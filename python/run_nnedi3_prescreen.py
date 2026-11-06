#!/usr/bin/env python3
"""Validate the NNEDI3 prescreener kernel in
src/opencl/nnedi3/kernels/nnedi3_prescreen.cl against the CPU mirror
sim/nnedi3_prescreen_ref.cpp with an independent Python golden.

Independence note: upstream (and therefore the port and the mirror) spells
the neighbourhood dot product as an asymmetric per-vector lane split
(x == 0 takes taps 0..1 from lanes z,w; x < 4 takes 4 taps; x == 4 takes
taps 14..15 from lanes x,y).  This golden does NOT transcribe that split —
it derives the equivalent flat form from first principles: row y consumes
the 16 pixels starting at pixel (xbase*4 + 2) with tap j at weight index
(j + y*16).  Agreement therefore also proves the lane split was read
correctly.

Float tail is emulated per operation in float32 (mirror built with
-ffp-contract=off); the work-list compaction is checked as an exclusive
add-scan over tid order, which is what upstream's dev_scan produces.

Run:  python3 python/run_nnedi3_prescreen.py
"""
import os, random, struct, subprocess, tempfile, sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BIN = os.path.join(tempfile.gettempdir(), "nnedi3_prescreen_ref")
PRE_W, PRE_H = 32, 16
PRE_N = PRE_W * PRE_H


def run_mirror(nums):
    inf = os.path.join(tempfile.gettempdir(), "nnp2_in.txt")
    outf = os.path.join(tempfile.gettempdir(), "nnp2_out.txt")
    with open(inf, "w") as f:
        f.write(" ".join(map(str, nums)))
    subprocess.run([BIN, inf, outf], check=True)
    with open(outf) as f:
        return [int(line) for line in f]


def F(v):
    return struct.unpack('f', struct.pack('f', v))[0]


def FB(v):
    return struct.unpack('i', struct.pack('f', v))[0]


def nblk(n, d):
    return (n + d - 1) // d


def main():
    subprocess.run(["g++", "-O2", "-std=gnu++17", "-Wl,--allow-multiple-definition", "-ffp-contract=off", "-w",
                    os.path.join(REPO, "sim", "nnedi3_prescreen_ref.cpp"),
                    "-o", BIN], check=True)
    rng = random.Random(9021)
    ok = True
    total = 0
    stats = {"rej": 0, "lanes": 0}

    def check(name, got, exp, info):
        if got != exp:
            for i, (g, e) in enumerate(zip(got, exp)):
                if g != e:
                    print(name, "MISMATCH", info, "i", i, g, e)
                    break
            if len(got) != len(exp):
                print(name, "LENGTH MISMATCH", info, len(got), len(exp))
            return False
        return True

    # The last CASES_ZERO cases zero the output layer so every in-range lane
    # lands on EXACTLY 0.0 (or -0.0, or the smallest denormals either side).
    # Without them `result <= 0` and `result < 0` are indistinguishable —
    # random weights never hit the boundary, as a mutation check confirmed.
    N_CASES, CASES_ZERO = 66, 6
    DEN = struct.unpack('f', struct.pack('i', 1))[0]   # +1.4e-45
    ZERO_BIASES = [0.0, -0.0, DEN, -DEN]

    for case in range(N_CASES):
        zero_case = case >= N_CASES - CASES_ZERO
        bits = rng.choice([8, 8, 16])
        maxv = 255 if bits == 8 else 65535
        # keep frames small but cross the 32x16 group grid in both axes
        w4 = rng.choice([1, 2, 5, 17, 31, 32, 33, 40, 64, 65])
        h = rng.choice([1, 2, 3, 7, 15, 16, 17, 20, 32, 33])
        rp4 = w4 + 4 + rng.choice([0, 0, 1, 3])
        dp4 = w4 + rng.choice([0, 1])
        val_min = rng.choice([0, 0, 16, 16 << (bits - 8)])
        val_max = rng.choice([maxv, maxv, 235 << (bits - 8), 240 << (bits - 8)])
        if val_max < val_min:
            val_min, val_max = 0, maxv

        nref = rp4 * 4 * (h + 3)
        ref = []
        for _ in range(nref):
            r = rng.random()
            if r < 0.2:
                ref.append(rng.choice([0, 1, maxv - 1, maxv]))
            else:
                ref.append(rng.randint(0, maxv))

        # int weights kept small so the exact int32 dot cannot overflow
        # (48 taps * 32767 * 65535 would); real prescreener weights are
        # quantised small too.
        wmag = rng.choice([1, 4, 40, 300])
        ws = [rng.randint(-wmag, wmag) for _ in range(256)]
        # float tail: scale chosen vs the observed sum magnitude so the
        # squash sees a useful range, bias sweeps the accept/reject mix
        scale = rng.choice([1e-6, 1e-5, 1e-4, 1e-3]) / max(1, wmag)
        wfv = []
        for i in range(7):
            for _ in range(4):
                if i == 0:
                    wfv.append(scale * rng.uniform(0.2, 3.0))
                elif i == 1:
                    wfv.append(rng.uniform(-1.0, 1.0))
                elif i == 6:
                    wfv.append(rng.choice([rng.uniform(-0.6, 0.6),
                                           rng.uniform(-3.0, 3.0)]))
                else:
                    wfv.append(rng.uniform(-1.5, 1.5))
        if zero_case:
            # output layer -> 0, biases -> {+0, -0, +denorm, -denorm}:
            # components 0/1/3 must reject (<= 0), component 2 must not.
            for i in range(8, 24):
                wfv[i] = 0.0
            order = list(ZERO_BIASES)
            rng.shuffle(order)
            for c in range(4):
                wfv[24 + c] = order[c]
        # the mirror receives these as f32 bit patterns: round first so the
        # golden multiplies by exactly what the mirror sees
        wfv = [F(v) for v in wfv]
        wf_bits = [FB(v) for v in wfv]

        hdr = [ord('P'), w4, h, rp4, dp4, val_min, val_max,
               256] + ws + [28] + wf_bits + [nref]
        got = run_mirror(hdr + ref)

        gx, gy = nblk(w4, PRE_W), nblk(h, PRE_H)
        nb = gx * gy
        dst = {}
        numb = [0] * nb
        work = [[] for _ in range(nb)]

        for by in range(gy):
            for bx in range(gx):
                bid = bx + by * gx
                res = []
                nums = []
                for tid in range(PRE_N):
                    tx, ty = tid % PRE_W, tid // PRE_W
                    xbase, ybase = tx + bx * PRE_W, ty + by * PRE_H
                    r = [1.0, 1.0, 1.0, 1.0]
                    if xbase < w4 and ybase < h:
                        s = [0, 0, 0, 0]
                        for y in range(4):
                            row = (ybase + y) * rp4 * 4
                            p0 = xbase * 4 + 2
                            for j in range(16):
                                pix = ref[row + p0 + j]
                                wi = (j + y * 16) * 4
                                for c in range(4):
                                    s[c] += ws[wi + c] * pix
                        val = []
                        for c in range(4):
                            tt = F(F(float(s[c]) * wfv[0 * 4 + c])
                                   + wfv[1 * 4 + c])
                            val.append(F(tt / F(abs(tt) + 1.0)))
                        r = []
                        for c in range(4):
                            acc = 0.0
                            for k in range(4):
                                acc = F(acc + F(wfv[(2 + k) * 4 + c] * val[k]))
                            r.append(F(acc + wfv[6 * 4 + c]))
                    res.append(r)
                    nums.append(sum(1 for c in range(4) if r[c] <= 0.0))

                idx = 0
                for tid in range(PRE_N):
                    tx, ty = tid % PRE_W, tid // PRE_W
                    for c in range(4):
                        if res[tid][c] <= 0.0:
                            work[bid] += [tx * 4 + c, ty]
                            idx += 1
                numb[bid] = idx
                stats["rej"] += idx
                stats["lanes"] += PRE_N * 4

                for tid in range(PRE_N):
                    tx, ty = tid % PRE_W, tid // PRE_W
                    xbase, ybase = tx + bx * PRE_W, ty + by * PRE_H
                    if nums[tid] < 4 and xbase < w4 and ybase < h:
                        for k in range(4):
                            col = (xbase + 2) * 4 + k
                            s3p = ref[col + (ybase + 0) * rp4 * 4]
                            s2 = ref[col + (ybase + 1) * rp4 * 4]
                            s4 = ref[col + (ybase + 2) * rp4 * 4]
                            s6 = ref[col + (ybase + 3) * rp4 * 4]
                            tmp = ((s2 + s4) * 19 - (s3p + s6) * 3 + 16) >> 5
                            tmp = max(val_min, min(tmp, val_max))
                            dst[(xbase * 4 + k, ybase)] = tmp

        exp = [dst.get((xx, yy), -1) for yy in range(h)
               for xx in range(w4 * 4)]
        exp += numb
        for i in range(nb):
            exp += work[i]

        total += 1
        if not check("prescreen", got, exp, (w4, h, bits, wmag)):
            ok = False

    frac = stats["rej"] / max(1, stats["lanes"])
    print(f"NNEDI3 prescreening: {'PASS' if ok else 'FAIL'} ({total} cases, "
          f"reject mix {frac:.1%} of lanes)")
    if ok and not (0.02 < frac < 0.98):
        print("  WARNING: degenerate accept/reject mix")
        return 1
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
