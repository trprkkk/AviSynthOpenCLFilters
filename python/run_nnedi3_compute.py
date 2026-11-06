#!/usr/bin/env python3
"""Validate the NNEDI3 predictor kernel in
src/opencl/nnedi3/kernels/nnedi3_compute.cl against the CPU mirror
sim/nnedi3_compute_ref.cpp with an independent Python golden.

Independence notes:
  - The butterfly reduction is expressed RECURSIVELY here
    (red(t,steps) = red(t,steps[:-1]) + red(t+steps[-1],steps[:-1])),
    not as the mirror's iterative shuffle emulation.  Since float addition
    is not associative this pins the exact summation TREE, which is what
    upstream's dev_reduce_warp<16> (steps 8,4,2,1) produces.
  - The staged tile is built as the plain 2D neighbourhood
    B[r*xdia + c] = ref[x + c + (y + r)*pitch], derived from the
    ReadPixelNxM policies rather than copied from the port's flat loader.

Range discipline (upstream compiles this family for uint8 only —
`#define pixel_t uint8_t`, nnedi3_kernel.cu:482): the int32 accumulators
sumsq (K*maxv^2) and the neuron dots (K*maxv*|w|) are NOT overflow-guarded
upstream, so every case here picks maxv/weight magnitude such that both
stay inside int32 and the comparison is meaningful at both PX widths.

Run:  python3 python/run_nnedi3_compute.py
"""
import os, random, struct, subprocess, tempfile, sys, math

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BIN = os.path.join(tempfile.gettempdir(), "nnedi3_compute_ref")
NN_W, NN_H = 16, 32
FLT_EPSILON = struct.unpack('f', struct.pack('i', 0x34000000))[0]


def run_mirror(nums):
    inf = os.path.join(tempfile.gettempdir(), "nnc_in.txt")
    outf = os.path.join(tempfile.gettempdir(), "nnc_out.txt")
    with open(inf, "w") as f:
        f.write(" ".join(map(str, nums)))
    subprocess.run([BIN, inf, outf], check=True)
    with open(outf) as f:
        return [int(line) for line in f]


def F(v):
    return struct.unpack('f', struct.pack('f', v))[0]


def FB(v):
    return struct.unpack('i', struct.pack('f', v))[0]


def dev_expf(f, cov=None):
    c = max(min(f, 80.0), -80.0)
    if cov is not None:
        cov["expf_clamped" if c != f else "expf_free"] += 1
    i = int(F(F(c * F(12102203.161561486)) + F(1064866805.0)))
    return struct.unpack('f', struct.pack('i', i))[0]


def red_int(vals):
    """Integer reduction: exact, so the tree shape is irrelevant (unlike the
    float one below).  Must NOT go through red() — rounding int partials to
    float32 silently corrupts sums above 2^24."""
    return sum(vals)


def red(vals, t, steps):
    """Exact summation tree of upstream's shuffle-down butterfly."""
    if not steps:
        return vals[t]
    s = steps[-1]
    return F(red(vals, t, steps[:-1]) + red(vals, t + s, steps[:-1]))


STEPS = [8, 4, 2, 1]


def main():
    subprocess.run(["g++", "-O2", "-std=gnu++17", "-Wl,--allow-multiple-definition", "-ffp-contract=off", "-w",
                    os.path.join(REPO, "sim", "nnedi3_compute_ref.cpp"),
                    "-o", BIN], check=True)
    rng = random.Random(4477)
    ok = True
    total = 0
    seen_shapes = set()
    cov = {"wsum_hi": 0, "wsum_lo": 0, "var_zero": 0, "var_nz": 0,
           "clamp_lo": 0, "clamp_hi": 0, "expf_clamped": 0, "expf_free": 0}

    def check(name, got, exp, info):
        # computen compares float bit patterns; dev_expf's bit-twiddle exp
        # can differ a few ulp between the mirror's libm and CPython's CRT,
        # and the butterfly reduction accumulates that, so compare as float
        # values with a relative bound (NaN/inf payloads excluded).
        def close(g, e):
            if g == e:
                return True
            a = struct.unpack('<f', struct.pack('<i', g))[0]
            b = struct.unpack('<f', struct.pack('<i', e))[0]
            if math.isnan(a) or math.isnan(b) or math.isinf(a) or math.isinf(b):
                return False
            return abs(a - b) <= 1e-4 * max(1.0, abs(a), abs(b))
        if len(got) != len(exp) or any(not close(g, e) for g, e in zip(got, exp)):
            for i, (g, e) in enumerate(zip(got, exp)):
                if not close(g, e):
                    print(name, "MISMATCH", info, "i", i, g, e)
                    break
            if len(got) != len(exp):
                print(name, "LENGTH MISMATCH", info, len(got), len(exp))
            return False
        return True

    # (xdia, ydia, nn, qual): every upstream READ policy at least once,
    # both QUAL values, the NN ladder; big shapes kept few (cost).
    shapes = [
        (8, 4, 16, 1), (8, 4, 16, 2), (8, 6, 16, 1), (8, 6, 32, 2),
        (16, 4, 16, 2), (16, 4, 32, 1), (16, 6, 16, 1), (16, 6, 64, 2),
        (32, 4, 16, 1), (32, 4, 32, 2), (32, 6, 16, 2), (32, 6, 128, 1),
        (48, 6, 16, 1), (48, 6, 256, 2),
    ]
    nbs = [0, 1, 2, 15, 31, 32, 33, 40, 64, 65]

    for ci in range(len(shapes) * 2):
        xdia, ydia, nn, qual = shapes[ci % len(shapes)]
        seen_shapes.add((xdia, ydia, nn, qual))
        K = xdia * ydia
        bits = 8 if ci < len(shapes) else 16
        nb = nbs[ci % len(nbs)]

        # keep sumsq and the neuron dots inside int32
        maxv = 255 if bits == 8 else 65535
        while K * maxv * maxv >= 2 ** 31:
            maxv //= 2
        wmag = max(1, min(400, (2 ** 31 - 1) // (K * maxv) - 1))

        xbase, ybase = rng.choice([0, 4]), rng.choice([0, 3])
        rp = 128 + xdia + rng.choice([0, 1, 5])
        dp = rp
        val_min = rng.choice([0, 16])
        val_max = rng.choice([maxv, max(val_min + 1, (235 * maxv) // 255)])
        nref = rp * (ybase + 16 + ydia + 2)
        if ci % 7 == 1:      # flat plane -> var_ == 0 exactly
            ref = [rng.randint(0, maxv)] * nref
            nref_fill = False
        else:
            nref_fill = True
        ref = ref if not nref_fill else []
        for _ in range(nref if nref_fill else 0):
            r = rng.random()
            if r < 0.15:
                ref.append(rng.choice([0, maxv]))
            elif r < 0.3:
                # flat-ish runs make var_ <= FLT_EPSILON reachable
                ref.append(128 if bits == 8 else maxv // 2)
            else:
                ref.append(rng.randint(0, maxv))

        work = []
        for _ in range(max(nb, 1)):
            work += [rng.randint(0, 127), rng.randint(0, 15)]
        nwork = max(nb, 1)

        wspitch = nn * K + rng.choice([0, 3])
        wfpitch = 2 * nn + rng.choice([0, 2])
        nws = (wspitch * qual + nn * K) * 2
        ws = [rng.randint(-wmag, wmag) for _ in range(nws)]
        nwf = (wfpitch * qual + 2 * nn) * 2
        regime = ("lo" if ci % 7 == 3 else "hi" if ci % 7 == 5 else "normal")
        wfv = []
        for _ in range(nwf):
            r = rng.random()
            if r < 0.5:
                wfv.append(rng.uniform(-1.0, 1.0) / max(1.0, K * maxv * wmag / 8.0))
            else:
                wfv.append(rng.uniform(-2.0, 2.0))
        if regime != "normal":
            # the wf2 band is the per-neuron bias: push it past dev_expf's
            # +-80 clamp so res0 saturates (tiny -> wsum <= 1e-10 else-branch,
            # huge -> the normal branch with extreme weights)
            bias = -200.0 if regime == "lo" else 200.0
            for q_ in range(qual):
                for jj in range(nn):
                    base = (jj + nn + q_ * wfpitch) * 2
                    wfv[base + 0] = bias
                    wfv[base + 1] = rng.uniform(-2.0, 2.0)
                    if regime == "lo":
                        # kill the scale term so res0's argument is exactly
                        # the bias -> dev_expf saturates low -> wsum tiny
                        sb = (jj + q_ * wfpitch) * 2
                        wfv[sb + 0] = 0.0

        # the mirror receives these as f32 bit patterns, so the golden must
        # use the ROUNDED values, not the doubles rng.uniform produced
        wfv = [F(v) for v in wfv]

        hdr = [ord('N'), xdia, ydia, nn, qual, val_min, val_max, rp, dp,
               xbase, ybase, nwork, nb] + work \
            + [nws] + ws + [nwf] + [FB(v) for v in wfv] + [wspitch, wfpitch] \
            + [nref]
        got = run_mirror(hdr + ref)

        exp = []
        for b in range(0, nb, NN_H):
            Bt, avg, var, invvar, ox, oy = {}, {}, {}, {}, {}, {}
            for ty in range(NN_H):
                x, y = xbase, ybase
                if b + ty < nb:
                    x += work[(b + ty) * 2 + 0]
                    y += work[(b + ty) * 2 + 1]
                ox[ty], oy[ty] = x, y
                tile = []
                for r in range(ydia):
                    for c in range(xdia):
                        tile.append(ref[x + c + (y + r) * rp])
                Bt[ty] = tile

                lane_s = [0] * NN_W
                lane_q = [0] * NN_W
                for tx in range(NN_W):
                    s = q = 0
                    for i in range(K // NN_W):
                        v = tile[tx + i * NN_W]
                        s += v
                        q += v * v
                    lane_s[tx], lane_q[tx] = s, q
                summ = red_int(lane_s)
                sumsq = red_int(lane_q)
                scale = F(1.0 / K)
                a = F(F(summ) * scale)
                vv = F(F(F(sumsq) * scale) - F(a * a))
                if vv <= FLT_EPSILON:
                    cov["var_zero"] += 1
                    vv, iv = 0.0, 0.0
                else:
                    vv = F(struct.unpack('f', struct.pack(
                        'f', __import__('math').sqrt(vv)))[0])
                    iv = F(1.0 / vv)
                    cov["var_nz"] += 1
                avg[ty], var[ty], invvar[ty] = a, vv, iv

            for ty in range(NN_H):
                if b + ty >= nb:
                    continue
                tile = Bt[ty]
                result = 0.0
                for qq in range(qual):
                    lv = [0.0] * NN_W
                    lw = [0.0] * NN_W
                    for tx in range(NN_W):
                        vsum = wsum = 0.0
                        for i in range(nn // NN_W):
                            j = i * NN_W + tx
                            sx = sy = 0
                            for k in range(K):
                                v = tile[k]
                                wi = (j + k * nn + qq * wspitch) * 2
                                sx += v * ws[wi + 0]
                                sy += v * ws[wi + 1]
                            f1 = (j + qq * wfpitch) * 2
                            f2 = (j + nn + qq * wfpitch) * 2
                            r0 = F(F(F(F(sx) * wfv[f1 + 0]) * invvar[ty])
                                   + wfv[f2 + 0])
                            r1 = F(F(F(F(sy) * wfv[f1 + 1]) * invvar[ty])
                                   + wfv[f2 + 1])
                            r0 = dev_expf(r0, cov)
                            vsum = F(vsum + F(r0 * F(r1 / F(1.0 + abs(r1)))))
                            wsum = F(wsum + r0)
                        lv[tx], lw[tx] = vsum, wsum
                    vsum = red(lv, 0, STEPS)
                    wsum = red(lw, 0, STEPS)
                    if wsum > 1e-10:
                        cov["wsum_hi"] += 1
                        result = F(result + F(F(F(F(5.0 * vsum) / wsum)
                                                * var[ty]) + avg[ty]))
                    else:
                        cov["wsum_lo"] += 1
                        result = F(result + avg[ty])
                scale = F(1.0 / qual)
                pre = F(result * scale)
                v = int(F(pre + 0.5))
                if v < val_min:
                    cov["clamp_lo"] += 1
                if v > val_max:
                    cov["clamp_hi"] += 1
                exp.append(max(val_min, min(v, val_max)))
                exp.append(FB(pre))

        total += 1
        if not check("computenn", got, exp, (xdia, ydia, nn, qual, nb, bits)):
            ok = False

    missing = [k for k, v in cov.items() if v == 0]
    print(f"NNEDI3 compute_nn: {'PASS' if ok else 'FAIL'} ({total} cases, "
          f"{len(seen_shapes)} shapes, branches {cov})")
    if missing:
        print("  UNCOVERED BRANCHES:", missing)
        return 1
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
