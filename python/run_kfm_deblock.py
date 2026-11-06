#!/usr/bin/env python3
"""Validate the KFM KDeblock core kernel (kf_deblock) in
src/opencl/kfm/kernels/kfm_deblock.cl against the CPU mirror
sim/kfm_deblock_ref.cpp with an independent Python golden.

The 8x8 DCT/IDCT is a fixed float32 butterfly (Devblock.cu dev_dct8/dev_idct8,
S1..S2 constants).  The mirror is compiled -ffp-contract=off and the golden
emulates float32 per operation (every add/sub/mul rounds), so the two must match
bit-for-bit.  Output is the 16-bit block-parity accumulator plane (pre-merge);
only the core kl_deblock is exercised (pad/qp/merge are separate host steps).

Run:  python3 python/run_kfm_deblock.py
"""
import os, random, struct, subprocess, tempfile, sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BIN = os.path.join(tempfile.gettempdir(), "kfm_deblock_ref")


def F(v):
    return struct.unpack('f', struct.pack('f', v))[0]
def FI(v):
    return struct.unpack('i', struct.pack('f', v))[0]

S1 = F(0.19509032201612825)
C1 = F(0.9807852804032304)
S3 = F(0.5555702330196022)
C3 = F(0.8314696123025452)
S2S6 = F(1.3065629648763766)
S2C6 = F(0.5411961001461971)
S2 = F(1.4142135623730951)

_OFFX = ("0,0,4, 0,2,6,4, 0,5,2,7,4,1,6,3, 0,4,1,5,3,7,2,6,0,4,1,5,3,7,2,6,"
         "0,0,0,0,1,1,1,1,2,2,2,2,3,3,3,3,4,4,4,4,5,5,5,5,6,6,6,6,7,7,7,7,"
         "0,4,0,4,2,6,2,6,0,4,0,4,2,6,2,6,1,5,1,5,3,7,3,7,1,5,1,5,3,7,3,7,"
         "0,4,0,4,2,6,2,6,0,4,0,4,2,6,2,6,1,5,1,5,3,7,3,7,1,5,1,5,3,7,3,7,")
_OFFY = ("0,0,4, 0,2,4,6, 0,1,2,3,4,5,6,7, 0,0,1,1,2,2,3,3,4,4,5,5,6,6,7,7,"
         "0,2,4,6,1,3,5,7,0,2,4,6,1,3,5,7,0,2,4,6,1,3,5,7,0,2,4,6,1,3,5,7,"
         "0,4,4,0,2,6,6,2,2,6,6,2,0,4,4,0,1,5,5,1,3,7,7,3,3,7,7,3,1,5,5,1,"
         "1,5,5,1,3,7,7,3,3,7,7,3,1,5,5,1,0,4,4,0,2,6,6,2,2,6,6,2,0,4,4,0,")
g_ox = [int(s) for s in _OFFX.split(',') if s.strip()]
g_oy = [int(s) for s in _OFFY.split(',') if s.strip()]
assert len(g_ox) == len(g_oy) == 127, (len(g_ox), len(g_oy))


def dct1(v):
    """one 1D 8-pt forward DCT on a length-8 list (dev_dct8, stride-order)."""
    a0 = F(v[7] + v[0]); a1 = F(v[6] + v[1]); a2 = F(v[5] + v[2]); a3 = F(v[4] + v[3])
    a4 = F(v[3] - v[4]); a5 = F(v[2] - v[5]); a6 = F(v[1] - v[6]); a7 = F(v[0] - v[7])
    b0 = F(a3 + a0); b1 = F(a2 + a1); b2 = F(a1 - a2); b3 = F(a0 - a3)
    b4 = F(F(F(S3 - C3) * a7) + F(C3 * F(a4 + a7)))
    b5 = F(F(F(S1 - C1) * a6) + F(C1 * F(a5 + a6)))
    b6 = F(F(F(-F(C1 + S1)) * a5) + F(C1 * F(a5 + a6)))
    b7 = F(F(F(-F(C3 + S3)) * a4) + F(C3 * F(a4 + a7)))
    c0 = F(b1 + b0); c1 = F(b0 - b1)
    c2 = F(F(F(S2S6 - S2C6) * b3) + F(S2C6 * F(b2 + b3)))
    c3 = F(F(F(-F(S2C6 + S2S6)) * b2) + F(S2C6 * F(b2 + b3)))
    c4 = F(b6 + b4); c5 = F(b7 - b5); c6 = F(b4 - b6); c7 = F(b5 + b7)
    d4 = F(c7 - c4); d5 = F(c5 * S2); d6 = F(c6 * S2); d7 = F(c4 + c7)
    # dev_dct8 store: [0]=c0 [4]=c1 [2]=c2 [6]=c3 [7]=d4 [3]=d5 [5]=d6 [1]=d7
    r = [0.0]*8
    r[0] = c0; r[1] = d7; r[2] = c2; r[3] = d5
    r[4] = c1; r[5] = d6; r[6] = c3; r[7] = d4
    return r


def idct1(v):
    """one 1D 8-pt inverse DCT on a length-8 list (dev_idct8, stride-order)."""
    c0 = v[0]; c1 = v[4]; c2 = v[2]; c3 = v[6]
    d4 = v[7]; d5 = v[3]; d6 = v[5]; d7 = v[1]
    c4 = F(d7 - d4); c5 = F(d5 * S2); c6 = F(d6 * S2); c7 = F(d4 + d7)
    b0 = F(c1 + c0); b1 = F(c0 - c1)
    b2 = F(F(F(-F(S2C6 + S2S6)) * c3) + F(S2C6 * F(c2 + c3)))
    b3 = F(F(F(S2S6 - S2C6) * c2) + F(S2C6 * F(c2 + c3)))
    b4 = F(c6 + c4); b5 = F(c7 - c5); b6 = F(c4 - c6); b7 = F(c5 + c7)
    a0 = F(b3 + b0); a1 = F(b2 + b1); a2 = F(b1 - b2); a3 = F(b0 - b3)
    a4 = F(F(F(-F(C3 + S3)) * b7) + F(C3 * F(b4 + b7)))
    a5 = F(F(F(-F(C1 + S1)) * b6) + F(C1 * F(b5 + b6)))
    a6 = F(F(F(S1 - C1) * b5) + F(C1 * F(b5 + b6)))
    a7 = F(F(F(S3 - C3) * b4) + F(C3 * F(b4 + b7)))
    return [F(a7 + a0), F(a6 + a1), F(a5 + a2), F(a4 + a3),
            F(a3 - a4), F(a2 - a5), F(a1 - a6), F(a0 - a7)]


def hardthresh(d, th):
    for i in range(1, 64):
        if not (d[i] < -th or d[i] > th):
            d[i] = 0.0


def block_dct_idct(d):
    # forward: rows then cols
    for r in range(8):
        d[r*8:(r+1)*8] = dct1(d[r*8:(r+1)*8])
    for c in range(8):
        col = [d[c + r*8] for r in range(8)]
        col = dct1(col)
        for r in range(8):
            d[c + r*8] = col[r]


def block_idct(d):
    # inverse: cols then rows (matches cpu_idct8x8)
    for c in range(8):
        col = [d[c + r*8] for r in range(8)]
        col = idct1(col)
        for r in range(8):
            d[c + r*8] = col[r]
    for r in range(8):
        d[r*8:(r+1)*8] = idct1(d[r*8:(r+1)*8])


def qp_thresh(qpv, ta, tb):
    v = F(F(F(qpv) * ta) + tb)
    lo, hi = F(0.0), F(float(qpv))
    return lo if v < lo else (hi if v > hi else v)


def golden(sw, bh, out_pitch, count_minus_1, shift, maxv,
           strength, ta, tb, qp, src):
    # infer bw from qp length and sw layout: caller ensures qp has bw*bh
    nqp = len(qp)
    bw = nqp // bh
    rows = 32 * bh + 8
    out = [-1] * (out_pitch * rows)
    for by in range(bh):
        for bx in range(bw):
            local_out = [[0]*16 for _ in range(16)]
            for ty in range(count_minus_1 + 1):
                ox0, oy0 = g_ox[count_minus_1 + ty], g_oy[count_minus_1 + ty]
                ox, oy = bx*8 + ox0, by*8 + oy0
                d = [float(src[(ox + x) + (oy + y)*sw])
                     for y in range(8) for x in range(8)]
                qpv = qp[bx + by*bw]
                thresh = F(qp_thresh(qpv, ta, tb) * F(F(4.0) + strength))
                thresh = F(thresh - F(1.0))
                block_dct_idct(d)
                hardthresh(d, thresh)
                block_idct(d)
                half = (1 << shift) >> 1
                for y in range(8):
                    for x in range(8):
                        tmp = int(F(d[x + y*8] + float(half))) >> shift
                        tmp = 0 if tmp < 0 else (maxv if tmp > maxv else tmp)
                        local_out[oy0 + y][ox0 + x] += tmp
            offz = (bx & 1) + (by & 1) * 2
            offx = bx * 8
            offy = (bh * offz + by) * 8
            for y in range(16):
                for x in range(16):
                    out[(offx + x) + (offy + y) * out_pitch] = local_out[y][x]
    return out


def run_mirror(nums):
    inf = os.path.join(tempfile.gettempdir(), "kfdb_in.txt")
    outf = os.path.join(tempfile.gettempdir(), "kfdb_out.txt")
    with open(inf, "w") as f:
        f.write(" ".join(map(str, nums)))
    subprocess.run([BIN, inf, outf], check=True)
    with open(outf) as f:
        return [int(line) for line in f]


def main():
    # -mfpmath=sse -msse2: the sim MUST be per-op float32 (the golden rounds
    # every operation).  On 32-bit x87 toolchains (mingw32) intermediates keep
    # 80-bit precision between ops, which shifts threshold/truncation
    # boundaries and breaks bit-exactness — force SSE math.
    subprocess.run(["g++", "-O2", "-std=gnu++17", "-Wl,--allow-multiple-definition", "-ffp-contract=off",
                    "-mfpmath=sse", "-msse2", "-w",
                    os.path.join(REPO, "sim", "kfm_deblock_ref.cpp"),
                    "-o", BIN], check=True)
    rng = random.Random(29)
    ok = True
    total = 0
    for _ in range(300):
        bits = rng.choice([8, 8, 16])
        maxvpx = (1 << bits) - 1
        bw = rng.randint(1, 4)
        bh = rng.randint(1, 3)
        quality = rng.choice([0, 0, 1, 2])
        count = 1 << quality
        count_minus_1 = count - 1
        shift = max(0, quality + bits - 10)
        deblock_maxv = (1 << (bits + 6 - shift)) - 1
        # padded src plane generously sized
        sw = bw * 8 + 16
        sh = bh * 8 + 16
        src = [rng.randint(0, maxvpx) for _ in range(sh * sw)]
        qp_pitch = bw
        qp = [rng.randint(0, 40) for _ in range(bw * bh)]
        strength = F(rng.choice([4.0, 8.0, 20.0, 30.0]))
        ta = F(rng.choice([0.02, 0.05, 0.08]))
        tb = F(rng.choice([-1.0, -0.5, 0.0]))
        out_pitch = sw   # accumulator plane pitch (>= bw*8+16)
        hdr = [ord('D'), sw, sh, bh, out_pitch, qp_pitch, count_minus_1,
               shift, deblock_maxv, FI(strength), FI(ta), FI(tb),
               bw, bw * bh]
        got = run_mirror(hdr + qp + [sh * sw] + src)
        exp = golden(sw, bh, out_pitch, count_minus_1, shift, deblock_maxv,
                     strength, ta, tb, qp, src)
        total += 1
        if got != exp:
            ok = False
            mism = 0
            for i, (g, e) in enumerate(zip(got, exp)):
                if g != e:
                    mism += 1
                    if mism <= 5:
                        print("KDeblock MISMATCH bw", bw, "bh", bh, "q", quality,
                              "px", i, g, e)
            if mism:
                print("  total mism", mism)
            if total >= 3:
                break
    print(f"KFM KDeblock core deblock: {'PASS' if ok else 'FAIL'} "
          f"({total} cases)")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
