#!/usr/bin/env python3
"""Validate kt_rb2b_bilinear_filtered_with_pad in
src/opencl/ktgmc/kernels/ktgmc_motion.cl.

The .cl kernel is a scalar transliteration of the CUDA-only FUSED anti-aliased
1:2 downsample kl_RB2B_bilinear_filtered_with_pad (MVKernel.cu / MV.cpp
ReduceToPad): a single 4x4-tap weighted filter with one +32/64 rounding that
also fills the destination plane's hpad/vpad border.  It is numerically
DISTINCT from the two-phase separable kt_rb2b_bilinear_filtered (which the CPU
ReduceToPad twin uses + a separate Pad()); there is no separable host routine to
match, so this test independently re-implements the same fused CUDA algorithm
in Python and cross-checks the CPU mirror sim/ktgmc_rb2b_pad_ref.cpp against it
bit-for-bit over random padded domains (both 8- and 16-bit sample ranges, pads
on every edge exercised by small planes).

Run:  python3 python/run_mv_rb2b_pad.py
"""
import os, random, subprocess, tempfile, sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BIN = os.path.join(tempfile.gettempdir(), "rb2b_pad_ref")


def run_mirror(nums):
    inf = os.path.join(tempfile.gettempdir(), "rbp_in.txt")
    outf = os.path.join(tempfile.gettempdir(), "rbp_out.txt")
    with open(inf, "w") as f:
        f.write(" ".join(map(str, nums)))
    subprocess.run([BIN, inf, outf], check=True)
    with open(outf) as f:
        return [int(line) for line in f]


def taps(coord, n, mul0, mul1):
    """Return (src_base, weights[4] for tap -1..2, active bool[4]).

    Edge cases (coord <= 0 or coord >= n-1) replicate: src_base clamped, and
    only taps 0,1 are active with weight mul1 (mul0 taps are zeroed).  Interior
    always uses the (1,3,3,1) weights regardless of the passed edge multipliers.
    """
    if coord <= 0:
        return 0, [mul0, mul1, mul1, mul0], [False, True, True, False]
    if coord >= n - 1:
        return (n - 1) * 2, [mul0, mul1, mul1, mul0], [False, True, True, False]
    return coord * 2, [1, 3, 3, 1], [True, True, True, True]


def golden(nw, nh, srcp, src, hpad, vpad):
    W = nw + 2 * hpad
    H = nh + 2 * vpad
    out = [0] * (W * H)
    for dsty in range(-vpad, nh + vpad):
        sy, yw, ya = taps(dsty, nh, 0, 4)
        for dstx in range(-hpad, nw + hpad):
            sx, xw, xa = taps(dstx, nw, 0, 4)
            s = 0
            for j in range(4):
                if ya[j]:
                    for i in range(4):
                        if xa[i]:
                            pix = src[(sx + i - 1) + (sy + j - 1) * srcp]
                            s += pix * yw[j] * xw[i]
            s = (s + 32) // 64
            out[(dsty + vpad) * W + (dstx + hpad)] = s
    return out


def main():
    subprocess.run(["g++", "-O2", "-std=gnu++17", "-Wl,--allow-multiple-definition", "-w",
                    os.path.join(REPO, "sim", "ktgmc_rb2b_pad_ref.cpp"),
                    "-o", BIN], check=True)
    rng = random.Random(23)
    ok = True
    total = 0
    hpad_vals = [1, 2, 4, 8]
    for _ in range(250):
        nw = rng.randint(2, 12)
        nh = rng.randint(2, 12)
        hpad = rng.choice(hpad_vals)
        vpad = rng.choice(hpad_vals)
        srcp = 2 * nw + rng.choice([0, 0, 2, 5, 9])
        maxv = rng.choice([255, 255, 65535])
        src = [rng.randint(0, maxv) for _ in range(srcp * 2 * nh)]
        nums = [nw, nh, srcp, W := (nw + 2 * hpad), hpad, vpad] + src
        got = run_mirror(nums)
        exp = golden(nw, nh, srcp, src, hpad, vpad)
        total += 1
        if got != exp:
            ok = False
            print("rb2b_pad MISMATCH nw,nh,srcp,hpad,vpad", nw, nh, srcp, hpad, vpad)
            for i, (g, e) in enumerate(zip(got, exp)):
                if g != e:
                    print("  px", i, "got", g, "exp", e)
            if total >= 3:
                break
    print(f"rb2b_bilinear_filtered_with_pad: {'PASS' if ok else 'FAIL'} ({total} cases)")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
