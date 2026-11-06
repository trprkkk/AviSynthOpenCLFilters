#!/usr/bin/env python3
"""Validate kt_rb2b_bilinear_filtered in src/opencl/ktgmc/kernels/ktgmc_motion.cl.

The .cl kernel is a *single-pass* recomputation of the separable (1,3,3,1)/8
two-phase filter (vertical-then-horizontal, each phase rounding separately) --
the same integer algorithm as the CPU RB2BilinearFiltered reference. Here we
check that an independent single-pass golden (mirroring the .cl) agrees
bit-for-bit with the CPU reference mirror sim/ktgmc_rb2b_ref.cpp over random
planes (edges exercised by using the full 1:2 reduction). Only values that occur
make a difference, so both must round identically.

Run:  python3 python/run_mv_rb2b.py
"""
import os, random, subprocess, tempfile, sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BIN = os.path.join(tempfile.gettempdir(), "rb2b_ref")


def run_mirror(nums):
    inf = os.path.join(tempfile.gettempdir(), "rb_in.txt")
    outf = os.path.join(tempfile.gettempdir(), "rb_out.txt")
    with open(inf, "w") as f:
        f.write(" ".join(map(str, nums)))
    subprocess.run([BIN, inf, outf], check=True)
    with open(outf) as f:
        return [int(line) for line in f]


def vv(src, sp, col, y, nh, nw):
    """single-pass vertical value (intermediate) at (y,col)."""
    if y == 0:
        return (src[col] + src[col + sp] + 1) >> 1
    if y < nh - 1:
        r0 = src[col + (2 * y - 1) * sp]
        r1 = src[col + (2 * y) * sp]
        r2 = src[col + (2 * y + 1) * sp]
        r3 = src[col + (2 * y + 2) * sp]
        return (r0 + r1 * 3 + r2 * 3 + r3 + 4) // 8
    a = src[col + (2 * y) * sp]
    b = src[col + (2 * y + 1) * sp]
    return (a + b + 1) >> 1


def golden(nw, nh, srcp, src):
    out = []
    for y in range(nh):
        for x in range(nw):
            if x == 0 or x == nw - 1:
                a = vv(src, srcp, 2 * x, y, nh, nw)
                b = vv(src, srcp, 2 * x + 1, y, nh, nw)
                v = (a + b + 1) >> 1
            else:
                a = vv(src, srcp, 2 * x - 1, y, nh, nw)
                b = vv(src, srcp, 2 * x, y, nh, nw)
                c = vv(src, srcp, 2 * x + 1, y, nh, nw)
                d = vv(src, srcp, 2 * x + 2, y, nh, nw)
                v = (a + b * 3 + c * 3 + d + 4) // 8
            out.append(v)
    return out


def main():
    subprocess.run(["g++", "-O2", "-std=gnu++17", "-Wl,--allow-multiple-definition", "-w",
                    os.path.join(REPO, "sim", "ktgmc_rb2b_ref.cpp"),
                    "-o", BIN], check=True)
    rng = random.Random(19)
    ok = True
    total = 0
    for _ in range(200):
        nw = rng.randint(2, 12)
        nh = rng.randint(2, 12)
        srcp = 2 * nw + rng.choice([0, 0, 2, 5])   # include some padding
        maxv = rng.choice([255, 255, 65535])
        src = [rng.randint(0, maxv) for _ in range(srcp * 2 * nh)]
        nums = [nw, nh, srcp, nw] + src
        got = run_mirror(nums)
        exp = golden(nw, nh, srcp, src)
        total += 1
        if got != exp:
            ok = False
            print("rb2b MISMATCH nw,nh,srcp", nw, nh, srcp)
            for i, (g, e) in enumerate(zip(got, exp)):
                if g != e:
                    print("  px", i, "got", g, "exp", e)
            if total >= 3:
                break
    print(f"rb2b_bilinear_filtered: {'PASS' if ok else 'FAIL'} ({total} cases)")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
