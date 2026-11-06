#!/usr/bin/env python3
"""Validate kt_interpolate_prediction in src/opencl/ktgmc/kernels/ktgmc_motion.cl
against the CPU mirror sim/ktgmc_ip_ref.cpp with an independent Python golden.

Run:  python3 python/run_mv_interp.py
"""
import os, random, subprocess, tempfile, sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BIN = os.path.join(tempfile.gettempdir(), "ip_ref")


def arshift(v, n):
    # arithmetic shift right for int32 semantics
    if v >= 0:
        return v >> n
    return -((-v) >> n) - (1 if (-v) & ((1 << n) - 1) else 0)


def idiv(a, b):
    """C-style integer division truncating toward zero (matches CUDA/OpenCL/C++)."""
    q = abs(a) // abs(b)
    return q if (a >= 0) == (b >= 0) else -q


def run_ref(nums):
    inf = os.path.join(tempfile.gettempdir(), "ip_in.txt")
    outf = os.path.join(tempfile.gettempdir(), "ip_out.txt")
    with open(inf, "w") as f:
        f.write(" ".join(map(str, nums)))
    subprocess.run([BIN, inf, outf], check=True)
    out = []
    with open(outf) as f:
        for line in f:
            out.append(tuple(map(int, line.split())))
    return out


def golden(params, srcv):
    nSX, nSY, nDX, nDY, normFactor, normov, atotal, aodd, aeven = params
    out = []
    for y in range(nDY):
        for X in range(nDX):
            i, j = X, y
            if i >= 2 * nSX: i = 2 * nSX - 1
            if j >= 2 * nSY: j = 2 * nSY - 1
            offy = -1 + 2 * (j % 2)
            offx = -1 + 2 * (i % 2)
            ip2, jp2 = i >> 1, j >> 1
            A = ip2 + jp2 * nSX
            def V(k): return srcv[k]
            if (i == 0) or (i >= 2 * nSX - 1):
                if (j == 0) or (j >= 2 * nSY - 1):
                    v1 = v2 = v3 = v4 = V(A)
                else:
                    B = ip2 + (jp2 + offy) * nSX
                    v1 = v2 = V(A); v3 = v4 = V(B)
            elif (j == 0) or (j >= 2 * nSY - 1):
                B = ip2 + offx + jp2 * nSX
                v1 = v2 = V(A); v3 = v4 = V(B)
            else:
                B = ip2 + offx + jp2 * nSX
                C = ip2 + (jp2 + offy) * nSX
                D = ip2 + offx + (jp2 + offy) * nSX
                v1, v2, v3, v4 = V(A), V(B), V(C), V(D)
            ax1 = aodd if offx > 0 else aeven
            ax2 = atotal - ax1
            ay1 = aodd if offy > 0 else aeven
            ay2 = atotal - ay1
            a11 = ax1 * ay1; a12 = ax1 * ay2; a21 = ax2 * ay1; a22 = ax2 * ay2
            vx = idiv(a11 * v1[0] + a21 * v2[0] + a12 * v3[0] + a22 * v4[0], normov)
            vy = idiv(a11 * v1[1] + a21 * v2[1] + a12 * v3[1] + a22 * v4[1], normov)
            ts = idiv(a11 * v1[2] + a21 * v2[2] + a12 * v3[2] + a22 * v4[2], normov)
            if normFactor > 0:
                vx = arshift(vx, normFactor); vy = arshift(vy, normFactor)
            else:
                vx <<= -normFactor; vy <<= -normFactor
            out.append((vx, vy, arshift(ts, 4)))
    return out


def main():
    subprocess.run(["g++", "-O2", "-std=gnu++17", "-Wl,--allow-multiple-definition", "-w",
                    os.path.join(REPO, "sim", "ktgmc_ip_ref.cpp"),
                    "-o", BIN], check=True)
    rng = random.Random(11)
    ok = True
    total = 0
    for _ in range(200):
        nSX = rng.randint(1, 8); nSY = rng.randint(1, 8)
        # fine grid approx 2x coarse, but can be arbitrary
        nDX = rng.randint(nSX, 2 * nSX + 1)
        nDY = rng.randint(nSY, 2 * nSY + 1)
        normFactor = rng.choice([-2, -1, 0, 1, 2])
        normov = rng.randint(1, 3000)
        atotal = rng.randint(1, 100)
        aodd = rng.randint(0, atotal); aeven = rng.randint(0, atotal)
        params = (nSX, nSY, nDX, nDY, normFactor, normov, atotal, aodd, aeven)
        srcv = [(rng.randint(-2000, 2000), rng.randint(-2000, 2000),
                 rng.randint(0, 3000)) for _ in range(nSX * nSY)]
        nums = list(params) + [v for t in srcv for v in t]
        got = run_ref(nums)
        exp = golden(params, srcv)
        total += 1
        if got != exp:
            ok = False
            print("interp MISMATCH", params)
            if total <= 3:
                for g, e in zip(got, exp):
                    if g != e:
                        print("  got", g, "exp", e)
    print(f"interpolate_prediction: {'PASS' if ok else 'FAIL'} ({total} cases)")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
