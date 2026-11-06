#!/usr/bin/env python3
"""Validate kt_mean_global_mv in src/opencl/ktgmc/kernels/ktgmc_motion.cl
against the CPU mirror sim/ktgmc_mean_ref.cpp with an independent Python golden.

Run:  python3 python/run_mv_mean.py
"""
import os, random, subprocess, tempfile, sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BIN = os.path.join(tempfile.gettempdir(), "mean_ref")


def idiv(a, b):
    q = abs(a) // abs(b)
    return q if (a >= 0) == (b >= 0) else -q


def run_ref(nums):
    inf = os.path.join(tempfile.gettempdir(), "mean_in.txt")
    outf = os.path.join(tempfile.gettempdir(), "mean_out.txt")
    with open(inf, "w") as f:
        f.write(" ".join(map(str, nums)))
    subprocess.run([BIN, inf, outf], check=True)
    out = []
    with open(outf) as f:
        for line in f:
            out.append(tuple(map(int, line.split())))
    return out


def golden(rows):
    res = []
    for (medianx, mediany, vecs) in rows:
        sx = sy = 0
        num = 0
        for (vx, vy) in vecs:
            dx = vx - medianx; dy = vy - mediany
            if abs(dx) < 6 and abs(dy) < 6:
                sx += vx; sy += vy; num += 1
        res.append((idiv(2 * sx, num), idiv(2 * sy, num)))
    return res


def main():
    subprocess.run(["g++", "-O2", "-std=gnu++17", "-Wl,--allow-multiple-definition", "-w",
                    os.path.join(REPO, "sim", "ktgmc_mean_ref.cpp"),
                    "-o", BIN], check=True)
    rng = random.Random(7)
    ok = True
    total = 0
    for _ in range(400):
        nRows = rng.randint(1, 6)
        rows = []
        nums = [nRows]
        for _ in range(nRows):
            nVec = rng.randint(1, 300)
            # a cluster near a random median so num>=1 (avoid div-by-0)
            mx = rng.randint(-2000, 2000); my = rng.randint(-2000, 2000)
            vecs = []
            for _ in range(nVec):
                if rng.random() < 0.7:
                    vecs.append((mx + rng.randint(-3, 3), my + rng.randint(-3, 3)))
                else:
                    vecs.append((mx + rng.randint(-40, 40), my + rng.randint(-40, 40)))
            # ensure at least one exact median
            vecs[0] = (mx, my)
            rows.append((mx, my, vecs))
            nums += [mx, my, nVec]
            for (vx, vy) in vecs:
                nums += [vx, vy]
        got = run_ref(nums)
        exp = golden(rows)
        total += 1
        if got != exp:
            ok = False
            print("mean MISMATCH rowgroup", total)
            for g, e in zip(got, exp):
                if g != e:
                    print("  got", g, "exp", e)
    print(f"mean_global_mv: {'PASS' if ok else 'FAIL'} ({total} rowgroups)")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
