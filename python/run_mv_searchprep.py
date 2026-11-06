#!/usr/bin/env python3
"""Validate kt_prepare_search in src/opencl/ktgmc/kernels/ktgmc_motion.cl
against the CPU mirror sim/ktgmc_searchprep_ref.cpp with an independent
Python golden (ANALYZE_SYNC == 1).

Run:  python3 python/run_mv_searchprep.py
"""
import os, random, subprocess, tempfile, sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BIN = os.path.join(tempfile.gettempdir(), "searchprep_ref")


def idiv(a, b):
    q = abs(a) // abs(b)
    return q if (a >= 0) == (b >= 0) else -q


def run_ref(nums):
    inf = os.path.join(tempfile.gettempdir(), "sp_in.txt")
    outf = os.path.join(tempfile.gettempdir(), "sp_out.txt")
    with open(inf, "w") as f:
        f.write(" ".join(map(str, nums)))
    subprocess.run([BIN, inf, outf], check=True)
    out = []
    with open(outf) as f:
        for line in f:
            out.append(list(map(int, line.split())))
    return out


def golden(par, vecs):
    (nBlkX, nBlkY, nBlkSize, nLogScale, nLambdaLevel, lsad,
     penaltyZero, penaltyGlobal, penaltyNew, nPel, nPad, nBlkSizeOvr,
     nExtendedWidth, nExtendedHeight) = par
    nBlk = nBlkX * nBlkY
    npad_s = nPad >> nLogScale
    out = []
    for by in range(nBlkY):
        for bx in range(nBlkX):
            blkIdx = bx + by * nBlkX
            X = nPad + nBlkSizeOvr * bx
            Y = nPad + nBlkSizeOvr * by
            dxmax = nPel * (nExtendedWidth - X - nBlkSize - nPad + npad_s) - 1
            dymax = nPel * (nExtendedHeight - Y - nBlkSize - nPad + npad_s) - 1
            dxmin = -nPel * (X - nPad + npad_s)
            dymin = -nPel * (Y - nPad + npad_s)
            p1 = -2
            if bx > 0:
                # ANALYZE_SYNC=1 (shipped): left predictor = FRESH own-region
                # neighbour.  Corrected 2026-10-28 from the copy-region
                # spelling (blkIdx - 1 + nBlk, the ANALYZE_SYNC=0 alternative)
                # after the CUDA twin measurement (cuda_search/
                # SEARCH_MEASURE_REPORT.md §3); see BLOCKSEARCH_MODEL.md §8b.
                p1 = blkIdx - 1
            pp2 = -2
            if by > 0:
                pp2 = blkIdx - nBlkX
            else:
                pp2 = p1
            p3 = -2
            if (by < nBlkY - 1) and (bx < nBlkX - 1):
                p3 = blkIdx + nBlkX + 1 + nBlk
            (vx, vy, s) = vecs[blkIdx]
            # lambda: left-assoc C int division -> truncating (sad_t=int)
            lam = idiv(nLambdaLevel * lsad, lsad + (s >> 1))
            lam = idiv(lam * lsad, lsad + (s >> 1))
            if by == 0:
                lam = 0
            out.append([dxmax, dymax, dxmin, dymin, -2, -1, blkIdx, p1, pp2, p3,
                        vx, vy, penaltyZero, penaltyGlobal, 0, penaltyNew, lam])
    return out


def main():
    subprocess.run(["g++", "-O2", "-std=gnu++17", "-Wl,--allow-multiple-definition", "-w",
                    os.path.join(REPO, "sim", "ktgmc_searchprep_ref.cpp"),
                    "-o", BIN], check=True)
    rng = random.Random(3)
    ok = True
    total = 0
    for _ in range(300):
        nBlkX = rng.randint(1, 10); nBlkY = rng.randint(1, 10)
        nBlkSize = rng.choice([8, 16, 32])
        nLogScale = rng.choice([1, 2, 3])
        nPel = rng.choice([1, 2, 4])
        nBlkSizeOvr = rng.randint(1, nBlkSize)
        nPad = rng.choice([0, 8, 16])
        # frame sized to keep everything valid/non-negative-ish; allow negatives too
        frame = 64 * nBlkX
        nExtendedWidth = frame
        nExtendedHeight = frame
        nLambdaLevel = rng.randint(1, 100)
        lsad = rng.randint(1, 2000)
        pz = rng.randint(0, 500); pg = rng.randint(0, 500); pn = rng.randint(0, 500)
        par = (nBlkX, nBlkY, nBlkSize, nLogScale, nLambdaLevel, lsad,
               pz, pg, pn, nPel, nPad, nBlkSizeOvr, nExtendedWidth, nExtendedHeight)
        vecs = [(rng.randint(-1000, 1000), rng.randint(-1000, 1000),
                 rng.randint(0, 2000)) for _ in range(nBlkX * nBlkY)]
        nums = list(par) + [v for t in vecs for v in t]
        got = run_ref(nums)
        exp = golden(par, vecs)
        total += 1
        if got != exp:
            ok = False
            print("prepare_search MISMATCH", par)
            for g, e in zip(got, exp):
                if g != e:
                    print("  got", g, "\n  exp", e)
            if total >= 3:
                break
    print(f"prepare_search: {'PASS' if ok else 'FAIL'} ({total} cases)")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
