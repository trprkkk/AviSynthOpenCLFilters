#!/usr/bin/env python3
"""Validate kt_load_mv / kt_store_mv / kt_init_const_vec in
src/opencl/ktgmc/kernels/ktgmc_motion.cl against the CPU mirror
sim/ktgmc_mvio_ref.cpp with an independent Python golden.

Run:  python3 python/run_mv_io.py
"""
import os, random, subprocess, tempfile, sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BIN = os.path.join(tempfile.gettempdir(), "mvio_ref")


def build():
    subprocess.run(["g++", "-O2", "-std=gnu++17", "-Wl,--allow-multiple-definition", "-w",
                    os.path.join(REPO, "sim", "ktgmc_mvio_ref.cpp"),
                    "-o", BIN], check=True)


def run(mode_args, input_text):
    p = subprocess.run([BIN] + [str(a) for a in mode_args],
                       input=input_text, capture_output=True, text=True, check=True)
    return [line.split() for line in p.stdout.splitlines()]


def main():
    build()
    rng = random.Random(5)
    ok = True
    total = 0

    # mode 0: load  (VECTOR int3 -> int2 vector + int sad). pass-through.
    for _ in range(60):
        nBlk = rng.randint(1, 200)
        tri = [(rng.randint(-5000, 5000), rng.randint(-5000, 5000),
                rng.randint(0, 4000)) for _ in range(nBlk)]
        inp = "".join(f"{a} {b} {c}\n" for (a, b, c) in tri)
        got = run([0, nBlk], inp)
        # load: vector = (x,y), sad = sad -> re-emit same triplet
        exp = [list(map(str, t)) for t in tri]
        total += 1
        if got != exp:
            ok = False; print("load MISMATCH")

    # mode 1: store (int2 vector + int sad -> VECTOR int3). pass-through.
    for _ in range(60):
        nBlk = rng.randint(1, 200)
        tri = [(rng.randint(-5000, 5000), rng.randint(-5000, 5000),
                rng.randint(0, 4000)) for _ in range(nBlk)]
        inp = "".join(f"{a} {b} {c}\n" for (a, b, c) in tri)
        got = run([1, nBlk], inp)
        exp = [list(map(str, t)) for t in tri]
        total += 1
        if got != exp:
            ok = False; print("store MISMATCH")

    # mode 2: init_const_vec. each row: slot -2 = (0,0), slot -1 = (gx*nPel, gy*nPel)
    for _ in range(80):
        nRows = rng.randint(1, 30)
        pitch = rng.randint(nRows, 50)   # >= rows so sentinels land in-buffer
        gx = rng.randint(-2000, 2000); gy = rng.randint(-2000, 2000)
        nPel = rng.choice([1, 2, 4])
        got = run([2, nRows, pitch, gx, gy, nPel], "")
        exp = [["0", "0", str(gx * nPel), str(gy * nPel)] for _ in range(nRows)]
        total += 1
        if got != exp:
            ok = False
            print("init_const_vec MISMATCH", got, exp)

    # mode 3: load_mv_batch. split VECTOR to vec+sad AND copy through to out.
    # pass-through of (x,y,sad), same arithmetic as load.
    for _ in range(60):
        nBlk = rng.randint(1, 200)
        tri = [(rng.randint(-5000, 5000), rng.randint(-5000, 5000),
                rng.randint(0, 4000)) for _ in range(nBlk)]
        inp = "".join(f"{a} {b} {c}\n" for (a, b, c) in tri)
        got = run([3, nBlk], inp)
        exp = [list(map(str, t)) for t in tri]
        total += 1
        if got != exp:
            ok = False; print("load_mv_batch MISMATCH")

    print(f"MV IO kernels (load/store/init_const_vec/load_mv_batch): "
          f"{'PASS' if ok else 'FAIL'} ({total} cases)")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
