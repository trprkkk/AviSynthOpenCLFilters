#!/usr/bin/env python3
"""Validate the MV-aux integer kernels from src/opencl/ktgmc/kernels/ktgmc_motion.cl:
    kt_write_default_mv   (def)
    kt_scene_change       (sc)
    kt_scene_change_x2    (scx2)
    kt_short_to_byte      (stb, shift 5 for 8-bit / 11 for 16-bit)
against the CPU mirror sim/ktgmc_mvaux_ref.cpp with an independent Python golden.

Run:  python3 python/run_mv_aux.py
"""
import os, random, subprocess, tempfile, sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BIN = os.path.join(tempfile.gettempdir(), "mvaux_ref")


def run_ref(typ, nums):
    inf = os.path.join(tempfile.gettempdir(), "mvaux_in.txt")
    outf = os.path.join(tempfile.gettempdir(), "mvaux_out.txt")
    with open(inf, "w") as f:
        f.write(" ".join(map(str, nums)))
    subprocess.run([BIN, typ, inf, outf], check=True)
    with open(outf) as f:
        return [l.rstrip("\n") for l in f if l.strip() != ""]


def main():
    subprocess.run(["g++", "-O2", "-std=gnu++17", "-Wl,--allow-multiple-definition", "-w",
                    os.path.join(REPO, "sim", "ktgmc_mvaux_ref.cpp"),
                    "-o", BIN], check=True)
    rng = random.Random(7)
    ok = True
    total = 0

    # def: n, verybigSAD
    for _ in range(50):
        n = rng.randint(1, 200); big = rng.randint(1, 1000000)
        got = run_ref("def", [n, big])
        exp = ["0 0 %d" % big] * n
        total += 1
        if got != exp: ok = False; print("def MISMATCH", n, big)

    # sc
    for _ in range(100):
        n = rng.randint(1, 500); nTh1 = rng.randint(0, 5000)
        sads = [rng.randint(0, 10000) for _ in range(n)]
        got = run_ref("sc", [nTh1] + sads)
        exp = [str(sum(1 for s in sads if s > nTh1))]
        total += 1
        if got != exp: ok = False; print("sc MISMATCH")

    # scx2
    for _ in range(100):
        n = rng.randint(1, 500); nTh1 = rng.randint(0, 5000)
        A = [rng.randint(0, 10000) for _ in range(n)]
        B = [rng.randint(0, 10000) for _ in range(n)]
        got = run_ref("scx2", [nTh1] + A + B)
        exp = [str(sum(1 for s in A if s > nTh1)), str(sum(1 for s in B if s > nTh1))]
        total += 1
        if got != exp: ok = False; print("scx2 MISMATCH")

    # stb : width height pitch shift maxval tmp...
    # height full pitch rows; tmp non-negative.
    for _ in range(60):
        w = rng.randint(1, 20); pitch = w + rng.randint(0, 5); h = rng.randint(1, 20)
        shift = rng.choice([5, 11]); maxv = 255 if shift == 5 else 65535
        tmp = [rng.randint(0, (1 << 21) - 1) for _ in range(pitch * h)]
        got = run_ref("stb", [w, h, pitch, shift, maxv] + tmp)
        exp = []
        for i in range(h):
            for x in range(w):
                v = tmp[x + i * pitch] >> shift
                v = min(v, maxv)
                v = max(v, 0)
                exp.append(str(v))
        total += 1
        if got != exp: ok = False; print("stb MISMATCH", w, h, shift)

    print(f"MV-aux kernels: {'PASS' if ok else 'FAIL'} ({total} cases)")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
