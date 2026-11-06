#!/usr/bin/env python3
"""Validate the four KFM MergeStatic.cu kernels (kf_compare_frames,
kf_min_frames, kf_and_coefs, kf_merge_static) in
src/opencl/kfm/kernels/kfm_mergestatic.cl against the CPU mirror
sim/kfm_mergestatic_ref.cpp with an independent Python golden.

compare_frames / min_frames / merge_static are integer-exact (exact CPU twins
exist upstream).  and_coefs is float32 (no FMA); the mirror is compiled with
-ffp-contract=off and the golden emulates float32 per operation, so the two
match bit-for-bit.

Run:  python3 python/run_kfm_mergestatic.py
"""
import os, random, struct, subprocess, tempfile, sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BIN = os.path.join(tempfile.gettempdir(), "kfm_mergestatic_ref")


def F(v):
    return struct.unpack('f', struct.pack('f', v))[0]


def FI(v):
    """reinterpret float32 as int32 bits (to pass floats through the int file)"""
    return struct.unpack('i', struct.pack('f', v))[0]


def run_mirror(nums):
    inf = os.path.join(tempfile.gettempdir(), "kfms_in.txt")
    outf = os.path.join(tempfile.gettempdir(), "kfms_out.txt")
    with open(inf, "w") as f:
        f.write(" ".join(map(str, nums)))
    subprocess.run([BIN, inf, outf], check=True)
    with open(outf) as f:
        return [int(line) for line in f]


def fclamp(v, a, b):
    return a if v < a else (b if v > b else v)


def golden_compare(width, height, pitch, planes):
    out = []
    for yy in range(height):
        for xx in range(width):
            o = yy * pitch + xx
            vals = [p[o] for p in planes]
            out.append(max(vals) - min(vals))
    return out


def golden_min(width, height, pitch, planes):
    return [min(p[yy * pitch + xx] for p in planes)
            for yy in range(height) for xx in range(width)]


def golden_andcoefs(width, height, pitch, ic, idc, dstp, diffp):
    # ic, idc are the exact float32 inputs (host-computed 1.0f/thcombe, 1.0f/thdiff)
    neg1 = F(-1.0)
    one = F(1.0)
    nidc = F(-idc)
    f128 = F(128.0)
    half = F(0.5)
    out = []
    for yy in range(height):
        for xx in range(width):
            o = yy * pitch + xx
            # (float)dstp * ic + (-1.0f), per-op float32
            combe = fclamp(F(F(F(dstp[o]) * ic) + neg1), -0.5, 0.5)
            diffc = fclamp(F(F(F(diffp[o]) * nidc) + one), -0.5, 0.5)
            s = F(combe + diffc)
            if s < 0.0:
                s = 0.0
            tmp = F(F(F(s) * f128) + half)
            out.append(int(tmp))
    return out


def golden_merge(width, height, pitch, s60, s30, flag):
    out = []
    for yy in range(height):
        for xx in range(width):
            o = yy * pitch + xx
            coef, v30, v60 = flag[o], s30[o], s60[o]
            out.append((coef * v30 + (128 - coef) * v60 + 64) >> 7)
    return out


def main():
    subprocess.run(["g++", "-O2", "-std=gnu++17", "-Wl,--allow-multiple-definition", "-ffp-contract=off", "-w",
                    os.path.join(REPO, "sim", "kfm_mergestatic_ref.cpp"),
                    "-o", BIN], check=True)
    rng = random.Random(17)
    ok = True
    total = 0

    # --- C: compare_frames (KTemporalDiff) ---
    for _ in range(160):
        bits = rng.choice([8, 8, 16])
        maxv = 255 if bits == 8 else 65535
        width = rng.randint(1, 20) * 4      # width%4==0 (faithful config)
        height = rng.randint(1, 10)
        pitch = width + rng.choice([0, 2])
        n = pitch * height
        # 5 correlated frames with occasional motion bursts
        base = [rng.randint(0, maxv) for _ in range(n)]
        planes = []
        for _ in range(5):
            pl = [max(0, min(maxv, b + rng.randint(-maxv // 8, maxv // 8)))
                  for b in base]
            planes.append(pl)
        hdr = [ord('C'), width, height, pitch, n, n, n, n, n]
        got = run_mirror(hdr + planes[0] + planes[1] + planes[2] +
                         planes[3] + planes[4])
        exp = golden_compare(width, height, pitch, planes)
        total += 1
        if got != exp:
            ok = False
            for i, (g, e) in enumerate(zip(got, exp)):
                if g != e:
                    print("compare_frames MISMATCH", width, height, "px", i,
                          g, e)
                    break
            if total >= 3: break

    # --- N: min_frames (KAnalyzeStatic) ---
    for _ in range(120):
        bits = rng.choice([8, 8, 16])
        maxv = 255 if bits == 8 else 65535
        width = rng.randint(1, 20) * 4
        height = rng.randint(1, 10)
        pitch = width + rng.choice([0, 2])
        n = pitch * height
        base = [rng.randint(0, maxv) for _ in range(n)]
        planes = [[max(0, min(maxv, b + rng.randint(0, maxv // 8)))
                   for b in base] for _ in range(3)]
        hdr = [ord('N'), width, height, pitch, n, n, n]
        got = run_mirror(hdr + planes[0] + planes[1] + planes[2])
        exp = golden_min(width, height, pitch, planes)
        total += 1
        if got != exp:
            ok = False
            for i, (g, e) in enumerate(zip(got, exp)):
                if g != e:
                    print("min_frames MISMATCH", width, height, "px", i, g, e)
                    break
            if total >= 3: break

    # --- A: and_coefs (KAnalyzeStatic) ---
    for _ in range(140):
        bits = rng.choice([8, 8, 16])
        maxv = 255 if bits == 8 else 65535
        width = rng.randint(1, 20) * 4
        height = rng.randint(1, 10)
        pitch = width
        n = pitch * height
        # combe flag plane and diff flag plane, small values (coefficients)
        dstp = [rng.randint(0, 255) for _ in range(n)]
        diffp = [rng.randint(0, 255) for _ in range(n)]
        thcombe = rng.choice([10.0, 20.0, 30.0, 60.0])
        thdiff = rng.choice([5.0, 15.0, 30.0])
        invcombe = F(1.0 / thcombe)
        invdiff = F(1.0 / thdiff)
        hdr = [ord('A'), width, height, pitch, FI(invcombe), FI(invdiff), n, n]
        got = run_mirror(hdr + dstp + diffp)
        exp = golden_andcoefs(width, height, pitch, invcombe, invdiff, dstp,
                              diffp)
        total += 1
        if got != exp:
            ok = False
            for i, (g, e) in enumerate(zip(got, exp)):
                if g != e:
                    print("and_coefs MISMATCH", width, height, "px", i, g, e)
                    break
            if total >= 3: break

    # --- M: merge_static (KMergeStatic) ---
    for _ in range(160):
        bits = rng.choice([8, 8, 16])
        maxv = 255 if bits == 8 else 65535
        width = rng.randint(1, 20) * 4
        height = rng.randint(1, 10)
        pitch = width + rng.choice([0, 2])
        n = pitch * height
        s60 = [rng.randint(0, maxv) for _ in range(n)]
        s30 = [rng.randint(0, maxv) for _ in range(n)]
        # static coefficient plane in [0,128]
        flag = [rng.randint(0, 128) for _ in range(n)]
        hdr = [ord('M'), width, height, pitch, n, n, n]
        got = run_mirror(hdr + s60 + s30 + flag)
        exp = golden_merge(width, height, pitch, s60, s30, flag)
        total += 1
        if got != exp:
            ok = False
            for i, (g, e) in enumerate(zip(got, exp)):
                if g != e:
                    print("merge_static MISMATCH", width, height, "px", i, g, e)
                    break
            if total >= 3: break

    print(f"KFM MergeStatic (compare/min/and_coefs/merge): "
          f"{'PASS' if ok else 'FAIL'} ({total} cases)")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
