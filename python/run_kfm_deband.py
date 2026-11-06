#!/usr/bin/env python3
"""Validate kf_deband_reduce_banding in src/opencl/kfm/kernels/kfm_deband.cl
against the CPU mirror sim/kfm_deband_ref.cpp with an independent Python golden.

The mirror transliterates the authoritative cpu_reduce_banding (KFM/KDeband.cu)
verbatim; this python golden is an independent implementation of the same
algorithm.  The pseudo-random byte stream `rand` is generated HERE (the CUDA
XorShift, seed 0 — the same one KDeband::CreateDebandRandom uses) and fed to the
mirror, so the two reduce-band implementations stay independent of it.  Matching
them bit-for-bit over the 3 sample_modes x blur_first, 8- and 16-bit, various
range/thresh, verifies the integer math that the .cl kernel transliterates.

Run:  python3 python/run_kfm_deband.py
"""
import os, random, subprocess, tempfile, sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BIN = os.path.join(tempfile.gettempdir(), "kfm_deband_ref")


def run_mirror(nums):
    inf = os.path.join(tempfile.gettempdir(), "kfd_in.txt")
    outf = os.path.join(tempfile.gettempdir(), "kfd_out.txt")
    with open(inf, "w") as f:
        f.write(" ".join(map(str, nums)))
    subprocess.run([BIN, inf, outf], check=True)
    with open(outf) as f:
        return [int(line) for line in f]


def xorshift_bytes(length, seed=0):
    x = (123456789 - seed) & 0xFFFFFFFF
    y, z, w = 362436069, 521288629, 88675123

    def nxt():
        nonlocal x, y, z, w
        t = x ^ ((x << 11) & 0xFFFFFFFF)
        x, y, z = y, z, w
        w ^= (t ^ (t >> 8) ^ (w >> 19)) & 0xFFFFFFFF
        w &= 0xFFFFFFFF
        return w

    buf = bytearray()
    while len(buf) + 4 <= length:
        buf += nxt().to_bytes(4, "little")
    if len(buf) < length:
        buf += nxt().to_bytes(4, "little")[: length - len(buf)]
    return bytes(buf)


def random_range(r, rng):
    return ((((rng << 1) + 1) * r) >> 8) - rng


def golden(width, height, pitch, rngv, thresh, mode, blur, maxv, src, rand):
    rs = width * height
    out = []
    for y in range(height):
        for x in range(width):
            off = y * pitch + x
            rl = min(rngv, y, height - y - 1, x, width - x - 1)
            refA = random_range(rand[off + rs * 0], rl)
            refB = random_range(rand[off + rs * 1], rl)
            s = src[off]
            if mode == 0:
                ref = refA * pitch + refB
                avg = src[off + ref]
                diff = abs(s - avg)
            elif mode == 1:
                ref = refA * pitch + refB
                rp, rm = src[off + ref], src[off - ref]
                avg = (rp + rm) >> 1
                diff = abs(s - avg) if blur else max(abs(s - rp), abs(s - rm))
            else:
                r0 = refA * pitch + refB
                r1 = refA - refB * pitch
                p0, m0, p1, m1 = src[off + r0], src[off - r0], src[off + r1], src[off - r1]
                avg = (p0 + m0 + p1 + m1) >> 2
                diff = (abs(s - avg) if blur
                        else max(abs(s - p0), abs(s - m0), abs(s - p1), abs(s - m1)))
            out.append(avg if diff <= thresh else s)
    return out


def main():
    subprocess.run(["g++", "-O2", "-std=gnu++17", "-Wl,--allow-multiple-definition", "-w",
                    os.path.join(REPO, "sim", "kfm_deband_ref.cpp"),
                    "-o", BIN], check=True)
    rng = random.Random(7)
    ok = True
    total = 0
    for _ in range(300):
        bits = rng.choice([8, 8, 16])
        maxv = 255 if bits == 8 else 65535
        width = rng.randint(2, 20)
        height = rng.randint(2, 20)
        pitch = width                       # faithful config (see .cl header)
        rangev = rng.randint(1, min(127, min(width, height) // 2) + 1)
        if rangev < 1: rangev = 1
        threshf = rng.choice([0.5, 1.0, 1.0, 3.0, 8.0])
        thresh = int(threshf * (1 << (bits - 8)) + 0.5)   # scaleParam
        mode = rng.randint(0, 2)
        blur = rng.choice([0, 1])
        rand = xorshift_bytes(2 * width * height)
        src = [rng.randint(0, maxv) for _ in range(pitch * height)]
        header = [width, height, pitch, rangev, thresh, mode, blur, maxv,
                  2 * width * height, pitch * height]
        tokens = header + list(rand) + src
        got = run_mirror(tokens)
        exp = golden(width, height, pitch, rangev, thresh, mode, blur, maxv,
                     src, rand)
        total += 1
        if got != exp:
            ok = False
            print("kfm_deband MISMATCH", width, height, pitch, rangev, thresh,
                  mode, blur, bits)
            for i, (g, e) in enumerate(zip(got, exp)):
                if g != e:
                    print("  px", i, "got", g, "exp", e)
                    if i > 10: break
            if total >= 3: break
    print(f"KFM KDeband reduce_banding: {'PASS' if ok else 'FAIL'} ({total} cases)")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
