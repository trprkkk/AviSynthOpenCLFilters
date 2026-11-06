#!/usr/bin/env python3
"""Validate the KFM KNoiseClip kernel (kf_noise_clip) in
src/opencl/kfm/kernels/kfm_noiseclip.cl against the CPU mirror
sim/kfm_noiseclip_ref.cpp with an independent Python golden.

Integer-exact, 8-bit only.  Per pixel:
  s = (src - noise + 256) >> 1   (128 == equal)
  out = dev_limitter(s, nmin, range) in {0,56,128,199,255}.

Run:  python3 python/run_kfm_noiseclip.py
"""
import os, random, subprocess, tempfile, sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BIN = os.path.join(tempfile.gettempdir(), "kfm_noiseclip_ref")


def run_mirror(nums):
    inf = os.path.join(tempfile.gettempdir(), "kfnc_in.txt")
    outf = os.path.join(tempfile.gettempdir(), "kfnc_out.txt")
    with open(inf, "w") as f:
        f.write(" ".join(map(str, nums)))
    subprocess.run([BIN, inf, outf], check=True)
    with open(outf) as f:
        return [int(line) for line in f]


def limitter(s, nmin, rng):
    if s == 128:
        return 128
    if s < 128:
        return 0 if ((127 - rng) < s and s < (128 - nmin)) else 56
    return 255 if ((128 + nmin) < s and s < (129 + rng)) else 199


def golden(width, height, pitch, nmin, rng, src, noise):
    out = []
    for yy in range(height):
        for xx in range(width):
            off = xx + yy * pitch
            s = (src[off] - noise[off] + 256) >> 1
            out.append(limitter(s, nmin, rng))
    return out


def main():
    subprocess.run(["g++", "-O2", "-std=gnu++17", "-Wl,--allow-multiple-definition", "-ffp-contract=off", "-w",
                    os.path.join(REPO, "sim", "kfm_noiseclip_ref.cpp"),
                    "-o", BIN], check=True)
    rng = random.Random(23)
    ok = True
    total = 0
    for _ in range(300):
        width = rng.randint(1, 24) * 4
        height = rng.randint(1, 10)
        pitch = width + rng.choice([0, 1])
        n = pitch * height
        nmin = rng.randint(0, 40)
        rangev = rng.choice([0, 1, 5, 20, 60, 128])
        src = [rng.randint(0, 255) for _ in range(n)]
        noise = [rng.randint(0, 255) for _ in range(n)]
        hdr = [ord('N'), width, height, pitch, nmin, rangev, n]
        got = run_mirror(hdr + src + noise)
        exp = golden(width, height, pitch, nmin, rangev, src, noise)
        total += 1
        if got != exp:
            ok = False
            for i, (g, e) in enumerate(zip(got, exp)):
                if g != e:
                    print("KNoiseClip MISMATCH", width, height, nmin, rangev,
                          "px", i, g, e)
                    break
            if total >= 3: break
    print(f"KFM KNoiseClip noise_clip: {'PASS' if ok else 'FAIL'} "
          f"({total} cases)")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
