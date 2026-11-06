#!/usr/bin/env python3
"""Validate the four NNEDI3 pad/copy kernels in
src/opencl/nnedi3/kernels/nnedi3_pad.cl against the CPU mirror
sim/nnedi3_pad_ref.cpp with an independent Python golden.

Production invariant pinned by the case generator: pad_h/v reads are
interior-only iff pad < dim (hPad < w, vPad < h); upstream runs hPad=32 /
vPad=3 on full frames, so margin-reading updates are uncharted and
untested.  kl_pad_ref_and_copy_half reflects into range, which needs the
weaker hpad4 <= w4 / vpad <= h (single reflection overshoots otherwise —
also true upstream, whose frames dwarf the pads).

Run:  python3 python/run_nnedi3_pad.py
"""
import os, random, subprocess, tempfile, sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BIN = os.path.join(tempfile.gettempdir(), "nnedi3_pad_ref")


def run_mirror(nums):
    inf = os.path.join(tempfile.gettempdir(), "nnp_in.txt")
    outf = os.path.join(tempfile.gettempdir(), "nnp_out.txt")
    with open(inf, "w") as f:
        f.write(" ".join(map(str, nums)))
    subprocess.run([BIN, inf, outf], check=True)
    with open(outf) as f:
        return [int(line) for line in f]


def main():
    subprocess.run(["g++", "-O2", "-std=gnu++17", "-Wl,--allow-multiple-definition", "-ffp-contract=off", "-w",
                    os.path.join(REPO, "sim", "nnedi3_pad_ref.cpp"),
                    "-o", BIN], check=True)
    rng = random.Random(31)
    ok = True
    total = 0

    def check(name, got, exp, info):
        if got != exp:
            for i, (g, e) in enumerate(zip(got, exp)):
                if g != e:
                    print(name, "MISMATCH", info, "i", i, g, e)
                    break
            if len(got) != len(exp):
                print(name, "LENGTH MISMATCH", info, len(got), len(exp))
            return False
        return True

    def vals(n, maxv):
        out = []
        for _ in range(n):
            r = rng.random()
            if r < 0.25:
                out.append(rng.choice([0, 1, maxv - 1, maxv]))
            else:
                out.append(rng.randint(0, maxv))
        return out

    # A: pad_h (mirror, in-place; hPad < w required)
    for _ in range(150):
        bits = rng.choice([8, 8, 16])
        maxv = 255 if bits == 8 else 65535
        w = rng.randint(2, 40)
        h = rng.randint(1, 24)
        hPad = rng.choice([1, 2, 3, 4, 8, 16, 31, 32, 33])
        hPad = min(hPad, w - 1)
        pitch = w + 2 * hPad + rng.choice([0, 0, 1, 3])
        buf = vals(pitch * h, maxv)
        hdr = [ord('A'), w, h, pitch, hPad, pitch * h]
        got = run_mirror(hdr + buf)
        exp = list(buf)
        for yy in range(h):
            for i in range(hPad):
                exp[hPad - (i + 1) + yy * pitch] = buf[hPad + (i + 1) + yy * pitch]
                exp[hPad + w + i + yy * pitch] = buf[hPad + w - (i + 2) + yy * pitch]
        total += 1
        if not check("padh", got, exp, (w, h, hPad, bits)):
            ok = False

    # B: pad_v (mirror, in-place; vPad < h required)
    for _ in range(150):
        bits = rng.choice([8, 8, 16])
        maxv = 255 if bits == 8 else 65535
        w = rng.randint(1, 40)
        h = rng.randint(2, 24)
        vPad = rng.choice([1, 2, 3, 3, 4, 5, 6, 7, 8])
        vPad = min(vPad, h - 1)
        pitch = w + rng.choice([0, 0, 1, 3])
        buf = vals(pitch * (h + 2 * vPad), maxv)
        hdr = [ord('B'), w, h, pitch, vPad, pitch * (h + 2 * vPad)]
        got = run_mirror(hdr + buf)
        exp = list(buf)
        for i in range(vPad):
            for xx in range(w):
                exp[xx + (vPad - (i + 1)) * pitch] = buf[xx + (vPad + (i + 1)) * pitch]
                exp[xx + (vPad + h + i) * pitch] = buf[xx + (vPad + h - (i + 2)) * pitch]
        total += 1
        if not check("padv", got, exp, (w, h, vPad, bits)):
            ok = False

    # C: copy
    for _ in range(100):
        bits = rng.choice([8, 8, 16])
        maxv = 255 if bits == 8 else 65535
        w = rng.randint(1, 48)
        h = rng.randint(1, 24)
        sp = w + rng.choice([0, 0, 1, 4])
        dp = w + rng.choice([0, 1])
        src = vals(sp * h, maxv)
        hdr = [ord('C'), w, h, dp, sp, sp * h]
        got = run_mirror(hdr + src)
        exp = [src[xx + yy * sp] for yy in range(h) for xx in range(w)]
        total += 1
        if not check("copy", got, exp, (w, h, bits)):
            ok = False

    # D: pad_ref_and_copy_half (reflection + lane reversal on x-mirror)
    for _ in range(200):
        bits = rng.choice([8, 8, 16])
        maxv = 255 if bits == 8 else 65535
        w4 = rng.randint(1, 10)
        h = rng.randint(1, 20)
        hpad4 = min(rng.choice([1, 2, 3, 4, 5, 6, 7, 8, 8]), w4)
        vpad = min(rng.choice([1, 2, 3, 3, 4, 5, 6]), h)
        sp = w4 * 4 + rng.choice([0, 0, 1, 2])
        rw = (w4 + 2 * hpad4) * 4
        rh = h + 2 * vpad
        rp = rw + rng.choice([0, 0, 1, 2])
        dp = w4 * 4 + rng.choice([0, 1])
        src = vals(sp * h, maxv)
        hdr = [ord('D'), w4, h, rp, dp, sp, hpad4, vpad, sp * h]
        got = run_mirror(hdr + src)
        ref = {}
        dst = {}
        for yy in range(-vpad, h + vpad):
            for xx in range(-hpad4, w4 + hpad4):
                sx, padx = xx, False
                if sx < 0:
                    sx, padx = -sx - 1, True
                elif sx >= w4:
                    sx, padx = w4 - (sx - w4) - 1, True
                sy, pady = yy, False
                if sy < 0:
                    sy, pady = -sy - 1, True
                elif sy >= h:
                    sy, pady = h - (sy - h) - 1, True
                v = [src[sx * 4 + k + sy * sp] for k in range(4)]
                if padx:
                    v = v[::-1]
                for k in range(4):
                    ref[((xx + hpad4) * 4 + k, yy + vpad)] = v[k]
                    if not padx and not pady:
                        dst[(xx * 4 + k, yy)] = v[k]
        exp = [ref[(xx, yy)] for yy in range(rh) for xx in range(rw)]
        exp += [dst[(xx, yy)] for yy in range(h) for xx in range(w4 * 4)]
        total += 1
        if not check("padref", got, exp, (w4, h, hpad4, vpad, bits)):
            ok = False

    print(f"NNEDI3 pad/copy (padh/padv/copy/padref): "
          f"{'PASS' if ok else 'FAIL'} ({total} cases)")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
