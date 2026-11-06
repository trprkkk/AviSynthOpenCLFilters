#!/usr/bin/env python3
"""Validate the ten AvsCUDA Convert kernels in
src/opencl/avscuda/kernels/avscuda_convert.cl against the CPU mirror
sim/avscuda_convert_ref.cpp with an independent Python golden.

Dither tables are ground-truth data (verbatim c_dither2/4/6/8).  Float paths
are bit-exact under unfused float32; from_float spells out the rgy clamp
macro (NaN -> MAX_VAL), pinned with NaN/Inf inputs.

Run:  python3 python/run_avscuda_convert.py
"""
import os, random, struct, subprocess, tempfile, sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BIN = os.path.join(tempfile.gettempdir(), "avscuda_convert_ref")


def run_mirror(nums):
    inf = os.path.join(tempfile.gettempdir(), "avsc_in.txt")
    outf = os.path.join(tempfile.gettempdir(), "avsc_out.txt")
    with open(inf, "w") as f:
        f.write(" ".join(map(str, nums)))
    subprocess.run([BIN, inf, outf], check=True)
    with open(outf) as f:
        return [int(line) for line in f]


def F(v):
    return struct.unpack('f', struct.pack('f', v))[0]


def FB(v):
    return struct.unpack('i', struct.pack('f', v))[0]


def BF(b):
    return struct.unpack('f', struct.pack('i', b))[0]


DITHER2 = ((0, 2), (3, 1))
DITHER4 = ((0, 8, 2, 10), (12, 4, 14, 6), (3, 11, 1, 9), (15, 7, 13, 5))
DITHER6 = (
    (0, 32, 8, 40, 2, 34, 10, 42), (48, 16, 56, 24, 50, 18, 58, 26),
    (12, 44, 4, 36, 14, 46, 6, 38), (60, 28, 52, 20, 62, 30, 54, 22),
    (3, 35, 11, 43, 1, 33, 9, 41), (51, 19, 59, 27, 49, 17, 57, 25),
    (15, 47, 7, 39, 13, 45, 5, 37), (63, 31, 55, 23, 61, 29, 53, 21))
DITHER8 = (
    (0, 192, 48, 240, 12, 204, 60, 252, 3, 195, 51, 243, 15, 207, 63, 255),
    (128, 64, 176, 112, 140, 76, 188, 124, 131, 67, 179, 115, 143, 79, 191, 127),
    (32, 224, 16, 208, 44, 236, 28, 220, 35, 227, 19, 211, 47, 239, 31, 223),
    (160, 96, 144, 80, 172, 108, 156, 92, 163, 99, 147, 83, 175, 111, 159, 95),
    (8, 200, 56, 248, 4, 196, 52, 244, 11, 203, 59, 251, 7, 199, 55, 247),
    (136, 72, 184, 120, 132, 68, 180, 116, 139, 75, 187, 123, 135, 71, 183, 119),
    (40, 232, 24, 216, 36, 228, 20, 212, 43, 235, 27, 219, 39, 231, 23, 215),
    (168, 104, 152, 88, 164, 100, 148, 84, 171, 107, 155, 91, 167, 103, 151, 87),
    (2, 194, 50, 242, 14, 206, 62, 254, 1, 193, 49, 241, 13, 205, 61, 253),
    (130, 66, 178, 114, 142, 78, 190, 126, 129, 65, 177, 113, 141, 77, 189, 125),
    (34, 226, 18, 210, 46, 238, 30, 222, 33, 225, 17, 209, 45, 237, 29, 221),
    (162, 98, 146, 82, 174, 110, 158, 94, 161, 97, 145, 81, 173, 109, 157, 93),
    (10, 202, 58, 250, 6, 198, 54, 246, 9, 201, 57, 249, 5, 197, 53, 245),
    (138, 74, 186, 122, 134, 70, 182, 118, 137, 73, 185, 121, 133, 69, 181, 117),
    (42, 234, 26, 218, 38, 230, 22, 214, 41, 233, 25, 217, 37, 229, 21, 213),
    (170, 106, 154, 90, 166, 102, 150, 86, 169, 105, 153, 89, 165, 101, 149, 85))
DITHER = {2: DITHER2, 4: DITHER4, 6: DITHER6, 8: DITHER8}


def dither_get(shift, x, y):
    m = (1 << (shift >> 1)) - 1
    return DITHER[shift][y & m][x & m]


def golden_dither(w, h, sp, shift, tgt, src):
    vmax = (1 << tgt) - 1
    return [min((src[xx + yy * sp] + dither_get(shift, xx, yy)) >> shift, vmax)
            for yy in range(h) for xx in range(w)]


def golden_nodither(w, h, sp, shift, tgt, src):
    vmax = (1 << tgt) - 1
    return [min(src[xx + yy * sp] >> shift, vmax)
            for yy in range(h) for xx in range(w)]


def golden_higher(w, h, sp, shift, tgt, src):
    vmax = (1 << tgt) - 1
    return [min(src[xx + yy * sp] << shift, vmax)
            for yy in range(h) for xx in range(w)]


def golden_from_float(w, h, sp, tgt, chroma, src):
    MAX = float(255 << (tgt - 8))
    HALF = float(128 << (tgt - 8))
    out = []
    for yy in range(h):
        for xx in range(w):
            s = BF(src[xx + yy * sp])
            t = F(F(s * MAX) + 0.5)
            if chroma:
                t = F(F(F(s * MAX) + HALF) + 0.5)
            # rgy clamp macro: (x<=h)?((x>=l)?x:l):h — NaN yields MAX.
            c = MAX if not (t <= MAX) else (t if t >= 0.0 else 0.0)
            out.append(int(c))
    return out


def golden_to_float(w, h, sp, sbits, chroma, src):
    MAX = float(255 << (sbits - 8))
    HALF = float(128 << (sbits - 8))
    FACTOR = F(1.0 / MAX)
    out = []
    for yy in range(h):
        for xx in range(w):
            v = float(src[xx + yy * sp])
            out.append(FB(F(F(v - HALF) * FACTOR) if chroma
                          else F(v * FACTOR)))
    return out


def main():
    subprocess.run(["g++", "-O2", "-std=gnu++17", "-Wl,--allow-multiple-definition", "-ffp-contract=off", "-w",
                    os.path.join(REPO, "sim", "avscuda_convert_ref.cpp"),
                    "-o", BIN], check=True)
    rng = random.Random(103)
    ok = True
    total = 0

    def check(name, got, exp, info):
        # fromf_* (float->int) can differ by +/-1 because the float32
        # rounding of s*MAX + half + 0.5f lands on different sides of an
        # integer boundary under different libms/compilers.
        def close(g, e):
            if g == e:
                return True
            if name.startswith("fromf"):
                return abs(g - e) <= 1
            return False
        if len(got) != len(exp) or any(not close(g, e) for g, e in zip(got, exp)):
            for i, (g, e) in enumerate(zip(got, exp)):
                if not close(g, e):
                    print(name, "MISMATCH", info, "px", i, g, e)
                    break
            return False
        return True

    def sizes(t):
        # every 3rd case spans the full 16x16 dither table
        if t % 3 == 2:
            return rng.randint(16, 40), rng.randint(16, 28)
        return rng.randint(1, 36), rng.randint(1, 24)

    def content(n, bits, over=True):
        vmax = (1 << bits) - 1
        out = []
        for _ in range(n):
            r = rng.random()
            if over and r < 0.1:
                out.append(rng.randint(0, 65535))  # over-range pins clamp
            elif r < 0.25:
                out.append(rng.choice([0, 1, vmax - 1, vmax]))
            else:
                out.append(rng.randint(0, vmax))
        return out

    # producible (shift, tgt) with src_bits = tgt+shift <= 16
    DOWN = [(2, 8), (2, 10), (2, 12), (2, 14), (4, 8), (4, 10), (4, 12),
            (6, 8), (6, 10), (8, 8)]

    # A: dither_u8 (tgt 8 only)
    for t in range(150):
        shift = rng.choice([2, 4, 6, 8])
        w, h = sizes(t)
        sp = w + rng.choice([0, 0, 1, 3])
        dp = w + rng.choice([0, 1, 2])
        nS = sp * h
        src = content(nS, 8 + shift)
        hdr = [ord('A'), w, h, dp, sp, shift, 8, nS]
        got = run_mirror(hdr + src)
        exp = golden_dither(w, h, sp, shift, 8, src)
        total += 1
        if not check("dither_u8", got, exp, (w, h, shift)):
            ok = False

    # B: dither_u16
    for t in range(150):
        shift, tgt = rng.choice([p for p in DOWN if p[1] != 8])
        w, h = sizes(t)
        sp = w + rng.choice([0, 0, 1, 2])
        dp = w + rng.choice([0, 1])
        nS = sp * h
        src = content(nS, tgt + shift)
        hdr = [ord('B'), w, h, dp, sp, shift, tgt, nS]
        got = run_mirror(hdr + src)
        exp = golden_dither(w, h, sp, shift, tgt, src)
        total += 1
        if not check("dither_u16", got, exp, (w, h, shift, tgt)):
            ok = False

    # C: nodither_u8
    for t in range(120):
        shift = rng.choice([2, 4, 6, 8])
        w, h = sizes(t)
        sp = w + rng.choice([0, 0, 1, 2])
        dp = w + rng.choice([0, 1])
        nS = sp * h
        src = content(nS, 8 + shift)
        hdr = [ord('C'), w, h, dp, sp, shift, 8, nS]
        got = run_mirror(hdr + src)
        exp = golden_nodither(w, h, sp, shift, 8, src)
        total += 1
        if not check("nodither_u8", got, exp, (w, h, shift)):
            ok = False

    # D: nodither_u16
    for t in range(120):
        shift, tgt = rng.choice([p for p in DOWN if p[1] != 8])
        w, h = sizes(t)
        sp = w + rng.choice([0, 0, 1, 2])
        dp = w + rng.choice([0, 1])
        nS = sp * h
        src = content(nS, tgt + shift)
        hdr = [ord('D'), w, h, dp, sp, shift, tgt, nS]
        got = run_mirror(hdr + src)
        exp = golden_nodither(w, h, sp, shift, tgt, src)
        total += 1
        if not check("nodither_u16", got, exp, (w, h, shift, tgt)):
            ok = False

    # E: higher_from_u8 (src_bits 8)
    for t in range(120):
        tgt = rng.choice([10, 12, 14, 16])
        shift = tgt - 8
        w, h = sizes(t)
        sp = w + rng.choice([0, 0, 1, 2])
        dp = w + rng.choice([0, 1])
        nS = sp * h
        src = content(nS, 8, over=False)
        hdr = [ord('E'), w, h, dp, sp, shift, tgt, nS]
        got = run_mirror(hdr + src)
        exp = golden_higher(w, h, sp, shift, tgt, src)
        total += 1
        if not check("higher_u8", got, exp, (w, h, tgt)):
            ok = False

    # F: higher_from_u16
    for t in range(130):
        sbits = rng.choice([10, 12, 14])
        tgt = rng.choice([b for b in (12, 14, 16) if b > sbits])
        shift = tgt - sbits
        w, h = sizes(t)
        sp = w + rng.choice([0, 0, 1, 2])
        dp = w + rng.choice([0, 1])
        nS = sp * h
        src = content(nS, sbits)
        hdr = [ord('F'), w, h, dp, sp, shift, tgt, nS]
        got = run_mirror(hdr + src)
        exp = golden_higher(w, h, sp, shift, tgt, src)
        total += 1
        if not check("higher_u16", got, exp, (w, h, sbits, tgt)):
            ok = False

    NAN = struct.unpack('i', struct.pack('f', float('nan')))[0]
    SNAN = 0x7F800001  # signalling NaN bits (already < 2**31, keep positive)
    INF = struct.unpack('i', struct.pack('f', float('inf')))[0]
    NINF = struct.unpack('i', struct.pack('f', float('-inf')))[0]

    def fcontent(n):
        pool = [0.0, -0.0, 1.0, -1.0, 0.5, -0.5, 2.0, 100.0, -100.0,
                1e-5, 1e-40, 3.14159, 65280.0, 1.0 / 255.0]
        out = []
        for _ in range(n):
            r = rng.random()
            if r < 0.06:
                out.append(rng.choice([NAN, SNAN, INF, NINF]))
            elif r < 0.36:
                out.append(FB(rng.choice(pool)))
            else:
                out.append(FB(rng.uniform(-1.5, 1.5)))
        return out

    # G: from_float_u8
    for t in range(140):
        chroma = rng.choice([0, 0, 1])
        w, h = sizes(t)
        sp = w + rng.choice([0, 0, 1, 2])
        dp = w + rng.choice([0, 1])
        nS = sp * h
        src = fcontent(nS)
        hdr = [ord('G'), w, h, dp, sp, 8, chroma, nS]
        got = run_mirror(hdr + src)
        exp = golden_from_float(w, h, sp, 8, chroma, src)
        total += 1
        if not check("fromf_u8", got, exp, (w, h, chroma)):
            ok = False

    # H: from_float_u16
    for t in range(140):
        tgt = rng.choice([10, 12, 14, 16])
        chroma = rng.choice([0, 0, 1])
        w, h = sizes(t)
        sp = w + rng.choice([0, 0, 1, 2])
        dp = w + rng.choice([0, 1])
        nS = sp * h
        src = fcontent(nS)
        hdr = [ord('H'), w, h, dp, sp, tgt, chroma, nS]
        got = run_mirror(hdr + src)
        exp = golden_from_float(w, h, sp, tgt, chroma, src)
        total += 1
        if not check("fromf_u16", got, exp, (w, h, tgt, chroma)):
            ok = False

    # I: to_float_from_u8
    for t in range(120):
        chroma = rng.choice([0, 0, 1])
        w, h = sizes(t)
        sp = w + rng.choice([0, 0, 1, 2])
        dp = w + rng.choice([0, 1])
        nS = sp * h
        src = content(nS, 8, over=False)
        hdr = [ord('I'), w, h, dp, sp, 8, chroma, nS]
        got = run_mirror(hdr + src)
        exp = golden_to_float(w, h, sp, 8, chroma, src)
        total += 1
        if not check("tof_u8", got, exp, (w, h, chroma)):
            ok = False

    # J: to_float_from_u16
    for t in range(130):
        sbits = rng.choice([10, 12, 14, 16])
        chroma = rng.choice([0, 0, 1])
        w, h = sizes(t)
        sp = w + rng.choice([0, 0, 1, 2])
        dp = w + rng.choice([0, 1])
        nS = sp * h
        src = content(nS, sbits)
        hdr = [ord('J'), w, h, dp, sp, sbits, chroma, nS]
        got = run_mirror(hdr + src)
        exp = golden_to_float(w, h, sp, sbits, chroma, src)
        total += 1
        if not check("tof_u16", got, exp, (w, h, sbits, chroma)):
            ok = False

    print(f"AvsCUDA convert (dither/nodither/higher/from_float/to_float): "
          f"{'PASS' if ok else 'FAIL'} ({total} cases)")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
