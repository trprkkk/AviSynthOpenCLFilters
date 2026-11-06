#!/usr/bin/env python3
"""Validate the AvsCUDA Invert kernels (ka_invert_plane_u8/u16/f32,
ka_invert_rgb) in src/opencl/avscuda/kernels/avscuda_filters.cl against the
CPU mirror sim/avscuda_filters_ref.cpp with an independent Python golden.

Lane extraction: the mirror shifts the word masks; the golden uses
precomputed lane tables.  Float is bit-exact under unfused float32.

Run:  python3 python/run_avscuda_filters.py
"""
import os, random, struct, subprocess, tempfile, sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BIN = os.path.join(tempfile.gettempdir(), "avscuda_filters_ref")


def run_mirror(nums):
    inf = os.path.join(tempfile.gettempdir(), "avsf_in.txt")
    outf = os.path.join(tempfile.gettempdir(), "avsf_out.txt")
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


def to_signed(u):
    return u - 2**32 if u >= 2**31 else u


def golden_inv_u8(w, h, pitch, mask0, src):
    lanes = [(mask0 >> (8 * i)) & 0xFF for i in range(4)]
    return [src[xx + yy * pitch] ^ lanes[xx & 3]
            for yy in range(h) for xx in range(w)]


def golden_inv_u16(w, h, pitch, mask0, mask1, src):
    lanes = [mask0 & 0xFFFF, (mask0 >> 16) & 0xFFFF,
             mask1 & 0xFFFF, (mask1 >> 16) & 0xFFFF]
    return [src[xx + yy * pitch] ^ lanes[xx & 3]
            for yy in range(h) for xx in range(w)]


def golden_inv_f32(w, h, pitch, src):
    return [FB(F(1.0 - BF(src[xx + yy * pitch])))
            for yy in range(h) for xx in range(w)]


def golden_inv_rgb(w, h, elp, masks, maxv, src):
    out = []
    for yy in range(h):
        for xx in range(w):
            for c in range(3):
                out.append((src[3 * xx + c + yy * elp] ^ masks[c]) & maxv)
    return out


def main():
    subprocess.run(["g++", "-O2", "-std=gnu++17", "-Wl,--allow-multiple-definition", "-ffp-contract=off", "-w",
                    os.path.join(REPO, "sim", "avscuda_filters_ref.cpp"),
                    "-o", BIN], check=True)
    rng = random.Random(102)
    ok = True
    total = 0

    def check(name, got, exp, info):
        if got != exp:
            for i, (g, e) in enumerate(zip(got, exp)):
                if g != e:
                    print(name, "MISMATCH", info, "px", i, g, e)
                    break
            return False
        return True

    # A: invert u8 (incl. non-mult-4 widths: no word overhang)
    for _ in range(200):
        w = rng.randint(1, 40)
        h = rng.randint(1, 24)
        pitch = w + rng.choice([0, 0, 1, 3])
        mask0 = rng.choice([0xFFFFFFFF, 0xFFFFFFFF, 0, 0x00FF0000,
                            0x0000FF00, 0x000000FF, 0xFF000000,
                            0x00FFFFFF, rng.randint(0, 0xFFFFFFFF)])
        n = pitch * h
        src = [rng.randint(0, 255) for _ in range(n)]
        hdr = [ord('A'), w, h, pitch, to_signed(mask0), n]
        got = run_mirror(hdr + src)
        exp = golden_inv_u8(w, h, pitch, mask0, src)
        total += 1
        if not check("inv_u8", got, exp, (w, h, hex(mask0))):
            ok = False

    # B: invert u16 (uniform planar masks + RGB64 channel + random)
    for t in range(200):
        w = rng.randint(1, 36)
        h = rng.randint(1, 24)
        pitch = w + rng.choice([0, 0, 1, 2])
        pick = t % 4
        if pick == 0:
            m = rng.choice([0x0000, 0xFFFF, rng.randint(0, 0xFFFF)])
            mask0 = mask1 = (m << 16) | m  # uniform planar
        elif pick == 1:  # RGB64 single-channel
            ch = rng.randint(0, 3)
            lanes = [0, 0, 0, 0]
            lanes[ch] = 0xFFFF
            mask0 = (lanes[1] << 16) | lanes[0]
            mask1 = (lanes[3] << 16) | lanes[2]
        else:
            mask0 = rng.randint(0, 0xFFFFFFFF)
            mask1 = rng.randint(0, 0xFFFFFFFF)
        n = pitch * h
        src = [rng.randint(0, 65535) for _ in range(n)]
        hdr = [ord('B'), w, h, pitch, to_signed(mask0), to_signed(mask1), n]
        got = run_mirror(hdr + src)
        exp = golden_inv_u16(w, h, pitch, mask0, mask1, src)
        total += 1
        if not check("inv_u16", got, exp, (w, h, hex(mask0), hex(mask1))):
            ok = False

    # C: invert f32
    for _ in range(150):
        w = rng.randint(1, 32)
        h = rng.randint(1, 20)
        pitch = w + rng.choice([0, 0, 1, 2])
        n = pitch * h
        pool = [0.0, -0.0, 1.0, -1.0, 0.5, 2.0, 1e-5, 1e-40, 3.14159]
        src = [FB(rng.choice(pool) if rng.random() < 0.3
                  else rng.uniform(-1.0, 2.0)) for _ in range(n)]
        hdr = [ord('C'), w, h, pitch, n]
        got = run_mirror(hdr + src)
        exp = golden_inv_f32(w, h, pitch, src)
        total += 1
        if not check("inv_f32", got, exp, (w, h)):
            ok = False

    # D: invert rgb (RGB24 u8 / RGB48 u16; full + partial masks)
    for _ in range(200):
        bits = rng.choice([8, 8, 16])
        maxv = 255 if bits == 8 else 65535
        w = rng.randint(1, 28)
        h = rng.randint(1, 20)
        elp = 3 * w + rng.choice([0, 0, 1, 3])
        full = 0xFF if bits == 8 else 0xFFFF
        partial = [0x0F, 0xF0, 0x55, 0xFF00] if bits == 8 else \
            [0x00FF, 0xFF00, 0x0F0F, 0x5555]
        masks = [rng.choice([0, 0, full, full, rng.choice(partial),
                             rng.randint(0, full)]) for _ in range(3)]
        n = elp * h
        src = [rng.randint(0, maxv) for _ in range(n)]
        hdr = [ord('D'), w, h, elp] + masks + [maxv, n]
        got = run_mirror(hdr + src)
        exp = golden_inv_rgb(w, h, elp, masks, maxv, src)
        total += 1
        if not check("inv_rgb", got, exp, (w, h, bits, masks)):
            ok = False

    print(f"AvsCUDA filters (inv_u8/inv_u16/inv_f32/inv_rgb): "
          f"{'PASS' if ok else 'FAIL'} ({total} cases)")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
