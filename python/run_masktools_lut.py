#!/usr/bin/env python3
"""Validate the five masktools kernels in
src/opencl/masktools/kernels/masktools_lut.cl against the CPU mirror
sim/masktools_lut_ref.cpp with an independent Python golden.

The interesting part is the LUT index arithmetic, so the golden derives it
WITHOUT the shift/mask spelling the port uses: it computes the index in
mixed radix (x*B^2 + y*B + z with B = 2**lut_bits) and applies the 16-bit
masking as a modulo (idx % (mask+1)), which is the same map for the
power-of-two masks upstream builds.

Pinned upstream defect (see the kernel header): the CUDA dispatcher passes
bits_per_pixel = 8 for every 16-bit depth while the host table is built
with the real depth, so on the 16-bit path lut_xy collapses to Y & 255 and
lut_xyz to Z & 255.  Dedicated cases assert that collapse explicitly, so
the behaviour cannot change silently.

Run:  python3 python/run_masktools_lut.py
"""
import os, random, struct, subprocess, tempfile, sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BIN = os.path.join(tempfile.gettempdir(), "masktools_lut_ref")


def run_mirror(nums):
    inf = os.path.join(tempfile.gettempdir(), "mtl_in.txt")
    outf = os.path.join(tempfile.gettempdir(), "mtl_out.txt")
    with open(inf, "w") as f:
        f.write(" ".join(map(str, nums)))
    subprocess.run([BIN, inf, outf], check=True)
    with open(outf) as f:
        return [int(line) for line in f]


def FB(v):
    return struct.unpack('i', struct.pack('f', v))[0]


def proc_lut(i, px):
    """Procedural stand-in table (same definition as the mirror)."""
    h = (i * 2654435761) & 0xFFFFFFFF
    h ^= h >> 15
    return h & (0xFF if px == 1 else 0xFFFF)


def main():
    subprocess.run(["g++", "-O2", "-std=gnu++17", "-Wl,--allow-multiple-definition", "-ffp-contract=off", "-w",
                    os.path.join(REPO, "sim", "masktools_lut_ref.cpp"),
                    "-o", BIN], check=True)
    rng = random.Random(1717)
    ok = True
    total = 0
    collapse_checked = 0

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

    def plane(nv, maxv):
        out = []
        for _ in range(nv):
            r = rng.random()
            if r < 0.25:
                out.append(rng.choice([0, 1, maxv - 1, maxv]))
            else:
                out.append(rng.randint(0, maxv))
        return out

    # A: fill (uchar4 / ushort4)
    for _ in range(120):
        px = rng.choice([1, 1, 2])
        maxv = 255 if px == 1 else 65535
        w4 = rng.randint(1, 30)
        h = rng.randint(1, 20)
        pitch4 = w4 + rng.choice([0, 0, 1, 3])
        v = rng.choice([0, 1, 127, maxv, rng.randint(0, maxv)])
        got = run_mirror([ord('F'), v, w4, h, pitch4])
        exp = [-1] * (pitch4 * 4 * h)
        for yy in range(h):
            for xx in range(w4):
                for k in range(4):
                    exp[(xx + yy * pitch4) * 4 + k] = v
        total += 1
        if not check("fill", got, exp, (w4, h, px)):
            ok = False

    # B: fill f32 (memset_plane_32_cuda)
    for _ in range(60):
        w4 = rng.randint(1, 24)
        h = rng.randint(1, 16)
        pitch4 = w4 + rng.choice([0, 1, 2])
        v = rng.choice([0.0, -0.0, 1.0, -1.0, 0.5, float('inf'),
                        float('-inf'), rng.uniform(-1e3, 1e3)])
        vb = FB(v)
        got = run_mirror([ord('G'), vb, w4, h, pitch4])
        exp = [-1] * (pitch4 * 4 * h)
        for yy in range(h):
            for xx in range(w4):
                for k in range(4):
                    exp[(xx + yy * pitch4) * 4 + k] = vb
        total += 1
        if not check("fillf32", got, exp, (w4, h)):
            ok = False

    # C: byte copy (uchar4, independent of pixel format)
    for _ in range(100):
        w4 = rng.randint(1, 32)
        h = rng.randint(1, 20)
        sp4 = w4 + rng.choice([0, 0, 1, 2])
        dp4 = w4 + rng.choice([0, 1])
        src = plane(sp4 * 4 * h, 255)
        got = run_mirror([ord('C'), w4, h, dp4, sp4, len(src)] + src)
        exp = [-1] * (dp4 * 4 * h)
        for yy in range(h):
            for xx in range(w4):
                for k in range(4):
                    exp[(xx + yy * dp4) * 4 + k] = src[(xx + yy * sp4) * 4 + k]
        total += 1
        if not check("copy", got, exp, (w4, h)):
            ok = False

    # D: lut_x  (8-bit: raw index; 16-bit: idx & mask)
    for _ in range(100):
        px = rng.choice([1, 1, 2])
        maxv = 255 if px == 1 else 65535
        mask = 255                      # upstream always (1 << 8) - 1
        w4 = rng.randint(1, 20)
        h = rng.randint(1, 14)
        pitch4 = w4 + rng.choice([0, 0, 1, 2])
        lut = [rng.randint(0, maxv) for _ in range(256)]
        src = plane(pitch4 * 4 * h, maxv)
        got = run_mirror([ord('X'), px, pitch4, w4, h, mask, len(lut)]
                         + lut + [len(src)] + src)
        exp = [-1] * (pitch4 * 4 * h)
        for yy in range(h):
            for xx in range(w4):
                for k in range(4):
                    o = (xx + yy * pitch4) * 4 + k
                    idx = src[o] if px == 1 else src[o] % (mask + 1)
                    exp[o] = lut[idx]
        total += 1
        if not check("lut_x", got, exp, (w4, h, px)):
            ok = False

    # E: lut_xy  (mixed-radix index, procedural table for the 8-bit range)
    for _ in range(100):
        px = rng.choice([1, 1, 2])
        maxv = 255 if px == 1 else 65535
        bits, mask = 8, 255
        B = 2 ** bits
        w4 = rng.randint(1, 16)
        h = rng.randint(1, 12)
        pitch4 = w4 + rng.choice([0, 0, 1, 2])
        n = pitch4 * 4 * h
        s0, s1 = plane(n, maxv), plane(n, maxv)
        got = run_mirror([ord('Y'), px, pitch4, w4, h, mask, bits, 0]
                         + [n] + s0 + s1)
        exp = [-1] * n
        for yy in range(h):
            for xx in range(w4):
                for k in range(4):
                    o = (xx + yy * pitch4) * 4 + k
                    idx = s0[o] * B + s1[o]
                    if px != 1:
                        idx %= mask + 1
                    exp[o] = proc_lut(idx, px)
        total += 1
        if not check("lut_xy", got, exp, (w4, h, px)):
            ok = False

    # F: lut_xyz
    for _ in range(100):
        px = rng.choice([1, 1, 2])
        maxv = 255 if px == 1 else 65535
        bits, mask = 8, 255
        B = 2 ** bits
        w4 = rng.randint(1, 14)
        h = rng.randint(1, 10)
        pitch4 = w4 + rng.choice([0, 0, 1, 2])
        n = pitch4 * 4 * h
        s0, s1, s2 = plane(n, maxv), plane(n, maxv), plane(n, maxv)
        got = run_mirror([ord('Z'), px, pitch4, w4, h, mask, bits, 0]
                         + [n] + s0 + s1 + s2)
        exp = [-1] * n
        for yy in range(h):
            for xx in range(w4):
                for k in range(4):
                    o = (xx + yy * pitch4) * 4 + k
                    idx = s0[o] * B * B + s1[o] * B + s2[o]
                    if px != 1:
                        idx %= mask + 1
                    exp[o] = proc_lut(idx, px)
        total += 1
        if not check("lut_xyz", got, exp, (w4, h, px)):
            ok = False

    # G: pin the 16-bit collapse explicitly (upstream defect, see header):
    #    with bits=8/mask=255 the xy index loses X entirely and xyz loses
    #    X and Y.  Same planes, different leading clips -> same output.
    for _ in range(30):
        w4, h, pitch4 = rng.randint(1, 8), rng.randint(1, 6), rng.randint(1, 10)
        pitch4 = max(pitch4, w4)
        n = pitch4 * 4 * h
        base = plane(n, 65535)
        alt = plane(n, 65535)
        other = plane(n, 65535)
        a = run_mirror([ord('Y'), 2, pitch4, w4, h, 255, 8, 0, n]
                       + base + other)
        b = run_mirror([ord('Y'), 2, pitch4, w4, h, 255, 8, 0, n]
                       + alt + other)
        c = run_mirror([ord('Z'), 2, pitch4, w4, h, 255, 8, 0, n]
                       + base + alt + other)
        total += 1
        if a != b:
            print("COLLAPSE xy: X still affects the 16-bit result")
            ok = False
        if a != c:
            print("COLLAPSE xyz: does not match the xy/Z-only collapse")
            ok = False
        # and it really is lut[Z & 255], not something else
        exp = [-1] * n
        for yy in range(h):
            for xx in range(w4):
                for k in range(4):
                    o = (xx + yy * pitch4) * 4 + k
                    exp[o] = proc_lut(other[o] % 256, 2)
        if not check("collapse_value", c, exp, (w4, h)):
            ok = False
        collapse_checked += 1

    print(f"masktools (fill/copy/lut x,xy,xyz): {'PASS' if ok else 'FAIL'} "
          f"({total} cases, {collapse_checked} 16-bit collapse pins)")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
