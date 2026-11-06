"""Independent golden vs sim/kfm_decombeucf_ref.cpp (DecombeUCF.cu twins).

Modes: U init_uint64, F field_diff, A add_block_sum, X block_sum_max,
N analyze_noise, D analyze_diff.  All reductions are serial here (the reduced
ops are int add / int max, so every tree shape is value-identical).
"""
import os
import random
import subprocess
import sys
import tempfile

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BIN = os.path.join(tempfile.gettempdir(), "kfm_decombeucf_ref")


def run_mirror(nums):
    inf = os.path.join(tempfile.gettempdir(), "kdu_in.txt")
    outf = os.path.join(tempfile.gettempdir(), "kdu_out.txt")
    with open(inf, "w") as f:
        f.write(" ".join(map(str, nums)) + "\n")
    subprocess.run([BIN, inf, outf], check=True)
    with open(outf) as f:
        return [int(line) for line in f]


def combe5(a, b, c, d, e):
    return abs(a + 4 * c + e - 3 * (b + d))


def golden_field_diff(width, height, pitch, nt, init, src):
    base = 2 * pitch
    total = init
    for y in range(height):
        for x in range(width):
            v = combe5(src[base + x + (y - 2) * pitch],
                       src[base + x + (y - 1) * pitch],
                       src[base + x + (y + 0) * pitch],
                       src[base + x + (y + 1) * pitch],
                       src[base + x + (y + 2) * pitch])
            if v > nt:
                total += v
    return [total]


def golden_add_block_sum(width, height, pitch, bw, bh, bpitch, bs,
                         s0, s1, init_abs, init_sig):
    ab = list(init_abs)
    sg = list(init_sig)
    for by in range(bh):
        for bx in range(bw):
            a_sum = 0
            s_sum = 0
            for ty in range(bs):
                y = by * bs + ty
                for tx in range(bs):
                    x = bx * bs + tx
                    if x >= width or y >= height:
                        continue
                    d = s0[y * pitch + x] - s1[y * pitch + x]
                    a_sum += abs(d)
                    s_sum += d
            cell = bx + by * bpitch
            ab[cell] += a_sum
            sg[cell] += s_sum
    return ab + sg


def golden_block_sum_max(bw, bh, bpitch, init_max, ab, sg):
    best = 0
    for y in range(bh):
        for x in range(bw):
            c4 = (x + y * bpitch) * 4
            for k in range(4):
                m = ab[c4 + k] + sg[c4 + k] * 4
                if m > best:
                    best = m
    return [init_max if init_max > best else best]


def golden_analyze_noise(width, height, pitch, init, s0, s1, s2):
    r = list(init)
    for y in range(height):
        for x in range(width):
            a = s0[y * pitch + x]
            b = s1[y * pitch + x]
            c = s2[y * pitch + x]
            r[0] += abs(a - 128)
            r[1] += abs(b - 128)
            r[2] += abs(b - a)
            r[3] += abs(c - b)
    return r


def golden_analyze_diff(width, height, pitch, i0, i1, f0, f1):
    base = 2 * pitch
    r0, r1 = i0, i1
    for y in range(height):
        for x in range(width):
            r0 += combe5(f0[base + x + (y - 2) * pitch],
                         f0[base + x + (y - 1) * pitch],
                         f0[base + x + (y + 0) * pitch],
                         f0[base + x + (y + 1) * pitch],
                         f0[base + x + (y + 2) * pitch])
            if y & 1:
                taps = (f0[base + x + (y - 2) * pitch],
                        f1[base + x + (y - 1) * pitch],
                        f0[base + x + (y + 0) * pitch],
                        f1[base + x + (y + 1) * pitch],
                        f0[base + x + (y + 2) * pitch])
            else:
                taps = (f1[base + x + (y - 2) * pitch],
                        f0[base + x + (y - 1) * pitch],
                        f1[base + x + (y + 0) * pitch],
                        f0[base + x + (y + 1) * pitch],
                        f1[base + x + (y + 2) * pitch])
            r1 += combe5(*taps)
    return [r0, r1]


def main():
    subprocess.run(["g++", "-O2", "-std=gnu++17", "-Wl,--allow-multiple-definition", "-ffp-contract=off", "-w",
                    os.path.join(REPO, "sim", "kfm_decombeucf_ref.cpp"),
                    "-o", BIN], check=True)
    rng = random.Random(97)
    ok = True
    total = 0

    def check(name, got, exp, info):
        nonlocal total, ok
        total += 1
        if got != exp:
            ok = False
            print(name, "MISMATCH", info, "got", got[:8], "exp", exp[:8])
            return False
        return True

    # U: init_uint64 zero-fill (upstream launch counts: 1 and 2*N)
    for length in [1, 2, 3, 4, 7, 8, 9, 16, 17, 64]:
        check("U", run_mirror(["U", length]), [0] * length, length)

    # F: field_diff (production widths are px multiples of 4; a few odd
    # widths pin the guard path)
    nt_pool = [-1, 0, 1, 2, 5, 63, 64, 255, 256, 1500, 1530, 393209,
               393210, 1000000]
    for t in range(170):
        maxv = rng.choice([255, 65535])
        if t % 17 == 16:
            width = rng.choice([1, 2, 3, 5, 6, 7, 9, 31])
        else:
            width = rng.choice([4, 8, 12, 16, 20, 28, 32, 36, 40, 48, 64])
        height = rng.choice([1, 2, 3, 4, 5, 7, 8, 13, 16, 20, 31, 40])
        pitch = width + rng.choice([0, 0, 1, 2, 3])
        nt = rng.choice(nt_pool) if t % 3 else rng.randint(-2, 400000)
        init = rng.choice([0, 0, 1, 123456789, 2**40 + 7])
        nP = pitch * (height + 4)
        craft = t % 5
        if craft == 0:  # constant field -> combe 0 -> fires iff nt < 0
            src = [rng.randint(0, maxv)] * nP
        elif craft == 1:  # vertical ramp
            src = [((r * 7 + c * 13) % (maxv + 1))
                   for r in range(height + 4) for c in range(pitch)]
        elif craft == 2:  # row-alternating comb
            src = [0 if r % 2 == 0 else maxv
                   for r in range(height + 4) for _ in range(pitch)]
        else:
            src = [rng.randint(0, maxv) for _ in range(nP)]
        got = run_mirror(["F", width, height, pitch, nt, init, nP] + src)
        exp = golden_field_diff(width, height, pitch, nt, init, src)
        check("F", got, exp, (width, height, pitch, nt, maxv))

    # A: add_block_sum, BLOCK_SIZE 32/16/8/4 x depths 8/16
    for t in range(190):
        bs = [32, 16, 8, 4][t % 4]
        maxv = 255 if t % 2 == 0 else 65535
        bw = rng.choice([1, 2, 3, 4])
        bh = rng.choice([1, 2, 3])
        width = rng.choice([1, bs - 1, bs, bs + 1, 2 * bs - 1, 2 * bs,
                            2 * bs + 1, 3 * bs])
        height = rng.choice([1, bs - 1, bs, bs + 1, 2 * bs - 1, 2 * bs,
                             2 * bs + 1])
        width = min(width, bw * bs + rng.choice([0, 1, 3]))
        height = min(height, bh * bs + rng.choice([0, 1]))
        width = max(width, 1)
        height = max(height, 1)
        pitch = width + rng.choice([0, 1, 2])
        bpitch = bw + rng.choice([0, 0, 1])
        nP = pitch * height
        nB = bpitch * bh
        craft = t % 4
        if craft == 0:  # identical planes -> out == init
            s0 = [rng.randint(0, maxv) for _ in range(nP)]
            s1 = list(s0)
        elif craft == 1:  # constant offset k -> sig pins sign
            k = rng.choice([-7, -1, 1, 5])
            s0 = [rng.randint(0, maxv) for _ in range(nP)]
            s1 = [min(max(v + k, 0), maxv) for v in s0]
        else:
            s0 = [rng.randint(0, maxv) for _ in range(nP)]
            s1 = [rng.randint(0, maxv) for _ in range(nP)]
        init_abs = [rng.choice([0, 0, 5, 1000000]) for _ in range(nB)]
        init_sig = [rng.choice([0, 0, -999999, -3, 7, 1000000])
                    for _ in range(nB)]
        nums = (["A", width, height, pitch, bw, bh, bpitch, bs, nP, nB] +
                s0 + s1 + init_abs + init_sig)
        exp = golden_add_block_sum(width, height, pitch, bw, bh, bpitch, bs,
                                   s0, s1, init_abs, init_sig)
        check("A", run_mirror(nums), exp, (width, height, bw, bh, bs, maxv))

    # X: block_sum_max (dims count interleaved int quads)
    for t in range(130):
        bw = rng.choice([1, 2, 3, 5, 8, 9, 16])
        bh = rng.choice([1, 2, 3, 4, 7])
        bpitch = bw + rng.choice([0, 0, 1, 2])
        nQ = bpitch * bh * 4
        init_max = rng.choice([0, 0, -5, -100, 7, 10**9, 2**31 - 1])
        craft = t % 5
        if craft == 0:  # all-negative metrics -> max(init, 0 floor rule)
            ab = [0] * nQ
            sg = [-rng.randint(1, 1000) for _ in range(nQ)]
        elif craft == 1:  # single spike (pins max location incl edges)
            ab = [0] * nQ
            sg = [0] * nQ
            spike = rng.randrange(bw * bh * 4)
            ab[spike] = rng.randint(1, 10**6)
        elif craft == 2:  # zeros -> init rule
            ab = [0] * nQ
            sg = [0] * nQ
        else:
            ab = [rng.randint(0, 200000) for _ in range(nQ)]
            sg = [rng.randint(-100000, 100000) for _ in range(nQ)]
        got = run_mirror(["X", bw, bh, bpitch, init_max, nQ] + ab + sg)
        exp = golden_block_sum_max(bw, bh, bpitch, init_max, ab, sg)
        check("X", got, exp, (bw, bh, bpitch, init_max))

    # N: analyze_noise (8-bit only upstream)
    for t in range(150):
        width = rng.choice([1, 2, 3, 4, 5, 8, 13, 16, 31, 32, 33, 48, 64])
        height = rng.choice([1, 2, 3, 5, 8, 16, 24, 33])
        pitch = width + rng.choice([0, 0, 1, 2])
        init = [rng.choice([0, 0, 1, 999999999999]) for _ in range(4)]
        nP = pitch * height
        craft = t % 4
        if craft == 0:  # all-128 -> all four deltas 0 -> out == init
            s0 = [128] * nP
            s1 = [128] * nP
            s2 = [128] * nP
        elif craft == 1:  # ramps
            s0 = [(i * 3) % 256 for i in range(nP)]
            s1 = [(i * 5 + 1) % 256 for i in range(nP)]
            s2 = [(i * 7 + 2) % 256 for i in range(nP)]
        else:
            s0 = [rng.randint(0, 255) for _ in range(nP)]
            s1 = [rng.randint(0, 255) for _ in range(nP)]
            s2 = [rng.randint(0, 255) for _ in range(nP)]
        nums = ["N", width, height, pitch] + init + [nP] + s0 + s1 + s2
        exp = golden_analyze_noise(width, height, pitch, init, s0, s1, s2)
        check("N", run_mirror(nums), exp, (width, height, pitch))

    # D: analyze_diff (8-bit only upstream; +-2-row padded inputs)
    for t in range(160):
        width = rng.choice([1, 2, 4, 5, 8, 12, 16, 20, 32, 40, 48, 64])
        height = rng.choice([1, 2, 3, 4, 5, 6, 7, 8, 9, 15, 16, 24, 33])
        pitch = width + rng.choice([0, 0, 1, 2, 3])
        i0 = rng.choice([0, 0, 1, 2**40 + 3])
        i1 = rng.choice([0, 0, 2, 2**40 + 5])
        nP = pitch * (height + 4)
        craft = t % 5
        if craft == 0:  # equal constants -> both sums 0 -> out == init
            c = rng.randint(0, 255)
            f0 = [c] * nP
            f1 = [c] * nP
        elif craft == 1:  # interlaced rows (pins y&1 branches + combe)
            f0 = [0 if r % 2 == 0 else 255
                  for r in range(height + 4) for _ in range(pitch)]
            f1 = [255 if r % 2 == 0 else 0
                  for r in range(height + 4) for _ in range(pitch)]
        elif craft == 2:  # f1 = f0 (weave == plain taps on even/odd rows)
            f0 = [rng.randint(0, 255) for _ in range(nP)]
            f1 = list(f0)
        else:
            f0 = [rng.randint(0, 255) for _ in range(nP)]
            f1 = [rng.randint(0, 255) for _ in range(nP)]
        nums = ["D", width, height, pitch, i0, i1, nP] + f0 + f1
        exp = golden_analyze_diff(width, height, pitch, i0, i1, f0, f1)
        check("D", run_mirror(nums), exp, (width, height, pitch))

    print("kfm_decombeucf: %d/%d PASS" % (total if ok else -1, total))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
