#!/usr/bin/env python3
"""Validate the CombingAnalyze.cu kernels
(kf_copy_first, kf_combe_to_flag, kf_sum_box3x3, kf_binary_flag,
kf_bilinear_h, kf_bilinear_v, kf_temporal_soften, kf_remove_combe2,
kf_clean_super, kf_init_contains_durty_block/kf_contains_durty_block,
kf_super_analyze, kf_init_fmcount/kf_count_cmflags/kf_count_cmflags_2planes)
and the 8-tap helpers (kf_calc_combe8/kf_calc_diff8) in
src/opencl/kfm/kernels/kfm_combinganalyze.cl against the CPU mirror
sim/kfm_combinganalyze_ref.cpp with an independent Python golden.

All are integer-exact except kf_temporal_soften, which is float32 with no FMA
pattern (the mirror is compiled with -ffp-contract=off and the golden emulates
float32 per operation; output depends only on the byte sum t, so the t =
0..765 sweep is exhaustive).  sum_box3x3 / bilinear_h/v / remove_combe2 read a
1-px halo unguarded and are verified over ALL outputs with halo-padded inputs.
binary_flag is verified in-place (dst == srcY, as upstream).  The (1.0f/3.0f)
constant fold is a // RIG-VERIFY item (must fold to 0x3EAAAAAB).
kf_super_analyze transcribes the warp-reduce per cell serially (value-identical
for integer sums); unwritten border cells stay sentinel.  The FMCount census
verifies serially (== reduce+atomics for ints); the 2planes golden runs two
single-plane passes, proving fused == U+V composition.

Run:  python3 python/run_kfm_combinganalyze.py
"""
import os, random, struct, subprocess, tempfile, sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BIN = os.path.join(tempfile.gettempdir(), "kfm_combinganalyze_ref")


def run_mirror(nums):
    inf = os.path.join(tempfile.gettempdir(), "kca_in.txt")
    outf = os.path.join(tempfile.gettempdir(), "kca_out.txt")
    with open(inf, "w") as f:
        f.write(" ".join(map(str, nums)) + "\n")
    subprocess.run([BIN, inf, outf], check=True)
    with open(outf) as f:
        return [int(line) for line in f]


def f32(v):
    return struct.unpack('f', struct.pack('f', v))[0]


def absdiff(a, b):
    return abs(a - b)


def golden_copy_first(width, height, spitch, lanes, src):
    return [src[(xx + yy * spitch) * lanes]
            for yy in range(height) for xx in range(width)]


def golden_combe_to_flag(nBlkX, nBlkY, cpitch, combe):
    out = []
    for yy in range(nBlkY):
        for xx in range(nBlkX):
            s = (combe[(2 * xx) + (2 * yy) * cpitch] +
                 combe[(2 * xx + 1) + (2 * yy) * cpitch] +
                 combe[(2 * xx) + (2 * yy + 1) * cpitch] +
                 combe[(2 * xx + 1) + (2 * yy + 1) * cpitch])
            out.append((s + 2) >> 2)
    return out


def golden_box3x3(width, height, pitch, maxv, src):
    org = 1 + pitch
    out = []
    for yy in range(height):
        for xx in range(width):
            off = org + xx + yy * pitch
            s = (src[off - 1 - pitch] + src[off - pitch] +
                 src[off + 1 - pitch] + src[off - 1] + src[off] +
                 src[off + 1] + src[off - 1 + pitch] + src[off + pitch] +
                 src[off + 1 + pitch])
            out.append(min(s >> 2, maxv))
    return out


def golden_binary(nBlkX, nBlkY, pitch, thY, thC, srcY, srcC):
    d = list(srcY)  # in-place on a copy, as upstream (dst == srcY)
    for yy in range(nBlkY):
        for xx in range(nBlkX):
            off = xx + yy * pitch
            d[off] = 128 if (d[off] >= thY or srcC[off] >= thC) else 0
    return [d[xx + yy * pitch] for yy in range(nBlkY) for xx in range(nBlkX)]


def golden_bilinear_h(width, height, spitch, scale, shift, src):
    half = scale // 2
    org = 1
    out = []
    for yy in range(height):
        for xx in range(width):
            x0 = (xx - half) >> shift
            c0 = ((x0 + 1) << shift) - (xx - half)
            c1 = scale - c0
            s0 = src[org + x0 + yy * spitch]
            s1 = src[org + x0 + 1 + yy * spitch]
            out.append((s0 * c0 + s1 * c1 + half) >> shift)
    return out


def golden_bilinear_v(width, height, spitch, scale, shift, src):
    half = scale // 2
    org = spitch
    out = []
    for yy in range(height):
        for xx in range(width):
            y0 = (yy - half) >> shift
            c0 = ((y0 + 1) << shift) - (yy - half)
            c1 = scale - c0
            s0 = src[org + xx + y0 * spitch]
            s1 = src[org + xx + (y0 + 1) * spitch]
            out.append((s0 * c0 + s1 * c1 + half) >> shift)
    return out


def golden_soften_val(a, b, c):
    t = f32(f32(a + b) + c)
    return int(f32(t * f32(1.0 / 3.0))) & 0xFF


def golden_soften(width, height, pitch, s0, s1, s2):
    out = []
    for yy in range(height):
        for xx in range(width):
            off = xx + yy * pitch
            out.append(golden_soften_val(s0[off], s1[off], s2[off]))
    return out


def golden_remove_combe2(width, height, pitch, cpitch, thcombe, src, combe):
    org = pitch
    out = []
    for yy in range(height):
        for xx in range(width):
            off = org + xx + yy * pitch
            score = combe[(xx >> 2) + (yy >> 2) * cpitch]
            if score >= thcombe:
                v = (src[off - pitch] + 2 * src[off] +
                     src[off + pitch] + 2) >> 2
                out.append(v)
            else:
                out.append(src[off])
    return out


def golden_clean_super(width, height, pitch, thresh, px, py, cx, cy):
    ox, oy = [], []
    for yy in range(height):
        for xx in range(width):
            off = xx + yy * pitch
            oy.append(cy[off])
            ox.append(0 if (py[off] <= thresh and cy[off] <= thresh)
                      else cx[off])
    return ox + oy


def golden_durty(width, height, pitch, flagp):
    for yy in range(height):
        for xx in range(width):
            if flagp[xx + yy * pitch]:
                return [1]
    return [0]


def golden_combe8(L):
    diff8 = absdiff(L[0], L[7])
    diffT = (absdiff(L[0], L[1]) + absdiff(L[1], L[2]) +
             absdiff(L[2], L[3]) + absdiff(L[3], L[4]) +
             absdiff(L[4], L[5]) + absdiff(L[5], L[6]) +
             absdiff(L[6], L[7]) - diff8)
    diffE = (absdiff(L[0], L[2]) + absdiff(L[2], L[4]) +
             absdiff(L[4], L[6]) + absdiff(L[6], L[7]) - diff8)
    diffO = (absdiff(L[0], L[1]) + absdiff(L[1], L[3]) +
             absdiff(L[3], L[5]) + absdiff(L[5], L[7]) - diff8)
    return [diffT - diffE - diffO]


def golden_diff8(L):
    return [absdiff(L[0], L[1]) + absdiff(L[2], L[3]) +
            absdiff(L[4], L[5]) + absdiff(L[6], L[7])]


def clamp_u8(v):
    return 0 if v < 0 else (255 if v > 255 else v)


def combe8(L):
    diff8 = absdiff(L[0], L[7])
    diffT = (absdiff(L[0], L[1]) + absdiff(L[1], L[2]) +
             absdiff(L[2], L[3]) + absdiff(L[3], L[4]) +
             absdiff(L[4], L[5]) + absdiff(L[5], L[6]) +
             absdiff(L[6], L[7]) - diff8)
    diffE = (absdiff(L[0], L[2]) + absdiff(L[2], L[4]) +
             absdiff(L[4], L[6]) + absdiff(L[6], L[7]) - diff8)
    diffO = (absdiff(L[0], L[1]) + absdiff(L[1], L[3]) +
             absdiff(L[3], L[5]) + absdiff(L[5], L[7]) - diff8)
    return diffT - diffE - diffO


def golden_super(nBlkX, nBlkY, fpitch_f, fpitch, shift, parity, f0, f1):
    nFl = fpitch * 2 * nBlkY
    fx = [-1] * nFl   # col 0 / rows 0-1 stay sentinel (never written)
    fy = [-1] * nFl
    for by in range(nBlkY - 1):
        for bx in range(nBlkX - 1):
            x0, y0 = bx * 4, by * 4
            sums = [0, 0, 0, 0]
            for tx in range(8):
                x = x0 + tx
                r0 = [f0[x + (y0 + k) * fpitch_f] for k in range(8)]
                r1 = [f1[x + (y0 + k) * fpitch_f] for k in range(8)]
                c0 = combe8(r0)
                if parity:    # TFF: top = f0-combe, bottom = f1/f0-mixed
                    c2 = combe8([r1[0], r0[1], r1[2], r0[3],
                                 r1[4], r0[5], r1[6], r0[7]])
                    sums[0] += c0
                    sums[2] += c2
                else:         # BFF: mirrored
                    c0m = combe8([r0[0], r1[1], r0[2], r1[3],
                                  r0[4], r1[5], r0[6], r1[7]])
                    sums[2] += c0
                    sums[0] += c0m
                sums[1] += sum(absdiff(r0[r], r1[r]) for r in (0, 2, 4, 6))
                sums[3] += sum(absdiff(r0[r], r1[r]) for r in (1, 3, 5, 7))
            c = bx + 1
            r0w, r1w = 2 * (by + 1), 2 * (by + 1) + 1
            fx[c + r0w * fpitch] = clamp_u8(sums[0] >> shift)
            fy[c + r0w * fpitch] = clamp_u8(sums[1] >> shift)
            fx[c + r1w * fpitch] = clamp_u8(sums[2] >> shift)
            fy[c + r1w * fpitch] = clamp_u8(sums[3] >> shift)
    return fx + fy


def golden_count_single(width, height, pitch, parity, thM, thS, thLS,
                        init, c0x, c0y, c1x, c1y):
    dst = list(init)  # [m0,s0,l0,m1,s1,l1]
    np_ = 0 if parity else 1   # !parity
    for by in range(height):
        for bx in range(width):
            off = bx + by * pitch
            for i in (0, 1):
                vx = c0x[off] if i == 0 else c1x[off]
                vy = c0y[off] if i == 0 else c1y[off]
                slot = i ^ np_
                if vy >= thM:
                    dst[slot * 3 + 0] += 1
                if vx >= thS:
                    dst[slot * 3 + 1] += 1
                if vx >= thLS:
                    dst[slot * 3 + 2] += 1
    return dst


def main():
    subprocess.run(["g++", "-O2", "-std=gnu++17", "-Wl,--allow-multiple-definition", "-ffp-contract=off", "-w",
                    os.path.join(REPO, "sim", "kfm_combinganalyze_ref.cpp"),
                    "-o", BIN], check=True)
    rng = random.Random(41)
    ok = True
    total = 0

    def check(name, got, exp, info):
        nonlocal total, ok
        total += 1
        if got != exp:
            ok = False
            for i, (g, e) in enumerate(zip(got, exp)):
                if g != e:
                    print(name, "MISMATCH", info, "px", i, g, e)
                    break
            return False
        return True

    # F: copy_first (.x lane extract; lanes 2 = production uchar2, 4 = uchar4)
    for _ in range(120):
        lanes = rng.choice([2, 2, 4])
        width = rng.randint(1, 32)
        height = rng.randint(1, 16)
        spitch = width + rng.choice([0, 1, 3])
        dpitch = width + rng.choice([0, 2])
        nS = spitch * height * lanes
        src = [rng.randint(0, 255) for _ in range(nS)]
        hdr = [ord('F'), width, height, spitch, dpitch, lanes, nS]
        got = run_mirror(hdr + src)
        exp = golden_copy_first(width, height, spitch, lanes, src)
        if not check("copy_first", got, exp, (width, height, lanes)):
            if total >= 3: break

    # G: combe_to_flag (2x2 round-half-up quarter mean)
    for _ in range(120):
        nBlkX = rng.randint(1, 16)
        nBlkY = rng.randint(1, 16)
        cpitch = 2 * nBlkX + rng.choice([0, 1, 3])
        nC = cpitch * 2 * nBlkY
        if rng.random() < 0.25:   # constant combe c -> exactly c
            c = rng.randint(0, 255)
            combe = [c] * nC
        else:
            combe = [rng.randint(0, 255) for _ in range(nC)]
        hdr = [ord('G'), nBlkX, nBlkY, 0, cpitch, nC]
        got = run_mirror(hdr + combe)
        exp = golden_combe_to_flag(nBlkX, nBlkY, cpitch, combe)
        if not check("combe_to_flag", got, exp, (nBlkX, nBlkY)):
            if total >= 3: break

    # B: sum_box3x3 (quartered 3x3, min with maxv; halo-padded src)
    for _ in range(150):
        width = rng.randint(1, 20)
        height = rng.randint(1, 20)
        maxv = rng.choice([0, 1, 100, 127, 128, 255, 256, 572, 573, 574,
                           1000])
        pitch = width + 2 + rng.choice([0, 1, 3])
        nS = pitch * (height + 2)
        if rng.random() < 0.2:    # all-255 -> 573, pins the maxv boundary
            src = [255] * nS
        else:
            src = [rng.randint(0, 255) for _ in range(nS)]
        hdr = [ord('B'), width, height, pitch, maxv, nS]
        got = run_mirror(hdr + src)
        exp = golden_box3x3(width, height, pitch, maxv, src)
        if not check("sum_box3x3", got, exp, (width, height, maxv)):
            if total >= 3: break

    # N: binary_flag (in-place; (Y>=thY||C>=thC)?128:0)
    for _ in range(120):
        nBlkX = rng.randint(1, 24)
        nBlkY = rng.randint(1, 24)
        pitch = nBlkX + rng.choice([0, 1, 3])
        nP = pitch * nBlkY
        thY = rng.choice([-1, 0, 1, 127, 128, 255, 256])
        thC = rng.choice([-1, 0, 1, 127, 128, 255, 256])
        srcY = [rng.randint(0, 255) for _ in range(nP)]
        srcC = [rng.randint(0, 255) for _ in range(nP)]
        # pin the >= boundary: force == th values in
        for i in rng.sample(range(nP), min(nP, 6)):
            (srcY if i % 2 == 0 else srcC)[i] = max(
                0, min(255, (thY if i % 2 == 0 else thC)))
        hdr = [ord('N'), nBlkX, nBlkY, pitch, thY, thC, nP]
        got = run_mirror(hdr + srcY + srcC)
        exp = golden_binary(nBlkX, nBlkY, pitch, thY, thC, srcY, srcC)
        if not check("binary_flag", got, exp, (nBlkX, nBlkY, thY, thC)):
            if total >= 3: break

    # H: bilinear_h ((4,2)/(8,3); widths need not be multiples of scale)
    for _ in range(150):
        scale, shift = rng.choice([(4, 2), (8, 3)])
        half = scale // 2
        width = rng.randint(1, 40)
        height = rng.randint(1, 12)
        readcols = ((width - 1 - half) >> shift) + 2
        spitch = 1 + readcols + rng.choice([0, 1, 3])
        nS = spitch * height
        if rng.random() < 0.3:    # ramp: blend direction visible
            src = [(xx * 37) % 256 for _ in range(height)
                   for xx in range(spitch)]
        else:
            src = [rng.randint(0, 255) for _ in range(nS)]
        hdr = [ord('H'), width, height, spitch, scale, shift, nS]
        got = run_mirror(hdr + src)
        exp = golden_bilinear_h(width, height, spitch, scale, shift, src)
        if not check("bilinear_h", got, exp, (width, height, scale, shift)):
            if total >= 3: break

    # V: bilinear_v (transposed halo: rows)
    for _ in range(150):
        scale, shift = rng.choice([(4, 2), (8, 3)])
        half = scale // 2
        width = rng.randint(1, 24)
        height = rng.randint(1, 40)
        readrows = ((height - 1 - half) >> shift) + 2
        spitch = width + rng.choice([0, 1, 3])
        nS = spitch * (1 + readrows)
        if rng.random() < 0.3:
            src = [(yy * 37) % 256 for yy in range(1 + readrows)
                   for _ in range(spitch)]
        else:
            src = [rng.randint(0, 255) for _ in range(nS)]
        hdr = [ord('V'), width, height, spitch, scale, shift, nS]
        got = run_mirror(hdr + src)
        exp = golden_bilinear_v(width, height, spitch, scale, shift, src)
        if not check("bilinear_v", got, exp, (width, height, scale, shift)):
            if total >= 3: break

    # S: temporal_soften — EXHAUSTIVE t sweep (output depends only on t)
    for t in range(766):
        a = rng.randint(max(0, t - 510), min(255, t))
        b = rng.randint(max(0, t - a - 255), min(255, t - a))
        c = t - a - b
        hdr = [ord('S'), 1, 1, 1, 1]
        got = run_mirror(hdr + [a] + [b] + [c])
        exp = [golden_soften_val(a, b, c)]
        if not check("soften_t", got, exp, (t, a, b, c)):
            if total >= 3: break
    # S: random multi-pixel frames (incl. split-independence spot checks)
    for _ in range(60):
        width = rng.randint(1, 24)
        height = rng.randint(1, 16)
        pitch = width + rng.choice([0, 2])
        nP = pitch * height
        pool = [0, 0, 1, 127, 128, 254, 255, 255]
        s0 = [rng.choice(pool) if rng.random() < 0.5
              else rng.randint(0, 255) for _ in range(nP)]
        s1 = [rng.choice(pool) if rng.random() < 0.5
              else rng.randint(0, 255) for _ in range(nP)]
        s2 = [rng.choice(pool) if rng.random() < 0.5
              else rng.randint(0, 255) for _ in range(nP)]
        hdr = [ord('S'), width, height, pitch, nP]
        got = run_mirror(hdr + s0 + s1 + s2)
        exp = golden_soften(width, height, pitch, s0, s1, s2)
        if not check("soften", got, exp, (width, height)):
            if total >= 3: break

    # R: remove_combe2 (4x4 combe gate + vertical binomial; padded src)
    for _ in range(160):
        bits = rng.choice([8, 8, 16])
        maxv = 255 if bits == 8 else 65535
        width = rng.randint(1, 32)
        height = rng.randint(1, 20)
        pitch = width + rng.choice([0, 2])
        nS = pitch * (height + 2)
        cw = ((width - 1) >> 2) + 1
        ch = ((height - 1) >> 2) + 1
        cpitch = cw + rng.choice([0, 1])
        nC = cpitch * ch
        thcombe = rng.choice([-1, 0, 1, 127, 128, 255, 256])
        if rng.random() < 0.3:    # row gradient: filter vs passthrough differ
            src = [(yy * 521) % (maxv + 1) for yy in range(height + 2)
                   for _ in range(pitch)]
        else:
            src = [rng.randint(0, maxv) for _ in range(nS)]
        combe = [rng.randint(0, 255) for _ in range(nC)]
        for i in rng.sample(range(nC), min(nC, 4)):  # pin score == th
            combe[i] = max(0, min(255, thcombe))
        hdr = [ord('R'), width, height, pitch, cpitch, thcombe, nS, nC]
        got = run_mirror(hdr + src + combe)
        exp = golden_remove_combe2(width, height, pitch, cpitch, thcombe,
                                   src, combe)
        if not check("remove_combe2", got, exp,
                     (width, height, bits, thcombe)):
            if total >= 3: break

    # C: clean_super (zero .x where prev.y<=th && cur.y<=th)
    for _ in range(130):
        width = rng.randint(1, 24)
        height = rng.randint(1, 24)
        pitch = width + rng.choice([0, 1, 3])
        nP = pitch * height
        thresh = rng.choice([-1, 0, 1, 10, 127, 128, 255, 256])
        px = [rng.randint(0, 255) for _ in range(nP)]
        py = [rng.randint(0, 255) for _ in range(nP)]
        cx = [rng.randint(0, 255) for _ in range(nP)]
        cy = [rng.randint(0, 255) for _ in range(nP)]
        for i in rng.sample(range(nP), min(nP, 6)):  # pin == th
            v = max(0, min(255, thresh))
            py[i] = v
            cy[i] = v
        hdr = [ord('C'), width, height, pitch, thresh, nP]
        got = run_mirror(hdr + px + py + cx + cy)
        exp = golden_clean_super(width, height, pitch, thresh, px, py, cx,
                                 cy)
        if not check("clean_super", got, exp, (width, height, thresh)):
            if total >= 3: break

    # D: durty_block (OR-reduction to one int)
    for case in range(80):
        width = rng.randint(1, 32)
        height = rng.randint(1, 32)
        pitch = width + rng.choice([0, 1, 3])
        nP = pitch * height
        if case % 4 == 0:
            flagp = [0] * nP                       # -> 0
        elif case % 4 == 1:                        # single hit -> 1
            flagp = [0] * nP
            flagp[rng.randrange(nP)] = rng.choice([1, 128, 255])
        else:
            flagp = [rng.randint(0, 255) for _ in range(nP)]
        hdr = [ord('D'), width, height, pitch, nP]
        got = run_mirror(hdr + flagp)
        exp = golden_durty(width, height, pitch, flagp)
        if not check("durty_block", got, exp, (width, height)):
            if total >= 3: break

    # M: calc_combe8 taps (8/16-bit; sign flips pinned by crafts)
    crafts8 = [
        [0] * 8,
        [255] * 8,
        [0, 255, 0, 255, 0, 255, 0, 255],   # alternating -> large +
        [0, 0, 255, 255, 0, 0, 0, 0],       # -> negative
        [0, 1, 2, 3, 4, 5, 6, 7],           # ramp -> 0
        [7, 6, 5, 4, 3, 2, 1, 0],
    ]
    for case in range(150):
        bits = rng.choice([8, 8, 16])
        maxv = 255 if bits == 8 else 65535
        if case < len(crafts8) and bits == 8:
            L = list(crafts8[case])
        elif rng.random() < 0.3:
            v = rng.randint(0, maxv)
            L = [v] * 8                            # constant -> 0
        else:
            L = [rng.randint(0, maxv) for _ in range(8)]
        got = run_mirror([ord('M')] + L)
        exp = golden_combe8(L)
        if not check("calc_combe8", got, exp, (bits, L)):
            if total >= 3: break

    # A: calc_diff8 taps
    for _ in range(120):
        bits = rng.choice([8, 8, 16])
        maxv = 255 if bits == 8 else 65535
        if rng.random() < 0.2:
            L = [0, maxv] * 4                      # -> 4*maxv
        else:
            L = [rng.randint(0, maxv) for _ in range(8)]
        got = run_mirror([ord('A')] + L)
        exp = golden_diff8(L)
        if not check("calc_diff8", got, exp, (bits,)):
            if total >= 3: break

    # J: super_analyze (serial-cell == warp-reduce; border cells unwritten)
    for _ in range(170):
        bits = rng.choice([8, 8, 16])
        maxv = 255 if bits == 8 else 65535
        nBlkX = rng.randint(1, 8)   # 1 -> empty grid, all sentinel
        nBlkY = rng.randint(1, 8)
        fpitch_f = 4 * nBlkX + rng.choice([0, 2])
        fpitch = nBlkX + rng.choice([0, 1, 3])
        shift = rng.choice([0, 1, 4, 8, 12, 16])
        parity = rng.choice([0, 1])
        nF = fpitch_f * 4 * nBlkY
        nFl = fpitch * 2 * nBlkY
        pick = rng.random()
        if pick < 0.45:
            f0 = [rng.randint(0, maxv) for _ in range(nF)]
            f1 = [rng.randint(0, maxv) for _ in range(nF)]
        elif pick < 0.65:           # interlaced: combe path fires hard
            f0 = [(maxv if r % 2 else 0)
                  for r in range(4 * nBlkY) for _ in range(fpitch_f)]
            f1 = [(0 if r % 2 else maxv)
                  for r in range(4 * nBlkY) for _ in range(fpitch_f)]
        elif pick < 0.8:            # split frames: diff top-clamp pins
            f0 = [0] * nF
            f1 = [maxv] * nF
        else:                       # constant: all sums 0 -> flags 0
            v = rng.randint(0, maxv)
            f0 = [v] * nF
            f1 = [v] * nF
        hdr = [ord('J'), nBlkX, nBlkY, fpitch_f, fpitch, shift, parity,
               nF, nFl]
        got = run_mirror(hdr + f0 + f1)
        exp = golden_super(nBlkX, nBlkY, fpitch_f, fpitch, shift, parity,
                           f0, f1)
        if not check("super_analyze", got, exp,
                     (nBlkX, nBlkY, bits, shift, parity)):
            if total >= 3: break

    # W: count_cmflags (serial golden == reduce+atomics; nonzero init piles)
    for _ in range(150):
        width = rng.randint(1, 40)    # need not tile 32x16: guards cover
        height = rng.randint(1, 40)
        pitch = width + rng.choice([0, 1, 3])
        nP = pitch * height
        parity = rng.choice([0, 1])
        thM = rng.choice([-1, 0, 1, 127, 128, 255, 256])
        thS = rng.choice([-1, 0, 1, 127, 128, 255, 256])
        thLS = rng.choice([-1, 0, 1, 127, 128, 255, 256])
        if rng.random() < 0.5:
            init = [0] * 6
        else:
            init = [rng.randint(0, 100000) for _ in range(6)]
        c0x = [rng.randint(0, 255) for _ in range(nP)]
        c0y = [rng.randint(0, 255) for _ in range(nP)]
        c1x = [rng.randint(0, 255) for _ in range(nP)]
        c1y = [rng.randint(0, 255) for _ in range(nP)]
        for i in rng.sample(range(nP), min(nP, 4)):  # pin == th (fires)
            c0y[i] = max(0, min(255, thM))
            c1x[i] = max(0, min(255, thS))
        hdr = [ord('W'), width, height, pitch, parity, thM, thS, thLS] + \
            init + [nP]
        got = run_mirror(hdr + c0x + c0y + c1x + c1y)
        exp = golden_count_single(width, height, pitch, parity, thM, thS,
                                  thLS, init, c0x, c0y, c1x, c1y)
        if not check("count_cmflags", got, exp,
                     (width, height, parity, thM, thS, thLS)):
            if total >= 3: break

    # Q: count_cmflags_2planes (fused mirror vs two-singles golden:
    # proves fused == U+V composition)
    for _ in range(130):
        width = rng.randint(1, 24)
        height = rng.randint(1, 24)
        pitch = width + rng.choice([0, 1, 3])
        nP = pitch * height
        parity = rng.choice([0, 1])
        thM = rng.choice([-1, 0, 1, 127, 128, 255, 256])
        thS = rng.choice([-1, 0, 1, 127, 128, 255, 256])
        thLS = rng.choice([-1, 0, 1, 127, 128, 255, 256])
        if rng.random() < 0.5:
            init = [0] * 6
        else:
            init = [rng.randint(0, 100000) for _ in range(6)]
        planes = [[rng.randint(0, 255) for _ in range(nP)]
                  for _ in range(8)]
        hdr = [ord('Q'), width, height, pitch, parity, thM, thS, thLS] + \
            init + [nP]
        got = run_mirror(hdr + [v for pl in planes for v in pl])
        mid = golden_count_single(width, height, pitch, parity, thM, thS,
                                  thLS, init, *planes[0:4])
        exp = golden_count_single(width, height, pitch, parity, thM, thS,
                                  thLS, mid, *planes[4:8])
        if not check("count_2planes", got, exp,
                     (width, height, parity, thM, thS, thLS)):
            if total >= 3: break

    # I: init_fmcount (zero-fill)
    for _ in range(5):
        got = run_mirror([ord('I')])
        if not check("init_fmcount", got, [0, 0, 0, 0, 0, 0], ()):
            if total >= 3: break

    print(f"KFM CombingAnalyze (copy_first/combe_to_flag/sum_box3x3/"
          f"binary_flag/bilinear_h/bilinear_v/soften/remove_combe2/"
          f"clean_super/durty_block/combe8/diff8/super_analyze/"
          f"count_cmflags/count_2planes/init_fmcount): "
          f"{'PASS' if ok else 'FAIL'} ({total} cases)")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
