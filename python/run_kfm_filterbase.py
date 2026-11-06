#!/usr/bin/env python3
"""Validate the sixteen KFM KFMFilterBase.cu kernels
(kf_calc_combe, kf_merge_uvcoefs, kf_extend_coef2, kf_apply_uvcoefs_420,
kf_padv, kf_padh, kf_merge_block, kf_average, kf_max, kf_merge_uvflags,
kf_copy_border, kf_analyze_frame, kf_copy_pad, kf_copy_pad_2plane,
kf_max_extend_blocks_h, kf_max_extend_blocks_v) in
src/opencl/kfm/kernels/kfm_filterbase.cl against the CPU mirror
sim/kfm_filterbase_ref.cpp with an independent Python golden.

All sixteen are integer-exact.  kf_calc_combe is verified over its interior rows
(y in [2,height-3]); its border rows read a host-padded plane (VPAD) and are
RIG-VERIFY (mirror/golden put a -1 sentinel there).  kf_extend_coef2 is the CUDA
kl_extend_coef2 device kernel (upstream's CPU fallback differs at rows 0 and
height-1; the OpenCL target is the device kernel).  kf_padv/kf_padh are the
in-place mirror pads (verified solo plus the composed padv-then-padh 2D pad in
upstream Deblock order).  kf_merge_block is the MergeBlock masked blender
(flag is uchar at both bit depths; full 0..255 flag sweep pins the
negative-invcombe wrap path too).  kf_average is the floor mean (odd sums pin
the floor path).  kf_max is scalar uint8 upstream, so arbitrary widths are
exact parity.  kf_merge_uvflags folds UV flags into Y with a mod-256 |= wrap
(full uchar sweep pins the wrap).  kf_copy_border copies the extreme rows
straight through (dst pre-filled to differ everywhere, so strays show; the
vborder*2 > height lane overlap is covered).  kf_analyze_frame classifies
SHIMA/LSHIMA/MOVE flags from unshifted CalcCombe + |mref-base| taps over
padded sources, verified over ALL rows with the strict-> threshold boundaries
pinned by crafted fields (t/diff = 0, 1, maxv, 6*maxv).  kf_copy_pad /
kf_copy_pad_2plane are the padded-frame copies (no upstream CPU twin; the
mirror runs the VECTOR algorithm with the lane swap while the golden is the
plain pixel mirror, so their agreement proves the swap cancels — plus a
copy->padv->padh composition cross-check per case).  kf_max_extend_blocks_h/v
are the ExtendBlocks ping-pong passes (verified solo, incl. the nBlk == 1
branch-order pins, plus composed h->v against the in-place 3-pass CPU
algorithm — different code paths, same result).

Run:  python3 python/run_kfm_filterbase.py
"""
import os, random, struct, subprocess, tempfile, sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BIN = os.path.join(tempfile.gettempdir(), "kfm_filterbase_ref")


def run_mirror(nums):
    inf = os.path.join(tempfile.gettempdir(), "kffb_in.txt")
    outf = os.path.join(tempfile.gettempdir(), "kffb_out.txt")
    with open(inf, "w") as f:
        f.write(" ".join(map(str, nums)))
    subprocess.run([BIN, inf, outf], check=True)
    with open(outf) as f:
        return [int(line) for line in f]


def calc_combe_val(a, b, c, d, e):
    v = a + c * 4 + e - (b + d) * 3
    return -v if v < 0 else v


def golden_calc_combe(width, height, pitch, src):
    out = []
    for yy in range(height):
        for xx in range(width):
            if 2 <= yy <= height - 3:
                off = xx + yy * pitch
                c = calc_combe_val(src[off - 2 * pitch], src[off - pitch],
                                   src[off], src[off + pitch],
                                   src[off + 2 * pitch]) >> 2
                out.append(max(0, min(255, c)))
            else:
                out.append(-1)  # border sentinel
    return out


def golden_merge(width, height, pitchY, pitchUV, lx, ly, fY, fU, fV):
    # returns logical width*height of the folded Y plane
    fY = list(fY)
    for yy in range(height):
        for xx in range(width):
            oY = xx + yy * pitchY
            oUV = (xx >> lx) + (yy >> ly) * pitchUV
            u, v = fU[oUV], fV[oUV]
            fY[oY] = max(fY[oY], max(u, v))
    return [fY[xx + yy * pitchY] for yy in range(height) for xx in range(width)]


def golden_extend(width, height, pitch, src):
    out = []
    for yy in range(height):
        for xx in range(width):
            y0 = max(yy - 1, 0)
            y2 = min(yy + 1, height - 1)
            m = max(src[xx + y0 * pitch], src[xx + yy * pitch],
                    src[xx + y2 * pitch])
            out.append(m)
    return out


def golden_apply420(widthUV, heightUV, pitchY, pitchUV, fY):
    uout, vout = [], []
    for yy in range(heightUV):
        for xx in range(widthUV):
            oY0 = (2 * xx) + (2 * yy) * pitchY
            v = (fY[oY0] + fY[oY0 + 1] + fY[oY0 + pitchY] +
                 fY[oY0 + pitchY + 1])
            avg = (v + 2) >> 2
            uout.append(avg)
            vout.append(avg)
    return uout + vout


def golden_padv(buf, width, height, pitch, vpad, org):
    # out-of-place w.r.t. the pristine input: reads never observe writes,
    # which holds iff the pad is race-free (reads interior, writes pad rows)
    b = list(buf)
    for yy in range(vpad):
        top_dst = org + (-yy - 1) * pitch
        top_src = org + yy * pitch
        bot_dst = org + (height + yy) * pitch
        bot_src = org + (height - yy - 1) * pitch
        for xx in range(width):
            b[top_dst + xx] = buf[top_src + xx]
            b[bot_dst + xx] = buf[bot_src + xx]
    return b


def golden_padh(buf, width, height, pitch, hpad, org):
    b = list(buf)
    for yy in range(height):
        row = org + yy * pitch
        for xx in range(hpad):
            b[row + (-xx - 1)] = buf[row + xx]
            b[row + (width + xx)] = buf[row + (width - xx - 1)]
    return b


def golden_merge_block(width, height, pitch, fpitch, bits, s24, s60, flag):
    mask = 0xFF if bits == 8 else 0xFFFF
    out = []
    for yy in range(height):
        for xx in range(width):
            combe = flag[xx + yy * fpitch]
            t = (combe * s60[xx + yy * pitch] +
                 (128 - combe) * s24[xx + yy * pitch] + 64) >> 7
            out.append(t & mask)  # the (PX) cast wrap
    return out


def golden_average(width, height, pitch, s0, s1):
    out = []
    for yy in range(height):
        for xx in range(width):
            off = xx + yy * pitch
            out.append((s0[off] + s1[off]) // 2)
    return out


def golden_max(width, height, pitch, s0, s1):
    out = []
    for yy in range(height):
        for xx in range(width):
            off = xx + yy * pitch
            a, b = s0[off], s1[off]
            out.append(a if a >= b else b)
    return out


def golden_merge_uvflags(width, height, pitchY, pitchUV, lx, ly, fY, fU, fV):
    fY = list(fY)
    for yy in range(height):
        for xx in range(width):
            oY = xx + yy * pitchY
            oUV = (xx >> lx) + (yy >> ly) * pitchUV
            fY[oY] = (fY[oY] | ((fU[oUV] | fV[oUV]) * 16)) % 256
    return [fY[xx + yy * pitchY] for yy in range(height)
            for xx in range(width)]


def golden_copy_border(src, dst, width, height, pitch, vborder):
    d = list(dst)
    for yy in range(vborder):
        top_src = yy * pitch
        bot_src = (height - yy - 1) * pitch
        for xx in range(width):
            d[top_src + xx] = src[top_src + xx]
            d[bot_src + xx] = src[bot_src + xx]
    return d


def golden_analyze_frame(width, height, pitch, dpitch, tM, tS, tLS,
                         base, sref, mref):
    out = [-1] * (dpitch * height)  # strided output; gaps stay sentinel
    org = pitch  # interior origin: 1 pad row above
    for yy in range(height):
        for xx in range(width):
            a = base[org + xx + (yy - 1) * pitch]
            b = sref[org + xx + yy * pitch]
            c = base[org + xx + yy * pitch]
            d = sref[org + xx + (yy + 1) * pitch]
            e = base[org + xx + (yy + 1) * pitch]
            t = calc_combe_val(a, b, c, d, e)  # unshifted: no >>2 here
            diff = abs(mref[org + xx + yy * pitch] - c)
            flag = 0
            if t > tS:
                flag |= 2   # SHIMA
            if t > tLS:
                flag |= 4   # LSHIMA
            if diff > tM:
                flag |= 1   # MOVE
            out[xx + yy * dpitch] = flag
    return out


def golden_copy_pad(src, width, height, srcpitch, dstpitch, hpad, vpad):
    # plain per-pixel mirror (scalar form); the mirror runs the VECTOR form
    # with the padx lane swap, so mirror-vs-golden agreement proves the swap
    # cancels (see kf_copy_pad's proof comment)
    nD = dstpitch * (height + 2 * vpad)
    out = [-1] * nD
    org = hpad + vpad * dstpitch  # dst interior origin
    for yy in range(-vpad, height + vpad):
        if yy < 0:
            sy = -yy - 1
        elif yy >= height:
            sy = height - (yy - height) - 1
        else:
            sy = yy
        for xx in range(-hpad, width + hpad):
            if xx < 0:
                sx = -xx - 1
            elif xx >= width:
                sx = width - (xx - width) - 1
            else:
                sx = xx
            out[org + xx + yy * dstpitch] = src[sx + sy * srcpitch]
    return out


def composed_copy_pad(src, width, height, srcpitch, dstpitch, hpad, vpad):
    # independent cross-check path: copy the interior, then padv, then padh
    # (upstream Deblock order) — must equal golden_copy_pad every case
    nD = dstpitch * (height + 2 * vpad)
    org = hpad + vpad * dstpitch
    buf = [-1] * nD
    for yy in range(height):
        for xx in range(width):
            buf[org + xx + yy * dstpitch] = src[xx + yy * srcpitch]
    step1 = golden_padv(buf, width, height, dstpitch, vpad, org)
    return golden_padh(step1, width, height + 2 * vpad, dstpitch, hpad,
                       org - vpad * dstpitch)


def golden_extend_h(src, nBlkX, nBlkY, pitch):
    out = []
    for by in range(nBlkY):
        for bx in range(nBlkX):
            off = bx + by * pitch
            if bx == nBlkX - 1:   # no right neighbour: self copy
                out.append(src[off])
            elif bx == 0:         # col 0 takes the neighbour outright
                out.append(src[off + 1])
            else:
                out.append(max(src[off], src[off + 1]))
    return out


def golden_extend_v(src, nBlkX, nBlkY, pitch):
    out = []
    for by in range(nBlkY):
        for bx in range(nBlkX):
            off = bx + by * pitch
            if by == nBlkY - 1:
                out.append(src[off])
            elif by == 0:
                out.append(src[off + pitch])
            else:
                out.append(max(src[off], src[off + pitch]))
    return out


def golden_extend_cpu(src, nBlkX, nBlkY, pitch):
    # the upstream IN-PLACE 3-pass algorithm (independent path from the h->v
    # ping-pong the mirror runs): pass 1 spreads right over rows by >= 1
    # (col 0 takes the neighbour outright), pass 2 copies row 1 into row 0,
    # pass 3 spreads down over rows by in [1, nBlkY-2]
    d = list(src)
    for by in range(1, nBlkY):
        d[0 + by * pitch] = d[1 + by * pitch]
        for bx in range(1, nBlkX - 1):
            d[bx + by * pitch] = max(d[bx + by * pitch],
                                     d[bx + 1 + by * pitch])
    for bx in range(nBlkX):
        d[bx] = d[bx + pitch]
    for by in range(1, nBlkY - 1):
        for bx in range(nBlkX):
            d[bx + by * pitch] = max(d[bx + by * pitch],
                                     d[bx + (by + 1) * pitch])
    return [d[xx + yy * pitch] for yy in range(nBlkY) for xx in range(nBlkX)]


def main():
    subprocess.run(["g++", "-O2", "-std=gnu++17", "-Wl,--allow-multiple-definition", "-ffp-contract=off", "-w",
                    os.path.join(REPO, "sim", "kfm_filterbase_ref.cpp"),
                    "-o", BIN], check=True)
    rng = random.Random(19)
    ok = True
    total = 0

    # K: calc_combe (interior only)
    for _ in range(180):
        bits = rng.choice([8, 8, 16])
        maxv = 255 if bits == 8 else 65535
        width = rng.randint(1, 20) * 4
        height = rng.randint(7, 18)          # >= 5 so interior exists
        pitch = width + rng.choice([0, 2])
        n = pitch * height
        # interlaced-ish content to exercise combing (row pitch layout)
        src = [0] * n
        for yy in range(height):
            for xx in range(width):
                field = (yy % 2)
                base = rng.randint(0, maxv)
                v = base + (rng.randint(0, maxv // 6) if field else 0)
                src[yy * pitch + xx] = max(0, min(maxv, v))
        hdr = [ord('K'), width, height, pitch, n]
        got = run_mirror(hdr + src)
        exp = golden_calc_combe(width, height, pitch, src)
        total += 1
        # compare interior rows only
        interior_ok = True
        for i, (g, e) in enumerate(zip(got, exp)):
            if e != -1 and g != e:
                interior_ok = False
                if ok:
                    print("calc_combe MISMATCH", width, height, "px", i, g, e)
        if not interior_ok:
            ok = False
            if total >= 3: break

    # M: merge_uvcoefs (YV12 lx=ly=1)
    for _ in range(160):
        bits = rng.choice([8, 8, 16])
        maxv = 255 if bits == 8 else 65535
        width = rng.randint(1, 20) * 4
        height = rng.randint(1, 6) * 2        # even (YV12)
        lx, ly = 1, 1
        pitchY = width + rng.choice([0, 2])
        uvwidth = width >> lx
        uvheight = height >> ly
        pitchUV = uvwidth + rng.choice([0, 2])
        nY = pitchY * height
        nU = pitchUV * uvheight
        fY = [rng.randint(0, 128) for _ in range(nY)]
        fU = [rng.randint(0, 128) for _ in range(nU)]
        fV = [rng.randint(0, 128) for _ in range(nU)]
        hdr = [ord('M'), width, height, pitchY, pitchUV, lx, ly, nY, nU]
        got = run_mirror(hdr + fY + fU + fV)
        exp = golden_merge(width, height, pitchY, pitchUV, lx, ly, fY, fU, fV)
        total += 1
        if got != exp:
            ok = False
            for i, (g, e) in enumerate(zip(got, exp)):
                if g != e:
                    print("merge_uvcoefs MISMATCH", width, height, "px", i, g, e)
                    break
            if total >= 3: break

    # E: extend_coef2
    for _ in range(180):
        bits = rng.choice([8, 8, 16])
        maxv = 255 if bits == 8 else 65535
        width = rng.randint(1, 20) * 4
        height = rng.randint(1, 14)
        pitch = width + rng.choice([0, 2])
        n = pitch * height
        src = [rng.randint(0, 128) for _ in range(n)]
        hdr = [ord('E'), width, height, pitch, n]
        got = run_mirror(hdr + src)
        exp = golden_extend(width, height, pitch, src)
        total += 1
        if got != exp:
            ok = False
            for i, (g, e) in enumerate(zip(got, exp)):
                if g != e:
                    print("extend_coef2 MISMATCH", width, height, "px", i, g, e)
                    break
            if total >= 3: break

    # A: apply_uvcoefs_420
    for _ in range(160):
        bits = rng.choice([8, 8, 16])
        maxv = 255 if bits == 8 else 65535
        uvwidth = rng.randint(1, 12) * 4
        uvheight = rng.randint(1, 10)
        width = uvwidth * 2
        pitchY = width + rng.choice([0, 2])
        pitchUV = uvwidth + rng.choice([0, 2])
        nY = pitchY * (uvheight * 2)
        nUV = pitchUV * uvheight
        fY = [rng.randint(0, 128) for _ in range(nY)]
        fU = [rng.randint(0, 128) for _ in range(nUV)]
        fV = [rng.randint(0, 128) for _ in range(nUV)]
        hdr = [ord('A'), uvwidth, uvheight, pitchY, pitchUV, nY, nUV]
        got = run_mirror(hdr + fY + fU + fV)
        exp = golden_apply420(uvwidth, uvheight, pitchY, pitchUV, fY)
        total += 1
        if got != exp:
            ok = False
            for i, (g, e) in enumerate(zip(got, exp)):
                if g != e:
                    print("apply_uvcoefs_420 MISMATCH", uvwidth, uvheight,
                          "px", i, g, e)
                    break
            if total >= 3: break

    # V: padv (in-place vertical mirror pad)
    for _ in range(150):
        bits = rng.choice([8, 8, 16])
        maxv = 255 if bits == 8 else 65535
        width = rng.randint(1, 40)
        height = rng.randint(2, 24)
        vpad = min(rng.choice([1, 2, 4, 8]), height)
        pitch = width + rng.choice([0, 1, 3])
        n = pitch * (height + 2 * vpad)
        buf = [rng.randint(0, maxv) for _ in range(n)]
        org = vpad * pitch
        hdr = [ord('V'), width, height, pitch, vpad, n]
        got = run_mirror(hdr + buf)
        exp = golden_padv(buf, width, height, pitch, vpad, org)
        total += 1
        if got != exp:
            ok = False
            for i, (g, e) in enumerate(zip(got, exp)):
                if g != e:
                    print("padv MISMATCH", width, height, pitch, vpad,
                          "px", i, g, e)
                    break
            if total >= 3: break

    # H: padh (in-place horizontal mirror pad)
    for _ in range(150):
        bits = rng.choice([8, 8, 16])
        maxv = 255 if bits == 8 else 65535
        width = rng.randint(2, 40)
        height = rng.randint(1, 24)
        hpad = min(rng.choice([1, 2, 4, 8]), width)
        pitch = width + 2 * hpad + rng.choice([0, 1, 3])
        n = pitch * height
        buf = [rng.randint(0, maxv) for _ in range(n)]
        org = hpad
        hdr = [ord('H'), width, height, pitch, hpad, n]
        got = run_mirror(hdr + buf)
        exp = golden_padh(buf, width, height, pitch, hpad, org)
        total += 1
        if got != exp:
            ok = False
            for i, (g, e) in enumerate(zip(got, exp)):
                if g != e:
                    print("padh MISMATCH", width, height, pitch, hpad,
                          "px", i, g, e)
                    break
            if total >= 3: break

    # B: padv then padh (composed 2D pad, upstream Deblock order)
    for _ in range(150):
        bits = rng.choice([8, 8, 16])
        maxv = 255 if bits == 8 else 65535
        width = rng.randint(2, 32)
        height = rng.randint(2, 20)
        vpad = min(rng.choice([1, 2, 4, 8]), height)
        hpad = min(rng.choice([1, 2, 4, 8]), width)
        pitch = width + 2 * hpad + rng.choice([0, 1, 3])
        n = pitch * (height + 2 * vpad)
        buf = [rng.randint(0, maxv) for _ in range(n)]
        org = hpad + vpad * pitch
        hdr = [ord('B'), width, height, pitch, vpad, hpad, n]
        got = run_mirror(hdr + buf)
        step1 = golden_padv(buf, width, height, pitch, vpad, org)
        exp = golden_padh(step1, width, height + 2 * vpad, pitch, hpad,
                          org - vpad * pitch)
        total += 1
        if got != exp:
            ok = False
            for i, (g, e) in enumerate(zip(got, exp)):
                if g != e:
                    print("pad-both MISMATCH", width, height, pitch,
                          vpad, hpad, "px", i, g, e)
                    break
            if total >= 3: break

    # G: merge_block (MergeBlock masked blender; flag uchar at both depths)
    for _ in range(200):
        bits = rng.choice([8, 8, 16])
        maxv = 255 if bits == 8 else 65535
        width = rng.randint(1, 16) * 4      # mult of 4: exact CUDA coverage
        height = rng.randint(1, 24)
        pitch = width + rng.choice([0, 2])
        fpitch = width + rng.choice([0, 2])
        nP = pitch * height
        nF = fpitch * height
        s24 = [rng.randint(0, maxv) for _ in range(nP)]
        s60 = [rng.randint(0, maxv) for _ in range(nP)]
        # production domain [0,128] + full uchar sweep (wrap path)
        if rng.random() < 0.5:
            flag = [rng.randint(0, 128) for _ in range(nF)]
        else:
            flag = [rng.randint(0, 255) for _ in range(nF)]
        hdr = [ord('G'), width, height, pitch, fpitch, bits, nP, nF]
        got = run_mirror(hdr + s24 + s60 + flag)
        exp = golden_merge_block(width, height, pitch, fpitch, bits,
                                 s24, s60, flag)
        total += 1
        if got != exp:
            ok = False
            for i, (g, e) in enumerate(zip(got, exp)):
                if g != e:
                    print("merge_block MISMATCH", width, height, bits,
                          "px", i, g, e)
                    break
            if total >= 3: break

    # R: average (floor mean; mult-of-4 widths = exact CUDA coverage)
    for _ in range(140):
        bits = rng.choice([8, 8, 16])
        maxv = 255 if bits == 8 else 65535
        width = rng.randint(1, 16) * 4
        height = rng.randint(1, 20)
        pitch = width + rng.choice([0, 2])
        n = pitch * height
        # extremes + odd sums (floor path) + random
        pool = [0, 0, 1, maxv - 1, maxv, maxv]
        s0 = [rng.choice(pool) if rng.random() < 0.4
              else rng.randint(0, maxv) for _ in range(n)]
        s1 = [rng.choice(pool) if rng.random() < 0.4
              else rng.randint(0, maxv) for _ in range(n)]
        hdr = [ord('R'), width, height, pitch, n]
        got = run_mirror(hdr + s0 + s1)
        exp = golden_average(width, height, pitch, s0, s1)
        total += 1
        if got != exp:
            ok = False
            for i, (g, e) in enumerate(zip(got, exp)):
                if g != e:
                    print("average MISMATCH", width, height, bits,
                          "px", i, g, e)
                    break
            if total >= 3: break

    # X: max (scalar uint8 twin: arbitrary widths, ties forced)
    for _ in range(120):
        width = rng.randint(1, 40)
        height = rng.randint(1, 20)
        pitch = width + rng.choice([0, 1, 3])
        n = pitch * height
        pool = [0, 1, 127, 128, 254, 255]
        s0 = [rng.choice(pool) if rng.random() < 0.4
              else rng.randint(0, 255) for _ in range(n)]
        s1 = [rng.choice(pool) if rng.random() < 0.4
              else rng.randint(0, 255) for _ in range(n)]
        for i in rng.sample(range(n), min(n, 5)):  # ties
            s1[i] = s0[i]
        hdr = [ord('X'), width, height, pitch, n]
        got = run_mirror(hdr + s0 + s1)
        exp = golden_max(width, height, pitch, s0, s1)
        total += 1
        if got != exp:
            ok = False
            for i, (g, e) in enumerate(zip(got, exp)):
                if g != e:
                    print("max MISMATCH", width, height, "px", i, g, e)
                    break
            if total >= 3: break

    # F: merge_uvflags (in-place fold, mod-256 |= wrap; 444/422/420-style lx/ly)
    for _ in range(130):
        lx = rng.choice([0, 1])
        ly = rng.choice([0, 1])
        width = rng.randint(1, 24)
        height = rng.randint(1, 16)
        pitchY = width + rng.choice([0, 2])
        uvw = ((width - 1) >> lx) + 1
        uvh = ((height - 1) >> ly) + 1
        pitchUV = uvw + rng.choice([0, 2])
        nY = pitchY * height
        nUV = pitchUV * uvh
        fY = [rng.randint(0, 255) for _ in range(nY)]
        pick = rng.random()
        if pick < 0.4:       # production domain: small flag bits, no wrap
            fU = [rng.randint(0, 7) for _ in range(nUV)]
            fV = [rng.randint(0, 7) for _ in range(nUV)]
        elif pick < 0.7:     # full uchar sweep: wrap path
            fU = [rng.randint(0, 255) for _ in range(nUV)]
            fV = [rng.randint(0, 255) for _ in range(nUV)]
        else:                # mixed
            fU = [rng.randint(0, 7) for _ in range(nUV)]
            fV = [rng.randint(0, 255) for _ in range(nUV)]
        hdr = [ord('F'), width, height, pitchY, pitchUV, lx, ly, nY, nUV]
        got = run_mirror(hdr + fY + fU + fV)
        exp = golden_merge_uvflags(width, height, pitchY, pitchUV, lx, ly,
                                   fY, fU, fV)
        total += 1
        if got != exp:
            ok = False
            for i, (g, e) in enumerate(zip(got, exp)):
                if g != e:
                    print("merge_uvflags MISMATCH", width, height, lx, ly,
                          "px", i, g, e)
                    break
            if total >= 3: break

    # C: copy_border (extreme rows through; dst pre-filled to catch strays)
    for _ in range(120):
        bits = rng.choice([8, 8, 16])
        maxv = 255 if bits == 8 else 65535
        width = rng.randint(1, 32)   # scalar twin: arbitrary
        height = rng.randint(1, 16)
        vborder = rng.randint(1, height)  # incl. vborder*2 > height overlap
        pitch = width + rng.choice([0, 1, 3])
        n = pitch * height
        src = [rng.randint(0, maxv) for _ in range(n)]
        # dst differs from src at EVERY pixel: border must be overwritten,
        # interior must survive
        off = (maxv + 1) // 2
        dst = [(s + off) % (maxv + 1) for s in src]
        hdr = [ord('C'), width, height, pitch, vborder, n]
        got = run_mirror(hdr + src + dst)
        exp = golden_copy_border(src, dst, width, height, pitch, vborder)
        total += 1
        if got != exp:
            ok = False
            for i, (g, e) in enumerate(zip(got, exp)):
                if g != e:
                    print("copy_border MISMATCH", width, height, vborder,
                          bits, "px", i, g, e)
                    break
            if total >= 3: break

    # N: analyze_frame (all rows incl. borders via padded sources; the strict->
    # threshold boundaries are pinned by crafted fields, not luck)
    for case in range(160):
        bits = rng.choice([8, 8, 16])
        maxv = 255 if bits == 8 else 65535
        width = rng.randint(1, 12) * 4   # 4-wide twin: mult of 4
        height = rng.randint(1, 18)
        pitch = width + rng.choice([0, 2])
        dpitch = width + rng.choice([0, 1, 3])
        nS = pitch * (height + 2)
        mode = case % 6
        if mode == 0:
            # uniform field: t = 0, diff = 0 -> pins the 0 boundary
            v = rng.randint(0, maxv)
            base = [v] * nS
            sref = [v] * nS
            mref = [v] * nS
            tM = rng.choice([-1, 0, 1])
            tS = rng.choice([-1, 0, 1])
            tLS = rng.choice([-1, 0, 1])
        elif mode == 1:
            # LSB-flipped mref: diff = 1 everywhere -> pins the 1 boundary
            base = [rng.randint(0, maxv) for _ in range(nS)]
            sref = [rng.randint(0, maxv) for _ in range(nS)]
            mref = [b ^ 1 for b in base]
            tM = rng.choice([-1, 0, 1, 2])
            tS = rng.choice([0, 100, 6 * maxv, 10 ** 9])
            tLS = rng.choice([0, 100, 6 * maxv, 10 ** 9])
        elif mode == 2:
            # opposed-phase stripes: t = maxv everywhere
            base = [(maxv if r % 2 else 0)
                    for r in range(height + 2) for _ in range(pitch)]
            sref = [(0 if r % 2 else maxv)
                    for r in range(height + 2) for _ in range(pitch)]
            mref = list(base)  # diff = 0
            tM = rng.choice([-1, 0])
            tS = rng.choice([maxv - 1, maxv, maxv + 1])
            tLS = rng.choice([maxv - 1, maxv, maxv + 1])
        elif mode == 3:
            # base = maxv, sref = 0: t = 6*maxv (ceiling) everywhere
            base = [maxv] * nS
            sref = [0] * nS
            mref = [rng.randint(0, maxv) for _ in range(nS)]
            tM = rng.choice([0, maxv])
            tS = rng.choice([6 * maxv - 1, 6 * maxv])
            tLS = rng.choice([6 * maxv - 1, 6 * maxv])
        else:
            # random fuzz with wide thresholds
            base = [rng.randint(0, maxv) for _ in range(nS)]
            sref = [rng.randint(0, maxv) for _ in range(nS)]
            mref = [rng.randint(0, maxv) for _ in range(nS)]
            tM = rng.choice([-1, 0, 1, 100, maxv, maxv + 1, 10 ** 9])
            tS = rng.choice([-1, 0, 1, 1000, 6 * maxv, 10 ** 9])
            tLS = rng.choice([-1, 0, 1, 1000, 6 * maxv, 10 ** 9])
        hdr = [ord('N'), width, height, pitch, dpitch, tM, tS, tLS, nS]
        got = run_mirror(hdr + base + sref + mref)
        exp = golden_analyze_frame(width, height, pitch, dpitch, tM, tS, tLS,
                                   base, sref, mref)
        total += 1
        if got != exp:
            ok = False
            for i, (g, e) in enumerate(zip(got, exp)):
                if g != e:
                    print("analyze_frame MISMATCH", width, height, bits,
                          tM, tS, tLS, "px", i, g, e)
                    break
            if total >= 3: break

    # P: copy_pad (vector-form mirror vs scalar-form golden proves the lane
    # swap cancels; plus a copy->padv->padh composition cross-check per case)
    for _ in range(150):
        bits = rng.choice([8, 8, 16])
        maxv = 255 if bits == 8 else 65535
        width = rng.randint(1, 16) * 4      # mult of 4: vector twin
        height = rng.randint(1, 24)
        hpad = rng.choice([0, 0, 4, 4, 8])  # hpad4 vectors; 0 = production
        hpad = min(hpad, width)             # single-reflection precondition
        vpad = min(rng.choice([0, 1, 2, 4, 8]), height)
        srcpitch = width + rng.choice([0, 2])
        dstpitch = width + 2 * hpad + rng.choice([0, 1, 3])
        nS = srcpitch * height
        nD = dstpitch * (height + 2 * vpad)
        pick = rng.random()
        if pick < 0.7:
            src = [rng.randint(0, maxv) for _ in range(nS)]
        elif pick < 0.85:                   # row gradient: mirror-visible
            src = [(yy * 257) % (maxv + 1)
                   for yy in range(height) for _ in range(srcpitch)]
        else:                               # col gradient
            src = [(xx * 257) % (maxv + 1)
                   for _ in range(height) for xx in range(srcpitch)]
        hdr = [ord('P'), width, height, srcpitch, dstpitch, hpad, vpad,
               nS, nD]
        got = run_mirror(hdr + src)
        exp = golden_copy_pad(src, width, height, srcpitch, dstpitch,
                              hpad, vpad)
        comp = composed_copy_pad(src, width, height, srcpitch, dstpitch,
                                 hpad, vpad)
        total += 1
        if comp != exp:
            ok = False
            print("copy_pad PROOF MISMATCH (composition vs direct)",
                  width, height, hpad, vpad, bits)
            for i, (g, e) in enumerate(zip(comp, exp)):
                if g != e:
                    print("  px", i, g, e)
                    break
            if total >= 3: break
        if got != exp:
            ok = False
            for i, (g, e) in enumerate(zip(got, exp)):
                if g != e:
                    print("copy_pad MISMATCH", width, height, hpad, vpad,
                          bits, "px", i, g, e)
                    break
            if total >= 3: break

    # O: copy_pad_2plane (per-plane copy_pad over a U/V pair)
    for _ in range(120):
        bits = rng.choice([8, 8, 16])
        maxv = 255 if bits == 8 else 65535
        width = rng.randint(1, 12) * 4
        height = rng.randint(1, 20)
        hpad = min(rng.choice([0, 0, 4, 8]), width)
        vpad = min(rng.choice([0, 1, 2, 4, 8]), height)
        srcpitch = width + rng.choice([0, 2])
        dstpitch = width + 2 * hpad + rng.choice([0, 1, 3])
        nS = srcpitch * height
        nD = dstpitch * (height + 2 * vpad)
        src0 = [rng.randint(0, maxv) for _ in range(nS)]
        src1 = [rng.randint(0, maxv) for _ in range(nS)]
        hdr = [ord('O'), width, height, srcpitch, dstpitch, hpad, vpad,
               nS, nD]
        got = run_mirror(hdr + src0 + src1)
        exp = (golden_copy_pad(src0, width, height, srcpitch, dstpitch,
                               hpad, vpad) +
               golden_copy_pad(src1, width, height, srcpitch, dstpitch,
                               hpad, vpad))
        comp = (composed_copy_pad(src0, width, height, srcpitch, dstpitch,
                                  hpad, vpad) +
                composed_copy_pad(src1, width, height, srcpitch, dstpitch,
                                  hpad, vpad))
        total += 1
        if comp != exp:
            ok = False
            print("copy_pad_2plane PROOF MISMATCH (composition vs direct)",
                  width, height, hpad, vpad, bits)
            if total >= 3: break
        if got != exp:
            ok = False
            for i, (g, e) in enumerate(zip(got, exp)):
                if g != e:
                    print("copy_pad_2plane MISMATCH", width, height, hpad,
                          vpad, bits, "px", i, g, e)
                    break
            if total >= 3: break

    # S: extend_blocks_h solo (incl. nBlkX == 1 branch-order pin)
    for _ in range(110):
        nBlkX = rng.choice([1, 1, 2, 2, 3, 5, 9, 16, 24])
        nBlkY = rng.randint(1, 16)
        pitch = nBlkX + rng.choice([0, 1, 3])
        n = pitch * nBlkY
        src = [rng.randint(0, 255) for _ in range(n)]
        hdr = [ord('S'), nBlkX, nBlkY, pitch, n]
        got = run_mirror(hdr + src)
        exp = golden_extend_h(src, nBlkX, nBlkY, pitch)
        total += 1
        if got != exp:
            ok = False
            for i, (g, e) in enumerate(zip(got, exp)):
                if g != e:
                    print("extend_blocks_h MISMATCH", nBlkX, nBlkY,
                          "px", i, g, e)
                    break
            if total >= 3: break

    # T: extend_blocks_v solo (incl. nBlkY == 1 branch-order pin)
    for _ in range(110):
        nBlkX = rng.randint(1, 16)
        nBlkY = rng.choice([1, 1, 2, 2, 3, 5, 9, 16, 24])
        pitch = nBlkX + rng.choice([0, 1, 3])
        n = pitch * nBlkY
        src = [rng.randint(0, 255) for _ in range(n)]
        hdr = [ord('T'), nBlkX, nBlkY, pitch, n]
        got = run_mirror(hdr + src)
        exp = golden_extend_v(src, nBlkX, nBlkY, pitch)
        total += 1
        if got != exp:
            ok = False
            for i, (g, e) in enumerate(zip(got, exp)):
                if g != e:
                    print("extend_blocks_v MISMATCH", nBlkX, nBlkY,
                          "px", i, g, e)
                    break
            if total >= 3: break

    # U: extend_blocks composed (ping-pong mirror vs in-place-CPU golden —
    # different code paths, same result, proves h->v equals the CPU twin)
    for _ in range(160):
        nBlkX = rng.randint(2, 24)   # >= 2: the CPU twin is OOB at 1
        nBlkY = rng.randint(2, 24)
        pitch = nBlkX + rng.choice([0, 1, 3])
        n = pitch * nBlkY
        pick = rng.random()
        if pick < 0.5:
            src = [rng.randint(0, 255) for _ in range(n)]
        elif pick < 0.75:            # production flag domain
            src = [rng.randint(0, 7) for _ in range(n)]
        else:                        # ramps: max-spread direction visible
            src = [(xx + 3 * yy) % 256 for yy in range(nBlkY)
                   for xx in range(pitch)]
        hdr = [ord('U'), nBlkX, nBlkY, pitch, n]
        got = run_mirror(hdr + src)
        exp = golden_extend_cpu(src, nBlkX, nBlkY, pitch)
        total += 1
        if got != exp:
            ok = False
            for i, (g, e) in enumerate(zip(got, exp)):
                if g != e:
                    print("extend_blocks_hv MISMATCH", nBlkX, nBlkY,
                          "px", i, g, e)
                    break
            if total >= 3: break

    # Y: copy (incl. the CombingAnalyze field-copy shape: doubled pitch)
    for t in range(150):
        bits = rng.choice([8, 8, 16])
        maxv = 255 if bits == 8 else 65535
        width = rng.randint(1, 48)
        height = rng.randint(1, 32)
        if t % 4 == 3:  # field-copy shape: even height, doubled pitch
            height = rng.choice([2, 4, 6, 8, 12, 16, 24])
            pitch = 2 * width + rng.choice([0, 0, 2, 4])
        else:
            pitch = width + rng.choice([0, 0, 1, 2, 3])
        n = pitch * height
        src = [rng.randint(0, maxv) for _ in range(n)]
        hdr = [ord('Y'), width, height, pitch, n]
        got = run_mirror(hdr + src)
        exp = [src[xx + yy * pitch]
               for yy in range(height) for xx in range(width)]
        total += 1
        if got != exp:
            ok = False
            for i, (g, e) in enumerate(zip(got, exp)):
                if g != e:
                    print("copy MISMATCH", width, height, "px", i, g, e)
                    break

    # Q: copy_2plane (plane independence both ways)
    for t in range(150):
        bits = rng.choice([8, 8, 16])
        maxv = 255 if bits == 8 else 65535
        width = rng.randint(1, 40)
        height = rng.randint(1, 28)
        pitch = width + rng.choice([0, 0, 1, 2])
        n = pitch * height
        s0 = [rng.randint(0, maxv) for _ in range(n)]
        craft = t % 3
        if craft == 0:
            s1 = [rng.randint(0, maxv) for _ in range(n)]
        elif craft == 1:  # identical planes
            s1 = list(s0)
        else:  # complementary: every pixel differs
            s1 = [(v + 1) % (maxv + 1) for v in s0]
        hdr = [ord('Q'), width, height, pitch, n]
        got = run_mirror(hdr + s0 + s1)
        exp = ([s0[xx + yy * pitch] for yy in range(height)
                for xx in range(width)] +
               [s1[xx + yy * pitch] for yy in range(height)
                for xx in range(width)])
        total += 1
        if got != exp:
            ok = False
            for i, (g, e) in enumerate(zip(got, exp)):
                if g != e:
                    print("copy_2plane MISMATCH", width, height, "px", i, g, e)
                    break

    # L: fill (v sweep incl. 0/255/256/65535 edges + production zero)
    for t in range(150):
        bits = rng.choice([8, 8, 16])
        maxv = 255 if bits == 8 else 65535
        width = rng.randint(1, 48)
        height = rng.randint(1, 32)
        pitch = width + rng.choice([0, 1, 2])
        v = rng.choice([0, 0, 0, 1, 2, 127, 128, 255, 256, 1000, 32767,
                        32768, 65534, 65535, rng.randint(0, maxv)])
        v = min(v, maxv)
        hdr = [ord('L'), width, height, pitch, v]
        got = run_mirror(hdr)
        exp = [v] * (width * height)
        total += 1
        if got != exp:
            ok = False
            print("fill MISMATCH", width, height, v, got[:4], exp[:4])

    print(f"KFM FilterBase (calc_combe/merge_uvcoefs/extend_coef2/"
          f"apply_uvcoefs_420/padv/padh/merge_block/average/max/"
          f"merge_uvflags/copy_border/analyze_frame/copy_pad/"
          f"copy_pad_2plane/extend_blocks_h/extend_blocks_v/"
          f"extend_blocks_hv/copy/copy_2plane/fill): "
          f"{'PASS' if ok else 'FAIL'} ({total} cases)")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
