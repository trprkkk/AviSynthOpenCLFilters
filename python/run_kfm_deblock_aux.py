"""Independent golden vs sim/kfm_deblock_aux_ref.cpp (graduated KDeblock helpers).

Modes: S scale_qp, C sharpen_coeff, H max_h, V max_v, B max_vh, G merge,
P sharpen, W show_sharpen_coeff.  P/W pin the deterministic
(manual-bilinear) behaviour only; vs CUDA texture filtering a device run
is still required, so the kernels stay // RIG-VERIFY (not graduated).
The B golden
is the separable composition max_h o max_v (box-max separability), which
cross-checks the direct box form in the mirror per the handoff recipe.
"""
import os
import random
import struct
import subprocess
import sys
import tempfile

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BIN = os.path.join(tempfile.gettempdir(), "kfm_deblock_aux_ref")
BIN_DB = os.path.join(tempfile.gettempdir(), "kfm_deblock_ref_e2e")

SHARPEN_COEFF = [
    0, 0, 0, 0, 0,
    0, 0, 0, 0, 10,
    50, 90, 120, 150, 160,
    170, 180, 190, 200, 210,
    220, 230, 240, 245, 250,
    255, 255, 255, 255, 255,
]


def F(v):
    return struct.unpack('f', struct.pack('f', v))[0]


def FI(v):
    return struct.unpack('i', struct.pack('f', v))[0]


LDITHER = [
    [[0, 48, 12, 60], [3, 51, 15, 63]],
    [[32, 16, 44, 28], [35, 19, 47, 31]],
    [[8, 56, 4, 52], [11, 59, 7, 55]],
    [[40, 24, 36, 20], [43, 27, 39, 23]],
    [[2, 50, 14, 62], [1, 49, 13, 61]],
    [[34, 18, 46, 30], [33, 17, 45, 29]],
    [[10, 58, 6, 54], [9, 57, 5, 53]],
    [[42, 26, 38, 22], [41, 25, 37, 21]],
]


def run_mirror(nums):
    inf = os.path.join(tempfile.gettempdir(), "kdx_in.txt")
    outf = os.path.join(tempfile.gettempdir(), "kdx_out.txt")
    with open(inf, "w") as f:
        f.write(" ".join(map(str, nums)) + "\n")
    subprocess.run([BIN, inf, outf], check=True)
    with open(outf) as f:
        return [int(line) for line in f]


def norm_qscale(q, t):
    if t == 0:
        return q << 2
    if t == 1:
        return q << 1
    if t == 2:
        return q
    if t == 3:
        return 63 - q + 2
    return q


def golden_scale_qp(width, height, src_pitch, stype, src):
    return [norm_qscale(src[x + y * src_pitch], stype) % 256
            for y in range(height) for x in range(width)]


def golden_sharpen_coeff(width, height, qp_pitch, qp):
    out = []
    for y in range(height):
        for x in range(width):
            q = qp[x + y * qp_pitch] >> 3
            out.append(255 if q >= 25 else SHARPEN_COEFF[q])
    return out


def golden_max_h(width, height, pitch, radius, src, org):
    return [max([0] + [src[org + (x + d) + y * pitch]
                       for d in range(-radius, radius + 1)])
            for y in range(height) for x in range(width)]


def golden_max_v(width, height, pitch, radius, src, org):
    return [max([0] + [src[org + x + (y + d) * pitch]
                       for d in range(-radius, radius + 1)])
            for y in range(height) for x in range(width)]


def run_deblock_mirror(nums):
    inf = os.path.join(tempfile.gettempdir(), "kdx_db_in.txt")
    outf = os.path.join(tempfile.gettempdir(), "kdx_db_out.txt")
    with open(inf, "w") as f:
        f.write(" ".join(map(str, nums)) + "\n")
    subprocess.run([BIN_DB, inf, outf], check=True)
    with open(outf) as f:
        return [int(line) for line in f]


def golden_bilinear(coeff, coeff_pitch, x, y):
    fx = F(F(float(x)) * F(1.0 / 8.0))
    fy = F(F(float(y)) * F(1.0 / 8.0))
    ix, iy = int(fx), int(fy)
    c00 = F(float(coeff[ix + iy * coeff_pitch]))
    c01 = F(float(coeff[ix + 1 + iy * coeff_pitch]))
    c10 = F(float(coeff[ix + (iy + 1) * coeff_pitch]))
    c11 = F(float(coeff[ix + 1 + (iy + 1) * coeff_pitch]))
    fracx = F(fx - F(float(ix)))
    fracy = F(fy - F(float(iy)))
    top = F(F(c00 * F(1.0 - fracx)) + F(c01 * fracx))
    bot = F(F(c10 * F(1.0 - fracx)) + F(c11 * fracx))
    return F(F(top * F(1.0 - fracy)) + F(bot * fracy))


def golden_sharpen(width, height, pitch, src_pitch, coeff_pitch, src, coeff,
                   unsharp, quirk):
    # quirk=True: device form min(x+1,height-1); False: width-1 clamp.
    xcap = height - 1 if quirk else width - 1
    out = []
    for y in range(height):
        for x in range(width):
            s = src[x + y * src_pitch]
            l = h = s
            xm1 = x - 1 if x - 1 >= 0 else 0
            xp1 = x + 1 if x + 1 <= xcap else xcap
            ym1 = y - 1 if y - 1 >= 0 else 0
            yp1 = y + 1 if y + 1 <= height - 1 else height - 1
            for tx, ty in ((xm1, ym1), (x, ym1), (xp1, ym1), (xm1, y),
                           (xp1, y), (xm1, yp1), (x, yp1), (xp1, yp1)):
                vv = src[tx + ty * src_pitch]
                l = vv if vv < l else l
                h = vv if vv > h else h
            c = F(golden_bilinear(coeff, coeff_pitch, x, y) * F(1.0 / 255.0))
            u = unsharp[x + y * pitch]
            r = F(F(F(float(s)) + F(F(float(s - u)) * c)) + 0.5)
            r = F(float(l)) if r < F(float(l)) else r
            r = F(float(h)) if r > F(float(h)) else r
            out.append(int(r))
    return out


def golden_show(coeff, coeff_pitch, width, height):
    return [int(golden_bilinear(coeff, coeff_pitch, x, y))
            for y in range(height) for x in range(width)]


def golden_merge(vis_w, vis_h, pitch_u4, ipitch, shift, maxv, tmp):
    # packed-quad addressing from the CUDA form (quad (X, row) holds lanes
    # (X + row*pitch_u4)*4 + L); float32 per op; fmin; C truncation.
    pitch_us = pitch_u4 * 4
    org = 8 + 8 * pitch_us
    inv = F(1.0 / float(1 << shift))
    sixth = F(1.0 / 64.0)
    maxv_f = F(float(maxv))
    out = []
    for y in range(vis_h):
        for x in range(vis_w):
            X = x >> 2
            L = x & 3
            s = 0
            for k in range(4):
                row = ipitch * k + y
                s += tmp[org + (X + row * pitch_u4) * 4 + L]
            v = F(F(F(float(s)) * inv) + F(F(float(LDITHER[y & 7][X & 1][L])) * sixth))
            out.append(int(v if v < maxv_f else maxv_f))
    return out


def golden_max_vh_sep(width, height, pitch, radius, src, org):
    # separable composition: horizontal pass over interior columns and the
    # radius-halo rows (the only cells the vertical pass consumes), then a
    # vertical pass.  All reads stay inside the 8px margin (radius <= 8).
    H2 = height + 2 * radius
    hpass = [0] * (width * H2)
    for yy in range(-radius, height + radius):
        for xx in range(width):
            hpass[xx + (yy + radius) * width] = max(
                [0] + [src[org + (xx + d) + yy * pitch]
                       for d in range(-radius, radius + 1)])
    return [max([0] + [hpass[x + (y + j + radius) * width]
                       for j in range(-radius, radius + 1)])
            for y in range(height) for x in range(width)]


def main():
    # -mfpmath=sse -msse2: per-op float32 required for bit-exactness vs the
    # golden (32-bit x87 toolchains keep 80-bit intermediates — see
    # run_kfm_deblock.py).
    subprocess.run(["g++", "-O2", "-std=gnu++17", "-Wl,--allow-multiple-definition", "-ffp-contract=off",
                    "-mfpmath=sse", "-msse2", "-w",
                    os.path.join(REPO, "sim", "kfm_deblock_aux_ref.cpp"),
                    "-o", BIN], check=True)
    subprocess.run(["g++", "-O2", "-std=gnu++17", "-Wl,--allow-multiple-definition", "-ffp-contract=off",
                    "-mfpmath=sse", "-msse2", "-w",
                    os.path.join(REPO, "sim", "kfm_deblock_ref.cpp"),
                    "-o", BIN_DB], check=True)
    rng = random.Random(131)
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

    # S: scale_qp (wrap pins: type0 q=64->0, q=255->252; OOR types 4/7)
    for t in range(200):
        width = rng.choice([1, 2, 3, 4, 5, 8, 13, 16, 31, 32, 40])
        height = rng.choice([1, 2, 3, 5, 8, 16, 24, 33])
        src_pitch = width + rng.choice([0, 0, 1, 2])
        dst_pitch = width + rng.choice([0, 1, 3])
        stype = rng.choice([0, 1, 2, 3, 3, 4, 7]) if t % 4 else t % 8
        nS = src_pitch * height
        craft = t % 4
        if craft == 0:
            src = [rng.randint(0, 255) for _ in range(nS)]
        elif craft == 1:  # full-range sweep hits every wrap edge
            src = [(i * 37 + t) % 256 for i in range(nS)]
        elif craft == 2:  # wrap-boundary values
            src = [rng.choice([0, 1, 62, 63, 64, 65, 127, 128, 254, 255])
                   for _ in range(nS)]
        else:
            src = [rng.choice([0, 51, 255])] * nS
        nums = ["S", width, height, dst_pitch, src_pitch, stype, nS] + src
        check("S", run_mirror(nums),
              golden_scale_qp(width, height, src_pitch, stype, src),
              (width, height, stype))

    # C: sharpen_coeff (boundary emphasis qp 191..215 i.e. q 23/24/25)
    for t in range(200):
        width = rng.choice([1, 2, 3, 4, 5, 8, 12, 16, 24, 32])
        height = rng.choice([1, 2, 3, 4, 7, 8, 12, 16])
        qp_pitch = width + rng.choice([0, 0, 1, 2])
        pitch = width + rng.choice([0, 1])
        nQ = qp_pitch * height
        craft = t % 5
        if craft == 0:
            qp = [rng.randint(0, 65535) for _ in range(nQ)]
        elif craft == 1:
            qp = [rng.randint(191, 215) for _ in range(nQ)]
        elif craft == 2:  # every q in 0..29 reachable (qp = 8q+r)
            qp = [(rng.randint(0, 29) * 8 + rng.randint(0, 7))
                  for _ in range(nQ)]
        elif craft == 3:  # production QP range + saturation tail
            qp = [rng.choice([0, 1, 40, 51 * 4, 200, 400, 1000, 65535])
                  for _ in range(nQ)]
        else:
            qp = [(i * 257 + t * 13) % 65536 for i in range(nQ)]
        nums = ["C", width, height, pitch, qp_pitch, nQ] + qp
        check("C", run_mirror(nums),
              golden_sharpen_coeff(width, height, qp_pitch, qp),
              (width, height))

    # H/V/B: max trio on 8px/side-padded planes, radius 1..8 (5 = production)
    for t in range(450):
        mode = "HVB"[t % 3]
        width = rng.choice([1, 2, 3, 4, 5, 8, 13, 16, 31, 32, 40])
        height = rng.choice([1, 2, 3, 5, 8, 13, 16, 24, 32])
        pitch = width + 16 + rng.choice([0, 0, 1, 2])
        radius = rng.choice([1, 2, 3, 5, 5, 5, 6, 8])
        nP = pitch * (height + 16)
        org = 8 + 8 * pitch
        craft = (t // 3) % 5
        if craft == 0:
            src = [rng.randint(0, 255) for _ in range(nP)]
        elif craft == 1:  # single spike pins the window extent exactly
            src = [0] * nP
            sx = rng.randrange(width)
            sy = rng.randrange(height)
            src[org + sx + sy * pitch] = 255
        elif craft == 2:  # all zeros -> 0-seed rule
            src = [0] * nP
        elif craft == 3:  # spikes on the margin edge (radius reach)
            src = [0] * nP
            for _ in range(3):
                mx = rng.randrange(-8, width + 8)
                my = rng.randrange(-8, height + 8)
                src[org + mx + my * pitch] = rng.randint(1, 255)
        else:  # ramp
            src = [(i * 11) % 256 for i in range(nP)]
        nums = [mode, width, height, pitch, radius, nP] + src
        if mode == "H":
            exp = golden_max_h(width, height, pitch, radius, src, org)
        elif mode == "V":
            exp = golden_max_v(width, height, pitch, radius, src, org)
        else:
            exp = golden_max_vh_sep(width, height, pitch, radius, src, org)
        check(mode, run_mirror(nums), exp, (width, height, radius))

    # G: merge_deblock (vis_w mult of 4; vis_h <= ipitch; shift/maxv from
    # quality 1..6 x bits 8/10/12/16; margins carry distinctive sentinels so
    # a wrong +8/+8 pre-offset would corrupt the result)
    for t in range(250):
        bits = rng.choice([8, 10, 12, 16])
        maxv = (1 << bits) - 1
        quality = rng.randint(1, 6)
        dbs = max(0, quality + bits - 10)
        shift = quality + 6 - dbs
        ipitch = 8 * rng.choice([1, 1, 2, 3])
        vis_w = rng.choice([4, 8, 12, 16, 20, 24, 32, 36, 40, 44, 48])
        vis_h = rng.randint(1, ipitch)
        pitch_us = 8 + vis_w + rng.choice([0, 0, 1, 2, 3])
        pitch_us += (-pitch_us) % 4
        pitch_u4 = pitch_us // 4
        rows = 8 + 4 * ipitch + rng.choice([0, 1])
        nT = pitch_us * rows
        org = 8 + 8 * pitch_us
        craft = t % 7
        if craft == 0:
            tmp = [rng.randint(0, 65535) for _ in range(nT)]
        elif craft == 1:  # zeros -> dither-only path, all outputs 0
            tmp = [0] * nT
        elif craft == 2:  # maxed acc -> fmin clamp pins
            tmp = [65535] * nT
        elif craft == 3:  # single-cell spike pins k/X/L/y at once
            tmp = [0] * nT
            k = rng.randrange(4)
            sx = rng.randrange(vis_w)
            sy = rng.randrange(vis_h)
            tmp[org + sx + (ipitch * k + sy) * pitch_us] = rng.randint(1, 65535)
        elif craft == 4:  # single-quad spike pins lane/column fan-out
            tmp = [0] * nT
            X0 = rng.randrange(vis_w // 4)
            k = rng.randrange(4)
            sy = rng.randrange(vis_h)
            for L in range(4):
                tmp[org + (X0 * 4 + L) + (ipitch * k + sy) * pitch_us] = \
                    rng.randint(1, 65535)
        elif craft == 5:  # uniform slices summing to 64k+63 at shift 6:
            # outputs flip k0 vs k0+1 exactly where dither >= 1, pinning
            # all three dither indices plus the float boundary
            bits = 10
            maxv = 1023
            shift = 6  # quality+6-(quality+0) for bits=10, any quality
            k0 = rng.choice([0, 1, 2])
            c = [16 + 16 * k0] * 3 + [15 + 16 * k0]
            tmp = [rng.randint(0, 65535) for _ in range(nT)]
            for k in range(4):
                for yy in range(4 * ipitch):
                    for xx in range(vis_w):
                        tmp[org + xx + (ipitch * k + yy % ipitch) * pitch_us] = c[k]
            for yy in range(vis_h, ipitch):  # keep unused rows distinctive
                for k in range(4):
                    for xx in range(vis_w):
                        tmp[org + xx + (ipitch * k + yy) * pitch_us] = \
                            rng.randint(0, 65535)
        else:  # sparse spikes on zero field
            tmp = [0] * nT
            for _ in range(20):
                k = rng.randrange(4)
                sx = rng.randrange(vis_w)
                sy = rng.randrange(vis_h)
                tmp[org + sx + (ipitch * k + sy) * pitch_us] = rng.randint(1, 65535)
        if craft != 5:
            for r in range(8):  # top-margin sentinels
                for c in range(pitch_us):
                    tmp[c + r * pitch_us] = rng.randint(0, 65535)
            for r in range(8, rows):  # left-margin sentinels
                for c in range(8):
                    tmp[c + r * pitch_us] = rng.randint(0, 65535)
        nums = ["G", vis_w, vis_h, pitch_u4, ipitch, vis_w, shift, maxv, nT] + tmp
        check("G", run_mirror(nums),
              golden_merge(vis_w, vis_h, pitch_u4, ipitch, shift, maxv, tmp),
              (vis_w, vis_h, shift, maxv, craft))

    # E2E (handoff section 4.1/section 5): real kf_deblock accumulator fed to
    # the merge mirror with tmp_pitch_u4 = acc_pitch_ushort >> 2, compared
    # against the packed-quad golden.  Unwritten (-1) mirror cells map to 0
    # (production tmpOut is written fully over the visible region; the vis
    # window stays inside written cols [0, bw*8+8)).
    for t in range(15):
        bits = rng.choice([8, 10, 12, 16])
        maxvpx = (1 << bits) - 1
        bw = rng.choice([1, 2])
        bh = rng.choice([1, 2])
        quality = rng.choice([1, 2, 3])
        dbs = max(0, quality + bits - 10)
        mshift = quality + 6 - dbs
        mmaxv = (1 << bits) - 1
        deblock_maxv = (1 << (bits + 6 - dbs)) - 1
        sw = bw * 8 + 16
        sh = bh * 8 + 16
        src = [rng.randint(0, maxvpx) for _ in range(sh * sw)]
        qp = [rng.randint(0, 40) for _ in range(bw * bh)]
        strength = F(rng.choice([4.0, 8.0, 20.0]))
        ta = F(rng.choice([0.02, 0.05, 0.08]))
        tb = F(rng.choice([-1.0, -0.5, 0.0]))
        out_pitch = sw
        hdr = [68, sw, sh, bh, out_pitch, bw, (1 << quality) - 1, dbs,
               deblock_maxv, FI(strength), FI(ta), FI(tb), bw, bw * bh]
        acc = run_deblock_mirror(hdr + qp + [sh * sw] + src)
        rows_acc = 32 * bh + 8
        assert len(acc) == out_pitch * rows_acc, (len(acc), out_pitch, rows_acc)
        ipitch = 8 * bh
        vis_w = 4 * rng.randint(1, (bw * 8 + 8) // 4)
        vis_h = rng.randint(1, 8 * bh)
        pitch_us = 8 + out_pitch
        assert pitch_us % 4 == 0
        pitch_u4 = pitch_us // 4
        rows = 8 + rows_acc
        nT = pitch_us * rows
        tmp = [rng.randint(0, 65535) for _ in range(nT)]
        for r in range(rows_acc):
            for c in range(out_pitch):
                a = acc[c + r * out_pitch]
                tmp[(8 + c) + (8 + r) * pitch_us] = a if a >= 0 else 0
        nums = ["G", vis_w, vis_h, pitch_u4, ipitch, vis_w, mshift, mmaxv, nT] + tmp
        check("E", run_mirror(nums),
              golden_merge(vis_w, vis_h, pitch_u4, ipitch, mshift, mmaxv, tmp),
              (bw, bh, quality, bits))

    # section 5 packing identity: CUDA packed-ushort2 writes == scalar ushort
    # writes when the byte stride matches (P = 2*P2), incl. the tile index
    # correspondence (bbx*4+tx)*2+j == bbx*8+2*tx+j.  Distinctive values.
    for t in range(5):
        W2 = rng.choice([4, 8, 16, 32])
        H = rng.choice([1, 2, 8, 16])
        P2 = W2 + rng.choice([0, 1, 2])
        P = 2 * P2
        assert P * H <= 65536
        def v(r, i):
            return (r * P + i) * 3 + 1
        scalar = [v(r, i) for r in range(H) for i in range(2 * W2)]
        words = [0] * (P2 * H)
        for r in range(H):  # emulate CUDA packed ushort2 writes
            for u in range(W2):
                words[r * P2 + u] = v(r, 2 * u) | (v(r, 2 * u + 1) << 16)
        back = []
        for r in range(H):  # unpack little-endian, read as scalar ushorts
            for u in range(W2):
                w = words[r * P2 + u]
                back += [w & 0xFFFF, (w >> 16) & 0xFFFF]
        idx_ok = all((bbx * 4 + tx) * 2 + j == bbx * 8 + 2 * tx + j
                     for bbx in range(8) for tx in range(8) for j in range(2))
        check("L", back + [1 if idx_ok else 0], scalar + [1], (W2, H, P2))

    # P: sharpen pin (width mult of 8; quirk configs width>height; bits 8/16)
    for t in range(250):
        maxv = rng.choice([255, 65535])
        width = rng.choice([8, 16, 24, 32, 40, 48, 64])
        height = rng.choice([1, 2, 3, 4, 5, 7, 8, 9, 12, 13, 15, 16, 20,
                             24, 32, 33, 40])
        pitch = width + rng.choice([0, 0, 1, 2])
        # +1 guard column: the quirk tap min(x+1,height-1) reads column
        # `width` when height > width (production's pad margin covers it).
        src_pitch = width + 1 + rng.choice([0, 0, 1, 2])
        qpw = (width + 15) >> 3
        qph = (height + 15) >> 3
        coeff_pitch = qpw + rng.choice([0, 0, 1])
        nS = src_pitch * height
        nC = coeff_pitch * qph
        nU = pitch * height
        # taps-in-bounds algebra (host contract, qp-sized + margin)
        assert ((width - 1) >> 3) + 1 < qpw + 1
        assert ((width - 1) >> 3) + 1 <= qpw
        assert ((height - 1) >> 3) + 1 <= qph
        craft = t % 6
        if craft == 0:
            src = [rng.randint(0, maxv) for _ in range(nS)]
            coeff = [rng.randint(0, 255) for _ in range(nC)]
            unsharp = [rng.randint(0, maxv) for _ in range(nU)]
        elif craft == 1:  # coeff 0 -> c=0 -> out == src (no-branch identity)
            src = [rng.randint(0, maxv) for _ in range(nS)]
            coeff = [0] * nC
            unsharp = [rng.randint(0, maxv) for _ in range(nU)]
        elif craft == 2:  # quirk-forcing: ramp cols, u=0, c~1 -> out=h
            src = [min(x % src_pitch, maxv) if (x % src_pitch) < width else 0
                   for x in range(nS)]
            coeff = [255] * nC
            unsharp = [0] * nU
        elif craft == 3:  # coeff ramp 0->255 hits every trunc boundary
            src = [rng.randint(0, maxv) for _ in range(nS)]
            coeff = [(i * 7 + t) % 256 for i in range(nC)]
            unsharp = [rng.randint(0, maxv) for _ in range(nU)]
        elif craft == 4:  # flat src -> window collapses, out==s always
            c0 = rng.randint(0, maxv)
            src = [c0] * nS
            coeff = [rng.randint(0, 255) for _ in range(nC)]
            unsharp = [rng.randint(0, maxv) for _ in range(nU)]
        else:  # checker src, extreme unsharp
            src = [0 if (x // 1 + x // max(1, src_pitch)) % 2 == 0 else maxv
                   for x in range(nS)]
            coeff = [rng.choice([0, 1, 127, 128, 254, 255]) for _ in range(nC)]
            unsharp = [rng.choice([0, maxv]) for _ in range(nU)]
        nums = (["P", width, height, pitch, src_pitch, coeff_pitch, qph,
                 nS, nC, nU] + src + coeff + unsharp)
        exp = golden_sharpen(width, height, pitch, src_pitch, coeff_pitch,
                             src, coeff, unsharp, True)
        info = (width, height, maxv, craft)
        check("P", run_mirror(nums), exp, info)
        if craft == 1:  # c==0 identity: deterministic out == src visible
            total += 1
            vis = [src[x + y * src_pitch]
                   for y in range(height) for x in range(width)]
            if exp != vis:
                ok = False
                print("P", "C0-IDENTITY MISMATCH", info)
        if craft == 2 and width > height:  # quirk must bite, or test is void
            total += 1
            alt = golden_sharpen(width, height, pitch, src_pitch, coeff_pitch,
                                 src, coeff, unsharp, False)
            if alt == exp:
                ok = False
                print("P", "QUIRK-NOT-EXERCISED", info)

    # W: show_sharpen_coeff pin (fractional x/8, ramps, bits 8/16)
    for t in range(150):
        width = rng.choice([8, 16, 24, 32, 40, 48, 64])
        height = rng.choice([1, 2, 3, 5, 8, 9, 13, 16, 24, 31, 32])
        pitch = width + rng.choice([0, 0, 1])
        qpw = (width + 15) >> 3
        qph = (height + 15) >> 3
        coeff_pitch = qpw + rng.choice([0, 0, 1])
        nC = coeff_pitch * qph
        assert ((width - 1) >> 3) + 1 <= qpw
        assert ((height - 1) >> 3) + 1 <= qph
        craft = t % 4
        if craft == 0:
            coeff = [rng.randint(0, 255) for _ in range(nC)]
        elif craft == 1:  # full 0->255 ramp per row
            coeff = [(x * 255) // max(1, coeff_pitch - 1)
                     for x in range(nC)]
            coeff = [coeff[i % coeff_pitch] for i in range(nC)]
        elif craft == 2:  # extremes checkerboard
            coeff = [0 if (i + i // max(1, coeff_pitch)) % 2 == 0 else 255
                     for i in range(nC)]
        else:  # single-texel spike (bilinear fan-out)
            coeff = [0] * nC
            coeff[rng.randrange(nC)] = 255
        nums = (["W", width, height, pitch, coeff_pitch, qph, nC] + coeff)
        check("W", run_mirror(nums),
              golden_show(coeff, coeff_pitch, width, height),
              (width, height, craft))

    print("KFM KDeblock aux (scale_qp/sharpen_coeff/max_h/max_v/max_vh/"
          "merge_deblock+e2e/sharpen+show pins): PASS (%d cases)" % total)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
