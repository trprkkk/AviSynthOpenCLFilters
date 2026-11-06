#!/usr/bin/env python3
"""Validate the KFM KEdgeLevel kernels in src/opencl/kfm/kernels/kfm_edgelevel.cl
against the CPU mirror sim/kfm_edgelevel_ref.cpp with an independent Python
golden: kf_edgelevel (all check/selective/uv combos), kf_edgelevel_repair
(N=1..4, interior), kf_el_to444 and kf_el_from444.

The edge/factor/enhance math is float32 (same as CUDA/CPU).  The mirror is
compiled with -ffp-contract=off and the golden rounds every float32 operation,
so the two must match bit-for-bit.

Run:  python3 python/run_kfm_edgelevel.py
"""
import os, random, struct, subprocess, tempfile, sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BIN = os.path.join(tempfile.gettempdir(), "kfm_edgelevel_ref")


def F(v):
    return struct.unpack('f', struct.pack('f', v))[0]


def run_mirror(nums):
    inf = os.path.join(tempfile.gettempdir(), "kfe_in.txt")
    outf = os.path.join(tempfile.gettempdir(), "kfe_out.txt")
    with open(inf, "w") as f:
        f.write(" ".join(map(str, nums)))
    subprocess.run([BIN, inf, outf], check=True)
    with open(outf) as f:
        return [int(line) for line in f]


def el_scale(c, maxv):
    return int(F(F(F(c) / F(255.0)) * maxv))


def edgelevel_golden(width, height, pitch, maxv, st, stUV, thrs, check,
                     selective, uv, src, u, v):
    S = 1 if selective else 0
    dY = [0] * (pitch * height)
    dU = [0] * (pitch * height) if uv else None
    dV = [0] * (pitch * height) if uv else None
    for yy in range(height):
        for xx in range(width):
            off = yy * pitch + xx
            if yy <= (1 + S) or yy >= height - (2 + S) or xx <= (1 + S) or xx >= width - (2 + S):
                dY[off] = el_scale(16, maxv) if check else src[off]
                if uv:
                    dU[off] = u[off]; dV[off] = v[off]
                continue
            hmax = hmin = src[off - (2 + S)]
            vmax = vmin = src[off - (2 + S) * pitch]
            hdiffmax = vdiffmax = 0
            hprev, vprev = hmax, vmax
            for i in range(-(1 + S), 3 + S):
                hc, vc = src[off + i], src[off + i * pitch]
                hmax = max(hmax, hc); hmin = min(hmin, hc)
                vmax = max(vmax, vc); vmin = min(vmin, vc)
                if selective:
                    hdiffmax = max(hdiffmax, abs(hc - hprev))
                    vdiffmax = max(vdiffmax, abs(vc - vprev))
                hprev, vprev = hc, vc
            if hmax - hmin < vmax - vmin:
                hmax, hmin = vmax, vmin
            hdiffmax = max(hdiffmax, vdiffmax)
            factor = 1.0
            if selective:
                rdiff = F(F(hdiffmax) / F(hmax - hmin))
                a = F((0.55 - rdiff) * 10.0)
                b = F((0.35 - rdiff) * 10.0)
                a = 0.0 if a < 0.0 else (1.0 if a > 1.0 else a)
                b = 0.0 if b < 0.0 else (1.0 if b > 1.0 else b)
                factor = F(a - b)
            srcvY = src[off]
            if check:
                if hmax - hmin > thrs and factor > 0.0:
                    avgY = (hmax + hmin) >> 1
                    if srcvY > avgY:
                        dstvY = el_scale(50, maxv) if factor == 1.0 else el_scale(120, maxv)
                    else:
                        dstvY = el_scale(240, maxv) if factor == 1.0 else el_scale(180, maxv)
                else:
                    dstvY = el_scale(16, maxv)
                dstvU, dstvV = u[off], v[off]
            else:
                if hmax - hmin > thrs and factor > 0.0:
                    factorY = F(F(F(st) * factor) * 0.0625)
                    avgY = (hmax + hmin) >> 1
                    yv = srcvY + int(F(F(srcvY - avgY) * factorY))
                    yv = max(hmin, min(hmax, yv))
                    yv = max(0, min(maxv, yv))
                    dstvY = yv
                    if uv:
                        factorUV = F(F(stUV) * 0.0625)
                        # U
                        Uhmax = Uhmin = u[off - (2 + S)]
                        Uvmax = Uvmin = u[off - (2 + S) * pitch]
                        for i in range(-(1 + S), 3 + S):
                            hc, vc = u[off + i], u[off + i * pitch]
                            Uhmax = max(Uhmax, hc); Uhmin = min(Uhmin, hc)
                            Uvmax = max(Uvmax, vc); Uvmin = min(Uvmin, vc)
                        if Uhmax - Uhmin < Uvmax - Uvmin:
                            Uhmax, Uhmin = Uvmax, Uvmin
                        avgU = (Uhmax + Uhmin) >> 1
                        uout = u[off] + int(F(F(u[off] - avgU) * factorUV))
                        uout = max(Uhmin, min(Uhmax, uout)); uout = max(0, min(maxv, uout))
                        dstvU = uout
                        Vhmax = Vhmin = v[off - (2 + S)]
                        Vvmax = Vvmin = v[off - (2 + S) * pitch]
                        for i in range(-(1 + S), 3 + S):
                            hc, vc = v[off + i], v[off + i * pitch]
                            Vhmax = max(Vhmax, hc); Vhmin = min(Vhmin, hc)
                            Vvmax = max(Vvmax, vc); Vvmin = min(Vvmin, vc)
                        if Vhmax - Vhmin < Vvmax - Vvmin:
                            Vhmax, Vhmin = Vvmax, Vvmin
                        avgV = (Vhmax + Vhmin) >> 1
                        vout = v[off] + int(F(F(v[off] - avgV) * factorUV))
                        vout = max(Vhmin, min(Vhmax, vout)); vout = max(0, min(maxv, vout))
                        dstvV = vout
                else:
                    dstvY = srcvY
                    dstvU, dstvV = u[off], v[off]
            dY[off] = dstvY
            if uv:
                dU[off] = dstvU; dV[off] = dstvV
    if uv:
        return dY + dU + dV
    return dY


def repair_golden(width, height, pitch, N, el, src):
    # interior only (borders need a padded plane; kernel is rig-bound there)
    out = []
    for yy in range(1, height - 1):
        for xx in range(1, width - 1):
            off = yy * pitch + xx
            sv, ev = src[off], el[off]
            dv = sv
            if ev != sv:
                a = [src[(xx - 1) + (yy - 1) * pitch], src[xx + (yy - 1) * pitch],
                     src[(xx + 1) + (yy - 1) * pitch], src[(xx - 1) + yy * pitch],
                     src[(xx + 1) + yy * pitch], src[(xx - 1) + (yy + 1) * pitch],
                     src[xx + (yy + 1) * pitch], src[(xx + 1) + (yy + 1) * pitch]]
                a.sort()
                lo = min(sv, a[N - 1])
                hi = max(sv, a[8 - N])
                dv = max(lo, min(hi, ev))
            out.append(dv)
    return out


def to444_golden(width, height, sp, lx, ly, src):
    BW, BH = 1 << lx, 1 << ly
    DW, DH = BW * width, BH * height
    dst = [[0] * DW for _ in range(DH)]
    for yy in range(height):
        for xx in range(width):
            v00 = src[xx + yy * sp]
            dst[BH * yy + 0][BW * xx + 0] = v00
            if lx:
                v10 = src[(xx + 1) + yy * sp] if xx + 1 < width else v00
                dst[BH * yy + 0][BW * xx + 1] = (v00 + v10 + 1) >> 1
                if ly:
                    if yy + 1 < height:
                        v01 = src[xx + (yy + 1) * sp]
                        v11 = src[(xx + 1) + (yy + 1) * sp] if xx + 1 < width else v01
                    else:
                        v01 = v00
                        v11 = v10 if xx + 1 < width else v00
                    dst[BH * yy + 1][BW * xx + 0] = (v00 + v01 + 1) >> 1
                    dst[BH * yy + 1][BW * xx + 1] = (v00 + v10 + v01 + v11 + 2) >> 2
    return [dst[r][c] for r in range(DH) for c in range(DW)]


def from444_golden(width, height, sp, lx, ly, src):
    BW, BH = 1 << lx, 1 << ly
    return [src[BW * xx + BH * yy * sp] for yy in range(height) for xx in range(width)]


def main():
    subprocess.run(["g++", "-O2", "-std=gnu++17", "-Wl,--allow-multiple-definition", "-ffp-contract=off", "-w",
                    os.path.join(REPO, "sim", "kfm_edgelevel_ref.cpp"),
                    "-o", BIN], check=True)
    rng = random.Random(11)
    ok = True
    total = 0

    # --- E: edgelevel ---
    for _ in range(220):
        bits = rng.choice([8, 8, 16])
        maxv = 255 if bits == 8 else 65535
        width = rng.randint(8, 24)
        height = rng.randint(8, 24)
        pitch = width + rng.choice([0, 2])
        selective = rng.choice([0, 1])
        check = rng.choice([0, 1])
        uv = rng.choice([0, 1])
        st = rng.randint(1, 60)
        stUV = rng.randint(1, 60) if uv else 0
        thrs = rng.randint(1, max(2, maxv // 16))
        n = pitch * height
        src = [rng.randint(0, maxv) for _ in range(n)]
        u = [rng.randint(0, maxv) for _ in range(n)]
        v = [rng.randint(0, maxv) for _ in range(n)]
        # structured edges so the window spreads/thresholds are interesting
        for yy in range(height):
            base = rng.randint(0, maxv)
            for xx in range(width):
                s = base + (30 if (xx // 4) % 2 else 0) + ((yy * 7) % 9)
                src[yy * pitch + xx] = max(0, min(maxv, s))
        hdr = [ord('E'), width, height, pitch, maxv, st, stUV, thrs, check, selective, uv, n, n, n]
        got = run_mirror(hdr + src + u + v)
        exp = edgelevel_golden(width, height, pitch, maxv, st, stUV, thrs, check,
                               selective, uv, src, u, v)
        total += 1
        if got != exp:
            ok = False
            print("edgelevel MISMATCH", width, height, pitch, maxv, st, stUV, thrs,
                  "check", check, "sel", selective, "uv", uv)
            for i, (g, e) in enumerate(zip(got, exp)):
                if g != e:
                    print("  px", i, "got", g, "exp", e)
                    if i > 14: break
            if total >= 3: break

    # --- R: edgelevel_repair (compare interior region only) ---
    for _ in range(120):
        bits = rng.choice([8, 8, 16])
        maxv = 255 if bits == 8 else 65535
        width = rng.randint(8, 24)
        height = rng.randint(8, 24)
        pitch = width
        N = rng.randint(1, 4)
        n = pitch * height
        el = [rng.randint(0, maxv) for _ in range(n)]
        src = [rng.randint(0, maxv) for _ in range(n)]
        # force some el != src
        for i in range(0, n, 3):
            el[i] = (el[i] + 40) % (maxv + 1)
        hdr = [ord('R'), width, height, pitch, N, n, n]
        got = run_mirror(hdr + el + src)
        exp = repair_golden(width, height, pitch, N, el, src)
        # got is full plane; exp is interior. compare interior row-major
        got_int = []
        for yy in range(1, height - 1):
            for xx in range(1, width - 1):
                got_int.append(got[yy * width + xx])
        total += 1
        if got_int != exp:
            ok = False
            for g, e in zip(got_int, exp):
                if g != e:
                    print("repair MISMATCH", width, height, N, "got", g, "exp", e)
                    break
            if total >= 3: break

    # --- U: el_to444 ---
    for _ in range(80):
        bits = rng.choice([8, 8, 16])
        maxv = 255 if bits == 8 else 65535
        width = rng.randint(3, 16)
        height = rng.randint(3, 16)
        lx = rng.choice([0, 1])
        ly = rng.choice([0, 1]) if lx else 0
        sp = width
        n = sp * height
        src = [rng.randint(0, maxv) for _ in range(n)]
        BW, BH = 1 << lx, 1 << ly
        dp = BW * width
        hdr = [ord('U'), width, height, sp, dp, lx, ly, n]
        got = run_mirror(hdr + src)
        exp = to444_golden(width, height, sp, lx, ly, src)
        total += 1
        if got != exp:
            ok = False
            print("el_to444 MISMATCH", width, height, lx, ly)
            for i, (g, e) in enumerate(zip(got, exp)):
                if g != e:
                    print("  px", i, "got", g, "exp", e)
                    if i > 10: break
            if total >= 3: break

    # --- D: el_from444 ---
    for _ in range(80):
        bits = rng.choice([8, 8, 16])
        maxv = 255 if bits == 8 else 65535
        width = rng.randint(3, 16)
        height = rng.randint(3, 16)
        lx = rng.choice([0, 1])
        ly = rng.choice([0, 1]) if lx else 0
        BW, BH = 1 << lx, 1 << ly
        sp = BW * width
        n = sp * (BH * height)
        src = [rng.randint(0, maxv) for _ in range(n)]
        dp = width
        hdr = [ord('D'), width, height, sp, dp, lx, ly, n]
        got = run_mirror(hdr + src)
        exp = from444_golden(width, height, sp, lx, ly, src)
        total += 1
        if got != exp:
            ok = False
            print("el_from444 MISMATCH", width, height, lx, ly)
            for i, (g, e) in enumerate(zip(got, exp)):
                if g != e:
                    print("  px", i, "got", g, "exp", e)
                    if i > 10: break
            if total >= 3: break

    print(f"KFM KEdgeLevel (edgelevel/repair/to444/from444): "
          f"{'PASS' if ok else 'FAIL'} ({total} cases)")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
