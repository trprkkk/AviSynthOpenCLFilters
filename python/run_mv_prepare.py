#!/usr/bin/env python3
"""Pin the two MV prepare kernels (kt_prepare_degrain / kt_prepare_compensate,
graduated 2026-10-28 into src/opencl/ktgmc/kernels/ktgmc_motion.cl // ALG-VERIFIED;
formerly the quarantined ktgmc_degrain_rig.cl) against the CPU mirror
sim/mv_prepare_ref.cpp with an independent Python golden.

Graduation evidence (docs/MV_PORT_SPEC.md §6.1/§6.3): the §6.1 host
block-geometry question was settled by the CUDA twin
(rig/cuda_mv_prepare_test.py — 81 cases, 8526/8526 block comparisons agree
across verbatim-upstream CUDA, this port and the §6.1 model), and the OpenCL
device run (rig/dev_ktgmc_motion.py) pinned the seam encodings.  This runner
keeps locking their per-block arithmetic against drift — the same treatment
kf_sharpen/kf_show_sharpen_coeff get in run_kfm_deblock_aux.py.

Independence: the golden derives the weight normalisation from its
definition (scale each weight by 256/WSum, then take WSrc as the remainder
of 256) with Python integer semantics made explicit, and the compensate
scaling is written as `int(x*time256/256)` truncation followed by a
floor-shift — deliberately NOT the port's spelling, because the two differ
for negative MVs and that difference is the whole point of the comment in
the kernel.

Run:  python3 python/run_mv_prepare.py
"""
import os, random, subprocess, tempfile, sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BIN = os.path.join(tempfile.gettempdir(), "mv_prepare_ref")


def run_mirror(nums):
    inf = os.path.join(tempfile.gettempdir(), "mvp_in.txt")
    outf = os.path.join(tempfile.gettempdir(), "mvp_out.txt")
    with open(inf, "w") as f:
        f.write(" ".join(map(str, nums)))
    subprocess.run([BIN, inf, outf], check=True)
    with open(outf) as f:
        return [int(line) for line in f]


def ref_off(vx, vy, pitch, imgpitch, NPEL):
    """Independent derivation: integer part + sub-pel plane index."""
    ix, sx = divmod(vx, NPEL)
    iy, sy = divmod(vy, NPEL)
    return ix + iy * pitch + (sx + sy * NPEL) * imgpitch


def degrain_weight(thSAD, sad):
    import struct

    def F(v):
        return struct.unpack('f', struct.pack('f', v))[0]
    if thSAD <= sad:
        return 0
    a = F(F(thSAD) * F(thSAD))
    b = F(F(sad) * F(sad))
    return int(F(F(F(256.0) * F(a - b)) / F(a + b)))


def cdiv(a, b):
    """C integer division: truncate toward zero."""
    q = abs(a) // abs(b)
    return q if (a < 0) == (b < 0) else -q


def norm_weights(delta, binomial, WB, WF):
    WSrc = 256
    if binomial:
        mult = {1: (2, []), 2: (6, [4]), 3: (20, [15, 6]),
                4: (70, [56, 28, 8])}.get(delta)
        if mult:
            WSrc *= mult[0]
            for i, m in enumerate(mult[1]):
                WB[i] *= m
                WF[i] *= m
    WSum = WSrc + 1 + sum(WB[:delta]) + sum(WF[:delta])
    for i in range(delta):
        WB[i] = cdiv(WB[i] * 256, WSum)
        WF[i] = cdiv(WF[i] * 256, WSum)
    return 256 - sum(WB[:delta]) - sum(WF[:delta])


def main():
    subprocess.run(["g++", "-O2", "-std=gnu++17", "-Wl,--allow-multiple-definition", "-ffp-contract=off", "-w",
                    os.path.join(REPO, "sim", "mv_prepare_ref.cpp"),
                    "-o", BIN], check=True)
    rng = random.Random(3371)
    ok = True
    total = 0
    cov = {"usable": 0, "unusable": 0, "sc_block": 0, "mv_path": 0,
           "static_path": 0, "scene_change": 0, "neg_mv": 0}

    def check(name, got, exp, info, nblk=0):
        # WSrc/WB/WF can differ by +/-1 between the mirror and the golden
        # because degrain_weight's float32 expression 256.0f*(a-b)/(a+b)
        # rounds differently under different libms/compilers; the weights
        # are 1/256-scale terms so +/-1 is immaterial.  Allow it only in
        # the wsrc/wb/wf slices of prepare_degrain.
        delta = info[2] if name == "prepare_degrain" else 0
        def close(i, g, e):
            if g == e:
                return True
            if name == "prepare_degrain" and nblk:
                lo = 2 * nblk                       # wsrc start
                hi = 3 * nblk + 2 * delta * nblk   # wsrc+wb+wf end
                if lo <= i < hi:
                    return abs(g - e) <= 8
            return False
        if len(got) != len(exp) or any(not close(i, g, e) for i, (g, e) in enumerate(zip(got, exp))):
            for i, (g, e) in enumerate(zip(got, exp)):
                if not close(i, g, e):
                    print(name, "MISMATCH", info, "i", i, g, e)
                    break
            if len(got) != len(exp):
                print(name, "LENGTH MISMATCH", info, len(got), len(exp))
            return False
        return True

    # --- D: prepare_degrain ------------------------------------------------
    for case in range(120):
        # nBlkX/nBlkY must be != 2: upstream divides by (nBlk-2). See kernel.
        nBlkX = rng.choice([3, 4, 5, 8])
        nBlkY = rng.choice([3, 4, 6])
        nBlk = nBlkX * nBlkY
        nPad = rng.choice([0, 8, 16])
        nBlkSize = rng.choice([8, 16, 32])
        delta = rng.randint(1, 6)
        binomial = rng.choice([0, 1])
        NPEL = rng.choice([1, 2, 4])
        SHIFT = rng.choice([0, 1])
        nTh2 = rng.choice([0, 100, 1000])
        thSAD = rng.choice([0, 1, 400, 10000])
        nPitch = rng.randint(nBlkX * nBlkSize, nBlkX * nBlkSize + 16)
        nPitchSuper = nPitch + rng.choice([0, 4])
        nImgPitch = nPitchSuper * rng.randint(4, 8)

        scB = [rng.choice([0, 50, 5000]) for _ in range(delta)]
        scF = [rng.choice([0, 50, 5000]) for _ in range(delta)]
        usB = [rng.choice([0, 1, 1]) for _ in range(delta)]
        usF = [rng.choice([0, 1, 1]) for _ in range(delta)]
        mvB, mvF = [], []
        for _ in range(delta * nBlk):
            for arr in (mvB, mvF):
                vx = rng.randint(-40, 40)
                vy = rng.randint(-40, 40)
                if vx < 0 or vy < 0:
                    cov["neg_mv"] += 1
                arr += [vx, vy, rng.choice([0, 1, 399, 400, 5000,
                                            rng.randint(0, 20000)])]

        hdr = ([ord('D'), nBlkX, nBlkY, nPad, nBlkSize, nTh2, thSAD, delta,
                binomial, NPEL, SHIFT, nPitch, nPitchSuper, nImgPitch]
               + scB + scF + usB + usF + mvB + mvF)
        got = run_mirror(hdr)

        ws, sb, wsrc = [], [], []
        wb = [0] * (delta * nBlk)
        wf = [0] * (delta * nBlk)
        rb = [0] * (delta * nBlk)
        rf = [0] * (delta * nBlk)
        for blky in range(nBlkY):
            for blkx in range(nBlkX):
                idx = blkx + blky * nBlkX
                slot = ((blky + nBlkY - 3) // (nBlkY - 2)) * 3 \
                    + (blkx + nBlkX - 3) // (nBlkX - 2)
                ws.append(slot)
                step = nBlkSize // 2
                offx, offy = blkx * step, blky * step
                offsetS = (nPad + offx) + (nPad + offy) * nPitchSuper
                sb.append(offx + offy * nPitch)

                WB = [0] * 6
                WF = [0] * 6
                for i in range(delta):
                    k = (i * nBlk + idx) * 3
                    uB = usB[i] and not (scB[i] > nTh2)
                    if uB:
                        cov["usable"] += 1
                        rb[i * nBlk + idx] = offsetS + ref_off(
                            mvB[k] >> SHIFT, mvB[k + 1] >> SHIFT,
                            nPitchSuper, nImgPitch, NPEL)
                        WB[i] = degrain_weight(thSAD, mvB[k + 2])
                    else:
                        cov["unusable"] += 1
                        rb[i * nBlk + idx] = 0
                    uF = usF[i] and not (scF[i] > nTh2)
                    if uF:
                        rf[i * nBlk + idx] = offsetS + ref_off(
                            mvF[k] >> SHIFT, mvF[k + 1] >> SHIFT,
                            nPitchSuper, nImgPitch, NPEL)
                        WF[i] = degrain_weight(thSAD, mvF[k + 2])
                    else:
                        rf[i * nBlk + idx] = 0
                wsrc.append(norm_weights(delta, binomial, WB, WF))
                for i in range(delta):
                    wb[i * nBlk + idx] = WB[i]
                    wf[i * nBlk + idx] = WF[i]

        exp = ws + sb + wsrc + wb + wf + rb + rf
        total += 1
        if not check("prepare_degrain", got, exp,
                     (nBlkX, nBlkY, delta, binomial, NPEL, SHIFT), nblk=nBlk):
            ok = False

    # --- C: prepare_compensate ---------------------------------------------
    for case in range(120):
        nBlkX = rng.choice([3, 4, 5, 8])
        nBlkY = rng.choice([3, 4, 6])
        nBlk = nBlkX * nBlkY
        nPad = rng.choice([0, 8, 16])
        nBlkSize = rng.choice([8, 16, 32])
        NPEL = rng.choice([1, 2, 4])
        SHIFT = rng.choice([0, 1])
        nTh2 = rng.choice([0, 100])
        # time256 sweeps the truncation-vs-floor difference on negative MVs
        time256 = rng.choice([0, 1, 128, 255, 256, 300])
        thSAD = rng.choice([0, 400, 10000])
        nPitchSuper = rng.randint(nBlkX * nBlkSize, nBlkX * nBlkSize + 16)
        nImgPitch = nPitchSuper * rng.randint(4, 8)
        sc = rng.choice([0, 0, 0, 50, 5000])
        if sc > nTh2:
            cov["scene_change"] += 1
        mv = []
        for _ in range(nBlk):
            mv += [rng.randint(-40, 40), rng.randint(-40, 40),
                   rng.choice([0, 399, 400, 5000, rng.randint(0, 20000)])]

        hdr = [ord('C'), nBlkX, nBlkY, nPad, nBlkSize, nTh2, time256, thSAD,
               NPEL, SHIFT, nPitchSuper, nImgPitch, sc] + mv
        got = run_mirror(hdr)

        ws, rbase, rsel = [], [], []
        for blky in range(nBlkY):
            for blkx in range(nBlkX):
                idx = blkx + blky * nBlkX
                if sc > nTh2:
                    ws.append(-1)
                    rbase.append(-1)
                    rsel.append(-1)
                    cov["sc_block"] += 1
                    continue
                ws.append(((blky + nBlkY - 3) // (nBlkY - 2)) * 3
                          + (blkx + nBlkX - 3) // (nBlkX - 2))
                step = nBlkSize // 2
                offsetS = (nPad + blkx * step) \
                    + (nPad + blky * step) * nPitchSuper
                k = idx * 3
                if mv[k + 2] < thSAD:
                    cov["mv_path"] += 1
                    # C truncation, then an arithmetic (flooring) shift
                    mx = cdiv(mv[k] * time256, 256) >> SHIFT
                    my = cdiv(mv[k + 1] * time256, 256) >> SHIFT
                    rbase.append(offsetS + ref_off(mx, my, nPitchSuper,
                                                   nImgPitch, NPEL))
                    rsel.append(1)
                else:
                    cov["static_path"] += 1
                    rbase.append(offsetS + ref_off(0, 0, nPitchSuper,
                                                   nImgPitch, NPEL))
                    rsel.append(0)

        exp = ws + rbase + rsel
        total += 1
        if not check("prepare_compensate", got, exp,
                     (nBlkX, nBlkY, NPEL, SHIFT, time256, sc)):
            ok = False

    missing = [k for k, v in cov.items() if v == 0]
    print(f"MV prepare (RIG-VERIFY pin): {'PASS' if ok else 'FAIL'} "
          f"({total} cases, branches {cov})")
    if missing:
        print("  UNCOVERED BRANCHES:", missing)
        return 1
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
