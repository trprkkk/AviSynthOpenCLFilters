#!/usr/bin/env python3
"""Validate the degrain/compensate OVERLAP pixel-combiner core: the two .cl
kernels kt_degrain_patch + kt_overlap_out in src/opencl/ktgmc/kernels/
ktgmc_motion.cl against the CPU staging mirror sim/ktgmc_degrain_ref.cpp.

The mirror reproduces the MV.cpp CPU overlap path (KMDegrainCore::Proc: per
block Degrain1to6_C -> tmpBlock, Overlaps_C feathered window accumulation into
a global tmp, then Short2Bytes + edge source-copy).  The .cl kernels compute the
same result per output pixel by summing the (<= 2x2) covering blocks' window
terms.  This python golden is an independent transcription of the .cl
per-pixel formulation; matching it to the staging mirror bit-for-bit cross-checks
the two summation groupings and therefore both kernels' integer math.

The ref-plane element offsets (refBaseB/refBaseF) are treated as resolved inputs
(the host/rig seam from docs/MV_PORT_SPEC.md 6.1); here they are set so the
block reads its own block-origin region of a ref plane, exercising the full
u/v local addressing.

Run:  python3 python/run_mv_degrain.py
"""
import os, random, subprocess, tempfile, sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BIN = os.path.join(tempfile.gettempdir(), "degrain_ref")


def run_mirror(nums):
    inf = os.path.join(tempfile.gettempdir(), "dg_in.txt")
    outf = os.path.join(tempfile.gettempdir(), "dg_out.txt")
    with open(inf, "w") as f:
        f.write(" ".join(map(str, nums)))
    subprocess.run([BIN, inf, outf], check=True)
    with open(outf) as f:
        return [int(line) for line in f]


def gen_case(rng):
    """Build one case's geometry + arrays. Returns mirror token list and the
    python arrays needed by the per-pixel golden."""
    bits = rng.choice([8, 8, 16])
    maxv = 255 if bits == 8 else 65535
    nBlkSize = rng.choice([4, 8])
    overlap = rng.choice([0, nBlkSize // 2])
    step = nBlkSize - overlap
    nBlkX = rng.randint(3, 4)   # window formula needs nBlkX,nBlkY >= 3
    nBlkY = rng.randint(3, 4)
    delta = rng.randint(1, min(3, 6))
    # extra uncovered margin so edge copy is exercised
    ex = rng.randint(0, 3)
    ey = rng.randint(0, 3)
    W_B = nBlkX * step + overlap
    H_B = nBlkY * step + overlap
    width = W_B + ex
    height = H_B + ey
    shift = 5 if bits == 8 else 11
    nBlk = nBlkX * nBlkY
    srcPitch = width
    refFPitch = width
    refBPitch = width
    dstPitch = width
    winSize = nBlkSize * nBlkSize

    src = [rng.randint(0, maxv) for _ in range(srcPitch * height)]
    refF = [rng.randint(0, maxv) for _ in range(refFPitch * height)]
    refB = [rng.randint(0, maxv) for _ in range(refBPitch * height)]

    WSrc = [rng.randint(32, 300) for _ in range(nBlk)]
    WB = [rng.randint(0, 200) for _ in range(delta * nBlk)]
    WF = [rng.randint(0, 200) for _ in range(delta * nBlk)]
    # base = block origin region of the plane (resolved seam, host supplies)
    baseB = []
    baseF = []
    for by in range(nBlkY):
        for bx in range(nBlkX):
            for _k in range(delta):
                baseB.append(bx * step + (by * step) * width)
                baseF.append(bx * step + (by * step) * width)
    win = [rng.randint(0, 300) for _ in range(9 * winSize)]

    tokens = [nBlkX, nBlkY, nBlkSize, step, step, overlap, overlap, delta,
              width, height, maxv, shift,
              srcPitch, refFPitch, refBPitch, dstPitch, winSize,
              srcPitch * height, refFPitch * height, refBPitch * height]
    for a in (WSrc, WF, WB, baseF, baseB, win, refF, refB, src):
        tokens += a
    return dict(bits=bits, maxv=maxv, nBlkSize=nBlkSize, step=step,
                overlap=overlap, nBlkX=nBlkX, nBlkY=nBlkY, delta=delta,
                width=width, height=height, W_B=W_B, H_B=H_B, shift=shift,
                srcPitch=srcPitch, winSize=winSize, tokens=tokens,
                src=src, refF=refF, refB=refB, WSrc=WSrc, WF=WF, WB=WB,
                baseB=baseB, baseF=baseF, win=win)


def golden(c):
    """per-pixel (.cl-equivalent) formulation."""
    nBlkSize = c["nBlkSize"]; step = c["step"]; nBlkX = c["nBlkX"]
    nBlkY = c["nBlkY"]; delta = c["delta"]; width = c["width"]
    height = c["height"]; W_B = c["W_B"]; H_B = c["H_B"]
    shift = c["shift"]; sp = c["srcPitch"]; nBlk = nBlkX * nBlkY
    maxv = c["maxv"]; winSize = c["winSize"]
    src = c["src"]; refF = c["refF"]; refB = c["refB"]
    WSrc = c["WSrc"]; WF = c["WF"]; WB = c["WB"]
    baseB = c["baseB"]; baseF = c["baseF"]; win = c["win"]

    def dg(blk, u, v):
        bx = blk % nBlkX; by = blk // nBlkX
        val = src[(by * step + v) * sp + (bx * step + u)] * WSrc[blk]
        for k in range(delta):
            o = k * nBlk + blk
            val += refB[baseB[o] + u + v * sp] * WB[o]
            val += refF[baseF[o] + u + v * sp] * WF[o]
        return (val + (0 if shift == 11 else 128)) >> 8


    out = []
    for y in range(height):
        for x in range(width):
            if x >= W_B or y >= H_B:
                out.append(src[y * sp + x])
                continue
            tmp = 0
            byc = y // step
            for dby in range(3):
                by = byc - dby
                if by < 0 or by >= nBlkY: continue
                if by * step > y or y >= by * step + nBlkSize: continue
                v = y - by * step
                wby = 3 * ((by + nBlkY - 3) // (nBlkY - 2))
                bxc = x // step
                for dbx in range(3):
                    bx = bxc - dbx
                    if bx < 0 or bx >= nBlkX: continue
                    if bx * step > x or x >= bx * step + nBlkSize: continue
                    u = x - bx * step
                    wbx = (bx + nBlkX - 3) // (nBlkX - 2)
                    blk = bx + by * nBlkX
                    pv = dg(blk, u, v)
                    w = win[(wby + wbx) * winSize + v * nBlkSize + u]
                    tmp += (pv * w) if shift == 11 else ((pv * w + 256) >> 6)
            a = tmp >> shift
            out.append(max(0, min(maxv, a)))
    return out


def main():
    subprocess.run(["g++", "-O2", "-std=gnu++17", "-Wl,--allow-multiple-definition", "-w",
                    os.path.join(REPO, "sim", "ktgmc_degrain_ref.cpp"),
                    "-o", BIN], check=True)
    rng = random.Random(41)
    ok = True
    total = 0
    for _ in range(300):
        c = gen_case(rng)
        got = run_mirror(c["tokens"])
        exp = golden(c)
        total += 1
        if got != exp:
            ok = False
            print("degrain MISMATCH", {k: c[k] for k in
                  ("nBlkX", "nBlkY", "nBlkSize", "overlap", "delta",
                   "width", "height", "shift")})
            for i, (g, e) in enumerate(zip(got, exp)):
                if g != e:
                    print("  px", i, "got", g, "exp", e)
                    if i > 10: break
            if total >= 3:
                break
    print(f"degrain/compensate overlap combiner: {'PASS' if ok else 'FAIL'} "
          f"({total} cases)")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
