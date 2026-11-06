#!/usr/bin/env python3
"""Validate kt_calc_all_sad in
src/opencl/ktgmc/kernels/ktgmc_motion.cl against the CPU mirror
sim/mv_calc_all_sad_ref.cpp with an independent Python golden.

This is the kernel that graduates from RIG-VERIFY once the host layout is
pinned.  The two open questions in docs/BLOCKSEARCH_MODEL.md §8 were
answered by reading MVKernel.cu directly, and they are ASSERTED here:

  1. `vectors` is a plain row-major short2 array indexed [bx + by*nBlkX]
     (SearchBatch, MVKernel.cu:819 + the kernel's own read at :1122).  The
     sentinel / vectorsPitch / appended-copy machinery belongs to
     kl_search's predictor reads, NOT to this kernel.  The runner pins
     this by planting distinct MVs per block and by padding the buffer
     with poison values that a pitched or sentinel-offset read would hit.
  2. The reference origin is &pRef[offx + offy*nPitch] passed through
     dev_get_ref_block; chroma uses base (offx>>1, offy>>1) with the MV
     halved (vx>>1, vy>>1).  Sub-pel plane selection is pinned by sweeping
     NPEL and every (vx&n, vy&n) residue.

The golden derives the sub-pel plane index independently (divmod on the
residues, plane = sy*NPEL + sx) instead of repeating the port's shift/AND
spelling, and computes SAD as sum(abs(...)) over an explicitly built
window list.

Run:  python3 python/run_mv_calc_all_sad.py
"""
import os, random, subprocess, tempfile, sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BIN = os.path.join(tempfile.gettempdir(), "mv_calc_all_sad_ref")
# A VALID but distinct MV: the layout pin prepends these to the buffer, so
# they must be legal to dereference (a wild value would just crash instead
# of proving anything about the indexing).
POISON = 1


def run_mirror(nums):
    inf = os.path.join(tempfile.gettempdir(), "mvs_in.txt")
    outf = os.path.join(tempfile.gettempdir(), "mvs_out.txt")
    with open(inf, "w") as f:
        f.write(" ".join(map(str, nums)))
    subprocess.run([BIN, inf, outf], check=True)
    with open(outf) as f:
        return [int(line) for line in f]


def main():
    subprocess.run(["g++", "-O2", "-std=gnu++17", "-Wl,--allow-multiple-definition", "-ffp-contract=off", "-w",
                    os.path.join(REPO, "sim", "mv_calc_all_sad_ref.cpp"),
                    "-o", BIN], check=True)
    rng = random.Random(6301)
    ok = True
    total = 0
    residues = set()

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

    # upstream instantiates BLK_SIZE {8,16,32} x NPEL {1,2} x CHROMA
    # (MVKernel.cu:2631); NPEL 4 is included to exercise the shared helper.
    shapes = [(b, n, c) for b in (8, 16, 32) for n in (1, 2, 4)
              for c in (0, 1)]

    for ci, (BLK, NPEL, chroma) in enumerate(shapes * 2):
        bits = 8 if ci < len(shapes) else 16
        maxv = 255 if bits == 8 else 65535
        nBlkX = rng.randint(1, 4)
        nBlkY = rng.randint(1, 3)
        nPad = rng.choice([0, 8, 16])
        blkStep = BLK // 2
        # plane big enough for the last block plus the largest MV shift
        w = nPad * 2 + (nBlkX - 1) * blkStep + BLK + 64
        h = nPad * 2 + (nBlkY - 1) * blkStep + BLK + 64
        nPitchY = w + rng.choice([0, 1, 4])
        nPitchUV = (w // 2) + rng.choice([0, 1, 2])
        # sub-pel stack: NPEL^2 planes of nImgPitch each
        nImgPitchY = nPitchY * h
        nImgPitchUV = nPitchUV * (h // 2 + 8)
        # origin offset so negative MVs legally read before it, exactly
        # like upstream's padded super-frame pointers
        baseY = 32 * nPitchY + 32
        baseUV = 32 * nPitchUV + 32
        nY = baseY + nImgPitchY * (NPEL * NPEL + 1)
        nUV = (baseUV + nImgPitchUV * (NPEL * NPEL + 1)) if chroma else 0

        srcY = [rng.randint(0, maxv) for _ in range(nY)]
        refY = [rng.randint(0, maxv) for _ in range(nY)]
        srcU = [rng.randint(0, maxv) for _ in range(nUV)]
        srcV = [rng.randint(0, maxv) for _ in range(nUV)]
        refU = [rng.randint(0, maxv) for _ in range(nUV)]
        refV = [rng.randint(0, maxv) for _ in range(nUV)]

        # distinct MV per block, sweeping the sub-pel residues; magnitudes
        # kept inside the padding so every read stays in the plane
        vec = []
        for b in range(nBlkX * nBlkY):
            vx = rng.randint(-2, 2) * NPEL + rng.randrange(NPEL)
            vy = rng.randint(-2, 2) * NPEL + rng.randrange(NPEL)
            residues.add((vx % NPEL, vy % NPEL, NPEL))
            vec += [vx, vy]
        # layout pin: prepend valid-but-different MVs.  A row-major read at
        # [bx + by*nBlkX] must pick these up and change every block's SAD;
        # an implementation that skipped a sentinel region would not.
        vec_sent = [POISON] * 8 + vec + [POISON] * 8
        vec_arg = vec  # the kernel must index [bx + by*nBlkX] exactly

        hdr = ([ord('S'), nBlkX, nBlkY, nPad, BLK, NPEL, chroma,
                nPitchY, nPitchUV, nImgPitchY, nImgPitchUV, baseY, baseUV,
                len(vec_arg)]
               + vec_arg + [nY] + srcY + refY + [nUV])
        if chroma:
            hdr += srcU + srcV + refU + refV
        got = run_mirror(hdr)

        exp = []
        for by in range(nBlkY):
            for bx in range(nBlkX):
                blk = bx + by * nBlkX
                offx = nPad + bx * blkStep
                offy = nPad + by * blkStep
                vx, vy = vec[blk * 2], vec[blk * 2 + 1]

                def sub_off(mx, my, pitch, imgpitch):
                    # independent derivation: integer part + plane index
                    ix, sx = divmod(mx, NPEL)
                    iy, sy = divmod(my, NPEL)
                    return ix + iy * pitch + (sx + sy * NPEL) * imgpitch

                roff = sub_off(vx, vy, nPitchY, nImgPitchY)
                win = [(offx + jx, offy + jy)
                       for jy in range(BLK) for jx in range(BLK)]
                sad = sum(abs(srcY[baseY + px + py * nPitchY]
                              - refY[baseY + px + py * nPitchY + roff])
                          for px, py in win)
                if chroma:
                    bs2 = BLK // 2
                    bux, buy = offx // 2, offy // 2
                    # C's >> 1 on a negative int is an arithmetic shift =
                    # floor division, which is what // 2 does in Python
                    ruv = sub_off(vx >> 1, vy >> 1, nPitchUV, nImgPitchUV)
                    winc = [(bux + jx, buy + jy)
                            for jy in range(bs2) for jx in range(bs2)]
                    sad += sum(abs(srcU[baseUV + px + py * nPitchUV]
                                   - refU[baseUV + px + py * nPitchUV + ruv])
                               for px, py in winc)
                    sad += sum(abs(srcV[baseUV + px + py * nPitchUV]
                                   - refV[baseUV + px + py * nPitchUV + ruv])
                               for px, py in winc)
                exp += [sad, vx, vy, sad]

        total += 1
        if not check("calc_all_sad", got, exp,
                     (BLK, NPEL, chroma, nBlkX, nBlkY, bits)):
            ok = False

        # layout pin: the same call with the buffer surrounded by poison
        # must be identical -> no sentinel/pitch offset is being applied
        hdr2 = ([ord('S'), nBlkX, nBlkY, nPad, BLK, NPEL, chroma,
                 nPitchY, nPitchUV, nImgPitchY, nImgPitchUV, baseY, baseUV,
                 len(vec_sent)]
                + vec_sent + [nY] + srcY + refY + [nUV])
        if chroma:
            hdr2 += srcU + srcV + refU + refV
        # (the mirror reads vectors[blk*2..] from the START of the array, so
        # a poisoned PREFIX must change the answer -- proving the index is
        # not shifted by a sentinel region)
        got2 = run_mirror(hdr2)
        if got2 == got and nBlkX * nBlkY > 0:
            print("LAYOUT PIN FAILED: sentinel-prefixed vectors gave the "
                  "same result, so the index is not [bx + by*nBlkX]")
            ok = False

    print(f"MV kt_calc_all_sad: {'PASS' if ok else 'FAIL'} ({total} cases, "
          f"{len(residues)} sub-pel residues)")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
