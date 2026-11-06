#!/usr/bin/env python3
"""Validate the KTGMC block-search E2E port (kt_search_column in
src/opencl/ktgmc/kernels/ktgmc_search.cl, with the kt_prepare_search setup
in ktgmc_motion.cl) against the CPU mirror sim/ktgmc_search_ref.cpp with an
independent Python golden.

This closes docs/BLOCKSEARCH_MODEL.md §8b/§8c: the predictor-slot layout and
the prog/next work-stealing question were settled empirically by the CUDA
twin (cuda_search/SEARCH_MEASURE_REPORT.md — 9 configs, scalar model mism=0
at order0 AND order1).  The pinned semantics this golden enforces:

  * vectors layout: pitch 4+2*nBlkCount per batch; slot -2 = zero sentinel
    (0,0), slot -1 = global sentinel = globalMV*nPel (short2), slots 0..n-1
    own region (fresh), slots n..2n-1 copy region (pre-search snapshot,
    read ONLY by the bottom-right predictor);
  * REF_VECTOR_INDEX (ANALYZE_SYNC=1, shipped): left = blkIdx-1 OWN region
    FRESH, up = blkIdx-nBlkX OWN region FRESH (left slot when by==0), BR =
    blkIdx+nBlkX+1+nBlkCount COPY region (stale);
  * CPU_EMU=true predictor placement: raw slots 0..5 = zero, global, own,
    left, up, BR; median candidate placed at slot 3 (dst remap i>=3 -> i+1);
  * cost init (LAMBDA*sq_norm)>>8; candidate SADs: bx<3 REPLACE the cost
    (sad + (sad*PENALTIES[bx])>>8), bx>=3 ADD the SAD;
  * serial reduce, earlier index wins ties;
  * refine: prune iff lambda-cost >= stage-ENTRY best; survivors get
    cost += sad + ((sad*PENALTY_NEW)>>8); pruned keep LARGE_COST; the stage
    update is CONDITIONAL; EXHAUSTIVE expand1/expand2 share the pre-expand1
    centre, HEX chains hex2 -> expand1;
  * ordered dispatch (order0, column-major ascending) — measured
    result-bearing only through "fresh left/up before use".

The golden derives the sub-pel plane index independently (divmod) and the
search serially from the documented contract, NOT from the .cl spelling.

Run:  python3 python/run_mv_search.py
"""
import os, random, subprocess, sys, tempfile

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BIN = os.path.join(tempfile.gettempdir(), "ktgmc_search_ref")
LARGE = 2147483647  # INT_MAX
COV = {}

E1 = [(0, -1), (0, 1), (-1, 0), (1, 0), (-1, -1), (-1, 1), (1, -1), (1, 1)]
E2 = [(-1, -2), (-1, 2), (0, -2), (0, 2), (1, -2), (1, 2),
      (-2, -1), (2, -1), (-2, 0), (2, 0), (-2, 1), (2, 1),
      (-2, -2), (-2, 2), (2, -2), (2, 2)]
HEX = [(-2, 0), (-1, 2), (1, 2), (2, 0), (1, -2), (-1, -2)]


def cdiv(a, b):
    q = abs(a) // abs(b)
    return q if (a >= 0) == (b >= 0) else -q


def sub_off(vx, vy, pitch, imgpitch, NPEL):
    """Independent sub-pel offset: integer part + plane index (divmod)."""
    ix, sx = divmod(vx, NPEL)
    iy, sy = divmod(vy, NPEL)
    return ix + iy * pitch + (sx + sy * NPEL) * imgpitch


def golden_prepare(par, own, sads, gx, gy):
    """kl_prepare_search host-equivalent (ANALYZE_SYNC=1) + sentinels."""
    (nBlkX, nBlkY, BS, NPEL, nPad, nBlkSizeOvr, nLogScale, nExtW, nExtH,
     nLambdaLevel, lsad, pZero, pGlobal, pNew) = par
    n = nBlkX * nBlkY
    nps = nPad >> nLogScale
    zero = (0, 0)
    glob = (int(gx * NPEL), int(gy * NPEL))
    blocks = []
    for by in range(nBlkY):
        for bx in range(nBlkX):
            blk = bx + by * nBlkX
            X = nPad + nBlkSizeOvr * bx
            Y = nPad + nBlkSizeOvr * by
            d = [NPEL * (nExtW - X - BS - nPad + nps) - 1,
                 NPEL * (nExtH - Y - BS - nPad + nps) - 1,
                 -NPEL * (X - nPad + nps),
                 -NPEL * (Y - nPad + nps)]
            p1 = blk - 1 if bx > 0 else -2          # own region, FRESH
            p2 = blk - nBlkX if by > 0 else p1
            p3 = blk + nBlkX + 1 + n if (by < nBlkY - 1 and bx < nBlkX - 1) else -2
            d += [-2, -1, blk, p1, p2, p3, own[2 * blk], own[2 * blk + 1]]
            lam = cdiv(cdiv(nLambdaLevel * lsad, lsad + (sads[blk] >> 1)) * lsad,
                       lsad + (sads[blk] >> 1))
            if by == 0:
                lam = 0
            f = [pZero, pGlobal, 0, pNew, lam]
            blocks.append((d, f))
    return blocks, zero, glob


def golden_search(par, blocks, zero, glob, own, copy, planes, search):
    """Ordered (column-major) search; returns per-block (x, y, sad)."""
    (nBlkX, nBlkY, BS, NPEL, nPad, nBlkSizeOvr, nLogScale, nExtW, nExtH,
     nLambdaLevel, lsad, pZero, pGlobal, pNew) = par
    n = nBlkX * nBlkY
    srcY, refY, srcU, refV_, refU, srcV, pY, pUV, ipY, ipUV = planes
    chroma = srcU is not None
    vec = list(own) + list(copy)          # own fresh [0..n), copy stale [n..2n)

    def resolve(idx):
        if idx == -2:
            return zero
        if idx == -1:
            return glob
        return (vec[2 * idx], vec[2 * idx + 1])

    def sad_at(offx, offy, vx, vy):
        roff = sub_off(vx, vy, pY, ipY, NPEL)
        sad = 0
        for jy in range(BS):
            for jx in range(BS):
                sad += abs(srcY[(offx + jx) + (offy + jy) * pY]
                           - refY[(offx + jx) + (offy + jy) * pY + roff])
        if chroma:
            bs2 = BS // 2
            bux, buy = offx >> 1, offy >> 1
            ruv = sub_off(vx >> 1, vy >> 1, pUV, ipUV, NPEL)
            for jy in range(bs2):
                for jx in range(bs2):
                    sad += abs(srcU[(bux + jx) + (buy + jy) * pUV]
                               - refU[(bux + jx) + (buy + jy) * pUV + ruv])
                    sad += abs(srcV[(bux + jx) + (buy + jy) * pUV]
                               - refV_[(bux + jx) + (buy + jy) * pUV + ruv])
        return sad

    res_blk = [None] * n
    for blkx in range(nBlkX):
        for blky in range(nBlkY):
            blk = blky * nBlkX + blkx
            d, f = blocks[blk]
            offx = nPad + blkx * (BS // 2)
            offy = nPad + blky * (BS // 2)

            pred = []
            for k in range(6):
                x, y = resolve(d[4 + k])
                x = d[0] if x > d[0] else (d[2] if x < d[2] else x)
                y = d[1] if y > d[1] else (d[3] if y < d[3] else y)
                pred.append((x, y))
            # median candidate at slot 3 from raw slots 3..5 (CPU_EMU remap:
            # raw k>=3 -> dst k+1, median from px[4],px[5],px[6])
            mx = min(max(min(pred[3][0], pred[4][0]), pred[5][0]),
                     max(pred[3][0], pred[4][0]))
            my = min(max(min(pred[3][1], pred[4][1]), pred[5][1]),
                     max(pred[3][1], pred[4][1]))
            pred = pred[:3] + [(mx, my)] + pred[3:]

            r = []
            for i in range(7):
                x, y = pred[i]
                cost = (f[4] * ((x - d[10]) ** 2 + (y - d[11]) ** 2)) >> 8
                r.append([cost, x, y])
            for i in range(7):
                sad = sad_at(offx, offy, r[i][1], r[i][2])
                if i < 3:
                    r[i][0] = sad + ((sad * f[i]) >> 8)
                else:
                    r[i][0] += sad
            for i in range(1, 7):
                if r[0][0] > r[i][0]:
                    r[0] = r[i]
            best = (r[0][0], r[0][1], r[0][2])

            def refine(cx, cy, area):
                nonlocal best
                rr = []
                for dx, dy in area:
                    x, y = cx + dx, cy + dy
                    ok = (x <= d[0] and y <= d[1] and x >= d[2] and y >= d[3])
                    cost = (f[4] * ((x - d[10]) ** 2 + (y - d[11]) ** 2)) >> 8
                    if cost >= best[0]:
                        ok = False
                        COV["pruned"] += 1
                    rr.append([cost if ok else LARGE, x, y])
                any_live = False
                for e in rr:
                    if e[0] == LARGE:
                        continue
                    any_live = True
                    s = sad_at(offx, offy, e[1], e[2])
                    e[0] = e[0] + s + ((s * f[3]) >> 8)
                if not any_live:
                    COV["all_pruned"] += 1
                for i in range(1, len(rr)):
                    if rr[0][0] > rr[i][0]:
                        rr[0] = rr[i]
                if rr[0][0] < best[0]:
                    best = (rr[0][0], rr[0][1], rr[0][2])
                    COV["stage_update"] += 1

            if search == 1:
                bmx, bmy = best[1], best[2]      # snapshot
                refine(bmx, bmy, E1)
                refine(bmx, bmy, E2)             # SAME pre-expand1 centre
            else:
                refine(best[1], best[2], HEX)
                refine(best[1], best[2], E1)     # chained off hex2
            vec[2 * blk], vec[2 * blk + 1] = best[1], best[2]
            sad = sad_at(offx, offy, best[1], best[2])
            res_blk[blk] = (best[1], best[2], sad)
    # output in block (row-major) order, matching the mirror/kernel layout
    return [res_blk[b] for b in range(n)]


def run_mirror(nums):
    inf = os.path.join(tempfile.gettempdir(), "ksearch_in.txt")
    outf = os.path.join(tempfile.gettempdir(), "ksearch_out.txt")
    with open(inf, "w") as fh:
        fh.write(" ".join(map(str, nums)))
    subprocess.run([BIN, inf, outf], check=True)
    with open(outf) as fh:
        return [int(line) for line in fh]


def make_case(rng, BS, SEARCH, NPEL, chroma, flat, nBlkX, nBlkY, batch):
    """Build one case: planes, incoming own/copy regions, sads, params."""
    step = BS // 2
    nPad = rng.choice([4, 8, 16])
    nBlkSizeOvr = step
    nLogScale = 0
    # extended frame sized so the real block grid fits with pad on both sides
    nExtW = nPad + (nBlkX - 1) * step + BS + nPad + rng.choice([0, 3])
    nExtH = nPad + (nBlkY - 1) * step + BS + nPad + rng.choice([0, 3])
    # generous plane so every clipped-MV read stays in bounds
    slack = 2 * nPad * NPEL + 8
    W = (nExtW + 2 * nPad) * NPEL + slack
    H = (nExtH + 2 * nPad) * NPEL + slack
    pY = W + rng.choice([0, 2])
    ipY = pY * H
    base = slack * pY + slack
    nY = base + ipY * (NPEL * NPEL + 1)

    if flat:
        v = rng.randint(32, 200)
        srcY = [v] * nY
        refY = [v if rng.random() < 0.7 else rng.randint(0, 255) for _ in range(nY)]
        if rng.random() < 0.5:
            refY = [v] * nY
    else:
        srcY = [rng.randint(0, 255) for _ in range(nY)]
        refY = [rng.randint(0, 255) for _ in range(nY)]

    pUV = ipUV = baseUV = nUV = 0
    srcU = srcV = refU = refV = None
    if chroma:
        WU = ((nExtW + 2 * nPad) // 2) * NPEL + slack
        HU = ((nExtH + 2 * nPad) // 2) * NPEL + slack
        pUV = WU + rng.choice([0, 2])
        ipUV = pUV * HU
        baseUV = slack * pUV + slack
        nUV = baseUV + ipUV * (NPEL * NPEL + 1)
        if flat:
            srcU = [rng.randint(0, 255)] * nUV
            srcV = [rng.randint(0, 255)] * nUV
            refU = [srcU[0]] * nUV
            refV = [srcV[0]] * nUV
        else:
            srcU = [rng.randint(0, 255) for _ in range(nUV)]
            srcV = [rng.randint(0, 255) for _ in range(nUV)]
            refU = [rng.randint(0, 255) for _ in range(nUV)]
            refV = [rng.randint(0, 255) for _ in range(nUV)]

    n = nBlkX * nBlkY
    own, copy, sads = [], [], []
    for _ in range(batch):
        for _ in range(n):
            # MVs around the clip range so clipping and boundary candidates bite
            vx = rng.randint(-nPad, nPad) * NPEL
            vy = rng.randint(-nPad, nPad) * NPEL
            own += [vx, vy]
        for _ in range(n):
            vx = rng.randint(-nPad, nPad) * NPEL
            vy = rng.randint(-nPad, nPad) * NPEL
            copy += [vx, vy]
        for _ in range(n):
            sads.append(rng.choice([0, 1, 400, 900, rng.randint(0, 2000)]))

    par = (nBlkX, nBlkY, BS, NPEL, nPad, nBlkSizeOvr, nLogScale, nExtW, nExtH,
           rng.choice([0, 4, 100, 400, 1000]),   # nLambdaLevel (0/4 force pruning)
           rng.choice([400, 1600]),              # lsad
           rng.choice([0, 25, 300]),             # penaltyZero
           rng.choice([0, 50]),                  # penaltyGlobal
           rng.choice([0, 25, 300]))             # penaltyNew
    gx, gy = rng.randint(-2, 2), rng.randint(-2, 2)
    return par, gx, gy, own, copy, sads, \
        (srcY, refY, srcU, refV, refU, srcV, pY, pUV, ipY, ipUV), base, baseUV, nY, nUV


def main():
    subprocess.run(["g++", "-O2", "-std=gnu++17", "-Wl,--allow-multiple-definition", "-ffp-contract=off", "-w",
                    os.path.join(REPO, "sim", "ktgmc_search_ref.cpp"),
                    "-o", BIN], check=True)
    rng = random.Random(77007)
    ok = True
    total = 0
    cov_tot = {}

    shapes = []
    # full reachable grid at small size, plus extra multi-batch / tie cases
    for BS in (8, 16, 32):
        for SEARCH in (1, 2):
            for NPEL in (1, 2):
                for chroma in (0, 1):
                    shapes.append((BS, SEARCH, NPEL, chroma))

    for si, (BS, SEARCH, NPEL, chroma) in enumerate(shapes):
        for rep in range(2):
            flat = (rep == 1)          # second pass: tie-heavy / all-pruned
            nBlkX = rng.randint(2, 4)
            nBlkY = rng.randint(2, 3)
            batch = rng.choice([1, 2])
            (par, gx, gy, own, copy, sads, planes,
             base, baseUV, nY, nUV) = make_case(
                rng, BS, SEARCH, NPEL, chroma, flat, nBlkX, nBlkY, batch)
            n = nBlkX * nBlkY
            COV.clear()
            COV.update(pruned=0, all_pruned=0, stage_update=0)
            (nBlkX_, nBlkY_, BS_, NPEL_, nPad_, ovr, nls, nExtW, nExtH,
             lam, lsad, pz, pg, pn) = par

            nums = ([ord('S'), nBlkX, nBlkY, BS, SEARCH, NPEL, chroma,
                     nPad_, ovr, nls, nExtW, nExtH, lam, lsad, pz, pg, pn,
                     gx, gy, batch]
                    + own + copy + sads
                    + [nY] + planes[0] + [nY] + planes[1]
                    + [nUV])
            if chroma:
                nums += planes[2] + planes[5] + planes[4] + planes[3]
            nums += [planes[6], planes[7], planes[8], planes[9]]
            got = run_mirror(nums)

            exp = []
            for b in range(batch):
                ow = own[b * 2 * n:(b + 1) * 2 * n]
                cp = copy[b * 2 * n:(b + 1) * 2 * n]
                sd = sads[b * n:(b + 1) * n]
                blocks, zero, glob = golden_prepare(par, ow, sd, gx, gy)
                # plane slices per batch: same planes are shared; the search
                # reads are batch-independent here (single shared scene)
                r = golden_search(par, blocks, zero, glob, ow, cp, planes,
                                  SEARCH)
                exp += [v for t in r for v in t]
                total += 1
            for k, v in COV.items():
                cov_tot[k] = cov_tot.get(k, 0) + v

            if got != exp:
                ok = False
                print("search MISMATCH",
                      (BS, SEARCH, NPEL, chroma, nBlkX, nBlkY, batch, flat))
                for i, (g, e) in enumerate(zip(got, exp)):
                    if g != e:
                        blk, w = divmod(i, 3)
                        print("  first diff at block", blk, "field", w,
                              "got", g, "exp", e)
                        break
                if len(got) != len(exp):
                    print("  LENGTH", len(got), len(exp))

    # --- layout pins -------------------------------------------------------
    # 1. BR predictor reads the COPY region: changing the copy region must be
    #    able to change interior-block results (poisoned distinct values).
    (BS, SEARCH, NPEL, chroma) = (8, 2, 1, 0)
    nBlkX, nBlkY, batch = 4, 3, 1
    (par, gx, gy, own, copy, sads, planes,
     base, baseUV, nY, nUV) = make_case(rng, BS, SEARCH, NPEL, chroma, False,
                                        nBlkX, nBlkY, batch)
    n = nBlkX * nBlkY
    lam = 100
    par = par[:9] + (lam, 400, 25, 0, 25)
    base_nums = ([ord('S'), nBlkX, nBlkY, BS, SEARCH, NPEL, chroma]
                 + list(par[4:]) + [gx, gy, batch] + own + copy + sads
                 + [nY] + planes[0] + [nY] + planes[1] + [0]
                 + [planes[6], planes[7], planes[8], planes[9]])
    got_a = run_mirror(base_nums)
    n2 = len(copy)
    poisoned = [(-1 if copy[i] >= 0 else 1) * (abs(copy[i]) + 1) * NPEL
                for i in range(n2)]
    # keep copy values within legal MV range for the SAD reads
    poisoned = [max(-NPEL * 4, min(NPEL * 4, v)) for v in poisoned]
    nums_poison = ([ord('S'), nBlkX, nBlkY, BS, SEARCH, NPEL, chroma]
                   + list(par[4:]) + [gx, gy, batch] + own + poisoned + sads
                   + [nY] + planes[0] + [nY] + planes[1] + [0]
                   + [planes[6], planes[7], planes[8], planes[9]])
    got_b = run_mirror(nums_poison)
    if got_a == got_b:
        print("LAYOUT PIN FAILED: poisoning the copy region changed nothing; "
              "the BR predictor is not reading the copy region")
        ok = False
    else:
        total += 1

    # 2. left predictor reads the OWN region FRESH: shuffling the incoming
    #    own region (same multiset, different placement) must change the
    #    result (fresh-value dependence), while a pure stale-read model
    #    would disagree with the golden either way.
    own2 = list(own)
    own2[0], own2[2] = own2[2], own2[0]
    nums_own = ([ord('S'), nBlkX, nBlkY, BS, SEARCH, NPEL, chroma]
                + list(par[4:]) + [gx, gy, batch] + own2 + copy + sads
                + [nY] + planes[0] + [nY] + planes[1] + [0]
                + [planes[6], planes[7], planes[8], planes[9]])
    got_c = run_mirror(nums_own)
    blocks2, zero, glob = golden_prepare(par, own2, sads, gx, gy)
    r2 = golden_search(par, blocks2, zero, glob, own2, copy, planes, SEARCH)
    exp_c = [v for t in r2 for v in t]
    total += 1
    if got_c != exp_c:
        ok = False
        print("search MISMATCH (own-shuffle pin)")
    if got_c == got_a:
        print("FRESHNESS PIN WARNING: swapping incoming own MVs changed "
              "nothing (own predictor may be inert in this case)")

    missing = [k for k in ("pruned", "all_pruned", "stage_update")
               if cov_tot.get(k, 0) == 0]
    print(f"MV block-search E2E: {'PASS' if ok else 'FAIL'} ({total} cases, "
          f"branches pruned={cov_tot.get('pruned', 0)} "
          f"all_pruned={cov_tot.get('all_pruned', 0)} "
          f"stage_update={cov_tot.get('stage_update', 0)})")
    if missing:
        print("  UNCOVERED BRANCHES:", missing)
        return 1
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
