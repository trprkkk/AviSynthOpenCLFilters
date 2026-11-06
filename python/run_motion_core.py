#!/usr/bin/env python3
"""Validate the motion-core weight helpers AND the kt_most_freq_mv mode
tie-break from src/opencl/ktgmc/kernels/ktgmc_motion.cl:
    kt_degrain_weight  (dev_degrain_weight)
    kt_norm_weights    (dev_norm_weights, delta 1..6)
    kt_most_freq_mv    (kl_most_freq_mv mode + staged-fold tie-break)
against the CPU mirror sim/ktgmc_weight_ref.cpp. Pure integer arithmetic, so
they must match the independent Python golden exactly.

The most_freq golden below implements the upstream CUDA staged-fold tie rule
independently of the mirror: per-thread first-strictly-greater pre-scan at
stride 1024 over the 8192-bin histogram (comp+4096), then the up-sweep folds
512..32 (shared) and 16..1 (warp) with MaxIndexReducer's strict `>` keeping
the earlier element in fold order (MVKernel.cu:1368-1427 + ReduceKernel.cuh
MaxIndexReducer/dev_reduce2, upstream @ 68aef6e).  The rule was validated
239/239 against the verbatim upstream CUDA kernel on the rig
(rig/cuda_mv_tie_test.py, tree_rule); the port's previous port-side
"smallest mode" rule diverged on 93/239 tie cases and was replaced
2026-10-28.

Run:  python3 python/run_motion_core.py
"""
import os, random, subprocess, sys, tempfile

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WREF_BIN = os.path.join(tempfile.gettempdir(), "wref")

def degw(th, bs):
    if th <= bs:
        return 0
    sqt = float(th) * th
    sqb = float(bs) * bs
    return int(256.0 * (sqt - sqb) / (sqt + sqb))

def norm(d, b, flat):
    flat = list(flat) + [0] * (12 - len(flat))
    WB = [flat[2 * i] for i in range(6)]
    WF = [flat[2 * i + 1] for i in range(6)]
    WSrc = 256
    if b:
        if d == 1: WSrc *= 2
        elif d == 2: WSrc *= 6; WB[0] *= 4; WF[0] *= 4
        elif d == 3: WSrc *= 20; WB[0] *= 15; WF[0] *= 15; WB[1] *= 6; WF[1] *= 6
        elif d == 4: WSrc *= 70; WB[0] *= 56; WF[0] *= 56; WB[1] *= 28; WF[1] *= 28; WB[2] *= 8; WF[2] *= 8
    if d == 6: WS = sum(WB[:6]) + sum(WF[:6]) + WSrc + 1
    elif d == 5: WS = sum(WB[:5]) + sum(WF[:5]) + WSrc + 1
    elif d == 4: WS = sum(WB[:4]) + sum(WF[:4]) + WSrc + 1
    elif d == 3: WS = sum(WB[:3]) + sum(WF[:3]) + WSrc + 1
    elif d == 2: WS = sum(WB[:2]) + sum(WF[:2]) + WSrc + 1
    else:        WS = WB[0] + WF[0] + WSrc + 1
    for i in range(6):
        WB[i] = WB[i] * 256 // WS
        WF[i] = WF[i] * 256 // WS
    if d == 6: WSrc = 256 - sum(WB[:6]) - sum(WF[:6])
    elif d == 5: WSrc = 256 - sum(WB[:5]) - sum(WF[:5])
    elif d == 4: WSrc = 256 - sum(WB[:4]) - sum(WF[:4])
    elif d == 3: WSrc = 256 - sum(WB[:3]) - sum(WF[:3])
    elif d == 2: WSrc = 256 - sum(WB[:2]) - sum(WF[:2])
    else:        WSrc = 256 - WB[0] - WF[0]
    return [WSrc] + [x for pair in zip(WB[:d], WF[:d]) for x in pair]

def staged_fold_tie(comps_counts):
    """Independent golden for kt_most_freq_mv's tie-break: the exact staged
    fold upstream CUDA performs (see module docstring).  comps_counts is a
    [(component, count)] list of the histogram entries with count > 0."""
    freq = dict(comps_counts)

    def cnt(b):                      # bin b holds component b - 4096
        return freq.get(b - 4096, 0)

    # per-thread pre-scan (MVKernel.cu:1406-1413): first strictly-greater bin
    k = []
    for t in range(1024):
        maxc, idx = 0, 0
        for i in range(8):
            b = t + i * 1024
            if cnt(b) > maxc:
                maxc, idx = cnt(b), b
        k.append((maxc, idx))

    def red(a, b):                   # MaxIndexReducer: strict > keeps a
        return b if b[0] > a[0] else a

    for s in (512, 256, 128, 64, 32):
        for t in range(s):
            k[t] = red(k[t], k[t + s])
    for s in (16, 8, 4, 2, 1):
        for t in range(s):
            k[t] = red(k[t], k[t + s])
    return k[0][1] - 4096


def most_freq_golden(vecs, isY):
    """Golden wrapper: build the histogram of the selected component, then
    apply the staged-fold tie rule."""
    counts = {}
    for v in vecs:
        c = v[1] if isY else v[0]
        counts[c] = counts.get(c, 0) + 1
    return staged_fold_tie(sorted(counts.items()))


def most_freq_cases():
    """Tie-heavy rows for the most_freq section (same construction ideas as
    the CUDA twin rig/cuda_mv_tie_test.py gen_cases: same-tid ties, cross-tid
    ties, cross-warp ties, multi-way ties, fuzz over a small comp pool whose
    members are multiples of a large stride so ties land on distinct tids)."""
    rng = random.Random(4242)
    cases = []                       # (name, [ (x,y) ] )

    def add(name, xs, ys=None):
        ys = xs if ys is None else ys
        cases.append((name, list(zip(xs, ys))))

    add("unique", [5, 5, 5, 7, 9])
    add("tie_same_tid", [3, 3, 3, 1027, 1027, 1027, 11])
    add("tie_same_tid_neg", [0, 0, 0, -1024, -1024, -1024, 2])
    add("tie_cross_tid", [3, 3, 3, 1024, 1024, 1024, 99])
    add("tie_cross_warp", [5, 5, 5, 2000, 2000, 2000])
    add("tie_cross_warp2", [7, 7, 7, 3000, 3000, 3000, -3, -3, -3])
    add("tie_3way", [-100, -100, -100, 924, 924, 924, 1948, 1948, 1948, 4])
    add("tie_gap", [0, 0, 0, 0, 4095, 4095, 4095, 4095])
    big = [rng.randint(-3000, 3000) for _ in range(2000)]
    for i in range(300):
        big[i] = -2000
    for i in range(300, 600):
        big[i] = -2000 + 1024        # same tid as -2000, larger bin
    big[600:900] = [-976] * 300
    add("big_2k", big, [rng.randint(-3000, 3000) for _ in range(2000)])
    for t in range(300):             # fuzz: forced ties from a small pool
        n = rng.randint(4, 260)
        pool = rng.sample([c * 160 for c in range(-24, 24)],
                          rng.randint(2, 6))
        xs = [pool[rng.randrange(len(pool))] for _ in range(n)]
        add("fuzz%04d" % t, xs)
    return cases


def main():
    rng = random.Random(99)
    lines = []
    for _ in range(300):
        th, bs = rng.randint(0, 5000), rng.randint(0, 5000)
        lines.append(f"w {th} {bs}")
    lines += ["w 0 0", "w 1 0", "w 0 1", "w 100 100", "w 100 99", "w 1000 700"]
    for _ in range(400):
        d = rng.randint(1, 6); b = rng.randint(0, 1)
        w = [rng.randint(0, 256) for _ in range(2 * d)]
        lines.append("n %d %d %s" % (d, b, " ".join(map(str, w))))

    # most_freq section: same rows run on both axes (isY 0/1)
    mf_cases = most_freq_cases()
    mf_lines = []
    mf_exp = []
    for name, vecs in mf_cases:
        for isY in (0, 1):
            flat = " ".join(str(c) for v in vecs for c in (v[0], v[1]))
            mf_lines.append("f %d %d %s" % (isY, len(vecs), flat))
            mf_exp.append(most_freq_golden(vecs, isY))

    exp = []
    for ln in lines:
        if ln[0] == 'w':
            th, bs = map(int, ln.split()[1:]); exp.append("w %d" % degw(th, bs))
        else:
            p = ln.split(); r = norm(int(p[1]), int(p[2]), list(map(int, p[3:])))
            exp.append("n %d %s" % (r[0], " ".join(map(str, r[1:]))))
    exp += ["f %d" % v for v in mf_exp]

    subprocess.run(["g++", "-O2", "-std=gnu++17", "-w", "-Wl,--allow-multiple-definition",
                    os.path.join(REPO, "sim", "ktgmc_weight_ref.cpp"),
                    "-o", WREF_BIN], check=True)
    out = subprocess.run([WREF_BIN], input="\n".join(lines + mf_lines) + "\n",
                         capture_output=True, text=True).stdout.splitlines()
    mism = sum(1 for e, g in zip(exp, out) if e != g)
    ok = mism == 0 and len(exp) == len(out)
    print(f"motion-core weight helpers + most_freq tie-break: "
          f"{'PASS' if ok else 'FAIL'} ({len(exp)} cases, mism={mism})")
    return 0 if ok else 1

if __name__ == "__main__":
    sys.exit(main())
