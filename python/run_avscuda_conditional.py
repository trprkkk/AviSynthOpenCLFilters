#!/usr/bin/env python3
"""Validate the fifteen AvsCUDA Conditional kernels in
src/opencl/avscuda/kernels/avscuda_conditional.cl against the CPU mirror
sim/avscuda_conditional_ref.cpp with an independent Python golden.

Integer sums are order-free (u32 wraparound pinned with oversized frames);
the float hist index uses fmin/fmax NaN rules (NaN -> 65535).  The float
sum/SAD reductions are // RIG-VERIFY (no golden can define their
arrival-order-dependent result) and are not covered here.

Run:  python3 python/run_avscuda_conditional.py
"""
import math
import os, random, struct, subprocess, tempfile, sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BIN = os.path.join(tempfile.gettempdir(), "avscuda_conditional_ref")


def run_mirror(nums):
    inf = os.path.join(tempfile.gettempdir(), "avcd_in.txt")
    outf = os.path.join(tempfile.gettempdir(), "avcd_out.txt")
    with open(inf, "w") as f:
        f.write(" ".join(map(str, nums)))
    subprocess.run([BIN, inf, outf], check=True)
    with open(outf) as f:
        return [int(line) for line in f]


def F(v):
    return struct.unpack('f', struct.pack('f', v))[0]


def FB(v):
    return struct.unpack('i', struct.pack('f', v))[0]


def BF(b):
    return struct.unpack('f', struct.pack('i', b))[0]


def fmin(a, b):  # CUDA-intrinsic / C fminf NaN rules
    if math.isnan(a):
        return b
    if math.isnan(b):
        return a
    return a if a < b else b


def fmax(a, b):
    if math.isnan(a):
        return b
    if math.isnan(b):
        return a
    return a if a > b else b


def main():
    subprocess.run(["g++", "-O2", "-std=gnu++17", "-Wl,--allow-multiple-definition", "-ffp-contract=off", "-w",
                    os.path.join(REPO, "sim", "avscuda_conditional_ref.cpp"),
                    "-o", BIN], check=True)
    rng = random.Random(104)
    ok = True
    total = 0

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

    W4 = [4, 8, 12, 16, 20, 24, 28, 32, 40]

    def dims():
        return rng.choice(W4), rng.randint(1, 24)

    # A/B/C/O: inits
    for mode, name in (('A', 'init_u32'), ('B', 'init_u64'),
                       ('C', 'init_f32'), ('O', 'init_hist')):
        for _ in range(40):
            ln = rng.choice([1, 1, 2, 3, 7, 16, 63, 64, 65, 255, 256,
                             1024, 65536])
            got = run_mirror([ord(mode), ln])
            total += 1
            if not check(name, got, [0] * ln, ln):
                ok = False

    # D/E: sum u8 (E gets a >2^32 case via 512x512 — still u64-exact)
    for mode, wide, name in (('D', False, 'sum8_32'), ('E', True, 'sum8_64')):
        for t in range(110):
            if t == 0:  # giant wrap probe (pitch == width for a fast golden)
                w = h = 4096 if not wide else 512
                pitch = w
            else:
                w, h = dims()
                pitch = w + rng.choice([0, 0, 4])
            n = pitch * h
            if t == 0:
                src = [255] * n
            elif t % 6 == 5:
                src = [rng.choice([0, 1, 254, 255]) for _ in range(n)]
            else:
                src = [rng.randint(0, 255) for _ in range(n)]
            got = run_mirror([ord(mode), w, h, pitch, 255, n] + src)
            exp = [sum(src[xx + yy * pitch] for yy in range(h)
                       for xx in range(w))]
            if not wide:
                exp = [exp[0] & 0xFFFFFFFF]
            total += 1
            if not check(name, got, exp, (w, h)):
                ok = False

    # F/G: sum u16 (512x512 maxval wraps u32, pins u64 >2^32)
    for mode, wide, name in (('F', False, 'sum16_32'), ('G', True, 'sum16_64')):
        for t in range(110):
            maxv = rng.choice([1023, 4095, 16383, 65535])
            if t == 0:
                w = h = 512
                pitch = w
                maxv = 65535
            else:
                w, h = dims()
                pitch = w + rng.choice([0, 0, 4])
            n = pitch * h
            if t == 0:
                src = [65535] * n
            elif t % 6 == 5:
                src = [rng.choice([0, 1, maxv - 1, maxv, maxv + 1, 65535])
                       for _ in range(n)]
            else:
                src = [rng.randint(0, 65535) for _ in range(n)]
            got = run_mirror([ord(mode), w, h, pitch, maxv, n] + src)
            exp = [sum(min(src[xx + yy * pitch], maxv) for yy in range(h)
                       for xx in range(w))]
            if not wide:
                exp = [exp[0] & 0xFFFFFFFF]
            total += 1
            if not check(name, got, exp, (w, h, maxv)):
                ok = False

    # H/I: sad u8
    for mode, wide, name in (('H', False, 'sad8_32'), ('I', True, 'sad8_64')):
        for t in range(110):
            if t == 0:
                w = h = 4096 if not wide else 512
                pitch = w
            else:
                w, h = dims()
                pitch = w + rng.choice([0, 0, 4])
            n = pitch * h
            if t == 0:
                s0, s1 = [255] * n, [0] * n
            elif t % 6 == 5:
                s0 = [rng.choice([0, 1, 254, 255]) for _ in range(n)]
                s1 = [rng.choice([0, 1, 254, 255]) for _ in range(n)]
            else:
                s0 = [rng.randint(0, 255) for _ in range(n)]
                s1 = [rng.randint(0, 255) for _ in range(n)]
            got = run_mirror([ord(mode), w, h, pitch, 255, n] + s0 + s1)
            exp = [sum(abs(s0[xx + yy * pitch] - s1[xx + yy * pitch])
                       for yy in range(h) for xx in range(w))]
            if not wide:
                exp = [exp[0] & 0xFFFFFFFF]
            total += 1
            if not check(name, got, exp, (w, h)):
                ok = False

    # J/K: sad u16
    for mode, wide, name in (('J', False, 'sad16_32'), ('K', True, 'sad16_64')):
        for t in range(110):
            maxv = rng.choice([1023, 4095, 16383, 65535])
            if t == 0:
                w = h = 512
                pitch = w
                maxv = 65535
            else:
                w, h = dims()
                pitch = w + rng.choice([0, 0, 4])
            n = pitch * h
            if t == 0:
                s0, s1 = [65535] * n, [0] * n
            elif t % 6 == 5:
                s0 = [rng.choice([0, 1, maxv - 1, maxv, maxv + 1, 65535])
                      for _ in range(n)]
                s1 = [rng.choice([0, 1, maxv - 1, maxv, maxv + 1, 65535])
                      for _ in range(n)]
            else:
                s0 = [rng.randint(0, 65535) for _ in range(n)]
                s1 = [rng.randint(0, 65535) for _ in range(n)]
            got = run_mirror([ord(mode), w, h, pitch, maxv, n] + s0 + s1)
            exp = [sum(abs(min(s0[xx + yy * pitch], maxv) -
                           min(s1[xx + yy * pitch], maxv))
                       for yy in range(h) for xx in range(w))]
            if not wide:
                exp = [exp[0] & 0xFFFFFFFF]
            total += 1
            if not check(name, got, exp, (w, h, maxv)):
                ok = False

    # L: hist u8
    for _ in range(100):
        w, h = dims()
        pitch = w + rng.choice([0, 0, 4])
        n = pitch * h
        src = [rng.randint(0, 255) for _ in range(n)]
        got = run_mirror([ord('L'), w, h, pitch, 255, 256, n] + src)
        exp = [0] * 256
        for yy in range(h):
            for xx in range(w):
                exp[src[xx + yy * pitch]] += 1
        total += 1
        if not check("hist8", got, exp, (w, h)):
            ok = False

    # M: hist u16 (production lens)
    for t in range(100):
        bits = rng.choice([10, 12, 14, 16])
        maxv = (1 << bits) - 1
        ln = 1 << bits
        w, h = dims()
        pitch = w + rng.choice([0, 0, 4])
        n = pitch * h
        if t % 4 == 3:
            src = [rng.choice([0, 1, maxv - 1, maxv, maxv + 1, 65535])
                   for _ in range(n)]
        else:
            src = [rng.randint(0, 65535) for _ in range(n)]
        got = run_mirror([ord('M'), w, h, pitch, maxv, ln, n] + src)
        exp = [0] * ln
        for yy in range(h):
            for xx in range(w):
                exp[min(src[xx + yy * pitch], maxv)] += 1
        total += 1
        if not check("hist16", got, exp, (w, h, bits)):
            ok = False

    # N: hist f32 (NaN/Inf/negative/out-of-range pinned)
    NAN = struct.unpack('i', struct.pack('f', float('nan')))[0]
    INF = struct.unpack('i', struct.pack('f', float('inf')))[0]
    NINF = struct.unpack('i', struct.pack('f', float('-inf')))[0]
    for _ in range(80):
        w, h = dims()
        pitch = w + rng.choice([0, 0, 4])
        n = pitch * h
        pool = [0.0, -0.0, 1.0, -1.0, 0.5, 2.0, 100.0, 1e-5, 1e-40]
        src = []
        for _ in range(n):
            r = rng.random()
            if r < 0.08:
                src.append(rng.choice([NAN, INF, NINF]))
            elif r < 0.4:
                src.append(FB(rng.choice(pool)))
            else:
                src.append(FB(rng.uniform(-0.5, 1.5)))
        got = run_mirror([ord('N'), w, h, pitch, 0, 65536, n] + src)
        exp = [0] * 65536
        for yy in range(h):
            for xx in range(w):
                s = BF(src[xx + yy * pitch])
                t = F(F(s * 65535.0) + 0.5)
                c = fmax(0.0, fmin(t, 65535.0))
                exp[int(c)] += 1
        total += 1
        if not check("histf32", got, exp, (w, h)):
            ok = False

    print(f"AvsCUDA conditional (init/sum/sad/hist): "
          f"{'PASS' if ok else 'FAIL'} ({total} cases)")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
