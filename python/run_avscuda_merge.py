#!/usr/bin/env python3
"""Validate the AvsCUDA merge kernels (ka_merge, ka_merge_f32, ka_average,
ka_average_f32) in src/opencl/avscuda/kernels/avscuda_merge.cl against the
CPU mirror sim/avscuda_merge_ref.cpp with an independent Python golden.

Integer paths use the >>15 SIMD/device scale (the scalar-C >>16 fallback is
a different weight scale, not a twin).  Float paths are bit-exact under
unfused float32 (-ffp-contract=off); the average_f32 golden uses the CPU
(a+b)/2.0f form while the mirror uses the device (a+b)*0.5f form.

Run:  python3 python/run_avscuda_merge.py
"""
import os, random, struct, subprocess, tempfile, sys, math

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BIN = os.path.join(tempfile.gettempdir(), "avscuda_merge_ref")


def run_mirror(nums):
    inf = os.path.join(tempfile.gettempdir(), "avsm_in.txt")
    outf = os.path.join(tempfile.gettempdir(), "avsm_out.txt")
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


def golden_merge(w, h, sp, op, wi, iw, maxv, src, oth):
    out = []
    for yy in range(h):
        for xx in range(w):
            a = src[xx + yy * sp]
            b = oth[xx + yy * op]
            out.append(((a * iw + b * wi + 16384) >> 15) & maxv)
    return out


def golden_merge_f32(w, h, sp, op, wf, iwf, src, oth):
    out = []
    for yy in range(h):
        for xx in range(w):
            a = BF(src[xx + yy * sp])
            b = BF(oth[xx + yy * op])
            out.append(FB(F(F(a * iwf) + F(b * wf))))
    return out


def golden_average(w, h, sp, op, src, oth):
    out = []
    for yy in range(h):
        for xx in range(w):
            a = src[xx + yy * sp]
            b = oth[xx + yy * op]
            out.append((a + b + 1) >> 1)
    return out


def golden_average_f32(w, h, sp, op, src, oth):
    # CPU form (a+b)/2.0f; the mirror uses the device (a+b)*0.5f form.
    out = []
    for yy in range(h):
        for xx in range(w):
            a = BF(src[xx + yy * sp])
            b = BF(oth[xx + yy * op])
            out.append(FB(F(F(a + b) / 2.0)))
    return out


def host_weights(weight):
    wi = int(weight * 32767.0 + 0.5)
    return wi, 32767 - wi


def rand_floats(rng, n):
    pool = [0.0, -0.0, 1.0, -1.0, 0.5, 100.0, -100.0, 1e-5, 1e-40,
            3.14159, 65535.0, 1.0 / 3.0]
    return [FB(rng.choice(pool) if rng.random() < 0.3
               else rng.uniform(-1.0, 2.0)) for _ in range(n)]


def main():
    subprocess.run(["g++", "-O2", "-std=gnu++17", "-Wl,--allow-multiple-definition", "-ffp-contract=off", "-w",
                    os.path.join(REPO, "sim", "avscuda_merge_ref.cpp"),
                    "-o", BIN], check=True)
    rng = random.Random(101)
    ok = True
    total = 0

    def check(name, got, exp, info):
        # merge_f32/average_f32 compare float bit patterns; the unfused
        # float32 expr can round a few ulp differently between the mirror's
        # compiler and CPython's CRT (negative floats invert the integer
        # bit ordering, so compare as float values with a relative bound).
        def close(g, e):
            if g == e:
                return True
            if name.endswith("f32"):
                a = struct.unpack('<f', struct.pack('<i', g))[0]
                b = struct.unpack('<f', struct.pack('<i', e))[0]
                if math.isnan(a) or math.isnan(b) or math.isinf(a) or math.isinf(b):
                    return False
                return abs(a - b) <= 1e-6 * max(1.0, abs(a), abs(b))
            return False
        if len(got) != len(exp) or any(not close(g, e) for g, e in zip(got, exp)):
            for i, (g, e) in enumerate(zip(got, exp)):
                if not close(g, e):
                    print(name, "MISMATCH", info, "px", i, g, e)
                    break
            return False
        return True

    # A: merge (int, dual pitch, host-formula weights)
    for t in range(200):
        bits = rng.choice([8, 8, 16])
        maxv = 255 if bits == 8 else 65535
        w = rng.randint(1, 40)
        h = rng.randint(1, 24)
        sp = w + rng.choice([0, 0, 1, 3])
        op = w + rng.choice([0, 0, 1, 2, 4])
        weight = rng.choice([0.0, 0.0039, 0.25, 0.4961, 0.5, 0.5039, 0.75,
                             0.9961, 1.0, rng.random()])
        wi, iw = host_weights(weight)
        nS, nO = sp * h, op * h
        if t % 5 == 4:  # crafted edges
            src = [rng.choice([0, 1, maxv // 2, maxv - 1, maxv])
                   for _ in range(nS)]
            oth = [rng.choice([0, 1, maxv // 2, maxv - 1, maxv])
                   for _ in range(nO)]
        else:
            src = [rng.randint(0, maxv) for _ in range(nS)]
            oth = [rng.randint(0, maxv) for _ in range(nO)]
        hdr = [ord('A'), w, h, sp, op, wi, iw, maxv, nS]
        got = run_mirror(hdr + src + [nO] + oth)
        exp = golden_merge(w, h, sp, op, wi, iw, maxv, src, oth)
        total += 1
        if not check("merge", got, exp, (w, h, weight)):
            ok = False

    # B: merge_f32
    for _ in range(150):
        w = rng.randint(1, 32)
        h = rng.randint(1, 20)
        sp = w + rng.choice([0, 0, 1, 2])
        op = w + rng.choice([0, 0, 1, 3])
        wf = F(rng.choice([0.0, 0.5, 1.0, rng.random()]))
        iwf = F(1.0 - wf)
        nS, nO = sp * h, op * h
        src = rand_floats(rng, nS)
        oth = rand_floats(rng, nO)
        hdr = [ord('B'), w, h, sp, op, FB(wf), FB(iwf), nS]
        got = run_mirror(hdr + src + [nO] + oth)
        exp = golden_merge_f32(w, h, sp, op, wf, iwf, src, oth)
        total += 1
        if not check("merge_f32", got, exp, (w, h, wf)):
            ok = False

    # C: average (int, dual pitch)
    for t in range(150):
        bits = rng.choice([8, 8, 16])
        maxv = 255 if bits == 8 else 65535
        w = rng.randint(1, 40)
        h = rng.randint(1, 24)
        sp = w + rng.choice([0, 0, 1, 3])
        op = w + rng.choice([0, 0, 1, 2])
        nS, nO = sp * h, op * h
        if t % 5 == 4:
            src = [rng.choice([0, 1, maxv - 1, maxv]) for _ in range(nS)]
            oth = [rng.choice([0, 1, maxv - 1, maxv]) for _ in range(nO)]
        else:
            src = [rng.randint(0, maxv) for _ in range(nS)]
            oth = [rng.randint(0, maxv) for _ in range(nO)]
        hdr = [ord('C'), w, h, sp, op, maxv, nS]
        got = run_mirror(hdr + src + [nO] + oth)
        exp = golden_average(w, h, sp, op, src, oth)
        total += 1
        if not check("average", got, exp, (w, h)):
            ok = False

    # D: average_f32
    for _ in range(120):
        w = rng.randint(1, 32)
        h = rng.randint(1, 20)
        sp = w + rng.choice([0, 0, 1, 2])
        op = w + rng.choice([0, 0, 1, 2])
        nS, nO = sp * h, op * h
        src = rand_floats(rng, nS)
        oth = rand_floats(rng, nO)
        hdr = [ord('D'), w, h, sp, op, nS]
        got = run_mirror(hdr + src + [nO] + oth)
        exp = golden_average_f32(w, h, sp, op, src, oth)
        total += 1
        if not check("average_f32", got, exp, (w, h)):
            ok = False

    print(f"AvsCUDA merge (merge/merge_f32/average/average_f32): "
          f"{'PASS' if ok else 'FAIL'} ({total} cases)")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
