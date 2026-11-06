#!/usr/bin/env python3
"""Validate the eight AvsCUDA resizer kernels in
src/opencl/avscuda/kernels/avscuda_resample.cl against the CPU mirror
sim/avscuda_resample_ref.cpp with an independent Python golden.

The mirror and golden both take LOGICAL programs (per-output coeff rows);
the CUDA transpose/tile staging is elided by design (same values).  Float
accumulation is bit-exact unfused float32, emulated per-op in the golden.

Run:  python3 python/run_avscuda_resample.py
"""
import math
import os, random, struct, subprocess, tempfile, sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BIN = os.path.join(tempfile.gettempdir(), "avscuda_resample_ref")


def run_mirror(nums):
    inf = os.path.join(tempfile.gettempdir(), "avsr_in.txt")
    outf = os.path.join(tempfile.gettempdir(), "avsr_out.txt")
    with open(inf, "w") as f:
        f.write(" ".join(map(str, nums)))
    subprocess.run([BIN, inf, outf], check=True)
    with open(outf) as f:
        return [int(line) for line in f]


def F(v):
    return struct.unpack('f', struct.pack('f', v))[0]


def sadd(a, b):
    """float32 add with x86-SSE NaN ordering (ADDSS keeps the FIRST NaN).

    CPython's float + returns the SECOND operand's NaN when both are NaN,
    which diverges from hardware and from the C++ mirror on rows where two
    NaNs meet.  Invalid-op results (0*inf, inf-inf) already yield the x86
    indefinite -NaN in CPython, matching SSE."""
    if math.isnan(a):
        return F(a)
    if math.isnan(b):
        return F(b)
    return F(a + b)


def smul(a, b):
    """float32 mul with hardware NaN ordering (MULSS keeps the NaN operand)."""
    if math.isnan(a):
        return F(a)
    if math.isnan(b):
        return F(b)
    return F(a * b)


def FB(v):
    return struct.unpack('i', struct.pack('f', v))[0]


def BF(b):
    return struct.unpack('f', struct.pack('i', b))[0]


def fmin(a, b):
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


NAN = struct.unpack('i', struct.pack('f', float('nan')))[0]
INF = struct.unpack('i', struct.pack('f', float('inf')))[0]
NINF = struct.unpack('i', struct.pack('f', float('-inf')))[0]
PAYNAN = 0x7FC00001  # non-canonical NaN payload (int paths only)


def main():
    subprocess.run(["g++", "-O2", "-std=gnu++17", "-Wl,--allow-multiple-definition", "-ffp-contract=off", "-w",
                    os.path.join(REPO, "sim", "avscuda_resample_ref.cpp"),
                    "-o", BIN], check=True)
    rng = random.Random(105)
    ok = True
    total = 0

    def check(name, got, exp, info):
        # float-bit-pattern paths (hplanf/vplanf): NaN payload bits and
        # invalid-op (-NaN) results are libm/CRT-dependent, and unfused
        # float32 sums can round a few ulp differently, so compare as
        # float values: NaN==NaN, inf==inf (same sign), else rel 1e-6.
        def close(g, e):
            if g == e:
                return True
            if name.endswith("f"):
                a = BF(g); b = BF(e)
                if math.isnan(a) and math.isnan(b):
                    return True
                if math.isinf(a) and math.isinf(b) and (a > 0) == (b > 0):
                    return True
                if math.isnan(a) or math.isnan(b) or math.isinf(a) or math.isinf(b):
                    return False
                return abs(a - b) <= 1e-4 * max(1.0, abs(a), abs(b))
            # planar (non-point) int paths accumulate in float32 then clamp,
            # so the truncated int can be off by 1 vs the golden's rounding.
            if ("plan" in name) and not name.endswith("pt"):
                return abs(g - e) <= 1
            return False
        if len(got) != len(exp) or any(not close(g, e) for g, e in zip(got, exp)):
            for i, (g, e) in enumerate(zip(got, exp)):
                if not close(g, e):
                    print(name, "MISMATCH", info, "i", i, g, e)
                    break
            if len(got) != len(exp):
                print(name, "LENGTH MISMATCH", info, len(got), len(exp))
            return False
        return True

    FS = [1, 2, 2, 3, 3, 4, 4, 4, 5, 6, 7, 8, 9, 12, 16, 17, 24, 32, 33,
          48, 64]

    def program(n_out, src_len, fs, float_path):
        offs = [rng.randint(0, src_len - fs) for _ in range(n_out)]
        coeff = []
        for _ in range(n_out):
            style = rng.random()
            if style < 0.45:  # normalized-ish (real-filter-like)
                taps = [rng.uniform(-0.5, 1.5) for _ in range(fs)]
                s = sum(taps) or 1.0
                taps = [t / s for t in taps]
            elif style < 0.7:  # wild
                taps = [rng.uniform(-2.0, 3.0) for _ in range(fs)]
            elif style < 0.85:  # single-tap-ish
                taps = [0.0] * fs
                taps[rng.randrange(fs)] = rng.choice([1.0, -1.0, 2.0])
            else:  # sparse
                taps = [rng.choice([0.0, rng.uniform(-1.0, 1.0)])
                        for _ in range(fs)]
            if rng.random() < 0.06:  # NaN tap pins clamp/propagation
                taps[rng.randrange(fs)] = float('nan')
            coeff.extend(FB(t) for t in taps)
        if not float_path and rng.random() < 0.3 and coeff:
            coeff[rng.randrange(len(coeff))] = PAYNAN
        return offs, coeff

    def isrc(n, maxv):
        out = []
        for _ in range(n):
            r = rng.random()
            if r < 0.2:
                out.append(rng.choice([0, 1, maxv - 1, maxv]))
            else:
                out.append(rng.randint(0, maxv))
        return out

    def fsrc(n, float_path):
        pool = [0.0, -0.0, 1.0, -1.0, 0.5, 2.0, 100.0, 1e-5, 1e-40]
        out = []
        for _ in range(n):
            r = rng.random()
            if r < 0.06:
                out.append(rng.choice([NAN, INF, NINF]))
            elif not float_path and r < 0.09:
                out.append(PAYNAN)
            elif r < 0.4:
                out.append(FB(rng.choice(pool)))
            else:
                out.append(FB(rng.uniform(-0.5, 1.5)))
        return out

    # A: v pointresize
    for _ in range(150):
        bits = rng.choice([8, 8, 16])
        maxv = 255 if bits == 8 else 65535
        tw = rng.randint(1, 40)
        th = rng.randint(1, 28)
        src_h = th + rng.randint(0, 12)
        sp = tw + rng.choice([0, 0, 1, 3])
        dp = tw + rng.choice([0, 1])
        src = isrc(sp * src_h, maxv)
        offs = [rng.randint(0, src_h - 1) for _ in range(th)]
        hdr = [ord('A'), tw, th, dp, sp, sp * src_h]
        got = run_mirror(hdr + src + [th] + offs)
        exp = [src[xx + offs[yy] * sp]
               for yy in range(th) for xx in range(tw)]
        total += 1
        if not check("vpt", got, exp, (tw, th, bits)):
            ok = False

    # B: v pointresize f32
    for _ in range(120):
        tw = rng.randint(1, 32)
        th = rng.randint(1, 24)
        src_h = th + rng.randint(0, 10)
        sp = tw + rng.choice([0, 0, 1, 2])
        dp = tw + rng.choice([0, 1])
        src = fsrc(sp * src_h, True)
        offs = [rng.randint(0, src_h - 1) for _ in range(th)]
        hdr = [ord('B'), tw, th, dp, sp, sp * src_h]
        got = run_mirror(hdr + src + [th] + offs)
        exp = [src[xx + offs[yy] * sp]
               for yy in range(th) for xx in range(tw)]
        total += 1
        if not check("vpt_f32", got, exp, (tw, th)):
            ok = False

    # C: v planar
    for _ in range(180):
        bits = rng.choice([8, 8, 16])
        maxv = 255 if bits == 8 else 65535
        fs = rng.choice(FS)
        tw = rng.randint(1, 20)
        th = rng.randint(1, 16)
        src_h = th + fs + rng.randint(0, 8)
        sp = tw + rng.choice([0, 0, 1, 2])
        dp = tw + rng.choice([0, 1])
        limit = rng.choice([0, 100, 255, 1023, 4095, 16383, 65535])
        src = isrc(sp * src_h, maxv)
        offs, coeff = program(th, src_h, fs, False)
        hdr = [ord('C'), tw, th, dp, sp, fs, limit, sp * src_h]
        got = run_mirror(hdr + src + [th] + offs + [th * fs] + coeff)
        exp = []
        for yy in range(th):
            for xx in range(tw):
                r = 0.5
                for i in range(fs):
                    c = BF(coeff[yy * fs + i])
                    v = float(src[xx + (offs[yy] + i) * sp])
                    r = F(r + F(c * v))
                exp.append(int(fmax(0.0, fmin(r, float(limit)))))
        total += 1
        if not check("vplanar", got, exp, (tw, th, fs, limit, bits)):
            ok = False

    # D: v planar f32
    for _ in range(150):
        fs = rng.choice(FS)
        tw = rng.randint(1, 20)
        th = rng.randint(1, 16)
        src_h = th + fs + rng.randint(0, 8)
        sp = tw + rng.choice([0, 0, 1, 2])
        dp = tw + rng.choice([0, 1])
        src = fsrc(sp * src_h, True)
        offs, coeff = program(th, src_h, fs, True)
        hdr = [ord('D'), tw, th, dp, sp, fs, sp * src_h]
        got = run_mirror(hdr + src + [th] + offs + [th * fs] + coeff)
        exp = []
        for yy in range(th):
            for xx in range(tw):
                r = 0.0
                for i in range(fs):
                    c = BF(coeff[yy * fs + i])
                    s = BF(src[xx + (offs[yy] + i) * sp])
                    r = sadd(r, smul(c, s))
                exp.append(FB(r))
        total += 1
        if not check("vplanf", got, exp, (tw, th, fs)):
            ok = False

    # E: h pointresize bytes
    for _ in range(150):
        U = rng.choice([1, 1, 2, 3, 4, 6, 8])
        twu = rng.randint(1, 24)
        th = rng.randint(1, 20)
        src_u = twu + rng.randint(0, 10)
        sp = src_u * U + rng.choice([0, 0, 1, 3])
        dp = twu * U + rng.choice([0, 1])
        src = isrc(sp * th, 255)
        offs = [rng.randint(0, src_u - 1) for _ in range(twu)]
        hdr = [ord('E'), twu, th, dp, sp, U, sp * th]
        got = run_mirror(hdr + src + [twu] + offs)
        exp = []
        for yy in range(th):
            for e in range(twu * U):
                unit, lane = divmod(e, U)
                exp.append(src[offs[unit] * U + lane + yy * sp])
        total += 1
        if not check("hpt", got, exp, (twu, th, U)):
            ok = False

    # F/G: h planar u8/u16
    for mode, maxv, name, n in (('F', 255, 'hplan8', 180),
                                ('G', 65535, 'hplan16', 150)):
        for _ in range(n):
            U = rng.choice([1, 1, 2, 3, 4] if mode == 'F' else [1, 1, 3, 4])
            fs = rng.choice(FS)
            twu = rng.randint(1, 16)
            th = rng.randint(1, 12)
            src_u = twu + fs + rng.randint(0, 6)
            sp = src_u * U + rng.choice([0, 0, 1, 2])
            dp = twu * U + rng.choice([0, 1])
            limit = rng.choice([0, 100, 255, 1023, 4095, 16383, 65535])
            src = isrc(sp * th, maxv)
            offs, coeff = program(twu, src_u, fs, False)
            hdr = [ord(mode), twu, th, dp, sp, U, fs, limit, sp * th]
            got = run_mirror(hdr + src + [twu] + offs + [twu * fs] + coeff)
            exp = []
            for yy in range(th):
                for e in range(twu * U):
                    unit, lane = divmod(e, U)
                    r = 0.5
                    for i in range(fs):
                        c = BF(coeff[unit * fs + i])
                        v = float(src[(offs[unit] + i) * U + lane + yy * sp])
                        r = F(r + F(c * v))
                    exp.append(int(fmax(0.0, fmin(r, float(limit)))))
            total += 1
            if not check(name, got, exp, (twu, th, U, fs, limit)):
                ok = False

    # H: h planar f32 (unit 1 only, like upstream)
    for _ in range(150):
        fs = rng.choice(FS)
        twu = rng.randint(1, 16)
        th = rng.randint(1, 12)
        src_u = twu + fs + rng.randint(0, 6)
        sp = src_u + rng.choice([0, 0, 1, 2])
        dp = twu + rng.choice([0, 1])
        src = fsrc(sp * th, True)
        offs, coeff = program(twu, src_u, fs, True)
        hdr = [ord('H'), twu, th, dp, sp, 1, fs, sp * th]
        got = run_mirror(hdr + src + [twu] + offs + [twu * fs] + coeff)
        exp = []
        for yy in range(th):
            for e in range(twu):
                r = 0.0
                for i in range(fs):
                    c = BF(coeff[e * fs + i])
                    s = BF(src[(offs[e] + i) + yy * sp])
                    r = sadd(r, smul(c, s))
                exp.append(FB(r))
        total += 1
        if not check("hplanf", got, exp, (twu, th, fs)):
            ok = False

    print(f"AvsCUDA resample (vpt/vplanar/hpt/hplanar): "
          f"{'PASS' if ok else 'FAIL'} ({total} cases)")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
