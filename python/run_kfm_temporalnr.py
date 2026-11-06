#!/usr/bin/env python3
"""Validate the KFM KTemporalNR kernel (kf_temporal_nr) in
src/opencl/kfm/kernels/kfm_temporalnr.cl against the CPU mirror
sim/kfm_temporalnr_ref.cpp with an independent Python golden.

The final `avg = (float)sum/count + 0.5f` is float32 (same as the CPU twin); the
mirror is compiled with -ffp-contract=off and the golden emulates float32 per
operation so the two match bit-for-bit.

Run:  python3 python/run_kfm_temporalnr.py
"""
import os, random, struct, subprocess, tempfile, sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BIN = os.path.join(tempfile.gettempdir(), "kfm_temporalnr_ref")


def F(v):
    return struct.unpack('f', struct.pack('f', v))[0]


def run_mirror(nums):
    inf = os.path.join(tempfile.gettempdir(), "kftn_in.txt")
    outf = os.path.join(tempfile.gettempdir(), "kftn_out.txt")
    with open(inf, "w") as f:
        f.write(" ".join(map(str, nums)))
    subprocess.run([BIN, inf, outf], check=True)
    with open(outf) as f:
        return [int(line) for line in f]


def golden(width, height, pitch, frame_stride, nframes, mid, thresh, frames):
    out = []
    for yy in range(height):
        for xx in range(width):
            idx = xx + yy * pitch
            center = frames[mid * frame_stride + idx]
            count = 0
            s = 0
            for i in range(nframes):
                ref = frames[i * frame_stride + idx]
                if abs(ref - center) <= thresh:
                    count += 1
                    s += ref
            avg = F(F(s / float(count)) + 0.5)
            out.append(int(avg))
    return out


def main():
    subprocess.run(["g++", "-O2", "-std=gnu++17", "-Wl,--allow-multiple-definition", "-ffp-contract=off", "-w",
                    os.path.join(REPO, "sim", "kfm_temporalnr_ref.cpp"),
                    "-o", BIN], check=True)
    rng = random.Random(13)
    ok = True
    total = 0
    for _ in range(300):
        bits = rng.choice([8, 8, 16])
        maxv = 255 if bits == 8 else 65535
        width = rng.randint(1, 32)
        height = rng.randint(1, 16)
        pitch = width + rng.choice([0, 1])
        dist = rng.randint(0, 6)
        nframes = 2 * dist + 1
        mid = dist
        # thresh: scaleParam of a float in [0, ~8]; keep it small enough to matter
        thresh = rng.randint(0, max(2, maxv // 32))
        frame_stride = pitch * height
        n = frame_stride * nframes
        # temporally correlated frames: base scene + small noise, with some
        # bursts/outliers so the thresh gate selects a varying subset
        base = [rng.randint(0, maxv) for _ in range(pitch * height)]
        frames = []
        for i in range(nframes):
            f = []
            for v in base:
                nz = rng.randint(0, maxv // 16)
                f.append(max(0, min(maxv, v + nz - (maxv // 32))))
            # occasionally corrupt a whole row to exercise the gate
            if rng.random() < 0.3:
                r0 = rng.randrange(height)
                for xx in range(pitch):
                    f[r0 * pitch + xx] = rng.randint(0, maxv)
            frames.extend(f)
        hdr = [ord('T'), width, height, pitch, frame_stride, nframes, mid,
               thresh, n]
        got = run_mirror(hdr + frames)
        exp = golden(width, height, pitch, frame_stride, nframes, mid, thresh,
                     frames)
        total += 1
        if got != exp:
            ok = False
            for i, (g, e) in enumerate(zip(got, exp)):
                if g != e:
                    print("KTemporalNR MISMATCH", width, height, pitch,
                          "dist", dist, "thresh", thresh, "px", i, g, e)
                    break
            if total >= 3:
                break
    print(f"KFM KTemporalNR temporal_nr: {'PASS' if ok else 'FAIL'} "
          f"({total} cases)")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
