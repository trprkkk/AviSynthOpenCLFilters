#!/usr/bin/env python3
"""Cross-language validation of the KTGMC simple-kernel port.

Two independent implementations of the exact same algorithm (the CUDA kernels
in KTGMC/Kernel.cu):
  1. Python golden below (double precision, mirrors the CUDA host/kernel math).
  2. sim/ktgmc_cpu_ref.cpp (the CPU reference mirror used to build the port).

Both operate on identical raw planes; their integer outputs must match bit-for-
bit.  The OpenCL kernels in src/opencl/ktgmc/kernels/ktgmc_simple.cl are a
line-for-line transliteration of this same math (float accumulation there only
matches the original CUDA float path and is irrelevant for these integer
comparisons except resample, which is double here and float on the device).

Run:  python3 python/run_validation.py
Builds the C++ ref if needed.  No third-party python packages required.
"""
import os, random, struct, subprocess, sys, math, tempfile

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WORK = os.path.join(REPO, "build", "val")
os.makedirs(WORK, exist_ok=True)
REF_BIN = os.path.join(tempfile.gettempdir(), "ktgmc_ref")

W, PITCH = 16, 20
A_H = B_H = REF_H = 10
FIELD_H = 5
GH_H = 12  # gaussian plane height (non-square vs W=16: keeps V/H programs distinct)
NBR = ("a", "b", "c", "ref", "n2", "n1", "p1", "p2", "field", "gsrc")

# ---------------------------------------------------------------- generation
def gen_plane(rng, h, maxval, blocky=True):
    n = PITCH * h
    arr = [0] * n
    for y in range(h):
        base = rng.randrange(0, maxval + 1)
        for x in range(W):
            v = base + rng.randrange(-6, 7)
            v = min(maxval, max(0, v))
            arr[y * PITCH + x] = v
    return arr

def make_inputs(bits):
    maxval = 255 if bits == 8 else 65535
    d = os.path.join(WORK, f"in_{bits}")
    os.makedirs(d, exist_ok=True)
    rng = random.Random(1234 + bits)
    planes = {}
    planes["a"]     = gen_plane(rng, A_H,     maxval)
    planes["b"]     = gen_plane(rng, B_H,     maxval)
    planes["c"]     = gen_plane(rng, REF_H,   maxval)
    planes["ref"]   = gen_plane(rng, REF_H,   maxval)
    planes["n2"]    = gen_plane(rng, A_H,     maxval)
    planes["n1"]    = gen_plane(rng, A_H,     maxval)
    planes["p1"]    = gen_plane(rng, A_H,     maxval)
    planes["p2"]    = gen_plane(rng, A_H,     maxval)
    planes["field"] = gen_plane(rng, FIELD_H, maxval)
    planes["gsrc"]  = gen_plane(rng, GH_H,    maxval)
    fmt = "B" if bits == 8 else "H"
    for name in NBR:
        with open(os.path.join(d, name + ".raw"), "wb") as f:
            f.write(struct.pack(fmt * len(planes[name]), *planes[name]))
    h = {"a": A_H, "b": B_H, "c": REF_H, "ref": REF_H, "n2": A_H,
         "n1": A_H, "p1": A_H, "p2": A_H, "field": FIELD_H, "gsrc": GH_H}
    with open(os.path.join(d, "plane.info"), "w") as f:
        for name in NBR:
            f.write(f"{name} {W} {h[name]} {PITCH}\n")
    return d

# ------------------------------------------------------- golden implementation
def clamp(v, lo, hi): return hi if v > hi else (lo if v < lo else v)

def build_resampling_program(source_size, crop_start, crop_size, target_size, b, c):
    def mitf(x, b, c):
        p0 = (6 - 2*b)/6; p2 = (-18 + 12*b + 6*c)/6; p3 = (12 - 9*b - 6*c)/6
        q0 = (8*b + 24*c)/6; q1 = (-12*b - 48*c)/6; q2 = (6*b + 30*c)/6; q3 = (-b - 6*c)/6
        x = abs(x)
        return (p0 + x*x*(p2 + x*p3)) if x < 1 else ((q0 + x*(q1 + x*(q2 + x*q3))) if x < 2 else 0.0)
    support = 2.0
    filter_scale = target_size / crop_size
    filter_step = min(filter_scale, 1.0)
    filter_support = support / filter_step
    fir = int(math.ceil(filter_support * 2))
    off, coef = [], []
    pos_step = crop_size / target_size
    pos = crop_start if fir == 1 else crop_start + ((crop_size - target_size) / (target_size * 2))
    for i in range(target_size):
        end_pos = int(pos + filter_support)
        if end_pos > source_size - 1: end_pos = source_size - 1
        start_pos = end_pos - fir + 1
        if start_pos < 0: start_pos = 0
        off.append(start_pos)
        ok_pos = clamp(pos, 0.0, float(source_size - 1))
        # naive left-to-right accumulation: builtins.sum() is Neumaier
        # compensated on CPython >= 3.12 and would diverge from the C++
        # mirror's sequential total in the last ulp (the coefs feed a
        # bit-exact compare).
        total = 0.0
        for j in range(fir):
            total += mitf((start_pos + j - ok_pos) * filter_step, b, c)
        if total == 0.0: total = 1.0
        value = 0.0
        row = []
        for k in range(fir):
            nv = value + mitf((start_pos + k - ok_pos) * filter_step, b, c) / total
            row.append(nv - value); value = nv
        coef.append(row)
        pos += pos_step
    return fir, off, coef

def build_gaussian_program(source_size, crop_start, crop_size, target_size, p):
    # NOTE: math.pow delegates to the C library pow (the same call the C++
    # mirror makes) — bit-identical on the same host libm. ** is avoided so
    # no interpreter fast path can interpose. Same cross-libm caveat as the
    # mirror applies (host-computed, float-rounded coefs).
    param = clamp(p, 0.1, 100.0)
    def gf(x):
        pp = param * 0.1
        return math.pow(2.0, -pp * x * x)
    support = 4.0
    filter_scale = target_size / crop_size
    filter_step = min(filter_scale, 1.0)
    filter_support = support / filter_step
    fir = int(math.ceil(filter_support * 2))
    off, coef = [], []
    pos_step = crop_size / target_size
    pos = crop_start if fir == 1 else crop_start + ((crop_size - target_size) / (target_size * 2))
    for i in range(target_size):
        end_pos = int(pos + filter_support)
        if end_pos > source_size - 1: end_pos = source_size - 1
        start_pos = end_pos - fir + 1
        if start_pos < 0: start_pos = 0
        off.append(start_pos)
        ok_pos = clamp(pos, 0.0, float(source_size - 1))
        # naive accumulation (see build_resampling_program note re sum())
        total = 0.0
        for j in range(fir):
            total += gf((start_pos + j - ok_pos) * filter_step)
        if total == 0.0: total = 1.0
        value = 0.0
        row = []
        for k in range(fir):
            nv = value + gf((start_pos + k - ok_pos) * filter_step) / total
            row.append(nv - value); value = nv
        coef.append(row)
        pos += pos_step
    return fir, off, coef

def kernel_makediff(a, b, w, h, mode, range_half, maxval):
    out = [0]*(PITCH*h)
    for y in range(h):
        for x in range(w):
            i = y*PITCH+x
            va, vb = a[i], b[i]
            v = (va - vb + range_half) if mode == 0 else (va + vb - range_half)
            out[i] = clamp(v, 0, maxval)
    return out

def kernel_rg_box3x3(src, w, h, mode, maxval):
    out = [0]*(PITCH*h)
    for y in range(h):
        for x in range(w):
            i = y*PITCH+x
            s = src[i]; v = s
            if 1 <= x < w-1 and 1 <= y < h-1:
                g = lambda dx, dy: src[(y+dy)*PITCH + (x+dx)]
                ul,uc,ur = g(-1,-1),g(0,-1),g(1,-1)
                ml,     mr = g(-1,0),          g(1,0)
                ll,lc,lr = g(-1,1),g(0,1),g(1,1)
                if mode == 20:
                    vt = (ul+uc+ur + ml+s+mr + ll+lc+lr + 4)//9
                else:
                    h0=ul+2*uc+ur; h1=ml+2*s+mr; h2=ll+2*lc+lr
                    vt=(h0+2*h1+h2+8)>>4
                v = clamp(vt, 0, maxval)
            out[i] = v
    return out

def _sort8(a):
    def cas(i,j):
        if a[i]>a[j]: a[i],a[j]=a[j],a[i]
    cas(0,1);cas(2,3);cas(4,5);cas(6,7);cas(0,2);cas(1,3);cas(4,6);cas(5,7)
    cas(1,2);cas(5,6);cas(0,4);cas(1,5);cas(2,6);cas(3,7);cas(2,4);cas(3,5)
    cas(1,2);cas(3,4);cas(5,6)

def kernel_removegrain(src, w, h, n, maxval):
    out=[0]*(PITCH*h)
    for y in range(h):
        for x in range(w):
            i=y*PITCH+x; s=src[i]; v=s
            if 1<=x<w-1 and 1<=y<h-1:
                a=[src[(y-1)*PITCH+(x-1)],src[(y-1)*PITCH+x],src[(y-1)*PITCH+(x+1)],
                   src[y*PITCH+(x-1)],                src[y*PITCH+(x+1)],
                   src[(y+1)*PITCH+(x-1)],src[(y+1)*PITCH+x],src[(y+1)*PITCH+(x+1)]]
                _sort8(a); v=clamp(s,a[n-1],a[7-(n-1)])
            out[i]=v
    return out

def _sort9(a):
    def cas(i,j):
        if a[i]>a[j]: a[i],a[j]=a[j],a[i]
    cas(0,1);cas(3,4);cas(6,7);cas(1,2);cas(4,5);cas(7,8);cas(0,1);cas(3,4)
    cas(6,7);cas(0,3);cas(1,4);cas(2,5);cas(3,6);cas(4,7);cas(5,8);cas(0,3)
    cas(1,4);cas(2,5);cas(1,3);cas(5,7);cas(2,6);cas(4,6);cas(2,4);cas(2,3)
    cas(5,6)

def kernel_repair(src, ref, w, h, n, maxval):
    out=[0]*(PITCH*h)
    for y in range(h):
        for x in range(w):
            i=y*PITCH+x; s=src[i]; v=s
            if 1<=x<w-1 and 1<=y<h-1:
                a=[ref[(y-1)*PITCH+(x-1)],ref[(y-1)*PITCH+x],ref[(y-1)*PITCH+(x+1)],
                   ref[y*PITCH+(x-1)], s, ref[y*PITCH+(x+1)],
                   ref[(y+1)*PITCH+(x-1)],ref[(y+1)*PITCH+x],ref[(y+1)*PITCH+(x+1)]]
                _sort9(a); v=clamp(s,a[n-1],a[8-(n-1)])
            out[i]=v
    return out

def kernel_vclean(src, w, h):
    out=[0]*(PITCH*h)
    for y in range(h):
        for x in range(w):
            i=y*PITCH+x; b=src[i]; v=b
            if 1<=y<h-1:
                a=src[(y-1)*PITCH+x]; c=src[(y+1)*PITCH+x]
                v=min(max(min(a,b),c),max(a,b))
            out[i]=v
    return out

def kernel_tfr(src, w, h, is_uv, maxval):
    out=[0]*(PITCH*h)
    for y in range(h):
        for x in range(w):
            i=y*PITCH+x; s=float(src[i])
            d=(s-128.0)*(128.0/112.0)+128.0 if is_uv else (s-16.0)*(255.0/219.0)
            out[i]=int(clamp(d+0.5,0.0,float(maxval)))
    return out

def kernel_merge(a, b, w, h, weight, maxval):
    inv=32767-weight
    out=[0]*(PITCH*h)
    for y in range(h):
        for x in range(w):
            i=y*PITCH+x
            v=(a[i]*inv + b[i]*weight + 16384) >> 15
            out[i]=clamp(v,0,maxval)
    return out

def kernel_resample_v(field, w, fir, off, coef, maxval, h_in, h_out):
    out=[0]*(PITCH*h_out)
    for y in range(h_out):
        begin=off[y]
        for x in range(w):
            acc=0.0
            for i in range(fir):
                acc += float(field[(begin+i)*PITCH+x])*coef[y][i]
            acc=clamp(acc,0.0,float(maxval))
            out[y*PITCH+x]=int(acc+0.5)
    return out

def kernel_resample_h(src, w, h, fir, off, coef, maxval):
    out=[0]*(PITCH*h)
    for y in range(h):
        for x in range(w):
            begin=off[x]; acc=0.0
            for i in range(fir):
                acc += float(src[y*PITCH+(begin+i)])*coef[x][i]
            acc=clamp(acc,0.0,float(maxval))
            out[y*PITCH+x]=int(acc+0.5)
    return out

def kernel_box5(src, w, h, is_min, maxval):
    out=[0]*(PITCH*h)
    for y in range(h):
        for x in range(w):
            i=y*PITCH+x; v2=src[i]
            v0=src[(y-2)*PITCH+x] if y-2>=0 else v2
            v1=src[(y-1)*PITCH+x] if y-1>=0 else v2
            v3=src[(y+1)*PITCH+x] if y+1<h   else v2
            v4=src[(y+2)*PITCH+x] if y+2<h   else v2
            out[i]=min(min(min(v0,v1),min(v2,v3)),v4) if is_min else max(max(max(v0,v1),max(v2,v3)),v4)
    return out

def kernel_logic(a,b,w,h,is_min):
    out=[0]*(PITCH*h)
    for y in range(h):
        for x in range(w):
            i=y*PITCH+x
            out[i]=min(a[i],b[i]) if is_min else max(a[i],b[i])
    return out

def kernel_vresharpen(src,w,h):
    out=[0]*(PITCH*h)
    for y in range(h):
        for x in range(w):
            i=y*PITCH+x; v1=src[i]
            v0=src[(y-1)*PITCH+x] if y!=0 else v1
            v2=src[(y+1)*PITCH+x] if y!=h-1 else v1
            out[i]=(min(v0,min(v1,v2))+max(v0,max(v1,v2))+1)>>1
    return out

def kernel_resharpen(a,b,w,h,sharpAdj,maxval):
    out=[0]*(PITCH*h)
    for y in range(h):
        for x in range(w):
            i=y*PITCH+x
            lut=int(clamp(a[i]+(a[i]-b[i])*sharpAdj+0.5,0.0,float(maxval)))
            out[i]=lut
    return out

def kernel_limitos(src,ref,cb,cf,w,h,osv):
    out=[0]*(PITCH*h)
    for y in range(h):
        for x in range(w):
            i=y*PITCH+x; s=src[i];r=ref[i];b=cb[i];f=cf[i]
            mn=min(r,min(b,f)); mx=max(r,max(b,f))
            out[i]=clamp(s,mn-osv,mx+osv)
    return out

def kernel_lossless(xa,ya,w,h,half,maxval):
    out=[0]*(PITCH*h)
    for y in range(h):
        for x in range(w):
            i=y*PITCH+x; vx=float(xa[i]); vy=float(ya[i])
            if (vx-half)*(vy-half)<0: v=half
            elif abs(vx-half)<abs(vy-half): v=vx
            else: v=vy
            out[i]=int(clamp(v,0.0,maxval))
    return out

def kernel_tweak(rep,bob,blr,w,h,scale,invscale,maxval):
    out=[0]*(PITCH*h)
    for y in range(h):
        for x in range(w):
            i=y*PITCH+x
            repair=float(rep[i])*invscale; bobbed=float(bob[i])*invscale; blur=float(blr[i])*invscale
            tweaked=clamp(bobbed,repair-3,repair+3)
            if (blur+7)<tweaked: ret=blur+2
            elif (blur-7)>tweaked: ret=blur-2
            else: ret=(blur*51+tweaked*49)*(1.0/100.0)
            out[i]=int(clamp(ret*scale+0.5,0.0,float(maxval)))
    return out

def kernel_erroradjust(src,mt,w,h,errorAdj,maxval):
    out=[0]*(PITCH*h)
    for y in range(h):
        for x in range(w):
            i=y*PITCH+x
            lut=int(clamp(src[i]*(errorAdj+1)-mt[i]*errorAdj+0.5,0.0,float(maxval)))
            out[i]=lut
    return out

def kernel_bobshimmer(src,diff,c1,c2,w,h,scale,maxval):
    hsh=128<<scale
    out=[0]*(PITCH*h)
    for y in range(h):
        for x in range(w):
            i=y*PITCH+x; s=src[i]; df=diff[i]; c1v=c1[i]; c2v=c2[i]
            df = df if df<(129<<scale) else (hsh if c1v<hsh else c1v)
            df = df if df>(127<<scale) else (hsh if c2v>hsh else c2v)
            out[i]=clamp(s+df-hsh,0,maxval)
    return out

def kernel_soften1(src,r0,r1,w,h,sc0,sc1,maxval):
    out=[0]*(PITCH*h)
    for y in range(h):
        for x in range(w):
            i=y*PITCH+x; s=src[i]; a0=s if sc0 else r0[i]; a1=s if sc1 else r1[i]
            out[i]=clamp((a0+2*s+a1+2)>>2,0,maxval)
    return out

def kernel_soften2(src,r0,r1,r2,r3,w,h,sc0,sc1,sc2,sc3,maxval):
    out=[0]*(PITCH*h)
    for y in range(h):
        for x in range(w):
            i=y*PITCH+x; s=src[i]
            a0=s if sc0 else r0[i]; a1=s if sc1 else r1[i]
            a2=s if sc2 else r2[i]; a3=s if sc3 else r3[i]
            out[i]=clamp((a2+4*a0+6*s+4*a1+a3+4)>>4,0,maxval)
    return out

def kernel_weave(top,bottom,w,h2):
    # h2 = number of input rows per field (each field h2 rows); out = 2*h2
    out=[0]*(PITCH*2*h2)
    for y in range(h2):
        for x in range(w):
            out[(2*y+0)*PITCH+x]=top[y*PITCH+x]
            out[(2*y+1)*PITCH+x]=bottom[y*PITCH+x]
    return out

def kernel_copy(src,w,h):
    return list(src)

def kernel_plane_sad(a,b,w,h):
    total=0
    for y in range(h):
        for x in range(w):
            i=y*PITCH+x
            total += abs(a[i]-b[i])
    return total

def kernel_wiener_v(src,w,h,maxval):
    out=[0]*(PITCH*h)
    for y in range(h):
        for x in range(w):
            P=lambda dy: src[y*PITCH+x+dy*PITCH]
            if y<2: v=(P(0)+P(1)+1)>>1
            elif y<h-4:
                num=P(-2)+((-P(-1)+4*P(0)+4*P(1)-P(2))*5)+P(3)+16
                v=clamp(num>>5,0,maxval)
            elif y<h-1: v=(P(0)+P(1)+1)>>1
            else: v=P(0)
            out[y*PITCH+x]=v
    return out

def kernel_wiener_h(src,w,h,maxval):
    out=[0]*(PITCH*h)
    for y in range(h):
        for x in range(w):
            P=lambda dx: src[y*PITCH+(x+dx)]
            if x<2: v=(P(0)+P(1)+1)>>1
            elif x<w-4:
                num=P(-2)+((-P(-1)+4*P(0)+4*P(1)-P(2))*5)+P(3)+16
                v=clamp(num>>5,0,maxval)
            elif x<w-1: v=(P(0)+P(1)+1)>>1
            else: v=P(0)
            out[y*PITCH+x]=v
    return out

# ---------------------------------------------------------------- comparison
def read_plane(path, w, h, bits):
    with open(path,"rb") as f: raw=f.read()
    fmt="B" if bits==8 else "H"
    n=PITCH*h
    return list(struct.unpack(fmt*n, raw))

def run_bits(bits):
    maxval=255 if bits==8 else 65535
    in_dir=make_inputs(bits)
    out_dir=os.path.join(WORK, f"ref_{bits}")
    os.makedirs(out_dir, exist_ok=True)
    subprocess.run([REF_BIN, in_dir, out_dir, str(bits)], check=True)
    # reload inputs
    def ld(name,h):
        return read_plane(os.path.join(in_dir,name+".raw"), W, h, bits)
    a=ld("a",A_H); b=ld("b",A_H); c=ld("c",REF_H); ref=ld("ref",REF_H)
    n2=ld("n2",A_H); n1=ld("n1",A_H); p1=ld("p1",A_H); p2=ld("p2",A_H)
    field=ld("field",FIELD_H)
    gsrc=ld("gsrc",GH_H)
    RANGE=1<<(bits-1)
    out_h={}      # name -> output height
    _gold={}
    def setg(name,h,golden): out_h[name]=h; _gold[name]=golden
    setg("makediff",A_H,kernel_makediff(a,b,W,A_H,0,RANGE,maxval))
    setg("adddiff",A_H,kernel_makediff(a,b,W,A_H,1,RANGE,maxval))
    setg("rg11",A_H,kernel_rg_box3x3(a,W,A_H,11,maxval))
    setg("rg20",A_H,kernel_rg_box3x3(a,W,A_H,20,maxval))
    for n in range(1,5):
        setg(f"rgclip{n}",A_H,kernel_removegrain(a,W,A_H,n,maxval))
        setg(f"repair{n}",A_H,kernel_repair(a,ref,W,A_H,n,maxval))
    setg("vclean",A_H,kernel_vclean(a,W,A_H))
    setg("tfr_y",A_H,kernel_tfr(a,W,A_H,0,maxval))
    setg("tfr_uv",A_H,kernel_tfr(a,W,A_H,1,maxval))
    setg("merge",A_H,kernel_merge(a,b,W,A_H,int(0.5*32767),maxval))

    fir,off,coef=build_resampling_program(5,0.25,5,10,0.0,0.5)
    setg("resample_v",10,kernel_resample_v(field,W,fir,off,coef,maxval,FIELD_H,10))
    firh,offh,coefh=build_resampling_program(16,0,16,16,0.0,0.5)
    setg("resample_h",A_H,kernel_resample_h(a,W,A_H,firh,offh,coefh,maxval))
    # KGaussResize gaussian programs (upstream: same-size, crop + 0.0001, p = 30;
    # fir 8/9 dispatch asserted with the table compares below).
    gV9=build_gaussian_program(GH_H,0,GH_H+0.0001,GH_H,30.0)
    gH9=build_gaussian_program(W,0,W+0.0001,W,30.0)
    gV8=build_gaussian_program(GH_H,0,GH_H,GH_H,30.0)
    gH8=build_gaussian_program(W,0,W,W,30.0)
    gV9_p01=build_gaussian_program(GH_H,0,GH_H+0.0001,GH_H,0.1)
    gV9_p100=build_gaussian_program(GH_H,0,GH_H+0.0001,GH_H,100.0)
    gV9_neg5=build_gaussian_program(GH_H,0,GH_H+0.0001,GH_H,-5.0)
    gV9_p250=build_gaussian_program(GH_H,0,GH_H+0.0001,GH_H,250.0)
    gprogs={"v9":gV9,"h9":gH9,"v8":gV8,"h8":gH8,"v9_p01":gV9_p01,
            "v9_p100":gV9_p100,"v9_neg5":gV9_neg5,"v9_p250":gV9_p250}
    setg("gres_v9",GH_H,kernel_resample_v(gsrc,W,gV9[0],gV9[1],gV9[2],maxval,GH_H,GH_H))
    setg("gres_h9",GH_H,kernel_resample_h(gsrc,W,GH_H,gH9[0],gH9[1],gH9[2],maxval))
    setg("gres_v8",GH_H,kernel_resample_v(gsrc,W,gV8[0],gV8[1],gV8[2],maxval,GH_H,GH_H))
    setg("gres_h8",GH_H,kernel_resample_h(gsrc,W,GH_H,gH8[0],gH8[1],gH8[2],maxval))
    setg("box5min",A_H,kernel_box5(a,W,A_H,1,maxval))
    setg("box5max",A_H,kernel_box5(a,W,A_H,0,maxval))
    setg("logicmin",A_H,kernel_logic(a,b,W,A_H,1))
    setg("logicmax",A_H,kernel_logic(a,b,W,A_H,0))
    setg("vresharpen",A_H,kernel_vresharpen(a,W,A_H))
    setg("resharpen",A_H,kernel_resharpen(a,b,W,A_H,0.2,maxval))
    setg("limitos",A_H,kernel_limitos(a,ref,b,c,W,A_H,3))
    setg("lossless",A_H,kernel_lossless(a,b,W,A_H,float(RANGE),float(maxval)))
    setg("tweak",A_H,kernel_tweak(a,b,ref,W,A_H,float(1<<(bits-8)),1.0/float(1<<(bits-8)),maxval))
    setg("erroradj",A_H,kernel_erroradjust(a,b,W,A_H,0.05,maxval))
    setg("bobshimmer",A_H,kernel_bobshimmer(a,b,ref,c,W,A_H,bits-8,maxval))
    setg("soften1",A_H,kernel_soften1(a,n1,p1,W,A_H,1,0,maxval))
    setg("soften2",A_H,kernel_soften2(a,n2,n1,p1,p2,W,A_H,0,1,0,1,maxval))
    setg("weave",2*A_H,kernel_weave(n1,p1,W,A_H))
    setg("copy",A_H,kernel_copy(a,W,A_H))
    setg("wiener_v",A_H,kernel_wiener_v(a,W,A_H,maxval))
    setg("wiener_h",A_H,kernel_wiener_h(a,W,A_H,maxval))

    ok=True
    # plane SAD is a scalar: check separately
    sad_ref=[int(l) for l in open(os.path.join(out_dir,"plane_sad.txt"))][0]
    sad_gold=kernel_plane_sad(a,b,W,A_H)
    if sad_ref!=sad_gold: ok=False; print(f"  MISMATCH plane_sad: {sad_ref} vs {sad_gold}")
    for N in (1,6,12,18,24,30,61,63,64,65,255,256,257,2048):
        vals=[float(l) for l in open(os.path.join(out_dir,f"init_sad_{N}.txt"))]
        if len(vals)!=N or any(v!=0.0 for v in vals):
            ok=False; print(f"  MISMATCH init_sad N={N}: got {len(vals)} vals")

    for name,golden in _gold.items():
        got=read_plane(os.path.join(out_dir,name+".raw"), W, out_h[name], bits)
        if got!=golden:
            ok=False
            diffs=[i for i in range(len(golden)) if got[i]!=golden[i]][:5]
            print(f"  MISMATCH {name}: {len(diffs)}+ diffs, e.g. {diffs}")
    # compare resample program tables (vertical)
    if bits==8:
        coff=[int(l) for l in open(os.path.join(out_dir,"prog_offset.txt"))]
        cf=[float(l) for l in open(os.path.join(out_dir,"prog_coef.txt"))]
        if coff!=off: ok=False; print("  MISMATCH prog_offset")
        cf2=[cf[y*fir+i] for y in range(10) for i in range(fir)]
        gold=[c for row in coef for c in row]
        # cross-libm/rounding tolerance (same as gprog below): relative 1e-9
        if len(cf2)!=len(gold) or any(abs(a-b) > 1e-9*max(1.0,abs(a),abs(b)) for a,b in zip(cf2,gold)):
            ok=False; print("  MISMATCH prog_coef")
        # KGaussResize gaussian program tables (+ fir dispatch + clamp probes)
        for tag,(gfir,goff,gcoef) in gprogs.items():
            want=8 if tag in ("v8","h8") else 9
            if gfir!=want: ok=False; print(f"  MISMATCH gprog_{tag}: fir {gfir} != {want}")
            roff=[int(l) for l in open(os.path.join(out_dir,f"gprog_{tag}_offset.txt"))]
            if roff!=goff: ok=False; print(f"  MISMATCH gprog_{tag}_offset")
            rcf=[float(l) for l in open(os.path.join(out_dir,f"gprog_{tag}_coef.txt"))]
            flat=[c for row in gcoef for c in row]
            # cross-libm tolerance: the mirror and this golden call C pow on
            # DIFFERENT libms on some rigs (here: mingw-w64 pow vs the
            # interpreter's CRT pow), which may differ in the last double
            # ulp; for tiny coefs (1e-9..1e-15) the absolute ulp count is
            # huge even though the relative error is ~1e-16, so use a
            # relative-error bound instead of a raw ulp bound.  The device
            # only consumes the float-rounded coefs; the f32 word table
            # below allows 1 ulp (the float rounding of a 1-ulp-off double
            # can cross a float boundary) and ignores sub-1e-6 magnitudes
            # (they round to 0 at 8-bit anyway).
            def coef_close(a, b):
                if a == b:
                    return True
                return abs(a - b) <= 1e-9 * max(1.0, abs(a), abs(b))
            def ulpdiff32(a, b):
                ia = struct.unpack('<i', struct.pack('<f', a))[0]
                ib = struct.unpack('<i', struct.pack('<f', b))[0]
                return abs(ia - ib)
            if len(rcf) != len(flat) or any(not coef_close(a, b) for a, b in zip(rcf, flat)):
                ok = False; print(f"  MISMATCH gprog_{tag}_coef")
            rf32 = [l.strip() for l in open(os.path.join(out_dir, f"gprog_{tag}_coef_f32.txt"))]
            gf32 = [struct.pack(">f", c).hex() for c in flat]  # BE bytes == %08x word
            bad = False
            for a_hex, b_hex in zip(gf32, rf32):
                if a_hex == b_hex:
                    continue
                av = struct.unpack('>f', bytes.fromhex(a_hex))[0]
                bv = struct.unpack('>f', bytes.fromhex(b_hex))[0]
                if abs(av) < 1e-6 and abs(bv) < 1e-6:
                    continue  # both round to ~0 at 8-bit
                if ulpdiff32(av, bv) > 1:
                    bad = True
                    break
            if bad:
                ok = False; print(f"  MISMATCH gprog_{tag}_coef_f32")
        # clamp probes: out-of-range p must reproduce the clamped edge tables
        for a,b in (("v9_neg5","v9_p01"),("v9_p250","v9_p100")):
            for suf in ("offset.txt","coef.txt","coef_f32.txt"):
                la=open(os.path.join(out_dir,f"gprog_{a}_{suf}")).read()
                lb=open(os.path.join(out_dir,f"gprog_{b}_{suf}")).read()
                if la!=lb: ok=False; print(f"  MISMATCH gprog clamp {a} != {b} ({suf})")
    print(f"[bitdepth {bits}] {'PASS' if ok else 'FAIL'}")
    return ok

if __name__=="__main__":
    # build C++ ref
    subprocess.run(["g++","-O2","-std=gnu++17","-w","-Wl,--allow-multiple-definition",
                    os.path.join(REPO,"sim","ktgmc_cpu_ref.cpp"),
                    "-o",REF_BIN], check=True)
    allok=True
    for b in (8,16):
        allok &= run_bits(b)
    sys.exit(0 if allok else 1)
