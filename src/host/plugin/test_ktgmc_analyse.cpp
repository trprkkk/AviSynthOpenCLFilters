// test_ktgmc_analyse.cpp -headless verification of the KMAnalyseOCL
// AviSynth+ plugin (ninth host plugin / second motion-stage plugin of the
// OpenCL port), chained behind the real KMSuperOCL plugin exactly like a
// script would:  KMSuperOCL(src,...).KMAnalyseOCL(...)
//
// REFERENCE CHOICE (stated, per the task): the reference is computed HERE, in
// this exe -an in-exe CPU implementation of the PROVEN multi-level
// GroupOfPlanes host loop (cuda_mv/KMANALYSE_HOST_CONTRACT.md -10):
//   * the super pyramid planes come from the already-verified CPU super
//     reference (transcribed from test_ktgmc_super.cpp -33/33 cases there,
//     and the mirror of the wiener/RB2B halves is pinned by
//     python/run_validation.py + python/run_mv_rb2b_pad.py);
//   * the per-level Search call is the CPU mirror of kt_search_column +
//     kt_prepare_search + kt_calc_all_sad (adapted from
//     sim/ktgmc_search_ref.cpp, which python/run_mv_search.py validated
//     bit-exact vs the CUDA twin on 9 configs);
//   * the global-MV chain (kt_most_freq_mv staged-fold tie-break +
//     kt_mean_global_mv) and kt_interpolate_prediction are transcribed from
//     src/opencl/ktgmc/kernels/ktgmc_motion.cl (ALG-VERIFIED by
//     python/run_motion_core.py / run_mv_mean.py / run_mv_interp.py and
//     CUDA-twin-measured by rig/dev_ktgmc_analyse.py, 39/39).
// Every VECTOR (x, y, sad) of the produced MV clip -which contains EVERY
// level's output region, coarsest level first (MV.cpp:2333-2364/2664-2668) -// is compared bit-exactly against the reference, plus the KMVIsValid frame
// property on computed and WriteDefault frames.
//
// Coverage: YV12/YUV420P16 x pel {1,2} x chroma {on,off} x fine search
// {HEX2SEARCH default, EXHAUSTIVE} on the 3-level 64x48 default geometry,
// the 2-level 32x24 geometry (the twin-measured level count), the parameter
// defaults (blksize 16), backward search, the alias registration, a real
// script string, the WriteDefault out-of-range frame, and the constructor
// throws (batch!=1, partial, non-super input).
//
// Build: see build_host_analyse.bat.  Usage:
//   test_ktgmc_analyse.exe [analyse_dll] [super_dll] [plugins_dir]
#define NOMINMAX
#include <windows.h>

#include "avisynth.h"
#include <algorithm>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

typedef IScriptEnvironment* (__stdcall* CSE_t)(int);
const AVS_Linkage* AVS_linkage = nullptr;

static LONG WINAPI eh(EXCEPTION_POINTERS* ep) {
    HMODULE m = nullptr;
    GetModuleHandleExA(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                       (LPCSTR)ep->ExceptionRecord->ExceptionAddress, &m);
    char path[MAX_PATH] = "?";
    GetModuleFileNameA(m, path, MAX_PATH);
    fprintf(stderr, "EXC %08lX at %p in %s\n", ep->ExceptionRecord->ExceptionCode,
            ep->ExceptionRecord->ExceptionAddress, path);
    fflush(stderr);
    return EXCEPTION_CONTINUE_SEARCH;
}

static int g_failures = 0;
static int g_cases = 0;
static int g_vec_checks = 0;

// ===========================================================================
// MV.cpp geometry helpers + CPU super reference (from test_ktgmc_super.cpp)
// ===========================================================================
static int PlaneHeightLuma(int src_height, int level, int yRatioUV, int vpad) {
    int height = src_height;
    for (int i = 1; i <= level; i++)
        height = vpad >= yRatioUV ? ((height / yRatioUV + 1) / 2) * yRatioUV
                                  : ((height / yRatioUV) / 2) * yRatioUV;
    return height;
}
static int PlaneWidthLuma(int src_width, int level, int xRatioUV, int hpad) {
    int width = src_width;
    for (int i = 1; i <= level; i++)
        width = hpad >= xRatioUV ? ((width / xRatioUV + 1) / 2) * xRatioUV
                                 : ((width / xRatioUV) / 2) * xRatioUV;
    return width;
}
static unsigned int PlaneSuperOffset(bool chroma, int src_height, int level, int pel,
                                     int vpad, int plane_pitch, int yRatioUV) {
    int height = src_height;
    unsigned int offset;
    if (level == 0) {
        offset = 0;
    } else {
        offset = (unsigned)pel * pel * plane_pitch * (src_height + vpad * 2);
        for (int i = 1; i < level; i++) {
            height = chroma
                ? PlaneHeightLuma(src_height * yRatioUV, i, yRatioUV, vpad * yRatioUV) / yRatioUV
                : PlaneHeightLuma(src_height, i, yRatioUV, vpad);
            offset += (unsigned)plane_pitch * (height + vpad * 2);
        }
    }
    return offset;
}

struct SuperGeom {
    int origW, origH, nHPad, nVPad, nPel, nLevels, xRatioUV, yRatioUV, logUVx, logUVy;
    int nSuperWidth, nSuperHeight;
    bool chroma;
};

static SuperGeom makeGeom(const VideoInfo& vi, int nHPad, int nVPad, int nPel,
                          int nLevels, bool chroma) {
    SuperGeom g;
    g.origW = vi.width; g.origH = vi.height;
    g.nHPad = nHPad; g.nVPad = nVPad; g.nPel = nPel; g.chroma = chroma;
    g.xRatioUV = 1 << vi.GetPlaneWidthSubsampling(PLANAR_U);
    g.yRatioUV = 1 << vi.GetPlaneHeightSubsampling(PLANAR_U);
    g.logUVx = vi.GetPlaneWidthSubsampling(PLANAR_U);
    g.logUVy = vi.GetPlaneHeightSubsampling(PLANAR_U);
    int nLevelsMax = 0;
    while (PlaneHeightLuma(g.origH, nLevelsMax, g.yRatioUV, nVPad) >= g.yRatioUV * 2 &&
           PlaneWidthLuma(g.origW, nLevelsMax, g.xRatioUV, nHPad) >= g.xRatioUV * 2)
        nLevelsMax++;
    if (nLevels <= 0 || nLevels > nLevelsMax) nLevels = nLevelsMax;
    if (nLevels < 1) nLevels = 1;
    g.nLevels = nLevels;
    g.nSuperWidth = g.origW + 2 * nHPad;
    g.nSuperHeight = (int)(PlaneSuperOffset(false, g.origH, nLevels, nPel, nVPad,
                                            g.nSuperWidth, g.yRatioUV) / (unsigned)g.nSuperWidth);
    if (g.yRatioUV == 2 && (g.nSuperHeight & 1)) g.nSuperHeight++;
    return g;
}

static int mirrorIdx(int c, int n) {
    if (c < 0) return -c - 1;
    if (c >= n) return n - (c - n) - 1;
    return c;
}

template <typename PX>
static void buildPlaneRef(const SuperGeom& g, bool isChroma,
                          const std::vector<PX>& src, int srcW, int srcH,
                          std::vector<PX>& plane, int& outPitch, int& outH) {
    const int P = isChroma ? (g.nSuperWidth >> g.logUVx) : g.nSuperWidth;
    const int planeH = isChroma ? (g.nSuperHeight >> g.logUVy) : g.nSuperHeight;
    const int hpad = isChroma ? g.nHPad / g.xRatioUV : g.nHPad;
    const int vpad = isChroma ? g.nVPad / g.yRatioUV : g.nVPad;
    const int H0 = srcH, W0 = srcW;
    const int L = g.nLevels;
    const PX MAXV = sizeof(PX) == 1 ? (PX)255 : (PX)65535;

    std::vector<int> off(L);
    for (int l = 0; l < L; l++)
        off[l] = (int)PlaneSuperOffset(isChroma, H0, l, g.nPel, vpad, P, g.yRatioUV);
    std::vector<std::pair<int,int>> dim(L);
    dim[0] = std::make_pair(W0, H0);
    for (int l = 1; l < L; l++) {
        int lw = PlaneWidthLuma(g.origW, l, g.xRatioUV, g.nHPad);
        int lh = PlaneHeightLuma(g.origH, l, g.yRatioUV, g.nVPad);
        dim[l] = isChroma ? std::make_pair(lw / g.xRatioUV, lh / g.yRatioUV)
                          : std::make_pair(lw, lh);
    }

    plane.assign((size_t)P * planeH, (PX)0);
    auto at = [&](int base, int x, int y) -> PX& { return plane[(size_t)base + x + (size_t)y * P]; };

    // 1. copy_pad (reflection) into level-0 sub-plane 0
    {
        int int0 = off[0] + P * vpad + hpad;
        int extW = W0 + 2 * hpad, extH = H0 + 2 * vpad;
        for (int gy = 0; gy < extH; gy++) {
            int y = gy - vpad;
            for (int gx = 0; gx < extW; gx++) {
                int x = gx - hpad;
                if (x < W0 + hpad && y < H0 + vpad) {
                    int sx = mirrorIdx(x, W0), sy = mirrorIdx(y, H0);
                    at(int0, x, y) = src[(size_t)sx + (size_t)sy * srcW];
                }
            }
        }
    }

    // 2. Refine (nPel==2): wiener
    if (g.nPel == 2) {
        int extW = W0 + 2 * hpad, extH = H0 + 2 * vpad;
        int ext = P * extH;
        auto wienerV = [&](int dstOff, int srcOff) {
            for (int y = 0; y < extH; y++) for (int x = 0; x < extW; x++) {
                auto S = [&](int dy) { return (int)at(srcOff, x, y + dy); };
                int v;
                if (y < 2) v = (S(0) + S(1) + 1) >> 1;
                else if (y < extH - 4) {
                    int num = S(-2) + ((-S(-1) + 4 * S(0) + 4 * S(1) - S(2)) * 5) + S(3) + 16;
                    v = (int)std::min<int>(MAXV, std::max(0, num >> 5));
                } else if (y < extH - 1) v = (S(0) + S(1) + 1) >> 1;
                else v = S(0);
                at(dstOff, x, y) = (PX)v;
            }
        };
        auto wienerH = [&](int dstOff, int srcOff) {
            for (int y = 0; y < extH; y++) for (int x = 0; x < extW; x++) {
                auto S = [&](int dx) { return (int)at(srcOff, x + dx, y); };
                int v;
                if (x < 2) v = (S(0) + S(1) + 1) >> 1;
                else if (x < extW - 4) {
                    int num = S(-2) + ((-S(-1) + 4 * S(0) + 4 * S(1) - S(2)) * 5) + S(3) + 16;
                    v = (int)std::min<int>(MAXV, std::max(0, num >> 5));
                } else if (x < extW - 1) v = (S(0) + S(1) + 1) >> 1;
                else v = S(0);
                at(dstOff, x, y) = (PX)v;
            }
        };
        wienerH(off[0] + 1 * ext, off[0]);
        wienerV(off[0] + 2 * ext, off[0]);
        wienerH(off[0] + 3 * ext, off[0] + 2 * ext);
    }

    // 3. coarse levels: fused RB2B reduce + pad
    for (int l = 0; l + 1 < L; l++) {
        int Wd = dim[l + 1].first, Hd = dim[l + 1].second;
        int srcInt = off[l] + P * vpad + hpad;
        int dstInt = off[l + 1] + P * vpad + hpad;
        int extW = Wd + 2 * hpad, extH = Hd + 2 * vpad;
        for (int gy = 0; gy < extH; gy++) {
            int dsty = gy - vpad;
            for (int gx = 0; gx < extW; gx++) {
                int dstx = gx - hpad;
                if (!(dstx < Wd + hpad && dsty < Hd + vpad)) continue;
                int xmul0 = 0, xmul1 = 4, srcx;
                if (dstx <= 0) srcx = 0;
                else if (dstx >= Wd - 1) srcx = (Wd - 1) * 2;
                else { srcx = dstx * 2; xmul0 = 1; xmul1 = 3; }
                int ymul0 = 0, ymul1 = 4, srcy;
                if (dsty <= 0) srcy = 0;
                else if (dsty >= Hd - 1) srcy = (Hd - 1) * 2;
                else { srcy = dsty * 2; ymul0 = 1; ymul1 = 3; }
                int sum = 0;
                for (int j = -1; j <= 2; j++) {
                    int ymul = (j == 0 || j == 1) ? ymul1 : ymul0;
                    if (ymul <= 0) continue;
                    for (int i = -1; i <= 2; i++) {
                        int xmul = (i == 0 || i == 1) ? xmul1 : xmul0;
                        if (xmul <= 0) continue;
                        sum += (int)at(srcInt, srcx + i, srcy + j) * ymul * xmul;
                    }
                }
                sum = (sum + 32) / 64;
                at(dstInt, dstx, dsty) = (PX)sum;
            }
        }
    }

    outPitch = P; outH = planeH;
}

// ===========================================================================
// deterministic source clips (from test_ktgmc_super.cpp)
// ===========================================================================
static uint32_t mix32(uint32_t a) {
    a ^= a >> 16; a *= 0x7feb352du;
    a ^= a >> 15; a *= 0x846ca68bu;
    a ^= a >> 16;
    return a;
}

template <typename PX>
static void fillPlane(uint8_t* dstp, int pitch, int w, int h, uint32_t salt, int maxv) {
    for (int y = 0; y < h; ++y) {
        PX* row = (PX*)(dstp + (size_t)y * pitch);
        for (int x = 0; x < w; ++x) {
            int v = (int)(((int64_t)x * (maxv - 40)) / (w > 1 ? w - 1 : 1)
                          + ((int64_t)y * (maxv - 40)) / (h > 1 ? h - 1 : 1)
                          + 20);
            v += (int)(mix32((uint32_t)x * 17u + (uint32_t)y * 1013u + salt) % 9u) - 4;
            if (v < 0) v = 0;
            if (v > maxv) v = maxv;
            row[x] = (PX)v;
        }
    }
}

class RandSource : public IClip {
    VideoInfo vi_;
    uint32_t salt_;
    int px_;
public:
    RandSource(VideoInfo vi, uint32_t salt) : vi_(vi), salt_(salt), px_(vi.ComponentSize()) {}
    PVideoFrame __stdcall GetFrame(int n, IScriptEnvironment* env) {
        PVideoFrame f = env->NewVideoFrame(vi_);
        const int w = vi_.width, h = vi_.height;
        const int wUV = w >> vi_.GetPlaneWidthSubsampling(PLANAR_U);
        const int hUV = h >> vi_.GetPlaneHeightSubsampling(PLANAR_U);
        const int maxv = (px_ == 1) ? 255 : 65535;
        uint32_t s = salt_ + (uint32_t)n * 7919u;
        if (px_ == 1) {
            fillPlane<uint8_t>(f->GetWritePtr(PLANAR_Y), f->GetPitch(PLANAR_Y), w, h, s, maxv);
            fillPlane<uint8_t>(f->GetWritePtr(PLANAR_U), f->GetPitch(PLANAR_U), wUV, hUV, s + 11u, maxv);
            fillPlane<uint8_t>(f->GetWritePtr(PLANAR_V), f->GetPitch(PLANAR_V), wUV, hUV, s + 23u, maxv);
        } else {
            fillPlane<uint16_t>(f->GetWritePtr(PLANAR_Y), f->GetPitch(PLANAR_Y), w, h, s, maxv);
            fillPlane<uint16_t>(f->GetWritePtr(PLANAR_U), f->GetPitch(PLANAR_U), wUV, hUV, s + 11u, maxv);
            fillPlane<uint16_t>(f->GetWritePtr(PLANAR_V), f->GetPitch(PLANAR_V), wUV, hUV, s + 23u, maxv);
        }
        return f;
    }
    bool __stdcall GetParity(int) { return false; }
    void __stdcall GetAudio(void*, int64_t, int64_t, IScriptEnvironment*) {}
    int __stdcall SetCacheHints(int, int) { return 0; }
    const VideoInfo& __stdcall GetVideoInfo() { return vi_; }
};

// ===========================================================================
// CPU reference of ONE Search call (prepare + ordered column search +
// calc_all_sad) -adapted from sim/ktgmc_search_ref.cpp (the model
// python/run_mv_search.py validated bit-exact vs the CUDA twin on 9 configs)
// ===========================================================================
static int ref_block_offset(int vx, int vy, int nPitch, int nImgPitch, int NPEL) {
    if (NPEL == 1) return vx + vy * nPitch;
    if (NPEL == 2) {
        int sx = vx & 1, sy = vy & 1, si = sx + sy * 2;
        return (vx >> 1) + (vy >> 1) * nPitch + si * nImgPitch;
    }
    int sx = vx & 3, sy = vy & 3, si = sx + sy * 4;
    return (vx >> 2) + (vy >> 2) * nPitch + si * nImgPitch;
}

static int sad_of(const std::vector<int>& srcY, const std::vector<int>& refY,
                  const std::vector<int>* srcU, const std::vector<int>* refU,
                  const std::vector<int>* srcV, const std::vector<int>* refV,
                  int baseY, int baseUV,
                  int offx, int offy, int BS, int chroma, int nPitchY, int nPitchUV,
                  int nImgPitchY, int nImgPitchUV, int NPEL, int vx, int vy) {
    int roff = ref_block_offset(vx, vy, nPitchY, nImgPitchY, NPEL);
    int sad = 0;
    for (int jy = 0; jy < BS; jy++) for (int jx = 0; jx < BS; jx++) {
        int a = srcY[(size_t)baseY + (offx + jx) + (offy + jy) * nPitchY];
        int b = refY[(size_t)baseY + (offx + jx) + (offy + jy) * nPitchY + roff];
        int d = a - b; if (d < 0) d = -d; sad += d;
    }
    if (chroma) {
        int bs2 = BS / 2;
        int bux = offx >> 1, buy = offy >> 1;
        int roffUV = ref_block_offset(vx >> 1, vy >> 1, nPitchUV, nImgPitchUV, NPEL);
        for (int jy = 0; jy < bs2; jy++) for (int jx = 0; jx < bs2; jx++) {
            int a = (*srcU)[(size_t)baseUV + (bux + jx) + (buy + jy) * nPitchUV];
            int b = (*refU)[(size_t)baseUV + (bux + jx) + (buy + jy) * nPitchUV + roffUV];
            int d = a - b; if (d < 0) d = -d; sad += d;
            a = (*srcV)[(size_t)baseUV + (bux + jx) + (buy + jy) * nPitchUV];
            b = (*refV)[(size_t)baseUV + (bux + jx) + (buy + jy) * nPitchUV + roffUV];
            d = a - b; if (d < 0) d = -d; sad += d;
        }
    }
    return sad;
}

struct CR { int cost; int x, y; };

// The whole proven one-Search assembly for one level.  own[] is the incoming
// vector region (2n ints) and is updated in place to the post-search own
// region; out receives the packed (x,y,sad) triples.
static void ref_search_level(
    int nBlkX, int nBlkY, int BS, int SEARCH, int NPEL, int chroma,
    int nPad, int nBlkSizeOvr, int nLogScale, int nExtW, int nExtH,
    int nLambdaLevel, int lsadN, int pZero, int pGlobal, int pNew,
    int gx, int gy,
    std::vector<int>& own, std::vector<int>& copy, const std::vector<int>& sads,
    const std::vector<int>& srcY, const std::vector<int>& refY,
    const std::vector<int>* srcU, const std::vector<int>* refU,
    const std::vector<int>* srcV, const std::vector<int>* refV,
    int baseY, int baseUV,
    int nPitchY, int nPitchUV, int nImgPitchY, int nImgPitchUV,
    std::vector<int>& out)
{
    int n = nBlkX * nBlkY;
    const int zeroX = 0, zeroY = 0;
    const int gsx = gx * NPEL, gsy = gy * NPEL;   // slot -1 = globalMV*nPel (int; MVs are small)

    // ---- prepare host-equivalent (ANALYZE_SYNC=1; kt_prepare_search) ----
    std::vector<int> data((size_t)n * 12), dataf((size_t)n * 5);
    int nPaddingScaled = nPad >> nLogScale;
    for (int by = 0; by < nBlkY; by++) for (int bx = 0; bx < nBlkX; bx++) {
        int blkIdx = bx + by * nBlkX;
        size_t o = (size_t)blkIdx * 12, fo = (size_t)blkIdx * 5;
        int X = nPad + nBlkSizeOvr * bx, Y = nPad + nBlkSizeOvr * by;
        data[o + 0] = NPEL * (nExtW - X - BS - nPad + nPaddingScaled) - 1;
        data[o + 1] = NPEL * (nExtH - Y - BS - nPad + nPaddingScaled) - 1;
        data[o + 2] = -NPEL * (X - nPad + nPaddingScaled);
        data[o + 3] = -NPEL * (Y - nPad + nPaddingScaled);
        int p1 = -2;
        if (bx > 0) p1 = blkIdx - 1;                 // own region, FRESH
        int p2 = (by > 0) ? blkIdx - nBlkX : p1;
        int p3 = -2;
        if ((by < nBlkY - 1) && (bx < nBlkX - 1)) p3 = blkIdx + nBlkX + 1 + n;
        data[o + 4] = -2; data[o + 5] = -1; data[o + 6] = blkIdx;
        data[o + 7] = p1; data[o + 8] = p2; data[o + 9] = p3;
        data[o + 10] = own[2 * blkIdx + 0];
        data[o + 11] = own[2 * blkIdx + 1];
        dataf[fo + 0] = pZero; dataf[fo + 1] = pGlobal; dataf[fo + 2] = 0; dataf[fo + 3] = pNew;
        int sad = sads[blkIdx];
        int lambda = nLambdaLevel * lsadN / (lsadN + (sad >> 1))
                   * lsadN / (lsadN + (sad >> 1));
        if (by == 0) lambda = 0;
        dataf[fo + 4] = lambda;
    }
    // copy-region snapshot (vectors_copy = incoming own)
    copy = own;

    // ---- ordered column search (ascending blkx; fresh left/up) ----
    std::vector<int> vx(2 * n), vy(2 * n);
    for (int i = 0; i < n; i++) { vx[i] = own[2 * i]; vy[i] = own[2 * i + 1]; }
    for (int i = 0; i < n; i++) { vx[n + i] = copy[2 * i]; vy[n + i] = copy[2 * i + 1]; }

    static const int e1x[8] = { 0,0,-1,1,-1,-1,1,1 };
    static const int e1y[8] = { -1,1,0,0,-1,1,-1,1 };
    static const int e2x[16] = { -1,-1,0,0,1,1,-2,2,-2,2,-2,2,-2,-2,2,2 };
    static const int e2y[16] = { -2,2,-2,2,-2,2,-1,-1,0,0,1,1,-2,2,-2,2 };
    static const int hx[6] = { -2,-1,1,2,1,-1 };
    static const int hy[6] = { 0,2,2,0,-2,-2 };

    for (int blkx = 0; blkx < nBlkX; blkx++) for (int blky = 0; blky < nBlkY; blky++) {
        int blkIdx = blky * nBlkX + blkx;
        const int* D = &data[(size_t)blkIdx * 12];
        const int* F = &dataf[(size_t)blkIdx * 5];
        int offx = nPad + blkx * (BS / 2), offy = nPad + blky * (BS / 2);

        auto resolve = [&](int idx, int& x, int& y) {
            if (idx == -2) { x = zeroX; y = zeroY; }
            else if (idx == -1) { x = gsx; y = gsy; }
            else { x = vx[idx]; y = vy[idx]; }
        };
        auto clip = [&](int& x, int& y) {
            x = (x > D[0]) ? D[0] : ((x < D[2]) ? D[2] : x);
            y = (y > D[1]) ? D[1] : ((y < D[3]) ? D[3] : y);
        };

        int px[7], py[7];
        for (int k = 0; k < 6; k++) {
            int tx, ty; resolve(D[4 + k], tx, ty); clip(tx, ty);
            int dst = (k < 3) ? k : k + 1;
            px[dst] = tx; py[dst] = ty;
        }
        px[3] = std::min(std::max(std::min(px[4], px[5]), px[6]), std::max(px[4], px[5]));
        py[3] = std::min(std::max(std::min(py[4], py[5]), py[6]), std::max(py[4], py[5]));

        CR r[7];
        for (int i = 0; i < 7; i++) {
            r[i].x = px[i]; r[i].y = py[i];
            r[i].cost = (F[4] * ((px[i] - D[10]) * (px[i] - D[10])
                                 + (py[i] - D[11]) * (py[i] - D[11]))) >> 8;
        }
        for (int i = 0; i < 7; i++) {
            int sad = sad_of(srcY, refY, srcU, refU, srcV, refV,
                             baseY, baseUV, offx, offy, BS, chroma,
                             nPitchY, nPitchUV, nImgPitchY, nImgPitchUV, NPEL,
                             r[i].x, r[i].y);
            if (i < 3) r[i].cost = sad + ((sad * F[i]) >> 8);
            else       r[i].cost += sad;
        }
        for (int i = 1; i < 7; i++) if (r[0].cost > r[i].cost) r[0] = r[i];
        CR best = r[0];

        auto refine = [&](int cx, int cy, const int* adx, const int* ady, int nC) {
            std::vector<CR> rr(nC);
            for (int i = 0; i < nC; i++) {
                int X = cx + adx[i], Y = cy + ady[i];
                bool ok = (X <= D[0]) && (Y <= D[1]) && (X >= D[2]) && (Y >= D[3]);
                int c = (F[4] * ((X - D[10]) * (X - D[10]) + (Y - D[11]) * (Y - D[11]))) >> 8;
                if (c >= best.cost) ok = false;
                rr[i].x = X; rr[i].y = Y;
                rr[i].cost = ok ? c : 2147483647;
            }
            for (int i = 0; i < nC; i++) {
                if (rr[i].cost == 2147483647) continue;
                int sad = sad_of(srcY, refY, srcU, refU, srcV, refV,
                                 baseY, baseUV, offx, offy, BS, chroma,
                                 nPitchY, nPitchUV, nImgPitchY, nImgPitchUV, NPEL,
                                 rr[i].x, rr[i].y);
                rr[i].cost += sad + ((sad * F[3]) >> 8);
            }
            for (int i = 1; i < nC; i++) if (rr[0].cost > rr[i].cost) rr[0] = rr[i];
            if (rr[0].cost < best.cost) best = rr[0];
        };

        if (SEARCH == 1) {
            int bmx = best.x, bmy = best.y;      // snapshot: expand2 shares centre
            refine(bmx, bmy, e1x, e1y, 8);
            refine(bmx, bmy, e2x, e2y, 16);
        } else {
            refine(best.x, best.y, hx, hy, 6);
            refine(best.x, best.y, e1x, e1y, 8); // chained off hex2
        }
        vx[blkIdx] = best.x; vy[blkIdx] = best.y;
    }

    for (int i = 0; i < n; i++) { own[2 * i] = vx[i]; own[2 * i + 1] = vy[i]; }

    // ---- kl_calc_all_sad (SAD of the final MV) ----
    out.resize((size_t)3 * n);
    for (int i = 0; i < n; i++) {
        int bx = i % nBlkX, by = i / nBlkX;
        int offx = nPad + bx * (BS / 2), offy = nPad + by * (BS / 2);
        int sad = sad_of(srcY, refY, srcU, refU, srcV, refV,
                         baseY, baseUV, offx, offy, BS, chroma,
                         nPitchY, nPitchUV, nImgPitchY, nImgPitchUV, NPEL,
                         own[2 * i], own[2 * i + 1]);
        out[3 * i + 0] = own[2 * i];
        out[3 * i + 1] = own[2 * i + 1];
        out[3 * i + 2] = sad;
    }
}

// ===========================================================================
// CPU reference of the global-MV chain + InterpolatePrediction
// (transcribed from ktgmc_motion.cl: kt_most_freq_mv 709-771,
//  kt_mean_global_mv 424-452, kt_interpolate_prediction 331-411)
// ===========================================================================
static int ref_most_freq(const std::vector<int>& own, int n, int isY) {
    int kcnt[1024], kbin[1024];
    for (int t = 0; t < 1024; t++) {
        int maxc = 0, idx = 0;
        for (int i = 0; i < 8; i++) {
            int b = t + i * 1024;
            int c = 0;
            for (int j = 0; j < n; j++) {
                int v = isY ? own[2 * j + 1] : own[2 * j];
                if (v == b - 4096) c++;
            }
            if (c > maxc) { maxc = c; idx = b; }
        }
        kcnt[t] = maxc; kbin[t] = idx;
    }
    for (int s = 512; s >= 32; s >>= 1)
        for (int t = 0; t < s; t++)
            if (kcnt[t + s] > kcnt[t]) { kcnt[t] = kcnt[t + s]; kbin[t] = kbin[t + s]; }
    for (int s = 16; s >= 1; s >>= 1)
        for (int t = 0; t < s; t++)
            if (kcnt[t + s] > kcnt[t]) { kcnt[t] = kcnt[t + s]; kbin[t] = kbin[t + s]; }
    return kbin[0] - 4096;
}

static void ref_mean_global(const std::vector<int>& own, int n,
                            int medianx, int mediany, int& mx, int& my) {
    long long meanvx = 0, meanvy = 0; int num = 0;
    for (int j = 0; j < n; j++) {
        int vx = own[2 * j], vy = own[2 * j + 1];
        int dx = vx - medianx; if (dx < 0) dx = -dx;
        int dy = vy - mediany; if (dy < 0) dy = -dy;
        if (dx < 6 && dy < 6) { meanvx += vx; meanvy += vy; num += 1; }
    }
    mx = (int)(2 * meanvx) / num;   // CUDA form: no num==0 guard (ktgmc_motion.cl:450)
    my = (int)(2 * meanvy) / num;
}

static void ref_interp(const std::vector<int>& srcOwn, const std::vector<int>& srcSads,
                       std::vector<int>& dstOwn, std::vector<int>& dstSads,
                       int nSX, int nSY, int nDX, int nDY,
                       int normFactor, int normov, int atotal, int aodd, int aeven) {
    for (int y = 0; y < nDY; y++) for (int x = 0; x < nDX; x++) {
        int i = x, j = y;
        if (i >= 2 * nSX) i = 2 * nSX - 1;
        if (j >= 2 * nSY) j = 2 * nSY - 1;
        int offy = -1 + 2 * (j % 2);
        int offx = -1 + 2 * (i % 2);
        int iper2 = i >> 1, jper2 = j >> 1;
        int v1x, v1y, v2x, v2y, v3x, v3y, v4x, v4y;
        int sad1, sad2, sad3, sad4;
        if ((i == 0) || (i >= 2 * nSX - 1)) {
            if ((j == 0) || (j >= 2 * nSY - 1)) {
                int vi = iper2 + jper2 * nSX;
                v1x = v2x = v3x = v4x = srcOwn[2 * vi]; v1y = v2y = v3y = v4y = srcOwn[2 * vi + 1];
                sad1 = sad2 = sad3 = sad4 = srcSads[vi];
            } else {
                int va = iper2 + jper2 * nSX, vb = iper2 + (jper2 + offy) * nSX;
                v1x = v2x = srcOwn[2 * va]; v1y = v2y = srcOwn[2 * va + 1];
                v3x = v4x = srcOwn[2 * vb]; v3y = v4y = srcOwn[2 * vb + 1];
                sad1 = sad2 = srcSads[va]; sad3 = sad4 = srcSads[vb];
            }
        } else if ((j == 0) || (j >= 2 * nSY - 1)) {
            int va = iper2 + jper2 * nSX, vb = iper2 + offx + jper2 * nSX;
            v1x = v2x = srcOwn[2 * va]; v1y = v2y = srcOwn[2 * va + 1];
            v3x = v4x = srcOwn[2 * vb]; v3y = v4y = srcOwn[2 * vb + 1];
            sad1 = sad2 = srcSads[va]; sad3 = sad4 = srcSads[vb];
        } else {
            int v1 = iper2 + jper2 * nSX;
            int v2 = iper2 + offx + jper2 * nSX;
            int v3 = iper2 + (jper2 + offy) * nSX;
            int v4 = iper2 + offx + (jper2 + offy) * nSX;
            v1x = srcOwn[2 * v1]; v1y = srcOwn[2 * v1 + 1];
            v2x = srcOwn[2 * v2]; v2y = srcOwn[2 * v2 + 1];
            v3x = srcOwn[2 * v3]; v3y = srcOwn[2 * v3 + 1];
            v4x = srcOwn[2 * v4]; v4y = srcOwn[2 * v4 + 1];
            sad1 = srcSads[v1]; sad2 = srcSads[v2]; sad3 = srcSads[v3]; sad4 = srcSads[v4];
        }
        int ax1 = (offx > 0) ? aodd : aeven;
        int ax2 = atotal - ax1;
        int ay1 = (offy > 0) ? aodd : aeven;
        int ay2 = atotal - ay1;
        int a11 = ax1 * ay1, a12 = ax1 * ay2, a21 = ax2 * ay1, a22 = ax2 * ay2;
        int vx = (a11 * v1x + a21 * v2x + a12 * v3x + a22 * v4x) / normov;
        int vy = (a11 * v1y + a21 * v2y + a12 * v3y + a22 * v4y) / normov;
        int tmp_sad = (a11 * sad1 + a21 * sad2 + a12 * sad3 + a22 * sad4) / normov;
        if (normFactor > 0) { vx >>= normFactor; vy >>= normFactor; }
        else { vx <<= -normFactor; vy <<= -normFactor; }
        int index = x + y * nDX;
        dstOwn[2 * index] = vx;
        dstOwn[2 * index + 1] = vy;
        dstSads[index] = (tmp_sad >> 4);
    }
}

// ===========================================================================
// CPU reference of the WHOLE proven KMAnalyse loop (contract -10)
// ===========================================================================
struct AnalParams {
    int BS;                 // blksize (== blksizeV)
    int overlap;            // == BS/2 (upstream-enforced)
    int searchArg;          // script arg (0/1/2/3/4/5/6/7)
    bool chroma, isb, global;
    int delta, lambda, lsad, plevel, pnew;
    int bpp;                // bits per component
    const SuperGeom* g;     // super geometry (hpad/vpad/pel/levels of the super)
};

// computes the per-level levelInfo + plane params; runs the coarse->fine loop;
// returns the packed VECTOR array (coarsest level first), exactly the output
// frame's byte content
static std::vector<int> ref_analyse(const AnalParams& ap,
                                    const std::vector<int>& srcY,
                                    const std::vector<int>* srcU,
                                    const std::vector<int>* srcV,
                                    const std::vector<int>& refY,
                                    const std::vector<int>* refU,
                                    const std::vector<int>* refV) {
    const SuperGeom& g = *ap.g;
    const int BS = ap.BS, ovr = ap.overlap;
    const int nBlkSizeX = BS, nOverlapX = ovr;
    const int nBlkSizeY = BS, nOverlapY = ovr;
    const int chroma = ap.chroma ? 1 : 0;

    // MV.cpp:2511-2514 covered grid
    const int nBlkX0 = (g.origW - nOverlapX) / (nBlkSizeX - nOverlapX);
    const int nBlkY0 = (g.origH - nOverlapY) / (nBlkSizeX - nOverlapX);
    const int nWidth_B = (nBlkSizeX - nOverlapX) * nBlkX0 + nOverlapX;
    const int nHeight_B = (nBlkSizeX - nOverlapX) * nBlkY0 + nOverlapX;

    // MV.cpp:2541-2547 valid levels (lv pinned 0)
    int nLevelsMax = 0;
    while (((nWidth_B >> nLevelsMax) - nOverlapX) / (nBlkSizeX - nOverlapX) > 0 &&
           ((nHeight_B >> nLevelsMax) - nOverlapY) / (nBlkSizeX - nOverlapY) > 0)
        ++nLevelsMax;
    const int L = nLevelsMax;   // <= g.nLevels guaranteed by the plugin's check

    const int nPad = g.nHPad;   // luma padding at EVERY level (contract -1.2)
    const int pitchY = g.nSuperWidth;
    const int pitchUV = pitchY >> g.logUVx;
    const int vpadU = g.nVPad / g.yRatioUV;

    const int lsadN = ap.lsad * (BS * BS) / 64 * (1 << (ap.bpp - 8));   // MV.cpp:2577
    const int pNew = ap.pnew, pZero = ap.pnew, pGlobal = 0;             // MV.cpp:2832/2833

    // upstream search arg -SearchType (MV.cpp:2586-2619)
    int st;
    switch (ap.searchArg) {
    case 0: st = 1; break;    // ONETIME
    case 1: st = 2; break;    // NSTEP
    case 3: st = 8; break;    // EXHAUSTIVE
    case 5: st = 32; break;   // UMHSEARCH
    case 6: st = 64; break;   // HSEARCH
    case 7: st = 128; break;  // VSEARCH
    case 2: default: st = 4; break;  // LOGARITHMIC
    }

    struct Lvl {
        int nBlkX, nBlkY, n, nPelLevel, devSearch, nls, lam;
        int offY, offU, Hl, Hc, extW, extH;
        std::vector<int> ownIn, postOwn, sadsIn, out;
    };
    std::vector<Lvl> lv(L);
    for (int i = 0; i < L; i++) {
        Lvl& l = lv[i];
        l.nBlkX = ((nWidth_B >> i) - nOverlapX) / (nBlkSizeX - nOverlapX);
        l.nBlkY = ((nHeight_B >> i) - nOverlapY) / (nBlkSizeX - nOverlapY);
        l.n = l.nBlkX * l.nBlkY;
        int nActualLevel = i;
        l.nPelLevel = (nActualLevel == 0) ? g.nPel : 1;
        int stl = (nActualLevel == 0 || st == 64 || st == 128) ? st : 8;
        l.devSearch = (stl == 8) ? 1 : 2;
        l.nls = i;
        l.lam = ap.lambda / (l.nPelLevel * l.nPelLevel);
        int nScale = 1 << i;
        if (ap.plevel == 1) l.lam *= nScale;
        else if (ap.plevel == 2) l.lam *= nScale * nScale;
        l.Hl = (i == 0) ? g.origH : PlaneHeightLuma(g.origH, i, g.yRatioUV, g.nVPad);
        int Wl = (i == 0) ? g.origW : PlaneWidthLuma(g.origW, i, g.xRatioUV, g.nHPad);
        l.Hc = l.Hl / g.yRatioUV;
        l.extW = Wl + 2 * g.nHPad;
        l.extH = l.Hl + 2 * g.nVPad;
        l.offY = (int)PlaneSuperOffset(false, g.origH, i, g.nPel, g.nVPad, pitchY, g.yRatioUV);
        l.offU = (int)PlaneSuperOffset(true, g.origH / g.yRatioUV, i, g.nPel, vpadU, pitchUV, g.yRatioUV);
    }

    std::vector<int> frame;
    int gmvx = 0, gmvy = 0;   // per-contract the coarsest globalMV sentinel is (0,0)

    for (int i = L - 1; i >= 0; i--) {
        Lvl& l = lv[i];
        const int n = l.n;
        l.ownIn.assign((size_t)2 * n, 0);
        l.sadsIn.assign((size_t)n, 0);
        int normFactor = 0, normov = 0, atotal = 0, aodd = 0, aeven = 0;

        if (i < L - 1) {
            Lvl& c = lv[i + 1];
            if (ap.global) {
                // EstimateGlobalMVDoubled on level i+1's post-search vectors
                int fx = ref_most_freq(c.postOwn, c.n, 0);
                int fy = ref_most_freq(c.postOwn, c.n, 1);
                ref_mean_global(c.postOwn, c.n, fx, fy, gmvx, gmvy);
            }
            // InterpolatePrediction c -> l (x-axis weights for both axes)
            normov = (nBlkSizeX - nOverlapX) * (nBlkSizeX - nOverlapX);
            atotal = (nBlkSizeX - nOverlapX) * 4;
            aodd = 3 * nBlkSizeX - 2 * nOverlapX;
            aeven = 3 * nBlkSizeX - 4 * nOverlapX;
            normFactor = 3 - (l.nPelLevel == 2 ? 1 : 0) + (c.nPelLevel == 2 ? 1 : 0);
            ref_interp(c.postOwn, c.sadsIn, l.ownIn, l.sadsIn,
                       c.nBlkX, c.nBlkY, l.nBlkX, l.nBlkY,
                       normFactor, normov, atotal, aodd, aeven);
        }
        // (coarsest level: own/sads stay zeroed -the twin's deterministic
        //  substitution for upstream's undefined work buffer)

        std::vector<int> copy;
        ref_search_level(
            l.nBlkX, l.nBlkY, BS, l.devSearch, l.nPelLevel, chroma,
            nPad, BS - ovr, l.nls, l.extW, l.extH,
            l.lam, lsadN, pZero, pGlobal, pNew,
            gmvx, gmvy,
            l.ownIn, copy, l.sadsIn,
            srcY, refY, srcU, refU, srcV, refV,
            l.offY, chroma ? l.offU : 0,
            pitchY, chroma ? pitchUV : 0,
            pitchY * l.extH, chroma ? pitchUV * (l.Hc + 2 * vpadU) : 0,
            l.out);
        l.postOwn = l.ownIn;

        // output region: coarsest level first
        frame.insert(frame.end(), l.out.begin(), l.out.end());
    }
    return frame;
}

// ===========================================================================
// driver
// ===========================================================================
static VideoInfo viFor(IScriptEnvironment* env, const char* pt, int w, int h, int len) {
    char script[160];
    sprintf_s(script, "BlankClip(width=%d, height=%d, pixel_type=\"%s\", length=%d)", w, h, pt, len);
    return env->Invoke("Eval", AVSValue(script)).AsClip()->GetVideoInfo();
}

// read a source frame's planes as ints
static void grabPlaneInt(PVideoFrame& f, int pl, int pw, int ph, std::vector<int>& out) {
    out.resize((size_t)pw * ph);
    const uint8_t* fp = f->GetReadPtr(pl);
    int pitch = f->GetPitch(pl);
    for (int y = 0; y < ph; ++y) {
        const uint8_t* row = fp + (size_t)y * pitch;
        if (ph && 0) {}
        for (int x = 0; x < pw; ++x) out[(size_t)y * pw + x] = row[x];   // 8-bit path
    }
}
static void grabPlaneInt16(PVideoFrame& f, int pl, int pw, int ph, std::vector<int>& out) {
    out.resize((size_t)pw * ph);
    const uint8_t* fp = f->GetReadPtr(pl);
    int pitch = f->GetPitch(pl);
    for (int y = 0; y < ph; ++y) {
        const uint16_t* row = (const uint16_t*)(fp + (size_t)y * pitch);
        for (int x = 0; x < pw; ++x) out[(size_t)y * pw + x] = (int)row[x];
    }
}

// the covered-grid analyse level list (MV.cpp:2511-2547; lv pinned 0) -// shared by the output-vi check; ref_analyse recomputes it identically
static int analyseLevelBlocks(const SuperGeom& g, int blksize,
                              std::vector<std::pair<int,int>>* info) {
    const int ovr = blksize / 2;
    const int nBlkX0 = (g.origW - ovr) / (blksize - ovr);
    const int nBlkY0 = (g.origH - ovr) / (blksize - ovr);
    const int nWB = (blksize - ovr) * nBlkX0 + ovr;
    const int nHB = (blksize - ovr) * nBlkY0 + ovr;
    int L = 0;
    while (((nWB >> L) - ovr) / (blksize - ovr) > 0 &&
           ((nHB >> L) - ovr) / (blksize - ovr) > 0) {
        if (info) info->push_back(std::make_pair(((nWB >> L) - ovr) / (blksize - ovr),
                                                 ((nHB >> L) - ovr) / (blksize - ovr)));
        ++L;
    }
    return L;
}

template <typename PX>
static void grabPlane(PVideoFrame& f, int pl, int pw, int ph, std::vector<PX>& out) {
    out.resize((size_t)pw * ph);
    const uint8_t* fp = f->GetReadPtr(pl);
    int pitch = f->GetPitch(pl);
    for (int y = 0; y < ph; ++y)
        memcpy(&out[(size_t)y * pw], fp + (size_t)y * pitch, (size_t)pw * sizeof(PX));
}

template <typename PX>
static std::vector<int> toInt(const std::vector<PX>& in) {
    std::vector<int> out(in.size());
    for (size_t k = 0; k < in.size(); k++) out[k] = (int)in[k];
    return out;
}

// one end-to-end case: build the chain via Invoke, compare every VECTOR of
// the produced frame against the CPU reference of the proven loop
template <typename PX>
static void runCase(IScriptEnvironment* env, const char* tag,
                    PClip srcClip, const SuperGeom& g,
                    int n, const char* fnName,
                    int blksize, int searchArg, bool chroma, bool isb,
                    int delta, int lambda, int lsad, int plevel, bool global,
                    int pnew, int batch, int len, bool expectInvalid)
{
    // ---- build the chain: KMSuperOCL(...).KMAnalyseOCL(...) ----
    AVSValue sargs[8] = { AVSValue(srcClip), 8, 8, g.nPel, g.nLevels,
                          AVSValue(g.chroma), 2, 2 };
    AVSValue sup;
    try {
        sup = env->Invoke("KMSuperOCL", AVSValue(sargs, 8));
    } catch (const AvisynthError& e) {
        printf("[FAIL] %s: KMSuperOCL Invoke threw: %s\n", tag, e.msg);
        ++g_failures; ++g_cases; return;
    }
    if (!sup.IsClip()) { printf("[FAIL] %s: super not a clip\n", tag); ++g_failures; ++g_cases; return; }
    PClip superClip = sup.AsClip();

    AVSValue aargs[15];
    aargs[0] = AVSValue(superClip);
    aargs[1] = blksize;
    aargs[2] = blksize / 2;
    aargs[3] = searchArg;
    aargs[4] = AVSValue(isb);
    aargs[5] = AVSValue(chroma);
    aargs[6] = delta;
    aargs[7] = lambda;
    aargs[8] = lsad;
    aargs[9] = AVSValue(global);
    aargs[10] = plevel;
    aargs[11] = pnew;
    aargs[12] = AVSValue(true);   // meander (dead on the CUDA path, D7)
    aargs[13] = AVSValue();       // partial: not given
    aargs[14] = batch;
    AVSValue ana;
    try {
        ana = env->Invoke(fnName, AVSValue(aargs, 15));
    } catch (const AvisynthError& e) {
        printf("[FAIL] %s: KMAnalyse Invoke threw: %s\n", tag, e.msg);
        ++g_failures; ++g_cases; return;
    }
    if (!ana.IsClip()) { printf("[FAIL] %s: not a clip\n", tag); ++g_failures; ++g_cases; return; }
    PClip mvClip = ana.AsClip();

    // ---- output vi check (BGR32, 2048 x ceil(bytes/8192)) ----
    std::vector<std::pair<int,int>> info;
    const int nLv = analyseLevelBlocks(g, blksize, &info);
    long long totalBlocks = 0;
    for (int i = 0; i < nLv; i++) totalBlocks += (long long)info[i].first * info[i].second;
    const int outFrameBytes = (int)(totalBlocks * 12);
    const VideoInfo& ovi = mvClip->GetVideoInfo();
    if (ovi.width != 2048 || ovi.height != (outFrameBytes + 8191) / 8192 ||
        !ovi.IsRGB32() || ovi.BitsPerPixel() != 32) {
        printf("[FAIL] %s: output vi %dx%d != BGR32 2048x%d\n", tag,
               ovi.width, ovi.height, (outFrameBytes + 8191) / 8192);
        ++g_failures; ++g_cases; return;
    }

    PVideoFrame outf;
    try {
        outf = mvClip->GetFrame(n, env);
    } catch (const AvisynthError& e) {
        printf("[FAIL] %s: GetFrame threw: %s\n", tag, e.msg);
        ++g_failures; ++g_cases; return;
    }

    // ---- KMVIsValid frame property ----
    {
        const AVSMap* m = env->getFramePropsRO(outf);
        int err = 0;
        int64_t v = env->propGetInt(m, "KMVIsValid", 0, &err);
        int expect = expectInvalid ? 0 : 1;
        if (err || (int)v != expect) {
            printf("[FAIL] %s: KMVIsValid=%d (err=%d) expected %d\n", tag,
                   (int)v, err, expect);
            ++g_failures; ++g_cases; return;
        }
    }

    // ---- read the frame's VECTOR bytes ----
    std::vector<int> got((size_t)outFrameBytes / 4);
    {
        const uint8_t* fp = outf->GetReadPtr();
        int pitch = outf->GetPitch();
        for (size_t k = 0; k < got.size(); k++) {
            size_t bytePos = k * 4;
            const uint8_t* q = fp + (bytePos / (size_t)pitch) * pitch + (bytePos % (size_t)pitch);
            memcpy(&got[k], q, 4);
        }
    }

    ++g_vec_checks;

    if (expectInvalid) {
        // WriteDefault (D2): {0,0,verybigSAD} per block, coarsest first;
        // verybigSAD = 3*BS^2*(1<<bpp) with the SUPER's bpp (MV.cpp:999)
        const int verybig = 3 * blksize * blksize * (sizeof(PX) == 1 ? 256 : 65536);
        bool ok = true;
        for (size_t k = 0; k + 2 < got.size(); k += 3) {
            if (got[k] != 0 || got[k + 1] != 0 || got[k + 2] != verybig) {
                printf("  MISMATCH default vec %zu: (%d,%d,%d) expected (0,0,%d)\n",
                       k / 3, got[k], got[k + 1], got[k + 2], verybig);
                ok = false; break;
            }
        }
        printf("[%s] %s (WriteDefault, %d VECTORs)\n", ok ? "PASS" : "FAIL", tag,
               (int)(got.size() / 3));
        if (!ok) ++g_failures;
        ++g_cases;
        return;
    }


// ===========================================================================
// helpers: read a plane as PX, convert PX plane -> int plane (search model's
// arithmetic is plain int)
// ===========================================================================

    // ---- CPU reference continued: super pyramid -> proven loop -> compare ----
    const int offset = isb ? delta : -delta;
    const int nsrc = n, nref = n + offset;
    AnalParams ap;
    ap.BS = blksize; ap.overlap = blksize / 2; ap.searchArg = searchArg;
    ap.chroma = chroma; ap.isb = isb; ap.global = global;
    ap.delta = delta; ap.lambda = lambda; ap.lsad = lsad; ap.plevel = plevel;
    ap.pnew = pnew;
    ap.bpp = (sizeof(PX) == 1) ? 8 : 16;
    ap.g = &g;

    const int wUV = g.origW / g.xRatioUV, hUV = g.origH / g.yRatioUV;
    std::vector<PX> sYp, rYp, sUp, sVp, rUp, rVp;
    for (int which = 0; which < 2; which++) {
        PVideoFrame f = srcClip->GetFrame(which ? nref : nsrc, env);
        grabPlane<PX>(f, PLANAR_Y, g.origW, g.origH, which ? rYp : sYp);
        if (g.chroma) {
            grabPlane<PX>(f, PLANAR_U, wUV, hUV, which ? rUp : sUp);
            grabPlane<PX>(f, PLANAR_V, wUV, hUV, which ? rVp : sVp);
        }
    }

    int pY, pUH;
    std::vector<PX> supSY, supRY, supSU, supSV, supRU, supRV;
    buildPlaneRef<PX>(g, false, sYp, g.origW, g.origH, supSY, pY, pUH);
    buildPlaneRef<PX>(g, false, rYp, g.origW, g.origH, supRY, pY, pUH);
    if (g.chroma) {
        buildPlaneRef<PX>(g, true, sUp, wUV, hUV, supSU, pY, pUH);
        buildPlaneRef<PX>(g, true, sVp, wUV, hUV, supSV, pY, pUH);
        buildPlaneRef<PX>(g, true, rUp, wUV, hUV, supRU, pY, pUH);
        buildPlaneRef<PX>(g, true, rVp, wUV, hUV, supRV, pY, pUH);
    }

    std::vector<int> cpuSY = toInt<PX>(supSY), cpuRY = toInt<PX>(supRY);
    std::vector<int> cpuSU, cpuSV, cpuRU, cpuRV;
    if (g.chroma) {
        cpuSU = toInt<PX>(supSU); cpuSV = toInt<PX>(supSV);
        cpuRU = toInt<PX>(supRU); cpuRV = toInt<PX>(supRV);
    }

    std::vector<int> ref = ref_analyse(ap, cpuSY,
                                       g.chroma ? &cpuSU : (std::vector<int>*)nullptr,
                                       g.chroma ? &cpuSV : (std::vector<int>*)nullptr,
                                       cpuRY,
                                       g.chroma ? &cpuRU : (std::vector<int>*)nullptr,
                                       g.chroma ? &cpuRV : (std::vector<int>*)nullptr);

    // ---- bit-exact comparison of EVERY VECTOR (x,y,sad) ----
    bool ok = (ref.size() == got.size());
    if (!ok) {
        printf("  MISMATCH size: got %zu ints, ref %zu ints\n", got.size(), ref.size());
    } else {
        for (size_t k = 0; k < got.size(); k++) {
            if (got[k] != ref[k]) {
                printf("  MISMATCH vec %zu (field %zu): got %d ref %d\n",
                       k / 3, k % 3, got[k], ref[k]);
                ok = false; break;
            }
        }
    }
    printf("[%s] %s (%d VECTORs, %d levels)\n", ok ? "PASS" : "FAIL", tag,
           (int)(got.size() / 3), nLv);
    if (!ok) ++g_failures;
    ++g_cases;
}

// ===========================================================================
// main
// ===========================================================================
static const int W3 = 64, H3 = 48;    // 3 covered levels (15x11, 7x5, 3x2 @ BS8)
static const int W2 = 32, H2 = 24;    // 2 covered levels (7x5, 3x2 @ BS8)

int main(int argc, char** argv) {
    setvbuf(stdout, nullptr, _IONBF, 0);
    AddVectoredExceptionHandler(1, eh);
    const char* analyseDll = argc > 1 ? argv[1] : nullptr;
    const char* superDll = argc > 2 ? argv[2] : nullptr;
    std::string pluginsDir = argc > 3 ? argv[3]
        : "Q:\\Downloads\\openclfilter\\sdk\\avs\\plugins64";

    if (analyseDll) {
        std::string dst = pluginsDir + "\\ktgmc_analyse_avs.dll";
        if (!CopyFileA(analyseDll, dst.c_str(), FALSE)) {
            printf("cannot copy analyse plugin '%s' (%lu)\n", analyseDll, GetLastError());
            return 1;
        }
        printf("analyse plugin installed: %s\n", dst.c_str());
    }
    if (superDll) {
        std::string dst = pluginsDir + "\\ktgmc_super_avs.dll";
        if (!CopyFileA(superDll, dst.c_str(), FALSE)) {
            printf("cannot copy super plugin '%s' (%lu)\n", superDll, GetLastError());
            return 1;
        }
        printf("super plugin installed: %s\n", dst.c_str());
    }

    HMODULE dll = LoadLibraryA("AviSynth.dll");
    if (!dll) { printf("cannot load AviSynth.dll (%lu)\n", GetLastError()); return 1; }
    FARPROC pv = GetProcAddress(dll, "?AVS_linkage@@3QEBUAVS_Linkage@@EB");
    if (!pv) { printf("AVS_linkage export not found\n"); return 1; }
    AVS_linkage = *(const AVS_Linkage**)pv;
    CSE_t cse = (CSE_t)GetProcAddress(dll, "CreateScriptEnvironment");
    if (!cse) { printf("CreateScriptEnvironment not exported\n"); return 1; }
    IScriptEnvironment* env = cse(AVISYNTH_INTERFACE_VERSION);
    if (!env) { printf("env creation failed\n"); return 1; }

    if (!env->FunctionExists("KMAnalyseOCL") || !env->FunctionExists("KMAnalyse")) {
        printf("[FAIL] autoload: KMAnalyseOCL / KMAnalyse not registered\n");
        return 1;
    }
    if (!env->FunctionExists("KMSuperOCL")) {
        printf("[FAIL] autoload: KMSuperOCL not registered\n");
        return 1;
    }
    printf("autoload OK (KMAnalyseOCL + KMAnalyse alias + KMSuperOCL registered)\n");

    // ---- main sweep ----
    {
        VideoInfo vi = viFor(env, "YV12", W3, H3, 4);
        SuperGeom g = makeGeom(vi, 8, 8, 2, 0, true);   // pel 2, auto levels
        PClip src(new RandSource(vi, 0x5EEDu));
        runCase<uint8_t>(env, "YV12 64x48 pel1 chroma hex(default search)", src, g,
                         1, "KMAnalyseOCL", 8, 4, true, false, 1, 400, 400, 0, true, 25, 1, 4, false);
        runCase<uint8_t>(env, "YV12 64x48 pel2 chroma hex(default search)", src, g,
                         1, "KMAnalyseOCL", 8, 4, true, false, 1, 400, 400, 0, true, 25, 1, 4, false);
        runCase<uint8_t>(env, "YV12 64x48 pel2 chroma=off hex", src, makeGeom(vi, 8, 8, 2, 0, false),
                         1, "KMAnalyseOCL", 8, 4, false, false, 1, 400, 400, 0, true, 25, 1, 4, false);
        runCase<uint8_t>(env, "YV12 64x48 pel1 chroma search=3 (fine EXHAUSTIVE)", src, makeGeom(vi, 8, 8, 1, 0, true),
                         1, "KMAnalyseOCL", 8, 3, true, false, 1, 400, 400, 0, true, 25, 1, 4, false);
        runCase<uint8_t>(env, "YV12 64x48 pel2 chroma isb(backward)", src, g,
                         1, "KMAnalyseOCL", 8, 4, true, true, 1, 400, 400, 0, true, 25, 1, 4, false);
        runCase<uint8_t>(env, "YV12 64x48 pel2 chroma plevel=2", src, g,
                         1, "KMAnalyseOCL", 8, 4, true, false, 1, 400, 400, 2, true, 25, 1, 4, false);
        runCase<uint8_t>(env, "YV12 64x48 pel2 chroma global=off", src, g,
                         1, "KMAnalyseOCL", 8, 4, true, false, 1, 400, 400, 0, false, 25, 1, 4, false);
        runCase<uint8_t>(env, "YV12 64x48 defaults blksize=16 (2 covered levels)", src, g,
                         2, "KMAnalyseOCL", 16, 4, true, false, 1, 400, 400, 0, true, 25, 1, 4, false);
    }

    // ---- 16-bit ----
    {
        VideoInfo vi = viFor(env, "YUV420P16", W3, H3, 4);
        SuperGeom g = makeGeom(vi, 8, 8, 2, 0, true);
        PClip src(new RandSource(vi, 0x16B1Du));
        runCase<uint16_t>(env, "YUV420P16 64x48 pel2 chroma hex", src, g,
                          1, "KMAnalyseOCL", 8, 4, true, false, 1, 400, 400, 0, true, 25, 1, 4, false);
        runCase<uint16_t>(env, "YUV420P16 64x48 pel1 chroma=off hex", src, makeGeom(vi, 8, 8, 1, 0, false),
                          1, "KMAnalyseOCL", 8, 4, false, false, 1, 400, 400, 0, true, 25, 1, 4, false);
    }

    // ---- 2-level 32x24 (the twin-measured level count) ----
    {
        VideoInfo vi = viFor(env, "YV12", W2, H2, 4);
        SuperGeom g = makeGeom(vi, 8, 8, 2, 0, true);
        PClip src(new RandSource(vi, 0x24C0Du));
        runCase<uint8_t>(env, "YV12 32x24 pel2 chroma hex (2 levels)", src, g,
                         1, "KMAnalyseOCL", 8, 4, true, false, 1, 400, 400, 0, true, 25, 1, 4, false);
        runCase<uint8_t>(env, "YV12 32x24 pel1 chroma=off search=3", src, makeGeom(vi, 8, 8, 1, 0, false),
                         1, "KMAnalyseOCL", 8, 3, false, false, 1, 400, 400, 0, true, 25, 1, 4, false);
    }

    // ---- upstream alias invocation (KMAnalyse) ----
    {
        VideoInfo vi = viFor(env, "YV12", W3, H3, 4);
        SuperGeom g = makeGeom(vi, 8, 8, 2, 0, true);
        PClip src(new RandSource(vi, 0xA1A5u));
        runCase<uint8_t>(env, "YV12 64x48 pel2 KMAnalyse alias", src, g,
                         1, "KMAnalyse", 8, 4, true, false, 1, 400, 400, 0, true, 25, 1, 4, false);
    }

    // ---- WriteDefault out-of-range frame (n=3, delta=1 forward, len=4) ----
    {
        VideoInfo vi = viFor(env, "YV12", W3, H3, 4);
        SuperGeom g = makeGeom(vi, 8, 8, 2, 0, true);
        PClip src(new RandSource(vi, 0xD00Du));
        runCase<uint8_t>(env, "YV12 64x48 pel2 out-of-range frame n=0 (WriteDefault)", src, g,
                         0, "KMAnalyseOCL", 8, 4, true, false, 1, 400, 400, 0, true, 25, 1, 4, true);
    }

    // ---- real script string chain ----
    try {
        const char* script =
            "KMSuperOCL(ColorBars(pixel_type=\"YV12\", width=64, height=48).Trim(0,3), "
            "pel=2, chroma=true).KMAnalyseOCL(blksize=8)";
        AVSValue ret = env->Invoke("Eval", AVSValue(script));
        if (!ret.IsClip()) throw AvisynthError("Eval did not return a clip");
        PClip mvClip = ret.AsClip();
        const VideoInfo& ovi = mvClip->GetVideoInfo();
        // YV12 64x48 BS8: 3 levels, blocks 15*11 + 7*5 + 3*2 = 206 -> 2472 bytes
        bool viok = (ovi.width == 2048 && ovi.pixel_type == VideoInfo::CS_BGR32);
        PVideoFrame f = mvClip->GetFrame(1, env);
        const AVSMap* m = env->getFramePropsRO(f);
        int err = 0;
        int64_t v = env->propGetInt(m, "KMVIsValid", 0, &err);
        bool ok = viok && !err && v == 1;
        printf("[%s] script-string ColorBars chain (vi %dx%d, KMVIsValid=%d)\n",
               ok ? "PASS" : "FAIL", ovi.width, ovi.height, (int)v);
        if (!ok) ++g_failures;
        ++g_cases; ++g_vec_checks;
    } catch (const AvisynthError& e) {
        printf("[FAIL] script-string chain: %s\n", e.msg); ++g_failures; ++g_cases;
    }

    // ---- constructor throws ----
    auto expectThrow = [&](const char* what, const char* script) {
        ++g_cases;
        try {
            env->Invoke("Eval", AVSValue(script));
            printf("[FAIL] %s accepted (expected throw)\n", what);
            ++g_failures;
        } catch (const AvisynthError&) {
            printf("[PASS] %s rejected\n", what);
        }
    };
    expectThrow("batch=2",
        "KMSuperOCL(BlankClip(width=64, height=48, pixel_type=\"YV12\", length=4))"
        ".KMAnalyseOCL(batch=2)");
    expectThrow("partial given",
        "KMSuperOCL(BlankClip(width=64, height=48, pixel_type=\"YV12\", length=4))"
        ".KMAnalyseOCL(partial=KMSuperOCL(BlankClip(width=64, height=48, "
        "pixel_type=\"YV12\", length=4)))");
    expectThrow("non-super input",
        "KMAnalyseOCL(BlankClip(width=64, height=48, pixel_type=\"YV12\", length=4))");

    printf("\n=== KMAnalyseOCL: %d cases (%d VECTOR-frame checks), %d failures ===\n",
           g_cases, g_vec_checks, g_failures);

    env->DeleteScriptEnvironment();
    // deliberately not FreeLibrary (AviSynth+ shutdown threads).
    return g_failures ? 1 : 0;
}
