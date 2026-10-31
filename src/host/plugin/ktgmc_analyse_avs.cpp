// ktgmc_analyse_avs.cpp — AviSynth+ host plugin for the OpenCL port of KTGMC's
// KMAnalyse (upstream: rigaya/AviSynthCUDAFilters KTGMC/MV.cpp + MVKernel.cu,
// GPL).  Ninth real AviSynth+ host plugin of the port, SECOND motion-stage
// plugin, chained after KMSuperOCL (eighth):  KMSuperOCL(src,...).KMAnalyseOCL(...)
//
// It replays the upstream KMAnalyse pipeline — the multi-level GroupOfPlanes
// coarse→fine loop — on the PROVEN ported kernels, embedded into the DLL at
// build time (ktgmc_motion.cl + ktgmc_search.cl).  The loop is the one proven
// bit-exact vs the CUDA twin (cuda_mv/KMANALYSE_HOST_CONTRACT.md §9/§10,
// KMANALYSE_MEASURE_REPORT.md; rig/dev_ktgmc_analyse.py 39/39).  Per level
// i = nLevels-1 (coarsest) … 0 (finest), per batch row b:
//
//   1. (i < coarsest, global) EstimateGlobalMVDoubled on level i+1's
//      POST-SEARCH vectors: kt_most_freq_mv (isY=0 then isY=1, tie-break =
//      upstream staged fold) → kt_mean_global_mv (doubled)
//      (MV.cpp:2336-2339 → MVKernel.cu:2664-2673).
//   2. (i < coarsest) kt_interpolate_prediction level i+1 → level i, writing
//      the fine level's ENTIRE own vector region + incoming sads
//      (MV.cpp:2341 → MVKernel.cu:1547-1627; x-axis weights for both axes).
//      The coarsest level instead starts from the work buffer state — here
//      deterministically ZEROED (upstream: undefined, MV.cpp:2080-2082; the
//      twin and rig zero it, measure report §4).
//   3. Host-written sentinels: vectors[row+2] = (0,0), vectors[row+3] =
//      globalMV·nPelLevel (kl_init_const_vec's arithmetic is verified but its
//      negative sub-relative addressing is unusable on this driver,
//      HOST_CONTRACT §3; upstream MVKernel.cu:1708-1717).
//   4. One Search call: kt_prepare_search (Program B) → ordered per-column
//      kt_search_column dispatches ascending blkx on the in-order queue
//      (Program C, ANALYZE_SYNC=1 fresh left/up) → kt_calc_all_sad
//      (BLOCKSEARCH_MODEL §8, HOST_CONTRACT §3).
//   5. The level's packed VECTOR out array is copied into the BGR32 output
//      frame, COARSEST LEVEL FIRST (MV.cpp:2333-2364 + 2664-2668; consumer
//      parse MV.cpp:3838-3842).
//
// Per-level parameters (contract §1.1/§1.2, MV.cpp:2248-2282):
//   nPelLevel      = (i == 0) ? nPel : 1
//   searchType     = coarse levels FORCED to EXHAUSTIVE unless HSEARCH/
//                    VSEARCH (MV.cpp:2255-2258); device dispatch is on the
//                    enum value (EXHAUSTIVE==8 → exhaustive table, everything
//                    else → the hex table), which the port's kernels spell
//                    SEARCH=1 / SEARCH=2.
//   nLambdaLevel   = nLambda/(nPelLevel²), then *= (1<<i) for plevel==1,
//                    *= (1<<i)² for plevel==2 (MV.cpp:1001-1006).
//   nLogScale = i; nPad = the super's nHPad at EVERY level (8; MV.cpp:553-575).
//
// MEASURE-REPORT §5 DECISIONS (each documented here as required):
//   D1. Coarsest-level global-MV sentinel: the CPU-defined (0,0) — upstream
//       CUDA leaves it UNINITIALIZED (MV.cpp:2080-2082 no-op); the twin and
//       the rig zero it deterministically (contract §8.1).  Implemented as
//       (0,0).
//   D2. WriteDefault frames (n outside [minframe,maxframe)): the INTENDED
//       per-block {0,0,verybigSAD} (verybigSAD = 3*blksize²*(1<<bpp),
//       MV.cpp:999) + KMVIsValid=false.  Upstream CUDA's kl_write_default_mv
//       writes .x twice (MVKernel.cu:1704 typo), so upstream CUDA default
//       frames carry x=verybigSAD(!) and a stale sad — NOT reproduced
//       (MV_PORT_SPEC §6 note).
//   D3. Level count: upstream pins lv=0 (MV.cpp:2820 "levels skip"), so
//       nAnalyzeLevels = the covered-grid nLevelsMax (MV.cpp:2541-2547),
//       validated ≤ the super's nLevels (MV.cpp:2549-2561).  The plugin
//       implements the GENERIC level loop (per-level math identical by
//       construction); the CUDA-twin end-to-end measurement covers 2 levels
//       (BS=8/overlap=4); 3-level configs (e.g. 64×48) are covered by the
//       plugin's headless verifier against an in-exe CPU reference of the
//       proven loop.  Not CUDA-cross-checked beyond 2 levels.
//   D4. batch: the per-row plumbing (vectors/sads/prog/next/out rows) is
//       implemented per contract §10 and stage-1-proven per level, BUT
//       kt_interpolate_prediction's port drops the batch dimension (row
//       pointers only, no grid y), so batch>1 cannot be assembled.  This
//       plugin THROWS on batch != 1 (unmeasured extension, measure report
//       §5; upstream's ANALYZE_MAX_BATCH plumbing stays out of scope).
//   D5. 16-bit: supported (PX=ushort program instantiations; the loop is
//       pixel-size agnostic host-side, lsad/verybigSAD carry the
//       (1<<(bpp-8)) factor).  Not CUDA-twin-measured; covered by the
//       in-exe CPU reference.
//   D6. partial pre-analyzed input (SetMVs path, MV.cpp:2317-2330): REJECTED
//       with an explicit throw.  The [partial]c parameter is kept in the
//       signature for upstream compatibility; the copy path is unmeasured
//       (measure report §5) and trivially replaceable later.
//   D7. Dead-on-the-CUDA-path parameters are KEPT in the signature with
//       upstream defaults and IGNORED, exactly like upstream CUDA: meander
//       (CPU-path scan order only, MV.cpp:1706-1727), badSAD/badrange
//       (normalized then never passed to kernel->Search, MV.cpp:2578/
//       MVKernel.cu:2517-2525), tryMany, and the search parameters (stp
//       fixed 2; nSearchParam/nPelSearch are CPU-path-only knobs).
//   D8. The level's sads work buffer is NOT updated by kt_calc_all_sad (the
//       recomputed SADs go to the output VECTORs only): InterpolatePrediction
//       reads the INCOMING (pre-search) sads — proven by the twin/rig
//       bit-exact interp dumps (the twin's calc dst_sad is a separate buffer,
//       kmanalyse_host.cu:189).
//   D9. No batch-result cache (upstream curBatch/batchFrames): output frame
//       n is recomputed per call — a pure function of child(n), child(n±delta)
//       and the config (KNoiseClipOCL/KMSuperOCL precedent).
//   D10. The BGR32 output frame is 2048 × ceil(out_frame_bytes/8192) rows;
//        only the out_frame_bytes vector bytes are written (row-scattered to
//        the frame pitch); the tail of the last row is whatever
//        NewVideoFrame left (upstream: undefined).
//
// SUPER-FRAME INPUT (produced by KMSuperOCL, inventory in its header): the
// child clip IS the super clip — a planar pyramid whose plane P (nSuperWidth
// for Y, >>logUVx for chroma) holds level L at element offset
// PlaneSuperOffset(chroma, H0, L, pel, vpad, P, yRatioUV), interior origin
// off(L) + P*vpad + hpad; level 0 stacks nPel² sub-pel planes P*(H+2*vpad)
// apart.  Frames are staged row-by-row into tight-pitch SVM pyramid buffers
// (KMSuperOCL's staging, reversed) and the search kernels are handed
// PER-LEVEL PADDED PLANE BASES (upstream GetAbsolutePelPointer(0,0) =
// pPlane[0] incl. border, MV.cpp:2143-2151/426-428; contract §7).
//
// PARAMETERS (upstream KMAnalyse::Create verbatim, MV.cpp:2796-2851;
// registration MV.cpp:5232-5234):
//   clip, blksize(int,16), overlap(int,blksize/2), search(int,4=HEX2SEARCH),
//   isb(bool,false), chroma(bool,true), delta(int,1), lambda(int,400),
//   lsad(int,400), global(bool,true), plevel(int,0), pnew(int,25),
//   meander(bool,true,ignored), partial(clip,unsupported→throw D6),
//   batch(int,1,only 1 supported D4)
//   Upstream validation kept (MV.cpp:2449-2505): float type throw, YUV
//   check, blksizex==blksizey / overlapx==overlapy / blksizex==overlapx*2
//   (fixed by the single blksize/overlap script args), blksize in {8,16,32},
//   pel in {1,2} (from the super), overlap ≤ blksize/2, overlap % subsampling,
//   Y-only input forces chroma=false.
#include <cstring>
#include <algorithm>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

#include "avisynth.h"
#include "../ocl_host.h"
#include "ktgmc_motion_cl.h"  // GENERATED from .../ktgmc/kernels/ktgmc_motion.cl
#include "ktgmc_search_cl.h"  // GENERATED from .../ktgmc/kernels/ktgmc_search.cl
#include "ktgmc_kmparam.h"    // KMVParam smuggling (upstream common/KMV.h)

namespace {

using oclhost::Context;
using oclhost::Program;
using oclhost::SvmBuffer;

Context& sharedContext() {
    static Context ctx;  // one device/queue per process
    return ctx;
}

// ---------------------------------------------------------------------------
// MV.cpp geometry helpers (verbatim transcription; shared with KMSuperOCL)
// ---------------------------------------------------------------------------
int PlaneHeightLuma(int src_height, int level, int yRatioUV, int vpad) {
    int height = src_height;
    for (int i = 1; i <= level; i++)
        height = vpad >= yRatioUV ? ((height / yRatioUV + 1) / 2) * yRatioUV
                                  : ((height / yRatioUV) / 2) * yRatioUV;
    return height;
}

int PlaneWidthLuma(int src_width, int level, int xRatioUV, int hpad) {
    int width = src_width;
    for (int i = 1; i <= level; i++)
        width = hpad >= xRatioUV ? ((width / xRatioUV + 1) / 2) * xRatioUV
                                 : ((width / xRatioUV) / 2) * xRatioUV;
    return width;
}

unsigned int PlaneSuperOffset(bool chroma, int src_height, int level, int pel,
                              int vpad, int plane_pitch, int yRatioUV) {
    int height = src_height;
    unsigned int offset;
    if (level == 0) {
        offset = 0;
    } else {
        offset = (unsigned int)pel * pel * plane_pitch * (src_height + vpad * 2);
        for (int i = 1; i < level; i++) {
            height = chroma
                ? PlaneHeightLuma(src_height * yRatioUV, i, yRatioUV, vpad * yRatioUV) / yRatioUV
                : PlaneHeightLuma(src_height, i, yRatioUV, vpad);
            offset += (unsigned int)plane_pitch * (height + vpad * 2);
        }
    }
    return offset;
}

static int nblocks(int v, int block) { return (v + block - 1) / block; }

// upstream VECTOR (common/KMV.h:6-11): three ints {x, y, sad} — the output
// frame is a byte-exact array of these (BGR32, 2048 px = 8192 B rows)
struct VECTOR12 { cl_int x, y, sad; };

// upstream SearchType enum (MV.cpp:859-868) and the script-arg mapping
// (MV.cpp:2586-2619, verbatim)
enum SearchType {
    KT_ONETIME = 1,
    KT_NSTEP = 2,
    KT_LOGARITHMIC = 4,
    KT_EXHAUSTIVE = 8,
    KT_HEX2SEARCH = 16,
    KT_UMHSEARCH = 32,
    KT_HSEARCH = 64,
    KT_VSEARCH = 128
};

static SearchType mapSearchArg(int st) {
    switch (st) {
    case 0:  return KT_ONETIME;
    case 1:  return KT_NSTEP;
    case 3:  return KT_EXHAUSTIVE;
    case 4:  return KT_HEX2SEARCH;
    case 5:  return KT_UMHSEARCH;
    case 6:  return KT_HSEARCH;
    case 7:  return KT_VSEARCH;
    case 2:
    default: return KT_LOGARITHMIC;
    }
}

// ---------------------------------------------------------------------------
// one analysed level's description + device work buffers
// ---------------------------------------------------------------------------
struct LevelWork {
    // geometry / params (contract §1.2)
    int nBlkX = 0, nBlkY = 0, nBlkCount = 0;
    int nPelLevel = 1, nLogScale = 0, nLambdaLevel = 0;
    int deviceSearch = 2;   // 1 = exhaustive table, 2 = hex table
    int nExtW = 0, nExtH = 0;
    int offY = 0, offU = 0;               // padded plane base offsets (elements)
    int Hl = 0, Hc = 0;                   // interior heights (luma/chroma)

    // buffers (SVM; batch == 1, D4)
    SvmBuffer vec;       // int2 alloc, pitch 4+2n: [-2 zero][-1 global][own n][copy n]
    SvmBuffer sads;      // int, pitch 4+n (incoming; never updated by calc_all_sad, D8)
    SvmBuffer data;      // 12n ints
    SvmBuffer dataf;     // 5n ints
    SvmBuffer prog;      // nBlkX ints
    SvmBuffer next;      // 1 int
    SvmBuffer shortvec;  // short2 per block (post-search own, calc_all_sad input)
    SvmBuffer sadout;    // n ints (calc_all_sad dst_sad, discarded, D8)
    SvmBuffer out;       // 3n ints (VECTOR x,y,sad, calc_all_sad out)
};

class KMAnalyseOCL : public GenericVideoFilter {
    // ---- config (upstream MV.cpp:2430-2445 + 2468-2619) ----
    int nBlkSizeX_, nBlkSizeY_, nOverlapX_, nOverlapY_;
    int nDeltaFrame_;
    bool isBackward_, chroma_;
    int nLambda_, lsad_, plevel_, pnew_, penaltyZero_, pglobal_;
    bool global_;
    int maxBatch_;
    SearchType searchType_;

    // ---- super clip params (KMVParam smuggled by KMSuperOCL) ----
    const KMVParam* super_;
    int nPel_, xRatioUV_, yRatioUV_, logUVx_, logUVy_;
    int nHPad_, nVPad_;
    int nPixelSize_, nBitsPerPixel_, nPixelShift_;
    int origW_, origH_;             // pre-super frame dims (params.nWidth/Height)
    int superW_, superH_;           // super clip dims (the child's vi)
    int pitchY_, pitchUV_, planeHY_, planeHUV_;
    int vpadU_, hpadU_;

    // ---- analyse pyramid ----
    int nAnalyzeLevels_;
    std::vector<LevelWork> lv_;
    int verybigSAD_ = 0;
    int outFrameBytes_ = 0;
    SvmBuffer srcY_, srcU_, srcV_, refY_, refU_, refV_;  // super plane staging
    SvmBuffer globalMV_;            // int2 per batch row (contract §2)

    std::vector<uint8_t> stage_;
    std::mutex* mtx_;

    Program& motionProgram() {
        return oclhost::ProgramRegistry::instance().get(
            sharedContext(), "ktgmc_motion", ktgmc_motion_cl_src, nPixelSize_);
    }
    Program& searchProgram() {
        return oclhost::ProgramRegistry::instance().get(
            sharedContext(), "ktgmc_search", ktgmc_search_cl_src, nPixelSize_);
    }

    // ------------------------------------------------------------------
    // per-frame level loop (contract §10; batch == 1, D4)
    // ------------------------------------------------------------------
    void runLoop(int nsrc, int nref, uint8_t* outFrame, int framePitch,
                 IScriptEnvironment* env) {
        Context& ctx = sharedContext();
        const int px = nPixelSize_;
        const int batch = maxBatch_;  // == 1 (D4)
        Program& PM = motionProgram();
        Program& KS = searchProgram();

        // ---- stage the two super frames into tight-pitch SVM pyramids ----
        auto stageFrame = [&](PVideoFrame& f, SvmBuffer& dY, SvmBuffer& dU,
                              SvmBuffer& dV) {
            auto copyPlane = [&](SvmBuffer& dst, const uint8_t* sp, int spitch,
                                 int w, int h) {
                const size_t rowbytes = (size_t)w * px;
                for (int y = 0; y < h; y++)
                    memcpy(stage_.data() + (size_t)y * rowbytes,
                           sp + (size_t)y * spitch, rowbytes);
                dst.write(stage_.data(), 0, rowbytes * h);
            };
            copyPlane(dY, f->GetReadPtr(PLANAR_Y), f->GetPitch(PLANAR_Y),
                      superW_, planeHY_);
            if (super_->chroma) {
                copyPlane(dU, f->GetReadPtr(PLANAR_U), f->GetPitch(PLANAR_U),
                          superW_ >> logUVx_, planeHUV_);
                copyPlane(dV, f->GetReadPtr(PLANAR_V), f->GetPitch(PLANAR_V),
                          superW_ >> logUVx_, planeHUV_);
            }
        };
        {
            PVideoFrame sf = child->GetFrame(nsrc, env);
            stageFrame(sf, srcY_, srcU_, srcV_);
            PVideoFrame rf = child->GetFrame(nref, env);
            stageFrame(rf, refY_, refU_, refV_);
        }
        ctx.finish();

        const int chromaK = super_->chroma ? 1 : 0;

        for (int i = nAnalyzeLevels_ - 1; i >= 0; i--) {
            LevelWork& L = lv_[i];
            const int n = L.nBlkCount;
            const int vecPitch = 4 + 2 * n;
            const int sadPitch = 4 + n;
            const size_t allocBytes = sizeof(cl_int2) * vecPitch * batch;
            const size_t sadBytes = sizeof(cl_int) * sadPitch * batch;

            // ---- 1. global-MV chain on level i+1 (contract §2) ----
            if (i < nAnalyzeLevels_ - 1 && global_) {
                LevelWork& C = lv_[i + 1];
                const int cn = C.nBlkCount;
                const int cPitch = 4 + 2 * cn;
                void* cOwn = C.vec.at(4 * sizeof(cl_int2));  // own-region base
                cl_kernel kfreq = PM.kernel("kt_most_freq_mv");
                oclhost::set_arg_svm(kfreq, 0, cOwn);
                oclhost::set_arg(kfreq, 1, cPitch);
                oclhost::set_arg(kfreq, 2, cn);
                oclhost::set_arg(kfreq, 3, 0);               // isY = 0 (x mode)
                oclhost::set_arg_svm(kfreq, 4, globalMV_.ptr());
                oclhost::dispatch2D(ctx, kfreq, 1, batch);
                oclhost::set_arg(kfreq, 3, 1);               // isY = 1 (y mode)
                oclhost::dispatch2D(ctx, kfreq, 1, batch);

                cl_kernel kmean = PM.kernel("kt_mean_global_mv");
                oclhost::set_arg_svm(kmean, 0, cOwn);
                oclhost::set_arg(kmean, 1, cPitch);
                oclhost::set_arg(kmean, 2, cn);
                oclhost::set_arg_svm(kmean, 3, globalMV_.ptr());
                oclhost::dispatch2D(ctx, kmean, 1, batch);
                ctx.finish();
            }

            // ---- 2. coarse→fine prediction (contract §3) ----
            if (i < nAnalyzeLevels_ - 1) {
                LevelWork& C = lv_[i + 1];
                const int cn = C.nBlkCount;
                const int cPitch = 4 + 2 * cn;
                const int cSadPitch = 4 + cn;
                const int normFactor = 3 - nlog2(L.nPelLevel) + nlog2(C.nPelLevel);
                const int normov = (nBlkSizeX_ - nOverlapX_) * (nBlkSizeY_ - nOverlapY_);
                const int atotal = (nBlkSizeX_ - nOverlapX_) * 4;
                const int aodd = 3 * nBlkSizeX_ - 2 * nOverlapX_;
                const int aeven = 3 * nBlkSizeX_ - 4 * nOverlapX_;
                cl_kernel k = PM.kernel("kt_interpolate_prediction");
                oclhost::set_arg_svm(k, 0, C.vec.at(4 * sizeof(cl_int2)));
                oclhost::set_arg_svm(k, 1, C.sads.at(4 * sizeof(cl_int)));
                oclhost::set_arg_svm(k, 2, L.vec.at(4 * sizeof(cl_int2)));
                oclhost::set_arg_svm(k, 3, L.sads.at(4 * sizeof(cl_int)));
                oclhost::set_arg(k, 4, C.nBlkX);
                oclhost::set_arg(k, 5, C.nBlkY);
                oclhost::set_arg(k, 6, L.nBlkX);
                oclhost::set_arg(k, 7, L.nBlkY);
                oclhost::set_arg(k, 8, normFactor);
                oclhost::set_arg(k, 9, normov);
                oclhost::set_arg(k, 10, atotal);
                oclhost::set_arg(k, 11, aodd);
                oclhost::set_arg(k, 12, aeven);
                oclhost::dispatch2D(ctx, k, L.nBlkX, L.nBlkY);
                ctx.finish();
            } else {
                // D1/D8: deterministic zero substitution for the undefined
                // coarsest work buffer (twin/rig behaviour).
                std::vector<uint8_t> z(allocBytes + sadBytes, 0);
                L.vec.write(z.data(), 0, allocBytes);
                L.sads.write(z.data(), 0, sadBytes);
            }

            // ---- 3. host-written sentinels (slot -2 / slot -1) ----
            {
                std::vector<cl_int2> gmv(batch);
                if (i < nAnalyzeLevels_ - 1 && global_) {
                    globalMV_.read(gmv.data(), 0, sizeof(cl_int2) * batch);
                } else {
                    for (int b = 0; b < batch; b++) { gmv[b].x = 0; gmv[b].y = 0; }  // D1
                }
                std::vector<cl_int2> sent(2 * batch);
                for (int b = 0; b < batch; b++) {
                    sent[2 * b + 0].x = 0;                          // slot -2: zero MV
                    sent[2 * b + 0].y = 0;
                    sent[2 * b + 1].x = gmv[b].x * L.nPelLevel;     // slot -1: globalMV·nPel
                    sent[2 * b + 1].y = gmv[b].y * L.nPelLevel;
                }
                L.vec.write(sent.data(), 0, sizeof(cl_int2) * 2 * batch);
                // batch == 1 (D4): only row 0's slots exist to write; the
                // per-row generalisation is dropped with the batch dimension.
            }

            // ---- 4a. prepare (Program B) ----
            {
                cl_kernel k = PM.kernel("kt_prepare_search");
                oclhost::set_arg(k, 0, L.nBlkX);
                oclhost::set_arg(k, 1, L.nBlkY);
                oclhost::set_arg(k, 2, nBlkSizeX_);
                oclhost::set_arg(k, 3, L.nLogScale);
                oclhost::set_arg(k, 4, L.nLambdaLevel);
                oclhost::set_arg(k, 5, lsad_);
                oclhost::set_arg(k, 6, penaltyZero_);
                oclhost::set_arg(k, 7, pglobal_);
                oclhost::set_arg(k, 8, pnew_);
                oclhost::set_arg(k, 9, L.nPelLevel);
                oclhost::set_arg(k, 10, nHPad_);
                oclhost::set_arg(k, 11, nBlkSizeX_ - nOverlapX_);
                oclhost::set_arg(k, 12, L.nExtW);
                oclhost::set_arg(k, 13, L.nExtH);
                oclhost::set_arg_svm(k, 14, L.vec.at(4 * sizeof(cl_int2)));      // own
                oclhost::set_arg_svm(k, 15, L.sads.at(4 * sizeof(cl_int)));      // sads own
                oclhost::set_arg_svm(k, 16, L.vec.at((4 + n) * sizeof(cl_int2)));// copy
                oclhost::set_arg_svm(k, 17, L.data.ptr());
                oclhost::set_arg_svm(k, 18, L.dataf.ptr());
                oclhost::set_arg_svm(k, 19, L.prog.ptr());
                oclhost::set_arg_svm(k, 20, L.next.ptr());
                oclhost::dispatch2D(ctx, k, L.nBlkX, L.nBlkY);
            }

            // ---- 4b. ordered per-column search (Program C) ----
            for (int blkx = 0; blkx < L.nBlkX; blkx++) {
                cl_kernel k = KS.kernel("kt_search_column");
                oclhost::set_arg_svm(k, 0, srcY_.at((size_t)L.offY * px));
                oclhost::set_arg_svm(k, 1, chromaK ? srcU_.at((size_t)L.offU * px) : srcY_.ptr());
                oclhost::set_arg_svm(k, 2, chromaK ? srcV_.at((size_t)L.offU * px) : srcY_.ptr());
                oclhost::set_arg_svm(k, 3, refY_.at((size_t)L.offY * px));
                oclhost::set_arg_svm(k, 4, chromaK ? refU_.at((size_t)L.offU * px) : refY_.ptr());
                oclhost::set_arg_svm(k, 5, chromaK ? refV_.at((size_t)L.offU * px) : refY_.ptr());
                oclhost::set_arg_svm(k, 6, L.vec.ptr());               // alloc base
                oclhost::set_arg_svm(k, 7, L.data.ptr());
                oclhost::set_arg_svm(k, 8, L.dataf.ptr());
                oclhost::set_arg(k, 9, blkx);
                oclhost::set_arg(k, 10, L.nBlkX);
                oclhost::set_arg(k, 11, L.nBlkY);
                oclhost::set_arg(k, 12, nHPad_);
                oclhost::set_arg(k, 13, nBlkSizeX_);
                oclhost::set_arg(k, 14, L.deviceSearch);
                oclhost::set_arg(k, 15, L.nPelLevel);
                oclhost::set_arg(k, 16, chromaK);
                oclhost::set_arg(k, 17, pitchY_);
                oclhost::set_arg(k, 18, chromaK ? pitchUV_ : 0);
                oclhost::set_arg(k, 19, pitchY_ * L.nExtH);
                oclhost::set_arg(k, 20, chromaK ? pitchUV_ * (L.Hc + 2 * vpadU_) : 0);
                oclhost::dispatch1D(ctx, k, 1);
            }

            // ---- 4c. build the short2 post-search own region + calc_all_sad ----
            {
                // upstream keeps the vectors allocation as native short2; the
                // port's search writes int2, so the components are narrowed
                // exactly like the rig harness (they always fit: search MVs
                // are clip-rect bounded)
                std::vector<cl_int2> ov((size_t)n * batch);
                L.vec.read(ov.data(), 4 * sizeof(cl_int2), sizeof(cl_int2) * n * batch);
                std::vector<cl_short2> sv((size_t)n * batch);
                for (size_t k = 0; k < (size_t)n * batch; k++) {
                    sv[k].x = (cl_short)ov[k].x;
                    sv[k].y = (cl_short)ov[k].y;
                }
                L.shortvec.write(sv.data(), 0, sizeof(cl_short2) * n * batch);

                cl_kernel k = PM.kernel("kt_calc_all_sad");
                oclhost::set_arg_svm(k, 0, srcY_.at((size_t)L.offY * px));
                oclhost::set_arg_svm(k, 1, chromaK ? srcU_.at((size_t)L.offU * px) : srcY_.ptr());
                oclhost::set_arg_svm(k, 2, chromaK ? srcV_.at((size_t)L.offU * px) : srcY_.ptr());
                oclhost::set_arg_svm(k, 3, refY_.at((size_t)L.offY * px));
                oclhost::set_arg_svm(k, 4, chromaK ? refU_.at((size_t)L.offU * px) : refY_.ptr());
                oclhost::set_arg_svm(k, 5, chromaK ? refV_.at((size_t)L.offU * px) : refY_.ptr());
                oclhost::set_arg_svm(k, 6, L.shortvec.ptr());
                oclhost::set_arg_svm(k, 7, L.sadout.ptr());
                oclhost::set_arg_svm(k, 8, L.out.ptr());
                oclhost::set_arg(k, 9, L.nBlkX);
                oclhost::set_arg(k, 10, L.nBlkY);
                oclhost::set_arg(k, 11, nHPad_);
                oclhost::set_arg(k, 12, nBlkSizeX_);
                oclhost::set_arg(k, 13, L.nPelLevel);
                oclhost::set_arg(k, 14, chromaK);
                oclhost::set_arg(k, 15, pitchY_);
                oclhost::set_arg(k, 16, chromaK ? pitchUV_ : 0);
                oclhost::set_arg(k, 17, pitchY_ * L.nExtH);
                oclhost::set_arg(k, 18, chromaK ? pitchUV_ * (L.Hc + 2 * vpadU_) : 0);
                oclhost::dispatch2D(ctx, k, L.nBlkX, L.nBlkY);
                ctx.finish();
            }

            // ---- 5. copy the level's VECTORs into the BGR32 frame ----
            // (coarsest level first; this loop's order IS the layout)
            {
                std::vector<cl_int> o((size_t)3 * n * batch);
                L.out.read(o.data(), 0, sizeof(cl_int) * 3 * n * batch);
                // region byte offset = the count of VECTORs already placed
                int placed = 0;
                for (int j = nAnalyzeLevels_ - 1; j > i; j--)
                    placed += lv_[j].nBlkCount;
                copyVectorsToFrame(outFrame, framePitch, o.data(),
                                   (size_t)3 * n * batch, (size_t)placed * 3);
            }
        }
    }

    // scatter the packed 3-int VECTOR triples into the frame's byte grid
    // (the frame is a 2048-px-wide BGR32 image; upstream treats it as a flat
    // byte array of out_frame_bytes, MV.cpp:2664-2668)
    static void copyVectorsToFrame(uint8_t* frame, int pitch,
                                   const cl_int* triples, size_t count3,
                                   size_t dstWord) {
        for (size_t k = 0; k < count3; k++) {
            size_t bytePos = dstWord * sizeof(cl_int) + k * sizeof(cl_int);
            uint8_t* p = frame + (size_t)(bytePos / (size_t)pitch) * pitch
                             + (bytePos % (size_t)pitch);
            cl_int v = triples[k];
            memcpy(p, &v, sizeof(cl_int));
        }
    }

    static int nlog2(int v) { int l = 0; while ((1 << l) < v) l++; return l; }

public:
    KMAnalyseOCL(PClip childClip, PClip partial, int blksize, int overlap,
                 int searchArg, bool isb, bool chroma, int delta, int lambda,
                 int lsad, int plevel, bool global, int pnew, int batch,
                 IScriptEnvironment* env_)
        : GenericVideoFilter(childClip)
        , nBlkSizeX_(blksize), nBlkSizeY_(blksize)
        , nOverlapX_(overlap), nOverlapY_(overlap)
        , nDeltaFrame_(delta)
        , isBackward_(isb)
        , nLambda_(lambda), plevel_(plevel), pnew_(pnew)
        , global_(global), maxBatch_(batch)
        , searchType_(mapSearchArg(searchArg))
        , super_(nullptr)
        , ownParams_(KMVParam::MV_FRAME) {
        // ---- upstream validation (MV.cpp:2449-2505, verbatim behaviour) ----
        if (vi.ComponentSize() == 4)
            env_->ThrowError("KMAnalyse: Clip with float pixel type is not supported");
        if (!vi.IsYUV() && !vi.IsY())
            env_->ThrowError("KMAnalyse: Clip must be YUV");
        if (nBlkSizeX_ != nBlkSizeY_)
            env_->ThrowError("KMAnalyse: blksizex != blksizey");
        if (nOverlapX_ != nOverlapY_)
            env_->ThrowError("KMAnalyse: overlapx != overlapy");
        if (nBlkSizeX_ != nOverlapX_ * 2)
            env_->ThrowError("KMAnalyse: blksizex != overlapx * 2");
        if (nBlkSizeX_ != 8 && nBlkSizeX_ != 16 && nBlkSizeX_ != 32)
            env_->ThrowError("KMAnalyse: blksizex must be 8 or 16 or 32");
        if (partial)
            env_->ThrowError("KMAnalyse: partial is not supported by the "
                             "OpenCL port (unmeasured extension, "
                             "KMANALYSE_MEASURE_REPORT §5 / deviation D6)");
        if (batch != 1)
            env_->ThrowError("KMAnalyse: only batch=1 is supported by the "
                             "OpenCL port (unmeasured extension, "
                             "KMANALYSE_MEASURE_REPORT §5 / deviation D4)");

        // the child MUST be a KMSuper(OCL) clip (upstream reads the super's
        // KMVParam through the same vi smuggling, common/KMV.h)
        try {
            super_ = KMVParam::GetParam(vi, env_);
        } catch (const AvisynthError&) {
            env_->ThrowError("KMAnalyse: Invalid source (sample_type signature "
                             "does not match); the clip must be a KMSuper(OCL) "
                             "super frame");
        }
        if (super_->nPixelSize != vi.ComponentSize())
            env_->ThrowError("KMAnalyse: super clip pixel size mismatch");

        nPel_ = super_->nPel;
        if (nPel_ != 1 && nPel_ != 2)
            env_->ThrowError("KMAnalyse: pel has to be 1 or 2 or 4");
        if (nOverlapX_ < 0 || nOverlapX_ > nBlkSizeX_ / 2)
            env_->ThrowError("KMAnalyse: overlap must be less or equal than half block size");
        if (nOverlapX_ % super_->xRatioUV || nOverlapY_ % super_->yRatioUV)
            env_->ThrowError("KMAnalyse: wrong overlap for the colorspace subsampling");

        chroma_ = chroma;
        if (vi.IsY()) chroma_ = false;
        if (chroma_ && !super_->chroma)
            env_->ThrowError("KMAnalyse: chroma=true but the super clip was "
                             "built with chroma=false");
        if (!chroma_ && super_->chroma) {
            // allowed (upstream: chroma is a per-filter knob; the search then
            // ignores the super's chroma planes)
        }

        origW_ = super_->nWidth;
        origH_ = super_->nHeight;
        xRatioUV_ = super_->xRatioUV;
        yRatioUV_ = super_->yRatioUV;
        logUVx_ = nlog2(xRatioUV_);
        logUVy_ = nlog2(yRatioUV_);
        nPixelSize_ = super_->nPixelSize;
        nBitsPerPixel_ = super_->nBitsPerPixel;
        nPixelShift_ = super_->nPixelShift;
        nHPad_ = super_->nHPad;
        nVPad_ = super_->nVPad;
        hpadU_ = nHPad_ / xRatioUV_;
        vpadU_ = nVPad_ / yRatioUV_;
        superW_ = vi.width;
        superH_ = vi.height;
        pitchY_ = superW_;
        pitchUV_ = superW_ >> logUVx_;
        planeHY_ = superH_;
        planeHUV_ = superH_ >> logUVy_;

        if (super_->nLevels < 1)
            env_->ThrowError("KMAnalyse: super clip has no levels");

        // ---- covered block grid (MV.cpp:2508-2514, verbatim) ----
        const int nBlkX0 = (origW_ - nOverlapX_) / (nBlkSizeX_ - nOverlapX_);
        const int nBlkY0 = (origH_ - nOverlapY_) / (nBlkSizeY_ - nOverlapY_);
        const int nWidth_B = (nBlkSizeX_ - nOverlapX_) * nBlkX0 + nOverlapX_;
        const int nHeight_B = (nBlkSizeY_ - nOverlapY_) * nBlkY0 + nOverlapY_;

        // ---- valid level count (MV.cpp:2541-2547, verbatim; lv pinned 0) ----
        int nLevelsMax = 0;
        while (((nWidth_B >> nLevelsMax) - nOverlapX_) / (nBlkSizeX_ - nOverlapX_) > 0
            && ((nHeight_B >> nLevelsMax) - nOverlapY_) / (nBlkSizeY_ - nOverlapY_) > 0) {
            ++nLevelsMax;
        }
        // upstream lv is pinned to 0 by KMAnalyse::Create (MV.cpp:2820) →
        // nAnalyzeLevel = nLevelsMax + 0
        int nAnalyzeLevel = nLevelsMax;
        if (nAnalyzeLevel > super_->nLevels)
            env_->ThrowError("MAnalyse: it is not enough levels  in super clip (%d), "
                             "while MAnalyse asks %d", super_->nLevels, nAnalyzeLevel);
        if (nAnalyzeLevel < 1 || nAnalyzeLevel > nLevelsMax)
            env_->ThrowError("MAnalyse: non-valid number of levels (%d)", nAnalyzeLevel);
        nAnalyzeLevels_ = nAnalyzeLevel;

        // ---- per-level levelInfo (MV.cpp:2567-2573) + plane params ----
        penaltyZero_ = pnew_;      // MV.cpp:2832 (pzero = pnew)
        pglobal_ = 0;              // MV.cpp:2833
        lsad_ = lsad * (nBlkSizeX_ * nBlkSizeY_) / 64
                * (1 << (nBitsPerPixel_ - 8));   // MV.cpp:2577
        verybigSAD_ = 3 * nBlkSizeX_ * nBlkSizeY_
                      * (1 << nBitsPerPixel_);     // MV.cpp:999 (8/16-bit)

        lv_.resize(nAnalyzeLevels_);
        for (int i = 0; i < nAnalyzeLevels_; i++) {
            LevelWork& L = lv_[i];
            L.nBlkX = ((nWidth_B >> i) - nOverlapX_) / (nBlkSizeX_ - nOverlapX_);
            L.nBlkY = ((nHeight_B >> i) - nOverlapY_) / (nBlkSizeY_ - nOverlapY_);
            L.nBlkCount = L.nBlkX * L.nBlkY;
            int nActualLevel = i;                       // nDropLevels == 0 (MV.cpp:2820)
            L.nPelLevel = (nActualLevel == 0) ? nPel_ : 1;   // MV.cpp:2250
            // searchType per level (MV.cpp:2255-2258) → device dispatch
            // (MVKernel.cu:2616: EXHAUSTIVE(8) → exhaustive table, else hex)
            SearchType st = (nActualLevel == 0 || searchType_ == KT_HSEARCH
                             || searchType_ == KT_VSEARCH)
                            ? searchType_ : KT_EXHAUSTIVE;
            L.deviceSearch = (st == KT_EXHAUSTIVE) ? 1 : 2;
            L.nLogScale = i;
            L.nLambdaLevel = nLambda_ / (L.nPelLevel * L.nPelLevel);  // MV.cpp:1001
            int nScale = 1 << i;
            if (plevel_ == 1) L.nLambdaLevel *= nScale;
            else if (plevel_ == 2) L.nLambdaLevel *= nScale * nScale;
            L.Hl = (i == 0) ? origH_ : PlaneHeightLuma(origH_, i, yRatioUV_, nVPad_);
            int Wl = (i == 0) ? origW_ : PlaneWidthLuma(origW_, i, xRatioUV_, nHPad_);
            L.Hc = L.Hl / yRatioUV_;
            int Wc = Wl / xRatioUV_;
            L.nExtW = Wl + 2 * nHPad_;   // MV.cpp:382
            L.nExtH = L.Hl + 2 * nVPad_;
            L.offY = (int)PlaneSuperOffset(false, origH_, i, nPel_, nVPad_, pitchY_, yRatioUV_);
            L.offU = (int)PlaneSuperOffset(true, origH_ / yRatioUV_, i, nPel_, vpadU_,
                                           pitchUV_, yRatioUV_);
        }

        // ---- output vi: BGR32 VECTOR array (MV.cpp:2664-2668) ----
        outFrameBytes_ = 0;
        for (int i = 0; i < nAnalyzeLevels_; i++)
            outFrameBytes_ += lv_[i].nBlkCount * (int)sizeof(VECTOR12);
        vi.pixel_type = VideoInfo::CS_BGR32;
        vi.width = 2048;
        vi.height = nblocks(outFrameBytes_, vi.width * 4);

        // ---- smuggle the MV_FRAME params for downstream KMCompensate etc.
        //      (upstream MV.cpp:2468/2575) ----
        ownParams_ = *super_;
        ownParams_.nDataType = KMVParam::MV_FRAME;
        ownParams_.chroma = chroma_;
        ownParams_.nBlkSizeX = nBlkSizeX_;
        ownParams_.nBlkSizeY = nBlkSizeY_;
        ownParams_.nOverlapX = nOverlapX_;
        ownParams_.nOverlapY = nOverlapY_;
        ownParams_.nDeltaFrame = nDeltaFrame_;
        ownParams_.isBackward = isBackward_;
        ownParams_.nAnalyzeLevels = nAnalyzeLevels_;
        for (int i = 0; i < nAnalyzeLevels_; i++) {
            ownParams_.levelInfo[i].nBlkX = lv_[i].nBlkX;
            ownParams_.levelInfo[i].nBlkY = lv_[i].nBlkY;
        }
        ownParams_.chromaSADScale = 0;
        KMVParam::SetParam(vi, &ownParams_);

        // ---- device buffers ----
        static std::mutex mtx;
        mtx_ = &mtx;
        try {
            Context& ctx = sharedContext();
            const size_t px = (size_t)nPixelSize_;
            const size_t bytesY = (size_t)pitchY_ * planeHY_ * px;
            const size_t bytesUV = (size_t)pitchUV_ * planeHUV_ * px;
            srcY_ = SvmBuffer(ctx, bytesY);
            refY_ = SvmBuffer(ctx, bytesY);
            if (super_->chroma) {
                srcU_ = SvmBuffer(ctx, bytesUV);
                srcV_ = SvmBuffer(ctx, bytesUV);
                refU_ = SvmBuffer(ctx, bytesUV);
                refV_ = SvmBuffer(ctx, bytesUV);
            }
            stage_.resize(std::max(bytesY, bytesUV));
            globalMV_ = SvmBuffer(ctx, sizeof(cl_int2) * maxBatch_);
            for (LevelWork& L : lv_) {
                const size_t n = L.nBlkCount;
                L.vec = SvmBuffer(ctx, sizeof(cl_int2) * (4 + 2 * n));
                L.sads = SvmBuffer(ctx, sizeof(cl_int) * (4 + n));
                L.data = SvmBuffer(ctx, sizeof(cl_int) * 12 * n);
                L.dataf = SvmBuffer(ctx, sizeof(cl_int) * 5 * n);
                L.prog = SvmBuffer(ctx, sizeof(cl_int) * L.nBlkX);
                L.next = SvmBuffer(ctx, sizeof(cl_int));
                L.shortvec = SvmBuffer(ctx, sizeof(cl_short2) * n);
                L.sadout = SvmBuffer(ctx, sizeof(cl_int) * n);
                L.out = SvmBuffer(ctx, sizeof(cl_int) * 3 * n);
            }
            motionProgram();   // force both builds up front (both PX of this size)
            searchProgram();
        } catch (const std::exception& e) {
            env_->ThrowError("[KMAnalyseOCL] OpenCL init failed: %s", e.what());
        }
    }

    PVideoFrame __stdcall GetFrame(int n, IScriptEnvironment* env_) {
        try {
            return GetFrameT(n, env_);
        } catch (const std::exception& e) {
            env_->ThrowError("[KMAnalyseOCL] %s", e.what());
        }
        return PVideoFrame();
    }

    int __stdcall SetCacheHints(int cachehints, int frame_range) {
        // mirror upstream KMAnalyse::SetCacheHints (MV.cpp:2787-2794)
        if (cachehints == CACHE_GET_MTMODE) return MT_MULTI_INSTANCE;
        return 0;
    }

private:
    KMVParam ownParams_;  // MV_FRAME params smuggled to downstream filters

    PVideoFrame GetFrameT(int n, IScriptEnvironment* env_) {
        const int nbr_src_frames = child->GetVideoInfo().num_frames;
        const int offset = (isBackward_) ? nDeltaFrame_ : -nDeltaFrame_;
        const int minframe = std::max(-offset, 0);
        const int maxframe = nbr_src_frames + std::min(-offset, 0);

        PVideoFrame dst = env_->NewVideoFrame(vi);
        uint8_t* pDst = dst->GetWritePtr();
        const int pitch = dst->GetPitch();

        if (n < minframe || n >= maxframe) {
            // WriteDefault (contract §5/D2): {0,0,verybigSAD} per block,
            // coarsest level first
            std::vector<cl_int> def;
            def.reserve(outFrameBytes_ / 4);
            for (int i = nAnalyzeLevels_ - 1; i >= 0; i--) {
                for (int k = 0; k < lv_[i].nBlkCount; k++) {
                    def.push_back(0);
                    def.push_back(0);
                    def.push_back(verybigSAD_);
                }
            }
            scatterFrame(pDst, pitch, def.data(), (size_t)outFrameBytes_ / 4);
            AVSMap* m = env_->getFramePropsRW(dst);
            env_->propSetInt(m, "KMVIsValid", 0, AVSPropAppendMode::PROPAPPENDMODE_REPLACE);
            return dst;
        }

        std::lock_guard<std::mutex> lock(*mtx_);
        runLoop(n, n + offset, pDst, pitch, env_);

        AVSMap* m = env_->getFramePropsRW(dst);
        env_->propSetInt(m, "KMVIsValid", 1, AVSPropAppendMode::PROPAPPENDMODE_REPLACE);
        return dst;
    }

    // scatter count 12-byte VECTORs (packed x,y,sad triples) into the frame
    void scatterFrame(uint8_t* frame, int pitch, const cl_int* triples, size_t count) {
        copyVectorsToFrame(frame, pitch, triples, (size_t)count * 3, 0);
    }
};

}  // namespace

// creator (upstream KMAnalyse::Create, MV.cpp:2796-2851; the script-visible
// arg order/defaults are verbatim, the dead-on-CUDA-path knobs are kept and
// ignored per deviation D7)
static AVSValue __cdecl CreateKMAnalyseOCL(AVSValue args, void*, IScriptEnvironment* env) {
    try {
        int blksize = args[1].AsInt(16);          // block size (square)
        int overlap = args[2].AsInt(blksize / 2); // overlap (default blksize/2)
        if (args[13].Defined())                   // partial super clip: D6
            env->ThrowError("KMAnalyse: partial is not supported by the "
                            "OpenCL port (unmeasured extension, deviation D6)");
        return new KMAnalyseOCL(
            args[0].AsClip(),       // super
            args[13].Defined() ? args[13].AsClip() : nullptr,  // partial (unused)
            blksize,
            overlap,                // overlapx/y (MV.cpp:2806/2834-2835)
            args[3].AsInt(4),       // search type (4 = HEX2SEARCH)
            args[4].AsBool(false),  // isb
            args[5].AsBool(true),   // chroma
            args[6].AsInt(1),       // delta
            args[7].AsInt(400),     // lambda
            args[8].AsInt(400),     // lsad
            args[10].AsInt(0),      // plevel
            args[9].AsBool(true),   // global
            args[11].AsInt(25),     // pnew
            args[14].AsInt(1),      // batch
            env);
    } catch (const std::exception& e) {
        env->ThrowError("[KMAnalyseOCL] %s", e.what());
    }
    return AVSValue();
}

// AviSynth+ linkage (no import lib): core passes the function table.
const AVS_Linkage* AVS_linkage{ nullptr };

extern "C" __declspec(dllexport) const char* __stdcall
AvisynthPluginInit3(IScriptEnvironment* env, const AVS_Linkage* const vectors) {
    AVS_linkage = vectors;
    try {
        // exact upstream signature (MV.cpp:5232-5234)
        const char* sig =
            "c[blksize]i[overlap]i[search]i[isb]b[chroma]b[delta]i[lambda]i"
            "[lsad]i[global]b[plevel]i[pnew]i[meander]b[partial]c[batch]i";
        env->AddFunction("KMAnalyseOCL", sig, CreateKMAnalyseOCL, 0);
        env->AddFunction("KMAnalyse", sig, CreateKMAnalyseOCL, 0);  // upstream alias
    } catch (const std::exception& e) {
        return e.what();
    }
    return "KMAnalyse (OpenCL port of KTGMC/MV.cpp KMAnalyse, GPL; ninth host plugin / second motion-stage plugin)";
}
