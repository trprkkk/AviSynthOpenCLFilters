// ktgmc_compensate_avs.cpp — AviSynth+ host plugin for the OpenCL port of
// KTGMC's motion-compensation / degrain consumers (upstream:
// rigaya/AviSynthCUDAFilters KTGMC/MV.cpp + MVKernel.cu, GPL).  Tenth real
// AviSynth+ host plugin of the port, THIRD (and final) motion-stage plugin,
// chained behind the proven pair:  KMSuperOCL(src).KMAnalyseOCL(...).KMCompensateOCL(...)
// and .KMDegrain1OCL(...) / .KMDegrain2OCL(...).
//
// Registers (upstream signatures verbatim, MV.cpp:5236-5246):
//   KMCompensateOCL / KMCompensate  "ccc[thSAD]i[time]f[thSCD1]i[thSCD2]i"
//   KMDegrain1OCL  / KMDegrain1
//       "cccc[thSAD]i[thSADC]i[plane]i[limit]i[limitC]i[thSCD1]i[thSCD2]i[binomial]b[useFlag]i"
//   KMDegrain2OCL  / KMDegrain2
//       "cccccc[thSAD]i[thSADC]i[plane]i[limit]i[limitC]i[thSCD1]i[thSCD2]i[binomial]b[useFlag]i"
//
// ============================================================ LAUNCH SEQUENCE
// Per output frame n (KMCompensate), per frame n with per-delta refs
// (KMDegrain1/2):
//   1. MV-clip read:  each MV clip frame at index n (upstream reads the MV
//      clip at n, NOT at a shifted index — MV.cpp:5162/4569/4591), parse the
//      BGR32 VECTOR frame COARSEST LEVEL FIRST (KMVClip::SetData,
//      MV.cpp:3835-3843) and the KMVIsValid frame prop (MV.cpp:2390-2392);
//      compute each clip's reference index (KMVClip::GetRefFrameIndex,
//      MV.cpp:3856-3866:  n + nDeltaFrame*(isBackward?1:-1), or the constant
//      -nDeltaFrame when nDeltaFrame <= 0).  isUsable[j] = isValid && ref
//      index within the super clip.  Only level-0 VECTORs are consumed
//      downstream (GetVectors(0), MV.cpp:4425/5066).
//   2. Stage the level-0 VECTORs as int3-strided SVM buffers, run the
//      ALG-VERIFIED kt_load_mv_batch (vectors/sads split + out copy-through),
//      then repack host-side into the packed int triples the prepare kernels
//      consume (BLOCKSEARCH_MODEL §8a packed-VECTOR contract; deviation V7).
//   3. Scene change, in kernels (upstream MVKernel.cu:3022-3048/3141-3149):
//      kt_init_scene_change zeroes the flags; degrain runs kt_scene_change_x2
//      per delta over the B/F level-0 arrays, compensate runs kt_scene_change,
//      with nTh1 = the SCALED thSCD1 (KMVClip ctor MV.cpp:3811-3825:
//      min(nSCD1,2040), 16-bit rescale, *BlkSizeX*BlkSizeY/64,
//      + ScaleSadChroma(2*nSCD1, chromaSADScale)/4 when chroma; and
//      nSCD2 = thSCD2*nBlkCount/256, MV.cpp:3824-3825).  The nTh2 GATE lives
//      inside the prepare kernels:  usable = isUsable && !(sceneChange>nTh2).
//   4. Stage the source frame (child, original resolution) and, per usable
//      ref, the super clip frame at the ref index (tight-pitch SVM planes;
//      only level 0 of the pyramid is addressed).
//   5. Per plane p (Y, then U/V; SHIFT = 0/1, NPEL = nPel for BOTH planes,
//      nPad and block size scaled by >>SHIFT, thSAD for Y / thSADC for U/V):
//      degrain:  kt_prepare_degrain (grid (nBlkX,nBlkY)) — the graduated
//        ALG-VERIFIED kernel emitting per-block win_slot/src_base/WSrc/
//        WRefB/WRefF/refBaseB/refBaseF with the §6.3 encodings (unusable ref
//        = offset 0 + weight 0; the nTh2 scene gate) — then a host refBase
//        slice fixup (V4), kt_degrain_patch (grid (BS,BS,nBlk)) and
//        kt_overlap_out (grid (width,height); full plane, edge source-copy).
//      compensate:  kt_prepare_compensate (motion ref when sad < thSAD —
//        time-scaled as (vec*time256/256)>>SHIFT with the truncate-then-floor
//        double rounding; else static ref0 at zero motion; -1 sentinels on
//        scene change), then the patch pass + kt_overlap_out (V5).
//   6. Copy the staged output planes back into the AVS frame rows; planes
//      disabled by the `plane` bitmask are copied from the source frame (V8).
//
// ================================================================ STATE MODEL
// No cross-frame device state:  output frame n is a pure function of
// child(n), the super clip frames (n and the ref indices), the MV clip frames
// and the config (the KNoiseClipOCL/KMAnalyseOCL D9 precedent).  All device
// dispatch is serialized by a process-wide mutex (one context/queue per
// process, sharedContext()).
//
// =================================================== BLOCK-GEOMETRY (SPEC §6.1)
// The prepare/patch/overlap kernels use the CUDA basis blkStep = nBlkSize/2,
// origin nPad + blk*blkStep in the super plane (hard-coded inside
// kt_prepare_degrain / kt_prepare_compensate; CUDA-twin-verified, rig
// cuda_mv_prepare_test.py, 8526 comparisons).  The MV.cpp CPU staging uses
// StepX = nBlkSizeX − nOverlapX (KMVFrame::GetBlock, MV.cpp:3781) — the two
// coincide BECAUSE upstream KMAnalyse enforces blksizex == overlapx*2
// (MV.cpp:2449-2505, reproduced by KMAnalyseOCL), so StepX = nBlkSize/2
// exactly (and overlap <= half is also enforced there).  The host feeds the
// kernels stepX = stepY = (nBlkSize − nOverlap)>>SHIFT (== nBlkSize/2 >>SHIFT)
// and overlapX/Y = nOverlap>>SHIFT, consistent with §6.1 on both bases.
//
// Super-plane addressing:  the level-0 super plane stacks nPel² sub-pel
// planes nPitchSuper*(H+2*vpad) apart (KMPlane::SetTarget, §6.2); a block's
// reference base is offsetS = (nPad>>SHIFT + bx*step) + (...)*nPitchSuper
// plus the ALG-VERIFIED dev_get_ref_block mapping (kd_ref_block_offset of
// mv>>SHIFT with NPEL=nPel).  UV planes use the same NPEL with SHIFT=1
// (upstream launch_prepare_compensate<NPEL, 1>, MVKernel.cu:3167-3175).
//
// ================================================================= DEVIATIONS
//  V1. limit/limitC are parsed (upstream defaults:  bit-adaptive (1<<bpp)-1)
//      but NOT applied:  upstream's shipped CUDA DegrainN path has no
//      LimitChanges call (MVKernel.cu:2820-3006 — the CPU KMDegrainCore::Proc
//      MV.cpp:4230-4232 does apply it).  The port follows the shipped CUDA
//      path; this is an upstream CPU/CUDA divergence, not a port bug.
//  V2. binomial follows the CUDA dev_norm_weights<N,BINOMIAL> form
//      (MVKernel.cu:1815-1881, the ALG-VERIFIED kd_norm_weights twin);
//      upstream's CPU norm_weights (MV.cpp:3551) has no binomial branch.  The
//      port implements the CUDA path (the in-exe verifier reference does too).
//  V3. delta is pinned to the registered 1/2 (upstream KMDegrain1/2; the
//      DegrainN template supports up to 6 but GetDegrainFunction
//      MV.cpp:3983-3993 instantiates 1/2 only).
//  V4. Reference staging:  the per-delta super frames are staged as SLICES of
//      one flat buffer per (plane, direction); kt_prepare_degrain computes
//      slice-0-relative refBase entries, and the host adds i*sliceElems for
//      each usable ref i (a plain SVM fixup).  Unusable refs keep offset 0
//      with weight 0 — never read with a nonzero weight (§6.3 encoding).
//  V5. Compensate patches are produced with kt_degrain_patch as an exact
//      copy:  delta=1, WSrc=0, WRefF=time256 gives (time256*v+128)>>8 for 8-bit
//      and (time256*v)>>8 for 16-bit, i.e. a pure time-blend.  No new kernel
//      is introduced for the compensate fetch; the verified combiner +
//      overlap path is reused.
//  V6. Compensate:  an invalid MV frame (KMVIsValid=0) or out-of-range ref
//      index returns the child frame untouched — the upstream CPU behavior
//      (MV.cpp:5176-5178; upstream's CUDA path would proceed with null
//      planes there, so the CPU behavior is the defined one).  A scene change
//      (count > nTh2) is handled INSIDE the kernel model:  prepare emits the
//      -1 sentinels and the output plane is the source plane — identical to
//      upstream CPU `return child->GetFrame(n)` and to the CUDA
//      isCopySrc/kl_short_to_byte_or_copy_src path (MVKernel.cu:2224-2233/
//      2315-2332), which also copies the uncovered region from src.
//  V7. kt_scene_change(_x2) consume int3-STRIDED MV buffers (OpenCL int3
//      stride 16 — NOT the packed 12-byte VECTOR layout; BLOCKSEARCH_MODEL
//      §8a); the prepare kernels consume packed int triples.  The host keeps
//      both stagings and bridges them with kt_load_mv_batch + repack.
//  V8. U/V are processed iff the child clip has UV planes AND the plane
//      bitmask selects them; disabled planes are copied from the source.
//  V9. thSAD normalization is verbatim (MV.cpp:4536-4537 / 5123):
//      thSAD = thSAD_arg * scaledThSCD1 / thSCD1_arg (int64 intermediate;
//      no overflow for any in-range upstream argument set).
//
// ============================================================ NOT IMPLEMENTED
//  NI1. KMDegrain3..6 (delta 3-6): upstream's GetDegrainFunction only
//      instantiates delta 1/2; the port follows suit.
//  NI2. batch > 1:  kt_interpolate_prediction dropped the batch dimension
//      (see KMAnalyseOCL deviation D4), so batch>1 MV clips cannot be
//      assembled through the real chain.
//  NI3. 16-bit (PX=ushort) is supported in the kernels but the verifier only
//      exercises 8-bit for lack of a 16-bit CPU reference.

#include <cstring>
#include <algorithm>
#include <cmath>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

#include "avisynth.h"
#include "../ocl_host.h"
#include "ktgmc_motion_cl.h"  // GENERATED from .../ktgmc/kernels/ktgmc_motion.cl
#include "ktgmc_kmparam.h"    // KMVParam smuggling (upstream common/KMV.h)

namespace {

using oclhost::Context;
using oclhost::Program;
using oclhost::SvmBuffer;

Context& sharedContext() {
    static Context ctx;  // one device/queue per process
    return ctx;
}

static int nlog2(int v) { int l = 0; while ((1 << l) < v) l++; return l; }

// upstream VECTOR (common/KMV.h): three packed ints {x, y, sad}
struct VECTOR12 { cl_int x, y, sad; };

// upstream PI (common/CommonFunctions.h:25)
static const double kPI = 3.1415926535897932384626433832795;

// ---------------------------------------------------------------------------
// OverlapWindows  — MV.cpp:3442-3526, verbatim (9 feathered windows, 0..2048)
// ---------------------------------------------------------------------------
static void BuildOverlapWindows(int nx, int ny, int ox, int oy,
                                std::vector<short>& out9) {
    std::vector<float> wx(nx, 1), wxf(nx, 1), wxl(nx, 1),
                      wy(ny, 1), wyf(ny, 1), wyl(ny, 1);
    for (int i = 0; i < ox; i++) {
        wx[i] = float(std::cos(kPI * (i - ox + 0.5f) / (ox * 2)));
        wx[i] = wx[i] * wx[i];
        wxf[i] = 1;
        wxl[i] = wx[i];
    }
    for (int i = ox; i < nx - ox; i++) { wx[i] = 1; wxf[i] = 1; wxl[i] = 1; }
    for (int i = nx - ox; i < nx; i++) {
        wx[i] = float(std::cos(kPI * (i - nx + ox + 0.5f) / (ox * 2)));
        wx[i] = wx[i] * wx[i];
        wxf[i] = wx[i]; wxl[i] = 1;
    }
    for (int i = 0; i < oy; i++) {
        wy[i] = float(std::cos(kPI * (i - oy + 0.5f) / (oy * 2)));
        wy[i] = wy[i] * wy[i];
        wyf[i] = 1; wyl[i] = wy[i];
    }
    for (int i = oy; i < ny - oy; i++) { wy[i] = 1; wyf[i] = 1; wyl[i] = 1; }
    for (int i = ny - oy; i < ny; i++) {
        wy[i] = float(std::cos(kPI * (i - ny + oy + 0.5f) / (oy * 2)));
        wy[i] = wy[i] * wy[i];
        wyf[i] = wy[i]; wyl[i] = 1;
    }
    const int size = nx * ny;
    out9.assign((size_t)size * 9, 0);
    for (int j = 0; j < ny; j++) {
        for (int i = 0; i < nx; i++) {
            out9[0 * size + (size_t)j * nx + i] = (int)(wyf[j] * wxf[i] * 2048 + 0.5f);  // TL
            out9[1 * size + (size_t)j * nx + i] = (int)(wyf[j] * wx[i]  * 2048 + 0.5f);  // TM
            out9[2 * size + (size_t)j * nx + i] = (int)(wyf[j] * wxl[i] * 2048 + 0.5f);  // TR
            out9[3 * size + (size_t)j * nx + i] = (int)(wy[j]  * wxf[i] * 2048 + 0.5f);  // ML
            out9[4 * size + (size_t)j * nx + i] = (int)(wy[j]  * wx[i]  * 2048 + 0.5f);  // MM
            out9[5 * size + (size_t)j * nx + i] = (int)(wy[j]  * wxl[i] * 2048 + 0.5f);  // MR
            out9[6 * size + (size_t)j * nx + i] = (int)(wyl[j] * wxf[i] * 2048 + 0.5f);  // BL
            out9[7 * size + (size_t)j * nx + i] = (int)(wyl[j] * wx[i]  * 2048 + 0.5f);  // BM
            out9[8 * size + (size_t)j * nx + i] = (int)(wyl[j] * wxl[i] * 2048 + 0.5f);  // BR
        }
    }
}

// ---------------------------------------------------------------------------
// shared per-frame helpers
// ---------------------------------------------------------------------------
struct PlaneGeom {   // per-plane launch geometry (Y: shift 0, UV: shift 1)
    int shift = 0;
    int bs = 0;           // block size on this plane (nBlkSize >> shift)
    int step = 0;         // (nBlkSize - overlap) >> shift
    int ovr = 0;          // overlap >> shift
    int pitch = 0;        // SOURCE plane pitch (tight)
    int superPitch = 0;   // super plane pitch (tight)
    int imgPitch = 0;     // superPitch * extH (one sub-pel plane)
    int sliceElems = 0;   // one staged whole super plane (delta slice stride)
    int width = 0, height = 0;
    int nPad = 0;
};

static void stagePlaneRows(SvmBuffer& dst, std::vector<uint8_t>& stage,
                           const uint8_t* sp, int spitch, int w, int h,
                           size_t px) {
    const size_t rowbytes = (size_t)w * px;
    for (int y = 0; y < h; y++)
        memcpy(stage.data() + (size_t)y * rowbytes,
               sp + (size_t)y * spitch, rowbytes);
    dst.write(stage.data(), 0, rowbytes * h);
}

static void copyPlaneToFrame(uint8_t* dp, int dpitch, const uint8_t* sp,
                             int spitch, int w, int h, size_t px) {
    for (int y = 0; y < h; y++)
        memcpy(dp + (size_t)y * dpitch, sp + (size_t)y * spitch,
               (size_t)w * px);
}

static void readPlaneFromStaged(const SvmBuffer& src, uint8_t* dp, int dpitch,
                                int w, int h, size_t px,
                                std::vector<uint8_t>& stage) {
    const size_t rowbytes = (size_t)w * px;
    src.read(stage.data(), 0, rowbytes * h);
    for (int y = 0; y < h; y++)
        memcpy(dp + (size_t)y * dpitch, stage.data() + (size_t)y * rowbytes,
               rowbytes);
}

// stage one delta-slice of level-0 VECTOR triples as int3-strided SVM, run
// the ALG-VERIFIED kt_load_mv_batch (vectors/sads split + out copy-through),
// read the split arrays back and repack into the packed int triples the
// prepare kernels consume (BLOCKSEARCH_MODEL §8a; deviation V7)
static void stageMvSlice(SvmBuffer& mvI3, size_t sliceOff4, SvmBuffer& mvPack,
                         size_t sliceOff3, SvmBuffer& vecSc, SvmBuffer& sadSc,
                         SvmBuffer& outSc, const std::vector<cl_int>& lvl0,
                         int nb, Context& ctx, Program& PM) {
    for (int b = 0; b < nb; b++) {
        cl_int quad[4] = { lvl0[3 * b + 0], lvl0[3 * b + 1],
                           lvl0[3 * b + 2], 0 };
        mvI3.write(quad, sliceOff4 + (size_t)b * sizeof(cl_int) * 4,
                   sizeof(cl_int) * 4);
    }
    {
        cl_kernel k = PM.kernel("kt_load_mv_batch");
        oclhost::set_arg_svm(k, 0, outSc.ptr());
        oclhost::set_arg_svm(k, 1, mvI3.at(sliceOff4));
        oclhost::set_arg_svm(k, 2, vecSc.ptr());
        oclhost::set_arg_svm(k, 3, sadSc.ptr());
        oclhost::set_arg(k, 4, nb);
        oclhost::dispatch1D(ctx, k, nb);
    }
    ctx.finish();
    std::vector<cl_int2> vecs((size_t)nb);
    std::vector<cl_int> sads((size_t)nb);
    vecSc.read(vecs.data(), 0, sizeof(cl_int2) * nb);
    sadSc.read(sads.data(), 0, sizeof(cl_int) * nb);
    for (int b = 0; b < nb; b++) {
        cl_int trip[3] = { vecs[b].x, vecs[b].y, sads[b] };
        mvPack.write(trip, sliceOff3 + (size_t)b * sizeof(cl_int) * 3,
                     sizeof(cl_int) * 3);
    }
}

// stageRefSlice — one delta-slice of a super frame as a tight-pitch plane
static void stageRefSlice(SvmBuffer& buf, int j, const PlaneGeom& g, size_t px,
                          PVideoFrame& f, int plane,
                          std::vector<uint8_t>& stage) {
    const size_t rowbytes = (size_t)g.width * px;
    for (int y = 0; y < g.height; y++)
        memcpy(stage.data() + (size_t)y * rowbytes,
               f->GetReadPtr(plane) + (size_t)y * f->GetPitch(plane),
               rowbytes);
    buf.write(stage.data(), (size_t)j * g.sliceElems * px,
              rowbytes * g.height);
}

// ============================================================================
// MVClip  — upstream KMVClip (MV.cpp:3749-3887), the consumer half.
// Holds a COPY of the clip's MV_FRAME KMVParam, the scaled SCD thresholds,
// and parses per-frame VECTOR data out of the BGR32 clip frames
// (coarsest level first).
// ============================================================================
struct MVClip {
    KMVParam par;
    int nSCD1 = 0, nSCD2 = 0;
    int nBlk0 = 0;          // level-0 block count
    int frameBytes = 0;     // total VECTOR bytes carried by one MV frame
    int nLevels = 0;

    MVClip(const KMVParam& p, int userSCD1, int userSCD2) : par(p) {
        nLevels = par.nAnalyzeLevels;
        // SCD thresholds (MV.cpp:3811-3825, verbatim)
        nSCD1 = std::min(userSCD1, 8 * 8 * (255 - 0));
        if (par.nPixelSize == 2)
            nSCD1 = int(nSCD1 / 255.0 * ((1 << par.nBitsPerPixel) - 1));
        nSCD1 = (int)((uint64_t)nSCD1 * (par.nBlkSizeX * par.nBlkSizeY) / (8 * 8));
        if (par.chroma)
            nSCD1 += ScaleSadChroma(nSCD1 * 2, par.chromaSADScale) / 4;
        const int nBlkCount = par.levelInfo[0].nBlkX * par.levelInfo[0].nBlkY;
        nBlk0 = nBlkCount;
        nSCD2 = userSCD2 * nBlkCount / 256;   // MV.cpp:3824-3825
        frameBytes = 0;
        for (int i = 0; i < nLevels; i++)
            frameBytes += par.levelInfo[i].nBlkX * par.levelInfo[i].nBlkY
                          * (int)sizeof(VECTOR12);
    }

    static int ScaleSadChroma(int sad, int effective_scale) {
        if (effective_scale == 0) return sad;
        if (effective_scale > 0) return sad >> effective_scale;
        return sad << (-effective_scale);
    }

    // KMVClip::GetRefFrameIndex (MV.cpp:3856-3866)
    int refIndex(int n) const {
        int off = par.nDeltaFrame;
        if (off > 0) {
            off *= par.isBackward ? 1 : -1;
            return n + off;
        }
        return -off;
    }

    // Read one clip frame:  returns KMVIsValid and fills lvl0 with the
    // level-0 (finest) packed (x,y,sad) triples — the LAST nBlk0 triples of
    // the coarsest-first layout (MV.cpp:3838-3842).
    bool readFrame(PClip& clip, int n, IScriptEnvironment* env,
                   std::vector<cl_int>& lvl0) const {
        PVideoFrame f = clip->GetFrame(n, env);
        int error = 0;
        const int64_t propVal = env->propGetInt(
            env->getFramePropsRO(f), "KMVIsValid", 0, &error);
        const bool isValid = !error && (propVal != 0);
        std::vector<cl_int> data((size_t)frameBytes / 4);
        {
            const uint8_t* fp = f->GetReadPtr();
            const int pitch = f->GetPitch();
            for (size_t k = 0; k < data.size(); k++) {
                const size_t bytePos = k * 4;
                const uint8_t* q = fp + (bytePos / (size_t)pitch) * pitch
                                       + (bytePos % (size_t)pitch);
                memcpy(&data[k], q, 4);
            }
        }
        lvl0.assign((size_t)nBlk0 * 3, 0);
        size_t pos = 0;
        for (int i = nLevels - 1; i >= 0; i--) {
            const int blks = par.levelInfo[i].nBlkX * par.levelInfo[i].nBlkY;
            if (i == 0)
                memcpy(lvl0.data(), &data[pos], sizeof(cl_int) * 3 * blks);
            pos += (size_t)blks * 3;
        }
        return isValid;
    }
};

// ============================================================================
// KMDegrainOCL  — upstream KMDegrainX (MV.cpp:4236-4712) + KMDegrainCore
// (MV.cpp:3927-4234) + the CUDA Degrain path (MVKernel.cu:3008-3070 +
// DegrainN 2820-3006), delta 1/2.
// ============================================================================
class KMDegrainOCL : public GenericVideoFilter {
    int delta_;
    int YUVplanes_;
    int useFlag_;
    int thSAD_, thSADC_;
    int nTh1_, nTh2_;
    PClip super_;
    PClip mvB_[2], mvF_[2];
    MVClip* clipB_[2];
    MVClip* clipF_[2];

    const KMVParam* par_;       // the mvbw clip's MV_FRAME params (drive the geometry)
    const KMVParam* superPar_;
    int BS_, ovr_;
    int nPel_, nHPad_;
    int px_, logUVx_, logUVy_;
    int pitchY_, pitchUV_, planeHY_, planeHUV_, vpadU_;
    bool hasUV_;
    PlaneGeom gY_, gU_;

    std::vector<short> winY_, winUV_;
    SvmBuffer winYd_, winUVd_;

    // staging
    SvmBuffer srcY_, srcU_, srcV_;
    SvmBuffer refBY_, refBU_, refBV_;   // delta slices each
    SvmBuffer refFY_, refFU_, refFV_;
    SvmBuffer dstY_, dstU_, dstV_;
    std::vector<uint8_t> stage_;

    // MV / decision buffers
    SvmBuffer mvI3B_, mvI3F_;        // delta*nBlk0 int3-strided (input + sc)
    SvmBuffer mvPackB_, mvPackF_;    // delta*nBlk0*3 packed (prepare input)
    SvmBuffer vecSc_, sadSc_, outSc_;    // kt_load_mv_batch scratch
    SvmBuffer scB_, scF_, useB_, useF_;
    SvmBuffer wsA_, sbA_, wsl_, wbA_, wfA_, rbB_, rbF_, patch_;

    std::mutex* mtx_;

    Program& motionProgram() {
        return oclhost::ProgramRegistry::instance().get(
            sharedContext(), "ktgmc_motion", ktgmc_motion_cl_src, px_);
    }

    int nBlkX() const { return par_->levelInfo[0].nBlkX; }
    int nBlkY() const { return par_->levelInfo[0].nBlkY; }
    int nBlk0() const { return nBlkX() * nBlkY(); }

    PlaneGeom planeGeom(int shift, int planeRows) const {
        PlaneGeom g;
        g.shift = shift;
        g.bs = BS_ >> shift;
        g.step = (BS_ - ovr_) >> shift;
        g.ovr = ovr_ >> shift;
        g.pitch = shift ? pitchUV_ : pitchY_;
        g.superPitch = shift ? pitchUV_ : pitchY_;
        const int extH = shift
            ? (superPar_->nHeight / superPar_->yRatioUV + 2 * vpadU_)
            : (superPar_->nHeight + 2 * superPar_->nVPad);
        g.imgPitch = g.superPitch * extH;
        g.sliceElems = g.superPitch * planeRows;
        g.width = par_->nWidth >> shift;
        g.height = par_->nHeight >> shift;
        g.nPad = nHPad_ >> shift;
        return g;
    }

    // stage lvl0 triples as int3-strided slices + kt_load_mv_batch per
    // delta and repack into the packed triples (V7)
    void stageMv(SvmBuffer& mvI3, SvmBuffer& mvPack,
                 std::vector<cl_int> lvl0[2], Context& ctx, Program& PM) {
        const int nb = nBlk0();
        for (int j = 0; j < delta_; j++) {
            if (lvl0[j].size() < (size_t)nb * 3)
                lvl0[j].assign((size_t)nb * 3, 0);  // skipped direction (useFlag)
            for (int b = 0; b < nb; b++) {
                cl_int quad[4] = { lvl0[j][3 * b + 0], lvl0[j][3 * b + 1],
                                   lvl0[j][3 * b + 2], 0 };
                mvI3.write(quad, ((size_t)j * nb + b) * sizeof(cl_int) * 4,
                           sizeof(cl_int) * 4);
            }
        }
        for (int j = 0; j < delta_; j++) {
            cl_kernel k = PM.kernel("kt_load_mv_batch");
            oclhost::set_arg_svm(k, 0, outSc_.ptr());
            oclhost::set_arg_svm(k, 1, mvI3.at((size_t)j * nb * 4 * sizeof(cl_int)));
            oclhost::set_arg_svm(k, 2, vecSc_.ptr());
            oclhost::set_arg_svm(k, 3, sadSc_.ptr());
            oclhost::set_arg(k, 4, nb);
            oclhost::dispatch1D(ctx, k, nb);
        }
        ctx.finish();
        std::vector<cl_int2> vecs((size_t)nb);
        std::vector<cl_int> sads((size_t)nb);
        vecSc_.read(vecs.data(), 0, sizeof(cl_int2) * nb);
        sadSc_.read(sads.data(), 0, sizeof(cl_int) * nb);
        for (int j = 0; j < delta_; j++)
            for (int b = 0; b < nb; b++) {
                cl_int trip[3] = { vecs[b].x, vecs[b].y, sads[b] };
                mvPack.write(trip, ((size_t)j * nb + b) * sizeof(cl_int) * 3,
                             sizeof(cl_int) * 3);
            }
    }

    // one plane's full degrain path
    void runPlane(const PlaneGeom& g, int thsadPlane, SvmBuffer& src,
                  SvmBuffer& dst, SvmBuffer& refB, SvmBuffer& refF,
                  SvmBuffer& winBuf, bool* refUsableB, bool* refUsableF,
                  Context& ctx, Program& PM) {
        const int nb = nBlk0();
        {   // ---- kt_prepare_degrain (26 args, .cl order) ----
            cl_kernel k = PM.kernel("kt_prepare_degrain");
            oclhost::set_arg(k, 0, nBlkX());
            oclhost::set_arg(k, 1, nBlkY());
            oclhost::set_arg(k, 2, g.nPad);
            oclhost::set_arg(k, 3, g.bs);
            oclhost::set_arg(k, 4, nTh2_);
            oclhost::set_arg(k, 5, thsadPlane);
            oclhost::set_arg(k, 6, delta_);
            oclhost::set_arg(k, 7, binomial_ ? 1 : 0);
            oclhost::set_arg(k, 8, nPel_);
            oclhost::set_arg(k, 9, g.shift);
            oclhost::set_arg(k, 10, g.pitch);
            oclhost::set_arg(k, 11, g.superPitch);
            oclhost::set_arg(k, 12, g.imgPitch);
            oclhost::set_arg_svm(k, 13, scB_.ptr());
            oclhost::set_arg_svm(k, 14, scF_.ptr());
            oclhost::set_arg_svm(k, 15, useB_.ptr());
            oclhost::set_arg_svm(k, 16, useF_.ptr());
            oclhost::set_arg_svm(k, 17, mvPackB_.ptr());
            oclhost::set_arg_svm(k, 18, mvPackF_.ptr());
            oclhost::set_arg_svm(k, 19, wsl_.ptr());
            oclhost::set_arg_svm(k, 20, sbA_.ptr());
            oclhost::set_arg_svm(k, 21, wsA_.ptr());
            oclhost::set_arg_svm(k, 22, wbA_.ptr());
            oclhost::set_arg_svm(k, 23, wfA_.ptr());
            oclhost::set_arg_svm(k, 24, rbB_.ptr());
            oclhost::set_arg_svm(k, 25, rbF_.ptr());
            oclhost::dispatch2D(ctx, k, nBlkX(), nBlkY());
        }
        ctx.finish();

        // ---- V4 refBase slice fixup (usable refs only) ----
        {
            std::vector<cl_int> rbB((size_t)nb * delta_), rbF((size_t)nb * delta_);
            rbB_.read(rbB.data(), 0, sizeof(cl_int) * nb * delta_);
            rbF_.read(rbF.data(), 0, sizeof(cl_int) * nb * delta_);
            for (int j = 0; j < delta_; j++) {
                if (!refUsableB[j]) continue;
                const size_t base = (size_t)j * nb;
                for (int b = 0; b < nb; b++)
                    rbB[base + b] += j * g.sliceElems;
            }
            for (int j = 0; j < delta_; j++) {
                if (!refUsableF[j]) continue;
                const size_t base = (size_t)j * nb;
                for (int b = 0; b < nb; b++)
                    rbF[base + b] += j * g.sliceElems;
            }
            rbB_.write(rbB.data(), 0, sizeof(cl_int) * nb * delta_);
            rbF_.write(rbF.data(), 0, sizeof(cl_int) * nb * delta_);
        }

        patchOverlap(g, src, dst, refB, refF, winBuf, wsA_, wfA_, wbA_,
                     rbF_, rbB_, ctx, PM);
    }

    // shared combiner + overlap tail:  kt_degrain_patch (18 args) over the
    // prepared weight/base arrays, then kt_overlap_out (17 args; full plane,
    // edge source-copy).  Used by BOTH KMDegrainOCL (prepare-degrain arrays)
    // and KMCompensateOCL (V5: prebuilt WSrc=0 / WB=0 / WF=256 arrays with
    // refBaseF = the compensated block bases, refBaseB = 0).
    void patchOverlap(const PlaneGeom& g, SvmBuffer& src, SvmBuffer& dst,
                      SvmBuffer& refB, SvmBuffer& refF, SvmBuffer& winBuf,
                      SvmBuffer& wsrcArr, SvmBuffer& wfArr, SvmBuffer& wbArr,
                      SvmBuffer& rbF, SvmBuffer& rbB,
                      Context& ctx, Program& PM) {
        const int nb = nBlk0();
        {   // ---- kt_degrain_patch (18 args) ----
            cl_kernel k = PM.kernel("kt_degrain_patch");
            oclhost::set_arg_svm(k, 0, src.ptr());
            oclhost::set_arg(k, 1, g.pitch);
            oclhost::set_arg(k, 2, nBlkX());
            oclhost::set_arg(k, 3, nBlkY());
            oclhost::set_arg(k, 4, g.bs);
            oclhost::set_arg(k, 5, g.step);
            oclhost::set_arg(k, 6, g.step);
            oclhost::set_arg(k, 7, 1);   // delta
            oclhost::set_arg_svm(k, 8, wsrcArr.ptr());
            oclhost::set_arg_svm(k, 9, wfArr.ptr());
            oclhost::set_arg_svm(k, 10, wbArr.ptr());
            oclhost::set_arg_svm(k, 11, rbF.ptr());
            oclhost::set_arg_svm(k, 12, rbB.ptr());
            oclhost::set_arg_svm(k, 13, refF.ptr());
            oclhost::set_arg(k, 14, g.superPitch);
            oclhost::set_arg_svm(k, 15, refB.ptr());
            oclhost::set_arg(k, 16, g.superPitch);
            oclhost::set_arg_svm(k, 17, patch_.ptr());
            oclhost::dispatch3D(ctx, k, g.bs, g.bs, nb);
        }

        {   // ---- kt_overlap_out (17 args; full plane, edge src-copy) ----
            cl_kernel k = PM.kernel("kt_overlap_out");
            oclhost::set_arg_svm(k, 0, src.ptr());
            oclhost::set_arg(k, 1, g.pitch);
            oclhost::set_arg_svm(k, 2, patch_.ptr());
            oclhost::set_arg(k, 3, g.bs * g.bs);
            oclhost::set_arg(k, 4, nBlkX());
            oclhost::set_arg(k, 5, nBlkY());
            oclhost::set_arg(k, 6, g.bs);
            oclhost::set_arg(k, 7, g.step);
            oclhost::set_arg(k, 8, g.step);
            oclhost::set_arg(k, 9, g.ovr);
            oclhost::set_arg(k, 10, g.ovr);
            oclhost::set_arg_svm(k, 11, winBuf.ptr());
            oclhost::set_arg(k, 12, g.bs * g.bs);
            oclhost::set_arg(k, 13, g.width);
            oclhost::set_arg(k, 14, g.height);
            oclhost::set_arg_svm(k, 15, dst.ptr());
            oclhost::set_arg(k, 16, g.pitch);
            oclhost::dispatch2D(ctx, k, g.width, g.height);
        }
        ctx.finish();
    }

public:
    KMDegrainOCL(PClip child, PClip superClip,
                 PClip mvbw, PClip mvfw, PClip mvbw2, PClip mvfw2,
                 int thSADArg, int thSADCArg, int YUVplanes,
                 int nLimit, int nLimitC, int nSCD1Arg, int nSCD2Arg,
                 bool binomial, int useFlag, int delta, IScriptEnvironment* env_)
        : GenericVideoFilter(child)
        , delta_(delta)
        , YUVplanes_(YUVplanes)
        , useFlag_(useFlag)
        , clipB_{ nullptr, nullptr }, clipF_{ nullptr, nullptr } {
        (void)nLimit; (void)nLimitC;   // V1: parsed, not applied (CUDA parity)
        if (delta != 1 && delta != 2)
            env_->ThrowError("KMDegrain: only delta 1/2 is supported");
        mvB_[0] = mvbw; mvF_[0] = mvfw;
        mvB_[1] = mvbw2; mvF_[1] = mvfw2;

        const KMVParam* p0 = KMVParam::GetParam(mvbw->GetVideoInfo(), env_);
        if (p0->nDataType != KMVParam::MV_FRAME)
            env_->ThrowError("KMDegrain%d: mvbw is not a KMAnalyse(OCL) MV clip", delta);
        par_ = p0;
        for (int j = 0; j < delta_; j++) {
            const KMVParam* pb = KMVParam::GetParam(mvB_[j]->GetVideoInfo(), env_);
            const KMVParam* pf = KMVParam::GetParam(mvF_[j]->GetVideoInfo(), env_);
            if (pb->nDataType != KMVParam::MV_FRAME ||
                pf->nDataType != KMVParam::MV_FRAME)
                env_->ThrowError("KMDegrain%d: MV clips must be KMAnalyse(OCL) clips", delta);
            clipB_[j] = new MVClip(*pb, nSCD1Arg, nSCD2Arg);
            clipF_[j] = new MVClip(*pf, nSCD1Arg, nSCD2Arg);
        }
        // thSAD normalization (MV.cpp:4536-4537, verbatim; V9)
        thSAD_ = (int)((uint64_t)thSADArg * clipB_[0]->nSCD1 / nSCD1Arg);
        thSADC_ = (int)((uint64_t)thSADCArg * clipB_[0]->nSCD1 / nSCD1Arg);
        nTh1_ = clipB_[0]->nSCD1;      // upstream Degrain uses clipB[0]'s
        nTh2_ = clipB_[0]->nSCD2;
        binomial_ = binomial;

        if (par_->nPixelSize != 1 && par_->nPixelSize != 2)
            env_->ThrowError("KMDegrain%d: unsupported pixel size", delta_);
        if (vi.ComponentSize() != par_->nPixelSize)
            env_->ThrowError("KMDegrain%d: source pixel size does not match the MV clip", delta_);
        if (vi.width != par_->nWidth || vi.height != par_->nHeight)
            env_->ThrowError("KMDegrain%d: source dims %dx%d do not match the MV clip's %dx%d",
                             delta_, vi.width, vi.height, par_->nWidth, par_->nHeight);
        if (nBlkX() < 3 || nBlkY() < 3)
            env_->ThrowError("KMDegrain%d: the 9-window slot divides by "
                             "(nBlkX-2)/(nBlkY-2); grids with fewer than 3 "
                             "blocks per axis are not valid (upstream divides "
                             "by zero there too)", delta_);
        try {
            const KMVParam* sp = KMVParam::GetParam(superClip->GetVideoInfo(), env_);
            superPar_ = sp;
        } catch (const AvisynthError&) {
            env_->ThrowError("KMDegrain%d: the super clip must be a KMSuper(OCL) clip", delta_);
        }
        super_ = superClip;

        BS_ = par_->nBlkSizeX; ovr_ = par_->nOverlapX;
        nPel_ = par_->nPel;
        nHPad_ = par_->nHPad;
        px_ = par_->nPixelSize;
        logUVx_ = nlog2(par_->xRatioUV);
        logUVy_ = nlog2(par_->yRatioUV);
        pitchY_ = super_->GetVideoInfo().width;
        planeHY_ = super_->GetVideoInfo().height;
        pitchUV_ = pitchY_ >> logUVx_;
        planeHUV_ = planeHY_ >> logUVy_;
        vpadU_ = superPar_->nVPad / superPar_->yRatioUV;
        hasUV_ = (vi.IsYUV() || vi.IsYUVA()) &&
                 super_->GetVideoInfo().IsPlanar() &&
                 super_->GetVideoInfo().IsYUV();
        if (hasUV_ && superPar_->nWidth >> logUVx_ == 0) hasUV_ = false;

        gY_ = planeGeom(0, planeHY_);
        gU_ = planeGeom(1, planeHUV_);

        BuildOverlapWindows(BS_, BS_, ovr_, ovr_, winY_);
        if (hasUV_)
            BuildOverlapWindows(BS_ >> 1, BS_ >> 1, ovr_ >> 1, ovr_ >> 1, winUV_);

        try {
            Context& ctx = sharedContext();
            const size_t pxs = (size_t)px_;
            const int nb = nBlk0();
            const size_t bytesY = (size_t)pitchY_ * planeHY_ * pxs;
            const size_t bytesUV = (size_t)pitchUV_ * planeHUV_ * pxs;
            srcY_ = SvmBuffer(ctx, bytesY);
            dstY_ = SvmBuffer(ctx, bytesY);
            refBY_ = SvmBuffer(ctx, bytesY * delta_);
            refFY_ = SvmBuffer(ctx, bytesY * delta_);
            if (hasUV_) {
                srcU_ = SvmBuffer(ctx, bytesUV);
                srcV_ = SvmBuffer(ctx, bytesUV);
                dstU_ = SvmBuffer(ctx, bytesUV);
                dstV_ = SvmBuffer(ctx, bytesUV);
                refBU_ = SvmBuffer(ctx, bytesUV * delta_);
                refBV_ = SvmBuffer(ctx, bytesUV * delta_);
                refFU_ = SvmBuffer(ctx, bytesUV * delta_);
                refFV_ = SvmBuffer(ctx, bytesUV * delta_);
            }
            stage_.resize(std::max(bytesY, bytesUV));
            mvI3B_ = SvmBuffer(ctx, sizeof(cl_int) * 4 * nb * delta_);
            mvI3F_ = SvmBuffer(ctx, sizeof(cl_int) * 4 * nb * delta_);
            mvPackB_ = SvmBuffer(ctx, sizeof(cl_int) * 3 * nb * delta_);
            mvPackF_ = SvmBuffer(ctx, sizeof(cl_int) * 3 * nb * delta_);
            outSc_ = SvmBuffer(ctx, sizeof(cl_int) * 4 * nb);
            vecSc_ = SvmBuffer(ctx, sizeof(cl_int2) * nb);
            sadSc_ = SvmBuffer(ctx, sizeof(cl_int) * nb);
            scB_ = SvmBuffer(ctx, sizeof(cl_int) * delta_);
            scF_ = SvmBuffer(ctx, sizeof(cl_int) * delta_);
            useB_ = SvmBuffer(ctx, sizeof(cl_int) * delta_);
            useF_ = SvmBuffer(ctx, sizeof(cl_int) * delta_);
            wsA_ = SvmBuffer(ctx, sizeof(cl_int) * nb);
            sbA_ = SvmBuffer(ctx, sizeof(cl_int) * nb);
            wsl_ = SvmBuffer(ctx, sizeof(cl_int) * nb);
            wbA_ = SvmBuffer(ctx, sizeof(cl_int) * nb * delta_);
            wfA_ = SvmBuffer(ctx, sizeof(cl_int) * nb * delta_);
            rbB_ = SvmBuffer(ctx, sizeof(cl_int) * nb * delta_);
            rbF_ = SvmBuffer(ctx, sizeof(cl_int) * nb * delta_);
            patch_ = SvmBuffer(ctx, pxs * (size_t)BS_ * BS_ * nb);
            winYd_ = SvmBuffer(ctx, winY_.size() * sizeof(short));
            winYd_.write(winY_.data(), 0, winY_.size() * sizeof(short));
            if (hasUV_) {
                winUVd_ = SvmBuffer(ctx, winUV_.size() * sizeof(short));
                winUVd_.write(winUV_.data(), 0, winUV_.size() * sizeof(short));
            }
            motionProgram();
        } catch (const std::exception& e) {
            env_->ThrowError("[KMDegrainOCL] OpenCL init failed: %s", e.what());
        }
        static std::mutex mtx;
        mtx_ = &mtx;
    }

    PVideoFrame __stdcall GetFrame(int n, IScriptEnvironment* env_) {
        try {
            return GetFrameT(n, env_);
        } catch (const std::exception& e) {
            env_->ThrowError("[KMDegrainOCL] %s", e.what());
        }
        return PVideoFrame();
    }

    int __stdcall SetCacheHints(int cachehints, int frame_range) {
        if (cachehints == CACHE_GET_MTMODE) return MT_MULTI_INSTANCE;
        return 0;
    }

private:
    bool binomial_ = false;

    PVideoFrame GetFrameT(int n, IScriptEnvironment* env_) {
        std::lock_guard<std::mutex> lock(*mtx_);
        Context& ctx = sharedContext();
        Program& PM = motionProgram();
        const int nb = nBlk0();
        const int nf = super_->GetVideoInfo().num_frames;

        // ---- 1. MV frames + usability (upstream GetFrame, MV.cpp:4560-4605;
        //         useFlag gates which direction is read, MV.cpp:4567/4589)
        bool isUsableB[2] = { false, false };
        bool isUsableF[2] = { false, false };
        int refIdxB[2] = { 0, 0 }, refIdxF[2] = { 0, 0 };
        std::vector<cl_int> lvl0B[2], lvl0F[2];
        if (useFlag_ != 2 /* USE_ONLY_AFTER: F skipped, MV.cpp:4567 */) {
            for (int j = delta_ - 1; j >= 0; j--) {
                const bool v = clipF_[j]->readFrame(mvF_[j], n, env_, lvl0F[j]);
                refIdxF[j] = clipF_[j]->refIndex(n);
                isUsableF[j] = v && refIdxF[j] >= 0 && refIdxF[j] < nf;
            }
        }
        if (useFlag_ != 1 /* USE_ONLY_BEFORE: B skipped, MV.cpp:4589 */) {
            for (int j = 0; j < delta_; j++) {
                const bool v = clipB_[j]->readFrame(mvB_[j], n, env_, lvl0B[j]);
                refIdxB[j] = clipB_[j]->refIndex(n);
                isUsableB[j] = v && refIdxB[j] >= 0 && refIdxB[j] < nf;
            }
        }

        // ---- 2/3. MV staging + scene-change kernels
        for (int j = 0; j < delta_; j++) {
            cl_int uB = isUsableB[j] ? 1 : 0;
            cl_int uF = isUsableF[j] ? 1 : 0;
            useB_.write(&uB, sizeof(cl_int) * j, sizeof(cl_int));
            useF_.write(&uF, sizeof(cl_int) * j, sizeof(cl_int));
        }
        stageMv(mvI3B_, mvPackB_, lvl0B, ctx, PM);
        stageMv(mvI3F_, mvPackF_, lvl0F, ctx, PM);
        {
            cl_kernel ki = PM.kernel("kt_init_scene_change");
            oclhost::set_arg_svm(ki, 0, scB_.ptr());
            oclhost::dispatch1D(ctx, ki, delta_);
            oclhost::set_arg_svm(ki, 0, scF_.ptr());
            oclhost::dispatch1D(ctx, ki, delta_);
            for (int j = 0; j < delta_; j++) {
                // upstream runs the count only for usable refs
                // (MVKernel.cu:3028-3047); non-usable flags stay zeroed and
                // are gated by isUsable anyway.
                if (!isUsableB[j] && !isUsableF[j]) continue;
                cl_kernel k = PM.kernel("kt_scene_change_x2");
                oclhost::set_arg_svm(k, 0, mvI3B_.at((size_t)j * nb * 4 * sizeof(cl_int)));
                oclhost::set_arg_svm(k, 1, mvI3F_.at((size_t)j * nb * 4 * sizeof(cl_int)));
                oclhost::set_arg(k, 2, nb);
                oclhost::set_arg(k, 3, nTh1_);
                oclhost::set_arg_svm(k, 4, scB_.at((size_t)j * sizeof(cl_int)));
                oclhost::set_arg_svm(k, 5, scF_.at((size_t)j * sizeof(cl_int)));
                oclhost::dispatch1D(ctx, k, nb);
            }
        }

        // ---- 4. stage source + usable super refs
        PVideoFrame src = child->GetFrame(n, env_);
        stagePlaneRows(srcY_, stage_, src->GetReadPtr(PLANAR_Y),
                       src->GetPitch(PLANAR_Y), vi.width, vi.height, px_);
        if (hasUV_) {
            stagePlaneRows(srcU_, stage_, src->GetReadPtr(PLANAR_U),
                           src->GetPitch(PLANAR_U), vi.width >> logUVx_,
                           vi.height >> logUVy_, px_);
            stagePlaneRows(srcV_, stage_, src->GetReadPtr(PLANAR_V),
                           src->GetPitch(PLANAR_V), vi.width >> logUVx_,
                           vi.height >> logUVy_, px_);
        }
        for (int j = 0; j < delta_; j++) {
            if (isUsableB[j]) {
                PVideoFrame r = super_->GetFrame(refIdxB[j], env_);
                stageRefSlice(refBY_, j, gY_, (size_t)px_, r, PLANAR_Y, stage_);
                if (hasUV_) {
                    stageRefSlice(refBU_, j, gU_, (size_t)px_, r, PLANAR_U, stage_);
                    stageRefSlice(refBV_, j, gU_, (size_t)px_, r, PLANAR_V, stage_);
                }
            }
            if (isUsableF[j]) {
                PVideoFrame r = super_->GetFrame(refIdxF[j], env_);
                stageRefSlice(refFY_, j, gY_, (size_t)px_, r, PLANAR_Y, stage_);
                if (hasUV_) {
                    stageRefSlice(refFU_, j, gU_, (size_t)px_, r, PLANAR_U, stage_);
                    stageRefSlice(refFV_, j, gU_, (size_t)px_, r, PLANAR_V, stage_);
                }
            }
        }
        ctx.finish();

        // ---- 5. per-plane prepare + patch + overlap
        const bool enY = (YUVplanes_ & 1) != 0;
        const bool enU = (YUVplanes_ & 2) != 0;
        const bool enV = (YUVplanes_ & 4) != 0;
        bool refUsableB[2], refUsableF[2];
        if (enY) {
            // the kernel re-derives the gate from scB_/scF_/useB_/useF_
            for (int j = 0; j < delta_; j++) {
                cl_int scb = 0, scf = 0;
                scB_.read(&scb, sizeof(cl_int) * j, sizeof(cl_int));
                scF_.read(&scf, sizeof(cl_int) * j, sizeof(cl_int));
                refUsableB[j] = isUsableB[j] && !(scb > nTh2_);
                refUsableF[j] = isUsableF[j] && !(scf > nTh2_);
            }
            runPlane(gY_, thSAD_, srcY_, dstY_, refBY_, refFY_, winYd_,
                     refUsableB, refUsableF, ctx, PM);
        }
        if (hasUV_ && (enU || enV)) {
            for (int j = 0; j < delta_; j++) {
                cl_int scb = 0, scf = 0;
                scB_.read(&scb, sizeof(cl_int) * j, sizeof(cl_int));
                scF_.read(&scf, sizeof(cl_int) * j, sizeof(cl_int));
                refUsableB[j] = isUsableB[j] && !(scb > nTh2_);
                refUsableF[j] = isUsableF[j] && !(scf > nTh2_);
            }
            if (enU)
                runPlane(gU_, thSADC_, srcU_, dstU_, refBU_, refFU_, winUVd_,
                         refUsableB, refUsableF, ctx, PM);
            if (enV)
                runPlane(gU_, thSADC_, srcV_, dstV_, refBV_, refFV_, winUVd_,
                         refUsableB, refUsableF, ctx, PM);
        }

        // ---- 6. copy-back
        PVideoFrame dst = env_->NewVideoFrame(vi);
        const int wUV = vi.width >> logUVx_, hUV = vi.height >> logUVy_;
        if (enY)
            readPlaneFromStaged(dstY_, dst->GetWritePtr(PLANAR_Y),
                                dst->GetPitch(PLANAR_Y), vi.width, vi.height,
                                px_, stage_);
        else
            copyPlaneToFrame(dst->GetWritePtr(PLANAR_Y), dst->GetPitch(PLANAR_Y),
                             src->GetReadPtr(PLANAR_Y), src->GetPitch(PLANAR_Y),
                             vi.width, vi.height, px_);
        if (hasUV_) {
            const bool cpU = !enU, cpV = !enV;
            if (cpU)
                copyPlaneToFrame(dst->GetWritePtr(PLANAR_U), dst->GetPitch(PLANAR_U),
                                 src->GetReadPtr(PLANAR_U), src->GetPitch(PLANAR_U),
                                 wUV, hUV, px_);
            else
                readPlaneFromStaged(dstU_, dst->GetWritePtr(PLANAR_U),
                                    dst->GetPitch(PLANAR_U), wUV, hUV, px_, stage_);
            if (cpV)
                copyPlaneToFrame(dst->GetWritePtr(PLANAR_V), dst->GetPitch(PLANAR_V),
                                 src->GetReadPtr(PLANAR_V), src->GetPitch(PLANAR_V),
                                 wUV, hUV, px_);
            else
                readPlaneFromStaged(dstV_, dst->GetWritePtr(PLANAR_V),
                                    dst->GetPitch(PLANAR_V), wUV, hUV, px_, stage_);
        }
        return dst;
    }
};

// ============================================================================
// KMCompensateOCL  — upstream KMCompensate (MV.cpp:4917-5204)
// delta=1 fixed, single MV clip, time-based blend (thSAD only, no thSADC).
// V5: kt_degrain_patch with WSrc=0/WB=0/WF=time256 reuses the verified combiner.
// ============================================================================
class KMCompensateOCL : public GenericVideoFilter {
    int YUVplanes_;
    int thSAD_, nTh1_, nTh2_;
    int time256_;
    PClip super_;
    PClip mvClip_;
    MVClip* clip_;
    const KMVParam* par_;
    const KMVParam* superPar_;
    int BS_, ovr_;
    int nPel_, nHPad_;
    int px_, logUVx_, logUVy_;
    int pitchY_, pitchUV_, planeHY_, planeHUV_, vpadU_;
    bool hasUV_;
    PlaneGeom gY_, gU_;

    std::vector<short> winY_, winUV_;
    SvmBuffer winYd_, winUVd_;

    SvmBuffer srcY_, srcU_, srcV_;
    SvmBuffer dstY_, dstU_, dstV_;
    SvmBuffer refBY_, refFY_, refBU_, refBV_, refFU_, refFV_;
    std::vector<uint8_t> stage_;

    SvmBuffer mvI3_, mvPack_;
    SvmBuffer vecSc_, sadSc_, outSc_;
    SvmBuffer sc_, use_;
    SvmBuffer wsl_, wsA_, wbA_, wfA_, rbB_, rbF_, patch_;

    std::mutex* mtx_;

    Program& motionProgram() {
        return oclhost::ProgramRegistry::instance().get(
            sharedContext(), "ktgmc_motion", ktgmc_motion_cl_src, px_);
    }

    int nBlkX() const { return par_->levelInfo[0].nBlkX; }
    int nBlkY() const { return par_->levelInfo[0].nBlkY; }
    int nBlk0() const { return nBlkX() * nBlkY(); }

    PlaneGeom planeGeom(int shift, int planeRows) const {
        PlaneGeom g;
        g.shift = shift;
        g.bs = BS_ >> shift;
        g.step = (BS_ - ovr_) >> shift;
        g.ovr = ovr_ >> shift;
        g.pitch = shift ? pitchUV_ : pitchY_;
        g.superPitch = shift ? pitchUV_ : pitchY_;
        const int extH = shift
            ? (superPar_->nHeight / superPar_->yRatioUV + 2 * vpadU_)
            : (superPar_->nHeight + 2 * superPar_->nVPad);
        g.imgPitch = g.superPitch * extH;
        g.sliceElems = g.superPitch * planeRows;
        g.width = par_->nWidth >> shift;
        g.height = par_->nHeight >> shift;
        g.nPad = nHPad_ >> shift;
        return g;
    }

    // one delta's MV triples as int3-strided + kt_load_mv_batch + repack
    void stageMv(SvmBuffer& mvI3, SvmBuffer& mvPack,
                 const std::vector<cl_int>& lvl0, Context& ctx, Program& PM) {
        const int nb = nBlk0();
        for (int b = 0; b < nb; b++) {
            cl_int quad[4] = { lvl0[3 * b + 0], lvl0[3 * b + 1],
                               lvl0[3 * b + 2], 0 };
            mvI3.write(quad, (size_t)b * sizeof(cl_int) * 4,
                       sizeof(cl_int) * 4);
        }
        {
            cl_kernel k = PM.kernel("kt_load_mv_batch");
            oclhost::set_arg_svm(k, 0, outSc_.ptr());
            oclhost::set_arg_svm(k, 1, mvI3.ptr());
            oclhost::set_arg_svm(k, 2, vecSc_.ptr());
            oclhost::set_arg_svm(k, 3, sadSc_.ptr());
            oclhost::set_arg(k, 4, nb);
            oclhost::dispatch1D(ctx, k, nb);
        }
        ctx.finish();
        std::vector<cl_int2> vecs((size_t)nb);
        std::vector<cl_int> sads((size_t)nb);
        vecSc_.read(vecs.data(), 0, sizeof(cl_int2) * nb);
        sadSc_.read(sads.data(), 0, sizeof(cl_int) * nb);
        for (int b = 0; b < nb; b++) {
            cl_int trip[3] = { vecs[b].x, vecs[b].y, sads[b] };
            mvPack.write(trip, (size_t)b * sizeof(cl_int) * 3,
                         sizeof(cl_int) * 3);
        }
    }

    void stageRefSlice(SvmBuffer& buf, PVideoFrame& f, int plane) {
        const PlaneGeom& g = plane ? gU_ : gY_;
        const size_t rowbytes = (size_t)g.width * px_;
        for (int y = 0; y < g.height; y++)
            memcpy(stage_.data() + (size_t)y * rowbytes,
                   f->GetReadPtr(plane) + (size_t)y * f->GetPitch(plane),
                   rowbytes);
        buf.write(stage_.data(), 0, rowbytes * g.height);
    }

    // one plane's compensate path: prepare_compensate -> patch + overlap_out
    void runPlane(const PlaneGeom& g, SvmBuffer& src, SvmBuffer& dst,
                  SvmBuffer& refB, SvmBuffer& refF,
                  SvmBuffer& winBuf, bool refUsableB, bool refUsableF,
                  Context& ctx, Program& PM) {
        const int nb = nBlk0();
        {   // ---- kt_prepare_compensate (16 args, .cl order) ----
            cl_kernel k = PM.kernel("kt_prepare_compensate");
            oclhost::set_arg(k, 0, nBlkX());
            oclhost::set_arg(k, 1, nBlkY());
            oclhost::set_arg(k, 2, g.nPad);
            oclhost::set_arg(k, 3, g.bs);
            oclhost::set_arg(k, 4, nTh2_);
            oclhost::set_arg(k, 5, time256_);
            oclhost::set_arg(k, 6, thSAD_);
            oclhost::set_arg(k, 7, nPel_);
            oclhost::set_arg(k, 8, g.shift);
            oclhost::set_arg(k, 9, g.superPitch);
            oclhost::set_arg(k, 10, g.imgPitch);
            oclhost::set_arg_svm(k, 11, sc_.ptr());
            oclhost::set_arg_svm(k, 12, mvPack_.ptr());
            oclhost::set_arg_svm(k, 13, wsl_.ptr());
            oclhost::set_arg_svm(k, 14, rbF_.ptr());
            oclhost::set_arg_svm(k, 15, rbB_.ptr());
            oclhost::dispatch2D(ctx, k, nBlkX(), nBlkY());
        }
        ctx.finish();

        patchOverlap(g, src, dst, refB, refF, winBuf,
                     wsA_, wfA_, wbA_, rbF_, rbB_, refUsableF, ctx, PM);
    }

    // shared combiner + overlap tail: kt_degrain_patch (18 args) then
    // kt_overlap_out (17 args).  For compensate: WSrc=0, WB=0, WF=time256_,
    // refBaseB=0 (unused), refBaseF = prepare_compensate output.
    void patchOverlap(const PlaneGeom& g, SvmBuffer& src, SvmBuffer& dst,
                      SvmBuffer& refB, SvmBuffer& refF, SvmBuffer& winBuf,
                      SvmBuffer& wsrcArr, SvmBuffer& wfArr, SvmBuffer& wbArr,
                      SvmBuffer& rbF, SvmBuffer& rbB, bool refUsableF,
                      Context& ctx, Program& PM) {
        const int nb = nBlk0();
        const int delta = 1;
        // build the constant weight arrays on the host (WSrc=0, WB=0, WF=time256)
        std::vector<cl_int> zeroW((size_t)nb, 0);
        std::vector<cl_int> timeW((size_t)nb, time256_);
        wsA_.write(zeroW.data(), 0, sizeof(cl_int) * nb);
        wbA_.write(zeroW.data(), 0, sizeof(cl_int) * nb);
        wfA_.write(timeW.data(), 0, sizeof(cl_int) * nb);
        // unusable refs: refBase stays 0 with weight 0 (V4)
        if (!refUsableF) {
            std::vector<cl_int> zeroRb((size_t)nb, 0);
            rbF_.write(zeroRb.data(), 0, sizeof(cl_int) * nb);
        }
        std::vector<cl_int> zeroRbB((size_t)nb, 0);
        rbB_.write(zeroRbB.data(), 0, sizeof(cl_int) * nb);

        {   // ---- kt_degrain_patch (18 args) ----
            cl_kernel k = PM.kernel("kt_degrain_patch");
            oclhost::set_arg_svm(k, 0, src.ptr());
            oclhost::set_arg(k, 1, g.pitch);
            oclhost::set_arg(k, 2, nBlkX());
            oclhost::set_arg(k, 3, nBlkY());
            oclhost::set_arg(k, 4, g.bs);
            oclhost::set_arg(k, 5, g.step);
            oclhost::set_arg(k, 6, g.step);
            oclhost::set_arg(k, 7, delta);
            oclhost::set_arg_svm(k, 8, wsA_.ptr());
            oclhost::set_arg_svm(k, 9, wfA_.ptr());
            oclhost::set_arg_svm(k, 10, wbA_.ptr());
            oclhost::set_arg_svm(k, 11, rbF.ptr());
            oclhost::set_arg_svm(k, 12, rbB.ptr());
            oclhost::set_arg_svm(k, 13, refF.ptr());
            oclhost::set_arg(k, 14, g.superPitch);
            oclhost::set_arg_svm(k, 15, refB.ptr());
            oclhost::set_arg(k, 16, g.superPitch);
            oclhost::set_arg_svm(k, 17, patch_.ptr());
            oclhost::dispatch3D(ctx, k, g.bs, g.bs, nb);
        }

        {   // ---- kt_overlap_out (17 args; full plane, edge src-copy) ----
            cl_kernel k = PM.kernel("kt_overlap_out");
            oclhost::set_arg_svm(k, 0, src.ptr());
            oclhost::set_arg(k, 1, g.pitch);
            oclhost::set_arg_svm(k, 2, patch_.ptr());
            oclhost::set_arg(k, 3, g.bs * g.bs);
            oclhost::set_arg(k, 4, nBlkX());
            oclhost::set_arg(k, 5, nBlkY());
            oclhost::set_arg(k, 6, g.bs);
            oclhost::set_arg(k, 7, g.step);
            oclhost::set_arg(k, 8, g.step);
            oclhost::set_arg(k, 9, g.ovr);
            oclhost::set_arg(k, 10, g.ovr);
            oclhost::set_arg_svm(k, 11, winBuf.ptr());
            oclhost::set_arg(k, 12, g.bs * g.bs);
            oclhost::set_arg(k, 13, g.width);
            oclhost::set_arg(k, 14, g.height);
            oclhost::set_arg_svm(k, 15, dst.ptr());
            oclhost::set_arg(k, 16, g.pitch);
            oclhost::dispatch2D(ctx, k, g.width, g.height);
        }
        ctx.finish();
    }

public:
    KMCompensateOCL(PClip child, PClip superClip, PClip vectors,
                    int thSADArg, double timeArg, int thSCD1Arg, int thSCD2Arg,
                    IScriptEnvironment* env_)
        : GenericVideoFilter(child), clip_(nullptr) {
        if (timeArg < 0.0 || timeArg > 100.0)
            env_->ThrowError("KMCompensate: time must be 0.0 to 100.0");
        time256_ = (int)(256.0 * timeArg / 100.0 + 0.5);

        const KMVParam* p0 = KMVParam::GetParam(vectors->GetVideoInfo(), env_);
        if (p0->nDataType != KMVParam::MV_FRAME)
            env_->ThrowError("KMCompensate: vectors is not a KMAnalyse(OCL) MV clip");
        par_ = p0;

        clip_ = new MVClip(*p0, thSCD1Arg, thSCD2Arg);
        thSAD_ = (int)((uint64_t)thSADArg * clip_->nSCD1 / thSCD1Arg);
        nTh1_ = clip_->nSCD1;
        nTh2_ = clip_->nSCD2;

        if (par_->nPixelSize != 1 && par_->nPixelSize != 2)
            env_->ThrowError("KMCompensate: unsupported pixel size");
        if (vi.ComponentSize() != par_->nPixelSize)
            env_->ThrowError("KMCompensate: source pixel size does not match the MV clip");
        if (vi.width != par_->nWidth || vi.height != par_->nHeight)
            env_->ThrowError("KMCompensate: source dims %dx%d do not match the MV clip's %dx%d",
                             vi.width, vi.height, par_->nWidth, par_->nHeight);
        if (nBlkX() < 3 || nBlkY() < 3)
            env_->ThrowError("KMCompensate: grids with fewer than 3 blocks per axis are not valid");

        try {
            const KMVParam* sp = KMVParam::GetParam(superClip->GetVideoInfo(), env_);
            superPar_ = sp;
        } catch (const AvisynthError&) {
            env_->ThrowError("KMCompensate: the super clip must be a KMSuper(OCL) clip");
        }
        super_ = superClip;
        mvClip_ = vectors;

        BS_ = par_->nBlkSizeX; ovr_ = par_->nOverlapX;
        nPel_ = par_->nPel;
        nHPad_ = par_->nHPad;
        px_ = par_->nPixelSize;
        logUVx_ = nlog2(par_->xRatioUV);
        logUVy_ = nlog2(par_->yRatioUV);
        pitchY_ = super_->GetVideoInfo().width;
        planeHY_ = super_->GetVideoInfo().height;
        pitchUV_ = pitchY_ >> logUVx_;
        planeHUV_ = planeHY_ >> logUVy_;
        vpadU_ = superPar_->nVPad / superPar_->yRatioUV;
        hasUV_ = (vi.IsYUV() || vi.IsYUVA()) &&
                 super_->GetVideoInfo().IsPlanar() &&
                 super_->GetVideoInfo().IsYUV();
        if (hasUV_ && superPar_->nWidth >> logUVx_ == 0) hasUV_ = false;

        YUVplanes_ = hasUV_ ? 7 : 1;

        gY_ = planeGeom(0, planeHY_);
        gU_ = planeGeom(1, planeHUV_);

        BuildOverlapWindows(BS_, BS_, ovr_, ovr_, winY_);
        if (hasUV_)
            BuildOverlapWindows(BS_ >> 1, BS_ >> 1, ovr_ >> 1, ovr_ >> 1, winUV_);

        try {
            Context& ctx = sharedContext();
            const size_t pxs = (size_t)px_;
            const int nb = nBlk0();
            const size_t bytesY = (size_t)pitchY_ * planeHY_ * pxs;
            const size_t bytesUV = (size_t)pitchUV_ * planeHUV_ * pxs;
            srcY_ = SvmBuffer(ctx, bytesY);
            dstY_ = SvmBuffer(ctx, bytesY);
            refBY_ = SvmBuffer(ctx, bytesY);
            refFY_ = SvmBuffer(ctx, bytesY);
            if (hasUV_) {
                srcU_ = SvmBuffer(ctx, bytesUV);
                srcV_ = SvmBuffer(ctx, bytesUV);
                dstU_ = SvmBuffer(ctx, bytesUV);
                dstV_ = SvmBuffer(ctx, bytesUV);
                refBU_ = SvmBuffer(ctx, bytesUV);
                refBV_ = SvmBuffer(ctx, bytesUV);
                refFU_ = SvmBuffer(ctx, bytesUV);
                refFV_ = SvmBuffer(ctx, bytesUV);
            }
            stage_.resize(std::max(bytesY, bytesUV));
            mvI3_ = SvmBuffer(ctx, sizeof(cl_int) * 4 * nb);
            mvPack_ = SvmBuffer(ctx, sizeof(cl_int) * 3 * nb);
            outSc_ = SvmBuffer(ctx, sizeof(cl_int) * 4 * nb);
            vecSc_ = SvmBuffer(ctx, sizeof(cl_int2) * nb);
            sadSc_ = SvmBuffer(ctx, sizeof(cl_int) * nb);
            sc_ = SvmBuffer(ctx, sizeof(cl_int));
            use_ = SvmBuffer(ctx, sizeof(cl_int));
            wsl_ = SvmBuffer(ctx, sizeof(cl_int) * nb);
            wsA_ = SvmBuffer(ctx, sizeof(cl_int) * nb);
            wbA_ = SvmBuffer(ctx, sizeof(cl_int) * nb);
            wfA_ = SvmBuffer(ctx, sizeof(cl_int) * nb);
            rbB_ = SvmBuffer(ctx, sizeof(cl_int) * nb);
            rbF_ = SvmBuffer(ctx, sizeof(cl_int) * nb);
            patch_ = SvmBuffer(ctx, pxs * (size_t)BS_ * BS_ * nb);
            winYd_ = SvmBuffer(ctx, winY_.size() * sizeof(short));
            winYd_.write(winY_.data(), 0, winY_.size() * sizeof(short));
            if (hasUV_) {
                winUVd_ = SvmBuffer(ctx, winUV_.size() * sizeof(short));
                winUVd_.write(winUV_.data(), 0, winUV_.size() * sizeof(short));
            }
            motionProgram();
        } catch (const std::exception& e) {
            env_->ThrowError("[KMCompensateOCL] OpenCL init failed: %s", e.what());
        }
        static std::mutex mtx;
        mtx_ = &mtx;
    }

    PVideoFrame __stdcall GetFrame(int n, IScriptEnvironment* env_) {
        try {
            return GetFrameT(n, env_);
        } catch (const std::exception& e) {
            env_->ThrowError("[KMCompensateOCL] %s", e.what());
        }
        return PVideoFrame();
    }

    int __stdcall SetCacheHints(int cachehints, int frame_range) {
        if (cachehints == CACHE_GET_MTMODE) return MT_MULTI_INSTANCE;
        return 0;
    }

private:
    PVideoFrame GetFrameT(int n, IScriptEnvironment* env_) {
        std::lock_guard<std::mutex> lock(*mtx_);
        Context& ctx = sharedContext();
        Program& PM = motionProgram();
        const int nb = nBlk0();
        const int nf = super_->GetVideoInfo().num_frames;

        // ---- 1. MV frame + usability (single clip at index n)
        std::vector<cl_int> lvl0;
        const bool isValid = clip_->readFrame(mvClip_, n, env_, lvl0);
        const int refIdx = clip_->refIndex(n);
        const bool isUsable = isValid && refIdx >= 0 && refIdx < nf;
        cl_int u = isUsable ? 1 : 0;
        use_.write(&u, 0, sizeof(cl_int));

        // ---- 2/3. MV staging + scene-change (single kt_scene_change)
        stageMv(mvI3_, mvPack_, lvl0, ctx, PM);
        cl_int sc = 0;
        sc_.write(&sc, 0, sizeof(cl_int));   // zero by kt_init_scene_change
        {
            cl_kernel ki = PM.kernel("kt_init_scene_change");
            oclhost::set_arg_svm(ki, 0, sc_.ptr());
            oclhost::dispatch1D(ctx, ki, 1);
        }
        if (isUsable) {
            cl_kernel k = PM.kernel("kt_scene_change");
            oclhost::set_arg_svm(k, 0, mvI3_.ptr());
            oclhost::set_arg(k, 1, nb);
            oclhost::set_arg(k, 2, nTh1_);
            oclhost::set_arg_svm(k, 3, sc_.ptr());
            oclhost::dispatch1D(ctx, k, nb);
        }
        ctx.finish();
        cl_int scv = 0;
        sc_.read(&scv, 0, sizeof(cl_int));
        const bool refUsable = isUsable && !(scv > nTh2_);

        // ---- 4. stage source + usable super ref
        PVideoFrame src = child->GetFrame(n, env_);
        stagePlaneRows(srcY_, stage_, src->GetReadPtr(PLANAR_Y),
                       src->GetPitch(PLANAR_Y), vi.width, vi.height, px_);
        if (hasUV_) {
            stagePlaneRows(srcU_, stage_, src->GetReadPtr(PLANAR_U),
                           src->GetPitch(PLANAR_U), vi.width >> logUVx_,
                           vi.height >> logUVy_, px_);
            stagePlaneRows(srcV_, stage_, src->GetReadPtr(PLANAR_V),
                           src->GetPitch(PLANAR_V), vi.width >> logUVx_,
                           vi.height >> logUVy_, px_);
        }
        if (refUsable) {
            PVideoFrame r = super_->GetFrame(refIdx, env_);
            stageRefSlice(refBY_, r, PLANAR_Y);
            if (hasUV_) {
                stageRefSlice(refBU_, r, PLANAR_U);
                stageRefSlice(refBV_, r, PLANAR_V);
            }
        }
        ctx.finish();

        // ---- 5. per-plane prepare + patch + overlap
        const bool enY = (YUVplanes_ & 1) != 0;
        const bool enU = (YUVplanes_ & 2) != 0;
        const bool enV = (YUVplanes_ & 4) != 0;
        if (enY)
            runPlane(gY_, srcY_, dstY_, refBY_, refFY_, winYd_,
                     refUsable, refUsable, ctx, PM);
        if (hasUV_ && (enU || enV)) {
            if (enU)
                runPlane(gU_, srcU_, dstU_, refBU_, refFU_, winUVd_,
                         refUsable, refUsable, ctx, PM);
            if (enV)
                runPlane(gU_, srcV_, dstV_, refBV_, refFV_, winUVd_,
                         refUsable, refUsable, ctx, PM);
        }

        // ---- 6. copy-back
        PVideoFrame dst = env_->NewVideoFrame(vi);
        const int wUV = vi.width >> logUVx_, hUV = vi.height >> logUVy_;
        if (enY)
            readPlaneFromStaged(dstY_, dst->GetWritePtr(PLANAR_Y),
                                dst->GetPitch(PLANAR_Y), vi.width, vi.height,
                                px_, stage_);
        else
            copyPlaneToFrame(dst->GetWritePtr(PLANAR_Y), dst->GetPitch(PLANAR_Y),
                             src->GetReadPtr(PLANAR_Y), src->GetPitch(PLANAR_Y),
                             vi.width, vi.height, px_);
        if (hasUV_) {
            const bool cpU = !enU, cpV = !enV;
            if (cpU)
                copyPlaneToFrame(dst->GetWritePtr(PLANAR_U), dst->GetPitch(PLANAR_U),
                                 src->GetReadPtr(PLANAR_U), src->GetPitch(PLANAR_U),
                                 wUV, hUV, px_);
            else
                readPlaneFromStaged(dstU_, dst->GetWritePtr(PLANAR_U),
                                    dst->GetPitch(PLANAR_U), wUV, hUV, px_, stage_);
            if (cpV)
                copyPlaneToFrame(dst->GetWritePtr(PLANAR_V), dst->GetPitch(PLANAR_V),
                                 src->GetReadPtr(PLANAR_V), src->GetPitch(PLANAR_V),
                                 wUV, hUV, px_);
            else
                readPlaneFromStaged(dstV_, dst->GetWritePtr(PLANAR_V),
                                    dst->GetPitch(PLANAR_V), wUV, hUV, px_, stage_);
        }
        return dst;
    }
};

}  // namespace

// ============================================================ REGISTRATION
static AVSValue __cdecl CreateKMCompensateOCL(AVSValue args, void*, IScriptEnvironment* env) {
    try {
        return new KMCompensateOCL(
            args[0].AsClip(),       // source
            args[1].AsClip(),       // super
            args[2].AsClip(),       // vectors (KMAnalyse MV clip)
            args[3].AsInt(10000),   // thSAD
            args[4].AsFloat(100),   // time (percent)
            args[5].AsInt(400),     // thSCD1
            args[6].AsInt(130),     // thSCD2
            env);
    } catch (const std::exception& e) {
        env->ThrowError("[KMCompensateOCL] %s", e.what());
    }
    return AVSValue();
}

static int planeToYUVplanes(int plane) {
    switch (plane) {
    case 0: return 1;
    case 1: return 2;
    case 2: return 4;
    case 3: return 6;
    default: return 7;   // plane==4 (upstream default) -> all planes
    }
}

static AVSValue __cdecl CreateKMDegrain1OCL(AVSValue args, void*, IScriptEnvironment* env) {
    try {
        // delta=1: script args are source,super,mvbw,mvfw,thSAD,thSADC,plane,limit,limitC,thSCD1,thSCD2,binomial,useFlag
        const int bpc = args[0].AsClip()->GetVideoInfo().BitsPerComponent();
        const int defLimit = (1 << bpc) - 1;
        return new KMDegrainOCL(
            args[0].AsClip(), args[1].AsClip(),
            args[2].AsClip(), args[3].AsClip(),
            nullptr, nullptr,                 // mvbw2/mvfw2 unused for delta 1
            args[4].AsInt(400),               // thSAD
            args[5].AsInt(400),               // thSADC
            planeToYUVplanes(args[6].AsInt(4)), // plane -> YUVplanes (upstream MV.cpp:4669-4685)
            args[7].AsInt(defLimit),          // limit
            args[8].AsInt(defLimit),          // limitC
            args[9].AsInt(400),               // thSCD1
            args[10].AsInt(130),              // thSCD2
            args[11].AsBool(false),           // binomial
            args[12].AsInt(0),                // useFlag
            1,                                // delta
            env);
    } catch (const std::exception& e) {
        env->ThrowError("[KMDegrain1OCL] %s", e.what());
    }
    return AVSValue();
}

static AVSValue __cdecl CreateKMDegrain2OCL(AVSValue args, void*, IScriptEnvironment* env) {
    try {
        // delta=2: script args are source,super,mvbw,mvfw,mvbw2,mvfw2,thSAD,thSADC,plane,limit,limitC,thSCD1,thSCD2,binomial,useFlag
        const int bpc = args[0].AsClip()->GetVideoInfo().BitsPerComponent();
        const int defLimit = (1 << bpc) - 1;
        return new KMDegrainOCL(
            args[0].AsClip(), args[1].AsClip(),
            args[2].AsClip(), args[3].AsClip(),
            args[4].AsClip(), args[5].AsClip(),
            args[6].AsInt(400),               // thSAD
            args[7].AsInt(400),               // thSADC
            planeToYUVplanes(args[8].AsInt(4)), // plane -> YUVplanes
            args[9].AsInt(defLimit),          // limit
            args[10].AsInt(defLimit),         // limitC
            args[11].AsInt(400),              // thSCD1
            args[12].AsInt(130),              // thSCD2
            args[13].AsBool(false),           // binomial
            args[14].AsInt(0),                // useFlag
            2,                                // delta
            env);
    } catch (const std::exception& e) {
        env->ThrowError("[KMDegrain2OCL] %s", e.what());
    }
    return AVSValue();
}

// AviSynth+ linkage (no import lib): core passes the function table.
const AVS_Linkage* AVS_linkage{ nullptr };

extern "C" __declspec(dllexport) const char* __stdcall
AvisynthPluginInit3(IScriptEnvironment* env, const AVS_Linkage* const vectors) {
    AVS_linkage = vectors;
    try {
        env->AddFunction("KMCompensateOCL",
            "ccc[thSAD]i[time]f[thSCD1]i[thSCD2]i",
            CreateKMCompensateOCL, 0);
        env->AddFunction("KMCompensate",
            "ccc[thSAD]i[time]f[thSCD1]i[thSCD2]i",
            CreateKMCompensateOCL, 0);
        env->AddFunction("KMDegrain1OCL",
            "cccc[thSAD]i[thSADC]i[plane]i[limit]i[limitC]i[thSCD1]i[thSCD2]i[binomial]b[useFlag]i",
            CreateKMDegrain1OCL, 0);
        env->AddFunction("KMDegrain1",
            "cccc[thSAD]i[thSADC]i[plane]i[limit]i[limitC]i[thSCD1]i[thSCD2]i[binomial]b[useFlag]i",
            CreateKMDegrain1OCL, 0);
        env->AddFunction("KMDegrain2OCL",
            "cccccc[thSAD]i[thSADC]i[plane]i[limit]i[limitC]i[thSCD1]i[thSCD2]i[binomial]b[useFlag]i",
            CreateKMDegrain2OCL, 0);
        env->AddFunction("KMDegrain2",
            "cccccc[thSAD]i[thSADC]i[plane]i[limit]i[limitC]i[thSCD1]i[thSCD2]i[binomial]b[useFlag]i",
            CreateKMDegrain2OCL, 0);
    } catch (const std::exception& e) {
        return e.what();
    }
    return "KMCompensate/KMDegrain1/KMDegrain2 (OpenCL port of KTGMC/MV.cpp, GPL; tenth host plugin / third motion-stage plugin)";
}
