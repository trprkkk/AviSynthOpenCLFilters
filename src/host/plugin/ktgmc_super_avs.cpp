// ktgmc_super_avs.cpp — AviSynth+ host plugin for the OpenCL port of KTGMC's
// KMSuper (upstream: rigaya/AviSynthCUDAFilters KTGMC/MV.cpp + MVKernel.cu,
// GPL).  Eighth real AviSynth+ host plugin of the port (after the seven KFM
// ones), and the FIRST motion-stage plugin.
//
// It replays the upstream KMSuper pipeline on the ALG-VERIFIED / device-
// confirmed ported kernels, embedded into the DLL at build time:
//
//   KMFrame::FillPad  -> kl_copy_pad        -> kt_copy_pad          (motion.cl)
//   KMPlane::Refine   -> kl_horizontal_wiener -> kt_horizontal_wiener (simple.cl)
//                        kl_vertical_wiener   -> kt_vertical_wiener   (simple.cl)
//   KMFrame::ReduceToPad -> kl_RB2B_bilinear_filtered_with_pad
//                                          -> kt_rb2b_bilinear_filtered_with_pad
//                                                                   (motion.cl)
//
// SUPER-FRAME LAYOUT (MV.cpp KMPlane / KMFrame / KMSuperFrame / KMSuper, all
// transcribed here; see docs/MV_PORT_SPEC.md §5 and STAGE-0 inventory):
//   * The output clip's video info is enlarged to
//       nSuperWidth  = width  + 2*nHPad
//       nSuperHeight = PlaneSuperOffset(false, height, nLevels, nPel,
//                                       nVPad, nSuperWidth, yRatioUV)
//                      / nSuperWidth, rounded up to even when yRatioUV==2.
//   * Each plane (Y at nSuperWidth x nSuperHeight, U/V at the subsampled size)
//     is ONE contiguous buffer holding the whole pyramid.  Level 0 holds
//     nPel*nPel sub-pel planes of extent nPitch x (H+2*vpad) stacked nPitch*
//     (H+2*vpad) apart; levels L>=1 each hold one reduced plane.  Level L's
//     element offset is PlaneSuperOffset(...).  All offsets are computed
//     against a TIGHT device pitch (nSuperWidth for Y, nSuperWidth>>logUVx for
//     chroma); the produced frame is copied back row-by-row.
//   * Interior origin of level L sub-plane s (nOffsetPadding):
//       off(L) + s*nPitch*(H_L+2*vpad) + nPitch*vpad + hpad.
//
// INTERIOR-ORIGIN POINTERS / SVM (documented deviation 1): upstream's
// FillPad/ReduceToPad call sites pass the plane INTERIOR as the base pointer
// (the kernels index dst[x + y*pitch] with x=gid-hPad allowed negative, i.e.
// the border is written through negative offsets).  cl_mem sub-buffers are
// broken on this driver (RIG_HANDOFF §9.4), so this plugin allocates the
// whole pyramid in coarse-grain buffer SVM (probe_svm.cpp, 2026-10-31) and
// passes base+offset interior pointers, exactly reproducing upstream's pointer
// arithmetic.  This is NOT a host-staging substitution: both pad kernels still
// run on the device.
//
// DOCUMENTED DEVIATIONS (none change any produced pixel for the pinned params):
//   1. SVM pyramid buffers instead of cl_mem + interior sub-buffers (above).
//   2. The child's CUDA/device-match check is meaningless on the one OpenCL
//      queue and is dropped (KNoiseClipOCL precedent).
//   3. When chroma=false the U/V planes of the output frame are left zero
//      (upstream leaves them uninitialised; they are never read downstream).
//   4. Output frame is copied back row-by-row into the AVS pitch (the device
//      pyramid uses a tight pitch); identical values.
//   5. The pyramid buffers are zero-filled at construction so the coarse-level
//      tail columns (beyond each level's extended width — never read by the
//      Search/Compensate consumers) are deterministic; upstream leaves them
//      undefined.
//   6. (2026-11-01, ninth-plugin support) The SUPER_FRAME KMVParam is
//      smuggled into the VideoInfo exactly like upstream KMSuper
//      (common/KMV.h SetParam: sample_type/num_audio_samples point at the
//      per-instance param).  No produced pixel changes; without it a chained
//      KMAnalyseOCL could not learn hpad/vpad/pel/levels/chroma the way
//      upstream KMAnalyse does.
//
// PARAMETERS (upstream KMSuper::Create verbatim):
//   clip, hpad(int,8), vpad(int,8), pel(int,2), levels(int,0=auto),
//   chroma(bool,true), sharp(int,2), filter(int,2)
//   upstream validation: hpad==8, vpad==8, sharp==2, filter==2, pel in {1,2}.
//
// STATE MODEL: like upstream, the class caches its pyramid buffers (pFrames)
// across GetFrame calls but overwrites them wholesale each call; output frame n
// is a pure function of child(n) + the fixed config.  The device buffers are
// per-instance scratch, serialised through the shared in-order queue + a
// process mutex.
//
// UPSTREAM REGISTRATION (MV.cpp:5228):
//   env->AddFunction("KMSuper",
//       "c[hpad]i[vpad]i[pel]i[levels]i[chroma]b[sharp]i[filter]i",
//       KMSuper::Create, 0);
// This DLL registers KMSuperOCL with that exact signature and "KMSuper" as an
// alias.
#include <cstdint>
#include <cstring>
#include <algorithm>
#include <mutex>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "avisynth.h"
#include "../ocl_host.h"
#include "ktgmc_simple_cl.h"  // GENERATED from .../ktgmc/kernels/ktgmc_simple.cl
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

// ---------------------------------------------------------------------------
// MV.cpp geometry helpers (verbatim transcription)
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

// PlaneSuperOffset: element offset of `level` inside a plane whose row stride
// is plane_pitch.  chroma selects the chroma height-chain; src_height/vpad are
// the plane's own (already subsampled for chroma).
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

// ---------------------------------------------------------------------------
// one plane's build description
// ---------------------------------------------------------------------------
struct PlaneGeom {
    int pitch = 0;     // tight device pitch (elements)
    int H = 0;         // interior height at level 0 (elements)
    int W = 0;         // interior width at level 0
    int hpad = 0, vpad = 0;
    bool chroma = false;
    std::vector<int> off;             // level element offsets
    std::vector<std::pair<int,int>> dim;  // (W,H) per level
};

class KMSuperOCL : public GenericVideoFilter {
    // ---- config ----
    int nHPad_, nVPad_, nPel_, nLevels_, nSharp_, nRFilter_;
    bool chroma_;
    int xRatioUV_, yRatioUV_, logUVx_, logUVy_;
    int nPixelSize_, nBitsPerPixel_, nPixelShift_;
    int nSuperWidth_, nSuperHeight_;

    // ---- device ----
    SvmBuffer pY_, pU_, pV_;      // output pyramid planes
    SvmBuffer sY_, sU_, sV_;      // source scratch (tight)
    int pitchY_ = 0, pitchUV_ = 0;
    int planeHY_ = 0, planeHUV_ = 0;
    PlaneGeom gY_, gU_, gV_;

    std::vector<uint8_t> stage_;  // host staging (byte sized = max plane bytes)
    std::mutex* mtx_;
    KMVParam kmv_;  // SUPER_FRAME params smuggled to KMAnalyseOCL via the vi
                    // (upstream KMSuper MV.cpp:721/781: SetParam(vi, &params))

    Program& simpleProgram() {
        return oclhost::ProgramRegistry::instance().get(
            sharedContext(), "ktgmc_simple", ktgmc_simple_cl_src, nPixelSize_);
    }
    Program& motionProgram() {
        return oclhost::ProgramRegistry::instance().get(
            sharedContext(), "ktgmc_motion", ktgmc_motion_cl_src, nPixelSize_);
    }

    void computeGeom(PlaneGeom& g, bool isChroma) {
        g.hpad = isChroma ? nHPad_ / xRatioUV_ : nHPad_;
        g.vpad = isChroma ? nVPad_ / yRatioUV_ : nVPad_;
        g.pitch = isChroma ? pitchUV_ : pitchY_;
        g.chroma = isChroma;
        g.W = isChroma ? (origW_ / xRatioUV_) : origW_;
        g.H = isChroma ? (origH_ / yRatioUV_) : origH_;
        g.off.assign(nLevels_, 0);
        g.dim.assign(nLevels_, std::make_pair(0, 0));
        g.dim[0] = std::make_pair(g.W, g.H);
        for (int L = 1; L < nLevels_; L++) {
            int lw = PlaneWidthLuma(origW_, L, xRatioUV_, nHPad_);
            int lh = PlaneHeightLuma(origH_, L, yRatioUV_, nVPad_);
            g.dim[L] = isChroma ? std::make_pair(lw / xRatioUV_, lh / yRatioUV_)
                                : std::make_pair(lw, lh);
        }
    }

    // ---- device launches (PX-templated so the .cl PX instantiation matches) --
    template <typename PX>
    void dispatchPlane(PlaneGeom& g, SvmBuffer& plane, SvmBuffer& src, int W, int H) {
        Context& ctx = sharedContext();
        const size_t px = sizeof(PX);
        const int P = g.pitch;
        const int hpad = g.hpad, vpad = g.vpad;
        const int nLevels = nLevels_;
        const bool isChroma = g.chroma;

        // level offsets / dims
        for (int L = 0; L < nLevels; L++) {
            g.off[L] = (int)PlaneSuperOffset(isChroma, g.H, L, nPel_,
                                             g.vpad, P, yRatioUV_);
        }

        // ---- 1. mirror-pad level-0 sub-plane 0 (kt_copy_pad, reflection) ----
        {
            const size_t int0 = (size_t)g.off[0] + (size_t)P * vpad + hpad;
            const int extW = W + 2 * hpad, extH = H + 2 * vpad;
            cl_kernel k = motionProgram().kernel("kt_copy_pad");
            oclhost::set_arg_svm(k, 0, plane.at(int0 * px));   // dst interior origin
            oclhost::set_arg(k, 1, P);
            oclhost::set_arg_svm(k, 2, src.ptr());             // src tight
            oclhost::set_arg(k, 3, W);
            oclhost::set_arg(k, 4, hpad);
            oclhost::set_arg(k, 5, vpad);
            oclhost::set_arg(k, 6, W);
            oclhost::set_arg(k, 7, H);
            oclhost::dispatch2D(ctx, k, extW, extH);
        }

        // ---- 2. Refine (nPel==2): wiener sub-planes 1,2,3 ----
        if (nPel_ == 2) {
            const int extW = W + 2 * hpad, extH = H + 2 * vpad;
            const size_t ext = (size_t)P * extH;
            const size_t o0 = (size_t)g.off[0];
            auto hw = [&](size_t dstoff, size_t srcoff) {
                cl_kernel k = simpleProgram().kernel("kt_horizontal_wiener");
                oclhost::set_arg_svm(k, 0, plane.at(srcoff * px));
                oclhost::set_arg(k, 1, P);
                oclhost::set_arg_svm(k, 2, plane.at(dstoff * px));
                oclhost::set_arg(k, 3, P);
                oclhost::set_arg(k, 4, extW);
                oclhost::set_arg(k, 5, extH);
                oclhost::dispatch2D(ctx, k, extW, extH);
            };
            auto vw = [&](size_t dstoff, size_t srcoff) {
                cl_kernel k = simpleProgram().kernel("kt_vertical_wiener");
                oclhost::set_arg_svm(k, 0, plane.at(srcoff * px));
                oclhost::set_arg(k, 1, P);
                oclhost::set_arg_svm(k, 2, plane.at(dstoff * px));
                oclhost::set_arg(k, 3, P);
                oclhost::set_arg(k, 4, extW);
                oclhost::set_arg(k, 5, extH);
                oclhost::dispatch2D(ctx, k, extW, extH);
            };
            hw(o0 + 1 * ext, o0);        // sub1 = H(sub0)
            vw(o0 + 2 * ext, o0);        // sub2 = V(sub0)
            hw(o0 + 3 * ext, o0 + 2 * ext);  // sub3 = H(sub2)
        }

        // ---- 3. coarse levels: fused RB2B reduce + pad ----
        for (int L = 0; L + 1 < nLevels; L++) {
            const int Wd = g.dim[L + 1].first, Hd = g.dim[L + 1].second;
            const size_t srcInt = (size_t)g.off[L] + (size_t)P * vpad + hpad;
            const size_t dstInt = (size_t)g.off[L + 1] + (size_t)P * vpad + hpad;
            cl_kernel k = motionProgram().kernel("kt_rb2b_bilinear_filtered_with_pad");
            oclhost::set_arg_svm(k, 0, plane.at(srcInt * px));
            oclhost::set_arg(k, 1, P);
            oclhost::set_arg_svm(k, 2, plane.at(dstInt * px));
            oclhost::set_arg(k, 3, P);
            oclhost::set_arg(k, 4, Wd);
            oclhost::set_arg(k, 5, Hd);
            oclhost::set_arg(k, 6, hpad);
            oclhost::set_arg(k, 7, vpad);
            oclhost::dispatch2D(ctx, k, Wd + 2 * hpad, Hd + 2 * vpad);
        }
    }

    void buildGeomTable(PlaneGeom& g, bool isChroma) { computeGeom(g, isChroma); }

public:
    int origW_, origH_;

    KMSuperOCL(PClip child, int nHPad, int nVPad, int nPel, int nLevels, bool chroma,
               int nSharp, int nRFilter, IScriptEnvironment* env_)
        : GenericVideoFilter(child)
        , nHPad_(nHPad), nVPad_(nVPad), nPel_(nPel), nLevels_(nLevels)
        , chroma_(chroma), nSharp_(nSharp), nRFilter_(nRFilter)
        , kmv_(KMVParam::SUPER_FRAME) {
        // upstream parameter validation (verbatim)
        if (nHPad_ != 8) env_->ThrowError("[KMSuper] hpad must be 8.");
        if (nVPad_ != 8) env_->ThrowError("[KMSuper] vpad must be 8.");
        if (nSharp_ != 2) env_->ThrowError("[KMSuper] sharp must be 2.");
        if (nRFilter_ != 2) env_->ThrowError("[KMSuper] rfilter must be 2.");
        if (nPel_ != 1 && nPel_ != 2) env_->ThrowError("[KMSuper] pel must be 1 or 2");

        if (!vi.IsPlanar())
            env_->ThrowError("[KMSuperOCL] planar input required");

        origW_ = vi.width;
        origH_ = vi.height;
        yRatioUV_ = 1 << vi.GetPlaneHeightSubsampling(PLANAR_U);
        xRatioUV_ = 1 << vi.GetPlaneWidthSubsampling(PLANAR_U);
        logUVy_ = vi.GetPlaneHeightSubsampling(PLANAR_U);
        logUVx_ = vi.GetPlaneWidthSubsampling(PLANAR_U);
        nPixelSize_ = vi.ComponentSize();
        nBitsPerPixel_ = vi.BitsPerComponent();
        nPixelShift_ = (nPixelSize_ == 1) ? 0 : 1;

        // nLevelsMax (verbatim)
        int nLevelsMax = 0;
        while (PlaneHeightLuma(origH_, nLevelsMax, yRatioUV_, nVPad_) >= yRatioUV_ * 2 &&
               PlaneWidthLuma(origW_, nLevelsMax, xRatioUV_, nHPad_) >= xRatioUV_ * 2) {
            nLevelsMax++;
        }
        if (nLevels_ <= 0 || nLevels_ > nLevelsMax) nLevels_ = nLevelsMax;
        if (nLevels_ < 1) nLevels_ = 1;

        // super dimensions (verbatim)
        nSuperWidth_ = origW_ + 2 * nHPad_;
        nSuperHeight_ = (int)(PlaneSuperOffset(false, origH_, nLevels_, nPel_, nVPad_,
                                               nSuperWidth_, yRatioUV_) / (unsigned)nSuperWidth_);
        if (yRatioUV_ == 2 && (nSuperHeight_ & 1)) nSuperHeight_++;

        vi.width = nSuperWidth_;
        vi.height = nSuperHeight_;

        // smuggle the SUPER_FRAME params into the vi exactly like upstream
        // KMSuper (MV.cpp:740-781) so a chained KMAnalyseOCL can read them
        // via KMVParam::GetParam — nWidth/nHeight are the PRE-super dims.
        kmv_.nWidth = origW_;
        kmv_.nHeight = origH_;
        kmv_.nActualWidth = 0;
        kmv_.nActualHeight = 0;
        kmv_.yRatioUV = yRatioUV_;
        kmv_.xRatioUV = xRatioUV_;
        kmv_.nHPad = nHPad_;
        kmv_.nVPad = nVPad_;
        kmv_.nPel = nPel_;
        kmv_.chroma = chroma_;
        kmv_.nLevels = nLevels_;
        kmv_.nDropLevels = 0;
        kmv_.nPixelSize = nPixelSize_;
        kmv_.nBitsPerPixel = nBitsPerPixel_;
        kmv_.nPixelShift = nPixelShift_;
        kmv_.pixelType = vi.pixel_type;
        KMVParam::SetParam(vi, &kmv_);

        // device pitches / plane sizes (tight)
        pitchY_ = nSuperWidth_;
        planeHY_ = nSuperHeight_;
        pitchUV_ = nSuperWidth_ >> logUVx_;
        planeHUV_ = nSuperHeight_ >> logUVy_;

        static std::mutex mtx;
        mtx_ = &mtx;

        try {
            Context& ctx = sharedContext();
            const size_t bytesY = (size_t)pitchY_ * planeHY_ * nPixelSize_;
            const size_t bytesUV = (size_t)pitchUV_ * planeHUV_ * nPixelSize_;
            pY_ = SvmBuffer(ctx, bytesY);
            sY_ = SvmBuffer(ctx, (size_t)origW_ * origH_ * nPixelSize_);
            if (chroma_) {
                pU_ = SvmBuffer(ctx, bytesUV);
                pV_ = SvmBuffer(ctx, bytesUV);
                sU_ = SvmBuffer(ctx, (size_t)(origW_ / xRatioUV_) * (origH_ / yRatioUV_) * nPixelSize_);
                sV_ = SvmBuffer(ctx, (size_t)(origW_ / xRatioUV_) * (origH_ / yRatioUV_) * nPixelSize_);
            }
            stage_.resize(std::max(bytesY, bytesUV));
            // Deterministic zero-fill: each level writes only its extended
            // width; the tail columns of coarse levels (beyond the level's
            // extended width, never read by the Search/Compensate consumers) are
            // otherwise uninitialised.  Upstream leaves them undefined; zeroing
            // is a harmless, documented deviation (deviation 5).
            {
                std::vector<uint8_t> z(bytesY, 0);
                pY_.write(z.data(), 0, bytesY);
                if (chroma_) { std::vector<uint8_t> zu(bytesUV, 0); pU_.write(zu.data(), 0, bytesUV); pV_.write(zu.data(), 0, bytesUV); }
            }
            buildGeomTable(gY_, false);
            if (chroma_) buildGeomTable(gU_, true);
            simpleProgram();  // force both builds up front
            motionProgram();
        } catch (const std::exception& e) {
            env_->ThrowError("[KMSuperOCL] OpenCL init failed: %s", e.what());
        }
    }

    PVideoFrame __stdcall GetFrame(int n, IScriptEnvironment* env_) {
        try {
            return GetFrameT(n, env_);
        } catch (const std::exception& e) {
            env_->ThrowError("[KMSuperOCL] %s", e.what());
        }
        return PVideoFrame();
    }

    int __stdcall SetCacheHints(int cachehints, int frame_range) {
        // mirror upstream KMSuper::SetCacheHints
        if (cachehints == CACHE_GET_MTMODE) return MT_MULTI_INSTANCE;
        return 0;
    }

private:
    // upload a PX plane row-by-row into a tight SVM source buffer
    template <typename PX>
    void uploadPlane(SvmBuffer& dst, const uint8_t* srcp, int srcPitchBytes,
                     int w, int h, std::vector<uint8_t>& stage) {
        const size_t rowbytes = (size_t)w * sizeof(PX);
        for (int y = 0; y < h; y++)
            memcpy(stage.data() + (size_t)y * rowbytes,
                   srcp + (size_t)y * srcPitchBytes, rowbytes);
        dst.write(stage.data(), 0, rowbytes * h);
    }

    template <typename PX>
    void buildAndDownload(PlaneGeom& g, SvmBuffer& plane, SvmBuffer& src,
                          int w, int h) {
        dispatchPlane<PX>(g, plane, src, w, h);
    }

    PVideoFrame GetFrameT(int n, IScriptEnvironment* env_) {
        std::lock_guard<std::mutex> lock(*mtx_);
        PVideoFrame srcFrame = child->GetFrame(n, env_);

        const int w = origW_, h = origH_;
        const int wUV = w / xRatioUV_, hUV = h / yRatioUV_;
        Context& ctx = sharedContext();

        // ---- upload source planes ----
        if (nPixelSize_ == 1) {
            uploadPlane<uint8_t>(sY_, srcFrame->GetReadPtr(PLANAR_Y),
                                 srcFrame->GetPitch(PLANAR_Y), w, h, stage_);
            if (chroma_) {
                uploadPlane<uint8_t>(sU_, srcFrame->GetReadPtr(PLANAR_U),
                                     srcFrame->GetPitch(PLANAR_U), wUV, hUV, stage_);
                uploadPlane<uint8_t>(sV_, srcFrame->GetReadPtr(PLANAR_V),
                                     srcFrame->GetPitch(PLANAR_V), wUV, hUV, stage_);
            }
            buildAndDownload<uint8_t>(gY_, pY_, sY_, w, h);
            if (chroma_) {
                buildAndDownload<uint8_t>(gU_, pU_, sU_, wUV, hUV);
                buildAndDownload<uint8_t>(gU_, pV_, sV_, wUV, hUV);
            }
        } else {
            uploadPlane<uint16_t>(sY_, srcFrame->GetReadPtr(PLANAR_Y),
                                  srcFrame->GetPitch(PLANAR_Y), w, h, stage_);
            if (chroma_) {
                uploadPlane<uint16_t>(sU_, srcFrame->GetReadPtr(PLANAR_U),
                                      srcFrame->GetPitch(PLANAR_U), wUV, hUV, stage_);
                uploadPlane<uint16_t>(sV_, srcFrame->GetReadPtr(PLANAR_V),
                                      srcFrame->GetPitch(PLANAR_V), wUV, hUV, stage_);
            }
            buildAndDownload<uint16_t>(gY_, pY_, sY_, w, h);
            if (chroma_) {
                buildAndDownload<uint16_t>(gU_, pU_, sU_, wUV, hUV);
                buildAndDownload<uint16_t>(gU_, pV_, sV_, wUV, hUV);
            }
        }
        ctx.finish();

        // ---- download into a fresh super frame ----
        PVideoFrame dstFrame = env_->NewVideoFrame(vi);
        auto downloadPlane = [&](SvmBuffer& plane, int pl, int pw, int ph, int devPitch) {
            const size_t rowbytes = (size_t)pw * nPixelSize_;
            std::vector<uint8_t> tmp((size_t)devPitch * ph * nPixelSize_);
            plane.read(tmp.data(), 0, tmp.size());
            uint8_t* dp = dstFrame->GetWritePtr(pl);
            int dpitch = dstFrame->GetPitch(pl);
            for (int y = 0; y < ph; y++)
                memcpy(dp + (size_t)y * dpitch, tmp.data() + (size_t)y * devPitch * nPixelSize_,
                       rowbytes);
        };
        downloadPlane(pY_, PLANAR_Y, pitchY_, planeHY_, pitchY_);
        if (chroma_) {
            downloadPlane(pU_, PLANAR_U, pitchUV_, planeHUV_, pitchUV_);
            downloadPlane(pV_, PLANAR_V, pitchUV_, planeHUV_, pitchUV_);
        } else {
            // deviation 3: zero the unread U/V planes
            for (int pl : {PLANAR_U, PLANAR_V}) {
                uint8_t* dp = dstFrame->GetWritePtr(pl);
                int dpitch = dstFrame->GetPitch(pl);
                int ph = dstFrame->GetHeight(pl);
                int pw = dstFrame->GetRowSize(pl);
                std::memset(dp, 0, (size_t)dpitch * ph);
                (void)pw;
            }
        }
        return dstFrame;
    }
};

AVSValue __cdecl CreateKMSuperOCL(AVSValue args, void*, IScriptEnvironment* env) {
    try {
        return new KMSuperOCL(
            args[0].AsClip(),
            args[1].AsInt(8),     // hpad
            args[2].AsInt(8),     // vpad
            args[3].AsInt(2),     // pel
            args[4].AsInt(0),     // levels
            args[5].AsBool(true), // chroma
            args[6].AsInt(2),     // sharp
            args[7].AsInt(2),     // filter
            env);
    } catch (const std::exception& e) {
        env->ThrowError("[KMSuperOCL] %s", e.what());
    }
    return AVSValue();
}

}  // namespace

// AviSynth+ linkage (no import lib): core passes the function table.
const AVS_Linkage* AVS_linkage{ nullptr };

extern "C" __declspec(dllexport) const char* __stdcall
AvisynthPluginInit3(IScriptEnvironment* env, const AVS_Linkage* const vectors) {
    AVS_linkage = vectors;
    try {
        // exact upstream signature (MV.cpp)
        env->AddFunction("KMSuperOCL", "c[hpad]i[vpad]i[pel]i[levels]i[chroma]b[sharp]i[filter]i",
                         CreateKMSuperOCL, 0);
        env->AddFunction("KMSuper", "c[hpad]i[vpad]i[pel]i[levels]i[chroma]b[sharp]i[filter]i",
                         CreateKMSuperOCL, 0);
    } catch (const std::exception& e) {
        return e.what();
    }
    return "KMSuper (OpenCL port of KTGMC/MV.cpp KMSuper, GPL; eighth host plugin / first motion-stage plugin)";
}
