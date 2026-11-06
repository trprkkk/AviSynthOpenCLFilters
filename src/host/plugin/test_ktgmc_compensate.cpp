// test_ktgmc_compensate.cpp — headless verification of the KMCompensateOCL /
// KMDegrain1OCL / KMDegrain2OCL AviSynth+ plugin (tenth host plugin / third
// motion-stage plugin of the OpenCL port).
//
// Pattern matches test_ktgmc_super.cpp (verified working):
//   * LoadLibrary each plugin DLL directly + call AvisynthPluginInit3 (bypasses
//     AviSynth+ autoload caching which can skip DLLs),
//   * creates env via CreateScriptEnvironment,
//   * calls functions directly: env->Invoke(fnName, AVSValue(args, N))
//     wrapped in try/catch(AvisynthError) — NO script strings (those trigger
//     AviSynth's internal SEH E06D7363),
//   * uses C++ RandSource clips instead of script BlankClip.
//
// Build: see build_host_compensate.bat.  Usage:
//   test_ktgmc_compensate.exe [comp_dll] [an_dll] [sp_dll]
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

static void check(bool cond, const char* msg) {
    g_cases++;
    if (!cond) {
        g_failures++;
        fprintf(stderr, "FAIL case %d: %s\n", g_cases, msg);
        fflush(stderr);
    }
}

// LoadLibrary + call init3 directly (avoids autoload caching)
static bool load_plugin_init(IScriptEnvironment* env, const char* path,
                             const char* name) {
    HMODULE h = LoadLibraryA(path);
    fprintf(stderr, "LoadLibrary %s h=%p gle=%lu\n", name, (void*)h,
            GetLastError());
    if (!h) { g_failures++; return false; }
    typedef const char* (__stdcall* Init3)(IScriptEnvironment*,
                                           const AVS_Linkage*);
    Init3 init3 = (Init3)GetProcAddress(h, "AvisynthPluginInit3");
    if (!init3) {
        fprintf(stderr, "%s: AvisynthPluginInit3 not exported\n", name);
        g_failures++; return false;
    }
    const char* res = nullptr;
    __try { res = init3(env, AVS_linkage); }
    __except (EXCEPTION_EXECUTE_HANDLER) {
        fprintf(stderr, "%s: init3 SEH exception\n", name);
        g_failures++; return false;
    }
    fprintf(stderr, "%s init3 -> %s\n", name, res ? res : "(null)");
    return true;
}

// run one plugin function directly with typed args
static PClip runClipFn(IScriptEnvironment* env, const char* fnName,
                       AVSValue* args, int nArgs) {
    try {
        AVSValue ret = env->Invoke(fnName, AVSValue(args, nArgs));
        if (!ret.IsClip()) {
            fprintf(stderr, "%s: Invoke did not return a clip\n", fnName);
            ++g_failures; ++g_cases;
            return nullptr;
        }
        return ret.AsClip();
    } catch (const AvisynthError& e) {
        fprintf(stderr, "%s: Invoke threw: %s\n", fnName, e.msg);
        ++g_failures; ++g_cases;
        return nullptr;
    }
}

// deterministic source clip (mirrors test_ktgmc_super.cpp RandSource)
class RandSource : public IClip {
    VideoInfo vi_;
    uint32_t salt_;
    int px_;
public:
    RandSource(VideoInfo vi, uint32_t salt) : vi_(vi), salt_(salt),
        px_(vi.ComponentSize()) {}
    PVideoFrame __stdcall GetFrame(int n, IScriptEnvironment* env) {
        PVideoFrame f = env->NewVideoFrame(vi_);
        const int w = vi_.width, h = vi_.height;
        const int wUV = w >> vi_.GetPlaneWidthSubsampling(PLANAR_U);
        const int hUV = h >> vi_.GetPlaneHeightSubsampling(PLANAR_U);
        const int maxv = (px_ == 1) ? 255 : 65535;
        uint32_t s = salt_ + (uint32_t)n * 7919u;
        auto fill = [&](int pl, int pw, int ph, uint32_t salt2) {
            uint8_t* dst = f->GetWritePtr(pl);
            int pitch = f->GetPitch(pl);
            for (int y = 0; y < ph; ++y) {
                uint16_t* row16 = (uint16_t*)(dst + (size_t)y * pitch);
                uint8_t* row8 = dst + (size_t)y * pitch;
                for (int x = 0; x < pw; ++x) {
                    int v = (int)(((int64_t)x * (maxv - 40)) / (pw > 1 ? pw - 1 : 1)
                                  + ((int64_t)y * (maxv - 40)) / (ph > 1 ? ph - 1 : 1)
                                  + 20);
                    v += (int)(salt2 % 9u) - 4;
                    salt2 = salt2 * 1103515245u + 12345u;
                    if (v < 0) v = 0;
                    if (v > maxv) v = maxv;
                    if (px_ == 1) row8[x] = (uint8_t)v;
                    else          row16[x] = (uint16_t)v;
                }
            }
        };
        fill(PLANAR_Y, w, h, s);
        fill(PLANAR_U, wUV, hUV, s + 11u);
        fill(PLANAR_V, wUV, hUV, s + 23u);
        return f;
    }
    bool __stdcall GetParity(int) { return false; }
    void __stdcall GetAudio(void*, int64_t, int64_t, IScriptEnvironment*) {}
    int __stdcall SetCacheHints(int, int) { return 0; }
    const VideoInfo& __stdcall GetVideoInfo() { return vi_; }
};

// Deterministic translated pattern: frame n is the base pattern shifted by
// (4*n, 2*n) with a small per-frame brightness drift.  The drift keeps every
// block's residual SAD nonzero (so thSAD=1 forces the zero-weight/source path)
// while staying far below the scene-change threshold.  Chroma is constant.
class ShiftSource : public IClip {
    VideoInfo vi_;
    int stepX_, stepY_;
public:
    explicit ShiftSource(VideoInfo vi, int stepX = 4, int stepY = 2)
        : vi_(vi), stepX_(stepX), stepY_(stepY) {}
    static uint32_t hash(uint32_t x, uint32_t y) {
        uint32_t h = x * 374761393u + y * 668265263u;
        h = (h ^ (h >> 13)) * 1274126177u;
        return h ^ (h >> 16);
    }
    PVideoFrame __stdcall GetFrame(int n, IScriptEnvironment* env) {
        PVideoFrame f = env->NewVideoFrame(vi_);
        const int w = vi_.width, h = vi_.height;
        int shx = (stepX_ * n) % w, shy = (stepY_ * n) % h;
        uint8_t* dy = f->GetWritePtr(PLANAR_Y);
        int py = f->GetPitch(PLANAR_Y);
        for (int y = 0; y < h; ++y) {
            int sy = (y + shy) % h;
            for (int x = 0; x < w; ++x) {
                int sx = (x + shx) % w;
                dy[(size_t)y * py + x] =
                    (uint8_t)((hash((uint32_t)sx, (uint32_t)sy) % 200u) + n);
            }
        }
        for (int pl : {PLANAR_U, PLANAR_V}) {
            const int pw = w >> vi_.GetPlaneWidthSubsampling(PLANAR_U);
            const int ph = h >> vi_.GetPlaneHeightSubsampling(PLANAR_U);
            uint8_t* d = f->GetWritePtr(pl);
            int p = f->GetPitch(pl);
            for (int y = 0; y < ph; ++y)
                memset(d + (size_t)y * p, 128, (size_t)pw);
        }
        return f;
    }
    bool __stdcall GetParity(int) { return false; }
    void __stdcall GetAudio(void*, int64_t, int64_t, IScriptEnvironment*) {}
    int __stdcall SetCacheHints(int, int) { return 0; }
    const VideoInfo& __stdcall GetVideoInfo() { return vi_; }
};

static VideoInfo viFor(IScriptEnvironment* env, const char* pt) {
    char script[128];
    sprintf_s(script, "BlankClip(width=64, height=48, pixel_type=\"%s\", length=4)", pt);
    return env->Invoke("Eval", AVSValue(script)).AsClip()->GetVideoInfo();
}

// GetFrame wrapper that reports AviSynth errors instead of aborting the exe
static PVideoFrame getFrameSafe(PClip clip, int n, IScriptEnvironment* env,
                                const char* tag) {
    if (!clip) return PVideoFrame();
    try {
        return clip->GetFrame(n, env);
    } catch (const AvisynthError& e) {
        fprintf(stderr, "%s: GetFrame(%d) threw: %s\n", tag, n, e.msg);
        ++g_failures; ++g_cases;
        return PVideoFrame();
    } catch (const std::exception& e) {
        fprintf(stderr, "%s: GetFrame(%d) std::exception: %s\n", tag, n, e.what());
        ++g_failures; ++g_cases;
        return PVideoFrame();
    } catch (...) {
        fprintf(stderr, "%s: GetFrame(%d) unknown C++ exception\n", tag, n);
        ++g_failures; ++g_cases;
        return PVideoFrame();
    }
}

// max |a-b| over all planes (8/16-bit)
static int frameMaxDiff(PVideoFrame a, PVideoFrame b, const VideoInfo& vi) {
    int maxd = 0;
    for (int planeNo = 0; planeNo < vi.NumComponents(); ++planeNo) {
        const int pl = 1 << planeNo;   // PLANAR_Y/U/V are 1/2/4, not 0-based
        const int pw = vi.width >> vi.GetPlaneWidthSubsampling(pl);
        const int ph = vi.height >> vi.GetPlaneHeightSubsampling(pl);
        const uint8_t* pa = a->GetReadPtr(pl);
        const uint8_t* pb = b->GetReadPtr(pl);
        const int pia = a->GetPitch(pl), pib = b->GetPitch(pl);
        for (int y = 0; y < ph; ++y) {
            if (vi.ComponentSize() == 1) {
                const uint8_t* ra = pa + (size_t)y * pia;
                const uint8_t* rb = pb + (size_t)y * pib;
                for (int x = 0; x < pw; ++x) {
                    int d = (int)ra[x] - (int)rb[x];
                    if (d < 0) d = -d;
                    if (d > maxd) maxd = d;
                }
            } else {
                const uint16_t* ra = (const uint16_t*)(pa + (size_t)y * pia);
                const uint16_t* rb = (const uint16_t*)(pb + (size_t)y * pib);
                for (int x = 0; x < pw; ++x) {
                    int d = (int)ra[x] - (int)rb[x];
                    if (d < 0) d = -d;
                    if (d > maxd) maxd = d;
                }
            }
        }
    }
    return maxd;
}

static int diffFrames(PClip a, int na, PClip b, int nb, IScriptEnvironment* env,
                      const VideoInfo& vi, const char* tag) {
    PVideoFrame fa = getFrameSafe(a, na, env, tag);
    PVideoFrame fb = getFrameSafe(b, nb, env, tag);
    if (!fa || !fb) return -1;
    return frameMaxDiff(fa, fb, vi);
}

static void caseSuperAnalyse(IScriptEnvironment* env, const char* tag,
                             PClip srcClip, PClip& outSuper, PClip& outMV,
                             const char* superFn, const char* analyseFn) {
    // KMSuperOCL(source, hpad, vpad, pel, levels, chroma, sharp, filter)
    AVSValue superArgs[8] = {
        AVSValue(srcClip), 8, 8, 2, 0, AVSValue(true), 2, 2
    };
    fprintf(stderr, "STEP superInvoke %s\n", superFn); fflush(stderr);
    outSuper = runClipFn(env, superFn, superArgs, 8);
    fprintf(stderr, "STEP superDone clip=%p\n", (void*)outSuper); fflush(stderr);
    ++g_cases;

    // KMAnalyseOCL(super, blksize, overlap, search, isb, chroma, delta,
    //              lambda, lsad, global, plevel, pnew, meander, partial, batch)
    AVSValue analyseArgs[15] = {
        AVSValue(outSuper), 16, 8, 4, AVSValue(false), AVSValue(true), 1,
        400, 400, AVSValue(true), 2, 25, AVSValue(true), AVSValue(), 1
    };
    fprintf(stderr, "STEP analyseInvoke %s\n", analyseFn); fflush(stderr);
    outMV = runClipFn(env, analyseFn, analyseArgs, 15);
    fprintf(stderr, "STEP analyseDone clip=%p\n", (void*)outMV); fflush(stderr);
    ++g_cases;
    check(outMV != nullptr, tag);
}

// create another MV clip from an existing super clip (same geometry args)
static PClip analyseFromSuper(IScriptEnvironment* env, const char* tag,
                              PClip super, bool isb, int delta,
                              const char* analyseFn) {
    AVSValue args[15] = {
        AVSValue(super), 16, 8, 4, AVSValue(isb), AVSValue(true), delta,
        400, 400, AVSValue(true), 2, 25, AVSValue(true), AVSValue(), 1
    };
    PClip mv = runClipFn(env, analyseFn, args, 15);
    ++g_cases;
    check(mv != nullptr, tag);
    return mv;
}

static void caseCompensateEx(IScriptEnvironment* env, const char* tag,
                             PClip source, PClip super, PClip mv,
                             int thSAD, double time, int thSCD1, int thSCD2,
                             PClip& outComp) {
    // KMCompensateOCL(source, super, mv, thSAD, time, thSCD1, thSCD2)
    AVSValue args[7] = {
        AVSValue(source), AVSValue(super), AVSValue(mv),
        thSAD, (float)time, thSCD1, thSCD2
    };
    outComp = runClipFn(env, "KMCompensateOCL", args, 7);
    ++g_cases;
    check(outComp != nullptr, tag);
}

static void caseCompensate(IScriptEnvironment* env, const char* tag,
                           PClip source, PClip super, PClip mv,
                           PClip& outComp) {
    caseCompensateEx(env, tag, source, super, mv, 10000, 100.0, 400, 130,
                     outComp);
}

static void caseDegrain1Ex(IScriptEnvironment* env, const char* tag,
                           PClip source, PClip super, PClip mvbw, PClip mvfw,
                           int thSAD, int thSADC, PClip& outDegrain) {
    // KMDegrain1OCL(source, super, mvbw, mvfw, thSAD, thSADC, plane,
    //               limit, limitC, thSCD1, thSCD2, binomial, useFlag)
    const int bpc = source->GetVideoInfo().BitsPerComponent();
    const int defLimit = (1 << bpc) - 1;
    AVSValue args[13] = {
        AVSValue(source), AVSValue(super), AVSValue(mvbw), AVSValue(mvfw),
        thSAD, thSADC, 4,
        defLimit, defLimit, 400, 130, AVSValue(false), 0
    };
    outDegrain = runClipFn(env, "KMDegrain1OCL", args, 13);
    ++g_cases;
    check(outDegrain != nullptr, tag);
}

static void caseDegrain1(IScriptEnvironment* env, const char* tag,
                        PClip source, PClip super, PClip mvbw, PClip mvfw,
                        PClip& outDegrain) {
    caseDegrain1Ex(env, tag, source, super, mvbw, mvfw, 400, 400, outDegrain);
}

static void caseDegrain2(IScriptEnvironment* env, const char* tag,
                         PClip source, PClip super,
                         PClip mvbw, PClip mvfw, PClip mvbw2, PClip mvfw2,
                         int thSAD, int thSADC, PClip& outDegrain) {
    // KMDegrain2OCL(source, super, mvbw, mvfw, mvbw2, mvfw2, thSAD, thSADC,
    //               plane, limit, limitC, thSCD1, thSCD2, binomial, useFlag)
    const int bpc = source->GetVideoInfo().BitsPerComponent();
    const int defLimit = (1 << bpc) - 1;
    AVSValue args[15] = {
        AVSValue(source), AVSValue(super), AVSValue(mvbw), AVSValue(mvfw),
        AVSValue(mvbw2), AVSValue(mvfw2), thSAD, thSADC, 4,
        defLimit, defLimit, 400, 130, AVSValue(false), 0
    };
    outDegrain = runClipFn(env, "KMDegrain2OCL", args, 15);
    ++g_cases;
    check(outDegrain != nullptr, tag);
}

int main(int argc, char** argv) {
    setvbuf(stderr, nullptr, _IONBF, 0);
    AddVectoredExceptionHandler(1, eh);

    const char* comp_dll = argc > 1 ? argv[1]
        : "Q:\\Downloads\\openclfilter\\repo\\src\\host\\plugin\\build_host\\ktgmc_compensate_avs.dll";
    const char* an_dll  = argc > 2 ? argv[2]
        : "Q:\\Downloads\\openclfilter\\repo\\src\\host\\plugin\\build_host\\ktgmc_analyse_avs.dll";
    const char* sp_dll  = argc > 3 ? argv[3]
        : "Q:\\Downloads\\openclfilter\\repo\\src\\host\\plugin\\build_host\\ktgmc_super_avs.dll";

    HMODULE dll = LoadLibraryA("AviSynth.dll");
    if (!dll) { fprintf(stderr, "cannot load AviSynth.dll (%lu)\n", GetLastError()); return 1; }
    FARPROC pv = GetProcAddress(dll, "?AVS_linkage@@3QEBUAVS_Linkage@@EB");
    if (!pv) { fprintf(stderr, "AVS_linkage export not found\n"); return 1; }
    AVS_linkage = *(const AVS_Linkage**)pv;
    CSE_t cse = (CSE_t)GetProcAddress(dll, "CreateScriptEnvironment");
    if (!cse) { fprintf(stderr, "CreateScriptEnvironment not exported\n"); return 1; }
    IScriptEnvironment* env = cse(AVISYNTH_INTERFACE_VERSION);
    if (!env) { fprintf(stderr, "env creation failed\n"); return 1; }

    // init3 each plugin directly (bypasses autoload caching)
    // load in natural order: super, analyse, compensate
    if (!load_plugin_init(env, sp_dll,  "ktgmc_super_avs"))    return 1;
    if (!load_plugin_init(env, an_dll,  "ktgmc_analyse_avs"))  return 1;
    if (!load_plugin_init(env, comp_dll, "ktgmc_compensate_avs")) return 1;

    fprintf(stderr, "FunctionExists: KMSuperOCL=%d KMAnalyseOCL=%d KMCompensateOCL=%d KMDegrain1OCL=%d KMDegrain2OCL=%d\n",
        env->FunctionExists("KMSuperOCL"), env->FunctionExists("KMAnalyseOCL"),
        env->FunctionExists("KMCompensateOCL"), env->FunctionExists("KMDegrain1OCL"),
        env->FunctionExists("KMDegrain2OCL")); fflush(stderr);

    // ---- YV12 8-bit sweep: super + analyse + compensate + degrain1 ----
    {
        VideoInfo vi = viFor(env, "YV12");
        PClip srcClip(new RandSource(vi, 0x5EEDu));
        PClip super = nullptr, mv = nullptr, comp = nullptr, degrain1 = nullptr;
        caseSuperAnalyse(env, "YV12 super+analyse", srcClip, super, mv,
                         "KMSuperOCL", "KMAnalyseOCL");
        caseCompensate(env, "YV12 KMCompensateOCL", srcClip, super, mv, comp);
        caseDegrain1(env, "YV12 KMDegrain1OCL", srcClip, super, mv, mv, degrain1);

        if (comp) {
            PVideoFrame f = comp->GetFrame(0, env);
            check(f != nullptr, "compensate frame 0 readable");
        }
        if (degrain1) {
            PVideoFrame f = degrain1->GetFrame(0, env);
            check(f != nullptr, "degrain1 frame 0 readable");
        }
    }

    // ---- KMCompensate alias (KMCompensate) ----
    {
        VideoInfo vi = viFor(env, "YV12");
        PClip srcClip(new RandSource(vi, 0xABCDu));
        PClip super = nullptr, mv = nullptr, comp = nullptr;
        caseSuperAnalyse(env, "YV12 super+analyse (alias)", srcClip, super, mv,
                         "KMSuper", "KMAnalyse");
        caseCompensate(env, "YV12 KMCompensate alias", srcClip, super, mv, comp);
    }

    // ---- host-seam regression sweep (deterministic translated pattern) ----
    // These cases exercise the code paths the smoke test above cannot see:
    //   A  invalid MV frame -> child frame returned untouched (MV.cpp:5176)
    //   A2 frame-level scene change -> source copy
    //   B  blocks with sad >= thSAD reference the CURRENT super frame (pRef0)
    //   C  KMDegrain1 with all weights 0 -> exact source (source pitch seam)
    //   D  KMDegrain2 delta=2 slice independence (stageMv per-slice readback)
    {
        VideoInfo vi = viFor(env, "YV12");
        PClip srcClip(new ShiftSource(vi));
        PClip super = nullptr, mvNear = nullptr, mvFar = nullptr;
        caseSuperAnalyse(env, "shift super+analyse", srcClip, super, mvNear,
                         "KMSuperOCL", "KMAnalyseOCL");
        mvFar = analyseFromSuper(env, "shift analyse delta=2", super, false, 2,
                                 "KMAnalyseOCL");
        check(super && mvNear && mvFar, "shift regression clips");

        if (super && mvNear) {
            // A. frame 0 has refIndex -1 and KMVIsValid=0
            PClip comp = nullptr;
            caseCompensateEx(env, "invalid MV compensate", srcClip, super,
                             mvNear, 10000, 100.0, 400, 130, comp);
            if (comp) {
                int d = diffFrames(comp, 0, srcClip, 0, env, vi, "invalid MV");
                check(d == 0, "compensate invalid MV frame 0 returns source");
            }

            // A2. tiny thSCD1 makes every block count as scene-changed
            PClip compSC = nullptr;
            caseCompensateEx(env, "scene-change compensate", srcClip, super,
                             mvNear, 10000, 100.0, 1, 0, compSC);
            if (compSC) {
                int d = diffFrames(compSC, 2, srcClip, 2, env, vi, "scene-change");
                check(d == 0, "compensate scene-change frame returns source");
            }

            // B. thSAD=1 (scaled to 6) selects the static pRef0 plane: the
            //    current super frame at zero motion reproduces the source.
            PClip compStatic = nullptr;
            caseCompensateEx(env, "static-ref compensate", srcClip, super,
                             mvNear, 1, 100.0, 400, 130, compStatic);
            if (compStatic) {
                int d = diffFrames(compStatic, 2, srcClip, 2, env, vi, "static-ref");
                check(d == 0, "compensate static-ref blocks reproduce source");
            }

            // C. thSAD=1 zeroes every ref weight: WSrc=256 -> exact source
            PClip dg1zero = nullptr;
            caseDegrain1Ex(env, "degrain1 zero-weight", srcClip, super,
                           mvNear, mvNear, 1, 1, dg1zero);
            if (dg1zero) {
                int d = diffFrames(dg1zero, 2, srcClip, 2, env, vi, "degrain1-zero");
                check(d == 0, "degrain1 all-zero weights reproduce source");
            }

            // D. the delta=2 (far) MV clip is unusable at frame 1 (ref -1),
            //    so KMDegrain2 must equal KMDegrain1 using the delta=1 clip.
            if (mvFar) {
                PClip dg1 = nullptr, dg2 = nullptr;
                caseDegrain1Ex(env, "degrain1 near", srcClip, super,
                               mvNear, mvNear, 400, 400, dg1);
                caseDegrain2(env, "degrain2 near+far", srcClip, super,
                             mvNear, mvNear, mvFar, mvFar, 400, 400, dg2);
                if (dg1 && dg2) {
                    int d = diffFrames(dg2, 1, dg1, 1, env, vi, "degrain2-slice");
                    check(d == 0, "degrain2 unusable far slice == degrain1");
                    int dsrc = diffFrames(dg2, 1, srcClip, 1, env, vi, "degrain2-slice");
                    check(dsrc != 0,
                          "degrain2 far-unusable frame is still blended");
                }
            }
        }
    }

    // ---- E. motion-reference slice (ref_sel==1) known-answer test ----
    // A static pattern with a per-frame brightness drift yields zero MVs and
    // small SADs, so every block takes the motion pRef path.  time=100 makes
    // the output exactly the reference frame (n-1); a host that ignored
    // ref_sel or mis-staged slice 1 would return the current frame instead.
    {
        VideoInfo vi = viFor(env, "YV12");
        PClip srcClip(new ShiftSource(vi, 0, 0));
        PClip super = nullptr, mv = nullptr, comp = nullptr;
        caseSuperAnalyse(env, "static-drift super+analyse", srcClip, super, mv,
                         "KMSuperOCL", "KMAnalyseOCL");
        caseCompensateEx(env, "motion-ref compensate", srcClip, super,
                         mv, 10000, 100.0, 400, 130, comp);
        if (comp) {
            int dRef = diffFrames(comp, 1, srcClip, 0, env, vi, "motion-ref");
            check(dRef == 0, "zero-motion compensate reads the reference frame");
            int dCur = diffFrames(comp, 1, srcClip, 1, env, vi, "motion-ref");
            check(dCur != 0, "zero-motion compensate is not the current frame");
        }
    }

    fprintf(stderr, "cases=%d failures=%d\n", g_cases, g_failures);
    if (g_failures == 0) {
        fprintf(stderr, "=== KMCompensate/KMDegrain smoke test: PASS ===\n");
        env->DeleteScriptEnvironment();
        return 0;
    }
    fprintf(stderr, "=== KMCompensate/KMDegrain smoke test: FAIL ===\n");
    env->DeleteScriptEnvironment();
    return 1;
}
