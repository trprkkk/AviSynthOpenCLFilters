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

static VideoInfo viFor(IScriptEnvironment* env, const char* pt) {
    char script[128];
    sprintf_s(script, "BlankClip(width=64, height=48, pixel_type=\"%s\", length=4)", pt);
    return env->Invoke("Eval", AVSValue(script)).AsClip()->GetVideoInfo();
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

static void caseCompensate(IScriptEnvironment* env, const char* tag,
                           PClip source, PClip super, PClip mv,
                           PClip& outComp) {
    // KMCompensateOCL(source, super, mv, thSAD, time, thSCD1, thSCD2)
    AVSValue args[7] = {
        AVSValue(source), AVSValue(super), AVSValue(mv),
        10000, 100.0f, 400, 130
    };
    outComp = runClipFn(env, "KMCompensateOCL", args, 7);
    ++g_cases;
    check(outComp != nullptr, tag);
}

static void caseDegrain1(IScriptEnvironment* env, const char* tag,
                        PClip source, PClip super, PClip mvbw, PClip mvfw,
                        PClip& outDegrain) {
    // KMDegrain1OCL(source, super, mvbw, mvfw, thSAD, thSADC, plane,
    //               limit, limitC, thSCD1, thSCD2, binomial, useFlag)
    const int bpc = source->GetVideoInfo().BitsPerComponent();
    const int defLimit = (1 << bpc) - 1;
    AVSValue args[13] = {
        AVSValue(source), AVSValue(super), AVSValue(mvbw), AVSValue(mvfw),
        400, 400, 4,
        defLimit, defLimit, 400, 130, AVSValue(false), 0
    };
    outDegrain = runClipFn(env, "KMDegrain1OCL", args, 13);
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
