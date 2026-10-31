# Port plan — KTGMC (and beyond) from CUDA → OpenCL

This document is the engineering map for porting
[`rigaya/AviSynthCUDAFilters`](https://github.com/rigaya/AviSynthCUDAFilters)
(→ `nekopanda/AviSynthCUDAFilters`) to OpenCL. It records the verified source
inventory, the chosen strategy, per-kernel status, and how to extend the repo.

## 1. Source inventory (verified from `master`)

KTGMC is split into three CUDA files plus AVS glue:

| File | Lines | Contents |
|---|---|---|
| `KTGMC/Kernel.cu` | ~3560 | **27** `__global__` kernels: the per-plane/no-MV pixel kernels **and** all the AviSynth `KTGMC_*`/`K*` filter classes (`KTGMC_Bob`, `KBinomialTemporalSoften`, `KRemoveGrain`, `KRepair`, `KVerticalCleaner`, `KGaussResize`, masktools-style `KMakeDiff/KAddDiff/KLogic/KMerge`, `KXpand/Expand`, bob shimmer fixes, VResharpen/Resharpen, LimitOverSharpen, ToFullRange, TweakSearchClip, LosslessProc, ErrorAdjust, `KDoubleWeave/KWeave`, `KCopy`, and the `ResamplingFunction`/`ResamplingProgram` framework). Registers 25 AVS functions. |
| `KTGMC/MVKernel.cu` | ~3540 | **44** device functions/kernels for motion estimation: `KMSuper` (super-sampled pyramid), `KMAnalyse` (SAD, block search, MV vector prediction/refinement, scene-change), and `KMCompensate` motion compensation. |
| `KTGMC/MV.cpp` | ~5550 | Host pipeline + `KMSuper`/`KMAnalyse`/`KMDegrain1/2`/`KMCompensate` classes + AVS registration. This is a port of AviSynth `mvtools` to CUDA. |

Other suite projects (next milestones after KTGMC): `KNNEDI3` (neural-net
upscaler), `KFM` (KDeband, Deblock, CombingAnalyze, DecombeUCF, MergeStatic…),
`AvsCUDA` (AviSynthNeo CUDA plumbing), `GRunT` (no CUDA upstream — vendored
verbatim at `third_party/grunt/`), `masktools` (**fully ported, 5/5** — see
`docs/MASKTOOLS_PORT_SPEC.md`).

## 2. Strategy

1. **Per-pixel separable kernels first.** Most `Kernel.cu` kernels process a
   plane pixel-by-pixel (or 4 pixels per CUDA thread via `uchar4`/`ushort4`).
   Each output element is an *independent* dot/point/sort operation that never
   reads a sibling channel, so a **scalar translation is bit-identical**. This is
   exactly what Milestone 1 does. Vectorization is a later, pure-performance pass.
2. **Double implementation for correctness.** Every kernel exists as (a) a scalar
   OpenCL `.cl` and (b) a scalar CPU mirror. An independent Python golden is the
   third, cross-language check. `make test` runs CPU-vs-Python bit-for-bit.
3. **Reuse QSVEnc's OpenCL idioms.** `rigaya/QSVEnc/QSVPipeline/rgy_filter_*.cl`
   (e.g. the OpenCL `--vpp-kfm`, `--vpp-degrain` filters) show the house style:
   `rgy_CL*` buffer/host helpers, `#pragma` / build-option handling, and the
   kernel-source-as-string approach. Mirror that for a drop-in feel.
4. **Host glue separated from kernels.** The AviSynth host adaptation and the
   motion-vector pipeline are ported separately from the pixel kernels, so each
   milestone is independently testable without AviSynth.

### Precision rules preserved from CUDA (do not "fix")
- Resample accumulates in `float` (CUDA) — scalar CPU reference uses `double`
  only because that is the *algorithm*; the `.cl` resampler uses `float` to match
  real CUDA bit behaviour. All other kernels are integer-exact.
- Rounding is always `clamp(x, 0, maxval)` then `+0.5` before truncation
  (`(int)(x + 0.5)`), matching CUDA `cast_to(x + 0.5f)`.
- Pixel range: `maxval = 255` (8-bit) or `65535` (16-bit); plane `ComponentSize`
  selects the kernel instantiation, exactly like the `switch (pixelSize)` in the
  CUDA `Proc()` templates.
- Fixed point: `KMerge` uses `(w*32767.0f)` → `32767` scale, `>>15`.
- `MakeDiff`: `a-b+range_half`, `range_half = 1 << (bits-1)`.

## 3. Milestone status map (Kernel.cu simple kernels)

`src/opencl/ktgmc/kernels/ktgmc_simple.cl` currently contains (✔ = validated
bit-for-bit via `make test`, both 8-bit and 16-bit):

| CUDA kernel (Kernel.cu) | OpenCL | Status |
|---|---|---|
| `kl_resample_v` + `ResamplingFunction`/`Program` | `kt_resample_v` | ✔ |
| `kl_resample_h` | `kt_resample_h` | ✔ |
| `kl_makediff` (`MakeDiffOp`) | `kt_makediff mode 0` | ✔ |
| `kl_makediff` (`AddDiffOp`) | `kt_makediff mode 1` | ✔ |
| `kl_box3x3_filter` RG11/RG20 | `kt_rg_box3x3` | ✔ |
| `kl_rg_clip` N=1..4 | `kt_removegrain_clip` | ✔ |
| `kl_repair_clip` N=1..4 | `kt_repair_clip` | ✔ |
| `kl_vertical_cleaner_median` | `kt_vertical_cleaner_median` | ✔ |
| `kl_box5_v_and_border` (Min5/Max5) | `kt_box5_minmax` | ✔ |
| `kl_logic2` (LogicMin/Max) | `kt_logic_minmax` | ✔ |
| `kl_box3_v` (Resharpen) | `kt_vresharpen` | ✔ |
| `kl_resharpen` | `kt_resharpen` | ✔ |
| `kl_limit_over_sharpen` | `kt_limit_over_sharpen` | ✔ |
| `kl_to_full_range` (Y / UV) | `kt_to_full_range` | ✔ |
| `kl_bobshimmerfixes_merge` | `kt_bobshimmerfixes_merge` | ✔ |
| `kl_tweak_search_clip` | `kt_tweak_search_clip` | ✔ |
| `kl_error_adjust` | `kt_error_adjust` | ✔ |
| `kl_lossless_proc` | `kt_lossless_proc` | ✔ |
| `kl_merge` | `kt_merge` | ✔ |
| `kl_binomial_temporal_soften_1` | `kt_temporal_soften_1` | ✔¹ |
| `kl_binomial_temporal_soften_2` | `kt_temporal_soften_2` | ✔¹ |
| `kl_weave` (KDoubleWeave) | `kt_weave` | ✔ |
| `kl_copy` / `kl_elementwise Copy` | `kt_copy` | ✔ |
| `kl_vertical_wiener` (MVKernel.cu, KMSuper "sharp") | `kt_vertical_wiener` | ✔ |
| `kl_horizontal_wiener` (MVKernel.cu, KMSuper "sharp") | `kt_horizontal_wiener` | ✔ |
| `kl_calculate_sad` frame-level (Kernel.cu temporal soften) | `kt_plane_sad` | ✔² |
| `kl_init_sad` (temporal soften SAD zeroing, `<<<1, radius*2*3>>>`) | `kt_init_sad` | ✔ |

² `kt_plane_sad` is the frame/plane absolute-diff sum (scene-change gate). The
CUDA block/warp reductions are intrinsic/layout-specific; the *metric* is the
scalar `sum |a-b|` which is validated here. Wiring it into the temporal-soften
`scN` scene flags is host logic shown in the kernel comment.

¹ Scene-change replacement (the CUDA shared-memory `isSC[]` reduction) is exposed
as scalar per-frame flags `scN` here; wiring the full SAD-based
`kl_calculate_sad` reduction is part of the motion stage.

Motion / super-sampling kernels now live in `src/opencl/ktgmc/kernels/ktgmc_motion.cl`
(see `docs/MV_PORT_SPEC.md`). ALG-VERIFIED: the degrain weight helpers
(`dev_degrain_weight`, `dev_norm_weights`), the MV-aux integer kernels
(`kl_write_default_mv`, `kl_scene_change`/`_x2`, `kl_short_to_byte`,
`kl_short_to_byte_or_copy_src`), the coarse→fine MV upsampler
`kl_interpolate_prediction`, the global-MV refinement `kl_mean_global_mv`, and
the per-block search setup `kl_prepare_search` (ANALYZE_SYNC=1), the MV
I/O quartet `kl_load_mv`/`kl_store_mv`/`kl_load_mv_batch`/`kl_init_const_vec`,
and the reduced-plane
builder `kl_RB2B_bilinear_filtered` (separable 1:2 downsample) and its CUDA-only
fused twin `kl_RB2B_bilinear_filtered_with_pad` (single 4×4-tap +32/64 pass that
also fills the hpad/vpad border, MV.cpp `ReduceToPad`) are both ALG-VERIFIED
(`run_mv_rb2b.py`, `run_mv_rb2b_pad.py`).
Source-ported (RIG-VERIFY, device run pending):
`kl_copy_pad`, `kl_pad_frame_h/v`, `kl_init_scene_change`, the block-search
pure helpers `dev_clip_mv`/`dev_check_mv`/`dev_sq_norm`/`dev_get_ref_block`,
and the first block-level kernel `kl_calc_all_sad` (per-block SAD vs the
MV-selected ref block; host model in `docs/BLOCKSEARCH_MODEL.md`).
`kl_most_freq_mv` is ALG-VERIFIED (staged-fold tie-break transcription,
CUDA-twin 239/239 tie axes, `rig/cuda_mv_tie_test.py`); the prepare pair
`kl_prepare_degrain`/`kl_prepare_compensate` graduated from the deleted
`ktgmc_degrain_rig.cl` into `ktgmc_motion.cl` byte-identical (`// RIG-VERIFY`,
run_mv_prepare pin retained).

TODO (not yet ported, device-bound — see `docs/CODEX_HANDOFF.md`):
~~the block-search driver kernels (`Search`, expanding/hex2, `dev_read_pixels`,
`dev_calc_sad`, `MinCost`, `dev_reduce_result`)~~ — **DONE 2026-10-28**: ported
as `kt_search_column` in `src/opencl/ktgmc/kernels/ktgmc_search.cl`
(**ALG-VERIFIED** on device, `rig/dev_ktgmc_search.py`; golden
`python/run_mv_search.py` 81 cases; CPU mirror `sim/ktgmc_search_ref.cpp`;
`docs/BLOCKSEARCH_MODEL.md` §8b/§8c resolved via host-side column
serialisation) — and the degrain / compensate
host wiring (`kl_prepare_degrain`, `kl_degrain_2x3`, `kl_prepare_compensate`,
`kl_compensate_2x3`), plus the MV.cpp host state machine.  Their *integer math*
is now covered: the ALG-VERIFIED `kt_degrain_patch` + `kt_overlap_out`
combiner reproduces the overlap path, and only the MV.cpp `KMPlane`
`nPel`/padding pel-model + `(nPatternX,nPatternY,M)` host launch geometry block
them (see `docs/MV_PORT_SPEC.md` §6.1/§6.2 and `docs/BLOCKSEARCH_MODEL.md` §8).

The `Kernel.cu` no-motion AVS filter-function kernels are all covered by
`ktgmc_simple.cl` already. Of the extra names there, `kl_logic1`/`kl_logic3`
(KLogic1/KLogic3) are never instantiated, `kl_copy_boarder1` has no caller and `kl_copy_boarder1_v` is inside `#if 0`
(both dead, no port), `kl_calculate_sad` (the temporal-soften scene-change
SAD) maps to the scalar `kt_plane_sad` plus a host-side threshold compare,
and `kl_init_sad` is ported 1:1 as `kt_init_sad` (ALG-VERIFIED) — no further
kernels needed here.

`GaussianFilter` (KGaussResize) needed no new `.cl`: upstream launches the
same `kl_resample_h/v` templates at `filter_size` 8/9, which `kt_resample_h/v`
already cover with a runtime `filter_size`. What was added is the gaussian
program builder (`build_gaussian_program` in `sim/ktgmc_cpu_ref.cpp` +
`python/run_validation.py`: `pow(2,-p·x²)`, support 4.0, `p` clamped to
[0.1,100], crop `+0.0001` ⇒ fir 9 / exact crop ⇒ fir 8) with fir-8/9
resample planes, int-offset + double-coef + float-coef-bit tables, and
clamp-probe cross-checks — all compared bit-exact by `run_validation.py`
(same-machine libm `pow`; the `.cl` float-accumulation path stays
device-side like the existing Mitchell resample).

## 4. Motion-compensation stages (the big remaining work)

To get a working deinterlacer you must port the mvtools-equivalent layers:

1. **Super sampling** (`KMSuper`, `MVKernel.cu`): separable upscale to 4× pel,
   levels pyramid. — **in progress**: the `kl_vertical_wiener`/`kl_horizontal_wiener`
   ("sharp") kernels are ported & validated; `ktgmc_motion.cl` adds the frame
   padding / mirror-copy kernels (source port, **RIG-VERIFY**) and both RB2B
   anti-aliased 1:2 downsamplers (`kt_rb2b_bilinear_filtered` separable and its
   `_with_pad` fused twin; ALG-VERIFIED). What remains device-bound is the
   sub-pel super-frame host buffer layout used by the Search kernels, whose
   semantics come from `MV.cpp`. Full data-model + verification plan:
   [`docs/MV_PORT_SPEC.md`](MV_PORT_SPEC.md).
2. **Analysis** (`KMAnalyse`, `kl_calculate_sad`, block search, temporal/vector
   prediction, scene-change detection, `dev_reduce` block reductions → OpenCL
   `barrier`/local reduce). — **frame-level SAD started** (`kt_plane_sad`,
   validated); the block-level SAD (`kt_calc_all_sad`) and the block *search*
   (`kt_search_column`, 2026-10-28) are now ported and **ALG-VERIFIED**;
   scene-change detection is ported (`kt_scene_change`/`_x2`).
3. **Compensation / Degrain** (`KMCompensate`, `kl_compensate_2x3`,
   `kl_degrain_2x3`, `kl_prepare_*`).
4. **Assembly** — the QTGMC AVS script wires individual `KTGMC_*`/`K*` filters
   (Bob → … → Repair/Resharpen → weave). The AviSynth graph lives on the host
   side and is device-independent once each leaf filter runs on OpenCL.

## 5. Beyond KTGMC

- **KNNEDI3**: self-contained neural-net 2× scaler; 6 `__global__` kernels
  (`docs/NNEDI3_PORT_SPEC.md`). Batch-1 done — the pad/copy helpers
  (`kn_pad_h/v`, `kn_copy`, `kn_pad_ref_and_copy_half`, 600 cases,
  batch-2 (`kn_prescreening`) and batch-3 (`kn_compute_nn`) all
  ALG-VERIFIED. **KNNEDI3 is fully ported (6/6)**; the 70 compute_nn
  template instantiations collapse into one runtime-parameterised kernel.
- **KFM**: MIT → cleanest to reuse/redistribute. **KDeband** (the smallest
  self-contained filter) was ported first, then **KEdgeLevel**,
  **KTemporalNR** and the **MergeStatic.cu** kernels. KDeband's
  `kf_deband_reduce_banding`, KEdgeLevel's four kernels (`kf_edgelevel`,
  `kf_edgelevel_repair`, `kf_el_to444`, `kf_el_from444`), KTemporalNR's
  `kf_temporal_nr`, and MergeStatic.cu's four (`kf_compare_frames`,
  `kf_min_frames`, `kf_and_coefs`, `kf_merge_static`) are ALG-VERIFIED in
  `src/opencl/kfm/kernels/kfm_deband.cl` / `kfm_edgelevel.cl` /
  `kfm_temporalnr.cl` / `kfm_mergestatic.cl` / `kfm_filterbase.cl`. MergeStatic:
  KTemporalDiff and KMergeStatic are kernel-complete (only AviSynth host glue
  remains), and KAnalyzeStatic's full kernel set is ported+verified too — the
  four KFMFilterBase coefficient kernels (`kf_calc_combe`, `kf_merge_uvcoefs`,
  `kf_extend_coef2`, `kf_apply_uvcoefs_420`, in `kfm_filterbase.cl`) plus
  `kf_min_frames`/`kf_and_coefs` (in `kfm_mergestatic.cl`); only the VPAD-pad
  host assembly of the KAnalyzeStatic pipeline is left (`// RIG-VERIFY`).
  The same file also hosts the shared mirror pads `kf_padv`/`kf_padh`
  (ALG-VERIFIED, solo + composed padv→padh), closing the KDeblock pad-kernel
  gap, the MergeBlock blender `kf_merge_block` (ALG-VERIFIED, making
  KPatchCombe/KFMSwitch kernel-complete with the KFMKernel.cu inventory),
  and five more CombingAnalyze/CompareFields helpers from the same file
  (`kf_average`, `kf_max`, `kf_merge_uvflags`, `kf_copy_border`,
  `kf_analyze_frame`, all ALG-VERIFIED), plus the padded-frame copies
  `kf_copy_pad`/`kf_copy_pad_2plane`, the ExtendBlocks ping-pong
  `kf_max_extend_blocks_h/v` and the plain plane utilities
  `kf_copy`/`kf_copy_2plane`/`kf_fill` (all ALG-VERIFIED) — `KFMFilterBase.cu` is now
  fully ported (KFMKernel.cu itself is inventoried host-only, no device
  kernels; see `docs/KFM_PORT_SPEC.md`).
  **KNoiseClip** (`kf_noise_clip`, from DecombeUCF.cu, in `kfm_noiseclip.cl`)
  is also ALG-VERIFIED — a self-contained 8-bit filter that is
  kernel-complete. **CombingAnalyze** (in `kfm_combinganalyze.cl`) is fully
  ported (15/15 device kernels, ALG-VERIFIED): the batch-1 stages
  (`kf_copy_first`, `kf_combe_to_flag`, `kf_sum_box3x3`, `kf_binary_flag`,
  `kf_bilinear_h/v`, `kf_temporal_soften`, `kf_remove_combe2`,
  `kf_clean_super`, `kf_contains_durty_block` + the 8-tap helpers) plus the
  KFMSuper block analyzer `kf_super_analyze` and the FMCount census
  `kf_init_fmcount`/`kf_count_cmflags`/`kf_count_cmflags_2planes`.
  The **DecombeUCF reductions** (`kf_init_uint64`,
  `kf_calculate_field_diff`, `kf_init_block_sum`, `kf_add_block_sum`,
  `kf_block_sum_max`, `kf_analyze_noise`, `kf_analyze_diff`, in
  `kfm_decombeucf.cl`) are ALG-VERIFIED too (810 cases, integer-exact),
  completing all 8 DecombeUCF.cu device kernels alongside `kf_noise_clip`.
  The **KDeblock core** `kf_deblock` (Deblock.cu `kl_deblock`,
  in `kfm_deblock.cl`) is ALG-VERIFIED too: the fixed float32 8×8 DCT/
  hard-threshold/IDCT deblocking stage, bit-exact vs an independent float32
  Python golden. Its QP-table builder `kf_make_qp_table` and `show==2`
  visualiser `kf_deblock_show` are ALG-VERIFIED as well (integer-exact;
  `python/run_kfm_deblock_qp.py`). Graduated from the rig file, the DC-mask
  `kf_max_vh/v/h`, ShowQP `kf_scale_qp`, sharpen-LUT `kf_sharpen_coeff` and
  the Bayer `kf_merge_deblock` (+`g_ldither`) are ALG-VERIFIED too
  (`python/run_kfm_deblock_aux.py`, 1593 cases, integer/float32-exact,
  incl. merge end-to-end + layout proofs and sharpen/show deterministic
  pins). The SharpenFilter pair
  `kf_sharpen` / `kf_show_sharpen_coeff` graduated 2026-10-28 into
  `kfm_deblock.cl` as `// ALG-VERIFIED` (mirror+golden + device run + CUDA
  texture-precision comparison; see `docs/RIG_HANDOFF_KDEBLOCK.md`, closed);
  all 11 Deblock.cu device kernels are transcribed, with only host
  sequencing remaining open.
  Full KFM map + next candidates (DecombeUCF host pipelines / Deblock
  helpers) are in `docs/KFM_PORT_SPEC.md`.
- **AvsCUDA** (`AvsCUDA/`, MIT): batch-1 done — the Merge planar core
  (`ka_merge`/`ka_merge_f32`/`ka_average`/`ka_average_f32`, in
  `avscuda_merge.cl`) and Invert (`ka_invert_plane_u8/u16/f32`,
  `ka_invert_rgb`, in `avscuda_filters.cl`), all ALG-VERIFIED (620+750
  cases), plus batch-2 ConvertBits (`ka_convert_lower_dither/nodither`,
  `ka_convert_higher`, `ka_convert_from/to_float`, 10 kernels in
  `avscuda_convert.cl`, 1320 cases), plus batch-3 Conditional metrics (15
  kernels in `avscuda_conditional.cl`, 1320 cases; the 2 float reductions
  are `// RIG-COMPARED` 2026-10-28 in `avscuda_conditional_rig.cl`), plus batch-4
  FilteredResizeH/V (8 kernels in `avscuda_resample.cl`, 1230 cases).
  AvsCUDA is now fully ported (18/18 templates); host plumbings
  (`Copy.cu`, `memcpy_kernel`) and the debug OSD were excluded by census.
  Rig host for the 2 float reductions: `src/host/run_avscuda_rig.cpp`;
  device-run handoff: `docs/RIG_HANDOFF_AVSCUDA_CONDITIONAL.md`; static
  perf pass over all families: `docs/PERF_NOTES.md`.
  See `docs/AVSCUDA_PORT_SPEC.md`.

## 6. AviSynth integration (device glue)

Real plugins need AviSynthNeo host code. That layer (frame `GetFrame`, plane
pitch/read/write pointers, `RegisterFunction`) is **separate** from the kernels
and must be written/run where AviSynthNeo exists (Windows, or Linux AviSynth+).
The kernels here are pure `plane → plane` operations so they can be unit-tested
with no AviSynth. When wiring up, follow the original classes' plane loop
(`PLANAR_Y/U/V`, `uvSamePitch` short-circuit, `logUVx/logUVy` subsampling).

2026-10-29: the first such plugin exists — `src/host/plugin/kfm_edgelevel_avs.cpp`
(KEdgeLevelOCL, loadable AviSynth+ 3.7.3 DLL, verified headless 79/79 cases;
see `src/host/plugin/build_host_edgelevel.bat` and the plugin header's
documented deviations). The second followed the same day —
`kfm_temporalnr_avs.cpp` (KTemporalNROCL, first temporal filter: stateless
windowed over the clamped frame window, 116/116 headless cases; the 16-bit
cases pinned that `kf_temporal_nr`'s `frame_stride` is an ELEMENT stride per
the goldens, not the byte stride its `.cl` comment claims — host fixed, kernel
untouched). The third, same day — `kfm_deband_avs.cpp` (KDebandOCL: one
`kf_deband_reduce_banding` launch per plane over one shared luma-dims XorShift
rand stream, matching upstream `CreateDebandRandom`; 141/141 headless cases;
one documented deviation — range > 127 throws instead of upstream's silent
`char` truncation). The fourth, same day — `kfm_mergestatic_avs.cpp`
(KTemporalDiffOCL + KMergeStaticOCL, the two script-assemblable filters of
upstream KFM/MergeStatic.cu, exact upstream signatures `"c"` / `"ccc"`;
stateless cross-frame semantics — a clamped 5-frame `max-min` window and the
60fps↔30fps `n>>1` coefficient blend; upstream KAnalyzeStatic stays a separate
filter, its two kernels already ported and verified; 63/63 headless cases;
documented deviations: `clamp(ref, 0, vi.num_frames)` upper bound fixed to
`num_frames-1`, redundant pre-merge CopyFrame skipped, CUDA device-match check
dropped). The fifth, same day — `kfm_noiseclip_avs.cpp` (KNoiseClipOCL: one
`kf_noise_clip` launch per plane, `dev_limitter((src-noise+256)>>1, nmin,
range)` against a same-resolution noise clip; 8-bit only upstream; the class
is stateless and launches exactly one kernel — 100% coverage; 170/170 headless
cases over YV12/YV16/YV24 source pairs incl. extremes 0/255; documented
deviations: upstream width/height %4 checks kept although the scalar kernel is
exact for any width, CUDA device-match check dropped). The sixth, same day —
`kfm_deblock_avs.cpp` (KDeblockOCL + `KDeblock` alias, the full KFM
Deblock.cu assembly: QPForDeblock (props/qpclip/KFM_SourceStart mapping,
force_qp constant table, `kf_make_qp_table` with DC blending), KDeblock
(`kf_deblock` parity-accumulator DCT/threshold/IDCT — dispatched as by=0 row
then by>=1 so the odd-bh parity-plane accumulator overlap resolves
deterministically like upstream's CPU loop order, where upstream CUDA itself
races — then `kf_merge_deblock`) and the SharpenFilter tail
(`kf_sharpen_coeff` + `kf_sharpen` behind the real core GaussResize);
mirror pads run as host staging; 137/137 headless cases over 8/16-bit x 6
formats incl. show=0..3; the three `run_kfm_deblock*` goldens PASS (their sim
builds force `-mfpmath=sse -msse2`: 32-bit x87 intermediates are not per-op
float32); documented deviations in the plugin header, incl. zero-filled pad
rows beyond h+16 — uninitialized (read) upstream).

The seventh, 2026-10-31 — `kfm_analyze_avs.cpp` (the remaining KFM analysis /
reduction filters of `CombingAnalyze.cu` + `DecombeUCF.cu` in one DLL:
`KFMSuperOCL`, `KCleanSuperOCL`, `KPreCycleAnalyzeOCL`,
`KPreCycleAnalyzeShowOCL`, `KContainsCombeOCL`, `KCombeMaskOCL`,
`KRemoveCombeOCL`, `KCFieldDiffOCL`, `KCFrameDiffDupOCL`, `KAnalyzeNoiseOCL`,
each with its exact upstream name as an alias/signature/defaults, over the
ALG-VERIFIED `kfm_combinganalyze.cl` (15/15) + `kfm_filterbase.cl` +
`kfm_decombeucf.cl` (8/8) kernels; the two `ConditionalFilter` helpers read
`current_frame` like upstream `CFunc`; stateless classes — per-frame data from
child/auxiliary clips, all device buffers per-call scratch; 65 headless cases /
106092 checks, `run_kfm_combinganalyze.py` 2731 cases + `run_kfm_decombeucf.py`
810 cases PASS; documented deviations: pair-packed uchar2 planes split into
`.x`/`.y` buffers, interior-origin pointers expressed as global-work offsets
over whole-buffer allocations (no sub-buffers on this driver), KCombeMask's
separable bilinear as the exact upstream CPU twin in host staging (its
pre-offset interior-origin pointers need a sub-buffer), KRemoveCombe runs
upstream's own `cpu_remove_combe2` twin, the DecombeUCF ±2-row pad ring). Not
assembled (documented with reasons in the plugin header): `KSwitchFlag` /
`KFMSuperShow`, the `KTelecine*`/`KPatchCombe`/`KFMSwitch`/`KFMPad`/
`KFMDecimate`/`AssumeDevice` KFMKernel.cu family (no ported `kfm_kernel.cl`),
the `KDecombUCF*` host decision machinery (no device kernels) and
`KAnalyzeStatic` (kernels ported; shifted-buffer chain formulation deferred).
No implemented class requires GRunT.

The eighth, 2026-10-30 — `ktgmc_super_avs.cpp` (KMSuperOCL, upstream `KMSuper`
alias, exact signature): the FIRST KTGMC motion-pipeline host plugin, building
the super-sampled pyramid frame over the ALG-VERIFIED `ktgmc_simple.cl` /
`ktgmc_motion.cl` kernels (pel-upscale wiener pair for sharp levels, separable/
fused RB2B 1:2 downsamplers for coarse levels, frame padding kernels, pyramid
cached per source frame). 33 headless cases / 31 frame checks, every plane of
the produced super frame vs an in-exe CPU reference (pel 1/2/4 x sharp on/off x
rfilter x chroma, P16 + 8-bit); a replication-border mutation fails every case.
Uses new SVM (`clSVMAlloc`) staging in `src/host/ocl_host.*` (interior
sub-buffer origins unreliable on this driver). Documented deviations: SVM
pyramid buffers, chroma=false planes zero-filled, tail columns zero-filled,
row-wise copy-back into the AVS pitch, CUDA-reflection border form. Its
successor KMAnalyse is inventoried but deliberately not assembled (the
multi-level `GroupOfPlanes` host loop needs an extension of the
`run_mv_search.py` one-Search-call assembly or a rig CUDA cross-check).

The ninth, 2026-11-01 — `ktgmc_analyse_avs.cpp` (KMAnalyseOCL, upstream
`KMAnalyse` alias, exact upstream signature; the SECOND motion-stage plugin,
chained directly behind KMSuperOCL as `KMSuperOCL(...).KMAnalyseOCL(...)`):
the proven multi-level `GroupOfPlanes` coarse→fine loop from
`cuda_mv/KMANALYSE_HOST_CONTRACT.md` §10 driven over the ALG-VERIFIED
`ktgmc_motion.cl` / `ktgmc_search.cl` kernels with NO new kernels — per level
i = nLevels-1…0: `kt_most_freq_mv`+`kt_mean_global_mv` (global-MV chain,
doubled) on the coarse level's post-search vectors, `kt_interpolate_prediction`
into the fine level, host-written sentinels, the one-Search assembly
(`kt_prepare_search` → ordered per-column `kt_search_column` →
`kt_calc_all_sad`), and the BGR32 VECTOR output frame assembled coarsest
level first. The SUPER_FRAME `KMVParam` is smuggled KMSuperOCL → KMAnalyseOCL
through the VideoInfo exactly like upstream (`ktgmc_kmparam.h`, common/KMV.h
SetParam), and the MV clip's vi carries the MV_FRAME params for downstream
KMCompensate/KMDegrain; frames carry the `KMVIsValid` prop. 18 headless cases
/ 15 VECTOR-frame checks / 0 failures via `test_ktgmc_analyse.cpp` (real
AviSynth+ 3.7.3; the reference is an in-exe CPU implementation of the proven
loop: verified CPU super planes + the `sim/ktgmc_search_ref.cpp` Search model
+ the global-MV/interp kernels transcribed from `ktgmc_motion.cl`; every
(x,y,sad) of every level bit-exact, 3-level 64x48 default + 2-level 32x24 +
blksize=16, pel 1/2, chroma on/off, HEX/EXHAUSTIVE fine, plevel, global=off,
backward, 16-bit, WriteDefault frame, alias + script-string + throws).
`rig/dev_ktgmc_analyse.py` (the CUDA-twin cross-check, 39/39) stays green.
Build with `build_host_analyse.bat`. Documented decisions D1-D10 in the
plugin header (measure report §5): CPU-defined coarsest globalMV (0,0),
intended WriteDefault {0,0,verybigSAD} (upstream CUDA `.x`-twice typo not
reproduced), generic level loop (2-level CUDA-twin-measured, 3-level
CPU-reference-covered), batch=1 only (port interp kernel has no batch dim),
16-bit supported, `partial` rejected (unmeasured), CUDA-dead params kept and
ignored, incoming-sads interp model (calc_all_sad writes only the outputs),
no batch cache, only the vector bytes of the BGR32 frame written.

## 7. Licensing

KTGMC-derived files follow the upstream **GPL** (per
`AviSynthCUDAFilters` README); KFM-derived files are **MIT**. Keep provenance
per file/directory. Nothing in this repo should be released without matching the
upstream license of the code it derives from.

## 8. Validation runner details

- `python/run_validation.py` generates deterministic raw planes (8 & 16 bit),
  computes the Python golden, builds `sim/ktgmc_cpu_ref.cpp`, runs it, and
  compares every output **and** the resampling-program `offset`/`coef` tables
  bit-for-bit.
- Add a new kernel by: implement it in `ktgmc_simple.cl`, mirror it in
  `sim/ktgmc_cpu_ref.cpp`, add the Python golden + a comparison entry.
