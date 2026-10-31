# opencl_filter

Vendor-agnostic **OpenCL** re-implementations of video filters originally written
for **NVIDIA CUDA** in [`rigaya/AviSynthCUDAFilters`](https://github.com/rigaya/AviSynthCUDAFilters)
(AviSynthNeo CUDA filters). The goal is to run the same algorithms on AMD /
Intel / Apple / CPU OpenCL devices — no NVIDIA GPU required.

The OpenCL porting style follows the proven OpenCL filters already shipped in
[`rigaya/QSVEnc`](https://github.com/rigaya/QSVEnc) (`QSVPipeline/rgy_filter_*`,
e.g. the `--vpp-kfm`/`--vpp-degrain` OpenCL pipelines): one `.cl` kernel set per
filter stage plus a small host wrapper that compiles the kernels, sets up
buffers and dispatches them, with a scalar CPU reference used for validation.

## Scope & status (honest picture)

`AviSynthCUDAFilters` is a **suite**, not one filter:

| Project | What it is | Port status here |
|---|---|---|
| **KTGMC** | QTGMC-style deinterlacer (motion compensated) | In progress — **Milestone 1 done** (see below); all motion kernels ported & device-verified, and the first motion host plugin KMSuperOCL ships (2026-10-30) |
| KNNEDI3 | NNEDI3 neural-net upscaler | **complete — all 6 device kernels ALG-VERIFIED** (`src/opencl/nnedi3/`, `docs/NNEDI3_PORT_SPEC.md`) |
| KFM | filter family (KDeband, Deblock, CombingAnalyze, …) | **KDeband/KEdgeLevel/KTemporalNR + MergeStatic/KNoiseClip + KDeblock + CombingAnalyze/DecombeUCF analysis done** (all 11 Deblock.cu device kernels ALG-VERIFIED, incl. the sharpen pair graduated 2026-10-28; the CombingAnalyze/DecombeUCF analysis kernels ALG-VERIFIED, seventh host plugin 2026-10-31) (`src/opencl/kfm/kernels/`) |
| masktools | mask/LUT helper suite | **complete — all 5 CUDA kernels ALG-VERIFIED** (`src/opencl/masktools/`, `docs/MASKTOOLS_PORT_SPEC.md`) |
| AvsCUDA | CUDA-aware dispatch + helpers | ported (18/18 families); 2 float reductions are rig-only (`docs/AVSCUDA_PORT_SPEC.md`) |
| GRunT | AviSynth runtime helper | **no CUDA to port** — vendored verbatim at `third_party/grunt/` |

A faithful KTGMC port is large: `KTGMC/Kernel.cu` (~3,560 lines) + `MVKernel.cu`
(~3,540) + `MV.cpp` (~5,550) implement the whole QTGMC algorithm including
multi-level motion estimation, block search and motion compensation. This repo
therefore progresses in **milestones**; each milestone's kernels are translated
faithfully and validated bit-for-bit against an independent reference.

### Milestone 1 (validated core)

A validated set of the *per-plane, no-motion* KTGMC kernels (the parts that need
no motion vectors), plus the host-side `ResamplingProgram`. All 22 kernels below
are present in `src/opencl/ktgmc/kernels/ktgmc_simple.cl` and cross-validated
bit-for-bit (8-bit and 16-bit) against independent C++ and Python references via
`make test`.

Motion super-sampling (stage 2, from `KTGMC/MVKernel.cu`, the `KMSuper`
"sharp" interpolation path):
- `kt_vertical_wiener` / `kt_horizontal_wiener` — 6-tap Wiener interpolation.

Resample / resize (host FIR `ResamplingProgram`, Mitchell/Catmull-Rom + more):
- `kt_resample_v` / `kt_resample_h` — the vertical & horizontal resamplers used
  by the "Bob" field-interpolation engine and the GaussResize-style path.

masktools-style combining:
- `kt_makediff` / add-diff (mt_makediff / mt_adddiff equivalent),
- `kt_logic_minmax` (KLogic min/max), `kt_merge` (32767 fixed point, KMerge).

Denoise / spatial:
- `kt_rg_box3x3` (RemoveGrain 11/12 & 20), `kt_removegrain_clip` (modes 1–4,
  8-neighbour sort network), `kt_repair_clip` (modes 1–4, 9-neighbour sort
  network), `kt_vertical_cleaner_median` (mode 1), `kt_box5_minmax`
  (Inpand/Expand ×2 vertical).

Bob/weave / sharper / misc:
- `kt_vresharpen`, `kt_resharpen`, `kt_limit_over_sharpen`,
  `kt_to_full_range` (Y & UV), `kt_bobshimmerfixes_merge`,
  `kt_tweak_search_clip`, `kt_error_adjust`, `kt_lossless_proc`,
  `kt_temporal_soften_1/2` (binomial, scene-change flags as scalars),
  `kt_weave` (KDoubleWeave), `kt_copy`.

Each is a faithful scalar OpenCL port (see the per-file notes about why scalar is
mathematically identical). See [`docs/PORT_PLAN.md`](docs/PORT_PLAN.md) for the
full kernel-by-kernel roadmap and how to extend the set.

## Layout

```
docs/PORT_PLAN.md              # port strategy + per-kernel roadmap + licensing
docs/RIG_HANDOFF_BRINGUP.md    # ** START HERE for rig work ** repo-wide bring-up handoff
docs/HANDOFF_PROMPT.md         # copy-paste prompt for handing this repo to the next agent
docs/MV_PORT_SPEC.md           # motion-engine data model + kernel inventory + rig test plan
docs/HOST_CONTRACT.md          # on-rig OpenCL host runner spec (builds, grids, buffer layout)
docs/BLOCKSEARCH_MODEL.md      # KTGMC block-search host model (SearchBatch, super-frame, CPU_EMU)
docs/CODEX_HANDOFF.md          # step-by-step recipe for the rig-bound MV remainder
docs/KFM_PORT_SPEC.md          # KFM filter-family map + verified/next status
docs/RIG_HANDOFF_KDEBLOCK.md   # verification handoff spec for the provisional KDeblock kernels
docs/AVSCUDA_PORT_SPEC.md      # AvsCUDA family map + batch status (18/18 ported)
docs/RIG_HANDOFF_AVSCUDA_CONDITIONAL.md  # device-run handoff for the 2 float RIG-VERIFY reductions
docs/MASKTOOLS_PORT_SPEC.md    # masktools CUDA surface (5/5) + the 16-bit LUT defect
docs/NNEDI3_PORT_SPEC.md       # NNEDI3 6-kernel census + batch verification records
docs/PERF_NOTES.md             # static perf pass: work-group contracts, atomics, vectorization
src/opencl/masktools/kernels/  # OpenCL kernel sources: masktools fill/copy/LUT
src/opencl/nnedi3/kernels/     # OpenCL kernel sources: NNEDI3 pad/prescreen/compute
src/opencl/ktgmc/kernels/      # OpenCL kernel sources: KTGMC motion/simple
src/opencl/kfm/kernels/        # OpenCL kernel sources: KFM (deband/edgelevel/temporalnr/mergestatic/filterbase/combinganalyze/decombeucf/noiseclip/deblock/deblock_rig .cl)
src/opencl/avscuda/kernels/    # OpenCL kernel sources: AvsCUDA (merge/filters/convert/conditional/conditional_rig/resample)
src/host/                      # OpenCL host harnesses + real AviSynth+ plugins (KEdgeLevelOCL, KTemporalNROCL, KDebandOCL, KTemporalDiffOCL+KMergeStaticOCL, KNoiseClipOCL, KDeblockOCL, the CombingAnalyze/DecombeUCF analysis family, and the KTGMC motion starter KMSuperOCL; src/host/plugin/, see build_host_*.bat)
third_party/grunt/             # verbatim upstream GRunT (CPU-only AviSynth plugin, no CUDA to port)
sim/ktgmc_cpu_ref.cpp          # scalar CPU mirror of the kernels (validates logic)
python/run_validation.py       # independent Python golden + cross-check harness
Makefile                       # make test  (no OpenCL required)
```

## Port status

- `ktgmc_simple.cl`: 25 per-plane KTGMC kernels — **bit-for-bit validated**
  (8/16-bit) via CPU + Python references (`make test` PASS). KGaussResize
  needs no new kernels (same `kt_resample_h/v` at fir 8/9); its gaussian
  program builder + fir-8/9 planes/tables are covered by `run_validation.py`.
- `ktgmc_motion.cl`: motion / super-sampling kernels. **ALG-VERIFIED**: the
  degrain weight helpers (`kt_degrain_weight`, `kt_norm_weights`), MV-aux
  kernels (`kt_write_default_mv`, `kt_scene_change`/`_x2`, `kt_short_to_byte`,
  `kt_short_to_byte_or_copy_src`), the coarse→fine MV upsampler
  `kt_interpolate_prediction`, the global-MV refinement `kt_mean_global_mv`,
  the per-block search setup `kt_prepare_search`, the MV I/O quartet
  `kt_load_mv`/`kt_store_mv`/`kt_load_mv_batch`/`kt_init_const_vec`, and both
  reduced-plane builders `kt_rb2b_bilinear_filtered` (separable) and
  `kt_rb2b_bilinear_filtered_with_pad` (CUDA-only fused twin that also fills the
  hpad/vpad border), and the degrain/compensate pixel-combiner core
  `kt_degrain_patch` + `kt_overlap_out` (the full Degrain1to6_C/Overlaps_C/
  Short2Bytes overlap arithmetic, ALG-VERIFIED vs the MV.cpp staging mirror).
  `kt_calc_all_sad` (per-block SAD vs the MV-selected ref block) is
  **ALG-VERIFIED** via `python/run_mv_calc_all_sad.py` (36 cases vs
  `sim/mv_calc_all_sad_ref.cpp`): its host layout was settled by reading
  MVKernel.cu, which also exposed two ABI bugs in the port (`vectors` is
  `short2`, not `int2`; `out` is a packed 12-byte `VECTOR`, not `int3`) —
  see `docs/BLOCKSEARCH_MODEL.md` §8a.
  The full `CPU_EMU=true` block-search driver is ported 2026-10-28 as
  `kt_search_column` in `src/opencl/ktgmc/kernels/ktgmc_search.cl`
  (**ALG-VERIFIED**, device: `rig/dev_ktgmc_search.py` — golden E2E over
  BS 8/16/32 × SEARCH 1/2 × NPEL × chroma, both PX, plus an 8-config
  three-way vs the Python golden and the stage-1 CUDA twin dumps; off-device:
  golden runner `python/run_mv_search.py`, 81 cases, CPU mirror
  `sim/ktgmc_search_ref.cpp`; `docs/BLOCKSEARCH_MODEL.md` §8b/§8c resolved —
  the `prog`/`next` work-steal handshake is replaced by host-side column
  serialisation).
  `kt_prepare_degrain` / `kt_prepare_compensate` are ported in
  `ktgmc_motion.cl` (`// RIG-VERIFY`; graduated 2026-10-28 from the deleted
  quarantine `ktgmc_degrain_rig.cl`, byte-identical): they produce
  the flat per-block offset/weight arrays that the ALG-VERIFIED
  `kt_degrain_patch` consumes, closing the `docs/MV_PORT_SPEC.md` §6.2 seam,
  with their arithmetic pinned by `python/run_mv_prepare.py` (240 cases).
  **RIG-VERIFY** (device run pending): frame padding / mirror copy
  (`kt_copy_pad`, `kt_pad_frame_h/v`) and `kt_init_scene_change`; the
  block-search pure helpers and the whole search driver are now covered by
  `kt_search_column` (**ALG-VERIFIED**, above). `kt_most_freq_mv` is **ALG-VERIFIED**
  (staged-fold tie-break, CUDA-twin 239/239 tie axes — `rig/cuda_mv_tie_test.py`).
  The remaining degrain-block /
  compensate kernels need the MV.cpp host state machine + super-frame sub-pel
  layout (see `docs/MV_PORT_SPEC.md`, `docs/CODEX_HANDOFF.md`).
- `kfm_deband.cl`: **KFM KDeband core** — `kf_deband_reduce_banding`, faithful to
  the authoritative CPU twin `cpu_reduce_banding` (KFM/KDeband.cu, MIT).
  **ALG-VERIFIED** via `python/run_kfm_deband.py` (300 cases: verbatim CPU
  mirror `sim/kfm_deband_ref.cpp` vs independent Python golden; sample modes
  0-2, blur_first, 8/16-bit).
- `kfm_edgelevel.cl`: **KFM KEdgeLevel** — `kf_edgelevel`,
  `kf_edgelevel_repair`, `kf_el_to444`, `kf_el_from444`, faithful to their CPU
  twins in KFM/KDeband.cu (MIT). Edge detection/enhance + visualise (`check`),
  the 3×3 `repair` limiter (interior ALG-VERIFIED; border needs a padded plane
  on the rig), and the 4:2:x↔4:4:4 chroma pack/unpack helpers. Float math is
  IEEE float32 with no FMA contraction. **ALG-VERIFIED** via
  `python/run_kfm_edgelevel.py` (500 cases: CPU mirror
  `sim/kfm_edgelevel_ref.cpp` vs an independent float32-exact Python golden,
  all 8 check/selective/uv combos + repair N=1-4 + to/from444, 8/16-bit).
- `kfm_temporalnr.cl`: **KFM KTemporalNR** — `kf_temporal_nr`, faithful to
  `cpu_temporal_nr` (KFM/KDeband.cu, MIT). Temporal noise reduction: for each
  pixel averages the same-location pixels across `2*dist+1` frames that are
  within `thresh` of the centre (`avg = (float)sum/count + 0.5f`). Column-wise,
  so a scalar port is identical to the CUDA vectorised one. **ALG-VERIFIED** via
  `python/run_kfm_temporalnr.py` (300 cases: CPU mirror
  `sim/kfm_temporalnr_ref.cpp` vs an independent float32-exact Python golden,
  dist 0-6, 8/16-bit). NOTE: OpenCL uses normal fp32 `/` (matches the CPU twin);
  the real CUDA device kernel uses the approximate `__fdividef` (≈ulp-level CUDA
  HW nuance, no OpenCL equivalent) — see spec.
- `kfm_mergestatic.cl`: **KFM MergeStatic.cu kernels** — `kf_compare_frames`
  (KTemporalDiff core, 5-frame max−min), `kf_min_frames` (KAnalyzeStatic
  temporal-min sub-step), `kf_and_coefs` (KAnalyzeStatic combing∧static
  sub-step, float32), `kf_merge_static` (KMergeStatic core,
  `(coef*v30+(128-coef)*v60+64)>>7`). KTemporalDiff and KMergeStatic are
  kernel-complete (only AviSynth host glue remains); KAnalyzeStatic's remaining
  coefficient kernels are in kfm_filterbase.cl. **ALG-VERIFIED** via
  `python/run_kfm_mergestatic.py` (580 cases: CPU mirror
  `sim/kfm_mergestatic_ref.cpp` vs an independent float32-exact Python golden).
- `kfm_filterbase.cl`: **KFM KFMFilterBase coefficient + pad + merge kernels** —
  `kf_calc_combe` (CompareFields combing, `|a+4c+e-3(b+d)|>>2` clamped [0,255];
  interior ALG-VERIFIED, border needs a VPAD-padded plane on the rig),
  `kf_merge_uvcoefs` (fold UV into Y), `kf_extend_coef2` (ExtendCoefs; = the
  CUDA `kl_extend_coef2` device kernel — upstream's CPU fallback differs at the
  extreme rows), `kf_apply_uvcoefs_420` (YV12 Y→UV), the shared in-place mirror
  pads `kf_padv`/`kf_padh` (exact `cpu_padv`/`cpu_padh` twins, race-free; 2D pad
  = padv-then-padh host sequencing, as DeblockPlane does), and the MergeBlock
  blender `kf_merge_block` (`(flag*src60+(128-flag)*src24+64)>>7`, uchar flag at
  both depths, full 0..255 flag sweep; makes KPatchCombe/KFMSwitch
  kernel-complete), plus the CombingAnalyze/CompareFields helpers `kf_average`
  (floor mean), `kf_max` (uint8 max; scalar twin, so any width),
  `kf_merge_uvflags` (in-place `fY|=((fU|fV)<<4)`, mod-256 wrap covered),
  `kf_copy_border` (extreme rows straight through) and `kf_analyze_frame`
  (SHIMA/LSHIMA/MOVE threshold fold over padded taps, all rows verified),
  the padded-frame copies `kf_copy_pad`/`kf_copy_pad_2plane` (lane-swap
  quirk proven to cancel; no upstream CPU twin) and the ExtendBlocks
  ping-pong `kf_max_extend_blocks_h/v` (composed h->v proven equal to the
  in-place CPU twin), and the plain plane utilities `kf_copy` /
  `kf_copy_2plane` (3D grid over the plane pair) / `kf_fill` (runtime value;
  also covers `kl_fill<*,0>`). These plus `kf_min_frames`/`kf_and_coefs`
  complete
  KAnalyzeStatic's kernel set (only the VPAD-pad host assembly remains,
  `// RIG-VERIFY`); `KFMFilterBase.cu` itself is now fully ported.
  **ALG-VERIFIED** via `python/run_kfm_filterbase.py` (3100 cases,
  integer-exact).
- `kfm_noiseclip.cl`: **KFM KNoiseClip** — `kf_noise_clip`, faithful to
  `cpu_noise_clip`/`dev_limitter` (KFM/DecombeUCF.cu, MIT). Self-contained
  **8-bit-only** filter that maps each src pixel against a `noise` pixel into
  band markers `{0,56,128,199,255}`: `out = dev_limitter((src-noise+256)>>1,
  nmin, range)` (128 == equal). Driven by one per-plane kernel, so it is
  kernel-complete (only AviSynth host glue remains). **ALG-VERIFIED** via
  `python/run_kfm_noiseclip.py` (300 cases, integer-exact, nmin/range sweeps).
- `kfm_decombeucf.cl`: **KFM DecombeUCF reductions (7/7 remaining kernels)** —
  `kf_init_uint64`, `kf_calculate_field_diff`, `kf_init_block_sum`,
  `kf_add_block_sum` (BLOCK_SIZE 4/8/16/32), `kf_block_sum_max`,
  `kf_analyze_noise`, `kf_analyze_diff` (KFM/DecombeUCF.cu, MIT): gated 5-tap
  combe sums, per-block sumAbs/sumSig accumulation, and the analyze-noise/
  diff census with uint64 atomics. Together with `kf_noise_clip`, all 8
  DecombeUCF.cu device kernels are ported (only multi-clip host pipelines
  remain). **ALG-VERIFIED** via `python/run_kfm_decombeucf.py` (810 cases,
  integer-exact).
- `kfm_combinganalyze.cl`: **KFM CombingAnalyze, complete (15/15 kernels)** —
  `kf_copy_first`, `kf_combe_to_flag`, `kf_sum_box3x3`, `kf_binary_flag`,
  `kf_bilinear_h/v`, `kf_temporal_soften` (float32, exhaustive t sweep),
  `kf_remove_combe2` (8/16-bit), `kf_clean_super`, `kf_contains_durty_block`
  (+ init), the 8-tap `kf_calc_combe8`/`kf_calc_diff8` helpers, the KFMSuper
  block analyzer `kf_super_analyze` (serial-cell == warp-reduce) and the
  FMCount census `kf_init_fmcount`/`kf_count_cmflags`/`kf_count_cmflags_2planes`
  (group-reduce + atomics; fused == U+V proven). **ALG-VERIFIED** via
  `python/run_kfm_combinganalyze.py` (2731 cases).
- `kfm_deblock.cl`: **KFM KDeblock** — `kf_deblock` (the CUDA `kl_deblock`
  device kernel, KFM/Deblock.cu, MIT): the fixed float32 8×8 DCT →
  hard-threshold (AC coeffs only, DC untouched) → IDCT deblocking stage that
  runs `count = 1<<quality` shifted passes per block and accumulates into a
  16-bit block-parity plane (transcribed from the *device* transform path; the
  upstream CPU fallback's `thresh<=0` identity shortcut is not used). Plus the
  QP-table builder `kf_make_qp_table` (`kl_make_qp_table`, macroblock-QP → block
  QP with optional DC blend + `norm_qscale` 0-3) and the `show==2` visualiser
  `kf_deblock_show` (`kl_deblock_show`, paints enabled/disabled blocks 230/16).
  **ALG-VERIFIED** via `python/run_kfm_deblock.py` (300 cases, core, vs
  `sim/kfm_deblock_ref.cpp` + float32-exact golden) and
  `python/run_kfm_deblock_qp.py` (make_qp_table + deblock_show, 200+200 cases,
  vs `sim/kfm_deblock_qp_ref.cpp`) — bit-exact. Graduated from the rig file:
  the DC-mask dilation `kf_max_vh/v/h`, the ShowQP scaler `kf_scale_qp`, the
  sharpen LUT `kf_sharpen_coeff` and the Bayer accumulator merge
  `kf_merge_deblock` (+`g_ldither`), **ALG-VERIFIED** via
  `python/run_kfm_deblock_aux.py` (1593 cases vs `sim/kfm_deblock_aux_ref.cpp`,
  incl. merge end-to-end + layout proofs and sharpen/show deterministic pins).
  The SharpenFilter pair `kf_sharpen` /
  `kf_show_sharpen_coeff` (manual bilinear replacing the CUDA texture fetch)
  graduated 2026-10-28 into `kfm_deblock.cl` as `// ALG-VERIFIED` (mirror+golden
  + the OpenCL device run + the CUDA texture-precision twin comparison; the ±1
  texture-grid divergence on `kf_sharpen` is characterised and accepted) — the
  provisional `kfm_deblock_rig.cl` was emptied and deleted (see git history);
  all 11 Deblock.cu device kernels are transcribed, with the host sequencing
  remaining open.
  Verification handoff spec (closed):
  `docs/RIG_HANDOFF_KDEBLOCK.md`. Documented in `docs/KFM_PORT_SPEC.md`.
- `avscuda_merge.cl`: **AvsCUDA Merge/MergeChroma/MergeLuma planar core** —
  `ka_merge` (in-place weighted merge, dual pitch, >>15 SIMD/device scale;
  the scalar-C >>16 fallback is a different weight scale, not a twin),
  `ka_merge_f32`, `ka_average` (exact `average_plane_c` twin) and
  `ka_average_f32` (device `(a+b)*0.5f` vs CPU `(a+b)/2.0f`, proven equal).
  Host dispatch contract (average band / early-outs / YUY2-throws) pinned in
  the file header. **ALG-VERIFIED** via `python/run_avscuda_merge.py`
  (620 cases vs `sim/avscuda_merge_ref.cpp`, incl. band-edge weights).
- `avscuda_filters.cl`: **AvsCUDA Invert** — `ka_invert_plane_u8/u16/f32`
  (word-XOR lane transcription; planar full masks + packed channel masks;
  exact row widths, no word overhang) and `ka_invert_rgb` (RGB24/48
  interleaved per-channel XOR). **ALG-VERIFIED** via
  `python/run_avscuda_filters.py` (750 cases vs
  `sim/avscuda_filters_ref.cpp`). Family map in `docs/AVSCUDA_PORT_SPEC.md`.
- `avscuda_convert.cl`: **AvsCUDA ConvertBits** — ordered-Bayer down-convert
  (`ka_convert_lower_dither_u8/u16`, verbatim tables), truncating down-convert
  (`ka_convert_lower_nodither_u8/u16`), shift up-convert
  (`ka_convert_higher_from_u8/from_u16`), float→int (`ka_convert_from_float_u8/
  u16`, rgy clamp macro spelled out: NaN yields MAX_VAL) and int→float
  (`ka_convert_to_float_from_u8/from_u16`). **ALG-VERIFIED** via
  `python/run_avscuda_convert.py` (1320 cases vs
  `sim/avscuda_convert_ref.cpp`, incl. NaN/Inf edges).
- `avscuda_conditional.cl`: **AvsCUDA Conditional metrics** — counter/hist
  inits, AveragePlane sums, PlaneDifference SAD (u8/u16 x u32/u64) and MinMax
  histograms (u8/u16/f32; NaN index yields 65535). Integer reductions use
  per-item atomics (order-free); u32 wraparound pinned. **ALG-VERIFIED** via
  `python/run_avscuda_conditional.py` (1320 cases vs
  `sim/avscuda_conditional_ref.cpp`). The float sum/SAD reductions are
  faithful transcriptions in `avscuda_conditional_rig.cl`,
  **`// RIG-COMPARED` 2026-10-28** (arrival order undefined upstream too;
  CUDA twin comparison done — 200 CUDA reps x 10 CL reps, Phase A bit-exact
  11/11, Phase B within the accepted leaf-association envelope; not covered
  by `make test` by design).
  Verification handoff spec (closed, appendix filled):
  `docs/RIG_HANDOFF_AVSCUDA_CONDITIONAL.md`.
  OpenCL-side host harness: `src/host/run_avscuda_rig.cpp` (canonical input
  fills + §4.4 launch; build via cmake when OpenCL exists).
- `avscuda_resample.cl`: **AvsCUDA FilteredResizeH/V** — row/unit select
  (`ka_resize_v_pointresize[_f32]`, `ka_resize_h_pointresize_bytes`) and
  separable filters (`ka_resize_v_planar[_f32]`, `ka_resize_h_planar_u8/u16/
  f32`; logical programs, shared-mem/transpose staging elided, unfused f32).
  **ALG-VERIFIED** via `python/run_avscuda_resample.py` (1230 cases vs
  `sim/avscuda_resample_ref.cpp`). AvsCUDA is now fully ported (18/18).
- `nnedi3_pad.cl`: **NNEDI3 batch-1 (pad/copy)** — in-place mirror pads
  `kn_pad_h/v` (group-id idiom, interior origin), plain `kn_copy`, and the
  fused `kn_pad_ref_and_copy_half` (per-vector grid, lane reversal on
  x-mirror). **ALG-VERIFIED** via `python/run_nnedi3_pad.py` (600 cases vs
  `sim/nnedi3_pad_ref.cpp`). Family map in `docs/NNEDI3_PORT_SPEC.md`.
- `nnedi3_prescreen.cl`: **NNEDI3 batch-2 (prescreener)** — `kn_prescreening`,
  the 48-tap int neighbourhood net + unfused-f32 squash tail, the
  scan-compacted `workNN`/`numblocks` work list (no atomics; Hillis-Steele
  stands in for upstream's warp-shuffle scan, bit-identically) and the
  bicubic fast path. Fixed `reqd_work_group_size(32,16,1)`.
  **ALG-VERIFIED** via `python/run_nnedi3_prescreen.py` (66 cases vs
  `sim/nnedi3_prescreen_ref.cpp`, 7 mutants caught).
- `nnedi3_compute.cl`: **NNEDI3 batch-3 (predictor net)** — `kn_compute_nn`,
  one runtime-parameterised kernel replacing upstream's 70 template
  instantiations (all seven `ReadPixelNxM` policies stage the same logical
  tile). Reproduces the order-significant butterfly reduction tree and the
  `dev_expf` bit-twiddle exactly. **ALG-VERIFIED** via
  `python/run_nnedi3_compute.py` (28 cases vs `sim/nnedi3_compute_ref.cpp`,
  comparing pre-rounding float bits, with enforced branch coverage).
  **KNNEDI3 is now fully ported (6/6).**
- `masktools_lut.cl`: **masktools, entire CUDA surface (5/5)** — `km_fill`
  (+`_f32`), `km_copy_bytes`, and the LUT kernels `km_lut_x/xy/xyz`.
  **ALG-VERIFIED** via `python/run_masktools_lut.py` (610 cases vs
  `sim/masktools_lut_ref.cpp`). Transcribes — and pins with dedicated
  assertions — an upstream CUDA defect where every 16-bit depth is
  dispatched as 8-bit, collapsing the multi-clip LUT index
  (`docs/MASKTOOLS_PORT_SPEC.md` §3).

## How the port is validated (no GPU/OpenCL needed)

This sandbox has no NVIDIA GPU, no CUDA toolkit, and no AviSynthNeo — and the
Debian package mirrors are unreachable, so even a CPU OpenCL ICD (pocl) could not
be installed. To still make the port *provably* correct, each algorithm exists in
**two independent implementations** that must agree bit-for-bit:

1. `python/run_validation.py` — a Python golden reference written directly from
   the CUDA math.
2. `sim/ktgmc_cpu_ref.cpp` — the scalar CPU mirror that the OpenCL kernels are
   transliterated from.

The OpenCL `.cl` bodies mirror the CPU mirror expression-for-expression
(float accumulation only in the resampler, matching the original CUDA `float`
path). On any machine with an OpenCL ICD you can additionally compile/run the
`.cl` and diff against the CPU mirror.

```sh
make test        # structural .cl lint + CPU-ref + 8/16-bit + motion cross-checks
```

`make test` also runs `make lint`, which does a *genuine* C parse of every
`.cl` (gcc does not parse `.cl` files by themselves — they are copied to `.c`
and parsed through `lint/oc_shim.h`) to catch syntax/typo errors before the
rig's OpenCL compiler is available. This is a structural check only; real
OpenCL semantics are validated on a device.

Expected output:

```
[bitdepth 8] PASS
[bitdepth 16] PASS
```

## Running on a real rig (future milestones)

- Port the motion-search / compensation stages (`MVKernel.cu`, `MV.cpp`) so the
  full KTGMC deinterlacer can be assembled.
- Add AviSynthNeo plugin glue (AviSynth host adaptation is separate from the
  kernels; see `docs/PORT_PLAN.md` §"AviSynth integration").
- Optional: reintroduce 4-channel vectorization as a pure performance pass.

## 2026-10-29 — first real AviSynth+ host plugin (KEdgeLevelOCL)

`src/host/plugin/` now contains a loadable 64-bit AviSynth+ plugin
(`kfm_edgelevel_avs.dll`, registers `KEdgeLevelOCL` / alias `KEdgeLevel`) that
runs the ALG-VERIFIED `kfm_edgelevel.cl` kernels on an OpenCL device through
`src/host/ocl_host.*`, replaying the upstream KEdgeLevel host pipeline
(strengthening passes, repair ping-pong, to444/from444, show/ClearUV). Verified
headless by `test_kfm_edgelevel.cpp` (real autoload via AviSynth+ 3.7.3, 6
formats × 12 param cases + defaults + script-string; every plane pixel checked
against an in-exe CPU reference): 79 cases / 144 frame checks / 0 failures.
Build with `src/host/plugin/build_host_edgelevel.bat`. Documented deviations
(subsampled GetUVFrame dims, interior-only repair dispatch with host-defined
border ring, show+uv frame type) are listed in the plugin header.

Same day, the second host plugin: `kfm_temporalnr_avs.dll`
(`KTemporalNROCL` / alias `KTemporalNR`) runs the ALG-VERIFIED
`kfm_temporalnr.cl` temporal noise reducer — the port's first temporal filter
(stateless windowed over `clamp(n-dist+i)`, no persistent cross-frame state;
state model documented in the plugin header). Verified headless by
`test_kfm_temporalnr.cpp` (6 formats × random/constant/scene-change sources ×
6 param sets, 8/16-bit, first-frame + heavy-boundary clamping; every plane
pixel vs an in-exe CPU reference): 116 cases / 310 frame checks / 0 failures.
Build with `build_host_temporalnr.bat`. It also surfaced a host-contract
discrepancy in the `.cl` header comment (frame_stride is ELEMENT-stride per
the goldens, not byte-stride — pinned by the 16-bit cases; kernel untouched,
documented as plugin deviation 4).

Same day, the third host plugin: `kfm_deband_avs.dll` (`KDebandOCL` / alias
`KDeband`) runs the ALG-VERIFIED `kfm_deband.cl` debander
(`kf_deband_reduce_banding`) — one reduce_banding launch per plane, one shared
luma-dims XorShift rand stream (seed 0) exactly like upstream KDeband's
`CreateDebandRandom`; the upstream KDeband class launches exactly one device
kernel, so the ported kernel covers 100% of its pipeline. Verified headless by
`test_kfm_deband.cpp` (6 formats × smooth/noisy/banded sources × 7 param sets
incl. defaults and range 0/127, thresh 0/large boundaries; every plane pixel vs
an in-exe CPU reference): 141 cases / 256 frame checks / 0 failures. Build with
`build_host_deband.bat`. One documented deviation (plugin header): range > 127
throws instead of upstream's silent `char` truncation.

Same day, the fourth host plugin: `kfm_mergestatic_avs.dll` registers
`KTemporalDiffOCL` + `KMergeStaticOCL` (with the upstream `KTemporalDiff` /
`KMergeStatic` names as aliases, exact upstream signatures `"c"` / `"ccc"`)
and runs the ALG-VERIFIED `kfm_mergestatic.cl` kernels — `kf_compare_frames`
for KTemporalDiff (stateless 5-frame `clamp(n-2+i)` window, per plane
`max-min` temporal spread) and `kf_merge_static` for KMergeStatic (per plane
`(coef*v30 + (128-coef)*v60 + 64) >> 7` over child(n), clip30(n>>1),
sttclip(n>>1)); neither upstream class holds cross-frame state (the state
model is documented in the plugin header, following the KTemporalNROCL
precedent). Upstream's third MergeStatic.cu filter, KAnalyzeStatic, is
deliberately not assembled here (its `kf_min_frames`/`kf_and_coefs` kernels
are already ported and ALG-VERIFIED; its CompareFields/ExtendCoefs pipeline
machinery is a later milestone, and `KMergeStatic` never instantiates it).
Verified headless by `test_kfm_mergestatic.cpp` (6 formats x random/constant/
scene-change sources, 8/16-bit, first-frame + 2-frame-clip clamping, flag
coefficient corners 0/128, frame-count-mismatch rejection, script-string
cases; every plane pixel vs an in-exe CPU reference): 63 cases / 242 frame
checks / 0 failures. Build with `build_host_mergestatic.bat`. Documented
deviations (plugin header): the off-by-one `clamp(ref, 0, vi.num_frames)`
upper bound fixed to `num_frames - 1`, the redundant pre-merge CopyFrame
skipped (scalar kernel writes every column), the CUDA-only device-match
constructor check dropped.

Same day, the fifth host plugin: `kfm_noiseclip_avs.dll` (`KNoiseClipOCL` /
alias `KNoiseClip`, exact upstream signature
`"cc[nmin_y]i[range_y]i[nmin_uv]i[range_uv]i"`, defaults 1/128/1/128) runs the
ALG-VERIFIED `kfm_noiseclip.cl` band-marker kernel (`kf_noise_clip`) — one
launch per plane (Y with nmin_y/range_y, U/V with nmin_uv/range_uv):
`out = dev_limitter((src - noise + 256) >> 1, nmin, range)` against a
same-location `noise` clip, exactly upstream DecombeUCF.cu's KNoiseClip (whose
class launches exactly one kernel, so the port covers 100% of its pipeline).
The class is stateless (output n = f(child(n), noise(n)); state model in the
plugin header). Verified headless by `test_kfm_noiseclip.cpp` (8-bit only, as
upstream: YV12/YV16/YV24 x smooth/noisy/mixed/extremes-0/255 source pairs x 5
param sets incl. defaults and nmin/range boundaries; every plane pixel vs an
in-exe CPU reference; plus width/height %4, resolution-mismatch and 16-bit
rejections, defaults-only/alias/script-string cases): 170 cases / 322 frame
checks / 0 failures. Build with `build_host_noiseclip.bat`. Documented
deviations (plugin header): the scalar OpenCL kernel is exact for any width
but the upstream width/height %4 constructor checks are kept verbatim, plain
pitch==width device planes, the CUDA device-match check dropped.

Same day, the sixth host plugin: `kfm_deblock_avs.dll` (`KDeblockOCL` / alias
`KDeblock`, exact upstream signature
`"c[quality]i[str]f[thr]f[bratio]f[qpclip]c[qp]i[badap]b[sharp]b[show]i"`,
defaults quality 3, str 0, thr 0, bratio = badap ? 2 : 0.5, qp -1, badap true,
sharp false, show 0) assembles the full KFM Deblock.cu host pipeline around the
ALG-VERIFIED `kfm_deblock.cl` kernels: `QPForDeblock` (props / qpclip /
`KFM_SourceStart` frame mapping with `frameRateConv`, force_qp constant table,
`kf_make_qp_table` with `b_ratio` DC blending and optional `DC_Table` masks),
`KDeblock` (`kf_deblock` DCT + hard-threshold + IDCT over the
`(w+15)>>3 x (h+15)>>3` parity-accumulator grid — the by=0 row launches first
and by>=1 in a second in-order dispatch so the odd-bh parity-plane overlap
resolves deterministically like upstream's CPU loop order — then
`kf_merge_deblock` 4-parity-slice merge with ordered dither) and the
`SharpenFilter` tail (`kf_sharpen_coeff` + `kf_sharpen` behind the real core
GaussResize, chained exactly like upstream's Create). Mirror pads (MakeMask
radius-5 box max and the DeblockPlane 8px pad) run as host staging (upstream's
own CPU-fallback path does the same); show=1/2 text is drawn bit-exactly with
the ported DrawText font. Verified headless by `test_kfm_deblock.cpp` (8-bit +
16-bit x 6 formats x qpclip/props/force_qp/DC/sourceStart/show 0-3/sharp/alias
sweeps incl. the 64x24 sharpen quirk config; every plane pixel vs an in-exe CPU
reference): 137 cases / 268 frame checks / 0 failures; the three `run_kfm_deblock*`
golden runners PASS (their sim builds now force `-mfpmath=sse -msse2` — 32-bit
x87 intermediates are not per-op float32 and broke bit-exactness vs the
golden). Build with `build_host_deblock.bat`. Documented deviations (plugin
header): the accumulator origin/merge-offset bookkeeping (no +8 base offset
needed), the deterministic kf_deblock split dispatch (upstream CUDA has a
genuine write race there), host-staged pads and QP-table margin zero-fill,
zero-filled pad rows beyond h+16 (uninitialized upstream), the fused
SharpenFilter tail, and the documented kf_sharpen bilinear-vs-texture ±1 class.

Same day, the seventh host plugin: `kfm_analyze_avs.dll` assembles the REMAINING
KFM analysis/reduction filters of CombingAnalyze.cu + DecombeUCF.cu in one DLL,
registering `KFMSuperOCL`, `KCleanSuperOCL`, `KPreCycleAnalyzeOCL`,
`KPreCycleAnalyzeShowOCL`, `KContainsCombeOCL`, `KCombeMaskOCL`,
`KRemoveCombeOCL`, `KCFieldDiffOCL`, `KCFrameDiffDupOCL` and `KAnalyzeNoiseOCL`
(each with the exact upstream name as an alias and the exact upstream signature
and defaults) around the ALG-VERIFIED `kfm_combinganalyze.cl` (15/15 kernels),
`kfm_filterbase.cl` and `kfm_decombeucf.cl` (8/8) kernels. The two
`ConditionalFilter` helpers (`KCFieldDiff` / `KCFrameDiffDup`) read
`current_frame` exactly like upstream's `CFunc` and return the double. The
classes are stateless (per-frame data from child/auxiliary clips; all device
buffers are per-call scratch). Verified headless by `test_kfm_analyze.cpp`
(YV12/YV16/YV24 (+YUV420P16/YUV444P16 for KRemoveCombe/KCFieldDiff/
KCFrameDiffDup/KFMSuper), smooth/noisy/combed sources, parity both ways, th
sweeps incl. boundaries, alias invocations, border-only KContainsCombe,
staged+superclip KAnalyzeNoise, written-cell KFMSuper after the real core
SeparateFields; every plane pixel/counter vs an in-exe CPU reference): 65 cases
/ 106092 checks / 0 failures. The `run_kfm_combinganalyze.py` (2731 cases) and
`run_kfm_decombeucf.py` (810 cases) goldens PASS. Build with
`build_host_analyze.bat`. Documented deviations (plugin header): the pair-packed
uchar2 frames are split into separate `.x`/`.y` planes (layout bookkeeping,
bit-identical); interior-origin pointers expressed as global-work offsets over
whole-buffer allocations (no sub-buffers on this driver, RIG_HANDOFF 9.4);
KCombeMask's separable bilinear upscale runs as the exact upstream CPU twin in
host staging (its `src-1row`/`dsttmp+1row` interior-origin pointer math needs a
sub-buffer this driver cannot express); KRemoveCombe runs upstream's own
`cpu_remove_combe2` twin; the DecombeUCF padded planes carry the +/-2-row ring
the kernels tap; the `best` floor rule and blocksize-4 mapping of
KCFrameDiffDup. Upstream registrations NOT implemented (documented, with
reasons, in the plugin header): `KSwitchFlag`/`KFMSuperShow` (upstream host-side
flag dims bug / debug-only visualiser), `KTelecine`/`KTelecineSuper`/
`KPatchCombe`/`KFMSwitch`/`KFMPad`/`KFMDecimate`/`AssumeDevice` (no ported
`kfm_kernel.cl`), the `KDecombUCF*` decision machinery (pure host, no device
kernels) and `KAnalyzeStatic` (all kernels ported but its shifted-buffer chain
formulation is deferred). No implemented KFM class requires GRunT script
evaluation; the GRunT pairings only wrap the un-ported `KDecombUCF*` consumers.

## 2026-10-30 — first KTGMC MOTION host plugin (KMSuperOCL)

`src/host/plugin/ktgmc_super_avs.cpp` (`ktgmc_super_avs.dll`, registers
`KMSuperOCL` with the upstream `KMSuper` name as alias and the exact upstream
signature) is the first plugin of the KTGMC motion pipeline: it builds the
super-sampled pyramid frame around the ALG-VERIFIED `ktgmc_simple.cl` /
`ktgmc_motion.cl` kernels — the pel-upscale wiener pair
(`kt_vertical_wiener`/`kt_horizontal_wiener`) for sharp levels, the
separable/fused RB2B 1:2 downsamplers (`kt_rb2b_bilinear_filtered[_with_pad]`)
for coarse levels, and the frame padding kernels (`kt_copy_pad` CUDA-reflection
form), with the pyramid cached per source frame. Verified headless by
`test_ktgmc_super.cpp` (real AviSynth+ 3.7.3 autoload; pel 1/2/4 x sharp
on/off x rfilter variants x chroma on/off, P16 + 8-bit; every plane of the
produced super frame vs an in-exe CPU reference; a replication-border mutation
fails every case, so the reference discriminates): 33 cases / 31 frame checks
/ 0 failures. Build with `build_host_super.bat` (embeds both `.cl`). The plugin
uses the new SVM (`clSVMAlloc`) staging added to `src/host/ocl_host.*` for the
pyramid buffers (the driver's interior sub-buffer origins are unreliable,
RIG_HANDOFF 9.4). Documented deviations (plugin header): SVM pyramid buffers
instead of interior sub-buffers, chroma=false output planes zero-filled
(unread downstream; uninitialised upstream), tail columns beyond each level's
extended width zero-filled, row-wise copy-back into the AVS pitch, and the
CUDA-reflection border form. The KMAnalyse successor is inventoried but not
assembled (its multi-level `GroupOfPlanes` host loop — lsad/badSAD scaling,
lambda/plevel schedules, batch/caching, BGR32 vector-frame layout — needs either
an extension of the `run_mv_search.py` one-Search-call assembly or a rig CUDA
cross-check; recorded rather than shipping a speculative assembly).

## 2026-11-01 — second KTGMC motion host plugin (KMAnalyseOCL, ninth overall)

`src/host/plugin/ktgmc_analyse_avs.cpp` (`ktgmc_analyse_avs.dll`, registers
`KMAnalyseOCL` with the upstream `KMAnalyse` name as alias and the exact
upstream signature `"c[blksize]i[overlap]i[search]i[isb]b[chroma]b[delta]i[lambda]i[lsad]i[global]b[plevel]i[pnew]i[meander]b[partial]c[batch]i"`)
chains directly behind KMSuperOCL — `KMSuperOCL(src,...).KMAnalyseOCL(...)`,
the motion-analysis half of the KTGMC pipeline. It replays the proven
multi-level `GroupOfPlanes` coarse→fine loop (cuda_mv/KMANALYSE_HOST_CONTRACT.md
§10) per level i = nLevels-1…0 with NO new kernels: the global-MV chain
(`kt_most_freq_mv` staged-fold tie-break → `kt_mean_global_mv` doubled),
`kt_interpolate_prediction` into the fine level, host-written
sentinels (slot −2 zero / slot −1 globalMV·nPel), the proven one-Search
assembly (`kt_prepare_search` → ordered per-column `kt_search_column` →
`kt_calc_all_sad`), and the BGR32 VECTOR output frame assembled coarsest level
first. Per-level forcing (EXHAUSTIVE on coarse levels, nPel=1 on coarse,
nLambdaLevel = λ/(nPelLevel²) with the plevel scaling, nPad = the super's 8 at
every level, covered-grid level count validated against the super's nLevels)
is transcribed from MV.cpp:2248-2282/2508-2573. The SUPER_FRAME `KMVParam` is
now smuggled KMSuperOCL → KMAnalyseOCL through the VideoInfo exactly like
upstream (common/KMV.h SetParam: sample_type/num_audio_samples), and the MV
clip's vi carries the MV_FRAME params for the later KMCompensate/KMDegrain
stages; computed frames carry `KMVIsValid=true` (false on WriteDefault
frames). Verified headless by `test_ktgmc_analyse.cpp` (real AviSynth+ 3.7.3
autoload, chain built through the real `KMSuperOCL` plugin; reference = an
in-exe CPU implementation of the proven loop — CPU super planes from the
verified super reference + the `sim/ktgmc_search_ref.cpp` Search model +
the global-MV/interp kernels transcribed from `ktgmc_motion.cl`): 18 cases /
15 VECTOR-frame checks / 0 failures — EVERY (x,y,sad) of EVERY level's output
region and the final frame bit-exact (206 VECTORs x 3 levels on the 64x48
default, 41 x 2 levels on 32x24 and blksize=16), covering pel 1/2, chroma
on/off, fine-level HEX2SEARCH and EXHAUSTIVE, plevel=2, global=off, backward
search, 16-bit, the parameter defaults, the `KMAnalyse` alias, a real script
string, the WriteDefault out-of-range frame, and the constructor throws.
Build with `build_host_analyse.bat` (embeds `ktgmc_motion.cl` +
`ktgmc_search.cl`). Documented plugin-level decisions (measure report §5,
plugin header D1-D10): coarsest global-MV sentinel = CPU-defined (0,0);
WriteDefault = the intended {0,0,verybigSAD} (the upstream CUDA `.x`-twice
typo is NOT reproduced); the generic level loop (upstream pins lv=0; 2-level
configurations are CUDA-twin-measured, 3-level via the CPU reference); batch
pinned to 1 (the port's `kt_interpolate_prediction` has no batch dimension);
16-bit supported; `partial` rejected (unmeasured SetMVs path); the
CUDA-dead parameters (meander, badSAD/badrange, tryMany, stp/nSearchParam/
nPelSearch) kept in the signature and ignored exactly like upstream CUDA;
the level's sads work buffer keeps the INCOMING sads (calc_all_sad writes
only the output VECTORs — twin-proven); no batch-result cache; only the
out_frame_bytes vector bytes of the BGR32 frame are written.

## Licensing

The ported kernels are derived from `rigaya/AviSynthCUDAFilters`, which is a fork
of `nekopanda/AviSynthCUDAFilters`. Per that repository's README the KTGMC code
is **GPL** (KFM is MIT). Derived code should therefore be distributed under the
same terms as the upstream KTGMC sources. Confirm against the exact upstream
`LICENSE` before any release; this repo currently carries no license by default.
