# MV motion engine — CUDA → OpenCL port specification

This document grounds the port of the KTGMC motion-compensation engine
(`KTGMC/MVKernel.cu` ≈3540 lines + `KTGMC/MV.cpp` ≈5550 lines) so each kernel
can be transliterated faithfully and audited against the original. It is the
reference for `src/opencl/ktgmc/kernels/ktgmc_motion.cl`.

> Status: the full MV engine is a large, multi-stage effort. Everything in the
> motion file is a faithful source port of the CUDA kernels **but is NOT
> runtime-verified here** (no OpenCL/CUDA device in the build sandbox). All
> motion kernels carry a `// RIG-VERIFY` marker until they are cross-checked on
> real hardware against the CUDA implementation (see §7).

## 1. Pipeline overview

The KTGMC motion path mirrors AviSynth `mvtools`, split into three AVS filter
stages plus the QTGMC script that chains them:

1. **KMSuper** — builds the *super frame*: the current frame upsampled (2× in
   each axis for `pel=2`, or 4× for `pel=4`) and stored **separated by sub-pel
   offset**, plus padded borders, as a small pyramid of levels. Each level is
   used as the reference for motion search at that resolution.
2. **KMAnalyse** — per level and per **batch** of rows, runs block motion
   search and writes one `VECTOR{x,y,sad}` per block, coarse→fine
   (predictors from the coarser level + spatial/temporal neighbours).
3. **KMCompensate / KMDegrain1/2** — uses the MV arrays to fetch each block
   from the super frames and write a compensated / weighted denoised frame.

## 2. Core data structures (from source)

From `common/KMV.h` and `KTGMC/MVKernel.cu`:

```cpp
struct VECTOR { int x; int y; int sad; };           // KMV.h (also {x,y} used as short2 internally)
struct LevelInfo { int nBlkX; int nBlkY; };          // KMV.h

// MVKernel.cu
struct SearchBlock {          // one per block, passed in a "search batch"
  int   data[12];  // [0..3]=CLIP_RECT (DxMax,DyMax,DxMin,DyMin); [4..9]=ref-MV indices
                   //  (Left, Up, bottom-right-from-coarse); [10..11]=PRED_X/Y
  sad_t dataf[5];  // [0..3]=penalties (zero,global,0,predictor/new); [4]=lambda
};
struct CostResult { sad_t cost; short2 xy; };
struct SearchBatch { /* holds per-block pointers + ref/MV arrays (see source) */ };

// MVKernel.cu template structs used by degrain/compensate kernels
template<typename pixel_t,int N> struct DegrainBlockData{ const short* winOver; const pixel_t* pSrc; const pixel_t* pB[N],*pF[N]; int WSrc,WRefB[N],WRefF[N]; };
template<typename pixel_t,int N> struct DegrainArgData { const VECTOR* mvB[N],*mvF[N]; const pixel_t* pSrc,*pRefB[N],*pRefF[N]; bool isUsableB[N],isUsableF[N]; };
// both are passed by "union { struct d; uint32 m[LEN]; }" (pointer arithmetic trick)
```

The AVS-side plane parameters live in `MV.cpp` (`MVPlaneParam`, block sizes,
`nBlkX/nBlkY`, overlap, `nPel`, level chain). The search host state machine
(predictor selection, meander direction, level loop, global-MV estimation,
scene-change) is in `MV.cpp` and must be reproduced on the host for a full
plugin; the kernels below are the device-side pieces.

## 3. Kernel inventory (MVKernel.cu, by line) and role

| CUDA | Role | MV stage |
|---|---|---|
| `memcpy_kernel` (26) | plain byte copy | generic |
| `kl_copy_pad` (39) | copy plane into padded buffer, mirror edges | super build |
| `kl_pad_frame_h/v` (65/84) | in-place horizontal/vertical edge replication | super build |
| `kl_vertical/horizontal_wiener` (107/135) | 6-tap Wiener interpolation (already ported → `kt_*_wiener` in ktgmc_simple.cl) | super "sharp" |
| `kl_RB2B_bilinear_filtered(_with_pad)` (175/229) | bilinear 1/8,3/8,3/8,1/8 sub-pel sampler | super |
| `dev_clip_mv`, `dev_check_mv`, `dev_sq_norm` (315/321/326) | MV helpers | search |
| `dev_get_ref_block` (333) | pick ref block pointer incl. sub-pel plane offset | search |
| `load4pix(_Aligned)` (357/368) | 4-byte vector load + funnel-shift (alignment trick) | search |
| `dev_calc_sad` (375) | block SAD (Y + optional chroma) + reduce | search |
| `MinCost`, `dev_reduce_result` (479/490) | keep lowest-cost candidate, block reduce | search |
| `dev_expanding_search_1/2` (523/635) | square expanding diamond pattern cost | search |
| `dev_hex2_search_1` (715) | hexagonal (hex2) refinement pattern | search |
| `dev_read_pixels` (782) | shared-memory tile load for hex search | search |
| `Search` (860) | one block batch search driver kernel (uses all above) | search |
| `kl_calc_all_sad` (1135) | compute SAD across full refs (initial) | search/scene |
| `kl_prepare_search` (1267) | build per-block SearchBlock data | search |
| `kl_most_freq_mv` / `kl_mean_global_mv` (1383/1447) | global-MV estimation | search |
| `kl_interpolate_prediction` (1567) | coarse→fine MV upscale/prediction | search |
| `kl_load_mv_batch/load/store`, `kl_write_default_mv`, `kl_init_const_vec` | MV array I/O between levels | search |
| `kl_init_scene_change`, `kl_scene_change(_x2)` (1761/1767/1790) | per-frame scene change | search |
| `dev_degrain_weight`, `dev_norm_weights` (1843/1857) | per-block weight → 256 scale | degrain |
| `kl_prepare_degrain` (1930), `kl_degrain_2x3` (2021) | block-weighted temporal denoise | degrain |
| `kl_short_to_byte` (2157) | convert tmp to output | degrain |
| `kl_prepare_compensate` (2188), `kl_compensate_2x3` (2239) | motion compensation fetch | compensate |
| `kl_short_to_byte_or_copy_src` (2371) | compensate output | compensate |

## 4. CUDA → OpenCL equivalence rules (logical identity)

These make each kernel transliteratable while keeping **identical results**:

| CUDA device construct | OpenCL equivalent |
|---|---|
| `threadIdx/blockIdx/blockDim` (x,y,z), `gridDim.z` for A/B plane pairs | `get_local_id`, `get_group_id`, `get_local_size`, `get_global_id`; keep same indexing arithmetic |
| `__shared__ T s[N]` | `__local T s[N]`; same `barrier(CLK_LOCAL_MEM_FENCE)` for `__syncthreads` |
| warp shuffle reduce `dev_reduce_warp*` over integer SAD | reduction to `__local`, then a final partial by lane 0 — integer addition is order-independent ⇒ identical |
| `__vabsdiff4(a,b,sad)` / `__sad` | `sad += |aByte-bByte|` per pixel; the packed-4 / funnel-shift (`__funnelshift_rc`) / 4-byte-aligned load are *load* optimizations only — scalar per-pixel absolute difference yields the same sums |
| `atomicAdd(int* , int)` | `atomic_add(global int*, int)` (order irrelevant for integer sums) |
| `uchar4/ushort4/int4` packed channels | scalar loops over the 4 channels with identical per-channel math |
| `short2` | store as 2-element or packed via two ints in `int2` |
| `min/max/clamp`, `>>5` on non-negative int | identical in OpenCL C |
| `__restrict__` | `__restrict` |

Caveat for floats (rare here): keep *identical* operation order inside each
output (e.g. `dev_degrain_weight` truncates `(int)`, order is fixed).

## 5. Super-frame layout notes (needed before Search is meaningful)

`dev_get_ref_block` (NPEL 2 / 4) treats the reference as a set of **NPEL×NPEL
sub-pel planes** stacked `nImgPitch` apart: integer MV maps to plane offset
`si = sx + sy*NPEL`, then `x=vx/NPEL, y=vy/NPEL` into that plane. Porting the
`Search` kernels therefore requires the host to lay the super frame out with
`NPEL*NPEL` planes each `nPitch×nImgHeight`; this layout is constructed in
`MV.cpp` `KMSuper::Create` (the Wiener/RB2B kernels write these planes). Getting
the plane pitch/stride arithmetic exact is the crux of on-rig verification.

## 6. What has been ported / verified

In `src/opencl/ktgmc/kernels/ktgmc_simple.cl` (bit-for-bit CPU+Python verified,
8 & 16 bit): the two Wiener interpolation kernels (`kt_vertical_wiener`,
`kt_horizontal_wiener`), `kt_plane_sad` (frame SAD metric) and `kt_init_sad`
(SAD-buffer zeroing) plus the 22
per-plane KTGMC kernels.

`src/opencl/ktgmc/kernels/ktgmc_motion.cl` holds the self-contained MV pieces:
- **ALG-VERIFIED** (CPU + Python golden, `make test`): the degrain weight
  helpers `kt_degrain_weight`/`kt_norm_weights` (`run_motion_core.py`, 706
  cases); the MV-aux integer kernels `kt_write_default_mv`, `kt_scene_change`
  /`_x2`, `kt_short_to_byte`, `kt_short_to_byte_or_copy_src` (`run_mv_aux.py`,
  310 cases); and the coarse→fine MV upsampler `kt_interpolate_prediction`
  (`run_mv_interp.py`, 200 random cases, bilinear parity 4-neighbour weights);
  and the per-row global-MV refinement `kt_mean_global_mv` (average of vectors
  within 6 of the median estimate; `run_mv_mean.py`, 400 rowgroups — upstream
  does a 1024-thread staged/shuffle reduction, but the integer sums are
  order-independent so a serial per-row accumulation matches exactly).
  The per-block search setup `kt_prepare_search` (search bounds, predictor
  slot indices, prior-level vector copy, penalties and the lambda schedule;
  ANALYZE_SYNC=1) is ALG-VERIFIED (`run_mv_searchprep.py`, 300 cases;
  2026-10-28: prior-column read `p1` corrected to `blkIdx-1` — under
  ANALYZE_SYNC=1 a block's own region stays FRESH, see kernel comment and
  `python/run_mv_search.py`).
  The MV I/O quartet `kt_load_mv`, `kt_store_mv`, `kt_load_mv_batch` (the degrain
  / compensate per-batch split + out copy-through), `kt_init_const_vec` (split /
  recombine VECTOR int3 with the int2-vector + int-sad buffers; write the two
  per-row sentinels slot -2 = zero-vector, slot -1 = globalMV*nPel) is
  ALG-VERIFIED (`run_mv_io.py`, 200+60 cases).
  The reduced-plane builder `kt_rb2b_bilinear_filtered` (separable (1,3,3,1)/8
  anti-aliased 1:2 downsample, KMSuper `ReduceTo`) is ALG-VERIFIED
  (`run_mv_rb2b.py`, 200 cases): the single-pass OpenCL form recomputes the two
  separate phase roundings and matches the CPU `RB2BilinearFiltered` reference
  bit-for-bit.
  Its CUDA-only fused twin `kt_rb2b_bilinear_filtered_with_pad` — the same
  1:2 downsample done as ONE weighted 4×4-tap filter with a single +32/64
  rounding that also fills the destination hpad/vpad border (MV.cpp
  `ReduceToPad`; the CPU path instead runs the separable core + a separate
  `Pad()`, so the two forms are numerically distinct) — is ALG-VERIFIED
  (`run_mv_rb2b_pad.py`, 250 cases) against an independent C++ mirror that
  transliterates the fused CUDA kernel verbatim.
  Note: upstream `kl_write_default_mv` sets `.x` twice (a typo for `.sad`); we
  implement the intended default.
  The degrain/compensate OVERLAP pixel-combiner core — `kt_degrain_patch`
  (Degrain1to6_C weighted denoise of a block) + `kt_overlap_out` (feathered
  Overlaps_C window accumulation + Short2Bytes, per output pixel over its <= 2x2
  covering blocks, with edge source-copy) — is ALG-VERIFIED (`run_mv_degrain.py`,
  300 cases): its per-pixel summation matches a faithful MV.cpp-staging CPU
  mirror (`sim/ktgmc_degrain_ref.cpp`) bit-for-bit for 8/16-bit, overlap=0 and
  overlap=nBlkSize/2.  These are the only degrain/compensate kernels that are
  verifiable in-sandbox; they take the per-block reference-plane element offsets
  (refBaseB/F) as resolved inputs — that resolution is the host/rig seam (§6.1).
- **RIG-VERIFY** (faithful source ports, device run pending): the frame padding
  / mirror-copy kernels (`kt_copy_pad`, `kt_pad_frame_h/v`), `kt_init_scene_change`,
  `kt_most_freq_mv`, and the block-search pure helpers (`kt_clip_mv`/`kt_check_mv`/
  `kt_sq_norm`/`kt_ref_block_offset`). (`kt_calc_all_sad` has since GRADUATED to
  ALG-VERIFIED — its host layout was resolved from MVKernel.cu; see
  `docs/BLOCKSEARCH_MODEL.md` §8a and `python/run_mv_calc_all_sad.py`.) These follow
  the authoritative host model in `docs/BLOCKSEARCH_MODEL.md`. That doc records
  the key fact that the shipped CUDA launches compile with `CPU_EMU=true` — the
  search runs the deterministic, CPU-ordered scalar path (not warp shuffles), so
  a faithful scalar OpenCL port can reproduce it exactly.
  `kt_most_freq_mv` returns the most-frequent component using the staged-fold
  tie-break rule, now transcribed from the CUDA reduction tree and marked
  `// ALG-VERIFIED` (2026-10-28: CUDA-twin cross-check `rig/cuda_mv_tie_test.py`
  — 239 tie axes, CUDA vs OpenCL agree on 239/239, previously 93 divergences
  under the old smallest-mode guess; 8526/8526 rows overall). The earlier
  "smallest most-frequent component" rule was a guess that was bit-exact only
  when the mode is unique; when several values tie for the mode, CUDA's winner
  comes from its 1024-thread reduction tree, and the port now folds entries in
  the same staged order (first strictly-greater bin at stride 1024, then
  strict-`>` folds 512→32 and 16→1), reproducing it exactly.
All remaining kernels are **not** independently dispatchable/verifiable in this
sandbox because their inputs come from the `SearchBatchData` super-frame host
struct (built by the `MV.cpp` host state machine) and/or use warp/block
reduction geometry. Concretely, the unported set splits into:

- **Now ported:** `kl_calc_all_sad` (**ALG-VERIFIED**, §8a of
  `docs/BLOCKSEARCH_MODEL.md`) and the pure helpers (RIG-VERIFY)
  `dev_check_mv` / `dev_clip_mv` / `dev_sq_norm` / `dev_get_ref_block`
  (`kt_ref_block_offset`), which the block kernels share.
- **Intrinsic to the warp/block `Search` driver** (only meaningful inside the
  block kernel; cannot be validated standalone): `Search`, `dev_read_pixels`,
  `dev_expanding_search_1/2`, `dev_hex2_search_1`, `MinCost`,
  `dev_reduce_result`, `load4pix(_Aligned)`, and `kl_prepare_search`'s consumer
  loop. **(2026-10-28: this set is now ported and verified** — the whole
  `CPU_EMU=true` `Search` block loop lives in
  `kt_search_column` (`src/opencl/ktgmc/kernels/ktgmc_search.cl`), one
  work-item per (batch, column), host-enqueued in ascending column order on
  the in-order queue (the `prog`/`next` work-steal handshake is replaced by
  host serialisation; `docs/BLOCKSEARCH_MODEL.md` §8b/§8c). It is
  **ALG-VERIFIED** on the device: `rig/dev_ktgmc_search.py` chains
  `kt_prepare_search` → ordered `kt_search_column` → `kt_calc_all_sad` and
  passes a 48-case golden E2E (both PX) plus an 8-config three-way
  (device vs Python golden vs stage-1 CUDA twin dumps,
  `cuda_search/out/*_C_search.txt` / `*_F_calc_sad.txt`); CPU mirror
  `sim/ktgmc_search_ref.cpp` and golden runner `python/run_mv_search.py`
  (81 cases) pin the same arithmetic off-device. Sentinel slots are
  host-written in the E2E — `kt_init_const_vec` through a row-base sub view
  hits this driver's broken negative sub-relative offsets; its arithmetic
  itself is verified in `rig/dev_ktgmc_motion.py` S13. Note:
  `kt_prepare_search`'s prior-column read `p1` was corrected to
  `blkIdx-1` (ANALYZE_SYNC=1 leaves a block's own region FRESH) with a
  dated comment in the kernel, mirror and golden.)
- **Degrain / compensate block kernels** (need `DegrainBlockData`/`ArgData`
  super-frame pointers + MV arrays): `kl_prepare_degrain` and
  `kl_prepare_compensate` are ported as
  `kt_prepare_degrain` / `kt_prepare_compensate` in
  `src/opencl/ktgmc/kernels/ktgmc_motion.cl` (`// RIG-VERIFY`), with
  their arithmetic pinned by `python/run_mv_prepare.py` (240 cases, 8
  mutants) — see §6.3. `kl_degrain_2x3` and `kl_compensate_2x3` remain
  unported on purpose (§6.1).

### 6.1 Why the degrain / compensate per-pixel kernels stay rig-bound

The two `_prepare` kernels (`kl_prepare_degrain`, `kl_prepare_compensate`) are
per-block and self-contained in principle: for block (bx,by) they pick one of 9
overlap windows via `wby = 3*((by+nBlkY-3)/(nBlkY-2))`, `wbx = (bx+nBlkX-3)/(nBlkX-2)`,
slot `wby+wbx`; compute block origins `offx=bx*blkStep`, `offy=by*blkStep`
(`blkStep=nBlkSize/2`) and `offsetS = nPad+offx + (nPad+offy)*nPitchSuper`;
resolve each B/F ref to an element offset via the ALG-verified `dev_get_ref_block`
(= `kt_ref_block_offset`); and normalise `WSrc/WRefB/WRefF` via the ALG-verified
weight helpers. In MV.cpp the CPU reference (`KMDegrainCore::Proc`,
`KMCompensateCore::Proc`, overlap branch) instead spaces blocks at `StepX =
nBlkSizeX-nOverlapX` and reads `GetBlock(ix,iy).x = ix*StepX`, then maps an
output pixel to the super plane at `block.x*nPel + mv…` through
`KMPlane::GetPointer` (adds padding). So a *faithful device transliteration* must
decide which block-geometry / super-plane coordinate model it implements — the
MV.cpp `nPel`/padding/pel semantics (`nOverlap`, `nPel`, `nHPad/nVPad`,
`nLogxRatio/yRatio`) are the ground truth, and they are exactly the super-frame
host model this doc §5 flags as rig-only.

The per-pixel `kl_degrain_2x3` / `kl_compensate_2x3` kernels are additionally
defined by a **host launch geometry**, not by their own indexing: they accumulate
weighted block patches (`Degrain1to6_C`-style `>>8` denoise, then an
`Overlaps_C`-style `(px*winOver+256)>>6` overlap-add into a shared tmp, later
`Short2Bytes`-style `>>5`/`>>11`) but tile the plane via `basex =
(blockIdx.x*M+nPatternX)*SPAN_X`, `basey=(blockIdx.y*2+nPatternY)*SPAN_Y` with
`SPAN_X=3, SPAN_Y=2` and dispatch several `(nPatternX,nPatternY,M)` instances
from MV.cpp. Reproducing their exact global-tmp writes therefore requires that
host pattern/grid, plus the window generation (`OverlapWindows`, 9 feathered
windows of `nBlkSize*nBlkSize`), the block-geometry model of §6.1, and the
per-plane `pDst` tmp layout — i.e. the full on-rig host assembly, not a
self-contained kernel. We therefore record these as rig-bound rather than ship a
speculative scalar re-derivation. The arithmetic they need is now *not* missing:
the ALG-VERIFIED combiner core `kt_degrain_patch` + `kt_overlap_out` (§6, above)
is the order-equivalent, verified way to produce identical output once the host
supplies the per-block ref-plane base offsets and windows (see §6.2).
- **Separate KTGMC temporal-filter path** (Kernel.cu, not MV): `kl_init_sad`,
  `kl_calculate_sad`, `kl_copy_boarder1(_v)`, `kl_logic1/2/3`, `kl_box3_v`,
  `kl_box5_v_and_border`, `kl_binomial_temporal_soften_1/2` — several use packed
  `vpixel_t`/`__vabsdiff4` or float `atomicAdd` reductions whose output ordering
  is not bit-deterministic, so they are ported scalar / as flags only.

### 6.3 The two prepare kernels, ported (graduated from quarantine)

`kl_prepare_degrain` (:1884) and `kl_prepare_compensate` (:2137) were
transcribed in the quarantined `ktgmc_degrain_rig.cl` and are now graduated:
the kernels moved byte-identical into `src/opencl/ktgmc/kernels/ktgmc_motion.cl`
(mirroring the KFM Deblock graduation, commit ecfcca3) and the quarantine
file was deleted. They remain `// RIG-VERIFY` rather than ALG-VERIFIED
because §6.1's block-geometry question is a property of the CALL SITE, not
of the kernel: the transcription follows the CUDA model
(`blkStep = nBlkSize/2`, origin `nPad + blk*blkStep`) and only a rig can
confirm that is what the host means.

What the port changes, deliberately, is the output encoding. Upstream fills
a struct of raw POINTERS; OpenCL cannot store those portably and the
ALG-VERIFIED consumer `kt_degrain_patch` does not want them — it takes flat
per-block element offsets and weight arrays. These kernels emit exactly the
arrays that consumer declares, which is what §6.2 below calls the remaining
seam. Two encodings need host cooperation, both provably output-equivalent:

- **Unusable reference**: upstream parks the pointer on `arg.pSrc` (another
  plane, another pitch) with weight 0, noting the values are never used.
  The port emits offset 0 with weight 0 — different pixels, identical
  output, since `kt_degrain_patch` multiplies by that zero.
- **Scene change** (compensate): upstream writes `nullptr` for `winOver`
  and `pRef`; the port writes the sentinel `-1` to `win_slot`/`ref_base`
  and `ref_sel`, which the host must treat as "skip", exactly as a null
  check would.

Two upstream details the pin protects and a rewrite would lose:

- The 9-window slot divides by `(nBlkX-2)`/`(nBlkY-2)`, so a 2-block-wide
  or 2-block-tall grid divides by zero **upstream too**. The port does not
  paper over it; the host must not call with `nBlkX == 2` or `nBlkY == 2`.
- Compensate scales the MV as `(vec.x * time256 / 256) >> SHIFT`: a C
  division that truncates toward zero followed by an arithmetic shift that
  floors. For negative MVs those are different maps, and a mutant that
  "simplifies" the division into a shift is caught by the runner.

### 6.2 The one seam between verified kernels and the rig

The degrain/compensate combiner math is verified; the *only* thing left to wire
is turning the MV.cpp `KMPlane`/`GetPointer` reference semantics into the flat
per-block element offsets (`refBaseB/refBaseF`, indexed `k*nBlk+blk`) that
`kt_degrain_patch` consumes, plus the host-side 9-window generation. Concretely:
`KMPlane::SetTarget` stacks `nPel*nPel` sub-pel planes `nPitch*nExtendedHeight`
apart; `GetPointer(nX,nY)` folds the low `log2(nPel)` bits of `nX,nY` into the
plane index, adds `nHPadPel/nVPadPel`, and the block base is
`block.x*nPel + mv[k][blk].x` (block.x = `bx*StepX`, StepX = `nBlkSize-overlap`).
That offset maps directly onto the verified `kt_ref_block_offset` helper. A
concise step-by-step recipe for the follow-up agent / rig lives in
`docs/CODEX_HANDOFF.md`.

So the isolated, per-output deterministic MV kernels are complete
(24 in `ktgmc_motion.cl` — including the graduated `kt_prepare_degrain` /
`kt_prepare_compensate` pair — plus 25 in `ktgmc_simple.cl`), and `kt_calc_all_sad`
extends the set into the first block-level (MV + super-frame) kernel.
Finishing KTGMC from here means assembling the `SearchBatchData` + super-frame
layout on a real rig (see `docs/HOST_CONTRACT.md`) and then porting/validating
the remaining groups above there.

## 7. Verification plan on a real rig

1. Bring up an OpenCL build (see top-level CMakeLists `OpenCL::OpenCL` block;
   install `ocl-icd-opencl-dev` + a device, e.g. `pocl-opencl-icd` or vendor
   runtime) and compile `ktgmc_simple.cl`/`ktgmc_motion.cl`.
2. First run `make lint` (or CTest `ktgmc_cl_lint`): it does a genuine C parse of
   every `.cl` through `lint/oc_shim.h` to catch structural/typo errors. gcc does
   NOT parse a `.cl` file on its own (it treats it as a linker input), so the
   script copies each source to `.c` and uses `-x c`. The shim only emulates
   syntax (vector `.x/.y` members, work-item ids, `atomic_add`, scalar builtins),
   not OpenCL semantics.
3. Add a host runner that applies each kernel to raw planes and diffs against the
   existing `sim/ktgmc_cpu_ref.cpp` output (the same vectors already used by
   `python/run_validation.py`). This validates every kernel not tied to the MV
   search data model. See `docs/HOST_CONTRACT.md` for the exact program builds,
   kernel→grid maps, argument order and — importantly — the `int2`/`int3` host
   buffer-layout probe to run before exercising the MV kernels.
4. For the search/degrain/compensate kernels, build the `MV.cpp` host state
   machine (predictors, meander, level loop) and compare `VECTOR` arrays + output
   frames against the CUDA build of `AviSynthCUDAFilters` on identical inputs
   (or against `mvtools` output where the parameterization matches).
5. Remove each `RIG-VERIFY` marker as its test passes.

## 8. Licensing

`KTGMC` (incl. the MV engine) is **GPL** per the upstream `AviSynthCUDAFilters`
README; `common/KMV.h` type defs come from the same tree. Distribute derived
files under the matching upstream terms.

## 9. 2026-11-01: KMAnalyse multi-level host loop PROVEN (measurement slice)

The `GroupOfPlanes` coarse→fine loop that KMAnalyse runs around the (already
ALG-VERIFIED) Search quartet has been settled on the rig — evidence-first, no
plugin written yet:

- Contract: `cuda_mv/KMANALYSE_HOST_CONTRACT.md` (level order coarse→fine,
  per-level searchType/nPel/nLambdaLevel forcing MV.cpp:2250-2261, the
  global-MV chain `kl_most_freq_mv`→`kl_mean_global_mv`×2→`slot −1 =
  globalMV·nPel` MVKernel.cu:2664-2673/1708-1717, `kl_interpolate_prediction`
  using the x-axis weights for both axes MV.cpp:2110-2118, BGR32 VECTOR frame
  coarsest-level-first MV.cpp:2664-2668/3838-3842, scene change NOT part of
  Analyse, per-level padding model).
- Measurement: `cuda_mv/KMANALYSE_MEASURE_REPORT.md` — CUDA twin
  (`cuda_mv/kmanalyse_host.cu`, verbatim MVKernel.cu 269..1627) vs the port's
  OpenCL chain (`rig/dev_ktgmc_analyse.py`) is **bit-exact at every stage of
  every level** for pel=1/pel=2 × chroma on/off (39/39 checks, incl. the
  final MV frame). A twin bug caught en route (level-0 init wiping the
  interpolated predictors) is exactly the plausible-but-wrong-MV failure
  mode; the golden adjudication and fix are recorded in the report §3.
- Consequence: a KMAnalyse host plugin needs NO new kernels — it is the
  proven one-Search assembly (§6, Program B/C) plus
  `kt_most_freq_mv`/`kt_mean_global_mv`/`kt_interpolate_prediction`
  (all ALG-VERIFIED) driven per level per §10 of the contract. Remaining
  plugin-level decisions (undefined coarsest globalMV, `WriteDefault` typo
  handling, ≥3 levels / batch>1 / partial-input extensions) are listed in
  the measure report §5.
