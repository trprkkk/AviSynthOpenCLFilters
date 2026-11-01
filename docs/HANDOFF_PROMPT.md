# Handoff prompt — paste this to the next agent

Two versions below. **A** is the full prompt (use this by default). **B** is a
short form for a quick continuation in the same context.

Both deliberately point at documents instead of restating them: the docs are
the source of truth and are machine-checked by `lint/audit_inventory.py`.

---

## A. Full handoff prompt

> You are continuing a port of `rigaya/AviSynthCUDAFilters` (CUDA, GPL) to
> OpenCL. The repository is `trprkkk/AviSynthOpenCLFilters`. The previous agent's
> work is finished and committed; nothing is half-done in the tree.
>
> **Read these three documents before touching anything, in this order:**
> 1. `docs/RIG_HANDOFF_BRINGUP.md` — the entry point. §0 is the inventory,
>    §2 is the five findings you must act on, §2.7 defines "done", §3 is the
>    suggested order, §5 maps every verified area to its runner.
> 2. `README.md` — the method and the per-family status table.
> 3. The spec for whatever family you touch (`docs/*_PORT_SPEC.md`).
>
> **Current state (verify, don't trust):** run `make test` and
> `./lint/lint_opencl.sh`. You should see 33 PASS / 0 FAIL, 46 lint checks /
> 0 failures, and `audit: OK`. 168 OpenCL kernels, 32 runners, 32 CPU
> mirrors. If those numbers differ, the audit will tell you what moved.
>
> **What is already proven — do not redo it.** Every kernel marked
> `// ALG-VERIFIED` has a scalar CPU mirror in `sim/` plus an *independent*
> Python golden in `python/`, agreeing bit-exactly, mutation-tested. If a
> device run ever disagrees with a mirror, suspect your harness, the launch
> geometry, or a compiler flag first — the mirror+golden pair is the more
> trustworthy artifact.
>
> **The method is not negotiable. A kernel graduates only with BOTH:**
> - a scalar CPU mirror in `sim/`, built `-ffp-contract=off`; and
> - an *independent* Python golden in `python/run_*.py`, wired into
>   `make test`. "Independent" means derived from the upstream semantics by
>   a different route — not a transcription of your own port. Several real
>   bugs in this repo were caught only because the two disagreed.
>
> Then **mutation-test the proof, not just the port**: deliberately break the
> mirror in ~6 ways and confirm the runner catches each. If a mutant
> survives, either your test is inadequate (fix it) or the mutant is
> genuinely equivalent (prove it and write the proof down). The cautionary
> tale is in `docs/NNEDI3_PORT_SPEC.md` §5: comparing only integer pixel
> output let four mutants live, and two of the bugs were in the *golden*.
>
> **Rules that have already cost time when broken:**
> - Buffer ABI is not inferable from kernel arithmetic. Upstream `short2` is
>   4 bytes and `VECTOR{int x,y,sad}` is a packed 12; OpenCL `int2`/`int3`
>   are 8 and 16. Treat every 3-component buffer as a packed triple and take
>   `docs/HOST_CONTRACT.md` as the ABI contract.
> - Upstream defects are transcribed and pinned with assertions, never
>   silently corrected (see the masktools 16-bit LUT collapse,
>   `docs/MASKTOOLS_PORT_SPEC.md` §3). Fixing one is a decision to record,
>   not a cleanup to perform.
> - Where float order is observable (reduction trees, `dev_expf`), reproduce
>   it exactly; do not "simplify" to a sequential sum or a library `exp`.
> - Unverifiable-by-construction kernels go in a quarantined `*_rig.cl` with
>   `// RIG-VERIFY`, never into a verified file. Three such files exist.
>
> **Pick your track:**
> - *No GPU (sandbox only):* the portable work is essentially exhausted.
>   What remains is `kl_search` (blocked on two host decisions —
>   `docs/BLOCKSEARCH_MODEL.md` §8b) and `kl_degrain_2x3` /
>   `kl_compensate_2x3` (defined by a host launch pattern). Do **not** guess
>   either: a wrong predictor layout yields plausible-but-wrong motion
>   vectors, the worst failure mode here. Prefer improving coverage of
>   existing pins, or say so and stop.
> - *With a GPU:* follow `docs/RIG_HANDOFF_BRINGUP.md` §3 — toolchain smoke
>   test, then masktools (5 elementwise kernels, no float: the cheapest
>   harness shakedown), then NNEDI3, then the bulk, then the two package
>   handoffs (`RIG_HANDOFF_KDEBLOCK.md`, `RIG_HANDOFF_AVSCUDA_CONDITIONAL.md`).
>
> **Definition of done** is in `docs/RIG_HANDOFF_BRINGUP.md` §2.7. The rule
> that trips people: a green device run closes the *external* fact only; the
> mirror+golden pair closes the *arithmetic*. A successful device run alone
> does not graduate a kernel to `ALG-VERIFIED`.
>
> **Operational notes:** `make test` and the lint are hermetic (no network,
> no GPU, ~100 s). Upstream is pinned at `68aef6e` with submodules `NNEDI3`
> @`01931aa` and `masktools` @`24ba826`; clone it to `/tmp` when you need to
> read source — it is volatile, re-clone freely. Before every commit, check
> `git log --oneline -1` and `git status`: this workspace has had its `.git`
> reset to the initial commit mid-session more than once. Recovery is
> `git fetch origin <your-branch> && git reset FETCH_HEAD`, which keeps the
> working tree and re-attaches your work.
>
> Report what you changed, what you proved, and what you deliberately did
> not do — the last one matters as much as the first.

---

## B. Short form

> Continuing the OpenCL port of `rigaya/AviSynthCUDAFilters` in
> `trprkkk/AviSynthOpenCLFilters`. Start at `docs/RIG_HANDOFF_BRINGUP.md` (§2 =
> findings to act on, §2.7 = definition of done, §3 = order). Verify state
> with `make test` (expect 33 PASS, `audit: OK`) and `./lint/lint_opencl.sh`
> (46/0).
>
> Non-negotiable method: a kernel graduates only with a `-ffp-contract=off`
> CPU mirror in `sim/` **and** an independent Python golden in `python/`
> wired into `make test`; then mutation-test the proof and record any
> equivalent mutants. Transcribe upstream defects and pin them rather than
> fixing them. Unverifiable kernels go in a quarantined `*_rig.cl` with
> `// RIG-VERIFY`. Don't guess `kl_search`'s predictor layout.
>
> Check `git log -1` and `git status` before committing — this workspace's
> `.git` has been reset mid-session before; recover with
> `git fetch origin <branch> && git reset FETCH_HEAD`.

---

## Why the prompt is shaped this way

- **Points at docs, does not restate them.** Restated facts rot; the docs are
  audited by `make test`, so a prompt that defers to them cannot go stale.
- **Leads with "verify, don't trust".** The numbers in §0 are checked
  automatically, but the next agent should still run the suite first — it is
  cheap and it catches a broken environment immediately.
- **States what NOT to redo, and what not to guess.** Both are failure modes
  that cost real time in this project: re-deriving verified arithmetic, and
  speculating on host layouts that only a rig can settle.
- **Makes the proof, not the port, the object of scrutiny.** The single most
  valuable lesson from this work is that a passing test can be wrong; the
  prompt encodes mutation testing as a requirement rather than a nicety.
- **Carries the environment hazard.** The `.git` reset is not in any spec
  document because it is not a property of the code, but an agent that loses
  a commit to it wastes a cycle rediscovering the recovery recipe.
