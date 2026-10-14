# FORK_NOTES

Engineering ledger for this fork. Factual, updated as the fork diverges.

## Baseline

- Upstream Strata: `Niko1221/Strata` (a CUDA/HIP engine for Qwen3.8-Flash-Next / Qwen3.6-35B-A3B MoE
  models + a Python installer/server/web app). This fork derives from upstream commit
  `296b59d2` ("Version 0.1.44.1", engine version 0.1.44.1).
- Reference inspected: `UnbiasedStrata` at commit `b124b59` (its own fork of upstream engine 0.1.39,
  with a C++ llama.cpp-based frontend). Reference only; not copied byte-for-byte.

## Target hardware / intent

The fork exists to run Strata's inference engine on the maintainer's PC: one 16 GB NVIDIA GPU,
32 GB DDR4. It must stay buildable and recognizably upstream so future synchronization stays cheap.

## Intended retained surface

- The engine itself: `src/`, `include/`, `cmake/`, `CMakeLists.txt`, `third_party/ggml` headers,
  `ref/`, `tests/`, engine parity/bench sources in `bench/micro/` and `bench/gfx906_*`.
- Engine build backends kept as-is (same sources, backend chosen at configure time): CUDA, HIP,
  CPU (AVX2/AVX-512/AVX1 floors). Windows source paths kept (in-source, inseparable without a rewrite).
- ggml/llama.cpp dependency: **vendored as a git submodule** `third_party/llama.cpp`, pinned at
  tag `b11624` = commit `af0579c78a07a555d84830d01da7b61cfadbc7ef` (ggml-org/llama.cpp, 2026-10;
  chosen by the maintainer, newer than upstream Strata's pin). CMake takes ggml, the MMQ kernel
  sources and `ggml-common.h` from it (`STRATA_GGML_DIR`, overridable with `-DSTRATA_GGML_DIR`);
  the build fetches nothing and fails with a `git submodule update --init` hint when the submodule
  is not checked out. `.gitmodules` is marked `shallow = true`.
- Model preparation tools (`tools/`): the pack/draft/profile pipeline (`iq_pack`, `mtp_*`,
  `make_profile`, `pack_*`, `strata_pack`, `strata_tokenizer`, `draft_vocab`, `gguf_*`,
  `embd_bf16_pack`, `ple_fp8_pack`, `nvfp4_*`, `quant_formats`) and their unit tests.
- `tools/vision/` (the `strata-vision` mmproj encoder binary, standalone CMake against llama.cpp),
  `tools/qwen35/` parity helpers, `tools/hip/` HIPBLASLt tuning tables + `tune_hipblaslt`,
  `tools/check_isa_guard.py`.
- Runtime data: `data/draft_vocab.bin`, `data/draft_vocab_en.bin`, `data/expert-profile.bin`,
  `data/expert-profile-coder.bin`.
- The engine's `strata --serve` command mode is retained in source (it is the engine's process
  interface; a future HTTP frontend will drive it the same way).

## Removed (phase 1 pruning commit)

- Installer/bootstrap: `setup.py`, `setup.sh`, `START-HERE.bat`, `SETUP.bat`, `UPDATE.bat`,
  `BENCH.bat`, `update.sh`, `bench.sh`, `requirements.txt`, `flake.nix`, `flake.lock` — the
  one-click download/install machinery is out of scope for a controlled local build.
- Python serving stack: `serve/` (OpenAI/Anthropic server, web app, MCP server, prometheus
  telemetry, replicas, its test suite) and `chat.py` — replaced later by a llama.cpp-based
  frontend (see Deferred). The engine CLI (`strata`, `strata-plan`, `strata-q36`, ...) remains.
- Platform ports as separate trees: `sycl/` (experimental Intel port, upstream marks it
  EXPERIMENTAL; its `expert-profile-sycl.bin` removed too) and `metal/` (Apple Silicon port,
  standalone CMake, macOS-only). The `STRATA_ENABLE_SYCL` option block was removed from
  `CMakeLists.txt`. CUDA/HIP stay.
- Distribution packaging: `Dockerfile`, `.dockerignore`, `docker-entrypoint.sh`, `.github/`
  (issue templates, funding), `SECURITY.md`.
- Docs: `docs/` entirely (installer/user/community/platform docs, benchmarks, media, paper) and
  the translated `README.*.md`; `README.md` and `AGENTS.md` rewritten for the fork.
- Benchmark results and research data: `bench/results/` (~2k files, 23 MB of historical runs),
  `bench/bench_vs_llama.py` + `bench/profile_corpus.json` (both drive the removed server),
  `data/quant_quality/`, `data/experimental-speed-projection/`, `data/gguf_fingerprints.json`
  (download-verification pins for the removed installer).
- Tools tied to removed systems: serving/MCP/bench/research tools (`strata_mcp`, `strata_bench`,
  `strata_inspect`, `serve_load`, `ab_engine`, `calibrate`, `canonical_xcheck`, `research_run`,
  `conversation_cache_*`, `*_bench.py` probes, `quant_bound`, `quant_quality`, `quantscope`),
  all `test_setup_*` and their fixtures, `tools/image/` (serve-side image generation/preprocessing),
  `tools/hip/{build_windows.bat,dll_closure.py,package_windows.py}` (Windows HIP packaging),
  `tools/split_mtp_batch_parity.py` (HTTP against the removed server).
- Data variants tied to the installer's language menu: `data/draft_vocab_cyrillic.bin`,
  `data/draft_vocab_fr.bin`.

## Divergences from upstream

- Deletions listed above; no engine source (`src/`, `include/`) was modified.
- **llama.cpp vendoring** (vs upstream's build-time `FetchContent` of `3cf0325...`): submodule
  `third_party/llama.cpp` pinned at `af0579c78a07a555d84830d01da7b61cfadbc7ef` (tag `b11624`),
  checked out locally; the submodule is pristine (no patches applied to it). The old
  `third_party/ggml/` header snapshot (a verbatim `ggml-common.h` from `3cf0325`) was deleted; the
  engine includes `ggml-common.h` from the submodule's `ggml/src`, so header and library always
  come from the same commit. Diffed old-vs-new pin: `ggml-common.h`, `ggml.h`, `ggml-cpu.h`, the
  context/pool structs and `quantize_mmq_q8_1_cuda` are unchanged; the only API break was in
  `mmq.cuh` (see next point).
- **MMQ call-site adaptation** (`src/prefill/moe_mmq.cu`, `mmq_kq_lib.cu`, `mmq_nvfp4_w4a8.cu`,
  `mmq_nvfp4_w4a4.cu`): `mul_mat_q_case` takes a second template parameter (the activations'
  precision; Q8 everywhere, Q4 only for the W4A4 NVFP4 unit), `mmq_args`' `ncols_opt` became a
  required `J_best` tile width (chosen by the same config probe llama.cpp's own launcher makes),
  and `ggml_cuda_mmq_get_config` takes the precision. Upstream Strata still compiles against the
  OLD (3cf0325) MMQ API — when porting future upstream changes to these four files, re-apply this
  adaptation on top of their text. The submodule itself is deliberately not patched (unlike
  UnbiasedStrata, whose one patch exists because its frontend needs a fix inside llama.cpp's chat
  parser, code we do not use yet).
- **Upstream synchronization infrastructure** (`.fork/`, `.github/workflows/sync-upstream.yml`): a
  scheduled workflow prepares a rolling `sync/upstream` pull request. The accepted upstream
  revision lives in `.fork/upstream.lock` and is passed to git as the *merge base* of the content
  merge (`git merge-tree --write-tree --merge-base=<lock>`), so the fork's own deletions cannot be
  silently undone and a **squash merge** of the sync pull request still advances the baseline
  correctly (git's own merge base goes stale after a squash; the lock file does not). Path rules
  live in `.fork/path-policy` (`mirror`/`dropped`/`ours`/`protected`, plus the built-in `pruned`
  rule for everything the fork deleted). The candidate also records the upstream tip as its second
  parent, so merging with a **merge commit** keeps `git merge-base main upstream/main` — and
  GitHub's "N commits behind" — accurate. That second parent is an assertion that the whole
  range was accounted for, so it is only ever attached after every changed path was classified
  and resolved: any protected/unclassified path or mirrored conflict stops the run before a
  commit object exists. Candidates are self-describing (`Fork-Sync-Base:`/`Fork-Sync-Upstream:`
  trailers checked against the actual parents, plus the lock inside the candidate matching its
  second parent), which is also how the rolling branch is proved to be automation-owned before
  it is replaced. Nothing from upstream is ever executed while synchronizing. See
  `.fork/README.md`.
- `.gitignore`/`.gitattributes`/`README.md`/`AGENTS.md` trimmed to the retained tree.
- Deliberately NOT copied from UnbiasedStrata: its C++ `frontend/`, its engine-source patches
  (`robust_read.hpp`, `expert_source.cpp`, `ngram.cpp` changes), its llama.cpp submodule/patch
  machinery, its older-CPU path removals. Those are its own reworkings against engine 0.1.39.

## Active source divergences

Path-indexed on purpose: the synchronization report flags any of these paths when upstream touches
them, and this is where the reason for that flag lives. Whenever a new patch is carried on a
mirrored file, add an entry here. (Derivable facts live in the tree; "why" only exists here.)

### D001 — vendored llama.cpp (the fork's own dependency integration)

- Paths: `CMakeLists.txt` (upstream's `FetchContent` block replaced by the submodule),
  `tools/_paths.py` (one comment: `gguf-py` comes from the vendored submodule, not `setup.py`).
  Additionally `.gitmodules` and the `third_party/llama.cpp` gitlink, both `protected` by the
  path policy rather than mirrored.
- Introduced: `f4343dd9` (vendoring).
- Why: this fork pins llama.cpp itself (`b11624` = `af0579c`, newer than upstream's `3cf0325`) and
  builds ggml from that checkout instead of letting the build fetch a tag.
- Expected when upstream touches it: `CMakeLists.txt` changes surface as a merge conflict and stop
  the run, because the fork's edits occupy the same region as upstream's dependency pin. That is
  the intended failure mode — do not "fix" it by classifying the file `ours`, which would discard
  upstream's other build changes.
- Retire when: only if upstream adopts a pinned dependency the fork can follow instead.

### D002 — MMQ call-site adaptation for the vendored llama.cpp

- Paths: `src/prefill/moe_mmq.cu`, `src/prefill/mmq_kq_lib.cu`, `src/prefill/mmq_nvfp4_w4a8.cu`,
  `src/prefill/mmq_nvfp4_w4a4.cu`.
- Introduced: `02ed2cda`.
- Why: llama.cpp `b11624` changed `mul_mat_q_case` (second template parameter = the activations'
  precision), `mmq_args` (`ncols_opt` became the required `J_best` tile width) and
  `ggml_cuda_mmq_get_config` (takes the precision), while upstream Strata still compiles against
  the old `3cf0325` API. These four files are the fork's porting cost of the newer pin; the
  submodule itself stays unpatched. See "Divergences from upstream" for the recipe.
- Expected when upstream touches it: a clean textual merge *only* if the edits are on different
  lines, so the flag means "read the merged file" rather than "it broke".
- Retire when: upstream Strata moves to the same or a newer llama.cpp API, which makes these four
  files pristine mirrors again.

## Deferred

- **Serving/tokenizer side of llama.cpp**: the vendored submodule supplies ggml, the MMQ kernel
  sources and `ggml-common.h` only. Borrowing more from it (tokenizer/chat parsing for a future
  frontend, as UnbiasedStrata does) is a later phase; the submodule stays pristine until then.
- **Serving/API**: no HTTP/OpenAI endpoint exists until the llama.cpp-based frontend lands. Until
  then the engine is driven via its CLI/token-ids interface.
- **GitHub Actions / CI**: only the upstream-synchronization workflow exists
  (`.github/workflows/sync-upstream.yml`; roughly every third day, plus manual dispatch). Engine
  build/test CI is still absent on purpose — a synchronization candidate must be built and tested
  by hand before its pull request is merged. The workflow needs the repository setting
  *Settings -> Actions -> General -> "Allow GitHub Actions to create and approve pull requests"*.
- **Porting upstream changes**: keep following upstream through the tooling in `.fork/` (see
  "Divergences from upstream" and `.fork/README.md`). Do **not** `git merge upstream/main` by hand:
  the content merge needs `.fork/upstream.lock` as its base and the path policy applied on top.
  Engine sources stay untouched, so `git diff upstream/main -- src include` remains meaningful, and
  deleted add-on trees show as ours-only removals. The vendored submodule may need re-pinning when
  an upstream port requires a newer or older llama.cpp.
- **Multi-GPU and the vision encoder** (`tools/vision`, engine mmproj path) are planned features,
  not leftovers. Windows: the maintainer works from WSL2 but wants native Windows builds to keep
  working (MSVC paths, `include/strata/windows/`, `tools/windows/` are retained for that) because
  serving from WSL has latency problems — treat the Windows build as supported,
  verification-pending.
- **Upstream sync**: engine sources are untouched, so `git diff upstream/main -- src include` stays
  meaningful; deleted add-on trees will show as ours-only removals. Local patches to mirrored
  files are listed under "Active source divergences" above.

## Uncertainties

- The engine is compiled against a llama.cpp commit **newer than upstream Strata pins**, with the
  MMQ call-site adaptation above carried in-tree. Verification still to run: configure + build
  `STRATA_ENABLE_CUDA` with the submodule checked out, then `ctest` (in particular
  `prefill_fused_moe_test`, `prefill_fused_iq_test`, `prefill_mmq_kquant_test`,
  `mmq_nvfp4_parity`, the CPU parity tests, and the HIP MMQ parity tests on AMD hardware).
- Whether `tools/nvfp4_verify*.py` (import `torch`) are wanted long-term; kept as model-prep
  verification tooling for the retained NVFP4 engine path.
- `tools/windows/enable_large_pages.ps1` kept as a tiny runtime helper for the retained Windows
  engine paths.
- Engine `--serve` mode and vision encoder are kept in source although nothing in-tree launches
  them anymore until the frontend phase.
- HIP/AMD, Windows, older-CPU (ISA floor) and gfx906 paths are retained for upstream parity and
  untested here.
- The upstream-synchronization machinery is exercised by `.fork/tests/run.sh` (153 checks over
  miniature repositories: no-op, clean ranges, dropped/protected/ours/unclassified paths, rewritten
  history, merge conflicts, squash vs merge commit, rolling-branch ownership, a patched mirrored
  file merging cleanly, and non-leaking working directories) plus `.fork/scripts/validate-sync.sh`
  (59 checks against this repository). A real dry run reports "up to date" because upstream `main`
  currently equals this fork's baseline `296b59d2`. A 300-commit window of real upstream history
  was classified to audit the path policy: 1804 paths, 1542 dropped / 254 mirror / 8 ours, none
  unclassified. The fork's own divergences on mirrored paths are `CMakeLists.txt`, the four MMQ
  call sites (D001/D002 above) and `tools/_paths.py`.
- **Not yet exercised**: the GitHub workflow itself (no run yet — the first scheduled or manual
  dispatch is the first real test of `actions/checkout` + push + `gh pr create`), and a real
  synchronization of a non-empty upstream range with the real remote. A *stale* lock was simulated
  against real history to confirm it fails closed (it conflicts on `CMakeLists.txt`, where the
  fork's vendoring edits meet upstream's own ggml integration pin).
