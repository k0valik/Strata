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
- ggml/llama.cpp dependency: still fetched by CMake `FetchContent` at the pinned commit
  (`third_party/ggml/VERSION.txt`) — to be replaced by a vendored submodule in a later phase.
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

- Deletions above; `CMakeLists.txt` lost only the SYCL option block; no engine source
  (`src/`, `include/`) was modified.
- `.gitignore`/`.gitattributes`/`README.md`/`AGENTS.md` trimmed to the retained tree.
- Deliberately NOT copied from UnbiasedStrata: its C++ `frontend/`, its engine-source patches
  (`robust_read.hpp`, `expert_source.cpp`, `ngram.cpp` changes), its llama.cpp submodule/patch
  machinery, its older-CPU path removals. Those are its own reworkings against engine 0.1.39.

## Deferred

- **llama.cpp vendoring**: replace the CMake `FetchContent` of the pinned llama.cpp commit with a
  vendored `third_party/llama.cpp` submodule (and bring the serving/tokenizer side from it, as
  UnbiasedStrata does). Dedicated later phase; until then the build still fetches the pinned commit.
  Target pin chosen by the maintainer: `a0e6dc7384edb8f5d927155e1d26c10b4a310e6d`
  (ggml-org/llama.cpp, 2026-10). Current pin in this tree: `3cf03257f219afbe7334045ff7c6a06ac68c627d`
  (`third_party/ggml/VERSION.txt` and the `GIT_TAG` in `CMakeLists.txt`; both must move together).
  Note: the new pin changes ggml behaviour the engine's parity tests compare against — after
  vendoring, run the CUDA/CPU parity tests and re-check the i-quant/Q2_0 contracts before trusting
  results.
- **Serving/API**: no HTTP/OpenAI endpoint exists after this phase until the llama.cpp-based
  frontend lands. Until then the engine is driven via its CLI/token-ids interface.
- **GitHub Actions / CI**: intentionally not set up in this phase.
- **HIP/AMD, Windows, multi-GPU, older-CPU (ISA floor), gfx906**: retained in source/build options
  for upstream parity. Multi-GPU and the vision encoder (`tools/vision`, engine mmproj path) are
  planned features, not leftovers. Windows: the maintainer works from WSL2 but wants native Windows
  builds to keep working (MSVC paths, `include/strata/windows/`, `tools/windows/` are retained for
  that) because serving from WSL has latency problems — treat the Windows build as supported,
  verification-pending.
- **Upstream sync**: engine sources are untouched, so `git diff upstream/main -- src include` stays
  meaningful; deleted add-on trees will show as ours-only removals.

## Uncertainties

- Whether `tools/nvfp4_verify*.py` (import `torch`) are wanted long-term; kept as model-prep
  verification tooling for the retained NVFP4 engine path.
- `tools/windows/enable_large_pages.ps1` kept as a tiny runtime helper for the retained Windows
  engine paths.
- Engine `--serve` mode and vision encoder are kept in source although nothing in-tree launches
  them anymore until the frontend phase.
