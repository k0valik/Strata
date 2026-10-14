# AGENTS.md (fork)

This is the pruned fork of Strata described in [FORK_NOTES.md](FORK_NOTES.md). Read that file
first: it records the upstream baseline, what was removed, what is retained, and what is deferred.

- The engine lives in `src/` and `include/`; keep it byte-identical to upstream where possible so
  `git diff upstream/main -- src include` stays meaningful.
- Do not reintroduce the removed installer/server/web machinery, the Intel SYCL port, the Apple
  Metal port, or benchmark archives. The serving/API layer will come from llama.cpp in a later
  phase.
- Model preparation goes through `tools/` (`iq_pack`, `mtp_*`, `make_profile`, ...); their unit
  tests run without a GPU or downloads: `python -m unittest tools.test_<name>`.
- Build: `cmake -S . -B build -DSTRATA_ENABLE_CUDA=ON -DSTRATA_BUILD_TESTS=ON` then
  `cmake --build build`. ggml comes from llama.cpp at the pinned commit (CMake FetchContent until
  the vendoring phase).
