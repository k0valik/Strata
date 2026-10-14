# Strata (fork)

A pruned fork of [Strata](https://github.com/Niko1221/Strata) (engine 0.1.44.1): the C++/CUDA/HIP
inference engine for Qwen3.8-Flash-Next and Qwen3.6-35B-A3B mixture-of-experts models, kept
buildable and close to upstream so it can follow upstream later.

The upstream installer, Python server, web app, Intel SYCL and Apple Metal ports, Docker packaging,
benchmark-result archives and user documentation are removed. See [FORK_NOTES.md](FORK_NOTES.md)
for exactly what was removed, what is kept, and what is deferred.

## Build

Requires CMake 3.24+, a C++20 compiler, and a CUDA toolkit (or AMD HIP, or CPU-only). ggml is taken
from llama.cpp at the pinned commit in `third_party/ggml/VERSION.txt` (fetched by CMake; a vendored
submodule replaces this in a later phase):

```
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DSTRATA_ENABLE_CUDA=ON -DSTRATA_BUILD_TESTS=ON
cmake --build build -j
ctest --test-dir build
```

## Layout

- `src/`, `include/` — the engine (core, kernels, prefill, speculative decoding, plan).
- `tests/` — engine tests (built when `STRATA_BUILD_TESTS=ON`).
- `tools/` — model preparation (pack, draft head, expert profiles) and small parity helpers.
- `data/` — small runtime data (draft vocabulary, expert profiles).
- `ref/` — reference GGUF loader used to validate the engine's reading of GGUF.
- `bench/micro/` — micro-benchmarks and parity harnesses referenced by the build.

## Status

There is no serving/API layer yet; the engine is driven through its CLI binaries
(`strata`, `strata-plan`, `strata-q36`, ...). A llama.cpp-based frontend is planned as a
separate phase (see FORK_NOTES.md).
