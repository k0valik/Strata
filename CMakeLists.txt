# Strata Engine - top-level build (P1.S1).
#
# Targets (phase-1 Ă‚Â§P1.S1): `strata_artifact` (no CUDA dependency), `tests`.  `strata_plan` and the C++
# `strata-pack` are added when their sources exist - a target with no sources would make the build graph
# claim something the tree does not have, which is worse than an absent target.
#
# CUDA targets arrive in Phase 2 and this build REJECTS any architecture other than 120: the engine is
# written against sm_120 (RTX 5000 / Blackwell) and silently building for an older arch would produce a
# binary that runs and is wrong about its own performance, which is the worst of both outcomes.
cmake_minimum_required(VERSION 3.24)
<<<<<<< this fork
project(strata VERSION 0.1.44.1 LANGUAGES CXX)
add_compile_definitions(STRATA_VERSION="${PROJECT_VERSION}")

# llama.cpp is vendored as a submodule (third_party/llama.cpp; see FORK_NOTES.md): ggml, the MMQ kernel
# sources and ggml-common.h all come from that one pinned commit, and the build fetches nothing.
set(STRATA_GGML_DIR "${CMAKE_CURRENT_SOURCE_DIR}/third_party/llama.cpp" CACHE PATH
    "llama.cpp checkout to take ggml, the MMQ sources and ggml-common.h from")
if(NOT EXISTS "${STRATA_GGML_DIR}/ggml/CMakeLists.txt")
  message(FATAL_ERROR "llama.cpp was not found at ${STRATA_GGML_DIR}: run `git submodule update --init` first")
endif()
# issue #31: release builds keep a PDB beside the exe (not shipped in it), so a minidump from a stalled engine can
# be read back to the exact source line
if(MSVC AND CMAKE_BUILD_TYPE STREQUAL "Release")
  add_compile_options($<$<COMPILE_LANGUAGE:CXX>:/Zi>)
  add_link_options(/DEBUG /OPT:REF /OPT:ICF)
endif()
=======
project(strata VERSION 0.1.43.1 LANGUAGES CXX)
add_compile_definitions(STRATA_VERSION="${PROJECT_VERSION}")
>>>>>>> their fork (github.com-eddoursul-Strata-custom)

# The prebuilt release (setup.py downloads it, so users need no compiler): ggml's CPU backend for any AVX2 CPU
# instead of this machine's (Strata's own AVX-512 kernels are per-file and chosen at run time either way), and
# the MSVC runtime linked in, so no Visual C++ redistributable is needed.
option(STRATA_PORTABLE "build for other PCs: AVX2 baseline, static MSVC runtime" OFF)
if(STRATA_PORTABLE AND MSVC)
  set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded")
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

option(STRATA_ENABLE_CUDA "Build the CUDA targets (Phase 2+)" OFF)
<<<<<<< this fork
# AMD gfx906 (MI50, wave64): the same .cu sources compiled as HIP through the gfx906 compat layer.  The CUDA runtime and
# cuBLAS names are mapped by the headers in include/strata/platform/hip_compat/, which go first on the include path,
# and CUDA::cudart / CUDA::cublas become aliases of HIP's targets, so every CUDA target below stays as it is.  This is
# separate from STRATA_ENABLE_HIP (the wave32 RDNA backend): with it on, STRATA_ENABLE_CUDA is on and
# STRATA_ENABLE_HIP stays off, so every "CUDA" branch of the tree is the one taken.
option(STRATA_HIP_GFX906 "Build the CUDA targets for AMD gfx906 through HIP (wave64 compat layer)" OFF)
if(STRATA_HIP_GFX906)
  if(STRATA_ENABLE_HIP)
    message(FATAL_ERROR "STRATA_HIP_GFX906 and STRATA_ENABLE_HIP are separate backends; enable only one")
  endif()
  set(STRATA_ENABLE_CUDA ON CACHE BOOL "" FORCE)
endif()
option(STRATA_ENABLE_HIP "Build the HIP targets for wave32 AMD GPUs (gfx1100, gfx1201; gfx1012/gfx1030 unvalidated)" OFF)
option(STRATA_PREFILL_MMQ "Build the optional GGML MMQ prefill path for HIP" OFF)
# MMQ prompt kernels for Q4_K / Q5_K / Q5_1 / Q6_K (Unsloth's UD-Q4/Q5/Q6_K_XL experts; on HIP also the dense GGUF
# projections, STRATA_DENSE_MMQ).  On by default since MODEL-UPDATE Phase 1: on CUDA they are their own library
# (strata_kq, loaded only for a model with these formats: the engine loads every kernel of its own binary at start,
# CUDA_MODULE_LOADING=EAGER, and these took ~3.8 MB of VRAM - 3 IQ3_XXS expert slots on a P100 - from every model); on
# HIP a code object loads on first use, so they cost other models nothing there (R9700: the same slots).  Used at run
# time with STRATA_MMQ_KQ=1 (opt-in: it changes the prompt's rounding against the FP16 path).
option(STRATA_MMQ_KQUANTS "Build the MMQ prompt kernels for Q4_K / Q5_K / Q5_1 / Q6_K: the experts of Unsloth's K-quant files, and on HIP the dense GGUF projections (STRATA_DENSE_MMQ)" ON)
# Q6_K gate/up in the grouped expert kernels (Unsloth's UD-Q6_K_XL, and layer 2 of UD-Q5_K_XL; docs/QUANTS.md).  On by
# default since MODEL-UPDATE Phase 1: one more instance per grouped kernel in iq_kernels.cu, +19 KB of sm_60 code,
# measured to leave every expert-cache slot count as it was (P100, R9700).  -DSTRATA_Q6K_EXPERTS=OFF drops it.
option(STRATA_Q6K_EXPERTS "Build the Q6_K gate/up grouped expert kernels (UD-Q6_K_XL, UD-Q5_K_XL)" ON)
if(STRATA_Q6K_EXPERTS)
  add_compile_definitions(STRATA_Q6K_EXPERTS=1)
endif()
option(STRATA_ORCA_Q4KS_MMQ "Build the GGML Q5_0 MMQ instance for OrcaRouter Q4_K_S" OFF)
if(STRATA_PREFILL_MMQ AND NOT STRATA_ENABLE_HIP)
  message(FATAL_ERROR "STRATA_PREFILL_MMQ requires STRATA_ENABLE_HIP=ON")
endif()
if(STRATA_ENABLE_CUDA AND STRATA_ENABLE_HIP AND NOT STRATA_HIP_GFX906)
  message(FATAL_ERROR "STRATA_ENABLE_CUDA and STRATA_ENABLE_HIP are separate backends; enable only one")
endif()
if(STRATA_ENABLE_HIP)
  include(cmake/hip_backend.cmake)
endif()
=======
>>>>>>> their fork (github.com-eddoursul-Strata-custom)
# The test suite and the micro benchmarks live in tests/ and bench/micro/, which the published source leaves out.
set(_strata_tests_default OFF)
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/tests/CMakeLists.txt AND EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/bench/micro)
  set(_strata_tests_default ON)
endif()
option(STRATA_BUILD_TESTS "Build the test suite" ${_strata_tests_default})
option(STRATA_WERROR "Treat compiler warnings as errors" OFF)

if(STRATA_ENABLE_CUDA)
  enable_language(CUDA)
  # `enable_language(CUDA)` gives the .cu compile rules but NOT the imported targets, so `CUDA::cudart` has to
  # be brought in explicitly for the host-side tests that call cudaMalloc themselves.
  find_package(CUDAToolkit REQUIRED)
  if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 120 CACHE STRING "CUDA architectures" FORCE)
  endif()
  foreach(_a IN LISTS CMAKE_CUDA_ARCHITECTURES)
    string(REGEX REPLACE "-.*$" "" _base "${_a}")     # 120-real / 120-virtual are still 120
    # Plan v0.3 P6: developed and measured on sm_120 (RTX 50); the kernels need sm_80 or newer (tf32 mma in
    # the attention scorer, bf16 math), so RTX 30 (86) and RTX 40 (89) build too.  Older cards are refused.
    if(_base STREQUAL "native" OR _base STREQUAL "all" OR _base STREQUAL "all-major")
      continue()
    endif()
    if(_base LESS 80)
      message(FATAL_ERROR
        "Strata needs an NVIDIA GPU of compute capability 8.0 or newer (RTX 30 / 40 / 50 series), but "
        "CMAKE_CUDA_ARCHITECTURES is '${CMAKE_CUDA_ARCHITECTURES}'.")
    endif()
    # #220 / #224: RTX 50 engines built with CUDA 12.8 crashed in the prompt path on Linux (13.0 and newer did not)
    if(_base GREATER_EQUAL 120 AND CMAKE_CUDA_COMPILER_VERSION VERSION_LESS 13.0)
      message(WARNING "Building for sm_${_base} (RTX 50) with CUDA ${CMAKE_CUDA_COMPILER_VERSION}: engines built with "
                      "CUDA 12.8 crashed on long prompts on these cards (issues #220, #224). Use CUDA 13.0 or newer.")
    endif()
  endforeach()
  message(STATUS "Strata: CUDA enabled, arch ${CMAKE_CUDA_ARCHITECTURES}")
endif()

# Warnings live on an INTERFACE target the project's own code links, NOT in add_compile_options: applied
# globally they would also hit FetchContent'd Catch2, whose -W4 warnings are not this project's to fix and
# whose /WX failures would be indistinguishable from our own.
#
# Every flag is guarded by COMPILE_LANGUAGE:CXX.  Without that guard nvcc receives bare `/W4 /utf-8` and dies
# with "A single input file is required for a non-link phase" - it parses them as inputs rather than host
# flags.  Host flags for CUDA must go through -Xcompiler, which is what COMPILE_LANGUAGE:CUDA does below.
add_library(strata_warnings INTERFACE)
target_compile_options(strata_warnings INTERFACE
  $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CXX_COMPILER_ID:MSVC>>:/W4>
  $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CXX_COMPILER_ID:MSVC>>:/utf-8>
  # C4324: a struct padded after an alignas member (the CPU pool's activation images), as intended.
  $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CXX_COMPILER_ID:MSVC>>:/wd4324>
  $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<NOT:$<CXX_COMPILER_ID:MSVC>>>:-Wall>
  $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<NOT:$<CXX_COMPILER_ID:MSVC>>>:-Wextra>
  $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CXX_COMPILER_ID:MSVC>>:-Xcompiler=/W4>
  # C4211 comes from the host stubs nvcc generates for kernels with internal linkage (anonymous namespaces).
  $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CXX_COMPILER_ID:MSVC>>:-Xcompiler=/wd4211>)
# fopen, sscanf and getenv are standard C; MSVC deprecates them in favour of its own _s variants.
target_compile_definitions(strata_warnings INTERFACE $<$<CXX_COMPILER_ID:MSVC>:_CRT_SECURE_NO_WARNINGS>)
if(STRATA_WERROR)
  target_compile_options(strata_warnings INTERFACE
    $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CXX_COMPILER_ID:MSVC>>:/WX>
    $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<NOT:$<CXX_COMPILER_ID:MSVC>>>:-Werror>)
endif()

# ---- strata_artifact -------------------------------------------------------
# Header-only, and that is a deliberate property rather than an unfinished split: the GGUF reader and the
# scalar dequantizers hold no state worth hiding, and a header-only boundary cannot introduce a
# duplicate-symbol or missing-declaration bug into code that is already validated against the real artifact.
add_library(strata_artifact INTERFACE)
target_include_directories(strata_artifact INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(strata_artifact INTERFACE strata_warnings)

add_executable(strata-gguf src/artifact/gguf_reader.cpp)
target_link_libraries(strata-gguf PRIVATE strata_artifact)

add_executable(strata-dequant src/artifact/dequant.cpp)
target_link_libraries(strata-dequant PRIVATE strata_artifact)
# this file is now nothing but the CLI entry point, and its inherited guard spells that as a "selftest"
target_compile_definitions(strata-dequant PRIVATE STRATA_DEQUANT_SELFTEST)
<<<<<<< this fork
if(STRATA_BUILD_TESTS)
  add_executable(logit_bias_test src/core/logit_bias_test.cpp)
  target_include_directories(logit_bias_test PRIVATE include)
  add_test(NAME logit_bias_test COMMAND logit_bias_test)
  # the reader refuses a duplicate tensor name at open, naming it (a synthetic GGUF: no model, no GPU)
  add_executable(gguf_reader_test src/artifact/gguf_reader_test.cpp)
  target_link_libraries(gguf_reader_test PRIVATE strata_artifact)
  add_test(NAME gguf_reader_test COMMAND gguf_reader_test)
  # a split model's shards and their joint view: missing shard, split keys, duplicates, truncation (synthetic)
  add_executable(gguf_split_test tests/core/gguf_split_test.cpp)
  target_link_libraries(gguf_split_test PRIVATE strata_artifact)
  add_test(NAME gguf_split_test COMMAND gguf_split_test)
  # the model families (qwen4exp, qwen35moe): the guards, the geometry from metadata, the layout tables (synthetic)
  add_executable(model_family_test tests/core/model_family_test.cpp)
  target_link_libraries(model_family_test PRIVATE strata_artifact)
  add_test(NAME model_family_test COMMAND model_family_test)
  # how --batch cuts a prompt read into pieces (a layer split with no slot decoding must still reach a BYIELD)
  add_executable(batch_read_test tests/program/batch_read_test.cpp)
  target_include_directories(batch_read_test PRIVATE include)
  add_test(NAME batch_read_test COMMAND batch_read_test)
  # the model's geometry from its metadata, and the check against the kernels' compiled shape (synthetic)
  add_executable(gguf_geometry_test tests/core/gguf_geometry_test.cpp)
  target_link_libraries(gguf_geometry_test PRIVATE strata_artifact)
  add_test(NAME gguf_geometry_test COMMAND gguf_geometry_test)

  # the hipBLASLt tuning header's architecture gate: RDNA4 siblings share solution IDs, RDNA3 siblings share
  # theirs, cross-family pairs and version mismatches stay rejected (synthetic tables: no model, no GPU)
  add_executable(tuning_header_arch_family_test tests/prefill/tuning_header_arch_family_test.cpp)
  target_include_directories(tuning_header_arch_family_test PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src/prefill")
  add_test(NAME tuning_header_arch_family_test COMMAND tuning_header_arch_family_test)

  # the auto layer split's minimum stage size (#1616): header-only rule, no GPU
  add_executable(split_rules_test tests/program/split_rules_test.cpp)
  target_include_directories(split_rules_test PRIVATE include)
  add_test(NAME split_rules_test COMMAND split_rules_test)

  # the default CPU prefill share is not armed when every expert is resident (#1595): header-only rule, no GPU
  add_executable(share_rules_test tests/prefill/share_rules_test.cpp)
  target_include_directories(share_rules_test PRIVATE include)
  add_test(NAME share_rules_test COMMAND share_rules_test)

  # the Windows HIP auto prefill chunk cap (#1630) and the commit-limit verdict (#1607): header-only, no GPU
  add_executable(host_limits_test tests/platform/host_limits_test.cpp)
  target_include_directories(host_limits_test PRIVATE include)
  add_test(NAME host_limits_test COMMAND host_limits_test)
endif()
=======
>>>>>>> their fork (github.com-eddoursul-Strata-custom)

# ---- strata_plan ----------------------------------------------------------
# Also header-only.  A "planner" that hides state would be one whose decision is not inspectable, and the
# whole point of P1.S9 is that the decision is arithmetic you can check by hand.
add_library(strata_plan INTERFACE)
target_include_directories(strata_plan INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(strata_plan INTERFACE strata_warnings)

add_executable(strata-plan src/plan/plan_main.cpp)
target_link_libraries(strata-plan PRIVATE strata_plan)

# ---- strata_core (CUDA, Phase 2) ------------------------------------------
# Only built when CUDA is on, so the Phase 1 targets keep working on a machine with no toolkit.  The arch
# guard above has already refused anything but sm_120 by the time this runs.
if(STRATA_ENABLE_CUDA)
  add_library(strata_core STATIC src/core/device.cu src/core/pinned.cu src/platform/memory.cpp src/core/graph.cpp src/core/weights.cpp
                               src/core/layout.cpp)
  target_include_directories(strata_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
  # CUDA::cudart is PUBLIC because graph.cpp is a HOST translation unit that calls cudaGraphLaunch and
  # cudaEventQuery: the .cu sources get cudart through the CUDA language rules, a .cpp does not, and the
  # linker error ("unresolved external symbol cudaGraphInstantiate") names the function rather than the cause.
  target_link_libraries(strata_core PUBLIC strata_plan strata_warnings CUDA::cudart)

  add_executable(strata-device src/core/device_main.cpp)
  target_link_libraries(strata-device PRIVATE strata_core)

  add_executable(strata-load src/core/load_main.cpp)
  target_link_libraries(strata-load PRIVATE strata_core strata_kernels)

  add_executable(strata-concurrent src/core/concurrent_main.cpp)
  target_include_directories(strata-concurrent PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
  target_link_libraries(strata-concurrent PRIVATE strata_core strata_kernels CUDA::cudart)

  add_executable(strata-overlap src/core/overlap_main.cpp)
  # a .cpp that calls cudaMemcpyAsync itself needs the toolkit headers and cudart explicitly
  target_include_directories(strata-overlap PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
  target_link_libraries(strata-overlap PRIVATE strata_core strata_kernels CUDA::cudart)

  # The load test needs the 34 GB artifact, so it is registered ONLY where that artifact exists.  A test that
  # silently skips is a test that stops running; an absent registration is visible in the configure output.
  if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/pack/full/experts.bin)
    add_test(NAME expert_arena_load
             COMMAND strata-load --file ${CMAKE_CURRENT_SOURCE_DIR}/pack/full/experts.bin --layers 1)
    set_tests_properties(expert_arena_load PROPERTIES PASS_REGULAR_EXPRESSION "per-layer checksums")
  else()
    message(STATUS "strata: pack/full/experts.bin absent - the expert arena load test is NOT registered")
  endif()

  # ---- strata_kernels (P2.S2) ----
  # Naive first, each with a parity test against a reference that was itself validated against ggml.  The
  # kernels link `strata_artifact` for the scalar dequantizers, because using the VALIDATED scalar path as the
  # reference is the whole point - a kernel checked against another copy of itself proves nothing.
  add_library(strata_kernels STATIC src/kernels/cuda/dequant_s2.cu src/kernels/cuda/s2_gemv.cu
                          src/kernels/cuda/s_gemv.cu
                          src/kernels/cuda/s2_gemv_quads.cu
                          src/kernels/cuda/s2_gemv_fast.cu
                          src/kernels/cuda/router_top10.cu
                          src/kernels/cuda/quantize_act.cu
                          src/kernels/cuda/rope.cu
                          src/kernels/cuda/sampler.cu
                          src/kernels/cuda/s2_gemv_q8.cu
    src/kernels/cuda/s2_expert_grouped.cu
                          src/kernels/cuda/shared_expert.cu
                          src/kernels/cuda/gr.cu
                          src/kernels/cuda/gdn.cu
                          src/kernels/cuda/qsa.cu
                          src/kernels/cuda/elementwise.cu
                          src/kernels/cuda/bf16_gemv.cu
                          src/kernels/cuda/native_mmvq.cu
                          src/kernels/cuda/native_bf16.cu
                          src/kernels/cuda/native_gr_norm.cu
                          src/kernels/cuda/native_moe.cu
                          src/kernels/cuda/native_gdn.cu
                          src/kernels/cuda/native_gr_postops.cu
                          src/kernels/cuda/native_router.cu
                          src/kernels/cuda/native_gdn_preprocess.cu
                          src/kernels/cuda/native_qsa.cu
                          src/kernels/cuda/native_rope.cu
                          src/kernels/cuda/native_ple_postops.cu
                          src/kernels/cuda/native_flash_attn.cu
                          src/kernels/cuda/native_qsa_indexer.cu
                          src/kernels/cuda/native_qsa_score.cu
                          src/kernels/cuda/ple.cu
                          src/kernels/cuda/kv_q8.cu
                          src/kernels/cuda/kv_q4.cu
                          src/kernels/cuda/fused_gr.cu
                          src/kernels/cuda/qsa_decode_attn.cu
                          src/kernels/cuda/fused_gdn.cu
                          src/kernels/cuda/dequant_bf16.cu
                          src/kernels/cuda/qsa_select.cu
                          src/kernels/cuda/verify_kernels.cu
                          src/kernels/cuda/iq_kernels.cu
                          src/kernels/ngram.cpp
                          src/ngram/ple_reader.cpp
                          src/platform/direct_file.cpp)
  target_link_libraries(strata_kernels PUBLIC strata_core strata_artifact)
<<<<<<< this fork
  # perf-review D-1: the tensor-core prompt attention vs the FP32 kernel and FP64 (GPU, synthetic, no model); with
  # the tests, or alone with -DSTRATA_PARITY_PROMPT_ATTN=ON.  CUDA only: HIP compiles the tensor-core kernel out.
  if((STRATA_BUILD_TESTS OR STRATA_PARITY_PROMPT_ATTN) AND STRATA_ENABLE_CUDA)
    add_executable(qsa_prompt_attn_parity src/kernels/qsa_prompt_attn_parity.cpp)
    target_include_directories(qsa_prompt_attn_parity PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
    target_link_libraries(qsa_prompt_attn_parity PRIVATE strata_kernels CUDA::cudart)
  endif()
  # #512: the top-k's active-block bound (Turing's dispatch, forced on any card by STRATA_TOPK_ACTIVE_ANY) against
  # the capacity rule, selected ids bitwise
  if(STRATA_BUILD_TESTS AND STRATA_ENABLE_CUDA)
    add_executable(qsa_topk_active_parity src/kernels/qsa_topk_active_parity.cpp)
    target_include_directories(qsa_topk_active_parity PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
    target_link_libraries(qsa_topk_active_parity PRIVATE strata_kernels CUDA::cudart)
    add_test(NAME qsa_topk_active_parity COMMAND qsa_topk_active_parity)
    set_tests_properties(qsa_topk_active_parity PROPERTIES SKIP_RETURN_CODE 77)
  endif()
  # the QSA selection (block scores on the warp kernel and the tensor-core one, top-k): a GPU bench with a
  # scorer-vs-scorer comparison and an FP64 accuracy gate (synthetic, no model); the ctest runs a 32K context under a
  # 262144-cell capacity (the top-k dispatch follows the capacity) and exits non-zero on a failed gate
  if((STRATA_BUILD_TESTS OR STRATA_PARITY_PROMPT_ATTN) AND STRATA_ENABLE_HIP)
    add_executable(qsa_select_bench src/kernels/qsa_select_bench.cpp)
    set_source_files_properties(src/kernels/qsa_select_bench.cpp PROPERTIES LANGUAGE HIP)
    target_link_libraries(qsa_select_bench PRIVATE strata_kernels strata_hip_runtime)
    add_test(NAME qsa_select_bench COMMAND qsa_select_bench 32768 256 3 262144)
  endif()
  # the same on CUDA: the tensor-core scorer (sm_80+) or the FP32 tiled one (below sm_80, and forced on any card by
  # STRATA_QSA_WARP=1 in the second test) against the warp kernel and FP64, at 32K and 131K
  if(STRATA_BUILD_TESTS AND STRATA_ENABLE_CUDA)
    add_executable(qsa_select_bench src/kernels/qsa_select_bench.cpp)
    target_include_directories(qsa_select_bench PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
    target_link_libraries(qsa_select_bench PRIVATE strata_kernels CUDA::cudart)
    add_test(NAME qsa_select_bench COMMAND qsa_select_bench 32768 256 3 262144)
    add_test(NAME qsa_select_bench_fp32_tiled COMMAND qsa_select_bench 131072 256 3 262144)
    set_tests_properties(qsa_select_bench_fp32_tiled PROPERTIES ENVIRONMENT "STRATA_QSA_WARP=1;STRATA_SELECT_SIMT=1")
  endif()
  # plan v0.3 P6: llama.cpp's block layouts and codebook grids for the i-quant formats (MIT; ggml-common.h
  # comes from the vendored llama.cpp submodule's ggml/src, the same commit the library is built from)
  target_include_directories(strata_kernels PRIVATE ${STRATA_GGML_DIR}/ggml/src)
=======
  # plan v0.3 P6: llama.cpp's block layouts and codebook grids for the i-quant formats (MIT, third_party/ggml)
  target_include_directories(strata_kernels PRIVATE third_party/ggml)
>>>>>>> their fork (github.com-eddoursul-Strata-custom)
  # Match the pinned llama.cpp CUDA arithmetic contract for this adapter only.
  set_source_files_properties(src/kernels/cuda/native_mmvq.cu src/kernels/cuda/native_bf16.cu
    src/kernels/cuda/native_gr_norm.cu
    src/kernels/cuda/native_moe.cu
    src/kernels/cuda/native_gdn.cu
    src/kernels/cuda/native_gr_postops.cu
    src/kernels/cuda/native_router.cu
    src/kernels/cuda/native_gdn_preprocess.cu
    src/kernels/cuda/native_qsa.cu
    src/kernels/cuda/native_rope.cu
    src/kernels/cuda/native_ple_postops.cu
    src/kernels/cuda/native_flash_attn.cu
    src/kernels/cuda/native_qsa_indexer.cu
    src/kernels/cuda/native_qsa_score.cu
    PROPERTIES COMPILE_OPTIONS "--use_fast_math")

  set(STRATA_ORACLE_SOURCE_DIR "" CACHE PATH "Pinned llama.cpp source for independent CUDA parity tests")
  set(STRATA_ORACLE_BUILD_DIR "" CACHE PATH "Matching existing shared llama.cpp build")
  if(STRATA_ORACLE_SOURCE_DIR AND STRATA_ORACLE_BUILD_DIR)
    foreach(parity IN ITEMS native_mmvq native_bf16 native_shared native_gr_norm native_moe native_gdn native_gr_postops native_router native_gdn_preprocess native_gdn_pipeline native_qsa native_rope native_ple_postops native_flash_attn native_qsa_indexer native_qsa_score)
    add_executable(${parity}_parity bench/micro/${parity}_parity.cpp)
    target_include_directories(${parity}_parity PRIVATE
      "${STRATA_ORACLE_SOURCE_DIR}/ggml/include" ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
    target_compile_definitions(${parity}_parity PRIVATE GGML_SHARED GGML_BACKEND_SHARED)
    target_link_libraries(${parity}_parity PRIVATE strata_kernels CUDA::cudart)
    foreach(component IN ITEMS ggml ggml-base ggml-cpu ggml-cuda)
      string(REPLACE "-" "_" variable "STRATA_ORACLE_${component}")
      find_library(${variable} NAMES ${component} PATHS
        "${STRATA_ORACLE_BUILD_DIR}/ggml/src" "${STRATA_ORACLE_BUILD_DIR}/ggml/src/${component}"
        "${STRATA_ORACLE_BUILD_DIR}/bin" NO_DEFAULT_PATH REQUIRED)
      target_link_libraries(${parity}_parity PRIVATE "${${variable}}")
    endforeach()
    add_test(NAME ${parity}_parity COMMAND ${parity}_parity --selftest)
    if(WIN32)
      set_tests_properties(${parity}_parity PROPERTIES
        ENVIRONMENT_MODIFICATION "PATH=path_list_prepend:${STRATA_ORACLE_BUILD_DIR}/bin")
    endif()
    endforeach()
  endif()

  # ---- strata_engine (P2.S5) -----------------------------------------------------
  # The COMPOSED layer lives here and not in `strata_core`, because it calls kernels and `strata_kernels`
  # already links `strata_core` - putting `gdn_layer` in core would make the two mutually dependent.  A third
  # library is the honest shape of the dependency graph: core knows about devices and weights, kernels know
  # about arithmetic, and this one knows the ORDER the arithmetic happens in.
  add_library(strata_engine STATIC src/core/layer.cpp src/core/session.cpp src/core/expert_source.cpp
    src/core/expert_cache.cpp src/core/native_head.cpp src/core/native_dense.cpp src/core/verify.cpp src/core/mtp.cpp
    src/core/prompt_cache.cpp src/core/adaptive_tier.cpp src/core/second_gpu.cpp
    src/core/split_head.cpp src/core/vmm.cpp)
  target_include_directories(strata_engine PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
<<<<<<< this fork
  target_link_libraries(strata_engine PUBLIC strata_core strata_kernels strata_kernels_cpu ${_strata_gpu_runtime_target})
  if(STRATA_BUILD_TESTS AND STRATA_ENABLE_CUDA AND NOT STRATA_HIP_GFX906)
    add_executable(host_graph_launch_test tests/core/host_graph_launch_test.cu)
    target_include_directories(host_graph_launch_test PRIVATE include)
    target_link_libraries(host_graph_launch_test PRIVATE CUDA::cudart Threads::Threads)
    add_test(NAME host_graph_launch_test COMMAND host_graph_launch_test)
    set_tests_properties(host_graph_launch_test PROPERTIES TIMEOUT 30)
  endif()
  if(STRATA_BUILD_TESTS AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/core/exchange_storage_test.cpp")
    add_executable(exchange_storage_test tests/core/exchange_storage_test.cpp)
    target_include_directories(exchange_storage_test PRIVATE include)
    target_link_libraries(exchange_storage_test PRIVATE Threads::Threads)
    add_test(NAME exchange_storage_test COMMAND exchange_storage_test)
    set_tests_properties(exchange_storage_test PROPERTIES TIMEOUT 60)
  endif()

  # available_host_bytes(): MemAvailable, cgroup limits and the operator's --memory-limit-mib cap on fixture directories
  # - no GPU, no CUDA
  if(STRATA_BUILD_TESTS AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/core/memory_guard_test.cpp")
    add_executable(memory_guard_test tests/core/memory_guard_test.cpp src/core/conversation_memory.cpp)
    target_include_directories(memory_guard_test PRIVATE include)
    add_test(NAME memory_guard_test COMMAND memory_guard_test)
    set_tests_properties(memory_guard_test PROPERTIES TIMEOUT 60)
  endif()

  if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/core/file_expert_source_test.cpp")
    enable_testing()
    add_executable(file_expert_source_test tests/core/file_expert_source_test.cpp)
    target_link_libraries(file_expert_source_test PRIVATE strata_engine)
    add_test(NAME file_expert_source_test COMMAND file_expert_source_test)
    set_tests_properties(file_expert_source_test PROPERTIES TIMEOUT 60)
    add_executable(vmm_test tests/core/vmm_test.cpp)
    target_link_libraries(vmm_test PRIVATE strata_engine)
    add_test(NAME vmm_test COMMAND vmm_test)
    set_tests_properties(vmm_test PROPERTIES TIMEOUT 60)
    # The resident RAM mode with a layer split: the copy keeps each stage's lend region, and the experts the
    # prompt path borrows are served from RAM (the file source reads none of them).  The device calls are
    # wrapped - conversation_transfer_test's pattern - and `pin = false` keeps the copy pageable, so this
    # runs on a CPU-only PC.
    if(NOT STRATA_ENABLE_HIP AND UNIX AND NOT APPLE AND CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
      add_executable(split_lend_complement_test tests/core/split_lend_complement_test.cpp)
      target_link_libraries(split_lend_complement_test PRIVATE strata_engine)
      target_link_options(split_lend_complement_test PRIVATE
        -Wl,--wrap=cudaMalloc -Wl,--wrap=cudaFree -Wl,--wrap=cudaMemset -Wl,--wrap=cudaMemcpy
        -Wl,--wrap=cudaMemcpyAsync -Wl,--wrap=cudaStreamSynchronize -Wl,--wrap=cudaDeviceSynchronize
        -Wl,--wrap=cudaGetLastError -Wl,--wrap=cudaGetDevice -Wl,--wrap=cudaSetDevice
        -Wl,--wrap=cudaMemGetInfo -Wl,--wrap=cudaHostAlloc -Wl,--wrap=cudaFreeHost)
      add_test(NAME split_lend_complement_test COMMAND split_lend_complement_test)
      set_tests_properties(split_lend_complement_test PROPERTIES ENVIRONMENT "CUDA_VISIBLE_DEVICES=-1" TIMEOUT 120)
    endif()
  endif()
  # #477: the learned expert profile (--expert-profile-save): its order and make_profile.py's file format (CPU only)
  if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/core/expert_profile_save_test.cpp")
    enable_testing()
    add_executable(expert_profile_save_test tests/core/expert_profile_save_test.cpp)
    target_link_libraries(expert_profile_save_test PRIVATE strata_engine)
    add_test(NAME expert_profile_save_test COMMAND expert_profile_save_test)
    set_tests_properties(expert_profile_save_test PROPERTIES TIMEOUT 60)
  endif()
  # the profile header's version is enforced: a version other than 1 is refused by its number (CPU only)
  if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/core/expert_profile_version_test.cpp")
    enable_testing()
    add_executable(expert_profile_version_test tests/core/expert_profile_version_test.cpp)
    target_link_libraries(expert_profile_version_test PRIVATE strata_engine)
    add_test(NAME expert_profile_version_test COMMAND expert_profile_version_test)
    set_tests_properties(expert_profile_version_test PROPERTIES TIMEOUT 60)
  endif()
  if(STRATA_BUILD_TESTS AND STRATA_NATIVE_EXPERTS)
    # native_experts.txt v2-v4 (the real packs' copies and the per-role shard column) and the arena filled from
    # synthetic GGUF shards at the real expert dimensions
    add_executable(expert_layout_test tests/core/expert_layout_test.cpp)
    target_link_libraries(expert_layout_test PRIVATE strata_engine)
    add_test(NAME expert_layout_test COMMAND expert_layout_test ${CMAKE_CURRENT_SOURCE_DIR}/tests/data/native_experts)
    set_tests_properties(expert_layout_test PROPERTIES TIMEOUT 60)
    # #326: a native pack's BF16 (--compat-bf16) PLE key stays in the arena; a quantized key row loads as before
    add_executable(native_dense_ple_key_test tests/core/native_dense_ple_key_test.cpp)
    target_link_libraries(native_dense_ple_key_test PRIVATE strata_engine)
    add_test(NAME native_dense_ple_key_test COMMAND native_dense_ple_key_test)
    set_tests_properties(native_dense_ple_key_test PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 60)
    # #369: per-layer admission on a sized (native pack) cache starts each layer in its own range
    add_executable(expert_cache_per_layer_test tests/core/expert_cache_per_layer_test.cpp)
    target_link_libraries(expert_cache_per_layer_test PRIVATE strata_engine)
    add_test(NAME expert_cache_per_layer_test COMMAND expert_cache_per_layer_test)
    set_tests_properties(expert_cache_per_layer_test PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 60)
    add_executable(expert_cache_segmented_test tests/core/expert_cache_segmented_test.cpp)
    target_link_libraries(expert_cache_segmented_test PRIVATE strata_engine)
    add_test(NAME expert_cache_segmented_test COMMAND expert_cache_segmented_test)
    set_tests_properties(expert_cache_segmented_test PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 60)
  endif()
  if(STRATA_BUILD_CONVERSATION_TESTS)
    add_executable(conversation_snapshot_test src/core/conversation_snapshot_test.cpp)
    target_link_libraries(conversation_snapshot_test PRIVATE strata_engine)
    add_test(NAME conversation_snapshot_test COMMAND conversation_snapshot_test)
    set_tests_properties(conversation_snapshot_test PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 120)
    add_executable(conversation_validation_test src/core/conversation_validation_test.cpp)
    target_link_libraries(conversation_validation_test PRIVATE strata_engine)
    add_test(NAME conversation_validation_test COMMAND conversation_validation_test)
    set_tests_properties(conversation_validation_test PROPERTIES ENVIRONMENT "CUDA_VISIBLE_DEVICES=-1" TIMEOUT 120)
    if(NOT STRATA_ENABLE_HIP AND UNIX AND NOT APPLE AND CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
      add_executable(conversation_transfer_test src/core/conversation_validation_test.cpp)
      target_compile_definitions(conversation_transfer_test PRIVATE CONVERSATION_TEST_TRANSFERS)
      target_link_options(conversation_transfer_test PRIVATE
        -Wl,--wrap=cudaMemcpy -Wl,--wrap=cudaMemcpyAsync -Wl,--wrap=cudaStreamSynchronize
        -Wl,--wrap=cudaGetDevice -Wl,--wrap=cudaStreamCreateWithFlags
        -Wl,--wrap=cudaDeviceSynchronize -Wl,--wrap=cudaGetLastError)
      target_link_libraries(conversation_transfer_test PRIVATE strata_engine)
      add_test(NAME conversation_transfer_test COMMAND conversation_transfer_test)
      set_tests_properties(conversation_transfer_test PROPERTIES ENVIRONMENT "CUDA_VISIBLE_DEVICES=-1" TIMEOUT 120)
    endif()
  endif()
=======
  target_link_libraries(strata_engine PUBLIC strata_core strata_kernels strata_kernels_cpu CUDA::cudart)
>>>>>>> their fork (github.com-eddoursul-Strata-custom)

  # `strata generate` (P2.S6).  The FIRST user-facing program: everything under it is a component, and this is
  # what composes them into a token.  No tokenizer - the prompt arrives as ids - because Gate C1 compares logits
  # against llama.cpp on the SAME ids and a tokenizer on one side of that comparison is a second variable.
  # plan v0.3 P5: batched prompt processing (tensor-core GEMMs through cuBLAS).
  add_library(strata_prefill STATIC src/prefill/gemm.cu src/prefill/kernels.cu src/prefill/experts.cpp
                                    src/prefill/prefill.cpp)
  target_include_directories(strata_prefill PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
  target_link_libraries(strata_prefill PUBLIC strata_engine CUDA::cublas CUDA::cudart)

  add_executable(strata src/program/generate.cpp)
  target_include_directories(strata PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
  target_link_libraries(strata PRIVATE strata_engine strata_prefill strata_spec CUDA::cudart)

<<<<<<< this fork
  # NEW-MODELS-PLAN.md T1: Qwen3.6-35B-A3B's reference path (one GPU, the model resident, plain kernels), checked
  # against llama.cpp by tools/qwen35/; the engine itself does not run the family yet
  add_executable(strata-q36ref src/qwen35/q36ref.cu)
  target_include_directories(strata-q36ref PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include ${_strata_gpu_include_directories})
  target_link_libraries(strata-q36ref PRIVATE strata_kernels strata_warnings ${_strata_gpu_runtime_target})
  # the engine path for the same family (serve protocol, prompt chunks, decode graph): src/qwen35/q36_engine.cu, a
  # library that `strata` switches to for a qwen35moe model (the family switch at the top of its main) and that
  # strata-q36 wraps on its own
  add_library(strata_qwen35 STATIC src/qwen35/q36_engine.cu)
  target_include_directories(strata_qwen35 PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include PRIVATE ${_strata_gpu_include_directories})
  target_link_libraries(strata_qwen35 PRIVATE strata_prefill strata_kernels strata_warnings ${_strata_gpu_runtime_target})
  if(STRATA_NATIVE_EXPERTS)
    # the hybrid mode (experts that do not all fit in VRAM): the CPU expert pool on ggml-cpu's arithmetic
    target_link_libraries(strata_qwen35 PRIVATE strata_kernels_cpu)
    target_compile_definitions(strata_qwen35 PRIVATE STRATA_Q36_CPU_POOL=1)
  endif()
  add_executable(strata-q36 src/qwen35/q36_main.cpp)
  target_link_libraries(strata-q36 PRIVATE strata_qwen35 ${_strata_gpu_runtime_target})
  target_link_libraries(strata PRIVATE strata_qwen35)
  target_compile_definitions(strata PRIVATE STRATA_HAVE_QWEN35=1)

  if(STRATA_ENABLE_HIP)
    enable_testing()
    set(_strata_hip_tests)
    add_executable(hip_prefill_gemm tests/hip/prefill_gemm.cpp)
    target_link_libraries(hip_prefill_gemm PRIVATE strata_prefill strata_hip_runtime)
    add_test(NAME hip_prefill_gemm COMMAND hip_prefill_gemm)
    list(APPEND _strata_hip_tests hip_prefill_gemm)
    add_executable(hip_gemm_f16_io_parity tests/hip/gemm_f16_io_parity.cpp)
    target_link_libraries(hip_gemm_f16_io_parity PRIVATE strata_prefill strata_hip_runtime)
    add_test(NAME hip_gemm_f16_io_parity COMMAND hip_gemm_f16_io_parity)
    list(APPEND _strata_hip_tests hip_gemm_f16_io_parity)
    add_executable(hip_q2_zero tests/hip/q2_zero.cpp)
    target_link_libraries(hip_q2_zero PRIVATE strata_kernels strata_hip_runtime)
    add_test(NAME hip_q2_zero COMMAND hip_q2_zero)
    list(APPEND _strata_hip_tests hip_q2_zero)
    add_executable(hip_expert_cache_staging tests/hip/expert_cache_staging.cpp)
    target_link_libraries(hip_expert_cache_staging PRIVATE strata_engine)
    add_test(NAME hip_expert_cache_staging COMMAND hip_expert_cache_staging)
    list(APPEND _strata_hip_tests hip_expert_cache_staging)
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/hip/intrinsics.cpp")
      add_executable(hip_intrinsics tests/hip/intrinsics.cpp)
      target_link_libraries(hip_intrinsics PRIVATE strata_hip_runtime)
      add_test(NAME hip_intrinsics COMMAND hip_intrinsics)
      list(APPEND _strata_hip_tests hip_intrinsics)
    endif()
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/hip/handoff.cpp")
      add_executable(hip_handoff tests/hip/handoff.cpp)
      target_link_libraries(hip_handoff PRIVATE strata_kernels strata_hip_runtime)
      add_test(NAME hip_handoff COMMAND hip_handoff)
      list(APPEND _strata_hip_tests hip_handoff)
    endif()
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/hip/mapped_alias.cpp")
      # #325: reports what a hipHostGetDevicePointer alias is on this stack (on Windows: the host pointer, and a
      # D2D copy into it does not land - hip_handoff's timeout there). A diagnosis: exits 0 whenever it ran.
      add_executable(hip_mapped_alias tests/hip/mapped_alias.cpp)
      target_link_libraries(hip_mapped_alias PRIVATE strata_kernels strata_hip_runtime)
      add_test(NAME hip_mapped_alias COMMAND hip_mapped_alias)
      set_tests_properties(hip_mapped_alias PROPERTIES SKIP_RETURN_CODE 77)
      list(APPEND _strata_hip_tests hip_mapped_alias)
    endif()
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/hip/native_qsa_score.cpp")
      add_executable(hip_native_qsa_score tests/hip/native_qsa_score.cpp)
      target_link_libraries(hip_native_qsa_score PRIVATE strata_kernels strata_hip_runtime)
      add_test(NAME hip_native_qsa_score COMMAND hip_native_qsa_score)
      list(APPEND _strata_hip_tests hip_native_qsa_score)
    endif()
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/hip/prefill_native_batch.cpp")
      add_executable(hip_prefill_native_batch tests/hip/prefill_native_batch.cpp)
      target_link_libraries(hip_prefill_native_batch PRIVATE strata_kernels strata_hip_runtime)
      add_test(NAME hip_prefill_native_batch COMMAND hip_prefill_native_batch)
      list(APPEND _strata_hip_tests hip_prefill_native_batch)
    endif()
    # Aurora: the prompt GDN recurrence's per-head kernel against the column-split + norm kernels, bitwise
    add_executable(hip_gdn_rec_head tests/hip/gdn_rec_head.cpp)
    target_link_libraries(hip_gdn_rec_head PRIVATE strata_prefill strata_hip_runtime)
    add_test(NAME hip_gdn_rec_head COMMAND hip_gdn_rec_head)
    list(APPEND _strata_hip_tests hip_gdn_rec_head)
    # #313: the RDNA3 / RDNA3.5 WMMA dense GEMMs (opt-in STRATA_WMMA_GEMM=1) against a reference
    add_executable(hip_prefill_wmma_gemm_parity tests/hip/prefill_wmma_gemm_parity.cpp)
    target_include_directories(hip_prefill_wmma_gemm_parity PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src/prefill")
    target_link_libraries(hip_prefill_wmma_gemm_parity PRIVATE strata_prefill strata_hip_runtime)
    add_test(NAME hip_prefill_wmma_gemm_parity COMMAND hip_prefill_wmma_gemm_parity)
    set_tests_properties(hip_prefill_wmma_gemm_parity PROPERTIES SKIP_RETURN_CODE 77)   # no matrix cores: skipped
    list(APPEND _strata_hip_tests hip_prefill_wmma_gemm_parity)
    # S (STRATA_HCD_EXACT): the exact HC down kernel against hipBLASLt solution 1176 / 1177 (needs STRATA_HIPBLASLT_TUNING; exit 77 else)
    add_executable(hip_prefill_hcd_exact_parity tests/hip/prefill_hcd_exact_parity.cpp)
    target_link_libraries(hip_prefill_hcd_exact_parity PRIVATE strata_prefill strata_hip_runtime)
    add_test(NAME hip_prefill_hcd_exact_parity COMMAND hip_prefill_hcd_exact_parity)
    set_tests_properties(hip_prefill_hcd_exact_parity PROPERTIES SKIP_RETURN_CODE 77)
    list(APPEND _strata_hip_tests hip_prefill_hcd_exact_parity)
    # S6: the AMD fast router against the portable kernel, bitwise
    add_executable(hip_router_fast tests/hip/router_fast.cpp)
    target_link_libraries(hip_router_fast PRIVATE strata_kernels strata_hip_runtime)
    add_test(NAME hip_router_fast COMMAND hip_router_fast)
    list(APPEND _strata_hip_tests hip_router_fast)
    # S6: the RDNA4 matrix-core prompt attention (STRATA_HIP_WMMA) against the FP32 kernel and FP64; skips off gfx12
    add_executable(hip_prompt_attn_wmma src/kernels/qsa_prompt_attn_parity.cpp)
    target_link_libraries(hip_prompt_attn_wmma PRIVATE strata_kernels strata_hip_runtime)
    add_test(NAME hip_prompt_attn_wmma COMMAND hip_prompt_attn_wmma 32768 2048 3)
    set_tests_properties(hip_prompt_attn_wmma PROPERTIES TIMEOUT 600 SKIP_RETURN_CODE 77)
    list(APPEND _strata_hip_tests hip_prompt_attn_wmma)
    if(STRATA_HIPBLASLT_AVAILABLE AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/hip/prefill_hipblaslt_gemm.cpp")
      add_executable(hip_prefill_hipblaslt_gemm tests/hip/prefill_hipblaslt_gemm.cpp)
      target_include_directories(hip_prefill_hipblaslt_gemm PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src/prefill")
      target_link_libraries(hip_prefill_hipblaslt_gemm PRIVATE strata_prefill strata_hip_runtime roc::hipblaslt roc::hipblas)
      add_test(NAME hip_prefill_hipblaslt_gemm COMMAND hip_prefill_hipblaslt_gemm)
      set_tests_properties(hip_prefill_hipblaslt_gemm PROPERTIES TIMEOUT 120 SKIP_RETURN_CODE 77)
      list(APPEND _strata_hip_tests hip_prefill_hipblaslt_gemm)
    endif()
    if(_strata_hip_tests)
      set_tests_properties(${_strata_hip_tests} PROPERTIES TIMEOUT 60)
    endif()
    if(TARGET hip_prefill_native_batch)
      set_tests_properties(hip_prefill_native_batch PROPERTIES TIMEOUT 120)
    endif()
    if(TARGET hip_prefill_hipblaslt_gemm)
      set_tests_properties(hip_prefill_hipblaslt_gemm PROPERTIES TIMEOUT 120 SKIP_RETURN_CODE 77)
    endif()
=======
  add_executable(adaptive_policy_test tests/core/adaptive_policy_test.cpp)
  target_include_directories(adaptive_policy_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
  target_link_libraries(adaptive_policy_test PRIVATE strata_artifact)
  if(STRATA_BUILD_TESTS)
    add_test(NAME adaptive_policy_test COMMAND adaptive_policy_test)
    add_test(NAME deterministic_adapt_help COMMAND strata --deterministic-adapt --help)
    set_tests_properties(deterministic_adapt_help PROPERTIES
                         PASS_REGULAR_EXPRESSION "--deterministic-adapt")
>>>>>>> their fork (github.com-eddoursul-Strata-custom)
  endif()

  add_executable(dequant_s2_parity src/kernels/dequant_s2_parity.cpp)
  # The tests are HOST C++ files that call cudaMalloc/cudaMemcpy themselves, so they need the toolkit headers
  # and cudart explicitly: enabling the CUDA language gives those to .cu sources automatically, but a .cpp
  # target is just a C++ target and gets neither.
  target_include_directories(dequant_s2_parity PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
  target_link_libraries(dequant_s2_parity PRIVATE strata_kernels CUDA::cudart)
  add_test(NAME dequant_s2_parity COMMAND dequant_s2_parity --selftest)
  # the coalesced Q8_0 dequant against the per-block kernel: the same bytes in BF16, FP16 and FP32
  add_executable(dequant_q8_0_identity src/kernels/dequant_q8_0_identity.cu)
  target_link_libraries(dequant_q8_0_identity PRIVATE strata_kernels CUDA::cudart)
  add_test(NAME dequant_q8_0_identity COMMAND dequant_q8_0_identity)

  add_executable(s2_gemv_parity src/kernels/s2_gemv_parity.cpp)
  target_include_directories(s2_gemv_parity PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
  target_link_libraries(s2_gemv_parity PRIVATE strata_kernels CUDA::cudart)
  add_test(NAME s2_gemv_parity COMMAND s2_gemv_parity --selftest)

  add_executable(iq_parity src/kernels/iq_parity.cpp)
<<<<<<< this fork
  target_include_directories(iq_parity PRIVATE ${_strata_gpu_include_directories})
  target_link_libraries(iq_parity PRIVATE strata_kernels ${_strata_gpu_runtime_target})
  # TODO 24: the i-quant fixtures are generated, not shipped.  tools/iq_fixture.py needs python3 + numpy
  # and the repository's vendored gguf-py (no pip installs); it writes deterministic fixtures into the
  # BUILD directory (the source tree is not touched).  The CTest fixture makes generation-to-test order
  # explicit even under parallel execution; a fixture-generation failure fails iq_parity with it.
  find_package(Python3 COMPONENTS Interpreter QUIET)
  # the generator needs an interpreter WITH numpy: probe the configured one, the setup's .venv, and the
  # system python3; -DIQ_FIXTURE_PYTHON=... overrides the search outright.
  set(_iq_python "")
  set(_iq_candidates "")
  if(IQ_FIXTURE_PYTHON)
    list(APPEND _iq_candidates ${IQ_FIXTURE_PYTHON})
  else()
    if(Python3_Interpreter_FOUND)
      list(APPEND _iq_candidates ${Python3_EXECUTABLE})
    endif()
    list(APPEND _iq_candidates "${CMAKE_CURRENT_SOURCE_DIR}/.venv/bin/python"
                                "${CMAKE_CURRENT_SOURCE_DIR}/.venv/Scripts/python.exe"
                                /usr/bin/python3
                                python3)
  endif()
  foreach(_cand IN LISTS _iq_candidates)
    if(_cand STREQUAL "")
      continue()
    endif()
    execute_process(COMMAND ${_cand} -c "import numpy" OUTPUT_QUIET ERROR_QUIET RESULT_VARIABLE _iq_probe)
    if(_iq_probe EQUAL 0)
      set(_iq_python "${_cand}")
      break()
    endif()
  endforeach()
  if(NOT _iq_python STREQUAL "")
    add_test(NAME iq_parity_fixtures
             COMMAND ${_iq_python} ${CMAKE_CURRENT_SOURCE_DIR}/tools/iq_fixture.py
                     --out ${CMAKE_CURRENT_BINARY_DIR}/iq_fixture)
    # exit 3: numpy or the vendored gguf-py is missing (a trimmed third_party) - a skip, not a failure
    set_tests_properties(iq_parity_fixtures PROPERTIES FIXTURES_SETUP iq_fixture_data SKIP_RETURN_CODE 3)
    add_test(NAME iq_parity COMMAND iq_parity ${CMAKE_CURRENT_BINARY_DIR}/iq_fixture)
    set_tests_properties(iq_parity PROPERTIES FIXTURES_REQUIRED iq_fixture_data SKIP_RETURN_CODE 3)
  else()
    message(STATUS "STRATA tests: no python3 with numpy was found (the configured interpreter and .venv "
                   "were probed) - the iq_parity fixtures and its test are not registered; the fixtures "
                   "need python3 with numpy and the repository's vendored gguf-py")
  endif()
  if(STRATA_ENABLE_CUDA)
    # native_mmvq's multi_exact contract, bitwise: a T-column call against T single-column calls on the same Q8_1
    # bytes, with the flag off as the negative control (synthetic, GPU, no model; the control assumes 32-lane warps)
    add_executable(mmvq_multi_parity src/kernels/mmvq_multi_parity.cpp)
    target_include_directories(mmvq_multi_parity PRIVATE ${_strata_gpu_include_directories})
    target_link_libraries(mmvq_multi_parity PRIVATE strata_kernels ${_strata_gpu_runtime_target})
    add_test(NAME mmvq_multi_parity COMMAND mmvq_multi_parity)
    if(STRATA_HIP_GFX906)   # the CUDA layout, where the negative control has power (the wave64 one is exact at every T)
      add_test(NAME mmvq_multi_parity_cuda_layout COMMAND mmvq_multi_parity)
      set_tests_properties(mmvq_multi_parity_cuda_layout PROPERTIES ENVIRONMENT "STRATA_MMVQ_WAVE=0")
    endif()
  endif()
  if(STRATA_ENABLE_CUDA)
    # fork F4: the 2-4 column dense projections from the interleaved q8_1 copy, bitwise vs native_mmvq (GPU, synthetic)
    add_executable(mmvq_il_parity src/kernels/mmvq_il_parity.cpp)
    target_include_directories(mmvq_il_parity PRIVATE ${_strata_gpu_include_directories})
    target_link_libraries(mmvq_il_parity PRIVATE strata_kernels ${_strata_gpu_runtime_target})
    add_test(NAME mmvq_il_parity COMMAND mmvq_il_parity)
    # the wide dequant kernels (a thread per 8 values; sm_61's default), bitwise vs the per-group ones (GPU, synthetic,
    # no model; --bench adds old/new timings)
    add_executable(dequant_wide_parity src/kernels/dequant_wide_parity.cpp)
    target_include_directories(dequant_wide_parity PRIVATE ${_strata_gpu_include_directories})
    target_link_libraries(dequant_wide_parity PRIVATE strata_kernels ${_strata_gpu_runtime_target})
    add_test(NAME dequant_wide_parity COMMAND dequant_wide_parity)
  endif()
  # programmatic dependent launch (pdl.hpp) in a captured verify-shaped chain: bitwise vs ordinary launches over
  # replays with poisoned buffers, and only kernel-to-kernel programmatic edges (GPU, synthetic, no model; the edges
  # are only made on sm_90+, elsewhere it checks that there are none).  NVIDIA only: PDL is a CUDA feature.
  if(STRATA_ENABLE_CUDA AND NOT STRATA_HIP_GFX906)
    add_executable(pdl_parity src/kernels/pdl_parity.cpp)
    target_include_directories(pdl_parity PRIVATE ${_strata_gpu_include_directories})
    target_link_libraries(pdl_parity PRIVATE strata_kernels ${_strata_gpu_runtime_target})
    add_test(NAME pdl_parity COMMAND pdl_parity)
    # PDL is opt-in (STRATA_DF_PDL); the test checks the opted-in path, whose PDL capture forces programmatic launches
    set_tests_properties(pdl_parity PROPERTIES ENVIRONMENT "STRATA_DF_PDL=1")
  endif()
  # the IQ kernels that decode a weight part once for every column/entry, bitwise vs the per-column ones
  # (GPU, synthetic, no model; --bench adds old/new timings)
  add_executable(iq_multi_parity src/kernels/iq_multi_parity.cpp)
  target_include_directories(iq_multi_parity PRIVATE ${_strata_gpu_include_directories})
  target_link_libraries(iq_multi_parity PRIVATE strata_kernels ${_strata_gpu_runtime_target})
  add_test(NAME iq_multi_parity COMMAND iq_multi_parity)
  # the prompt path's DeltaNet recurrence: gdn_rec_kh_kernel against gdn_rec_cols_pipe_kernel, bitwise (GPU,
  # synthetic, no model; --bench adds timings).  NVIDIA only: cp.async and the CUDA occupancy API (not on gfx906)
  if(STRATA_ENABLE_CUDA AND NOT STRATA_HIP_GFX906)
    # Independent pinned expert uploads: byte parity and stream/event ordering.
    add_executable(dma_batch_parity tests/core/dma_batch_parity.cu)
    target_include_directories(dma_batch_parity PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
    target_link_libraries(dma_batch_parity PRIVATE ${_strata_gpu_runtime_target})
    add_test(NAME dma_batch_parity COMMAND dma_batch_parity)
    set_tests_properties(dma_batch_parity PROPERTIES TIMEOUT 120)
    if(WIN32)
      add_test(NAME dma_batch_registered_parity COMMAND dma_batch_parity --registered)
      set_tests_properties(dma_batch_registered_parity PROPERTIES TIMEOUT 120)
    endif()
    add_executable(gdn_rec_parity src/prefill/gdn_rec_parity.cu)
    target_link_libraries(gdn_rec_parity PRIVATE ${_strata_gpu_runtime_target})
    add_test(NAME gdn_rec_parity COMMAND gdn_rec_parity)
    # the prompt path's BF16 products against cuBLAS's own BF16 GEMM, within fp32 rounding (sm_7x runs them on the
    # FP16 tensor cores; GPU, synthetic, no model; --bench adds timings)
    add_executable(gemm_bf16_parity src/prefill/gemm_bf16_parity.cu)
    target_link_libraries(gemm_bf16_parity PRIVATE strata_prefill )
    add_test(NAME gemm_bf16_parity COMMAND gemm_bf16_parity)
    # Pascal, opt-in STRATA_DENSE_F32=1: the dense projections' FP16 GEMMs as cublasSgemm on FP32 copies, bitwise vs
    # the cuBLAS FP16 call over a sweep of chunk sizes (GPU, synthetic, no model; --full sweeps every size, --bench
    # adds timings).  Not a ctest: whether the two agree is a property of the cuBLAS build, which this checks.
    add_executable(gemm_f32_parity src/prefill/gemm_f32_parity.cu)
    target_link_libraries(gemm_f32_parity PRIVATE strata_prefill )
  endif()
  # native_expert_grouped's group stride and fused SwiGLU + q8_1, bitwise vs the launches before them (GPU, synthetic,
  # no model; --bench adds v1/new timings of a window's 48 calls in a graph)
  add_executable(native_grouped_parity src/kernels/native_grouped_parity.cpp)
  target_include_directories(native_grouped_parity PRIVATE ${_strata_gpu_include_directories})
  target_link_libraries(native_grouped_parity PRIVATE strata_kernels ${_strata_gpu_runtime_target})
  add_test(NAME native_grouped_parity COMMAND native_grouped_parity)
=======
  target_include_directories(iq_parity PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
  target_link_libraries(iq_parity PRIVATE strata_kernels CUDA::cudart)
>>>>>>> their fork (github.com-eddoursul-Strata-custom)
  add_executable(shared_expert_parity src/kernels/shared_expert_parity.cpp)
  target_include_directories(shared_expert_parity PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
  target_link_libraries(shared_expert_parity PRIVATE strata_kernels CUDA::cudart)
  add_test(NAME shared_expert_parity COMMAND shared_expert_parity --selftest)

  add_executable(gr_parity src/kernels/gr_parity.cpp)
  target_include_directories(gr_parity PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
  target_link_libraries(gr_parity PRIVATE strata_kernels CUDA::cudart)
  add_test(NAME gr_parity COMMAND gr_parity --selftest)

  add_executable(gdn_parity src/kernels/gdn_parity.cpp)
  target_include_directories(gdn_parity PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
  target_link_libraries(gdn_parity PRIVATE strata_kernels CUDA::cudart)
  add_test(NAME gdn_parity COMMAND gdn_parity --selftest)

  add_executable(s2_gemv_q8_parity src/kernels/s2_gemv_q8_parity.cpp)
  target_include_directories(s2_gemv_q8_parity PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
  target_link_libraries(s2_gemv_q8_parity PRIVATE strata_kernels CUDA::cudart)
  add_test(NAME s2_gemv_q8_parity COMMAND s2_gemv_q8_parity --selftest)

  add_executable(sampler_parity src/kernels/sampler_parity.cpp)
  target_include_directories(sampler_parity PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
  target_link_libraries(sampler_parity PRIVATE strata_kernels CUDA::cudart)
  add_test(NAME sampler_parity COMMAND sampler_parity --selftest)
  set_tests_properties(sampler_parity PROPERTIES ENVIRONMENT "STRATA_OLD_SAMPLER=0;STRATA_SAMPLER_ONE_BLOCK=0")
  # the same fixtures on the other two sampled paths: the one-block kernel and the 0.1.20 kernel
  add_test(NAME sampler_parity_one_block COMMAND sampler_parity --selftest)
  set_tests_properties(sampler_parity_one_block PROPERTIES ENVIRONMENT "STRATA_OLD_SAMPLER=0;STRATA_SAMPLER_ONE_BLOCK=1")
  add_test(NAME sampler_parity_old COMMAND sampler_parity --selftest)
  set_tests_properties(sampler_parity_old PROPERTIES ENVIRONMENT "STRATA_OLD_SAMPLER=1;STRATA_SAMPLER_ONE_BLOCK=0")

  add_executable(rope_parity src/kernels/rope_parity.cpp)
  target_include_directories(rope_parity PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
  target_link_libraries(rope_parity PRIVATE strata_kernels CUDA::cudart)
  add_test(NAME rope_parity COMMAND rope_parity --selftest)

  add_executable(quantize_act_parity src/kernels/quantize_act_parity.cpp)
  target_include_directories(quantize_act_parity PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
  target_link_libraries(quantize_act_parity PRIVATE strata_kernels CUDA::cudart)
  add_test(NAME quantize_act_parity COMMAND quantize_act_parity --selftest)

  add_executable(router_top10_parity src/kernels/router_top10_parity.cpp)
  target_include_directories(router_top10_parity PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
  target_link_libraries(router_top10_parity PRIVATE strata_kernels CUDA::cudart)
  add_test(NAME router_top10_parity COMMAND router_top10_parity --selftest)

  add_executable(s_gemv_parity src/kernels/s_gemv_parity.cpp)
  target_include_directories(s_gemv_parity PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
  target_link_libraries(s_gemv_parity PRIVATE strata_kernels CUDA::cudart)
  add_test(NAME s_gemv_parity COMMAND s_gemv_parity --selftest)

  add_executable(elementwise_parity src/kernels/elementwise_parity.cpp)
  target_include_directories(elementwise_parity PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
  target_link_libraries(elementwise_parity PRIVATE strata_kernels CUDA::cudart)
  add_test(NAME elementwise_parity COMMAND elementwise_parity --selftest)

  add_executable(bf16_gemv_parity src/kernels/bf16_gemv_parity.cpp)
  target_include_directories(bf16_gemv_parity PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
  target_link_libraries(bf16_gemv_parity PRIVATE strata_kernels CUDA::cudart)
  add_test(NAME bf16_gemv_parity COMMAND bf16_gemv_parity --selftest)

  add_executable(s_gemv_q8k_parity src/kernels/s_gemv_q8k_parity.cpp)
  target_include_directories(s_gemv_q8k_parity PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
  target_link_libraries(s_gemv_q8k_parity PRIVATE strata_kernels CUDA::cudart)
  add_test(NAME s_gemv_q8k_parity COMMAND s_gemv_q8k_parity --selftest)

  add_executable(qsa_parity src/kernels/qsa_parity.cpp)
  target_include_directories(qsa_parity PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
  target_link_libraries(qsa_parity PRIVATE strata_kernels CUDA::cudart)
  add_test(NAME qsa_parity COMMAND qsa_parity --selftest)

  # ---- P2.S4: the PLE n-gram hash, the IQ4_NL table read, and the PLE block ----
  # `ngram.cpp` is host-only but lives in `strata_kernels` because `ple.cu` shares its geometry constants, and
  # splitting them across two libraries would only add a layer.
  #
  # The pack, the original GGUF and the ggml capture are all REQUIRED inputs: this test's oracles are real
  # artifacts, so the working directory is pinned to the source tree and a missing input is a hard error
  # INSIDE the test.  A test that passes by finding nothing to compare is the failure mode this project keeps
  # re-learning.
  add_executable(ple_parity src/kernels/ple_parity.cpp)
  target_include_directories(ple_parity PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
  target_link_libraries(ple_parity PRIVATE strata_kernels CUDA::cudart)
  set(STRATA_PLE_FIXTURE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/bench/micro" CACHE PATH
      "Directory containing required ple_in.bin and ple_out.bin parity fixtures")
  add_test(NAME ple_parity COMMAND ple_parity --selftest
    --in "${STRATA_PLE_FIXTURE_DIR}/ple_in.bin" --out "${STRATA_PLE_FIXTURE_DIR}/ple_out.bin"
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})

  # Plan v0.3 P7: INT8 KV storage vs a host reference and vs the FP16 path (GPU, synthetic, no model).
  add_executable(kv_q8_parity src/kernels/kv_q8_parity.cpp)
  target_include_directories(kv_q8_parity PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
  target_link_libraries(kv_q8_parity PRIVATE strata_kernels CUDA::cudart)
  add_test(NAME kv_q8_parity COMMAND kv_q8_parity)

  # The verify window's batched kernels vs the per-token kernel sequences they replace, bitwise (synthetic data).
  add_executable(verify_parity src/kernels/verify_parity.cpp)
  target_include_directories(verify_parity PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
  target_link_libraries(verify_parity PRIVATE strata_kernels strata_prefill strata_kernels_cpu CUDA::cudart)
  add_test(NAME verify_parity COMMAND verify_parity --selftest)

  # The QSA block scores and top-k against the kernels they replaced, kept in the test: bitwise (synthetic data).
  add_executable(qsa_select_parity src/kernels/qsa_select_parity.cu)
  target_link_libraries(qsa_select_parity PRIVATE strata_kernels CUDA::cudart)
  add_test(NAME qsa_select_parity COMMAND qsa_select_parity)
  # The MMVQ of 2..8 tokens from interleaved activations against native_mmvq: bitwise (synthetic matrices).
  add_executable(mmvq_il_parity src/kernels/mmvq_il_parity.cu)
  target_link_libraries(mmvq_il_parity PRIVATE strata_kernels CUDA::cudart)
  add_test(NAME mmvq_il_parity COMMAND mmvq_il_parity)

  # native_mmvq's multi_exact contract, bitwise: a T-column call against T single-column calls on the same Q8_1
  # bytes, with the flag off as the negative control (synthetic, GPU, no model; the control assumes 32-lane warps)
  add_executable(mmvq_multi_parity src/kernels/mmvq_multi_parity.cpp)
  target_include_directories(mmvq_multi_parity PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
  target_link_libraries(mmvq_multi_parity PRIVATE strata_kernels CUDA::cudart)
  add_test(NAME mmvq_multi_parity COMMAND mmvq_multi_parity)

  # Plan v0.3 P3: native MMVQ for 1..8 tokens (bitwise selftest + --bench timing); synthetic data, no model.
  if(STRATA_BUILD_TESTS AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/bench/micro/native_mmvq_multi.cpp")
    add_executable(native_mmvq_multi bench/micro/native_mmvq_multi.cpp)
    target_include_directories(native_mmvq_multi PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
    target_link_libraries(native_mmvq_multi PRIVATE strata_kernels CUDA::cudart)
    add_test(NAME native_mmvq_multi COMMAND native_mmvq_multi --selftest)
  endif()

  # Plan v0.3 P0.1: working-set lock of a host region (CPU only).
  add_executable(platform_memory_test src/platform/memory_test.cpp)
  target_link_libraries(platform_memory_test PRIVATE strata_core CUDA::cudart)
  add_test(NAME platform_memory_test COMMAND platform_memory_test)

  # Plan v0.3 P2: the n-gram table read straight from the SSD. The selftest needs only the CPU and a temp file.
  add_executable(ple_reader_test src/ngram/ple_reader_test.cpp)
  target_include_directories(ple_reader_test PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
  target_link_libraries(ple_reader_test PRIVATE strata_kernels CUDA::cudart)
  add_test(NAME ple_reader_selftest COMMAND ple_reader_test --selftest)

  # The GPU test is registered only when CUDA is on, and it runs ON THE TARGET PC.  `--selftest` allocates
  # and writes real VRAM, so a machine without a device fails this test rather than skipping it silently -
  # which is correct: an engine that targets one GPU should not report a pass for having no GPU.
  add_test(NAME cuda_device_selftest COMMAND strata-device --selftest)

  # ---- P2.S5: the GraphRegistry --------------------------------------------------
  # A .cu because the test needs real kernels to capture; it links `strata_core` and not `strata_kernels`,
  # because what it exercises is the capture machinery and not any kernel in it.
  if(STRATA_BUILD_TESTS)
  add_executable(graph_registry_test bench/micro/graph_registry_test.cu)
  target_include_directories(graph_registry_test PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
  target_link_libraries(graph_registry_test PRIVATE strata_core CUDA::cudart)
  add_test(NAME graph_registry_test COMMAND graph_registry_test)

  # ---- P2.S5's GATE: is a device<->pinned-memory handoff graph-capturable? --------------------
  # The doorbell protocol puts the D2H of `x_f` and the miss list INSIDE G[l] - that is the whole overlap.  The
  # phase doc says to verify this "before building on it", and this project has already lost two rounds to
  # building first (a memory-only spin that DEADLOCKS on Windows, and a `cudaEventRecord` inside a capture that
  # is SILENTLY DROPPED).  It links only cudart: it is about the capture machinery, not about any kernel.
  add_executable(pinned_capture bench/micro/pinned_capture.cu)
  target_include_directories(pinned_capture PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
  target_link_libraries(pinned_capture PRIVATE CUDA::cudart)
  add_test(NAME pinned_capture COMMAND pinned_capture)
  endif()

  # Tells tests/CMakeLists.txt to add the CUDA-dependent Catch2 target.  It cannot be declared there
  # unconditionally: tests/ is added at the end of this file, but the flag has to be known before it runs.
  set(STRATA_CUDA_TESTS ON)
endif()

# ---- strata_kernels_cpu (P2.S3) ----
#
# A SEPARATE LIBRARY from `strata_kernels`, and `/arch:AVX512` is set on the SOURCE rather than on the target
# so that the flag applies only to the TU that needs it.  Both parts matter:
#
#   * the CPU expert kernel uses `_mm256_dpbusd_epi32` (AVX512-VNNI) and `_mm256_multishift_epi64_epi8`
#     (AVX512-VBMI); without the flag MSVC reports undeclared intrinsics and never mentions AVX-512;
#   * a TU compiled WITH `/arch:AVX512` may use AVX-512 in ANY of its code, so the scalar fallback and the
#     test drivers must NOT be in it - otherwise the fallback is not a fallback, it is the same bet with
#     worse opt-out.  `cpu_require_expert_support()` therefore refuses at startup on a CPU without the
#     features, and the fallback exists for tests, exactly as P2.S3 specifies.
# Plan v0.3 P6: speculation components that need no GPU.
add_library(strata_spec STATIC src/spec/suffix_drafter.cpp src/spec/controller.cpp)
target_include_directories(strata_spec PUBLIC include)
add_executable(suffix_drafter_test src/spec/suffix_drafter_test.cpp)
target_link_libraries(suffix_drafter_test PRIVATE strata_spec)
add_test(NAME suffix_drafter_test COMMAND suffix_drafter_test)
if(STRATA_BUILD_TESTS)
  add_executable(controller_test src/spec/controller_test.cpp)
  target_link_libraries(controller_test PRIVATE strata_spec)
  add_test(NAME controller_test COMMAND controller_test)
  # the f32 -> bf16 rounding against ggml's, NaN included, over every f32 bit pattern (header-only, CPU)
  add_executable(bf16_bits_test src/kernels/bf16_bits_test.cpp)
  target_include_directories(bf16_bits_test PRIVATE include)
  add_test(NAME bf16_bits_test COMMAND bf16_bits_test)
endif()

add_library(strata_kernels_cpu STATIC src/kernels/cpu/expert.cpp src/kernels/cpu/pool.cpp
            src/kernels/cpu/expert_layout.cpp src/kernels/cpu/q2_avx2.cpp)
target_include_directories(strata_kernels_cpu PUBLIC include)
# Plan v0.3 P6: only the kernel files get the AVX-512 / AVX2 flags; the pool and the probe run on any x86-64,
# and the engine picks the kernels at run time (cpu_avx512_ok).
if(MSVC)
  set_source_files_properties(src/kernels/cpu/expert.cpp PROPERTIES COMPILE_OPTIONS "/arch:AVX512")
  set_source_files_properties(src/kernels/cpu/q2_avx2.cpp PROPERTIES COMPILE_OPTIONS "/arch:AVX2")
else()
  # GCC/Clang spell it differently, and `-mavx512vnni` alone is not enough: VBMI is a separate feature bit.
  set_source_files_properties(src/kernels/cpu/expert.cpp
                              PROPERTIES COMPILE_OPTIONS "-mavx512f;-mavx512bw;-mavx512vl;-mavx512dq;-mavx512vnni;-mavx512vbmi;-mf16c;-mfma")
  set_source_files_properties(src/kernels/cpu/q2_avx2.cpp PROPERTIES COMPILE_OPTIONS "-mavx2;-mfma;-mf16c")
endif()
# AVX-VNNI (Alder Lake, Zen 5 and later): the AVX-2 expert kernels' vpdpwssd / vpdpbusd copies (q2_avx2.cpp,
# iq_avx2.cpp; *_rows.inl), taken at run time where cpu_avxvnni_ok(); the same integer sums.  GCC and Clang compile
# only those functions with target("avxvnni"), MSVC takes the intrinsics under /arch:AVX2.  A compiler without them
# (GCC < 11, Clang < 12) builds the files without the copies.  (Upstream's, b211ba17.)
include(CheckCXXSourceCompiles)
if(MSVC)
  set(CMAKE_REQUIRED_FLAGS "/arch:AVX2")
else()
  set(CMAKE_REQUIRED_FLAGS "-mavx2 -mfma -mf16c")
endif()
check_cxx_source_compiles([=[
#include <immintrin.h>
#if defined(__GNUC__) || defined(__clang__)
__attribute__((target("avxvnni")))
#endif
static __m256i dots(__m256i a, __m256i b) { return _mm256_dpbusd_avx_epi32(_mm256_dpwssd_avx_epi32(a, a, b), a, b); }
int main() {
    volatile int x = 1;
    return _mm256_extract_epi32(dots(_mm256_set1_epi32(x), _mm256_set1_epi32(x)), 0) == 0;
}
]=] STRATA_HAVE_AVXVNNI_INTRINSICS)
unset(CMAKE_REQUIRED_FLAGS)
if(STRATA_HAVE_AVXVNNI_INTRINSICS)
  set(_strata_avxvnni STRATA_AVXVNNI=1)
else()
  set(_strata_avxvnni STRATA_AVXVNNI=0)
endif()
set_property(SOURCE src/kernels/cpu/q2_avx2.cpp APPEND PROPERTY COMPILE_DEFINITIONS ${_strata_avxvnni})
find_package(Threads REQUIRED)
target_link_libraries(strata_kernels_cpu PUBLIC Threads::Threads)
# The expert pool against serial runs; `--stress` needs no pack (synthetic experts, idle gaps that put the workers
# to sleep).
add_executable(pool_test src/kernels/cpu/pool_test.cpp)
target_link_libraries(pool_test PRIVATE strata_kernels_cpu)

# ---- plan v0.3 P6: the i-quant models' CPU experts (IQ2_XS / IQ3_XXS files) ----
#
# The CPU computes a native expert with ggml-cpu's own dot products (`ggml_get_type_traits_cpu`), so an IQ
<<<<<<< this fork
# weight means exactly what it means in llama.cpp.  ggml is built static from the vendored llama.cpp
# submodule (STRATA_GGML_DIR, pinned above).
if(STRATA_PREFILL_MMQ AND NOT STRATA_NATIVE_EXPERTS)
  message(FATAL_ERROR "STRATA_PREFILL_MMQ requires STRATA_NATIVE_EXPERTS=ON for GGML sources")
endif()
if(STRATA_ORCA_Q4KS_MMQ AND NOT STRATA_NATIVE_EXPERTS)
  message(FATAL_ERROR "STRATA_ORCA_Q4KS_MMQ requires STRATA_NATIVE_EXPERTS=ON for GGML sources")
endif()
=======
# weight means exactly what it means in llama.cpp.  ggml is built static from a llama.cpp checkout: a local one
# with -DSTRATA_GGML_DIR=..., otherwise the pinned upstream commit (third_party/ggml/VERSION.txt).
option(STRATA_NATIVE_EXPERTS "Native (IQ) expert support through ggml-cpu" ON)
set(STRATA_GGML_DIR "" CACHE PATH "llama.cpp checkout to take ggml from (empty: fetch the pinned commit)")
>>>>>>> their fork (github.com-eddoursul-Strata-custom)
if(STRATA_NATIVE_EXPERTS)
  enable_language(C)
  set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
  set(GGML_STATIC ON CACHE BOOL "" FORCE)
  set(GGML_CUDA OFF CACHE BOOL "" FORCE)
  set(GGML_OPENMP OFF CACHE BOOL "" FORCE)
  set(GGML_BACKEND_DL OFF CACHE BOOL "" FORCE)
  set(GGML_BUILD_TESTS OFF CACHE BOOL "" FORCE)
  set(GGML_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
  if(STRATA_PORTABLE)
    set(GGML_NATIVE OFF CACHE BOOL "" FORCE)
    foreach(_f GGML_AVX GGML_AVX2 GGML_FMA GGML_F16C GGML_BMI2)
      set(${_f} ON CACHE BOOL "" FORCE)
    endforeach()
  else()
    set(GGML_NATIVE ON CACHE BOOL "" FORCE)
  endif()
  add_subdirectory(${STRATA_GGML_DIR}/ggml ${CMAKE_BINARY_DIR}/ggml EXCLUDE_FROM_ALL)
  # part of strata_kernels_cpu (the pool calls these and they call its CPU probe: one library, so GNU ld's
  # one-pass static linking has no cycle to resolve)
  target_sources(strata_kernels_cpu PRIVATE src/kernels/cpu/native_expert.cpp src/kernels/cpu/iq_avx512.cpp
                                           src/kernels/cpu/iq_avx2.cpp src/kernels/cpu/kq_avx2.cpp
                                           src/kernels/cpu/kq_avx1.cpp src/kernels/cpu/q2_avx1.cpp src/kernels/cpu/iq_avx1.cpp
                                           src/kernels/cpu/nvfp4_avx512.cpp src/kernels/cpu/nvfp4_avx2.cpp)
  target_include_directories(strata_kernels_cpu PRIVATE ${STRATA_GGML_DIR}/ggml/src)
  if(MSVC)
    set_source_files_properties(src/kernels/cpu/iq_avx512.cpp src/kernels/cpu/nvfp4_avx512.cpp
                                PROPERTIES COMPILE_OPTIONS "/arch:AVX512")
    set_source_files_properties(src/kernels/cpu/iq_avx2.cpp PROPERTIES COMPILE_OPTIONS "/arch:AVX2")
    set_source_files_properties(src/kernels/cpu/kq_avx2.cpp PROPERTIES COMPILE_OPTIONS "/arch:AVX2")
<<<<<<< this fork
    set_source_files_properties(src/kernels/cpu/nvfp4_avx2.cpp PROPERTIES COMPILE_OPTIONS "/arch:AVX2")
    # the router lookahead's AVX1 dot (older CPUs; called behind cpu_avx1_ok): no FMA, no 256-bit integer ops
    set_source_files_properties(src/kernels/cpu/kq_avx1.cpp PROPERTIES COMPILE_OPTIONS "/arch:AVX")
    # the older CPUs' Q2_0 rows and activation quantizer (q2_avx1.cpp, issue #1699): 128-bit integer lanes,
    # no FMA/F16C, its own #error guard; called only behind cpu_avx1_ok() && !cpu_avx2_ok()
    set_source_files_properties(src/kernels/cpu/q2_avx1.cpp PROPERTIES COMPILE_OPTIONS "/arch:AVX")
    # the older CPUs' i-quant expert rows (iq_avx1.hpp): 128-bit SSSE3/SSE2 integer, SSE2 float, no FMA/F16C
    set_source_files_properties(src/kernels/cpu/iq_avx1.cpp PROPERTIES COMPILE_OPTIONS "/arch:AVX")
=======
    # older CPUs' kernels (AVX1, called behind cpu_avx1_ok() && !cpu_avx2_ok()): no FMA, no 256-bit integer ops
    set_source_files_properties(src/kernels/cpu/kq_avx1.cpp src/kernels/cpu/q2_avx1.cpp src/kernels/cpu/iq_avx1.cpp
                                PROPERTIES COMPILE_OPTIONS "/arch:AVX")
>>>>>>> their fork (github.com-eddoursul-Strata-custom)
  else()
    set_source_files_properties(src/kernels/cpu/iq_avx512.cpp src/kernels/cpu/nvfp4_avx512.cpp PROPERTIES COMPILE_OPTIONS
                                "-mavx512f;-mavx512bw;-mavx512vl;-mavx512dq;-mavx512vnni;-mavx512vbmi;-mf16c;-mfma")
    set_source_files_properties(src/kernels/cpu/iq_avx2.cpp PROPERTIES COMPILE_OPTIONS "-mavx2;-mfma;-mf16c")
    set_source_files_properties(src/kernels/cpu/kq_avx2.cpp PROPERTIES COMPILE_OPTIONS "-mavx2;-mfma;-mf16c")
    set_source_files_properties(src/kernels/cpu/nvfp4_avx2.cpp PROPERTIES COMPILE_OPTIONS "-mavx2;-mfma;-mf16c")
    set_source_files_properties(src/kernels/cpu/kq_avx1.cpp PROPERTIES COMPILE_OPTIONS "-mavx")
    set_source_files_properties(src/kernels/cpu/q2_avx1.cpp src/kernels/cpu/iq_avx1.cpp PROPERTIES COMPILE_OPTIONS
                                "-msse4.2;-mavx;-mno-avx2;-mno-fma;-mno-f16c")
  endif()
  set_property(SOURCE src/kernels/cpu/iq_avx2.cpp APPEND PROPERTY COMPILE_DEFINITIONS ${_strata_avxvnni})
  target_link_libraries(strata_kernels_cpu PUBLIC ggml-cpu ggml-base)
  target_compile_definitions(strata_kernels_cpu PUBLIC STRATA_NATIVE_EXPERTS=1)
  # the AVX-2 expert kernels' variants (iq_avx2.cpp, q2_avx2.cpp; with AVX-VNNI where the CPU has it) bit for bit
  # against each other at 1..8 tokens and against ggml-cpu's dot, on random blocks; --bench: ms per expert on one
  # thread, the weights streamed from DRAM
  add_executable(iq_avx2_parity src/kernels/cpu/iq_avx2_parity.cpp)
  target_link_libraries(iq_avx2_parity PRIVATE strata_kernels_cpu)
<<<<<<< this fork
  # MODEL-UPDATE: the pool's expert rows per format on T threads against the RAM bandwidth (a benchmark, no test)
  add_executable(kq_cpu_bench src/kernels/cpu/kq_cpu_bench.cpp)
  target_link_libraries(kq_cpu_bench PRIVATE strata_kernels_cpu)
  if(STRATA_BUILD_TESTS)
    add_test(NAME iq_avx2_parity COMMAND iq_avx2_parity)
    add_test(NAME iq_avx2_parity_iq3s_mt1 COMMAND ${CMAKE_COMMAND} -E env STRATA_IQ3S_MT1=1 $<TARGET_FILE:iq_avx2_parity>)
  endif()
  # the older CPUs' AVX1 expert rows (iq_avx1.cpp) against ggml-cpu's dots and against their own one-token
  # rows at every width; --bench: ms per expert, the weights streamed from DRAM.  Runs on any AVX CPU:
  # on an AVX2 machine it checks the older-CPU path as an AVX-only machine would run it.
  add_executable(iq_avx1_parity src/kernels/cpu/iq_avx1_parity.cpp)
  target_link_libraries(iq_avx1_parity PRIVATE strata_kernels_cpu)
  if(STRATA_BUILD_TESTS)
    add_test(NAME iq_avx1_parity COMMAND iq_avx1_parity)
  endif()
=======
  add_test(NAME iq_avx2_parity COMMAND iq_avx2_parity)
>>>>>>> their fork (github.com-eddoursul-Strata-custom)
  # the AVX-2 Q8_K activation quantizer against ggml's, byte for byte (--bench: its time)
  add_executable(q8k_quant_parity src/kernels/cpu/q8k_quant_parity.cpp)
  target_link_libraries(q8k_quant_parity PRIVATE strata_kernels_cpu)
  add_test(NAME q8k_quant_parity COMMAND q8k_quant_parity)
  if(STRATA_ENABLE_CUDA)
<<<<<<< this fork
    # prompt-speed plan step 2b: llama.cpp's MMQ kernels (ggml-cuda, MIT) for the prompt path's experts, compiled
    # from the same checkout as ggml above - only mmq.cuh's instances for the packs' types and the quantizer, with
    # the few host symbols of ggml-cuda.cu they use in src/prefill/ggml_cuda_host.cu
    set(_strata_gg ${STRATA_GGML_DIR}/ggml)
    set(_strata_mmq ${_strata_gg}/src/ggml-cuda)
    # #136: the Q2_0 pack's prompt experts on Strata's own int8 kernels (moe_fused.cu, opt-in STRATA_PF_FUSED=1; CUDA
    # only: the HIP branch below keeps MMQ, and an sm_75 device or an sm_75-only build reports them unavailable)
    add_library(strata_mmq STATIC src/prefill/moe_mmq.cu src/prefill/moe_fused.cu src/prefill/moe_fused_iq.cu
                src/prefill/mmq_nvfp4_w4a8.cu
                src/prefill/ggml_cuda_host.cu ${_strata_mmq}/quantize.cu)
    # q8_0: the drafter's matrices (E-9); STRATA_ORCA_Q4KS_MMQ adds q5_0 (#296, in _strata_mmq_types), and
    # STRATA_MMQ_KQUANTS q4_k q5_k q5_1 q6_k, the experts of Unsloth's UD-Q4/Q5/Q6_K_XL (docs/QUANTS.md; the dispatch
    # follows eddoursul/Strata 7706c1b).  MODEL-UPDATE Phase 1: on CUDA they are a library of their own, strata_kq
    # (below), loaded only for a model with these experts: the engine loads every kernel of its own binary at start
    # (CUDA_MODULE_LOADING=EAGER), and these four took ~3.8 MB of VRAM (3 expert slots of IQ3_XXS on a P100) from every
    # model.  gfx906 (HIP: code objects load on first use) keeps them in strata_mmq.
    set(_strata_mmq_cuda_types ${_strata_mmq_types})
    set(_strata_kq_types q4_k q5_k q5_1 q6_k)
    set(_strata_kq_dso OFF)
    if(STRATA_MMQ_KQUANTS AND STRATA_HIP_GFX906)
      list(APPEND _strata_mmq_cuda_types ${_strata_kq_types})
    elseif(STRATA_MMQ_KQUANTS)
      set(_strata_kq_dso ON)
    endif()
    foreach(_t IN LISTS _strata_mmq_cuda_types)
=======
    # llama.cpp's MMQ kernels (ggml-cuda, MIT) for the prompt path's experts, from the same checkout as ggml: mmq.cuh's
    # instances for the expert types of ISTA-DASLab's and unsloth's files and the quantizer, with the few host
    # symbols of ggml-cuda.cu they use in src/prefill/ggml_cuda_host.cu
    if(STRATA_GGML_DIR)
      set(_strata_gg ${STRATA_GGML_DIR}/ggml)
    else()
      set(_strata_gg ${strata_llamacpp_SOURCE_DIR}/ggml)
    endif()
    set(_strata_mmq ${_strata_gg}/src/ggml-cuda)
    add_library(strata_mmq STATIC src/prefill/moe_mmq.cu src/prefill/ggml_cuda_host.cu ${_strata_mmq}/quantize.cu)
    foreach(_t q2_0 iq1_s iq2_xxs iq2_xs iq2_s iq3_xxs iq3_s iq4_nl iq4_xs q4_k q5_k q6_k q5_1 q8_0)
>>>>>>> their fork (github.com-eddoursul-Strata-custom)
      target_sources(strata_mmq PRIVATE ${_strata_mmq}/template-instances/mmq-instance-${_t}.cu)
    endforeach()
    # NVFP4: W4A8 (default) in mmq_nvfp4_w4a8.cu with the plain architectures; the opt-in W4A4 needs RTX 50's FP4 MMA,
    # which exists only in the arch-specific target, so its one unit alone is built with 12x -> 12xa (the default
    # binary is unchanged and runs on sm_121; a card without that image falls back to W4A8 at run time)
    set(_strata_w4a4_archs ${CMAKE_CUDA_ARCHITECTURES})
    # (two steps: CMake before 3.29 rejects a \2 whose optional group did not match, "120" alone)
    list(TRANSFORM _strata_w4a4_archs REPLACE "^(12[0-9])$" "\\1a")
    list(TRANSFORM _strata_w4a4_archs REPLACE "^(12[0-9])-(real|virtual)$" "\\1a-\\2")
    add_library(strata_mmq_w4a4 STATIC src/prefill/mmq_nvfp4_w4a4.cu)
    set_target_properties(strata_mmq_w4a4 PROPERTIES CUDA_ARCHITECTURES "${_strata_w4a4_archs}")
    target_include_directories(strata_mmq_w4a4 PRIVATE ${_strata_gg}/include ${_strata_gg}/src ${_strata_mmq})
    target_compile_options(strata_mmq_w4a4 PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda -use_fast_math>)
    if(MSVC)
      target_compile_options(strata_mmq_w4a4 PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/bigobj>)
    endif()
    target_link_libraries(strata_mmq_w4a4 PUBLIC ggml-base ${_strata_gpu_runtime_target})
    target_link_libraries(strata_mmq PRIVATE strata_mmq_w4a4)
    target_include_directories(strata_mmq PRIVATE ${_strata_gg}/include ${_strata_gg}/src ${_strata_mmq}
                               PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
    target_compile_options(strata_mmq PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda -use_fast_math>)
    if(MSVC)
      target_compile_options(strata_mmq PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/bigobj>)
    endif()
<<<<<<< this fork
    # GGML's CUDA error reporting calls cuGetErrorString from the driver API.
    target_link_libraries(strata_mmq PUBLIC ggml-base ${_strata_gpu_runtime_target})
    if(TARGET CUDA::cuda_driver)   # not in the HIP (gfx906) build, whose CUDA::cudart is a stand-in
      target_link_libraries(strata_mmq PUBLIC CUDA::cuda_driver)
    endif()
    if(STRATA_MMQ_KQUANTS)
      target_compile_definitions(strata_mmq PRIVATE STRATA_MMQ_KQUANTS=1)
    endif()
    if(_strata_kq_dso)
      # strata_kq: the K-quant MMQ instances with their own copy of ggml_cuda_host.cu's symbols and the static CUDA
      # runtime, next to the engine (libstrata_kq.so / strata_kq.dll); moe_mmq.cu opens it when a model asks for one
      # of these formats (STRATA_MMQ_KQ=1).  Without the file the engine runs them on the FP16 prompt path, as before.
      target_compile_definitions(strata_mmq PRIVATE STRATA_KQ_DSO=1)
      target_sources(strata_mmq PRIVATE src/prefill/mmq_kq_loader.cpp)
      if(NOT WIN32)
        target_link_libraries(strata_mmq PUBLIC ${CMAKE_DL_LIBS})
      endif()
      add_library(strata_kq SHARED src/prefill/mmq_kq_lib.cu src/prefill/ggml_cuda_host.cu)
      foreach(_t IN LISTS _strata_kq_types)
        target_sources(strata_kq PRIVATE ${_strata_mmq}/template-instances/mmq-instance-${_t}.cu)
      endforeach()
      target_include_directories(strata_kq PRIVATE ${_strata_gg}/include ${_strata_gg}/src ${_strata_mmq}
                                 ${CMAKE_CURRENT_SOURCE_DIR}/include)
      target_compile_options(strata_kq PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda -use_fast_math>)
      if(MSVC)
        target_compile_options(strata_kq PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/bigobj>)
      endif()
      set_target_properties(strata_kq PROPERTIES CUDA_RUNTIME_LIBRARY Static POSITION_INDEPENDENT_CODE ON
                            CXX_VISIBILITY_PRESET hidden CUDA_VISIBILITY_PRESET hidden)
      # ggml-base is static: position-independent so a shared library can hold it (the same code otherwise)
      set_target_properties(ggml-base PROPERTIES POSITION_INDEPENDENT_CODE ON)
      target_link_libraries(strata_kq PRIVATE ggml-base)
      if(TARGET CUDA::cuda_driver)
        target_link_libraries(strata_kq PRIVATE CUDA::cuda_driver)
      endif()
      add_dependencies(strata_mmq strata_kq)
    endif()
    if(STRATA_ORCA_Q4KS_MMQ)
      target_compile_definitions(strata_mmq PRIVATE STRATA_ORCA_Q4KS_MMQ=1)
    endif()
    target_link_libraries(strata_prefill PUBLIC strata_mmq)
    target_compile_definitions(strata_prefill PUBLIC STRATA_PREFILL_MMQ=1 STRATA_PREFILL_FUSED=1)
    if(STRATA_BUILD_TESTS)
      # #136: the fused int8 experts against a double-precision reference and the MMQ path (exit 77 below sm_80)
      add_executable(prefill_fused_moe_test tests/cuda/prefill_fused_moe_test.cpp)
      target_link_libraries(prefill_fused_moe_test PRIVATE strata_mmq ggml-base ${_strata_gpu_runtime_target})
      add_test(NAME prefill_fused_moe_test COMMAND prefill_fused_moe_test)
      set_tests_properties(prefill_fused_moe_test PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 300)
      # #136: the same for the native packs' formats (IQ2_XXS IQ2_XS IQ2_S IQ3_XXS IQ3_S IQ4_XS, down Q2_0 IQ4_NL)
      add_executable(prefill_fused_iq_test tests/cuda/prefill_fused_iq_test.cpp)
      target_link_libraries(prefill_fused_iq_test PRIVATE strata_mmq ggml-base ${_strata_gpu_runtime_target})
      add_test(NAME prefill_fused_iq_test COMMAND prefill_fused_iq_test)
      set_tests_properties(prefill_fused_iq_test PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 600)
    endif()
    if(STRATA_BUILD_TESTS AND STRATA_MMQ_KQUANTS)
      # the UD-Q4_K_XL expert formats through MMQ against ggml's dequantized weights (synthetic, a few seconds)
      add_executable(prefill_mmq_kquant_test tests/cuda/prefill_mmq_kquant_test.cpp)
      target_link_libraries(prefill_mmq_kquant_test PRIVATE strata_mmq ggml-base ${_strata_gpu_runtime_target})
      add_test(NAME prefill_mmq_kquant_test COMMAND prefill_mmq_kquant_test)
      # the K-quant MMQ is opt-in at run time (STRATA_MMQ_KQ=1), and on CUDA it is strata_kq next to the test
      set_tests_properties(prefill_mmq_kquant_test PROPERTIES TIMEOUT 120 ENVIRONMENT STRATA_MMQ_KQ=1)
    endif()
    # the NVFP4 expert products (w4a8 / w4a4, STRATA_PREFILL_NVFP4) against a double-precision reference
    add_executable(mmq_nvfp4_parity src/prefill/mmq_nvfp4_parity.cpp)
    target_link_libraries(mmq_nvfp4_parity PRIVATE strata_mmq ggml-base CUDA::cudart)
    if(NOT MSVC)   # its FP16 rounding helper uses the F16C intrinsics (GCC/Clang need the target flag)
      target_compile_options(mmq_nvfp4_parity PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-mf16c>)
    endif()
    # the routed-expert products (MMQ) timed at the prompt path's shapes (16 experts a launch)
    add_executable(mmq_expert_bench src/prefill/mmq_expert_bench.cpp)
    target_link_libraries(mmq_expert_bench PRIVATE strata_mmq ggml-base CUDA::cudart)
  elseif(STRATA_PREFILL_MMQ)
    # Compile only the pinned GGML MMQ and activation-quantizer sources used by prefill. This uses
    # common.cuh's native HIP path without enabling a second GGML HIP backend.
    set(_strata_gg "${STRATA_GGML_DIR}/ggml")
    set(_strata_mmq "${_strata_gg}/src/ggml-cuda")
    if(NOT EXISTS "${_strata_mmq}/common.cuh" OR NOT EXISTS "${_strata_mmq}/quantize.cu")
      message(FATAL_ERROR "The selected GGML source is missing the ggml-cuda MMQ sources: ${_strata_mmq}")
    endif()
    # Aurora: moe_fused.cu's gfx11 and moe_fused_iq.cu's gfx11 / gfx12 WMMA kernels for the prompt experts (opt-in STRATA_PF_FUSED=1)
    set(_strata_mmq_sources
      src/prefill/moe_mmq.cu
      src/prefill/moe_fused.cu
      src/prefill/moe_fused_iq.cu
      src/prefill/ggml_cuda_host.cu
      "${_strata_mmq}/quantize.cu")
    if(STRATA_MMQ_KQUANTS)
      # the same K-quant instances as the CUDA branch: on HIP they serve the dense GGUF projections
      # of Gemm::native (STRATA_DENSE_MMQ=1), the mixed-quant packs' attention/PLE tensors
      list(APPEND _strata_mmq_types q4_k q5_k q5_1 q6_k)
    endif()
    foreach(_t IN LISTS _strata_mmq_types)
      list(APPEND _strata_mmq_sources "${_strata_mmq}/template-instances/mmq-instance-${_t}.cu")
    endforeach()
    set_source_files_properties(${_strata_mmq_sources} PROPERTIES LANGUAGE HIP)
    # moe_fused.cu is Strata's own CUDA-shaped source: it takes the HIP compat headers, ggml's MMQ sources do not
    set_source_files_properties(src/prefill/moe_fused.cu src/prefill/moe_fused_iq.cu
                                PROPERTIES INCLUDE_DIRECTORIES "${STRATA_HIP_COMPAT_INCLUDE_DIR}")
    add_library(strata_mmq STATIC ${_strata_mmq_sources})
    target_include_directories(strata_mmq
      PRIVATE "${_strata_gg}/include" "${_strata_gg}/src" "${_strata_mmq}"
      PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
    target_compile_definitions(strata_mmq PRIVATE GGML_USE_HIP=1)
    if(STRATA_MMQ_KQUANTS)
      target_compile_definitions(strata_mmq PRIVATE STRATA_MMQ_KQUANTS=1)
    endif()
    if(STRATA_ORCA_Q4KS_MMQ)
      target_compile_definitions(strata_mmq PRIVATE STRATA_ORCA_Q4KS_MMQ=1)
    endif()
    target_link_libraries(strata_mmq PUBLIC ggml-base hip::host roc::hipblas)
    target_link_libraries(strata_prefill PUBLIC strata_mmq)
    target_compile_definitions(strata_prefill PUBLIC STRATA_PREFILL_MMQ=1 STRATA_PREFILL_FUSED=1)
    # Aurora: the gfx11 fused experts against the FP64 reference and MMQ (tests/cuda's test; exit 77 off gfx11)
    add_executable(hip_prefill_fused_moe tests/cuda/prefill_fused_moe_test.cpp)
    target_include_directories(hip_prefill_fused_moe PRIVATE "${_strata_gg}/include")
    target_link_libraries(hip_prefill_fused_moe PRIVATE strata_mmq ggml-base strata_hip_runtime)
    add_test(NAME hip_prefill_fused_moe COMMAND hip_prefill_fused_moe)
    set_tests_properties(hip_prefill_fused_moe PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 300)
    list(APPEND _strata_hip_tests hip_prefill_fused_moe)
    add_executable(hip_prefill_fused_iq tests/cuda/prefill_fused_iq_test.cpp)
    target_include_directories(hip_prefill_fused_iq PRIVATE "${_strata_gg}/include")
    target_link_libraries(hip_prefill_fused_iq PRIVATE strata_mmq ggml-base strata_hip_runtime)
    add_test(NAME hip_prefill_fused_iq COMMAND hip_prefill_fused_iq)
    set_tests_properties(hip_prefill_fused_iq PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 600)
    list(APPEND _strata_hip_tests hip_prefill_fused_iq)

    # Numerical diagnostic: synthetic layouts by default, optional first-expert IQ3_XXS/IQ4_NL check with a pack.
    add_executable(hip_prefill_mmq_parity tests/hip/prefill_mmq_parity.cpp)
    target_include_directories(hip_prefill_mmq_parity PRIVATE "${_strata_gg}/include")
    target_link_libraries(hip_prefill_mmq_parity PRIVATE strata_prefill)
    add_test(NAME hip_prefill_mmq_parity COMMAND hip_prefill_mmq_parity)
    set_tests_properties(hip_prefill_mmq_parity PROPERTIES TIMEOUT 120)
    list(APPEND _strata_hip_tests hip_prefill_mmq_parity)
  endif()
  if(STRATA_ENABLE_HIP)
    # Optional real-model diagnostic: pass the GGUF shard containing the native PLE key.
    add_executable(hip_ple_iq4 tests/hip/ple_iq4.cpp)
    target_link_libraries(hip_ple_iq4 PRIVATE strata_kernels strata_hip_runtime ggml-base)
  endif()
  # the NVFP4 AVX-512 and AVX2 rows vs ggml-cpu's vec_dot, and their speed (CPU only; optional arg: a pack's experts.bin)
  add_executable(nvfp4_avx512_parity src/kernels/cpu/nvfp4_avx512_parity.cpp)
  target_link_libraries(nvfp4_avx512_parity PRIVATE strata_kernels_cpu)
  add_test(NAME nvfp4_cpu_parity COMMAND nvfp4_avx512_parity)   # random blocks; the AVX2 rows bit-equal to ggml's
  set_tests_properties(nvfp4_cpu_parity PROPERTIES TIMEOUT 120)
  if(STRATA_ENABLE_CUDA)   # the NVFP4 decode kernels vs ggml's reference
    add_executable(nvfp4_expert_gpu_parity src/kernels/nvfp4_expert_gpu_parity.cpp)
    target_link_libraries(nvfp4_expert_gpu_parity PRIVATE strata_kernels ggml-base CUDA::cudart)
    # the decode experts' weight bandwidth on random blobs of a format pair, and an output hash (bit-for-bit checks)
    add_executable(nvfp4_decode_bench src/kernels/nvfp4_decode_bench.cpp)
    target_link_libraries(nvfp4_decode_bench PRIVATE strata_kernels CUDA::cudart)
  endif()
  if(STRATA_ENABLE_CUDA OR STRATA_ENABLE_HIP)
    add_executable(native_expert_parity src/kernels/native_expert_parity.cpp)
    target_include_directories(native_expert_parity PRIVATE ${_strata_gpu_include_directories})
    target_link_libraries(native_expert_parity PRIVATE strata_kernels strata_kernels_cpu strata_artifact ${_strata_gpu_runtime_target})
    add_executable(native_expert_bench src/kernels/native_expert_bench.cpp)
    target_include_directories(native_expert_bench PRIVATE ${_strata_gpu_include_directories})
    target_link_libraries(native_expert_bench PRIVATE strata_kernels strata_kernels_cpu strata_artifact ${_strata_gpu_runtime_target})
    # MODEL-UPDATE: the dense Q8_0 GEMVs at the engine's shapes against the memory bandwidth (a benchmark, no test)
    add_executable(q8_gemv_bench src/kernels/q8_gemv_bench.cpp)
    target_include_directories(q8_gemv_bench PRIVATE ${_strata_gpu_include_directories})
    target_link_libraries(q8_gemv_bench PRIVATE strata_kernels ${_strata_gpu_runtime_target})
    if(STRATA_BUILD_TESTS AND STRATA_ENABLE_CUDA)
      # Unsloth UD-Q4_K_XL's three expert pairs (and Q8_0 alone, its embedding) on synthetic ggml-quantized weights
      # at 2560/640: GPU and CPU experts against the float reference, GPU dequantizers against to_float.
      # q4_K/q5_0 and q5_0/q8_0: a plain llama-quantize "_M" tier GGUF (not Unsloth's dynamic method) commonly
      # puts Q5_0 on down-projections (sometimes gate/up too) where this model's native_experts.txt had no GPU
      # kernel for it before; q5_0/q5_0 (both roles Q5_0) is left out of this list since stacking the same
      # lower-precision format on both roles pushes the GPU path over this test's 3% threshold (quantization
      # noise, not a kernel bug -- see the PR).
      # q4_0/q4_0 and q4_0/q4_1: a plain llama-quantize Q4_0 file (bartowski's Qwen3.8-Flash-Next-Q4_0: gate/up Q4_0,
      # down Q4_0 on 42 layers and Q4_1 on 6)
      # q6_K/q8_0: Unsloth UD-Q6_K_XL gate/up Q6_K with Q8_0 down (docs/UNSLOTH_Q6.md). Layer 2 is q8_0/q8_0, already listed.
      set(_strata_parity_pairs q4_K/q5_1 q4_K/q8_0 q5_K/q8_0 q8_0/q8_0 q4_K/q5_0 q5_0/q8_0 q4_0/q4_0 q4_0/q4_1)
      if(STRATA_Q6K_EXPERTS)
        list(APPEND _strata_parity_pairs q6_K/q8_0)
      else()   # a pair without GPU kernels (Q6_K gate/up) is refused by native_expert_supported, not run
        add_test(NAME native_expert_parity_refuses_q6_K COMMAND native_expert_parity --synthetic q6_K/q8_0)
        set_tests_properties(native_expert_parity_refuses_q6_K PROPERTIES WILL_FAIL TRUE)
      endif()
      foreach(_pair ${_strata_parity_pairs})
        string(REPLACE "/" "_" _name "${_pair}")
        add_test(NAME native_expert_parity_${_name} COMMAND native_expert_parity --synthetic ${_pair})
        set_tests_properties(native_expert_parity_${_name} PROPERTIES TIMEOUT 120)
      endforeach()
      add_test(NAME native_expert_parity_q5_1_min COMMAND native_expert_parity --q5_1-min)
      # #290: the BF16 token embedding (--embd-gguf) gathered and dequantized bit for bit
      add_test(NAME native_expert_parity_bf16_embd COMMAND native_expert_parity --bf16-embd)
    endif()
  endif()
  # #296's real-weight checks (OrcaRouter Q4_K_S files; docs/ORCA_Q4_K_S.md) - test programs, not part of a release
  if(STRATA_ENABLE_CUDA AND STRATA_BUILD_TESTS)
    add_executable(ple_q5_parity src/kernels/ple_q5_parity.cpp)
    target_link_libraries(ple_q5_parity PRIVATE strata_kernels strata_artifact ggml-base ${_strata_gpu_runtime_target})
    add_executable(q5_projection_parity src/kernels/q5_projection_parity.cpp)
    target_include_directories(q5_projection_parity PRIVATE ${_strata_gpu_include_directories})
    target_link_libraries(q5_projection_parity PRIVATE strata_kernels strata_artifact ggml-base ${_strata_gpu_runtime_target})
    # #291: the FP8 PLE table (tools/ple_fp8_pack.py) through PleTable vs the checkpoint's own values
    add_executable(ple_fp8_parity src/kernels/ple_fp8_parity.cpp)
    target_link_libraries(ple_fp8_parity PRIVATE strata_kernels ${_strata_gpu_runtime_target})
=======
    target_link_libraries(strata_mmq PUBLIC ggml-base CUDA::cudart)
    target_link_libraries(strata_prefill PUBLIC strata_mmq)
    add_executable(native_expert_parity src/kernels/native_expert_parity.cpp)
    target_include_directories(native_expert_parity PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
    target_link_libraries(native_expert_parity PRIVATE strata_kernels strata_kernels_cpu strata_artifact CUDA::cudart)
    add_test(NAME native_expert_parity_k_quants COMMAND native_expert_parity --synthetic q4_K/q5_1 q5_K/q8_0)
    # --expert-math exact's expf: the CPU pool's (strata_kernels_cpu) against the GPU's, over every float
    add_executable(swiglu_expf_parity src/kernels/swiglu_expf_parity.cu)
    target_link_libraries(swiglu_expf_parity PRIVATE strata_kernels_cpu CUDA::cudart)
    add_test(NAME swiglu_expf_parity COMMAND swiglu_expf_parity)
    # The grouped native expert kernels against the ones they replaced, kept in the test: bitwise (synthetic experts).
    add_executable(expert_grouped_parity src/kernels/expert_grouped_parity.cu)
    target_include_directories(expert_grouped_parity PRIVATE third_party/ggml src/kernels/cuda)
    target_link_libraries(expert_grouped_parity PRIVATE strata_kernels strata_kernels_cpu CUDA::cudart)
    add_test(NAME expert_grouped_parity COMMAND expert_grouped_parity)
>>>>>>> their fork (github.com-eddoursul-Strata-custom)
  endif()
endif()

if(STRATA_ENABLE_CUDA AND STRATA_BUILD_TESTS AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/bench/micro/hit_cpu_order_parity.cu")
  add_executable(hit_cpu_order_parity bench/micro/hit_cpu_order_parity.cu)
  target_link_libraries(hit_cpu_order_parity PRIVATE strata_kernels strata_kernels_cpu CUDA::cudart)
endif()

if(STRATA_BUILD_TESTS)
  # Plan v0.3 P6: one expert applied to several tokens, bitwise vs the single-token kernel (synthetic, CPU only).
  # plan v0.3 P5: the device dequantizers vs the artifact's CPU dequantizers on the real shards (not a ctest: needs
  # the 38 GB model; run it by hand or from tools/gpu_session.py)
  add_executable(dequant_bf16_test src/kernels/dequant_bf16_test.cpp)
  target_link_libraries(dequant_bf16_test PRIVATE strata_kernels strata_artifact CUDA::cudart)

  add_executable(expert_multi_test src/kernels/cpu/expert_multi_test.cpp)
  target_link_libraries(expert_multi_test PRIVATE strata_kernels_cpu)
  add_test(NAME expert_multi_test COMMAND expert_multi_test)

  add_executable(expert_parity src/kernels/cpu/expert_parity.cpp)
  target_link_libraries(expert_parity PRIVATE strata_kernels_cpu)
  # The pack is a REQUIRED input, not an optional one: this test compares against a real expert blob, and a
  # missing file would otherwise be reported as a pass by a test that skipped everything.
  add_test(NAME expert_parity COMMAND expert_parity --selftest
           WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})

  add_test(NAME pool_test COMMAND pool_test --selftest WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
  add_test(NAME pool_stress COMMAND pool_test --stress)
endif()

if(STRATA_BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()
