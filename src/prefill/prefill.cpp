// src/prefill/prefill.cpp - see include/strata/prefill/prefill.hpp.
#include "strata/prefill/prefill.hpp"

#include "strata/core/layout.hpp"
#include "strata/kernels/dequant_bf16.hpp"
#include "strata/kernels/kv_q4.hpp"
#include "strata/kernels/native_qsa_indexer.hpp"
#include "strata/kernels/native_ple_postops.hpp"
#include "strata/kernels/ngram.hpp"
#include "strata/kernels/qsa.hpp"
<<<<<<< this fork
#include "strata/kernels/kv_stream.hpp"
#include "strata/kernels/cvec.hpp"
#include "strata/kernels/kv_q4.hpp"
#include "strata/core/layer.hpp"
#include "strata/core/native_dense.hpp"
#include "strata/core/native_head.hpp"
#include "strata/kernels/verify_kernels.hpp"
=======
>>>>>>> their fork (github.com-eddoursul-Strata-custom)
#include "strata/kernels/qsa_decode_attn.hpp"
#include "strata/kernels/qsa_select.hpp"
#include "strata/kernels/verify_kernels.hpp"
#include "strata/prefill/gemm.hpp"
#include "strata/prefill/kernels.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <thread>
#include <vector>

<<<<<<< this fork
#ifndef STRATA_PREFILL_MMQ
// A build without the llama.cpp sources (no STRATA_NATIVE_EXPERTS): no MMQ, the FP16 expert path everywhere.
namespace strata::prefill::mmq {
bool built() { return false; }
bool supported(int) { return false; }
bool fits(int, int64_t) { return false; }
size_t matrix_bytes(int, int64_t, int64_t) { return 0; }
size_t q8_bytes(int64_t, int64_t) { return 0; }
void quantize(const float*, const int32_t*, void*, int, int64_t, int64_t, int64_t, void*, float*) {}
bool fp4_activations(int) { return false; }
Context::Context() {}
Context::~Context() {}
void Context::run(const Product&, void*) {}
void gather_native(const void*, const void*, size_t, const void*, size_t, void*, void*, void*, const void*, void*, size_t) {}
bool gather_native_group(const GatherGroup&, size_t, size_t, size_t, size_t, void*, size_t, void*, size_t, void*, size_t,
                         void*, size_t) { return false; }
void gather_strata_q2(const uint8_t*, void*, void*, void*) {}
void swiglu(const float*, float*, int64_t, int64_t, bool, void*) {}
void swiglu_scaled(const float*, float*, int64_t, int64_t, const int32_t*, int, const float*, int64_t, void*) {}
void down_row_scales(float*, const int32_t*, int, const float*, int64_t, void*) {}
void iota(int32_t*, int64_t, void*) {}
}  // namespace strata::prefill::mmq
#endif
#ifndef STRATA_PREFILL_FUSED
// #136: the fused int8 experts are CUDA-only (HIP and builds without MMQ keep the MMQ / FP16 paths)
namespace strata::prefill::fused {
bool built() { return false; }
bool available() { return false; }
bool enabled() { return false; }
bool requested() { return false; }
size_t act_bytes(int64_t, int64_t) { return 0; }
size_t group_bytes(int64_t, int) { return 0; }
void quantize_act(const float*, int64_t, int64_t, void*, void*) {}
void group(const int32_t*, int64_t, int, int, void*, int32_t*, int32_t*, void*) {}
void experts(const Batch&, int, int64_t, const void*, const void*, const int32_t*, void*, float*, void*) {}
bool native_supported(int, int) { return false; }
void quantize_act_native(const float*, int64_t, int64_t, void*, void*) {}
void experts_native(const Batch&, const NativeGeom&, int, int64_t, const void*, const void*, const int32_t*, void*,
                    float*, void*) {}
}  // namespace strata::prefill::fused
#endif

#if defined(STRATA_GPU_METAL)
// The Metal backend reads a prompt through its decode window (metal/src/engine/window.cpp): init keeps the chunk, run
// hands over, draft_kv declines (the MTP hook reads the window's rows itself).
namespace strata::metal_engine {
bool window_ensure(const strata::core::ModelGeometry& g, strata::core::SessionState& ss, int64_t max_cells, std::string& err);
bool prompt_run(const int64_t* tokens, int64_t n, int64_t pos0, const float* const* embd_rows,
                const std::function<bool(const float*, int64_t, int64_t, std::string&)>& on_chunk, std::string& err);
}  // namespace strata::metal_engine
#endif

=======
>>>>>>> their fork (github.com-eddoursul-Strata-custom)
namespace strata::prefill {
namespace {

using Clock = std::chrono::steady_clock;
constexpr float EPS = 1e-6f;
constexpr int64_t N = 2560, HC = 4, D = N * HC, LR = 320, K = 10, NE = 512;
constexpr int64_t C = 10240, ZV = 6144, HV = 48;
constexpr int64_t SUB = 2048;                 // tokens of a sub-chunk: everything but the routed experts
constexpr int64_t SEL_BATCH = 256;            // queries per block-score launch
// A layer's quantized projections, dequantized to FP16 once for all its sub-chunks: the PLE key, or the mixer's
// (GDN: qkv, gate, out; QSA: q, k, v, out) and the shared expert's
constexpr int64_t W16 = std::max(D * N, std::max(C + ZV + ZV, 12288 + 512 + 512 + ZV) * N + 3 * 640 * N);
constexpr size_t GEMM_WS = 32u << 20;         // cuBLAS workspace
constexpr int GGML_Q8_0 = 8;

double ms_since(Clock::time_point t) { return std::chrono::duration<double, std::milli>(Clock::now() - t).count(); }

// Bump allocation from a region; without one it only counts (bytes_needed).
struct Alloc {
    uint8_t* base = nullptr;
    uint64_t cap = 0, used = 0;
<<<<<<< this fork
    uint64_t failed_bytes = 0;          ///< the take that could not be allocated; the failure message names it
    uint64_t granule = 0;               ///< count_only: each take rounds up to this (cudaMalloc's 2 MiB pages: owned buffers)
    bool count_only = false;
    bool spill = false, spilled = false;
    std::vector<void*>* owned = nullptr;
    template <typename T> T* take(size_t n, bool& ok) {
        uint64_t bytes = ((uint64_t) n * sizeof(T) + 256 + 255) & ~255ull;
        if (count_only) {
            if (granule > 0) bytes = (bytes + granule - 1) / granule * granule;
            used += bytes;
            return nullptr;
        }
        if (base != nullptr && !spilled && used + bytes <= cap) {
            T* p = (T*) (base + used);
            used += bytes;
            return p;
        }
        if (base != nullptr && !spill) { ok = false; failed_bytes = bytes; return nullptr; }
        void* p = nullptr;
        if (cudaMalloc(&p, bytes) != cudaSuccess) { ok = false; failed_bytes = bytes; return nullptr; }
        spilled = true;
        owned->push_back(p);
=======
    bool ok = true;
    template <typename T> T* take(size_t n) {
        const uint64_t bytes = ((uint64_t) n * sizeof(T) + 256 + 255) & ~255ull;
        T* p = base != nullptr ? (T*) (base + used) : nullptr;
>>>>>>> their fork (github.com-eddoursul-Strata-custom)
        used += bytes;
        if (base != nullptr && used > cap) ok = false;
        return p;
    }
};

// The device buffers of one binding.
struct Bufs {
    uint16_t* w16 = nullptr;
    void* gemm_ws = nullptr;
    // the chunk's: residual stream, the MoE's input (FP16) and output, the FFN half's injection, routing, steps, PLE
    // rows, the second GPU's sums (FP16)
    float *R = nullptr, *bo_moe = nullptr, *inj_f = nullptr, *w = nullptr, *ple_emb = nullptr;
    uint16_t *mixed_h = nullptr, *sums = nullptr;
    int32_t *ids = nullptr, *steps = nullptr;
    // a sub-chunk's: the mixer's BF16 input, its output, the mixer half's injection, a read's row scales
    uint16_t* mixed_bf = nullptr;
    float *bo = nullptr, *inj_a = nullptr, *rs = nullptr;
    // the union: what one step needs at a time
    float* emb = nullptr;
    float *xn = nullptr, *lo = nullptr, *gated = nullptr;
    uint16_t *xn16 = nullptr, *lo16 = nullptr;
    float *ple_kp = nullptr, *ple_key = nullptr, *ple_q = nullptr, *ple_gated = nullptr;
    float *qkv = nullptr, *z = nullptr, *ab = nullptr, *gate = nullptr, *beta = nullptr, *hbuf = nullptr, *y = nullptr;
    uint16_t* y_h = nullptr;
    float *Kc = nullptr, *Vc = nullptr, *Qf = nullptr, *q = nullptr, *idx_raw = nullptr, *q_idx = nullptr,
          *attn = nullptr, *sel_scores = nullptr;
    uint16_t* attn_h = nullptr;
    int32_t* sel_ids = nullptr;
    float *logits = nullptr, *sgate = nullptr, *sup = nullptr, *sg = nullptr;
    uint16_t* sh_h = nullptr;
    uint8_t* local = nullptr;     // this GPU's expert runner
    uint64_t local_bytes = 0;
};

// Chunks of T tokens: the chunk-long buffers, the sub-chunk ones, then the union of the steps a layer takes one after
// the other on the stream.  (Staging copies run ahead on a copy stream only while the routed experts run, after the
// host has waited for the stream, and the stream waits for them before the next step.)
Bufs layout(Alloc& a, int64_t T, int64_t cap, int64_t max_blocks, bool offload) {
    const size_t t = (size_t) T, p = (size_t) std::min(T, SUB);
    Bufs b;
    b.w16 = a.take<uint16_t>((size_t) W16);
    b.gemm_ws = a.take<uint8_t>(GEMM_WS);
    b.R = a.take<float>(t * D);
    b.mixed_h = a.take<uint16_t>(t * N);
    b.bo_moe = a.take<float>(t * N);
    b.inj_f = a.take<float>(t * HC);
    b.ids = a.take<int32_t>(t * K);
    b.w = a.take<float>(t * K);
    b.steps = a.take<int32_t>(t * strata::kernels::kStepCount);
    b.ple_emb = a.take<float>(t * N);
    b.sums = offload ? a.take<uint16_t>(t * N) : nullptr;   // read by the next layer's sub-chunks
    b.mixed_bf = a.take<uint16_t>(p * N);
    b.bo = a.take<float>(p * N);
    b.inj_a = a.take<float>(p * HC);
    b.rs = a.take<float>(p * HC);
    const uint64_t u0 = a.used;
    uint64_t u1 = u0;
    auto next = [&] { u1 = std::max(u1, a.used); a.used = u0; };
    b.emb = a.take<float>(p * N);
    next();   // the hyper-connection read
    b.xn = a.take<float>(p * D); b.xn16 = a.take<uint16_t>(p * D); b.lo = a.take<float>(p * LR);
    b.lo16 = a.take<uint16_t>(p * LR); b.gated = a.take<float>(p * D);
    next();   // the PLE block
    b.ple_kp = a.take<float>(p * D); b.ple_key = a.take<float>(p * D);
    b.ple_q = a.take<float>(p * D); b.ple_gated = a.take<float>(p * D);
    next();   // GDN
    b.qkv = a.take<float>(p * C); b.z = a.take<float>(p * ZV); b.ab = a.take<float>(p * 2 * HV);
    b.gate = a.take<float>(p * HV); b.beta = a.take<float>(p * HV); b.hbuf = a.take<float>(p * C);
    b.y = a.take<float>(p * ZV); b.y_h = a.take<uint16_t>(p * ZV);
    next();   // QSA
    b.Kc = a.take<float>(p * 512); b.Vc = a.take<float>(p * 512); b.Qf = a.take<float>(p * 12288);
    b.q = a.take<float>(p * ZV); b.idx_raw = a.take<float>(p * 128); b.q_idx = a.take<float>(p * 512);
    b.attn = a.take<float>(p * ZV); b.attn_h = a.take<uint16_t>(p * ZV); b.sel_ids = a.take<int32_t>(p * (size_t) cap);
    b.sel_scores = a.take<float>((size_t) SEL_BATCH * (size_t) max_blocks);
    next();   // the router and the shared expert
    b.logits = a.take<float>(p * NE); b.sgate = a.take<float>(p * 640); b.sup = a.take<float>(p * 640);
    b.sh_h = a.take<uint16_t>(p * 640); b.sg = a.take<float>(p);
    next();   // the routed experts
    b.local_bytes = ExpertRunner::bytes_needed(T, NE, false, !offload);
    b.local = a.take<uint8_t>(b.local_bytes);
    next();
    a.used = u1;
    return b;
}

// K and V of T cells from pos0 ([T, 2, 256] each) into a state's pools, in its format (the decode append's arithmetic)
void append_kv(const core::QsaState& st, const float* Kr, const float* Vr, int64_t T, int64_t pos0,
               const strata::kernels::QsaShapes& s, cudaStream_t cs) {
    using strata::kernels::kv_append_q4_rows;
    switch (st.kv) {
    case core::KvFormat::Q4:
        kv_append_q4_rows(st.k_q4, st.k_q4s, st.v_q4, st.v_q4s, st.page_table, pos0, T, Kr, Vr, 512, s, cs);
        break;
    case core::KvFormat::K8V4:
        kv_append(Kr, Vr, T, pos0, st.page_table, s.page_size, nullptr, nullptr, st.k_q, nullptr, st.k_scale, nullptr, cs,
                  1);
        kv_append_q4_rows(nullptr, nullptr, st.v_q4, st.v_q4s, st.page_table, pos0, T, Kr, Vr, 512, s, cs);
        break;
    default:
        kv_append(Kr, Vr, T, pos0, st.page_table, s.page_size, st.kv_int8 ? nullptr : st.k_pool,
                  st.kv_int8 ? nullptr : st.v_pool, st.k_q, st.v_q, st.k_scale, st.v_scale, cs);
    }
}

strata::kernels::QsaShapes shapes_of(const core::ModelGeometry& g) {
    strata::kernels::QsaShapes s = strata::kernels::qsa_real_shapes();
    s.n_head = g.n_head; s.n_head_kv = g.n_head_kv; s.head_dim = g.head_dim; s.idx_n_head = g.idx_q_heads;
    s.idx_dim = g.idx_key_dim;
    return s;
}

}  // namespace

struct Prefill::Impl {
    const core::WeightTable* wt = nullptr;
    const core::ModelGeometry* g = nullptr;
    core::SessionState* ss = nullptr;
    const core::ExpertCache* cache = nullptr;
    const int32_t* host_res = nullptr;
<<<<<<< this fork
    int64_t T = 0, T_max = 0;
    bool borrowed = false, borrow_spill = false;
    cudaStream_t cs = nullptr, copy = nullptr;
    Gemm gemm;
    std::vector<void*> owned;
    // chunk buffers
    float *emb = nullptr, *R = nullptr, *xn = nullptr, *lo = nullptr, *gated = nullptr, *inj = nullptr;
    float* grs = nullptr;                    // F-1: the hyper-connection read's row scales (T x 4)
    uint16_t *xn16 = nullptr, *lo16 = nullptr;
    float* mixed = nullptr;
    uint16_t *mixed_bf = nullptr, *mixed_h = nullptr;
    bool f16_io = false;   // HIP, STRATA_HIP_PROMPT_F16=1: this stage's 16-bit GEMMs run FP16 in and out (set in init)
    uint16_t *xn16_lo = nullptr, *lo16_lo = nullptr, *mixed_bf_lo = nullptr;   // bf16x2(): the BF16 GEMMs' low parts
    float* bo = nullptr;
    // GDN
    float *qkv = nullptr, *z = nullptr, *ab = nullptr, *gate = nullptr, *beta = nullptr, *hbuf = nullptr, *y = nullptr;
    uint16_t* y_h = nullptr;
    // QSA
    float *Kc = nullptr, *Vc = nullptr, *Qf = nullptr, *q = nullptr, *idx_raw = nullptr, *q_idx = nullptr, *attn = nullptr;
    uint16_t* attn_h = nullptr;
    int32_t* steps_dev = nullptr;
    std::vector<int32_t> steps_host;
    PinnedUpload steps_up;
    int32_t* sel_ids = nullptr;
    float* sel_scores = nullptr;          // [sel_batch, max_blocks]
    int64_t sel_batch = 256, max_blocks = 0;
    float* attn_scratch = nullptr;
    int64_t attn_batch = 32, cap = 0;
    // MoE
    float *logits = nullptr, *w = nullptr, *GU = nullptr, *Dm = nullptr, *sgate = nullptr, *sup = nullptr,
          *shared = nullptr, *sg = nullptr;
    int32_t *ids = nullptr, *slot_dev = nullptr, *src_dev = nullptr;
    uint16_t *Xs = nullptr, *Hh = nullptr, *sh_h = nullptr;
    // step 2b (MMQ): the activations quantized per layer, H in FP32 and its group's quantized rows, the identity
    // row map, the group bounds, the group buffers of gathered experts
    void *Xq = nullptr, *Hq = nullptr;
    float* H = nullptr;
    int32_t *ids_identity = nullptr, *bounds_dev = nullptr;
    uint8_t *grp_gu = nullptr, *grp_d = nullptr;
    float *Xscale = nullptr, *Hscale = nullptr, *grp_tail = nullptr;   // NVFP4: activation scales, expert tails
    float* row_sd = nullptr;          // NVFP4 in MMQ: each Dm row's s_down, applied by the combine as it reads the row
    std::vector<int32_t> bounds_host;
    std::unique_ptr<mmq::Context> mmq_ctx;
    std::vector<int32_t> ids_host, slot_host, src_host, cnt, off;
    // set_cpu_pool: a small chunk's CPU experts. The layer's MoE input (T x N, from `mixed`), the rows the pool writes
    // (Dm's tail, in Dm's row order), ones for row_sd (the CPU's rows carry s_down already), all pinned; and the
    // pool's per-token activations and jobs.
    float *cpu_x = nullptr, *cpu_rows = nullptr, *cpu_ones = nullptr;
    size_t cpu_x_n = 0, cpu_rows_n = 0, cpu_ones_n = 0;
    bool cpu_dead = false;   // a pinned buffer of the share could not be had (host memory is short): the GPU does it all
    std::vector<uint8_t> cpu_nact;
    std::vector<kernels::cpu::ActQ> cpu_actq;   // a Q2_0 layer's activations (the pool's Q2_0 kernels read ActQ)
    std::vector<kernels::cpu::ExpertJobMulti> cpu_jobs;
    // the measured share: running means of the CPU's ms per expert and the GPU's per streamed expert, and the share
    // they balance at (cpu_share_env)
    double cpu_c_ms = 0, cpu_g_ms = 0, cpu_share_now = 0.5;
    cudaEvent_t cpu_ev[2] = {};   // the GPU's expert work of the last CPU-sharing layer, read once the stream is done
    bool cpu_pend = false;
    double pend_cpu_ms = 0;
    int64_t pend_n_cpu = 0, pend_n_gpu = 0;
    // ... and whether sharing pays at all (auto): an eligible layer's wall time per non-resident expert, from before
    // its routing sync to its combine (CUDA events, two sets: a layer's start is recorded before the last one's is
    // read); two adjacent layers of the two arms give a ratio (with / without), and the median of the last
    // CPU_RATIOS decides.  Layers differ up to 8x in that cost, the same in every run, which is why adjacent layers
    // are compared rather than a running mean per arm.
    static constexpr int CPU_RATIOS = 5;
    int64_t cpu_layers = 0;   // the eligible layers so far (the arms' schedule)
    int64_t cpu_first_l = -1;   // the model's first eligible layer: 2x the others' cost in both arms, never read
    cudaEvent_t cpu_wall[4] = {};
    int pend_wall = 0, pend_set = 0, pend_l = 0;   // pend_wall 1: the last eligible layer ran without the share, 2: with
    int64_t pend_wall_n = 0;
    int last_arm = 0, last_l = 0;
    double last_w = 0;
    double cpu_ratio[CPU_RATIOS] = {};
    int cpu_nratio = 0;
    bool cpu_gate = false;   // the median ratio is below 1: the layers that share are the cheaper ones
    // The grouping tables in mapped pinned memory, [ids | slot | src] of T_max * K each, then the MMQ bounds: kernels
    // read and write them in place.  A cudaMemcpyAsync of them queues behind the expert blobs the copy stream already
    // holds (up to `ring` of them, ~70 us each), and the GPU idles meanwhile - measured 4.2 s of a 128K prompt's
    // 70 s at the default ring, 0.7 s with a 16-slot one.  STRATA_GROUP_COPY=1: the copies (the A/B arm).
    int32_t* grp_host = nullptr;
    int32_t* grp_dev = nullptr;          // its device alias
    size_t grp_n = 0, grp_tk = 0;        // int32s allocated; T_max * K (the offset of slot, and of src past it)
    uint16_t* dq_gu[DQ] = {};
    uint16_t* dq_d[DQ] = {};
    uint8_t* stage_dev[RING_MAX] = {};
    int ring = STAGE;                        // the slots of this layout's ring (ring_slots)
    // this layout's GU, H and Xq are the fused path's (moe_bufs' `fused`), so a layer that runs MMQ or the FP16 path may
    // only write stream_all_min() - 1 of its rows into them; fused_layout() keeps the two in step (#583, #954)
    bool fused_bufs = false;
    std::unique_ptr<Stager> stager;          // the unpinned experts' host copies (step 4)
    cudaEvent_t copied[RING_MAX] = {}, used[RING_MAX] = {};
    bool stage_live[RING_MAX] = {};
    // the event that releases each ring slot: its own `used`, or - when an MMQ group is gathered in one launch - the
    // `used` of the last slot gathered with it, recorded once for all of them (a later record only waits longer)
    int used_of[RING_MAX] = {};
    // PLE
    float* ple_emb = nullptr;
    std::vector<float> ple_pageable[2];      // the fallback when no more RAM can be pinned
    float* ple_emb_host[2] = {};             // pinned, double-buffered: the next chunk's rows are read while this
    cudaEvent_t ple_copied[2] = {};          // one runs; the event marks that buffer's upload done
=======
    ExpertRunner* offload = nullptr;
    ExpertRunner local;
    int64_t max_chunk = 0, T = 0;   // T: the chunk the buffers are bound for
    strata::kernels::QsaShapes s;
    int64_t cap = 0, max_blocks = 0;
    cudaStream_t cs = nullptr;
    // the second GPU's share, a sub-chunk at a time: the input goes out on `xfer`, the sums come back on `xsum`
    cudaStream_t xfer = nullptr, xsum = nullptr;
    cudaEvent_t ev_mixed = nullptr;
    std::vector<cudaEvent_t> ev_in, ev_piece;   // a sub-chunk's input is in host memory, its sums on this GPU
    Gemm gemm;
    void* owned = nullptr;   // the buffers' own allocation (no region)
    Bufs b;
    std::vector<std::pair<const void*, const uint16_t*>> w16_have;   // this layer's dequantized weights
    int64_t w16_used = 0;
    int32_t* ids_host = nullptr;   // pinned [max_chunk, K]
    float* w_host = nullptr;
    std::vector<int32_t> steps_host, cnt, off, entry_of;
    std::vector<int32_t> ex1, off1, src1, ex2, off2, src2;   // this GPU's experts, the second GPU's
    std::vector<float> w1, w2;
    // a chunk's PLE rows, read on a thread while the chunk before runs: two pinned buffers, their row indices
    float* ple_host[2] = {};
>>>>>>> their fork (github.com-eddoursul-Strata-custom)
    std::vector<uint32_t> ple_rows[2];
};

Prefill::Prefill() : impl_(new Impl) {}
Prefill::~Prefill() {
    if (!impl_) return;
<<<<<<< this fork
    if (impl_->cs) cudaStreamSynchronize(impl_->cs);
    if (impl_->copy) cudaStreamSynchronize(impl_->copy);
    if (impl_->kv_copy) cudaStreamSynchronize(impl_->kv_copy);
    for (int i = 0; i < RING_MAX; ++i) {
        if (impl_->copied[i]) cudaEventDestroy(impl_->copied[i]);
        if (impl_->used[i]) cudaEventDestroy(impl_->used[i]);
    }
    for (int b = 0; b < 2; ++b) {
        if (impl_->hand[b]) cudaFreeHost(impl_->hand[b]);
        if (impl_->ple_copied[b]) cudaEventDestroy(impl_->ple_copied[b]);
        if (impl_->ple_emb_host[b] && impl_->ple_pageable[b].empty()) cudaFreeHost(impl_->ple_emb_host[b]);
    }
    if (impl_->copy) cudaStreamDestroy(impl_->copy);
    if (impl_->kv_copy) cudaStreamDestroy(impl_->kv_copy);
    if (impl_->kv_released) cudaEventDestroy(impl_->kv_released);
    if (impl_->kv_ready) cudaEventDestroy(impl_->kv_ready);
    if (impl_->grp_host) cudaFreeHost(impl_->grp_host);
    impl_->steps_up.release();
    impl_->tok_up.release();
    impl_->up_ring.release();
    for (float* p : {impl_->cpu_x, impl_->cpu_rows, impl_->cpu_ones})
        if (p) cudaFreeHost(p);
    for (cudaEvent_t e : impl_->cpu_ev)
        if (e) cudaEventDestroy(e);
    for (cudaEvent_t e : impl_->cpu_wall)
        if (e) cudaEventDestroy(e);
    for (void* p : impl_->owned) cudaFree(p);
}

namespace {
constexpr int64_t GEMM_SCRATCH = 32ll << 20;        // FP16 elements for the largest dequantized dense weight
constexpr size_t GEMM_WS = 32u << 20;               // cuBLAS workspace

// THE ATTENTION HALF AND THE MoE HALF SHARE THEIR BUFFERS.  A layer runs its attention (GDN or QSA), writes it back
// into the residual, and only then its MoE, so the three sets of scratch are never live at once: one region the size
// of the largest holds them all.  That is ~260 KB of the ~680 KB a prompt token cost - which is what lets a chunk
// grow (every expert is streamed once per chunk, so a bigger chunk streams fewer bytes per token).  The sizes are
// counted with the same `take` sequence `init` uses; a mismatch makes `init` fail with "do not fit", never overlap.
uint64_t gdn_set_bytes(size_t T) {
    Alloc a; a.count_only = true; bool ok = true;
    a.take<float>(T * C, ok); a.take<float>(T * ZV, ok); a.take<float>(T * 2 * HV, ok); a.take<float>(T * HV, ok);
    a.take<float>(T * HV, ok); a.take<float>(T * C, ok); a.take<float>(T * ZV, ok);
    a.take<uint16_t>(T * (ZV + (pf_pad() ? ZV_PAD : 0)), ok);
    return a.used;
}
uint64_t qsa_set_bytes(size_t T, int64_t cap, int64_t max_blocks, int64_t sel_batch, int64_t attn_batch,
                       const strata::kernels::QsaShapes& s) {
    Alloc a; a.count_only = true; bool ok = true;
    a.take<float>(T * 512, ok); a.take<float>(T * 512, ok); a.take<float>(T * 12288, ok); a.take<float>(T * ZV, ok);
    a.take<float>(T * 128, ok); a.take<float>(T * 512, ok); a.take<float>(T * ZV, ok);
    a.take<uint16_t>(T * (ZV + (pf_pad() ? ZV_PAD : 0)), ok);
    a.take<int32_t>(T * (size_t) cap, ok);
    a.take<float>((size_t) sel_batch * (size_t) max_blocks, ok);
    a.take<float>((size_t) attn_batch * strata::kernels::qsa_decode_attn_scratch_floats(cap, s), ok);
    return a.used;
}
// Step 2b: which layers' experts go through MMQ (both weight types covered; the Strata Q2_0 pack always - its blob
// is converted to GGUF Q2_0 blocks on the gather), whether any layer keeps the FP16 path (IQ1_M), and the largest
// gate/up and down matrices a group buffer slot holds.  STRATA_PREFILL_MMQ=0: the FP16 path everywhere (the A/B).
constexpr int MMQ_GROUP = 16;                  // experts per MMQ launch (the gather is per expert, as blobs arrive)
// MMQ reads up to one 256-value tile past a matrix's last row when the row length is not a multiple of it (the down
// product: 640 values).  Those bytes meet zero activations, which is harmless only if they decode to finite numbers -
// llama.cpp zero-pads after every tensor, and so does a group buffer: this many zeroed bytes follow its last expert.
constexpr size_t MMQ_TAIL = 4096;
struct MmqPlan {
    bool any = false, fallback = true;
    std::vector<char> layer;                   // per layer: MMQ
    std::vector<char> fo;                      // per layer: no MMQ here but the native fused kernels cover its formats
                                               // (gfx11: UD-Q4_K_XL's Q4_K / Q5_K and Q5_1 / Q8_0 experts, STRATA_PF_FUSED=1)
    size_t gu_max = 0, d_max = 0;
};
const MmqPlan& mmq_plan() {
    static const MmqPlan plan = [] {
        MmqPlan p;
        const auto& lay = strata::kernels::cpu::expert_layout();
        const char* env = std::getenv("STRATA_PREFILL_MMQ");
        const bool on = mmq::built() && (env == nullptr || std::atoi(env) != 0);
        const int64_t layers = lay.native ? (int64_t) lay.fmt.size() : lay.n_layers;
        p.layer.assign((size_t) std::max<int64_t>(layers, 0), 0);
        p.fo.assign(p.layer.size(), 0);
        p.fallback = !on || layers <= 0;
        for (int64_t l = 0; on && l < layers; ++l) {
            const int gt = lay.native ? lay.fmt[(size_t) l].gu_type : 42, dt = lay.native ? lay.fmt[(size_t) l].d_type : 42;
            // #420: a tile on every GPU for these shapes (gate+up: 1280 rows, down: N rows), else the FP16 path
            if (!mmq::fits(gt, 1280) || !mmq::fits(dt, N)) {
                p.fallback = true;
                // the fused path's buffers (Xq, H) exist for it; a chunk it does not take (a small one) keeps the FP16 path
                if (on && lay.native && fused::native_supported(gt, dt)) { p.fo[(size_t) l] = 1; p.any = true; }
                continue;
            }
            p.layer[(size_t) l] = 1;
            p.any = true;
            p.gu_max = std::max(p.gu_max, mmq::matrix_bytes(gt, 1280, N));
            p.d_max = std::max(p.d_max, mmq::matrix_bytes(dt, N, 640));
        }
        return p;
    }();
    return plan;
}
// #136 P3: a layout whose chunks run the fused experts (STRATA_PF_FUSED=1, the Q2_0 pack, a streamed chunk of
// stream_all_min() tokens or more).  Its GU, H and Xq hold only the fused path's grouping tables, int8 H and per-token
// int8 activations, and Hq nothing: ~100 KB a token less than MMQ's FP32 GU / H and per-slot q8_1 rows, which is what
// lets a bigger chunk or ring fit in the slots the prompt path borrows.  A last chunk below stream_all_min() still runs
// MMQ in the same buffers, so each keeps MMQ's size for stream_all_min() - 1 tokens.  `src`: the layout streams experts
// (Prefill::init got an ExpertSource; without one no chunk takes the streamed walk, so no chunk is fused).
bool fused_layout(size_t T, bool src) {
    return src && fused_ring() && mmq_plan().any && ring_slots(T) > STAGE && (int64_t) T >= stream_all_min();
}
// The MoE buffers MMQ and the fused path share: GU and H in floats, Xq and Hq in bytes.  Without `fused` (the
// default): MMQ's, for T tokens.
struct MoeBufs { size_t gu, h, xq, hq; };
MoeBufs moe_bufs(size_t T, int64_t n_expert, bool fused) {
    if (!fused) return {T * K * 1280, T * K * 640, mmq::q8_bytes((int64_t) (T * K), N), mmq::q8_bytes((int64_t) (T * K), 640)};
    const size_t ts = (size_t) std::min<int64_t>((int64_t) T, stream_all_min() - 1);   // MMQ's last small chunk
    return {std::max<size_t>(ts * K * 1280, (fused::group_bytes((int64_t) (T * K), (int) n_expert) + 3) / 4),
            std::max<size_t>(ts * K * 640, (fused::act_bytes((int64_t) (T * K), 640) + 3) / 4),
            std::max(mmq::q8_bytes((int64_t) (ts * K), N), fused::act_bytes((int64_t) T, N)),
            mmq::q8_bytes((int64_t) (ts * K), 640)};
}
uint64_t moe_set_bytes(size_t T, int64_t n_expert, bool fused) {
    const MmqPlan& mp = mmq_plan();
    const MoeBufs mb = moe_bufs(T, n_expert, fused);
    Alloc a; a.count_only = true; bool ok = true;
    a.take<float>(T * n_expert, ok); a.take<float>(T * K, ok); a.take<int32_t>(T * K, ok); a.take<int32_t>(T * K, ok);
    a.take<int32_t>(T * K, ok);
    if (mp.fallback) a.take<uint16_t>(T * K * N, ok);
    a.take<float>(mb.gu, ok);
    if (mp.fallback) a.take<uint16_t>(T * K * 640, ok);
    a.take<float>(T * K * N, ok); a.take<float>(T * 640, ok);
    a.take<float>(T * 640, ok); a.take<uint16_t>(T * 640, ok); a.take<float>(T * N, ok); a.take<float>(T, ok);
    if (mp.any) {
        a.take<uint8_t>(mb.xq, ok);
        a.take<float>(mb.h, ok);
        a.take<uint8_t>(mb.hq, ok);
        a.take<float>(T * K, ok); a.take<float>(T * K, ok); a.take<float>(MMQ_GROUP * 4, ok);
        a.take<float>(T * K, ok);
    }
    return a.used;
}
}

bool Prefill::init(const core::WeightTable& wt, const core::ModelGeometry& g, core::SessionState& ss,
                   core::ExpertSource* src, const core::ExpertCache* cache, const int32_t* host_res, int64_t chunk,
                   void* stream, std::string& err, void* borrow, uint64_t borrow_bytes, bool borrow_spill) {
    Impl& m = *impl_;
#if defined(STRATA_GPU_METAL)
    (void) wt; (void) src; (void) cache; (void) host_res; (void) stream; (void) borrow;
    (void) borrow_bytes;
    m.T = chunk > 0 ? chunk : 1;
    return strata::metal_engine::window_ensure(g, ss, ss.qsa_states[ss.qsa_primary()].max_cells, err);
#endif
    m.wt = &wt; m.g = &g; m.ss = &ss; m.src = src; m.cache = cache; m.host_res = host_res;
    m.T = chunk; m.cs = (cudaStream_t) stream; m.stats = &stats_;
    if (g.n_embd != N || g.hc != HC || g.hc_lr != LR || g.n_expert < 1 || ss.k != K) {
=======
    Impl& m = *impl_;
    if (m.cs) cudaStreamSynchronize(m.cs);
    if (m.xfer) cudaStreamSynchronize(m.xfer);
    if (m.xsum) cudaStreamSynchronize(m.xsum);
    if (m.ev_mixed) cudaEventDestroy(m.ev_mixed);
    for (cudaEvent_t e : m.ev_in) if (e) cudaEventDestroy(e);
    for (cudaEvent_t e : m.ev_piece) if (e) cudaEventDestroy(e);
    if (m.xfer) cudaStreamDestroy(m.xfer);
    if (m.xsum) cudaStreamDestroy(m.xsum);
    for (float* h : m.ple_host) if (h) cudaFreeHost(h);
    if (m.ids_host) cudaFreeHost(m.ids_host);
    if (m.w_host) cudaFreeHost(m.w_host);
    if (m.owned) cudaFree(m.owned);
}

bool Prefill::init(const core::WeightTable& wt, const core::ModelGeometry& g, core::SessionState& ss,
                   core::ExpertSource* src, const core::ExpertCache* cache, const int32_t* host_res, int64_t max_chunk,
                   void* stream, ExpertRunner* offload, std::string& err) {
    Impl& m = *impl_;
    m.wt = &wt; m.g = &g; m.ss = &ss; m.cache = cache; m.host_res = host_res; m.offload = offload;
    m.max_chunk = max_chunk; m.cs = (cudaStream_t) stream;
    if (g.n_embd != N || g.hc != HC || g.hc_lr != LR || g.n_expert != NE || ss.k != K) {
>>>>>>> their fork (github.com-eddoursul-Strata-custom)
        err = "prefill: geometry differs from the artifact's"; return false;
    }
    m.s = shapes_of(g);
    m.cap = strata::kernels::qsa_selection_width(strata::kernels::kTopkMaxCells, m.s);
    m.max_blocks = ss.qsa_states[0].max_cells / m.s.idx_block + 2;
    m.ev_in.assign((size_t) ((max_chunk + SUB - 1) / SUB), nullptr);
    m.ev_piece.assign(m.ev_in.size(), nullptr);
    bool pieces = true;
    for (auto* v : {&m.ev_in, &m.ev_piece})
        for (cudaEvent_t& e : *v)
            pieces = pieces && cudaEventCreateWithFlags(&e, cudaEventDisableTiming) == cudaSuccess;
    if (!pieces || cudaStreamCreateWithFlags(&m.xfer, cudaStreamNonBlocking) != cudaSuccess ||
        cudaStreamCreateWithFlags(&m.xsum, cudaStreamNonBlocking) != cudaSuccess ||
        cudaEventCreateWithFlags(&m.ev_mixed, cudaEventDisableTiming) != cudaSuccess ||
        cudaHostAlloc((void**) &m.ids_host, (size_t) max_chunk * K * 4, cudaHostAllocDefault) != cudaSuccess ||
        cudaHostAlloc((void**) &m.w_host, (size_t) max_chunk * K * 4, cudaHostAllocDefault) != cudaSuccess) {
        err = "prefill: streams, events and pinned routing buffers";
        return false;
    }
<<<<<<< this fork
    // the CPU share's staged-chunk limit, before the ring is sized below (a caller that sizes loans first armed it)
    arm_cpu_share(cpu_pool_ != nullptr);
    for (int b = 0; next_ != nullptr && b < 2; ++b)
        if (!m.hand[b] && cudaHostAlloc((void**) &m.hand[b], (size_t) chunk * D * 4, cudaHostAllocPortable) != cudaSuccess) {
            err = "prefill: the layer split's hand-off buffers";
            return false;
        }
    if (m.tok_dev == nullptr) {
        if (const cudaError_t e = cudaMalloc((void**) &m.tok_dev, (size_t) chunk * sizeof(int32_t)); e != cudaSuccess) {
            err = std::string("prefill: the token id buffer (") + cudaGetErrorString(e) + ")";
            return false;
        }
        m.owned.push_back(m.tok_dev);
        m.tok_host.resize((size_t) chunk);
        m.tok_up.init((size_t) chunk);
    }
    if (cudaStreamCreateWithFlags(&m.copy, cudaStreamNonBlocking) != cudaSuccess) { err = "prefill: copy stream"; return false; }
    const size_t T = (size_t) chunk;
    m.T_max = chunk;
    m.borrowed = borrow != nullptr;
    m.borrow_spill = m.borrowed && borrow_spill;
    bool ok = true;
    // one-time: events, the stager, the host buffers (for the largest chunk), the identity page table
    for (int i = 0; i < ring_cap(); ++i) {
        if (cudaEventCreateWithFlags(&m.copied[i], cudaEventDisableTiming) != cudaSuccess) ok = false;
        if (cudaEventCreateWithFlags(&m.used[i], cudaEventDisableTiming) != cudaSuccess) ok = false;
    }
    if (!m.stager) {
        m.stager = std::make_unique<Stager>();
        const int hw = (int) std::thread::hardware_concurrency();
        const char* stv = std::getenv("STRATA_STAGER_THREADS");   // D-5: the host copy threads of unpinned blobs
        // A GGUF read in place (UD-Q4_K_XL beyond its RAM budget): most of a chunk's blobs are page faults on the
        // SSD, so the copies need many reads in flight - 32 threads and a 128-deep ring read a 4K chunk in 29 s
        // instead of 71 s on an RTX 5070 / NVMe PC (4 threads, 16 deep: the defaults, kept for every other source)
        bool files = false;
        for (int64_t l = 0; src != nullptr && !files && l < g.n_layers; ++l)
            for (int64_t e = 0; !files && e < g.n_expert; ++e) files = src->transient(l, e);
        // #1353: ... but a source answers transient() for every blob it copies out of the GGUF, including the ones whose
        // pages are already in the page cache (a Q8 pack on a big-RAM box with a warm cache).  Those copies are memcpy
        // from RAM, where 32 threads and a 128-deep ring ran 4-7x slower than the defaults.  So ask the OS: the SSD
        // profile only when fewer than 90% of a sample of the blobs' pages are resident (-1: it cannot tell, keep it).
        if (files && !stv) {
            const char* ev = std::getenv("STRATA_STAGER_SSD");   // 1 / 0: force the profile (A/B, support)
            if (ev != nullptr && ev[0] != '\0') files = ev[0] != '0';
            else if (const double warm = src->cached_share(256); warm >= 0.9) {
                files = false;
                std::fprintf(stderr, "prefill: the GGUF experts are %.0f%% in the page cache; the stager takes the RAM profile\n", warm * 100.0);
            }
        }
        const int threads = stv ? std::clamp(std::atoi(stv), 1, 32) : files ? 32 : std::max(2, std::min(4, hw / 4));
        if (files && std::getenv("STRATA_STAGER_RING") == nullptr) m.stager->kRing = 4 * threads;
        // pp-opt: with files, each thread claims runs of neighbouring blobs and reads them as one request (copy_blobs);
        // STRATA_STAGER_BATCH=1 is the A/B arm (one blob per request, as before)
        {
            const char* bv = std::getenv("STRATA_STAGER_BATCH");
            const int want = bv ? std::atoi(bv) : files ? 8 : 1;
            m.stager->batch = std::clamp(want, 1, Stager::kMaxBatch);
        }
        if (!m.stager->init((size_t) MAXBLOB(), threads)) ok = false;
        m.stager->batch = std::min(m.stager->batch, m.stager->kRing);
    }
=======
    const size_t T = (size_t) max_chunk;
>>>>>>> their fork (github.com-eddoursul-Strata-custom)
    m.steps_host.resize(T * strata::kernels::kStepCount);
    m.cnt.resize(NE);
    m.off.resize(NE + 1);
    m.entry_of.resize(T * K);
    for (int i = 0; i < 2; ++i) {
        m.ple_rows[i].resize(T * strata::kernels::PLE_N_HEADS);
        if (cudaHostAlloc((void**) &m.ple_host[i], T * N * 4, cudaHostAllocDefault) != cudaSuccess) {
            err = "prefill: pinned PLE row buffers";
            return false;
        }
<<<<<<< this fork
        if (!m.ple_copied[b] && cudaEventCreateWithFlags(&m.ple_copied[b], cudaEventDisableTiming) != cudaSuccess)
            ok = false;
        m.ple_rows[b].resize(T * strata::kernels::PLE_N_HEADS);
    }
    if (ss.qsa_states[ss.qsa_primary()].kv_mode == 1) {   // KV streaming: the staging pool's identity page table
        const int64_t pages = ss.qsa_states[ss.qsa_primary()].n_pages;
        std::vector<int32_t> ident((size_t) pages);
        for (int64_t i = 0; i < pages; ++i) ident[(size_t) i] = (int32_t) i;
        if (cudaMalloc((void**) &m.ident_table, ident.size() * 4) != cudaSuccess ||
            cudaMemcpy(m.ident_table, ident.data(), ident.size() * 4, cudaMemcpyHostToDevice) != cudaSuccess)
            ok = false;
        else
            m.owned.push_back(m.ident_table);
    }
    if (!ok) { err = "prefill: host buffers or events for a chunk of " + std::to_string(chunk) + " tokens"; return false; }
    Alloc o;
    o.base = (uint8_t*) borrow;
    o.cap = borrow_bytes;
    o.spill = m.borrow_spill;
    o.owned = &m.owned;
    {
        uint16_t* gs = o.take<uint16_t>((size_t) GEMM_SCRATCH, ok);
        void* ws = o.take<uint8_t>(GEMM_WS, ok);
        if (!ok) { err = "prefill: GEMM scratch does not fit"; return false; }
        if (!m.gemm.init_external(stream, gs, GEMM_SCRATCH, ws, GEMM_WS, err)) return false;
        // this stage's device decides, once, here (init runs on the stage's own device): its gr_* kernels write the
        // image its GEMMs read, and set_act_f16 writes that device's symbol
        const core::OnDevice on_stage(m.device);
        m.f16_io = prompt_f16();
        m.gemm.set_f16_io(m.f16_io);
        set_act_f16(m.f16_io);
=======
>>>>>>> their fork (github.com-eddoursul-Strata-custom)
    }
    int dev = 0;
    cudaGetDevice(&dev);
    return m.gemm.init(stream, err) &&
           m.local.init(dev, dev, stream, src, cache, host_res, NE, max_chunk, offload == nullptr, err);
}

<<<<<<< this fork
// Every device buffer of a chunk of T tokens, from the Alloc `alloc` (after the GEMM scratch and workspace): `init`
// once, and `relayout` for a request's own chunk.  The order is `bytes_needed`'s.
bool Prefill::carve(size_t T, void* alloc) {
    Impl& m = *impl_;
    Alloc& o = *static_cast<Alloc*>(alloc);
    const core::ModelGeometry& g = *m.g;
    core::SessionState& ss = *m.ss;
    bool ok = true;
    m.emb = o.take<float>(T * N, ok); m.R = o.take<float>(T * D, ok);
    m.xn = gr_unfused() ? o.take<float>(T * D, ok) : nullptr;   // F-1: not needed (gr_mix_r reads R)
    m.grs = o.take<float>(T * HC, ok);
    m.xn16 = o.take<uint16_t>(T * (D + (hc_pad() ? XN_PAD : 0)), ok); m.lo = o.take<float>(T * LR, ok); m.lo16 = o.take<uint16_t>(T * LR, ok);
    m.gated = o.take<float>(T * D, ok); m.inj = o.take<float>(T * HC, ok);
    m.mixed = o.take<float>(T * N, ok); m.mixed_bf = o.take<uint16_t>(T * N, ok);
    m.mixed_h = o.take<uint16_t>(T * N, ok);
    // Embedding rows are dead after gr_broadcast on m.cs; subsequent half outputs use the same stream.
    m.bo = m.emb;
    if (bf16x2_hc(m.f16_io)) { m.xn16_lo = o.take<uint16_t>(T * D, ok); m.lo16_lo = o.take<uint16_t>(T * LR, ok); }
    if (bf16x2(m.f16_io)) m.mixed_bf_lo = o.take<uint16_t>(T * N, ok);
    m.steps_dev = o.take<int32_t>(T * strata::kernels::kStepCount, ok);
    strata::kernels::QsaShapes s = core::qsa_shapes(g);
    m.cap = strata::kernels::qsa_selection_width(strata::kernels::kTopkMaxCells, s);
    m.max_blocks = ss.qsa_states[ss.qsa_primary()].max_cells / s.idx_block + 2;
    {
        // one region for the attention half's and the MoE half's scratch (see gdn_set_bytes)
        const bool fz = fused_layout(T, m.src != nullptr);
        m.fused_bufs = fz;
        const MoeBufs mb = moe_bufs(T, m.g->n_expert, fz);
        const uint64_t region = std::max({gdn_set_bytes(T), qsa_set_bytes(T, m.cap, m.max_blocks, m.sel_batch,
                                                                           m.attn_batch, s), moe_set_bytes(T, m.g->n_expert, fz)});
        uint8_t* base = o.take<uint8_t>((size_t) region, ok);
        m.region = base;
        m.region_bytes = region;
        Alloc a;
        a.base = base; a.cap = region; a.owned = &m.owned;
        m.qkv = a.take<float>(T * C, ok); m.z = a.take<float>(T * ZV, ok); m.ab = a.take<float>(T * 2 * HV, ok);
        m.gate = a.take<float>(T * HV, ok); m.beta = a.take<float>(T * HV, ok); m.hbuf = a.take<float>(T * C, ok);
        m.y = a.take<float>(T * ZV, ok); m.y_h = a.take<uint16_t>(T * (ZV + (pf_pad() ? ZV_PAD : 0)), ok);
        Alloc b;
        b.base = base; b.cap = region; b.owned = &m.owned;
        m.Kc = b.take<float>(T * 512, ok); m.Vc = b.take<float>(T * 512, ok); m.Qf = b.take<float>(T * 12288, ok);
        m.q = b.take<float>(T * ZV, ok); m.idx_raw = b.take<float>(T * 128, ok); m.q_idx = b.take<float>(T * 512, ok);
        m.attn = b.take<float>(T * ZV, ok); m.attn_h = b.take<uint16_t>(T * (ZV + (pf_pad() ? ZV_PAD : 0)), ok);
        m.sel_ids = b.take<int32_t>(T * (size_t) m.cap, ok);
        m.sel_scores = b.take<float>((size_t) m.sel_batch * (size_t) m.max_blocks, ok);
        m.attn_scratch = b.take<float>((size_t) m.attn_batch * strata::kernels::qsa_decode_attn_scratch_floats(m.cap, s), ok);
        Alloc c;
        c.base = base; c.cap = region; c.owned = &m.owned;
        m.logits = c.take<float>(T * m.g->n_expert, ok); m.w = c.take<float>(T * K, ok); m.ids = c.take<int32_t>(T * K, ok);
        m.slot_dev = c.take<int32_t>(T * K, ok); m.src_dev = c.take<int32_t>(T * K, ok);
        const MmqPlan& mp = mmq_plan();
        m.Xs = mp.fallback ? c.take<uint16_t>(T * K * N, ok) : nullptr;
        m.GU = c.take<float>(mb.gu, ok);
        m.Hh = mp.fallback ? c.take<uint16_t>(T * K * 640, ok) : nullptr;
        m.Dm = c.take<float>(T * K * N, ok);
        m.sgate = c.take<float>(T * 640, ok); m.sup = c.take<float>(T * 640, ok); m.sh_h = c.take<uint16_t>(T * 640, ok);
        m.shared = c.take<float>(T * N, ok); m.sg = c.take<float>(T, ok);
        if (mp.any) {
            m.Xq = c.take<uint8_t>(mb.xq, ok);
            m.H = c.take<float>(mb.h, ok);
            m.Hq = c.take<uint8_t>(mb.hq, ok);
            m.Xscale = c.take<float>(T * K, ok); m.Hscale = c.take<float>(T * K, ok);
            m.grp_tail = c.take<float>(MMQ_GROUP * 4, ok);
            m.row_sd = c.take<float>(T * K, ok);
        }
        if (base == nullptr) ok = false;
    }
    for (int i = 0; i < DQ; ++i) { m.dq_gu[i] = o.take<uint16_t>(1280 * 2560, ok); m.dq_d[i] = o.take<uint16_t>(2560 * 640, ok); }
    if (mmq_plan().any) {
        const MmqPlan& mp = mmq_plan();
        m.ids_identity = o.take<int32_t>(T * K, ok);
        m.bounds_dev = o.take<int32_t>((size_t) (2 * (m.g->n_expert + m.g->n_expert / MMQ_GROUP + 2)), ok);
        m.grp_gu = o.take<uint8_t>(MMQ_GROUP * mp.gu_max + MMQ_TAIL, ok);
        m.grp_d = o.take<uint8_t>(MMQ_GROUP * mp.d_max + MMQ_TAIL, ok);
        // (written at every run's start, not here: when serving, these are live expert-cache slots until a request
        // lends them - a write now would corrupt a resident expert)
        if (!m.mmq_ctx) m.mmq_ctx = std::make_unique<mmq::Context>();
    }
    m.ring = ring_slots(T);
    if (o.base == nullptr && m.ring > 0) {
        // OWNED buffers: the ring in ONE allocation.  384 separate 2.7 MiB cudaMallocs each round up to a 2 MiB page
        // (~1.3 MiB a slot, ~0.5 GiB in all) that no count ever saw.  A borrowed region keeps its per-slot layout (and
        // so its price, `bytes_needed`: the loans' slot counts do not move).
        uint8_t* ring_base = o.take<uint8_t>((size_t) m.ring * (size_t) MAXBLOB(), ok);
        for (int i = 0; ok && i < m.ring; ++i) {
            m.stage_dev[i] = ring_base + (size_t) i * (size_t) MAXBLOB();
            m.stage_live[i] = false;
            m.used_of[i] = i;
        }
    } else {
        for (int i = 0; i < m.ring; ++i) {
            m.stage_dev[i] = o.take<uint8_t>((size_t) MAXBLOB(), ok);
            m.stage_live[i] = false;                        // a new buffer: nothing of an earlier layout to wait for
            m.used_of[i] = i;
        }
    }
    m.ple_emb = o.take<float>(T * N, ok);
    m.ple_norm = o.take<float>((size_t) strata::kernels::NG_HC_DIM, ok);
    take_stage(o, ss, s, m.stage, ok);
    m.T = (int64_t) T;
    return ok;
}

bool Prefill::relayout(int64_t chunk, void* borrow, uint64_t borrow_bytes, std::string& err) {
    Impl& m = *impl_;
    if (m.borrow_spill) {
        err = "prefill: a hybrid borrowed/owned layout cannot be changed between requests";
        return false;
    }
    if (!m.borrowed || borrow == nullptr || chunk <= 0 || chunk > m.T_max) {
        err = "prefill: relayout needs borrowed buffers and a chunk of at most " + std::to_string(m.T_max);
        return false;
    }
    if (cudaStreamSynchronize(m.cs) != cudaSuccess || cudaStreamSynchronize(m.copy) != cudaSuccess ||
        (m.kv_copy && cudaStreamSynchronize(m.kv_copy) != cudaSuccess)) {
        err = "prefill: relayout: the stream failed";
        return false;
    }
    bool ok = true;
    Alloc o;
    o.base = (uint8_t*) borrow;
    o.cap = borrow_bytes;
    o.owned = &m.owned;
    uint16_t* gs = o.take<uint16_t>((size_t) GEMM_SCRATCH, ok);
    void* ws = o.take<uint8_t>(GEMM_WS, ok);
    if (ok) m.gemm.rebind(gs, GEMM_SCRATCH, ws, GEMM_WS);
    if (!ok || !carve((size_t) chunk, &o)) {
        size_t fb = 0, tb = 0;
        cudaMemGetInfo(&fb, &tb);
        (void) cudaGetLastError();
        err = "prefill: device buffers for a chunk of " + std::to_string(chunk) + " tokens do not fit (" +
              std::to_string(fb >> 20) + " of " + std::to_string(tb >> 20) + " MiB free at the failure; " +
              std::to_string(o.used >> 20) + " MiB taken, the failing buffer wanted " +
              std::to_string(o.failed_bytes >> 20) + " MiB)";
        return false;
    }
    return true;
=======
uint64_t Prefill::bytes_needed(const core::ModelGeometry& g, const core::SessionState& ss, int64_t chunk, bool offload) {
    const strata::kernels::QsaShapes s = shapes_of(g);
    Alloc a;
    layout(a, chunk, strata::kernels::qsa_selection_width(strata::kernels::kTopkMaxCells, s),
           ss.qsa_states[0].max_cells / s.idx_block + 2, offload);
    return a.used;
}

uint64_t Prefill::bytes_for(int64_t chunk) const {
    const Impl& m = *impl_;
    return bytes_needed(*m.g, *m.ss, chunk, m.offload != nullptr) +
           (m.offload == nullptr && chunk >= ExpertRunner::kPrefetchMin ? m.local.prefetch_need(nullptr) : 0);
>>>>>>> their fork (github.com-eddoursul-Strata-custom)
}

bool Prefill::bind(void* region, uint64_t bytes, int64_t chunk, std::string& err) {
    Impl& m = *impl_;
<<<<<<< this fork
#if defined(STRATA_GPU_METAL)
    (void) m; (void) mtp; (void) R_rows; (void) next_tokens; (void) n; (void) cell0; (void) err;
    if (why) *why = "the Metal window feeds the draft layer itself";
    return false;
#endif
    const core::OnDevice on_device(m.device);   // the stage's own GPU current (a layer split's drafter: the last stage)
    auto decline = [&](const char* reason) { if (why != nullptr) *why = reason; return false; };
    static const bool off = [] { const char* v = std::getenv("STRATA_MTP_BATCH"); return v != nullptr && v[0] == '0'; }();
    // A ring (KV streaming: the drafter's window, page p in slot p % n_slots over a host copy) takes the same appends
    // with its own page table and host copy, as a streamed main layer does; the cells written are those the window can
    // still reach (r0 below), which the ring holds, so no two of them share a slot.  STRATA_MTP_BATCH_RING=0: the
    // drafter's own pass for a ring (the A/B).
    static const bool ring_ok = [] { const char* v = std::getenv("STRATA_MTP_BATCH_RING"); return v == nullptr || v[0] != '0'; }();
    core::QsaState& st = mtp.kv_state_rw();
    if (off) return decline("STRATA_MTP_BATCH=0");
    if (n <= 0 || m.g == nullptr || m.region == nullptr) return decline("no prompt scratch region on this path");
    if (st.kv_mode != 0 && !(st.kv_mode == 2 && ring_ok))
        return decline(st.kv_mode == 2 ? "the drafter's K/V is a ring and STRATA_MTP_BATCH_RING=0" : "the drafter's K/V mode is not paged/ring");
    if (st.kv_hybrid) return decline("the drafter's K/V is hybrid (kv_hybrid)");
    if (mtp.device() != m.device) return decline("the drafter is on another device than this prefill path");
    const auto t0 = Clock::now();
    const core::ModelGeometry& g = *m.g;
    const int64_t Nn = g.n_embd, HCN = g.hc * g.n_embd, KV = g.n_head_kv * g.head_dim;
    constexpr int kQ8_0 = 8;   // GGML_TYPE_Q8_0
    const float* w_ne = mtp.tensor_f32("pre_fc_norm_embedding.weight");
    const void* w_fe = mtp.tensor_q8("fc_embedding.weight");
    const float* w_nh = mtp.tensor_f32("pre_fc_norm_hidden.weight");
    const void* w_fh = mtp.tensor_q8("fc_hidden.weight");
    const float* w_hn = mtp.tensor_f32("attn_hyper_connection.hc_norm.weight");
    const uint16_t* w_dn = mtp.tensor_bf16("attn_hyper_connection.input_mix_weight_down.weight");
    const uint16_t* w_up = mtp.tensor_bf16("attn_hyper_connection.input_mix_weight_up.weight");
    const void* w_k = mtp.tensor_q8("self_attn.k_proj.weight");
    const void* w_v = mtp.tensor_q8("self_attn.v_proj.weight");
    const float* w_kn = mtp.tensor_f32("self_attn.k_norm.weight");
    if (!w_ne || !w_fe || !w_nh || !w_fh || !w_hn || !w_dn || !w_up || !w_k || !w_v || !w_kn)
        return decline("a drafter tensor is missing (Q8_0 projections / norms)");
    const core::NativeEmbed* nemb = core::native_embed();
    const core::WeightRef* wemb = nemb ? nullptr : m.wt->find("token_embd.weight");
    if (!nemb && (wemb == nullptr || wemb->codebook_iq4nl || wemb->ne0 != g.n_embd || wemb->group_elems <= 0 ||
                  (wemb->code_bits != 2 && wemb->code_bits != 4 && wemb->code_bits != 8)))
        return decline("the token embedding is not gatherable on this path");
    // the cells the drafter's window can still reach
    const int64_t r0 = std::max<int64_t>(0, mtp.first_needed() - cell0);
    if (r0 >= n) return true;
    // per row: emb/e2 (N), en16 (N half), hn/h2/Rm/gated (HCN), hn16/xn16 (HCN half), lo (LR) + lo16, grs, mixed (N) +
    // mixed_h, K and V (KV each), the token id
    // E-9: the drafter's Q8_0 matrices through Q8_1 x Q8_0 MMQ - its own pass's integer dot products (mmvq), so
    // its K/V stay close to what the drafter computes itself; STRATA_MTP_BATCH_F16=1: FP16 GEMMs (the A/B)
    static const bool f16_only = [] { const char* v = std::getenv("STRATA_MTP_BATCH_F16"); return v && v[0] == '1'; }();
    const bool q8 = !f16_only && mmq::built() && mmq::fits(kQ8_0, Nn) && mmq::fits(kQ8_0, KV);   // #420
    const uint64_t per_row = 4 * (2 * Nn + 4 * HCN + LR + HC + Nn + 2 * KV + 1) + 2 * (Nn + 2 * HCN + LR + Nn) + 64 +
                             (q8 ? (uint64_t) mmq::q8_bytes(g.hc, Nn) + 4 * g.hc : 0);
    int64_t B = std::min<int64_t>(n - r0, (int64_t) (m.region_bytes / per_row) & ~(int64_t) 63);
    if (st.kv_mode == 2)   // a ring: one batch's cells must not share a slot (a batch can straddle one page more)
        B = std::min<int64_t>(B, ((st.n_slots - 1) * strata::kernels::qsa_real_shapes().page_size) & ~(int64_t) 63);
    if (B < 64) return decline("too little scratch for a 64-row batch");
    uint8_t* q = m.region;
    auto carve = [&](size_t bytes) { void* p = q; q += (bytes + 255) & ~(size_t) 255; return p; };
    float* emb = (float*) carve((size_t) B * Nn * 4);
    float* e2 = (float*) carve((size_t) B * Nn * 4);
    uint16_t* en16 = (uint16_t*) carve((size_t) B * Nn * 2);
    float* hn = (float*) carve((size_t) B * HCN * 4);
    float* h2 = (float*) carve((size_t) B * HCN * 4);
    float* Rm = (float*) carve((size_t) B * HCN * 4);
    float* gated = (float*) carve((size_t) B * HCN * 4);
    uint16_t* hn16 = (uint16_t*) carve((size_t) B * HCN * 2);
    uint16_t* xn16 = (uint16_t*) carve((size_t) B * HCN * 2);
    float* lo = (float*) carve((size_t) B * LR * 4);
    uint16_t* lo16 = (uint16_t*) carve((size_t) B * LR * 2);
    float* grs = (float*) carve((size_t) B * HC * 4);
    float* mixed = (float*) carve((size_t) B * Nn * 4);
    uint16_t* mixed_h = (uint16_t*) carve((size_t) B * Nn * 2);
    float* Kc = (float*) carve((size_t) B * KV * 4);
    float* Vc = (float*) carve((size_t) B * KV * 4);
    int32_t* tok = (int32_t*) carve((size_t) B * 4);
    void* xq = q8 ? carve(mmq::q8_bytes(B * g.hc, Nn)) : nullptr;
    int32_t* ident = q8 ? (int32_t*) carve((size_t) B * g.hc * 4) : nullptr;
    int32_t* bnd = q8 ? (int32_t*) carve(16) : nullptr;
    if ((uint64_t) (q - m.region) > m.region_bytes) return decline("the batch buffers do not fit the scratch region");
    static const bool timing = std::getenv("STRATA_DRAFT_TIMING") != nullptr;   // debug: where this pass's time goes
    if (timing) cudaStreamSynchronize(m.cs);
    const auto ti0 = Clock::now();
    if (!mtp.idle(err)) return false;   // the drafter's own stream (its graph uploads) before this writes its K/V
    const double ms_idle = ms_since(ti0);
    const auto tl0 = Clock::now();
    strata::kernels::QsaShapes s = core::qsa_shapes(g);
    std::vector<int32_t> tk((size_t) B);
    if (q8) {
        if (!m.mmq_ctx) m.mmq_ctx = std::make_unique<mmq::Context>();
        mmq::iota(ident, B * g.hc, m.cs);
    }
    // y[rows, n_out] = x[rows, k] . w^T for a Q8_0 matrix: MMQ from the FP32 rows (bounds slot 0: rows, 1: rows*hc)
    auto proj = [&](const float* x, const uint16_t* x16, const void* w, float* y, int64_t rows, int64_t n_out,
                    int64_t k, int slot) {
        if (!q8) { m.gemm.native(x16, kQ8_0, w, y, rows, n_out, k); return; }
        mmq::quantize(x, nullptr, xq, kQ8_0, k, k, rows, m.cs);
        mmq::Product p;
        p.w = w; p.type = kQ8_0; p.w_rows = n_out; p.w_cols = k; p.expert_bytes = mmq::matrix_bytes(kQ8_0, n_out, k);
        p.n = 1; p.xq = xq; p.bounds = bnd + 2 * slot; p.ids = ident; p.total_rows = rows; p.max_rows = rows;
        p.dst = y; p.ld_dst = n_out;
        m.mmq_ctx->run(p, m.cs);
    };
    for (int64_t b0 = r0; b0 < n; b0 += B) {
        const int64_t nb = std::min(B, n - b0), c0 = cell0 + b0;
        for (int64_t i = 0; i < nb; ++i) tk[(size_t) i] = next_tokens[b0 + i];
        const int32_t bh[4] = {0, (int32_t) nb, 0, (int32_t) (nb * g.hc)};
        // #1996: bh is on this stack and tk is refilled every pass: both go up through pinned slots
        if (q8 && m.up_ring.h2d(bnd, bh, sizeof bh, m.cs) != cudaSuccess) {
            err = "prefill: the draft bounds' upload failed";
            return false;
        }
        if (m.up_ring.h2d(tok, tk.data(), (size_t) nb * 4, m.cs) != cudaSuccess) {
            err = "prefill: the draft tokens' upload failed";
=======
    if (region == nullptr) {
        if (m.owned != nullptr) return true;   // allocated once, for the longest chunk
        chunk = m.max_chunk;
        bytes = bytes_needed(*m.g, *m.ss, chunk, m.offload != nullptr);
        if (cudaMalloc(&m.owned, bytes) != cudaSuccess) {
            m.owned = nullptr;
            err = "prefill: " + std::to_string(bytes >> 20) + " MiB of device buffers";
>>>>>>> their fork (github.com-eddoursul-Strata-custom)
            return false;
        }
        region = m.owned;
    }
    if (chunk <= 0 || chunk > m.max_chunk) {
        err = "prefill: a chunk of " + std::to_string(chunk) + " tokens (set up for " + std::to_string(m.max_chunk) + ")";
        return false;
    }
    Alloc a;
    a.base = (uint8_t*) region;
    a.cap = bytes;
    m.b = layout(a, chunk, m.cap, m.max_blocks, m.offload != nullptr);
    if (!a.ok) {
        err = "prefill: device buffers for a chunk of " + std::to_string(chunk) + " tokens do not fit";
        return false;
<<<<<<< this fork
    PeerPrefill& P = *m.help_pp;
    P.T_max = T;
    P.cap_rows = T * K;
    P.G = T;                                  // one expert never has more rows than the chunk has tokens
    P.mixed = h.mixed;
    P.Xq_g = h.Xq;                            // the helper's per-layer buffers hold T * K rows: a group's T fit
    P.Hq_g = h.Hq;
    P.GU_g = h.GU;
    P.H_g = h.H;
    P.Dm_b[0] = h.Dm;
    P.Dm_b[1] = h.Dm + (size_t) T * N;
    P.src = h.src_dev;
    P.ident = h.ids_identity;
    P.bounds = h.bounds_dev;
    P.grp_gu = h.grp_gu;
    P.grp_d = h.grp_d;
    P.run_ctx = h.mmq_ctx.get();
    P.RP = std::min(h.ring, kSplitHelpRing);
    P.pstage.assign(h.stage_dev, h.stage_dev + P.RP);
    P.plive.assign((size_t) P.RP, 0);
    P.dm_live[0] = P.dm_live[1] = false;
    P.out_pending = false;
    P.ps_frac = frac;
    if (!P.p2p) {   // grown once to the largest chunk it helped
        auto grow = [](auto*& p, size_t& cap, size_t n) {
            if (cap >= n) return true;
            if (p) cudaFreeHost(p);
            p = nullptr;
            cap = 0;
            if (cudaHostAlloc((void**) &p, n * sizeof(*p), cudaHostAllocPortable | cudaHostAllocMapped) != cudaSuccess) {
                cudaGetLastError();
                p = nullptr;
                return false;
            }
            cap = n;
            return true;
        };
        size_t bounds_cap = P.host_bounds ? PeerPrefill::kHostBounds : 0;
        if (!grow(P.host_x, P.host_x_cap, (size_t) T * N) || !grow(P.host_rows, P.host_rows_cap, (size_t) T * K * N) ||
            !grow(P.host_src, P.host_src_cap, (size_t) T * K) || !grow(P.host_bounds, bounds_cap, PeerPrefill::kHostBounds))
            return false;
    }
    const core::OnDevice on(P.dev);
    mmq::iota(P.ident, T * K, P.s);           // the helper's row table: a refill may have overwritten its slots
    return true;
}

bool Prefill::set_peer(core::PeerExperts* peer, int64_t cap_rows, std::string& err) {
    Impl& m = *impl_;
    if (peer == nullptr || !peer->valid()) { m.pp.reset(); return true; }
    // Without P2P the activations and the result rows take the mapped host route of the layer split's helper
    // (STRATA_PF_PEER_HOST=0: refuse as before, the prompt path then stays on the primary)
    const bool host_route_ok = [] { const char* v = std::getenv("STRATA_PF_PEER_HOST"); return v == nullptr || std::atoi(v) != 0; }();
    if (!peer->p2p() && !host_route_ok) { err = "prefill peer: the two GPUs cannot access each other (no P2P)"; return false; }
    if (!mmq_plan().any) { err = "prefill peer: needs the MMQ prompt path"; return false; }
    {   // NVFP4: the peer's MMQ products have no per-expert scales (grp_tail, row_sd); its decode rows do
        const auto& lay = strata::kernels::cpu::expert_layout();
        for (const auto& f : lay.fmt)
            if (lay.native && f.tail_off) { err = "prefill peer: not for NVFP4 experts yet"; return false; }
    }
    auto pp = std::make_unique<PeerPrefill>();
    pp->peer = peer;
    pp->p2p = peer->p2p();
    pp->T_max = m.T_max;
    pp->cap_rows = std::max<int64_t>(1, std::min<int64_t>(cap_rows, m.T_max * K));
    const int64_t R = pp->cap_rows;
    const MmqPlan& mp = mmq_plan();
    if (cudaEventCreateWithFlags(&pp->ev_in, cudaEventDisableTiming) != cudaSuccess) { err = "prefill peer: event"; return false; }
    if (!pp->p2p) {   // the host route's copies on a stream of their own (STRATA_PF_PEER_XSTREAM=0: the compute stream)
        const char* xv = std::getenv("STRATA_PF_PEER_XSTREAM");
        if ((xv == nullptr || std::atoi(xv) != 0) &&
            (cudaStreamCreateWithFlags(&pp->x_stream, cudaStreamNonBlocking) != cudaSuccess ||
             cudaEventCreateWithFlags(&pp->ev_x, cudaEventDisableTiming) != cudaSuccess)) {
            err = "prefill peer: copy stream";
            return false;
        }
    }
    int prev = 0;
    cudaGetDevice(&prev);
    pp->dev = peer->device();
    cudaSetDevice(pp->dev);
    bool ok = cudaStreamCreateWithFlags(&pp->s, cudaStreamNonBlocking) == cudaSuccess &&
              cudaEventCreateWithFlags(&pp->ev_done, cudaEventDisableTiming) == cudaSuccess;
    {
        const char* v = std::getenv("STRATA_PF_PEER_OUT_PIPE");   // =0: one copy after the last group (the A/B)
        pp->out_pipe = (v == nullptr || std::atoi(v) != 0) || pp->compact;   // compact always pipes
    }
    if (ok && pp->out_pipe) {
        ok = cudaStreamCreateWithFlags(&pp->s_out, cudaStreamNonBlocking) == cudaSuccess;
        for (int i = 0; ok && i < PeerPrefill::kGrpEv; ++i)
            ok = cudaEventCreateWithFlags(&pp->ev_grp[i], cudaEventDisableTiming) == cudaSuccess;
=======
>>>>>>> their fork (github.com-eddoursul-Strata-custom)
    }
    m.gemm.set_buffers(nullptr, 0, m.b.gemm_ws, GEMM_WS);
    // the rest of a lent region takes this GPU's prefetched experts (outside the union: copies run beside the steps)
    const bool area = region != m.owned && bytes > a.used;
    if (!m.local.bind(m.b.local, m.b.local_bytes, chunk, err, area ? (uint8_t*) region + a.used : nullptr,
                      area ? bytes - a.used : 0))
        return false;
    m.T = chunk;
    return true;
}

<<<<<<< this fork
void Prefill::set_pinned_share(double share) { g_pinned_share = share; }
void Prefill::set_ring_override(int slots) { g_ring_override = slots > 0 ? slots : 0; }
void Prefill::set_ring_budget(int slots, int64_t small_max) {
    g_ring_budget = slots > 0 ? slots : 0;
    g_ring_small_max = small_max > 0 ? small_max : 0;
}
double Prefill::pinned_share() { return g_pinned_share; }
int64_t Prefill::stream_all_min_tokens() { return stream_all_min(); }

uint64_t Prefill::bytes_needed(const core::ModelGeometry& g, const core::SessionState& ss, int64_t chunk, bool src) {
    return bytes_needed_impl(g, ss, chunk, false, src);
}

// What `init` really allocates when the prompt path owns its buffers (no loan): every cudaMalloc rounds up to a 2 MiB
// page, and the ring is one allocation (carve).  `bytes_needed` stays the borrowed region's sum.
uint64_t Prefill::bytes_needed_owned(const core::ModelGeometry& g, const core::SessionState& ss, int64_t chunk,
                                      bool src) {
    return bytes_needed_impl(g, ss, chunk, true, src);
}

uint64_t Prefill::bytes_needed_impl(const core::ModelGeometry& g, const core::SessionState& ss, int64_t chunk,
                                    bool owned_pages, bool src) {
    // the same allocation sequence as `init` (`carve`), counted - with the MoE layout `carve` picks for the same source
    const size_t T = (size_t) chunk;
    bool ok = true;
    Alloc o;
    o.count_only = true;
    if (owned_pages) o.granule = 2ull << 20;
    o.take<uint16_t>((size_t) GEMM_SCRATCH, ok);
    o.take<uint8_t>(GEMM_WS, ok);
    auto f = [&](size_t n) { o.take<float>(n, ok); };
    // `carve`'s order, buffer for buffer: emb, R, xn, grs, xn16, lo, lo16, gated, inj, mixed, mixed_bf, mixed_h,
    // bo aliases emb. This counted `xn` unconditionally (carve takes it only under STRATA_GR_UNFUSED) and never counted
    // `grs`.  Net over-count T*(D-HC)*4 bytes: 42 MB at a 1024-token chunk, 252 MB (48 Q8_0 slots) at 6144 - the
    // prompt path was told it had less room than it did.  Safe - the direction is over-estimating, and `take`
    // still bounds-checks - but it under-sizes every loan, so every chunk the scan picks is one step smaller.
    if (ring_bytes_on()) {
        f(T * N); f(T * D);
        if (gr_unfused()) f(T * D);
        f(T * HC);
    } else {   // STRATA_RING_BYTES=0: 0.1.39's count
        f(T * N); f(T * D); f(T * D);
    }
    o.take<uint16_t>(T * (D + (hc_pad() ? XN_PAD : 0)), ok); f(T * LR); o.take<uint16_t>(T * LR, ok);
    f(T * D); f(T * HC); f(T * N); o.take<uint16_t>(T * N, ok); o.take<uint16_t>(T * N, ok);
    if (!emb_reuse_account()) f(T * N);   // bo: aliases emb in carve; still counted unless STRATA_EMB_REUSE_ACCOUNT=1
    const bool f16_io = prompt_f16();   // the current device's mode (the stage's), as Prefill::init will decide it
    if (bf16x2_hc(f16_io)) { o.take<uint16_t>(T * D, ok); o.take<uint16_t>(T * LR, ok); }
    if (bf16x2(f16_io)) o.take<uint16_t>(T * N, ok);
    o.take<int32_t>(T * strata::kernels::kStepCount, ok);
    strata::kernels::QsaShapes s = core::qsa_shapes(g);
    const int64_t cap = strata::kernels::qsa_selection_width(strata::kernels::kTopkMaxCells, s);
    const int64_t max_blocks = ss.qsa_states[ss.qsa_primary()].max_cells / s.idx_block + 2;
    o.take<uint8_t>((size_t) std::max({gdn_set_bytes(T), qsa_set_bytes(T, cap, max_blocks, 256, 32, s),
                                       moe_set_bytes(T, g.n_expert, fused_layout(T, src))}), ok);
    for (int i = 0; i < DQ; ++i) { o.take<uint16_t>(1280 * 2560, ok); o.take<uint16_t>(2560 * 640, ok); }
    if (mmq_plan().any) {
        const MmqPlan& mp = mmq_plan();
        o.take<int32_t>(T * K, ok);
        o.take<int32_t>((size_t) (2 * (g.n_expert + g.n_expert / MMQ_GROUP + 2)), ok);
        o.take<uint8_t>(MMQ_GROUP * mp.gu_max + MMQ_TAIL, ok);
        o.take<uint8_t>(MMQ_GROUP * mp.d_max + MMQ_TAIL, ok);
    }
    if (owned_pages) {
        if (ring_slots(T) > 0) o.take<uint8_t>((size_t) ring_slots(T) * (size_t) MAXBLOB(), ok);   // one allocation
    } else {
        for (int i = 0; i < ring_slots(T); ++i) o.take<uint8_t>((size_t) MAXBLOB(), ok);
    }
    f(T * N);
    f((size_t) strata::kernels::NG_HC_DIM);
    strata::kernels::KvHostPools stage;
    take_stage(o, ss, s, stage, ok);
    return o.used + (8u << 20);   // alignment slack
}

uint64_t Prefill::bytes_needed_no_ring(const core::ModelGeometry& g, const core::SessionState& ss, int64_t chunk,
                                        bool src) {
    return bytes_needed(g, ss, chunk, src) - (uint64_t) ring_slots((size_t) chunk) * (uint64_t) MAXBLOB();
}

int64_t Prefill::ring_default_slots() {
    const int r = g_pinned_share >= 0.9 ? (fused_ring() ? 1024 : 384) : 96;
    return (int64_t) std::min(r, ring_cap());
}

int64_t Prefill::ring_cap_for(int64_t old_chunk) {
    // 0.1.39b (#583, measured on the RTX 5070): giving up ring slots for a bigger chunk paid where 0.1.39's ring held
    // the chunk at 4096 or less (IQ3_XXS 32K prompts +14% to +26%: 4096/384 -> 6656/56) and lost where 0.1.39 already
    // read 6144-token chunks (the Coder: 6144/384 -> 7936/199, 32K -12%).  From kKeepRingChunk on the ring keeps its
    // 0.1.39 size and the scan only looks for a bigger chunk next to it (IQ3_XXS unpinned 6144/96 -> 6912/96: +9%).
    constexpr int64_t kKeepRingChunk = 6144;
    return old_chunk >= kKeepRingChunk ? ring_default_slots() : ring_max_slots();
}

// the unpinned arm keeps its measured 96 (the PR's rule; the byte budget would have been ~49 slots on IQ3_S)
int64_t Prefill::ring_max_slots() {
    return g_pinned_share >= 0.9 ? (int64_t) ring_budget_slots() : (int64_t) std::min(96, ring_cap());
}

int64_t Prefill::ring_slots_for(int64_t chunk) { return ring_slots((size_t) chunk); }
void Prefill::set_cpu_pool(kernels::cpu::ExpertPool* pool) { cpu_pool_ = pool; }
void Prefill::arm_cpu_share(bool applies, bool by_default) {
#if !defined(STRATA_USE_HIP) && !defined(STRATA_HIP_GFX906)
    if (applies && by_default && cpu_share_explicit() == -2.0 && !g_share_default) {
        g_share_default = true;
        std::fprintf(stderr, "prefill: the CPU share is ON by default for prompt chunks below 1024 tokens (the idle CPU "
                             "takes some of the experts the GPU would stream; answers can differ slightly from 0.1.40.3, "
                             "mean KL ~0.004). STRATA_PREFILL_CPU_SHARE=0 turns it off.\n");
    }
#else
    (void) by_default;
#endif
    if (applies && cpu_share_on()) g_stream_min_share = cpu_share_max();
}

bool Prefill::ring_bytes_enabled() { return ring_bytes_on(); }

=======
>>>>>>> their fork (github.com-eddoursul-Strata-custom)
namespace {

const core::WeightRef* need(const core::LayerView& v, const char* suffix, std::string& err) {
    const core::WeightRef* r = v.get(suffix);
    if (!r) err = v.name(suffix) + " is missing";
    return r;
}
bool bf16_proj(Gemm& gm, const core::WeightRef* w, const uint16_t* X, float* Y, int64_t T, const std::string& name,
               std::string& err, int64_t ldy = 0) {
    if (w->kind != core::WeightKind::Bf16InF32 || !w->data) { err = "prefill: " + name + " is not a resident BF16 tensor"; return false; }
    gm.bf16(X, (const uint16_t*) w->data, Y, T, w->ne1 > 0 ? w->ne1 : 1, w->ne0, ldy);
    return true;
}

}  // namespace

bool Prefill::run(const int64_t* tokens, int64_t n, int64_t pos0, std::string& err) {
    Impl& m = *impl_;
    const Bufs& b = m.b;
    const core::ModelGeometry& g = *m.g;
    core::SessionState& ss = *m.ss;
    const strata::kernels::QsaShapes& s = m.s;
    constexpr int64_t SC = strata::kernels::kStepCount;
    if (m.T <= 0) { err = "prefill: no buffers bound"; return false; }
    const auto t_start = Clock::now();
<<<<<<< this fork
    const int64_t LB = stage_lb_, LE = stage_le_;
    // The direct successor's future lives on the Prefill object. Intermediate
    // stages therefore do not drain the complete remaining GPU chain here.
    double host_sync_ms = 0, host_chunk_ms = 0, host_setup_ms = 0;   // STRATA_PREFILL_TIMING: the host's share
    strata::kernels::QsaShapes s = core::qsa_shapes(g);
    // read once (STRATA_KV_PREFETCH=1/0: the next streamed layer's KV prefix is staged on its own stream)
    static const bool kv_prefetch_on = [] {
        const char* e = std::getenv("STRATA_KV_PREFETCH");
        return e ? std::atoi(e) != 0 : false;
    }();
    const bool kv_prefetch = kv_prefetch_on;
    if (kv_prefetch) {
        if ((!m.kv_copy && cudaStreamCreateWithFlags(&m.kv_copy, cudaStreamNonBlocking) != cudaSuccess) ||
            (!m.kv_released && cudaEventCreateWithFlags(&m.kv_released, cudaEventDisableTiming) != cudaSuccess) ||
            (!m.kv_ready && cudaEventCreateWithFlags(&m.kv_ready, cudaEventDisableTiming) != cudaSuccess)) {
            err = "prefill: KV prefetch stream/events";
            return false;
        }
    }
    // An early return must drain DMA before the caller refills the borrowed expert slots.
    struct KvDrain {
        cudaStream_t stream;
        ~KvDrain() { if (stream) cudaStreamSynchronize(stream); }
    } kv_drain{kv_prefetch ? m.kv_copy : nullptr};
    int64_t kv_prefetches = 0;
=======
>>>>>>> their fork (github.com-eddoursul-Strata-custom)
    const uint64_t gdn_floats = (uint64_t) g.ssm_state_size * g.ssm_v_heads * g.ssm_state_size +
                                (uint64_t) g.ssm_conv_channels * (g.ssm_d_conv - 1);
    int32_t prev[2] = {ss.ple_prev[0], ss.ple_prev[1]};
    // Y[T, rows] = X . W^T for a quantized W [rows, cols]: dequantized at its first use in the layer
    auto w16 = [&](int type, const void* blocks, int64_t rows, int64_t cols) -> const uint16_t* {
        for (const auto& [k, p] : m.w16_have)
            if (k == blocks) return p;
        if (m.w16_used + rows * cols > W16) return nullptr;
        uint16_t* p = b.w16 + m.w16_used;
        strata::kernels::dequant_f16(type, blocks, 0, rows, cols, p, m.cs);
        m.w16_used += (rows * cols + 127) / 128 * 128;
        m.w16_have.emplace_back(blocks, p);
        return p;
    };
    auto native_proj = [&](const core::WeightRef* w, const uint16_t* X, float* Y, int64_t T, const std::string& name,
                           std::string& e) {
        const uint16_t* W = w->native_data ? w16(w->native_type, w->native_data, w->ne1, w->ne0) : nullptr;
        if (W == nullptr) {
            e = "prefill: " + name + (w->native_data ? " does not fit the dequantization buffer"
                                                     : " has no native GGUF blocks (run with --native)");
            return false;
        }
        m.gemm.f16(X, W, Y, T, w->ne1, w->ne0);
        return true;
    };
    auto new_layer = [&] {
        m.w16_have.clear();
        m.w16_used = 0;
    };
    // --prefill-profile: an event at the start of each section; the time to the next event is charged to it
    std::vector<std::pair<int, cudaEvent_t>> ev;
    auto mark = [&](int section) {
        if (!profile) return;
        cudaEvent_t e;
        cudaEventCreate(&e);
        cudaEventRecord(e, m.cs);
        ev.push_back({section, e});
    };

    // The MTP draft layer's K/V over a chunk's cells: its input norms and projections (e = fc_embedding(rms(the
    // embedding of the token at t + 1)), h = fc_hidden(rms(R_t)) per stream, h + e on every stream), the attention
    // hyper-connection read, then K (normed and rotated) and V into its cells, from the window's first cell on.  The
    // chunk's buffers are free once its last layer is done; its residual rows are normed in place.
    const core::MtpPromptKv dk = draft != nullptr ? draft->prompt_kv() : core::MtpPromptKv{};
    if (draft != nullptr && (!dk.norm_emb || !dk.norm_hidden || !dk.fc_emb || !dk.fc_hidden || !dk.hc_norm ||
                             !dk.hc_down || !dk.hc_up || !dk.k_proj || !dk.v_proj || !dk.k_norm)) {
        err = "prefill: a draft layer weight is missing";
        return false;
    }
    auto draft_pass = [&](const int64_t* toks, int64_t T, int64_t p0) -> bool {
        const int64_t from = std::clamp<int64_t>(dk.first_cell - p0, 0, T);
        if (from == T) return true;
        mark(kPsDraft);
        new_layer();
        const uint16_t *We = w16(GGML_Q8_0, dk.fc_emb, N, N), *Wh = w16(GGML_Q8_0, dk.fc_hidden, N, N),
                       *Wk = w16(GGML_Q8_0, dk.k_proj, 512, N), *Wv = w16(GGML_Q8_0, dk.v_proj, 512, N);
        if (!We || !Wh || !Wk || !Wv) {
            err = "prefill: the draft layer's projections do not fit the dequantization buffer";
            return false;
        }
        for (int64_t t = from; t < T; ++t) m.ids_host[t - from] = (int32_t) toks[t + 1];
        cudaMemcpyAsync(b.ids, m.ids_host, (size_t) (T - from) * 4, cudaMemcpyHostToDevice, m.cs);
        const core::QsaState& st = *dk.kv;
        for (int64_t t0 = from; t0 < T; t0 += SUB) {
            const int64_t P = std::min(SUB, T - t0), pp = p0 + t0;
            float* R = b.R + t0 * D;
            if (!core::embed_rows(*m.wt, g, b.ids + (t0 - from), P, b.emb, m.cs, err)) return false;
            rms_rows(b.emb, dk.norm_emb, P, N, N, EPS, m.cs);
            to_f16(b.emb, b.mixed_h, P * N, m.cs);
            m.gemm.f16(b.mixed_h, We, b.bo, P, N, N);
            rms_rows(R, dk.norm_hidden, P, D, D, EPS, m.cs);
            to_f16(R, b.xn16, P * D, m.cs);
            m.gemm.f16(b.xn16, Wh, b.gated, P * HC, N, N);
            strata::kernels::add_streams_broadcast(b.gated, b.bo, b.gated, N, (int) HC, (int) P, m.cs);
            gr_norm(b.gated, dk.hc_norm, EPS, b.xn, b.xn16, P, m.cs);
            m.gemm.bf16(b.xn16, dk.hc_down, b.lo, P, LR, D);
            gr_silu(b.lo, b.lo16, P, m.cs);
            m.gemm.bf16(b.lo16, dk.hc_up, b.gated, P, D, LR);
            gr_mix(b.xn, b.gated, nullptr, nullptr, P, m.cs, b.mixed_h);
            m.gemm.f16(b.mixed_h, Wk, b.Kc, P, 512, N);
            m.gemm.f16(b.mixed_h, Wv, b.Vc, P, 512, N);
            rms_rows(b.Kc, dk.k_norm, P * 2, 256, 256, EPS, m.cs);
            rope(b.Kc, P, 2, 256, 512, pp, (float) strata::kernels::qsa_freq_base(), m.cs);
            append_kv(st, b.Kc, b.Vc, P, pp, s, m.cs);
        }
        return true;
    };

    // ---- a chunk's PLE rows (one batched SSD request) on a thread: the first chunk's while its layer 0 runs, each
    // next chunk's while the one before runs; waited for before the PLE block at layer 1
    struct PleRead {
        std::thread th;
        bool ok = true;
        std::string err;
        bool wait(std::string& e) {
            if (th.joinable()) th.join();
            if (!ok) e = err;
            return ok;
        }
        ~PleRead() { if (th.joinable()) th.join(); }
    } ple_read;
    int32_t pa[2] = {prev[0], prev[1]};   // the rows' tokens before the next chunk to read
    int ple_cur = 0;
    auto ple_start = [&](int64_t c, int buf) {
        const int64_t Tc = std::min(m.T, n - c);
        uint32_t* rows = m.ple_rows[buf].data();
        for (int64_t t = 0; t < Tc; ++t) {
            const int32_t tok = (int32_t) tokens[c + t];
            strata::kernels::ngram_rows(&tok, pa, 1, ss.ple.consts, rows + t * strata::kernels::PLE_N_HEADS);
            pa[0] = pa[1];
            pa[1] = tok;
        }
        ple_read.ok = true;
        ple_read.th = std::thread([&m, &ss, &ple_read, rows, Tc, out = m.ple_host[buf]] {
            ple_read.ok = ss.ple.table->gather_batch(rows, (size_t) Tc, out, ple_read.err);
        });
    };
    if (ss.ple.ready() && n > 0) ple_start(0, 0);

    static const bool trace = std::getenv("STRATA_TRACE") != nullptr;
    processed_ = 0;
    int64_t c0 = 0;
    for (; c0 < n; c0 += m.T) {
        if (should_stop && should_stop()) break;   // the chunks before stay: `prev` and the state hold them
        if (trace) { std::fprintf(stderr, "strata trace: prompt chunk %lld of %lld\n", (long long) c0, (long long) n); std::fflush(stderr); }
        const int64_t T = std::min(m.T, n - c0), p0 = pos0 + c0;
        ++stats_.chunks;
        // ---- the embeddings, broadcast to the four streams
        for (int64_t t0 = 0; t0 < T; t0 += SUB) {
            const int64_t P = std::min(SUB, T - t0);
            for (int64_t t = t0; t < t0 + P; ++t) {
                const float* row = embd_rows ? embd_rows[p0 + t] : nullptr;
                if (row) {
                    if (cudaMemcpyAsync(b.emb + (t - t0) * N, row, (size_t) N * 4, cudaMemcpyHostToDevice, m.cs) != cudaSuccess) {
                        err = "prefill: the image embedding upload failed";
                        return false;
                    }
                } else if (!core::embed_row(*m.wt, g, tokens[c0 + t], b.emb + (t - t0) * N, m.cs, err)) {
                    return false;
                }
            }
            gr_broadcast(b.emb, b.R + t0 * D, P, m.cs);
        }
        const bool ple_on = ss.ple.ready();
        // ---- the QSA step records of every position in the chunk
        for (int64_t t = 0; t < T; ++t) strata::kernels::qsa_step_fill(m.steps_host.data() + t * SC, p0 + t, s);
        cudaMemcpyAsync(b.steps, m.steps_host.data(), (size_t) T * SC * 4, cudaMemcpyHostToDevice, m.cs);

        // ---- a layer's FFN write, deferred to the sub-chunk that reads its rows next: the second GPU's sums come back
        // a sub-chunk at a time while this GPU works on the sub-chunks before (with `norm`, the next read's norm in
        // the same pass)
        bool pending = false;
        auto ffn_write = [&](int64_t t0, int64_t P, const float* norm) {
            const uint16_t* part = nullptr;
            if (m.offload) {
                cudaStreamWaitEvent(m.cs, m.ev_piece[(size_t) (t0 / SUB)], 0);
                part = b.sums + t0 * N;
            }
            float *R = b.R + t0 * D, *bo = b.bo_moe + t0 * N, *inj = b.inj_f + t0 * HC;
            if (norm) gr_write_norm_rs(R, bo, inj, HC, norm, EPS, b.rs, b.xn16, P, m.cs, part);
            else gr_write(R, bo, inj, HC, P, m.cs, part);
        };
        int64_t qsa_index = 0, gdn_index = 0;
        for (int64_t l = 0; l < g.n_layers; ++l) {
            const core::LayerView v(*m.wt, l);
            const bool qsa = core::is_qsa_layer(g, l);
            new_layer();
            // a long chunk uses nearly every expert: the ones this GPU's cache lacks are copied ahead, here while
            // the dense steps run, or on the second GPU
            if (T >= ExpertRunner::kPrefetchMin) {
                m.ex1.clear();
                for (int32_t e = 0; e < NE; ++e)
                    if (!(m.host_res && m.cache && m.host_res[(size_t) l * NE + e] >= 0)) m.ex1.push_back(e);
                if (!(m.offload ? m.offload : &m.local)->prefetch(l, m.ex1, err)) return false;
            }
            mark(kPsPle);
            // ---- the PLE block at layer 1: the key and value projections, then the decode path's per-token
            // arithmetic (the conv reads the previous tokens' normalized rows)
            if (l == 1 && ple_on) {
                // the chunk's rows, then the next chunk's read on the thread (the other buffer)
                const auto tp = Clock::now();
                if (!ple_read.wait(err)) return false;
                cudaMemcpyAsync(b.ple_emb, m.ple_host[ple_cur], (size_t) T * N * 4, cudaMemcpyHostToDevice, m.cs);
                if (c0 + T < n) ple_start(c0 + T, 1 - ple_cur);
                ple_cur = 1 - ple_cur;
                stats_.ms_ple += ms_since(tp);
                const strata::kernels::PleWeights& pw = ss.ple.w;
                for (int64_t t0 = 0; t0 < T; t0 += SUB) {
                    const int64_t P = std::min(SUB, T - t0);
                    if (pending) ffn_write(t0, P, nullptr);
                    const float* ple_emb = b.ple_emb + t0 * N;
                    to_bf16(ple_emb, b.mixed_bf, P * N, m.cs);
                    if (pw.key_bf16 != nullptr) {
                        m.gemm.bf16(b.mixed_bf, pw.key_bf16, b.ple_kp, P, D, N);
                    } else if (const uint16_t* W = pw.key_native_data ? w16(pw.key_native_type, pw.key_native_data, D, N)
                                                                      : nullptr) {
                        to_f16(ple_emb, b.mixed_h, P * N, m.cs);
                        m.gemm.f16(b.mixed_h, W, b.ple_kp, P, D, N);
                    } else {
                        err = "prefill: the PLE key has neither a BF16 nor a native GGUF form (run with --native)";
                        return false;
                    }
                    m.gemm.bf16(b.mixed_bf, pw.value_bf16, b.bo, P, N, N);
                    try {
                        strata::kernels::native_ple_postops_tokens(
                            b.ple_kp, b.R + t0 * D, b.bo, ss.ple.hist, pw,
                            {b.ple_key, b.ple_q, b.inj_a, b.ple_gated, b.ple_q, b.R + t0 * D}, (int) P, m.cs);
                    } catch (const std::exception& e) { err = std::string("prefill PLE: ") + e.what(); return false; }
                }
                pending = false;
                new_layer();
            }
<<<<<<< this fork
            for (int half = 0; half < 2; ++half) {
                // ---- the hyper-connection read of this half
                const char* pre = half == 0 ? "hc_attn_" : "hc_ffn_";
                const std::string sn = std::string(pre) + "norm.weight", sd = std::string(pre) + "down.weight",
                                  su = std::string(pre) + "up.weight", si = std::string(pre) + "inject.weight";
                const core::WeightRef *wn = need(v, sn.c_str(), err), *wd = need(v, sd.c_str(), err),
                                      *wu = need(v, su.c_str(), err), *wi = need(v, si.c_str(), err);
                if (!wn || !wd || !wu || !wi) return false;
                // STRATA_HC_Q8 without a bf16 copy: this path's GEMMs read bf16, so the matrices are dequantized per half
                core::WeightRef wd_q8, wu_q8;
                if (!wd->data && wd->hc_q8 && !wu->data && wu->hc_q8) {
                    wd_q8 = *wd; wu_q8 = *wu;
                    wd_q8.data = core::NativeDense::hc_bf16_view(*wd, m.cs, 0, 0);
                    wu_q8.data = core::NativeDense::hc_bf16_view(*wu, m.cs, 1, 0);
                    if (!wd_q8.data || !wu_q8.data) { err = "prefill: the Q8_0 hyper-connection projections could not be dequantized"; return false; }
                    wd = &wd_q8; wu = &wu_q8;
                }
                pt.mark(kPfHc, cs);
                // the xn16 token stride of this chunk (the writers of both halves and of the fused write-backs use it)
                const int64_t ldx = hc_pad() && T >= std::max<int64_t>(pf_switch_min_t(), 64) && !gr_unfused() &&
                                            !m.xn16_lo ? D + XN_PAD : D;
                if (gr_unfused()) gr_norm(m.R, (const float*) wn->data, EPS, m.xn, m.xn16, T, m.cs, m.xn16_lo);
                else if (!normed) gr_norm_rs(m.R, (const float*) wn->data, EPS, m.grs, m.xn16, T, m.cs, m.xn16_lo, ldx);
                normed = false;
                bool hcd = false;
                bool hdown = false;
                if (ldx != D && !pf_hcdown()) {   // STRATA_HCD_EXACT: the exact-order down kernel when hipBLASLt would take 1176 / 1177
                    hdown = wd->kind == core::WeightKind::Bf16InF32 && wd->data && wd->ne0 == D && wd->ne1 == LR &&
                            m.gemm.bf16_hcd_exact(m.xn16, ldx, (const uint16_t*) wd->data, m.lo, T, LR, D);
                } else if (ldx != D) {
                    if (wd->kind != core::WeightKind::Bf16InF32 || !wd->data || wd->ne0 != D || wd->ne1 != LR ||
                        wi->kind != core::WeightKind::Bf16InF32 || !wi->data || wi->ne0 != D || wi->ne1 != HC ||
                        !strata_pf_hcdown_bf16(m.xn16, ldx, (const uint16_t*) wd->data, (const uint16_t*) wi->data, LR,
                                               HC, m.lo, m.inj, T, D, m.cs)) {
                        err = "prefill: STRATA_PF_HCDOWN could not run the hyper-connection projections";
                        return false;
                    }
                    hcd = true;
                }
                if (!hcd && !hdown && !bf16_proj(m.gemm, wd, m.xn16, m.lo, T, sd, err, 0, m.xn16_lo, ldx != D ? ldx : 0)) return false;
                gr_silu(m.lo, m.lo16, T, m.cs, m.lo16_lo);
                bool upmixed = false;
                if (hc_upmix() && T >= pf_switch_min_t() && !gr_unfused() && !m.lo16_lo && !m.mixed_bf_lo &&
                    wu->kind == core::WeightKind::Bf16InF32 && wu->data && wu->ne0 == LR && wu->ne1 == D) {
                    static int checks = [] { const char* e = std::getenv("STRATA_HC_UPMIX_CHECK"); return e ? std::atoi(e) : 0; }();
                    if (checks > 0) {   // the default pair first, kept for the comparison
                        --checks;
                        if (!bf16_proj(m.gemm, wu, m.lo16, m.gated, T, su, err)) return false;
                        gr_mix_r(m.R, m.grs, (const float*) wn->data, m.gated, m.mixed, m.mixed_bf, T, m.cs, m.mixed_h);
                        std::vector<float> ref((size_t) T * N), got((size_t) T * N);
                        cudaMemcpyAsync(ref.data(), m.mixed, ref.size() * 4, cudaMemcpyDeviceToHost, m.cs);
                        upmixed = gr_upmix(m.lo16, (const uint16_t*) wu->data, m.R, m.grs, (const float*) wn->data,
                                           m.mixed, m.mixed_bf, m.mixed_h, T, m.cs);
                        cudaMemcpyAsync(got.data(), m.mixed, got.size() * 4, cudaMemcpyDeviceToHost, m.cs);
                        cudaStreamSynchronize(m.cs);
                        double e2 = 0, r2 = 0, emax = 0;
                        for (size_t i = 0; i < ref.size(); ++i) {
                            const double dd = (double) got[i] - ref[i];
                            e2 += dd * dd; r2 += (double) ref[i] * ref[i]; emax = std::max(emax, std::fabs(dd));
                        }
                        std::fprintf(stderr, "strata: STRATA_HC_UPMIX_CHECK layer %lld half %d T %lld: mixed rel RMS %.3e, "
                                     "max |diff| %.3e (%s)\n", (long long) l, half, (long long) T,
                                     std::sqrt(e2 / std::max(r2, 1e-30)), emax, upmixed ? "upmix" : "upmix refused");
                    } else {
                        upmixed = gr_upmix(m.lo16, (const uint16_t*) wu->data, m.R, m.grs, (const float*) wn->data,
                                           m.mixed, m.mixed_bf, m.mixed_h, T, m.cs);
                    }
                }
                if (!upmixed && !bf16_proj(m.gemm, wu, m.lo16, m.gated, T, su, err, 0, m.lo16_lo)) return false;
                if (!hcd && !bf16_proj(m.gemm, wi, m.xn16, m.inj, T, si, err, 0, m.xn16_lo, ldx != D ? ldx : 0)) return false;
                if (upmixed) {
                } else if (gr_unfused()) {
                    gr_mix(m.xn, m.gated, m.mixed, m.mixed_bf, T, m.cs, m.mixed_h, m.mixed_bf_lo);
                } else {
                    gr_mix_r(m.R, m.grs, (const float*) wn->data, m.gated, m.mixed, m.mixed_bf, T, m.cs, m.mixed_h,
                             m.mixed_bf_lo);
                }

                if (wd_q8.data) {   // the HC GEMMs that read the scratch are enqueued: the next user waits for them
                    core::NativeDense::hc_bf16_release(m.cs, 0, 0);
                    core::NativeDense::hc_bf16_release(m.cs, 1, 0);
                }
                if (half == 0 && !core::is_qsa_layer(g, l)) {
                    // ======================= GDN =======================
                    const core::WeightRef *wqkv = need(v, "attn_qkv.weight", err), *wg = need(v, "attn_gate.weight", err),
                                          *wo = need(v, "ssm_out.weight", err), *wa = need(v, "ssm_alpha.weight", err),
                                          *wb = need(v, "ssm_beta.weight", err), *wc = need(v, "ssm_conv1d.weight", err),
                                          *wnm = need(v, "ssm_norm.weight", err), *wdt = need(v, "ssm_dt.bias", err),
                                          *wsa = need(v, "ssm_a", err);
                    if (!wqkv || !wg || !wo || !wa || !wb || !wc || !wnm || !wdt || !wsa) return false;
                    pt.mark(kPfGdn, cs);
                    float* state = ss.gdn_state + (size_t) (gdn_index - ss.gdn_ord0) * gdn_floats;
                    float* conv = state + (uint64_t) g.ssm_state_size * g.ssm_v_heads * g.ssm_state_size;
                    if (!native_proj(m.gemm, wqkv, m.mixed_h, m.qkv, T, v.name("attn_qkv.weight"), err)) return false;
                    if (!native_proj(m.gemm, wg, m.mixed_h, m.z, T, v.name("attn_gate.weight"), err)) return false;
                    if (!bf16_proj(m.gemm, wa, m.mixed_bf, m.ab, T, v.name("ssm_alpha.weight"), err, 2 * HV, m.mixed_bf_lo)) return false;
                    if (!bf16_proj(m.gemm, wb, m.mixed_bf, m.ab + HV, T, v.name("ssm_beta.weight"), err, 2 * HV, m.mixed_bf_lo)) return false;
                    pt.mark(kPfGdnConv, cs);   // "gdn" is the projections in; the rest on their own lines
                    gdn_gates(m.ab, (const float*) wdt->data, (const float*) wsa->data, m.gate, m.beta, T, m.cs);
                    gdn_conv(conv, m.qkv, (const float*) wc->data, m.hbuf, T, EPS, m.cs);
                    pt.mark(kPfGdnRec, cs);
                    const int64_t ld_y = pf_pad() && T >= std::max<int64_t>(pf_switch_min_t(), 64) ? ZV + ZV_PAD : 0;
                    gdn_recurrence(state, m.hbuf, m.gate, m.beta, m.z, (const float*) wnm->data, EPS, m.y, m.y_h, T, m.cs,
                                   ld_y);
                    pt.mark(kPfGdnOut, cs);
                    if (!native_proj(m.gemm, wo, m.y_h, m.bo, T, v.name("ssm_out.weight"), err, 0, ld_y)) return false;
                    ++gdn_index;
                } else if (half == 0) {
                    // ======================= QSA =======================
                    const core::QsaState& st = ss.qsa_states[qsa_index];
                    const core::WeightRef *wq = need(v, "attn_q.weight", err), *wk = need(v, "attn_k.weight", err),
                                          *wv = need(v, "attn_v.weight", err), *wo = need(v, "attn_output.weight", err),
                                          *wik = need(v, "indexer.k_proj.weight", err),
                                          *wiq = need(v, "indexer.q_proj.weight", err),
                                          *wqn = need(v, "attn_q_norm.weight", err), *wkn = need(v, "attn_k_norm.weight", err),
                                          *wiqn = need(v, "indexer.q_norm.weight", err),
                                          *wikn = need(v, "indexer.k_norm.weight", err);
                    if (!wq || !wk || !wv || !wo || !wik || !wiq || !wqn || !wkn || !wiqn || !wikn) return false;
                    pt.mark(kPfQsa, cs);
                    if (!native_proj(m.gemm, wk, m.mixed_h, m.Kc, T, v.name("attn_k.weight"), err)) return false;
                    if (!native_proj(m.gemm, wv, m.mixed_h, m.Vc, T, v.name("attn_v.weight"), err)) return false;
                    if (!native_proj(m.gemm, wq, m.mixed_h, m.Qf, T, v.name("attn_q.weight"), err)) return false;
                    if (!bf16_proj(m.gemm, wik, m.mixed_bf, m.idx_raw, T, v.name("indexer.k_proj.weight"), err, 0, m.mixed_bf_lo)) return false;
                    if (!bf16_proj(m.gemm, wiq, m.mixed_bf, m.q_idx, T, v.name("indexer.q_proj.weight"), err, 0, m.mixed_bf_lo)) return false;
                    rms_rows(m.Kc, (const float*) wkn->data, T * 2, 256, 256, EPS, m.cs);
                    rope(m.Kc, T, 2, 256, 512, p0, strata::kernels::rope_scaling(), m.cs);
                    // KV streaming: this layer's cells [0, p0) come in from the host copy to the staging pool, and the
                    // chunk's cells go to the host copy, the staging pool, and the VRAM slots of resident blocks
                    const bool staged = st.kv_mode == 1;
                    if (staged) {
                        pt.mark(kPfKvStage, cs);
                        if (kv_pending == qsa_index) {
                            if (cudaStreamWaitEvent(cs, m.kv_ready, 0) != cudaSuccess) {
                                err = "prefill: waiting for KV prefetch";
                                return false;
                            }
                            kv_pending = -1;
                        } else strata::kernels::kv_stage_from_host(pools_of(m.stage, m.ident_table), st.host,
                                                            core::qsa_kv_format(st),
                                                            (p0 + s.page_size - 1) / s.page_size, s, m.cs);
                        pt.mark(kPfQsa, cs);
                        pf_step("reading the prompt (batched, step sync): the K/V staged from RAM at layer", l);
                    }
                    // #579 #613 (HIP, opt-in A/B, STRATA_KV_HOST_DMA=1): the append writes the staging pool and the
                    // resident slots only, and one DMA copies the chunk's blocks from the staging pool to the host copy
                    // - no kernel writes host memory over PCIe.  The bytes every reader sees are the same (the staged
                    // first block is complete; past the chunk's last cell nothing is read until a later append writes
                    // it).  CUDA: never.
#if defined(STRATA_USE_HIP)
                    static const bool kv_host_dma = [] { const char* e = std::getenv("STRATA_KV_HOST_DMA"); return e && e[0] == '1'; }();
#else
                    constexpr bool kv_host_dma = false;
#endif
                    const bool host_by_dma = staged && kv_host_dma;
                    const strata::kernels::KvHostPools* host_w = host_by_dma ? nullptr : &st.host;
                    if (st.kv_hybrid) {   // K8V4: K INT8 unrotated, V rotated Q4_0 (only V and the output rotate)
                        // streamed: each half writes its part of the host copy and of the staging pool
                        strata::kernels::fwht256_inplace_cuda(m.Vc, T * 2, m.cs);
                        const strata::kernels::KvHostPools hk = strata::kernels::kv_hybrid_k_half(st.host),
                                                           hv = strata::kernels::kv_hybrid_v_half(st.host),
                                                           sk = strata::kernels::kv_hybrid_k_half(m.stage),
                                                           sv = strata::kernels::kv_hybrid_v_half(m.stage);
                        const bool mirror = st.host.present();
                        kv_append(m.Kc, m.Kc, T, p0, st.page_table, s.page_size, nullptr, nullptr,
                                  st.k_q, st.k_q, st.k_scale, st.k_scale, m.cs, mirror ? &hk : nullptr,
                                  staged ? &sk : nullptr);
                        strata::kernels::kv_append_q4(st.v_q4, st.v_q4, st.page_table, p0, T, m.Vc, m.Vc, s, m.cs,
                                                      mirror ? &hv : nullptr, staged ? &sv : nullptr);
                    } else {
                        if (st.kv_rot) {   // rotated K and V (kv_q4.hpp), the queries below too, the output back
                            strata::kernels::fwht256_inplace_cuda(m.Kc, T * 2, m.cs);
                            strata::kernels::fwht256_inplace_cuda(m.Vc, T * 2, m.cs);
                        }
                        if (st.kv_q4)
                            strata::kernels::kv_append_q4(st.k_q4, st.v_q4, st.page_table, p0, T, m.Kc, m.Vc, s, m.cs,
                                                          host_w, staged ? &m.stage : nullptr);
                        else
                            kv_append(m.Kc, m.Vc, T, p0, st.page_table, s.page_size, st.kv_int8 ? nullptr : st.k_pool,
                                      st.kv_int8 ? nullptr : st.v_pool, st.k_q, st.v_q, st.k_scale, st.v_scale, m.cs,
                                      host_w, staged ? &m.stage : nullptr);
                        if (host_by_dma)
                            strata::kernels::kv_unstage_to_host(pools_of(m.stage, m.ident_table), st.host,
                                                                core::qsa_kv_format(st), p0 / s.page_size,
                                                                (p0 + T + s.page_size - 1) / s.page_size, s, m.cs);
                    }
                    if (staged) pf_step("reading the prompt (batched, step sync): the K/V append at layer", l);
                    split_q(m.Qf, m.q, T, m.cs);
                    rms_rows(m.q, (const float*) wqn->data, T * 24, 256, 256, EPS, m.cs);
                    rope(m.q, T, 24, 256, 6144, p0, strata::kernels::rope_scaling(), m.cs);
                    if (st.kv_rot) strata::kernels::fwht256_inplace_cuda(m.q, T * 24, m.cs);
                    rms_rows(m.q_idx, (const float*) wiqn->data, T * 4, 128, 128, EPS, m.cs);
                    rope(m.q_idx, T, 4, 128, 512, p0, strata::kernels::rope_scaling(), m.cs);
                    // the indexer appends, token by token; then scores + selection for many queries at once:
                    // a query reads completed blocks (final once completed) and `dead` for its own tail block
                    const strata::kernels::QsaIndexerBuffers ib{st.idx_tail, st.idx_dead, st.idx_pooled, st.idx_block_pos};
                    pt.mark(kPfQsaIdx, cs);
                    // C-2: the chunk's appends in three launches instead of one per token (the same end state:
                    // the queries below read it only after the whole chunk is appended). STRATA_INDEXER_PER_TOKEN=1: the old
                    try {
                        static const bool per_token = std::getenv("STRATA_INDEXER_PER_TOKEN") != nullptr;
                        if (!per_token) {
                            strata::kernels::native_qsa_indexer_append_batch(m.idx_raw, T, p0, 0, (const float*) wikn->data,
                                                                             EPS, ib, s, st.max_cells,
                                                                             strata::kernels::rope_scaling(), m.cs);
                        }
                        for (int64_t t = 0; per_token && t < T; ++t) {
                            const int32_t* step_t = m.steps_dev + t * strata::kernels::kStepCount;
                            strata::kernels::native_qsa_indexer_append(m.idx_raw + t * 128, step_t + strata::kernels::kStepPos, 0,
                                                                       (const float*) wikn->data, EPS, ib, s, st.max_cells,
                                                                       strata::kernels::rope_scaling(), m.cs);
                        }
                    } catch (const std::exception& e) { err = std::string("prefill indexer: ") + e.what(); return false; }
                    pt.mark(kPfQsaSel, cs);
                    for (int64_t t0 = 0; t0 < T; t0 += m.sel_batch) {
                        const int64_t nb = std::min(m.sel_batch, T - t0);
                        const int32_t* steps0 = m.steps_dev + t0 * strata::kernels::kStepCount;
                        // C-1: the grid reaches the batch's last query's n_bid (they rise with the position)
                        const int64_t active = (int64_t) m.steps_host[(size_t) ((t0 + nb - 1) * strata::kernels::kStepCount +
                                                                                strata::kernels::kStepNBid)] + 1;
                        // the scores on tensor cores (3xTF32: FP32-level, not bitwise), below sm_80 the FP32 tiled kernel with STRATA_SELECT_SIMT=1;
                        // STRATA_SELECT_OLD=1: the warp kernel
                        static const bool old_sel = std::getenv("STRATA_SELECT_OLD") != nullptr;
                        if (old_sel || !strata::kernels::qsa_block_scores_tc(st.idx_pooled, st.idx_dead, m.q_idx + t0 * 512,
                                                                             steps0, nb, m.max_blocks, s, m.sel_scores,
                                                                             m.cs, active))
                            strata::kernels::qsa_block_scores(st.idx_pooled, st.idx_dead, m.q_idx + t0 * 512, steps0, nb,
                                                              m.max_blocks, s, m.sel_scores, m.cs, active);
                        strata::kernels::qsa_block_topk(m.sel_scores, steps0, nb, m.max_blocks, m.cap, s,
                                                        m.sel_ids + t0 * m.cap, m.cs, active);
                    }
                    // STRATA_SEL_OVERLAP (debug, D-1's question): how much do neighbouring queries' selections share?
                    // Per tile of 16 queries: the union of their selected cells against the sum of their widths.
                    if (static const bool ovl = std::getenv("STRATA_SEL_OVERLAP") != nullptr; ovl && qsa_index == 0) {
                        std::vector<int32_t> ids((size_t) (T * m.cap));
                        cudaMemcpyAsync(ids.data(), m.sel_ids, ids.size() * 4, cudaMemcpyDeviceToHost, m.cs);
                        cudaStreamSynchronize(m.cs);
                        double sum_w = 0, sum_u = 0;
                        for (int64_t t0 = 0; t0 + 16 <= T; t0 += 16) {
                            std::vector<int32_t> u;
                            for (int64_t t = t0; t < t0 + 16; ++t) {
                                const int64_t w = m.steps_host[(size_t) (t * strata::kernels::kStepCount + strata::kernels::kStepWidth)];
                                sum_w += (double) w;
                                u.insert(u.end(), ids.begin() + t * m.cap, ids.begin() + t * m.cap + w);
                            }
                            std::sort(u.begin(), u.end());
                            sum_u += (double) (std::unique(u.begin(), u.end()) - u.begin());
                        }
                        std::fprintf(stderr, "strata prefill: selection overlap at %lld: 16-query tiles read %.1f%% of the "
                                             "cells one query at a time does\n", (long long) p0, sum_w > 0 ? 100.0 * sum_u / sum_w : 0.0);
                    }
                    // STRATA_IDX_FP16_CHECK: would FP16 pooled indexer keys select the same cells? (the KV-streaming
                    // design's last question). Every query is selected again from the pooled keys and `dead` rounded
                    // to fp16 (exactly what an fp16 store reads back); the agreement with the fp32 selection is
                    // printed cumulatively after each chunk's last QSA layer. Debug: syncs per layer.
                    if (static const bool f16chk = std::getenv("STRATA_IDX_FP16_CHECK") != nullptr; f16chk) {
                        static float *pooled16 = nullptr, *dead16 = nullptr;
                        static int32_t* ids16 = nullptr;
                        static double shared = 0, cells = 0;
                        static long long queries = 0, same = 0, sel_queries = 0;
                        const int64_t rows = st.idx_pooled_rows;
                        if (pooled16 == nullptr &&
                            (cudaMalloc((void**) &pooled16, (size_t) rows * s.idx_dim * 4) != cudaSuccess ||
                             cudaMalloc((void**) &dead16, (size_t) s.idx_dim * 4) != cudaSuccess ||
                             cudaMalloc((void**) &ids16, (size_t) (m.T * m.cap) * 4) != cudaSuccess)) {
                            err = "STRATA_IDX_FP16_CHECK: no room for its buffers";
                            return false;
                        }
                        round_f16(st.idx_pooled, pooled16, rows * s.idx_dim, m.cs);
                        round_f16(st.idx_dead, dead16, s.idx_dim, m.cs);
                        for (int64_t t0 = 0; t0 < T; t0 += m.sel_batch) {
                            const int64_t nb = std::min(m.sel_batch, T - t0);
                            const int32_t* steps0 = m.steps_dev + t0 * strata::kernels::kStepCount;
                            strata::kernels::qsa_block_scores(pooled16, dead16, m.q_idx + t0 * 512, steps0, nb,
                                                              m.max_blocks, s, m.sel_scores, m.cs);
                            strata::kernels::qsa_block_topk(m.sel_scores, steps0, nb, m.max_blocks, m.cap, s,
                                                            ids16 + t0 * m.cap, m.cs);
                        }
                        std::vector<int32_t> a((size_t) (T * m.cap)), b((size_t) (T * m.cap));
                        cudaMemcpyAsync(a.data(), m.sel_ids, a.size() * 4, cudaMemcpyDeviceToHost, m.cs);
                        cudaMemcpyAsync(b.data(), ids16, b.size() * 4, cudaMemcpyDeviceToHost, m.cs);
                        cudaStreamSynchronize(m.cs);
                        for (int64_t t = 0; t < T; ++t) {
                            const int64_t w = m.steps_host[(size_t) (t * strata::kernels::kStepCount + strata::kernels::kStepWidth)];
                            const int32_t *x = a.data() + t * m.cap, *y = b.data() + t * m.cap;
                            int64_t i = 0, j = 0, c = 0;
                            while (i < w && j < w) {
                                if (x[i] == y[j]) { ++c; ++i; ++j; } else if (x[i] < y[j]) ++i; else ++j;
                            }
                            ++queries;
                            same += c == w;
                            if (p0 + t + 1 > m.cap) { ++sel_queries; shared += (double) c; cells += (double) w; }
                        }
                        if (qsa_index + 1 == g.n_qsa_layers())
                            std::fprintf(stderr, "strata prefill: FP16 indexer keys: %lld of %lld selections identical; "
                                                 "where the selection is sparse, %.4f%% of cells shared (%lld queries)\n",
                                         same, queries, cells > 0 ? 100.0 * shared / cells : 100.0, sel_queries);
                    }
                    // STRATA_QSA_DUMP=<file>: append every QSA layer's selected cells for the prompt's last
                    // STRATA_QSA_DUMP_LAST (4096) positions - records of int32 {qsa layer, pos0, T, cap} + T*cap cells,
                    // for tools/qsa_locality.py (how local the sparse attention's reads are: the KV-streaming question)
                    if (static const char* dump = std::getenv("STRATA_QSA_DUMP"); dump != nullptr) {
                        static const long long last = std::getenv("STRATA_QSA_DUMP_LAST")
                                                          ? std::atoll(std::getenv("STRATA_QSA_DUMP_LAST")) : 4096;
                        if (p0 + T > pos0 + n - last) {
                            std::vector<int32_t> h((size_t) (T * m.cap));
                            cudaMemcpyAsync(h.data(), m.sel_ids, h.size() * 4, cudaMemcpyDeviceToHost, m.cs);
                            cudaStreamSynchronize(m.cs);
                            if (std::FILE* f = std::fopen(dump, "ab")) {
                                const int32_t hdr[4] = {(int32_t) qsa_index, (int32_t) p0, (int32_t) T, (int32_t) m.cap};
                                std::fwrite(hdr, 4, 4, f);
                                std::fwrite(h.data(), 4, h.size(), f);
                                std::fclose(f);
                            }
                        }
                    }
                    const strata::kernels::QsaAttnPools pools = staged ? pools_of(m.stage, m.ident_table)
                                                                       : core::qsa_attn_pools(st);
                    pt.mark(kPfQsaAttn, cs);
                    // perf-review D-1: the whole chunk on tensor cores, one block per (query, KV head), FP32-level
                    // accuracy but not bitwise (qsa_prompt_attn.hpp). Q4_0 KV, or STRATA_PROMPT_ATTN_OLD=1: the
                    // decode kernel, 32 queries at a time (K8V4 runs the tensor kernel's mode 3: INT8 K,
                    // V dequantized from its q4_0 blocks to fp16 at gather)
                    static const bool old_attn = std::getenv("STRATA_PROMPT_ATTN_OLD") != nullptr;
                    if (old_attn || !strata::kernels::qsa_prompt_attn_batch(m.q, pools, m.sel_ids, m.steps_dev, m.cap, s,
                                                                            m.attn, T, m.cs))
                        for (int64_t t0 = 0; t0 < T; t0 += m.attn_batch) {
                            const int64_t nb = std::min(m.attn_batch, T - t0);
                            strata::kernels::qsa_decode_attn_batch(m.q + t0 * ZV, pools, m.sel_ids + t0 * m.cap,
                                                                   m.steps_dev + t0 * strata::kernels::kStepCount, m.cap,
                                                                   s, m.attn_scratch, m.attn + t0 * ZV, nb, m.cs);
                        }
                    if (st.kv_rot || st.kv_hybrid) strata::kernels::fwht256_inplace_cuda(m.attn, T * 24, m.cs);
                    pt.mark(kPfQsa, cs);
                    const int64_t ld_a = pf_pad() && T >= std::max<int64_t>(pf_switch_min_t(), 64) ? ZV + ZV_PAD : 0;
                    gate_attn(m.attn, m.Qf, m.attn_h, T, m.cs, ld_a);
                    if (!native_proj(m.gemm, wo, m.attn_h, m.bo, T, v.name("attn_output.weight"), err, 0, ld_a)) return false;
                    ++qsa_index;
                    if (!kv_prefetch_after(l + 1, qsa_index)) return false;
                } else {
                    // ======================= MoE =======================
                    const core::WeightRef *wr = need(v, "ffn_gate_inp.weight", err),
                                          *wgi = need(v, "ffn_gate_inp_shexp.weight", err),
                                          *wsg = need(v, "ffn_gate_shexp.weight", err),
                                          *wsu = need(v, "ffn_up_shexp.weight", err),
                                          *wsd = need(v, "ffn_down_shexp.weight", err);
                    if (!wr || !wgi || !wsg || !wsu || !wsd) return false;
                    pt.mark(kPfRouter, cs);
                    if (!bf16_proj(m.gemm, wr, m.mixed_bf, m.logits, T, v.name("ffn_gate_inp.weight"), err, 0, m.mixed_bf_lo)) return false;
                    route(m.logits, m.ids, m.w, T, m.g->n_expert, m.cs);
                    // RESEARCH HOOK (141-research, opt-in, never on by default): STRATA_DBG_FEAT=<file> dumps, per MoE layer and chunk,
                    // the routed ids + weights (and for the layers in STRATA_DBG_FEAT_LAYERS the router-input hidden state, bf16);
                    // STRATA_DBG_ROUTE_OVERRIDE=<file> replaces the routing of the layers listed in that file with the file's ids + weights.
                    {
                        static const char* feat_path = std::getenv("STRATA_DBG_FEAT");
                        static const char* ovr_path = std::getenv("STRATA_DBG_ROUTE_OVERRIDE");
                        if (feat_path != nullptr || ovr_path != nullptr) {
                            static std::FILE* ff = feat_path ? std::fopen(feat_path, "wb") : nullptr;
                            static std::vector<bool> featl = [] {
                                std::vector<bool> r(64, false);
                                const char* e = std::getenv("STRATA_DBG_FEAT_LAYERS");
                                std::string s = e ? e : "3,4,15,16,27,28,39,40";
                                size_t i = 0;
                                while (i < s.size()) { size_t j = s.find(',', i); if (j == std::string::npos) j = s.size(); int v = std::atoi(s.substr(i, j - i).c_str()); if (v >= 0 && v < 64) r[(size_t) v] = true; i = j + 1; }
                                return r;
                            }();
                            static int64_t posl[64] = {};
                            struct Ovr { std::vector<int32_t> ids; std::vector<float> w; };
                            static std::map<std::pair<int32_t, int32_t>, Ovr> ovr = [] {
                                std::map<std::pair<int32_t, int32_t>, Ovr> mp;
                                const char* p = std::getenv("STRATA_DBG_ROUTE_OVERRIDE");
                                std::FILE* f = p ? std::fopen(p, "rb") : nullptr;
                                if (f == nullptr) return mp;
                                int32_t h[3];
                                while (std::fread(h, 4, 3, f) == 3) {
                                    Ovr o; o.ids.resize((size_t) h[2] * 10); o.w.resize((size_t) h[2] * 10);
                                    if (std::fread(o.ids.data(), 4, o.ids.size(), f) != o.ids.size()) break;
                                    if (std::fread(o.w.data(), 4, o.w.size(), f) != o.w.size()) break;
                                    mp[{h[0], h[1]}] = std::move(o);
                                }
                                std::fclose(f);
                                std::fprintf(stderr, "route override: %zu records\n", mp.size());
                                return mp;
                            }();
                            const int32_t pos = (int32_t) posl[l & 63];
                            posl[l & 63] += T;
                            cudaStreamSynchronize((cudaStream_t) m.cs);
                            if (ff != nullptr) {
                                std::vector<int32_t> hid((size_t) T * K); std::vector<float> hw((size_t) T * K);
                                cudaMemcpy(hid.data(), m.ids, hid.size() * 4, cudaMemcpyDeviceToHost);
                                cudaMemcpy(hw.data(), m.w, hw.size() * 4, cudaMemcpyDeviceToHost);
                                std::vector<uint16_t> ids16(hid.size()), w16(hw.size());
                                for (size_t i = 0; i < hid.size(); ++i) { ids16[i] = (uint16_t) hid[i]; w16[i] = (uint16_t) (((*(const uint32_t*) &hw[i]) + 0x8000u) >> 16); }
                                const int32_t hd[4] = {(int32_t) l, pos, (int32_t) T, featl[(size_t) (l & 63)] ? 1 : 0};
                                std::fwrite(hd, 4, 4, ff);
                                std::fwrite(ids16.data(), 2, ids16.size(), ff);
                                std::fwrite(w16.data(), 2, w16.size(), ff);   // bf16 weights
                                if (hd[3]) {
                                    std::vector<uint16_t> hx((size_t) T * N);
                                    cudaMemcpy(hx.data(), m.mixed_bf, hx.size() * 2, cudaMemcpyDeviceToHost);
                                    std::fwrite(hx.data(), 2, hx.size(), ff);
                                }
                            }
                            auto it = ovr.find({(int32_t) l, pos});
                            if (it != ovr.end() && (int64_t) it->second.ids.size() == T * K) {
                                cudaMemcpy(m.ids, it->second.ids.data(), (size_t) T * K * 4, cudaMemcpyHostToDevice);
                                cudaMemcpy(m.w, it->second.w.data(), (size_t) T * K * 4, cudaMemcpyHostToDevice);
                            }
                        }
                    }
                    // the shared expert and its scalar gate
                    auto shared_expert = [&]() -> bool {
                        if (!native_proj(m.gemm, wsg, m.mixed_h, m.sgate, T, v.name("ffn_gate_shexp.weight"), err)) return false;
                        if (!native_proj(m.gemm, wsu, m.mixed_h, m.sup, T, v.name("ffn_up_shexp.weight"), err)) return false;
                        swiglu_pair(m.sgate, m.sup, m.sh_h, T, m.cs);
                        if (!native_proj(m.gemm, wsd, m.sh_h, m.shared, T, v.name("ffn_down_shexp.weight"), err)) return false;
                        if (wgi->kind != core::WeightKind::Bf16InF32) { err = "prefill: shared gate is not BF16"; return false; }
                        m.gemm.bf16(m.mixed_bf, (const uint16_t*) wgi->data, m.sg, T, 1, N);
                        if (m.mixed_bf_lo) m.gemm.bf16(m.mixed_bf_lo, (const uint16_t*) wgi->data, m.sg, T, 1, N, 0, 1.0f);
                        return true;
                    };
                    // Only the router needs to finish before readback. Queue the shared expert afterwards
                    // so it can overlap CPU grouping and the routed-only path's initial expert uploads.
                    const bool defer_shared = !stream_all && stream_ahead_enabled();
                    if (!defer_shared && !shared_expert()) return false;
                    // #136: STRATA_PF_FUSED=1 - the Q2_0 pack's experts on the fused int8 kernels (moe_fused.hpp),
                    // grouped on the GPU: no host sync.  Only where every expert's place is known before the routing -
                    // the streamed walk, in which every non-resident expert of the layer comes through the ring in id
                    // order - and where the MMQ buffers exist: they hold the fused path's own (the per-token int8
                    // activations in Xq, the int8 H in H, the grouping tables in GU).  Chunks below stream_all_min()
                    // keep MMQ; without the variable nothing here runs.  A native pack's layer takes the native kernels
                    // (moe_fused_iq.hpp) where they cover its two formats, else MMQ (or the FP16 path: IQ1_M).
                    const strata::kernels::cpu::ExpertLayout& lay = strata::kernels::cpu::expert_layout();
                    const bool use_mmq = mmq_plan().any && mmq_plan().layer[(size_t) l];
                    const int mmq_gt = lay.native ? lay.fmt[(size_t) l].gu_type : 42;
                    const int mmq_dt = lay.native ? lay.fmt[(size_t) l].d_type : 42;
                    // --peer-device: MMQ only, whether or not the peer took the prompt path (set_peer can decline), as
                    // fused_ring() sized the ring and the buffers for it
                    const bool no_peer = !core::peer_portable();
                    const bool fused_only = mmq_plan().any && mmq_plan().fo[(size_t) l];
                    const bool fused_nat = (use_mmq || fused_only) && stream_all && no_peer && lay.native && fused::native_supported(mmq_gt, mmq_dt);
                    const bool fused_l = (use_mmq && stream_all && no_peer && !lay.native && fused::enabled()) || fused_nat;
                    // #583 / #954: the fused layout's GU/H/Xq hold the grouping tables and the int8 rows, sized for
                    // stream_all_min() - 1 tokens of MMQ's rows, so a layer that takes MMQ or the FP16 path at the FULL
                    // chunk would write T*K rows into a (stream_all_min() - 1)*K-row buffer: an illegal access, or a kernel
                    // that never returns (the first ck() to see it is `prefill mmq: iota`, one prompt later).  fused_layout()
                    // and fused_ring() keep the shrink and this decision in step; this is the backstop for anything that
                    // drifts: a clear error, never a hang.
                    // STRATA_DBG_FORCE_SLOW_LAYER=1 (a test of the backstop): layer 0 counts as one the fused path does not take
                    static const bool force_slow = [] { const char* e = std::getenv("STRATA_DBG_FORCE_SLOW_LAYER"); return e && e[0] == '1'; }();
                    if ((!fused_l || (force_slow && l == 0)) && m.fused_bufs && T >= stream_all_min()) {
                        err = "prefill: the fused layout's MoE buffers are too small for layer " + std::to_string(l) +
                              "'s expert path at a chunk of " + std::to_string(T) + " tokens";
                        return false;
=======
            const core::WeightRef *wr = nullptr, *wgi = nullptr, *wsg = nullptr, *wsu = nullptr, *wsd = nullptr;
            if (!(wr = need(v, "ffn_gate_inp.weight", err)) || !(wgi = need(v, "ffn_gate_inp_shexp.weight", err)) ||
                !(wsg = need(v, "ffn_gate_shexp.weight", err)) || !(wsu = need(v, "ffn_up_shexp.weight", err)) ||
                !(wsd = need(v, "ffn_down_shexp.weight", err)))
                return false;
            if (wgi->kind != core::WeightKind::Bf16InF32) { err = "prefill: shared gate is not BF16"; return false; }
            // ================ per sub-chunk: the mixer half, then the FFN half up to the routing ================
            for (int64_t t0 = 0; t0 < T; t0 += SUB) {
                const int64_t P = std::min(SUB, T - t0), pp = p0 + t0;
                uint16_t* mixed_h = b.mixed_h + t0 * N;
                for (int half = 0; half < 2; ++half) {
                    mark(half == 0 ? kPsHcRead : kPsHcFfn);
                    // ---- the hyper-connection read of this half (the FFN half's norm came with the mixer's write)
                    const char* pre = half == 0 ? "hc_attn_" : "hc_ffn_";
                    const std::string sn = std::string(pre) + "norm.weight", sd = std::string(pre) + "down.weight",
                                      su = std::string(pre) + "up.weight", si = std::string(pre) + "inject.weight";
                    const core::WeightRef *wn = need(v, sn.c_str(), err), *wd = need(v, sd.c_str(), err),
                                          *wu = need(v, su.c_str(), err), *wi = need(v, si.c_str(), err);
                    if (!wn || !wd || !wu || !wi) return false;
                    float* inj = half == 0 ? b.inj_a : b.inj_f + t0 * HC;
                    if (half == 0 && pending) {
                        mark(kPsCombine);
                        ffn_write(t0, P, (const float*) wn->data);
                        mark(kPsHcRead);
                    } else if (half == 0) {
                        gr_norm_rs(b.R + t0 * D, (const float*) wn->data, EPS, b.rs, b.xn16, P, m.cs);
>>>>>>> their fork (github.com-eddoursul-Strata-custom)
                    }
                    if (!bf16_proj(m.gemm, wd, b.xn16, b.lo, P, sd, err)) return false;
                    gr_silu(b.lo, b.lo16, P, m.cs);
                    if (!bf16_proj(m.gemm, wu, b.lo16, b.gated, P, su, err)) return false;
                    if (!bf16_proj(m.gemm, wi, b.xn16, inj, P, si, err)) return false;
                    gr_mix_r(b.R + t0 * D, b.rs, (const float*) wn->data, b.gated, nullptr, b.mixed_bf, P, m.cs,
                             mixed_h);
                    if (half == 1) break;

                    if (!qsa) {
                        // ======================= GDN =======================
                        const core::WeightRef *wqkv = need(v, "attn_qkv.weight", err), *wg = need(v, "attn_gate.weight", err),
                                              *wo = need(v, "ssm_out.weight", err), *wa = need(v, "ssm_alpha.weight", err),
                                              *wb = need(v, "ssm_beta.weight", err), *wc = need(v, "ssm_conv1d.weight", err),
                                              *wnm = need(v, "ssm_norm.weight", err), *wdt = need(v, "ssm_dt.bias", err),
                                              *wsa = need(v, "ssm_a", err);
                        if (!wqkv || !wg || !wo || !wa || !wb || !wc || !wnm || !wdt || !wsa) return false;
                        mark(kPsGdn);
                        float* state = ss.gdn_state + (size_t) gdn_index * gdn_floats;
                        float* conv = state + (uint64_t) g.ssm_state_size * g.ssm_v_heads * g.ssm_state_size;
                        if (!native_proj(wqkv, mixed_h, b.qkv, P, v.name("attn_qkv.weight"), err)) return false;
                        if (!native_proj(wg, mixed_h, b.z, P, v.name("attn_gate.weight"), err)) return false;
                        if (!bf16_proj(m.gemm, wa, b.mixed_bf, b.ab, P, v.name("ssm_alpha.weight"), err, 2 * HV)) return false;
                        if (!bf16_proj(m.gemm, wb, b.mixed_bf, b.ab + HV, P, v.name("ssm_beta.weight"), err, 2 * HV)) return false;
                        gdn_gates(b.ab, (const float*) wdt->data, (const float*) wsa->data, b.gate, b.beta, P, m.cs);
                        mark(kPsGdnConv);
                        gdn_conv(conv, b.qkv, (const float*) wc->data, b.hbuf, P, EPS, m.cs);
                        mark(kPsGdnScan);
                        gdn_scan(state, b.hbuf, b.gate, b.beta, b.y, P, m.cs);
                        mark(kPsGdnNorm);
                        gdn_out_norm(b.y, b.z, (const float*) wnm->data, EPS, b.y_h, P, m.cs);
                        mark(kPsGdnOut);
                        if (!native_proj(wo, b.y_h, b.bo, P, v.name("ssm_out.weight"), err)) return false;
                    } else {
<<<<<<< this fork
                        // group the (token, k) pairs by expert on the host
                        pt.mark(kPfHostGroup, cs);
                        // (the sync below also orders this layer's writes of slot/src/bounds after the previous
                        // layer's kernels that read them)
                        const bool grp_mapped = m.grp_host != nullptr;
                        int32_t* ids_h = grp_mapped ? m.grp_host : m.ids_host.data();
                        int32_t* slot_h = grp_mapped ? m.grp_host + m.grp_tk : m.slot_host.data();
                        int32_t* src_h = grp_mapped ? m.grp_host + 2 * m.grp_tk : m.src_host.data();
                        if (grp_mapped) copy_i32(m.grp_dev, m.ids, T * K, m.cs);
                        else if (m.up_ring.d2h_sync(m.ids_host.data(), m.ids, (size_t) T * K * 4, m.cs) != cudaSuccess) {
                            err = "prefill: the routed ids' download failed";   // #1996
                            return false;
                        }
                        bool cpu_maybe = pool_hold.held && !stream_all && m.src != nullptr && !m.cpu_dead &&
                                         lay.native && !m.pp && !lay.fmt.empty();
                        // The share's two pinned buffers are small (~50 MB) but are had mid-request, when the host may have
                        // nothing left (0.1.41 report: tight RAM, chats failing).  A buffer that cannot be had ends the share
                        // for this engine - the GPU takes every expert, as with STRATA_PREFILL_CPU_SHARE=0 - instead of
                        // failing the request.
                        auto cpu_share_off = [&](const char* what) {
                            m.cpu_dead = true;
                            cpu_maybe = false;
                            std::fprintf(stderr, "prefill: the CPU share is off for this run: the host could not give %s "
                                                 "(RAM is short); the GPU streams every expert\n", what);
                        };
                        // auto shares only while the layers that share cost less per non-resident expert than their
                        // neighbours that do not.  #1282 (RX 7900 GRE + 5700X3D): every share lost, auto's 0.47 by
                        // 4.5%, the host's issue time per streamed expert rising 107 -> 200 us with the CPU busy -
                        // a cost g / (c + g) cannot see, as both sides are measured with the share on.  The first
                        // eligible layers alternate, without first, until CPU_MIN_RATIOS ratios are in; then the
                        // cheaper arm, and every CPU_PROBE-th layer the other one (a prime: the probes move across
                        // the layers from one chunk to the next), so both stay measured.
                        if (cpu_maybe && cpu_share_env() < 0.0) {
                            constexpr int64_t CPU_PROBE = 29;
                            constexpr int CPU_MIN_RATIOS = 3;
                            const int64_t i = m.cpu_layers++;
                            cpu_arm = m.cpu_nratio < CPU_MIN_RATIOS ? (i & 1) != 0 : (i % CPU_PROBE == 0) != m.cpu_gate;
                            if (m.cpu_first_l < 0) m.cpu_first_l = l;
                            cpu_cold = l == m.cpu_first_l;
                            if (m.cpu_wall[0] == nullptr)
                                for (cudaEvent_t& e : m.cpu_wall) cudaEventCreate(&e);
                            cpu_set = (int) (i & 1);
                            cudaEventRecord(m.cpu_wall[2 * cpu_set], m.cs);
                        }
                        if (cpu_maybe && cpu_arm) {
                            const size_t want = (size_t) T * N;
                            if (m.cpu_x_n < want) {
                                cpu_cold = true;
                                if (m.cpu_x) cudaFreeHost(m.cpu_x);
                                m.cpu_x = nullptr;
                                m.cpu_x_n = 0;
                                if (cudaHostAlloc((void**) &m.cpu_x, want * sizeof(float), cudaHostAllocDefault) != cudaSuccess) {
                                    (void) cudaGetLastError();
                                    m.cpu_x = nullptr;
                                    cpu_share_off("the activation buffer");
                                } else {
                                    m.cpu_x_n = want;
                                }
                            }
                            if (cpu_maybe) cudaMemcpyAsync(m.cpu_x, m.mixed, want * sizeof(float), cudaMemcpyDeviceToHost, m.cs);
                        }
                        // #579: a stall here is the GPU (this layer's attention and router, or the previous layer's
                        // work), not the host: the watchdog's report says so (only its text changes)
                        core::progress_at("reading the prompt (batched): waiting for the GPU (attention, router) at layer",
                                          l, p0);
                        cudaStreamSynchronize(m.cs);
                        core::progress_at("reading the prompt (batched): layer", l, p0);
                        if (cpu_maybe) {   // the rows buffer, once, at its largest (an expert takes at most MAXT tokens' rows)
                            const size_t cap = (size_t) std::min<int64_t>(T * K, (int64_t) strata::kernels::cpu::MAXT * m.g->n_expert) * N;
                            if (m.cpu_rows_n < cap) {
                                if (m.cpu_rows) cudaFreeHost(m.cpu_rows);
                                m.cpu_rows = nullptr;
                                m.cpu_rows_n = 0;
                                if (cudaHostAlloc((void**) &m.cpu_rows, cap * sizeof(float), cudaHostAllocDefault) != cudaSuccess) {
                                    (void) cudaGetLastError();
                                    m.cpu_rows = nullptr;
                                    cpu_share_off("the rows buffer");
                                } else {
                                    m.cpu_rows_n = cap;
                                }
                            }
                        }
                        if (m.cpu_pend) {   // the measured share: the last CPU-sharing layer's GPU time is final now
                            m.cpu_pend = false;
                            float g_ms = 0;
                            if (cudaEventElapsedTime(&g_ms, m.cpu_ev[0], m.cpu_ev[1]) == cudaSuccess) {
                                constexpr double a = 0.25;
                                const double c = m.pend_cpu_ms / (double) m.pend_n_cpu, g = g_ms / (double) m.pend_n_gpu;
                                m.cpu_c_ms = m.cpu_c_ms > 0 ? m.cpu_c_ms + a * (c - m.cpu_c_ms) : c;
                                m.cpu_g_ms = m.cpu_g_ms > 0 ? m.cpu_g_ms + a * (g - m.cpu_g_ms) : g;
                                m.cpu_share_now = std::clamp(m.cpu_g_ms / (m.cpu_c_ms + m.cpu_g_ms), 0.05, 0.9);
                            }
                        }
                        if (m.pend_wall) {   // the last eligible layer's wall, and its ratio to the one before it
                            float w_ms = 0;
                            if (cudaEventElapsedTime(&w_ms, m.cpu_wall[2 * m.pend_set], m.cpu_wall[2 * m.pend_set + 1]) == cudaSuccess) {
                                const double w = w_ms / (double) m.pend_wall_n;
                                if (m.last_arm != 0 && m.last_arm != m.pend_wall && m.pend_l > m.last_l) {   // same chunk
                                    std::rotate(m.cpu_ratio, m.cpu_ratio + 1, m.cpu_ratio + Impl::CPU_RATIOS);
                                    m.cpu_ratio[Impl::CPU_RATIOS - 1] = m.pend_wall == 2 ? w / m.last_w : m.last_w / w;
                                    m.cpu_nratio = std::min(m.cpu_nratio + 1, Impl::CPU_RATIOS);
                                    double v[Impl::CPU_RATIOS];
                                    std::copy(m.cpu_ratio + Impl::CPU_RATIOS - m.cpu_nratio, m.cpu_ratio + Impl::CPU_RATIOS, v);
                                    std::nth_element(v, v + m.cpu_nratio / 2, v + m.cpu_nratio);   // even: the upper one
                                    m.cpu_gate = v[m.cpu_nratio / 2] < 1.0;
                                }
                                // debug: STRATA_DBG_CPU_GATE=1 prints each reading and what the next layers do
                                if (static const bool dbg = std::getenv("STRATA_DBG_CPU_GATE") != nullptr; dbg)
                                    std::fprintf(stderr, "cpu gate: layer %d %s, %lld non-resident, %.2f ms (%.4f a expert) -> %s\n",
                                                 m.pend_l, m.pend_wall == 2 ? "shared" : "not shared", (long long) m.pend_wall_n,
                                                 w_ms, w, m.cpu_gate ? "share" : "do not share");
                                m.last_arm = m.pend_wall;
                                m.last_l = m.pend_l;
                                m.last_w = w;
                            }
                            m.pend_wall = 0;
                        }
                        pt.fold();
                        if (pe.on) {   // the peer's marks so far are done: the primary waited for its last rows
                            int pd = 0; cudaGetDevice(&pd); cudaSetDevice(pe.dev); cudaStreamSynchronize(m.pp->s); pe.fold(); cudaSetDevice(pd);
                        }
                        if (defer_shared) {
                            pt.mark(kPfRouter, cs);   // keep the existing router+shared timing attribution
                            if (!shared_expert()) return false;
                            pt.mark(kPfHostGroup, cs);
                        }
                        std::fill(m.cnt.begin(), m.cnt.end(), 0);
                        for (int64_t i = 0; i < T * K; ++i) {
                            const int32_t e = ids_h[(size_t) i];
                            if (e < 0 || e >= m.g->n_expert) { err = "prefill: routed id out of range"; return false; }
                            ++m.cnt[(size_t) e];
                        }
                        if (cpu_maybe) {
                            std::vector<std::pair<int32_t, int32_t>> cand;   // (tokens, expert)
                            int64_t nstream = 0;
                            for (int32_t e = 0; e < m.g->n_expert; ++e) {
                                const int32_t c = m.cnt[(size_t) e];
                                if (c == 0 || (m.host_res && m.cache && m.host_res[(size_t) l * m.g->n_expert + e] >= 0))
                                    continue;
                                ++nstream;
                                // a blob the CPU reads from RAM: page-locked, or any other that is not assembled into a
                                // short-lived buffer (the arena's unpinned part, the mapped experts.bin's pages)
                                if (c <= strata::kernels::cpu::MAXT && (m.src->pinned(l, e) || !m.src->transient(l, e)))
                                    cand.emplace_back(c, e);
                            }
                            std::sort(cand.begin(), cand.end());
                            const double share = cpu_share_env() >= 0.0 ? cpu_share_env() : cpu_arm ? m.cpu_share_now : 0.0;
                            const size_t take = std::min(cand.size(), (size_t) std::llround(share * (double) nstream));
                            n_stream = nstream;
                            // The CPU thread touches no ExpertSource (its blob() counts reads without a lock, and its
                            // calls race with this thread's own blob_stable()/pinned() ones): the pointers are taken
                            // here, and an expert without a stable blob stays on the GPU.
                            for (size_t i = 0; i < cand.size() && n_cpu < (int64_t) take; ++i) {
                                const int32_t e = cand[i].second;
                                const uint8_t* b = m.src->blob_stable(l, e);
                                if (b == nullptr) continue;
                                if (on_cpu.empty()) {
                                    on_cpu.assign((size_t) m.g->n_expert, 0);
                                    cpu_blob.assign((size_t) m.g->n_expert, nullptr);
                                }
                                on_cpu[(size_t) e] = 1;
                                cpu_blob[(size_t) e] = b;
                                rows_cpu += cand[i].first;
                                ++n_cpu;
                            }
                        }
                        // multi-GPU: the rows of the experts the peer computes go last, as one block [rows_local, T*K)
                        const bool pre_mmq = mmq_plan().any && mmq_plan().layer[(size_t) l];
                        std::vector<char> on_peer;
                        int64_t rows_local = T * K, rows_peer = 0;
                        if (m.pp && pre_mmq) {
                            on_peer.assign((size_t) m.g->n_expert, 0);
                            for (int32_t e = 0; e < m.g->n_expert; ++e) {
                                const int32_t c = m.cnt[(size_t) e];
                                if (c > 0 && !m.pp->ps_flag.empty() && m.pp->ps_flag[(size_t) l * m.g->n_expert + e]) {   // peer-streamed
                                    on_peer[(size_t) e] = 2;
                                    rows_peer += c;
                                    continue;
                                }
                                if (c == 0 || (m.host_res && m.cache && m.host_res[(size_t) l * m.g->n_expert + e] >= 0) ||
                                    !m.pp->peer || !m.pp->peer->has(l, e))
                                    continue;
                                if (rows_peer + c > m.pp->cap_rows) { ++m.pp->over_cap; continue; }
                                on_peer[(size_t) e] = 1;
                                rows_peer += c;
                            }
                            rows_local = T * K - rows_peer;
                        }
                        {
                            int32_t r = 0;
                            for (int32_t e = 0; e < m.g->n_expert; ++e)
                                if ((on_peer.empty() || !on_peer[(size_t) e]) && (on_cpu.empty() || !on_cpu[(size_t) e])) {
                                    m.off[(size_t) e] = r;
                                    r += m.cnt[(size_t) e];
                                }
                            // the peer's rows in order_peer's order (peer-held, then peer-streamed): its groups need them contiguous
                            for (int kind = 1; kind <= 2 && !on_peer.empty(); ++kind)
                                for (int32_t e = 0; e < m.g->n_expert; ++e)
                                    if (on_peer[(size_t) e] == kind) { m.off[(size_t) e] = r; r += m.cnt[(size_t) e]; }
                            for (int32_t e = 0; e < m.g->n_expert && !on_cpu.empty(); ++e)   // the CPU's block, last
                                if (on_cpu[(size_t) e]) { m.off[(size_t) e] = r; r += m.cnt[(size_t) e]; }
                            m.off[(size_t) m.g->n_expert] = r;
                            rows_local -= rows_cpu;
                        }
                        std::vector<int32_t> fill(m.off.begin(), m.off.end() - 1);
                        for (int64_t i = 0; i < T * K; ++i) {
                            const int32_t e = ids_h[(size_t) i];
                            const int32_t p = fill[(size_t) e]++;
                            slot_h[(size_t) i] = p;
                            src_h[(size_t) p] = (int32_t) (i / K);
                        }
                        // the experts, in id order: resident ones from VRAM, the others through the staging ring
                        std::vector<int32_t> order, order_peer;
                        for (int32_t e = 0; e < m.g->n_expert; ++e)
                            if (m.cnt[(size_t) e] > 0 && (on_peer.empty() || on_peer[(size_t) e] != 2) &&
                                (on_cpu.empty() || !on_cpu[(size_t) e]))
                                (!on_peer.empty() && on_peer[(size_t) e] ? order_peer : order).push_back(e);
                        for (int32_t e = 0; e < m.g->n_expert && !on_peer.empty(); ++e)   // the peer-streamed ones last:
                            if (on_peer[(size_t) e] == 2) order_peer.push_back(e);        // their copies get the most time
                        // Aurora (STRATA_MMQ_RESIDENT_SORT_NE=1, opt-in): a layer whose experts are ALL resident and
                        // run through MMQ groups takes them in row-count order, so each 16-expert group's max_rows
                        // (its padded tile rows) is close to its experts' own; slot/src/off are rebuilt coherently
                        // before the upload, the routed ids, weights and the within-expert row order stay as they are.
                        if (detail::mmq_resident_sort_eligible(detail::mmq_resident_sort_requested() && on_peer.empty(), use_mmq, lay.native,
                                m.cache != nullptr, m.host_res ? m.host_res + (size_t) l * m.g->n_expert : nullptr,
                                m.g->n_expert, stream_all, !stream_all || seq_start[(size_t) l] == seq_start[(size_t) l + 1]))
                            detail::mmq_resident_sort_rows(ids_h, T * K, (int32_t) K, m.cnt, m.off, order, slot_h, src_h);
                        if (grp_mapped) {
                            copy_i32(m.slot_dev, m.grp_dev + m.grp_tk, T * K, m.cs);
                            copy_i32(m.src_dev, m.grp_dev + 2 * m.grp_tk, T * K, m.cs);
                        } else {
                            if (m.up_ring.h2d(m.slot_dev, m.slot_host.data(), (size_t) T * K * 4, m.cs) != cudaSuccess ||   // #1996
                                m.up_ring.h2d(m.src_dev, m.src_host.data(), (size_t) T * K * 4, m.cs) != cudaSuccess) {
                                err = "prefill: the routing tables' upload failed";
                                return false;
                            }
                        }
                        // Diagnostics: STRATA_DUMP_MOE_INPUT=path, STRATA_DUMP_MOE_LAYER=l - the first chunk's MoE input
                        // of layer l ({T, N, K} int64, T x N floats, T x K expert ids) for testing products on real rows
                        static const char* dump_moe = std::getenv("STRATA_DUMP_MOE_INPUT");
                        static bool dumped_moe = false;
                        if (dump_moe && !dumped_moe && l == std::atoi(std::getenv("STRATA_DUMP_MOE_LAYER") ? std::getenv("STRATA_DUMP_MOE_LAYER") : "0")) {
                            dumped_moe = true;
                            std::vector<float> x((size_t) (T * N));
                            cudaMemcpyAsync(x.data(), m.mixed, x.size() * sizeof(float), cudaMemcpyDeviceToHost, m.cs);
                            cudaStreamSynchronize(m.cs);
                            if (std::FILE* f = std::fopen(dump_moe, "wb")) {
                                const int64_t hdr[3] = {(int64_t) T, (int64_t) N, (int64_t) K};
                                std::fwrite(hdr, sizeof hdr, 1, f);
                                std::fwrite(x.data(), sizeof(float), x.size(), f);
                                std::fwrite(ids_h, sizeof(int32_t), (size_t) (T * K), f);
                                std::fclose(f);
                            }
                        }
                        n_order = order.size();
                        // set_cpu_pool: the CPU's experts on a thread while this one streams and runs the GPU's; their
                        // unweighted rows land in Dm's row order, so the combine weights them as any other row
                        if (rows_cpu > 0) {
                            const size_t want = (size_t) rows_cpu * N;
                            if (m.cpu_rows_n < want) {
                                cpu_cold = true;
                                if (m.cpu_rows) cudaFreeHost(m.cpu_rows);
                                m.cpu_rows = nullptr;
                                m.cpu_rows_n = 0;
                                const size_t cap_rows = (size_t) rows_cpu * 2;   // headroom: few reallocations
                                if (cudaHostAlloc((void**) &m.cpu_rows, cap_rows * N * sizeof(float), cudaHostAllocDefault) != cudaSuccess) {
                                    err = "prefill: cannot allocate the CPU experts' rows";
                                    return false;
                                }
                                m.cpu_rows_n = cap_rows * N;
                            }
                            // this fork (NVFP4): row_sd 1 for the CPU's rows, which carry s_down already - sized on its
                            // own, as the rows buffer is now taken once at its largest above (0.1.42)
                            if (m.cpu_ones_n < (size_t) rows_cpu) {
                                if (m.cpu_ones) cudaFreeHost(m.cpu_ones);
                                m.cpu_ones = nullptr;
                                m.cpu_ones_n = 0;
                                const size_t n1 = std::max<size_t>((size_t) rows_cpu * 2, m.cpu_rows_n / N);
                                if (cudaHostAlloc((void**) &m.cpu_ones, n1 * sizeof(float), cudaHostAllocDefault) != cudaSuccess) {
                                    err = "prefill: cannot allocate the CPU experts' row scales";
                                    return false;
                                }
                                std::fill(m.cpu_ones, m.cpu_ones + n1, 1.0f);
                                m.cpu_ones_n = n1;
                            }
                            const int64_t r0c = T * K - rows_cpu;
                            const strata::kernels::cpu::NativeFmt& cf = lay.fmt[(size_t) l];
                            if (cpu_share_env() < 0.0) {
                                if (m.cpu_ev[0] == nullptr) { cudaEventCreate(&m.cpu_ev[0]); cudaEventCreate(&m.cpu_ev[1]); }
                                cudaEventRecord(m.cpu_ev[0], m.cs);
                            }
                            cpu_fut = std::async(std::launch::async, [&m, &on_cpu, &cpu_blob, &cf, &cpu_ms, src_h, l, T, r0c, this]() -> bool {
                                const auto t0 = std::chrono::steady_clock::now();
                                constexpr size_t AB = strata::kernels::cpu::kNativeActBytes;
                                // a Q2_0 layer (gate/up on the pool's Q2_0 kernels): ActQ activations, as decode's
                                const bool q2 = strata::kernels::cpu::q2_native_kernels(cf.gu_type);
                                if (q2 && m.cpu_actq.size() < (size_t) T) m.cpu_actq.resize((size_t) T);
                                if (!q2 && m.cpu_nact.size() < (size_t) T * AB) m.cpu_nact.resize((size_t) T * AB);
                                std::vector<char> need((size_t) T, 0);
                                for (int32_t e = 0; e < m.g->n_expert; ++e)   // the tokens the CPU's experts read
                                    if (on_cpu[(size_t) e])
                                        for (int32_t r = 0; r < m.cnt[(size_t) e]; ++r)
                                            need[(size_t) src_h[(size_t) m.off[(size_t) e] + (size_t) r]] = 1;
                                for (int64_t t = 0; t < T; ++t)
                                    if (need[(size_t) t] && q2)
                                        strata::kernels::cpu::act_quant_any(m.cpu_x + (size_t) t * N, (int) N, m.cpu_actq[(size_t) t]);
                                    else if (need[(size_t) t])
                                        strata::kernels::cpu::native_quant_act(cf, m.cpu_x + (size_t) t * N,
                                                                               m.cpu_nact.data() + (size_t) t * AB);
                                m.cpu_jobs.clear();
                                for (int32_t e = 0; e < m.g->n_expert; ++e) {
                                    if (!on_cpu[(size_t) e]) continue;
                                    strata::kernels::cpu::ExpertJobMulti j;
                                    j.blob = cpu_blob[(size_t) e];
                                    j.nt = m.cnt[(size_t) e];
                                    for (int r = 0; r < j.nt; ++r) {
                                        const int64_t p = (int64_t) m.off[(size_t) e] + r;
                                        if (q2) j.act[r] = &m.cpu_actq[(size_t) src_h[(size_t) p]];
                                        else j.nact[r] = m.cpu_nact.data() + (size_t) src_h[(size_t) p] * AB;
                                        j.out[r] = m.cpu_rows + (size_t) (p - r0c) * N;
                                    }
                                    m.cpu_jobs.push_back(j);
                                }
                                cpu_pool_->run_split_multi_native(cf, m.cpu_jobs.data(), (int) m.cpu_jobs.size());
                                cpu_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
                                return true;
                            });
                        }
                        const size_t mmq_gub = use_mmq ? mmq::matrix_bytes(mmq_gt, 1280, N) : 0;
                        const size_t mmq_db = use_mmq ? mmq::matrix_bytes(mmq_dt, N, 640) : 0;
                        pt.mark(kPfGather, cs);
                        if (use_mmq) {
                            // step 2b: the layer's activations as q8_1 rows in expert order, straight from `mixed`
                            mmq::quantize(m.mixed, m.src_dev, m.Xq, mmq_gt, N, N, T * K, m.cs, m.Xscale);
                            // each group's rows: absolute bounds (gate/up reads the layer's rows), relative ones (down
                            // reads the group's own quantized H)
                            const size_t n = order.size(), ng = (n + MMQ_GROUP - 1) / MMQ_GROUP;
                            m.bounds_host.resize(n + 1 + ng * (MMQ_GROUP + 1));
                            for (size_t j = 0; j < n; ++j) m.bounds_host[j] = m.off[(size_t) order[j]];
                            m.bounds_host[n] = (int32_t) rows_local;
                            for (size_t g = 0; g < ng; ++g)
                                for (size_t i = 0; i <= MMQ_GROUP; ++i)
                                    m.bounds_host[n + 1 + g * (MMQ_GROUP + 1) + i] =
                                        m.bounds_host[std::min(n, g * MMQ_GROUP + i)] - m.bounds_host[g * MMQ_GROUP];
                            if (grp_mapped) {
                                int32_t* bh = m.grp_host + 3 * m.grp_tk;
                                std::memcpy(bh, m.bounds_host.data(), m.bounds_host.size() * 4);
                                copy_i32(m.bounds_dev, m.grp_dev + 3 * m.grp_tk, (int64_t) m.bounds_host.size(), m.cs);
                            } else {
                                if (m.up_ring.h2d(m.bounds_dev, m.bounds_host.data(), m.bounds_host.size() * 4, m.cs) != cudaSuccess) {   // #1996
                                    err = "prefill: the group bounds' upload failed";
                                    return false;
                                }
                            }
                        } else {
                            gather_rows16(m.mixed_h, m.src_dev, m.Xs, T * K, N, m.cs);
                        }
                        // multi-GPU: the peer's share, enqueued before the primary's own experts so both cards work at once
                        peer_now = use_mmq && !order_peer.empty();
                        if (peer_now) {
                            PeerPrefill& P = *m.pp;
                            // the host route's copies beside the compute stream: the primary's own experts start at once
                            // (mixed_h and w are rewritten only after this layer's combine, which waits for the peer)
                            cudaStream_t xs = m.cs;
                            if (P.x_stream != nullptr) {
                                cudaEventRecord(P.ev_x, m.cs);
                                cudaStreamWaitEvent(P.x_stream, P.ev_x, 0);
                                xs = P.x_stream;
                            }
                            if (!P.p2p && P.f16) cudaMemcpyAsync(P.host_x16, m.mixed_h, (size_t) T * N * 2, cudaMemcpyDeviceToHost, xs);
                            else if (!P.p2p) cudaMemcpyAsync(P.host_x, m.mixed, (size_t) T * N * 4, cudaMemcpyDeviceToHost, xs);
                            if (P.sums) {   // the routing weights, and each peer row's routed pair (t * K + k)
                                cudaMemcpyAsync(P.host_w, m.w, (size_t) T * K * 4, cudaMemcpyDeviceToHost, xs);
                                for (int64_t i = 0; i < T * K; ++i)
                                    if (const int32_t r = slot_h[(size_t) i]; r >= rows_local) P.host_pair[r - rows_local] = (int32_t) i;
                            }
                            cudaEventRecord(P.ev_in, xs);
                            int prevd = 0;
                            cudaGetDevice(&prevd);
                            cudaSetDevice(P.dev);
                            const cudaStream_t ps = P.s;
                            cudaStreamWaitEvent(ps, P.ev_in, 0);
                            if (P.out_pending) { cudaStreamWaitEvent(ps, P.ev_done, 0); P.out_pending = false; }
                            pe.mark(kPeMoeIn, ps);
                            if (P.p2p) cudaMemcpyPeerAsync(P.mixed, P.dev, m.mixed, prevd, (size_t) T * N * 4, ps);
                            else if (P.f16) f16_to_f32_wide(P.mixed, P.host_x16, T * N, ps);
                            else copy_f32_wide(P.mixed, P.host_x, T * N, ps);
                            P.back_at = rows_local;
                            P.back_rows = rows_peer;
                            if (P.p2p) {
                                // multi-GPU: the mapped table when on; src_host (a pageable vector) goes up from a pinned slot (#1996)
                                if ((grp_mapped ? cudaMemcpyAsync(P.src, src_h + rows_local, (size_t) rows_peer * 4,
                                                                  cudaMemcpyHostToDevice, ps)
                                                : P.up_ring.h2d(P.src, src_h + rows_local, (size_t) rows_peer * 4, ps)) !=
                                    cudaSuccess) {
                                    cudaSetDevice(prevd);
                                    err = "prefill: the peer's routing table upload failed";
                                    return false;
                                }
                            } else {
                                std::memcpy(P.host_src, src_h + rows_local, (size_t) rows_peer * 4);
                                copy_i32(P.src, P.host_src, rows_peer, ps);
                            }
                            const size_t n = order_peer.size();
                            if (P.sums) {
                                copy_f32_wide(P.wk, P.host_w, T * K, ps);
                                copy_i32(P.pair, P.host_pair, rows_peer, ps);
                                cudaMemsetAsync(P.sum, 0, (size_t) T * N * 4, ps);
                            }
                            if (P.compact) {
                                // groups of up to MMQ_GROUP experts and at most G rows, each computed from its own rows
                                P.groups.clear();
                                {
                                    size_t g0 = 0;
                                    int64_t gr = 0;
                                    for (size_t j = 0; j < n; ++j) {
                                        const int64_t c = m.cnt[(size_t) order_peer[j]];
                                        if (j > g0 && (j - g0 == (size_t) MMQ_GROUP || gr + c > P.G)) { P.groups.push_back({g0, j}); g0 = j; gr = 0; }
                                        gr += c;
                                    }
                                    if (n > g0) P.groups.push_back({g0, n});
                                }
                                const size_t ng = P.groups.size();
                                P.bounds_host.resize(n + 1 + ng * (MMQ_GROUP + 1));
                                for (size_t j = 0; j < n; ++j) P.bounds_host[j] = m.off[(size_t) order_peer[j]] - (int32_t) rows_local;
                                P.bounds_host[n] = (int32_t) rows_peer;
                                for (size_t g2 = 0; g2 < ng; ++g2)
                                    for (size_t i = 0; i <= MMQ_GROUP; ++i)
                                        P.bounds_host[n + 1 + g2 * (MMQ_GROUP + 1) + i] =
                                            P.bounds_host[std::min(P.groups[g2].second, P.groups[g2].first + i)] -
                                            P.bounds_host[P.groups[g2].first];
                                if (P.p2p || P.bounds_host.size() > PeerPrefill::kHostBounds) {
                                    if (P.up_ring.h2d(P.bounds, P.bounds_host.data(), P.bounds_host.size() * 4, ps) != cudaSuccess) {   // #1996
                                        cudaSetDevice(prevd);
                                        err = "prefill: the peer's group bounds upload failed";
                                        return false;
                                    }
                                } else {
                                    std::memcpy(P.host_bounds, P.bounds_host.data(), P.bounds_host.size() * 4);
                                    copy_i32(P.bounds, P.host_bounds, (int64_t) P.bounds_host.size(), ps);
                                }
                                if (P.sums && P.gather) {   // each group's tokens, their row lists in the experts' order
                                    P.adds_at.resize(ng);
                                    P.ntok.resize(ng);
                                    size_t at = 0;
                                    for (size_t g2 = 0; g2 < ng; ++g2) {
                                        const int32_t r0 = P.bounds_host[P.groups[g2].first], r1 = P.bounds_host[P.groups[g2].second];
                                        P.touched.clear();
                                        for (int32_t r = r0; r < r1; ++r) {
                                            const int32_t t = P.host_pair[r] / (int32_t) K;
                                            if (P.tcur[(size_t) t]++ == 0) P.touched.push_back(t);
                                        }
                                        const size_t nt = P.touched.size();
                                        int32_t* a = P.host_adds + at;   // [tokens nt | starts nt + 1 | rows]
                                        a[nt] = 0;
                                        for (size_t i = 0; i < nt; ++i) {
                                            const int32_t t = P.touched[i];
                                            a[i] = t;
                                            a[nt + 1 + i] = a[nt + i] + P.tcur[(size_t) t];
                                            P.tcur[(size_t) t] = a[nt + i];   // the fill cursor
                                        }
                                        int32_t* list = a + 2 * nt + 1;
                                        for (int32_t r = r0; r < r1; ++r)
                                            list[P.tcur[(size_t) (P.host_pair[r] / (int32_t) K)]++] = r;
                                        for (int32_t t : P.touched) P.tcur[(size_t) t] = 0;
                                        P.adds_at[g2] = at;
                                        P.ntok[g2] = (int32_t) nt;
                                        at += 2 * nt + 1 + (size_t) (r1 - r0);
                                    }
                                    if (at > P.adds_cap) {
                                        err = "prefill: the peer's group tables overflow";
                                        cudaSetDevice(prevd);
                                        return false;
                                    }
                                    copy_i32(P.adds, P.host_adds, (int64_t) at, ps);
                                }
                                pe.mark(kPeMoeGemm, ps);
                                const auto& f = lay.fmt[(size_t) l];
                                for (size_t g2 = 0; g2 < ng; ++g2) {
                                    const size_t j0 = P.groups[g2].first, j1 = P.groups[g2].second;
                                    const int ngx = (int) (j1 - j0);
                                    int64_t maxr = 0;
                                    for (size_t j = j0; j < j1; ++j) {
                                        const int32_t e = order_peer[j];
                                        maxr = std::max<int64_t>(maxr, m.cnt[(size_t) e]);
                                        const uint8_t* bd = nullptr;
                                        int psl = -1;
                                        if (on_peer[(size_t) e] == 2) {   // from the peer's ring
                                            p_release_to(l, e, ps);
                                            if (P.pk >= P.pseq_start[(size_t) l + 1] || P.pseq[P.pk].e != e) {
                                                err = "prefill: a peer-streamed expert is not next in the peer's ring";
                                                cudaSetDevice(prevd);
                                                return false;
                                            }
                                            psl = (int) (P.pk % (size_t) P.RP);
                                            cudaStreamWaitEvent(ps, P.pcopied[(size_t) psl], 0);
                                            bd = P.pstage[(size_t) psl];
                                            ++P.ps_experts;
                                        } else {
                                            bd = P.peer->slot_ptr(l, e);
                                        }
                                        const size_t q = j - j0;
                                        if (lay.native)
                                            mmq::gather_native(bd, bd + f.up_off, mmq_gub / 2, bd + f.down_off, mmq_db,
                                                               P.grp_gu + q * mmq_gub, P.grp_d + q * mmq_db, ps);
                                        else
                                            mmq::gather_strata_q2(bd, P.grp_gu + q * mmq_gub, P.grp_d + q * mmq_db, ps);
                                        if (psl >= 0) {   // the slot is free once gathered
                                            cudaEventRecord(P.pused[(size_t) psl], ps);
                                            ++P.pk;
                                            p_issue_until(P.pk + (size_t) P.RP);
                                        }
                                    }
                                    const int64_t r0 = P.bounds_host[j0], nr = P.bounds_host[j1] - r0;
                                    if (nr <= 0) continue;
                                    const int32_t* rel = P.bounds + n + 1 + g2 * (MMQ_GROUP + 1);
                                    cudaMemsetAsync(P.grp_gu + (size_t) ngx * mmq_gub, 0, MMQ_TAIL, ps);
                                    cudaMemsetAsync(P.grp_d + (size_t) ngx * mmq_db, 0, MMQ_TAIL, ps);
                                    mmq::quantize(P.mixed, P.src + r0, P.Xq_g, mmq_gt, N, N, nr, ps);
                                    mmq::Product gu;
                                    gu.w = P.grp_gu; gu.type = mmq_gt; gu.w_rows = 1280; gu.w_cols = N; gu.expert_bytes = mmq_gub;
                                    gu.n = ngx; gu.xq = P.Xq_g; gu.bounds = rel; gu.ids = P.ident;
                                    gu.total_rows = nr; gu.max_rows = maxr; gu.dst = P.GU_g; gu.ld_dst = 1280;
                                    P.run_ctx->run(gu, ps);
                                    mmq::swiglu(P.GU_g, P.H_g, nr, 640, !lay.native, ps);
                                    mmq::quantize(P.H_g, nullptr, P.Hq_g, mmq_dt, 640, 640, nr, ps);
                                    const int b = (int) (g2 & 1);
                                    if (P.dm_live[b]) cudaStreamWaitEvent(ps, P.ev_dm[b], 0);   // its last rows have left
                                    mmq::Product dn;
                                    dn.w = P.grp_d; dn.type = mmq_dt; dn.w_rows = N; dn.w_cols = 640; dn.expert_bytes = mmq_db;
                                    dn.n = ngx; dn.xq = P.Hq_g; dn.bounds = rel;
                                    dn.ids = P.ident; dn.total_rows = nr; dn.max_rows = maxr; dn.dst = P.Dm_b[b];
                                    dn.ld_dst = N;
                                    P.run_ctx->run(dn, ps);
                                    if (P.sums && P.gather) {   // into the per-token sums, one launch for the group
                                        const int32_t* a = P.adds + P.adds_at[g2];
                                        const int64_t nt = P.ntok[g2];
                                        peer_gather_add(P.sum, P.Dm_b[b], r0, P.wk, P.pair, a, a + nt, a + 2 * nt + 1, nt, ps);
                                        continue;
                                    }
                                    if (P.sums) {   // into the per-token sums on this stream: Dm_b[b] is free after it
                                        for (size_t j = j0; j < j1; ++j)
                                            peer_scatter_add(P.sum, P.Dm_b[b] + (size_t) (P.bounds_host[j] - r0) * N, P.wk,
                                                             P.pair + P.bounds_host[j], P.bounds_host[j + 1] - P.bounds_host[j], ps);
                                        continue;
                                    }
                                    cudaEventRecord(P.ev_grp[g2 % PeerPrefill::kGrpEv], ps);
                                    cudaStreamWaitEvent(P.s_out, P.ev_grp[g2 % PeerPrefill::kGrpEv], 0);
                                    if (P.p2p)
                                        cudaMemcpyPeerAsync(m.Dm + (size_t) (rows_local + r0) * N, prevd, P.Dm_b[b], P.dev,
                                                            (size_t) nr * N * 4, P.s_out);
                                    else   // the primary reads them when it combines (below)
                                        cudaMemcpyAsync(P.host_rows + (size_t) r0 * N, P.Dm_b[b], (size_t) nr * N * 4,
                                                        cudaMemcpyDeviceToHost, P.s_out);
                                    cudaEventRecord(P.ev_dm[b], P.s_out);
                                    P.dm_live[b] = true;
                                }
                            } else {
                                const size_t ng = (n + MMQ_GROUP - 1) / MMQ_GROUP;
                                P.bounds_host.resize(n + 1 + ng * (MMQ_GROUP + 1));
                                for (size_t j = 0; j < n; ++j) P.bounds_host[j] = m.off[(size_t) order_peer[j]] - (int32_t) rows_local;
                                P.bounds_host[n] = (int32_t) rows_peer;
                                for (size_t g2 = 0; g2 < ng; ++g2)
                                    for (size_t i = 0; i <= MMQ_GROUP; ++i)
                                        P.bounds_host[n + 1 + g2 * (MMQ_GROUP + 1) + i] =
                                            P.bounds_host[std::min(n, g2 * MMQ_GROUP + i)] - P.bounds_host[g2 * MMQ_GROUP];
                                if (P.p2p || P.bounds_host.size() > PeerPrefill::kHostBounds) {
                                    if (P.up_ring.h2d(P.bounds, P.bounds_host.data(), P.bounds_host.size() * 4, ps) != cudaSuccess) {   // #1996
                                        cudaSetDevice(prevd);
                                        err = "prefill: the peer's group bounds upload failed";
                                        return false;
                                    }
                                } else {
                                    std::memcpy(P.host_bounds, P.bounds_host.data(), P.bounds_host.size() * 4);
                                    copy_i32(P.bounds, P.host_bounds, (int64_t) P.bounds_host.size(), ps);
                                }
                                mmq::quantize(P.mixed, P.src, P.Xq, mmq_gt, N, N, rows_peer, ps);
                                pe.mark(kPeMoeGemm, ps);
                                const auto& f = lay.fmt[(size_t) l];
                                for (size_t j = 0; j < n; ++j) {
                                    const int32_t e = order_peer[j];
                                    const uint8_t* bd = P.peer->slot_ptr(l, e);
                                    const size_t q = j % MMQ_GROUP;
                                    if (lay.native)
                                        mmq::gather_native(bd, bd + f.up_off, mmq_gub / 2, bd + f.down_off, mmq_db,
                                                           P.grp_gu + q * mmq_gub, P.grp_d + q * mmq_db, ps);
                                    else
                                        mmq::gather_strata_q2(bd, P.grp_gu + q * mmq_gub, P.grp_d + q * mmq_db, ps);
                                    if (q + 1 < MMQ_GROUP && j + 1 < n) continue;
                                    const size_t j0 = j - q, g2 = j0 / MMQ_GROUP;
                                    const int ngx = (int) (q + 1);
                                    const int64_t r0 = P.bounds_host[j0], nr = P.bounds_host[j + 1] - r0;
                                    int64_t maxr = 0;
                                    for (size_t i = j0; i <= j; ++i) maxr = std::max<int64_t>(maxr, m.cnt[(size_t) order_peer[i]]);
                                    cudaMemsetAsync(P.grp_gu + (size_t) ngx * mmq_gub, 0, MMQ_TAIL, ps);
                                    cudaMemsetAsync(P.grp_d + (size_t) ngx * mmq_db, 0, MMQ_TAIL, ps);
                                    mmq::Product gu;
                                    gu.w = P.grp_gu; gu.type = mmq_gt; gu.w_rows = 1280; gu.w_cols = N; gu.expert_bytes = mmq_gub;
                                    gu.n = ngx; gu.xq = P.Xq; gu.bounds = P.bounds + j0; gu.ids = P.ident;
                                    gu.total_rows = rows_peer; gu.max_rows = maxr; gu.dst = P.GU; gu.ld_dst = 1280;
                                    P.run_ctx->run(gu, ps);
                                    mmq::swiglu(P.GU + r0 * 1280, P.H + r0 * 640, nr, 640, !lay.native, ps);
                                    mmq::quantize(P.H + r0 * 640, nullptr, P.Hq, mmq_dt, 640, 640, nr, ps);
                                    mmq::Product dn;
                                    dn.w = P.grp_d; dn.type = mmq_dt; dn.w_rows = N; dn.w_cols = 640; dn.expert_bytes = mmq_db;
                                    dn.n = ngx; dn.xq = P.Hq; dn.bounds = P.bounds + n + 1 + g2 * (MMQ_GROUP + 1);
                                    dn.ids = P.ident; dn.total_rows = nr; dn.max_rows = maxr; dn.dst = P.Dm + r0 * N;
                                    dn.ld_dst = N;
                                    P.run_ctx->run(dn, ps);
                                    if (P.out_pipe && nr > 0) {   // this group's rows go back while the next group computes
                                        cudaEventRecord(P.ev_grp[g2], ps);
                                        cudaStreamWaitEvent(P.s_out, P.ev_grp[g2], 0);
                                        cudaMemcpyPeerAsync(m.Dm + (size_t) (rows_local + r0) * N, prevd, P.Dm + (size_t) r0 * N,
                                                            P.dev, (size_t) nr * N * 4, P.s_out);
                                    }
                                }
                            }
                            pe.mark(kPeMoeOut, ps);
                            if (P.sums) {   // the sums go back: T x N
                                if (P.f16) {
                                    sums_to_f16(P.sum, P.sum16, T * N, ps);
                                    cudaMemcpyAsync(P.host_sum16, P.sum16, (size_t) T * N * 2, cudaMemcpyDeviceToHost, ps);
                                } else {
                                    cudaMemcpyAsync(P.host_sum, P.sum, (size_t) T * N * 4, cudaMemcpyDeviceToHost, ps);
                                }
                                cudaEventRecord(P.ev_done, ps);
                                P.out_pending = true;
                            } else if (P.out_pipe) {
                                cudaEventRecord(P.ev_done, P.s_out);
                                P.out_pending = true;
                            } else {
                                cudaMemcpyPeerAsync(m.Dm + (size_t) rows_local * N, prevd, P.Dm, P.dev, (size_t) rows_peer * N * 4, ps);
                                cudaEventRecord(P.ev_done, ps);
                            }
                            pe.mark(kPeIdle, ps);
                            cudaSetDevice(prevd);
                            ++P.layers;
                            P.experts += (int64_t) n;
                            P.rows += rows_peer;
                        }
                        if (ps_on) {   // multi-GPU: this layer's peer-ring entries the routing did not pick give their slots back
                            int pd = 0;
                            cudaGetDevice(&pd);
                            cudaSetDevice(m.pp->dev);
                            p_release_to(l, (int32_t) m.g->n_expert, m.pp->s);
                            cudaSetDevice(pd);
                        }
                        // Stage ahead: the copy stream moves blobs host -> device while the compute stream works.
                        int stage_next = 0;
                        std::vector<int> stage_of(order.size(), -1);
                        // the unpinned ones are copied to pinned buffers by the stager's threads, in this order
                        std::vector<int> job_of(order.size(), -1);
                        if (!stream_all) {
                            std::vector<Stager::Job> js;
                            for (size_t j = 0; j < order.size(); ++j) {
                                const int32_t e = order[j];
                                if (m.host_res && m.cache && m.host_res[(size_t) l * m.g->n_expert + e] >= 0) continue;
                                if (m.src->pinned(l, e)) continue;
                                job_of[j] = (int) js.size();
                                if (m.src->transient(l, e)) {   // CS-T: copied by the source
                                    js.push_back({nullptr, (size_t) lay.blob_bytes(l), m.src, (int32_t) l, e});
                                    continue;
                                }
                                const uint8_t* b = m.src->blob_stable(l, e);
                                if (!b) { err = "prefill: expert source has no blob"; return false; }
                                js.push_back({b, (size_t) lay.blob_bytes(l)});
                            }
                            m.stager->start(std::move(js));
                        }
                        StagerDone stager_done{stream_all ? nullptr : m.stager.get()};
                        auto stage_one = [&](size_t j) -> bool {
                            const int32_t e = order[j];
                            const bool resident = m.host_res && m.cache && m.host_res[(size_t) l * m.g->n_expert + e] >= 0;
                            if (resident) return true;
                            const int sl = stage_next;
                            stage_next = (stage_next + 1) % STAGE;
                            const auto th = Clock::now();
                            const bool pinned = m.src->pinned(l, e);   // pinned: never transient
                            const uint8_t* b = pinned ? m.src->blob_stable(l, e) : nullptr;
                            if (pinned && !b) { err = "prefill: expert source has no blob"; return false; }
                            if (pinned) {
                                // DMA straight from the page-locked arena: the copy stream only waits for the slot
                                if (m.stage_live[sl]) cudaStreamWaitEvent(m.copy, m.used[m.used_of[sl]], 0);
                                cudaMemcpyAsync(m.stage_dev[sl], b, (size_t) lay.blob_bytes(l), cudaMemcpyHostToDevice, m.copy);
                                ++stats_.experts_dma;
                            } else {
                                // copied to a pinned buffer by the stager (waits only if it is behind), then DMA
                                const uint8_t* hb = m.stager->wait(job_of[j]);
                                if (m.stage_live[sl]) cudaStreamWaitEvent(m.copy, m.used[m.used_of[sl]], 0);
                                cudaMemcpyAsync(m.stage_dev[sl], hb, (size_t) lay.blob_bytes(l), cudaMemcpyHostToDevice, m.copy);
                                m.stager->issued_one(job_of[j], m.copy);
                            }
                            cudaEventRecord(m.copied[sl], m.copy);
                            m.stage_live[sl] = true;
                            stage_of[j] = sl;
                            stats_.ms_experts_host += ms_since(th);
                            ++stats_.experts_streamed;
                            return true;
                        };
                        // In the streamed walk an MMQ group is gathered in ONE launch, after ONE wait on its last
                        // streamed copy (the copy stream is in order), and its ring slots are released by ONE event.  A
                        // wait and a record per expert put ~10 us of GPU bubble each on the compute stream under WDDM
                        // (226 ms of a 32K prompt on an NVFP4 pack).  The same bytes into the same group slots.  A ring
                        // entry the routing skipped inside an open group first gathers what the group holds so far
                        // (`flush`), so no more than a group's entries are ever held back from the issuer.
                        // STRATA_PREFILL_GROUP_GATHER=0: one gather, one wait and one record per expert.
                        static const bool group_env = [] {
                            const char* v = std::getenv("STRATA_PREFILL_GROUP_GATHER");
                            return v == nullptr || std::atoi(v) != 0;
                        }();
                        const bool group_gather = group_env && stream_all && use_mmq && lay.native &&
                                                  MMQ_GROUP <= mmq::kGatherGroupMax;
                        mmq::GatherGroup gg;
                        int gg_slots[MMQ_GROUP];
                        int gg_nslots = 0;   // ring slots gathered by the next flush
                        // `zero`: the MMQ tail zeroed after the gathered experts (the group's last flush)
                        auto flush = [&](size_t zero) {
                            if (gg.n <= gg.first) return;
                            const auto& f = lay.fmt[(size_t) l];
                            if (gg_nslots > 0) {   // the copies land in order: the last one covers the others
                                pt.mark(kPfWaitCopy, cs);
                                cudaStreamWaitEvent(m.cs, m.copied[gg_slots[gg_nslots - 1]], 0);
                                pt.mark(kPfDequant, cs);
                            }
                            // NVFP4: each expert's scales go next to its group slot in the same launch
                            if (!mmq::gather_native_group(gg, f.up_off, mmq_gub / 2, f.down_off, mmq_db, m.grp_gu, mmq_gub,
                                                          m.grp_d, mmq_db, m.cs, f.tail_off,
                                                          f.tail_off ? m.grp_tail : nullptr, zero)) {
                                for (int i = gg.first; i < gg.n; ++i) {   // not 16-byte aligned: one at a time
                                    const uint8_t* b = gg.blob[i];
                                    mmq::gather_native(b, b + f.up_off, mmq_gub / 2, b + f.down_off, mmq_db,
                                                       m.grp_gu + i * mmq_gub, m.grp_d + i * mmq_db, m.cs,
                                                       f.tail_off ? b + f.tail_off : nullptr,
                                                       f.tail_off ? m.grp_tail + i * 4 : nullptr, i + 1 == gg.n ? zero : 0);
                                }
                            }
                            if (gg_nslots > 0) {
                                const int rel = gg_slots[gg_nslots - 1];
                                cudaEventRecord(m.used[rel], m.cs);
                                for (int i = 0; i < gg_nslots; ++i) m.used_of[gg_slots[i]] = rel;
                            }
                            gg_nslots = 0;
                            gg.first = gg.n;
                        };
                        // one expert's products from its blob on the device; `slot` (a ring slot, or -1 for a resident
                        // expert) is released once the blob is read
                        auto compute = [&](size_t j, const uint8_t* blob_dev, int slot) -> bool {
                            const int32_t e = order[j];
                            pt.mark(kPfDequant, cs);
                            if (use_mmq) {
                                // gather the expert into its group slot (GGUF blocks, unchanged or converted)
                                const size_t q = j % MMQ_GROUP;
                                // the group's last expert also zeroes the MMQ tail after its slot (see MMQ_TAIL)
                                const bool last_of_group = q + 1 == MMQ_GROUP || j + 1 == order.size();
                                if (group_gather) {
                                    gg.blob[q] = blob_dev;
                                    gg.n = (int) q + 1;
                                    if (slot >= 0) gg_slots[gg_nslots++] = slot;
                                    if (!last_of_group) return true;
                                    flush(MMQ_TAIL);
                                    gg = mmq::GatherGroup{};
                                } else if (lay.native) {
                                    const auto& f = lay.fmt[(size_t) l];
                                    // NVFP4: this expert's scales go next to its group slot in the same launch
                                    mmq::gather_native(blob_dev, blob_dev + f.up_off, mmq_gub / 2, blob_dev + f.down_off,
                                                       mmq_db, m.grp_gu + q * mmq_gub, m.grp_d + q * mmq_db, m.cs,
                                                       f.tail_off ? blob_dev + f.tail_off : nullptr,
                                                       f.tail_off ? m.grp_tail + q * 4 : nullptr, last_of_group ? MMQ_TAIL : 0);
                                } else {
                                    mmq::gather_strata_q2(blob_dev, m.grp_gu + q * mmq_gub, m.grp_d + q * mmq_db, m.cs);
                                    if (last_of_group) {
                                        cudaMemsetAsync(m.grp_gu + (size_t) (q + 1) * mmq_gub, 0, MMQ_TAIL, m.cs);
                                        cudaMemsetAsync(m.grp_d + (size_t) (q + 1) * mmq_db, 0, MMQ_TAIL, m.cs);
                                    }
                                }
                                if (slot >= 0 && !group_gather) { cudaEventRecord(m.used[slot], m.cs); m.used_of[slot] = slot; }
                                if (q + 1 < MMQ_GROUP && j + 1 < order.size()) return true;
                                // the group's products: gate/up, swiglu, the group's H to q8_1, down
                                const size_t j0 = j - q, g = j0 / MMQ_GROUP, n = order.size();
                                const int ngx = (int) (q + 1);
                                const int64_t r0 = m.bounds_host[j0], nr = m.bounds_host[j + 1] - r0;
                                int64_t maxr = 0;
                                for (size_t i = j0; i <= j; ++i) maxr = std::max<int64_t>(maxr, m.cnt[(size_t) order[i]]);
                                pt.mark(kPfGemmGU, cs);
                                mmq::Product gu;
                                gu.w = m.grp_gu; gu.type = mmq_gt; gu.w_rows = 1280; gu.w_cols = N; gu.expert_bytes = mmq_gub;
                                gu.n = ngx; gu.xq = m.Xq; gu.bounds = m.bounds_dev + j0; gu.ids = m.ids_identity;
                                gu.total_rows = T * K; gu.max_rows = maxr; gu.dst = m.GU; gu.ld_dst = 1280;
                                gu.y_scale = mmq::fp4_activations(mmq_gt) ? m.Xscale : nullptr;
                                m.mmq_ctx->run(gu, m.cs);
                                if (lay.native && lay.fmt[(size_t) l].tail_off)   // gate/up scales applied as read
                                    mmq::swiglu_scaled(m.GU + r0 * 1280, m.H + r0 * 640, nr, 640, m.bounds_dev + j0, ngx,
                                                       m.grp_tail, r0, m.cs);
                                else
                                    mmq::swiglu(m.GU + r0 * 1280, m.H + r0 * 640, nr, 640, !lay.native, m.cs);
                                pt.mark(kPfGemmD, cs);
                                mmq::quantize(m.H + r0 * 640, nullptr, m.Hq, mmq_dt, 640, 640, nr, m.cs, m.Hscale);
                                mmq::Product dn;
                                dn.w = m.grp_d; dn.type = mmq_dt; dn.w_rows = N; dn.w_cols = 640; dn.expert_bytes = mmq_db;
                                dn.n = ngx; dn.xq = m.Hq; dn.bounds = m.bounds_dev + n + 1 + g * (MMQ_GROUP + 1);
                                dn.ids = m.ids_identity; dn.total_rows = nr; dn.max_rows = maxr; dn.dst = m.Dm + r0 * N;
                                dn.ld_dst = N;
                                dn.y_scale = mmq::fp4_activations(mmq_dt) ? m.Hscale : nullptr;
                                m.mmq_ctx->run(dn, m.cs);
                                if (lay.native && lay.fmt[(size_t) l].tail_off)   // s_down: applied by the combine
                                    mmq::down_row_scales(m.row_sd + r0, dn.bounds, ngx, m.grp_tail, nr, m.cs);
                                return true;
                            }
                            const int q = (int) (j % DQ);
                            if (lay.native) {
                                // plan v0.3 P6: a native pack's layer, dequantized by llama.cpp's own formulas
                                const auto& f = lay.fmt[(size_t) l];
                                strata::kernels::iq_dequant_gu_f16(f.gu_type, blob_dev, blob_dev + f.up_off, f.n_ff, f.n_embd,
                                                                   m.dq_gu[q], m.cs,
                                                                   f.tail_off ? (const float*) (blob_dev + f.tail_off) : nullptr);
                                strata::kernels::iq_dequant_f16(f.d_type, blob_dev + f.down_off, f.n_embd * f.n_ff, m.dq_d[q], m.cs,
                                                                f.tail_off ? (const float*) (blob_dev + f.tail_off) + 2 : nullptr);
                            } else {
                                blob_dequant_f16(blob_dev, m.dq_gu[q], m.dq_d[q], m.cs);
                            }
                            if (slot >= 0) { cudaEventRecord(m.used[slot], m.cs); m.used_of[slot] = slot; }
                            const int64_t o0 = m.off[(size_t) e], ne = m.cnt[(size_t) e];
                            pt.mark(kPfGemmGU, cs);
                            m.gemm.f16(m.Xs + o0 * N, m.dq_gu[q], m.GU + o0 * 1280, ne, 1280, N);
                            swiglu_interleaved(m.GU + o0 * 1280, m.Hh + o0 * 640, ne, m.cs);
                            pt.mark(kPfGemmD, cs);
                            m.gemm.f16(m.Hh + o0 * 640, m.dq_d[q], m.Dm + o0 * N, ne, N, 640);
                            return true;
                        };
                        if (!stream_all) {
                            size_t staged = 0;
                            size_t pending = 0;
                            const bool stream_ahead = stream_ahead_enabled();
                            const size_t lookahead = STAGE - 1;
                            for (size_t j = 0; j < order.size(); ++j) {
                                // Resident experts occupy no staging slot. Keep STAGE actual transfers ahead,
                                // rather than STAGE positions in the mixed resident/streamed order.
                                while (staged < order.size() && (stream_ahead ? pending < STAGE : staged <= j + lookahead)) {
                                    if (!stage_one(staged)) return false;
                                    if (stage_of[staged] >= 0) ++pending;
                                    ++staged;
                                }
                                const int32_t e = order[j];
                                if (stage_of[j] < 0) {
                                    ++stats_.experts_resident;
                                    if (!compute(j, m.cache->device_slot(m.host_res[(size_t) l * m.g->n_expert + e]), -1)) return false;
                                } else {
                                    pt.mark(kPfWaitCopy, cs);
                                    cudaStreamWaitEvent(m.cs, m.copied[stage_of[j]], 0);
                                    if (!compute(j, m.stage_dev[stage_of[j]], stage_of[j])) return false;
                                    --pending;   // compute recorded the slot's release event before any reuse
                                }
                            }
                        } else {
                            // the streamed walk: this layer's entries [k, kend) in id order; an entry the routing did not
                            // pick only gives its slot back
                            size_t k = seq_start[(size_t) l];
                            const size_t kend = seq_start[(size_t) l + 1];
                            auto release_to = [&](int32_t e_stop) {
                                while (k < kend && seq[k].e < e_stop) {
                                    if (gg_nslots > 0) flush(0);   // the open group's slots get their event first
                                    const int sl = (int) (k % (size_t) m.ring);
                                    cudaEventRecord(m.used[sl], m.cs);
                                    m.used_of[sl] = sl;
                                    consumed = ++k;
                                    give_back(consumed);
                                }
                            };
                            for (size_t j = 0; j < order.size(); ++j) {
                                const int32_t e = order[j];
                                release_to(e);
                                if (k < kend && seq[k].e == e) {
                                    const int sl = (int) (k % (size_t) m.ring);
                                    wait_issued(k);
                                    if (!group_gather) {
                                        pt.mark(kPfWaitCopy, cs);
                                        cudaStreamWaitEvent(m.cs, m.copied[sl], 0);
                                    }
                                    if (!compute(j, m.stage_dev[sl], sl)) return false;
                                    consumed = ++k;
                                    if (!group_gather || gg_nslots == 0) give_back(consumed);   // its group was gathered
                                } else {
                                    const bool r0 = m.host_res && m.cache && m.host_res[(size_t) l * m.g->n_expert + e] >= 0;
                                    const uint8_t* bp = r0 ? m.cache->device_slot(m.host_res[(size_t) l * m.g->n_expert + e])
                                                           : (m.pp && m.pp->peer ? m.pp->peer->slot_ptr(l, e) : nullptr);   // over the cap: P2P
                                    if (bp == nullptr) { err = "prefill: an expert is neither resident, streamed nor on the peer"; return false; }
                                    ++stats_.experts_resident;
                                    if (!compute(j, bp, -1)) return false;
                                    if (group_gather && gg_nslots == 0) give_back(consumed);
                                }
                            }
                            release_to(m.g->n_expert);
                        }
                    }
                    pt.mark(kPfCombine, cs);
                    if (cpu_fut.valid()) {   // set_cpu_pool: the CPU's rows into Dm's tail, row_sd 1 (s_down is in them)
                        if (cpu_share_env() < 0.0) cudaEventRecord(m.cpu_ev[1], m.cs);   // after the GPU's expert work
                        if (!cpu_fut.get()) { err = "prefill: a CPU expert has no blob"; return false; }
                        if (cpu_share_env() < 0.0 && n_stream > n_cpu) {
                            m.cpu_pend = true;
                            m.pend_cpu_ms = cpu_ms;
                            m.pend_n_cpu = n_cpu;
                            m.pend_n_gpu = n_stream - n_cpu;
                        }
                        stats_.cpu_share = cpu_share_env() >= 0.0 ? cpu_share_env() : m.cpu_share_now;
                        const int64_t r0c = T * K - rows_cpu;
                        cudaMemcpyAsync(m.Dm + (size_t) r0c * N, m.cpu_rows, (size_t) rows_cpu * N * sizeof(float),
                                        cudaMemcpyHostToDevice, m.cs);
                        if (use_mmq && !fused_l && lay.native && lay.fmt[(size_t) l].tail_off)
                            cudaMemcpyAsync(m.row_sd + r0c, m.cpu_ones, (size_t) rows_cpu * sizeof(float),
                                            cudaMemcpyHostToDevice, m.cs);
                        stats_.experts_cpu += n_cpu;
                    }
                    // no reading from a layer the share could not change (nothing taken) or with one-time costs
                    if (cpu_set >= 0 && n_stream > 0 && (!cpu_arm || n_cpu > 0) && !cpu_cold) {
                        cudaEventRecord(m.cpu_wall[2 * cpu_set + 1], m.cs);
                        m.pend_wall = cpu_arm ? 2 : 1;
                        m.pend_set = cpu_set;
                        m.pend_l = (int) l;
                        m.pend_wall_n = n_stream;
                    }
                    if (peer_now) {   // multi-GPU: the peer's rows are in Dm (or, without P2P, in host memory)
                        cudaStreamWaitEvent(m.cs, m.pp->ev_done, 0);
                        if (!m.pp->p2p && !m.pp->sums)
                            copy_f32_wide(m.Dm + (size_t) m.pp->back_at * N, m.pp->host_rows, m.pp->back_rows * N, m.cs);
                    }
                    if (peer_now && m.pp->sums && m.pp->f16)
                        moe_combine_peer16(m.Dm, m.slot_dev, m.w, m.shared, m.sg, m.pp->host_sum16, m.pp->back_at, m.bo, T, m.cs);
                    else if (peer_now && m.pp->sums)
                        moe_combine_peer(m.Dm, m.slot_dev, m.w, m.shared, m.sg, m.pp->host_sum, m.pp->back_at, m.bo, T, m.cs);
                    else
                        moe_combine(m.Dm, m.slot_dev, m.w, m.shared, m.sg, m.bo, T, m.cs,
                                    use_mmq && !fused_l && lay.native && lay.fmt[(size_t) l].tail_off ? m.row_sd : nullptr);
                    // debug: STRATA_DBG_NAN=1 reports the first layer of a chunk whose MoE produced non-finite values
                    if (static const bool dbg = std::getenv("STRATA_DBG_NAN") != nullptr; dbg) {
                        cudaStreamSynchronize(m.cs);
                        auto bad = [&](const float* d, int64_t n) {
                            std::vector<float> h((size_t) n);
                            cudaMemcpy(h.data(), d, (size_t) n * 4, cudaMemcpyDeviceToHost);
                            int64_t c = 0;
                            for (float v : h) c += !std::isfinite(v);
                            return c;
                        };
                        // (fused: GU and H hold the grouping tables and int8 rows, not floats)
                        // (the CPU's rows, the last rows_cpu of GU and H, are never written by the GPU: not checked)
                        const int64_t gpu_rows = T * K - rows_cpu;
                        const int64_t bgu = fused_l ? 0 : bad(m.GU, gpu_rows * 1280), bdm = bad(m.Dm, T * K * N),
                                      bbo = bad(m.bo, T * N);
                        const int64_t bh = m.H && !fused_l ? bad(m.H, gpu_rows * 640) : -1;
                        static int64_t reported = -1;
                        if ((bgu || bdm || bbo || bh > 0) && reported != stats_.chunks) {
                            reported = stats_.chunks;
                            std::fprintf(stderr, "strata dbg: layer %lld (mmq %d, types %d/%d, %zu experts): non-finite GU %lld "
                                         "H %lld Dm %lld bo %lld of T %lld\n", (long long) l, (int) use_mmq, mmq_gt, mmq_dt,
                                         n_order, (long long) bgu, (long long) bh, (long long) bdm, (long long) bbo,
                                         (long long) T);
                        }
=======
                        // ======================= QSA =======================
                        const core::QsaState& st = ss.qsa_states[qsa_index];
                        const core::WeightRef *wq = need(v, "attn_q.weight", err), *wk = need(v, "attn_k.weight", err),
                                              *wv = need(v, "attn_v.weight", err), *wo = need(v, "attn_output.weight", err),
                                              *wik = need(v, "indexer.k_proj.weight", err),
                                              *wiq = need(v, "indexer.q_proj.weight", err),
                                              *wqn = need(v, "attn_q_norm.weight", err), *wkn = need(v, "attn_k_norm.weight", err),
                                              *wiqn = need(v, "indexer.q_norm.weight", err),
                                              *wikn = need(v, "indexer.k_norm.weight", err);
                        if (!wq || !wk || !wv || !wo || !wik || !wiq || !wqn || !wkn || !wiqn || !wikn) return false;
                        const int32_t* steps = b.steps + t0 * SC;
                        mark(kPsQsaProj);
                        if (!native_proj(wk, mixed_h, b.Kc, P, v.name("attn_k.weight"), err)) return false;
                        if (!native_proj(wv, mixed_h, b.Vc, P, v.name("attn_v.weight"), err)) return false;
                        if (!native_proj(wq, mixed_h, b.Qf, P, v.name("attn_q.weight"), err)) return false;
                        if (!bf16_proj(m.gemm, wik, b.mixed_bf, b.idx_raw, P, v.name("indexer.k_proj.weight"), err)) return false;
                        if (!bf16_proj(m.gemm, wiq, b.mixed_bf, b.q_idx, P, v.name("indexer.q_proj.weight"), err)) return false;
                        rms_rows(b.Kc, (const float*) wkn->data, P * 2, 256, 256, EPS, m.cs);
                        rope(b.Kc, P, 2, 256, 512, pp, (float) strata::kernels::qsa_freq_base(), m.cs);
                        append_kv(st, b.Kc, b.Vc, P, pp, s, m.cs);
                        split_q(b.Qf, b.q, P, m.cs);
                        rms_rows(b.q, (const float*) wqn->data, P * 24, 256, 256, EPS, m.cs);
                        rope(b.q, P, 24, 256, 6144, pp, (float) strata::kernels::qsa_freq_base(), m.cs);
                        rms_rows(b.q_idx, (const float*) wiqn->data, P * 4, 128, 128, EPS, m.cs);
                        rope(b.q_idx, P, 4, 128, 512, pp, (float) strata::kernels::qsa_freq_base(), m.cs);
                        // the indexer appends of the sub-chunk; then scores + selection for many queries at once:
                        // a query reads completed blocks (final once completed) and `dead` for its own tail block
                        const strata::kernels::QsaIndexerBuffers ib{st.idx_tail, st.idx_dead, st.idx_pooled, st.idx_block_pos};
                        mark(kPsQsaIndexer);
                        try {
                            strata::kernels::native_qsa_indexer_append_multi(
                                b.idx_raw, steps + strata::kernels::kStepPos, (int) SC, (int) P, 0,
                                (const float*) wikn->data, EPS, ib, s, st.max_cells, (float) strata::kernels::qsa_freq_base(), m.cs);
                        } catch (const std::exception& e) { err = std::string("prefill indexer: ") + e.what(); return false; }
                        mark(kPsQsaScores);
                        for (int64_t q0 = 0; q0 < P; q0 += SEL_BATCH) {
                            const int64_t nb = std::min(SEL_BATCH, P - q0);
                            // the batch's last query has completed the most blocks
                            const int64_t grid_blocks =
                                m.steps_host[(size_t) (t0 + q0 + nb - 1) * SC + strata::kernels::kStepNBid] + 1;
                            strata::kernels::qsa_block_scores(st.idx_pooled, st.idx_dead, b.q_idx + q0 * 512, steps + q0 * SC,
                                                              nb, m.max_blocks, s, b.sel_scores, m.cs, grid_blocks);
                            strata::kernels::qsa_block_topk(b.sel_scores, steps + q0 * SC, nb, m.max_blocks, m.cap, s,
                                                            b.sel_ids + q0 * m.cap, m.cs);
                        }
                        const strata::kernels::QsaAttnPools pools = core::qsa_attn_pools(st);
                        mark(kPsQsaAttn);
                        strata::kernels::qsa_prefill_attn(b.q, pools, b.sel_ids, steps, m.cap, s, b.attn, P, m.cs);
                        mark(kPsQsaOut);
                        gate_attn(b.attn, b.Qf, b.attn_h, P, m.cs);
                        if (!native_proj(wo, b.attn_h, b.bo, P, v.name("attn_output.weight"), err)) return false;
>>>>>>> their fork (github.com-eddoursul-Strata-custom)
                    }
                    // ---- the hyper-connection write of the mixer half, with the FFN half's norm
                    mark(kPsHcFfn);
                    const core::WeightRef* wfn = need(v, "hc_ffn_norm.weight", err);
                    if (!wfn) return false;
                    gr_write_norm_rs(b.R + t0 * D, b.bo, b.inj_a, HC, (const float*) wfn->data, EPS, b.rs, b.xn16, P,
                                     m.cs);
                }
                // ---- the FFN half's input is ready: its trip to the second GPU starts here
                if (m.offload) {
                    cudaEventRecord(m.ev_mixed, m.cs);
                    cudaStreamWaitEvent(m.xfer, m.ev_mixed, 0);
                    cudaMemcpyAsync(m.offload->host_input() + t0 * N, mixed_h, (size_t) P * N * 2,
                                    cudaMemcpyDeviceToHost, m.xfer);
                    cudaEventRecord(m.ev_in[(size_t) (t0 / SUB)], m.xfer);
                    m.offload->stage_input(t0, P, m.ev_in[(size_t) (t0 / SUB)]);
                }
                // ---- the router, and the shared expert (its gated output starts the MoE sum)
                mark(kPsRouter);
                if (!bf16_proj(m.gemm, wr, b.mixed_bf, b.logits, P, v.name("ffn_gate_inp.weight"), err)) return false;
                route(b.logits, b.ids + t0 * K, b.w + t0 * K, P, m.cs);
                if (!native_proj(wsg, mixed_h, b.sgate, P, v.name("ffn_gate_shexp.weight"), err)) return false;
                if (!native_proj(wsu, mixed_h, b.sup, P, v.name("ffn_up_shexp.weight"), err)) return false;
                swiglu_pair(b.sgate, b.sup, b.sh_h, P, m.cs);
                if (!native_proj(wsd, b.sh_h, b.bo_moe + t0 * N, P, v.name("ffn_down_shexp.weight"), err)) return false;
                m.gemm.bf16(b.mixed_bf, (const uint16_t*) wgi->data, b.sg, P, 1, N);
                moe_shared_gate(b.bo_moe + t0 * N, b.sg, P, m.cs);
                cudaMemcpyAsync(m.ids_host + t0 * K, b.ids + t0 * K, (size_t) P * K * 4, cudaMemcpyDeviceToHost, m.cs);
                cudaMemcpyAsync(m.w_host + t0 * K, b.w + t0 * K, (size_t) P * K * 4, cudaMemcpyDeviceToHost, m.cs);
            }
            pending = false;
            if (!qsa) ++gdn_index;
            else ++qsa_index;
            // ================ the routed experts of the whole chunk ================
            if (cudaStreamSynchronize(m.cs) != cudaSuccess) {
                err = std::string("prefill: ") + cudaGetErrorString(cudaGetLastError());
                return false;
            }
            mark(kPsExperts);
            // group the (token, k) entries by expert: this GPU takes the experts its cache holds (all of them without
            // a second GPU), the second GPU the others
            std::fill(m.cnt.begin(), m.cnt.end(), 0);
            for (int64_t i = 0; i < T * K; ++i) {
                const int32_t e = m.ids_host[i];
                if (e < 0 || e >= NE) { err = "prefill: routed id out of range"; return false; }
                ++m.cnt[(size_t) e];
            }
            m.off[0] = 0;
            for (int64_t e = 0; e < NE; ++e) m.off[(size_t) e + 1] = m.off[(size_t) e] + m.cnt[(size_t) e];
            {
                std::vector<int32_t> fill(m.off.begin(), m.off.end() - 1);
                for (int64_t i = 0; i < T * K; ++i) m.entry_of[(size_t) fill[(size_t) m.ids_host[i]]++] = (int32_t) i;
            }
            for (auto* v2 : {&m.ex1, &m.src1, &m.ex2, &m.src2}) v2->clear();
            m.w1.clear();
            m.w2.clear();
            m.off1.assign(1, 0);
            m.off2.assign(1, 0);
            for (int32_t e = 0; e < NE; ++e) {
                if (m.cnt[(size_t) e] == 0) continue;
                const bool here = !m.offload || (m.host_res && m.cache && m.host_res[(size_t) l * NE + e] >= 0);
                std::vector<int32_t>& ex = here ? m.ex1 : m.ex2;
                std::vector<int32_t>& of = here ? m.off1 : m.off2;
                std::vector<int32_t>& sr = here ? m.src1 : m.src2;
                std::vector<float>& wv = here ? m.w1 : m.w2;
                ex.push_back(e);
                for (int32_t p = m.off[(size_t) e]; p < m.off[(size_t) e + 1]; ++p) {
                    const int32_t i = m.entry_of[(size_t) p];
                    sr.push_back(i / (int32_t) K);
                    wv.push_back(m.w_host[i]);
                }
                of.push_back((int32_t) sr.size());
            }
            if (m.offload && !m.offload->run_layer(l, T, m.ex2, m.off2, m.src2, m.w2, SUB, nullptr, nullptr, err))
                return false;
            if (!m.local.run_layer(l, T, m.ex1, m.off1, m.src1, m.w1, SUB, b.mixed_h, b.bo_moe, err)) return false;
            // ---- the MoE output (+ the second GPU's sums) goes into the residual stream as the next layer reads it
            if (m.offload) {
                for (int64_t t0 = 0; t0 < T; t0 += SUB) {
                    cudaStreamWaitEvent(m.xsum, m.offload->piece_done(t0 / SUB), 0);
                    const size_t bytes = (size_t) std::min(SUB, T - t0) * N * 2;
                    cudaMemcpyAsync(b.sums + t0 * N, m.offload->host_sum() + t0 * N, bytes, cudaMemcpyHostToDevice,
                                    m.xsum);
                    cudaEventRecord(m.ev_piece[(size_t) (t0 / SUB)], m.xsum);
                }
            }
            pending = true;
        }
        mark(kPsCombine);   // the last layer's write
        for (int64_t t0 = 0; t0 < T; t0 += SUB) ffn_write(t0, std::min(SUB, T - t0), nullptr);
        if (draft != nullptr && !draft_pass(tokens + c0, T, p0)) return false;
        if (profile) {
            mark(kPsPle);
            cudaEventSynchronize(ev.back().second);
            for (size_t i = 0; i + 1 < ev.size(); ++i) {
                float ms = 0;
                cudaEventElapsedTime(&ms, ev[i].second, ev[i + 1].second);
                stats_.ms_section[ev[i].first] += ms;
            }
            for (auto& e : ev) cudaEventDestroy(e.second);
            ev.clear();
        }
        stats_.tokens += T;
        for (int64_t t = 0; t < T; ++t) { prev[0] = prev[1]; prev[1] = (int32_t) tokens[c0 + t]; }
    }
    ss.ple_prev[0] = prev[0];
    ss.ple_prev[1] = prev[1];
    if (cudaStreamSynchronize(m.cs) != cudaSuccess) {
        err = std::string("prefill: ") + cudaGetErrorString(cudaGetLastError());
        return false;
    }
    stats_.experts_streamed = m.local.experts_streamed;
    stats_.experts_prefetched = m.local.experts_prefetched;
    stats_.experts_resident = m.local.experts_resident;
    stats_.ms_experts_host = m.local.ms_host;
    stats_.ms_total += ms_since(t_start);
    processed_ = std::min(c0, n);
    if (c0 < n) {
        err = "cancelled";
        return false;
    }
    return true;
}

<<<<<<< this fork
bool Prefill::drain_pipeline(std::string& err) {
    bool ok = true;

    if (next_run_.valid()) {
        if (!next_run_.get()) {
            err = next_err_;
            ok = false;
        }
    }

    if (next_ != nullptr) {
        std::string tail_err;
        if (!next_->drain_pipeline(tail_err)) {
            if (ok) err = tail_err;
            ok = false;
        }
    }

    return ok;
}

bool Prefill::run(const int64_t* tokens, int64_t n, int64_t pos0, std::string& err) {
#if defined(STRATA_GPU_METAL)
    {
        const auto t0 = std::chrono::steady_clock::now();
        const bool ok = strata::metal_engine::prompt_run(tokens, n, pos0, embd_rows, on_chunk, err);
        stats_.tokens += n;
        stats_.chunks += 1;
        stats_.ms_total += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        return ok;
    }
#endif
    const bool body_ok = run_impl(tokens, n, pos0, err);

    std::string drain_err;
    const bool drain_ok = drain_pipeline(drain_err);

    if (!body_ok) return false;
    if (!drain_ok) {
        err = drain_err;
        return false;
    }
    return true;
=======
const char* prefill_section_name(int section) {
    static const char* const names[kPsCount] = {
        "PLE block", "HC read (mixer)", "GDN projections", "GDN conv", "GDN recurrence", "GDN output norm",
        "GDN output projection",
        "QSA projections, norms, KV", "QSA indexer appends",
        "QSA block scores + top-k", "QSA attention", "QSA gate + output", "HC write + HC read (FFN)",
        "router, shared expert", "grouping, experts", "combine + HC write", "draft layer K/V"};
    return section >= 0 && section < kPsCount ? names[section] : "?";
>>>>>>> their fork (github.com-eddoursul-Strata-custom)
}

}  // namespace strata::prefill
