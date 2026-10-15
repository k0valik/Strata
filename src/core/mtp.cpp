// src/core/mtp.cpp - see include/strata/core/mtp.hpp.
#include "strata/core/mtp.hpp"

#include "strata/core/native_head.hpp"
#include "strata/kernels/bf16_gemv.hpp"
#include "strata/kernels/cpu/expert.hpp"
#include "strata/kernels/elementwise.hpp"
#include "strata/kernels/fused_gr.hpp"
#include "strata/kernels/gr.hpp"
#include "strata/kernels/kv_q8.hpp"
#include "strata/kernels/native_moe.hpp"
#include "strata/kernels/native_mmvq.hpp"
#include "strata/kernels/native_qsa.hpp"
#include "strata/kernels/native_rope.hpp"
#include "strata/kernels/native_router.hpp"
#include "strata/kernels/qsa.hpp"
#include "strata/kernels/qsa_decode_attn.hpp"
#include "strata/kernels/quantize_act.hpp"
#include "strata/kernels/rope.hpp"
#include "strata/kernels/router_top10.hpp"
#include "strata/kernels/s2_expert_grouped.hpp"
#include "strata/kernels/sampler.hpp"
#include "strata/kernels/shared_expert.hpp"
#include "strata/kernels/verify_kernels.hpp"
<<<<<<< this fork
#include "strata/kernels/ngram.hpp"
#include "strata/kernels/gfx_arch.hpp"
=======
>>>>>>> their fork (github.com-eddoursul-Strata-custom)

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <fstream>
#include <sstream>
#include <vector>

#if defined(STRATA_GPU_METAL)
// The Metal backend's MTP drafter (metal/src/engine/window.cpp, beside the decode window it reads): load, prefill and
// draft hand over to it; bind has nothing to bind there.
namespace strata::metal_engine {
bool mtp_load(const std::string& rt_dir, int max_t, std::string& err);
bool mtp_prefill(const int32_t* next_tokens, int64_t n, int64_t cell0, std::string& err);
bool mtp_draft(int T, const int32_t* tokens, int64_t p, int a, int max_steps, int32_t* drafts, float* probs, float min_p,
               int* n_drafts, std::string& err);
}  // namespace strata::metal_engine
#endif

namespace strata::core {
namespace {

constexpr float EPS = 1e-6f;
constexpr int GGML_Q8_0 = 8;
using Clock = std::chrono::steady_clock;
double ms_since(Clock::time_point t) { return std::chrono::duration<double, std::milli>(Clock::now() - t).count(); }

struct Bump {
    uint8_t* base = nullptr;
    uint64_t used = 0;
    template <typename T> T* take(uint64_t n) {
        T* p = base ? (T*) (base + used) : nullptr;
        used += (n * sizeof(T) + 255) & ~255ull;
        return p;
    }
};

bool mapped(size_t bytes, void** h, void** d) {
    if (cudaHostAlloc(h, bytes, cudaHostAllocMapped) != cudaSuccess) return false;
    std::memset(*h, 0, bytes);
    return cudaHostGetDevicePointer(d, *h, 0) == cudaSuccess;
}

strata::kernels::QsaShapes shapes_of(const ModelGeometry& g) { return qsa_shapes(g); }

bool read_file(const std::string& path, std::vector<uint8_t>& out) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) return false;
    const std::streamsize n = f.tellg();
    f.seekg(0);
    out.resize((size_t) n);
    return (bool) f.read((char*) out.data(), n);
}

}  // namespace

MtpDrafter::~MtpDrafter() {
    if (cs_) cudaStreamSynchronize(cs_);
    for (auto& e : round_exec_) if (e) cudaGraphExecDestroy(e);
    for (auto& e : step_exec_) if (e) cudaGraphExecDestroy(e);
    if (cs_) cudaStreamDestroy(cs_);
    if (side_) cudaStreamDestroy(side_);
    if (sh_fork_) cudaEventDestroy(sh_fork_);
    if (sh_join_) cudaEventDestroy(sh_join_);
    if (dense_) cudaFree(dense_);
    if (experts_) cudaFree(experts_);
    if (state_arena_) cudaFree(state_arena_);
    if (arena_) cudaFree(arena_);
    if (head_logits_) cudaFree(head_logits_);
    if (dhead_) cudaFree(dhead_);
    if (dvocab_) cudaFree(dvocab_);
    void* hosts[] = {h_tok_, h_step_, h_pos_, h_row_, h_out_, h_prob_};
    for (void* h : hosts) if (h) cudaFreeHost(h);
}

const float* MtpDrafter::f32(const char* name) const {
    for (const auto& t : tensors_) if (t.name == name && t.kind == "f32") return (const float*) (dense_ + t.off);
    return nullptr;
}
const uint16_t* MtpDrafter::bf16(const char* name) const {
    for (const auto& t : tensors_) if (t.name == name && t.kind == "bf16") return (const uint16_t*) (dense_ + t.off);
    return nullptr;
}
const void* MtpDrafter::q8(const char* name) const {
    for (const auto& t : tensors_) if (t.name == name && t.kind == "q8_0") return dense_ + t.off;
    return nullptr;
}

bool MtpDrafter::load(const std::string& rt_dir, const ModelGeometry& g, SessionState& ss, int max_t, std::string& err,
<<<<<<< this fork
                      int64_t window, const MtpDrafter* shared) {
#if defined(STRATA_GPU_METAL)
    (void) window; (void) shared;
    g_ = &g;
    ss_ = &ss;
    max_t_ = max_t;
    return strata::metal_engine::mtp_load(rt_dir, max_t, err);
#endif
    cudaGetDevice(&device_);   // a layer split's last stage on another GPU: the drafter lives there
=======
                      int64_t window) {
>>>>>>> their fork (github.com-eddoursul-Strata-custom)
    g_ = &g;
    ss_ = &ss;
    max_t_ = max_t;
    rt_dir_ = rt_dir;
    if (max_t < 1 || max_t > strata::kernels::kVerifyMaxT) { err = "mtp: max_t out of range"; return false; }
    // ---- the index and the dense weights
    {
        std::ifstream idx(rt_dir + "/dense.txt");
        if (!idx) { err = "mtp: cannot open " + rt_dir + "/dense.txt (run tools/mtp_rt.py)"; return false; }
        std::string line;
        while (std::getline(idx, line)) {
            if (line.empty()) continue;
            std::istringstream is(line);
            Tensor t;
            is >> t.name >> t.kind >> t.rows >> t.cols >> t.off >> t.bytes;
            if (!is) { err = "mtp: malformed dense.txt line: " + line; return false; }
            tensors_.push_back(t);
        }
        std::vector<uint8_t> blob;
        if (!read_file(rt_dir + "/dense.bin", blob)) { err = "mtp: cannot read dense.bin"; return false; }
        if (cudaMalloc((void**) &dense_, blob.size()) != cudaSuccess) { err = "mtp: dense weights do not fit"; return false; }
        cudaMemcpy(dense_, blob.data(), blob.size(), cudaMemcpyHostToDevice);
        vram_ += blob.size();
    }
    // ---- the 512 routed experts, one blob each: raw GGUF blocks [gate | up | down] in the formats experts.txt names
    //      (the ggml type ids of gate/up and down), or without it the Q2_0 blob layout of cpu/expert.hpp
    {
        blob_ = strata::kernels::cpu::BLOB;
        if (std::ifstream fmt(rt_dir + "/experts.txt"); fmt) {
            int gu = -1, dn = -1;
            fmt >> gu >> dn;
            if (!strata::kernels::native_expert_supported(gu, dn, g.n_embd, g.n_ff)) {
                err = "mtp: experts.txt names formats the GPU expert kernels lack";
                return false;
            }
            // the fast kernels under either --expert-math: drafts are verified, and both runs of a comparison
            // compute them here
            nat_ = strata::kernels::native_expert_layout(gu, dn, g.n_embd, g.n_ff);
            blob_ = nat_.bytes;
        }
        const uint64_t bytes = (uint64_t) g.n_expert * blob_;
        std::ifstream f(rt_dir + "/experts.bin", std::ios::binary | std::ios::ate);
        if (!f) { err = "mtp: cannot open experts.bin"; return false; }
        if ((uint64_t) f.tellg() != bytes) { err = "mtp: experts.bin does not match its format"; return false; }
        f.seekg(0);
        if (cudaMalloc((void**) &experts_, bytes) != cudaSuccess) { err = "mtp: the 512 experts do not fit in VRAM"; return false; }
        std::vector<uint8_t> chunk(64u << 20);
        for (uint64_t off = 0; off < bytes;) {
            const uint64_t n = std::min<uint64_t>(chunk.size(), bytes - off);
            if (!f.read((char*) chunk.data(), (std::streamsize) n)) { err = "mtp: experts.bin is truncated"; return false; }
            cudaMemcpy(experts_ + off, chunk.data(), n, cudaMemcpyHostToDevice);
            off += n;
        }
        vram_ += bytes;
    }
    const char* required[] = {"fc_embedding.weight", "fc_hidden.weight", "self_attn.q_proj.weight", "self_attn.k_proj.weight",
                              "self_attn.v_proj.weight", "self_attn.o_proj.weight", "mlp.shared_expert.gate_proj.weight",
                              "mlp.shared_expert.up_proj.weight", "mlp.shared_expert.down_proj.weight"};
    for (const char* n : required) if (!q8(n)) { err = std::string("mtp: ") + n + " is missing (q8_0)"; return false; }

    // ---- the layer's own K/V (dense attention: no indexer state is read), FP16 or INT8 (qsa_draft_kv_format)
    const strata::kernels::QsaShapes s = shapes_of(g);
    const int64_t max_cells = ss.qsa_states[0].max_cells;
    const uint64_t sb = qsa_state_bytes(g, max_cells, false, qsa_draft_kv_format());
    if (cudaMalloc(&state_arena_, sb) != cudaSuccess) { err = "mtp: the K/V state does not fit"; return false; }
    if (qsa_state_init(g, max_cells, state_arena_, st_, &ss.qsa_states[0], qsa_draft_kv_format()) == 0) {
        err = "mtp: state init failed";
        return false;
    }
    qsa_state_zero(st_, g, nullptr);
    cudaDeviceSynchronize();
    vram_ += sb;

    // ---- buffers
    window_ = (window > 0 && window < max_cells) ? window : 0;
    cap_ = (((window_ > 0 ? window_ : max_cells) + 63) / 64) * 64;
    attn_scratch_floats_ = (int64_t) strata::kernels::qsa_decode_attn_scratch_floats(cap_, s);
    const uint64_t T = (uint64_t) max_t, N = (uint64_t) g.n_embd, HC = (uint64_t) g.hc, K = (uint64_t) ss.k;
    const uint64_t NH = (uint64_t) g.n_head, HD = (uint64_t) g.head_dim, NKV = (uint64_t) g.n_head_kv;
    const uint64_t R2 = 2 * T;   // step/pos rows: T catch-up rows + up to T-2 chain steps
    bool ok = mapped(T * 4 + 64, (void**) &h_tok_, (void**) &m_tok_) &&
              mapped(R2 * 4 * 4 + 64, (void**) &h_step_, (void**) &m_step_) &&
              mapped(R2 * NH * 4 + 64, (void**) &h_pos_, (void**) &m_pos_) &&
              mapped(64, (void**) &h_row_, (void**) &m_row_) &&
              mapped(T * 4 + 64, (void**) &h_out_, (void**) &m_out_) &&
              mapped(T * 4 + 64, (void**) &h_prob_, (void**) &m_prob_);
    if (!ok) { err = "mtp: mapped staging failed"; return false; }
    auto carve = [&](Bump& b) {
        tok_ = b.take<int32_t>(T); step_ = b.take<int32_t>(R2 * 4); pos_ = b.take<int32_t>(R2 * NH); row_ = b.take<int32_t>(4);
        ident_ = b.take<int32_t>(T * (uint64_t) cap_);
        Rin_ = b.take<float>(T * HC * N); R_ = b.take<float>(T * HC * N);
        emb_ = b.take<float>(T * N); en_ = b.take<float>(T * N); e2_ = b.take<float>(T * N);
        hn_ = b.take<float>(T * HC * N); h2_ = b.take<float>(T * HC * N);
        mixed_ = b.take<float>(T * N); inj_ = b.take<float>(T * HC); inj2_ = b.take<float>(T * HC);
        lo_ = b.take<float>(T * (uint64_t) g.hc_lr); rs_ = b.take<float>(T * HC); bo_ = b.take<float>(T * N);
        grs_ = (float*) b.take<uint8_t>(strata::kernels::fused_gr_scratch_bytes());
        xq_ = b.take<uint8_t>(strata::kernels::native_q8_1_bytes((int) (NH * HD), 8));
        qfull_ = b.take<float>(T * NH * 2 * HD); qcur_ = b.take<float>(T * NH * HD);
        kcur_ = b.take<float>(T * NKV * HD); vcur_ = b.take<float>(T * NKV * HD);
        attn32_ = b.take<float>(T * NH * HD);
        attn_scratch_ = b.take<float>((uint64_t) attn_scratch_floats_);   // the full layer runs one row at a time
        logits_ = b.take<float>(T * (uint64_t) g.n_expert); w_ = b.take<float>(T * K); ids_ = b.take<int32_t>(T * K);
        shared_ = b.take<float>(T * N); parts_ = b.take<float>(T * K * N); y_ = b.take<float>(T * N);
        sample_ = b.take<float>(T * N);
        hit_slot_ = b.take<int32_t>(T * K); hit_dst_ = b.take<int32_t>(T * K); hit_count_ = b.take<int32_t>(4);
        grp_ptr_ = b.take<unsigned long long>(T * K); grp_start_ = b.take<int32_t>(T * K + 1);
        grp_counts_ = b.take<int32_t>(4);
        hit_xq_ = b.take<uint8_t>(T * (N / 32) * 34); hit_xs_ = b.take<float>(T * (N / 32));
        nat_xq_ = b.take<uint8_t>(T * (N / 32) * 36);
        rseq_ = b.take<uint32_t>(4);
        rcount_ = b.take<unsigned>(4);
        hit_scratch_ = b.take<uint8_t>(std::max<uint64_t>(
            strata::kernels::moe_hit_grouped_scratch_bytes((int64_t) (T * K), g.n_embd, g.n_ff),
            strata::kernels::native_expert_scratch_bytes((int64_t) (T * K), g.n_ff)));
        sh_scratch_ = (float*) b.take<uint8_t>(strata::kernels::shared_expert_scratch_bytes(g.n_ff));
        x_bf16_ = b.take<uint16_t>(N);
        out_ids_ = b.take<int32_t>(T + 4);
        probs_ = b.take<float>(T + 4);
        arg_scratch_ = b.take<uint8_t>(strata::kernels::argmax_rows_scratch_bytes((int) T));
        top_scratch_ = b.take<uint8_t>(strata::kernels::row_top_prob_scratch_bytes((int) T));
        dummy_inj_ = b.take<float>(HC);
    };
    Bump count;
    carve(count);
    if (cudaMalloc(&arena_, count.used) != cudaSuccess) { err = "mtp: buffers do not fit"; return false; }
    cudaMemset(arena_, 0, count.used);
    Bump real;
    real.base = (uint8_t*) arena_;
    carve(real);
    vram_ += count.used;
    {
        std::vector<int32_t> id((size_t) (T * (uint64_t) cap_));
        for (uint64_t t = 0; t < T; ++t)
            for (int64_t i = 0; i < cap_; ++i) id[(size_t) (t * (uint64_t) cap_ + (uint64_t) i)] = (int32_t) i;
        cudaMemcpy(ident_, id.data(), id.size() * 4, cudaMemcpyHostToDevice);
    }
<<<<<<< this fork
    // --pipeline-windows 2 (the forcing graphs): the chain shares its card with stage 1's windows and the next launch
    // on stage 0 waits for it, so its stream gets the highest priority there (STRATA_MTP_PRIORITY=0: the default)
    bool prio = false;
#if !defined(STRATA_USE_HIP) && !defined(STRATA_HIP_GFX906) && !defined(STRATA_GPU_METAL)   // (HIP, Metal: the default priority)
    static const bool hi_prio = [] { const char* v = std::getenv("STRATA_MTP_PRIORITY"); return v == nullptr || std::atoi(v) != 0; }();
    int prio_lo = 0, prio_hi = 0;
    if (force_on_ && hi_prio && cudaDeviceGetStreamPriorityRange(&prio_lo, &prio_hi) == cudaSuccess)
        prio = cudaStreamCreateWithPriority(&cs_, cudaStreamNonBlocking, prio_hi) == cudaSuccess;
    cudaGetLastError();
#endif
#if defined(STRATA_USE_HIP) && !defined(STRATA_HIP_GFX906)
    // engine-latency, STRATA_MTP_HIP_PRIO (default on gfx1200/gfx1201, =1 elsewhere opt-in, =0 off): the draft stream at the
    // highest priority.  ROCm hands out hardware queues per priority (GPU_MAX_HW_QUEUES per level), so the drafter gets a
    // queue of its own: its round no longer waits behind the commit graph that the verifier's stream launched just before
    // it (the commit is only needed by the next window, and now runs beside the draft).  R9700, UD-Q4_K_XL: draft wall
    // 5.8 -> 3.5 ms a window (notes-latency.md).
    {
        static const bool hip_prio = [] {
            if (const char* v = std::getenv("STRATA_MTP_HIP_PRIO"); v != nullptr && *v != 0) return std::atoi(v) != 0;
            int dev = 0;
            cudaDeviceProp prop{};
            return cudaGetDevice(&dev) == cudaSuccess && cudaGetDeviceProperties(&prop, dev) == cudaSuccess &&
                   strata::kernels::gfx_arch_is_gfx12(prop.gcnArchName);
        }();
        int lo = 0, hi = 0;
        if (hip_prio && hipDeviceGetStreamPriorityRange(&lo, &hi) == hipSuccess)
            prio = hipStreamCreateWithPriority(&cs_, hipStreamNonBlocking, hi) == hipSuccess;
        (void) cudaGetLastError();
    }
#endif
    if (!prio && cudaStreamCreateWithFlags(&cs_, cudaStreamNonBlocking) != cudaSuccess) { err = "mtp: stream"; return false; }
#if !defined(STRATA_USE_HIP)
    if (cudaStreamCreateWithFlags(&side_, cudaStreamNonBlocking) != cudaSuccess ||
=======
    if (cudaStreamCreateWithFlags(&cs_, cudaStreamNonBlocking) != cudaSuccess ||
        cudaStreamCreateWithFlags(&side_, cudaStreamNonBlocking) != cudaSuccess ||
>>>>>>> their fork (github.com-eddoursul-Strata-custom)
        cudaEventCreateWithFlags(&sh_fork_, cudaEventDisableTiming) != cudaSuccess ||
        cudaEventCreateWithFlags(&sh_join_, cudaEventDisableTiming) != cudaSuccess) {
        err = "mtp: streams";
        return false;
    }
    char fmts[48] = "Q2_0 blobs";
    if (nat_.gu_type >= 0) std::snprintf(fmts, sizeof fmts, "ggml types %d/%d", nat_.gu_type, nat_.d_type);
    std::fprintf(stderr, "strata mtp: draft layer loaded, %.0f MiB of VRAM (experts %.0f as %s, dense %.0f)\n",
                 (double) vram_ / 1048576.0, (double) g.n_expert * blob_ / 1048576.0, fmts,
                 (double) tensors_.back().off / 1048576.0);
    return true;
}

<<<<<<< this fork
namespace {
// STRATA_MTP_FULL_HEAD=1 (a diagnostic): draft over the whole vocabulary (the main head itself) instead of
// rt/draft_vocab.bin's subset - the reference head, to measure what the subset costs in acceptance
bool full_head_env() {
    static const bool on = [] { const char* v = std::getenv("STRATA_MTP_FULL_HEAD"); return v && v[0] == '1'; }();
    return on;
}
}  // namespace

bool MtpDrafter::top2_env() {
    static const bool on = [] { const char* v = std::getenv("STRATA_MTP_TOP2"); return v && v[0] == '1'; }();
    return on;
}

// STRATA_MTP_TOP2=1 (a diagnostic for tree drafts): draft j's runner-up under the draft layer, from the head logits
// the step just wrote (row 0), on the host.  Slow; only for measuring what a second branch would have caught.
void MtpDrafter::record_top2(int j) {
    const int64_t nv = dhead_ != nullptr ? n_dvocab_ : n_vocab_;
    if ((int64_t) lg_host_.size() < nv) lg_host_.resize((size_t) nv);
    if ((int) top2_.size() < max_t_) top2_.assign((size_t) max_t_, -1);
    if (j < 0 || j >= max_t_) return;
    if (cudaMemcpy(lg_host_.data(), head_logits_, (size_t) nv * sizeof(float), cudaMemcpyDeviceToHost) != cudaSuccess) {
        top2_[(size_t) j] = -1;
        return;
    }
    int64_t b1 = -1, b2 = -1;
    for (int64_t i = 0; i < nv; ++i) {
        const float v = lg_host_[(size_t) i];
        if (b1 < 0 || v > lg_host_[(size_t) b1]) { b2 = b1; b1 = i; }
        else if (b2 < 0 || v > lg_host_[(size_t) b2]) b2 = i;
    }
    top2_[(size_t) j] = b2 < 0 ? -1 : dhead_ != nullptr ? dvocab_host_[(size_t) b2] : (int32_t) b2;
}

uint64_t MtpDrafter::bind_bytes(uint64_t head_row_bytes, int64_t n_vocab) const {
    uint64_t bytes = head_logits_ ? 0 : (uint64_t) max_t_ * (uint64_t) n_vocab * sizeof(float);
    if (dhead_ == nullptr && owns_draft_head_ && !full_head_env()) {
        if (FILE* f = std::fopen(vocab_file().c_str(), "rb")) {
            std::fseek(f, 0, SEEK_END);
            const long size = std::ftell(f);
            std::fclose(f);
            if (size >= 4 && size % 4 == 0) bytes += (uint64_t) (size / 4) * head_row_bytes + (uint64_t) size;
        }
    }
    // coupled draft sampling (STRATA_SPEC_COUPLED=1 only): an upper bound - the id -> subset map, the penalty ring,
    // the split scratch (64 lists of 64 entries at most) and the parameters
    if (coupled_draft_env() && cparams_ == nullptr)
        bytes += (uint64_t) n_vocab * 4 + (uint64_t) (kCoupledHistCap + max_t_) * 4 + 64ull * 64ull * 8ull + 4096;
    return bytes;
}

bool MtpDrafter::setup_coupled(std::string& err) {
    const int64_t nv = dhead_ != nullptr ? n_dvocab_ : n_vocab_;
    const size_t scratch = strata::kernels::coupled_draft_scratch_bytes((int) nv);
    if (scratch == 0) {
        std::fprintf(stderr, "strata mtp: STRATA_SPEC_COUPLED: %lld draft logits are too wide for the coupled sampler; "
                             "argmax drafts\n", (long long) nv);
        return true;
    }
    const size_t ring = (size_t) (kCoupledHistCap + max_t_) * sizeof(int32_t);
    if (cudaMalloc((void**) &cparams_, sizeof(strata::kernels::SamplerParams)) != cudaSuccess ||
        cudaMalloc((void**) &cring_, ring) != cudaSuccess || cudaMalloc(&cscratch_, scratch) != cudaSuccess ||
        !mapped(sizeof(strata::kernels::SamplerParams), (void**) &h_cparams_, (void**) &m_cparams_) ||
        !mapped((size_t) kCoupledHistCap * sizeof(int32_t), (void**) &h_chist_, (void**) &m_chist_)) {
        err = "mtp: the coupled draft sampler's buffers do not fit";
        return false;
    }
    if (spec_prob_env() &&
        !mapped((size_t) max_t_ * (size_t) kSpecQStride * sizeof(int32_t), (void**) &h_q_, (void**) &m_q_)) {
        err = "mtp: the draft distributions' buffer does not fit";
        return false;
    }
    cudaMemset(cring_, 0xff, ring);   // -1: no token
    std::fill(h_chist_, h_chist_ + kCoupledHistCap, -1);
    vram_ += sizeof(strata::kernels::SamplerParams) + ring + scratch;
    if (dhead_ != nullptr) {   // token id -> subset index (-1: not in the draft head), for the penalties
        std::vector<int32_t> sub((size_t) n_dvocab_), inv((size_t) n_vocab_, -1);
        cudaMemcpy(sub.data(), dvocab_, sub.size() * sizeof(int32_t), cudaMemcpyDeviceToHost);
        for (size_t i = 0; i < sub.size(); ++i)
            if (sub[i] >= 0 && sub[i] < n_vocab_ && inv[(size_t) sub[i]] < 0) inv[(size_t) sub[i]] = (int32_t) i;
        if (cudaMalloc((void**) &dinv_, inv.size() * sizeof(int32_t)) != cudaSuccess) {
            err = "mtp: the coupled draft sampler's token map does not fit";
            return false;
        }
        cudaMemcpy(dinv_, inv.data(), inv.size() * sizeof(int32_t), cudaMemcpyHostToDevice);
        vram_ += inv.size() * sizeof(int32_t);
    }
    if (cudaDeviceSynchronize() != cudaSuccess) {
        err = std::string("mtp: coupled setup: ") + cudaGetErrorString(cudaGetLastError());
        return false;
    }
    coupled_ok_ = true;
    if (spec_prob_env())
        std::fprintf(stderr, "strata mtp: probabilistic draft acceptance on (STRATA_SPEC_PROB): sampled requests draft by "
                             "sampling the draft head's distribution over %lld tokens; the verifier accepts with "
                             "min(1, p/q) and resamples the residual\n", (long long) nv);
    else
        std::fprintf(stderr, "strata mtp: coupled draft sampling on (STRATA_SPEC_COUPLED): sampled requests draft with the "
                             "target's chain and Philox draw over %lld tokens\n", (long long) nv);
    return true;
}

void MtpDrafter::set_draft_sampling(const strata::kernels::SamplerParams& sp) {
    coupled_active_ = coupled_ok_ && !sp.greedy && sp.temperature > 0.0f && sp.temperature >= spec_min_temp();
    if (!coupled_active_) return;
    *h_cparams_ = sp;   // read by the next round graph (after the previous one has synced)
    std::fill(h_chist_, h_chist_ + kCoupledHistCap, -1);
}

void MtpDrafter::set_draft_history(const int32_t* tail, int64_t n_tail, int32_t next) {
    if (!coupled_active_) return;
    const int h = coupled_hist_len(h_cparams_->penalty_last_n, kCoupledHistCap);
    coupled_hist_base(tail, n_tail, next, h, h_chist_ + (kCoupledHistCap - h));
}

namespace {
// #474: what to change when the draft head's token subset does not fit the VRAM left - a smaller subset that setup
// ships (data/draft_vocab_*.bin; their token counts below), and how to pick it.  The start used to stop at "the draft
// head does not fit" with no hint.  Text only, after the failure: nothing changes for a start that fits.
void draft_head_hint(int64_t n_tokens, int64_t row_bytes) {
    size_t free_b = 0, total_b = 0;
    const bool have_free = cudaMemGetInfo(&free_b, &total_b) == cudaSuccess;
    auto mib = [&](int64_t n) { return (double) (n * row_bytes) / 1048576.0; };
    std::string free_s;
    if (have_free) {
        char b[64];
        std::snprintf(b, sizeof(b), " and %.0f MiB is free", (double) free_b / 1048576.0);
        free_s = b;
    }
    std::fprintf(stderr, "strata mtp: the draft head over %lld tokens needs %.0f MiB of VRAM%s.\n",
                 (long long) n_tokens, mib(n_tokens), free_s.c_str());
    struct Subset { const char* name; int64_t tokens; const char* what; };
    static const Subset smaller[] = {{"cyrillic", 58963, "English, code and the Cyrillic script"},
                                     {"en", 40525, "English and code"}};
    std::string opts;
    for (const Subset& s : smaller)
        if (s.tokens < n_tokens) {
            char b[160];
            std::snprintf(b, sizeof(b), "%s--draft-vocab %s (%s, ~%.0f MiB)", opts.empty() ? "" : " or ", s.name,
                          s.what, mib(s.tokens));
            opts += b;
        }
    if (!opts.empty())
        std::fprintf(stderr, "strata mtp: hint: a smaller draft vocabulary needs less VRAM: %s. Start once with it - "
                             "START-HERE.bat --draft-vocab en (Windows) or ./setup.sh --draft-vocab en - and the model "
                             "keeps it (\"draft_vocab\" in its strata-*.json config); or a smaller --context in "
                             "setup.\n",
                     opts.c_str());
    else
        std::fprintf(stderr, "strata mtp: hint: this is already the smallest shipped draft vocabulary: a smaller "
                             "--context (or closing what else uses the GPU) leaves it room.\n");
}
}  // namespace

bool MtpDrafter::bind(const WeightTable& wt, const NativeHead* head, const float* window_R, std::string& err,
                      const MtpDrafter* shared) {
#if defined(STRATA_GPU_METAL)
    (void) wt; (void) head; (void) window_R; (void) err; (void) shared;
    return true;
#endif
    const OnDevice on_device(device_);
=======
bool MtpDrafter::bind(const WeightTable& wt, const NativeHead* head, const float* window_R, std::string& err) {
>>>>>>> their fork (github.com-eddoursul-Strata-custom)
    wt_ = &wt;
    head_ = head;
    window_R_ = window_R;
    const WeightRef* wo = wt.find("output.weight");
    if (!wo) { err = "mtp: output.weight is missing"; return false; }
    n_vocab_ = wo->ne1;
    if (head == nullptr || !head->loaded()) { err = "mtp: the draft layer needs the native head (--native)"; return false; }
    if (dhead_ == nullptr && !bind_head(head, err)) return false;
    if (dhead_ == nullptr && head->rows() != n_vocab_) {
        err = "mtp: the main head holds part of the vocabulary and the draft layer has no draft head of its own";
        return false;
    }
    if (head_logits_ == nullptr &&
        cudaMalloc((void**) &head_logits_, (size_t) max_t_ * (size_t) n_vocab_ * sizeof(float)) != cudaSuccess) {
        err = "mtp: the draft logits do not fit";
        return false;
    }
    return true;
}

bool MtpDrafter::bind_head(const NativeHead* head, std::string& err) {
    if (head == nullptr || !head->loaded()) {
        err = "mtp: the draft layer needs the native head (--native)";
        return false;
    }
    // the draft head's token subset, when tools/draft_vocab.py wrote one
    if (dhead_ == nullptr) {
        std::vector<uint8_t> raw;
        if (read_file(rt_dir_ + "/draft_vocab.bin", raw) && raw.size() >= 4 && raw.size() % 4 == 0) {
            n_dvocab_ = (int64_t) (raw.size() / 4);
            const int64_t row_bytes = (int64_t) head->row_bytes();   // a vocabulary row of the native head
            if (cudaMalloc((void**) &dvocab_, raw.size()) != cudaSuccess ||
                cudaMalloc((void**) &dhead_, (size_t) (n_dvocab_ * row_bytes)) != cudaSuccess) {
                err = "mtp: the draft head does not fit";
                return false;
            }
            cudaMemcpy(dvocab_, raw.data(), raw.size(), cudaMemcpyHostToDevice);
<<<<<<< this fork
            dvocab_host_.resize((size_t) n_dvocab_);
            std::memcpy(dvocab_host_.data(), raw.data(), raw.size());
            const int head_dev = head->device();
            if (head_dev >= 0 && head_dev != device_) {
                // the main head is on another card (the drafter on the first stage of a layer split, 0.1.44): the rows are
                // gathered where the head is and the subset copied over once
                void* tmp = nullptr;
                int32_t* dv = nullptr;
                bool ok_x = true;
                {
                    const OnDevice on_head(head_dev);
                    ok_x = cudaMalloc((void**) &tmp, (size_t) (n_dvocab_ * row_bytes)) == cudaSuccess &&
                           cudaMalloc((void**) &dv, raw.size()) == cudaSuccess &&
                           cudaMemcpy(dv, raw.data(), raw.size(), cudaMemcpyHostToDevice) == cudaSuccess;
                    if (ok_x) {
                        strata::kernels::gather_rows((const uint8_t*) head->weights(), row_bytes, dv, n_dvocab_,
                                                     (uint8_t*) tmp, nullptr);
                        ok_x = cudaDeviceSynchronize() == cudaSuccess;
                    }
                }
                ok_x = ok_x && cudaMemcpy(dhead_, tmp, (size_t) (n_dvocab_ * row_bytes), cudaMemcpyDeviceToDevice) == cudaSuccess;
                {
                    const OnDevice on_head(head_dev);
                    if (tmp) cudaFree(tmp);
                    if (dv) cudaFree(dv);
                }
                if (!ok_x) { err = "mtp: the draft head's rows could not be copied from the head's card"; return false; }
            } else {
                strata::kernels::gather_rows((const uint8_t*) head->weights(), row_bytes, dvocab_, n_dvocab_, dhead_, nullptr);
            }
=======
            strata::kernels::gather_rows((const uint8_t*) head->weights(), row_bytes, dvocab_, n_dvocab_, dhead_, nullptr);
>>>>>>> their fork (github.com-eddoursul-Strata-custom)
            cudaDeviceSynchronize();
            vram_ += (uint64_t) (n_dvocab_ * row_bytes) + raw.size();
            std::fprintf(stderr, "strata mtp: draft head over %lld tokens (%.1f MiB)\n", (long long) n_dvocab_,
                         (double) (n_dvocab_ * row_bytes) / 1048576.0);
        }
    }
    return true;
}

// STRATA_MTP_PROFILE=1 (measurement only): GPU stamps between the draft layer's stages in the round graph and in every
// step graph (each stamp is one more node, so the totals run a little high); the per-stage averages are printed every
// 300 rounds with STRATA_DRAFT_TIMING's line.  Off: no stamp is recorded and nothing changes.
namespace {
constexpr int kMpSlots = 24;
constexpr int kMpGraphs = 17;   // the round graph (0) and the steps 1..16
const char* const kMpNames[kMpSlots] = {"-", "embed+fc_emb", "hnorm+fc_hidden+add", "hc-read attn", "k/v+append",
                                        "q+norm-rope", "attention", "gate+o_proj", "hc-read mlp", "router",
                                        "experts", "shared", "combine+write", "hc-read mixer", "head gemv",
                                        "argmax+prob", "select", "", "", "", "", "", "", ""};
struct MtpProf {
    bool on = false;
    unsigned long long* d = nullptr;
    std::vector<unsigned long long> h;
    int base = -1;   // the graph being captured (-1: none, no stamps)
    double sum[2][kMpSlots] = {};
    int64_t n[2] = {0, 0};
    MtpProf() {
        const char* e = std::getenv("STRATA_MTP_PROFILE");
        on = e != nullptr && e[0] == '1';
    }
    void init() {
        if (!on || d != nullptr) return;
        if (cudaMalloc((void**) &d, (size_t) kMpSlots * kMpGraphs * 8) != cudaSuccess) { on = false; d = nullptr; (void) cudaGetLastError(); return; }
        cudaMemset(d, 0, (size_t) kMpSlots * kMpGraphs * 8);
        h.assign((size_t) kMpSlots * kMpGraphs, 0);
    }
    void stamp(int i, cudaStream_t cs) const {
        if (on && d != nullptr && base >= 0) strata::kernels::gpu_stamp(d, base * kMpSlots + i, cs);
    }
    void collect(int n_steps_run) {   // after the round's sync: the round graph and steps 1..n_steps_run-1
        if (!on || d == nullptr) return;
        cudaMemcpy(h.data(), d, h.size() * 8, cudaMemcpyDeviceToHost);
        for (int gi = 0; gi < n_steps_run && gi < kMpGraphs; ++gi) {
            const unsigned long long* s = h.data() + (size_t) gi * kMpSlots;
            const int k = gi == 0 ? 0 : 1;
            unsigned long long prev = s[0];
            if (prev == 0) continue;
            for (int i = 1; i < kMpSlots; ++i) {
                if (s[i] == 0 || s[i] < prev) continue;
                sum[k][i] += (double) (s[i] - prev);
                prev = s[i];
            }
            ++n[k];
        }
    }
    void report() {
        for (int k = 0; k < 2; ++k) {
            if (n[k] == 0) continue;
            std::string out;
            char b[96];
            double tot = 0;
            for (int i = 1; i < kMpSlots; ++i) {
                if (sum[k][i] <= 0) continue;
                tot += sum[k][i];
                std::snprintf(b, sizeof b, " %s %.3f", kMpNames[i], sum[k][i] / 1e6 / (double) n[k]);
                out += b;
            }
            std::fprintf(stderr, "strata mtp profile (%s, ms per graph over %lld):%s | total %.3f\n", k == 0 ? "round" : "step",
                         (long long) n[k], out.c_str(), tot / 1e6 / (double) n[k]);
            for (double& x : sum[k]) x = 0;
            n[k] = 0;
        }
    }
};
MtpProf g_mp;

// STRATA_VERIFY_NODES=1 also lists the draft graphs' nodes (kernel names on HIP; counts on CUDA)
void mtp_list_nodes(cudaGraph_t graph, const char* what) {
    if (std::getenv("STRATA_VERIFY_NODES") == nullptr || graph == nullptr) return;
#if defined(STRATA_GPU_METAL)
    (void) what;    // Metal has no graph nodes to list
#else
#if defined(STRATA_USE_HIP)
    using NodeT = hipGraphNode_t;
    using NodeTypeT = hipGraphNodeType;
#define MTP_GET_NODES hipGraphGetNodes
#define MTP_NODE_TYPE hipGraphNodeGetType
#else
    using NodeT = cudaGraphNode_t;
    using NodeTypeT = cudaGraphNodeType;
#define MTP_GET_NODES cudaGraphGetNodes
#define MTP_NODE_TYPE cudaGraphNodeGetType
#endif
    size_t nn = 0;
    MTP_GET_NODES(graph, nullptr, &nn);
    std::vector<NodeT> nodes(nn);
    MTP_GET_NODES(graph, nodes.data(), &nn);
    std::vector<std::pair<std::string, int>> kinds;
    for (NodeT nd : nodes) {
        NodeTypeT ty;
        MTP_NODE_TYPE(nd, &ty);
        std::string name = "type" + std::to_string((int) ty);
#if defined(STRATA_USE_HIP)
        if (ty == hipGraphNodeTypeKernel) {
            hipKernelNodeParams kp{};
            if (hipGraphKernelNodeGetParams(nd, &kp) == hipSuccess && kp.func != nullptr) {
                const char* fn = hipKernelNameRefByPtr(kp.func, nullptr);
                if (fn != nullptr) name = fn;
            }
        }
#else
        if (ty == cudaGraphNodeTypeKernel) name = "kernel";
#endif
        bool found = false;
        for (auto& k : kinds)
            if (k.first == name) { ++k.second; found = true; break; }
        if (!found) kinds.push_back({name, 1});
    }
#undef MTP_GET_NODES
#undef MTP_NODE_TYPE
    std::fprintf(stderr, "strata mtp: the %s graph has %zu nodes:", what, nn);
    for (auto& k : kinds) std::fprintf(stderr, "\n %4d x %.110s", k.second, k.first.c_str());
    std::fprintf(stderr, "\n");
#endif
}
}  // namespace

// The layer for T rows.  full = false stops after the K/V append (the prompt only needs the cache).
bool MtpDrafter::record_forward(int T, int step_row0, cudaStream_t cs, std::string& err) {
    // step_row0 >= 0: the full layer on step rows [step_row0, +T); step_row0 < 0: K/V only on rows [-1 - step_row0, +T)
    const bool full = step_row0 >= 0;
    if (full && T != 1) { err = "mtp: the full layer runs one row at a time (its attention scratch is sized for one)"; return false; }
    if (!record_front(T, full ? step_row0 : -1 - step_row0, cs, err)) return false;
    return !full || record_rest(step_row0, cs, err);
}

bool MtpDrafter::record_front(int T, int row0, cudaStream_t cs, std::string& err) {
    using namespace strata::kernels;
    const ModelGeometry& g = *g_;
    const int64_t N = g.n_embd, HC = g.hc, NH = g.n_head, HD = g.head_dim, NKV = g.n_head_kv;
    const QsaShapes s = shapes_of(g);
    const int32_t* step = step_ + row0 * 4;
    const int32_t* pos = pos_ + row0 * NH;
    try {
        // ---- the two input branches
<<<<<<< this fork
        const WeightRef* we = wt_->find("token_embd.weight");
        if (!we) { err = "mtp: token_embd.weight is missing"; return false; }
        g_mp.stamp(0, cs);
        if (const NativeEmbed* ne = native_embed()) {   // plan v0.3 P6: the GGUF-form table
            ne->gather_dev(tok_, T, emb_, cs);
        } else {
            const auto* codes = (const uint8_t*) we->data;
            const auto* scales = (const float*) (codes + we->codes_bytes);
            const auto* offsets = we->has_offset ? (const float*) (codes + we->codes_bytes + we->scales_bytes) : nullptr;
            embedding_gather_dev(codes, scales, offsets, tok_, T, we->ne0, we->code_bits, we->code_bias, we->group_elems,
                                 (uint64_t) (we->ne0 / (8 / we->code_bits)), (uint64_t) (we->ne0 / we->group_elems), emb_, cs);
        }
        native_qsa_rms_norm_weighted(emb_, f32("pre_fc_norm_embedding.weight"), en_, (int) N, T, EPS, cs);
        native_quantize_q8_1(en_, xq_, (int) N, T, cs);
        native_mmvq(wt_fc_embedding, wp_fc_embedding, xq_, e2_, (int) N, (int) N, T, cs);
        g_mp.stamp(1, cs);
        if (hnorm_stream_)   // --mtp-hnorm stream: one RMS per stream, each scaled by its slice of the weight
            native_qsa_rms_norm_grouped(Rin_, f32("pre_fc_norm_hidden.weight"), hn_, (int) N, (int) HC, (int) (T * HC), EPS, cs);
        else
            native_qsa_rms_norm_weighted(Rin_, f32("pre_fc_norm_hidden.weight"), hn_, (int) (HC * N), T, EPS, cs);
=======
        if (!embed_rows(*wt_, g, tok_, T, emb_, cs, err)) return false;
        native_qsa_rms_norm_weighted(emb_, f32("pre_fc_norm_embedding.weight"), en_, (int) N, T, EPS, cs);
        native_quantize_q8_1(en_, xq_, (int) N, T, cs);
        native_mmvq(GGML_Q8_0, q8("fc_embedding.weight"), xq_, e2_, (int) N, (int) N, T, cs);
        native_qsa_rms_norm_weighted(Rin_, f32("pre_fc_norm_hidden.weight"), hn_, (int) (HC * N), T, EPS, cs);
>>>>>>> their fork (github.com-eddoursul-Strata-custom)
        for (int c0 = 0; c0 < T * HC; c0 += 8) {
            const int nc = (int) std::min<int64_t>(8, T * HC - c0);
            native_quantize_q8_1(hn_ + (size_t) c0 * N, xq_, (int) N, nc, cs);
            native_mmvq(GGML_Q8_0, q8("fc_hidden.weight"), xq_, h2_ + (size_t) c0 * N, (int) N, (int) N, nc, cs);
        }
        add_streams_broadcast(h2_, e2_, R_, N, (int) HC, T, cs);
        g_mp.stamp(2, cs);
        // ---- the attention hyper-connection
        {
            FusedGrArgs fa[kFusedGrMaxT];
            for (int t = 0; t < T; ++t) {
                fa[t].R = R_ + (size_t) t * HC * N; fa[t].R_out = R_ + (size_t) t * HC * N; fa[t].apply = false;
                fa[t].w_norm = f32("attn_hyper_connection.hc_norm.weight");
                fa[t].w_down = bf16("attn_hyper_connection.input_mix_weight_down.weight");
                fa[t].w_up = bf16("attn_hyper_connection.input_mix_weight_up.weight");
                fa[t].w_inject = bf16("attn_hyper_connection.block_inject_weight.weight");
                fa[t].eps = EPS; fa[t].lo = lo_ + t * g.hc_lr; fa[t].rs = rs_ + t * HC;
                fa[t].inject_out = inj_ + t * HC; fa[t].mixed = mixed_ + t * N;
            }
            fused_gr_read_multi(fa, T, grs_, cs);
        }
        g_mp.stamp(3, cs);
        // ---- attention: K/V into the layer's own cache
        native_quantize_q8_1(mixed_, xq_, (int) N, T, cs);
        native_mmvq(GGML_Q8_0, q8("self_attn.k_proj.weight"), xq_, kcur_, (int) N, (int) (NKV * HD), T, cs);
        native_mmvq(GGML_Q8_0, q8("self_attn.v_proj.weight"), xq_, vcur_, (int) N, (int) (NKV * HD), T, cs);
        for (int t = 0; t < T; ++t) {
            norm_rope(kcur_ + t * NKV * HD, f32("self_attn.k_norm.weight"), (int) NKV, (int) HD, pos + t * NH, cs);
            if (st_.kv_int8)
                kv_append_q8_step(st_.k_q, st_.v_q, st_.k_scale, st_.v_scale, st_.page_table, step + t * 4,
                                  kcur_ + t * NKV * HD, vcur_ + t * NKV * HD, s, cs);
            else
                kv_append_step(st_.k_pool, st_.v_pool, st_.page_table, step + t * 4, kcur_ + t * NKV * HD,
                               vcur_ + t * NKV * HD, s, cs);
        }
<<<<<<< this fork
        // stored in the state's own format (#293 appended rotated INT8 K/V as Q4_0, into pools INT8 never has)
        if (!no_batch_kv && st_.kv_q4)
            kv_append_q4_steps(st_.k_q4, st_.v_q4, st_.page_table, step, 4, T, kcur_, vcur_, s, cs, &st_.host);
        else if (!no_batch_kv && st_.kv_int8)
            kv_append_q8_steps(st_.k_q, st_.v_q, st_.k_scale, st_.v_scale, st_.page_table, step, 4, kcur_, vcur_,
                               (int) (NKV * HD), T, s, cs, &st_.host);
        else
            for (int t = 0; t < T; ++t) {
                if (st_.kv_q4)
                    kv_append_q4_step(st_.k_q4, st_.v_q4, st_.page_table, step + t * 4, kcur_ + t * NKV * HD,
                                      vcur_ + t * NKV * HD, s, cs, &st_.host);
                else if (st_.kv_int8)
                    kv_append_q8_step(st_.k_q, st_.v_q, st_.k_scale, st_.v_scale, st_.page_table, step + t * 4,
                                      kcur_ + t * NKV * HD, vcur_ + t * NKV * HD, s, cs, &st_.host);
                else
                    kv_append_step(st_.k_pool, st_.v_pool, st_.page_table, step + t * 4, kcur_ + t * NKV * HD,
                                   vcur_ + t * NKV * HD, s, cs, &st_.host);
            }
        g_mp.stamp(4, cs);
=======
>>>>>>> their fork (github.com-eddoursul-Strata-custom)
    } catch (const std::exception& e) {
        err = std::string("mtp: ") + e.what();
        return false;
    }
    return true;
}

void MtpDrafter::norm_rope(float* data, const float* gamma, int rows, int cols, const int32_t* p, cudaStream_t cs) {
    using namespace strata::kernels;
    const QsaShapes s = shapes_of(*g_);
    native_qsa_rms_norm_weighted(data, gamma, data, cols, rows, EPS, cs);
    if (native_rope_enabled()) native_rope_apply(data, data, rows, cols, (int) s.n_rot, (float) qsa_freq_base(), p, cs);
    else rope_neox_apply(data, data, rows, cols, (int) s.n_rot, st_.cos_tab, st_.sin_tab, p, cs);
}

bool MtpDrafter::record_rest(int step_row, cudaStream_t cs, std::string& err) {
    using namespace strata::kernels;
    const ModelGeometry& g = *g_;
    SessionState& ss = *ss_;
    const int T = 1;
    const int64_t N = g.n_embd, HC = g.hc, K = ss.k, NH = g.n_head, HD = g.head_dim;
    const QsaShapes s = shapes_of(g);
    const GrShapes gs{g.n_embd, g.hc, g.hc_lr};
    const int32_t* step = step_ + step_row * 4;
    const int32_t* pos = pos_ + step_row * NH;
    try {
        // ---- dense attention over every cell
        native_mmvq(GGML_Q8_0, q8("self_attn.q_proj.weight"), xq_, qfull_, (int) N, (int) (NH * 2 * HD), T, cs);
        for (int t = 0; t < T; ++t) {
            float* qc = qcur_ + t * NH * HD;
            if (cudaMemcpy2DAsync(qc, (size_t) HD * 4, qfull_ + t * NH * 2 * HD, (size_t) HD * 2 * 4, (size_t) HD * 4,
                                  (size_t) NH, cudaMemcpyDeviceToDevice, cs) != cudaSuccess) {
                err = "mtp: q split failed";
                return false;
            }
            norm_rope(qc, f32("self_attn.q_norm.weight"), (int) NH, (int) HD, pos + t * NH, cs);
        }
<<<<<<< this fork
        if (st_.kv_rot) fwht256_inplace_cuda(qcur_, (int64_t) T * NH, cs);
        g_mp.stamp(5, cs);
        const QsaAttnPools pools = qsa_attn_pools(st_);
        if (window_ > 0) window_ids(const_cast<int32_t*>(step), T, (int) window_, ident_, cap_, cs);
        qsa_decode_attn_batch(qcur_, pools, ident_, step, cap_, s, attn_scratch_, attn_, T, cs);
        if (st_.kv_rot) fwht256_inplace_cuda(attn_, (int64_t) T * NH, cs);
        g_mp.stamp(6, cs);
        native_qsa_gate_apply(attn_, qfull_, attn32_, (int) (T * NH), (int) HD, cs);
        native_quantize_q8_1(attn32_, xq_, (int) (NH * HD), T, cs);
        native_mmvq(wt_o_proj, wp_o_proj, xq_, bo_, (int) (NH * HD), (int) N, T, cs);
        g_mp.stamp(7, cs);
=======
        QsaAttnPools pools;
        pools.page_table = st_.page_table;
        if (st_.kv_int8) { pools.k_q = st_.k_q; pools.v_q = st_.v_q; pools.k_scale = st_.k_scale; pools.v_scale = st_.v_scale; }
        else { pools.k_pool = st_.k_pool; pools.v_pool = st_.v_pool; }
        if (window_ > 0) window_ids(const_cast<int32_t*>(step), T, (int) window_, ident_, cap_, cs);
        qsa_decode_attn_batch(qcur_, pools, ident_, step, cap_, s, attn_scratch_, attn32_, T, cs, qfull_);
        native_quantize_q8_1(attn32_, xq_, (int) (NH * HD), T, cs);
        native_mmvq(GGML_Q8_0, q8("self_attn.o_proj.weight"), xq_, bo_, (int) (NH * HD), (int) N, T, cs);
>>>>>>> their fork (github.com-eddoursul-Strata-custom)
        // ---- the MLP hyper-connection (the attention write folded in)
        {
            FusedGrArgs fa[kFusedGrMaxT];
            for (int t = 0; t < T; ++t) {
                fa[t].R = R_ + (size_t) t * HC * N; fa[t].R_out = R_ + (size_t) t * HC * N; fa[t].apply = true;
                fa[t].bo_prev = bo_ + t * N; fa[t].inj_prev = inj_ + t * HC;
                fa[t].w_norm = f32("mlp_hyper_connection.hc_norm.weight");
                fa[t].w_down = bf16("mlp_hyper_connection.input_mix_weight_down.weight");
                fa[t].w_up = bf16("mlp_hyper_connection.input_mix_weight_up.weight");
                fa[t].w_inject = bf16("mlp_hyper_connection.block_inject_weight.weight");
                fa[t].eps = EPS; fa[t].lo = lo_ + t * g.hc_lr; fa[t].rs = rs_ + t * HC;
                fa[t].inject_out = inj2_ + t * HC; fa[t].mixed = mixed_ + t * N;
            }
            fused_gr_read_multi(fa, T, grs_, cs);
        }
<<<<<<< this fork
        g_mp.stamp(8, cs);
        // ---- MoE: router, the 512 resident experts, the shared expert (on a branch beside them on CUDA), the
        // combine, the write
=======
        // ---- MoE: the shared expert on a branch beside the router and the 512 resident experts, then the combine
        // and the write
>>>>>>> their fork (github.com-eddoursul-Strata-custom)
        NativeSharedWeights nsw;
        nsw.gate_type = GGML_Q8_0; nsw.gate_data = q8("mlp.shared_expert.gate_proj.weight");
        nsw.up_type = GGML_Q8_0; nsw.up_data = q8("mlp.shared_expert.up_proj.weight");
        nsw.down_type = GGML_Q8_0; nsw.down_data = q8("mlp.shared_expert.down_proj.weight");
        nsw.q8_1 = xq_;   // the routed experts read nat_xq_ / hit_xq_
        const SForm none{};
        if (cudaEventRecord(sh_fork_, cs) != cudaSuccess || cudaStreamWaitEvent(side_, sh_fork_, 0) != cudaSuccess) {
            err = "mtp: the shared expert's branch";
            return false;
        }
        for (int t = 0; t < T; ++t) {
            f32_to_bf16_bulk(mixed_ + t * N, x_bf16_, N, side_);
            shared_expert(nullptr, nullptr, x_bf16_, none, nullptr, nullptr, nullptr, none, nullptr, nullptr, nullptr, none,
                          nullptr, nullptr, nullptr, bf16("mlp.shared_expert_gate.weight"), sh_scratch_, shared_ + t * N,
                          N, g.n_ff, 32, side_, mixed_ + t * N, &nsw);
        }
        if (cudaEventRecord(sh_join_, side_) != cudaSuccess) { err = "mtp: the shared expert's branch"; return false; }
        bool xq1 = false;   // the router kernel wrote the native experts' q8_1 rows
        if (native_router_enabled()) {
            VerifyRouterArgs ra;
            ra.x = mixed_; ra.w = bf16("mlp.gate.weight"); ra.logits = logits_; ra.ids = ids_; ra.weights = w_;
            ra.xq1 = nat_.gu_type >= 0 ? nat_xq_ : nullptr;
            ra.seq = rseq_; ra.counter = rcount_;
            ra.n_tok = T; ra.n_embd = (int) N; ra.n_expert = (int) g.n_expert;
            verify_router(ra, cs);
            xq1 = ra.xq1 != nullptr;
        } else {
            for (int t = 0; t < T; ++t) {
                bf16_gemv_fp32_mmvf(mixed_ + t * N, bf16("mlp.gate.weight"), logits_ + t * g.n_expert, (int) N,
                                    (int) g.n_expert, cs);
                router_top10(logits_ + t * g.n_expert, 1, (int) g.n_expert, (int) K, ids_ + t * K, w_ + t * K, cs);
            }
        }
        moe_group_resident(ids_, (int) (T * K), (int) K, experts_, (int64_t) blob_, grp_ptr_, grp_start_, grp_counts_,
                           hit_dst_, hit_slot_, cs);
        if (nat_.gu_type >= 0) {
            if (!xq1) quantize_q8_1_rows(mixed_, T, N, nat_xq_, cs);
            native_expert_grouped(nat_, grp_ptr_, grp_start_, grp_counts_, hit_dst_, hit_slot_, (int64_t) T * K,
                                  (int64_t) T * K, nat_xq_, (int64_t) (N / 32) * 36, hit_scratch_, parts_, cs);
        } else {
            quantize_q8_0_scaled(mixed_, hit_xq_, hit_xs_, (int64_t) T * N, cs);
            moe_grouped_s2(grp_ptr_, grp_start_, grp_counts_, hit_dst_, hit_slot_, (int64_t) T * K, (int64_t) T * K,
                           hit_xq_, hit_xs_, hit_scratch_, parts_, cs);
        }
<<<<<<< this fork
        g_mp.stamp(9, cs);
        moe_group_resident(ids_, (int) (T * K), (int) K, experts_, (int64_t) strata::kernels::cpu::BLOB, grp_ptr_,
                           grp_start_, grp_counts_, hit_dst_, hit_slot_, cs);
        quantize_q8_0_scaled(mixed_, hit_xq_, hit_xs_, (int64_t) T * N, cs);
        moe_grouped_s2(grp_ptr_, grp_start_, grp_counts_, hit_dst_, hit_slot_, (int64_t) T * K, (int64_t) T * K, hit_xq_,
                       hit_xs_, hit_scratch_, parts_, cs);
        g_mp.stamp(10, cs);
        if (branch_on) {
            if (cudaStreamWaitEvent(cs, sh_join_, 0) != cudaSuccess) { err = "mtp: the shared expert's join"; return false; }
        } else {
            shared_rows();
        }
        g_mp.stamp(11, cs);
        static const bool no_multi_gr = [] {   // #783 PR-g: STRATA_NO_MULTI_GR=1 keeps the per-token GR calls
            const char* v = std::getenv("STRATA_NO_MULTI_GR");
            return v != nullptr && v[0] != '\0' && v[0] != '0';
        }();
=======
        if (cudaStreamWaitEvent(cs, sh_join_, 0) != cudaSuccess) { err = "mtp: the shared expert's join"; return false; }
>>>>>>> their fork (github.com-eddoursul-Strata-custom)
        for (int t = 0; t < T; ++t) {
            if (native_moe_combine_enabled())
                native_moe_combine(parts_ + (size_t) t * K * N, w_ + t * K, shared_ + t * N, y_ + t * N, N, K, cs);
            else
                moe_combine(parts_ + (size_t) t * K * N, w_ + t * K, shared_ + t * N, y_ + t * N, N, K, cs);
            gr_write(R_ + (size_t) t * HC * N, y_ + t * N, inj2_ + t * HC, gs, R_ + (size_t) t * HC * N, cs);
        }
<<<<<<< this fork
        if (T > 1 && native_moe_combine_enabled())
            native_moe_combine_multi(parts_, w_, shared_, y_, N, K, T, cs);
        // #783 PR-g (stuchapin909): the T residual writes in one launch (each token's R, block output and injection are
        // its own, so doing them after the loop changes nothing)
        if (!fuse_head_gr && !no_multi_gr)
            gr_write_multi(R_, y_, inj2_, gs, R_, T, cs);
        g_mp.stamp(12, cs);
=======
>>>>>>> their fork (github.com-eddoursul-Strata-custom)
        // ---- the final mixer and the main model's head
        {
            FusedGrArgs fa[kFusedGrMaxT];
            for (int t = 0; t < T; ++t) {
                fa[t].R = R_ + (size_t) t * HC * N; fa[t].R_out = R_ + (size_t) t * HC * N; fa[t].apply = false;
                fa[t].w_norm = f32("hyper_connection_mixer.hc_norm.weight");
                fa[t].w_down = bf16("hyper_connection_mixer.input_mix_weight_down.weight");
                fa[t].w_up = bf16("hyper_connection_mixer.input_mix_weight_up.weight");
                fa[t].eps = EPS; fa[t].lo = lo_ + t * g.hc_lr; fa[t].rs = rs_ + t * HC; fa[t].mixed = sample_ + t * N;
            }
            fused_gr_read_multi(fa, T, grs_, cs);
        }
        g_mp.stamp(13, cs);
        native_quantize_q8_1(sample_, xq_, (int) N, T, cs);
        const bool sub = dhead_ != nullptr;
        const int64_t nv = sub ? n_dvocab_ : n_vocab_;
<<<<<<< this fork
        native_mmvq(sub ? dhead_type_ : head_->type(), sub ? dhead_ : head_->weights(), xq_, head_logits_, (int) N,
                    (int) nv, T, cs);
        g_mp.stamp(14, cs);
        if (coupled_rec_) {
            // coupled draft sampling: the target's chain and Philox draw for the row that will verify this draft
            // (counter = this cell + 1, from its step record), penalties over the ring; T == 1 (full layer)
            if (m_q_ != nullptr)   // STRATA_SPEC_PROB: a draw from q with its own stream, and q's list for the verifier
                spec_draft_sample(head_logits_, (int) nv, sub ? dvocab_ : nullptr, sub ? dinv_ : nullptr, (int) n_vocab_,
                                  cparams_, cring_, kCoupledHistCap, coupled_j_, step, cscratch_, out_ids_, probs_, m_q_,
                                  spec_gate_pick(), spec_draft_temp_scale(), cs);
            else
                coupled_draft_sample(head_logits_, (int) nv, sub ? dvocab_ : nullptr, sub ? dinv_ : nullptr, (int) n_vocab_,
                                     cparams_, cring_, kCoupledHistCap, coupled_j_, step, cscratch_, out_ids_, probs_, cs);
            return true;
        }
        if (argmax_rows_wanted()) {
            argmax_rows(head_logits_, T, (int) nv, arg_scratch_, out_ids_, cs);
        } else {
            SamplerParams sp;
            sp.greedy = true;
            sp.temperature = 0.0f;
            sample_tokens(head_logits_, T, (int) nv, nullptr, 0, sp, out_ids_, cs);
        }
        if (multi_block_head_ops()) row_top_prob_split(head_logits_, T, (int) nv, out_ids_, probs_, top_scratch_, cs);
        else row_top_prob(head_logits_, T, (int) nv, out_ids_, probs_, cs);
=======
        native_mmvq(head_->type(), sub ? dhead_ : head_->weights(), xq_, head_logits_, (int) N, (int) nv, T, cs);
        argmax_rows(head_logits_, T, (int) nv, arg_scratch_, out_ids_, cs);
        row_top_prob_split(head_logits_, T, (int) nv, out_ids_, probs_, top_scratch_, cs);
>>>>>>> their fork (github.com-eddoursul-Strata-custom)
        if (sub) map_ids(out_ids_, dvocab_, T, cs);
        g_mp.stamp(15, cs);
    } catch (const std::exception& e) {
        err = std::string("mtp: ") + e.what();
        return false;
    }
    return true;
}

namespace {
bool finish_capture(cudaStream_t cs, bool ok, cudaGraphExec_t& exec, const char* what, std::string& err) {
    cudaGraph_t graph = nullptr;
    const cudaError_t ce = cudaStreamEndCapture(cs, &graph);
    if (!ok) {
        if (graph) cudaGraphDestroy(graph);
        return false;
    }
    if (ce != cudaSuccess || cudaGraphInstantiate(&exec, graph, 0) != cudaSuccess) {
        if (graph) cudaGraphDestroy(graph);
        err = std::string("mtp: ") + what + " capture: " + cudaGetErrorString(ce);
        return false;
    }
    mtp_list_nodes(graph, what);
    cudaGraphDestroy(graph);
    // an explicit upload: the first launch's implicit one blocked behind a device-side spin (verify.cpp)
    cudaGraphUpload(exec, cs);
    cudaStreamSynchronize(cs);
    return true;
}
}  // namespace

bool MtpDrafter::capture_round(int T, std::string& err) {
    if (round_exec_[T]) return true;
    using namespace strata::kernels;
    const int64_t HCN = g_->hc * g_->n_embd;
    g_mp.init();   // (its allocation before the capture)
    if (cudaStreamBeginCapture(cs_, cudaStreamCaptureModeThreadLocal) != cudaSuccess) { err = "mtp: begin capture"; return false; }
    g_mp.base = 0;
    bool ok = true;
    // the catch-up: the layer's front for the window's T cells (their K/V), then its rest for row a only, on row
    // a's intermediates copied to row 0, at the cell the host staged in step row 2*max_t - 1; the draft chain is
    // one graph per step (`capture_step`) so the host can stop it when a draft is unlikely
    const int ra = 2 * max_t_ - 1;
    const int64_t N = g_->n_embd, NH = g_->n_head;
    const MappedCopy in[7] = {{tok_, m_tok_, T}, {step_, m_step_, (int64_t) 2 * T * 4}, {pos_, m_pos_, 2 * T * NH},
                              {row_, m_row_, 2}, {Rin_, window_R_, T * HCN}, {step_ + ra * 4, m_step_ + ra * 4, 4},
                              {pos_ + ra * NH, m_pos_ + ra * NH, NH}};
    copy_from_mapped_multi(in, 7, cs_);
    ok = record_front(T, 0, cs_, err);
    if (ok) {
        if (T > 1) {
            copy_row_to_first(row_, R_, HCN, inj_, g_->hc, mixed_, N, cs_);
            native_quantize_q8_1(mixed_, xq_, (int) N, 1, cs_);
        }
        ok = record_rest(ra, cs_, err);
    }
    if (ok) mtp_select(R_, HCN, out_ids_, row_ + 1, Rin_, tok_, m_out_, 0, cs_, probs_, m_prob_);
<<<<<<< this fork
    if (ok) g_mp.stamp(16, cs_);
    g_mp.base = -1;
    if (ok && force_on_ && !coupled) force_token(tok_, m_force_, 0, cs_);   // chain_launch: step 1's input
    return finish_capture(cs_, ok, exec, coupled ? "round (coupled)" : "round", err);
=======
    return finish_capture(cs_, ok, round_exec_[T], "round", err);
>>>>>>> their fork (github.com-eddoursul-Strata-custom)
}

// Chain step j (1..max_t-2): one row at the cell staged in step row `max_t + j - 1`, from the previous step's
// residual and token (left in Rin_[0] / tok_[0] by mtp_select); draft j and its probability to the mapped outputs.
bool MtpDrafter::capture_step(int j, std::string& err) {
    if (step_exec_[j]) return true;
    using namespace strata::kernels;
    const int64_t HCN = g_->hc * g_->n_embd;
    const int row = max_t_ + j - 1;
    g_mp.init();
    if (cudaStreamBeginCapture(cs_, cudaStreamCaptureModeThreadLocal) != cudaSuccess) { err = "mtp: begin capture"; return false; }
    const MappedCopy in[2] = {{step_ + row * 4, m_step_ + row * 4, 4},
                              {pos_ + row * g_->n_head, m_pos_ + row * g_->n_head, g_->n_head}};
    copy_from_mapped_multi(in, 2, cs_);
<<<<<<< this fork
#endif
    coupled_rec_ = coupled;
    coupled_j_ = j;
    g_mp.base = j < kMpGraphs ? j : -1;
=======
>>>>>>> their fork (github.com-eddoursul-Strata-custom)
    bool ok = record_forward(1, row, cs_, err);
    if (ok) mtp_select(R_, HCN, out_ids_, row_ + 1, Rin_, tok_, m_out_, j, cs_, probs_, m_prob_);
<<<<<<< this fork
    if (ok) g_mp.stamp(16, cs_);
    g_mp.base = -1;
    if (ok && force_on_ && !coupled) force_token(tok_, m_force_, j, cs_);   // chain_launch: step j+1's input
    return finish_capture(cs_, ok, exec, coupled ? "step (coupled)" : "step", err);
}

void MtpDrafter::kv_restore(int64_t upto) {
    const OnDevice on_device(device_);
    if (st_.kv_mode != 2 || upto <= 0) return;
    // the ring's blocks below `upto`, from the host copy: a checkpoint resume may have left later cells in them
    const strata::kernels::QsaShapes s = shapes_of(*g_);
    const int64_t b1 = (upto + s.page_size - 1) / s.page_size, b0 = std::max<int64_t>(0, b1 - st_.n_slots);
    strata::kernels::kv_ring_restore(qsa_attn_pools(st_), st_.host, qsa_kv_format(st_), b0, b1, st_.n_slots, s, cs_);
    cudaStreamSynchronize(cs_);
}

bool MtpDrafter::prepare_prefill(std::string& err) {
    const OnDevice on_device(device_);
    const int64_t per_row = 1 + 4 + g_->n_head;
    if (pf_cap_ < (int64_t) max_t_ * per_row) {
        if (pf_dev_) cudaFree(pf_dev_);
        pf_dev_ = nullptr;
        pf_cap_ = 0;
        if (cudaMalloc((void**) &pf_dev_, (size_t) (max_t_ * per_row) * sizeof(int32_t)) != cudaSuccess) {
            err = "mtp prefill: the input records do not fit";
            return false;
        }
        pf_cap_ = max_t_ * per_row;
    }
    for (int T = 1; T <= max_t_; ++T)
        if (!capture_prefill_dev(T, err)) return false;   // (STRATA_MTP_PREFILL_SYNC's loop syncs anyway)
    return true;
}

bool MtpDrafter::prefill(const float* R_rows, const int32_t* next_tokens, int64_t n, int64_t cell0, std::string& err,
                         bool sync) {
#if defined(STRATA_GPU_METAL)
    (void) R_rows; (void) sync;   // the rows are the decode window's own (the Metal drafter reads them in place)
    return strata::metal_engine::mtp_prefill(next_tokens, n, cell0, err);
#endif
    const OnDevice on_device(device_);
    const Clock::time_point t0 = Clock::now();
    const int64_t HCN = g_->hc * g_->n_embd;
    // cells the window can never reach again need no K/V
    const int64_t first_needed = (window_ > 0 && prompt_len_ > 0) ? prompt_len_ - window_ - 64 : 0;
    // E-4: every group's token / step / position records uploaded at once; each group is then device copies and a
    // graph on the one stream, with a single sync at the end (a group of <= max_t rows used to be staged in mapped
    // memory and synced before the next: ~5,500 host round trips on a 32K prompt).  The same work in the same
    // order.  STRATA_MTP_PREFILL_SYNC=1 keeps the old loop, =0 forces E-4.
    // HIP defaults to the old loop.  This pass runs whenever E-9 (Prefill::draft_kv) declines, which it does for the
    // drafter's ring (KV streaming, --kv-resident), and on gfx1201 / ROCm 7.2.4 the E-4 queue (a graph launch and four
    // device copies per group, ~2,000 groups per 8192-token chunk, no sync) sometimes never completes: the prompt hangs
    // in hipGraphLaunch or the final sync until the watchdog ends the engine.  R9700, full IQ3_XXS, --kv-resident
    // 32768, mixed load (chats, 32K and 7K prompts, deep follow-ups): E-4 hung in the first round on 2 of 2 tries,
    // the old loop ran 30 of 30 rounds clean, prompt speed unchanged.
    static const bool per_group_sync = [] {
        if (const char* v = std::getenv("STRATA_MTP_PREFILL_SYNC")) return std::atoi(v) != 0;
#if defined(STRATA_USE_HIP)
        return true;
#else
        return false;
#endif
    }();
    const int64_t NHp = g_->n_head, per_row = 1 + 4 + NHp;
    if (!per_group_sync && n > 0) {
        if (pf_cap_ < n * per_row) {
            if (pf_dev_) cudaFree(pf_dev_);
            pf_dev_ = nullptr;
            pf_cap_ = 0;
            if (cudaMalloc((void**) &pf_dev_, (size_t) (n * per_row) * sizeof(int32_t)) != cudaSuccess) {
                err = "mtp prefill: the input records do not fit";
                return false;
            }
            pf_cap_ = n * per_row;
        }
        std::vector<int32_t> rec((size_t) (n * per_row));
        int32_t* tk = rec.data();
        int32_t* stp = tk + n;
        int32_t* ps = stp + 4 * n;
        for (int64_t i = 0; i < n; ++i) {
            const int64_t cell = cell0 + i;
            tk[i] = next_tokens[i];
            stp[i * 4 + 0] = (int32_t) cell;
            stp[i * 4 + 1] = (int32_t) (cell + 1);
            stp[i * 4 + 2] = (int32_t) ((cell + 1) / 4);
            stp[i * 4 + 3] = (int32_t) (cell + 1);
            for (int64_t h = 0; h < NHp; ++h) ps[i * NHp + h] = (int32_t) cell;
        }
        if (pf_ring_.h2d(pf_dev_, rec.data(), rec.size() * sizeof(int32_t), cs_) != cudaSuccess) {
            err = "mtp prefill: the input records upload failed";
            return false;
        }
        const int32_t* d_tk = pf_dev_;
        const int32_t* d_stp = d_tk + n;
        const int32_t* d_ps = d_stp + 4 * n;
        for (int64_t c = 0; c < n; c += max_t_) {
            const int T = (int) std::min<int64_t>(max_t_, n - c);
            if (cell0 + c + T <= first_needed) continue;
            if (!capture_prefill_dev(T, err)) return false;
            if (cudaMemcpyAsync(tok_, d_tk + c, (size_t) T * 4, cudaMemcpyDeviceToDevice, cs_) != cudaSuccess ||
                cudaMemcpyAsync(step_, d_stp + c * 4, (size_t) T * 16, cudaMemcpyDeviceToDevice, cs_) != cudaSuccess ||
                cudaMemcpyAsync(pos_, d_ps + c * NHp, (size_t) (T * NHp) * 4, cudaMemcpyDeviceToDevice, cs_) != cudaSuccess ||
                cudaMemcpyAsync(Rin_, R_rows + (size_t) c * HCN, (size_t) T * HCN * sizeof(float), cudaMemcpyDeviceToDevice,
                                cs_) != cudaSuccess ||
                cudaGraphLaunch(prefill_dev_exec_[T], cs_) != cudaSuccess) {
                err = std::string("mtp prefill: ") + cudaGetErrorString(cudaGetLastError());
                return false;
            }
        }
        if (!sync) {   // the caller orders on stream(); the records are staged (a pageable copy returns once taken)
            (void) cudaStreamQuery(cs_);
            ms_prefill += ms_since(t0);
            return true;
        }
        if (cudaStreamSynchronize(cs_) != cudaSuccess) {
            err = std::string("mtp prefill: ") + cudaGetErrorString(cudaGetLastError());
            return false;
        }
        ms_prefill += ms_since(t0);
        return true;
    }
    for (int64_t c = 0; c < n; c += max_t_) {
        const int T = (int) std::min<int64_t>(max_t_, n - c);
        if (cell0 + c + T <= first_needed) continue;
        if (!capture_prefill(T, err)) return false;
        for (int t = 0; t < T; ++t) {
            const int64_t cell = cell0 + c + t;
            h_tok_[t] = next_tokens[c + t];
            h_step_[t * 4 + 0] = (int32_t) cell;
            h_step_[t * 4 + 1] = (int32_t) (cell + 1);
            h_step_[t * 4 + 2] = (int32_t) ((cell + 1) / 4);
            h_step_[t * 4 + 3] = (int32_t) (cell + 1);
            for (int64_t h = 0; h < g_->n_head; ++h) h_pos_[t * g_->n_head + h] = (int32_t) cell;
        }
        if (cudaMemcpyAsync(Rin_, R_rows + (size_t) c * HCN, (size_t) T * HCN * sizeof(float), cudaMemcpyDeviceToDevice,
                            cs_) != cudaSuccess ||
            cudaGraphLaunch(prefill_exec_[T], cs_) != cudaSuccess ||
            cudaStreamSynchronize(cs_) != cudaSuccess) {
            err = std::string("mtp prefill: ") + cudaGetErrorString(cudaGetLastError());
            return false;
        }
    }
    ms_prefill += ms_since(t0);
    return true;
=======
    return finish_capture(cs_, ok, step_exec_[j], "step", err);
}

MtpPromptKv MtpDrafter::prompt_kv() {
    MtpPromptKv k;
    k.norm_emb = f32("pre_fc_norm_embedding.weight");
    k.norm_hidden = f32("pre_fc_norm_hidden.weight");
    k.fc_emb = q8("fc_embedding.weight");
    k.fc_hidden = q8("fc_hidden.weight");
    k.hc_norm = f32("attn_hyper_connection.hc_norm.weight");
    k.hc_down = bf16("attn_hyper_connection.input_mix_weight_down.weight");
    k.hc_up = bf16("attn_hyper_connection.input_mix_weight_up.weight");
    k.k_proj = q8("self_attn.k_proj.weight");
    k.v_proj = q8("self_attn.v_proj.weight");
    k.k_norm = f32("self_attn.k_norm.weight");
    k.kv = &st_;
    k.first_cell = (window_ > 0 && prompt_len_ > 0) ? prompt_len_ - window_ - 64 : 0;
    return k;
>>>>>>> their fork (github.com-eddoursul-Strata-custom)
}

// STRATA_DRAFT_TIMING=1: where a draft round's time goes, from cudaEvent pairs around each graph launch (GPU time of the round
// graph and of each step, and the idle gap the host's decision leaves between two steps) and the host's own wall time.
namespace {
struct DraftTiming {
    bool on = false;
    cudaEvent_t eb[16] = {}, ea[16] = {};
    double round_ms = 0, step_ms = 0, gap_ms = 0, wall_ms = 0, tail_ms = 0;
    double pre_ms = 0, launch_ms = 0, ple_ms = 0, wait0_ms = 0, waitj_ms = 0;   // host: staging, graph launch calls, PLE prefetch, spins
    int64_t rounds = 0, steps = 0;
    DraftTiming() {
        const char* e = std::getenv("STRATA_DRAFT_TIMING");
        on = e != nullptr && e[0] == '1';
    }
    void init() {
        if (!on || eb[0] != nullptr) return;
        for (int i = 0; i < 16; ++i)
            if (cudaEventCreate(&eb[i]) != cudaSuccess || cudaEventCreate(&ea[i]) != cudaSuccess) { on = false; (void) cudaGetLastError(); return; }
    }
    void report() const {
        if (rounds == 0) return;
        const double r = (double) rounds;
        std::fprintf(stderr, "strata mtp draft timing: %lld rounds, %.2f steps/round: wall %.3f ms = round graph %.3f + steps %.3f GPU "
                             "(%.3f per step) + host gaps between graphs %.3f + tail %.3f (per round)\n",
                     (long long) rounds, (double) steps / r, wall_ms / r, round_ms / r, step_ms / r,
                     steps > 0 ? step_ms / (double) steps : 0.0, gap_ms / r, tail_ms / r);
        std::fprintf(stderr, "strata mtp draft timing (host, per round): staging %.3f, graph launch calls %.3f, PLE prefetch %.3f, "
                             "wait for draft 0 %.3f, waits for the steps %.3f\n",
                     pre_ms / r, launch_ms / r, ple_ms / r, wait0_ms / r, waitj_ms / r);
    }
};
DraftTiming g_dt;
}  // namespace

bool MtpDrafter::draft(int T, const int32_t* tokens, int64_t p, int a, int32_t* drafts, std::string& err,
                       float* probs, float min_p, int* n_drafts) {
<<<<<<< this fork
#if defined(STRATA_GPU_METAL)
    {
        if (T < 1 || T > max_t_ || a < 0 || a >= T) { err = "mtp: draft arguments out of range"; return false; }
        const int max_steps = (int) std::min<int64_t>(std::min(max_t_ - 1, max_drafts_), (int64_t) 1 << 30);
        const Clock::time_point t0 = Clock::now();
        int n = 0;
        if (!strata::metal_engine::mtp_draft(T, tokens, p, a, max_steps, drafts, probs, min_p, &n, err)) return false;
        for (int j = n; j < max_t_ - 1; ++j) { drafts[j] = 0; if (probs) probs[j] = 0.0f; }
        if (n_drafts) *n_drafts = n;
        ms_draft += ms_since(t0);
        ++rounds;
        return true;
    }
#endif
    const OnDevice on_device(device_);
=======
>>>>>>> their fork (github.com-eddoursul-Strata-custom)
    if (T < 1 || T > max_t_ || a < 0 || a >= T) { err = "mtp: draft arguments out of range"; return false; }
    if (!capture_round(T, err)) return false;
    const Clock::time_point t0 = Clock::now();
    const int64_t NH = g_->n_head;
    auto put = [&](int row, int64_t cell) {
        h_step_[row * 4 + 0] = (int32_t) cell;
        h_step_[row * 4 + 1] = (int32_t) (cell + 1);
        h_step_[row * 4 + 2] = (int32_t) ((cell + 1) / 4);
        h_step_[row * 4 + 3] = (int32_t) (cell + 1);
        for (int64_t h = 0; h < NH; ++h) h_pos_[row * NH + h] = (int32_t) cell;
    };
    for (int t = 0; t < T; ++t) {
        h_tok_[t] = tokens[t];
        put(t, p + t);
    }
    put(2 * max_t_ - 1, p + a);
    h_row_[0] = a;
    h_row_[1] = 0;
    std::atomic_thread_fence(std::memory_order_seq_cst);
<<<<<<< this fork
    if (!stage_source_R(T, err)) return false;

    SessionState* const pss = ple_ss_ ? ple_ss_ : ss_;
    const bool do_ple = pss != nullptr && pss->ple.ready() && pss->ple.table != nullptr;
    int32_t ple_prev[2] = {do_ple ? pss->ple_prev[0] : 0, do_ple ? pss->ple_prev[1] : 0};
    const bool dth = g_dt.on;
    auto hms = [](Clock::time_point a0) { return std::chrono::duration<double, std::milli>(Clock::now() - a0).count(); };
    auto prefetch_ple = [&](int32_t tok) {
        if (!do_ple || tok < 0) return;
        const Clock::time_point tp = Clock::now();
        struct PleT { bool on; double& acc; Clock::time_point t; ~PleT() { if (on) acc += std::chrono::duration<double, std::milli>(Clock::now() - t).count(); } } ple_t{dth, g_dt.ple_ms, tp};
        uint32_t rows16[strata::kernels::PLE_N_HEADS];
        strata::kernels::ngram_rows(&tok, ple_prev, 1, pss->ple.consts, rows16);
        ple_prev[0] = ple_prev[1];
        ple_prev[1] = tok;
        pss->ple.table->prefetch_rows(rows16);
    };

    int n = 0;
    if (min_p <= 0.0f && max_steps > 0) {
        if (cudaGraphLaunch(cp ? round_exec_c_[T] : round_exec_[T], cs_) != cudaSuccess) {
            err = std::string("mtp draft: ") + cudaGetErrorString(cudaGetLastError());
            return false;
        }
        for (int j = 1; j < max_steps; ++j) {
            if (cudaGraphLaunch(cp ? step_exec_c_[j] : step_exec_[j], cs_) != cudaSuccess) {
                err = std::string("mtp draft step: ") + cudaGetErrorString(cudaGetLastError());
                return false;
            }
        }
        (void) cudaStreamQuery(cs_);
        prefetch_ple(tokens[a]);
        progress_at("mtp draft: waiting for the GPU");   // #2065
        for (int j = 0; j + 1 < max_steps; ++j) {
            uint32_t spins = 0;
            while (((volatile int32_t*) h_out_)[j] < 0) {
#if defined(_WIN32) || defined(__x86_64__)
                _mm_pause();
#endif
                if ((++spins & 1023u) == 0 && cudaStreamQuery(cs_) != cudaErrorNotReady) break;
            }
            drafts[j] = ((volatile int32_t*) h_out_)[j];
            if (probs) probs[j] = ((volatile float*) h_prob_)[j];
            prefetch_ple(drafts[j]);
        }
        if (cudaStreamSynchronize(cs_) != cudaSuccess) {
            err = std::string("mtp draft: ") + cudaGetErrorString(cudaGetLastError());
            return false;
        }
        const int last = max_steps - 1;
        drafts[last] = ((volatile int32_t*) h_out_)[last];
        if (probs) probs[last] = ((volatile float*) h_prob_)[last];
        prefetch_ple(drafts[last]);
        n = max_steps;
    } else if (max_steps > 0) {
        g_dt.init();
        const bool dt = g_dt.on;
        if (dt) g_dt.pre_ms += hms(t0);
        if (dt) cudaEventRecord(g_dt.eb[0], cs_);
        Clock::time_point tl = Clock::now();
        if (cudaGraphLaunch(cp ? round_exec_c_[T] : round_exec_[T], cs_) != cudaSuccess) {
            err = std::string("mtp draft: ") + cudaGetErrorString(cudaGetLastError());
            return false;
        }
        if (dt) g_dt.launch_ms += hms(tl);
        if (dt) cudaEventRecord(g_dt.ea[0], cs_);
        (void) cudaStreamQuery(cs_);
        prefetch_ple(tokens[a]);
        progress_at("mtp draft: waiting for the GPU");   // #2065: a stall here read as "decode", the verify window before it
        auto wait_step = [&](int j) {
            uint32_t spins = 0;
            while (((volatile int32_t*) h_out_)[j] < 0) {
#if defined(_WIN32) || defined(__x86_64__)
                _mm_pause();
#endif
                if ((++spins & 1023u) == 0 && cudaStreamQuery(cs_) != cudaErrorNotReady) break;
            }
        };
        tl = Clock::now();
        wait_step(0);
        if (dt) g_dt.wait0_ms += hms(tl);
        drafts[0] = ((volatile int32_t*) h_out_)[0];
        float pj = ((volatile float*) h_prob_)[0];
        if (probs) probs[0] = pj;
        n = 1;
        for (int j = 1; j < max_steps && pj >= min_p; ++j) {
            if (dt) cudaEventRecord(g_dt.eb[j], cs_);
            tl = Clock::now();
            if (cudaGraphLaunch(cp ? step_exec_c_[j] : step_exec_[j], cs_) != cudaSuccess) {
                err = std::string("mtp draft step: ") + cudaGetErrorString(cudaGetLastError());
                return false;
            }
            if (dt) g_dt.launch_ms += hms(tl);
            if (dt) cudaEventRecord(g_dt.ea[j], cs_);
            (void) cudaStreamQuery(cs_);
            prefetch_ple(drafts[j - 1]);
            tl = Clock::now();
            wait_step(j);
            if (dt) g_dt.waitj_ms += hms(tl);
            drafts[j] = ((volatile int32_t*) h_out_)[j];
            pj = ((volatile float*) h_prob_)[j];
            if (probs) probs[j] = pj;
            ++n;
        }
        prefetch_ple(drafts[n - 1]);
        const Clock::time_point t_pre_sync = Clock::now();
        if (cudaStreamSynchronize(cs_) != cudaSuccess) {
            err = std::string("mtp draft: ") + cudaGetErrorString(cudaGetLastError());
            return false;
        }
        if (dt) {
            float ms = 0;
            double r = 0, s = 0, gp = 0;
            if (cudaEventElapsedTime(&ms, g_dt.eb[0], g_dt.ea[0]) == cudaSuccess) r = ms;
            for (int j = 1; j < n; ++j) {
                if (cudaEventElapsedTime(&ms, g_dt.eb[j], g_dt.ea[j]) == cudaSuccess) s += ms;
                if (cudaEventElapsedTime(&ms, g_dt.ea[j - 1], g_dt.eb[j]) == cudaSuccess) gp += ms;
            }
            g_dt.round_ms += r; g_dt.step_ms += s; g_dt.gap_ms += gp;
            g_dt.steps += n - 1; ++g_dt.rounds;
            g_dt.wall_ms += ms_since(t0);
            g_dt.tail_ms += std::chrono::duration<double, std::milli>(Clock::now() - t_pre_sync).count();
            if (g_dt.rounds % 300 == 0) g_dt.report();
        }
        if (g_mp.on) {
            g_mp.collect(n);
            if (g_mp.n[0] % 300 == 0) g_mp.report();
        }
    }
    for (int j = n; j < max_t_ - 1; ++j) { drafts[j] = 0; if (probs) probs[j] = 0.0f; }
    if (n_drafts) *n_drafts = n;
    ms_draft += ms_since(t0);
    ++rounds;
    return true;
}

bool MtpDrafter::stage_source_R(int T, std::string& err) {
    if (src_R_ == nullptr || src_R_ == window_R_) return true;
    const Clock::time_point t_src = Clock::now();
    struct SrcTime { MtpDrafter& d; Clock::time_point t; ~SrcTime() { d.ms_src += ms_since(t); ++d.n_src; } } src_time{*this, t_src};
    if (cudaMemcpyAsync((void*) window_R_, src_R_, (size_t) T * (size_t) (g_->hc * g_->n_embd) * sizeof(float),
                        cudaMemcpyDeviceToDevice, cs_) != cudaSuccess) {
        err = "mtp: staging the window's residual rows failed";
=======
    if (cudaGraphLaunch(round_exec_[T], cs_) != cudaSuccess || cudaStreamSynchronize(cs_) != cudaSuccess) {
        err = std::string("mtp draft: ") + cudaGetErrorString(cudaGetLastError());
>>>>>>> their fork (github.com-eddoursul-Strata-custom)
        return false;
    }
    drafts[0] = ((volatile int32_t*) h_out_)[0];
    float pj = ((volatile float*) h_prob_)[0];
    if (probs) probs[0] = pj;
    if (on_draft) on_draft(0, drafts[0], pj);
    int n = 1;
    // the chain continues while the last draft is likely enough to be verified; draft j's K/V goes to cell p + a + j,
    // so near the context's end it stops at the last cell
    const int chain = (int) std::min<int64_t>(max_drafts_ > 0 ? std::min(max_drafts_, max_t_ - 1) : max_t_ - 1,
                                              st_.max_cells - (p + a));
    for (int j = 1; j < chain && pj >= min_p; ++j) {
        if (!capture_step(j, err)) return false;
        put(max_t_ + j - 1, p + a + j);
        std::atomic_thread_fence(std::memory_order_seq_cst);
        if (cudaGraphLaunch(step_exec_[j], cs_) != cudaSuccess || cudaStreamSynchronize(cs_) != cudaSuccess) {
            err = std::string("mtp draft step: ") + cudaGetErrorString(cudaGetLastError());
            return false;
        }
        drafts[j] = ((volatile int32_t*) h_out_)[j];
        pj = ((volatile float*) h_prob_)[j];
        if (probs) probs[j] = pj;
        if (on_draft) on_draft(j, drafts[j], pj);
        ++n;
    }
    for (int j = n; j < max_t_ - 1; ++j) { drafts[j] = 0; if (probs) probs[j] = 0.0f; }
    if (n_drafts) *n_drafts = n;
    ms_draft += ms_since(t0);
    ++rounds;
    return true;
}

bool MtpDrafter::draft_first(int T, const float* R_row, int32_t token, int64_t cell, int32_t* drafts, std::string& err,
                             float* probs, float min_p, int* n_drafts) {
<<<<<<< this fork
#if defined(STRATA_GPU_METAL)
    {   // the decode window's row 0 is the residual (the window that just ran)
        (void) R_row;
        std::vector<int32_t> toks((size_t) T, token);
        return draft(1, toks.data(), cell, 0, drafts, err, probs, min_p, n_drafts);
    }
#endif
    const OnDevice on_device(device_);
=======
>>>>>>> their fork (github.com-eddoursul-Strata-custom)
    // row 0 is the real pair; rows 1.. repeat it and only write cells the next round overwrites
    const int64_t HCN = g_->hc * g_->n_embd;
    for (int t = 0; t < T; ++t)
        if (cudaMemcpy((void*) (window_R_ + (size_t) t * HCN), R_row, (size_t) HCN * sizeof(float),
                       cudaMemcpyDeviceToDevice) != cudaSuccess) {
            err = "mtp: staging the first residual failed";
            return false;
        }
    std::vector<int32_t> toks((size_t) T, token);
    return draft(T, toks.data(), cell, 0, drafts, err, probs, min_p, n_drafts);
}

}  // namespace strata::core
