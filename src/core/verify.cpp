// src/core/verify.cpp - see include/strata/core/verify.hpp.
#include "strata/core/verify.hpp"
<<<<<<< this fork
#if !defined(STRATA_USE_HIP) && !defined(STRATA_HIP_GFX906)
#include "strata/core/host_graph_launch.hpp"
#endif
#include "strata/core/remote_expert_opt.hpp"
#include "strata/core/dma_batch.hpp"
#include "strata/core/spec_prob.hpp"
=======
>>>>>>> their fork (github.com-eddoursul-Strata-custom)
#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <intrin.h>
#endif

#include "strata/core/native_head.hpp"
#include "strata/kernels/iq_kernels.hpp"
#include "strata/kernels/cpu/expert_layout.hpp"
#include "strata/kernels/bf16_gemv.hpp"
#include "strata/kernels/cpu/expert.hpp"
#include "strata/kernels/elementwise.hpp"
#include "strata/kernels/fused_gr.hpp"
#include "strata/kernels/gr.hpp"
#include "strata/kernels/kv_q8.hpp"
#include "strata/kernels/native_mmvq.hpp"
#include "strata/kernels/native_moe.hpp"
#include "strata/kernels/native_ple_postops.hpp"
#include "strata/kernels/native_qsa.hpp"
#include "strata/kernels/native_qsa_indexer.hpp"
#include "strata/kernels/native_router.hpp"
#include "strata/kernels/native_rope.hpp"
#include "strata/kernels/ngram.hpp"
#include "strata/kernels/ple.hpp"
#include "strata/kernels/qsa.hpp"
#include "strata/kernels/qsa_decode_attn.hpp"
#include "strata/kernels/qsa_select.hpp"
#include "strata/kernels/quantize_act.hpp"
#include "strata/kernels/s2_expert_grouped.hpp"
#include "strata/kernels/sampler.hpp"
#include "strata/kernels/shared_expert.hpp"
#include "strata/kernels/verify_kernels.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <immintrin.h>

#if defined(STRATA_GPU_METAL)
// The Metal backend's decode window (metal/src/engine/window.cpp): this file's init/run/commit hand over to it.
namespace strata::metal_engine {
void window_destroy(void* w);
void* window_create();
bool window_init(void* w, const strata::core::ModelGeometry& g, strata::core::SessionState& ss, int max_t, int64_t max_cells,
                 int64_t* n_vocab, std::string& err);
bool window_run(void* w, int T, const int32_t* tokens, int64_t pos0, bool head, int32_t* out, std::string& err);
bool window_commit(void* w, int n_keep, std::string& err);
bool window_copy_logits(void* w, int t, float* host);
}  // namespace strata::metal_engine
#endif

namespace strata::core {
namespace {

constexpr float EPS = 1e-6f;
using Clock = std::chrono::steady_clock;
double ms_since(Clock::time_point t) { return std::chrono::duration<double, std::milli>(Clock::now() - t).count(); }
const bool g_dbg = std::getenv("STRATA_VERIFY_DEBUG") != nullptr;
#define VDBG(...) do { if (g_dbg) { std::fprintf(stderr, "verify dbg: " __VA_ARGS__); std::fflush(stderr); } } while (0)

struct Bump {
    uint8_t* base = nullptr;
    uint64_t used = 0;
    template <typename T> T* take(uint64_t n) {
        T* p = base ? (T*) (base + used) : nullptr;
        used += (n * sizeof(T) + 255) & ~255ull;
        return p;
    }
};

bool mapped(size_t bytes, void** h, void** d, unsigned flags = cudaHostAllocMapped) {
    if (cudaHostAlloc(h, bytes, flags) != cudaSuccess) return false;
    std::memset(*h, 0, bytes);
    return cudaHostGetDevicePointer(d, *h, 0) == cudaSuccess;
}

strata::kernels::QsaShapes shapes_of(const ModelGeometry& g) { return qsa_shapes(g); }

const WeightRef* need(const LayerView& v, const char* suffix, std::string& err) {
    const WeightRef* r = v.get(suffix);
    if (r == nullptr && err.empty()) err = v.name(suffix) + " is missing";
    return r;
}

bool native_of(const WeightRef* w, const std::string& name, std::string& err) {
    if (w == nullptr) return false;
    if (w->native_data == nullptr) {
        err = "verify: " + name + " is not served natively (run with --native)";
        return false;
    }
    return true;
}

}  // namespace

<<<<<<< this fork
namespace {
std::atomic<const Verifier*> g_diag_verifier{nullptr};
void diag_active_verifier(std::FILE* f) {
    if (const Verifier* v = g_diag_verifier.load()) v->diag(f);
}
// #267: every live verifier (a layer split has one per stage), for the release before the engine ends
constexpr int kLiveMax = 16;
std::atomic<Verifier*> g_live[kLiveMax];
void release_live_verifiers(std::FILE* f) {
    for (auto& slot : g_live)
        if (Verifier* v = slot.load()) {
            const Clock::time_point t0 = Clock::now();
            const bool done = v->release_gpu_waits(5000);
            if (f != nullptr)
                std::fprintf(f, "strata: released the verify window's GPU waits (#267): the GPU %s\n",
                             done ? ("finished in " + std::to_string((long long) ms_since(t0)) + " ms").c_str()
                                  : "did not finish within 5 s");
        }
    if (f != nullptr) std::fflush(f);
}
// #649: the host side of STRATA_VERIFY_TRACE - a ring of the last kTraceN handshake events of every verifier
struct TraceEv {
    int64_t t_ns;
    const void* who;
    int64_t window, step, layer, aux;
    uint32_t seq, flag, a, b;
    const char* what;
};
constexpr uint64_t kTraceN = 4096;
TraceEv g_trace_ring[kTraceN];
std::atomic<uint64_t> g_trace_next{0};
int64_t trace_now_ns() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now().time_since_epoch()).count();
}
std::string released_note(bool drained) {
    return drained ? "; its GPU waits were released and the GPU finished (#267)"
                   : "; its GPU waits were released but the GPU did not finish within 5 s (#267)";
}
// #267 test hook: STRATA_TEST_VERIFY_STALL=N withholds the last layer's flag in the N-th window (1-based), so the
// GPU spins on a flag nobody raises - the bounded window wait and the release are then what ends it.  Unset: never.
const int64_t g_test_stall = [] {
    const char* e = std::getenv("STRATA_TEST_VERIFY_STALL");
    return e != nullptr ? (int64_t) std::atoll(e) : (int64_t) 0;
}();
}  // namespace

bool Verifier::release_gpu_waits(int timeout_ms) {
    released_.store(true);
    trace_ev("RELEASE", -1, -1, timeout_ms);
    // the words the spin kernels read (wait_flag_ge, wait_flag_ge_or) are mapped host memory, so a store here
    // reaches them with no API call; UINT32_MAX is past every ring.  (E-6's skip words are device memory, but
    // wait_flag_ge_or also returns on its flag.)  A host function raising flag B later only raises.
    for (uint32_t* p : {h_flag_, h_flagA_, h_flagB_})
        if (p != nullptr) *(volatile uint32_t*) p = UINT32_MAX;
    std::atomic_thread_fence(std::memory_order_seq_cst);
    _mm_sfence();
    const OnDevice on_device(device_);
    const Clock::time_point t0 = Clock::now();
    for (cudaStream_t s : {cs_, copy_}) {
        if (s == nullptr) continue;
        while (cudaStreamQuery(s) == cudaErrorNotReady) {
            if (ms_since(t0) > timeout_ms) {
                trace_ev("RELEASE-NOT-DRAINED", -1, -1, (int64_t) ms_since(t0));
                return false;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    }
    trace_ev("RELEASE-DRAINED", -1, -1, (int64_t) ms_since(t0));   // aux: ms the GPU took to finish once released
    return true;
}

void Verifier::trace_ev(const char* what, int64_t step, int64_t layer, int64_t aux) const {
    if (!g_trace) return;
    auto rd = [](const uint32_t* p) { return p ? *(const volatile uint32_t*) p : 0u; };
    TraceEv& e = g_trace_ring[g_trace_next.fetch_add(1) % kTraceN];
    e = {trace_now_ns(), this, windows, step, layer, aux, rd(h_seq_), rd(h_flag_), rd(h_flagA_), rd(h_flagB_), what};
}

void Verifier::trace_dump(std::FILE* f) const {
    if (!g_trace || f == nullptr) return;
    static const char* names[kProfPer] = {"pre", "hc-read0", "qkv gemv", "conv", "ab", "z", "rec", "kv-idx",
                                          "k/v rope", "kv append", "q", "scores+topk", "kv-resolve", "attention",
                                          "gate", "", "out-proj", "router+ring", "shared", "waitA", "VRAM hits", "waitB",
                                          "PCIe grp", "waitCPU", "combine", "", "head", "", "", "", "", "", ""};
    const uint64_t end = g_trace_next.load();
    const uint64_t n = end < 160 ? end : 160;
    std::fprintf(f, "strata verify trace (#649): the last %llu handshake events (host clock, us before the last; "
                    "words: seq = GPU rang, flag = CPU served, A = plan, B = copies)\n", (unsigned long long) n);
    const int64_t t_last = n ? g_trace_ring[(end - 1) % kTraceN].t_ns : 0;
    for (uint64_t i = end - n; i < end; ++i) {
        const TraceEv& e = g_trace_ring[i % kTraceN];
        if (e.what == nullptr) continue;
        std::fprintf(f, "  %10.1f %-20s v%p window %lld step %lld layer %lld aux %lld | seq %u flag %u A %u B %u\n",
                     (double) (t_last - e.t_ns) / 1000.0, e.what, e.who, (long long) e.window, (long long) e.step,
                     (long long) e.layer, (long long) e.aux, e.seq, e.flag, e.a, e.b);
    }
    if (trace_h_ == nullptr || g_ == nullptr) return;
    // the breadcrumbs: every stamp point the GPU passed in this window (ns of the GPU clock after the window's first)
    unsigned long long t0 = ~0ull;
    for (size_t i = 0; i < trace_n_; ++i) {
        const unsigned long long t = *(const volatile unsigned long long*) (trace_h_ + i);
        if (t != 0 && t < t0) t0 = t;
    }
    if (t0 == ~0ull) {
        std::fprintf(f, "strata verify trace: no GPU breadcrumb in this window (the GPU never started it)\n");
        return;
    }
    int64_t last_l = -1;
    for (int64_t l = 0; l <= g_->n_layers; ++l)
        for (int i = 0; i < kProfPer; ++i)
            for (int grp = 0; grp < 2; ++grp)
                if (trace_h_[(size_t) ((l * kProfPer + i) * 2 + grp)] != 0) last_l = l;
    std::fprintf(f, "strata verify trace: GPU breadcrumbs (us after the window's first), the GPU got as far as layer "
                    "%lld:\n", (long long) last_l);
    for (int64_t l = std::max<int64_t>(0, last_l - 2); l <= std::min<int64_t>(g_->n_layers, last_l + 1); ++l)
        for (int grp = 0; grp < 2; ++grp) {
            std::string line;
            char b[64];
            for (int i = 0; i < kProfPer; ++i) {
                const unsigned long long t = trace_h_[(size_t) ((l * kProfPer + i) * 2 + grp)];
                if (t == 0) continue;
                std::snprintf(b, sizeof b, " %s(%d)=%.1f", names[i], i, (double) (t - t0) / 1000.0);
                line += b;
            }
            if (!line.empty())
                std::fprintf(f, "  layer %lld group %d:%s\n", (long long) l, grp, line.c_str());
        }
    std::fflush(f);
}

void Verifier::diag(std::FILE* f) const {
    auto rd = [](const uint32_t* p) { return p ? *(const volatile uint32_t*) p : 0u; };
    // #251: outside a verify stage these are the LAST window's numbers (it finished), not the stalled work's
    const char* where = progress().where.load();
    const bool current = where != nullptr && std::strncmp(where, "verify window", 13) == 0;
    std::fprintf(f, "  verify window%s: %d tokens at position %lld, host at layer step %u; the GPU rang %u; flags: "
                    "served %u, plan (A) %u, copies (B) %u\n", current ? "" : " (last window, not the current stage)",
                 last_t_, (long long) last_pos0_, cur_layer_ + 1, rd(h_seq_), rd(h_flag_), rd(h_flagA_), rd(h_flagB_));
    trace_dump(f);   // #649: STRATA_VERIFY_TRACE=1 only
}


Verifier::~Verifier() {
#if defined(STRATA_GPU_METAL)
    strata::metal_engine::window_destroy(metal_);
    metal_ = nullptr;
#endif
    const Verifier* self = this;
    g_diag_verifier.compare_exchange_strong(self, nullptr);
    for (auto& slot : g_live) {
        Verifier* me = this;
        slot.compare_exchange_strong(me, nullptr);
    }
=======
Verifier::~Verifier() {
>>>>>>> their fork (github.com-eddoursul-Strata-custom)
    if (cs_) cudaStreamSynchronize(cs_);
    for (auto& e : exec_)
        if (e) cudaGraphExecDestroy(e);
    if (commit_exec_) cudaGraphExecDestroy(commit_exec_);
    if (cs_) cudaStreamDestroy(cs_);
    if (copy_) { cudaStreamSynchronize(copy_); cudaStreamDestroy(copy_); }
    if (side_) cudaStreamDestroy(side_);
    if (shs_) cudaStreamDestroy(shs_);
    if (ples_) cudaStreamDestroy(ples_);
    if (g2s_) cudaStreamDestroy(g2s_);
    if (cps_) cudaStreamDestroy(cps_);
    if (kst_) cudaStreamDestroy(kst_);
    if (sgs_) cudaStreamDestroy(sgs_);
    for (cudaEvent_t e : {join_, bfork_, bjoin_, pfork_, pjoin_, shfork_, shjoin_, res_ready_, done_, g2fork_, plefork_,
                          plejoin_,
                          g2join_[0], g2join_[1], cpfork_, cpjoin_[0], cpjoin_[1], efork_, rdone_, kjoin_, iproj_,
                          sgjoin_})
        if (e) cudaEventDestroy(e);
    if (arena_) cudaFree(arena_);
    if (xk_dev_) cudaFree(xk_dev_);
    if (stamps_) cudaFree(stamps_);
    void* hosts[] = {h_tok_, h_val_, h_step_, h_pos_, h_commit_, h_ple_, h_out_, h_x_, h_ids_, h_w_, h_seq_, h_flag_,
                     h_ymiss_, h_bad_,
                     h_flagA_, h_plan_, h_flagB_, h_pids_, h_pw_, h_pseq_, h_pleflag_, h_rows_, h_res_, h_flag2_,
                     h_list2_, h_xk_, h_xq1_};
    for (void* h : hosts)
        if (h) cudaFreeHost(h);
}

int Verifier::nonfinite(int T) {
    if (!nan_guard_) return -1;
    ++guard_calls_;
    if (std::find(nan_inject_.begin(), nan_inject_.end(), guard_calls_) != nan_inject_.end()) return 0;
    for (int t = 0; t < T; ++t)
        if (((volatile const uint32_t*) h_bad_)[t] != 0 ||
            (shead_ != nullptr && ((volatile const uint32_t*) shead_->nonfinite())[t] != 0))
            return t;
    return -1;
}

bool Verifier::init(const WeightTable& wt, const ModelGeometry& g, SessionState& ss, const VerifyHits& hits,
                    const NativeHead* head, int max_t, std::string& err) {
    if (const char* v = std::getenv("STRATA_NAN_GUARD")) nan_guard_ = !(v[0] == '0' && v[1] == '\0');
    if (const char* v = std::getenv("STRATA_NAN_INJECT"))
        for (const char* c = v; *c != '\0';) {
            char* e = nullptr;
            const unsigned long long n = std::strtoull(c, &e, 10);
            if (e == c) break;
            nan_inject_.push_back(n);
            c = *e == ',' ? e + 1 : e;
        }
    wt_ = &wt;
    g_ = &g;
    ss_ = &ss;
    hits_ = hits;
    head_ = head;
    max_t_ = max_t;
<<<<<<< this fork
    sampling_.greedy = true;      // a fresh verifier samples greedily until set_sampling says otherwise
    sampling_.temperature = 0.0f;
#if defined(STRATA_GPU_METAL)
    if (metal_ == nullptr) metal_ = strata::metal_engine::window_create();
    return strata::metal_engine::window_init(metal_, g, ss, max_t, ss.qsa_states[ss.qsa_primary()].max_cells, &n_vocab_,
                                             err);
#endif
=======
>>>>>>> their fork (github.com-eddoursul-Strata-custom)
    if (max_t < 2 || max_t > strata::kernels::kVerifyMaxT || max_t > strata::kernels::cpu::MAXT) {
        err = "verify: the window must hold 2.." + std::to_string(strata::kernels::kVerifyMaxT) + " tokens";
        return false;
    }
    if (hits.res == nullptr || hits.cache_base == nullptr || hits.blob <= 0 || hits.slots <= 0) {
        err = "verify: needs the profile-filled VRAM expert tier (--expert-profile and --expert-cache)";
        return false;
    }
    std::string why;
    if (!layer_verify_compatible(why)) {
        err = "verify: " + why + " (the verify window reproduces the default native decode path)";
        return false;
    }
    if (!strata::kernels::fused_gr_supported(g.n_embd, g.hc, g.hc_lr) || ss.k != 10 || g.ssm_state_size != 128 ||
        g.ssm_d_conv != 4) {
        err = "verify: geometry differs from the artifact's";
        return false;
    }
    if (ss.ple.ready() && (!strata::kernels::ple_native_postops_enabled() || !strata::kernels::ple_native_bf16() ||
                           (ss.ple.w.key_bf16 == nullptr && ss.ple.w.key_native_data == nullptr))) {
        err = "verify: the PLE block needs the native postops and value and a BF16 or native key";
        return false;
    }
    const WeightRef* wo = wt.find("output.weight");
    if (wo == nullptr) { err = "verify: output.weight is missing"; return false; }
    n_vocab_ = wo->ne1;

    const strata::kernels::QsaShapes s = shapes_of(g);
    cap_ = strata::kernels::qsa_selection_width(strata::kernels::kTopkMaxCells, s);
    max_blocks_ = ss.qsa_states[0].max_cells / s.idx_block + 2;
    attn_scratch_floats_ = (int64_t) strata::kernels::qsa_decode_attn_scratch_floats(cap_, s);

    const uint64_t T = (uint64_t) max_t, N = (uint64_t) g.n_embd, HC = (uint64_t) g.hc, K = (uint64_t) ss.k;
    head_rows_ = (head != nullptr && head->loaded()) ? head->rows() : n_vocab_;
    if (head_rows_ != n_vocab_ && (shead_ == nullptr || shead_->split() != head_rows_)) {
        err = "verify: the head holds part of the vocabulary without its other part";
        return false;
    }
    const uint64_t C = (uint64_t) g.ssm_conv_channels, ZV = (uint64_t) g.ssm_value_dim, HV = (uint64_t) g.ssm_v_heads;
    const uint64_t NH = (uint64_t) g.n_head, HD = (uint64_t) g.head_dim, NKV = (uint64_t) g.n_head_kv;
    const uint64_t IQ = (uint64_t) g.idx_q_heads, ID = (uint64_t) g.idx_key_dim;
    const uint64_t nG = (uint64_t) g.n_gdn_layers(), nQ = (uint64_t) g.n_qsa_layers();
    const uint64_t HS = (uint64_t) strata::kernels::NG_HIST * strata::kernels::NG_HC_DIM;
    const uint64_t TS = (uint64_t) (s.idx_block - 1) * ID;
    const int max_in = (int) std::max<uint64_t>(std::max<uint64_t>(N, ZV), NH * HD);

    // ---- mapped staging (portable where the second GPU writes: its rows and flags)
    const unsigned portable = cudaHostAllocMapped | cudaHostAllocPortable;
    list2_stride_ = (4 + (int64_t) (T * K) + 3) & ~3ll;
    const auto& lay = strata::kernels::cpu::expert_layout();
    quant_router_ = strata::kernels::native_router_enabled() && lay.native;
    xk_layer_.assign((size_t) g.n_layers, 0);
    for (int64_t l = 0; quant_router_ && l < g.n_layers; ++l)
        xk_layer_[(size_t) l] = pool_takes_q8k(lay.fmt[(size_t) l]);
    xk_stride_ = ((int64_t) (N / 256) * 292 + 15) & ~15ll;
    bool ok = mapped(T * 4, (void**) &h_tok_, (void**) &m_tok_) &&
              mapped(T * strata::kernels::kStepCount * 4, (void**) &h_step_, (void**) &m_step_) &&
              mapped(T * NH * 4, (void**) &h_pos_, (void**) &m_pos_) &&
              mapped((2 + T) * 4 + 16, (void**) &h_commit_, (void**) &m_commit_) &&
              mapped(T * N * 4, (void**) &h_ple_, (void**) &m_ple_) &&
              mapped(T * 4 + 16, (void**) &h_out_, (void**) &m_out_) &&
              mapped(T * 4 + 16, (void**) &h_val_, (void**) &m_val_) &&
              mapped(T * 4 + 16, (void**) &h_bad_, (void**) &m_bad_) &&
              mapped(T * N * 4, (void**) &h_x_, (void**) &m_x_) &&
              mapped(T * K * 4, (void**) &h_ids_, (void**) &m_ids_) &&
              mapped(T * K * 4, (void**) &h_w_, (void**) &m_w_) &&
              mapped(64, (void**) &h_seq_, (void**) &m_seq_) &&
              mapped(64, (void**) &h_flag_, (void**) &m_flag_) &&
              mapped(64, (void**) &h_flagA_, (void**) &m_flagA_) &&
              mapped(64, (void**) &h_flagB_, (void**) &m_flagB_) &&
              mapped(T * K * N * 4, (void**) &h_ymiss_, (void**) &m_ymiss_, portable) &&
              mapped(128, (void**) &h_flag2_, (void**) &m_flag2_, portable) &&
              mapped((size_t) (2 * list2_stride_) * 4, (void**) &h_list2_, (void**) &m_list2_) &&
              mapped(T * (uint64_t) xk_stride_, (void**) &h_xk_, (void**) &m_xk_) &&
              mapped(T * (N / 32) * 36, (void**) &h_xq1_, (void**) &m_xq1_) &&
              mapped(2 * T * K * 4, (void**) &h_pids_, (void**) &m_pids_) &&
              mapped(2 * T * K * 4, (void**) &h_pw_, (void**) &m_pw_) &&
              mapped(64, (void**) &h_pseq_, (void**) &m_pseq_) &&
              mapped(64, (void**) &h_pleflag_, (void**) &m_pleflag_) &&
              mapped(T * HC * N * 4, (void**) &h_rows_, (void**) &m_rows_);
    if (!ok) { err = "verify: mapped staging allocation failed"; return false; }
    // A plan block, one per token group: counts(4) | start(cap+1) | dst(cap) | tok(cap) | pad | ptr(cap u64) |
    // ptr2(cap u64) | start2(cap+1).  The main GPU's hits (`verify_hit_plan`, device) fill counts [groups, entries],
    // start, dst, tok and ptr; the pool's PCIe share (mapped, `sink_`) counts [-, entries, groups], dst, tok, ptr2 and
    // start2.
    {
        const int64_t cap = (int64_t) (T * K);
        plan_ptr_off_ = (4 + (cap + 1) + cap + cap + 1) & ~1ll;
        plan_i32_ = (plan_ptr_off_ + 4 * cap + (cap + 1) + 2) & ~1ll;
        plan_stride_ = plan_i32_ + 16;
        if (!mapped((size_t) plan_i32_ * 4 * 2 + 64, (void**) &h_plan_, (void**) &m_plan_)) {
            err = "verify: mapped plan allocation failed";
            return false;
        }
        sink_.counts = h_plan_;
        sink_.dst = h_plan_ + 4 + cap + 1;
        sink_.tok = sink_.dst + cap;
        sink_.ptr2 = (unsigned long long*) (h_plan_ + plan_ptr_off_) + cap;
        sink_.start2 = h_plan_ + plan_ptr_off_ + 4 * cap;
        sink_.cap = cap;
        sink_.publish = &Verifier::publish_plan;
        sink_.host_rows_only = true;                      // the gather-combine (post) reads the GPUs' from VRAM
        sink_.fetch = &Verifier::fetch_dma;
        sink_.ctx = this;
    }
    // the residency snapshot, rounded up to whole 16-byte copies
    res_words_ = (g.n_layers * g.n_expert + 3) & ~3ll;
    if (!mapped((size_t) res_words_ * 4, (void**) &h_res_, (void**) &m_res_)) {
        err = "verify: mapped residency allocation failed";
        return false;
    }
    std::memcpy(h_res_, hits.res, (size_t) (g.n_layers * g.n_expert) * sizeof(int32_t));

    // ---- the device arena: the same sequence counted, then carved
    auto carve = [&](Bump& b) {
        tok_ = b.take<int32_t>(T); step_ = b.take<int32_t>(T * strata::kernels::kStepCount);
        pos_ = b.take<int32_t>(T * NH); commit_ = b.take<int32_t>(2 + T);
        ple_ = b.take<float>(T * N); emb_ = b.take<float>(T * N); R_ = b.take<float>(T * HC * N);
        mixed_ = b.take<float>(T * N); bo_ = b.take<float>(T * N);
        inj_ = b.take<float>(T * HC); inj2_ = b.take<float>(T * HC);
        lo_ = b.take<float>(T * (uint64_t) g.hc_lr); rs_ = b.take<float>(T * HC);
        grs_ = (float*) b.take<uint8_t>(strata::kernels::fused_gr_scratch_bytes());
        xq_ = b.take<uint8_t>(strata::kernels::native_q8_1_bytes(max_in, (int) T));
        xil_ = b.take<uint8_t>(strata::kernels::native_q8_1_il_bytes(max_in, (int) T));
        qkv_L_ = b.take<float>(nG * T * C); h_L_ = b.take<float>(nG * T * C);
        gate_L_ = b.take<float>(nG * T * HV); beta_L_ = b.take<float>(nG * T * HV);
        z_ = b.take<float>(T * ZV); y_ = b.take<float>(T * ZV); y_dummy_ = b.take<float>(T * ZV);
        qfull_ = b.take<float>(T * NH * 2 * HD); qcur_ = b.take<float>(T * NH * HD);
        kcur_ = b.take<float>(T * NKV * HD); vcur_ = b.take<float>(T * NKV * HD);
        idx_raw_L_ = b.take<float>(nQ * T * ID); qidx_ = b.take<float>(T * IQ * ID);
        scores_ = b.take<float>(T * (uint64_t) max_blocks_); sel_ = b.take<int32_t>(T * (uint64_t) cap_);
        attn32_ = b.take<float>(T * NH * HD);
        attn_scratch_ = b.take<float>(T * (uint64_t) attn_scratch_floats_);
        tail_snap_ = b.take<float>(nQ * TS);
        logits_ = b.take<float>(T * (uint64_t) g.n_expert); w_ = b.take<float>(T * K); ids_ = b.take<int32_t>(T * K);
        plogits_ = b.take<float>(2 * T * (uint64_t) g.n_expert); pw_ = b.take<float>(T * K);
        pids_ = b.take<int32_t>(T * K);
        pmixed_ = b.take<float>(T * N); gate_ema_ = b.take<float>((uint64_t) g.n_layers * HC * N);
        pbias_ = b.take<float>((uint64_t) (g.n_layers * g.n_expert));
        shared_ = b.take<float>(T * N); hit_out_ = b.take<float>(T * K * N);
        hit_slot_ = b.take<int32_t>(T * K); hit_dst_ = b.take<int32_t>(T * K); hit_count_ = b.take<int32_t>(4);
        list2_ = b.take<int32_t>(2 * (uint64_t) list2_stride_);
        ring_count_ = b.take<unsigned>(2);
        plan_ = b.take<int32_t>(2 * (uint64_t) plan_stride_); dplan_ = b.take<int32_t>(2 * (uint64_t) plan_stride_);
        res_ = b.take<int32_t>((uint64_t) res_words_); slot_ptr_ = b.take<unsigned long long>((uint64_t) hits.slots);
        staging_ = b.take<uint8_t>((uint64_t) kStagingBlobs * strata::kernels::cpu::expert_layout().max_blob);
        hit_xq_ = b.take<uint8_t>(T * (N / 32) * 34); hit_xs_ = b.take<float>(T * (N / 32));
        nat_xq_ = b.take<uint8_t>(T * (N / 32) * 36);
        hit_scratch_ = b.take<uint8_t>(std::max<uint64_t>(
            strata::kernels::moe_hit_grouped_scratch_bytes((int64_t) (T * K), g.n_embd, g.n_ff),
            strata::kernels::native_expert_scratch_bytes((int64_t) (T * K), g.n_ff)));
        head_mixed_ = b.take<float>(T * N); head_inj_ = b.take<float>(HC);
        sh_gate_ = b.take<float>(T * (uint64_t) g.n_ff);
        sh_up_ = b.take<float>(T * (uint64_t) g.n_ff); sh_g_ = b.take<float>(T + 4);
        head_logits_ = b.take<float>(T * (uint64_t) n_vocab_);
        arg_scratch_ = b.take<uint8_t>(strata::kernels::argmax_rows_scratch_bytes((int) T));
        if (shead_ != nullptr) head_full_ = b.take<float>(T * (uint64_t) n_vocab_);
        one_ = b.take<int32_t>(4);
        hist_snap_ = b.take<float>(T * HS);
        const uint64_t PD = (uint64_t) strata::kernels::NG_HC_DIM;
        ple_key_ = b.take<float>(T * PD); ple_val_ = b.take<float>(T * N); ple_nkey_ = b.take<float>(T * PD);
        ple_norm_ = b.take<float>(T * PD); ple_gated_ = b.take<float>(T * PD); ple_gate_ = b.take<float>(T * HC);
    };
    Bump count;
    carve(count);
    if (cudaMalloc(&arena_, count.used) != cudaSuccess) {
        err = "verify: the device arena (" + std::to_string(count.used >> 20) + " MiB) does not fit";
        return false;
    }
    cudaMemset(arena_, 0, count.used);
    Bump real;
    real.base = (uint8_t*) arena_;
    carve(real);
    if (lay.native && strata::kernels::expert_math() == strata::kernels::ExpertMath::Exact &&
        cudaMalloc((void**) &xk_dev_, (size_t) T * (size_t) (N / 256) * 292) != cudaSuccess) {
        err = "verify: the exact hits' Q8_K rows";
        return false;
    }
    {
        const int32_t one = 1;
        if (cudaMemcpy(one_, &one, sizeof one, cudaMemcpyHostToDevice) != cudaSuccess) {
            err = "verify: the arena could not be set";
            return false;
        }
    }
    sink_.staging = (unsigned long long) staging_;
    sink_.staging_cap = kStagingBlobs;
    (void) TS;
<<<<<<< this fork
    bool copy_made = false;
#if defined(STRATA_USE_HIP) && !defined(STRATA_HIP_GFX906)
    // engine-latency, STRATA_COPY_HIP_PRIO=1 (opt-in): the copy stream at high priority, so it gets a hardware queue of
    // its own (ROCm hands queues out per priority).  Sharing one with the window's stream, --pcie-mode dma's copies and
    // their host function queue behind the window's graph, which spins on the flag they raise (a timeout at layer 1).
    {
        static const bool cp = [] { const char* v = std::getenv("STRATA_COPY_HIP_PRIO"); return v != nullptr && std::atoi(v) != 0; }();
        int lo = 0, hi = 0;
        if (cp && hipDeviceGetStreamPriorityRange(&lo, &hi) == hipSuccess)
            copy_made = hipStreamCreateWithPriority(&copy_, hipStreamNonBlocking, hi) == hipSuccess;
        (void) cudaGetLastError();
    }
#endif
    if (!copy_made && cudaStreamCreateWithFlags(&copy_, cudaStreamNonBlocking) != cudaSuccess) {
=======
    {   // the gates' running averages start at a gate's middle
        const std::vector<float> half((size_t) (g.n_layers * g.hc * g.n_embd), 0.5f);
        if (cudaMemcpy(gate_ema_, half.data(), half.size() * sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess) {
            err = "verify: the gates' averages could not be set";
            return false;
        }
    }
    {
        std::vector<unsigned long long> sp((size_t) hits.slots);
        for (int64_t i = 0; i < hits.slots; ++i)
            sp[(size_t) i] = (unsigned long long) (hits.cache_base + (hits.slot_off != nullptr ? hits.slot_off[i]
                                                                                              : (uint64_t) i * (uint64_t) hits.blob));
        if (cudaMemcpy(slot_ptr_, sp.data(), sp.size() * sizeof(unsigned long long), cudaMemcpyHostToDevice) != cudaSuccess) {
            err = "verify: the slot table could not be uploaded";
            return false;
        }
    }
    if (cudaStreamCreateWithFlags(&copy_, cudaStreamNonBlocking) != cudaSuccess) {
>>>>>>> their fork (github.com-eddoursul-Strata-custom)
        err = "verify: copy stream create failed";
        return false;
    }
    if (cudaStreamCreateWithFlags(&cs_, cudaStreamNonBlocking) != cudaSuccess ||
        cudaStreamCreateWithFlags(&side_, cudaStreamNonBlocking) != cudaSuccess ||
        cudaStreamCreateWithFlags(&shs_, cudaStreamNonBlocking) != cudaSuccess ||
        cudaStreamCreateWithFlags(&ples_, cudaStreamNonBlocking) != cudaSuccess ||
        cudaEventCreateWithFlags(&plefork_, cudaEventDisableTiming) != cudaSuccess ||
        cudaEventCreateWithFlags(&plejoin_, cudaEventDisableTiming) != cudaSuccess ||
        cudaStreamCreateWithFlags(&g2s_, cudaStreamNonBlocking) != cudaSuccess ||
        cudaStreamCreateWithFlags(&cps_, cudaStreamNonBlocking) != cudaSuccess ||
        cudaStreamCreateWithFlags(&kst_, cudaStreamNonBlocking) != cudaSuccess ||
        cudaEventCreateWithFlags(&kjoin_, cudaEventDisableTiming) != cudaSuccess ||
        cudaEventCreateWithFlags(&iproj_, cudaEventDisableTiming) != cudaSuccess ||
        cudaStreamCreateWithFlags(&sgs_, cudaStreamNonBlocking) != cudaSuccess ||
        cudaEventCreateWithFlags(&sgjoin_, cudaEventDisableTiming) != cudaSuccess ||
        cudaEventCreateWithFlags(&cpfork_, cudaEventDisableTiming) != cudaSuccess ||
        cudaEventCreateWithFlags(&cpjoin_[0], cudaEventDisableTiming) != cudaSuccess ||
        cudaEventCreateWithFlags(&cpjoin_[1], cudaEventDisableTiming) != cudaSuccess ||
        cudaEventCreateWithFlags(&efork_, cudaEventDisableTiming) != cudaSuccess ||
        cudaEventCreateWithFlags(&rdone_, cudaEventDisableTiming) != cudaSuccess ||
        cudaEventCreateWithFlags(&g2fork_, cudaEventDisableTiming) != cudaSuccess ||
        cudaEventCreateWithFlags(&g2join_[0], cudaEventDisableTiming) != cudaSuccess ||
        cudaEventCreateWithFlags(&g2join_[1], cudaEventDisableTiming) != cudaSuccess ||
        cudaEventCreateWithFlags(&join_, cudaEventDisableTiming) != cudaSuccess ||
        cudaEventCreateWithFlags(&bfork_, cudaEventDisableTiming) != cudaSuccess ||
        cudaEventCreateWithFlags(&bjoin_, cudaEventDisableTiming) != cudaSuccess ||
        cudaEventCreateWithFlags(&pfork_, cudaEventDisableTiming) != cudaSuccess ||
        cudaEventCreateWithFlags(&pjoin_, cudaEventDisableTiming) != cudaSuccess ||
        cudaEventCreateWithFlags(&shfork_, cudaEventDisableTiming) != cudaSuccess ||
        cudaEventCreateWithFlags(&shjoin_, cudaEventDisableTiming) != cudaSuccess ||
        cudaEventCreateWithFlags(&res_ready_, cudaEventDisableTiming) != cudaSuccess ||
        cudaEventCreateWithFlags(&done_, cudaEventDisableTiming) != cudaSuccess) {
        err = "verify: stream create failed";
        return false;
    }
    std::fprintf(stderr, "strata verify: window up to %d tokens, %.1f MiB of device buffers\n", max_t,
                 (double) count.used / 1048576.0);
    return true;
}

const float* Verifier::final_R(int t) const { return R_ + (size_t) t * (size_t) (g_->hc * g_->n_embd); }

// ================================ THE WINDOW, AS CAPTURED ================================
//
// Plan v0.3 P6 (split window): with `groups_ == 2` the window's tokens are cut into two groups A = [0, T/2 up) and
// B = the rest, and the stream is ordered
//
//     pre(0,A) pre(0,B) | post(0,A) pre(1,A) | post(0,B) pre(1,B) | post(1,A) pre(2,A) | ...
//
// so the CPU computes A's experts of layer l while the GPU runs B's mixer and router of layer l, and B's experts
// while the GPU combines A and runs A's layer l+1.  B's mixer only needs A's mixer of the same layer (K/V, GDN
// state), never A's experts, so nothing waits that did not wait before.  Every token's arithmetic is unchanged.
bool Verifier::record_window(int T, cudaStream_t cs, std::string& err) {
    using namespace strata::kernels;
    const ModelGeometry& g = *g_;
    const WeightTable& wt = *wt_;
    SessionState& ss = *ss_;
    const int64_t N = g.n_embd, HC = g.hc, K = ss.k, C = g.ssm_conv_channels, ZV = g.ssm_value_dim;
    const int64_t HV = g.ssm_v_heads, HK = g.ssm_k_heads, NH = g.n_head, HD = g.head_dim, NKV = g.n_head_kv;
    const int64_t IQ = g.idx_q_heads, ID = g.idx_key_dim, NE = g.n_expert, MT = max_t_;
    const QsaShapes s = shapes_of(g);
    const GrShapes gs{g.n_embd, g.hc, g.hc_lr};
    const uint64_t gdn_floats = (uint64_t) g.ssm_state_size * g.ssm_v_heads * g.ssm_state_size +
                                (uint64_t) g.ssm_conv_channels * (g.ssm_d_conv - 1);
    const int64_t HS = (int64_t) NG_HIST * NG_HC_DIM;
    const int64_t TS = (s.idx_block - 1) * ID;
    const bool ple_on = ss.ple.ready();
    auto Rt = [&](int t) { return R_ + (size_t) t * HC * N; };
    const int G = (split_ && T >= 2) ? 2 : 1;
    const int tb_[2] = {0, (T + 1) / 2}, te_[2] = {G == 2 ? (T + 1) / 2 : T, T};
    groups_[T] = G;
    // --window-profile: row l < n_layers holds layer l's stages, row n_layers the window's own stamps
    auto stamp = [&](int64_t row, int i, cudaStream_t st) {
        if (stamps_ != nullptr && G == 1)
            gpu_stamp(stamps_ + ((size_t) T * (size_t) (g.n_layers + 1) + (size_t) row) * kStamps + i, st);
    };
    stamp(g.n_layers, 0, cs);
    if (gpu2_ && !quant_router_) {
        err = "verify: the second GPU takes the native router's q8_1 rows (a native pack)";
        return false;
    }
    // the layer's short branches: `fork` starts stream `to` from the main stream's work so far, `mark` records what
    // `from` has queued, `wait` makes the main stream wait for a mark
    auto fork = [&](cudaStream_t to, cudaEvent_t ev) {
        if (cudaEventRecord(ev, cs) == cudaSuccess && cudaStreamWaitEvent(to, ev, 0) == cudaSuccess) return true;
        err = "verify: a branch could not fork";
        return false;
    };
    auto mark = [&](cudaStream_t from, cudaEvent_t ev) {
        if (cudaEventRecord(ev, from) == cudaSuccess) return true;
        err = "verify: a branch could not join";
        return false;
    };
    auto wait = [&](cudaEvent_t ev) {
        if (cudaStreamWaitEvent(cs, ev, 0) == cudaSuccess) return true;
        err = "verify: a branch could not join";
        return false;
    };

    // ---- the window's inputs, from mapped staging; the residency snapshot on the shared expert's branch (the first
    // hit plan waits for it)
    copy_i32_from_mapped(tok_, m_tok_, T, cs);
    copy_i32_from_mapped(step_, m_step_, (int64_t) T * kStepCount, cs);
    copy_i32_from_mapped(pos_, m_pos_, (int64_t) T * NH, cs);
    if (cudaEventRecord(shfork_, cs) != cudaSuccess || cudaStreamWaitEvent(shs_, shfork_, 0) != cudaSuccess) {
        err = "verify: the residency copy could not fork";
        return false;
    }
    copy_from_mapped((float*) res_, (const float*) m_res_, res_words_, shs_);   // int32 words, copied as they are
    if (cudaEventRecord(res_ready_, shs_) != cudaSuccess) {
        err = "verify: the residency copy could not join";
        return false;
    }
    // the PLE block's key and value projections read only the window's PLE rows: on a branch from the window's start,
    // beside layer 0 (from jesdga95/basalt 2ad5813), for every token at once (each row's bits as in its own call);
    // layer 1 joins it
    const bool ple_beside = ple_on && ss.ple.w.key_bf16 != nullptr;
    if (ple_beside) {
        if (!fork(ples_, plefork_)) return false;
        wait_flag_ge(m_pleflag_, 1, ples_);   // the host reads the rows while layer 0 runs
        copy_from_mapped(ple_, m_ple_, (int64_t) T * N, ples_);
        try {
            bf16_gemv_fp32_mmvf_multi(ple_, ss.ple.w.key_bf16, ple_key_, N, NG_HC_DIM, T, ples_);
            bf16_gemv_fp32_mmvf_multi(ple_, ss.ple.w.value_bf16, ple_val_, N, N, T, ples_);
        } catch (const std::exception& e) {
            err = std::string("verify PLE: ") + e.what();
            return false;
        }
        if (!mark(ples_, plejoin_)) return false;
    }

    // ---- the embeddings, broadcast to the hc streams
    if (const NativeEmbed* ne = native_embed()) {       // plan v0.3 P6: the GGUF-form table
        ne->gather_dev(tok_, T, emb_, cs);
        broadcast_streams(emb_, R_, N, (int) HC, T, cs);
    } else {
        const WeightRef* w = wt.find("token_embd.weight");
        if (w == nullptr || w->codebook_iq4nl || (w->code_bits != 2 && w->code_bits != 4 && w->code_bits != 8)) {
            err = "verify: token_embd.weight is missing or not an S2/S4/S8 tensor";
            return false;
        }
        const auto* codes = (const uint8_t*) w->data;
        const auto* scales = (const float*) (codes + w->codes_bytes);
        const auto* offsets = w->has_offset ? (const float*) (codes + w->codes_bytes + w->scales_bytes) : nullptr;
        const uint64_t row_codes = (uint64_t) (w->ne0 / (8 / w->code_bits));
        const uint64_t row_groups = (uint64_t) (w->ne0 / w->group_elems);
        embedding_gather_dev(codes, scales, offsets, tok_, T, w->ne0, w->code_bits, w->code_bias, w->group_elems,
                             row_codes, row_groups, emb_, cs);
        broadcast_streams(emb_, R_, N, (int) HC, T, cs);
    }

    // set_predict: this layer's FFN read also estimates the next layer's FFN input (that layer's read of this state,
    // its gates their running average over the windows), and on a side branch forked after the read the next layer's
    // router runs on it with that layer's logit bias.  When learning, the FFN read of a layer the previous one
    // predicts keeps its gates' average, and the layer corrects its bias by its own logits on that branch.
    auto predicts = [&](int64_t l) {
        return predict_ != nullptr && G == 1 && l + 1 < g.n_layers && native_router_enabled();
    };
    auto corrects = [&](int64_t l) { return learn_ && l > 0 && predicts(l - 1); };
    auto branches = [&](int64_t l) { return predicts(l) || corrects(l); };

    // per-layer state indices (GDN and QSA layers are numbered separately)
    std::vector<int64_t> gdn_idx((size_t) g.n_layers, -1), qsa_idx((size_t) g.n_layers, -1);
    {
        int64_t qi = 0, gi = 0;
        for (int64_t l = 0; l < g.n_layers; ++l) {
            if (is_qsa_layer(g, l)) qsa_idx[(size_t) l] = qi++;
            else gdn_idx[(size_t) l] = gi++;
        }
    }

    // the shared expert: in a window of one token group on a branch of its own forked after the FFN read (`efork_`),
    // beside the router and the routed experts (joined before the combine); in a split window inline.  It ends with
    // its down projection: the combine applies its scalar gate, whose logits come from a branch of their own.
    const bool beside = G == 1;
    auto shared = [&](int64_t l, int grp) -> bool {
        const int tb = tb_[grp], te = te_[grp], n = te - tb;
        const LayerView v(wt, l);
        const WeightRef *wgi = need(v, "ffn_gate_inp_shexp.weight", err), *wsg = need(v, "ffn_gate_shexp.weight", err),
                        *wsu = need(v, "ffn_up_shexp.weight", err), *wsd = need(v, "ffn_down_shexp.weight", err);
        if (!wgi || !wsg || !wsu || !wsd) return false;
        if (!native_of(wsg, v.name("ffn_gate_shexp.weight"), err) || !native_of(wsu, v.name("ffn_up_shexp.weight"), err) ||
            !native_of(wsd, v.name("ffn_down_shexp.weight"), err))
            return false;
        NativeSharedWeights nsw;
        nsw.gate_type = wsg->native_type; nsw.gate_data = wsg->native_data;
        nsw.up_type = wsu->native_type; nsw.up_data = wsu->native_data;
        nsw.down_type = wsd->native_type; nsw.down_data = wsd->native_data;
        nsw.q8_1 = xq_;                                   // the main stream's next use is the next layer's mixer
        nsw.q8_1_il = xil_;
        cudaStream_t ws = cs, gs = cs;
        if (beside) {
            if (cudaStreamWaitEvent(shs_, efork_, 0) != cudaSuccess ||
                cudaStreamWaitEvent(sgs_, efork_, 0) != cudaSuccess) {
                err = "verify: the shared expert's branch could not fork";
                return false;
            }
            ws = shs_;
            gs = sgs_;
        }
        try {
            shared_expert_multi(n, mixed_ + tb * N, nsw, (const uint16_t*) wgi->data, sh_gate_ + (size_t) tb * g.n_ff,
                                sh_up_ + (size_t) tb * g.n_ff, sh_g_ + tb, shared_ + tb * N, N, g.n_ff, ws, gs);
        } catch (const std::exception& e) {
            err = std::string("verify shared expert: ") + e.what();
            return false;
        }
        if (beside) {
            stamp(l, kLayerStamps, shs_);
            if (cudaEventRecord(shjoin_, shs_) != cudaSuccess || cudaEventRecord(sgjoin_, sgs_) != cudaSuccess) {
                err = "verify: the shared expert's branch could not join";
                return false;
            }
        }
        return true;
    };

    // ---------------------------------------------------------------- pre(l, group): up to the ring
    auto pre = [&](int64_t l, int grp) -> bool {
        const int tb = tb_[grp], te = te_[grp], n = te - tb;
        stamp(l, 0, cs);
        const LayerView v(wt, l);
        const char* pfx[2] = {"hc_attn_", "hc_ffn_"};
        const WeightRef *wn[2], *wd[2], *wu[2], *wi[2];
        for (int h = 0; h < 2; ++h) {
            wn[h] = need(v, (std::string(pfx[h]) + "norm.weight").c_str(), err);
            wd[h] = need(v, (std::string(pfx[h]) + "down.weight").c_str(), err);
            wu[h] = need(v, (std::string(pfx[h]) + "up.weight").c_str(), err);
            wi[h] = need(v, (std::string(pfx[h]) + "inject.weight").c_str(), err);
            if (!wn[h] || !wd[h] || !wu[h] || !wi[h]) return false;
        }
        bool pending = l > 0;   // the previous layer's FFN write, folded into this layer's first read
        // the projections of the group's tokens: 2+ tokens are quantized with the interleaved copy the multi-token
        // kernels read
        auto quant = [&](const float* x, int64_t n_in) {
            if (n >= 2) native_quantize_q8_1_il(x, xq_, xil_, (int) n_in, n, cs);
            else native_quantize_q8_1(x, xq_, (int) n_in, n, cs);
        };
        auto mm = [&](int type, const void* w, float* y, int64_t n_in, int64_t n_out, cudaStream_t ps) {
            native_mmvq_il(type, w, xq_, xil_, y, (int) n_in, (int) n_out, n, ps);
        };
        auto proj = [&](const WeightRef* w, float* y, int64_t n_in, int64_t n_out, cudaStream_t ps) {
            mm(w->native_type, w->native_data, y, n_in, n_out, ps);
        };
        if (l == 1 && ple_on) {
            if (grp == 0 && ple_beside) {
                if (!wait(plejoin_)) return false;
            } else if (grp == 0) {                         // the host reads the rows while layer 0 runs
                wait_flag_ge(m_pleflag_, 1, cs);
                copy_from_mapped(ple_, m_ple_, (int64_t) T * N, cs);
            }
            for (int t = tb; t < te; ++t) gr_write(Rt(t), bo_ + t * N, inj2_ + t * HC, gs, Rt(t), cs);
            // the PLE block: the group's projections with each weight read once, then the rest of the block over its
            // tokens in order, which leaves the history after each of them for the commit
            const PleWeights& pw = ss.ple.w;
            const float* emb = ple_ + tb * N;
            float* key = ple_key_ + (size_t) tb * NG_HC_DIM;
            float* norm = ple_norm_ + (size_t) tb * NG_HC_DIM;
            try {
                if (!ple_beside) {
                    quant(emb, N);
                    mm(pw.key_native_type, pw.key_native_data, key, N, NG_HC_DIM, cs);
                    bf16_gemv_fp32_mmvf_multi(emb, pw.value_bf16, ple_val_ + tb * N, N, N, n, cs);
                }
                const NativePleTokensBuffers pb{ple_nkey_ + (size_t) tb * NG_HC_DIM, norm, ple_gate_ + tb * HC,
                                                ple_gated_ + (size_t) tb * NG_HC_DIM, norm, Rt(tb)};
                native_ple_postops_tokens(key, Rt(tb), ple_val_ + tb * N, ss.ple.hist, pw, pb, n, cs,
                                          hist_snap_ + (size_t) tb * HS);
            } catch (const std::exception& e) {
                err = std::string("verify PLE: ") + e.what();
                return false;
            }
            pending = false;
        }
        auto gr_read_group = [&](int half, bool apply, float* inj_prev, float* inj_out, float* gate_ema,
                                 const float* est_norm, const float* est_gates) {
            FusedGrArgs fa[kFusedGrMaxT];
            for (int t = tb; t < te; ++t) {
                FusedGrArgs& a = fa[t - tb];
                a.R = Rt(t); a.R_out = Rt(t); a.apply = apply;
                a.bo_prev = bo_ + t * N; a.inj_prev = inj_prev + t * HC;
                a.w_norm = (const float*) wn[half]->data; a.w_down = (const uint16_t*) wd[half]->data;
                a.w_up = (const uint16_t*) wu[half]->data; a.w_inject = (const uint16_t*) wi[half]->data;
                a.eps = EPS; a.lo = lo_ + t * g.hc_lr; a.rs = rs_ + t * HC;
                a.inject_out = inj_out + t * HC; a.mixed = mixed_ + t * N; a.gate_ema = gate_ema;
                a.est_norm = est_norm; a.est_gates = est_gates; a.est = pmixed_ + t * N;
            }
            fused_gr_read_multi(fa, n, grs_, cs);
        };
        gr_read_group(0, pending, inj2_, inj_, nullptr, nullptr, nullptr);
        stamp(l, 1, cs);
        float* xm = mixed_ + tb * N;
        try {
            if (!is_qsa_layer(g, l)) {
                // ======================= GDN =======================
                const WeightRef *wqkv = need(v, "attn_qkv.weight", err), *wg = need(v, "attn_gate.weight", err),
                                *wout = need(v, "ssm_out.weight", err), *wa = need(v, "ssm_alpha.weight", err),
                                *wb = need(v, "ssm_beta.weight", err), *wc = need(v, "ssm_conv1d.weight", err),
                                *wnm = need(v, "ssm_norm.weight", err), *wdt = need(v, "ssm_dt.bias", err),
                                *wsa = need(v, "ssm_a", err);
                if (!wqkv || !wg || !wout || !wa || !wb || !wc || !wnm || !wdt || !wsa) return false;
                if (!native_of(wqkv, v.name("attn_qkv.weight"), err) || !native_of(wg, v.name("attn_gate.weight"), err) ||
                    !native_of(wout, v.name("ssm_out.weight"), err))
                    return false;
                const int64_t gi = gdn_idx[(size_t) l];
                float* state = ss.gdn_state + (size_t) gi * gdn_floats;
                float* conv = state + (uint64_t) g.ssm_state_size * g.ssm_v_heads * g.ssm_state_size;
                float* qkv = qkv_L_ + (size_t) gi * MT * C;
                float* hb = h_L_ + (size_t) gi * MT * C;
                float* gate = gate_L_ + (size_t) gi * MT * HV;
                float* beta = beta_L_ + (size_t) gi * MT * HV;
                // beside qkv and the conv: alpha/beta (a few blocks, latency-bound) and the gate projection, joined
                // before the recurrence.  qkv's grid is recorded first, so it is dispatched first and the conv runs
                // beside the gate projection's last blocks.
                if (!fork(side_, bfork_)) return false;
                gdn_ab_multi(xm, (const uint16_t*) wa->data, (const uint16_t*) wb->data, (const float*) wdt->data,
                             (const float*) wsa->data, gate + (size_t) tb * HV, beta + (size_t) tb * HV, (int) N, (int) HV,
                             n, side_);
                if (!mark(side_, bjoin_)) return false;
                quant(xm, N);
                if (!fork(shs_, pfork_)) return false;
                proj(wqkv, qkv + (size_t) tb * C, N, C, cs);
                proj(wg, z_ + (size_t) tb * ZV, N, ZV, shs_);
                if (!mark(shs_, pjoin_)) return false;
                // a one-token window keeps its token: it advances the conv history and the state itself (commit)
                gdn_conv_l2_multi(conv, qkv, (const float*) wc->data, hb, (int) C, (int) (2 * HK), EPS, n, cs, tb,
                                  T == 1);
                if (!wait(bjoin_) || !wait(pjoin_)) return false;
                // the recurrence from the untouched state over tokens [0, te); outputs only for this group's
                gdn_step_norm_multi(state, hb, (int) C, gate, beta, z_, (const float*) wnm->data, EPS, y_, (int) HK,
                                    (int) HV, te, T == 1 ? one_ : nullptr, cs, tb);
                quant(y_ + (size_t) tb * ZV, ZV);
                proj(wout, bo_ + tb * N, ZV, N, cs);
            } else {
                // ======================= QSA =======================
                const int64_t qi = qsa_idx[(size_t) l];
                const QsaState& st = ss.qsa_states[qi];
                const WeightRef *wik = need(v, "indexer.k_proj.weight", err), *wq = need(v, "attn_q.weight", err),
                                *wk = need(v, "attn_k.weight", err), *wv = need(v, "attn_v.weight", err),
                                *wo = need(v, "attn_output.weight", err), *wiq = need(v, "indexer.q_proj.weight", err),
                                *wqn = need(v, "attn_q_norm.weight", err), *wkn = need(v, "attn_k_norm.weight", err),
                                *wiqn = need(v, "indexer.q_norm.weight", err), *wikn = need(v, "indexer.k_norm.weight", err);
                if (!wik || !wq || !wk || !wv || !wo || !wiq || !wqn || !wkn || !wiqn || !wikn) return false;
                if (!native_of(wq, v.name("attn_q.weight"), err) || !native_of(wk, v.name("attn_k.weight"), err) ||
                    !native_of(wv, v.name("attn_v.weight"), err) || !native_of(wo, v.name("attn_output.weight"), err))
                    return false;
                // the rms norm and rope of the group's heads in one launch: rows of `cols` read at in + r * stride,
                // token t's positions at pos_ + t * NH
                auto norm_rope = [&](const float* in, int stride, float* out, const WeightRef* norm, int heads,
                                     int cols, cudaStream_t ns) {
                    native_qsa_norm_rope_tokens(in, stride, (const float*) norm->data, out, cols, n * heads, EPS,
                                                (int) s.n_rot, (float) qsa_freq_base(), pos_ + tb * NH, heads, (int) NH,
                                                ns);
                };
                // three branches beside the values, joined before the attention: the indexer (its key and query
                // projections in one launch, the keys appended, the queries' block scores and top-k), the keys and the
                // queries.  A grid is dispatched whole before a later kernel's blocks, so the keys' and values'
                // projections go ahead of the queries' grid, and that grid waits for the indexer's projections: the
                // append is dispatched ahead of it, and the indexer's last kernels end about when the queries' do.
                if (!fork(side_, bfork_)) return false;
                float* idx_raw = idx_raw_L_ + (size_t) qi * MT * ID;
                bf16_gemv_fp32_mmvf_multi(xm, (const uint16_t*) wik->data, idx_raw + tb * ID, N, ID, n, side_,
                                          (const uint16_t*) wiq->data, qidx_ + tb * IQ * ID, IQ * ID);
                if (!mark(side_, iproj_)) return false;
                const QsaIndexerBuffers ib{st.idx_tail, st.idx_dead, st.idx_pooled, st.idx_block_pos};
                native_qsa_indexer_append_multi(idx_raw + tb * ID, step_ + tb * kStepCount + kStepPos, (int) kStepCount, n,
                                                0, (const float*) wikn->data, EPS, ib, s, st.max_cells,
                                                (float) qsa_freq_base(), side_,
                                                grp == 0 ? tail_snap_ + (size_t) qi * TS : nullptr);
                norm_rope(qidx_ + tb * IQ * ID, (int) ID, qidx_ + tb * IQ * ID, wiqn, (int) IQ, (int) ID, side_);
                qsa_block_scores(st.idx_pooled, st.idx_dead, qidx_ + tb * IQ * ID, step_ + tb * kStepCount, n, max_blocks_,
                                 s, scores_ + (size_t) tb * max_blocks_, side_, qsa_score_grid_blocks);
                qsa_block_topk(scores_ + (size_t) tb * max_blocks_, step_ + tb * kStepCount, n, max_blocks_, cap_, s,
                               sel_ + (size_t) tb * cap_, side_);
                if (!mark(side_, bjoin_)) return false;
                quant(xm, N);
                if (cudaEventRecord(pfork_, cs) != cudaSuccess || cudaStreamWaitEvent(kst_, pfork_, 0) != cudaSuccess ||
                    cudaStreamWaitEvent(shs_, pfork_, 0) != cudaSuccess ||
                    cudaStreamWaitEvent(shs_, iproj_, 0) != cudaSuccess) {
                    err = "verify: a branch could not fork";
                    return false;
                }
                proj(wk, kcur_ + tb * NKV * HD, N, NKV * HD, kst_);
                norm_rope(kcur_ + tb * NKV * HD, (int) HD, kcur_ + tb * NKV * HD, wkn, (int) NKV, (int) HD, kst_);
                if (!mark(kst_, kjoin_)) return false;
                proj(wv, vcur_ + tb * NKV * HD, N, NKV * HD, cs);
                proj(wq, qfull_ + tb * NH * 2 * HD, N, NH * 2 * HD, shs_);
                // the query halves of every token's heads (rows of 2 * HD: query, then gate)
                norm_rope(qfull_ + tb * NH * 2 * HD, (int) (2 * HD), qcur_ + tb * NH * HD, wqn, (int) NH, (int) HD,
                          shs_);
                if (!mark(shs_, pjoin_) || !wait(kjoin_)) return false;
                qsa_kv_append_steps(st, g, step_ + tb * kStepCount, (int) kStepCount, kcur_ + tb * NKV * HD,
                                    vcur_ + tb * NKV * HD, n, cs);
                if (!wait(bjoin_) || !wait(pjoin_)) return false;
                const QsaAttnPools pools = qsa_attn_pools(st);
                qsa_decode_attn_batch(qcur_ + tb * NH * HD, pools, sel_ + (size_t) tb * cap_, step_ + tb * kStepCount, cap_,
                                      s, attn_scratch_ + (size_t) tb * attn_scratch_floats_, attn32_ + tb * NH * HD,
                                      n, cs, qfull_ + tb * NH * 2 * HD);
                quant(attn32_ + tb * NH * HD, NH * HD);
                proj(wo, bo_ + tb * N, NH * HD, N, cs);
            }
        } catch (const std::exception& e) {
            err = "verify layer " + std::to_string(l) + ": " + e.what();
            return false;
        }
        stamp(l, 2, cs);
        // the FFN read, and the next layer's estimated FFN input for the prediction (set_predict)
        const WeightRef* wn1 = predicts(l) ? need(LayerView(wt, l + 1), "hc_ffn_norm.weight", err) : nullptr;
        if (predicts(l) && !wn1) return false;
        gr_read_group(1, true, inj_, inj2_, corrects(l) ? gate_ema_ + (size_t) l * HC * N : nullptr,
                      wn1 != nullptr ? (const float*) wn1->data : nullptr,
                      wn1 != nullptr ? gate_ema_ + (size_t) (l + 1) * HC * N : nullptr);
        stamp(l, 3, cs);
        // the routed experts' activations: from the router kernel, else beside it (joined before the experts)
        if (!quant_router_) {
            if (!fork(side_, bfork_)) return false;
            if (strata::kernels::cpu::expert_layout().native)
                quantize_q8_1_rows(xm, n, N, nat_xq_ + (size_t) tb * (N / 32) * 36, side_);
            else
                quantize_q8_0_scaled(xm, hit_xq_ + (size_t) tb * (N / 32) * 34, hit_xs_ + (size_t) tb * (N / 32),
                                     (int64_t) n * N, side_);
            if (!mark(side_, bjoin_)) return false;
        }
        // the branches that need only the FFN read fork here; they are recorded after the router, whose blocks are
        // then dispatched first, and run beside its routing, the ring and the hit plan
        if (beside && cudaEventRecord(efork_, cs) != cudaSuccess) {
            err = "verify: the FFN read's branches could not fork";
            return false;
        }
        // the router, the doorbell and the main GPU's routed experts - the ones its VRAM tier holds, decided here from
        // the residency snapshot the pool reads too (it leaves their rows to the GPU)
        if (l == 0 && grp == 0 && cudaStreamWaitEvent(cs, res_ready_, 0) != cudaSuccess) {
            err = "verify: the residency copy could not join";
            return false;
        }
        int32_t* hit_plan = dplan_ + (size_t) grp * (size_t) plan_stride_;
        if (native_router_enabled()) {             // one kernel: logits, top 10, ring, hit plan
            const WeightRef* wr = need(v, "ffn_gate_inp.weight", err);
            if (!wr) return false;
            VerifyRouterArgs ra;
            ra.x = xm; ra.w = (const uint16_t*) wr->data; ra.logits = logits_ + tb * NE;
            ra.ids = ids_ + tb * K; ra.weights = w_ + tb * K;
            ra.x_out = xk_layer_[(size_t) l] ? nullptr : m_x_ + tb * N;
            ra.ids_out = m_ids_ + tb * K; ra.w_out = m_w_ + tb * K; ra.seq = m_seq_;
            if (quant_router_) {   // the q8_1 rows for both GPUs' experts, the Q8_K rows for the CPU's
                ra.xq1 = nat_xq_ + (size_t) tb * (N / 32) * 36;
                ra.xq1_out = gpu2_ ? m_xq1_ + (size_t) tb * (N / 32) * 36 : nullptr;
                ra.xk_out = xk_layer_[(size_t) l] ? m_xk_ + (size_t) tb * (size_t) xk_stride_ : nullptr;
                ra.xk_stride = (int) xk_stride_;
            }
            ra.ring = (uint32_t) (l * G + grp + 1);   // the rings so far (the host waits for this one)
            ra.res = res_ + (size_t) l * NE; ra.slot_ptr = slot_ptr_; ra.plan = hit_plan;
            ra.cap = (int) (MT * K); ra.ptr_off = (int) plan_ptr_off_;
            ra.counter = ring_count_; ra.n_tok = n; ra.n_embd = (int) N; ra.n_expert = (int) NE;
            try {
                verify_router(ra, cs);
            } catch (const std::exception& e) {
                err = "verify layer " + std::to_string(l) + " router: " + e.what();
                return false;
            }
        } else {
            for (int t = tb; t < te; ++t) {
                MoEBuffers mb = ss.moe;
                mb.logits = logits_ + t * NE; mb.ids = ids_ + t * K; mb.weights = w_ + t * K;
                if (!moe_route(wt, g, l, K, mb, mixed_ + t * N, cs, err, nullptr)) return false;
            }
            doorbell_publish(xm, ids_ + tb * K, w_ + tb * K, (int64_t) n * N, (int64_t) n * K, m_x_ + tb * N,
                             m_ids_ + tb * K, m_w_ + tb * K, m_seq_, cs);
            verify_hit_plan(ids_ + tb * K, (int) (n * K), (int) K, res_ + (size_t) l * NE, (int) NE, slot_ptr_, hit_plan,
                            MT * K, plan_ptr_off_, cs);
        }
        stamp(l, 4, cs);
        // after the doorbell: the second GPU's rows of this token group, taken into VRAM once their flag rises, and
        // the wait for the CPU's rows, each on a branch beside the routed experts; then the branches forked after the
        // FFN read: the shared expert, and the next layer's prediction
        if (gpu2_) {
            if (!fork(g2s_, g2fork_)) return false;
            wait_flag_ge(m_flag2_ + (size_t) grp * 16, (uint32_t) (l * G + grp + 1), g2s_);
            fetch_listed_rows(m_list2_ + grp * list2_stride_, m_ymiss_ + (size_t) tb * K * N,
                              hit_out_ + (size_t) tb * K * N, list2_ + grp * list2_stride_, (int) (n * K), N, g2s_);
            stamp(l, kLayerStamps + 1, g2s_);
            if (!mark(g2s_, g2join_[grp])) return false;
        }
        if (!fork(cps_, cpfork_)) return false;
        wait_flag_ge(m_flag_, (uint32_t) (l * G + grp + 1), cps_);   // the CPU's share is in the mapped rows
        if (!mark(cps_, cpjoin_[grp])) return false;
        if (beside && cudaEventRecord(rdone_, cs) != cudaSuccess) {
            err = "verify: the router's end could not be marked";
            return false;
        }
        if (beside && !shared(l, grp)) return false;
        if (branches(l) && cudaStreamWaitEvent(side_, efork_, 0) != cudaSuccess) {
            err = "verify: the prediction branch could not fork";
            return false;
        }
        if (predicts(l)) {
            const WeightRef* wr1 = need(LayerView(wt, l + 1), "ffn_gate_inp.weight", err);
            if (!wr1) return false;
            const size_t half = (size_t) ((l + 1) & 1) * (size_t) (MT * K);
            VerifyRouterArgs ra;
            ra.x = pmixed_; ra.w = (const uint16_t*) wr1->data;
            ra.logits = plogits_ + (size_t) ((l + 1) & 1) * MT * NE; ra.ids = pids_; ra.weights = pw_;
            ra.ids_out = m_pids_ + half; ra.w_out = m_pw_ + half; ra.seq = m_pseq_;
            ra.ring = (uint32_t) (l + 1);              // the predictions published so far
            ra.counter = ring_count_ + 1; ra.n_tok = n; ra.n_embd = (int) N; ra.n_expert = (int) NE;
            if (learn_) ra.bias = pbias_ + (size_t) (l + 1) * NE;
            try {
                verify_router(ra, side_);
            } catch (const std::exception& e) {
                err = "verify layer " + std::to_string(l) + " prediction: " + e.what();
                return false;
            }
        }
        if (corrects(l)) {   // after the prediction and the router, from the prediction the previous layer made
            if (cudaStreamWaitEvent(side_, rdone_, 0) != cudaSuccess) {
                err = "verify: the prediction's bias could not wait for the router";
                return false;
            }
            try {
                verify_router_bias_update(logits_ + tb * NE, plogits_ + (size_t) (l & 1) * MT * NE, n,
                                          pbias_ + (size_t) l * NE, side_);
            } catch (const std::exception& e) {
                err = "verify layer " + std::to_string(l) + " prediction bias: " + e.what();
                return false;
            }
        }
        if (branches(l) && cudaEventRecord(join_, side_) != cudaSuccess) {
            err = "verify: the prediction branch could not join";
            return false;
        }
        if (!quant_router_ && !wait(bjoin_)) return false;
        stamp(l, 5, cs);
        return beside || shared(l, grp);
    };

    // ---------------------------------------------------------------- post(l, group): experts, combine
    auto post = [&](int64_t l, int grp) -> bool {
        const int tb = tb_[grp], te = te_[grp], n = te - tb;
        const uint32_t ring = (uint32_t) (l * G + grp + 1);
        const int64_t cap = (int64_t) n * K, capx = (int64_t) max_t_ * K;
        struct Plan {   // a plan block's fields (`init`)
            const int32_t *counts = nullptr, *start = nullptr, *dst = nullptr, *tok = nullptr, *start2 = nullptr;
            const unsigned long long *ptr = nullptr, *ptr2 = nullptr;
        };
        auto plan_at = [&](const int32_t* pl) {
            Plan p;
            p.counts = pl;
            p.start = pl + 4;
            p.dst = p.start + capx + 1;
            p.tok = p.dst + capx;
            p.ptr = (const unsigned long long*) (pl + plan_ptr_off_);
            p.ptr2 = p.ptr + capx;
            p.start2 = pl + plan_ptr_off_ + 4 * capx;
            return p;
        };
        float* hit_out = hit_out_ + (size_t) tb * K * N;
        const auto& lay = strata::kernels::cpu::expert_layout();
        bool xk_ready = false;   // --expert-math exact: this group's Q8_K rows quantized for the hits
        auto grouped = [&](const Plan& p, const unsigned long long* gp, const int32_t* gs, const int32_t* gn) {
            if (lay.native) {
                // the layer's GGUF formats (i-quant gate/up, Q2_0 / IQ4_NL down)
                const auto& f = lay.fmt[(size_t) l];
                const NativeExpertLayout L = native_expert_layout(f.gu_type, f.d_type, f.n_embd, f.n_ff,
                                                                  gpu_expert_act(f.gu_act), gpu_expert_act(f.d_act));
                // --expert-math exact on a K-quant layer: the CPU's Q8_K rows, quantized here from the FFN input
                // (quantize_q8_K is bitwise ggml's quantize_row_q8_K, as the router kernel's rows for the pool are)
                const uint8_t* xa = nat_xq_ + (size_t) tb * (N / 32) * 36;
                int64_t xs = (int64_t) (N / 32) * 36;
                if (L.gu_act == 15) {
                    xs = (int64_t) (N / 256) * 292;
                    xa = xk_dev_ + (size_t) tb * (size_t) xs;
                    if (!xk_ready) quantize_q8_K(mixed_ + (size_t) tb * N, xk_dev_ + (size_t) tb * (size_t) xs,
                                                 (int64_t) n * N, cs);
                    xk_ready = true;
                }
                native_expert_grouped(L, gp, gs, gn, p.dst, p.tok, cap, cap, xa, xs, hit_scratch_, hit_out, cs);
            } else {
                moe_grouped_s2(gp, gs, gn, p.dst, p.tok, cap, cap, hit_xq_ + (size_t) tb * (N / 32) * 34,
                               hit_xs_ + (size_t) tb * (N / 32), hit_scratch_, hit_out, cs);
            }
        };
        // plan v0.3 P6: the VRAM groups at once; the PCIe groups once the pool has published them and the copy engine
        // has landed them in staging
        const Plan vram = plan_at(dplan_ + (size_t) grp * (size_t) plan_stride_);
        grouped(vram, vram.ptr, vram.start, vram.counts);
        stamp(l, 6, cs);
        Plan pcie;
        if (pcie_share_) {
            wait_flag_ge(m_flagA_, ring, cs);                  // the pool published this group's PCIe share
            stamp(l, 7, cs);
            int32_t* pl = plan_ + (size_t) grp * (size_t) plan_stride_;
            copy_i32_from_mapped(pl, m_plan_ + (size_t) grp * (size_t) plan_i32_, plan_i32_, cs);
            pcie = plan_at(pl);
            wait_flag_ge(m_flagB_, ring, cs);                  // it is in staging (DMA) or mapped
            if (sink_.pcie_mode == 2) {                        // stage it with a copy kernel, then point at staging
                const int64_t per = G == 2 ? kStagingBlobs / 2 : kStagingBlobs;
                uint8_t* stage = staging_ + (size_t) (grp * per) * lay.max_blob;
                fetch_blobs(pcie.ptr2, pcie.counts + 2, stage, (int64_t) lay.blob_bytes(l), (int) per, cs);
                rebase_ptrs((unsigned long long*) pcie.ptr2, pcie.counts + 2, stage, (int64_t) lay.blob_bytes(l), cs);
            }
            grouped(pcie, pcie.ptr2, pcie.start2, pcie.counts + 2);
        } else {
            stamp(l, 7, cs);
        }
        stamp(l, 8, cs);
        if (!wait(cpjoin_[grp])) return false;                 // the CPU's share is in the mapped rows
        stamp(l, 9, cs);
        if (gpu2_ && !wait(g2join_[grp])) return false;        // the second GPU's, in `hit_out`
        stamp(l, 10, cs);
        if (beside && (cudaStreamWaitEvent(cs, shjoin_, 0) != cudaSuccess ||
                       cudaStreamWaitEvent(cs, sgjoin_, 0) != cudaSuccess)) {
            err = "verify: the shared expert's branch could not join";
            return false;
        }
        // only the CPU's rows cross PCIe here; the GPUs' come from `hit_out` (the pool leaves their rows unwritten)
        const int32_t* list2 = gpu2_ ? list2_ + grp * list2_stride_ : nullptr;
        try {
            native_moe_gather_combine(hit_out, m_ymiss_ + (size_t) tb * K * N, vram.dst, vram.counts + 1, pcie.dst,
                                      pcie.counts != nullptr ? pcie.counts + 1 : nullptr,
                                      list2 != nullptr ? list2 + 4 : nullptr, list2, w_ + tb * K, shared_ + tb * N,
                                      sh_g_ + tb, bo_ + tb * N, N, K, n, cs);
        } catch (const std::exception& e) {
            err = "verify layer " + std::to_string(l) + " combine: " + e.what();
            return false;
        }
        stamp(l, 11, cs);
        if (branches(l) && cudaStreamWaitEvent(cs, join_, 0) != cudaSuccess) {
            err = "verify: the prediction branch could not join";
            return false;
        }
        if (l == g.n_layers - 1)
            for (int t = tb; t < te; ++t) gr_write(Rt(t), bo_ + t * N, inj2_ + t * HC, gs, Rt(t), cs);
        stamp(l, 12, cs);
        return true;
    };

    stamp(g.n_layers, 1, cs);
    for (int grp = 0; grp < G; ++grp)
        if (!pre(0, grp)) return false;
    for (int64_t l = 0; l < g.n_layers; ++l)
        for (int grp = 0; grp < G; ++grp) {
            if (!post(l, grp)) return false;
            if (l + 1 < g.n_layers && !pre(l + 1, grp)) return false;
        }
    stamp(g.n_layers, 2, cs);

    // ---- the head, T columns, and the argmax of each
    {
        const WeightRef *hn = wt.find("output_hc_norm.weight"), *hd = wt.find("output_hc_down.weight"),
                        *hu = wt.find("output_hc_up.weight");
        if (!hn || !hd || !hu) { err = "verify: an output_hc_* weight is missing"; return false; }
        if (head_ != nullptr && head_->loaded()) {   // the final mixer, the window's tokens in one read
            if (hn->kind != WeightKind::F32 || hd->kind != WeightKind::Bf16InF32 || hu->kind != WeightKind::Bf16InF32) {
                err = "verify: the output_hc_* weights have the wrong engine forms";
                return false;
            }
            FusedGrArgs fa[kFusedGrMaxT];
            for (int t = 0; t < T; ++t) {
                fa[t].R = Rt(t); fa[t].R_out = Rt(t); fa[t].apply = false;
                fa[t].w_norm = (const float*) hn->data; fa[t].w_down = (const uint16_t*) hd->data;
                fa[t].w_up = (const uint16_t*) hu->data; fa[t].eps = EPS;
                fa[t].lo = lo_ + t * g.hc_lr; fa[t].rs = rs_ + t * HC; fa[t].mixed = head_mixed_ + t * N;
            }
            fused_gr_read_multi(fa, T, grs_, cs);
        } else {
            for (int t = 0; t < T; ++t) {
                BlockBuffers bb = ss.block;
                bb.R = Rt(t);
                bb.mixed = head_mixed_ + t * N;
                if (!lm_head(wt, g, bb, head_logits_ + (size_t) t * n_vocab_, cs, err)) return false;
            }
        }
        if (head_ != nullptr && head_->loaded()) {
            try {
                if (shead_ != nullptr) {   // the second GPU's part starts once the host sees the count
                    copy_from_mapped(shead_->input(), head_mixed_, (int64_t) T * N, cs);
                    mapped_bump(shead_->handoff(), cs);
                }
                if (T >= 2) {
                    native_quantize_q8_1_il(head_mixed_, xq_, xil_, (int) N, T, cs);
                    native_mmvq_il(head_->type(), head_->weights(), xq_, xil_, head_logits_, (int) N, (int) head_rows_,
                                   T, cs);
                } else {
                    native_quantize_q8_1(head_mixed_, xq_, (int) N, T, cs);
                    native_mmvq(head_->type(), head_->weights(), xq_, head_logits_, (int) N, (int) head_rows_, T, cs);
                }
            } catch (const std::exception& e) {
                err = std::string("verify head: ") + e.what();
                return false;
            }
        }
        argmax_rows(head_logits_, T, (int) head_rows_, arg_scratch_, m_out_, cs, m_val_, nan_guard_ ? m_bad_ : nullptr);
    }
    if (host_rows_) copy_from_mapped(m_rows_, R_, (int64_t) T * HC * N, cs);   // a plain copy kernel, here to host
    stamp(g.n_layers, 3, cs);
    stamp(g.n_layers, 4, cs);   // right after the last: one stamp's own cost
    return true;
}

bool Verifier::capture(int T, std::string& err) {
    if (exec_[T] != nullptr) return true;
    if (profile_ && stamps_ == nullptr &&
        cudaMalloc(&stamps_, (size_t) (strata::kernels::kVerifyMaxT + 1) * (size_t) (g_->n_layers + 1) * kStamps *
                                 sizeof(unsigned long long)) != cudaSuccess) {
        err = "verify: the profile's stamp buffer does not fit";
        return false;
    }
    if (cudaStreamBeginCapture(cs_, cudaStreamCaptureModeThreadLocal) != cudaSuccess) {
        err = "verify: begin capture failed";
        return false;
    }
    std::string rerr;
    const bool ok = record_window(T, cs_, rerr);
    cudaGraph_t graph = nullptr;
    const cudaError_t ce = cudaStreamEndCapture(cs_, &graph);
    if (!ok) {
        if (graph) cudaGraphDestroy(graph);
        err = rerr;
        return false;
    }
    if (ce != cudaSuccess) {
        err = std::string("verify: end capture: ") + cudaGetErrorString(ce);
        return false;
    }
<<<<<<< this fork
#if defined(STRATA_USE_HIP)   // 143-amdk: the same listing on HIP (kernel names by hipKernelNameRefByPtr); STRATA_VERIFY_NODES=2 also lists the first nodes in graph order
    if (std::getenv("STRATA_VERIFY_NODES") != nullptr) {
        size_t nn = 0;
        hipGraphGetNodes(graph, nullptr, &nn);
        std::vector<hipGraphNode_t> nodes(nn);
        hipGraphGetNodes(graph, nodes.data(), &nn);
        std::map<std::string, int> kinds;
        std::string seq;
        size_t shown = 0;
        for (hipGraphNode_t nd : nodes) {
            hipGraphNodeType ty;
            hipGraphNodeGetType(nd, &ty);
            std::string name = "type" + std::to_string((int) ty);
            if (ty == hipGraphNodeTypeKernel) {
                hipKernelNodeParams kp{};
                if (hipGraphKernelNodeGetParams(nd, &kp) == hipSuccess && kp.func != nullptr) {
#if defined(_WIN32)
                    using Fn_hipKernelNameRefByPtr = const char* (*)(const void*, hipStream_t);
                    static auto p_hipKernelNameRefByPtr = reinterpret_cast<Fn_hipKernelNameRefByPtr>(
                        GetProcAddress(GetModuleHandleA("amdhip64_7.dll"), "hipKernelNameRefByPtr"));
                    const char* fn = p_hipKernelNameRefByPtr ? p_hipKernelNameRefByPtr(kp.func, nullptr) : nullptr;
#else
                    const char* fn = hipKernelNameRefByPtr(kp.func, nullptr);
#endif
                    if (fn != nullptr) name = fn;
                }
            } else if (ty == hipGraphNodeTypeMemcpy) name = "memcpy";
            else if (ty == hipGraphNodeTypeMemset) name = "memset";
            ++kinds[name];
            if (std::atoi(std::getenv("STRATA_VERIFY_NODES")) >= 2 && shown++ < 400) seq += "\n  " + name.substr(0, 110);
        }
        std::vector<std::pair<int, std::string>> v;
        for (auto& [k2, c] : kinds) v.push_back({c, k2});
        std::sort(v.rbegin(), v.rend());
        std::fprintf(stderr, "strata verify: the %d-token window graph has %zu nodes:", T, nn);
        for (size_t i = 0; i < v.size() && i < 80; ++i) std::fprintf(stderr, "\n %4d x %.110s", v[i].first, v[i].second.c_str());
        std::fprintf(stderr, "\n%s\n", seq.c_str());
    }
#endif
#if !defined(STRATA_USE_HIP) && !defined(STRATA_GPU_METAL)   // a CUDA debug listing (node types, kernel names)
    if (std::getenv("STRATA_VERIFY_NODES") != nullptr) {   // what the window graph holds
        size_t nn = 0;
        cudaGraphGetNodes(graph, nullptr, &nn);
        std::vector<cudaGraphNode_t> nodes(nn);
        cudaGraphGetNodes(graph, nodes.data(), &nn);
        std::map<std::string, int> kinds;
        for (cudaGraphNode_t nd : nodes) {
            cudaGraphNodeType ty;
            cudaGraphNodeGetType(nd, &ty);
            std::string name = "type" + std::to_string((int) ty);
            if (ty == cudaGraphNodeTypeKernel) {
                cudaKernelNodeParams kp{};
                if (cudaGraphKernelNodeGetParams(nd, &kp) == cudaSuccess) {
#if CUDART_VERSION >= 12030   // cudaFuncGetName arrived in CUDA 12.3
                    const char* fn = nullptr;
                    if (cudaFuncGetName(&fn, kp.func) == cudaSuccess && fn) name = fn;
#endif
                }
            } else if (ty == cudaGraphNodeTypeMemcpy) name = "memcpy";
            else if (ty == cudaGraphNodeTypeMemset) name = "memset";
            ++kinds[name];
        }
        std::vector<std::pair<int, std::string>> v;
        for (auto& [k2, c] : kinds) v.push_back({c, k2});
        std::sort(v.rbegin(), v.rend());
        std::fprintf(stderr, "strata verify: the %d-token window graph has %zu nodes:", T, nn);
        for (size_t i = 0; i < v.size() && i < 40; ++i) std::fprintf(stderr, " %d x %.60s;", v[i].first, v[i].second.c_str());
        std::fprintf(stderr, "\n");
#if CUDART_VERSION >= 12030   // the programmatic (PDL) edges: CUDA allows them only from a kernel node to a kernel node
        size_t ne = 0;
#if CUDART_VERSION >= 13000
        cudaGraphGetEdges(graph, nullptr, nullptr, nullptr, &ne);
#else
        cudaGraphGetEdges_v2(graph, nullptr, nullptr, nullptr, &ne);
#endif
        std::vector<cudaGraphNode_t> from(ne), to(ne);
        std::vector<cudaGraphEdgeData> ed(ne);
#if CUDART_VERSION >= 13000
        const cudaError_t ge = cudaGraphGetEdges(graph, from.data(), to.data(), ed.data(), &ne);
#else
        const cudaError_t ge = cudaGraphGetEdges_v2(graph, from.data(), to.data(), ed.data(), &ne);
#endif
        size_t prog = 0, bad = 0;
        for (size_t i = 0; ge == cudaSuccess && i < ne; ++i) {
            if (ed[i].type != cudaGraphDependencyTypeProgrammatic) continue;
            ++prog;
            cudaGraphNodeType ta, tb2;
            if (cudaGraphNodeGetType(from[i], &ta) != cudaSuccess || cudaGraphNodeGetType(to[i], &tb2) != cudaSuccess ||
                ta != cudaGraphNodeTypeKernel || tb2 != cudaGraphNodeTypeKernel)
                ++bad;
        }
        if (ge != cudaSuccess) cudaGetLastError();
        std::fprintf(stderr, "strata verify: %zu edges, %zu programmatic (PDL), %zu of them not kernel to kernel%s\n", ne,
                     prog, bad, ge != cudaSuccess ? " (the edge query failed)" : "");
#endif
    }
#endif
    const bool made = instantiate_evicting(exec_t, graph, {}, "verify: instantiate: ", err);
=======
    const cudaError_t ie = cudaGraphInstantiate(&exec_[T], graph, 0);
>>>>>>> their fork (github.com-eddoursul-Strata-custom)
    cudaGraphDestroy(graph);
    if (ie != cudaSuccess) {
        err = std::string("verify: instantiate: ") + cudaGetErrorString(ie);
        return false;
    }
    const cudaError_t ue = cudaGraphUpload(exec_[T], cs_);
    const cudaError_t us = cudaStreamSynchronize(cs_);
    std::fprintf(stderr, "strata verify: captured the %d-token window (upload %s, sync %s)\n", T,
                 cudaGetErrorString(ue), cudaGetErrorString(us));
    return true;
}

bool Verifier::capture_commit(std::string& err) {
    if (commit_exec_ != nullptr) return true;
    using namespace strata::kernels;
    const ModelGeometry& g = *g_;
    SessionState& ss = *ss_;
    const QsaShapes s = shapes_of(g);
    const int64_t C = g.ssm_conv_channels, HV = g.ssm_v_heads, ID = g.idx_key_dim, MT = max_t_;
    const uint64_t gdn_floats = (uint64_t) g.ssm_state_size * g.ssm_v_heads * g.ssm_state_size +
                                (uint64_t) g.ssm_conv_channels * (g.ssm_d_conv - 1);
    const int64_t TS = (s.idx_block - 1) * ID;
    const int64_t HS = (int64_t) NG_HIST * NG_HC_DIM;
    if (cudaStreamBeginCapture(cs_, cudaStreamCaptureModeThreadLocal) != cudaSuccess) {
        err = "verify: begin commit capture failed";
        return false;
    }
    bool ok = true;
    try {
        copy_i32_from_mapped(commit_, m_commit_, 2 + MT, cs_);
        int64_t qsa_index = 0, gdn_index = 0;
<<<<<<< this fork
        for (int64_t l = 0; l < lb_; ++l) (is_qsa_layer(g, l) ? qsa_index : gdn_index) += 1;
        // engine-latency (STRATA_COMMIT_BATCH, default on; =0 keeps the per-layer loop): every GDN layer's conv history and
        // state in two launches instead of two per layer (~70 graph nodes fewer; the same arithmetic per layer, bitwise).
        // The commit runs between a window and the draft round, which on HIP waits for it.
#if defined(STRATA_GPU_METAL)   // no batched commit kernels on Metal: the per-layer loop (the Mac engine's gated path)
        static const bool commit_batch = false;
#else
        static const bool commit_batch = [] { const char* e = std::getenv("STRATA_COMMIT_BATCH"); return !e || e[0] != '0'; }();
#endif
        bool gdn_batched = false;
        if (commit_batch) {
            int64_t n_gdn = 0;
            bool norms_ok = true;   // the per-layer loop checks each layer's ssm_norm.weight: keep that check
            for (int64_t l = lb_; l < le_ && norms_ok; ++l)
                if (!is_qsa_layer(g, l)) {
                    ++n_gdn;
                    norms_ok = need(LayerView(*wt_, l), "ssm_norm.weight", err) != nullptr;
                }
            if (!norms_ok) ok = false;
            else if (n_gdn > 0)
                gdn_batched = gdn_commit_layers(
                    ss.gdn_state + (size_t) (gdn_index - ss.gdn_ord0) * gdn_floats, (int64_t) gdn_floats,
                    (int64_t) g.ssm_state_size * g.ssm_v_heads * g.ssm_state_size, qkv_L_ + (size_t) gdn_index * MT * C,
                    MT * C, h_L_ + (size_t) gdn_index * MT * C, MT * C, gate_L_ + (size_t) gdn_index * MT * HV,
                    beta_L_ + (size_t) gdn_index * MT * HV, MT * HV, (int) C, (int) g.ssm_k_heads, (int) HV, (int) MT,
                    (int) n_gdn, commit_, cs_);
        }
        // the QSA layers' tail restore + indexer append, one launch for all of them (one block per layer)
        bool qsa_batched = false;
        if (commit_batch && ok && !g_no_batch_kv) {
            QsaIndexerLayers il;
            int nq = 0;
            int64_t mc = -1;
            bool fits = true;
            for (int64_t l = lb_, qi = qsa_index; l < le_ && fits; ++l) {
                if (!is_qsa_layer(g, l)) continue;
                const WeightRef* wikn = need(LayerView(*wt_, l), "indexer.k_norm.weight", err);
                if (!wikn) { ok = false; fits = false; break; }
                const QsaState& st = ss.qsa_states[qi];
                if (nq >= QsaIndexerLayers::kMax || (mc >= 0 && st.max_cells != mc)) { fits = false; break; }
                mc = st.max_cells;
                il.gamma[nq] = (const float*) wikn->data;
                il.tail_src[nq] = tail_snap_ + (size_t) qi * TS;
                il.tail[nq] = st.idx_tail; il.dead[nq] = st.idx_dead; il.pooled[nq] = st.idx_pooled;
                il.block_pos[nq] = st.idx_block_pos;
                ++nq;
                ++qi;
            }
            if (ok && fits && nq > 0) {
                native_qsa_indexer_append_steps_layers(idx_raw_L_ + (size_t) qsa_index * MT * ID, MT * ID, commit_ + 2, 1,
                                                       (int) MT, 0, EPS, il, nq, s, mc, rope_scaling(), cs_);
                qsa_batched = true;
            }
        }
        for (int64_t l = lb_; l < le_ && ok; ++l) {
=======
        for (int64_t l = 0; l < g.n_layers && ok; ++l) {
>>>>>>> their fork (github.com-eddoursul-Strata-custom)
            const LayerView v(*wt_, l);
            if (is_qsa_layer(g, l) && qsa_batched) {
                ++qsa_index;
            } else if (!is_qsa_layer(g, l) && gdn_batched) {
                ++gdn_index;
            } else if (!is_qsa_layer(g, l)) {
                const WeightRef* wnm = need(v, "ssm_norm.weight", err);
                if (!wnm) { ok = false; break; }
                float* state = ss.gdn_state + (size_t) gdn_index * gdn_floats;
                float* conv = state + (uint64_t) g.ssm_state_size * g.ssm_v_heads * g.ssm_state_size;
                const float* qkv = qkv_L_ + (size_t) gdn_index * MT * C;
                gdn_conv_commit(conv, qkv, (int) C, commit_, cs_);
                gdn_step_norm_multi(state, h_L_ + (size_t) gdn_index * MT * C, (int) C, gate_L_ + (size_t) gdn_index * MT * HV,
                                    beta_L_ + (size_t) gdn_index * MT * HV, z_, (const float*) wnm->data, EPS, y_dummy_,
                                    (int) g.ssm_k_heads, (int) HV, (int) MT, commit_, cs_);
                ++gdn_index;
            } else {
                const QsaState& st = ss.qsa_states[qsa_index];
                const WeightRef* wikn = need(v, "indexer.k_norm.weight", err);
                if (!wikn) { ok = false; break; }
                copy_from_mapped(st.idx_tail, tail_snap_ + (size_t) qsa_index * TS, TS, cs_);
                const QsaIndexerBuffers ib{st.idx_tail, st.idx_dead, st.idx_pooled, st.idx_block_pos};
                native_qsa_indexer_append_multi(idx_raw_L_ + (size_t) (qsa_index * MT) * ID, commit_ + 2, 1, (int) MT, 0,
                                                (const float*) wikn->data, EPS, ib, s, st.max_cells,
                                                (float) qsa_freq_base(), cs_);
                ++qsa_index;
            }
        }
        if (ok && ss.ple.ready()) copy_indexed(ss.ple.hist, hist_snap_, HS, commit_ + 1, HS, cs_);
    } catch (const std::exception& e) {
        err = std::string("verify commit: ") + e.what();
        ok = false;
    }
    cudaGraph_t graph = nullptr;
    const cudaError_t ce = cudaStreamEndCapture(cs_, &graph);
    if (!ok) {
        if (graph) cudaGraphDestroy(graph);
        return false;
    }
    if (ce != cudaSuccess || cudaGraphInstantiate(&commit_exec_, graph, 0) != cudaSuccess) {
        if (graph) cudaGraphDestroy(graph);
        err = std::string("verify: commit capture: ") + cudaGetErrorString(ce);
        return false;
    }
    cudaGraphDestroy(graph);
    return true;
}

bool Verifier::run(int T, const int32_t* tokens, int64_t pos0, PoolMultiFn pool, void* user, int32_t* out,
                   std::string& err) {
#if defined(STRATA_GPU_METAL)
    (void) pool;
    (void) user;
    return strata::metal_engine::window_run(metal_, T, tokens, pos0, head_sampling_, out, err);
#endif
    using namespace strata::kernels;
    if (T < 1 || T > max_t_) { err = "verify: window size out of range"; return false; }
    const ModelGeometry& g = *g_;
    SessionState& ss = *ss_;
    if (pos0 + T > ss.qsa_states[0].max_cells) { err = "verify: the window runs past the context"; return false; }
    if (!capture(T, err) || !capture_commit(err)) return false;
    VDBG("captured; staging\n");
    const Clock::time_point t0 = Clock::now();
    const QsaShapes s = shapes_of(g);
    for (int t = 0; t < T; ++t) {
        h_tok_[t] = tokens[t];
        qsa_step_fill(h_step_ + t * kStepCount, pos0 + t, s);
        for (int64_t h = 0; h < g.n_head; ++h) h_pos_[t * g.n_head + h] = (int32_t) (pos0 + t);
    }
    const bool ple_on = ss.ple.ready();
    if (ple_on) {   // the reads start here unless ple_ahead started them
        int32_t prev[2] = {ss.ple_prev[0], ss.ple_prev[1]};
        for (int t = 0; t < T; ++t) {
            uint32_t rows[PLE_N_HEADS];
            ngram_rows(&tokens[t], prev, 1, ss.ple.consts, rows);
            if (!ss.ple.table->ahead_holds(t, rows)) ss.ple.table->ahead(t, rows);
            prev[0] = prev[1];
            prev[1] = tokens[t];
        }
    }
    // the residency this window decides the main GPU's hits by, and the pool reads (`residency`)
    std::memcpy(h_res_, hits_.res, (size_t) (g.n_layers * g.n_expert) * sizeof(int32_t));
    *(volatile uint32_t*) h_seq_ = 0;
    *(volatile uint32_t*) h_flag_ = 0;
    *(volatile uint32_t*) h_flagA_ = 0;
    *(volatile uint32_t*) h_flagB_ = 0;
    *(volatile uint32_t*) h_pseq_ = 0;
    *(volatile uint32_t*) h_pleflag_ = 0;
    h_flag2_[0] = h_flag2_[16] = 0;
    sink_.gpu2_ring = 0;
    std::atomic_thread_fence(std::memory_order_seq_cst);
    last_t_ = T;
    last_pos0_ = pos0;
    for (int t = 0; t < T; ++t) last_tokens_[t] = tokens[t];
    ms_host += ms_since(t0);
    VDBG("staged; launching\n");
<<<<<<< this fork
    if (ar_on() && h_plan_err_ != nullptr) *(volatile uint32_t*) h_plan_err_ = 0;
    // 0.1.43: with the all-resident graph the host serves no layer, so nothing can hold a layer back: every copy first
    if (adapt_gate_ != nullptr && adapt_gate_->active.load(std::memory_order_acquire) && ar_on() &&
        !adapt_gate_wait((int32_t) g.n_layers, err)) return false;
#if !defined(STRATA_USE_HIP) && !defined(STRATA_HIP_GFX906)
    // A blocking instrumented launch cannot also serve the graph's CPU doorbells.
    static const bool threaded_launch = [] {
        const char* value = std::getenv("STRATA_VERIFY_ASYNC_LAUNCH");
        return value != nullptr && value[0] == '1';
    }();
    HostGraphLaunch launch(ar_off_ ? exec_nr_[T] : exec_[T], cs_, device_, threaded_launch,
                           [this] { (void) release_gpu_waits(5000); });
    const cudaError_t le = launch.status();
#else
    const cudaError_t le = cudaGraphLaunch(ar_off_ ? exec_nr_[T] : exec_[T], cs_);
#endif
    trace_ev("LAUNCHED", -1, -1, (int64_t) le);
    if (le != cudaSuccess) { err = std::string("verify: launch: ") + cudaGetErrorString(le); return false; }
#if !defined(STRATA_USE_HIP) && !defined(STRATA_HIP_GFX906)
    (void) launch.query();
#else
=======
    const cudaError_t le = cudaGraphLaunch(exec_[T], cs_);
    if (le != cudaSuccess || cudaEventRecord(done_, cs_) != cudaSuccess) {
        err = std::string("verify: launch: ") + cudaGetErrorString(cudaGetLastError());
        return false;
    }
>>>>>>> their fork (github.com-eddoursul-Strata-custom)
    (void) cudaStreamQuery(cs_);
#endif
    VDBG("launched\n");
    // on a failure every wait of the graph passes, so it runs to its end before the next window resets the flags
    auto release = [&] {
        h_list2_[0] = h_list2_[list2_stride_] = 0;
        std::atomic_thread_fence(std::memory_order_seq_cst);
        for (uint32_t* f : {h_flag_, h_flagA_, h_flagB_, h_pleflag_, h_flag2_, h_flag2_ + 16})
            *(volatile uint32_t*) f = 0xffffffffu;
        cudaStreamSynchronize(cs_);
        return false;
    };
    if (ple_on) {   // before layer 1: the graph waits for the flag, raised even when a read failed
        const Clock::time_point tp = Clock::now();
        bool ok = true;
        for (int t = 0; t < T && ok; ++t) ok = ss.ple.table->ahead_collect(t, h_ple_ + (size_t) t * g.n_embd, err);
        std::atomic_thread_fence(std::memory_order_seq_cst);
        _mm_sfence();
        *(volatile uint32_t*) h_pleflag_ = 1;
        ms_host += ms_since(tp);
        if (!ok) return release();
    }
    volatile uint32_t* const seq = h_seq_;
    volatile uint32_t* const flag = h_flag_;
    const int G = groups_[T] > 0 ? groups_[T] : 1;
    const int gtb[2] = {0, (T + 1) / 2}, gte[2] = {G == 2 ? (T + 1) / 2 : T, T};
    // a pool whose second-GPU rows came after the CPU's: the host time until they were in
    const volatile uint32_t* late_flag = nullptr;
    uint32_t late_want = 0;
    Clock::time_point late_at{};
    auto settle = [&](bool now) {
        if (late_flag != nullptr && (now || *late_flag >= late_want)) {
            ms_late2 += ms_since(late_at);
            late_flag = nullptr;
        }
    };
    for (int64_t k = 0; k < g.n_layers * G; ++k) {
        const int64_t l = k / G;
        const int grp = (int) (k % G);
        const uint32_t want = (uint32_t) (k + 1);
        const Clock::time_point a = Clock::now();
        auto last_flush = a;
        uint32_t spins = 0;
        while (*seq < want) {
            settle(false);
            _mm_pause();
            if ((++spins & 1023u) != 0) continue;
            const auto now = Clock::now();
            if (now - last_flush > std::chrono::microseconds(2000)) {
                last_flush = now;
#if !defined(STRATA_USE_HIP) && !defined(STRATA_HIP_GFX906)
                const cudaError_t q = launch.query();
#else
                const cudaError_t q = cudaStreamQuery(cs_);
#endif
                if (q != cudaErrorNotReady && *seq < want) {
                    err = "verify: layer " + std::to_string(l) + " never rang (" +
                          (q == cudaSuccess ? std::string("graph finished") : std::string(cudaGetErrorString(q))) + ")";
                    return release();
                }
                if (watch_ != nullptr && !watch_(watch_user_, err)) return release();
            }
            if (now - a > std::chrono::seconds(20)) {
<<<<<<< this fork
                // #267: the caller ends the engine; no spin kernel may outlive it
#if !defined(STRATA_USE_HIP) && !defined(STRATA_HIP_GFX906)
                trace_ev("TIMEOUT", k, l, (int64_t) launch.query());
#else
                trace_ev("TIMEOUT", k, l, (int64_t) cudaStreamQuery(cs_));
#endif
                if (g_trace) {
                    diag(stderr);   // the words and the breadcrumbs before the release ...
                    const bool drained = release_gpu_waits(5000);
                    trace_dump(stderr);   // ... and after it: did the GPU move once its waits were raised?
                    err = "verify: timed out at layer " + std::to_string(l) + released_note(drained);
                    return false;
                }
                err = "verify: timed out at layer " + std::to_string(l) + released_note(release_gpu_waits(5000));
                return false;
=======
                err = "verify: timed out at layer " + std::to_string(l);
                return release();
>>>>>>> their fork (github.com-eddoursul-Strata-custom)
            }
        }
        const Clock::time_point b = Clock::now();
        VDBG("layer %lld rang\n", (long long) l);
        cur_layer_ = want - 1;
        set_plan_slot(grp);
        const int tb = gtb[grp], n = gte[grp] - gtb[grp];
        sink_.xk = xk_layer_[(size_t) l] ? h_xk_ + (size_t) tb * (size_t) xk_stride_ : nullptr;
        sink_.xk_stride = xk_stride_;
        sink_.x1 = h_xq1_ + (size_t) tb * (size_t) (g.n_embd / 32) * 36;
        if (pool != nullptr)
            pool(user, h_x_ + (size_t) tb * g.n_embd, h_ids_ + (size_t) tb * ss.k, h_w_ + (size_t) tb * ss.k, n,
                 ss.k, h_ymiss_ + (size_t) tb * ss.k * g.n_embd, l);
        VDBG("layer %lld served\n", (long long) l);
        std::atomic_thread_fence(std::memory_order_seq_cst);
        _mm_sfence();
        if (pcie_share_ && *(volatile uint32_t*) h_flagA_ != want) {   // no PCIe share published: an empty one
            sink_.counts[0] = 0;
            sink_.counts[1] = 0;
            sink_.counts[2] = 0;
            sink_.start2[0] = 0;
            std::atomic_thread_fence(std::memory_order_seq_cst);
            *(volatile uint32_t*) h_flagA_ = want;
            raise_flag(h_flagB_, want);
        }
        if (gpu2_ && sink_.gpu2_ring != want) {   // no second-GPU share went out (the pool failed): an empty one
            sink_.gpu2_list[0] = 0;
            std::atomic_thread_fence(std::memory_order_seq_cst);
            *(volatile uint32_t*) sink_.gpu2_flag = want;
        }
        *flag = want;
        ms_wait += std::chrono::duration<double, std::milli>(b - a).count();
        ms_pool += ms_since(b);
        if (gpu2_ && *(volatile uint32_t*) sink_.gpu2_flag < want) {
            settle(true);   // a split window's other token group
            late_flag = sink_.gpu2_flag;
            late_want = want;
            late_at = Clock::now();
            ++late2;
        }
        if (tail_ != nullptr && k + 1 == g.n_layers * G) tail_(tail_user_);
        // the next layer's prediction, published during this layer's pool, else microseconds later (its branch is
        // joined before the next layer): waited for, so the experts the second GPU takes never depend on timing
        if (predict_ != nullptr && G == 1 && l + 1 < g.n_layers && native_router_enabled()) {
            const Clock::time_point c = Clock::now();
            if (*(volatile uint32_t*) h_pseq_ < (uint32_t) (l + 1)) {
                ++predict_late;
                while (*(volatile uint32_t*) h_pseq_ < (uint32_t) (l + 1) &&
                       Clock::now() - c < std::chrono::milliseconds(50)) {
                    settle(false);
                    _mm_pause();
                }
            }
            if (*(volatile uint32_t*) h_pseq_ >= (uint32_t) (l + 1)) {
                std::atomic_thread_fence(std::memory_order_acquire);
                const size_t half = (size_t) ((l + 1) & 1) * (size_t) max_t_ * (size_t) ss.k;
                predict_(predict_user_, l + 1, h_pids_ + half, h_pw_ + half, T, ss.k);
            }
            ms_predict += ms_since(c);
        }
        // a guarded tier's pump never runs while the second GPU is behind: its enqueue can block on a GPU that is
        // spinning on the second-GPU flag, and this thread is what launches that GPU's next graph (deadlock)
        if (rows_in_ != nullptr && k + 1 < g.n_layers * G &&
            (!rows_after_gpu2_ || !gpu2_ || *(volatile uint32_t*) sink_.gpu2_flag >= want))
            rows_in_(rows_user_, l);
    }
    // the last layer's second-GPU rows: no later doorbell bounds the graph's wait for them, and a spin kernel left
    // running when the engine ends can leave the GPU lost until a power cycle (upstream #267)
    for (const Clock::time_point w0 = Clock::now();
         gpu2_ && *(volatile uint32_t*) sink_.gpu2_flag < (uint32_t) (g.n_layers * G);) {
        settle(false);
        _mm_pause();
        if (ms_since(w0) > 5000.0) {
            err = "verify: the second GPU's rows of the last layer never came";
            return release();
        }
    }
<<<<<<< this fork
    // (#646 staged the next stage's inputs here; 0.1.39b keeps the layer split's order: each stage stages its own)
    progress_at("verify window: waiting for the GPU to finish the window (flags A/B/M raised)", (int64_t) T);
    // #267: a window the GPU never finishes (a spin kernel that never sees its flag) holds the host here; the stall
    // watchdog then releases every verifier's GPU waits (release_live_verifiers) before it ends the engine, so no
    // spin kernel outlives the process - the case that left Windows GPUs "lost" until a power cycle.  The wait
    // itself stays a blocking sync: a cudaStreamQuery poll here cost IQ3_S ~3% decode (a core calling the driver
    // beside the expert workers).
#if !defined(STRATA_USE_HIP) && !defined(STRATA_HIP_GFX906)
    const cudaError_t launch_result = launch.finish();
    if (launch_result != cudaSuccess) {
        err = std::string("verify: launch: ") + cudaGetErrorString(launch_result);
        return false;
    }
#endif
    trace_ev("SYNC", -1, -1, 0);
    const cudaError_t se = cudaStreamSynchronize(cs_);
    trace_ev("SYNCED", -1, -1, (int64_t) se);
    if (se != cudaSuccess) { err = std::string("verify: ") + cudaGetErrorString(se); return false; }
    if (ar_on() && h_plan_err_ != nullptr && *(volatile uint32_t*) h_plan_err_ != 0) {   // #871: refresh_ar saw the table whole
        *(volatile uint32_t*) h_plan_err_ = 0;
        err = "verify: the all-resident plan met an expert that is not in VRAM (the residency table changed during the window)";
        return false;
=======
    while (late_flag != nullptr && ms_since(late_at) < 50.0) {   // the last layer's (the graph ends after them)
        settle(false);
        _mm_pause();
>>>>>>> their fork (github.com-eddoursul-Strata-custom)
    }
    settle(true);
    const bool sampled = head_sampling_ && ((!sampling_.greedy && sampling_.temperature > 0.0f) || hist_d_ != nullptr);
    const bool whole = shead_ != nullptr && (logits_wanted_ || sampled);
    if (shead_ != nullptr) {   // the head's input is handed over after the last layer: the second GPU's part then
        ++hand_n_;
        const Clock::time_point h0 = Clock::now();
        while (*(volatile uint32_t*) shead_->handoff() < hand_n_) {
            _mm_pause();
            if (ms_since(h0) > 5000.0) {
                err = std::string("verify: the head's input never came (") + cudaGetErrorString(cudaStreamQuery(cs_)) +
                      ")";
                return release();
            }
        }
        if (!shead_->submit(T, whole, err)) return false;
        ++done_n_;
    }
    const cudaError_t se = cudaStreamSynchronize(cs_);
    if (se != cudaSuccess) { err = std::string("verify: ") + cudaGetErrorString(se); return false; }
    cudaStreamSynchronize(copy_);   // no host function of this window may raise flag B in the next one
    if (shead_ != nullptr) {   // each token's larger logit; on equality this GPU's, the lower index
        if (!shead_->wait(done_n_, err)) return false;
        for (int t = 0; t < T; ++t) {
            const float v1 = ((volatile float*) h_val_)[t], v2 = ((volatile const float*) shead_->val())[t];
            if (v2 > v1) ((volatile int32_t*) h_out_)[t] = (int32_t) (shead_->split() + shead_->idx()[t]);
        }
        if (whole) join_rows(head_full_, n_vocab_, head_logits_, head_rows_, shead_->logits(), T, cs_);
    }
    // a sampled or penalized request: the head's sampling again, after the graph (see set_sampling)
    if (sampled) {
        SamplerParams sp = sampling_;
        sp.counter = (uint64_t) pos0;
        sample_tokens(shead_ != nullptr ? head_full_ : head_logits_, T, (int) n_vocab_, hist_d_, hist_len_, sp, m_out_,
                      cs_);
        if (cudaStreamSynchronize(cs_) != cudaSuccess) { err = "verify: the head sampling failed"; return false; }
    }
    for (int t = 0; t < T; ++t) out[t] = ((volatile int32_t*) h_out_)[t];
    if (stamps_ != nullptr && G == 1) {
        const int64_t NL = g.n_layers;
        h_stamps_.resize((size_t) (NL + 1) * kStamps);
        cudaMemcpy(h_stamps_.data(), stamps_ + (size_t) T * (size_t) (NL + 1) * kStamps,
                   h_stamps_.size() * sizeof(unsigned long long), cudaMemcpyDeviceToHost);
        if (prof_ns_.empty()) prof_ns_.assign(2 * kStamps, 0.0);
        const unsigned long long* w = h_stamps_.data() + (size_t) NL * kStamps;
        for (int64_t l = 0; l < NL; ++l) {
            const unsigned long long* st = h_stamps_.data() + (size_t) l * kStamps;
            const int kind = is_qsa_layer(g, l) ? 1 : 0;
            const unsigned long long before = l == 0 ? w[1] : (st - kStamps)[kLayerStamps - 1];
            prof_ns_[(size_t) kind * kStamps] += (double) (st[0] - before);
            for (int i = 1; i < kLayerStamps; ++i) prof_ns_[(size_t) kind * kStamps + i] += (double) (st[i] - st[i - 1]);
            // the shared expert's branch and the second GPU's rows', from the doorbell (they fork there) to their ends
            prof_ns_[(size_t) kind * kStamps + kLayerStamps] += (double) (st[kLayerStamps] - st[4]);
            if (gpu2_) prof_ns_[(size_t) kind * kStamps + kLayerStamps + 1] += (double) (st[kLayerStamps + 1] - st[4]);
            {   // the wait for the CPU's rows, by bucket
                const double us = (double) (st[9] - st[8]) / 1e3;
                int b = 0;
                while (b < kWaitBuckets - 1 && us >= kWaitEdges[b]) ++b;
                ++prof_wait_hist_[kind][b];
            }
            ++prof_layers_[kind];
        }
        prof_window_ns_[0] += (double) (w[1] - w[0]);
        prof_window_ns_[1] += (double) (w[3] - w[2]);
        prof_window_ns_[2] += (double) (w[3] - w[0]);
        prof_window_ns_[3] += (double) (w[4] - w[3]);
        ++prof_windows_;
    }
    VDBG("window done\n");
    ++windows;
    return true;
}

bool Verifier::copy_logits(int T, float* out, std::string& err) const {
    if (shead_ != nullptr && !logits_wanted_) { err = "verify: the logits need set_logits_wanted"; return false; }
    const float* src = shead_ != nullptr ? head_full_ : head_logits_;
    if (cudaMemcpyAsync(out, src, (size_t) T * (size_t) n_vocab_ * sizeof(float), cudaMemcpyDeviceToHost,
                        cs_) != cudaSuccess ||
        cudaStreamSynchronize(cs_) != cudaSuccess) {
        err = std::string("verify: reading the logits back: ") + cudaGetErrorString(cudaGetLastError());
        return false;
    }
    return true;
}

bool Verifier::candidate_logprobs(const int32_t* cand, int n, float* out, std::string& err) const {
    std::vector<float> row((size_t) n_vocab_);
    if (!copy_logits(1, row.data(), err)) return false;
    float mx = -INFINITY;
    for (float v : row)
        if (std::isfinite(v) && v > mx) mx = v;
    double sum = 0.0;
    for (float v : row)
        if (std::isfinite(v)) sum += std::exp((double) v - mx);
    const double lse = (double) mx + std::log(sum);
    for (int i = 0; i < n; ++i) {
        const int32_t id = cand[i];
        const bool ok = id >= 0 && id < n_vocab_ && std::isfinite(row[(size_t) id]);
        out[i] = ok ? (float) ((double) row[(size_t) id] - lse) : -INFINITY;
    }
    return true;
}

void Verifier::print_profile() const {
    if (prof_windows_ == 0) return;
    static const char* const names[kStamps] = {"stamp gap",      "HC read, mixer",   "mixer",          "HC read, FFN",
                                               "router, ring, hits", "expert inputs", "VRAM experts",
                                               "wait for PCIe plan", "PCIe experts", "wait for CPU",
                                               "wait for 2nd GPU", "rows + combine", "last HC write",
                                               "shared expert, beside", "2nd GPU rows, beside"};
    const double nw = (double) prof_windows_;
    std::printf("%-24s GPU time over %lld windows, ms per window: inputs %.3f, head %.3f, whole window %.3f; "
                "one stamp %.2f us (each stage below includes one; the last two run from the ring beside the "
                "six before \"rows + combine\")\n", "window profile", (long long) prof_windows_,
                prof_window_ns_[0] / nw / 1e6, prof_window_ns_[1] / nw / 1e6, prof_window_ns_[2] / nw / 1e6,
                prof_window_ns_[3] / nw / 1e3);
    std::printf("  %-22s %12s %12s %14s\n", "stage", "GDN us/layer", "QSA us/layer", "ms per window");
    for (int i = 0; i < kStamps; ++i) {
        const double a = prof_ns_[(size_t) i], b = prof_ns_[(size_t) kStamps + i];
        std::printf("  %-22s %12.1f %12.1f %14.3f\n", names[i],
                    prof_layers_[0] ? a / (double) prof_layers_[0] / 1e3 : 0.0,
                    prof_layers_[1] ? b / (double) prof_layers_[1] / 1e3 : 0.0, (a + b) / nw / 1e6);
    }
    std::printf("  %-22s", "wait for CPU, layers");
    for (int b = 0; b < kWaitBuckets; ++b) {
        const double f = 100.0 * (double) (prof_wait_hist_[0][b] + prof_wait_hist_[1][b]) /
                         (double) std::max<int64_t>(1, prof_layers_[0] + prof_layers_[1]);
        if (b < kWaitBuckets - 1) std::printf(" <%.0f us %.0f%%,", kWaitEdges[b], f);
        else std::printf(" more %.0f%%\n", f);
    }
}

void Verifier::ple_ahead(int t, int32_t token) {
    using namespace strata::kernels;
    SessionState& ss = *ss_;
    if (!ss.ple.ready() || t < 0 || t >= max_t_) return;
    ahead_tok_[t] = token;
    int32_t prev[2] = {t >= 2 ? ahead_tok_[t - 2] : ss.ple_prev[t], t >= 1 ? ahead_tok_[t - 1] : ss.ple_prev[1]};
    uint32_t rows[PLE_N_HEADS];
    ngram_rows(&token, prev, 1, ss.ple.consts, rows);
    ss.ple.table->ahead(t, rows);
}

void Verifier::set_plan_slot(int grp) {
    const int64_t cap = sink_.cap;
    int32_t* base = h_plan_ + (size_t) grp * (size_t) plan_i32_;
    sink_.counts = base;
    sink_.dst = base + 4 + cap + 1;
    sink_.tok = sink_.dst + cap;
    sink_.ptr2 = (unsigned long long*) (base + plan_ptr_off_) + cap;
    sink_.start2 = base + plan_ptr_off_ + 4 * cap;
    const int G = groups_[last_t_] > 0 ? groups_[last_t_] : 1;
    const int64_t per = G == 2 ? kStagingBlobs / 2 : kStagingBlobs;
    sink_.staging = (unsigned long long) (staging_ + (size_t) (grp * per) * strata::kernels::cpu::expert_layout().max_blob);
    sink_.staging_cap = per;
    if (gpu2_) {
        sink_.gpu2_list = h_list2_ + (size_t) grp * (size_t) list2_stride_;
        sink_.gpu2_flag = h_flag2_ + (size_t) grp * 16;
        sink_.ring = cur_layer_ + 1;
    }
}

// Flag B only rises: a host function of an earlier layer may run after a later layer already raised it directly.
void Verifier::raise_flag(uint32_t* flag, uint32_t value) {
    volatile long* f = (volatile long*) flag;
#if defined(_WIN32)
    long cur = *f;
    while ((uint32_t) cur < value) {
        const long prev = _InterlockedCompareExchange(f, (long) value, cur);
        if (prev == cur) break;
        cur = prev;
    }
#else
    uint32_t cur = __atomic_load_n((uint32_t*) flag, __ATOMIC_SEQ_CST);
    while (cur < value && !__atomic_compare_exchange_n((uint32_t*) flag, &cur, value, false, __ATOMIC_SEQ_CST, __ATOMIC_SEQ_CST)) {}
#endif
}

// Plan v0.3 P6: the PCIe share by DMA.  The copy engine moves the blobs while the CPU computes its own share and the
// GPU its VRAM experts; a host function raises flag B when they have landed (the graph waits for it before the PCIe
// groups).  Staging is split between the two token groups of a split window.
void Verifier::fetch_dma(void* ctx, const uint8_t* const* src, int n, size_t bytes) {
    Verifier* v = (Verifier*) ctx;
    const uint32_t want = v->cur_layer_ + 1;
    if (n <= 0) { raise_flag(v->h_flagB_, want); return; }
    uint8_t* stage = (uint8_t*) v->sink_.staging;                  // this group's half in a split window
    for (int i = 0; i < n; ++i) cudaMemcpyAsync(stage + (size_t) i * bytes, src[i], bytes, cudaMemcpyHostToDevice, v->copy_);
    FlagSet& fs = v->flag_sets_[v->cur_layer_ % (sizeof v->flag_sets_ / sizeof v->flag_sets_[0])];
    fs.flag = v->h_flagB_;
    fs.value = want;
    cudaLaunchHostFunc(v->copy_, [](void* p) { FlagSet* s = (FlagSet*) p; raise_flag(s->flag, s->value); }, &fs);
}

void Verifier::publish_plan(void* ctx) {
    Verifier* v = (Verifier*) ctx;
    _mm_sfence();
    *(volatile uint32_t*) v->h_flagA_ = v->cur_layer_ + 1;
}

<<<<<<< this fork
bool Verifier::window_logprobs(const int32_t* targets, int T, int64_t pos0, int32_t extra_id, std::FILE* out,
                               std::string& err) {
    if (next_ != nullptr) return next_->window_logprobs(targets, T, pos0, extra_id, out, err);
    const OnDevice on_device(device_);
#if defined(STRATA_GPU_METAL)
    // the head logits are the Metal window's (window_copy_logits, row by row)
    if (n_vocab_ <= 0 || T <= 0 || targets == nullptr || out == nullptr) {
        err = "window_logprobs: no head logits for this window";
        return false;
    }
    std::vector<float> h((size_t) T * (size_t) n_vocab_);
    for (int t = 0; t < T; ++t)
        if (!strata::metal_engine::window_copy_logits(metal_, t, h.data() + (size_t) t * (size_t) n_vocab_)) {
            err = "window_logprobs: the head logits copy failed";
            return false;
        }
#else
    if (head_logits_ == nullptr || n_vocab_ <= 0 || T <= 0 || targets == nullptr || out == nullptr) {
        err = "window_logprobs: no head logits for this window";
        return false;
    }
    // run() synchronized cs_ before returning, so the head of this window is complete
    std::vector<float> h((size_t) T * (size_t) n_vocab_);
    if (cudaMemcpy(h.data(), head_logits_, h.size() * sizeof(float), cudaMemcpyDeviceToHost) != cudaSuccess) {
        err = "window_logprobs: the head logits copy failed";
        return false;
    }
#endif
    for (int t = 0; t < T; ++t) {
        const float* row = h.data() + (size_t) t * (size_t) n_vocab_;
        const int32_t tgt = targets[t];
        if (tgt < 0 || (int64_t) tgt >= n_vocab_) continue;
        int64_t top = 0;
        for (int64_t v = 1; v < n_vocab_; ++v)
            if (row[v] > row[top]) top = v;
        const double maxv = row[top];
        const bool has_extra = extra_id >= 0 && (int64_t) extra_id < n_vocab_;
        double sum = 0.0, sum_without = 0.0;   // the second skips extra_id: no cancellation when it holds ~all mass
        for (int64_t v = 0; v < n_vocab_; ++v) {
            const double e = std::exp((double) row[v] - maxv);
            sum += e;
            if (v != (int64_t) extra_id) sum_without += e;
        }
        const double lse = maxv + std::log(sum);
        const double extra = has_extra ? (double) row[extra_id] - lse : NAN;
        const double without = has_extra && tgt != extra_id && sum_without > 0.0
                                   ? (double) row[tgt] - (maxv + std::log(sum_without)) : NAN;
        std::fprintf(out, "%lld\t%d\t%.9f\t%lld\t%.9f\t%d\t%.9f\t%.9f", (long long) (pos0 + t), (int) tgt,
                     (double) row[tgt] - lse, (long long) top, maxv - lse, (int) (top == (int64_t) tgt), extra,
                     without);
        // STRATA_LOGPOS_TOPK=K: the K most likely tokens and their log-probabilities too (`id:logprob`), for a
        // top-k comparison with another engine on the same tokens (docs/UNSLOTH_Q4.md)
        static const int topk = [] {
            const char* v = std::getenv("STRATA_LOGPOS_TOPK");
            return v != nullptr ? std::max(0, std::min(256, std::atoi(v))) : 0;
        }();
        if (topk > 0) {
            std::vector<int32_t> order((size_t) n_vocab_);
            for (int64_t v = 0; v < n_vocab_; ++v) order[(size_t) v] = (int32_t) v;
            std::partial_sort(order.begin(), order.begin() + topk, order.end(),
                              [&](int32_t a, int32_t b) { return row[a] > row[b]; });
            for (int j = 0; j < topk; ++j)
                std::fprintf(out, "\t%d:%.6f", order[(size_t) j], (double) row[order[(size_t) j]] - lse);
        }
        std::fprintf(out, "\n");
    }
    std::fflush(out);
    return true;
}

namespace { bool g_commit_async = false; }
void Verifier::set_commit_async(bool on) { g_commit_async = on && std::getenv("STRATA_COMMIT_SYNC") == nullptr; }

bool Verifier::commit(int n_keep, std::string& err) {
#if defined(STRATA_GPU_METAL)
    return strata::metal_engine::window_commit(metal_, n_keep, err);
#endif
    const OnDevice on_device(device_);
=======
bool Verifier::commit(int n_keep, std::string& err, bool wait) {
>>>>>>> their fork (github.com-eddoursul-Strata-custom)
    if (n_keep < 1 || n_keep > last_t_) { err = "verify: commit count out of range"; return false; }
    const Clock::time_point t0 = Clock::now();
    h_commit_[0] = n_keep;
    h_commit_[1] = n_keep - 1;
    for (int t = 0; t < max_t_; ++t) h_commit_[2 + t] = t < n_keep ? (int32_t) (last_pos0_ + t) : -1;
    if (last_t_ > 1) {   // a one-token window has advanced the state itself (record_window)
        std::atomic_thread_fence(std::memory_order_seq_cst);
        // STRATA_COMMIT_TIMING=1 (measurement only): the commit graph's GPU time, from an event pair around it (read at the
        // next commit, when the pair has long completed); printed every 300 commits
        static struct CommitTiming {
            bool on = false, pending = false;
            cudaEvent_t b = nullptr, e = nullptr;
            double ms = 0;
            int64_t n = 0;
            CommitTiming() { const char* v = std::getenv("STRATA_COMMIT_TIMING"); on = v != nullptr && v[0] == '1'; }
        } ct;
        if (ct.on && ct.b == nullptr &&
            (cudaEventCreate(&ct.b) != cudaSuccess || cudaEventCreate(&ct.e) != cudaSuccess)) { ct.on = false; (void) cudaGetLastError(); }
        if (ct.on && ct.pending) {
            float ms = 0;
            if (cudaEventSynchronize(ct.e) == cudaSuccess && cudaEventElapsedTime(&ms, ct.b, ct.e) == cudaSuccess) { ct.ms += ms; ++ct.n; }
            ct.pending = false;
            if (ct.n > 0 && ct.n % 300 == 0)
                std::fprintf(stderr, "strata verify: commit graph %.3f ms GPU on average over %lld commits\n", ct.ms / (double) ct.n,
                             (long long) ct.n);
        }
        if (ct.on) cudaEventRecord(ct.b, cs_);
        const cudaError_t le = cudaGraphLaunch(commit_exec_, cs_);
        if (ct.on) { cudaEventRecord(ct.e, cs_); ct.pending = true; }
        if (le != cudaSuccess) { err = std::string("verify: commit launch: ") + cudaGetErrorString(le); return false; }
        (void) cudaStreamQuery(cs_);   // submit now, not at the next driver call
        commit_pending_ = true;
    }
    for (int t = 0; t < n_keep; ++t) {
        ss_->ple_prev[0] = ss_->ple_prev[1];
        ss_->ple_prev[1] = last_tokens_[t];
    }
    ms_commit += ms_since(t0);
    return !wait || wait_commit(err);
}

bool Verifier::wait_commit(std::string& err) {
<<<<<<< this fork
#if defined(STRATA_GPU_METAL)
    (void) err;
    return true;
#endif
    if (commit_pending_) {
        const OnDevice on_device(device_);
        commit_pending_ = false;
        const cudaError_t se = cudaEventSynchronize(commit_done_);
        if (se != cudaSuccess) { err = std::string("verify: commit: ") + cudaGetErrorString(se); return false; }
    }
    return next_ == nullptr || next_->wait_commit(err);
}


// ================================ BATCH WINDOWS (see init_slots) ================================

bool Verifier::init_slots(const std::vector<SessionState*>& slots, std::string& err) {
    const OnDevice on_device(device_);
    if (g_ == nullptr || ss_ == nullptr) { err = "verify: init_slots before init"; return false; }
    if (slots.empty()) {
        err = "verify: init_slots needs at least one session";
        return false;
    }
    for (SessionState* x : slots) {
        if (x == nullptr || x->layer_lo != ss_->layer_lo || x->layer_hi != ss_->layer_hi ||
            x->max_cells != ss_->max_cells || x->gdn_ord0 != ss_->gdn_ord0 || x->qsa_ord0 != ss_->qsa_ord0) {
            err = "verify: a slot's session is not carved like the stage's own (layer range, context)";
            return false;
        }
    }
    const strata::kernels::QsaShapes s = shapes_of(*g_);
    const int64_t S = (int64_t) slots.size(), CB = 2 + max_t_;
    const int64_t TS = (s.idx_block - 1) * g_->idx_key_dim, nQ = g_->n_qsa_layers();
    void* d = nullptr;
    if (!mapped((size_t) (S * CB * 4 + 16), (void**) &h_commitb_, (void**) &m_commitb_)) {
        err = "verify: the batch commit staging failed";
        return false;
    }
    const uint64_t a = ((uint64_t) S * CB * 4 + 255) & ~255ull;
    if (cudaMalloc(&d, a + (uint64_t) S * std::max<int64_t>(nQ, 1) * TS * 4) != cudaSuccess) {
        err = "verify: the batch buffers do not fit";
        return false;
    }
    arena_b_ = d;
    commitb_ = (int32_t*) d;
    tail_snap_b_ = (float*) ((uint8_t*) d + a);
    slots_ = slots;
    slot_sp_.assign(slots.size(), sampling_);   // greedy until set_slot_sampling
    std::fprintf(stderr, "strata verify: batch windows of up to %lld sequences (layers [%lld, %lld))\n", (long long) S,
                 (long long) lb_, (long long) le_);
    return true;
}

bool Verifier::capture_batch(const int* rows, int S, int hbase, std::string& err) {
    cudaGraphExec_t& ex = exec_bm_[bkey(rows, S, hbase)];
    if (ex != nullptr) return true;
    if (cudaStreamBeginCapture(cs_, cudaStreamCaptureModeThreadLocal) != cudaSuccess) {
        err = "verify: begin batch capture failed";
        return false;
    }
    batch_rec_ = true;
    row_base_ = hbase;
    for (int t = 0; t < S; ++t) brow_[t] = rows[t];
    std::string rerr;
    const bool ok = record_window(S, cs_, rerr);
    batch_rec_ = false;
    row_base_ = 0;
    cudaGraph_t graph = nullptr;
    const cudaError_t ce = cudaStreamEndCapture(cs_, &graph);
    if (!ok || ce != cudaSuccess) {
        if (graph) cudaGraphDestroy(graph);
        err = !ok ? rerr : std::string("verify: end batch capture: ") + cudaGetErrorString(ce);
        return false;
    }
    const bool made = instantiate_evicting(ex, graph, bkey(rows, S, hbase), "verify: batch instantiate: ", err);
    cudaGraphDestroy(graph);
    if (!made) return false;
    cudaGraphUpload(ex, cs_);
    cudaStreamSynchronize(cs_);
    std::string list;
    for (int t = 0; t < S; ++t) list += (t ? "," : "") + std::to_string(rows[t]);
    std::fprintf(stderr, "strata verify: captured the batch window over slots %s\n", list.c_str());
    return true;
}

bool Verifier::capture_commit_batch(const int* rows, int S, int hbase, std::string& err) {
    cudaGraphExec_t& cex = commit_bm_[bkey(rows, S, hbase)];
    if (cex != nullptr) return true;
    using namespace strata::kernels;
    const ModelGeometry& g = *g_;
    const QsaShapes s = shapes_of(g);
    const int64_t C = g.ssm_conv_channels, HV = g.ssm_v_heads, ZV = g.ssm_value_dim, ID = g.idx_key_dim, MT = max_t_;
    const int64_t CB = 2 + MT, nQ = g.n_qsa_layers();
    const uint64_t gdn_floats = (uint64_t) g.ssm_state_size * g.ssm_v_heads * g.ssm_state_size +
                                (uint64_t) g.ssm_conv_channels * (g.ssm_d_conv - 1);
    const int64_t TS = (s.idx_block - 1) * ID;
    const int64_t HS = (int64_t) NG_HIST * NG_HC_DIM;
    if (cudaStreamBeginCapture(cs_, cudaStreamCaptureModeThreadLocal) != cudaSuccess) {
        err = "verify: begin batch commit capture failed";
        return false;
    }
    bool ok = true;
    try {
        for (int t = 0; t < S; ++t)
            if (t == 0 || rows[t] != rows[t - 1])
                copy_i32_from_mapped(commitb_ + (size_t) rows[t] * CB, m_commitb_ + (size_t) rows[t] * CB, CB, cs_);
        int64_t qsa_index = 0, gdn_index = 0;
        for (int64_t l = 0; l < lb_; ++l) (is_qsa_layer(g, l) ? qsa_index : gdn_index) += 1;
        for (int64_t l = lb_; l < le_ && ok; ++l) {
            const LayerView v(*wt_, l);
            if (!is_qsa_layer(g, l)) {
                const WeightRef* wnm = need(v, "ssm_norm.weight", err);
                if (!wnm) { ok = false; break; }
                for (int t = 0; t < S;) {
                    const int first = t;
                    while (t < S && rows[t] == rows[first]) ++t;
                    SessionState& sx = *slots_[(size_t) rows[first]];
                    const int32_t* keep = commitb_ + (size_t) rows[first] * CB;
                    float* state = sx.gdn_state + (size_t) (gdn_index - sx.gdn_ord0) * gdn_floats;
                    float* conv = state + (uint64_t) g.ssm_state_size * g.ssm_v_heads * g.ssm_state_size;
                    gdn_conv_commit(conv, qkv_L_ + (size_t) gdn_index * MT * C + (size_t) first * C,
                                    (int) C, keep, cs_);
                    gdn_step_norm_multi(state, h_L_ + (size_t) gdn_index * MT * C + (size_t) first * C, (int) C,
                                        gate_L_ + (size_t) gdn_index * MT * HV + (size_t) first * HV,
                                        beta_L_ + (size_t) gdn_index * MT * HV + (size_t) first * HV,
                                        z_ + (size_t) first * ZV, (const float*) wnm->data, EPS,
                                        y_dummy_ + (size_t) first * ZV, (int) g.ssm_k_heads,
                                        (int) HV, t - first, keep, cs_,
                                        t - first > 1 ? t - first : 0);   // a 1-row group keeps the 0.1.39 kernel
                }
                ++gdn_index;
            } else {
                const WeightRef* wikn = need(v, "indexer.k_norm.weight", err);
                if (!wikn) { ok = false; break; }
                for (int t = 0; t < S;) {
                    const int first = t;
                    while (t < S && rows[t] == rows[first]) ++t;
                    const QsaState& st = slots_[(size_t) rows[first]]->qsa_states[qsa_index];
                    copy_from_mapped(st.idx_tail, tail_snap_b_ + ((size_t) rows[first] * nQ + qsa_index) * TS,
                                     TS, cs_);
                    const QsaIndexerBuffers ib{st.idx_tail, st.idx_dead, st.idx_pooled, st.idx_block_pos};
                    for (int u = first; u < t; ++u)
                        native_qsa_indexer_append(idx_raw_L_ + (size_t) (qsa_index * MT + u) * ID,
                                                  commitb_ + (size_t) rows[first] * CB + 2 + (u - first),
                                                  0, (const float*) wikn->data, EPS, ib, s, st.max_cells,
                                                  rope_scaling(), cs_);
                }
                ++qsa_index;
            }
        }
        if (ok && ss_->ple.ready() && ple_stage())
            for (int t = 0; t < S;) {
                const int first = t;
                while (t < S && rows[t] == rows[first]) ++t;
                copy_indexed(slots_[(size_t) rows[first]]->ple_hist, hist_snap_ + (size_t) first * HS,
                             HS, commitb_ + (size_t) rows[first] * CB + 1, HS, cs_);
            }
    } catch (const std::exception& e) {
        err = std::string("verify batch commit: ") + e.what();
        ok = false;
    }
    cudaGraph_t graph = nullptr;
    const cudaError_t ce = cudaStreamEndCapture(cs_, &graph);
    if (!ok) {
        if (graph) cudaGraphDestroy(graph);
        return false;
    }
    if (ce != cudaSuccess) {
        if (graph) cudaGraphDestroy(graph);
        err = std::string("verify: batch commit capture: ") + cudaGetErrorString(ce);
        return false;
    }
    const bool made = instantiate_evicting(cex, graph, bkey(rows, S, hbase), "verify: batch commit instantiate: ", err);
    cudaGraphDestroy(graph);
    return made;
}

bool Verifier::evict_batch_graph(const std::vector<int>& keep, bool& evicted, std::string& err) {
    evicted = false;
    auto old = exec_bm_.end();
    uint64_t oldest = UINT64_MAX;
    for (auto it = exec_bm_.begin(); it != exec_bm_.end(); ++it) {
        if (it->first == keep) continue;
        const auto u = bm_used_.find(it->first);
        const uint64_t t = u == bm_used_.end() ? 0 : u->second;
        if (t < oldest) { oldest = t; old = it; }
    }
    if (old == exec_bm_.end()) return true;
    if (cudaStreamSynchronize(cs_) != cudaSuccess) {
        err = "verify: synchronizing before batch graph eviction failed";
        return false;
    }
    const auto old_key = old->first;
    if (old->second) cudaGraphExecDestroy(old->second);
    exec_bm_.erase(old);
    bm_used_.erase(old_key);
    auto commit_old = commit_bm_.find(old_key);
    if (commit_old != commit_bm_.end()) {
        if (commit_old->second) cudaGraphExecDestroy(commit_old->second);
        commit_bm_.erase(commit_old);
    }
    evicted = true;
    return true;
}

bool Verifier::instantiate_evicting(cudaGraphExec_t& ex, cudaGraph_t graph, const std::vector<int>& key,
                                   const char* what, std::string& err) {
    // #997: every layout of active slots keeps its own graph pair (20-30 MiB each on an L40S), and with an expert cache
    // sized to the reserve the next window graph - a batch layout or a one-request window captured on first use - can
    // find no VRAM left: free the least recently used layouts until it fits (they are captured again when needed)
    cudaError_t ie = cudaGraphInstantiate(&ex, graph, 0);
    int freed = 0;
    for (bool evicted = true; ie == cudaErrorMemoryAllocation && evicted;) {
        (void) cudaGetLastError();
        if (!evict_batch_graph(key, evicted, err)) return false;
        if (evicted) {
            ++freed;
            ie = cudaGraphInstantiate(&ex, graph, 0);
        }
    }
    if (freed > 0)
        std::fprintf(stderr, "strata verify: no VRAM for a new window graph: freed the graphs of %d older batch slot "
                             "layouts, %zu kept (a larger --vram-reserve-mib keeps more)\n", freed, exec_bm_.size());
    if (ie != cudaSuccess) {
        err = std::string(what) + cudaGetErrorString(ie);
        return false;
    }
    return true;
}

bool Verifier::stage_batch(const int* rows, int S, int hbase, const int32_t* tokens, const int64_t* pos,
                           std::string& err) {
    using namespace strata::kernels;
    // the row layout: a slot may own a contiguous group of rows (--batch-mtp), also across the stages of a layer split,
    // whose hand-off buffers hold kVerifyMaxT rows (a stage that hands rows on or takes them)
    if (const char* why = batch_rows_error(rows, S, pos, (int) slots_.size(), max_t_, hbase,
                                           next_ != nullptr || lb_ > 0, kVerifyMaxT)) {
        err = why;
        return false;
    }
    if (released_.load()) { err = "verify: an earlier window never finished on the GPU (#267); restart the engine"; return false; }
    const ModelGeometry& g = *g_;
    for (int t = 0; t < S; ++t)
        if (pos[t] < 0 || pos[t] + 1 > slots_[(size_t) rows[t]]->max_cells) {
            err = "verify: slot " + std::to_string(rows[t]) + " runs past its context";
            return false;
        }
    refresh_ar();
    // --batch-mtp only (limit 0 = 0.1.39: no eviction): slot rotation creates new layouts; bound the captured graph
    // pairs, evicting the least recently used layout.  Without a limit the same eviction frees VRAM for a capture.
    const auto key = bkey(rows, S, hbase);
    bool evicted = false;
    if (batch_graph_limit_ > 0 && exec_bm_.find(key) == exec_bm_.end() && exec_bm_.size() >= batch_graph_limit_ &&
        !evict_batch_graph(key, evicted, err))
        return false;
    bm_used_[key] = ++bm_tick_;
    if (!capture_batch(rows, S, hbase, err) || !capture_commit_batch(rows, S, hbase, err)) return false;
=======
    if (!commit_pending_) return true;
>>>>>>> their fork (github.com-eddoursul-Strata-custom)
    const Clock::time_point t0 = Clock::now();
    commit_pending_ = false;
    const cudaError_t se = cudaStreamSynchronize(cs_);
<<<<<<< this fork
    if (se != cudaSuccess) { err = std::string("verify: batch commit: ") + cudaGetErrorString(se); return false; }
    if (ple_stage())
        for (int t = 0; t < S;) {
            const int first = t;
            while (t < S && last_rows_[t] == last_rows_[first]) ++t;
            SessionState& sx = *slots_[(size_t) last_rows_[first]];
            for (int u = first; u < first + keep[last_rows_[first]]; ++u) {
                sx.ple_prev[0] = sx.ple_prev[1];
                sx.ple_prev[1] = last_tokens_[u];
            }
        }
    ms_commit += ms_since(t0);
    return next_ == nullptr || next_->commit_slot_prefixes(keep, err);
}

bool Verifier::sample_rows(int S, std::string& err) {
    bool any = false;
    for (int t = 0; t < S; ++t) {
        strata::kernels::SamplerParams sp = slot_sp_[(size_t) last_rows_[t]];
        if (sp.greedy || sp.temperature <= 0.0f) continue;
        sp.counter = (uint64_t) last_pos_b_[t];   // Philox(seed, position): the solo window's draw for this position
        sp.penalty_last_n = 0;
        strata::kernels::sample_tokens(head_logits_ + (size_t) t * (size_t) n_vocab_, 1, (int) n_vocab_, nullptr, 0, sp,
                                       m_out_ + t, cs_);
        any = true;
    }
    if (any && cudaStreamSynchronize(cs_) != cudaSuccess) { err = "verify batch: the row sampling failed"; return false; }
    return true;
}

bool Verifier::batch_launch(int base, int S, const int32_t* tokens, const int64_t* pos, std::string& err) {
    const OnDevice on_device(device_);
    if (b_running_) { err = "verify: batch_launch while this stage is busy"; return false; }
    int rows[8] = {};
    for (int t = 0; t < S && t < 8; ++t) rows[t] = base + t;
    if (!stage_batch(rows, S, base, tokens, pos, err)) return false;
    cudaError_t le = cudaGraphLaunch(exec_bm_[bkey(rows, S, base)], cs_);
    if (le == cudaSuccess) le = cudaGraphLaunch(commit_bm_[bkey(rows, S, base)], cs_);   // right behind it: every row is kept
    if (le != cudaSuccess) { err = std::string("verify: batch launch: ") + cudaGetErrorString(le); return false; }
    (void) cudaStreamQuery(cs_);
    if (ple_stage())   // the host's side of the commit (the hash's last two tokens)
        for (int t = 0; t < S; ++t) {
            SessionState& sx = *slots_[(size_t) rows[t]];
            sx.ple_prev[0] = sx.ple_prev[1];
            sx.ple_prev[1] = tokens[t];
        }
    b_running_ = true;
    b_k_ = 0;
    b_steps_ = le_ - lb_;
    b_last_ = Clock::now();
    return true;
}

int Verifier::batch_poll(PoolMultiFn pool, void* user, std::string& err) {
    if (!b_running_) return 1;
    const OnDevice on_device(device_);
    volatile uint32_t* const seq = h_seq_;
    const int S = last_t_;
    // #646: an all-resident stage's graph raises no host doorbells (see run_slot_rows): nothing to serve per layer,
    // so the poll is just "has the graph finished" - except the PLE flag, which the graph's first wait reads and
    // only the host can raise (the same raise run_slot_rows makes).  Per-window (ar_on): with a loan in flight the
    // doorbell graph needs the per-layer service below.
    if (ar_on() && ss_->ple.ready() && ple_stage()) {
        std::atomic_thread_fence(std::memory_order_seq_cst);
        _mm_sfence();
        *h_flag_ = 1;
    }
    while (!ar_on() && b_k_ < b_steps_) {
        const uint32_t want = (uint32_t) (b_k_ + 1);
        if (*seq < want) {
            const auto now = Clock::now();
            if (now - b_last_ > std::chrono::milliseconds(2)) {
                const cudaError_t q = cudaStreamQuery(cs_);
                if (q != cudaErrorNotReady && *seq < want) {
                    err = "verify batch: layer " + std::to_string(lb_ + b_k_) + " never rang (" +
                          (q == cudaSuccess ? std::string("graph finished") : std::string(cudaGetErrorString(q))) + ")";
                    b_running_ = false;
                    return -1;
                }
                if (now - b_last_ > std::chrono::seconds(20)) {
                    err = "verify batch: timed out at layer " + std::to_string(lb_ + b_k_) + released_note(release_gpu_waits(5000));
                    b_running_ = false;
                    return -1;
                }
            }
            return 0;
        }
        const Clock::time_point b = Clock::now();
        cur_layer_ = want - 1;
        set_plan_slot(0);
        if (remote_opt_) remote_opt_->begin(h_w_, 0, S);
        if (pool != nullptr) pool(user, h_x_, h_ids_, S, ss_->k, h_ymiss_, lb_ + b_k_);
        if (remote_opt_) remote_opt_->end();
        progress_tick();
        std::atomic_thread_fence(std::memory_order_seq_cst);
        _mm_sfence();
        if (*(volatile uint32_t*) h_flagA_ != want) {        // the pool did not publish a plan: an empty one
            sink_.counts[0] = 0;
            sink_.counts[1] = 0;
            sink_.counts[2] = 0;
            sink_.start[0] = 0;
            sink_.start2[0] = 0;
            std::atomic_thread_fence(std::memory_order_seq_cst);
            *(volatile uint32_t*) h_flagA_ = want;
            raise_flag(h_flagB_, want);
        }
        *(volatile uint32_t*) h_flag_ = want;
        ms_pool += ms_since(b);
        b_last_ = Clock::now();
        ++b_k_;
    }
    const cudaError_t q = cudaStreamQuery(cs_);
    if (q == cudaErrorNotReady) return 0;
    if (q != cudaSuccess) { err = std::string("verify batch: ") + cudaGetErrorString(q); b_running_ = false; return -1; }
    const cudaError_t qc = cudaStreamQuery(copy_);   // no host function of this window may raise flag B in the next
    if (qc == cudaErrorNotReady) return 0;
    if (prof_on_) collect_profile();
    if (last_stage()) {
        if (!sample_rows(S, err)) { b_running_ = false; return -1; }
        for (int t = 0; t < S; ++t) b_out_[t] = ((volatile int32_t*) h_out_)[t];
        note_nonfinite(S, last_pos_b_);
    }
    ++windows;
    b_running_ = false;
    progress_beat();
    return 1;
}

// #879: STRATA_NAN_GUARD=0 turns the guard off (no kernel is recorded; the graphs are captured once, so it is read once)
bool Verifier::nan_guard_on() {
    static const bool on = [] {
        const char* v = std::getenv("STRATA_NAN_GUARD");
        return v == nullptr || std::atoi(v) != 0;
    }();
    return on;
}

namespace {
// STRATA_NAN_INJECT=<n>[,<n>...] (debug): the windows (counted from 1 over the process) the guard reports as non-finite
const std::vector<long long>& nan_inject_list() {
    static const std::vector<long long> list = [] {
        std::vector<long long> v;
        if (const char* e = std::getenv("STRATA_NAN_INJECT"))
            for (const char* q = e; *q != 0;) {
                char* end = nullptr;
                const long long n = std::strtoll(q, &end, 10);
                if (end == q) break;
                if (n > 0) v.push_back(n);
                q = *end == ',' ? end + 1 : end;
            }
        return v;
    }();
    return list;
}
std::atomic<long long> g_nf_windows{0};
}  // namespace

void Verifier::note_nonfinite(int T, const int64_t* pos_b) {
    nf_row_ = -1;
    nf_pos_ = -1;
    nf_mask_ = 0;
    if (!nan_guard_on() || h_nf_ == nullptr || !last_stage()) return;   // (a stage without the head checks nothing)
    const volatile uint32_t* f = h_nf_;
    for (int t = 0; t < T && t < strata::kernels::kVerifyMaxT; ++t)
        for (int b = 0; b < strata::kernels::kNonfiniteBlocks; ++b)
            if (f[t * strata::kernels::kNonfiniteBlocks + b] != 0) { nf_mask_ |= 1u << t; break; }
    const std::vector<long long>& inject = nan_inject_list();
    if (!inject.empty()) {
        const long long k = g_nf_windows.fetch_add(1) + 1;
        if (nf_mask_ == 0 && std::find(inject.begin(), inject.end(), k) != inject.end()) {
            nf_mask_ = 1u;
            std::fprintf(stderr, "strata verify: STRATA_NAN_INJECT: window %lld reported as non-finite\n", k);
        }
    }
    for (int t = 0; t < T && nf_row_ < 0; ++t)
        if ((nf_mask_ >> t) & 1u) nf_row_ = t;
    if (nf_row_ >= 0) nf_pos_ = pos_b != nullptr ? pos_b[nf_row_] : last_pos0_ + nf_row_;
}

bool Verifier::copy_logits(int t, float* host) const {
#if defined(STRATA_GPU_METAL)
    return strata::metal_engine::window_copy_logits(metal_, t, host);
#endif
    if (next_ != nullptr) return next_->copy_logits(t, host);   // a layer split: the head is on the last stage
    if (head_logits_ == nullptr || host == nullptr || t < 0 || n_vocab_ <= 0) return false;
    return cudaMemcpy(host, head_logits_ + (size_t) t * (size_t) n_vocab_, (size_t) n_vocab_ * sizeof(float),
                      cudaMemcpyDeviceToHost) == cudaSuccess;
}


// ================================ PIPELINED WINDOWS (see verify.hpp) ================================
//
// The window is run()'s, step for step: the same graph, the same staging, the same per-layer service (the PLE rows
// gathered while layer 0 is served, the zero-doorbell graph's one flag), the same commit graph.  Only the waits are
// split up: the host polls instead of spinning, so it can serve the other stage's window in between.

namespace {
double now_ms() { return std::chrono::duration<double, std::milli>(Clock::now().time_since_epoch()).count(); }
}  // namespace

void Verifier::diag_pipelined(std::FILE* f, const char* name) const {
    auto rd = [](const uint32_t* p) { return p ? *(const volatile uint32_t*) p : 0u; };
    auto ev = [](cudaEvent_t e) {
        if (e == nullptr) return "none";
        const cudaError_t q = cudaEventQuery(e);
        return q == cudaSuccess ? "done" : q == cudaErrorNotReady ? "PENDING" : "error";
    };
    std::fprintf(f, "  %s: %s T=%d pos %lld served %lld/%lld; GPU rang %u, flags served %u A %u B %u; window event %s, "
                    "commit event %s (commit launched %d)\n", name, fl_active_ ? "IN FLIGHT" : "idle", last_t_,
                 (long long) last_pos0_, (long long) fl_k_, (long long) fl_total_, rd(h_seq_), rd(h_flag_), rd(h_flagA_),
                 rd(h_flagB_), ev(ev_done_), ev(ev_commit_), (int) commit_live_);
}

bool Verifier::capture_all(std::string& err) {
    const OnDevice on_device(device_);
    if (g_ == nullptr) { err = "verify: capture_all before init"; return false; }
    if (remote_opt_ != nullptr) { err = "verify: pipelined windows do not serve --remote-expert-opt"; return false; }
    if (released_.load()) { err = "verify: an earlier window never finished on the GPU (#267); restart the engine"; return false; }
    for (int T = 1; T <= max_t_; ++T)
        if (!capture(T, err)) return false;
    if (!capture_commit(err)) return false;
    if ((ev_done_ == nullptr && cudaEventCreateWithFlags(&ev_done_, cudaEventDisableTiming) != cudaSuccess) ||
        (ev_commit_ == nullptr && cudaEventCreateWithFlags(&ev_commit_, cudaEventDisableTiming) != cudaSuccess)) {
        err = "verify: event create failed";
        return false;
    }
    if (prof_on_ && prof_pin_ == nullptr &&
        cudaHostAlloc((void**) &prof_pin_, prof_h_.size() * 8, cudaHostAllocDefault) != cudaSuccess) {
        prof_pin_ = nullptr;   // the pipelined windows go unprofiled
        cudaGetLastError();
    }
    pl_ple_rows_.assign((size_t) strata::kernels::kVerifyMaxT * strata::kernels::PLE_N_HEADS, 0u);
    return true;
}

// The host staging of a pipelined window: run()'s (stage_inputs), and its PLE rows from `ple_prev` with their pages
// prefetched (they are gathered into the mapped rows when layer 0 is served, as run() does).
void Verifier::pl_stage(int T, const int32_t* tokens, int64_t pos0, const int32_t ple_prev[2]) {
    using namespace strata::kernels;
    stage_inputs(T, tokens, pos0);
    staged_ = false;   // a later run() stages its own window
    SessionState& ss = *ss_;
    fl_ple_ = ss.ple.ready() && ple_stage();
    if (fl_ple_) {
        int32_t prev[2] = {ple_prev[0], ple_prev[1]};
        for (int t = 0; t < T; ++t) {
            ngram_rows(&tokens[t], prev, 1, ss.ple.consts, pl_ple_rows_.data() + t * PLE_N_HEADS);
            prev[0] = prev[1];
            prev[1] = tokens[t];
            ss.ple.table->prefetch_rows(pl_ple_rows_.data() + t * PLE_N_HEADS);
        }
    }
    pl_prev_[0] = ple_prev[0];
    pl_prev_[1] = ple_prev[1];
}

bool Verifier::prestage(int T, const int32_t* tokens, int64_t pos0, const int32_t ple_prev[2], std::string& err) {
    if (fl_active_) { err = "verify: a window is in flight on this verifier"; return false; }
    if (T < 1 || T > max_t_) { err = "verify: window size out of range"; return false; }
    if (pl_ple_rows_.empty()) { err = "verify: pipelined window not prepared (capture_all)"; return false; }
    const Clock::time_point t0 = Clock::now();
    pl_stage(T, tokens, pos0, ple_prev);
    pl_prestaged_ = true;
    ms_host += ms_since(t0);
    return true;
}

bool Verifier::pl_launch(int T, const int32_t* tokens, int64_t pos0, std::string& err) {
    const OnDevice on_device(device_);
    if (fl_active_) { err = "verify: a window is already in flight on this verifier"; return false; }
    if (released_.load()) { err = "verify: an earlier window never finished on the GPU (#267); restart the engine"; return false; }
    if (T < 1 || T > max_t_) { err = "verify: window size out of range"; return false; }
    SessionState& ss = *ss_;
    if (pos0 + T > ss.qsa_states[ss.qsa_primary()].max_cells) { err = "verify: the window runs past the context"; return false; }
    if (exec_[T] == nullptr || commit_exec_ == nullptr || ev_done_ == nullptr || pl_ple_rows_.empty()) {
        err = "verify: pipelined window not prepared (capture_all)";
        return false;
    }
    const Clock::time_point t0 = Clock::now();
    last_batch_ = false;
    bool staged = pl_prestaged_ && last_t_ == T && last_pos0_ == pos0;
    for (int t = 0; staged && t < T; ++t) staged = last_tokens_[t] == tokens[t];
    if (staged && ss.ple.ready() && ple_stage())
        staged = pl_prev_[0] == ss.ple_prev[0] && pl_prev_[1] == ss.ple_prev[1];
    pl_prestaged_ = false;
    if (!staged) pl_stage(T, tokens, pos0, ss.ple_prev);
    const int G = groups_[T] > 0 ? groups_[T] : 1;
    fl_T_ = T;
    fl_k_ = 0;
    fl_total_ = (le_ - lb_) * G;
    fl_prof_ = prof_on_ && G == 1 && prof_pin_ != nullptr;
    if (trace_h_ != nullptr) std::memset(trace_h_, 0, trace_n_ * 8);   // #649: this window's breadcrumbs only
    std::atomic_thread_fence(std::memory_order_seq_cst);
    trace_ev("WINDOW (pipelined)", -1, -1, pos0 * 16 + T);
    ms_host += ms_since(t0);
    const cudaError_t le = cudaGraphLaunch(exec_[T], cs_);
    trace_ev("LAUNCHED", -1, -1, (int64_t) le);
    if (le != cudaSuccess) { err = std::string("verify: launch: ") + cudaGetErrorString(le); return false; }
    if (fl_prof_) cudaMemcpyAsync(prof_pin_, prof_, prof_h_.size() * 8, cudaMemcpyDeviceToHost, cs_);
    if (cudaEventRecord(ev_done_, cs_) != cudaSuccess) { err = "verify: event record failed"; return false; }
    (void) cudaStreamQuery(cs_);   // WDDM: submit now
    fl_active_ = true;
    fl_since_ms_ = fl_flush_ms_ = fl_launch_ms_ = now_ms();
    return true;
}

int Verifier::service(PoolMultiFn pool, void* user, std::string& err) {
    if (!fl_active_) return 1;
    if (fl_k_ >= fl_total_) return 1;
    // Nothing rang yet and neither the WDDM flush nor the timeout below is due: return before any CUDA call.  The
    // pipelined loop asks both stages' windows this millions of times a second while they wait on their doorbells,
    // and the device switch (OnDevice: cudaGetDevice and two cudaSetDevice) made that a few driver calls each time:
    // 919 million passes before the host stalled inside one of those cudaSetDevice calls with both stages waiting
    // on it (0.1.41, Windows, an RTX 3060 + RTX 5070 Ti split).  The same answer as the loop below gives.
    if (!ar_on() && *(volatile uint32_t*) h_seq_ < (uint32_t) (fl_k_ + 1)) {
        const double now = now_ms();
        if (now - fl_flush_ms_ <= 2.0 && now - fl_since_ms_ <= 20000.0) return 0;
    }
    const OnDevice on_device(device_);
    const ModelGeometry& g = *g_;
    SessionState& ss = *ss_;
    const int T = fl_T_;
    // the PLE rows into the mapped staging (layer 1's pre() copies them once flag 1 is up), as run() does at k == 0
    auto gather_ple = [&]() -> bool {
        if (!fl_ple_) return true;
        const Clock::time_point tp = Clock::now();
        if (!ss.ple.table->gather_batch(pl_ple_rows_.data(), (size_t) T, h_ple_, err)) return false;
        std::atomic_thread_fence(std::memory_order_seq_cst);
        _mm_sfence();
        ms_host += ms_since(tp);
        return true;
    };
    if (ar_on()) {   // the zero-doorbell graph never rings: it waits only for flag 1 (stage 0's PLE rows).
        // Per-window: with a loan in flight (ar_off_) the doorbell graph needs the per-layer service below.
        if (!gather_ple()) return -1;
        *(volatile uint32_t*) h_flag_ = 1;
        fl_k_ = fl_total_;
        progress_tick();
        return 1;
    }
    const int G = groups_[T] > 0 ? groups_[T] : 1;
    const int gtb[2] = {0, (T + 1) / 2}, gte[2] = {G == 2 ? (T + 1) / 2 : T, T};
    while (fl_k_ < fl_total_) {
        const int64_t l = lb_ + fl_k_ / G;
        const uint32_t want = (uint32_t) (fl_k_ + 1);
        if (*(volatile uint32_t*) h_seq_ < want) {
            const double now = now_ms();
            if (now - fl_flush_ms_ > 2.0) {   // flush WDDM and notice a dead graph, as run() does
                fl_flush_ms_ = now;
                const cudaError_t q = cudaEventQuery(ev_done_);
                if (q != cudaErrorNotReady && *(volatile uint32_t*) h_seq_ < want) {
                    trace_ev("NEVER-RANG", fl_k_, l, (int64_t) q);
                    err = "verify: layer " + std::to_string(l) + " never rang (" +
                          (q == cudaSuccess ? std::string("graph finished") : std::string(cudaGetErrorString(q))) + ")";
                    return -1;
                }
                (void) cudaStreamQuery(cs_);
            }
            if (now - fl_since_ms_ > 20000.0) {   // #267: no spin kernel may outlive the engine
                trace_ev("TIMEOUT", fl_k_, l, 0);
                err = "verify: timed out at layer " + std::to_string(l) + released_note(release_gpu_waits(5000));
                return -1;
            }
            return 0;
        }
        const Clock::time_point b = Clock::now();
        ms_wait += now_ms() - fl_since_ms_;
        const int grp = (int) (fl_k_ % G);
        cur_layer_ = want - 1;
        set_plan_slot(grp);
        const int tb = gtb[grp], n = gte[grp] - gtb[grp];
        progress_at("verify window (pipelined): the CPU experts of layer", l);
        if (pool != nullptr)
            pool(user, h_x_ + (size_t) tb * g.n_embd, h_ids_ + (size_t) tb * ss.k, n, ss.k,
                 h_ymiss_ + (size_t) tb * ss.k * g.n_embd, l);
        if (g_trace) trace_ev(*(volatile uint32_t*) h_flagA_ == want ? "SERVED" : "SERVED-NO-PLAN-YET", fl_k_, l,
                              (int64_t) ms_since(b));
        progress_tick();
        std::atomic_thread_fence(std::memory_order_seq_cst);
        _mm_sfence();
        if (*(volatile uint32_t*) h_flagA_ != want) {        // the pool did not publish a plan: an empty one
            sink_.counts[0] = 0;
            sink_.counts[1] = 0;
            sink_.counts[2] = 0;
            sink_.start[0] = 0;
            sink_.start2[0] = 0;
            std::atomic_thread_fence(std::memory_order_seq_cst);
            *(volatile uint32_t*) h_flagA_ = want;
            raise_flag(h_flagB_, want);
        }
        if (fl_k_ == 0 && !gather_ple()) return -1;
        *(volatile uint32_t*) h_flag_ = want;
        ++fl_k_;
        ms_pool += ms_since(b);
        fl_since_ms_ = fl_flush_ms_ = now_ms();
    }
    return 1;
}

bool Verifier::done(std::string& err) {
    if (!fl_active_ || fl_k_ < fl_total_) return false;
    const OnDevice on_device(device_);
    const cudaError_t q = cudaEventQuery(ev_done_);
    if (q == cudaErrorNotReady) {
        const double now = now_ms();
        if (now - fl_flush_ms_ > 2.0) { fl_flush_ms_ = now; (void) cudaStreamQuery(cs_); }
        if (now - fl_since_ms_ > 20000.0) {   // every layer served, and the graph still runs: #267 as above
            err = "verify: the window never finished" + released_note(release_gpu_waits(5000));
        }
        return false;
    }
    if (q != cudaSuccess) { err = std::string("verify: ") + cudaGetErrorString(q); return false; }
    if (copy_used_) {   // no host function of this window may raise flag B in the next one
        if (cudaStreamQuery(copy_) == cudaErrorNotReady) return false;
        copy_used_ = false;
    }
    return true;
}

bool Verifier::pl_finish(int32_t* out, std::string& err) {
    using namespace strata::kernels;
    const OnDevice on_device(device_);
    const ModelGeometry& g = *g_;
    fl_active_ = false;
    if (fl_prof_) accumulate_profile(prof_pin_);
    ++windows;
    trace_ev("DONE (pipelined)", -1, -1, (int64_t) (now_ms() - fl_launch_ms_));
    if (le_ < g.n_layers) return true;   // an earlier stage: the hand-off is written
    const int T = fl_T_;
    const bool sampled = !sampling_.greedy && sampling_.temperature > 0.0f;
    if (head_sampling_ && (sampled || hist_d_ != nullptr || sampling_.logit_bias != nullptr)) {   // run()'s host-side sampling, Philox(seed, pos0 + t)
        SamplerParams sp = sampling_;
        sp.counter = (uint64_t) last_pos0_;
        sample_tokens(head_logits_, T, (int) n_vocab_, hist_d_, hist_len_, sp, m_out_, cs_);
        if (cudaStreamSynchronize(cs_) != cudaSuccess) {
            err = "verify: the head sampling failed";
            return false;
        }
    }
    if (out != nullptr)
        for (int t = 0; t < T; ++t) out[t] = ((volatile int32_t*) h_out_)[t];
    note_nonfinite(T, nullptr);
    progress_beat();
    return true;
}

bool Verifier::pl_commit_async(int n_keep, std::string& err) {
    const OnDevice on_device(device_);
    if (n_keep < 1 || n_keep > last_t_) { err = "verify: commit count out of range"; return false; }
    if (commit_exec_ == nullptr || ev_commit_ == nullptr) { err = "verify: pipelined commit not prepared"; return false; }
    const Clock::time_point t0 = Clock::now();
    if (commit_live_) {   // its graph reads the words below when it starts: the previous one has (a window ago)
        cudaError_t q;
        while ((q = cudaEventQuery(ev_commit_)) == cudaErrorNotReady) _mm_pause();
        if (q != cudaSuccess) { err = std::string("verify: commit: ") + cudaGetErrorString(q); return false; }
    }
    h_commit_[0] = n_keep;
    h_commit_[1] = n_keep - 1;
    for (int t = 0; t < max_t_; ++t) h_commit_[2 + t] = t < n_keep ? (int32_t) (last_pos0_ + t) : -1;
    std::atomic_thread_fence(std::memory_order_seq_cst);
    const cudaError_t le = cudaGraphLaunch(commit_exec_, cs_);
    if (le != cudaSuccess) { err = std::string("verify: commit launch: ") + cudaGetErrorString(le); return false; }
    if (cudaEventRecord(ev_commit_, cs_) != cudaSuccess) { err = "verify: event record failed"; return false; }
    (void) cudaStreamQuery(cs_);
    commit_live_ = true;
    if (ple_stage())
        for (int t = 0; t < n_keep; ++t) {
            ss_->ple_prev[0] = ss_->ple_prev[1];
            ss_->ple_prev[1] = last_tokens_[t];
        }
=======
>>>>>>> their fork (github.com-eddoursul-Strata-custom)
    ms_commit += ms_since(t0);
    if (se != cudaSuccess) { err = std::string("verify: commit: ") + cudaGetErrorString(se); return false; }
    return true;
}

}  // namespace strata::core
