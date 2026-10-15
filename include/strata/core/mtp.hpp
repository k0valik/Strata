// include/strata/core/mtp.hpp - plan v0.3 P6: the MTP draft layer on the GPU.
//
// The model ships one multi-token-prediction layer (vLLM 0.30.0 `qwen4_exp` MTP; transcribed and validated in
// `tools/mtp_probe.py`, where it accepts 0.89 / 0.86 / 0.85 of greedy drafts at steps 1-3).  Cell i of its
// sequence pairs the main model's final multi-stream residual at position i with the token at position i+1, at
// rope position i (vLLM's convention); its output predicts the token at position i+2, and its own residual feeds
// the next draft step.
//
//   e = fc_embedding(rms(embed(tok)) * (1+w))           h = fc_hidden per stream(rms_10240(R) * (1+w))
//   R = h + e  ->  attention hyper-connection  ->  QSA attention (own K/V)  ->  MLP hyper-connection  ->
//   MoE (512 experts, top-10, shared expert)  ->  R += y * 2 sigmoid(inject / hc)  ->  final mixer  ->  head
//
// Runtime choices, each a trade of exactness for simplicity that only affects DRAFT quality, never the output
// (the verify window decides every emitted token):
//   * the large projections run as Q8_0 through the multi-column MMVQ (`tools/mtp_rt.py` quantizes them);
//   * the attention is DENSE over every cell the layer has seen - identical to the model's sparse selection
//     below 2,051 cells - so speculative cells (draft steps, rejected window rows) never touch indexer state and
//     are simply overwritten when their positions are processed again;
//   * all 512 routed experts live in VRAM and run through the grouped kernels: raw GGUF blocks in the formats
//     `experts.txt` names, through the main model's native expert kernels, or else the Q2_0 blob layout (708 MB);
//   * the K/V of a prompt's cells come from the batched prompt path (`prompt_kv`), in its GEMM arithmetic, as the
//     model's own layers there.
#pragma once

#include "strata/core/layer.hpp"
#include "strata/core/session.hpp"
#include "strata/kernels/iq_kernels.hpp"

#include <cuda_runtime.h>

#include <cstdint>
#include <functional>
#include <string>
#include <vector>

namespace strata::core {

class NativeHead;

/// What the batched prompt path needs for the draft layer's K/V over a prompt (strata/prefill/prefill.hpp): the two
/// input norms and projections, the attention hyper-connection read, the K and V projections and the K norm.
struct MtpPromptKv {
    const float *norm_emb = nullptr, *norm_hidden = nullptr;   ///< [n_embd], [hc * n_embd]
    const void *fc_emb = nullptr, *fc_hidden = nullptr;        ///< Q8_0 [n_embd, n_embd]
    const float* hc_norm = nullptr;                             ///< [hc * n_embd]
    const uint16_t *hc_down = nullptr, *hc_up = nullptr;        ///< BF16 [hc_lr, hc * n_embd], [hc * n_embd, hc_lr]
    const void *k_proj = nullptr, *v_proj = nullptr;            ///< Q8_0 [n_head_kv * head_dim, n_embd]
    const float* k_norm = nullptr;                              ///< [head_dim]
    QsaState* kv = nullptr;                                     ///< the layer's K/V cells
    int64_t first_cell = 0;                                     ///< cells before it the window never reaches again
};

class MtpDrafter {
public:
    MtpDrafter() = default;
    ~MtpDrafter();
    MtpDrafter(const MtpDrafter&) = delete;
    MtpDrafter& operator=(const MtpDrafter&) = delete;

    /// Loads `rt_dir` (from tools/mtp_rt.py) and allocates the layer's K/V and buffers for up to `max_t` rows.
    /// Call before the VRAM expert tier is sized: this takes ~0.9 GB.
    bool load(const std::string& rt_dir, const ModelGeometry& g, SessionState& ss, int max_t, std::string& err,
              int64_t window = 32768);
    /// The prompt's length: the prompt's K/V skip the cells the attention window can never reach again.
    void set_prompt_len(int64_t n) { prompt_len_ = n; }
    /// The draft chain stops after `n` drafts (0: max_t - 1).
    void set_max_drafts(int n) { max_drafts_ = n; }
    uint64_t vram_bytes() const { return vram_; }
    /// The layer's own K/V cells (the prompt cache stashes them with the model's).
    QsaState& kv_state() { return st_; }
    /// The weights and cells the batched prompt path computes the prompt's K/V with (after load and set_prompt_len).
    MtpPromptKv prompt_kv();
    /// The main model's embedding and head, and the verify window's final residuals (T rows, hc*n_embd each).
<<<<<<< this fork
    bool bind(const WeightTable& wt, const NativeHead* head, const float* window_R, std::string& err,
              const MtpDrafter* shared = nullptr);
    void set_window_R(const float* window_R) { window_R_ = window_R; }

    /// Prompt cells [cell0, cell0 + n): residual rows `R_rows` (device, hc*n_embd each) and `next_tokens` (host,
    /// the token at position cell+1).  Runs in batches of up to max_t rows.  `sync` false: returns without waiting
    /// for the drafter's stream (the caller orders on it: `stream()`).
    bool prefill(const float* R_rows, const int32_t* next_tokens, int64_t n, int64_t cell0, std::string& err,
                 bool sync = true);
    /// --pipeline-windows: capture the prefill graphs and size its input records for windows of up to max_t rows now,
    /// so a later prefill(sync = false) of a window neither captures nor allocates (both wait for the device).
    bool prepare_prefill(std::string& err);
    cudaStream_t stream() const { return cs_; }
=======
    bool bind(const WeightTable& wt, const NativeHead* head, const float* window_R, std::string& err);
>>>>>>> their fork (github.com-eddoursul-Strata-custom)

    /// One round: catch-up over T cells from `p` (rows = the window's final residuals, `tokens` = the window's
    /// argmaxes: row t pairs R_{p+t} with the token at p+t+1), then the draft chain from row `a` (the last
    /// accepted row) for T-1 drafts at cells p+a+1 ...  `drafts` gets T-1 tokens.
    bool draft(int T, const int32_t* tokens, int64_t p, int a, int32_t* drafts, std::string& err,
               float* probs = nullptr, float min_p = 0.0f, int* n_drafts = nullptr);

    /// The first round: one cell (`cell`) from `R_row` (device) and `token` -> T-1 drafts.
    /// The draft head's rows gathered from the main head now (before the main GPU keeps only part of it); `bind`
    /// then takes them as they are.
    bool bind_head(const NativeHead* head, std::string& err);
    bool has_draft_head() const { return dhead_ != nullptr; }
    bool draft_first(int T, const float* R_row, int32_t token, int64_t cell, int32_t* drafts, std::string& err,
                     float* probs = nullptr, float min_p = 0.0f, int* n_drafts = nullptr);

    /// Called with each draft as soon as it is known (j = 0.., its probability under the draft layer), while the
    /// chain goes on: the next window's PLE rows start there.
    std::function<void(int j, int32_t token, float prob)> on_draft;

<<<<<<< this fork
    // ---- --pipeline-windows 2: the chain as one asynchronous launch (the round and its steps back to back, no host
    // wait), its first `n_force` steps fed the given tokens instead of their own picks (teacher forcing: the drafts of
    // a window already in flight), each output read once the event after its step has completed.
    /// Before `load`: the round/step graphs get the forcing kernel (a no-op while its token is -1), and the drafter's
    /// stream the highest priority (STRATA_MTP_PRIORITY=0: the default priority).
    void set_force_capture(bool on) { force_on_ = on; }
    /// The rows the next round reads (null: the bound `window_R`), copied into the bound buffer before the round.
    void set_source_R(const float* rows) { src_R_ = rows; }
    /// Capture the round graphs for T = 1..max_t and every step graph now (a capture syncs the drafter's stream).
    bool prepare_chain(std::string& err);
    /// The catch-up over T cells from `p` (as draft()), then `n_out` outputs from row `a`: output j is the chain's
    /// own pick, and step j (1 <= j <= n_force) takes force[j - 1] as its input token instead of output j - 1.
    /// `n_early`: an event after each of the first `n_early` outputs (chain_outputs_ready).
    bool chain_launch(int T, const int32_t* tokens, int64_t p, int a, const int32_t* force, int n_force, int n_out,
                      int n_early, std::string& err);
    /// 1: every output is in chain_tok()/chain_prob(); 0: still running; -1: an error.
    int chain_poll(std::string& err);
    /// How many of the chain's first outputs (up to `n_early`; all of them once the chain is done) have landed in
    /// chain_tok()/chain_prob(); -1: an error.  A window's size is often decided by its first outputs (a draft below
    /// spec_min_p ends it), so the next launch need not wait for all of them.
    int chain_outputs_ready(std::string& err);
    bool chain_live() const { return chain_live_; }
    const int32_t* chain_tok() const { return chain_tok_; }
    const float* chain_prob() const { return chain_prob_; }

    double ms_draft = 0, ms_prefill = 0;
    double ms_src = 0;      ///< the host time spent staging the source rows (a cross-card copy when the drafter is on another card)
    int64_t n_src = 0;
=======
    double ms_draft = 0;
>>>>>>> their fork (github.com-eddoursul-Strata-custom)
    int64_t rounds = 0;

private:
    bool record_forward(int T, int step_row0, cudaStream_t cs, std::string& err);
    /// The layer's front for T rows at step rows [row0, +T): the embedding, the fc projections, the attention
    /// hyper-connection read (R_, inj_, mixed_) and the K/V appended.
    bool record_front(int T, int row0, cudaStream_t cs, std::string& err);
    /// The rest for row 0 at step row `step_row`: the attention from the q projection, the MLP, the final mixer,
    /// the head, the draft and its probability.  Needs row 0's R_, inj_ and mixed_ (its q8_1 rows in xq_).
    bool record_rest(int step_row, cudaStream_t cs, std::string& err);
    void norm_rope(float* data, const float* gamma, int rows, int cols, const int32_t* p, cudaStream_t cs);
    bool capture_round(int T, std::string& err);
    bool capture_step(int j, std::string& err);
    cudaGraphExec_t step_exec_[9] = {};
    const float* f32(const char* name) const;
    const uint16_t* bf16(const char* name) const;
    const void* q8(const char* name) const;

    const ModelGeometry* g_ = nullptr;
    SessionState* ss_ = nullptr;
    const WeightTable* wt_ = nullptr;
    const NativeHead* head_ = nullptr;
    const float* window_R_ = nullptr;
    int max_t_ = 0, max_drafts_ = 0;
    int64_t n_vocab_ = 0;
    uint64_t vram_ = 0;
    cudaStream_t cs_ = nullptr;
    cudaStream_t side_ = nullptr;                    // the shared expert's branch of the draft graphs
    cudaEvent_t sh_fork_ = nullptr, sh_join_ = nullptr;
    uint32_t* rseq_ = nullptr;                       // verify_router's sequence word and counter (device)
    unsigned* rcount_ = nullptr;
    cudaGraphExec_t round_exec_[9] = {};

    struct Tensor { std::string name, kind; int64_t rows = 0, cols = 0; uint64_t off = 0, bytes = 0; };
    std::vector<Tensor> tensors_;
    uint8_t* dense_ = nullptr;
    uint8_t* experts_ = nullptr;
    uint64_t blob_ = 0;                               // bytes per expert
    strata::kernels::NativeExpertLayout nat_;         // gu_type >= 0: GGUF blocks (experts.txt); else the Q2_0 blobs
    void* state_arena_ = nullptr;
    QsaState st_;
    void* arena_ = nullptr;

    // mapped staging: tokens, step records (2*max_t rows), positions per head (2*max_t rows), the selected row,
    // the drafts out
    int32_t *h_tok_ = nullptr, *m_tok_ = nullptr, *h_step_ = nullptr, *m_step_ = nullptr;
    int32_t *h_pos_ = nullptr, *m_pos_ = nullptr, *h_row_ = nullptr, *m_row_ = nullptr;
    int32_t *h_out_ = nullptr, *m_out_ = nullptr;
    float *h_prob_ = nullptr, *m_prob_ = nullptr;   // each draft's probability under the draft layer
    // the draft head: the main head's rows for a token subset (rt/draft_vocab.bin), or the whole head
    uint8_t* dhead_ = nullptr;
    int32_t* dvocab_ = nullptr;
    int64_t n_dvocab_ = 0;
    std::string rt_dir_;
    int64_t window_ = 0;        // attention over the last window_ cells (0 = every cell)
    int64_t prompt_len_ = 0;
    float* probs_ = nullptr;
    uint8_t* arg_scratch_ = nullptr;   ///< argmax_rows' and row_top_prob_split's partials and counters
    uint8_t* top_scratch_ = nullptr;
    // device
    int32_t *tok_ = nullptr, *step_ = nullptr, *pos_ = nullptr, *row_ = nullptr, *ident_ = nullptr;
    float *Rin_ = nullptr, *R_ = nullptr, *emb_ = nullptr, *en_ = nullptr, *e2_ = nullptr, *hn_ = nullptr, *h2_ = nullptr;
    float *mixed_ = nullptr, *inj_ = nullptr, *inj2_ = nullptr, *lo_ = nullptr, *rs_ = nullptr, *bo_ = nullptr;
    float* grs_ = nullptr;   // the hyper-connection reads' scratch (zeroed with the arena)
    uint8_t* xq_ = nullptr;
    float *qfull_ = nullptr, *qcur_ = nullptr, *kcur_ = nullptr, *vcur_ = nullptr, *attn32_ = nullptr;
    float* attn_scratch_ = nullptr;
    float *logits_ = nullptr, *w_ = nullptr, *shared_ = nullptr, *parts_ = nullptr, *y_ = nullptr, *sample_ = nullptr;
    int32_t *ids_ = nullptr, *hit_slot_ = nullptr, *hit_dst_ = nullptr, *hit_count_ = nullptr, *out_ids_ = nullptr;
    uint8_t *hit_xq_ = nullptr, *nat_xq_ = nullptr;
    unsigned long long* grp_ptr_ = nullptr;
    int32_t *grp_start_ = nullptr, *grp_counts_ = nullptr;
    float* hit_xs_ = nullptr;
    void* hit_scratch_ = nullptr;
    float* sh_scratch_ = nullptr;
    uint16_t* x_bf16_ = nullptr;
    float* head_logits_ = nullptr;
    float* dummy_inj_ = nullptr;
    int64_t cap_ = 0, attn_scratch_floats_ = 0;
};

}  // namespace strata::core
