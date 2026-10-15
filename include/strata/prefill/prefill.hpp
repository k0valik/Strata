// include/strata/prefill/prefill.hpp - plan v0.3 P5: batched prompt processing.
//
// The prompt's positions [pos0, pos0 + n) are processed in chunks through all 48 layers, leaving the session state
// (GDN recurrence and conv state, QSA KV pools and indexer, PLE history) where the token path would have left it; the
// decode loop then continues with the next token.  Per layer: the projections are tensor-core GEMMs (quantized
// weights dequantized to FP16 on the fly, BF16 weights as they are) and the recurrences walk the tokens inside one
// kernel, in sub-chunks of up to 2048 tokens whose buffers are reused; the routed experts then run over the whole
// chunk (strata/prefill/experts.hpp), so an expert the cache does not hold is streamed once per chunk.  Long chunks
// cost little VRAM besides the residual stream (~55 KB a token).
//
// Requires the native weights (`--native`): every quantized projection must carry its GGUF blocks.
#pragma once

#include "strata/core/expert_cache.hpp"
#include "strata/core/expert_source.hpp"
#include "strata/core/layer.hpp"
#include "strata/core/mtp.hpp"
#include "strata/core/session.hpp"
#include "strata/core/weights.hpp"
#include "strata/prefill/experts.hpp"

#include <cstdint>
#include <functional>
#include <memory>
#include <string>

namespace strata::prefill {

/// The sections `Prefill::profile` times, in the order a layer runs them.
enum PrefillSection {
    kPsPle, kPsHcRead, kPsGdn, kPsGdnConv, kPsGdnScan, kPsGdnNorm, kPsGdnOut, kPsQsaProj, kPsQsaIndexer, kPsQsaScores, kPsQsaAttn, kPsQsaOut, kPsHcFfn, kPsRouter,
    kPsExperts, kPsCombine, kPsDraft, kPsCount
};
const char* prefill_section_name(int section);

struct PrefillStats {
    int64_t tokens = 0;
    int64_t chunks = 0;
    double ms_total = 0;
    double ms_experts_host = 0;     ///< host time queuing this GPU's experts
    int64_t experts_streamed = 0;   ///< expert blobs copied host -> this GPU
    int64_t experts_prefetched = 0; ///< ...of which ahead of the routing
    int64_t experts_resident = 0;   ///< expert-layer groups served from its VRAM tier
    double ms_ple = 0;
    double ms_section[kPsCount] = {};   ///< GPU ms per section, with `Prefill::profile`
};

class Prefill {
public:
    Prefill();
    ~Prefill();
    Prefill(const Prefill&) = delete;
    Prefill& operator=(const Prefill&) = delete;

    /// For chunks of up to `max_chunk` tokens.  `host_res`: the static residency table (n_layers x n_expert, slot
    /// or -1) or null; `cache` its slots.  `offload`: a runner on a second GPU (initialized, bound by the caller)
    /// that computes the experts this GPU's cache does not hold; null: they stream here.
    bool init(const core::WeightTable& wt, const core::ModelGeometry& g, core::SessionState& ss,
<<<<<<< this fork
              core::ExpertSource* src, const core::ExpertCache* cache, const int32_t* host_res, int64_t chunk,
              void* stream, std::string& err, void* borrow = nullptr, uint64_t borrow_bytes = 0,
              bool borrow_spill = false);

    /// With borrowed buffers: lay them out again for chunks of `chunk` tokens (at most `init`'s) in `borrow` - a
    /// request lends only the slots its prompt needs.  The stream must be idle (between prompts).
    bool relayout(int64_t chunk, void* borrow, uint64_t borrow_bytes, std::string& err);
    int64_t chunk() const;

    /// The share of the streamed experts' bytes DMA-able straight from pinned RAM (1 = all).  Sizes the streamed
    /// ring (a big one only pays when the copy engine, not the host copies, is the limit); set before bytes_needed.
    static void set_pinned_share(double share);
    static double pinned_share();
    /// The chunk size from which a chunk streams every expert the GPU does not hold (1024; STRATA_PREFILL_STREAM_MIN).
    static int64_t stream_all_min_tokens();
    /// #340: the streamed ring's slot count for chunks that stream every expert, instead of the pinned-share rule
    /// (0 = that rule). Set before any `bytes_needed`/`init` (both count the ring); STRATA_PREFILL_RING still wins.
    static void set_ring_override(int slots);
    /// #583: the auto chunk scan's byte-budget ring, for chunks above `small_max` (0.1.39's auto chunk: a prompt that
    /// fits it keeps 0.1.39's ring).  0 slots = none.  A layer split's set_ring_override and STRATA_PREFILL_RING win.
    static void set_ring_budget(int slots, int64_t small_max);

    /// Device bytes `init` needs for a chunk of `chunk` tokens (what a borrowed region must hold).  `src`: init gets an
    /// ExpertSource (carve picks the fused MoE layout only then).
=======
              core::ExpertSource* src, const core::ExpertCache* cache, const int32_t* host_res, int64_t max_chunk,
              void* stream, ExpertRunner* offload, std::string& err);

    /// Device bytes of the buffers for chunks of `chunk` tokens.
>>>>>>> their fork (github.com-eddoursul-Strata-custom)
    static uint64_t bytes_needed(const core::ModelGeometry& g, const core::SessionState& ss, int64_t chunk,
                                 bool offload);
    /// The same with the area its experts are prefetched into (without a second GPU), for the residency as it is now.
    uint64_t bytes_for(int64_t chunk) const;
    /// The buffers for chunks of up to `chunk` tokens, carved from `region` (`bytes` long: lent expert-cache
    /// slots; what is left over takes prefetched experts), or with a null region allocated for the longest chunk
    /// (once).  Before the first chunk of every prompt that uses a region.
    bool bind(void* region, uint64_t bytes, int64_t chunk, std::string& err);

    /// Positions [pos0, pos0 + n) holding `tokens`, in chunks of the bound length; `ss.ple_prev` must be the two
    /// tokens before pos0 (oldest first, -1 for none) and is advanced to the last two of these.  With a `draft`
    /// layer tokens[n], the token after the last position, is read too.
    bool run(const int64_t* tokens, int64_t n, int64_t pos0, std::string& err);
    /// Positions the last `run` completed: n, or fewer after `should_stop` (the session state holds them).
    int64_t processed() const { return processed_; }

    const PrefillStats& stats() const { return stats_; }

    /// The MTP draft layer (or null): after every chunk, its K/V over the chunk's cells in the chunk's batched
    /// arithmetic (cell t pairs the chunk's final residual at t with the token at t + 1).
    core::MtpDrafter* draft = nullptr;

    /// Checked before every chunk: true stops the prompt there (`run` returns false with err "cancelled", and the
    /// state and `ss.ple_prev` hold the chunks before it, `processed()` positions).
    std::function<bool()> should_stop;

    /// The vision path: HOST rows (n_embd floats) indexed by absolute position, read in place of the token
    /// embedding where non-null (an image's <|image_pad|> cells).  Null (default): every position embeds its token.
    const float* const* embd_rows = nullptr;

    /// GPU time per section into `stats().ms_section` (events between the sections; the host waits for the last
    /// one after every chunk).
    bool profile = false;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
    PrefillStats stats_;
    int64_t processed_ = 0;
};

}  // namespace strata::prefill
