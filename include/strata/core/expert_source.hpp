// include/strata/core/expert_source.hpp - P2.S3/P2.S5: the expert pool's adapter to the host loop.
//
// `session_loop` publishes, per layer, the NORMED activation `x_f`, the ten routed expert ids and their router
// weights, and expects `k x n_embd` floats back.  This is the piece that turns those four things into an
// `ExpertPool::run` call.  Nothing here is clever, and that is the point: the pool, the kernel and the loop are
// each already verified, so this file has exactly one job - get the CONTRACT between them right.
//
// THE CONTRACT, and each clause is a way to be wrong:
//
//   1. **`x_f` IS THE SAME PINNED BUFFER EVERY LAYER.**  Its ADDRESS never changes, so the `ActQ` cannot be
//      cached by pointer - a cache keyed on `x_f` would quantize layer 0 and reuse it for all 47 remaining
//      layers, which is a finite, plausible, completely wrong token.  There is no cache here at all: the
//      conversion is one 2560-element pass against a 0.3 ms/layer budget, and a correct answer is worth more
//      than the microseconds.
//   2. **THE POOL DOES NOT APPLY THE ROUTER WEIGHT.**  `moe_combine` (`src/core/layer.cpp:479`) sums
//      `w[i] * parts[i]` on the device.  `ExpertJob::weight` is a diagnostic field; setting it here would
//      apply the weight twice, which is invisible in a single layer and compounds over 48.
//   3. **THE KERNEL'S GEOMETRY IS FIXED BY THE ARTIFACT** (`H = 2560`, `FF = 640`, `BLOB = 1,382,400`).  A
//      different `n_embd` or a different expert width is REFUSED rather than mis-indexed, because the blob's
//      internal offsets are compile-time constants and reading a 640-wide expert as a 2560-wide one walks off
//      the end into the next expert's bytes without faulting.
#pragma once

#include "strata/core/expert_cache.hpp"
<<<<<<< this fork
#include "strata/core/pinned.hpp"
#include "strata/core/exchange_storage.hpp"
=======
>>>>>>> their fork (github.com-eddoursul-Strata-custom)
#include "strata/core/hit_hook.hpp"
#include "strata/kernels/cpu/pool.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace strata::core {

class SecondGpu;

/// Where one routed expert's bytes come from.
///
/// Phase 2 has NO cache (`phase-2-correct-engine.md`: hit rate `h = 0`), so the only implementation is a
/// file-backed reader.  The interface exists anyway because Phase 3 replaces exactly this object with the VRAM
/// cache, and because a test can supply an in-memory source without a 34 GB artifact.
class ExpertSource {
public:
    virtual ~ExpertSource() = default;

    /// The 1,382,400-byte blob for `(layer, expert)`, or nullptr if it cannot be produced.
    ///
    /// The pointer only has to stay valid until the next `blob()` call: with `h = 0` every expert is computed
    /// immediately and nothing is retained.  A CACHING source must return pointers into the cache, not into a
    /// reused staging buffer - otherwise the pool would read the next expert's bytes while computing this one.
    virtual const uint8_t* blob(int64_t layer, int64_t expert) = 0;

    /// Blobs touched, for the driver to report.  A source that does not count returns 0.
    virtual int64_t reads() const { return 0; }

    /// Plan v0.3 P5: whether `blob(layer, expert)` lies in page-locked, CUDA-registered memory, so an
    /// asynchronous host-to-device copy can DMA it directly (no staging copy on the CPU).
    virtual bool pinned(int64_t layer, int64_t expert) const { (void) layer; (void) expert; return false; }

    /// Called once before the first expert of a layer.  A source that reads from disk wants to start the read
    /// here so it overlaps the quantisation, and a prefetching source in Phase 3 wants the ids.
    virtual void begin_layer(int64_t layer, const int32_t* ids, int64_t k) { (void) layer; (void) ids; (void) k; }
    /// Plan v0.3 P6: the DEVICE address of a pinned, mapped blob (the GPU can read it over PCIe), or null.
    virtual const uint8_t* device_alias(int64_t layer, int64_t expert) const { (void) layer; (void) expert; return nullptr; }
    /// Whether `layer`'s blobs in RAM have device aliases (the PCIe share can read them).  Not device_alias(layer, 0):
    /// that expert may live only in VRAM.
    virtual bool layer_mapped(int64_t layer) const { return device_alias(layer, 0) != nullptr; }

    /// Whether `blob(layer, expert)` exists.  False for an expert that lives only in VRAM (--vram-pin-gib): a GPU
    /// holds it for good and computes it, so no path may ask the RAM for it.
    virtual bool in_ram(int64_t layer, int64_t expert) const { (void) layer; (void) expert; return true; }
};

/// Plan v0.3 P6: the PCIe share of a verify window's layer - missed experts the main GPU reads over PCIe - written by
/// the pool (mapped host memory) right after the ring and published before the CPU starts its own share.  (The
/// experts in its VRAM tier the window decides itself, from the residency the pool reads too.)  Groups of entries
/// that share a blob: `ptr2[q]` is staging slot q (DMA) or the arena's device alias, its entries dst/tok
/// [start2[q], start2[q + 1]).  counts[1] = entries, counts[2] = groups.
struct GpuPlanSink {
    int32_t* counts = nullptr;
    int32_t* dst = nullptr;                ///< cap: the entry's row of parts (token * k + j)
    int32_t* tok = nullptr;                ///< cap: the entry's token
    unsigned long long* ptr2 = nullptr;    ///< cap
    int32_t* start2 = nullptr;             ///< cap + 1
    unsigned long long staging = 0;
    int64_t staging_cap = 0;
    int64_t cap = 0;
    /// Whether the window takes a PCIe share (`--pcie-frac` > 0): the pool publishes one every layer, else none.
    bool pcie = false;
    void (*publish)(void* ctx) = nullptr;
    /// Starts the DMA copies of `n` host blobs (pinned) into staging slots 0..n-1 and signals the GPU when they land.
    void (*fetch)(void* ctx, const uint8_t* const* src, int n, size_t bytes) = nullptr;
    void* ctx = nullptr;
    /// Plan v0.3 P6: how a PCIe group reaches the GPU.  0 = the copy engine stages it (DMA, `fetch`); 1 = the grouped
    /// kernel reads the mapped arena directly; 2 = a copy kernel stages it inside the graph.  For 1 and 2 `ptr2`
    /// holds the arena's device alias.
    int pcie_mode = 0;
    /// The GPU reads only the rows it did not compute, so the pool leaves the GPU's rows of `out` unwritten.
    bool host_rows_only = false;
    /// The second GPU's share of this token group (mapped; null without one): its entries, [n, pad x3, entries], and
    /// the flag its graph raises to `ring` once their rows are in `out`, which the main GPU waits for.  The pool
    /// raises the flag itself when the second GPU takes no share, and sets `gpu2_ring` to `ring` once the flag is
    /// raised or a share is submitted.
    int32_t* gpu2_list = nullptr;
    uint32_t* gpu2_flag = nullptr;
    uint32_t ring = 0, gpu2_ring = 0;
    /// This token group's activations as the window's router wrote them (mapped): ggml's Q8_K rows, `xk_stride` bytes
    /// apart, when the layer takes them (`pool_takes_q8k`), else null (the pool quantizes the floats); and the q8_1
    /// rows (n_embd / 32 blocks each) the second GPU takes.
    const uint8_t* xk = nullptr;
    int64_t xk_stride = 0;
    const uint8_t* x1 = nullptr;
};

/// Whether the pool takes a native layer's gate/up activation as ggml's Q8_K rows (`GpuPlanSink::xk`): its format's
/// vec_dot_type (the i-quants', the K-quants'), and not the Q2_0 kernels' own.
inline bool pool_takes_q8k(const strata::kernels::cpu::NativeFmt& f) { return f.gu_type != 42 && f.gu_act == 15; }

/// The adapter's own state.  One per session, reused every layer so the token path allocates nothing (P2.T10).
struct ExpertDispatch {
    strata::kernels::cpu::ExpertPool* pool = nullptr;
    ExpertSource* src = nullptr;
    int64_t n_expert = strata::kernels::cpu::NE;

    /// Counters, for the driver to report rather than for control flow.
    int64_t layers = 0;
    int64_t experts = 0;
    int64_t missing = 0;

    // ================================ R4: THE VRAM TIER, MEASURED BEFORE IT IS USED ================================
    //
    // **THE CACHE IS CONSULTED AND FILLED HERE, AND NOTHING IS COMPUTED FROM IT YET.**  That is deliberate and
    // it is `R4-design-note.md` §7 step 2: the *dispatch* is measured on its own before any kernel is written,
    // because a hit rate measured after the kernel exists cannot say whether a disappointing result was the
    // policy, the split or the kernel.
    //
    // So every expert is still computed by the CPU, exactly as before, and the run is numerically identical
    // with the cache on or off.  What changes is that the engine now reports **h on its own routing, on a real
    // workload** - which is the number `R4.1` asks for and which until now existed only from offline traces.
    //
    // The policy is compulsory-miss: the first time `(layer, expert)` is routed, if a slot is free it is
    // admitted and filled from the arena.  No eviction, because eviction policy is the measured question
    // (`R4.1`'s LFU-decay vs LRU sweep) and a placeholder would set the hit rate everything is sized against.
    ExpertCache* cache = nullptr;
    int64_t cache_hits = 0;      ///< lookups already resident
    int64_t cache_admitted = 0;  ///< lookups that took a slot
    int64_t cache_refused = 0;   ///< lookups with no slot free (the cache is full)
    void* cache_stream = nullptr;
    const char* cache_fail = nullptr;

    // ================================ R4.2c: THE HITS GO TO THE GPU ================================
    //
    // **THE SPLIT IS BY ROUTER INDEX, AND THAT IS THE WHOLE TRICK.**  A layer routes ten experts; the resident
    // ones are computed on the GPU and the rest on the CPU, and both answers have to end up in `parts` at the
    // index the ROUTER gave them, because that is the order `moe_combine` weights against.  So the CPU's `out`
    // rows are zeroed for the hits, and each hit carries its routed index to the kernel as `dst`.
    //
    // Everything below is per-session state rather than per-call, so the token path allocates nothing (P2.T10).
    const uint8_t* cache_base = nullptr;   ///< the slot arena on the DEVICE
    int64_t cache_blob = 0;                ///< bytes per slot
    const uint64_t* cache_slot_off = nullptr;   ///< plan v0.3 P6: per-slot offsets when the slots differ in size
    void* hit_scratch = nullptr;           ///< `moe_hit_grouped_scratch_bytes(K, ...)`
    float* parts_out = nullptr;            ///< the graph's `parts` buffer, on the device
    /// Where the GPU's hits land, `K x n_embd`, DEVICE and separate from `parts_out` on purpose: see
    /// `HitPhase`.  Zeroed by the hit path each layer before the kernel writes it.
    float* hit_out = nullptr;
    int64_t parts_elems = 0;               ///< `K * n_embd`, the length of both buffers
    const float* mixed = nullptr;          ///< the layer's normed activation, for the hit kernel's Q8_0
    uint8_t* x_q8_0_hit = nullptr;         ///< `(n_embd/32) * 34` bytes, its own buffer
    /// **R4.2h: THE CPU's fp32 ACTIVATION SCALES, `(n_embd/32)` FLOATS.**  Without them the GPU's hits are
    /// computed with the `block_q8_0`'s fp16 `d` while the CPU's misses use the fp32 `ActQ::scale`
    /// (`cpu/expert.cpp:92`) - **4.761e-04 relative on 80 of 80 chunks**, measured with both real
    /// implementations linked in `bench/micro/act_quant_parity.cu`.  That disagreement is why turning the
    /// cache on changed the generated tokens.  Required whenever the hit path runs.
    float* x_q8_0_hit_scale = nullptr;
    int32_t* d_slot = nullptr;             ///< device, K entries
    int32_t* d_dst = nullptr;              ///< device, K entries
    std::vector<int32_t> h_slot, h_dst;    ///< host staging, sized at session setup
    /// **PER ROUTER INDEX, DECIDED IN `Launch` AND CONSUMED BY THE POOL.**  The two callbacks share it
    /// so the decision is made exactly once, on this layer's ids, and neither side can re-decide it.
    std::vector<uint8_t> is_hit;
    bool decided = false;
    /// Plan v0.3 P4 token graph: the residency table (`n_layers x n_expert`, slot or -1), the host's copy of what the
    /// device hit path reads (in verify windows `Verifier::residency`).  When set, the pool leaves a resident
    /// expert's row to the GPU without any `Launch` callback.
    const int32_t* host_res = nullptr;
    /// Plan v0.3 P4: split every expert by rows across the pool's threads (default on; A/B `--no-split-rows`).
    bool split_rows = true;

    // ================================ R4.2d: DID THE GPU ACTUALLY START? ================================
    //
    // **THE HIT PATH IS 0.91 ms SLOWER AND THE ONLY EXPLANATION LEFT IS THAT IT NEVER OVERLAPS.**  The kernel
    // is 159 GB/s against the CPU's 35, the CPU's half of the drain falls 6 ms, and none of it reaches the
    // token - which is what it would look like if the enqueued hit kernels did not BEGIN until the host next
    // entered the driver, i.e. after `pool()` returned.  That is the third sighting of this driver behaving
    // that way (rounds 195/287 on the doorbell, round 36 on the head and sampler) and it decides whether R4 can
    // pay at all, so it gets measured rather than argued.
    //
    // `hit_done` is recorded on the stream straight after the hit kernel.  `Combine` - which runs after the
    // pool - queries it: SUCCESS means the GPU finished while the CPU was working, NOT-READY means it had not.
    // Same shape as the doorbell's `rings_mid_graph`, and for the same reason.
    void* hit_done = nullptr;      ///< cudaEvent_t, created at session setup
    int64_t hit_ready = 0;         ///< layers where the hit work was DONE by the time the pool returned
    int64_t hit_late = 0;          ///< layers where it was not
    /// The candidate fix, as an A/B arm: enter the driver once right after the hit launch.  If submission is
    /// lazy, this starts the GPU work before the pool instead of after it.
    bool hit_poke = false;
    bool hit_cpu_order = false;    ///< experimental CPU-order GPU expert arithmetic; opt-in only
    int64_t n_hits = 0;                    ///< this layer's hits
    /// Set by `Launch` and consumed by `Combine`, so a `Combine` with no `Launch` in front of it cannot
    /// add a stale buffer into `parts`.
    bool hit_pending = false;
    const char* hit_fail = nullptr;

    /// Whether the hit path is wired up.  All of it or none of it: a half-configured hit path would compute
    /// some experts twice and others not at all, which is a wrong token rather than an error.
    bool hits_ready() const {
        return cache != nullptr && cache_base != nullptr && parts_out != nullptr && hit_out != nullptr &&
               mixed != nullptr &&
               hit_scratch != nullptr && x_q8_0_hit != nullptr && x_q8_0_hit_scale != nullptr && d_slot != nullptr && d_dst != nullptr;
    }

    std::vector<strata::kernels::cpu::ExpertJob> jobs;
    strata::kernels::cpu::ActQ act;
    /// Plan v0.3 P6 verify window: one quantized activation per token, the multi-token jobs, and the expert ->
    /// job map (reset after every layer).
    std::vector<strata::kernels::cpu::ActQ> act_multi;
    /// Plan v0.3 P6: a native pack's per-token activations (MAXT x kNativeActBytes).
    std::vector<uint8_t> nact_multi;
    std::vector<strata::kernels::cpu::ExpertJobMulti> jobs_multi;
    std::vector<int16_t> job_of;
    /// Plan v0.3 P6: the verify window's PCIe share; `pcie_num`/256 of each layer's distinct missed experts (the last
    /// ones in routing order) are read by the GPU over PCIe.
    GpuPlanSink* plan = nullptr;
    int pcie_num = 0;
    int64_t pcie_experts = 0;      ///< distinct experts the GPU read over PCIe in verify windows
    double ms_plan = 0, ms_actq = 0, ms_jobs = 0, ms_run = 0;   ///< verify-window dispatch sections
    /// Plan v0.3 P6: decayed routing counts per (layer, expert) during decode (sized by the caller; empty = off),
    /// which the driver uses to swap the most-routed missing experts into the VRAM tier between rounds.
    std::vector<float> usage;
    int64_t multi_misses = 0;      ///< distinct (layer, expert) pairs the CPU computed in verify windows
    int64_t multi_entries = 0;     ///< routed (token, expert) entries the CPU served in verify windows
    /// Routed (token, expert) entries per (layer, expert) in verify windows, when the caller sizes it: how many of
    /// the entries a cache of the N most-routed experts would serve.
    std::vector<uint32_t> routed;
    /// A second GPU's expert tier (verify windows) and its residency table (n_layers x n_expert, slot or -1): the
    /// experts it holds or has prefetched and the first GPU's cache does not are computed there, beside the CPU pool.
    SecondGpu* gpu2 = nullptr;
    const int32_t* host_res2 = nullptr;
    int64_t gpu2_entries = 0;      ///< routed entries the second GPU served
    int64_t gpu2_skipped = 0;      ///< layers the CPU took the second GPU's share of (it was quicker)
    bool gpu2_used = false;        ///< this layer's decision
    uint64_t gpu2_min_bytes = 0;   ///< a layer's miss bytes from which the second GPU takes its share
    uint64_t gpu2_prefetch_bytes = 0;   ///< per layer, its likeliest experts no GPU holds, copied there ahead
    std::string gpu2_err;          ///< `fail` points here when the second GPU failed
    /// Set when `dispatch` could not produce an answer.  The loop itself has no error channel, so this is
    /// where a source failure surfaces: the driver checks it after `session_loop` returns rather than the
    /// engine computing from a half-filled `parts`.
    ///
    /// **`failed` LATCHES AND `fail` IS ONLY THE MESSAGE.**  Clearing the message must not re-arm the adapter:
    /// a session that failed at layer 5 has already fed `moe_combine` whatever `parts` held, so every layer
    /// after it is built on a hole - and resuming into a plausible-looking token is the exact outcome this
    /// whole mechanism exists to prevent.  `failed` is what the short-circuit reads.
    bool failed = false;
    const char* fail = nullptr;
    int64_t fail_layer = -1;
    int64_t fail_expert = -1;
};

/// `strata::core::PoolFn`, exactly.  Silent on failure BY SIGNATURE - see `ExpertDispatch::fail`.
void expert_pool_dispatch(void* user, const float* x_f, const int32_t* ids, const float* weights, int64_t n_embd,
                          int64_t k, float* out);

/// Plan v0.3 P6: the pool for a verify window of `n_tok` tokens.  `x_f` is (n_tok, n_embd), `ids` (n_tok, k) and
/// `out` (n_tok * k, n_embd).  Each distinct missed expert is computed once for all the tokens routed to it;
/// resident experts' rows are the main GPU's (zeroed, or left unwritten with `GpuPlanSink::host_rows_only`), and the
/// second GPU writes its share's rows itself, after this returns (`GpuPlanSink::gpu2_flag`).
void expert_pool_dispatch_multi(ExpertDispatch& d, const float* x_f, const int32_t* ids, int64_t n_tok, int64_t k,
                                float* out);
/// Copies `layer`'s likeliest experts that neither GPU holds to the second GPU's prefetch slots, where that layer's
/// dispatch finds them; `ids`/`w` (n_tok, k) are the layer's predicted routing (`Verifier::set_predict`).
void expert_prefetch_multi(ExpertDispatch& d, int64_t layer, const int32_t* ids, const float* w, int64_t n_tok,
                           int64_t k);

/// **THE HITS, LAUNCHED AFTER THE MISSES ARE STAGED AND BEFORE `post[l]`.**  Same shape as `PoolFn` and for the
/// same reason: `session_loop` owns the ORDER and this owns the work, so the loop needs to know nothing about
/// expert caches.  Returns immediately when there are no hits, which is every layer until the cache is warm.
///
/// It must run AFTER the host has copied the misses into `parts_dev` (it writes into the same buffer, on rows
/// the CPU zeroed) and BEFORE `post[l]` (which reads it).  Both are stream-ordered on the loop's own stream.
void expert_hit_run(void* user, void* stream, HitPhase phase, const int32_t* ids, int64_t k);
/// The pool half of the same decision; see `ExpertDispatch::is_hit`.

/// **PHASE 2'S ONLY SOURCE: `experts.bin`, memory-mapped, no cache.**
///
/// `experts.bin` is 33,973,862,400 B and `BLOB` is 1,382,400, so it holds exactly `48 x 512 = 24,576` blobs and
/// the index is `layer * 512 + expert` with NO padding.  A mapping is therefore the whole implementation: the
/// blob pointer is base plus a multiply, and the page fault that follows is the read.
///
/// **AND THAT IS NOT AS SLOW AS IT SOUNDS, WHICH IS THE POINT.**  The expert set is 34 GB and this machine has
/// 64 GB of DDR5, so a warm OS page cache holds ALL of it - the second token onward is a DRAM read at the
/// measured 44.14 GB/s, not a disk read.  The first token's 663.6 MB comes off the disk and is the cold-path
/// number; `benches` should report the two separately rather than blending them.
///
/// IT IS NOT AN LRU, AN LFU OR A PREFETCHER.  Phase 3 replaces this object with those.  Phase 2 is hit rate
/// `h = 0` on purpose, so that the CPU path and the host round trip are exercised on every layer of every
/// token - which is the only way they get debugged before speed matters.
class FileExpertSource : public ExpertSource {
public:
    FileExpertSource() = default;
    ~FileExpertSource() override;
    FileExpertSource(const FileExpertSource&) = delete;
    FileExpertSource& operator=(const FileExpertSource&) = delete;

    /// Maps `<pack_dir>/experts.bin` and checks its size against the loaded `expert_layout().total` - the
    /// canonical `n_layers * n_expert * BLOB` for a Q2_0 pack, or the sum of a native pack's per-layer blobs.
    ///
    /// The size check is not a formality: a short file would fault at the END of a long sequence, and an
    /// over-long one means the pack is not the one the geometry came from.  Refuses with the two numbers.  The
    /// blob offsets come from the same layout, so a native (IQ4 / mixed K-quant) pack reads its real per-layer
    /// slices rather than a fixed `BLOB` stride.
    bool open(const std::string& pack_dir, int64_t n_layers, int64_t n_expert, std::string& err);
    void close();

    bool mapped() const { return base_ != nullptr; }
    int64_t blobs() const { return blobs_; }
<<<<<<< this fork
    uint64_t pinned_bytes() const { return complement_pinned_ ? complement_pin_limit_ : 0; }
    uint64_t resident_bytes() const { return complement_bytes_; }
    bool complement_pinned() const { return complement_pinned_; }
    bool complement_ready() const { return complement_ready_; }
    uint64_t locked_bytes() const { return complement_locked_; }
    /// Lend-region slots whose experts the compact copy holds (the last ones of the cache).
    int64_t resident_lent_slots() const { return complement_lent_slots_; }

    // ---- the resident RAM mode and the adaptive tier.  A swap puts `in` (held here) into a GPU slot and evicts
    // `out` (held only by that slot).  Before the slot is overwritten the caller copies it back into an exchange
    // buffer and calls `stage_exchange`: `out` is then read from that buffer, and `in` still from here (the CPU
    // computes both until the swap lands).  Once the slot copy has landed, `commit_exchanges` moves `out` into
    // `in`'s place, so the copy keeps holding exactly the experts the GPU does not - with no read of the file.
    // STRATA_EXCHANGE_ROTATE=1 can instead transfer buffer ownership (uniform, fully pinned/mapped slots).
    /// Whether the compact copy holds `(layer, expert)`.
    bool has_resident(int64_t layer, int64_t expert) const;
    /// Host room for `n` evicted blobs (page-locked when possible).  Idempotent for the same or a smaller `n`.
    bool reserve_exchanges(int64_t n, std::string& err);
    int64_t exchange_capacity() const { return xstage_cap_; }
    /// The exchange buffers are page-locked (cudaHostAlloc): copies to and from them are asynchronous.
    bool exchange_pinned() const { return xstage_pinned_; }
    uint8_t* exchange_buffer(int64_t q) const;
    /// Requires `has_resident(layer, in)`, `!has_resident(layer, out)` and `exchange_buffer(q)` holding out's blob.
    bool stage_exchange(int64_t layer, int64_t in, int64_t out, int64_t q);
    /// --pipeline-windows: size the exchange table now, so a `stage_exchange` on the adaptive tier's thread never
    /// reallocates it under a concurrent `blob` (the pool reads it while windows are in flight).
    void prepare_overrides() { if (override_.empty()) override_.assign((size_t) blobs_, nullptr); }
    /// After the GPU copies of every staged swap have landed.  Returns how many exchanges were applied.
    int64_t commit_exchanges();
    /// 0.1.43 (STRATA_ADAPT_OVERLAP): BEFORE the GPU copies of the staged swaps have landed, make each staged `out` look to the
    /// planner as it will after `commit_exchanges` (pinned, with a device alias into its exchange buffer, so the PCIe share
    /// treats it exactly as a committed one) without touching `in`'s place, which the copy still reads.  `commit_exchanges`
    /// (after the copies have landed) ends it.  False: not possible here (rotation, exchange buffers not page-locked): the
    /// caller waits for the copies instead.
    bool soft_commit_exchanges();
    /// `commit_exchanges` in two halves, for the asynchronous adaptive tier (--adapt-async): `commit_copies` moves
    /// every staged evicted blob into its `in`'s place in the copy (on another thread: safe once nothing computes `in`
    /// from RAM - it is resident on the GPU - while `out` is still read from its exchange buffer), then `commit_flip`
    /// (the caller's thread, between windows) points `out` there and drops the staging.  Returns how many were applied.
    void commit_copies();
    int64_t commit_flip();
    /// #2134: the Foresight swap space (STRATA_FS_SLOTS), null = none.  Each commit above holds the staged experts in it
    /// (ForesightSwap::hold: their slots are dropped and not refilled) while their bytes move, so no slot is served
    /// with bytes copied from a place an exchange rewrote.
    void set_foresight_swap(ForesightSwap* fs) { fs_swap_.store(fs, std::memory_order_release); }
    /// The compact copy's blob of `(layer, expert)`, or null; not counted as a read (any thread).
    const uint8_t* resident_blob(int64_t layer, int64_t expert) const;
    int64_t exchanges() const { return exchanges_; }
    bool exchange_rotation() const { return exchange_storage_.active(); }
    uint64_t rotated_exchanges() const { return exchange_storage_.exchanges(); }
    uint64_t avoided_exchange_copy_bytes() const { return exchange_storage_.avoided_bytes(); }
    /// With the compact copy ready: blobs read from the mapped file since (what the plain mmap mode may read from
    /// the SSD).  0 in a steady resident mode; lend-region experts that did not fit the RAM count here.
    int64_t file_reads() const { return file_reads_.load(std::memory_order_relaxed); }
    /// CS-T per-tier counters: blobs served from the RAM copy, and the bytes read from the mapped files (the blobs
    /// `file_reads` counts, plus the prompt path's copies of them).
    int64_t ram_reads() const { return ram_reads_.load(std::memory_order_relaxed); }
    uint64_t file_read_bytes() const { return file_read_bytes_.load(std::memory_order_relaxed); }
    /// Of those, the bytes `blob` read (decode windows, adaptive swaps; the rest are the prompt path's copies), and
    /// the time spent reading the files, summed over threads.
    uint64_t file_blob_bytes() const { return file_blob_bytes_.load(std::memory_order_relaxed); }
    double file_ms() const { return (double) file_us_.load(std::memory_order_relaxed) / 1000.0; }
    /// Threads `prefetch` reads the GGUF with (STRATA_FETCH_THREADS, default 8).
    void set_fetch_threads(int n) { fetch_threads_ = n < 1 ? 1 : n; }
    /// #286: read the experts straight from the drive (Windows: FILE_FLAG_NO_BUFFERING, overlapped; Linux: O_DIRECT and
    /// the kernel's asynchronous reads) instead of
    /// through the mapped files - the GGUF in place, or a pack's experts.bin - when the file cache could not keep them
    /// beside `ram_bytes` (the RAM budget), cached now or not (their mapped pages would land in the working set); see
    /// experts_unbuffered.  The mapped reads' page faults are one small request each, and the pages they bring in
    /// take the RAM the budget was sized for.  `why` says what decided.
    /// #577: `ram_bytes` counts up to every expert, and only the experts outside it are what the cache must keep.
    bool set_unbuffered(uint64_t ram_bytes, std::string& why);
    /// #577: the same decision once the RAM copy is built (pin_cache_complement), from the RAM it really holds and the
    /// expert bytes outside it; switches either way (startup only, nothing reading).  Returns whether unbuffered.
    bool recheck_unbuffered(std::string& why);
    bool unbuffered() const { return !direct_.empty(); }
    /// MODEL-UPDATE round 2 (Linux, opt-in STRATA_PROMPT_DIRECT=1): with the file tier read through the file cache (the
    /// decode's choice: re-routed experts come back from it), the prompt path's batched stager reads (copy_blobs) go
    /// unbuffered instead - the same bytes, read in merged O_DIRECT requests.  False when not opened (and why).
    bool open_prompt_direct(std::string& why);
    /// pp-opt (Windows, experts.bin, unbuffered): close the mapped view once startup no longer needs it.  NTFS runs
    /// the unbuffered reads of a mapped file one at a time - on a PCIe 4 NVMe (WD SN580) 2 MiB reads at queue depth
    /// 32 measured 2.35 GB/s with experts.bin mapped and 3.57 GB/s without.  After this every file read goes through
    /// read_direct (mapped_blob answers nullptr).  Returns whether the view was closed; `why` says why not.
    bool drop_mapping(std::string& why);
    /// Windows, experts.bin, a file tier that reads through the file cache (--resident-experts): the startup reads -
    /// the GPU cache's fill from the profile and the RAM copy (pin_cache_complement) - go unbuffered (read_direct)
    /// with the view closed meanwhile.  Through the mapping they put the file's pages in the working set beside the
    /// RAM copy, and on a PC short of RAM that pages other memory out.  The bytes and their places are the same.
    /// end_startup_unbuffered maps the view again and closes the unbuffered handles and the stage buffers, as before
    /// the startup.  STRATA_STARTUP_UNBUFFERED=0: the mapped copy (A/B).  False (and `why`) when it does not apply.
    bool begin_startup_unbuffered(std::string& why);
    /// False when the view could not be mapped again: then the file tier stays unbuffered, `why` says so.
    bool end_startup_unbuffered(std::string& why);
    bool startup_unbuffered() const { return startup_ub_; }
    /// Every expert's bytes (n_layers x n_expert blobs).
    uint64_t expert_bytes() const;
    /// #286, unbuffered: assembles the blobs of these pairs ahead of the `blob` calls that will ask for them (the
    /// GPU cache's fill from the profile) - one batch of reads instead of one blob at a time.  At most 64 pairs.
    void prefetch_pairs(const std::pair<int32_t, int32_t>* pairs, int64_t n);
    /// Mapped reads: asks the OS for these pairs' blobs ahead (platform::advise_willneed).  False when unbuffered,
    /// when read-ahead is off, or before `open`.
    bool advise_pairs(const std::pair<int32_t, int32_t>* pairs, int64_t n) const override;
=======
>>>>>>> their fork (github.com-eddoursul-Strata-custom)

    const uint8_t* blob(int64_t layer, int64_t expert) override;

    /// Blobs touched, for the driver to report.  With `h = 0` this is `48 * k` per token and the number is only
    /// interesting once Phase 3 makes it not so.
    int64_t reads() const override { return reads_; }

private:
<<<<<<< this fork
    const uint8_t* resident_blob(size_t index) const;
    const uint8_t* mapped_blob(int64_t layer, int64_t expert) const;
    /// The blob's bytes from the mapped file(s) - experts.bin, or the three GGUF role slices - into `dst`.
    bool copy_from_files(int64_t layer, int64_t expert, uint8_t* dst) const;
    bool open_gguf(std::string& err);
    const uint8_t* staged_blob(int64_t layer, int64_t expert);
    bool claim_stage(int64_t key, size_t& v, bool& fill, bool ahead = false);
    bool fill_stage(size_t v, int64_t layer, int64_t expert, uint8_t* dst);
    void publish_stage(size_t v, int64_t layer, bool ok, double us);
    struct Fill { size_t v; int64_t layer, e; uint8_t* dst; };
    /// The claimed buffers' blobs: one overlapped batch when unbuffered, else the fetch threads' mapped copies.
    void fill_many(const std::vector<Fill>& todo);
    /// #286: the blobs from the drive, unbuffered: every role's 4 KiB-aligned window is read at once (overlapped on
    /// Windows, io_submit on Linux) into this thread's aligned buffer, then copied into place.  False when a read fails.
    bool read_direct(const Fill* fills, size_t n) const;
    bool open_direct(std::string& why);
    /// Windows: experts.bin mapped again as `open` maps it (begin/end_startup_unbuffered).
    bool map_view(std::string& why);
    std::vector<std::string> paths_;          ///< the mapped files, as maps_
    std::vector<void*> pdirect_;              ///< open_prompt_direct: O_DIRECT fds for copy_blobs only
    std::vector<void*> direct_;               ///< #286: per file, an unbuffered handle (Windows) or O_DIRECT fd (Linux)
    std::vector<int> role_file_;              ///< 3 x n_layers: index into maps_ / direct_
    /// blobs assembled in the stage buffers (`blob` hands those out): the GGUF in place, or any unbuffered source
    bool staged() const { return !role_ptr_.empty() || !direct_.empty() || io_pf_; }
    static constexpr uint64_t kNoComplement = detail::kNoCacheComplement;
    // ---- CS-T: the GGUF shards in place
    std::string gguf_;
    struct Map {
        const uint8_t* base = nullptr;
        uint64_t bytes = 0;
        bool closed = false;   ///< drop_mapping unmapped the view (base stays for offsets)
#if defined(_WIN32)
        void* file = nullptr;
        void* mapping = nullptr;
#else
        int fd = -1;
#endif
    };
    std::vector<Map> maps_;
    std::vector<const uint8_t*> role_ptr_;    ///< 3 x n_layers: gate / up / down of the layer's expert 0
    std::vector<uint64_t> role_bytes_;        ///< 3 x n_layers: bytes per expert of that role
    // the blobs assembled for `blob()`: a small pool of buffers, one per recent (layer, expert).  A buffer is
    // reused only once `kStageAge` layer changes have passed since its blob was last asked for, so a pointer holds
    // through the layer it was asked in and the next ones (the pool computes a layer's misses before the next).
    static constexpr uint64_t kStageAge = 3;
    // A stage buffer feeds cudaMemcpyAsync (the cache fill), so it is cudaHostAlloc'd where the driver allows:
    // a pageable source would be staged through the driver's bounce buffer - an extra copy at a fraction of
    // the transfer rate, with the calling thread doing it.  `pinned` picks the free (defined in the .cpp: the
    // header has no cuda_runtime.h).
    struct StageBufFree {
        bool pinned = false;
        void operator()(uint8_t* p) const noexcept;
    };
    std::mutex stage_mu_;
    std::vector<std::unique_ptr<uint8_t[], StageBufFree>> stage_buf_;
    std::vector<int64_t> stage_key_;
    std::vector<uint64_t> stage_epoch_, stage_used_;
    std::vector<char> stage_busy_;            ///< being filled (outside stage_mu_): never a victim
    std::condition_variable stage_cv_;
    int fetch_threads_ = 8;
    std::atomic<uint64_t> file_blob_bytes_{0}, file_us_{0};
    std::unique_ptr<std::atomic<uint32_t>[]> warm_stamp_;   ///< per (layer, expert): epoch_ + 1 when warmed
    std::atomic<int64_t> warm_hits_{0}, warm_count_{0};
    mutable std::atomic<int64_t> direct_fallbacks_{0};
    std::unordered_map<int64_t, size_t> stage_of_;
    uint64_t stage_blob_ = 0;
    uint64_t stage_seq_ = 0;
    uint64_t epoch_ = 0;
    int64_t last_layer_ = -1;
    bool stage_grew_ = false;
    bool stage_pin_said_ = false, stage_pin_failed_said_ = false;   ///< the one-time notes in claim_stage
    // #1237: per stage buffer, the event recorded after the last asynchronous copy out of it (note_async_read); a
    // buffer is refilled only once that event has completed.  cudaEvent_t, stored untyped (no cuda_runtime.h here).
    std::vector<void*> stage_ev_;
    std::vector<char> stage_ev_live_;                 ///< a copy was queued since the buffer was last waited for
    std::unordered_map<const uint8_t*, size_t> stage_idx_;   ///< buffer address -> index (the buffers never move)
    void stage_free_events();
    std::atomic<uint64_t> stage_fence_waits_{0};      ///< refills that had to wait for a queued copy
    std::atomic<int64_t> ram_reads_{0};
    std::atomic<uint64_t> file_read_bytes_{0};
    // ---- the Linux I/O path (set_io_prefetch)
    bool io_pf_ = false;
    bool io_fill_ = false;                            ///< experts.bin: the workers only fill the page cache (large preads), no staging, nothing waits
    bool io_ahead_ = true;                            ///< STRATA_IO_PF_AHEAD=0: no read-ahead workers' queue, demand preads only
    bool io_stats_ = false;
    int io_threads_ = 4;
    std::vector<std::thread> io_workers_;
    std::mutex io_mu_;
    std::condition_variable io_cv_;
    std::deque<std::pair<int32_t, int32_t>> io_q_;   ///< (layer, expert) to read ahead
    bool io_quit_ = false;
    std::vector<char> stage_pf_;                      ///< per stage buffer: filled ahead, not yet used by a layer
    std::vector<int> io_fds_;                         ///< per mapped file: a plain descriptor (POSIX_FADV_RANDOM) for the preads
    mutable std::atomic<uint64_t> io_cached_{0}, io_uncached_{0}, io_pread_bytes_{0}, io_pread_us_{0}, io_pread_n_{0};
    std::atomic<uint64_t> io_pf_jobs_{0}, io_pf_demand_{0}, io_pf_blobs_{0}, io_pf_skips_{0}, io_pf_dropped_{0}, io_pf_used_{0}, io_pf_unused_{0},
        io_crit_us_{0}, io_crit_n_{0};
    struct Span { const uint8_t* p; uint64_t n; int fd; uint64_t off; uint64_t at; };
    /// The (up to 3) file ranges of a blob: pointer in the mapping, size, descriptor and file offset, offset in the blob.
    int spans(int64_t layer, int64_t expert, Span* out) const;
    /// Bytes of the blob's pages in the page cache (mincore); `total` the blob's bytes.
    /// `exact` false: four sampled pages (a blob is cached when they are; a handful of syscalls, not hundreds of pages' walk).
    uint64_t cached_bytes(int64_t layer, int64_t expert, uint64_t& total, bool exact = true) const;
    void io_worker();
    void io_enqueue(int64_t layer, int64_t expert);
    void io_stop();
    const uint8_t* io_blob(int64_t layer, int64_t expert);
=======
>>>>>>> their fork (github.com-eddoursul-Strata-custom)
    const uint8_t* base_ = nullptr;
    int64_t blobs_ = 0;
    int64_t n_layers_ = 0;
    int64_t n_expert_ = 0;
<<<<<<< this fork
    uint64_t mapped_bytes_ = 0;
    std::vector<uint64_t> layer_offsets_, layer_blob_bytes_;
    void* complement_arena_ = nullptr;
    const uint8_t* complement_host_ = nullptr;
    const uint8_t* complement_device_ = nullptr;
    uint64_t complement_bytes_ = 0;
    std::vector<uint64_t> complement_offsets_;
    detail::ExchangeStorage exchange_storage_; // authoritative when active; original arenas still own memory
    bool complement_pinned_ = false;
    bool complement_partial_ = false;         ///< CS-T: only the first complement_pin_limit_ bytes are registered
    bool complement_registered_ = false;      ///< the arena is ordinary memory registered with cudaHostRegister (not cudaHostAlloc)
    uint64_t complement_pin_limit_ = 0;
    uint64_t complement_lock_off_ = 0;        ///< the working-set lock covers [lock_off, lock_off + locked)
    bool complement_ready_ = false;
    uint64_t complement_locked_ = 0;          ///< bytes held in the working set (pin refused)
    int64_t complement_lent_slots_ = 0;
    std::vector<std::vector<std::pair<int32_t, int32_t>>> stage_lend_regions_;  ///< a split: each stage's lend region
    std::vector<const uint8_t*> override_;    ///< staged exchanges: an evicted expert read from its exchange buffer
    struct Exchange { size_t in, out; int64_t q; uint64_t bytes; };
    std::vector<Exchange> staged_;
    std::atomic<ForesightSwap*> fs_swap_{nullptr};   ///< set_foresight_swap
    void fs_hold_staged(bool hold);
    std::vector<size_t> soft_in_;             ///< soft_commit_exchanges: per `out` index, the `in` whose place it will take (SIZE_MAX: none)
    const uint8_t* xstage_dev_ = nullptr;     ///< the exchange buffers' device alias (soft_commit_exchanges)
    uint8_t* xstage_ = nullptr;               ///< exchange buffers, `xstage_cap_ x xstage_blob_`
    bool xstage_pinned_ = false;
    int64_t xstage_cap_ = 0;
    uint64_t xstage_blob_ = 0;
    int64_t exchanges_ = 0;
    std::atomic<int64_t> file_reads_{0};
=======
    uint64_t bytes_ = 0;   ///< the mapped length (`expert_layout().total`), for munmap and the bounds check
>>>>>>> their fork (github.com-eddoursul-Strata-custom)
    int64_t reads_ = 0;
#if defined(_WIN32)
    void* file_ = nullptr;
    void* mapping_ = nullptr;
#else
    int fd_ = -1;
#endif
};

// ================================ THE RESIDENT ARENA (R2.1) ================================
//
// **THE MMAP ABOVE IS THE REVIEW'S FINDING C1 AND IT IS WORTH 2.2x.**  Read the comment on `FileExpertSource`
// about the page cache holding all 34 GB: that is true of the expert file ALONE, and it is not what this engine
// does.  The PLE/n-gram shard is a 26.8 GB file that the engine also maps, so 34 GB of experts plus 26.8 GB of
// n-gram is 60.8 GB of mapped, file-backed pages on a 63 GB machine - and file-backed pages are exactly the ones
// the OS drops from the standby list when it wants memory.  The expert stream then re-faults from disk.
//
// Measured, this round: the pool runs at **~19 GB/s in the engine against 42.8 GB/s in `bench/micro/cpu_s2.cpp`
// on the same machine, reading the same 34 GB**.  The micro does `std::fread` into a heap arena and reads
// ordinary (anonymous, resident) memory; the engine reads `MapViewOfFile`.  That is the whole difference, and it
// is why this class exists.
//
// It reads the experts once at startup into anonymous memory the OS has no cheaper reason to evict: one pinned
// block per layer (cudaHostAlloc, portable and mapped), which the GPUs need for cache fills, the prompt path and the
// PCIe share of the misses.  Registering one reserved range instead (cudaHostRegister) is limited to ~42 GiB on
// Windows once a second GPU's context exists, and past that limit every CUDA allocation fails; allocated blocks reach
// ~75 GiB with one or two GPUs.  The layers past the pinning budget share one locked, pageable range; the note says
// how many layers were pinned.
class ArenaExpertSource : public ExpertSource {
public:
    ArenaExpertSource() = default;
    ~ArenaExpertSource() override;
    ArenaExpertSource(const ArenaExpertSource&) = delete;
    ArenaExpertSource& operator=(const ArenaExpertSource&) = delete;

    /// Allocates and loads `<pack_dir>/experts.bin`.  Prints nothing; the caller reports `note()` and the load
    /// rate, because those are the two numbers that say whether the arena is the one that was asked for.
    bool open(const std::string& pack_dir, int64_t n_layers, int64_t n_expert, int threads, std::string& err);
    /// Plan v0.3 P6: a native pack without experts.bin takes its experts from the model's GGUF shards
    /// (strata::gguf_split_paths), found by tensor name.
    void set_gguf(const std::vector<std::string>& shards) { gguf_ = shards; }
    /// Before open(): the experts (n_layers x n_expert, non-zero) that live only in VRAM.  The arena leaves them
    /// out (a 119.5 GiB Q8_0 then fits a 125.7 GiB machine), blob() has none for them, and `read_file_blob` reads
    /// one from the model for the GPU that keeps it.  A native pack read from its GGUF only.
    void set_vram_only(std::vector<uint8_t> mask) { vram_only_ = std::move(mask); }
    void close();

    bool mapped() const { return !layer_base_.empty(); }
    int64_t blobs() const { return blobs_; }
    const uint8_t* blob(int64_t layer, int64_t expert) override;
    /// `blob` without the read counter: a pure pointer into the arena, safe from another thread (the async adaptive
    /// tier's job thread, which must not touch the non-atomic `reads_`)
    const uint8_t* blob_at(int64_t layer, int64_t expert) const;
    int64_t reads() const { return reads_; }
    bool pinned(int64_t layer, int64_t expert) const override;
    const uint8_t* device_alias(int64_t layer, int64_t expert) const override;
<<<<<<< this fork
    /// Prefix pinning: the old rule (device_alias(layer, 0)); pin by need: at least one pinned range in the layer.
    bool pcie_layer(int64_t layer) const override;
    void prefetch(int64_t layer, int64_t expert) override;
    uint64_t release(int64_t layer, int64_t expert) override;
=======
    bool layer_mapped(int64_t layer) const override {
        return layer >= 0 && layer < pinned_layers_ && layer_dev_[(size_t) layer] != nullptr;
    }
    bool in_ram(int64_t layer, int64_t expert) const override {
        return idx_.empty() || idx_[(size_t) (layer * n_expert_ + expert)] >= 0;
    }
    /// One expert's blob read from the model into `dst` (blob_bytes(layer) bytes; any expert, thread-safe).
    bool read_file_blob(int64_t layer, int64_t expert, uint8_t* dst, std::string& err) const;
    /// The bytes the arena left out (set_vram_only).
    uint64_t vram_only_bytes() const { return vram_only_bytes_; }
>>>>>>> their fork (github.com-eddoursul-Strata-custom)

    /// Pin by need (STRATA_PIN_BY_NEED=1): BEFORE open(), the (layer, expert) pairs worth pinning, by decreasing
    /// priority - the profile's ranked pairs the GPU cache will NOT hold, then the unranked ones.  open() merges
    /// neighbours into byte ranges and registers only those (PinnedArena::ByNeed); pinned()/device_alias()/pcie_layer()
    /// then follow the registered ranges instead of a prefix.  Never called: the old prefix pinning, unchanged.
    /// `n_first`: the first n_first pairs are the non-resident estimate; the rest (the pairs taken as GPU-resident, which
    /// only get what the cap leaves over) never merge into a run with them, so their ranges come after all of the first.
    void set_pin_priority(std::vector<std::pair<int32_t, int32_t>> ranked, size_t n_first = (size_t) -1) {
        pin_pairs_ = std::move(ranked);
        pin_first_ = n_first;
        pin_by_need_ = true;
    }
    /// Pin by need, after open(): the registered ranges (0 when the prefix rule, or nothing, is in force)
    size_t pin_ranges() const { return pin_start_.size(); }

    /// What backing was obtained and why, for the startup print.  "The engine adapts to the machine it is on" is
    /// only true if the engine says what it got.
    const std::string& note() const { return note_; }
    double load_gib_per_second() const { return gib_per_s_; }

private:
    std::vector<uint8_t*> layer_base_;       ///< each layer's experts: a pinned block, or a part of `tail_`
    std::vector<const uint8_t*> layer_dev_;  ///< the device alias of each pinned layer (null for the others)
    std::vector<uint8_t*> blocks_;           ///< the pinned blocks, owned (layers [0, blocks_.size()))
    void* tail_ = nullptr;                   ///< the layers past the pinning budget, locked resident, owned
    uint64_t tail_bytes_ = 0;
    /// Linux (STRATA_PIN_AFTER_COPY, default on): the arena as one anonymous range the load's threads write into,
    /// each layer page-locked with cudaHostRegister after the load (cudaHostAlloc zeroes and locks it on one thread
    /// first: ~16 s for 55 GiB).  The layers past the first refused registration are locked resident instead.
    void* anon_ = nullptr;
    uint64_t anon_bytes_ = 0;
    std::vector<std::pair<uint8_t*, uint64_t>> registered_;
    uint8_t* anon_lock_ = nullptr;
    uint64_t anon_lock_bytes_ = 0;
    int64_t pinned_layers_ = 0;              ///< layers [0, pinned_layers_) are page-locked and mapped
    int64_t blobs_ = 0;
    int64_t n_expert_ = 0;
    int64_t reads_ = 0;
    std::string note_;
    double gib_per_s_ = 0.0;
<<<<<<< this fork
    double load_seconds_ = 0.0;
    double load_read_s_ = 0.0;
    double load_copy_s_ = 0.0;
    uint64_t pinned_bytes_ = 0;
    // pin by need: set_pin_priority's pairs, and (after open) the registered ranges - sorted starts, ends and the
    // device alias of each start - plus pcie_layer's answer per layer.  `ranges_` false = the prefix rule above.
    bool pin_by_need_ = false;
    bool ranges_ = false;
    // pin by need: the arena's page-aligned blob stride (see PadMap); `padded_` false = the packed experts.bin layout
    bool padded_ = false;
    PadMap pad_;
    /// where blob (layer, expert) starts in the arena: the ONE address function of this class
    uint64_t arena_off(int64_t layer, int64_t expert) const;
    std::vector<std::pair<int32_t, int32_t>> pin_pairs_;
    size_t pin_first_ = (size_t) -1;   // pin_pairs_[0, pin_first_) are the first tier (see set_pin_priority)
    std::vector<uint64_t> pin_start_, pin_end_;
    std::vector<const uint8_t*> pin_dev_;
    std::vector<uint8_t> layer_pcie_;
    /// index of the registered range holding [off, off + len), or -1
    int64_t pin_range_of(uint64_t off, uint64_t len) const;
    std::string gguf_;
};

/// Plan v0.3 P6: checks native_experts.txt's GGUF spans against the files, before anything is read: each layer's
/// gate/up/down at its recorded (file, offset) must be that tensor (`blk.L.ffn_<role>_exps.weight`), of the
/// layout's type and dimensions, and inside the file.  `native` is the --native shard (see set_gguf).
bool check_experts_gguf(const std::string& native, const strata::kernels::cpu::ExpertLayout& lay, std::string& err);
/// Fills `dst` (lay.total bytes, the experts.bin layout) from the GGUF files, one role at a time.
/// `unbuffered`: each chunk read past the file cache (Windows); `ready`: layer l is written only once
/// *ready > l + 1 (an arena that is still being registered).
/// `pad` (optional): the destination is the page-aligned padded layout (pin by need), see PadMap.
LoadStats load_experts_gguf(const std::string& native, uint8_t* dst, const strata::kernels::cpu::ExpertLayout& lay,
                            int threads, bool unbuffered = false, const std::atomic<int>* ready = nullptr,
                            const PadMap* pad = nullptr);

=======
    std::vector<std::string> gguf_;
    std::vector<uint8_t> vram_only_;
    /// per (layer, expert): its place among its layer's blobs in the arena, -1 for one only in VRAM (empty: all
    /// in the arena, at their expert index)
    std::vector<int32_t> idx_;
    uint64_t vram_only_bytes_ = 0;
    std::vector<std::pair<size_t, uint64_t>> spans_;   ///< per (layer, role): its shard and the tensor's offset
    std::vector<int> fds_;                              ///< the shards, open for read_file_blob
};

>>>>>>> their fork (github.com-eddoursul-Strata-custom)
}  // namespace strata::core
