// include/strata/prefill/kernels.hpp - plan v0.3 P5: the element-wise and stateful kernels of a prompt chunk.
//
// T tokens at a time, all row-major with the token as the slow index.  The arithmetic mirrors the per-token native
// kernels (the decode path), batched; where a step is a recurrence in time (the GDN conv and state, the PLE conv)
// the kernel walks the chunk's tokens in order inside one launch.
#pragma once

#include <cstdint>

namespace strata::prefill {

// ---- hyper-connection (n_embd 2560, hc 4, hc_lr 320)
/// xn[t, c*2560 + d] = R[t,c,d] * rsqrt(mean_d R[t,c,:]^2 + eps) * w_norm[c*2560 + d]; also its BF16 image.
void gr_norm(const float* R, const float* w_norm, float eps, float* xn, uint16_t* xn16, int64_t T, void* stream);
/// gr_norm without the FP32 rows: the row scales rs[t*4 + c] and the BF16 image; gr_mix_r reads R instead.
void gr_norm_rs(const float* R, const float* w_norm, float eps, float* rs, uint16_t* xn16, int64_t T, void* stream);
/// gr_mix with xn recomputed from R, rs and w_norm as gr_norm computes it (the same bits).
void gr_mix_r(const float* R, const float* rs, const float* w_norm, const float* gated, float* mixed, uint16_t* mixed16,
              int64_t T, void* stream, uint16_t* mixed_h = nullptr);
/// gr_write, then gr_norm_rs of the written rows with the next read's norm weights, R read once.
void gr_write_norm_rs(float* R, const float* bo, const float* inj, int64_t inj_ld, const float* w_norm, float eps,
                      float* rs, uint16_t* xn16, int64_t T, void* stream, const uint16_t* partial = nullptr);
/// lo16[t, k] = bf16(silu(lo[t, k] / hc))
void gr_silu(const float* lo, uint16_t* lo16, int64_t T, void* stream);
/// mixed[t, d] = mean_c xn[t, c, d] * sigmoid(gated[t, c, d]); FP32, BF16 and FP16 (any of them may be null).
void gr_mix(const float* xn, const float* gated, float* mixed, uint16_t* mixed16, int64_t T, void* stream,
            uint16_t* mixed_h = nullptr);
/// R[t, c, d] += (bo[t, d] (+ partial[t, d], FP16)) * 2 sigmoid(inj[t, c] / hc)   (inj has row stride inj_ld)
void gr_write(float* R, const float* bo, const float* inj, int64_t inj_ld, int64_t T, void* stream,
              const uint16_t* partial = nullptr);
/// R[t, c, :] = e[t, :] for all four streams (the embedding broadcast).
void gr_broadcast(const float* e, float* R, int64_t T, void* stream);

// ---- GDN (state 128, 16 k heads, 48 v heads, 10240 conv channels, 4 taps)
/// gate[t,h] = softplus(ab[t,h] + dt[h]) * ssm_a[h];  beta[t,h] = sigmoid(ab[t, 48 + h])  (ab: [T, 96])
void gdn_gates(const float* ab, const float* dt, const float* ssm_a, float* gate, float* beta, int64_t T, void* stream);
/// The 4-tap causal conv + SiLU over the chunk (history [C][3] in, updated to the chunk's last three inputs), then
/// the L2 norm of the q and k heads of every token: the verify window's gdn_conv_l2_multi over all T tokens at once.
/// h: [T, C].
void gdn_conv(float* history, const float* qkv, const float* conv_w, float* h, int64_t T, float eps, void* stream);
<<<<<<< this fork
/// The recurrence over the chunk, state in registers; y16[t] = rmsnorm(o) * gamma * sigmoid(z) in FP16 (what the
/// out projection reads); y is FP32 scratch.
void gdn_recurrence(float* state, const float* h, const float* gate, const float* beta, const float* z,
                    const float* gamma, float eps, float* y, uint16_t* y16, int64_t T, void* stream,
                    int64_t ld16 = 0);   // ld16: y16's row stride (0 = 6144; S23 STRATA_PF_PAD pads it)
/// The kernels behind gdn_recurrence, for the parity test: 0 = the column-split kernels + the norm kernel (or the
/// one-block-per-head kernel under STRATA_GDN_REC_HEADS), 1 = four lanes per column (no barrier per token) + the grid-stride norm, 2 = the
/// one-block-per-head kernel (its fused norm rounds differently).  0 and 1 give the same bits.
/// SYCL only: 6 = the Xe sub-group recurrence (16-lane sub-groups, one barrier per token) + the norm kernel, the same
/// bits as 0 on an Arc A770; gdn_recurrence takes it by default on Alchemist (STRATA_GDN_XE=0 off, =1 any Intel GPU).
void gdn_recurrence_variant(int variant, float* state, const float* h, const float* gate, const float* beta,
                            const float* z, const float* gamma, float eps, float* y, uint16_t* y16, int64_t T,
                            void* stream, int64_t ld16 = 0);
=======
/// The recurrence over the chunk (state in registers): y[t] = o (before the norm).  With gdn_out_norm, bitwise
/// gdn_step_norm_multi over the tokens.
void gdn_scan(float* state, const float* h, const float* gate, const float* beta, float* y, int64_t T, void* stream);
/// In place: y[t] = rmsnorm(o) * gamma * sigmoid(z) per head, and its FP16 bits (the out projection is quantized).
void gdn_out_norm(float* y, const float* z, const float* gamma, float eps, uint16_t* y16, int64_t T, void* stream);
>>>>>>> their fork (github.com-eddoursul-Strata-custom)

// ---- MoE
/// softmax over 512, top-10 (ties to the lower id), weights renormalised over the ten (the native router).
void route(const float* logits, int32_t* ids, float* weights, int64_t T, void* stream);
/// Expert blob (Strata pack layout, Q2_0) -> BF16 matrices: gate/up interleaved [1280, 2560], down [2560, 640].
void blob_dequant(const uint8_t* blob, uint16_t* gu16, uint16_t* down16, void* stream);
/// h16[n, r] = fp16(silu(gu[n, 2r]) * gu[n, 2r + 1])   (the interleaved expert gate/up)
void swiglu_interleaved(const float* gu, uint16_t* h16, int64_t n, void* stream);
/// h16[n, r] = fp16(silu(g[n, r]) * u[n, r])   (the shared expert, gate and up separate, width 640)
void swiglu_pair(const float* g, const float* u, uint16_t* h16, int64_t n, void* stream);
/// Gather rows: dst16[i, :] = x16[src[i], :] (n rows of `width` BF16).
void gather_rows16(const uint16_t* x16, const int32_t* src, uint16_t* dst16, int64_t n, int64_t width, void* stream);
<<<<<<< this fork
/// bo[t, :] = shared[t, :] * sigmoid(sg[t]) + sum_k w[t, k] * D[slot[t, k], :]
void moe_combine(const float* D, const int32_t* slot, const float* w, const float* shared, const float* sg, float* bo,
                 int64_t T, void* stream, const float* row_sd = nullptr);
/// --peer-device's prompt share as sums: bo[t, :] = (sum over the k with slot[t, k] < rows_local of w[t, k] *
/// D[slot[t, k], :], in k order) + peer[t, :] + shared[t, :] * sigmoid(sg[t]).  `peer` (the peer's per-token sums)
/// may be mapped host memory.
void moe_combine_peer(const float* D, const int32_t* slot, const float* w, const float* shared, const float* sg,
                      const float* peer, int64_t rows_local, float* bo, int64_t T, void* stream);
/// sum[t, :] += wk[p] * rows[r, :] with p = pair[r], t = p / 10 (the routed pair's token), for n rows of ONE expert
/// (no token twice: launches in a fixed expert order make the sums repeatable).  After eddoursul/Strata's
/// moe_scatter_add (f8de703).
void peer_scatter_add(float* sum, const float* rows, const float* wk, const int32_t* pair, int64_t n, void* stream);
/// The same for a group of experts in one launch (a token may recur): token tok[b] adds its rows list[start[b] ..
/// start[b + 1]) in that order (the experts' order), row r at rows[r - r0]: bitwise the per-expert launches'.
/// eddoursul/Strata's moe_gather_add, with the weight looked up through the row's pair.
void peer_gather_add(float* sum, const float* rows, int64_t r0, const float* wk, const int32_t* pair, const int32_t* tok,
                     const int32_t* start, const int32_t* list, int64_t n_tok, void* stream);
/// moe_combine_peer with the peer's sums in FP16 (`peer16`, may be mapped host memory)
void moe_combine_peer16(const float* D, const int32_t* slot, const float* w, const float* shared, const float* sg,
                        const uint16_t* peer16, int64_t rows_local, float* bo, int64_t T, void* stream);
/// y = FP16 bits of x, saturated at +-65504 (the peer's sums on their way back; eddoursul/Strata 2acfac4)
=======
/// The same from FP16 rows into FP32 ones (MMQ's quantizer reads floats).
void gather_rows16_f32(const uint16_t* x16, const int32_t* src, float* dst, int64_t n, int64_t width, void* stream);
/// y = FP16 bits of x, saturated at +-65504 (the second GPU's expert sums on their way back: below ~20 measured)
>>>>>>> their fork (github.com-eddoursul-Strata-custom)
void sums_to_f16(const float* x, uint16_t* y, int64_t n, void* stream);
/// In place: x[t, :] *= sigmoid(sg[t])   (the shared expert's scalar gate)
void moe_shared_gate(float* x, const float* sg, int64_t T, void* stream);
/// sum[src[r], :] += w[r] * rows[r, :] for n rows (one expert's rows: no token twice).
void moe_scatter_add(float* sum, const float* rows, const float* w, const int32_t* src, int64_t n, void* stream);
/// The same for several experts' rows (row r at rows[r - r0]), where a token may recur: token tok[b] adds its rows
/// list[start[b] .. start[b + 1]) in that order, so it matches moe_scatter_add per expert in the list's order.
void moe_gather_add(float* sum, const float* rows, int64_t r0, const float* w, const int32_t* tok, const int32_t* start,
                    const int32_t* list, int64_t n_tok, void* stream);

// ---- QSA helpers
/// In place: x[r, :] = x[r, :] * rsqrt(mean x^2 + eps) * w  over rows of `cols` (row stride `ld`).
void rms_rows(float* x, const float* w, int64_t rows, int64_t cols, int64_t ld, float eps, void* stream);
/// NEOX rotary on the first 64 dims of each head: x [T, heads, dim] at positions pos0 + t.
void rope(float* x, int64_t T, int64_t heads, int64_t dim, int64_t ld, int64_t pos0, float freq_base, void* stream);
/// q_full [T, 24, 512] (q | gate per head) -> q [T, 24, 256]
void split_q(const float* q_full, float* q, int64_t T, void* stream);
/// attn[t, h, d] *= sigmoid(q_full[t, h, 256 + d]) -> out16 (fp16 bits: the o-projection is quantized)
void gate_attn(const float* attn, const float* q_full, uint16_t* out16, int64_t T, void* stream);

/// K and V of T consecutive cells (positions pos0..pos0+T-1; K normed and rotated) into the paged pools: FP16
/// (`k_pool`/`v_pool`) or INT8 codes + FP16 scale per 64 (`k_q`...), the decode append's arithmetic.
/// K, V: [T, 2, 256].  `sides`: 3 K and V, 1 K alone (k8v4's INT8 half), 2 V alone.
void kv_append(const float* K, const float* V, int64_t T, int64_t pos0, const int32_t* page_table, int64_t page_size,
               uint16_t* k_pool, uint16_t* v_pool, int8_t* k_q, int8_t* v_q, uint16_t* k_scale, uint16_t* v_scale,
               void* stream, int sides = 3);

/// fp32 -> fp16 bits and fp32 -> bf16, n elements (the two activation images of the prompt GEMMs).
void to_f16(const float* x, uint16_t* y, int64_t n, void* stream);
void to_bf16(const float* x, uint16_t* y, int64_t n, void* stream);
/// Expert blob -> FP16 (Q2_0 values are exact in FP16).
void blob_dequant_f16(const uint8_t* blob, uint16_t* gu16, uint16_t* down16, void* stream);

}  // namespace strata::prefill
