// include/strata/kernels/fused_gr.hpp - the hyper-connection read in two kernels, with the previous half's write
// folded in, for up to 8 tokens that share the weights.
//
//   down : R' = R + bo_prev * 2 sigmoid(inj_prev / hc)          (only when `apply`, computed on the fly)
//          rs[c] = rsqrt(mean(R'[c]^2) + eps)
//          lo[k] = silu((w_down[k] . xn) / hc),  xn = R' * w_norm * rs    k < hc_lr
//          inject[c] = w_inject[c] . xn                               when w_inject is given
//   up   : R <- R' in place (when `apply`)
//          mixed[d] = mean_c  xn[c,d] * sigmoid(w_up[c*n_embd + d] . lo)
//          gate_ema[c,d] += (mean over the tokens of sigmoid(w_up[c*n_embd + d] . lo) - gate_ema[c,d]) / 4
//                                                                     when `gate_ema` is given
//          est[d] = mean_c R'[c,d] * rs[c] * est_norm[c,d] * est_gates[c,d]   when `est_norm` is given: the mixed of
//                   another read of R' (its norm weights) with the gates `est_gates`
//
// The down projection is split by input slices: 80 blocks each take a quarter of the 324 rows over a twentieth of
// the input, so each block reads 1/20 of every token's R' instead of all of it, and the stream's rs multiplies the
// slice sums (a slice lies in one stream).  The last block of each quarter adds the partials in a fixed order.
// FP32 activations and BF16 weights; every token's outputs are the same whatever its place in the window and the
// window's size.  Geometry is the artifact's: n_embd 2560, hc 4, hc_lr 320.  `inj_prev` and `inject_out` must be
// different buffers (every block reads the former while one block writes the latter).
#pragma once

#include <cstddef>
#include <cstdint>

namespace strata::kernels {

struct FusedGrArgs {
    const float* R = nullptr;          ///< (hc, n_embd); `up` updates it in place when apply
    float* R_out = nullptr;            ///< == R for the in-place update
    bool apply = false;                ///< fold the previous half's gr_write
    const float* bo_prev = nullptr;    ///< that half's block output, n_embd
    const float* inj_prev = nullptr;   ///< that half's injection, hc
    const float* w_norm = nullptr;     ///< (hc * n_embd) f32
    const uint16_t* w_down = nullptr;  ///< bf16 [hc_lr][hc*n_embd]
    const uint16_t* w_up = nullptr;    ///< bf16 [hc*n_embd][hc_lr]
    const uint16_t* w_inject = nullptr;///< bf16 [hc][hc*n_embd], or null (the final mixer)
    float eps = 1e-6f;
    float* lo = nullptr;               ///< workspace, hc_lr floats
    float* rs = nullptr;               ///< workspace, hc floats
    float* inject_out = nullptr;       ///< hc floats (when w_inject)
    float* mixed = nullptr;            ///< n_embd
    float* gate_ema = nullptr;         ///< (hc, n_embd): the gates' running average over the reads, or null
    const float* est_norm = nullptr;   ///< another read's w_norm, or null
    const float* est_gates = nullptr;  ///< (hc, n_embd): the gates for its estimate
    float* est = nullptr;              ///< n_embd: the estimate
};

bool fused_gr_supported(int64_t n_embd, int64_t hc, int64_t hc_lr);
<<<<<<< this fork
void fused_gr_read(const FusedGrArgs& a, void* stream);
/// STRATA_HC_Q8 on a bf16 pack: `out` (n_blocks * 34 bytes) = the Q8_0 blocks (fp16 d = amax / 127, round(x / d), ggml's
/// reference rounding) of `n_blocks * 32` bf16 values on the device.
/// The inverse for a path with no Q8 kernel: bf16 (round to nearest even) of d * q for `n_blocks` Q8_0 blocks.
void fused_gr_dequant_q8_0(const uint8_t* q8, uint16_t* bf16, int64_t n_blocks, void* stream);
void fused_gr_quantize_q8_0(const uint16_t* bf16, uint8_t* out, int64_t n_blocks, void* stream);
=======
>>>>>>> their fork (github.com-eddoursul-Strata-custom)

/// The read of `n_tok` tokens (1-8): `a[t]` is token t's arguments (its own R, pending write, lo, rs, inject,
/// mixed, est; the four weight pointers, gate_ema, est_norm, est_gates and eps must be the same for every t).
/// `scratch` is `fused_gr_scratch_bytes()`, zeroed once when allocated (the kernels leave it zeroed); reads that share
/// it must run in stream order.
constexpr int kFusedGrMaxT = 8;
size_t fused_gr_scratch_bytes();
void fused_gr_read_multi(const FusedGrArgs* a, int n_tok, float* scratch, void* stream);

}  // namespace strata::kernels
