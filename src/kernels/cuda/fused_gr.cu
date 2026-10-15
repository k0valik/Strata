// src/kernels/cuda/fused_gr.cu - see include/strata/kernels/fused_gr.hpp.
<<<<<<< this fork
#include "strata/core/emulate.hpp"
#include "strata/kernels/q8_1_finite.hpp"   // #606: q8_1_ds
#include "strata/core/geometry.hpp"
=======
>>>>>>> their fork (github.com-eddoursul-Strata-custom)
#include "strata/kernels/fused_gr.hpp"
#include "xor_scatter.cuh"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>

namespace strata::kernels {
namespace {

constexpr int N = 2560;         // n_embd
constexpr int HC = 4;           // streams
constexpr int D = N * HC;       // 10240
constexpr int LR = 320;         // hc_lr
<<<<<<< this fork
// the latency-hidden norm/up (gr_norm_fast_kernel / gr_up_fast_kernel, STRATA_GR_FAST): gfx906's, built for CUDA too
#if defined(STRATA_HIP_GFX906) || !defined(__HIPCC__)
#define STRATA_GR_FAST_BUILD 1
#else
#define STRATA_GR_FAST_BUILD 0
#endif
static_assert(N == strata::core::kCompiledGeometry.n_embd && HC == strata::core::kCompiledGeometry.hc && LR == strata::core::kCompiledGeometry.hc_lr, "fused_gr is compiled for core/geometry.hpp");
=======
constexpr int NR = LR + HC;     // the down projection's rows: w_down's 320, then w_inject's 4

// down: the NR rows in DG groups of DRB = DW warps x DR rows, the input in DS slices of DCS columns (a slice lies in
// one stream); a block per (slice, group), up to DT tokens a launch
constexpr int DG = 4, DS = 20, DW = 9, DR = 9, DT = 4;
constexpr int DTH = DW * 32, DRB = DW * DR, DCS = D / DS, DSPC = DS / HC, DSTEPS = DCS / 256, DNF = DCS / 4;
static_assert(DRB * DG == NR && DCS % 256 == 0 && DS % HC == 0 && DNF <= DTH, "down geometry");

// up: UPM_COLS columns of every stream a block, UPM_ROWS rows a warp
>>>>>>> their fork (github.com-eddoursul-Strata-custom)
constexpr int THREADS = 256;
constexpr int WARPS = THREADS / 32;
constexpr int UPM_COLS = 16;
constexpr int UPM_BLOCKS = N / UPM_COLS;          // 160
constexpr int UPM_ROWS = HC * UPM_COLS / WARPS;   // 8

// the scratch (floats): DG counters, the slices' sums of squares [DG][DS][DT], the partial dots [DS][DT][NR]
constexpr int SC_CTR = 0, SC_SSP = 64, SC_PART = SC_SSP + DG * DS * DT, SC_FLOATS = SC_PART + DS * DT * NR;

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}
__device__ __forceinline__ float sigmoidf_(float x) { return 1.0f / (1.0f + __expf(-x)); }

// 8 bf16 packed in a uint4 against 8 floats held as two float4
__device__ __forceinline__ float dot8h(const uint4 w, const float4 x0, const float4 x1) {
    float acc = 0.0f;
    acc = fmaf(__uint_as_float(w.x << 16), x0.x, acc);
    acc = fmaf(__uint_as_float(w.x & 0xffff0000u), x0.y, acc);
    acc = fmaf(__uint_as_float(w.y << 16), x0.z, acc);
    acc = fmaf(__uint_as_float(w.y & 0xffff0000u), x0.w, acc);
    acc = fmaf(__uint_as_float(w.z << 16), x1.x, acc);
    acc = fmaf(__uint_as_float(w.z & 0xffff0000u), x1.y, acc);
    acc = fmaf(__uint_as_float(w.w << 16), x1.z, acc);
    acc = fmaf(__uint_as_float(w.w & 0xffff0000u), x1.w, acc);
    return acc;
}

struct GrMulti {
    FusedGrArgs a[kFusedGrMaxT];
    int T;
};

__host__ __device__ constexpr int pow2_floor(int n) { return n < 2 ? 1 : 2 * pow2_floor(n / 2); }

// v[B, V) summed across the warp in power-of-two chunks of xor_scatter; emit(value, sum) on one lane per value
template <int V, int B, int VT, typename F>
__device__ __forceinline__ void reduce_chunks(const float (&v)[VT], int lane, F& emit) {
    if constexpr (B < V) {
        constexpr int P = pow2_floor(V - B > 32 ? 32 : V - B);
        float cv[P];
#pragma unroll
        for (int e = 0; e < P; ++e) cv[e] = v[B + e];
        xor_scatter<P>(cv, lane);
        if ((lane & (32 / P - 1)) == 0) emit(B + xor_scatter_index<P>(lane), cv[0]);
        reduce_chunks<V, B + P, VT>(v, lane, emit);
    }
}

// The down projection and the injection of TT tokens.  Block (s, g): warp w takes rows g DRB + w DR + [0, DR) of
// [w_down; w_inject] over the slice's DCS columns, a lane 8 columns a step, each activation chunk loaded once for its
// DR rows.  The slice's R' = R + bo_prev * 2 sigmoid(inj_prev / hc) times w_norm is staged in shared memory split into
// the halves of the 8-column chunks (conflict-free 16-byte reads), and the sums of squares of R' go to the scratch:
// a slice lies in one stream, so the stream's 1/rms multiplies the slice sums later.  The group's last block (a
// counter) adds each row's partials in slice order, per stream times the stream's 1/rms, and writes lo = silu(y / hc)
// or the injection; group 0's also writes rs.  Every token's arithmetic is the same whatever its place in the window
// and the window's size.
template <int TT>
__global__ void __launch_bounds__(DTH) gr_down_split_kernel(GrMulti m, float* __restrict__ scratch) {
    static_assert(TT >= 1 && TT <= DT, "tokens a launch");
    __shared__ __align__(16) float4 xs[TT][DNF];
    __shared__ float ssw[TT][DW];
    __shared__ float s_rs[TT][HC];
    __shared__ bool s_last;
    unsigned* ctr = reinterpret_cast<unsigned*>(scratch + SC_CTR);
    float* ssp = scratch + SC_SSP;
    float* part = scratch + SC_PART;
    const int t = threadIdx.x, lane = t & 31, warp = t >> 5;
    const int s = blockIdx.x, g = blockIdx.y;
    const int c = s / DSPC, col0 = s * DCS;
    const bool has_inj = m.a[0].w_inject != nullptr;
    const int rb0 = g * DRB + warp * DR;
    // the staging loads first, then the weights
    float4 pr[TT], pb[TT], pw = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    if (t < DNF) {
        pw = *reinterpret_cast<const float4*>(m.a[0].w_norm + col0 + 4 * t);
#pragma unroll
        for (int k = 0; k < TT; ++k) {
            const FusedGrArgs& a = m.a[k];
            pr[k] = *reinterpret_cast<const float4*>(a.R + col0 + 4 * t);
            if (a.apply) pb[k] = *reinterpret_cast<const float4*>(a.bo_prev + col0 - c * N + 4 * t);
        }
    }
    uint4 w[DR][DSTEPS];
#pragma unroll
    for (int i = 0; i < DR; ++i) {
        const int rb = rb0 + i;
        const bool ok = rb < LR || has_inj;
        const uint16_t* base =
            rb < LR ? m.a[0].w_down + (size_t) rb * D : m.a[0].w_inject + (size_t) (ok ? rb - LR : 0) * D;
        const uint4* w4 = reinterpret_cast<const uint4*>(base + col0) + lane;
#pragma unroll
        for (int j = 0; j < DSTEPS; ++j) w[i][j] = ok ? __ldg(w4 + j * 32) : make_uint4(0, 0, 0, 0);
    }
#pragma unroll
    for (int k = 0; k < TT; ++k) {
        const FusedGrArgs& a = m.a[k];
        const float gw = a.apply ? 2.0f * sigmoidf_(a.inj_prev[c] / (float) HC) : 0.0f;
        float sq = 0.0f;
        if (t < DNF) {
            float4 r = pr[k];
            if (a.apply) {
                r.x = fmaf(pb[k].x, gw, r.x); r.y = fmaf(pb[k].y, gw, r.y);
                r.z = fmaf(pb[k].z, gw, r.z); r.w = fmaf(pb[k].w, gw, r.w);
            }
            sq = fmaf(r.x, r.x, sq); sq = fmaf(r.y, r.y, sq); sq = fmaf(r.z, r.z, sq); sq = fmaf(r.w, r.w, sq);
            xs[k][(t & 1) * (DNF / 2) + (t >> 1)] = make_float4(r.x * pw.x, r.y * pw.y, r.z * pw.z, r.w * pw.w);
        }
        sq = warp_sum(sq);
        if (lane == 0) ssw[k][warp] = sq;
    }
    __syncthreads();
    if (t < TT) {
        float sq = 0.0f;
#pragma unroll
        for (int q = 0; q < DW; ++q) sq += ssw[t][q];
        ssp[(g * DS + s) * DT + t] = sq;
    }
    float acc[DR * TT];
#pragma unroll
    for (int e = 0; e < DR * TT; ++e) acc[e] = 0.0f;
#pragma unroll
    for (int j = 0; j < DSTEPS; ++j) {
        float4 x0[TT], x1[TT];
#pragma unroll
        for (int k = 0; k < TT; ++k) { x0[k] = xs[k][j * 32 + lane]; x1[k] = xs[k][DNF / 2 + j * 32 + lane]; }
#pragma unroll
        for (int i = 0; i < DR; ++i)
#pragma unroll
            for (int k = 0; k < TT; ++k) acc[i * TT + k] += dot8h(w[i][j], x0[k], x1[k]);
    }
    auto emit = [=](int vi, float sum) {
        const int i = vi / TT, k = vi - i * TT, rb = rb0 + i;
        if (rb < LR || has_inj) part[(s * DT + k) * NR + rb] = sum;
    };
    reduce_chunks<DR * TT, 0, DR * TT>(acc, lane, emit);
    // the group's last block
    __threadfence();
    __syncthreads();
    if (t == 0) s_last = atomicAdd(ctr + g, 1u) == (unsigned) (DS - 1);
    __syncthreads();
    if (!s_last) return;
    __threadfence();
    if (t < TT * HC) {
        const int k = t / HC, cc = t - k * HC;
        float sq = 0.0f;
#pragma unroll
        for (int q = 0; q < DSPC; ++q) sq += __ldcg(ssp + (g * DS + cc * DSPC + q) * DT + k);
        s_rs[k][cc] = rsqrtf(sq / (float) N + m.a[0].eps);
    }
    __syncthreads();
    for (int e = t; e < TT * DRB; e += DTH) {
        const int k = e / DRB, r = e - k * DRB, rb = g * DRB + r;
        if (rb >= LR && !has_inj) continue;
        float y = 0.0f;
#pragma unroll
        for (int cc = 0; cc < HC; ++cc) {
            float p = 0.0f;
#pragma unroll
            for (int q = 0; q < DSPC; ++q) p += __ldcg(part + ((cc * DSPC + q) * DT + k) * NR + rb);
            y = fmaf(s_rs[k][cc], p, y);
        }
        if (rb >= LR) {
            m.a[k].inject_out[rb - LR] = y;
        } else {
            const float x = y / (float) HC;
            m.a[k].lo[rb] = x / (1.0f + __expf(-x));
        }
    }
    if (g == 0 && t < TT * HC) m.a[t / HC].rs[t % HC] = s_rs[t / HC][t % HC];
    if (t == 0) ctr[g] = 0u;
}

// The up projection of TT tokens: each row of w_up read once, RW rows a warp at a time; the TT x RW dots of those rows
// summed by one xor tree (xor_scatter) and every (row, token) epilogue on the lane holding its sum, its inputs fetched
// while the dots run: R <- R' for this block's columns (when apply), mixed[d] = mean_c xn[c,d] * sigmoid(u[c,d]), the
// gates' running average (when gate_ema) and the other read's estimate (when est_norm).  The lo vectors are staged
// split into chunk halves.
template <int TT>
__global__ void __launch_bounds__(THREADS) gr_up_multi_kernel(GrMulti m) {
    constexpr int RW = TT <= 4 ? 4 : 2;
    constexpr int NV = pow2_at_least(TT * RW);    // sums a lane reduces, zero-padded
    constexpr int LPV = 32 / NV;                  // lanes that end up holding each sum
    __shared__ __align__(16) float4 lo4[TT][2][LR / 8];
    __shared__ float g[TT][HC][UPM_COLS];
    __shared__ float gs[TT][HC][UPM_COLS];        // the gates
    __shared__ float ge[TT][HC][UPM_COLS];        // the estimate's terms
    const bool est = m.a[0].est_norm != nullptr;
    const int t = threadIdx.x, lane = t & 31, warp = t >> 5;
    const int d0 = blockIdx.x * UPM_COLS;
    for (int i = t; i < TT * LR / 4; i += THREADS) {
        const int k = i / (LR / 4), f = i - k * (LR / 4);
        lo4[k][f & 1][f >> 1] = reinterpret_cast<const float4*>(m.a[k].lo)[f];
    }
    __syncthreads();
    const int vi = xor_scatter_index<NV>(lane);   // the sum this lane ends with: row vi / TT, token vi % TT
    const int vq = vi / TT, vk = vi - vq * TT;
    const bool epi = (lane & (LPV - 1)) == 0 && vi < TT * RW;
    for (int r0 = warp * UPM_ROWS; r0 < (warp + 1) * UPM_ROWS; r0 += RW) {
        uint4 wa[RW], wb[RW];
#pragma unroll
        for (int q = 0; q < RW; ++q) {
            const int r = r0 + q, c = r / UPM_COLS, dd = r - c * UPM_COLS, i = c * N + d0 + dd;
            const uint4* w4 = reinterpret_cast<const uint4*>(m.a[0].w_up + (size_t) i * LR);
            wa[q] = __ldg(w4 + lane);
            wb[q] = lane < LR / 8 - 32 ? __ldg(w4 + 32 + lane) : make_uint4(0, 0, 0, 0);
        }
        const int er = r0 + vq, ec = er / UPM_COLS, edd = er - ec * UPM_COLS, ei = ec * N + d0 + edd;
        float rv = 0.0f, wn = 0.0f, rsc = 0.0f, bo = 0.0f, ip = 0.0f, en = 0.0f, eg = 0.0f;
        bool apply = false;
        if (epi) {
            const FusedGrArgs& a = m.a[vk];
            rv = a.R[ei];
            wn = a.w_norm[ei];
            rsc = a.rs[ec];
            apply = a.apply;
            if (apply) { bo = a.bo_prev[d0 + edd]; ip = a.inj_prev[ec]; }
            if (est) { en = a.est_norm[ei]; eg = a.est_gates[ei]; }
        }
        float v[NV];
#pragma unroll
        for (int q = 0; q < RW; ++q)
#pragma unroll
            for (int k = 0; k < TT; ++k) {
                float acc = dot8h(wa[q], lo4[k][0][lane], lo4[k][1][lane]);
                if (lane < LR / 8 - 32) acc += dot8h(wb[q], lo4[k][0][32 + lane], lo4[k][1][32 + lane]);
                v[q * TT + k] = acc;
            }
#pragma unroll
        for (int e = TT * RW; e < NV; ++e) v[e] = 0.0f;
        xor_scatter<NV>(v, lane);
        if (epi) {
            float x = rv;
            if (apply) {
                x = fmaf(bo, 2.0f * sigmoidf_(ip / (float) HC), x);
                m.a[vk].R_out[ei] = x;
            }
            const float y = x * wn * rsc;
            const float gate = sigmoidf_(v[0]);
            g[vk][ec][edd] = y * gate;
            gs[vk][ec][edd] = gate;
            ge[vk][ec][edd] = x * rsc * en * eg;
        }
    }
    __syncthreads();
    for (int i = t; i < TT * UPM_COLS; i += THREADS) {
        const int k = i / UPM_COLS, col = i - k * UPM_COLS;
<<<<<<< this fork
        float sum = 0.0f;
#pragma unroll
        for (int c = 0; c < HC; ++c) sum += g[k][c][col];
        m.a[k].mixed[d0 + col] = sum / (float) HC;
    }
    if (m.a[0].q8_mixed != nullptr) gr_q8_tail(m, d0);   // S26 STRATA_QFUSE
}


// STRATA_HC_Q8 on HIP gfx12: gr_up_q8_kernel with 32 columns / 512 threads a block, so a block owns a whole q8_1 group
// (STRATA_QFUSE_HC: no counter, no fence; the same arithmetic per value as gr_up_q8_kernel)
template <int T>
__global__ void __launch_bounds__(Q32_THREADS) gr_up_q8_q32_kernel(GrMulti m, const float* __restrict__ part,
                                                           const float* __restrict__ ssg) {
    __shared__ __align__(16) float lo[T][LR];
    __shared__ float rsS[T][HC];
    __shared__ float g[T][HC][Q32_COLS];
    const int t = threadIdx.x, lane = t & 31, warp = t >> 5;
    const int d0 = blockIdx.x * Q32_COLS;
    if (t < T * HC) {
        const int k = t / HC, c = t - k * HC;
        float ss = 0.0f;
#pragma unroll
        for (int j = 0; j < Q8_CPS; ++j) ss += ssg[k * Q8_NKC + c * Q8_CPS + j];
        const float r = rsqrtf(ss / (float) N + m.a[k].eps);
        rsS[k][c] = r;
        if (blockIdx.x == 0) m.a[k].rs[c] = r;
    }
    __syncthreads();
    for (int i = t; i < T * PR; i += Q32_THREADS) {
        const int k = i / PR, r = i - k * PR;
        if (r >= LR && (blockIdx.x != 0 || m.a[k].w_inject == nullptr)) continue;
        float sum = 0.0f;
#pragma unroll
        for (int c = 0; c < HC; ++c) {
            float p = 0.0f;
#pragma unroll
            for (int j = 0; j < Q8_CPS; ++j) p += part[((size_t) k * Q8_NKC + c * Q8_CPS + j) * PR + r];
            sum = fmaf(rsS[k][c], p, sum);
        }
        if (r < LR) {
            const float x = sum / (float) HC;
            lo[k][r] = x / (1.0f + __expf(-x));
        } else {
            m.a[k].inject_out[r - LR] = sum;
        }
    }
    __syncthreads();
    constexpr int RPW8 = HC * Q32_COLS / Q32_WARPS;     // 8 rows per warp
#pragma unroll 1
    for (int qq = 0; qq < RPW8; ++qq) {
        const int r = warp + qq * Q32_WARPS;
        const int c = r / Q32_COLS, dd = r - c * Q32_COLS, i = c * N + d0 + dd;
        const uint8_t* row = m.a[0].q8_up + (size_t) i * (LR / 32) * Q8B;
        float4 w[3];
#pragma unroll
        for (int s = 0; s < 3; ++s) {   // lane: values 4 lane + 128 s (the third step: lanes 0-15)
            const int v = 4 * lane + 128 * s;
            if (v < LR) {
                const uint8_t* blk = row + (v >> 5) * Q8B;
                w[s] = q8_four(q8_ld16(blk + 2 + (v & 31)) | q8_ld16(blk + 4 + (v & 31)) << 16, q8_half(q8_ld16(blk)));
            } else {
                w[s] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            }
        }
        float rv = 0.0f, wn = 0.0f, bo = 0.0f, ip = 0.0f;
        bool apply = false;
        if (lane < T) {
            const FusedGrArgs& a = m.a[lane];
            rv = a.R[i];
            wn = a.w_norm[i];
            apply = a.apply;
            if (apply) { bo = a.bo_prev[d0 + dd]; ip = a.inj_prev[c]; }
        }
        float mine = 0.0f;
#pragma unroll
        for (int k = 0; k < T; ++k) {
            float acc = 0.0f;
#pragma unroll
            for (int s = 0; s < 3; ++s) {
                const int v = 4 * lane + 128 * s;
                if (v < LR) {
                    const float4 x = *reinterpret_cast<const float4*>(&lo[k][v]);
                    acc = fmaf(w[s].x, x.x, acc); acc = fmaf(w[s].y, x.y, acc);
                    acc = fmaf(w[s].z, x.z, acc); acc = fmaf(w[s].w, x.w, acc);
                }
            }
            acc = warp_sum(acc);
            if (lane == k) mine = acc;
        }
        if (lane < T) {
            if (apply) {
                rv = fmaf(bo, 2.0f * sigmoidf_(ip / (float) HC), rv);
                m.a[lane].R_out[i] = rv;
            }
            const float x = rv * wn * rsS[lane][c];
            g[lane][c][dd] = x * sigmoidf_(mine);
        }
    }
    __syncthreads();
    for (int i = t; i < T * Q32_COLS; i += Q32_THREADS) {
        const int k = i / Q32_COLS, col = i - k * Q32_COLS;
        float sum = 0.0f;
#pragma unroll
        for (int c = 0; c < HC; ++c) sum += g[k][c][col];
        m.a[k].mixed[d0 + col] = sum / (float) HC;
    }
    __syncthreads();
    // the block owns the whole q8_1 group (32 columns): quantize it here (gr_up_q32_kernel's tail), no counter, no fence
    if (m.a[0].q8_mixed != nullptr && warp < T) {
        const float xi = m.a[warp].mixed[d0 + lane];
        float amax = fabsf(xi), sum = xi;
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) sum += __shfl_xor_sync(0xffffffffu, sum, o);
        const float d = q8_1_finite(amax / 127.0f);
        const int8_t q = q8_1_quant(xi, d, amax);
        GrQ81* y = reinterpret_cast<GrQ81*>(m.a[warp].q8_mixed) + d0 / 32;
        y->qs[lane] = q;
        if (lane == 0) y->ds = q8_1_ds(d, sum);
    }
}



// S26 STRATA_TSUM=1: gr_down_q8_kernel / gr_up_q8_kernel with each row's T per-token warp sums (and the sums of
// squares) as one transposed butterfly (s26_tsum.cuh): bitwise the same values (S26 ws4 harness, every output equal,
// T 1-4: 1.00 / 1.03 / 1.03 / 1.15x per read). PF (all rows' weights in registers before the part / lo phase) was
// slower (0.92-1.03x) and is off.
using s26ts::tsum;
using s26ts::tsum_token;
using s26ts::tsum_lane;
using s26ts::pow2_ceil;
// gr_down_q8_kernel / gr_up_q8_kernel with every per-token warp sum done as one transposed butterfly per row (tsum)
template <int T, int RPW = Q8_RPW>
__global__ void __launch_bounds__(THREADS) gr_down_q8_fast_kernel(GrMulti m, float* __restrict__ part, float* __restrict__ ssg) {
    constexpr int P = pow2_ceil(T);
    constexpr int RG_ = LR / (WARPS * RPW);
    __shared__ __align__(16) float xs[T][Q8_KC];
    __shared__ float red[WARPS][T];
    const int t = threadIdx.x, lane = t & 31, warp = t >> 5;
    const int rg = blockIdx.x, kc = blockIdx.y, c = kc / Q8_CPS;
    const bool inj = rg == RG_;
    const bool inj_bf16 = inj && m.a[0].q8_inject == nullptr;
    const int nrows = inj ? (((m.a[0].q8_inject != nullptr || m.a[0].w_inject != nullptr) && warp < HC) ? 1 : 0) : RPW;
    const int row0 = inj ? warp : (rg * WARPS + warp) * RPW;
    const uint8_t* wb = inj ? m.a[0].q8_inject : m.a[0].q8_down;
    const int sub = lane & 7, bl = lane >> 3;
    const int tk = tsum_token<P>(lane);
    const bool owner = lane == tsum_lane<P>(tk) && tk < T;
    uint32_t q[RPW][Q8_SPB];
    float dq[RPW][Q8_SPB];
#pragma unroll
    for (int r = 0; r < RPW; ++r) {
        if (r >= nrows || inj_bf16) break;
        const uint8_t* row = wb + (size_t) (row0 + r) * (D / 32) * Q8B + (size_t) kc * (Q8_KC / 32) * Q8B;
#pragma unroll
        for (int s = 0; s < Q8_SPB; ++s) {
            const uint8_t* blk = row + (4 * s + bl) * Q8B;
            dq[r][s] = q8_half(q8_ld16(blk));
            q[r][s] = q8_ld16(blk + 2 + 4 * sub) | q8_ld16(blk + 4 + 4 * sub) << 16;
        }
    }
    float ssp[P];
#pragma unroll
    for (int k = 0; k < P; ++k) ssp[k] = 0.0f;
#pragma unroll
    for (int k = 0; k < T; ++k) {
        const FusedGrArgs& a = m.a[k];
        const float gw = a.apply ? 2.0f * sigmoidf_(a.inj_prev[c] / (float) HC) : 0.0f;
        for (int i = t; i < Q8_KC / 4; i += THREADS) {
            const int col = kc * Q8_KC + 4 * i;
            float4 r = *reinterpret_cast<const float4*>(a.R + col);
            if (a.apply) {
                const float4 b = *reinterpret_cast<const float4*>(a.bo_prev + (col - c * N));
                r.x = fmaf(b.x, gw, r.x); r.y = fmaf(b.y, gw, r.y);
                r.z = fmaf(b.z, gw, r.z); r.w = fmaf(b.w, gw, r.w);
            }
            const float4 g = *reinterpret_cast<const float4*>(a.w_norm + col);
            ssp[k] += r.x * r.x + r.y * r.y + r.z * r.z + r.w * r.w;
            *reinterpret_cast<float4*>(&xs[k][4 * i]) = make_float4(r.x * g.x, r.y * g.y, r.z * g.z, r.w * g.w);
        }
    }
    {
        const float v = tsum<P>(ssp, lane);
        if (owner) red[warp][tk] = v;
    }
    __syncthreads();
    if (rg == 0 && t < T) {
=======
>>>>>>> their fork (github.com-eddoursul-Strata-custom)
        float s = 0.0f;
#pragma unroll
        for (int c = 0; c < HC; ++c) s += g[k][c][col];
        m.a[k].mixed[d0 + col] = s / (float) HC;
    }
    if (est)
        for (int i = t; i < TT * UPM_COLS; i += THREADS) {
            const int k = i / UPM_COLS, col = i - k * UPM_COLS;
            float s = 0.0f;
#pragma unroll
            for (int c = 0; c < HC; ++c) s += ge[k][c][col];
            m.a[k].est[d0 + col] = s / (float) HC;
        }
    if (m.a[0].gate_ema != nullptr && t < HC * UPM_COLS) {
        const int c = t / UPM_COLS, col = t - c * UPM_COLS;
        float s = 0.0f;
#pragma unroll
<<<<<<< this fork
        for (int c = 0; c < HC; ++c) sum += g[k][c][col];
        m.a[k].mixed[d0 + col] = sum / (float) HC;
    }
    if (m.a[0].q8_mixed != nullptr) gr_q8_tail(m, d0);   // S26 STRATA_QFUSE
}

template <int T> void launch_q8_t(const GrMulti& m, float* part, float* ssg, cudaStream_t st, unsigned long long* stamp_buf,
                                  int stamp_i0) {
    static const bool ts = [] { const char* v = std::getenv("STRATA_TSUM"); return v && v[0] == '1'; }();
    if (stamp_buf) gpu_stamp(stamp_buf, stamp_i0, (void*) st);
    if (ts) gr_down_q8_fast_kernel<T, 4><<<dim3(LR / (WARPS * 4) + 1, Q8_NKC), THREADS, 0, st>>>(m, part, ssg);
    else gr_down_q8_kernel<T><<<dim3(Q8_RG + 1, Q8_NKC), THREADS, 0, st>>>(m, part, ssg);
    if (stamp_buf) gpu_stamp(stamp_buf, stamp_i0 + 1, (void*) st);
#if !STRATA_GR_FAST_BUILD
    if (m.a[0].q8_mixed != nullptr && qfuse_hc_local()) {   // HIP (not gfx906): the block owns a q8_1 group
        gr_up_q8_q32_kernel<T><<<Q32_BLOCKS, Q32_THREADS, 0, st>>>(m, part, ssg);
        return;
    }
#endif
    if (ts) gr_up_q8_fast_kernel<T, false><<<UPM_BLOCKS, THREADS, 0, st>>>(m, part, ssg);
    else gr_up_q8_kernel<T><<<UPM_BLOCKS, THREADS, 0, st>>>(m, part, ssg);
}
void launch_q8(const GrMulti& m, float* scratch, cudaStream_t st, unsigned long long* stamp_buf, int stamp_i0) {
    float* part = scratch;
    float* ssg = scratch + (size_t) m.T * Q8_NKC * PR;
    switch (m.T) {
        case 1: launch_q8_t<1>(m, part, ssg, st, stamp_buf, stamp_i0); break;
        case 2: launch_q8_t<2>(m, part, ssg, st, stamp_buf, stamp_i0); break;
        case 3: launch_q8_t<3>(m, part, ssg, st, stamp_buf, stamp_i0); break;
        case 4: launch_q8_t<4>(m, part, ssg, st, stamp_buf, stamp_i0); break;
        case 5: launch_q8_t<5>(m, part, ssg, st, stamp_buf, stamp_i0); break;
        case 6: launch_q8_t<6>(m, part, ssg, st, stamp_buf, stamp_i0); break;
        case 7: launch_q8_t<7>(m, part, ssg, st, stamp_buf, stamp_i0); break;
        default: launch_q8_t<8>(m, part, ssg, st, stamp_buf, stamp_i0); break;
=======
        for (int k = 0; k < TT; ++k) s += gs[k][c][col];
        float* e = m.a[0].gate_ema + (size_t) c * N + d0 + col;
        *e += 0.25f * (s / (float) TT - *e);
>>>>>>> their fork (github.com-eddoursul-Strata-custom)
    }
}

}  // namespace

size_t fused_gr_scratch_bytes() { return (size_t) SC_FLOATS * sizeof(float); }

void fused_gr_read_multi(const FusedGrArgs* a, int n_tok, float* scratch, void* stream) {
    if (n_tok < 1 || n_tok > kFusedGrMaxT || scratch == nullptr) {
        std::fprintf(stderr, "fused_gr_read_multi: invalid arguments\n");
        std::exit(1);
    }
    GrMulti m;
    for (int t = 0; t < n_tok; ++t) {
        m.a[t] = a[t];
        const FusedGrArgs& x = a[t];
<<<<<<< this fork
        if (!x.R || !x.w_norm || !((x.w_down && x.w_up) || (x.q8_down && x.q8_up)) || !x.lo || !x.rs || !x.mixed || (x.w_inject && !x.inject_out) ||
            (x.apply && (!x.bo_prev || !x.inj_prev || !x.R_out)) || x.w_down != a[0].w_down || x.w_up != a[0].w_up ||
            x.w_inject != a[0].w_inject || x.w_norm != a[0].w_norm || x.q8_down != a[0].q8_down ||
            x.q8_up != a[0].q8_up || x.q8_inject != a[0].q8_inject) {
=======
        if (!x.R || !x.w_norm || !x.w_down || !x.w_up || !x.lo || !x.rs || !x.mixed || (x.w_inject && !x.inject_out) ||
            (x.apply && (!x.bo_prev || !x.inj_prev || !x.R_out || x.inj_prev == x.inject_out)) ||
            x.w_down != a[0].w_down || x.w_up != a[0].w_up || x.w_inject != a[0].w_inject ||
            x.w_norm != a[0].w_norm || x.gate_ema != a[0].gate_ema || x.eps != a[0].eps ||
            x.est_norm != a[0].est_norm || x.est_gates != a[0].est_gates || (x.est_norm && (!x.est_gates || !x.est))) {
>>>>>>> their fork (github.com-eddoursul-Strata-custom)
            std::fprintf(stderr, "fused_gr_read_multi: invalid arguments for token %d\n", t);
            std::exit(1);
        }
    }
    m.T = n_tok;
    cudaStream_t st = (cudaStream_t) stream;
    const dim3 down_grid(DS, DG);
    for (int t0 = 0; t0 < n_tok; t0 += DT) {   // DT tokens a down launch; each token's arithmetic is the same
        GrMulti md;
        md.T = std::min(DT, n_tok - t0);
        for (int k = 0; k < md.T; ++k) md.a[k] = a[t0 + k];
        switch (md.T) {
#define STRATA_DOWN(T) case T: gr_down_split_kernel<T><<<down_grid, DTH, 0, st>>>(md, scratch); break;
            STRATA_DOWN(1) STRATA_DOWN(2) STRATA_DOWN(3) STRATA_DOWN(4)
#undef STRATA_DOWN
        }
    }
    switch (n_tok) {
#define STRATA_UP(T) case T: gr_up_multi_kernel<T><<<UPM_BLOCKS, THREADS, 0, st>>>(m); break;
        STRATA_UP(1) STRATA_UP(2) STRATA_UP(3) STRATA_UP(4) STRATA_UP(5) STRATA_UP(6) STRATA_UP(7) STRATA_UP(8)
#undef STRATA_UP
    }
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        std::fprintf(stderr, "fused_gr_read_multi: %s\n", cudaGetErrorString(e));
        std::exit(1);
    }
}

// STRATA_HC_Q8 on a bf16 pack: one warp a Q8_0 block, ggml's reference rounding (d = amax / 127 as fp16, q = round(x * (1 / d)))
__global__ void __launch_bounds__(256) gr_quantize_q8_0_kernel(const uint16_t* __restrict__ w, uint8_t* __restrict__ out, int64_t nb) {
    const int64_t b = (int64_t) blockIdx.x * 8 + (threadIdx.x >> 5);
    const int lane = threadIdx.x & 31;
    if (b >= nb) return;
    const float v = __uint_as_float((uint32_t) w[b * 32 + lane] << 16);
    float amax = fabsf(v);
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));
    const float d = amax / 127.0f, id = d != 0.0f ? 1.0f / d : 0.0f;
    uint8_t* o = out + b * 34;
    if (lane == 0) {
        const unsigned short h = __half_as_ushort(__float2half_rn(d));
        o[0] = (uint8_t) (h & 255u);
        o[1] = (uint8_t) (h >> 8);
    }
    o[2 + lane] = (uint8_t) (int8_t) lrintf(roundf(v * id));
}

__global__ void __launch_bounds__(256) gr_dequant_q8_0_kernel(const uint8_t* __restrict__ q8, uint16_t* __restrict__ out, int64_t nb) {
    const int64_t b = (int64_t) blockIdx.x * 8 + (threadIdx.x >> 5);
    const int lane = threadIdx.x & 31;
    if (b >= nb) return;
    const uint8_t* p = q8 + b * 34;
    const float d = __half2float(__ushort_as_half((unsigned short) (p[0] | (p[1] << 8))));
    out[b * 32 + lane] = bf16_from_f32(d * (float) (int8_t) p[2 + lane]);
}

void fused_gr_dequant_q8_0(const uint8_t* q8, uint16_t* bf16, int64_t n_blocks, void* stream) {
    gr_dequant_q8_0_kernel<<<(unsigned) ((n_blocks + 7) / 8), 256, 0, (cudaStream_t) stream>>>(q8, bf16, n_blocks);
}

void fused_gr_quantize_q8_0(const uint16_t* bf16, uint8_t* out, int64_t n_blocks, void* stream) {
    gr_quantize_q8_0_kernel<<<(unsigned) ((n_blocks + 7) / 8), 256, 0, (cudaStream_t) stream>>>(bf16, out, n_blocks);
}

bool fused_gr_supported(int64_t n_embd, int64_t hc, int64_t hc_lr) {
    return n_embd == N && hc == HC && hc_lr == LR;
}

}  // namespace strata::kernels
