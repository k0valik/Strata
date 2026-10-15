// src/prefill/gemm.cu - see include/strata/prefill/gemm.hpp.
#include "strata/prefill/gemm.hpp"
#include "strata/kernels/dequant_bf16.hpp"

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace strata::prefill {
namespace {

void ck(cublasStatus_t s, const char* what) {
    if (s != CUBLAS_STATUS_SUCCESS) {
        std::fprintf(stderr, "prefill gemm: %s: cuBLAS status %d\n", what, (int) s);
        std::exit(1);
    }
}

}  // namespace

Gemm::~Gemm() {
    if (handle_) cublasDestroy((cublasHandle_t) handle_);
    for (int i = 0; i < AUX; ++i) {
        if (aux_handle_[i]) cublasDestroy((cublasHandle_t) aux_handle_[i]);
        if (join_[i]) cudaEventDestroy((cudaEvent_t) join_[i]);
        if (aux_stream_[i]) {
            cudaStreamSynchronize((cudaStream_t) aux_stream_[i]);
            cudaStreamDestroy((cudaStream_t) aux_stream_[i]);
        }
    }
    if (fork_) cudaEventDestroy((cudaEvent_t) fork_);
}

bool Gemm::init(void* stream, std::string& err, bool side_streams) {
    cublasHandle_t h = nullptr;
    if (cublasCreate(&h) != CUBLAS_STATUS_SUCCESS) { err = "prefill gemm: cublasCreate failed"; return false; }
    handle_ = h;
    stream_ = stream;
    cublasSetStream(h, (cudaStream_t) stream);
    cublasSetMathMode(h, CUBLAS_DEFAULT_MATH);
    if (!side_streams) return true;
    if (cudaEventCreateWithFlags((cudaEvent_t*) &fork_, cudaEventDisableTiming) != cudaSuccess) {
        err = "prefill gemm: events";
        return false;
    }
    for (int i = 0; i < AUX; ++i) {
        cublasHandle_t a = nullptr;
        if (cudaStreamCreateWithFlags((cudaStream_t*) &aux_stream_[i], cudaStreamNonBlocking) != cudaSuccess ||
            cudaEventCreateWithFlags((cudaEvent_t*) &join_[i], cudaEventDisableTiming) != cudaSuccess ||
            cublasCreate(&a) != CUBLAS_STATUS_SUCCESS) {
            err = "prefill gemm: side streams";
            return false;
        }
        aux_handle_[i] = a;
        cublasSetStream(a, (cudaStream_t) aux_stream_[i]);
        cublasSetMathMode(a, CUBLAS_DEFAULT_MATH);
    }
    return true;
}

<<<<<<< this fork
#if !defined(__HIPCC__)
// ---- below sm_80: BF16 GEMMs without BF16 tensor cores (PR #655, #540, #395) ----------------------------------------
// Turing and Volta have no BF16 tensor cores: cublasGemmEx on CUDA_R_16BF inputs falls back to a SIMT fp32 kernel
// (magma_sgemmEx), a fifth of a 4K prompt on an RTX 2080 Ti (#655) and ~10% of a V100's prompt (#540).  BF16 -> FP16
// is exact for every value inside FP16's normal range (the 7-bit mantissa fits in 10 bits); weights and normalized
// activations sit there.  With the FP16 path, the weight is converted once per call and the activations in row
// slices, into the instance's own buffers, and the product runs as the FP16 GEMM (fp32 accumulate) on the tensor
// cores.  Finite values beyond FP16's range are clamped to +-65504 (#540) instead of becoming Inf.  A beta = 1
// product (STRATA_PREFILL_BF16X2's remainder, ~2^-9 of the original, deep in FP16's subnormal band) keeps cuBLAS.
//   Default: on for compute capability 7.0 - 7.4 (Volta: only the experimental STRATA_EXPERIMENTAL_SM60 build runs
//   there), OFF for 7.5 (RTX 20 in the ready-made engine: the sums round differently, so it is opt-in until it has
//   been gated).  STRATA_BF16_TC=1 / =0 turns it on / off on any 7.x card (=2, a test mode: on any card).
// Pascal (6.x) has no tensor cores and cuBLAS has no BF16 GEMM for it (#395, measured NOT_SUPPORTED on a P40): both
// operands are widened to fp32 by an exact shift and the product is cublasSgemm with the same fp32 accumulator.
namespace {
__global__ void bf16_to_f16_kernel(const uint16_t* __restrict__ in, __half* __restrict__ out, int64_t n) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float f = __uint_as_float((uint32_t) in[i] << 16);
        if (f > 65504.0f && !isinf(f)) f = 65504.0f;
        else if (f < -65504.0f && !isinf(f)) f = -65504.0f;
        out[i] = __float2half_rn(f);
    }
}
void bf16_to_f16(const uint16_t* in, uint16_t* out, int64_t n, cudaStream_t st) {
    if (n <= 0) return;
    bf16_to_f16_kernel<<<(unsigned) ((n + 255) / 256), 256, 0, st>>>(in, reinterpret_cast<__half*>(out), n);
}
#if defined(STRATA_EXPERIMENTAL_SM60)   // Pascal runs only the experimental build
__global__ void bf16_to_f32_kernel(const uint16_t* __restrict__ in, float* __restrict__ out, int64_t n) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = __uint_as_float((uint32_t) in[i] << 16);
}
void bf16_to_f32(const uint16_t* in, float* out, int64_t n, cudaStream_t st) {
    if (n <= 0) return;
    bf16_to_f32_kernel<<<(unsigned) ((n + 255) / 256), 256, 0, st>>>(in, out, n);
}
#endif
// The current device's compute capability as 10 * major + minor, per device (a layer split can mix cards); 0: not
// known, read as "not an old card" so a failed query keeps the cuBLAS BF16 call.
int current_cc() {
    static std::atomic<int> cc[64] = {};
    int dev = 0;
    if (cudaGetDevice(&dev) != cudaSuccess || dev < 0 || dev >= 64) { cudaGetLastError(); return 0; }
    int v = cc[dev].load(std::memory_order_relaxed);
    if (v == 0) {
        int major = 0, minor = 0;
        if (cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev) != cudaSuccess ||
            cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, dev) != cudaSuccess) {
            cudaGetLastError();
            return 0;
        }
        v = 10 * major + minor;
        cc[dev].store(v, std::memory_order_relaxed);
    }
    return v;
}
// 0: cuBLAS's BF16 GEMM, 1: through FP16 (tensor cores), 2: through FP32 (Pascal)
int bf16_path() {
    static const int forced = [] {
        const char* v = std::getenv("STRATA_BF16_TC");
        return v != nullptr && v[0] != '\0' ? std::atoi(v) : -1;
    }();
    const int cc = current_cc();
    if (forced == 2 && cc > 0) return 1;   // a test mode: the FP16 path on any card (gemm_bf16_parity on sm_80+)
    if (cc <= 0 || cc >= 80) return 0;
    if (cc < 70) return 2;
    if (forced >= 0) return forced != 0 ? 1 : 0;
    return cc < 75 ? 1 : 0;
}
bool grow(uint16_t*& p, int64_t& have, int64_t want) {   // `have`, `want`: 2-byte elements
    if (have >= want) return true;
    if (p) cudaFree(p);
    p = nullptr;
    have = 0;
    if (cudaMalloc((void**) &p, (size_t) want * 2) != cudaSuccess) { cudaGetLastError(); p = nullptr; return false; }
    have = want;
    return true;
}
constexpr int64_t kXSliceElems = 16ll << 20;   // 32 MiB of FP16 activations per slice (64 MiB as fp32)

#if defined(STRATA_EXPERIMENTAL_SM60)
// ---- Pascal, opt-in (STRATA_DENSE_F32=1): the dense projections' FP16-operand GEMMs as cublasSgemm ----------------
// Measured on a Tesla P40 (sm_61) with cuBLAS 12.4.5.8 (bench/results/2026-10-12-p40-sm61-kernels).  For FP16
// operands with FP32 sums cuBLAS has two Maxwell kernels there, maxwell_sgemm_fp16_64x64 and _32x128, and picks the
// slow one for most chunk sizes: the 10240 x 2560 qkv projection runs at 7.6 TFLOPS for T 7030-7040 and at 4.3 from
// T 7041 (49 -> 85 ms), and at 4.3-4.6 for two thirds of the T between 256 and 8192; the 12288-row query projection
// likewise.  A 7K-token prompt reads in 16.3 or 19.5 s by its token count.  cublasSgemm on FP32 copies
// (maxwell_sgemm_128x64) runs at 8.4-9.8 TFLOPS for every T from 512 (6.7+ from 256).  The copies are exact
// (FP16 -> FP32), and for the shapes and sizes taken here the two kernels add every output's products in the same
// order: the result is bitwise the cuBLAS FP16 call's (gemm_f32_parity: 5,201 chunk sizes on a P40 with cuBLAS 12.4, a
// measurement of this library, not a guarantee of cuBLAS's; below the sizes taken here - T under
// ~144 at 6144+ rows, under ~408 at 2560 rows - cuBLAS switches one of them to another kernel family and the bits
// differ, so those keep the FP16 call).  The weight goes through the BF16 path's conversion buffer (126 MiB for the
// largest, 21 MiB more than that path's own), the activations through its slices.
// Opt-in, and on Pascal (compute capability 6.x) only: the equality belongs to the cuBLAS build, so run
// gemm_f32_parity with yours before turning it on; newer cards have tensor cores for the FP16 call.
__global__ void f16_to_f32_kernel(const uint16_t* __restrict__ in, float* __restrict__ out, int64_t n) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = __half2float(__ushort_as_half(in[i]));
}
void f16_to_f32(const uint16_t* in, float* out, int64_t n, cudaStream_t st) {
    if (n <= 0) return;
    f16_to_f32_kernel<<<(unsigned) ((n + 255) / 256), 256, 0, st>>>(in, out, n);
}
int g_dense_f32 = -1;   // gemm_set_dense_f32 (the parity test: 0 / 1 on any card); -1 = STRATA_DENSE_F32
bool dense_f32_on() {
    static const bool env = [] {
        const char* v = std::getenv("STRATA_DENSE_F32");
        return v != nullptr && std::atoi(v) != 0;
    }();
    if (g_dense_f32 >= 0) return g_dense_f32 != 0;
    if (!env) return false;
    const int cc = current_cc();
    return cc >= 60 && cc < 70;
}
// The smallest T taken for a weight of N rows and K columns (0: never): the measured shapes only - the 2560-column
// qkv (10240 rows), query (12288) and gate (6144) projections and the 2560-row output projections (6144 and 4096
// columns), each from a T where the FP32 route (its conversions included) is the faster one or within ~7% of the FP16
// call (the gate projection below ~1300 tokens, the 4096-column output projection near 4096) and well above the T
// where the two stop agreeing bitwise.  Every other shape (a row slice of a weight too) keeps the FP16 call.
int64_t dense_f32_min_t(int64_t N, int64_t K) {
    if (K == 2560 && (N == 10240 || N == 12288)) return 256;
    if (K == 2560 && N == 6144) return 512;
    if (N == 2560 && (K == 6144 || K == 4096)) return 4096;
    return 0;
}
#endif
}  // namespace
#endif
#if !defined(__HIPCC__) && defined(STRATA_EXPERIMENTAL_SM60)
void gemm_set_dense_f32(int mode) { g_dense_f32 = mode; }
#else
void gemm_set_dense_f32(int) {}
#endif
bool Gemm::bf16_hcd_exact(const uint16_t* X, int64_t ldx, const uint16_t* W, float* Y, int64_t T, int64_t N, int64_t K) {
#if defined(__HIPCC__) && defined(STRATA_HIPBLASLT_AVAILABLE)
    static std::atomic<bool> told{false};
    if (!hipblaslt_state_ || N != 320 || K != 10240 || T < 1) {
        if (N == 320 && K == 10240 && !told.exchange(true))
            std::fprintf(stderr, "strata: STRATA_HCD_EXACT: no hipBLASLt tuning table is loaded, so the exact kernel has nothing to match: "
                                 "the hyper-connection down projection stays on hipBLAS\n");
        return false;
    }
    // The kernel copies hipBLASLt solution 1176 / 1177's summation order, which is only true of the library build the
    // table was calibrated with: take it only when the table makes hipBLASLt pick one of those two for this T.
    const int id = hipblaslt_solution_for(hipblaslt_state_, strata::prefill::hipblaslt::InputType::bf16, T, N, K, N, 0.0f);
    if (id != 1176 && id != 1177) {
        if (T >= 4096 && !told.exchange(true))   // (a short chunk may legitimately get another solution: not worth a line)
            std::fprintf(stderr, "strata: STRATA_HCD_EXACT: the tuning table gives the hyper-connection down projection (N 320, K 10240) "
                                 "solution %d at T %lld, not 1176 / 1177: hipBLASLt runs it (a changed table or library; "
                                 "tools/hip/gfx1151-hipblaslt-100401.txt is the one measured)\n",
                         id, (long long) T);
        return false;
    }
    return strata_pf_hcdown_exact_bf16(X, ldx, W, Y, T, N, K, stream_);
#else
    (void) X; (void) ldx; (void) W; (void) Y; (void) T; (void) N; (void) K;
    return false;
#endif
=======
void Gemm::set_buffers(uint16_t* scratch, int64_t scratch_elems, void* workspace, size_t ws_bytes) {
    // a fixed workspace, so the handles never allocate on the way: a slice for each stream
    const int ns = aux_handle_[0] ? AUX + 1 : 1;
    const size_t slice = ws_bytes / (size_t) ns / 256 * 256;
    cublasSetWorkspace((cublasHandle_t) handle_, workspace, slice);
    for (int i = 0; i + 1 < ns; ++i)
        cublasSetWorkspace((cublasHandle_t) aux_handle_[i], (uint8_t*) workspace + (size_t) (i + 1) * slice, slice);
    scratch_ = scratch;
    scratch_elems_ = scratch_elems;
>>>>>>> their fork (github.com-eddoursul-Strata-custom)
}

void Gemm::bf16(const uint16_t* X, const uint16_t* W, float* Y, int64_t T, int64_t N, int64_t K, int64_t ldy,
                float beta) {
    if (T <= 0 || N <= 0) return;
    if (ldy <= 0) ldy = N;
    const float alpha = 1.0f;
    // Column-major view: Y^T[N, T] = W[N, K] (stored K x N col-major, transposed) . X^T[K, T].
    ck(cublasGemmEx((cublasHandle_t) handle_, CUBLAS_OP_T, CUBLAS_OP_N, (int) N, (int) T, (int) K, &alpha, W,
                    CUDA_R_16BF, (int) K, X, CUDA_R_16BF, (int) K, &beta, Y, CUDA_R_32F, (int) ldy,
                    CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT),
       "cublasGemmEx");
}

void Gemm::f16(const uint16_t* X, const uint16_t* W, float* Y, int64_t T, int64_t N, int64_t K, int64_t ldy,
               float beta) {
    if (T <= 0 || N <= 0) return;
    if (ldy <= 0) ldy = N;
    const float alpha = 1.0f;
<<<<<<< this fork
#if defined(__HIPCC__) && defined(STRATA_HIPBLASLT_AVAILABLE)
    if (try_hipblaslt(hipblaslt_state_, strata::prefill::hipblaslt::InputType::f16, X, W, Y, T, N, K, ldy,
                      beta, stream_)) {
        STRATA_ABSORB_HIPBLAS_STICKY("hipBLASLt f16");
        return;
    }
#endif
#if !defined(__HIPCC__) && defined(STRATA_EXPERIMENTAL_SM60)
    // Pascal, STRATA_DENSE_F32=1: cublasSgemm on FP32 copies (dense_f32_on above: where it is bitwise the FP16 call)
    if (const int64_t min_t = beta == 0.0f ? dense_f32_min_t(N, K) : 0; min_t > 0 && T >= min_t && dense_f32_on()) {
        const int64_t n_sl = (T * K + kXSliceElems - 1) / kXSliceElems;   // even slices of at most 64 MiB as fp32
        const int64_t x_rows = (T + n_sl - 1) / n_sl;
        // grow() frees a buffer an earlier GEMM on this stream may still read
        if (tc_w_elems_ < 2 * N * K || tc_x_elems_ < 2 * x_rows * K) cudaStreamSynchronize((cudaStream_t) stream_);
        if (grow(tc_w_, tc_w_elems_, 2 * N * K) && grow(tc_x_, tc_x_elems_, 2 * x_rows * K)) {
            static bool said = false;
            if (!said) {
                said = true;
                std::fprintf(stderr, "prefill gemm: the dense projections' FP16 GEMMs run as cublasSgemm on FP32 copies "
                                     "(STRATA_DENSE_F32=1; gemm_f32_parity checks the bits against this cuBLAS)\n");
            }
            float* const wf = reinterpret_cast<float*>(tc_w_);
            float* const xf = reinterpret_cast<float*>(tc_x_);
            f16_to_f32(W, wf, N * K, (cudaStream_t) stream_);
            for (int64_t t0 = 0; t0 < T; t0 += x_rows) {
                const int64_t n = std::min<int64_t>(x_rows, T - t0);
                f16_to_f32(X + t0 * K, xf, n * K, (cudaStream_t) stream_);
                ck(cublasSgemm((cublasHandle_t) handle_, CUBLAS_OP_T, CUBLAS_OP_N, (int) N, (int) n, (int) K, &alpha, wf, (int) K,
                               xf, (int) K, &beta, Y + t0 * ldy, (int) ldy),
                   "cublasSgemm (FP16 operands, STRATA_DENSE_F32)");
            }
            return;
        }
        cudaGetLastError();   // no room for the copies: the FP16 call below
    }
#endif
#if !defined(__HIPCC__)
    const auto f16_gemm = [&](cublasGemmAlgo_t algo) {
        return cublasGemmEx((cublasHandle_t) handle_, CUBLAS_OP_T, CUBLAS_OP_N, (int) N, (int) T, (int) K, &alpha,
                            W, CUDA_R_16F, (int) K, X, CUDA_R_16F, (int) K, &beta, Y, CUDA_R_32F, (int) ldy,
                            CUBLAS_COMPUTE_32F, algo);
    };
    // Issue #1650: with a second CUDA device initialized in the process, cuBLAS 12 answers the default algorithm of a
    // few shapes - narrow ones, and the set moves with K - with CUBLAS_STATUS_INTERNAL_ERROR; a fixed algorithm keeps
    // a kernel that state does not break.  Only the shapes that fail are moved (the failing call itself is harmless);
    // the wide shapes, for which the fixed algorithms are markedly slower, stay on the default.
    const int64_t shape = ((int64_t) K << 32) | (uint32_t) N;
    cublasGemmAlgo_t algo = CUBLAS_GEMM_DEFAULT;
    for (int i = 0; i < f16_algo_n_; ++i)
        if (f16_algo_shape_[i] == shape) { algo = CUBLAS_GEMM_ALGO2; break; }
    cublasStatus_t st = f16_gemm(algo);
    if (st == CUBLAS_STATUS_INTERNAL_ERROR && algo == CUBLAS_GEMM_DEFAULT) {
        if (f16_algo_n_ < kF16AlgoSlots) f16_algo_shape_[f16_algo_n_++] = shape;
        st = f16_gemm(CUBLAS_GEMM_ALGO2);
    }
    ck(st, "cublasGemmEx f16");
#else
=======
>>>>>>> their fork (github.com-eddoursul-Strata-custom)
    ck(cublasGemmEx((cublasHandle_t) handle_, CUBLAS_OP_T, CUBLAS_OP_N, (int) N, (int) T, (int) K, &alpha, W,
                    CUDA_R_16F, (int) K, X, CUDA_R_16F, (int) K, &beta, Y, CUDA_R_32F, (int) ldy,
                    CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT),
       "cublasGemmEx f16");
}

void Gemm::f16_grouped(const uint16_t* const* X, const uint16_t* const* W, float* const* Y, const uint16_t* const* X_host,
                       const uint16_t* const* W_host, float* const* Y_host, const int* rows, int n, int64_t N, int64_t K) {
    if (n <= 0) return;
    if (grouped_) {
        // column-major, as f16(): Y_i^T[N, rows] = W_i (K x N col-major, transposed) . X_i^T[K, rows]; a group per problem
        const std::vector<cublasOperation_t> ta((size_t) n, CUBLAS_OP_T), tb((size_t) n, CUBLAS_OP_N);
        const std::vector<int> m((size_t) n, (int) N), k((size_t) n, (int) K), ld_wx((size_t) n, (int) K),
            ld_y((size_t) n, (int) N), size((size_t) n, 1);
        const std::vector<float> alpha((size_t) n, 1.0f), beta((size_t) n, 0.0f);
        const cublasStatus_t st = cublasGemmGroupedBatchedEx(
            (cublasHandle_t) handle_, ta.data(), tb.data(), m.data(), rows, k.data(), alpha.data(), (const void* const*) W,
            CUDA_R_16F, ld_wx.data(), (const void* const*) X, CUDA_R_16F, ld_wx.data(), beta.data(), (void* const*) Y,
            CUDA_R_32F, ld_y.data(), n, size.data(), CUBLAS_COMPUTE_32F);
        if (st == CUBLAS_STATUS_SUCCESS) return;
        if (st != CUBLAS_STATUS_NOT_SUPPORTED && st != CUBLAS_STATUS_INVALID_VALUE) ck(st, "cublasGemmGroupedBatchedEx");
        grouped_ = false;
        std::fprintf(stderr, "prefill gemm: cuBLAS does not group FP16 -> FP32 GEMMs here; one call per expert\n");
    }
    // round robin over the stream and its side streams, which join it again
    int64_t sum_rows = 0;
    for (int i = 0; i < n; ++i) sum_rows += rows[i];
    const int ns = aux_handle_[0] && sum_rows >= 128 * (int64_t) n ? std::min(n, AUX + 1) : 1;
    if (ns > 1) {
        cudaEventRecord((cudaEvent_t) fork_, (cudaStream_t) stream_);
        for (int s = 1; s < ns; ++s) cudaStreamWaitEvent((cudaStream_t) aux_stream_[s - 1], (cudaEvent_t) fork_, 0);
    }
    void* const main = handle_;
    for (int i = 0; i < n; ++i) {
        handle_ = i % ns == 0 ? main : aux_handle_[i % ns - 1];
        f16(X_host[i], W_host[i], Y_host[i], rows[i], N, K);
    }
    handle_ = main;
    for (int s = 1; s < ns; ++s) {
        cudaEventRecord((cudaEvent_t) join_[s - 1], (cudaStream_t) aux_stream_[s - 1]);
        cudaStreamWaitEvent((cudaStream_t) stream_, (cudaEvent_t) join_[s - 1], 0);
    }
}

void Gemm::native(const uint16_t* X, int ggml_type, const void* W_blocks, float* Y, int64_t T, int64_t N, int64_t K,
                  int64_t ldy, float beta) {
    if (N * K > scratch_elems_) {
        // Too large for the scratch at once: in row slices.
        const int64_t rows = scratch_elems_ / K;
        if (rows <= 0) { std::fprintf(stderr, "prefill gemm: scratch too small for K=%lld\n", (long long) K); std::exit(1); }
        if (ldy <= 0) ldy = N;
        for (int64_t r0 = 0; r0 < N; r0 += rows) {
            const int64_t n = (N - r0 < rows) ? N - r0 : rows;
            strata::kernels::dequant_f16(ggml_type, W_blocks, r0, n, K, scratch_, stream_);
            f16(X, scratch_, Y + r0, T, n, K, ldy, beta);
        }
        return;
    }
    strata::kernels::dequant_f16(ggml_type, W_blocks, 0, N, K, scratch_, stream_);
    f16(X, scratch_, Y, T, N, K, ldy, beta);
}

}  // namespace strata::prefill
