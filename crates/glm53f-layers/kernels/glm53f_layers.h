// glm53f-layers: C ABI of the GLM-5.3-Flash coordinator layer kernels (mHC, router,
// RMSNorm, SwiGLU, FP8 block-128 projection GEMMs).
//
// Conventions for every entry point:
// - returns a cudaError_t value (0 = success); invalid arguments return
//   cudaErrorInvalidValue (1) before anything is launched;
// - launches on `stream`, allocates nothing and never synchronizes;
// - BF16 tensors are uint16_t bit patterns, E4M3 tensors are uint8_t codes (the `fn`
//   variant: no infinities, 0x7F/0xFF = NaN, largest finite 448);
// - all pointers are device pointers, 16-byte aligned; outputs are disjoint from inputs;
// - a row's result never depends on the other rows of the launch (every kernel computes
//   each row with the same operations whatever `rows` is).
//
// The CPU reference in src/ models each kernel's arithmetic operation by operation; the
// tests compare them bit for bit (the tensor-core prefill GEMM within a bound).
#pragma once
#include <stdint.h>
#include <cuda_runtime.h>

#ifdef __cplusplus
extern "C" {
#endif

// ---- mHC (4 streams; hidden % 128 == 0; streams are [rows][4][hidden] BF16) -------------

// Copy each embedding row [rows][hidden] into 4 identical streams [rows][4][hidden].
int32_t glm53f_hc_broadcast(const uint16_t* embed, uint16_t* streams, int32_t rows,
                            int32_t hidden, cudaStream_t stream);

// Slice partials of a boundary's projection: partials[rows][hidden/128][25] (24
// projections, then the sum of squares) of `streams` against `fn` [24][4*hidden] BF16.
//
// Expand fused in front (when `block_out` is non-null): first
//   streams_out[i] = bf16(bf16(bf16(post[i]) * h) + bf16(sum_j bf16(comb[j][i]) * streams_in[j]))
// with h = block_out (or bf16(block_out + block_out2)), post [rows][4] and comb [rows][16]
// (row-major [source][destination]) from the previous boundary; the projection then reads
// streams_out. Without `block_out`, streams_out must be null and the projection reads
// streams_in. `fn` may be null only when expanding (then partials is unused).
int32_t glm53f_hc_project(const uint16_t* streams_in, const uint16_t* block_out,
                          const uint16_t* block_out2, const float* post, const float* comb,
                          uint16_t* streams_out, const uint16_t* fn, float* partials,
                          int32_t rows, int32_t hidden, cudaStream_t stream);

// Finish a boundary: sum the slice partials, apply the RMS scale (eps 1e-5), compute
// pre [rows][4], post [rows][4] and comb [rows][16] (sigmoid, softmax + Sinkhorn 20,
// hc_eps 1e-6), collapse `streams` [rows][4][hidden] with pre into `collapsed`
// [rows][hidden] (BF16), and, when `norm_weight` is non-null, apply the sublayer's
// weighted RMSNorm into `normed` [rows][hidden]; when `normed_q` is non-null (hidden %
// 256 == 0), also quantize the normed row per 128-group to E4M3 (`normed_q`) with f32
// scales `normed_scales` [rows][hidden/128], the W8A8 input of the sublayer's FP8
// projections. Any of pre/post/comb/collapsed may be null. `comb` is computed by a
// separate warp, off the collapse's critical path; it is needed only by the next expansion.
int32_t glm53f_hc_finish(const float* partials, const float* base, const float* scale,
                         const uint16_t* streams, const uint16_t* norm_weight, float* pre,
                         float* post, float* comb, uint16_t* collapsed, uint16_t* normed,
                         uint8_t* normed_q, float* normed_scales, int32_t rows,
                         int32_t hidden, cudaStream_t stream);

// The model's end: out = RMSNorm(mean of the 4 streams) with `norm_weight`. With
// `block_out` non-null, the last sublayer's expansion (as in glm53f_hc_project) is applied
// first; its streams are not stored.
int32_t glm53f_hc_head(const uint16_t* streams, const uint16_t* block_out,
                       const uint16_t* block_out2, const float* post, const float* comb,
                       const uint16_t* norm_weight, uint16_t* out, int32_t rows,
                       int32_t hidden, cudaStream_t stream);

// ---- Router (sigmoid scores, bias-corrected top-k) ---------------------------------------

// logits [rows][experts] (f32) = x [rows][hidden] (BF16) . weight [experts][hidden] (BF16).
// hidden % 256 == 0.
int32_t glm53f_router_logits(const uint16_t* x, const uint16_t* weight, float* logits,
                             int32_t rows, int32_t experts, int32_t hidden,
                             cudaStream_t stream);

// ids [rows][top_k] and weights [rows][top_k]: experts in descending order of
// sigmoid(logit) + bias (ties to the lower index); weights = score / (sum + 1e-20) * scale.
// experts <= 1024, top_k <= 32.
int32_t glm53f_router_select(const float* logits, const float* bias, int32_t* ids,
                             float* weights, int32_t rows, int32_t experts, int32_t top_k,
                             float scale, cudaStream_t stream);

// ---- Elementwise ------------------------------------------------------------------------

// Weighted RMSNorm (eps 1e-5): out = bf16(w * bf16(x * r)). hidden % 8 == 0.
int32_t glm53f_rmsnorm(const uint16_t* x, const uint16_t* weight, uint16_t* out,
                       int32_t rows, int32_t hidden, cudaStream_t stream);

// Per-row, per-128-group E4M3 quantization: scale = amax / 448 (1 for an all-zero
// group), q = e4m3_rn_satfinite(x / scale). cols % 128 == 0.
int32_t glm53f_act_quant(const uint16_t* x, uint8_t* q, float* scales, int32_t rows,
                         int32_t cols, cudaStream_t stream);

// SwiGLU with the clamp (gate <= 10, -10 <= up <= 10) of gate_up [rows][2*inter]
// (gate columns first): act = bf16(bf16(silu(gate)) * up). Writes `act` [rows][inter] when
// non-null, and the down projection's W8A8 input (`q` [rows][inter], `scales`
// [rows][inter/128]) when `q` is non-null. inter % 128 == 0.
int32_t glm53f_swiglu(const uint16_t* gate_up, uint16_t* act, uint8_t* q, float* scales,
                      int32_t rows, int32_t inter, cudaStream_t stream);

// ---- FP8 block-128 projections: out [rows][n] = x [rows][k] . W^T, W [n][k] E4M3 with
// f32 scales [n/128][k/128]; n % 128 == 0, k % 128 == 0. ------------------------------------

// Decode GEMM, 1 <= rows <= 8, bandwidth-bound. a8 = 0: x is BF16 [rows][k]; a8 = 1: x is
// E4M3 [rows][k] with scales x_scales [rows][k/128] (glm53f_act_quant). ksplit divides
// k/128 (use the Rust `decode_ksplit(n, k)`, a function of the shape only). ksplit == 1
// writes BF16 `out`; ksplit > 1 writes f32 `partials` [ksplit][rows][n] for
// glm53f_splitk_reduce.
int32_t glm53f_fp8_gemm_decode(const void* x, const float* x_scales, int32_t a8,
                               const uint8_t* w, const float* w_scales, int32_t rows,
                               int32_t n, int32_t k, int32_t ksplit, float* partials,
                               uint16_t* out, cudaStream_t stream);

// out [rows][n] (BF16) = sum over splits, in split order, of partials [ksplit][rows][n].
int32_t glm53f_splitk_reduce(const float* partials, uint16_t* out, int32_t ksplit,
                             int32_t rows, int32_t n, cudaStream_t stream);

// Prefill GEMM (W8A8, FP8 tensor cores: mma.m16n8k32 e4m3). Each 128-wide K block is
// accumulated separately and added to the f32 output with fma(block, sx * sw, acc). x is
// E4M3 [rows][k] with x_scales [rows][k/128]. Writes BF16 `out` and, when non-null, the
// f32 values `out_f32` [rows][n]. Any rows >= 1.
//
// The tensor cores' FP8 accumulation keeps fewer bits than f32 (measured on sm_89: mean
// error 4e-6 of sum|x w|, worst 1.6e-4, over a 128-block). flags = 0 accumulates each
// 128-block inside the tensor core, as the reference's block-FP8 kernels do;
// GLM53F_PREFILL_PROMOTE_K32 adds every k32 product sum to the block sum in f32 instead
// (2.3x smaller error, about 9% slower on sm_89).
#define GLM53F_PREFILL_PROMOTE_K32 1
int32_t glm53f_fp8_gemm_prefill(const uint8_t* xq, const float* x_scales, const uint8_t* w,
                                const float* w_scales, int32_t rows, int32_t n, int32_t k,
                                int32_t flags, uint16_t* out, float* out_f32,
                                cudaStream_t stream);

// Shared memory the prefill GEMM needs (bytes); it opts in on first use.
int32_t glm53f_fp8_gemm_prefill_smem_bytes(void);

#ifdef __cplusplus
}
#endif
