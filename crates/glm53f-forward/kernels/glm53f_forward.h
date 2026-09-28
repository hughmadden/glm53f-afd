// glm53f-forward: C ABI of the forward's own kernels (the BF16 weight GEMV and the glue
// between the kernel crates).
//
// Conventions for every entry point:
// - returns a cudaError_t value (0 = success); invalid arguments return
//   cudaErrorInvalidValue (1) before anything is launched;
// - launches on `stream`, allocates nothing and never synchronizes;
// - BF16 tensors are uint16_t bit patterns; all pointers are device pointers (or host
//   pointers mapped into the device's address space where stated);
// - strides are in elements.
#pragma once
#include <stdint.h>
#include <cuda_runtime.h>

#ifdef __cplusplus
extern "C" {
#endif

// ---- BF16 weight GEMV (1 <= rows <= 8) --------------------------------------------------------
//
// out[g][m][o] = sum_k x[g][m][k] * w[g][o][k] for groups g < groups, rows m < rows, outputs
// o < n: x at x + g * x_gstride + m * ldx, w row o at w + g * w_gstride + o * ldw, out at
// out + g * o_gstride + m * ldo (BF16, or f32 when out_f32 != 0).
//
// Bandwidth-bound on CUDA cores. A CTA of 4 warps owns 8 outputs (2 per warp). Lane l of a warp
// streams 16 weight bytes (8 values) of each of its 2 rows at k = k_lo + 8 l + 256 i and
// accumulates fma(x, w, acc) in that order; lanes reduce by butterfly; K splits (grid.y) are
// added in split order by the last CTA of each 8-output block (a counter in `sync`, at least
// groups * n / 8 zeroed counters, left zeroed). A row's result is a function of that row's
// inputs and of (n, k) only, never of `rows`: a verify window gives the bits of serial steps.
//
// n % 8 == 0, k % 8 == 0, ldx, ldw, x_gstride, w_gstride % 8 == 0, pointers 16-byte aligned;
// ksplit divides k / 8 and each split is a multiple of 8 wide (use glm53f_fwd_gemv_ksplit).
// ksplit > 1 needs `partials` (f32 [groups][ksplit][rows][n]) and `sync`.
int32_t glm53f_fwd_gemv_bf16(const uint16_t* x, int64_t ldx, int64_t x_gstride,
                             const uint16_t* w, int64_t ldw, int64_t w_gstride,
                             int32_t groups, int32_t rows, int32_t n, int32_t k, int32_t ksplit,
                             float* partials, uint32_t* sync,
                             void* out, int32_t out_f32, int64_t ldo, int64_t o_gstride,
                             cudaStream_t stream);

// The K splits the GEMV uses for an [n][k] weight in `groups` groups: the fewest splits, each
// at least 1,024 wide and a multiple of 256, that give at least 512 CTAs. Shape only.
int32_t glm53f_fwd_gemv_ksplit(int32_t groups, int32_t n, int32_t k);

// ---- Embedding ------------------------------------------------------------------------------
//
// streams[r][s][:] = table[ids[r]][:] for s < 4 (the 4 identical mHC streams a token starts
// with), and rows_out[r][:] too when rows_out is non-null. `table` [vocab][hidden] BF16 may be
// page-locked host memory mapped into the device (only the rows used cross PCIe). An id
// outside [0, vocab) writes zeros. hidden % 8 == 0.
int32_t glm53f_fwd_embed_gather(const uint16_t* table, int64_t vocab, const int32_t* ids,
                                int32_t rows, int32_t hidden, uint16_t* streams, uint16_t* rows_out,
                                cudaStream_t stream);

// ---- Conversions ------------------------------------------------------------------------------

// out[r][c] = f32(in[r][c]) * scale (IEEE multiply; scale 1 is exact), r < rows, c < cols.
int32_t glm53f_fwd_bf16_to_f32(const uint16_t* in, int64_t ldi, float* out, int64_t ldo,
                               int32_t rows, int32_t cols, float scale, cudaStream_t stream);

// out[r][c] = bf16(in[r][c]) (round to nearest even), r < rows, c < cols.
int32_t glm53f_fwd_f32_to_bf16(const float* in, int64_t ldi, uint16_t* out, int64_t ldo,
                               int32_t rows, int32_t cols, cudaStream_t stream);

// ---- Selection ----------------------------------------------------------------------------------

// ids[r] = the first index of the largest value among logits[r][0 .. n_valid) (NaN never
// wins; all NaN gives 0), vals[r] its value (vals may be null).
int32_t glm53f_fwd_argmax(const float* logits, int64_t ld, int32_t rows, int32_t n_valid,
                          int32_t* ids, float* vals, cudaStream_t stream);

// ---- Row gather and scatter ------------------------------------------------------------------

// dst[i] = src[idx[i]] for i < rows: rows of `bytes` bytes, src rows `src_stride` bytes apart,
// dst rows `dst_stride` bytes apart. bytes, strides and pointers are multiples of 16.
int32_t glm53f_fwd_gather_rows(const uint8_t* src, int64_t src_stride, const int32_t* idx,
                               uint8_t* dst, int64_t dst_stride, int32_t rows, int64_t bytes,
                               cudaStream_t stream);

// dst[idx[i]] = src[i] for i < rows (same layout rules; idx entries distinct).
int32_t glm53f_fwd_scatter_rows(const uint8_t* src, int64_t src_stride, const int32_t* idx,
                                uint8_t* dst, int64_t dst_stride, int32_t rows, int64_t bytes,
                                cudaStream_t stream);

// ---- Routed-expert combine (the reference's eager expert loop) ---------------------------------
//
// out[r] = the BF16 sum over the row's slots j < top_k with ids[r][j] >= 0, in ascending
// expert id order (equal ids in slot order), of bf16(y[r][j] * weights[r][j]), rounding the
// running sum to BF16 after every addition, from zero. y [rows][top_k][hidden] BF16, ids and
// weights [rows][top_k]. hidden % 8 == 0, top_k <= 32.
int32_t glm53f_fwd_moe_combine(const uint16_t* y, const int32_t* ids, const float* weights,
                               int32_t rows, int32_t top_k, int32_t hidden, uint16_t* out,
                               cudaStream_t stream);

// ---- DFlash2 taps -----------------------------------------------------------------------------
//
// out[r][c] = bf16(((s0 + s1 + s2 + s3) summed left to right in f32) * 0.25) for r < rows,
// c < hidden, where s_j = streams[r][j][c] (the four mHC streams, BF16 [rows][4][hidden]); out
// rows are `ldo` elements apart. The mean of glm53f-layers' final head (hc_head) with the same
// order of operations; the drafter's taps are this mean of a layer's output streams.
// hidden % 8 == 0, ldo % 8 == 0, ldo >= hidden, pointers 16-byte aligned.
int32_t glm53f_fwd_stream_mean(const uint16_t* streams, int32_t rows, int32_t hidden,
                               uint16_t* out, int64_t ldo, cudaStream_t stream);

#ifdef __cplusplus
}
#endif
