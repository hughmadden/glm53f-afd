// Router kernels: f32 logits (one warp per expert and row, or per expert and 8 rows above 8
// rows), then a per-row top-k over sigmoid scores plus the correction bias.
//
// Logits: lane l of the expert's warp takes k = 8 l + 256 i + j (16-byte loads of 8 BF16),
// fma over i then j, butterfly across lanes (src/router.rs logits_row). Above 8 rows a
// warp reuses each weight load for 8 activation rows.
//
// Selection: argmax rounds with warp shuffles; the largest corrected score wins and an
// exact tie goes to the lower expert index. The structure follows ds41rt's
// select_fast_kernel (v41_router.cu), with GLM's sigmoid scores, any expert count up to
// 1024 and top-k up to 32 (see PROVENANCE.md).
#include "glm53f_layers.h"
#include "common.cuh"
#include <math_constants.h>

namespace glm53f {
namespace {

constexpr int kRows = 8;
constexpr int kExpertsPerCta = 4;

template <int R>
__global__ void __launch_bounds__(32 * kExpertsPerCta) router_logits_kernel(
    const uint16_t* __restrict__ x, const uint16_t* __restrict__ w, float* __restrict__ logits, int rows,
    int experts, int hidden) {
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int e = blockIdx.x * kExpertsPerCta + warp;
  if (e >= experts) return;
  const int r0 = blockIdx.y * R;
  const int nr = min(R, rows - r0);
  float acc[R];
#pragma unroll
  for (int r = 0; r < R; ++r) acc[r] = 0.0f;
  const uint16_t* wr = w + size_t(e) * hidden;
  const int steps = hidden >> 8;
  // One row per warp: keep several weight loads in flight. Eight rows: registers are the limit.
  constexpr int kUnroll = R == 1 ? 4 : 1;
#pragma unroll kUnroll
  for (int i = 0; i < steps; ++i) {
    const int k = 8 * lane + 256 * i;
    float wf[8];
    unpack_bf16x8(ld_stream(wr + k), wf);
#pragma unroll
    for (int r = 0; r < R; ++r) {
      if (r < nr) {
        float xf[8];
        unpack_bf16x8(ld_cached(x + size_t(r0 + r) * hidden + k), xf);
#pragma unroll
        for (int j = 0; j < 8; ++j) acc[r] = __fmaf_rn(xf[j], wf[j], acc[r]);
      }
    }
  }
#pragma unroll
  for (int r = 0; r < R; ++r) {
    if (r < nr) {
      const float v = warp_sum(acc[r]);
      if (lane == 0) logits[size_t(r0 + r) * experts + e] = v;
    }
  }
}

__global__ void router_select_kernel(const float* __restrict__ logits, const float* __restrict__ bias,
                                     int32_t* __restrict__ ids, float* __restrict__ weights, int experts, int top_k,
                                     float scale) {
  __shared__ float warp_v[32];
  __shared__ int warp_i[32];
  __shared__ float chosen[32];
  __shared__ int winner;
  const int row = blockIdx.x, tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int nw = blockDim.x >> 5;
  float score = 0.0f, cand = -CUDART_INF_F;
  if (tid < experts) {
    score = sigmoid_f32(logits[size_t(row) * experts + tid]);
    cand = __fadd_rn(score, bias[tid]);
  }
  for (int k = 0; k < top_k; ++k) {
    float v = cand;
    int i = tid < experts ? tid : 0x7fffffff;
#pragma unroll
    for (int off = 16; off; off >>= 1) {
      const float ov = __shfl_down_sync(0xffffffffu, v, off);
      const int oi = __shfl_down_sync(0xffffffffu, i, off);
      if (ov > v || (ov == v && oi < i)) { v = ov; i = oi; }
    }
    if (lane == 0) { warp_v[warp] = v; warp_i[warp] = i; }
    __syncthreads();
    if (warp == 0) {
      v = lane < nw ? warp_v[lane] : -CUDART_INF_F;
      i = lane < nw ? warp_i[lane] : 0x7fffffff;
#pragma unroll
      for (int off = 16; off; off >>= 1) {
        const float ov = __shfl_down_sync(0xffffffffu, v, off);
        const int oi = __shfl_down_sync(0xffffffffu, i, off);
        if (ov > v || (ov == v && oi < i)) { v = ov; i = oi; }
      }
      if (lane == 0) { winner = i; ids[size_t(row) * top_k + k] = i; }
    }
    __syncthreads();
    if (tid == winner) { chosen[k] = score; cand = -CUDART_INF_F; }
    __syncthreads();
  }
  if (tid < top_k) {
    float total = 0.0f;
    for (int j = 0; j < top_k; ++j) total = __fadd_rn(total, chosen[j]);
    weights[size_t(row) * top_k + tid] = __fmul_rn(__fdiv_rn(chosen[tid], __fadd_rn(total, kNormDenomEps)), scale);
  }
}

}  // namespace
}  // namespace glm53f

using namespace glm53f;

extern "C" int32_t glm53f_router_logits(const uint16_t* x, const uint16_t* weight, float* logits, int32_t rows,
                                        int32_t experts, int32_t hidden, cudaStream_t stream) {
  if (rows < 1 || experts < 1 || hidden < 256 || hidden % 256 || !aligned16(x) || !aligned16(weight) || !logits)
    return cudaErrorInvalidValue;
  const dim3 block(32 * kExpertsPerCta);
  const int eblocks = (experts + kExpertsPerCta - 1) / kExpertsPerCta;
  // Up to 8 rows (decode): one row per warp, so the work spreads over experts x rows warps.
  // More rows: 8 rows per warp, reusing each weight load 8 times. Each row's arithmetic is
  // the same either way.
  if (rows <= kRows)
    router_logits_kernel<1><<<dim3(eblocks, rows), block, 0, stream>>>(x, weight, logits, rows, experts, hidden);
  else
    router_logits_kernel<kRows><<<dim3(eblocks, (rows + kRows - 1) / kRows), block, 0, stream>>>(x, weight, logits, rows,
                                                                                                experts, hidden);
  return cudaGetLastError();
}

extern "C" int32_t glm53f_router_select(const float* logits, const float* bias, int32_t* ids, float* weights,
                                        int32_t rows, int32_t experts, int32_t top_k, float scale,
                                        cudaStream_t stream) {
  if (rows < 1 || experts < 1 || experts > 1024 || top_k < 1 || top_k > 32 || top_k > experts || !logits || !bias ||
      !ids || !weights)
    return cudaErrorInvalidValue;
  const int threads = (experts + 31) / 32 * 32;
  router_select_kernel<<<rows, threads, 0, stream>>>(logits, bias, ids, weights, experts, top_k, scale);
  return cudaGetLastError();
}
