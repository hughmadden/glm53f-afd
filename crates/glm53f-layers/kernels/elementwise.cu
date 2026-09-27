// Row-wise kernels: weighted RMSNorm, per-128-group E4M3 activation quantization, and
// SwiGLU with the clamp producing the down projection's input (BF16, and/or its W8A8
// quantization, so the MLP's activation never makes a separate pass).
#include "glm53f_layers.h"
#include "common.cuh"

namespace glm53f {
namespace {

// RMSNorm: 256 threads per row; thread t owns chunks t, t + 256, ... of 8 positions
// (src/norm.rs sum_squares), the warp totals added in order.
__global__ void __launch_bounds__(256) rmsnorm_kernel(const uint16_t* __restrict__ x, const uint16_t* __restrict__ nw,
                                                      uint16_t* __restrict__ out, int hidden) {
  __shared__ float scratch[9];
  const int row = blockIdx.x, tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const uint16_t* xr = x + size_t(row) * hidden;
  const int chunks = hidden >> 3;
  float sq = 0.0f;
  for (int c = tid; c < chunks; c += 256) {
    float v[8];
    unpack_bf16x8(*reinterpret_cast<const uint4*>(xr + 8 * c), v);
#pragma unroll
    for (int k = 0; k < 8; ++k) sq = __fmaf_rn(v[k], v[k], sq);
  }
  sq = warp_sum(sq);
  if (lane == 0) scratch[warp] = sq;
  __syncthreads();
  if (tid == 0) {
    float t = 0.0f;
#pragma unroll
    for (int i = 0; i < 8; ++i) t = __fadd_rn(t, scratch[i]);
    scratch[8] = t;
  }
  __syncthreads();
  const float r = rms_scale(scratch[8], float(hidden), kRmsEps);
  for (int c = tid; c < chunks; c += 256) {
    float v[8], wf[8];
    unpack_bf16x8(*reinterpret_cast<const uint4*>(xr + 8 * c), v);
    unpack_bf16x8(__ldg(reinterpret_cast<const uint4*>(nw + 8 * c)), wf);
    float o[8];
#pragma unroll
    for (int k = 0; k < 8; ++k) o[k] = __fmul_rn(wf[k], bf16_round(__fmul_rn(v[k], r)));
    uint4 ov;
    ov.x = pack_bf16x2(o[0], o[1]); ov.y = pack_bf16x2(o[2], o[3]);
    ov.z = pack_bf16x2(o[4], o[5]); ov.w = pack_bf16x2(o[6], o[7]);
    *reinterpret_cast<uint4*>(out + size_t(row) * hidden + 8 * c) = ov;
  }
}

// Quantize 4 values of a 128-group held by one lane of the group's warp.
__device__ __forceinline__ void quant4(const float (&v)[4], float amax_lane, uint8_t* __restrict__ q, float* __restrict__ s,
                                       int lane) {
  const float scale = group_scale(warp_max(amax_lane));
  uint32_t packed = 0;
#pragma unroll
  for (int i = 0; i < 4; ++i) packed |= f32_to_e4m3(__fdiv_rn(v[i], scale)) << (8 * i);
  *reinterpret_cast<uint32_t*>(q) = packed;
  if (lane == 0) *s = scale;
}

// One warp per (row, 128-group); lane l owns values 4 l .. 4 l + 3.
__global__ void __launch_bounds__(256) act_quant_kernel(const uint16_t* __restrict__ x, uint8_t* __restrict__ q,
                                                        float* __restrict__ scales, int rows, int cols) {
  const int lane = threadIdx.x & 31;
  const long gw = long(blockIdx.x) * 8 + (threadIdx.x >> 5);
  const int groups = cols >> 7;
  if (gw >= long(rows) * groups) return;
  const long row = gw / groups;
  const int g = int(gw % groups);
  const size_t at = size_t(row) * cols + g * 128 + 4 * lane;
  const uint2 a = *reinterpret_cast<const uint2*>(x + at);
  const float v[4] = {bf16_lo(a.x), bf16_hi(a.x), bf16_lo(a.y), bf16_hi(a.y)};
  const float amax = fmaxf(fmaxf(fabsf(v[0]), fabsf(v[1])), fmaxf(fabsf(v[2]), fabsf(v[3])));
  quant4(v, amax, q + at, scales + row * groups + g, lane);
}

// SwiGLU: gate_up [rows][2 inter] -> act [rows][inter] (BF16) and/or its W8A8 quantization.
__global__ void __launch_bounds__(256) swiglu_kernel(const uint16_t* __restrict__ gu, uint16_t* __restrict__ act,
                                                     uint8_t* __restrict__ q, float* __restrict__ scales, int rows,
                                                     int inter) {
  const int lane = threadIdx.x & 31;
  const long gw = long(blockIdx.x) * 8 + (threadIdx.x >> 5);
  const int groups = inter >> 7;
  if (gw >= long(rows) * groups) return;
  const long row = gw / groups;
  const int g = int(gw % groups);
  const int i0 = g * 128 + 4 * lane;
  const uint16_t* gr = gu + size_t(row) * 2 * inter;
  const uint2 ga = *reinterpret_cast<const uint2*>(gr + i0);
  const uint2 ua = *reinterpret_cast<const uint2*>(gr + inter + i0);
  const float gv[4] = {bf16_lo(ga.x), bf16_hi(ga.x), bf16_lo(ga.y), bf16_hi(ga.y)};
  const float uv[4] = {bf16_lo(ua.x), bf16_hi(ua.x), bf16_lo(ua.y), bf16_hi(ua.y)};
  float v[4];
  float amax = 0.0f;
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    v[i] = swiglu_f32(gv[i], uv[i]);
    amax = fmaxf(amax, fabsf(v[i]));
  }
  const size_t at = size_t(row) * inter + i0;
  if (act) *reinterpret_cast<uint2*>(act + at) = make_uint2(pack_bf16x2(v[0], v[1]), pack_bf16x2(v[2], v[3]));
  if (q) quant4(v, amax, q + at, scales + row * groups + g, lane);
}

}  // namespace
}  // namespace glm53f

using namespace glm53f;

extern "C" int32_t glm53f_rmsnorm(const uint16_t* x, const uint16_t* weight, uint16_t* out, int32_t rows, int32_t hidden,
                                  cudaStream_t stream) {
  if (rows < 1 || hidden < 8 || hidden % 8 || !aligned16(x) || !aligned16(weight) || !aligned16(out))
    return cudaErrorInvalidValue;
  rmsnorm_kernel<<<rows, 256, 0, stream>>>(x, weight, out, hidden);
  return cudaGetLastError();
}

extern "C" int32_t glm53f_act_quant(const uint16_t* x, uint8_t* q, float* scales, int32_t rows, int32_t cols,
                                    cudaStream_t stream) {
  if (rows < 1 || cols < 128 || cols % 128 || !aligned16(x) || !aligned16(q) || !scales) return cudaErrorInvalidValue;
  const long warps = long(rows) * (cols / 128);
  act_quant_kernel<<<unsigned((warps + 7) / 8), 256, 0, stream>>>(x, q, scales, rows, cols);
  return cudaGetLastError();
}

extern "C" int32_t glm53f_swiglu(const uint16_t* gate_up, uint16_t* act, uint8_t* q, float* scales, int32_t rows,
                                 int32_t inter, cudaStream_t stream) {
  if (rows < 1 || inter < 128 || inter % 128 || !aligned16(gate_up) || (!act && !q)) return cudaErrorInvalidValue;
  if (!aligned16_or_null(act) || !aligned16_or_null(q) || (q && !scales)) return cudaErrorInvalidValue;
  const long warps = long(rows) * (inter / 128);
  swiglu_kernel<<<unsigned((warps + 7) / 8), 256, 0, stream>>>(gate_up, act, q, scales, rows, inter);
  return cudaGetLastError();
}
