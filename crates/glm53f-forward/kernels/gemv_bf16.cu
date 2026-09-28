// BF16 weight GEMV for 1..8 activation rows: out[m][o] = sum_k x[m][k] * W[o][k].
//
// Bandwidth-bound on CUDA cores, modelled on glm53f-layers' FP8 decode GEMM
// (kernels/fp8_gemm.cu, same repository): a CTA of 4 warps owns 8 outputs (2 per warp); lane
// l streams 16 weight bytes (8 BF16 values) of each of its warp's 2 rows at
// k = k_lo + 8 l + 256 i, and for every activation row accumulates fma(x, w, acc) in that
// order (i outer, the 8 values inner); lanes reduce by butterfly; K splits (grid.y) go to f32
// partials that the last CTA of each 8-output block adds in split order. Every BF16 x BF16
// product is exact in f32, so only the sums round, and they round in an order fixed by the
// shape: a row's result never depends on how many rows share the launch.
//
// grid.z runs independent groups (the forget-gate and output-gate up-projections of a KDA
// layer share one launch).
#include "glm53f_forward.h"
#include "common.cuh"

namespace glm53f_fwd {
namespace {

constexpr int kWarps = 4;
constexpr int kRowsPerWarp = 2;
constexpr int kRowsPerCta = kWarps * kRowsPerWarp;
constexpr int kLaneValues = 8;                  // one 16-byte load
constexpr int kWarpStep = 32 * kLaneValues;     // 256 values of K per warp step

template <bool F32>
__device__ __forceinline__ void store_out(void* out, int64_t idx, float v) {
  if (F32)
    static_cast<float*>(out)[idx] = v;
  else
    static_cast<uint16_t*>(out)[idx] = f32_to_bf16(v);
}

template <int M, bool F32>
__global__ void __launch_bounds__(32 * kWarps) gemv_bf16_kernel(
    const uint16_t* __restrict__ x, int64_t ldx, int64_t x_gs, const uint16_t* __restrict__ w, int64_t ldw,
    int64_t w_gs, int n, int kc, float* __restrict__ partials, unsigned* __restrict__ sync, void* __restrict__ out,
    int64_t ldo, int64_t o_gs) {
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int g = blockIdx.z;
  const int split = blockIdx.y, nsplit = gridDim.y;
  const int n0 = (blockIdx.x * kWarps + warp) * kRowsPerWarp;  // n % 8 == 0: always < n
  const int lo = split * kc, hi = lo + kc;
  const uint16_t* xg = x + g * x_gs;
  const uint16_t* w0 = w + g * w_gs + int64_t(n0) * ldw;
  const uint16_t* w1 = w0 + ldw;

  float acc[kRowsPerWarp][M];
#pragma unroll
  for (int r = 0; r < kRowsPerWarp; ++r)
#pragma unroll
    for (int m = 0; m < M; ++m) acc[r][m] = 0.0f;

#pragma unroll 4
  for (int k0 = lo + kLaneValues * lane; k0 < hi; k0 += kWarpStep) {
    float wa[8], wb[8];
    unpack_bf16x8(ld_stream(w0 + k0), wa);
    unpack_bf16x8(ld_stream(w1 + k0), wb);
#pragma unroll
    for (int m = 0; m < M; ++m) {
      float xf[8];
      unpack_bf16x8(ld_cached(xg + int64_t(m) * ldx + k0), xf);
#pragma unroll
      for (int j = 0; j < 8; ++j) {
        acc[0][m] = __fmaf_rn(xf[j], wa[j], acc[0][m]);
        acc[1][m] = __fmaf_rn(xf[j], wb[j], acc[1][m]);
      }
    }
  }

  void* og = F32 ? static_cast<void*>(static_cast<float*>(out) + g * o_gs)
                 : static_cast<void*>(static_cast<uint16_t*>(out) + g * o_gs);
#pragma unroll
  for (int r = 0; r < kRowsPerWarp; ++r)
#pragma unroll
    for (int m = 0; m < M; ++m) {
      const float v = warp_sum(acc[r][m]);
      // Spread the stores over lanes.
      if (lane == ((r * M + m) & 31)) {
        if (nsplit > 1)
          partials[((int64_t(g) * nsplit + split) * M + m) * n + n0 + r] = v;
        else
          store_out<F32>(og, int64_t(m) * ldo + n0 + r, v);
      }
    }
  if (nsplit == 1) return;

  // The last CTA of this 8-output block to finish adds the splits in split order.
  __shared__ int last;
  __threadfence();
  __syncthreads();
  unsigned* counter = sync + int64_t(g) * gridDim.x + blockIdx.x;
  if (threadIdx.x == 0) last = atomicAdd(counter, 1u) == unsigned(nsplit - 1);
  __syncthreads();
  if (!last) return;
  __threadfence();
  if (threadIdx.x == 0) *counter = 0;  // every split of this block has signalled
  const int64_t split_stride = int64_t(M) * n;
  for (int t = threadIdx.x; t < kRowsPerCta * M; t += blockDim.x) {
    const int m = t / kRowsPerCta, o = blockIdx.x * kRowsPerCta + t % kRowsPerCta;
    const int64_t base = (int64_t(g) * nsplit * M + m) * n + o;
    float s = __ldcg(partials + base);
    for (int kk = 1; kk < nsplit; ++kk) s = __fadd_rn(s, __ldcg(partials + base + kk * split_stride));
    store_out<F32>(og, int64_t(m) * ldo + o, s);
  }
}

template <bool F32>
cudaError_t launch(const uint16_t* x, int64_t ldx, int64_t x_gs, const uint16_t* w, int64_t ldw, int64_t w_gs,
                   int groups, int rows, int n, int k, int ksplit, float* partials, unsigned* sync, void* out,
                   int64_t ldo, int64_t o_gs, cudaStream_t stream) {
  const dim3 grid(n / kRowsPerCta, ksplit, groups);
  const dim3 block(32 * kWarps);
  const int kc = k / ksplit;
  switch (rows) {
#define GLM53F_GEMV_CASE(M)                                                                                    \
  case M:                                                                                                      \
    gemv_bf16_kernel<M, F32><<<grid, block, 0, stream>>>(x, ldx, x_gs, w, ldw, w_gs, n, kc, partials, sync, out, \
                                                         ldo, o_gs);                                           \
    break;
    GLM53F_GEMV_CASE(1)
    GLM53F_GEMV_CASE(2)
    GLM53F_GEMV_CASE(3)
    GLM53F_GEMV_CASE(4)
    GLM53F_GEMV_CASE(5)
    GLM53F_GEMV_CASE(6)
    GLM53F_GEMV_CASE(7)
    GLM53F_GEMV_CASE(8)
#undef GLM53F_GEMV_CASE
    default:
      return cudaErrorInvalidValue;
  }
  return cudaGetLastError();
}

}  // namespace
}  // namespace glm53f_fwd

using namespace glm53f_fwd;

extern "C" int32_t glm53f_fwd_gemv_ksplit(int32_t groups, int32_t n, int32_t k) {
  if (groups < 1 || n < 8 || k < 8) return 1;
  const int64_t ctas = int64_t(n / kRowsPerCta) * groups;
  if (ctas >= 512) return 1;
  int best = 1;
  for (int d = 2; k / d >= 1024; ++d) {
    if (k % (kWarpStep * d) != 0) continue;
    best = d;
    if (ctas * d >= 512) break;
  }
  return best;
}

extern "C" int32_t glm53f_fwd_gemv_bf16(const uint16_t* x, int64_t ldx, int64_t x_gstride, const uint16_t* w,
                                        int64_t ldw, int64_t w_gstride, int32_t groups, int32_t rows, int32_t n,
                                        int32_t k, int32_t ksplit, float* partials, uint32_t* sync, void* out,
                                        int32_t out_f32, int64_t ldo, int64_t o_gstride, cudaStream_t stream) {
  if (groups < 1 || groups > 65535 || rows < 1 || rows > 8 || n < 8 || n % 8 || k < 8 || k % 8 || ksplit < 1 ||
      k % ksplit || (k / ksplit) % 8)
    return cudaErrorInvalidValue;
  if (!aligned16(x) || !aligned16(w) || !out || ldx % 8 || ldw % 8 || x_gstride % 8 || w_gstride % 8 ||
      ldx < k || ldw < k || ldo < n)
    return cudaErrorInvalidValue;
  if (ksplit > 1 && (!partials || !sync)) return cudaErrorInvalidValue;
  return out_f32 ? launch<true>(x, ldx, x_gstride, w, ldw, w_gstride, groups, rows, n, k, ksplit, partials, sync,
                                out, ldo, o_gstride, stream)
                 : launch<false>(x, ldx, x_gstride, w, ldw, w_gstride, groups, rows, n, k, ksplit, partials, sync,
                                 out, ldo, o_gstride, stream);
}
