// FP8 block-128 projection GEMMs: out[m][n] = sum_k x[m][k] * W[n][k], with W E4M3 [n][k]
// and one f32 scale per 128 x 128 block of W (the checkpoint's weight_scale_inv).
//
// Decode (rows <= 8): bandwidth-bound, CUDA cores. A CTA of 4 warps owns 8 output rows
// (2 per warp); lane l streams 16 weight bytes at k = k_lo + 16 l + 512 i for each of its
// warp's 2 rows, forms the 16-term dot product for each activation row with fma from 0,
// and adds dot * scale into its accumulator with fma (scale = sw, or sw * sx for W8A8).
// Lanes reduce by butterfly; K splits (grid.y) go to f32 partials that
// glm53f_splitk_reduce adds in split order (src/mlp.rs fp8_linear models this exactly).
// Every product is exact in f32 (E4M3 x BF16 or E4M3 x E4M3), so only the sums round.
//
// Prefill (any rows, W8A8): FP8 tensor cores (mma.sync m16n8k32 e4m3).
// 128 x 128 output tiles, 8 warps of 64 x 32, K steps of 128 (one scale block) through a
// 3-stage cp.async pipeline with XOR-swizzled shared memory and ldmatrix fragment loads.
// Each K block accumulates into fresh registers and is then added to the output with
// fma(c, sx * sw, acc): the per-block f32 promotion of block-scaled FP8 GEMMs. The tensor
// cores accumulate FP8 products with fewer bits than f32, so the block sum is either left
// to them (the reference kernels' choice) or, with GLM53F_PREFILL_PROMOTE_K32, formed in
// f32 from the four k32 partial sums.
#include "glm53f_layers.h"
#include "common.cuh"

namespace glm53f {
namespace {

constexpr int kDecodeWarps = 4;
constexpr int kRowsPerWarp = 2;
constexpr int kDecodeRowsPerCta = kDecodeWarps * kRowsPerWarp;

// FuseReduce: the K splits are added by the last CTA of each 8-output block to finish (a
// per-block counter in `sync`), in split order, exactly as splitk_reduce_kernel adds them.
template <bool A8, int M, bool FuseReduce = false>
__global__ void __launch_bounds__(32 * kDecodeWarps) fp8_gemm_decode_kernel(
    const void* __restrict__ xv, const float* __restrict__ xs, const uint8_t* __restrict__ w,
    const float* __restrict__ ws, int n, int k, int kc, float* partials, uint16_t* __restrict__ out,
    unsigned* __restrict__ sync = nullptr) {
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int n0 = (blockIdx.x * kDecodeWarps + warp) * kRowsPerWarp;
  if (!FuseReduce && n0 >= n) return;
  if (n0 < n) {
    const int split = blockIdx.y;
    const int lo = split * kc, hi = lo + kc;
    const int kb_count = k >> 7;
    const uint8_t* w0 = w + size_t(n0) * k;
    const uint8_t* w1 = w0 + k;
    const float* ws0 = ws + size_t(n0 >> 7) * kb_count;  // both rows share a 128-row block
    float acc[kRowsPerWarp][M];
#pragma unroll
    for (int r = 0; r < kRowsPerWarp; ++r)
#pragma unroll
      for (int m = 0; m < M; ++m) acc[r][m] = 0.0f;

#pragma unroll 2
    for (int k0 = lo + 16 * lane; k0 < hi; k0 += 512) {
      float wf[kRowsPerWarp][16];
      unpack_e4m3x16(ld_stream(w0 + k0), wf[0]);
      unpack_e4m3x16(ld_stream(w1 + k0), wf[1]);
      const int kb = k0 >> 7;
      const float sw = __ldg(ws0 + kb);
#pragma unroll
      for (int m = 0; m < M; ++m) {
        float xf[16];
        if (A8) {
          unpack_e4m3x16(ld_cached(static_cast<const uint8_t*>(xv) + size_t(m) * k + k0), xf);
        } else {
          const uint16_t* xp = static_cast<const uint16_t*>(xv) + size_t(m) * k + k0;
          float a[8], b[8];
          unpack_bf16x8(ld_cached(xp), a);
          unpack_bf16x8(ld_cached(xp + 8), b);
#pragma unroll
          for (int j = 0; j < 8; ++j) { xf[j] = a[j]; xf[8 + j] = b[j]; }
        }
        const float s = A8 ? __fmul_rn(sw, __ldg(xs + size_t(m) * kb_count + kb)) : sw;
#pragma unroll
        for (int r = 0; r < kRowsPerWarp; ++r) {
          float d = 0.0f;
#pragma unroll
          for (int j = 0; j < 16; ++j) d = __fmaf_rn(xf[j], wf[r][j], d);
          acc[r][m] = __fmaf_rn(d, s, acc[r][m]);
        }
      }
    }

#pragma unroll
    for (int r = 0; r < kRowsPerWarp; ++r)
#pragma unroll
      for (int m = 0; m < M; ++m) {
        const float v = warp_sum(acc[r][m]);
        // Spread the stores over lanes.
        if (lane == ((r * M + m) & 31)) {
          if (partials)
            partials[(size_t(split) * M + m) * n + n0 + r] = v;
          else
            out[size_t(m) * n + n0 + r] = f32_to_bf16(v);
        }
      }
  }
  if (FuseReduce) {
    __shared__ int last;
    __syncthreads();
    if (threadIdx.x == 0) last = atomic_add_acq_rel(sync + blockIdx.x, 1u) == gridDim.y - 1;
    __syncthreads();
    if (!last) return;
    if (threadIdx.x == 0) sync[blockIdx.x] = 0;  // every split of this block has signalled
    const size_t elements = size_t(M) * n;
    for (int t = threadIdx.x; t < kDecodeRowsPerCta * M; t += blockDim.x) {
      const size_t i = size_t(t / kDecodeRowsPerCta) * n + blockIdx.x * kDecodeRowsPerCta + t % kDecodeRowsPerCta;
      float s = __ldcg(partials + i);
      for (int kk = 1; kk < int(gridDim.y); ++kk) s = __fadd_rn(s, __ldcg(partials + size_t(kk) * elements + i));
      out[i] = f32_to_bf16(s);
    }
  }
}

__global__ void splitk_reduce_kernel(const float* __restrict__ partials, uint16_t* __restrict__ out, int ksplit,
                                     long elements) {
  for (long i = blockIdx.x * long(blockDim.x) + threadIdx.x; i < elements; i += long(gridDim.x) * blockDim.x) {
    float s = partials[i];
    for (int k = 1; k < ksplit; ++k) s = __fadd_rn(s, partials[size_t(k) * elements + i]);
    out[i] = f32_to_bf16(s);
  }
}

// ---- Prefill: FP8 tensor cores ----------------------------------------------------------------
constexpr int kBM = 128, kBN = 128, kBK = 128, kStages = 3;
constexpr int kTileBytes = kBM * kBK;                         // 16 KB (A) and 16 KB (B)
constexpr int kStageBytes = 2 * kTileBytes + kBM * 4;         // + 128 activation scales
constexpr int kPrefillSmem = kStages * kStageBytes;           // 99,840 bytes

// Byte offset of 16-byte chunk `col` (0..7) of row `row` in a swizzled 128 x 128 tile.
__device__ __forceinline__ int swz(int row, int col) { return row * 128 + ((col ^ (row & 7)) << 4); }

template <bool PromoteK32>
__global__ void __launch_bounds__(256, 1) fp8_gemm_prefill_kernel(
    const uint8_t* __restrict__ xq, const float* __restrict__ xs, const uint8_t* __restrict__ w,
    const float* __restrict__ ws, int rows, int n, int k, uint16_t* __restrict__ out, float* __restrict__ out32) {
  extern __shared__ __align__(128) uint8_t smem[];
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int wm = warp >> 2, wn = warp & 3;
  const int m0 = blockIdx.y * kBM, n0 = blockIdx.x * kBN;
  const int kbs = k / kBK;

  auto load_stage = [&](int slot, int kb) {
    uint8_t* a = smem + slot * kStageBytes;
    uint8_t* b = a + kTileBytes;
    float* sx = reinterpret_cast<float*>(b + kTileBytes);
#pragma unroll
    for (int q = 0; q < 4; ++q) {
      const int c = tid + 256 * q, row = c >> 3, col = c & 7;
      const bool valid = m0 + row < rows;
      cp_async16(a + swz(row, col), xq + size_t(valid ? m0 + row : 0) * k + kb * kBK + col * 16, valid);
      cp_async16(b + swz(row, col), w + size_t(n0 + row) * k + kb * kBK + col * 16, true);
    }
    if (tid < kBM) {
      const bool valid = m0 + tid < rows;
      cp_async4(sx + tid, xs + size_t(valid ? m0 + tid : 0) * kbs + kb, valid);
    }
  };

  float acc[4][4][4];
#pragma unroll
  for (int i = 0; i < 4; ++i)
#pragma unroll
    for (int j = 0; j < 4; ++j)
#pragma unroll
      for (int e = 0; e < 4; ++e) acc[i][j][e] = 0.0f;

#pragma unroll
  for (int s = 0; s < kStages - 1; ++s) {
    if (s < kbs) load_stage(s, s);
    cp_async_commit();
  }

  const int g = lane >> 2;
  for (int kb = 0; kb < kbs; ++kb) {
    cp_async_wait<kStages - 2>();
    __syncthreads();
    {
      const int next = kb + kStages - 1;
      if (next < kbs) load_stage(next % kStages, next);
      cp_async_commit();
    }
    const int slot = kb % kStages;
    const uint8_t* a_s = smem + slot * kStageBytes;
    const uint8_t* b_s = a_s + kTileBytes;
    const float* sx = reinterpret_cast<const float*>(b_s + kTileBytes);

    float c[4][4][4];
#pragma unroll
    for (int i = 0; i < 4; ++i)
#pragma unroll
      for (int j = 0; j < 4; ++j)
#pragma unroll
        for (int e = 0; e < 4; ++e) c[i][j][e] = 0.0f;

#pragma unroll
    for (int ks = 0; ks < 4; ++ks) {
      uint32_t af[4][4], bf[4][2];
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        const int row = wm * 64 + i * 16 + (lane & 7) + ((lane >> 3) & 1) * 8;
        const int col = ks * 2 + (lane >> 4);
        ldmatrix_x4(af[i][0], af[i][1], af[i][2], af[i][3], a_s + swz(row, col));
      }
#pragma unroll
      for (int jj = 0; jj < 2; ++jj) {
        const int row = wn * 32 + jj * 16 + (lane & 7) + (lane >> 4) * 8;
        const int col = ks * 2 + ((lane >> 3) & 1);
        ldmatrix_x4(bf[2 * jj][0], bf[2 * jj][1], bf[2 * jj + 1][0], bf[2 * jj + 1][1], b_s + swz(row, col));
      }
      if (PromoteK32) {
        // Each k32 product sum starts from zero and is added to the block sum in f32.
#pragma unroll
        for (int i = 0; i < 4; ++i) {
          float t[4][4];
#pragma unroll
          for (int j = 0; j < 4; ++j) {
            t[j][0] = t[j][1] = t[j][2] = t[j][3] = 0.0f;
            mma_e4m3_16832(t[j], af[i], bf[j]);
          }
#pragma unroll
          for (int j = 0; j < 4; ++j)
#pragma unroll
            for (int e = 0; e < 4; ++e) c[i][j][e] = __fadd_rn(c[i][j][e], t[j][e]);
        }
      } else {
#pragma unroll
        for (int i = 0; i < 4; ++i)
#pragma unroll
          for (int j = 0; j < 4; ++j) mma_e4m3_16832(c[i][j], af[i], bf[j]);
      }
    }

    const float sw = __ldg(ws + size_t(n0 >> 7) * kbs + kb);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const int r = wm * 64 + i * 16 + g;
      const float s_lo = __fmul_rn(sx[r], sw), s_hi = __fmul_rn(sx[r + 8], sw);
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        acc[i][j][0] = __fmaf_rn(c[i][j][0], s_lo, acc[i][j][0]);
        acc[i][j][1] = __fmaf_rn(c[i][j][1], s_lo, acc[i][j][1]);
        acc[i][j][2] = __fmaf_rn(c[i][j][2], s_hi, acc[i][j][2]);
        acc[i][j][3] = __fmaf_rn(c[i][j][3], s_hi, acc[i][j][3]);
      }
    }
  }
  cp_async_wait<0>();

#pragma unroll
  for (int i = 0; i < 4; ++i)
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const int row = m0 + wm * 64 + i * 16 + g;
      const int col = n0 + wn * 32 + j * 8 + (lane & 3) * 2;
#pragma unroll
      for (int h = 0; h < 2; ++h) {
        const int rr = row + 8 * h;
        if (rr < rows) {
          *reinterpret_cast<uint32_t*>(out + size_t(rr) * n + col) = pack_bf16x2(acc[i][j][2 * h], acc[i][j][2 * h + 1]);
          if (out32) *reinterpret_cast<float2*>(out32 + size_t(rr) * n + col) = make_float2(acc[i][j][2 * h], acc[i][j][2 * h + 1]);
        }
      }
    }
}

template <bool A8>
cudaError_t launch_decode(const void* x, const float* xs, const uint8_t* w, const float* ws, int rows, int n, int k,
                          int ksplit, float* partials, uint16_t* out, cudaStream_t stream) {
  const dim3 grid((n + kDecodeRowsPerCta - 1) / kDecodeRowsPerCta, ksplit);
  const dim3 block(32 * kDecodeWarps);
  const int kc = k / ksplit;
  float* p = ksplit > 1 ? partials : nullptr;
  switch (rows) {
#define GLM53F_DECODE_CASE(M) \
  case M: fp8_gemm_decode_kernel<A8, M><<<grid, block, 0, stream>>>(x, xs, w, ws, n, k, kc, p, out); break;
    GLM53F_DECODE_CASE(1)
    GLM53F_DECODE_CASE(2)
    GLM53F_DECODE_CASE(3)
    GLM53F_DECODE_CASE(4)
    GLM53F_DECODE_CASE(5)
    GLM53F_DECODE_CASE(6)
    GLM53F_DECODE_CASE(7)
    GLM53F_DECODE_CASE(8)
#undef GLM53F_DECODE_CASE
    default: return cudaErrorInvalidValue;
  }
  return cudaGetLastError();
}

// The decode GEMM with its K splits reduced by the last CTA of each output block.
template <bool A8>
cudaError_t launch_decode_fused(const void* x, const float* xs, const uint8_t* w, const float* ws, int rows, int n,
                                int k, int ksplit, float* partials, unsigned* sync, uint16_t* out, cudaStream_t stream) {
  if (ksplit == 1) return launch_decode<A8>(x, xs, w, ws, rows, n, k, 1, nullptr, out, stream);
  const dim3 grid((n + kDecodeRowsPerCta - 1) / kDecodeRowsPerCta, ksplit);
  const dim3 block(32 * kDecodeWarps);
  const int kc = k / ksplit;
  switch (rows) {
#define GLM53F_DECODE_CASE(M)                                                                               \
  case M:                                                                                                    \
    fp8_gemm_decode_kernel<A8, M, true><<<grid, block, 0, stream>>>(x, xs, w, ws, n, k, kc, partials, out, sync); \
    break;
    GLM53F_DECODE_CASE(1)
    GLM53F_DECODE_CASE(2)
    GLM53F_DECODE_CASE(3)
    GLM53F_DECODE_CASE(4)
    GLM53F_DECODE_CASE(5)
    GLM53F_DECODE_CASE(6)
    GLM53F_DECODE_CASE(7)
    GLM53F_DECODE_CASE(8)
#undef GLM53F_DECODE_CASE
    default: return cudaErrorInvalidValue;
  }
  return cudaGetLastError();
}

}  // namespace
}  // namespace glm53f

using namespace glm53f;

extern "C" int32_t glm53f_fp8_gemm_decode(const void* x, const float* x_scales, int32_t a8, const uint8_t* w,
                                          const float* w_scales, int32_t rows, int32_t n, int32_t k, int32_t ksplit,
                                          float* partials, uint16_t* out, cudaStream_t stream) {
  if (rows < 1 || rows > 8 || n < 128 || n % 128 || k < 128 || k % 128 || ksplit < 1 || (k / 128) % ksplit)
    return cudaErrorInvalidValue;
  if (!aligned16(x) || !aligned16(w) || !w_scales || (a8 && !x_scales)) return cudaErrorInvalidValue;
  if (ksplit > 1 ? !partials : !out) return cudaErrorInvalidValue;
  return a8 ? launch_decode<true>(x, x_scales, w, w_scales, rows, n, k, ksplit, partials, out, stream)
            : launch_decode<false>(x, nullptr, w, w_scales, rows, n, k, ksplit, partials, out, stream);
}

extern "C" int32_t glm53f_fp8_gemm_decode_fused(const void* x, const float* x_scales, int32_t a8, const uint8_t* w,
                                                const float* w_scales, int32_t rows, int32_t n, int32_t k,
                                                int32_t ksplit, float* partials, uint32_t* sync, uint16_t* out,
                                                cudaStream_t stream) {
  if (rows < 1 || rows > 8 || n < 128 || n % 128 || k < 128 || k % 128 || ksplit < 1 || (k / 128) % ksplit)
    return cudaErrorInvalidValue;
  if (!aligned16(x) || !aligned16(w) || !w_scales || (a8 && !x_scales) || !out) return cudaErrorInvalidValue;
  if (ksplit > 1 && (!partials || !sync)) return cudaErrorInvalidValue;
  return a8 ? launch_decode_fused<true>(x, x_scales, w, w_scales, rows, n, k, ksplit, partials, sync, out, stream)
            : launch_decode_fused<false>(x, nullptr, w, w_scales, rows, n, k, ksplit, partials, sync, out, stream);
}

extern "C" int32_t glm53f_splitk_reduce(const float* partials, uint16_t* out, int32_t ksplit, int32_t rows, int32_t n,
                                        cudaStream_t stream) {
  if (!partials || !out || ksplit < 1 || rows < 1 || n < 1) return cudaErrorInvalidValue;
  const long elements = long(rows) * n;
  const long want = (elements + 255) / 256;
  const int blocks = int(want < 8192 ? want : 8192);
  splitk_reduce_kernel<<<blocks, 256, 0, stream>>>(partials, out, ksplit, elements);
  return cudaGetLastError();
}

extern "C" int32_t glm53f_fp8_gemm_prefill_smem_bytes(void) { return kPrefillSmem; }

extern "C" int32_t glm53f_fp8_gemm_prefill(const uint8_t* xq, const float* x_scales, const uint8_t* w,
                                           const float* w_scales, int32_t rows, int32_t n, int32_t k, int32_t flags,
                                           uint16_t* out, float* out_f32, cudaStream_t stream) {
  if (rows < 1 || n < 128 || n % 128 || k < 128 || k % 128 || !aligned16(xq) || !aligned16(w) || !x_scales ||
      !w_scales || !aligned16(out) || !aligned16_or_null(out_f32))
    return cudaErrorInvalidValue;
  if (flags & ~GLM53F_PREFILL_PROMOTE_K32) return cudaErrorInvalidValue;
  // Opt in to the large shared-memory carve-out (a host-side attribute, cheap to repeat).
  const bool k32 = (flags & GLM53F_PREFILL_PROMOTE_K32) != 0;
  const cudaError_t e = k32 ? cudaFuncSetAttribute(fp8_gemm_prefill_kernel<true>, cudaFuncAttributeMaxDynamicSharedMemorySize, kPrefillSmem)
                            : cudaFuncSetAttribute(fp8_gemm_prefill_kernel<false>, cudaFuncAttributeMaxDynamicSharedMemorySize, kPrefillSmem);
  if (e != cudaSuccess) return e;
  const dim3 grid(n / kBN, (rows + kBM - 1) / kBM);
  if (k32) fp8_gemm_prefill_kernel<true><<<grid, 256, kPrefillSmem, stream>>>(xq, x_scales, w, w_scales, rows, n, k, out, out_f32);
  else fp8_gemm_prefill_kernel<false><<<grid, 256, kPrefillSmem, stream>>>(xq, x_scales, w, w_scales, rows, n, k, out, out_f32);
  return cudaGetLastError();
}
