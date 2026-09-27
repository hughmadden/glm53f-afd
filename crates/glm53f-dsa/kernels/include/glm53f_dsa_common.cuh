// Shared device helpers for the glm53f-dsa kernels.
#pragma once

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <stdint.h>

#include "glm53f_dsa.h"

namespace glm53f {

constexpr int kPool = 4;
constexpr int kTopPools = 512;
constexpr int kMaxTokens = 2051;
constexpr int kPageTokens = 64;
constexpr int kPagePools = 16;
constexpr int kLatentDim = 512;
constexpr int kLatentBytes = 528;
constexpr int64_t kPoolCodesOffset = 33792;   // 64 * 528
constexpr int64_t kPoolScalesOffset = 35840;  // + 16 * 128
constexpr int64_t kPageLayerBytes = 35904;    // + 16 * 4
constexpr int kTailBytes = 1552;
constexpr int kTailTokenBytes = 512;

__host__ __device__ inline uint64_t align_up(uint64_t x, uint64_t a) { return (x + a - 1) / a * a; }

inline bool valid_cache(const glm53f_dsa_cache_t& c) {
  return c.base && (reinterpret_cast<uintptr_t>(c.base) % 16) == 0 && c.page_stride >= kPageLayerBytes &&
         c.page_stride % 16 == 0 && c.page_tables && c.max_pages > 0 && c.n_pages > 0;
}

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
  for (int off = 16; off > 0; off >>= 1) v += __shfl_xor_sync(0xffffffffu, v, off);
  return v;
}

__device__ __forceinline__ float warp_max(float v) {
#pragma unroll
  for (int off = 16; off > 0; off >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, off));
  return v;
}

// 2^e for e in the normal range, exact.
__device__ __forceinline__ float exp2_int(int e) { return __uint_as_float(uint32_t(e + 127) << 23); }

__device__ __forceinline__ float bf16_round(float x) { return __bfloat162float(__float2bfloat16_rn(x)); }

// Power-of-two block scale: the smallest 2^k with amax <= 448 * 2^k (k clamped to
// [-126, 126]); scale = 1 for a zero or non-finite amax. Mirrors
// crates/glm53f-dsa/src/fp8.rs (pow2_scale_exponent, quantize_block).
__device__ __forceinline__ void pow2_scale(float amax, float& scale, float& inv) {
  if (!(amax > 0.f) || !isfinite(amax)) {
    scale = 1.f;
    inv = 1.f;
    return;
  }
  const uint32_t bits = __float_as_uint(amax) & 0x7fffffffu;
  const int biased = int(bits >> 23);
  int k;
  if (biased == 0) {
    k = -126;
  } else {
    const int e = biased - 127;
    k = (bits & 0x7fffffu) <= 0x600000u ? e - 8 : e - 7;
  }
  k = max(-126, min(126, k));
  scale = exp2_int(k);
  inv = exp2_int(-k);
}

// Round-to-nearest-even, saturating E4M3 (cvt.rn.satfinite.e4m3x2.f32).
__device__ __forceinline__ uint8_t fp8_e4m3(float x) {
  return static_cast<uint8_t>(__nv_cvt_float_to_fp8(x, __NV_SATFINITE, __NV_E4M3));
}

// Two floats to two E4M3 bytes (x.x in the low byte).
__device__ __forceinline__ uint16_t fp8x2_e4m3(float lo, float hi) {
  return static_cast<uint16_t>(__nv_cvt_float2_to_fp8x2(make_float2(lo, hi), __NV_SATFINITE, __NV_E4M3));
}

// (score, pool) -> sortable key; larger ranks first, ties to the lower pool, NaN -> 0.
// Mirrors crates/glm53f-dsa/src/select.rs (score_key).
__device__ __forceinline__ uint64_t score_key(float s, uint32_t pool) {
  if (isnan(s)) return 0;
  const uint32_t bits = s == 0.f ? 0u : __float_as_uint(s);
  const uint32_t ordered = (bits & 0x80000000u) ? ~bits : (bits ^ 0x80000000u);
  return (uint64_t(ordered) << 32) | uint64_t(~pool);
}

// Physical page of a request's logical page, or nullptr when out of range.
__device__ __forceinline__ uint8_t* page_base(const glm53f_dsa_cache_t& c, int req, int logical_page) {
  if (logical_page < 0 || logical_page >= c.max_pages) return nullptr;
  const int32_t phys = c.page_tables[int64_t(req) * c.max_pages + logical_page];
  if (phys < 0 || phys >= c.n_pages) return nullptr;
  return c.base + int64_t(phys) * c.page_stride;
}

// A pooled key's scale (NaN when its page is not mapped, so the pool never ranks).
__device__ __forceinline__ float pool_scale(const glm53f_dsa_cache_t& c, int req, int pool) {
  const uint8_t* page = page_base(c, req, pool / kPagePools);
  if (!page) return __int_as_float(0x7fc00000);
  return __ldg(reinterpret_cast<const float*>(page + kPoolScalesOffset + (pool % kPagePools) * 4));
}

// Byte offset of a token's latent record from the cache base, or -1.
__device__ __forceinline__ int64_t latent_offset(const glm53f_dsa_cache_t& c, int req, int token) {
  if (token < 0) return -1;
  const int lp = token / kPageTokens;
  if (lp >= c.max_pages) return -1;
  const int32_t phys = c.page_tables[int64_t(req) * c.max_pages + lp];
  if (phys < 0 || phys >= c.n_pages) return -1;
  return int64_t(phys) * c.page_stride + int64_t(token % kPageTokens) * kLatentBytes;
}

}  // namespace glm53f
