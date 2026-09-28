// Device helpers for the glm53f-forward kernels.
//
// The load helpers and the butterfly follow glm53f-layers' kernels/common.cuh (same
// repository); every floating-point operation a result depends on is an explicit rounding
// intrinsic, and the build passes --fmad=false.
#pragma once
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <stdint.h>

namespace glm53f_fwd {

// ---- BF16 ----------------------------------------------------------------------------------
__device__ __forceinline__ float bf16_lo(uint32_t w) { return __uint_as_float(w << 16); }
__device__ __forceinline__ float bf16_hi(uint32_t w) { return __uint_as_float(w & 0xFFFF0000u); }
__device__ __forceinline__ float bf16_to_f32(uint16_t b) { return __uint_as_float(uint32_t(b) << 16); }
__device__ __forceinline__ uint16_t f32_to_bf16(float x) {
  return __bfloat16_as_ushort(__float2bfloat16_rn(x));
}
__device__ __forceinline__ float bf16_round(float x) { return bf16_to_f32(f32_to_bf16(x)); }
// Eight BF16 values of a 16-byte vector, in memory order.
__device__ __forceinline__ void unpack_bf16x8(const uint4 v, float (&f)[8]) {
  f[0] = bf16_lo(v.x); f[1] = bf16_hi(v.x); f[2] = bf16_lo(v.y); f[3] = bf16_hi(v.y);
  f[4] = bf16_lo(v.z); f[5] = bf16_hi(v.z); f[6] = bf16_lo(v.w); f[7] = bf16_hi(v.w);
}

// ---- Warp reductions -------------------------------------------------------------------------
// Butterfly sum: every lane ends with the same bits.
__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
  for (int off = 16; off; off >>= 1) v = __fadd_rn(v, __shfl_xor_sync(0xffffffffu, v, off));
  return v;
}

// ---- Memory ----------------------------------------------------------------------------------
// A streamed read: through the non-coherent path, not kept in L1.
__device__ __forceinline__ uint4 ld_stream(const void* p) {
  uint4 v;
  asm("ld.global.nc.L1::no_allocate.v4.u32 {%0,%1,%2,%3}, [%4];"
      : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p));
  return v;
}
__device__ __forceinline__ uint4 ld_cached(const void* p) {
  return __ldg(reinterpret_cast<const uint4*>(p));
}

// ---- Argument checks -------------------------------------------------------------------------
inline bool aligned16(const void* p) { return p && (reinterpret_cast<uintptr_t>(p) & 15u) == 0; }

}  // namespace glm53f_fwd
