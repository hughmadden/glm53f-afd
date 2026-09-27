// Shared device helpers for the glm53f-layers kernels.
//
// Every floating-point operation that must match the CPU reference is written with an
// explicit rounding intrinsic (__fadd_rn, __fmul_rn, __fdiv_rn, __fsqrt_rn, __fmaf_rn), and
// the build passes --fmad=false, so nvcc neither contracts nor reorders them.
#pragma once
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <stdint.h>

namespace glm53f {

constexpr float kRmsEps = 1e-5f;
constexpr float kHcEps = 1e-6f;
constexpr float kSwigluLimit = 10.0f;
constexpr float kE4m3Max = 448.0f;
constexpr float kNormDenomEps = 1e-20f;

// ---- BF16 ----------------------------------------------------------------------------------
__device__ __forceinline__ float bf16_to_f32(uint32_t b) { return __uint_as_float(b << 16); }
__device__ __forceinline__ uint16_t f32_to_bf16(float x) {
  return __bfloat16_as_ushort(__float2bfloat16_rn(x));
}
__device__ __forceinline__ float bf16_round(float x) { return bf16_to_f32(f32_to_bf16(x)); }
// The two BF16 values of a 32-bit word (low half first).
__device__ __forceinline__ float bf16_lo(uint32_t w) { return __uint_as_float(w << 16); }
__device__ __forceinline__ float bf16_hi(uint32_t w) { return __uint_as_float(w & 0xFFFF0000u); }
__device__ __forceinline__ uint32_t pack_bf16x2(float lo, float hi) {
  return uint32_t(f32_to_bf16(lo)) | (uint32_t(f32_to_bf16(hi)) << 16);
}
// Eight BF16 values of a 16-byte vector.
__device__ __forceinline__ void unpack_bf16x8(const uint4 v, float (&f)[8]) {
  f[0] = bf16_lo(v.x); f[1] = bf16_hi(v.x); f[2] = bf16_lo(v.y); f[3] = bf16_hi(v.y);
  f[4] = bf16_lo(v.z); f[5] = bf16_hi(v.z); f[6] = bf16_lo(v.w); f[7] = bf16_hi(v.w);
}

// ---- E4M3 ----------------------------------------------------------------------------------
// Two E4M3 codes (low byte first) to f32, exactly (through f16, which holds every E4M3 value).
__device__ __forceinline__ float2 e4m3x2_to_f32x2(uint32_t two) {
  const __half2_raw h = __nv_cvt_fp8x2_to_halfraw2(static_cast<__nv_fp8x2_storage_t>(two & 0xFFFFu), __NV_E4M3);
  return __half22float2(*reinterpret_cast<const __half2*>(&h));
}
// Sixteen E4M3 codes of a 16-byte vector to f32.
__device__ __forceinline__ void unpack_e4m3x16(const uint4 v, float (&f)[16]) {
  const uint32_t w[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const float2 a = e4m3x2_to_f32x2(w[i]);
    const float2 b = e4m3x2_to_f32x2(w[i] >> 16);
    f[4 * i] = a.x; f[4 * i + 1] = a.y; f[4 * i + 2] = b.x; f[4 * i + 3] = b.y;
  }
}
// f32 to E4M3 with round-to-nearest-even and saturation to +-448 (NaN stays NaN).
__device__ __forceinline__ uint32_t f32_to_e4m3(float x) {
  return __nv_cvt_float_to_fp8(x, __NV_SATFINITE, __NV_E4M3);
}

// ---- exp, sigmoid, silu: the same operations as src/math.rs --------------------------------
__device__ __forceinline__ float exp_f32(float x) {
  if (isnan(x)) return x;
  if (x > __int_as_float(0x42b17218)) return __int_as_float(0x7f800000);
  if (x < __int_as_float(0xc2aeac4f)) return 0.0f;
  const float kLog2e = __int_as_float(0x3fb8aa3b), kMagic = __int_as_float(0x4b400000);
  const float kLn2Hi = __int_as_float(0x3f318000), kLn2Lo = __int_as_float(0xb95e8083);
  const float t = __fmaf_rn(x, kLog2e, kMagic);
  const float n = __fsub_rn(t, kMagic);
  float r = __fmaf_rn(n, -kLn2Hi, x);
  r = __fmaf_rn(n, -kLn2Lo, r);
  const float z = __fmul_rn(r, r);
  float p = __int_as_float(0x39506967);
  p = __fmaf_rn(p, r, __int_as_float(0x3ab743ce));
  p = __fmaf_rn(p, r, __int_as_float(0x3c088908));
  p = __fmaf_rn(p, r, __int_as_float(0x3d2aa9c1));
  p = __fmaf_rn(p, r, __int_as_float(0x3e2aaaaa));
  p = __fmaf_rn(p, r, __int_as_float(0x3f000000));
  const float y = __fadd_rn(__fmaf_rn(p, z, r), 1.0f);
  const int ni = __float2int_rn(n);
  if (ni > 127) return __fmul_rn(__fmul_rn(y, __int_as_float(254 << 23)), 2.0f);
  return __fmul_rn(y, __int_as_float((ni + 127) << 23));
}
__device__ __forceinline__ float sigmoid_f32(float x) {
  return __fdiv_rn(1.0f, __fadd_rn(1.0f, exp_f32(-x)));
}
__device__ __forceinline__ float silu_f32(float x) {
  return __fdiv_rn(x, __fadd_rn(1.0f, exp_f32(-x)));
}

// SwiGLU of BF16 gate and up (as f32): bf16(bf16(silu(min(g, 10))) * clamp(u, -10, 10)).
__device__ __forceinline__ float swiglu_f32(float g, float u) {
  g = g > kSwigluLimit ? kSwigluLimit : g;
  u = u > kSwigluLimit ? kSwigluLimit : (u < -kSwigluLimit ? -kSwigluLimit : u);
  return bf16_round(__fmul_rn(bf16_round(silu_f32(g)), u));
}

// 1 / sqrt(sum_sq / n + eps), IEEE throughout.
__device__ __forceinline__ float rms_scale(float sum_sq, float n, float eps) {
  return __fdiv_rn(1.0f, __fsqrt_rn(__fadd_rn(__fdiv_rn(sum_sq, n), eps)));
}

// ---- Warp reductions -------------------------------------------------------------------------
// Butterfly sum: every lane ends with the same bits (src/math.rs warp_sum).
__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
  for (int off = 16; off; off >>= 1) v = __fadd_rn(v, __shfl_xor_sync(0xffffffffu, v, off));
  return v;
}
__device__ __forceinline__ float group16_max(float v) {
#pragma unroll
  for (int off = 8; off; off >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, off));
  return v;
}
__device__ __forceinline__ float warp_max(float v) {
#pragma unroll
  for (int off = 16; off; off >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, off));
  return v;
}

// E4M3 scale of a 128-group from its absolute maximum (src/fp8.rs quantize_group).
__device__ __forceinline__ float group_scale(float amax) {
  return amax > 0.0f ? __fdiv_rn(amax, kE4m3Max) : 1.0f;
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

__device__ __forceinline__ uint32_t smem_addr(const void* p) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}
__device__ __forceinline__ void cp_async16(void* dst, const void* src, bool valid) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(smem_addr(dst)), "l"(src),
               "r"(valid ? 16 : 0));
}
__device__ __forceinline__ void cp_async4(void* dst, const void* src, bool valid) {
  asm volatile("cp.async.ca.shared.global [%0], [%1], 4, %2;\n" ::"r"(smem_addr(dst)), "l"(src),
               "r"(valid ? 4 : 0));
}
__device__ __forceinline__ void cp_async_commit() { asm volatile("cp.async.commit_group;\n" ::); }
template <int N>
__device__ __forceinline__ void cp_async_wait() { asm volatile("cp.async.wait_group %0;\n" ::"n"(N)); }

__device__ __forceinline__ void ldmatrix_x4(uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3,
                                            const void* p) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(smem_addr(p)));
}

// D += A (16x32 E4M3, row) * B (32x8 E4M3, col), f32 accumulation.
__device__ __forceinline__ void mma_e4m3_16832(float (&d)[4], const uint32_t (&a)[4], const uint32_t (&b)[2]) {
  asm volatile(
      "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
      "{%0,%1,%2,%3};\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

// ---- Argument checks -------------------------------------------------------------------------
inline bool aligned16(const void* p) { return p && (reinterpret_cast<uintptr_t>(p) & 15u) == 0; }
inline bool aligned16_or_null(const void* p) { return !p || (reinterpret_cast<uintptr_t>(p) & 15u) == 0; }

}  // namespace glm53f
