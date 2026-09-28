// Coordinator-side wire kernels for the expert exchange: the FP8 E4M3 / UE8M0-K32 quantizer of
// the hidden rows sent to the ranks, the device fill of a mapped request frame, and the sum of
// the ranks' BF16 return planes.
//
// Copied from mimo26f-afd v1.2.0 (PROVENANCE.md): `quant_scales_kernel`, `rank_sum_bf16_kernel`,
// `frame_fill_kernel` and `grid_for` from `crates/mimo26-coordinator/kernels/glue.cu`;
// `e4m3_encode` from `crates/mimo26-attn/kernels/include/mimo26_attn_device.cuh`;
// `hidden_quantize_encode_kernel` from `crates/mimo26-attn/kernels/kv_cache_fp8.cu`. Changed: the
// entry points' names; the rank sum multiplies by a routed scale after the sum (1.0 leaves the
// source's bits); comments that pointed at private documents.
//
// Each kernel mirrors its host twin in `src/wire.rs` (and `glm53f-wire`'s `CoordinatorSum`) bit
// for bit. Every entry point is asynchronous on `stream` and returns the launch status.

#include <cuda_runtime.h>
#include <stdint.h>
#include <math.h>

namespace {

constexpr int kThreads = 256;

/* Nearest representable E4M3, ties to the larger magnitude, clamped at 448 (`src/fp8.rs`
 * `encode_e4m3` is the host twin). `clip` (optional, a device counter) counts inputs whose
 * magnitude exceeds 448. The magnitude grid is (1 + m/8) 2^(e-7) for normals and m 2^-9 for
 * subnormals. */
__device__ __forceinline__ uint8_t e4m3_encode(float x, unsigned long long* clip) {
  const uint8_t sign = (x < 0.0f) ? 0x80u : 0x00u;
  float mag = fabsf(x);
  if (mag > 448.0f) {
    if (clip) atomicAdd(clip, 1ULL);
    mag = 448.0f;
  }
  if (mag == 0.0f) return sign;
  int code;
  const int E = ilogbf(mag);
  if (E <= -7) {
    /* subnormal band: magnitudes q·2^-9, q in 0..8 (q == 8 == smallest normal) */
    code = (int)floorf(mag * 512.0f + 0.5f); /* half-up: ties to larger */
  } else {
    const float step = ldexpf(1.0f, E - 3);
    int q = (int)floorf(mag / step + 0.5f); /* half-up: ties to larger */
    int ee = E + 7;
    if (q >= 16) { q = 8; ee += 1; }
    code = (ee << 3) | (q - 8);
    if (code > 0x7E) code = 0x7E;
  }
  return (uint8_t)(sign | (uint8_t)code);
}


// src/wire.rs quantize_hidden_scales: per K32 block, amax in FP64,
// s = amax == 0 ? 0 : clamp(ceil(log2(amax / 448)) + 127, 0, 254),
// scale_inv = 2^(127 - s) (exact in f32 for s in 0..=254).
__global__ void quant_scales_kernel(const float* __restrict__ x, long n_blocks,
                                    unsigned char* __restrict__ scales, float* __restrict__ scale_inv) {
    for (long b = blockIdx.x * (long)blockDim.x + threadIdx.x; b < n_blocks; b += (long)gridDim.x * blockDim.x) {
        const float* blk = x + b * 32;
        double amax = 0.0;
        for (int k = 0; k < 32; ++k) amax = fmax(amax, fabs((double)blk[k]));
        int s = 0;
        if (amax != 0.0) {
            double e = ceil(log2(amax / 448.0)) + 127.0;
            e = fmin(fmax(e, 0.0), 254.0);
            s = (int)e;
        }
        scales[b] = (unsigned char)s;
        scale_inv[b] = ldexpf(1.0f, 127 - s);
    }
}


/* Hidden quantize encode (the MoE wire-out Fp8E4m3Ue8m0K32): one thread per
 * element, `payload[i] = e4m3_encode(x[i] * scale_inv[i>>5])`. The scale_inv is a
 * power of two (2^(127-s)), so the f32 multiply is exact and bit-identical to the
 * CPU's f64 `v * scale_inv` (same mantissa, shifted exponent). */
__global__ void hidden_quantize_encode_kernel(
    const float* x, const float* scale_inv, uint8_t* payload, int64_t n_elem) {
  const int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n_elem) return;
  payload[i] = e4m3_encode(x[i] * scale_inv[i >> 5], nullptr);
}

// l4.rs CoordinatorSum (R8): s = 0; then s += f32(bf16) for ranks 0,1,2,3, FP32,
// no fused ops. The same order and start value as the host sum, so bit-identical.
// Then the routed scale: out = s * scale (the host path multiplies the same way; 1.0 is exact).
__global__ void rank_sum_bf16_kernel(const uint16_t* __restrict__ p0, const uint16_t* __restrict__ p1,
                                     const uint16_t* __restrict__ p2, const uint16_t* __restrict__ p3,
                                     float* __restrict__ out, long n, float scale) {
    for (long i = blockIdx.x * (long)blockDim.x + threadIdx.x; i < n; i += (long)gridDim.x * blockDim.x) {
        float s = 0.0f;
        s = __fadd_rn(s, __uint_as_float((uint32_t)p0[i] << 16));
        s = __fadd_rn(s, __uint_as_float((uint32_t)p1[i] << 16));
        s = __fadd_rn(s, __uint_as_float((uint32_t)p2[i] << 16));
        s = __fadd_rn(s, __uint_as_float((uint32_t)p3[i] << 16));
        out[i] = __fmul_rn(s, scale);
    }
}


// Perf reset P9: write one MoE request's route entries and hidden rows straight
// into the page-locked, device-mapped RDMA frame body (DS41RTE3 v3 layout: route
// entry = row u32, expert u32, weight f32 bits; hidden row = hid E4M3 payload
// then hid/32 UE8M0 scales at `pitch`). One kernel instead of two D2H route
// downloads and two strided copies; the bytes equal the host encoder's.
__global__ void frame_fill_kernel(const int* __restrict__ idx, const float* __restrict__ wts,
                                  const uint8_t* __restrict__ payload, const uint8_t* __restrict__ scales, int t,
                                  int topk, int hid, uint8_t* __restrict__ routes, uint8_t* __restrict__ hidden,
                                  int pitch) {
    // 8-byte stores: the hidden rows start at 136 * t bytes into the body (40 B
    // descriptor + 8 x 12 B routes per row), 8- but not always 16-byte aligned.
    const long nr = (long)t * topk;
    const int row8 = hid / 8, sc8 = hid / 32 / 8;
    const long nh = (long)t * (row8 + sc8);
    for (long i = blockIdx.x * (long)blockDim.x + threadIdx.x; i < nr + nh; i += (long)gridDim.x * blockDim.x) {
        if (i < nr) {
            uint32_t* e = reinterpret_cast<uint32_t*>(routes + i * 12);
            e[0] = uint32_t(i / topk);
            e[1] = uint32_t(idx[i]);
            e[2] = __float_as_uint(wts[i]);
        } else {
            const long j = i - nr;
            const long r = j / (row8 + sc8), c = j % (row8 + sc8);
            const uint2* src = c < row8 ? reinterpret_cast<const uint2*>(payload + r * hid) + c
                                        : reinterpret_cast<const uint2*>(scales + r * (hid / 32)) + (c - row8);
            *reinterpret_cast<uint2*>(hidden + r * pitch + c * 8) = *src;
        }
    }
}

inline int grid_for(long n) {
    long g = (n + kThreads - 1) / kThreads;
    if (g > 65535L * 8) g = 65535L * 8;
    if (g < 1) g = 1;
    return (int)g;
}

}  // namespace

extern "C" {

// Per K32 block of x [n_blocks * 32]: the UE8M0 scale byte and its exact inverse 2^(127 - s).
cudaError_t glm53f_coord_quant_scales(const float* x, long n_blocks, unsigned char* scales, float* scale_inv,
                                      cudaStream_t stream) {
    if (n_blocks <= 0) return cudaSuccess;
    quant_scales_kernel<<<grid_for(n_blocks), kThreads, 0, stream>>>(x, n_blocks, scales, scale_inv);
    return cudaGetLastError();
}

// payload[i] = e4m3(x[i] * scale_inv[i / 32]) for i in [0, n_elem).
cudaError_t glm53f_coord_quantize_hidden(const float* x, const float* scale_inv, uint8_t* payload, int64_t n_elem,
                                         cudaStream_t stream) {
    if (n_elem <= 0) return cudaSuccess;
    const int64_t blocks = (n_elem + 255) / 256;
    hidden_quantize_encode_kernel<<<(int)blocks, 256, 0, stream>>>(x, scale_inv, payload, n_elem);
    return cudaGetLastError();
}

// out[i] = (((0 + p0[i]) + p1[i]) + p2[i]) + p3[i], FP32, times `scale`, for i in [0, n).
cudaError_t glm53f_coord_rank_sum_bf16(const uint16_t* p0, const uint16_t* p1, const uint16_t* p2, const uint16_t* p3,
                                       float* out, long n, float scale, cudaStream_t stream) {
    if (n <= 0) return cudaSuccess;
    rank_sum_bf16_kernel<<<grid_for(n), kThreads, 0, stream>>>(p0, p1, p2, p3, out, n, scale);
    return cudaGetLastError();
}

// One request frame's route entries and hidden rows written into a (device-mapped) frame body.
cudaError_t glm53f_coord_frame_fill(const int* idx, const float* wts, const uint8_t* payload, const uint8_t* scales,
                                    int t, int topk, int hid, uint8_t* routes, uint8_t* hidden, int pitch,
                                    cudaStream_t stream) {
    if (t <= 0) return cudaSuccess;
    const long n = (long)t * topk + (long)t * (hid / 8 + hid / 32 / 8);
    frame_fill_kernel<<<grid_for(n), kThreads, 0, stream>>>(idx, wts, payload, scales, t, topk, hid, routes, hidden,
                                                           pitch);
    return cudaGetLastError();
}

}  // extern "C"
