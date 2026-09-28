// Glue between the kernel crates: the embedding gather (from page-locked host memory into
// the 4 mHC streams), dtype conversions, the greedy argmax, row gather/scatter, the
// routed-expert combine of the reference's eager expert loop, and the mean of the 4 streams
// (the DFlash2 drafter's taps).
#include "glm53f_forward.h"
#include "common.cuh"

namespace glm53f_fwd {
namespace {

__global__ void __launch_bounds__(256) embed_gather_kernel(const uint16_t* table, int64_t vocab, const int32_t* ids,
                                                           int hidden, uint16_t* streams, uint16_t* rows_out) {
  const int r = blockIdx.x;
  const int id = ids[r];
  const bool ok = id >= 0 && id < vocab;
  const uint4* src = reinterpret_cast<const uint4*>(table + int64_t(ok ? id : 0) * hidden);
  const int chunks = hidden / 8;
  uint4* dst = reinterpret_cast<uint4*>(streams + int64_t(r) * 4 * hidden);
  uint4* row = rows_out ? reinterpret_cast<uint4*>(rows_out + int64_t(r) * hidden) : nullptr;
  for (int c = threadIdx.x; c < chunks; c += blockDim.x) {
    const uint4 v = ok ? src[c] : make_uint4(0u, 0u, 0u, 0u);
#pragma unroll
    for (int s = 0; s < 4; ++s) dst[s * chunks + c] = v;
    if (row) row[c] = v;
  }
}

__global__ void bf16_to_f32_kernel(const uint16_t* in, int64_t ldi, float* out, int64_t ldo, int rows, int cols,
                                   float scale) {
  const int64_t total = int64_t(rows) * cols;
  for (int64_t i = blockIdx.x * int64_t(blockDim.x) + threadIdx.x; i < total; i += int64_t(gridDim.x) * blockDim.x) {
    const int64_t r = i / cols, c = i % cols;
    out[r * ldo + c] = __fmul_rn(bf16_to_f32(in[r * ldi + c]), scale);
  }
}

__global__ void f32_to_bf16_kernel(const float* in, int64_t ldi, uint16_t* out, int64_t ldo, int rows, int cols) {
  const int64_t total = int64_t(rows) * cols;
  for (int64_t i = blockIdx.x * int64_t(blockDim.x) + threadIdx.x; i < total; i += int64_t(gridDim.x) * blockDim.x) {
    const int64_t r = i / cols, c = i % cols;
    out[r * ldo + c] = f32_to_bf16(in[r * ldi + c]);
  }
}

// (value, index) with the larger value first, then the lower index; index < 0 is empty.
__device__ __forceinline__ bool better(float v, int i, float bv, int bi) {
  if (i < 0) return false;
  if (bi < 0) return true;
  return v > bv || (v == bv && i < bi);
}

__global__ void __launch_bounds__(1024) argmax_kernel(const float* logits, int64_t ld, int n_valid, int32_t* ids,
                                                      float* vals) {
  __shared__ float sv[32];
  __shared__ int si[32];
  const int r = blockIdx.x, tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const float* row = logits + int64_t(r) * ld;
  float bv = 0.0f;
  int bi = -1;
  for (int i = tid; i < n_valid; i += blockDim.x) {
    const float v = row[i];
    if (!isnan(v) && (bi < 0 || v > bv)) {  // ascending i per thread: a tie keeps the lower index
      bv = v;
      bi = i;
    }
  }
#pragma unroll
  for (int off = 16; off; off >>= 1) {
    const float ov = __shfl_xor_sync(0xffffffffu, bv, off);
    const int oi = __shfl_xor_sync(0xffffffffu, bi, off);
    if (better(ov, oi, bv, bi)) {
      bv = ov;
      bi = oi;
    }
  }
  if (lane == 0) {
    sv[warp] = bv;
    si[warp] = bi;
  }
  __syncthreads();
  if (warp == 0) {
    const int nw = blockDim.x >> 5;
    bv = lane < nw ? sv[lane] : 0.0f;
    bi = lane < nw ? si[lane] : -1;
#pragma unroll
    for (int off = 16; off; off >>= 1) {
      const float ov = __shfl_xor_sync(0xffffffffu, bv, off);
      const int oi = __shfl_xor_sync(0xffffffffu, bi, off);
      if (better(ov, oi, bv, bi)) {
        bv = ov;
        bi = oi;
      }
    }
    if (lane == 0) {
      ids[r] = bi < 0 ? 0 : bi;
      if (vals) vals[r] = bi < 0 ? __int_as_float(0x7fc00000) : bv;
    }
  }
}

__global__ void __launch_bounds__(128) gather_rows_kernel(const uint8_t* src, int64_t src_stride, const int32_t* idx,
                                                          uint8_t* dst, int64_t dst_stride, int64_t chunks,
                                                          int scatter) {
  const int i = blockIdx.x;
  const int64_t s_row = scatter ? i : idx[i];
  const int64_t d_row = scatter ? idx[i] : i;
  const uint4* s = reinterpret_cast<const uint4*>(src + s_row * src_stride);
  uint4* d = reinterpret_cast<uint4*>(dst + d_row * dst_stride);
  for (int64_t c = threadIdx.x; c < chunks; c += blockDim.x) d[c] = s[c];
}

constexpr int kMaxTopK = 32;

__global__ void __launch_bounds__(256) moe_combine_kernel(const uint16_t* y, const int32_t* ids, const float* weights,
                                                          int top_k, int hidden, uint16_t* out) {
  __shared__ int order[kMaxTopK];
  __shared__ int count;
  const int r = blockIdx.x;
  if (threadIdx.x == 0) {
    // The row's valid slots in ascending expert id (equal ids in slot order): insertion sort.
    int c = 0;
    for (int j = 0; j < top_k; ++j) {
      const int id = ids[int64_t(r) * top_k + j];
      if (id < 0) continue;
      int p = c++;
      while (p > 0 && ids[int64_t(r) * top_k + order[p - 1]] > id) {
        order[p] = order[p - 1];
        --p;
      }
      order[p] = j;
    }
    count = c;
  }
  __syncthreads();
  const int chunks = hidden / 8;
  for (int ch = threadIdx.x; ch < chunks; ch += blockDim.x) {
    float acc[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
    for (int t = 0; t < count; ++t) {
      const int j = order[t];
      const float wj = weights[int64_t(r) * top_k + j];
      float v[8];
      unpack_bf16x8(*reinterpret_cast<const uint4*>(y + (int64_t(r) * top_k + j) * hidden + ch * 8), v);
#pragma unroll
      for (int e = 0; e < 8; ++e) acc[e] = bf16_round(__fadd_rn(acc[e], bf16_round(__fmul_rn(v[e], wj))));
    }
    uint4 o;
    o.x = uint32_t(f32_to_bf16(acc[0])) | (uint32_t(f32_to_bf16(acc[1])) << 16);
    o.y = uint32_t(f32_to_bf16(acc[2])) | (uint32_t(f32_to_bf16(acc[3])) << 16);
    o.z = uint32_t(f32_to_bf16(acc[4])) | (uint32_t(f32_to_bf16(acc[5])) << 16);
    o.w = uint32_t(f32_to_bf16(acc[6])) | (uint32_t(f32_to_bf16(acc[7])) << 16);
    *reinterpret_cast<uint4*>(out + int64_t(r) * hidden + ch * 8) = o;
  }
}

__global__ void __launch_bounds__(256) stream_mean_kernel(const uint16_t* streams, int hidden, uint16_t* out,
                                                          int64_t ldo) {
  const int r = blockIdx.x;
  const int chunks = hidden / 8;
  const uint16_t* row = streams + int64_t(r) * 4 * hidden;
  for (int c = threadIdx.x; c < chunks; c += blockDim.x) {
    float s[4][8];
#pragma unroll
    for (int j = 0; j < 4; ++j) unpack_bf16x8(*reinterpret_cast<const uint4*>(row + int64_t(j) * hidden + c * 8), s[j]);
    uint32_t w[4];
#pragma unroll
    for (int e = 0; e < 8; e += 2) {
      const float a = __fmul_rn(__fadd_rn(__fadd_rn(__fadd_rn(s[0][e], s[1][e]), s[2][e]), s[3][e]), 0.25f);
      const float b =
          __fmul_rn(__fadd_rn(__fadd_rn(__fadd_rn(s[0][e + 1], s[1][e + 1]), s[2][e + 1]), s[3][e + 1]), 0.25f);
      w[e / 2] = uint32_t(f32_to_bf16(a)) | (uint32_t(f32_to_bf16(b)) << 16);
    }
    *reinterpret_cast<uint4*>(out + int64_t(r) * ldo + c * 8) = make_uint4(w[0], w[1], w[2], w[3]);
  }
}

int grid_for(int64_t total, int threads) {
  const int64_t want = (total + threads - 1) / threads;
  return int(want < 65535 ? (want < 1 ? 1 : want) : 65535);
}

bool aligned16_or_null(const void* p) { return !p || (reinterpret_cast<uintptr_t>(p) & 15u) == 0; }

}  // namespace
}  // namespace glm53f_fwd

using namespace glm53f_fwd;

extern "C" int32_t glm53f_fwd_embed_gather(const uint16_t* table, int64_t vocab, const int32_t* ids, int32_t rows,
                                           int32_t hidden, uint16_t* streams, uint16_t* rows_out,
                                           cudaStream_t stream) {
  if (rows < 0 || hidden < 8 || hidden % 8 || vocab < 1 || !aligned16(table) || !ids || !aligned16(streams) ||
      !aligned16_or_null(rows_out))
    return cudaErrorInvalidValue;
  if (rows == 0) return cudaSuccess;
  embed_gather_kernel<<<rows, 256, 0, stream>>>(table, vocab, ids, hidden, streams, rows_out);
  return cudaGetLastError();
}

extern "C" int32_t glm53f_fwd_bf16_to_f32(const uint16_t* in, int64_t ldi, float* out, int64_t ldo, int32_t rows,
                                          int32_t cols, float scale, cudaStream_t stream) {
  if (rows < 0 || cols < 0 || ldi < cols || ldo < cols || !in || !out) return cudaErrorInvalidValue;
  if (rows == 0 || cols == 0) return cudaSuccess;
  bf16_to_f32_kernel<<<grid_for(int64_t(rows) * cols, 256), 256, 0, stream>>>(in, ldi, out, ldo, rows, cols, scale);
  return cudaGetLastError();
}

extern "C" int32_t glm53f_fwd_f32_to_bf16(const float* in, int64_t ldi, uint16_t* out, int64_t ldo, int32_t rows,
                                          int32_t cols, cudaStream_t stream) {
  if (rows < 0 || cols < 0 || ldi < cols || ldo < cols || !in || !out) return cudaErrorInvalidValue;
  if (rows == 0 || cols == 0) return cudaSuccess;
  f32_to_bf16_kernel<<<grid_for(int64_t(rows) * cols, 256), 256, 0, stream>>>(in, ldi, out, ldo, rows, cols);
  return cudaGetLastError();
}

extern "C" int32_t glm53f_fwd_argmax(const float* logits, int64_t ld, int32_t rows, int32_t n_valid, int32_t* ids,
                                     float* vals, cudaStream_t stream) {
  if (rows < 0 || n_valid < 1 || ld < n_valid || !logits || !ids) return cudaErrorInvalidValue;
  if (rows == 0) return cudaSuccess;
  argmax_kernel<<<rows, 1024, 0, stream>>>(logits, ld, n_valid, ids, vals);
  return cudaGetLastError();
}

namespace {
int32_t gather_scatter(const uint8_t* src, int64_t src_stride, const int32_t* idx, uint8_t* dst, int64_t dst_stride,
                       int32_t rows, int64_t bytes, int scatter, cudaStream_t stream) {
  if (rows < 0 || bytes < 0 || bytes % 16 || src_stride % 16 || dst_stride % 16 || src_stride < bytes ||
      dst_stride < bytes)
    return cudaErrorInvalidValue;
  if (rows == 0 || bytes == 0) return cudaSuccess;
  if (!aligned16(src) || !aligned16(dst) || !idx) return cudaErrorInvalidValue;
  gather_rows_kernel<<<rows, 128, 0, stream>>>(src, src_stride, idx, dst, dst_stride, bytes / 16, scatter);
  return cudaGetLastError();
}
}  // namespace

extern "C" int32_t glm53f_fwd_gather_rows(const uint8_t* src, int64_t src_stride, const int32_t* idx, uint8_t* dst,
                                          int64_t dst_stride, int32_t rows, int64_t bytes, cudaStream_t stream) {
  return gather_scatter(src, src_stride, idx, dst, dst_stride, rows, bytes, 0, stream);
}

extern "C" int32_t glm53f_fwd_scatter_rows(const uint8_t* src, int64_t src_stride, const int32_t* idx, uint8_t* dst,
                                           int64_t dst_stride, int32_t rows, int64_t bytes, cudaStream_t stream) {
  return gather_scatter(src, src_stride, idx, dst, dst_stride, rows, bytes, 1, stream);
}

extern "C" int32_t glm53f_fwd_moe_combine(const uint16_t* y, const int32_t* ids, const float* weights, int32_t rows,
                                          int32_t top_k, int32_t hidden, uint16_t* out, cudaStream_t stream) {
  if (rows < 0 || top_k < 1 || top_k > kMaxTopK || hidden < 8 || hidden % 8 || !aligned16(y) || !ids || !weights ||
      !aligned16(out))
    return cudaErrorInvalidValue;
  if (rows == 0) return cudaSuccess;
  moe_combine_kernel<<<rows, 256, 0, stream>>>(y, ids, weights, top_k, hidden, out);
  return cudaGetLastError();
}

extern "C" int32_t glm53f_fwd_stream_mean(const uint16_t* streams, int32_t rows, int32_t hidden, uint16_t* out,
                                          int64_t ldo, cudaStream_t stream) {
  if (rows < 0 || hidden < 8 || hidden % 8 || ldo < hidden || ldo % 8 || !aligned16(streams) || !aligned16(out))
    return cudaErrorInvalidValue;
  if (rows == 0) return cudaSuccess;
  stream_mean_kernel<<<rows, 256, 0, stream>>>(streams, hidden, out, ldo);
  return cudaGetLastError();
}
