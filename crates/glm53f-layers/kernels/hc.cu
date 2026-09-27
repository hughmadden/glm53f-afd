// mHC boundary kernels: projection partials (with the previous sublayer's expand fused in
// front), finish (weights, Sinkhorn, collapse, the sublayer's RMSNorm and optional W8A8
// quantization of its input), the final mean + norm, and the embedding broadcast.
//
// The order of operations is the one src/mhc.rs models:
// - projection slice s covers hidden positions [128 s, 128 s + 128) of all 4 streams; warp w
//   of 8 computes projections 3w..3w+2 (warp 0 also the sum of squares); lane l takes
//   positions 128 s + 4 l + i, i in 0..4, stream-major;
// - the finish sums the slices in order; its Sinkhorn is the 16-lane butterfly of
//   ds41rt's v41_hc.cu (see PROVENANCE.md);
// - the row reductions of the collapse norm use 256 threads over chunks of 8 positions.
#include "glm53f_layers.h"
#include "common.cuh"

namespace glm53f {
namespace {

constexpr int kSlice = 128;
constexpr int kPartial = 25;
constexpr int kRowsPerGroup = 8;  // rows per CTA group for more than 8 rows (1 below)

// The expanded (new) values of stream positions d..d+3 for all 4 destination streams.
__device__ __forceinline__ void expand4(const uint16_t* __restrict__ res_row,  // [4][hidden]
                                        const uint16_t* __restrict__ h1, const uint16_t* __restrict__ h2,
                                        const float* __restrict__ post, const float* __restrict__ comb,
                                        int hidden, int d, float (&out)[4][4]) {
  float h[4], res[4][4];
  {
    const uint2 a = *reinterpret_cast<const uint2*>(h1 + d);
    h[0] = bf16_lo(a.x); h[1] = bf16_hi(a.x); h[2] = bf16_lo(a.y); h[3] = bf16_hi(a.y);
    if (h2) {
      const uint2 b = *reinterpret_cast<const uint2*>(h2 + d);
      h[0] = bf16_round(__fadd_rn(h[0], bf16_lo(b.x)));
      h[1] = bf16_round(__fadd_rn(h[1], bf16_hi(b.x)));
      h[2] = bf16_round(__fadd_rn(h[2], bf16_lo(b.y)));
      h[3] = bf16_round(__fadd_rn(h[3], bf16_hi(b.y)));
    }
  }
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    const uint2 a = *reinterpret_cast<const uint2*>(res_row + j * hidden + d);
    res[j][0] = bf16_lo(a.x); res[j][1] = bf16_hi(a.x); res[j][2] = bf16_lo(a.y); res[j][3] = bf16_hi(a.y);
  }
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const float pb = bf16_round(post[i]);
    float cb[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) cb[j] = bf16_round(comb[4 * j + i]);
#pragma unroll
    for (int k = 0; k < 4; ++k) {
      const float e = bf16_round(__fmul_rn(pb, h[k]));
      float m = 0.0f;
#pragma unroll
      for (int j = 0; j < 4; ++j) m = __fmaf_rn(cb[j], res[j][k], m);
      out[i][k] = bf16_round(__fadd_rn(e, bf16_round(m)));
    }
  }
}

template <bool Expand, bool Project, int kRowsPerGroup>
__global__ void __launch_bounds__(256, 2) hc_project_kernel(
    const uint16_t* __restrict__ streams_in, const uint16_t* __restrict__ h1,
    const uint16_t* __restrict__ h2, const float* __restrict__ post, const float* __restrict__ comb,
    uint16_t* __restrict__ streams_out, const uint16_t* __restrict__ fn, float* __restrict__ partials,
    int rows, int hidden, int groups_per_cta) {
  __shared__ float4 xs[kRowsPerGroup][4][32];
  const int s = blockIdx.x, slices = gridDim.x;
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int d = s * kSlice + 4 * lane;
  const int flat = 4 * hidden;

  // This warp's fn values: projections 3*warp + q, streams st, positions d..d+3.
  float w[3][4][4];
  if (Project) {
#pragma unroll
    for (int q = 0; q < 3; ++q)
#pragma unroll
      for (int st = 0; st < 4; ++st) {
        const uint2 a = __ldg(reinterpret_cast<const uint2*>(fn + size_t(3 * warp + q) * flat + st * hidden + d));
        w[q][st][0] = bf16_lo(a.x); w[q][st][1] = bf16_hi(a.x); w[q][st][2] = bf16_lo(a.y); w[q][st][3] = bf16_hi(a.y);
      }
  }

  for (int g = 0; g < groups_per_cta; ++g) {
    const int r0 = (blockIdx.y * groups_per_cta + g) * kRowsPerGroup;
    if (r0 >= rows) break;
    const int nr = min(kRowsPerGroup, rows - r0);
    // Stage the group's stream values: warp rr handles row r0 + rr, lane l positions d..d+3.
    {
      const int rr = warp;
      if (rr < nr) {
        const size_t row = size_t(r0 + rr);
        if (Expand) {
          float v[4][4];
          expand4(streams_in + row * flat, h1 + row * hidden, h2 ? h2 + row * hidden : nullptr,
                  post + row * 4, comb + row * 16, hidden, d, v);
#pragma unroll
          for (int i = 0; i < 4; ++i) {
            uint2 o;
            o.x = pack_bf16x2(v[i][0], v[i][1]);
            o.y = pack_bf16x2(v[i][2], v[i][3]);
            *reinterpret_cast<uint2*>(streams_out + row * flat + i * hidden + d) = o;
            xs[rr][i][lane] = make_float4(v[i][0], v[i][1], v[i][2], v[i][3]);
          }
        } else {
#pragma unroll
          for (int st = 0; st < 4; ++st) {
            const uint2 a = *reinterpret_cast<const uint2*>(streams_in + row * flat + st * hidden + d);
            xs[rr][st][lane] = make_float4(bf16_lo(a.x), bf16_hi(a.x), bf16_lo(a.y), bf16_hi(a.y));
          }
        }
      }
    }
    if (!Project) continue;
    __syncthreads();
    // All rows of the group at once, so their FMA and butterfly chains interleave. Rows past
    // `nr` accumulate zeros and are not stored; each row's arithmetic is unchanged.
    float acc[kRowsPerGroup][3], sq[kRowsPerGroup];
#pragma unroll
    for (int rr = 0; rr < kRowsPerGroup; ++rr) {
      acc[rr][0] = acc[rr][1] = acc[rr][2] = sq[rr] = 0.0f;
      if (rr < nr) {
        float x[4][4];
#pragma unroll
        for (int st = 0; st < 4; ++st) {
          const float4 v = xs[rr][st][lane];
          x[st][0] = v.x; x[st][1] = v.y; x[st][2] = v.z; x[st][3] = v.w;
        }
#pragma unroll
        for (int st = 0; st < 4; ++st)
#pragma unroll
          for (int i = 0; i < 4; ++i) {
#pragma unroll
            for (int q = 0; q < 3; ++q) acc[rr][q] = __fmaf_rn(x[st][i], w[q][st][i], acc[rr][q]);
            sq[rr] = __fmaf_rn(x[st][i], x[st][i], sq[rr]);
          }
      }
    }
#pragma unroll
    for (int off = 16; off; off >>= 1)
#pragma unroll
      for (int rr = 0; rr < kRowsPerGroup; ++rr) {
        if (rr >= nr) continue;  // uniform across the CTA
#pragma unroll
        for (int q = 0; q < 3; ++q) acc[rr][q] = __fadd_rn(acc[rr][q], __shfl_xor_sync(0xffffffffu, acc[rr][q], off));
        if (warp == 0) sq[rr] = __fadd_rn(sq[rr], __shfl_xor_sync(0xffffffffu, sq[rr], off));
      }
    if (lane < kRowsPerGroup && lane < nr) {
      float* out = partials + (size_t(r0 + lane) * slices + s) * kPartial;
      // Lane rr stores row rr (every lane holds every row's totals).
#pragma unroll
      for (int rr = 0; rr < kRowsPerGroup; ++rr) {
        if (rr == lane) {
#pragma unroll
          for (int q = 0; q < 3; ++q) out[3 * warp + q] = acc[rr][q];
          if (warp == 0) out[24] = sq[rr];
        }
      }
    }
    __syncthreads();
  }
}

// The 16-lane Sinkhorn of one row's comb logits (lanes 0..15 hold row-major [i][j]).
__device__ __forceinline__ float sinkhorn16(float v) {
  float m = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, 1));
  m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, 2));
  v = exp_f32(__fsub_rn(v, m));
  float s = __fadd_rn(v, __shfl_xor_sync(0xffffffffu, v, 1));
  s = __fadd_rn(s, __shfl_xor_sync(0xffffffffu, s, 2));
  v = __fadd_rn(__fdiv_rn(v, s), kHcEps);
  s = __fadd_rn(v, __shfl_xor_sync(0xffffffffu, v, 4));
  s = __fadd_rn(s, __shfl_xor_sync(0xffffffffu, s, 8));
  v = __fdiv_rn(v, __fadd_rn(s, kHcEps));
  for (int it = 1; it < 20; ++it) {
    s = __fadd_rn(v, __shfl_xor_sync(0xffffffffu, v, 1));
    s = __fadd_rn(s, __shfl_xor_sync(0xffffffffu, s, 2));
    v = __fdiv_rn(v, __fadd_rn(s, kHcEps));
    s = __fadd_rn(v, __shfl_xor_sync(0xffffffffu, v, 4));
    s = __fadd_rn(s, __shfl_xor_sync(0xffffffffu, s, 8));
    v = __fdiv_rn(v, __fadd_rn(s, kHcEps));
  }
  return v;
}

// Block-wide sum of per-thread values in the norm's order: butterfly per warp, then the 8
// warp totals in order from 0. Every thread gets the result.
__device__ __forceinline__ float block_sum_256(float v, float* scratch) {
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  v = warp_sum(v);
  if (lane == 0) scratch[warp] = v;
  __syncthreads();
  if (threadIdx.x == 0) {
    float t = 0.0f;
#pragma unroll
    for (int i = 0; i < 8; ++i) t = __fadd_rn(t, scratch[i]);
    scratch[8] = t;
  }
  __syncthreads();
  return scratch[8];
}

// Normalize a row with weight `nw` (global or shared), write BF16 `normed`, and optionally
// its per-128 E4M3 quantization. `chunk(c, v)` yields the row's values at 8c..8c+7.
// Thread t owns chunks t, t + 256, ...
template <class Chunk>
__device__ __forceinline__ void norm_row(Chunk chunk, float r, const uint16_t* nw, uint16_t* __restrict__ normed,
                                         uint8_t* __restrict__ q, float* __restrict__ qs, int hidden) {
  const int chunks = hidden >> 3;
  for (int c = threadIdx.x; c < chunks; c += 256) {
    float v[8], wf[8];
    chunk(c, v);
    unpack_bf16x8(*reinterpret_cast<const uint4*>(nw + 8 * c), wf);
    float o[8];
    float amax = 0.0f;
#pragma unroll
    for (int k = 0; k < 8; ++k) {
      o[k] = bf16_round(__fmul_rn(wf[k], bf16_round(__fmul_rn(v[k], r))));
      amax = fmaxf(amax, fabsf(o[k]));
    }
    if (normed) {
      uint4 ov;
      ov.x = pack_bf16x2(o[0], o[1]); ov.y = pack_bf16x2(o[2], o[3]);
      ov.z = pack_bf16x2(o[4], o[5]); ov.w = pack_bf16x2(o[6], o[7]);
      *reinterpret_cast<uint4*>(normed + 8 * c) = ov;
    }
    if (q) {
      // A 128-group is 16 consecutive chunks, owned by 16 consecutive lanes.
      const float scale = group_scale(group16_max(amax));
      uint32_t lo = 0, hi = 0;
#pragma unroll
      for (int k = 0; k < 4; ++k) lo |= f32_to_e4m3(__fdiv_rn(o[k], scale)) << (8 * k);
#pragma unroll
      for (int k = 0; k < 4; ++k) hi |= f32_to_e4m3(__fdiv_rn(o[4 + k], scale)) << (8 * k);
      *reinterpret_cast<uint2*>(q + 8 * c) = make_uint2(lo, hi);
      if ((c & 15) == 0) qs[c >> 4] = scale;
    }
  }
}

// Barrier over warps 0..7 only (named barrier 1), so a ninth warp can work independently.
__device__ __forceinline__ void sync256() { asm volatile("bar.sync 1, 256;\n" ::: "memory"); }

// Block-wide sum over warps 0..7 in the norm's order: butterfly per warp, then the 8 warp
// totals in order from 0. Every participating thread gets the result.
__device__ __forceinline__ float block_sum_256_named(float v, float* scratch) {
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  v = warp_sum(v);
  if (lane == 0) scratch[warp] = v;
  sync256();
  if (threadIdx.x == 0) {
    float t = 0.0f;
#pragma unroll
    for (int i = 0; i < 8; ++i) t = __fadd_rn(t, scratch[i]);
    scratch[8] = t;
  }
  sync256();
  return scratch[8];
}

// One row per CTA of 9 warps. Warp 8 runs the Sinkhorn for comb (needed only by the next
// expansion), off the critical path; warps 0..7 compute pre and post, the collapse into
// shared memory and the sublayer's RMSNorm.
__global__ void __launch_bounds__(288) hc_finish_kernel(
    const float* __restrict__ partials, int slices, const float* __restrict__ base,
    const float* __restrict__ scale, const uint16_t* __restrict__ streams, const uint16_t* __restrict__ nw,
    float* __restrict__ pre_out, float* __restrict__ post_out, float* __restrict__ comb_out,
    uint16_t* __restrict__ collapsed, uint16_t* __restrict__ normed, uint8_t* __restrict__ q,
    float* __restrict__ qs, int hidden) {
  extern __shared__ float vals[];  // [hidden]: the collapsed row
  __shared__ float proj[kPartial];
  __shared__ float pre[4];
  __shared__ float scratch[9];
  const int row = blockIdx.x, tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  if (tid < kPartial) {
    const float* p = partials + size_t(row) * slices * kPartial + tid;
    float t = 0.0f;
#pragma unroll 8
    for (int k = 0; k < slices; ++k) t = __fadd_rn(t, p[size_t(k) * kPartial]);
    proj[tid] = t;
  }
  __syncthreads();
  const float r = rms_scale(proj[24], float(4 * hidden), kRmsEps);
  if (warp == 8) {
    float v = 0.0f;
    if (lane < 16) v = __fadd_rn(__fmul_rn(__fmul_rn(proj[8 + lane], r), scale[2]), base[8 + lane]);
    v = sinkhorn16(v);
    if (lane < 16 && comb_out) comb_out[row * 16 + lane] = v;
    return;
  }
  if (warp == 0) {
    if (lane < 4) {
      const float a = __fadd_rn(__fmul_rn(__fmul_rn(proj[lane], r), scale[0]), base[lane]);
      const float p = __fadd_rn(sigmoid_f32(a), kHcEps);
      pre[lane] = p;
      if (pre_out) pre_out[row * 4 + lane] = p;
    } else if (lane < 8) {
      const int j = lane - 4;
      const float b = __fadd_rn(__fmul_rn(__fmul_rn(proj[4 + j], r), scale[1]), base[4 + j]);
      if (post_out) post_out[row * 4 + j] = __fmul_rn(2.0f, sigmoid_f32(b));
    }
  }
  sync256();

  // Collapse: bf16((((0 + p0 s0) + p1 s1) + p2 s2) + p3 s3); the sum of squares of the result.
  const float p0 = pre[0], p1 = pre[1], p2 = pre[2], p3 = pre[3];
  const uint16_t* srow = streams + size_t(row) * 4 * hidden;
  const int chunks = hidden >> 3;
  float sq = 0.0f;
  for (int c = tid; c < chunks; c += 256) {
    float x[4][8];
#pragma unroll
    for (int j = 0; j < 4; ++j) unpack_bf16x8(*reinterpret_cast<const uint4*>(srow + j * hidden + 8 * c), x[j]);
    float o[8];
#pragma unroll
    for (int k = 0; k < 8; ++k) {
      float a = 0.0f;
      a = __fadd_rn(a, __fmul_rn(p0, x[0][k]));
      a = __fadd_rn(a, __fmul_rn(p1, x[1][k]));
      a = __fadd_rn(a, __fmul_rn(p2, x[2][k]));
      a = __fadd_rn(a, __fmul_rn(p3, x[3][k]));
      o[k] = bf16_round(a);
      vals[8 * c + k] = o[k];
      sq = __fmaf_rn(o[k], o[k], sq);
    }
    if (collapsed) {
      uint4 ov;
      ov.x = pack_bf16x2(o[0], o[1]); ov.y = pack_bf16x2(o[2], o[3]);
      ov.z = pack_bf16x2(o[4], o[5]); ov.w = pack_bf16x2(o[6], o[7]);
      *reinterpret_cast<uint4*>(collapsed + size_t(row) * hidden + 8 * c) = ov;
    }
  }
  if (!nw) return;  // uniform across the block
  const float total = block_sum_256_named(sq, scratch);
  const float rn = rms_scale(total, float(hidden), kRmsEps);
  auto from_vals = [&](int c, float (&v)[8]) {
#pragma unroll
    for (int k = 0; k < 8; ++k) v[k] = vals[8 * c + k];
  };
  norm_row(from_vals, rn, nw, normed ? normed + size_t(row) * hidden : nullptr, q ? q + size_t(row) * hidden : nullptr,
           qs ? qs + size_t(row) * (hidden / 128) : nullptr, hidden);
}

template <bool Expand>
__global__ void __launch_bounds__(256) hc_head_kernel(
    const uint16_t* __restrict__ streams, const uint16_t* __restrict__ h1, const uint16_t* __restrict__ h2,
    const float* __restrict__ post, const float* __restrict__ comb, const uint16_t* __restrict__ nw,
    uint16_t* __restrict__ out, int hidden) {
  extern __shared__ float vals[];
  __shared__ float scratch[9];
  const int row = blockIdx.x, tid = threadIdx.x;
  const uint16_t* srow = streams + size_t(row) * 4 * hidden;
  const int quads = hidden >> 2;
  // Mean of the 4 streams, 4 positions at a time; thread t owns quads t, t + 256, ...
  // The norm's sum of squares runs in chunks of 8 (two quads), so accumulate per chunk.
  for (int qd = tid; qd < quads; qd += 256) {
    float s[4][4];
    if (Expand) {
      expand4(srow, h1 + size_t(row) * hidden, h2 ? h2 + size_t(row) * hidden : nullptr, post + row * 4,
              comb + row * 16, hidden, 4 * qd, s);
    } else {
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        const uint2 a = *reinterpret_cast<const uint2*>(srow + j * hidden + 4 * qd);
        s[j][0] = bf16_lo(a.x); s[j][1] = bf16_hi(a.x); s[j][2] = bf16_lo(a.y); s[j][3] = bf16_hi(a.y);
      }
    }
#pragma unroll
    for (int k = 0; k < 4; ++k)
      vals[4 * qd + k] = bf16_round(__fmul_rn(__fadd_rn(__fadd_rn(__fadd_rn(s[0][k], s[1][k]), s[2][k]), s[3][k]), 0.25f));
  }
  __syncthreads();
  float sq = 0.0f;
  for (int c = tid; c < (hidden >> 3); c += 256)
#pragma unroll
    for (int k = 0; k < 8; ++k) sq = __fmaf_rn(vals[8 * c + k], vals[8 * c + k], sq);
  const float total = block_sum_256(sq, scratch);
  const float r = rms_scale(total, float(hidden), kRmsEps);
  auto from_vals = [&](int c, float (&v)[8]) {
#pragma unroll
    for (int k = 0; k < 8; ++k) v[k] = vals[8 * c + k];
  };
  norm_row(from_vals, r, nw, out + size_t(row) * hidden, nullptr, nullptr, hidden);
}

__global__ void hc_broadcast_kernel(const uint4* __restrict__ embed, uint4* __restrict__ streams, int per_row,
                                    long total) {
  for (long i = blockIdx.x * long(blockDim.x) + threadIdx.x; i < total; i += long(gridDim.x) * blockDim.x) {
    const long row = i / per_row, c = i % per_row;
    const uint4 v = embed[row * per_row + c];
#pragma unroll
    for (int j = 0; j < 4; ++j) streams[(row * 4 + j) * per_row + c] = v;
  }
}

}  // namespace
}  // namespace glm53f

using namespace glm53f;

extern "C" int32_t glm53f_hc_broadcast(const uint16_t* embed, uint16_t* streams, int32_t rows, int32_t hidden,
                                       cudaStream_t stream) {
  if (rows < 1 || hidden < 8 || hidden % 8 || !aligned16(embed) || !aligned16(streams)) return cudaErrorInvalidValue;
  const int per_row = hidden / 8;
  const long total = long(rows) * per_row;
  const long want = (total + 255) / 256;
  const int blocks = int(want < 4096 ? want : 4096);
  hc_broadcast_kernel<<<blocks, 256, 0, stream>>>(reinterpret_cast<const uint4*>(embed),
                                                  reinterpret_cast<uint4*>(streams), per_row, total);
  return cudaGetLastError();
}

extern "C" int32_t glm53f_hc_project(const uint16_t* streams_in, const uint16_t* block_out, const uint16_t* block_out2,
                                     const float* post, const float* comb, uint16_t* streams_out, const uint16_t* fn,
                                     float* partials, int32_t rows, int32_t hidden, cudaStream_t stream) {
  if (rows < 1 || hidden < kSlice || hidden % kSlice || !aligned16(streams_in)) return cudaErrorInvalidValue;
  const bool expand = block_out != nullptr;
  if (expand && (!aligned16(block_out) || !aligned16_or_null(block_out2) || !post || !comb || !aligned16(streams_out)))
    return cudaErrorInvalidValue;
  if (!expand && (streams_out || block_out2 || !fn)) return cudaErrorInvalidValue;
  if (fn && (!aligned16(fn) || !aligned16(partials))) return cudaErrorInvalidValue;
  const int slices = hidden / kSlice;
  // Up to 8 rows (decode): one row per CTA, so the rows spread over slices x rows CTAs. More
  // rows: groups of 8 per CTA step, each CTA keeping its fn slice in registers across
  // several groups. Each row's arithmetic is the same either way.
  if (rows <= kRowsPerGroup) {
    const dim3 grid(slices, rows);
    if (expand && fn)
      hc_project_kernel<true, true, 1><<<grid, 256, 0, stream>>>(streams_in, block_out, block_out2, post, comb,
                                                                 streams_out, fn, partials, rows, hidden, 1);
    else if (expand)
      hc_project_kernel<true, false, 1><<<grid, 256, 0, stream>>>(streams_in, block_out, block_out2, post, comb,
                                                                  streams_out, fn, partials, rows, hidden, 1);
    else
      hc_project_kernel<false, true, 1><<<grid, 256, 0, stream>>>(streams_in, nullptr, nullptr, nullptr, nullptr,
                                                                  nullptr, fn, partials, rows, hidden, 1);
    return cudaGetLastError();
  }
  const int groups = (rows + kRowsPerGroup - 1) / kRowsPerGroup;
  const int gpc = groups <= 16 ? 1 : (groups + 15) / 16;
  const dim3 grid(slices, (groups + gpc - 1) / gpc);
  if (expand && fn)
    hc_project_kernel<true, true, kRowsPerGroup><<<grid, 256, 0, stream>>>(
        streams_in, block_out, block_out2, post, comb, streams_out, fn, partials, rows, hidden, gpc);
  else if (expand)
    hc_project_kernel<true, false, kRowsPerGroup><<<grid, 256, 0, stream>>>(
        streams_in, block_out, block_out2, post, comb, streams_out, fn, partials, rows, hidden, gpc);
  else
    hc_project_kernel<false, true, kRowsPerGroup><<<grid, 256, 0, stream>>>(
        streams_in, nullptr, nullptr, nullptr, nullptr, nullptr, fn, partials, rows, hidden, gpc);
  return cudaGetLastError();
}


extern "C" int32_t glm53f_hc_finish(const float* partials, const float* base, const float* scale,
                                    const uint16_t* streams, const uint16_t* norm_weight, float* pre, float* post,
                                    float* comb, uint16_t* collapsed, uint16_t* normed, uint8_t* normed_q,
                                    float* normed_scales, int32_t rows, int32_t hidden, cudaStream_t stream) {
  if (rows < 1 || hidden < kSlice || hidden % kSlice || !partials || !base || !scale || !aligned16(streams))
    return cudaErrorInvalidValue;
  if (!aligned16_or_null(collapsed) || !aligned16_or_null(normed) || !aligned16_or_null(norm_weight) ||
      !aligned16_or_null(normed_q))
    return cudaErrorInvalidValue;
  if ((normed || normed_q) && !norm_weight) return cudaErrorInvalidValue;
  // The quantization's 128-groups are 16 lanes wide; whole warps need hidden % 256 == 0.
  if (normed_q && (!normed_scales || hidden % 256)) return cudaErrorInvalidValue;
  const size_t smem = size_t(hidden) * sizeof(float);  // the collapsed row
  if (smem > 48 * 1024) {
    const cudaError_t e = cudaFuncSetAttribute(hc_finish_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, int(smem));
    if (e != cudaSuccess) return e;
  }
  hc_finish_kernel<<<rows, 288, smem, stream>>>(partials, hidden / kSlice, base, scale, streams, norm_weight, pre, post,
                                                comb, collapsed, normed, normed_q, normed_scales, hidden);
  return cudaGetLastError();
}

extern "C" int32_t glm53f_hc_head(const uint16_t* streams, const uint16_t* block_out, const uint16_t* block_out2,
                                  const float* post, const float* comb, const uint16_t* norm_weight, uint16_t* out,
                                  int32_t rows, int32_t hidden, cudaStream_t stream) {
  if (rows < 1 || hidden < 8 || hidden % 8 || !aligned16(streams) || !aligned16(norm_weight) || !aligned16(out))
    return cudaErrorInvalidValue;
  if (block_out && (!aligned16(block_out) || !aligned16_or_null(block_out2) || !post || !comb))
    return cudaErrorInvalidValue;
  const size_t smem = size_t(hidden) * sizeof(float);
  if (block_out) {
    if (smem > 48 * 1024) {
      const cudaError_t e = cudaFuncSetAttribute(hc_head_kernel<true>, cudaFuncAttributeMaxDynamicSharedMemorySize, int(smem));
      if (e != cudaSuccess) return e;
    }
    hc_head_kernel<true><<<rows, 256, smem, stream>>>(streams, block_out, block_out2, post, comb, norm_weight, out, hidden);
  } else {
    if (smem > 48 * 1024) {
      const cudaError_t e = cudaFuncSetAttribute(hc_head_kernel<false>, cudaFuncAttributeMaxDynamicSharedMemorySize, int(smem));
      if (e != cudaSuccess) return e;
    }
    hc_head_kernel<false><<<rows, 256, smem, stream>>>(streams, nullptr, nullptr, nullptr, nullptr, norm_weight, out, hidden);
  }
  return cudaGetLastError();
}
