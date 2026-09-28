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

// A row's post and comb as the expansion applies them, rounded to BF16.
__device__ __forceinline__ void load_mix(const float* __restrict__ post, const float* __restrict__ comb, float (&pb)[4],
                                         float (&cb)[16]) {
#pragma unroll
  for (int i = 0; i < 4; ++i) pb[i] = bf16_round(post[i]);
#pragma unroll
  for (int i = 0; i < 16; ++i) cb[i] = bf16_round(comb[i]);
}

// The expanded (new) values of stream positions d..d+3 for all 4 destination streams, with
// the row's load_mix values.
__device__ __forceinline__ void expand4(const uint16_t* __restrict__ res_row,  // [4][hidden]
                                        const uint16_t* __restrict__ h1, const uint16_t* __restrict__ h2,
                                        const float (&pb4)[4], const float (&cb16)[16], int hidden, int d,
                                        float (&out)[4][4]) {
  // Every input is loaded before any is used, so they take one round trip together.
  const uint2 a = *reinterpret_cast<const uint2*>(h1 + d);
  const uint2 b = h2 ? *reinterpret_cast<const uint2*>(h2 + d) : make_uint2(0u, 0u);
  uint2 rv[4];
#pragma unroll
  for (int j = 0; j < 4; ++j) rv[j] = *reinterpret_cast<const uint2*>(res_row + j * hidden + d);
  float h[4] = {bf16_lo(a.x), bf16_hi(a.x), bf16_lo(a.y), bf16_hi(a.y)};
  if (h2) {
    h[0] = bf16_round(__fadd_rn(h[0], bf16_lo(b.x)));
    h[1] = bf16_round(__fadd_rn(h[1], bf16_hi(b.x)));
    h[2] = bf16_round(__fadd_rn(h[2], bf16_lo(b.y)));
    h[3] = bf16_round(__fadd_rn(h[3], bf16_hi(b.y)));
  }
  float res[4][4];
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    res[j][0] = bf16_lo(rv[j].x); res[j][1] = bf16_hi(rv[j].x); res[j][2] = bf16_lo(rv[j].y); res[j][3] = bf16_hi(rv[j].y);
  }
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const float pb = pb4[i];
    float cb[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) cb[j] = cb16[4 * j + i];
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

  // This warp's fn values: projections 3*warp + q, streams st, positions d..d+3. Unpacked only
  // when used, so the expansion's loads go out without waiting for them.
  uint2 wraw[3][4];
  if (Project) {
#pragma unroll
    for (int q = 0; q < 3; ++q)
#pragma unroll
      for (int st = 0; st < 4; ++st)
        wraw[q][st] = __ldg(reinterpret_cast<const uint2*>(fn + size_t(3 * warp + q) * flat + st * hidden + d));
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
          float pb[4], cb[16];
          load_mix(post + row * 4, comb + row * 16, pb, cb);
          expand4(streams_in + row * flat, h1 + row * hidden, h2 ? h2 + row * hidden : nullptr, pb, cb, hidden, d, v);
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
    float w[3][4][4];
#pragma unroll
    for (int q = 0; q < 3; ++q)
#pragma unroll
      for (int st = 0; st < 4; ++st) {
        const uint2 a = wraw[q][st];
        w[q][st][0] = bf16_lo(a.x); w[q][st][1] = bf16_hi(a.x); w[q][st][2] = bf16_lo(a.y); w[q][st][3] = bf16_hi(a.y);
      }
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
    for (int rr = 0; rr < kRowsPerGroup; ++rr) {
      if (rr >= nr) continue;  // uniform across the CTA
#pragma unroll
      for (int q = 0; q < 3; ++q) acc[rr][q] = warp_sum(acc[rr][q]);
      if (warp == 0) sq[rr] = warp_sum(sq[rr]);
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

// The 16-lane Sinkhorn of one row's comb logits (lanes 0..15 hold row-major [i][j]). A row
// sum is the butterfly over lanes xor 1, 2 and a column sum the one over xor 4, 8, each
// taken as sum2_xor (half the shuffle latency, the same bits).
__device__ __forceinline__ float sinkhorn16(float v) {
  const float m = fmaxf(fmaxf(v, __shfl_xor_sync(0xffffffffu, v, 1)),
                        fmaxf(__shfl_xor_sync(0xffffffffu, v, 2), __shfl_xor_sync(0xffffffffu, v, 3)));
  v = exp_f32(__fsub_rn(v, m));
  v = __fadd_rn(__fdiv_rn(v, sum2_xor(v, 1, 2)), kHcEps);
  v = __fdiv_rn(v, __fadd_rn(sum2_xor(v, 4, 8), kHcEps));
  for (int it = 1; it < 20; ++it) {
    v = __fdiv_rn(v, __fadd_rn(sum2_xor(v, 1, 2), kHcEps));
    v = __fdiv_rn(v, __fadd_rn(sum2_xor(v, 4, 8), kHcEps));
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

// ---- The finish, in pieces shared by every kernel that finishes a boundary, so they all
// compute the same bits. ----

// A row's partials [slices][25] (n = slices * 25 floats) are summed in steps, so a kernel can
// issue the loads early and do other work before the sums:
// - load_partials4: thread t's floats [4t, 4t + 4), one 16-byte load when `vec` (the row is
//   16-byte aligned and n % 4 == 0), else scalars; floats at or past n read as 0. `Coherent`
//   reads through L2 only (partials other CTAs of the same launch wrote);
// - store_partials4 puts them in shared `stage` (16-byte aligned);
// - after a barrier, partial_total(j) is total j, summed over the slices in slice order.
__device__ __forceinline__ bool partials_vec(const float* p, int n) {
  return (n & 3) == 0 && (reinterpret_cast<uintptr_t>(p) & 15) == 0;
}
template <bool Coherent>
__device__ __forceinline__ float4 load_partials4(const float* p, int n, bool vec, int t) {
  float4 v = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
  const int e = 4 * t;
  if (e >= n) return v;
  if (vec) return Coherent ? __ldcg(reinterpret_cast<const float4*>(p + e)) : *reinterpret_cast<const float4*>(p + e);
  v.x = Coherent ? __ldcg(p + e) : p[e];
  if (e + 1 < n) v.y = Coherent ? __ldcg(p + e + 1) : p[e + 1];
  if (e + 2 < n) v.z = Coherent ? __ldcg(p + e + 2) : p[e + 2];
  if (e + 3 < n) v.w = Coherent ? __ldcg(p + e + 3) : p[e + 3];
  return v;
}
__device__ __forceinline__ void store_partials4(float* stage, int n, int t, float4 v) {
  const int e = 4 * t;
  if (e + 4 <= n) {
    *reinterpret_cast<float4*>(stage + e) = v;
  } else if (e < n) {
    stage[e] = v.x;
    if (e + 1 < n) stage[e + 1] = v.y;
    if (e + 2 < n) stage[e + 2] = v.z;
  }
}
// Every thread of the block: the whole row into `stage` (the caller synchronizes).
template <bool Coherent>
__device__ __forceinline__ void stage_partials(const float* p, int slices, float* stage) {
  const int n = slices * kPartial;
  const bool vec = partials_vec(p, n);
  for (int t = threadIdx.x; 4 * t < n; t += blockDim.x) store_partials4(stage, n, t, load_partials4<Coherent>(p, n, vec, t));
}
__device__ __forceinline__ float partial_total(const float* stage, int slices, int j) {
  float t = 0.0f;
#pragma unroll 8
  for (int k = 0; k < slices; ++k) t = __fadd_rn(t, stage[k * kPartial + j]);
  return t;
}

// The base and scale a lane of a finishing warp uses: for pre/post (warp 0), lane j < 8 takes
// base[j] and scale[j / 4]; for comb (warp 8), lane j < 16 takes base[8 + j] and scale[2].
struct MixWeights {
  float base, scale;
};
__device__ __forceinline__ MixWeights pre_post_weights(const float* base, const float* scale, int lane) {
  return lane < 8 ? MixWeights{base[lane], scale[lane >> 2]} : MixWeights{0.0f, 0.0f};
}
__device__ __forceinline__ MixWeights comb_weights(const float* base, const float* scale, int lane) {
  return lane < 16 ? MixWeights{base[8 + lane], scale[2]} : MixWeights{0.0f, 0.0f};
}

// pre (lanes 0..3) and post (lanes 4..7) of a warp whose lane j holds projection total j;
// pre also goes to shared `pre_s`. r is the RMS scale.
__device__ __forceinline__ void pre_post(float t, float r, MixWeights w, int lane, float* pre_s, float* pre_out,
                                         float* post_out) {
  if (lane < 4) {
    const float a = __fadd_rn(__fmul_rn(__fmul_rn(t, r), w.scale), w.base);
    const float p = __fadd_rn(sigmoid_f32(a), kHcEps);
    pre_s[lane] = p;
    if (pre_out) pre_out[lane] = p;
  } else if (lane < 8) {
    const float b = __fadd_rn(__fmul_rn(__fmul_rn(t, r), w.scale), w.base);
    if (post_out) post_out[lane - 4] = __fmul_rn(2.0f, sigmoid_f32(b));
  }
}

// Named barriers: 1 = warps 0..7 (sync256 below), 2 = warp 0 hands the projection totals to
// the comb warp (8).
__device__ __forceinline__ void comb_handoff_arrive() { asm volatile("bar.arrive 2, 64;\n" ::: "memory"); }
__device__ __forceinline__ void comb_handoff_wait() { asm volatile("bar.sync 2, 64;\n" ::: "memory"); }

// Warp 0 of a finishing block, given its lane's projection total t (lane j < 25 holds total
// j; total 24 is the sum of squares): hands the totals to the comb warp through shared
// `proj` when `comb` (the comb warp waits for them rather than summing them itself, which
// measured slower), then lanes 0..7 compute pre and post.
__device__ __forceinline__ void warp0_finish(float t, int flat, MixWeights w, int lane, bool comb, float* proj,
                                             float* pre_s, float* pre_out, float* post_out) {
  if (comb) {
    if (lane < kPartial) proj[lane] = t;
    comb_handoff_arrive();
  }
  const float r = rms_scale(__shfl_sync(0xffffffffu, t, 24), float(flat), kRmsEps);
  pre_post(t, r, w, lane, pre_s, pre_out, post_out);
}

// comb entry `lane` (lanes 0..15; the whole warp must call it).
__device__ __forceinline__ float comb_value(const float* proj, float r, MixWeights w, int lane) {
  float v = 0.0f;
  if (lane < 16) v = __fadd_rn(__fmul_rn(__fmul_rn(proj[8 + lane], r), w.scale), w.base);
  return sinkhorn16(v);
}

// Collapse of 8 positions: bf16((((0 + p0 s0) + p1 s1) + p2 s2) + p3 s3).
__device__ __forceinline__ void collapse8(const float (&x)[4][8], float p0, float p1, float p2, float p3,
                                          float (&o)[8]) {
#pragma unroll
  for (int k = 0; k < 8; ++k) {
    float a = 0.0f;
    a = __fadd_rn(a, __fmul_rn(p0, x[0][k]));
    a = __fadd_rn(a, __fmul_rn(p1, x[1][k]));
    a = __fadd_rn(a, __fmul_rn(p2, x[2][k]));
    a = __fadd_rn(a, __fmul_rn(p3, x[3][k]));
    o[k] = bf16_round(a);
  }
}

__device__ __forceinline__ void store_bf16x8(uint16_t* p, const float (&o)[8]) {
  uint4 ov;
  ov.x = pack_bf16x2(o[0], o[1]); ov.y = pack_bf16x2(o[2], o[3]);
  ov.z = pack_bf16x2(o[4], o[5]); ov.w = pack_bf16x2(o[6], o[7]);
  *reinterpret_cast<uint4*>(p) = ov;
}

// Normalize chunk c (8 positions) with its weights: bf16(w * bf16(v * r)); write it to
// `normed` and, when `q` is set, its E4M3 codes and the 128-group scale. With `q`, the 16
// lanes that own a 128-group must call this together.
__device__ __forceinline__ void norm8(const float (&v)[8], const float (&wf)[8], float r, uint16_t* __restrict__ normed,
                                      uint8_t* __restrict__ q, float* __restrict__ qs, int c) {
  float o[8];
  float amax = 0.0f;
#pragma unroll
  for (int k = 0; k < 8; ++k) {
    o[k] = bf16_round(__fmul_rn(wf[k], bf16_round(__fmul_rn(v[k], r))));
    amax = fmaxf(amax, fabsf(o[k]));
  }
  if (normed) store_bf16x8(normed + 8 * c, o);
  if (q) {
    const float scale = group_scale(group16_max(amax));
    uint32_t code[8];
    e4m3_of_quotients(o, scale, __fdiv_rn(1.0f, scale), code);
    uint32_t lo = 0, hi = 0;
#pragma unroll
    for (int k = 0; k < 4; ++k) lo |= code[k] << (8 * k);
#pragma unroll
    for (int k = 0; k < 4; ++k) hi |= code[4 + k] << (8 * k);
    *reinterpret_cast<uint2*>(q + 8 * c) = make_uint2(lo, hi);
    if ((c & 15) == 0) qs[c >> 4] = scale;
  }
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
    norm8(v, wf, r, normed, q, qs, c);
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
  extern __shared__ __align__(16) float vals[];  // [hidden]: the collapsed row, then [slices][25] partials
  __shared__ float proj[kPartial];
  __shared__ float pre[4];
  __shared__ float scratch[9];
  const int row = blockIdx.x, tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const MixWeights w = warp == 8 ? comb_weights(base, scale, lane) : pre_post_weights(base, scale, lane);
  float* stage = vals + hidden;
  stage_partials<false>(partials + size_t(row) * slices * kPartial, slices, stage);
  __syncthreads();
  if (warp == 8) {
    if (comb_out) {
      comb_handoff_wait();
      const float r = rms_scale(proj[24], float(4 * hidden), kRmsEps);
      const float v = comb_value(proj, r, w, lane);
      if (lane < 16) comb_out[row * 16 + lane] = v;
    }
    return;
  }
  if (warp == 0)
    warp0_finish(lane < kPartial ? partial_total(stage, slices, lane) : 0.0f, 4 * hidden, w, lane, comb_out != nullptr,
                 proj, pre, pre_out ? pre_out + row * 4 : nullptr, post_out ? post_out + row * 4 : nullptr);
  sync256();

  const float p0 = pre[0], p1 = pre[1], p2 = pre[2], p3 = pre[3];
  const uint16_t* srow = streams + size_t(row) * 4 * hidden;
  const int chunks = hidden >> 3;
  float sq = 0.0f;
  for (int c = tid; c < chunks; c += 256) {
    float x[4][8];
#pragma unroll
    for (int j = 0; j < 4; ++j) unpack_bf16x8(*reinterpret_cast<const uint4*>(srow + j * hidden + 8 * c), x[j]);
    float o[8];
    collapse8(x, p0, p1, p2, p3, o);
#pragma unroll
    for (int k = 0; k < 8; ++k) {
      vals[8 * c + k] = o[k];
      sq = __fmaf_rn(o[k], o[k], sq);
    }
    if (collapsed) store_bf16x8(collapsed + size_t(row) * hidden + 8 * c, o);
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

// Up to this many rows per decode boundary launch, and this hidden size at most (the
// finishing CTA keeps two 8-position chunks per thread in registers).
constexpr int kDecodeRows = 8;
constexpr int kDecodeMaxHidden = 4096;

// A whole decode boundary in one launch: CTA (s, r) is hc_project_kernel's CTA for slice s
// of row r (with the expansion in front); the last CTA of a row to finish its slice, found
// with a per-row counter, then runs hc_finish_kernel's work for that row. The arithmetic is
// the same code as the two kernels', so the results are the same bits. Streams and weights
// may be updated in place (each (row, slice) region is read and written by one CTA, and the
// finish reads only after every CTA of its row has signalled).
template <bool Expand>
__global__ void __launch_bounds__(288, 2) hc_boundary_decode_kernel(
    const uint16_t* streams_in, const uint16_t* __restrict__ h1, const uint16_t* __restrict__ h2,
    const float* post_in, const float* comb_in, uint16_t* streams_out, const uint16_t* __restrict__ fn,
    const float* __restrict__ base, const float* __restrict__ scale, const uint16_t* __restrict__ nw,
    float* __restrict__ partials, unsigned* __restrict__ sync, float* __restrict__ pre_out, float* post_out,
    float* comb_out, uint16_t* __restrict__ collapsed, uint16_t* __restrict__ normed, uint8_t* __restrict__ q,
    float* __restrict__ qs, int hidden) {
  __shared__ float4 xs[4][32];
  __shared__ __align__(16) float stage[kDecodeMaxHidden / kSlice * kPartial];
  __shared__ float proj[kPartial];
  __shared__ float pre[4];
  __shared__ float scratch[9];
  __shared__ int last;
  const int s = blockIdx.x, slices = gridDim.x, row = blockIdx.y;
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int flat = 4 * hidden;
  float* prow = partials + size_t(row) * slices * kPartial;

  // ---- This slice's partials, as hc_project_kernel<Expand, true, 1> computes them. ----
  {
    const int d = s * kSlice + 4 * lane;
    // fn first, unpacked only after the barrier, so warp 0's expansion loads go out with it.
    uint2 wraw[3][4];
    if (warp < 8) {
#pragma unroll
      for (int qq = 0; qq < 3; ++qq)
#pragma unroll
        for (int st = 0; st < 4; ++st)
          wraw[qq][st] = __ldg(reinterpret_cast<const uint2*>(fn + size_t(3 * warp + qq) * flat + st * hidden + d));
    }
    if (warp == 0) {
      const size_t r = size_t(row);
      if (Expand) {
        float v[4][4];
        float pb[4], cb[16];
        load_mix(post_in + r * 4, comb_in + r * 16, pb, cb);
        expand4(streams_in + r * flat, h1 + r * hidden, h2 ? h2 + r * hidden : nullptr, pb, cb, hidden, d, v);
#pragma unroll
        for (int i = 0; i < 4; ++i) {
          uint2 o;
          o.x = pack_bf16x2(v[i][0], v[i][1]);
          o.y = pack_bf16x2(v[i][2], v[i][3]);
          *reinterpret_cast<uint2*>(streams_out + r * flat + i * hidden + d) = o;
          xs[i][lane] = make_float4(v[i][0], v[i][1], v[i][2], v[i][3]);
        }
      } else {
#pragma unroll
        for (int st = 0; st < 4; ++st) {
          const uint2 a = *reinterpret_cast<const uint2*>(streams_in + r * flat + st * hidden + d);
          xs[st][lane] = make_float4(bf16_lo(a.x), bf16_hi(a.x), bf16_lo(a.y), bf16_hi(a.y));
        }
      }
    }
    __syncthreads();
    if (warp < 8) {
      float w[3][4][4];
#pragma unroll
      for (int qq = 0; qq < 3; ++qq)
#pragma unroll
        for (int st = 0; st < 4; ++st) {
          const uint2 a = wraw[qq][st];
          w[qq][st][0] = bf16_lo(a.x); w[qq][st][1] = bf16_hi(a.x); w[qq][st][2] = bf16_lo(a.y); w[qq][st][3] = bf16_hi(a.y);
        }
      float x[4][4];
#pragma unroll
      for (int st = 0; st < 4; ++st) {
        const float4 v = xs[st][lane];
        x[st][0] = v.x; x[st][1] = v.y; x[st][2] = v.z; x[st][3] = v.w;
      }
      float acc[3] = {0.0f, 0.0f, 0.0f}, sq = 0.0f;
#pragma unroll
      for (int st = 0; st < 4; ++st)
#pragma unroll
        for (int i = 0; i < 4; ++i) {
#pragma unroll
          for (int qq = 0; qq < 3; ++qq) acc[qq] = __fmaf_rn(x[st][i], w[qq][st][i], acc[qq]);
          sq = __fmaf_rn(x[st][i], x[st][i], sq);
        }
#pragma unroll
      for (int qq = 0; qq < 3; ++qq) acc[qq] = warp_sum(acc[qq]);
      if (warp == 0) sq = warp_sum(sq);
      if (lane == 0) {
        float* out = prow + size_t(s) * kPartial;
#pragma unroll
        for (int qq = 0; qq < 3; ++qq) out[3 * warp + qq] = acc[qq];
        if (warp == 0) out[24] = sq;
      }
    }
  }

  // ---- Signal (atomic_add_acq_rel); the last CTA of the row finishes it, reading the other
  // CTAs' writes through L2. ----
  __syncthreads();
  if (tid == 0) last = atomic_add_acq_rel(sync + row, 1u) == unsigned(slices - 1);
  __syncthreads();
  if (!last) return;

  // hc_finish_kernel's work for this row. The partials first: everything below waits on them.
  // Every thread loads 4 of them into shared memory; after the barrier warp 0 sums them while
  // warps 1..7 load the streams and norm weights of their two chunks (loading those together
  // with the partials delayed the partials), and warp 8 runs the Sinkhorn.
  const int n = slices * kPartial;
  const float4 pv = load_partials4<true>(prow, n, partials_vec(prow, n), tid);
  const MixWeights w = warp == 8 ? comb_weights(base, scale, lane) : pre_post_weights(base, scale, lane);
  store_partials4(stage, n, tid, pv);
  __syncthreads();
  if (tid == 0) sync[row] = 0;  // every CTA of this row has signalled: re-arm for the next launch
  if (warp == 8) {
    if (comb_out) {
      comb_handoff_wait();
      const float r = rms_scale(proj[24], float(flat), kRmsEps);
      const float v = comb_value(proj, r, w, lane);
      if (lane < 16) comb_out[row * 16 + lane] = v;
    }
    return;
  }
  const uint16_t* srow = (Expand ? streams_out : streams_in) + size_t(row) * flat;
  const int chunks = hidden >> 3;
  // Chunk i of thread t (warps 0..7) is t + 256 i; its validity is uniform per warp.
  const bool has[2] = {tid < chunks, tid + 256 < chunks};
  uint4 sv[2][4], wv[2];
#pragma unroll
  for (int i = 0; i < 2; ++i) {
    const int c = tid + 256 * i;
    if (has[i]) {
#pragma unroll
      for (int j = 0; j < 4; ++j) sv[i][j] = __ldcg(reinterpret_cast<const uint4*>(srow + j * hidden + 8 * c));
      if (nw) wv[i] = __ldg(reinterpret_cast<const uint4*>(nw + 8 * c));
    }
  }
  if (warp == 0)
    warp0_finish(lane < kPartial ? partial_total(stage, slices, lane) : 0.0f, flat, w, lane, comb_out != nullptr, proj,
                 pre, pre_out ? pre_out + row * 4 : nullptr, post_out ? post_out + row * 4 : nullptr);
  sync256();

  // The collapse, kept in registers: thread t owns chunks t and t + 256, as the finish kernel's
  // 256 threads do, so the norm's sum of squares runs in the same order.
  const float p0 = pre[0], p1 = pre[1], p2 = pre[2], p3 = pre[3];
  float o[2][8];
  float sq = 0.0f;
#pragma unroll
  for (int i = 0; i < 2; ++i) {
    if (has[i]) {
      float x[4][8];
#pragma unroll
      for (int j = 0; j < 4; ++j) unpack_bf16x8(sv[i][j], x[j]);
      collapse8(x, p0, p1, p2, p3, o[i]);
#pragma unroll
      for (int k = 0; k < 8; ++k) sq = __fmaf_rn(o[i][k], o[i][k], sq);
      if (collapsed) store_bf16x8(collapsed + size_t(row) * hidden + 8 * (tid + 256 * i), o[i]);
    } else {
#pragma unroll
      for (int k = 0; k < 8; ++k) o[i][k] = 0.0f;
    }
  }
  if (!nw) return;  // uniform across the block
  // block_sum_256_named's sum (butterfly per warp, then the 8 warp totals in order from 0),
  // with every thread adding the 8 totals itself instead of waiting for thread 0.
  {
    const float ws = warp_sum(sq);
    if (lane == 0) scratch[warp] = ws;
  }
  sync256();
  float total = 0.0f;
#pragma unroll
  for (int w = 0; w < 8; ++w) total = __fadd_rn(total, scratch[w]);
  const float rn = rms_scale(total, float(hidden), kRmsEps);

  // norm8 for both chunks together: one pass of group maxima, scales and codes.
  float y[2][8], amax[2] = {0.0f, 0.0f};
#pragma unroll
  for (int i = 0; i < 2; ++i) {
    float wf[8];
    unpack_bf16x8(has[i] ? wv[i] : make_uint4(0u, 0u, 0u, 0u), wf);
#pragma unroll
    for (int k = 0; k < 8; ++k) {
      y[i][k] = bf16_round(__fmul_rn(wf[k], bf16_round(__fmul_rn(o[i][k], rn))));
      amax[i] = fmaxf(amax[i], fabsf(y[i][k]));
    }
    if (normed && has[i]) store_bf16x8(normed + size_t(row) * hidden + 8 * (tid + 256 * i), y[i]);
  }
  if (!q) return;
#pragma unroll
  for (int off = 8; off; off >>= 1)  // group16_max of both chunks
#pragma unroll
    for (int i = 0; i < 2; ++i) amax[i] = fmaxf(amax[i], __shfl_xor_sync(0xffffffffu, amax[i], off));
  float sc[2], inv[2];
#pragma unroll
  for (int i = 0; i < 2; ++i) sc[i] = group_scale(amax[i]);
#pragma unroll
  for (int i = 0; i < 2; ++i) inv[i] = __fdiv_rn(1.0f, sc[i]);
  uint32_t code[2][8];
  e4m3_of_quotients(y, sc, inv, code);
#pragma unroll
  for (int i = 0; i < 2; ++i) {
    if (!has[i]) continue;
    const int c = tid + 256 * i;
    uint32_t lo = 0, hi = 0;
#pragma unroll
    for (int k = 0; k < 4; ++k) lo |= code[i][k] << (8 * k);
#pragma unroll
    for (int k = 0; k < 4; ++k) hi |= code[i][4 + k] << (8 * k);
    *reinterpret_cast<uint2*>(q + size_t(row) * hidden + 8 * c) = make_uint2(lo, hi);
    if ((c & 15) == 0) qs[size_t(row) * (hidden / 128) + (c >> 4)] = sc[i];
  }
}

// comb alone from a boundary's partials (the Sinkhorn off the boundary's critical path).
// Every thread loads partials; warp 0 sums them and runs the Sinkhorn.
__global__ void __launch_bounds__(256) hc_comb_kernel(const float* __restrict__ partials, int slices,
                                                      const float* __restrict__ base, const float* __restrict__ scale,
                                                      float* __restrict__ comb_out, int hidden) {
  extern __shared__ __align__(16) float stage[];  // [slices][25] partials
  __shared__ float proj[kPartial];
  const int row = blockIdx.x, lane = threadIdx.x;
  stage_partials<false>(partials + size_t(row) * slices * kPartial, slices, stage);
  __syncthreads();
  if (threadIdx.x >= 32) return;
  if (lane < kPartial) proj[lane] = partial_total(stage, slices, lane);
  __syncwarp();
  const float r = rms_scale(proj[24], float(4 * hidden), kRmsEps);
  const float v = comb_value(proj, r, comb_weights(base, scale, lane), lane);
  if (lane < 16) comb_out[row * 16 + lane] = v;
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
  float pb[4], cb[16];  // the row's post and comb, loaded once
  if (Expand) load_mix(post + row * 4, comb + row * 16, pb, cb);
  // Mean of the 4 streams, 4 positions at a time; thread t owns quads t, t + 256, ...
  // The norm's sum of squares runs in chunks of 8 (two quads), so accumulate per chunk.
  for (int qd = tid; qd < quads; qd += 256) {
    float s[4][4];
    if (Expand) {
      expand4(srow, h1 + size_t(row) * hidden, h2 ? h2 + size_t(row) * hidden : nullptr, pb, cb, hidden, 4 * qd, s);
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
  // The collapsed row, then the row's partials.
  const size_t smem = (size_t(hidden) + size_t(hidden / kSlice) * kPartial) * sizeof(float);
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

extern "C" int32_t glm53f_hc_boundary_decode(const uint16_t* streams_in, const uint16_t* block_out,
                                             const uint16_t* block_out2, const float* post_in, const float* comb_in,
                                             uint16_t* streams_out, const uint16_t* fn, const float* base,
                                             const float* scale, const uint16_t* norm_weight, float* partials,
                                             uint32_t* sync, float* pre, float* post, float* comb, uint16_t* collapsed,
                                             uint16_t* normed, uint8_t* normed_q, float* normed_scales, int32_t rows,
                                             int32_t hidden, cudaStream_t stream) {
  if (rows < 1 || rows > kDecodeRows || hidden < 256 || hidden % 256 || hidden > kDecodeMaxHidden)
    return cudaErrorInvalidValue;
  if (!aligned16(streams_in) || !aligned16(fn) || !base || !scale || !partials || !sync) return cudaErrorInvalidValue;
  const bool expand = block_out != nullptr;
  if (expand && (!aligned16(block_out) || !aligned16_or_null(block_out2) || !post_in || !comb_in || !aligned16(streams_out)))
    return cudaErrorInvalidValue;
  if (!expand && (streams_out || block_out2)) return cudaErrorInvalidValue;
  if (!aligned16_or_null(collapsed) || !aligned16_or_null(normed) || !aligned16_or_null(norm_weight) ||
      !aligned16_or_null(normed_q))
    return cudaErrorInvalidValue;
  if ((normed || normed_q) && !norm_weight) return cudaErrorInvalidValue;
  if (normed_q && !normed_scales) return cudaErrorInvalidValue;
  const dim3 grid(hidden / kSlice, rows);
  if (expand)
    hc_boundary_decode_kernel<true><<<grid, 288, 0, stream>>>(streams_in, block_out, block_out2, post_in, comb_in,
                                                              streams_out, fn, base, scale, norm_weight, partials, sync,
                                                              pre, post, comb, collapsed, normed, normed_q,
                                                              normed_scales, hidden);
  else
    hc_boundary_decode_kernel<false><<<grid, 288, 0, stream>>>(streams_in, nullptr, nullptr, nullptr, nullptr, nullptr,
                                                               fn, base, scale, norm_weight, partials, sync, pre, post,
                                                               comb, collapsed, normed, normed_q, normed_scales, hidden);
  return cudaGetLastError();
}

extern "C" int32_t glm53f_hc_comb(const float* partials, const float* base, const float* scale, float* comb,
                                  int32_t rows, int32_t hidden, cudaStream_t stream) {
  if (rows < 1 || hidden < kSlice || hidden % kSlice || !partials || !base || !scale || !comb)
    return cudaErrorInvalidValue;
  const size_t smem = size_t(hidden / kSlice) * kPartial * sizeof(float);
  if (smem > 48 * 1024) {
    const cudaError_t e = cudaFuncSetAttribute(hc_comb_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, int(smem));
    if (e != cudaSuccess) return e;
  }
  hc_comb_kernel<<<rows, 256, smem, stream>>>(partials, hidden / kSlice, base, scale, comb, hidden);
  return cudaGetLastError();
}
