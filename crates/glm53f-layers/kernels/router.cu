// Router kernels: f32 logits (one warp per expert and row, or per expert and 8 rows above 8
// rows), then a per-row top-k over sigmoid scores plus the correction bias.
//
// Logits: lane l of the expert's warp takes k = 8 l + 256 i + j (16-byte loads of 8 BF16),
// fma over i then j, butterfly across lanes (src/router.rs logits_row). Above 8 rows a
// warp reuses each weight load for 8 activation rows.
//
// Selection: one warp per row, top_k rounds of a warp-wide argmax (two redux instructions:
// the largest key, then the lowest expert holding it); the largest corrected score wins and
// an exact tie goes to the lower expert index, as src/router.rs select_row decides every
// case (NaN, -0, repeated winners). Any expert count up to 1024, top-k up to 32.
#include "glm53f_layers.h"
#include "common.cuh"
#include <math_constants.h>

namespace glm53f {
namespace {

constexpr int kRows = 8;
constexpr int kExpertsPerCta = 4;

// One warp's logit for expert `e` and rows r0 .. r0 + nr - 1, stored by lane 0. With one row,
// the weight and activation loads of all 16 steps of a 4,096-wide row are issued together (a
// step past the row's end reloads its last step and is not used), so the row streams in one
// round trip instead of one per step. With 8 rows each weight load serves 8 activation rows,
// one step at a time.
template <int R>
__device__ __forceinline__ void logits_warp(const uint16_t* __restrict__ x, const uint16_t* __restrict__ w,
                                            float* __restrict__ logits, int experts, int hidden, int e, int r0,
                                            int nr, int lane) {
  float acc[R];
#pragma unroll
  for (int r = 0; r < R; ++r) acc[r] = 0.0f;
  const uint16_t* wr = w + size_t(e) * hidden + 8 * lane;
  const int steps = hidden >> 8;
  if constexpr (R == 1) {
    constexpr int kBatch = 16;
    const uint16_t* xr = x + size_t(r0) * hidden + 8 * lane;
#pragma unroll 1
    for (int i0 = 0; i0 < steps; i0 += kBatch) {
      uint4 wv[kBatch], xv[kBatch];
#pragma unroll
      for (int i = 0; i < kBatch; ++i) {
        const int step = min(i0 + i, steps - 1);
        wv[i] = ld_stream(wr + 256 * step);
        xv[i] = ld_cached(xr + 256 * step);
      }
#pragma unroll
      for (int i = 0; i < kBatch; ++i) {
        if (i0 + i < steps) {
          float wf[8], xf[8];
          unpack_bf16x8(wv[i], wf);
          unpack_bf16x8(xv[i], xf);
#pragma unroll
          for (int j = 0; j < 8; ++j) acc[0] = __fmaf_rn(xf[j], wf[j], acc[0]);
        }
      }
    }
  } else {
#pragma unroll 1
    for (int i = 0; i < steps; ++i) {
      const int k = 256 * i;
      float wf[8];
      unpack_bf16x8(ld_stream(wr + k), wf);
#pragma unroll
      for (int r = 0; r < R; ++r) {
        if (r < nr) {
          float xf[8];
          unpack_bf16x8(ld_cached(x + size_t(r0 + r) * hidden + 8 * lane + k), xf);
#pragma unroll
          for (int j = 0; j < 8; ++j) acc[r] = __fmaf_rn(xf[j], wf[j], acc[r]);
        }
      }
    }
  }
#pragma unroll
  for (int r = 0; r < R; ++r) {
    if (r < nr) {
      const float v = warp_sum(acc[r]);
      if (lane == 0) logits[size_t(r0 + r) * experts + e] = v;
    }
  }
}


template <int R>
__global__ void __launch_bounds__(32 * kExpertsPerCta) router_logits_kernel(
    const uint16_t* __restrict__ x, const uint16_t* __restrict__ w, float* __restrict__ logits, int rows,
    int experts, int hidden) {
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int e = blockIdx.x * kExpertsPerCta + warp;
  if (e >= experts) return;
  const int r0 = blockIdx.y * R;
  logits_warp<R>(x, w, logits, experts, hidden, e, r0, min(R, rows - r0), lane);
}

// ---- Selection: one warp per row ------------------------------------------------------------
//
// src/router.rs select_row scans the experts from 0 keeping the first strictly greatest
// corrected score, and marks each winner with -inf. As one unsigned key (larger first; equal
// keys go to the lower expert) that order is: corrected scores in float order with -0 == +0;
// a NaN at expert 0 above everything (nothing compares greater than the scan's start); any
// other NaN below everything (it never compares greater). Lane l holds experts l, l + 32, ...
// (S slots); each round takes the warp's largest key with a redux, then the lowest expert
// holding it with a second redux, so the winners are the reference's in every case.
__device__ __forceinline__ uint32_t select_key(float v, int e) {
  const uint32_t b = v == 0.0f ? 0u : __float_as_uint(v);
  uint32_t k = (b & 0x80000000u) ? ~b : (b | 0x80000000u);
  const uint32_t nan_key = e == 0 ? 0xffffffffu : 0u;
  k = v != v ? nan_key : k;
  return k;
}
constexpr uint32_t kChosenKey = 0x007fffffu;  // select_key(-inf, e)

// Slots per lane for up to `experts` experts.
__host__ __device__ constexpr int select_slots(int experts) { return experts <= 288 ? 9 : experts <= 512 ? 16 : 32; }

// Scores (sigmoid of the logit) and keys (of score + bias) of N experts e[i]; experts at or
// past `experts` get key 0 (never chosen). The sigmoids go without a division each (with one
// they would run one after another); the rare ones outside recip_ge1's range (logits below
// about -87, NaN) are divided behind one branch.
template <int N>
__device__ __forceinline__ void scores_keys(const float (&x)[N], const float (&b)[N], const int (&e)[N], int experts,
                                            float (&score)[N], uint32_t (&key)[N]) {
  uint32_t slow = 0;
#pragma unroll
  for (int i = 0; i < N; ++i) {
    bool sl;
    score[i] = sigmoid_fast(x[i], sl);
    slow |= sl ? 1u << i : 0u;
  }
  if (slow) {
#pragma unroll
    for (int i = 0; i < N; ++i)
      if (slow & (1u << i)) score[i] = sigmoid_f32(x[i]);
  }
#pragma unroll
  for (int i = 0; i < N; ++i) key[i] = e[i] < experts ? select_key(__fadd_rn(score[i], b[i]), e[i]) : 0u;
}

// The top_k rounds for one row, by one warp whose lane l holds the keys and scores of experts
// l + 32 i (slot i). Each lane keeps its best slot (the largest key, the lowest slot among
// equals); a round takes the warp's best with two reduxes (the largest key, then the lowest
// expert holding it), marks the winner chosen (the reference sets it to -inf and it stays a
// candidate) and the winning lane rescans its slots. Writes ids[k] and weights[k] (k < top_k).
template <int S>
__device__ __forceinline__ void select_rounds(uint32_t (&key)[S], const float (&score)[S], int top_k, float scale,
                                              int32_t* __restrict__ ids, float* __restrict__ weights, int lane) {
  uint32_t bk;
  int bs;
  float bc;
  auto best = [&]() {
    bk = key[0];
    bs = 0;
    bc = score[0];
#pragma unroll
    for (int i = 1; i < S; ++i)
      if (key[i] > bk) {
        bk = key[i];
        bs = i;
        bc = score[i];
      }
  };
  best();
  float total = 0.0f, mine = 0.0f;
  uint32_t myid = 0;
  for (int k = 0; k < top_k; ++k) {
    const uint32_t mk = __reduce_max_sync(0xffffffffu, bk);
    const uint32_t win = __reduce_min_sync(0xffffffffu, bk == mk ? uint32_t(lane + 32 * bs) : 0xffffffffu);
    const int owner = int(win & 31u);
    const float s = __shfl_sync(0xffffffffu, bc, owner);
    total = __fadd_rn(total, s);  // the weights' sum, in selection order
    if (lane == k) {
      mine = s;
      myid = win;
    }
    if (lane == owner) {
#pragma unroll
      for (int i = 0; i < S; ++i)
        if (i == bs) key[i] = kChosenKey;
      best();
    }
  }
  if (lane < top_k) {
    ids[lane] = int32_t(myid);
    weights[lane] = __fmul_rn(__fdiv_rn(mine, __fadd_rn(total, kNormDenomEps)), scale);
  }
}

// One row's selection by one warp, from logits in global memory. `Coherent` reads them through
// L2 only (other CTAs of the same launch wrote them).
template <int S, bool Coherent>
__device__ __forceinline__ void select_row_warp(const float* logits_row, const float* __restrict__ bias, int experts,
                                                int top_k, float scale, int32_t* __restrict__ ids,
                                                float* __restrict__ weights, int lane) {
  float x[S], b[S];
  int e[S];
#pragma unroll
  for (int i = 0; i < S; ++i) {
    e[i] = lane + 32 * i;
    x[i] = e[i] < experts ? (Coherent ? __ldcg(logits_row + e[i]) : logits_row[e[i]]) : 0.0f;
    b[i] = e[i] < experts ? __ldg(bias + e[i]) : 0.0f;
  }
  // Every load issued before any is used: the barrier keeps the compiler from sinking each load
  // to its use (one round trip per slot).
  __syncwarp();
  float score[S];
  uint32_t key[S];
  scores_keys(x, b, e, experts, score, key);
  select_rounds<S>(key, score, top_k, scale, ids, weights, lane);
}

// Selection of `rows` rows from logits a previous launch wrote: one warp per row, 4 per CTA.
template <int S>
__global__ void __launch_bounds__(128) router_select_kernel(const float* __restrict__ logits,
                                                            const float* __restrict__ bias, int32_t* __restrict__ ids,
                                                            float* __restrict__ weights, int rows, int experts,
                                                            int top_k, float scale) {
  const int row = blockIdx.x * 4 + (threadIdx.x >> 5);
  if (row >= rows) return;
  select_row_warp<S, false>(logits + size_t(row) * experts, bias, experts, top_k, scale, ids + size_t(row) * top_k,
                            weights + size_t(row) * top_k, threadIdx.x & 31);
}

// Logits and selection in one launch: CTA (b, y) computes router_logits_kernel's logits for
// experts kFusedExperts * b .. and the rows of block-row y; the last CTA of each block-row to
// finish (a per-block-row counter) selects that block-row's experts, one warp per row.
constexpr int kFusedExperts = 4;
template <int R, int S>
__global__ void __launch_bounds__(32 * kFusedExperts) router_fused_kernel(
    const uint16_t* __restrict__ x, const uint16_t* __restrict__ w, const float* __restrict__ bias, float* logits,
    unsigned* __restrict__ sync, int32_t* __restrict__ ids, float* __restrict__ weights, int rows, int experts,
    int hidden, int top_k, float scale) {
  __shared__ int last;
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int e = blockIdx.x * kFusedExperts + warp;
  const int r0 = blockIdx.y * R;
  const int nr = min(R, rows - r0);
  if (e < experts) logits_warp<R>(x, w, logits, experts, hidden, e, r0, nr, lane);
  // Signal (atomic_add_acq_rel); the last CTA of the block-row selects.
  __syncthreads();
  if (threadIdx.x == 0) last = atomic_add_acq_rel(sync + blockIdx.y, 1u) == gridDim.x - 1;
  __syncthreads();
  if (!last) return;
  if (threadIdx.x == 0) sync[blockIdx.y] = 0;  // every CTA of the block-row has signalled
  if (R == 1) {
    // One row: all 128 threads compute its scores and keys (the logits read through L2), then
    // warp 0 selects.
    constexpr int kPer = (32 * S + 32 * kFusedExperts - 1) / (32 * kFusedExperts);
    __shared__ uint32_t skey[32 * S];
    __shared__ float sscore[32 * S];
    const float* lrow = logits + size_t(r0) * experts;
    float xv[kPer], bv[kPer];
    int ev[kPer];
#pragma unroll
    for (int j = 0; j < kPer; ++j) {
      ev[j] = threadIdx.x + 32 * kFusedExperts * j;
      xv[j] = ev[j] < experts ? __ldcg(lrow + ev[j]) : 0.0f;
      bv[j] = ev[j] < experts ? __ldg(bias + ev[j]) : 0.0f;
    }
    __syncwarp();  // every load issued before any is used
    float sc[kPer];
    uint32_t ky[kPer];
    scores_keys(xv, bv, ev, experts, sc, ky);
#pragma unroll
    for (int j = 0; j < kPer; ++j)
      if (ev[j] < 32 * S) {
        skey[ev[j]] = ky[j];
        sscore[ev[j]] = sc[j];
      }
    __syncthreads();
    if (warp == 0) {
      uint32_t key[S];
      float score[S];
#pragma unroll
      for (int i = 0; i < S; ++i) {
        key[i] = skey[lane + 32 * i];
        score[i] = sscore[lane + 32 * i];
      }
      select_rounds<S>(key, score, top_k, scale, ids + size_t(r0) * top_k, weights + size_t(r0) * top_k, lane);
    }
  } else {
    for (int rr = warp; rr < nr; rr += kFusedExperts) {
      const size_t row = size_t(r0 + rr);
      select_row_warp<S, true>(logits + row * experts, bias, experts, top_k, scale, ids + row * top_k,
                               weights + row * top_k, lane);
    }
  }
}

}  // namespace
}  // namespace glm53f

using namespace glm53f;

extern "C" int32_t glm53f_router_logits(const uint16_t* x, const uint16_t* weight, float* logits, int32_t rows,
                                        int32_t experts, int32_t hidden, cudaStream_t stream) {
  if (rows < 1 || experts < 1 || hidden < 256 || hidden % 256 || !aligned16(x) || !aligned16(weight) || !logits)
    return cudaErrorInvalidValue;
  const dim3 block(32 * kExpertsPerCta);
  const int eblocks = (experts + kExpertsPerCta - 1) / kExpertsPerCta;
  // Up to 8 rows (decode): one row per warp, so the work spreads over experts x rows warps.
  // More rows: 8 rows per warp, reusing each weight load 8 times. Each row's arithmetic is
  // the same either way.
  if (rows <= kRows)
    router_logits_kernel<1><<<dim3(eblocks, rows), block, 0, stream>>>(x, weight, logits, rows, experts, hidden);
  else
    router_logits_kernel<kRows><<<dim3(eblocks, (rows + kRows - 1) / kRows), block, 0, stream>>>(x, weight, logits, rows,
                                                                                                experts, hidden);
  return cudaGetLastError();
}

extern "C" int32_t glm53f_router_select(const float* logits, const float* bias, int32_t* ids, float* weights,
                                        int32_t rows, int32_t experts, int32_t top_k, float scale,
                                        cudaStream_t stream) {
  if (rows < 1 || experts < 1 || experts > 1024 || top_k < 1 || top_k > 32 || top_k > experts || !logits || !bias ||
      !ids || !weights)
    return cudaErrorInvalidValue;
  const unsigned blocks = unsigned((rows + 3) / 4);
  if (experts <= 288)
    router_select_kernel<select_slots(288)><<<blocks, 128, 0, stream>>>(logits, bias, ids, weights, rows, experts, top_k, scale);
  else if (experts <= 512)
    router_select_kernel<select_slots(512)><<<blocks, 128, 0, stream>>>(logits, bias, ids, weights, rows, experts, top_k, scale);
  else
    router_select_kernel<select_slots(1024)><<<blocks, 128, 0, stream>>>(logits, bias, ids, weights, rows, experts, top_k,
                                                                         scale);
  return cudaGetLastError();
}

template <int S>
static int32_t launch_router_fused(const uint16_t* x, const uint16_t* weight, const float* bias, float* logits,
                                   uint32_t* sync, int32_t* ids, float* weights, int32_t rows, int32_t experts,
                                   int32_t hidden, int32_t top_k, float scale, cudaStream_t stream) {
  const dim3 block(32 * kFusedExperts);
  const int eblocks = (experts + kFusedExperts - 1) / kFusedExperts;
  if (rows <= kRows)
    router_fused_kernel<1, S><<<dim3(eblocks, rows), block, 0, stream>>>(x, weight, bias, logits, sync, ids, weights,
                                                                             rows, experts, hidden, top_k, scale);
  else
    router_fused_kernel<kRows, S><<<dim3(eblocks, (rows + kRows - 1) / kRows), block, 0, stream>>>(
        x, weight, bias, logits, sync, ids, weights, rows, experts, hidden, top_k, scale);
  return cudaGetLastError();
}

extern "C" int32_t glm53f_router_fused(const uint16_t* x, const uint16_t* weight, const float* bias, float* logits,
                                       uint32_t* sync, int32_t* ids, float* weights, int32_t rows, int32_t experts,
                                       int32_t hidden, int32_t top_k, float scale, cudaStream_t stream) {
  if (rows < 1 || experts < 1 || experts > 1024 || hidden < 256 || hidden % 256 || top_k < 1 || top_k > 32 ||
      top_k > experts || !aligned16(x) || !aligned16(weight) || !bias || !logits || !sync || !ids || !weights)
    return cudaErrorInvalidValue;
  if (experts <= 288)
    return launch_router_fused<select_slots(288)>(x, weight, bias, logits, sync, ids, weights, rows, experts, hidden,
                                                  top_k, scale, stream);
  if (experts <= 512)
    return launch_router_fused<select_slots(512)>(x, weight, bias, logits, sync, ids, weights, rows, experts, hidden,
                                                  top_k, scale, stream);
  return launch_router_fused<select_slots(1024)>(x, weight, bias, logits, sync, ids, weights, rows, experts, hidden,
                                                 top_k, scale, stream);
}
