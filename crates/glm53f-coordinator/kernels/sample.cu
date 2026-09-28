// Served sampling (mimo26f-afd perf reset V3): DS41RT v15's target-sampling contract on the
// coordinator GPU. Copied from mimo26f-afd v1.2.0 `crates/mimo26-coordinator/kernels/sample.cu`;
// only the entry point's name changed (PROVENANCE.md).
//
// One CTA of 1024 threads per sampled logit row; the row's greedy id (from `glm53f_coord_argmax_rows`) is
// overwritten with the draw. Filters in vLLM's order: temperature, min_p, top_k, top_p, then an
// exact categorical draw (`src/sampling.rs` has the contract and the CPU reference):
//   s_i = logit_i * (1 / T), m = max s_i;
//   min_p keeps s_i >= m + ln(min_p) (ln taken on the host, as DS41RT);
//   top_k keeps every token at or above the k-th largest s_i (ties at the boundary kept, as vLLM);
//   top_p keeps the smallest descending prefix of what is left whose weight reaches
//     ceil(top_p * its total) (ties at the boundary kept);
//   the draw is r = floor(rnd * Z / 2^64), Z the kept weight, and the token where the cumulative
//     weight in token order first exceeds r.
// Weights are fixed point, q_i = floor(exp(s_i - m) * 2^40), so every sum is an integer and a draw
// is the same on every run and at every batch position. A token below m - 28 has q_i = 0 (under
// 2^-40 of the top token's weight) and is never drawn, so the threshold searches start there.
// Thresholds come from a 16-way search over order-preserving float keys: no atomics.

#include <cuda_runtime.h>
#include <math.h>
#include <stdint.h>

namespace {

constexpr int kThreads = 1024;
constexpr int kWarps = kThreads / 32;
constexpr int kSplits = 15;                 // thresholds per search pass
constexpr float kQScale = 1099511627776.f;  // 2^40
constexpr float kFloor = 28.f;              // exp(-28) * 2^40 < 1

// One sampled row; `sampling::DeviceRow` on the host (32 bytes).
struct Row {
  int32_t row;
  float inv_t;
  float top_p;     // >= 1: off
  float ln_min_p;  // -inf: off
  int32_t top_k;   // 0: off
  int32_t pad;
  uint64_t rnd;
};
static_assert(sizeof(Row) == 32, "Row layout");

// Order-preserving key: a larger float has a larger key; -0 == +0; NaN is 0 (never kept).
__device__ __forceinline__ uint32_t okey(float s) {
  if (s != s) return 0u;
  const uint32_t b = __float_as_uint(s == 0.f ? 0.f : s);
  return (b & 0x80000000u) ? ~b : (b | 0x80000000u);
}

__device__ __forceinline__ unsigned long long weight(float s, float m) {
  return __float2ull_rz(expf(s - m) * kQScale);
}

__device__ float block_max(float v, float* sh) {
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
  __syncthreads();
  if (lane == 0) sh[warp] = v;
  __syncthreads();
  v = sh[lane];
  for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
  return v;
}

__device__ unsigned long long block_sum(unsigned long long v, unsigned long long* sh) {
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
  __syncthreads();
  if (lane == 0) sh[warp] = v;
  __syncthreads();
  v = sh[lane];
  for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
  return v;
}

// The largest key t in [lo, hi) with W(t) >= target, where W(t) sums the weight (1, or q_i when
// `kMass`) of every token with key >= t. Requires W(lo) >= target > W(hi).
template <bool kMass>
__device__ uint32_t search(const float* row, int vocab, float inv_t, float m, uint32_t lo, uint32_t hi,
                           unsigned long long target, unsigned long long (*sh)[kSplits + 1],
                           unsigned long long* tot, uint32_t* ts) {
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  while (hi - lo > 1) {
    __syncthreads();
    if (threadIdx.x < kSplits) {
      const uint64_t w = hi - lo;
      ts[threadIdx.x] = lo + (uint32_t)((w * (uint64_t)(threadIdx.x + 1) + kSplits) / (kSplits + 1));
    }
    __syncthreads();
    uint32_t t[kSplits];
#pragma unroll
    for (int j = 0; j < kSplits; ++j) t[j] = ts[j];
    unsigned long long acc[kSplits + 1] = {};
    for (int i = threadIdx.x; i < vocab; i += kThreads) {
      const float s = row[i] * inv_t;
      const uint32_t k = okey(s);
      if (k < lo) continue;
      const unsigned long long w = kMass ? weight(s, m) : 1ull;
      if (k >= hi) {
        acc[kSplits] += w;
        continue;
      }
#pragma unroll
      for (int j = 0; j < kSplits; ++j) acc[j] += k >= t[j] ? w : 0ull;
    }
#pragma unroll
    for (int j = 0; j <= kSplits; ++j)
      for (int o = 16; o > 0; o >>= 1) acc[j] += __shfl_xor_sync(0xffffffffu, acc[j], o);
    if (lane == 0)
      for (int j = 0; j <= kSplits; ++j) sh[warp][j] = acc[j];
    __syncthreads();
    if (threadIdx.x <= kSplits) {
      unsigned long long v = 0;
      for (int w = 0; w < kWarps; ++w) v += sh[w][threadIdx.x];
      tot[threadIdx.x] = v;
    }
    __syncthreads();
    // W(t_j) = tot[j] + tot[kSplits] is non-increasing in j: lo moves to the last split that still
    // reaches the target, hi to the first that does not.
    uint32_t nlo = lo, nhi = hi;
    for (int j = 0; j < kSplits; ++j) {
      if (tot[j] + tot[kSplits] >= target) {
        nlo = t[j] > nlo ? t[j] : nlo;
      } else {
        nhi = t[j] < nhi ? t[j] : nhi;
        break;
      }
    }
    lo = nlo;
    hi = nhi;
  }
  return lo;
}

__global__ void __launch_bounds__(kThreads) sample_kernel(const float* __restrict__ x, int64_t ld, int vocab,
                                                         const Row* __restrict__ rows, int* __restrict__ out) {
  __shared__ float shf[kWarps];
  __shared__ unsigned long long shu[kWarps];
  __shared__ unsigned long long shv[kWarps][kSplits + 1];
  __shared__ unsigned long long tot[kSplits + 1];
  __shared__ uint32_t ts[kSplits];
  const Row p = rows[blockIdx.x];
  const float* row = x + (int64_t)p.row * ld;
  const float inv_t = p.inv_t;

  float m = -INFINITY;
  for (int i = threadIdx.x; i < vocab; i += kThreads) m = fmaxf(m, row[i] * inv_t);
  m = block_max(m, shf);
  if (!(m > -INFINITY && m < INFINITY)) return;  // a non-finite row keeps its argmax
  const float smin = p.ln_min_p > -INFINITY ? m + p.ln_min_p : -INFINITY;
  const uint32_t kfloor = okey(fmaxf(smin, m - kFloor));
  const uint32_t hi = okey(m) + 1u;

  uint32_t kk = kfloor;
  if (p.top_k > 0) {
    unsigned long long n = 0;
    for (int i = threadIdx.x; i < vocab; i += kThreads) n += okey(row[i] * inv_t) >= kfloor;
    n = block_sum(n, shu);
    if (n > (unsigned long long)p.top_k)
      kk = search<false>(row, vocab, inv_t, m, kfloor, hi, (unsigned long long)p.top_k, shv, tot, ts);
  }
  uint32_t kp = kk;
  if (p.top_p < 1.f) {
    unsigned long long q = 0;
    for (int i = threadIdx.x; i < vocab; i += kThreads) {
      const float s = row[i] * inv_t;
      if (okey(s) >= kk) q += weight(s, m);
    }
    q = block_sum(q, shu);
    unsigned long long target = (unsigned long long)ceil((double)p.top_p * (double)q);
    target = target < 1ull ? 1ull : (target > q ? q : target);
    kp = search<true>(row, vocab, inv_t, m, kk, hi, target, shv, tot, ts);
  }

  // The draw: each thread owns a contiguous run of token ids; an exclusive scan of the runs' weights
  // finds the run holding r, and that thread walks it in token order.
  const int per = (vocab + kThreads - 1) / kThreads;
  const int b = threadIdx.x * per;
  const int e = min(b + per, vocab);
  unsigned long long run = 0;
  for (int i = b; i < e; ++i) {
    const float s = row[i] * inv_t;
    if (okey(s) >= kp) run += weight(s, m);
  }
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  unsigned long long incl = run;
  for (int o = 1; o < 32; o <<= 1) {
    const unsigned long long y = __shfl_up_sync(0xffffffffu, incl, o);
    if (lane >= o) incl += y;
  }
  __syncthreads();
  if (lane == 31) shu[warp] = incl;
  __syncthreads();
  if (warp == 0) {
    unsigned long long v = shu[lane];
    for (int o = 1; o < 32; o <<= 1) {
      const unsigned long long y = __shfl_up_sync(0xffffffffu, v, o);
      if (lane >= o) v += y;
    }
    shu[lane] = v;
  }
  __syncthreads();
  const unsigned long long before = (warp ? shu[warp - 1] : 0ull) + incl - run;
  const unsigned long long z = shu[kWarps - 1];
  const unsigned long long r = __umul64hi(p.rnd, z);
  if (run > 0 && r >= before && r - before < run) {
    unsigned long long acc = before;
    for (int i = b; i < e; ++i) {
      const float s = row[i] * inv_t;
      if (okey(s) < kp) continue;
      acc += weight(s, m);
      if (r < acc) {
        out[p.row] = i;
        break;
      }
    }
  }
}

}  // namespace

// Overwrite `out[rows[i].row]` with row i's draw over ids [0, vocab) of `x` (row stride `ld`).
extern "C" cudaError_t glm53f_coord_sample_rows(const float* x, int64_t ld, int vocab, const void* rows, int n, int* out,
                                       cudaStream_t s) {
  if (n <= 0) return cudaSuccess;
  sample_kernel<<<n, kThreads, 0, s>>>(x, ld, vocab, static_cast<const Row*>(rows), out);
  return cudaGetLastError();
}
