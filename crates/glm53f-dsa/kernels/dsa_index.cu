// DSA indexer kernels for GLM-5.3-Flash: pooled-key write, tail commit, and the
// score + top-512 pool selection over up to 256K pools (1M tokens).
//
// Selection (see crates/glm53f-dsa/README.md). glm53f_dsa_index_select runs one
// kernel, grid (chunks, rows): a block owns one row and a contiguous chunk of
// pools. Warps stream 8-pool tiles: 32 B of FP8 codes per lane, f16
// tensor-core products against the row's 32 x 128 query (held in registers),
// ReLU and head weights in f32, then a key (ordered score << 32 | ~pool). Keys
// above the block's running threshold go to a 2,048-entry shared buffer; when
// it fills, a radix select keeps the best 512. Blocks then merge through a
// fan-in-8 tree of arrival counters, and the block that completes a row's root
// writes the kept pools ascending, the expanded tokens and the tail.
// glm53f_dsa_index_select_v1 is the first implementation (a scoring kernel,
// merge kernels, a finalize kernel), kept for comparison.
// No score matrix is materialized; the workspace is about rows x chunks x 4 KiB.
// Ties (equal scores) go to the lower pool, exactly as the CPU reference.
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <stdint.h>

#include "glm53f_dsa.h"
#include "glm53f_dsa_common.cuh"

namespace glm53f {
namespace {

constexpr int kHeads = 32;
constexpr int kDim = 128;
constexpr int kTileThreads = 256;
constexpr int kTileWarps = kTileThreads / 32;
constexpr int kRoundTiles = 4;  // 8-pool tiles per warp between capacity checks
constexpr int kRoundPools = kTileWarps * kRoundTiles * 8;
constexpr int kCap = 2048;      // candidate buffer (keys)
constexpr int kMergeFanIn = 8;
constexpr int kMergeThreads = 256;

__device__ __forceinline__ float relu(float x) { return x > 0.f ? x : 0.f; }

__device__ __forceinline__ uint32_t fp8x2_to_f16x2(uint32_t two) {
  __half2_raw h = __nv_cvt_fp8x2_to_halfraw2(static_cast<__nv_fp8x2_storage_t>(two & 0xFFFFu), __NV_E4M3);
  return uint32_t(h.x) | (uint32_t(h.y) << 16);
}

// D = A (16x16 f16, row) * B (16x8 f16, col) + D, f32 accumulate.
__device__ __forceinline__ void mma_f16(float (&d)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

// ---------------------------------------------------------------------------
// Block-wide radix select over unique 64-bit keys in shared memory.
// Returns theta such that exactly k of keys[0..n) are >= theta (requires k <= n).
struct SelectScratch {
  uint32_t hist[256];
  int digit;
  int remaining;
  int done;
  int write;
};

__device__ uint64_t block_kth_key(const uint64_t* keys, int n, int k, SelectScratch& s) {
  uint64_t prefix = 0;
  int remaining = k;
  for (int shift = 56; shift >= 0; shift -= 8) {
    for (int i = threadIdx.x; i < 256; i += blockDim.x) s.hist[i] = 0;
    __syncthreads();
    const uint64_t hi = shift == 56 ? 0ull : (~0ull << (shift + 8));
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
      const uint64_t key = keys[i];
      if ((key & hi) == prefix) atomicAdd(&s.hist[(key >> shift) & 255u], 1u);
    }
    __syncthreads();
    if (threadIdx.x < 32) {
      const int lane = threadIdx.x;
      uint32_t c[8];
      uint32_t local = 0;
#pragma unroll
      for (int j = 0; j < 8; ++j) {
        c[j] = s.hist[255 - 8 * lane - j];
        local += c[j];
      }
      uint32_t incl = local;
#pragma unroll
      for (int off = 1; off < 32; off <<= 1) {
        const uint32_t v = __shfl_up_sync(0xffffffffu, incl, off);
        if (lane >= off) incl += v;
      }
      const uint32_t above = incl - local;
      if (above < uint32_t(remaining) && uint32_t(remaining) <= above + local) {
        uint32_t acc = above;
#pragma unroll
        for (int j = 0; j < 8; ++j) {
          if (acc + c[j] >= uint32_t(remaining)) {
            s.digit = 255 - 8 * lane - j;
            s.remaining = remaining - int(acc);
            s.done = c[j] == uint32_t(remaining) - acc;
            break;
          }
          acc += c[j];
        }
      }
    }
    __syncthreads();
    prefix |= uint64_t(s.digit) << shift;
    remaining = s.remaining;
    const int done = s.done;
    __syncthreads();
    if (done) break;
  }
  return prefix;
}

// Keep the best `k` of keys[0..n) at keys[0..k) (order unspecified); returns theta.
__device__ uint64_t block_keep_top(uint64_t* keys, int n, int k, uint64_t* stage, SelectScratch& s) {
  const uint64_t theta = block_kth_key(keys, n, k, s);
  if (threadIdx.x == 0) s.write = 0;
  __syncthreads();
  for (int i = threadIdx.x; i < n; i += blockDim.x) {
    const uint64_t key = keys[i];
    if (key >= theta) stage[atomicAdd(&s.write, 1)] = key;
  }
  __syncthreads();
  for (int i = threadIdx.x; i < k; i += blockDim.x) keys[i] = stage[i];
  __syncthreads();
  return theta;
}

// ---------------------------------------------------------------------------
// Pooled-key write.

__device__ float block_sum_128(float v, float* red) {
  // 128 threads = 4 warps.
  v = warp_sum(v);
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  __syncthreads();
  if (lane == 0) red[warp] = v;
  __syncthreads();
  return ((red[0] + red[1]) + red[2]) + red[3];
}

__device__ float block_max_128(float v, float* red) {
  v = warp_max(v);
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  __syncthreads();
  if (lane == 0) red[warp] = v;
  __syncthreads();
  return fmaxf(fmaxf(red[0], red[1]), fmaxf(red[2], red[3]));
}

// LayerNorm of one 128-wide row (thread = channel), rounded to BF16.
__device__ float layer_norm_bf16(const float* x_row, const float* w, const float* b, float eps, float* red) {
  const int c = threadIdx.x;
  const float x = x_row[c];
  const float mean = block_sum_128(x, red) / float(kDim);
  const float d = x - mean;
  const float var = block_sum_128(d * d, red) / float(kDim);
  const float inv = 1.0f / sqrtf(var + eps);
  return bf16_round(d * inv * w[c] + b[c]);
}

__device__ __forceinline__ float tail_value(const uint8_t* tail, int token, int offset_elems) {
  const uint16_t bits = *reinterpret_cast<const uint16_t*>(tail + 16 + token * kTailTokenBytes + 2 * offset_elems);
  return __uint_as_float(uint32_t(bits) << 16);
}

__global__ __launch_bounds__(128) void pool_write_kernel(const float* k_raw, const float* gate,
    const float* ln_w, const float* ln_b, float eps, const float* ape, const uint8_t* tails,
    const glm53f_dsa_window_t* windows, const int32_t* row_req, glm53f_dsa_cache_t cache) {
  __shared__ float red[4];
  const int row = blockIdx.x, c = threadIdx.x;
  const int req = row_req[row];
  const glm53f_dsa_window_t win = windows[req];
  const int pos = win.start + (row - win.first_row);
  if ((pos + 1) % kPool != 0) return;
  const int pool = (pos + 1) / kPool - 1;
  const int tail_count = win.start % kPool;
  float kv[kPool], gv[kPool];
#pragma unroll
  for (int j = 0; j < kPool; ++j) {
    const int t = pos - (kPool - 1) + j;
    if (t >= win.start) {
      const int src = win.first_row + (t - win.start);
      kv[j] = layer_norm_bf16(k_raw + int64_t(src) * kDim, ln_w, ln_b, eps, red);
      gv[j] = bf16_round(gate[int64_t(src) * kDim + c]);
    } else {
      const uint8_t* tail = tails + int64_t(req) * kTailBytes;
      const int ti = t - (win.start - tail_count);
      kv[j] = tail_value(tail, ti, c);
      gv[j] = tail_value(tail, ti, kDim + c);
    }
  }
  // Per-channel softmax over the pool's tokens of gate + ape, applied to the keys.
  float l[kPool];
  float m = -INFINITY;
#pragma unroll
  for (int j = 0; j < kPool; ++j) {
    l[j] = gv[j] + ape[j * kDim + c];
    m = fmaxf(m, l[j]);
  }
  float sum = 0.f;
#pragma unroll
  for (int j = 0; j < kPool; ++j) {
    l[j] = expf(l[j] - m);
    sum += l[j];
  }
  float key = 0.f;
#pragma unroll
  for (int j = 0; j < kPool; ++j) key += (l[j] / sum) * kv[j];
  // FP8 with a power-of-two scale.
  const float amax = block_max_128(fabsf(key), red);
  float scale = 1.f, inv = 1.f;
  pow2_scale(amax, scale, inv);
  uint8_t* page = page_base(cache, req, pool / kPagePools);
  if (!page) return;
  page[kPoolCodesOffset + (pool % kPagePools) * kDim + c] = fp8_e4m3(key * inv);
  if (c == 0) *reinterpret_cast<float*>(page + kPoolScalesOffset + (pool % kPagePools) * 4) = scale;
}

__global__ __launch_bounds__(128) void tail_commit_kernel(const float* k_raw, const float* gate,
    const float* ln_w, const float* ln_b, float eps, uint8_t* tails, const glm53f_dsa_window_t* windows) {
  __shared__ float red[4];
  const int req = blockIdx.x, c = threadIdx.x;
  const glm53f_dsa_window_t win = windows[req];
  const int new_len = win.start + win.accepted;
  const int count = new_len % kPool;
  const int old_count = win.start % kPool;
  uint8_t* tail = tails + int64_t(req) * kTailBytes;
  float kv[kPool - 1], gv[kPool - 1];
  for (int i = 0; i < count; ++i) {
    const int t = new_len - count + i;
    if (t >= win.start) {
      const int src = win.first_row + (t - win.start);
      kv[i] = layer_norm_bf16(k_raw + int64_t(src) * kDim, ln_w, ln_b, eps, red);
      gv[i] = bf16_round(gate[int64_t(src) * kDim + c]);
    } else {
      const int ti = t - (win.start - old_count);
      kv[i] = tail_value(tail, ti, c);
      gv[i] = tail_value(tail, ti, kDim + c);
    }
  }
  __syncthreads();  // every read of the old tail precedes the rewrite
  for (int i = 0; i < count; ++i) {
    uint16_t* dst = reinterpret_cast<uint16_t*>(tail + 16 + i * kTailTokenBytes);
    dst[c] = uint16_t(__float_as_uint(kv[i]) >> 16);
    dst[kDim + c] = uint16_t(__float_as_uint(gv[i]) >> 16);
  }
  if (c < 4) reinterpret_cast<uint32_t*>(tail)[c] = c == 0 ? uint32_t(count) : 0u;
}

// ---------------------------------------------------------------------------
// Score + per-chunk top-512.

struct TileShared {
  __half q[kHeads * kDim];  // 8 KiB, natural channel order
  float w[kHeads];
  uint64_t buf[kCap];       // 16 KiB
  uint64_t stage[kTopPools];  // 4 KiB
  SelectScratch sel;
  int count;
  uint64_t threshold;
};

__global__ __launch_bounds__(kTileThreads, 2) void index_tiles_kernel(const float* q, const float* w,
    float score_scale, const int32_t* row_pos, const int32_t* row_req, glm53f_dsa_cache_t cache,
    int chunk_pools, int chunks, uint64_t* lists, int32_t* counts, float* debug_scores, int debug_stride) {
  __shared__ TileShared sh;
  const int chunk = blockIdx.x, row = blockIdx.y;
  const int pos = row_pos[row], req = row_req[row];
  const int n = (pos + 1) / kPool;
  const int begin = chunk * chunk_pools;
  const int end = min(n, begin + chunk_pools);
  if (n <= kTopPools || begin >= end) {
    if (threadIdx.x == 0) counts[int64_t(row) * chunks + chunk] = 0;
    return;
  }
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int g = lane >> 2, t = lane & 3;

  // Query to f16 with a per-head power-of-two scale (exact), folded into the head weight.
  for (int h = warp; h < kHeads; h += kTileWarps) {
    const float* qh = q + (int64_t(row) * kHeads + h) * kDim;
    float v[4];
    float amax = 0.f;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      v[i] = qh[lane + 32 * i];
      amax = fmaxf(amax, fabsf(v[i]));
    }
    amax = warp_max(amax);
    int e = 0;
    if (amax > 0.f && isfinite(amax)) {
      int E;
      frexpf(amax, &E);
      e = max(-120, min(120, 15 - E));
    }
    const float up = exp2_int(e);
#pragma unroll
    for (int i = 0; i < 4; ++i) sh.q[h * kDim + lane + 32 * i] = __float2half_rn(v[i] * up);
    if (lane == 0) sh.w[h] = w[int64_t(row) * kHeads + h] * exp2_int(-e) * score_scale;
  }
  if (threadIdx.x == 0) {
    sh.count = 0;
    sh.threshold = 0;
  }
  __syncthreads();

  // A fragments (32 heads x 128 channels) in registers. Channel permutation:
  // logical k (tile kt, thread t, half r, element e) <-> physical 32 t + 4 kt + 2 r + e,
  // so a lane's B operand for all 8 k-tiles is bytes [32 t, 32 t + 32) of a pooled key.
  uint32_t a[2][8][4];
#pragma unroll
  for (int mt = 0; mt < 2; ++mt) {
#pragma unroll
    for (int kt = 0; kt < 8; ++kt) {
      const int col = 32 * t + 4 * kt;
      const int h0 = mt * 16 + g, h1 = h0 + 8;
      a[mt][kt][0] = *reinterpret_cast<const uint32_t*>(&sh.q[h0 * kDim + col]);
      a[mt][kt][1] = *reinterpret_cast<const uint32_t*>(&sh.q[h1 * kDim + col]);
      a[mt][kt][2] = *reinterpret_cast<const uint32_t*>(&sh.q[h0 * kDim + col + 2]);
      a[mt][kt][3] = *reinterpret_cast<const uint32_t*>(&sh.q[h1 * kDim + col + 2]);
    }
  }
  const float w0 = sh.w[g], w1 = sh.w[g + 8], w2 = sh.w[16 + g], w3 = sh.w[16 + g + 8];

  const int n_tiles = (end - begin + 7) / 8;
  const unsigned lt_mask = (1u << lane) - 1u;
  // Warp w takes tiles w, w + 8, w + 16, ...; the next tile's codes and scales
  // are loaded before the current tile's products (software pipelining). Every
  // kRoundTiles steps the block checks the candidate buffer's capacity.
  const int steps = (n_tiles + kTileWarps - 1) / kTileWarps;
  auto load_tile = [&](int tile, uint4& lo, uint4& hi, float& sa, float& sb) {
    lo = make_uint4(0, 0, 0, 0);
    hi = lo;
    sa = sb = __int_as_float(0x7fc00000);
    if (tile >= n_tiles) return;
    const int p0 = begin + tile * 8;
    const int pb = p0 + g;
    if (pb < end) {
      const uint8_t* page = page_base(cache, req, pb / kPagePools);
      if (page) {
        const uint4* src = reinterpret_cast<const uint4*>(page + kPoolCodesOffset + (pb % kPagePools) * kDim + 32 * t);
        lo = __ldg(src);
        hi = __ldg(src + 1);
      }
    }
    const int pa = p0 + 2 * t;
    if (pa < end) sa = pool_scale(cache, req, pa);
    if (pa + 1 < end) sb = pool_scale(cache, req, pa + 1);
  };
  uint4 lo, hi;
  float sa, sb;
  load_tile(warp, lo, hi, sa, sb);
  uint64_t threshold = 0;
  for (int step = 0; step < steps; ++step) {
    if (step % kRoundTiles == 0) {
      __syncthreads();
      if (sh.count > kCap - kRoundPools) {
        const uint64_t theta = block_keep_top(sh.buf, sh.count, kTopPools, sh.stage, sh.sel);
        if (threadIdx.x == 0) {
          sh.count = kTopPools;
          sh.threshold = theta;
        }
        __syncthreads();
      }
      threshold = sh.threshold;
    }
    const int tile = warp + step * kTileWarps;
    uint4 nlo, nhi;
    float nsa, nsb;
    load_tile(tile + kTileWarps, nlo, nhi, nsa, nsb);
    if (tile < n_tiles) {
      const int p0 = begin + tile * 8;
      const uint32_t words[8] = {lo.x, lo.y, lo.z, lo.w, hi.x, hi.y, hi.z, hi.w};
      float c0[4] = {0.f, 0.f, 0.f, 0.f}, c1[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
      for (int kt = 0; kt < 8; ++kt) {
        const uint32_t b0 = fp8x2_to_f16x2(words[kt]);
        const uint32_t b1 = fp8x2_to_f16x2(words[kt] >> 16);
        mma_f16(c0, a[0][kt], b0, b1);
        mma_f16(c1, a[1][kt], b0, b1);
      }
      // c*[0]: (head g, pool 2t), [1]: (head g, pool 2t+1), [2]/[3]: head g+8.
      float s0 = ((w0 * relu(c0[0]) + w1 * relu(c0[2])) + w2 * relu(c1[0])) + w3 * relu(c1[2]);
      float s1 = ((w0 * relu(c0[1]) + w1 * relu(c0[3])) + w2 * relu(c1[1])) + w3 * relu(c1[3]);
#pragma unroll
      for (int off = 4; off < 32; off <<= 1) {
        s0 += __shfl_xor_sync(0xffffffffu, s0, off);
        s1 += __shfl_xor_sync(0xffffffffu, s1, off);
      }
      uint64_t k0 = 0, k1 = 0;
      if (g == 0) {
        const int pa = p0 + 2 * t;
        if (pa < end) {
          const float s = s0 * sa;
          k0 = score_key(s, uint32_t(pa));
          if (debug_scores) debug_scores[int64_t(row) * debug_stride + pa] = s;
        }
        if (pa + 1 < end) {
          const float s = s1 * sb;
          k1 = score_key(s, uint32_t(pa + 1));
          if (debug_scores) debug_scores[int64_t(row) * debug_stride + pa + 1] = s;
        }
      }
      const bool take0 = k0 > threshold, take1 = k1 > threshold;
      const unsigned m0 = __ballot_sync(0xffffffffu, take0), m1 = __ballot_sync(0xffffffffu, take1);
      const int total = __popc(m0) + __popc(m1);
      if (total) {
        int at = 0;
        if (lane == 0) at = atomicAdd(&sh.count, total);
        at = __shfl_sync(0xffffffffu, at, 0);
        if (take0) sh.buf[at + __popc(m0 & lt_mask)] = k0;
        if (take1) sh.buf[at + __popc(m0) + __popc(m1 & lt_mask)] = k1;
      }
    }
    lo = nlo;
    hi = nhi;
    sa = nsa;
    sb = nsb;
  }
  __syncthreads();
  int kept = sh.count;
  if (kept > kTopPools) {
    block_keep_top(sh.buf, kept, kTopPools, sh.stage, sh.sel);
    kept = kTopPools;
  }
  uint64_t* out = lists + (int64_t(row) * chunks + chunk) * kTopPools;
  for (int i = threadIdx.x; i < kept; i += kTileThreads) out[i] = sh.buf[i];
  if (threadIdx.x == 0) counts[int64_t(row) * chunks + chunk] = kept;
}

// Merge groups of up to 8 lists into one list of at most 512 keys.
__global__ __launch_bounds__(kMergeThreads) void index_merge_kernel(const uint64_t* in, const int32_t* in_counts,
    int in_lists, uint64_t* out, int32_t* out_counts, int out_lists) {
  __shared__ uint64_t keys[kMergeFanIn * kTopPools];  // 32 KiB
  __shared__ uint64_t stage[kTopPools];
  __shared__ SelectScratch sel;
  __shared__ int offsets[kMergeFanIn + 1];
  const int group = blockIdx.x, row = blockIdx.y;
  const int first = group * kMergeFanIn;
  const int lists = min(kMergeFanIn, in_lists - first);
  if (threadIdx.x == 0) {
    int o = 0;
    for (int i = 0; i < kMergeFanIn; ++i) {
      offsets[i] = o;
      if (i < lists) o += in_counts[int64_t(row) * in_lists + first + i];
    }
    offsets[kMergeFanIn] = o;
  }
  __syncthreads();
  for (int i = 0; i < lists; ++i) {
    const uint64_t* src = in + (int64_t(row) * in_lists + first + i) * kTopPools;
    const int cnt = offsets[i + 1] - offsets[i];
    for (int j = threadIdx.x; j < cnt; j += kMergeThreads) keys[offsets[i] + j] = src[j];
  }
  __syncthreads();
  int n = offsets[kMergeFanIn];
  if (n > kTopPools) {
    block_keep_top(keys, n, kTopPools, stage, sel);
    n = kTopPools;
  }
  uint64_t* dst = out + (int64_t(row) * out_lists + group) * kTopPools;
  for (int j = threadIdx.x; j < n; j += kMergeThreads) dst[j] = keys[j];
  if (threadIdx.x == 0) out_counts[int64_t(row) * out_lists + group] = n;
}

// Kept pools ascending, expanded tokens and the tail.
__global__ __launch_bounds__(kTopPools) void index_finalize_kernel(const int32_t* row_pos, const uint64_t* list,
    const int32_t* list_counts, int32_t* pools_out, int32_t* tokens_out, int32_t* counts_out) {
  __shared__ uint32_t pools[kTopPools];
  const int row = blockIdx.x, i = threadIdx.x;
  const int pos = row_pos[row];
  const int n = (pos + 1) / kPool;
  int kept;
  if (n <= kTopPools) {
    kept = n;
    pools[i] = i < n ? uint32_t(i) : 0xFFFFFFFFu;
  } else {
    kept = list_counts[row];
    pools[i] = i < kept ? ~uint32_t(list[int64_t(row) * kTopPools + i]) : 0xFFFFFFFFu;
  }
  __syncthreads();
  // Bitonic sort of 512 u32 ascending (unused slots are 0xFFFFFFFF and sort last).
  for (int size = 2; size <= kTopPools; size <<= 1) {
    for (int stride = size >> 1; stride > 0; stride >>= 1) {
      const int j = i ^ stride;
      if (j > i) {
        const bool up = (i & size) == 0;
        const uint32_t x = pools[i], y = pools[j];
        if ((x > y) == up) {
          pools[i] = y;
          pools[j] = x;
        }
      }
      __syncthreads();
    }
  }
  pools_out[int64_t(row) * kTopPools + i] = i < kept ? int32_t(pools[i]) : -1;
  const int tail_start = n * kPool;
  const int tail_len = pos + 1 - tail_start;
  const int total = kept * kPool + tail_len;
  for (int j = i; j < kMaxTokens; j += kTopPools) {
    int32_t tok = -1;
    if (j < kept * kPool) tok = int32_t(pools[j / kPool]) * kPool + (j % kPool);
    else if (j < total) tok = tail_start + (j - kept * kPool);
    tokens_out[int64_t(row) * kMaxTokens + j] = tok;
  }
  if (i == 0) {
    counts_out[row * 2] = kept;
    counts_out[row * 2 + 1] = total;
  }
}

// ---------------------------------------------------------------------------
// v2: score, select and merge in one kernel (the default since the
// decode-latency pass).
//
// Scoring is index_tiles' with the chunk's page ids resolved once into shared
// memory (each tile costs one global load, not a page-table load followed by a
// dependent code load), loads issued two tiles ahead, and the f16 query laid
// out so its fragment loads are free of bank conflicts. Each block then
// publishes its kept keys and arrives at its parent node of a fan-in-8 tree
// (one arrival counter per node). The last block to arrive at a node gathers
// the node's (at most 8) lists in one round trip, keeps the best 512 and
// climbs to the next level; the block that completes the root writes the row's
// outputs. There are no further launches, and the merges of finished subtrees
// overlap the scoring of blocks still running.
//
// At these sizes the block-wide steps are bound by latency (barriers, shared
// memory round trips), not arithmetic (measured with per-phase clock stamps),
// so each step is arranged to need few of them: the radix select starts at
// the keys' highest differing bit (merged lists hold similar scores) with a
// double-buffered histogram, the compaction after it is one scan, and the
// ascending pool order comes from a bitmap with per-thread prefix counts.

constexpr int kFanIn = 8;
constexpr int kMaxLevels = 6;         // 8^6 leaves
constexpr int kLeafCap = 2048;        // candidate buffer while scoring
constexpr int kMaxChunkPages = 1728;  // page ids cached per chunk (chunks up to 27,648 pools)
// The f16 query in shared memory: head rows of kQStride halves; key channels
// 32 t .. 32 t + 31 (lane group t's eight k-steps) start at 32 t + 16 (t / 2).
// With this padding a half-warp's 8-byte fragment loads hit 32 distinct banks
// (unpadded rows put 16 lanes on 2 banks).
constexpr int kQStride = 148;
__device__ __forceinline__ int q_slot(int c) { return c + 16 * (c >> 6); }

// Dynamic shared memory (bytes). [0, 32 KiB) holds, in turn: the candidate
// buffer with the f16 query and the chunk's page ids (scoring), up to 8 x 512
// gathered keys (merging), and the pool bitmap (finalize).
constexpr int kSmIn = 0;
constexpr int kSmQ = 16384;
constexpr int kSmPages = kSmQ + kHeads * kQStride * 2;  // 25,856
constexpr int kSmPools = 32768;   // 512 x u32: the row's pools, ascending
constexpr int kSmStage = 34816;   // 4 KiB: 512 keys (keep's compaction), then the finalize's prefix counts
constexpr int kSmScratch = 38912;
static_assert(kSmPages + kMaxChunkPages * 4 <= kSmPools, "page ids fit below the pool list");

// Radix-select state. The histograms are double-buffered: pass p counts into
// hist[p] while the other is cleared for the next pass.
constexpr int kRadixBins = 256;
static_assert(kRadixBins <= kTileThreads, "one bin per thread when clearing");
struct RadixScratch {
  uint32_t hist[2][kRadixBins];
  int digit[2];
  int remaining[2];
  int done[2];
};

struct FusedScratch {
  RadixScratch sel;
  float w[kHeads];
  uint64_t red[kTileWarps];
  int count;
  int flag;
  int scan[kTileWarps + 1];
  uint64_t threshold;
};
constexpr int kFusedSmem = kSmScratch + int(sizeof(FusedScratch));
static_assert(kFusedSmem <= 48 * 1024, "within the default dynamic shared-memory limit (no opt-in needed)");

struct FusedLayout {
  int levels;                      // merge levels; nodes[levels] == 1
  int nodes[kMaxLevels + 1];       // nodes per row at each level; nodes[0] = chunks
  uint64_t lists[kMaxLevels];      // byte offset: [rows][nodes[l]][512] keys
  uint64_t counts[kMaxLevels];     // byte offset: [rows][nodes[l]] int32
  uint64_t counters[kMaxLevels];   // byte offset: [rows][nodes[l + 1]] uint32 arrivals
  uint64_t counter_begin, counter_bytes, total;
};

FusedLayout fused_layout(int rows, int chunks) {
  FusedLayout L{};
  L.nodes[0] = chunks;
  while (L.nodes[L.levels] > 1 && L.levels < kMaxLevels) {
    L.nodes[L.levels + 1] = (L.nodes[L.levels] + kFanIn - 1) / kFanIn;
    ++L.levels;
  }
  uint64_t off = 0;
  for (int l = 0; l < L.levels; ++l) {
    L.lists[l] = off;
    off += uint64_t(rows) * L.nodes[l] * kTopPools * 8;
    L.counts[l] = off;
    off += align_up(uint64_t(rows) * L.nodes[l] * 4, 256);
  }
  L.counter_begin = off;
  for (int l = 0; l < L.levels; ++l) {
    L.counters[l] = off;
    off += uint64_t(rows) * L.nodes[l + 1] * 4;
  }
  L.counter_bytes = align_up(off - L.counter_begin, 256);
  L.total = L.counter_begin + L.counter_bytes;
  return L;
}

// Returns theta such that exactly k of the block's unique keys are >= theta (k
// <= their number): radix select, 8 bits a pass, starting at the highest bit
// where the keys differ (the highest set bit of `d`; `ref` is any of the keys),
// which skips the digits every key shares (merged lists hold similar scores).
// While the digit lies in the keys' upper word (the score) a pass works on
// 32-bit values. `for_each_key(f)` calls f(key) on each of the calling
// thread's keys (kTileThreads threads).
// Precondition: s.hist[0] is zero and a barrier has passed since.
template <class ForEachKey>
__device__ uint64_t radix_kth(uint64_t ref, uint64_t d, int k, RadixScratch& s, ForEachKey&& for_each_key) {
  if (d == 0) return ref;
  const int tid = threadIdx.x, lane = tid & 31;
  int top = 63 - __clzll(d);
  uint64_t prefix = top == 63 ? 0ull : (ref & (~0ull << (top + 1)));
  int remaining = k;
  for (int p = 0;; p ^= 1) {
    uint32_t* h = s.hist[p];
    if (tid < kRadixBins) s.hist[p ^ 1][tid] = 0u;
    const int shift = top >= 7 ? top - 7 : 0;
    const uint32_t dmask = (2u << (top - shift)) - 1u;
    const uint64_t hm = top == 63 ? 0ull : (~0ull << (top + 1));
    if (shift >= 32) {
      const uint32_t pre = uint32_t(prefix >> 32), m = uint32_t(hm >> 32);
      const int sh = shift - 32;
      for_each_key([&](uint64_t v) {
        const uint32_t x = uint32_t(v >> 32);
        if ((x & m) == pre) atomicAdd(&h[(x >> sh) & dmask], 1u);
      });
    } else {
      for_each_key([&](uint64_t v) {
        if ((v & hm) == prefix) atomicAdd(&h[uint32_t(v >> shift) & dmask], 1u);
      });
    }
    __syncthreads();
    if (tid < 32) {
      // Lane l owns bins 255 - 8 l - j (j = 0..7), read as two 16-byte vectors.
      const uint4 b_hi = *reinterpret_cast<const uint4*>(h + 252 - 8 * lane);
      const uint4 b_lo = *reinterpret_cast<const uint4*>(h + 248 - 8 * lane);
      const uint32_t c[8] = {b_hi.w, b_hi.z, b_hi.y, b_hi.x, b_lo.w, b_lo.z, b_lo.y, b_lo.x};
      uint32_t local = 0;
#pragma unroll
      for (int j = 0; j < 8; ++j) local += c[j];
      uint32_t incl = local;
#pragma unroll
      for (int off = 1; off < 32; off <<= 1) {
        const uint32_t v = __shfl_up_sync(0xffffffffu, incl, off);
        if (lane >= off) incl += v;
      }
      const uint32_t above = incl - local;
      if (above < uint32_t(remaining) && uint32_t(remaining) <= above + local) {
        uint32_t acc = above;
#pragma unroll
        for (int j = 0; j < 8; ++j) {
          if (acc + c[j] >= uint32_t(remaining)) {
            s.digit[p] = 255 - 8 * lane - j;
            s.remaining[p] = remaining - int(acc);
            s.done[p] = c[j] == uint32_t(remaining) - acc;
            break;
          }
          acc += c[j];
        }
      }
    }
    __syncthreads();
    prefix |= uint64_t(s.digit[p]) << shift;
    remaining = s.remaining[p];
    if (s.done[p] || shift == 0) return prefix;
    top = shift - 1;
  }
}

// The keys[0..n) of a shared-memory array as radix_kth's for_each_key: each
// thread reads two keys per 16-byte load and issues four loads before using
// any of them, so the latency of one does not serialize the next.
struct SmemKeys {
  const uint64_t* keys;  // 16-byte aligned
  int n;
  template <class F>
  __device__ void operator()(F&& f) const {
    constexpr int kU = 4;
    for (int base = 0; base < n; base += 2 * kU * kTileThreads) {
      uint4 v[kU];
#pragma unroll
      for (int u = 0; u < kU; ++u) {
        const int i = base + 2 * (u * kTileThreads + int(threadIdx.x));
        v[u] = i < n ? *reinterpret_cast<const uint4*>(keys + i) : make_uint4(0u, 0u, 0u, 0u);
      }
#pragma unroll
      for (int u = 0; u < kU; ++u) {
        const int i = base + 2 * (u * kTileThreads + int(threadIdx.x));
        if (i < n) f((uint64_t(v[u].y) << 32) | v[u].x);
        if (i + 1 < n) f((uint64_t(v[u].w) << 32) | v[u].z);
      }
    }
  }
};

// Clears the state radix_kth expects (a barrier must follow).
__device__ __forceinline__ void radix_reset(RadixScratch& s) {
  if (threadIdx.x < kRadixBins) s.hist[0][threadIdx.x] = 0u;
}

// radix_kth over keys[0..n) in shared memory (nonzero, unique; k <= n), with
// the highest differing bit from one pass over the keys.
__device__ uint64_t block_kth_key_fast(const uint64_t* keys, int n, int k, RadixScratch& s, uint64_t* red) {
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, nw = blockDim.x >> 5;
  const uint64_t k0 = keys[0];
  uint64_t diff = 0;
#pragma unroll 4
  for (int i = threadIdx.x; i < n; i += blockDim.x) diff |= keys[i] ^ k0;
  const uint32_t dlo = __reduce_or_sync(0xffffffffu, uint32_t(diff));
  const uint32_t dhi = __reduce_or_sync(0xffffffffu, uint32_t(diff >> 32));
  if (lane == 0) red[warp] = (uint64_t(dhi) << 32) | dlo;
  radix_reset(s);
  __syncthreads();
  uint64_t d = 0;
  for (int i = 0; i < nw; ++i) d |= red[i];
  return radix_kth(k0, d, k, s, SmemKeys{keys, n});
}

// Moves the k keys >= theta of keys[0..n) to keys[0..k) (order unspecified):
// every thread counts its keys, a block-wide scan gives each thread its output
// offset, and the keys go out through `stage` (k entries). One scan instead of
// a shared write counter per warp and round; the keys are re-read rather than
// held in registers, which the scoring loop that calls this cannot spare.
__device__ void keep_at_least(uint64_t* keys, int n, int k, uint64_t theta, uint64_t* stage, int* wsum) {
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  int mine = 0;
#pragma unroll 4
  for (int i = tid; i < n; i += kTileThreads) mine += keys[i] >= theta ? 1 : 0;
  int incl = mine;
#pragma unroll
  for (int o = 1; o < 32; o <<= 1) {
    const int t = __shfl_up_sync(0xffffffffu, incl, o);
    if (lane >= o) incl += t;
  }
  if (lane == 31) wsum[warp] = incl;
  __syncthreads();
  int at = incl - mine;
#pragma unroll
  for (int i = 0; i < kTileWarps; ++i) at += i < warp ? wsum[i] : 0;
#pragma unroll 4
  for (int i = tid; i < n; i += kTileThreads) {
    const uint64_t v = keys[i];
    if (v >= theta) stage[at++] = v;
  }
  __syncthreads();
  for (int i = tid; i < k; i += kTileThreads) keys[i] = stage[i];
  __syncthreads();
}

// Keep the best `k` of keys[0..n) at keys[0..k) (order unspecified); returns theta.
__device__ uint64_t block_keep_top_fast(uint64_t* keys, int n, int k, uint64_t* stage, RadixScratch& s,
                                        uint64_t* red, int* wsum) {
  const uint64_t theta = block_kth_key_fast(keys, n, k, s, red);
  keep_at_least(keys, n, k, theta, stage, wsum);
  return theta;
}

// Scores a chunk of pools into the candidate buffer; returns the number of keys
// kept (<= 512, unsorted, at smem + kSmIn).
__device__ int fused_score_chunk(unsigned char* smem, FusedScratch& sc, const float* q, const float* w,
                                 float score_scale, int row, int req, const glm53f_dsa_cache_t& cache, int begin,
                                 int end, float* debug_scores, int debug_stride) {
  uint64_t* buf = reinterpret_cast<uint64_t*>(smem + kSmIn);
  __half* q16 = reinterpret_cast<__half*>(smem + kSmQ);
  int32_t* pages = reinterpret_cast<int32_t*>(smem + kSmPages);
  uint64_t* stage = reinterpret_cast<uint64_t*>(smem + kSmStage);
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int g = lane >> 2, t = lane & 3;
  const int page0 = begin / kPagePools;
  const int npages = (end - 1) / kPagePools - page0 + 1;
  const bool cached = npages <= kMaxChunkPages;
  if (cached) {
    for (int i = threadIdx.x; i < npages; i += blockDim.x) {
      const int lp = page0 + i;
      int32_t phys = -1;
      if (lp < cache.max_pages) {
        phys = __ldg(cache.page_tables + int64_t(req) * cache.max_pages + lp);
        if (phys >= cache.n_pages) phys = -1;
      }
      pages[i] = phys;
    }
  }
  // Warp `warp` prepares heads warp + 8 j. All of its query and weight loads are
  // issued before the first reduction (one memory round trip, not one per head).
  constexpr int kWarpHeads = kHeads / kTileWarps;  // 4
  float v[kWarpHeads][4];
#pragma unroll
  for (int j = 0; j < kWarpHeads; ++j) {
    const float* qh = q + (int64_t(row) * kHeads + warp + kTileWarps * j) * kDim;
#pragma unroll
    for (int i = 0; i < 4; ++i) v[j][i] = __ldg(qh + lane + 32 * i);
  }
  const float wv = lane < kWarpHeads ? __ldg(w + int64_t(row) * kHeads + warp + kTileWarps * lane) : 0.f;
#pragma unroll
  for (int j = 0; j < kWarpHeads; ++j) {
    const int h = warp + kTileWarps * j;
    float amax = 0.f;
#pragma unroll
    for (int i = 0; i < 4; ++i) amax = fmaxf(amax, fabsf(v[j][i]));
    amax = warp_max(amax);
    int e = 0;
    if (amax > 0.f && isfinite(amax)) {
      int E;
      frexpf(amax, &E);
      e = max(-120, min(120, 15 - E));
    }
    const float up = exp2_int(e);
#pragma unroll
    for (int i = 0; i < 4; ++i) q16[h * kQStride + q_slot(lane + 32 * i)] = __float2half_rn(v[j][i] * up);
    const float wh = __shfl_sync(0xffffffffu, wv, j);
    if (lane == 0) sc.w[h] = wh * exp2_int(-e) * score_scale;
  }
  if (threadIdx.x == 0) {
    sc.count = 0;
    sc.threshold = 0;
  }
  __syncthreads();
  uint32_t a[2][8][4];
#pragma unroll
  for (int mt = 0; mt < 2; ++mt) {
#pragma unroll
    for (int kt = 0; kt < 8; ++kt) {
      const int col = 32 * t + 4 * kt;
      const int h0 = mt * 16 + g, h1 = h0 + 8;
      a[mt][kt][0] = *reinterpret_cast<const uint32_t*>(&q16[h0 * kQStride + q_slot(col)]);
      a[mt][kt][1] = *reinterpret_cast<const uint32_t*>(&q16[h1 * kQStride + q_slot(col)]);
      a[mt][kt][2] = *reinterpret_cast<const uint32_t*>(&q16[h0 * kQStride + q_slot(col) + 2]);
      a[mt][kt][3] = *reinterpret_cast<const uint32_t*>(&q16[h1 * kQStride + q_slot(col) + 2]);
    }
  }
  const float w0 = sc.w[g], w1 = sc.w[g + 8], w2 = sc.w[16 + g], w3 = sc.w[16 + g + 8];
  const int n_tiles = (end - begin + 7) / 8;
  const unsigned lt_mask = (1u << lane) - 1u;
  const int steps = (n_tiles + kTileWarps - 1) / kTileWarps;
  auto page_of = [&](int pool) -> const uint8_t* {
    if (cached) {
      const int32_t p = pages[pool / kPagePools - page0];
      return p < 0 ? nullptr : cache.base + int64_t(p) * cache.page_stride;
    }
    return page_base(cache, req, pool / kPagePools);
  };
  auto load_tile = [&](int tile, uint4& lo, uint4& hi, float2& s) {
    lo = make_uint4(0, 0, 0, 0);
    hi = lo;
    s = make_float2(__int_as_float(0x7fc00000), __int_as_float(0x7fc00000));
    if (tile >= n_tiles) return;
    const int p0 = begin + tile * 8;
    const int pb = p0 + g;
    if (pb < end) {
      const uint8_t* page = page_of(pb);
      if (page) {
        const uint4* src = reinterpret_cast<const uint4*>(page + kPoolCodesOffset + (pb % kPagePools) * kDim + 32 * t);
        lo = __ldg(src);
        hi = __ldg(src + 1);
      }
    }
    const int pa = p0 + 2 * t;  // even: pa and pa + 1 share a page
    if (pa < end) {
      const uint8_t* page = page_of(pa);
      if (page) {
        const float* sp = reinterpret_cast<const float*>(page + kPoolScalesOffset) + pa % kPagePools;
        if (pa + 1 < end) s = __ldg(reinterpret_cast<const float2*>(sp));
        else s.x = __ldg(sp);
      }
    }
  };
  uint4 lo, hi, lo1, hi1;
  float2 sa, sa1;
  load_tile(warp, lo, hi, sa);
  load_tile(warp + kTileWarps, lo1, hi1, sa1);
  uint64_t threshold = 0;
  for (int step = 0; step < steps; ++step) {
    if (step % kRoundTiles == 0) {
      __syncthreads();
      if (sc.count > kLeafCap - kRoundPools) {
        const uint64_t theta = block_keep_top_fast(buf, sc.count, kTopPools, stage, sc.sel, sc.red, sc.scan);
        if (threadIdx.x == 0) {
          sc.count = kTopPools;
          sc.threshold = theta;
        }
        __syncthreads();
      }
      threshold = sc.threshold;
    }
    const int tile = warp + step * kTileWarps;
    uint4 lo2, hi2;
    float2 sa2;
    load_tile(tile + 2 * kTileWarps, lo2, hi2, sa2);
    if (tile < n_tiles) {
      const int p0 = begin + tile * 8;
      const uint32_t words[8] = {lo.x, lo.y, lo.z, lo.w, hi.x, hi.y, hi.z, hi.w};
      float c0[4] = {0.f, 0.f, 0.f, 0.f}, c1[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
      for (int kt = 0; kt < 8; ++kt) {
        const uint32_t b0 = fp8x2_to_f16x2(words[kt]);
        const uint32_t b1 = fp8x2_to_f16x2(words[kt] >> 16);
        mma_f16(c0, a[0][kt], b0, b1);
        mma_f16(c1, a[1][kt], b0, b1);
      }
      float s0 = ((w0 * relu(c0[0]) + w1 * relu(c0[2])) + w2 * relu(c1[0])) + w3 * relu(c1[2]);
      float s1 = ((w0 * relu(c0[1]) + w1 * relu(c0[3])) + w2 * relu(c1[1])) + w3 * relu(c1[3]);
#pragma unroll
      for (int off = 4; off < 32; off <<= 1) {
        s0 += __shfl_xor_sync(0xffffffffu, s0, off);
        s1 += __shfl_xor_sync(0xffffffffu, s1, off);
      }
      uint64_t k0 = 0, k1 = 0;
      if (g == 0) {
        const int pa = p0 + 2 * t;
        if (pa < end) {
          const float s = s0 * sa.x;
          k0 = score_key(s, uint32_t(pa));
          if (debug_scores) debug_scores[int64_t(row) * debug_stride + pa] = s;
        }
        if (pa + 1 < end) {
          const float s = s1 * sa.y;
          k1 = score_key(s, uint32_t(pa + 1));
          if (debug_scores) debug_scores[int64_t(row) * debug_stride + pa + 1] = s;
        }
      }
      const bool take0 = k0 > threshold, take1 = k1 > threshold;
      const unsigned m0 = __ballot_sync(0xffffffffu, take0), m1 = __ballot_sync(0xffffffffu, take1);
      const int total = __popc(m0) + __popc(m1);
      if (total) {
        int at = 0;
        if (lane == 0) at = atomicAdd(&sc.count, total);
        at = __shfl_sync(0xffffffffu, at, 0);
        if (take0) buf[at + __popc(m0 & lt_mask)] = k0;
        if (take1) buf[at + __popc(m0) + __popc(m1 & lt_mask)] = k1;
      }
    }
    lo = lo1;
    hi = hi1;
    sa = sa1;
    lo1 = lo2;
    hi1 = hi2;
    sa1 = sa2;
  }
  __syncthreads();
  int kept = sc.count;
  if (kept > kTopPools) {
    block_keep_top_fast(buf, kept, kTopPools, stage, sc.sel, sc.red, sc.scan);
    kept = kTopPools;
  }
  return kept;
}

// Row outputs: every visible pool (dense rows).
__device__ void write_dense_row(int row, int pos, int n, int32_t* pools_out, int32_t* tokens_out,
                                int32_t* counts_out) {
  for (int i = threadIdx.x; i < kTopPools; i += blockDim.x) pools_out[int64_t(row) * kTopPools + i] = i < n ? i : -1;
  for (int j = threadIdx.x; j < kMaxTokens; j += blockDim.x)
    tokens_out[int64_t(row) * kMaxTokens + j] = j <= pos ? j : -1;
  if (threadIdx.x == 0) {
    counts_out[row * 2] = n;
    counts_out[row * 2 + 1] = pos + 1;
  }
}


// Row outputs from the kept keys (in shared memory, possibly aliasing `bitmap`):
// pools ascending, their tokens, then the tail. A pool's position is the number
// of kept pools below it, read from a bitmap over the visible pools: thread t
// owns bitmap words [32 t, 32 t + 32) and publishes the exclusive prefix counts
// of its four 8-word groups (`prefix`, 4 x kTileThreads ints), so a position
// costs at most 7 word popcounts after one block-wide scan.
constexpr int kBitmapWords = 32 * kTileThreads;  // 8,192: pools < 262,144
__device__ void write_sparse_row(const uint64_t* keys, int kept, uint32_t* bitmap, uint32_t* pools, int* prefix,
                                 int* wsum, int row, int pos, int n, int32_t* pools_out, int32_t* tokens_out,
                                 int32_t* counts_out) {
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int words = min((n + 31) / 32, kBitmapWords);
  const uint32_t limit = uint32_t(words) * 32u;
  uint32_t p0 = tid < kept ? ~uint32_t(keys[tid]) : 0xFFFFFFFFu;
  uint32_t p1 = tid + kTileThreads < kept ? ~uint32_t(keys[tid + kTileThreads]) : 0xFFFFFFFFu;
  if (p0 >= limit) p0 = 0xFFFFFFFFu;  // only when the caller breaks the pool-count precondition
  if (p1 >= limit) p1 = 0xFFFFFFFFu;
  __syncthreads();  // the keys are in registers; the bitmap may reuse their storage
  for (int i = tid; i < words; i += kTileThreads) bitmap[i] = 0u;
  __syncthreads();
  if (p0 != 0xFFFFFFFFu) atomicOr(&bitmap[p0 >> 5], 1u << (p0 & 31));
  if (p1 != 0xFFFFFFFFu) atomicOr(&bitmap[p1 >> 5], 1u << (p1 & 31));
  __syncthreads();
  int g0 = 0, g1 = 0, g2 = 0, g3 = 0;
  if (32 * tid < words) {
#pragma unroll
    for (int j = 0; j < 32; ++j) {
      const int jj = (j + lane) & 31;  // lane-rotated: a warp's 32 reads hit 32 banks
      const int wi = 32 * tid + jj;
      const int c = wi < words ? __popc(bitmap[wi]) : 0;
      g0 += jj < 8 ? c : 0;
      g1 += (jj >> 3) == 1 ? c : 0;
      g2 += (jj >> 3) == 2 ? c : 0;
      g3 += jj >= 24 ? c : 0;
    }
  }
  const int mine = g0 + g1 + g2 + g3;
  int incl = mine;
#pragma unroll
  for (int o = 1; o < 32; o <<= 1) {
    const int v = __shfl_up_sync(0xffffffffu, incl, o);
    if (lane >= o) incl += v;
  }
  if (lane == 31) wsum[warp] = incl;
  __syncthreads();
  int before = incl - mine;
#pragma unroll
  for (int i = 0; i < kTileWarps; ++i) before += i < warp ? wsum[i] : 0;
  prefix[4 * tid] = before;
  prefix[4 * tid + 1] = before + g0;
  prefix[4 * tid + 2] = before + g0 + g1;
  prefix[4 * tid + 3] = before + g0 + g1 + g2;
  __syncthreads();
  auto place = [&](uint32_t p) {
    const int wi = int(p >> 5), grp = wi >> 3;
    int at = prefix[grp] + __popc(bitmap[wi] & ((1u << (p & 31)) - 1u));
#pragma unroll
    for (int j = 0; j < 7; ++j) at += grp * 8 + j < wi ? __popc(bitmap[grp * 8 + j]) : 0;
    pools[at] = p;
  };
  if (p0 != 0xFFFFFFFFu) place(p0);
  if (p1 != 0xFFFFFFFFu) place(p1);
  __syncthreads();
  for (int i = tid; i < kTopPools; i += kTileThreads)
    pools_out[int64_t(row) * kTopPools + i] = i < kept ? int32_t(pools[i]) : -1;
  const int tail_start = n * kPool;
  const int total = kept * kPool + (pos + 1 - tail_start);
  for (int j = tid; j < kMaxTokens; j += kTileThreads) {
    int32_t tok = -1;
    if (j < kept * kPool) tok = int32_t(pools[j / kPool]) * kPool + (j % kPool);
    else if (j < total) tok = tail_start + (j - kept * kPool);
    tokens_out[int64_t(row) * kMaxTokens + j] = tok;
  }
  if (tid == 0) {
    counts_out[row * 2] = kept;
    counts_out[row * 2 + 1] = total;
  }
}

__global__ __launch_bounds__(kTileThreads, 2) void index_fused_kernel(const float* q, const float* w,
    float score_scale, const int32_t* row_pos, const int32_t* row_req, glm53f_dsa_cache_t cache, int chunk_pools,
    FusedLayout lay, uint8_t* ws, int32_t* pools_out, int32_t* tokens_out, int32_t* counts_out, float* debug_scores,
    int debug_stride) {
  extern __shared__ __align__(16) unsigned char smem[];
  FusedScratch& sc = *reinterpret_cast<FusedScratch*>(smem + kSmScratch);
  uint64_t* in = reinterpret_cast<uint64_t*>(smem + kSmIn);
  uint64_t* stage = reinterpret_cast<uint64_t*>(smem + kSmStage);
  const int chunk = blockIdx.x, row = blockIdx.y;
  const int pos = row_pos[row], req = row_req[row];
  const int n = (pos + 1) / kPool;
  if (n <= kTopPools) {
    if (chunk == 0) write_dense_row(row, pos, n, pools_out, tokens_out, counts_out);
    return;
  }
  const int begin = chunk * chunk_pools;
  const int end = min(n, begin + chunk_pools);
  int kept = 0;
  if (begin < end) kept = fused_score_chunk(smem, sc, q, w, score_scale, row, req, cache, begin, end, debug_scores, debug_stride);

  int idx = chunk;
  for (int level = 0; level < lay.levels; ++level) {
    const int nodes = lay.nodes[level], parents = lay.nodes[level + 1];
    uint64_t* lists = reinterpret_cast<uint64_t*>(ws + lay.lists[level]);
    int32_t* counts = reinterpret_cast<int32_t*>(ws + lay.counts[level]);
    uint32_t* arrivals = reinterpret_cast<uint32_t*>(ws + lay.counters[level]);
    uint64_t* mine = lists + (int64_t(row) * nodes + idx) * kTopPools;
    for (int i = threadIdx.x; i < kept; i += blockDim.x) mine[i] = in[i];
    if (threadIdx.x == 0) counts[int64_t(row) * nodes + idx] = kept;
    __threadfence();
    __syncthreads();
    const int parent = idx / kFanIn;
    const int fan = min(kFanIn, nodes - parent * kFanIn);
    if (threadIdx.x == 0) {
      uint32_t* c = arrivals + int64_t(row) * parents + parent;
      const uint32_t before = atomicAdd(c, 1u);
      sc.flag = before == uint32_t(fan - 1);
      if (sc.flag) *c = 0u;  // every arrival has happened; ready for the next call
    }
    __syncthreads();
    if (!sc.flag) return;
    __threadfence();
    // Gather: the children's counts and every key slot (fixed addresses) are
    // loaded together, one L2 round trip, then packed into `in`. Thread t's
    // slot r is entry (r % 2) * 256 + t of child r / 2; slots past a child's
    // count hold stale data and are dropped. The radix select's first digit
    // comes from the keys' differences to child 0's first key, which is valid
    // whenever any child has keys (empty children are the rightmost).
    const int64_t child0 = int64_t(row) * nodes + parent * kFanIn;
    const uint64_t* first = lists + child0 * kTopPools;
    constexpr int kPer = kFanIn * kTopPools / kTileThreads;  // 16
    static_assert(kTopPools == 2 * kTileThreads, "two slots of each child per thread");
    int cnt[kFanIn];
    uint64_t got[kPer];
    const uint64_t ref = __ldcg(first);
#pragma unroll
    for (int c = 0; c < kFanIn; ++c) cnt[c] = c < fan ? __ldcg(counts + child0 + c) : 0;
#pragma unroll
    for (int r = 0; r < kPer; ++r) got[r] = r / 2 < fan ? __ldcg(first + r * kTileThreads + threadIdx.x) : 0ull;
    int total = 0;
    uint64_t diff = 0;
#pragma unroll
    for (int r = 0; r < kPer; ++r) {
      const int i = (r % 2) * kTileThreads + int(threadIdx.x);
      if (i < cnt[r / 2]) {
        in[total + i] = got[r];
        diff |= got[r] ^ ref;
      }
      if (r % 2 == 1) total += cnt[r / 2];
    }
    {
      const uint32_t dlo = __reduce_or_sync(0xffffffffu, uint32_t(diff));
      const uint32_t dhi = __reduce_or_sync(0xffffffffu, uint32_t(diff >> 32));
      if ((threadIdx.x & 31) == 0) sc.red[threadIdx.x >> 5] = (uint64_t(dhi) << 32) | dlo;
    }
    radix_reset(sc.sel);
    __syncthreads();
    kept = total;
    if (kept > kTopPools) {
      uint64_t d = 0;
#pragma unroll
      for (int i = 0; i < kTileWarps; ++i) d |= sc.red[i];
      const uint64_t theta = radix_kth(ref, d, kTopPools, sc.sel, SmemKeys{in, kept});
      keep_at_least(in, kept, kTopPools, theta, stage, sc.scan);
      kept = kTopPools;
    }
    idx = parent;
  }
  write_sparse_row(in, kept, reinterpret_cast<uint32_t*>(in), reinterpret_cast<uint32_t*>(smem + kSmPools),
                   reinterpret_cast<int*>(smem + kSmStage), sc.scan,
                   row, pos, n, pools_out, tokens_out, counts_out);
}

struct Workspace {
  uint64_t* lists[2];
  int32_t* counts[2];
};

uint64_t workspace_layout(int rows, int chunks, Workspace* ws, void* base) {
  const uint64_t a_lists = uint64_t(rows) * chunks * kTopPools * 8;
  const uint64_t a_counts = align_up(uint64_t(rows) * chunks * 4, 256);
  const int b_chunks = (chunks + kMergeFanIn - 1) / kMergeFanIn;
  const uint64_t b_lists = uint64_t(rows) * b_chunks * kTopPools * 8;
  const uint64_t b_counts = align_up(uint64_t(rows) * b_chunks * 4, 256);
  if (ws && base) {
    auto* p = static_cast<uint8_t*>(base);
    ws->lists[0] = reinterpret_cast<uint64_t*>(p);
    ws->counts[0] = reinterpret_cast<int32_t*>(p + a_lists);
    ws->lists[1] = reinterpret_cast<uint64_t*>(p + a_lists + a_counts);
    ws->counts[1] = reinterpret_cast<int32_t*>(p + a_lists + a_counts + b_lists);
  }
  return a_lists + a_counts + b_lists + b_counts;
}

}  // namespace
}  // namespace glm53f

using namespace glm53f;

extern "C" int32_t glm53f_dsa_index_pool_write(const float* k_raw, const float* gate, const float* ln_w,
    const float* ln_b, float ln_eps, const float* ape, const uint8_t* tails,
    const glm53f_dsa_window_t* windows, const int32_t* row_req, int32_t rows,
    glm53f_dsa_cache_t cache, void* stream) {
  if (rows < 0 || !valid_cache(cache)) return cudaErrorInvalidValue;
  if (rows == 0) return cudaSuccess;
  if (!k_raw || !gate || !ln_w || !ln_b || !ape || !tails || !windows || !row_req) return cudaErrorInvalidValue;
  pool_write_kernel<<<rows, 128, 0, static_cast<cudaStream_t>(stream)>>>(k_raw, gate, ln_w, ln_b, ln_eps, ape,
                                                                         tails, windows, row_req, cache);
  return cudaGetLastError();
}

extern "C" int32_t glm53f_dsa_index_tail_commit(const float* k_raw, const float* gate, const float* ln_w,
    const float* ln_b, float ln_eps, uint8_t* tails, const glm53f_dsa_window_t* windows, int32_t n_req,
    void* stream) {
  if (n_req < 0) return cudaErrorInvalidValue;
  if (n_req == 0) return cudaSuccess;
  if (!k_raw || !gate || !ln_w || !ln_b || !tails || !windows) return cudaErrorInvalidValue;
  tail_commit_kernel<<<n_req, 128, 0, static_cast<cudaStream_t>(stream)>>>(k_raw, gate, ln_w, ln_b, ln_eps, tails,
                                                                           windows);
  return cudaGetLastError();
}

extern "C" uint64_t glm53f_dsa_index_workspace_bytes(int32_t rows, int32_t chunks) {
  if (rows < 1 || chunks < 1) return 0;
  const uint64_t v1 = workspace_layout(rows, chunks, nullptr, nullptr);
  const uint64_t v2 = fused_layout(rows, chunks).total;
  return v1 > v2 ? v1 : v2;
}

extern "C" void glm53f_dsa_index_plan(int32_t rows, int32_t max_pools, int32_t sms, int32_t* chunk_pools,
                                      int32_t* chunks) {
  rows = rows < 1 ? 1 : rows;
  sms = sms < 1 ? 1 : sms;
  max_pools = max_pools < 1 ? 1 : max_pools;
  // Measured on the RTX 4090 with the fused kernel: up to 64K pools, one or two
  // rows do best with about one block per multiprocessor (a shallower merge
  // tree), everything else with two. A chunk holds at least max_pools / 64
  // pools (64 to 256), which keeps short rows to at most 64 chunks.
  const int blocks = (rows <= 2 && max_pools <= 65536) ? sms : 2 * sms;
  const int per_row = blocks / rows > 1 ? blocks / rows : 1;
  int cp = (max_pools + per_row - 1) / per_row;
  cp = (cp + 63) / 64 * 64;
  int min_cp = (max_pools / 64 + 63) / 64 * 64;
  min_cp = min_cp < 64 ? 64 : (min_cp > 256 ? 256 : min_cp);
  if (cp < min_cp) cp = min_cp;
  *chunk_pools = cp;
  *chunks = (max_pools + cp - 1) / cp;
}

namespace {
int32_t check_select_args(const float* q, const float* w, const int32_t* row_pos, const int32_t* row_req,
                          int32_t rows, int32_t max_pools, const glm53f_dsa_cache_t& cache, int32_t chunk_pools,
                          int32_t chunks, void* workspace, int32_t* pools_out, int32_t* tokens_out,
                          int32_t* counts_out) {
  if (rows < 0 || max_pools < 0 || chunk_pools < 64 || chunk_pools % 64 || chunks < 1 ||
      int64_t(chunk_pools) * chunks < max_pools || !valid_cache(cache))
    return cudaErrorInvalidValue;
  if (rows == 0) return cudaSuccess;
  if (!q || !w || !row_pos || !row_req || !pools_out || !tokens_out || !counts_out || !workspace)
    return cudaErrorInvalidValue;
  return cudaSuccess;
}

int32_t launch_fused(const float* q, const float* w, float score_scale, const int32_t* row_pos,
                     const int32_t* row_req, int32_t rows, int32_t max_pools, glm53f_dsa_cache_t cache,
                     int32_t chunk_pools, int32_t chunks, void* workspace, uint64_t workspace_bytes,
                     int32_t* pools_out, int32_t* tokens_out, int32_t* counts_out, float* debug_scores,
                     cudaStream_t s, bool reset_counters) {
  const FusedLayout lay = fused_layout(rows, chunks);
  if (lay.nodes[lay.levels] != 1 || lay.total > workspace_bytes || max_pools > 32 * kBitmapWords)
    return cudaErrorInvalidValue;
  auto* ws = static_cast<uint8_t*>(workspace);
  if (reset_counters && lay.counter_bytes) {
    const cudaError_t st = cudaMemsetAsync(ws + lay.counter_begin, 0, lay.counter_bytes, s);
    if (st != cudaSuccess) return st;
  }
  index_fused_kernel<<<dim3(chunks, rows), kTileThreads, kFusedSmem, s>>>(q, w, score_scale, row_pos, row_req, cache,
                                                                          chunk_pools, lay, ws, pools_out, tokens_out,
                                                                          counts_out, debug_scores, max_pools);
  return cudaGetLastError();
}
}  // namespace

extern "C" int32_t glm53f_dsa_index_select(const float* q, const float* w, float score_scale,
    const int32_t* row_pos, const int32_t* row_req, int32_t rows, int32_t max_pools,
    glm53f_dsa_cache_t cache, int32_t chunk_pools, int32_t chunks, void* workspace, uint64_t workspace_bytes,
    int32_t* pools_out, int32_t* tokens_out, int32_t* counts_out, float* debug_scores, void* stream) {
  const int32_t bad = check_select_args(q, w, row_pos, row_req, rows, max_pools, cache, chunk_pools, chunks,
                                        workspace, pools_out, tokens_out, counts_out);
  if (bad != cudaSuccess || rows == 0) return bad;
  return launch_fused(q, w, score_scale, row_pos, row_req, rows, max_pools, cache, chunk_pools, chunks, workspace,
                      workspace_bytes, pools_out, tokens_out, counts_out, debug_scores,
                      static_cast<cudaStream_t>(stream), true);
}

extern "C" int32_t glm53f_dsa_index_select_prepared(const float* q, const float* w, float score_scale,
    const int32_t* row_pos, const int32_t* row_req, int32_t rows, int32_t max_pools,
    glm53f_dsa_cache_t cache, int32_t chunk_pools, int32_t chunks, void* workspace, uint64_t workspace_bytes,
    int32_t* pools_out, int32_t* tokens_out, int32_t* counts_out, float* debug_scores, void* stream) {
  const int32_t bad = check_select_args(q, w, row_pos, row_req, rows, max_pools, cache, chunk_pools, chunks,
                                        workspace, pools_out, tokens_out, counts_out);
  if (bad != cudaSuccess || rows == 0) return bad;
  return launch_fused(q, w, score_scale, row_pos, row_req, rows, max_pools, cache, chunk_pools, chunks, workspace,
                      workspace_bytes, pools_out, tokens_out, counts_out, debug_scores,
                      static_cast<cudaStream_t>(stream), false);
}

extern "C" int32_t glm53f_dsa_index_select_v1(const float* q, const float* w, float score_scale,
    const int32_t* row_pos, const int32_t* row_req, int32_t rows, int32_t max_pools,
    glm53f_dsa_cache_t cache, int32_t chunk_pools, int32_t chunks, void* workspace, uint64_t workspace_bytes,
    int32_t* pools_out, int32_t* tokens_out, int32_t* counts_out, float* debug_scores, void* stream) {
  const int32_t bad = check_select_args(q, w, row_pos, row_req, rows, max_pools, cache, chunk_pools, chunks,
                                        workspace, pools_out, tokens_out, counts_out);
  if (bad != cudaSuccess || rows == 0) return bad;
  Workspace ws;
  if (workspace_layout(rows, chunks, &ws, workspace) > workspace_bytes) return cudaErrorInvalidValue;
  auto s = static_cast<cudaStream_t>(stream);
  index_tiles_kernel<<<dim3(chunks, rows), kTileThreads, 0, s>>>(q, w, score_scale, row_pos, row_req, cache,
                                                                chunk_pools, chunks, ws.lists[0], ws.counts[0],
                                                                debug_scores, max_pools);
  cudaError_t st = cudaGetLastError();
  if (st != cudaSuccess) return st;
  int cur = 0, lists = chunks;
  while (lists > 1) {
    const int next = (lists + kMergeFanIn - 1) / kMergeFanIn;
    index_merge_kernel<<<dim3(next, rows), kMergeThreads, 0, s>>>(ws.lists[cur], ws.counts[cur], lists,
                                                                  ws.lists[cur ^ 1], ws.counts[cur ^ 1], next);
    st = cudaGetLastError();
    if (st != cudaSuccess) return st;
    cur ^= 1;
    lists = next;
  }
  index_finalize_kernel<<<rows, kTopPools, 0, s>>>(row_pos, ws.lists[cur], ws.counts[cur], pools_out, tokens_out,
                                                  counts_out);
  return cudaGetLastError();
}
