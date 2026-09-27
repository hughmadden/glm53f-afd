// DSA indexer kernels for GLM-5.3-Flash: pooled-key write, tail commit, and the
// score + top-512 pool selection over up to 256K pools (1M tokens).
//
// Selection (see crates/glm53f-dsa/README.md, "Long-context selection"):
//   1. index_tiles: grid (chunks, rows). A block owns one row and a contiguous
//      chunk of pools. Warps stream 8-pool tiles: 32 B of FP8 codes per lane,
//      f16 tensor-core products against the row's 32 x 128 query (held in
//      registers), ReLU and head weights in f32, then a key
//      (ordered score << 32 | ~pool). Keys above the block's running threshold
//      go to a 2,048-entry shared buffer; when it fills, a radix select keeps the
//      best 512 and raises the threshold. The block writes at most 512 keys.
//   2. index_merge: groups of 8 lists -> 1 list of 512, repeated.
//   3. index_finalize: kept pools ascending, expanded tokens plus the tail.
// No score matrix is materialized; the workspace is rows x chunks x 4 KiB.
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
  return workspace_layout(rows, chunks, nullptr, nullptr);
}

extern "C" void glm53f_dsa_index_plan(int32_t rows, int32_t max_pools, int32_t sms, int32_t* chunk_pools,
                                      int32_t* chunks) {
  rows = rows < 1 ? 1 : rows;
  sms = sms < 1 ? 1 : sms;
  max_pools = max_pools < 1 ? 1 : max_pools;
  const int per_row = (2 * sms) / rows > 1 ? (2 * sms) / rows : 1;
  int cp = (max_pools + per_row - 1) / per_row;
  cp = (cp + 63) / 64 * 64;
  if (cp < 256) cp = 256;
  *chunk_pools = cp;
  *chunks = (max_pools + cp - 1) / cp;
}

extern "C" int32_t glm53f_dsa_index_select(const float* q, const float* w, float score_scale,
    const int32_t* row_pos, const int32_t* row_req, int32_t rows, int32_t max_pools,
    glm53f_dsa_cache_t cache, int32_t chunk_pools, int32_t chunks, void* workspace, uint64_t workspace_bytes,
    int32_t* pools_out, int32_t* tokens_out, int32_t* counts_out, float* debug_scores, void* stream) {
  if (rows < 0 || max_pools < 0 || chunk_pools < 64 || chunk_pools % 64 || chunks < 1 ||
      int64_t(chunk_pools) * chunks < max_pools || !valid_cache(cache))
    return cudaErrorInvalidValue;
  if (rows == 0) return cudaSuccess;
  if (!q || !w || !row_pos || !row_req || !pools_out || !tokens_out || !counts_out || !workspace)
    return cudaErrorInvalidValue;
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
