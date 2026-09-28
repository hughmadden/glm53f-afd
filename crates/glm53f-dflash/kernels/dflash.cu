/* glm53f-dflash kernels (see glm53f_dflash.h for the contract of each entry point).
 *
 * The attention's split-K structure (one block per request, KV head and key range, then a merge
 * kernel) and the per-request ring follow the DFlash drafter of mimo26f-afd v1.2.0
 * (crates/mimo26-coordinator/kernels/dflash.cu, MIT); the code is new: grouped-query heads, lanes
 * over the keys of a tile, no sink or value scale, a per-request lowest position. See
 * PROVENANCE.md.
 */
#include "glm53f_dflash.h"

#include <cuda_bf16.h>
#include <math.h>

namespace {

constexpr int kHeadDim = 128;
constexpr int kBlock = 8;
constexpr int kTopK = 16;
constexpr int kTile = 32;

__device__ __forceinline__ float bf2f(uint16_t b) { return __uint_as_float(((uint32_t)b) << 16); }
__device__ __forceinline__ uint16_t f2bf(float x) { return __bfloat16_as_ushort(__float2bfloat16_rn(x)); }

/* ---------------------------------------------------------------------------------------------
 * RMSNorm: one block of 256 threads per row; the mean of squares in f64. */
__global__ void __launch_bounds__(256) rmsnorm_kernel(const float* __restrict__ x, int64_t ldx,
                                                      const uint16_t* __restrict__ w, int n, float eps,
                                                      float* y, int64_t ldy, uint16_t* yb, int64_t ldb) {
  __shared__ double part[8];
  const float* row = x + (int64_t)blockIdx.x * ldx;
  double s = 0.0;
  for (int i = threadIdx.x; i < n; i += blockDim.x) {
    const double v = (double)row[i];
    s += v * v;
  }
  for (int o = 16; o > 0; o >>= 1) s += __shfl_xor_sync(0xffffffffu, s, o);
  if ((threadIdx.x & 31) == 0) part[threadIdx.x >> 5] = s;
  __syncthreads();
  double tot = 0.0;
  for (int i = 0; i < (int)(blockDim.x >> 5); ++i) tot += part[i];
  const float ms = (float)(tot / (double)n);
  const float r = 1.0f / sqrtf(ms + eps);
  for (int i = threadIdx.x; i < n; i += blockDim.x) {
    const float v = bf2f(w[i]) * (row[i] * r);
    if (yb) yb[(int64_t)blockIdx.x * ldb + i] = f2bf(v);
    if (y) y[(int64_t)blockIdx.x * ldy + i] = v;
  }
}

/* ---------------------------------------------------------------------------------------------
 * RoPE table: cos and sin of pos[row] * inv_freq[i] per row, the angle an f32 product, its sine
 * and cosine in f64 rounded to f32. cs[row][i] = cos, cs[row][half + i] = sin. */
__global__ void rope_table_kernel(const int64_t* __restrict__ pos, int rows, const float* __restrict__ inv_freq,
                                  int half, float* cs) {
  const int64_t total = (int64_t)rows * half;
  for (int64_t t = blockIdx.x * (int64_t)blockDim.x + threadIdx.x; t < total; t += (int64_t)gridDim.x * blockDim.x) {
    const int row = (int)(t / half), i = (int)(t % half);
    const float angle = (float)pos[row] * inv_freq[i];
    double sd, cd;
    sincos((double)angle, &sd, &cd);
    cs[(int64_t)row * 2 * half + i] = (float)cd;
    cs[(int64_t)row * 2 * half + half + i] = (float)sd;
  }
}

/* ---------------------------------------------------------------------------------------------
 * Per-head RMSNorm + RoPE: one warp per (row, head); lane owns dims [4 lane, 4 lane + 4). */
__global__ void __launch_bounds__(256) head_norm_rope_kernel(float* x, int64_t ldx, int rows, int heads,
                                                             const uint16_t* __restrict__ w,
                                                             const float* __restrict__ cs, float eps,
                                                             uint16_t* out, int64_t ldo) {
  const int item = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
  if (item >= rows * heads) return;
  const int lane = threadIdx.x & 31;
  const int row = item / heads, head = item % heads;
  float* p = x + (int64_t)row * ldx + head * kHeadDim;
  float v[4];
  double s = 0.0;
#pragma unroll
  for (int d = 0; d < 4; ++d) {
    v[d] = p[lane * 4 + d];
    s += (double)v[d] * (double)v[d];
  }
  for (int o = 16; o > 0; o >>= 1) s += __shfl_xor_sync(0xffffffffu, s, o);
  const float ms = (float)(s / (double)kHeadDim);
  const float r = 1.0f / sqrtf(ms + eps);
#pragma unroll
  for (int d = 0; d < 4; ++d) v[d] = bf2f(w[lane * 4 + d]) * (v[d] * r);
  /* The partner of dim i (i < 64) is i + 64, 16 lanes away. */
  float other[4];
#pragma unroll
  for (int d = 0; d < 4; ++d) other[d] = __shfl_xor_sync(0xffffffffu, v[d], 16);
  const float* c = cs + (int64_t)row * kHeadDim;
  const bool low = lane < 16;
#pragma unroll
  for (int d = 0; d < 4; ++d) {
    const int i = (lane * 4 + d) & 63;
    const float cn = c[i], sn = c[64 + i];
    /* low: x[i] * cos - x[i + 64] * sin; high: x[i + 64] * cos + x[i] * sin */
    const float o = low ? v[d] * cn + (-other[d]) * sn : v[d] * cn + other[d] * sn;
    p[lane * 4 + d] = o;
    if (out) out[(int64_t)row * ldo + head * kHeadDim + lane * 4 + d] = f2bf(o);
  }
}

/* ---------------------------------------------------------------------------------------------
 * K/V rows into rings. */
__global__ void store_kv_kernel(const float* __restrict__ k, const float* __restrict__ v, int64_t ld, int rows,
                                int kvw, const int32_t* __restrict__ req, const int64_t* __restrict__ pos,
                                const uint64_t* __restrict__ bases, int layer, int ring) {
  const int64_t total = (int64_t)rows * kvw;
  for (int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x; i < total; i += (int64_t)gridDim.x * blockDim.x) {
    const int r = (int)(i / kvw), e = (int)(i % kvw);
    uint16_t* base = (uint16_t*)bases[req[r]];
    const int64_t slot = pos[r] % ring;
    base[((int64_t)(layer * 2) * ring + slot) * kvw + e] = f2bf(k[(int64_t)r * ld + e]);
    base[((int64_t)(layer * 2 + 1) * ring + slot) * kvw + e] = f2bf(v[(int64_t)r * ld + e]);
  }
}

/* ---------------------------------------------------------------------------------------------
 * Dynamic convolution: one thread per (row, channel). */
__global__ void dyn_conv_kernel(const float* __restrict__ x, int64_t ldx, const float* __restrict__ dyn, int64_t lddyn,
                                const uint16_t* __restrict__ base, int rows, int n, int gs, int taps, int block,
                                float* out, int64_t ldo, uint16_t* outb, int64_t ldb, float* resid, int64_t ldr) {
  const int64_t total = (int64_t)rows * n;
  const int groups = n / gs;
  for (int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x; i < total; i += (int64_t)gridDim.x * blockDim.x) {
    const int l = (int)(i / n), c = (int)(i % n);
    const int in_block = l % block;
    float acc = 0.0f;
    for (int o = 0; o < taps && o <= in_block; ++o) {
      const float src = x[(int64_t)(l - o) * ldx + c];
      acc = acc + bf2f(base[o * n + c]) * src;
      acc = acc + dyn[(int64_t)l * lddyn + o * groups + c / gs] * src;
    }
    if (out) out[(int64_t)l * ldo + c] = acc;
    if (outb) outb[(int64_t)l * ldb + c] = f2bf(acc);
    if (resid) resid[(int64_t)l * ldr + c] += acc;
  }
}

/* ---------------------------------------------------------------------------------------------
 * Attention, split over key ranges. Grid (nreq * kv_heads, splits); block = group warps (group <=
 * 4). Warp w is query head kvh * group + w and carries the 8 block rows. Keys come in tiles of 32
 * (K as bf16, V as f32, in shared memory); a lane scores one key against the 8 rows, the warp
 * updates each row's online softmax once per tile, then a lane accumulates 4 of the 128 output
 * dims over the tile's keys. Partials [req][split][row][head][2 + 128]: (max, sum, unnormalized
 * output). Sums use fused multiply-adds (__fmaf_rn); the attention is not bit-compared with the
 * CPU. */
constexpr int kKStride = 130; /* u16 per shared K row: 65 words, so a column read is conflict-free */

__global__ void __launch_bounds__(128) attn_split_kernel(const float* __restrict__ q, int64_t ldq, int heads,
                                                         int kv_heads, const int64_t* __restrict__ start,
                                                         const int64_t* __restrict__ lo,
                                                         const uint64_t* __restrict__ bases, int layer, int ring,
                                                         int window_left, float scale, int split_keys, int splits,
                                                         float* __restrict__ part) {
  __shared__ uint16_t ks[kTile][kKStride];
  __shared__ __align__(16) float vs[kTile][kHeadDim];
  __shared__ float qs[4][kBlock][kHeadDim];
  __shared__ float ps[4][kBlock][kTile];
  const int req = blockIdx.x / kv_heads;
  const int kvh = blockIdx.x % kv_heads;
  const int split = blockIdx.y;
  const int group = heads / kv_heads;
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int qh = kvh * group + warp;
  const int kvw = kv_heads * kHeadDim;
  const int64_t st = start[req];
  int64_t klo = st - window_left;
  if (klo < lo[req]) klo = lo[req];
  if (klo < 0) klo = 0;
  const int64_t khi = st + kBlock;
  const int64_t k0 = klo + (int64_t)split * split_keys;
  const int64_t k1 = k0 + split_keys < khi ? k0 + split_keys : khi;
  const uint16_t* base = (const uint16_t*)bases[req];
  const uint16_t* rk = base + (int64_t)(layer * 2) * ring * kvw + kvh * kHeadDim;
  const uint16_t* rv = base + (int64_t)(layer * 2 + 1) * ring * kvw + kvh * kHeadDim;

  for (int i = lane; i < kBlock * kHeadDim; i += 32) {
    const int r = i / kHeadDim, d = i % kHeadDim;
    qs[warp][r][d] = q[(int64_t)(req * kBlock + r) * ldq + qh * kHeadDim + d];
  }
  float acc[kBlock][4], m[kBlock], l[kBlock];
  int64_t rlo[kBlock];
#pragma unroll
  for (int r = 0; r < kBlock; ++r) {
#pragma unroll
    for (int d = 0; d < 4; ++d) acc[r][d] = 0.0f;
    int64_t b = st + r - window_left;
    if (b < lo[req]) b = lo[req];
    rlo[r] = b;
    m[r] = -INFINITY;
    l[r] = 0.0f;
  }
  for (int64_t t0 = k0; t0 < k1; t0 += kTile) {
    const int nk = (int)(k1 - t0 < kTile ? k1 - t0 : kTile);
    const int row0 = (int)(t0 % ring);
    __syncthreads();
    /* 8 values (16 bytes) of one key per step. */
    for (int i = threadIdx.x; i < kTile * (kHeadDim / 8); i += blockDim.x) {
      const int j = i / (kHeadDim / 8), c8 = (i % (kHeadDim / 8)) * 8;
      uint4 kr = make_uint4(0, 0, 0, 0), vr = make_uint4(0, 0, 0, 0);
      if (j < nk) {
        int row = row0 + j;
        if (row >= ring) row -= ring;
        kr = *reinterpret_cast<const uint4*>(rk + (int64_t)row * kvw + c8);
        vr = *reinterpret_cast<const uint4*>(rv + (int64_t)row * kvw + c8);
      }
      const uint32_t kw[4] = {kr.x, kr.y, kr.z, kr.w}, vw[4] = {vr.x, vr.y, vr.z, vr.w};
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        ks[j][c8 + 2 * e] = (uint16_t)(kw[e] & 0xffffu);
        ks[j][c8 + 2 * e + 1] = (uint16_t)(kw[e] >> 16);
        vs[j][c8 + 2 * e] = __uint_as_float(vw[e] << 16);
        vs[j][c8 + 2 * e + 1] = __uint_as_float(vw[e] & 0xffff0000u);
      }
    }
    __syncthreads();
    if (warp >= group) continue;
    float sc[kBlock];
#pragma unroll
    for (int r = 0; r < kBlock; ++r) sc[r] = 0.0f;
    if (lane < nk) {
      for (int d = 0; d < kHeadDim; d += 2) {
        const uint32_t kp2 = *reinterpret_cast<const uint32_t*>(&ks[lane][d]);
        const float ka = __uint_as_float(kp2 << 16), kb = __uint_as_float(kp2 & 0xffff0000u);
#pragma unroll
        for (int r = 0; r < kBlock; ++r) sc[r] = __fmaf_rn(qs[warp][r][d + 1], kb, __fmaf_rn(qs[warp][r][d], ka, sc[r]));
      }
    }
    const int64_t kp = t0 + lane;
#pragma unroll
    for (int r = 0; r < kBlock; ++r) {
      const bool valid = lane < nk && kp >= rlo[r];
      const float s = valid ? sc[r] * scale : -INFINITY;
      float mt = s;
      for (int o = 16; o > 0; o >>= 1) mt = fmaxf(mt, __shfl_xor_sync(0xffffffffu, mt, o));
      const float mn = fmaxf(m[r], mt);
      float p = 0.0f, c = 1.0f;
      if (mn != -INFINITY) {
        p = valid ? expf(s - mn) : 0.0f;
        c = expf(m[r] - mn);
      }
      float sum = p;
      for (int o = 16; o > 0; o >>= 1) sum += __shfl_xor_sync(0xffffffffu, sum, o);
      l[r] = l[r] * c + sum;
#pragma unroll
      for (int d = 0; d < 4; ++d) acc[r][d] *= c;
      m[r] = mn;
      ps[warp][r][lane] = p;
    }
    __syncwarp();
    for (int j = 0; j < nk; ++j) {
      const float4 vv = *reinterpret_cast<const float4*>(&vs[j][lane * 4]);
#pragma unroll
      for (int r = 0; r < kBlock; ++r) {
        const float pj = ps[warp][r][j];
        acc[r][0] = __fmaf_rn(pj, vv.x, acc[r][0]);
        acc[r][1] = __fmaf_rn(pj, vv.y, acc[r][1]);
        acc[r][2] = __fmaf_rn(pj, vv.z, acc[r][2]);
        acc[r][3] = __fmaf_rn(pj, vv.w, acc[r][3]);
      }
    }
    __syncwarp();
  }
  if (warp >= group) return;
#pragma unroll
  for (int r = 0; r < kBlock; ++r) {
    float* out = part + ((((int64_t)req * splits + split) * kBlock + r) * heads + qh) * (2 + kHeadDim);
    if (lane == 0) {
      out[0] = m[r];
      out[1] = l[r];
    }
#pragma unroll
    for (int d = 0; d < 4; ++d) out[2 + lane * 4 + d] = acc[r][d];
  }
}

/* Merge the splits: one block of 128 threads per (row, head). */
__global__ void attn_merge_kernel(const float* __restrict__ part, int heads, int splits, uint16_t* outb, float* outf,
                                  int64_t ldo) {
  const int row = blockIdx.x / heads, qh = blockIdx.x % heads;
  const int req = row / kBlock, r = row % kBlock;
  const int d = threadIdx.x;
  float mx = -INFINITY;
  for (int s = 0; s < splits; ++s) {
    const float* p = part + ((((int64_t)req * splits + s) * kBlock + r) * heads + qh) * (2 + kHeadDim);
    if (p[1] > 0.0f) mx = fmaxf(mx, p[0]);
  }
  float den = 0.0f, num = 0.0f;
  for (int s = 0; s < splits; ++s) {
    const float* p = part + ((((int64_t)req * splits + s) * kBlock + r) * heads + qh) * (2 + kHeadDim);
    if (!(p[1] > 0.0f)) continue;
    const float w = expf(p[0] - mx);
    den = den + w * p[1];
    num = num + w * p[2 + d];
  }
  const float o = num / den;
  if (outb) outb[(int64_t)row * ldo + qh * kHeadDim + d] = f2bf(o);
  if (outf) outf[(int64_t)row * ldo + qh * kHeadDim + d] = o;
}

/* ---------------------------------------------------------------------------------------------
 * Elementwise. */
__global__ void silu_mul_kernel(const float* __restrict__ gu, int64_t ldgu, int rows, int inter, uint16_t* out,
                                int64_t ldo) {
  const int64_t total = (int64_t)rows * inter;
  for (int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x; i < total; i += (int64_t)gridDim.x * blockDim.x) {
    const int r = (int)(i / inter), c = (int)(i % inter);
    const float g = gu[(int64_t)r * ldgu + c], u = gu[(int64_t)r * ldgu + inter + c];
    out[(int64_t)r * ldo + c] = f2bf((g / (1.0f + expf(-g))) * u);
  }
}

__global__ void block_embed_kernel(const uint16_t* __restrict__ anchor, const uint16_t* __restrict__ mask, int nreq,
                                   int block, int n, float* h) {
  const int64_t total = (int64_t)nreq * block * n;
  for (int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x; i < total; i += (int64_t)gridDim.x * blockDim.x) {
    const int64_t row = i / n;
    const int c = (int)(i % n);
    const int r = (int)(row / block), j = (int)(row % block);
    h[i] = bf2f(j == 0 ? anchor[(int64_t)r * n + c] : mask[c]);
  }
}

__global__ void gather_drafts_kernel(const uint16_t* __restrict__ src, int64_t lds, int nreq, int block, int n,
                                     uint16_t* dst) {
  const int drafts = block - 1;
  const int64_t total = (int64_t)nreq * drafts * n;
  for (int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x; i < total; i += (int64_t)gridDim.x * blockDim.x) {
    const int64_t row = i / n;
    const int c = (int)(i % n);
    const int r = (int)(row / drafts), j = (int)(row % drafts) + 1;
    dst[i] = src[(int64_t)(r * block + j) * lds + c];
  }
}

/* ---------------------------------------------------------------------------------------------
 * Top-16 per row in two stages. Stage 1: block (row, chunk) of 256 threads; each thread keeps a
 * sorted list of its stride's best 16 (it sees ids in increasing order, so only a strictly larger
 * value displaces), then a tree of pairwise merges gives the chunk's 16. Stage 2: block `row`
 * merges its chunks' lists the same way. Order: larger value first, ties to the lower id. */
constexpr int kTopkChunks = 32;

__device__ __forceinline__ bool better(float va, int ia, float vb, int ib) {
  return va > vb || (va == vb && ia < ib);
}

/* Merge sorted lists t and u of (sv, si) into t. */
__device__ void merge16(float (*sv)[kTopK], int (*si)[kTopK], int t, int u) {
  float mv[kTopK];
  int mi[kTopK];
  int a = 0, b = 0;
  for (int k = 0; k < kTopK; ++k) {
    if (better(sv[t][a], si[t][a], sv[u][b], si[u][b])) {
      mv[k] = sv[t][a];
      mi[k] = si[t][a];
      ++a;
    } else {
      mv[k] = sv[u][b];
      mi[k] = si[u][b];
      ++b;
    }
  }
  for (int k = 0; k < kTopK; ++k) {
    sv[t][k] = mv[k];
    si[t][k] = mi[k];
  }
}

__global__ void __launch_bounds__(256) topk16_stage1(const float* __restrict__ logits, int64_t ld, int limit,
                                                     float* wv, int* wi) {
  __shared__ float sv[256][kTopK];
  __shared__ int si[256][kTopK];
  const int row = blockIdx.x, chunk = blockIdx.y;
  const int per = (limit + kTopkChunks - 1) / kTopkChunks;
  const int lo = chunk * per, hi = lo + per < limit ? lo + per : limit;
  const float* x = logits + (int64_t)row * ld;
  float lv[kTopK];
  int li[kTopK];
#pragma unroll
  for (int k = 0; k < kTopK; ++k) {
    lv[k] = -INFINITY;
    li[k] = 0x7fffffff;
  }
  for (int i = lo + threadIdx.x; i < hi; i += blockDim.x) {
    const float v = x[i];
    if (v > lv[kTopK - 1]) {
      lv[kTopK - 1] = v;
      li[kTopK - 1] = i;
#pragma unroll
      for (int k = kTopK - 1; k > 0; --k) {
        if (lv[k] > lv[k - 1]) {
          const float tv = lv[k];
          lv[k] = lv[k - 1];
          lv[k - 1] = tv;
          const int ti = li[k];
          li[k] = li[k - 1];
          li[k - 1] = ti;
        }
      }
    }
  }
#pragma unroll
  for (int k = 0; k < kTopK; ++k) {
    sv[threadIdx.x][k] = lv[k];
    si[threadIdx.x][k] = li[k];
  }
  for (int w = blockDim.x / 2; w >= 1; w >>= 1) {
    __syncthreads();
    if ((int)threadIdx.x < w) merge16(sv, si, threadIdx.x, threadIdx.x + w);
  }
  __syncthreads();
  if (threadIdx.x < kTopK) {
    const int64_t o = ((int64_t)row * kTopkChunks + chunk) * kTopK + threadIdx.x;
    wv[o] = sv[0][threadIdx.x];
    wi[o] = si[0][threadIdx.x];
  }
}

__global__ void topk16_stage2(const float* __restrict__ wv, const int* __restrict__ wi, float* vals, int32_t* ids) {
  __shared__ float sv[kTopkChunks][kTopK];
  __shared__ int si[kTopkChunks][kTopK];
  const int row = blockIdx.x, t = threadIdx.x;
  for (int k = 0; k < kTopK; ++k) {
    sv[t][k] = wv[((int64_t)row * kTopkChunks + t) * kTopK + k];
    si[t][k] = wi[((int64_t)row * kTopkChunks + t) * kTopK + k];
  }
  for (int w = kTopkChunks / 2; w >= 1; w >>= 1) {
    __syncthreads();
    if (t < w) merge16(sv, si, t, t + w);
  }
  __syncthreads();
  if (t < kTopK) {
    vals[(int64_t)row * kTopK + t] = sv[0][t];
    ids[(int64_t)row * kTopK + t] = si[0][t];
  }
}

/* ---------------------------------------------------------------------------------------------
 * Selector walk: one block per request, one thread per rank index (blockDim = rank rounded up to
 * a multiple of 32). The 16 sums use the tree selector.rs models: within each warp the xor
 * butterfly (16, 8, 4, 2, 1), then over the warps' totals halving strides. */
__global__ void __launch_bounds__(1024) select_kernel(const float* __restrict__ hproj, const float* __restrict__ vals,
                                                      const int32_t* __restrict__ ids,
                                                      const int32_t* __restrict__ anchors,
                                                      const uint16_t* __restrict__ pred,
                                                      const uint16_t* __restrict__ succ, int rank, int slots,
                                                      const float* __restrict__ temperature,
                                                      const float* __restrict__ uniforms, int32_t* tokens,
                                                      int32_t* index, float* scores, float* q, float* conf) {
  __shared__ float wsum[32][kTopK];
  __shared__ float sc[kTopK];
  __shared__ int prev;
  const int req = blockIdx.x;
  const int t = threadIdx.x, lane = t & 31, warp = t >> 5;
  const int nw = blockDim.x >> 5;
  int nw2 = 1;
  while (nw2 < nw) nw2 <<= 1;
  const float temp = temperature[req];
  if (t == 0) prev = anchors[req];
  for (int e = 0; e < slots; ++e) {
    __syncthreads();
    const int64_t row = (int64_t)req * slots + e;
    const int p = prev;
    float a_h = 0.0f;
    if (t < rank) a_h = bf2f(pred[(int64_t)p * rank + t]) * hproj[row * rank + t];
    for (int c = 0; c < kTopK; ++c) {
      const int cand = ids[row * kTopK + c];
      float v = t < rank ? a_h * bf2f(succ[(int64_t)cand * rank + t]) : 0.0f;
      for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
      if (lane == 0) wsum[warp][c] = v;
    }
    __syncthreads();
    if (t < kTopK) {
      float w[32];
      for (int i = 0; i < nw2; ++i) w[i] = i < nw ? wsum[i][t] : 0.0f;
      for (int s = nw2 / 2; s >= 1; s >>= 1)
        for (int i = 0; i < s; ++i) w[i] = w[i] + w[i + s];
      sc[t] = vals[row * kTopK + t] + w[0];
    }
    __syncthreads();
    if (t == 0) {
      int chosen = 0;
      float qrow[kTopK];
      if (!(temp > 0.0f)) {
        float best = sc[0];
        for (int c = 1; c < kTopK; ++c)
          if (sc[c] > best) {
            best = sc[c];
            chosen = c;
          }
        for (int c = 0; c < kTopK; ++c) qrow[c] = c == chosen ? 1.0f : 0.0f;
      } else {
        const float tt = temp < 1e-5f ? 1e-5f : temp;
        float s[kTopK], mx = -INFINITY;
        for (int c = 0; c < kTopK; ++c) {
          s[c] = sc[c] / tt;
          mx = fmaxf(mx, s[c]);
        }
        float z = 0.0f;
        for (int c = 0; c < kTopK; ++c) {
          qrow[c] = expf(s[c] - mx);
          z = z + qrow[c];
        }
        const float u = uniforms[row];
        float cum = 0.0f;
        for (int c = 0; c < kTopK; ++c) {
          qrow[c] = qrow[c] / z;
          cum = cum + qrow[c];
          if (u >= cum) ++chosen;
        }
        if (chosen > kTopK - 1) chosen = kTopK - 1;
      }
      float mx = -INFINITY;
      for (int c = 0; c < kTopK; ++c) mx = fmaxf(mx, sc[c]);
      float z = 0.0f, pc = 0.0f;
      for (int c = 0; c < kTopK; ++c) {
        const float ex = expf(sc[c] - mx);
        z = z + ex;
        if (c == chosen) pc = ex;
      }
      const int tok = ids[row * kTopK + chosen];
      tokens[row] = tok;
      index[row] = chosen;
      conf[row] = pc / z;
      for (int c = 0; c < kTopK; ++c) {
        scores[row * kTopK + c] = sc[c];
        q[row * kTopK + c] = qrow[c];
      }
      prev = tok;
    }
  }
}

int grid_for(int64_t n, int threads) {
  int64_t b = (n + threads - 1) / threads;
  if (b > 8192) b = 8192;
  if (b < 1) b = 1;
  return (int)b;
}

}  // namespace

extern "C" cudaError_t g53d_rmsnorm(const float* x, int64_t ldx, const uint16_t* w, int rows, int n, float eps,
                                    float* y_f32, int64_t ldy, uint16_t* y_bf16, int64_t ldb, cudaStream_t s) {
  if (rows <= 0) return cudaSuccess;
  rmsnorm_kernel<<<rows, 256, 0, s>>>(x, ldx, w, n, eps, y_f32, ldy, y_bf16, ldb);
  return cudaGetLastError();
}

extern "C" cudaError_t g53d_rope_table(const int64_t* pos, int rows, const float* inv_freq, float* cs,
                                       cudaStream_t s) {
  if (rows <= 0) return cudaSuccess;
  rope_table_kernel<<<grid_for((int64_t)rows * (kHeadDim / 2), 256), 256, 0, s>>>(pos, rows, inv_freq, kHeadDim / 2, cs);
  return cudaGetLastError();
}

extern "C" cudaError_t g53d_head_norm_rope(float* x, int64_t ldx, int rows, int heads, const uint16_t* w,
                                           const float* cs, float eps, uint16_t* out, int64_t ldo, cudaStream_t s) {
  if (rows <= 0 || heads <= 0) return cudaSuccess;
  const int items = rows * heads;
  head_norm_rope_kernel<<<(items + 7) / 8, 256, 0, s>>>(x, ldx, rows, heads, w, cs, eps, out, ldo);
  return cudaGetLastError();
}

extern "C" cudaError_t g53d_store_kv(const float* k, const float* v, int64_t ld, int rows, int kv_width,
                                     const int32_t* req, const int64_t* pos, const uint64_t* bases, int layer,
                                     int ring, cudaStream_t s) {
  if (rows <= 0) return cudaSuccess;
  store_kv_kernel<<<grid_for((int64_t)rows * kv_width, 256), 256, 0, s>>>(k, v, ld, rows, kv_width, req, pos, bases,
                                                                         layer, ring);
  return cudaGetLastError();
}

extern "C" cudaError_t g53d_dyn_conv(const float* x, int64_t ldx, const float* dyn, int64_t lddyn, const uint16_t* base,
                                     int rows, int n, int group_size, int taps, int block, float* out_f32, int64_t ldo,
                                     uint16_t* out_bf16, int64_t ldb, float* resid, int64_t ldr, cudaStream_t s) {
  if (rows <= 0) return cudaSuccess;
  if (group_size <= 0 || n % group_size != 0 || block <= 0 || rows % block != 0 || taps < 1 || taps > block)
    return cudaErrorInvalidValue;
  dyn_conv_kernel<<<grid_for((int64_t)rows * n, 256), 256, 0, s>>>(x, ldx, dyn, lddyn, base, rows, n, group_size, taps,
                                                                  block, out_f32, ldo, out_bf16, ldb, resid, ldr);
  return cudaGetLastError();
}

extern "C" int64_t g53d_attention_partial_floats(int nreq, int heads, int splits) {
  return (int64_t)nreq * splits * kBlock * heads * (2 + kHeadDim);
}

extern "C" cudaError_t g53d_attention(const float* q, int64_t ldq, int nreq, int heads, int kv_heads,
                                      const int64_t* start, const int64_t* lo, const uint64_t* bases, int layer,
                                      int ring, int window_left, float scale, int split_keys, int splits,
                                      float* partial, uint16_t* out_bf16, float* out_f32, int64_t ldo,
                                      cudaStream_t s) {
  if (nreq <= 0) return cudaSuccess;
  if (kv_heads <= 0 || heads % kv_heads != 0 || heads / kv_heads > 4 || splits <= 0 || split_keys <= 0)
    return cudaErrorInvalidValue;
  const int group = heads / kv_heads;
  attn_split_kernel<<<dim3(nreq * kv_heads, splits), group * 32, 0, s>>>(
      q, ldq, heads, kv_heads, start, lo, bases, layer, ring, window_left, scale, split_keys, splits, partial);
  cudaError_t e = cudaGetLastError();
  if (e != cudaSuccess) return e;
  attn_merge_kernel<<<nreq * kBlock * heads, kHeadDim, 0, s>>>(partial, heads, splits, out_bf16, out_f32, ldo);
  return cudaGetLastError();
}

extern "C" cudaError_t g53d_silu_mul(const float* gu, int64_t ldgu, int rows, int inter, uint16_t* out, int64_t ldo,
                                     cudaStream_t s) {
  if (rows <= 0) return cudaSuccess;
  silu_mul_kernel<<<grid_for((int64_t)rows * inter, 256), 256, 0, s>>>(gu, ldgu, rows, inter, out, ldo);
  return cudaGetLastError();
}

extern "C" cudaError_t g53d_block_embed(const uint16_t* anchor_rows, const uint16_t* mask_row, int nreq, int block,
                                        int n, float* h, cudaStream_t s) {
  if (nreq <= 0) return cudaSuccess;
  block_embed_kernel<<<grid_for((int64_t)nreq * block * n, 256), 256, 0, s>>>(anchor_rows, mask_row, nreq, block, n, h);
  return cudaGetLastError();
}

extern "C" cudaError_t g53d_gather_drafts(const uint16_t* src, int64_t lds, int nreq, int block, int n, uint16_t* dst,
                                          cudaStream_t s) {
  if (nreq <= 0 || block < 2) return cudaSuccess;
  gather_drafts_kernel<<<grid_for((int64_t)nreq * (block - 1) * n, 256), 256, 0, s>>>(src, lds, nreq, block, n, dst);
  return cudaGetLastError();
}

extern "C" int64_t g53d_topk16_workspace_bytes(int rows) {
  return (int64_t)rows * kTopkChunks * kTopK * 8;
}

extern "C" cudaError_t g53d_topk16(const float* logits, int64_t ld, int rows, int limit, void* workspace, float* vals,
                                   int32_t* ids, cudaStream_t s) {
  if (rows <= 0) return cudaSuccess;
  if (limit < kTopK) return cudaErrorInvalidValue;
  float* wv = (float*)workspace;
  int* wi = (int*)(wv + (int64_t)rows * kTopkChunks * kTopK);
  topk16_stage1<<<dim3(rows, kTopkChunks), 256, 0, s>>>(logits, ld, limit, wv, wi);
  cudaError_t e = cudaGetLastError();
  if (e != cudaSuccess) return e;
  topk16_stage2<<<rows, kTopkChunks, 0, s>>>(wv, wi, vals, ids);
  return cudaGetLastError();
}

extern "C" cudaError_t g53d_select(const float* hproj, const float* vals, const int32_t* ids, const int32_t* anchors,
                                   const uint16_t* pred, const uint16_t* succ, int rank, int nreq, int slots,
                                   const float* temperature, const float* uniforms, int32_t* tokens, int32_t* index,
                                   float* scores, float* q, float* conf, cudaStream_t s) {
  if (nreq <= 0 || slots <= 0) return cudaSuccess;
  if (rank <= 0 || rank > 1024) return cudaErrorInvalidValue;
  const int threads = (rank + 31) / 32 * 32;
  select_kernel<<<nreq, threads, 0, s>>>(hproj, vals, ids, anchors, pred, succ, rank, slots, temperature, uniforms,
                                         tokens, index, scores, q, conf);
  return cudaGetLastError();
}
