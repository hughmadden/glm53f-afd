// Sparse MLA kernels for GLM-5.3-Flash (absorbed form, FP8 528-byte latents).
//
// sparse_attn follows the structure of ds41rt's v41_sparse_attention.cu attend<>
// kernel (MIT; see crates/glm53f-dsa/PROVENANCE.md): a 64-token tile of
// selected latents is decoded once into padded BF16 shared memory and shared by
// 16-head groups; each warp scores 16 tokens x 16 heads with BF16 WMMA over the
// 512-wide latent; the online softmax runs in f32; the running output is
// rescaled in registers through a factor fragment loaded in the accumulator's
// own layout; P (BF16) x V accumulates in f32 with each warp owning 128 output
// channels. GLM-5.3-Flash differs from DeepSeek-V4.1 here: no sink, no
// window/compressed split, one list of <= 2,051 tokens per row, E4M3 codes with
// four f32 scales per record, and scale 256^-0.5.
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <mma.h>
#include <stdint.h>

#include "glm53f_dsa.h"
#include "glm53f_dsa_common.cuh"

namespace glm53f {
namespace {

using namespace nvcuda;

constexpr int kHeads = 64;
constexpr int kNope = 256;
constexpr int kKvRows = 512;  // kv_b_proj rows per head (256 key + 256 value)
constexpr int kTile = 64;     // tokens per tile
constexpr int kKvStride = 520;  // padded BF16 row of the decoded tile
constexpr int kProbStride = 80;
constexpr int kTileBytes = kTile * kKvStride * 2;  // 66,560
constexpr int kGroupBytes = 16 * kTile * 4 + 16 * 16 * 4 + 3 * 16 * 4;  // scores/probs, factors, max/sum/rescale

constexpr uint64_t smem_bytes(int groups) { return uint64_t(kTileBytes) + uint64_t(groups) * kGroupBytes + kTile * 8; }

// ---------------------------------------------------------------------------
// Latent write: optional RMSNorm, then E4M3 with a power-of-two scale per 128.
__global__ __launch_bounds__(128) void latent_write_kernel(const float* latent, const float* norm_w, float eps,
    const int32_t* row_pos, const int32_t* row_req, glm53f_dsa_cache_t cache) {
  __shared__ float red[4];
  const int row = blockIdx.x;
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int c0 = warp * 128 + lane * 4;
  float4 v = *reinterpret_cast<const float4*>(latent + int64_t(row) * kLatentDim + c0);
  if (norm_w) {
    float ss = v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
    ss = warp_sum(ss);
    if (lane == 0) red[warp] = ss;
    __syncthreads();
    const float total = ((red[0] + red[1]) + red[2]) + red[3];
    const float inv = 1.0f / sqrtf(total / float(kLatentDim) + eps);
    v.x = norm_w[c0 + 0] * (v.x * inv);
    v.y = norm_w[c0 + 1] * (v.y * inv);
    v.z = norm_w[c0 + 2] * (v.z * inv);
    v.w = norm_w[c0 + 3] * (v.w * inv);
  }
  const float amax = warp_max(fmaxf(fmaxf(fabsf(v.x), fabsf(v.y)), fmaxf(fabsf(v.z), fabsf(v.w))));
  float scale, inv;
  pow2_scale(amax, scale, inv);
  const int64_t off = latent_offset(cache, row_req[row], row_pos[row]);
  if (off < 0) return;
  uint8_t* rec = cache.base + off;
  const uint32_t lo = fp8x2_e4m3(v.x * inv, v.y * inv), hi = fp8x2_e4m3(v.z * inv, v.w * inv);
  *reinterpret_cast<uint32_t*>(rec + c0) = lo | (hi << 16);
  if (lane == 0) *reinterpret_cast<float*>(rec + kLatentDim + 4 * warp) = scale;
}

// ---------------------------------------------------------------------------
// Absorb: q_abs[r][h][l] = sum_i q[r][h][i] kv_b[h*512 + i][l], sequential in i.
constexpr int kAbsorbRows = 8;
__global__ __launch_bounds__(256) void absorb_kernel(const float* q, const __nv_bfloat16* kv_b, int rows,
    __nv_bfloat16* q_abs_bf16, float* q_abs_f32) {
  __shared__ float qs[kAbsorbRows][kNope];
  const int h = blockIdx.x, r0 = blockIdx.y * kAbsorbRows;
  const int nr = min(kAbsorbRows, rows - r0);
  for (int i = threadIdx.x; i < kAbsorbRows * kNope; i += 256) {
    const int r = i / kNope, c = i % kNope;
    qs[r][c] = r < nr ? q[(int64_t(r0 + r) * kHeads + h) * kNope + c] : 0.f;
  }
  __syncthreads();
  const int l = 2 * threadIdx.x;
  float acc[kAbsorbRows][2];
#pragma unroll
  for (int r = 0; r < kAbsorbRows; ++r) acc[r][0] = acc[r][1] = 0.f;
  const __nv_bfloat162* w = reinterpret_cast<const __nv_bfloat162*>(kv_b + int64_t(h) * kKvRows * kLatentDim + l);
  for (int i = 0; i < kNope; ++i) {
    const float2 wv = __bfloat1622float2(w[int64_t(i) * (kLatentDim / 2)]);
#pragma unroll
    for (int r = 0; r < kAbsorbRows; ++r) {
      acc[r][0] = __fadd_rn(acc[r][0], __fmul_rn(qs[r][i], wv.x));
      acc[r][1] = __fadd_rn(acc[r][1], __fmul_rn(qs[r][i], wv.y));
    }
  }
  for (int r = 0; r < nr; ++r) {
    const int64_t o = (int64_t(r0 + r) * kHeads + h) * kLatentDim + l;
    if (q_abs_bf16)
      *reinterpret_cast<__nv_bfloat162*>(q_abs_bf16 + o) = __floats2bfloat162_rn(acc[r][0], acc[r][1]);
    if (q_abs_f32) *reinterpret_cast<float2*>(q_abs_f32 + o) = make_float2(acc[r][0], acc[r][1]);
  }
}

// ---------------------------------------------------------------------------
// Un-absorb: o[r][h][v] = sum_l kv_b[h*512 + 256 + v][l] o_lat[r][h][l].
__global__ __launch_bounds__(256) void unabsorb_kernel(const float* o_lat, const __nv_bfloat16* kv_b, int rows,
    float* o) {
  __shared__ float ol[kAbsorbRows][kLatentDim];
  const int h = blockIdx.x, r0 = blockIdx.y * kAbsorbRows;
  const int nr = min(kAbsorbRows, rows - r0);
  for (int i = threadIdx.x; i < kAbsorbRows * kLatentDim; i += 256) {
    const int r = i / kLatentDim, c = i % kLatentDim;
    ol[r][c] = r < nr ? o_lat[(int64_t(r0 + r) * kHeads + h) * kLatentDim + c] : 0.f;
  }
  __syncthreads();
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  for (int v = warp; v < kNope; v += 8) {
    const __nv_bfloat162* wr =
        reinterpret_cast<const __nv_bfloat162*>(kv_b + (int64_t(h) * kKvRows + kNope + v) * kLatentDim);
    float acc[kAbsorbRows];
#pragma unroll
    for (int r = 0; r < kAbsorbRows; ++r) acc[r] = 0.f;
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      const int l2 = lane + 32 * j;  // bf16x2 index
      const float2 wv = __bfloat1622float2(wr[l2]);
#pragma unroll
      for (int r = 0; r < kAbsorbRows; ++r) acc[r] += wv.x * ol[r][2 * l2] + wv.y * ol[r][2 * l2 + 1];
    }
#pragma unroll
    for (int r = 0; r < kAbsorbRows; ++r) {
      const float s = warp_sum(acc[r]);
      if (lane == 0 && r < nr) o[(int64_t(r0 + r) * kHeads + h) * kNope + v] = s;
    }
  }
}

// ---------------------------------------------------------------------------
// Sparse attention.

struct GroupSmem {
  float* scores;    // [16][64] f32, aliased by probabilities [16][80] bf16
  float* factors;   // [16][16]
  float* maximum;   // [16]
  float* sum;       // [16]
  float* rescale;   // [16]
};

__device__ __forceinline__ GroupSmem group_smem(unsigned char* smem, int groups, int sub) {
  unsigned char* p = smem + kTileBytes + sub * kGroupBytes;
  GroupSmem g;
  g.scores = reinterpret_cast<float*>(p);
  g.factors = reinterpret_cast<float*>(p + 16 * kTile * 4);
  g.maximum = reinterpret_cast<float*>(p + 16 * kTile * 4 + 16 * 16 * 4);
  g.sum = g.maximum + 16;
  g.rescale = g.sum + 16;
  return g;
}

// Decode 4 E4M3 codes (one u32) times a scale to 4 BF16 values (8 bytes).
__device__ __forceinline__ uint2 decode4(uint32_t codes, float scale) {
  const __half2_raw a = __nv_cvt_fp8x2_to_halfraw2(static_cast<__nv_fp8x2_storage_t>(codes & 0xFFFFu), __NV_E4M3);
  const __half2_raw b = __nv_cvt_fp8x2_to_halfraw2(static_cast<__nv_fp8x2_storage_t>(codes >> 16), __NV_E4M3);
  const float2 fa = __half22float2(*reinterpret_cast<const __half2*>(&a));
  const float2 fb = __half22float2(*reinterpret_cast<const __half2*>(&b));
  const __nv_bfloat162 x = __floats2bfloat162_rn(fa.x * scale, fa.y * scale);
  const __nv_bfloat162 y = __floats2bfloat162_rn(fb.x * scale, fb.y * scale);
  return make_uint2(*reinterpret_cast<const uint32_t*>(&x), *reinterpret_cast<const uint32_t*>(&y));
}

template <int Groups>
__global__ __launch_bounds__(128 * Groups, 1) void sparse_attn_kernel(const __nv_bfloat16* q_abs,
    const int32_t* tokens, int token_stride, const int32_t* counts, const int32_t* row_req, float scale,
    glm53f_dsa_cache_t cache, int splits, float* partial, float* o_lat, float* lse) {
  extern __shared__ __align__(128) unsigned char smem[];
  auto* kv = reinterpret_cast<__nv_bfloat16*>(smem);
  auto* refs = reinterpret_cast<int64_t*>(smem + kTileBytes + Groups * kGroupBytes);
  const int row = blockIdx.x;
  const int sub = Groups == 1 ? 0 : threadIdx.x / 128;
  const int group = blockIdx.y * Groups + sub;  // 16-head group
  const int split = blockIdx.z;
  const int tid = threadIdx.x % 128, gtid = threadIdx.x;
  const int warp = tid / 32, lane = tid % 32;
  const GroupSmem gs = group_smem(smem, Groups, sub);
  const int req = row_req[row];
  const int n_tok = counts[2 * row + 1];
  const int tiles = (n_tok + kTile - 1) / kTile;
  const int per = (tiles + splits - 1) / splits;
  const int t_begin = split * per, t_end = min(tiles, t_begin + per);
  const int head0 = group * 16;
  const __nv_bfloat16* qrow = q_abs + (int64_t(row) * kHeads + head0) * kLatentDim;

  if (tid < 16) {
    gs.maximum[tid] = -1e30f;
    gs.sum[tid] = 0.f;
  }
  __syncthreads();
  wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[8];
#pragma unroll
  for (int t = 0; t < 8; ++t) wmma::fill_fragment(acc[t], 0.f);

  for (int tile = t_begin; tile < t_end; ++tile) {
    const int first = tile * kTile;
    if (gtid < kTile) {
      const int j = first + gtid;
      refs[gtid] = j < n_tok ? latent_offset(cache, req, tokens[int64_t(row) * token_stride + j]) : -1;
    }
    __syncthreads();
    // Decode the tile: a warp takes whole records, a lane 16 codes (one 16-byte
    // load) and its group's scale; unrolled so several records are in flight.
    {
      const int wg = gtid >> 5, nw = 4 * Groups;
#pragma unroll 4
      for (int key = wg; key < kTile; key += nw) {
        const int64_t ref = refs[key];
        uint4 codes = make_uint4(0, 0, 0, 0);
        float sc = 0.f;
        if (ref >= 0) {
          const uint8_t* rec = cache.base + ref;
          codes = __ldg(reinterpret_cast<const uint4*>(rec) + lane);
          sc = __ldg(reinterpret_cast<const float*>(rec + kLatentDim) + (lane >> 3));
        }
        const uint2 a = decode4(codes.x, sc), b = decode4(codes.y, sc), c = decode4(codes.z, sc), d = decode4(codes.w, sc);
        uint4* dst = reinterpret_cast<uint4*>(kv + key * kKvStride + 16 * lane);
        dst[0] = make_uint4(a.x, a.y, b.x, b.y);
        dst[1] = make_uint4(c.x, c.y, d.x, d.y);
      }
    }
    __syncthreads();
    // Scores: warp w -> tokens [16 w, 16 w + 16) x 16 heads.
    {
      wmma::fragment<wmma::accumulator, 16, 16, 16, float> s;
      wmma::fill_fragment(s, 0.f);
#pragma unroll 4
      for (int k = 0; k < kLatentDim; k += 16) {
        wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> a;
        wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::col_major> b;
        wmma::load_matrix_sync(a, qrow + k, kLatentDim);
        wmma::load_matrix_sync(b, kv + warp * 16 * kKvStride + k, kKvStride);
        wmma::mma_sync(s, a, b, s);
      }
      wmma::store_matrix_sync(gs.scores + warp * 16, s, kTile, wmma::mem_row_major);
    }
    __syncthreads();
    // Online softmax: warp w handles heads w, w+4, w+8, w+12; lane handles tokens lane, lane+32.
    const bool va = refs[lane] >= 0, vb = refs[lane + 32] >= 0;
    __nv_bfloat16 pa16[4], pb16[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const int h = warp + 4 * i;
      const float a = va ? gs.scores[h * kTile + lane] * scale : -INFINITY;
      const float b = vb ? gs.scores[h * kTile + lane + 32] * scale : -INFINITY;
      const float prev = gs.maximum[h];
      const float m = fmaxf(prev, warp_max(fmaxf(a, b)));
      const float pa = va ? __expf(a - m) : 0.f, pb = vb ? __expf(b - m) : 0.f;
      const float total = warp_sum(pa + pb);
      pa16[i] = __float2bfloat16_rn(pa);
      pb16[i] = __float2bfloat16_rn(pb);
      if (lane == 0) {
        const float r = __expf(prev - m);
        gs.maximum[h] = m;
        gs.rescale[h] = r;
        gs.sum[h] = gs.sum[h] * r + total;
      }
    }
    __syncthreads();
    // Rescale the running output (the factor fragment shares the accumulator layout).
    for (int i = tid; i < 256; i += 128) gs.factors[i] = gs.rescale[i >> 4];
    __syncthreads();
    {
      wmma::fragment<wmma::accumulator, 16, 16, 16, float> f;
      wmma::load_matrix_sync(f, gs.factors, 16, wmma::mem_row_major);
#pragma unroll
      for (int t = 0; t < 8; ++t)
#pragma unroll
        for (int e = 0; e < acc[t].num_elements; ++e) acc[t].x[e] *= f.x[e];
    }
    // Probabilities (BF16) over the score buffer.
    auto* prob = reinterpret_cast<__nv_bfloat16*>(gs.scores);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const int h = warp + 4 * i;
      prob[h * kProbStride + lane] = pa16[i];
      prob[h * kProbStride + lane + 32] = pb16[i];
    }
    __syncthreads();
    // P x V: warp w owns latent channels [128 w, 128 w + 128).
#pragma unroll
    for (int k = 0; k < kTile; k += 16) {
      wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> p;
      wmma::load_matrix_sync(p, prob + k, kProbStride);
#pragma unroll
      for (int t = 0; t < 8; ++t) {
        wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::row_major> v;
        wmma::load_matrix_sync(v, kv + k * kKvStride + warp * 128 + t * 16, kKvStride);
        wmma::mma_sync(acc[t], p, v, acc[t]);
      }
    }
    __syncthreads();
  }

  // Epilogue. Non-split: normalize in registers and store. Split: store the
  // unnormalized partial and (max, sum) for the merge.
  if (splits == 1) {
    for (int i = tid; i < 256; i += 128) {
      const float s = gs.sum[i >> 4];
      gs.factors[i] = s > 0.f ? 1.f / s : 0.f;
    }
    __syncthreads();
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> f;
    wmma::load_matrix_sync(f, gs.factors, 16, wmma::mem_row_major);
    float* dst = o_lat + (int64_t(row) * kHeads + head0) * kLatentDim + warp * 128;
#pragma unroll
    for (int t = 0; t < 8; ++t) {
#pragma unroll
      for (int e = 0; e < acc[t].num_elements; ++e) acc[t].x[e] *= f.x[e];
      wmma::store_matrix_sync(dst + t * 16, acc[t], kLatentDim, wmma::mem_row_major);
    }
    if (tid < 16 && lse) {
      const float s = gs.sum[tid];
      lse[int64_t(row) * kHeads + head0 + tid] = s > 0.f ? gs.maximum[tid] + logf(s) : -INFINITY;
    }
  } else {
    float* dst = partial + ((int64_t(row) * splits + split) * kHeads + head0) * kLatentDim + warp * 128;
#pragma unroll
    for (int t = 0; t < 8; ++t) wmma::store_matrix_sync(dst + t * 16, acc[t], kLatentDim, wmma::mem_row_major);
    if (tid < 16) {
      float* ms = partial + int64_t(gridDim.x) * splits * kHeads * kLatentDim +
                  ((int64_t(row) * splits + split) * kHeads + head0 + tid) * 2;
      ms[0] = gs.maximum[tid];
      ms[1] = gs.sum[tid];
    }
  }
}

__global__ __launch_bounds__(128) void attn_merge_kernel(const float* partial, int rows, int splits, float* o_lat,
                                                         float* lse) {
  const int row = blockIdx.x, h = blockIdx.y;
  const float* ms = partial + int64_t(rows) * splits * kHeads * kLatentDim;
  float m = -1e30f;
  for (int s = 0; s < splits; ++s) m = fmaxf(m, ms[((int64_t(row) * splits + s) * kHeads + h) * 2]);
  float sum = 0.f;
  float acc[4] = {0.f, 0.f, 0.f, 0.f};
  for (int s = 0; s < splits; ++s) {
    const int64_t i = (int64_t(row) * splits + s) * kHeads + h;
    const float part_sum = ms[i * 2 + 1];
    if (!(part_sum > 0.f)) continue;
    const float f = __expf(ms[i * 2] - m);
    sum += part_sum * f;
    const float4 v = *reinterpret_cast<const float4*>(partial + i * kLatentDim + threadIdx.x * 4);
    acc[0] += v.x * f;
    acc[1] += v.y * f;
    acc[2] += v.z * f;
    acc[3] += v.w * f;
  }
  const float inv = sum > 0.f ? 1.f / sum : 0.f;
  *reinterpret_cast<float4*>(o_lat + (int64_t(row) * kHeads + h) * kLatentDim + threadIdx.x * 4) =
      make_float4(acc[0] * inv, acc[1] * inv, acc[2] * inv, acc[3] * inv);
  if (threadIdx.x == 0 && lse) lse[int64_t(row) * kHeads + h] = sum > 0.f ? m + logf(sum) : -INFINITY;
}

}  // namespace
}  // namespace glm53f

using namespace glm53f;

extern "C" int32_t glm53f_dsa_init(void) {
  cudaError_t st = cudaFuncSetAttribute(sparse_attn_kernel<1>, cudaFuncAttributeMaxDynamicSharedMemorySize, int(smem_bytes(1)));
  if (st != cudaSuccess) return st;
  st = cudaFuncSetAttribute(sparse_attn_kernel<2>, cudaFuncAttributeMaxDynamicSharedMemorySize, int(smem_bytes(2)));
  if (st != cudaSuccess) return st;
  return cudaFuncSetAttribute(sparse_attn_kernel<4>, cudaFuncAttributeMaxDynamicSharedMemorySize, int(smem_bytes(4)));
}

extern "C" int32_t glm53f_dsa_mla_latent_write(const float* latent, const float* norm_w, float eps,
    const int32_t* row_pos, const int32_t* row_req, int32_t rows, glm53f_dsa_cache_t cache, void* stream) {
  if (rows < 0 || !valid_cache(cache)) return cudaErrorInvalidValue;
  if (rows == 0) return cudaSuccess;
  if (!latent || !row_pos || !row_req || (reinterpret_cast<uintptr_t>(latent) % 16)) return cudaErrorInvalidValue;
  latent_write_kernel<<<rows, 128, 0, static_cast<cudaStream_t>(stream)>>>(latent, norm_w, eps, row_pos, row_req,
                                                                           cache);
  return cudaGetLastError();
}

extern "C" int32_t glm53f_dsa_mla_absorb_q(const float* q, const uint16_t* kv_b, int32_t rows, uint16_t* q_abs_bf16,
                                           float* q_abs_f32, void* stream) {
  if (rows < 0 || !q || !kv_b || (!q_abs_bf16 && !q_abs_f32)) return cudaErrorInvalidValue;
  if (rows == 0) return cudaSuccess;
  absorb_kernel<<<dim3(kHeads, (rows + kAbsorbRows - 1) / kAbsorbRows), 256, 0, static_cast<cudaStream_t>(stream)>>>(
      q, reinterpret_cast<const __nv_bfloat16*>(kv_b), rows, reinterpret_cast<__nv_bfloat16*>(q_abs_bf16), q_abs_f32);
  return cudaGetLastError();
}

extern "C" int32_t glm53f_dsa_mla_unabsorb_v(const float* o_lat, const uint16_t* kv_b, int32_t rows, float* o,
                                             void* stream) {
  if (rows < 0 || !o_lat || !kv_b || !o) return cudaErrorInvalidValue;
  if (rows == 0) return cudaSuccess;
  unabsorb_kernel<<<dim3(kHeads, (rows + kAbsorbRows - 1) / kAbsorbRows), 256, 0, static_cast<cudaStream_t>(stream)>>>(
      o_lat, reinterpret_cast<const __nv_bfloat16*>(kv_b), rows, o);
  return cudaGetLastError();
}

extern "C" uint64_t glm53f_dsa_mla_workspace_bytes(int32_t rows, int32_t splits) {
  if (rows < 1 || splits <= 1) return 0;
  return uint64_t(rows) * splits * kHeads * (kLatentDim + 2) * 4;
}

extern "C" int32_t glm53f_dsa_mla_sparse_attn(const uint16_t* q_abs, const int32_t* tokens, int32_t token_stride,
    const int32_t* counts, const int32_t* row_req, int32_t rows, float scale, glm53f_dsa_cache_t cache,
    int32_t splits, int32_t head_groups, void* workspace, uint64_t workspace_bytes, float* o_lat, float* lse,
    void* stream) {
  if (rows < 0 || splits < 1 || splits > 64 || token_stride < 1 || !valid_cache(cache) ||
      (head_groups != 1 && head_groups != 2 && head_groups != 4))
    return cudaErrorInvalidValue;
  if (rows == 0) return cudaSuccess;
  if (!q_abs || !tokens || !counts || !row_req || !o_lat) return cudaErrorInvalidValue;
  if (splits > 1 && (!workspace || workspace_bytes < glm53f_dsa_mla_workspace_bytes(rows, splits)))
    return cudaErrorInvalidValue;
  auto s = static_cast<cudaStream_t>(stream);
  float* partial = static_cast<float*>(workspace);
  const dim3 grid(rows, 4 / head_groups, splits);
  const auto* q = reinterpret_cast<const __nv_bfloat16*>(q_abs);
  switch (head_groups) {
    case 1:
      sparse_attn_kernel<1><<<grid, 128, smem_bytes(1), s>>>(q, tokens, token_stride, counts, row_req, scale, cache,
                                                             splits, partial, o_lat, lse);
      break;
    case 2:
      sparse_attn_kernel<2><<<grid, 256, smem_bytes(2), s>>>(q, tokens, token_stride, counts, row_req, scale, cache,
                                                             splits, partial, o_lat, lse);
      break;
    default:
      sparse_attn_kernel<4><<<grid, 512, smem_bytes(4), s>>>(q, tokens, token_stride, counts, row_req, scale, cache,
                                                             splits, partial, o_lat, lse);
  }
  cudaError_t st = cudaGetLastError();
  if (st != cudaSuccess || splits == 1) return st;
  attn_merge_kernel<<<dim3(rows, kHeads), 128, 0, s>>>(partial, rows, splits, o_lat, lse);
  return cudaGetLastError();
}
