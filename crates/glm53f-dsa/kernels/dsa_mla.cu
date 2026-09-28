// Sparse MLA kernels for GLM-5.3-Flash (absorbed form, FP8 528-byte latents).
//
// glm53f_dsa_mla_sparse_attn runs attn_v2_kernel (described at its definition):
// F16 tensor-core products straight from the E4M3 codes, tiles of 32 tokens
// shared by up to four 16-head groups, splits merged by attn_merge_v2_kernel.
//
// glm53f_dsa_mla_sparse_attn_v1 (sparse_attn_kernel, kept for comparison)
// follows the structure of ds41rt's v41_sparse_attention.cu attend<> kernel
// (MIT; see crates/glm53f-dsa/PROVENANCE.md): a 64-token tile of selected
// latents is decoded once into padded BF16 shared memory and shared by 16-head
// groups; each warp scores 16 tokens x 16 heads with BF16 WMMA over the
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
// Absorb for up to R rows per block. Block (head, row group, column half): 128
// threads x 2 columns. Rows of W^K are loaded eight at a time before their
// multiply-adds, which stay in the CPU reference's sequential order (bit-exact).
template <int R>
__global__ __launch_bounds__(128) void absorb_kernel(const float* q, const __nv_bfloat16* kv_b, int rows,
    __nv_bfloat16* q_abs_bf16, float* q_abs_f32) {
  __shared__ float qs[R][kNope];
  const int h = blockIdx.x, r0 = blockIdx.y * R;
  const int nr = min(R, rows - r0);
  for (int i = threadIdx.x; i < R * kNope; i += blockDim.x) {
    const int r = i / kNope, c = i % kNope;
    qs[r][c] = r < nr ? q[(int64_t(r0 + r) * kHeads + h) * kNope + c] : 0.f;
  }
  __syncthreads();
  const int l = 256 * blockIdx.z + 2 * threadIdx.x;
  float acc[R][2];
#pragma unroll
  for (int r = 0; r < R; ++r) acc[r][0] = acc[r][1] = 0.f;
  const __nv_bfloat162* w = reinterpret_cast<const __nv_bfloat162*>(kv_b + int64_t(h) * kKvRows * kLatentDim + l);
  for (int i0 = 0; i0 < kNope; i0 += 8) {
    float2 wv[8];
#pragma unroll
    for (int u = 0; u < 8; ++u) wv[u] = __bfloat1622float2(w[int64_t(i0 + u) * (kLatentDim / 2)]);
#pragma unroll
    for (int u = 0; u < 8; ++u) {
#pragma unroll
      for (int r = 0; r < R; ++r) {
        acc[r][0] = __fadd_rn(acc[r][0], __fmul_rn(qs[r][i0 + u], wv[u].x));
        acc[r][1] = __fadd_rn(acc[r][1], __fmul_rn(qs[r][i0 + u], wv[u].y));
      }
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
// Block (head, row group, value quarter): 8 warps x 8 value rows, R rows of
// o_lat in shared memory; each warp loads two value rows before reducing them.
template <int R>
__global__ __launch_bounds__(256) void unabsorb_kernel(const float* o_lat, const __nv_bfloat16* kv_b, int rows,
    float* o) {
  __shared__ float ol[R][kLatentDim];
  const int h = blockIdx.x, r0 = blockIdx.y * R;
  const int nr = min(R, rows - r0);
  for (int i = threadIdx.x; i < R * kLatentDim; i += 256) {
    const int r = i / kLatentDim, c = i % kLatentDim;
    ol[r][c] = r < nr ? o_lat[(int64_t(r0 + r) * kHeads + h) * kLatentDim + c] : 0.f;
  }
  __syncthreads();
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int v0 = 64 * blockIdx.z + 8 * warp;
  for (int vp = 0; vp < 8; vp += 2) {
    float2 wv[2][8];
#pragma unroll
    for (int e = 0; e < 2; ++e) {
      const __nv_bfloat162* wr =
          reinterpret_cast<const __nv_bfloat162*>(kv_b + (int64_t(h) * kKvRows + kNope + v0 + vp + e) * kLatentDim);
#pragma unroll
      for (int j = 0; j < 8; ++j) wv[e][j] = __bfloat1622float2(wr[lane + 32 * j]);
    }
#pragma unroll
    for (int e = 0; e < 2; ++e) {
      float acc[R];
#pragma unroll
      for (int r = 0; r < R; ++r) acc[r] = 0.f;
#pragma unroll
      for (int j = 0; j < 8; ++j) {
        const int l2 = lane + 32 * j;  // bf16x2 index
#pragma unroll
        for (int r = 0; r < R; ++r) acc[r] += wv[e][j].x * ol[r][2 * l2] + wv[e][j].y * ol[r][2 * l2 + 1];
      }
#pragma unroll
      for (int r = 0; r < R; ++r) {
        const float s = warp_sum(acc[r]);
        if (lane == 0 && r < nr) o[(int64_t(r0 + r) * kHeads + h) * kNope + v0 + vp + e] = s;
      }
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

// ---------------------------------------------------------------------------
// Sparse attention, v2 (the default since the decode-latency pass).
//
// Per block: one row, G 16-head groups (4 warps each), one token split; tiles
// of 32 tokens. The FP8 records of tile i+1 are loaded into registers while
// tile i is computed. At the top of each iteration the codes are converted to
// F16 (cvt.rn.f16x2.e4m3x2, exact) into one tile shared by the head groups,
// its 16-byte units XOR-swizzled so the conversion stores and the ldmatrix
// reads avoid bank conflicts; each token's four scales stay beside it in f32.
// Splits get consecutive runs of tiles of equal size (to within one tile).
// Scores: warp w of a head group owns channel group w. It multiplies the
// group's 16 query rows (F16 with a per-head power-of-two scale, so the BF16
// query converts exactly; held in registers for the whole block) by the codes
// of its 128 channels (mma.m16n8k16, B fragments by ldmatrix) and applies each
// token's group-w scale in f32; the four partial score tiles are summed through
// shared memory. Every warp of the group then runs the same online softmax in
// registers, forms P x (the tokens' group-w scales) directly as F16 A
// fragments, and accumulates them against the codes of its 128 output channels
// (ldmatrix.trans). Three barriers per tile; the block resolves its record
// offsets once. Compared with a BF16 tile of scaled values this removes the
// E4M3 -> f32 -> BF16 conversion (sm_89 has no direct E4M3 -> BF16), and the
// F16 operands carry more precision than BF16.
namespace v2 {
constexpr int kT = 32;
constexpr int kStride = 520;  // F16 elements per tile row: 16-byte rows land on distinct banks
constexpr int kPartStride = 40;
constexpr int kTileBytes = kT * kStride * 2;          // 33,280
constexpr int kScaleBytes = kT * 4 * 4;               // 512: f32 scale per token and channel group
constexpr int kPartBytes = 4 * 16 * kPartStride * 4;  // 10,240 per 16-head group
constexpr int kAmaxBytes = 4 * 16 * 4;                // 256 per 16-head group: per-warp head maxima
__host__ __device__ constexpr uint64_t smem_bytes(int groups, int max_tiles) {
  return uint64_t(kTileBytes) + kScaleBytes + uint64_t(groups) * (kPartBytes + kAmaxBytes) +
         (uint64_t(max_tiles) * kT * 4 + 15) / 16 * 16;
}
}  // namespace v2

__device__ __forceinline__ void mma_f16_16816(float (&d)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

__device__ __forceinline__ void ldsm_x4(uint32_t (&r)[4], const void* p) {
  const uint32_t a = static_cast<uint32_t>(__cvta_generic_to_shared(p));
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
               : "r"(a));
}

__device__ __forceinline__ void ldsm_x4_trans(uint32_t (&r)[4], const void* p) {
  const uint32_t a = static_cast<uint32_t>(__cvta_generic_to_shared(p));
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
               : "r"(a));
}

__device__ __forceinline__ uint32_t pack_f16(float lo, float hi) {
  const __half2 v = __floats2half2_rn(lo, hi);
  return *reinterpret_cast<const uint32_t*>(&v);
}

// Two BF16 values (one 32-bit word) times 2^e as an F16 pair.
__device__ __forceinline__ uint32_t bf16x2_to_f16x2(uint32_t v, float up) {
  const float lo = __uint_as_float(v << 16), hi = __uint_as_float(v & 0xFFFF0000u);
  return pack_f16(lo * up, hi * up);
}

// Swizzled position of 16-byte unit u (8 F16 values) within a tile row.
__device__ __forceinline__ int swz(int u) { return u ^ ((u >> 3) & 7); }

__device__ __forceinline__ uint32_t e4m3x2_to_f16x2(uint32_t two) {
  const __half2_raw h = __nv_cvt_fp8x2_to_halfraw2(static_cast<__nv_fp8x2_storage_t>(two & 0xFFFFu), __NV_E4M3);
  return uint32_t(h.x) | (uint32_t(h.y) << 16);
}

template <int G>
__global__ __launch_bounds__(128 * G, G == 1 ? 2 : 1) void attn_v2_kernel(const __nv_bfloat16* __restrict__ q_abs,
    const int32_t* __restrict__ tokens, int token_stride, const int32_t* __restrict__ counts,
    const int32_t* __restrict__ row_req, float scale, glm53f_dsa_cache_t cache, int splits,
    float* __restrict__ partial, float* __restrict__ o_lat, float* __restrict__ lse) {
  constexpr int kT = v2::kT, kStride = v2::kStride, kPartStride = v2::kPartStride;
  extern __shared__ __align__(128) unsigned char smem[];
  __half* tile = reinterpret_cast<__half*>(smem);
  float* tscale = reinterpret_cast<float*>(smem + v2::kTileBytes);  // [kT][4]; -1 marks a missing record
  float* part = reinterpret_cast<float*>(smem + v2::kTileBytes + v2::kScaleBytes);
  float* amax_x = reinterpret_cast<float*>(smem + v2::kTileBytes + v2::kScaleBytes + G * v2::kPartBytes);
  uint32_t* refs =
      reinterpret_cast<uint32_t*>(smem + v2::kTileBytes + v2::kScaleBytes + G * (v2::kPartBytes + v2::kAmaxBytes));

  const int row = blockIdx.x, split = blockIdx.z;
  const int wi = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
  const int hg = wi >> 2, w = wi & 3;
  const int head0 = (blockIdx.y * G + hg) * 16;
  const int req = row_req[row];
  const int n_tok = min(counts[2 * row + 1], token_stride);
  const int tiles = (n_tok + kT - 1) / kT;
  // Tiles spread evenly over the splits (each gets floor or ceil of tiles / splits).
  const int t_begin = int(int64_t(split) * tiles / splits);
  const int my_tiles = int(int64_t(split + 1) * tiles / splits) - t_begin;

  // Query (BF16) first: nothing depends on it until the first products.
  uint32_t qraw[8][4];
  {
    const __nv_bfloat16* qh = q_abs + (int64_t(row) * kHeads + head0) * kLatentDim + 128 * w + 2 * t;
#pragma unroll
    for (int kk = 0; kk < 8; ++kk) {
      qraw[kk][0] = __ldg(reinterpret_cast<const uint32_t*>(qh + g * kLatentDim + 16 * kk));
      qraw[kk][1] = __ldg(reinterpret_cast<const uint32_t*>(qh + (g + 8) * kLatentDim + 16 * kk));
      qraw[kk][2] = __ldg(reinterpret_cast<const uint32_t*>(qh + g * kLatentDim + 16 * kk + 8));
      qraw[kk][3] = __ldg(reinterpret_cast<const uint32_t*>(qh + (g + 8) * kLatentDim + 16 * kk + 8));
    }
  }
  // Decode assignment: one token slot and 128 / G codes of one scale group per thread.
  constexpr int kChunks = 8 / G;
  const int ctok = threadIdx.x / (4 * G);
  const int ccode = (threadIdx.x % (4 * G)) * (128 / G);
  const int32_t* trow = tokens + int64_t(row) * token_stride + t_begin * kT;
  auto resolve = [&](int i) -> uint32_t {  // record offset of the block's i-th token, 16-byte units
    uint32_t r = 0xFFFFFFFFu;
    if (i < my_tiles * kT && t_begin * kT + i < n_tok) {
      const int64_t off = latent_offset(cache, req, __ldg(trow + i));
      if (off >= 0) r = uint32_t(off >> 4);
    }
    return r;
  };
  // Offsets of tiles 1.. go to shared memory; each thread resolves its own
  // tile-0 token directly, so the first records are requested without a barrier.
  for (int i = kT + threadIdx.x; i < my_tiles * kT; i += blockDim.x) refs[i] = resolve(i);
  const uint32_t ref0 = resolve(ctok);
  uint4 raw[kChunks];
  float rscale = 0.f;
  auto fetch_ref = [&](uint32_t r) {
    if (r != 0xFFFFFFFFu) {
      const uint8_t* rec = cache.base + (int64_t(r) << 4);
#pragma unroll
      for (int c = 0; c < kChunks; ++c) raw[c] = __ldg(reinterpret_cast<const uint4*>(rec + ccode) + c);
      rscale = __ldg(reinterpret_cast<const float*>(rec + kLatentDim) + (ccode >> 7));
    } else {
#pragma unroll
      for (int c = 0; c < kChunks; ++c) raw[c] = make_uint4(0, 0, 0, 0);
      rscale = -1.f;
    }
  };
  if (my_tiles > 0) fetch_ref(ref0);

  // Per-head power of two that puts the head's largest |q| in [2^14, 2^15): the
  // BF16 values then convert to F16 exactly; 2^-e goes into the softmax scale.
  {
    float m0 = 0.f, m1 = 0.f;
#pragma unroll
    for (int kk = 0; kk < 8; ++kk) {
#pragma unroll
      for (int r = 0; r < 4; r += 2) {
        m0 = fmaxf(m0, fmaxf(fabsf(__uint_as_float(qraw[kk][r] << 16)), fabsf(__uint_as_float(qraw[kk][r] & 0xFFFF0000u))));
        m1 = fmaxf(m1, fmaxf(fabsf(__uint_as_float(qraw[kk][r + 1] << 16)),
                             fabsf(__uint_as_float(qraw[kk][r + 1] & 0xFFFF0000u))));
      }
    }
    m0 = fmaxf(m0, __shfl_xor_sync(0xffffffffu, m0, 1));
    m0 = fmaxf(m0, __shfl_xor_sync(0xffffffffu, m0, 2));
    m1 = fmaxf(m1, __shfl_xor_sync(0xffffffffu, m1, 1));
    m1 = fmaxf(m1, __shfl_xor_sync(0xffffffffu, m1, 2));
    if (t == 0) {
      amax_x[(hg * 4 + w) * 16 + g] = m0;
      amax_x[(hg * 4 + w) * 16 + g + 8] = m1;
    }
  }
  __syncthreads();  // head maxima and the refs of tiles 1.. are written
  float up[2], sm_scale[2];
#pragma unroll
  for (int h = 0; h < 2; ++h) {
    const int head = g + 8 * h;
    float m = 0.f;
#pragma unroll
    for (int q = 0; q < 4; ++q) m = fmaxf(m, amax_x[(hg * 4 + q) * 16 + head]);
    int e = 0;
    if (m > 0.f && isfinite(m)) {
      int E;
      frexpf(m, &E);
      e = max(-120, min(120, 15 - E));
    }
    up[h] = exp2_int(e);
    sm_scale[h] = scale * exp2_int(-e);
  }
  uint32_t qa[8][4];
#pragma unroll
  for (int kk = 0; kk < 8; ++kk) {
    qa[kk][0] = bf16x2_to_f16x2(qraw[kk][0], up[0]);
    qa[kk][1] = bf16x2_to_f16x2(qraw[kk][1], up[1]);
    qa[kk][2] = bf16x2_to_f16x2(qraw[kk][2], up[0]);
    qa[kk][3] = bf16x2_to_f16x2(qraw[kk][3], up[1]);
  }

  float acc[16][4];
#pragma unroll
  for (int j = 0; j < 16; ++j) acc[j][0] = acc[j][1] = acc[j][2] = acc[j][3] = 0.f;
  float m_run[2] = {-1e30f, -1e30f}, l_run[2] = {0.f, 0.f};
  const int lm = lane >> 3, lr = lane & 7;
  float* pw = part + (hg * 4 + w) * 16 * kPartStride;
  const float* pgrp = part + hg * 4 * 16 * kPartStride;

  for (int ti = 0; ti < my_tiles; ++ti) {
    {
      // 16-byte units are stored swizzled (swz): the 4 G threads of a token would
      // otherwise write 256 / G bytes apart, into the same bank group.
      __half* row_base = tile + ctok * kStride;
      const int u_first = ccode / 8;
#pragma unroll
      for (int c = 0; c < kChunks; ++c) {
        *reinterpret_cast<uint4*>(row_base + 8 * swz(u_first + 2 * c)) = make_uint4(e4m3x2_to_f16x2(raw[c].x), e4m3x2_to_f16x2(raw[c].x >> 16),
                                    e4m3x2_to_f16x2(raw[c].y), e4m3x2_to_f16x2(raw[c].y >> 16));
        *reinterpret_cast<uint4*>(row_base + 8 * swz(u_first + 2 * c + 1)) = make_uint4(e4m3x2_to_f16x2(raw[c].z), e4m3x2_to_f16x2(raw[c].z >> 16),
                                    e4m3x2_to_f16x2(raw[c].w), e4m3x2_to_f16x2(raw[c].w >> 16));
      }
      if ((ccode & 127) == 0) tscale[ctok * 4 + (ccode >> 7)] = rscale;
    }
    if (ti + 1 < my_tiles) fetch_ref(refs[(ti + 1) * kT + ctok]);
    __syncthreads();

    float sc[4][4];
#pragma unroll
    for (int j = 0; j < 4; ++j) sc[j][0] = sc[j][1] = sc[j][2] = sc[j][3] = 0.f;
#pragma unroll
    for (int kk = 0; kk < 8; ++kk) {
#pragma unroll
      for (int jp = 0; jp < 2; ++jp) {
        uint32_t b[4];
        ldsm_x4(b, tile + (16 * jp + (lm >> 1) * 8 + lr) * kStride + 8 * swz(16 * w + 2 * kk + (lm & 1)));
        mma_f16_16816(sc[2 * jp], qa[kk], b[0], b[1]);
        mma_f16_16816(sc[2 * jp + 1], qa[kk], b[2], b[3]);
      }
    }
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const float s0 = tscale[(8 * j + 2 * t) * 4 + w], s1 = tscale[(8 * j + 2 * t + 1) * 4 + w];
      const float x0 = s0 >= 0.f ? sc[j][0] * s0 : -INFINITY, x1 = s1 >= 0.f ? sc[j][1] * s1 : -INFINITY;
      const float x2 = s0 >= 0.f ? sc[j][2] * s0 : -INFINITY, x3 = s1 >= 0.f ? sc[j][3] * s1 : -INFINITY;
      *reinterpret_cast<float2*>(pw + g * kPartStride + 8 * j + 2 * t) = make_float2(x0, x1);
      *reinterpret_cast<float2*>(pw + (g + 8) * kPartStride + 8 * j + 2 * t) = make_float2(x2, x3);
    }
    __syncthreads();

    const int tok0 = (t_begin + ti) * kT;
    float mx[2] = {-INFINITY, -INFINITY};
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      float2 lo = make_float2(0.f, 0.f), hi = make_float2(0.f, 0.f);
#pragma unroll
      for (int q = 0; q < 4; ++q) {
        const float2 a = *reinterpret_cast<const float2*>(pgrp + (q * 16 + g) * kPartStride + 8 * j + 2 * t);
        const float2 b = *reinterpret_cast<const float2*>(pgrp + (q * 16 + g + 8) * kPartStride + 8 * j + 2 * t);
        lo.x += a.x;
        lo.y += a.y;
        hi.x += b.x;
        hi.y += b.y;
      }
      const int tok = tok0 + 8 * j + 2 * t;
      const bool v0 = tok < n_tok, v1 = tok + 1 < n_tok;
      sc[j][0] = v0 ? lo.x * sm_scale[0] : -INFINITY;
      sc[j][1] = v1 ? lo.y * sm_scale[0] : -INFINITY;
      sc[j][2] = v0 ? hi.x * sm_scale[1] : -INFINITY;
      sc[j][3] = v1 ? hi.y * sm_scale[1] : -INFINITY;
      mx[0] = fmaxf(mx[0], fmaxf(sc[j][0], sc[j][1]));
      mx[1] = fmaxf(mx[1], fmaxf(sc[j][2], sc[j][3]));
    }
    float alpha[2];
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      mx[h] = fmaxf(mx[h], __shfl_xor_sync(0xffffffffu, mx[h], 1));
      mx[h] = fmaxf(mx[h], __shfl_xor_sync(0xffffffffu, mx[h], 2));
      const float m_new = fmaxf(m_run[h], mx[h]);
      alpha[h] = __expf(m_run[h] - m_new);
      m_run[h] = m_new;
      l_run[h] *= alpha[h];
    }
    uint32_t pa[2][4];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const float p0 = __expf(sc[j][0] - m_run[0]), p1 = __expf(sc[j][1] - m_run[0]);
      const float p2 = __expf(sc[j][2] - m_run[1]), p3 = __expf(sc[j][3] - m_run[1]);
      l_run[0] += p0 + p1;
      l_run[1] += p2 + p3;
      // Fold this warp's channel-group scale of each token into P.
      const float s0 = tscale[(8 * j + 2 * t) * 4 + w], s1 = tscale[(8 * j + 2 * t + 1) * 4 + w];
      pa[j >> 1][(j & 1) * 2] = pack_f16(p0 * s0, p1 * s1);
      pa[j >> 1][(j & 1) * 2 + 1] = pack_f16(p2 * s0, p3 * s1);
    }
    if (alpha[0] != 1.f || alpha[1] != 1.f) {
#pragma unroll
      for (int j = 0; j < 16; ++j) {
        acc[j][0] *= alpha[0];
        acc[j][1] *= alpha[0];
        acc[j][2] *= alpha[1];
        acc[j][3] *= alpha[1];
      }
    }
#pragma unroll
    for (int ks = 0; ks < 2; ++ks) {
#pragma unroll
      for (int jp = 0; jp < 8; ++jp) {
        uint32_t b[4];
        ldsm_x4_trans(b, tile + (16 * ks + (lm & 1) * 8 + lr) * kStride + 8 * swz(16 * w + 2 * jp + (lm >> 1)));
        mma_f16_16816(acc[2 * jp], pa[ks], b[0], b[1]);
        mma_f16_16816(acc[2 * jp + 1], pa[ks], b[2], b[3]);
      }
    }
    __syncthreads();
  }

#pragma unroll
  for (int h = 0; h < 2; ++h) {
    l_run[h] += __shfl_xor_sync(0xffffffffu, l_run[h], 1);
    l_run[h] += __shfl_xor_sync(0xffffffffu, l_run[h], 2);
  }
  if (splits == 1) {
    const float i0 = l_run[0] > 0.f ? 1.f / l_run[0] : 0.f, i1 = l_run[1] > 0.f ? 1.f / l_run[1] : 0.f;
    float* dst = o_lat + (int64_t(row) * kHeads + head0) * kLatentDim + 128 * w + 2 * t;
#pragma unroll
    for (int j = 0; j < 16; ++j) {
      *reinterpret_cast<float2*>(dst + g * kLatentDim + 8 * j) = make_float2(acc[j][0] * i0, acc[j][1] * i0);
      *reinterpret_cast<float2*>(dst + (g + 8) * kLatentDim + 8 * j) = make_float2(acc[j][2] * i1, acc[j][3] * i1);
    }
    if (lse && w == 0 && t == 0) {
      lse[int64_t(row) * kHeads + head0 + g] = l_run[0] > 0.f ? m_run[0] + logf(l_run[0]) : -INFINITY;
      lse[int64_t(row) * kHeads + head0 + g + 8] = l_run[1] > 0.f ? m_run[1] + logf(l_run[1]) : -INFINITY;
    }
  } else {
    float* dst = partial + ((int64_t(row) * splits + split) * kHeads + head0) * kLatentDim + 128 * w + 2 * t;
#pragma unroll
    for (int j = 0; j < 16; ++j) {
      *reinterpret_cast<float2*>(dst + g * kLatentDim + 8 * j) = make_float2(acc[j][0], acc[j][1]);
      *reinterpret_cast<float2*>(dst + (g + 8) * kLatentDim + 8 * j) = make_float2(acc[j][2], acc[j][3]);
    }
    if (w == 0 && t == 0) {
      float* ms = partial + int64_t(gridDim.x) * splits * kHeads * kLatentDim +
                  ((int64_t(row) * splits + split) * kHeads + head0) * 2;
      ms[2 * g] = m_run[0];
      ms[2 * g + 1] = l_run[0];
      ms[2 * (g + 8)] = m_run[1];
      ms[2 * (g + 8) + 1] = l_run[1];
    }
  }
}

// Merge of v2's split partials: the split weights exp(m_s - M) are computed once
// per head in shared memory, and each thread's loads for eight splits are in
// flight together (the v1 merge walks the splits one dependent load at a time).
__global__ __launch_bounds__(128) void attn_merge_v2_kernel(const float* partial, int rows, int splits, float* o_lat,
                                                            float* lse) {
  __shared__ float wgt[64];
  __shared__ float red[4];
  const int row = blockIdx.x, h = blockIdx.y, tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const float* ms = partial + int64_t(rows) * splits * kHeads * kLatentDim;
  float m = -INFINITY, l = 0.f;
  if (tid < splits) {
    const int64_t i = (int64_t(row) * splits + tid) * kHeads + h;
    m = ms[2 * i];
    l = ms[2 * i + 1];
    if (!(l > 0.f)) m = -INFINITY;
  }
  const float wm = warp_max(m);
  if (lane == 0 && warp < 2) red[warp] = wm;
  __syncthreads();
  const float M = fmaxf(red[0], red[1]);
  const float f = (tid < splits && l > 0.f) ? __expf(m - M) : 0.f;
  if (tid < 64) wgt[tid] = f;
  const float ws = warp_sum(f * l);
  __syncthreads();
  if (lane == 0 && warp < 2) red[2 + warp] = ws;
  __syncthreads();
  const float sum = red[2] + red[3];
  const float* src = partial + (int64_t(row) * splits * kHeads + h) * kLatentDim + tid * 4;
  const int64_t split_stride = int64_t(kHeads) * kLatentDim;
  float4 acc = make_float4(0.f, 0.f, 0.f, 0.f);
  for (int s0 = 0; s0 < splits; s0 += 8) {
    float4 v[8];
#pragma unroll
    for (int u = 0; u < 8; ++u)
      v[u] = s0 + u < splits && wgt[s0 + u] != 0.f ? *reinterpret_cast<const float4*>(src + (s0 + u) * split_stride)
                                                   : make_float4(0.f, 0.f, 0.f, 0.f);
#pragma unroll
    for (int u = 0; u < 8; ++u) {
      const float c = s0 + u < splits ? wgt[s0 + u] : 0.f;
      acc.x += v[u].x * c;
      acc.y += v[u].y * c;
      acc.z += v[u].z * c;
      acc.w += v[u].w * c;
    }
  }
  const float inv = sum > 0.f ? 1.f / sum : 0.f;
  *reinterpret_cast<float4*>(o_lat + (int64_t(row) * kHeads + h) * kLatentDim + tid * 4) =
      make_float4(acc.x * inv, acc.y * inv, acc.z * inv, acc.w * inv);
  if (tid == 0 && lse) lse[int64_t(row) * kHeads + h] = sum > 0.f ? M + logf(sum) : -INFINITY;
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

namespace {
// Opt-in shared memory per block on the current device (set by glm53f_dsa_init).
int g_smem_optin = 0;
}  // namespace

extern "C" int32_t glm53f_dsa_init(void) {
  cudaError_t st = cudaFuncSetAttribute(sparse_attn_kernel<1>, cudaFuncAttributeMaxDynamicSharedMemorySize, int(smem_bytes(1)));
  if (st != cudaSuccess) return st;
  st = cudaFuncSetAttribute(sparse_attn_kernel<2>, cudaFuncAttributeMaxDynamicSharedMemorySize, int(smem_bytes(2)));
  if (st != cudaSuccess) return st;
  st = cudaFuncSetAttribute(sparse_attn_kernel<4>, cudaFuncAttributeMaxDynamicSharedMemorySize, int(smem_bytes(4)));
  if (st != cudaSuccess) return st;
  int dev = 0, optin = 0;
  st = cudaGetDevice(&dev);
  if (st != cudaSuccess) return st;
  st = cudaDeviceGetAttribute(&optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev);
  if (st != cudaSuccess) return st;
  st = cudaFuncSetAttribute(attn_v2_kernel<1>, cudaFuncAttributeMaxDynamicSharedMemorySize, optin);
  if (st != cudaSuccess) return st;
  st = cudaFuncSetAttribute(attn_v2_kernel<2>, cudaFuncAttributeMaxDynamicSharedMemorySize, optin);
  if (st != cudaSuccess) return st;
  st = cudaFuncSetAttribute(attn_v2_kernel<4>, cudaFuncAttributeMaxDynamicSharedMemorySize, optin);
  if (st != cudaSuccess) return st;
  g_smem_optin = optin;
  return cudaSuccess;
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
  auto s = static_cast<cudaStream_t>(stream);
  const auto* w = reinterpret_cast<const __nv_bfloat16*>(kv_b);
  auto* qb = reinterpret_cast<__nv_bfloat16*>(q_abs_bf16);
  // Rows per block: the smallest of 1, 2, 4, 8 that covers `rows` (8 beyond).
  if (rows == 1) absorb_kernel<1><<<dim3(kHeads, 1, 2), 128, 0, s>>>(q, w, rows, qb, q_abs_f32);
  else if (rows == 2) absorb_kernel<2><<<dim3(kHeads, 1, 2), 128, 0, s>>>(q, w, rows, qb, q_abs_f32);
  else if (rows <= 4) absorb_kernel<4><<<dim3(kHeads, 1, 2), 128, 0, s>>>(q, w, rows, qb, q_abs_f32);
  else absorb_kernel<8><<<dim3(kHeads, (rows + 7) / 8, 2), 128, 0, s>>>(q, w, rows, qb, q_abs_f32);
  return cudaGetLastError();
}

extern "C" int32_t glm53f_dsa_mla_unabsorb_v(const float* o_lat, const uint16_t* kv_b, int32_t rows, float* o,
                                             void* stream) {
  if (rows < 0 || !o_lat || !kv_b || !o) return cudaErrorInvalidValue;
  if (rows == 0) return cudaSuccess;
  auto s = static_cast<cudaStream_t>(stream);
  const auto* w = reinterpret_cast<const __nv_bfloat16*>(kv_b);
  if (rows == 1) unabsorb_kernel<1><<<dim3(kHeads, 1, 4), 256, 0, s>>>(o_lat, w, rows, o);
  else if (rows == 2) unabsorb_kernel<2><<<dim3(kHeads, 1, 4), 256, 0, s>>>(o_lat, w, rows, o);
  else if (rows <= 4) unabsorb_kernel<4><<<dim3(kHeads, 1, 4), 256, 0, s>>>(o_lat, w, rows, o);
  else unabsorb_kernel<8><<<dim3(kHeads, (rows + 7) / 8, 4), 256, 0, s>>>(o_lat, w, rows, o);
  return cudaGetLastError();
}

extern "C" uint64_t glm53f_dsa_mla_workspace_bytes(int32_t rows, int32_t splits) {
  if (rows < 1 || splits <= 1) return 0;
  return uint64_t(rows) * splits * kHeads * (kLatentDim + 2) * 4;
}

extern "C" void glm53f_dsa_mla_plan(int32_t rows, int32_t max_tokens, int32_t sms, int32_t* splits,
                                    int32_t* head_groups) {
  rows = rows < 1 ? 1 : rows;
  sms = sms < 1 ? 1 : sms;
  // Measured on the RTX 4090: about one block per multiprocessor, one 16-head
  // group per block up to 4 rows and two beyond. (Four groups per block, 512
  // threads, is no faster at prefill sizes and spills registers.)
  const int groups = rows <= 4 ? 1 : 2;
  const int per_split = rows * (4 / groups);
  const int tiles = max_tokens > 0 ? (max_tokens + v2::kT - 1) / v2::kT : 1;
  int s = sms / per_split;
  s = s < 1 ? 1 : s;
  s = s > 64 ? 64 : s;
  s = s > tiles ? tiles : s;
  *splits = s;
  *head_groups = groups;
}

namespace {

int32_t check_attn_args(const uint16_t* q_abs, const int32_t* tokens, int32_t token_stride, const int32_t* counts,
                        const int32_t* row_req, int32_t rows, const glm53f_dsa_cache_t& cache, int32_t splits,
                        int32_t head_groups, void* workspace, uint64_t workspace_bytes, float* o_lat) {
  if (rows < 0 || splits < 1 || splits > 64 || token_stride < 1 || !valid_cache(cache) ||
      (head_groups != 1 && head_groups != 2 && head_groups != 4))
    return cudaErrorInvalidValue;
  if (rows == 0) return cudaSuccess;
  if (!q_abs || !tokens || !counts || !row_req || !o_lat) return cudaErrorInvalidValue;
  if (splits > 1 && (!workspace || workspace_bytes < glm53f_dsa_mla_workspace_bytes(rows, splits)))
    return cudaErrorInvalidValue;
  return cudaSuccess;
}

int32_t launch_merge_v2(float* partial, int32_t rows, int32_t splits, float* o_lat, float* lse, cudaStream_t s) {
  attn_merge_v2_kernel<<<dim3(rows, kHeads), 128, 0, s>>>(partial, rows, splits, o_lat, lse);
  return cudaGetLastError();
}

}  // namespace

extern "C" int32_t glm53f_dsa_mla_sparse_attn(const uint16_t* q_abs, const int32_t* tokens, int32_t token_stride,
    const int32_t* counts, const int32_t* row_req, int32_t rows, float scale, glm53f_dsa_cache_t cache,
    int32_t splits, int32_t head_groups, void* workspace, uint64_t workspace_bytes, float* o_lat, float* lse,
    void* stream) {
  const int32_t bad = check_attn_args(q_abs, tokens, token_stride, counts, row_req, rows, cache, splits, head_groups,
                                      workspace, workspace_bytes, o_lat);
  if (bad != cudaSuccess || rows == 0) return bad;
  const int max_tiles = ((token_stride + v2::kT - 1) / v2::kT + splits - 1) / splits;
  const uint64_t smem = v2::smem_bytes(head_groups, max_tiles);
  if (g_smem_optin == 0 || smem > uint64_t(g_smem_optin))  // not initialized, or rows too long for v2
    return glm53f_dsa_mla_sparse_attn_v1(q_abs, tokens, token_stride, counts, row_req, rows, scale, cache, splits,
                                         head_groups, workspace, workspace_bytes, o_lat, lse, stream);
  auto s = static_cast<cudaStream_t>(stream);
  float* partial = static_cast<float*>(workspace);
  const dim3 grid(rows, 4 / head_groups, splits);
  const auto* q = reinterpret_cast<const __nv_bfloat16*>(q_abs);
  switch (head_groups) {
    case 1:
      attn_v2_kernel<1><<<grid, 128, smem, s>>>(q, tokens, token_stride, counts, row_req, scale, cache, splits, partial,
                                               o_lat, lse);
      break;
    case 2:
      attn_v2_kernel<2><<<grid, 256, smem, s>>>(q, tokens, token_stride, counts, row_req, scale, cache, splits, partial,
                                               o_lat, lse);
      break;
    default:
      attn_v2_kernel<4><<<grid, 512, smem, s>>>(q, tokens, token_stride, counts, row_req, scale, cache, splits, partial,
                                               o_lat, lse);
  }
  const cudaError_t st = cudaGetLastError();
  if (st != cudaSuccess || splits == 1) return st;
  return launch_merge_v2(partial, rows, splits, o_lat, lse, s);
}

extern "C" int32_t glm53f_dsa_mla_sparse_attn_v1(const uint16_t* q_abs, const int32_t* tokens, int32_t token_stride,
    const int32_t* counts, const int32_t* row_req, int32_t rows, float scale, glm53f_dsa_cache_t cache,
    int32_t splits, int32_t head_groups, void* workspace, uint64_t workspace_bytes, float* o_lat, float* lse,
    void* stream) {
  const int32_t bad = check_attn_args(q_abs, tokens, token_stride, counts, row_req, rows, cache, splits, head_groups,
                                      workspace, workspace_bytes, o_lat);
  if (bad != cudaSuccess || rows == 0) return bad;
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
