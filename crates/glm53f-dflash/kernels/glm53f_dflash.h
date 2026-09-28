/* glm53f-dflash kernels: the DFlash2 drafter's non-GEMM work on the coordinator GPU.
 *
 * Every pointer is a device pointer. bf16 values are passed as their uint16_t bits. Every
 * function launches on `stream` and returns cudaGetLastError() (or cudaErrorInvalidValue for
 * arguments it rejects); none synchronizes.
 *
 * Numerics: compiled with --fmad=false --prec-div=true --prec-sqrt=true, so a kernel's f32
 * arithmetic is the IEEE operation sequence its source spells out. Where a result feeds a
 * comparison with the CPU reference (crates/glm53f-dflash/src/reference.rs, cpu.rs,
 * selector.rs), the kernel uses the reference's order of operations: the RMSNorm mean of squares
 * in f64, RoPE angles as f32 products with sine and cosine in f64, the convolution's
 * `acc + base * x` then `acc + dyn * x` per tap, the selector's sums in the tree `selector.rs`
 * models. The attention's softmax is online (flash-decoding style) and uses expf, so it agrees
 * with the reference to f32 rounding, not bit for bit.
 *
 * Fixed sizes: head dimension 128, block 8 rows, convolution taps <= 8, 16 candidates.
 *
 * A request's context ring is one allocation of [layers][2 (K, V)][ring][kv_width] bf16: the
 * key (after k_norm and RoPE) and value of position p at row p % ring. `bases[r]` is request r's
 * allocation.
 */
#ifndef GLM53F_DFLASH_H
#define GLM53F_DFLASH_H

#include <cuda_runtime.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* y = w * (x * rsqrt(mean(x^2) + eps)) per row of n; writes y_f32 and/or y_bf16 (either may be
 * null; y_f32 may alias x). One block per row. */
cudaError_t g53d_rmsnorm(const float* x, int64_t ldx, const uint16_t* w, int rows, int n, float eps,
                         float* y_f32, int64_t ldy, uint16_t* y_bf16, int64_t ldb, cudaStream_t stream);

/* RoPE table for `rows` positions: cs[row * 128 + i] = cos, cs[row * 128 + 64 + i] = sin of the
 * angle pos[row] * inv_freq[i] formed in f32 (i < 64), the sine and cosine taken in f64 and
 * rounded to f32 (the CPU reference's `rope`). */
cudaError_t g53d_rope_table(const int64_t* pos, int rows, const float* inv_freq, float* cs, cudaStream_t stream);

/* Per row and head of 128 values at x[row * ldx + head * 128]: RMSNorm with weight w[128] in
 * place, then RoPE (rotate_half pairs i, i + 64) with the row's table cs[row * 128 ..]
 * (g53d_rope_table). Optionally a bf16 copy to out_bf16[row * ldo + head * 128]. One warp per
 * (row, head). */
cudaError_t g53d_head_norm_rope(float* x, int64_t ldx, int rows, int heads, const uint16_t* w, const float* cs,
                                float eps, uint16_t* out_bf16, int64_t ldo, cudaStream_t stream);

/* Store rows of keys k[row * ld ..] and values v[row * ld ..] (f32, kv_width each) as bf16 into
 * request req[row]'s ring for `layer`, at row pos[row] % ring. */
cudaError_t g53d_store_kv(const float* k, const float* v, int64_t ld, int rows, int kv_width, const int32_t* req,
                          const int64_t* pos, const uint64_t* bases, int layer, int ring, cudaStream_t stream);

/* The grouped dynamic convolution over blocks of `block` rows (rows = requests x block), one side:
 *   out[l][c] = sum over o < taps with o <= l % block of (base[o][c] + dyn[l][o * groups + c / gs]) * x[l - o][c]
 * accumulated tap by tap as acc = acc + base * x; acc = acc + dyn * x. `base` is the side's
 * [taps][n] bf16; `dyn` points at the side's first column of rows of stride lddyn. Writes out_f32
 * (=), out_bf16 (=) and resid (+=), each when non-null. */
cudaError_t g53d_dyn_conv(const float* x, int64_t ldx, const float* dyn, int64_t lddyn, const uint16_t* base,
                          int rows, int n, int group_size, int taps, int block, float* out_f32, int64_t ldo,
                          uint16_t* out_bf16, int64_t ldb, float* resid, int64_t ldr, cudaStream_t stream);

/* Scratch floats g53d_attention needs for `partial` (nreq x splits x 8 rows x heads x 130). */
int64_t g53d_attention_partial_floats(int nreq, int heads, int splits);

/* Attention of each request's 8 block rows over its ring for `layer` (heads / kv_heads <= 4). q: [nreq * 8][ldq] f32,
 * head h at column h * 128 (after q_norm and RoPE). Row j of request r (position start[r] + j)
 * sees the ring rows at positions max(lo[r], start[r] + j - window_left) .. start[r] + 7 (the
 * context below start[r], the block's own rows at start[r].., which the caller stored).
 * Scores are q . k * scale. Query head h reads KV head h / (heads / kv_heads).
 * Keys are processed in `splits` ranges of split_keys per request (flash-decoding), merged by a
 * second kernel. Writes out_bf16 [row * ldo + h * 128] and/or out_f32 (same layout). */
cudaError_t g53d_attention(const float* q, int64_t ldq, int nreq, int heads, int kv_heads, const int64_t* start,
                           const int64_t* lo, const uint64_t* bases, int layer, int ring, int window_left,
                           float scale, int split_keys, int splits, float* partial, uint16_t* out_bf16,
                           float* out_f32, int64_t ldo, cudaStream_t stream);

/* out[r][i] = bf16(silu(gu[r][i]) * gu[r][inter + i]), silu(g) = g / (1 + expf(-g)). */
cudaError_t g53d_silu_mul(const float* gu, int64_t ldgu, int rows, int inter, uint16_t* out, int64_t ldo,
                          cudaStream_t stream);

/* h[r * block + 0] = anchor_rows[r], h[r * block + j] = mask_row for 0 < j < block; bf16 to f32. */
cudaError_t g53d_block_embed(const uint16_t* anchor_rows, const uint16_t* mask_row, int nreq, int block, int n,
                             float* h, cudaStream_t stream);

/* dst[r * (block - 1) + j - 1] = src[r * block + j] for 0 < j < block (rows of n bf16). */
cudaError_t g53d_gather_drafts(const uint16_t* src, int64_t lds, int nreq, int block, int n, uint16_t* dst,
                               cudaStream_t stream);

/* Workspace bytes g53d_topk16 needs for `rows` rows. */
int64_t g53d_topk16_workspace_bytes(int rows);

/* The 16 largest of logits[row * ld + 0 .. limit), descending, ties to the lower id (two passes:
 * 32 chunks per row, then their lists merged). */
cudaError_t g53d_topk16(const float* logits, int64_t ld, int rows, int limit, void* workspace, float* vals,
                        int32_t* ids, cudaStream_t stream);

/* The selector's walk, one block per request over `slots` positions of 16 candidates:
 *   score[c] = vals[e][c] + tree_sum_r((pred[p][r] * hproj[e][r]) * succ[ids[e][c]][r])
 * from p = anchors[req] at e = 0, then the chosen token. temperature[req] <= 0: the first
 * maximum (q one-hot); else a draw by inverse CDF from softmax(score / T) with uniforms[req][e].
 * Writes tokens, index [nreq][slots], scores and q [nreq][slots][16], conf [nreq][slots]
 * (softmax(score)[chosen] at T = 1). rank <= 1024. */
cudaError_t g53d_select(const float* hproj, const float* vals, const int32_t* ids, const int32_t* anchors,
                        const uint16_t* pred, const uint16_t* succ, int rank, int nreq, int slots,
                        const float* temperature, const float* uniforms, int32_t* tokens, int32_t* index,
                        float* scores, float* q, float* conf, cudaStream_t stream);

#ifdef __cplusplus
}
#endif

#endif
