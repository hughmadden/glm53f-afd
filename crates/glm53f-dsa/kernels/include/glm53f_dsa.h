// glm53f-dsa CUDA ABI: the DSA indexer (k-pool 4, top-512 pools plus tail) and
// sparse MLA over the FP8 528-byte latent record, for GLM-5.3-Flash.
//
// Every entry point returns a cudaError_t as int32_t (0 = success) and never
// synchronizes. Pointers are device pointers unless stated otherwise. Shapes are
// fixed to GLM-5.3-Flash: 64 heads, latent 512, index 32 heads x 128, pool 4,
// 512 pools kept, 2,051 selected tokens at most.
//
// Cache layout (see crates/glm53f-dsa/src/cache.rs): per DSA layer, physical page
// p starts at base + p * page_stride and holds 64 latent records of 528 bytes
// (token slot t at 528 t), then 16 x 128 pooled-key codes at 33,792, then 16 f32
// pooled-key scales at 35,840. A request's logical page i (tokens 64 i .. 64 i + 63,
// pools 16 i .. 16 i + 15) is physical page page_tables[req * max_pages + i].
#ifndef GLM53F_DSA_H
#define GLM53F_DSA_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// One layer's paged cache.
typedef struct {
  uint8_t* base;                // physical page 0 of this layer (16-byte aligned)
  int64_t page_stride;          // bytes between physical pages (multiple of 16, >= 35,904)
  const int32_t* page_tables;   // [n_req][max_pages]
  int32_t max_pages;            // row stride of page_tables
  int32_t n_pages;              // physical pages (bounds check)
} glm53f_dsa_cache_t;

// One request's rows in a window (prefill chunk, decode step or verify window).
// Rows of a request are contiguous: rows first_row .. first_row + rows - 1 sit at
// absolute positions start .. start + rows - 1. `start` is the committed length
// before the window. `accepted` is used by tail_commit only.
typedef struct {
  int32_t first_row;
  int32_t rows;
  int32_t start;
  int32_t accepted;
} glm53f_dsa_window_t;

// ---- Indexer -----------------------------------------------------------------

// Pools completed inside the window: LayerNorm(k_raw) and the gates (both
// rounded to BF16, as the tail stores them), softmax(gate + ape) per channel over
// the pool's 4 tokens, weighted sum, FP8 E4M3 with a power-of-two scale, written
// to the cache. Tokens before the window come from `tails` (one 1,552-byte record
// per request). One block per row; rows that do not end a pool exit.
//   k_raw, gate: [rows][128] f32; ln_w, ln_b: [128] f32; ape: [4][128] f32.
//   row_req: [rows] request index of each row (also indexes windows and tails).
int32_t glm53f_dsa_index_pool_write(const float* k_raw, const float* gate, const float* ln_w,
                                    const float* ln_b, float ln_eps, const float* ape,
                                    const uint8_t* tails, const glm53f_dsa_window_t* windows,
                                    const int32_t* row_req, int32_t rows,
                                    glm53f_dsa_cache_t cache, void* stream);

// After acceptance: rewrite each request's tail from its old tail and its first
// `accepted` window rows (in place; one block per request).
int32_t glm53f_dsa_index_tail_commit(const float* k_raw, const float* gate, const float* ln_w,
                                     const float* ln_b, float ln_eps, uint8_t* tails,
                                     const glm53f_dsa_window_t* windows, int32_t n_req,
                                     void* stream);

// Workspace bytes for index_select with `rows` query rows and `chunks` chunks
// per row (from glm53f_dsa_index_plan): rows x chunks x 4 KiB plus one merge level.
uint64_t glm53f_dsa_index_workspace_bytes(int32_t rows, int32_t chunks);

// Chunk plan: pools per chunk (a multiple of 64) and chunks per row, for rows
// that see up to `max_pools` pools on a device with `sms` multiprocessors.
void glm53f_dsa_index_plan(int32_t rows, int32_t max_pools, int32_t sms, int32_t* chunk_pools,
                           int32_t* chunks);

// Score and select. For each row r at position row_pos[r] (visible pools
// n = (row_pos + 1) / 4), score every visible pool
//   s = sum_h w[r][h] * relu(score_scale * q[r][h] . key)
// with the FP8 pooled keys (f16 tensor-core products, f32 accumulation), keep
// the top min(512, n) by (score desc, pool asc), and write:
//   pools_out [rows][512]  kept pool ids ascending, -1 padded
//   tokens_out[rows][2051] selected tokens ascending (kept pools, then the tail), -1 padded
//   counts_out[rows][2]    {kept pools, selected tokens}
// Rows with n <= 512 keep every pool without scoring. q: [rows][32][128] f32,
// w: [rows][32] f32 (weights_proj(x) * 32^-0.5). debug_scores (optional,
// [rows][max_pools] f32) receives every computed score.
// Precondition (not checked on the device): every row's n <= max_pools <=
// chunk_pools * chunks, and every visible pool's page is mapped.
int32_t glm53f_dsa_index_select(const float* q, const float* w, float score_scale,
                                const int32_t* row_pos, const int32_t* row_req, int32_t rows,
                                int32_t max_pools, glm53f_dsa_cache_t cache, int32_t chunk_pools,
                                int32_t chunks, void* workspace, uint64_t workspace_bytes,
                                int32_t* pools_out, int32_t* tokens_out, int32_t* counts_out,
                                float* debug_scores, void* stream);

// ---- Sparse MLA --------------------------------------------------------------

// Latent records: optional RMSNorm (norm_w != NULL: x * rsqrt(mean(x^2) + eps) * w),
// then 4 groups of 128 channels to E4M3 with power-of-two scales, written to
// token row_pos[r] of request row_req[r]. latent: [rows][512] f32.
int32_t glm53f_dsa_mla_latent_write(const float* latent, const float* norm_w, float eps,
                                    const int32_t* row_pos, const int32_t* row_req,
                                    int32_t rows, glm53f_dsa_cache_t cache, void* stream);

// q_abs[r][h][l] = sum_i q[r][h][i] * kv_b[h * 512 + i][l] (i < 256, the key rows).
// q: [rows][64][256] f32; kv_b: [64 * 512][512] BF16 (checkpoint kv_b_proj);
// q_abs_bf16: [rows][64][512] BF16 (for sparse_attn); q_abs_f32 optional.
int32_t glm53f_dsa_mla_absorb_q(const float* q, const uint16_t* kv_b, int32_t rows,
                                uint16_t* q_abs_bf16, float* q_abs_f32, void* stream);

// Sparse attention in latent space. For row r, over tokens[r][0 .. counts[r*2+1])
// (ascending), per head h: o[r][h] = sum_j softmax_j(scale q_abs[r][h] . c_j) c_j,
// with c_j decoded from the FP8 record (BF16 tensor-core products, FP32 softmax
// and accumulation). Writes o_lat [rows][64][512] f32 and lse [rows][64]
// (natural log of the sum of exp(scale * score)). `splits` > 1 splits each row's
// tokens across blocks (partials in the workspace, merged at the end);
// `head_groups` in {1, 2, 4} is how many 16-head groups share one decoded tile.
uint64_t glm53f_dsa_mla_workspace_bytes(int32_t rows, int32_t splits);
int32_t glm53f_dsa_mla_sparse_attn(const uint16_t* q_abs, const int32_t* tokens,
                                   int32_t token_stride, const int32_t* counts,
                                   const int32_t* row_req, int32_t rows, float scale,
                                   glm53f_dsa_cache_t cache, int32_t splits,
                                   int32_t head_groups, void* workspace,
                                   uint64_t workspace_bytes, float* o_lat, float* lse,
                                   void* stream);

// o[r][h][v] = sum_l kv_b[h * 512 + 256 + v][l] * o_lat[r][h][l]. o: [rows][64][256] f32.
int32_t glm53f_dsa_mla_unabsorb_v(const float* o_lat, const uint16_t* kv_b, int32_t rows,
                                  float* o, void* stream);

// One-time per-process setup (shared-memory limits). Call before any launch.
int32_t glm53f_dsa_init(void);

#ifdef __cplusplus
}
#endif

#endif  // GLM53F_DSA_H
