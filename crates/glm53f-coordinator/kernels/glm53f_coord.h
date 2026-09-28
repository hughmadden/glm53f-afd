/* SPDX-License-Identifier: MIT
 *
 * glm53f-coordinator: the serving shell's kernels, C ABI.
 *
 * Every entry point takes raw device pointers and a CUDA stream, returns a cudaError_t value
 * (0 on success) and is asynchronous on the stream. Row strides (`ld`) are in elements.
 *
 * Selection (kernels/select.cu, kernels/sample.cu). A model selects a batch's tokens from its
 * logit rows in three launches, in this order:
 *   1. glm53f_coord_mask_rows     rows with a grammar mask: disallowed logits become -inf;
 *   2. glm53f_coord_argmax_rows   every row's argmax over ids [0, vocab);
 *   3. glm53f_coord_sample_rows   each sampled row's draw over ids [0, vocab) overwrites its argmax.
 * `vocab` is the tokenizer's id bound (154,856 for GLM-5.3-Flash), below the LM head's padded
 * row count `ld` (154,880), so a padding id is never selected. The CPU reference of the whole
 * selection is `select_pick` in src/sampling.rs.
 *
 * Wire (kernels/wire.cu): the hidden rows' FP8 E4M3 / UE8M0-K32 quantizer, the fill of a mapped
 * DS41RTE3 v3 request frame, and the rank-ordered FP32 sum of the four ranks' BF16 return
 * planes, each bit-identical to its host twin in src/wire.rs.
 */
#ifndef GLM53F_COORD_H
#define GLM53F_COORD_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

struct CUstream_st;
/** Same type as cudaStream_t. NULL is the legacy default stream. */
typedef struct CUstream_st* glm53f_coord_stream_t;

/** One sampled row (32 bytes; `DeviceRow` in src/sampling.rs). */
struct glm53f_coord_row {
  int32_t row;      /* the logit row, and the index of `out` it overwrites */
  float inv_t;      /* 1 / temperature */
  float top_p;      /* >= 1: off */
  float ln_min_p;   /* -inf: off */
  int32_t top_k;    /* 0: off */
  int32_t pad;
  uint64_t rnd;     /* the draw's 64 random bits for this (seed, position) */
};

/** out[r] = the first index of the largest non-NaN value of row r over [0, vocab); 0 when all
 *  are NaN. One block of 1,024 threads per row. */
int glm53f_coord_argmax_rows(const float* x, int64_t ld, int32_t rows, int32_t vocab, int32_t* out,
                             glm53f_coord_stream_t stream);

/** As glm53f_coord_argmax_rows, plus prob[r] = the softmax probability of that maximum. */
int glm53f_coord_argmax_prob_rows(const float* x, int64_t ld, int32_t rows, int32_t vocab, int32_t* out, float* prob,
                                  glm53f_coord_stream_t stream);

/** masks[r] (a device array of `rows` device pointers): row r's allowed-token bitset, bit i % 32
 *  of word i / 32 for token i, over `bits` tokens; NULL leaves the row alone. Every logit in
 *  columns [0, cols) whose token is not allowed (tokens at or past `bits` never are) becomes -inf.
 *  cols <= ld. */
int glm53f_coord_mask_rows(float* x, int64_t ld, int32_t rows, int32_t cols, const uint32_t* const* masks, int32_t bits,
                           glm53f_coord_stream_t stream);

/** For each of the `n` rows (a device array of struct glm53f_coord_row): overwrite out[row] with
 *  the row's draw over ids [0, vocab) (temperature, min_p, top_k, top_p in vLLM's order, then an
 *  exact categorical draw in fixed point). A row without a finite maximum keeps its argmax. */
int glm53f_coord_sample_rows(const float* x, int64_t ld, int32_t vocab, const void* rows, int32_t n, int32_t* out,
                             glm53f_coord_stream_t stream);

/** Per K32 block of x [n_blocks * 32]: scales[b] = the UE8M0 byte clamp(ceil(log2(amax / 448)) + 127,
 *  0, 254) (0 for an all-zero block) and scale_inv[b] = 2^(127 - scales[b]). */
int glm53f_coord_quant_scales(const float* x, long n_blocks, uint8_t* scales, float* scale_inv,
                              glm53f_coord_stream_t stream);

/** payload[i] = E4M3(x[i] * scale_inv[i / 32]), nearest, ties to the larger magnitude, clamped at 448. */
int glm53f_coord_quantize_hidden(const float* x, const float* scale_inv, uint8_t* payload, int64_t n_elem,
                                 glm53f_coord_stream_t stream);

/** out[i] = ((((0 + p0[i]) + p1[i]) + p2[i]) + p3[i]) * scale in FP32, each p a BF16 plane of n values. */
int glm53f_coord_rank_sum_bf16(const uint16_t* p0, const uint16_t* p1, const uint16_t* p2, const uint16_t* p3,
                               float* out, long n, float scale, glm53f_coord_stream_t stream);

/** Write t rows' route entries (row u32, expert u32 = idx, weight f32 = wts; topk per row, 12 bytes
 *  each, at `routes`) and hidden rows (hid E4M3 payload bytes then hid / 32 scale bytes, at `hidden`
 *  with row pitch `pitch`) of one request frame. `hidden` must be 8-byte aligned. */
int glm53f_coord_frame_fill(const int32_t* idx, const float* wts, const uint8_t* payload, const uint8_t* scales,
                            int32_t t, int32_t topk, int32_t hid, uint8_t* routes, uint8_t* hidden, int32_t pitch,
                            glm53f_coord_stream_t stream);

#ifdef __cplusplus
}
#endif

#endif /* GLM53F_COORD_H */
