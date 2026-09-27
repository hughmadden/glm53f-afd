/* SPDX-License-Identifier: MIT
 *
 * glm53f-kda: GLM-5.3-Flash KDA (Kimi delta attention) kernels, C ABI.
 *
 * The kernels are a port of TensorFold's fused KDA kernels (see PROVENANCE.md and
 * LICENSE.tensorfold). Every entry point takes raw device pointers, explicit strides
 * (in elements, not bytes) and a CUDA stream, and returns a cudaError_t value:
 * 0 on success, 1 (cudaErrorInvalidValue) when an argument is rejected on the host,
 * otherwise the launch error. Launches are asynchronous on `stream`.
 *
 * Geometry: DK = DV = 128 (head dims), TAPS = 4 (short convolution), any number of
 * heads H >= 1 (64 for GLM-5.3-Flash). C = 3 * H * 128 conv channels, q | k | v.
 *
 * Layouts (all row-major, elements):
 *   recurrent state  f32 [H][128 (v)][128 (k)]: value row major. This is the transpose,
 *                    per head, of the reference's [H][K][V] (`recurrent_states`).
 *   conv window      bf16 [3][C]: the q|k|v projection rows of the 3 positions before
 *                    the window's first row, oldest first. The reference caches the last
 *                    4 positions channel-major, [C][4]; the window is its last 3 columns.
 *   conv weight      bf16 [C][4] (q_conv1d | k_conv1d | v_conv1d); tap 3 multiplies the
 *                    current row.
 *   replay saves     row r, head h: k (L2-normalized, f32) and exp(g) (f32) at
 *                    (r * H + h) * 128, v (bf16) at the same index, beta (f32) at r * H + h.
 *
 * Numerics (identical to the source kernels): conv + SiLU in f32 with one bf16 rounding;
 * q, k L2-normalized in f32 (eps 1e-6 inside the sum, q scaled by 128^-0.5); per-channel
 * decay exp(lower * sigmoid(exp(A_log) * (a + dt_bias))); beta = bf16(sigmoid(b)); the
 * delta rule in f32; the read-out rounded to bf16; gated RMSNorm
 * bf16(w * (y * 1/sqrt(mean(y^2) + eps)) * sigmoid(gate)). Built with --fmad=false: the
 * state update is one routine shared by the chain and the replays, so a replayed prefix of
 * a window has the bits of serial single-row steps.
 */
#ifndef GLM53F_KDA_H
#define GLM53F_KDA_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

struct CUstream_st;
/** Same type as cudaStream_t. NULL is the legacy default stream. */
typedef struct CUstream_st* glm53f_stream_t;
/** Raw bfloat16 bits. */
typedef uint16_t glm53f_bf16;

#define GLM53F_KDA_DK 128
#define GLM53F_KDA_DV 128
#define GLM53F_KDA_TAPS 4

/**
 * One layer, one request: `rows` consecutive rows from the committed state.
 *
 * p, p_stride   bf16 projection rows. Row r holds q | k | v at [0, C) of p + r * p_stride and
 *               the H beta logits (b_proj output, before the sigmoid) at [b_off, b_off + H).
 *               b_off may be negative.
 * a, a_stride   bf16 forget-gate rows f_b_proj(f_a_proj(x)), [H * 128] each, before dt_bias.
 * g, g_stride   bf16 output-gate rows g_b_proj(g_a_proj(x)), [H * 128] each.
 * conv          bf16 conv window [3][C] (read only; advance it with glm53f_kda_conv_shift).
 * conv_w        bf16 [C][4].
 * state_in      f32 [H][128][128].
 * state_out     f32 [H][128][128], the state after the last row. May be NULL (not written)
 *               and may equal state_in (in place).
 * a_log         f32 [H]. dt_bias: f32 [H * 128]. norm_w: bf16 [128] (o_norm.weight).
 * eps           rms_norm_eps of the gated RMSNorm (1e-5). lower: gate_lower_bound (-5).
 * out           bf16 rows of [H * 128] at out + r * out_stride: the gated RMSNorm output,
 *               which is o_proj's input.
 * k_save, v_save, g_save, b_save
 *               replay inputs for rows 0 .. rows-1 (layout above). All NULL or all set.
 */
int glm53f_kda_chain(int32_t heads, int32_t rows,
                     const glm53f_bf16* p, int64_t p_stride, int64_t b_off,
                     const glm53f_bf16* a, int64_t a_stride,
                     const glm53f_bf16* g, int64_t g_stride,
                     const glm53f_bf16* conv, const glm53f_bf16* conv_w,
                     const float* state_in, float* state_out,
                     const float* a_log, const float* dt_bias, const glm53f_bf16* norm_w,
                     float eps, float lower,
                     glm53f_bf16* out, int64_t out_stride,
                     float* k_save, glm53f_bf16* v_save, float* g_save, float* b_save,
                     glm53f_stream_t stream);

/**
 * One layer, a batch of requests. Block (h, b) runs head h of request b.
 *
 * cu_rows    device int32 [batch + 1], non-decreasing: request b owns rows
 *            [cu_rows[b], cu_rows[b + 1]) of p, a, g, out and the replay saves (the saves are
 *            indexed by that global row). A request may have 0 rows (its state is copied).
 * conv_off   device int64 [batch]: request b's conv window is conv + conv_off[b].
 * state_off  device int64 [batch]: request b's state is state_in + state_off[b], written to
 *            state_out + state_off[b] (state_out may be NULL or equal state_in). The requests'
 *            state regions must not overlap.
 * Other arguments as in glm53f_kda_chain. The device arrays are not checked on the host.
 */
int glm53f_kda_chain_batch(int32_t heads, int32_t batch, const int32_t* cu_rows,
                           const glm53f_bf16* p, int64_t p_stride, int64_t b_off,
                           const glm53f_bf16* a, int64_t a_stride,
                           const glm53f_bf16* g, int64_t g_stride,
                           const glm53f_bf16* conv, const int64_t* conv_off,
                           const glm53f_bf16* conv_w,
                           const float* state_in, float* state_out, const int64_t* state_off,
                           const float* a_log, const float* dt_bias, const glm53f_bf16* norm_w,
                           float eps, float lower,
                           glm53f_bf16* out, int64_t out_stride,
                           float* k_save, glm53f_bf16* v_save, float* g_save, float* b_save,
                           glm53f_stream_t stream);

/**
 * One layer, one request: the state after the first `rows` saved rows of a chain
 * (rows >= 0; 0 copies the state). state_out may equal state_in.
 */
int glm53f_kda_replay(int32_t heads, int32_t rows, const float* state_in, float* state_out,
                      const float* k_save, const glm53f_bf16* v_save, const float* g_save,
                      const float* b_save, glm53f_stream_t stream);

/**
 * Every layer of one request at once: layer l's states at state_in/state_out + l * state_stride,
 * its saves at k_save/v_save/g_save + l * kv_stride and b_save + l * b_stride.
 */
int glm53f_kda_replay_layers(int32_t heads, int32_t layers, int32_t rows,
                             const float* state_in, float* state_out, int64_t state_stride,
                             const float* k_save, const glm53f_bf16* v_save, const float* g_save,
                             const float* b_save, int64_t kv_stride, int64_t b_stride,
                             glm53f_stream_t stream);

/**
 * Every layer of every request of a batch (the commit after a verify round). Block
 * (l * H + h, b) replays keep[b] rows of request b, starting at its row cu_rows[b], from
 * state_in + l * state_stride + state_off[b] into state_out + (same offset).
 * cu_rows: device int32 [batch + 1] (as in glm53f_kda_chain_batch); keep: device int32
 * [batch], 0 <= keep[b] <= cu_rows[b + 1] - cu_rows[b]; state_off: device int64 [batch].
 */
int glm53f_kda_replay_batch(int32_t heads, int32_t layers, int32_t batch,
                            const int32_t* cu_rows, const int32_t* keep,
                            const float* state_in, float* state_out, int64_t state_stride,
                            const int64_t* state_off,
                            const float* k_save, const glm53f_bf16* v_save, const float* g_save,
                            const float* b_save, int64_t kv_stride, int64_t b_stride,
                            glm53f_stream_t stream);

/**
 * Advance a conv window past `keep` kept rows, in place: window rows j = 0..2 become rows
 * keep + j of [window; new rows], where new row r is the q|k|v part ([0, channels)) of
 * p + r * p_stride. 0 <= keep; keep must not exceed the rows the chain ran.
 */
int glm53f_kda_conv_shift(int32_t channels, int32_t keep, glm53f_bf16* conv,
                          const glm53f_bf16* p, int64_t p_stride, glm53f_stream_t stream);

/**
 * glm53f_kda_conv_shift for every layer of every request: the window of (layer l, request b)
 * is conv + l * conv_stride + conv_off[b]; its new rows start at p + l * p_layer_stride +
 * cu_rows[b] * p_stride; it keeps keep[b] rows (device arrays as in glm53f_kda_replay_batch).
 */
int glm53f_kda_conv_shift_batch(int32_t channels, int32_t layers, int32_t batch,
                                const int32_t* cu_rows, const int32_t* keep,
                                glm53f_bf16* conv, int64_t conv_stride, const int64_t* conv_off,
                                const glm53f_bf16* p, int64_t p_layer_stride, int64_t p_stride,
                                glm53f_stream_t stream);

#ifdef __cplusplus
}
#endif

#endif /* GLM53F_KDA_H */
