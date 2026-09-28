// SPDX-License-Identifier: MIT
//
// GLM-5.3-Flash's KDA (Kimi delta attention) on CUDA: one block of 1024 threads per (head, request), a chain
// of R rows. Ported from TensorFold (families/glm5_next/cuda/kda.cu, MIT: see LICENSE.tensorfold and
// PROVENANCE.md). The per-row arithmetic below is the source's, unchanged; the port changes the launcher (a C
// ABI instead of a torch extension), adds a batch of requests per launch, explicit strides, in-place states,
// and the conv-window shift that the source ran as a Triton kernel at commit.
//
// Per row, following the Hugging Face definition: the depthwise conv over [conv state; q|k|v rows] (4 taps,
// fp32) with SiLU and one bf16 rounding; fp32 L2 norms of q and k (eps inside the sum, q times DK^-0.5); the
// per-channel decay g_i = exp(lower * sigmoid(exp(A_log) * (a_i + dt_bias_i))) in fp32; beta = bf16(sigmoid(b));
// the delta rule in fp32 (decay along the key channel, read with k, correct toward v, read out with q);
// read-out rounded to bf16; then the gated RMSNorm bf16(w * (y * rsqrt(mean(y^2) + eps)) * sigmoid(gate)).
// Warp w owns value rows 4w .. 4w + 3, lane l the key columns 4l .. 4l + 3. The state update is one routine
// (``update``) shared with the replays, compiled without FMA contraction (--fmad=false), so a replayed prefix of
// a window gives the bits of the serial steps.

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <stdint.h>

#include "glm53f_kda.h"

namespace {

constexpr int DK = 128, DV = 128, TAPS = 4;

__device__ __forceinline__ float bf(float x) { return __bfloat162float(__float2bfloat16_rn(x)); }

__device__ __forceinline__ float warp_sum(float x) {
    for (int o = 16; o; o >>= 1) x += __shfl_xor_sync(0xffffffffu, x, o);
    return x;
}

__device__ __forceinline__ float sigmoidf_(float x) { return 1.0f / (1.0f + expf(-x)); }

// One delta-rule step on this thread's 4 x 4 block of the state: per-channel decay, read (k), correct toward v.
__device__ __forceinline__ void update(float (&s)[4][4], const float (&kk)[4], const float (&gg)[4],
                                       const float* vrow, int warp, float beta) {
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        float kv = 0.0f;
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            s[j][i] = s[j][i] * gg[i];
            kv = kv + s[j][i] * kk[i];
        }
        kv = warp_sum(kv);
        const float delta = (vrow[warp * 4 + j] - kv) * beta;
#pragma unroll
        for (int i = 0; i < 4; ++i) s[j][i] = s[j][i] + kk[i] * delta;
    }
}

// How a state is stored: f32 (the reference's), or bf16 (the `_bf16state` entry points). The kernels compute in
// f32 either way; a bf16 state is rounded to bf16 after every row (after the row's read-out), so a window of R rows
// gives the bits of R serial single-row steps, each of which stores its state in bf16.
template <typename S>
struct StateIO;
template <>
struct StateIO<float> {
    static constexpr bool kRound = false;
    static __device__ __forceinline__ float load(const float* p) { return *p; }
    static __device__ __forceinline__ void store(float* p, float v) { *p = v; }
};
template <>
struct StateIO<__nv_bfloat16> {
    static constexpr bool kRound = true;
    static __device__ __forceinline__ float load(const __nv_bfloat16* p) { return __bfloat162float(*p); }
    static __device__ __forceinline__ void store(__nv_bfloat16* p, float v) { *p = __float2bfloat16_rn(v); }
};

// A bf16 state's rounding after a row (nothing for an f32 state).
template <typename S>
__device__ __forceinline__ void round_state(float (&s)[4][4]) {
    if constexpr (StateIO<S>::kRound) {
#pragma unroll
        for (int j = 0; j < 4; ++j)
#pragma unroll
            for (int i = 0; i < 4; ++i) s[j][i] = bf(s[j][i]);
    }
}

// Request b's rows: [row0, row0 + rows) of the shared row buffers. Without cu_rows: one request, `rows` rows.
struct Rows {
    long long row0;
    int rows;
};

__device__ __forceinline__ Rows request_rows(const int32_t* cu_rows, int b, int rows) {
    if (cu_rows == nullptr) return {0, rows};
    const int r0 = cu_rows[b];
    return {(long long)r0, cu_rows[b + 1] - r0};
}

// The state pointers are not __restrict__: a chain or replay may run in place (state_out == state_in). Each
// thread reads its 16 state values before the first row and writes the same 16 after the last.
template <typename S>
__global__ void __launch_bounds__(1024) chain_kernel(
        int H, const int32_t* __restrict__ cu_rows, int rows1,
        const __nv_bfloat16* __restrict__ P, long long p_stride, long long b_off,
        const __nv_bfloat16* __restrict__ A, long long a_stride, const __nv_bfloat16* __restrict__ G,
        long long g_stride, const __nv_bfloat16* __restrict__ conv, const long long* __restrict__ conv_off,
        const __nv_bfloat16* __restrict__ cw, const S* state_in, S* state_out,
        const long long* __restrict__ state_off, const float* __restrict__ a_log,
        const float* __restrict__ dt_bias, const __nv_bfloat16* __restrict__ norm_w, float eps, float lower,
        __nv_bfloat16* __restrict__ out, long long out_stride, float* __restrict__ k_save,
        __nv_bfloat16* __restrict__ v_save, float* __restrict__ g_save, float* __restrict__ b_save) {
    const int C = 3 * H * DK;                         // conv channels: q | k | v
    const int h = blockIdx.x, b = blockIdx.y;
    const int t = threadIdx.x, warp = t >> 5, lane = t & 31;
    const Rows rr = request_rows(cu_rows, b, rows1);
    const int rows = rr.rows;
    const __nv_bfloat16* cs = conv + (conv_off != nullptr ? conv_off[b] : 0);
    const long long soff = state_off != nullptr ? state_off[b] : 0;
    __shared__ float qs[DK], ks[DK], vs[DV], ys[DV], gs[DK];
    __shared__ float beta_s, rinv;
    int c = -1;
    if (t < 3 * DK) c = (t / DK) * H * DK + h * DK + (t % DK);
    float s[4][4];
    const long long sbase = soff + (long long)h * DV * DK;
#pragma unroll
    for (int j = 0; j < 4; ++j)
#pragma unroll
        for (int i = 0; i < 4; ++i) s[j][i] = StateIO<S>::load(state_in + sbase + (long long)(warp * 4 + j) * DK + lane * 4 + i);
    const float decay_rate = expf(a_log[h]);
    for (int r = 0; r < rows; ++r) {
        const long long gr = rr.row0 + r;             // this row in the shared row buffers
        if (c >= 0) {
            float acc = 0.0f;
#pragma unroll
            for (int tap = 0; tap < TAPS; ++tap) {
                const int at = r + tap;
                const float x = at < TAPS - 1
                                    ? __bfloat162float(cs[(long long)at * C + c])
                                    : __bfloat162float(P[(rr.row0 + at - (TAPS - 1)) * p_stride + c]);
                acc = acc + __bfloat162float(cw[(long long)c * TAPS + tap]) * x;
            }
            const float act = bf(acc / (1.0f + expf(-acc)));
            if (t < DK) qs[t] = act;
            else if (t < 2 * DK) ks[t - DK] = act;
            else vs[t - 2 * DK] = act;
        } else if (t >= 512 && t < 512 + DK) {
            const int i = t - 512;
            const float a = __bfloat162float(A[gr * a_stride + h * DK + i]) + dt_bias[h * DK + i];
            gs[i] = expf(lower * sigmoidf_(decay_rate * a));
        } else if (t == 1023) {
            beta_s = bf(sigmoidf_(__bfloat162float(P[gr * p_stride + b_off + h])));
        }
        __syncthreads();
        if (warp < 2) {
            float* x = warp == 0 ? qs : ks;
            float v4[4], ss = 0.0f;
#pragma unroll
            for (int i = 0; i < 4; ++i) { v4[i] = x[lane * 4 + i]; ss = ss + v4[i] * v4[i]; }
            ss = warp_sum(ss);
            float inv = 1.0f / sqrtf(ss + 1e-6f);
            __syncwarp();
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                float y = v4[i] * inv;
                if (warp == 0) y = y * (1.0f / sqrtf((float)DK));
                x[lane * 4 + i] = y;
            }
        }
        __syncthreads();
        const float beta = beta_s;
        float kk[4], qq[4], gg[4];
#pragma unroll
        for (int i = 0; i < 4; ++i) { kk[i] = ks[lane * 4 + i]; qq[i] = qs[lane * 4 + i]; gg[i] = gs[lane * 4 + i]; }
        update(s, kk, gg, vs, warp, beta);
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            float o = 0.0f;
#pragma unroll
            for (int i = 0; i < 4; ++i) o = o + s[j][i] * qq[i];
            o = warp_sum(o);
            if (lane == 0) ys[warp * 4 + j] = bf(o);
        }
        round_state<S>(s);
        if (k_save != nullptr) {
            const long long base = (gr * H + h) * DK;
            if (t < DK) { k_save[base + t] = ks[t]; g_save[base + t] = gs[t]; }
            if (t >= DK && t < DK + DV) v_save[base + t - DK] = __float2bfloat16_rn(vs[t - DK]);
            if (t == 0) b_save[gr * H + h] = beta;
        }
        __syncthreads();
        if (warp == 0) {
            float ss = 0.0f;
#pragma unroll
            for (int i = 0; i < 4; ++i) { const float y = ys[lane * 4 + i]; ss = ss + y * y; }
            ss = warp_sum(ss);
            if (lane == 0) rinv = 1.0f / sqrtf(ss / (float)DV + eps);
        }
        __syncthreads();
        if (t < DV) {
            const float yn = ys[t] * rinv;
            const float yw = __bfloat162float(norm_w[t]) * yn;
            const float gate = __bfloat162float(G[gr * g_stride + h * DV + t]);
            out[gr * out_stride + h * DV + t] = __float2bfloat16_rn(yw * sigmoidf_(gate));
        }
        __syncthreads();
    }
    if (state_out != nullptr) {
#pragma unroll
        for (int j = 0; j < 4; ++j)
#pragma unroll
            for (int i = 0; i < 4; ++i) StateIO<S>::store(state_out + sbase + (long long)(warp * 4 + j) * DK + lane * 4 + i, s[j][i]);
    }
}

// The source's replay and replay_layers in one kernel: block (layer * H + h, b) replays `rows` saved rows of
// request b (keep[b] with a batch) from its row cu_rows[b] on; per-layer strides of the states and the saves.
template <typename S>
__global__ void __launch_bounds__(1024) replay_kernel(
        int H, const int32_t* __restrict__ cu_rows, const int32_t* __restrict__ keep, int rows1,
        const S* state_in, S* state_out, long long state_stride, const long long* __restrict__ state_off,
        const float* __restrict__ k_save, const __nv_bfloat16* __restrict__ v_save,
        const float* __restrict__ g_save, const float* __restrict__ b_save, long long kv_stride,
        long long b_stride) {
    const int layer = blockIdx.x / H, h = blockIdx.x % H, b = blockIdx.y;
    const int t = threadIdx.x, warp = t >> 5, lane = t & 31;
    const long long row0 = cu_rows != nullptr ? cu_rows[b] : 0;
    const int rows = keep != nullptr ? keep[b] : rows1;
    const long long soff = (long long)layer * state_stride + (state_off != nullptr ? state_off[b] : 0);
    const float* ksv = k_save + layer * kv_stride;
    const __nv_bfloat16* vsv = v_save + layer * kv_stride;
    const float* gsv = g_save + layer * kv_stride;
    const float* bsv = b_save + layer * b_stride;
    __shared__ float vs[DV];
    float s[4][4];
    const long long sbase = soff + (long long)h * DV * DK;
#pragma unroll
    for (int j = 0; j < 4; ++j)
#pragma unroll
        for (int i = 0; i < 4; ++i) s[j][i] = StateIO<S>::load(state_in + sbase + (long long)(warp * 4 + j) * DK + lane * 4 + i);
    for (int r = 0; r < rows; ++r) {
        const long long gr = row0 + r;
        const long long base = (gr * H + h) * DK;
        if (t < DV) vs[t] = __bfloat162float(vsv[base + t]);
        __syncthreads();
        float kk[4], gg[4];
#pragma unroll
        for (int i = 0; i < 4; ++i) { kk[i] = ksv[base + lane * 4 + i]; gg[i] = gsv[base + lane * 4 + i]; }
        update(s, kk, gg, vs, warp, bsv[gr * H + h]);
        round_state<S>(s);
        __syncthreads();
    }
#pragma unroll
    for (int j = 0; j < 4; ++j)
#pragma unroll
        for (int i = 0; i < 4; ++i) StateIO<S>::store(state_out + sbase + (long long)(warp * 4 + j) * DK + lane * 4 + i, s[j][i]);
}

// The source's commit-time conv shift (a Triton kernel there): thread (channel c, layer, request) sets the 3
// window rows to rows keep .. keep + 2 of [old window; new rows]. Each thread reads its three values before it
// writes them, so the shift runs in place.
__global__ void conv_shift_kernel(int C, const int32_t* __restrict__ cu_rows, const int32_t* __restrict__ keep,
                                  int keep1, __nv_bfloat16* conv, long long conv_stride,
                                  const long long* __restrict__ conv_off, const __nv_bfloat16* P,
                                  long long p_layer_stride, long long p_stride) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= C) return;
    const int layer = blockIdx.y, b = blockIdx.z;
    const long long row0 = cu_rows != nullptr ? cu_rows[b] : 0;
    const int k = keep != nullptr ? keep[b] : keep1;
    __nv_bfloat16* win = conv + (long long)layer * conv_stride + (conv_off != nullptr ? conv_off[b] : 0);
    const __nv_bfloat16* rows = P + (long long)layer * p_layer_stride + row0 * p_stride;
    __nv_bfloat16 v[TAPS - 1];
#pragma unroll
    for (int j = 0; j < TAPS - 1; ++j) {
        const int src = k + j;
        v[j] = src < TAPS - 1 ? win[(long long)src * C + c] : rows[(long long)(src - (TAPS - 1)) * p_stride + c];
    }
#pragma unroll
    for (int j = 0; j < TAPS - 1; ++j) win[(long long)j * C + c] = v[j];
}

// cudaErrorInvalidValue.
constexpr int kInvalid = 1;

inline int launched() { return (int)cudaGetLastError(); }

template <typename T>
inline const __nv_bfloat16* bfp(const T* x) { return reinterpret_cast<const __nv_bfloat16*>(x); }
template <typename T>
inline __nv_bfloat16* bfp(T* x) { return reinterpret_cast<__nv_bfloat16*>(x); }

// Host-side checks shared by the chain entry points.
inline bool chain_args_ok(int32_t heads, const glm53f_bf16* p, const glm53f_bf16* a, const glm53f_bf16* g,
                          const glm53f_bf16* conv, const glm53f_bf16* conv_w, const void* state_in,
                          const float* a_log, const float* dt_bias, const glm53f_bf16* norm_w,
                          const glm53f_bf16* out, const float* k_save, const glm53f_bf16* v_save,
                          const float* g_save, const float* b_save) {
    if (heads < 1 || heads > 65535) return false;
    if (!p || !a || !g || !conv || !conv_w || !state_in || !a_log || !dt_bias || !norm_w || !out) return false;
    const int saves = (k_save != nullptr) + (v_save != nullptr) + (g_save != nullptr) + (b_save != nullptr);
    return saves == 0 || saves == 4;
}

}  // namespace

extern "C" int glm53f_kda_chain(int32_t heads, int32_t rows, const glm53f_bf16* p, int64_t p_stride,
                                int64_t b_off, const glm53f_bf16* a, int64_t a_stride, const glm53f_bf16* g,
                                int64_t g_stride, const glm53f_bf16* conv, const glm53f_bf16* conv_w,
                                const float* state_in, float* state_out, const float* a_log,
                                const float* dt_bias, const glm53f_bf16* norm_w, float eps, float lower,
                                glm53f_bf16* out, int64_t out_stride, float* k_save, glm53f_bf16* v_save,
                                float* g_save, float* b_save, glm53f_stream_t stream) {
    if (rows < 1 || !chain_args_ok(heads, p, a, g, conv, conv_w, state_in, a_log, dt_bias, norm_w, out, k_save,
                                   v_save, g_save, b_save))
        return kInvalid;
    chain_kernel<float><<<dim3(heads, 1), 1024, 0, (cudaStream_t)stream>>>(
        heads, nullptr, rows, bfp(p), p_stride, b_off, bfp(a), a_stride, bfp(g), g_stride, bfp(conv), nullptr,
        bfp(conv_w), state_in, state_out, nullptr, a_log, dt_bias, bfp(norm_w), eps, lower, bfp(out), out_stride,
        k_save, bfp(v_save), g_save, b_save);
    return launched();
}

namespace {

template <typename S>
int chain_batch_launch(int32_t heads, int32_t batch, const int32_t* cu_rows, const glm53f_bf16* p, int64_t p_stride,
                       int64_t b_off, const glm53f_bf16* a, int64_t a_stride, const glm53f_bf16* g, int64_t g_stride,
                       const glm53f_bf16* conv, const int64_t* conv_off, const glm53f_bf16* conv_w, const S* state_in,
                       S* state_out, const int64_t* state_off, const float* a_log, const float* dt_bias,
                       const glm53f_bf16* norm_w, float eps, float lower, glm53f_bf16* out, int64_t out_stride,
                       float* k_save, glm53f_bf16* v_save, float* g_save, float* b_save, glm53f_stream_t stream) {
    if (batch < 1 || batch > 65535 || !cu_rows || !conv_off || !state_off ||
        !chain_args_ok(heads, p, a, g, conv, conv_w, state_in, a_log, dt_bias, norm_w, out, k_save, v_save, g_save,
                       b_save))
        return kInvalid;
    chain_kernel<S><<<dim3(heads, batch), 1024, 0, (cudaStream_t)stream>>>(
        heads, cu_rows, 0, bfp(p), p_stride, b_off, bfp(a), a_stride, bfp(g), g_stride, bfp(conv),
        reinterpret_cast<const long long*>(conv_off), bfp(conv_w), state_in, state_out,
        reinterpret_cast<const long long*>(state_off), a_log, dt_bias, bfp(norm_w), eps, lower, bfp(out),
        out_stride, k_save, bfp(v_save), g_save, b_save);
    return launched();
}

}  // namespace

extern "C" int glm53f_kda_chain_batch(int32_t heads, int32_t batch, const int32_t* cu_rows, const glm53f_bf16* p,
                                      int64_t p_stride, int64_t b_off, const glm53f_bf16* a, int64_t a_stride,
                                      const glm53f_bf16* g, int64_t g_stride, const glm53f_bf16* conv,
                                      const int64_t* conv_off, const glm53f_bf16* conv_w, const float* state_in,
                                      float* state_out, const int64_t* state_off, const float* a_log,
                                      const float* dt_bias, const glm53f_bf16* norm_w, float eps, float lower,
                                      glm53f_bf16* out, int64_t out_stride, float* k_save, glm53f_bf16* v_save,
                                      float* g_save, float* b_save, glm53f_stream_t stream) {
    return chain_batch_launch<float>(heads, batch, cu_rows, p, p_stride, b_off, a, a_stride, g, g_stride, conv,
                                     conv_off, conv_w, state_in, state_out, state_off, a_log, dt_bias, norm_w, eps,
                                     lower, out, out_stride, k_save, v_save, g_save, b_save, stream);
}

extern "C" int glm53f_kda_chain_batch_bf16state(int32_t heads, int32_t batch, const int32_t* cu_rows,
                                                const glm53f_bf16* p, int64_t p_stride, int64_t b_off,
                                                const glm53f_bf16* a, int64_t a_stride, const glm53f_bf16* g,
                                                int64_t g_stride, const glm53f_bf16* conv, const int64_t* conv_off,
                                                const glm53f_bf16* conv_w, const glm53f_bf16* state_in,
                                                glm53f_bf16* state_out, const int64_t* state_off,
                                                const float* a_log, const float* dt_bias,
                                                const glm53f_bf16* norm_w, float eps, float lower,
                                                glm53f_bf16* out, int64_t out_stride, float* k_save,
                                                glm53f_bf16* v_save, float* g_save, float* b_save,
                                                glm53f_stream_t stream) {
    return chain_batch_launch<__nv_bfloat16>(heads, batch, cu_rows, p, p_stride, b_off, a, a_stride, g, g_stride,
                                             conv, conv_off, conv_w, bfp(state_in), bfp(state_out), state_off,
                                             a_log, dt_bias, norm_w, eps, lower, out, out_stride, k_save, v_save,
                                             g_save, b_save, stream);
}

namespace {

template <typename S>
inline int replay_launch(int32_t heads, int32_t layers, int32_t batch, const int32_t* cu_rows, const int32_t* keep,
                         int32_t rows, const S* state_in, S* state_out, int64_t state_stride,
                         const int64_t* state_off, const float* k_save, const glm53f_bf16* v_save,
                         const float* g_save, const float* b_save, int64_t kv_stride, int64_t b_stride,
                         glm53f_stream_t stream) {
    if (heads < 1 || layers < 1 || batch < 1 || batch > 65535 || (long long)heads * layers > 0x7fffffffLL ||
        !state_in || !state_out || !k_save || !v_save || !g_save || !b_save || rows < 0)
        return kInvalid;
    replay_kernel<S><<<dim3(heads * layers, batch), 1024, 0, (cudaStream_t)stream>>>(
        heads, cu_rows, keep, rows, state_in, state_out, state_stride,
        reinterpret_cast<const long long*>(state_off), k_save, bfp(v_save), g_save, b_save, kv_stride, b_stride);
    return launched();
}

}  // namespace

extern "C" int glm53f_kda_replay(int32_t heads, int32_t rows, const float* state_in, float* state_out,
                                 const float* k_save, const glm53f_bf16* v_save, const float* g_save,
                                 const float* b_save, glm53f_stream_t stream) {
    return replay_launch(heads, 1, 1, nullptr, nullptr, rows, state_in, state_out, 0, nullptr, k_save, v_save,
                         g_save, b_save, 0, 0, stream);
}

extern "C" int glm53f_kda_replay_layers(int32_t heads, int32_t layers, int32_t rows, const float* state_in,
                                        float* state_out, int64_t state_stride, const float* k_save,
                                        const glm53f_bf16* v_save, const float* g_save, const float* b_save,
                                        int64_t kv_stride, int64_t b_stride, glm53f_stream_t stream) {
    return replay_launch(heads, layers, 1, nullptr, nullptr, rows, state_in, state_out, state_stride, nullptr,
                         k_save, v_save, g_save, b_save, kv_stride, b_stride, stream);
}

extern "C" int glm53f_kda_replay_batch(int32_t heads, int32_t layers, int32_t batch, const int32_t* cu_rows,
                                       const int32_t* keep, const float* state_in, float* state_out,
                                       int64_t state_stride, const int64_t* state_off, const float* k_save,
                                       const glm53f_bf16* v_save, const float* g_save, const float* b_save,
                                       int64_t kv_stride, int64_t b_stride, glm53f_stream_t stream) {
    if (!cu_rows || !keep || !state_off) return kInvalid;
    return replay_launch(heads, layers, batch, cu_rows, keep, 0, state_in, state_out, state_stride, state_off,
                         k_save, v_save, g_save, b_save, kv_stride, b_stride, stream);
}

extern "C" int glm53f_kda_replay_batch_bf16state(int32_t heads, int32_t layers, int32_t batch,
                                                 const int32_t* cu_rows, const int32_t* keep,
                                                 const glm53f_bf16* state_in, glm53f_bf16* state_out,
                                                 int64_t state_stride, const int64_t* state_off,
                                                 const float* k_save, const glm53f_bf16* v_save,
                                                 const float* g_save, const float* b_save, int64_t kv_stride,
                                                 int64_t b_stride, glm53f_stream_t stream) {
    if (!cu_rows || !keep || !state_off) return kInvalid;
    return replay_launch(heads, layers, batch, cu_rows, keep, 0, bfp(state_in), bfp(state_out), state_stride,
                         state_off, k_save, v_save, g_save, b_save, kv_stride, b_stride, stream);
}

namespace {

constexpr int kShiftThreads = 256;

inline int conv_shift_launch(int32_t channels, int32_t layers, int32_t batch, const int32_t* cu_rows,
                             const int32_t* keep, int32_t keep1, glm53f_bf16* conv, int64_t conv_stride,
                             const int64_t* conv_off, const glm53f_bf16* p, int64_t p_layer_stride,
                             int64_t p_stride, glm53f_stream_t stream) {
    if (channels < 1 || layers < 1 || layers > 65535 || batch < 1 || batch > 65535 || !conv || !p || keep1 < 0)
        return kInvalid;
    const dim3 grid((channels + kShiftThreads - 1) / kShiftThreads, layers, batch);
    conv_shift_kernel<<<grid, kShiftThreads, 0, (cudaStream_t)stream>>>(
        channels, cu_rows, keep, keep1, bfp(conv), conv_stride, reinterpret_cast<const long long*>(conv_off),
        bfp(p), p_layer_stride, p_stride);
    return launched();
}

}  // namespace

extern "C" int glm53f_kda_conv_shift(int32_t channels, int32_t keep, glm53f_bf16* conv, const glm53f_bf16* p,
                                     int64_t p_stride, glm53f_stream_t stream) {
    return conv_shift_launch(channels, 1, 1, nullptr, nullptr, keep, conv, 0, nullptr, p, 0, p_stride, stream);
}

extern "C" int glm53f_kda_conv_shift_batch(int32_t channels, int32_t layers, int32_t batch, const int32_t* cu_rows,
                                           const int32_t* keep, glm53f_bf16* conv, int64_t conv_stride,
                                           const int64_t* conv_off, const glm53f_bf16* p, int64_t p_layer_stride,
                                           int64_t p_stride, glm53f_stream_t stream) {
    if (!cu_rows || !keep || !conv_off) return kInvalid;
    return conv_shift_launch(channels, layers, batch, cu_rows, keep, 0, conv, conv_stride, conv_off, p,
                             p_layer_stride, p_stride, stream);
}
