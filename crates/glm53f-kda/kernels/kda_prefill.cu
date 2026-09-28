// SPDX-License-Identifier: MIT
//
// GLM-5.3-Flash KDA prefill: the delta rule over a prompt segment in the chunked (WY/UT) form. New code; the
// algebra follows the reference's chunk_kimi_delta_attention and flash-linear-attention's chunk_kda (see
// PROVENANCE.md). src/chunked.rs (`prefill_head`) is its host model.
//
// Per chunk of CH = 16 rows (n <= CH real rows at a segment's end):
//   1. prologue, the chain's per-row arithmetic with its bits: conv + SiLU (one bf16 rounding), q/k L2 norms,
//      the decay multipliers d = exp(lower * sigmoid(exp(A_log) * (a + dt_bias))), beta = bf16(sigmoid(b));
//   2. decay products from the chunk's first row, Lf_i = d_1 ... d_i (the -5 lower bound keeps them within
//      e^+-75 over 16 rows), and A = k * Lf, A' = q * Lf, B = k / Lf, so that exp(G_i - G_j) = Lf_i / Lf_j;
//   3. L[i][j] = beta_i A_i . B_j (j < i), M[i][j] = A'_i . B_j (j <= i);
//   4. T' = (I + L)^-1 diag(beta) by forward substitution;
//   5. W = d_0 * T'A, U = T'V, Qg = d_0 * A';
//   6. from the state S entering the chunk: D = U - W S, Y = Qg S + M D,
//      S <- d_0 Lf_last * S + Lf_last * (B^T D);
//   7. y = bf16(Y), then the chain's gated RMSNorm.
//
// Two passes, repeated over sub-segments of rows as large as the workspace between them allows (each
// repetition costs a round trip of the state and a fixed overhead):
//   intra_kernel, one block per (chunk, head, request), steps 1-5 (nothing there needs the state), written to
//     the workspace;
//   inter_kernel, one block per (head, value-column block, request), the state in registers, the chunks in
//     order: each chunk's workspace is staged into shared memory one chunk ahead (cp.async), then steps 6-7.
// Built with the chain's flags (--fmad=false), so the prologue has the chain's bits; the products use fmaf.
// In step 6 thread (v, kp) owns S[k][v] for its KW = 128 / KP keys k: KP threads per value column,
// VB = 256 / KP value columns per block, 128 / VB blocks per head.

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <stdint.h>

#include "glm53f_kda.h"

namespace {

constexpr int DK = 128, DV = 128, TAPS = 4, CH = 16, THREADS = 256;
constexpr unsigned FULL = 0xffffffffu;

// Workspace per (chunk, head, request), in floats: W [CH][DK], Qg [CH][DK], U [CH][DV], B^T [DK][CH],
// M [CH][CH], elast = d_0 Lf_last [DK], last = Lf_last [DK].
constexpr int WS_W = 0, WS_Q = WS_W + CH * DK, WS_U = WS_Q + CH * DK, WS_B = WS_U + CH * DV;
constexpr int WS_M = WS_B + DK * CH, WS_E = WS_M + CH * CH, WS_L = WS_E + DK, WS_FLOATS = WS_L + DK;

__device__ __forceinline__ float bf(float x) { return __bfloat162float(__float2bfloat16_rn(x)); }
__device__ __forceinline__ float b2f(__nv_bfloat16 x) { return __bfloat162float(x); }

__device__ __forceinline__ float warp_sum(float x) {
    for (int o = 16; o; o >>= 1) x += __shfl_xor_sync(FULL, x, o);
    return x;
}

__device__ __forceinline__ float sigmoidf_(float x) { return 1.0f / (1.0f + expf(-x)); }

__device__ __forceinline__ void cp_async16(float* smem, const float* gmem) {
    const unsigned s = (unsigned)__cvta_generic_to_shared(smem);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(s), "l"(gmem));
}
__device__ __forceinline__ void cp_async_commit() { asm volatile("cp.async.commit_group;\n" ::); }
template <int N>
__device__ __forceinline__ void cp_async_wait() {
    asm volatile("cp.async.wait_group %0;\n" ::"n"(N));
}

struct Rows {
    long long row0;
    int rows;
};

__device__ __forceinline__ Rows request_rows(const int32_t* cu_rows, int b, int rows1) {
    if (cu_rows == nullptr) return {0, rows1};
    const int r0 = cu_rows[b];
    return {(long long)r0, cu_rows[b + 1] - r0};
}

// Pass 1: one chunk of one head of one request, steps 1-5, into its workspace slot.
__global__ void __launch_bounds__(THREADS) intra_kernel(
        int H, const int32_t* __restrict__ cu_rows, int rows1, int sub0, const __nv_bfloat16* __restrict__ P,
        long long p_stride, long long b_off, const __nv_bfloat16* __restrict__ A, long long a_stride,
        const __nv_bfloat16* __restrict__ conv, const long long* __restrict__ conv_off,
        const __nv_bfloat16* __restrict__ cw, const float* __restrict__ a_log, const float* __restrict__ dt_bias,
        float lower, float* __restrict__ ws) {
    __shared__ __align__(16) float kq[CH * DK];   // q_raw -> q -> A'
    __shared__ __align__(16) float ka[CH * DK];   // k_raw -> k -> A
    __shared__ __align__(16) float kbt[DK * CH];  // B^T [k][i]
    __shared__ __align__(16) float vv[CH * DV];   // v
    __shared__ float dd[CH * DK];                 // decay multipliers
    __shared__ float lt[CH * CH];                 // L -> T'
    __shared__ float d0s[DK];
    __shared__ float betas[CH];

    const int c = blockIdx.x, h = blockIdx.y, b = blockIdx.z;
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const Rows rr = request_rows(cu_rows, b, rows1);
    const int c0 = sub0 + c * CH;  // this chunk's first row within the request
    if (c0 >= rr.rows) return;
    const int n = min(CH, rr.rows - c0);
    const long long rbase = rr.row0 + c0;
    const int C = 3 * H * DK;
    const __nv_bfloat16* cs = conv + (conv_off != nullptr ? conv_off[b] : 0);
    float* w = ws + ((long long)(b * gridDim.x + c) * H + h) * WS_FLOATS;

    // 1. Prologue: every global load first (per channel the three rows before the chunk, from the conv
    // window before the segment, else segment rows; the chunk's rows; the gate inputs; beta's logit).
    constexpr int NCH = 3 * DK, CPT = NCH / THREADS + (NCH % THREADS != 0), GPT = CH * DK / THREADS;
    constexpr int XN = TAPS - 1 + CH;
    const __nv_bfloat16 zero = __float2bfloat16_rn(0.0f);
    __nv_bfloat16 xin[CPT][XN], win[CPT][TAPS];
#pragma unroll
    for (int u = 0; u < CPT; ++u) {
        const int ch = tid + u * THREADS;
        if (ch < NCH) {
            const long long cc = (long long)(ch / DK) * H * DK + h * DK + ch % DK;
#pragma unroll
            for (int t = 0; t < TAPS; ++t) win[u][t] = cw[cc * TAPS + t];
#pragma unroll
            for (int t = 0; t < XN; ++t) {
                const int rho = c0 - (TAPS - 1) + t;
                xin[u][t] = t >= TAPS - 1 && t - (TAPS - 1) >= n ? zero
                            : rho < 0 ? cs[(long long)(TAPS - 1 + rho) * C + cc]
                                      : P[(rr.row0 + rho) * p_stride + cc];
            }
        }
    }
    __nv_bfloat16 ain[GPT];
#pragma unroll
    for (int q = 0; q < GPT; ++q) {
        const int r = tid / DK + q * (THREADS / DK);
        ain[q] = r < n ? A[(rbase + r) * a_stride + h * DK + tid % DK] : zero;
    }
    const __nv_bfloat16 blogit = tid < n ? P[(rbase + tid) * p_stride + b_off + h] : zero;
    const float dtb = dt_bias[h * DK + tid % DK];
    const float decay_rate = expf(a_log[h]);
    // Conv + SiLU, the chain's arithmetic: acc = 0 + w0 x0 + w1 x1 + w2 x2 + w3 x3, one bf16 rounding.
#pragma unroll
    for (int u = 0; u < CPT; ++u) {
        const int ch = tid + u * THREADS;
        if (ch < NCH) {
            const int part = ch / DK, idx = ch % DK;
            const float w0 = b2f(win[u][0]), w1 = b2f(win[u][1]), w2 = b2f(win[u][2]), w3 = b2f(win[u][3]);
            float* dst = part == 0 ? kq : (part == 1 ? ka : vv);
#pragma unroll
            for (int r = 0; r < CH; ++r) {
                float act = 0.0f;
                if (r < n) {
                    float acc = 0.0f;
                    acc = acc + w0 * b2f(xin[u][r]);
                    acc = acc + w1 * b2f(xin[u][r + 1]);
                    acc = acc + w2 * b2f(xin[u][r + 2]);
                    acc = acc + w3 * b2f(xin[u][r + 3]);
                    act = bf(acc / (1.0f + expf(-acc)));
                }
                dst[r * DK + idx] = act;
            }
        }
    }
    // Decay multipliers and beta; padding rows decay by 1 and have beta 0.
#pragma unroll
    for (int q = 0; q < GPT; ++q) {
        const int r = tid / DK + q * (THREADS / DK);
        float dv = 1.0f;
        if (r < n) {
            const float a = b2f(ain[q]) + dtb;
            dv = expf(lower * sigmoidf_(decay_rate * a));
        }
        dd[r * DK + tid % DK] = dv;
    }
    if (tid < CH) betas[tid] = tid < n ? bf(sigmoidf_(b2f(blogit))) : 0.0f;
    __syncthreads();

    // L2 norms, in the chain's order: warp per (row, q or k), lane l owns keys 4l .. 4l + 3.
    for (int task = warp; task < 2 * CH; task += THREADS / 32) {
        const int r = task >> 1;
        const bool isq = (task & 1) == 0;
        float* xr = (isq ? kq : ka) + r * DK;
        float v4[4], ss = 0.0f;
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            v4[i] = xr[lane * 4 + i];
            ss = ss + v4[i] * v4[i];
        }
        ss = warp_sum(ss);
        const float inv = 1.0f / sqrtf(ss + 1e-6f);
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            float y = v4[i] * inv;
            if (isq) y = y * (1.0f / sqrtf((float)DK));
            xr[lane * 4 + i] = y;
        }
    }
    __syncthreads();

    // 2. Decay products per key channel, and A, A', B^T. lo carries the running product's rounding error
    // (exact two-product with fmaf), so every product is rounded once, from lf + lo. Products of multipliers
    // just below 1 tend to round the same way, and the chunk's whole decay reaches the state at every chunk:
    // rounded once per row, that bias would accumulate over a slowly decaying channel's memory.
    if (tid < DK) {
        const int k = tid;
        float lf = 1.0f, lo = 0.0f;
#pragma unroll
        for (int i = 0; i < CH; ++i) {
            if (i > 0) {
                const float d = dd[i * DK + k];
                const float p = lf * d;
                lo = fmaf(lo, d, fmaf(lf, d, -p));
                lf = p;
            }
            const float li = lf + lo;
            const float kn = ka[i * DK + k], qn = kq[i * DK + k];
            ka[i * DK + k] = kn * li;
            kq[i * DK + k] = qn * li;
            kbt[k * CH + i] = kn / li;
        }
        d0s[k] = dd[k];
        w[WS_E + k] = fmaf(dd[k], lf, dd[k] * lo);
        w[WS_L + k] = lf + lo;
    }
    __syncthreads();

    // 3. L and M: thread (i, j), dot products over the keys with four partial sums.
    {
        const int i = tid / CH, j = tid % CH;
        float l4[4] = {0.0f, 0.0f, 0.0f, 0.0f}, m4[4] = {0.0f, 0.0f, 0.0f, 0.0f};
#pragma unroll 8
        for (int k = 0; k < DK; k += 4) {
            const float4 a = *reinterpret_cast<const float4*>(&ka[i * DK + k]);
            const float4 q = *reinterpret_cast<const float4*>(&kq[i * DK + k]);
            const float b0 = kbt[k * CH + j], b1 = kbt[(k + 1) * CH + j];
            const float b2 = kbt[(k + 2) * CH + j], b3 = kbt[(k + 3) * CH + j];
            l4[0] = fmaf(a.x, b0, l4[0]);
            l4[1] = fmaf(a.y, b1, l4[1]);
            l4[2] = fmaf(a.z, b2, l4[2]);
            l4[3] = fmaf(a.w, b3, l4[3]);
            m4[0] = fmaf(q.x, b0, m4[0]);
            m4[1] = fmaf(q.y, b1, m4[1]);
            m4[2] = fmaf(q.z, b2, m4[2]);
            m4[3] = fmaf(q.w, b3, m4[3]);
        }
        const float sl = (l4[0] + l4[1]) + (l4[2] + l4[3]);
        const float sm = (m4[0] + m4[1]) + (m4[2] + m4[3]);
        w[WS_M + i * CH + j] = j <= i ? sm : 0.0f;
        lt[i * CH + j] = j < i ? betas[i] * sl : 0.0f;
    }
    __syncthreads();

    // 4. T = (I + L)^-1 by forward substitution: lane j of warp 0 keeps column j of T in registers.
    if (warp == 0) {
        const int j = lane;
        float tc[CH];
#pragma unroll
        for (int m = 0; m < CH; ++m) tc[m] = m == j ? 1.0f : 0.0f;
#pragma unroll
        for (int i = 1; i < CH; ++i) {
            float acc = lt[i * CH + (j < CH ? j : 0)];
#pragma unroll
            for (int m = 1; m < CH - 1; ++m)
                if (m > j && m < i) acc = fmaf(lt[i * CH + m], tc[m], acc);
            if (j < i) tc[i] = -acc;
        }
        __syncwarp();
        if (j < CH) {
            const float bj = betas[j];
#pragma unroll
            for (int m = 0; m < CH; ++m) lt[m * CH + j] = tc[m] * bj;  // T' = T diag(beta)
        }
    }
    __syncthreads();

    // 5. W = d0 * T'A and Qg = d0 * A' (threads 0..127, key column k); U = T'V (threads 128..255, value
    // column); then B^T.
    if (tid < DK) {
        const int k = tid;
        const float d0 = d0s[k];
        float acol[CH];
#pragma unroll
        for (int j = 0; j < CH; ++j) acol[j] = ka[j * DK + k];
#pragma unroll
        for (int i = 0; i < CH; ++i) {
            float acc = 0.0f;
#pragma unroll
            for (int j = 0; j <= i; ++j) acc = fmaf(lt[i * CH + j], acol[j], acc);
            w[WS_W + i * DK + k] = d0 * acc;
            w[WS_Q + i * DK + k] = kq[i * DK + k] * d0;
        }
    } else {
        const int v = tid - DK;
        float vcol[CH];
#pragma unroll
        for (int j = 0; j < CH; ++j) vcol[j] = vv[j * DV + v];
#pragma unroll
        for (int i = 0; i < CH; ++i) {
            float acc = 0.0f;
#pragma unroll
            for (int j = 0; j <= i; ++j) acc = fmaf(lt[i * CH + j], vcol[j], acc);
            w[WS_U + i * DV + v] = acc;
        }
    }
    for (int e = tid; e < DK * CH / 4; e += THREADS)
        reinterpret_cast<float4*>(w + WS_B)[e] = reinterpret_cast<const float4*>(kbt)[e];
}

// Shared-memory geometry of inter_kernel: KP threads per value-column group, COLS columns per thread (each
// value loaded from shared memory feeds COLS multiply-adds). K rows are split into KP segments of KW keys,
// each followed by 4 pad floats, so the KP threads of a group read distinct banks.
template <int KP, int COLS>
struct Geo {
    static constexpr int KW = DK / KP;
    static constexpr int VB = THREADS / KP * COLS;
    static constexpr int KROW = DK + 4 * KP;
    static constexpr int BT = DK * CH + 4 * KP;
    // One stage: W, Qg (padded rows), B^T (padded), U (this block's columns), M, elast, last.
    static constexpr int OW = 0, OQ = OW + CH * KROW, OB = OQ + CH * KROW, OU = OB + BT, OM = OU + CH * VB;
    static constexpr int OE = OM + CH * CH, OL = OE + DK, STAGE = OL + DK;
    static constexpr int YS = 2 * STAGE;  // bf16(Y) for this block's columns
    static constexpr int FLOATS = YS + CH * VB;
    __device__ static __forceinline__ int kx(int k) { return k + 4 * (k / KW); }
    __device__ static __forceinline__ int bx(int k, int i) { return k * CH + i + 4 * (k / KW); }
};

// Copy one chunk's workspace into a stage (16-byte cp.async; every offset is a multiple of four floats).
template <typename Gm>
__device__ __forceinline__ void stage_chunk(float* st, const float* src, int vb0, int tid) {
    for (int e = tid; e < CH * DK / 4; e += THREADS) {
        const int i = e / (DK / 4), k = (e % (DK / 4)) * 4;
        cp_async16(st + Gm::OW + i * Gm::KROW + Gm::kx(k), src + WS_W + i * DK + k);
        cp_async16(st + Gm::OQ + i * Gm::KROW + Gm::kx(k), src + WS_Q + i * DK + k);
    }
    for (int e = tid; e < DK * CH / 4; e += THREADS) {
        const int k = e / (CH / 4), i = (e % (CH / 4)) * 4;
        cp_async16(st + Gm::OB + Gm::bx(k, i), src + WS_B + k * CH + i);
    }
    for (int e = tid; e < CH * Gm::VB / 4; e += THREADS) {
        const int i = e / (Gm::VB / 4), v = (e % (Gm::VB / 4)) * 4;
        cp_async16(st + Gm::OU + i * Gm::VB + v, src + WS_U + i * DV + vb0 + v);
    }
    for (int e = tid; e < (CH * CH + 2 * DK) / 4; e += THREADS) cp_async16(st + Gm::OM + 4 * e, src + WS_M + 4 * e);
}

// Partial products of one chunk matrix (W or Qg, padded rows) with this thread's state: acc[c][i] =
// sum over this thread's keys of X[i][k] S[k][v_c], then summed over the KP lanes of the group.
template <typename Gm, int COLS>
__device__ __forceinline__ void chunk_times_state(const float* x, const float (&s)[COLS][Gm::KW], int kp,
                                                  float (&acc)[COLS][CH]) {
#pragma unroll
    for (int c = 0; c < COLS; ++c)
#pragma unroll
        for (int i = 0; i < CH; ++i) acc[c][i] = 0.0f;
#pragma unroll
    for (int j = 0; j < Gm::KW; j += 4) {
        const int kk = Gm::kx(kp * Gm::KW + j);
#pragma unroll
        for (int i = 0; i < CH; ++i) {
            const float4 xv = *reinterpret_cast<const float4*>(&x[i * Gm::KROW + kk]);
#pragma unroll
            for (int c = 0; c < COLS; ++c) {
                acc[c][i] = fmaf(xv.x, s[c][j], acc[c][i]);
                acc[c][i] = fmaf(xv.y, s[c][j + 1], acc[c][i]);
                acc[c][i] = fmaf(xv.z, s[c][j + 2], acc[c][i]);
                acc[c][i] = fmaf(xv.w, s[c][j + 3], acc[c][i]);
            }
        }
    }
    constexpr int KP = DK / Gm::KW;
#pragma unroll
    for (int o = 1; o < KP; o <<= 1)
#pragma unroll
        for (int c = 0; c < COLS; ++c)
#pragma unroll
            for (int i = 0; i < CH; ++i) acc[c][i] += __shfl_xor_sync(FULL, acc[c][i], o);
}

// Pass 2: the chunks of one sub-segment in order, for one (head, value-column block, request).
template <int KP, int COLS>
__global__ void __launch_bounds__(THREADS) inter_kernel(
        int H, const int32_t* __restrict__ cu_rows, int rows1, int sub0, int chunks,
        const __nv_bfloat16* __restrict__ G, long long g_stride, const float* state_in, float* state_out,
        const long long* __restrict__ state_off, const __nv_bfloat16* __restrict__ norm_w, float eps,
        __nv_bfloat16* __restrict__ out, long long out_stride, const float* __restrict__ ws) {
    using Gm = Geo<KP, COLS>;
    constexpr int KW = Gm::KW, VB = Gm::VB, NB = DV / VB;
    extern __shared__ __align__(16) float smem[];
    float* ys = smem + Gm::YS;

    const int h = blockIdx.x / NB, vb0 = (blockIdx.x % NB) * VB, b = blockIdx.y;
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const Rows rr = request_rows(cu_rows, b, rows1);
    const long long sbase = (state_off != nullptr ? state_off[b] : 0) + (long long)h * DV * DK;
    // Thread (group, kp): value columns vg * COLS .. vg * COLS + COLS - 1 of the block, keys kp * KW ...
    const int vg = warp * (32 / KP) + lane / KP, kp = lane % KP;
    const int here = min(chunks * CH, rr.rows - sub0);  // this request's rows in the sub-segment
    // A request with no rows here keeps its state (copied on the first pass when out and in differ).
    if (here <= 0 && !(sub0 == 0 && state_out != state_in)) return;
    float s[COLS][KW];
#pragma unroll
    for (int c = 0; c < COLS; ++c)
#pragma unroll
        for (int j = 0; j < KW; ++j) s[c][j] = state_in[sbase + (long long)(vb0 + vg * COLS + c) * DK + kp * KW + j];
    const int nch = here > 0 ? (here + CH - 1) / CH : 0;
    const float* wsb = ws + (long long)b * chunks * H * WS_FLOATS + (long long)h * WS_FLOATS;
    float nw[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) nw[i] = b2f(norm_w[lane * 4 + i]);
    if (nch > 0) stage_chunk<Gm>(smem, wsb, vb0, tid);
    cp_async_commit();

    for (int c = 0; c < nch; ++c) {
        const int n = min(CH, here - c * CH);
        const long long rbase = rr.row0 + sub0 + c * CH;
        if (c + 1 < nch) stage_chunk<Gm>(smem + ((c + 1) & 1) * Gm::STAGE, wsb + (long long)(c + 1) * H * WS_FLOATS, vb0, tid);
        cp_async_commit();
        // The output gates, needed at the end of the chunk.
        __nv_bfloat16 gin[CH / (THREADS / 32)][4];
        if (NB == 1) {
#pragma unroll
            for (int u = 0; u < CH / (THREADS / 32); ++u) {
                const int r = warp + u * (THREADS / 32);
#pragma unroll
                for (int i = 0; i < 4; ++i)
                    gin[u][i] = r < n ? G[(rbase + r) * g_stride + h * DV + lane * 4 + i] : __float2bfloat16_rn(0.0f);
            }
        }
        cp_async_wait<1>();
        __syncthreads();
        const float* st = smem + (c & 1) * Gm::STAGE;
        const float* su = st + Gm::OU;
        const float* sm = st + Gm::OM;
        const float* sb = st + Gm::OB;
        const float* se = st + Gm::OE;
        const float* sl = st + Gm::OL;

        // 6. D = U - W S, then Y = Qg S + M D.
        float dl[COLS][CH];
        {
            float acc[COLS][CH];
            chunk_times_state<Gm, COLS>(st + Gm::OW, s, kp, acc);
#pragma unroll
            for (int cc = 0; cc < COLS; ++cc)
#pragma unroll
                for (int i = 0; i < CH; ++i) dl[cc][i] = su[i * VB + vg * COLS + cc] - acc[cc][i];
            chunk_times_state<Gm, COLS>(st + Gm::OQ, s, kp, acc);
            if (kp == 0) {
#pragma unroll
                for (int cc = 0; cc < COLS; ++cc)
#pragma unroll
                    for (int i = 0; i < CH; ++i) {
                        float y = acc[cc][i];
#pragma unroll
                        for (int j = 0; j <= i; ++j) y = fmaf(sm[i * CH + j], dl[cc][j], y);
                        ys[i * VB + vg * COLS + cc] = bf(y);
                    }
            }
        }
        // S <- elast * S + last * (B^T D), per owned key.
#pragma unroll
        for (int j = 0; j < KW; ++j) {
            const int k = kp * KW + j;
            float t[COLS];
#pragma unroll
            for (int cc = 0; cc < COLS; ++cc) t[cc] = 0.0f;
#pragma unroll
            for (int i = 0; i < CH; i += 4) {
                const float4 b4 = *reinterpret_cast<const float4*>(&sb[Gm::bx(k, i)]);
#pragma unroll
                for (int cc = 0; cc < COLS; ++cc) {
                    t[cc] = fmaf(b4.x, dl[cc][i], t[cc]);
                    t[cc] = fmaf(b4.y, dl[cc][i + 1], t[cc]);
                    t[cc] = fmaf(b4.z, dl[cc][i + 2], t[cc]);
                    t[cc] = fmaf(b4.w, dl[cc][i + 3], t[cc]);
                }
            }
            const float e = se[k], l = sl[k];
#pragma unroll
            for (int cc = 0; cc < COLS; ++cc) s[cc][j] = fmaf(s[cc][j], e, l * t[cc]);
        }
        __syncthreads();

        // 7. Outputs. With the whole head in this block, the chain's gated RMSNorm (warp per row); otherwise
        // bf16(Y) for this block's columns, normalized by norm_kernel afterwards.
        if (NB == 1) {
#pragma unroll
            for (int u = 0; u < CH / (THREADS / 32); ++u) {
                const int r = warp + u * (THREADS / 32);
                if (r < n) {
                    float ss = 0.0f;
#pragma unroll
                    for (int i = 0; i < 4; ++i) {
                        const float y = ys[r * VB + lane * 4 + i];
                        ss = ss + y * y;
                    }
                    ss = warp_sum(ss);
                    const float rinv = 1.0f / sqrtf(ss / (float)DV + eps);
#pragma unroll
                    for (int i = 0; i < 4; ++i) {
                        const int t = lane * 4 + i;
                        const float yn = ys[r * VB + t] * rinv;
                        const float yw = nw[i] * yn;
                        out[(rbase + r) * out_stride + h * DV + t] =
                            __float2bfloat16_rn(yw * sigmoidf_(b2f(gin[u][i])));
                    }
                }
            }
        } else {
            for (int e = tid; e < n * VB; e += THREADS) {
                const int r = e / VB, cc = e % VB;
                out[(rbase + r) * out_stride + h * DV + vb0 + cc] = __float2bfloat16_rn(ys[r * VB + cc]);
            }
        }
        __syncthreads();
    }
    cp_async_wait<0>();
#pragma unroll
    for (int c = 0; c < COLS; ++c)
#pragma unroll
        for (int j = 0; j < KW; ++j) state_out[sbase + (long long)(vb0 + vg * COLS + c) * DK + kp * KW + j] = s[c][j];
}

// The gated RMSNorm over the bf16(Y) that value-split blocks left in `out`: warp per (row, head), the chain's
// arithmetic. Block (x, b) strides over request b's rows.
__global__ void norm_kernel(int H, const int32_t* __restrict__ cu_rows, int rows1, const __nv_bfloat16* __restrict__ G,
                            long long g_stride, const __nv_bfloat16* __restrict__ norm_w, float eps,
                            __nv_bfloat16* out, long long out_stride) {
    const Rows rr = request_rows(cu_rows, blockIdx.y, rows1);
    const long long tasks = (long long)rr.rows * H;
    const int wpb = blockDim.x >> 5, lane = threadIdx.x & 31;
    for (long long task = (long long)blockIdx.x * wpb + (threadIdx.x >> 5); task < tasks;
         task += (long long)gridDim.x * wpb) {
        const long long r = rr.row0 + task / H;
        const int h = (int)(task % H);
        __nv_bfloat16* o = out + r * out_stride + h * DV;
        float y[4], ss = 0.0f;
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            y[i] = b2f(o[lane * 4 + i]);
            ss = ss + y[i] * y[i];
        }
        ss = warp_sum(ss);
        const float rinv = 1.0f / sqrtf(ss / (float)DV + eps);
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int t = lane * 4 + i;
            const float yn = y[i] * rinv;
            const float yw = b2f(norm_w[t]) * yn;
            const float gate = b2f(G[r * g_stride + h * DV + t]);
            o[t] = __float2bfloat16_rn(yw * sigmoidf_(gate));
        }
    }
}

// Advance each request's conv window past all its rows (rows `rows .. rows + 2` of [window; rows]), in place:
// each thread reads its three values before writing them.
__global__ void conv_advance_kernel(int C, const int32_t* __restrict__ cu_rows, int rows1, __nv_bfloat16* conv,
                                    const long long* __restrict__ conv_off, const __nv_bfloat16* __restrict__ P,
                                    long long p_stride) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= C) return;
    const int b = blockIdx.y;
    const Rows rr = request_rows(cu_rows, b, rows1);
    __nv_bfloat16* win = conv + (conv_off != nullptr ? conv_off[b] : 0);
    __nv_bfloat16 x[TAPS - 1];
#pragma unroll
    for (int j = 0; j < TAPS - 1; ++j) {
        const int src = rr.rows + j;
        x[j] = src < TAPS - 1 ? win[(long long)src * C + c] : P[(rr.row0 + src - (TAPS - 1)) * p_stride + c];
    }
#pragma unroll
    for (int j = 0; j < TAPS - 1; ++j) win[(long long)j * C + c] = x[j];
}

// cudaErrorInvalidValue.
constexpr int kInvalid = 1;
// Blocks per request for norm_kernel.
constexpr int kNormBlocks = 256;

template <typename T>
inline const __nv_bfloat16* bfp(const T* x) { return reinterpret_cast<const __nv_bfloat16*>(x); }
template <typename T>
inline __nv_bfloat16* bfp(T* x) { return reinterpret_cast<__nv_bfloat16*>(x); }

int sm_count() {
    static int n = 0;
    if (n == 0) {
        int dev = 0, v = 0;
        if (cudaGetDevice(&dev) == cudaSuccess &&
            cudaDeviceGetAttribute(&v, cudaDevAttrMultiProcessorCount, dev) == cudaSuccess && v > 0)
            n = v;
        else {
            cudaGetLastError();
            return 1;
        }
    }
    return n;
}

template <int KP, int COLS>
int inter_launch(dim3 grid, cudaStream_t st, int H, const int32_t* cu_rows, int rows1, int sub0, int chunks,
                 const __nv_bfloat16* G, long long g_stride, const float* s_in, float* s_out, const long long* soff,
                 const __nv_bfloat16* norm_w, float eps, __nv_bfloat16* out, long long out_stride, const float* ws) {
    const size_t bytes = sizeof(float) * Geo<KP, COLS>::FLOATS;
    cudaFuncSetAttribute(inter_kernel<KP, COLS>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)bytes);
    inter_kernel<KP, COLS><<<grid, THREADS, bytes, st>>>(H, cu_rows, rows1, sub0, chunks, G, g_stride, s_in, s_out, soff,
                                                   norm_w, eps, out, out_stride, ws);
    return (int)cudaGetLastError();
}

// The passes over every sub-segment, the norm for value-split blocks, and the conv-window advance, in
// stream order.
int launch(int32_t heads, int32_t batch, const int32_t* cu_rows, int32_t rows1, int32_t max_rows,
           const glm53f_bf16* p, int64_t p_stride, int64_t b_off, const glm53f_bf16* a, int64_t a_stride,
           const glm53f_bf16* g, int64_t g_stride, glm53f_bf16* conv, const int64_t* conv_off,
           const glm53f_bf16* conv_w, const float* state_in, float* state_out, const int64_t* state_off,
           const float* a_log, const float* dt_bias, const glm53f_bf16* norm_w, float eps, float lower,
           glm53f_bf16* out, int64_t out_stride, int32_t value_blocks, float* workspace, int64_t workspace_bytes,
           glm53f_stream_t stream) {
    if (heads < 1 || heads > 65535 || batch < 1 || batch > 65535 || rows1 < 0 || max_rows < 0 || !p || !a ||
        !g || !conv || !conv_w || !state_in || !state_out || !a_log || !dt_bias || !norm_w || !out || !workspace)
        return kInvalid;
    // Fifteen rows of decay must stay above the smallest normal f32 (e^-87.3) in the in-chunk products.
    if (!(lower >= -5.8f && lower <= 0.0f)) return kInvalid;
    const long long per_chunk = (long long)WS_FLOATS * sizeof(float) * heads * batch;
    long long chunks = workspace_bytes / per_chunk;
    if (chunks < 1) return kInvalid;
    const long long needed = ((long long)max_rows + CH - 1) / CH;
    if (chunks > needed) chunks = needed > 0 ? needed : 1;
    if (chunks > 65535) chunks = 65535;
    int nb = value_blocks;
    if (nb == 0) {
        // The most value blocks per head that still run as one wave (inter_kernel fits one block per SM):
        // a second wave costs more than splitting the columns saves.
        const long long blocks = (long long)heads * batch;
        const int sms = sm_count();
        nb = 4 * blocks <= sms ? 4 : (2 * blocks <= sms ? 2 : 1);
    }
    if (nb != 1 && nb != 2 && nb != 4) return kInvalid;
    cudaStream_t st = (cudaStream_t)stream;
    const long long* coff = reinterpret_cast<const long long*>(conv_off);
    const long long* soff = reinterpret_cast<const long long*>(state_off);
    const int step = (int)chunks * CH;
    for (int sub0 = 0; sub0 == 0 || sub0 < max_rows; sub0 += step) {
        if (max_rows > 0) {
            intra_kernel<<<dim3((unsigned)chunks, heads, batch), THREADS, 0, st>>>(
                heads, cu_rows, rows1, sub0, bfp(p), p_stride, b_off, bfp(a), a_stride, bfp(conv), coff, bfp(conv_w),
                a_log, dt_bias, lower, workspace);
            const int rc = (int)cudaGetLastError();
            if (rc != 0) return rc;
        }
        const float* s_in = sub0 == 0 ? state_in : state_out;
        const dim3 grid(heads * nb, batch);
        const int rc = nb == 1   ? inter_launch<2, 1>(grid, st, heads, cu_rows, rows1, sub0, (int)chunks, bfp(g), g_stride,
                                                   s_in, state_out, soff, bfp(norm_w), eps, bfp(out), out_stride, workspace)
                       : nb == 2 ? inter_launch<8, 2>(grid, st, heads, cu_rows, rows1, sub0, (int)chunks, bfp(g), g_stride,
                                                   s_in, state_out, soff, bfp(norm_w), eps, bfp(out), out_stride, workspace)
                                 : inter_launch<8, 1>(grid, st, heads, cu_rows, rows1, sub0, (int)chunks, bfp(g), g_stride,
                                                   s_in, state_out, soff, bfp(norm_w), eps, bfp(out), out_stride, workspace);
        if (rc != 0) return rc;
        if (max_rows == 0) break;
    }
    if (nb > 1) {
        norm_kernel<<<dim3(kNormBlocks, batch), 256, 0, st>>>(heads, cu_rows, rows1, bfp(g), g_stride, bfp(norm_w), eps,
                                                               bfp(out), out_stride);
        const int rc = (int)cudaGetLastError();
        if (rc != 0) return rc;
    }
    const int C = 3 * heads * DK;
    conv_advance_kernel<<<dim3((C + 255) / 256, batch), 256, 0, st>>>(C, cu_rows, rows1, bfp(conv), coff, bfp(p),
                                                                      p_stride);
    return (int)cudaGetLastError();
}

}  // namespace

extern "C" int64_t glm53f_kda_prefill_workspace_bytes(int32_t heads, int32_t batch, int32_t rows_per_pass) {
    if (heads < 1 || batch < 1 || rows_per_pass < 1) return 0;
    return (((int64_t)rows_per_pass + CH - 1) / CH) * WS_FLOATS * (int64_t)sizeof(float) * heads * batch;
}

extern "C" int glm53f_kda_prefill(int32_t heads, int32_t rows, const glm53f_bf16* p, int64_t p_stride,
                                  int64_t b_off, const glm53f_bf16* a, int64_t a_stride, const glm53f_bf16* g,
                                  int64_t g_stride, glm53f_bf16* conv, const glm53f_bf16* conv_w,
                                  const float* state_in, float* state_out, const float* a_log, const float* dt_bias,
                                  const glm53f_bf16* norm_w, float eps, float lower, glm53f_bf16* out,
                                  int64_t out_stride, int32_t value_blocks, float* workspace, int64_t workspace_bytes,
                                  glm53f_stream_t stream) {
    return launch(heads, 1, nullptr, rows, rows, p, p_stride, b_off, a, a_stride, g, g_stride, conv, nullptr, conv_w,
                  state_in, state_out, nullptr, a_log, dt_bias, norm_w, eps, lower, out, out_stride, value_blocks,
                  workspace, workspace_bytes, stream);
}

extern "C" int glm53f_kda_prefill_batch(int32_t heads, int32_t batch, const int32_t* cu_rows, int32_t max_rows,
                                        const glm53f_bf16* p, int64_t p_stride, int64_t b_off,
                                        const glm53f_bf16* a, int64_t a_stride, const glm53f_bf16* g,
                                        int64_t g_stride, glm53f_bf16* conv, const int64_t* conv_off,
                                        const glm53f_bf16* conv_w, const float* state_in, float* state_out,
                                        const int64_t* state_off, const float* a_log, const float* dt_bias,
                                        const glm53f_bf16* norm_w, float eps, float lower, glm53f_bf16* out,
                                        int64_t out_stride, int32_t value_blocks, float* workspace,
                                        int64_t workspace_bytes, glm53f_stream_t stream) {
    if (!cu_rows || !conv_off || !state_off) return kInvalid;
    return launch(heads, batch, cu_rows, 0, max_rows, p, p_stride, b_off, a, a_stride, g, g_stride, conv, conv_off,
                  conv_w, state_in, state_out, state_off, a_log, dt_bias, norm_w, eps, lower, out, out_stride,
                  value_blocks, workspace, workspace_bytes, stream);
}
