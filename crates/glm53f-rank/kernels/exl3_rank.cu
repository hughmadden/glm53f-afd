// glm53f-rank: one expert rank's share of GLM-5.3-Flash's routed-expert FFN, EXL3 K4 (ExLlamaV3 trellis,
// "mcg" codebook) at TP4: every expert's intermediate channels [512 r, 512 r + 512), 288 experts a layer.
//
// Per call: `rows` wire rows (FP8 E4M3 values, one UE8M0 scale per 32) with their top-8 expert ids and FP32
// gate weights  ->  plan (group the rows x 8 routes by expert, on the GPU, one CTA)  ->  gate/up (the wire row
// widened to FP32, times suh, rotated by a 128-point Hadamard transform, rounded to FP16 and multiplied by the
// trellis tiles on the tensor cores)  ->  epilogue (rotate back, times svh, GLM's clamped SwiGLU with BF16
// roundings, times the down projection's suh, rotate, FP16)  ->  down (tensor cores again)  ->  reduce (rotate
// back, times svh, the 8 slots summed in slot order with their gate weights, BF16)  =  the rank's partial row.
// `g53r_ffn_f32` leaves that row in FP32, for the prefill reduce-scatter.
//
// Sources. The tile decoder (`mcg2`, `decode_tile`), the MMA wrapper and the warp butterfly (`fwht128`) are
// TensorFold's `src/tensorfold/families/glm5_next/cuda/exl3.cu` at bb4b4a3 (MIT, Copyright (c) 2026 TensorFold
// contributors; LICENSE.tensorfold), verbatim. TensorFold's kernels read the format of ExLlamaV3
// (https://github.com/turboderp-org/exllamav3, MIT, Copyright (c) 2025 Turboderp). The grouped GEMMs follow
// TensorFold's `grouped_kernel` (a warp decodes 16x16 tiles straight into B fragments; K splits summed in a
// fixed order) with these changes: TP4 slices (N = 512 for gate/up, K = 512 for down); each block's four
// warps cover the slice's four 128-column Hadamard blocks, so the input rotation is computed once per block
// in shared memory, straight from the FP8 wire row (no rotated-input buffer); up to two 16-row tiles per
// block for prefill; the next k tile's weights are loaded while the current one is multiplied. The planner
// takes MiMo's one-CTA route plan (mimo26f-afd v1.2.0 `crates/mimo26-spark/kernels/b1_serve.cu`, MIT) to 288
// experts. See PROVENANCE.md.
//
// Every output depends only on its own row: the tensor cores keep rows independent, and every sum (K splits,
// the 8 slots) runs in a fixed order. With the same (MT, SK, SKD) configuration a row gets the same bits
// whatever else is in the batch. Build with --fmad=false: the CPU reference (src/reference.rs) repeats the
// epilogues' separate multiplies and adds.

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

#ifndef G53R_BAKED_ARCH
#define G53R_BAKED_ARCH 0
#endif

namespace {

constexpr int H = 4096;      // model width: gate/up K, down N
constexpr int WID = 512;     // rank width: gate/up N, down K
constexpr int E = 288;       // routed experts per layer
constexpr int TOPK = 8;
constexpr int ROW_PITCH = H + H / 32;  // device copy of a wire row: 4,096 E4M3 then 128 UE8M0
constexpr int MAX_ROWS = 4096;
constexpr float HAD_SCALE = 0.08838834764831845f;  // 1 / sqrt(128)
constexpr float LIMIT = 10.0f;                      // swiglu_limit

// Layer image layout glm53f-exl3-k4-tp4-e1 (src/layout.rs).
constexpr size_t TRELLIS = 1048576;
constexpr size_t OFF_GATE_T = 0, OFF_UP_T = TRELLIS, OFF_DOWN_T = 2 * TRELLIS;
constexpr size_t OFF_SUH_G = 3 * TRELLIS, OFF_SUH_U = OFF_SUH_G + 2 * H, OFF_SVH_G = OFF_SUH_U + 2 * H;
constexpr size_t OFF_SVH_U = OFF_SVH_G + 2 * WID, OFF_SUH_D = OFF_SVH_U + 2 * WID, OFF_SVH_D = OFF_SUH_D + 2 * WID;
constexpr size_t EXPERT_BYTES = OFF_SVH_D + 2 * H;
constexpr size_t LAYER_BYTES = size_t(E) * EXPERT_BYTES;
static_assert(EXPERT_BYTES == 3173376, "layout e1");
static_assert(E % 32 == 0, "the plan scans the experts a warp at a time");

// Fault bits (OR-ed into one word).
constexpr unsigned F_ROUTE = 1u;   // bad expert id, weight or a duplicate id in a row
constexpr unsigned F_ROW = 2u;     // an E4M3 NaN value or a UE8M0 NaN scale in a wire row
constexpr unsigned F_GATEUP = 4u;  // a non-finite gate/up output
constexpr unsigned F_OUT = 8u;     // a non-finite output sum
constexpr unsigned F_SCALE = 16u;  // a non-finite scale vector in a layer image

struct Group {
    int expert;  // expert id (== slot in the image)
    int start;   // first grouped pair
    int count;   // pairs (<= 16 * MT)
    int pad;
};

// ---- TensorFold exl3.cu (bb4b4a3), verbatim ------------------------------------------------------------------

// Two values of the "mcg" codebook from two 16-bit states, as a half2 (first state in .x).
__device__ __forceinline__ uint32_t mcg2(uint32_t s0, uint32_t s1) {
    uint32_t x0 = s0 * 0xCBAC1FEDu;
    uint32_t x1 = s1 * 0xCBAC1FEDu;
    x0 = (x0 & 0x8FFF8FFFu) ^ 0x3B603B60u;
    x1 = (x1 & 0x8FFF8FFFu) ^ 0x3B603B60u;
    uint32_t lo = __byte_perm(x0, x1, 0x5410);
    uint32_t hi = __byte_perm(x0, x1, 0x7632);
    half2 r = __hadd2(*reinterpret_cast<half2*>(&lo), *reinterpret_cast<half2*>(&hi));
    return *reinterpret_cast<uint32_t*>(&r);
}

// This lane's eight values of a 4-bit tile (word = tile[lane]) as the B fragments of its two n8 halves.
__device__ __forceinline__ void decode_tile(uint32_t w, int lane, uint32_t (&b0)[2], uint32_t (&b1)[2]) {
    uint32_t p = __shfl_sync(0xffffffffu, w, (lane + 31) & 31);
    uint32_t s = __funnelshift_r(w, p, 20);
    b0[0] = mcg2((s >> 8) & 0xffffu, (s >> 4) & 0xffffu);
    b0[1] = mcg2(s & 0xffffu, w >> 16);
    b1[0] = mcg2((w >> 12) & 0xffffu, (w >> 8) & 0xffffu);
    b1[1] = mcg2((w >> 4) & 0xffffu, w & 0xffffu);
}

__device__ __forceinline__ void mma16816(float (&d)[4], const uint32_t (&a)[4], const uint32_t (&b)[2]) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
                 "{%0,%1,%2,%3};\n"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

// Fast Walsh-Hadamard transform of 128 values held 4 per lane (lane L: values 4L..4L+3), natural order, fixed
// butterfly order: strides 1, 2 in registers, 4..64 across lanes.
__device__ __forceinline__ void fwht128(float (&v)[4], int lane) {
    float a = v[0] + v[1], b = v[0] - v[1], c = v[2] + v[3], d = v[2] - v[3];
    v[0] = a + c; v[1] = b + d; v[2] = a - c; v[3] = b - d;
#pragma unroll
    for (int m = 1; m < 32; m <<= 1) {
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            float o = __shfl_xor_sync(0xffffffffu, v[j], m);
            v[j] = (lane & m) ? o - v[j] : v[j] + o;
        }
    }
}

// ---- end of the TensorFold functions -------------------------------------------------------------------------

__device__ __forceinline__ float bf16r(float x) { return __bfloat162float(__float2bfloat16_rn(x)); }

// E4M3 (FN) code to f32, exact; NaN codes are screened by the caller.
__device__ __forceinline__ float e4m3(uint32_t c) {
    const uint32_t e = (c >> 3) & 0xFu, m = c & 7u;
    const float mag = e ? __uint_as_float(((e + 120u) << 23) | (m << 20)) : float(m) * 0x1p-9f;
    return (c & 0x80u) ? -mag : mag;
}

// UE8M0 byte to 2^(b - 127); byte 0 is the subnormal 2^-127. Byte 255 (NaN) is screened by the caller.
__device__ __forceinline__ float ue8m0(uint32_t b) { return b ? __uint_as_float(b << 23) : __uint_as_float(0x00400000u); }

__device__ __forceinline__ uint32_t ld32(const half* p) { return *reinterpret_cast<const uint32_t*>(p); }

// ---- plan: rows x 8 routes -> pairs grouped by expert ---------------------------------------------------------
// One 1,024-thread CTA (after MiMo's plan_parallel). Pairs of expert e occupy [base[e], base[e] + cnt[e]) in
// ascending expert order, cut into groups of at most `gr` pairs. The order of an expert's pairs follows the
// atomics and is not stable; that cannot change a value, since every pair is computed on its own rows.
__global__ void __launch_bounds__(1024) plan_kernel(const int32_t* __restrict__ ids, const float* __restrict__ w,
                                                    int rows, int gr, Group* __restrict__ groups,
                                                    int* __restrict__ counts, int* __restrict__ pair_row,
                                                    int* __restrict__ pair_route, int* __restrict__ inverse,
                                                    unsigned* __restrict__ fault) {
    __shared__ int cnt[E], base[E], gbase[E], cursor[E];
    __shared__ int bad;
    const int tid = threadIdx.x, nt = blockDim.x, routes = rows * TOPK;
    for (int e = tid; e < E; e += nt) { cnt[e] = 0; cursor[e] = 0; }
    if (tid == 0) bad = 0;
    __syncthreads();
    for (int r = tid; r < routes; r += nt) {
        const int e = ids[r];
        const float wt = w[r];
        bool ok = e >= 0 && e < E && wt >= 0.0f && wt <= 0x1.fffffep127f;  // finite, not negative, not NaN
        if (ok)
            for (int k = (r / TOPK) * TOPK; k < r; ++k) ok = ok && ids[k] != e;  // no expert twice in a row
        if (!ok) atomicOr(&bad, 1); else atomicAdd(&cnt[e], 1);
    }
    __syncthreads();
    if (bad) {
        if (tid == 0) { atomicOr(fault, F_ROUTE); counts[0] = 0; }
        return;
    }
    // Exclusive prefix sums of the pair and group counts over the 288 experts: nine warps scan 32 experts each
    // with shuffles, then every warp adds the totals of the warps before it (a serial scan here cost ~10 us).
    __shared__ int wsum[E / 32], wgsum[E / 32];
    const int warp = tid >> 5, lane = tid & 31;
    int c = 0, gc = 0, ic = 0, igc = 0;
    if (warp < E / 32) {
        c = cnt[warp * 32 + lane];
        gc = (c + gr - 1) / gr;
        ic = c;
        igc = gc;
#pragma unroll
        for (int d = 1; d < 32; d <<= 1) {
            const int u = __shfl_up_sync(0xffffffffu, ic, d), ug = __shfl_up_sync(0xffffffffu, igc, d);
            if (lane >= d) { ic += u; igc += ug; }
        }
        if (lane == 31) { wsum[warp] = ic; wgsum[warp] = igc; }
    }
    __syncthreads();
    if (warp < E / 32) {
        int off = 0, goff = 0;
        for (int w = 0; w < warp; ++w) { off += wsum[w]; goff += wgsum[w]; }
        base[warp * 32 + lane] = off + ic - c;
        gbase[warp * 32 + lane] = goff + igc - gc;
        if (warp == E / 32 - 1 && lane == 31) counts[0] = goff + igc;
    }
    __syncthreads();
    for (int r = tid; r < routes; r += nt) {
        const int e = ids[r];
        const int gp = base[e] + atomicAdd(&cursor[e], 1);
        pair_row[gp] = r / TOPK;
        pair_route[gp] = r;
        inverse[r] = gp;
    }
    for (int e = tid; e < E; e += nt) {
        const int c = cnt[e];
        for (int k = 0; k * gr < c; ++k)
            groups[gbase[e] + k] = Group{e, base[e] + k * gr, c - k * gr < gr ? c - k * gr : gr, 0};
    }
}

// ---- gate and up: Z[mat][split][pair][512] = rot(x * suh) @ W_q over this split's K range ---------------------
// Block (group, mat, split), 4 warps; warp w owns the slice's columns [128 w, 128 w + 128) (8 n tiles). For
// each 128-wide K block the warps first rotate the block's rows into shared memory (one warp per row), then
// multiply. MT 16-row tiles per block.
template <int MT>
__global__ void __launch_bounds__(128) gateup_kernel(const uint8_t* __restrict__ image, const uint8_t* __restrict__ xrows,
                                                     const Group* __restrict__ groups, const int* __restrict__ counts,
                                                     const int* __restrict__ pair_row, float* __restrict__ Z, int P,
                                                     int SK, unsigned* __restrict__ fault) {
    constexpr int R = MT * 16;
    const int g = blockIdx.x;
    if (g >= counts[0]) return;
    const int mat = blockIdx.y, split = blockIdx.z;
    const Group G = groups[g];
    __shared__ __align__(16) half As[R][136];  // 272-byte rows: conflict-free fragment loads
    __shared__ int rows_sh[R];
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, g8 = lane >> 2, t4 = lane & 3;
    if (tid < R) rows_sh[tid] = tid < G.count ? pair_row[G.start + tid] : -1;
    __syncthreads();

    const uint8_t* ex = image + size_t(G.expert) * EXPERT_BYTES;
    const uint32_t* T = reinterpret_cast<const uint32_t*>(ex + (mat ? OFF_UP_T : OFF_GATE_T));
    const half* suh = reinterpret_cast<const half*>(ex + (mat ? OFF_SUH_U : OFF_SUH_G));
    const int kbs = 32 / SK;  // 128-wide K blocks per split
    const int kt_begin = split * kbs * 8, kt_end = kt_begin + kbs * 8;

    float acc[MT][8][2][4];
#pragma unroll
    for (int m = 0; m < MT; ++m)
#pragma unroll
        for (int i = 0; i < 8; ++i)
#pragma unroll
            for (int h = 0; h < 2; ++h)
#pragma unroll
                for (int c = 0; c < 4; ++c) acc[m][i][h][c] = 0.f;

    // Trellis words of tile (kt, nt) are at (kt * 32 + nt) * 32 + lane; this warp's 8 n tiles are contiguous.
    const uint32_t* tp = T + (size_t(kt_begin) * 32 + warp * 8) * 32 + lane;
    uint32_t wn[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) wn[i] = __ldg(tp + i * 32);
    bool bad = false;

    for (int kt = kt_begin; kt < kt_end; ++kt) {
        const int ktl = kt & 7;
        if (ktl == 0) {
            __syncthreads();  // the previous block's fragments have been read
            const int kb = kt >> 3;
            for (int rr = warp; rr < R; rr += 4) {
                const int r = rows_sh[rr];
                uint2 packed = make_uint2(0u, 0u);
                if (r >= 0) {
                    const uint8_t* xr = xrows + size_t(r) * ROW_PITCH;
                    const uint32_t q = *reinterpret_cast<const uint32_t*>(xr + kb * 128 + lane * 4);
                    const uint32_t sb = xr[H + kb * 4 + (lane >> 3)];
                    bad = bad || sb == 0xFFu;
                    const float s = ue8m0(sb);
                    const half* su = suh + kb * 128 + lane * 4;
                    float v[4];
#pragma unroll
                    for (int j = 0; j < 4; ++j) {
                        const uint32_t c = (q >> (8 * j)) & 0xFFu;
                        bad = bad || (c & 0x7Fu) == 0x7Fu;
                        v[j] = e4m3(c) * s * __half2float(su[j]);
                    }
                    fwht128(v, lane);
                    const half2 h0 = __floats2half2_rn(v[0] * HAD_SCALE, v[1] * HAD_SCALE);
                    const half2 h1 = __floats2half2_rn(v[2] * HAD_SCALE, v[3] * HAD_SCALE);
                    packed = make_uint2(*reinterpret_cast<const uint32_t*>(&h0), *reinterpret_cast<const uint32_t*>(&h1));
                }
                *reinterpret_cast<uint2*>(&As[rr][lane * 4]) = packed;
            }
            __syncthreads();
        }
        uint32_t w[8];
#pragma unroll
        for (int i = 0; i < 8; ++i) w[i] = wn[i];
        if (kt + 1 < kt_end) {
            tp += 32 * 32;
#pragma unroll
            for (int i = 0; i < 8; ++i) wn[i] = __ldg(tp + i * 32);
        }
        uint32_t a[MT][4];
#pragma unroll
        for (int m = 0; m < MT; ++m) {
            const half* r0 = &As[m * 16 + g8][ktl * 16 + 2 * t4];
            const half* r1 = &As[m * 16 + g8 + 8][ktl * 16 + 2 * t4];
            a[m][0] = ld32(r0);
            a[m][1] = ld32(r1);
            a[m][2] = ld32(r0 + 8);
            a[m][3] = ld32(r1 + 8);
        }
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            uint32_t b0[2], b1[2];
            decode_tile(w[i], lane, b0, b1);
#pragma unroll
            for (int m = 0; m < MT; ++m) {
                mma16816(acc[m][i][0], a[m], b0);
                mma16816(acc[m][i][1], a[m], b1);
            }
        }
    }
    if (bad) atomicOr(fault, F_ROW);

    float* zbase = Z + (size_t(mat) * SK + split) * size_t(P) * WID;
#pragma unroll
    for (int m = 0; m < MT; ++m) {
        const int r0 = m * 16 + g8, r1 = r0 + 8;
#pragma unroll
        for (int i = 0; i < 8; ++i)
#pragma unroll
            for (int h = 0; h < 2; ++h) {
                const int col = warp * 128 + i * 16 + h * 8 + 2 * t4;
                if (r0 < G.count)
                    *reinterpret_cast<float2*>(zbase + size_t(G.start + r0) * WID + col) =
                        make_float2(acc[m][i][h][0], acc[m][i][h][1]);
                if (r1 < G.count)
                    *reinterpret_cast<float2*>(zbase + size_t(G.start + r1) * WID + col) =
                        make_float2(acc[m][i][h][2], acc[m][i][h][3]);
            }
    }
}

// ---- epilogue: splits summed in order, rotated, svh, GLM's SwiGLU, down suh, rotated, FP16 --------------------
// One warp per (pair, 128-block of the rank width); after TensorFold's gateup_epilogue_kernel.
__global__ void __launch_bounds__(128) gateup_epilogue(const float* __restrict__ Z, const int* __restrict__ pair_route,
                                                       const int32_t* __restrict__ ids, const uint8_t* __restrict__ image,
                                                       const int* __restrict__ counts, half* __restrict__ Xd, int P,
                                                       int SK, int bf16, unsigned* __restrict__ fault) {
    const int item = blockIdx.x * 4 + (threadIdx.x >> 5), lane = threadIdx.x & 31;
    const int gp = item >> 2, blk = item & 3;
    if (gp >= P || counts[0] <= 0) return;  // no plan (invalid routes): ids may be out of range
    const uint8_t* ex = image + size_t(ids[pair_route[gp]]) * EXPERT_BYTES;
    const half* svh_g = reinterpret_cast<const half*>(ex + OFF_SVH_G);
    const half* svh_u = reinterpret_cast<const half*>(ex + OFF_SVH_U);
    const half* suh_d = reinterpret_cast<const half*>(ex + OFF_SUH_D);
    const int n = blk * 128 + lane * 4;
    float gv[4], uv[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        float sg = 0.f, su = 0.f;
        for (int s = 0; s < SK; ++s) {
            sg += Z[(size_t(0 * SK + s) * P + gp) * WID + n + j];
            su += Z[(size_t(1 * SK + s) * P + gp) * WID + n + j];
        }
        gv[j] = sg;
        uv[j] = su;
    }
    bool bad = false;
#pragma unroll
    for (int j = 0; j < 4; ++j) bad = bad || !isfinite(gv[j]) || !isfinite(uv[j]);
    fwht128(gv, lane);
    fwht128(uv, lane);
    float v[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        float gg = gv[j] * HAD_SCALE * __half2float(svh_g[n + j]);
        float uu = uv[j] * HAD_SCALE * __half2float(svh_u[n + j]);
        float act;
        if (bf16) {  // the reference model's BF16 SwiGLU (TensorFold's default)
            gg = fminf(bf16r(gg), LIMIT);
            uu = fminf(fmaxf(bf16r(uu), -LIMIT), LIMIT);
            act = bf16r(bf16r(gg / (1.f + expf(-gg))) * uu);
        } else {     // FP32 throughout (an option for the KL gate and for tight kernel tests)
            gg = fminf(gg, LIMIT);
            uu = fminf(fmaxf(uu, -LIMIT), LIMIT);
            act = gg / (1.f + expf(-gg)) * uu;
        }
        v[j] = act * __half2float(suh_d[n + j]);
    }
    fwht128(v, lane);
    const half2 h0 = __floats2half2_rn(v[0] * HAD_SCALE, v[1] * HAD_SCALE);
    const half2 h1 = __floats2half2_rn(v[2] * HAD_SCALE, v[3] * HAD_SCALE);
    *reinterpret_cast<uint2*>(Xd + size_t(gp) * WID + n) =
        make_uint2(*reinterpret_cast<const uint32_t*>(&h0), *reinterpret_cast<const uint32_t*>(&h1));
    if (__any_sync(0xffffffffu, bad) && lane == 0) atomicOr(fault, F_GATEUP);
}

// ---- down: Zd[split][pair][4096] = Xd[pair] @ W_q(down) over this split's K range -----------------------------
// Block (group, 512-column chunk of the output, split); warp w owns the chunk's columns [128 w, 128 w + 128).
// A group's pairs are contiguous in Xd.
template <int MT>
__global__ void __launch_bounds__(128) down_kernel(const uint8_t* __restrict__ image, const half* __restrict__ Xd,
                                                   const Group* __restrict__ groups, const int* __restrict__ counts,
                                                   float* __restrict__ Zd, int P, int SKD) {
    const int g = blockIdx.x;
    if (g >= counts[0]) return;
    const int chunk = blockIdx.y, split = blockIdx.z;
    const Group G = groups[g];
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, g8 = lane >> 2, t4 = lane & 3;
    const uint32_t* T = reinterpret_cast<const uint32_t*>(image + size_t(G.expert) * EXPERT_BYTES + OFF_DOWN_T);
    const int kts = 32 / SKD, kt_begin = split * kts, kt_end = kt_begin + kts;

    float acc[MT][8][2][4];
#pragma unroll
    for (int m = 0; m < MT; ++m)
#pragma unroll
        for (int i = 0; i < 8; ++i)
#pragma unroll
            for (int h = 0; h < 2; ++h)
#pragma unroll
                for (int c = 0; c < 4; ++c) acc[m][i][h][c] = 0.f;

    // Down tiles are [32 k tiles][256 n tiles]; this warp's 8 n tiles start at chunk * 32 + warp * 8.
    const uint32_t* tp = T + (size_t(kt_begin) * 256 + chunk * 32 + warp * 8) * 32 + lane;
    uint32_t wn[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) wn[i] = __ldg(tp + i * 32);
    const half* xg = Xd + size_t(G.start) * WID;
    for (int kt = kt_begin; kt < kt_end; ++kt) {
        uint32_t w[8];
#pragma unroll
        for (int i = 0; i < 8; ++i) w[i] = wn[i];
        if (kt + 1 < kt_end) {
            tp += 256 * 32;
#pragma unroll
            for (int i = 0; i < 8; ++i) wn[i] = __ldg(tp + i * 32);
        }
        const int k = kt * 16 + 2 * t4;
        uint32_t a[MT][4];
#pragma unroll
        for (int m = 0; m < MT; ++m) {
            const int r0 = m * 16 + g8, r1 = r0 + 8;
            const bool ok0 = r0 < G.count, ok1 = r1 < G.count;
            a[m][0] = ok0 ? ld32(xg + size_t(r0) * WID + k) : 0u;
            a[m][1] = ok1 ? ld32(xg + size_t(r1) * WID + k) : 0u;
            a[m][2] = ok0 ? ld32(xg + size_t(r0) * WID + k + 8) : 0u;
            a[m][3] = ok1 ? ld32(xg + size_t(r1) * WID + k + 8) : 0u;
        }
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            uint32_t b0[2], b1[2];
            decode_tile(w[i], lane, b0, b1);
#pragma unroll
            for (int m = 0; m < MT; ++m) {
                mma16816(acc[m][i][0], a[m], b0);
                mma16816(acc[m][i][1], a[m], b1);
            }
        }
    }
    float* zbase = Zd + size_t(split) * size_t(P) * H;
#pragma unroll
    for (int m = 0; m < MT; ++m) {
        const int r0 = m * 16 + g8, r1 = r0 + 8;
#pragma unroll
        for (int i = 0; i < 8; ++i)
#pragma unroll
            for (int h = 0; h < 2; ++h) {
                const int col = chunk * 512 + warp * 128 + i * 16 + h * 8 + 2 * t4;
                if (r0 < G.count)
                    *reinterpret_cast<float2*>(zbase + size_t(G.start + r0) * H + col) =
                        make_float2(acc[m][i][h][0], acc[m][i][h][1]);
                if (r1 < G.count)
                    *reinterpret_cast<float2*>(zbase + size_t(G.start + r1) * H + col) =
                        make_float2(acc[m][i][h][2], acc[m][i][h][3]);
            }
    }
}

// ---- reduce: out[row] = BF16(sum over slots, in order, of w * (rot(sum of splits) * svh)) ---------------------
// One warp per (row, 128-block of the model width). F32 keeps the sums in FP32 (the prefill reduce-scatter adds
// them across ranks before rounding): the same arithmetic, so their BF16 rounding is the default output's bits.
template <bool F32>
__global__ void __launch_bounds__(128) reduce_kernel(const float* __restrict__ Zd, const int* __restrict__ inverse,
                                                     const int32_t* __restrict__ ids, const float* __restrict__ wts,
                                                     const uint8_t* __restrict__ image, const int* __restrict__ counts,
                                                     void* __restrict__ out, int P, int SKD, int rows,
                                                     unsigned* __restrict__ fault) {
    const int item = blockIdx.x * 4 + (threadIdx.x >> 5), lane = threadIdx.x & 31;
    const int row = item >> 5, blk = item & 31;
    if (row >= rows || counts[0] <= 0) return;  // no plan (invalid routes)
    const int n = blk * 128 + lane * 4;
    float acc[4] = {0.f, 0.f, 0.f, 0.f};
    for (int slot = 0; slot < TOPK; ++slot) {
        const int r = row * TOPK + slot;
        const int gp = inverse[r];
        const float wt = wts[r];
        const half* svh = reinterpret_cast<const half*>(image + size_t(ids[r]) * EXPERT_BYTES + OFF_SVH_D);
        float v[4];
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            float s = 0.f;
            for (int k = 0; k < SKD; ++k) s += Zd[(size_t(k) * P + gp) * H + n + j];
            v[j] = s;
        }
        fwht128(v, lane);
#pragma unroll
        for (int j = 0; j < 4; ++j) acc[j] = acc[j] + wt * (v[j] * HAD_SCALE * __half2float(svh[n + j]));
    }
    bool bad = false;
#pragma unroll
    for (int j = 0; j < 4; ++j) bad = bad || !isfinite(acc[j]);
    if (F32) {
        *reinterpret_cast<float4*>(static_cast<float*>(out) + size_t(row) * H + n) =
            make_float4(acc[0], acc[1], acc[2], acc[3]);
    } else {
        uint16_t o[4];
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const __nv_bfloat16 b = __float2bfloat16_rn(acc[j]);
            o[j] = *reinterpret_cast<const uint16_t*>(&b);
        }
        *reinterpret_cast<uint2*>(static_cast<uint16_t*>(out) + size_t(row) * H + n) =
            make_uint2(uint32_t(o[0]) | (uint32_t(o[1]) << 16), uint32_t(o[2]) | (uint32_t(o[3]) << 16));
    }
    if (__any_sync(0xffffffffu, bad) && lane == 0) atomicOr(fault, F_OUT);
}

// Every F16 scale vector of every expert of a layer image must be finite.
__global__ void scale_scan(const uint8_t* __restrict__ image, unsigned* __restrict__ fault) {
    const size_t per = (2 * H + 3 * WID + H);  // halves of scale vectors per expert, contiguous from OFF_SUH_G
    for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x; i < size_t(E) * per;
         i += size_t(gridDim.x) * blockDim.x) {
        const size_t e = i / per, k = i % per;
        const uint16_t h = reinterpret_cast<const uint16_t*>(image + e * EXPERT_BYTES + OFF_SUH_G)[k];
        if ((h & 0x7C00u) == 0x7C00u) atomicOr(fault, F_SCALE);
    }
}

struct Dev {
    void* p = nullptr;
    size_t n = 0;
    cudaError_t ensure(size_t bytes) {
        if (bytes <= n) return cudaSuccess;
        if (p) cudaFree(p);
        p = nullptr;
        n = 0;
        const size_t want = bytes < 256 ? 256 : bytes;
        const cudaError_t e = cudaMalloc(&p, want);
        if (e == cudaSuccess) n = want;
        return e;
    }
    template <class T> T* as() const { return static_cast<T*>(p); }
    ~Dev() { if (p) cudaFree(p); }
};

void set_err(char* err, size_t len, const char* what, cudaError_t e) {
    if (err && len) std::snprintf(err, len, "%s: %s", what, cudaGetErrorString(e));
}

void set_msg(char* err, size_t len, const char* msg) {
    if (err && len) std::snprintf(err, len, "%s", msg);
}

}  // namespace

struct g53r_layer {
    uint8_t* image = nullptr;
};

struct g53r_scratch {
    cudaStream_t st = nullptr;
    cudaEvent_t e0 = nullptr, ep = nullptr, ea = nullptr, eb = nullptr, ec = nullptr, e1 = nullptr;
    Dev xrows, iw, groups, meta, pair_row, pair_route, inverse, z, xd, zd, out;
    std::vector<uint8_t> hiw;
};

// A kernel configuration: 16-row tiles per group (MT), gate/up K splits (SK, dividing 32), down K splits (SKD,
// dividing 32); zero fields take the defaults for the row count. `fp32_swiglu` = 1 computes the SwiGLU in FP32
// instead of with the reference model's BF16 roundings (the default).
struct g53r_cfg {
    int mt, sk, skd, fp32_swiglu;
};

extern "C" {

// The architecture the kernels were compiled for (major * 10 + minor, e.g. 89 or 121) and the live device's.
int g53r_baked_arch(void) { return G53R_BAKED_ARCH; }

int g53r_device_identity(int* arch, int* sms, char* name, size_t namelen) {
    int dev = 0;
    cudaDeviceProp p{};
    if (cudaGetDevice(&dev) != cudaSuccess || cudaGetDeviceProperties(&p, dev) != cudaSuccess) return 1;
    if (arch) *arch = p.major * 10 + p.minor;
    if (sms) *sms = p.multiProcessorCount;
    if (name && namelen) std::snprintf(name, namelen, "%s", p.name);
    return 0;
}

// The default configuration for `rows` rows. Up to 64 rows (decode and verify windows) share one
// configuration, so a row's bits do not depend on the window size.
void g53r_default_cfg(uint32_t rows, g53r_cfg* out) {
    if (rows <= 64) *out = g53r_cfg{1, 8, 2, 0};
    else *out = g53r_cfg{2, 1, 1, 0};
}

// Upload one layer image (LAYER_BYTES, host memory) and check its scale vectors.
int g53r_layer_new(const uint8_t* host_image, uint64_t bytes, g53r_layer** out, char* err, size_t errlen) {
    if (!host_image || !out || bytes != LAYER_BYTES) {
        set_msg(err, errlen, "layer: the image is not one layer of layout glm53f-exl3-k4-tp4-e1");
        return 1;
    }
    auto* L = new g53r_layer();
    cudaError_t e = cudaMalloc(&L->image, LAYER_BYTES);
    if (e != cudaSuccess) { set_err(err, errlen, "layer alloc", e); delete L; return 2; }
    unsigned* flag = nullptr;
    unsigned h = 0;
    if ((e = cudaMemcpy(L->image, host_image, LAYER_BYTES, cudaMemcpyHostToDevice)) != cudaSuccess ||
        (e = cudaMalloc(&flag, 4)) != cudaSuccess || (e = cudaMemset(flag, 0, 4)) != cudaSuccess) {
        set_err(err, errlen, "layer upload", e);
        if (flag) cudaFree(flag);
        cudaFree(L->image);
        delete L;
        return 2;
    }
    scale_scan<<<256, 256>>>(L->image, flag);
    if ((e = cudaGetLastError()) != cudaSuccess || (e = cudaMemcpy(&h, flag, 4, cudaMemcpyDeviceToHost)) != cudaSuccess) {
        set_err(err, errlen, "layer scale scan", e);
        cudaFree(flag);
        cudaFree(L->image);
        delete L;
        return 2;
    }
    cudaFree(flag);
    if (h) {
        set_msg(err, errlen, "layer: a scale vector holds a non-finite value");
        cudaFree(L->image);
        delete L;
        return 3;
    }
    *out = L;
    return 0;
}

void g53r_layer_free(g53r_layer* L) {
    if (!L) return;
    cudaFree(L->image);
    delete L;
}

int g53r_scratch_new(g53r_scratch** out, char* err, size_t errlen) {
    auto* S = new g53r_scratch();
    cudaError_t e;
    if ((e = cudaStreamCreateWithFlags(&S->st, cudaStreamNonBlocking)) != cudaSuccess ||
        (e = cudaEventCreate(&S->e0)) != cudaSuccess || (e = cudaEventCreate(&S->ep)) != cudaSuccess ||
        (e = cudaEventCreate(&S->ea)) != cudaSuccess ||
        (e = cudaEventCreate(&S->eb)) != cudaSuccess || (e = cudaEventCreate(&S->ec)) != cudaSuccess ||
        (e = cudaEventCreate(&S->e1)) != cudaSuccess) {
        set_err(err, errlen, "scratch stream/events", e);
        delete S;
        return 1;
    }
    *out = S;
    return 0;
}

void g53r_scratch_free(g53r_scratch* S) {
    if (!S) return;
    for (cudaEvent_t ev : {S->e0, S->ep, S->ea, S->eb, S->ec, S->e1})
        if (ev) cudaEventDestroy(ev);
    if (S->st) cudaStreamDestroy(S->st);
    delete S;
}

// The body of g53r_ffn and g53r_ffn_f32: exactly one of `bf16_out` and `f32_out` is set.
static int ffn_impl(const g53r_layer* L, g53r_scratch* S, const uint8_t* payload, size_t payload_pitch,
                    const uint8_t* scales, size_t scales_pitch, const int32_t* ids, const float* weights, uint32_t rows,
                    uint16_t* bf16_out, float* f32_out, const g53r_cfg* cfg_in, float* ms, char* err, size_t errlen) {
    if (!L || !S || !payload || !scales || !ids || !weights || (!bf16_out == !f32_out) || rows == 0 ||
        rows > MAX_ROWS || payload_pitch < size_t(H) || scales_pitch < size_t(H / 32)) {
        set_msg(err, errlen, "ffn: bad handles, row count or pitches");
        return 1;
    }
    const bool f32 = f32_out != nullptr;
    const size_t out_bytes = size_t(rows) * H * (f32 ? 4 : 2);
    g53r_cfg cfg;
    g53r_default_cfg(rows, &cfg);
    if (cfg_in) {
        if (cfg_in->mt) cfg.mt = cfg_in->mt;
        if (cfg_in->sk) cfg.sk = cfg_in->sk;
        if (cfg_in->skd) cfg.skd = cfg_in->skd;
        cfg.fp32_swiglu = cfg_in->fp32_swiglu;
    }
    const bool ok_cfg = (cfg.mt == 1 || cfg.mt == 2) && cfg.sk > 0 && 32 % cfg.sk == 0 && cfg.skd > 0 && 32 % cfg.skd == 0;
    if (!ok_cfg) {
        set_msg(err, errlen, "ffn: bad configuration (mt 1 or 2; sk and skd divide 32)");
        return 1;
    }
    const int routes = int(rows) * TOPK, gr = 16 * cfg.mt;
    // Groups of one expert hold at most gr pairs: at most routes / gr full groups plus one partial group per
    // distinct expert, and never more groups than pairs.
    const int bound = routes / gr + (routes < E ? routes : E);
    const int max_groups = routes < bound ? routes : bound;
    cudaError_t e;
    auto t0 = std::chrono::steady_clock::now();
#define G53R_CK(expr, what)                  \
    do {                                     \
        if ((e = (expr)) != cudaSuccess) {   \
            set_err(err, errlen, what, e);   \
            return 2;                        \
        }                                    \
    } while (0)
    G53R_CK(S->xrows.ensure(size_t(rows) * ROW_PITCH), "xrows alloc");
    G53R_CK(S->iw.ensure(size_t(routes) * 8), "ids/weights alloc");
    G53R_CK(S->groups.ensure(size_t(max_groups) * sizeof(Group)), "groups alloc");
    G53R_CK(S->meta.ensure(16), "meta alloc");
    G53R_CK(S->pair_row.ensure(size_t(routes) * 4), "pair_row alloc");
    G53R_CK(S->pair_route.ensure(size_t(routes) * 4), "pair_route alloc");
    G53R_CK(S->inverse.ensure(size_t(routes) * 4), "inverse alloc");
    G53R_CK(S->z.ensure(size_t(2) * cfg.sk * routes * WID * 4), "z alloc");
    G53R_CK(S->xd.ensure(size_t(routes) * WID * 2), "xd alloc");
    G53R_CK(S->zd.ensure(size_t(cfg.skd) * routes * H * 4), "zd alloc");
    G53R_CK(S->out.ensure(out_bytes), "out alloc");
    int* const counts = S->meta.as<int>();
    unsigned* const fault = reinterpret_cast<unsigned*>(counts + 1);
    int32_t* const ids_d = S->iw.as<int32_t>();
    float* const w_d = reinterpret_cast<float*>(S->iw.as<uint8_t>() + size_t(routes) * 4);

    G53R_CK(cudaMemsetAsync(S->meta.p, 0, 16, S->st), "meta clear");
    G53R_CK(cudaMemcpy2DAsync(S->xrows.p, ROW_PITCH, payload, payload_pitch, H, rows, cudaMemcpyHostToDevice, S->st),
            "payload upload");
    G53R_CK(cudaMemcpy2DAsync(S->xrows.as<uint8_t>() + H, ROW_PITCH, scales, scales_pitch, H / 32, rows,
                              cudaMemcpyHostToDevice, S->st),
            "scales upload");
    S->hiw.resize(size_t(routes) * 8);
    std::memcpy(S->hiw.data(), ids, size_t(routes) * 4);
    std::memcpy(S->hiw.data() + size_t(routes) * 4, weights, size_t(routes) * 4);
    G53R_CK(cudaMemcpyAsync(S->iw.p, S->hiw.data(), size_t(routes) * 8, cudaMemcpyHostToDevice, S->st),
            "ids/weights upload");
    auto t1 = std::chrono::steady_clock::now();

    G53R_CK(cudaEventRecord(S->e0, S->st), "event 0");
    plan_kernel<<<1, 1024, 0, S->st>>>(ids_d, w_d, int(rows), gr, S->groups.as<Group>(), counts, S->pair_row.as<int>(),
                                       S->pair_route.as<int>(), S->inverse.as<int>(), fault);
    G53R_CK(cudaGetLastError(), "plan launch");
    G53R_CK(cudaEventRecord(S->ep, S->st), "event p");
    const Group* groups = S->groups.as<Group>();
    if (cfg.mt == 1)
        gateup_kernel<1><<<dim3(max_groups, 2, cfg.sk), 128, 0, S->st>>>(L->image, S->xrows.as<uint8_t>(), groups, counts,
                                                                       S->pair_row.as<int>(), S->z.as<float>(), routes,
                                                                       cfg.sk, fault);
    else
        gateup_kernel<2><<<dim3(max_groups, 2, cfg.sk), 128, 0, S->st>>>(L->image, S->xrows.as<uint8_t>(), groups, counts,
                                                                       S->pair_row.as<int>(), S->z.as<float>(), routes,
                                                                       cfg.sk, fault);
    G53R_CK(cudaGetLastError(), "gate/up launch");
    G53R_CK(cudaEventRecord(S->ea, S->st), "event a");
    gateup_epilogue<<<routes, 128, 0, S->st>>>(S->z.as<float>(), S->pair_route.as<int>(), ids_d, L->image, counts,
                                               S->xd.as<half>(), routes, cfg.sk, cfg.fp32_swiglu ? 0 : 1, fault);
    G53R_CK(cudaGetLastError(), "epilogue launch");
    G53R_CK(cudaEventRecord(S->eb, S->st), "event b");
    if (cfg.mt == 1)
        down_kernel<1><<<dim3(max_groups, 8, cfg.skd), 128, 0, S->st>>>(L->image, S->xd.as<half>(), groups, counts,
                                                                      S->zd.as<float>(), routes, cfg.skd);
    else
        down_kernel<2><<<dim3(max_groups, 8, cfg.skd), 128, 0, S->st>>>(L->image, S->xd.as<half>(), groups, counts,
                                                                      S->zd.as<float>(), routes, cfg.skd);
    G53R_CK(cudaGetLastError(), "down launch");
    G53R_CK(cudaEventRecord(S->ec, S->st), "event c");
    if (f32)
        reduce_kernel<true><<<rows * 8, 128, 0, S->st>>>(S->zd.as<float>(), S->inverse.as<int>(), ids_d, w_d, L->image,
                                                         counts, S->out.p, routes, cfg.skd, int(rows), fault);
    else
        reduce_kernel<false><<<rows * 8, 128, 0, S->st>>>(S->zd.as<float>(), S->inverse.as<int>(), ids_d, w_d, L->image,
                                                          counts, S->out.p, routes, cfg.skd, int(rows), fault);
    G53R_CK(cudaGetLastError(), "reduce launch");
    G53R_CK(cudaEventRecord(S->e1, S->st), "event 1");
    G53R_CK(cudaMemcpyAsync(f32 ? static_cast<void*>(f32_out) : static_cast<void*>(bf16_out), S->out.p, out_bytes,
                            cudaMemcpyDeviceToHost, S->st),
            "output download");
    int meta[2] = {0, 0};  // groups, fault word
    G53R_CK(cudaMemcpyAsync(meta, S->meta.p, 8, cudaMemcpyDeviceToHost, S->st), "meta download");
    G53R_CK(cudaStreamSynchronize(S->st), "ffn sync");
#undef G53R_CK
    const unsigned f = unsigned(meta[1]);
    if (f) {
        if (err && errlen)
            std::snprintf(err, errlen, "ffn fault %#x:%s%s%s%s", f, f & F_ROUTE ? " invalid routes (id, weight or duplicate)" : "",
                          f & F_ROW ? " NaN in a wire row" : "", f & F_GATEUP ? " non-finite gate/up output" : "",
                          f & F_OUT ? " non-finite output" : "");
        return 3;
    }
    if (meta[0] <= 0 || meta[0] > max_groups) {
        set_msg(err, errlen, "ffn: the plan's group count is out of range");
        return 3;
    }
    auto t2 = std::chrono::steady_clock::now();
    if (ms) {
        float gpu = 0, ph[5] = {0, 0, 0, 0, 0};
        cudaEventElapsedTime(&gpu, S->e0, S->e1);
        cudaEventElapsedTime(&ph[0], S->e0, S->ep);
        cudaEventElapsedTime(&ph[1], S->ep, S->ea);
        cudaEventElapsedTime(&ph[2], S->ea, S->eb);
        cudaEventElapsedTime(&ph[3], S->eb, S->ec);
        cudaEventElapsedTime(&ph[4], S->ec, S->e1);
        ms[0] = std::chrono::duration<float, std::milli>(t1 - t0).count();
        ms[1] = gpu;
        ms[2] = std::chrono::duration<float, std::milli>(t2 - t1).count() - gpu;
        ms[3] = float(meta[0]);
        for (int i = 0; i < 5; ++i) ms[4 + i] = ph[i];
    }
    return 0;
}

// One rank's FFN for `rows` rows. `payload`: rows of 4,096 E4M3 bytes, `payload_pitch` apart; `scales`: rows of
// 128 UE8M0 bytes, `scales_pitch` apart (4,096 / 128 for separate arrays; the request frame's interleaved rows
// use its row stride for both). `ids`, `weights`: [rows * 8], row-major top-8. Writes the rank's partial rows,
// BF16 [rows * 4,096], to `bf16_out` (host). `cfg` may be null (defaults). `ms` (8 floats, may be null):
// host staging and uploads, GPU time, download and checks, groups; then the GPU phases plan, gate/up,
// epilogue, down, reduce (9 floats in all). Returns 0 on success.
int g53r_ffn(const g53r_layer* L, g53r_scratch* S, const uint8_t* payload, size_t payload_pitch, const uint8_t* scales,
             size_t scales_pitch, const int32_t* ids, const float* weights, uint32_t rows, uint16_t* bf16_out,
             const g53r_cfg* cfg_in, float* ms, char* err, size_t errlen) {
    return ffn_impl(L, S, payload, payload_pitch, scales, scales_pitch, ids, weights, rows, bf16_out, nullptr, cfg_in,
                    ms, err, errlen);
}

// g53r_ffn with the partial rows left in FP32 [rows * 4,096] (`f32_out`, host): the reduce's sums before their
// BF16 rounding, for the prefill reduce-scatter. Rounding them to BF16 (nearest even) gives g53r_ffn's bits.
int g53r_ffn_f32(const g53r_layer* L, g53r_scratch* S, const uint8_t* payload, size_t payload_pitch,
                 const uint8_t* scales, size_t scales_pitch, const int32_t* ids, const float* weights, uint32_t rows,
                 float* f32_out, const g53r_cfg* cfg_in, float* ms, char* err, size_t errlen) {
    return ffn_impl(L, S, payload, payload_pitch, scales, scales_pitch, ids, weights, rows, nullptr, f32_out, cfg_in,
                    ms, err, errlen);
}

// Test hook: copy the last call's intermediates out of the scratch: the gate/up split partials Z
// [2][sk][routes][512] (f32), each grouped pair's route (row * 8 + slot) [routes], the down input Xd
// [routes][512] (FP16 bits), and the down split partials Zd [skd][routes][4096] (f32). Counts are elements.
int g53r_debug_copy(const g53r_scratch* S, float* z, size_t z_count, int32_t* pair_route, size_t pr_count,
                    uint16_t* xd, size_t xd_count, float* zd, size_t zd_count, char* err, size_t errlen) {
    if (!S || z_count * 4 > S->z.n || pr_count * 4 > S->pair_route.n || xd_count * 2 > S->xd.n ||
        zd_count * 4 > S->zd.n) {
        set_msg(err, errlen, "debug copy: counts exceed the scratch");
        return 1;
    }
    cudaError_t e;
    if ((e = cudaMemcpy(z, S->z.p, z_count * 4, cudaMemcpyDeviceToHost)) != cudaSuccess ||
        (e = cudaMemcpy(pair_route, S->pair_route.p, pr_count * 4, cudaMemcpyDeviceToHost)) != cudaSuccess ||
        (e = cudaMemcpy(xd, S->xd.p, xd_count * 2, cudaMemcpyDeviceToHost)) != cudaSuccess ||
        (e = cudaMemcpy(zd, S->zd.p, zd_count * 4, cudaMemcpyDeviceToHost)) != cudaSuccess) {
        set_err(err, errlen, "debug copy", e);
        return 2;
    }
    return 0;
}

}  // extern "C"
