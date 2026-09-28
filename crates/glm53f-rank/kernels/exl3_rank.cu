// glm53f-rank: one expert rank's share of GLM-5.3-Flash's routed-expert FFN, EXL3 K4 (ExLlamaV3 trellis,
// "mcg" codebook) at TP4: every expert's intermediate channels [512 r, 512 r + 512), 288 experts a layer.
//
// Per call: `rows` wire rows (FP8 E4M3 values, one UE8M0 scale per 32) with their top-8 expert ids and FP32
// gate weights  ->  plan (group the rows x 8 routes by expert, on the GPU)  ->  gate/up (the wire row widened
// to FP32, times suh, rotated by a 128-point Hadamard transform, rounded to FP16 and multiplied by the trellis
// tiles on the tensor cores)  ->  epilogue (rotate back, times svh, GLM's clamped SwiGLU with BF16 roundings,
// times the down projection's suh, rotate, FP16)  ->  down (tensor cores again)  ->  reduce (rotate back, times
// svh, the 8 slots summed in slot order with their gate weights, BF16)  =  the rank's partial row.
// `g53r_ffn_f32` leaves that row in FP32, for the prefill reduce-scatter.
//
// Two kernel families share every piece of arithmetic outside the matrix products:
// - the split kernels (`gateup_kernel`, `down_kernel`: 16 or 32 rows a group, four warps of 128 columns, K split
//   into partial sums) serve decode and verify windows; at those sizes the gate/up blocks can plan the call
//   themselves (`self_plan`) instead of waiting for `plan_kernel`, the trellis words are loaded 1, 2 or 4 k tiles
//   ahead, a group's blocks can run together so that the fused steps find the partial sums in L2, and the down
//   kernel can start before the gate/up kernel ends (a programmatic dependent launch, sm_90 and later);
// - the large-M kernels (`gateup_big`, `down_big`: 32 or 64 rows a group, 8 or 16 warps, persistent blocks)
//   serve prefill: each decoded trellis tile feeds up to 8 MMAs, the rotated input rows and the down input are
//   double-buffered in shared memory, the trellis words are loaded several k tiles ahead, and m tiles past the
//   group's rows are skipped.
// The epilogue and the reduce run either as kernels of their own or fused: the last block to finish a group's
// gate/up (or a row's down columns) runs the same epilogue (reduce) code on the partial sums the other blocks
// left in L2, and can drop those lines from L2 afterwards instead of letting them be written back.
//
// Sources. The tile decoder (`mcg2`, `decode_tile`), the MMA wrapper and the warp butterfly (`fwht128`) are
// TensorFold's `src/tensorfold/families/glm5_next/cuda/exl3.cu` at bb4b4a3 (MIT, Copyright (c) 2026 TensorFold
// contributors; LICENSE.tensorfold), verbatim. TensorFold's kernels read the format of ExLlamaV3
// (https://github.com/turboderp-org/exllamav3, MIT, Copyright (c) 2025 Turboderp). The split GEMMs follow
// TensorFold's `grouped_kernel` (a warp decodes 16x16 tiles straight into B fragments; K splits summed in a
// fixed order) with these changes: TP4 slices (N = 512 for gate/up, K = 512 for down); each block's four
// warps cover the slice's four 128-column Hadamard blocks, so the input rotation is computed once per block
// in shared memory, straight from the FP8 wire row (no rotated-input buffer); up to two 16-row tiles per
// block for prefill; the next k tiles' weights are loaded while the current one is multiplied. The planner
// takes MiMo's one-CTA route plan (mimo26f-afd v1.2.0 `crates/mimo26-spark/kernels/b1_serve.cu`, MIT) to 288
// experts. The large-M kernels, the plan in the gate/up blocks, the fused epilogue and reduce, and the split
// kernels' prefetch ring, block orders and dependent launch are written here. See PROVENANCE.md.
//
// Every output depends only on its own row: the tensor cores keep rows independent, and every sum (K splits,
// the 8 slots) runs in a fixed order. With the same K splits (SK, SKD) and SwiGLU a row gets the same bits
// whatever else is in the batch; the kernel family, the rows per group, the tilings, the plan, the fusion, the
// L2 policies, the prefetch depth, the block order and the dependent launch do not change a bit (the same
// products are accumulated in the same order into the same sums). Build with --fmad=false: the CPU reference
// (src/reference.rs) repeats the epilogues' separate multiplies and adds.

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

// The A fragment of an m16n8k16 MMA (rows `r0..r0+15`, columns `c0..c0+15` of a row-major FP16 tile in shared
// memory) with one ldmatrix: lane L gives the address of row (L & 7) + 8 ((L >> 3) & 1), column 8 (L >> 4).
// The same registers as four 32-bit loads (a[0] rows g, cols 2t; a[1] rows g + 8; a[2], a[3] cols + 8).
__device__ __forceinline__ void ldsm_a(uint32_t (&a)[4], const half* p) {
    const uint32_t s = static_cast<uint32_t>(__cvta_generic_to_shared(p));
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(a[0]), "=r"(a[1]), "=r"(a[2]), "=r"(a[3])
                 : "r"(s));
}

__device__ __forceinline__ void cp_async16(void* smem, const void* gmem) {
    const uint32_t s = static_cast<uint32_t>(__cvta_generic_to_shared(smem));
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(s), "l"(gmem) : "memory");
}
__device__ __forceinline__ void cp_async_commit() { asm volatile("cp.async.commit_group;\n" ::: "memory"); }
template <int N>
__device__ __forceinline__ void cp_async_wait() { asm volatile("cp.async.wait_group %0;\n" ::"n"(N) : "memory"); }

// A trellis word through L2 with an evict-first policy (`pol` from `evict_first_policy`): streamed weights then
// give way to the partial sums and down inputs that are read again soon. Plain `__ldg` otherwise.
__device__ __forceinline__ uint64_t evict_first_policy() {
    uint64_t pol = 0;
#if __CUDA_ARCH__ >= 800
    asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;\n" : "=l"(pol));
#endif
    return pol;
}

__device__ __forceinline__ uint32_t ldg_w(const uint32_t* p, bool hint, uint64_t pol) {
#if __CUDA_ARCH__ >= 800
    if (hint) {
        uint32_t v;
        asm("ld.global.nc.L2::cache_hint.b32 %0, [%1], %2;\n" : "=r"(v) : "l"(p), "l"(pol));
        return v;
    }
#endif
    return __ldg(p);
}

// Programmatic dependent launch (sm_90 and later; nothing on older devices, where the host launches the dependent
// kernel in stream order): the gate/up kernel lets the down kernel start early, and the down kernel waits for the
// gate/up grid to complete (its memory visible) before it reads what that grid wrote.
__device__ __forceinline__ void grid_launch_dependents() {
#if __CUDA_ARCH__ >= 900
    asm volatile("griddepcontrol.launch_dependents;\n" ::: "memory");
#endif
}
__device__ __forceinline__ void grid_dependency_wait() {
#if __CUDA_ARCH__ >= 900
    asm volatile("griddepcontrol.wait;\n" ::: "memory");
#endif
}

// Fetch the 128-byte line holding `p` into L2 (no register is written, nothing waits for it).
__device__ __forceinline__ void prefetch_l2(const void* p) { asm volatile("prefetch.global.L2 [%0];\n" ::"l"(p)); }

// Drop one 128-byte line of scratch from L2 without writing it back (its contents become undefined).
__device__ __forceinline__ void discard_line(const void* p) {
#if __CUDA_ARCH__ >= 800
    asm volatile("discard.global.L2 [%0], 128;\n" ::"l"(p) : "memory");
#endif
}

// Everything a call's kernels read and write (passed by value to every kernel).
struct Args {
    const uint8_t* image;    // layer image
    const uint8_t* xrows;    // wire rows [rows][ROW_PITCH]
    const Group* groups;
    const int* counts;       // counts[0]: groups the plan formed (0: invalid routes)
    const int* pair_row;     // per grouped pair: its row
    const int* pair_route;   // per grouped pair: its route (row * 8 + slot)
    const int* inverse;      // per route: its grouped pair
    const int32_t* ids;      // [rows * 8]
    const float* wts;        // [rows * 8]
    float* z;                // gate/up split partials [2][sk][P][512]
    half* xd;                // down input [P][512]
    float* zd;               // down split partials [skd][P][4096]
    void* out;               // [rows][4096], BF16 or FP32
    int* gcnt;               // fused epilogue: arrivals per group
    int* rcnt;               // fused reduce: arrivals per (row, output chunk)
    int* work;               // large-M kernels: the next item of gate/up [0] and down [1] (zero at the start)
    unsigned* fault;
    int P, rows, sk, skd;
    int bf16;                // BF16 SwiGLU (the default) or FP32
    int f32;                 // FP32 output (g53r_ffn_f32)
    int fused;               // epilogue and reduce fused into gate/up and down
    int discard;             // drop consumed partial sums from L2
    int nch;                 // down output chunks (4,096 / chunk width)
    int selfplan;            // the split gate/up blocks plan the call (no plan_kernel)
    int l2hint;              // trellis words loaded evict-first
    int order;               // split kernels' block order: 1 split slowest, 2 a group's (a chunk's) blocks adjacent
    int pdl;                 // the down blocks plan themselves and start before the gate/up kernel ends
};

// ---- the arithmetic outside the matrix products, shared by every kernel ---------------------------------------

// A lane's inputs for rotating one wire row's 128-wide K block `kb`: 4 E4M3 codes, their UE8M0 scale byte and
// 4 suh values, loaded ahead of `row_rotate` so that several rows' loads are in flight at once.
struct RowIn {
    uint32_t q, sb;
    uint2 su;
};

__device__ __forceinline__ RowIn row_load(const uint8_t* __restrict__ xr, const half* __restrict__ suh, int kb, int lane) {
    RowIn r;
    r.q = *reinterpret_cast<const uint32_t*>(xr + kb * 128 + lane * 4);
    r.sb = xr[H + kb * 4 + (lane >> 3)];
    r.su = *reinterpret_cast<const uint2*>(suh + kb * 128 + lane * 4);
    return r;
}

// The row block rotated for the tensor cores: E4M3 widened to FP32, x UE8M0 scale, x suh, the butterfly in FP32,
// x 1/sqrt(128), FP16. Lane L gets values 4L..4L+3. `bad` collects NaN codes.
__device__ __forceinline__ uint2 row_rotate(const RowIn& in, int lane, bool& bad) {
    bad = bad || in.sb == 0xFFu;
    const float s = ue8m0(in.sb);
    const half2 s01 = *reinterpret_cast<const half2*>(&in.su.x), s23 = *reinterpret_cast<const half2*>(&in.su.y);
    const float su[4] = {__low2float(s01), __high2float(s01), __low2float(s23), __high2float(s23)};
    float v[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const uint32_t c = (in.q >> (8 * j)) & 0xFFu;
        bad = bad || (c & 0x7Fu) == 0x7Fu;
        v[j] = e4m3(c) * s * su[j];
    }
    fwht128(v, lane);
    const half2 h0 = __floats2half2_rn(v[0] * HAD_SCALE, v[1] * HAD_SCALE);
    const half2 h1 = __floats2half2_rn(v[2] * HAD_SCALE, v[3] * HAD_SCALE);
    return make_uint2(*reinterpret_cast<const uint32_t*>(&h0), *reinterpret_cast<const uint32_t*>(&h1));
}

__device__ __forceinline__ uint2 rotate_row(const uint8_t* __restrict__ xr, const half* __restrict__ suh, int kb,
                                            int lane, bool& bad) {
    return row_rotate(row_load(xr, suh, kb, lane), lane, bad);
}

// Four FP16 values (8 bytes) as floats.
__device__ __forceinline__ void half4(uint2 h, float (&f)[4]) {
    const half2 h01 = *reinterpret_cast<const half2*>(&h.x), h23 = *reinterpret_cast<const half2*>(&h.y);
    f[0] = __low2float(h01);
    f[1] = __high2float(h01);
    f[2] = __low2float(h23);
    f[3] = __high2float(h23);
}

// The gate/up epilogue of one (pair gp, 128-block blk), one warp (after TensorFold's gateup_epilogue_kernel): the
// splits summed in order, rotated back, x svh, GLM's clamped SwiGLU, x down suh, rotated, FP16 into Xd. `ex` is
// the pair's expert block. Returns whether a gate/up sum was not finite. The scale vectors are loaded with the
// first splits' partial sums, not one value at a time after them.
__device__ __forceinline__ bool epilogue_item(const Args& a, const uint8_t* __restrict__ ex, int gp, int blk, int lane) {
    const int n = blk * 128 + lane * 4;
    const uint2 sg = *reinterpret_cast<const uint2*>(ex + OFF_SVH_G + 2 * n);
    const uint2 su = *reinterpret_cast<const uint2*>(ex + OFF_SVH_U + 2 * n);
    const uint2 sd = *reinterpret_cast<const uint2*>(ex + OFF_SUH_D + 2 * n);
    float gv[4] = {0.f, 0.f, 0.f, 0.f}, uv[4] = {0.f, 0.f, 0.f, 0.f};
    for (int s = 0; s < a.sk; ++s) {
        const float4 zg = __ldcg(reinterpret_cast<const float4*>(a.z + (size_t(0 * a.sk + s) * a.P + gp) * WID + n));
        const float4 zu = __ldcg(reinterpret_cast<const float4*>(a.z + (size_t(1 * a.sk + s) * a.P + gp) * WID + n));
        gv[0] += zg.x; gv[1] += zg.y; gv[2] += zg.z; gv[3] += zg.w;
        uv[0] += zu.x; uv[1] += zu.y; uv[2] += zu.z; uv[3] += zu.w;
    }
    float svh_g[4], svh_u[4], suh_d[4];
    half4(sg, svh_g);
    half4(su, svh_u);
    half4(sd, suh_d);
    bool bad = false;
#pragma unroll
    for (int j = 0; j < 4; ++j) bad = bad || !isfinite(gv[j]) || !isfinite(uv[j]);
    fwht128(gv, lane);
    fwht128(uv, lane);
    float v[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        float gg = gv[j] * HAD_SCALE * svh_g[j];
        float uu = uv[j] * HAD_SCALE * svh_u[j];
        float act;
        if (a.bf16) {  // the reference model's BF16 SwiGLU (TensorFold's default)
            gg = fminf(bf16r(gg), LIMIT);
            uu = fminf(fmaxf(bf16r(uu), -LIMIT), LIMIT);
            act = bf16r(bf16r(gg / (1.f + expf(-gg))) * uu);
        } else {       // FP32 throughout (an option for the KL gate and for tight kernel tests)
            gg = fminf(gg, LIMIT);
            uu = fminf(fmaxf(uu, -LIMIT), LIMIT);
            act = gg / (1.f + expf(-gg)) * uu;
        }
        v[j] = act * suh_d[j];
    }
    fwht128(v, lane);
    const half2 h0 = __floats2half2_rn(v[0] * HAD_SCALE, v[1] * HAD_SCALE);
    const half2 h1 = __floats2half2_rn(v[2] * HAD_SCALE, v[3] * HAD_SCALE);
    *reinterpret_cast<uint2*>(a.xd + size_t(gp) * WID + n) =
        make_uint2(*reinterpret_cast<const uint32_t*>(&h0), *reinterpret_cast<const uint32_t*>(&h1));
    return bad;
}

// The route reduce of one (row, 128-block blk), one warp: out[row] = sum over the 8 slots, in slot order, of
// w * (rot(sum of the down splits) * svh), in FP32, then BF16 (or left in FP32). Every slot's inputs are loaded
// before the first is used. Returns whether a sum was not finite.
__device__ __forceinline__ bool reduce_item(const Args& a, int row, int blk, int lane) {
    const int n = blk * 128 + lane * 4;
    int gp[TOPK];
    float wt[TOPK];
    uint2 sv[TOPK];
    float4 z0[TOPK];
#pragma unroll
    for (int s = 0; s < TOPK; ++s) {
        const int r = row * TOPK + s;
        gp[s] = a.inverse[r];
        wt[s] = a.wts[r];
        sv[s] = *reinterpret_cast<const uint2*>(
            reinterpret_cast<const half*>(a.image + size_t(a.ids[r]) * EXPERT_BYTES + OFF_SVH_D) + n);
    }
#pragma unroll
    for (int s = 0; s < TOPK; ++s) z0[s] = __ldcg(reinterpret_cast<const float4*>(a.zd + size_t(gp[s]) * H + n));
    float acc[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
    for (int s = 0; s < TOPK; ++s) {
        float v[4] = {0.f + z0[s].x, 0.f + z0[s].y, 0.f + z0[s].z, 0.f + z0[s].w};
        for (int k = 1; k < a.skd; ++k) {
            const float4 t = __ldcg(reinterpret_cast<const float4*>(a.zd + (size_t(k) * a.P + gp[s]) * H + n));
            v[0] += t.x; v[1] += t.y; v[2] += t.z; v[3] += t.w;
        }
        fwht128(v, lane);
        const half2 s01 = *reinterpret_cast<const half2*>(&sv[s].x), s23 = *reinterpret_cast<const half2*>(&sv[s].y);
        const float svh[4] = {__low2float(s01), __high2float(s01), __low2float(s23), __high2float(s23)};
#pragma unroll
        for (int j = 0; j < 4; ++j) acc[j] = acc[j] + wt[s] * (v[j] * HAD_SCALE * svh[j]);
    }
    bool bad = false;
#pragma unroll
    for (int j = 0; j < 4; ++j) bad = bad || !isfinite(acc[j]);
    if (a.f32) {
        *reinterpret_cast<float4*>(static_cast<float*>(a.out) + size_t(row) * H + n) =
            make_float4(acc[0], acc[1], acc[2], acc[3]);
    } else {
        uint16_t o[4];
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const __nv_bfloat16 b = __float2bfloat16_rn(acc[j]);
            o[j] = *reinterpret_cast<const uint16_t*>(&b);
        }
        *reinterpret_cast<uint2*>(static_cast<uint16_t*>(a.out) + size_t(row) * H + n) =
            make_uint2(uint32_t(o[0]) | (uint32_t(o[1]) << 16), uint32_t(o[2]) | (uint32_t(o[3]) << 16));
    }
    return bad;
}

// Fused epilogue: after a gate/up block has stored its partial sums, the last of the group's 2 * SK blocks (both
// matrices, every split) runs the epilogue for all of the group's pairs, reading the others' partials from L2
// (`last` is a shared flag). It also clears the group's counter for the next call (counters are zeroed when
// allocated and every arrival of a call comes before its last). Called by every thread of the block.
template <int NW>
__device__ __forceinline__ void gateup_tail(const Args& a, const Group& G, int g, const uint8_t* ex, int* last) {
    __threadfence();
    __syncthreads();
    if (threadIdx.x == 0) {
        *last = atomicAdd(&a.gcnt[g], 1) == 2 * a.sk - 1;
        if (*last) a.gcnt[g] = 0;
    }
    __syncthreads();
    if (!*last) return;
    __threadfence();
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    bool bad = false;
    for (int item = warp; item < G.count * 4; item += NW) bad = epilogue_item(a, ex, G.start + (item >> 2), item & 3, lane) || bad;
    if (__any_sync(0xffffffffu, bad) && lane == 0) atomicOr(a.fault, F_GATEUP);
    if (a.discard) {
        __syncthreads();  // every item has read its partials
        const int lines = 2 * a.sk * G.count * (WID / 32);  // 16 lines of 128 bytes per pair and (matrix, split)
        for (int i = threadIdx.x; i < lines; i += NW * 32) {
            const int line = i & 15, rest = i >> 4, pr = rest % G.count, ms = rest / G.count;
            discard_line(a.z + (size_t(ms) * a.P + G.start + pr) * WID + line * 32);
        }
    }
}

// Fused reduce: after a down block has stored its partial sums (chunk `chunk` of the output, `cwb` 128-blocks
// wide), each (row, chunk) of its group counts one arrival; the block that brings a (row, chunk) to all 8 * SKD
// arrivals (8 slots, every split) reduces it and clears its counter. Thread t holds the row of the group's pair t
// in `my_row` (loaded when the block started; a group has at most 64 pairs), -1 past the group's pairs.
// `list` holds up to the group's pair count, `nl` is shared.
template <int NW>
__device__ __forceinline__ void down_tail(const Args& a, int chunk, int cwb, int my_row, int* list, int* nl) {
    static_assert(NW * 32 >= 64, "a thread per pair of a group of up to 64");
    __threadfence();
    if (threadIdx.x == 0) *nl = 0;
    __syncthreads();
    const int target = TOPK * a.skd;
    if (my_row >= 0 && atomicAdd(&a.rcnt[my_row * a.nch + chunk], 1) == target - 1) {
        a.rcnt[my_row * a.nch + chunk] = 0;
        list[atomicAdd(nl, 1)] = my_row;
    }
    __syncthreads();
    const int nrows = *nl;
    if (nrows == 0) return;
    __threadfence();
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    bool bad = false;
    for (int item = warp; item < nrows * cwb; item += NW)
        bad = reduce_item(a, list[item / cwb], chunk * cwb + item % cwb, lane) || bad;
    if (__any_sync(0xffffffffu, bad) && lane == 0) atomicOr(a.fault, F_OUT);
    if (a.discard) {
        __syncthreads();  // every item has read its partials
        const int lines = cwb * 4, per = TOPK * a.skd * lines;  // 4 lines of 128 bytes per 128-block
        for (int i = threadIdx.x; i < nrows * per; i += NW * 32) {
            const int row = list[i / per], rem = i % per, line = rem % lines, ks = rem / lines;
            const int k = ks % a.skd, s = ks / a.skd;
            discard_line(a.zd + (size_t(k) * a.P + a.inverse[row * TOPK + s]) * H + chunk * cwb * 128 + line * 32);
        }
    }
}

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

// ---- the plan inside the split gate/up blocks (decode sizes, up to 512 routes) --------------------------------
// Every block of the split gate/up kernel can plan the call itself instead of waiting for plan_kernel: it checks
// the routes as the planner does, counts the pairs per expert, and finds its group. Each expert's pairs are taken
// in route order, so every block agrees on the plan; the group's (mat 0, split 0) block writes its part of the
// plan for the down kernel (groups, pair_row, pair_route, inverse) and group 0's block the group count. Returns
// false (in every thread) for invalid routes or a block past the last group.
struct PlanSmem {
    int ids[512];
    int cnt[E], pbase[E], gbase[E];
    int wsum[4], wgsum[4];
    int bad, ngroups;
    Group G;
};

// Exclusive prefix sums over a 128-thread block of two values (x, y); `ws`, `wgs` are 4-entry scratch.
__device__ __forceinline__ void block_scan128(int& x, int& y, int* ws, int* wgs) {
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    int ix = x, iy = y;
#pragma unroll
    for (int d = 1; d < 32; d <<= 1) {
        const int u = __shfl_up_sync(0xffffffffu, ix, d), v = __shfl_up_sync(0xffffffffu, iy, d);
        if (lane >= d) { ix += u; iy += v; }
    }
    if (lane == 31) { ws[warp] = ix; wgs[warp] = iy; }
    __syncthreads();
    int ox = 0, oy = 0;
    for (int w = 0; w < warp; ++w) { ox += ws[w]; oy += wgs[w]; }
    x = ox + ix - x;
    y = oy + iy - y;
    __syncthreads();
}

__device__ bool self_plan(const Args& a, int g, int gr, bool writer, int* rows_sh, int R, PlanSmem& sm) {
    const int tid = threadIdx.x, routes = a.rows * TOPK;
    for (int e = tid; e < E; e += 128) sm.cnt[e] = 0;
    if (tid == 0) sm.bad = 0;
    // Thread t checks routes t, t + 128, ... (ids and weights loaded together, one memory round trip).
    bool wok[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const int r = tid + 128 * j;
        if (r < routes) {
            const int e = a.ids[r];
            const float wt = a.wts[r];
            sm.ids[r] = e;
            wok[j] = e >= 0 && e < E && wt >= 0.0f && wt <= 0x1.fffffep127f;  // finite, not negative, not NaN
        }
    }
    __syncthreads();
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const int r = tid + 128 * j;
        if (r < routes) {
            const int e = sm.ids[r];
            bool ok = wok[j];
            if (ok)
                for (int k = (r / TOPK) * TOPK; k < r; ++k) ok = ok && sm.ids[k] != e;  // no expert twice in a row
            if (!ok) atomicOr(&sm.bad, 1); else atomicAdd(&sm.cnt[e], 1);
        }
    }
    __syncthreads();
    if (sm.bad) {
        if (tid == 0 && g == 0 && writer) atomicOr(a.fault, F_ROUTE);
        return false;
    }
    // Pair and group bases per expert: thread t sums experts 3t .. 3t + 2 (t < 96), then a block scan.
    int c3 = 0, g3 = 0;
    if (tid < E / 3)
        for (int j = 0; j < 3; ++j) {
            const int c = sm.cnt[3 * tid + j];
            c3 += c;
            g3 += (c + gr - 1) / gr;
        }
    int pb = c3, gb = g3;
    block_scan128(pb, gb, sm.wsum, sm.wgsum);
    if (tid < E / 3)
        for (int j = 0; j < 3; ++j) {
            const int e = 3 * tid + j, c = sm.cnt[e], gc = (c + gr - 1) / gr;
            sm.pbase[e] = pb;
            sm.gbase[e] = gb;
            if (g >= gb && g < gb + gc) {
                const int k = g - gb;
                sm.G = Group{e, pb + k * gr, c - k * gr < gr ? c - k * gr : gr, k};  // pad: the group's index in its expert
            }
            if (e == E - 1) sm.ngroups = gb + gc;
            pb += c;
            gb += gc;
        }
    __syncthreads();
    if (g >= sm.ngroups) return false;
    const Group G = sm.G;
    if (writer) {
        if (tid == 0) const_cast<Group*>(a.groups)[g] = Group{G.expert, G.start, G.count, 0};
        if (tid == 0 && g == 0) const_cast<int*>(a.counts)[0] = sm.ngroups;
    }
    // The expert's routes ranked in route order (thread t holds routes 4t .. 4t + 3); ranks [k gr, k gr + count)
    // are this group's pairs.
    int f[4], n = 0;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const int r = 4 * tid + j;
        f[j] = r < routes && sm.ids[r] == G.expert;
        n += f[j];
    }
    int rank = n, dummy = 0;
    block_scan128(rank, dummy, sm.wsum, sm.wgsum);
    for (int i = tid; i < R; i += 128) rows_sh[i] = -1;
    __syncthreads();
    const int lo = G.pad * gr;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        if (f[j]) {
            const int r = 4 * tid + j, slot = rank - lo;
            if (slot >= 0 && slot < G.count) {
                rows_sh[slot] = r / TOPK;
                if (writer) {
                    const int gp = G.start + slot;
                    const_cast<int*>(a.pair_row)[gp] = r / TOPK;
                    const_cast<int*>(a.pair_route)[gp] = r;
                    const_cast<int*>(a.inverse)[r] = gp;
                }
            }
            ++rank;
        }
    }
    __syncthreads();
    return true;
}

// ---- gate and up, split kernel: Z[mat][split][pair][512] = rot(x * suh) @ W_q over this split's K range --------
// Block (mat, group, split), 4 warps; warp w owns the slice's columns [128 w, 128 w + 128) (8 n tiles). For each
// 128-wide K block the warps first rotate the block's rows into shared memory (one warp per row), then multiply.
// MT 16-row tiles per block. The trellis words are loaded PF k tiles ahead, each register refilled as soon as its
// word is decoded, with the L2 evict-first policy if HINT (a template parameter: a runtime choice costs a second
// load instruction per word). Blocks are (mat, group, split) with the split slowest, or with `order` 2 (grid
// (2 SK, groups)) a group's 2 SK blocks are adjacent, so the fused epilogue reads their partial sums soon after
// they are written.
template <int MT, int PF, bool HINT>
__global__ void __launch_bounds__(128) gateup_kernel(const Args a) {
    constexpr int R = MT * 16;
    static_assert(8 % PF == 0, "k tiles of a K block are taken PF at a time");
    if (a.pdl) grid_launch_dependents();  // every block has started: the down kernel may be scheduled
    const bool o2 = a.order == 2;
    const int mat = o2 ? int(blockIdx.x & 1) : int(blockIdx.x), g = blockIdx.y;
    const int split = o2 ? int(blockIdx.x >> 1) : int(blockIdx.z);
    __shared__ __align__(16) half As[R][136];  // 272-byte rows: conflict-free fragment loads
    __shared__ int rows_sh[R];
    __shared__ int last_sh;
    __shared__ PlanSmem plan_sh;
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, g8 = lane >> 2, t4 = lane & 3;
    Group G;
    if (a.selfplan) {
        if (!self_plan(a, g, R, mat == 0 && split == 0, rows_sh, R, plan_sh)) return;
        G = plan_sh.G;
    } else {
        if (g >= a.counts[0]) return;
        G = a.groups[g];
        if (tid < R) rows_sh[tid] = tid < G.count ? a.pair_row[G.start + tid] : -1;
        __syncthreads();
    }

    const uint8_t* ex = a.image + size_t(G.expert) * EXPERT_BYTES;
    const uint32_t* T = reinterpret_cast<const uint32_t*>(ex + (mat ? OFF_UP_T : OFF_GATE_T));
    const half* suh = reinterpret_cast<const half*>(ex + (mat ? OFF_SUH_U : OFF_SUH_G));
    const int kbs = 32 / a.sk;  // 128-wide K blocks per split
    const int kt_begin = split * kbs * 8, kt_end = kt_begin + kbs * 8;
    // Only this block reads the suh values of the split's later K blocks (two 128-byte lines each): fetch them into
    // L2 now, so each K block's rotation does not wait for memory. The group's first block does the same for the
    // scale vectors of the fused epilogue (svh of gate and up, suh of down: 1 KB each).
    for (int i = 2 + tid; i < 2 * kbs; i += 128) prefetch_l2(suh + (split * kbs) * 128 + i * 64);
    if (a.fused && mat == 0 && split == 0 && tid < 24) prefetch_l2(ex + OFF_SVH_G + tid * 128);

    float acc[MT][8][2][4];
#pragma unroll
    for (int m = 0; m < MT; ++m)
#pragma unroll
        for (int i = 0; i < 8; ++i)
#pragma unroll
            for (int h = 0; h < 2; ++h)
#pragma unroll
                for (int c = 0; c < 4; ++c) acc[m][i][h][c] = 0.f;

    // Trellis words of tile (kt, nt) are at (kt * 32 + nt) * 32 + lane; this warp's 8 n tiles are contiguous. The
    // ring: tile kt's words in wr[(kt - kt_begin) % PF] (a split has at least 8 k tiles).
    const uint64_t pol = HINT ? evict_first_policy() : 0;
    const uint32_t* Tw = T + size_t(warp * 8) * 32 + lane;
    uint32_t wr[PF][8];
#pragma unroll
    for (int p = 0; p < PF; ++p)
#pragma unroll
        for (int i = 0; i < 8; ++i) wr[p][i] = ldg_w(Tw + (size_t(kt_begin + p) * 32 + i) * 32, HINT, pol);
    bool bad = false;

    for (int kt = kt_begin; kt < kt_end; kt += PF) {
        if ((kt & 7) == 0) {
            __syncthreads();  // the previous block's fragments have been read
            const int kb = kt >> 3;
            // Every row's inputs are loaded before the first is rotated: one memory round trip, not R / 4.
            RowIn in[R / 4];
#pragma unroll
            for (int j = 0; j < R / 4; ++j) {
                const int r = rows_sh[warp + 4 * j];
                if (r >= 0) in[j] = row_load(a.xrows + size_t(r) * ROW_PITCH, suh, kb, lane);
            }
#pragma unroll
            for (int j = 0; j < R / 4; ++j) {
                const int rr = warp + 4 * j, r = rows_sh[rr];
                const uint2 packed = r >= 0 ? row_rotate(in[j], lane, bad) : make_uint2(0u, 0u);
                *reinterpret_cast<uint2*>(&As[rr][lane * 4]) = packed;
            }
            __syncthreads();
        }
#pragma unroll
        for (int p = 0; p < PF; ++p) {
            const int ktl = (kt + p) & 7;
            uint32_t af[MT][4];
#pragma unroll
            for (int m = 0; m < MT; ++m) {
                const half* r0 = &As[m * 16 + g8][ktl * 16 + 2 * t4];
                const half* r1 = &As[m * 16 + g8 + 8][ktl * 16 + 2 * t4];
                af[m][0] = ld32(r0);
                af[m][1] = ld32(r1);
                af[m][2] = ld32(r0 + 8);
                af[m][3] = ld32(r1 + 8);
            }
            // Tile kt + p + PF, or the split's last tile again past its end (the loop has no branch).
            const int ktn = kt + p + PF < kt_end ? kt + p + PF : kt_end - 1;
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                uint32_t b0[2], b1[2];
                decode_tile(wr[p][i], lane, b0, b1);
                wr[p][i] = ldg_w(Tw + (size_t(ktn) * 32 + i) * 32, HINT, pol);
#pragma unroll
                for (int m = 0; m < MT; ++m) {
                    mma16816(acc[m][i][0], af[m], b0);
                    mma16816(acc[m][i][1], af[m], b1);
                }
            }
        }
    }
    if (bad) atomicOr(a.fault, F_ROW);

    float* zbase = a.z + (size_t(mat) * a.sk + split) * size_t(a.P) * WID;
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
    if (a.fused) gateup_tail<4>(a, G, g, ex, &last_sh);
}

// ---- gate and up, large-M kernel -----------------------------------------------------------------------------
// Persistent blocks of NW MMA warps (8 or 16) and NP rotation warps (0 or 2) take (mat, group, split) items in
// grid order (launch_persistent). An item covers the group's rows (up to 16 MT) and one matrix's 512 columns;
// MMA warp w owns columns [16 NT w, 16 NT (w + 1)) with NT = 32 / NW, so each decoded tile feeds 2 MT MMAs. The
// rotated rows are double-buffered in shared memory with one barrier per 128-wide K block: while the MMA warps
// multiply one K block, the next one is rotated, either by the rotation warps (NP > 0: the MMA warps then only
// load, decode and multiply) or by the MMA warps themselves, one row per k tile of the first R / NW k tiles with
// its inputs loaded a k tile ahead (NP = 0). The trellis words are loaded PF k tiles ahead; the A fragments come
// from ldmatrix and the MMAs of m tiles past the group's rows are predicated off. The products are the split
// kernel's, accumulated in the same order.
template <int MT, int NW, int NP>
__global__ void __launch_bounds__((NW + NP) * 32, (MT == 2 && NW == 8 && NP == 0) ? 2 : 1) gateup_big(const Args a) {
    constexpr int NT = 32 / NW, R = MT * 16, RPW = NP ? 0 : R / NW, PF = NT >= 4 ? 2 : 4;
    static_assert(NP > 0 || (RPW >= 1 && RPW <= 8), "a row a warp per k tile rotates a whole K block");
    __shared__ __align__(16) half As[2][R][136];
    __shared__ int rows_sh[R];
    __shared__ int last_sh, item_sh;
    // Persistent blocks take (mat, group, split) items in grid order, the matrix varying fastest.
    const int ngroups = a.counts[0], items = 2 * ngroups * a.sk;
    for (;;) {
        if (threadIdx.x == 0) item_sh = atomicAdd(a.work, 1);
        __syncthreads();
        const int item = item_sh;
        if (item >= items) break;
        const int mat = item & 1, g = (item >> 1) % ngroups, split = (item >> 1) / ngroups;
        const Group G = a.groups[g];
        const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, g8 = lane >> 2, t4 = lane & 3;
        if (tid < R) rows_sh[tid] = tid < G.count ? a.pair_row[G.start + tid] : -1;
        __syncthreads();

        const uint8_t* ex = a.image + size_t(G.expert) * EXPERT_BYTES;
        const uint32_t* T = reinterpret_cast<const uint32_t*>(ex + (mat ? OFF_UP_T : OFF_GATE_T));
        const half* suh = reinterpret_cast<const half*>(ex + (mat ? OFF_SUH_U : OFF_SUH_G));
        const int mt_used = (G.count + 15) >> 4, rows_used = mt_used * 16;
        const int kbs = 32 / a.sk, kb0 = split * kbs, kt0 = kb0 * 8, kt_last = kt0 + kbs * 8 - 1;
        bool bad = false;

        // The first K block, every warp a row at a time.
        for (int rr = warp; rr < rows_used; rr += NW + NP) {
            const int r = rows_sh[rr];
            const uint2 packed = r >= 0 ? rotate_row(a.xrows + size_t(r) * ROW_PITCH, suh, kb0, lane, bad) : make_uint2(0u, 0u);
            *reinterpret_cast<uint2*>(&As[0][rr][lane * 4]) = packed;
        }

        if (warp >= NW) {
            // Rotation warps: during K block kbi, rotate K block kbi + 1 into the other buffer, 4 rows at a time with
            // every row's inputs loaded before the first is rotated.
            const int pw = warp - NW;
            for (int kbi = 0; kbi < kbs; ++kbi) {
                __syncthreads();
                if (kbi + 1 < kbs) {
                    const int kb = kb0 + kbi + 1, buf = (kbi + 1) & 1;
                    for (int r0 = pw; r0 < rows_used; r0 += 4 * NP) {
                        RowIn in[4];
#pragma unroll
                        for (int j = 0; j < 4; ++j) {
                            const int rr = r0 + j * NP;
                            const int r = rr < rows_used ? rows_sh[rr] : -1;
                            if (r >= 0) in[j] = row_load(a.xrows + size_t(r) * ROW_PITCH, suh, kb, lane);
                        }
#pragma unroll
                        for (int j = 0; j < 4; ++j) {
                            const int rr = r0 + j * NP;
                            if (rr < rows_used) {
                                const int r = rows_sh[rr];
                                const uint2 packed = r >= 0 ? row_rotate(in[j], lane, bad) : make_uint2(0u, 0u);
                                *reinterpret_cast<uint2*>(&As[buf][rr][lane * 4]) = packed;
                            }
                        }
                    }
                }
            }
        } else {
            // Trellis words of tile (kt, nt) are at (kt * 32 + nt) * 32 + lane; this warp's NT n tiles are contiguous.
            // A ring of PF k tiles of words in flight.
            const uint32_t* Tw = T + size_t(warp * NT) * 32 + lane;
            const bool hint = a.l2hint != 0;
            const uint64_t pol = hint ? evict_first_policy() : 0;
            uint32_t wr[PF][NT];
#pragma unroll
            for (int p = 0; p < PF; ++p)
#pragma unroll
                for (int i = 0; i < NT; ++i)
                    wr[p][i] = ldg_w(Tw + (size_t(kt0 + p < kt_last ? kt0 + p : kt_last) * 32 + i) * 32, hint, pol);
            // Without rotation warps: the inputs of row warp + NW j of K block kb0 + kbi + 1 (rows past the group read
            // row 0's inputs; their results are not stored).
            auto load_next = [&](int kbi, int j) -> RowIn {
                const int r = rows_sh[warp + NW * j];
                const int kb = kb0 + kbi + 1 < 32 ? kb0 + kbi + 1 : 31;
                return row_load(a.xrows + size_t(r >= 0 ? r : rows_sh[0]) * ROW_PITCH, suh, kb, lane);
            };
            RowIn nx;
            if constexpr (RPW > 0) nx = load_next(0, 0);

            float acc[MT][NT][2][4];
#pragma unroll
            for (int m = 0; m < MT; ++m)
#pragma unroll
                for (int i = 0; i < NT; ++i)
#pragma unroll
                    for (int h = 0; h < 2; ++h)
#pragma unroll
                        for (int c = 0; c < 4; ++c) acc[m][i][h][c] = 0.f;
            const int lrow = (lane & 7) + ((lane >> 3) & 1) * 8, lcol = (lane >> 4) * 8;

            for (int kbi = 0; kbi < kbs; ++kbi) {
                __syncthreads();  // buffer kbi & 1 is complete; the other one's last readers are done
                const int buf = kbi & 1;
                const bool more = kbi + 1 < kbs;
#pragma unroll
                for (int ktl = 0; ktl < 8; ++ktl) {
                    if (ktl < RPW) {  // rotate this warp's row ktl of the next K block, then load the next one's inputs
                        const int rr = warp + NW * ktl;
                        bool b = false;
                        const uint2 packed = row_rotate(nx, lane, b);
                        if (more && rr < rows_used) {
                            const bool real = rows_sh[rr] >= 0;
                            *reinterpret_cast<uint2*>(&As[buf ^ 1][rr][lane * 4]) = real ? packed : make_uint2(0u, 0u);
                            bad = bad || (real && b);
                        }
                        nx = ktl + 1 < RPW ? load_next(kbi, ktl + 1) : load_next(kbi + 1, 0);
                    }
                    const int kt = kt0 + kbi * 8 + ktl, slot = ktl % PF;
                    uint32_t w[NT];
#pragma unroll
                    for (int i = 0; i < NT; ++i) w[i] = wr[slot][i];
                    const int ktn = kt + PF < kt_last ? kt + PF : kt_last;
#pragma unroll
                    for (int i = 0; i < NT; ++i) wr[slot][i] = ldg_w(Tw + (size_t(ktn) * 32 + i) * 32, hint, pol);
                    uint32_t af[MT][4];
#pragma unroll
                    for (int m = 0; m < MT; ++m) ldsm_a(af[m], &As[buf][m * 16 + lrow][ktl * 16 + lcol]);
#pragma unroll
                    for (int i = 0; i < NT; ++i) {
                        uint32_t b0[2], b1[2];
                        decode_tile(w[i], lane, b0, b1);
#pragma unroll
                        for (int m = 0; m < MT; ++m)
                            if (m < mt_used) {
                                mma16816(acc[m][i][0], af[m], b0);
                                mma16816(acc[m][i][1], af[m], b1);
                            }
                    }
                }
            }

            float* zbase = a.z + (size_t(mat) * a.sk + split) * size_t(a.P) * WID;
#pragma unroll
            for (int m = 0; m < MT; ++m) {
                if (m >= mt_used) break;
                const int r0 = m * 16 + g8, r1 = r0 + 8;
#pragma unroll
                for (int i = 0; i < NT; ++i)
#pragma unroll
                    for (int h = 0; h < 2; ++h) {
                        const int col = warp * NT * 16 + i * 16 + h * 8 + 2 * t4;
                        if (r0 < G.count)
                            *reinterpret_cast<float2*>(zbase + size_t(G.start + r0) * WID + col) =
                                make_float2(acc[m][i][h][0], acc[m][i][h][1]);
                        if (r1 < G.count)
                            *reinterpret_cast<float2*>(zbase + size_t(G.start + r1) * WID + col) =
                                make_float2(acc[m][i][h][2], acc[m][i][h][3]);
                    }
            }
        }
        if (bad) atomicOr(a.fault, F_ROW);
        if (a.fused) gateup_tail<NW + NP>(a, G, g, ex, &last_sh);
        __syncthreads();  // the next item reuses the shared buffers and item_sh
    }
}

// ---- epilogue kernel (unfused): one warp per (pair, 128-block of the rank width) --------------------------------
__global__ void __launch_bounds__(128) gateup_epilogue(const Args a) {
    const int item = blockIdx.x * 4 + (threadIdx.x >> 5), lane = threadIdx.x & 31;
    const int gp = item >> 2, blk = item & 3;
    if (gp >= a.P || a.counts[0] <= 0) return;  // no plan (invalid routes): ids may be out of range
    const uint8_t* ex = a.image + size_t(a.ids[a.pair_route[gp]]) * EXPERT_BYTES;
    const bool bad = epilogue_item(a, ex, gp, blk, lane);
    if (__any_sync(0xffffffffu, bad) && lane == 0) atomicOr(a.fault, F_GATEUP);
}

// ---- down, split kernel: Zd[split][pair][4096] = Xd[pair] @ W_q(down) over this split's K range ---------------
// Block (group, 512-column chunk of the output, split); warp w owns the chunk's columns [128 w, 128 w + 128). A
// group's pairs are contiguous in Xd; their K range is staged in shared memory with cp.async when the block starts
// (rows past the group zeroed). The trellis words are loaded PF k tiles ahead, evict-first if HINT, as in the
// gate/up kernel. Blocks are (group, chunk, split) with the split slowest, or with `order` 2 (grid (SKD, groups,
// chunks)) chunk by chunk with the splits fastest, so a (row, chunk)'s partial sums are reduced (fused) soon after
// they are written. Four blocks a multiprocessor at 16 rows (the self-planning path would otherwise take more
// registers on some targets).
template <int MT, int PF, bool HINT>
__global__ void __launch_bounds__(128, MT == 1 ? 4 : 1) down_kernel(const Args a) {
    constexpr int R = MT * 16;
    const bool o2 = a.order == 2;
    const int g = o2 ? blockIdx.y : blockIdx.x, chunk = o2 ? blockIdx.z : blockIdx.y;
    const int split = o2 ? blockIdx.x : blockIdx.z;
    __shared__ __align__(16) half As[R][WID + 8];  // 1,040-byte rows: conflict-free fragment loads
    __shared__ int list_sh[R], nl_sh, rows_sh[R];
    __shared__ PlanSmem plan_sh;
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, g8 = lane >> 2, t4 = lane & 3;
    // The group and this thread's pair row (for the fused reduce): from the plan the gate/up kernel wrote (the group
    // loaded with the group count; an entry past the count is never used), or, with `pdl`, planned here from the
    // routes as the gate/up blocks plan (the same plan), before the gate/up kernel has ended.
    Group G;
    int my_row;
    if (a.pdl) {
        if (!self_plan(a, g, R, false, rows_sh, R, plan_sh)) return;
        G = plan_sh.G;
        my_row = tid < R ? rows_sh[tid] : -1;
    } else {
        const int ngroups = a.counts[0];
        G = a.groups[g];
        if (g >= ngroups) return;
        my_row = tid < G.count ? a.pair_row[G.start + tid] : -1;
    }
    const uint32_t* T = reinterpret_cast<const uint32_t*>(a.image + size_t(G.expert) * EXPERT_BYTES + OFF_DOWN_T);
    const int kts = 32 / a.skd, kt0 = split * kts;
    // The svh values of this expert and chunk (1 KB), which only the fused reduce reads: into L2 now.
    if (a.fused && split == 0 && tid < 8)
        prefetch_l2(a.image + size_t(G.expert) * EXPERT_BYTES + OFF_SVH_D + chunk * 1024 + tid * 128);

    // Down tiles are [32 k tiles][256 n tiles]; this warp's 8 n tiles start at chunk * 32 + warp * 8. The ring: tile
    // kt's words in wr[(kt - kt0) % PF] (a split can have fewer k tiles than PF).
    const uint64_t pol = HINT ? evict_first_policy() : 0;
    const uint32_t* Tw = T + size_t(chunk * 32 + warp * 8) * 32 + lane;
    uint32_t wr[PF][8];
#pragma unroll
    for (int p = 0; p < PF; ++p)
        if (p < kts)
#pragma unroll
            for (int i = 0; i < 8; ++i) wr[p][i] = ldg_w(Tw + (size_t(kt0 + p) * 256 + i) * 32, HINT, pol);

    if (a.pdl) grid_dependency_wait();  // the gate/up kernel has ended: its down input is complete
    // The group's rows of Xd over this split's K range, 16-byte copies.
    {
        const half* xg = a.xd + size_t(G.start) * WID + kt0 * 16;
        const int cpr = kts * 2;
        for (int i = tid; i < R * cpr; i += 128) {
            const int r = i / cpr, c = i % cpr;
            half* dst = &As[r][c * 8];
            if (r < G.count) cp_async16(dst, xg + size_t(r) * WID + c * 8);
            else *reinterpret_cast<uint4*>(dst) = make_uint4(0u, 0u, 0u, 0u);
        }
        cp_async_commit();
        cp_async_wait<0>();
        __syncthreads();
    }

    float acc[MT][8][2][4];
#pragma unroll
    for (int m = 0; m < MT; ++m)
#pragma unroll
        for (int i = 0; i < 8; ++i)
#pragma unroll
            for (int h = 0; h < 2; ++h)
#pragma unroll
                for (int c = 0; c < 4; ++c) acc[m][i][h][c] = 0.f;

    for (int kl = 0; kl < kts; kl += PF) {
#pragma unroll
        for (int p = 0; p < PF; ++p) {
            if (p == 0 || kl + p < kts) {
                const int kt = kt0 + kl + p;
                uint32_t af[MT][4];
#pragma unroll
                for (int m = 0; m < MT; ++m) {
                    const half* r0 = &As[m * 16 + g8][(kl + p) * 16 + 2 * t4];
                    const half* r1 = &As[m * 16 + g8 + 8][(kl + p) * 16 + 2 * t4];
                    af[m][0] = ld32(r0);
                    af[m][1] = ld32(r1);
                    af[m][2] = ld32(r0 + 8);
                    af[m][3] = ld32(r1 + 8);
                }
                // Tile kt + PF, or the split's last tile again past its end (the loop has no branch).
                const int ktn = kl + p + PF < kts ? kt + PF : kt0 + kts - 1;
#pragma unroll
                for (int i = 0; i < 8; ++i) {
                    uint32_t b0[2], b1[2];
                    decode_tile(wr[p][i], lane, b0, b1);
                    wr[p][i] = ldg_w(Tw + (size_t(ktn) * 256 + i) * 32, HINT, pol);
#pragma unroll
                    for (int m = 0; m < MT; ++m) {
                        mma16816(acc[m][i][0], af[m], b0);
                        mma16816(acc[m][i][1], af[m], b1);
                    }
                }
            }
        }
    }
    float* zbase = a.zd + size_t(split) * size_t(a.P) * H;
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
    if (a.fused) down_tail<4>(a, chunk, 4, my_row, list_sh, &nl_sh);
}

// ---- down, large-M kernel -------------------------------------------------------------------------------------
// Persistent blocks of 8 warps take (group, chunk of 128 NT output columns, split) items in grid order: the
// group's rows (up to 16 MT), warp w owning NT n tiles of the chunk, so each decoded tile feeds 2 MT MMAs. The
// group's down input is staged in shared memory with cp.async, 128 K columns at a time, double-buffered; the A
// fragments come from ldmatrix and the MMAs of m tiles past the group's rows are predicated off (the loop body
// has no branch); the trellis words are loaded PF k tiles ahead. Items run chunk by chunk (the group index
// varies fastest), so a chunk's partial sums are written and reduced (fused) while still in L2.
template <int MT, int NT>
__global__ void __launch_bounds__(256) down_big(const Args a) {
    constexpr int NW = 8, R = MT * 16, CW = NW * NT * 16, KC = 8, PF = 4;
    __shared__ __align__(16) half As[2][R][136];
    __shared__ int list_sh[R], nl_sh, item_sh;
    // Persistent blocks take (group, chunk, split) items in grid order, the group varying fastest.
    const int ngroups = a.counts[0], items = ngroups * a.nch * a.skd;
    for (;;) {
        if (threadIdx.x == 0) item_sh = atomicAdd(a.work + 1, 1);
        __syncthreads();
        const int item = item_sh;
        if (item >= items) break;
        const int g = item % ngroups, chunk = (item / ngroups) % a.nch, split = item / (ngroups * a.nch);
        const Group G = a.groups[g];
        const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, g8 = lane >> 2, t4 = lane & 3;
        const int my_row = tid < G.count ? a.pair_row[G.start + tid] : -1;  // for the fused reduce
        const int mt_used = (G.count + 15) >> 4;
        const int kts = 32 / a.skd, kt0 = split * kts, nkc = kts / KC, kt_last = kt0 + kts - 1;
        const half* xg = a.xd + size_t(G.start) * WID;

        // Stage K columns [16 (kt0 + KC c), + 16 KC) of the group's rows into buffer b: 16-byte copies; rows past the
        // group are zeroed (their products are never stored).
        auto stage = [&](int c, int b) {
            const int k0 = (kt0 + c * KC) * 16;
            for (int i = tid; i < mt_used * 16 * 2 * KC; i += NW * 32) {
                const int r = i / (2 * KC), p = i % (2 * KC);
                half* dst = &As[b][r][p * 8];
                if (r < G.count) cp_async16(dst, xg + size_t(r) * WID + k0 + p * 8);
                else *reinterpret_cast<uint4*>(dst) = make_uint4(0u, 0u, 0u, 0u);
            }
            cp_async_commit();
        };
        stage(0, 0);

        const uint32_t* T = reinterpret_cast<const uint32_t*>(a.image + size_t(G.expert) * EXPERT_BYTES + OFF_DOWN_T);
        // Down tiles are [32 k tiles][256 n tiles]; this warp's NT n tiles start at chunk * CW / 16 + warp * NT.
        // A ring of PF k tiles of words in flight.
        const uint32_t* Tw = T + size_t(chunk * (CW / 16) + warp * NT) * 32 + lane;
        const bool hint = a.l2hint != 0;
        const uint64_t pol = hint ? evict_first_policy() : 0;
        uint32_t wr[PF][NT];
#pragma unroll
        for (int p = 0; p < PF; ++p)
#pragma unroll
            for (int i = 0; i < NT; ++i)
                wr[p][i] = ldg_w(Tw + (size_t(kt0 + p < kt_last ? kt0 + p : kt_last) * 256 + i) * 32, hint, pol);

        float acc[MT][NT][2][4];
#pragma unroll
        for (int m = 0; m < MT; ++m)
#pragma unroll
            for (int i = 0; i < NT; ++i)
#pragma unroll
                for (int h = 0; h < 2; ++h)
#pragma unroll
                    for (int c = 0; c < 4; ++c) acc[m][i][h][c] = 0.f;
        const int lrow = (lane & 7) + ((lane >> 3) & 1) * 8, lcol = (lane >> 4) * 8;

        for (int c = 0; c < nkc; ++c) {
            if (c + 1 < nkc) {
                stage(c + 1, (c + 1) & 1);
                cp_async_wait<1>();
            } else {
                cp_async_wait<0>();
            }
            __syncthreads();
            const int b = c & 1;
#pragma unroll
            for (int ktl = 0; ktl < KC; ++ktl) {
                const int kt = kt0 + c * KC + ktl, slot = ktl % PF;
                uint32_t w[NT];
#pragma unroll
                for (int i = 0; i < NT; ++i) w[i] = wr[slot][i];
                const int ktn = kt + PF < kt_last ? kt + PF : kt_last;
#pragma unroll
                for (int i = 0; i < NT; ++i) wr[slot][i] = ldg_w(Tw + (size_t(ktn) * 256 + i) * 32, hint, pol);
                uint32_t af[MT][4];
#pragma unroll
                for (int m = 0; m < MT; ++m) ldsm_a(af[m], &As[b][m * 16 + lrow][ktl * 16 + lcol]);
#pragma unroll
                for (int i = 0; i < NT; ++i) {
                    uint32_t b0[2], b1[2];
                    decode_tile(w[i], lane, b0, b1);
#pragma unroll
                    for (int m = 0; m < MT; ++m)
                        if (m < mt_used) {
                            mma16816(acc[m][i][0], af[m], b0);
                            mma16816(acc[m][i][1], af[m], b1);
                        }
                }
            }
            __syncthreads();  // buffer b is read; the next stage may overwrite it
        }
        float* zbase = a.zd + size_t(split) * size_t(a.P) * H;
#pragma unroll
        for (int m = 0; m < MT; ++m) {
            if (m >= mt_used) break;
            const int r0 = m * 16 + g8, r1 = r0 + 8;
#pragma unroll
            for (int i = 0; i < NT; ++i)
#pragma unroll
                for (int h = 0; h < 2; ++h) {
                    const int col = chunk * CW + warp * NT * 16 + i * 16 + h * 8 + 2 * t4;
                    if (r0 < G.count)
                        *reinterpret_cast<float2*>(zbase + size_t(G.start + r0) * H + col) =
                            make_float2(acc[m][i][h][0], acc[m][i][h][1]);
                    if (r1 < G.count)
                        *reinterpret_cast<float2*>(zbase + size_t(G.start + r1) * H + col) =
                            make_float2(acc[m][i][h][2], acc[m][i][h][3]);
                }
        }
        if (a.fused) down_tail<NW>(a, chunk, CW / 128, my_row, list_sh, &nl_sh);
        __syncthreads();  // the next item reuses the shared buffers and item_sh
    }
}

// ---- reduce kernel (unfused): one warp per (row, 128-block of the model width) ---------------------------------
__global__ void __launch_bounds__(128) reduce_kernel(const Args a) {
    const int item = blockIdx.x * 4 + (threadIdx.x >> 5), lane = threadIdx.x & 31;
    const int row = item >> 5, blk = item & 31;
    if (row >= a.rows || a.counts[0] <= 0) return;  // no plan (invalid routes)
    const bool bad = reduce_item(a, row, blk, lane);
    if (__any_sync(0xffffffffu, bad) && lane == 0) atomicOr(a.fault, F_OUT);
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

// Launch a persistent large-M kernel: as many blocks as fit on the device at once, at most one per item.
template <class K>
cudaError_t launch_persistent(K* fn, int threads, int items, int sms, cudaStream_t st, const Args& a) {
    int occ = 0;
    if (cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, fn, threads, 0) != cudaSuccess || occ < 1) occ = 1;
    int n = sms * occ;
    if (n > items) n = items;
    if (n < 1) n = 1;
    fn<<<n, threads, 0, st>>>(a);
    return cudaGetLastError();
}

// Launch the split kernels with the trellis prefetch depth `pf` (1, 2 or 4), 16-row tiles `mt` (1 or 2) and the
// evict-first policy `hint`, 128 threads a block; the down kernel with programmatic stream serialization when
// `pdl` (it may then start before the gate/up kernel ends, and waits for it in the kernel).
template <class K>
cudaError_t launch_one(K* fn, bool pdl, dim3 grid, cudaStream_t st, const Args& a) {
    if (!pdl) {
        fn<<<grid, 128, 0, st>>>(a);
        return cudaGetLastError();
    }
    cudaLaunchConfig_t lc = {};
    lc.gridDim = grid;
    lc.blockDim = dim3(128);
    lc.dynamicSmemBytes = 0;
    lc.stream = st;
    cudaLaunchAttribute at[1];
    at[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    at[0].val.programmaticStreamSerializationAllowed = 1;
    lc.attrs = at;
    lc.numAttrs = 1;
    return cudaLaunchKernelEx(&lc, fn, a);
}

template <int MT, bool HINT>
cudaError_t launch_split_mt(bool down, int pf, bool pdl, dim3 grid, cudaStream_t st, const Args& a) {
    if (down) {
        if (pf == 4) return launch_one(down_kernel<MT, 4, HINT>, pdl, grid, st, a);
        if (pf == 2) return launch_one(down_kernel<MT, 2, HINT>, pdl, grid, st, a);
        return launch_one(down_kernel<MT, 1, HINT>, pdl, grid, st, a);
    }
    if (pf == 4) return launch_one(gateup_kernel<MT, 4, HINT>, false, grid, st, a);
    if (pf == 2) return launch_one(gateup_kernel<MT, 2, HINT>, false, grid, st, a);
    return launch_one(gateup_kernel<MT, 1, HINT>, false, grid, st, a);
}

cudaError_t launch_split(bool down, int mt, int pf, bool hint, bool pdl, dim3 grid, cudaStream_t st, const Args& a) {
    if (mt == 1) {
        if (hint) return launch_split_mt<1, true>(down, pf, pdl, grid, st, a);
        return launch_split_mt<1, false>(down, pf, pdl, grid, st, a);
    }
    if (hint) return launch_split_mt<2, true>(down, pf, pdl, grid, st, a);
    return launch_split_mt<2, false>(down, pf, pdl, grid, st, a);
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
    // `ensure`, and the whole buffer zeroed (on `st`) whenever it is (re)allocated.
    cudaError_t ensure_zeroed(size_t bytes, cudaStream_t st) {
        if (bytes <= n) return cudaSuccess;
        const cudaError_t e = ensure(bytes);
        return e != cudaSuccess ? e : cudaMemsetAsync(p, 0, n, st);
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
    Dev xrows, iw, groups, meta, pair_row, pair_route, inverse, z, xd, zd, out, gcnt, rcnt;
    std::vector<uint8_t> hiw;
    int sms = 1;             // the device's multiprocessors (persistent grids)
    int cc = 0;              // its compute capability, major * 10 + minor (programmatic dependent launch from 90)
    bool discarded = false;  // the last call dropped its partial sums from L2 (g53r_debug_copy refuses)
};

// A kernel configuration; zero fields take the defaults for the row count.
// - mt: 16-row tiles per group: 1 or 2 with the split kernels, 2 or 4 with the large-M kernels;
// - sk, skd: gate/up and down K splits (dividing 32; the large-M down kernel takes 1, 2 or 4);
// - fp32_swiglu: 1 computes the SwiGLU in FP32 instead of with the reference model's BF16 roundings;
// - big: 1 the split kernels, 2 the large-M kernels;
// - nt: large-M down, 16-column tiles per warp (1, 2 or 4): blocks of 128 nt output columns;
// - gw: large-M gate/up, MMA warps per block (8 or 16): 512 / gw columns a warp;
// - gp: large-M gate/up, rotation warps per block (0: the MMA warps rotate; 2, with gw 8 and mt 2);
// - plan: 1 plan_kernel, 2 the split gate/up blocks plan the call themselves (up to 512 routes; plan_kernel
//   above);
// - l2: 1 default caching of the trellis words, 2 an L2 evict-first policy for them (both families);
// - fuse: 1 separate epilogue and reduce kernels, 2 fused into gate/up and down;
// - discard: 1 keep the partial sums, 2 drop them from L2 once consumed (fused only);
// - pf: split kernels, trellis words loaded 1, 2 or 4 k tiles ahead (1 with the large-M kernels, which have their
//   own depth);
// - ord: split kernels' block order, 1 the split slowest, 2 a group's gate/up blocks adjacent and the down blocks
//   chunk by chunk with the splits fastest (1 with the large-M kernels);
// - pdl: 2 launches the split down kernel as a programmatic dependent launch (sm_90 and later): its blocks plan
//   themselves and load their first trellis words while the gate/up kernel ends (with plan 2 and fuse 2; up to 512
//   routes, as the plan in the gate/up blocks; older devices run the same kernels in stream order).
// Only sk, skd and fp32_swiglu change a bit of the output; the rest are schedules of the same arithmetic.
struct g53r_cfg {
    int mt, sk, skd, fp32_swiglu, big, nt, fuse, discard, gw, gp, plan, l2, pf, ord, pdl;
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

// The default configuration for `rows` rows. Up to 64 rows (decode and verify windows) the split kernels share
// one configuration, so a row's bits do not depend on the window size. Above 64 rows the large-M kernels run
// with one K split each (the bits of the split kernels' (2, 1, 1), whatever the row count): 32-row groups up to
// 2,048 rows, 64-row groups above (measured on the development GPU; README "The kernel"). On GB10 the L2
// evict-first policy up to 2,048 rows and 512-column down chunks above were faster (README "Measured on GB10");
// neither changes a bit. Up to 64 rows a group's gate/up blocks run together and the down blocks chunk by chunk,
// and the consumed partial sums are dropped from L2 (ord 2, discard 2: never slower on the development GPU, 3-9%
// faster at 4-64 rows; the same bits).
// On GB10 (sm_121, the ranks' GPU) the evict-first split loads, a prefetch ring of 2 and programmatic dependent
// launch were fastest up to 64 rows on the target hardware (1.02-1.35x over the development default; README
// "Measured on GB10"); the same bits. On the development GPU l2 2 was slower at 1-2 rows, so other GPUs keep it.
void g53r_default_cfg(uint32_t rows, g53r_cfg* out) {
    if (rows <= 64) {
#if G53R_BAKED_ARCH == 121
        *out = g53r_cfg{1, 8, 2, 0, 1, 2, 2, 2, 8, 0, 2, 2, 2, 2, 2};
#else
        *out = g53r_cfg{1, 8, 2, 0, 1, 2, 2, 2, 8, 0, 2, 1, 1, 2, 1};
#endif
    } else if (rows <= 2048) *out = g53r_cfg{2, 1, 1, 0, 2, 2, 2, 2, 8, 0, 1, 2, 1, 1, 1};
    else *out = g53r_cfg{4, 1, 1, 0, 2, 4, 2, 2, 16, 0, 1, 1, 1, 1, 1};
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
    int dev = 0;
    if (cudaGetDevice(&dev) != cudaSuccess || cudaDeviceGetAttribute(&S->sms, cudaDevAttrMultiProcessorCount, dev) != cudaSuccess ||
        S->sms < 1)
        S->sms = 1;
    int major = 0, minor = 0;
    if (cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev) == cudaSuccess &&
        cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, dev) == cudaSuccess)
        S->cc = major * 10 + minor;
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

// The configuration a call of `rows` rows runs: the defaults with `cfg_in`'s nonzero fields over them. Returns
// 0 when it is valid.
int g53r_resolve_cfg(uint32_t rows, const g53r_cfg* cfg_in, g53r_cfg* out) {
    g53r_cfg c;
    g53r_default_cfg(rows, &c);
    if (cfg_in) {
        const g53r_cfg& i = *cfg_in;
        if (i.big) {
            // Another family takes that family's defaults for the fields not given.
            g53r_cfg d;
            g53r_default_cfg(i.big == 2 ? 4096u : 1u, &d);
            if (d.big == i.big) c = d;
            c.big = i.big;
        }
        if (i.mt) c.mt = i.mt;
        if (i.sk) c.sk = i.sk;
        if (i.skd) c.skd = i.skd;
        if (i.nt) c.nt = i.nt;
        if (i.fuse) c.fuse = i.fuse;
        if (i.discard) c.discard = i.discard;
        if (i.gw) c.gw = i.gw;
        if (i.gp) c.gp = i.gp < 0 ? 0 : i.gp;
        if (i.plan) c.plan = i.plan;
        if (i.l2) c.l2 = i.l2;
        if (i.pf) c.pf = i.pf;
        if (i.ord) c.ord = i.ord;
        if (i.pdl) c.pdl = i.pdl;
        c.fp32_swiglu = i.fp32_swiglu;
    }
    if (out) *out = c;
    const bool div = c.sk > 0 && 32 % c.sk == 0 && c.skd > 0 && 32 % c.skd == 0;
    const bool fam = (c.big == 1 && (c.mt == 1 || c.mt == 2)) ||
                     (c.big == 2 && (c.mt == 2 || c.mt == 4) && (c.nt == 1 || c.nt == 2 || c.nt == 4) && c.skd <= 4 &&
                      (c.gw == 8 || c.gw == 16) && (c.gp == 0 || (c.gp == 2 && c.gw == 8 && c.mt == 2)));
    const bool rest = (c.fuse == 1 || c.fuse == 2) && (c.discard == 1 || c.discard == 2) &&
                      (c.fp32_swiglu == 0 || c.fp32_swiglu == 1) && (c.plan == 1 || (c.plan == 2 && c.big == 1)) &&
                      (c.l2 == 1 || c.l2 == 2) && (c.pf == 1 || ((c.pf == 2 || c.pf == 4) && c.big == 1)) &&
                      (c.ord == 1 || (c.ord == 2 && c.big == 1)) &&
                      (c.pdl == 1 || (c.pdl == 2 && c.big == 1 && c.plan == 2 && c.fuse == 2));
    return div && fam && rest ? 0 : 1;
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
    if (g53r_resolve_cfg(rows, cfg_in, &cfg) != 0) {
        set_msg(err, errlen,
                "ffn: bad configuration (split kernels: mt 1 or 2; large-M kernels: mt 2 or 4, nt 1, 2 or 4, gw 8 or "
                "16, gp 0 (-1) or 2 with gw 8 and mt 2, skd at most 4; sk and skd divide 32; fuse and discard 1 or 2; plan 1, or "
                "2 with the split kernels; l2 1 or 2; pf 1, or 2 or 4 with the split kernels; ord 1, or 2 with the "
                "split kernels; pdl 1, or 2 with the split kernels, plan 2 and fuse 2)");
        return 1;
    }
    const bool big = cfg.big == 2, fused = cfg.fuse == 2, discard = fused && cfg.discard == 2;
    const int routes = int(rows) * TOPK, gr = 16 * cfg.mt;
    const int nch = big ? 32 / cfg.nt : 8;  // output chunks of the down kernel (128 nt or 512 columns)
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
    // The arrival counters are zero between calls: zeroed when allocated, cleared by their last arrival.
    G53R_CK(S->gcnt.ensure_zeroed(size_t(max_groups) * 4, S->st), "group counters alloc");
    G53R_CK(S->rcnt.ensure_zeroed(size_t(rows) * nch * 4, S->st), "row counters alloc");
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

    Args a;
    a.image = L->image;
    a.xrows = S->xrows.as<uint8_t>();
    a.groups = S->groups.as<Group>();
    a.counts = counts;
    a.pair_row = S->pair_row.as<int>();
    a.pair_route = S->pair_route.as<int>();
    a.inverse = S->inverse.as<int>();
    a.ids = ids_d;
    a.wts = w_d;
    a.z = S->z.as<float>();
    a.xd = S->xd.as<half>();
    a.zd = S->zd.as<float>();
    a.out = S->out.p;
    a.gcnt = S->gcnt.as<int>();
    a.rcnt = S->rcnt.as<int>();
    a.work = counts + 2;  // meta words 2 and 3, cleared with the fault word
    a.fault = fault;
    a.P = routes;
    a.rows = int(rows);
    a.sk = cfg.sk;
    a.skd = cfg.skd;
    a.bf16 = cfg.fp32_swiglu ? 0 : 1;
    a.f32 = f32 ? 1 : 0;
    a.fused = fused ? 1 : 0;
    a.discard = discard ? 1 : 0;
    a.nch = nch;
    a.selfplan = !big && cfg.plan == 2 && routes <= 512;
    a.l2hint = cfg.l2 == 2 ? 1 : 0;
    a.order = cfg.ord;
    a.pdl = a.selfplan && fused && cfg.pdl == 2 ? 1 : 0;  // the down blocks plan themselves (up to 512 routes)

    // Phase events: one is recorded only where a phase that ran a kernel ends; an empty phase (the plan inside the
    // gate/up blocks, the fused epilogue and reduce) ends at the event before it and times zero. Every record is a
    // command of its own on the stream (about a microsecond of GPU time between two kernels).
    // With `pdl` nothing may stand between the two products' kernels: the gate/up phase then times both.
    G53R_CK(cudaEventRecord(S->e0, S->st), "event 0");
    cudaEvent_t ev_p = S->e0, ev_a = a.pdl ? S->ec : S->ea, ev_b = ev_a, ev_1 = S->ec;
    if (!a.selfplan) {
        plan_kernel<<<1, 1024, 0, S->st>>>(ids_d, w_d, int(rows), gr, S->groups.as<Group>(), counts, S->pair_row.as<int>(),
                                           S->pair_route.as<int>(), S->inverse.as<int>(), fault);
        G53R_CK(cudaGetLastError(), "plan launch");
        G53R_CK(cudaEventRecord(S->ep, S->st), "event p");
        ev_p = S->ep;
    }
    const dim3 gu_grid = cfg.ord == 2 ? dim3(2 * cfg.sk, max_groups) : dim3(2, max_groups, cfg.sk);
    if (big) {
        const int items = 2 * max_groups * cfg.sk;
        if (cfg.gp == 0) {
            if (cfg.gw == 8) {
                if (cfg.mt == 2) e = launch_persistent(gateup_big<2, 8, 0>, 256, items, S->sms, S->st, a);
                else e = launch_persistent(gateup_big<4, 8, 0>, 256, items, S->sms, S->st, a);
            } else {
                if (cfg.mt == 2) e = launch_persistent(gateup_big<2, 16, 0>, 512, items, S->sms, S->st, a);
                else e = launch_persistent(gateup_big<4, 16, 0>, 512, items, S->sms, S->st, a);
            }
        } else {
            e = launch_persistent(gateup_big<2, 8, 2>, 320, items, S->sms, S->st, a);
        }
        if (e != cudaSuccess) {
            set_err(err, errlen, "gate/up launch", e);
            return 2;
        }
    } else {
        G53R_CK(launch_split(false, cfg.mt, cfg.pf, a.l2hint != 0, false, gu_grid, S->st, a), "gate/up launch");
    }
    G53R_CK(cudaGetLastError(), "gate/up launch");
    if (!a.pdl) G53R_CK(cudaEventRecord(S->ea, S->st), "event a");
    if (!fused) {
        gateup_epilogue<<<routes, 128, 0, S->st>>>(a);
        G53R_CK(cudaGetLastError(), "epilogue launch");
        G53R_CK(cudaEventRecord(S->eb, S->st), "event b");
        ev_b = S->eb;
    }
    const dim3 dn_grid = cfg.ord == 2 ? dim3(cfg.skd, max_groups, nch) : dim3(max_groups, nch, cfg.skd);
    if (big) {
        const int items = max_groups * nch * cfg.skd;
        if (cfg.mt == 2) {
            if (cfg.nt == 1) e = launch_persistent(down_big<2, 1>, 256, items, S->sms, S->st, a);
            else if (cfg.nt == 2) e = launch_persistent(down_big<2, 2>, 256, items, S->sms, S->st, a);
            else e = launch_persistent(down_big<2, 4>, 256, items, S->sms, S->st, a);
        } else {
            if (cfg.nt == 1) e = launch_persistent(down_big<4, 1>, 256, items, S->sms, S->st, a);
            else if (cfg.nt == 2) e = launch_persistent(down_big<4, 2>, 256, items, S->sms, S->st, a);
            else e = launch_persistent(down_big<4, 4>, 256, items, S->sms, S->st, a);
        }
        if (e != cudaSuccess) {
            set_err(err, errlen, "down launch", e);
            return 2;
        }
    } else {
        const bool pdl_launch = a.pdl && S->cc >= 90;  // older devices: stream order (the kernel's wait is empty)
        G53R_CK(launch_split(true, cfg.mt, cfg.pf, a.l2hint != 0, pdl_launch, dn_grid, S->st, a), "down launch");
    }
    G53R_CK(cudaGetLastError(), "down launch");
    G53R_CK(cudaEventRecord(S->ec, S->st), "event c");
    if (!fused) {
        reduce_kernel<<<rows * 8, 128, 0, S->st>>>(a);
        G53R_CK(cudaGetLastError(), "reduce launch");
        G53R_CK(cudaEventRecord(S->e1, S->st), "event 1");
        ev_1 = S->e1;
    }
    S->discarded = discard;
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
        cudaEventElapsedTime(&gpu, S->e0, ev_1);
        if (ev_p != S->e0) cudaEventElapsedTime(&ph[0], S->e0, ev_p);
        cudaEventElapsedTime(&ph[1], ev_p, ev_a);
        if (ev_b != ev_a) cudaEventElapsedTime(&ph[2], ev_a, ev_b);
        if (ev_b != S->ec) cudaEventElapsedTime(&ph[3], ev_b, S->ec);
        if (ev_1 != S->ec) cudaEventElapsedTime(&ph[4], S->ec, ev_1);
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
// BF16 [rows * 4,096], to `bf16_out` (host). `cfg` may be null (defaults). `ms` (9 floats, may be null):
// host staging and uploads, GPU time, download and checks, groups; then the GPU phases plan, gate/up,
// epilogue, down, reduce (zero when no kernel of their own ran: the plan in the gate/up blocks, the fused
// epilogue and reduce; with `pdl` the gate/up phase times both products and the down phase is zero). Returns 0
// on success.
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
// Refused when the last call dropped its partial sums from L2 (discard 2).
int g53r_debug_copy(const g53r_scratch* S, float* z, size_t z_count, int32_t* pair_route, size_t pr_count,
                    uint16_t* xd, size_t xd_count, float* zd, size_t zd_count, char* err, size_t errlen) {
    if (!S || z_count * 4 > S->z.n || pr_count * 4 > S->pair_route.n || xd_count * 2 > S->xd.n ||
        zd_count * 4 > S->zd.n) {
        set_msg(err, errlen, "debug copy: counts exceed the scratch");
        return 1;
    }
    if (S->discarded) {
        set_msg(err, errlen, "debug copy: the last call discarded its partial sums (run it with discard 1)");
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
