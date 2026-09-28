//! Raw bindings to `kernels/glm53f_layers.h` (`cuda` feature). See the header for each entry
//! point's contract; [`crate::ops`] wraps them with shape checks.

use core::ffi::c_void;

/// `cudaError_t` (0 = success).
pub type CudaError = i32;
/// `cudaStream_t`.
pub type CudaStream = *mut c_void;

unsafe extern "C" {
    pub fn glm53f_hc_broadcast(
        embed: *const u16,
        streams: *mut u16,
        rows: i32,
        hidden: i32,
        stream: CudaStream,
    ) -> CudaError;
    #[allow(clippy::too_many_arguments)]
    pub fn glm53f_hc_project(
        streams_in: *const u16,
        block_out: *const u16,
        block_out2: *const u16,
        post: *const f32,
        comb: *const f32,
        streams_out: *mut u16,
        fn_: *const u16,
        partials: *mut f32,
        rows: i32,
        hidden: i32,
        stream: CudaStream,
    ) -> CudaError;
    #[allow(clippy::too_many_arguments)]
    pub fn glm53f_hc_finish(
        partials: *const f32,
        base: *const f32,
        scale: *const f32,
        streams: *const u16,
        norm_weight: *const u16,
        pre: *mut f32,
        post: *mut f32,
        comb: *mut f32,
        collapsed: *mut u16,
        normed: *mut u16,
        normed_q: *mut u8,
        normed_scales: *mut f32,
        rows: i32,
        hidden: i32,
        stream: CudaStream,
    ) -> CudaError;
    #[allow(clippy::too_many_arguments)]
    pub fn glm53f_hc_head(
        streams: *const u16,
        block_out: *const u16,
        block_out2: *const u16,
        post: *const f32,
        comb: *const f32,
        norm_weight: *const u16,
        out: *mut u16,
        rows: i32,
        hidden: i32,
        stream: CudaStream,
    ) -> CudaError;
    pub fn glm53f_router_logits(
        x: *const u16,
        weight: *const u16,
        logits: *mut f32,
        rows: i32,
        experts: i32,
        hidden: i32,
        stream: CudaStream,
    ) -> CudaError;
    #[allow(clippy::too_many_arguments)]
    pub fn glm53f_router_select(
        logits: *const f32,
        bias: *const f32,
        ids: *mut i32,
        weights: *mut f32,
        rows: i32,
        experts: i32,
        top_k: i32,
        scale: f32,
        stream: CudaStream,
    ) -> CudaError;
    pub fn glm53f_rmsnorm(
        x: *const u16,
        weight: *const u16,
        out: *mut u16,
        rows: i32,
        hidden: i32,
        stream: CudaStream,
    ) -> CudaError;
    pub fn glm53f_act_quant(
        x: *const u16,
        q: *mut u8,
        scales: *mut f32,
        rows: i32,
        cols: i32,
        stream: CudaStream,
    ) -> CudaError;
    pub fn glm53f_swiglu(
        gate_up: *const u16,
        act: *mut u16,
        q: *mut u8,
        scales: *mut f32,
        rows: i32,
        inter: i32,
        stream: CudaStream,
    ) -> CudaError;
    pub fn glm53f_selfcheck_division_free(mismatches: *mut u64, stream: CudaStream) -> CudaError;
    #[allow(clippy::too_many_arguments)]
    pub fn glm53f_fp8_gemm_decode(
        x: *const c_void,
        x_scales: *const f32,
        a8: i32,
        w: *const u8,
        w_scales: *const f32,
        rows: i32,
        n: i32,
        k: i32,
        ksplit: i32,
        partials: *mut f32,
        out: *mut u16,
        stream: CudaStream,
    ) -> CudaError;
    pub fn glm53f_splitk_reduce(
        partials: *const f32,
        out: *mut u16,
        ksplit: i32,
        rows: i32,
        n: i32,
        stream: CudaStream,
    ) -> CudaError;
    #[allow(clippy::too_many_arguments)]
    pub fn glm53f_fp8_gemm_prefill(
        xq: *const u8,
        x_scales: *const f32,
        w: *const u8,
        w_scales: *const f32,
        rows: i32,
        n: i32,
        k: i32,
        flags: i32,
        out: *mut u16,
        out_f32: *mut f32,
        stream: CudaStream,
    ) -> CudaError;
    pub fn glm53f_fp8_gemm_prefill_smem_bytes() -> i32;
    pub fn glm53f_fp8_quantize_weight(
        w: *const u16,
        n: i32,
        k: i32,
        q: *mut u8,
        scales: *mut f32,
        stream: CudaStream,
    ) -> CudaError;
    #[allow(clippy::too_many_arguments)]
    pub fn glm53f_fp8_dequant_bf16(
        w: *const u8,
        w_scales: *const f32,
        n: i32,
        k: i32,
        row0: i32,
        rows: i32,
        out: *mut u16,
        stream: CudaStream,
    ) -> CudaError;

    // Second revision: single-launch variants (see the header's `sync` convention).
    #[allow(clippy::too_many_arguments)]
    pub fn glm53f_hc_boundary_decode(
        streams_in: *const u16,
        block_out: *const u16,
        block_out2: *const u16,
        post_in: *const f32,
        comb_in: *const f32,
        streams_out: *mut u16,
        fn_: *const u16,
        base: *const f32,
        scale: *const f32,
        norm_weight: *const u16,
        partials: *mut f32,
        sync: *mut u32,
        pre: *mut f32,
        post: *mut f32,
        comb: *mut f32,
        collapsed: *mut u16,
        normed: *mut u16,
        normed_q: *mut u8,
        normed_scales: *mut f32,
        rows: i32,
        hidden: i32,
        stream: CudaStream,
    ) -> CudaError;
    pub fn glm53f_hc_comb(
        partials: *const f32,
        base: *const f32,
        scale: *const f32,
        comb: *mut f32,
        rows: i32,
        hidden: i32,
        stream: CudaStream,
    ) -> CudaError;
    #[allow(clippy::too_many_arguments)]
    pub fn glm53f_router_fused(
        x: *const u16,
        weight: *const u16,
        bias: *const f32,
        logits: *mut f32,
        sync: *mut u32,
        ids: *mut i32,
        weights: *mut f32,
        rows: i32,
        experts: i32,
        hidden: i32,
        top_k: i32,
        scale: f32,
        stream: CudaStream,
    ) -> CudaError;
    #[allow(clippy::too_many_arguments)]
    pub fn glm53f_fp8_gemm_decode_fused(
        x: *const c_void,
        x_scales: *const f32,
        a8: i32,
        w: *const u8,
        w_scales: *const f32,
        rows: i32,
        n: i32,
        k: i32,
        ksplit: i32,
        partials: *mut f32,
        sync: *mut u32,
        out: *mut u16,
        stream: CudaStream,
    ) -> CudaError;
}

/// `GLM53F_PREFILL_PROMOTE_K32`: add every k32 tensor-core product sum to the block sum in
/// f32, instead of accumulating the whole 128-block in the tensor core.
pub const PREFILL_PROMOTE_K32: i32 = 1;
