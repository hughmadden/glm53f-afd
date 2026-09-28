//! C ABI of the CUDA kernels (`kernels/include/glm53f_dsa.h`). Declarations only;
//! [`crate::gpu`] wraps them for tests and benchmarks.

use core::ffi::c_void;

/// `cudaError_t` (0 = success).
pub type CudaError = i32;
/// `cudaStream_t`.
pub type CudaStream = *mut c_void;

/// One layer's paged cache (`glm53f_dsa_cache_t`).
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct DsaCache {
    pub base: *mut u8,
    pub page_stride: i64,
    pub page_tables: *const i32,
    pub max_pages: i32,
    pub n_pages: i32,
}

/// One request's rows in a window (`glm53f_dsa_window_t`).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct DsaWindow {
    pub first_row: i32,
    pub rows: i32,
    pub start: i32,
    pub accepted: i32,
}

unsafe extern "C" {
    pub fn glm53f_dsa_init() -> CudaError;

    pub fn glm53f_dsa_index_pool_write(
        k_raw: *const f32,
        gate: *const f32,
        ln_w: *const f32,
        ln_b: *const f32,
        ln_eps: f32,
        ape: *const f32,
        tails: *const u8,
        windows: *const DsaWindow,
        row_req: *const i32,
        rows: i32,
        cache: DsaCache,
        stream: CudaStream,
    ) -> CudaError;

    pub fn glm53f_dsa_index_tail_commit(
        k_raw: *const f32,
        gate: *const f32,
        ln_w: *const f32,
        ln_b: *const f32,
        ln_eps: f32,
        tails: *mut u8,
        windows: *const DsaWindow,
        n_req: i32,
        stream: CudaStream,
    ) -> CudaError;

    pub fn glm53f_dsa_index_workspace_bytes(rows: i32, chunks: i32) -> u64;

    pub fn glm53f_dsa_index_plan(rows: i32, max_pools: i32, sms: i32, chunk_pools: *mut i32, chunks: *mut i32);

    pub fn glm53f_dsa_index_select(
        q: *const f32,
        w: *const f32,
        score_scale: f32,
        row_pos: *const i32,
        row_req: *const i32,
        rows: i32,
        max_pools: i32,
        cache: DsaCache,
        chunk_pools: i32,
        chunks: i32,
        workspace: *mut c_void,
        workspace_bytes: u64,
        pools_out: *mut i32,
        tokens_out: *mut i32,
        counts_out: *mut i32,
        debug_scores: *mut f32,
        stream: CudaStream,
    ) -> CudaError;

    /// `glm53f_dsa_index_select` without the counter reset: the workspace was zero-filled
    /// before its first use and serves only calls with the same `rows` and `chunks`.
    pub fn glm53f_dsa_index_select_prepared(
        q: *const f32,
        w: *const f32,
        score_scale: f32,
        row_pos: *const i32,
        row_req: *const i32,
        rows: i32,
        max_pools: i32,
        cache: DsaCache,
        chunk_pools: i32,
        chunks: i32,
        workspace: *mut c_void,
        workspace_bytes: u64,
        pools_out: *mut i32,
        tokens_out: *mut i32,
        counts_out: *mut i32,
        debug_scores: *mut f32,
        stream: CudaStream,
    ) -> CudaError;

    /// The first selection implementation (same contract and outputs), kept for A/B comparison.
    pub fn glm53f_dsa_index_select_v1(
        q: *const f32,
        w: *const f32,
        score_scale: f32,
        row_pos: *const i32,
        row_req: *const i32,
        rows: i32,
        max_pools: i32,
        cache: DsaCache,
        chunk_pools: i32,
        chunks: i32,
        workspace: *mut c_void,
        workspace_bytes: u64,
        pools_out: *mut i32,
        tokens_out: *mut i32,
        counts_out: *mut i32,
        debug_scores: *mut f32,
        stream: CudaStream,
    ) -> CudaError;

    pub fn glm53f_dsa_mla_latent_write(
        latent: *const f32,
        norm_w: *const f32,
        eps: f32,
        row_pos: *const i32,
        row_req: *const i32,
        rows: i32,
        cache: DsaCache,
        stream: CudaStream,
    ) -> CudaError;

    pub fn glm53f_dsa_mla_absorb_q(
        q: *const f32,
        kv_b: *const u16,
        rows: i32,
        q_abs_bf16: *mut u16,
        q_abs_f32: *mut f32,
        stream: CudaStream,
    ) -> CudaError;

    pub fn glm53f_dsa_mla_workspace_bytes(rows: i32, splits: i32) -> u64;

    pub fn glm53f_dsa_mla_plan(rows: i32, max_tokens: i32, sms: i32, splits: *mut i32, head_groups: *mut i32);

    pub fn glm53f_dsa_mla_sparse_attn(
        q_abs: *const u16,
        tokens: *const i32,
        token_stride: i32,
        counts: *const i32,
        row_req: *const i32,
        rows: i32,
        scale: f32,
        cache: DsaCache,
        splits: i32,
        head_groups: i32,
        workspace: *mut c_void,
        workspace_bytes: u64,
        o_lat: *mut f32,
        lse: *mut f32,
        stream: CudaStream,
    ) -> CudaError;

    /// The first sparse-attention implementation (same contract), kept for A/B comparison.
    pub fn glm53f_dsa_mla_sparse_attn_v1(
        q_abs: *const u16,
        tokens: *const i32,
        token_stride: i32,
        counts: *const i32,
        row_req: *const i32,
        rows: i32,
        scale: f32,
        cache: DsaCache,
        splits: i32,
        head_groups: i32,
        workspace: *mut c_void,
        workspace_bytes: u64,
        o_lat: *mut f32,
        lse: *mut f32,
        stream: CudaStream,
    ) -> CudaError;

    pub fn glm53f_dsa_mla_unabsorb_v(o_lat: *const f32, kv_b: *const u16, rows: i32, o: *mut f32, stream: CudaStream)
        -> CudaError;

    pub fn glm53f_dsa_mla_absorb_q_bf16(
        q: *const u16,
        ldq: i64,
        rows: i32,
        kv_b: *const u16,
        q_abs_bf16: *mut u16,
        q_abs_f32: *mut f32,
        stream: CudaStream,
    ) -> CudaError;

    pub fn glm53f_dsa_mla_unabsorb_v_rows(
        o_lat: *const f32,
        kv_b: *const u16,
        rows: i32,
        o_bf16: *mut u16,
        ldo: i64,
        o_f32: *mut f32,
        stream: CudaStream,
    ) -> CudaError;
}

// CUDA runtime (only what the tests and benchmarks use).
unsafe extern "C" {
    pub fn cudaMalloc(ptr: *mut *mut c_void, size: usize) -> CudaError;
    pub fn cudaFree(ptr: *mut c_void) -> CudaError;
    pub fn cudaMemcpy(dst: *mut c_void, src: *const c_void, count: usize, kind: i32) -> CudaError;
    pub fn cudaMemset(ptr: *mut c_void, value: i32, count: usize) -> CudaError;
    pub fn cudaMemsetAsync(ptr: *mut c_void, value: i32, count: usize, stream: CudaStream) -> CudaError;
    pub fn cudaMemcpy2D(
        dst: *mut c_void,
        dpitch: usize,
        src: *const c_void,
        spitch: usize,
        width: usize,
        height: usize,
        kind: i32,
    ) -> CudaError;
    pub fn cudaDeviceSynchronize() -> CudaError;
    pub fn cudaGetLastError() -> CudaError;
    pub fn cudaGetErrorString(err: CudaError) -> *const core::ffi::c_char;
    pub fn cudaMemGetInfo(free: *mut usize, total: *mut usize) -> CudaError;
    pub fn cudaDeviceGetAttribute(value: *mut i32, attr: i32, device: i32) -> CudaError;
    pub fn cudaEventCreate(event: *mut *mut c_void) -> CudaError;
    pub fn cudaEventDestroy(event: *mut c_void) -> CudaError;
    pub fn cudaEventRecord(event: *mut c_void, stream: CudaStream) -> CudaError;
    pub fn cudaEventSynchronize(event: *mut c_void) -> CudaError;
    pub fn cudaEventElapsedTime(ms: *mut f32, start: *mut c_void, end: *mut c_void) -> CudaError;
    pub fn cudaStreamCreateWithFlags(stream: *mut CudaStream, flags: u32) -> CudaError;
    pub fn cudaStreamDestroy(stream: CudaStream) -> CudaError;
    pub fn cudaStreamSynchronize(stream: CudaStream) -> CudaError;
    pub fn cudaStreamBeginCapture(stream: CudaStream, mode: i32) -> CudaError;
    pub fn cudaStreamEndCapture(stream: CudaStream, graph: *mut *mut c_void) -> CudaError;
    pub fn cudaGraphInstantiate(exec: *mut *mut c_void, graph: *mut c_void, flags: u64) -> CudaError;
    pub fn cudaGraphLaunch(exec: *mut c_void, stream: CudaStream) -> CudaError;
    pub fn cudaGraphExecDestroy(exec: *mut c_void) -> CudaError;
    pub fn cudaGraphDestroy(graph: *mut c_void) -> CudaError;
}

pub const MEMCPY_H2D: i32 = 1;
pub const MEMCPY_D2H: i32 = 2;
/// `cudaStreamNonBlocking`: no implicit synchronization with the legacy default stream.
pub const STREAM_NON_BLOCKING: u32 = 1;
/// `cudaStreamCaptureModeThreadLocal`: a capture restricts only its own thread.
pub const CAPTURE_THREAD_LOCAL: i32 = 1;
/// `cudaDevAttrMultiProcessorCount`.
pub const ATTR_SM_COUNT: i32 = 16;
/// `cudaDevAttrComputeCapabilityMajor` / `Minor`.
pub const ATTR_CC_MAJOR: i32 = 75;
pub const ATTR_CC_MINOR: i32 = 76;
