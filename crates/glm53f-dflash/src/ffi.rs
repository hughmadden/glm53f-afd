//! Raw bindings to `kernels/glm53f_dflash.h` (feature `cuda`). See the header for each contract;
//! [`crate::gpu`] is the checked caller.

use crate::cuda::{CudaError, RawStream};

unsafe extern "C" {
    pub fn g53d_rmsnorm(
        x: *const f32,
        ldx: i64,
        w: *const u16,
        rows: i32,
        n: i32,
        eps: f32,
        y_f32: *mut f32,
        ldy: i64,
        y_bf16: *mut u16,
        ldb: i64,
        stream: RawStream,
    ) -> CudaError;
    pub fn g53d_rope_table(
        pos: *const i64,
        rows: i32,
        inv_freq: *const f32,
        cs: *mut f32,
        stream: RawStream,
    ) -> CudaError;
    pub fn g53d_head_norm_rope(
        x: *mut f32,
        ldx: i64,
        rows: i32,
        heads: i32,
        w: *const u16,
        cs: *const f32,
        eps: f32,
        out_bf16: *mut u16,
        ldo: i64,
        stream: RawStream,
    ) -> CudaError;
    pub fn g53d_store_kv(
        k: *const f32,
        v: *const f32,
        ld: i64,
        rows: i32,
        kv_width: i32,
        req: *const i32,
        pos: *const i64,
        bases: *const u64,
        layer: i32,
        ring: i32,
        stream: RawStream,
    ) -> CudaError;
    pub fn g53d_dyn_conv(
        x: *const f32,
        ldx: i64,
        dyn_: *const f32,
        lddyn: i64,
        base: *const u16,
        rows: i32,
        n: i32,
        group_size: i32,
        taps: i32,
        block: i32,
        out_f32: *mut f32,
        ldo: i64,
        out_bf16: *mut u16,
        ldb: i64,
        resid: *mut f32,
        ldr: i64,
        stream: RawStream,
    ) -> CudaError;
    pub fn g53d_attention_partial_floats(nreq: i32, heads: i32, splits: i32) -> i64;
    pub fn g53d_attention(
        q: *const f32,
        ldq: i64,
        nreq: i32,
        heads: i32,
        kv_heads: i32,
        start: *const i64,
        lo: *const i64,
        bases: *const u64,
        layer: i32,
        ring: i32,
        window_left: i32,
        scale: f32,
        split_keys: i32,
        splits: i32,
        partial: *mut f32,
        out_bf16: *mut u16,
        out_f32: *mut f32,
        ldo: i64,
        stream: RawStream,
    ) -> CudaError;
    pub fn g53d_silu_mul(
        gu: *const f32,
        ldgu: i64,
        rows: i32,
        inter: i32,
        out: *mut u16,
        ldo: i64,
        stream: RawStream,
    ) -> CudaError;
    pub fn g53d_block_embed(
        anchor_rows: *const u16,
        mask_row: *const u16,
        nreq: i32,
        block: i32,
        n: i32,
        h: *mut f32,
        stream: RawStream,
    ) -> CudaError;
    pub fn g53d_gather_drafts(
        src: *const u16,
        lds: i64,
        nreq: i32,
        block: i32,
        n: i32,
        dst: *mut u16,
        stream: RawStream,
    ) -> CudaError;
    pub fn g53d_topk16_workspace_bytes(rows: i32) -> i64;
    pub fn g53d_topk16(
        logits: *const f32,
        ld: i64,
        rows: i32,
        limit: i32,
        workspace: *mut core::ffi::c_void,
        vals: *mut f32,
        ids: *mut i32,
        stream: RawStream,
    ) -> CudaError;
    pub fn g53d_select(
        hproj: *const f32,
        vals: *const f32,
        ids: *const i32,
        anchors: *const i32,
        pred: *const u16,
        succ: *const u16,
        rank: i32,
        nreq: i32,
        slots: i32,
        temperature: *const f32,
        uniforms: *const f32,
        tokens: *mut i32,
        index: *mut i32,
        scores: *mut f32,
        q: *mut f32,
        conf: *mut f32,
        stream: RawStream,
    ) -> CudaError;
}
