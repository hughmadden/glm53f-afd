//! Raw bindings to `kernels/glm53f_forward.h` (feature `cuda`). The safety contract of every
//! call is the header's: device pointers to buffers large enough for the declared geometry,
//! alive until the stream's work completes.

use core::ffi::c_void;

use crate::cuda::RawStream;

unsafe extern "C" {
    pub fn glm53f_fwd_gemv_bf16(
        x: *const u16,
        ldx: i64,
        x_gstride: i64,
        w: *const u16,
        ldw: i64,
        w_gstride: i64,
        groups: i32,
        rows: i32,
        n: i32,
        k: i32,
        ksplit: i32,
        partials: *mut f32,
        sync: *mut u32,
        out: *mut c_void,
        out_f32: i32,
        ldo: i64,
        o_gstride: i64,
        stream: RawStream,
    ) -> i32;

    pub fn glm53f_fwd_gemv_ksplit(groups: i32, n: i32, k: i32) -> i32;

    pub fn glm53f_fwd_embed_gather(
        table: *const u16,
        vocab: i64,
        ids: *const i32,
        rows: i32,
        hidden: i32,
        streams: *mut u16,
        rows_out: *mut u16,
        stream: RawStream,
    ) -> i32;

    pub fn glm53f_fwd_bf16_to_f32(
        input: *const u16,
        ldi: i64,
        out: *mut f32,
        ldo: i64,
        rows: i32,
        cols: i32,
        scale: f32,
        stream: RawStream,
    ) -> i32;

    pub fn glm53f_fwd_f32_to_bf16(
        input: *const f32,
        ldi: i64,
        out: *mut u16,
        ldo: i64,
        rows: i32,
        cols: i32,
        stream: RawStream,
    ) -> i32;

    pub fn glm53f_fwd_argmax(
        logits: *const f32,
        ld: i64,
        rows: i32,
        n_valid: i32,
        ids: *mut i32,
        vals: *mut f32,
        stream: RawStream,
    ) -> i32;

    pub fn glm53f_fwd_gather_rows(
        src: *const u8,
        src_stride: i64,
        idx: *const i32,
        dst: *mut u8,
        dst_stride: i64,
        rows: i32,
        bytes: i64,
        stream: RawStream,
    ) -> i32;

    pub fn glm53f_fwd_scatter_rows(
        src: *const u8,
        src_stride: i64,
        idx: *const i32,
        dst: *mut u8,
        dst_stride: i64,
        rows: i32,
        bytes: i64,
        stream: RawStream,
    ) -> i32;

    pub fn glm53f_fwd_stream_mean(
        streams: *const u16,
        rows: i32,
        hidden: i32,
        out: *mut u16,
        ldo: i64,
        stream: RawStream,
    ) -> i32;

    pub fn glm53f_fwd_moe_combine(
        y: *const u16,
        ids: *const i32,
        weights: *const f32,
        rows: i32,
        top_k: i32,
        hidden: i32,
        out: *mut u16,
        stream: RawStream,
    ) -> i32;
}
