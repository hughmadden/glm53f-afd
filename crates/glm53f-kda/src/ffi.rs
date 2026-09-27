//! Raw bindings to `kernels/glm53f_kda.h` (feature `cuda`). The safety contract of every call
//! is the header's: the pointers are device pointers to buffers large enough for the declared
//! geometry, which stay alive until the stream's work completes. [`crate::kernel`] checks the
//! sizes on the host and is the safe way in.

use core::ffi::c_void;

/// `cudaError_t`.
pub type CudaError = i32;
/// `cudaStream_t`.
pub type Stream = *mut c_void;

/// Raw bfloat16 bits.
pub type Bf16 = u16;

unsafe extern "C" {
    pub fn glm53f_kda_chain(
        heads: i32,
        rows: i32,
        p: *const Bf16,
        p_stride: i64,
        b_off: i64,
        a: *const Bf16,
        a_stride: i64,
        g: *const Bf16,
        g_stride: i64,
        conv: *const Bf16,
        conv_w: *const Bf16,
        state_in: *const f32,
        state_out: *mut f32,
        a_log: *const f32,
        dt_bias: *const f32,
        norm_w: *const Bf16,
        eps: f32,
        lower: f32,
        out: *mut Bf16,
        out_stride: i64,
        k_save: *mut f32,
        v_save: *mut Bf16,
        g_save: *mut f32,
        b_save: *mut f32,
        stream: Stream,
    ) -> CudaError;

    pub fn glm53f_kda_chain_batch(
        heads: i32,
        batch: i32,
        cu_rows: *const i32,
        p: *const Bf16,
        p_stride: i64,
        b_off: i64,
        a: *const Bf16,
        a_stride: i64,
        g: *const Bf16,
        g_stride: i64,
        conv: *const Bf16,
        conv_off: *const i64,
        conv_w: *const Bf16,
        state_in: *const f32,
        state_out: *mut f32,
        state_off: *const i64,
        a_log: *const f32,
        dt_bias: *const f32,
        norm_w: *const Bf16,
        eps: f32,
        lower: f32,
        out: *mut Bf16,
        out_stride: i64,
        k_save: *mut f32,
        v_save: *mut Bf16,
        g_save: *mut f32,
        b_save: *mut f32,
        stream: Stream,
    ) -> CudaError;

    pub fn glm53f_kda_replay(
        heads: i32,
        rows: i32,
        state_in: *const f32,
        state_out: *mut f32,
        k_save: *const f32,
        v_save: *const Bf16,
        g_save: *const f32,
        b_save: *const f32,
        stream: Stream,
    ) -> CudaError;

    pub fn glm53f_kda_replay_layers(
        heads: i32,
        layers: i32,
        rows: i32,
        state_in: *const f32,
        state_out: *mut f32,
        state_stride: i64,
        k_save: *const f32,
        v_save: *const Bf16,
        g_save: *const f32,
        b_save: *const f32,
        kv_stride: i64,
        b_stride: i64,
        stream: Stream,
    ) -> CudaError;

    pub fn glm53f_kda_replay_batch(
        heads: i32,
        layers: i32,
        batch: i32,
        cu_rows: *const i32,
        keep: *const i32,
        state_in: *const f32,
        state_out: *mut f32,
        state_stride: i64,
        state_off: *const i64,
        k_save: *const f32,
        v_save: *const Bf16,
        g_save: *const f32,
        b_save: *const f32,
        kv_stride: i64,
        b_stride: i64,
        stream: Stream,
    ) -> CudaError;

    pub fn glm53f_kda_conv_shift(
        channels: i32,
        keep: i32,
        conv: *mut Bf16,
        p: *const Bf16,
        p_stride: i64,
        stream: Stream,
    ) -> CudaError;

    pub fn glm53f_kda_conv_shift_batch(
        channels: i32,
        layers: i32,
        batch: i32,
        cu_rows: *const i32,
        keep: *const i32,
        conv: *mut Bf16,
        conv_stride: i64,
        conv_off: *const i64,
        p: *const Bf16,
        p_layer_stride: i64,
        p_stride: i64,
        stream: Stream,
    ) -> CudaError;
}

/// The source kernels, compiled verbatim from `kernels/parity/tensorfold_kda.cu`, for the
/// parity tests. Same arguments as the source's launchers (32-bit strides).
pub mod parity {
    use super::{Bf16, CudaError, Stream};

    unsafe extern "C" {
        pub fn glm53f_kda_parity_chain(
            heads: i32,
            p: *const Bf16,
            p_stride: i32,
            b_off: i32,
            a: *const Bf16,
            a_stride: i32,
            g: *const Bf16,
            g_stride: i32,
            conv: *const Bf16,
            conv_w: *const Bf16,
            state_in: *const f32,
            a_log: *const f32,
            dt_bias: *const f32,
            norm_w: *const Bf16,
            eps: f32,
            lower: f32,
            rows: i32,
            out: *mut Bf16,
            state_out: *mut f32,
            k_save: *mut f32,
            v_save: *mut Bf16,
            g_save: *mut f32,
            b_save: *mut f32,
            stream: Stream,
        ) -> CudaError;

        pub fn glm53f_kda_parity_replay(
            heads: i32,
            state_in: *const f32,
            k_save: *const f32,
            v_save: *const Bf16,
            g_save: *const f32,
            b_save: *const f32,
            rows: i32,
            state_out: *mut f32,
            stream: Stream,
        ) -> CudaError;

        pub fn glm53f_kda_parity_replay_layers(
            heads: i32,
            state_in: *const f32,
            state_stride: i64,
            k_save: *const f32,
            v_save: *const Bf16,
            g_save: *const f32,
            b_save: *const f32,
            kv_stride: i64,
            b_stride: i64,
            layers: i32,
            rows: i32,
            state_out: *mut f32,
            stream: Stream,
        ) -> CudaError;
    }
}
