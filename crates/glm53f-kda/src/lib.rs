//! `glm53f-kda`: GLM-5.3-Flash's KDA (Kimi delta attention) layers for decode, verify and
//! prefill.
//!
//! GLM-5.3-Flash has 34 KDA layers (every layer except 3, 7, …, 43). Each has 64 heads with
//! 128-wide q, k and v; a causal depthwise convolution (4 taps, SiLU) over the concatenated
//! q | k | v projections; a per-channel forget gate `lower * sigmoid(exp(A_log) * (a + dt_bias))`
//! with `lower = -5`; `beta = sigmoid(b)`; L2-normalized q and k; the delta-rule recurrent
//! state (64 × 128 × 128 per layer, f32); and a gated RMSNorm before `o_proj`.
//!
//! What this crate holds:
//!
//! - [`cpu`]: an f32 reference of the recurrent (per-token) path. It follows the reference
//!   model's definition and the kernels' order of operations, so it doubles as an exact model
//!   of the kernels' state update.
//! - [`chunked`]: the chunked (WY) form of the same recurrence, as the reference writes it for
//!   prefill and as the prefill kernel evaluates it (the kernel's host model).
//! - `kernels/`: the CUDA kernels behind a C ABI (`kernels/glm53f_kda.h`). Ported from
//!   TensorFold: a fused per-layer chain over R rows, replays that rebuild the state after a
//!   kept prefix bit for bit, and the conv-window shift. New: the chunked prefill, which runs a
//!   prompt segment of any length from a committed state and agrees with the chain to f32
//!   rounding. With the `cuda` feature they are compiled by `build.rs` and exposed as [`ffi`]
//!   (raw) and [`kernel`] (checked wrappers over [`device`] buffers).
//! - [`goldens`]: a loader for the oracle's golden fixtures.
//!
//! See `README.md` for the numerics contract and the prefill's design, accuracy and
//! throughput, and `PROVENANCE.md` for where each unit comes from.

pub mod bf16;
pub mod chunked;
pub mod cpu;
pub mod goldens;
pub mod json;
pub mod sha256;
pub mod synth;

#[cfg(feature = "cuda")]
pub mod cuda;
#[cfg(feature = "cuda")]
pub mod device;
#[cfg(feature = "cuda")]
pub mod ffi;
#[cfg(feature = "cuda")]
pub mod kernel;

/// Key head dimension.
pub const DK: usize = 128;
/// Value head dimension.
pub const DV: usize = 128;
/// Taps of the short convolution.
pub const TAPS: usize = 4;
/// Rows of conv history a request carries between steps (`TAPS - 1`).
pub const WINDOW: usize = TAPS - 1;
/// KDA heads per layer in GLM-5.3-Flash.
pub const HEADS: usize = 64;
/// KDA layers in GLM-5.3-Flash.
pub const LAYERS: usize = 34;
/// Decoder layers in GLM-5.3-Flash (KDA and DSA).
pub const DECODER_LAYERS: usize = 45;
/// `gate_lower_bound` of GLM-5.3-Flash.
pub const LOWER_BOUND: f32 = -5.0;
/// `rms_norm_eps` of GLM-5.3-Flash (the gated RMSNorm's epsilon).
pub const RMS_EPS: f32 = 1e-5;
/// The epsilon inside the q/k L2 norms (fixed in the reference).
pub const L2_EPS: f32 = 1e-6;

/// Whether decoder layer `i` is a KDA layer (`linear_attn_config.kda_layers`): every layer
/// except 3, 7, 11, …, 43, which are DSA layers.
pub fn is_kda_layer(i: usize) -> bool {
    i < DECODER_LAYERS && i % 4 != 3
}

/// The conv channels of a layer with `heads` heads: q | k | v.
pub fn channels(heads: usize) -> usize {
    3 * heads * DK
}

/// Elements of one layer's recurrent state for `heads` heads.
pub fn state_len(heads: usize) -> usize {
    heads * DV * DK
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kda_layer_schedule_matches_the_config() {
        // linear_attn_config.kda_layers of zai-org/GLM-5.3-Flash.
        let expected = [
            0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, 16, 17, 18, 20, 21, 22, 24, 25, 26, 28, 29, 30,
            32, 33, 34, 36, 37, 38, 40, 41, 42, 44,
        ];
        let got: Vec<usize> = (0..DECODER_LAYERS).filter(|&i| is_kda_layer(i)).collect();
        assert_eq!(got, expected);
        assert_eq!(got.len(), LAYERS);
        // One request's state: 34 × 64 × 128 × 128 × 4 B = 136 MiB.
        assert_eq!(LAYERS * state_len(HEADS) * 4, 136 << 20);
        assert_eq!(channels(HEADS), 24_576);
    }
}
