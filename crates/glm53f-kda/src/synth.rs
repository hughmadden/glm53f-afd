//! Deterministic synthetic layers and rows for tests and benchmarks.
//!
//! Magnitudes are chosen to exercise the whole range of every gate: decay multipliers from
//! `exp(-5)` to almost 1, beta from almost 0 to almost 1, and states of order one. Every value
//! that the model stores in bfloat16 is generated bfloat16-exact.

use crate::cpu::{LayerParams, Rows};
use crate::{bf16, channels, state_len, DK, DV, LOWER_BOUND, RMS_EPS, WINDOW};

/// SplitMix64.
#[derive(Clone, Debug)]
pub struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Self {
        Rng(seed ^ 0x6a09_e667_f3bc_c909)
    }

    pub fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }

    /// Uniform in [lo, hi).
    pub fn uniform(&mut self, lo: f32, hi: f32) -> f32 {
        let u = (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32;
        lo + (hi - lo) * u
    }

    /// Uniform in [lo, hi), rounded to bfloat16.
    pub fn uniform_bf16(&mut self, lo: f32, hi: f32) -> f32 {
        bf16::round(self.uniform(lo, hi))
    }

    pub fn fill(&mut self, n: usize, lo: f32, hi: f32) -> Vec<f32> {
        (0..n).map(|_| self.uniform(lo, hi)).collect()
    }

    pub fn fill_bf16(&mut self, n: usize, lo: f32, hi: f32) -> Vec<f32> {
        (0..n).map(|_| self.uniform_bf16(lo, hi)).collect()
    }
}

/// A layer's parameters: conv weights and the norm weight bfloat16, `A_log` and `dt_bias` f32,
/// as in the official checkpoint.
pub fn layer(heads: usize, seed: u64) -> LayerParams {
    let mut rng = Rng::new(seed.wrapping_mul(3) ^ 0x11);
    LayerParams {
        heads,
        conv_w: rng.fill_bf16(channels(heads) * crate::TAPS, -0.6, 0.6),
        a_log: rng.fill(heads, -1.5, 1.5),
        dt_bias: rng.fill(heads * DK, -1.0, 1.0),
        norm_w: rng.fill_bf16(DV, 0.25, 1.75),
        eps: RMS_EPS,
        lower: LOWER_BOUND,
    }
}

/// `rows` rows of projection outputs (all bfloat16-exact).
pub fn rows(heads: usize, rows: usize, seed: u64) -> Rows {
    let mut rng = Rng::new(seed.wrapping_mul(5) ^ 0x22);
    Rows {
        heads,
        rows,
        qkv: rng.fill_bf16(rows * channels(heads), -2.0, 2.0),
        a: rng.fill_bf16(rows * heads * DK, -4.0, 4.0),
        b: rng.fill_bf16(rows * heads, -4.0, 4.0),
        gate: rng.fill_bf16(rows * heads * DV, -4.0, 4.0),
    }
}

/// A conv window (`WINDOW` rows of q | k | v, bfloat16-exact).
pub fn conv_window(heads: usize, seed: u64) -> Vec<f32> {
    Rng::new(seed.wrapping_mul(7) ^ 0x33).fill_bf16(WINDOW * channels(heads), -2.0, 2.0)
}

/// A recurrent state with entries uniform in [-scale, scale).
pub fn state(heads: usize, seed: u64, scale: f32) -> Vec<f32> {
    Rng::new(seed.wrapping_mul(11) ^ 0x44).fill(state_len(heads), -scale, scale)
}
