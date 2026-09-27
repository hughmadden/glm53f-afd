//! A small deterministic generator for test and benchmark data (SplitMix64).

/// SplitMix64: fast, reproducible, good enough for test data.
#[derive(Clone, Debug)]
pub struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Self {
        Self(seed ^ 0x9E37_79B9_7F4A_7C15)
    }

    pub fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform in [0, 1).
    pub fn uniform(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 * (1.0 / (1u64 << 24) as f32)
    }

    /// Uniform integer in [0, n).
    pub fn below(&mut self, n: usize) -> usize {
        (self.next_u64() % n as u64) as usize
    }

    /// Standard normal (Box-Muller).
    pub fn normal(&mut self) -> f32 {
        let u1 = (self.next_u64() >> 11) as f64 * (1.0 / (1u64 << 53) as f64);
        let u2 = (self.next_u64() >> 11) as f64 * (1.0 / (1u64 << 53) as f64);
        let r = (-2.0 * (u1.max(1e-300)).ln()).sqrt();
        (r * (2.0 * std::f64::consts::PI * u2).cos()) as f32
    }

    /// `n` normals times `scale`.
    pub fn normals(&mut self, n: usize, scale: f32) -> Vec<f32> {
        (0..n).map(|_| self.normal() * scale).collect()
    }
}
