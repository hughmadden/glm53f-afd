//! Support for the tests and the benchmark: a JSON reader, SHA-256, a safetensors reader,
//! the golden-fixture loader and a seeded random source. Standard library only.

pub mod goldens;
pub mod json;
pub mod safetensors;
pub mod sha256;

use crate::bf16;
use crate::fp8::{f32_to_e4m3, Fp8Matrix};

/// SplitMix64: a small, seeded random source for test data.
pub struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Self {
        Rng(seed)
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

    /// Standard normal (Box-Muller).
    pub fn normal(&mut self) -> f32 {
        let u1 = (self.uniform() as f64).max(1e-12);
        let u2 = self.uniform() as f64;
        ((-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()) as f32
    }

    /// `n` BF16 values drawn from `N(0, sigma^2)`.
    pub fn bf16_vec(&mut self, n: usize, sigma: f32) -> Vec<u16> {
        (0..n)
            .map(|_| bf16::from_f32(self.normal() * sigma))
            .collect()
    }

    /// `n` f32 values drawn from `N(0, sigma^2)`.
    pub fn f32_vec(&mut self, n: usize, sigma: f32) -> Vec<f32> {
        (0..n).map(|_| self.normal() * sigma).collect()
    }

    /// A random FP8 weight `[rows][cols]` whose block scales vary over two decades, as a
    /// quantized checkpoint's do: `N(0, 1)` values quantized with `block_amax / 448`.
    pub fn fp8_matrix(&mut self, rows: usize, cols: usize) -> Fp8Matrix {
        let (sr, sc) = (rows.div_ceil(128), cols.div_ceil(128));
        let mut data = vec![0u8; rows * cols];
        let mut scale = vec![0f32; sr * sc];
        let target: Vec<f32> = (0..sr * sc)
            .map(|_| 0.02 * 10f32.powf(self.uniform() * 2.0 - 1.0))
            .collect();
        let vals: Vec<f32> = (0..rows * cols).map(|_| self.normal()).collect();
        for br in 0..sr {
            for bc in 0..sc {
                let mut amax = 0f32;
                for r in br * 128..((br + 1) * 128).min(rows) {
                    for c in bc * 128..((bc + 1) * 128).min(cols) {
                        amax = amax.max(vals[r * cols + c].abs());
                    }
                }
                let t = target[br * sc + bc];
                let s = amax * t / 448.0;
                scale[br * sc + bc] = s;
                for r in br * 128..((br + 1) * 128).min(rows) {
                    for c in bc * 128..((bc + 1) * 128).min(cols) {
                        data[r * cols + c] = f32_to_e4m3(vals[r * cols + c] * t / s);
                    }
                }
            }
        }
        Fp8Matrix::new(rows, cols, data, scale)
    }
}

/// The largest absolute difference between two f32 slices, and where it is.
pub fn max_abs_diff(a: &[f32], b: &[f32]) -> (f32, usize) {
    assert_eq!(a.len(), b.len());
    let mut worst = (0f32, 0usize);
    for (i, (&x, &y)) in a.iter().zip(b).enumerate() {
        let d = (x - y).abs();
        if d > worst.0 || d.is_nan() {
            worst = (d, i);
        }
    }
    worst
}

/// How two BF16 slices differ: the count of unequal elements and the largest ULP distance.
pub fn bf16_mismatch(a: &[u16], b: &[u16]) -> (usize, u32) {
    assert_eq!(a.len(), b.len());
    let mut count = 0;
    let mut worst = 0;
    for (&x, &y) in a.iter().zip(b) {
        if x != y {
            count += 1;
            worst = worst.max(bf16::ulp_distance(x, y));
        }
    }
    (count, worst)
}

/// The repository root (two levels above this crate).
pub fn repo_root() -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}
