//! RMSNorm, with the reference's rounding points and the kernels' reduction order.
//!
//! The reference (`Glm5NextTextRMSNorm`) computes, for a BF16 input `x`:
//! `weight * (x_f32 * rsqrt(mean(x_f32^2) + eps)).to(bf16)`, i.e. two roundings to BF16:
//! the normalized value, then its product with the BF16 weight. The kernels keep both
//! roundings and fix the order of the sum of squares (see [`sum_squares`]).

use crate::bf16;
use crate::math::warp_sum;

/// `rms_norm_eps` of GLM-5.3-Flash.
pub const RMS_EPS: f32 = 1e-5;
/// Threads of the kernels' per-row reductions.
pub const ROW_THREADS: usize = 256;
/// Consecutive values each thread handles per step (one 16-byte BF16 load).
pub const ROW_CHUNK: usize = 8;

/// Sum of squares in the kernels' order: thread `t` of 256 accumulates chunks `t`,
/// `t + 256`, ... of 8 consecutive values with `fma(v, v, acc)`; each warp reduces by
/// butterfly; the 8 warp totals are added in warp order. `x.len()` must be a multiple of 8.
pub fn sum_squares(x: &[f32]) -> f32 {
    assert_eq!(x.len() % ROW_CHUNK, 0, "row length must be a multiple of 8");
    let chunks = x.len() / ROW_CHUNK;
    let mut thread = [0f32; ROW_THREADS];
    for (t, acc) in thread.iter_mut().enumerate() {
        let mut c = t;
        while c < chunks {
            for &v in &x[c * ROW_CHUNK..(c + 1) * ROW_CHUNK] {
                *acc = v.mul_add(v, *acc);
            }
            c += ROW_THREADS;
        }
    }
    let mut total = 0f32;
    for w in 0..ROW_THREADS / 32 {
        let lanes: [f32; 32] = thread[w * 32..(w + 1) * 32].try_into().unwrap();
        total += warp_sum(&lanes);
    }
    total
}

/// `1 / sqrt(sum_squares / n + eps)`, with IEEE division and square root.
pub fn rms_scale(sum_squares: f32, n: usize, eps: f32) -> f32 {
    1.0 / (sum_squares / n as f32 + eps).sqrt()
}

/// Weighted RMSNorm of one BF16 row: `bf16(w * bf16(x * r))`.
pub fn rms_norm_row(x: &[u16], weight: &[u16], eps: f32) -> Vec<u16> {
    assert_eq!(x.len(), weight.len());
    let xf = bf16::widen(x);
    let r = rms_scale(sum_squares(&xf), xf.len(), eps);
    apply_scale(&xf, weight, r)
}

/// `bf16(w * bf16(x * r))` for a row whose scale `r` is known.
pub fn apply_scale(x: &[f32], weight: &[u16], r: f32) -> Vec<u16> {
    x.iter()
        .zip(weight)
        .map(|(&v, &w)| bf16::from_f32(bf16::to_f32(w) * bf16::round(v * r)))
        .collect()
}

/// Weighted RMSNorm of `rows` BF16 rows of width `weight.len()`.
pub fn rms_norm(x: &[u16], weight: &[u16], rows: usize, eps: f32) -> Vec<u16> {
    let d = weight.len();
    assert_eq!(x.len(), rows * d);
    x.chunks_exact(d)
        .flat_map(|row| rms_norm_row(row, weight, eps))
        .collect()
}
