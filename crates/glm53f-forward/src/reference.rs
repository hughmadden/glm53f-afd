//! Host models of this crate's kernels' arithmetic, for tests.

use glm53f_layers::bf16;
use glm53f_layers::math::warp_sum;

/// The GEMV's K splits for an `[n][k]` weight in `groups` groups (the same rule as
/// `glm53f_fwd_gemv_ksplit`).
pub fn gemv_ksplit(groups: usize, n: usize, k: usize) -> usize {
    let ctas = (n / 8) * groups;
    if ctas >= 512 {
        return 1;
    }
    let mut best = 1;
    let mut d = 2;
    while k / d >= 1024 {
        if k.is_multiple_of(256 * d) {
            best = d;
            if ctas * d >= 512 {
                break;
            }
        }
        d += 1;
    }
    best
}

/// `out[m][o] = x[m] . w[o]` in the GEMV's order, f32 (before the output rounding): lane `l`
/// accumulates `fma(x[k], w[k], acc)` over `k = lo + 8 l + 256 i + j` (i outer, j in 0..8), the
/// 32 lanes are summed by butterfly, and the splits are added in order.
#[allow(clippy::too_many_arguments)]
pub fn gemv_bf16(
    x: &[u16],
    ldx: usize,
    w: &[u16],
    ldw: usize,
    rows: usize,
    n: usize,
    k: usize,
    ksplit: usize,
) -> Vec<f32> {
    assert!(k.is_multiple_of(ksplit) && (k / ksplit).is_multiple_of(8));
    let kc = k / ksplit;
    let mut out = vec![0f32; rows * n];
    for m in 0..rows {
        let xr: Vec<f32> = x[m * ldx..m * ldx + k]
            .iter()
            .map(|&b| bf16::to_f32(b))
            .collect();
        for o in 0..n {
            let wr = &w[o * ldw..o * ldw + k];
            let mut total = 0f32;
            for s in 0..ksplit {
                let (lo, hi) = (s * kc, (s + 1) * kc);
                let mut lanes = [0f32; 32];
                for (l, acc) in lanes.iter_mut().enumerate() {
                    let mut k0 = lo + 8 * l;
                    while k0 < hi {
                        for j in 0..8 {
                            *acc = xr[k0 + j].mul_add(bf16::to_f32(wr[k0 + j]), *acc);
                        }
                        k0 += 256;
                    }
                }
                let part = warp_sum(&lanes);
                total = if s == 0 { part } else { total + part };
            }
            out[m * n + o] = total;
        }
    }
    out
}

/// The routed-expert combine of the reference's eager loop, as `glm53f_fwd_moe_combine`
/// computes it: per row, over the valid slots in ascending expert id,
/// `acc = bf16(acc + bf16(y * w))`.
pub fn moe_combine(
    y: &[u16],
    ids: &[i32],
    weights: &[f32],
    rows: usize,
    top_k: usize,
    hidden: usize,
) -> Vec<u16> {
    let mut out = vec![0u16; rows * hidden];
    for r in 0..rows {
        let mut order: Vec<usize> = (0..top_k).filter(|&j| ids[r * top_k + j] >= 0).collect();
        order.sort_by_key(|&j| (ids[r * top_k + j], j));
        for d in 0..hidden {
            let mut acc = 0f32;
            for &j in &order {
                let v = bf16::to_f32(y[(r * top_k + j) * hidden + d]);
                acc = bf16::round(acc + bf16::round(v * weights[r * top_k + j]));
            }
            out[r * hidden + d] = bf16::from_f32(acc);
        }
    }
    out
}

/// The mean of the four mHC streams (BF16 `[rows][4][hidden]`) as `glm53f_fwd_stream_mean`
/// computes it, the drafter's taps: per value `bf16((((s0 + s1) + s2) + s3) * 0.25)` in f32.
pub fn stream_mean(streams: &[u16], rows: usize, hidden: usize) -> Vec<u16> {
    assert_eq!(streams.len(), rows * 4 * hidden);
    let mut out = vec![0u16; rows * hidden];
    for r in 0..rows {
        let s = |j: usize, c: usize| bf16::to_f32(streams[(r * 4 + j) * hidden + c]);
        for c in 0..hidden {
            out[r * hidden + c] =
                bf16::from_f32((((s(0, c) + s(1, c)) + s(2, c)) + s(3, c)) * 0.25);
        }
    }
    out
}

/// Relative RMS difference `||a - b|| / ||b||` (f64 sums).
pub fn rel_rms(a: &[f32], b: &[f32]) -> f64 {
    assert_eq!(a.len(), b.len());
    let (mut num, mut den) = (0f64, 0f64);
    for (&x, &y) in a.iter().zip(b) {
        num += ((x - y) as f64).powi(2);
        den += (y as f64).powi(2);
    }
    (num / den.max(1e-300)).sqrt()
}

/// Largest absolute difference.
pub fn max_abs(a: &[f32], b: &[f32]) -> f64 {
    a.iter()
        .zip(b)
        .map(|(&x, &y)| (x as f64 - y as f64).abs())
        .fold(0.0, f64::max)
}
