//! f32 building blocks of the CPU reference: a threaded GEMM against BF16 weights, RMSNorm,
//! RoPE and SiLU, each as the reference defines it (transformers' `Qwen3RMSNorm`,
//! `Qwen3RotaryEmbedding` + `apply_rotary_pos_emb`, `SiLU`).

use crate::bf16;

/// Threads for [`matmul`]: the machine's parallelism, at most 32.
fn threads() -> usize {
    std::thread::available_parallelism()
        .map_or(4, |n| n.get())
        .min(32)
}

/// `a . b` with sixteen partial sums (so the loop vectorizes), summed pairwise.
#[inline]
pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let mut acc = [0f32; 16];
    let n = a.len() / 16 * 16;
    for (ca, cb) in a[..n]
        .as_chunks::<16>()
        .0
        .iter()
        .zip(b[..n].as_chunks::<16>().0)
    {
        for l in 0..16 {
            acc[l] += ca[l] * cb[l];
        }
    }
    let mut tail = 0f32;
    for i in n..a.len() {
        tail += a[i] * b[i];
    }
    let mut w = 16;
    while w > 1 {
        w /= 2;
        for l in 0..w {
            acc[l] += acc[l + w];
        }
    }
    acc[0] + tail
}

/// `y [m][n] = x [m][k] . w^T`, `w [n][k]` in BF16 bits. Output columns are split over threads;
/// each weight row is widened once per block of 32 rows of `x`.
pub fn matmul(x: &[f32], m: usize, k: usize, w: &[u16], n: usize) -> Vec<f32> {
    assert_eq!(x.len(), m * k, "matmul: x is not [{m}][{k}]");
    assert_eq!(w.len(), n * k, "matmul: w is not [{n}][{k}]");
    let mut y = vec![0f32; m * n];
    if m == 0 || n == 0 {
        return y;
    }
    let t = threads().min(n);
    let per = n.div_ceil(t);
    let parts: Vec<(usize, Vec<f32>)> = std::thread::scope(|s| {
        let handles: Vec<_> = (0..t)
            .map(|ti| {
                let j0 = (ti * per).min(n);
                let j1 = ((ti + 1) * per).min(n);
                s.spawn(move || {
                    let nj = j1 - j0;
                    let mut out = vec![0f32; m * nj];
                    let mut wf = vec![0f32; k];
                    for i0 in (0..m).step_by(32) {
                        let i1 = (i0 + 32).min(m);
                        for j in j0..j1 {
                            for (d, &b) in wf.iter_mut().zip(&w[j * k..(j + 1) * k]) {
                                *d = bf16::to_f32(b);
                            }
                            for i in i0..i1 {
                                out[i * nj + (j - j0)] = dot(&x[i * k..(i + 1) * k], &wf);
                            }
                        }
                    }
                    (j0, out)
                })
            })
            .collect();
        handles
            .into_iter()
            .map(|h| h.join().expect("matmul thread"))
            .collect()
    });
    for (j0, out) in parts {
        let nj = out.len() / m;
        for i in 0..m {
            y[i * n + j0..i * n + j0 + nj].copy_from_slice(&out[i * nj..(i + 1) * nj]);
        }
    }
    y
}

/// `Qwen3RMSNorm` on every `w.len()`-wide row of `x`: `w * (x * rsqrt(mean(x^2) + eps))`.
pub fn rmsnorm(x: &[f32], w: &[u16], eps: f32) -> Vec<f32> {
    let n = w.len();
    assert_eq!(x.len() % n, 0);
    let mut y = vec![0f32; x.len()];
    for (row, out) in x.chunks_exact(n).zip(y.chunks_exact_mut(n)) {
        let ms = (row.iter().map(|&v| (v as f64) * (v as f64)).sum::<f64>() / n as f64) as f32;
        let r = 1.0 / (ms + eps).sqrt();
        for ((o, &v), &g) in out.iter_mut().zip(row).zip(w) {
            *o = bf16::to_f32(g) * (v * r);
        }
    }
    y
}

/// The default RoPE's inverse frequencies `1 / theta^(2i / dim)`, i < dim / 2, with the
/// reference's roundings: the exponent `(2i) / dim` in f32, the power in f64 rounded to f32, the
/// reciprocal in f32 (`1.0 / (base ** (arange(0, dim, 2).float() / dim))` in transformers'
/// `compute_default_rope_parameters`). The goldens record the reference's table; this reproduces
/// all 64 values bit for bit, where a correctly rounded `1 / theta^e` differs in 19 of them.
pub fn inv_freq(theta: f64, dim: usize) -> Vec<f32> {
    (0..dim / 2)
        .map(|i| {
            let e = (2 * i) as f32 / dim as f32;
            1.0f32 / (theta.powf(e as f64) as f32)
        })
        .collect()
}

/// RoPE (`rotate_half` convention) on one head of `x` in place at position `pos`:
/// `x[i] * cos - x[i + d/2] * sin`, `x[i + d/2] * cos + x[i] * sin`, with the angle
/// `pos * inv_freq[i]` formed in f32 as the reference does and its cosine and sine rounded to f32.
pub fn rope(x: &mut [f32], pos: usize, inv_freq: &[f32]) {
    let half = inv_freq.len();
    debug_assert_eq!(x.len(), 2 * half);
    for i in 0..half {
        let angle = (pos as f32) * inv_freq[i];
        let (s, c) = (angle as f64).sin_cos();
        let (s, c) = (s as f32, c as f32);
        let (a, b) = (x[i], x[i + half]);
        x[i] = a * c + (-b) * s;
        x[i + half] = b * c + a * s;
    }
}

/// SiLU `x * sigmoid(x) = x / (1 + exp(-x))`.
#[inline]
pub fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matmul_matches_naive() {
        let (m, k, n) = (37, 45, 70);
        let x: Vec<f32> = (0..m * k)
            .map(|i| ((i * 7 % 13) as f32 - 6.0) * 0.25)
            .collect();
        let w: Vec<u16> = (0..n * k)
            .map(|i| bf16::from_f32(((i * 5 % 11) as f32 - 5.0) * 0.5))
            .collect();
        let y = matmul(&x, m, k, &w, n);
        for i in 0..m {
            for j in 0..n {
                let e: f64 = (0..k)
                    .map(|t| x[i * k + t] as f64 * bf16::to_f32(w[j * k + t]) as f64)
                    .sum();
                // Small integers times quarters: every partial sum is exact.
                assert_eq!(y[i * n + j] as f64, e);
            }
        }
    }

    #[test]
    fn rope_rotates_pairs() {
        let f = inv_freq(10_000.0, 4);
        assert_eq!(f, vec![1.0, 0.01]);
        let mut x = vec![1.0, 0.0, 0.0, 0.0];
        rope(&mut x, 0, &f);
        assert_eq!(x, vec![1.0, 0.0, 0.0, 0.0]);
        let mut x = vec![1.0, 2.0, 0.0, 0.0];
        rope(&mut x, 3, &f);
        let (c0, s0) = ((3.0f64).cos() as f32, (3.0f64).sin() as f32);
        assert!((x[0] - c0).abs() < 1e-7 && (x[2] - s0).abs() < 1e-7);
        // A rotation keeps the norm of each pair.
        assert!(((x[1] * x[1] + x[3] * x[3]).sqrt() - 2.0).abs() < 1e-6);
    }

    #[test]
    fn rmsnorm_hand_values() {
        let x = [3.0, 4.0];
        let w = [bf16::from_f32(1.0), bf16::from_f32(2.0)];
        let y = rmsnorm(&x, &w, 0.0);
        let r = (12.5f32).sqrt();
        assert!((y[0] - 3.0 / r).abs() < 1e-6 && (y[1] - 8.0 / r).abs() < 1e-6);
    }
}
