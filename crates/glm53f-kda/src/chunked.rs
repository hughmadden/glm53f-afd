//! The chunked (WY / UT) form of the KDA recurrence, as the reference evaluates prefill.
//!
//! This follows the reference's pure-torch `chunk_kimi_delta_attention` step by step, in f32,
//! for one head. With `C` rows per chunk and `G_i` the cumulative log decay within the chunk
//! (`G_i = g_0 + … + g_i`, per key channel):
//!
//! 1. `A[i][j] = -beta_i * sum_d k_i[d] k_j[d] exp(G_i[d] - G_j[d])` for `j < i` (strictly
//!    lower triangular); forward substitution then gives `T = (I - A)^-1` (the UT transform
//!    of the WY representation);
//! 2. `U = T (beta * V)` and `W = T (beta * K * exp(G))` (per row, per channel);
//! 3. per chunk, from the state `S` entering it: `V' = U - W S`;
//!    `O = (Q * exp(G)) S + M V'` with `M[i][j] = sum_d q_i[d] k_j[d] exp(G_i[d] - G_j[d])`
//!    for `j <= i`; and `S <- exp(G_last) * S + (K * exp(G_last - G))^T V'`.
//!
//! In exact arithmetic this equals the per-row recurrence of [`crate::cpu`]: row by row,
//! `S_t = diag(exp(g_t)) S_{t-1} + beta_t k_t (v_t - k_t^T diag(exp(g_t)) S_{t-1})^T`. In f32
//! it differs only in rounding. Every exponent is a difference `G_i - G_j` with `j <= i` or a
//! prefix `G_i`, so no exponential exceeds 1: the form is safe for any decay, and with
//! `lower = -5` whole chunks of strongly decaying channels simply underflow to zero.
//!
//! This is the numerics reference for a chunked prefill kernel (see the crate README).

// Sums are written `acc = acc + x`, as in the reference and the kernels, for side-by-side reading.
#![allow(clippy::assign_op_pattern)]

use crate::{cpu, DK, DV};

/// One head's chunked KDA over `t` rows. `q` and `k` are already L2-normalized (`q` also
/// scaled by `1 / sqrt(DK)`), `[t][DK]`; `v` is `[t][DV]`; `g` is the per-row log decay
/// `[t][DK]` (not cumulative); `beta` is `[t]`. `state` is the reference's key-major `[DK][DV]`
/// state, updated in place. Returns the read-out `[t][DV]`, not rounded.
pub fn head(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    g: &[f32],
    beta: &[f32],
    state: &mut [f32],
    chunk: usize,
) -> Vec<f32> {
    let t = beta.len();
    assert!(chunk >= 1);
    assert_eq!(q.len(), t * DK);
    assert_eq!(k.len(), t * DK);
    assert_eq!(v.len(), t * DV);
    assert_eq!(g.len(), t * DK);
    assert_eq!(state.len(), DK * DV);
    let c = chunk;
    let mut out = vec![0.0f32; t * DV];
    let mut start = 0;
    while start < t {
        let n = c.min(t - start);
        // Rows past the end are the reference's zero padding (beta 0, g 0): they contribute
        // nothing, so the chunk is simply shorter.
        let at = |w: usize, i: usize| (start + i) * w;
        // Cumulative log decay within the chunk.
        let mut gc = vec![0.0f32; n * DK];
        for i in 0..n {
            for d in 0..DK {
                let prev = if i == 0 { 0.0 } else { gc[(i - 1) * DK + d] };
                gc[i * DK + d] = prev + g[at(DK, i) + d];
            }
        }
        // 1. A, strictly lower triangular, and the forward substitution.
        let mut a = vec![0.0f32; n * n];
        for i in 0..n {
            let bi = beta[start + i];
            for j in 0..i {
                let mut acc = 0.0f32;
                for d in 0..DK {
                    let kb = k[at(DK, i) + d] * bi;
                    acc = acc + kb * k[at(DK, j) + d] * (gc[i * DK + d] - gc[j * DK + d]).exp();
                }
                a[i * n + j] = -acc;
            }
        }
        for i in 1..n {
            let rowi: Vec<f32> = a[i * n..i * n + i].to_vec();
            for col in 0..i {
                let mut acc = 0.0f32;
                for m in 0..i {
                    acc = acc + rowi[m] * a[m * n + col];
                }
                a[i * n + col] = rowi[col] + acc;
            }
        }
        for i in 0..n {
            a[i * n + i] = a[i * n + i] + 1.0;
        }
        // 2. U = T (beta V), W = T (beta K exp(G)).
        let mut u = vec![0.0f32; n * DV];
        let mut w = vec![0.0f32; n * DK];
        for i in 0..n {
            for vv in 0..DV {
                let mut acc = 0.0f32;
                for j in 0..n {
                    acc = acc + a[i * n + j] * (v[at(DV, j) + vv] * beta[start + j]);
                }
                u[i * DV + vv] = acc;
            }
            for d in 0..DK {
                let mut acc = 0.0f32;
                for j in 0..n {
                    acc = acc
                        + a[i * n + j]
                            * (k[at(DK, j) + d] * beta[start + j] * gc[j * DK + d].exp());
                }
                w[i * DK + d] = acc;
            }
        }
        // 3. The chunk against the incoming state.
        let mut vnew = vec![0.0f32; n * DV];
        for i in 0..n {
            for vv in 0..DV {
                let mut acc = 0.0f32;
                for d in 0..DK {
                    acc = acc + w[i * DK + d] * state[d * DV + vv];
                }
                vnew[i * DV + vv] = u[i * DV + vv] - acc;
            }
        }
        for i in 0..n {
            let mut m = vec![0.0f32; n];
            for (j, mj) in m.iter_mut().enumerate().take(i + 1) {
                let mut acc = 0.0f32;
                for d in 0..DK {
                    acc = acc
                        + q[at(DK, i) + d]
                            * k[at(DK, j) + d]
                            * (gc[i * DK + d] - gc[j * DK + d]).exp();
                }
                *mj = acc;
            }
            for vv in 0..DV {
                let mut inter = 0.0f32;
                for d in 0..DK {
                    inter = inter + q[at(DK, i) + d] * gc[i * DK + d].exp() * state[d * DV + vv];
                }
                let mut intra = 0.0f32;
                for j in 0..=i {
                    intra = intra + m[j] * vnew[j * DV + vv];
                }
                out[(start + i) * DV + vv] = inter + intra;
            }
        }
        let last = n - 1;
        for d in 0..DK {
            let decay_all = gc[last * DK + d].exp();
            for vv in 0..DV {
                let mut acc = 0.0f32;
                for i in 0..n {
                    acc = acc
                        + k[at(DK, i) + d]
                            * (gc[last * DK + d] - gc[i * DK + d]).exp()
                            * vnew[i * DV + vv];
                }
                state[d * DV + vv] = state[d * DV + vv] * decay_all + acc;
            }
        }
        start += n;
    }
    out
}

/// A whole layer's KDA core in the chunked form, from the conv outputs: `q`, `k`, `v`
/// `[t][H][128]` (before the L2 norms), `g` `[t][H][DK]` in log space, `beta` `[t][H]`, and the
/// state `[H][DK][DV]` (reference layout), updated in place. Returns the read-out
/// `[t][H][DV]`, not rounded.
#[allow(clippy::too_many_arguments)]
pub fn layer(
    heads: usize,
    q: &[f32],
    k: &[f32],
    v: &[f32],
    g: &[f32],
    beta: &[f32],
    state: &mut [f32],
    chunk: usize,
) -> Vec<f32> {
    let t = beta.len() / heads;
    let per_head = |x: &[f32], h: usize, w: usize| -> Vec<f32> {
        (0..t)
            .flat_map(|r| {
                x[(r * heads + h) * w..(r * heads + h + 1) * w]
                    .iter()
                    .copied()
            })
            .collect()
    };
    let mut out = vec![0.0f32; t * heads * DV];
    for h in 0..heads {
        let qh: Vec<f32> = (0..t)
            .flat_map(|r| {
                let n = cpu::literal::l2norm(&q[(r * heads + h) * DK..(r * heads + h + 1) * DK]);
                n.map(|x| x * cpu::q_scale())
            })
            .collect();
        let kh: Vec<f32> = (0..t)
            .flat_map(|r| cpu::literal::l2norm(&k[(r * heads + h) * DK..(r * heads + h + 1) * DK]))
            .collect();
        let vh = per_head(v, h, DV);
        let gh = per_head(g, h, DK);
        let bh: Vec<f32> = (0..t).map(|r| beta[r * heads + h]).collect();
        let o = head(
            &qh,
            &kh,
            &vh,
            &gh,
            &bh,
            &mut state[h * DK * DV..(h + 1) * DK * DV],
            chunk,
        );
        for r in 0..t {
            out[(r * heads + h) * DV..(r * heads + h + 1) * DV]
                .copy_from_slice(&o[r * DV..(r + 1) * DV]);
        }
    }
    out
}
