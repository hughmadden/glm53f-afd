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
//! # The prefill kernel's form
//!
//! [`prefill`] and [`prefill_head`] are a host model of the prefill kernel
//! (`glm53f_kda_prefill*`). The kernel uses the same WY/UT algebra, with the decays in a form
//! that stays at f32 accuracy against the per-row recurrence:
//!
//! - **Chunks of [`PREFILL_CHUNK`] = 16 rows.** The gate's lower bound (−5 per row) keeps
//!   every decay within a chunk inside `e^±75`, which f32 represents. So the pairwise decay
//!   `e^{G_i - G_j}` factors exactly through the chunk's first row:
//!   `Lf_i / Lf_j`, with `Lf_i = d_1 · … · d_i`.
//! - **Decays as products** of the chain's own per-row multipliers `d_t = exp(g_t)`, not
//!   exponentials of cumulative sums. A difference of two large cumulative sums loses about
//!   `ulp(G)`, some 3e-5 at `G ≈ −300`. A product of at most 16 multipliers loses a few f32
//!   ulps. Every factor is `≤ 1`, except `1 / Lf_j`, which is at most `e^75`.
//! - **Every product rounded once.** Products of multipliers just below 1 tend to round the
//!   same way, so a plain running product is biased, and the chunk's whole decay (`Lf_{n-1}`,
//!   `e^{G_last}`) reaches the state at every chunk: over a slowly decaying channel's memory
//!   the bias compounds. The running product therefore carries its rounding error exactly (a
//!   two-product with a fused multiply-add), and each `Lf_i` is rounded once from it. Without
//!   this, on slowly decaying channels the state's error against exact arithmetic was 2.5
//!   times the chain's; with it, it is about a quarter of the chain's (README.md, prefill).
//!
//! Per chunk of n rows:
//!
//! - `A_i = k_i ⊙ Lf_i`, `A'_i = q_i ⊙ Lf_i`, `B_j = k_j ⊘ Lf_j`;
//! - `L[i][j] = β_i A_i·B_j` for `j < i`, and `M[i][j] = A'_i·B_j` for `j ≤ i`;
//! - `T = (I + L)^-1`;
//! - `W = d_0 ⊙ T diag(β) A`, `U = T diag(β) V`, `Qg = d_0 ⊙ A'`, `Kg = Lf_{n-1} ⊙ B`, and
//!   `e^{G_last} = d_0 ⊙ Lf_{n-1}`.
//!
//! Then, from the state `S` entering the chunk:
//!
//! - `Δ = U − W S`;
//! - `Y = Qg S + M Δ`;
//! - `S ← e^{G_last} ⊙ S + Kgᵀ Δ`.

// Sums are written `acc = acc + x`, as in the reference and the kernels, for side-by-side reading.
#![allow(clippy::assign_op_pattern)]

use crate::cpu::{self, LayerParams, Rounding, Rows};
use crate::{state_len, DK, DV};

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

/// Rows per chunk of the prefill kernel.
pub const PREFILL_CHUNK: usize = 16;

/// One head's recurrence over `t` rows in the prefill kernel's chunked form (see the module
/// notes). `qn`, `kn`: the chain's normalized q (scaled) and k, `[t][DK]`; `v`: `[t][DV]`; `d`:
/// the chain's decay multipliers `exp(g)`, `[t][DK]`; `beta`: `[t]`. `s` is this head's state in
/// the kernels' layout `[DV][DK]`, updated in place. Returns the read-out `[t][DV]`, not rounded.
pub fn prefill_head(
    qn: &[f32],
    kn: &[f32],
    v: &[f32],
    d: &[f32],
    beta: &[f32],
    s: &mut [f32],
) -> Vec<f32> {
    let t = beta.len();
    assert_eq!(qn.len(), t * DK);
    assert_eq!(kn.len(), t * DK);
    assert_eq!(v.len(), t * DV);
    assert_eq!(d.len(), t * DK);
    assert_eq!(s.len(), DV * DK);
    let mut y = vec![0.0f32; t * DV];
    let mut start = 0;
    while start < t {
        let n = PREFILL_CHUNK.min(t - start);
        let row = |x: &[f32], i: usize| (start + i) * x.len() / t;
        // Decay products from the chunk's first row: Lf_0 = 1, Lf_i = d_1 ... d_i. The running
        // product `hi` carries its rounding error in `lo` (exact two-product), and each Lf_i is
        // rounded once, from hi + lo.
        let mut lf = vec![1.0f32; n * DK];
        let mut hi = vec![1.0f32; DK];
        let mut lo = vec![0.0f32; DK];
        for i in 1..n {
            for k in 0..DK {
                let dk = d[row(d, i) + k];
                let p = hi[k] * dk;
                lo[k] = lo[k].mul_add(dk, hi[k].mul_add(dk, -p));
                hi[k] = p;
                lf[i * DK + k] = hi[k] + lo[k];
            }
        }
        let d0: Vec<f32> = (0..DK).map(|k| d[row(d, 0) + k]).collect();
        let last: Vec<f32> = (0..DK).map(|k| lf[(n - 1) * DK + k]).collect();
        let elast: Vec<f32> = (0..DK)
            .map(|k| d0[k].mul_add(hi[k], d0[k] * lo[k]))
            .collect();
        let mut a = vec![0.0f32; n * DK];
        let mut aq = vec![0.0f32; n * DK];
        let mut b = vec![0.0f32; n * DK];
        for i in 0..n {
            for k in 0..DK {
                let f = lf[i * DK + k];
                a[i * DK + k] = kn[row(kn, i) + k] * f;
                aq[i * DK + k] = qn[row(qn, i) + k] * f;
                b[i * DK + k] = kn[row(kn, i) + k] / f;
            }
        }
        // L (strictly lower, with beta_i) and M (lower with the diagonal).
        let mut l = vec![0.0f32; n * n];
        let mut m = vec![0.0f32; n * n];
        for i in 0..n {
            for j in 0..=i {
                let (mut sl, mut sm) = (0.0f32, 0.0f32);
                for k in 0..DK {
                    sl = sl + a[i * DK + k] * b[j * DK + k];
                    sm = sm + aq[i * DK + k] * b[j * DK + k];
                }
                m[i * n + j] = sm;
                if j < i {
                    l[i * n + j] = beta[start + i] * sl;
                }
            }
        }
        // T = (I + L)^-1 by forward substitution; then T' = T diag(beta).
        let mut tm = vec![0.0f32; n * n];
        for i in 0..n {
            tm[i * n + i] = 1.0;
            for j in 0..i {
                let mut acc = l[i * n + j];
                for mm in j + 1..i {
                    acc = acc + l[i * n + mm] * tm[mm * n + j];
                }
                tm[i * n + j] = -acc;
            }
        }
        for i in 0..n {
            for j in 0..=i {
                tm[i * n + j] = tm[i * n + j] * beta[start + j];
            }
        }
        // W = d0 ⊙ T'A, U = T'V.
        let mut w = vec![0.0f32; n * DK];
        let mut u = vec![0.0f32; n * DV];
        for i in 0..n {
            for k in 0..DK {
                let mut acc = 0.0f32;
                for j in 0..=i {
                    acc = acc + tm[i * n + j] * a[j * DK + k];
                }
                w[i * DK + k] = d0[k] * acc;
            }
            for vv in 0..DV {
                let mut acc = 0.0f32;
                for j in 0..=i {
                    acc = acc + tm[i * n + j] * v[row(v, j) + vv];
                }
                u[i * DV + vv] = acc;
            }
        }
        // The chunk against the incoming state (s[v][k] = S[k][v]).
        let mut delta = vec![0.0f32; n * DV];
        for i in 0..n {
            for vv in 0..DV {
                let mut acc = 0.0f32;
                for k in 0..DK {
                    acc = acc + w[i * DK + k] * s[vv * DK + k];
                }
                delta[i * DV + vv] = u[i * DV + vv] - acc;
            }
        }
        for i in 0..n {
            for vv in 0..DV {
                let mut acc = 0.0f32;
                for k in 0..DK {
                    acc = acc + (aq[i * DK + k] * d0[k]) * s[vv * DK + k];
                }
                for j in 0..=i {
                    acc = acc + m[i * n + j] * delta[j * DV + vv];
                }
                y[(start + i) * DV + vv] = acc;
            }
        }
        for vv in 0..DV {
            for k in 0..DK {
                let mut acc = s[vv * DK + k] * elast[k];
                for i in 0..n {
                    acc = acc + (b[i * DK + k] * last[k]) * delta[i * DV + vv];
                }
                s[vv * DK + k] = acc;
            }
        }
        start += n;
    }
    y
}

/// A whole layer in the prefill kernel's form: the chain's per-row prologue (conv, norms,
/// decay, beta, with its bits), the chunked recurrence of [`prefill_head`], then the chain's
/// rounding and gated RMSNorm. Returns what [`cpu::chain`] returns, without replay inputs.
pub fn prefill(
    p: &LayerParams,
    conv: &[f32],
    state: &[f32],
    rows: &Rows,
    mode: Rounding,
) -> cpu::ChainOut {
    let (hn, rn) = (p.heads, rows.rows);
    assert_eq!(state.len(), state_len(hn));
    let mut st = state.to_vec();
    let mut out = vec![0.0f32; rn * hn * DV];
    let mut ys = vec![0.0f32; rn * hn * DV];
    for h in 0..hn {
        let decay_rate = p.a_log[h].exp();
        let (mut qn, mut kn, mut v, mut d) = (
            vec![0.0f32; rn * DK],
            vec![0.0f32; rn * DK],
            vec![0.0f32; rn * DV],
            vec![0.0f32; rn * DK],
        );
        let mut beta = vec![0.0f32; rn];
        for r in 0..rn {
            let (q, k, vr) = cpu::conv_qkv(p, conv, rows, r, h, mode);
            qn[r * DK..(r + 1) * DK].copy_from_slice(&cpu::l2norm(&q, Some(cpu::q_scale())));
            kn[r * DK..(r + 1) * DK].copy_from_slice(&cpu::l2norm(&k, None));
            v[r * DV..(r + 1) * DV].copy_from_slice(&vr);
            for i in 0..DK {
                d[r * DK + i] = cpu::decay(
                    rows.a[(r * hn + h) * DK + i],
                    p.dt_bias[h * DK + i],
                    decay_rate,
                    p.lower,
                );
            }
            beta[r] = mode.round(cpu::sigmoid(rows.b[r * hn + h]));
        }
        let y = prefill_head(
            &qn,
            &kn,
            &v,
            &d,
            &beta,
            &mut st[h * DV * DK..(h + 1) * DV * DK],
        );
        for r in 0..rn {
            let yr: Vec<f32> = y[r * DV..(r + 1) * DV]
                .iter()
                .map(|&x| mode.round(x))
                .collect();
            let gate = &rows.gate[(r * hn + h) * DV..(r * hn + h + 1) * DV];
            let o = cpu::gated_rmsnorm_as(&yr, &p.norm_w, gate, p.eps, mode);
            let at = (r * hn + h) * DV;
            out[at..at + DV].copy_from_slice(&o);
            ys[at..at + DV].copy_from_slice(&yr);
        }
    }
    cpu::ChainOut {
        out,
        y: ys,
        state: st,
        saves: cpu::Saves::zeros(hn, 0),
    }
}
