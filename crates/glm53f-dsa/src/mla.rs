//! Multi-head latent attention without RoPE (f32 reference), in the expanded
//! form of the `transformers` reference and in the absorbed form the engine runs.
//!
//! Query path: `q_resid = RMSNorm(q_a_proj x)`, `q = q_b_proj q_resid`
//! (heads x 256). Latent path: `c = RMSNorm(kv_a_proj_with_mqa x)` (512, no RoPE
//! part); `c` is what the cache holds. `kv_b_proj` (heads x (256 k + 256 v) x 512)
//! expands a latent into per-head keys and values.
//!
//! * Expanded (reference): `k_h = W^K_h c`, `v_h = W^V_h c`,
//!   `o_h = sum_j softmax_j(scale q_h . k_hj) v_hj`.
//! * Absorbed (engine): `q'_h = (W^K_h)^T q_h` (512 wide),
//!   `o'_h = sum_j softmax_j(scale q'_h . c_j) c_j`, `o_h = W^V_h o'_h`.
//!
//! The two are equal in exact arithmetic (associativity of the matrix
//! products); see [`score_error_bound`] for the f32 difference.

use crate::config::DsaConfig;
use crate::num::{dot, gamma, matvec, rms_norm, Rounding};
use crate::rng::Rng;

/// MLA weights, dequantized to f32 (`[out, in]` row-major, as in the checkpoint).
#[derive(Clone, Debug)]
pub struct MlaWeights {
    /// `q_a_proj`: `[q_lora_rank, hidden]` (FP8 block-128 in the checkpoint).
    pub q_a_proj: Vec<f32>,
    /// `q_a_layernorm.weight`: `[q_lora_rank]`.
    pub q_a_norm: Vec<f32>,
    /// `q_b_proj`: `[n_heads * qk_nope_head_dim, q_lora_rank]` (FP8).
    pub q_b_proj: Vec<f32>,
    /// `kv_a_proj_with_mqa`: `[kv_lora_rank, hidden]` (FP8).
    pub kv_a_proj: Vec<f32>,
    /// `kv_a_layernorm.weight`: `[kv_lora_rank]`.
    pub kv_a_norm: Vec<f32>,
    /// `kv_b_proj`: `[n_heads * (qk_nope_head_dim + v_head_dim), kv_lora_rank]` (BF16).
    pub kv_b_proj: Vec<f32>,
    /// `o_proj`: `[hidden, n_heads * v_head_dim]` (FP8).
    pub o_proj: Vec<f32>,
}

impl MlaWeights {
    /// Random weights with roughly unit-variance activations.
    pub fn random(cfg: &DsaConfig, rng: &mut Rng) -> Self {
        let (h, ql, kl, nh, dn, dv) =
            (cfg.hidden, cfg.q_lora_rank, cfg.kv_lora_rank, cfg.n_heads, cfg.qk_nope_head_dim, cfg.v_head_dim);
        Self {
            q_a_proj: rng.normals(ql * h, 1.0 / (h as f32).sqrt()),
            q_a_norm: (0..ql).map(|_| 1.0 + 0.2 * rng.normal()).collect(),
            q_b_proj: rng.normals(nh * dn * ql, 1.0 / (ql as f32).sqrt()),
            kv_a_proj: rng.normals(kl * h, 1.0 / (h as f32).sqrt()),
            kv_a_norm: (0..kl).map(|_| 1.0 + 0.2 * rng.normal()).collect(),
            kv_b_proj: rng.normals(nh * (dn + dv) * kl, 1.0 / (kl as f32).sqrt()),
            o_proj: rng.normals(h * nh * dv, 1.0 / ((nh * dv) as f32).sqrt()),
        }
    }

    pub fn check(&self, cfg: &DsaConfig) -> Result<(), String> {
        let (h, ql, kl, nh, dn, dv) =
            (cfg.hidden, cfg.q_lora_rank, cfg.kv_lora_rank, cfg.n_heads, cfg.qk_nope_head_dim, cfg.v_head_dim);
        let want = [
            ("q_a_proj", self.q_a_proj.len(), ql * h),
            ("q_a_layernorm", self.q_a_norm.len(), ql),
            ("q_b_proj", self.q_b_proj.len(), nh * dn * ql),
            ("kv_a_proj_with_mqa", self.kv_a_proj.len(), kl * h),
            ("kv_a_layernorm", self.kv_a_norm.len(), kl),
            ("kv_b_proj", self.kv_b_proj.len(), nh * (dn + dv) * kl),
            ("o_proj", self.o_proj.len(), h * nh * dv),
        ];
        for (n, got, exp) in want {
            if got != exp {
                return Err(format!("mla {n}: {got} values, expected {exp}"));
            }
        }
        Ok(())
    }

    /// Row `r` of head `h`'s key block of `kv_b_proj` (a `kv_lora_rank` vector).
    #[inline]
    pub fn wk_row(&self, cfg: &DsaConfig, h: usize, r: usize) -> &[f32] {
        let kl = cfg.kv_lora_rank;
        let base = (h * (cfg.qk_nope_head_dim + cfg.v_head_dim) + r) * kl;
        &self.kv_b_proj[base..base + kl]
    }

    /// Row `r` of head `h`'s value block of `kv_b_proj`.
    #[inline]
    pub fn wv_row(&self, cfg: &DsaConfig, h: usize, r: usize) -> &[f32] {
        let kl = cfg.kv_lora_rank;
        let base = (h * (cfg.qk_nope_head_dim + cfg.v_head_dim) + cfg.qk_nope_head_dim + r) * kl;
        &self.kv_b_proj[base..base + kl]
    }
}

/// Query path: returns `(q_resid, q)` with `q` laid out `[n_heads][qk_nope_head_dim]`.
pub fn project_q(cfg: &DsaConfig, w: &MlaWeights, hidden: &[f32], rounding: Rounding) -> (Vec<f32>, Vec<f32>) {
    let mut qa = matvec(&w.q_a_proj, cfg.q_lora_rank, cfg.hidden, hidden);
    rounding.apply(&mut qa);
    let q_resid = rms_norm(&qa, &w.q_a_norm, cfg.rms_norm_eps, rounding);
    let mut q = matvec(&w.q_b_proj, cfg.n_heads * cfg.qk_nope_head_dim, cfg.q_lora_rank, &q_resid);
    rounding.apply(&mut q);
    (q_resid, q)
}

/// Latent path: `RMSNorm(kv_a_proj_with_mqa x)` (the cached latent).
pub fn project_latent(cfg: &DsaConfig, w: &MlaWeights, hidden: &[f32], rounding: Rounding) -> Vec<f32> {
    let mut c = matvec(&w.kv_a_proj, cfg.kv_lora_rank, cfg.hidden, hidden);
    rounding.apply(&mut c);
    rms_norm(&c, &w.kv_a_norm, cfg.rms_norm_eps, rounding)
}

/// Expand one latent into keys `[n_heads][qk_nope]` and values `[n_heads][v]`.
pub fn expand_kv(cfg: &DsaConfig, w: &MlaWeights, latent: &[f32], rounding: Rounding) -> (Vec<f32>, Vec<f32>) {
    let (nh, dn, dv) = (cfg.n_heads, cfg.qk_nope_head_dim, cfg.v_head_dim);
    let mut k = Vec::with_capacity(nh * dn);
    let mut v = Vec::with_capacity(nh * dv);
    for h in 0..nh {
        for r in 0..dn {
            k.push(rounding.at(dot(w.wk_row(cfg, h, r), latent)));
        }
        for r in 0..dv {
            v.push(rounding.at(dot(w.wv_row(cfg, h, r), latent)));
        }
    }
    (k, v)
}

/// Output of one query row's attention.
#[derive(Clone, Debug)]
pub struct AttnRow {
    /// Per-head attention output `[n_heads][v_head_dim]` (before `o_proj`).
    pub out: Vec<f32>,
    /// Per-head log-sum-exp of the scaled scores (natural log).
    pub lse: Vec<f32>,
}

/// Softmax over `scores` in place (max-subtracted); returns `(max, sum)`.
fn softmax_inplace(scores: &mut [f32]) -> (f32, f32) {
    let m = scores.iter().fold(f32::NEG_INFINITY, |a, b| a.max(*b));
    let mut sum = 0.0f32;
    for s in scores.iter_mut() {
        *s = (*s - m).exp();
        sum += *s;
    }
    for s in scores.iter_mut() {
        *s /= sum;
    }
    (m, sum)
}

/// Expanded attention (the reference form) for one query row over the given
/// latents. `keys`/`values` are the expanded caches for those latents
/// (see [`expand_kv`]), in the order of `latents`.
pub fn attend_expanded(cfg: &DsaConfig, q: &[f32], keys: &[Vec<f32>], values: &[Vec<f32>], rounding: Rounding) -> AttnRow {
    let (nh, dn, dv) = (cfg.n_heads, cfg.qk_nope_head_dim, cfg.v_head_dim);
    let scale = cfg.attn_scale();
    let n = keys.len();
    let mut out = vec![0.0f32; nh * dv];
    let mut lse = vec![0.0f32; nh];
    let mut s = vec![0.0f32; n];
    for h in 0..nh {
        let qh = &q[h * dn..(h + 1) * dn];
        for j in 0..n {
            // The reference's matmul output is BF16, then scaled.
            s[j] = rounding.at(dot(qh, &keys[j][h * dn..(h + 1) * dn])) * scale;
        }
        let (m, sum) = softmax_inplace(&mut s);
        lse[h] = m + sum.ln();
        let o = &mut out[h * dv..(h + 1) * dv];
        for j in 0..n {
            let p = rounding.at(s[j]);
            let vj = &values[j][h * dv..(h + 1) * dv];
            for r in 0..dv {
                o[r] += p * vj[r];
            }
        }
        rounding.apply(o);
    }
    AttnRow { out, lse }
}

/// Absorbed query: `q'_h = (W^K_h)^T q_h`, laid out `[n_heads][kv_lora_rank]`.
pub fn absorb_q(cfg: &DsaConfig, w: &MlaWeights, q: &[f32]) -> Vec<f32> {
    let (nh, dn, kl) = (cfg.n_heads, cfg.qk_nope_head_dim, cfg.kv_lora_rank);
    let mut qa = vec![0.0f32; nh * kl];
    for h in 0..nh {
        let o = &mut qa[h * kl..(h + 1) * kl];
        for r in 0..dn {
            let qr = q[h * dn + r];
            let row = w.wk_row(cfg, h, r);
            for l in 0..kl {
                o[l] += qr * row[l];
            }
        }
    }
    qa
}

/// Absorbed attention over latents: returns `o'` `[n_heads][kv_lora_rank]` and
/// the per-head log-sum-exp.
pub fn attend_absorbed_latent(cfg: &DsaConfig, q_abs: &[f32], latents: &[&[f32]]) -> (Vec<f32>, Vec<f32>) {
    let (nh, kl) = (cfg.n_heads, cfg.kv_lora_rank);
    let scale = cfg.attn_scale();
    let n = latents.len();
    let mut out = vec![0.0f32; nh * kl];
    let mut lse = vec![0.0f32; nh];
    let mut s = vec![0.0f32; n];
    for h in 0..nh {
        let qh = &q_abs[h * kl..(h + 1) * kl];
        for j in 0..n {
            s[j] = dot(qh, latents[j]) * scale;
        }
        let (m, sum) = softmax_inplace(&mut s);
        lse[h] = m + sum.ln();
        let o = &mut out[h * kl..(h + 1) * kl];
        for j in 0..n {
            let p = s[j];
            for l in 0..kl {
                o[l] += p * latents[j][l];
            }
        }
    }
    (out, lse)
}

/// `o_h = W^V_h o'_h`: from latent space to value space, `[n_heads][v_head_dim]`.
pub fn unabsorb_v(cfg: &DsaConfig, w: &MlaWeights, o_lat: &[f32]) -> Vec<f32> {
    let (nh, dv, kl) = (cfg.n_heads, cfg.v_head_dim, cfg.kv_lora_rank);
    let mut out = vec![0.0f32; nh * dv];
    for h in 0..nh {
        let ol = &o_lat[h * kl..(h + 1) * kl];
        for r in 0..dv {
            out[h * dv + r] = dot(w.wv_row(cfg, h, r), ol);
        }
    }
    out
}

/// Absorbed attention for one query row (absorb, attend, un-absorb).
pub fn attend_absorbed(cfg: &DsaConfig, w: &MlaWeights, q: &[f32], latents: &[&[f32]]) -> AttnRow {
    let qa = absorb_q(cfg, w, q);
    let (ol, lse) = attend_absorbed_latent(cfg, &qa, latents);
    AttnRow { out: unabsorb_v(cfg, w, &ol), lse }
}

/// `o_proj` over the concatenated heads.
pub fn o_proj(cfg: &DsaConfig, w: &MlaWeights, attn: &[f32]) -> Vec<f32> {
    matvec(&w.o_proj, cfg.hidden, cfg.n_heads * cfg.v_head_dim, attn)
}

/// Bound on `|s_absorbed - s_expanded|` for one pre-scale score `q_h . (W^K_h c)`
/// computed both ways in f32 with sequential sums:
/// `2 gamma(dn + kl + 1) * sum_{r,l} |q_r| |W_rl| |c_l|`.
///
/// Each form is a two-stage product (a `kl`-term then a `dn`-term sum, or the
/// reverse); each stage's forward error is bounded by `gamma` of its length
/// times the absolute products, which compose to at most `gamma(dn + kl + 1)` of the
/// triple-absolute sum. The two forms each satisfy that bound against the exact
/// value, hence the factor 2.
pub fn score_error_bound(cfg: &DsaConfig, w: &MlaWeights, h: usize, q_h: &[f32], latent: &[f32]) -> f64 {
    let mut s = 0.0f64;
    for (r, qr) in q_h.iter().enumerate().take(cfg.qk_nope_head_dim) {
        let row = w.wk_row(cfg, h, r);
        let inner: f64 = row.iter().zip(latent).map(|(a, b)| (*a as f64 * *b as f64).abs()).sum();
        s += (*qr as f64).abs() * inner;
    }
    2.0 * gamma(cfg.qk_nope_head_dim + cfg.kv_lora_rank + 1) * s
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn absorbed_equals_expanded_tiny() {
        let cfg = DsaConfig::tiny();
        let mut rng = Rng::new(7);
        let w = MlaWeights::random(&cfg, &mut rng);
        let q = rng.normals(cfg.n_heads * cfg.qk_nope_head_dim, 1.0);
        let lat: Vec<Vec<f32>> = (0..9).map(|_| rng.normals(cfg.kv_lora_rank, 1.0)).collect();
        let (ks, vs): (Vec<_>, Vec<_>) = lat.iter().map(|c| expand_kv(&cfg, &w, c, Rounding::F32)).unzip();
        let e = attend_expanded(&cfg, &q, &ks, &vs, Rounding::F32);
        let refs: Vec<&[f32]> = lat.iter().map(|v| v.as_slice()).collect();
        let a = attend_absorbed(&cfg, &w, &q, &refs);
        for (x, y) in e.out.iter().zip(&a.out) {
            assert!((x - y).abs() <= 1e-5 * (1.0 + x.abs()), "{x} vs {y}");
        }
        for (x, y) in e.lse.iter().zip(&a.lse) {
            assert!((x - y).abs() <= 1e-5 * (1.0 + x.abs()));
        }
    }
}
