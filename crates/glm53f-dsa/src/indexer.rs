//! The DSA indexer with k-pool compression (f32 reference).
//!
//! Semantics follow `Glm5NextTextIndexer` in `transformers`
//! (`models/glm5_next/modeling_glm5_next.py`, see PROVENANCE.md):
//!
//! * per token: index query `q = wq_b(q_resid)` (heads x dim), index key
//!   `k = LayerNorm(wk(x))`, pool gate `g = compress_gate(x)` (dim wide, one gate
//!   per key channel) and head weights `w = weights_proj(x) * heads^-0.5`;
//! * every `kpool` consecutive tokens (aligned to the first valid token) form a
//!   pool whose key is, per channel `d`, `sum_i softmax_i(g_i[d] + ape[i][d]) k_i[d]`;
//! * a query at position `p` scores the complete pools that end at or before
//!   `p`: `s = sum_h w_h relu(scale * q_h . key)`, keeps the top
//!   `index_topk / kpool` pools and appends the incomplete tail pool's visible
//!   tokens.
//!
//! A pooled key depends only on its own `kpool` tokens, so it can be computed
//! once, when its last token arrives, and cached. That is what [`IndexerCache`]
//! does (and what the engine does).

use crate::config::DsaConfig;
use crate::fp8::{self, ScaleMode};
use crate::num::{bf16_round, dot, layer_norm, matvec, Rounding};
use crate::rng::Rng;
use crate::select::Selection;

/// Indexer weights, dequantized to f32. Shapes are `[out, in]` row-major, as in
/// the checkpoint.
#[derive(Clone, Debug)]
pub struct IndexerWeights {
    /// `wq_b`: `[index_n_heads * index_head_dim, q_lora_rank]`.
    pub wq_b: Vec<f32>,
    /// `wk`: `[index_head_dim, hidden]`.
    pub wk: Vec<f32>,
    /// `k_norm.weight`: `[index_head_dim]`.
    pub k_norm_w: Vec<f32>,
    /// `k_norm.bias`: `[index_head_dim]`.
    pub k_norm_b: Vec<f32>,
    /// `weights_proj`: `[index_n_heads, hidden]`.
    pub weights_proj: Vec<f32>,
    /// `index_kpool_compress_gate`: `[index_head_dim, hidden]`.
    pub gate: Vec<f32>,
    /// `index_kpool_compress_ape`: `[index_kpool, index_head_dim]`.
    pub ape: Vec<f32>,
}

impl IndexerWeights {
    /// Random weights with roughly unit-variance outputs.
    pub fn random(cfg: &DsaConfig, rng: &mut Rng) -> Self {
        let (h, d, nh, ql) = (cfg.hidden, cfg.index_head_dim, cfg.index_n_heads, cfg.q_lora_rank);
        Self {
            wq_b: rng.normals(nh * d * ql, 1.0 / (ql as f32).sqrt()),
            wk: rng.normals(d * h, 1.0 / (h as f32).sqrt()),
            k_norm_w: (0..d).map(|_| 1.0 + 0.2 * rng.normal()).collect(),
            k_norm_b: rng.normals(d, 0.1),
            weights_proj: rng.normals(nh * h, 1.0 / (h as f32).sqrt()),
            gate: rng.normals(d * h, 1.0 / (h as f32).sqrt()),
            ape: rng.normals(cfg.index_kpool * d, 0.5),
        }
    }

    pub fn check(&self, cfg: &DsaConfig) -> Result<(), String> {
        let (h, d, nh, ql) = (cfg.hidden, cfg.index_head_dim, cfg.index_n_heads, cfg.q_lora_rank);
        let want = [
            ("wq_b", self.wq_b.len(), nh * d * ql),
            ("wk", self.wk.len(), d * h),
            ("k_norm.weight", self.k_norm_w.len(), d),
            ("k_norm.bias", self.k_norm_b.len(), d),
            ("weights_proj", self.weights_proj.len(), nh * h),
            ("compress_gate", self.gate.len(), d * h),
            ("compress_ape", self.ape.len(), cfg.index_kpool * d),
        ];
        for (n, got, exp) in want {
            if got != exp {
                return Err(format!("indexer {n}: {got} values, expected {exp}"));
            }
        }
        Ok(())
    }
}

/// One token's indexer projections.
#[derive(Clone, Debug, PartialEq)]
pub struct IndexerToken {
    /// Index query, `[index_n_heads * index_head_dim]`.
    pub q: Vec<f32>,
    /// Head weights including the `heads^-0.5` factor, `[index_n_heads]`.
    pub w: Vec<f32>,
    /// Index key after `k_norm`, `[index_head_dim]`.
    pub k: Vec<f32>,
    /// Pool gate logits, `[index_head_dim]`.
    pub gate: Vec<f32>,
}

/// Project one token. `hidden` is the layer's normalized input, `q_resid` is
/// `q_a_layernorm(q_a_proj(hidden))` from the MLA query path.
pub fn project_token(
    cfg: &DsaConfig,
    wts: &IndexerWeights,
    hidden: &[f32],
    q_resid: &[f32],
    rounding: Rounding,
) -> IndexerToken {
    let (h, d, nh) = (cfg.hidden, cfg.index_head_dim, cfg.index_n_heads);
    let mut q = matvec(&wts.wq_b, nh * d, cfg.q_lora_rank, q_resid);
    rounding.apply(&mut q);
    let mut k_raw = matvec(&wts.wk, d, h, hidden);
    rounding.apply(&mut k_raw);
    let mut k = layer_norm(&k_raw, &wts.k_norm_w, &wts.k_norm_b, cfg.index_k_norm_eps);
    rounding.apply(&mut k);
    let mut gate = matvec(&wts.gate, d, h, hidden);
    rounding.apply(&mut gate);
    let mut w = matvec(&wts.weights_proj, nh, h, hidden);
    rounding.apply(&mut w);
    let ws = cfg.index_weight_scale();
    for v in w.iter_mut() {
        *v *= ws;
    }
    IndexerToken { q, w, k, gate }
}

/// Pooled key of one complete pool: per channel, the softmax over the pool's
/// tokens of `gate + ape`, applied to the tokens' keys.
///
/// With `Rounding::Bf16Ref` the probabilities, the products and the sum are
/// rounded to BF16 as in the reference (which pools BF16 keys).
pub fn pool_key(cfg: &DsaConfig, ape: &[f32], keys: &[&[f32]], gates: &[&[f32]], rounding: Rounding) -> Vec<f32> {
    let (d, kp) = (cfg.index_head_dim, cfg.index_kpool);
    assert_eq!(keys.len(), kp);
    assert_eq!(gates.len(), kp);
    let mut out = vec![0.0f32; d];
    let mut logit = vec![0.0f32; kp];
    for c in 0..d {
        let mut m = f32::NEG_INFINITY;
        for i in 0..kp {
            logit[i] = gates[i][c] + ape[i * d + c];
            m = m.max(logit[i]);
        }
        let mut sum = 0.0f32;
        for l in logit.iter_mut() {
            *l = (*l - m).exp();
            sum += *l;
        }
        let mut acc = 0.0f32;
        for i in 0..kp {
            let p = rounding.at(logit[i] / sum);
            acc += rounding.at(p * keys[i][c]);
        }
        out[c] = rounding.at(acc);
    }
    out
}

/// Index score of one pooled key for one query row:
/// `sum_h w_h * relu(scale * (q_h . key))`.
pub fn index_score(cfg: &DsaConfig, q: &[f32], w: &[f32], key: &[f32]) -> f32 {
    let d = cfg.index_head_dim;
    let scale = cfg.index_scale();
    let mut s = 0.0f32;
    for h in 0..cfg.index_n_heads {
        let dq = dot(&q[h * d..(h + 1) * d], key) * scale;
        s += w[h] * dq.max(0.0);
    }
    s
}

/// How pooled keys are stored before scoring.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum KeyFormat {
    /// Keep f32 (the reference).
    F32,
    /// Round to BF16.
    Bf16,
    /// FP8 E4M3 with one f32 scale per key.
    Fp8(ScaleMode),
}

impl KeyFormat {
    /// The key as the scorer will see it.
    pub fn store(self, key: &[f32]) -> Vec<f32> {
        match self {
            KeyFormat::F32 => key.to_vec(),
            KeyFormat::Bf16 => key.iter().map(|v| bf16_round(*v)).collect(),
            KeyFormat::Fp8(mode) => {
                let mut codes = vec![0u8; key.len()];
                let s = fp8::quantize_block(key, &mut codes, mode);
                let mut out = vec![0.0f32; key.len()];
                fp8::dequantize_block(&codes, s, &mut out);
                out
            }
        }
    }
}

/// The per-request indexer state the engine keeps: pooled keys of complete
/// pools and the raw keys and gates of the incomplete tail pool (fewer than
/// `kpool` tokens).
#[derive(Clone, Debug)]
pub struct IndexerCache {
    /// Pooled keys as stored (after [`KeyFormat::store`]).
    pub pool_keys: Vec<Vec<f32>>,
    /// Pooled keys before storage rounding (for error measurements).
    pub pool_keys_exact: Vec<Vec<f32>>,
    /// Raw `(key, gate)` of the tail tokens.
    pub tail: Vec<(Vec<f32>, Vec<f32>)>,
    /// Tokens appended so far.
    pub len: usize,
    pub format: KeyFormat,
    pub rounding: Rounding,
}

impl IndexerCache {
    pub fn new(format: KeyFormat, rounding: Rounding) -> Self {
        Self { pool_keys: Vec::new(), pool_keys_exact: Vec::new(), tail: Vec::new(), len: 0, format, rounding }
    }

    /// Append one token's key and gate; completes a pool every `kpool` tokens.
    /// Returns the index of the pool completed by this token, if any.
    pub fn push(&mut self, cfg: &DsaConfig, ape: &[f32], k: &[f32], gate: &[f32]) -> Option<usize> {
        self.tail.push((k.to_vec(), gate.to_vec()));
        self.len += 1;
        if self.tail.len() < cfg.index_kpool {
            return None;
        }
        let keys: Vec<&[f32]> = self.tail.iter().map(|(k, _)| k.as_slice()).collect();
        let gates: Vec<&[f32]> = self.tail.iter().map(|(_, g)| g.as_slice()).collect();
        let key = pool_key(cfg, ape, &keys, &gates, self.rounding);
        self.pool_keys.push(self.format.store(&key));
        self.pool_keys_exact.push(key);
        self.tail.clear();
        Some(self.pool_keys.len() - 1)
    }

    /// Scores of the pools visible to a query at `position` (which must already
    /// be in the cache).
    pub fn scores(&self, cfg: &DsaConfig, position: usize, q: &[f32], w: &[f32]) -> Vec<f32> {
        assert!(position < self.len, "query position {position} not in the cache ({})", self.len);
        let visible = cfg.visible_pools(position);
        self.pool_keys[..visible].iter().map(|key| index_score(cfg, q, w, key)).collect()
    }

    /// Select for a query at `position`.
    pub fn select(&self, cfg: &DsaConfig, position: usize, q: &[f32], w: &[f32]) -> (Selection, Vec<f32>) {
        let scores = self.scores(cfg, position, q, w);
        let sel = Selection::from_scores(position, cfg.index_kpool, cfg.topk_pools(), cfg.always_select_tail, &scores);
        (sel, scores)
    }
}

/// The `transformers` indexer's output for one sequence (batch 1), including
/// left padding: `mask[t]` says whether token `t` is a real token.
///
/// `score(q, pool)` gives the index score of kept pool `pool` (numbered from the
/// first valid token) for query row `q`. Returns `[seq_len][selection_width]`
/// token indices (absolute positions in the padded sequence, -1 unused), with
/// this crate's tie rule for equal scores.
pub fn reference_topk_indices(
    cfg: &DsaConfig,
    mask: &[bool],
    mut score: impl FnMut(usize, usize) -> f32,
) -> Vec<Vec<i32>> {
    let s = mask.len();
    let kp = cfg.index_kpool;
    let width = cfg.selection_width();
    let first_key = mask.iter().position(|v| *v).unwrap_or(s);
    let n_pools = s.div_ceil(kp);
    // pool_indices[p][i] = first_key + kp p + i, valid if inside the sequence and a real token.
    let mut kept: Vec<usize> = Vec::new(); // kept (complete, valid) pools
    for p in 0..n_pools {
        let ok = (0..kp).all(|i| {
            let t = first_key + kp * p + i;
            t < s && mask[t]
        });
        if ok {
            kept.push(p);
        }
    }
    let select_k = cfg.topk_pools().min(kept.len());
    let mut out = Vec::with_capacity(s);
    for q in 0..s {
        let visible = |t: usize| t <= q && mask[t];
        // Candidate scores over kept pools; invisible ones get f32::MIN (finfo.min).
        let mut keys: Vec<(u64, usize, bool)> = kept
            .iter()
            .enumerate()
            .map(|(j, p)| {
                let end = first_key + kp * p + kp - 1;
                let cand = visible(end);
                let sc = if cand { score(q, j) } else { f32::MIN };
                (crate::select::score_key(sc, j as u32), *p, cand)
            })
            .collect();
        keys.sort_unstable_by(|a, b| b.0.cmp(&a.0));
        let mut row: Vec<i32> = Vec::with_capacity(width);
        for (_, p, cand) in keys.iter().take(select_k) {
            for i in 0..kp {
                row.push(if *cand { (first_key + kp * p + i) as i32 } else { -1 });
            }
        }
        if cfg.always_select_tail && kp > 1 {
            let visible_count = (0..s).filter(|t| visible(*t)).count();
            let tail_count = visible_count % kp;
            let tail_start = first_key + visible_count - tail_count;
            for i in 0..kp - 1 {
                let t = tail_start + i;
                let ok = i < tail_count && t < s && visible(t.min(s.saturating_sub(1)));
                row.push(if ok { t as i32 } else { -1 });
            }
        }
        row.resize(width, -1);
        row.truncate(width);
        if !mask[q] {
            row.iter_mut().for_each(|v| *v = -1);
        }
        out.push(row);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pool_key_uniform_gate_is_mean() {
        let cfg = DsaConfig::tiny();
        let d = cfg.index_head_dim;
        let ape = vec![0.0f32; cfg.index_kpool * d];
        let keys: Vec<Vec<f32>> = (0..4).map(|i| vec![i as f32; d]).collect();
        let gates: Vec<Vec<f32>> = (0..4).map(|_| vec![0.3f32; d]).collect();
        let kr: Vec<&[f32]> = keys.iter().map(|v| v.as_slice()).collect();
        let gr: Vec<&[f32]> = gates.iter().map(|v| v.as_slice()).collect();
        let pk = pool_key(&cfg, &ape, &kr, &gr, Rounding::F32);
        assert!(pk.iter().all(|v| (v - 1.5).abs() < 1e-6));
    }

    #[test]
    fn pool_key_gate_is_per_channel() {
        let cfg = DsaConfig::tiny();
        let d = cfg.index_head_dim;
        let ape = vec![0.0f32; cfg.index_kpool * d];
        let keys: Vec<Vec<f32>> = (0..4).map(|i| vec![i as f32; d]).collect();
        // Channel 0 gates token 3 hard; channel 1 gates token 0 hard.
        let mut gates: Vec<Vec<f32>> = (0..4).map(|_| vec![0.0f32; d]).collect();
        gates[3][0] = 100.0;
        gates[0][1] = 100.0;
        let kr: Vec<&[f32]> = keys.iter().map(|v| v.as_slice()).collect();
        let gr: Vec<&[f32]> = gates.iter().map(|v| v.as_slice()).collect();
        let pk = pool_key(&cfg, &ape, &kr, &gr, Rounding::F32);
        assert!((pk[0] - 3.0).abs() < 1e-6);
        assert!(pk[1].abs() < 1e-6);
        assert!((pk[2] - 1.5).abs() < 1e-6);
    }
}
