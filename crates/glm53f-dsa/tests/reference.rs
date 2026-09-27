//! CPU reference tests: absorbed vs expanded MLA, k-pool/tail/mask edge cases,
//! window-size independence, deterministic ties, and agreement with the
//! `transformers` indexer's output layout.

use glm53f_dsa::config::DsaConfig;
use glm53f_dsa::indexer::{self, reference_topk_indices, IndexerCache, KeyFormat};
use glm53f_dsa::layer::{prefill, DsaLayerWeights, DsaState, LayerOptions, LatentFormat, MlaForm};
use glm53f_dsa::mla::{self, MlaWeights};
use glm53f_dsa::num::{dot, Rounding};
use glm53f_dsa::rng::Rng;
use glm53f_dsa::select::{score_key, top_k, Selection};

fn hidden_rows(cfg: &DsaConfig, n: usize, seed: u64) -> Vec<Vec<f32>> {
    let mut rng = Rng::new(seed);
    (0..n).map(|_| rng.normals(cfg.hidden, 1.0)).collect()
}

/// Scores: the absorbed and expanded forms differ by at most the f32 bound
/// `2 gamma(dn + kl + 1) sum |q| |W| |c|`, at the real GLM-5.3-Flash dimensions.
/// Outputs agree to about 1e-6 relative.
#[test]
fn absorbed_matches_expanded_at_glm_dims() {
    let cfg = DsaConfig::glm53_flash();
    let mut rng = Rng::new(11);
    let (nh, dn, kl) = (cfg.n_heads, cfg.qk_nope_head_dim, cfg.kv_lora_rank);
    // Only kv_b_proj matters here; scale it like the checkpoint's (std ~ 0.05).
    let mut w = MlaWeights {
        q_a_proj: vec![],
        q_a_norm: vec![],
        q_b_proj: vec![],
        kv_a_proj: vec![],
        kv_a_norm: vec![],
        kv_b_proj: rng.normals(nh * (dn + cfg.v_head_dim) * kl, 0.05),
        o_proj: vec![],
    };
    // A few large weights, as real checkpoints have.
    for i in (0..w.kv_b_proj.len()).step_by(4099) {
        w.kv_b_proj[i] *= 20.0;
    }
    let q = rng.normals(nh * dn, 1.0);
    let n = 12;
    let lat: Vec<Vec<f32>> = (0..n).map(|_| rng.normals(kl, 1.0)).collect();
    let qa = mla::absorb_q(&cfg, &w, &q);
    let (ks, vs): (Vec<_>, Vec<_>) = lat.iter().map(|c| mla::expand_kv(&cfg, &w, c, Rounding::F32)).unzip();
    let mut worst = 0.0f64;
    for h in 0..nh {
        for j in 0..n {
            let se = dot(&q[h * dn..(h + 1) * dn], &ks[j][h * dn..(h + 1) * dn]);
            let sa = dot(&qa[h * kl..(h + 1) * kl], &lat[j]);
            let bound = mla::score_error_bound(&cfg, &w, h, &q[h * dn..(h + 1) * dn], &lat[j]);
            let d = (se as f64 - sa as f64).abs();
            assert!(d <= bound, "head {h} latent {j}: |{se} - {sa}| = {d} > bound {bound}");
            worst = worst.max(d / bound);
        }
    }
    // Full attention outputs.
    let e = mla::attend_expanded(&cfg, &q, &ks, &vs, Rounding::F32);
    let refs: Vec<&[f32]> = lat.iter().map(|v| v.as_slice()).collect();
    let a = mla::attend_absorbed(&cfg, &w, &q, &refs);
    let mut num = 0.0f64;
    let mut den = 0.0f64;
    let mut max_rel = 0.0f64;
    for (x, y) in e.out.iter().zip(&a.out) {
        num += ((x - y) as f64).powi(2);
        den += (*x as f64).powi(2);
        max_rel = max_rel.max(((x - y).abs() / (1e-3 + x.abs())) as f64);
    }
    let rel = (num / den).sqrt();
    eprintln!("absorbed vs expanded: worst score diff / bound = {worst:.3e}; output rel L2 = {rel:.3e}; max rel = {max_rel:.3e}");
    assert!(rel < 2e-6, "output relative L2 {rel}");
    for (x, y) in e.lse.iter().zip(&a.lse) {
        assert!((x - y).abs() < 1e-5 * (1.0 + x.abs()));
    }
}

/// The layer in both forms (tiny dims, whole path from hidden states).
#[test]
fn layer_forms_agree() {
    let cfg = DsaConfig::tiny();
    let mut rng = Rng::new(5);
    let w = DsaLayerWeights::random(&cfg, &mut rng);
    let x = hidden_rows(&cfg, 75, 6);
    let mut oe = LayerOptions::f32_absorbed();
    oe.form = MlaForm::Expanded;
    let (_, te) = prefill(&cfg, &w, &x, oe);
    let (_, ta) = prefill(&cfg, &w, &x, LayerOptions::f32_absorbed());
    for (a, e) in ta.iter().zip(&te) {
        assert_eq!(a.selection, e.selection);
        for (u, v) in a.out.iter().zip(&e.out) {
            assert!((u - v).abs() <= 1e-5 * (1.0 + v.abs()), "row {}: {u} vs {v}", a.position);
        }
    }
}

/// Processing in one window, token by token, or in windows of 3 and 8 gives
/// bit-identical outputs, scores and selections (pools complete inside a
/// window are visible to the window's later rows only).
#[test]
fn window_size_does_not_change_results() {
    let cfg = DsaConfig::tiny();
    let mut rng = Rng::new(21);
    let w = DsaLayerWeights::random(&cfg, &mut rng);
    let x = hidden_rows(&cfg, 61, 22);
    for opts in [LayerOptions::f32_absorbed(), LayerOptions::engine()] {
        let (_, whole) = prefill(&cfg, &w, &x, opts);
        for win in [1usize, 3, 8] {
            let mut st = DsaState::new(opts);
            let mut traces = Vec::new();
            for chunk in x.chunks(win) {
                traces.extend(st.forward(&cfg, &w, chunk));
            }
            assert_eq!(traces.len(), whole.len());
            for (a, b) in traces.iter().zip(&whole) {
                assert_eq!(a.selection, b.selection, "window {win} row {}", a.position);
                assert_eq!(a.scores, b.scores);
                assert_eq!(a.out, b.out, "window {win} row {}", a.position);
            }
        }
    }
}

/// k-pool and tail edge cases: prompts shorter than a pool, exact multiples of
/// the pool size, tails of 1 to 3, and rows below and above the dense limit.
#[test]
fn kpool_tail_edges() {
    let cfg = DsaConfig::tiny(); // kpool 4, 8 pools kept
    let mut rng = Rng::new(31);
    let w = DsaLayerWeights::random(&cfg, &mut rng);
    for len in [1usize, 2, 3, 4, 5, 7, 8, 9, 12, 31, 32, 33, 34, 35, 36, 37, 64] {
        let x = hidden_rows(&cfg, len, 100 + len as u64);
        let (st, tr) = prefill(&cfg, &w, &x, LayerOptions::f32_absorbed());
        assert_eq!(st.index.pool_keys.len(), len / 4, "complete pools for len {len}");
        assert_eq!(st.index.tail.len(), len % 4, "tail for len {len}");
        for t in &tr {
            let p = t.position;
            let s = &t.selection;
            assert_eq!(s.visible_pools, (p + 1) / 4);
            assert_eq!(s.tail_start, 4 * ((p + 1) / 4));
            assert_eq!(s.tail_len, (p + 1) % 4);
            assert_eq!(s.pools.len(), s.visible_pools.min(cfg.topk_pools()));
            let toks = s.tokens(4);
            assert!(toks.iter().all(|x| (*x as usize) <= p), "causal");
            assert!(toks.windows(2).all(|w| w[0] < w[1]), "ascending, no duplicates");
            assert!(toks.len() <= cfg.selection_width());
            if s.visible_pools <= cfg.topk_pools() {
                // Dense: every token up to the query.
                assert_eq!(toks, (0..=p as u32).collect::<Vec<_>>(), "dense row {p} of {len}");
            } else {
                // Sparse: exactly the budget plus the tail; the tail is always kept.
                assert_eq!(toks.len(), cfg.index_topk + s.tail_len);
                for i in 0..s.tail_len {
                    assert!(toks.contains(&((s.tail_start + i) as u32)));
                }
                // Nothing outside the kept set scores above the weakest kept pool.
                let kth = s.pools.iter().map(|q| t.scores[*q as usize]).fold(f32::INFINITY, f32::min);
                for (q, sc) in t.scores.iter().enumerate() {
                    if !s.pools.contains(&(q as u32)) {
                        assert!(*sc <= kth);
                    }
                }
            }
        }
    }
}

/// The engine-style cache (pooled keys computed once, tail kept raw) returns the
/// same rows as a recomputation of the `transformers` indexer from the full
/// sequence, including its output layout, for lengths around pool boundaries.
#[test]
fn matches_reference_layout() {
    let cfg = DsaConfig::tiny();
    let mut rng = Rng::new(41);
    let w = DsaLayerWeights::random(&cfg, &mut rng);
    for len in [1usize, 3, 4, 6, 33, 34, 35, 36, 50] {
        let x = hidden_rows(&cfg, len, 200 + len as u64);
        let (st, tr) = prefill(&cfg, &w, &x, LayerOptions::f32_absorbed());
        let complete = len / 4;
        let select_k = cfg.topk_pools().min(complete);
        let mask = vec![true; len];
        let ref_rows = reference_topk_indices(&cfg, &mask, |q, j| {
            indexer::index_score(&cfg, &tr[q].proj.idx.q, &tr[q].proj.idx.w, &st.index.pool_keys[j])
        });
        for t in &tr {
            let ours = t.selection.reference_row(4, select_k, true, cfg.selection_width());
            assert_eq!(ours, ref_rows[t.position], "len {len} row {}", t.position);
        }
    }
}

/// Left padding (the reference's batched case): pools align to the first real
/// token, so a padded sequence selects the unpadded sequence's tokens shifted by
/// the pad, and padded rows select nothing.
#[test]
fn padding_aligns_pools_to_first_valid_token() {
    let cfg = DsaConfig::tiny();
    let mut rng = Rng::new(51);
    let w = DsaLayerWeights::random(&cfg, &mut rng);
    let len = 45;
    let pad = 3;
    let x = hidden_rows(&cfg, len, 300);
    let (st, tr) = prefill(&cfg, &w, &x, LayerOptions::f32_absorbed());
    let mut mask = vec![false; pad];
    mask.extend(std::iter::repeat_n(true, len));
    let padded = reference_topk_indices(&cfg, &mask, |q, j| {
        if q < pad {
            0.0
        } else {
            indexer::index_score(&cfg, &tr[q - pad].proj.idx.q, &tr[q - pad].proj.idx.w, &st.index.pool_keys[j])
        }
    });
    for q in 0..pad {
        assert!(padded[q].iter().all(|v| *v == -1), "padded row {q}");
    }
    let select_k = cfg.topk_pools().min(len / 4);
    for t in &tr {
        let ours = t.selection.reference_row(4, select_k, true, cfg.selection_width());
        let shifted: Vec<i32> = ours.iter().map(|v| if *v < 0 { -1 } else { v + pad as i32 }).collect();
        assert_eq!(shifted, padded[t.position + pad], "row {}", t.position);
    }
}

/// Ties are broken by the lower pool, whatever order the candidates arrive in:
/// selecting per chunk and merging (as the GPU does) equals one global selection.
#[test]
fn ties_are_deterministic_under_chunking() {
    let mut rng = Rng::new(61);
    let n = 5000;
    // Heavy ties: scores drawn from 7 values, including +0 and -0.
    let vals = [0.0f32, -0.0, 1.0, 1.0, 2.5, -3.0, 0.5];
    let scores: Vec<f32> = (0..n).map(|_| vals[rng.below(vals.len())]).collect();
    let direct = top_k(&scores, 512);
    for chunk in [1usize, 7, 64, 700, 5000] {
        let mut cands: Vec<u64> = Vec::new();
        for (ci, c) in scores.chunks(chunk).enumerate() {
            for p in top_k(c, 512) {
                let pool = (ci * chunk) as u32 + p;
                cands.push(score_key(scores[pool as usize], pool));
            }
        }
        // Merge in a scrambled order.
        let mut r2 = Rng::new(chunk as u64);
        for i in (1..cands.len()).rev() {
            cands.swap(i, r2.below(i + 1));
        }
        cands.sort_unstable_by(|a, b| b.cmp(a));
        let merged: Vec<u32> = cands.iter().take(512).map(|k| !(*k as u32)).collect();
        assert_eq!(merged, direct, "chunk {chunk}");
    }
    // The tie rule: among equal scores the lower pools are kept.
    let two_five: Vec<u32> = (0..n as u32).filter(|p| scores[*p as usize] == 2.5).collect();
    let kept: Vec<u32> = direct.iter().copied().filter(|p| scores[*p as usize] == 2.5).collect();
    assert_eq!(&kept[..], &two_five[..kept.len()]);
}

/// Selection at the real budget (512 pools) over thousands of pools, driven by
/// pooled keys directly (the indexer at GLM-5.3-Flash dimensions).
#[test]
fn long_context_selection_at_glm_dims() {
    let cfg = DsaConfig::glm53_flash();
    let mut rng = Rng::new(71);
    let d = cfg.index_head_dim;
    let n_pools = 3000;
    let mut cache = IndexerCache::new(KeyFormat::F32, Rounding::F32);
    cache.pool_keys = (0..n_pools).map(|_| rng.normals(d, 1.0)).collect();
    cache.len = n_pools * 4 + 2; // two tail tokens
    let q = rng.normals(cfg.index_n_heads * d, 1.0);
    let w: Vec<f32> = rng.normals(cfg.index_n_heads, 1.0);
    for pos in [2047usize, 2048, 2050, 2051, 2052, 5000, n_pools * 4 + 1] {
        let (sel, scores) = cache.select(&cfg, pos, &q, &w);
        let vis = (pos + 1) / 4;
        assert_eq!(scores.len(), vis);
        assert_eq!(sel.pools.len(), vis.min(512));
        let toks = sel.tokens(4);
        assert_eq!(toks.len(), 4 * vis.min(512) + (pos + 1) % 4);
        assert!(toks.len() <= 2051);
        if (pos + 1) % 4 != 0 || vis <= 512 {
            // The query's own token is attended whenever it is in the tail or the row is dense.
            assert_eq!(*toks.last().unwrap() as usize, pos);
        }
        if vis > 512 {
            let kth = sel.pools.iter().map(|p| scores[*p as usize]).fold(f32::INFINITY, f32::min);
            let above = scores.iter().filter(|s| **s > kth).count();
            assert!(above < 512);
        }
    }
    // Dense limit: position 2050 sees 512 pools (all kept, 2051 tokens); 2051 sees 513.
    assert_eq!(cache.select(&cfg, 2050, &q, &w).0.tokens(4).len(), 2051);
    assert_eq!(cache.select(&cfg, 2051, &q, &w).0.tokens(4).len(), 2048);
}

/// The FP8 engine storage changes selections only near ties and changes the
/// output by a small relative amount (tiny dims; the real-data numbers are in
/// the cache-format tests).
#[test]
fn engine_storage_is_close_to_f32() {
    let cfg = DsaConfig::tiny();
    let mut rng = Rng::new(81);
    let w = DsaLayerWeights::random(&cfg, &mut rng);
    let x = hidden_rows(&cfg, 90, 82);
    let (_, exact) = prefill(&cfg, &w, &x, LayerOptions::f32_absorbed());
    let mut o = LayerOptions::f32_absorbed();
    o.latent = LatentFormat::Fp8(glm53f_dsa::fp8::ScaleMode::Pow2);
    let (_, fp8) = prefill(&cfg, &w, &x, o);
    let mut num = 0.0f64;
    let mut den = 0.0f64;
    for (a, b) in exact.iter().zip(&fp8) {
        assert_eq!(a.selection, b.selection, "latent format does not touch the indexer");
        for (u, v) in a.out.iter().zip(&b.out) {
            num += ((u - v) as f64).powi(2);
            den += (*u as f64).powi(2);
        }
    }
    let rel = (num / den).sqrt();
    assert!(rel < 0.08, "FP8 latent output relative L2 {rel}");
    let _ = Selection::dense(0, 4, true);
}
