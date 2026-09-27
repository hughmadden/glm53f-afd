//! Real GLM-5.3-Flash layer-3 weights: the error of each cache format against
//! the f32 reference (latent and pooled index key), the selection change the
//! FP8 pooled keys cause, and the attention/layer output error.
//!
//! Needs `GLM53F_CHECKPOINT` (a directory of the official checkpoint's
//! safetensors, or a subset holding layer 3's attention tensors and the token
//! embedding) and a release build; skips otherwise:
//!
//!   GLM53F_CHECKPOINT=/path/to/GLM-5.3-Flash cargo test --release --test real_data -- --nocapture
//!
//! Inputs: when oracle goldens exist, their real layer-3 inputs are used for a
//! short-context check. The long-context measurement uses **proxy inputs**:
//! token-embedding rows (pseudo-random ids) through layer 3's input RMSNorm. They
//! have the checkpoint's per-channel structure but are not real layer-3
//! activations; the numbers are labelled accordingly.

use glm53f_dsa::config::DsaConfig;
use glm53f_dsa::fp8::{round_trip_grouped, ScaleMode};
use glm53f_dsa::indexer::{index_score, pool_key};
use glm53f_dsa::layer::DsaLayerWeights;
use glm53f_dsa::mla;
use glm53f_dsa::num::{bf16_round, layer_norm, matvec, rms_norm, Rounding};
use glm53f_dsa::rng::Rng;
use glm53f_dsa::safetensors::Checkpoint;
use glm53f_dsa::select::top_k;
use glm53f_dsa::weights;

fn rel_l2(a: &[f32], b: &[f32]) -> f64 {
    let num: f64 = a.iter().zip(b).map(|(x, y)| ((x - y) as f64).powi(2)).sum();
    let den: f64 = a.iter().map(|x| (*x as f64).powi(2)).sum();
    (num / den.max(1e-300)).sqrt()
}

fn setup() -> Option<(Checkpoint, DsaConfig, DsaLayerWeights)> {
    if cfg!(debug_assertions) {
        eprintln!("skipping: real-data tests need --release");
        return None;
    }
    let Some(ck) = weights::checkpoint_from_env() else {
        eprintln!("skipping: GLM53F_CHECKPOINT not set or unreadable");
        return None;
    };
    if !weights::layer_available(&ck, 3) {
        eprintln!("skipping: layer 3 tensors not present in GLM53F_CHECKPOINT");
        return None;
    }
    let cfg = DsaConfig::glm53_flash();
    let w = weights::load_dsa_layer(&ck, &cfg, 3).expect("load layer 3");
    Some((ck, cfg, w))
}

/// Proxy layer-3 inputs: embedding rows of pseudo-random token ids through the
/// layer's input RMSNorm.
fn proxy_inputs(ck: &Checkpoint, cfg: &DsaConfig, n: usize, seed: u64) -> Vec<Vec<f32>> {
    let emb = ["model.language_model.embed_tokens.weight", "model.embed_tokens.weight"]
        .into_iter()
        .find(|name| ck.find(name).is_some())
        .expect("token embedding in the checkpoint");
    let norm = ["model.language_model.layers.3.input_layernorm.weight", "model.layers.3.input_layernorm.weight"]
        .into_iter()
        .find(|name| ck.find(name).is_some())
        .expect("layer 3 input_layernorm");
    let (w_in, _) = ck.read_f32(norm).unwrap();
    let mut rng = Rng::new(seed);
    (0..n)
        .map(|_| {
            let id = 1000 + rng.below(140_000);
            let row = ck.read_rows(emb, id, 1).unwrap();
            rms_norm(&row, &w_in, cfg.rms_norm_eps, Rounding::F32)
        })
        .collect()
}

struct TokenData {
    latent: Vec<f32>,
    k: Vec<f32>,
    gate: Vec<f32>,
}

fn token_data(cfg: &DsaConfig, w: &DsaLayerWeights, x: &[f32]) -> TokenData {
    let latent = mla::project_latent(cfg, &w.mla, x, Rounding::F32);
    let d = cfg.index_head_dim;
    let k_raw = matvec(&w.idx.wk, d, cfg.hidden, x);
    let k = layer_norm(&k_raw, &w.idx.k_norm_w, &w.idx.k_norm_b, cfg.index_k_norm_eps);
    let gate = matvec(&w.idx.gate, d, cfg.hidden, x);
    TokenData { latent, k, gate }
}

struct QueryData {
    q: Vec<f32>,
    iq: Vec<f32>,
    iw: Vec<f32>,
}

fn query_data(cfg: &DsaConfig, w: &DsaLayerWeights, x: &[f32]) -> QueryData {
    let (q_resid, q) = mla::project_q(cfg, &w.mla, x, Rounding::F32);
    let t = glm53f_dsa::indexer::project_token(cfg, &w.idx, x, &q_resid, Rounding::F32);
    QueryData { q, iq: t.q, iw: t.w }
}

#[test]
fn layer3_cache_format_error_on_proxy_inputs() {
    let Some((ck, cfg, w)) = setup() else { return };
    let n = 2400; // positions 0..2399; the last rows see 600 pools and keep 512
    let xs = proxy_inputs(&ck, &cfg, n, 17);
    let toks: Vec<TokenData> = xs.iter().map(|x| token_data(&cfg, &w, x)).collect();
    let n_pools = n / 4;
    let pools: Vec<Vec<f32>> = (0..n_pools)
        .map(|p| {
            let keys: Vec<&[f32]> = (0..4).map(|i| toks[4 * p + i].k.as_slice()).collect();
            let gates: Vec<&[f32]> = (0..4).map(|i| toks[4 * p + i].gate.as_slice()).collect();
            pool_key(&cfg, &w.idx.ape, &keys, &gates, Rounding::F32)
        })
        .collect();

    eprintln!("\n== layer 3, real weights, PROXY inputs ({n} tokens: embedding rows through input_layernorm) ==");
    // 1. Stored-value error per format.
    let lat_all: Vec<f32> = toks.iter().flat_map(|t| t.latent.iter().copied()).collect();
    let key_all: Vec<f32> = pools.iter().flatten().copied().collect();
    let bf16 = |v: &[f32]| v.iter().map(|x| bf16_round(*x)).collect::<Vec<_>>();
    eprintln!("latent (512 per token), relative L2 error of the stored values:");
    eprintln!("  BF16                         {:.3e}", rel_l2(&lat_all, &bf16(&lat_all)));
    for (label, g, m) in [
        ("FP8, pow2 scale per 128 (default)", 128, ScaleMode::Pow2),
        ("FP8, amax/448 scale per 128", 128, ScaleMode::Amax),
        ("FP8, pow2 scale per 512", 512, ScaleMode::Pow2),
        ("FP8, pow2 scale per 32", 32, ScaleMode::Pow2),
    ] {
        let rt: Vec<f32> = lat_all.chunks(512).flat_map(|c| round_trip_grouped(c, g, m)).collect();
        eprintln!("  {label:34} {:.3e}", rel_l2(&lat_all, &rt));
    }
    let key_fp8: Vec<f32> = key_all.chunks(128).flat_map(|c| round_trip_grouped(c, 128, ScaleMode::Pow2)).collect();
    eprintln!("pooled index key (128 per pool):");
    eprintln!("  BF16                         {:.3e}", rel_l2(&key_all, &bf16(&key_all)));
    eprintln!("  FP8, pow2 scale per key      {:.3e}", rel_l2(&key_all, &key_fp8));

    // 2. Index scores and selections for the last query rows.
    let q_rows = 8;
    let key_formats: Vec<(&str, Vec<Vec<f32>>)> = vec![
        ("f32", pools.clone()),
        ("bf16", pools.iter().map(|k| bf16(k)).collect()),
        ("fp8", key_fp8.chunks(128).map(|c| c.to_vec()).collect()),
    ];
    let lat_formats: Vec<(&str, Vec<Vec<f32>>)> = vec![
        ("f32", toks.iter().map(|t| t.latent.clone()).collect()),
        ("bf16", toks.iter().map(|t| bf16(&t.latent)).collect()),
        ("fp8 pow2/128", toks.iter().map(|t| round_trip_grouped(&t.latent, 128, ScaleMode::Pow2)).collect()),
        ("fp8 amax/128", toks.iter().map(|t| round_trip_grouped(&t.latent, 128, ScaleMode::Amax)).collect()),
        ("fp8 pow2/512", toks.iter().map(|t| round_trip_grouped(&t.latent, 512, ScaleMode::Pow2)).collect()),
    ];
    let mut score_err = vec![0.0f64; key_formats.len()];
    let mut overlap = vec![0.0f64; key_formats.len()];
    let mut attn_err = vec![0.0f64; lat_formats.len()];
    let mut out_err = vec![0.0f64; lat_formats.len()];
    let mut e2e_err = 0.0f64;
    for pos in n - q_rows..n {
        let qd = query_data(&cfg, &w, &xs[pos]);
        let vis = (pos + 1) / 4;
        let scores: Vec<Vec<f32>> =
            key_formats.iter().map(|(_, ks)| ks[..vis].iter().map(|k| index_score(&cfg, &qd.iq, &qd.iw, k)).collect()).collect();
        let sels: Vec<Vec<u32>> = scores.iter().map(|s| top_k(s, 512)).collect();
        for f in 0..key_formats.len() {
            score_err[f] += rel_l2(&scores[0], &scores[f]) / q_rows as f64;
            let common = sels[f].iter().filter(|p| sels[0].contains(p)).count();
            overlap[f] += common as f64 / 512.0 / q_rows as f64;
        }
        // Attention over the f32-key selection, per latent format.
        let sel = glm53f_dsa::select::Selection::from_scores(pos, 4, 512, true, &scores[0]);
        let tokens = sel.tokens(4);
        let mut outs = Vec::new();
        for (_, lats) in &lat_formats {
            let refs: Vec<&[f32]> = tokens.iter().map(|t| lats[*t as usize].as_slice()).collect();
            let a = mla::attend_absorbed(&cfg, &w.mla, &qd.q, &refs);
            let o = mla::o_proj(&cfg, &w.mla, &a.out);
            outs.push((a.out, o));
        }
        for f in 0..lat_formats.len() {
            attn_err[f] += rel_l2(&outs[0].0, &outs[f].0) / q_rows as f64;
            out_err[f] += rel_l2(&outs[0].1, &outs[f].1) / q_rows as f64;
        }
        // End to end: FP8 keys (their own selection) and FP8 latents.
        let sel8 = glm53f_dsa::select::Selection::from_scores(pos, 4, 512, true, &scores[2]);
        let refs: Vec<&[f32]> = sel8.tokens(4).iter().map(|t| lat_formats[2].1[*t as usize].as_slice()).collect();
        let a = mla::attend_absorbed(&cfg, &w.mla, &qd.q, &refs);
        e2e_err += rel_l2(&outs[0].1, &mla::o_proj(&cfg, &w.mla, &a.out)) / q_rows as f64;
    }
    eprintln!("index scores over {} to {} visible pools, {q_rows} query rows (mean):", (n - q_rows + 1) / 4, n / 4);
    for (f, (label, _)) in key_formats.iter().enumerate() {
        eprintln!("  keys {label:5}: score rel L2 {:.3e}; top-512 overlap with f32 keys {:.4}", score_err[f], overlap[f]);
    }
    eprintln!("attention over the f32-key selection (2,048 tokens + tail), mean rel L2 vs f32 latents:");
    for (f, (label, _)) in lat_formats.iter().enumerate() {
        eprintln!("  latents {label:13}: per-head output {:.3e}; layer output (o_proj) {:.3e}", attn_err[f], out_err[f]);
    }
    eprintln!("end to end (FP8 keys and their selection, FP8 pow2/128 latents): layer output rel L2 {e2e_err:.3e}");
    assert!(out_err[2] < 0.1, "FP8 latent layer-output error {}", out_err[2]);
    assert!(overlap[2] > 0.8, "FP8 key selection overlap {}", overlap[2]);
}
