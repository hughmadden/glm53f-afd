//! The CPU reference on the official checkpoint's tensors (layers 0, 3 and 4): invariants,
//! and agreement with direct f64 transcriptions of the reference formulas. Needs
//! `GLM53F_CHECKPOINT_DIR` (see tests/common); skips cleanly without it. Run with
//! `--release --nocapture` to see the weights' statistics.

mod common;

use glm53f_layers::bf16;
use glm53f_layers::fp8::{ActScheme, Fp8Matrix};
use glm53f_layers::math::sigmoid;
use glm53f_layers::mhc::{self, HcParams, HC_EPS};
use glm53f_layers::mlp;
use glm53f_layers::norm::RMS_EPS;
use glm53f_layers::router::{self, ROUTED_SCALE, TOP_K};

/// `Glm5NextTextHyperConnection` in f64: pre, post, comb.
fn mix_f64(streams: &[f32], p: &HcParams) -> ([f64; 4], [f64; 4], [f64; 16]) {
    let n = streams.len();
    let ms: f64 = streams.iter().map(|&x| (x as f64).powi(2)).sum::<f64>() / n as f64;
    let r = 1.0 / (ms + RMS_EPS as f64).sqrt();
    let proj: Vec<f64> = (0..24)
        .map(|q| {
            (0..n)
                .map(|k| streams[k] as f64 * r * bf16::to_f32(p.fn_[q * n + k]) as f64)
                .sum()
        })
        .collect();
    let sig = |x: f64| 1.0 / (1.0 + (-x).exp());
    let (s, b) = (p.scale, p.base);
    let pre = std::array::from_fn(|j| sig(proj[j] * s[0] as f64 + b[j] as f64) + HC_EPS as f64);
    let post = std::array::from_fn(|j| 2.0 * sig(proj[4 + j] * s[1] as f64 + b[4 + j] as f64));
    let mut c = [0f64; 16];
    for i in 0..4 {
        let l: Vec<f64> = (0..4)
            .map(|j| proj[8 + 4 * i + j] * s[2] as f64 + b[8 + 4 * i + j] as f64)
            .collect();
        let m = l.iter().cloned().fold(f64::MIN, f64::max);
        let e: Vec<f64> = l.iter().map(|x| (x - m).exp()).collect();
        let sum: f64 = e.iter().sum();
        for j in 0..4 {
            c[4 * i + j] = e[j] / sum + HC_EPS as f64;
        }
    }
    let norm = |c: &mut [f64; 16], by_row: bool| {
        for a in 0..4 {
            let idx = |b: usize| if by_row { 4 * a + b } else { 4 * b + a };
            let s: f64 = (0..4).map(|b| c[idx(b)]).sum::<f64>() + HC_EPS as f64;
            for b in 0..4 {
                c[idx(b)] /= s;
            }
        }
    };
    norm(&mut c, false);
    for _ in 1..20 {
        norm(&mut c, true);
        norm(&mut c, false);
    }
    (pre, post, c)
}

fn check_mix(streams: &[f32], p: &HcParams, what: &str) -> mhc::HcMix {
    let m = mhc::mix(streams, p, RMS_EPS);
    let (pre, post, comb) = mix_f64(streams, p);
    for j in 0..4 {
        assert!(
            (m.pre[j] as f64 - pre[j]).abs() <= 1e-5 * pre[j],
            "{what}: pre[{j}] {} vs {}",
            m.pre[j],
            pre[j]
        );
        assert!(
            (m.post[j] as f64 - post[j]).abs() <= 1e-5 * post[j].max(1e-3),
            "{what}: post[{j}] {} vs {}",
            m.post[j],
            post[j]
        );
    }
    for l in 0..16 {
        assert!(
            (m.comb[l] as f64 - comb[l]).abs() <= 1e-4 * comb[l] + 1e-7,
            "{what}: comb[{l}] {} vs {}",
            m.comb[l],
            comb[l]
        );
    }
    for j in 0..4 {
        let col: f32 = (0..4).map(|i| m.comb[4 * i + j]).sum();
        assert!((col - 1.0).abs() < 2e-6, "{what}: column {j} sums to {col}");
    }
    m
}

fn weight_stats(name: &str, m: &Fp8Matrix) {
    let nan = m.data.iter().filter(|&&b| b & 0x7F == 0x7F).count();
    assert_eq!(nan, 0, "{name}: {nan} NaN codes");
    assert!(
        m.scale_inv.iter().all(|s| s.is_finite() && *s > 0.0),
        "{name}: non-positive scale"
    );
    let (lo, hi) = m
        .scale_inv
        .iter()
        .fold((f32::MAX, 0f32), |(a, b), &s| (a.min(s), b.max(s)));
    let pow2 = m
        .scale_inv
        .iter()
        .filter(|s| s.to_bits() & 0x7F_FFFF == 0)
        .count();
    eprintln!(
        "{name}: {} x {}, scales {lo:.3e} .. {hi:.3e}, {pow2} of {} powers of two",
        m.rows,
        m.cols,
        m.scale_inv.len()
    );
}

#[test]
fn layer0_boundaries_on_embeddings() {
    let Some(ck) = common::checkpoint() else {
        return;
    };
    let Some(p) = common::layer_params(&ck, 0) else {
        return;
    };
    let Some(emb) = common::embeddings(&ck, &common::TOKENS) else {
        return;
    };
    for (t, &id) in common::TOKENS.iter().enumerate() {
        let x = &emb[t * 4096..(t + 1) * 4096];
        let streams = mhc::broadcast(x);
        let sf = bf16::widen(&streams);
        let m = check_mix(&sf, &p.attn_hc, &format!("layer 0 attn, token {id}"));
        if t == 0 {
            eprintln!("layer 0 attn token {id}: pre {:?} post {:?}", m.pre, m.post);
            eprintln!("  comb {:?}", m.comb);
        }
        // Identical streams collapse to (sum of pre) * x, to one BF16 step.
        let c = mhc::collapse(&sf, &m.pre, 4096);
        let total: f32 = m.pre.iter().sum();
        for d in 0..4096 {
            let want = bf16::from_f32(total * bf16::to_f32(x[d]));
            assert!(bf16::ulp_distance(c[d], want) <= 1, "token {id} d={d}");
        }
        // The FFN boundary after an attention output of zero: streams are the comb mix.
        let mid = mhc::expand(&vec![0.0; 4096], &sf, &m.post, &m.comb, 4096);
        check_mix(
            &bf16::widen(&mid),
            &p.ffn_hc,
            &format!("layer 0 ffn, token {id}"),
        );
    }
}

#[test]
fn layer0_dense_mlp_on_real_weights() {
    let Some(ck) = common::checkpoint() else {
        return;
    };
    let Some(p) = common::layer_params(&ck, 0) else {
        return;
    };
    let Some(m) = common::mlp(&ck, 0, false) else {
        return;
    };
    let Some(emb) = common::embeddings(&ck, &common::TOKENS[..1]) else {
        return;
    };
    weight_stats("layer 0 gate+up", &m.gate_up);
    weight_stats("layer 0 down", &m.down);
    assert_eq!((m.inter, m.hidden()), (12288, 4096));
    let x = mhc::boundary(&mhc::broadcast(&emb), &p.attn_hc, &p.input_norm, RMS_EPS).normed;
    let mut outs = Vec::new();
    for scheme in [ActScheme::Bf16, ActScheme::Fp8Dynamic128] {
        let got = mlp::mlp(&x, 1, &m, scheme);
        // Each projection, in the kernels' order, against exact block products.
        let gu = mlp::fp8_linear(&x, 1, &m.gate_up, scheme, mlp::decode_ksplit(24576, 4096));
        let (ex, mag) = mlp::fp8_linear_f64(&x, 1, &m.gate_up, scheme);
        for i in 0..gu.len() {
            assert!(
                (gu[i] as f64 - ex[i]).abs() <= 1e-6 * mag[i] + 1e-30,
                "{scheme:?} gate_up [{i}]"
            );
        }
        let dn = mlp::fp8_linear(&got.act, 1, &m.down, scheme, 1);
        let (ex, mag) = mlp::fp8_linear_f64(&got.act, 1, &m.down, scheme);
        for i in 0..dn.len() {
            assert!(
                (dn[i] as f64 - ex[i]).abs() <= 1e-6 * mag[i] + 1e-30,
                "{scheme:?} down [{i}]"
            );
        }
        outs.push(bf16::widen(&got.out));
    }
    // W8A8 against W8A16: the activation quantization's effect on this layer's output.
    let (a, b) = (&outs[0], &outs[1]);
    let num: f64 = a.iter().zip(b).map(|(x, y)| ((x - y) as f64).powi(2)).sum();
    let den: f64 = a.iter().map(|x| (*x as f64).powi(2)).sum();
    eprintln!(
        "layer 0 dense MLP: relative RMS difference W8A8 vs W8A16 = {:.3e}",
        (num / den).sqrt()
    );
    assert!((num / den).sqrt() < 0.1);
}

#[test]
fn layers3_4_router_and_shared_expert() {
    let Some(ck) = common::checkpoint() else {
        return;
    };
    let Some(emb) = common::embeddings(&ck, &common::TOKENS) else {
        return;
    };
    for layer in [3usize, 4] {
        let Some(p) = common::layer_params(&ck, layer) else {
            return;
        };
        let Some((rw, rb)) = common::router(&ck, layer) else {
            return;
        };
        let Some(shared) = common::mlp(&ck, layer, true) else {
            return;
        };
        assert_eq!((shared.inter, shared.hidden()), (2048, 4096));
        weight_stats(&format!("layer {layer} shared gate+up"), &shared.gate_up);
        let (lo, hi) = rb
            .iter()
            .fold((f32::MAX, f32::MIN), |(a, b), &v| (a.min(v), b.max(v)));
        eprintln!("layer {layer} e_score_correction_bias: {lo:.4} .. {hi:.4}");
        // Plausible FFN inputs: the FFN boundary of broadcast embeddings.
        let xs: Vec<u16> = (0..common::TOKENS.len())
            .flat_map(|t| {
                mhc::boundary(
                    &mhc::broadcast(&emb[t * 4096..(t + 1) * 4096]),
                    &p.ffn_hc,
                    &p.post_attn_norm,
                    RMS_EPS,
                )
                .normed
            })
            .collect();
        let routes = router::route(&xs, common::TOKENS.len(), &rw, &rb, TOP_K, ROUTED_SCALE);
        for (t, r) in routes.iter().enumerate() {
            let mut ids = r.ids.clone();
            ids.sort_unstable();
            ids.dedup();
            assert_eq!(ids.len(), TOP_K);
            let s: f32 = r.weights.iter().sum();
            assert!((s - ROUTED_SCALE).abs() < 1e-5, "weights sum {s}");
            // Logits against f64 dots; weights against the unbiased scores.
            let x = bf16::widen(&xs[t * 4096..(t + 1) * 4096]);
            let lg = router::logits_row(&x, &rw, 288);
            for e in 0..288 {
                let w = &rw[e * 4096..(e + 1) * 4096];
                let exact: f64 = (0..4096)
                    .map(|k| x[k] as f64 * bf16::to_f32(w[k]) as f64)
                    .sum();
                let mag: f64 = (0..4096)
                    .map(|k| (x[k] as f64 * bf16::to_f32(w[k]) as f64).abs())
                    .sum();
                assert!(
                    (lg[e] as f64 - exact).abs() <= 2e-7 * mag,
                    "layer {layer} expert {e}"
                );
            }
            let scores: Vec<f32> = r.ids.iter().map(|&e| sigmoid(lg[e as usize])).collect();
            let sum: f32 = scores.iter().sum();
            for (k, &w) in r.weights.iter().enumerate() {
                assert!((w - scores[k] / sum * ROUTED_SCALE).abs() <= 1e-6 * w);
            }
            if t == 0 {
                eprintln!(
                    "layer {layer} token {}: experts {:?} weights {:?}",
                    common::TOKENS[t],
                    r.ids,
                    r.weights
                );
            }
        }
        // The shared expert's projections against exact block products (one token).
        let x = &xs[..4096];
        let got = mlp::mlp(x, 1, &shared, ActScheme::Fp8Dynamic128);
        let (ex, mag) = mlp::fp8_linear_f64(&got.act, 1, &shared.down, ActScheme::Fp8Dynamic128);
        let dn = mlp::fp8_linear(&got.act, 1, &shared.down, ActScheme::Fp8Dynamic128, 1);
        for i in 0..4096 {
            assert!(
                (dn[i] as f64 - ex[i]).abs() <= 1e-6 * mag[i] + 1e-30,
                "layer {layer} shared down [{i}]"
            );
            assert_eq!(got.out[i], bf16::from_f32(dn[i]));
        }
    }
}

#[test]
fn final_norm_is_present() {
    let Some(ck) = common::checkpoint() else {
        return;
    };
    let Ok((w, s)) = ck.read_bf16(&format!("{}norm.weight", common::PREFIX)) else {
        eprintln!("skip: final norm not present");
        return;
    };
    assert_eq!(s, vec![4096]);
    assert!(w.iter().any(|&v| v != 0));
}
