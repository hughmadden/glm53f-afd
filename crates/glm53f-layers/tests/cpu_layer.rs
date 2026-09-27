//! The decoder layer's stream flow and the model's ends.

use glm53f_layers::bf16;
use glm53f_layers::layer::{self, LayerParams};
use glm53f_layers::mhc::{self, HcParams, HC_MULT, HC_PROJ};
use glm53f_layers::norm::{self, RMS_EPS};
use glm53f_layers::testkit::Rng;

const D: usize = 256;

fn hc(rng: &mut Rng) -> HcParams {
    HcParams::new(
        D,
        rng.bf16_vec(HC_PROJ * HC_MULT * D, 0.02),
        &rng.f32_vec(HC_PROJ, 0.5),
        &[1.0, 1.0, 1.0],
    )
}

fn params(rng: &mut Rng) -> LayerParams {
    LayerParams {
        attn_hc: hc(rng),
        ffn_hc: hc(rng),
        input_norm: rng.bf16_vec(D, 0.5),
        post_attn_norm: rng.bf16_vec(D, 0.5),
    }
}

#[test]
fn layer_flow_composes_the_boundaries() {
    let mut rng = Rng::new(31);
    let p = params(&mut rng);
    let rows = 3;
    let emb = rng.bf16_vec(rows * D, 1.0);
    let streams = layer::embed_streams(&emb, rows);
    assert_eq!(streams.len(), rows * HC_MULT * D);
    // Sublayers: fixed pseudo-random maps of their inputs.
    let mut attn = |x: &[u16]| -> Vec<u16> {
        x.iter()
            .map(|&v| bf16::from_f32(bf16::to_f32(v) * -0.5 + 0.125))
            .collect()
    };
    let mut ffn = |x: &[u16]| -> Vec<u16> { x.iter().rev().copied().collect() };
    let tr = layer::decoder_layer(&streams, rows, &p, &mut attn, &mut ffn, RMS_EPS);

    for t in 0..rows {
        let s = &streams[t * 4 * D..(t + 1) * 4 * D];
        // Attention boundary, by hand.
        let b = mhc::boundary(s, &p.attn_hc, &p.input_norm, RMS_EPS);
        assert_eq!(tr.attn[t].mix, b.mix);
        assert_eq!(tr.attn[t].normed, b.normed);
        let a_out: Vec<u16> = b
            .normed
            .iter()
            .map(|&v| bf16::from_f32(bf16::to_f32(v) * -0.5 + 0.125))
            .collect();
        assert_eq!(&tr.attn_out[t * D..(t + 1) * D], &a_out[..]);
        let mid = mhc::expand(
            &bf16::widen(&a_out),
            &bf16::widen(s),
            &b.mix.post,
            &b.mix.comb,
            D,
        );
        assert_eq!(&tr.mid[t * 4 * D..(t + 1) * 4 * D], &mid[..]);
        let f = mhc::boundary(&mid, &p.ffn_hc, &p.post_attn_norm, RMS_EPS);
        assert_eq!(tr.ffn[t].mix, f.mix);
    }
    // The FFN saw all tokens at once (reversed across the batch).
    let f_in: Vec<u16> = tr
        .ffn
        .iter()
        .flat_map(|b| b.normed.iter().copied())
        .collect();
    let rev: Vec<u16> = f_in.iter().rev().copied().collect();
    assert_eq!(tr.ffn_out, rev);
    for t in 0..rows {
        let out = mhc::expand(
            &bf16::widen(&tr.ffn_out[t * D..(t + 1) * D]),
            &bf16::widen(&tr.mid[t * 4 * D..(t + 1) * 4 * D]),
            &tr.ffn[t].mix.post,
            &tr.ffn[t].mix.comb,
            D,
        );
        assert_eq!(&tr.out[t * 4 * D..(t + 1) * 4 * D], &out[..]);
    }
}

#[test]
fn zero_sublayers_only_mix_the_streams() {
    // With zero sublayer outputs the layer is two comb mixes; comb is doubly stochastic, so
    // the mean over streams is preserved to rounding.
    let mut rng = Rng::new(32);
    let p = params(&mut rng);
    let streams = rng.bf16_vec(4 * D, 1.0);
    let mut zero = |x: &[u16]| vec![0u16; x.len()];
    let mut zero2 = |x: &[u16]| vec![0u16; x.len()];
    let tr = layer::decoder_layer(&streams, 1, &p, &mut zero, &mut zero2, RMS_EPS);
    let before = bf16::widen(&mhc::head_mean(&bf16::widen(&streams), D));
    let after = bf16::widen(&mhc::head_mean(&bf16::widen(&tr.out), D));
    let scale: f32 = before.iter().map(|v| v.abs()).sum::<f32>() / D as f32;
    for d in 0..D {
        assert!(
            (before[d] - after[d]).abs() <= 0.02 * scale + 0.02 * before[d].abs(),
            "d={d}"
        );
    }
}

#[test]
fn final_hidden_is_the_normed_mean() {
    let mut rng = Rng::new(33);
    let rows = 2;
    let streams = rng.bf16_vec(rows * 4 * D, 1.0);
    let w = rng.bf16_vec(D, 0.4);
    let got = layer::final_hidden(&streams, rows, &w, RMS_EPS);
    for t in 0..rows {
        let mean = mhc::head_mean(&bf16::widen(&streams[t * 4 * D..(t + 1) * 4 * D]), D);
        assert_eq!(
            &got[t * D..(t + 1) * D],
            &norm::rms_norm_row(&mean, &w, RMS_EPS)[..]
        );
    }
}
