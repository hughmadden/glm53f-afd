//! Kernels against the CPU reference on the official checkpoint's tensors (`--features
//! cuda`, `GLM53F_CHECKPOINT_DIR`; see tests/common). Real scale grids, real mHC and router
//! weights, real embedding rows: bitwise for every kernel but the tensor-core prefill GEMM.
#![cfg(feature = "cuda")]

mod common;

use glm53f_layers::cuda::{DeviceBuffer, Stream};
use glm53f_layers::fp8::ActScheme;
use glm53f_layers::mhc::{self, HcParams, PARTIAL};
use glm53f_layers::mlp::{self, Fp8Mlp};
use glm53f_layers::norm::{self, RMS_EPS};
use glm53f_layers::ops::{self, FinishOut, GemmInput, Promotion};
use glm53f_layers::router::{self, ROUTED_SCALE, TOP_K};

fn up<T: Copy>(v: &[T]) -> DeviceBuffer {
    DeviceBuffer::from_slice(v).unwrap()
}
fn zeros(bytes: usize) -> DeviceBuffer {
    DeviceBuffer::zeroed(bytes).unwrap()
}

/// One boundary on the device: partials, weights, collapse, normed input.
fn gpu_boundary(
    streams: &[u16],
    rows: usize,
    p: &HcParams,
    nw: &[u16],
    s: &Stream,
) -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<u16>) {
    let hidden = p.hidden;
    let slices = hidden / 128;
    let (dst, dfn, db, dsc, dnw) = (up(streams), up(&p.fn_), up(&p.base), up(&p.scale), up(nw));
    let parts = zeros(rows * slices * PARTIAL * 4);
    ops::hc_project(&dst, None, Some((&dfn, &parts)), rows, hidden, s).unwrap();
    let (pre, post, comb, normed) = (
        zeros(rows * 16),
        zeros(rows * 16),
        zeros(rows * 64),
        zeros(rows * hidden * 2),
    );
    let o = FinishOut {
        pre: Some(&pre),
        post: Some(&post),
        comb: Some(&comb),
        normed: Some(&normed),
        ..Default::default()
    };
    ops::hc_finish(&parts, &db, &dsc, &dst, Some(&dnw), &o, rows, hidden, s).unwrap();
    (
        pre.download(rows * 4).unwrap(),
        post.download(rows * 4).unwrap(),
        comb.download(rows * 16).unwrap(),
        normed.download(rows * hidden).unwrap(),
    )
}

/// An MLP on the device in the decode kernels (W8A8 or W8A16), BF16 output.
fn gpu_mlp(x: &[u16], rows: usize, m: &Fp8Mlp, scheme: ActScheme, s: &Stream) -> Vec<u16> {
    let (h, i) = (m.hidden(), m.inter);
    let (gw, gs, dw, ds) = (
        up(&m.gate_up.data),
        up(&m.gate_up.scale_inv),
        up(&m.down.data),
        up(&m.down.scale_inv),
    );
    let dx = up(x);
    let partials = zeros(8 * rows * 2 * i * 4);
    let gate_up = zeros(rows * 2 * i * 2);
    let out = zeros(rows * h * 2);
    match scheme {
        ActScheme::Bf16 => {
            ops::fp8_linear_decode(
                &GemmInput::Bf16(&dx),
                &gw,
                &gs,
                rows,
                2 * i,
                h,
                &partials,
                &gate_up,
                s,
            )
            .unwrap();
            let act = zeros(rows * i * 2);
            ops::swiglu(&gate_up, Some(&act), None, rows, i, s).unwrap();
            ops::fp8_linear_decode(
                &GemmInput::Bf16(&act),
                &dw,
                &ds,
                rows,
                h,
                i,
                &partials,
                &out,
                s,
            )
            .unwrap();
        }
        ActScheme::Fp8Dynamic128 => {
            let (xq, xs) = (zeros(rows * h), zeros(rows * h / 128 * 4));
            ops::act_quant(&dx, &xq, &xs, rows, h, s).unwrap();
            ops::fp8_linear_decode(
                &GemmInput::Fp8 {
                    q: &xq,
                    scales: &xs,
                },
                &gw,
                &gs,
                rows,
                2 * i,
                h,
                &partials,
                &gate_up,
                s,
            )
            .unwrap();
            let (aq, as_) = (zeros(rows * i), zeros(rows * i / 128 * 4));
            ops::swiglu(&gate_up, None, Some((&aq, &as_)), rows, i, s).unwrap();
            ops::fp8_linear_decode(
                &GemmInput::Fp8 {
                    q: &aq,
                    scales: &as_,
                },
                &dw,
                &ds,
                rows,
                h,
                i,
                &partials,
                &out,
                s,
            )
            .unwrap();
        }
    }
    out.download(rows * h).unwrap()
}

#[test]
fn layer0_on_real_weights() {
    let Some(ck) = common::checkpoint() else {
        return;
    };
    let Some(p) = common::layer_params(&ck, 0) else {
        return;
    };
    let Some(m) = common::mlp(&ck, 0, false) else {
        return;
    };
    let Some(emb) = common::embeddings(&ck, &common::TOKENS) else {
        return;
    };
    let s = Stream::new().unwrap();
    let rows = common::TOKENS.len();
    let streams: Vec<u16> = emb.chunks_exact(4096).flat_map(mhc::broadcast).collect();
    let (pre, post, comb, normed) = gpu_boundary(&streams, rows, &p.attn_hc, &p.input_norm, &s);
    for t in 0..rows {
        let b = mhc::boundary(
            &streams[t * 4 * 4096..(t + 1) * 4 * 4096],
            &p.attn_hc,
            &p.input_norm,
            RMS_EPS,
        );
        for j in 0..4 {
            assert_eq!(pre[t * 4 + j].to_bits(), b.mix.pre[j].to_bits(), "pre");
            assert_eq!(post[t * 4 + j].to_bits(), b.mix.post[j].to_bits(), "post");
        }
        for l in 0..16 {
            assert_eq!(comb[t * 16 + l].to_bits(), b.mix.comb[l].to_bits(), "comb");
        }
        assert_eq!(&normed[t * 4096..(t + 1) * 4096], &b.normed[..], "normed");
    }
    // The dense MLP (12,288 wide) on those inputs: decode kernels bitwise, both schemes.
    for scheme in [ActScheme::Bf16, ActScheme::Fp8Dynamic128] {
        let got = gpu_mlp(&normed, rows, &m, scheme, &s);
        assert_eq!(got, mlp::mlp(&normed, rows, &m, scheme).out, "{scheme:?}");
    }
    // The prefill GEMM on the real gate+up weight, within the tensor-core bound.
    let (dq, ds) = (zeros(rows * 4096), zeros(rows * 32 * 4));
    ops::act_quant(&up(&normed), &dq, &ds, rows, 4096, &s).unwrap();
    let (out, out32) = (zeros(rows * 24576 * 2), zeros(rows * 24576 * 4));
    ops::fp8_gemm_prefill(
        &dq,
        &ds,
        &up(&m.gate_up.data),
        &up(&m.gate_up.scale_inv),
        rows,
        24576,
        4096,
        Promotion::Block128,
        &out,
        Some(&out32),
        &s,
    )
    .unwrap();
    let got: Vec<f32> = out32.download(rows * 24576).unwrap();
    let (ex, mag) = mlp::fp8_linear_f64(&normed, rows, &m.gate_up, ActScheme::Fp8Dynamic128);
    for i in 0..got.len() {
        assert!(
            (got[i] as f64 - ex[i]).abs() <= 2f64.powi(-10) * mag[i],
            "prefill [{i}]"
        );
    }
}

#[test]
fn moe_layers_on_real_weights() {
    let Some(ck) = common::checkpoint() else {
        return;
    };
    let Some(emb) = common::embeddings(&ck, &common::TOKENS) else {
        return;
    };
    let s = Stream::new().unwrap();
    let rows = common::TOKENS.len();
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
        let streams: Vec<u16> = emb.chunks_exact(4096).flat_map(mhc::broadcast).collect();
        let (_, _, _, x) = gpu_boundary(&streams, rows, &p.ffn_hc, &p.post_attn_norm, &s);
        let cpu_x: Vec<u16> = (0..rows)
            .flat_map(|t| {
                mhc::boundary(
                    &streams[t * 4 * 4096..(t + 1) * 4 * 4096],
                    &p.ffn_hc,
                    &p.post_attn_norm,
                    RMS_EPS,
                )
                .normed
            })
            .collect();
        assert_eq!(x, cpu_x, "layer {layer} FFN input");
        // Router, bitwise.
        let (dx, dw, db) = (up(&x), up(&rw), up(&rb));
        let (lg, ids, wt) = (
            zeros(rows * 288 * 4),
            zeros(rows * TOP_K * 4),
            zeros(rows * TOP_K * 4),
        );
        ops::router_logits(&dx, &dw, &lg, rows, 288, 4096, &s).unwrap();
        ops::router_select(&lg, &db, &ids, &wt, rows, 288, TOP_K, ROUTED_SCALE, &s).unwrap();
        let ids: Vec<i32> = ids.download(rows * TOP_K).unwrap();
        let wt: Vec<f32> = wt.download(rows * TOP_K).unwrap();
        let routes = router::route(&x, rows, &rw, &rb, TOP_K, ROUTED_SCALE);
        for (t, r) in routes.iter().enumerate() {
            let gi: Vec<u32> = ids[t * TOP_K..(t + 1) * TOP_K]
                .iter()
                .map(|&v| v as u32)
                .collect();
            assert_eq!(gi, r.ids, "layer {layer} row {t}");
            for k in 0..TOP_K {
                assert_eq!(wt[t * TOP_K + k].to_bits(), r.weights[k].to_bits());
            }
        }
        // The shared expert, both schemes, bitwise.
        for scheme in [ActScheme::Bf16, ActScheme::Fp8Dynamic128] {
            assert_eq!(
                gpu_mlp(&x, rows, &shared, scheme, &s),
                mlp::mlp(&x, rows, &shared, scheme).out,
                "layer {layer} {scheme:?}"
            );
        }
        // The final RMSNorm kernel on these rows.
        let Some(fw) = ck
            .read_bf16(&format!("{}norm.weight", common::PREFIX))
            .ok()
            .map(|v| v.0)
        else {
            continue;
        };
        let out = zeros(rows * 4096 * 2);
        ops::rmsnorm(&dx, &up(&fw), &out, rows, 4096, &s).unwrap();
        assert_eq!(
            out.download::<u16>(rows * 4096).unwrap(),
            norm::rms_norm(&x, &fw, rows, RMS_EPS)
        );
    }
}
