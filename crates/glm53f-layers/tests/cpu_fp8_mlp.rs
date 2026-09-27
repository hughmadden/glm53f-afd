//! FP8 block-128 weights, activation quantization, the projection orders, SwiGLU and the
//! MLPs.

use glm53f_layers::bf16;
use glm53f_layers::fp8::{self, e4m3_to_f32, f32_to_e4m3, ActScheme, Fp8Matrix};
use glm53f_layers::layer::{self, MoeParams};
use glm53f_layers::mlp::{self, Fp8Mlp};
use glm53f_layers::router;
use glm53f_layers::testkit::Rng;

#[test]
fn block_dequantization_of_known_values() {
    // 130 x 260: a 2 x 3 scale grid with ragged last blocks.
    let (rows, cols) = (130, 260);
    let mut data = vec![0u8; rows * cols];
    data[0] = 0x38; // 1.0 in block (0, 0)
    data[127 * cols + 255] = 0xB9; // -1.125 in block (0, 1)
    data[128 * cols + 256] = 0x7E; // 448 in block (1, 2)
    data[129 * cols + 259] = 0x01; // 2^-9 in block (1, 2)
    data[5 * cols + 130] = 0x08; // 2^-6 in block (0, 1)
    let scale = vec![0.5, 0.25, 3.0, 0.125, 1.5, 2.0f32.powi(-8)];
    let w = Fp8Matrix::new(rows, cols, data, scale);
    let d = w.dequantize();
    assert_eq!(d[0], 0.5);
    assert_eq!(d[127 * cols + 255], -1.125 * 0.25);
    assert_eq!(d[128 * cols + 256], 448.0 * 2.0f32.powi(-8));
    assert_eq!(d[129 * cols + 259], 2.0f32.powi(-17));
    assert_eq!(d[5 * cols + 130], 2.0f32.powi(-6) * 0.25);
    assert_eq!(d.iter().filter(|&&v| v != 0.0).count(), 5);
    assert_eq!(w.scale(129, 259), 2.0f32.powi(-8));
    assert_eq!(w.scale(0, 128), 0.25);
}

#[test]
fn activation_quantization() {
    let mut rng = Rng::new(21);
    let mut x = rng.f32_vec(3 * 256, 2.0);
    for v in &mut x[256..384] {
        *v = 0.0; // an all-zero group
    }
    x[5] = 9.0; // a clear maximum in group (0, 0)
    let q = fp8::quantize_rows(&x, 3, 256);
    assert_eq!(q.scale[0], 9.0 / 448.0);
    assert_eq!(q.q[5], 0x7E);
    assert_eq!(q.scale[2], 1.0);
    assert!(q.q[256..384].iter().all(|&b| b == 0));
    let dq = q.dequantize();
    for (i, (&a, &b)) in x.iter().zip(&dq).enumerate() {
        // Within half an E4M3 step (relative 2^-4) or the subnormal step of the group.
        let s = q.scale[(i / 256) * 2 + (i % 256) / 128];
        assert!(
            (a - b).abs() <= a.abs() / 16.0 + s * 2f32.powi(-10),
            "i={i}: {a} vs {b}"
        );
    }
}

#[test]
fn ksplit_policy_for_glm_shapes() {
    // (n, k) -> splits. Shared expert, dense MLP, DSA projections.
    for (n, k, want) in [
        (4096, 4096, 1),
        (4096, 2048, 1),
        (24576, 4096, 1),
        (4096, 12288, 1),
        (1536, 4096, 4),
        (512, 4096, 4),
        (16384, 1536, 1),
        (4096, 16384, 1),
    ] {
        assert_eq!(mlp::decode_ksplit(n, k), want, "n={n} k={k}");
    }
}

#[test]
fn decode_order_is_close_to_exact_for_both_schemes() {
    let mut rng = Rng::new(22);
    let w = rng.fp8_matrix(256, 1024);
    let x = rng.bf16_vec(3 * 1024, 1.0);
    for scheme in [ActScheme::Bf16, ActScheme::Fp8Dynamic128] {
        let (exact, mag) = mlp::fp8_linear_f64(&x, 3, &w, scheme);
        for ksplit in [1, 2, 4, 8] {
            let got = mlp::fp8_linear(&x, 3, &w, scheme, ksplit);
            for i in 0..got.len() {
                assert!(
                    (got[i] as f64 - exact[i]).abs() <= 1e-6 * mag[i] + 1e-30,
                    "{scheme:?} ksplit {ksplit} [{i}]: {} vs {}",
                    got[i],
                    exact[i]
                );
            }
        }
    }
}

#[test]
fn a_row_does_not_depend_on_its_batch() {
    let mut rng = Rng::new(23);
    let w = rng.fp8_matrix(128, 2048);
    let x = rng.bf16_vec(8 * 2048, 1.0);
    for scheme in [ActScheme::Bf16, ActScheme::Fp8Dynamic128] {
        let all = mlp::fp8_linear(&x, 8, &w, scheme, 2);
        for r in 0..8 {
            let one = mlp::fp8_linear(&x[r * 2048..(r + 1) * 2048], 1, &w, scheme, 2);
            assert_eq!(&all[r * 128..(r + 1) * 128], &one[..]);
        }
    }
}

#[test]
fn w8a16_is_the_dequantized_product() {
    // With BF16 activations the projection is x @ dequant(W)^T up to f32 summation.
    let mut rng = Rng::new(24);
    let w = rng.fp8_matrix(128, 512);
    let x = rng.bf16_vec(512, 1.0);
    let got = mlp::fp8_linear(&x, 1, &w, ActScheme::Bf16, 1);
    let dq = w.dequantize();
    for n in 0..128 {
        let want: f64 = (0..512)
            .map(|k| bf16::to_f32(x[k]) as f64 * dq[n * 512 + k] as f64)
            .sum();
        let mag: f64 = (0..512)
            .map(|k| (bf16::to_f32(x[k]) as f64 * dq[n * 512 + k] as f64).abs())
            .sum();
        assert!((got[n] as f64 - want).abs() <= 1e-6 * mag);
    }
}

#[test]
fn swiglu_clamps() {
    let s = |g: f32, u: f32| bf16::to_f32(mlp::swiglu(bf16::round(g), bf16::round(u)));
    // Gate is clamped above only.
    assert_eq!(s(20.0, 1.0), s(10.0, 1.0));
    assert!(s(-30.0, 1.0).abs() < 1e-11);
    // Up is clamped on both sides.
    assert_eq!(s(1.0, 50.0), s(1.0, 10.0));
    assert_eq!(s(1.0, -50.0), s(1.0, -10.0));
    // Two BF16 roundings: silu, then the product.
    let g = bf16::round(1.3);
    let u = bf16::round(-2.7);
    let want = bf16::round(bf16::round(glm53f_layers::math::silu(g)) * u);
    assert_eq!(s(g, u), want);
    // NaN passes through the clamps.
    assert!(s(f32::NAN, 1.0).is_nan());
}

/// The MLP as the reference writes it, with dequantized weights and f64 products; BF16
/// rounding after each projection and inside SwiGLU.
fn mlp_reference(x: &[u16], m: &Fp8Mlp) -> Vec<u16> {
    let (h, i) = (m.hidden(), m.inter);
    let gu = m.gate_up.dequantize();
    let dn = m.down.dequantize();
    let xf = bf16::widen(x);
    let gate_up: Vec<u16> = (0..2 * i)
        .map(|n| {
            bf16::from_f32(
                (0..h)
                    .map(|k| xf[k] as f64 * gu[n * h + k] as f64)
                    .sum::<f64>() as f32,
            )
        })
        .collect();
    let act = mlp::swiglu_rows(&gate_up, 1, i);
    let af = bf16::widen(&act);
    (0..h)
        .map(|n| {
            bf16::from_f32(
                (0..i)
                    .map(|k| af[k] as f64 * dn[n * i + k] as f64)
                    .sum::<f64>() as f32,
            )
        })
        .collect()
}

#[test]
fn mlp_matches_the_reference_within_rounding() {
    let mut rng = Rng::new(25);
    let (h, i) = (512, 384);
    let m = Fp8Mlp::new(
        &rng.fp8_matrix(i, h),
        &rng.fp8_matrix(i, h),
        rng.fp8_matrix(h, i),
    );
    let x = rng.bf16_vec(h, 1.0);
    let got = mlp::mlp(&x, 1, &m, ActScheme::Bf16);
    let want = mlp_reference(&x, &m);
    let (count, worst) = glm53f_layers::testkit::bf16_mismatch(&got.out, &want);
    // Different f32 summation orders flip an occasional BF16 rounding (one step) in the
    // projections; those can move a few outputs by one step more.
    assert!(
        worst <= 2 && count * 20 <= h,
        "{count} mismatches, worst {worst} ulp"
    );
}

#[test]
fn stacked_gate_up_keeps_blocks() {
    let mut rng = Rng::new(26);
    let g = rng.fp8_matrix(256, 128);
    let u = rng.fp8_matrix(256, 128);
    let m = Fp8Mlp::new(&g, &u, rng.fp8_matrix(128, 256));
    assert_eq!(m.gate_up.rows, 512);
    assert_eq!(m.gate_up.value(300, 7), u.value(44, 7));
    assert_eq!(m.gate_up.scale(300, 7), u.scale(44, 7));
    assert_eq!(m.gate_up.scale(3, 7), g.scale(3, 7));
}

#[test]
fn eager_experts_and_the_moe_sum() {
    let mut rng = Rng::new(27);
    let (h, i, rows) = (256, 128, 3);
    let experts: Vec<Fp8Mlp> = (0..4)
        .map(|_| {
            Fp8Mlp::new(
                &rng.fp8_matrix(i, h),
                &rng.fp8_matrix(i, h),
                rng.fp8_matrix(h, i),
            )
        })
        .collect();
    let x = rng.bf16_vec(rows * h, 1.0);
    let routes = vec![
        router::Route {
            ids: vec![2, 0],
            weights: vec![1.5, 1.0],
        },
        router::Route {
            ids: vec![3, 2],
            weights: vec![0.5, 2.0],
        },
        router::Route {
            ids: vec![1, 3],
            weights: vec![1.25, 1.25],
        },
    ];
    let got = mlp::routed_experts_eager(
        &x,
        rows,
        &routes,
        &mut |e| &experts[e as usize],
        ActScheme::Bf16,
    );
    // Token 1: experts 2 then 3 in ascending order, each term rounded, the sum rounded.
    let t = 1;
    let xt = &x[t * h..(t + 1) * h];
    let y2 = mlp::mlp(xt, 1, &experts[2], ActScheme::Bf16).out;
    let y3 = mlp::mlp(xt, 1, &experts[3], ActScheme::Bf16).out;
    for d in 0..h {
        let a = bf16::round(bf16::to_f32(y2[d]) * 2.0);
        let b = bf16::round(bf16::to_f32(y3[d]) * 0.5);
        assert_eq!(
            got[t * h + d],
            bf16::from_f32(bf16::round(0.0 + a) + b),
            "d={d}"
        );
    }

    // The MoE sum adds the shared expert with one rounding.
    let hidden = 256;
    let rw = rng.bf16_vec(router::EXPERTS * hidden, 0.02);
    let rb = rng.f32_vec(router::EXPERTS, 0.01);
    let shared = Fp8Mlp::new(
        &rng.fp8_matrix(i, hidden),
        &rng.fp8_matrix(i, hidden),
        rng.fp8_matrix(hidden, i),
    );
    let p = MoeParams {
        router_weight: &rw,
        router_bias: &rb,
        shared: &shared,
    };
    let routed = rng.bf16_vec(rows * hidden, 0.1);
    let out = layer::moe_ffn(
        &x,
        rows,
        &p,
        &mut |_, _| routed.clone(),
        ActScheme::Fp8Dynamic128,
    );
    assert_eq!(out.routes.len(), rows);
    for k in 0..rows * hidden {
        assert_eq!(
            out.out[k],
            bf16::from_f32(bf16::to_f32(routed[k]) + bf16::to_f32(out.shared[k]))
        );
    }
}

#[test]
fn e4m3_products_are_exact_in_f32() {
    // Every product of two E4M3 values, and of an E4M3 and a BF16 value, is exact in f32:
    // the W8A8 and W8A16 inner products only round in their sums.
    for a in (0u8..=255).step_by(3) {
        for b in (0u8..=255).step_by(5) {
            let (x, y) = (e4m3_to_f32(a), e4m3_to_f32(b));
            if x.is_nan() || y.is_nan() {
                continue;
            }
            assert_eq!((x * y) as f64, x as f64 * y as f64);
        }
    }
    let mut rng = Rng::new(28);
    for _ in 0..10000 {
        let x = bf16::round(rng.normal() * 1000.0);
        let y = e4m3_to_f32(f32_to_e4m3(rng.normal() * 100.0));
        assert_eq!((x * y) as f64, x as f64 * y as f64);
    }
}
