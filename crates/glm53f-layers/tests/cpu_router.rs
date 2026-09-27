//! Router reference: tie order, scale, bias-for-selection-only, and agreement with a
//! direct transcription of `Glm5NextTextTopkRouter.forward`.

use glm53f_layers::bf16;
use glm53f_layers::math::sigmoid;
use glm53f_layers::router::{self, EXPERTS, ROUTED_SCALE, TOP_K};
use glm53f_layers::testkit::Rng;

#[test]
fn exact_ties_go_to_the_lower_expert() {
    let logits = vec![0.25f32; EXPERTS];
    let bias = vec![0.0f32; EXPERTS];
    let r = router::select_row(&logits, &bias, TOP_K, ROUTED_SCALE);
    assert_eq!(r.ids, (0..8).collect::<Vec<u32>>());
    // Equal scores: equal weights of 2.5 / 8.
    for &w in &r.weights {
        assert!((w - ROUTED_SCALE / 8.0).abs() < 1e-6);
    }
    // A three-way tie above everything else is taken in index order.
    let mut logits = vec![-4.0f32; EXPERTS];
    logits[200] = 1.0;
    logits[17] = 1.0;
    logits[5] = 1.0;
    let r = router::select_row(&logits, &bias, TOP_K, ROUTED_SCALE);
    assert_eq!(&r.ids[..3], &[5, 17, 200]);
    // The rest tie at sigmoid(-4); the lowest free indices fill the remaining places.
    assert_eq!(&r.ids[3..], &[0, 1, 2, 3, 4]);
    // A tie created by the bias counts the same way.
    let mut bias = vec![0.0f32; EXPERTS];
    let logits = vec![0.0f32; EXPERTS];
    bias[250] = 0.25;
    bias[9] = 0.25;
    let r = router::select_row(&logits, &bias, TOP_K, ROUTED_SCALE);
    assert_eq!(&r.ids[..3], &[9, 250, 0]);
}

#[test]
fn bias_chooses_but_does_not_weigh() {
    let mut rng = Rng::new(11);
    let logits = rng.f32_vec(EXPERTS, 1.0);
    let mut bias = vec![0f32; EXPERTS];
    // Push expert 3 into the top 8 through the bias alone.
    bias[3] = 10.0;
    let r = router::select_row(&logits, &bias, TOP_K, ROUTED_SCALE);
    assert_eq!(r.ids[0], 3);
    let scores: Vec<f32> = r.ids.iter().map(|&e| sigmoid(logits[e as usize])).collect();
    let sum: f32 = scores.iter().sum();
    for (k, &w) in r.weights.iter().enumerate() {
        let want = scores[k] / (sum + 1e-20) * ROUTED_SCALE;
        assert!((w - want).abs() <= 1e-6 * want, "weight {k}: {w} vs {want}");
    }
}

#[test]
fn weights_sum_to_the_routed_scale() {
    let mut rng = Rng::new(12);
    for _ in 0..32 {
        let logits = rng.f32_vec(EXPERTS, 3.0);
        let bias = rng.f32_vec(EXPERTS, 0.05);
        let r = router::select_row(&logits, &bias, TOP_K, ROUTED_SCALE);
        let s: f32 = r.weights.iter().sum();
        assert!((s - ROUTED_SCALE).abs() < 4e-6, "sum {s}");
        let mut ids = r.ids.clone();
        ids.sort_unstable();
        ids.dedup();
        assert_eq!(ids.len(), TOP_K, "distinct experts");
    }
}

#[test]
fn matches_the_reference_transcription() {
    // f32 logits by a plain sequential dot (another order), f32 sigmoid, f32 bias add,
    // a stable sort for top-k. Away from near-ties the chosen sets must agree, and the
    // weights to f32 accuracy.
    let hidden = 1024;
    let mut rng = Rng::new(13);
    let weight = rng.bf16_vec(EXPERTS * hidden, 0.02);
    let bias = rng.f32_vec(EXPERTS, 0.01);
    let mut compared = 0;
    for _ in 0..16 {
        let x = rng.bf16_vec(hidden, 1.0);
        let xf = bf16::widen(&x);
        let got = router::select_row(
            &router::logits_row(&xf, &weight, EXPERTS),
            &bias,
            TOP_K,
            ROUTED_SCALE,
        );
        let logits: Vec<f32> = (0..EXPERTS)
            .map(|e| {
                (0..hidden).fold(0f32, |a, k| {
                    a + xf[k] * bf16::to_f32(weight[e * hidden + k])
                })
            })
            .collect();
        let scores: Vec<f32> = logits
            .iter()
            .map(|&l| 1.0 / (1.0 + (-(l as f64)).exp() as f32))
            .collect();
        let corrected: Vec<f32> = scores.iter().zip(&bias).map(|(s, b)| s + b).collect();
        let mut order: Vec<usize> = (0..EXPERTS).collect();
        order.sort_by(|&a, &b| corrected[b].partial_cmp(&corrected[a]).unwrap());
        // Skip rows whose 8th and 9th choices are within rounding of each other.
        if corrected[order[7]] - corrected[order[8]] < 1e-5 {
            continue;
        }
        compared += 1;
        let mut want: Vec<u32> = order[..8].iter().map(|&e| e as u32).collect();
        let mut have = got.ids.clone();
        want.sort_unstable();
        have.sort_unstable();
        assert_eq!(have, want);
        let sum: f32 = order[..8].iter().map(|&e| scores[e]).sum();
        for (k, &e) in got.ids.iter().enumerate() {
            let w = scores[e as usize] / (sum + 1e-20) * ROUTED_SCALE;
            assert!(
                (got.weights[k] - w).abs() <= 1e-5 * w,
                "weight of expert {e}"
            );
        }
    }
    assert!(
        compared >= 12,
        "too many near-ties ({compared} rows compared)"
    );
}

#[test]
fn logits_are_close_to_f64_dots() {
    let hidden = 4096;
    let mut rng = Rng::new(14);
    let weight = rng.bf16_vec(16 * hidden, 0.02);
    let x = bf16::widen(&rng.bf16_vec(hidden, 1.0));
    let got = router::logits_row(&x, &weight, 16);
    for e in 0..16 {
        let w = &weight[e * hidden..(e + 1) * hidden];
        let exact: f64 = (0..hidden)
            .map(|k| x[k] as f64 * bf16::to_f32(w[k]) as f64)
            .sum();
        let mag: f64 = (0..hidden)
            .map(|k| (x[k] as f64 * bf16::to_f32(w[k]) as f64).abs())
            .sum();
        assert!((got[e] as f64 - exact).abs() <= 2e-7 * mag, "expert {e}");
    }
}

#[test]
fn routing_a_batch_is_routing_each_row() {
    let hidden = 512;
    let mut rng = Rng::new(15);
    let weight = rng.bf16_vec(EXPERTS * hidden, 0.02);
    let bias = rng.f32_vec(EXPERTS, 0.01);
    let x = rng.bf16_vec(5 * hidden, 1.0);
    let all = router::route(&x, 5, &weight, &bias, TOP_K, ROUTED_SCALE);
    for (t, r) in all.iter().enumerate() {
        let one = router::route(
            &x[t * hidden..(t + 1) * hidden],
            1,
            &weight,
            &bias,
            TOP_K,
            ROUTED_SCALE,
        );
        assert_eq!(&one[0], r);
    }
}
