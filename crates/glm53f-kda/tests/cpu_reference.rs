//! The CPU reference against hand-checked values, invariants of the recurrence, its own two
//! formulations, and the chunked form. No GPU needed.

use glm53f_kda::cpu::{self, ConvRounding, LayerParams, Rows};
use glm53f_kda::{bf16, chunked, synth, DK, DV, WINDOW};

fn bits(xs: &[f32]) -> Vec<u32> {
    xs.iter().map(|x| x.to_bits()).collect()
}

/// max |a - b| / max(max |b|, tiny).
fn rel_err(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    let d = a
        .iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max);
    let m = b.iter().map(|y| y.abs()).fold(0.0f32, f32::max);
    d / m.max(1e-30)
}

/// Largest bfloat16 ulp distance and the number of differing elements.
fn bf16_diff(a: &[f32], b: &[f32]) -> (u32, usize) {
    let mut worst = 0;
    let mut n = 0;
    for (x, y) in a.iter().zip(b) {
        let d = bf16::ulp_distance(bf16::from_f32(*x), bf16::from_f32(*y));
        worst = worst.max(d);
        n += (d != 0) as usize;
    }
    (worst, n)
}

#[test]
fn forget_gate_and_beta_hand_values() {
    // sigmoid(0) = 1/2 exactly, so the log decay is lower / 2.
    assert_eq!(cpu::log_decay(0.0, 0.0, 0.0, -5.0), -2.5);
    let d = cpu::decay(0.0, 0.0, 1.0, -5.0);
    assert!((d as f64 - (-2.5f64).exp()).abs() < 1e-8, "{d}");
    // Saturated gates: the strongest decay is exp(lower), the weakest exactly 1.
    assert!((cpu::decay(1e4, 0.0, 1.0, -5.0) as f64 - (-5.0f64).exp()).abs() < 1e-9);
    assert_eq!(cpu::decay(-1e4, 0.0, 1.0, -5.0), 1.0);
    // dt_bias shifts the argument; exp(A_log) scales it.
    assert_eq!(cpu::log_decay(1.0, -1.0, 3.0, -5.0), -2.5);
    assert_eq!(
        cpu::decay(2.0, -1.0, 0.5, -5.0),
        cpu::decay(0.5, 0.0, 1.0, -5.0)
    );
    // beta = bf16(sigmoid(b)).
    assert_eq!(cpu::beta(0.0), 0.5);
    assert_eq!(cpu::beta(20.0), 1.0);
    assert_eq!(cpu::beta(-200.0), 0.0);
    assert_eq!(cpu::beta(1.0), bf16::round(0.731_058_6));
}

/// A one-head layer whose conv weights are `w` on every channel.
fn one_head(w: [f32; 4]) -> LayerParams {
    let mut p = synth::layer(1, 1);
    for c in 0..p.channels() {
        p.conv_w[c * 4..c * 4 + 4].copy_from_slice(&w);
    }
    p
}

#[test]
fn conv_hand_values() {
    let p = one_head([1.0, 2.0, 3.0, 4.0]);
    let c = p.channels();
    // Window rows 1, 2, 3 (oldest first), new rows 4, 5 on every channel.
    let mut conv = vec![0.0; WINDOW * c];
    for (j, row) in conv.chunks_exact_mut(c).enumerate() {
        row.fill(j as f32 + 1.0);
    }
    let mut rows = synth::rows(1, 2, 2);
    rows.qkv[..c].fill(4.0);
    rows.qkv[c..].fill(5.0);
    // Row 0: 1*1 + 2*2 + 3*3 + 4*4 = 30; row 1: 1*2 + 2*3 + 3*4 + 4*5 = 40. SiLU(x) rounds to x
    // in bfloat16 for both.
    for (r, want) in [(0, 30.0), (1, 40.0)] {
        let (q, k, v) = cpu::conv_qkv(&p, &conv, &rows, r, 0, ConvRounding::Fused);
        assert!(q.iter().chain(&k).chain(&v).all(|&x| x == want), "row {r}");
    }
    // A negative sum is squashed: SiLU(-30) = -30 / (1 + e^30).
    rows.qkv[..c].fill(-11.0);
    let (q, _, _) = cpu::conv_qkv(&p, &conv, &rows, 0, 0, ConvRounding::Fused);
    let want = bf16::round(-30.0 / (1.0 + 30f32.exp()));
    assert_eq!(q[0], want);
    assert!(q[0] < 0.0 && q[0] > -1e-11);
    // The shift keeps rows [keep, keep + 3) of [window; rows].
    let rows2 = synth::rows(1, 2, 3);
    let w1 = cpu::conv_shift(&conv, &rows2, 1);
    assert_eq!(&w1[..c], &conv[c..2 * c]);
    assert_eq!(&w1[c..2 * c], &conv[2 * c..]);
    assert_eq!(&w1[2 * c..], &rows2.qkv[..c]);
    let w2 = cpu::conv_shift(&conv, &rows2, 2);
    assert_eq!(&w2[..c], &conv[2 * c..]);
    assert_eq!(&w2[c..], &rows2.qkv[..]);
    assert_eq!(cpu::conv_shift(&conv, &rows2, 0), conv);
}

#[test]
fn warp_sum_is_the_butterfly() {
    // Order matters: a sequential sum loses the 1 against 1e8; the butterfly first cancels
    // lanes 0 and 16, then adds the 1.
    let mut lanes = [0.0f32; 32];
    lanes[0] = 1e8;
    lanes[16] = -1e8;
    lanes[1] = 1.0;
    let sequential = lanes.iter().fold(0.0f32, |a, &x| a + x);
    assert_eq!(sequential, 0.0);
    assert_eq!(cpu::warp_sum(&lanes), 1.0);
    // Lanes 0 and 1 meet only in the last step, after each absorbed its own partner.
    let mut lanes = [0.0f32; 32];
    lanes[0] = 1.0;
    lanes[1] = 1e8;
    lanes[17] = -1e8;
    assert_eq!(cpu::warp_sum(&lanes), 1.0);
    let ints: [f32; 32] = std::array::from_fn(|i| i as f32);
    assert_eq!(cpu::warp_sum(&ints), 496.0);
}

#[test]
fn l2norm_and_gated_norm_hand_values() {
    let mut x = [0.0f32; DK];
    x[5] = 3.0;
    x[9] = 4.0;
    let y = cpu::l2norm(&x, None);
    let n = (25.0f32 + 1e-6).sqrt();
    assert_eq!(y[5], 3.0 * (1.0 / n));
    assert_eq!(y[9], 4.0 * (1.0 / n));
    let yq = cpu::l2norm(&x, Some(cpu::q_scale()));
    assert_eq!(yq[5], y[5] * (1.0 / 128f32.sqrt()));
    // The reference divides instead; the two agree to an ulp.
    let yl = cpu::literal::l2norm(&x);
    assert!((yl[9] - y[9]).abs() <= f32::EPSILON * y[9]);
    // Gated RMSNorm of a constant vector: w * c / sqrt(c^2 + eps) * sigmoid(gate).
    let yc = [0.5f32; DV];
    let w = [2.0f32; DV];
    let gate = [0.0f32; DV];
    let o = cpu::gated_rmsnorm(&yc, &w, &gate, 1e-5);
    let want = bf16::round(2.0 * (0.5 * (1.0 / (0.25f32 + 1e-5).sqrt())) * 0.5);
    assert!(o.iter().all(|&v| v == want));
    assert!((want - 1.0).abs() < 1e-2);
}

#[test]
fn delta_rule_writes_the_value_under_its_key() {
    // No decay, beta 1, a unit key: afterwards reading with that key returns v, and every
    // other key column is unchanged.
    let mut rng = synth::Rng::new(9);
    let s0 = rng.fill(DV * DK, -0.5, 0.5);
    let mut s = s0.clone();
    let mut k = [0.0f32; DK];
    k[3] = 1.0;
    let g = [1.0f32; DK];
    let v = rng.fill(DV, -2.0, 2.0);
    cpu::update(&mut s, &k, &g, &v, 1.0);
    let read = cpu::read_out(&s, &k);
    for row in 0..DV {
        assert!(
            (read[row] - v[row]).abs() <= 4.0 * f32::EPSILON * 2.5,
            "row {row}: {} vs {}",
            read[row],
            v[row]
        );
        for c in (0..DK).filter(|&c| c != 3) {
            assert_eq!(s[row * DK + c], s0[row * DK + c]);
        }
    }
}

#[test]
fn gates_saturated() {
    let heads = 2;
    let mut p = synth::layer(heads, 4);
    p.a_log.fill(0.0);
    p.dt_bias.fill(0.0);
    let conv = synth::conv_window(heads, 4);
    let s0 = synth::state(heads, 4, 0.5);
    // Strongest decay, beta 0: every row scales the state by exp(-5), nothing is written.
    let mut rows = synth::rows(heads, 20, 4);
    rows.a.fill(1e4);
    rows.b.fill(-200.0);
    let r = cpu::chain(&p, &conv, &s0, &rows, ConvRounding::Fused);
    assert!(r.saves.beta.iter().all(|&b| b == 0.0));
    let e5 = cpu::decay(1e4, 0.0, 1.0, -5.0);
    assert!(r.saves.g.iter().all(|&g| g == e5));
    let mut expect = s0.clone();
    for _ in 0..20 {
        for x in &mut expect {
            *x *= e5;
        }
    }
    // Compared as values: entries that underflow may end as +0 or -0.
    assert_eq!(r.state, expect);
    let max = r.state.iter().fold(0.0f32, |m, x| m.max(x.abs()));
    assert!(
        max < 1e-40,
        "state should have decayed to (sub)normal dust, max {max}"
    );
    // No decay, beta 0: the state is unchanged.
    rows.a.fill(-1e4);
    let r = cpu::chain(&p, &conv, &s0, &rows, ConvRounding::Fused);
    assert!(r.saves.g.iter().all(|&g| g == 1.0));
    assert_eq!(r.state, s0);
}

#[test]
fn replay_of_a_prefix_equals_the_shorter_chain_bitwise() {
    let heads = 4;
    let p = synth::layer(heads, 11);
    let conv = synth::conv_window(heads, 11);
    let s0 = synth::state(heads, 11, 0.5);
    let rows = synth::rows(heads, 8, 11);
    let full = cpu::chain(&p, &conv, &s0, &rows, ConvRounding::Fused);
    assert_eq!(bits(&cpu::replay(&s0, &full.saves, 8)), bits(&full.state));
    for keep in 0..=8 {
        let short = cpu::chain(&p, &conv, &s0, &rows.slice(0, keep), ConvRounding::Fused);
        assert_eq!(
            bits(&cpu::replay(&s0, &full.saves, keep)),
            bits(&short.state),
            "keep {keep}"
        );
        let n = keep * heads * DV;
        assert_eq!(
            bits(&full.out[..n]),
            bits(&short.out),
            "outputs of the first {keep} rows"
        );
    }
}

#[test]
fn serial_steps_equal_one_window_bitwise() {
    let heads = 3;
    let p = synth::layer(heads, 12);
    let conv0 = synth::conv_window(heads, 12);
    let s0 = synth::state(heads, 12, 0.5);
    let rows = synth::rows(heads, 8, 12);
    let window = cpu::chain(&p, &conv0, &s0, &rows, ConvRounding::Fused);
    let (mut conv, mut s) = (conv0.clone(), s0.clone());
    for r in 0..8 {
        let row = rows.slice(r, 1);
        let step = cpu::chain(&p, &conv, &s, &row, ConvRounding::Fused);
        let n = heads * DV;
        assert_eq!(
            bits(&step.out),
            bits(&window.out[r * n..(r + 1) * n]),
            "row {r}"
        );
        conv = cpu::conv_shift(&conv, &row, 1);
        s = step.state;
    }
    assert_eq!(bits(&s), bits(&window.state));
    assert_eq!(conv, cpu::conv_shift(&conv0, &rows, 8));
}

#[test]
fn kernel_order_agrees_with_the_literal_formulation() {
    let heads = 4;
    let p = synth::layer(heads, 13);
    let conv = synth::conv_window(heads, 13);
    let s0 = synth::state(heads, 13, 0.5);
    let rows = synth::rows(heads, 8, 13);
    let a = cpu::chain(&p, &conv, &s0, &rows, ConvRounding::Fused);
    let b = cpu::literal::chain(&p, &conv, &s0, &rows, ConvRounding::Fused);
    // The same conv and gates; only the order of the sums and the norms' division differ.
    assert_eq!(bits(&a.saves.v), bits(&b.saves.v));
    assert_eq!(bits(&a.saves.beta), bits(&b.saves.beta));
    let e_state = rel_err(&a.state, &b.state);
    let e_k = rel_err(&a.saves.k, &b.saves.k);
    let e_g = rel_err(&a.saves.g, &b.saves.g);
    let (ulp_y, n_y) = bf16_diff(&a.y, &b.y);
    let (ulp_o, n_o) = bf16_diff(&a.out, &b.out);
    eprintln!(
        "kernel order vs literal: state {e_state:.2e}, k {e_k:.2e}, g {e_g:.2e}; read-out {n_y}/{} differ (max {ulp_y} ulp); output {n_o}/{} differ (max {ulp_o} ulp)",
        a.y.len(),
        a.out.len()
    );
    assert!(e_state < 1e-5, "state {e_state}");
    assert!(e_k < 1e-6 && e_g < 1e-6);
    assert!(ulp_y <= 1 && ulp_o <= 1);
    assert!(n_y * 100 <= a.y.len() && n_o * 100 <= a.out.len());
}

#[test]
fn conv_roundings_differ_slightly() {
    // Rounding the conv sum to bfloat16 before the SiLU moves it by up to half a bfloat16 ulp
    // of the sum. Where SiLU is flat (large negative sums) that is many ulps of the tiny output
    // but a negligible absolute change.
    let heads = 2;
    let p = synth::layer(heads, 14);
    let conv = synth::conv_window(heads, 14);
    let rows = synth::rows(heads, 4, 14);
    let (mut differ, mut total, mut worst_ulp, mut worst_abs) = (0, 0, 0, 0.0f32);
    for r in 0..4 {
        for h in 0..heads {
            let (q1, k1, v1) = cpu::conv_qkv(&p, &conv, &rows, r, h, ConvRounding::Fused);
            let (q2, k2, v2) = cpu::conv_qkv(&p, &conv, &rows, r, h, ConvRounding::Unfused);
            let (a, b) = ([q1, k1, v1].concat(), [q2, k2, v2].concat());
            let (u, n) = bf16_diff(&a, &b);
            for (x, y) in a.iter().zip(&b) {
                let d = (x - y).abs();
                worst_abs = worst_abs.max(d);
                // Half an ulp of the sum through SiLU (slope <= 1.1), plus the outputs' own roundings.
                assert!(d <= x.abs().max(y.abs()).max(1.0) / 64.0, "{x} vs {y}");
            }
            worst_ulp = worst_ulp.max(u);
            differ += n;
            total += 3 * DK;
        }
    }
    eprintln!(
        "fused vs unfused conv rounding: {differ}/{total} conv outputs differ (max {worst_ulp} bf16 ulp, max |diff| {worst_abs:.2e})"
    );
    // About a quarter of the outputs differ: the two paths are not interchangeable bit for bit.
    assert!(differ > 0 && differ * 2 < total);
}

#[test]
fn state_layout_transpose() {
    let s = synth::state(2, 15, 1.0);
    let t = cpu::transpose_state(&s);
    assert_eq!(t[DV + 3], s[3 * DK + 1]);
    assert_eq!(t[DK * DV + 7 * DV + 2], s[DK * DV + 2 * DK + 7]);
    assert_eq!(cpu::transpose_state(&t), s);
}

/// q, k, v (conv outputs), log decays and betas for `t` rows of `heads` heads.
type CoreInputs = (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>);

fn core_inputs(heads: usize, t: usize, seed: u64) -> CoreInputs {
    let mut rng = synth::Rng::new(seed);
    let q = rng.fill_bf16(t * heads * DK, -1.0, 1.0);
    let k = rng.fill_bf16(t * heads * DK, -1.0, 1.0);
    let v = rng.fill_bf16(t * heads * DV, -1.0, 1.0);
    let g: Vec<f32> = (0..t * heads * DK)
        .map(|_| cpu::log_decay(rng.uniform(-4.0, 4.0), 0.0, 0.0, -5.0))
        .collect();
    let beta: Vec<f32> = (0..t * heads)
        .map(|_| cpu::beta(rng.uniform(-4.0, 4.0)))
        .collect();
    (q, k, v, g, beta)
}

#[test]
fn chunked_form_matches_the_recurrence() {
    let heads = 2;
    for (t, chunk) in [(1, 64), (7, 64), (64, 64), (65, 64), (150, 64), (40, 16)] {
        let (q, k, v, g, beta) = core_inputs(heads, t, 16 + t as u64);
        let s0 = synth::state(heads, t as u64, 0.5);
        // Recurrence, the reference's formulation, state [H][DK][DV].
        let mut s_rec = s0.clone();
        let mut y_rec = vec![0.0f32; t * heads * DV];
        for r in 0..t {
            for h in 0..heads {
                let i = r * heads + h;
                let o = cpu::literal::step(
                    &mut s_rec[h * DK * DV..(h + 1) * DK * DV],
                    &q[i * DK..(i + 1) * DK],
                    &k[i * DK..(i + 1) * DK],
                    &v[i * DV..(i + 1) * DV],
                    &g[i * DK..(i + 1) * DK],
                    beta[i],
                );
                y_rec[i * DV..(i + 1) * DV].copy_from_slice(&o);
            }
        }
        let mut s_chunk = s0.clone();
        let y_chunk = chunked::layer(heads, &q, &k, &v, &g, &beta, &mut s_chunk, chunk);
        let e_y = rel_err(&y_chunk, &y_rec);
        let e_s = rel_err(&s_chunk, &s_rec);
        let (ulp, n) = bf16_diff(&y_chunk, &y_rec);
        eprintln!(
            "chunked vs recurrent, T={t}, chunk {chunk}: read-out {e_y:.2e}, state {e_s:.2e}, bf16 read-out {n}/{} differ (max {ulp} ulp)",
            y_rec.len()
        );
        assert!(
            e_y < 2e-5 && e_s < 2e-5,
            "T={t}: read-out {e_y}, state {e_s}"
        );
    }
}

#[test]
fn rows_slice() {
    let rows: Rows = synth::rows(2, 5, 17);
    let s = rows.slice(1, 3);
    assert_eq!(s.rows, 3);
    assert_eq!(s.b, rows.b[2..8]);
    assert_eq!(s.slice(2, 1).gate, rows.slice(3, 1).gate);
}
