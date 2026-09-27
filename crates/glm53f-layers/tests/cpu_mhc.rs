//! mHC reference: invariants, and agreement with a direct f64 transcription of
//! `Glm5NextTextHyperConnection.forward` (norm before the projection, no fixed order).

use glm53f_layers::bf16;
use glm53f_layers::mhc::{self, HcParams, HC_EPS, HC_MULT, HC_PROJ};
use glm53f_layers::norm::RMS_EPS;
use glm53f_layers::testkit::Rng;

const D: usize = 256;

fn params(rng: &mut Rng, hidden: usize, logit_scale: f32) -> HcParams {
    let fn_ = rng.bf16_vec(HC_PROJ * HC_MULT * hidden, 0.02);
    let base: Vec<f32> = rng.f32_vec(HC_PROJ, 0.5);
    let scale = [0.8 * logit_scale, 1.1 * logit_scale, 1.3 * logit_scale];
    HcParams::new(hidden, fn_, &base, &scale)
}

/// The reference, transcribed in f64: flatten, RMS-normalize, project, then the formulas.
fn reference_f64(streams: &[f32], p: &HcParams) -> ([f64; 4], [f64; 4], [f64; 16]) {
    let n = streams.len();
    let ms: f64 = streams
        .iter()
        .map(|&x| (x as f64) * (x as f64))
        .sum::<f64>()
        / n as f64;
    let r = 1.0 / (ms + RMS_EPS as f64).sqrt();
    let proj: Vec<f64> = (0..HC_PROJ)
        .map(|q| {
            (0..n)
                .map(|k| streams[k] as f64 * r * bf16::to_f32(p.fn_[q * n + k]) as f64)
                .sum()
        })
        .collect();
    let sig = |x: f64| 1.0 / (1.0 + (-x).exp());
    let (s, b) = (p.scale, p.base);
    let mut pre = [0f64; 4];
    let mut post = [0f64; 4];
    for j in 0..4 {
        pre[j] = sig(proj[j] * s[0] as f64 + b[j] as f64) + HC_EPS as f64;
        post[j] = 2.0 * sig(proj[4 + j] * s[1] as f64 + b[4 + j] as f64);
    }
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
    let colnorm = |c: &mut [f64; 16]| {
        for j in 0..4 {
            let s: f64 = (0..4).map(|i| c[4 * i + j]).sum::<f64>() + HC_EPS as f64;
            for i in 0..4 {
                c[4 * i + j] /= s;
            }
        }
    };
    let rownorm = |c: &mut [f64; 16]| {
        for i in 0..4 {
            let s: f64 = (0..4).map(|j| c[4 * i + j]).sum::<f64>() + HC_EPS as f64;
            for j in 0..4 {
                c[4 * i + j] /= s;
            }
        }
    };
    colnorm(&mut c);
    for _ in 1..20 {
        rownorm(&mut c);
        colnorm(&mut c);
    }
    (pre, post, c)
}

#[test]
fn mix_matches_f64_reference() {
    let mut rng = Rng::new(1);
    for trial in 0..8 {
        let p = params(&mut rng, D, 1.0 + trial as f32);
        let streams = bf16::widen(&rng.bf16_vec(HC_MULT * D, 1.0 + trial as f32));
        let got = mhc::mix(&streams, &p, RMS_EPS);
        let (pre, post, comb) = reference_f64(&streams, &p);
        for j in 0..4 {
            assert!(
                (got.pre[j] as f64 - pre[j]).abs() <= 2e-6 * pre[j].abs() + 1e-7,
                "pre[{j}] {} vs {}",
                got.pre[j],
                pre[j]
            );
            assert!(
                (got.post[j] as f64 - post[j]).abs() <= 2e-6 * post[j].abs() + 1e-7,
                "post[{j}]"
            );
        }
        for l in 0..16 {
            let rel = (got.comb[l] as f64 - comb[l]).abs() / comb[l].abs().max(1e-6);
            assert!(
                rel < 2e-5,
                "trial {trial} comb[{l}] {} vs {} (rel {rel})",
                got.comb[l],
                comb[l]
            );
        }
    }
}

/// Largest |row sum - 1| of a comb matrix.
fn row_error(c: &[f32; 16]) -> f32 {
    (0..4)
        .map(|i| ((0..4).map(|j| c[4 * i + j]).sum::<f32>() - 1.0).abs())
        .fold(0.0, f32::max)
}

#[test]
fn comb_is_doubly_stochastic_after_20_iterations() {
    // The last step divides each column by (sum + 1e-6), so columns always sum to 1 within
    // a few ulp. Rows converge to within ~1e-6 for moderate logits (sigma 1: median 1e-6,
    // worst of 2,000 draws 2.2e-4); strongly peaked matrices (sigma >= 1.5) keep up to a
    // few percent of row error after the reference's 20 iterations, which is the
    // reference's behaviour, not rounding (see the next test).
    let mut rng = Rng::new(2);
    let mut rows: Vec<f32> = Vec::new();
    for _ in 0..256 {
        let mut logits = [0f32; 16];
        for l in &mut logits {
            *l = rng.normal();
        }
        let c = mhc::sinkhorn(&logits);
        for j in 0..4 {
            let col: f32 = (0..4).map(|i| c[4 * i + j]).sum();
            assert!((col - 1.0).abs() < 2e-6, "column {j} sums to {col}");
        }
        assert!(c.iter().all(|&v| v > 0.0 && v < 1.0));
        rows.push(row_error(&c));
    }
    rows.sort_by(|a, b| a.partial_cmp(b).unwrap());
    assert!(rows[128] < 2e-6, "median row error {}", rows[128]);
    assert!(rows[255] < 5e-4, "worst row error {}", rows[255]);
}

#[test]
fn peaked_comb_error_is_the_iteration_count_not_rounding() {
    // For peaked logits, an f64 Sinkhorn with the same 20 iterations leaves the same row
    // error; the f32 kernel order agrees with it entry by entry.
    let mut rng = Rng::new(8);
    for _ in 0..64 {
        let mut logits = [0f32; 16];
        for l in &mut logits {
            *l = rng.normal() * 3.0;
        }
        let c = mhc::sinkhorn(&logits);
        let mut r = [0f64; 16];
        for i in 0..4 {
            let m = (0..4)
                .map(|j| logits[4 * i + j] as f64)
                .fold(f64::MIN, f64::max);
            let e: Vec<f64> = (0..4)
                .map(|j| (logits[4 * i + j] as f64 - m).exp())
                .collect();
            let s: f64 = e.iter().sum();
            for j in 0..4 {
                r[4 * i + j] = e[j] / s + 1e-6;
            }
        }
        let norm = |r: &mut [f64; 16], by_row: bool| {
            for a in 0..4 {
                let idx = |b: usize| if by_row { 4 * a + b } else { 4 * b + a };
                let s: f64 = (0..4).map(|b| r[idx(b)]).sum::<f64>() + 1e-6;
                for b in 0..4 {
                    r[idx(b)] /= s;
                }
            }
        };
        norm(&mut r, false);
        for _ in 1..20 {
            norm(&mut r, true);
            norm(&mut r, false);
        }
        for l in 0..16 {
            assert!(
                (c[l] as f64 - r[l]).abs() <= 1e-5 * r[l] + 1e-9,
                "entry {l}: {} vs {}",
                c[l],
                r[l]
            );
        }
        let rf: [f32; 16] = std::array::from_fn(|l| r[l] as f32);
        assert!((row_error(&c) - row_error(&rf)).abs() < 1e-5);
    }
}

#[test]
fn pre_and_post_ranges() {
    let mut rng = Rng::new(3);
    for _ in 0..16 {
        let p = params(&mut rng, D, 20.0);
        let s = bf16::widen(&rng.bf16_vec(HC_MULT * D, 3.0));
        let m = mhc::mix(&s, &p, RMS_EPS);
        assert!(m.pre.iter().all(|&x| (HC_EPS..=1.0 + HC_EPS).contains(&x)));
        assert!(m.post.iter().all(|&x| (0.0..=2.0).contains(&x)));
    }
}

#[test]
fn collapse_of_identical_streams() {
    let mut rng = Rng::new(4);
    let x = rng.bf16_vec(D, 1.0);
    let streams = bf16::widen(&mhc::broadcast(&x));
    // Equal weights of 1/4 give the stream back exactly.
    assert_eq!(mhc::collapse(&streams, &[0.25; 4], D), x);
    // Any weights give (sum of weights) * x, to within one BF16 step.
    let pre = [0.3, 0.1 + HC_EPS, 0.72, 0.05];
    let got = mhc::collapse(&streams, &pre, D);
    let total: f32 = pre.iter().sum();
    for d in 0..D {
        let want = bf16::from_f32(total * bf16::to_f32(x[d]));
        assert!(bf16::ulp_distance(got[d], want) <= 1, "d={d}");
    }
    // The final mean of identical streams is the stream.
    assert_eq!(mhc::head_mean(&streams, D), x);
}

#[test]
fn identical_streams_project_like_the_folded_weight() {
    // Layer 0 starts from 4 copies of the embedding: the projection equals x times the
    // per-position sum of fn over the streams, and the RMS is the row's RMS.
    let mut rng = Rng::new(5);
    let p = params(&mut rng, D, 1.0);
    let x = rng.bf16_vec(D, 0.7);
    let streams = bf16::widen(&mhc::broadcast(&x));
    let parts = mhc::project(&streams, &p);
    for q in 0..HC_PROJ {
        let got: f64 = parts.iter().map(|s| s[q] as f64).sum();
        let want: f64 = (0..D)
            .map(|d| {
                let f: f64 = (0..4)
                    .map(|st| bf16::to_f32(p.fn_[q * 4 * D + st * D + d]) as f64)
                    .sum();
                bf16::to_f32(x[d]) as f64 * f
            })
            .sum();
        let mag: f64 = (0..4 * D)
            .map(|k| (streams[k] * bf16::to_f32(p.fn_[q * 4 * D + k])).abs() as f64)
            .sum();
        assert!(
            (got - want).abs() <= 1e-6 * mag + 1e-9,
            "projection {q}: {got} vs {want}"
        );
    }
    let sq: f64 = parts.iter().map(|s| s[HC_PROJ] as f64).sum();
    let want_sq: f64 = 4.0
        * x.iter()
            .map(|&v| (bf16::to_f32(v) as f64).powi(2))
            .sum::<f64>();
    assert!((sq - want_sq).abs() <= 1e-6 * want_sq);
}

#[test]
fn expand_edge_cases() {
    let mut rng = Rng::new(6);
    let res = rng.bf16_vec(HC_MULT * D, 1.0);
    let h = rng.bf16_vec(D, 1.0);
    let resf = bf16::widen(&res);
    let hf = bf16::widen(&h);
    let identity: [f32; 16] = std::array::from_fn(|l| if l / 4 == l % 4 { 1.0 } else { 0.0 });
    // No expansion, identity mix: the residual passes through.
    assert_eq!(mhc::expand(&hf, &resf, &[0.0; 4], &identity, D), res);
    // Unit expansion, no mix: every stream is the output.
    assert_eq!(
        mhc::expand(&hf, &resf, &[1.0; 4], &[0.0; 16], D),
        mhc::broadcast(&h)
    );
    // comb[j][i] routes source j to destination i: a permutation moves stream 0 to 2.
    let mut perm = [0f32; 16];
    for (j, i) in [(0, 2), (1, 0), (2, 3), (3, 1)] {
        perm[4 * j + i] = 1.0;
    }
    let out = mhc::expand(&hf, &resf, &[0.0; 4], &perm, D);
    assert_eq!(&out[2 * D..3 * D], &res[..D]);
    assert_eq!(&out[..D], &res[D..2 * D]);
    // Two-part outputs add with one BF16 rounding.
    let h2 = rng.bf16_vec(D, 1.0);
    let sum = mhc::block_output(&h, Some(&h2));
    for d in 0..D {
        assert_eq!(
            sum[d],
            bf16::round(bf16::to_f32(h[d]) + bf16::to_f32(h2[d]))
        );
    }
}

#[test]
fn boundary_normalizes_the_collapse() {
    let mut rng = Rng::new(7);
    let p = params(&mut rng, D, 1.0);
    let s = rng.bf16_vec(HC_MULT * D, 1.5);
    let w = rng.bf16_vec(D, 0.3);
    let b = mhc::boundary(&s, &p, &w, RMS_EPS);
    let x = bf16::widen(&b.collapsed);
    let ms: f64 = x.iter().map(|&v| (v as f64).powi(2)).sum::<f64>() / D as f64;
    let r = 1.0 / (ms + RMS_EPS as f64).sqrt();
    for d in 0..D {
        let want = bf16::to_f32(w[d]) as f64 * bf16::round((x[d] as f64 * r) as f32) as f64;
        let got = bf16::to_f32(b.normed[d]) as f64;
        assert!(
            (got - want).abs() <= want.abs() * 2f64.powi(-7) + 1e-30,
            "d={d}: {got} vs {want}"
        );
    }
}
