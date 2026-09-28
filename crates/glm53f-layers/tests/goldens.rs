//! Checks against the reference oracle's golden fixtures (`oracle/README.md`), when present.
//!
//! - Every tensor set's entries are verified against their sizes and SHA-256 digests.
//! - **FP32 sets** (`layerNN-prefill`, `layerNN-decode`; the reference run in FP32 on FP32-
//!   dequantised weights): the f32 functions here fed the reference's own FP32 inputs:
//!   mHC weights, collapse and expand, the router's logits, choice and weights, and the dense
//!   MLP or shared expert recomputed in f64 from the checkpoint. Tight tolerances; these pin
//!   the formulas and the weight layouts.
//! - **Native set** (`native`, the reference in BF16 on the same inputs cast to BF16): this
//!   crate's BF16 stream flow (boundaries, rounding points, expansion), fed the reference's
//!   attention and FFN outputs, against its BF16 output streams.
//! - **Head set**: the unweighted mean of the streams and the final RMSNorm.
//!
//! The numeric checks also need the checkpoint (`GLM53F_CHECKPOINT_DIR`, see tests/common).
//! Anything absent is reported and skipped.

mod common;

use glm53f_layers::bf16;
use glm53f_layers::fp8::Fp8Matrix;
use glm53f_layers::layer::LayerParams;
use glm53f_layers::math::silu;
use glm53f_layers::mhc::{self, HcMix, HcParams};
use glm53f_layers::mlp::{self, Fp8Mlp, SWIGLU_LIMIT};
use glm53f_layers::norm::RMS_EPS;
use glm53f_layers::router::{self, ROUTED_SCALE, TOP_K};
use glm53f_layers::testkit::goldens::{self, GoldenSet};
use glm53f_layers::testkit::repo_root;

const D: usize = 4096;

/// The oracle's golden sets: `GLM53F_GOLDENS` if set (as in the other crates' tests), else
/// `oracle/goldens` in this repository. A set whose payload files are absent (they are
/// regenerated, not committed) is skipped with a note rather than failing a test.
fn sets() -> Vec<GoldenSet> {
    let root = std::env::var_os("GLM53F_GOLDENS")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| repo_root().join("oracle/goldens"));
    goldens::discover(&root)
        .iter()
        .filter_map(|d| GoldenSet::load(d).ok())
        .filter(|s| !s.entries.is_empty())
        .filter(|s| {
            let complete = s.entries.iter().all(|e| s.dir.join(&e.file).is_file());
            if !complete {
                eprintln!("skip: {} has no payloads (regenerate with the oracle)", s.dir.display());
            }
            complete
        })
        .collect()
}

fn set_named<'a>(sets: &'a [GoldenSet], name: &str) -> Option<&'a GoldenSet> {
    sets.iter()
        .find(|s| s.dir.file_name().is_some_and(|n| n == name))
}

struct G<'a> {
    set: &'a GoldenSet,
    prefix: String,
}

impl G<'_> {
    fn entry(&self, key: &str) -> Option<&goldens::GoldenEntry> {
        self.set.get(&format!("{}{key}", self.prefix))
    }
    fn f32(&self, key: &str) -> Option<Vec<f32>> {
        self.entry(key).map(|e| self.set.read_f32(e).unwrap())
    }
    fn bf16(&self, key: &str) -> Option<Vec<u16>> {
        self.entry(key).map(|e| self.set.read_bf16(e).unwrap())
    }
    fn ids(&self, key: &str) -> Option<Vec<i64>> {
        self.entry(key).map(|e| self.set.read_i64(e).unwrap())
    }
}

fn close(what: &str, got: &[f32], want: &[f32], rel: f32, abs: f32) {
    assert_eq!(got.len(), want.len(), "{what}: length");
    let mut worst = 0f32;
    for (i, (&g, &w)) in got.iter().zip(want).enumerate() {
        let d = (g - w).abs();
        assert!(d <= rel * w.abs() + abs, "{what}[{i}]: {g} vs {w}");
        worst = worst.max(d / (w.abs() + abs));
    }
    eprintln!("  {what}: ok, worst {worst:.2e} of the tolerance scale");
}

/// Relative RMS difference.
fn rel_rms(got: &[f32], want: &[f32]) -> f64 {
    let num: f64 = got
        .iter()
        .zip(want)
        .map(|(a, b)| ((a - b) as f64).powi(2))
        .sum();
    let den: f64 = want.iter().map(|b| (*b as f64).powi(2)).sum();
    (num / den.max(1e-300)).sqrt()
}

fn mixes(streams: &[f32], hc: &HcParams) -> Vec<HcMix> {
    streams
        .chunks_exact(4 * D)
        .map(|s| mhc::mix(s, hc, RMS_EPS))
        .collect()
}

fn check_mix(g: &G<'_>, site: &str, m: &[HcMix]) {
    for (part, n) in [("pre", 4usize), ("post", 4), ("comb", 16)] {
        let Some(want) = g.f32(&format!("{site}.{part}")) else {
            eprintln!("  skip {site}.{part}: absent");
            continue;
        };
        let got: Vec<f32> = m
            .iter()
            .flat_map(|x| match part {
                "pre" => x.pre.to_vec(),
                "post" => x.post.to_vec(),
                _ => x.comb.to_vec(),
            })
            .collect();
        assert_eq!(got.len(), want.len() / n * n);
        close(
            &format!("{}{site}.{part}", g.prefix),
            &got,
            &want,
            2e-5,
            1e-7,
        );
    }
}

/// f32 collapse without the BF16 rounding (the FP32 reference's).
fn collapse_f32(streams: &[f32], pre: &[f32; 4]) -> Vec<f32> {
    (0..D)
        .map(|d| (0..4).fold(0f32, |a, j| a + pre[j] * streams[j * D + d]))
        .collect()
}

/// f32 expansion without BF16 rounding.
fn expand_f32(h: &[f32], res: &[f32], m: &HcMix) -> Vec<f32> {
    let mut out = vec![0f32; 4 * D];
    for i in 0..4 {
        for d in 0..D {
            let mix: f32 = (0..4).fold(0f32, |a, j| a + m.comb[4 * j + i] * res[j * D + d]);
            out[i * D + d] = m.post[i] * h[d] + mix;
        }
    }
    out
}

/// The MLP in f64 on FP32-dequantised weights (`Fp8Dequantize._dequantize_one`: one f32
/// product per element), as the FP32 reference computes it.
fn mlp_fp32(x: &[f32], m: &Fp8Mlp) -> Vec<f32> {
    let dq = |w: &Fp8Matrix| w.dequantize();
    let (gu, dn) = (dq(&m.gate_up), dq(&m.down));
    let (h, i) = (m.hidden(), m.inter);
    let lin = |x: &[f32], w: &[f32], n: usize, k: usize| -> Vec<f32> {
        (0..n)
            .map(|o| {
                (0..k)
                    .map(|kk| x[kk] as f64 * w[o * k + kk] as f64)
                    .sum::<f64>() as f32
            })
            .collect()
    };
    let y = lin(x, &gu, 2 * i, h);
    let act: Vec<f32> = (0..i)
        .map(|j| {
            let g = y[j].min(SWIGLU_LIMIT);
            let u = y[i + j].clamp(-SWIGLU_LIMIT, SWIGLU_LIMIT);
            silu(g) * u
        })
        .collect();
    lin(&act, &dn, h, i)
}

fn fp32_layer(
    g: &G<'_>,
    layer: usize,
    p: &LayerParams,
    ck: &glm53f_layers::testkit::safetensors::Checkpoint,
) {
    let Some(s_in) = g.f32("in_streams") else {
        eprintln!("  skip: {}in_streams absent", g.prefix);
        return;
    };
    let rows = s_in.len() / (4 * D);
    eprintln!(
        "layer {layer} {}: {rows} tokens",
        g.prefix.trim_end_matches('.')
    );
    let a = mixes(&s_in, &p.attn_hc);
    check_mix(g, "attn_hc", &a);
    if let Some(want) = g.f32("attn_hc.collapsed") {
        let got: Vec<f32> = (0..rows)
            .flat_map(|t| collapse_f32(&s_in[t * 4 * D..(t + 1) * 4 * D], &a[t].pre))
            .collect();
        close(
            &format!("{}attn_hc.collapsed", g.prefix),
            &got,
            &want,
            1e-5,
            1e-6,
        );
    }
    if let (Some(attn_out), Some(want)) = (g.f32("attn_out"), g.f32("mid_streams")) {
        let got: Vec<f32> = (0..rows)
            .flat_map(|t| {
                expand_f32(
                    &attn_out[t * D..(t + 1) * D],
                    &s_in[t * 4 * D..(t + 1) * 4 * D],
                    &a[t],
                )
            })
            .collect();
        close(&format!("{}mid_streams", g.prefix), &got, &want, 1e-5, 1e-6);
    }
    let Some(mid) = g.f32("mid_streams") else {
        return;
    };
    let f = mixes(&mid, &p.ffn_hc);
    check_mix(g, "ffn_hc", &f);
    let Some(x) = g.f32("ffn_norm") else {
        eprintln!("  skip FFN checks: {}ffn_norm absent", g.prefix);
        return;
    };
    // The FFN.
    let dense = layer < mlp::DENSE_LAYERS;
    let (m, key) = if dense {
        (common::mlp(ck, layer, false), "mlp_out")
    } else {
        (common::mlp(ck, layer, true), "moe.shared_out")
    };
    if let (Some(m), Some(want)) = (m, g.f32(key)) {
        let got: Vec<f32> = (0..rows)
            .flat_map(|t| mlp_fp32(&x[t * D..(t + 1) * D], &m))
            .collect();
        let e = rel_rms(&got, &want);
        eprintln!(
            "  {}{key}: relative RMS {e:.2e} (f64 on FP32-dequantised weights)",
            g.prefix
        );
        assert!(e < 1e-5, "{key}: relative RMS {e:.2e}");
    }
    if !dense {
        if let Some((rw, rb)) = common::router(ck, layer) {
            let logits: Vec<Vec<f32>> = (0..rows)
                .map(|t| router::logits_row(&x[t * D..(t + 1) * D], &rw, 288))
                .collect();
            if let Some(want) = g.f32("moe.router_logits") {
                let got: Vec<f32> = logits.iter().flatten().copied().collect();
                close(
                    &format!("{}moe.router_logits", g.prefix),
                    &got,
                    &want,
                    1e-4,
                    1e-5,
                );
            }
            if let (Some(ids), Some(w)) = (
                g.ids("moe.topk_ids_sorted"),
                g.f32("moe.topk_weights_sorted"),
            ) {
                let mut same = 0;
                for (t, lg) in logits.iter().enumerate() {
                    let r = router::select_row(lg, &rb, TOP_K, ROUTED_SCALE);
                    let mut got: Vec<(i64, f32)> = r
                        .ids
                        .iter()
                        .zip(&r.weights)
                        .map(|(&e, &v)| (e as i64, v))
                        .collect();
                    got.sort_by_key(|v| v.0);
                    if got
                        .iter()
                        .map(|v| v.0)
                        .eq(ids[t * TOP_K..(t + 1) * TOP_K].iter().copied())
                    {
                        same += 1;
                        for (k, gv) in got.iter().enumerate() {
                            let wv = w[t * TOP_K + k];
                            assert!(
                                (gv.1 - wv).abs() <= 1e-5 * wv,
                                "token {t} expert {}: {} vs {wv}",
                                gv.0,
                                gv.1
                            );
                        }
                    }
                }
                eprintln!(
                    "  {}moe.topk: {same} of {rows} tokens choose the same 8 experts",
                    g.prefix
                );
                assert!(
                    same * 50 >= rows * 49,
                    "router choices differ on {} of {rows} tokens",
                    rows - same
                );
            }
        }
    }
    if let (Some(ffn_out), Some(want)) = (g.f32("mlp_out"), g.f32("out_streams")) {
        let got: Vec<f32> = (0..rows)
            .flat_map(|t| {
                expand_f32(
                    &ffn_out[t * D..(t + 1) * D],
                    &mid[t * 4 * D..(t + 1) * 4 * D],
                    &f[t],
                )
            })
            .collect();
        close(&format!("{}out_streams", g.prefix), &got, &want, 1e-5, 1e-6);
    }
}

#[test]
fn golden_digests() {
    let sets = sets();
    if sets.is_empty() {
        eprintln!("skip: no tensor golden sets under oracle/goldens");
        return;
    }
    for s in &sets {
        for e in &s.entries {
            s.read(e).unwrap_or_else(|err| panic!("{err}"));
        }
        eprintln!("{}: {} entries verified", s.dir.display(), s.entries.len());
    }
}

#[test]
fn golden_fp32_layers() {
    let sets = sets();
    let Some(ck) = common::checkpoint() else {
        return;
    };
    let mut any = false;
    for layer in [0usize, 3, 4] {
        let Some(p) = common::layer_params(&ck, layer) else {
            return;
        };
        for phase in ["prefill", "decode"] {
            let Some(set) = set_named(&sets, &format!("layer{layer:02}-{phase}")) else {
                continue;
            };
            any = true;
            fp32_layer(
                &G {
                    set,
                    prefix: format!("{phase}."),
                },
                layer,
                &p,
                &ck,
            );
        }
    }
    if !any {
        eprintln!("skip: no layerNN-prefill/decode sets");
    }
}

/// How close (in f32 ulps) an f32 mHC weight must be to a BF16 rounding boundary for the
/// reference's own f32 value, from a different projection sum order and `exp`, to round to
/// the other side. The flips seen on the native set are 3, 29 and 41 ulps from the boundary.
const FLIP_ULPS: f32 = 256.0;

/// The BF16 value on the other side of the nearest BF16 rounding boundary, when `x` lies
/// within `ulps` f32 ulps of it.
fn other_rounding(x: f32, ulps: f32) -> Option<f32> {
    let b = bf16::round(x);
    if x == b || b == 0.0 || !x.is_normal() {
        return None;
    }
    let u = b.to_bits() >> 16;
    let up = (x > b) == (b > 0.0);
    let n = f32::from_bits((if up { u + 1 } else { u - 1 }) << 16);
    let boundary = (b + n) * 0.5; // exact: b and n are BF16 neighbours
    let ulp = f32::from_bits(x.abs().to_bits() & 0x7f80_0000) * f32::EPSILON;
    ((x - boundary).abs() <= ulps * ulp).then_some(n)
}

/// One token's stream flow (`[4][D]` streams) with given sublayer outputs, optionally with
/// one mHC weight replaced: (boundary 0 = attention, 1 = FFN; index 0..4 = `post`, 4..20 =
/// `comb`; the value).
fn token_flow(
    s: &[f32],
    attn_out: &[f32],
    mlp_out: &[f32],
    p: &LayerParams,
    flip: Option<(usize, usize, f32)>,
) -> Vec<u16> {
    let set = |m: &mut HcMix, b: usize| {
        if let Some((fb, i, v)) = flip {
            if fb == b {
                if i < 4 {
                    m.post[i] = v;
                } else {
                    m.comb[i - 4] = v;
                }
            }
        }
    };
    let mut a = mhc::mix(s, &p.attn_hc, RMS_EPS);
    set(&mut a, 0);
    let mid = bf16::widen(&mhc::expand(attn_out, s, &a.post, &a.comb, D));
    let mut f = mhc::mix(&mid, &p.ffn_hc, RMS_EPS);
    set(&mut f, 1);
    mhc::expand(mlp_out, &mid, &f.post, &f.comb, D)
}

/// The one mHC weight whose BF16 rounding, flipped across a boundary it lies within
/// [`FLIP_ULPS`] of, makes this token's output streams equal `want`.
fn explain_by_one_flip(
    s: &[f32],
    attn_out: &[f32],
    mlp_out: &[f32],
    p: &LayerParams,
    want: &[u16],
) -> Option<String> {
    let a = mhc::mix(s, &p.attn_hc, RMS_EPS);
    let mid = bf16::widen(&mhc::expand(attn_out, s, &a.post, &a.comb, D));
    let f = mhc::mix(&mid, &p.ffn_hc, RMS_EPS);
    for (b, m) in [(0usize, &a), (1, &f)] {
        let w: Vec<f32> = m.post.iter().chain(&m.comb).copied().collect();
        for (i, &x) in w.iter().enumerate() {
            let Some(v) = other_rounding(x, FLIP_ULPS) else {
                continue;
            };
            if token_flow(s, attn_out, mlp_out, p, Some((b, i, v))) == want {
                let site = if b == 0 { "attn" } else { "ffn" };
                let (name, k) = if i < 4 { ("post", i) } else { ("comb", i - 4) };
                return Some(format!(
                    "{site} {name}[{k}] = {x:e} rounds to {v:e} instead of {:e}",
                    bf16::round(x)
                ));
            }
        }
    }
    None
}

#[test]
fn golden_native_stream_flow() {
    // The reference's BF16 run of each recorded layer, fed bf16(FP32 in_streams): this crate's
    // BF16 boundaries and expansions, with the reference's attention and FFN outputs, must
    // give its output streams bit for bit, except on tokens where one f32 mHC weight (`post`
    // or `comb`, which the reference rounds to BF16 before the expansion) lies so close to a
    // BF16 rounding boundary that the reference's value, from its own projection sum order
    // and `exp`, rounds to the other side. Such a token must match exactly with that one
    // weight rounded the other way.
    let sets = sets();
    let Some(native) = set_named(&sets, "native") else {
        eprintln!("skip: no native set");
        return;
    };
    let Some(ck) = common::checkpoint() else {
        return;
    };
    for layer in [0usize, 3, 4] {
        let Some(p) = common::layer_params(&ck, layer) else {
            return;
        };
        let mut s_in: Vec<f32> = Vec::new();
        for phase in ["prefill", "decode"] {
            let Some(set) = set_named(&sets, &format!("layer{layer:02}-{phase}")) else {
                continue;
            };
            if let Some(v) = (G {
                set,
                prefix: format!("{phase}."),
            })
            .f32("in_streams")
            {
                s_in.extend(v);
            }
        }
        let n = G {
            set: native,
            prefix: format!("L{layer:02}."),
        };
        let (Some(attn_out), Some(mlp_out), Some(want)) =
            (n.bf16("attn_out"), n.bf16("mlp_out"), n.bf16("out_streams"))
        else {
            eprintln!("skip: native layer {layer} entries absent");
            continue;
        };
        let rows = want.len() / (4 * D);
        if s_in.len() != rows * 4 * D {
            eprintln!("skip: layer {layer} FP32 in_streams ({} rows) do not match the native rows ({rows})", s_in.len() / (4 * D));
            continue;
        }
        let streams: Vec<u16> = bf16::narrow(&s_in);
        let mut attn = |_: &[u16]| attn_out.clone();
        let mut ffn = |_: &[u16]| mlp_out.clone();
        let tr =
            glm53f_layers::layer::decoder_layer(&streams, rows, &p, &mut attn, &mut ffn, RMS_EPS);
        let (count, worst) = glm53f_layers::testkit::bf16_mismatch(&tr.out, &want);
        eprintln!(
            "layer {layer} native out_streams: {count} of {} differ, worst {worst} ulp",
            tr.out.len()
        );
        let per = 4 * D;
        let mut flipped = 0;
        for t in 0..rows {
            let (got, want) = (
                &tr.out[t * per..(t + 1) * per],
                &want[t * per..(t + 1) * per],
            );
            if got == want {
                continue;
            }
            let s = bf16::widen(&streams[t * per..(t + 1) * per]);
            let why = explain_by_one_flip(
                &s,
                &bf16::widen(&attn_out[t * D..(t + 1) * D]),
                &bf16::widen(&mlp_out[t * D..(t + 1) * D]),
                &p,
                want,
            );
            let n = got.iter().zip(want).filter(|(a, b)| a != b).count();
            let why = why.unwrap_or_else(|| {
                panic!("layer {layer} token {t}: {n} values differ and no single weight rounding explains them")
            });
            eprintln!("  token {t}: {n} values differ; all match when {why}");
            flipped += 1;
        }
        assert!(
            flipped * 4 <= rows,
            "layer {layer}: {flipped} of {rows} tokens need a flipped rounding"
        );
        // The MoE layers' routing on this crate's FFN input, against the native choice.
        if layer >= mlp::DENSE_LAYERS {
            if let (Some((rw, rb)), Some(ids)) =
                (common::router(&ck, layer), n.ids("moe.topk_ids_sorted"))
            {
                let x: Vec<u16> = tr
                    .ffn
                    .iter()
                    .flat_map(|b| b.normed.iter().copied())
                    .collect();
                let routes = router::route(&x, rows, &rw, &rb, TOP_K, ROUTED_SCALE);
                let same = routes
                    .iter()
                    .enumerate()
                    .filter(|(t, r)| {
                        let mut s: Vec<i64> = r.ids.iter().map(|&e| e as i64).collect();
                        s.sort_unstable();
                        s[..] == ids[t * TOP_K..(t + 1) * TOP_K]
                    })
                    .count();
                eprintln!("layer {layer} native routing: {same} of {rows} tokens choose the same 8 experts");
            }
        }
    }
}

#[test]
fn golden_head() {
    let sets = sets();
    let Some(head) = set_named(&sets, "head") else {
        eprintln!("skip: no head set");
        return;
    };
    let g = G {
        set: head,
        prefix: String::new(),
    };
    let (Some(s), Some(want)) = (g.f32("head.in_streams"), g.f32("head.collapsed")) else {
        eprintln!("skip: head entries absent");
        return;
    };
    let rows = want.len() / D;
    let got: Vec<f32> = (0..rows)
        .flat_map(|t| {
            let s = &s[t * 4 * D..(t + 1) * 4 * D];
            (0..D).map(move |d| (((s[d] + s[D + d]) + s[2 * D + d]) + s[3 * D + d]) * 0.25)
        })
        .collect();
    close("head.collapsed", &got, &want, 1e-6, 1e-7);
    // The BF16 path rounds the mean once.
    let b = mhc::head_mean(&s[..4 * D], D);
    for d in 0..D {
        assert!((bf16::to_f32(b[d]) - want[d]).abs() <= want[d].abs() * 2f32.powi(-8) + 1e-30);
    }
    let Some(ck) = common::checkpoint() else {
        return;
    };
    if let (Ok((w, _)), Some(want)) = (
        ck.read_bf16(&format!("{}norm.weight", common::PREFIX)),
        g.f32("head.norm"),
    ) {
        let got: Vec<f32> = got
            .chunks_exact(D)
            .flat_map(|x| {
                let ms: f64 = x.iter().map(|&v| (v as f64).powi(2)).sum::<f64>() / D as f64;
                let r = (1.0 / (ms + RMS_EPS as f64).sqrt()) as f32;
                x.iter()
                    .zip(&w)
                    .map(move |(&v, &wv)| bf16::to_f32(wv) * (v * r))
                    .collect::<Vec<_>>()
            })
            .collect();
        close("head.norm", &got, &want, 1e-5, 1e-6);
    }
}
