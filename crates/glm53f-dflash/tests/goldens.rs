//! The CPU reference against the oracle's DFlash2 goldens: the z-lab/dflash reference run in FP32
//! on the CPU (`oracle/golden_dflash.py`, sets `dflash-short`, `dflash-window`, `dflash-native`).
//!
//! Needs GLM53F_DFLASH_DIR (the drafter checkpoint), GLM53F_CHECKPOINT_DIR (a GLM-5.3-Flash
//! directory with `embed_tokens` and `lm_head`) and GLM53F_GOLDENS (the golden sets); each test
//! passes with a message when one is unset. Run with `--release`: the reference's GEMMs are slow
//! unoptimized.

use std::path::PathBuf;
use std::sync::OnceLock;

use glm53f_dflash::goldens::{rel_rms, Set};
use glm53f_dflash::reference::{Bf16Head, Context, Draft, DraftOptions, Reference, Trace};
use glm53f_dflash::selector::{softmax, Pick};
use glm53f_dflash::weights::{env_dir, Target, Weights};
use glm53f_dflash::{bf16, sha256, synth, Dims};

struct Env {
    w: Weights,
    head: Vec<u16>,
    target: Target,
    goldens: PathBuf,
}

fn env() -> Option<&'static Env> {
    static ENV: OnceLock<Option<Env>> = OnceLock::new();
    ENV.get_or_init(|| {
        let d = env_dir(
            "GLM53F_DFLASH_DIR",
            "the incoai/GLM-5.3-Flash-DFlash2 checkpoint directory",
        );
        let c = env_dir(
            "GLM53F_CHECKPOINT_DIR",
            "a GLM-5.3-Flash directory holding embed_tokens and lm_head",
        );
        let g = env_dir(
            "GLM53F_GOLDENS",
            "the oracle's golden directory (oracle/goldens)",
        );
        let (d, c, g) = (d?, c?, g?);
        let dims = Dims::GLM53F;
        let w = Weights::load(&d, dims).expect("drafter weights");
        let target = Target::open(&c, &dims).expect("target checkpoint");
        let head = target.lm_head().expect("lm_head");
        Some(Env {
            w,
            head,
            target,
            goldens: g,
        })
    })
    .as_ref()
}

/// A golden case's parameters, from its manifest's notes.
struct Case {
    context: usize,
    seed0: u64,
    anchor0: u32,
    seed1: u64,
    anchor1: u32,
    kept: usize,
}

fn case(set: &Set) -> Case {
    let m = set.manifest().unwrap();
    let c = m
        .get("notes")
        .and_then(|n| n.get("case"))
        .expect("notes.case");
    let u = |k: &str| {
        c.get(k)
            .and_then(|v| v.as_u64())
            .unwrap_or_else(|| panic!("notes.case.{k}"))
    };
    let kept = m
        .get("notes")
        .and_then(|n| n.get("kept_rows"))
        .and_then(|v| v.as_array())
        .expect("kept_rows")
        .len();
    Case {
        context: u("context") as usize,
        seed0: u("seed0"),
        anchor0: u("anchor0") as u32,
        seed1: u("seed1"),
        anchor1: u("anchor1") as u32,
        kept,
    }
}

/// Compare and print; fail above `tol` (relative RMS).
fn close(what: &str, got: &[f32], want: &[f32], tol: f64) -> f64 {
    let (r, m) = rel_rms(got, want);
    println!("  {what:<28} rel_rms {r:.3e}  max_abs {m:.3e}");
    assert!(r <= tol, "{what}: relative RMS {r:.3e} above {tol:.1e}");
    r
}

fn taps(seed: u64, start: usize, rows: usize) -> Vec<u16> {
    synth::taps(seed, start, rows, Dims::GLM53F.tap_width())
}

/// One draft of `anchor` at the context's end, as `dflash_generate` does it (full vocabulary for
/// the candidates, like the reference; greedy).
fn draft(
    e: &Env,
    r: &Reference<'_>,
    ctx: &Context,
    anchor: u32,
    trace: Option<&mut Trace>,
) -> Draft {
    let d = Dims::GLM53F;
    let rows = e.target.embed_rows(&[anchor, d.mask_token]).unwrap();
    let rows = bf16::decode(&rows);
    let head = Bf16Head {
        weight: &e.head,
        hidden: d.hidden,
    };
    let opts = DraftOptions {
        vocab_limit: d.vocab,
        pick: Pick::Greedy,
    };
    r.draft(
        ctx,
        anchor,
        &rows[..d.hidden],
        &rows[d.hidden..],
        &head,
        &opts,
        trace,
    )
}

fn ids(v: &[i32]) -> Vec<u32> {
    v.iter().map(|&x| x as u32).collect()
}

/// Check a draft against the golden step `p` ("s0", "s1").
fn check_draft(g: &Set, p: &str, dr: &Draft, tol: f64) {
    close(
        &format!("{p}.final"),
        &dr.hidden,
        &g.f32(&format!("{p}.final")).unwrap(),
        tol,
    );
    close(
        &format!("{p}.logits"),
        &dr.logits,
        &g.f32(&format!("{p}.logits")).unwrap(),
        tol,
    );
    close(
        &format!("{p}.topk_vals"),
        &dr.unary,
        &g.f32(&format!("{p}.topk_vals")).unwrap(),
        tol,
    );
    assert_eq!(
        dr.candidates,
        ids(&g.i32(&format!("{p}.topk_ids")).unwrap()),
        "{p}: top-16 candidates"
    );
    if g.has(&format!("{p}.hproj")) {
        close(
            &format!("{p}.hproj"),
            &dr.hproj,
            &g.f32(&format!("{p}.hproj")).unwrap(),
            tol,
        );
        close(
            &format!("{p}.path_scores"),
            &dr.walk.scores,
            &g.f32(&format!("{p}.path_scores")).unwrap(),
            tol,
        );
    }
    let path = ids(&g.i32(&format!("{p}.path")).unwrap());
    println!("  {p}.path {:?} (reference {:?})", dr.walk.tokens, path);
    assert_eq!(dr.walk.tokens, path, "{p}: the selector's path");
}

#[test]
fn synthetic_taps_match_the_oracle() {
    let Some(e) = env() else { return };
    for set in ["dflash-short", "dflash-window"] {
        let g = Set::open(&e.goldens, set).unwrap();
        let c = case(&g);
        let m = g.manifest().unwrap();
        let want = m.get("notes").and_then(|n| n.get("taps_sha256")).unwrap();
        for (step, t) in [
            ("s0", taps(c.seed0, 0, c.context)),
            ("s1", taps(c.seed1, c.context, c.kept)),
        ] {
            let bytes: Vec<u8> = t.iter().flat_map(|v| v.to_le_bytes()).collect();
            assert_eq!(
                sha256::hex(&bytes),
                want.get(step).and_then(|v| v.as_str()).unwrap(),
                "{set} {step} taps"
            );
        }
    }
}

#[test]
fn rope_table_matches_the_reference() {
    let Some(e) = env() else { return };
    let g = Set::open(&e.goldens, "dflash-short").unwrap();
    let want = g.f32("rope.inv_freq").unwrap();
    let got = Reference::new(&e.w).inv_freq;
    let differ = got
        .iter()
        .zip(&want)
        .filter(|(a, b)| a.to_bits() != b.to_bits())
        .count();
    println!(
        "  inv_freq: {differ} of {} differ in bits from the reference's",
        got.len()
    );
    assert_eq!(
        differ, 0,
        "inv_freq differs from the reference's in {differ} places"
    );
}

#[test]
fn short_case_matches_fp32_goldens() {
    let Some(e) = env() else { return };
    let d = Dims::GLM53F;
    let g = Set::open(&e.goldens, "dflash-short").unwrap();
    let c = case(&g);
    let r = Reference::new(&e.w);
    // Embedding rows: exactly the reference's (BF16 rows widened).
    let rows = bf16::decode(&e.target.embed_rows(&[c.anchor0, d.mask_token]).unwrap());
    let be = g.f32("s0.block_embed").unwrap();
    assert_eq!(&be[..d.hidden], &rows[..d.hidden]);
    assert_eq!(&be[d.hidden..2 * d.hidden], &rows[d.hidden..]);
    assert_eq!(g.f32("mask_embed").unwrap(), &rows[d.hidden..]);

    // Context: fc + hidden_norm, then every layer's keys and values.
    let t0 = bf16::decode(&taps(c.seed0, 0, c.context));
    let feats = r.context_features(&t0);
    close(
        "s0.ctx_features",
        &feats,
        &g.f32("s0.ctx_features").unwrap(),
        1e-5,
    );
    let mut ctx = Context::new(d);
    r.append_features(&mut ctx, &feats);
    let (gk, gv) = (g.f32("s0.ctx_k").unwrap(), g.f32("s0.ctx_v").unwrap());
    let (mut k, mut v) = (Vec::new(), Vec::new());
    for l in 0..d.layers {
        for p in 0..c.context {
            let (kr, vr) = ctx.row(l, p);
            k.extend_from_slice(kr);
            v.extend_from_slice(vr);
        }
    }
    close("s0.ctx_k", &k, &gk, 1e-5);
    close("s0.ctx_v", &v, &gv, 1e-5);

    // The block: every recorded intermediate of every layer.
    let mut trace = Trace::new();
    let dr = draft(e, &r, &ctx, c.anchor0, Some(&mut trace));
    for l in 0..d.layers {
        for key in [
            "in",
            "attn_norm",
            "attn_dyn",
            "attn_conv_in",
            "k_block",
            "v_block",
            "attn_raw",
            "attn_conv_out",
            "mlp_norm",
            "mlp_dyn",
            "mlp_conv_in",
            "mlp_raw",
            "mlp_conv_out",
            "out",
        ] {
            let name = format!("L{l}.{key}");
            close(
                &format!("s0.{name}"),
                &trace[&name],
                &g.f32(&format!("s0.{name}")).unwrap(),
                1e-5,
            );
        }
    }
    check_draft(&g, "s0", &dr, 1e-5);
    // A sampled walk's q along the greedy path: softmax(scores / 0.7) of each row it used.
    let q07: Vec<f32> = dr
        .walk
        .scores
        .chunks(d.top_k)
        .flat_map(|row| softmax(row, 0.7))
        .collect();
    close("s0.q_t07", &q07, &g.f32("s0.q_t07").unwrap(), 1e-5);
    let lattice = g.f32("s0.lattice").unwrap();
    // The lattice's slot-0 rows are all the anchor's row: the walk's first scores.
    close(
        "s0.lattice[0][0]",
        &dr.walk.scores[..d.top_k],
        &lattice[..d.top_k],
        1e-5,
    );

    // Step 1: the anchor and three accepted drafts become context; draft again.
    let t1 = bf16::decode(&taps(c.seed1, c.context, c.kept));
    let f1 = r.context_features(&t1);
    close(
        "s1.ctx_features",
        &f1,
        &g.f32("s1.ctx_features").unwrap(),
        1e-5,
    );
    r.append_features(&mut ctx, &f1);
    let (mut k, mut v) = (Vec::new(), Vec::new());
    for l in 0..d.layers {
        for p in c.context..c.context + c.kept {
            let (kr, vr) = ctx.row(l, p);
            k.extend_from_slice(kr);
            v.extend_from_slice(vr);
        }
    }
    close("s1.ctx_k", &k, &g.f32("s1.ctx_k").unwrap(), 1e-5);
    close("s1.ctx_v", &v, &g.f32("s1.ctx_v").unwrap(), 1e-5);
    let dr1 = draft(e, &r, &ctx, c.anchor1, None);
    check_draft(&g, "s1", &dr1, 1e-5);
}

#[test]
fn window_case_matches_fp32_goldens() {
    let Some(e) = env() else { return };
    let d = Dims::GLM53F;
    let g = Set::open(&e.goldens, "dflash-window").unwrap();
    let c = case(&g);
    let mut r = Reference::new(&e.w);
    let mut ctx = Context::new(d);
    r.append(&mut ctx, &bf16::decode(&taps(c.seed0, 0, c.context)));
    let mut trace = Trace::new();
    let dr = draft(e, &r, &ctx, c.anchor0, Some(&mut trace));
    for l in 0..d.layers {
        close(
            &format!("s0.L{l}.out"),
            &trace[&format!("L{l}.out")],
            &g.f32(&format!("s0.L{l}.out")).unwrap(),
            1e-5,
        );
    }
    let exact = close(
        "s0.final (window 2048)",
        &dr.hidden,
        &g.f32("s0.final").unwrap(),
        1e-5,
    );
    check_draft(&g, "s0", &dr, 1e-5);
    // The golden pins the window: one key more or less on the left moves the result far more.
    for wl in [d.window - 2, d.window] {
        r.window_left = wl;
        let other = draft(e, &r, &ctx, c.anchor0, None);
        let (off, _) = rel_rms(&other.hidden, &g.f32("s0.final").unwrap());
        println!("  s0.final with window_left {wl}: rel_rms {off:.3e}");
        assert!(
            off > 20.0 * exact.max(1e-9),
            "window_left {wl} is indistinguishable ({off:.3e} vs {exact:.3e})"
        );
    }
    r.window_left = d.window - 1;
    r.append(&mut ctx, &bf16::decode(&taps(c.seed1, c.context, c.kept)));
    let dr1 = draft(e, &r, &ctx, c.anchor1, None);
    check_draft(&g, "s1", &dr1, 1e-5);
}

#[test]
fn bf16_mode_against_fp32_and_native() {
    let Some(e) = env() else { return };
    let d = Dims::GLM53F;
    let g = Set::open(&e.goldens, "dflash-short").unwrap();
    let n = Set::open(&e.goldens, "dflash-native").unwrap();
    let c = case(&g);
    let mut r = Reference::new(&e.w);
    r.bf16_io = true;
    let mut ctx = Context::new(d);
    r.append(&mut ctx, &bf16::decode(&taps(c.seed0, 0, c.context)));
    let dr = draft(e, &r, &ctx, c.anchor0, None);
    let fp32 = g.f32("s0.final").unwrap();
    let (ours, _) = rel_rms(&dr.hidden, &fp32);
    let (native, _) = rel_rms(&n.f32("s0.final").unwrap(), &fp32);
    let (lo, _) = rel_rms(&dr.logits, &g.f32("s0.logits").unwrap());
    println!("  final vs FP32: BF16-io reference {ours:.3e}, the reference's native BF16 {native:.3e}; logits {lo:.3e}");
    println!(
        "  path {:?} vs FP32 {:?} vs native {:?}",
        dr.walk.tokens,
        g.i32("s0.path").unwrap(),
        n.i32("s0.path").unwrap()
    );
    // GEMM inputs and K/V in BF16 but an f32 residual: closer to FP32 than the all-BF16 reference.
    assert!(ours < native, "BF16-io {ours:.3e} vs native {native:.3e}");
}
