//! `glm53f-score` end to end on one GPU in development mode, and its files through
//! `harness/klgate.py` (feature `cuda`).
//!
//! 1. **Tokens.** The development model the runs use (all 45 decoder layers on repeats of layers
//!    0-4 of `GLM53F_CHECKPOINT_DIR`, the official FP8 experts of layers 3 and 4 from
//!    `GLM53F_EXPERTS_DIR`, zeros for the other 40 MoE layers), loaded through the scorer's own
//!    `engine::load`, decodes two windows of 160 tokens: every other next token its own greedy
//!    pick, the others random. So a teacher made of its output passes the harness's alignment
//!    canary (its top-1 is the next token at about half the rows).
//! 2. **A teacher panel** of those windows (14 rows each: 0-4 and spread to the end), in the
//!    layout `klgate_fetch.py` writes, then `klgate.py plan` and `glm53f-score` twice:
//!    `--pass-rows 8` (the decode path) and `--pass-rows 4096` (the prefill path: each window one
//!    pass of 160 rows, two lanes of 80). The files: every row the plan names, the metadata
//!    (`window_id`, `tokens_sha256`, `plan_sha256`, a `DEVELOPMENT` engine line) and `run.json`.
//! 3. **The harness.** The teacher's rows replaced by the 8-row run's own: `klgate.py score`
//!    gives KL exactly 0 at every row and top-1 agreement 1 for that run, and the prefill-path
//!    run's distance from it (the two paths' rounding on this model); `klgate.py compare`; and
//!    `klgate.py canary` on that teacher, reported.
//! 4. **The real teacher rows**, with `GLM53F_KL_TEACHER` (the subset `klgate_fetch.py` writes):
//!    window `final-0000` (2,048 tokens, 189 rows) planned, scored at both pass sizes with routed
//!    outputs of zeros, and read by `klgate.py score` against the teacher. The values mean
//!    nothing (5 layers' weights); the formats line up.
//!
//! Needs `GLM53F_CHECKPOINT_DIR`, `GLM53F_EXPERTS_DIR` (the routed experts of layers 3 and 4),
//! `python3` and about 9 GiB of free GPU memory; anything missing makes it print why and pass.
//!
//! ```sh
//! GLM53F_CHECKPOINT_DIR=... GLM53F_EXPERTS_DIR=... [GLM53F_KL_TEACHER=...] \
//!   cargo test --release -p glm53f-score --features cuda --test plumbing -- --nocapture
//! ```
#![cfg(feature = "cuda")]

mod common;

use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

use common::*;
use glm53f_forward::device;
use glm53f_forward::shape::VOCAB;
use glm53f_model::json::Json;
use glm53f_score::engine;
use glm53f_score::plan::{tokens_sha256, Plan};
use glm53f_score::Options;

const TOKENS: usize = 160;
const ROWS: [usize; 14] = [0, 1, 2, 3, 4, 9, 20, 40, 63, 64, 100, 127, 128, 158];

fn env_path(k: &str) -> Option<PathBuf> {
    std::env::var_os(k).map(PathBuf::from)
}

fn s(p: &Path) -> &str {
    p.to_str().unwrap()
}

/// Run the scorer; its exit status.
fn score(args: &[&str]) -> bool {
    let st = Command::new(env!("CARGO_BIN_EXE_glm53f-score"))
        .args(args)
        .status()
        .expect("run glm53f-score");
    st.success()
}

/// The development model's greedy continuation: every other next token its pick.
fn windows(ckpt: &Path, experts: &Path) -> Vec<Vec<u32>> {
    let args: Vec<String> = [
        "--checkpoint",
        s(ckpt),
        "--experts",
        "local",
        "--experts-dir",
        s(experts),
        "--local-experts-gib",
        "2",
        "--dev-load-layers",
        "5",
        "--pass-rows",
        "8",
        "--plan",
        "-",
        "--out",
        "-",
    ]
    .map(String::from)
    .to_vec();
    let o = Options::parse(&args, &|_| None).unwrap();
    let mut e = engine::load(&o, TOKENS).unwrap();
    let mut rng = Lcg(11);
    let mut out = Vec::new();
    for _ in 0..2 {
        let mut kv = e.fwd.kv.slot().unwrap();
        let mut t = vec![1000 + (rng.next() % 150_000) as u32];
        while t.len() < TOKENS {
            let pick = e.fwd.decode(&mut [(&mut kv, *t.last().unwrap())]).unwrap()[0];
            t.push(if t.len() % 2 == 1 {
                pick
            } else {
                1000 + (rng.next() % 150_000) as u32
            });
        }
        out.push(t);
    }
    out
}

/// A window's rows from a scorer's output, as a teacher window.
fn from_engine(dir: &Path, id: &str, tokens: &[u32]) -> TeacherWindow {
    let f = read_engine(&dir.join(format!("{id}.safetensors")));
    TeacherWindow {
        id: id.to_string(),
        tokens: tokens.to_vec(),
        positions: f.positions,
        rows: f.rows,
    }
}

fn klgate_score(teacher: &Path, engine: &Path, json: &Path, extra: &[&str]) -> Option<i32> {
    let mut a = vec![
        "score",
        "--teacher",
        s(teacher),
        "--engine",
        s(engine),
        "--json",
        s(json),
    ];
    a.extend_from_slice(extra);
    klgate(&a).status.code()
}

/// Every window file of a run: its rows and metadata as the plan and the run say.
fn check_run(dir: &Path, plan: &Plan, pass_rows: usize) {
    for w in &plan.windows {
        let f = read_engine(&dir.join(format!("{}.safetensors", w.window_id)));
        assert_eq!(f.positions, w.positions, "{}: rows", w.window_id);
        assert_eq!(f.cols, VOCAB);
        assert!(f.rows.iter().flatten().all(|x| x.is_finite()));
        assert_eq!(f.meta("window_id"), w.window_id);
        assert_eq!(f.meta("tokens_sha256"), tokens_sha256(&w.tokens));
        assert_eq!(f.meta("plan_sha256"), plan.sha256);
        let line = f.meta("engine");
        assert!(line.starts_with("DEVELOPMENT"), "{line}");
        assert!(
            line.contains(&format!("passes of {pass_rows} rows")),
            "{line}"
        );
    }
    let run = report(&dir.join("run.json"));
    assert_eq!(run.get("complete").and_then(Json::as_bool), Some(true));
    assert_eq!(
        run.get("windows")
            .and_then(Json::as_array)
            .map(<[Json]>::len),
        Some(plan.windows.len())
    );
    assert_eq!(
        run.get("plan")
            .and_then(|p| p.get("sha256"))
            .and_then(Json::as_str),
        Some(plan.sha256.as_str())
    );
}

#[test]
fn the_scorer_through_the_harness() {
    let (Some(ckpt), Some(experts)) = (
        env_path("GLM53F_CHECKPOINT_DIR"),
        env_path("GLM53F_EXPERTS_DIR"),
    ) else {
        eprintln!("skip: GLM53F_CHECKPOINT_DIR and GLM53F_EXPERTS_DIR are needed");
        return;
    };
    if !have_python() {
        return;
    }
    if device::device_count() == 0 {
        eprintln!("skip: no CUDA device");
        return;
    }
    let (free, _) = device::mem_info().unwrap();
    // The largest step, the 4096-row run, takes about 6.5 GiB (its weights, lanes of 2,048 rows,
    // the local experts' 2 GiB).
    if free < 9 << 30 {
        eprintln!(
            "skip: {:.1} GiB free on the GPU, the test needs 9",
            free as f64 / (1u64 << 30) as f64
        );
        return;
    }
    let root = scratch("plumbing");

    // 1-2. Tokens from the model, a teacher panel, the plan, two runs.
    let toks = windows(&ckpt, &experts);
    let ids = ["syn-0000", "syn-0001"];
    let tdir = root.join("teacher");
    let skeleton: Vec<TeacherWindow> = ids
        .iter()
        .zip(&toks)
        .map(|(id, t)| TeacherWindow {
            id: id.to_string(),
            tokens: t.clone(),
            positions: ROWS.to_vec(),
            rows: vec![vec![0.0; VOCAB]; ROWS.len()],
        })
        .collect();
    write_teacher(&tdir, &skeleton, VOCAB, Some(154_856));
    let plan_path = root.join("plan.json");
    assert!(
        klgate(&["plan", "--teacher", s(&tdir), "--out", s(&plan_path)])
            .status
            .success()
    );
    let plan = Plan::parse(&fs::read(&plan_path).unwrap(), VOCAB, 154_856).unwrap();
    let dev = [
        "--checkpoint",
        s(&ckpt),
        "--experts",
        "local",
        "--experts-dir",
        s(&experts),
        "--local-experts-gib",
        "2",
        "--dev-load-layers",
        "5",
        "--plan",
        s(&plan_path),
    ];
    let (o8, o4096) = (root.join("engine-8"), root.join("engine-4096"));
    for (pass, out) in [("8", &o8), ("4096", &o4096)] {
        let mut a = dev.to_vec();
        a.extend(["--pass-rows", pass, "--out", s(out)]);
        assert!(score(&a), "glm53f-score --pass-rows {pass} failed");
    }
    check_run(&o8, &plan, 8);
    check_run(&o4096, &plan, 4096);

    // 3. The teacher made of the 8-row run's rows.
    let t8 = root.join("teacher-8");
    let engine_teacher: Vec<TeacherWindow> = ids
        .iter()
        .zip(&toks)
        .map(|(id, t)| from_engine(&o8, id, t))
        .collect();
    write_teacher(&t8, &engine_teacher, VOCAB, Some(154_856));
    let (r8, r4096) = (root.join("self-8.json"), root.join("prefill-4096.json"));
    assert_eq!(
        klgate_score(&t8, &o8, &r8, &["--max-mean", "1e-12", "--min-top1", "1"]),
        Some(0),
        "the 8-row run against its own rows"
    );
    let rep = report(&r8);
    let kl = row_klds(&rep);
    assert_eq!(kl.len(), 2 * ROWS.len());
    assert!(kl.iter().all(|&k| k == 0.0), "{kl:?}");
    assert_eq!(summary_f64(&rep, "top1_agreement"), 1.0);
    assert_eq!(klgate_score(&t8, &o4096, &r4096, &[]), Some(0));
    let rep2 = report(&r4096);
    let (mean, top1) = (
        summary_f64(&rep2, "mean_kld"),
        summary_f64(&rep2, "top1_agreement"),
    );
    assert!(mean > 0.0 && mean.is_finite(), "the two paths: {mean}");
    let cmp = klgate(&["compare", s(&r4096), s(&r8)]);
    assert!(cmp.status.success());
    let canary = klgate(&["canary", "--teacher", s(&t8)]);
    eprintln!(
        "development model, 2 windows x 14 rows: the 8-row run against a teacher made of its own \
         rows: KL exactly 0 at {}/{} rows, top-1 1.0; the 4096-row run (prefill path) against it: \
         mean KL {mean:.3e} nats, top-1 {top1:.4}; canary on that teacher: exit {:?}",
        kl.len(),
        kl.len(),
        canary.status.code()
    );

    // 4. The real teacher rows: window final-0000 at both pass sizes, routed outputs of zeros.
    let Some(real) = env_path("GLM53F_KL_TEACHER") else {
        eprintln!(
            "GLM53F_KL_TEACHER is not set: the format check against the real teacher skipped"
        );
        fs::remove_dir_all(&root).unwrap();
        return;
    };
    let rplan = root.join("plan-real.json");
    assert!(klgate(&[
        "plan",
        "--teacher",
        s(&real),
        "--windows",
        "final-0000",
        "--out",
        s(&rplan),
    ])
    .status
    .success());
    let rp = Plan::parse(&fs::read(&rplan).unwrap(), VOCAB, 154_856).unwrap();
    for pass in ["4096", "8"] {
        let out = root.join(format!("real-{pass}"));
        assert!(score(&[
            "--checkpoint",
            s(&ckpt),
            "--experts",
            "zero",
            "--dev-load-layers",
            "5",
            "--plan",
            s(&rplan),
            "--pass-rows",
            pass,
            "--out",
            s(&out),
        ]));
        check_run(&out, &rp, pass.parse().unwrap());
        let json = root.join(format!("real-{pass}.json"));
        assert_eq!(
            klgate_score(&real, &out, &json, &["--windows", "final-0000"]),
            Some(0),
            "klgate.py score against the real teacher rows"
        );
        let r = report(&json);
        let rows = r
            .get("summary")
            .and_then(|x| x.get("scored_rows"))
            .and_then(Json::as_u64);
        assert_eq!(rows, Some(rp.windows[0].positions.len() as u64));
        eprintln!(
            "the real teacher, final-0000, --pass-rows {pass}: {} rows read by klgate.py score, \
             mean KL {:.3} nats (meaningless: 5 layers' weights, zero experts)",
            rows.unwrap(),
            summary_f64(&r, "mean_kld")
        );
    }
    fs::remove_dir_all(&root).unwrap();
}
