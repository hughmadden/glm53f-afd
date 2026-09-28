//! The scorer's plan reading and file format against `harness/klgate.py`, on the CPU.
//!
//! A teacher panel in the dataset's layout (three windows of 24 tokens, 40 stored columns of
//! which 36 are the tokenizer's; the subset form `klgate_fetch.py` writes), `klgate.py plan`, the
//! plan read back by [`Plan::parse`], and the teacher's own rows written by [`RowWriter`] as an
//! engine's output, with the metadata the scorer writes. Then `klgate.py score`:
//!
//! - the teacher against itself through the scorer's files: KL exactly 0 at every row, top-1
//!   agreement 1, the gate (`--max-mean 1e-12 --min-top1 1`) passing;
//! - one row changed by a small amount: that row's KL above 0, every other row 0, a tight gate
//!   failing (exit 3);
//! - a file whose token digest is not the teacher's, or that lacks a teacher row: refused
//!   (exit 1).
//!
//! Needs `python3` (or `GLM53F_PYTHON`); skips without it.

mod common;

use std::fs;

use common::*;
use glm53f_score::out::RowWriter;
use glm53f_score::plan::{tokens_sha256, Plan};

const VOCAB: usize = 40;
const TOKENIZER: usize = 36;
const TOKENS: usize = 24;

/// A teacher: the next token is the argmax at three rows in four (the harness's alignment
/// canary wants 0.2-0.995); the padded columns low.
fn teacher_windows() -> Vec<TeacherWindow> {
    let mut rng = Lcg(7);
    let sets: [&[usize]; 3] = [
        &[0, 1, 2, 3, 4, 7, 12, 17, 22],
        &[0, 5, 10, 15, 20],
        &(0..TOKENS - 1).collect::<Vec<_>>(),
    ];
    sets.iter()
        .enumerate()
        .map(|(k, pos)| {
            let tokens: Vec<u32> = (0..TOKENS)
                .map(|_| (rng.next() % TOKENIZER as u64) as u32)
                .collect();
            let rows = pos
                .iter()
                .map(|&r| {
                    let mut row: Vec<f32> = (0..VOCAB).map(|_| 2.0 * rng.unit()).collect();
                    if r % 4 != 0 {
                        row[tokens[r + 1] as usize] += 9.0;
                    }
                    for x in &mut row[TOKENIZER..] {
                        *x = -3.0;
                    }
                    row
                })
                .collect();
            TeacherWindow {
                id: format!("final-{k:04}"),
                tokens,
                positions: pos.to_vec(),
                rows,
            }
        })
        .collect()
}

/// The teacher's rows written as the scorer writes a window: with another token digest, without
/// the row `drop_row`, or with the row `change` (window, position) moved.
fn write_engine(
    dir: &std::path::Path,
    plan: &Plan,
    teacher: &[TeacherWindow],
    digest: Option<&str>,
    drop_row: Option<usize>,
    change: Option<(usize, usize)>,
) {
    fs::create_dir_all(dir).unwrap();
    for (k, (w, t)) in plan.windows.iter().zip(teacher).enumerate() {
        let fed = tokens_sha256(&w.tokens);
        let meta = [
            ("window_id", w.window_id.as_str()),
            ("tokens_sha256", digest.unwrap_or(&fed)),
            ("plan_sha256", plan.sha256.as_str()),
            ("engine", "test: the teacher's rows"),
        ];
        let positions: Vec<usize> = w
            .positions
            .iter()
            .copied()
            .filter(|&p| Some(p) != drop_row)
            .collect();
        let mut f = RowWriter::create(dir, &w.window_id, &positions, VOCAB, &meta).unwrap();
        for (i, &p) in w.positions.iter().enumerate() {
            if Some(p) == drop_row {
                continue;
            }
            let mut row = t.rows[i].clone();
            if change == Some((k, p)) {
                row[3] += 0.25;
            }
            f.push(p, &row).unwrap();
        }
        f.finish().unwrap();
    }
}

#[test]
fn the_scorers_files_read_by_the_harness() {
    if !have_python() {
        return;
    }
    let root = scratch("format");
    let tdir = root.join("teacher");
    let teacher = teacher_windows();
    write_teacher(&tdir, &teacher, VOCAB, Some(TOKENIZER));
    let plan_path = root.join("plan.json");
    let out = klgate(&[
        "plan",
        "--teacher",
        tdir.to_str().unwrap(),
        "--out",
        plan_path.to_str().unwrap(),
    ]);
    assert!(out.status.success(), "klgate.py plan failed");

    // The plan as the scorer reads it.
    let plan = Plan::parse(&fs::read(&plan_path).unwrap(), VOCAB, TOKENIZER).unwrap();
    assert_eq!(plan.windows.len(), 3);
    for (w, t) in plan.windows.iter().zip(&teacher) {
        assert_eq!(
            (&w.window_id, &w.tokens, &w.positions),
            (&t.id, &t.tokens, &t.positions)
        );
    }
    // A tokenizer bound below an id in the plan is refused.
    assert!(Plan::parse(&fs::read(&plan_path).unwrap(), VOCAB, 1).is_err());

    // The teacher against itself through the scorer's files.
    let same = root.join("engine-same");
    write_engine(&same, &plan, &teacher, None, None, None);
    let f = read_engine(&same.join("final-0000.safetensors"));
    assert_eq!(f.positions, teacher[0].positions);
    assert_eq!(f.cols, VOCAB);
    assert_eq!(f.meta("plan_sha256"), plan.sha256);
    let rep = root.join("same.json");
    let s = |engine: &std::path::Path, json: &std::path::Path, gate: bool| {
        let mut args = vec![
            "score".to_string(),
            "--teacher".into(),
            tdir.to_str().unwrap().into(),
            "--engine".into(),
            engine.to_str().unwrap().into(),
            "--json".into(),
            json.to_str().unwrap().into(),
            "--bootstrap".into(),
            "200".into(),
            "--jobs".into(),
            "1".into(),
        ];
        if gate {
            args.extend(["--max-mean", "1e-12", "--min-top1", "1"].map(String::from));
        }
        let a: Vec<&str> = args.iter().map(String::as_str).collect();
        klgate(&a).status.code()
    };
    assert_eq!(s(&same, &rep, true), Some(0), "the self-score gate");
    let r = report(&rep);
    let kl = row_klds(&r);
    let rows: usize = teacher.iter().map(|t| t.positions.len()).sum();
    assert_eq!(kl.len(), rows);
    assert!(
        kl.iter().all(|&k| k == 0.0),
        "KL(teacher || teacher) is not 0: {kl:?}"
    );
    assert_eq!(summary_f64(&r, "top1_agreement"), 1.0);
    let engine = r.get("engine").unwrap();
    assert_eq!(
        engine
            .get("final-0001")
            .and_then(|m| m.get("engine"))
            .and_then(|e| e.as_str()),
        Some("test: the teacher's rows")
    );

    // One row changed: its KL above 0, the others exactly 0; a tight gate fails.
    let moved = root.join("engine-moved");
    write_engine(&moved, &plan, &teacher, None, None, Some((1, 10)));
    let rep2 = root.join("moved.json");
    assert_eq!(
        s(&moved, &rep2, true),
        Some(3),
        "a tight gate over a changed row"
    );
    let kl2 = row_klds(&report(&rep2));
    let nonzero: Vec<f64> = kl2.iter().copied().filter(|&k| k != 0.0).collect();
    assert_eq!(nonzero.len(), 1, "{kl2:?}");
    assert!(nonzero[0] > 0.0);

    // Refused: another token digest, a missing teacher row.
    let digest = root.join("engine-digest");
    write_engine(&digest, &plan, &teacher, Some(&"0".repeat(64)), None, None);
    assert_eq!(s(&digest, &root.join("d.json"), false), Some(1));
    let missing = root.join("engine-missing");
    write_engine(&missing, &plan, &teacher, None, Some(12), None);
    assert_eq!(s(&missing, &root.join("m.json"), false), Some(1));
    eprintln!(
        "plan read back; the scorer's files: KL exactly 0 at {rows}/{rows} rows against the \
         teacher, one changed row alone above 0 ({:.3e}), a wrong digest and a missing row refused",
        nonzero[0]
    );
    fs::remove_dir_all(&root).unwrap();
}
