//! Shared by the scorer's tests: `harness/klgate.py`, a teacher panel in the dataset's layout
//! (as `klgate_fetch.py` leaves a subset of it: manifest, token arrays, `teacher-rows/`), and the
//! scorer's files read back.
#![allow(dead_code)]

use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};

use glm53f_dsa::sha256::sha256_hex;
use glm53f_model::dtype::DType;
use glm53f_model::json::{self, Json};
use glm53f_model::safetensors::{read_file_header, serialize};

pub fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

/// `GLM53F_PYTHON`, else `python3`.
pub fn python() -> String {
    std::env::var("GLM53F_PYTHON").unwrap_or_else(|_| "python3".into())
}

/// Whether the harness can run (printed when not).
pub fn have_python() -> bool {
    let ok = Command::new(python())
        .arg("--version")
        .output()
        .is_ok_and(|o| o.status.success());
    if !ok {
        eprintln!("skip: {} is not available", python());
    }
    ok
}

/// `harness/klgate.py` with `args`.
pub fn klgate(args: &[&str]) -> Output {
    let out = Command::new(python())
        .arg(repo_root().join("harness/klgate.py"))
        .args(args)
        .output()
        .expect("run klgate.py");
    eprintln!(
        "klgate.py {}: exit {:?}\n{}{}",
        args.first().copied().unwrap_or(""),
        out.status.code(),
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    out
}

/// A fresh directory for a test's files.
pub fn scratch(name: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("glm53f-score-{name}-{}", std::process::id()));
    let _ = fs::remove_dir_all(&d);
    fs::create_dir_all(&d).unwrap();
    d
}

/// Deterministic pseudo-random numbers (a 64-bit LCG).
pub struct Lcg(pub u64);

impl Lcg {
    pub fn next(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.0 >> 33
    }

    /// Uniform in [-1, 1).
    pub fn unit(&mut self) -> f32 {
        (self.next() % 2_000_000) as f32 / 1_000_000.0 - 1.0
    }
}

/// One window of a teacher panel: its tokens, and the rows it holds with their logits.
pub struct TeacherWindow {
    pub id: String,
    pub tokens: Vec<u32>,
    pub positions: Vec<usize>,
    pub rows: Vec<Vec<f32>>,
}

/// A `.npy` v1 file of int32.
pub fn write_npy(path: &Path, xs: &[u32]) {
    let mut header = format!(
        "{{'descr': '<i4', 'fortran_order': False, 'shape': ({},), }}",
        xs.len()
    );
    while (10 + header.len() + 1) % 64 != 0 {
        header.push(' ');
    }
    header.push('\n');
    let mut b = b"\x93NUMPY\x01\x00".to_vec();
    b.extend_from_slice(&(header.len() as u16).to_le_bytes());
    b.extend_from_slice(header.as_bytes());
    for &x in xs {
        b.extend_from_slice(&(x as i32).to_le_bytes());
    }
    fs::write(path, b).unwrap();
}

fn le_i32(xs: &[usize]) -> Vec<u8> {
    xs.iter().flat_map(|&x| (x as i32).to_le_bytes()).collect()
}

fn le_f32(rows: &[Vec<f32>]) -> Vec<u8> {
    rows.iter()
        .flatten()
        .flat_map(|x| x.to_le_bytes())
        .collect()
}

/// A teacher panel under `root` in the layout `klgate.py` reads: `dataset-manifest.json`, each
/// window's `calibration/panel-v1/arrays/<id>.tokens.npy` and `teacher-rows/<id>.safetensors`
/// (the subset form `klgate_fetch.py` writes), and the tokenizer receipt when `tok_vocab` is
/// given. A window predicts every token after its first.
pub fn write_teacher(
    root: &Path,
    windows: &[TeacherWindow],
    vocab: usize,
    tok_vocab: Option<usize>,
) {
    let arrays = root.join("calibration/panel-v1/arrays");
    fs::create_dir_all(&arrays).unwrap();
    fs::create_dir_all(root.join("teacher-rows")).unwrap();
    let source = "0".repeat(64);
    let mut files = Vec::new();
    for (k, w) in windows.iter().enumerate() {
        let npy = arrays.join(format!("{}.tokens.npy", w.id));
        write_npy(&npy, &w.tokens);
        let npy_sha = sha256_hex(&fs::read(&npy).unwrap());
        let k_rows = w.positions.len() as u64;
        let bytes = serialize(
            &[
                ("positions", DType::I32, &[k_rows], &le_i32(&w.positions)),
                (
                    "logits",
                    DType::F32,
                    &[k_rows, vocab as u64],
                    &le_f32(&w.rows),
                ),
            ],
            &[
                ("window_id", &w.id),
                ("token_ids_sha256", &npy_sha),
                ("source_sha256", &source),
            ],
        );
        fs::write(
            root.join(format!("teacher-rows/{}.safetensors", w.id)),
            bytes,
        )
        .unwrap();
        files.push(Json::Object(vec![
            ("window_id".into(), Json::Str(w.id.clone())),
            (
                "path".into(),
                Json::Str(format!("logits/window-{k:04}.safetensors")),
            ),
            ("role".into(), Json::Str("final".into())),
            ("domain".into(), Json::Str(format!("d{}", k % 2))),
            (
                "prediction_positions".into(),
                Json::Int(w.tokens.len() as i64 - 1),
            ),
            ("token_ids_sha256".into(), Json::Str(npy_sha)),
            ("sha256".into(), Json::Str(source.clone())),
            ("bytes".into(), Json::Int(0)),
        ]));
    }
    let manifest = Json::Object(vec![
        ("dataset_sha256".into(), Json::Str("synthetic".into())),
        ("model_revision".into(), Json::Str("synthetic".into())),
        ("vocab_size".into(), Json::Int(vocab as i64)),
        ("logit_files".into(), Json::Array(files)),
    ]);
    fs::write(
        root.join("dataset-manifest.json"),
        json::serialize(&manifest),
    )
    .unwrap();
    if let Some(t) = tok_vocab {
        fs::write(
            root.join("calibration/panel-v1/tokenizer.receipt.json"),
            format!("{{\"vocab_size\": {t}}}"),
        )
        .unwrap();
    }
}

/// A scorer's window file read back: metadata, positions and rows.
pub struct EngineFile {
    pub meta: Vec<(String, String)>,
    pub positions: Vec<usize>,
    pub cols: usize,
    pub rows: Vec<Vec<f32>>,
}

impl EngineFile {
    pub fn meta(&self, k: &str) -> &str {
        &self.meta.iter().find(|(x, _)| x == k).unwrap().1
    }
}

/// Read a scorer's window file with `glm53f-model`'s safetensors reader (which checks every
/// tensor's size, the data region's coverage and the file's length).
pub fn read_engine(path: &Path) -> EngineFile {
    let shard = read_file_header(path).unwrap();
    let h = &shard.header;
    let raw = fs::read(path).unwrap();
    let data = &raw[shard.data_start.unwrap() as usize..];
    let (p, l) = (&h.tensors["positions"], &h.tensors["logits"]);
    assert_eq!(p.dtype, DType::I32);
    assert_eq!(l.dtype, DType::F32);
    let positions: Vec<usize> = data[p.begin as usize..p.end as usize]
        .chunks(4)
        .map(|c| i32::from_le_bytes(c.try_into().unwrap()) as usize)
        .collect();
    let cols = l.shape[1] as usize;
    let vals: Vec<f32> = data[l.begin as usize..l.end as usize]
        .chunks(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect();
    assert_eq!(l.shape[0] as usize, positions.len());
    EngineFile {
        meta: h.metadata.clone(),
        positions,
        cols,
        rows: vals.chunks(cols).map(<[f32]>::to_vec).collect(),
    }
}

/// A score report's per-row KL values, and its summary.
pub fn report(path: &Path) -> Json {
    json::parse(&fs::read_to_string(path).unwrap()).unwrap()
}

/// Every per-row KL of a report.
pub fn row_klds(rep: &Json) -> Vec<f64> {
    rep.get("summary")
        .and_then(|s| s.get("per_window"))
        .and_then(Json::as_array)
        .unwrap()
        .iter()
        .flat_map(|w| {
            w.get("kld")
                .and_then(Json::as_array)
                .unwrap()
                .iter()
                .map(|x| x.as_f64().unwrap())
                .collect::<Vec<_>>()
        })
        .collect()
}

pub fn summary_f64(rep: &Json, k: &str) -> f64 {
    rep.get("summary")
        .and_then(|s| s.get(k))
        .and_then(Json::as_f64)
        .unwrap()
}
