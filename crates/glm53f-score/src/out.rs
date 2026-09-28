//! The scorer's output: one safetensors file per window, its logits streamed row by row as the
//! forward computes them, and `run.json`.

use std::fs::{self, File};
use std::io::{self, BufWriter, Write};
use std::path::{Path, PathBuf};

use glm53f_model::json::{self, Json};

/// A window's safetensors file, written as its rows come: the header and `positions` first
/// (the rows are known from the plan), then each row's logits in order, and the file renamed
/// from `<window>.safetensors.partial` to `<window>.safetensors` when every row is in.
pub struct RowWriter {
    file: BufWriter<File>,
    partial: PathBuf,
    path: PathBuf,
    positions: Vec<usize>,
    cols: usize,
    next: usize,
    bytes: u64,
    buf: Vec<u8>,
}

/// The header: `__metadata__` (strings), then `positions` I32 `[k]` and `logits` F32
/// `[k, cols]` in that order, padded with spaces so the data starts at a multiple of 8.
pub fn header(metadata: &[(&str, &str)], rows: usize, cols: usize) -> Vec<u8> {
    let int = |v: usize| Json::Int(v as i64);
    let entry = |dtype: &str, shape: Vec<Json>, a: usize, b: usize| {
        Json::Object(vec![
            ("dtype".to_string(), Json::Str(dtype.to_string())),
            ("shape".to_string(), Json::Array(shape)),
            (
                "data_offsets".to_string(),
                Json::Array(vec![int(a), int(b)]),
            ),
        ])
    };
    let meta = metadata
        .iter()
        .map(|(k, v)| (k.to_string(), Json::Str(v.to_string())))
        .collect();
    let p = rows * 4;
    let text = json::serialize(&Json::Object(vec![
        ("__metadata__".to_string(), Json::Object(meta)),
        ("positions".to_string(), entry("I32", vec![int(rows)], 0, p)),
        (
            "logits".to_string(),
            entry("F32", vec![int(rows), int(cols)], p, p + rows * cols * 4),
        ),
    ]));
    let mut h = text.into_bytes();
    while !(8 + h.len()).is_multiple_of(8) {
        h.push(b' ');
    }
    let mut out = (h.len() as u64).to_le_bytes().to_vec();
    out.extend_from_slice(&h);
    out
}

impl RowWriter {
    /// Start `<dir>/<name>.safetensors` for the rows `positions` of `cols` logits each.
    pub fn create(
        dir: &Path,
        name: &str,
        positions: &[usize],
        cols: usize,
        metadata: &[(&str, &str)],
    ) -> io::Result<RowWriter> {
        let path = dir.join(format!("{name}.safetensors"));
        let partial = dir.join(format!("{name}.safetensors.partial"));
        let mut file = BufWriter::with_capacity(8 << 20, File::create(&partial)?);
        let h = header(metadata, positions.len(), cols);
        file.write_all(&h)?;
        let mut p = Vec::with_capacity(positions.len() * 4);
        for &x in positions {
            let v = i32::try_from(x)
                .map_err(|_| io::Error::new(io::ErrorKind::InvalidInput, "a row past i32"))?;
            p.extend_from_slice(&v.to_le_bytes());
        }
        file.write_all(&p)?;
        Ok(RowWriter {
            file,
            partial,
            path,
            positions: positions.to_vec(),
            cols,
            next: 0,
            bytes: (h.len() + p.len()) as u64,
            buf: Vec::with_capacity(cols * 4),
        })
    }

    /// The next row's logits: it must be `positions[i]` for the i-th call.
    pub fn push(&mut self, row: usize, logits: &[f32]) -> io::Result<()> {
        if self.positions.get(self.next) != Some(&row) || logits.len() != self.cols {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!(
                    "row {row} of {} logits, expected row {:?} of {}",
                    logits.len(),
                    self.positions.get(self.next),
                    self.cols
                ),
            ));
        }
        self.buf.clear();
        for x in logits {
            self.buf.extend_from_slice(&x.to_le_bytes());
        }
        self.file.write_all(&self.buf)?;
        self.next += 1;
        self.bytes += self.buf.len() as u64;
        Ok(())
    }

    /// Every row written: flush, sync and rename into place. Returns the file's bytes.
    pub fn finish(self) -> io::Result<u64> {
        if self.next != self.positions.len() {
            return Err(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                format!("{} of {} rows written", self.next, self.positions.len()),
            ));
        }
        let file = self.file.into_inner().map_err(|e| e.into_error())?;
        file.sync_all()?;
        drop(file);
        fs::rename(&self.partial, &self.path)?;
        Ok(self.bytes)
    }
}

/// `v` as JSON text, objects and arrays of objects spread over lines (as `json.dump(indent=1)`
/// for the outer levels), everything else compact.
pub fn pretty(v: &Json) -> String {
    fn go(v: &Json, depth: usize, out: &mut String) {
        let pad = " ".repeat(depth + 1);
        match v {
            Json::Object(pairs) if !pairs.is_empty() => {
                out.push_str("{\n");
                for (i, (k, x)) in pairs.iter().enumerate() {
                    out.push_str(&pad);
                    out.push_str(&json::serialize(&Json::Str(k.clone())));
                    out.push_str(": ");
                    go(x, depth + 1, out);
                    out.push_str(if i + 1 < pairs.len() { ",\n" } else { "\n" });
                }
                out.push_str(&" ".repeat(depth));
                out.push('}');
            }
            Json::Array(items) if items.iter().any(|x| matches!(x, Json::Object(_))) => {
                out.push_str("[\n");
                for (i, x) in items.iter().enumerate() {
                    out.push_str(&pad);
                    go(x, depth + 1, out);
                    out.push_str(if i + 1 < items.len() { ",\n" } else { "\n" });
                }
                out.push_str(&" ".repeat(depth));
                out.push(']');
            }
            other => out.push_str(&json::serialize(other)),
        }
    }
    let mut out = String::new();
    go(v, 0, &mut out);
    out.push('\n');
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use glm53f_model::safetensors::read_file_header;

    fn tmp(name: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("glm53f-score-{name}-{}", std::process::id()));
        let _ = fs::remove_dir_all(&d);
        fs::create_dir_all(&d).unwrap();
        d
    }

    #[test]
    fn a_window_file_reads_back() {
        let dir = tmp("write");
        let (pos, cols) = ([0usize, 3, 7], 5);
        let meta = [("window_id", "w-1"), ("tokens_sha256", "ab\"c")];
        let mut w = RowWriter::create(&dir, "w-1", &pos, cols, &meta).unwrap();
        for (i, &r) in pos.iter().enumerate() {
            let row: Vec<f32> = (0..cols).map(|c| (i * 10 + c) as f32 - 0.5).collect();
            w.push(r, &row).unwrap();
        }
        let bytes = w.finish().unwrap();
        let path = dir.join("w-1.safetensors");
        assert!(!dir.join("w-1.safetensors.partial").exists());
        let raw = fs::read(&path).unwrap();
        assert_eq!(raw.len() as u64, bytes);
        let n = u64::from_le_bytes(raw[..8].try_into().unwrap()) as usize;
        assert_eq!((8 + n) % 8, 0, "the data starts at a multiple of 8");
        // An independent reader: glm53f-model's, which checks every tensor's size and coverage.
        let shard = read_file_header(&path).unwrap();
        let h = &shard.header;
        assert_eq!(
            h.metadata,
            vec![
                ("window_id".to_string(), "w-1".to_string()),
                ("tokens_sha256".to_string(), "ab\"c".to_string())
            ]
        );
        assert_eq!(h.tensors["positions"].shape, vec![3]);
        assert_eq!(h.tensors["logits"].shape, vec![3, 5]);
        let data = &raw[8 + n..];
        let p: Vec<i32> = data[..12]
            .chunks(4)
            .map(|c| i32::from_le_bytes(c.try_into().unwrap()))
            .collect();
        assert_eq!(p, vec![0, 3, 7]);
        let l: Vec<f32> = data[12..]
            .chunks(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect();
        assert_eq!(l.len(), 15);
        assert_eq!((l[0], l[6], l[14]), (-0.5, 10.5, 23.5));
        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn rows_out_of_order_or_missing_are_refused() {
        let dir = tmp("refuse");
        let mut w = RowWriter::create(&dir, "w", &[1, 2], 3, &[]).unwrap();
        assert!(w.push(2, &[0.0; 3]).is_err(), "row 2 before row 1");
        assert!(w.push(1, &[0.0; 4]).is_err(), "a row of the wrong width");
        w.push(1, &[0.0; 3]).unwrap();
        assert!(w.finish().is_err(), "a row missing");
        assert!(!dir.join("w.safetensors").exists());
        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn pretty_json_parses_back() {
        let v = Json::Object(vec![
            ("a".to_string(), Json::Int(1)),
            (
                "w".to_string(),
                Json::Array(vec![
                    Json::Object(vec![("x".to_string(), Json::Num(0.5))]),
                    Json::Object(vec![]),
                ]),
            ),
            ("s".to_string(), Json::Str("q\"\n".to_string())),
            (
                "n".to_string(),
                Json::Array(vec![Json::Int(1), Json::Int(2)]),
            ),
        ]);
        let text = pretty(&v);
        assert!(text.contains("\n \"w\": [\n  {\n   \"x\": 0.5\n  },\n  {}\n ],"));
        assert_eq!(json::parse(&text).unwrap(), v);
    }
}
