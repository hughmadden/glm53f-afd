//! The oracle's DFlash2 golden sets (`oracle/golden_dflash.py`): `dflash-short`, `dflash-window`,
//! `dflash-native` under the directory `GLM53F_GOLDENS` names.
//!
//! Each set is a directory with `tensors.tsv` (`name, file, dtype, shape, sha256`) and one raw
//! little-endian `.bin` per tensor (`f32` or `i32`). This reader checks each file's length against
//! its shape and its sha256 against the manifest.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

/// One tensor of a set.
#[derive(Clone, Debug)]
pub struct Entry {
    pub file: String,
    pub dtype: String,
    pub shape: Vec<usize>,
    pub sha256: String,
}

/// A golden set.
#[derive(Clone, Debug)]
pub struct Set {
    pub dir: PathBuf,
    pub entries: BTreeMap<String, Entry>,
}

impl Set {
    /// Open `root/name` (reads `tensors.tsv`).
    pub fn open(root: &Path, name: &str) -> Result<Set, String> {
        let dir = root.join(name);
        let tsv = dir.join("tensors.tsv");
        let text = std::fs::read_to_string(&tsv).map_err(|e| format!("{}: {e}", tsv.display()))?;
        let mut entries = BTreeMap::new();
        for line in text
            .lines()
            .filter(|l| !l.starts_with('#') && !l.trim().is_empty())
        {
            let f: Vec<&str> = line.split('\t').collect();
            if f.len() != 5 {
                return Err(format!("{}: bad line {line:?}", tsv.display()));
            }
            let shape = if f[3].is_empty() {
                Vec::new()
            } else {
                f[3].split(',')
                    .map(|s| s.parse::<usize>().map_err(|e| format!("{line:?}: {e}")))
                    .collect::<Result<_, _>>()?
            };
            entries.insert(
                f[0].to_string(),
                Entry {
                    file: f[1].to_string(),
                    dtype: f[2].to_string(),
                    shape,
                    sha256: f[4].to_string(),
                },
            );
        }
        Ok(Set { dir, entries })
    }

    /// The set's `manifest.json`, parsed (its `source` and `notes` blocks).
    pub fn manifest(&self) -> Result<glm53f_model::json::Json, String> {
        let path = self.dir.join("manifest.json");
        let text =
            std::fs::read_to_string(&path).map_err(|e| format!("{}: {e}", path.display()))?;
        glm53f_model::json::parse(&text).map_err(|e| format!("{}: {e}", path.display()))
    }

    pub fn has(&self, name: &str) -> bool {
        self.entries.contains_key(name)
    }

    pub fn shape(&self, name: &str) -> Result<&[usize], String> {
        Ok(&self.entry(name)?.shape)
    }

    fn entry(&self, name: &str) -> Result<&Entry, String> {
        self.entries
            .get(name)
            .ok_or_else(|| format!("{}: no tensor {name}", self.dir.display()))
    }

    fn raw(&self, name: &str, dtype: &str) -> Result<Vec<u8>, String> {
        let e = self.entry(name)?;
        if e.dtype != dtype {
            return Err(format!("{name}: {} (wanted {dtype})", e.dtype));
        }
        let path = self.dir.join(&e.file);
        let raw = std::fs::read(&path).map_err(|e| format!("{}: {e}", path.display()))?;
        let n: usize = e.shape.iter().product();
        if raw.len() != n * 4 {
            return Err(format!(
                "{}: {} bytes for shape {:?}",
                path.display(),
                raw.len(),
                e.shape
            ));
        }
        let digest = crate::sha256::hex(&raw);
        if digest != e.sha256 {
            return Err(format!(
                "{}: sha256 {digest}, manifest {}",
                path.display(),
                e.sha256
            ));
        }
        Ok(raw)
    }

    pub fn f32(&self, name: &str) -> Result<Vec<f32>, String> {
        Ok(self
            .raw(name, "f32")?
            .as_chunks::<4>()
            .0
            .iter()
            .map(|b| f32::from_le_bytes(*b))
            .collect())
    }

    pub fn i32(&self, name: &str) -> Result<Vec<i32>, String> {
        Ok(self
            .raw(name, "i32")?
            .as_chunks::<4>()
            .0
            .iter()
            .map(|b| i32::from_le_bytes(*b))
            .collect())
    }
}

/// Relative RMS difference `rms(a - b) / rms(b)` and the largest absolute difference.
pub fn rel_rms(a: &[f32], b: &[f32]) -> (f64, f64) {
    assert_eq!(a.len(), b.len(), "rel_rms: lengths differ");
    let (mut num, mut den, mut max) = (0f64, 0f64, 0f64);
    for (&x, &y) in a.iter().zip(b) {
        let d = x as f64 - y as f64;
        num += d * d;
        den += (y as f64) * (y as f64);
        max = max.max(d.abs());
    }
    ((num / den.max(f64::MIN_POSITIVE)).sqrt(), max)
}
