//! Oracle fixtures: `<dir>/manifest.json` mapping a tensor name to
//! `{file, dtype, shape, sha256}` with raw little-endian `.bin` data next to it.
//!
//! The fixture root defaults to `oracle/goldens` at the workspace root and can
//! be moved with `GLM53F_GOLDENS`. Tests that need fixtures skip when none exist.

use std::path::{Path, PathBuf};

use crate::json::Json;
use crate::safetensors::bytes_to_f32;
use crate::sha256::sha256_hex;

#[derive(Clone, Debug)]
pub struct GoldenTensor {
    pub name: String,
    pub file: PathBuf,
    pub dtype: String,
    pub shape: Vec<usize>,
    pub sha256: String,
}

#[derive(Clone, Debug)]
pub struct GoldenSet {
    pub dir: PathBuf,
    pub tensors: Vec<GoldenTensor>,
}

impl GoldenSet {
    /// Read `<dir>/manifest.json`.
    pub fn open(dir: &Path) -> Result<Self, String> {
        let m = dir.join("manifest.json");
        let text = std::fs::read_to_string(&m).map_err(|e| format!("{}: {e}", m.display()))?;
        let j = Json::parse(&text).map_err(|e| format!("{}: {e}", m.display()))?;
        // Accept either {name: entry} or {"tensors": {name: entry}}.
        let obj = j.get("tensors").and_then(|t| t.as_object()).or(j.as_object()).ok_or("manifest is not an object")?;
        let mut tensors = Vec::new();
        for (name, e) in obj {
            let (Some(file), Some(dtype), Some(shape)) =
                (e.get("file").and_then(|v| v.as_str()), e.get("dtype").and_then(|v| v.as_str()), e.get("shape").and_then(|v| v.as_array()))
            else {
                continue; // metadata entries
            };
            let shape = shape.iter().map(|v| v.as_u64().map(|u| u as usize).ok_or("bad shape")).collect::<Result<Vec<_>, _>>()?;
            let sha256 = e.get("sha256").and_then(|v| v.as_str()).unwrap_or("").to_lowercase();
            tensors.push(GoldenTensor { name: name.clone(), file: dir.join(file), dtype: dtype.to_string(), shape, sha256 });
        }
        Ok(Self { dir: dir.to_path_buf(), tensors })
    }

    /// The fixture root: `GLM53F_GOLDENS`, else `<workspace>/oracle/goldens`.
    pub fn root() -> PathBuf {
        if let Some(d) = std::env::var_os("GLM53F_GOLDENS") {
            return PathBuf::from(d);
        }
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../../oracle/goldens")
    }

    /// Every fixture set under the root (sorted by directory name).
    pub fn discover() -> Vec<GoldenSet> {
        let Ok(rd) = std::fs::read_dir(Self::root()) else { return Vec::new() };
        let mut dirs: Vec<PathBuf> = rd.filter_map(|e| e.ok().map(|e| e.path())).filter(|p| p.join("manifest.json").is_file()).collect();
        dirs.sort();
        dirs.iter().filter_map(|d| Self::open(d).ok()).collect()
    }

    pub fn get(&self, name: &str) -> Option<&GoldenTensor> {
        self.tensors.iter().find(|t| t.name == name)
    }

    /// Names containing every fragment in `parts`.
    pub fn matching(&self, parts: &[&str]) -> Vec<&GoldenTensor> {
        self.tensors.iter().filter(|t| parts.iter().all(|p| t.name.contains(p))).collect()
    }

    /// Load a tensor as f32, verifying its digest when the manifest has one.
    pub fn load_f32(&self, t: &GoldenTensor) -> Result<Vec<f32>, String> {
        let b = std::fs::read(&t.file).map_err(|e| format!("{}: {e}", t.file.display()))?;
        if !t.sha256.is_empty() {
            let got = sha256_hex(&b);
            if got != t.sha256 {
                return Err(format!("{}: sha256 {got}, manifest says {}", t.name, t.sha256));
            }
        }
        let v = bytes_to_f32(&t.dtype, &b)?;
        let n: usize = t.shape.iter().product();
        if v.len() != n {
            return Err(format!("{}: {} values, shape {:?}", t.name, v.len(), t.shape));
        }
        Ok(v)
    }
}
