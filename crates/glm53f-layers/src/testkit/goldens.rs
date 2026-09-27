//! Golden fixtures recorded by the reference oracle.
//!
//! Layout: `oracle/goldens/<set>/manifest.json` maps tensor names to
//! `{"file", "dtype", "shape", "sha256"}`; each file holds the raw little-endian values.
//! The manifest may also nest that map under a `"tensors"` key. Every read checks the
//! byte count and the SHA-256 digest.

use std::path::{Path, PathBuf};

use super::json::{self, Json};
use super::sha256;

#[derive(Clone, Debug)]
pub struct GoldenEntry {
    pub name: String,
    pub file: String,
    pub dtype: String,
    pub shape: Vec<usize>,
    pub sha256: String,
}

impl GoldenEntry {
    pub fn numel(&self) -> usize {
        self.shape.iter().product()
    }
}

#[derive(Clone, Debug)]
pub struct GoldenSet {
    pub dir: PathBuf,
    pub entries: Vec<GoldenEntry>,
}

/// Element size of a dtype name, in either safetensors (`BF16`) or torch (`bfloat16`)
/// spelling.
pub fn dtype_size(dtype: &str) -> Option<usize> {
    Some(
        match dtype.to_ascii_lowercase().trim_start_matches("torch.") {
            "bf16" | "bfloat16" | "f16" | "float16" | "half" | "i16" | "int16" => 2,
            "f32" | "float32" | "float" | "i32" | "int32" | "int" | "u32" | "uint32" => 4,
            "f64" | "float64" | "double" | "i64" | "int64" | "long" => 8,
            "u8" | "uint8" | "i8" | "int8" | "bool" | "f8_e4m3" | "float8_e4m3fn" => 1,
            _ => return None,
        },
    )
}

/// Every `<root>/<set>/manifest.json`, sorted.
pub fn discover(root: &Path) -> Vec<PathBuf> {
    let Ok(rd) = std::fs::read_dir(root) else {
        return Vec::new();
    };
    let mut sets: Vec<PathBuf> = rd
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.join("manifest.json").is_file())
        .collect();
    sets.sort();
    sets
}

impl GoldenSet {
    pub fn load(dir: &Path) -> Result<Self, String> {
        let path = dir.join("manifest.json");
        let text =
            std::fs::read_to_string(&path).map_err(|e| format!("{}: {e}", path.display()))?;
        let j = json::parse(&text).map_err(|e| format!("{}: {e}", path.display()))?;
        let map = j.get("tensors").unwrap_or(&j);
        let obj = map.as_object().ok_or("manifest is not an object")?;
        let mut entries = Vec::new();
        for (name, v) in obj {
            let Some(file) = v.get("file").and_then(Json::as_str) else {
                continue;
            };
            let dtype = v
                .get("dtype")
                .and_then(Json::as_str)
                .ok_or(format!("{name}: dtype"))?;
            let shape = v
                .get("shape")
                .and_then(Json::as_array)
                .ok_or(format!("{name}: shape"))?
                .iter()
                .map(|d| d.as_u64().map(|x| x as usize))
                .collect::<Option<Vec<_>>>()
                .ok_or(format!("{name}: shape"))?;
            let sha = v
                .get("sha256")
                .and_then(Json::as_str)
                .ok_or(format!("{name}: sha256"))?;
            entries.push(GoldenEntry {
                name: name.clone(),
                file: file.to_string(),
                dtype: dtype.to_string(),
                shape,
                sha256: sha.to_ascii_lowercase(),
            });
        }
        Ok(GoldenSet {
            dir: dir.to_path_buf(),
            entries,
        })
    }

    pub fn get(&self, name: &str) -> Option<&GoldenEntry> {
        self.entries.iter().find(|e| e.name == name)
    }

    /// The first entry whose name, with `/` read as `.`, contains the layer marker
    /// (`layers.{layer}.`) and ends with one of `suffixes`.
    pub fn find_layer(&self, layer: usize, suffixes: &[&str]) -> Option<&GoldenEntry> {
        let marker = format!("layers.{layer}.");
        for suf in suffixes {
            if let Some(e) = self.entries.iter().find(|e| {
                let n = e.name.replace('/', ".");
                n.contains(&marker) && n.ends_with(suf)
            }) {
                return Some(e);
            }
        }
        None
    }

    /// The raw bytes of an entry, checked against its size and digest.
    pub fn read(&self, e: &GoldenEntry) -> Result<Vec<u8>, String> {
        let path = self.dir.join(&e.file);
        let bytes = std::fs::read(&path).map_err(|err| format!("{}: {err}", path.display()))?;
        if let Some(sz) = dtype_size(&e.dtype) {
            if bytes.len() != e.numel() * sz {
                return Err(format!(
                    "{}: {} bytes, expected {}",
                    e.name,
                    bytes.len(),
                    e.numel() * sz
                ));
            }
        }
        let got = sha256::hex_digest(&bytes);
        if got != e.sha256 {
            return Err(format!("{}: sha256 {got}, manifest {}", e.name, e.sha256));
        }
        Ok(bytes)
    }

    /// An entry's values as f32 (from F32, BF16 or F16).
    pub fn read_f32(&self, e: &GoldenEntry) -> Result<Vec<f32>, String> {
        let b = self.read(e)?;
        let dt = e.dtype.to_ascii_lowercase();
        let dt = dt.trim_start_matches("torch.");
        Ok(match dt {
            "f32" | "float32" | "float" => b
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect(),
            "bf16" | "bfloat16" => crate::bf16::widen(&crate::bf16::from_le_bytes(&b)),
            "f16" | "float16" | "half" => b
                .chunks_exact(2)
                .map(|c| f16_to_f32(u16::from_le_bytes([c[0], c[1]])))
                .collect(),
            other => return Err(format!("{}: dtype {other} is not a float type", e.name)),
        })
    }

    /// An entry's values as BF16 bit patterns (BF16 only).
    pub fn read_bf16(&self, e: &GoldenEntry) -> Result<Vec<u16>, String> {
        let dt = e.dtype.to_ascii_lowercase();
        if !matches!(dt.trim_start_matches("torch."), "bf16" | "bfloat16") {
            return Err(format!("{}: dtype {} is not BF16", e.name, e.dtype));
        }
        Ok(crate::bf16::from_le_bytes(&self.read(e)?))
    }

    /// An entry's values as i64 (from I64 or I32).
    pub fn read_i64(&self, e: &GoldenEntry) -> Result<Vec<i64>, String> {
        let b = self.read(e)?;
        let dt = e.dtype.to_ascii_lowercase();
        Ok(match dt.trim_start_matches("torch.") {
            "i64" | "int64" | "long" => b
                .chunks_exact(8)
                .map(|c| i64::from_le_bytes(c.try_into().unwrap()))
                .collect(),
            "i32" | "int32" | "int" => b
                .chunks_exact(4)
                .map(|c| i32::from_le_bytes(c.try_into().unwrap()) as i64)
                .collect(),
            other => return Err(format!("{}: dtype {other} is not an integer type", e.name)),
        })
    }
}

fn f16_to_f32(h: u16) -> f32 {
    let s = if h & 0x8000 != 0 { -1.0 } else { 1.0 };
    let e = ((h >> 10) & 0x1F) as i32;
    let m = (h & 0x3FF) as f32;
    s * match e {
        0 => m * 2f32.powi(-24),
        31 => {
            if m == 0.0 {
                f32::INFINITY
            } else {
                f32::NAN
            }
        }
        _ => (1.0 + m / 1024.0) * 2f32.powi(e - 15),
    }
}
