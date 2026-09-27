//! A minimal safetensors reader (header parse and ranged tensor reads), enough to
//! load one DSA layer's tensors from a checkpoint directory.

use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};

use crate::fp8::e4m3_decode;
use crate::json::Json;
use crate::num::{bf16_bits_to_f32, f16_bits_to_f32};

/// One tensor's entry in a safetensors header.
#[derive(Clone, Debug)]
pub struct TensorInfo {
    pub name: String,
    pub dtype: String,
    pub shape: Vec<usize>,
    /// Byte range relative to the start of the data section.
    pub begin: u64,
    pub end: u64,
}

impl TensorInfo {
    pub fn numel(&self) -> usize {
        self.shape.iter().product()
    }
}

/// One safetensors file.
#[derive(Clone, Debug)]
pub struct SafeTensors {
    pub path: PathBuf,
    pub data_start: u64,
    pub tensors: Vec<TensorInfo>,
}

impl SafeTensors {
    pub fn open(path: &Path) -> Result<Self, String> {
        let mut f = File::open(path).map_err(|e| format!("{}: {e}", path.display()))?;
        let mut n = [0u8; 8];
        f.read_exact(&mut n).map_err(|e| format!("{}: {e}", path.display()))?;
        let n = u64::from_le_bytes(n);
        if n > (256 << 20) {
            return Err(format!("{}: header of {n} bytes", path.display()));
        }
        let mut h = vec![0u8; n as usize];
        f.read_exact(&mut h).map_err(|e| format!("{}: {e}", path.display()))?;
        let text = std::str::from_utf8(&h).map_err(|e| format!("{}: {e}", path.display()))?;
        let j = Json::parse(text.trim_end_matches(' ')).map_err(|e| format!("{}: {e}", path.display()))?;
        let mut tensors = Vec::new();
        for (name, v) in j.as_object().ok_or("header is not an object")? {
            if name == "__metadata__" {
                continue;
            }
            let dtype = v.get("dtype").and_then(|d| d.as_str()).ok_or("missing dtype")?.to_string();
            let shape = v
                .get("shape")
                .and_then(|s| s.as_array())
                .ok_or("missing shape")?
                .iter()
                .map(|x| x.as_u64().map(|u| u as usize).ok_or("bad shape"))
                .collect::<Result<Vec<_>, _>>()?;
            let off = v.get("data_offsets").and_then(|s| s.as_array()).ok_or("missing data_offsets")?;
            let begin = off.first().and_then(|x| x.as_u64()).ok_or("bad offsets")?;
            let end = off.get(1).and_then(|x| x.as_u64()).ok_or("bad offsets")?;
            tensors.push(TensorInfo { name: name.clone(), dtype, shape, begin, end });
        }
        Ok(Self { path: path.to_path_buf(), data_start: 8 + n, tensors })
    }

    pub fn get(&self, name: &str) -> Option<&TensorInfo> {
        self.tensors.iter().find(|t| t.name == name)
    }

    pub fn read_bytes(&self, t: &TensorInfo) -> Result<Vec<u8>, String> {
        let mut f = File::open(&self.path).map_err(|e| format!("{}: {e}", self.path.display()))?;
        f.seek(SeekFrom::Start(self.data_start + t.begin)).map_err(|e| e.to_string())?;
        let mut b = vec![0u8; (t.end - t.begin) as usize];
        f.read_exact(&mut b).map_err(|e| format!("{} {}: {e}", self.path.display(), t.name))?;
        Ok(b)
    }
}

/// Convert raw little-endian tensor bytes to f32 (FP8 is decoded without scales).
pub fn bytes_to_f32(dtype: &str, b: &[u8]) -> Result<Vec<f32>, String> {
    Ok(match dtype {
        "F32" | "float32" | "f32" => b.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect(),
        "BF16" | "bfloat16" | "bf16" => b.chunks_exact(2).map(|c| bf16_bits_to_f32(u16::from_le_bytes([c[0], c[1]]))).collect(),
        "F16" | "float16" | "f16" => b.chunks_exact(2).map(|c| f16_bits_to_f32(u16::from_le_bytes([c[0], c[1]]))).collect(),
        "F8_E4M3" | "float8_e4m3fn" => b.iter().map(|c| e4m3_decode(*c)).collect(),
        "I32" | "int32" | "i32" => b.chunks_exact(4).map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]) as f32).collect(),
        "I64" | "int64" | "i64" => b
            .chunks_exact(8)
            .map(|c| i64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]) as f32)
            .collect(),
        "U8" | "uint8" | "BOOL" | "bool" => b.iter().map(|c| *c as f32).collect(),
        _ => return Err(format!("unsupported dtype {dtype}")),
    })
}

/// A checkpoint directory: every `*.safetensors` file in it.
#[derive(Clone, Debug)]
pub struct Checkpoint {
    pub files: Vec<SafeTensors>,
}

impl Checkpoint {
    pub fn open_dir(dir: &Path) -> Result<Self, String> {
        let mut paths: Vec<PathBuf> = std::fs::read_dir(dir)
            .map_err(|e| format!("{}: {e}", dir.display()))?
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| p.extension().map(|x| x == "safetensors").unwrap_or(false))
            .collect();
        paths.sort();
        let files = paths.iter().map(|p| SafeTensors::open(p)).collect::<Result<Vec<_>, _>>()?;
        Ok(Self { files })
    }

    pub fn find(&self, name: &str) -> Option<(&SafeTensors, &TensorInfo)> {
        self.files.iter().find_map(|f| f.get(name).map(|t| (f, t)))
    }

    /// Read a tensor as f32 with its shape.
    pub fn read_f32(&self, name: &str) -> Result<(Vec<f32>, Vec<usize>), String> {
        let (f, t) = self.find(name).ok_or_else(|| format!("tensor {name} not found"))?;
        let b = f.read_bytes(t)?;
        Ok((bytes_to_f32(&t.dtype, &b)?, t.shape.clone()))
    }

    /// Read rows `[start, start + n)` of a 2-d BF16/F16/F32 tensor as f32.
    pub fn read_rows(&self, name: &str, start: usize, n: usize) -> Result<Vec<f32>, String> {
        let (f, t) = self.find(name).ok_or_else(|| format!("tensor {name} not found"))?;
        if t.shape.len() != 2 || start + n > t.shape[0] {
            return Err(format!("{name}: rows {start}..{} of shape {:?}", start + n, t.shape));
        }
        let elem = match t.dtype.as_str() {
            "BF16" | "F16" => 2,
            "F32" => 4,
            d => return Err(format!("{name}: read_rows does not support {d}")),
        };
        let row_bytes = (t.shape[1] * elem) as u64;
        let sub = TensorInfo {
            name: t.name.clone(),
            dtype: t.dtype.clone(),
            shape: vec![n, t.shape[1]],
            begin: t.begin + start as u64 * row_bytes,
            end: t.begin + (start + n) as u64 * row_bytes,
        };
        bytes_to_f32(&t.dtype, &f.read_bytes(&sub)?)
    }

    /// Read a weight, dequantizing FP8 E4M3 with its `weight_scale_inv`
    /// (one f32 multiplier per 128 x 128 block) when present.
    pub fn read_weight(&self, name: &str) -> Result<(Vec<f32>, Vec<usize>), String> {
        let (f, t) = self.find(name).ok_or_else(|| format!("tensor {name} not found"))?;
        let b = f.read_bytes(t)?;
        let mut v = bytes_to_f32(&t.dtype, &b)?;
        if t.dtype == "F8_E4M3" {
            let sname = format!("{}_scale_inv", name);
            let (s, sshape) = self.read_f32(&sname)?;
            if t.shape.len() != 2 || sshape.len() != 2 {
                return Err(format!("{name}: expected 2-d weight and scales"));
            }
            let (rows, cols) = (t.shape[0], t.shape[1]);
            let (sr, sc) = (rows.div_ceil(128), cols.div_ceil(128));
            if sshape != [sr, sc] {
                return Err(format!("{name}: scale shape {sshape:?}, expected [{sr}, {sc}]"));
            }
            for r in 0..rows {
                for c in 0..cols {
                    v[r * cols + c] *= s[(r / 128) * sc + c / 128];
                }
            }
        }
        Ok((v, t.shape.clone()))
    }

    /// Whether a tensor's bytes look fetched (not all zero in three sampled windows).
    pub fn looks_present(&self, name: &str) -> bool {
        let Some((f, t)) = self.find(name) else { return false };
        let Ok(mut file) = File::open(&f.path) else { return false };
        let len = t.end - t.begin;
        for off in [0, len / 2, len.saturating_sub(4096)] {
            let n = (len - off).min(4096) as usize;
            let mut b = vec![0u8; n];
            if file.seek(SeekFrom::Start(f.data_start + t.begin + off)).is_err() || file.read_exact(&mut b).is_err() {
                return false;
            }
            if b.iter().any(|x| *x != 0) {
                return true;
            }
        }
        false
    }
}
