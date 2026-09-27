//! A minimal safetensors reader: the header, and one tensor's bytes at a time.

use std::fs::File;
use std::io::{self, Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};

use super::json::{self, Json};
use crate::fp8::Fp8Matrix;

#[derive(Clone, Debug)]
pub struct TensorInfo {
    pub dtype: String,
    pub shape: Vec<usize>,
    /// Absolute byte offset of the tensor in the file.
    pub start: u64,
    pub len: u64,
}

#[derive(Clone, Debug)]
pub struct SafeTensors {
    pub path: PathBuf,
    pub tensors: Vec<(String, TensorInfo)>,
}

fn bad(msg: String) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, msg)
}

impl SafeTensors {
    pub fn open(path: &Path) -> io::Result<Self> {
        let mut f = File::open(path)?;
        let mut n = [0u8; 8];
        f.read_exact(&mut n)?;
        let n = u64::from_le_bytes(n);
        if n > 512 << 20 {
            return Err(bad(format!(
                "{}: implausible header size {n}",
                path.display()
            )));
        }
        let mut h = vec![0u8; n as usize];
        f.read_exact(&mut h)?;
        let text =
            std::str::from_utf8(&h).map_err(|e| bad(format!("{}: header: {e}", path.display())))?;
        let j = json::parse(text).map_err(|e| bad(format!("{}: {e}", path.display())))?;
        let data_start = 8 + n;
        let mut tensors = Vec::new();
        for (name, v) in j
            .as_object()
            .ok_or_else(|| bad("header is not an object".into()))?
        {
            if name == "__metadata__" {
                continue;
            }
            let dtype = v
                .get("dtype")
                .and_then(Json::as_str)
                .ok_or_else(|| bad(format!("{name}: dtype")))?;
            let shape = v
                .get("shape")
                .and_then(Json::as_array)
                .ok_or_else(|| bad(format!("{name}: shape")))?
                .iter()
                .map(|d| d.as_u64().map(|x| x as usize))
                .collect::<Option<Vec<_>>>()
                .ok_or_else(|| bad(format!("{name}: shape")))?;
            let off = v
                .get("data_offsets")
                .and_then(Json::as_array)
                .ok_or_else(|| bad(format!("{name}: offsets")))?;
            let (b, e) = match off {
                [b, e] => (b.as_u64(), e.as_u64()),
                _ => (None, None),
            };
            let (b, e) = (
                b.ok_or_else(|| bad(format!("{name}: offsets")))?,
                e.ok_or_else(|| bad(format!("{name}: offsets")))?,
            );
            tensors.push((
                name.clone(),
                TensorInfo {
                    dtype: dtype.to_string(),
                    shape,
                    start: data_start + b,
                    len: e - b,
                },
            ));
        }
        Ok(SafeTensors {
            path: path.to_path_buf(),
            tensors,
        })
    }

    pub fn get(&self, name: &str) -> Option<&TensorInfo> {
        self.tensors.iter().find(|(n, _)| n == name).map(|(_, t)| t)
    }

    pub fn read(&self, info: &TensorInfo) -> io::Result<Vec<u8>> {
        let mut f = File::open(&self.path)?;
        f.seek(SeekFrom::Start(info.start))?;
        let mut buf = vec![0u8; info.len as usize];
        f.read_exact(&mut buf)?;
        Ok(buf)
    }
}

/// A directory of safetensors files, searched by tensor name.
pub struct Checkpoint {
    pub files: Vec<SafeTensors>,
}

impl Checkpoint {
    pub fn open_dir(dir: &Path) -> io::Result<Self> {
        let mut paths: Vec<PathBuf> = std::fs::read_dir(dir)?
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| p.extension().is_some_and(|x| x == "safetensors"))
            .collect();
        paths.sort();
        let files = paths
            .iter()
            .map(|p| SafeTensors::open(p))
            .collect::<io::Result<Vec<_>>>()?;
        Ok(Checkpoint { files })
    }

    pub fn find(&self, name: &str) -> Option<(&SafeTensors, &TensorInfo)> {
        self.files.iter().find_map(|f| f.get(name).map(|t| (f, t)))
    }

    fn read_checked(&self, name: &str, dtype: &str) -> io::Result<(Vec<u8>, Vec<usize>)> {
        let (f, t) = self
            .find(name)
            .ok_or_else(|| io::Error::new(io::ErrorKind::NotFound, name.to_string()))?;
        if t.dtype != dtype {
            return Err(bad(format!("{name}: dtype {} (expected {dtype})", t.dtype)));
        }
        Ok((f.read(t)?, t.shape.clone()))
    }

    /// Rows `first..first + count` of a 2-D BF16 tensor (for example embedding rows), read
    /// without loading the whole tensor.
    pub fn read_bf16_rows(&self, name: &str, first: usize, count: usize) -> io::Result<Vec<u16>> {
        let (f, t) = self
            .find(name)
            .ok_or_else(|| io::Error::new(io::ErrorKind::NotFound, name.to_string()))?;
        if t.dtype != "BF16" || t.shape.len() != 2 || first + count > t.shape[0] {
            return Err(bad(format!(
                "{name}: rows {first}+{count} of {:?} {}",
                t.shape, t.dtype
            )));
        }
        let row = t.shape[1] as u64 * 2;
        let sub = TensorInfo {
            dtype: t.dtype.clone(),
            shape: vec![count, t.shape[1]],
            start: t.start + first as u64 * row,
            len: count as u64 * row,
        };
        Ok(crate::bf16::from_le_bytes(&f.read(&sub)?))
    }

    pub fn read_bf16(&self, name: &str) -> io::Result<(Vec<u16>, Vec<usize>)> {
        let (b, s) = self.read_checked(name, "BF16")?;
        Ok((crate::bf16::from_le_bytes(&b), s))
    }

    pub fn read_f32(&self, name: &str) -> io::Result<(Vec<f32>, Vec<usize>)> {
        let (b, s) = self.read_checked(name, "F32")?;
        Ok((
            b.chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect(),
            s,
        ))
    }

    /// An FP8 weight `<name>` with its `<name>_scale_inv` (for example
    /// `...mlp.gate_proj.weight`).
    pub fn read_fp8(&self, weight_name: &str) -> io::Result<Fp8Matrix> {
        let (w, s) = self.read_checked(weight_name, "F8_E4M3")?;
        let (scale, ss) = self.read_f32(&format!("{weight_name}_scale_inv"))?;
        if s.len() != 2
            || ss.len() != 2
            || ss[0] != s[0].div_ceil(128)
            || ss[1] != s[1].div_ceil(128)
        {
            return Err(bad(format!("{weight_name}: shape {s:?} with scale {ss:?}")));
        }
        Ok(Fp8Matrix::new(s[0], s[1], w, scale))
    }
}
