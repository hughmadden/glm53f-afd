//! Safetensors: strict header parsing, multi-shard checkpoints and positional
//! reads of whole tensors or byte runs of them.
//!
//! Adapted from mimo26f-afd v1.2.0 `crates/mimo26-repack/src/safetensors.rs`
//! (MIT); see this crate's `PROVENANCE.md`. Changes from the source: every
//! dtype (not only `U8`); the header is parsed with [`crate::json`] instead of a
//! private mini-parser; each entry's byte length must equal its shape times its
//! dtype size; the data region must be exactly covered, without holes or
//! overlaps (as the format requires); multi-shard checkpoints, including a check
//! of `model.safetensors.index.json` when present; header bundles; and strided
//! byte-run reads for tensor-parallel slices.
//!
//! Format: `<u64 LE header_len><header_len bytes of JSON><data>`. Each header
//! entry is `{"dtype": .., "shape": [..], "data_offsets": [begin, end]}` with
//! offsets relative to the start of the data region; `__metadata__` holds
//! string-to-string metadata.

use std::collections::BTreeMap;
use std::fs::File;
use std::path::{Path, PathBuf};

use crate::dtype::{numel, DType};
use crate::error::{io_err, Error, Result};
use crate::json::{self, Json};

/// The format's own cap on the header size.
pub const MAX_HEADER_BYTES: u64 = 100 * 1024 * 1024;

/// One tensor's header entry.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TensorEntry {
    pub dtype: DType,
    pub shape: Vec<u64>,
    /// Byte range `[begin, end)` relative to the start of the data region.
    pub begin: u64,
    pub end: u64,
}

impl TensorEntry {
    pub fn byte_len(&self) -> u64 {
        self.end - self.begin
    }
    pub fn numel(&self) -> u64 {
        numel(&self.shape)
    }
}

/// A parsed, validated safetensors header.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Header {
    pub tensors: BTreeMap<String, TensorEntry>,
    /// `__metadata__`, in file order.
    pub metadata: Vec<(String, String)>,
}

impl Header {
    /// Parse and validate header JSON text.
    pub fn parse(text: &str) -> Result<Header> {
        let v = json::parse(text).map_err(Error::Json)?;
        Header::from_json(&v)
    }

    /// Validate an already-parsed header object.
    pub fn from_json(v: &Json) -> Result<Header> {
        let pairs = v
            .as_object()
            .ok_or_else(|| Error::Safetensors("header is not a JSON object".into()))?;
        let mut h = Header::default();
        let mut seen_metadata = false;
        for (name, value) in pairs {
            if name == "__metadata__" {
                if seen_metadata {
                    return Err(Error::Safetensors("duplicate __metadata__".into()));
                }
                seen_metadata = true;
                let m = value
                    .as_object()
                    .ok_or_else(|| Error::Safetensors("__metadata__ is not an object".into()))?;
                json::check_unique_keys(value).map_err(Error::Safetensors)?;
                for (k, x) in m {
                    let s = x.as_str().ok_or_else(|| {
                        Error::Safetensors(format!("__metadata__.{k} is not a string"))
                    })?;
                    h.metadata.push((k.clone(), s.to_string()));
                }
                continue;
            }
            let entry = tensor_entry(name, value)?;
            if h.tensors.insert(name.clone(), entry).is_some() {
                return Err(Error::Safetensors(format!("duplicate tensor {name:?}")));
            }
        }
        h.check_coverage()?;
        Ok(h)
    }

    /// Length of the data region the header describes.
    pub fn data_len(&self) -> u64 {
        self.tensors.values().map(|t| t.end).max().unwrap_or(0)
    }

    /// The format requires the data region to be fully indexed: no holes, no
    /// overlaps.
    fn check_coverage(&self) -> Result<()> {
        let mut spans: Vec<(u64, u64, &str)> = self
            .tensors
            .iter()
            .map(|(n, t)| (t.begin, t.end, n.as_str()))
            .collect();
        spans.sort_unstable();
        let mut at = 0u64;
        let mut prev = "<start of data>";
        for (begin, end, name) in spans {
            if begin != at {
                let what = if begin > at { "hole" } else { "overlap" };
                return Err(Error::Safetensors(format!(
                    "{what} in the data region: {prev} ends at {at}, {name} begins at {begin}"
                )));
            }
            at = end;
            prev = name;
        }
        Ok(())
    }
}

fn tensor_entry(name: &str, v: &Json) -> Result<TensorEntry> {
    let bad = |m: &str| Error::Safetensors(format!("tensor {name:?}: {m}"));
    let pairs = v.as_object().ok_or_else(|| bad("entry is not an object"))?;
    let (mut dtype, mut shape, mut offsets) = (None, None, None);
    for (k, x) in pairs {
        let slot_taken = match k.as_str() {
            "dtype" => {
                let s = x.as_str().ok_or_else(|| bad("dtype is not a string"))?;
                let d = DType::parse(s).ok_or_else(|| bad(&format!("unknown dtype {s:?}")))?;
                dtype.replace(d).is_some()
            }
            "shape" => shape
                .replace(u64_list(x).ok_or_else(|| bad("shape is not a list of integers"))?)
                .is_some(),
            "data_offsets" => {
                let o = u64_list(x).ok_or_else(|| bad("data_offsets is not a list of integers"))?;
                if o.len() != 2 {
                    return Err(bad("data_offsets must have two entries"));
                }
                offsets.replace((o[0], o[1])).is_some()
            }
            other => return Err(bad(&format!("unknown field {other:?}"))),
        };
        if slot_taken {
            return Err(bad(&format!("duplicate field {k:?}")));
        }
    }
    let dtype = dtype.ok_or_else(|| bad("missing dtype"))?;
    let shape = shape.ok_or_else(|| bad("missing shape"))?;
    let (begin, end) = offsets.ok_or_else(|| bad("missing data_offsets"))?;
    if end < begin {
        return Err(bad(&format!("data_offsets [{begin}, {end}] are reversed")));
    }
    let want = shape
        .iter()
        .try_fold(dtype.size(), |acc, &d| acc.checked_mul(d))
        .ok_or_else(|| bad("shape overflows"))?;
    if end - begin != want {
        return Err(bad(&format!(
            "{} bytes declared, {dtype} {shape:?} needs {want}",
            end - begin
        )));
    }
    Ok(TensorEntry {
        dtype,
        shape,
        begin,
        end,
    })
}

fn u64_list(v: &Json) -> Option<Vec<u64>> {
    v.as_array()?.iter().map(Json::as_u64).collect()
}

/// One checkpoint file: its name, header, and (for a file on disk) where its
/// data region starts.
#[derive(Clone, Debug)]
pub struct Shard {
    /// File name within the checkpoint directory.
    pub file: String,
    pub header: Header,
    /// Absolute file offset of the data region: `8 + header_len`. Unknown for a
    /// shard read from a header bundle.
    pub data_start: Option<u64>,
}

/// Parse a header bundle: a JSON object mapping each shard's file name to its
/// safetensors header, as fetched from a model hub without the tensor data.
/// Shards come back sorted by file name; data starts are unknown.
pub fn parse_header_bundle(text: &str) -> Result<Vec<Shard>> {
    let v = json::parse(text).map_err(Error::Json)?;
    let pairs = v
        .as_object()
        .ok_or_else(|| Error::Safetensors("header bundle is not a JSON object".into()))?;
    let mut shards = Vec::with_capacity(pairs.len());
    for (file, header) in pairs {
        let header = Header::from_json(header).map_err(|e| match e {
            Error::Safetensors(m) => Error::Safetensors(format!("{file}: {m}")),
            other => other,
        })?;
        shards.push(Shard {
            file: file.clone(),
            header,
            data_start: None,
        });
    }
    shards.sort_by(|a, b| a.file.cmp(&b.file));
    if let Some(w) = shards.windows(2).find(|w| w[0].file == w[1].file) {
        return Err(Error::Safetensors(format!(
            "header bundle lists {} twice",
            w[0].file
        )));
    }
    Ok(shards)
}

/// Read and validate the header of one safetensors file.
pub fn read_file_header(path: &Path) -> Result<Shard> {
    let f = File::open(path).map_err(|e| io_err(path, e))?;
    let file_len = f.metadata().map_err(|e| io_err(path, e))?.len();
    let bad = |m: String| Error::Safetensors(format!("{}: {m}", path.display()));
    if file_len < 8 {
        return Err(bad(format!(
            "{file_len} bytes is shorter than the length prefix"
        )));
    }
    let mut len_buf = [0u8; 8];
    read_exact_at(&f, &mut len_buf, 0).map_err(|e| io_err(path, e))?;
    let header_len = u64::from_le_bytes(len_buf);
    if header_len == 0 || header_len > MAX_HEADER_BYTES || header_len > file_len - 8 {
        return Err(bad(format!(
            "header length {header_len} does not fit a {file_len}-byte file"
        )));
    }
    let mut buf = vec![0u8; header_len as usize];
    read_exact_at(&f, &mut buf, 8).map_err(|e| io_err(path, e))?;
    let text = std::str::from_utf8(&buf).map_err(|e| bad(format!("header is not UTF-8: {e}")))?;
    let header = Header::parse(text).map_err(|e| bad(e.to_string()))?;
    let data_start = 8 + header_len;
    if data_start + header.data_len() != file_len {
        return Err(bad(format!(
            "data region is {} bytes but the file holds {}",
            header.data_len(),
            file_len - data_start
        )));
    }
    let file = path
        .file_name()
        .and_then(|n| n.to_str())
        .ok_or_else(|| bad("file name is not UTF-8".into()))?
        .to_string();
    Ok(Shard {
        file,
        header,
        data_start: Some(data_start),
    })
}

/// A byte gather within one tensor: `count` runs of `len` bytes, run `i`
/// starting `offset + i * stride` bytes after the tensor's first byte.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Runs {
    pub offset: u64,
    pub len: u64,
    pub stride: u64,
    pub count: u64,
}

impl Runs {
    /// The whole tensor of `bytes` bytes.
    pub fn whole(bytes: u64) -> Runs {
        Runs {
            offset: 0,
            len: bytes,
            stride: bytes,
            count: 1,
        }
    }
    /// Bytes gathered.
    pub fn bytes(&self) -> u64 {
        self.len * self.count
    }
    /// Bytes from the first gathered byte to one past the last.
    pub fn span(&self) -> u64 {
        if self.count == 0 {
            0
        } else {
            (self.count - 1) * self.stride + self.len
        }
    }
    /// Whether the runs lie inside a tensor of `tensor_bytes` bytes without
    /// overlapping one another.
    pub fn fits(&self, tensor_bytes: u64) -> bool {
        (self.count <= 1 || self.stride >= self.len) && self.offset + self.span() <= tensor_bytes
    }
    /// Gather the runs out of `tensor`, the tensor's full bytes.
    pub fn gather(&self, tensor: &[u8]) -> Vec<u8> {
        let mut out = Vec::with_capacity(self.bytes() as usize);
        for i in 0..self.count {
            let a = (self.offset + i * self.stride) as usize;
            out.extend_from_slice(&tensor[a..a + self.len as usize]);
        }
        out
    }
}

/// A checkpoint directory: every `*.safetensors` file in it.
#[derive(Clone, Debug)]
pub struct Checkpoint {
    pub dir: PathBuf,
    pub shards: Vec<Shard>,
    /// Tensor name -> index into `shards`.
    names: BTreeMap<String, usize>,
}

impl Checkpoint {
    /// Open every `*.safetensors` file in `dir` (sorted by name). If
    /// `model.safetensors.index.json` is present, its `weight_map` must list
    /// exactly the tensors found, each in the file that holds it.
    pub fn open(dir: &Path) -> Result<Checkpoint> {
        let mut files = Vec::new();
        for e in std::fs::read_dir(dir).map_err(|e| io_err(dir, e))? {
            let e = e.map_err(|e| io_err(dir, e))?;
            let name = e.file_name();
            if let Some(name) = name.to_str() {
                if name.ends_with(".safetensors") && e.path().is_file() {
                    files.push(e.path());
                }
            }
        }
        files.sort();
        if files.is_empty() {
            return Err(Error::Safetensors(format!(
                "{}: no .safetensors files",
                dir.display()
            )));
        }
        let shards = files
            .iter()
            .map(|p| read_file_header(p))
            .collect::<Result<Vec<_>>>()?;
        let cp = Checkpoint::from_shards(dir, shards)?;
        let index = dir.join("model.safetensors.index.json");
        if index.is_file() {
            cp.check_index(&index)?;
        }
        Ok(cp)
    }

    fn from_shards(dir: &Path, shards: Vec<Shard>) -> Result<Checkpoint> {
        let mut names = BTreeMap::new();
        for (i, s) in shards.iter().enumerate() {
            for name in s.header.tensors.keys() {
                if let Some(j) = names.insert(name.clone(), i) {
                    return Err(Error::Safetensors(format!(
                        "{name} is in both {} and {}",
                        shards[j].file, s.file
                    )));
                }
            }
        }
        Ok(Checkpoint {
            dir: dir.to_path_buf(),
            shards,
            names,
        })
    }

    fn check_index(&self, path: &Path) -> Result<()> {
        let text = std::fs::read_to_string(path).map_err(|e| io_err(path, e))?;
        let v = json::parse(&text).map_err(Error::Json)?;
        let bad = |m: String| Error::Safetensors(format!("{}: {m}", path.display()));
        let map = v
            .get("weight_map")
            .and_then(Json::as_object)
            .ok_or_else(|| bad("no weight_map object".into()))?;
        json::check_unique_keys(v.get("weight_map").unwrap()).map_err(bad)?;
        for (name, file) in map {
            let file = file
                .as_str()
                .ok_or_else(|| bad(format!("{name}: file is not a string")))?;
            match self.names.get(name) {
                Some(&i) if self.shards[i].file == file => {}
                Some(&i) => {
                    return Err(bad(format!(
                        "{name} is indexed in {file} but found in {}",
                        self.shards[i].file
                    )))
                }
                None => return Err(bad(format!("{name} is indexed in {file} but not found"))),
            }
        }
        if map.len() != self.names.len() {
            return Err(bad(format!(
                "index lists {} tensors, the files hold {}",
                map.len(),
                self.names.len()
            )));
        }
        Ok(())
    }

    /// Every tensor name, sorted.
    pub fn names(&self) -> impl Iterator<Item = &str> {
        self.names.keys().map(String::as_str)
    }

    /// The shard holding `name` and its header entry.
    pub fn get(&self, name: &str) -> Option<(&Shard, &TensorEntry)> {
        let s = &self.shards[*self.names.get(name)?];
        Some((s, &s.header.tensors[name]))
    }

    /// Absolute byte range `[begin, end)` of `name` in its file.
    pub fn file_range(&self, name: &str) -> Result<(String, u64, u64)> {
        let (s, t) = self.get(name).ok_or_else(|| missing(name))?;
        let base = s
            .data_start
            .expect("a checkpoint opened from disk knows its data start");
        Ok((s.file.clone(), base + t.begin, base + t.end))
    }

    /// Read one tensor's bytes.
    pub fn read_tensor(&self, name: &str) -> Result<Vec<u8>> {
        let (_, t) = self.get(name).ok_or_else(|| missing(name))?;
        self.read_runs(name, &Runs::whole(t.byte_len()))
    }

    /// Read a byte gather of one tensor (see [`Runs`]).
    pub fn read_runs(&self, name: &str, runs: &Runs) -> Result<Vec<u8>> {
        let (shard, t) = self.get(name).ok_or_else(|| missing(name))?;
        if !runs.fits(t.byte_len()) {
            return Err(Error::Safetensors(format!(
                "{name}: runs {runs:?} do not fit its {} bytes",
                t.byte_len()
            )));
        }
        let path = self.dir.join(&shard.file);
        let f = File::open(&path).map_err(|e| io_err(&path, e))?;
        let at = shard.data_start.expect("opened from disk") + t.begin + runs.offset;
        let mut span = vec![0u8; runs.span() as usize];
        read_exact_at(&f, &mut span, at).map_err(|e| io_err(&path, e))?;
        if runs.count <= 1 || runs.stride == runs.len {
            return Ok(span);
        }
        Ok(Runs { offset: 0, ..*runs }.gather(&span))
    }
}

fn missing(name: &str) -> Error {
    Error::Safetensors(format!("no tensor named {name:?}"))
}

#[cfg(unix)]
fn read_exact_at(f: &File, buf: &mut [u8], offset: u64) -> std::io::Result<()> {
    use std::os::unix::fs::FileExt;
    f.read_exact_at(buf, offset)
}

#[cfg(not(unix))]
fn read_exact_at(f: &File, buf: &mut [u8], offset: u64) -> std::io::Result<()> {
    use std::io::{Read, Seek, SeekFrom};
    let mut f = f;
    f.seek(SeekFrom::Start(offset))?;
    f.read_exact(buf)
}

/// Serialize tensors into safetensors bytes (a header padded to 8 bytes, then
/// the data in the given order). For tests and small fixtures.
pub fn serialize(tensors: &[(&str, DType, &[u64], &[u8])], metadata: &[(&str, &str)]) -> Vec<u8> {
    let mut pairs = Vec::new();
    if !metadata.is_empty() {
        let m = metadata
            .iter()
            .map(|(k, v)| (k.to_string(), Json::Str(v.to_string())))
            .collect();
        pairs.push(("__metadata__".to_string(), Json::Object(m)));
    }
    let mut at = 0u64;
    for (name, dtype, shape, data) in tensors {
        let end = at + data.len() as u64;
        let entry = vec![
            ("dtype".to_string(), Json::Str(dtype.as_str().to_string())),
            (
                "shape".to_string(),
                Json::Array(shape.iter().map(|&d| Json::Int(d as i64)).collect()),
            ),
            (
                "data_offsets".to_string(),
                Json::Array(vec![Json::Int(at as i64), Json::Int(end as i64)]),
            ),
        ];
        pairs.push((name.to_string(), Json::Object(entry)));
        at = end;
    }
    let mut header = json::serialize(&Json::Object(pairs)).into_bytes();
    while !header.len().is_multiple_of(8) {
        header.push(b' ');
    }
    let mut out = (header.len() as u64).to_le_bytes().to_vec();
    out.extend_from_slice(&header);
    for (_, _, _, data) in tensors {
        out.extend_from_slice(data);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn entry_json(dtype: &str, shape: &str, a: u64, b: u64) -> String {
        format!(r#"{{"dtype":"{dtype}","shape":{shape},"data_offsets":[{a},{b}]}}"#)
    }

    #[test]
    fn parses_a_well_formed_header() {
        let text = format!(
            r#"{{"__metadata__":{{"format":"pt"}},"a":{},"b":{}}}"#,
            entry_json("BF16", "[2,3]", 0, 12),
            entry_json("F8_E4M3", "[4]", 12, 16)
        );
        let h = Header::parse(&text).unwrap();
        assert_eq!(h.metadata, vec![("format".to_string(), "pt".to_string())]);
        assert_eq!(h.tensors["a"].dtype, DType::BF16);
        assert_eq!(h.tensors["b"].byte_len(), 4);
        assert_eq!(h.data_len(), 16);
    }

    #[test]
    fn rejects_malformed_headers() {
        let good_b = entry_json("U8", "[4]", 12, 16);
        let cases = [
            // byte length disagrees with the shape
            format!(
                r#"{{"a":{},"b":{good_b}}}"#,
                entry_json("BF16", "[2,3]", 0, 11)
            ),
            // a hole between a and b
            format!(
                r#"{{"a":{},"b":{}}}"#,
                entry_json("BF16", "[2,3]", 0, 12),
                entry_json("U8", "[4]", 13, 17)
            ),
            // overlapping tensors
            format!(
                r#"{{"a":{},"b":{}}}"#,
                entry_json("BF16", "[2,3]", 0, 12),
                entry_json("U8", "[4]", 8, 12)
            ),
            // duplicate tensor
            format!(r#"{{"b":{good_b},"b":{good_b}}}"#),
            // unknown dtype
            format!(r#"{{"a":{}}}"#, entry_json("F4", "[4]", 0, 4)),
            // float offset
            r#"{"a":{"dtype":"U8","shape":[4],"data_offsets":[0,4.0]}}"#.to_string(),
            // unknown field
            r#"{"a":{"dtype":"U8","shape":[4],"data_offsets":[0,4],"x":1}}"#.to_string(),
            // data does not start at 0
            format!(r#"{{"b":{good_b}}}"#),
        ];
        for text in &cases {
            assert!(Header::parse(text).is_err(), "accepted: {text}");
        }
    }

    #[test]
    fn runs_gather_and_bounds() {
        let data: Vec<u8> = (0..32).collect();
        let r = Runs {
            offset: 2,
            len: 3,
            stride: 8,
            count: 4,
        };
        assert!(r.fits(32));
        assert_eq!(
            r.gather(&data),
            vec![2, 3, 4, 10, 11, 12, 18, 19, 20, 26, 27, 28]
        );
        assert!(!Runs {
            offset: 8,
            len: 3,
            stride: 8,
            count: 4
        }
        .fits(32));
        assert!(
            !Runs {
                offset: 0,
                len: 9,
                stride: 8,
                count: 2
            }
            .fits(32),
            "overlapping runs"
        );
        assert_eq!(Runs::whole(32).gather(&data), data);
    }
}
