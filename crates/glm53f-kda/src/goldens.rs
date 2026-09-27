//! Golden fixtures from the reference oracle (`oracle/`), and the KDA checks against them.
//!
//! # Format
//!
//! A set is a directory `oracle/goldens/<set>/` holding a `manifest.json` next to raw
//! little-endian `.bin` files. The manifest's `tensors` object maps names to
//! `{"file", "dtype", "shape", "sha256"}`; a manifest that is that map itself is accepted too.
//! Its `notes` object carries facts about the set, such as `state_heads`. Every file is checked
//! against its size and digest before use.
//!
//! # KDA tensors
//!
//! The oracle names its sets `layerNN-prefill` and `layerNN-decode`, and their tensors
//! `prefill.kda.q`, `decode.kda.core_out` and so on. A decode set's tensors are stacked over
//! the decode steps. A tensor belongs to a KDA layer when it names that layer, either as a
//! component of its own name (`layers.4.`, `L04.`, `layer4.`) or through its set's name.
//!
//! Its role is what remains after:
//!
//! - the phase prefix (`prefill.`, `decode.`);
//! - the layer component;
//! - module names (`kda`, `self_attn`, `linear_attn`, `forget_gate`).
//!
//! Per-step names (`decode.sN.`) are ignored. [`ROLES`] lists the roles and their names.
//!
//! # Checks
//!
//! - [`check_core`]: q, k, v (conv outputs), g (log decay) and beta against the core output,
//!   the state after the rows and, where recorded, the per-step states of a few heads.
//! - [`check_conv_cache`]: the conv cache after the rows against the last inputs.
//! - [`check_layer`]: the whole layer, from the projections and the layer's KDA weights, against
//!   the gated norm's output. It runs only where a set carries those weights. The oracle's sets
//!   record activations and list the weights they read by digest.
//!
//! The state before the rows is the set's `initial_state` if it has one. Otherwise, for a decode
//! set it is the matching prefill set's final state ([`Init::from_prefill`]), and for a prefill
//! set it is zero.

use std::fmt;
use std::path::{Path, PathBuf};

use crate::cpu::{self, ConvRounding, LayerParams, Rows};
use crate::json::{self, Value};
use crate::{bf16, channels, is_kda_layer, state_len, DK, DV, TAPS, WINDOW};

#[derive(Debug)]
pub enum GoldenError {
    Io(PathBuf, std::io::Error),
    Json(PathBuf, json::Error),
    Manifest(String),
    Digest {
        name: String,
        expected: String,
        actual: String,
    },
    Size {
        name: String,
        expected: usize,
        actual: usize,
    },
    Missing(String),
    Shape(String),
}

impl fmt::Display for GoldenError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            GoldenError::Io(p, e) => write!(f, "{}: {e}", p.display()),
            GoldenError::Json(p, e) => write!(f, "{}: {e}", p.display()),
            GoldenError::Manifest(m) => write!(f, "manifest: {m}"),
            GoldenError::Digest {
                name,
                expected,
                actual,
            } => {
                write!(
                    f,
                    "{name}: sha256 {actual} does not match the manifest's {expected}"
                )
            }
            GoldenError::Size {
                name,
                expected,
                actual,
            } => {
                write!(
                    f,
                    "{name}: {actual} bytes, the manifest's dtype and shape need {expected}"
                )
            }
            GoldenError::Missing(m) => write!(f, "missing tensor: {m}"),
            GoldenError::Shape(m) => write!(f, "shape: {m}"),
        }
    }
}

impl std::error::Error for GoldenError {}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DType {
    F64,
    F32,
    F16,
    BF16,
    I64,
    I32,
    U8,
}

impl DType {
    pub fn parse(s: &str) -> Option<DType> {
        Some(match s.to_ascii_lowercase().trim_start_matches("torch.") {
            "f64" | "float64" | "double" => DType::F64,
            "f32" | "float32" | "float" => DType::F32,
            "f16" | "float16" | "half" => DType::F16,
            "bf16" | "bfloat16" => DType::BF16,
            "i64" | "int64" | "long" => DType::I64,
            "i32" | "int32" | "int" => DType::I32,
            "u8" | "uint8" | "bool" => DType::U8,
            _ => return None,
        })
    }

    pub fn size(self) -> usize {
        match self {
            DType::F64 | DType::I64 => 8,
            DType::F32 | DType::I32 => 4,
            DType::F16 | DType::BF16 => 2,
            DType::U8 => 1,
        }
    }

    fn manifest_name(self) -> &'static str {
        match self {
            DType::F64 => "f64",
            DType::F32 => "f32",
            DType::F16 => "f16",
            DType::BF16 => "bf16",
            DType::I64 => "i64",
            DType::I32 => "i32",
            DType::U8 => "u8",
        }
    }
}

/// One manifest entry.
#[derive(Clone, Debug)]
pub struct Entry {
    pub name: String,
    pub file: String,
    pub dtype: DType,
    pub shape: Vec<usize>,
    pub sha256: String,
}

impl Entry {
    pub fn elements(&self) -> usize {
        self.shape.iter().product()
    }
}

/// A tensor read from a set, widened to f32 (integers converted as values).
#[derive(Clone, Debug)]
pub struct Tensor {
    pub name: String,
    pub dtype: DType,
    pub shape: Vec<usize>,
    pub data: Vec<f32>,
}

/// The phase a set records.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Phase {
    /// The prompt in one forward (the reference's chunked KDA path).
    Prefill,
    /// One token per forward, continuing from the prompt (the recurrent path).
    Decode,
}

/// A golden set: a directory and its manifest.
#[derive(Clone, Debug)]
pub struct Set {
    pub dir: PathBuf,
    pub entries: Vec<Entry>,
    /// The manifest's `notes` object, if any.
    pub notes: Option<Value>,
}

impl Set {
    /// Read `dir/manifest.json`.
    pub fn load(dir: &Path) -> Result<Set, GoldenError> {
        let path = dir.join("manifest.json");
        let text = std::fs::read_to_string(&path).map_err(|e| GoldenError::Io(path.clone(), e))?;
        let doc = json::parse(&text).map_err(|e| GoldenError::Json(path.clone(), e))?;
        let map = doc.get("tensors").unwrap_or(&doc);
        let members = map
            .as_object()
            .ok_or_else(|| GoldenError::Manifest("not a JSON object".into()))?;
        let mut entries = Vec::new();
        for (name, v) in members {
            // Other members (metadata) are skipped; tensor entries have a "file".
            let Some(file) = v.get("file").and_then(Value::as_str) else {
                continue;
            };
            let bad = |what: &str| GoldenError::Manifest(format!("{name}: {what}"));
            let dtype = v
                .get("dtype")
                .and_then(Value::as_str)
                .ok_or_else(|| bad("no dtype"))?;
            let dtype = DType::parse(dtype).ok_or_else(|| bad("unknown dtype"))?;
            let shape = v
                .get("shape")
                .and_then(Value::as_array)
                .ok_or_else(|| bad("no shape"))?
                .iter()
                .map(|x| x.as_usize().ok_or_else(|| bad("bad shape")))
                .collect::<Result<Vec<_>, _>>()?;
            let sha256 = v
                .get("sha256")
                .and_then(Value::as_str)
                .ok_or_else(|| bad("no sha256"))?
                .to_ascii_lowercase();
            if file.contains("..") || file.starts_with('/') {
                return Err(bad("file must stay inside the set"));
            }
            entries.push(Entry {
                name: name.clone(),
                file: file.to_string(),
                dtype,
                shape,
                sha256,
            });
        }
        Ok(Set {
            dir: dir.to_path_buf(),
            entries,
            notes: doc.get("notes").cloned(),
        })
    }

    /// The set's directory name.
    pub fn name(&self) -> String {
        self.dir
            .file_name()
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_default()
    }

    /// The layer and phase the set's name gives (`layerNN-prefill`, `layerNN-decode`).
    pub fn layer_phase(&self) -> Option<(usize, Phase)> {
        let name = self.name();
        let (layer, phase) = name.strip_prefix("layer")?.split_once('-')?;
        let phase = match phase {
            "prefill" => Phase::Prefill,
            "decode" => Phase::Decode,
            _ => return None,
        };
        Some((layer.parse().ok()?, phase))
    }

    /// `notes.state_heads`: the heads whose per-step states a decode set records.
    pub fn state_heads(&self) -> Option<Vec<usize>> {
        self.notes
            .as_ref()?
            .get("state_heads")?
            .as_array()?
            .iter()
            .map(Value::as_usize)
            .collect()
    }

    pub fn entry(&self, name: &str) -> Option<&Entry> {
        self.entries.iter().find(|e| e.name == name)
    }

    /// The raw bytes of `name`, checked against the manifest's size and digest.
    pub fn bytes(&self, name: &str) -> Result<Vec<u8>, GoldenError> {
        let e = self
            .entry(name)
            .ok_or_else(|| GoldenError::Missing(name.to_string()))?;
        let path = self.dir.join(&e.file);
        let data = std::fs::read(&path).map_err(|err| GoldenError::Io(path, err))?;
        let expected = e.elements() * e.dtype.size();
        if data.len() != expected {
            return Err(GoldenError::Size {
                name: name.to_string(),
                expected,
                actual: data.len(),
            });
        }
        let actual = crate::sha256::hex(&data);
        if actual != e.sha256 {
            return Err(GoldenError::Digest {
                name: name.to_string(),
                expected: e.sha256.clone(),
                actual,
            });
        }
        Ok(data)
    }

    /// `name`, widened to f32.
    pub fn read(&self, name: &str) -> Result<Tensor, GoldenError> {
        let e = self
            .entry(name)
            .ok_or_else(|| GoldenError::Missing(name.to_string()))?
            .clone();
        let b = self.bytes(name)?;
        let data: Vec<f32> = match e.dtype {
            DType::F64 => b
                .as_chunks::<8>()
                .0
                .iter()
                .map(|c| f64::from_le_bytes(*c) as f32)
                .collect(),
            DType::F32 => b
                .as_chunks::<4>()
                .0
                .iter()
                .map(|c| f32::from_le_bytes(*c))
                .collect(),
            DType::F16 => b
                .as_chunks::<2>()
                .0
                .iter()
                .map(|c| f16_to_f32(u16::from_le_bytes(*c)))
                .collect(),
            DType::BF16 => b
                .as_chunks::<2>()
                .0
                .iter()
                .map(|c| bf16::to_f32(u16::from_le_bytes(*c)))
                .collect(),
            DType::I64 => b
                .as_chunks::<8>()
                .0
                .iter()
                .map(|c| i64::from_le_bytes(*c) as f32)
                .collect(),
            DType::I32 => b
                .as_chunks::<4>()
                .0
                .iter()
                .map(|c| i32::from_le_bytes(*c) as f32)
                .collect(),
            DType::U8 => b.iter().map(|&x| x as f32).collect(),
        };
        Ok(Tensor {
            name: e.name,
            dtype: e.dtype,
            shape: e.shape,
            data,
        })
    }

    /// Check every entry's size and digest.
    pub fn verify_all(&self) -> Result<(), GoldenError> {
        for e in &self.entries {
            self.bytes(&e.name)?;
        }
        Ok(())
    }
}

fn f16_to_f32(h: u16) -> f32 {
    let sign = ((h >> 15) as u32) << 31;
    let exp = ((h >> 10) & 0x1f) as u32;
    let man = (h & 0x3ff) as u32;
    let bits = match (exp, man) {
        (0, 0) => sign,
        (0, m) => {
            // Subnormal: normalize.
            let mut e = 127 - 15 + 1;
            let mut m = m;
            while m & 0x400 == 0 {
                m <<= 1;
                e -= 1;
            }
            sign | (e << 23) | ((m & 0x3ff) << 13)
        }
        (0x1f, m) => sign | 0x7f80_0000 | (m << 13),
        (e, m) => sign | ((e + 127 - 15) << 23) | (m << 13),
    };
    f32::from_bits(bits)
}

/// The directory that holds the sets: `GLM53F_GOLDENS` if set, else the repository's
/// `oracle/goldens`.
pub fn default_root() -> PathBuf {
    match std::env::var_os("GLM53F_GOLDENS") {
        Some(p) => PathBuf::from(p),
        None => Path::new(env!("CARGO_MANIFEST_DIR")).join("../../oracle/goldens"),
    }
}

/// The sets under `root` (subdirectories with a `manifest.json`), sorted; empty when `root`
/// does not exist.
pub fn discover(root: &Path) -> Vec<PathBuf> {
    let Ok(rd) = std::fs::read_dir(root) else {
        return Vec::new();
    };
    let mut sets: Vec<PathBuf> = rd
        .filter_map(Result::ok)
        .map(|e| e.path())
        .filter(|p| p.join("manifest.json").is_file())
        .collect();
    sets.sort();
    sets
}

/// The KDA tensors a check can use.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Role {
    /// q after the conv, before the L2 norm: `[T, H, DK]`.
    Q,
    /// k after the conv: `[T, H, DK]`.
    K,
    /// v after the conv: `[T, H, DV]`.
    V,
    /// Log-space decay (the forget gate's output): `[T, H, DK]`.
    LogDecay,
    /// `sigmoid(b_proj(x))`: `[T, H]`.
    Beta,
    /// Recurrent state before the rows, reference layout `[H, DK, DV]`.
    StateIn,
    /// Recurrent state after the rows, reference layout `[H, DK, DV]`.
    StateOut,
    /// The states after each row for the heads in `notes.state_heads`: `[T, n, DK, DV]`.
    StateSteps,
    /// The delta rule's read-out before the gated norm: `[T, H, DV]`.
    CoreOut,
    /// The reference's KDA path: 0 chunked, 1 recurrent (`[1]` or one per row).
    Path,
    /// q | k | v projections before the conv: `[T, 3 * H * DK]`.
    MixedQkv,
    /// The conv cache before the rows, reference layout `[C, W]`, oldest first, W >= 3.
    ConvStateIn,
    /// The conv cache after the rows, `[C, W]`.
    ConvStateOut,
    /// `f_b_proj(f_a_proj(x))`: `[T, H * DK]`.
    ForgetProj,
    /// `b_proj(x)`: `[T, H]`.
    BetaLogits,
    /// `g_b_proj(g_a_proj(x))`: `[T, H * DV]`.
    GateProj,
    /// The gated RMSNorm's output (`o_proj`'s input): `[T, H * DV]`.
    NormOut,
    /// The depthwise conv weight over q | k | v: `[3 * H * DK, 1, TAPS]` or `[3 * H * DK, TAPS]`.
    ConvW,
    /// Per-projection conv weights, as the checkpoint names them.
    ConvWQ,
    ConvWK,
    ConvWV,
    ALog,
    DtBias,
    NormW,
}

/// The names each role is recognized by (lowercase, after the phase, layer and module names).
pub const ROLES: &[(Role, &[&str])] = &[
    (Role::Q, &["q", "query"]),
    (Role::K, &["k", "key"]),
    (Role::V, &["v", "value"]),
    (Role::LogDecay, &["g", "log_decay"]),
    (Role::Beta, &["beta"]),
    (Role::StateIn, &["initial_state", "state_in"]),
    (
        Role::StateOut,
        &[
            "state",
            "state_final",
            "final_state",
            "state_out",
            "last_recurrent_state",
        ],
    ),
    (Role::StateSteps, &["state_heads"]),
    (Role::CoreOut, &["core_out", "core_attn_out"]),
    (Role::Path, &["path"]),
    (Role::MixedQkv, &["qkv_preconv", "mixed_qkv"]),
    (Role::ConvStateIn, &["conv_state_in"]),
    (
        Role::ConvStateOut,
        &["conv_state", "conv_state_final", "conv_state_out"],
    ),
    (Role::ForgetProj, &["f_proj", "forget_proj", "f_b_proj"]),
    (Role::BetaLogits, &["b_logits", "b_proj", "beta_logits"]),
    (Role::GateProj, &["gate", "g_proj", "g_b_proj"]),
    (Role::NormOut, &["norm_out", "o_norm_out"]),
    (Role::ConvW, &["conv1d.weight", "conv_w"]),
    (Role::ConvWQ, &["q_conv1d.weight"]),
    (Role::ConvWK, &["k_conv1d.weight"]),
    (Role::ConvWV, &["v_conv1d.weight"]),
    (Role::ALog, &["a_log"]),
    (Role::DtBias, &["dt_bias"]),
    (Role::NormW, &["o_norm.weight", "norm_w"]),
];

const MODULES: &[&str] = &["kda", "self_attn", "linear_attn", "forget_gate"];

/// The (layer, role) of a tensor name. `set_layer` is the layer the set's name gives, used when
/// the tensor name holds none. Only KDA layers qualify.
pub fn classify(name: &str, set_layer: Option<usize>) -> Option<(usize, Role)> {
    let lower = name.to_ascii_lowercase();
    let mut parts: Vec<&str> = lower.split(['.', '/']).collect();
    if matches!(parts.first(), Some(&"prefill") | Some(&"decode")) {
        parts.remove(0);
        // Per-step tensors (`decode.sN.<name>`) are not used.
        if parts.first().is_some_and(|p| {
            p.len() > 1 && p.starts_with('s') && p[1..].bytes().all(|b| b.is_ascii_digit())
        }) {
            return None;
        }
    }
    let found = parts.iter().enumerate().find_map(|(i, p)| {
        let digits = p
            .strip_prefix("layer")
            .or_else(|| p.strip_prefix('l'))
            .unwrap_or(p);
        (!digits.is_empty() && digits.bytes().all(|b| b.is_ascii_digit()))
            .then(|| digits.parse::<usize>().ok().map(|n| (i, n)))
            .flatten()
    });
    let (rest, layer) = match found {
        Some((i, n)) => (&parts[i + 1..], n),
        None => (&parts[..], set_layer?),
    };
    if !is_kda_layer(layer) {
        return None;
    }
    let rest: Vec<&str> = rest
        .iter()
        .copied()
        .filter(|p| !MODULES.contains(p))
        .collect();
    let key = rest.join(".");
    ROLES
        .iter()
        .find(|(_, names)| names.contains(&key.as_str()))
        .map(|(r, _)| (layer, *r))
}

/// One KDA layer's tensors in a set.
#[derive(Clone, Debug)]
pub struct LayerFixtures {
    pub layer: usize,
    /// The phase the set's name gives, if any.
    pub phase: Option<Phase>,
    pub tensors: Vec<(Role, String)>,
}

impl LayerFixtures {
    pub fn name(&self, role: Role) -> Option<&str> {
        self.tensors
            .iter()
            .find(|(r, _)| *r == role)
            .map(|(_, n)| n.as_str())
    }

    pub fn has(&self, roles: &[Role]) -> bool {
        roles.iter().all(|r| self.name(*r).is_some())
    }
}

/// The KDA layers a set has tensors for, by layer.
pub fn kda_layers(set: &Set) -> Vec<LayerFixtures> {
    let (set_layer, phase) = match set.layer_phase() {
        Some((l, p)) => (Some(l), Some(p)),
        None => (None, None),
    };
    let mut layers: Vec<LayerFixtures> = Vec::new();
    for e in &set.entries {
        let Some((layer, role)) = classify(&e.name, set_layer) else {
            continue;
        };
        match layers.iter_mut().find(|l| l.layer == layer) {
            Some(l) => {
                if l.name(role).is_none() {
                    l.tensors.push((role, e.name.clone()));
                }
            }
            None => layers.push(LayerFixtures {
                layer,
                phase: if Some(layer) == set_layer {
                    phase
                } else {
                    None
                },
                tensors: vec![(role, e.name.clone())],
            }),
        }
    }
    layers.sort_by_key(|l| l.layer);
    layers
}

/// What precedes a set's rows when the set does not record it.
#[derive(Clone, Debug, Default)]
pub struct Init {
    /// Recurrent state, reference layout `[H, DK, DV]`.
    pub state: Option<Vec<f32>>,
    /// Conv cache, reference layout `[C, W]`.
    pub conv: Option<Vec<f32>>,
}

impl Init {
    /// For a decode set: the state and conv cache after the prompt, from the matching
    /// `layerNN-prefill` set next to it. Empty for any other set, or when there is none.
    pub fn from_prefill(set: &Set, layer: usize) -> Result<Init, GoldenError> {
        if set.layer_phase() != Some((layer, Phase::Decode)) {
            return Ok(Init::default());
        }
        let Some(dir) = set
            .dir
            .parent()
            .map(|p| p.join(format!("layer{layer:02}-prefill")))
        else {
            return Ok(Init::default());
        };
        if !dir.join("manifest.json").is_file() {
            return Ok(Init::default());
        }
        let prefill = Set::load(&dir)?;
        let Some(lf) = kda_layers(&prefill).into_iter().find(|l| l.layer == layer) else {
            return Ok(Init::default());
        };
        let read = |role| -> Result<Option<Vec<f32>>, GoldenError> {
            match lf.name(role) {
                Some(n) => Ok(Some(prefill.read(n)?.data)),
                None => Ok(None),
            }
        };
        Ok(Init {
            state: read(Role::StateOut)?,
            conv: read(Role::ConvStateOut)?,
        })
    }
}

/// A comparison of a computed tensor against its golden.
#[derive(Clone, Debug)]
pub struct Comparison {
    pub what: String,
    pub elements: usize,
    /// Largest absolute difference.
    pub max_abs: f32,
    /// Largest absolute golden value.
    pub max_ref: f32,
    /// Largest difference in bfloat16 units in the last place.
    pub max_ulp: u32,
    /// Elements that differ at all.
    pub differing: usize,
    /// The tolerance applied: `max_abs <= tol * max(max_ref, 1e-3)`.
    pub tol: f32,
}

impl Comparison {
    pub fn compare(what: &str, got: &[f32], want: &[f32], tol: f32) -> Comparison {
        assert_eq!(got.len(), want.len(), "{what}: length");
        let mut c = Comparison {
            what: what.to_string(),
            elements: got.len(),
            max_abs: 0.0,
            max_ref: 0.0,
            max_ulp: 0,
            differing: 0,
            tol,
        };
        for (&g, &w) in got.iter().zip(want) {
            let d = (g - w).abs();
            if d.is_nan() {
                c.max_abs = f32::INFINITY;
            } else {
                c.max_abs = c.max_abs.max(d);
            }
            c.max_ref = c.max_ref.max(w.abs());
            c.max_ulp = c
                .max_ulp
                .max(bf16::ulp_distance(bf16::from_f32(g), bf16::from_f32(w)));
            if g.to_bits() != w.to_bits() {
                c.differing += 1;
            }
        }
        c
    }

    pub fn passes(&self) -> bool {
        self.max_abs <= self.tol * self.max_ref.max(1e-3)
    }
}

impl fmt::Display for Comparison {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}: {} elements, max |diff| {:.3e} (max |ref| {:.3e}, tolerance {:.1e} relative), max {} bf16 ulp, {} differ",
            self.what, self.elements, self.max_abs, self.max_ref, self.tol, self.max_ulp, self.differing
        )
    }
}

/// Tolerance for f32 goldens (the oracle's primary contract): the reference and this crate
/// evaluate the same f32 recurrence, or its chunked form, and differ by rounding only (the
/// chunked form measures about 1e-5 against the recurrence, see `src/chunked.rs`).
pub const TOL_F32: f32 = 1e-4;
/// Tolerance for bfloat16 goldens taken on the per-token path: a bfloat16 rounding flip.
pub const TOL_RECURRENT: f32 = 1e-2;
/// Tolerance for bfloat16 goldens taken on the chunked path, which a fused library kernel may
/// compute with bfloat16 intermediates.
pub const TOL_CHUNKED: f32 = 5e-2;

fn tensor(set: &Set, lf: &LayerFixtures, role: Role) -> Result<Tensor, GoldenError> {
    let name = lf
        .name(role)
        .ok_or_else(|| GoldenError::Missing(format!("layer {} {:?}", lf.layer, role)))?;
    set.read(name)
}

fn optional(set: &Set, lf: &LayerFixtures, role: Role) -> Result<Option<Tensor>, GoldenError> {
    match lf.name(role) {
        Some(_) => tensor(set, lf, role).map(Some),
        None => Ok(None),
    }
}

fn expect_len(t: &Tensor, n: usize) -> Result<(), GoldenError> {
    if t.data.len() != n {
        return Err(GoldenError::Shape(format!(
            "{}: {} elements, expected {n}",
            t.name,
            t.data.len()
        )));
    }
    Ok(())
}

/// Whether a set's rows came from the recurrent path: its `path` tensor says so, else one row.
fn recurrent(set: &Set, lf: &LayerFixtures, rows: usize) -> Result<bool, GoldenError> {
    Ok(match optional(set, lf, Role::Path)? {
        Some(p) => p.data.iter().all(|&x| x == 1.0),
        None => rows == 1,
    })
}

/// The tolerance tier for a golden of `dtype` (f32 or not) on the given path.
fn tolerance(dtype: DType, recurrent: bool) -> f32 {
    match (dtype, recurrent) {
        (DType::F32 | DType::F64, _) => TOL_F32,
        (_, true) => TOL_RECURRENT,
        (_, false) => TOL_CHUNKED,
    }
}

/// The inputs of the recurrent core, as the reference passes them to the delta-rule function.
struct CoreInputs {
    heads: usize,
    rows: usize,
    q: Tensor,
    k: Tensor,
    v: Tensor,
    g: Tensor,
    beta: Tensor,
    /// Reference layout `[H, DK, DV]`.
    state: Vec<f32>,
}

fn core_inputs(set: &Set, lf: &LayerFixtures, init: &Init) -> Result<CoreInputs, GoldenError> {
    let q = tensor(set, lf, Role::Q)?;
    let k = tensor(set, lf, Role::K)?;
    let v = tensor(set, lf, Role::V)?;
    let g = tensor(set, lf, Role::LogDecay)?;
    let beta = tensor(set, lf, Role::Beta)?;
    // q: T * H * DK elements, beta: T * H.
    if beta.data.is_empty() || q.data.len() != beta.data.len() * DK {
        return Err(GoldenError::Shape(format!(
            "{} and {} disagree on T * H",
            q.name, beta.name
        )));
    }
    let th = beta.data.len();
    // H: beta's last dimension ([.., T, H]), else q's second to last ([.., T, H, DK]).
    let heads = match (beta.shape.len(), q.shape.len()) {
        (nb, _) if nb >= 2 => beta.shape[nb - 1],
        (_, nq) if nq >= 2 && q.shape[nq - 1] == DK => q.shape[nq - 2],
        _ => {
            return Err(GoldenError::Shape(format!(
                "{}: cannot tell the number of heads",
                q.name
            )))
        }
    };
    if heads == 0 || th % heads != 0 {
        return Err(GoldenError::Shape(format!(
            "{}: {heads} heads do not divide T * H = {th}",
            q.name
        )));
    }
    for x in [&k, &g] {
        expect_len(x, th * DK)?;
    }
    expect_len(&v, th * DV)?;
    let state = match optional(set, lf, Role::StateIn)? {
        Some(s) => {
            expect_len(&s, state_len(heads))?;
            s.data
        }
        None => match &init.state {
            Some(s) if s.len() == state_len(heads) => s.clone(),
            Some(s) => {
                return Err(GoldenError::Shape(format!(
                    "initial state: {} elements, expected {}",
                    s.len(),
                    state_len(heads)
                )))
            }
            None => vec![0.0; state_len(heads)],
        },
    };
    Ok(CoreInputs {
        heads,
        rows: th / heads,
        q,
        k,
        v,
        g,
        beta,
        state,
    })
}

/// The recurrent core against a golden: q, k, v, g, beta through [`cpu::literal::step`] row
/// by row, compared with the core output, the state after the rows and, when the set records
/// them, the per-step states of `notes.state_heads`.
pub fn check_core(
    set: &Set,
    lf: &LayerFixtures,
    init: &Init,
) -> Result<Vec<Comparison>, GoldenError> {
    let c = core_inputs(set, lf, init)?;
    let core = tensor(set, lf, Role::CoreOut)?;
    expect_len(&core, c.rows * c.heads * DV)?;
    let steps = optional(set, lf, Role::StateSteps)?;
    let step_heads = set.state_heads();
    let mut state = c.state;
    let mut y = vec![0.0f32; c.rows * c.heads * DV];
    let mut per_step: Vec<f32> = Vec::new();
    for r in 0..c.rows {
        for h in 0..c.heads {
            let i = r * c.heads + h;
            let o = cpu::literal::step(
                &mut state[h * DK * DV..(h + 1) * DK * DV],
                &c.q.data[i * DK..(i + 1) * DK],
                &c.k.data[i * DK..(i + 1) * DK],
                &c.v.data[i * DV..(i + 1) * DV],
                &c.g.data[i * DK..(i + 1) * DK],
                c.beta.data[i],
            );
            y[i * DV..(i + 1) * DV].copy_from_slice(&o);
        }
        if let (Some(_), Some(hs)) = (&steps, &step_heads) {
            for &h in hs {
                per_step.extend_from_slice(&state[h * DK * DV..(h + 1) * DK * DV]);
            }
        }
    }
    let rec = recurrent(set, lf, c.rows)?;
    let tol = tolerance(core.dtype, rec);
    let path = if rec { "recurrent" } else { "chunked" };
    let label = |what: &str| format!("layer {} {what} ({} rows, {path} path)", lf.layer, c.rows);
    // The reference returns the core output in the activations' dtype.
    let y = if core.dtype == DType::BF16 {
        y.iter().map(|&x| bf16::round(x)).collect()
    } else {
        y
    };
    let mut out = vec![Comparison::compare(
        &label("core output"),
        &y,
        &core.data,
        tol,
    )];
    if let Some(so) = optional(set, lf, Role::StateOut)? {
        expect_len(&so, state_len(c.heads))?;
        out.push(Comparison::compare(
            &label("state after the rows"),
            &state,
            &so.data,
            tol,
        ));
    }
    if let (Some(st), Some(hs)) = (steps, step_heads) {
        expect_len(&st, c.rows * hs.len() * DK * DV)?;
        out.push(Comparison::compare(
            &label(&format!("per-row states of heads {hs:?}")),
            &per_step,
            &st.data,
            tol,
        ));
    }
    Ok(out)
}

/// The conv cache after the rows against the last `W` of the cache before them followed by the
/// rows' q | k | v projections, per channel (reference layout `[C, W]`, oldest first). This is
/// data movement: it pins the cache layout, and must match exactly.
pub fn check_conv_cache(
    set: &Set,
    lf: &LayerFixtures,
    init: &Init,
) -> Result<Option<Comparison>, GoldenError> {
    let (Some(_), Some(_)) = (lf.name(Role::MixedQkv), lf.name(Role::ConvStateOut)) else {
        return Ok(None);
    };
    let after = tensor(set, lf, Role::ConvStateOut)?;
    let qkv = tensor(set, lf, Role::MixedQkv)?;
    let n = after.shape.len();
    if n < 2 || after.shape[n - 1] < WINDOW {
        return Err(GoldenError::Shape(format!(
            "{}: expected [C, W >= {WINDOW}]",
            after.name
        )));
    }
    let (c, w) = (after.shape[n - 2], after.shape[n - 1]);
    if c == 0 || qkv.data.len() % c != 0 {
        return Err(GoldenError::Shape(format!(
            "{}: not a multiple of {c} channels",
            qkv.name
        )));
    }
    let t = qkv.data.len() / c;
    let before = match optional(set, lf, Role::ConvStateIn)? {
        Some(b) => b.data,
        None => init.conv.clone().unwrap_or_else(|| vec![0.0; c * w]),
    };
    if before.len() != c * w {
        return Err(GoldenError::Shape(format!(
            "conv cache before the rows: {} elements, expected {}",
            before.len(),
            c * w
        )));
    }
    let mut want = vec![0.0f32; c * w];
    for ch in 0..c {
        // The sequence [before (w); rows (t)]; keep its last w.
        for j in 0..w {
            let s = t + j; // index into the sequence of length w + t
            want[ch * w + j] = if s < w {
                before[ch * w + s]
            } else {
                qkv.data[(s - w) * c + ch]
            };
        }
    }
    Ok(Some(Comparison::compare(
        &format!("layer {} conv cache after {t} rows", lf.layer),
        &want,
        &after.data,
        0.0,
    )))
}

/// A golden's rows as kernel replay inputs, with the states around them.
#[derive(Clone, Debug)]
pub struct ReplayCase {
    /// Normalized k ([`cpu::l2norm`]), v rounded to bfloat16 (the kernels keep v in bfloat16),
    /// `exp(g)`, beta.
    pub saves: cpu::Saves,
    /// The state before the rows, kernel layout `[H][DV][DK]`.
    pub state: Vec<f32>,
    /// The golden state after the rows, kernel layout, if the set has it.
    pub after: Option<Vec<f32>>,
}

/// Kernel-format replay inputs for a golden's rows, for checking the device recurrence on the
/// oracle's activations. With f32 goldens the bfloat16 v costs about 1e-3 of the state.
pub fn replay_inputs(
    set: &Set,
    lf: &LayerFixtures,
    init: &Init,
) -> Result<ReplayCase, GoldenError> {
    let c = core_inputs(set, lf, init)?;
    let (h, rows) = (c.heads, c.rows);
    let mut s = cpu::Saves::zeros(h, rows);
    for i in 0..rows * h {
        let k: [f32; DK] = c.k.data[i * DK..(i + 1) * DK].try_into().unwrap();
        s.k[i * DK..(i + 1) * DK].copy_from_slice(&cpu::l2norm(&k, None));
        for d in 0..DK {
            s.g[i * DK + d] = c.g.data[i * DK + d].exp();
            s.v[i * DV + d] = bf16::round(c.v.data[i * DV + d]);
        }
        s.beta[i] = c.beta.data[i];
    }
    let after = optional(set, lf, Role::StateOut)?
        .map(|t| cpu::transpose_state(&t.data))
        .filter(|x| x.len() == state_len(h));
    Ok(ReplayCase {
        saves: s,
        state: cpu::transpose_state(&c.state),
        after,
    })
}

/// The inputs of a full-layer golden, in this crate's layouts.
#[derive(Clone, Debug)]
pub struct LayerCase {
    pub params: LayerParams,
    /// `[WINDOW][C]`.
    pub conv: Vec<f32>,
    /// `[H][DV][DK]`.
    pub state: Vec<f32>,
    pub rows: Rows,
    /// Golden gated-norm output `[T][H * DV]`.
    pub norm_out: Vec<f32>,
    /// Golden state after the rows `[H][DV][DK]`, if the set has it.
    pub state_out: Option<Vec<f32>>,
}

/// Assemble a full-layer case (projections, weights, caches and the norm output), converting
/// the reference's layouts to this crate's.
pub fn layer_case(
    set: &Set,
    lf: &LayerFixtures,
    init: &Init,
    eps: f32,
    lower: f32,
) -> Result<LayerCase, GoldenError> {
    let a_log = tensor(set, lf, Role::ALog)?.data;
    let heads = a_log.len();
    let c = channels(heads);
    let conv_w = if lf.name(Role::ConvW).is_some() {
        tensor(set, lf, Role::ConvW)?.data
    } else {
        let mut w = Vec::with_capacity(c * TAPS);
        for r in [Role::ConvWQ, Role::ConvWK, Role::ConvWV] {
            w.extend(tensor(set, lf, r)?.data);
        }
        w
    };
    if conv_w.len() != c * TAPS {
        return Err(GoldenError::Shape(format!(
            "layer {} conv weight: {} elements, expected {}",
            lf.layer,
            conv_w.len(),
            c * TAPS
        )));
    }
    let dt_bias = tensor(set, lf, Role::DtBias)?.data;
    let norm_w = tensor(set, lf, Role::NormW)?.data;
    let qkv = tensor(set, lf, Role::MixedQkv)?;
    if qkv.data.len() % c != 0 {
        return Err(GoldenError::Shape(format!(
            "{}: not a multiple of {c}",
            qkv.name
        )));
    }
    let t = qkv.data.len() / c;
    let a = tensor(set, lf, Role::ForgetProj)?;
    let b = tensor(set, lf, Role::BetaLogits)?;
    let gate = tensor(set, lf, Role::GateProj)?;
    let norm_out = tensor(set, lf, Role::NormOut)?;
    expect_len(&a, t * heads * DK)?;
    expect_len(&b, t * heads)?;
    expect_len(&gate, t * heads * DV)?;
    expect_len(&norm_out, t * heads * DV)?;
    // The conv cache: [C][W], oldest first; the window is its last WINDOW columns.
    let cache = match optional(set, lf, Role::ConvStateIn)? {
        Some(cs) => Some(cs.data),
        None => init.conv.clone(),
    };
    let conv = match cache {
        Some(cs) => {
            if cs.len() % c != 0 || cs.len() / c < WINDOW {
                return Err(GoldenError::Shape(format!(
                    "conv cache: expected [{c}][W >= {WINDOW}]"
                )));
            }
            let w = cs.len() / c;
            let mut win = vec![0.0f32; WINDOW * c];
            for j in 0..WINDOW {
                for ch in 0..c {
                    win[j * c + ch] = cs[ch * w + (w - WINDOW + j)];
                }
            }
            win
        }
        None => vec![0.0; WINDOW * c],
    };
    let state = match optional(set, lf, Role::StateIn)? {
        Some(s) => {
            expect_len(&s, state_len(heads))?;
            cpu::transpose_state(&s.data)
        }
        None => match &init.state {
            Some(s) if s.len() == state_len(heads) => cpu::transpose_state(s),
            _ => vec![0.0; state_len(heads)],
        },
    };
    let state_out = match optional(set, lf, Role::StateOut)? {
        Some(s) => {
            expect_len(&s, state_len(heads))?;
            Some(cpu::transpose_state(&s.data))
        }
        None => None,
    };
    Ok(LayerCase {
        params: LayerParams {
            heads,
            conv_w,
            a_log,
            dt_bias,
            norm_w,
            eps,
            lower,
        },
        conv,
        state,
        rows: Rows {
            heads,
            rows: t,
            qkv: qkv.data,
            a: a.data,
            b: b.data,
            gate: gate.data,
        },
        norm_out: norm_out.data,
        state_out,
    })
}

/// The whole layer ([`cpu::chain`]) against a golden's gated-norm output (and final state).
/// Runs both conv roundings and reports them; the fused one is the kernels'.
pub fn check_layer(
    set: &Set,
    lf: &LayerFixtures,
    init: &Init,
    eps: f32,
    lower: f32,
) -> Result<Vec<Comparison>, GoldenError> {
    let case = layer_case(set, lf, init, eps, lower)?;
    let t = case.rows.rows;
    let dtype = set
        .entry(lf.name(Role::NormOut).unwrap_or_default())
        .map(|e| e.dtype)
        .unwrap_or(DType::BF16);
    let tol = tolerance(dtype, recurrent(set, lf, t)?);
    let mut out = Vec::new();
    for (mode, label) in [
        (ConvRounding::Fused, "fused conv"),
        (ConvRounding::Unfused, "unfused conv"),
    ] {
        let r = cpu::chain(&case.params, &case.conv, &case.state, &case.rows, mode);
        out.push(Comparison::compare(
            &format!("layer {} norm output, {label} ({t} rows)", lf.layer),
            &r.out,
            &case.norm_out,
            tol,
        ));
        if let Some(so) = &case.state_out {
            out.push(Comparison::compare(
                &format!(
                    "layer {} state after the rows, {label} ({t} rows)",
                    lf.layer
                ),
                &r.state,
                so,
                tol,
            ));
        }
    }
    Ok(out)
}

/// The roles [`check_core`] needs.
pub const CORE_ROLES: &[Role] = &[
    Role::Q,
    Role::K,
    Role::V,
    Role::LogDecay,
    Role::Beta,
    Role::CoreOut,
];
/// The roles [`check_layer`] needs (plus the conv weight, whole or per projection).
pub const LAYER_ROLES: &[Role] = &[
    Role::MixedQkv,
    Role::ForgetProj,
    Role::BetaLogits,
    Role::GateProj,
    Role::NormOut,
    Role::ALog,
    Role::DtBias,
    Role::NormW,
];

/// Whether a layer has what [`check_layer`] needs.
pub fn has_layer_inputs(lf: &LayerFixtures) -> bool {
    lf.has(LAYER_ROLES)
        && (lf.has(&[Role::ConvW]) || lf.has(&[Role::ConvWQ, Role::ConvWK, Role::ConvWV]))
}

/// A tensor to write with [`write_set`]: name, dtype (f32, bf16 or i32), shape, values.
pub type NewTensor<'a> = (&'a str, DType, Vec<usize>, Vec<f32>);

/// Write a set in the oracle's layout (`manifest.json` with `tensors` and `notes`, one `.bin`
/// per tensor) to `dir`: for tests of the loader and for fixtures made from this crate's own
/// reference. bfloat16 tensors are rounded; i32 values truncated. `notes` is a JSON object.
pub fn write_set(dir: &Path, tensors: &[NewTensor<'_>], notes: &str) -> std::io::Result<()> {
    std::fs::create_dir_all(dir)?;
    let mut manifest = String::from("{\n  \"tensors\": {\n");
    for (i, (name, dtype, shape, values)) in tensors.iter().enumerate() {
        let bytes: Vec<u8> = match dtype {
            DType::F32 => values.iter().flat_map(|x| x.to_le_bytes()).collect(),
            DType::BF16 => values
                .iter()
                .flat_map(|&x| bf16::from_f32(x).to_le_bytes())
                .collect(),
            DType::I32 => values
                .iter()
                .flat_map(|&x| (x as i32).to_le_bytes())
                .collect(),
            _ => panic!("write_set writes f32, bf16 and i32"),
        };
        assert_eq!(
            values.len(),
            shape.iter().product::<usize>(),
            "{name}: shape"
        );
        let file = format!("{name}.bin");
        std::fs::write(dir.join(&file), &bytes)?;
        let shape: Vec<String> = shape.iter().map(|d| d.to_string()).collect();
        manifest.push_str(&format!(
            "    \"{name}\": {{\"file\": \"{file}\", \"dtype\": \"{}\", \"shape\": [{}], \"sha256\": \"{}\"}}{}\n",
            dtype.manifest_name(),
            shape.join(", "),
            crate::sha256::hex(&bytes),
            if i + 1 < tensors.len() { "," } else { "" }
        ));
    }
    manifest.push_str(&format!("  }},\n  \"notes\": {notes}\n}}\n"));
    std::fs::write(dir.join("manifest.json"), manifest)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn classifies_names() {
        assert_eq!(classify("layers.4.kda.q", None), Some((4, Role::Q)));
        assert_eq!(
            classify("model.language_model.layers.4.self_attn.A_log", None),
            Some((4, Role::ALog))
        );
        assert_eq!(
            classify(
                "model.language_model.layers.0.self_attn.forget_gate.dt_bias",
                None
            ),
            Some((0, Role::DtBias))
        );
        assert_eq!(
            classify(
                "model.language_model.layers.0.self_attn.q_conv1d.weight",
                None
            ),
            Some((0, Role::ConvWQ))
        );
        assert_eq!(classify("L06.kda.core_out", None), Some((6, Role::CoreOut)));
        // The oracle's names: the layer comes from the set.
        assert_eq!(classify("prefill.kda.q", Some(4)), Some((4, Role::Q)));
        assert_eq!(
            classify("decode.kda.state_final", Some(0)),
            Some((0, Role::StateOut))
        );
        assert_eq!(
            classify("decode.kda.state_heads", Some(0)),
            Some((0, Role::StateSteps))
        );
        assert_eq!(
            classify("prefill.kda.conv_state", Some(4)),
            Some((4, Role::ConvStateOut))
        );
        assert_eq!(
            classify("prefill.kda.qkv_preconv", Some(4)),
            Some((4, Role::MixedQkv))
        );
        assert_eq!(classify("decode.s3.kda.q", Some(4)), None);
        assert_eq!(classify("prefill.kda.q", None), None);
        // Layer 3 is a DSA layer; unknown roles are not KDA tensors.
        assert_eq!(classify("prefill.mla.q", Some(3)), None);
        assert_eq!(classify("layers.3.self_attn.q", None), None);
        assert_eq!(classify("prefill.moe.gate", Some(4)), None);
        assert_eq!(classify("logits", None), None);
        assert_eq!(classify("L04.prefill.kda.state_heads", None), None);
    }

    #[test]
    fn f16_widening() {
        assert_eq!(f16_to_f32(0x3c00), 1.0);
        assert_eq!(f16_to_f32(0xc000), -2.0);
        assert_eq!(f16_to_f32(0x0001), 2f32.powi(-24));
        assert_eq!(f16_to_f32(0x7c00), f32::INFINITY);
        assert_eq!(f16_to_f32(0x7bff), 65504.0);
    }
}
