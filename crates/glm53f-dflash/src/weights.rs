//! The drafter's tensors, and the target rows it borrows (embedding rows and the LM head).
//!
//! Everything stays in the checkpoint's BF16 (as bits); the CPU reference widens on the fly and
//! the GPU forward uploads the bits. Names are the checkpoint's (`model.safetensors` of
//! `incoai/GLM-5.3-Flash-DFlash2`); the target's are `model.language_model.embed_tokens.weight`
//! and `lm_head.weight` of `zai-org/GLM-5.3-Flash`.

use std::path::{Path, PathBuf};

use glm53f_model::dtype::DType;
use glm53f_model::safetensors::{Checkpoint, Runs};

use crate::synth::splitmix64;
use crate::{bf16, Dims};

/// The target's input embedding.
pub const TARGET_EMBED: &str = "model.language_model.embed_tokens.weight";
/// The target's output head.
pub const TARGET_LM_HEAD: &str = "lm_head.weight";

/// One decoder layer's tensors (BF16 bits, row-major `[out][in]` for projections).
#[derive(Clone, Debug, Default)]
pub struct LayerWeights {
    pub q: Vec<u16>,
    pub k: Vec<u16>,
    pub v: Vec<u16>,
    pub o: Vec<u16>,
    pub q_norm: Vec<u16>,
    pub k_norm: Vec<u16>,
    pub input_ln: Vec<u16>,
    pub post_ln: Vec<u16>,
    pub gate: Vec<u16>,
    pub up: Vec<u16>,
    pub down: Vec<u16>,
    /// `attention_conv.base_kernel` `[2 sides][taps][hidden]`.
    pub attn_base: Vec<u16>,
    /// `attention_conv.kernel_projection.weight` `[2 * taps * groups][hidden]`.
    pub attn_kp: Vec<u16>,
    pub mlp_base: Vec<u16>,
    pub mlp_kp: Vec<u16>,
}

/// All of the drafter's own tensors.
#[derive(Clone, Debug)]
pub struct Weights {
    pub dims: Dims,
    /// `fc.weight` `[hidden][taps * hidden]`.
    pub fc: Vec<u16>,
    pub hidden_norm: Vec<u16>,
    pub norm: Vec<u16>,
    pub layers: Vec<LayerWeights>,
    /// `candidate_selector.hidden_projection.weight` `[rank][hidden]`.
    pub hproj: Vec<u16>,
    /// `candidate_selector.predecessor_codebook` `[vocab][rank]`.
    pub pred: Vec<u16>,
    /// `candidate_selector.successor_codebook` `[vocab][rank]`.
    pub succ: Vec<u16>,
}

/// Where a test finds data: the directory named by environment variable `var`, or `None` with a
/// message on stderr (the test then passes without running).
pub fn env_dir(var: &str, what: &str) -> Option<PathBuf> {
    match std::env::var_os(var) {
        Some(v) if !v.is_empty() => Some(PathBuf::from(v)),
        _ => {
            eprintln!("skipped: set {var} to {what}");
            None
        }
    }
}

fn read_bf16(ck: &Checkpoint, name: &str, shape: &[usize]) -> Result<Vec<u16>, String> {
    let (_, t) = ck
        .get(name)
        .ok_or_else(|| format!("missing tensor {name}"))?;
    let want: Vec<u64> = shape.iter().map(|&d| d as u64).collect();
    if t.dtype != DType::BF16 || t.shape != want {
        return Err(format!(
            "{name}: {} {:?}, expected BF16 {want:?}",
            t.dtype.as_str(),
            t.shape
        ));
    }
    Ok(bf16::from_le_bytes(
        &ck.read_tensor(name).map_err(|e| e.to_string())?,
    ))
}

impl Weights {
    /// Load the drafter from its checkpoint directory (`config.json`, `model.safetensors`),
    /// checking the config and every tensor's dtype and shape against `dims`.
    pub fn load(dir: &Path, dims: Dims) -> Result<Weights, String> {
        dims.validate()?;
        let cfg = glm53f_model::config::DraftConfig::load(&dir.join("config.json"))
            .map_err(|e| e.to_string())?;
        dims.check_config(&cfg)?;
        let ck = Checkpoint::open(dir).map_err(|e| e.to_string())?;
        let d = dims;
        let h = d.hidden;
        let g = |name: &str, shape: &[usize]| read_bf16(&ck, name, shape);
        let mut layers = Vec::with_capacity(d.layers);
        for l in 0..d.layers {
            let p = format!("layers.{l}.");
            layers.push(LayerWeights {
                q: g(&format!("{p}self_attn.q_proj.weight"), &[d.q_width(), h])?,
                k: g(&format!("{p}self_attn.k_proj.weight"), &[d.kv_width(), h])?,
                v: g(&format!("{p}self_attn.v_proj.weight"), &[d.kv_width(), h])?,
                o: g(&format!("{p}self_attn.o_proj.weight"), &[h, d.q_width()])?,
                q_norm: g(&format!("{p}self_attn.q_norm.weight"), &[d.head_dim])?,
                k_norm: g(&format!("{p}self_attn.k_norm.weight"), &[d.head_dim])?,
                input_ln: g(&format!("{p}input_layernorm.weight"), &[h])?,
                post_ln: g(&format!("{p}post_attention_layernorm.weight"), &[h])?,
                gate: g(&format!("{p}mlp.gate_proj.weight"), &[d.inter, h])?,
                up: g(&format!("{p}mlp.up_proj.weight"), &[d.inter, h])?,
                down: g(&format!("{p}mlp.down_proj.weight"), &[h, d.inter])?,
                attn_base: g(
                    &format!("{p}attention_conv.base_kernel"),
                    &[2, d.conv_taps, h],
                )?,
                attn_kp: g(
                    &format!("{p}attention_conv.kernel_projection.weight"),
                    &[d.dyn_width(), h],
                )?,
                mlp_base: g(&format!("{p}mlp_conv.base_kernel"), &[2, d.conv_taps, h])?,
                mlp_kp: g(
                    &format!("{p}mlp_conv.kernel_projection.weight"),
                    &[d.dyn_width(), h],
                )?,
            });
        }
        let w = Weights {
            dims: d,
            fc: g("fc.weight", &[h, d.tap_width()])?,
            hidden_norm: g("hidden_norm.weight", &[h])?,
            norm: g("norm.weight", &[h])?,
            layers,
            hproj: g("candidate_selector.hidden_projection.weight", &[d.rank, h])?,
            pred: g(
                "candidate_selector.predecessor_codebook",
                &[d.vocab, d.rank],
            )?,
            succ: g("candidate_selector.successor_codebook", &[d.vocab, d.rank])?,
        };
        let expected = 6 + 15 * d.layers;
        let found = ck.names().count();
        if found != expected {
            return Err(format!(
                "{}: {found} tensors, expected {expected}",
                dir.display()
            ));
        }
        Ok(w)
    }

    /// Random weights of the right shapes, for tests without the checkpoint. Projections are
    /// uniform with the variance of `1 / fan_in`; norms near 1; the convolutions near the identity
    /// (tap 0 near 1, tap 1 and the dynamic part small), so activations stay in range.
    pub fn random(dims: Dims, seed: u64) -> Weights {
        let d = dims;
        let h = d.hidden;
        let mut n = 0u64;
        let mut fill = |len: usize, scale: f32, offset: f32| -> Vec<u16> {
            n += 1;
            let base = (seed << 32) ^ (n << 56);
            (0..len as u64)
                .map(|i| {
                    let u = (splitmix64(base.wrapping_add(i)) >> 40) as f32 / (1u32 << 24) as f32;
                    bf16::from_f32(offset + scale * (2.0 * u - 1.0))
                })
                .collect()
        };
        let lin = |fan_in: usize| (3.0 / fan_in as f32).sqrt();
        let conv_base = |fill: &mut dyn FnMut(usize, f32, f32) -> Vec<u16>| {
            // [side][tap][hidden]: tap 0 near 1, the rest near 0.
            let mut v = Vec::with_capacity(2 * d.conv_taps * h);
            for _side in 0..2 {
                for tap in 0..d.conv_taps {
                    v.extend(fill(h, 0.1, if tap == 0 { 1.0 } else { 0.0 }));
                }
            }
            v
        };
        let mut layers = Vec::with_capacity(d.layers);
        for _ in 0..d.layers {
            let attn_base = conv_base(&mut fill);
            let mlp_base = conv_base(&mut fill);
            layers.push(LayerWeights {
                q: fill(d.q_width() * h, lin(h), 0.0),
                k: fill(d.kv_width() * h, lin(h), 0.0),
                v: fill(d.kv_width() * h, lin(h), 0.0),
                o: fill(h * d.q_width(), lin(d.q_width()), 0.0),
                q_norm: fill(d.head_dim, 0.2, 1.0),
                k_norm: fill(d.head_dim, 0.2, 1.0),
                input_ln: fill(h, 0.2, 1.0),
                post_ln: fill(h, 0.2, 1.0),
                gate: fill(d.inter * h, lin(h), 0.0),
                up: fill(d.inter * h, lin(h), 0.0),
                down: fill(h * d.inter, lin(d.inter), 0.0),
                attn_base,
                attn_kp: fill(d.dyn_width() * h, 0.2 * lin(h), 0.0),
                mlp_base,
                mlp_kp: fill(d.dyn_width() * h, 0.2 * lin(h), 0.0),
            });
        }
        Weights {
            dims: d,
            fc: fill(h * d.tap_width(), lin(d.tap_width()), 0.0),
            hidden_norm: fill(h, 0.2, 1.0),
            norm: fill(h, 0.2, 1.0),
            layers,
            hproj: fill(d.rank * h, lin(h), 0.0),
            pred: fill(d.vocab * d.rank, 1.0, 0.0),
            succ: fill(d.vocab * d.rank, 1.0, 0.0),
        }
    }
}

/// The target rows the drafter borrows, read from a GLM-5.3-Flash checkpoint directory (the
/// official one, or a subset holding the non-expert tensors).
pub struct Target {
    ck: Checkpoint,
    hidden: usize,
    vocab: usize,
}

impl Target {
    pub fn open(dir: &Path, dims: &Dims) -> Result<Target, String> {
        let ck = Checkpoint::open(dir).map_err(|e| e.to_string())?;
        for name in [TARGET_EMBED, TARGET_LM_HEAD] {
            let (_, t) = ck
                .get(name)
                .ok_or_else(|| format!("{}: no {name}", dir.display()))?;
            if t.dtype != DType::BF16 || t.shape != [dims.vocab as u64, dims.hidden as u64] {
                return Err(format!("{name}: {} {:?}", t.dtype.as_str(), t.shape));
            }
        }
        Ok(Target {
            ck,
            hidden: dims.hidden,
            vocab: dims.vocab,
        })
    }

    /// Embedding rows of `ids`, BF16 bits `[ids.len()][hidden]`.
    pub fn embed_rows(&self, ids: &[u32]) -> Result<Vec<u16>, String> {
        let row = (self.hidden * 2) as u64;
        let mut out = Vec::with_capacity(ids.len() * self.hidden);
        for &id in ids {
            if id as usize >= self.vocab {
                return Err(format!("token {id} beyond the vocabulary"));
            }
            let runs = Runs {
                offset: id as u64 * row,
                len: row,
                stride: row,
                count: 1,
            };
            out.extend(bf16::from_le_bytes(
                &self
                    .ck
                    .read_runs(TARGET_EMBED, &runs)
                    .map_err(|e| e.to_string())?,
            ));
        }
        Ok(out)
    }

    /// The whole LM head, BF16 bits `[vocab][hidden]` (1.27 GB), read in slices.
    pub fn lm_head(&self) -> Result<Vec<u16>, String> {
        let row = (self.hidden * 2) as u64;
        let mut out = Vec::with_capacity(self.vocab * self.hidden);
        let chunk = 8192usize;
        let mut r = 0usize;
        while r < self.vocab {
            let n = chunk.min(self.vocab - r);
            let runs = Runs {
                offset: r as u64 * row,
                len: n as u64 * row,
                stride: n as u64 * row,
                count: 1,
            };
            out.extend(bf16::from_le_bytes(
                &self
                    .ck
                    .read_runs(TARGET_LM_HEAD, &runs)
                    .map_err(|e| e.to_string())?,
            ));
            r += n;
        }
        Ok(out)
    }
}
