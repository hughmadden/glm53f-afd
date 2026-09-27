//! Loading GLM-5.3-Flash tensors for the real-weight tests.
//!
//! Set `GLM53F_CHECKPOINT_DIR` to a directory of the official checkpoint's safetensors files
//! (all of them, or a subset holding the coordinator's tensors). Without it, or when a
//! tensor is missing or not yet downloaded (all zero), the tests print why and pass.
#![allow(dead_code)]

use glm53f_layers::fp8::Fp8Matrix;
use glm53f_layers::layer::LayerParams;
use glm53f_layers::mhc::HcParams;
use glm53f_layers::mlp::Fp8Mlp;
use glm53f_layers::testkit::safetensors::Checkpoint;

pub const PREFIX: &str = "model.language_model.";

pub fn checkpoint() -> Option<Checkpoint> {
    let Some(dir) = std::env::var_os("GLM53F_CHECKPOINT_DIR") else {
        eprintln!("skip: GLM53F_CHECKPOINT_DIR is not set");
        return None;
    };
    match Checkpoint::open_dir(std::path::Path::new(&dir)) {
        Ok(c) if !c.files.is_empty() => Some(c),
        Ok(_) => {
            eprintln!("skip: no safetensors files in GLM53F_CHECKPOINT_DIR");
            None
        }
        Err(e) => {
            eprintln!("skip: cannot open GLM53F_CHECKPOINT_DIR: {e}");
            None
        }
    }
}

fn name(layer: usize, rest: &str) -> String {
    format!("{PREFIX}layers.{layer}.{rest}")
}

/// A boundary's parameters (`which` = "attn" or "ffn").
pub fn hc(ck: &Checkpoint, layer: usize, which: &str) -> Option<HcParams> {
    let (f, fs) = ck.read_bf16(&name(layer, &format!("hc_{which}_fn"))).ok()?;
    let (b, _) = ck
        .read_f32(&name(layer, &format!("hc_{which}_base")))
        .ok()?;
    let (s, _) = ck
        .read_f32(&name(layer, &format!("hc_{which}_scale")))
        .ok()?;
    assert_eq!(fs, vec![24, 4 * 4096], "hc fn shape");
    if f.iter().all(|&v| v == 0) || s.iter().all(|&v| v == 0.0) {
        eprintln!("skip: layer {layer} hc_{which} data is all zero (not downloaded?)");
        return None;
    }
    Some(HcParams::new(4096, f, &b, &s))
}

pub fn norm(ck: &Checkpoint, layer: usize, which: &str) -> Option<Vec<u16>> {
    ck.read_bf16(&name(layer, &format!("{which}.weight")))
        .ok()
        .map(|(w, _)| w)
}

pub fn layer_params(ck: &Checkpoint, layer: usize) -> Option<LayerParams> {
    Some(LayerParams {
        attn_hc: hc(ck, layer, "attn")?,
        ffn_hc: hc(ck, layer, "ffn")?,
        input_norm: norm(ck, layer, "input_layernorm")?,
        post_attn_norm: norm(ck, layer, "post_attention_layernorm")?,
    })
}

pub fn fp8(ck: &Checkpoint, full: &str) -> Option<Fp8Matrix> {
    let m = ck.read_fp8(full).ok()?;
    if m.scale_inv.iter().all(|&s| s == 0.0) {
        eprintln!("skip: {full} scales are all zero (not downloaded?)");
        return None;
    }
    Some(m)
}

/// The dense MLP of layers 0-2 (`mlp.*`) or the shared expert of an MoE layer
/// (`mlp.shared_experts.*`).
pub fn mlp(ck: &Checkpoint, layer: usize, shared: bool) -> Option<Fp8Mlp> {
    let p = if shared { "mlp.shared_experts" } else { "mlp" };
    let g = fp8(ck, &name(layer, &format!("{p}.gate_proj.weight")))?;
    let u = fp8(ck, &name(layer, &format!("{p}.up_proj.weight")))?;
    let d = fp8(ck, &name(layer, &format!("{p}.down_proj.weight")))?;
    Some(Fp8Mlp::new(&g, &u, d))
}

/// The router weight (BF16 `[288][4096]`) and correction bias (f32 `[288]`).
pub fn router(ck: &Checkpoint, layer: usize) -> Option<(Vec<u16>, Vec<f32>)> {
    let (w, ws) = ck.read_bf16(&name(layer, "mlp.gate.weight")).ok()?;
    let (b, _) = ck
        .read_f32(&name(layer, "mlp.gate.e_score_correction_bias"))
        .ok()?;
    assert_eq!(ws, vec![288, 4096], "router weight shape");
    if w.iter().all(|&v| v == 0) {
        eprintln!("skip: layer {layer} router weight is all zero (not downloaded?)");
        return None;
    }
    Some((w, b))
}

/// Embedding rows of the given token ids.
pub fn embeddings(ck: &Checkpoint, ids: &[usize]) -> Option<Vec<u16>> {
    let mut out = Vec::new();
    for &id in ids {
        out.extend(
            ck.read_bf16_rows(&format!("{PREFIX}embed_tokens.weight"), id, 1)
                .ok()?,
        );
    }
    if out.iter().all(|&v| v == 0) {
        eprintln!("skip: embedding rows are all zero (not downloaded?)");
        return None;
    }
    Some(out)
}

/// A few token ids spread over the vocabulary.
pub const TOKENS: [usize; 5] = [0, 42, 1000, 77_777, 154_819];
