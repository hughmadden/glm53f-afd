//! Loading one DSA layer's weights from the official checkpoint.
//!
//! In the official FP8 checkpoint the DSA projections `q_a_proj`, `q_b_proj`,
//! `kv_a_proj_with_mqa` and `o_proj` are FP8 E4M3 with `weight_scale_inv` over
//! 128 x 128 blocks; `kv_b_proj`, the norms and every indexer tensor are BF16
//! (they are in the checkpoint's `modules_to_not_convert`).

use crate::config::DsaConfig;
use crate::indexer::IndexerWeights;
use crate::layer::DsaLayerWeights;
use crate::mla::MlaWeights;
use crate::safetensors::Checkpoint;

/// Tensor-name prefixes tried for layer `l`'s attention module.
pub fn attn_prefixes(l: usize) -> [String; 2] {
    [format!("model.language_model.layers.{l}.self_attn."), format!("model.layers.{l}.self_attn.")]
}

/// The attention prefix the checkpoint uses for layer `l`, if any.
pub fn find_prefix(ck: &Checkpoint, l: usize) -> Option<String> {
    attn_prefixes(l).into_iter().find(|p| ck.find(&format!("{p}kv_b_proj.weight")).is_some())
}

/// Names of every tensor one DSA layer needs (relative to the attention prefix).
pub const DSA_TENSORS: [&str; 18] = [
    "q_a_proj.weight",
    "q_a_proj.weight_scale_inv",
    "q_a_layernorm.weight",
    "q_b_proj.weight",
    "q_b_proj.weight_scale_inv",
    "kv_a_proj_with_mqa.weight",
    "kv_a_proj_with_mqa.weight_scale_inv",
    "kv_a_layernorm.weight",
    "kv_b_proj.weight",
    "o_proj.weight",
    "o_proj.weight_scale_inv",
    "indexer.wq_b.weight",
    "indexer.wk.weight",
    "indexer.k_norm.weight",
    "indexer.k_norm.bias",
    "indexer.weights_proj.weight",
    "indexer.index_kpool_compress_gate",
    "indexer.index_kpool_compress_ape",
];

/// Whether every tensor of layer `l` is in the checkpoint and looks fetched.
pub fn layer_available(ck: &Checkpoint, l: usize) -> bool {
    let Some(p) = find_prefix(ck, l) else { return false };
    DSA_TENSORS.iter().all(|n| ck.looks_present(&format!("{p}{n}")))
}

fn expect_shape(name: &str, got: &[usize], want: &[usize]) -> Result<(), String> {
    if got != want {
        return Err(format!("{name}: shape {got:?}, expected {want:?}"));
    }
    Ok(())
}

/// Load (and dequantize to f32) the weights of DSA layer `l`.
pub fn load_dsa_layer(ck: &Checkpoint, cfg: &DsaConfig, l: usize) -> Result<DsaLayerWeights, String> {
    let p = find_prefix(ck, l).ok_or_else(|| format!("layer {l}: no attention tensors in the checkpoint"))?;
    let (h, ql, kl, nh, dn, dv) =
        (cfg.hidden, cfg.q_lora_rank, cfg.kv_lora_rank, cfg.n_heads, cfg.qk_nope_head_dim, cfg.v_head_dim);
    let (id, inh, kp) = (cfg.index_head_dim, cfg.index_n_heads, cfg.index_kpool);
    let w = |n: &str, shape: &[usize]| -> Result<Vec<f32>, String> {
        let name = format!("{p}{n}");
        let (v, s) = ck.read_weight(&name)?;
        expect_shape(&name, &s, shape)?;
        Ok(v)
    };
    let mla = MlaWeights {
        q_a_proj: w("q_a_proj.weight", &[ql, h])?,
        q_a_norm: w("q_a_layernorm.weight", &[ql])?,
        q_b_proj: w("q_b_proj.weight", &[nh * dn, ql])?,
        kv_a_proj: w("kv_a_proj_with_mqa.weight", &[kl, h])?,
        kv_a_norm: w("kv_a_layernorm.weight", &[kl])?,
        kv_b_proj: w("kv_b_proj.weight", &[nh * (dn + dv), kl])?,
        o_proj: w("o_proj.weight", &[h, nh * dv])?,
    };
    let idx = IndexerWeights {
        wq_b: w("indexer.wq_b.weight", &[inh * id, ql])?,
        wk: w("indexer.wk.weight", &[id, h])?,
        k_norm_w: w("indexer.k_norm.weight", &[id])?,
        k_norm_b: w("indexer.k_norm.bias", &[id])?,
        weights_proj: w("indexer.weights_proj.weight", &[inh, h])?,
        gate: w("indexer.index_kpool_compress_gate", &[id, h])?,
        ape: w("indexer.index_kpool_compress_ape", &[kp, id])?,
    };
    let out = DsaLayerWeights { mla, idx };
    out.check(cfg)?;
    Ok(out)
}

/// The checkpoint directory from `GLM53F_CHECKPOINT_DIR` (the name every crate's tests use), or
/// the older `GLM53F_CHECKPOINT`, if set and readable.
pub fn checkpoint_from_env() -> Option<Checkpoint> {
    let dir = std::env::var_os("GLM53F_CHECKPOINT_DIR").or_else(|| std::env::var_os("GLM53F_CHECKPOINT"))?;
    Checkpoint::open_dir(std::path::Path::new(&dir)).ok()
}
