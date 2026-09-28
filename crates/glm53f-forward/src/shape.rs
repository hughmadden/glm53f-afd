//! The GLM-5.3-Flash dimensions the forward is built for, and which decoder layers a forward
//! runs.
//!
//! A forward runs the first `layers` decoder layers (all 45 in serving; a prefix such as
//! layers 0-4 in the golden tests) and then the head. [`ModelShape`] records each layer's
//! attention and MLP kind and its index among the layers of that kind, which is how the KV
//! and the weights are laid out.

use glm53f_model::config::{glm53_flash as g, AttnKind, MlpKind, TextConfig};

use crate::error::{invalid, Result};

/// Hidden size.
pub const HIDDEN: usize = g::HIDDEN_SIZE as usize;
/// mHC residual streams.
pub const HC: usize = g::HC_MULT as usize;
/// LM head rows (padding included).
pub const VOCAB: usize = g::VOCAB_SIZE as usize;
/// Token ids a pick may return: one past the tokenizer's largest id. LM head rows at and above
/// it are padding, never sampled or argmaxed.
pub const SAMPLE_VOCAB: usize = 154_856;
/// KDA heads and head size.
pub const KDA_HEADS: usize = g::KDA_HEADS as usize;
pub const KDA_DIM: usize = g::KDA_HEAD_DIM as usize;
/// KDA q, k or v width (8,192).
pub const KDA_WIDTH: usize = KDA_HEADS * KDA_DIM;
/// KDA conv channels, q | k | v (24,576).
pub const KDA_QKV: usize = 3 * KDA_WIDTH;
/// Columns of the stacked KDA input projection: q | k | v | beta logits (24,640).
pub const KDA_P_COLS: usize = KDA_QKV + KDA_HEADS;
/// Conv history a request carries (taps - 1).
pub const KDA_WINDOW: usize = g::KDA_CONV_KERNEL as usize - 1;
/// MLA heads, query latent, KV latent, head sizes.
pub const MLA_HEADS: usize = g::MLA_HEADS as usize;
pub const Q_LORA: usize = g::Q_LORA_RANK as usize;
pub const KV_LORA: usize = g::KV_LORA_RANK as usize;
pub const QK_HEAD: usize = g::QK_NOPE_HEAD_DIM as usize;
pub const V_HEAD: usize = g::V_HEAD_DIM as usize;
/// Indexer heads and head size.
pub const INDEX_HEADS: usize = g::INDEX_HEADS as usize;
pub const INDEX_DIM: usize = g::INDEX_HEAD_DIM as usize;
/// Columns of the stacked indexer projection of the layer input: wk | compress gate |
/// weights_proj (288).
pub const IDX_PROJ_COLS: usize = 2 * INDEX_DIM + INDEX_HEADS;
/// Selected tokens per query row at most: 512 pools of 4 plus a tail of 3.
pub const MAX_SELECTED: usize = 2051;
/// Kept pools per query row.
pub const TOP_POOLS: usize = 512;
/// Routed experts, experts per token, expert width, shared width, dense width.
pub const EXPERTS: usize = g::ROUTED_EXPERTS as usize;
pub const TOP_K: usize = g::EXPERTS_PER_TOKEN as usize;
pub const MOE_INTER: usize = g::MOE_INTERMEDIATE as usize;
pub const SHARED_INTER: usize = MOE_INTER * g::SHARED_EXPERTS as usize;
pub const DENSE_INTER: usize = g::DENSE_INTERMEDIATE as usize;
/// `routed_scaling_factor`.
pub const ROUTED_SCALE: f32 = g::ROUTED_SCALING_FACTOR as f32;
/// `rms_norm_eps`.
pub const RMS_EPS: f32 = 1e-5;
/// `gate_lower_bound`.
pub const KDA_LOWER: f32 = -5.0;
/// The indexer's `k_norm` LayerNorm epsilon (fixed in the reference).
pub const INDEX_LN_EPS: f32 = 1e-6;
/// Softmax scale of MLA (`qk_head_dim^-0.5`).
pub const MLA_SCALE: f32 = 0.0625;

/// The layers a forward runs: decoder layers `0 .. layers`, each with its kinds and its index
/// among the layers of the same kind.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ModelShape {
    pub layers: usize,
    pub attn: Vec<AttnKind>,
    pub mlp: Vec<MlpKind>,
    /// Per layer: its index among the KDA layers, if it is one.
    pub kda_index: Vec<Option<usize>>,
    /// Per layer: its index among the DSA layers, if it is one.
    pub dsa_index: Vec<Option<usize>>,
    pub kda_layers: usize,
    pub dsa_layers: usize,
}

impl ModelShape {
    /// The first `layers` decoder layers of `text` (a validated GLM-5.3-Flash config).
    pub fn new(text: &TextConfig, layers: usize) -> Result<ModelShape> {
        if layers == 0 || layers > text.num_hidden_layers as usize {
            return Err(invalid!(
                "layers = {layers}: the model has {} decoder layers",
                text.num_hidden_layers
            ));
        }
        let attn: Vec<AttnKind> = (0..layers as u64).map(|l| text.attn_kind(l)).collect();
        let mlp: Vec<MlpKind> = (0..layers as u64)
            .map(|l| {
                if text.is_moe(l) {
                    MlpKind::Moe
                } else {
                    MlpKind::Dense
                }
            })
            .collect();
        let (mut kda, mut dsa) = (0, 0);
        let mut kda_index = Vec::with_capacity(layers);
        let mut dsa_index = Vec::with_capacity(layers);
        for a in &attn {
            match a {
                AttnKind::Kda => {
                    kda_index.push(Some(kda));
                    dsa_index.push(None);
                    kda += 1;
                }
                AttnKind::Dsa => {
                    kda_index.push(None);
                    dsa_index.push(Some(dsa));
                    dsa += 1;
                }
            }
        }
        Ok(ModelShape {
            layers,
            attn,
            mlp,
            kda_index,
            dsa_index,
            kda_layers: kda,
            dsa_layers: dsa,
        })
    }

    /// Every decoder layer.
    pub fn full(text: &TextConfig) -> Result<ModelShape> {
        ModelShape::new(text, text.num_hidden_layers as usize)
    }

    pub fn is_moe(&self, layer: usize) -> bool {
        self.mlp[layer] == MlpKind::Moe
    }

    /// The MoE layers run.
    pub fn moe_layers(&self) -> Vec<usize> {
        (0..self.layers).filter(|&l| self.is_moe(l)).collect()
    }
}
