//! Typed GLM-5.3-Flash `config.json` (architecture `glm5_next`) and DFlash2
//! drafter config, with the invariants the engine relies on.
//!
//! Parsing and validation are separate steps. [`ModelConfig::parse`] reads any
//! `glm5_next` config into typed fields (a missing key or a wrong type is an
//! error); [`ModelConfig::validate`] then checks the GLM-5.3-Flash shape the
//! engine's kernels, wire frames and memory plan are built for, and lists
//! every violated invariant at once. [`ModelConfig::load`] does both.

use std::ops::Range;
use std::path::Path;

use crate::dtype::DType;
use crate::error::{io_err, Error, Result};
use crate::json::{self, Json};

/// The GLM-5.3-Flash shape the engine is built for.
pub mod glm53_flash {
    pub const NUM_HIDDEN_LAYERS: u64 = 45;
    pub const KDA_LAYERS: usize = 34;
    pub const DSA_LAYERS: usize = 11;
    pub const HIDDEN_SIZE: u64 = 4096;
    pub const VOCAB_SIZE: u64 = 154_880;
    pub const MAX_POSITIONS: u64 = 1_048_576;
    pub const KDA_HEADS: u64 = 64;
    pub const KDA_HEAD_DIM: u64 = 128;
    pub const KDA_CONV_KERNEL: u64 = 4;
    pub const MLA_HEADS: u64 = 64;
    pub const Q_LORA_RANK: u64 = 1536;
    pub const KV_LORA_RANK: u64 = 512;
    pub const QK_NOPE_HEAD_DIM: u64 = 256;
    pub const QK_ROPE_HEAD_DIM: u64 = 0;
    pub const V_HEAD_DIM: u64 = 256;
    pub const INDEX_HEADS: u64 = 32;
    pub const INDEX_HEAD_DIM: u64 = 128;
    pub const INDEX_TOPK: u64 = 2048;
    pub const INDEX_KPOOL: u64 = 4;
    pub const HC_MULT: u64 = 4;
    pub const HC_SINKHORN_ITERS: u64 = 20;
    pub const ROUTED_EXPERTS: u64 = 288;
    pub const EXPERTS_PER_TOKEN: u64 = 8;
    pub const MOE_INTERMEDIATE: u64 = 2048;
    pub const SHARED_EXPERTS: u64 = 1;
    pub const FIRST_K_DENSE: u64 = 3;
    pub const DENSE_INTERMEDIATE: u64 = 12_288;
    pub const ROUTED_SCALING_FACTOR: f64 = 2.5;
    pub const SWIGLU_LIMIT: f64 = 10.0;
    pub const MTP_LAYERS: u64 = 1;
}

/// Attention type of a decoder layer.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum AttnKind {
    /// `linear_attention`: Kimi delta attention, a fixed-size recurrent state.
    Kda,
    /// `deepseek_sparse_attention`: MLA over a 512-dim latent plus the indexer.
    Dsa,
}

/// MLP type of a decoder layer.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum MlpKind {
    Dense,
    Moe,
}

/// The checkpoint's `quantization_config`, as far as the catalog needs it.
#[derive(Clone, Debug, PartialEq)]
pub enum Quantization {
    /// No `quantization_config`.
    None,
    /// `quant_method: fp8` with block scales (the official checkpoint).
    Fp8Block {
        fmt: String,
        block: [u64; 2],
        activation_scheme: String,
    },
    /// `quant_method: exl3` (ExLlamaV3 trellis).
    Exl3 { bits: u64, codebook: String },
    /// `quant_method: modelopt`, e.g. NVFP4 with its weight group size.
    Modelopt { algo: String, group_size: u64 },
    /// Anything else, by `quant_method`.
    Other(String),
}

#[derive(Clone, Debug, PartialEq)]
pub struct ModelConfig {
    pub model_type: String,
    pub architectures: Vec<String>,
    pub tie_word_embeddings: bool,
    pub text: TextConfig,
    pub vision: VisionConfig,
    pub quantization: Quantization,
}

#[derive(Clone, Debug, PartialEq)]
pub struct TextConfig {
    pub model_type: String,
    pub hidden_size: u64,
    pub num_hidden_layers: u64,
    pub vocab_size: u64,
    pub max_position_embeddings: u64,
    pub rms_norm_eps: f64,
    pub hidden_act: String,
    pub tie_word_embeddings: bool,
    pub eos_token_ids: Vec<u64>,
    pub pad_token_id: u64,
    /// One per decoder layer (MTP layers not included).
    pub layer_types: Vec<AttnKind>,
    pub mlp_layer_types: Vec<MlpKind>,
    /// `full` (the layer runs its own indexer) or `shared`.
    pub indexer_types: Vec<String>,
    pub kda: KdaConfig,
    pub mla: MlaConfig,
    pub indexer: IndexerConfig,
    pub mhc: MhcConfig,
    pub moe: MoeConfig,
    pub num_nextn_predict_layers: u64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct KdaConfig {
    pub num_heads: u64,
    pub head_dim: u64,
    pub short_conv_kernel_size: u64,
    pub gate_lower_bound: f64,
    pub kda_layers: Vec<u64>,
    pub full_attn_layers: Vec<u64>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct MlaConfig {
    pub num_attention_heads: u64,
    pub num_key_value_heads: u64,
    pub q_lora_rank: u64,
    pub kv_lora_rank: u64,
    pub qk_nope_head_dim: u64,
    pub qk_rope_head_dim: u64,
    pub qk_head_dim: u64,
    pub v_head_dim: u64,
    pub use_nope: bool,
    pub attention_bias: bool,
}

#[derive(Clone, Debug, PartialEq)]
pub struct IndexerConfig {
    pub n_heads: u64,
    pub head_dim: u64,
    pub topk: u64,
    pub kpool: u64,
    pub kpool_compress: bool,
    pub always_select_tail: bool,
    pub share_for_mtp_iteration: bool,
}

#[derive(Clone, Debug, PartialEq)]
pub struct MhcConfig {
    pub enabled: bool,
    pub hc_mult: u64,
    pub sinkhorn_iters: u64,
    pub eps: f64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct MoeConfig {
    pub n_routed_experts: u64,
    pub num_experts_per_tok: u64,
    pub moe_intermediate_size: u64,
    pub n_shared_experts: u64,
    pub first_k_dense_replace: u64,
    /// `intermediate_size`: the width of the dense MLP layers.
    pub dense_intermediate_size: u64,
    pub scoring_func: String,
    pub topk_method: String,
    pub routed_scaling_factor: f64,
    pub norm_topk_prob: bool,
    pub n_group: u64,
    pub topk_group: u64,
    pub swiglu_limit: f64,
    pub router_dtype: String,
}

#[derive(Clone, Debug, PartialEq)]
pub struct VisionConfig {
    pub depth: u64,
    pub hidden_size: u64,
    pub num_heads: u64,
    pub intermediate_size: u64,
    pub out_hidden_size: u64,
    pub projection_intermediate_size: u64,
    pub patch_size: u64,
    pub temporal_patch_size: u64,
    pub spatial_merge_size: u64,
    pub in_channels: u64,
    pub image_size: u64,
    pub attention_bias: bool,
    pub swiglu_limit: f64,
    pub rms_norm_eps: f64,
}

/// A JSON object being read, with its path for error messages.
struct Obj<'a> {
    path: String,
    v: &'a Json,
}

impl<'a> Obj<'a> {
    fn root(v: &'a Json) -> Result<Obj<'a>> {
        v.as_object()
            .ok_or_else(|| Error::Json("config is not a JSON object".into()))?;
        Ok(Obj {
            path: String::new(),
            v,
        })
    }
    fn at(&self, key: &str) -> String {
        if self.path.is_empty() {
            key.to_string()
        } else {
            format!("{}.{key}", self.path)
        }
    }
    fn opt(&self, key: &str) -> Option<&'a Json> {
        self.v.get(key)
    }
    fn req(&self, key: &str) -> Result<&'a Json> {
        self.opt(key)
            .ok_or_else(|| Error::Json(format!("{}: missing", self.at(key))))
    }
    fn wrong(&self, key: &str, want: &str) -> Error {
        Error::Json(format!("{}: expected {want}", self.at(key)))
    }
    fn obj(&self, key: &str) -> Result<Obj<'a>> {
        let v = self.req(key)?;
        v.as_object().ok_or_else(|| self.wrong(key, "an object"))?;
        Ok(Obj {
            path: self.at(key),
            v,
        })
    }
    fn u64(&self, key: &str) -> Result<u64> {
        self.req(key)?
            .as_u64()
            .ok_or_else(|| self.wrong(key, "a non-negative integer"))
    }
    fn f64(&self, key: &str) -> Result<f64> {
        self.req(key)?
            .as_f64()
            .ok_or_else(|| self.wrong(key, "a number"))
    }
    fn bool(&self, key: &str) -> Result<bool> {
        self.req(key)?
            .as_bool()
            .ok_or_else(|| self.wrong(key, "true or false"))
    }
    fn str(&self, key: &str) -> Result<String> {
        Ok(self
            .req(key)?
            .as_str()
            .ok_or_else(|| self.wrong(key, "a string"))?
            .to_string())
    }
    fn u64_list(&self, key: &str) -> Result<Vec<u64>> {
        let a = self
            .req(key)?
            .as_array()
            .ok_or_else(|| self.wrong(key, "a list"))?;
        a.iter()
            .map(|x| {
                x.as_u64()
                    .ok_or_else(|| self.wrong(key, "a list of integers"))
            })
            .collect()
    }
    /// An integer or a list of integers (e.g. `eos_token_id`).
    fn u64_or_list(&self, key: &str) -> Result<Vec<u64>> {
        match self.req(key)?.as_u64() {
            Some(n) => Ok(vec![n]),
            None => self.u64_list(key),
        }
    }
    fn str_list(&self, key: &str) -> Result<Vec<String>> {
        let a = self
            .req(key)?
            .as_array()
            .ok_or_else(|| self.wrong(key, "a list"))?;
        a.iter()
            .map(|x| {
                x.as_str()
                    .map(str::to_string)
                    .ok_or_else(|| self.wrong(key, "a list of strings"))
            })
            .collect()
    }
}

fn read_json(path: &Path) -> Result<Json> {
    let bytes = std::fs::read(path).map_err(|e| io_err(path, e))?;
    let v =
        json::parse_bytes(&bytes).map_err(|m| Error::Json(format!("{}: {m}", path.display())))?;
    json::check_unique_keys(&v).map_err(|m| Error::Json(format!("{}: {m}", path.display())))?;
    Ok(v)
}

impl ModelConfig {
    /// Read, parse and validate a GLM-5.3-Flash `config.json`.
    pub fn load(path: &Path) -> Result<ModelConfig> {
        let c = ModelConfig::from_json(&read_json(path)?)?;
        c.validate()?;
        Ok(c)
    }

    /// Parse config text into typed fields, without the GLM-5.3-Flash checks.
    pub fn parse(text: &str) -> Result<ModelConfig> {
        let v = json::parse(text).map_err(Error::Json)?;
        json::check_unique_keys(&v).map_err(Error::Json)?;
        ModelConfig::from_json(&v)
    }

    pub fn from_json(v: &Json) -> Result<ModelConfig> {
        let root = Obj::root(v)?;
        let t = root.obj("text_config")?;
        let lin = t.obj("linear_attn_config")?;
        let layer_types = t
            .str_list("layer_types")?
            .iter()
            .map(|s| match s.as_str() {
                "linear_attention" => Ok(AttnKind::Kda),
                "deepseek_sparse_attention" => Ok(AttnKind::Dsa),
                other => Err(Error::Json(format!(
                    "text_config.layer_types: unknown type {other:?}"
                ))),
            })
            .collect::<Result<Vec<_>>>()?;
        let mlp_layer_types = t
            .str_list("mlp_layer_types")?
            .iter()
            .map(|s| match s.as_str() {
                "dense" => Ok(MlpKind::Dense),
                "sparse" => Ok(MlpKind::Moe),
                other => Err(Error::Json(format!(
                    "text_config.mlp_layer_types: unknown type {other:?}"
                ))),
            })
            .collect::<Result<Vec<_>>>()?;
        let text = TextConfig {
            model_type: t.str("model_type")?,
            hidden_size: t.u64("hidden_size")?,
            num_hidden_layers: t.u64("num_hidden_layers")?,
            vocab_size: t.u64("vocab_size")?,
            max_position_embeddings: t.u64("max_position_embeddings")?,
            rms_norm_eps: t.f64("rms_norm_eps")?,
            hidden_act: t.str("hidden_act")?,
            tie_word_embeddings: t.bool("tie_word_embeddings")?,
            eos_token_ids: t.u64_or_list("eos_token_id")?,
            pad_token_id: t.u64("pad_token_id")?,
            layer_types,
            mlp_layer_types,
            indexer_types: t.str_list("indexer_types")?,
            kda: KdaConfig {
                num_heads: lin.u64("num_heads")?,
                head_dim: lin.u64("head_dim")?,
                short_conv_kernel_size: lin.u64("short_conv_kernel_size")?,
                gate_lower_bound: lin.f64("gate_lower_bound")?,
                kda_layers: lin.u64_list("kda_layers")?,
                full_attn_layers: lin.u64_list("full_attn_layers")?,
            },
            mla: MlaConfig {
                num_attention_heads: t.u64("num_attention_heads")?,
                num_key_value_heads: t.u64("num_key_value_heads")?,
                q_lora_rank: t.u64("q_lora_rank")?,
                kv_lora_rank: t.u64("kv_lora_rank")?,
                qk_nope_head_dim: t.u64("qk_nope_head_dim")?,
                qk_rope_head_dim: t.u64("qk_rope_head_dim")?,
                qk_head_dim: t.u64("qk_head_dim")?,
                v_head_dim: t.u64("v_head_dim")?,
                use_nope: t.bool("mla_use_nope")?,
                attention_bias: t.bool("attention_bias")?,
            },
            indexer: IndexerConfig {
                n_heads: t.u64("index_n_heads")?,
                head_dim: t.u64("index_head_dim")?,
                topk: t.u64("index_topk")?,
                kpool: t.u64("index_kpool")?,
                kpool_compress: t.bool("index_kpool_compress")?,
                always_select_tail: t.bool("index_kpool_always_select_tail")?,
                share_for_mtp_iteration: t.bool("index_share_for_mtp_iteration")?,
            },
            mhc: MhcConfig {
                enabled: t.bool("mhc")?,
                hc_mult: t.u64("hc_mult")?,
                sinkhorn_iters: t.u64("hc_sinkhorn_iters")?,
                eps: t.f64("hc_eps")?,
            },
            moe: MoeConfig {
                n_routed_experts: t.u64("n_routed_experts")?,
                num_experts_per_tok: t.u64("num_experts_per_tok")?,
                moe_intermediate_size: t.u64("moe_intermediate_size")?,
                n_shared_experts: t.u64("n_shared_experts")?,
                first_k_dense_replace: t.u64("first_k_dense_replace")?,
                dense_intermediate_size: t.u64("intermediate_size")?,
                scoring_func: t.str("scoring_func")?,
                topk_method: t.str("topk_method")?,
                routed_scaling_factor: t.f64("routed_scaling_factor")?,
                norm_topk_prob: t.bool("norm_topk_prob")?,
                n_group: t.u64("n_group")?,
                topk_group: t.u64("topk_group")?,
                swiglu_limit: t.f64("swiglu_limit")?,
                router_dtype: t.str("moe_router_dtype")?,
            },
            num_nextn_predict_layers: t.u64("num_nextn_predict_layers")?,
        };
        let vc = root.obj("vision_config")?;
        let vision = VisionConfig {
            depth: vc.u64("depth")?,
            hidden_size: vc.u64("hidden_size")?,
            num_heads: vc.u64("num_heads")?,
            intermediate_size: vc.u64("intermediate_size")?,
            out_hidden_size: vc.u64("out_hidden_size")?,
            projection_intermediate_size: vc.u64("projection_intermediate_size")?,
            patch_size: vc.u64("patch_size")?,
            temporal_patch_size: vc.u64("temporal_patch_size")?,
            spatial_merge_size: vc.u64("spatial_merge_size")?,
            in_channels: vc.u64("in_channels")?,
            image_size: vc.u64("image_size")?,
            attention_bias: vc.bool("attention_bias")?,
            swiglu_limit: vc.f64("swiglu_limit")?,
            rms_norm_eps: vc.f64("rms_norm_eps")?,
        };
        let quantization = match root.opt("quantization_config") {
            None => Quantization::None,
            Some(_) => parse_quantization(&root.obj("quantization_config")?)?,
        };
        Ok(ModelConfig {
            model_type: root.str("model_type")?,
            architectures: root.str_list("architectures")?,
            tie_word_embeddings: root.bool("tie_word_embeddings")?,
            text,
            vision,
            quantization,
        })
    }

    /// Check the GLM-5.3-Flash invariants. Lists every violation.
    pub fn validate(&self) -> Result<()> {
        use glm53_flash as g;
        let mut errs = Vec::new();
        let mut eq = |what: &str, got: String, want: String| {
            if got != want {
                errs.push(format!("{what}: expected {want}, got {got}"));
            }
        };
        let t = &self.text;
        eq("model_type", self.model_type.clone(), "glm5_next".into());
        eq(
            "text_config.model_type",
            t.model_type.clone(),
            "glm5_next_text".into(),
        );
        eq(
            "tie_word_embeddings",
            format!("{}", self.tie_word_embeddings || t.tie_word_embeddings),
            "false".into(),
        );
        eq(
            "text_config.num_hidden_layers",
            t.num_hidden_layers.to_string(),
            g::NUM_HIDDEN_LAYERS.to_string(),
        );
        eq(
            "text_config.hidden_size",
            t.hidden_size.to_string(),
            g::HIDDEN_SIZE.to_string(),
        );
        eq(
            "text_config.vocab_size",
            t.vocab_size.to_string(),
            g::VOCAB_SIZE.to_string(),
        );
        eq(
            "text_config.max_position_embeddings",
            t.max_position_embeddings.to_string(),
            g::MAX_POSITIONS.to_string(),
        );
        eq(
            "text_config.hidden_act",
            t.hidden_act.clone(),
            "silu".into(),
        );
        // Layer layout: 34 KDA + 11 DSA, and the three lists that describe it agree.
        let n = t.num_hidden_layers as usize;
        eq(
            "text_config.layer_types (length)",
            t.layer_types.len().to_string(),
            n.to_string(),
        );
        let kda = t.kda_layer_ids();
        let dsa = t.dsa_layer_ids();
        eq(
            "linear_attention layers",
            kda.len().to_string(),
            g::KDA_LAYERS.to_string(),
        );
        eq(
            "deepseek_sparse_attention layers",
            dsa.len().to_string(),
            g::DSA_LAYERS.to_string(),
        );
        eq(
            "linear_attn_config.kda_layers",
            format!("{:?}", t.kda.kda_layers),
            format!("{kda:?}"),
        );
        eq(
            "linear_attn_config.full_attn_layers",
            format!("{:?}", t.kda.full_attn_layers),
            format!("{dsa:?}"),
        );
        // KDA
        eq(
            "linear_attn_config.num_heads",
            t.kda.num_heads.to_string(),
            g::KDA_HEADS.to_string(),
        );
        eq(
            "linear_attn_config.head_dim",
            t.kda.head_dim.to_string(),
            g::KDA_HEAD_DIM.to_string(),
        );
        eq(
            "linear_attn_config.short_conv_kernel_size",
            t.kda.short_conv_kernel_size.to_string(),
            g::KDA_CONV_KERNEL.to_string(),
        );
        // MLA (no RoPE)
        let m = &t.mla;
        eq(
            "text_config.num_attention_heads",
            m.num_attention_heads.to_string(),
            g::MLA_HEADS.to_string(),
        );
        eq(
            "text_config.num_key_value_heads",
            m.num_key_value_heads.to_string(),
            g::MLA_HEADS.to_string(),
        );
        eq(
            "text_config.q_lora_rank",
            m.q_lora_rank.to_string(),
            g::Q_LORA_RANK.to_string(),
        );
        eq(
            "text_config.kv_lora_rank",
            m.kv_lora_rank.to_string(),
            g::KV_LORA_RANK.to_string(),
        );
        eq(
            "text_config.qk_nope_head_dim",
            m.qk_nope_head_dim.to_string(),
            g::QK_NOPE_HEAD_DIM.to_string(),
        );
        eq(
            "text_config.qk_rope_head_dim",
            m.qk_rope_head_dim.to_string(),
            g::QK_ROPE_HEAD_DIM.to_string(),
        );
        eq(
            "text_config.qk_head_dim",
            m.qk_head_dim.to_string(),
            (g::QK_NOPE_HEAD_DIM + g::QK_ROPE_HEAD_DIM).to_string(),
        );
        eq(
            "text_config.v_head_dim",
            m.v_head_dim.to_string(),
            g::V_HEAD_DIM.to_string(),
        );
        eq(
            "text_config.mla_use_nope",
            m.use_nope.to_string(),
            "true".into(),
        );
        eq(
            "text_config.attention_bias",
            m.attention_bias.to_string(),
            "false".into(),
        );
        // Indexer: k-pool 4 with the tail; every DSA layer runs its own.
        let ix = &t.indexer;
        eq(
            "text_config.index_n_heads",
            ix.n_heads.to_string(),
            g::INDEX_HEADS.to_string(),
        );
        eq(
            "text_config.index_head_dim",
            ix.head_dim.to_string(),
            g::INDEX_HEAD_DIM.to_string(),
        );
        eq(
            "text_config.index_topk",
            ix.topk.to_string(),
            g::INDEX_TOPK.to_string(),
        );
        eq(
            "text_config.index_kpool",
            ix.kpool.to_string(),
            g::INDEX_KPOOL.to_string(),
        );
        eq(
            "text_config.index_kpool_compress",
            ix.kpool_compress.to_string(),
            "true".into(),
        );
        eq(
            "text_config.index_kpool_always_select_tail",
            ix.always_select_tail.to_string(),
            "true".into(),
        );
        eq(
            "text_config.indexer_types (length)",
            t.indexer_types.len().to_string(),
            n.to_string(),
        );
        let shared: Vec<u64> = dsa
            .iter()
            .copied()
            .filter(|&l| t.indexer_types.get(l as usize).map(String::as_str) != Some("full"))
            .collect();
        eq(
            "DSA layers without their own indexer",
            format!("{shared:?}"),
            "[]".into(),
        );
        // mHC
        eq("text_config.mhc", t.mhc.enabled.to_string(), "true".into());
        eq(
            "text_config.hc_mult",
            t.mhc.hc_mult.to_string(),
            g::HC_MULT.to_string(),
        );
        eq(
            "text_config.hc_sinkhorn_iters",
            t.mhc.sinkhorn_iters.to_string(),
            g::HC_SINKHORN_ITERS.to_string(),
        );
        // MoE
        let e = &t.moe;
        eq(
            "text_config.n_routed_experts",
            e.n_routed_experts.to_string(),
            g::ROUTED_EXPERTS.to_string(),
        );
        eq(
            "text_config.num_experts_per_tok",
            e.num_experts_per_tok.to_string(),
            g::EXPERTS_PER_TOKEN.to_string(),
        );
        eq(
            "text_config.moe_intermediate_size",
            e.moe_intermediate_size.to_string(),
            g::MOE_INTERMEDIATE.to_string(),
        );
        eq(
            "text_config.n_shared_experts",
            e.n_shared_experts.to_string(),
            g::SHARED_EXPERTS.to_string(),
        );
        eq(
            "text_config.first_k_dense_replace",
            e.first_k_dense_replace.to_string(),
            g::FIRST_K_DENSE.to_string(),
        );
        eq(
            "text_config.intermediate_size",
            e.dense_intermediate_size.to_string(),
            g::DENSE_INTERMEDIATE.to_string(),
        );
        let want_mlp: Vec<MlpKind> = (0..n as u64)
            .map(|l| {
                if l < g::FIRST_K_DENSE {
                    MlpKind::Dense
                } else {
                    MlpKind::Moe
                }
            })
            .collect();
        eq(
            "text_config.mlp_layer_types",
            format!("{:?}", t.mlp_layer_types),
            format!("{want_mlp:?}"),
        );
        eq(
            "text_config.scoring_func",
            e.scoring_func.clone(),
            "sigmoid".into(),
        );
        eq(
            "text_config.topk_method",
            e.topk_method.clone(),
            "noaux_tc".into(),
        );
        eq(
            "text_config.routed_scaling_factor",
            e.routed_scaling_factor.to_string(),
            g::ROUTED_SCALING_FACTOR.to_string(),
        );
        eq(
            "text_config.norm_topk_prob",
            e.norm_topk_prob.to_string(),
            "true".into(),
        );
        eq("text_config.n_group", e.n_group.to_string(), "1".into());
        eq(
            "text_config.topk_group",
            e.topk_group.to_string(),
            "1".into(),
        );
        eq(
            "text_config.swiglu_limit",
            e.swiglu_limit.to_string(),
            g::SWIGLU_LIMIT.to_string(),
        );
        eq(
            "text_config.moe_router_dtype",
            e.router_dtype.clone(),
            "float32".into(),
        );
        eq(
            "text_config.num_nextn_predict_layers",
            t.num_nextn_predict_layers.to_string(),
            g::MTP_LAYERS.to_string(),
        );
        // Vision tower (24 layers, 1024 hidden, patch 14, merge 2), feeding the text hidden size.
        let v = &self.vision;
        eq("vision_config.depth", v.depth.to_string(), "24".into());
        eq(
            "vision_config.hidden_size",
            v.hidden_size.to_string(),
            "1024".into(),
        );
        eq(
            "vision_config.num_heads",
            v.num_heads.to_string(),
            "16".into(),
        );
        eq(
            "vision_config.intermediate_size",
            v.intermediate_size.to_string(),
            "4096".into(),
        );
        eq(
            "vision_config.out_hidden_size",
            v.out_hidden_size.to_string(),
            t.hidden_size.to_string(),
        );
        eq(
            "vision_config.projection_intermediate_size",
            v.projection_intermediate_size.to_string(),
            "10240".into(),
        );
        eq(
            "vision_config.patch_size",
            v.patch_size.to_string(),
            "14".into(),
        );
        eq(
            "vision_config.temporal_patch_size",
            v.temporal_patch_size.to_string(),
            "2".into(),
        );
        eq(
            "vision_config.spatial_merge_size",
            v.spatial_merge_size.to_string(),
            "2".into(),
        );
        eq(
            "vision_config.in_channels",
            v.in_channels.to_string(),
            "3".into(),
        );
        if errs.is_empty() {
            Ok(())
        } else {
            Err(Error::Invariant(errs))
        }
    }
}

fn parse_quantization(q: &Obj<'_>) -> Result<Quantization> {
    let method = q.str("quant_method")?;
    Ok(match method.as_str() {
        "fp8" => {
            let b = q.u64_list("weight_block_size")?;
            if b.len() != 2 {
                return Err(q.wrong("weight_block_size", "two integers"));
            }
            Quantization::Fp8Block {
                fmt: q.str("fmt")?,
                block: [b[0], b[1]],
                activation_scheme: q.str("activation_scheme")?,
            }
        }
        "exl3" => Quantization::Exl3 {
            bits: q.u64("bits")?,
            codebook: q.str("codebook")?,
        },
        "modelopt" => {
            let groups = q.obj("config_groups")?;
            let pairs = groups.v.as_object().unwrap_or_default();
            if pairs.len() != 1 {
                return Err(q.wrong("config_groups", "exactly one group"));
            }
            let g = Obj {
                path: groups.at(&pairs[0].0),
                v: &pairs[0].1,
            };
            let w = g.obj("weights")?;
            Quantization::Modelopt {
                algo: q.str("quant_algo")?,
                group_size: w.u64("group_size")?,
            }
        }
        other => Quantization::Other(other.to_string()),
    })
}

impl TextConfig {
    /// Attention type of `layer`. MTP layers (after the decoder layers) use DSA.
    pub fn attn_kind(&self, layer: u64) -> AttnKind {
        self.layer_types
            .get(layer as usize)
            .copied()
            .unwrap_or(AttnKind::Dsa)
    }
    /// Whether `layer` has a mixture-of-experts MLP. MTP layers do.
    pub fn is_moe(&self, layer: u64) -> bool {
        self.mlp_layer_types
            .get(layer as usize)
            .is_none_or(|&k| k == MlpKind::Moe)
    }
    pub fn kda_layer_ids(&self) -> Vec<u64> {
        self.layer_ids(AttnKind::Kda)
    }
    pub fn dsa_layer_ids(&self) -> Vec<u64> {
        self.layer_ids(AttnKind::Dsa)
    }
    fn layer_ids(&self, kind: AttnKind) -> Vec<u64> {
        (0..self.layer_types.len() as u64)
            .filter(|&l| self.layer_types[l as usize] == kind)
            .collect()
    }
    /// Decoder layers with routed experts (not counting MTP).
    pub fn moe_layers(&self) -> Vec<u64> {
        (0..self.num_hidden_layers)
            .filter(|&l| self.is_moe(l))
            .collect()
    }
    /// The MTP layer indices, after the decoder layers.
    pub fn mtp_layers(&self) -> Range<u64> {
        self.num_hidden_layers..self.num_hidden_layers + self.num_nextn_predict_layers
    }
    /// KDA channels per projection: heads x head dim (8,192).
    pub fn kda_dim(&self) -> u64 {
        self.kda.num_heads * self.kda.head_dim
    }
    /// Width of an mHC projection's output: pre (N) + post (N) + comb (N x N).
    pub fn hc_mix(&self) -> u64 {
        (2 + self.mhc.hc_mult) * self.mhc.hc_mult
    }
}

/// The DFlash2 drafter's `config.json` (`incoai/GLM-5.3-Flash-DFlash2`): five
/// Qwen3-style layers with grouped two-tap dynamic convolutions and a candidate
/// selector, reading the target's hidden states at `target_layer_ids`.
#[derive(Clone, Debug, PartialEq)]
pub struct DraftConfig {
    pub architectures: Vec<String>,
    pub hidden_size: u64,
    pub num_hidden_layers: u64,
    pub num_attention_heads: u64,
    pub num_key_value_heads: u64,
    pub head_dim: u64,
    pub intermediate_size: u64,
    pub vocab_size: u64,
    pub sliding_window: u64,
    pub layer_types: Vec<String>,
    pub num_target_layers: u64,
    pub block_size: u64,
    pub conv_group_size: u64,
    pub conv_kernel_size: u64,
    pub selector_rank: u64,
    pub selector_top_k: u64,
    pub mask_token_id: u64,
    pub target_layer_ids: Vec<u64>,
}

/// Branches of each DFlash2 dynamic convolution. Not a config field: the
/// published checkpoint stores `base_kernel` as [2, conv_kernel_size, hidden]
/// and `kernel_projection` with 2 x conv_kernel_size x (hidden / group) rows.
pub const DFLASH2_CONV_BRANCHES: u64 = 2;

impl DraftConfig {
    pub fn load(path: &Path) -> Result<DraftConfig> {
        DraftConfig::from_json(&read_json(path)?)
    }

    pub fn parse(text: &str) -> Result<DraftConfig> {
        let v = json::parse(text).map_err(Error::Json)?;
        json::check_unique_keys(&v).map_err(Error::Json)?;
        DraftConfig::from_json(&v)
    }

    pub fn from_json(v: &Json) -> Result<DraftConfig> {
        let r = Obj::root(v)?;
        let d = r.obj("dflash_config")?;
        Ok(DraftConfig {
            architectures: r.str_list("architectures")?,
            hidden_size: r.u64("hidden_size")?,
            num_hidden_layers: r.u64("num_hidden_layers")?,
            num_attention_heads: r.u64("num_attention_heads")?,
            num_key_value_heads: r.u64("num_key_value_heads")?,
            head_dim: r.u64("head_dim")?,
            intermediate_size: r.u64("intermediate_size")?,
            vocab_size: r.u64("vocab_size")?,
            sliding_window: r.u64("sliding_window")?,
            layer_types: r.str_list("layer_types")?,
            num_target_layers: r.u64("num_target_layers")?,
            block_size: d.u64("block_size")?,
            conv_group_size: d.u64("conv_group_size")?,
            conv_kernel_size: d.u64("conv_kernel_size")?,
            selector_rank: d.u64("selector_rank")?,
            selector_top_k: d.u64("selector_top_k")?,
            mask_token_id: d.u64("mask_token_id")?,
            target_layer_ids: d.u64_list("target_layer_ids")?,
        })
    }

    /// Check that the drafter fits `target` and the shapes the engine uses.
    pub fn validate(&self, target: &ModelConfig) -> Result<()> {
        let t = &target.text;
        let mut errs = Vec::new();
        let mut eq = |what: &str, got: String, want: String| {
            if got != want {
                errs.push(format!("drafter {what}: expected {want}, got {got}"));
            }
        };
        eq(
            "architectures",
            format!("{:?}", self.architectures),
            "[\"DFlash2DraftModel\"]".into(),
        );
        eq(
            "hidden_size",
            self.hidden_size.to_string(),
            t.hidden_size.to_string(),
        );
        eq(
            "vocab_size",
            self.vocab_size.to_string(),
            t.vocab_size.to_string(),
        );
        eq(
            "num_target_layers",
            self.num_target_layers.to_string(),
            t.num_hidden_layers.to_string(),
        );
        eq(
            "layer_types (length)",
            self.layer_types.len().to_string(),
            self.num_hidden_layers.to_string(),
        );
        let not_sliding = self
            .layer_types
            .iter()
            .filter(|s| *s != "sliding_attention")
            .count();
        eq("non-sliding layers", not_sliding.to_string(), "0".into());
        let bad_taps: Vec<u64> = self
            .target_layer_ids
            .iter()
            .copied()
            .filter(|&l| l >= t.num_hidden_layers)
            .collect();
        eq(
            "target_layer_ids beyond the target",
            format!("{bad_taps:?}"),
            "[]".into(),
        );
        eq(
            "hidden_size % conv_group_size",
            (self.hidden_size % self.conv_group_size.max(1)).to_string(),
            "0".into(),
        );
        if errs.is_empty() {
            Ok(())
        } else {
            Err(Error::Invariant(errs))
        }
    }

    /// Tokens of draft KV kept per request: the sliding window plus one block.
    pub fn kv_window_tokens(&self) -> u64 {
        self.sliding_window + self.block_size
    }

    /// Draft KV per slot: layers x KV heads x head dim x (K, V), BF16, over
    /// [`Self::kv_window_tokens`].
    pub fn kv_bytes_per_slot(&self) -> u64 {
        self.num_hidden_layers
            * self.num_key_value_heads
            * self.head_dim
            * 2
            * 2
            * self.kv_window_tokens()
    }

    /// Every tensor of the drafter checkpoint (names, dtypes, shapes), derived
    /// from the config. The drafter shares the target's embedding and LM head,
    /// so neither appears here.
    pub fn tensors(&self) -> Vec<(String, DType, Vec<u64>)> {
        let (h, bf) = (self.hidden_size, DType::BF16);
        let conv_rows = DFLASH2_CONV_BRANCHES * self.conv_kernel_size * (h / self.conv_group_size);
        let mut out = vec![
            (
                "candidate_selector.hidden_projection.weight".to_string(),
                bf,
                vec![self.selector_rank, h],
            ),
            (
                "candidate_selector.predecessor_codebook".to_string(),
                bf,
                vec![self.vocab_size, self.selector_rank],
            ),
            (
                "candidate_selector.successor_codebook".to_string(),
                bf,
                vec![self.vocab_size, self.selector_rank],
            ),
            (
                "fc.weight".to_string(),
                bf,
                vec![h, h * self.target_layer_ids.len() as u64],
            ),
            ("hidden_norm.weight".to_string(), bf, vec![h]),
            ("norm.weight".to_string(), bf, vec![h]),
        ];
        let q = self.num_attention_heads * self.head_dim;
        let kv = self.num_key_value_heads * self.head_dim;
        for l in 0..self.num_hidden_layers {
            let p = format!("layers.{l}.");
            for conv in ["attention_conv", "mlp_conv"] {
                out.push((
                    format!("{p}{conv}.base_kernel"),
                    bf,
                    vec![DFLASH2_CONV_BRANCHES, self.conv_kernel_size, h],
                ));
                out.push((
                    format!("{p}{conv}.kernel_projection.weight"),
                    bf,
                    vec![conv_rows, h],
                ));
            }
            out.push((format!("{p}input_layernorm.weight"), bf, vec![h]));
            out.push((format!("{p}post_attention_layernorm.weight"), bf, vec![h]));
            out.push((
                format!("{p}mlp.gate_proj.weight"),
                bf,
                vec![self.intermediate_size, h],
            ));
            out.push((
                format!("{p}mlp.up_proj.weight"),
                bf,
                vec![self.intermediate_size, h],
            ));
            out.push((
                format!("{p}mlp.down_proj.weight"),
                bf,
                vec![h, self.intermediate_size],
            ));
            out.push((format!("{p}self_attn.q_proj.weight"), bf, vec![q, h]));
            out.push((format!("{p}self_attn.k_proj.weight"), bf, vec![kv, h]));
            out.push((format!("{p}self_attn.v_proj.weight"), bf, vec![kv, h]));
            out.push((format!("{p}self_attn.o_proj.weight"), bf, vec![h, q]));
            out.push((
                format!("{p}self_attn.q_norm.weight"),
                bf,
                vec![self.head_dim],
            ));
            out.push((
                format!("{p}self_attn.k_norm.weight"),
                bf,
                vec![self.head_dim],
            ));
        }
        out.sort();
        out
    }

    /// Bytes of the drafter's own weights.
    pub fn weight_bytes(&self) -> u64 {
        self.tensors()
            .iter()
            .map(|(_, d, s)| d.size() * s.iter().product::<u64>())
            .sum()
    }
}
