//! The tensor catalog: every tensor of a GLM-5.3-Flash checkpoint with its
//! role (layer, component, part), dtype, shape, file and byte offsets.
//!
//! Three checkpoint formats are covered:
//!
//! - [`CheckpointFormat::OfficialFp8`] (`zai-org/GLM-5.3-Flash`): FP8 E4M3
//!   weights with `weight_scale_inv`, one F32 scale per 128 x 128 block, for
//!   the DSA projections (not `kv_b_proj`), the shared and dense MLPs and the
//!   routed experts. KDA attention, the indexer, routers, norms, embedding and
//!   LM head are BF16 (or F32 for a few small parameters).
//! - [`CheckpointFormat::Exl3`] (`Mia-AiLab/GLM-5.3-Flash-EXL3-TR3-4bpw`, the
//!   same weights as `brandonmusic/GLM-5.3-Flash-tr3-4bpw`): routed experts as
//!   ExLlamaV3 trellis tensors with the `mcg` codebook (`trellis`, `suh`,
//!   `svh`, `mcg`); every other tensor BF16.
//! - [`CheckpointFormat::Nvfp4`] (`LibertAIDAI/GLM-5.3-Flash-NVFP4`): routed
//!   experts in modelopt NVFP4 (packed `U8` weight, `F8_E4M3` scale per 16
//!   values, F32 `weight_scale_2` and `input_scale`); every other tensor BF16.
//!
//! The expected tensor list of each format is derived from `config.json`
//! ([`spec`]). A catalog read from checkpoint headers must match it exactly:
//! same names, dtypes and shapes, nothing missing, nothing extra. Quantized
//! linears are paired with their scales ([`Linear`]).

use std::cmp::Ordering;
use std::collections::BTreeMap;

use crate::config::{AttnKind, ModelConfig, Quantization, TextConfig};
use crate::dtype::{numel, DType};
use crate::error::{Error, Result};
use crate::safetensors::{Shard, TensorEntry};

/// Name prefix of the decoder (and MTP) layers.
pub const LAYERS: &str = "model.language_model.layers.";
pub const EMBED: &str = "model.language_model.embed_tokens.weight";
pub const FINAL_NORM: &str = "model.language_model.norm.weight";
pub const LM_HEAD: &str = "lm_head.weight";
/// Name prefix of the vision tower.
pub const VISION: &str = "model.visual.";

/// Rows and columns covered by one F32 scale of an official FP8 weight.
pub const FP8_BLOCK: u64 = 128;
/// Values per `F8_E4M3` scale in modelopt NVFP4.
pub const NVFP4_GROUP: u64 = 16;
/// Edge of an EXL3 trellis tile (16 x 16 weights).
pub const EXL3_TILE: u64 = 16;
/// Size of the Hadamard blocks EXL3 rotates inputs and outputs with.
pub const EXL3_HADAMARD: u64 = 128;

/// Which published checkpoint layout a catalog describes.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum CheckpointFormat {
    OfficialFp8,
    Exl3 { bits: u64 },
    Nvfp4,
}

impl CheckpointFormat {
    /// How the routed experts are stored.
    pub fn expert_quant(self) -> Quant {
        match self {
            CheckpointFormat::OfficialFp8 => Quant::Fp8Block,
            CheckpointFormat::Exl3 { bits } => Quant::Exl3 { bits },
            CheckpointFormat::Nvfp4 => Quant::Nvfp4,
        }
    }

    /// The format a `quantization_config` announces, if it is one of the three.
    pub fn from_config(cfg: &ModelConfig) -> Option<CheckpointFormat> {
        match &cfg.quantization {
            Quantization::Fp8Block {
                block: [FP8_BLOCK, FP8_BLOCK],
                ..
            } => Some(CheckpointFormat::OfficialFp8),
            Quantization::Exl3 { bits, codebook } if codebook == "mcg" => {
                Some(CheckpointFormat::Exl3 { bits: *bits })
            }
            Quantization::Modelopt {
                algo,
                group_size: NVFP4_GROUP,
            } if algo == "NVFP4" => Some(CheckpointFormat::Nvfp4),
            _ => None,
        }
    }

    pub fn label(self) -> String {
        match self {
            CheckpointFormat::OfficialFp8 => "official FP8".into(),
            CheckpointFormat::Exl3 { bits } => format!("EXL3 {bits} bpw"),
            CheckpointFormat::Nvfp4 => "NVFP4".into(),
        }
    }
}

/// How one linear layer's weight is stored.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Quant {
    Bf16,
    /// FP8 E4M3 weight + F32 `weight_scale_inv` per 128 x 128 block.
    Fp8Block,
    /// ExLlamaV3 trellis (`mcg` codebook) at `bits` per weight.
    Exl3 {
        bits: u64,
    },
    /// modelopt NVFP4: packed E2M1 pairs, E4M3 scale per 16, F32 global scales.
    Nvfp4,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Proj {
    Gate,
    Up,
    Down,
}

impl Proj {
    pub const ALL: [Proj; 3] = [Proj::Gate, Proj::Up, Proj::Down];
    pub fn stem(self) -> &'static str {
        match self {
            Proj::Gate => "gate_proj",
            Proj::Up => "up_proj",
            Proj::Down => "down_proj",
        }
    }
    fn from_stem(s: &str) -> Option<Proj> {
        Proj::ALL.into_iter().find(|p| p.stem() == s)
    }
}

/// A KDA (linear attention) tensor.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum KdaPart {
    Q,
    K,
    V,
    QConv,
    KConv,
    VConv,
    /// Decay (forget) gate, low-rank down and up projections.
    FA,
    FB,
    /// Output gate, low-rank down and up projections.
    GA,
    GB,
    /// Beta (delta-rule write strength).
    B,
    O,
    ALog,
    DtBias,
    /// Gated RMSNorm on the head outputs.
    ONorm,
}

/// A DSA (MLA) tensor.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum DsaPart {
    QA,
    QANorm,
    QB,
    KvA,
    KvANorm,
    KvB,
    O,
}

/// A DSA indexer tensor.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum IndexerPart {
    WqB,
    Wk,
    /// LayerNorm on the keys (weight and bias).
    KNorm,
    WeightsProj,
    /// The k-pool gate (`index_kpool_compress_gate`).
    KpoolGate,
    /// The k-pool position embedding (`index_kpool_compress_ape`).
    KpoolApe,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum MhcSite {
    Attn,
    Ffn,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum MhcParam {
    Base,
    Fn,
    Scale,
}

/// A tensor only the MTP layer has.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum MtpPart {
    EhProj,
    ENorm,
    HNorm,
    SharedHeadNorm,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Component {
    Embedding,
    LmHead,
    FinalNorm,
    InputNorm,
    PostAttnNorm,
    Mhc(MhcSite, MhcParam),
    Kda(KdaPart),
    Dsa(DsaPart),
    Indexer(IndexerPart),
    /// Router weight ([`Part::Weight`]) and correction bias ([`Part::Param`]).
    Router,
    DenseMlp(Proj),
    SharedExpert(Proj),
    RoutedExpert {
        expert: u64,
        proj: Proj,
    },
    Mtp(MtpPart),
    Vision,
}

/// Which stored piece of a component a tensor is.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Part {
    /// `*.weight` in its storage dtype (BF16 or FP8 E4M3).
    Weight,
    Bias,
    /// A bare parameter (`A_log`, `hc_attn_base`, ...).
    Param,
    /// `weight_scale_inv`: F32 per 128 x 128 block of an FP8 weight.
    Fp8ScaleInv,
    /// EXL3 `trellis`: I16 [in/16, out/16, 16 * bits].
    Exl3Trellis,
    /// EXL3 `suh`: F16 input scales [in].
    Exl3Suh,
    /// EXL3 `svh`: F16 output scales [out].
    Exl3Svh,
    /// EXL3 `mcg`: I32 [1], marks the `mcg` codebook.
    Exl3Mcg,
    /// NVFP4 `weight`: U8 [out, in/2], two E2M1 values per byte.
    Nvfp4Weight,
    /// NVFP4 `weight_scale`: F8_E4M3 [out, in/16].
    Nvfp4Scale,
    /// NVFP4 `weight_scale_2`: F32 scalar, the per-tensor global scale.
    Nvfp4Scale2,
    /// NVFP4 `input_scale`: F32 scalar, the activation global scale.
    Nvfp4InputScale,
}

impl Part {
    /// The name suffix after the module name.
    pub fn suffix(self) -> &'static str {
        match self {
            Part::Weight | Part::Nvfp4Weight => "weight",
            Part::Bias => "bias",
            Part::Param => "",
            Part::Fp8ScaleInv => "weight_scale_inv",
            Part::Exl3Trellis => "trellis",
            Part::Exl3Suh => "suh",
            Part::Exl3Svh => "svh",
            Part::Exl3Mcg => "mcg",
            Part::Nvfp4Scale => "weight_scale",
            Part::Nvfp4Scale2 => "weight_scale_2",
            Part::Nvfp4InputScale => "input_scale",
        }
    }
}

/// What a tensor is.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Role {
    /// Decoder or MTP layer; `None` for top-level and vision tensors.
    pub layer: Option<u64>,
    pub component: Component,
    pub part: Part,
}

/// Byte-accounting groups (the rows of docs/SIZING.md section 4, split finer).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Group {
    KdaAttention,
    DsaAttention,
    DsaIndexer,
    SharedExperts,
    DenseMlp,
    Routers,
    Mhc,
    /// Decoder-layer norms and the final norm (not the DSA-internal and KDA
    /// output norms, which count with their attention).
    Norms,
    LmHead,
    Embedding,
    RoutedExperts,
    /// The MTP layer, except its routed experts.
    MtpLayer,
    MtpRoutedExperts,
    Vision,
}

impl Group {
    pub const ALL: [Group; 14] = [
        Group::KdaAttention,
        Group::DsaAttention,
        Group::DsaIndexer,
        Group::SharedExperts,
        Group::DenseMlp,
        Group::Routers,
        Group::Mhc,
        Group::Norms,
        Group::LmHead,
        Group::Embedding,
        Group::RoutedExperts,
        Group::MtpLayer,
        Group::MtpRoutedExperts,
        Group::Vision,
    ];

    pub fn label(self) -> &'static str {
        match self {
            Group::KdaAttention => "KDA attention",
            Group::DsaAttention => "DSA attention",
            Group::DsaIndexer => "DSA indexer",
            Group::SharedExperts => "shared experts",
            Group::DenseMlp => "dense MLP",
            Group::Routers => "routers",
            Group::Mhc => "mHC",
            Group::Norms => "norms",
            Group::LmHead => "LM head",
            Group::Embedding => "embedding",
            Group::RoutedExperts => "routed experts",
            Group::MtpLayer => "MTP layer (non-expert)",
            Group::MtpRoutedExperts => "MTP routed experts",
            Group::Vision => "vision tower",
        }
    }

    /// Groups that live on the coordinator GPU in every layout (the embedding
    /// depends on the layout; MTP and vision are optional).
    pub fn always_on_coordinator(self) -> bool {
        !matches!(
            self,
            Group::Embedding
                | Group::RoutedExperts
                | Group::MtpLayer
                | Group::MtpRoutedExperts
                | Group::Vision
        )
    }
}

fn group_of(role: &Role, text: &TextConfig) -> Group {
    let mtp = role.layer.is_some_and(|l| l >= text.num_hidden_layers);
    match role.component {
        Component::Vision => Group::Vision,
        Component::RoutedExpert { .. } if mtp => Group::MtpRoutedExperts,
        _ if mtp => Group::MtpLayer,
        Component::Embedding => Group::Embedding,
        Component::LmHead => Group::LmHead,
        Component::FinalNorm | Component::InputNorm | Component::PostAttnNorm => Group::Norms,
        Component::Mhc(..) => Group::Mhc,
        Component::Kda(_) => Group::KdaAttention,
        Component::Dsa(_) => Group::DsaAttention,
        Component::Indexer(_) => Group::DsaIndexer,
        Component::Router => Group::Routers,
        Component::DenseMlp(_) => Group::DenseMlp,
        Component::SharedExpert(_) => Group::SharedExperts,
        Component::RoutedExpert { .. } => Group::RoutedExperts,
        Component::Mtp(_) => Group::MtpLayer,
    }
}

/// One expected tensor.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TensorSpec {
    pub name: String,
    pub role: Role,
    pub dtype: DType,
    pub shape: Vec<u64>,
}

/// One expected linear layer, `y = x W^T` with W [out_features, in_features].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LinearSpec {
    /// Tensor-name prefix before the part suffix, e.g.
    /// `model.language_model.layers.3.mlp.experts.0.gate_proj`.
    pub module: String,
    pub layer: Option<u64>,
    pub component: Component,
    pub out_features: u64,
    pub in_features: u64,
    pub quant: Quant,
}

impl LinearSpec {
    /// The parts this linear is stored as.
    pub fn parts(&self) -> &'static [Part] {
        match self.quant {
            Quant::Bf16 => &[Part::Weight],
            Quant::Fp8Block => &[Part::Weight, Part::Fp8ScaleInv],
            Quant::Exl3 { .. } => &[
                Part::Exl3Trellis,
                Part::Exl3Suh,
                Part::Exl3Svh,
                Part::Exl3Mcg,
            ],
            Quant::Nvfp4 => &[
                Part::Nvfp4Weight,
                Part::Nvfp4Scale,
                Part::Nvfp4Scale2,
                Part::Nvfp4InputScale,
            ],
        }
    }
}

struct Builder<'a> {
    cfg: &'a ModelConfig,
    fmt: CheckpointFormat,
    tensors: Vec<TensorSpec>,
    linears: Vec<LinearSpec>,
}

fn ceil_div(a: u64, b: u64) -> u64 {
    a.div_ceil(b)
}

impl Builder<'_> {
    fn param(
        &mut self,
        name: String,
        layer: Option<u64>,
        component: Component,
        part: Part,
        dtype: DType,
        shape: Vec<u64>,
    ) {
        self.tensors.push(TensorSpec {
            name,
            role: Role {
                layer,
                component,
                part,
            },
            dtype,
            shape,
        });
    }

    /// How the published checkpoint of this format stores a linear.
    fn quant_for(&self, c: Component) -> Quant {
        match (self.fmt, c) {
            (fmt, Component::RoutedExpert { .. }) => fmt.expert_quant(),
            (
                CheckpointFormat::OfficialFp8,
                Component::Dsa(DsaPart::QA | DsaPart::QB | DsaPart::KvA | DsaPart::O)
                | Component::SharedExpert(_)
                | Component::DenseMlp(_),
            ) => Quant::Fp8Block,
            _ => Quant::Bf16,
        }
    }

    fn linear(
        &mut self,
        module: String,
        layer: Option<u64>,
        component: Component,
        out: u64,
        inp: u64,
    ) {
        let quant = self.quant_for(component);
        let t = |part: Part| format!("{module}.{}", part.suffix());
        let spec: Vec<(Part, DType, Vec<u64>)> = match quant {
            Quant::Bf16 => vec![(Part::Weight, DType::BF16, vec![out, inp])],
            Quant::Fp8Block => vec![
                (Part::Weight, DType::F8E4M3, vec![out, inp]),
                (
                    Part::Fp8ScaleInv,
                    DType::F32,
                    vec![ceil_div(out, FP8_BLOCK), ceil_div(inp, FP8_BLOCK)],
                ),
            ],
            Quant::Exl3 { bits } => vec![
                (
                    Part::Exl3Trellis,
                    DType::I16,
                    vec![inp / EXL3_TILE, out / EXL3_TILE, EXL3_TILE * bits],
                ),
                (Part::Exl3Suh, DType::F16, vec![inp]),
                (Part::Exl3Svh, DType::F16, vec![out]),
                (Part::Exl3Mcg, DType::I32, vec![1]),
            ],
            Quant::Nvfp4 => vec![
                (Part::Nvfp4Weight, DType::U8, vec![out, inp / 2]),
                (
                    Part::Nvfp4Scale,
                    DType::F8E4M3,
                    vec![out, inp / NVFP4_GROUP],
                ),
                (Part::Nvfp4Scale2, DType::F32, vec![]),
                (Part::Nvfp4InputScale, DType::F32, vec![]),
            ],
        };
        for (part, dtype, shape) in spec {
            self.param(t(part), layer, component, part, dtype, shape);
        }
        self.linears.push(LinearSpec {
            module,
            layer,
            component,
            out_features: out,
            in_features: inp,
            quant,
        });
    }

    fn text(&mut self) {
        let t = &self.cfg.text;
        let (h, v) = (t.hidden_size, t.vocab_size);
        self.param(
            EMBED.into(),
            None,
            Component::Embedding,
            Part::Weight,
            DType::BF16,
            vec![v, h],
        );
        self.param(
            FINAL_NORM.into(),
            None,
            Component::FinalNorm,
            Part::Weight,
            DType::BF16,
            vec![h],
        );
        self.linear("lm_head".into(), None, Component::LmHead, v, h);
        for l in 0..t.num_hidden_layers {
            self.layer(l, false);
        }
        for l in t.mtp_layers() {
            self.layer(l, true);
        }
    }

    fn layer(&mut self, l: u64, mtp: bool) {
        let t = &self.cfg.text;
        let h = t.hidden_size;
        let p = format!("{LAYERS}{l}.");
        let lay = Some(l);
        let norm = |b: &mut Self, name: &str, c: Component, n: u64| {
            b.param(
                format!("{p}{name}"),
                lay,
                c,
                Part::Weight,
                DType::BF16,
                vec![n],
            );
        };
        norm(self, "input_layernorm.weight", Component::InputNorm, h);
        norm(
            self,
            "post_attention_layernorm.weight",
            Component::PostAttnNorm,
            h,
        );
        if mtp {
            self.linear(
                format!("{p}eh_proj"),
                lay,
                Component::Mtp(MtpPart::EhProj),
                h,
                2 * h,
            );
            norm(self, "enorm.weight", Component::Mtp(MtpPart::ENorm), h);
            norm(self, "hnorm.weight", Component::Mtp(MtpPart::HNorm), h);
            norm(
                self,
                "shared_head.norm.weight",
                Component::Mtp(MtpPart::SharedHeadNorm),
                h,
            );
        } else {
            let (mix, m) = (t.hc_mix(), t.mhc.hc_mult);
            for (site, s) in [(MhcSite::Attn, "attn"), (MhcSite::Ffn, "ffn")] {
                self.param(
                    format!("{p}hc_{s}_fn"),
                    lay,
                    Component::Mhc(site, MhcParam::Fn),
                    Part::Param,
                    DType::BF16,
                    vec![mix, m * h],
                );
                self.param(
                    format!("{p}hc_{s}_base"),
                    lay,
                    Component::Mhc(site, MhcParam::Base),
                    Part::Param,
                    DType::F32,
                    vec![mix],
                );
                self.param(
                    format!("{p}hc_{s}_scale"),
                    lay,
                    Component::Mhc(site, MhcParam::Scale),
                    Part::Param,
                    DType::F32,
                    vec![3],
                );
            }
        }
        match t.attn_kind(l) {
            AttnKind::Kda => self.kda(l),
            AttnKind::Dsa => self.dsa(l),
        }
        if t.is_moe(l) {
            self.moe(l);
        } else {
            let w = t.moe.dense_intermediate_size;
            for proj in Proj::ALL {
                let (o, i) = if proj == Proj::Down { (h, w) } else { (w, h) };
                self.linear(
                    format!("{p}mlp.{}", proj.stem()),
                    lay,
                    Component::DenseMlp(proj),
                    o,
                    i,
                );
            }
        }
    }

    fn kda(&mut self, l: u64) {
        let t = &self.cfg.text;
        let (h, d, hd, heads) = (t.hidden_size, t.kda_dim(), t.kda.head_dim, t.kda.num_heads);
        let p = format!("{LAYERS}{l}.self_attn.");
        let lay = Some(l);
        for (part, s) in [(KdaPart::Q, "q"), (KdaPart::K, "k"), (KdaPart::V, "v")] {
            self.linear(format!("{p}{s}_proj"), lay, Component::Kda(part), d, h);
        }
        for (part, s) in [
            (KdaPart::QConv, "q"),
            (KdaPart::KConv, "k"),
            (KdaPart::VConv, "v"),
        ] {
            let shape = vec![d, 1, t.kda.short_conv_kernel_size];
            self.param(
                format!("{p}{s}_conv1d.weight"),
                lay,
                Component::Kda(part),
                Part::Weight,
                DType::BF16,
                shape,
            );
        }
        self.linear(
            format!("{p}f_a_proj"),
            lay,
            Component::Kda(KdaPart::FA),
            hd,
            h,
        );
        self.linear(
            format!("{p}f_b_proj"),
            lay,
            Component::Kda(KdaPart::FB),
            d,
            hd,
        );
        self.linear(
            format!("{p}g_a_proj"),
            lay,
            Component::Kda(KdaPart::GA),
            hd,
            h,
        );
        self.linear(
            format!("{p}g_b_proj"),
            lay,
            Component::Kda(KdaPart::GB),
            d,
            hd,
        );
        self.linear(
            format!("{p}b_proj"),
            lay,
            Component::Kda(KdaPart::B),
            heads,
            h,
        );
        self.linear(format!("{p}o_proj"), lay, Component::Kda(KdaPart::O), h, d);
        self.param(
            format!("{p}A_log"),
            lay,
            Component::Kda(KdaPart::ALog),
            Part::Param,
            DType::F32,
            vec![heads],
        );
        self.param(
            format!("{p}dt_bias"),
            lay,
            Component::Kda(KdaPart::DtBias),
            Part::Param,
            DType::F32,
            vec![d],
        );
        self.param(
            format!("{p}o_norm.weight"),
            lay,
            Component::Kda(KdaPart::ONorm),
            Part::Weight,
            DType::BF16,
            vec![hd],
        );
    }

    fn dsa(&mut self, l: u64) {
        let t = &self.cfg.text;
        let (h, m, ix) = (t.hidden_size, &t.mla, &t.indexer);
        let p = format!("{LAYERS}{l}.self_attn.");
        let lay = Some(l);
        let heads = m.num_attention_heads;
        self.linear(
            format!("{p}q_a_proj"),
            lay,
            Component::Dsa(DsaPart::QA),
            m.q_lora_rank,
            h,
        );
        self.param(
            format!("{p}q_a_layernorm.weight"),
            lay,
            Component::Dsa(DsaPart::QANorm),
            Part::Weight,
            DType::BF16,
            vec![m.q_lora_rank],
        );
        self.linear(
            format!("{p}q_b_proj"),
            lay,
            Component::Dsa(DsaPart::QB),
            heads * m.qk_head_dim,
            m.q_lora_rank,
        );
        self.linear(
            format!("{p}kv_a_proj_with_mqa"),
            lay,
            Component::Dsa(DsaPart::KvA),
            m.kv_lora_rank + m.qk_rope_head_dim,
            h,
        );
        self.param(
            format!("{p}kv_a_layernorm.weight"),
            lay,
            Component::Dsa(DsaPart::KvANorm),
            Part::Weight,
            DType::BF16,
            vec![m.kv_lora_rank],
        );
        self.linear(
            format!("{p}kv_b_proj"),
            lay,
            Component::Dsa(DsaPart::KvB),
            heads * (m.qk_nope_head_dim + m.v_head_dim),
            m.kv_lora_rank,
        );
        self.linear(
            format!("{p}o_proj"),
            lay,
            Component::Dsa(DsaPart::O),
            h,
            heads * m.v_head_dim,
        );
        let q = format!("{p}indexer.");
        let c = |x| Component::Indexer(x);
        self.linear(
            format!("{q}wq_b"),
            lay,
            c(IndexerPart::WqB),
            ix.n_heads * ix.head_dim,
            m.q_lora_rank,
        );
        self.linear(format!("{q}wk"), lay, c(IndexerPart::Wk), ix.head_dim, h);
        self.param(
            format!("{q}k_norm.weight"),
            lay,
            c(IndexerPart::KNorm),
            Part::Weight,
            DType::BF16,
            vec![ix.head_dim],
        );
        self.param(
            format!("{q}k_norm.bias"),
            lay,
            c(IndexerPart::KNorm),
            Part::Bias,
            DType::BF16,
            vec![ix.head_dim],
        );
        self.linear(
            format!("{q}weights_proj"),
            lay,
            c(IndexerPart::WeightsProj),
            ix.n_heads,
            h,
        );
        self.param(
            format!("{q}index_kpool_compress_gate"),
            lay,
            c(IndexerPart::KpoolGate),
            Part::Param,
            DType::BF16,
            vec![ix.head_dim, h],
        );
        self.param(
            format!("{q}index_kpool_compress_ape"),
            lay,
            c(IndexerPart::KpoolApe),
            Part::Param,
            DType::BF16,
            vec![ix.kpool, ix.head_dim],
        );
    }

    fn moe(&mut self, l: u64) {
        let t = &self.cfg.text;
        let (h, e) = (t.hidden_size, &t.moe);
        let p = format!("{LAYERS}{l}.mlp.");
        let lay = Some(l);
        self.linear(
            format!("{p}gate"),
            lay,
            Component::Router,
            e.n_routed_experts,
            h,
        );
        self.param(
            format!("{p}gate.e_score_correction_bias"),
            lay,
            Component::Router,
            Part::Param,
            DType::F32,
            vec![e.n_routed_experts],
        );
        let ws = e.moe_intermediate_size * e.n_shared_experts;
        for proj in Proj::ALL {
            let (o, i) = if proj == Proj::Down { (h, ws) } else { (ws, h) };
            self.linear(
                format!("{p}shared_experts.{}", proj.stem()),
                lay,
                Component::SharedExpert(proj),
                o,
                i,
            );
        }
        let w = e.moe_intermediate_size;
        for expert in 0..e.n_routed_experts {
            for proj in Proj::ALL {
                let (o, i) = if proj == Proj::Down { (h, w) } else { (w, h) };
                let c = Component::RoutedExpert { expert, proj };
                self.linear(format!("{p}experts.{expert}.{}", proj.stem()), lay, c, o, i);
            }
        }
    }

    fn vision(&mut self) {
        let v = &self.cfg.vision;
        let (h, o, hd) = (
            v.hidden_size,
            v.out_hidden_size,
            v.hidden_size / v.num_heads,
        );
        let mut add = |name: String, shape: Vec<u64>| {
            let part = if name.ends_with(".bias") {
                Part::Bias
            } else {
                Part::Weight
            };
            self.tensors.push(TensorSpec {
                name,
                role: Role {
                    layer: None,
                    component: Component::Vision,
                    part,
                },
                dtype: DType::BF16,
                shape,
            });
        };
        for b in 0..v.depth {
            let p = format!("{VISION}blocks.{b}.");
            add(format!("{p}attn.qkv.weight"), vec![3 * h, h]);
            add(format!("{p}attn.proj.weight"), vec![h, h]);
            if v.attention_bias {
                add(format!("{p}attn.qkv.bias"), vec![3 * h]);
                add(format!("{p}attn.proj.bias"), vec![h]);
            }
            add(format!("{p}attn.q_norm.weight"), vec![hd]);
            add(format!("{p}attn.k_norm.weight"), vec![hd]);
            for s in ["gate_proj", "up_proj"] {
                add(format!("{p}mlp.{s}.weight"), vec![v.intermediate_size, h]);
                add(format!("{p}mlp.{s}.bias"), vec![v.intermediate_size]);
            }
            add(
                format!("{p}mlp.down_proj.weight"),
                vec![h, v.intermediate_size],
            );
            add(format!("{p}mlp.down_proj.bias"), vec![h]);
            add(format!("{p}norm1.weight"), vec![h]);
            add(format!("{p}norm2.weight"), vec![h]);
        }
        let m = v.spatial_merge_size;
        add(format!("{VISION}downsample.weight"), vec![o, h, m, m]);
        add(format!("{VISION}downsample.bias"), vec![o]);
        add(format!("{VISION}merger.proj.weight"), vec![o, o]);
        add(
            format!("{VISION}merger.gate_proj.weight"),
            vec![v.projection_intermediate_size, o],
        );
        add(
            format!("{VISION}merger.up_proj.weight"),
            vec![v.projection_intermediate_size, o],
        );
        add(
            format!("{VISION}merger.down_proj.weight"),
            vec![o, v.projection_intermediate_size],
        );
        add(
            format!("{VISION}merger.post_projection_norm.weight"),
            vec![o],
        );
        add(format!("{VISION}merger.post_projection_norm.bias"), vec![o]);
        let pe = vec![
            h,
            v.in_channels,
            v.temporal_patch_size,
            v.patch_size,
            v.patch_size,
        ];
        add(format!("{VISION}patch_embed.proj.weight"), pe);
        add(format!("{VISION}patch_embed.proj.bias"), vec![h]);
        add(format!("{VISION}post_layernorm.weight"), vec![h]);
    }
}

/// Every tensor (sorted by name) and every linear (sorted by module) that a
/// checkpoint of format `fmt` holds, derived from the config.
pub fn spec(cfg: &ModelConfig, fmt: CheckpointFormat) -> (Vec<TensorSpec>, Vec<LinearSpec>) {
    let mut b = Builder {
        cfg,
        fmt,
        tensors: Vec::new(),
        linears: Vec::new(),
    };
    b.text();
    b.vision();
    b.tensors.sort_by(|x, y| x.name.cmp(&y.name));
    b.linears.sort_by(|x, y| x.module.cmp(&y.module));
    (b.tensors, b.linears)
}

/// Map a tensor name to its role, from the name alone (plus the layer types
/// in the config, and the format for NVFP4's packed `weight`). `None` for a
/// name this model does not have. Independent of [`spec`]; the tests check
/// that the two agree on every tensor.
pub fn classify(name: &str, text: &TextConfig, fmt: CheckpointFormat) -> Option<Role> {
    let top = |component, part| {
        Some(Role {
            layer: None,
            component,
            part,
        })
    };
    match name {
        LM_HEAD => return top(Component::LmHead, Part::Weight),
        EMBED => return top(Component::Embedding, Part::Weight),
        FINAL_NORM => return top(Component::FinalNorm, Part::Weight),
        _ => {}
    }
    if let Some(rest) = name.strip_prefix(VISION) {
        let part = if rest.ends_with(".bias") {
            Part::Bias
        } else if rest.ends_with(".weight") {
            Part::Weight
        } else {
            return None;
        };
        return top(Component::Vision, part);
    }
    let rest = name.strip_prefix(LAYERS)?;
    let (l, rest) = rest.split_once('.')?;
    if l.len() > 1 && l.starts_with('0') {
        return None;
    }
    let l: u64 = l.parse().ok()?;
    let mtp = l >= text.num_hidden_layers;
    if mtp && !text.mtp_layers().contains(&l) {
        return None;
    }
    let role = |component, part| {
        Some(Role {
            layer: Some(l),
            component,
            part,
        })
    };
    // Norms and mHC.
    match rest {
        "input_layernorm.weight" => return role(Component::InputNorm, Part::Weight),
        "post_attention_layernorm.weight" => return role(Component::PostAttnNorm, Part::Weight),
        _ => {}
    }
    if mtp {
        let c = match rest {
            "eh_proj.weight" => Some(MtpPart::EhProj),
            "enorm.weight" => Some(MtpPart::ENorm),
            "hnorm.weight" => Some(MtpPart::HNorm),
            "shared_head.norm.weight" => Some(MtpPart::SharedHeadNorm),
            _ => None,
        };
        if let Some(c) = c {
            return role(Component::Mtp(c), Part::Weight);
        }
    } else if let Some(hc) = rest.strip_prefix("hc_") {
        let (site, param) = hc.split_once('_')?;
        let site = match site {
            "attn" => MhcSite::Attn,
            "ffn" => MhcSite::Ffn,
            _ => return None,
        };
        let param = match param {
            "base" => MhcParam::Base,
            "fn" => MhcParam::Fn,
            "scale" => MhcParam::Scale,
            _ => return None,
        };
        return role(Component::Mhc(site, param), Part::Param);
    }
    // MLP.
    if let Some(m) = rest.strip_prefix("mlp.") {
        match m {
            "gate.weight" => return role(Component::Router, Part::Weight),
            "gate.e_score_correction_bias" => return role(Component::Router, Part::Param),
            _ => {}
        }
        if let Some(x) = m.strip_prefix("shared_experts.") {
            let (proj, part) = linear_suffix(x, Quant::Bf16)?;
            return role(Component::SharedExpert(proj), part);
        }
        if let Some(x) = m.strip_prefix("experts.") {
            let (e, x) = x.split_once('.')?;
            if e.len() > 1 && e.starts_with('0') {
                return None;
            }
            let expert: u64 = e.parse().ok()?;
            if expert >= text.moe.n_routed_experts {
                return None;
            }
            let (proj, part) = linear_suffix(x, fmt.expert_quant())?;
            return role(Component::RoutedExpert { expert, proj }, part);
        }
        let (proj, part) = linear_suffix(m, Quant::Bf16)?;
        return if text.is_moe(l) {
            None
        } else {
            role(Component::DenseMlp(proj), part)
        };
    }
    // Attention.
    let a = rest.strip_prefix("self_attn.")?;
    if let Some(x) = a.strip_prefix("indexer.") {
        if text.attn_kind(l) != AttnKind::Dsa {
            return None;
        }
        let c = |p| Component::Indexer(p);
        return match x {
            "wq_b.weight" => role(c(IndexerPart::WqB), Part::Weight),
            "wk.weight" => role(c(IndexerPart::Wk), Part::Weight),
            "k_norm.weight" => role(c(IndexerPart::KNorm), Part::Weight),
            "k_norm.bias" => role(c(IndexerPart::KNorm), Part::Bias),
            "weights_proj.weight" => role(c(IndexerPart::WeightsProj), Part::Weight),
            "index_kpool_compress_gate" => role(c(IndexerPart::KpoolGate), Part::Param),
            "index_kpool_compress_ape" => role(c(IndexerPart::KpoolApe), Part::Param),
            _ => None,
        };
    }
    match text.attn_kind(l) {
        AttnKind::Kda => {
            let c = |p| Component::Kda(p);
            match a {
                "A_log" => return role(c(KdaPart::ALog), Part::Param),
                "dt_bias" => return role(c(KdaPart::DtBias), Part::Param),
                "o_norm.weight" => return role(c(KdaPart::ONorm), Part::Weight),
                "q_conv1d.weight" => return role(c(KdaPart::QConv), Part::Weight),
                "k_conv1d.weight" => return role(c(KdaPart::KConv), Part::Weight),
                "v_conv1d.weight" => return role(c(KdaPart::VConv), Part::Weight),
                _ => {}
            }
            let (module, suffix) = a.split_once('.')?;
            let part = match (module, suffix) {
                ("q_proj", "weight") => KdaPart::Q,
                ("k_proj", "weight") => KdaPart::K,
                ("v_proj", "weight") => KdaPart::V,
                ("f_a_proj", "weight") => KdaPart::FA,
                ("f_b_proj", "weight") => KdaPart::FB,
                ("g_a_proj", "weight") => KdaPart::GA,
                ("g_b_proj", "weight") => KdaPart::GB,
                ("b_proj", "weight") => KdaPart::B,
                ("o_proj", "weight") => KdaPart::O,
                _ => return None,
            };
            role(c(part), Part::Weight)
        }
        AttnKind::Dsa => {
            let (module, suffix) = a.split_once('.')?;
            let (part, norm) = match module {
                "q_a_proj" => (DsaPart::QA, false),
                "q_a_layernorm" => (DsaPart::QANorm, true),
                "q_b_proj" => (DsaPart::QB, false),
                "kv_a_proj_with_mqa" => (DsaPart::KvA, false),
                "kv_a_layernorm" => (DsaPart::KvANorm, true),
                "kv_b_proj" => (DsaPart::KvB, false),
                "o_proj" => (DsaPart::O, false),
                _ => return None,
            };
            let p = match suffix {
                "weight" => Part::Weight,
                "weight_scale_inv" if !norm => Part::Fp8ScaleInv,
                _ => return None,
            };
            role(Component::Dsa(part), p)
        }
    }
}

/// `<gate|up|down>_proj.<suffix>` -> (projection, part) for a linear stored as
/// `q` (or as FP8 blocks, which every non-expert MLP of the official
/// checkpoint uses).
fn linear_suffix(x: &str, q: Quant) -> Option<(Proj, Part)> {
    let (stem, suffix) = x.split_once('.')?;
    let proj = Proj::from_stem(stem)?;
    let part = match (q, suffix) {
        (Quant::Nvfp4, "weight") => Part::Nvfp4Weight,
        (Quant::Nvfp4, "weight_scale") => Part::Nvfp4Scale,
        (Quant::Nvfp4, "weight_scale_2") => Part::Nvfp4Scale2,
        (Quant::Nvfp4, "input_scale") => Part::Nvfp4InputScale,
        (Quant::Exl3 { .. }, "trellis") => Part::Exl3Trellis,
        (Quant::Exl3 { .. }, "suh") => Part::Exl3Suh,
        (Quant::Exl3 { .. }, "svh") => Part::Exl3Svh,
        (Quant::Exl3 { .. }, "mcg") => Part::Exl3Mcg,
        (Quant::Bf16 | Quant::Fp8Block, "weight") => Part::Weight,
        (Quant::Bf16 | Quant::Fp8Block, "weight_scale_inv") => Part::Fp8ScaleInv,
        _ => return None,
    };
    Some((proj, part))
}

/// Where a tensor's bytes are.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TensorLoc {
    /// Shard file name.
    pub file: String,
    /// Byte range relative to the shard's data region.
    pub begin: u64,
    pub end: u64,
    /// Absolute offset of the shard's data region, when read from a file.
    pub data_start: Option<u64>,
}

impl TensorLoc {
    /// Absolute byte range in the shard file, when known.
    pub fn file_range(&self) -> Option<(u64, u64)> {
        self.data_start.map(|s| (s + self.begin, s + self.end))
    }
}

/// One catalogued tensor.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TensorInfo {
    pub name: String,
    pub role: Role,
    pub group: Group,
    pub dtype: DType,
    pub shape: Vec<u64>,
    pub bytes: u64,
    /// `None` for a catalog derived from the config alone.
    pub loc: Option<TensorLoc>,
}

/// One linear layer with its stored parts (indices into [`Catalog::tensors`]).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Linear {
    pub spec: LinearSpec,
    pub parts: Vec<(Part, usize)>,
}

impl Linear {
    pub fn tensor(&self, part: Part) -> Option<usize> {
        self.parts.iter().find(|(p, _)| *p == part).map(|&(_, i)| i)
    }
}

/// A checkpoint's tensors, each with its role.
#[derive(Clone, Debug)]
pub struct Catalog {
    pub format: CheckpointFormat,
    /// Sorted by name.
    pub tensors: Vec<TensorInfo>,
    /// Sorted by module name.
    pub linears: Vec<Linear>,
    /// Whether every tensor of the format is present (not a subset).
    pub complete: bool,
}

/// How strictly [`Catalog::from_shards`] matches a checkpoint.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Coverage {
    /// Every expected tensor must be present.
    Complete,
    /// A subset of the checkpoint (for example the coordinator's tensors
    /// fetched on their own); every present tensor must still be expected.
    Subset,
}

impl Catalog {
    /// The catalog a checkpoint of `fmt` has, derived from the config alone
    /// (no file locations).
    pub fn from_config(cfg: &ModelConfig, fmt: CheckpointFormat) -> Catalog {
        let (tensors, linears) = spec(cfg, fmt);
        let tensors: Vec<TensorInfo> = tensors
            .into_iter()
            .map(|s| TensorInfo {
                group: group_of(&s.role, &cfg.text),
                bytes: numel(&s.shape) * s.dtype.size(),
                name: s.name,
                role: s.role,
                dtype: s.dtype,
                shape: s.shape,
                loc: None,
            })
            .collect();
        let linears =
            link(&tensors, linears).expect("a derived catalog holds every part of its linears");
        Catalog {
            format: fmt,
            tensors,
            linears,
            complete: true,
        }
    }

    /// Catalog the tensors of `shards`, checking them against the config. The
    /// format is detected from the tensor names unless given.
    pub fn from_shards(
        cfg: &ModelConfig,
        shards: &[Shard],
        fmt: Option<CheckpointFormat>,
        coverage: Coverage,
    ) -> Result<Catalog> {
        let fmt = match fmt {
            Some(f) => f,
            None => detect_format(shards)?,
        };
        let mut found: Vec<(&str, &Shard, &TensorEntry)> = shards
            .iter()
            .flat_map(|s| {
                s.header
                    .tensors
                    .iter()
                    .map(move |(n, e)| (n.as_str(), s, e))
            })
            .collect();
        found.sort_by(|a, b| a.0.cmp(b.0));
        if let Some(w) = found.windows(2).find(|w| w[0].0 == w[1].0) {
            return Err(Error::Catalog(format!(
                "{} is in both {} and {}",
                w[0].0, w[0].1.file, w[1].1.file
            )));
        }
        let (expected, linears) = spec(cfg, fmt);
        let (mut missing, mut unexpected, mut wrong) = (Vec::new(), Vec::new(), Vec::new());
        let mut tensors = Vec::with_capacity(found.len());
        let (mut i, mut j) = (0, 0);
        while i < expected.len() || j < found.len() {
            let ord = match (expected.get(i), found.get(j)) {
                (Some(e), Some(f)) => e.name.as_str().cmp(f.0),
                (Some(_), None) => Ordering::Less,
                _ => Ordering::Greater,
            };
            match ord {
                Ordering::Less => {
                    missing.push(expected[i].name.clone());
                    i += 1;
                }
                Ordering::Greater => {
                    unexpected.push(found[j].0.to_string());
                    j += 1;
                }
                Ordering::Equal => {
                    let (e, (name, shard, entry)) = (&expected[i], found[j]);
                    if e.dtype != entry.dtype || e.shape != entry.shape {
                        wrong.push(format!(
                            "{name}: {} {:?}, expected {} {:?}",
                            entry.dtype, entry.shape, e.dtype, e.shape
                        ));
                    }
                    tensors.push(TensorInfo {
                        name: name.to_string(),
                        role: e.role,
                        group: group_of(&e.role, &cfg.text),
                        dtype: entry.dtype,
                        shape: entry.shape.clone(),
                        bytes: entry.byte_len(),
                        loc: Some(TensorLoc {
                            file: shard.file.clone(),
                            begin: entry.begin,
                            end: entry.end,
                            data_start: shard.data_start,
                        }),
                    });
                    i += 1;
                    j += 1;
                }
            }
        }
        let complete = missing.is_empty();
        if coverage == Coverage::Subset {
            missing.clear();
        }
        if !(missing.is_empty() && unexpected.is_empty() && wrong.is_empty()) {
            let list = |what: &str, v: &[String]| {
                if v.is_empty() {
                    String::new()
                } else {
                    format!("\n  {} {what}, e.g. {:?}", v.len(), &v[..v.len().min(4)])
                }
            };
            return Err(Error::Catalog(format!(
                "checkpoint does not match the {} layout of this config:{}{}{}",
                fmt.label(),
                list("missing", &missing),
                list("unexpected", &unexpected),
                list("with the wrong dtype or shape", &wrong)
            )));
        }
        let linears = link(&tensors, linears)?;
        Ok(Catalog {
            format: fmt,
            tensors,
            linears,
            complete,
        })
    }

    /// Index of the tensor called `name`.
    pub fn find(&self, name: &str) -> Option<usize> {
        self.tensors
            .binary_search_by(|t| t.name.as_str().cmp(name))
            .ok()
    }

    pub fn get(&self, name: &str) -> Option<&TensorInfo> {
        self.find(name).map(|i| &self.tensors[i])
    }

    /// The linear whose tensors start with `module`.
    pub fn linear(&self, module: &str) -> Option<&Linear> {
        let i = self
            .linears
            .binary_search_by(|l| l.spec.module.as_str().cmp(module))
            .ok()?;
        Some(&self.linears[i])
    }

    /// A routed expert's projection.
    pub fn expert(&self, layer: u64, expert: u64, proj: Proj) -> Option<&Linear> {
        self.linear(&format!(
            "{LAYERS}{layer}.mlp.experts.{expert}.{}",
            proj.stem()
        ))
    }

    /// Bytes per group.
    pub fn bytes_by_group(&self) -> BTreeMap<Group, u64> {
        let mut out = BTreeMap::new();
        for t in &self.tensors {
            *out.entry(t.group).or_insert(0) += t.bytes;
        }
        out
    }

    /// Bytes of one group.
    pub fn group_bytes(&self, g: Group) -> u64 {
        self.tensors
            .iter()
            .filter(|t| t.group == g)
            .map(|t| t.bytes)
            .sum()
    }

    /// Bytes of one group, split by dtype.
    pub fn group_dtypes(&self, g: Group) -> BTreeMap<DType, u64> {
        let mut out = BTreeMap::new();
        for t in self.tensors.iter().filter(|t| t.group == g) {
            *out.entry(t.dtype).or_insert(0) += t.bytes;
        }
        out
    }

    pub fn total_bytes(&self) -> u64 {
        self.tensors.iter().map(|t| t.bytes).sum()
    }
}

/// Resolve each linear's part names to tensor indices. A linear with none of
/// its parts present is left out (a subset); one with only some is an error
/// (for example an FP8 weight without its scales).
fn link(tensors: &[TensorInfo], specs: Vec<LinearSpec>) -> Result<Vec<Linear>> {
    let find = |name: &str| tensors.binary_search_by(|t| t.name.as_str().cmp(name)).ok();
    let mut out = Vec::with_capacity(specs.len());
    for s in specs {
        let parts: Vec<(Part, Option<usize>)> = s
            .parts()
            .iter()
            .map(|&p| (p, find(&format!("{}.{}", s.module, p.suffix()))))
            .collect();
        let present = parts.iter().filter(|(_, i)| i.is_some()).count();
        if present == 0 {
            continue;
        }
        if present != parts.len() {
            let absent: Vec<&str> = parts
                .iter()
                .filter(|(_, i)| i.is_none())
                .map(|(p, _)| p.suffix())
                .collect();
            return Err(Error::Catalog(format!(
                "{} is missing its {absent:?}",
                s.module
            )));
        }
        out.push(Linear {
            parts: parts.into_iter().map(|(p, i)| (p, i.unwrap())).collect(),
            spec: s,
        });
    }
    Ok(out)
}

/// Tell the checkpoint format from the routed-expert tensor names.
pub fn detect_format(shards: &[Shard]) -> Result<CheckpointFormat> {
    let mut found = Vec::new();
    for s in shards {
        for (name, e) in &s.header.tensors {
            if !name.contains(".mlp.experts.") {
                continue;
            }
            let f = if name.ends_with(".trellis") {
                match e.shape.last() {
                    Some(&w) if w % EXL3_TILE == 0 && w > 0 => CheckpointFormat::Exl3 {
                        bits: w / EXL3_TILE,
                    },
                    _ => {
                        return Err(Error::Catalog(format!(
                            "{name}: trellis shape {:?}",
                            e.shape
                        )))
                    }
                }
            } else if name.ends_with(".weight_scale_2") {
                CheckpointFormat::Nvfp4
            } else if name.ends_with(".weight_scale_inv") {
                CheckpointFormat::OfficialFp8
            } else {
                continue;
            };
            if !found.contains(&f) {
                found.push(f);
            }
        }
    }
    match found.as_slice() {
        [f] => Ok(*f),
        [] => Err(Error::Catalog(
            "no routed-expert tensors: give the checkpoint format explicitly".into(),
        )),
        many => Err(Error::Catalog(format!(
            "routed experts in several formats: {many:?}"
        ))),
    }
}
