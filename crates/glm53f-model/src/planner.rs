//! The memory planner: docs/SIZING.md sections 2-6 as code.
//!
//! Every figure is computed from components: tensor bytes from a [`Catalog`]
//! (read from checkpoint headers, or derived from `config.json`), cache and
//! state geometry from the config, draft KV from the drafter config. The one
//! estimate is the runtime and workspace reserve, which is a parameter.
//!
//! - [`KvGeometry`]: bytes per token of context (section 2).
//! - [`SlotState`]: fixed bytes per active request (section 3).
//! - [`resident_weights`]: coordinator weights per [`WeightPolicy`]; the four
//!   [`Layout`]s of section 5 are named policies (section 4).
//! - [`plan_gpu`]: the coordinator GPU budget and KV pool (section 5).
//! - [`plan_spark`]: expert bytes per rank (section 6).

use crate::catalog::{
    Catalog, CheckpointFormat, Component, Group, KdaPart, Part, TensorInfo, FP8_BLOCK,
};
use crate::config::{DraftConfig, TextConfig};
use crate::dtype::DType;
use crate::error::{Error, Result};
use crate::slicing::{rank_plan, Tp};

pub const MIB: u64 = 1 << 20;
pub const GIB: u64 = 1 << 30;
pub const GB: u64 = 1_000_000_000;
pub const TOKENS_256K: u64 = 262_144;
pub const TOKENS_1M: u64 = 1_048_576;
/// Tokens per KV page: 16 indexer pools of 4 (docs/DESIGN.md section 8).
pub const PAGE_TOKENS: u64 = 64;
/// Latent values per F32 scale in an FP8 MLA record.
pub const MLA_FP8_SCALE_GROUP: u64 = 128;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum KvPrecision {
    Fp8,
    Bf16,
}

/// What each token of context costs in the DSA layers.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct KvGeometry {
    pub precision: KvPrecision,
    pub dsa_layers: u64,
    /// Tokens per pooled indexer key.
    pub kpool: u64,
    /// One token's MLA latent in one layer: 512 FP8 values and 4 F32 scales
    /// (528 B), or 512 BF16 values (1,024 B).
    pub mla_record: u64,
    /// One pooled indexer key in one layer: 128 FP8 values and an F32 scale
    /// (132 B), or 128 BF16 values (256 B).
    pub index_record: u64,
}

impl KvGeometry {
    pub fn new(text: &TextConfig, precision: KvPrecision) -> KvGeometry {
        let (kv, ih) = (text.mla.kv_lora_rank, text.indexer.head_dim);
        let (mla_record, index_record) = match precision {
            KvPrecision::Fp8 => (kv + kv.div_ceil(MLA_FP8_SCALE_GROUP) * 4, ih + 4),
            KvPrecision::Bf16 => (kv * 2, ih * 2),
        };
        KvGeometry {
            precision,
            dsa_layers: text.dsa_layer_ids().len() as u64,
            kpool: text.indexer.kpool,
            mla_record,
            index_record,
        }
    }

    /// MLA latent bytes per token, over all DSA layers.
    pub fn mla_per_token(&self) -> u64 {
        self.dsa_layers * self.mla_record
    }

    /// Indexer bytes per pool of `kpool` tokens, over all DSA layers.
    pub fn index_per_pool(&self) -> u64 {
        self.dsa_layers * self.index_record
    }

    /// Bytes of `tokens` tokens of context, a whole number of pools.
    pub fn bytes_for(&self, tokens: u64) -> u64 {
        assert_eq!(tokens % self.kpool, 0, "a whole number of pools");
        tokens * self.mla_per_token() + tokens / self.kpool * self.index_per_pool()
    }

    /// Bytes per token (the indexer share averaged over a pool).
    pub fn per_token(&self) -> f64 {
        self.bytes_for(self.kpool) as f64 / self.kpool as f64
    }

    /// Bytes of one KV page of [`PAGE_TOKENS`] tokens.
    pub fn page_bytes(&self) -> u64 {
        self.bytes_for(PAGE_TOKENS)
    }

    /// Bytes a request of `tokens` tokens occupies, in whole pages.
    pub fn request_bytes(&self, tokens: u64) -> u64 {
        tokens.div_ceil(PAGE_TOKENS) * self.page_bytes()
    }
}

/// Fixed state per active request (slot).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SlotState {
    /// KDA recurrent state, FP32: layers x heads x head_dim x head_dim.
    pub kda_state: u64,
    /// Short-convolution state, BF16: layers x (kernel - 1) x q, k, v channels.
    pub conv_state: u64,
    /// Drafter KV over its window plus one block (0 without a drafter).
    pub draft_kv: u64,
}

impl SlotState {
    pub fn new(text: &TextConfig, drafter: Option<&DraftConfig>) -> SlotState {
        let layers = text.kda_layer_ids().len() as u64;
        let (heads, d) = (text.kda.num_heads, text.kda.head_dim);
        SlotState {
            kda_state: layers * heads * d * d * 4,
            conv_state: layers
                * text.kda.short_conv_kernel_size.saturating_sub(1)
                * 3
                * text.kda_dim()
                * 2,
            draft_kv: drafter.map_or(0, DraftConfig::kv_bytes_per_slot),
        }
    }

    pub fn total(&self) -> u64 {
        self.kda_state + self.conv_state + self.draft_kv
    }

    /// One KDA snapshot (recurrent plus convolution state), as saved at the end
    /// of a prompt or a turn.
    pub fn kda_snapshot(&self) -> u64 {
        self.kda_state + self.conv_state
    }
}

/// Which non-expert weights the coordinator GPU holds (docs/SIZING.md section 5).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Layout {
    A,
    B,
    C,
    D,
}

impl Layout {
    pub const ALL: [Layout; 4] = [Layout::A, Layout::B, Layout::C, Layout::D];

    pub fn letter(self) -> char {
        match self {
            Layout::A => 'A',
            Layout::B => 'B',
            Layout::C => 'C',
            Layout::D => 'D',
        }
    }

    pub fn parse(s: &str) -> Option<Layout> {
        Layout::ALL
            .into_iter()
            .find(|l| s.eq_ignore_ascii_case(&l.letter().to_string()))
    }

    pub fn title(self) -> &'static str {
        match self {
            Layout::A => "official precision, embedding on GPU",
            Layout::B => "official precision, embedding in host RAM",
            Layout::C => "B plus FP8 KDA projections",
            Layout::D => {
                "all-BF16 non-expert (quantized checkpoints as shipped), embedding in host RAM"
            }
        }
    }

    /// The weight policy this layout stands for.
    pub fn policy(self) -> WeightPolicy {
        WeightPolicy {
            embedding_on_gpu: self == Layout::A,
            kda_fp8: self == Layout::C,
            shipped_bf16: self == Layout::D,
        }
    }
}

/// Which non-expert weights the coordinator GPU holds, and at what precision.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct WeightPolicy {
    /// The embedding table on the GPU (otherwise in host RAM, gathered by row).
    pub embedding_on_gpu: bool,
    /// KDA projections quantized to FP8 with 128 x 128 block scales (our own
    /// change; the official checkpoint ships them in BF16).
    pub kda_fp8: bool,
    /// Non-expert weights as the EXL3 and NVFP4 checkpoints ship them (all
    /// BF16), rather than from the official checkpoint.
    pub shipped_bf16: bool,
}

/// Whether a tensor is a BF16 KDA projection weight (what `kda_fp8` quantizes).
pub fn is_kda_projection(t: &TensorInfo) -> bool {
    use KdaPart::*;
    matches!(
        t.role.component,
        Component::Kda(Q | K | V | FA | FB | GA | GB | B | O)
    ) && t.role.part == Part::Weight
        && t.dtype == DType::BF16
}

/// Bytes of an [out, in] weight in FP8 E4M3 with F32 scales per 128 x 128 block.
pub fn fp8_block_bytes(out: u64, inp: u64) -> u64 {
    out * inp + out.div_ceil(FP8_BLOCK) * inp.div_ceil(FP8_BLOCK) * 4
}

/// The coordinator's resident weights under a policy.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ResidentWeights {
    pub policy: WeightPolicy,
    /// Bytes per group as held on the GPU (0 for a group kept elsewhere).
    pub by_group: Vec<(Group, u64)>,
    pub total: u64,
}

impl ResidentWeights {
    pub fn group(&self, g: Group) -> u64 {
        self.by_group
            .iter()
            .find(|(x, _)| *x == g)
            .map_or(0, |&(_, b)| b)
    }
}

/// The coordinator's resident weights under `policy`. `cat` supplies the
/// non-expert tensors: the official catalog, or an EXL3 or NVFP4 catalog when
/// `policy.shipped_bf16` (layout D).
pub fn resident_weights(cat: &Catalog, policy: WeightPolicy) -> Result<ResidentWeights> {
    let official = cat.format == CheckpointFormat::OfficialFp8;
    if official == policy.shipped_bf16 {
        return Err(Error::Plan(format!(
            "the policy needs {} non-expert weights, not a {} catalog",
            if policy.shipped_bf16 {
                "BF16-shipped (EXL3 or NVFP4)"
            } else {
                "official"
            },
            cat.format.label()
        )));
    }
    let mut by_group: Vec<(Group, u64)> = Group::ALL.iter().map(|&g| (g, 0)).collect();
    for t in &cat.tensors {
        let on_gpu = t.group.always_on_coordinator()
            || (t.group == Group::Embedding && policy.embedding_on_gpu);
        if !on_gpu {
            continue;
        }
        let bytes = if policy.kda_fp8 && is_kda_projection(t) {
            fp8_block_bytes(t.shape[0], t.shape[1])
        } else {
            t.bytes
        };
        by_group.iter_mut().find(|(g, _)| *g == t.group).unwrap().1 += bytes;
    }
    let total = by_group.iter().map(|(_, b)| b).sum();
    Ok(ResidentWeights {
        policy,
        by_group,
        total,
    })
}

/// Everything on the coordinator GPU except the KV pool.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GpuBudget {
    /// Device memory the plan may use (for an RTX 5090, 32,607 MiB).
    pub device: u64,
    pub weights: u64,
    /// Drafter weights (0 without one).
    pub drafter: u64,
    pub slots: u64,
    pub slot_state: SlotState,
    /// Runtime and workspace reserve: CUDA context, graphs, activations.
    pub runtime: u64,
}

impl GpuBudget {
    pub fn fixed(&self) -> u64 {
        self.weights + self.drafter + self.slots * self.slot_state.total() + self.runtime
    }
}

/// What a KV pool holds.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct KvCapacity {
    pub geometry: KvGeometry,
    pub page_bytes: u64,
    pub pages: u64,
    pub tokens: u64,
    /// Requests of 262,144 tokens that fit at once (at most one per slot).
    pub requests_256k: u64,
    /// Requests of 1,048,576 tokens that fit at once (at most one per slot).
    pub requests_1m: u64,
    /// The longest single request: the pool or the model's maximum context.
    pub longest_request: u64,
}

impl KvCapacity {
    pub fn new(text: &TextConfig, precision: KvPrecision, pool: u64, slots: u64) -> KvCapacity {
        let geometry = KvGeometry::new(text, precision);
        let page_bytes = geometry.page_bytes();
        let pages = pool / page_bytes;
        let tokens = pages * PAGE_TOKENS;
        KvCapacity {
            geometry,
            page_bytes,
            pages,
            tokens,
            requests_256k: (tokens / TOKENS_256K).min(slots),
            requests_1m: (tokens / TOKENS_1M).min(slots),
            longest_request: tokens.min(text.max_position_embeddings),
        }
    }
}

/// The coordinator GPU plan.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GpuPlan {
    pub budget: GpuBudget,
    /// What remains for the KV pool.
    pub kv_pool: u64,
    pub fp8: KvCapacity,
    pub bf16: KvCapacity,
}

/// Plan the coordinator GPU. Fails if the fixed parts do not fit.
pub fn plan_gpu(text: &TextConfig, budget: GpuBudget) -> Result<GpuPlan> {
    let fixed = budget.fixed();
    if fixed > budget.device {
        return Err(Error::Plan(format!(
            "weights, drafter, {} slots and runtime need {:.2} GiB; the device has {:.2} GiB",
            budget.slots,
            gib(fixed),
            gib(budget.device)
        )));
    }
    let kv_pool = budget.device - fixed;
    Ok(GpuPlan {
        budget,
        kv_pool,
        fp8: KvCapacity::new(text, KvPrecision::Fp8, kv_pool, budget.slots),
        bf16: KvCapacity::new(text, KvPrecision::Bf16, kv_pool, budget.slots),
    })
}

/// One expert rank's load (docs/SIZING.md section 6).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SparkRank {
    pub format: CheckpointFormat,
    pub world: u64,
    /// Routed-expert bytes of the decoder layers held by each rank.
    pub bytes: u64,
    /// Routed-expert bytes of the MTP layer held by each rank.
    pub mtp_bytes: u64,
    /// Of `bytes + mtp_bytes`, tensors held whole by every rank.
    pub replicated_bytes: u64,
    /// One expert's share on one rank.
    pub expert_slice: u64,
    /// Bytes each rank reads for one token (top-k experts in every MoE layer).
    pub read_per_token: u64,
}

/// Plan the expert ranks: every rank's share of `cat`'s routed experts.
pub fn plan_spark(cat: &Catalog, text: &TextConfig, world: u64) -> Result<SparkRank> {
    let mut first: Option<(u64, u64, u64)> = None;
    for rank in 0..world {
        let p = rank_plan(cat, Tp { rank, world })?;
        let got = (p.bytes, p.mtp_bytes, p.replicated_bytes);
        match first {
            None => first = Some(got),
            Some(f) if f != got => {
                return Err(Error::Plan(format!(
                    "rank {rank} holds {got:?} bytes, rank 0 holds {f:?}"
                )))
            }
            _ => {}
        }
    }
    let (bytes, mtp_bytes, replicated_bytes) =
        first.ok_or_else(|| Error::Plan("no ranks".into()))?;
    let experts = text.moe_layers().len() as u64 * text.moe.n_routed_experts;
    if experts == 0 || bytes % experts != 0 {
        return Err(Error::Plan(format!(
            "{bytes} bytes do not divide over {experts} experts"
        )));
    }
    let expert_slice = bytes / experts;
    Ok(SparkRank {
        format: cat.format,
        world,
        bytes,
        mtp_bytes,
        replicated_bytes,
        expert_slice,
        read_per_token: expert_slice
            * text.moe.num_experts_per_tok
            * text.moe_layers().len() as u64,
    })
}

pub fn gb(bytes: u64) -> f64 {
    bytes as f64 / GB as f64
}

pub fn gib(bytes: u64) -> f64 {
    bytes as f64 / GIB as f64
}

pub fn mib(bytes: u64) -> f64 {
    bytes as f64 / MIB as f64
}

/// Millions of tokens.
pub fn mtok(tokens: u64) -> f64 {
    tokens as f64 / 1e6
}
