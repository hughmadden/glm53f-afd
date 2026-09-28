//! GLM-5.3-Flash geometry as the expert rank sees it.
//!
//! Every value is from the model's `config.json` (`zai-org/GLM-5.3-Flash`,
//! architecture `glm5_next`) or from the TP4 split of docs/DESIGN.md. The rank
//! never reads the config at run time: a rank built for another geometry is a
//! different binary.

/// Model width: the hidden row on the wire, gate/up input, down output.
pub const HIDDEN: usize = 4096;
/// A routed expert's intermediate width (`moe_intermediate_size`).
pub const INTERMEDIATE: usize = 2048;
/// Expert ranks (tensor parallel over the intermediate dimension).
pub const WORLD: usize = 4;
/// Intermediate channels per rank: rank `r` owns `[512 r, 512 r + 512)`.
pub const RANK_WIDTH: usize = INTERMEDIATE / WORLD;
/// Routed experts per MoE layer (`n_routed_experts`).
pub const EXPERTS: usize = 288;
/// Routed experts per row (`num_experts_per_tok`).
pub const TOPK: usize = 8;
/// First MoE layer (`first_k_dense_replace`): layers 0-2 are dense.
pub const FIRST_MOE_LAYER: u32 = 3;
/// Last decoder MoE layer (`num_hidden_layers` = 45, so 3..=44: 42 layers).
pub const LAST_MOE_LAYER: u32 = 44;
/// The MTP layer, whose routed experts can also live on the ranks (optional).
pub const MTP_LAYER: u32 = 45;
/// Decoder MoE layers served by every rank.
pub const MOE_LAYERS: usize = (LAST_MOE_LAYER - FIRST_MOE_LAYER + 1) as usize;
/// `swiglu_limit`: gate clamped above at +10, up clamped to [-10, 10].
pub const SWIGLU_LIMIT: f32 = 10.0;
/// `routed_scaling_factor`. The rank never applies it: the reference folds it
/// into the router's top-k weights (`Glm5NextTextTopkRouter.forward`), so it
/// arrives inside the wire's FP32 gate weights or is applied by the
/// coordinator after the rank sum (the two are the same linear map).
pub const ROUTED_SCALE: f32 = 2.5;
/// Largest request the rank serves in one call (the RDMA receive slot size).
pub const MAX_ROWS: usize = 4096;

/// Whether `layer` is a MoE layer this rank can serve (`with_mtp` admits 45).
pub fn is_moe_layer(layer: u32, with_mtp: bool) -> bool {
    (FIRST_MOE_LAYER..=LAST_MOE_LAYER).contains(&layer) || (with_mtp && layer == MTP_LAYER)
}

const _: () = assert!(MOE_LAYERS == 42);
const _: () = assert!(RANK_WIDTH == 512);
const _: () = assert!(RANK_WIDTH.is_multiple_of(128), "a rank's share must be whole 128-channel Hadamard blocks");
