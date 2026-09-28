//! `glm53f-rank`: the expert rank of glm53f-afd.
//!
//! One DGX Spark serves a quarter of every routed expert of GLM-5.3-Flash:
//! intermediate channels `[512 r, 512 r + 512)` of each of the 288 experts of
//! the 42 MoE layers, stored as EXL3 K4 (README.md). The coordinator sends each
//! MoE layer's routed rows as `DS41RTE3` v3 request frames (FP8 E4M3 rows with
//! UE8M0 K32 scales, top-8 expert ids and FP32 gate weights per row); the rank
//! returns, per row, the BF16 sum over its 8 routed experts of
//! `gate_weight * expert_partial` (8,192 bytes a row). For prefill-sized
//! requests the coordinator can ask for the reduce-scatter instead (`DS41RTE3`
//! v4): the ranks add their partials among themselves over the peer mesh and
//! each returns only its quarter of the rows, summed.
//!
//! # Modules
//!
//! - Numerics and format: [`consts`], [`half`], [`fp8`], [`exl3`] (the EXL3
//!   decoder, pinned bit for bit to TensorFold's reference), [`layout`] (the
//!   rank's weight image and the TP4 slicing, through `glm53f-model`),
//!   [`reference`] (the dequantized FP32 expert FFN and a kernel-order
//!   emulation), [`kernel`] (the kernel interface and its CPU backend),
//!   [`testkit`] (synthetic layers, rows and routes).
//! - The prefill reduce-scatter: [`reduce_scatter`] (partition, exchange
//!   frames, the sum in rank order) and [`mesh`] (the links between the ranks).
//! - Serving shell, ported from mimo26f-afd v1.2.0 (`PROVENANCE.md`):
//!   [`transport`], [`server`], [`serve`], [`route`], [`boot`], [`resident`],
//!   [`manifest`], [`sha256`], [`wire`], [`timeline`], and, with the `cuda`
//!   feature, [`cuda`], [`device`] and the kernel backend `exl3_cuda`.
//! - Memory: [`pagecache`] gives a layer image's cached pages back once it is on the device, and
//!   reports `MemAvailable` beside `MemFree` for the boot log.
//!
//! The seam is the attention/FFN boundary: the rank never touches embeddings,
//! attention, the KV cache, the router or sampling.

pub mod boot;
pub mod consts;
#[cfg(feature = "cuda")]
pub mod cuda;
#[cfg(feature = "cuda")]
pub mod device;
pub mod exl3;
#[cfg(feature = "cuda")]
pub mod exl3_cuda;
pub mod fp8;
pub mod half;
pub mod kernel;
pub mod layout;
pub mod manifest;
pub mod mesh;
pub mod pagecache;
pub mod reduce_scatter;
pub mod reference;
pub mod resident;
pub mod route;
pub mod serve;
pub mod server;
pub mod sha256;
pub mod testkit;
pub mod timeline;
pub mod transport;
pub mod wire;
