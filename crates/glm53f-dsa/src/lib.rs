//! # glm53f-dsa
//!
//! The DeepSeek sparse attention (DSA) layers of GLM-5.3-Flash: an f32 CPU
//! reference of the full layer path, the cache formats, and (behind the `cuda`
//! feature) the CUDA kernels for the indexer's top-512 selection and sparse MLA.
//!
//! GLM-5.3-Flash has 11 DSA layers (3, 7, ..., 43). Each is MLA with a 512-dim
//! latent and no RoPE (64 heads of 256), plus its own indexer (32 heads of 128)
//! that pools index keys 4 tokens to 1 with a learned per-channel gate, keeps
//! the best 512 pools for each query and adds the incomplete tail pool, so
//! sparse attention reads at most 2,051 cached latents per query.
//!
//! Module map:
//! * [`config`]: dimensions and constants.
//! * [`indexer`], [`select`]: index projections, k-pool compression, scores,
//!   top-k with a deterministic tie-break, expansion and the tail.
//! * [`mla`]: MLA in the expanded (reference) and absorbed (engine) forms.
//! * [`layer`]: the whole layer over a per-request cache, with intermediates.
//! * [`cache`], [`fp8`]: the FP8 latent record, the pooled index key, the tail,
//!   the page layout.
//! * [`weights`], [`safetensors`], [`golden`]: loading real weights and oracle
//!   fixtures when they exist.
//! * `gpu` (feature `cuda`): device buffers and kernel launchers.

pub mod cache;
pub mod config;
pub mod fp8;
pub mod golden;
pub mod indexer;
pub mod json;
pub mod layer;
pub mod mla;
pub mod num;
pub mod rng;
pub mod safetensors;
pub mod select;
pub mod sha256;
pub mod weights;

#[cfg(feature = "cuda")]
pub mod ffi;
#[cfg(feature = "cuda")]
pub mod gpu;

pub use config::{DsaConfig, DSA_LAYERS};
