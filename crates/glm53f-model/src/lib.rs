//! What the glm53f-afd engine knows about GLM-5.3-Flash before it touches a
//! GPU.
//!
//! - [`config`]: `config.json` parsed into typed fields and checked against the
//!   invariants the engine relies on; the DFlash2 drafter config.
//! - [`safetensors`]: strict header parsing, multi-shard checkpoints, header
//!   bundles, and positional reads of tensors or byte runs of them.
//! - [`catalog`]: every checkpoint tensor mapped to its role (layer, component,
//!   part) with dtype, shape, file and offsets, for the official FP8, EXL3 and
//!   NVFP4 checkpoints; quantized weights paired with their scales.
//! - [`slicing`]: the TP4 split of each routed expert over the expert ranks,
//!   exact for all three formats.
//! - [`planner`]: the coordinator and expert-rank memory plans of
//!   docs/SIZING.md, computed from components.
//!
//! Standard library only.

pub mod catalog;
pub mod config;
pub mod dtype;
pub mod error;
pub mod json;
pub mod planner;
pub mod safetensors;
pub mod slicing;

pub use error::{Error, Result};
