//! `glm53f-coordinator`: the coordinator's serving shell, the model-agnostic half.
//!
//! The model plugs in through two traits ([`model`]): [`KvSlot`], one request's context state on
//! the device, and [`ModelForward`], batched passes over slots. Everything else is here:
//!
//! - [`scheduler`]: the batching scheduler. 16 slots by default; admission reserves a prompt
//!   plus an output allowance; a request that does not fit waits; long prompts prefill in
//!   segments of about 2 s with decode steps for the running requests in between; short prompts
//!   that arrive together prefill in one pass; every running request takes one batched step, a
//!   decode or a speculative verify window.
//! - [`pool`]: the slots, the device snapshot points (prompt end, turn end, abandoned prefill)
//!   and their banks; resuming a prompt in place or by a fork; eviction under pressure.
//! - [`radix`]: the prefix index over token ids (image identities included) that finds a
//!   prompt's longest snapshot point, and how much of it exists in whole pools.
//! - [`hostcache`]: the host RAM tier behind the device, least recently used first, over the
//!   slot's page and state export.
//! - [`sampling`] and `gpu` (feature `cuda`): DS41RT v15's sampling contract, the per-row token
//!   masks, the CPU reference and the GPU sampler.
//! - [`spec`]: the verify-length policy.
//! - [`queue`] and [`engine`]: the bounded request queue (429 before a response starts) and the
//!   [`glm53f_api::Engine`] implementation; [`glm_prompt`]: GLM-5.3-Flash's chat template and
//!   tokenizer for it.
//! - [`wire`] and [`fp8`]: the expert ranks' client (TCP or RDMA), its FP8 quantizer and the
//!   return handling.
//!
//! Ported from mimo26f-afd v1.2.0 `crates/mimo26-coordinator` (PROVENANCE.md records every
//! behavioural difference). Standard library only.

pub mod engine;
pub mod fp8;
pub mod glm_prompt;
pub mod hostcache;
pub mod model;
pub mod pool;
pub mod queue;
pub mod radix;
pub mod sampling;
pub mod scheduler;
pub mod spec;
pub mod streaming;
pub mod wire;

/// The shell's kernels and the CUDA runtime calls they need (feature `cuda`).
#[cfg(feature = "cuda")]
pub mod gpu;

pub use engine::{CoordinatorEngine, EngineConfig, PromptCodec};
pub use glm_prompt::GlmPrompts;
pub use hostcache::{HostCache, HostTierConfig, Kind};
pub use model::{
    DecodeRow, Draft, DraftRow, ImageSpan, KvSlot, Limits, ModelForward, Pick, Segment, SegmentOut, Token, Window,
    IMAGE_ID_BASE,
};
pub use queue::Queue;
pub use radix::RadixIndex;
pub use sampling::{After, Mask, Sampling};
pub use scheduler::{Job, SchedStats, Scheduler, SchedulerConfig};
pub use spec::SpecPolicy;
pub use wire::{ReturnPath, WireClient, WireConfig};

/// Slots by default (design decision D3: 16, in two lanes of 8).
pub const DEFAULT_SLOTS: usize = 16;

/// How many slots to build: `GLM53F_MAX_SLOTS`, default [`DEFAULT_SLOTS`], between 1 and 64.
/// The model creates the slots (they are its type); the queue's depth defaults to the same count.
pub fn slot_count() -> usize {
    std::env::var("GLM53F_MAX_SLOTS").ok().and_then(|v| v.parse().ok()).unwrap_or(DEFAULT_SLOTS).clamp(1, 64)
}
