//! Shared setup of the drafter's GPU tests: a forward over all 45 decoder layers with the DFlash2
//! drafter attached, on one GPU next to other work.
//!
//! - `GLM53F_CHECKPOINT_DIR` (the coordinator's tensors), `GLM53F_EXPERTS_DIR` (the routed
//!   experts of layers 3 and 4), `GLM53F_DFLASH_DIR` (the drafter's checkpoint).
//! - `GLM53F_DRAFT_TEST_LAYERS` (default 5): decoder layers loaded from the checkpoint; every
//!   later layer runs on the weights of the last loaded layer of its kinds
//!   (`DeviceModel::load_repeating`; layers 0-4 cover every kind), so the 45-layer forward and its
//!   taps at layers 5 to 42 fit in about 3 GB of weights. 45 loads every layer (13.96 GB).
//! - Routed experts: the official FP8 experts of layers 3 and 4 on this GPU, zeros for every
//!   other MoE layer.
//!
//! The text is garbage by design: the tests check the drafter's plumbing and that speculation
//! never changes a token, which holds for any weights. Anything missing makes the tests print
//! why and pass.
#![allow(dead_code)]

use std::sync::Arc;

use glm53f_forward::device::Stream;
use glm53f_forward::draft::Dflash;
use glm53f_forward::embed::HostEmbedding;
use glm53f_forward::experts::{ExpertBackend, ExpertCall, LocalFp8Experts, ZeroExperts};
use glm53f_forward::forward::{ForwardConfig, GlmForward};
use glm53f_forward::gemm::Fp8Act;
use glm53f_forward::kv::{KvConfig, KvPool};
use glm53f_forward::kvplan::KvLayout;
use glm53f_forward::shape::ModelShape;
use glm53f_forward::weights::{open_checkpoint, DeviceModel};
use glm53f_forward::Result;

use crate::common::{checkpoint_dir, env_dir, experts_dir, gpu_with, numerics};

/// The MoE layers whose routed experts run (the others give zeros).
pub const EXPERT_LAYERS: [usize; 2] = [3, 4];

/// Local FP8 experts for [`EXPERT_LAYERS`], a routed output of zeros for every other layer.
pub struct SomeExperts {
    pub local: LocalFp8Experts,
}

impl ExpertBackend for SomeExperts {
    fn submit(&mut self, call: &ExpertCall<'_>, stream: &Stream) -> Result<()> {
        if EXPERT_LAYERS.contains(&call.layer) {
            self.local.submit(call, stream)
        } else {
            ZeroExperts.submit(call, stream)
        }
    }

    fn finish(&mut self, call: &ExpertCall<'_>, stream: &Stream) -> Result<()> {
        if EXPERT_LAYERS.contains(&call.layer) {
            self.local.finish(call, stream)
        } else {
            ZeroExperts.finish(call, stream)
        }
    }

    /// Both run a call's work in `submit`: two lanes may have calls in flight.
    fn depth(&self) -> usize {
        2
    }
}

/// Deterministic token ids in the vocabulary.
pub fn ids(seed: u64, n: usize) -> Vec<u32> {
    let mut s = seed;
    (0..n)
        .map(|_| {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            1000 + ((s >> 33) % 150_000) as u32
        })
        .collect()
}

/// Snapshot marks the pool has room for beyond the tests' pages (a mark takes pages of the pool:
/// 376 for the whole model).
pub const MARKS: usize = 4;

/// A 45-layer forward with the drafter attached and a KV pool of `slots` slots of up to
/// `max_pages` pages (64 tokens each), `pages` in all plus room for [`MARKS`] snapshot marks;
/// local experts get `expert_gib` GiB. None (printed) when data or GPU memory is missing.
pub fn drafted_forward(
    cfg: ForwardConfig,
    slots: usize,
    max_pages: usize,
    pages: usize,
    expert_gib: f64,
) -> Option<GlmForward> {
    let dir = checkpoint_dir()?;
    let edir = experts_dir()?;
    let Some(ddir) = env_dir(&["GLM53F_DFLASH_DIR"]) else {
        eprintln!("skip: GLM53F_DFLASH_DIR is not set");
        return None;
    };
    let loaded: usize = std::env::var("GLM53F_DRAFT_TEST_LAYERS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(5);
    // Plus the marks' pages (0.6 GiB); FP8 KDA projections take 4.26 GiB less.
    let num = numerics();
    let need = if loaded >= 45 { 20.6 } else { 7.6 } + expert_gib - if num.kda_fp8 { 4.26 } else { 0.0 };
    if !gpu_with(need) {
        return None;
    }
    let (mcfg, ckpt) = match open_checkpoint(&dir) {
        Ok(x) => x,
        Err(e) => {
            eprintln!("skip: cannot open the checkpoint: {e}");
            return None;
        }
    };
    let shape = ModelShape::full(&mcfg.text).unwrap();
    let t0 = std::time::Instant::now();
    let model = DeviceModel::load_with(&ckpt, &shape, loaded, num.weights()).unwrap();
    let embed = HostEmbedding::load(&ckpt).unwrap();
    let stream = Arc::new(Stream::new().unwrap());
    let d = Dflash::load(&ddir, &model, &embed, &stream).unwrap();
    eprintln!(
        "45 decoder layers ({} loaded, the rest repeating them) and the head: {:.2} GB; the \
         drafter {:.2} GB; {:.1} s",
        loaded.min(45),
        model.bytes as f64 / 1e9,
        d.weight_bytes() as f64 / 1e9,
        t0.elapsed().as_secs_f64()
    );
    let layout = num.layout(KvLayout::new(&shape, Some(d.config())));
    let kv = KvPool::new(
        KvConfig {
            layout,
            max_slots: slots,
            pages: pages + MARKS * layout.mark_pages(),
            max_pages,
            base_pages: 1,
        },
        stream.clone(),
    )
    .unwrap();
    let local = LocalFp8Experts::new(
        &edir,
        (expert_gib * (1u64 << 30) as f64) as usize,
        cfg.max_rows.max(cfg.max_verify_rows),
        &stream,
        Fp8Act::Bf16,
    )
    .unwrap();
    let mut fwd = GlmForward::new(
        model,
        embed,
        kv,
        Box::new(SomeExperts { local }),
        num.config(cfg),
    )
    .unwrap();
    fwd.attach_drafter(d).unwrap();
    Some(fwd)
}
