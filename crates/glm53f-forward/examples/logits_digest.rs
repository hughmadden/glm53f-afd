//! Teacher-forced scoring of a fixed prompt through two-lane prefill passes, and a digest of the
//! logits: a change meant to move no bit (a kernel's schedule, where buffers live, row blocks) is
//! checked by running this before and after it on the same GPU and comparing the lines.
//!
//! ```sh
//! GLM53F_CHECKPOINT_DIR=/path/to/checkpoint \
//!   cargo run --release -p glm53f-forward --features cuda --example logits_digest
//! ```
//!
//! All 45 decoder layers run on the weights of layers 0-4, as in `prefill_bench`; the routed
//! experts return zeros. The prompt is longer than a DSA row's 2,051-token selection, so the
//! later passes' DSA layers select. Environment:
//! - `GLM53F_DIGEST_LANE_ROWS`: rows per lane (default 1000; passes of twice that);
//! - `GLM53F_DIGEST_TOKENS`: prompt tokens (default 6000);
//! - `GLM53F_DIGEST_EVERY`: score every this many positions (default 37);
//! - `GLM53F_DIGEST_MLA_BLOCK`: `ForwardConfig::mla_block_rows` (default: the default).

use std::path::PathBuf;
use std::sync::Arc;

use glm53f_forward::device::Stream;
use glm53f_forward::embed::HostEmbedding;
use glm53f_forward::experts::ZeroExperts;
use glm53f_forward::forward::{ForwardConfig, GlmForward};
use glm53f_forward::kv::{KvConfig, KvPool};
use glm53f_forward::kvplan::KvLayout;
use glm53f_forward::shape::{ModelShape, SAMPLE_VOCAB, VOCAB};
use glm53f_forward::weights::{open_checkpoint, DeviceModel};

fn env_usize(k: &str, d: usize) -> usize {
    std::env::var(k)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(d)
}

fn main() {
    let Some(dir) = std::env::var_os("GLM53F_CHECKPOINT_DIR").map(PathBuf::from) else {
        eprintln!("set GLM53F_CHECKPOINT_DIR");
        return;
    };
    let lane = env_usize("GLM53F_DIGEST_LANE_ROWS", 1000);
    let n = env_usize("GLM53F_DIGEST_TOKENS", 6000);
    let every = env_usize("GLM53F_DIGEST_EVERY", 37).max(1);
    let (cfg, ckpt) = open_checkpoint(&dir).unwrap();
    let shape = ModelShape::full(&cfg.text).unwrap();
    let model = DeviceModel::load_repeating(&ckpt, &shape, 5).unwrap();
    let embed = HostEmbedding::load(&ckpt).unwrap();
    let stream = Arc::new(Stream::new().unwrap());
    let pages = KvLayout::pages_for(n + 64);
    let kv = KvPool::new(
        KvConfig {
            layout: KvLayout::new(&shape, None),
            max_slots: 1,
            pages,
            max_pages: pages.div_ceil(4) * 4,
            base_pages: 0,
        },
        stream,
    )
    .unwrap();
    #[allow(unused_mut)]
    let mut fcfg = ForwardConfig {
        max_rows: 2 * lane,
        lanes: 2,
        max_requests: 1,
        max_verify_rows: 8,
        ..ForwardConfig::default()
    };
    if let Ok(v) = std::env::var("GLM53F_DIGEST_MLA_BLOCK") {
        fcfg.mla_block_rows = v.parse().unwrap();
    }
    let mut fwd = GlmForward::new(model, embed, kv, Box::new(ZeroExperts), fcfg).unwrap();
    let mut s = 7u64;
    let tokens: Vec<u32> = (0..n)
        .map(|_| {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            1000 + ((s >> 33) % 150_000) as u32
        })
        .collect();
    let rows: Vec<usize> = (0..n).filter(|r| r % every == every - 1).collect();
    let mut kv = fwd.kv.slot().unwrap();
    kv.reserve(n).unwrap();
    // FNV-1a over the logits' bytes, and each scored row's argmax.
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    let mut picks = Vec::with_capacity(rows.len());
    fwd.score_each(&mut kv, &tokens, &rows, 2 * lane, |_, l| {
        for v in l {
            for b in v.to_le_bytes() {
                h = (h ^ u64::from(b)).wrapping_mul(0x0000_0100_0000_01b3);
            }
        }
        let (mut best, mut at) = (f32::NEG_INFINITY, 0);
        for (i, &v) in l[..SAMPLE_VOCAB].iter().enumerate() {
            if v > best {
                (best, at) = (v, i);
            }
        }
        picks.push(at);
        Ok(())
    })
    .unwrap();
    let head: Vec<u64> = picks.iter().take(8).map(|&p| p as u64).collect();
    println!(
        "logits digest {h:016x}: {} rows x {VOCAB} of a {n}-token prompt in passes of {} rows \
         (two lanes); argmax of the first rows {head:?}",
        rows.len(),
        2 * lane
    );
}
