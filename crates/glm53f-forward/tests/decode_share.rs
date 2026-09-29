//! Decode during a long prefill over the real forward (features `coordinator` and `cuda`; real
//! weights of layers 0-4 and the head, zero routed experts; skips without them). The serving
//! shell's scheduler drives `ServedForward` with the daemon's prefill passes (8,192 rows in four
//! lanes of 2,048): a greedy request G decodes, then a prompt L of six passes arrives and G decodes
//! on while L prefills. A round holds one pass, as the daemon's 2 s rounds hold one 45-layer pass on
//! the target hardware. Once with `SchedulerConfig::decode_share` 0 and once with 0.2:
//!
//! 1. **The same bits.** G's and L's tokens are identical: the share moves only when G's steps
//!    run. L's passes are cut where they were, and each step carries one row a request, which every
//!    kernel computes independently of the step's other rows up to 8.
//! 2. **G's share of the time.** With 0.2, G keeps at least a tenth of its rate while L prefills
//!    (about 17% with six passes: the share, less the last round's), three times what it keeps with
//!    0 (one step a pass) at least, and L's time to first token grows by less than 60% (about 20%).
//!    The printed line gives the rates.
//!
//! ```sh
//! GLM53F_CHECKPOINT_DIR=... cargo test --release -p glm53f-forward --features coordinator \
//!   --test decode_share -- --nocapture
//! ```
#![cfg(feature = "coordinator")]

mod common;

use std::sync::mpsc::Receiver;
use std::time::Instant;

use common::*;
use glm53f_coordinator::{Job, Scheduler, SchedulerConfig};
use glm53f_forward::experts::ZeroExperts;
use glm53f_forward::forward::ForwardConfig;
use glm53f_forward::serve::ServedForward;

fn ids(seed: u64, n: usize) -> Vec<u32> {
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

/// Rows of one prefill pass: the daemon's four lanes of 2,048.
const PASS: usize = 8192;
/// L's passes.
const PASSES: usize = 6;
/// G's tokens: it decodes before L arrives, while L prefills, and after.
const G_TOKENS: usize = 256;

/// Tokens received so far (panics on an error).
fn drain(rx: &Receiver<Result<u32, String>>, into: &mut Vec<u32>) {
    into.extend(rx.try_iter().map(|t| t.expect("a token")));
}

/// One run: G's and L's tokens, G's rate alone and while L prefilled (tok/s), L's time to first
/// token (s).
struct Run {
    g: Vec<u32>,
    l: Vec<u32>,
    alone: f64,
    during: f64,
    ttft: f64,
}

impl Run {
    /// G's rate while L prefilled, against its rate alone.
    fn kept(&self) -> f64 {
        self.during / self.alone
    }
}

/// G, then L, with `share`. None (printed) without the weights.
fn run(share: f64) -> Option<Run> {
    let cfg = ForwardConfig {
        max_rows: PASS,
        lanes: 4,
        max_verify_rows: 8,
        max_requests: 2,
        ..ForwardConfig::default()
    };
    // L's 49,152 tokens and output allowance take 785 pages, its prompt and turn marks 486 each.
    let fwd = forward_with(5, cfg, |_| Box::new(ZeroExperts), 2, 4096, 1024)?;
    let model = ServedForward::new(fwd).unwrap();
    let slots: Vec<_> = (0..2).map(|_| model.fwd.kv.slot().unwrap()).collect();
    let mut sc = SchedulerConfig::new(Vec::new());
    // One pass a round: a round runs one segment at least, and the next would end past 1 ms.
    sc.segment_ms = 1.0;
    sc.decode_share = share;
    let (tx, rx) = std::sync::mpsc::channel();
    let mut sched = Scheduler::new(model, slots, None, sc, rx);
    let (mut g, mut l) = (Vec::new(), Vec::new());
    let (job, rg) = Job::new(ids(21, 64), G_TOKENS, None);
    tx.send(job).unwrap();
    // G alone: its rate over 32 tokens, after 8.
    while g.len() < 8 {
        sched.step(false);
        drain(&rg, &mut g);
    }
    let t0 = Instant::now();
    while g.len() < 40 {
        sched.step(false);
        drain(&rg, &mut g);
    }
    let alone = (g.len() - 8) as f64 / t0.elapsed().as_secs_f64();
    // L: G's tokens until L's first.
    let (job, rl) = Job::new(ids(22, PASSES * PASS), 4, None);
    tx.send(job).unwrap();
    let (sent, before) = (Instant::now(), g.len());
    while l.is_empty() {
        sched.step(false);
        drain(&rg, &mut g);
        drain(&rl, &mut l);
    }
    let ttft = sent.elapsed().as_secs_f64();
    let during = (g.len() - before) as f64 / ttft;
    assert!(g.len() < G_TOKENS, "G ended before L's first token");
    while !sched.is_idle() {
        sched.step(false);
    }
    drain(&rg, &mut g);
    drain(&rl, &mut l);
    assert_eq!(sched.stats.prefill_tokens, (64 + PASSES * PASS) as u64);
    Some(Run {
        g,
        l,
        alone,
        during,
        ttft,
    })
}

#[test]
fn a_running_request_keeps_its_share_while_a_long_prompt_prefills() {
    if !gpu_with(9.0) {
        return;
    }
    let Some(off) = run(0.0) else {
        return;
    };
    let on = run(0.2).expect("the weights were there a moment ago");
    assert_eq!((off.g.len(), off.l.len()), (G_TOKENS, 4));
    assert_eq!(on.g, off.g, "G's tokens moved with the share");
    assert_eq!(on.l, off.l, "L's tokens moved with the share");
    for (share, r) in [(0.0, &off), (0.2, &on)] {
        eprintln!(
            "decode share {share}: G {:.1} tok/s alone, {:.1} tok/s while L's {} tokens prefilled \
             ({:.1}% kept); L's first token after {:.3} s",
            r.alone,
            r.during,
            PASSES * PASS,
            100.0 * r.kept(),
            r.ttft
        );
    }
    assert!(
        on.kept() >= 0.1 && on.kept() >= 3.0 * off.kept(),
        "G kept {:.3} of its rate with the share, {:.3} without",
        on.kept(),
        off.kept()
    );
    assert!(
        on.ttft < 1.6 * off.ttft,
        "L's first token after {:.3} s with the share, {:.3} s without",
        on.ttft,
        off.ttft
    );
}
