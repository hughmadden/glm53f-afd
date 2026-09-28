//! Admission when device memory is tight (features `coordinator` and `cuda`; real weights of
//! layers 0-4 and the head, zero routed experts; skips without them). The serving shell's
//! scheduler drives `ServedForward` over a page pool with room for one request and one snapshot
//! mark:
//!
//! 1. a 600-token prompt is admitted and served; its prompt-end mark takes 486 pages of the pool,
//!    and its turn-end mark, which no longer fits, is skipped (the request completes);
//! 2. the same prompt again resumes from the prompt-end mark (a rewind out of pool pages) and
//!    gives the same tokens;
//! 3. a prompt the pool cannot hold even after evicting everything is refused at admission with
//!    the pool's message;
//!
//! and the forward allocates no device memory while it serves (its allocation counter; the
//! requests are greedy, so the shell's sampler is not used): passes (two-lane prefill included),
//! marks and admission run on what was allocated at start, so memory running short shows as a
//! refusal or a skipped snapshot, never as a CUDA out-of-memory error in a pass.
#![cfg(feature = "coordinator")]

mod common;

use common::*;
use glm53f_coordinator::{Job, Scheduler, SchedulerConfig};
use glm53f_forward::device;
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

/// Every token a job sent, or its error.
fn collect(rx: &std::sync::mpsc::Receiver<Result<u32, String>>) -> Result<Vec<u32>, String> {
    rx.try_iter().collect()
}

#[test]
fn admission_refuses_and_snapshots_skip_instead_of_running_out() {
    if !gpu_with(6.0) {
        return;
    }
    let cfg = ForwardConfig {
        max_rows: 256,
        lanes: 2,
        max_verify_rows: 8,
        max_requests: 4,
        ..ForwardConfig::default()
    };
    // A 600-token prompt with 8 new tokens reserves 600 + 1,024 + 64 tokens (the shell's output
    // allowance): 27 pages. One mark: 486 pages for these five layers (252 with BF16 KDA states,
    // GLM53F_TEST_NUMERICS=kda-state-bf16). Three to spare.
    let mark = if numerics().kda_state_bf16 { 252 } else { 486 };
    let request = 27;
    let Some(fwd) = forward_with(
        5,
        cfg,
        |_| Box::new(ZeroExperts),
        2,
        request + mark + 3,
        512,
    ) else {
        return;
    };
    assert_eq!(fwd.kv.config().layout.mark_pages(), mark);
    let model = ServedForward::new(fwd).unwrap();
    let slots: Vec<_> = (0..2).map(|_| model.fwd.kv.slot().unwrap()).collect();
    let page = slots[0].page_bytes();
    let (tx, rx) = std::sync::mpsc::channel();
    let mut sched = Scheduler::new(model, slots, None, SchedulerConfig::new(Vec::new()), rx);
    let run = |sched: &mut Scheduler<ServedForward>| {
        while !sched.is_idle() {
            sched.step(false);
        }
    };
    let before = device::allocations();

    // 1. Served; the prompt-end mark fits, the turn-end mark does not.
    let prompt = ids(1, 600);
    let (job, r1) = Job::new(prompt.clone(), 8, None);
    tx.send(job).unwrap();
    sched.step(false);
    run(&mut sched);
    let t1 = collect(&r1).expect("the first request");
    assert_eq!(t1.len(), 8);
    let free = sched.model().fwd.kv.free_pages();
    assert_eq!(sched.pool_stats().points, 1, "the prompt-end mark only");
    assert_eq!(free, 3, "the retained slot's pages and one mark");

    // 2. The same prompt resumes from the mark: the same tokens.
    let (job, r2) = Job::new(prompt.clone(), 8, None);
    tx.send(job).unwrap();
    sched.step(false);
    run(&mut sched);
    let t2 = collect(&r2).expect("the repeat");
    assert_eq!(t2, t1, "resumed from the mark, the tokens differ");
    assert_eq!(sched.pool_stats().device_hits, 1);

    // 3. A prompt no pool of this size holds: refused at admission.
    let (job, r3) = Job::new(ids(2, 32_000), 8, None);
    tx.send(job).unwrap();
    sched.step(false);
    run(&mut sched);
    let e = collect(&r3).expect_err("a 32,000-token prompt in a 516-page pool");
    assert!(e.contains("KV pool"), "{e}");
    assert_eq!(sched.stats.refused, 1);

    assert_eq!(
        device::allocations(),
        before,
        "serving allocated device memory"
    );
    eprintln!(
        "a pool of {} pages of {page} B: a 600-token request served with its prompt-end mark \
         ({mark} pages), its turn-end mark skipped ({free} pages free); the repeat resumed from \
         the mark with the same {} tokens; a 32,000-token prompt refused: {e}; no device \
         allocation while serving",
        request + mark + 3,
        t2.len()
    );
}
