//! Snapshots and the RAM tier over the real KV (features `coordinator` and `cuda`; real weights of
//! layers 0-4 and the head, zero routed experts; skips without them). The serving shell's
//! scheduler drives `ServedForward` with a host RAM tier and no bank cap, and serves the same three
//! 600-token requests in the same order twice: B runs long, A runs and finishes beside it, then C
//! arrives.
//!
//! 1. On a pool with room for everything, every snapshot mark stays on the device: six marks, no
//!    RAM traffic.
//! 2. On a pool that lacks pages for C's reservation, C's admission evicts exactly one point, the
//!    least recently used: B's prompt mark, while B is running. It is stored to RAM and its pool
//!    pages freed; B runs on. C's prompt mark, at the end of its prefill in the same step, finds
//!    too few pages too and evicts the next least recently used point the same way: A's prompt
//!    mark (it used to be skipped). Every request's tokens equal the run without pressure.
//! 3. B's prompt again restores from RAM (no prefill) and gives B's tokens.
//!
//! The forward allocates no device memory while serving: eviction, the RAM copies and the restore
//! run on the pool allocated at start.
#![cfg(feature = "coordinator")]

mod common;

use std::sync::mpsc::{Receiver, Sender};

use common::*;
use glm53f_coordinator::{HostCache, HostTierConfig, Job, Kind, Scheduler, SchedulerConfig};
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

/// A 600-token prompt reserves 600 + 1,024 + 64 tokens (the shell's output allowance): 27 pages.
const REQUEST: usize = 27;
/// Tokens B generates (it runs across A's whole life and C's admission).
const B_TOKENS: usize = 24;

/// A scheduler over a forward of layers 0-4 with a pool of `pages` pages, three slots and a RAM
/// tier. None (printed) without the weights or the GPU memory.
fn serve(pages: usize) -> Option<(Scheduler<ServedForward>, Sender<Job>)> {
    let cfg = ForwardConfig {
        max_rows: 256,
        lanes: 2,
        max_verify_rows: 8,
        max_requests: 4,
        ..ForwardConfig::default()
    };
    let fwd = forward_with(5, cfg, |_| Box::new(ZeroExperts), 3, pages, 64)?;
    let model = ServedForward::new(fwd).unwrap();
    let slots: Vec<_> = (0..3).map(|_| model.fwd.kv.slot().unwrap()).collect();
    let tier = HostTierConfig {
        pages: 64,
        states: 4,
        page_tokens: slots[0].page_tokens(),
        page_bytes: slots[0].page_bytes(),
        state_bytes: slots[0].state_bytes(),
        min_tokens: 512,
        pin: true,
    };
    let cache = HostCache::new(tier).unwrap();
    let mut cfg = SchedulerConfig::new(Vec::new());
    cfg.bank = 0;
    let (tx, rx) = std::sync::mpsc::channel();
    Some((Scheduler::new(model, slots, Some(cache), cfg, rx), tx))
}

fn submit(tx: &Sender<Job>, prompt: &[u32], max: usize) -> Receiver<Result<u32, String>> {
    let (job, rx) = Job::new(prompt.to_vec(), max, None);
    tx.send(job).unwrap();
    rx
}

/// Tokens received so far (panics on an error).
fn drain(rx: &Receiver<Result<u32, String>>, into: &mut Vec<u32>) {
    into.extend(rx.try_iter().map(|t| t.expect("a token")));
}

/// What one pass of the scene saw.
struct Scene {
    /// B's, A's and C's tokens.
    outs: [Vec<u32>; 3],
    /// Pool counters and RAM captures after the step that admitted C.
    evicted: (u64, u64),
    captures: u64,
    /// The RAM tier's snapshots after C's admission: (tokens, bank).
    held: Vec<(Vec<u32>, Kind)>,
    /// Device points at the end.
    points: usize,
}

/// B (600 tokens, 24 to generate) until it decodes; A (600, 4) beside it until A finishes, its
/// prompt and turn marks retained; then C (600, 4), admitted in one step, and everything to the
/// end.
fn scene(sched: &mut Scheduler<ServedForward>, tx: &Sender<Job>) -> Scene {
    let (b, a, c) = (ids(11, 600), ids(12, 600), ids(13, 600));
    let before = device::allocations();
    let mut outs: [Vec<u32>; 3] = Default::default();
    let rb = submit(tx, &b, B_TOKENS);
    while sched.in_flight().0 < 1 {
        sched.step(false);
    }
    let ra = submit(tx, &a, 4);
    while outs[1].len() < 4 {
        sched.step(false);
        drain(&ra, &mut outs[1]);
    }
    assert_eq!(sched.in_flight().0, 1, "B still decoding");
    assert_eq!(
        sched.device_points(),
        3,
        "B's prompt mark, A's prompt and turn marks"
    );
    let rc = submit(tx, &c, 4);
    sched.step(false);
    let st = sched.pool_stats();
    let hc = sched.host_cache().unwrap();
    let (captures, held) = (hc.stats.captures, hc.snapshots());
    while !sched.is_idle() {
        sched.step(false);
    }
    drain(&rb, &mut outs[0]);
    drain(&rc, &mut outs[2]);
    assert_eq!(
        (outs[0].len(), outs[1].len(), outs[2].len()),
        (B_TOKENS, 4, 4)
    );
    assert_eq!(
        device::allocations(),
        before,
        "serving allocated device memory"
    );
    Scene {
        outs,
        evicted: (st.evicted_points, st.evicted_in_flight),
        captures,
        held,
        points: sched.device_points(),
    }
}

#[test]
fn a_running_request_s_mark_goes_to_ram_under_pool_pressure() {
    if !gpu_with(6.0) {
        return;
    }
    let mark = if numerics().kda_state_bf16 { 252 } else { 486 };

    // 1. Room for everything: three reservations, six marks.
    let Some((mut sched, tx)) = serve(3 * REQUEST + 6 * mark + 16) else {
        return;
    };
    assert_eq!(sched.model().fwd.kv.config().layout.mark_pages(), mark);
    let calm = scene(&mut sched, &tx);
    assert_eq!(calm.evicted, (0, 0));
    assert_eq!(calm.captures, 0);
    assert_eq!(
        sched.host_cache().unwrap().stats.captures,
        0,
        "no pressure, no RAM traffic"
    );
    assert_eq!(calm.points, 6, "every mark on the device");
    drop((sched, tx));

    // 2. Ten pages spare once B and A hold theirs: C's 27 need one mark's pages, and C's prompt
    // mark then another's.
    let Some((mut sched, tx)) = serve(2 * REQUEST + 3 * mark + 10) else {
        return;
    };
    let tight = scene(&mut sched, &tx);
    assert_eq!(
        tight.evicted,
        (2, 1),
        "a running request's point for C's admission, a retained one for C's prompt mark"
    );
    assert_eq!(tight.captures, 2);
    assert_eq!(
        tight.held,
        vec![(ids(11, 600), Kind::Prompt), (ids(12, 600), Kind::Prompt)],
        "B's prompt mark, then A's, in RAM"
    );
    assert_eq!(
        tight.outs, calm.outs,
        "the tokens of the run without pressure"
    );

    // 3. B's prompt again: restored from RAM, B's tokens.
    let before = device::allocations();
    let rb = submit(&tx, &ids(11, 600), B_TOKENS);
    sched.step(false);
    while !sched.is_idle() {
        sched.step(false);
    }
    let mut again = Vec::new();
    drain(&rb, &mut again);
    let st = sched.pool_stats();
    assert_eq!(st.host_restores, 1);
    assert_eq!(again, calm.outs[0], "restored from RAM, B's tokens");
    assert_eq!(device::allocations(), before);
    eprintln!(
        "{mark}-page marks: without pressure 6 marks on the device, 0 RAM stores; with a pool 10 pages \
         short of C's {REQUEST}, C's admission stored B's prompt mark to RAM while B ran and C's \
         prompt mark stored A's (2 points evicted), every request's tokens unchanged; B's prompt \
         restored from RAM with the same {} tokens",
        again.len()
    );
}
