//! Two-lane decode and verify against the same requests as two passes (feature `cuda`; the real
//! weights of layers 0-4 and the head, the official FP8 experts of layers 3 and 4, and for test 2
//! the DFlash2 drafter; skips without them).
//!
//! **Exact.** A decode or verify pass in two lanes is cut between requests
//! (`ForwardConfig::decode_lane_rows`), and a request reads nothing of another's, so a lane is a
//! pass over its own requests. Two lanes must therefore give the bits of two passes, lane A's
//! requests then lane B's, in everything they leave: logits and picks, and each slot's KDA states,
//! conv windows, committed MLA latents and pooled keys, and DSA tails; with the drafter, its
//! contexts and rings and the drafts read from them. The tests check that bit for bit.
//!
//! **Against one pass** over every row, the lanes change the row counts of the tensor-core GEMMs
//! once a lane holds more than 8 rows, as any batch does: up to 8 rows every kernel is
//! row-independent (`tests/verify_commit.rs`), so the decode steps of six one-row requests equal
//! one pass's bit for bit, and a verify pass of 25 rows moves within rounding (the same tokens on
//! every row whose two best logits are at least [`TIE`] apart, logits within the chain test's 5%).
//!
//! 1. **Six requests** (prompts of 3 to 41 tokens), layers 0-4 and the head, the local FP8
//!    experts: 3 decode steps (lanes of 3 and 3 requests), a verify round with windows of 1 to 8
//!    rows and partial keeps (the lanes cut to balance the rows), its commit, and 2 decode steps
//!    from token ids on the device; against the two passes per step, with the experts' calls in
//!    flight two at a time and one at a time (lane A's head then runs before lane B's last call is
//!    collected), and against one pass. Then the bounds: passes outside
//!    `decode_lane_rows ..= decode_lane_max_rows`, and one request, run in one lane.
//! 2. **The drafter** (all 45 decoder layers with DFlash2 attached, `drafting`; routed experts of
//!    zeros): four prompts, a verify round in two lanes with its commit, 2 decode steps; against
//!    the two passes per step: picks and logits, every stored ring row, the greedy drafts after.
//!    With the drafter's memory reserved first, no pass allocates device memory.
//! 3. **On four rank daemons** (feature `coordinator`; `GLM53F_RANK_BIN` and `GLM53F_RANK_DIRS` as
//!    in `tests/remote_experts.rs`): test 1's script with `RemoteExperts`, layers 3 and 4's EXL3
//!    experts on four `glm53f-rank serve` processes on this GPU over loopback TCP (one exchange in
//!    flight): two lanes against two passes, bit for bit (the rank kernel gives a row the same bits
//!    in any batch of up to 64 rows, and a lane's exchange takes the return path a pass of its
//!    rows would). Three wire configurations: four planes with the default fast paths; row slices
//!    from 4 rows (the decode lanes' exchanges four planes, the verify lanes' row slices); requests
//!    written by the frame-fill kernel. The `STEP` lines, with the wire's record, show the schedule
//!    on a shared GPU, not the overlap of separate machines.
//!
//! ```sh
//! GLM53F_CHECKPOINT_DIR=... GLM53F_EXPERTS_DIR=... GLM53F_DFLASH_DIR=... \
//! GLM53F_RANK_BIN=... GLM53F_RANK_DIRS=a,b,c,d \
//!   cargo test --release -p glm53f-forward --features coordinator --test decode_lanes -- --nocapture --test-threads=1
//! ```
#![cfg(feature = "cuda")]

mod common;
mod drafting;

use std::sync::Arc;

use common::*;
use glm53f_dsa::cache::{
    LATENT_RECORD_BYTES, PAGE_POOL_CODES_OFFSET, PAGE_POOL_SCALES_OFFSET, PAGE_TOKENS,
    TAIL_TOKEN_BYTES,
};
use glm53f_forward::device::{self, DeviceBuffer, Stream};
use glm53f_forward::draft::DraftReq;
use glm53f_forward::experts::{ExpertBackend, ExpertCall, LocalFp8Experts, ZeroExperts};
use glm53f_forward::forward::{ForwardConfig, GlmForward, Mode};
use glm53f_forward::gemm::Fp8Act;
use glm53f_forward::kv::GlmKv;
use glm53f_forward::shape::{INDEX_DIM, SAMPLE_VOCAB, VOCAB};

const LAYERS: usize = 5;

/// Near ties (`tests/lanes.rs`): rows whose two best logits are closer may pick either with
/// rounding differences of the row counts' size.
const TIE: f32 = 0.25;

fn ids(seed: u64, n: usize) -> Vec<u32> {
    drafting::ids(seed, n)
}

/// The cut the forward documents: after the request that splits the rows most evenly, lane A the
/// larger on a tie. Returns lane A's requests.
fn balanced_cut(rows: &[usize]) -> usize {
    let total: usize = rows.iter().sum();
    let (mut best, mut k, mut at) = (usize::MAX, 0, 0);
    for (i, &r) in rows[..rows.len() - 1].iter().enumerate() {
        at += r;
        let off = (2 * at).abs_diff(total);
        if off <= best {
            (best, k) = (off, i + 1);
        }
    }
    k
}

/// Routed experts with one call in flight at most: lane A's call is collected before lane B's
/// goes out, and lane A's head runs while lane B's is out.
struct OneInFlight<B: ExpertBackend>(B);

impl<B: ExpertBackend> ExpertBackend for OneInFlight<B> {
    fn submit(&mut self, call: &ExpertCall<'_>, stream: &Stream) -> glm53f_forward::Result<()> {
        self.0.submit(call, stream)
    }
    fn finish(&mut self, call: &ExpertCall<'_>, stream: &Stream) -> glm53f_forward::Result<()> {
        self.0.finish(call, stream)
    }
    fn depth(&self) -> usize {
        1
    }
}

/// What a slot keeps, as bytes: every KDA layer's state and conv window, every DSA layer's tail
/// (its valid tokens) and the committed rows of its pages (latent records, complete pools' keys
/// and scales).
fn kept(fwd: &GlmForward, kv: &GlmKv) -> Vec<u8> {
    let shape = fwd.shape();
    let mut out = Vec::new();
    for j in 0..shape.kda_layers {
        out.extend(
            kv.download_state(j)
                .unwrap()
                .iter()
                .flat_map(|x| x.to_le_bytes()),
        );
        out.extend(
            kv.download_conv(j)
                .unwrap()
                .iter()
                .flat_map(|x| x.to_le_bytes()),
        );
    }
    let t = kv.tokens();
    for j in 0..shape.dsa_layers {
        let tail = kv.download_tail(j).unwrap();
        let n = u32::from_le_bytes(tail[..4].try_into().unwrap()) as usize;
        out.extend(&tail[..16 + n.min(3) * TAIL_TOKEN_BYTES]);
        for p in 0..t.div_ceil(PAGE_TOKENS) {
            let b = kv.download_page_block(p, j).unwrap();
            let rows = (t - p * PAGE_TOKENS).min(PAGE_TOKENS);
            let pools = rows / 4;
            out.extend(&b[..rows * LATENT_RECORD_BYTES]);
            out.extend(&b[PAGE_POOL_CODES_OFFSET..PAGE_POOL_CODES_OFFSET + pools * INDEX_DIM]);
            out.extend(&b[PAGE_POOL_SCALES_OFFSET..PAGE_POOL_SCALES_OFFSET + pools * 4]);
        }
    }
    out
}

/// A run's outputs: every pass's picks and logit rows in request order, and what each slot kept.
struct Outcome {
    picks: Vec<u32>,
    logits: Vec<f32>,
    kept: Vec<Vec<u8>>,
}

/// How a run takes each step: in two lanes, as two passes cut where the forward cuts, or as one
/// pass in one lane.
#[derive(Clone, Copy, PartialEq, Debug)]
enum How {
    Lanes,
    TwoPasses,
    OnePass,
}

/// Decode steps: one token per request.
fn decode_step(fwd: &mut GlmForward, kvs: &mut [GlmKv], tokens: &[u32], how: How, o: &mut Outcome) {
    let groups: Vec<std::ops::Range<usize>> = if how == How::TwoPasses {
        let k = balanced_cut(&vec![1; kvs.len()]);
        vec![0..k, k..kvs.len()]
    } else {
        vec![0..kvs.len()]
    };
    for g in groups {
        let mut rows: Vec<(&mut GlmKv, u32)> = kvs[g.clone()]
            .iter_mut()
            .zip(&tokens[g.clone()])
            .map(|(k, &t)| (k, t))
            .collect();
        o.picks.extend(fwd.decode(&mut rows).unwrap());
        o.logits.extend(fwd.logits(g.len()).unwrap());
    }
}

/// A verify round: `windows[i]` rows for request `i`, then its commit keeping `keep[i]`.
fn verify_round(
    fwd: &mut GlmForward,
    kvs: &mut [GlmKv],
    windows: &[Vec<u32>],
    keep: &[usize],
    how: How,
    o: &mut Outcome,
) {
    let lens: Vec<usize> = windows.iter().map(|w| w.len()).collect();
    let groups: Vec<std::ops::Range<usize>> = if how == How::TwoPasses {
        let k = balanced_cut(&lens);
        vec![0..k, k..kvs.len()]
    } else {
        vec![0..kvs.len()]
    };
    for g in groups {
        let picks = {
            let mut w: Vec<(&mut GlmKv, &[u32])> = kvs[g.clone()]
                .iter_mut()
                .zip(&windows[g.clone()])
                .map(|(k, w)| (k, &w[..]))
                .collect();
            fwd.verify(&mut w).unwrap()
        };
        o.picks.extend(picks.concat());
        o.logits
            .extend(fwd.logits(lens[g.clone()].iter().sum()).unwrap());
        let mut slots: Vec<&mut GlmKv> = kvs[g.clone()].iter_mut().collect();
        fwd.commit(&mut slots, &keep[g]).unwrap();
    }
}

/// Decode steps from token ids already on the device (`ids`, one i32 per request).
fn decode_device_step(
    fwd: &mut GlmForward,
    kvs: &mut [GlmKv],
    ids: &DeviceBuffer,
    how: How,
    o: &mut Outcome,
) {
    let groups: Vec<std::ops::Range<usize>> = if how == How::TwoPasses {
        let k = balanced_cut(&vec![1; kvs.len()]);
        vec![0..k, k..kvs.len()]
    } else {
        vec![0..kvs.len()]
    };
    for g in groups {
        let mut rows: Vec<&mut GlmKv> = kvs[g.clone()].iter_mut().collect();
        o.picks
            .extend(fwd.decode_device(&mut rows, ids.ptr(g.start)).unwrap());
        o.logits.extend(fwd.logits(g.len()).unwrap());
    }
}

const PROMPTS: [usize; 6] = [3, 17, 9, 41, 26, 5];
const WINDOWS: [usize; 6] = [8, 3, 5, 1, 6, 2];
const KEEP: [usize; 6] = [3, 3, 1, 1, 6, 2];

/// The script of test 1 over fresh slots: prefill each prompt (its own pass), 3 decode steps, a
/// verify round and its commit, 2 decode steps from device ids. With [`How::Lanes`], checks
/// that the decode and verify passes ran in two lanes, cut where [`balanced_cut`] says.
fn script(fwd: &mut GlmForward, how: How) -> Outcome {
    let prompts: Vec<Vec<u32>> = PROMPTS
        .iter()
        .enumerate()
        .map(|(i, &n)| ids(10 + i as u64, n))
        .collect();
    let mut kvs: Vec<GlmKv> = prompts
        .iter()
        .map(|p| {
            let mut kv = fwd.kv.slot().unwrap();
            kv.reserve(p.len() + 32).unwrap();
            kv
        })
        .collect();
    let mut o = Outcome {
        picks: Vec::new(),
        logits: Vec::new(),
        kept: Vec::new(),
    };
    for (kv, p) in kvs.iter_mut().zip(&prompts) {
        o.picks.extend(fwd.prefill(&mut [(kv, &p[..])]).unwrap());
        o.logits.extend(fwd.logits(1).unwrap());
    }
    let lanes_before = fwd.decode_lane_passes();
    let n = kvs.len();
    for s in 0..3 {
        decode_step(fwd, &mut kvs, &ids(100 + s, n), how, &mut o);
        if how == How::Lanes {
            let t = fwd.take_step_trace().expect("a traced decode pass");
            assert_eq!((t.mode, &t.requests[..]), (Mode::Decode, &[3, 3][..]));
            assert_eq!(t.rows, vec![3, 3]);
        }
    }
    let windows: Vec<Vec<u32>> = WINDOWS
        .iter()
        .enumerate()
        .map(|(i, &w)| ids(200 + i as u64, w))
        .collect();
    verify_round(fwd, &mut kvs, &windows, &KEEP, how, &mut o);
    if how == How::Lanes {
        let t = fwd.take_step_trace().expect("a traced verify pass");
        let k = balanced_cut(&WINDOWS);
        let a: usize = WINDOWS[..k].iter().sum();
        assert_eq!(t.mode, Mode::Verify);
        assert_eq!(t.requests, vec![k, n - k]);
        assert_eq!(t.rows, vec![a, 25 - a]);
        assert!(
            t.step.commit_ms > 0.0,
            "the verify pass's step closes at its commit"
        );
    }
    let dev = DeviceBuffer::alloc(n * 4).unwrap();
    for s in 0..2 {
        let t: Vec<i32> = ids(300 + s, n).iter().map(|&x| x as i32).collect();
        dev.upload(&t).unwrap();
        decode_device_step(fwd, &mut kvs, &dev, how, &mut o);
    }
    if how == How::Lanes {
        // 3 decode steps, the verify round, 2 device decode steps.
        assert_eq!(
            fwd.decode_lane_passes() - lanes_before,
            6,
            "passes in two lanes"
        );
    } else {
        assert_eq!(fwd.decode_lane_passes(), lanes_before, "passes in one lane");
    }
    o.kept = kvs.iter().map(|kv| kept(fwd, kv)).collect();
    o
}

fn bits(a: &[f32], b: &[f32]) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits())
}

fn argmax(v: &[f32]) -> usize {
    let mut b = 0;
    for i in 0..SAMPLE_VOCAB {
        if v[i] > v[b] {
            b = i;
        }
    }
    b
}

/// Rows of `reference` whose two best logits are at least [`TIE`] apart, and how many of them
/// `a` and `b` pick the same token on.
fn decided(reference: &[f32], a: &[u32], b: &[u32]) -> (usize, usize) {
    let (mut rows, mut agree) = (0, 0);
    for (r, (x, y)) in a.iter().zip(b).enumerate() {
        let v = &reference[r * VOCAB..r * VOCAB + SAMPLE_VOCAB];
        let best = argmax(v);
        let second = (0..SAMPLE_VOCAB)
            .filter(|&i| i != best)
            .map(|i| v[i])
            .fold(f32::NEG_INFINITY, f32::max);
        if v[best] - second >= TIE {
            rows += 1;
            agree += usize::from(x == y);
        }
    }
    (rows, agree)
}

#[test]
fn decode_and_verify_in_two_lanes() {
    if !gpu_with(9.0) {
        return;
    }
    let Some(edir) = experts_dir() else {
        return;
    };
    let cfg = ForwardConfig {
        max_rows: 64,
        lanes: 2,
        min_lane_rows: 8,
        decode_lane_rows: 2,
        max_verify_rows: 32,
        max_requests: 8,
        ..ForwardConfig::default()
    };
    let local = |st: &Arc<Stream>| {
        LocalFp8Experts::new(&edir, 3 << 30, 64, st, Fp8Act::Bf16).expect("local experts")
    };
    let Some(mut fwd) = forward_with(LAYERS, cfg, |st| Box::new(local(st)), 12, 96, 16) else {
        return;
    };
    fwd.set_lane_trace(true, true);
    let two = script(&mut fwd, How::Lanes);
    fwd.set_lane_trace(false, false);
    fwd.cfg.decode_lane_rows = 0;
    let seq = script(&mut fwd, How::TwoPasses);
    let one = script(&mut fwd, How::OnePass);
    // One call in flight at a time: lane A's head runs before lane B's last call is collected.
    let stream = fwd.stream().clone();
    fwd.set_experts(Box::new(OneInFlight(local(&stream))));
    fwd.cfg.decode_lane_rows = 2;
    fwd.set_lane_trace(true, false);
    let two1 = script(&mut fwd, How::Lanes);

    let exact = bits(&two.logits, &seq.logits) && two.picks == seq.picks;
    let kept_exact = two.kept == seq.kept;
    let depth1 =
        bits(&two1.logits, &seq.logits) && two1.picks == seq.picks && two1.kept == seq.kept;
    // Against one pass: the prompts and the 3 decode steps (at most 8 rows a pass) bit for bit;
    // from the verify round (25 rows) on, within rounding.
    let head_rows = PROMPTS.len() + 3 * PROMPTS.len();
    let before = bits(
        &two.logits[..head_rows * VOCAB],
        &one.logits[..head_rows * VOCAB],
    ) && two.picks[..head_rows] == one.picks[..head_rows];
    let (rows, same) = decided(&one.logits, &one.picks, &two.picks);
    let worst = (0..two.picks.len())
        .map(|r| {
            err(
                &two.logits[r * VOCAB..(r + 1) * VOCAB],
                &one.logits[r * VOCAB..(r + 1) * VOCAB],
            )
            .rel_rms
        })
        .fold(0f64, f64::max);
    eprintln!(
        "six requests, layers 0-4 and the head, local FP8 experts: 3 decode steps (lanes of 3 + 3), \
         a verify round of {WINDOWS:?} rows kept {KEEP:?} (lanes cut after request {}), 2 decode \
         steps from device ids: two lanes against two passes, logits and picks bit for bit \
         {exact}, the slots' KDA states, conv windows, pages and tails {kept_exact}; with one call \
         in flight {depth1}; against one pass: the prompts and the decode steps before the verify \
         bit for bit {before}, every row's logits within {worst:.2e} (relative RMS, worst row), \
         picks equal on {same}/{rows} rows decided by {TIE}",
        balanced_cut(&WINDOWS)
    );
    assert!(exact, "two lanes differ from two passes");
    assert!(kept_exact, "the slots' state differs from two passes'");
    assert!(depth1, "one call in flight changed the bits");
    assert!(
        before,
        "rows of passes of at most 8 rows differ from one pass"
    );
    assert_eq!(same, rows, "two lanes pick other tokens than one pass");
    assert!(worst < 5e-2, "two lanes against one pass: {worst:.3e}");

    // The bounds: under decode_lane_rows, over decode_lane_max_rows, one request.
    let mut kvs: Vec<GlmKv> = (0..6)
        .map(|i| {
            let mut kv = fwd.kv.slot().unwrap();
            kv.reserve(16).unwrap();
            fwd.prefill(&mut [(&mut kv, &ids(40 + i, 5)[..])]).unwrap();
            kv
        })
        .collect();
    let n0 = fwd.decode_lane_passes();
    let step = |fwd: &mut GlmForward, kvs: &mut [GlmKv]| {
        let mut rows: Vec<(&mut GlmKv, u32)> = kvs.iter_mut().map(|k| (k, 1234)).collect();
        fwd.decode(&mut rows).unwrap();
    };
    fwd.cfg.decode_lane_rows = 7;
    step(&mut fwd, &mut kvs);
    fwd.cfg.decode_lane_rows = 2;
    fwd.cfg.decode_lane_max_rows = 5;
    step(&mut fwd, &mut kvs);
    fwd.cfg.decode_lane_max_rows = usize::MAX;
    step(&mut fwd, &mut kvs[..1]);
    assert_eq!(
        fwd.decode_lane_passes(),
        n0,
        "passes outside the bounds ran in two lanes"
    );
    step(&mut fwd, &mut kvs);
    assert_eq!(fwd.decode_lane_passes(), n0 + 1);
}

/// Every stored ring row of two slots' contexts (positions `lo .. len`, the drafter's five
/// layers, keys and values) equal bit for bit, and their lengths and lowest positions.
fn same_context(fwd: &GlmForward, a: &GlmKv, b: &GlmKv) -> bool {
    let (x, y) = (a.draft_slot().unwrap(), b.draft_slot().unwrap());
    if (x.len(), x.lo(), x.context_rows()) != (y.len(), y.lo(), y.context_rows()) {
        return false;
    }
    let d = fwd.drafter().unwrap();
    (0..5).all(|l| {
        (x.lo()..x.len()).all(|p| d.ring_row(a, l, p).unwrap() == d.ring_row(b, l, p).unwrap())
    })
}

fn greedy_draft(fwd: &mut GlmForward, kv: &GlmKv, anchor: u32) -> glm53f_dflash::seam::Proposal {
    fwd.draft(&[DraftReq {
        kv,
        anchor,
        temperature: 0.0,
        uniforms: &[],
    }])
    .unwrap()
    .remove(0)
}

#[test]
fn the_drafter_in_two_decode_lanes() {
    let cfg = ForwardConfig {
        max_rows: 128,
        lanes: 2,
        min_lane_rows: 8,
        decode_lane_rows: 2,
        max_verify_rows: 32,
        max_requests: 4,
        ..ForwardConfig::default()
    };
    let Some(mut fwd) = drafting::drafted_forward(cfg, 8, 8, 64, 0.25) else {
        return;
    };
    // Routed outputs of zeros: the local experts' cache loads experts on demand, which would
    // hide the forward's own allocations (none) among its.
    fwd.set_experts(Box::new(ZeroExperts));
    let rows = fwd.pass_rows();
    let (tap_bytes, scratch) = fwd
        .drafter_mut()
        .unwrap()
        .reserve(rows, cfg.max_requests)
        .unwrap();
    let prompts = [ids(1, 20), ids(2, 33), ids(3, 12), ids(4, 45)];
    let windows = [ids(5, 8), ids(6, 4), ids(7, 6), ids(8, 2)];
    let keep = [3usize, 4, 1, 2];
    let run = |fwd: &mut GlmForward, how: How| -> (Outcome, Vec<GlmKv>) {
        let mut kvs: Vec<GlmKv> = prompts
            .iter()
            .map(|p| {
                let mut kv = fwd.kv.slot().unwrap();
                kv.reserve(p.len() + 16).unwrap();
                kv
            })
            .collect();
        let mut o = Outcome {
            picks: Vec::new(),
            logits: Vec::new(),
            kept: Vec::new(),
        };
        for (kv, p) in kvs.iter_mut().zip(&prompts) {
            o.picks.extend(fwd.prefill(&mut [(kv, &p[..])]).unwrap());
        }
        verify_round(fwd, &mut kvs, &windows, &keep, how, &mut o);
        for s in 0..2 {
            decode_step(fwd, &mut kvs, &ids(20 + s, 4), how, &mut o);
        }
        (o, kvs)
    };
    let allocs = device::allocations();
    let n0 = fwd.decode_lane_passes();
    let (two, kv2) = run(&mut fwd, How::Lanes);
    assert_eq!(
        fwd.decode_lane_passes() - n0,
        3,
        "the verify and 2 decode passes in two lanes"
    );
    fwd.cfg.decode_lane_rows = 0;
    let (seq, kv1) = run(&mut fwd, How::TwoPasses);
    let exact = bits(&two.logits, &seq.logits) && two.picks == seq.picks;
    let rings = kv2.iter().zip(&kv1).all(|(a, b)| same_context(&fwd, a, b));
    let drafts = kv2
        .iter()
        .zip(&kv1)
        .all(|(a, b)| greedy_draft(&mut fwd, a, 777) == greedy_draft(&mut fwd, b, 777));
    let lens: Vec<usize> = kv2.iter().map(|k| k.draft_slot().unwrap().len()).collect();
    eprintln!(
        "the drafter in two decode lanes, prompts of {:?} tokens: a verify round of {:?} rows \
         kept {keep:?} and 2 decode steps in two lanes against two passes each: picks and logits \
         {exact}, contexts and rings {rings} (lengths {lens:?}), greedy drafts {drafts}",
        prompts.iter().map(|p| p.len()).collect::<Vec<_>>(),
        windows.iter().map(|w| w.len()).collect::<Vec<_>>(),
    );
    assert!(exact, "two lanes pick other tokens than two passes");
    assert!(rings, "the drafter's contexts differ from two passes'");
    assert!(drafts, "the drafts differ from two passes'");
    let d = fwd.drafter().unwrap();
    assert_eq!(
        device::allocations(),
        allocs,
        "a pass allocated device memory"
    );
    assert_eq!(
        d.scratch_bytes(),
        scratch,
        "the drafter's working memory grew"
    );
    assert_eq!(d.tap_bytes(), tap_bytes);
}

/// The four rank daemons on loopback with their peer mesh, stopped when dropped.
#[cfg(feature = "coordinator")]
struct Daemons(Vec<std::process::Child>);

#[cfg(feature = "coordinator")]
impl Drop for Daemons {
    fn drop(&mut self) {
        for c in &mut self.0 {
            let _ = c.kill();
            let _ = c.wait();
        }
    }
}

/// `GLM53F_RANK_BIN` and the four `GLM53F_RANK_DIRS` started on loopback (`--peers` for the
/// ranks' mesh, `GLM53F_WIRE_ALLOW_LAN=1`, tests only): the daemons and their addresses in rank
/// order, or None (printed).
#[cfg(feature = "coordinator")]
fn spawn_ranks() -> Option<(Daemons, Vec<String>)> {
    use std::io::{BufRead, BufReader};
    use std::process::{Command, Stdio};
    let Some(bin) = env_dir(&["GLM53F_RANK_BIN"]).filter(|p| p.is_file()) else {
        eprintln!("skip: GLM53F_RANK_BIN does not name a glm53f-rank binary");
        return None;
    };
    let dirs: Vec<std::path::PathBuf> = std::env::var("GLM53F_RANK_DIRS")
        .unwrap_or_default()
        .split(',')
        .filter(|s| !s.is_empty())
        .map(std::path::PathBuf::from)
        .collect();
    if dirs.len() != 4 || !dirs.iter().all(|d| d.join("manifest.txt").is_file()) {
        eprintln!("skip: GLM53F_RANK_DIRS must name the four rank directories (with manifest.txt)");
        return None;
    }
    // Four free loopback ports for the peer mesh (bound, noted, released).
    let peers: Vec<String> = (0..4)
        .map(|_| {
            std::net::TcpListener::bind("127.0.0.1:0")
                .unwrap()
                .local_addr()
                .unwrap()
                .to_string()
        })
        .collect();
    let mut d = Daemons(Vec::new());
    let mut addrs = Vec::new();
    for (r, dir) in dirs.iter().enumerate() {
        let mut child = Command::new(&bin)
            .args(["serve", "--rank", &r.to_string(), "--dir"])
            .arg(dir)
            .args(["--listen", "127.0.0.1:0", "--allow-partial"])
            .args(["--peers", &peers.join(",")])
            .env("GLM53F_WIRE_ALLOW_LAN", "1")
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .spawn()
            .expect("spawn glm53f-rank");
        let mut out = BufReader::new(child.stdout.take().unwrap());
        d.0.push(child);
        let mut line = String::new();
        let addr = loop {
            line.clear();
            if out.read_line(&mut line).unwrap_or(0) == 0 {
                panic!("rank {r} exited before listening (see its log above)");
            }
            if let Some(rest) = line.strip_prefix("listening on ") {
                break rest.split(' ').next().unwrap().to_string();
            }
        };
        addrs.push(addr);
        // Keep draining stdout so the daemon never blocks on it.
        std::thread::spawn(move || for _ in out.lines() {});
    }
    Some((d, addrs))
}

#[cfg(feature = "coordinator")]
#[test]
fn decode_lanes_on_four_rank_daemons() {
    use glm53f_coordinator::wire::{ReturnPath, WireConfig};
    use glm53f_forward::remote::{FastPaths, RemoteExperts};
    use glm53f_wire::row_shard::ExchangeDtype;
    if !gpu_with(13.0) {
        return;
    }
    let Some((daemons, addrs)) = spawn_ranks() else {
        return;
    };
    let cfg = ForwardConfig {
        max_rows: 64,
        lanes: 2,
        min_lane_rows: 8,
        decode_lane_rows: 2,
        max_verify_rows: 32,
        max_requests: 8,
        ..ForwardConfig::default()
    };
    let four = WireConfig::glm53_flash();
    // Row slices from 4 rows: the decode lanes' exchanges of 3 rows return four planes, the verify
    // lanes' of 11 and 14 rows row slices (each exchange in flight keeps its own path).
    let sliced = WireConfig {
        return_path: ReturnPath::RowSharded {
            min_rows: 4,
            exchange: ExchangeDtype::Bf16,
        },
        ..four
    };
    // The frame-fill kernel for every request (the device routes), instead of the copies.
    let fill = FastPaths {
        fill_rows: 64,
        ..FastPaths::ON
    };
    let runs = [
        ("four planes, the default paths", four, FastPaths::ON),
        ("row slices from 4 rows", sliced, FastPaths::ON),
        ("four planes, frame-fill requests", four, fill),
    ];
    let mut failed = Vec::new();
    for (what, wire, fast) in runs {
        let remote =
            RemoteExperts::connect_with(&addrs, 64, wire, fast).expect("connect to the ranks");
        let depth = remote.depth();
        let Some(mut fwd) = forward_with(LAYERS, cfg, |_| Box::new(remote), 12, 96, 16) else {
            return;
        };
        fwd.set_lane_trace(true, true);
        let two = script(&mut fwd, How::Lanes);
        fwd.cfg.decode_lane_rows = 0;
        let seq = script(&mut fwd, How::TwoPasses);
        drop(fwd);
        let exact = bits(&two.logits, &seq.logits) && two.picks == seq.picks;
        let kept_exact = two.kept == seq.kept;
        eprintln!(
            "six requests on four rank daemons ({what}; {depth} exchange in flight), the script \
             of decode_and_verify_in_two_lanes: two lanes against two passes, logits and picks \
             bit for bit {exact}, the slots' state {kept_exact}"
        );
        if !(exact && kept_exact) {
            failed.push(what);
        }
    }
    drop(daemons);
    assert!(
        failed.is_empty(),
        "two lanes differ from two passes on the ranks: {failed:?}"
    );
}
