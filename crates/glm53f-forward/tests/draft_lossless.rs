//! Speculative decoding with the DFlash2 drafter never changes a token (feature `coordinator`;
//! the coordinator's weights, the experts of layers 3 and 4 and the drafter's checkpoint; skips
//! without them). The shell's scheduler runs the same requests over one forward of all 45
//! decoder layers (`drafting`: garbage text, which is enough, since the property holds for any
//! weights): plain (the model's block hidden, one token a step) and speculative (the drafter's
//! block of 8, every draft verified: `SpecPolicy::Fixed`). The token sequences must be equal:
//!
//! - greedy requests one at a time, for prompts of 5 to 150 tokens (a two-chunk prefill) and a
//!   low-entropy prompt;
//! - two greedy requests at once, drafting 3 a step, so a verify pass holds at most 8 rows;
//! - a sampled request with a fixed seed (every emitted token is the target's own draw), twice;
//! - a second turn that resumes the first's retained slot (the drafter's context kept in place).
//!
//! On these weights the target rejects nearly every draft of the real drafter, so the greedy
//! cases, the pair and the sampled request run once more with the drafter still drafting but its
//! proposals replaced by the plain run's own continuation, one draft in some windows made wrong:
//! whole and partial accepts, commits of up to 8 rows and of two windows at once, give the plain
//! tokens too.
//!
//! Exactness rests on the forward's row independence up to 8 rows (`tests/verify_commit.rs`): a
//! verify window then gives the bits of serial steps. A step whose verify pass holds more than 8
//! rows (several requests with long windows) runs the tensor-core path, as a plain decode batch
//! of more than 8 rows does.
//!
//! The acceptance printed for the real drafter measures nothing about quality: repeated layer
//! weights and zero experts for 40 of 42 MoE layers.
#![cfg(feature = "coordinator")]

mod common;
mod drafting;

use std::sync::mpsc;

use drafting::*;
use glm53f_coordinator::model::{
    DecodeRow, Draft, DraftRow, Limits, ModelForward, Pick, Segment, SegmentOut, Token, Window,
};
use glm53f_coordinator::sampling::Sampling;
use glm53f_coordinator::scheduler::{Job, SchedStats, Scheduler, SchedulerConfig};
use glm53f_coordinator::spec::SpecPolicy;
use glm53f_forward::forward::ForwardConfig;
use glm53f_forward::kv::GlmKv;
use glm53f_forward::serve::{DraftStats, ServedForward};
use glm53f_forward::shape::SAMPLE_VOCAB;

/// Drafts that are the plain run's own tokens: a request with a prompt of `plen` tokens whose
/// plain output is `out`.
#[derive(Clone)]
struct Oracle {
    plen: usize,
    out: Vec<Token>,
}

impl Oracle {
    /// Where a slot holding `tokens` rows with `last` next stands in this request's output: the
    /// index of the token after `last`.
    fn place(&self, tokens: usize, last: Token) -> Option<usize> {
        let g = (tokens + 1).checked_sub(self.plen)?;
        (g >= 1 && g <= self.out.len() && self.out[g - 1] == last).then_some(g)
    }
}

/// The served forward, with its block hidden (plain decoding) or not, and its drafts replaced by
/// the matching [`Oracle`]'s when there are oracles.
struct Lens<'a> {
    m: &'a mut ServedForward,
    spec: bool,
    oracles: Vec<Oracle>,
}

impl ModelForward for Lens<'_> {
    type Slot = GlmKv;

    fn limits(&self) -> Limits {
        let mut l = self.m.limits();
        if !self.spec {
            l.block = 0;
        }
        l
    }
    fn free_bytes(&self) -> Result<usize, String> {
        self.m.free_bytes()
    }
    fn prefill(&mut self, segs: &mut [Segment<'_, GlmKv>]) -> Result<Vec<SegmentOut>, String> {
        self.m.prefill(segs)
    }
    fn decode(&mut self, rows: &mut [DecodeRow<'_, GlmKv>]) -> Result<Vec<Token>, String> {
        self.m.decode(rows)
    }
    fn draft(&mut self, rows: &mut [DraftRow<'_, GlmKv>]) -> Result<Vec<Draft>, String> {
        let mut drafts = self.m.draft(rows)?;
        if !self.oracles.is_empty() {
            for (r, d) in rows.iter().zip(drafts.iter_mut()) {
                // The slot holds the prompt and all but the last emitted token.
                let found: Vec<(&Oracle, usize)> = self
                    .oracles
                    .iter()
                    .filter_map(|o| o.place(r.slot.tokens(), r.last).map(|g| (o, g)))
                    .collect();
                assert_eq!(found.len(), 1, "the oracles lost their place");
                let (o, g) = found[0];
                let mut t: Vec<Token> = o.out[g..].iter().take(r.max).copied().collect();
                // One wrong draft in some windows (g = 1, 6, 8, 16, ...: 4, 1, none, 4, ...):
                // partial and whole accepts.
                let wrong = (g * 3 + 1) % 9;
                if wrong < t.len() {
                    t[wrong] = (t[wrong] + 1) % SAMPLE_VOCAB as Token;
                }
                *d = Draft {
                    probs: vec![0.9; t.len()],
                    tokens: t,
                };
            }
        }
        Ok(drafts)
    }
    fn verify(&mut self, windows: &mut [Window<'_, GlmKv>]) -> Result<Vec<Vec<Token>>, String> {
        self.m.verify(windows)
    }
    fn commit(&mut self, slots: &mut [&mut GlmKv], keep: &[usize]) -> Result<(), String> {
        self.m.commit(slots, keep)
    }
    fn select_host(&mut self, logits: &[f32], pick: &Pick) -> Result<Token, String> {
        self.m.select_host(logits, pick)
    }
}

/// A request: prompt, token budget, sampling.
type Req = (Vec<Token>, usize, Option<Sampling>);

/// Run `waves` of requests through a scheduler over `m` (each wave submitted together and run
/// until idle, the next wave after it); every request's tokens, and the scheduler's counters.
fn run(
    m: &mut ServedForward,
    spec: bool,
    oracles: Vec<Oracle>,
    min_retain: usize,
    waves: &[Vec<Req>],
) -> (Vec<Vec<Token>>, SchedStats) {
    let slots: Vec<GlmKv> = (0..3).map(|_| m.fwd.kv.slot().unwrap()).collect();
    let mut cfg = SchedulerConfig::new(Vec::new());
    cfg.min_retain = min_retain;
    cfg.out_min = 16;
    cfg.policy = SpecPolicy::Fixed;
    let (tx, rx) = mpsc::channel();
    let mut sched = Scheduler::new(Lens { m, spec, oracles }, slots, None, cfg, rx);
    let mut out = Vec::new();
    for wave in waves {
        let rxs: Vec<_> = wave
            .iter()
            .map(|(p, max, s)| {
                let (job, r) = Job::new(p.clone(), *max, *s);
                tx.send(job).unwrap();
                r
            })
            .collect();
        let mut idle = 0;
        for _ in 0..100_000 {
            assert!(sched.step(false));
            idle = if sched.is_idle() { idle + 1 } else { 0 };
            if idle >= 3 {
                break;
            }
        }
        assert!(idle >= 3, "the scheduler did not go idle");
        for r in rxs {
            out.push(
                r.try_iter()
                    .collect::<Result<Vec<Token>, String>>()
                    .unwrap(),
            );
        }
    }
    (out, sched.stats)
}

fn delta(a: DraftStats, b: DraftStats) -> DraftStats {
    DraftStats {
        drafted: b.drafted - a.drafted,
        cold: b.cold - a.cold,
        proposed: b.proposed - a.proposed,
        rounds: b.rounds - a.rounds,
        windows: b.windows - a.windows,
        verified: b.verified - a.verified,
        kept: b.kept - a.kept,
    }
}

fn add(a: DraftStats, b: DraftStats) -> DraftStats {
    DraftStats {
        drafted: a.drafted + b.drafted,
        cold: a.cold + b.cold,
        proposed: a.proposed + b.proposed,
        rounds: a.rounds + b.rounds,
        windows: a.windows + b.windows,
        verified: a.verified + b.verified,
        kept: a.kept + b.kept,
    }
}

fn pct(a: u64, b: u64) -> f64 {
    100.0 * a as f64 / b.max(1) as f64
}

/// Plain and speculative runs of `waves`: equal tokens. Returns the tokens, the speculative
/// run's scheduler counters and the drafter's.
fn same(
    m: &mut ServedForward,
    what: &str,
    min_retain: usize,
    waves: &[Vec<Req>],
) -> (Vec<Vec<Token>>, SchedStats, DraftStats) {
    let t0 = std::time::Instant::now();
    let (plain, ps) = run(m, false, Vec::new(), min_retain, waves);
    let t1 = std::time::Instant::now();
    let before = m.stats;
    let (spec, ss) = run(m, true, Vec::new(), min_retain, waves);
    let ds = delta(before, m.stats);
    let t2 = std::time::Instant::now();
    assert_eq!(ps.spec_steps, 0);
    assert!(ss.spec_steps > 0 && ss.decode_steps == 0, "{ss:?}");
    for (i, (p, s)) in plain.iter().zip(&spec).enumerate() {
        assert_eq!(
            p, s,
            "{what}: request {i}: speculative decoding changed the tokens"
        );
    }
    let distinct: std::collections::HashSet<_> = plain.iter().flatten().collect();
    assert!(distinct.len() > 3, "{what}: degenerate outputs {plain:?}");
    // The scheduler's counters and the drafter's agree (no request stopped inside a run).
    assert_eq!(
        (ss.drafts_verified, ss.drafts_accepted),
        (ds.verified, ds.kept)
    );
    let n: usize = plain.iter().map(|v| v.len()).sum();
    eprintln!(
        "{what}: {n} tokens equal; plain {} steps in {:.1} s, speculative {} steps in {:.1} s; \
         real drafts: {} verified, {} accepted ({:.1}%, meaningless weights), {:.2} tokens a window",
        ps.decode_steps,
        (t1 - t0).as_secs_f64(),
        ss.spec_steps,
        (t2 - t1).as_secs_f64(),
        ss.drafts_verified,
        ss.drafts_accepted,
        pct(ss.drafts_accepted, ss.drafts_verified),
        ds.tokens_per_window()
    );
    (plain, ss, ds)
}

/// The requests of `wave` with the plain run's tokens `want` as drafts (one wrong in some
/// windows): the same tokens; the scheduler's counters.
fn with_oracle(m: &mut ServedForward, wave: &[Req], want: &[Vec<Token>]) -> SchedStats {
    let oracles = wave
        .iter()
        .zip(want)
        .map(|(r, w)| Oracle {
            plen: r.0.len(),
            out: w.clone(),
        })
        .collect();
    let (got, ss) = run(m, true, oracles, usize::MAX, &[wave.to_vec()]);
    assert_eq!(got, want, "oracle drafts changed the tokens");
    ss
}

#[test]
fn speculative_decoding_changes_no_token() {
    let cfg = ForwardConfig {
        max_rows: 128,
        max_verify_rows: 16,
        max_requests: 3,
        ..ForwardConfig::default()
    };
    let Some(fwd) = drafted_forward(cfg, 3, 16, 48, 2.0) else {
        return;
    };
    let mut m = ServedForward::new(fwd).unwrap();
    assert_eq!(m.limits().block, 8);
    let never = usize::MAX;

    // Greedy requests one at a time: prompts of 5, 37 and 150 tokens (a two-chunk prefill), and
    // a low-entropy prompt (a 4-token pattern repeated).
    let pattern: Vec<Token> = ids(14, 4).iter().cycle().take(48).copied().collect();
    let singles: Vec<Vec<Req>> = vec![
        vec![(ids(11, 5), 48, None)],
        vec![(ids(12, 37), 40, None)],
        vec![(ids(13, 150), 32, None)],
        vec![(pattern, 48, None)],
    ];
    let (greedy_out, _, ds) = same(&mut m, "greedy, one request at a time", never, &singles);
    assert!(ds.proposed >= ds.verified && ds.windows == ds.rounds);
    // The real drafter's counters over its own runs (not the oracle runs below).
    let mut real = ds;

    // Two greedy requests at once, 3 drafts a step: verify passes of at most 8 rows.
    m.max_drafts = 3;
    let pair = vec![vec![(ids(21, 9), 32, None), (ids(22, 17), 32, None)]];
    let (pair_out, _, ds) = same(
        &mut m,
        "two greedy requests at once, 3 drafts a step",
        never,
        &pair,
    );
    assert!(ds.verified <= 3 * ds.windows, "windows of at most 3 drafts");
    real = add(real, ds);
    let ss = with_oracle(&mut m, &pair[0], &pair_out);
    let accepted = (ss.drafts_accepted, ss.drafts_verified);
    assert!(accepted.0 > 0 && accepted.0 < accepted.1, "{ss:?}");
    eprintln!(
        "two greedy requests at once with oracle drafts: tokens equal in {} steps, {} of {} drafts \
         accepted",
        ss.spec_steps, accepted.0, accepted.1
    );
    m.max_drafts = 7;

    // A sampled request with a fixed seed: the target's own draws, speculation or not, and the
    // same drafts every run.
    let s = Sampling::new(0.9, 0.95, 0, 0.0, Some(5)).unwrap();
    assert!(s.is_some());
    let sampled = vec![vec![(ids(31, 25), 40, s)]];
    let (sampled_out, _, d1) = same(&mut m, "a sampled request, seed 5", never, &sampled);
    let before = m.stats;
    let (again, _) = run(&mut m, true, Vec::new(), never, &sampled);
    let d2 = delta(before, m.stats);
    assert_eq!(sampled_out, again, "a seeded request twice");
    assert_eq!(d1, d2, "the same drafts and acceptance every run");
    real = add(add(real, d1), d2);

    // A second turn resumes the first turn's retained slot in place (a turn snapshot at 16+
    // tokens), its drafter context kept.
    let p1 = ids(41, 20);
    let (t1, _) = run(
        &mut m,
        false,
        Vec::new(),
        16,
        &[vec![(p1.clone(), 16, None)]],
    );
    let mut p2 = p1.clone();
    p2.extend(&t1[0][..15]);
    p2.extend(ids(42, 6));
    let turns = vec![vec![(p1, 16, None)], vec![(p2, 24, None)]];
    let (out, _, ds) = same(&mut m, "two turns, the second resumed in place", 16, &turns);
    assert_eq!(out[0], t1[0]);
    assert_eq!(ds.cold, 0, "the resumed turn's context stayed warm");
    real = add(real, ds);

    // Drafts the target accepts: the plain tokens, one wrong in some windows.
    let t0 = std::time::Instant::now();
    let before = m.stats;
    let mut total = SchedStats::default();
    let cases = singles
        .iter()
        .zip(greedy_out.chunks(1))
        .chain([(&sampled[0], &sampled_out[..])]);
    for (wave, want) in cases {
        let ss = with_oracle(&mut m, wave, want);
        total.spec_steps += ss.spec_steps;
        total.drafts_verified += ss.drafts_verified;
        total.drafts_accepted += ss.drafts_accepted;
    }
    let ds = delta(before, m.stats);
    assert_eq!(
        (ds.verified, ds.kept),
        (total.drafts_verified, total.drafts_accepted)
    );
    let tokens: usize = greedy_out.iter().map(|v| v.len()).sum::<usize>() + sampled_out[0].len();
    assert!(total.drafts_accepted > 0 && total.drafts_accepted < total.drafts_verified);
    assert!(ds.tokens_per_window() > 3.0, "{ds:?}");
    eprintln!(
        "oracle drafts (greedy and sampled): {tokens} tokens equal in {} steps, {:.1} s; {} drafts \
         verified, {} accepted ({:.1}%), {:.2} tokens a window",
        total.spec_steps,
        t0.elapsed().as_secs_f64(),
        total.drafts_verified,
        total.drafts_accepted,
        pct(total.drafts_accepted, total.drafts_verified),
        ds.tokens_per_window()
    );
    eprintln!(
        "the real drafter over its speculative runs (the oracle runs left out): {} rounds, {} \
         windows, {} drafts verified, {} kept ({:.1}%, meaningless weights), {:.2} tokens a \
         window; {} requests drafted, {} cold",
        real.rounds,
        real.windows,
        real.verified,
        real.kept,
        100.0 * real.acceptance(),
        real.tokens_per_window(),
        real.drafted,
        real.cold
    );
}
