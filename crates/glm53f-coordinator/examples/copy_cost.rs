//! The copy index's host cost (`glm53f_coordinator::copy`): what a speculative step pays for copy
//! windows on the scheduler's thread.
//!
//! ```text
//! cargo run --release -p glm53f-coordinator --example copy_cost
//! ```
//!
//! - **Indexing:** host time per context token while the index catches up with a context (a
//!   prompt the request's first steps index, [`CATCH_UP`] positions a step), and the steps that
//!   takes.
//! - **A step when nothing matches:** fresh text (uniform random ids, so no 8-token gram repeats):
//!   per request, the step's new tokens are indexed and the tail's gram is looked up and missed.
//! - **A step when everything matches:** a context of one 4,096-token block repeated, so every
//!   tail has the most earlier occurrences the walk visits ([`CANDIDATES`]), each matching back to
//!   the [`EXTEND`] limit: the lookup's worst case.
//!
//! Each step appends 8 tokens per request (a whole window delivered) and proposes up to 7; the
//! per-step figure is the median over 2,000 steps of every request's proposal together, for 1
//! and 48 requests at each context length.

use std::time::Instant;

use glm53f_coordinator::copy::{CopyIndex, CANDIDATES, CATCH_UP, EXTEND};
use glm53f_coordinator::Token;

/// GLM-5.3-Flash's tokenizer ids.
const BOUND: usize = 154_856;
const STEPS: usize = 2000;

fn rng(seed: u64) -> impl FnMut() -> u64 {
    let mut s = seed;
    move || {
        s = s.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut x = s;
        x = (x ^ (x >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        x = (x ^ (x >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        x ^ (x >> 31)
    }
}

fn fresh(n: usize, seed: u64) -> Vec<Token> {
    let mut r = rng(seed);
    (0..n).map(|_| (r() % BOUND as u64) as Token).collect()
}

fn repeated(n: usize, seed: u64) -> Vec<Token> {
    let block = fresh(4096, seed);
    block.iter().cycle().take(n).copied().collect()
}

/// Index `ctx` from scratch as the scheduler would (one call a step until caught up): the calls it
/// took, the host time per token, and the longest call (ms).
fn build(c: &mut CopyIndex, ctx: &[Token]) -> (usize, f64, f64) {
    let t0 = Instant::now();
    let (mut calls, mut longest) = (0, 0f64);
    while c.indexed() + 1 < ctx.len() {
        let t = Instant::now();
        c.propose(ctx, 7, BOUND);
        longest = longest.max(t.elapsed().as_secs_f64() * 1e3);
        calls += 1;
    }
    (calls, t0.elapsed().as_secs_f64() * 1e9 / ctx.len() as f64, longest)
}

/// The `q` quantile of `v` (sorted in place).
fn quantile(v: &mut [f64], q: f64) -> f64 {
    v.sort_by(f64::total_cmp);
    v[((v.len() - 1) as f64 * q).round() as usize]
}

/// `requests` requests at `len` tokens each: the index build (with room reserved for the context and
/// the steps' tokens, as the scheduler reserves, or without), then STEPS steps. Returns the steps'
/// host times (all requests together, us) and the copies proposed.
fn run(requests: usize, len: usize, kind: &str, reserve: bool) -> (Vec<f64>, usize) {
    let mut ctxs: Vec<Vec<Token>> = (0..requests)
        .map(|i| if kind == "fresh" { fresh(len, i as u64) } else { repeated(len, i as u64) })
        .collect();
    let mut idx: Vec<CopyIndex> = (0..requests)
        .map(|_| {
            let mut c = CopyIndex::default();
            if reserve {
                c.reserve(len + 8 * STEPS);
            }
            c
        })
        .collect();
    let (calls, ns, longest) = build(&mut idx[0], &ctxs[0]);
    for (c, x) in idx.iter_mut().zip(&ctxs).skip(1) {
        build(c, x);
    }
    let mut more: Vec<Box<dyn FnMut() -> Token>> = (0..requests)
        .map(|i| {
            let mut r = rng(1000 + i as u64);
            let block = ctxs[i][..4096.min(len)].to_vec();
            let mut at = len;
            let f: Box<dyn FnMut() -> Token> = if kind == "fresh" {
                Box::new(move || (r() % BOUND as u64) as Token)
            } else {
                Box::new(move || {
                    let t = block[at % block.len()];
                    at += 1;
                    t
                })
            };
            f
        })
        .collect();
    let (mut times, mut copies) = (Vec::with_capacity(STEPS), 0usize);
    for _ in 0..STEPS {
        for (x, m) in ctxs.iter_mut().zip(more.iter_mut()) {
            for _ in 0..8 {
                x.push(m());
            }
        }
        let t0 = Instant::now();
        for (c, x) in idx.iter_mut().zip(&ctxs) {
            copies += usize::from(!c.propose(x, 7, BOUND).is_empty());
        }
        times.push(t0.elapsed().as_secs_f64() * 1e6);
    }
    if requests == 1 {
        println!(
            "  index build, {len} tokens{}: {ns:.1} ns a token ({:.2} ms in all, {bytes:.1} bytes a token held), \
             {calls} step(s), the longest {longest:.2} ms",
            if reserve { "" } else { " (no room reserved)" },
            ns * len as f64 / 1e6,
            bytes = idx[0].bytes() as f64 / len as f64
        );
    }
    (times, copies)
}

fn main() {
    println!(
        "copy index: entry 8 tokens, up to {CANDIDATES} candidates matched back up to {EXTEND}, \
         {CATCH_UP} positions indexed a step"
    );
    for kind in ["fresh", "repeated"] {
        println!("{kind} context ({}):", if kind == "fresh" { "nothing matches" } else { "every tail matches" });
        for (requests, len) in [(1, 1024), (1, 65_536), (1, 1_048_576), (48, 1024), (48, 21_845), (48, 1_048_576)] {
            for reserve in [true, false] {
                let (mut t, copies) = run(requests, len, kind, reserve);
                let (p50, p99, p999, max) =
                    (quantile(&mut t, 0.5), quantile(&mut t, 0.99), quantile(&mut t, 0.999), quantile(&mut t, 1.0));
                println!(
                    "  {requests:2} request(s) x {len:>9} tokens{}: a step {p50:.2} us median ({:.2} us a request), \
                     p99 {p99:.1} us, p99.9 {p999:.1} us, largest {max:.1} us; {copies} copies in {STEPS} steps",
                    if reserve { "" } else { ", no room reserved" },
                    p50 / requests as f64
                );
            }
        }
    }
}
