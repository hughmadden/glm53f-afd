//! Prefill in lanes against one lane (feature `cuda`; real weights of layers 0-4 and the head;
//! skips without them). Each test runs 2, 3 and 4 lanes ([`MAX_LANES`]).
//!
//! **Exact.** Lane i's attention at a layer reads only what the lanes before it left at that
//! layer (the KDA states and conv windows, the MLA latents and pooled keys, a split request's DSA
//! tail), so a pass in N lanes must give the bits of the same rows run as N passes one after the
//! other: lane 0's rows, then lane 1's, and so on. The tests check that bit for bit (logits,
//! picks, every slot's state: KDA states, conv windows, MLA latents and pooled keys, DSA tails;
//! the decode steps after), at every depth of calls in flight.
//!
//! **Within rounding of one pass.** Against one pass over all the rows, the lanes change the row
//! counts of the tensor-core GEMMs (cuBLAS picks its kernels by the row count; the FP8 GEMM
//! activations are quantized per pass), as a prompt cut into chunks already does. The tests hold
//! the two to the same tokens on every row whose best two logits are at least [`TIE`] apart
//! (closer calls are near ties that rounding of this size flips: the oracle's own first row is a
//! 0.042 tie, and one pass and the lanes land on either side of it), to logits within the chain
//! test's bound against the oracle (`tests/goldens_chain.rs`: 5e-2 relative RMS per row), and the
//! lanes themselves to the goldens with that test's bounds.
//!
//! 1. **The oracle's prompt** (33 tokens: lanes of 17 and 16 rows, 3 of 11, 9 + 3 of 8) through
//!    layers 0-4 and the head, then the 8 fixed decode steps, with the local FP8 experts behind
//!    the golden routes: N lanes against N passes of the lanes' rows (bit for bit), against one
//!    pass, and both against the golden logits (the chain test's bounds; from three lanes, the
//!    argmax on the rows with a clear winner: lanes of 11 rows move both of the golden's near
//!    ties).
//! 2. **Two prompts batched across the cuts** (150 and 211 tokens: 2 lanes cut the second prompt
//!    31 rows in, inside an indexer pool of 4; 3 lanes cut the first at 121 and the second at 91;
//!    4 lanes the first at 91 and the second at 31 and 121, so one lane holds a middle part of
//!    it), routed experts returning zeros: N lanes against the N passes (bit for bit) and against
//!    one pass (picks, logits, the pooled index key at each cut, the DSA tails), then 4 decode
//!    steps each; the lanes repeat bit for bit; no pass allocates device memory; a segment longer
//!    than a pass runs in chunks, each in N lanes; the lane trace.
//! 3. **The drafter in lanes** (all 45 decoder layers with the DFlash2 drafter attached,
//!    `drafting`): a prompt of 50 tokens and two prompts of 20 and 33, prefilled in N lanes, leave
//!    the drafter's taps, contexts and rings bit for bit as the same rows in N one-lane passes,
//!    and draft the same; 3 decode steps after, still the same (routed experts of zeros). With the
//!    drafter's memory reserved first (`Dflash::reserve`), nothing allocates: the forward's
//!    counter and the drafter's working memory do not move.
//! 4. **Every depth.** Test 2's batch in N lanes with 1 to N calls in flight, routed experts that
//!    check the calls (at most the depth out, finished oldest first, each into its own lane's
//!    output) and write each row's FFN input as its routed output when finished, so a call
//!    finished late or into another lane's buffers changes the bits: against the N passes, bit
//!    for bit, and the depth reached; then four lanes with the chunked KDA prefill against four
//!    passes.
//!
//! ```sh
//! GLM53F_CHECKPOINT_DIR=... GLM53F_EXPERTS_DIR=... \
//!   cargo test --release -p glm53f-forward --features cuda --test lanes -- --nocapture --test-threads=1
//! ```
#![cfg(feature = "cuda")]

mod common;
mod drafting;

use std::collections::{HashMap, VecDeque};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use common::*;
use glm53f_dsa::cache::{
    decode_index_key, PAGE_POOL_CODES_OFFSET, PAGE_POOL_SCALES_OFFSET, PAGE_TOKENS,
};
use glm53f_forward::device::{self, Stream};
use glm53f_forward::draft::{DraftReq, TAP_WIDTH};
use glm53f_forward::experts::{ExpertBackend, ExpertCall, LocalFp8Experts, ZeroExperts};
use glm53f_forward::forward::{ForwardConfig, GlmForward, MAX_LANES};
use glm53f_forward::gemm::Fp8Act;
use glm53f_forward::kv::GlmKv;
use glm53f_forward::shape::{HIDDEN, SAMPLE_VOCAB, TOP_K, VOCAB};

const LAYERS: usize = 5;
const PROMPT: usize = 33;
const STEPS: usize = 8;

/// Logits of two runs of the same rows: the relative RMS of every value, the largest per-row
/// relative RMS, and the rows whose argmax (below the sampled vocabulary) agrees.
fn close(a: &[f32], b: &[f32]) -> (f64, f64, usize) {
    let rows = a.len() / VOCAB;
    let mut worst = 0f64;
    let mut agree = 0;
    for r in 0..rows {
        let (x, y) = (
            &a[r * VOCAB..(r + 1) * VOCAB],
            &b[r * VOCAB..(r + 1) * VOCAB],
        );
        worst = worst.max(err(x, y).rel_rms);
        agree += usize::from(argmax(x) == argmax(y));
    }
    (err(a, b).rel_rms, worst, agree)
}

/// Near ties: a row whose two best logits (below the sampled vocabulary) are closer than this
/// may pick either with rounding differences of the row counts' size (the oracle's own prompt has
/// a 0.042 gap on its first row, and one pass and the lanes land on either side of it).
const TIE: f32 = 0.25;

/// Rows of `reference` (logit rows) whose two best logits are at least [`TIE`] apart, and
/// whether `picks` agree with `other_picks` on each of them.
fn decided(reference: &[f32], picks: &[u32], other_picks: &[u32]) -> (usize, usize) {
    let (mut rows, mut agree) = (0, 0);
    for (r, (a, b)) in picks.iter().zip(other_picks).enumerate() {
        let v = &reference[r * VOCAB..r * VOCAB + SAMPLE_VOCAB];
        let best = argmax(v);
        let second = (0..SAMPLE_VOCAB)
            .filter(|&i| i != best)
            .map(|i| v[i])
            .fold(f32::NEG_INFINITY, f32::max);
        if v[best] - second >= TIE {
            rows += 1;
            agree += usize::from(a == b);
        }
    }
    (rows, agree)
}

/// The rows whose pick differs from `reference`'s argmax, with `reference`'s gap between its two
/// best logits there.
fn misses(reference: &[f32], picks: &[u32]) -> Vec<(usize, f32)> {
    picks
        .iter()
        .enumerate()
        .filter_map(|(r, &p)| {
            let v = &reference[r * VOCAB..r * VOCAB + SAMPLE_VOCAB];
            let best = argmax(v);
            let second = (0..SAMPLE_VOCAB)
                .filter(|&i| i != best)
                .map(|i| v[i])
                .fold(f32::NEG_INFINITY, f32::max);
            (p as usize != best).then_some((r, v[best] - second))
        })
        .collect()
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

fn bits(a: &[f32], b: &[f32]) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits())
}

/// Golden routes of layers 3 and 4 for `runs` runs of the prompt then the 8 steps.
fn golden_routes(g: &Goldens, runs: usize) -> RouteQueue {
    let mut q = HashMap::new();
    for l in [3usize, 4] {
        let (sp, sd) = (
            format!("layer{l:02}-prefill"),
            format!("layer{l:02}-decode"),
        );
        let ids =
            |s: &str, n: &str| -> Vec<i32> { g.i64(s, n).into_iter().map(|x| x as i32).collect() };
        let (pi, pw) = (
            ids(&sp, "prefill.moe.topk_ids"),
            g.f32(&sp, "prefill.moe.topk_weights"),
        );
        let (di, dw) = (
            ids(&sd, "decode.moe.topk_ids"),
            g.f32(&sd, "decode.moe.topk_weights"),
        );
        let mut d = VecDeque::new();
        for _ in 0..runs {
            d.push_back((pi.clone(), pw.clone()));
            for s in 0..STEPS {
                d.push_back((
                    di[s * TOP_K..(s + 1) * TOP_K].to_vec(),
                    dw[s * TOP_K..(s + 1) * TOP_K].to_vec(),
                ));
            }
        }
        q.insert(l, d);
    }
    q
}

/// The prompt in one prefill (or with `cuts`, in one-lane passes cut at those rows), then the
/// steps: every row's logits (the prompt's last row, then each step) and picks, and what the slot
/// keeps ([`kept`]).
fn chain(
    fwd: &mut GlmForward,
    prompt: &[u32],
    steps: &[u32],
    cuts: &[usize],
) -> (Vec<f32>, Vec<u32>, Vec<u8>) {
    let mut kv = [fwd.kv.slot().unwrap()];
    kv[0].reserve(prompt.len() + steps.len()).unwrap();
    let (mut picks, mut logits) = prefill_in_passes(fwd, &mut kv, &[prompt], cuts);
    for &t in steps {
        picks.extend(fwd.decode(&mut [(&mut kv[0], t)]).unwrap());
        logits.extend(fwd.logits(1).unwrap());
    }
    let state = kept(fwd, &kv[0]);
    (logits, picks, state)
}

#[test]
fn the_oracles_prompt_in_lanes() {
    let names = [
        "layer03-prefill",
        "layer03-decode",
        "layer04-prefill",
        "layer04-decode",
        "layer00-prefill",
        "head",
    ];
    let Some(g) = Goldens::load(&names) else {
        return;
    };
    if !gpu_with(9.0) {
        return;
    }
    // Lanes of up to 33 rows, so one lane also holds the whole prompt.
    let cfg = ForwardConfig {
        max_rows: MAX_LANES * PROMPT,
        lanes: MAX_LANES,
        min_lane_rows: 8,
        max_verify_rows: 8,
        max_requests: 4,
        ..ForwardConfig::default()
    };
    let Some(edir) = experts_dir() else {
        return;
    };
    // One pass, then each lane count's lanes and passes.
    let routes = golden_routes(&g, 1 + 2 * (MAX_LANES - 1));
    let Some(mut fwd) = forward_with(
        LAYERS,
        cfg,
        |st| {
            let local =
                LocalFp8Experts::new(&edir, 3 << 30, cfg.max_rows, st, Fp8Act::Bf16).unwrap();
            let mut gr = GoldenRoutes::new(local, cfg.max_rows);
            gr.queue = routes;
            Box::new(gr)
        },
        4,
        64,
        16,
    ) else {
        return;
    };
    let (prompt, steps) = g.token_ids("layer00-prefill");
    assert_eq!((prompt.len(), steps.len()), (PROMPT, STEPS));
    let gl = g.f32("head", "head.logits");
    let golden_picks: Vec<u32> = (0..=STEPS)
        .map(|r| argmax(&gl[r * VOCAB..(r + 1) * VOCAB]) as u32)
        .collect();

    fwd.set_lane_trace(true, false);
    fwd.cfg.lanes = 1;
    let one = chain(&mut fwd, &prompt, &steps, &[]);
    assert!(
        fwd.take_lane_trace()
            .is_some_and(|t| t.rows == vec![PROMPT]),
        "one lane"
    );
    let e1 = err(&one.0, &gl).rel_rms;
    let a1 = close(&one.0, &gl).2;
    for n in 2..=MAX_LANES {
        fwd.cfg.lanes = n;
        let lanes = chain(&mut fwd, &prompt, &steps, &[]);
        let trace = fwd.take_lane_trace().expect("a traced prefill");
        assert_eq!(
            trace.rows,
            lane_rows(PROMPT, n),
            "the prompt ran in {n} lanes"
        );
        fwd.cfg.lanes = 1;
        let seq = chain(&mut fwd, &prompt, &steps, &lane_cuts(PROMPT, n));
        let exact = lanes.1 == seq.1 && bits(&lanes.0, &seq.0) && lanes.2 == seq.2;

        let en = err(&lanes.0, &gl).rel_rms;
        let an = close(&lanes.0, &gl).2;
        let (rel, worst, agree) = close(&lanes.0, &one.0);
        let same = lanes.1.iter().zip(&one.1).filter(|(x, y)| x == y).count();
        let (rows, decided_same) = decided(&one.0, &one.1, &lanes.1);
        eprintln!(
            "the oracle's prompt, layers 0-4, golden routes, local FP8 experts: {n} lanes of {:?} \
             rows against {n} passes of those rows bit for bit (logits, picks, the slot's KDA \
             states, conv windows, pages and tails): {exact}; logits against the \
             golden, relative RMS {en:.3e} ({n} lanes) / {e1:.3e} (one pass), argmax {an}/9 / \
             {a1}/9; {n} lanes against one pass: relative RMS {rel:.3e} (worst row {worst:.3e}), \
             argmax {agree}/9, picks {same}/9, and {decided_same}/{rows} of the rows whose best \
             two logits are at least {TIE} apart; the rows whose pick is not the golden's (row, \
             the golden's top-2 gap): {:?}",
            trace.rows,
            misses(&gl, &lanes.1)
        );
        assert!(exact, "{n} lanes differ from {n} passes of the same rows");
        // The chain test's bounds against the oracle (tests/goldens_chain.rs): logits within 5e-2
        // and the argmax on 8 of the 9 rows, or with numerics under test (GLM53F_TEST_NUMERICS) on
        // every row whose golden top-2 gap is at least TIE, as there. The golden's rows 0 and 2
        // are near ties (gaps 0.042 and 0.006) that lanes of 11 rows and fewer both move (three
        // lanes: 7/9, measured), so from three lanes the bound is the rows with a clear winner.
        let clear = [
            decided(&gl, &golden_picks, &lanes.1),
            decided(&gl, &golden_picks, &one.1),
        ];
        let eight = [n == 2, true];
        for (((e, a), (rows, same)), eight) in
            [(en, an), (e1, a1)].into_iter().zip(clear).zip(eight)
        {
            assert!(e < 5e-2, "against the golden: {e:.3e}");
            if numerics() == TestNumerics::default() && eight {
                assert!(a >= 8, "against the golden: {a}/9");
            } else {
                assert_eq!(
                    same, rows,
                    "against the golden, the rows with a clear winner"
                );
            }
        }
        // The lanes against one pass: the same tokens but on near ties, logits within the same
        // bound.
        assert!(rows >= 7, "{rows} rows decided");
        assert_eq!(
            decided_same, rows,
            "{n} lanes pick other tokens than one pass"
        );
        assert!(
            worst < 5e-2,
            "{n} lanes against one pass: {rel:.3e} (worst row {worst:.3e})"
        );
    }
}

/// Pooled index key `pool` of DSA layer `j` in a slot, dequantized.
fn pooled_key(kv: &GlmKv, j: usize, pool: usize) -> Vec<f32> {
    let per_page = PAGE_TOKENS / 4;
    let b = kv.download_page_block(pool / per_page, j).unwrap();
    let i = pool % per_page;
    let codes = &b[PAGE_POOL_CODES_OFFSET + i * 128..PAGE_POOL_CODES_OFFSET + (i + 1) * 128];
    let s = PAGE_POOL_SCALES_OFFSET + i * 4;
    decode_index_key(codes, f32::from_le_bytes(b[s..s + 4].try_into().unwrap()))
}

/// Two prompts in one prefill (or with `cuts`, in one-lane passes cut at those rows, as the lanes
/// hold them), then `steps` batched decode steps: the prompts' last rows' logits and their picks
/// and the steps', and the two slots. The steps take the tokens of `tokens` (a run's picks) when
/// given, else the run's own picks.
fn batched(
    fwd: &mut GlmForward,
    p: &[&[u32]; 2],
    steps: usize,
    cuts: &[usize],
    tokens: Option<&[u32]>,
) -> (Vec<f32>, Vec<u32>, [GlmKv; 2]) {
    let mut kvs = [fwd.kv.slot().unwrap(), fwd.kv.slot().unwrap()];
    for (kv, p) in kvs.iter_mut().zip(p) {
        kv.reserve(p.len() + steps).unwrap();
    }
    let (mut picks, mut logits) = prefill_in_passes(fwd, &mut kvs, p, cuts);
    let mut last = picks.clone();
    let [a, b] = &mut kvs;
    for s in 0..steps {
        let feed = tokens.map_or(last.clone(), |t| t[2 * s..2 * s + 2].to_vec());
        last = fwd
            .decode(&mut [(&mut *a, feed[0]), (&mut *b, feed[1])])
            .unwrap();
        picks.extend(&last);
        logits.extend(fwd.logits(2).unwrap());
    }
    (logits, picks, kvs)
}

/// The pools the cuts of a pass of the two prompts fall in, as (prompt, pool): where a cut splits
/// a prompt, the pool holding the next lane's first row.
fn cut_pools(p: &[&[u32]; 2], cuts: &[usize]) -> Vec<(usize, usize)> {
    let (a, b) = (p[0].len(), p[1].len());
    cuts.iter()
        .filter_map(|&c| {
            if c < a {
                Some((0, c / 4))
            } else if c > a && c < a + b {
                Some((1, (c - a) / 4))
            } else {
                None
            }
        })
        .collect()
}

/// A DSA tail record: its count of tokens, the bytes of its valid ones (a 16-byte header, then
/// each token's raw key and gate in BF16, 512 bytes), and their values.
fn tail_count(t: &[u8]) -> u32 {
    u32::from_le_bytes(t[..4].try_into().unwrap())
}

fn tail_valid(t: &[u8]) -> &[u8] {
    &t[..16 + (tail_count(t) as usize).min(3) * 512]
}

fn tail_values(t: &[u8]) -> Vec<f32> {
    tail_valid(t)[16..]
        .chunks(2)
        .map(|c| f32::from_bits(u32::from(u16::from_le_bytes([c[0], c[1]])) << 16))
        .collect()
}

/// The two prompts of tests 2 and 4, and a forward whose one lane holds both (lanes of 361 rows).
const BATCH: [usize; 2] = [150, 211];

fn batch_config() -> ForwardConfig {
    ForwardConfig {
        max_rows: MAX_LANES * (BATCH[0] + BATCH[1]),
        lanes: MAX_LANES,
        min_lane_rows: 16,
        max_verify_rows: 8,
        max_requests: 4,
        ..ForwardConfig::default()
    }
}

#[test]
fn prompts_batched_across_the_cuts() {
    if !gpu_with(7.0) {
        return;
    }
    let total = BATCH[0] + BATCH[1];
    let cfg = batch_config();
    // Rows for the slots below, and room for a snapshot mark.
    let Some(mut fwd) = forward_with(LAYERS, cfg, |_| Box::new(ZeroExperts), 8, 640, 16) else {
        return;
    };
    let (p1, p2) = (ids(1, BATCH[0]), ids(2, BATCH[1]));
    let prompts = [&p1[..], &p2[..]];
    let dsa = fwd.shape().dsa_index[3].expect("layer 3 is a DSA layer");
    let tails = |r: &[GlmKv; 2]| -> Vec<Vec<u8>> {
        r.iter().map(|kv| kv.download_tail(dsa).unwrap()).collect()
    };
    let values = |t: &[Vec<u8>]| -> Vec<f32> { t.iter().flat_map(|x| tail_values(x)).collect() };
    let counts = |t: &[Vec<u8>]| -> Vec<u32> { t.iter().map(|x| tail_count(x)).collect() };

    fwd.set_lane_trace(true, true);
    fwd.cfg.lanes = 1;
    let one = batched(&mut fwd, &prompts, 4, &[], None);
    // The other runs' steps take the tokens the one-pass run's steps took: the prompts' picks,
    // then each step's.
    let feed = one.1[..8].to_vec();
    let t1 = tails(&one.2);
    for n in 2..=MAX_LANES {
        let cuts = lane_cuts(total, n);
        fwd.cfg.lanes = 1;
        let seq = batched(&mut fwd, &prompts, 4, &cuts, Some(&feed));
        fwd.cfg.lanes = n;
        let allocs = device::allocations();
        let lanes = batched(&mut fwd, &prompts, 4, &[], Some(&feed));
        let trace = fwd.take_lane_trace().expect("a traced prefill");
        assert_eq!(trace.rows, lane_rows(total, n));
        let again = {
            let r = batched(&mut fwd, &prompts, 4, &[], Some(&feed));
            (r.0, r.1)
        };
        // A segment longer than a pass: chunks of `max_rows`, each in n lanes.
        fwd.cfg.max_rows = n * 100;
        let mut long = fwd.kv.slot().unwrap();
        long.reserve(900).unwrap();
        fwd.prefill(&mut [(&mut long, &ids(3, 900)[..])]).unwrap();
        let chunk = fwd.take_lane_trace().expect("a traced prefill");
        fwd.cfg.max_rows = cfg.max_rows;
        // Decode, verify and commit, and marks: no pass allocates device memory.
        fwd.verify(&mut [(&mut long, &ids(4, 5)[..])]).unwrap();
        fwd.commit(&mut [&mut long], &[3]).unwrap();
        let m = long.mark().unwrap();
        drop(m);
        drop(long);
        assert_eq!(
            device::allocations(),
            allocs,
            "a pass or a mark allocated device memory"
        );

        // The prompts' last rows (the steps after them are compared through their picks).
        let (rel, worst, agree) = close(&lanes.0[..2 * VOCAB], &one.0[..2 * VOCAB]);
        let same = lanes.1.iter().zip(&one.1).filter(|(x, y)| x == y).count();
        let (rows, decided_same) = decided(&one.0, &one.1, &lanes.1);
        let repeat = bits(&lanes.0, &again.0) && lanes.1 == again.1;
        let exact = bits(&lanes.0, &seq.0) && lanes.1 == seq.1;
        let state = (0..2).all(|i| kept(&fwd, &lanes.2[i]) == kept(&fwd, &seq.2[i]));
        // The pools the cuts fall in (a lane wrote the pool's first rows into the tail, the next
        // lane completed it), and the tails the prefill left.
        let pools = cut_pools(&prompts, &cuts);
        let key = |r: &[GlmKv; 2], &(i, pool): &(usize, usize)| pooled_key(&r[i], dsa, pool);
        let keys_exact = pools
            .iter()
            .all(|c| bits(&key(&lanes.2, c), &key(&seq.2, c)));
        let key_err = pools
            .iter()
            .map(|c| err(&key(&lanes.2, c), &key(&one.2, c)).rel_rms)
            .fold(0f64, f64::max);
        let (tn, ts) = (tails(&lanes.2), tails(&seq.2));
        let tails_exact = tn
            .iter()
            .zip(&ts)
            .all(|(x, y)| tail_valid(x) == tail_valid(y));
        let tail = err(&values(&tn), &values(&t1)).rel_rms;
        eprintln!(
            "two prompts of {} and {} tokens in one pass in {n} lanes of {:?} rows (cuts at \
             {cuts:?}, in the (prompt, pool)s {pools:?}), zero routed experts: against {n} passes \
             of the same rows: logits, picks, the pooled keys at the cuts, the tails and every \
             slot's state (KDA states, conv windows, pages, tails) bit for bit: {exact}, \
             {keys_exact}, {tails_exact}, {state}; against one pass: logits relative RMS \
             {rel:.3e} (worst row {worst:.3e}), argmax {agree}/2, picks {same}/{} \
             ({decided_same}/{rows} of the rows whose best two logits are {TIE} apart), the \
             pooled keys at the cuts {key_err:.3e} (worst), the tails left (counts {:?} / {:?}) \
             {tail:.3e}; a repeat bit for bit: {repeat}; a 900-token prompt in chunks of {} \
             rows, the last in lanes of {:?}",
            BATCH[0],
            BATCH[1],
            trace.rows,
            one.1.len(),
            counts(&tn),
            counts(&t1),
            n * 100,
            chunk.rows
        );
        eprintln!("lane trace, {n} lanes: {}", trace.summary());
        assert!(exact, "{n} lanes differ from {n} passes of the same rows");
        assert!(
            keys_exact && tails_exact && state,
            "the pooled keys, the tails or the slots' state differ from {n} passes"
        );
        assert!(repeat, "{n}-lane passes are not deterministic");
        assert_eq!(
            decided_same, rows,
            "{n} lanes pick other tokens than one pass"
        );
        assert!(
            worst < 5e-2,
            "{n} lanes against one pass: {rel:.3e} (worst {worst:.3e})"
        );
        assert_eq!(counts(&tn), counts(&t1), "tail counts");
        assert_eq!(
            counts(&t1),
            vec![(BATCH[0] % 4) as u32, (BATCH[1] % 4) as u32]
        );
        // The FP8 pooled keys and the BF16 tail within rounding (a key built from a stale tail
        // is off by its whole size).
        assert!(
            !pools.is_empty() && key_err < 5e-2,
            "the pooled keys at the cuts: {key_err:.3e}"
        );
        assert!(tail < 2e-2, "the tails: {tail:.3e}");
        // 900 rows in chunks of n * 100: the last chunk's rows, in n lanes.
        let last = 900 - (900 - 1) / (n * 100) * (n * 100);
        assert_eq!(chunk.rows, lane_rows(last, n), "the last chunk's lanes");
        assert_eq!(trace.layers.len(), 2, "two MoE layers traced");
    }
}

/// One one-lane pass over the rows `lo .. hi` of `prompts` laid end to end (the slots `kvs`, one
/// a prompt): the prompts that end in it, and their picks.
fn pass_of(
    fwd: &mut GlmForward,
    kvs: &mut [GlmKv],
    prompts: &[&[u32]],
    lo: usize,
    hi: usize,
) -> (Vec<usize>, Vec<u32>) {
    let (mut segs, mut row) = (Vec::new(), 0);
    for (i, p) in prompts.iter().enumerate() {
        let (a, b) = (lo.max(row), hi.min(row + p.len()));
        if a < b {
            segs.push((i, a - row..b - row));
        }
        row += p.len();
    }
    let first = segs[0].0;
    let got = {
        let mut pass: Vec<(&mut GlmKv, &[u32])> = kvs[first..first + segs.len()]
            .iter_mut()
            .zip(&segs)
            .map(|(k, (i, r))| (k, &prompts[*i][r.clone()]))
            .collect();
        fwd.prefill(&mut pass).unwrap()
    };
    segs.iter()
        .zip(got)
        .filter(|((i, r), _)| r.end == prompts[*i].len())
        .map(|((i, _), g)| (*i, g))
        .unzip()
}

/// Prefill `prompts` into fresh slots: in one call, or in one-lane passes cut at the rows
/// `cuts`, each pass's taps downloaded after it. Returns the slots, each prompt's pick, and the
/// taps (the passes' in order).
fn prefill_drafted(
    fwd: &mut GlmForward,
    prompts: &[Vec<u32>],
    cuts: &[usize],
) -> (Vec<GlmKv>, Vec<u32>, Vec<u16>) {
    let mut kvs: Vec<GlmKv> = prompts
        .iter()
        .map(|p| {
            let mut kv = fwd.kv.slot().unwrap();
            kv.reserve(p.len() + 8).unwrap();
            kv
        })
        .collect();
    let total: usize = prompts.iter().map(|p| p.len()).sum();
    let p: Vec<&[u32]> = prompts.iter().map(|p| &p[..]).collect();
    let (mut picks, mut taps) = (vec![0u32; prompts.len()], Vec::new());
    let mut lo = 0;
    for hi in cuts.iter().copied().chain(std::iter::once(total)) {
        let (ended, got) = pass_of(fwd, &mut kvs, &p, lo, hi);
        for (i, g) in ended.into_iter().zip(got) {
            picks[i] = g;
        }
        taps.extend(fwd.drafter().unwrap().taps(hi - lo).unwrap());
        lo = hi;
    }
    (kvs, picks, taps)
}

/// Every stored ring row of two slots' contexts (positions `0 .. len`, the drafter's five
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
fn the_drafter_in_lanes() {
    let cfg = ForwardConfig {
        max_rows: 128,
        lanes: MAX_LANES,
        min_lane_rows: 8,
        max_verify_rows: 16,
        max_requests: 4,
        ..ForwardConfig::default()
    };
    let Some(mut fwd) = drafting::drafted_forward(cfg, 8, 8, 64, 0.25) else {
        return;
    };
    // Routed outputs of zeros: the local experts' cache loads experts on demand, which would
    // hide the forward's own allocations (none) among its.
    fwd.set_experts(Box::new(ZeroExperts));
    // Everything the drafter uses while serving, allocated now.
    let rows = fwd.pass_rows();
    let (tap_bytes, scratch) = fwd
        .drafter_mut()
        .unwrap()
        .reserve(rows, cfg.max_requests)
        .unwrap();
    let allocs = device::allocations();
    let cases = [
        vec![drafting::ids(1, 50)],
        vec![drafting::ids(2, 20), drafting::ids(3, 33)],
    ];
    for n in 2..=MAX_LANES {
        for prompts in &cases {
            let total: usize = prompts.iter().map(|p| p.len()).sum();
            fwd.set_lane_trace(true, false);
            fwd.cfg.lanes = n;
            let (lanes, pn, tn) = prefill_drafted(&mut fwd, prompts, &[]);
            let trace = fwd.take_lane_trace().expect("a traced prefill");
            fwd.cfg.lanes = 1;
            let (one, p1, t1) = prefill_drafted(&mut fwd, prompts, &lane_cuts(total, n));
            assert_eq!(trace.rows, lane_rows(total, n), "{n} lanes");
            let taps = tn.len() == total * TAP_WIDTH && tn == t1;
            let contexts = lanes
                .iter()
                .zip(&one)
                .all(|(a, b)| same_context(&fwd, a, b));
            let drafts = lanes
                .iter()
                .zip(&one)
                .all(|(a, b)| greedy_draft(&mut fwd, a, 777) == greedy_draft(&mut fwd, b, 777));
            // Three decode steps after, both fed the passes' picks.
            let (mut lanes, mut one) = (lanes, one);
            let mut feed = p1.clone();
            let mut after = true;
            for _ in 0..3 {
                let a = {
                    let mut rows: Vec<(&mut GlmKv, u32)> =
                        lanes.iter_mut().zip(&feed).map(|(k, &t)| (k, t)).collect();
                    fwd.decode(&mut rows).unwrap()
                };
                let b = {
                    let mut rows: Vec<(&mut GlmKv, u32)> =
                        one.iter_mut().zip(&feed).map(|(k, &t)| (k, t)).collect();
                    fwd.decode(&mut rows).unwrap()
                };
                after &= a == b;
                feed = b;
            }
            let after = after
                && lanes
                    .iter()
                    .zip(&one)
                    .all(|(a, b)| same_context(&fwd, a, b) && kept(&fwd, a) == kept(&fwd, b));
            let lens: Vec<usize> = lanes
                .iter()
                .map(|k| k.draft_slot().unwrap().len())
                .collect();
            eprintln!(
                "the drafter in {n} lanes, prompts of {:?} tokens (lanes of {:?}): against {n} \
                 one-lane passes of the same rows, picks {}, taps {taps}, contexts and rings \
                 {contexts} (lengths {lens:?}), greedy drafts {drafts}; 3 decode steps after: \
                 tokens, rings and every slot's state {after}",
                prompts.iter().map(|p| p.len()).collect::<Vec<_>>(),
                trace.rows,
                if pn == p1 { "equal" } else { "differ" },
            );
            assert_eq!(pn, p1, "{n} lanes pick other tokens than {n} passes");
            assert!(taps, "the taps differ from {n} passes'");
            assert!(contexts, "the drafter's contexts differ from {n} passes'");
            assert!(drafts, "the drafts differ from {n} passes'");
            assert!(after, "the decode steps after differ");
        }
    }
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
    eprintln!(
        "drafter memory reserved up front: taps {:.1} MiB for {rows} rows, working memory {:.1} \
         MiB; no allocation in the passes, drafts and appends above",
        tap_bytes as f64 / (1u64 << 20) as f64,
        scratch as f64 / (1u64 << 20) as f64
    );
}

/// Routed experts that check the forward's calls in flight: `submit` takes a call while fewer
/// than `depth` are out, and `finish` must be the oldest's (its layer, rows and output buffer).
/// A call's routed output is its FFN input rows, written when it is finished, on the stream: a
/// call finished late, or into another lane's buffers, changes the bits.
struct Checked {
    depth: usize,
    out: VecDeque<(usize, usize, usize)>,
    /// The most calls seen in flight.
    most: Arc<AtomicUsize>,
}

impl ExpertBackend for Checked {
    fn submit(&mut self, call: &ExpertCall<'_>, _stream: &Stream) -> glm53f_forward::Result<()> {
        assert!(
            self.out.len() < self.depth,
            "a call submitted with {} in flight, depth {}",
            self.out.len(),
            self.depth
        );
        self.out
            .push_back((call.layer, call.rows, call.out as usize));
        self.most.fetch_max(self.out.len(), Ordering::Relaxed);
        Ok(())
    }

    fn finish(&mut self, call: &ExpertCall<'_>, stream: &Stream) -> glm53f_forward::Result<()> {
        let oldest = self
            .out
            .pop_front()
            .expect("a finish with no call in flight");
        assert_eq!(
            oldest,
            (call.layer, call.rows, call.out as usize),
            "a finish that is not the oldest call's"
        );
        // SAFETY: `x` and `out` hold `rows` BF16 rows of HIDDEN values on the device (the
        // trait's contract).
        device::check(
            unsafe {
                glm53f_forward::cuda::cudaMemcpyAsync(
                    call.out.cast(),
                    call.x.cast(),
                    call.rows * HIDDEN * 2,
                    glm53f_forward::cuda::MEMCPY_D2D,
                    stream.raw(),
                )
            },
            "cudaMemcpyAsync (the routed output)",
        )
    }

    fn depth(&self) -> usize {
        self.depth
    }
}

#[test]
fn lanes_at_every_depth() {
    if !gpu_with(7.0) {
        return;
    }
    let total = BATCH[0] + BATCH[1];
    let most = Arc::new(AtomicUsize::new(0));
    let checked = |depth: usize| -> Box<dyn ExpertBackend> {
        Box::new(Checked {
            depth,
            out: VecDeque::new(),
            most: most.clone(),
        })
    };
    let Some(mut fwd) = forward_with(LAYERS, batch_config(), |_| checked(1), 8, 640, 16) else {
        return;
    };
    let (p1, p2) = (ids(5, BATCH[0]), ids(6, BATCH[1]));
    let prompts = [&p1[..], &p2[..]];
    let mut report = Vec::new();
    for n in 2..=MAX_LANES {
        fwd.cfg.lanes = 1;
        fwd.set_experts(checked(1));
        let seq = batched(&mut fwd, &prompts, 2, &lane_cuts(total, n), None);
        let feed = seq.1[..4].to_vec();
        for depth in 1..=n {
            fwd.cfg.lanes = n;
            fwd.set_experts(checked(depth));
            most.store(0, Ordering::Relaxed);
            let lanes = batched(&mut fwd, &prompts, 2, &[], Some(&feed));
            let exact = bits(&lanes.0, &seq.0)
                && lanes.1 == seq.1
                && (0..2).all(|i| kept(&fwd, &lanes.2[i]) == kept(&fwd, &seq.2[i]));
            let reached = most.load(Ordering::Relaxed);
            report.push(format!(
                "{n} lanes at depth {depth}: bit for bit {exact}, {reached} in flight at most"
            ));
            assert!(exact, "{n} lanes at depth {depth} differ from {n} passes");
            assert_eq!(
                reached, depth,
                "{n} lanes at depth {depth}: calls in flight"
            );
        }
    }
    // The chunked KDA prefill (`ForwardConfig::kda_chunked_prefill`, a numerics change under
    // test) in four lanes, a request across three of them: against four passes.
    fwd.cfg.kda_chunked_prefill = true;
    fwd.cfg.lanes = 1;
    fwd.set_experts(checked(1));
    let seq = batched(&mut fwd, &prompts, 2, &lane_cuts(total, MAX_LANES), None);
    fwd.cfg.lanes = MAX_LANES;
    fwd.set_experts(checked(MAX_LANES));
    let lanes = batched(&mut fwd, &prompts, 2, &[], Some(&seq.1[..4]));
    let chunked = bits(&lanes.0, &seq.0)
        && lanes.1 == seq.1
        && (0..2).all(|i| kept(&fwd, &lanes.2[i]) == kept(&fwd, &seq.2[i]));
    report.push(format!(
        "{MAX_LANES} lanes with the chunked KDA prefill: bit for bit {chunked}"
    ));
    assert!(
        chunked,
        "{MAX_LANES} lanes with the chunked KDA prefill differ from {MAX_LANES} passes"
    );
    eprintln!(
        "two prompts of {} and {} tokens in N lanes, routed outputs written when finished, against \
         N passes of the same rows (logits and picks, 2 decode steps after, and every slot's \
         state): {}",
        BATCH[0],
        BATCH[1],
        report.join("; ")
    );
}
