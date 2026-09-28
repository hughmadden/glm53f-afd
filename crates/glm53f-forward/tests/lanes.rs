//! Two-lane prefill against one lane (feature `cuda`; real weights of layers 0-4 and the head;
//! skips without them).
//!
//! **Exact.** Lane B's attention at a layer reads only what lane A's attention at that layer
//! left (the KDA states and conv windows, the MLA latents and pooled keys, a split request's DSA
//! tail), so a two-lane pass must give the bits of the same rows run as two passes one after
//! the other: lane A's rows, then lane B's. The tests check that bit for bit (logits, picks, the
//! pooled keys and tails, the decode steps after).
//!
//! **Within rounding of one pass.** Against one pass over all the rows, the lanes change the row
//! counts of the tensor-core GEMMs (cuBLAS picks its kernels by the row count; the FP8 GEMM
//! activations are quantized per pass), as a prompt cut into chunks already does. The tests hold
//! the two to the same tokens on every row whose best two logits are at least [`TIE`] apart
//! (closer calls are near ties that rounding of this size flips: the oracle's own first row is a
//! 0.042 tie, and one pass and two lanes land on either side of it), to logits within the chain
//! test's bound against the oracle (`tests/goldens_chain.rs`: 5e-2 relative RMS per row), and the
//! two-lane path itself to the goldens with that test's bounds.
//!
//! 1. **The oracle's prompt** (33 tokens, lanes of 17 and 16 rows) through layers 0-4 and the
//!    head, then the 8 fixed decode steps, with the local FP8 experts behind the golden routes:
//!    two lanes against two passes of 17 and 16 rows (bit for bit), against one pass, and both
//!    against the golden logits.
//! 2. **Two prompts batched across the cut** (150 and 211 tokens: the cut at row 181 splits the
//!    second prompt 31 rows in, inside an indexer pool of 4), routed experts returning zeros:
//!    two lanes against the two passes (bit for bit) and against one pass (picks, logits, the
//!    pooled index key across the cut, the DSA tails), then 4 decode steps each; two-lane passes
//!    repeat bit for bit; no pass allocates device memory; a segment longer than a pass runs in
//!    two-lane chunks; the lane trace.
//!
//! ```sh
//! GLM53F_CHECKPOINT_DIR=... GLM53F_EXPERTS_DIR=... \
//!   cargo test --release -p glm53f-forward --features cuda --test lanes -- --nocapture --test-threads=1
//! ```
#![cfg(feature = "cuda")]

mod common;

use std::collections::{HashMap, VecDeque};

use common::*;
use glm53f_dsa::cache::{decode_index_key, PAGE_POOL_CODES_OFFSET, PAGE_POOL_SCALES_OFFSET};
use glm53f_forward::device;
use glm53f_forward::experts::{LocalFp8Experts, ZeroExperts};
use glm53f_forward::forward::{ForwardConfig, GlmForward};
use glm53f_forward::gemm::Fp8Act;
use glm53f_forward::kv::GlmKv;
use glm53f_forward::shape::{SAMPLE_VOCAB, TOP_K, VOCAB};

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
/// a 0.042 gap on its first row, and one pass and two lanes land on either side of it).
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

/// The prompt in one prefill (or with `cut`, its first `cut` tokens, then the rest), then the
/// steps: every row's logits (the prompt's last row, then each step) and picks.
fn chain(
    fwd: &mut GlmForward,
    prompt: &[u32],
    steps: &[u32],
    cut: Option<usize>,
) -> (Vec<f32>, Vec<u32>) {
    let mut kv = fwd.kv.slot().unwrap();
    kv.reserve(prompt.len() + steps.len()).unwrap();
    if let Some(k) = cut {
        fwd.prefill(&mut [(&mut kv, &prompt[..k])]).unwrap();
    }
    let rest = &prompt[cut.unwrap_or(0)..];
    let mut picks = fwd.prefill(&mut [(&mut kv, rest)]).unwrap();
    let mut logits = fwd.logits(1).unwrap();
    for &t in steps {
        picks.extend(fwd.decode(&mut [(&mut kv, t)]).unwrap());
        logits.extend(fwd.logits(1).unwrap());
    }
    (logits, picks)
}

#[test]
fn the_oracles_prompt_in_two_lanes() {
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
        max_rows: 2 * PROMPT,
        lanes: 2,
        min_lane_rows: 8,
        max_verify_rows: 8,
        max_requests: 4,
        ..ForwardConfig::default()
    };
    let Some(edir) = experts_dir() else {
        return;
    };
    let routes = golden_routes(&g, 3);
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

    fwd.set_lane_trace(true, false);
    let two = chain(&mut fwd, &prompt, &steps, None);
    let trace = fwd.take_lane_trace().expect("a traced prefill");
    assert_eq!(trace.rows, vec![17, 16], "the prompt ran in two lanes");
    fwd.cfg.lanes = 1;
    let seq = chain(&mut fwd, &prompt, &steps, Some(17));
    let one = chain(&mut fwd, &prompt, &steps, None);
    assert!(
        fwd.take_lane_trace()
            .is_some_and(|t| t.rows == vec![PROMPT]),
        "then in one lane"
    );
    let exact = two.1 == seq.1
        && two
            .0
            .iter()
            .zip(&seq.0)
            .all(|(x, y)| x.to_bits() == y.to_bits());

    let gl = g.f32("head", "head.logits");
    let (e2, e1) = (err(&two.0, &gl).rel_rms, err(&one.0, &gl).rel_rms);
    let (a2, a1) = (close(&two.0, &gl).2, close(&one.0, &gl).2);
    let (rel, worst, agree) = close(&two.0, &one.0);
    let same = two.1.iter().zip(&one.1).filter(|(x, y)| x == y).count();
    let (rows, decided_same) = decided(&one.0, &one.1, &two.1);
    eprintln!(
        "the oracle's prompt, layers 0-4, golden routes, local FP8 experts: two lanes against \
         two passes of 17 and 16 rows bit for bit: {exact}; logits against the golden, relative \
         RMS {e2:.3e} (two lanes) / {e1:.3e} (one pass), argmax {a2}/9 / {a1}/9; two lanes \
         against one pass: relative RMS {rel:.3e} (worst row {worst:.3e}), argmax {agree}/9, \
         picks {same}/9, and {decided_same}/{rows} of the rows whose best two logits are at \
         least {TIE} apart"
    );
    assert!(exact, "two lanes differ from two passes of the same rows");
    // The chain test's bounds against the oracle (tests/goldens_chain.rs).
    for (e, a) in [(e2, a2), (e1, a1)] {
        assert!(e < 5e-2 && a >= 8, "against the golden: {e:.3e}, {a}/9");
    }
    // Two lanes against one pass: the same tokens but on near ties, logits within the same
    // bound.
    assert!(rows >= 7, "{rows} rows decided");
    assert_eq!(
        decided_same, rows,
        "two lanes pick other tokens than one pass"
    );
    assert!(
        worst < 5e-2,
        "two lanes against one pass: {rel:.3e} (worst row {worst:.3e})"
    );
}

/// Pooled index key `pool` of DSA layer `j` in a slot's page 0, dequantized.
fn pooled_key(kv: &GlmKv, j: usize, pool: usize) -> Vec<f32> {
    let b = kv.download_page_block(0, j).unwrap();
    let codes = &b[PAGE_POOL_CODES_OFFSET + pool * 128..PAGE_POOL_CODES_OFFSET + (pool + 1) * 128];
    let s = PAGE_POOL_SCALES_OFFSET + pool * 4;
    decode_index_key(codes, f32::from_le_bytes(b[s..s + 4].try_into().unwrap()))
}

/// Two prompts in one prefill pass (or with `cut`, the rows before it in one pass and the
/// rest in a second, as lanes A and B hold them), then `steps` batched decode steps: the prompts'
/// last rows' logits and their picks and the steps', and the two slots. The steps take the
/// tokens of `tokens` (a run's picks) when given, else the run's own picks.
fn batched(
    fwd: &mut GlmForward,
    p: &[&[u32]; 2],
    steps: usize,
    cut: Option<usize>,
    tokens: Option<&[u32]>,
) -> (Vec<f32>, Vec<u32>, [GlmKv; 2]) {
    let mut a = fwd.kv.slot().unwrap();
    let mut b = fwd.kv.slot().unwrap();
    a.reserve(p[0].len() + steps).unwrap();
    b.reserve(p[1].len() + steps).unwrap();
    let (mut picks, logits) = match cut {
        None => {
            let picks = fwd.prefill(&mut [(&mut a, p[0]), (&mut b, p[1])]).unwrap();
            (picks, fwd.logits(2).unwrap())
        }
        Some(k) => {
            let k = k - p[0].len();
            let first = fwd
                .prefill(&mut [(&mut a, p[0]), (&mut b, &p[1][..k])])
                .unwrap();
            let mut logits = fwd.logits(1).unwrap();
            let second = fwd.prefill(&mut [(&mut b, &p[1][k..])]).unwrap();
            logits.extend(fwd.logits(1).unwrap());
            (vec![first[0], second[0]], logits)
        }
    };
    let mut logits = logits;
    let mut last = picks.clone();
    for s in 0..steps {
        let feed = tokens.map_or(last.clone(), |t| t[2 * s..2 * s + 2].to_vec());
        last = fwd
            .decode(&mut [(&mut a, feed[0]), (&mut b, feed[1])])
            .unwrap();
        picks.extend(&last);
        logits.extend(fwd.logits(2).unwrap());
    }
    (logits, picks, [a, b])
}

#[test]
fn two_prompts_batched_across_the_cut() {
    if !gpu_with(7.0) {
        return;
    }
    let (n1, n2) = (150usize, 211usize);
    let total = n1 + n2;
    let cfg = ForwardConfig {
        max_rows: 2 * total,
        lanes: 2,
        min_lane_rows: 16,
        max_verify_rows: 8,
        max_requests: 4,
        ..ForwardConfig::default()
    };
    // Rows for the slots below, and room for a snapshot mark.
    let Some(mut fwd) = forward_with(LAYERS, cfg, |_| Box::new(ZeroExperts), 8, 640, 16) else {
        return;
    };
    let (p1, p2) = (ids(1, n1), ids(2, n2));
    let prompts = [&p1[..], &p2[..]];
    let dsa = fwd.shape().dsa_index[3].expect("layer 3 is a DSA layer");

    let at = total.div_ceil(2);
    fwd.set_lane_trace(true, true);
    fwd.cfg.lanes = 1;
    let one = batched(&mut fwd, &prompts, 4, None, None);
    // The other runs' steps take the tokens the one-pass run's steps took: the prompts' picks,
    // then each step's.
    let feed = one.1[..8].to_vec();
    let seq = batched(&mut fwd, &prompts, 4, Some(at), Some(&feed));
    fwd.cfg.lanes = 2;
    let allocs = device::allocations();
    let two = batched(&mut fwd, &prompts, 4, None, Some(&feed));
    let trace = fwd.take_lane_trace().expect("a traced prefill");
    assert_eq!(trace.rows, vec![at, total - at]);
    let again = {
        let r = batched(&mut fwd, &prompts, 4, None, Some(&feed));
        (r.0, r.1)
    };
    // A segment longer than a pass: chunks of `max_rows`, each in two lanes.
    let mut long = fwd.kv.slot().unwrap();
    long.reserve(900).unwrap();
    fwd.prefill(&mut [(&mut long, &ids(3, 900)[..])]).unwrap();
    let chunk = fwd.take_lane_trace().expect("a traced prefill");
    // Decode, verify and commit, and marks: no pass allocates device memory.
    fwd.verify(&mut [(&mut long, &ids(4, 5)[..])]).unwrap();
    fwd.commit(&mut [&mut long], &[3]).unwrap();
    let m = long.mark().unwrap();
    drop(m);
    assert_eq!(
        device::allocations(),
        allocs,
        "a pass or a mark allocated device memory"
    );

    let bits = |a: &[f32], b: &[f32]| a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits());
    // The prompts' last rows (the steps after them are compared through their picks).
    let (rel, worst, agree) = close(&two.0[..2 * VOCAB], &one.0[..2 * VOCAB]);
    let same = two.1.iter().zip(&one.1).filter(|(x, y)| x == y).count();
    let (rows, decided_same) = decided(&one.0, &one.1, &two.1);
    let repeat = bits(&two.0, &again.0) && two.1 == again.1;
    let exact = bits(&two.0, &seq.0) && two.1 == seq.1;
    // The pool the cut falls in (the second prompt's tokens 28-31: lane A wrote 28-30 into the
    // tail, lane B completed the pool with 31), and the tails the prefill left.
    let pk = |r: &[GlmKv; 2]| pooled_key(&r[1], dsa, 7);
    let (k2, k1) = (pk(&two.2), pk(&one.2));
    let key = err(&k2, &k1).rel_rms;
    let key_exact = bits(&k2, &pk(&seq.2));
    let tails = |r: &[GlmKv; 2]| -> Vec<Vec<u8>> {
        r.iter().map(|kv| kv.download_tail(dsa).unwrap()).collect()
    };
    let (t2, t1) = (tails(&two.2), tails(&one.2));
    // A tail record: the count, then the raw keys and gates in BF16.
    let counts = |t: &[Vec<u8>]| -> Vec<u32> {
        t.iter()
            .map(|x| u32::from_le_bytes(x[..4].try_into().unwrap()))
            .collect()
    };
    // The valid tokens' keys and gates (512 bytes each after a 16-byte header).
    let tail_vals = |t: &[Vec<u8>]| -> Vec<f32> {
        t.iter()
            .flat_map(|x| {
                let n = u32::from_le_bytes(x[..4].try_into().unwrap()) as usize;
                x[16..16 + n.min(3) * 512]
                    .chunks(2)
                    .map(|c| f32::from_bits(u32::from(u16::from_le_bytes([c[0], c[1]])) << 16))
                    .collect::<Vec<_>>()
            })
            .collect()
    };
    let tail = err(&tail_vals(&t2), &tail_vals(&t1)).rel_rms;
    let tails_exact = t2.iter().zip(tails(&seq.2)).all(|(x, y)| {
        let n = 16 + (u32::from_le_bytes(x[..4].try_into().unwrap()) as usize).min(3) * 512;
        x[..n] == y[..n]
    });
    eprintln!(
        "two prompts of {n1} and {n2} tokens in one pass (lanes {:?}), zero routed experts: two \
         lanes against two passes of the same rows: logits, picks, the pooled key across the cut \
         and the tails bit for bit: {exact}, {key_exact}, {tails_exact}; against one pass: logits \
         relative RMS {rel:.3e} (worst row {worst:.3e}), argmax {agree}/2, picks {same}/{} \
         ({decided_same}/{rows} of the rows whose best two logits are {TIE} apart), the \
         pooled key across the cut {key:.3e}, the tails left (counts {:?} / {:?}) {tail:.3e}; a \
         two-lane repeat bit for bit: {repeat}; a 900-token prompt in two-lane chunks of {} rows",
        trace.rows,
        one.1.len(),
        counts(&t2),
        counts(&t1),
        chunk.rows.iter().sum::<usize>()
    );
    eprintln!("lane trace, two lanes: {}", trace.summary());
    assert!(exact, "two lanes differ from two passes of the same rows");
    assert!(
        key_exact && tails_exact,
        "the pooled key or the tails differ from two passes"
    );
    assert!(repeat, "two-lane passes are not deterministic");
    assert_eq!(
        decided_same, rows,
        "two lanes pick other tokens than one pass"
    );
    assert!(
        worst < 5e-2,
        "two lanes against one pass: {rel:.3e} (worst {worst:.3e})"
    );
    assert_eq!(counts(&t2), counts(&t1), "tail counts");
    assert_eq!(counts(&t1), vec![(n1 % 4) as u32, (n2 % 4) as u32]);
    // The FP8 pooled key and the BF16 tail within rounding (a key built from a stale tail is
    // off by its whole size).
    assert!(key < 5e-2, "the pooled key across the cut: {key:.3e}");
    assert!(tail < 2e-2, "the tails: {tail:.3e}");
    assert_eq!(chunk.rows.iter().sum::<usize>(), 900 - cfg.max_rows);
    assert_eq!(trace.layers.len(), 2, "two MoE layers traced");
}
