//! Teacher-forced scoring, [`GlmForward::score`] (the engine side of the KL gate,
//! `docs/KL-GATE.md` section 4; feature `cuda`, real weights of layers 0-4 and the head, routed
//! experts returning zeros; skips without them).
//!
//! 1. **The forward's own logits.** A 150-token prompt scored in passes of 64 rows (two lanes
//!    each: 32 + 32, 32 + 32, then 11 + 11) gives, at each pass's last row, the bits of the
//!    logits a prefill of the prompt up to that row writes for its last row (the same passes,
//!    the same kernels), whichever other rows are scored with it (the head's GEMV takes them in
//!    groups of up to 8), and leaves the slot as the prefill of the whole prompt does.
//! 2. **The decode path.** In passes of 8 and of 5 rows (the row-independent decode kernels, one
//!    lane), every row's logits are the bits of serial decode steps.
//! 3. **The two paths.** Passes of 8 rows against one pass of 200 rows (two lanes of 100, the
//!    tensor-core GEMMs): every row's logits within the chain test's bound (5e-2 relative RMS,
//!    `tests/goldens_chain.rs`), and the same argmax on every row whose best two logits are at
//!    least [`TIE`] apart (closer calls flip with rounding of this size, as in `tests/lanes.rs`);
//!    not bit for bit.
//!
//! ```sh
//! GLM53F_CHECKPOINT_DIR=... \
//!   cargo test --release -p glm53f-forward --features cuda --test score -- --nocapture
//! ```
#![cfg(feature = "cuda")]

mod common;

use common::*;
use glm53f_forward::experts::ZeroExperts;
use glm53f_forward::forward::{ForwardConfig, GlmForward};
use glm53f_forward::kv::GlmKv;
use glm53f_forward::shape::{SAMPLE_VOCAB, VOCAB};

const LAYERS: usize = 5;

/// Near ties: rows whose two best logits are closer than this may pick either token under the
/// two paths' rounding (as in `tests/lanes.rs`).
const TIE: f32 = 0.25;

/// Deterministic token ids in the vocabulary.
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

fn bits(v: &[f32]) -> Vec<u32> {
    v.iter().map(|x| x.to_bits()).collect()
}

fn row(v: &[f32], r: usize) -> &[f32] {
    &v[r * VOCAB..(r + 1) * VOCAB]
}

/// The first maximum below the sampled vocabulary, and the gap to the second best.
fn best(v: &[f32]) -> (usize, f32) {
    let mut b = 0;
    for i in 0..SAMPLE_VOCAB {
        if v[i] > v[b] {
            b = i;
        }
    }
    let second = (0..SAMPLE_VOCAB)
        .filter(|&i| i != b)
        .map(|i| v[i])
        .fold(f32::NEG_INFINITY, f32::max);
    (b, v[b] - second)
}

/// What a slot holds for its committed tokens: every KDA layer's state (as bits) and conv window,
/// the DSA tail of the first DSA layer and its page blocks.
type Image = (Vec<Vec<u32>>, Vec<Vec<u16>>, Vec<u8>, Vec<Vec<u8>>);

fn image(kv: &GlmKv, kda_layers: usize) -> Image {
    let states = (0..kda_layers)
        .map(|j| bits(&kv.download_state(j).unwrap()))
        .collect();
    let convs = (0..kda_layers)
        .map(|j| kv.download_conv(j).unwrap())
        .collect();
    let pages = (0..kv.tokens().div_ceil(64))
        .map(|i| kv.download_page_block(i, 0).unwrap())
        .collect();
    (states, convs, kv.download_tail(0).unwrap(), pages)
}

/// Serial decode steps from an empty slot: every row's logits.
fn serial(fwd: &mut GlmForward, toks: &[u32]) -> Vec<f32> {
    let mut kv = fwd.kv.slot().unwrap();
    let mut out = Vec::with_capacity(toks.len() * VOCAB);
    for &t in toks {
        fwd.decode(&mut [(&mut kv, t)]).unwrap();
        out.extend(fwd.logits(1).unwrap());
    }
    out
}

#[test]
fn scoring_a_teacher_forced_sequence() {
    if !gpu_with(7.0) {
        return;
    }
    let cfg = ForwardConfig {
        max_rows: 256,
        lanes: 2,
        min_lane_rows: 8,
        max_verify_rows: 8,
        max_requests: 4,
        ..ForwardConfig::default()
    };
    let Some(mut fwd) = forward_with(LAYERS, cfg, |_| Box::new(ZeroExperts), 6, 64, 8) else {
        return;
    };
    let kl = fwd.shape().kda_layers;

    // 1. The forward's own logits: passes of 64 rows, as a prefill with max_rows 64 cuts the
    //    prompt.
    fwd.cfg.max_rows = 64;
    let prompt = ids(1, 150);
    let rows = [0usize, 1, 2, 3, 4, 5, 31, 32, 63, 64, 100, 127, 128, 149];
    let ends = [63usize, 127, 149];
    fwd.set_lane_trace(true, false);
    let mut kv = fwd.kv.slot().unwrap();
    let got = fwd.score(&mut kv, &prompt, &rows, 64).unwrap();
    let last = fwd.take_lane_trace().expect("a traced pass");
    assert_eq!(last.rows, vec![11, 11], "the last pass ran in two lanes");
    assert_eq!(kv.tokens(), prompt.len());
    assert_eq!(got.len(), rows.len() * VOCAB);
    let mut exact = true;
    let mut picks = true;
    let mut full = None;
    for &e in &ends {
        let mut r = fwd.kv.slot().unwrap();
        let pick = fwd.prefill(&mut [(&mut r, &prompt[..=e])]).unwrap();
        let want = fwd.logits(1).unwrap();
        let i = rows.iter().position(|&x| x == e).unwrap();
        exact &= bits(row(&got, i)) == bits(&want);
        picks &= best(row(&got, i)).0 == pick[0] as usize;
        if e + 1 == prompt.len() {
            full = Some(r);
        }
    }
    // The same rows scored on their own (other groups of the head's GEMV).
    let mut alone = fwd.kv.slot().unwrap();
    let only = fwd.score(&mut alone, &prompt, &ends, 64).unwrap();
    let alone_exact = ends.iter().enumerate().all(|(k, &e)| {
        let i = rows.iter().position(|&x| x == e).unwrap();
        bits(row(&only, k)) == bits(row(&got, i))
    });
    let slot_same = image(&kv, kl) == image(full.as_ref().unwrap(), kl);
    eprintln!(
        "a 150-token prompt scored in passes of 64 rows (two lanes each, the last {:?}): the \
         last rows of the passes (63, 127, 149) bit for bit the logits of prefills up to them: \
         {exact}; their argmax the prefills' picks: {picks}; scored alone, the same bits: \
         {alone_exact}; the slot as the whole prompt's prefill leaves it: {slot_same}",
        last.rows
    );
    assert!(exact, "scored rows differ from the forward's own logits");
    assert!(
        picks,
        "scored rows' argmax differs from the forward's picks"
    );
    assert!(
        alone_exact,
        "a row's logits depend on the other rows scored"
    );
    assert!(slot_same, "scoring left the slot otherwise than a prefill");
    drop((kv, full, alone));

    // Every row of a two-lane pass: a pass of 128 rows in lanes of 64 gives the bits of two
    // one-lane passes of 64 (`tests/lanes.rs`), so each row's logits must too, lane B's rows
    // included.
    let first: Vec<usize> = (0..128).collect();
    fwd.cfg.max_rows = 128;
    let mut kv = fwd.kv.slot().unwrap();
    let two = fwd.score(&mut kv, &prompt[..128], &first, 128).unwrap();
    assert_eq!(
        fwd.take_lane_trace().expect("a traced pass").rows,
        vec![64, 64]
    );
    fwd.cfg.lanes = 1;
    fwd.cfg.max_rows = 64;
    let mut kv1 = fwd.kv.slot().unwrap();
    let one = fwd.score(&mut kv1, &prompt[..128], &first, 64).unwrap();
    fwd.cfg.lanes = 2;
    let lanes_exact = (0..128).all(|r| bits(row(&two, r)) == bits(row(&one, r)));
    eprintln!(
        "128 rows in one pass of two lanes (64 + 64) against two one-lane passes of 64: every \
         row bit for bit: {lanes_exact}"
    );
    assert!(
        lanes_exact,
        "a row of the two-lane pass differs from one-lane passes"
    );
    drop((kv, kv1));

    // 2. The decode path: passes of 8 rows or fewer give the bits of serial decode steps.
    fwd.cfg.max_rows = 256;
    let p2 = ids(2, 29);
    let all: Vec<usize> = (0..p2.len()).collect();
    let steps = serial(&mut fwd, &p2);
    for pass in [8usize, 5] {
        let mut kv = fwd.kv.slot().unwrap();
        let got = fwd.score(&mut kv, &p2, &all, pass).unwrap();
        let same = (0..p2.len()).all(|r| bits(row(&got, r)) == bits(row(&steps, r)));
        eprintln!(
            "29 tokens in passes of {pass} rows, every row: bit for bit serial decode steps: {same}"
        );
        assert!(
            same,
            "passes of {pass} rows differ from serial decode steps"
        );
    }

    // 3. The two paths: passes of 8 rows against one pass of 200 rows in two lanes.
    let p3 = ids(3, 200);
    let all: Vec<usize> = (0..p3.len()).collect();
    let mut kv = fwd.kv.slot().unwrap();
    let dec = fwd.score(&mut kv, &p3, &all, 8).unwrap();
    drop(kv);
    let mut kv = fwd.kv.slot().unwrap();
    let pre = fwd.score(&mut kv, &p3, &all, p3.len()).unwrap();
    drop(kv);
    let lanes = fwd.take_lane_trace().expect("a traced pass").rows;
    assert_eq!(lanes, vec![100, 100], "the 200-row pass ran in two lanes");
    let (mut worst, mut sum, mut identical) = (0f64, 0f64, 0);
    let (mut decided, mut agree) = (0, 0);
    for r in 0..p3.len() {
        let (a, b) = (row(&dec, r), row(&pre, r));
        let e = err(b, a).rel_rms;
        worst = worst.max(e);
        sum += e;
        identical += usize::from(bits(a) == bits(b));
        let (ta, gap) = best(a);
        if gap >= TIE {
            decided += 1;
            agree += usize::from(best(b).0 == ta);
        }
    }
    eprintln!(
        "200 tokens, every row, passes of 8 rows (decode kernels) against one pass of 200 rows \
         (prefill kernels, lanes {lanes:?}): logits relative RMS mean {:.3e}, worst row \
         {worst:.3e}; rows bit for bit equal {identical}/200; argmax equal on {agree}/{decided} \
         rows whose best two logits are at least {TIE} apart",
        sum / p3.len() as f64
    );
    assert!(worst < 5e-2, "the two paths differ by {worst:.3e} on a row");
    assert!(decided >= 100, "only {decided} rows decided");
    assert_eq!(agree, decided, "the two paths pick other tokens");

    // Refused: rows out of order or past the tokens, empty passes, passes the forward cannot
    // hold (prefill passes of 64 rows here).
    fwd.cfg.max_rows = 64;
    let mut kv = fwd.kv.slot().unwrap();
    for (rows, pass) in [
        (&[3usize, 2][..], 8),
        (&[1, 1][..], 8),
        (&[29][..], 8),
        (&[0][..], 0),
        (&[0][..], 65),
    ] {
        let t = if pass == 65 { &prompt[..] } else { &p2[..] };
        assert!(
            fwd.score(&mut kv, t, rows, pass).is_err(),
            "scoring rows {rows:?} in passes of {pass} was accepted"
        );
    }
    assert_eq!(kv.tokens(), 0, "a refused request appended rows");
}
