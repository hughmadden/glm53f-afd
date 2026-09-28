//! The DFlash2 drafter wired into the forward (feature `coordinator`; the coordinator's weights,
//! the experts of layers 3 and 4 and the drafter's checkpoint; skips without them). A forward
//! over all 45 decoder layers (`drafting`): layers past the loaded ones repeat their weights, so
//! the text is garbage, but the taps at layers 5 to 42, the drafter's context and its drafts are
//! the real plumbing.
//!
//! 1. **Taps.** In a prefill (20 rows), a decode row and a verify window (8 rows), the drafter's
//!    taps are bit for bit the mean of the four streams of the outputs of layers 5, 14, 24, 33
//!    and 42 (read through the forward's own `LayerOut` taps), in that order; the drafter's
//!    context follows the committed length.
//! 2. **Commits.** A verify window of 8 committed at 5 leaves the drafter's ring bit for bit as
//!    a window of those 5 committed whole (the same rows appended in one call): the rejected
//!    rows never reach it. Against 5 serial decode steps (appended a row at a time, so the
//!    drafter's GEMMs round differently) the ring agrees to a BF16 unit and the drafts are
//!    compared.
//! 3. **Fork and rewind.** A slot forked at an earlier mark of a slot that ran on, and that slot
//!    rewound to the mark, draft exactly as a fresh slot fed the same rows.
//! 4. **Host restore.** A slot restored from host images starts cold (the images hold no taps):
//!    no drafts until `warm_rows` rows are back in context; the target's next token is the
//!    source's.
#![cfg(feature = "coordinator")]

mod common;
mod drafting;

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use common::*;
use drafting::*;
use glm53f_coordinator::model::{DecodeRow, DraftRow, ModelForward, Pick};
use glm53f_dflash::TARGET_LAYERS;
use glm53f_forward::draft::{DraftReq, TAP_WIDTH};
use glm53f_forward::forward::{ForwardConfig, GlmForward, Tap, TapBuf, TapPoint};
use glm53f_forward::kv::GlmKv;
use glm53f_forward::reference::stream_mean;
use glm53f_forward::serve::ServedForward;
use glm53f_forward::shape::HIDDEN;

type Streams = Arc<Mutex<HashMap<usize, Vec<u16>>>>;

/// The taps a pass of `rows` rows should have written: the mean of each target layer's output
/// streams, layer 5 first.
fn expected_taps(streams: &Streams, rows: usize) -> Vec<u16> {
    let s = streams.lock().unwrap();
    let means: Vec<Vec<u16>> = TARGET_LAYERS
        .iter()
        .map(|l| stream_mean(&s[l][..rows * 4 * HIDDEN], rows, HIDDEN))
        .collect();
    let mut out = Vec::with_capacity(rows * TAP_WIDTH);
    for r in 0..rows {
        for m in &means {
            out.extend_from_slice(&m[r * HIDDEN..(r + 1) * HIDDEN]);
        }
    }
    out
}

fn check_taps(fwd: &GlmForward, streams: &Streams, rows: usize, what: &str) {
    let want = expected_taps(streams, rows);
    let got = fwd.drafter().unwrap().taps(rows).unwrap();
    let differ = got.iter().zip(&want).filter(|(a, b)| a != b).count();
    assert_eq!(differ, 0, "{what}: {differ} tap values differ");
    let nonzero = got.iter().filter(|&&v| v & 0x7fff != 0).count();
    assert!(nonzero > got.len() / 2, "{what}: taps mostly zero");
    eprintln!("{what}: {rows} rows of taps equal the mean of the layer-output streams of layers {TARGET_LAYERS:?}");
}

fn ctx(kv: &GlmKv) -> (usize, usize) {
    let d = kv.draft_slot().unwrap();
    (d.len(), d.lo())
}

fn greedy(fwd: &mut GlmForward, kv: &GlmKv, anchor: u32) -> glm53f_dflash::seam::Proposal {
    fwd.draft(&[DraftReq {
        kv,
        anchor,
        temperature: 0.0,
        uniforms: &[],
    }])
    .unwrap()
    .remove(0)
}

fn sampled(fwd: &mut GlmForward, kv: &GlmKv, anchor: u32) -> glm53f_dflash::seam::Proposal {
    let u = [0.11f32, 0.93, 0.47, 0.02, 0.66, 0.35, 0.81];
    fwd.draft(&[DraftReq {
        kv,
        anchor,
        temperature: 0.8,
        uniforms: &u,
    }])
    .unwrap()
    .remove(0)
}

/// Ring rows of two slots at positions `0 .. n`: (values differing, largest difference over the
/// row's RMS).
fn ring_diff(fwd: &GlmForward, a: &GlmKv, b: &GlmKv, n: usize) -> (usize, f32) {
    let d = fwd.drafter().unwrap();
    let (mut differ, mut worst) = (0usize, 0f32);
    for l in 0..5 {
        for p in 0..n {
            let (ka, va) = d.ring_row(a, l, p).unwrap();
            let (kb, vb) = d.ring_row(b, l, p).unwrap();
            for (x, y) in [(ka, kb), (va, vb)] {
                let (x, y) = (widen(&x), widen(&y));
                let rms = (y.iter().map(|v| v * v).sum::<f32>() / y.len() as f32).sqrt();
                for (p, q) in x.iter().zip(&y) {
                    if p.to_bits() != q.to_bits() {
                        differ += 1;
                        worst = worst.max((p - q).abs() / rms.max(1e-30));
                    }
                }
            }
        }
    }
    (differ, worst)
}

fn decode_all(fwd: &mut GlmForward, kv: &mut GlmKv, toks: &[u32]) -> Vec<u32> {
    toks.iter()
        .map(|&t| fwd.decode(&mut [(&mut *kv, t)]).unwrap()[0])
        .collect()
}

#[test]
fn drafter_taps_context_forks_and_restores() {
    let cfg = ForwardConfig {
        max_rows: 64,
        max_verify_rows: 16,
        max_requests: 4,
        ..ForwardConfig::default()
    };
    let Some(mut fwd) = drafted_forward(cfg, 8, 8, 64, 1.5) else {
        return;
    };
    let t0 = std::time::Instant::now();

    // 1. Taps against the forward's own LayerOut streams, in all three modes.
    let streams: Streams = Arc::new(Mutex::new(HashMap::new()));
    let sink = streams.clone();
    fwd.set_tap(Some(Box::new(move |t: &Tap<'_>| {
        if t.point == TapPoint::LayerOut && TARGET_LAYERS.contains(&t.layer) {
            sink.lock()
                .unwrap()
                .insert(t.layer, t.bf16(TapBuf::Streams)?);
        }
        Ok(())
    })));
    let mut a = fwd.kv.slot().unwrap();
    a.reserve(256).unwrap();
    let prompt = ids(1, 20);
    let first = fwd.prefill(&mut [(&mut a, &prompt[..])]).unwrap()[0];
    check_taps(&fwd, &streams, 20, "prefill");
    assert_eq!(ctx(&a), (20, 0));
    let next = fwd.decode(&mut [(&mut a, first)]).unwrap()[0];
    check_taps(&fwd, &streams, 1, "decode");
    assert_eq!(ctx(&a), (21, 0));
    let mut window = vec![next];
    window.extend(ids(2, 7));
    fwd.verify(&mut [(&mut a, &window[..])]).unwrap();
    check_taps(&fwd, &streams, 8, "verify");
    assert_eq!(
        ctx(&a),
        (21, 0),
        "a verify window is not context before its commit"
    );
    fwd.commit(&mut [&mut a], &[3]).unwrap();
    assert_eq!(ctx(&a), (24, 0));
    fwd.set_tap(None);
    drop(a);

    // 2. A window of 8 committed at 5, against the 5 rows committed whole and 5 decode steps.
    let p6 = ids(3, 6);
    let mut slots: Vec<GlmKv> = (0..3).map(|_| fwd.kv.slot().unwrap()).collect();
    for kv in slots.iter_mut() {
        kv.reserve(128).unwrap();
        let f = fwd.prefill(&mut [(&mut *kv, &p6[..])]).unwrap()[0];
        decode_all(&mut fwd, kv, &[f, 4242]);
    }
    let w8 = ids(4, 8);
    let [a, c, b] = &mut slots[..] else {
        unreachable!()
    };
    let va = fwd.verify(&mut [(&mut *a, &w8[..])]).unwrap();
    fwd.commit(&mut [&mut *a], &[5]).unwrap();
    let vc = fwd.verify(&mut [(&mut *c, &w8[..5])]).unwrap();
    fwd.commit(&mut [&mut *c], &[5]).unwrap();
    let sb = decode_all(&mut fwd, b, &w8[..5]);
    assert_eq!(&va[0][..5], &vc[0][..], "verify picks");
    assert_eq!(&va[0][..5], &sb[..], "verify picks against decode steps");
    let n = a.tokens();
    assert_eq!((n, b.tokens(), c.tokens()), (13, 13, 13));
    assert!(ctx(a) == (n, 0) && ctx(b) == (n, 0) && ctx(c) == (n, 0));
    let (differ, _) = ring_diff(&fwd, a, c, n);
    assert_eq!(
        differ, 0,
        "a window committed at 5 differs from the 5 rows committed whole"
    );
    let anchor = sb[4];
    assert_eq!(greedy(&mut fwd, a, anchor), greedy(&mut fwd, c, anchor));
    assert_eq!(sampled(&mut fwd, a, anchor), sampled(&mut fwd, c, anchor));
    let (differ, worst) = ring_diff(&fwd, a, b, n);
    let (pa, pb) = (greedy(&mut fwd, a, anchor), greedy(&mut fwd, b, anchor));
    eprintln!(
        "commit of 5 of 8 rows: the drafter's ring equals 5 rows committed whole, bit for bit; \
         against 5 decode steps {differ} of {} ring values differ (largest {worst:.2e} of the \
         row's RMS), drafts {:?} / {:?}",
        n * 5 * 2 * 1024,
        pa.tokens,
        pb.tokens
    );
    assert!(
        worst < 0.1,
        "a ring value is off by {worst:.2e} of its row's RMS"
    );
    drop(slots);

    // 2b. Rows of two requests in one pass reach their own rings: a prefill of 3 + 5 rows and a
    // decode batch of two, against each request alone (the forward's taps are the same bits up
    // to 8 rows; the drafter's GEMMs over 8 rows may round a value one unit apart).
    let (q3, q5) = (ids(9, 3), ids(10, 5));
    let mut four: Vec<GlmKv> = (0..4).map(|_| fwd.kv.slot().unwrap()).collect();
    let [a, b, c, d] = &mut four[..] else {
        unreachable!()
    };
    for kv in [&mut *a, &mut *b, &mut *c, &mut *d] {
        kv.reserve(64).unwrap();
    }
    let both = fwd
        .prefill(&mut [(&mut *a, &q3[..]), (&mut *b, &q5[..])])
        .unwrap();
    let alone = [
        fwd.prefill(&mut [(&mut *c, &q3[..])]).unwrap()[0],
        fwd.prefill(&mut [(&mut *d, &q5[..])]).unwrap()[0],
    ];
    assert_eq!(both, alone);
    fwd.decode(&mut [(&mut *a, both[0]), (&mut *b, both[1])])
        .unwrap();
    decode_all(&mut fwd, c, &[both[0]]);
    decode_all(&mut fwd, d, &[both[1]]);
    assert!(ctx(a) == (4, 0) && ctx(b) == (6, 0) && ctx(c) == (4, 0) && ctx(d) == (6, 0));
    let (da, wa) = ring_diff(&fwd, a, c, 4);
    let (db, wb) = ring_diff(&fwd, b, d, 6);
    assert!(
        wa < 0.1 && wb < 0.1,
        "batched rows off by {wa:.2e} / {wb:.2e} of the RMS"
    );
    let same = greedy(&mut fwd, a, 5) == greedy(&mut fwd, c, 5)
        && greedy(&mut fwd, b, 5) == greedy(&mut fwd, d, 5);
    eprintln!(
        "a prefill of 3 + 5 rows and a decode batch of 2: each request's ring against the request \
         alone, {da} and {db} values differ (largest {:.2e} of the row's RMS); drafts {}",
        wa.max(wb),
        if same { "equal" } else { "differ" }
    );
    drop(four);

    // 3. Fork and rewind against a fresh slot fed the same rows.
    let mut s = fwd.kv.slot().unwrap();
    let mut f = fwd.kv.slot().unwrap();
    let mut x = fwd.kv.slot().unwrap();
    let more = ids(5, 5);
    for kv in [&mut s, &mut f] {
        kv.reserve(128).unwrap();
        fwd.prefill(&mut [(&mut *kv, &p6[..])]).unwrap();
        decode_all(&mut fwd, kv, &more);
    }
    let l0 = s.tokens();
    let mark = GlmKv::mark(&s).unwrap();
    let t = decode_all(&mut fwd, &mut s, &ids(6, 4));
    let mut w = vec![t[3]];
    w.extend(ids(7, 7));
    fwd.verify(&mut [(&mut s, &w[..])]).unwrap();
    fwd.commit(&mut [&mut s], &[3]).unwrap();
    assert_eq!(ctx(&s), (l0 + 7, 0));
    x.reserve(128).unwrap();
    x.fork(&s, l0, &mark).unwrap();
    assert_eq!(ctx(&x), (l0, 0));
    let anchor = 777;
    let want = greedy(&mut fwd, &f, anchor);
    assert_eq!(greedy(&mut fwd, &x, anchor), want, "a forked slot's drafts");
    assert_eq!(sampled(&mut fwd, &x, anchor), sampled(&mut fwd, &f, anchor));
    assert_eq!(ring_diff(&fwd, &x, &f, l0).0, 0, "a forked slot's ring");
    s.rewind(l0, &mark).unwrap();
    assert_eq!(ctx(&s), (l0, 0));
    assert_eq!(
        greedy(&mut fwd, &s, anchor),
        want,
        "a rewound slot's drafts"
    );
    // Both go on with the same token: still the fresh slot's.
    let (ps, pf) = (
        decode_all(&mut fwd, &mut s, &[anchor]),
        decode_all(&mut fwd, &mut f, &[anchor]),
    );
    assert_eq!(ps, pf);
    assert_eq!(greedy(&mut fwd, &s, ps[0]), greedy(&mut fwd, &f, pf[0]));
    eprintln!("fork at {l0} of a slot that ran on to {}, and a rewind there: the fresh slot's drafts and ring", l0 + 7);
    drop((s, f, x, mark));

    // 4. A restore from host images starts cold.
    let mut m = ServedForward::new(fwd).unwrap();
    m.warm_rows = 4;
    let mut h = m.fwd.kv.slot().unwrap();
    h.reserve(128).unwrap();
    let p70 = ids(8, 70);
    let hn = m.fwd.prefill(&mut [(&mut h, &p70[..])]).unwrap()[0];
    let hmark = GlmKv::mark(&h).unwrap();
    let pb = h.page_bytes();
    let (mut pg0, mut pg1, mut st) = (vec![0u8; pb], vec![0u8; pb], vec![0u8; h.state_bytes()]);
    h.export_page(0, 64, &mut pg0).unwrap();
    h.export_page(64, 6, &mut pg1).unwrap();
    h.export_state(&hmark, &mut st).unwrap();
    GlmKv::sync(&h).unwrap();
    let mut r = m.fwd.kv.slot().unwrap();
    r.reserve(128).unwrap();
    r.import_page(64, &pg0).unwrap();
    r.import_page(6, &pg1).unwrap();
    r.import_state(70, &st).unwrap();
    assert_eq!(ctx(&r), (70, 70), "a restored slot's context restarts cold");
    assert!(m.cold(&r) && !m.cold(&h));
    let draft = |m: &mut ServedForward, kv: &mut GlmKv, last: u32| {
        let mut rows = [DraftRow {
            slot: kv,
            last,
            max: 7,
            pick: Pick::greedy(),
        }];
        m.draft(&mut rows).unwrap().remove(0)
    };
    assert!(
        draft(&mut m, &mut r, hn).tokens.is_empty(),
        "a cold slot drafts nothing"
    );
    assert_eq!(draft(&mut m, &mut h, hn).tokens.len(), 7);
    // The target state was restored: the same next token, then warm after warm_rows rows.
    let mut tok = (hn, hn);
    for i in 0..m.warm_rows {
        let next = |m: &mut ServedForward, kv: &mut GlmKv, t: u32| {
            let mut rows = [DecodeRow {
                slot: kv,
                token: t,
                pick: Pick::greedy(),
            }];
            m.decode(&mut rows).unwrap()[0]
        };
        tok = (next(&mut m, &mut h, tok.0), next(&mut m, &mut r, tok.1));
        assert_eq!(tok.0, tok.1, "the restored slot's step {i}");
        let rows_in = i + 1;
        let d = draft(&mut m, &mut r, tok.1);
        assert_eq!(
            d.tokens.is_empty(),
            rows_in < m.warm_rows,
            "after {rows_in} rows back in context"
        );
    }
    assert_eq!(ctx(&r), (74, 70));
    let st = m.stats;
    assert!(st.cold >= 4 && st.drafted >= 2, "{st:?}");
    eprintln!(
        "a restored slot: cold until {} rows are back in context, then drafts; the target's tokens \
         equal the source's ({:.1} s in all)",
        m.warm_rows,
        t0.elapsed().as_secs_f64()
    );
    let lim = m.limits();
    assert_eq!(lim.block, 8);
}
