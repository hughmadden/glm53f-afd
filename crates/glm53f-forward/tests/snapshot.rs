//! Marks, rewinds, forks with copy-on-write pages, and host images of pages and marks
//! (feature `cuda`, real weights of layers 0-4 and the head; skips without them). The routed
//! experts return zeros here: these checks are about the KV, not the experts.
//!
//! - A restore from host images (pages, then the mark's state) continues bit for bit like the
//!   slot it was exported from; images of the same history are byte-identical.
//! - A rewind to a mark and a fork from a mark continue bit for bit like the original run.
//! - A fork shares full pages; a slot that rewinds into a shared page gets its own copy, and
//!   the other slot's page is unchanged.
#![cfg(feature = "cuda")]

mod common;

use std::collections::HashMap;

use common::*;
use glm53f_forward::forward::{ForwardConfig, GlmForward};
use glm53f_forward::kv::GlmKv;

const LAYERS: usize = 5;

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

/// Decode `toks`: each step's pick and logits bits.
fn run(fwd: &mut GlmForward, kv: &mut GlmKv, toks: &[u32]) -> Vec<(u32, Vec<u32>)> {
    toks.iter()
        .map(|&t| {
            let p = fwd.decode(&mut [(kv, t)]).unwrap()[0];
            (
                p,
                fwd.logits(1).unwrap().iter().map(|x| x.to_bits()).collect(),
            )
        })
        .collect()
}

/// Host images of a slot at a mark: its pages, then the mark's state.
fn export(kv: &GlmKv, mark: &glm53f_forward::kv::KvMark) -> (Vec<(usize, Vec<u8>)>, Vec<u8>) {
    let t = mark.tokens;
    let mut pages = Vec::new();
    for first in (0..t).step_by(64) {
        let n = (t - first).min(64);
        let mut img = vec![0u8; kv.page_bytes()];
        kv.export_page(first, n, &mut img).unwrap();
        pages.push((n, img));
    }
    let mut state = vec![0u8; kv.state_bytes()];
    kv.export_state(mark, &mut state).unwrap();
    (pages, state)
}

#[test]
fn marks_rewinds_forks_and_host_images() {
    if !gpu_with(6.0) {
        return;
    }
    let cfg = ForwardConfig {
        max_rows: 64,
        max_verify_rows: 8,
        max_requests: 4,
        ..ForwardConfig::default()
    };
    let Some(mut su) = forward(LAYERS, cfg, HashMap::new(), 0) else {
        return;
    };
    let fwd = &mut su.fwd;
    let free0 = fwd.kv.free_pages();
    let mut a = fwd.kv.slot().unwrap();
    a.reserve(128).unwrap();
    // 60 prompt tokens, a mark, 10 more, 2 steps, a mark at 72.
    let p = ids(3, 70);
    fwd.prefill(&mut [(&mut a, &p[..60])]).unwrap();
    let m60 = a.mark().unwrap();
    fwd.prefill(&mut [(&mut a, &p[60..])]).unwrap();
    run(fwd, &mut a, &ids(4, 2));
    let m72 = a.mark().unwrap();
    assert_eq!((m60.tokens, m72.tokens, a.tokens()), (60, 72, 72));
    assert_eq!(m72.bytes(), a.state_bytes());
    let (pages, state) = export(&a, &m72);
    assert_eq!(pages.len(), 2);

    // A restore into a fresh slot continues exactly like the original.
    let mut b = fwd.kv.slot().unwrap();
    b.reserve(128).unwrap();
    for (n, img) in &pages {
        b.import_page(*n, img).unwrap();
    }
    b.import_state(72, &state).unwrap();
    let mb = b.mark().unwrap();
    let (pages_b, state_b) = export(&b, &mb);
    assert!(
        pages_b == pages && state_b == state,
        "re-exported images differ"
    );
    let t1 = ids(5, 3);
    let ra = run(fwd, &mut a, &t1);
    let rb = run(fwd, &mut b, &t1);
    assert!(ra == rb, "a restored slot continues differently");
    eprintln!(
        "restore from host images ({} pages of {} B, state {} B): continues bit for bit",
        pages.len(),
        pages[0].1.len(),
        state.len()
    );

    // Rewind to the mark at 72 and replay the same steps.
    a.rewind(72, &m72).unwrap();
    assert_eq!(a.tokens(), 72);
    let ra2 = run(fwd, &mut a, &t1);
    assert!(ra2 == ra, "a rewound slot continues differently");
    eprintln!("rewind to a mark: continues bit for bit");

    // Fork at 72 from `a` (now at 75): page 0 shared, page 1 (tokens 64..71) copied.
    let mut c = fwd.kv.slot().unwrap();
    c.fork(&a, 72, &m72).unwrap();
    assert_eq!(
        c.page_table()[0],
        a.page_table()[0],
        "the full page is shared"
    );
    assert_ne!(
        c.page_table()[1],
        a.page_table()[1],
        "the partial page is copied"
    );
    let rc = run(fwd, &mut c, &t1);
    assert!(rc == ra, "a fork continues differently");
    eprintln!(
        "fork at a mark: shares the full page, copies the partial one, continues bit for bit"
    );

    // `a` rewinds to 60, inside the shared page: it gets its own copy; `c`'s page stays.
    let c_page0 = c.download_page_block(0, 0).unwrap();
    let shared = a.page_table()[0];
    a.rewind(60, &m60).unwrap();
    assert_ne!(
        a.page_table()[0],
        shared,
        "rewinding into a shared page copies it"
    );
    assert_eq!(c.page_table()[0], shared);
    let alt = ids(6, 6);
    let ra3 = run(fwd, &mut a, &alt);
    assert_eq!(
        c.download_page_block(0, 0).unwrap(),
        c_page0,
        "the other slot's page changed"
    );
    // `a`'s run after the rewind equals a fresh prompt of those 60 tokens followed by the steps.
    let mut d = fwd.kv.slot().unwrap();
    d.reserve(128).unwrap();
    fwd.prefill(&mut [(&mut d, &p[..60])]).unwrap();
    let rd = run(fwd, &mut d, &alt);
    assert!(
        rd == ra3,
        "a rewound slot differs from a fresh prompt of the same tokens"
    );
    eprintln!("rewind into a shared page: copied first, the fork's page unchanged, equal to a fresh prompt");

    // Marks inside a verify window are refused; pages go back to the pool.
    fwd.verify(&mut [(&mut d, &ids(7, 3)[..])]).unwrap();
    assert!(d.mark().is_err());
    fwd.commit(&mut [&mut d], &[3]).unwrap();
    drop((a, b, c, d));
    assert_eq!(fwd.kv.free_pages(), free0, "pages leaked");
}
