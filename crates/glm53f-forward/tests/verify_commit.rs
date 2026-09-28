//! A verify window of R rows committed at k equals k serial decode steps, bit for bit
//! (feature `cuda`, real weights of layers 0-4 and the head; skips without them).
//!
//! Compared after every round: each kept row's pick and logits, the KDA states and conv
//! windows of every KDA layer, the DSA tail, and the MLA latent records and pooled keys of
//! every committed token. Rounds keep 5 of 8, 8 of 8 and 1 of 8 rows, then two requests share
//! one verify pass (3 and 5 rows, keeping 2 and 4).
#![cfg(feature = "cuda")]

mod common;

use std::collections::HashMap;

use common::*;
use glm53f_dsa::cache::{LATENT_RECORD_BYTES, PAGE_POOL_CODES_OFFSET, PAGE_POOL_SCALES_OFFSET};
use glm53f_forward::forward::{ForwardConfig, GlmForward};
use glm53f_forward::kv::GlmKv;
use glm53f_forward::shape::VOCAB;

const LAYERS: usize = 5;

/// Everything a slot holds for its committed tokens.
#[derive(PartialEq)]
struct Image {
    states: Vec<Vec<u32>>,
    convs: Vec<Vec<u16>>,
    tail: Vec<u8>,
    latents: Vec<u8>,
    pools: Vec<u8>,
}

fn image(kv: &GlmKv, kda_layers: usize) -> Image {
    let t = kv.tokens();
    let mut latents = Vec::new();
    let mut pools = Vec::new();
    for i in 0..t.div_ceil(64) {
        let b = kv.download_page_block(i, 0).unwrap();
        let n = (t - 64 * i).min(64);
        latents.extend_from_slice(&b[..n * LATENT_RECORD_BYTES]);
        let np = if 64 * i + 64 <= t { 16 } else { n / 4 };
        pools.extend_from_slice(&b[PAGE_POOL_CODES_OFFSET..PAGE_POOL_CODES_OFFSET + np * 128]);
        pools.extend_from_slice(&b[PAGE_POOL_SCALES_OFFSET..PAGE_POOL_SCALES_OFFSET + np * 4]);
    }
    Image {
        states: (0..kda_layers)
            .map(|j| {
                kv.download_state(j)
                    .unwrap()
                    .iter()
                    .map(|x| x.to_bits())
                    .collect()
            })
            .collect(),
        convs: (0..kda_layers)
            .map(|j| kv.download_conv(j).unwrap())
            .collect(),
        tail: valid_tail(kv.download_tail(0).unwrap()),
        latents,
        pools,
    }
}

/// Which parts of two images differ.
fn diff(a: &Image, b: &Image) -> String {
    let mut out = Vec::new();
    for (j, (x, y)) in a.states.iter().zip(&b.states).enumerate() {
        let n = x.iter().zip(y).filter(|(p, q)| p != q).count();
        if n > 0 {
            out.push(format!("KDA state {j}: {n} values"));
        }
    }
    for (j, (x, y)) in a.convs.iter().zip(&b.convs).enumerate() {
        let n = x.iter().zip(y).filter(|(p, q)| p != q).count();
        if n > 0 {
            out.push(format!("conv {j}: {n} values"));
        }
    }
    if a.tail != b.tail {
        out.push(format!(
            "tail: {} bytes",
            a.tail.iter().zip(&b.tail).filter(|(p, q)| p != q).count()
        ));
    }
    if a.latents != b.latents {
        let first = a
            .latents
            .iter()
            .zip(&b.latents)
            .position(|(p, q)| p != q)
            .unwrap();
        out.push(format!(
            "latents from token {}",
            first / LATENT_RECORD_BYTES
        ));
    }
    if a.pools != b.pools {
        out.push(format!(
            "pools ({} vs {} bytes)",
            a.pools.len(),
            b.pools.len()
        ));
    }
    out.join("; ")
}

/// A tail record up to its count: the slots past it hold whatever an earlier pool left there.
fn valid_tail(t: Vec<u8>) -> Vec<u8> {
    let n = u32::from_le_bytes([t[0], t[1], t[2], t[3]]) as usize;
    t[..16 + n * 512].to_vec()
}

fn bits(v: &[f32]) -> Vec<u32> {
    v.iter().map(|x| x.to_bits()).collect()
}

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

/// Serial decode steps: picks and logits.
fn serial(fwd: &mut GlmForward, kv: &mut GlmKv, toks: &[u32]) -> (Vec<u32>, Vec<Vec<u32>>) {
    let mut picks = Vec::new();
    let mut logits = Vec::new();
    for &t in toks {
        picks.extend(fwd.decode(&mut [(kv, t)]).unwrap());
        logits.push(bits(&fwd.logits(1).unwrap()));
    }
    (picks, logits)
}

#[test]
fn verify_then_commit_equals_serial_decode() {
    if !gpu_with(8.0) {
        return;
    }
    let cfg = ForwardConfig {
        max_rows: 64,
        max_verify_rows: 8,
        max_requests: 4,
        ..ForwardConfig::default()
    };
    let Some(mut su) = forward(LAYERS, cfg, HashMap::new(), 6 << 30) else {
        return;
    };
    let kl = su.fwd.shape().kda_layers;
    let fwd = &mut su.fwd;
    let prompt = ids(1, 40);
    let mut a = fwd.kv.slot().unwrap();
    let mut b = fwd.kv.slot().unwrap();
    for kv in [&mut a, &mut b] {
        kv.reserve(128).unwrap();
    }
    let pa = fwd.prefill(&mut [(&mut a, &prompt[..])]).unwrap();
    let pb = fwd.prefill(&mut [(&mut b, &prompt[..])]).unwrap();
    assert_eq!(pa, pb);
    assert!(
        image(&a, kl) == image(&b, kl),
        "two prefills of the same prompt differ"
    );

    // Rounds on one request: keep 5 of 8, all 8, 1 of 8.
    for (round, keep) in [5usize, 8, 1].into_iter().enumerate() {
        let window = ids(10 + round as u64, 8);
        let (sp, sl) = serial(fwd, &mut a, &window[..keep]);
        let vp = fwd.verify(&mut [(&mut b, &window[..])]).unwrap();
        let vl = fwd.logits(8).unwrap();
        assert_eq!(b.pending(), 8);
        fwd.commit(&mut [&mut b], &[keep]).unwrap();
        assert_eq!(
            (a.tokens(), b.tokens(), b.pending()),
            (b.tokens(), a.tokens(), 0)
        );
        assert_eq!(&vp[0][..keep], &sp[..], "round {round}: picks");
        for (j, l) in sl.iter().enumerate() {
            assert!(
                bits(&vl[j * VOCAB..(j + 1) * VOCAB]) == *l,
                "round {round}: logits of row {j} differ"
            );
        }
        let (ia, ib) = (image(&a, kl), image(&b, kl));
        assert!(
            ia == ib,
            "round {round}: the committed state differs from serial steps: {}",
            diff(&ia, &ib)
        );
        eprintln!("round {round}: verify 8 rows, keep {keep}: picks, logits and state bitwise equal to {keep} decode steps ({} tokens)", a.tokens());
    }
    // And the next step continues identically.
    let t = ids(99, 1);
    let (sa, _) = serial(fwd, &mut a, &t);
    let (sb, _) = serial(fwd, &mut b, &t);
    assert_eq!(sa, sb);
    assert!(image(&a, kl) == image(&b, kl));

    // Two requests in one verify pass (3 and 5 rows) against serial steps of two others.
    let mut c = fwd.kv.slot().unwrap();
    let mut d = fwd.kv.slot().unwrap();
    let p2 = ids(2, 37);
    for kv in [&mut c, &mut d] {
        kv.reserve(128).unwrap();
    }
    fwd.prefill(&mut [(&mut c, &p2[..])]).unwrap();
    fwd.prefill(&mut [(&mut d, &p2[..])]).unwrap();
    let (w1, w2) = (ids(20, 3), ids(21, 5));
    let (s1, l1) = serial(fwd, &mut a, &w1[..2]);
    let (s2, l2) = serial(fwd, &mut c, &w2[..4]);
    let vp = fwd
        .verify(&mut [(&mut b, &w1[..]), (&mut d, &w2[..])])
        .unwrap();
    let vl = fwd.logits(8).unwrap();
    fwd.commit(&mut [&mut b, &mut d], &[2, 4]).unwrap();
    assert_eq!((&vp[0][..2], &vp[1][..4]), (&s1[..], &s2[..]));
    for (j, l) in l1.iter().enumerate() {
        assert!(bits(&vl[j * VOCAB..(j + 1) * VOCAB]) == *l);
    }
    for (j, l) in l2.iter().enumerate() {
        assert!(bits(&vl[(3 + j) * VOCAB..(4 + j) * VOCAB]) == *l);
    }
    assert!(
        image(&a, kl) == image(&b, kl) && image(&c, kl) == image(&d, kl),
        "two-request verify"
    );
    eprintln!(
        "two requests in one verify pass (3 + 5 rows, keep 2 and 4): bitwise equal to serial steps"
    );

    // A short prefill (8 rows or fewer runs the same kernels) equals serial steps too.
    let t8 = ids(30, 6);
    fwd.prefill(&mut [(&mut b, &t8[..])]).unwrap();
    let pl = fwd.logits(1).unwrap();
    let (_, sl) = serial(fwd, &mut a, &t8);
    assert!(bits(&pl) == *sl.last().unwrap());
    assert!(
        image(&a, kl) == image(&b, kl),
        "a 6-row prefill differs from 6 decode steps"
    );
    eprintln!("a 6-row prefill: bitwise equal to 6 decode steps");

    // A decode batch of two requests equals each request alone (a = b and c = d here).
    let (x, y) = (ids(40, 1)[0], ids(41, 1)[0]);
    let batched = fwd.decode(&mut [(&mut a, x), (&mut c, y)]).unwrap();
    let bl = fwd.logits(2).unwrap();
    let (sb, lb) = serial(fwd, &mut b, &[x]);
    let (sd, ld) = serial(fwd, &mut d, &[y]);
    assert_eq!(batched, vec![sb[0], sd[0]]);
    assert!(bits(&bl[..VOCAB]) == lb[0] && bits(&bl[VOCAB..]) == ld[0]);
    assert!(
        image(&a, kl) == image(&b, kl) && image(&c, kl) == image(&d, kl),
        "a decode batch differs from single steps"
    );
    // Two short prompts in one prefill pass (5 + 3 rows) equal each prompt alone.
    let (q1, q2) = (ids(50, 5), ids(51, 3));
    let pp = fwd
        .prefill(&mut [(&mut a, &q1[..]), (&mut c, &q2[..])])
        .unwrap();
    let p1 = fwd.prefill(&mut [(&mut b, &q1[..])]).unwrap();
    let p2 = fwd.prefill(&mut [(&mut d, &q2[..])]).unwrap();
    assert_eq!(pp, vec![p1[0], p2[0]]);
    assert!(
        image(&a, kl) == image(&b, kl) && image(&c, kl) == image(&d, kl),
        "a batched prefill differs from single prefills"
    );
    eprintln!("a decode batch of 2 and a prefill batch of 2 short prompts: bitwise equal to each request alone");
}
