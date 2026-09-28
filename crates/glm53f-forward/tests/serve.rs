//! The serving shell's traits over this forward (features `coordinator` and `cuda`; real weights
//! of layers 0-4 and the head; skips without them): prefill, decode, verify and commit through
//! `ModelForward`, with greedy, sampled and masked picks, against the shell's CPU reference of
//! its selection contract.
#![cfg(feature = "coordinator")]

mod common;

use std::collections::HashMap;

use common::*;
use glm53f_coordinator::model::{DecodeRow, KvSlot, ModelForward, Pick, Segment, Window};
use glm53f_coordinator::sampling::{select_pick, Mask, Sampling};
use glm53f_forward::forward::ForwardConfig;
use glm53f_forward::serve::ServedForward;
use glm53f_forward::shape::{SAMPLE_VOCAB, VOCAB};

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

#[test]
fn model_forward_through_the_shell_traits() {
    if !gpu_with(6.0) {
        return;
    }
    let cfg = ForwardConfig {
        max_rows: 64,
        max_verify_rows: 8,
        max_requests: 4,
        ..ForwardConfig::default()
    };
    let Some(su) = forward(5, cfg, HashMap::new(), 0) else {
        return;
    };
    let mut m = ServedForward::new(su.fwd).unwrap();
    let lim = m.limits();
    assert_eq!(
        (lim.vocab, lim.sample_vocab, lim.block),
        (VOCAB, SAMPLE_VOCAB, 0)
    );
    // Admission counts the page pool's free pages (the pool is allocated up front): two slots of
    // 128 tokens take 4 of its 64-token pages.
    let page = m.fwd.kv.config().layout.page_bytes;
    let free = m.free_bytes().unwrap();
    assert_eq!(free, m.fwd.kv.free_pages() * page);
    let mut a = m.fwd.kv.slot().unwrap();
    let mut b = m.fwd.kv.slot().unwrap();
    KvSlot::reserve(&mut a, 128).unwrap();
    KvSlot::reserve(&mut b, 128).unwrap();
    assert_eq!(m.free_bytes().unwrap(), free - 4 * page);

    // Two prompts in one pass: one greedy, one sampled; both keep their logits.
    let sampled = Sampling::new(0.8, 0.95, 0, 0.0, Some(7)).unwrap().unwrap();
    let (p1, p2) = (ids(1, 20), ids(2, 13));
    let outs = {
        let mut segs = [
            Segment {
                slot: &mut a,
                tokens: &p1,
                images: &[],
                pick: Pick::greedy(),
                keep_logits: true,
            },
            Segment {
                slot: &mut b,
                tokens: &p2,
                images: &[],
                pick: Pick::at(Some(sampled), 0),
                keep_logits: true,
            },
        ];
        m.prefill(&mut segs).unwrap()
    };
    let (la, lb) = (
        outs[0].logits.as_ref().unwrap(),
        outs[1].logits.as_ref().unwrap(),
    );
    assert_eq!(outs[0].next, select_pick(la, SAMPLE_VOCAB, &Pick::greedy()));
    assert_eq!(
        outs[1].next,
        select_pick(lb, SAMPLE_VOCAB, &Pick::at(Some(sampled), 0))
    );
    assert_eq!((KvSlot::tokens(&a), KvSlot::tokens(&b)), (20, 13));

    // Decode: a greedy row and a masked row.
    let allowed = [11u32, 4242, 99_999];
    let mask = Mask::from_allowed(SAMPLE_VOCAB, allowed);
    let next = {
        let mut rows = [
            DecodeRow {
                slot: &mut a,
                token: outs[0].next,
                pick: Pick::greedy(),
            },
            DecodeRow {
                slot: &mut b,
                token: outs[1].next,
                pick: Pick {
                    draw: None,
                    mask: Some(mask.clone()),
                },
            },
        ];
        m.decode(&mut rows).unwrap()
    };
    assert!(allowed.contains(&next[1]), "a masked pick outside its mask");

    // Verify and commit through the traits.
    let w = ids(3, 4);
    let picks = vec![Pick::greedy(); 4];
    let v = {
        let mut win = [Window {
            slot: &mut a,
            tokens: &w,
            picks: &picks,
        }];
        m.verify(&mut win).unwrap()
    };
    assert_eq!(v[0].len(), 4);
    assert_eq!(KvSlot::pending(&a), 4);
    m.commit(&mut [&mut a], &[2]).unwrap();
    assert_eq!((KvSlot::tokens(&a), KvSlot::pending(&a)), (23, 0));
    eprintln!("ModelForward through the shell: prefill (greedy and sampled, logits kept), decode (greedy and masked), verify and commit");
}
