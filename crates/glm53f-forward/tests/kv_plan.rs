//! The device KV's byte accounting against the memory planner, and the page bookkeeping
//! (copy-on-write, forks, rewinds). CPU only.

use std::path::PathBuf;

use glm53f_forward::kvplan::{KvLayout, PageAlloc, PageCopy, SlotPages, PAGE};
use glm53f_forward::shape::ModelShape;
use glm53f_model::config::{DraftConfig, ModelConfig};
use glm53f_model::planner::{KvGeometry, KvPrecision, SlotState};

fn data(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../glm53f-model/tests/data")
        .join(name)
}

fn configs() -> (ModelConfig, DraftConfig) {
    let cfg = ModelConfig::load(&data("zai-org_GLM-5.3-Flash.config.json")).unwrap();
    let draft = DraftConfig::load(&data("incoai_GLM-5.3-Flash-DFlash2.config.json")).unwrap();
    (cfg, draft)
}

#[test]
fn accounting_matches_the_planner() {
    let (cfg, draft) = configs();
    let shape = ModelShape::full(&cfg.text).unwrap();
    assert_eq!((shape.kda_layers, shape.dsa_layers), (34, 11));
    let l = KvLayout::new(&shape, Some(&draft));
    let geo = KvGeometry::new(&cfg.text, KvPrecision::Fp8);
    let slot = SlotState::new(&cfg.text, Some(&draft));
    // A page: 64 tokens of every DSA layer = 64 x 6,171 B.
    assert_eq!(l.page_bytes as u64, geo.page_bytes());
    assert_eq!(l.page_bytes, 394_944);
    for tokens in [1usize, 63, 64, 65, 4096, 262_144, 1_048_576] {
        assert_eq!(
            l.paged_bytes(tokens) as u64,
            geo.request_bytes(tokens as u64),
            "{tokens} tokens"
        );
    }
    // Per slot: the KDA state and the conv windows as the planner counts them, the draft KV,
    // and the DSA tails (the planner does not count them: 11 x 1,552 B = 17,072 B per slot).
    assert_eq!(l.kda_state_bytes() as u64, slot.kda_state);
    assert_eq!(l.conv_bytes() as u64, slot.conv_state);
    assert_eq!(l.draft_kv_bytes as u64, slot.draft_kv);
    assert_eq!(l.tails_bytes(), 17_072);
    assert_eq!(l.mark_bytes() as u64, slot.kda_snapshot() + 17_072);
    assert_eq!(l.slot_fixed_bytes() as u64, slot.total() + 17_072);
    eprintln!(
        "page {} B; slot: KDA state {:.1} MiB + conv {:.2} MiB + tails {} B + draft KV {:.1} MiB; mark (host image) {:.2} MiB",
        l.page_bytes,
        l.kda_state_bytes() as f64 / 1048576.0,
        l.conv_bytes() as f64 / 1048576.0,
        l.tails_bytes(),
        l.draft_kv_bytes as f64 / 1048576.0,
        l.mark_bytes() as f64 / 1048576.0
    );
    // A mark on the device: pool pages, each part (KDA states, conv windows, tails) from a page
    // of its own: 362 + 13 + 1 = 376 pages, 0.6% over the state it holds.
    let regions = l.mark_regions();
    assert_eq!(regions.map(|r| r.2), [362, 13, 1]);
    assert_eq!(
        regions.map(|r| r.0),
        [0, l.kda_state_bytes(), l.kda_state_bytes() + l.conv_bytes()]
    );
    assert_eq!(regions.iter().map(|r| r.1).sum::<usize>(), l.mark_bytes());
    assert_eq!(l.mark_pages(), 376);
    assert!(l.mark_pages() * l.page_bytes >= l.mark_bytes());
    // The layer prefix the golden tests run: layers 0-4 (KDA 0, 1, 2, 4; DSA 3).
    let s5 = ModelShape::new(&cfg.text, 5).unwrap();
    assert_eq!((s5.kda_layers, s5.dsa_layers), (4, 1));
    assert_eq!(s5.kda_index, vec![Some(0), Some(1), Some(2), None, Some(3)]);
    assert_eq!(KvLayout::new(&s5, None).page_bytes, 35_904);
    assert_eq!(KvLayout::new(&s5, None).mark_pages(), 468 + 17 + 1);
}

fn check_refs(alloc: &PageAlloc, slots: &[&SlotPages]) {
    let mut refs = vec![0u32; alloc.total()];
    for s in slots {
        for &p in &s.pages {
            refs[p as usize] += 1;
        }
    }
    for (p, &r) in refs.iter().enumerate() {
        assert_eq!(alloc.refs(p as u32), r, "page {p}");
    }
    let used = refs.iter().filter(|&&r| r > 0).count();
    assert_eq!(alloc.free(), alloc.total() - used);
}

#[test]
fn pages_grow_share_and_copy_on_write() {
    let mut a = PageAlloc::new(16);
    let mut s1 = SlotPages::default();
    // Growth: fresh pages, table entries from 0.
    let ch = s1.reserve(&mut a, 2).unwrap();
    assert_eq!((ch.copies.len(), ch.first_changed), (0, Some(0)));
    let ch = s1.prepare_write(&mut a, 100, 140).unwrap();
    assert_eq!((s1.len(), ch.first_changed), (3, Some(2)));
    check_refs(&a, &[&s1]);

    // A fork at 150 tokens: pages 0 and 1 shared, page 2 (tokens 128..149) copied.
    let mut s2 = SlotPages::default();
    let ch = s2.fork(&mut a, &s1, 150).unwrap();
    assert_eq!(
        ch.copies,
        vec![PageCopy {
            src: s1.pages[2],
            dst: s2.pages[2]
        }]
    );
    assert_eq!(&s2.pages[..2], &s1.pages[..2]);
    assert_ne!(s2.pages[2], s1.pages[2]);
    assert!(s1.is_shared(&a, 0) && s1.is_shared(&a, 1) && !s1.is_shared(&a, 2));
    check_refs(&a, &[&s1, &s2]);

    // Writing past the shared pages copies nothing.
    let ch = s2.prepare_write(&mut a, 150, 200).unwrap();
    assert!(ch.copies.is_empty());
    // Writing into a shared page copies it first; the other slot keeps the original.
    let orig = s1.pages[1];
    let ch = s1.prepare_write(&mut a, 70, 72).unwrap();
    assert_eq!(
        ch.copies,
        vec![PageCopy {
            src: orig,
            dst: s1.pages[1]
        }]
    );
    assert_eq!(s2.pages[1], orig);
    assert!(!s1.is_shared(&a, 1) && !s2.is_shared(&a, 1));
    check_refs(&a, &[&s1, &s2]);

    // Rewind into a shared page: the page holding kept rows is copied, later shared pages
    // are replaced by fresh ones (their rows are dead).
    let mut s3 = SlotPages::default();
    s3.fork(&mut a, &s2, 192).unwrap();
    assert!(s3.pages[..3].iter().zip(&s2.pages).all(|(x, y)| x == y));
    let (p0, p1, p2) = (s3.pages[0], s3.pages[1], s3.pages[2]);
    let ch = s3.rewind(&mut a, 40).unwrap();
    assert_eq!(
        ch.copies,
        vec![PageCopy {
            src: p0,
            dst: s3.pages[0]
        }]
    );
    assert!(s3.pages[0] != p0 && s3.pages[1] != p1 && s3.pages[2] != p2);
    assert_eq!(ch.first_changed, Some(0));
    check_refs(&a, &[&s1, &s2, &s3]);
    // A page-aligned rewind copies nothing.
    let mut s4 = SlotPages::default();
    s4.fork(&mut a, &s2, 128).unwrap();
    let ch = s4.rewind(&mut a, 64).unwrap();
    assert!(ch.copies.is_empty());
    assert_eq!(s4.pages[0], s2.pages[0]);
    check_refs(&a, &[&s1, &s2, &s3, &s4]);

    // Truncation gives pages back; a full pool refuses without changing anything.
    s4.truncate(&mut a, 0);
    s3.truncate(&mut a, 1);
    check_refs(&a, &[&s1, &s2, &s3, &s4]);
    let before = (s1.clone(), a.free());
    let too_many = 64 * (a.free() + s1.len() + 1);
    let err = s1.prepare_write(&mut a, 0, too_many).unwrap_err();
    assert!(err.to_string().contains("free pages"), "{err}");
    assert_eq!((s1.clone(), a.free()), before);
    check_refs(&a, &[&s1, &s2, &s3, &s4]);
    assert_eq!(PAGE, 64);
}
