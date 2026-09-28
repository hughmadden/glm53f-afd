//! The host RAM tier over the mock slot's page and state export: page sharing, exact restores,
//! the lookup rules, and the eviction order under pressure.

mod common;

use common::*;
use glm53f_coordinator::sampling::After;
use glm53f_coordinator::{HostCache, HostTierConfig, Kind, KvSlot, Token};

fn tier(pages: usize, states: usize) -> HostCache {
    HostCache::new(HostTierConfig { pages, states, page_tokens: PAGE, page_bytes: PAGE * 4, state_bytes: 16, min_tokens: 16, pin: false })
        .expect("tier")
}

fn greedy_after(t: Token) -> After {
    After { greedy: Some(t as usize), logits: None }
}

/// A slot holding `prefix` then `rest`, with marks at both ends.
fn conversation(dev: &Dev, prefix: &[Token], rest: &[Token]) -> (MockSlot, MockMark, MockMark) {
    let mut s = MockSlot::new(0, dev, 0);
    s.fill(prefix);
    let m1 = s.mark().unwrap();
    s.fill(rest);
    let m2 = s.mark().unwrap();
    (s, m1, m2)
}

#[test]
fn prompt_and_turn_snapshots_share_pages_and_restore_exactly() {
    let dev = device(1 << 30);
    let mut hc = tier(32, 8);
    let (px, rest) = (prompt(1, 24), prompt(2, 20));
    let (s, m_prompt, m_turn) = conversation(&dev, &px, &rest);
    let tx: Vec<Token> = px.iter().chain(&rest).copied().collect();
    hc.capture(&s, &px, &greedy_after(3), Kind::Prompt, &m_prompt).unwrap();
    hc.capture(&s, &tx, &greedy_after(4), Kind::Turn, &m_turn).unwrap();
    // 24 tokens: three full pages; 44: five full pages (three shared) and a 4-token tail.
    assert_eq!(hc.pages_held(), 5);
    assert_eq!(hc.stats.pages_written, 5);
    assert_eq!(hc.free_slots(), (32 - 5 - 1, 8 - 2));
    // Restores rebuild the rows and the positional state (the mock checks they agree).
    for (want, after) in [(&tx, 4usize), (&px, 3usize)] {
        let q: Vec<Token> = want.iter().copied().chain([9, 9]).collect();
        let (id, n) = hc.lookup(&q).expect("a snapshot prefixes the query");
        assert_eq!(n, want.len());
        let mut fresh = MockSlot::new(1, &dev, 0);
        let (m, a, _) = hc.restore(id, &mut fresh).unwrap();
        assert_eq!((m, a.greedy), (want.len(), Some(after)));
        assert_eq!(&fresh.toks, want);
        fresh.check().unwrap();
    }
    assert_eq!(hc.stats.restores, 2);
    // An identical capture refreshes; a short one is not stored.
    hc.capture(&s, &tx, &greedy_after(4), Kind::Turn, &m_turn).unwrap();
    hc.capture(&s, &px[..10], &greedy_after(4), Kind::Prompt, &m_prompt).unwrap();
    assert_eq!(hc.len(), 2);
    // A snapshot of the whole prompt that cannot serve the request gives way to a shorter one:
    // a sampled request needs kept logits, and the turn snapshot has none.
    assert_eq!(hc.lookup_for(&tx, true).map(|x| x.1), Some(24));
    assert_eq!(hc.lookup_for(&tx, false).map(|x| x.1), Some(44));
}

/// Page pressure evicts the least recently used snapshot first. A prompt snapshot shares its
/// pages with its turn snapshot, so evicting it frees no page; the next victim is the stale turn
/// snapshot, not a fresh prompt snapshot of another conversation (the order mimo26f-afd v1.1.1
/// fixed: under "every prompt snapshot first", the fresh prompt snapshot went before the stale turn
/// snapshot, and an exact repeat of a recent prompt prefilled from cold).
#[test]
fn page_pressure_keeps_a_fresh_prompt_snapshot() {
    let dev = device(1 << 30);
    let mut hc = tier(10, 8);
    // Conversation X: prompt 24 tokens (3 pages), turn 40 tokens (5 pages, 3 shared).
    let (sx, mxp, mxt) = conversation(&dev, &prompt(10, 24), &prompt(11, 16));
    let px = sx.toks[..24].to_vec();
    let tx = sx.toks.clone();
    hc.capture(&sx, &px, &greedy_after(1), Kind::Prompt, &mxp).unwrap();
    hc.capture(&sx, &tx, &greedy_after(1), Kind::Turn, &mxt).unwrap();
    // Conversation Y's prompt (4 pages), captured later: 9 of 10 pages in use.
    let (sy, myp, _) = conversation(&dev, &prompt(12, 32), &[]);
    let py = sy.toks.clone();
    hc.capture(&sy, &py, &greedy_after(2), Kind::Prompt, &myp).unwrap();
    assert_eq!(hc.free_slots().0, 1);
    // Conversation Z's prompt needs 3 pages: X's prompt snapshot (oldest) frees none, then X's
    // turn snapshot frees 5.
    let (sz, mzp, _) = conversation(&dev, &prompt(13, 24), &[]);
    let pz = sz.toks.clone();
    hc.capture(&sz, &pz, &greedy_after(3), Kind::Prompt, &mzp).unwrap();
    let held: Vec<(usize, Kind)> = hc.snapshots().into_iter().map(|(t, k)| (t.len(), k)).collect();
    assert_eq!(held, vec![(32, Kind::Prompt), (24, Kind::Prompt)], "Y and Z kept, X evicted");
    assert_eq!(hc.stats.evicted, 2);
    assert_eq!(hc.lookup(&py).map(|x| x.1), Some(32), "the fresh prompt snapshot still serves a repeat");
    assert_eq!(hc.lookup(&tx), None);
    // A recently used snapshot outlives an older one: touch Y, then pressure evicts Z first.
    let mut fresh = MockSlot::new(2, &dev, 0);
    hc.restore(hc.lookup(&py).unwrap().0, &mut fresh).unwrap();
    let (sw, mwp, _) = conversation(&dev, &prompt(14, 40), &[]);
    let pw = sw.toks.clone();
    hc.capture(&sw, &pw, &greedy_after(4), Kind::Prompt, &mwp).unwrap();
    let held: Vec<usize> = hc.snapshots().into_iter().map(|(t, _)| t.len()).collect();
    assert_eq!(held, vec![32, 40], "Z (older use) went, Y (just restored) stayed");
}

/// A conversation's prompt snapshot, stored just before its turn snapshot (which shares its
/// pages), goes first when a state slot is needed. (The tie rule itself, prompt before turn at
/// equal use, is pinned by the unit test `eviction_is_least_recently_used_first`.)
#[test]
fn a_conversation_s_prompt_snapshot_goes_before_its_turn_snapshot() {
    let dev = device(1 << 30);
    let mut hc = tier(64, 2);
    let (s, mp, mt) = conversation(&dev, &prompt(20, 16), &prompt(21, 16));
    let p = s.toks[..16].to_vec();
    hc.capture(&s, &p, &greedy_after(1), Kind::Prompt, &mp).unwrap();
    hc.capture(&s, &s.toks.clone(), &greedy_after(1), Kind::Turn, &mt).unwrap();
    // Both states in use: a third snapshot evicts one; the older (prompt) goes.
    let (s2, m2, _) = conversation(&dev, &prompt(22, 16), &[]);
    hc.capture(&s2, &s2.toks.clone(), &greedy_after(1), Kind::Prompt, &m2).unwrap();
    let kinds: Vec<(usize, Kind)> = hc.snapshots().into_iter().map(|(t, k)| (t.len(), k)).collect();
    assert_eq!(kinds, vec![(32, Kind::Turn), (16, Kind::Prompt)]);
}
