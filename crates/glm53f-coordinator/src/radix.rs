//! A radix index over token ids: which retained snapshot points a new prompt can resume from.
//!
//! Keys are token sequences (image rows included: they are ids at or above
//! [`crate::model::IMAGE_ID_BASE`] that encode the image's identity, so two prompts share a key
//! prefix only when their images are the same). Each entry is stored at its exact key; several
//! entries may share one key. A [`RadixIndex::lookup`] walks the query once and answers two
//! questions:
//!
//! - **hit**: the longest stored key that is a whole prefix of the query (with an entry the
//!   caller accepts). A snapshot point is only usable at exactly its length, because the
//!   positional state (KDA) exists there and nowhere else; so the hit is exact, token by token.
//! - **shared**: how much of the query any stored key shares, rounded down to the index's
//!   granularity. For GLM-5.3-Flash the granularity is the indexer's pool of 4 tokens: the
//!   appendable KV (MLA latents and pooled keys) of that many tokens exists somewhere. The gap
//!   between `shared` and the hit is prefill a periodic checkpoint could have saved; the
//!   scheduler counts it (design decision D4 adds such checkpoints when branch reuse shows up).
//!
//! The tree is a compressed trie: an edge holds a run of tokens, children are keyed by their
//! first token, and removal prunes empty leaves and merges single-child chains. Insert, remove
//! and lookup cost O(key length) token comparisons, instead of the source's scan of every
//! retained slot's history per admission.

use std::collections::BTreeMap;

use crate::model::Token;

struct Node<V> {
    /// The tokens from the parent to this node (empty at the root only).
    edge: Vec<Token>,
    /// Entries whose key ends here, in insertion order.
    values: Vec<V>,
    /// Children by the first token of their edge.
    children: BTreeMap<Token, Node<V>>,
}

impl<V> Node<V> {
    fn new(edge: Vec<Token>) -> Self {
        Node { edge, values: Vec::new(), children: BTreeMap::new() }
    }
}

fn common_len(a: &[Token], b: &[Token]) -> usize {
    a.iter().zip(b).take_while(|(x, y)| x == y).count()
}

/// The result of [`RadixIndex::lookup`].
#[derive(Debug)]
pub struct Match<'a, V> {
    /// The longest prefix of the query that some stored key shares, rounded down to the
    /// granularity.
    pub shared: usize,
    /// The longest stored key that is a whole prefix of the query and holds an accepted entry:
    /// its length and those entries, in insertion order.
    pub hit: Option<(usize, Vec<&'a V>)>,
}

/// A radix index from token sequences to entries of type `V`.
pub struct RadixIndex<V> {
    granularity: usize,
    root: Node<V>,
    entries: usize,
}

impl<V> RadixIndex<V> {
    /// An empty index reporting shared prefixes in whole units of `granularity` tokens (at
    /// least 1).
    pub fn new(granularity: usize) -> Self {
        RadixIndex { granularity: granularity.max(1), root: Node::new(Vec::new()), entries: 0 }
    }

    pub fn granularity(&self) -> usize {
        self.granularity
    }

    /// Entries stored.
    pub fn len(&self) -> usize {
        self.entries
    }

    pub fn is_empty(&self) -> bool {
        self.entries == 0
    }

    /// Store `value` at `key` (beside any entries already there).
    pub fn insert(&mut self, key: &[Token], value: V) {
        let mut node = &mut self.root;
        let mut rest = key;
        loop {
            let Some(&first) = rest.first() else {
                node.values.push(value);
                self.entries += 1;
                return;
            };
            let child = match node.children.entry(first) {
                std::collections::btree_map::Entry::Vacant(slot) => {
                    let mut leaf = Node::new(rest.to_vec());
                    leaf.values.push(value);
                    slot.insert(leaf);
                    self.entries += 1;
                    return;
                }
                std::collections::btree_map::Entry::Occupied(o) => o.into_mut(),
            };
            let common = common_len(&child.edge, rest);
            if common < child.edge.len() {
                // Split the edge where the key leaves it.
                let tail = child.edge.split_off(common);
                let lower = Node {
                    edge: tail,
                    values: std::mem::take(&mut child.values),
                    children: std::mem::take(&mut child.children),
                };
                child.children.insert(lower.edge[0], lower);
            }
            node = child;
            rest = &rest[common..];
        }
    }

    /// Remove and return the first entry stored at exactly `key` for which `pred` holds.
    pub fn remove(&mut self, key: &[Token], mut pred: impl FnMut(&V) -> bool) -> Option<V> {
        let v = remove_at(&mut self.root, key, &mut pred)?;
        self.entries -= 1;
        Some(v)
    }

    /// The entries stored at exactly `key`.
    pub fn get(&self, key: &[Token]) -> &[V] {
        let mut node = &self.root;
        let mut rest = key;
        while let Some(&first) = rest.first() {
            let Some(child) = node.children.get(&first) else { return &[] };
            if !rest.starts_with(&child.edge) {
                return &[];
            }
            rest = &rest[child.edge.len()..];
            node = child;
        }
        &node.values
    }

    /// Walk `query`: the longest stored key that prefixes it with an entry `accept(len, entry)`
    /// holds for, and the shared prefix (see [`Match`]).
    pub fn lookup<'a>(&'a self, query: &[Token], mut accept: impl FnMut(usize, &V) -> bool) -> Match<'a, V> {
        let mut node = &self.root;
        let mut pos = 0;
        let mut hit = None;
        let shared = loop {
            let ok: Vec<&V> = node.values.iter().filter(|v| accept(pos, v)).collect();
            if !ok.is_empty() {
                hit = Some((pos, ok));
            }
            let Some(&t) = query.get(pos) else { break pos };
            let Some(child) = node.children.get(&t) else { break pos };
            let common = common_len(&child.edge, &query[pos..]);
            if common < child.edge.len() {
                break pos + common;
            }
            pos += common;
            node = child;
        };
        Match { shared: shared / self.granularity * self.granularity, hit }
    }
}

/// Remove below `node` (at `rest`), pruning empty leaves and merging single-child chains on the
/// way back up. The root is never pruned or merged.
fn remove_at<V>(node: &mut Node<V>, rest: &[Token], pred: &mut dyn FnMut(&V) -> bool) -> Option<V> {
    let Some(&first) = rest.first() else {
        let i = node.values.iter().position(&mut *pred)?;
        return Some(node.values.remove(i));
    };
    let child = node.children.get_mut(&first)?;
    if !rest.starts_with(&child.edge) {
        return None;
    }
    let n = child.edge.len();
    let v = remove_at(child, &rest[n..], pred)?;
    if child.values.is_empty() {
        match child.children.len() {
            0 => {
                node.children.remove(&first);
            }
            1 => {
                let (_, only) = child.children.pop_first().expect("one child");
                child.edge.extend(only.edge);
                child.values = only.values;
                child.children = only.children;
            }
            _ => {}
        }
    }
    Some(v)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn keys(ix: &RadixIndex<u32>, q: &[Token]) -> (usize, Option<(usize, Vec<u32>)>) {
        let m = ix.lookup(q, |_, _| true);
        (m.shared, m.hit.map(|(n, v)| (n, v.into_iter().copied().collect())))
    }

    #[test]
    fn splits_edges_and_finds_the_longest_whole_prefix() {
        let mut ix = RadixIndex::new(1);
        ix.insert(&[10, 20, 30, 40], 4);
        ix.insert(&[10, 20], 2);
        ix.insert(&[10, 20, 50], 3);
        assert_eq!(ix.len(), 3);
        assert_eq!(keys(&ix, &[10, 20, 30, 40, 60]), (4, Some((4, vec![4]))));
        assert_eq!(keys(&ix, &[10, 20, 30]), (3, Some((2, vec![2]))));
        assert_eq!(keys(&ix, &[10, 20, 50]), (3, Some((3, vec![3]))));
        assert_eq!(keys(&ix, &[10]), (1, None));
        assert_eq!(keys(&ix, &[99, 20]), (0, None));
        assert_eq!(keys(&ix, &[]), (0, None));
        // Duplicates live side by side, in insertion order.
        ix.insert(&[10, 20], 22);
        assert_eq!(ix.get(&[10, 20]), &[2, 22]);
        assert_eq!(keys(&ix, &[10, 20, 7]), (2, Some((2, vec![2, 22]))));
    }

    #[test]
    fn accept_filters_entries_and_falls_back_to_shorter_keys() {
        let mut ix = RadixIndex::new(1);
        ix.insert(&[1, 2, 3], 3);
        ix.insert(&[1, 2], 2);
        let m = ix.lookup(&[1, 2, 3, 4], |_, &v| v != 3);
        assert_eq!(m.hit.map(|(n, v)| (n, *v[0])), Some((2, 2)));
        assert_eq!(m.shared, 3);
        let m = ix.lookup(&[1, 2, 3], |len, _| len < 3);
        assert_eq!(m.hit.map(|(n, _)| n), Some(2));
    }

    #[test]
    fn remove_prunes_and_merges() {
        let mut ix = RadixIndex::new(1);
        ix.insert(&[1, 2, 3, 4], 'a');
        ix.insert(&[1, 2, 5], 'b');
        ix.insert(&[1, 2], 'c');
        assert_eq!(ix.remove(&[1, 2], |&v| v == 'x'), None);
        assert_eq!(ix.remove(&[1, 2, 3], |_| true), None, "no entry at an inner edge position");
        assert_eq!(ix.remove(&[1, 2], |_| true), Some('c'));
        assert_eq!(ix.remove(&[1, 2, 5], |_| true), Some('b'));
        // [1, 2] and [3, 4] merged back into one edge.
        assert_eq!(ix.root.children.len(), 1);
        assert_eq!(ix.root.children[&1].edge, vec![1, 2, 3, 4]);
        assert_eq!(ix.lookup(&[1, 2, 3, 4], |_, _| true).hit.map(|(n, _)| n), Some(4));
        assert_eq!(ix.remove(&[1, 2, 3, 4], |_| true), Some('a'));
        assert!(ix.is_empty());
        assert!(ix.root.children.is_empty());
    }

    /// GLM-5.3-Flash: pools of 4 tokens (the device index) and pages of 64 (the host tier).
    #[test]
    fn shared_prefixes_round_down_to_the_granularity() {
        let key: Vec<Token> = (0..200).collect();
        for (g, want) in [(1usize, 131usize), (4, 128), (64, 128)] {
            let mut ix = RadixIndex::new(g);
            ix.insert(&key, ());
            let mut q = key[..131].to_vec();
            q.push(9_999);
            let m = ix.lookup(&q, |_, _| true);
            assert_eq!(m.shared, want, "granularity {g}");
            assert!(m.hit.is_none(), "the key does not prefix the query");
        }
        let mut ix = RadixIndex::new(64);
        ix.insert(&key[..70], ());
        assert_eq!(ix.lookup(&key, |_, _| true).shared, 64);
        assert_eq!(ix.lookup(&key, |_, _| true).hit.map(|(n, _)| n), Some(70), "hits stay exact");
    }

    /// Image rows are ids with bit 31 set: different images never share a key.
    #[test]
    fn image_identities_are_part_of_the_key() {
        let img = |h: u32, n: u32| (0..n).map(move |i| 0x8000_0000 | (h * 7919 + i));
        let a: Vec<Token> = [1, 2].into_iter().chain(img(1, 4)).chain([3]).collect();
        let b: Vec<Token> = [1, 2].into_iter().chain(img(2, 4)).chain([3]).collect();
        let mut ix = RadixIndex::new(1);
        ix.insert(&a, 'a');
        assert_eq!(keys_char(&ix, &b), (2, None));
        assert_eq!(keys_char(&ix, &a).1, Some(7));
    }

    fn keys_char(ix: &RadixIndex<char>, q: &[Token]) -> (usize, Option<usize>) {
        let m = ix.lookup(q, |_, _| true);
        (m.shared, m.hit.map(|(n, _)| n))
    }

    struct Rng(u64);
    impl Rng {
        fn next(&mut self) -> u64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            self.0
        }
        fn below(&mut self, n: u64) -> u64 {
            self.next() % n
        }
    }

    /// Against the source's rule, a scan of every key: the hit is the longest key that
    /// prefixes the query, `shared` the longest common prefix with any key. Random keys over a
    /// small alphabet (so they share prefixes), random inserts and removes.
    #[test]
    fn agrees_with_a_linear_scan() {
        let mut rng = Rng(0x9e37_79b9_7f4a_7c15);
        for g in [1usize, 4, 64] {
            let mut ix = RadixIndex::new(g);
            let mut live: Vec<(Vec<Token>, u32)> = Vec::new();
            for step in 0..3000u32 {
                if !live.is_empty() && rng.below(3) == 0 {
                    let i = rng.below(live.len() as u64) as usize;
                    let (k, v) = live.swap_remove(i);
                    assert_eq!(ix.remove(&k, |&x| x == v), Some(v));
                } else {
                    let base = if live.is_empty() { Vec::new() } else { live[rng.below(live.len() as u64) as usize].0.clone() };
                    let keep = rng.below(base.len() as u64 + 1) as usize;
                    let mut k = base[..keep].to_vec();
                    for _ in 0..rng.below(150) {
                        k.push(rng.below(3) as Token);
                    }
                    ix.insert(&k, step);
                    live.push((k, step));
                }
                assert_eq!(ix.len(), live.len());
                let mut q: Vec<Token> = if live.is_empty() || rng.below(4) == 0 {
                    Vec::new()
                } else {
                    live[rng.below(live.len() as u64) as usize].0.clone()
                };
                let cut = rng.below(q.len() as u64 + 1) as usize;
                q.truncate(cut);
                for _ in 0..rng.below(40) {
                    q.push(rng.below(3) as Token);
                }
                let m = ix.lookup(&q, |_, _| true);
                let lcp = live.iter().map(|(k, _)| common_len(k, &q)).max().unwrap_or(0);
                assert_eq!(m.shared, lcp / g * g, "step {step}");
                let best = live.iter().filter(|(k, _)| q.starts_with(k)).map(|(k, _)| k.len()).max();
                assert_eq!(m.hit.as_ref().map(|h| h.0), best, "step {step}");
                if let Some((n, vals)) = m.hit {
                    let mut want: Vec<u32> = live.iter().filter(|(k, _)| k.len() == n && q.starts_with(k)).map(|x| x.1).collect();
                    let mut got: Vec<u32> = vals.into_iter().copied().collect();
                    want.sort_unstable();
                    got.sort_unstable();
                    assert_eq!(got, want, "step {step}");
                }
            }
        }
    }
}
