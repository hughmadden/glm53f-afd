//! Byte-pair merging in the reference's order.
//!
//! This follows `tokenizers`' `Word::merge_all` step for step, so its results match even where
//! a merges list would let two orders differ: a min-heap of candidate merges keyed by (rank,
//! position); a popped entry is skipped when its left symbol is gone, is last, or no longer forms
//! the pair the entry was queued for; after a merge the new pairs with the left and right
//! neighbours are queued. Cost is O(n log n) in the word's length, so a very long piece (a run
//! of one letter, a base64 blob) stays cheap.

use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap};

/// (left id, right id) -> (rank, merged id).
pub type Merges = HashMap<(u32, u32), (u32, u32)>;

#[derive(Clone, Copy, Debug)]
struct Symbol {
    id: u32,
    prev: isize,
    next: isize,
    /// Bytes covered; 0 marks a symbol merged into its left neighbour.
    len: usize,
}

#[derive(PartialEq, Eq)]
struct Candidate {
    pos: usize,
    rank: u32,
    new_id: u32,
}

impl Ord for Candidate {
    // BinaryHeap is a max-heap: reverse both keys so the lowest rank, then the lowest
    // position, pops first.
    fn cmp(&self, other: &Self) -> Ordering {
        other.rank.cmp(&self.rank).then_with(|| other.pos.cmp(&self.pos))
    }
}

impl PartialOrd for Candidate {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// Merge the initial symbols `ids` (one per byte of the piece) and append the result to `out`.
pub fn merge(ids: &[u32], merges: &Merges, out: &mut Vec<u32>) {
    let n = ids.len();
    let mut syms: Vec<Symbol> = ids
        .iter()
        .enumerate()
        .map(|(i, &id)| Symbol { id, prev: i as isize - 1, next: if i + 1 < n { i as isize + 1 } else { -1 }, len: 1 })
        .collect();
    let mut queue = BinaryHeap::with_capacity(n);
    for i in 0..n.saturating_sub(1) {
        if let Some(&(rank, new_id)) = merges.get(&(syms[i].id, syms[i + 1].id)) {
            queue.push(Candidate { pos: i, rank, new_id });
        }
    }
    while let Some(top) = queue.pop() {
        if syms[top.pos].len == 0 || syms[top.pos].next == -1 {
            continue;
        }
        let next_pos = syms[top.pos].next as usize;
        let right = syms[next_pos];
        // An entry queued for a pair that has since changed is stale.
        match merges.get(&(syms[top.pos].id, right.id)) {
            Some(&(_, new_id)) if new_id == top.new_id => {}
            _ => continue,
        }
        let cur = &mut syms[top.pos];
        cur.id = top.new_id;
        cur.len += right.len;
        cur.next = right.next;
        syms[next_pos].len = 0;
        if right.next >= 0 && (right.next as usize) < n {
            syms[right.next as usize].prev = top.pos as isize;
        }
        let cur = syms[top.pos];
        if cur.prev >= 0 {
            let prev = cur.prev as usize;
            if let Some(&(rank, new_id)) = merges.get(&(syms[prev].id, cur.id)) {
                queue.push(Candidate { pos: prev, rank, new_id });
            }
        }
        if cur.next >= 0 && (cur.next as usize) < n {
            if let Some(&(rank, new_id)) = merges.get(&(cur.id, syms[cur.next as usize].id)) {
                queue.push(Candidate { pos: top.pos, rank, new_id });
            }
        }
    }
    out.extend(syms.iter().filter(|s| s.len != 0).map(|s| s.id));
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A toy vocabulary: a=0 b=1 c=2 ab=3 bc=4 abc=5 aa=6; merges in rank order.
    fn toy() -> Merges {
        let mut m = Merges::new();
        m.insert((0, 1), (0, 3)); // a b -> ab
        m.insert((1, 2), (1, 4)); // b c -> bc
        m.insert((3, 2), (2, 5)); // ab c -> abc
        m.insert((0, 0), (3, 6)); // a a -> aa
        m
    }

    fn run(ids: &[u32]) -> Vec<u32> {
        let mut out = Vec::new();
        merge(ids, &toy(), &mut out);
        out
    }

    #[test]
    fn lowest_rank_first_then_leftmost() {
        assert_eq!(run(&[0, 1, 2]), [5]); // a b c: ab (rank 0) before bc (rank 1), then abc
        assert_eq!(run(&[1, 2]), [4]);
        assert_eq!(run(&[0, 0, 0]), [6, 0]); // aaa: the leftmost pair merges first
        assert_eq!(run(&[0, 0, 0, 0]), [6, 6]);
        assert_eq!(run(&[2]), [2]);
        assert_eq!(run(&[]), Vec::<u32>::new());
    }

    #[test]
    fn long_words_stay_cheap() {
        let ids = vec![0u32; 200_000];
        let out = run(&ids);
        assert_eq!(out, vec![6; 100_000]);
    }
}
