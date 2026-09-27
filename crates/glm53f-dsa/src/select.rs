//! Top-k pool selection with a deterministic tie-break, and the selection's
//! expansion to raw token indices.
//!
//! Order: higher score first; equal scores (including +0 and -0) go to the lower
//! pool index. NaN scores are never selected. The order is a total order on
//! `(score, pool)`, so the selected set does not depend on how the candidates
//! were split into tiles or merged, and a CPU and a GPU selection over the same
//! scores agree exactly. `torch.topk` leaves ties unspecified, so the reference
//! can differ from this rule only among pools with equal scores.

/// Sortable key for `(score, pool)`: larger keys rank first. 0 means invalid.
///
/// High 32 bits: the score's bits mapped to an unsigned order (both zeros equal).
/// Low 32 bits: `!pool`, so the lower pool wins a tie. The CUDA selector uses the
/// same encoding.
#[inline]
pub fn score_key(score: f32, pool: u32) -> u64 {
    if score.is_nan() {
        return 0;
    }
    let bits = if score == 0.0 { 0u32 } else { score.to_bits() };
    let ordered = if bits & 0x8000_0000 != 0 { !bits } else { bits ^ 0x8000_0000 };
    ((ordered as u64) << 32) | (!pool) as u64
}

/// Pool index of a key made by [`score_key`].
#[inline]
pub fn key_pool(key: u64) -> u32 {
    !(key as u32)
}

/// Score of a key made by [`score_key`] (a canonical zero is +0).
#[inline]
pub fn key_score(key: u64) -> f32 {
    let ordered = (key >> 32) as u32;
    let bits = if ordered & 0x8000_0000 != 0 { ordered ^ 0x8000_0000 } else { !ordered };
    f32::from_bits(bits)
}

/// The best `k` pools of `scores` (index = pool id), in rank order.
pub fn top_k(scores: &[f32], k: usize) -> Vec<u32> {
    let mut keys: Vec<u64> =
        scores.iter().enumerate().map(|(p, s)| score_key(*s, p as u32)).filter(|k| *k != 0).collect();
    if keys.len() > k {
        // Partial selection, then order the kept keys.
        let nth = keys.len() - k;
        keys.select_nth_unstable(nth);
        keys.drain(..nth);
    }
    keys.sort_unstable_by(|a, b| b.cmp(a));
    keys.into_iter().map(key_pool).collect()
}

/// One query row's selection: kept pools plus the raw tail.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Selection {
    /// Absolute position of the query token (0-based, the request's first token is 0).
    pub position: usize,
    /// Complete pools visible to the query: `(position + 1) / kpool`.
    pub visible_pools: usize,
    /// Kept pools in rank order (score descending, ties to the lower pool).
    pub pools: Vec<u32>,
    /// First token of the incomplete tail pool: `kpool * visible_pools`.
    pub tail_start: usize,
    /// Tail tokens `tail_start ..= position` (0 to kpool - 1 of them).
    pub tail_len: usize,
}

impl Selection {
    /// Build a selection from the scores of the pools visible to the row.
    pub fn from_scores(position: usize, kpool: usize, topk_pools: usize, tail: bool, scores: &[f32]) -> Self {
        let visible = (position + 1) / kpool;
        assert_eq!(scores.len(), visible, "one score per visible pool");
        let pools = top_k(scores, topk_pools);
        let tail_start = visible * kpool;
        let tail_len = if tail { position + 1 - tail_start } else { 0 };
        Self { position, visible_pools: visible, pools, tail_start, tail_len }
    }

    /// Dense selection (every visible pool) for rows with at most `topk_pools` pools.
    pub fn dense(position: usize, kpool: usize, tail: bool) -> Self {
        let visible = (position + 1) / kpool;
        let tail_start = visible * kpool;
        Self {
            position,
            visible_pools: visible,
            pools: (0..visible as u32).collect(),
            tail_start,
            tail_len: if tail { position + 1 - tail_start } else { 0 },
        }
    }

    /// Kept pools in ascending order.
    pub fn pools_ascending(&self) -> Vec<u32> {
        let mut p = self.pools.clone();
        p.sort_unstable();
        p
    }

    /// Selected token indices in ascending order: every kept pool's tokens, then
    /// the tail (which lies after every complete pool).
    pub fn tokens(&self, kpool: usize) -> Vec<u32> {
        let mut t = Vec::with_capacity(self.pools.len() * kpool + self.tail_len);
        for p in self.pools_ascending() {
            for i in 0..kpool {
                t.push(p * kpool as u32 + i as u32);
            }
        }
        for i in 0..self.tail_len {
            t.push((self.tail_start + i) as u32);
        }
        t
    }

    /// The row as the `transformers` indexer returns it (for a request without
    /// padding): `select_k` pools' tokens in rank order (-1 for pools not
    /// visible to this row), then the `kpool - 1` tail slots (-1 unused), padded
    /// with -1 to `width`. `select_k = min(topk_pools, complete pools in the
    /// sequence)` is a property of the whole call, not of the row.
    pub fn reference_row(&self, kpool: usize, select_k: usize, tail: bool, width: usize) -> Vec<i32> {
        let mut row = Vec::with_capacity(width);
        for j in 0..select_k {
            match self.pools.get(j) {
                Some(p) => {
                    for i in 0..kpool {
                        row.push((*p as usize * kpool + i) as i32);
                    }
                }
                None => row.extend(std::iter::repeat_n(-1, kpool)),
            }
        }
        if tail {
            for i in 0..kpool - 1 {
                row.push(if i < self.tail_len { (self.tail_start + i) as i32 } else { -1 });
            }
        }
        row.resize(width, -1);
        row.truncate(width);
        row
    }
}

/// Compare two selections that may differ only by near-ties at the boundary.
///
/// Returns the pools in the symmetric difference whose score differs from the
/// k-th kept score by more than `tol` (empty when the selections agree up to
/// ties within `tol`). `scores` are the scores used for the judgement.
pub fn boundary_mismatches(a: &[u32], b: &[u32], scores: &[f32], tol: f32) -> Vec<u32> {
    use std::collections::HashSet;
    let sa: HashSet<u32> = a.iter().copied().collect();
    let sb: HashSet<u32> = b.iter().copied().collect();
    if sa == sb {
        return Vec::new();
    }
    // The k-th score of the reference selection `a`.
    let kth = a.iter().map(|p| scores[*p as usize]).fold(f32::INFINITY, f32::min);
    let mut bad: Vec<u32> = sa
        .symmetric_difference(&sb)
        .copied()
        .filter(|p| (scores[*p as usize] - kth).abs() > tol)
        .collect();
    bad.sort_unstable();
    bad
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn key_order() {
        let a = score_key(1.0, 5);
        let b = score_key(1.0, 6);
        let c = score_key(2.0, 100);
        let d = score_key(-1.0, 0);
        assert!(c > a && a > b && b > d && d > 0);
        assert_eq!(score_key(0.0, 3), score_key(-0.0, 3));
        assert!(score_key(f32::MIN, 0) > 0);
        assert_eq!(score_key(f32::NAN, 0), 0);
        assert_eq!(key_pool(a), 5);
        for s in [0.0f32, 1.5, -2.25, f32::MAX, f32::MIN, 1e-40] {
            assert_eq!(key_score(score_key(s, 9)), s);
        }
    }

    #[test]
    fn ties_go_to_lower_pool() {
        let s = [3.0f32, 1.0, 3.0, 3.0, 0.0, -0.0, 0.0];
        assert_eq!(top_k(&s, 2), vec![0, 2]);
        assert_eq!(top_k(&s, 4), vec![0, 2, 3, 1]);
        assert_eq!(top_k(&s, 6), vec![0, 2, 3, 1, 4, 5]);
        assert_eq!(top_k(&[f32::NAN, 1.0], 2), vec![1]);
    }

    #[test]
    fn expansion_and_reference_row() {
        // Position 13: 14 visible tokens -> 3 complete pools, tail 12..=13.
        let s = Selection::from_scores(13, 4, 2, true, &[0.5, 2.0, 1.0]);
        assert_eq!(s.pools, vec![1, 2]);
        assert_eq!((s.tail_start, s.tail_len), (12, 2));
        assert_eq!(s.tokens(4), vec![4, 5, 6, 7, 8, 9, 10, 11, 12, 13]);
        let r = s.reference_row(4, 2, true, 11);
        assert_eq!(r, vec![4, 5, 6, 7, 8, 9, 10, 11, 12, 13, -1]);
        // A row with fewer visible pools than select_k gets -1 blocks.
        let s = Selection::from_scores(3, 4, 2, true, &[1.0]);
        let r = s.reference_row(4, 2, true, 11);
        assert_eq!(r, vec![0, 1, 2, 3, -1, -1, -1, -1, -1, -1, -1]);
    }
}
