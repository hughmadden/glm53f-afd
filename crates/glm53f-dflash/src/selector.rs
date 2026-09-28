//! The candidate selector: the top-k tokens per draft position and one path through them.
//!
//! Per draft position `e` (0..7, block rows 1..7) the LM head's logits give `k` = 16 candidates
//! and their logits (the *unary* scores). The selector scores the edge from a predecessor token
//! `p` to candidate `c` as
//!
//! ```text
//! score(e, p, c) = unary[e][c] + sum_r A[p][r] * h[e][r] * B[c][r]
//! ```
//!
//! with `A` the predecessor codebook, `B` the successor codebook (both `[vocab][rank]`) and
//! `h[e] = hidden_projection(draft_hidden[e])`. The walk starts from the anchor (the last
//! verified token) and at each position takes the candidate that the previous choice leads to:
//! the argmax (greedy), or a draw from `softmax(score / T)`. Reference: `CandidateSelector.select`
//! in z-lab/dflash `dflash/model.py` (lines 515-547); SGLang scores the same edges as a lattice
//! (`_score_edges`, `sample_path`, `selector_walk_triton`), which the walk here follows for the
//! sampled case (see `README.md`).

use crate::bf16;

/// The `k` largest values of `row[..limit]`, descending; ties go to the lower index.
pub fn top_k(row: &[f32], k: usize, limit: usize) -> (Vec<f32>, Vec<u32>) {
    let limit = limit.min(row.len());
    assert!(k <= limit, "top_k: {k} of {limit}");
    // Keep a sorted list of the best k seen so far.
    let mut best: Vec<(f32, u32)> = Vec::with_capacity(k + 1);
    for (i, &v) in row[..limit].iter().enumerate() {
        if best.len() == k && v.partial_cmp(&best[k - 1].0) != Some(std::cmp::Ordering::Greater) {
            continue; // NaN and ties with the k-th never displace an earlier index
        }
        let at = best.partition_point(|&(b, _)| b >= v);
        best.insert(at, (v, i as u32));
        best.truncate(k);
    }
    best.into_iter().unzip()
}

/// How a path is chosen.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Pick<'a> {
    /// The first maximum at every position.
    Greedy,
    /// A draw from `softmax(score / temperature)` per position, by inverse CDF: the chosen index
    /// is the number of cumulative probabilities not above `uniforms[e]` (capped at `k - 1`), as
    /// SGLang's `sample_path` and `selector_walk_triton` do.
    Sample {
        temperature: f32,
        uniforms: &'a [f32],
    },
}

/// One path through the candidates.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Walk {
    /// The chosen token per position.
    pub tokens: Vec<u32>,
    /// The chosen candidate slot per position (0 = the highest logit).
    pub index: Vec<u32>,
    /// The edge scores the walk used, `[positions][k]`: the row of the previous choice.
    pub scores: Vec<f32>,
    /// The distribution each token was drawn from, `[positions][k]`: one-hot when greedy, as
    /// SGLang stores it for greedy rows.
    pub q: Vec<f32>,
    /// `softmax(scores)[chosen]` at temperature 1: the drafter's confidence in each draft.
    pub conf: Vec<f32>,
}

/// `softmax(x / t)` in f32 (max subtracted first).
pub fn softmax(x: &[f32], t: f32) -> Vec<f32> {
    let s: Vec<f32> = x.iter().map(|&v| v / t).collect();
    let m = s.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let e: Vec<f32> = s.iter().map(|&v| (v - m).exp()).collect();
    let z: f32 = e.iter().sum();
    e.iter().map(|&v| v / z).collect()
}

/// The sum in the order the GPU selector forms it (`select_kernel`): each warp of 32 consecutive
/// terms by the xor butterfly (pairs 16 apart, then 8, 4, 2, 1), then the warps' totals, padded
/// with zeros to a power of two, by halving strides. The reference's einsum sums in some other
/// order; the difference is f32 rounding.
pub fn tree_sum(x: &[f32]) -> f32 {
    let warps = x.len().div_ceil(32).max(1);
    let mut totals = Vec::with_capacity(warps.next_power_of_two());
    for w in 0..warps {
        let mut v = [0f32; 32];
        for (d, &t) in v.iter_mut().zip(&x[w * 32..((w + 1) * 32).min(x.len())]) {
            *d = t;
        }
        let mut s = 16;
        while s >= 1 {
            for i in 0..s {
                v[i] += v[i + s];
            }
            s /= 2;
        }
        totals.push(v[0]);
    }
    totals.resize(warps.next_power_of_two(), 0.0);
    let mut s = totals.len() / 2;
    while s >= 1 {
        for i in 0..s {
            totals[i] += totals[i + s];
        }
        s /= 2;
    }
    totals[0]
}

/// Edge scores from predecessor `p` to each candidate: `unary[c] + sum_r (A[p][r] * h[r]) * B[c][r]`.
/// `a_h` is `A[p] * h` (the product formed first, as the reference does).
pub fn edge_row(a_h: &[f32], cands: &[u32], unary: &[f32], succ: &[u16], rank: usize) -> Vec<f32> {
    cands
        .iter()
        .zip(unary)
        .map(|(&c, &u)| {
            let b = &succ[c as usize * rank..(c as usize + 1) * rank];
            let terms: Vec<f32> = a_h
                .iter()
                .zip(b)
                .map(|(&x, &y)| x * bf16::to_f32(y))
                .collect();
            u + tree_sum(&terms)
        })
        .collect()
}

/// Walk the positions: `unary` and `cands` are `[positions][k]`, `h` is `[positions][rank]`.
#[allow(clippy::too_many_arguments)]
pub fn walk(
    unary: &[f32],
    cands: &[u32],
    h: &[f32],
    anchor: u32,
    pred: &[u16],
    succ: &[u16],
    rank: usize,
    k: usize,
    pick: Pick<'_>,
) -> Walk {
    let n = unary.len() / k;
    let mut out = Walk::default();
    let mut prev = anchor;
    for e in 0..n {
        let a = &pred[prev as usize * rank..(prev as usize + 1) * rank];
        let a_h: Vec<f32> = a
            .iter()
            .zip(&h[e * rank..(e + 1) * rank])
            .map(|(&x, &y)| bf16::to_f32(x) * y)
            .collect();
        let row = edge_row(
            &a_h,
            &cands[e * k..(e + 1) * k],
            &unary[e * k..(e + 1) * k],
            succ,
            rank,
        );
        let (index, q) = match pick {
            Pick::Greedy => {
                let best = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let i = row.iter().position(|&v| v == best).unwrap_or(0);
                let mut q = vec![0f32; k];
                q[i] = 1.0;
                (i, q)
            }
            Pick::Sample {
                temperature,
                uniforms,
            } => {
                let q = softmax(&row, temperature.max(1e-5));
                let mut c = 0f32;
                let mut i = 0usize;
                for &p in &q {
                    c += p;
                    if uniforms[e] >= c {
                        i += 1;
                    }
                }
                (i.min(k - 1), q)
            }
        };
        let conf = softmax(&row, 1.0)[index];
        prev = cands[e * k + index];
        out.tokens.push(prev);
        out.index.push(index as u32);
        out.scores.extend_from_slice(&row);
        out.q.extend(q);
        out.conf.push(conf);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn top_k_order_and_ties() {
        let row = [1.0, 5.0, 3.0, 5.0, -1.0, 4.0, 9.0];
        let (v, i) = top_k(&row, 4, row.len());
        assert_eq!(v, vec![9.0, 5.0, 5.0, 4.0]);
        assert_eq!(i, vec![6, 1, 3, 5]);
        // The limit excludes the tail of the row.
        let (_, i) = top_k(&row, 2, 6);
        assert_eq!(i, vec![1, 3]);
    }

    #[test]
    fn tree_sum_order() {
        // 32 terms: pairs 16 apart first.
        let x: Vec<f32> = (0..32).map(|i| i as f32).collect();
        assert_eq!(tree_sum(&x), 496.0);
        // Rounding shows the order: 1e8 + 1 + (-1e8) + ... in f32.
        let mut y = vec![0f32; 64];
        y[0] = 1e8;
        y[16] = 1.0;
        y[32] = -1e8;
        y[48] = 1.0;
        // Warp 0: 1e8 + 1 = 1e8 (rounded); warp 1: -1e8 + 1 = -1e8 + 1 -> -99999999 rounds to -1e8.
        assert_eq!(tree_sum(&y), 0.0);
        assert_eq!(tree_sum(&[]), 0.0);
        assert_eq!(tree_sum(&[2.5]), 2.5);
    }

    #[test]
    fn walk_follows_the_previous_choice() {
        // rank 1, vocab 4: A[p] = p, B[c] = c, h = 1, unary 0: score(p, c) = p * c.
        let pred = crate::bf16::encode(&[0.0, 1.0, 2.0, 3.0]);
        let succ = crate::bf16::encode(&[0.0, 1.0, -1.0, 2.0]);
        let (k, rank) = (2, 1);
        // Two positions with candidates {1, 2} then {2, 3}.
        let cands = [1, 2, 2, 3];
        let unary = [0.0, 0.0, 0.0, 0.0];
        let h = [1.0, 1.0];
        let w = walk(&unary, &cands, &h, 3, &pred, &succ, rank, k, Pick::Greedy);
        // From 3: scores 3*B[1] = 3, 3*B[2] = -3 -> token 1; from 1: B[2] = -1, B[3] = 2 -> token 3.
        assert_eq!(w.tokens, vec![1, 3]);
        assert_eq!(w.index, vec![0, 1]);
        assert_eq!(w.scores, vec![3.0, -3.0, -1.0, 2.0]);
        assert_eq!(w.q, vec![1.0, 0.0, 0.0, 1.0]);
        // Sampling with u = 0 takes the first candidate with any mass; u just under 1 the last.
        let s = walk(
            &unary,
            &cands,
            &h,
            3,
            &pred,
            &succ,
            rank,
            k,
            Pick::Sample {
                temperature: 1.0,
                uniforms: &[0.0, 0.999],
            },
        );
        assert_eq!(s.index, vec![0, 1]);
        let q0 = softmax(&[3.0, -3.0], 1.0);
        assert_eq!(&s.q[..2], &q0[..]);
    }
}
