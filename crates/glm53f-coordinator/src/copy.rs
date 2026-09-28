//! Copy windows: drafts copied from a request's own context (TensorFold's idea, ported from
//! mimo26f-afd v1.3.0's `copy.rs`; PROVENANCE.md).
//!
//! Much of a coding agent's output repeats its context: a file written back with an edit, an edit
//! call quoting the lines it replaces, a quoted passage. When the last [`ENTRY`] tokens of a greedy
//! request's context (its prompt, then its output so far) occurred earlier in it, the tokens that
//! followed that occurrence become the request's drafts for the step, in place of the drafter's
//! (`crate::scheduler`), up to the verify window's cap. The verify pass checks a copied token
//! exactly as it checks a draft, so a copy changes which rows are verified, never the output.
//!
//! - **The entry, 24 tokens.** TensorFold's chain proposer lets a copy replace the drafter's block
//!   only when 24 context tokens match (`confident_match`); its tree proposer and mimo26f-afd take 8.
//!   This engine's drafter is a chain cut at a confidence of 0.7, so its windows are short where it
//!   is unsure, while a copy verifies up to 8 rows: a coincidental match costs rows. Replayed on
//!   real text (`examples/copy_replay.rs`): with an entry of 8, a fresh-code reply of 3,959 tokens
//!   copied in 101 rounds that kept 37% of their copied tokens (3.6 tokens a round from 8 rows), and
//!   prose in 16 that kept 13%; with 24, fresh code copied in 2 rounds and prose in none, while
//!   rewrites, edits and quotes still copied 86-96% of their replies, 7.9 tokens a copy round.
//! - **Which occurrence.** Earlier occurrences are found by their last [`MATCH`] tokens; of the
//!   newest [`CANDIDATES`], the one whose match reaches furthest back wins (counted up to
//!   [`EXTEND`] tokens), ties to the most recent, and it must reach [`ENTRY`] tokens.
//! - **A periodic run.** A copy that runs into the tokens it is copying repeats them.
//! - **The index.** Every position's gram of [`MATCH`] tokens, keyed by a 64-bit fingerprint, with
//!   a link to the gram's previous occurrence. A candidate is checked against the context, so a
//!   fingerprint collision costs a candidate, never a wrong copy. The context only grows between
//!   calls; each call indexes the positions added since the last one, at most [`CATCH_UP`] of them,
//!   and copies nothing until the index has caught up. A step's lookup therefore never scans the
//!   context, and a long prompt (prefilled, or resumed from a snapshot without a prefill) is
//!   indexed over its request's first steps. The scheduler reserves room for every position a
//!   request can reach ([`CopyIndex::reserve`]), so the map never rehashes mid-request.
//! - **Ids the model cannot pick** (at or past `bound`, the model's `Limits::sample_vocab`: image
//!   rows) end a copy.

use std::collections::HashMap;

use crate::model::Token;

/// Tokens of the grams the index keys (an earlier occurrence is found by these).
pub const MATCH: usize = 8;
/// Context tokens that must match before a copy is proposed (TensorFold's `confident_match`).
pub const ENTRY: usize = 24;
/// Backward match length counted up to this when choosing among occurrences.
pub const EXTEND: usize = 64;
/// Earlier occurrences tried per proposal, newest first.
pub const CANDIDATES: usize = 64;
/// Positions indexed per call at most. A million-token prompt of fresh text is indexed over 64
/// steps, 1.1 ms of host time a step (the median) on a development machine; its first, which first
/// touches the reserved table's pages, took 13.6 ms (`examples/copy_cost.rs`).
pub const CATCH_UP: usize = 1 << 14;
/// The likelihood the step's row budget (`crate::spec::budget`) gives each copied token:
/// TensorFold's measured 94% for the next token of a copy backed by 8 or more tokens, for every
/// token of the copy.
pub const TOKEN_P: f32 = 0.94;
const NONE: u32 = u32::MAX;

/// A request's index of the [`MATCH`]-grams in its context, extended as the context grows.
pub struct CopyIndex {
    /// The last position (the gram's final token) of each gram fingerprint seen.
    head: HashMap<u64, u32>,
    /// `prev[p]`: the previous position with the fingerprint of the gram ending at `p`, or
    /// [`NONE`].
    prev: Vec<u32>,
    /// The match length behind the last proposal (0 when it proposed nothing).
    matched: usize,
    /// Matching tokens a copy needs.
    entry: usize,
}

impl Default for CopyIndex {
    fn default() -> Self {
        CopyIndex::with_entry(ENTRY)
    }
}

impl CopyIndex {
    /// An index whose copies need `entry` matching tokens ([`MATCH`] to [`EXTEND`]; the default is
    /// [`ENTRY`]).
    pub fn with_entry(entry: usize) -> Self {
        CopyIndex { head: HashMap::new(), prev: Vec::new(), matched: 0, entry: entry.clamp(MATCH, EXTEND) }
    }

    /// Up to `k` tokens to copy after `ctx`, each below `bound`; none when `k < 2`, when the last
    /// tokens of `ctx` match no earlier span over the entry's length, while the index is catching
    /// up with `ctx`, or when fewer than 2 tokens could be copied. The context must only grow
    /// between calls (a request's history does).
    pub fn propose(&mut self, ctx: &[Token], k: usize, bound: usize) -> Vec<Token> {
        self.matched = 0;
        let n = ctx.len();
        if k < 2 || n <= MATCH {
            return Vec::new();
        }
        debug_assert!(self.prev.len() < n, "the context shrank under its copy index");
        // Index every gram with at least one token after it, at most CATCH_UP per call.
        let end = (n - 1).min(self.prev.len() + CATCH_UP);
        for p in self.prev.len()..end {
            if p + 1 < MATCH {
                self.prev.push(NONE);
                continue;
            }
            let g = fingerprint(&ctx[p + 1 - MATCH..=p]);
            self.prev.push(self.head.insert(g, p as u32).unwrap_or(NONE));
        }
        if self.prev.len() < n - 1 {
            return Vec::new();
        }
        let tail = &ctx[n - MATCH..];
        let Some(&newest) = self.head.get(&fingerprint(tail)) else {
            return Vec::new();
        };
        // The occurrence matching furthest back (ties: the most recent).
        let (mut best, mut best_len, mut p) = (NONE, 0usize, newest);
        for _ in 0..CANDIDATES {
            if p == NONE {
                break;
            }
            let q = p as usize;
            if ctx[q + 1 - MATCH..=q] == *tail {
                let mut len = MATCH;
                while len < EXTEND && len <= q && ctx[q - len] == ctx[n - 1 - len] {
                    len += 1;
                }
                if len > best_len {
                    (best, best_len) = (p, len);
                    if len == EXTEND {
                        break;
                    }
                }
            }
            p = self.prev[q];
        }
        if best_len < self.entry {
            return Vec::new();
        }
        let from = best as usize + 1;
        let mut out = Vec::with_capacity(k);
        for j in 0..k {
            let t = if from + j < n { ctx[from + j] } else { out[from + j - n] };
            if t as usize >= bound {
                break;
            }
            out.push(t);
        }
        if out.len() < 2 {
            return Vec::new();
        }
        self.matched = best_len;
        out
    }

    /// Room for a context of `tokens` tokens, so the index never grows while it reaches them (a
    /// growing map rehashes every entry at once, a stall of tens of ms at a million positions).
    pub fn reserve(&mut self, tokens: usize) {
        self.head.reserve(tokens.saturating_sub(self.head.len()));
        self.prev.reserve(tokens.saturating_sub(self.prev.len()));
    }

    /// The match length behind the last proposal, the entry to [`EXTEND`] (0 when the last call
    /// proposed nothing).
    pub fn matched(&self) -> usize {
        self.matched
    }

    /// Positions indexed so far.
    pub fn indexed(&self) -> usize {
        self.prev.len()
    }

    /// Host bytes held, about (the map's table at the standard library's load factor of 7/8).
    pub fn bytes(&self) -> usize {
        let entry = std::mem::size_of::<(u64, u32)>() + 1;
        self.head.capacity() * 8 / 7 * entry + self.prev.capacity() * std::mem::size_of::<u32>()
    }
}

/// A 64-bit fingerprint of a gram (the map hashes it again with the standard library's keyed
/// hasher, so crafted prompts cannot flood a bucket).
fn fingerprint(g: &[Token]) -> u64 {
    let mut h = 0x243f_6a88_85a3_08d3u64;
    for pair in g.chunks(2) {
        let x = u64::from(pair[0]) | u64::from(pair.get(1).copied().unwrap_or(0)) << 32;
        h = mix(h ^ x);
    }
    h
}

/// SplitMix64's finalizer.
fn mix(mut x: u64) -> u64 {
    x = (x ^ (x >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    x ^ (x >> 31)
}

#[cfg(test)]
mod tests {
    use super::*;

    const BOUND: usize = 1 << 20;

    fn ids(s: &str) -> Vec<Token> {
        s.bytes().map(Token::from).collect()
    }

    /// mimo26f-afd's entry of 8 tokens (the index's gram).
    fn at8() -> CopyIndex {
        CopyIndex::with_entry(MATCH)
    }

    #[test]
    fn copies_what_followed_an_earlier_match() {
        let mut c = at8();
        let ctx = ids("fn alpha(x) { return x + 1; }\n// again: fn alpha(x) { ret");
        assert_eq!(c.propose(&ctx, 7, BOUND), ids("urn x +"));
        // Back to the start of the context (the 17 tokens of `fn alpha(x) { ret`).
        assert_eq!(c.matched(), 17);
        // Short matches do not count.
        let mut c = at8();
        assert!(c.propose(&ids("abcdefg xyz abcdefg"), 7, BOUND).is_empty());
        assert_eq!(c.matched(), 0);
        // Nothing when k < 2 or the context is short.
        let mut c = at8();
        assert!(c.propose(&ctx, 1, BOUND).is_empty());
        assert!(c.propose(&ctx[..MATCH], 7, BOUND).is_empty());
    }

    #[test]
    fn the_default_entry_is_24_matching_tokens() {
        // A 23-token span after `X`, then after `Y`: 23 tokens match, no copy by default.
        let span = "abcdefghijklmnopqrstuvw";
        let ctx = ids(&format!("X{span}1234567\nY{span}"));
        let mut c = CopyIndex::default();
        assert!(c.propose(&ctx, 7, BOUND).is_empty());
        assert_eq!(at8().propose(&ctx, 7, BOUND), ids("1234567"));
        // After `X` again: 24 match, a copy.
        let ctx = ids(&format!("X{span}1234567\nX{span}"));
        let mut c = CopyIndex::default();
        assert_eq!(c.propose(&ctx, 7, BOUND), ids("1234567"));
        assert_eq!(c.matched(), 24);
        // The entry is kept between the gram's length and the longest match counted.
        assert_eq!(CopyIndex::with_entry(1).entry, MATCH);
        assert_eq!(CopyIndex::with_entry(100).entry, EXTEND);
    }

    #[test]
    fn longest_backward_match_wins_then_most_recent() {
        // "12345678" occurs twice; only the first occurrence is preceded by "XY", as the tail is.
        let mut c = at8();
        let ctx = ids("XY12345678AB..zz12345678CD..XY12345678");
        assert_eq!(c.propose(&ctx, 2, BOUND), ids("AB"));
        assert_eq!(c.matched(), 10);
        // Equal evidence: the most recent occurrence.
        let mut c = at8();
        let ctx = ids("..12345678AB..12345678CD..12345678");
        assert_eq!(c.propose(&ctx, 2, BOUND), ids("CD"));
    }

    #[test]
    fn a_growing_context_is_indexed_incrementally() {
        let mut c = at8();
        let mut ctx = ids("the quick brown fox jumps over the lazy dog; ");
        for t in ids("the quick") {
            ctx.push(t);
            let got = c.propose(&ctx, 7, BOUND);
            let full = at8().propose(&ctx, 7, BOUND);
            assert_eq!(got, full);
        }
        assert_eq!(c.propose(&ctx, 7, BOUND), ids(" brown "));
        assert_eq!(c.indexed(), ctx.len() - 1);
    }

    #[test]
    fn a_periodic_run_repeats() {
        let mut c = at8();
        let ctx = ids("start: ab-ab-ab-ab-ab");
        assert_eq!(c.propose(&ctx, 7, BOUND), ids("-ab-ab-"));
    }

    #[test]
    fn ids_the_model_cannot_pick_end_a_copy() {
        // An image row (an id past the bound) three tokens into the copy.
        let mut ctx = ids("0123456789ab");
        ctx.push(0x8000_0001);
        ctx.extend(ids("cdef..0123456789"));
        let mut c = at8();
        assert_eq!(c.propose(&ctx, 7, BOUND), ids("ab"));
        // A copy cut below 2 tokens is none.
        ctx.extend(ids("a"));
        let mut c = at8();
        assert!(c.propose(&ctx, 7, BOUND).is_empty());
    }

    #[test]
    fn a_long_context_is_indexed_over_several_calls() {
        // A span, CATCH_UP + 1,000 tokens of filler that repeats no 8-gram, the span's start again.
        let span = ids("let copied = source.iter().map(|x| x + 1).collect();");
        let mut ctx = span.clone();
        ctx.extend((0..CATCH_UP as u32 + 1000).map(|i| 1000 + i));
        ctx.extend(&span[..30]);
        let mut c = CopyIndex::default();
        // The first call indexes CATCH_UP positions and copies nothing; the second catches up.
        assert!(c.propose(&ctx, 7, BOUND).is_empty());
        assert_eq!(c.indexed(), CATCH_UP);
        assert_eq!(c.propose(&ctx, 7, BOUND), span[30..37].to_vec());
        assert_eq!((c.indexed(), c.matched()), (ctx.len() - 1, 30));
    }
}
