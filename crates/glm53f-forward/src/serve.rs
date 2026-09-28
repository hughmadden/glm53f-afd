//! The serving shell's traits for this forward (features `coordinator` and `cuda`):
//! `glm53f_coordinator::KvSlot` for [`GlmKv`], and `glm53f_coordinator::ModelForward` for
//! [`ServedForward`], a [`GlmForward`] with the shell's GPU sampler.
//!
//! - Every pass selects its tokens on the device: greedy rows take the forward's own argmax
//!   (the first index of the largest logit below 154,856, the shell's greedy contract); rows
//!   with a draw or a mask go through the shell's sampler on the pass's device logits.
//! - Without a drafter [`Limits::block`] is 0 and the scheduler decodes one token per step.
//!   With the DFlash2 drafter attached (`GlmForward::attach_drafter`) the block is 8: each step
//!   drafts up to 7 tokens per request ([`ModelForward::draft`]), verifies every window in one
//!   pass and commits the rows of the delivered tokens, which the forward appends to the
//!   drafter's context. The shell accepts a draft only while it equals the target's own pick
//!   (the argmax, or the request's seeded draw at that row), so the tokens are the target's
//!   whatever is drafted. Greedy requests draft with the drafter's greedy walk; sampled ones
//!   with its sampled walk at the request's temperature, one seeded uniform per draft
//!   ([`draft_uniform`]; [`ServedForward::sampled_walk`] off: the greedy walk for every request).
//!   The drafter's `q` rows are not used: an exact-match verify needs only the tokens.
//! - A request whose drafter context lost rows (restored from the host tier, whose images hold
//!   no taps, or rewound or forked further back than its 2,048-row window holds) drafts nothing
//!   until [`ServedForward::warm_rows`] rows are back in context: its steps verify a window of
//!   one row, a decode step.
//! - Admission sees the page pool: slots grow into the pages the `KvPool` allocated up front,
//!   and snapshot marks take pages of the same pool (`crate::kv`), so `free_bytes` is the pool's
//!   free pages. The forward's buffers were allocated before the pool was sized, and a pass
//!   allocates nothing; `free_bytes` still subtracts any shortfall of device memory below the
//!   margin, for allocations outside both (the sampler's small buffers, kernel modules loaded on
//!   first use).
//! - Images are refused (the vision tower is a later phase).

use glm53f_coordinator::gpu::Sampler;
use glm53f_coordinator::model::{
    DecodeRow, Draft, DraftRow, KvSlot, Limits, ModelForward, Pick, Segment, SegmentOut, Token,
    Window,
};

use glm53f_dflash::Dims;

use crate::device;
use crate::draft::DraftReq;
use crate::forward::GlmForward;
use crate::kv::{GlmKv, KvMark};
use crate::shape::{SAMPLE_VOCAB, VOCAB};

fn s<T>(r: crate::Result<T>) -> Result<T, String> {
    r.map_err(|e| e.to_string())
}

impl KvSlot for GlmKv {
    type Mark = KvMark;

    fn tokens(&self) -> usize {
        GlmKv::tokens(self)
    }
    fn pending(&self) -> usize {
        GlmKv::pending(self)
    }
    fn capacity(&self) -> usize {
        GlmKv::capacity(self)
    }
    fn need_bytes(&self, tokens: usize) -> usize {
        GlmKv::need_bytes(self, tokens)
    }
    fn reserve(&mut self, tokens: usize) -> Result<(), String> {
        s(GlmKv::reserve(self, tokens))
    }
    fn bytes(&self) -> usize {
        GlmKv::bytes(self)
    }
    fn reset(&mut self) {
        if let Err(e) = GlmKv::reset(self) {
            eprintln!("glm53f-forward: slot reset failed: {e}");
        }
    }
    fn release(&mut self) {
        if let Err(e) = GlmKv::release(self) {
            eprintln!("glm53f-forward: slot release failed: {e}");
        }
    }
    fn mark(&self) -> Result<KvMark, String> {
        s(GlmKv::mark(self))
    }
    fn rewind(&mut self, to: usize, mark: &KvMark) -> Result<(), String> {
        s(GlmKv::rewind(self, to, mark))
    }
    fn fork(&mut self, src: &Self, to: usize, mark: &KvMark) -> Result<(), String> {
        s(GlmKv::fork(self, src, to, mark))
    }
    fn page_tokens(&self) -> usize {
        GlmKv::page_tokens(self)
    }
    fn page_bytes(&self) -> usize {
        GlmKv::page_bytes(self)
    }
    fn state_bytes(&self) -> usize {
        GlmKv::state_bytes(self)
    }
    fn export_page(&self, first: usize, n: usize, dst: &mut [u8]) -> Result<(), String> {
        s(GlmKv::export_page(self, first, n, dst))
    }
    fn export_state(&self, mark: &KvMark, dst: &mut [u8]) -> Result<(), String> {
        s(GlmKv::export_state(self, mark, dst))
    }
    fn sync(&self) -> Result<(), String> {
        s(GlmKv::sync(self))
    }
    fn import_page(&mut self, n: usize, src: &[u8]) -> Result<(), String> {
        s(GlmKv::import_page(self, n, src))
    }
    fn import_state(&mut self, tokens: usize, src: &[u8]) -> Result<(), String> {
        s(GlmKv::import_state(self, tokens, src))
    }
}

/// Drafts a DFlash2 block proposes (`block - 1`).
pub const DRAFTS: usize = 7;

/// The drafter's uniform for the draft at emitted-token `position` of a request seeded `seed`:
/// the SplitMix64 mix of the shell's target draw (`glm53f_coordinator::sampling::Sampling::draw`)
/// over a domain of the drafter's own, so it is independent of the target's draw at that
/// position; its top 24 bits over 2^24, in `[0, 1)`. A seeded request drafts the same tokens
/// every run.
pub fn draft_uniform(seed: u64, position: u64) -> f32 {
    const DOMAIN: u64 = 0x2d4f_1a6b_d3c5_9e17;
    let mut x = seed
        .wrapping_add(DOMAIN)
        .wrapping_add(position.wrapping_mul(0x9e37_79b9_7f4a_7c15))
        .wrapping_add(0x9e37_79b9_7f4a_7c15);
    x = (x ^ (x >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    x ^= x >> 31;
    (x >> 40) as f32 * (1.0 / 16_777_216.0)
}

/// Counters of the drafter's work, for logs and tests.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct DraftStats {
    /// Requests drafted for, and requests given no draft because their context is cold.
    pub drafted: u64,
    pub cold: u64,
    /// Drafts proposed (each request's cap at most).
    pub proposed: u64,
    /// Verify rounds committed, and their windows.
    pub rounds: u64,
    pub windows: u64,
    /// Drafts verified (every window's rows after its first) and kept (its kept rows after the
    /// first: the drafts the target agreed with, less any after a stop inside the run).
    pub verified: u64,
    pub kept: u64,
}

impl DraftStats {
    /// Kept drafts per verified draft.
    pub fn acceptance(&self) -> f64 {
        self.kept as f64 / self.verified.max(1) as f64
    }

    /// Tokens a window delivers on average (its kept rows).
    pub fn tokens_per_window(&self) -> f64 {
        (self.kept + self.windows) as f64 / self.windows.max(1) as f64
    }
}

/// Commits between two log lines of the drafter's counters.
const LOG_ROUNDS: u64 = 64;

/// A [`GlmForward`] with the shell's GPU sampler: the `ModelForward` the scheduler drives.
pub struct ServedForward {
    pub fwd: GlmForward,
    sampler: Sampler,
    /// Device bytes that should stay free outside the pool and the forward's buffers (the
    /// sampler's buffers, kernel modules loaded on first use); a shortfall below it comes off
    /// the pool's free pages in `free_bytes`.
    pub margin: usize,
    /// With a drafter: the most drafts a step proposes per request (0 to 7; the block is one
    /// more, 1 turning speculation off).
    pub max_drafts: usize,
    /// With a drafter: a request whose context lost rows drafts again once this many committed
    /// rows are back in it (at most its length).
    pub warm_rows: usize,
    /// With a drafter: sampled requests draft with the sampled walk (else the greedy walk).
    pub sampled_walk: bool,
    pub stats: DraftStats,
    /// The last verify's window lengths, for the counters at its commit.
    windows: Vec<usize>,
}

impl ServedForward {
    pub fn new(fwd: GlmForward) -> Result<ServedForward, String> {
        Ok(ServedForward {
            fwd,
            sampler: Sampler::new()?,
            margin: 512 << 20,
            max_drafts: DRAFTS,
            warm_rows: 64,
            sampled_walk: true,
            stats: DraftStats::default(),
            windows: Vec::new(),
        })
    }

    /// Whether `kv`'s drafter context is too thin to draft from: fewer readable rows than
    /// `warm_rows` (or its length, or the window's 2,047), which happens only after rows were
    /// lost (a fresh request's context holds all its rows); or not at the committed length.
    pub fn cold(&self, kv: &GlmKv) -> bool {
        match kv.draft_slot() {
            None => true,
            Some(d) => {
                let want = d.len().min(self.warm_rows).min(Dims::GLM53F.window - 1);
                d.len() != GlmKv::tokens(kv) || d.context_rows() < want
            }
        }
    }

    /// The tokens `picks` select from the last pass's logit rows, `greedy` its argmax.
    fn select(&mut self, greedy: Vec<u32>, picks: &[&Pick]) -> Result<Vec<u32>, String> {
        if picks.iter().all(|p| p.draw.is_none() && p.mask.is_none()) {
            return Ok(greedy);
        }
        // The pass has synchronized its stream (it downloaded the picks); the sampler runs on
        // the default stream.
        let owned: Vec<Pick> = picks.iter().map(|p| (*p).clone()).collect();
        self.sampler
            .select(self.fwd.device_logits(), VOCAB, SAMPLE_VOCAB, &owned)
    }
}

impl ModelForward for ServedForward {
    type Slot = GlmKv;

    fn limits(&self) -> Limits {
        Limits {
            vocab: VOCAB,
            sample_vocab: SAMPLE_VOCAB,
            batch_rows: self.fwd.prefill_rows(),
            block: if self.fwd.has_drafter() {
                self.max_drafts.min(DRAFTS) + 1
            } else {
                0
            },
        }
    }

    /// The page pool's free pages, in bytes: a slot grows into pages of the pool the `KvPool`
    /// allocated up front ([`KvSlot::need_bytes`] counts pages), and snapshot marks take pages of
    /// it too, so the device's free memory does not measure room for a request. Less any
    /// shortfall of the device's free memory below `margin` (allocations outside the pool and
    /// the forward's buffers); admission then evicts retained slots, which frees pages.
    fn free_bytes(&self) -> Result<usize, String> {
        let (free, _) = s(device::mem_info())?;
        let kv = &self.fwd.kv;
        let pages = kv.free_pages() * kv.config().layout.page_bytes;
        Ok(pages.saturating_sub(self.margin.saturating_sub(free)))
    }

    fn prefill(&mut self, segs: &mut [Segment<'_, GlmKv>]) -> Result<Vec<SegmentOut>, String> {
        if segs.iter().any(|g| !g.images.is_empty()) {
            return Err("this forward has no vision tower yet".into());
        }
        let total: usize = segs.iter().map(|g| g.tokens.len()).sum();
        let one_pass = segs.len() <= self.fwd.cfg.max_requests && total <= self.fwd.prefill_rows();
        let groups: Vec<std::ops::Range<usize>> = if one_pass {
            vec![0..segs.len()]
        } else {
            (0..segs.len()).map(|i| i..i + 1).collect()
        };
        let mut out = Vec::with_capacity(segs.len());
        for g in groups {
            let part = &mut segs[g];
            let greedy = {
                let mut pairs: Vec<(&mut GlmKv, &[u32])> =
                    part.iter_mut().map(|x| (&mut *x.slot, x.tokens)).collect();
                s(self.fwd.prefill(&mut pairs))?
            };
            let kept: Vec<Option<Vec<f32>>> = if part.iter().any(|x| x.keep_logits) {
                let all = s(self.fwd.logits(part.len()))?;
                part.iter()
                    .enumerate()
                    .map(|(i, x)| {
                        x.keep_logits
                            .then(|| all[i * VOCAB..(i + 1) * VOCAB].to_vec())
                    })
                    .collect()
            } else {
                vec![None; part.len()]
            };
            let picks: Vec<&Pick> = part.iter().map(|x| &x.pick).collect();
            let next = self.select(greedy, &picks)?;
            out.extend(
                next.into_iter()
                    .zip(kept)
                    .map(|(next, logits)| SegmentOut { next, logits }),
            );
        }
        Ok(out)
    }

    fn decode(&mut self, rows: &mut [DecodeRow<'_, GlmKv>]) -> Result<Vec<Token>, String> {
        let greedy = {
            let mut pairs: Vec<(&mut GlmKv, u32)> =
                rows.iter_mut().map(|r| (&mut *r.slot, r.token)).collect();
            s(self.fwd.decode(&mut pairs))?
        };
        let picks: Vec<&Pick> = rows.iter().map(|r| &r.pick).collect();
        self.select(greedy, &picks)
    }

    /// One drafter launch for every request in the step that can draft: at most `max` (and
    /// [`ServedForward::max_drafts`]) of the block's 7 proposals, with the drafter's per-draft
    /// confidence (a softmax over its 16 candidates) as `probs`. Cold contexts and requests with
    /// no room for a draft get an empty draft.
    fn draft(&mut self, rows: &mut [DraftRow<'_, GlmKv>]) -> Result<Vec<Draft>, String> {
        let mut out = vec![Draft::default(); rows.len()];
        let mut who = Vec::with_capacity(rows.len());
        for (i, r) in rows.iter().enumerate() {
            let max = r.max.min(self.max_drafts).min(DRAFTS);
            if max == 0 {
                continue;
            }
            if self.cold(r.slot) {
                self.stats.cold += 1;
                continue;
            }
            who.push((i, max));
        }
        if who.is_empty() {
            return Ok(out);
        }
        let walks: Vec<(f32, Vec<f32>)> = who
            .iter()
            .map(|&(i, _)| match rows[i].pick.draw {
                Some((s, pos)) if self.sampled_walk => (
                    s.temperature,
                    (0..DRAFTS as u64)
                        .map(|j| draft_uniform(s.seed, pos + j))
                        .collect(),
                ),
                _ => (0.0, Vec::new()),
            })
            .collect();
        let props = {
            let reqs: Vec<DraftReq<'_>> = who
                .iter()
                .zip(&walks)
                .map(|(&(i, _), (t, u))| DraftReq {
                    kv: &*rows[i].slot,
                    anchor: rows[i].last,
                    temperature: *t,
                    uniforms: u,
                })
                .collect();
            s(self.fwd.draft(&reqs))?
        };
        for (&(i, max), p) in who.iter().zip(props) {
            out[i] = Draft {
                tokens: p.tokens[..max].to_vec(),
                probs: p.conf[..max].to_vec(),
            };
            self.stats.drafted += 1;
            self.stats.proposed += max as u64;
        }
        Ok(out)
    }

    fn verify(&mut self, windows: &mut [Window<'_, GlmKv>]) -> Result<Vec<Vec<Token>>, String> {
        self.windows = windows.iter().map(|w| w.tokens.len()).collect();
        let greedy = {
            let mut pairs: Vec<(&mut GlmKv, &[u32])> = windows
                .iter_mut()
                .map(|w| (&mut *w.slot, w.tokens))
                .collect();
            s(self.fwd.verify(&mut pairs))?
        };
        let picks: Vec<&Pick> = windows.iter().flat_map(|w| w.picks.iter()).collect();
        let flat = self.select(greedy.concat(), &picks)?;
        let mut out = Vec::with_capacity(windows.len());
        let mut at = 0;
        for w in windows.iter() {
            out.push(flat[at..at + w.tokens.len()].to_vec());
            at += w.tokens.len();
        }
        Ok(out)
    }

    fn commit(&mut self, slots: &mut [&mut GlmKv], keep: &[usize]) -> Result<(), String> {
        s(self.fwd.commit(slots, keep))?;
        let windows = std::mem::take(&mut self.windows);
        if self.fwd.has_drafter() && windows.len() == keep.len() {
            let st = &mut self.stats;
            st.rounds += 1;
            st.windows += keep.len() as u64;
            st.verified += windows.iter().map(|&w| w as u64 - 1).sum::<u64>();
            st.kept += keep.iter().map(|&k| k as u64 - 1).sum::<u64>();
            if st.rounds.is_multiple_of(LOG_ROUNDS) {
                eprintln!(
                    "[drafter] {} rounds, {} windows: {} drafts verified, {} kept ({:.1}%), {:.2} \
                     tokens a window; {} requests drafted, {} cold",
                    st.rounds,
                    st.windows,
                    st.verified,
                    st.kept,
                    100.0 * st.acceptance(),
                    st.tokens_per_window(),
                    st.drafted,
                    st.cold
                );
            }
        }
        Ok(())
    }
}
