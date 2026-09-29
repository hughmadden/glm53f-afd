//! The cut between the serving shell and a model: [`KvSlot`] and [`ModelForward`].
//!
//! The shell (scheduler, slot pool, prefix index, host RAM tier, sampler, queue, API engine)
//! knows nothing about a model's layers. It sees two things:
//!
//! - a **slot** ([`KvSlot`]): one request's context state on the device, whatever layers it
//!   has, with its memory accounting, its saved positions and its host-tier image;
//! - a **forward** ([`ModelForward`]): batched passes over rows of several slots (prefill
//!   segments, decode rows, drafts, verify windows and the commit after them), each returning
//!   the token its [`Pick`] selects per row.
//!
//! # Two kinds of layer state
//!
//! A slot's state has two parts. The traits never name layers; they only rely on this split.
//!
//! - **Appendable** state grows by one record per token and can drop any suffix. In
//!   GLM-5.3-Flash these are the 11 DSA layers' MLA latent records and the indexer's pooled
//!   keys (one per complete pool of 4 tokens, the incomplete pool's keys held raw). In MiMo
//!   they were the global-attention rows. A drafter's context ring is appendable too.
//!   Appendable rows can be shared between slots, page by page, copy-on-write.
//! - **Positional** state exists for one position only and cannot go back. In GLM-5.3-Flash
//!   this is the 34 KDA layers' recurrent state and short-convolution window (141 MiB per
//!   request). In MiMo it was the sliding-window rings. Returning to an earlier position
//!   needs a copy saved there: a [`KvSlot::Mark`].
//!
//! # Positions and rewind
//!
//! [`KvSlot::tokens`] is the slot's committed length: its state is the model's state after
//! exactly that many tokens, and the next row goes at that position. A slot moves backwards
//! in exactly two ways; there is no general truncate.
//!
//! 1. **Inside an uncommitted verify window.** [`ModelForward::verify`] appends a window of
//!    `R` rows (the last token and `R - 1` drafts) as *pending*: the appendable layers hold
//!    the rows, and the positional state stays at the committed position. For KDA the fused
//!    chain runs from the committed state without writing it back and saves each row's replay
//!    inputs. [`ModelForward::commit`] then keeps the first `keep` rows: the positional state
//!    is rebuilt by replaying them from the committed state (KDA `replay_batch`, which gives the
//!    bits of `keep` serial steps), and the appendable layers drop the rows past them. Between
//!    the two calls [`KvSlot::pending`] is `R` and `tokens()` has not moved.
//! 2. **To a mark.** [`KvSlot::rewind`] restores the positional state a mark saved and drops
//!    the appendable rows past the mark's length: "restore the saved state and truncate the
//!    MLA pages". This is how a retained slot resumes at one of its snapshot points.
//!
//! Committed rows are never dropped any other way. When a request stops inside an accepted
//! run (an end-of-sequence token or its token budget), the scheduler commits only the rows of
//! the tokens it delivered, so the slot ends exactly at the request's history.
//!
//! # Snapshot points
//!
//! The scheduler keeps positions worth resuming at: the end of every prompt, the end of every
//! completed turn, and a prefill abandoned by its client. Each is a *point*: a length, a
//! [`KvSlot::Mark`] taken there, and what is known about the next token. A retained slot keeps
//! its appendable rows and its points on the device; a new prompt that starts with a point's
//! tokens resumes from it, either in the slot itself ([`KvSlot::rewind`]) or in a fresh slot
//! that shares the rows ([`KvSlot::fork`]). Under pressure a point moves to the host RAM tier:
//! the pages of its appendable rows ([`KvSlot::export_page`], shared between snapshots with
//! the same prefix) plus its mark's image ([`KvSlot::export_state`]). A restore reverses it
//! ([`KvSlot::import_page`], [`KvSlot::import_state`]).
//!
//! # Memory
//!
//! Admission reserves a request's prompt plus an output allowance ([`KvSlot::reserve`]), never
//! the model's whole context. [`KvSlot::need_bytes`] says what a reservation would cost and
//! [`ModelForward::free_bytes`] what the device has; when it does not fit the pool evicts
//! snapshot points, least recently used first, wherever they live (retained slots', and running
//! requests' marks, which the requests do without), each stored to the host RAM tier first when
//! it is on, until it fits; if nothing is left to evict, the request waits for running requests
//! to finish. A snapshot mark that finds too little room ([`KvSlot::mark_bytes`]) makes room the
//! same way, and is skipped only if it still does not fit. Nothing is evicted while nothing needs
//! the memory.
//!
//! # Selection
//!
//! Every pass takes a [`Pick`] per output row and returns the selected token id, so logits
//! need not leave the device. A model applies picks with the GPU sampler (`crate::gpu`, feature
//! `cuda`) or the CPU reference ([`crate::sampling::select_pick`]); both implement the contract
//! in [`crate::sampling`]: greedy is the first index of the maximum; a draw is a function of
//! `(seed, position)`; only ids below [`Limits::sample_vocab`] can come out; a mask removes
//! the tokens it does not allow.
//!
//! # Threads
//!
//! The scheduler owns the forward and every slot on its own thread; nothing here is shared.
//! A forward and its slots must be [`Send`] to move there.

use std::sync::Arc;

use glm53f_api::engine::ImageInput;

use crate::sampling::{Mask, Sampling};

/// A token id. Ids at or above [`IMAGE_ID_BASE`] stand for image rows.
pub type Token = u32;

/// Token ids with bit 31 set stand for the rows of an image (`crate::engine::image_token_id`):
/// past any vocabulary, and a function of the image's bytes, so the prefix index (which
/// compares ids) shares a prompt only when its images are the same.
pub const IMAGE_ID_BASE: Token = 0x8000_0000;

/// An image in a prompt: the prompt's tokens `[start, start + image.tokens)` stand for its
/// encoded rows.
#[derive(Clone, Debug)]
pub struct ImageSpan {
    pub start: usize,
    pub image: Arc<ImageInput>,
}

/// How one logit row becomes a token (the contract of [`crate::sampling`]).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Pick {
    /// `None`: greedy, the first index of the row's maximum. `Some((s, position))`: the draw
    /// of sampling `s` for emitted-token `position` (0 is the first token after the prompt).
    pub draw: Option<(Sampling, u64)>,
    /// The tokens the row may produce (a grammar's mask); `None`: every token.
    pub mask: Option<Mask>,
}

impl Pick {
    /// The argmax.
    pub fn greedy() -> Pick {
        Pick::default()
    }

    /// A request's pick for emitted-token `position`: the argmax when it is greedy (`None`).
    pub fn at(sampling: Option<Sampling>, position: u64) -> Pick {
        Pick { draw: sampling.map(|s| (s, position)), mask: None }
    }
}

/// A run of prompt tokens appended to a slot ([`ModelForward::prefill`]).
pub struct Segment<'a, S> {
    pub slot: &'a mut S,
    /// Appended at position `slot.tokens()`, and committed.
    pub tokens: &'a [Token],
    /// The prompt's images whose rows these tokens reach (positions are absolute). Ids at or
    /// above [`IMAGE_ID_BASE`] are rows of one of them.
    pub images: &'a [ImageSpan],
    /// The selection for the token after the segment's last row.
    pub pick: Pick,
    /// Also return the last row's logits, before any mask (a prompt snapshot keeps them so a
    /// sampled request resuming exactly there draws its first token without a forward).
    pub keep_logits: bool,
}

/// What [`ModelForward::prefill`] returns for one segment.
#[derive(Clone, Debug, PartialEq)]
pub struct SegmentOut {
    /// The segment's [`Segment::pick`] applied to its last row.
    pub next: Token,
    /// The last row's logits, `[Limits::vocab]`, when [`Segment::keep_logits`] asked for them.
    pub logits: Option<Vec<f32>>,
}

/// One decode row: `token` appended at `slot.tokens()` and committed ([`ModelForward::decode`]).
pub struct DecodeRow<'a, S> {
    pub slot: &'a mut S,
    pub token: Token,
    /// The selection for the token after it.
    pub pick: Pick,
}

/// One request to draft for: `last` is the token at position `slot.tokens()`, not yet in the
/// slot ([`ModelForward::draft`]).
pub struct DraftRow<'a, S> {
    pub slot: &'a mut S,
    pub last: Token,
    /// The most drafts worth proposing (`Limits::block - 1` at most).
    pub max: usize,
    /// The pick of the verify window's first row, which the first draft is checked against:
    /// greedy, or the request's draw at the first draft's emitted-token position (draft `j`,
    /// from 0, stands at that position plus `j`). A drafter may sample its proposals with it; a
    /// draft is accepted only when it equals the target's own pick either way.
    pub pick: Pick,
}

/// A drafter's proposal for one request.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Draft {
    /// The proposed tokens after `last`, in order.
    pub tokens: Vec<Token>,
    /// The drafter's probability of each, for the verify-length policy (`crate::spec`).
    pub probs: Vec<f32>,
}

/// One verify window ([`ModelForward::verify`]): `tokens` (`[last, d1, .., dk]`) appended as
/// pending rows at `slot.tokens()`.
pub struct Window<'a, S> {
    pub slot: &'a mut S,
    pub tokens: &'a [Token],
    /// One per row: row `j` selects the token after `tokens[j]`.
    pub picks: &'a [Pick],
}

/// What a forward can take.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Limits {
    /// Length of a logit row: the LM head's rows, padding included (154,880 for GLM-5.3-Flash).
    pub vocab: usize,
    /// Token ids a pick may return: one past the tokenizer's largest id (154,856 for
    /// GLM-5.3-Flash). Greedy and draws never return an id at or above it, so the LM head's
    /// padding rows never come out.
    pub sample_vocab: usize,
    /// Rows a batched prefill of several short prompts may hold in all. Prompts longer than
    /// this prefill one segment at a time.
    pub batch_rows: usize,
    /// Rows of a verify window: the last token plus up to `block - 1` drafts. 0 or 1: the
    /// model has no drafter and decodes one token per step.
    pub block: usize,
}

impl Limits {
    /// Drafts a window can carry.
    pub fn drafts(&self) -> usize {
        self.block.saturating_sub(1)
    }
}

/// One request's context state on the device. See the module documentation for the
/// appendable/positional split and the two ways a slot rewinds.
///
/// A slot starts empty (`tokens() == 0`) with some base capacity. The pool moves it between
/// requests: [`KvSlot::reset`] empties it for a new request, [`KvSlot::release`] also gives
/// back memory it grew past its base.
pub trait KvSlot {
    /// The positional state saved at one position, taken by [`KvSlot::mark`]. Usually a
    /// device copy (GLM-5.3-Flash: the KDA recurrent and convolution states, about 141 MiB);
    /// an implementation may keep it in page-locked host memory instead and copy it back in
    /// [`KvSlot::rewind`] and [`KvSlot::fork`]. It does not hold appendable rows: those stay in
    /// the slot the mark was taken in. Dropping a mark frees what it holds; the scheduler drops
    /// a point's mark when the point is evicted or its request fails.
    type Mark;

    // ---- Position ----------------------------------------------------------------------

    /// Committed tokens: the position of the next row.
    fn tokens(&self) -> usize;

    /// Rows of an uncommitted verify window ([`ModelForward::verify`]); 0 outside one.
    fn pending(&self) -> usize;

    // ---- Memory ------------------------------------------------------------------------

    /// Tokens the slot can hold without growing.
    fn capacity(&self) -> usize;

    /// Device bytes, beyond what the slot holds now, that holding `tokens` tokens and running
    /// one prefill in it can take at the peak: the growth to `tokens`, a transient while it
    /// grows (MiMo reallocated a layer's buffers and held both while copying), and a prefill
    /// working set the slot itself allocates. 0 for a slot that has all of it already.
    fn need_bytes(&self, tokens: usize) -> usize;

    /// Grow to hold at least `tokens` tokens. The scheduler calls it once at admission with the
    /// prompt plus the output allowance, again before a decode step when a request is about to
    /// outgrow it, and before a restore. It keeps every row. `Err` when the device cannot.
    fn reserve(&mut self, tokens: usize) -> Result<(), String>;

    /// Device bytes the slot holds now (its rows, shared ones included, and its positional
    /// state), for accounting and logs.
    fn bytes(&self) -> usize;

    /// A prefill is about to run in this slot: allocate any working set it needs now, outside
    /// the forward (MiMo grew its sliding-window rings to window + chunk here). Default: none.
    fn begin_prefill(&mut self) -> Result<(), String> {
        Ok(())
    }

    /// The prompt's prefill is over (finished or abandoned): give the working set back.
    /// Default: nothing to do.
    fn end_prefill(&mut self) {}

    /// Drop every token (a fresh request). Keeps the capacity.
    fn reset(&mut self);

    /// Drop every token (pending rows too) and give back memory grown past the slot's base
    /// capacity, a prefill working set and shared pages included (the slot goes to the free
    /// list). Called on slots in any state, also after a failed pass or a failed fork.
    fn release(&mut self);

    // ---- Marks and rewind --------------------------------------------------------------

    /// Save the positional state at the current position (`tokens()`, with no pending rows).
    /// `Err` when the device has no room for the copy: the pool then makes room
    /// ([`KvSlot::mark_bytes`]) and tries once more, and goes without the point if that fails too.
    fn mark(&self) -> Result<Self::Mark, String>;

    /// Device bytes a mark takes, as [`ModelForward::free_bytes`] counts them: a [`KvSlot::mark`]
    /// that finds too little room has snapshot points evicted until this much is free.
    fn mark_bytes(&self) -> usize;

    /// Go back to `to` tokens, where `mark` was taken in this slot: restore the mark's
    /// positional state and drop the appendable rows past `to`. `to <= tokens()`. The rows
    /// before `to` must be the ones the mark was taken after (the scheduler only rewinds to
    /// marks of this slot's own history).
    fn rewind(&mut self, to: usize, mark: &Self::Mark) -> Result<(), String>;

    /// Start this slot (reset, and reserved for its request) as `src`'s first `to` tokens at
    /// `mark`, a mark `src` took at `to`: the appendable rows `[0, to)` shared with `src`
    /// copy-on-write where the implementation pages them (a partly filled page is copied), or
    /// copied, and the mark's positional state loaded. Afterwards neither slot's writes change
    /// the other's rows: a slot that appends into, or rewinds below, a shared page copies it
    /// first.
    fn fork(&mut self, src: &Self, to: usize, mark: &Self::Mark) -> Result<(), String>;

    // ---- Host tier image ---------------------------------------------------------------

    /// Tokens per host page (64 for GLM-5.3-Flash: 16 indexer pools). Pages are shared between
    /// host snapshots whose tokens agree up to the page's end.
    fn page_tokens(&self) -> usize;

    /// Bytes of one host page: `page_tokens()` tokens of every appendable layer (a partial last
    /// page uses the same size). GLM-5.3-Flash with an FP8 cache: 64 x 6,171 B.
    fn page_bytes(&self) -> usize;

    /// Bytes of a mark's host image ([`KvSlot::export_state`]).
    fn state_bytes(&self) -> usize;

    /// Copy appendable rows `[first, first + n)` into a host page. `first` is a multiple of
    /// `page_tokens()` and `n <= page_tokens()`; `n` is short only for the last page, which then
    /// also carries whatever the implementation keeps for an incomplete unit (GLM: the raw
    /// indexer keys of an incomplete pool). The copy may complete asynchronously until
    /// [`KvSlot::sync`].
    fn export_page(&self, first: usize, n: usize, dst: &mut [u8]) -> Result<(), String>;

    /// Copy `mark`'s positional state into a host buffer of `state_bytes()`.
    fn export_state(&self, mark: &Self::Mark, dst: &mut [u8]) -> Result<(), String>;

    /// Wait for the copies queued by [`KvSlot::export_page`]. Default: they are synchronous.
    fn sync(&self) -> Result<(), String> {
        Ok(())
    }

    /// Append `n` appendable rows from a host page written by [`KvSlot::export_page`], at the
    /// slot's next position (a restore runs pages in order from a reset, reserved slot).
    fn import_page(&mut self, n: usize, src: &[u8]) -> Result<(), String>;

    /// Finish a restore of `tokens` tokens: load the positional state from `src` (written by
    /// [`KvSlot::export_state`]) and set the committed length to `tokens`.
    fn import_state(&mut self, tokens: usize, src: &[u8]) -> Result<(), String>;
}

/// A model's batched passes over slots. Every pass may be given rows of several slots; a
/// model is free to split them into lanes (MiMo ran two, each lane's attention overlapping the
/// other's expert exchange) as long as each row's result does not depend on the batch.
///
/// The passes and what they do to a slot:
///
/// | Pass | Rows | Slot afterwards |
/// |---|---|---|
/// | [`prefill`](Self::prefill) | a segment per slot | `tokens() += segment`, committed |
/// | [`decode`](Self::decode) | one per slot | `tokens() += 1`, committed |
/// | [`draft`](Self::draft) | none of the target's | unchanged (the drafter may keep scratch) |
/// | [`verify`](Self::verify) | a window of `R` per slot | `pending() == R`, `tokens()` unchanged |
/// | [`commit`](Self::commit) | none | `tokens() += keep`, `pending() == 0` |
///
/// A pass that fails leaves its slots in an unknown state: the scheduler fails their requests
/// and releases the slots.
pub trait ModelForward {
    /// The model's per-request state.
    type Slot: KvSlot;

    /// What this forward can take.
    fn limits(&self) -> Limits;

    /// Device bytes available for slot growth now, already less whatever margin the model keeps
    /// for its own allocations (MiMo: `cudaMemGetInfo` less 512 MiB). Admission compares
    /// [`KvSlot::need_bytes`] against it.
    fn free_bytes(&self) -> Result<usize, String>;

    /// Device bytes the model needs free to encode `images` during a prefill (a vision tower's
    /// transient weights and buffers; 0 for images it has encoded already, e.g. rows it keeps
    /// per slot across a prompt's segments). The scheduler evicts snapshot points until they are
    /// free before each segment that reaches an image. Default: 0.
    fn image_bytes(&self, images: &[ImageSpan]) -> usize {
        let _ = images;
        0
    }

    /// Append and commit each segment's tokens to its slot. One segment of any length is one
    /// prompt's next segment (the model cuts it into chunks as it needs); several segments are
    /// short prompts batched into one pass (together at most [`Limits::batch_rows`] rows).
    /// Returns one [`SegmentOut`] per segment, in order.
    fn prefill(&mut self, segs: &mut [Segment<'_, Self::Slot>]) -> Result<Vec<SegmentOut>, String>;

    /// Append and commit one token per slot; returns each row's pick.
    fn decode(&mut self, rows: &mut [DecodeRow<'_, Self::Slot>]) -> Result<Vec<Token>, String>;

    /// Propose up to `row.max` drafts per request. Only called when [`Limits::block`] > 1.
    /// Drafting reads the slot's drafter context and must not change what the target sees.
    fn draft(&mut self, rows: &mut [DraftRow<'_, Self::Slot>]) -> Result<Vec<Draft>, String> {
        let _ = rows;
        Err("this model has no drafter".into())
    }

    /// Append each window as pending rows (see the module documentation) and return every row's
    /// pick, per window, in row order. Row `j`'s pick is computed from the rows up to `j` only
    /// (causal), so it equals what serial decoding would select after `tokens[..=j]`.
    fn verify(&mut self, windows: &mut [Window<'_, Self::Slot>]) -> Result<Vec<Vec<Token>>, String>;

    /// Keep the first `keep[i]` pending rows of `slots[i]` (`1 <= keep[i] <= pending()`): replay
    /// them into the positional state from the committed state, drop the appendable rows past
    /// them, and give the kept rows to the drafter as context. Afterwards `tokens()` has grown
    /// by `keep[i]` and `pending() == 0`. An appendable layer that summarizes rows in units
    /// undoes a unit the dropped rows completed (GLM-5.3-Flash: an indexer pool of 4 closed
    /// inside the window but cut by the commit goes back to raw keys).
    fn commit(&mut self, slots: &mut [&mut Self::Slot], keep: &[usize]) -> Result<(), String>;

    /// Apply `pick` to one logit row held on the host (a prompt snapshot's kept logits: a
    /// sampled request resuming exactly there draws its first token without a forward). A GPU
    /// model should run its device sampler here, so the draw has the bits a forward would give.
    /// Default: the CPU reference.
    fn select_host(&mut self, logits: &[f32], pick: &Pick) -> Result<Token, String> {
        Ok(crate::sampling::select_pick(logits, self.limits().sample_vocab, pick))
    }
}
