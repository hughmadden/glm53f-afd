//! The seam between the drafter and the rest of the engine: what the target forward and the
//! scheduler call, and what they hand over. `README.md` ("Integration") maps each call to the
//! serving shell's passes (`ModelForward::prefill`, `decode`, `draft`, `verify`, `commit`,
//! `KvSlot::rewind`).
//!
//! A request's drafter state is a *slot*: its context ring and its committed length. The target
//! forward appends every committed row's taps to it; the scheduler asks for drafts after the last
//! verified token. Rows enter the context only when committed, so a verify window's rejected rows
//! never reach the drafter and nothing has to be rolled back after a verify. The one way back is
//! [`Drafter::rewind`], for a slot resumed at an earlier snapshot point.

use crate::reference::{Bf16Head, Context, DraftOptions, Reference};
use crate::selector::Pick;
use crate::weights::Weights;
use crate::{bf16, Dims};

/// Committed rows of one request, to append to its drafter context.
pub struct Append<'a, S> {
    pub slot: &'a mut S,
    /// BF16 bits `[rows][taps * hidden]`: for each committed row, in position order, the mean of
    /// the four mHC streams after target layers 5, 14, 24, 33 and 42, concatenated in that order.
    /// Row `i` sits at position `len(slot) + i`.
    pub taps: &'a [u16],
}

/// One request to draft for.
pub struct DraftRequest<'a, S> {
    pub slot: &'a S,
    /// The token at position `len(slot)`: the last verified token (the previous step's bonus, or
    /// the prompt's first sampled token), not yet in the context.
    pub anchor: u32,
    /// The target's embedding row of `anchor`, BF16 bits `[hidden]`.
    pub anchor_embed: &'a [u16],
    /// `<= 0`: greedy. Otherwise each draft is drawn from `softmax(score / temperature)`.
    pub temperature: f32,
    /// One uniform in `[0, 1)` per draft position (read only when sampling).
    pub uniforms: &'a [f32],
}

/// A drafter's proposal for one request.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Proposal {
    /// The drafts after the anchor, in order (`block - 1` of them).
    pub tokens: Vec<u32>,
    /// `softmax(score)` of each draft among its position's candidates at temperature 1: the
    /// confidence a verify-length policy can use (`glm53f_coordinator::model::Draft::probs`).
    pub conf: Vec<f32>,
    /// `[drafts][top_k]` candidates per position, descending logit.
    pub candidates: Vec<u32>,
    /// `[drafts][top_k]`: the distribution each draft was drawn from over its position's
    /// candidates (one-hot when greedy). Row `e` depends on the draft chosen at `e - 1`. A
    /// rejection-sampling verify needs it; an exact-match verify does not.
    pub q: Vec<f32>,
}

/// The calls a drafter serves.
pub trait Drafter {
    /// Per-request state (the context ring).
    type Slot;

    fn dims(&self) -> Dims;

    /// A slot with an empty context.
    fn new_slot(&mut self) -> Result<Self::Slot, String>;

    /// Committed rows in the slot's context: the next anchor's position.
    fn len(&self, slot: &Self::Slot) -> usize;

    /// Append committed rows, for any number of slots at once. Called after a prefill segment
    /// (its rows; only the last `window - 1` of a prompt are ever read), a decode step (one row)
    /// and a verify commit (the kept rows: the anchor and the accepted drafts).
    fn append(&mut self, rows: &mut [Append<'_, Self::Slot>]) -> Result<(), String>;

    /// Propose `block - 1` tokens per request. The slots do not change.
    fn draft(&mut self, reqs: &[DraftRequest<'_, Self::Slot>]) -> Result<Vec<Proposal>, String>;

    /// Return a slot to an earlier committed length. Rows the ring no longer holds (a rewind
    /// further back than the ring can serve) are left out of later drafts rather than read stale.
    fn rewind(&mut self, slot: &mut Self::Slot, len: usize) -> Result<(), String>;

    /// Empty a slot for a new request.
    fn reset(&mut self, slot: &mut Self::Slot);
}

/// The CPU reference behind the seam (for tests and as the model of the GPU drafter).
pub struct CpuDrafter {
    pub weights: Weights,
    /// The target's LM head, BF16 bits `[vocab][hidden]`.
    pub lm_head: Vec<u16>,
    /// The target's embedding row of the mask token, BF16 bits `[hidden]`.
    pub mask_embed: Vec<u16>,
    /// Token ids a draft may propose (see [`crate::SAMPLE_VOCAB`]).
    pub vocab_limit: usize,
    /// Round GEMM inputs and ring contents to BF16, as the GPU forward does.
    pub bf16_io: bool,
}

impl CpuDrafter {
    fn reference(&self) -> Reference<'_> {
        let mut r = Reference::new(&self.weights);
        r.bf16_io = self.bf16_io;
        r
    }
}

impl Drafter for CpuDrafter {
    type Slot = Context;

    fn dims(&self) -> Dims {
        self.weights.dims
    }

    fn new_slot(&mut self) -> Result<Context, String> {
        Ok(Context::new(self.weights.dims))
    }

    fn len(&self, slot: &Context) -> usize {
        slot.len()
    }

    fn append(&mut self, rows: &mut [Append<'_, Context>]) -> Result<(), String> {
        let r = self.reference();
        for a in rows.iter_mut() {
            if a.taps.len() % self.weights.dims.tap_width() != 0 {
                return Err(format!("append: {} values is not whole rows", a.taps.len()));
            }
            r.append(a.slot, &bf16::decode(a.taps));
        }
        Ok(())
    }

    fn draft(&mut self, reqs: &[DraftRequest<'_, Context>]) -> Result<Vec<Proposal>, String> {
        let d = self.weights.dims;
        let r = self.reference();
        let head = Bf16Head {
            weight: &self.lm_head,
            hidden: d.hidden,
        };
        let mask = bf16::decode(&self.mask_embed);
        let mut out = Vec::with_capacity(reqs.len());
        for q in reqs {
            let pick = if q.temperature > 0.0 {
                if q.uniforms.len() < d.drafts() {
                    return Err(format!(
                        "draft: {} uniforms for {} positions",
                        q.uniforms.len(),
                        d.drafts()
                    ));
                }
                Pick::Sample {
                    temperature: q.temperature,
                    uniforms: q.uniforms,
                }
            } else {
                Pick::Greedy
            };
            let opts = DraftOptions {
                vocab_limit: self.vocab_limit,
                pick,
            };
            let dr = r.draft(
                q.slot,
                q.anchor,
                &bf16::decode(q.anchor_embed),
                &mask,
                &head,
                &opts,
                None,
            );
            out.push(Proposal {
                tokens: dr.walk.tokens,
                conf: dr.walk.conf,
                candidates: dr.candidates,
                q: dr.walk.q,
            });
        }
        Ok(out)
    }

    fn rewind(&mut self, slot: &mut Context, len: usize) -> Result<(), String> {
        if len > slot.len() {
            return Err(format!("rewind to {len} past the context's {}", slot.len()));
        }
        slot.rewind(len);
        Ok(())
    }

    fn reset(&mut self, slot: &mut Context) {
        *slot = Context::new(self.weights.dims);
    }
}
