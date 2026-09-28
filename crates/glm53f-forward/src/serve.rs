//! The serving shell's traits for this forward (features `coordinator` and `cuda`):
//! `glm53f_coordinator::KvSlot` for [`GlmKv`], and `glm53f_coordinator::ModelForward` for
//! [`ServedForward`], a [`GlmForward`] with the shell's GPU sampler.
//!
//! - Every pass selects its tokens on the device: greedy rows take the forward's own argmax
//!   (the first index of the largest logit below 154,856, the shell's greedy contract); rows
//!   with a draw or a mask go through the shell's sampler on the pass's device logits.
//! - The model has no drafter yet ([`Limits::block`] is 0), so the scheduler decodes one token
//!   per step; `verify` and `commit` work for windows of up to 8 rows all the same.
//! - Admission sees the page pool: slots grow into the pages the `KvPool` allocated up front,
//!   so `free_bytes` is the pool's free pages, less any shortfall of device memory below the
//!   margin (marks and workspaces come from device memory).
//! - Images are refused (the vision tower is a later phase).

use glm53f_coordinator::gpu::Sampler;
use glm53f_coordinator::model::{
    DecodeRow, KvSlot, Limits, ModelForward, Pick, Segment, SegmentOut, Token, Window,
};

use crate::device;
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

/// A [`GlmForward`] with the shell's GPU sampler: the `ModelForward` the scheduler drives.
pub struct ServedForward {
    pub fwd: GlmForward,
    sampler: Sampler,
    /// Device bytes kept free for the forward's own growth (index and attention workspaces).
    pub margin: usize,
}

impl ServedForward {
    pub fn new(fwd: GlmForward) -> Result<ServedForward, String> {
        Ok(ServedForward {
            fwd,
            sampler: Sampler::new()?,
            margin: 512 << 20,
        })
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
            batch_rows: self.fwd.cfg.max_rows,
            block: 0,
        }
    }

    /// The page pool's free pages, in bytes: a slot grows into pages of the pool the `KvPool`
    /// allocated up front ([`KvSlot::need_bytes`] counts pages), so the device's free memory does
    /// not measure room for a request. Less any shortfall of the device's free memory below
    /// `margin`: marks and the forward's workspaces come from device memory, and admission then
    /// evicts retained slots, which frees both.
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
        let one_pass = segs.len() <= self.fwd.cfg.max_requests && total <= self.fwd.cfg.max_rows;
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

    fn verify(&mut self, windows: &mut [Window<'_, GlmKv>]) -> Result<Vec<Vec<Token>>, String> {
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
        s(self.fwd.commit(slots, keep))
    }
}
