//! The GLM device KV's layout, byte accounting and page bookkeeping, on the host.
//!
//! [`crate::kv`] (feature `cuda`) holds the device memory; everything here is plain data, so
//! the accounting and the copy-on-write rules are tested without a GPU.
//!
//! # Layout
//!
//! - **Pages** (appendable state): one physical page holds 64 tokens of every DSA layer: layer
//!   `j`'s 35,904-byte block (64 latent records of 528 B, 16 pooled keys of 128 B, 16 f32
//!   scales; `glm53f_dsa::cache`) at byte `35,904 j`. Over the model's 11 DSA layers a page is
//!   394,944 B = 64 x 6,171 B, the planner's `KvGeometry::page_bytes`.
//! - **Positional state**, per slot: the FP32 KDA states `[kda_layers][64][128 (v)][128 (k)]`,
//!   the BF16 conv windows `[kda_layers][3][24,576]`, and the DSA tails `[dsa_layers][1,552 B]`
//!   (the raw index keys and gates of the incomplete pool). A mark saves all three: the tail
//!   belongs to a position, like the KDA state, because the keys it holds are dropped once
//!   their pool completes.
//! - **Draft KV** (DFlash2): the drafter's context ring, reserved per slot when a drafter is
//!   configured (`crate::kv` keeps it at the committed length).
//!
//! # Pages and copy-on-write
//!
//! [`PageAlloc`] counts references per physical page. [`SlotPages`] maps a slot's logical
//! pages to physical ones. A fork shares the source's full pages and copies its partial last
//! page; a slot that is about to write into a shared page, or rewinds into one, gets its own
//! copy first ([`SlotPages::prepare_write`], [`SlotPages::rewind`]). The bookkeeping returns
//! the page copies ([`PageCopy`]) for the device side to run.

use glm53f_dsa::cache::{PAGE_LAYER_BYTES, PAGE_TOKENS, TAIL_BYTES};
use glm53f_model::config::DraftConfig;

use crate::error::{Error, Result};
use crate::shape::{ModelShape, KDA_DIM, KDA_HEADS, KDA_QKV, KDA_WINDOW};

/// Bytes of one DSA layer's block in a page.
pub const LAYER_PAGE_BYTES: usize = PAGE_LAYER_BYTES;
/// Tokens per page.
pub const PAGE: usize = PAGE_TOKENS;
/// Bytes of one DSA layer's tail.
pub const TAIL: usize = TAIL_BYTES;

/// Byte layout of the device KV for a model shape.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct KvLayout {
    pub kda_layers: usize,
    pub dsa_layers: usize,
    /// Bytes of one physical page (every DSA layer's 64-token block).
    pub page_bytes: usize,
    /// Bytes of the DFlash2 draft KV reserved per slot (0 without a drafter).
    pub draft_kv_bytes: usize,
}

impl KvLayout {
    pub fn new(shape: &ModelShape, draft: Option<&DraftConfig>) -> KvLayout {
        KvLayout {
            kda_layers: shape.kda_layers,
            dsa_layers: shape.dsa_layers,
            page_bytes: shape.dsa_layers.max(1) * LAYER_PAGE_BYTES,
            draft_kv_bytes: draft.map_or(0, |d| d.kv_bytes_per_slot() as usize),
        }
    }

    /// f32 elements of one layer's KDA state.
    pub const fn state_elems_per_layer() -> usize {
        KDA_HEADS * KDA_DIM * KDA_DIM
    }

    /// BF16 elements of one layer's conv window.
    pub const fn conv_elems_per_layer() -> usize {
        KDA_WINDOW * KDA_QKV
    }

    pub fn kda_state_bytes(&self) -> usize {
        self.kda_layers * Self::state_elems_per_layer() * 4
    }

    pub fn conv_bytes(&self) -> usize {
        self.kda_layers * Self::conv_elems_per_layer() * 2
    }

    pub fn tails_bytes(&self) -> usize {
        self.dsa_layers * TAIL
    }

    /// A mark's positional state: KDA states, conv windows and DSA tails (also the size of its
    /// host image).
    pub fn mark_bytes(&self) -> usize {
        self.kda_state_bytes() + self.conv_bytes() + self.tails_bytes()
    }

    /// A mark's three parts (KDA states, conv windows, DSA tails): `(offset in its host image,
    /// bytes, pool pages)`. On the device a mark is held in pages of the pool, each part from a
    /// page of its own.
    pub fn mark_regions(&self) -> [(usize, usize, usize); 3] {
        let pb = self.page_bytes;
        let (s, c, t) = (
            self.kda_state_bytes(),
            self.conv_bytes(),
            self.tails_bytes(),
        );
        [
            (0, s, s.div_ceil(pb)),
            (s, c, c.div_ceil(pb)),
            (s + c, t, t.div_ceil(pb)),
        ]
    }

    /// Pool pages one mark takes (GLM-5.3-Flash: 376 pages, 141 MiB).
    pub fn mark_pages(&self) -> usize {
        self.mark_regions().iter().map(|r| r.2).sum()
    }

    /// Device bytes a slot holds whatever its length: its positional state and draft KV.
    pub fn slot_fixed_bytes(&self) -> usize {
        self.mark_bytes() + self.draft_kv_bytes
    }

    /// Pages that hold `tokens` tokens.
    pub fn pages_for(tokens: usize) -> usize {
        tokens.div_ceil(PAGE)
    }

    /// Device bytes of the pages that hold `tokens` tokens.
    pub fn paged_bytes(&self, tokens: usize) -> usize {
        Self::pages_for(tokens) * self.page_bytes
    }
}

/// Reference counts and the free list of a pool of physical pages.
#[derive(Clone, Debug)]
pub struct PageAlloc {
    free: Vec<u32>,
    refs: Vec<u32>,
}

impl PageAlloc {
    pub fn new(pages: usize) -> PageAlloc {
        PageAlloc {
            // Popped from the end: the lowest page first.
            free: (0..pages as u32).rev().collect(),
            refs: vec![0; pages],
        }
    }

    pub fn total(&self) -> usize {
        self.refs.len()
    }

    pub fn free(&self) -> usize {
        self.free.len()
    }

    pub fn refs(&self, page: u32) -> u32 {
        self.refs[page as usize]
    }

    pub fn alloc(&mut self) -> Option<u32> {
        let p = self.free.pop()?;
        debug_assert_eq!(self.refs[p as usize], 0);
        self.refs[p as usize] = 1;
        Some(p)
    }

    /// One more slot maps `page`.
    pub fn share(&mut self, page: u32) {
        assert!(self.refs[page as usize] > 0, "sharing a free page");
        self.refs[page as usize] += 1;
    }

    /// One slot fewer maps `page`; true when it became free.
    pub fn release(&mut self, page: u32) -> bool {
        let r = &mut self.refs[page as usize];
        assert!(*r > 0, "releasing a free page");
        *r -= 1;
        if *r == 0 {
            self.free.push(page);
            true
        } else {
            false
        }
    }
}

/// A whole-page device copy the bookkeeping asks for.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PageCopy {
    pub src: u32,
    pub dst: u32,
}

/// What a bookkeeping step changed: page copies to run before anything reads the new pages,
/// and the lowest page-table index whose entry changed (for the device table upload).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct PageChange {
    pub copies: Vec<PageCopy>,
    pub first_changed: Option<usize>,
}

impl PageChange {
    fn touch(&mut self, index: usize) {
        self.first_changed = Some(self.first_changed.map_or(index, |f| f.min(index)));
    }
}

fn no_pages(need: usize, free: usize) -> Error {
    Error::OutOfMemory(format!("the KV pool has {free} free pages, {need} needed"))
}

/// One slot's logical-to-physical page table.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct SlotPages {
    pub pages: Vec<u32>,
}

impl SlotPages {
    /// Logical pages held.
    pub fn len(&self) -> usize {
        self.pages.len()
    }

    pub fn is_empty(&self) -> bool {
        self.pages.is_empty()
    }

    /// Tokens the slot can hold without growing.
    pub fn capacity(&self) -> usize {
        self.pages.len() * PAGE
    }

    /// Grow to at least `pages` logical pages (fresh pages, owned alone).
    pub fn reserve(&mut self, alloc: &mut PageAlloc, pages: usize) -> Result<PageChange> {
        let mut ch = PageChange::default();
        let need = pages.saturating_sub(self.pages.len());
        if need > alloc.free() {
            return Err(no_pages(need, alloc.free()));
        }
        while self.pages.len() < pages {
            ch.touch(self.pages.len());
            self.pages.push(alloc.alloc().expect("counted"));
        }
        Ok(ch)
    }

    /// Before a pass writes tokens `[first, end)`: make their pages exist and belong to this
    /// slot alone. Missing pages are allocated; a shared page is copied into a fresh one and
    /// the shared reference dropped. Nothing changes when the pool cannot cover it.
    pub fn prepare_write(
        &mut self,
        alloc: &mut PageAlloc,
        first: usize,
        end: usize,
    ) -> Result<PageChange> {
        let mut ch = PageChange::default();
        if end <= first {
            return Ok(ch);
        }
        let (lo, hi) = (first / PAGE, (end - 1) / PAGE);
        let shared = (lo..=hi.min(self.pages.len().saturating_sub(1)))
            .filter(|&i| i < self.pages.len() && alloc.refs(self.pages[i]) > 1)
            .count();
        let missing = (hi + 1).saturating_sub(self.pages.len());
        if shared + missing > alloc.free() {
            return Err(no_pages(shared + missing, alloc.free()));
        }
        for i in lo..=hi {
            if i < self.pages.len() {
                let p = self.pages[i];
                if alloc.refs(p) > 1 {
                    let q = alloc.alloc().expect("counted");
                    ch.copies.push(PageCopy { src: p, dst: q });
                    alloc.release(p);
                    self.pages[i] = q;
                    ch.touch(i);
                }
            } else {
                while self.pages.len() <= i {
                    ch.touch(self.pages.len());
                    self.pages.push(alloc.alloc().expect("counted"));
                }
            }
        }
        Ok(ch)
    }

    /// Going back to `to` tokens: the page holding token `to - 1` .. is written again later, so
    /// every page from the one holding position `to` on must belong to this slot alone. A
    /// shared page that still holds kept rows (`to` inside it) is copied; a shared page past
    /// them is replaced by a fresh page (its rows are dead).
    pub fn rewind(&mut self, alloc: &mut PageAlloc, to: usize) -> Result<PageChange> {
        let mut ch = PageChange::default();
        let lo = to / PAGE;
        let shared: Vec<usize> = (lo..self.pages.len())
            .filter(|&i| alloc.refs(self.pages[i]) > 1)
            .collect();
        if shared.len() > alloc.free() {
            return Err(no_pages(shared.len(), alloc.free()));
        }
        for i in shared {
            let p = self.pages[i];
            let q = alloc.alloc().expect("counted");
            if i == lo && !to.is_multiple_of(PAGE) {
                ch.copies.push(PageCopy { src: p, dst: q });
            }
            alloc.release(p);
            self.pages[i] = q;
            ch.touch(i);
        }
        Ok(ch)
    }

    /// Become `src`'s first `to` tokens: its full pages shared, its partial last page (if
    /// any) copied into a page of this slot's own. This slot's own pages below that are given
    /// back; the ones past it are kept (capacity).
    pub fn fork(
        &mut self,
        alloc: &mut PageAlloc,
        src: &SlotPages,
        to: usize,
    ) -> Result<PageChange> {
        let mut ch = PageChange::default();
        let full = to / PAGE;
        let partial = !to.is_multiple_of(PAGE);
        if src.pages.len() < full + usize::from(partial) {
            return Err(Error::Invalid(format!(
                "fork of {to} tokens from a slot with {} pages",
                src.pages.len()
            )));
        }
        // The partial page lands in this slot's own page at index `full` when it has one that
        // no other slot maps, else in a fresh page.
        let own = self.pages.len() > full && alloc.refs(self.pages[full]) == 1;
        let need_fresh = usize::from(partial && !own);
        if need_fresh > alloc.free() {
            return Err(no_pages(need_fresh, alloc.free()));
        }
        for i in 0..full {
            let s = src.pages[i];
            alloc.share(s);
            if i < self.pages.len() {
                alloc.release(self.pages[i]);
                self.pages[i] = s;
            } else {
                self.pages.push(s);
            }
            ch.touch(i);
        }
        if partial {
            if !own {
                let q = alloc.alloc().expect("counted");
                if self.pages.len() > full {
                    alloc.release(self.pages[full]);
                    self.pages[full] = q;
                } else {
                    self.pages.push(q);
                }
            }
            ch.copies.push(PageCopy {
                src: src.pages[full],
                dst: self.pages[full],
            });
            ch.touch(full);
        }
        Ok(ch)
    }

    /// Give back the pages from logical index `keep` on.
    pub fn truncate(&mut self, alloc: &mut PageAlloc, keep: usize) {
        while self.pages.len() > keep {
            let p = self.pages.pop().unwrap();
            alloc.release(p);
        }
    }

    /// Whether logical page `index` is shared with another slot.
    pub fn is_shared(&self, alloc: &PageAlloc, index: usize) -> bool {
        alloc.refs(self.pages[index]) > 1
    }
}
