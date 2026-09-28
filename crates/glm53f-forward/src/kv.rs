//! The GLM device KV (feature `cuda`): a pool of pages and per-slot positional state shared by
//! every request, and [`GlmKv`], one request's view of it.
//!
//! Layout and accounting are [`crate::kvplan`]'s. On the device:
//!
//! - `pages`: `pages x page_bytes`, the paged MLA latents and pooled index keys of every DSA
//!   layer (layer `j`'s block of physical page `p` at `p * page_bytes + 35,904 j`, which is the
//!   `glm53f_dsa_cache_t` view with `page_stride = page_bytes`);
//! - `table`: the page tables, `[max_slots][max_pages]` i32, one row per slot;
//! - `state`: FP32 KDA states `[max_slots][kda_layers][64][128][128]`;
//! - `conv`: BF16 conv windows `[max_slots][kda_layers][3][24,576]`;
//! - `tails`: DSA tails `[max_slots][dsa_layers][1,552 B]`;
//! - `draft`: the DFlash2 drafter's context rings, `[max_slots][draft_kv_bytes]` (when the layout
//!   has a drafter): each `[5 layers][K, V][2,056 rows][1,024]` BF16, 40.16 MiB.
//!
//! Every device operation is ordered on the pool's stream, which the forward (and its drafter)
//! shares.
//!
//! [`GlmKv`] mirrors the serving shell's `KvSlot` contract: committed tokens and pending verify
//! rows; `reserve`; marks of the positional state; `rewind` to a mark; `fork` from another slot
//! with copy-on-write pages; host images of pages and marks. The DSA tail travels with the
//! positional state (in a mark and its host image), not in the last page: a 63-token page has
//! 660 free bytes per layer, and a tail of 3 tokens needs 1,552.
//!
//! **The drafter's context** ([`GlmKv::draft_slot`], with a drafter in the layout) is the ring of
//! the committed rows' keys and values (`glm53f_dflash::gpu::GpuSlot` over the slot's `draft`
//! region), kept at the committed length: the forward appends every committed row, and the
//! slot's own moves follow `glm53f_dflash`'s rules. The ring is positional (row `p % 2,056` holds
//! position `p`, the last 2,048 positions stay intact) and is not saved in marks:
//!
//! | `KvSlot` call | The drafter's context |
//! |---|---|
//! | `reserve` | unchanged (the ring is fixed state, allocated with the pool) |
//! | `reset`, `release` | emptied |
//! | `rewind(to)` | rewound: positions the ring still holds (at or above `len - 2,048`) are kept, lower ones masked out |
//! | `fork(src, to)` | `src`'s ring copied, then rewound to `to` the same way |
//! | `import_page`, `import_state(tokens)` | restarted cold at `tokens` (the host image has no taps): drafts read only rows appended afterwards |

use std::sync::{Arc, Mutex, MutexGuard};

use glm53f_dflash::gpu::GpuSlot;
use glm53f_dflash::Dims;

use crate::device::{DeviceBuffer, Stream};
use crate::error::{invalid, Error, Result};
use crate::kvplan::{KvLayout, PageAlloc, PageChange, SlotPages, LAYER_PAGE_BYTES, PAGE, TAIL};

/// How big the pool is.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct KvConfig {
    pub layout: KvLayout,
    /// Slots (requests with device state at once).
    pub max_slots: usize,
    /// Physical pages in the pool.
    pub pages: usize,
    /// Most pages one slot maps (the page-table row width; a multiple of 4).
    pub max_pages: usize,
    /// Pages every slot keeps across `reset` (its base capacity).
    pub base_pages: usize,
}

impl KvConfig {
    /// Device bytes the pool allocates.
    pub fn device_bytes(&self) -> usize {
        let l = &self.layout;
        self.pages * l.page_bytes + self.max_slots * (self.max_pages * 4 + l.slot_fixed_bytes())
    }
}

struct Alloc {
    pages: PageAlloc,
    free_slots: Vec<usize>,
}

pub(crate) struct KvShared {
    pub cfg: KvConfig,
    pub pages: DeviceBuffer,
    pub table: DeviceBuffer,
    pub state: DeviceBuffer,
    pub conv: DeviceBuffer,
    pub tails: DeviceBuffer,
    pub draft: Option<DeviceBuffer>,
    pub stream: Arc<Stream>,
    alloc: Mutex<Alloc>,
}

impl KvShared {
    fn lock(&self) -> MutexGuard<'_, Alloc> {
        self.alloc.lock().unwrap_or_else(|e| e.into_inner())
    }
}

/// The shared KV pool. Cloning it shares the pool.
#[derive(Clone)]
pub struct KvPool {
    pub(crate) shared: Arc<KvShared>,
}

impl KvPool {
    pub fn new(cfg: KvConfig, stream: Arc<Stream>) -> Result<KvPool> {
        if !cfg.max_pages.is_multiple_of(4) || cfg.max_pages == 0 || cfg.max_slots == 0 {
            return Err(invalid!(
                "max_pages must be a positive multiple of 4, max_slots positive"
            ));
        }
        if cfg.base_pages > cfg.max_pages {
            return Err(invalid!(
                "base_pages {} > max_pages {}",
                cfg.base_pages,
                cfg.max_pages
            ));
        }
        let l = cfg.layout;
        if l.draft_kv_bytes != 0 && l.draft_kv_bytes != Dims::GLM53F.ring_bytes() {
            return Err(invalid!(
                "a draft region of {} bytes per slot: the DFlash2 ring is {}",
                l.draft_kv_bytes,
                Dims::GLM53F.ring_bytes()
            ));
        }
        let shared = KvShared {
            pages: DeviceBuffer::zeroed(cfg.pages * l.page_bytes)?,
            table: DeviceBuffer::zeroed(cfg.max_slots * cfg.max_pages * 4)?,
            state: DeviceBuffer::zeroed(cfg.max_slots * l.kda_state_bytes())?,
            conv: DeviceBuffer::zeroed(cfg.max_slots * l.conv_bytes())?,
            tails: DeviceBuffer::zeroed(cfg.max_slots * l.tails_bytes().max(16))?,
            draft: if l.draft_kv_bytes > 0 {
                Some(DeviceBuffer::zeroed(cfg.max_slots * l.draft_kv_bytes)?)
            } else {
                None
            },
            stream,
            alloc: Mutex::new(Alloc {
                pages: PageAlloc::new(cfg.pages),
                free_slots: (0..cfg.max_slots).rev().collect(),
            }),
            cfg,
        };
        Ok(KvPool {
            shared: Arc::new(shared),
        })
    }

    pub fn config(&self) -> &KvConfig {
        &self.shared.cfg
    }

    pub fn stream(&self) -> &Arc<Stream> {
        &self.shared.stream
    }

    pub fn free_pages(&self) -> usize {
        self.shared.lock().pages.free()
    }

    pub fn free_slots(&self) -> usize {
        self.shared.lock().free_slots.len()
    }

    /// A new empty slot with its base pages, or `Err` when every slot is taken.
    pub fn slot(&self) -> Result<GlmKv> {
        let s = &self.shared;
        let slot = s.lock().free_slots.pop().ok_or_else(|| {
            Error::OutOfMemory(format!("all {} KV slots are taken", s.cfg.max_slots))
        })?;
        let n = s.cfg.layout.draft_kv_bytes;
        let draft = s.draft.as_ref().map(|d| {
            // SAFETY: region `slot` of the pool's draft arena holds one ring (checked in `new`),
            // at a 16-byte multiple; the slot index is this GlmKv's alone while it lives, the
            // arena lives as long as the pool it holds, and only the forward's drafter writes it,
            // on the pool's stream.
            unsafe { GpuSlot::external(d.byte_ptr(slot * n), &Dims::GLM53F) }
        });
        let mut kv = GlmKv {
            pool: self.shared.clone(),
            slot,
            pages: SlotPages::default(),
            tokens: 0,
            pending: 0,
            draft,
        };
        kv.clear_state()?;
        let base = s.cfg.base_pages;
        kv.grow_pages(base)?;
        Ok(kv)
    }

    /// Device base pointer of the pages (layer `j`'s blocks at `+ 35,904 j`).
    pub fn pages_ptr(&self) -> *mut u8 {
        self.shared.pages.ptr(0)
    }
}

/// The positional state saved at one position: KDA states, conv windows and DSA tails.
pub struct KvMark {
    pub tokens: usize,
    pub(crate) buf: DeviceBuffer,
}

impl KvMark {
    pub fn bytes(&self) -> usize {
        self.buf.bytes()
    }
}

/// One request's context state on the device.
pub struct GlmKv {
    pub(crate) pool: Arc<KvShared>,
    pub(crate) slot: usize,
    pub(crate) pages: SlotPages,
    pub(crate) tokens: usize,
    pub(crate) pending: usize,
    /// The drafter's context over this slot's ring, when the pool has rings.
    pub(crate) draft: Option<GpuSlot>,
}

impl GlmKv {
    fn layout(&self) -> &KvLayout {
        &self.pool.cfg.layout
    }

    fn stream(&self) -> &Stream {
        &self.pool.stream
    }

    /// The slot's index in the pool's per-slot arrays.
    pub fn slot_index(&self) -> usize {
        self.slot
    }

    /// Committed tokens: the position of the next row.
    pub fn tokens(&self) -> usize {
        self.tokens
    }

    /// Rows of an uncommitted verify window (0 outside one).
    pub fn pending(&self) -> usize {
        self.pending
    }

    /// Tokens the slot can hold without growing.
    pub fn capacity(&self) -> usize {
        self.pages.capacity()
    }

    /// The slot's physical pages, in logical order.
    pub fn page_table(&self) -> &[u32] {
        &self.pages.pages
    }

    /// Device bytes that holding `tokens` tokens would add (pages only: the positional state
    /// and the drafter's ring are fixed per slot, allocated with the pool, and the forward's
    /// working set is its own).
    pub fn need_bytes(&self, tokens: usize) -> usize {
        KvLayout::pages_for(tokens).saturating_sub(self.pages.len()) * self.layout().page_bytes
    }

    /// Grow to hold at least `tokens` tokens; keeps every row.
    pub fn reserve(&mut self, tokens: usize) -> Result<()> {
        self.grow_pages(KvLayout::pages_for(tokens))
    }

    /// Device bytes held: pages (shared ones included) and the slot's fixed state (the drafter's
    /// ring included).
    pub fn bytes(&self) -> usize {
        self.pages.len() * self.layout().page_bytes + self.layout().slot_fixed_bytes()
    }

    /// Drop every token (a fresh request): positional state zeroed, pages kept, the drafter's
    /// context emptied.
    pub fn reset(&mut self) -> Result<()> {
        self.tokens = 0;
        self.pending = 0;
        if let Some(d) = self.draft.as_mut() {
            d.reset();
        }
        self.clear_state()
    }

    /// Drop every token and give back pages past the base capacity.
    pub fn release(&mut self) -> Result<()> {
        self.reset()?;
        let base = self.pool.cfg.base_pages;
        let mut a = self.pool.lock();
        self.pages.truncate(&mut a.pages, base);
        Ok(())
    }

    fn grow_pages(&mut self, pages: usize) -> Result<()> {
        if pages > self.pool.cfg.max_pages {
            return Err(Error::OutOfMemory(format!(
                "{pages} pages exceed a slot's page table of {}",
                self.pool.cfg.max_pages
            )));
        }
        let ch = {
            let mut a = self.pool.lock();
            self.pages.reserve(&mut a.pages, pages)?
        };
        self.apply(&ch)
    }

    /// Run a bookkeeping step's page copies and upload the changed page-table entries.
    pub(crate) fn apply(&self, ch: &PageChange) -> Result<()> {
        let pb = self.layout().page_bytes;
        let s = self.stream();
        for c in &ch.copies {
            self.pool.pages.copy_from(
                s,
                c.dst as usize * pb,
                &self.pool.pages,
                c.src as usize * pb,
                pb,
            )?;
        }
        if let Some(first) = ch.first_changed {
            let entries: Vec<i32> = self.pages.pages[first..]
                .iter()
                .map(|&p| p as i32)
                .collect();
            self.pool.table.upload_async(
                s,
                self.slot * self.pool.cfg.max_pages + first,
                &entries,
            )?;
        }
        Ok(())
    }

    /// Before a pass writes rows `[tokens, tokens + rows)`: their pages exist and belong to
    /// this slot alone.
    pub(crate) fn prepare_rows(&mut self, rows: usize) -> Result<()> {
        let end = self.tokens + rows;
        if KvLayout::pages_for(end) > self.pool.cfg.max_pages {
            return Err(Error::OutOfMemory(format!(
                "{end} tokens exceed a slot's page table of {} pages",
                self.pool.cfg.max_pages
            )));
        }
        let ch = {
            let mut a = self.pool.lock();
            self.pages.prepare_write(&mut a.pages, self.tokens, end)?
        };
        self.apply(&ch)
    }

    // ---- Positional state regions -------------------------------------------------------

    fn state_at(&self) -> usize {
        self.slot * self.layout().kda_state_bytes()
    }

    fn conv_at(&self) -> usize {
        self.slot * self.layout().conv_bytes()
    }

    fn tails_at(&self) -> usize {
        self.slot * self.layout().tails_bytes().max(16)
    }

    fn clear_state(&self) -> Result<()> {
        let (l, s) = (self.layout(), self.stream());
        self.pool
            .state
            .zero_async(s, self.state_at(), l.kda_state_bytes())?;
        self.pool
            .conv
            .zero_async(s, self.conv_at(), l.conv_bytes())?;
        self.pool
            .tails
            .zero_async(s, self.tails_at(), l.tails_bytes())
    }

    /// Copy the positional state into `buf` (`dir` true) or out of it into the slot.
    fn copy_state(&self, buf: &DeviceBuffer, save: bool) -> Result<()> {
        let (l, s, p) = (*self.layout(), self.stream(), &self.pool);
        let regions = [
            (&p.state, self.state_at(), 0, l.kda_state_bytes()),
            (&p.conv, self.conv_at(), l.kda_state_bytes(), l.conv_bytes()),
            (
                &p.tails,
                self.tails_at(),
                l.kda_state_bytes() + l.conv_bytes(),
                l.tails_bytes(),
            ),
        ];
        for (arena, at, off, n) in regions {
            if save {
                buf.copy_from(s, off, arena, at, n)?;
            } else {
                arena.copy_from(s, at, buf, off, n)?;
            }
        }
        Ok(())
    }

    // ---- Marks, rewind, fork -------------------------------------------------------------

    /// Save the positional state at the current position.
    pub fn mark(&self) -> Result<KvMark> {
        if self.pending != 0 {
            return Err(invalid!("mark inside an uncommitted verify window"));
        }
        let buf = DeviceBuffer::alloc(self.layout().mark_bytes())?;
        self.copy_state(&buf, true)?;
        Ok(KvMark {
            tokens: self.tokens,
            buf,
        })
    }

    /// Go back to `to` tokens, where `mark` was taken in this slot's history.
    pub fn rewind(&mut self, to: usize, mark: &KvMark) -> Result<()> {
        if mark.tokens != to || to > self.tokens {
            return Err(invalid!(
                "rewind to {to} with a mark at {} (the slot holds {})",
                mark.tokens,
                self.tokens
            ));
        }
        let ch = {
            let mut a = self.pool.lock();
            self.pages.rewind(&mut a.pages, to)?
        };
        self.apply(&ch)?;
        self.copy_state(&mark.buf, false)?;
        self.tokens = to;
        self.pending = 0;
        if let Some(d) = self.draft.as_mut() {
            // A context that does not reach `to` (it never lags the committed rows while the
            // forward has a drafter) restarts cold there.
            if d.len() >= to {
                d.rewind(to).map_err(Error::Other)?;
            } else {
                d.restart(to);
            }
        }
        Ok(())
    }

    /// Become `src`'s first `to` tokens at `mark` (taken by `src` at `to`): full pages shared
    /// copy-on-write, the partial last page copied, the mark's state loaded.
    pub fn fork(&mut self, src: &GlmKv, to: usize, mark: &KvMark) -> Result<()> {
        if !Arc::ptr_eq(&self.pool, &src.pool) {
            return Err(invalid!("fork across pools"));
        }
        if mark.tokens != to || to > src.tokens {
            return Err(invalid!(
                "fork of {to} tokens with a mark at {} from a slot of {}",
                mark.tokens,
                src.tokens
            ));
        }
        self.reset()?;
        let ch = {
            let mut a = self.pool.lock();
            self.pages.fork(&mut a.pages, &src.pages, to)?
        };
        self.apply(&ch)?;
        self.copy_state(&mark.buf, false)?;
        self.tokens = to;
        if let (Some(d), Some(s)) = (self.draft.as_mut(), src.draft.as_ref()) {
            // The ring is positional, not paged: copy it whole, then keep what it holds of
            // `src`'s first `to` rows.
            if s.len() >= to {
                let n = self.pool.cfg.layout.draft_kv_bytes;
                let arena = self.pool.draft.as_ref().expect("a pool with rings");
                arena.copy_from(&self.pool.stream, self.slot * n, arena, src.slot * n, n)?;
                d.follow(s, to).map_err(Error::Other)?;
            } else {
                d.restart(to);
            }
        }
        Ok(())
    }

    // ---- Host images ---------------------------------------------------------------------

    pub fn page_tokens(&self) -> usize {
        PAGE
    }

    pub fn page_bytes(&self) -> usize {
        self.layout().page_bytes
    }

    pub fn state_bytes(&self) -> usize {
        self.layout().mark_bytes()
    }

    /// Copy committed rows `[first, first + n)` (`first` page-aligned, `n <= 64`) into a host
    /// page image. Latent records past `n` and pooled keys of pools past `n / 4` are zeroed, so
    /// the image depends only on the committed rows.
    pub fn export_page(&self, first: usize, n: usize, dst: &mut [u8]) -> Result<()> {
        let pb = self.page_bytes();
        if !first.is_multiple_of(PAGE)
            || n == 0
            || n > PAGE
            || first + n > self.tokens
            || dst.len() != pb
        {
            return Err(invalid!(
                "export_page({first}, {n}) of a slot with {} tokens into {} bytes",
                self.tokens,
                dst.len()
            ));
        }
        let p = self.pages.pages[first / PAGE] as usize;
        self.pool.pages.download_bytes(self.stream(), p * pb, dst)?;
        if n < PAGE {
            let pools = n / 4;
            for j in 0..self.layout().dsa_layers {
                let b = &mut dst[j * LAYER_PAGE_BYTES..(j + 1) * LAYER_PAGE_BYTES];
                b[n * 528..64 * 528].fill(0);
                b[33_792 + pools * 128..35_840].fill(0);
                b[35_840 + pools * 4..35_904].fill(0);
            }
        }
        Ok(())
    }

    /// Copy `mark`'s positional state into a host buffer of [`GlmKv::state_bytes`]. The tail
    /// slots past each tail's count (keys of a pool that has since completed) are zeroed, so
    /// the image depends only on the committed tokens.
    pub fn export_state(&self, mark: &KvMark, dst: &mut [u8]) -> Result<()> {
        if dst.len() != mark.buf.bytes() {
            return Err(invalid!(
                "state image of {} bytes, the mark holds {}",
                dst.len(),
                mark.buf.bytes()
            ));
        }
        mark.buf.download_bytes(self.stream(), 0, dst)?;
        let l = self.layout();
        let base = l.kda_state_bytes() + l.conv_bytes();
        for j in 0..l.dsa_layers {
            let t = &mut dst[base + j * TAIL..base + (j + 1) * TAIL];
            let n = (u32::from_le_bytes([t[0], t[1], t[2], t[3]]) as usize).min(3);
            t[16 + n * 512..].fill(0);
        }
        Ok(())
    }

    /// Append `n` rows from a host page image at the slot's next position (a restore runs
    /// pages in order into a reset slot).
    pub fn import_page(&mut self, n: usize, src: &[u8]) -> Result<()> {
        let pb = self.page_bytes();
        if !self.tokens.is_multiple_of(PAGE)
            || n == 0
            || n > PAGE
            || src.len() != pb
            || self.pending != 0
        {
            return Err(invalid!(
                "import_page({n}) at {} tokens from {} bytes",
                self.tokens,
                src.len()
            ));
        }
        self.prepare_rows(n)?;
        let p = self.pages.pages[self.tokens / PAGE] as usize;
        self.pool
            .pages
            .upload_bytes_async(self.stream(), p * pb, src)?;
        self.tokens += n;
        Ok(())
    }

    /// Finish a restore of `tokens` tokens: load the positional state image. The drafter's
    /// context restarts cold at `tokens`: the image holds no taps to rebuild it from, so drafts
    /// read only the rows committed after the restore.
    pub fn import_state(&mut self, tokens: usize, src: &[u8]) -> Result<()> {
        let l = *self.layout();
        if tokens != self.tokens || src.len() != l.mark_bytes() {
            return Err(invalid!(
                "import_state({tokens}) after {} imported rows, {} bytes",
                self.tokens,
                src.len()
            ));
        }
        let s = self.stream();
        let p = &self.pool;
        p.state
            .upload_bytes_async(s, self.state_at(), &src[..l.kda_state_bytes()])?;
        p.conv.upload_bytes_async(
            s,
            self.conv_at(),
            &src[l.kda_state_bytes()..l.kda_state_bytes() + l.conv_bytes()],
        )?;
        p.tails.upload_bytes_async(
            s,
            self.tails_at(),
            &src[l.kda_state_bytes() + l.conv_bytes()..],
        )?;
        if let Some(d) = self.draft.as_mut() {
            d.restart(tokens);
        }
        Ok(())
    }

    /// Wait for the slot's queued copies.
    pub fn sync(&self) -> Result<()> {
        self.stream().synchronize()
    }

    /// The slot's DFlash2 ring on the device (pointer and bytes), when the pool was laid out
    /// with a drafter: `[5 layers][K, V][2,056 rows][1,024]` BF16.
    pub fn draft_kv(&self) -> Option<(*mut u8, usize)> {
        let n = self.layout().draft_kv_bytes;
        self.pool
            .draft
            .as_ref()
            .map(|d| (d.byte_ptr(self.slot * n), n))
    }

    /// The drafter's context (committed length, lowest readable position), when the pool has
    /// rings.
    pub fn draft_slot(&self) -> Option<&GpuSlot> {
        self.draft.as_ref()
    }

    // ---- Views for the forward and for tests ---------------------------------------------

    /// Element offset of this slot's KDA state (all its layers) in the pool's state arena.
    pub(crate) fn state_elem_offset(&self) -> usize {
        self.state_at() / 4
    }

    /// Element offset of this slot's conv windows in the pool's conv arena.
    pub(crate) fn conv_elem_offset(&self) -> usize {
        self.conv_at() / 2
    }

    /// Row index of this slot's tails in the pool's tail arena, in units of one layer's tail.
    pub(crate) fn tail_row(&self, dsa_layer: usize) -> usize {
        self.tails_at() / TAIL + dsa_layer
    }

    /// Download KDA layer `j`'s state, f32 `[64][128 (v)][128 (k)]`.
    pub fn download_state(&self, j: usize) -> Result<Vec<f32>> {
        self.stream().synchronize()?;
        let n = KvLayout::state_elems_per_layer();
        self.pool
            .state
            .download_at(self.state_elem_offset() + j * n, n)
    }

    /// Download KDA layer `j`'s conv window, BF16 `[3][24,576]`.
    pub fn download_conv(&self, j: usize) -> Result<Vec<u16>> {
        self.stream().synchronize()?;
        let n = KvLayout::conv_elems_per_layer();
        self.pool
            .conv
            .download_at(self.conv_elem_offset() + j * n, n)
    }

    /// Download DSA layer `j`'s tail record.
    pub fn download_tail(&self, j: usize) -> Result<Vec<u8>> {
        self.stream().synchronize()?;
        self.pool.tails.download_at(self.tail_row(j) * TAIL, TAIL)
    }

    /// Download logical page `i`'s block of DSA layer `j` (35,904 bytes).
    pub fn download_page_block(&self, i: usize, j: usize) -> Result<Vec<u8>> {
        self.stream().synchronize()?;
        let p = self.pages.pages[i] as usize;
        self.pool.pages.download_at(
            p * self.layout().page_bytes + j * LAYER_PAGE_BYTES,
            LAYER_PAGE_BYTES,
        )
    }
}

impl Drop for GlmKv {
    fn drop(&mut self) {
        let mut a = self.pool.lock();
        let alloc = &mut a.pages;
        for &p in &self.pages.pages {
            alloc.release(p);
        }
        self.pages.pages.clear();
        a.free_slots.push(self.slot);
    }
}
