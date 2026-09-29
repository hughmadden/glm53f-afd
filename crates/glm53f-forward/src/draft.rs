//! The DFlash2 drafter in the forward (feature `cuda`): the taps, the context appends and the
//! drafts, over `glm53f-dflash`'s GPU drafter. Its `README.md` cites every step below at pinned
//! commits (steps 1 and 4, and "Integration").
//!
//! - **Taps.** For every row of every pass through the head (prefill chunks, decode rows,
//!   verify windows): at the entry of layers 6, 15, 25, 34 and 43, the mean of the four mHC
//!   streams there, which is the completed output of target layers 5, 14, 24, 33 and 42
//!   (`dflash_config.target_layer_ids`), rounded to BF16 into a `[rows][5 x 4,096]` buffer,
//!   layer 5 first. SGLang captures "before layer k + 1" (`glm5_next.py:1403-1415`) and
//!   contracts the four streams with `hc_contract` (`glm5_next.py:1078-1084`),
//!   `x.unflatten(-1, (4, -1)).mean(dim=-2)` (`mhc.py:1571-1573`), concatenated layer 5 first
//!   (`aux_hidden_states.py:55-58`), at the commits the drafter's `README.md` pins (step 1);
//!   `glm53f_fwd_stream_mean` sums the streams left to right in f32 and scales by 1/4, the
//!   arithmetic of the model's own final mean (`glm53f-layers` `hc_head`).
//! - **Context.** Only committed rows reach the drafter: after a prefill pass (each chunk of a
//!   segment) or a decode pass, all its rows; after a verify pass, nothing until its commit,
//!   which appends the first `keep` rows of each window (the anchor and the accepted drafts).
//!   Appends run on the forward's stream right after the pass that wrote the taps, so a row's
//!   taps never wait in the buffer past the next pass. A two-lane prefill pass (`crate::forward`)
//!   captures each lane's rows into their rows of the buffer and appends lane by lane, lane A's
//!   rows first (a request the lanes split: lane A's part, then lane B's), the calls two one-lane
//!   passes of the same rows make.
//! - **Memory.** [`Dflash::reserve`] allocates the tap buffer for the forward's largest pass and
//!   grows the drafter's working buffers to their largest use before a server sizes its KV pool;
//!   appends and drafts then allocate nothing.
//! - **Drafts.** A batch of requests, each at its committed length with its last verified token
//!   as the anchor (the anchor's embedding row comes from the host table); the drafter reads the
//!   forward's LM head in place.
//! - **The FP8 drafter** ([`Dflash::load_with`] with `fp8`, `glm53f-serve --drafter-fp8`, off by
//!   default). The drafter's GEMM weights in FP8 E4M3 with 128 x 128 block scales (quantized at
//!   load) and its own FP8 copy of the LM head for drafting, quantized from the forward's once;
//!   the forward's head is untouched (`glm53f-dflash`'s `gpu` module, "The FP8 drafter"). It
//!   changes which drafts are proposed, never a committed token: the verify pass decides.
//!
//! The rings live in the KV pool (`KvLayout::new(shape, Some(drafter config))`), one per slot,
//! and follow the slot's rewinds, forks and restores (`crate::kv`). The forward must run decoder
//! layers `0 ..= 43` ([`LAYERS_NEEDED`]).

use std::path::Path;
use std::sync::Arc;

use glm53f_dflash::gpu::{GpuDrafter, GpuSlot, Taps};
use glm53f_dflash::seam::{DraftRequest, Proposal};
use glm53f_dflash::weights::Weights;
use glm53f_dflash::{Dims, TARGET_LAYERS};
use glm53f_model::config::DraftConfig;

use crate::device::{launched, DeviceBuffer, Stream};
use crate::embed::HostEmbedding;
use crate::error::{invalid, Error, Result};
use crate::ffi;
use crate::kv::GlmKv;
use crate::shape::{HIDDEN, VOCAB};
use crate::weights::DeviceModel;

/// Values in one row of taps: five target layers of 4,096 (20,480).
pub const TAP_WIDTH: usize = TARGET_LAYERS.len() * HIDDEN;

/// Decoder layers a forward with a drafter runs at least: the last tap is read at the entry of
/// layer 43.
pub const LAYERS_NEEDED: usize = TARGET_LAYERS[TARGET_LAYERS.len() - 1] + 2;

/// Which tap is read at the entry of decoder layer `layer`, if any.
pub fn tap_at(layer: usize) -> Option<usize> {
    TARGET_LAYERS.iter().position(|&k| k + 1 == layer)
}

/// One request to draft for.
pub struct DraftReq<'a> {
    /// At its committed length, with no verify pending.
    pub kv: &'a GlmKv,
    /// The last verified token, at position `kv.tokens()`, not yet in the slot.
    pub anchor: u32,
    /// `<= 0`: the greedy walk. Otherwise the sampled walk at this temperature.
    pub temperature: f32,
    /// One uniform in `[0, 1)` per draft (read only when sampling).
    pub uniforms: &'a [f32],
}

/// The drafter as the forward runs it: `glm53f-dflash`'s GPU drafter on the forward's stream,
/// and the tap buffer.
pub struct Dflash {
    // Dropped before `stream`, which it borrows.
    gpu: GpuDrafter,
    /// `[rows][TAP_WIDTH]` BF16, allocated when the drafter is attached to a forward.
    taps: Option<DeviceBuffer>,
    head: *const u16,
    stream: Arc<Stream>,
    config: DraftConfig,
}

// SAFETY: `head` is a device address; the drafter is driven by one thread at a time (its owner,
// the forward) and its device memory is usable from any thread.
unsafe impl Send for Dflash {}

impl Dflash {
    /// [`Dflash::new`] from the drafter's checkpoint directory (`config.json`,
    /// `model.safetensors`).
    pub fn load(
        dir: &Path,
        model: &DeviceModel,
        embed: &HostEmbedding,
        stream: &Arc<Stream>,
    ) -> Result<Dflash> {
        Self::load_with(dir, model, embed, stream, false)
    }

    /// [`Dflash::load`], the FP8 drafter when `fp8` ([`Dflash::new_with`]).
    pub fn load_with(
        dir: &Path,
        model: &DeviceModel,
        embed: &HostEmbedding,
        stream: &Arc<Stream>,
        fp8: bool,
    ) -> Result<Dflash> {
        let config = DraftConfig::load(&dir.join("config.json"))?;
        let w = Weights::load(dir, Dims::GLM53F).map_err(Error::Other)?;
        Self::new_with(&w, config, model, embed, stream, fp8)
    }

    /// Upload the drafter's weights (2.18 GiB) next to `model`, reading `model`'s LM head in
    /// place, on `stream` (the KV pool's, which the forward shares). `embed` gives the mask
    /// token's row; `config` is the checkpoint's (the KV pool's rings are laid out from it). It
    /// drafts once attached to the forward that owns `model`
    /// ([`crate::forward::GlmForward::attach_drafter`]).
    pub fn new(
        w: &Weights,
        config: DraftConfig,
        model: &DeviceModel,
        embed: &HostEmbedding,
        stream: &Arc<Stream>,
    ) -> Result<Dflash> {
        Self::new_with(w, config, model, embed, stream, false)
    }

    /// [`Dflash::new`]; with `fp8`, the FP8 drafter (module documentation): its weights quantized
    /// to FP8 block-128 at load (1.17 GiB instead of 2.18) and its own FP8 copy of `model`'s LM
    /// head (0.59 GiB), quantized from it once on `stream`; `model`'s head is not changed.
    pub fn new_with(
        w: &Weights,
        config: DraftConfig,
        model: &DeviceModel,
        embed: &HostEmbedding,
        stream: &Arc<Stream>,
        fp8: bool,
    ) -> Result<Dflash> {
        if w.dims != Dims::GLM53F {
            return Err(invalid!(
                "a drafter of shape {:?}: the forward takes GLM-5.3-Flash's DFlash2",
                w.dims
            ));
        }
        w.dims.check_config(&config).map_err(Error::Other)?;
        if config
            .target_layer_ids
            .iter()
            .map(|&l| l as usize)
            .ne(TARGET_LAYERS)
        {
            return Err(invalid!(
                "a drafter reading layers {:?}: the forward taps {TARGET_LAYERS:?}",
                config.target_layer_ids
            ));
        }
        let lm = &model.head.lm_head;
        if lm.n != VOCAB || lm.k != HIDDEN {
            return Err(invalid!("an LM head of {} x {}", lm.n, lm.k));
        }
        let head: *const u16 = lm.buf.ptr(0);
        let mask = embed.row(w.dims.mask_token as usize);
        // SAFETY: `head` is `model`'s LM head, [154,880][4,096] BF16 on this device, never
        // written; the drafter reads it only once attached to the forward that owns `model`
        // (`attach_drafter` checks the address), so it outlives every read; the FP8 drafter reads
        // it once, here, on `stream` (after the upload that wrote it). `stream` is kept in `self`,
        // and `gpu` is dropped first.
        let gpu = unsafe {
            if fp8 {
                GpuDrafter::fp8_with_head_on(w, head, mask, stream.raw())
            } else {
                GpuDrafter::with_borrowed_head_on(w, head, mask, stream.raw())
            }
        }
        .map_err(Error::Other)?;
        Ok(Dflash {
            gpu,
            taps: None,
            head,
            stream: stream.clone(),
            config,
        })
    }

    pub fn dims(&self) -> Dims {
        self.gpu.dims()
    }

    /// The checkpoint's config (`KvLayout::new(shape, Some(config))` lays out the rings).
    pub fn config(&self) -> &DraftConfig {
        &self.config
    }

    /// Device bytes of the drafter's weights (the LM head is the forward's; the FP8 drafter's
    /// own FP8 copy of it included).
    pub fn weight_bytes(&self) -> usize {
        self.gpu.weight_bytes()
    }

    /// Whether this is the FP8 drafter ([`Dflash::new_with`]).
    pub fn is_fp8(&self) -> bool {
        self.gpu.is_fp8()
    }

    /// Device bytes of the tap buffer (0 until attached or reserved).
    pub fn tap_bytes(&self) -> usize {
        self.taps.as_ref().map_or(0, |t| t.bytes())
    }

    /// Device bytes of the drafter's working buffers (grown by appends and drafts, or all at
    /// once by [`Dflash::reserve`]).
    pub fn scratch_bytes(&self) -> usize {
        self.gpu.scratch_bytes()
    }

    /// Allocate everything the drafter uses while serving, before a server sizes its KV pool
    /// from the memory left: the tap buffer for passes of up to `rows` rows (the forward's
    /// largest pass: `ForwardBuffers::pass_rows`), and the drafter's working buffers for drafts
    /// of up to `requests` requests and appends of any size (`GpuDrafter::reserve`). Later
    /// appends and drafts of no more then allocate nothing. Returns the bytes of the two.
    pub fn reserve(&mut self, rows: usize, requests: usize) -> Result<(usize, usize)> {
        self.reserve_taps(rows)?;
        let scratch = self.gpu.reserve(requests).map_err(Error::Other)?;
        Ok((self.tap_bytes(), scratch))
    }

    pub(crate) fn head(&self) -> *const u16 {
        self.head
    }

    pub(crate) fn stream(&self) -> &Arc<Stream> {
        &self.stream
    }

    /// The tap buffer for passes of up to `rows` rows (kept when it is large enough already).
    pub(crate) fn reserve_taps(&mut self, rows: usize) -> Result<()> {
        if self
            .taps
            .as_ref()
            .is_none_or(|t| t.bytes() < rows * TAP_WIDTH * 2)
        {
            self.taps = None;
            self.taps = Some(DeviceBuffer::alloc(rows * TAP_WIDTH * 2)?);
        }
        Ok(())
    }

    fn taps_buf(&self, rows: usize) -> Result<&DeviceBuffer> {
        let t = self
            .taps
            .as_ref()
            .ok_or_else(|| invalid!("the drafter is not attached to a forward"))?;
        if rows * TAP_WIDTH * 2 > t.bytes() {
            return Err(invalid!("taps for {rows} rows: the buffer holds fewer"));
        }
        Ok(t)
    }

    /// At the entry of decoder layer `layer`, before its attention: when a tap is read there,
    /// the mean of `streams` (the layer's input streams, BF16 `[rows][4][4096]`, the previous
    /// layer's completed output) into its columns of the tap buffer's rows `row0 ..` (a lane
    /// of a two-lane pass writes the pass's rows it holds).
    pub(crate) fn capture(
        &self,
        layer: usize,
        row0: usize,
        rows: usize,
        streams: *const u16,
        stream: &Stream,
    ) -> Result<()> {
        let Some(t) = tap_at(layer) else {
            return Ok(());
        };
        let taps = self.taps_buf(row0 + rows)?;
        // SAFETY: `streams` holds `rows` rows of the pass; the tap buffer holds rows `row0 ..
        // row0 + rows` (checked), its column block `t` at a 16-byte multiple.
        launched(
            unsafe {
                ffi::glm53f_fwd_stream_mean(
                    streams,
                    rows as i32,
                    HIDDEN as i32,
                    taps.ptr(row0 * TAP_WIDTH + t * HIDDEN),
                    TAP_WIDTH as i64,
                    stream.raw(),
                )
            },
            "glm53f_fwd_stream_mean",
        )
    }

    /// Append committed rows: for `kvs[i]`, the `rows[i].1` rows of the tap buffer from row
    /// `rows[i].0`, which are the slot's last committed rows (`kvs[i].tokens()` has moved past
    /// them). A context that is not where they start (the slot moved without its drafter)
    /// restarts cold there first.
    pub(crate) fn append(&mut self, kvs: &mut [&mut GlmKv], rows: &[(usize, usize)]) -> Result<()> {
        let total = rows.iter().map(|&(r0, n)| r0 + n).max().unwrap_or(0);
        let taps: *const u16 = self.taps_buf(total)?.ptr(0);
        let mut items: Vec<(&mut GpuSlot, Taps<'_>)> = Vec::with_capacity(kvs.len());
        for (kv, &(row0, n)) in kvs.iter_mut().zip(rows) {
            let start = kv.tokens - n;
            let Some(d) = kv.draft.as_mut() else {
                continue;
            };
            if d.len() != start {
                d.restart(start);
            }
            if n > 0 {
                items.push((
                    d,
                    Taps::Device {
                        ptr: taps.wrapping_add(row0 * TAP_WIDTH),
                        rows: n,
                    },
                ));
            }
        }
        if items.is_empty() {
            return Ok(());
        }
        // SAFETY: each item names `n` rows of the tap buffer (inside it, checked above), written
        // by the pass just queued on the forward's stream, which is the drafter's stream too: the
        // append is ordered after them.
        unsafe { self.gpu.append_taps(&mut items) }.map_err(Error::Other)
    }

    /// Proposals for `reqs`, `block - 1` per request.
    pub(crate) fn draft(
        &mut self,
        reqs: &[DraftReq<'_>],
        embed: &HostEmbedding,
    ) -> Result<Vec<Proposal>> {
        let mut dr = Vec::with_capacity(reqs.len());
        for r in reqs {
            let slot =
                r.kv.draft
                    .as_ref()
                    .ok_or_else(|| invalid!("a slot without a drafter context"))?;
            if slot.len() != r.kv.tokens || r.kv.pending != 0 {
                return Err(invalid!(
                    "a drafter context at {} for a slot of {} tokens (+{} pending)",
                    slot.len(),
                    r.kv.tokens,
                    r.kv.pending
                ));
            }
            if r.anchor as usize >= VOCAB {
                return Err(invalid!("anchor {} beyond the vocabulary", r.anchor));
            }
            dr.push(DraftRequest {
                slot,
                anchor: r.anchor,
                anchor_embed: embed.row(r.anchor as usize),
                temperature: r.temperature,
                uniforms: r.uniforms,
            });
        }
        self.gpu.launch(&dr).map_err(Error::Other)?;
        self.gpu.proposals(dr.len()).map_err(Error::Other)
    }

    /// The first `rows` rows of the tap buffer (tests; waits for the stream).
    pub fn taps(&self, rows: usize) -> Result<Vec<u16>> {
        let t = self.taps_buf(rows)?;
        self.stream.synchronize()?;
        t.download(rows * TAP_WIDTH)
    }

    /// The stored key and value rows of draft layer `layer` at position `pos` in `kv`'s ring
    /// (tests).
    pub fn ring_row(&self, kv: &GlmKv, layer: usize, pos: usize) -> Result<(Vec<u16>, Vec<u16>)> {
        let d = kv
            .draft
            .as_ref()
            .ok_or_else(|| invalid!("a slot without a drafter context"))?;
        self.gpu.ring_row(d, layer, pos).map_err(Error::Other)
    }
}
