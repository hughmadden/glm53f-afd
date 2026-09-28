//! The drafter on the GPU (feature `cuda`): BF16 weights on the device, cuBLAS GEMMs (BF16
//! inputs, f32 accumulation and outputs) and this crate's kernels for everything else.
//!
//! Numerics are the CPU reference's in [`crate::reference::Reference::bf16_io`] mode: every GEMM
//! input and the ring's keys and values are BF16; the residual stream, norms, RoPE, the
//! convolutions, attention, logits and the selector are f32.
//!
//! Per request a [`GpuSlot`] holds the context ring (`[layers][K, V][window + block][kv_width]`
//! BF16, 40.16 MiB at the checkpoint's shape) and the committed length.
//!
//! [`GpuDrafter::append`] (committed rows): `fc` over the taps, `hidden_norm`, then per layer the
//! key/value rows of the fused QKV weight, `k_norm` and RoPE, stored at `pos % ring`.
//!
//! [`GpuDrafter::launch`] then [`GpuDrafter::proposals`] ([`Drafter::draft`] does both; a batch
//! of requests, 8 rows each): the block embeddings; per layer the
//! input norm, the attention convolution's kernel projection and `prepare`, the fused QKV GEMM,
//! q/k norms and RoPE, the block's keys and values into the ring (their positions' rows are dead
//! context), split-K attention over the ring, `o_proj`, `finish` added to the residual; the same
//! for the MLP (fused gate/up GEMM, SiLU x up, down); the final norm; the LM head GEMM over each
//! block's 7 draft rows; the top-16; the selector's projection and walk.

use crate::blas::Blas;
use crate::cuda::check;
use crate::device::{DeviceBuffer, Scratch, Stream};
use crate::ffi;
use crate::seam::{Append, DraftRequest, Drafter, Proposal};
use crate::weights::Weights;
use crate::{cpu, Dims};

const TOP_K: usize = 16;

/// One layer's device weights.
struct Layer {
    /// `[q_width + 2 kv_width][hidden]`: q, then k, then v rows.
    qkv: DeviceBuffer,
    o: DeviceBuffer,
    /// `[2 inter][hidden]`: gate, then up rows.
    gate_up: DeviceBuffer,
    down: DeviceBuffer,
    attn_kp: DeviceBuffer,
    mlp_kp: DeviceBuffer,
    attn_base: DeviceBuffer,
    mlp_base: DeviceBuffer,
    input_ln: DeviceBuffer,
    post_ln: DeviceBuffer,
    q_norm: DeviceBuffer,
    k_norm: DeviceBuffer,
}

/// The LM head the drafter reads: its own upload, or the target forward's (borrowed).
enum Head {
    Owned(DeviceBuffer),
    Borrowed(*const u16),
}

/// One request's drafter state on the device.
pub struct GpuSlot {
    ring: DeviceBuffer,
    len: usize,
    lo: usize,
}

impl GpuSlot {
    /// Committed rows.
    pub fn len(&self) -> usize {
        self.len
    }
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
    /// The lowest position a draft may read (0 unless a rewind went past what the ring holds).
    pub fn lo(&self) -> usize {
        self.lo
    }
}

/// Where an append's taps are.
#[derive(Clone, Copy, Debug)]
pub enum Taps<'a> {
    /// BF16 bits on the host, `[rows][taps * hidden]`.
    Host(&'a [u16]),
    /// `rows` contiguous rows of BF16 taps on this device (the target forward's capture buffer).
    Device { ptr: *const u16, rows: usize },
}

impl Taps<'_> {
    fn rows(&self, width: usize) -> Result<usize, String> {
        match *self {
            Taps::Host(t) if t.len() % width == 0 => Ok(t.len() / width),
            Taps::Host(t) => Err(format!(
                "append: {} values is not whole rows of {width}",
                t.len()
            )),
            Taps::Device { rows, .. } => Ok(rows),
        }
    }
}

#[derive(Default)]
struct Buffers {
    taps: Scratch,
    feat: Scratch,
    feat_b: Scratch,
    kv: Scratch,
    meta_pos: Scratch,
    meta_req: Scratch,
    bases: Scratch,
    h: Scratch,
    xn: Scratch,
    xb: Scratch,
    wide_b: Scratch,
    dynk: Scratch,
    qkv: Scratch,
    att_b: Scratch,
    a: Scratch,
    gu: Scratch,
    part: Scratch,
    fin: Scratch,
    fin_b: Scratch,
    draft_b: Scratch,
    logits: Scratch,
    vals: Scratch,
    ids: Scratch,
    hp: Scratch,
    anchor_rows: Scratch,
    start: Scratch,
    lo: Scratch,
    anchors: Scratch,
    temps: Scratch,
    unif: Scratch,
    tokens: Scratch,
    index: Scratch,
    scores: Scratch,
    q: Scratch,
    conf: Scratch,
    trace: Scratch,
    rope: Scratch,
    topk_ws: Scratch,
}

/// What the last draft computed besides the proposals, for tests (downloaded on request).
pub struct Outputs {
    /// `norm(h)` `[nreq * block][hidden]`.
    pub hidden: Vec<f32>,
    /// `[nreq * drafts][vocab]`.
    pub logits: Vec<f32>,
    /// `[nreq * drafts][16]`.
    pub vals: Vec<f32>,
    pub ids: Vec<i32>,
    /// `[nreq * drafts][rank]`.
    pub hproj: Vec<f32>,
    /// `[nreq * drafts][16]` edge scores the walk used.
    pub scores: Vec<f32>,
    pub index: Vec<i32>,
}

/// The drafter on one GPU stream.
pub struct GpuDrafter {
    dims: Dims,
    stream: Stream,
    blas: Blas,
    layers: Vec<Layer>,
    fc: DeviceBuffer,
    hidden_norm: DeviceBuffer,
    norm: DeviceBuffer,
    hproj: DeviceBuffer,
    pred: DeviceBuffer,
    succ: DeviceBuffer,
    inv_freq: DeviceBuffer,
    mask_row: DeviceBuffer,
    head: Head,
    /// Token ids a draft may propose (default [`crate::SAMPLE_VOCAB`] for the checkpoint, the
    /// vocabulary for other shapes).
    pub vocab_limit: usize,
    /// Keys per attention split (flash-decoding); 256 by default.
    pub split_keys: usize,
    /// Rows of taps per `fc` pass of an append (bounds the append's scratch).
    pub append_chunk: usize,
    /// Keep a copy of the residual stream after every layer's attention and MLP sites
    /// ([`GpuDrafter::layer_trace`]), for tests.
    pub trace: bool,
    buf: Buffers,
    last_nreq: usize,
}

fn up(data: &[u16], s: &Stream) -> Result<DeviceBuffer, String> {
    DeviceBuffer::from_slice(data, s)
}

fn cat(parts: &[&[u16]]) -> Vec<u16> {
    let mut v = Vec::with_capacity(parts.iter().map(|p| p.len()).sum());
    for p in parts {
        v.extend_from_slice(p);
    }
    v
}

/// The scratch at least `bytes` long, as a raw pointer (the borrow ends here).
fn sp<T>(sc: &mut Scratch, bytes: usize) -> Result<*mut T, String> {
    Ok(sc.get(bytes)?.ptr(0))
}

/// Upload `data` into scratch `sc` and return its pointer.
fn put<T: crate::device::Pod>(sc: &mut Scratch, data: &[T], s: &Stream) -> Result<*mut T, String> {
    let b = sc.get(std::mem::size_of_val(data))?;
    b.upload(data, s)?;
    Ok(b.ptr(0))
}

fn i32c(v: usize, what: &str) -> Result<i32, String> {
    i32::try_from(v).map_err(|_| format!("{what} = {v} does not fit in 32 bits"))
}

impl GpuDrafter {
    /// Upload the drafter's weights and the LM head `head` (`[vocab][hidden]` BF16 bits). `mask_row`
    /// is the target's embedding row of the mask token.
    pub fn new(w: &Weights, head: &[u16], mask_row: &[u16]) -> Result<GpuDrafter, String> {
        let stream = Stream::new()?;
        if head.len() != w.dims.vocab * w.dims.hidden {
            return Err(format!(
                "LM head has {} values, expected {} x {}",
                head.len(),
                w.dims.vocab,
                w.dims.hidden
            ));
        }
        let h = Head::Owned(up(head, &stream)?);
        Self::build(w, h, mask_row, stream)
    }

    /// As [`GpuDrafter::new`], reading the target forward's LM head in place.
    ///
    /// # Safety
    ///
    /// `head` points at `[vocab][hidden]` BF16 values on this device that outlive the drafter and
    /// are not written while it drafts.
    pub unsafe fn with_borrowed_head(
        w: &Weights,
        head: *const u16,
        mask_row: &[u16],
    ) -> Result<GpuDrafter, String> {
        let stream = Stream::new()?;
        Self::build(w, Head::Borrowed(head), mask_row, stream)
    }

    fn build(
        w: &Weights,
        head: Head,
        mask_row: &[u16],
        stream: Stream,
    ) -> Result<GpuDrafter, String> {
        let d = w.dims;
        d.validate()?;
        if d.head_dim != 128
            || d.block != 8
            || d.top_k != TOP_K
            || d.conv_taps > d.block
            || d.group() > 4
            || d.rank > 1024
        {
            return Err(format!(
                "the kernels need head_dim 128, block 8, top_k 16, group <= 4, rank <= 1024: {d:?}"
            ));
        }
        if mask_row.len() != d.hidden {
            return Err("mask row width".into());
        }
        let s = &stream;
        let mut layers = Vec::with_capacity(d.layers);
        for lw in &w.layers {
            layers.push(Layer {
                qkv: up(&cat(&[&lw.q, &lw.k, &lw.v]), s)?,
                o: up(&lw.o, s)?,
                gate_up: up(&cat(&[&lw.gate, &lw.up]), s)?,
                down: up(&lw.down, s)?,
                attn_kp: up(&lw.attn_kp, s)?,
                mlp_kp: up(&lw.mlp_kp, s)?,
                attn_base: up(&lw.attn_base, s)?,
                mlp_base: up(&lw.mlp_base, s)?,
                input_ln: up(&lw.input_ln, s)?,
                post_ln: up(&lw.post_ln, s)?,
                q_norm: up(&lw.q_norm, s)?,
                k_norm: up(&lw.k_norm, s)?,
            });
        }
        let inv = cpu::inv_freq(d.rope_theta, d.head_dim);
        let blas = Blas::new(&stream, 32 << 20)?;
        let vocab_limit = if d == Dims::GLM53F {
            crate::SAMPLE_VOCAB
        } else {
            d.vocab
        };
        Ok(GpuDrafter {
            dims: d,
            fc: up(&w.fc, s)?,
            hidden_norm: up(&w.hidden_norm, s)?,
            norm: up(&w.norm, s)?,
            hproj: up(&w.hproj, s)?,
            pred: up(&w.pred, s)?,
            succ: up(&w.succ, s)?,
            inv_freq: DeviceBuffer::from_slice(&inv, s)?,
            mask_row: up(mask_row, s)?,
            layers,
            head,
            vocab_limit,
            split_keys: 256,
            append_chunk: 512,
            trace: false,
            buf: Buffers::default(),
            last_nreq: 0,
            blas,
            stream,
        })
    }

    pub fn stream(&self) -> &Stream {
        &self.stream
    }

    fn head_ptr(&self) -> *const u16 {
        match &self.head {
            Head::Owned(b) => b.ptr::<u16>(0),
            Head::Borrowed(p) => *p,
        }
    }

    /// A slot with an empty context (its ring allocated, not cleared).
    pub fn new_slot(&self) -> Result<GpuSlot, String> {
        Ok(GpuSlot {
            ring: DeviceBuffer::alloc(self.dims.ring_bytes())?,
            len: 0,
            lo: 0,
        })
    }

    /// The stored key and value row (BF16 bits) of `layer` at `pos`, for tests.
    pub fn ring_row(
        &self,
        slot: &GpuSlot,
        layer: usize,
        pos: usize,
    ) -> Result<(Vec<u16>, Vec<u16>), String> {
        let d = self.dims;
        let kvw = d.kv_width();
        let r = pos % d.ring();
        let k =
            slot.ring
                .download_at::<u16>(((layer * 2) * d.ring() + r) * kvw, kvw, &self.stream)?;
        let v = slot.ring.download_at::<u16>(
            ((layer * 2 + 1) * d.ring() + r) * kvw,
            kvw,
            &self.stream,
        )?;
        Ok((k, v))
    }

    #[allow(clippy::too_many_arguments)]
    unsafe fn gemm(
        &self,
        rows: usize,
        n: usize,
        k: usize,
        x: *const u16,
        w: *const u16,
        y: *mut f32,
        ldy: usize,
    ) -> Result<(), String> {
        // SAFETY: forwarded; the caller's buffers hold the shapes.
        unsafe { self.blas.gemm(rows, n, k, x, k, w, k, y, ldy) }
    }

    /// Append rows to slots: `items[i].1` is BF16 taps `[rows][taps * hidden]` (host) at positions
    /// `slot.len..`. Of an item with more than `window` rows only the last `window` are computed
    /// (no draft can read the others).
    pub fn append(&mut self, items: &mut [(&mut GpuSlot, &[u16])]) -> Result<(), String> {
        let mut t: Vec<(&mut GpuSlot, Taps<'_>)> = items
            .iter_mut()
            .map(|(s, t)| (&mut **s, Taps::Host(t)))
            .collect();
        // SAFETY: host taps only.
        unsafe { self.append_taps(&mut t) }
    }

    /// [`GpuDrafter::append`] from taps anywhere ([`Taps`]).
    ///
    /// # Safety
    ///
    /// A [`Taps::Device`] pointer names `rows * taps * hidden` BF16 values on this device whose
    /// writes have completed: the drafter works on its own stream ([`GpuDrafter::stream`]), so
    /// synchronize the stream that produced them first.
    pub unsafe fn append_taps(
        &mut self,
        items: &mut [(&mut GpuSlot, Taps<'_>)],
    ) -> Result<(), String> {
        let d = self.dims;
        let tw = d.tap_width();
        // (item, first row within the item, rows, first position)
        let mut work: Vec<(usize, usize, usize, usize)> = Vec::new();
        for (i, (slot, taps)) in items.iter().enumerate() {
            let rows = taps.rows(tw)?;
            let skip = rows.saturating_sub(d.window);
            let mut r = skip;
            while r < rows {
                let n = (rows - r).min(self.append_chunk);
                work.push((i, r, n, slot.len + r));
                r += n;
            }
        }
        // Batch the pieces into passes of at most append_chunk rows.
        let mut pass: Vec<(usize, usize, usize, usize)> = Vec::new();
        let mut pass_rows = 0;
        for w in work {
            if pass_rows + w.2 > self.append_chunk && !pass.is_empty() {
                self.append_pass(items, &pass)?;
                pass.clear();
                pass_rows = 0;
            }
            pass_rows += w.2;
            pass.push(w);
        }
        if !pass.is_empty() {
            self.append_pass(items, &pass)?;
        }
        for (slot, taps) in items.iter_mut() {
            slot.len += taps.rows(tw)?;
        }
        self.stream.synchronize()
    }

    fn append_pass(
        &mut self,
        items: &[(&mut GpuSlot, Taps<'_>)],
        pass: &[(usize, usize, usize, usize)],
    ) -> Result<(), String> {
        let d = self.dims;
        let (tw, h, kvw) = (d.tap_width(), d.hidden, d.kv_width());
        let n: usize = pass.iter().map(|p| p.2).sum();
        let s = &self.stream;
        let (mut pos, mut req, mut bases) =
            (Vec::with_capacity(n), Vec::with_capacity(n), Vec::new());
        let taps: *mut u16 = {
            let buf = self.buf.taps.get(n * tw * 2)?;
            let mut at = 0;
            for &(i, r0, rows, p0) in pass {
                match items[i].1 {
                    Taps::Host(t) => buf.upload_at(at * tw, &t[r0 * tw..(r0 + rows) * tw], s)?,
                    Taps::Device { ptr, .. } => check(
                        // SAFETY: the caller's pointer holds the item's rows (append_taps); the
                        // destination range is inside the scratch.
                        unsafe {
                            crate::cuda::cudaMemcpyAsync(
                                buf.ptr::<u16>(at * tw).cast(),
                                ptr.add(r0 * tw).cast(),
                                rows * tw * 2,
                                crate::cuda::MEMCPY_D2D,
                                s.raw(),
                            )
                        },
                        "taps D2D",
                    )?,
                }
                let bi = bases.len() as i32;
                bases.push(items[i].0.ring.ptr::<u16>(0) as u64);
                for r in 0..rows {
                    pos.push((p0 + r) as i64);
                    req.push(bi);
                }
                at += rows;
            }
            buf.ptr(0)
        };
        let pos_d = put(&mut self.buf.meta_pos, &pos, s)?;
        let req_d = put(&mut self.buf.meta_req, &req, s)?;
        let bases_d = put(&mut self.buf.bases, &bases, s)?;
        let feat: *mut f32 = sp(&mut self.buf.feat, n * h * 4)?;
        let feat_b: *mut u16 = sp(&mut self.buf.feat_b, n * h * 2)?;
        let kv: *mut f32 = sp(&mut self.buf.kv, n * 2 * kvw * 4)?;
        let cs: *mut f32 = sp(&mut self.buf.rope, n * d.head_dim * 4)?;
        let ni = i32c(n, "rows")?;
        let raw = s.raw();
        // SAFETY (whole block): every pointer is a live buffer sized above for n rows.
        unsafe {
            check(
                ffi::g53d_rope_table(pos_d, ni, self.inv_freq.ptr(0), cs, raw),
                "rope table",
            )?;
            self.gemm(n, h, tw, taps, self.fc.ptr(0), feat, h)?;
            check(
                ffi::g53d_rmsnorm(
                    feat,
                    h as i64,
                    self.hidden_norm.ptr(0),
                    ni,
                    h as i32,
                    d.eps,
                    core::ptr::null_mut(),
                    0,
                    feat_b,
                    h as i64,
                    raw,
                ),
                "hidden_norm",
            )?;
            for (l, lw) in self.layers.iter().enumerate() {
                // Rows q_width.. of the fused weight are k, then v.
                let wkv = lw.qkv.ptr::<u16>(d.q_width() * h);
                self.gemm(n, 2 * kvw, h, feat_b, wkv, kv, 2 * kvw)?;
                check(
                    ffi::g53d_head_norm_rope(
                        kv,
                        (2 * kvw) as i64,
                        ni,
                        d.kv_heads as i32,
                        lw.k_norm.ptr(0),
                        cs,
                        d.eps,
                        core::ptr::null_mut(),
                        0,
                        raw,
                    ),
                    "k_norm + rope (context)",
                )?;
                check(
                    ffi::g53d_store_kv(
                        kv,
                        kv.add(kvw),
                        (2 * kvw) as i64,
                        ni,
                        kvw as i32,
                        req_d,
                        pos_d,
                        bases_d,
                        l as i32,
                        d.ring() as i32,
                        raw,
                    ),
                    "store context K/V",
                )?;
            }
        }
        Ok(())
    }

    /// Queue a draft for a batch on the stream without waiting for it; [`GpuDrafter::proposals`]
    /// collects it ([`Drafter::draft`] does both).
    pub fn launch(&mut self, reqs: &[DraftRequest<'_, GpuSlot>]) -> Result<(), String> {
        let d = self.dims;
        let nreq = reqs.len();
        if nreq == 0 {
            self.last_nreq = 0;
            return Ok(());
        }
        let (h, bl, dr) = (d.hidden, d.block, d.drafts());
        let rows = nreq * bl;
        let drows = nreq * dr;
        let (qw, kvw, qkvw) = (d.q_width(), d.kv_width(), d.q_width() + 2 * d.kv_width());
        let dw = d.dyn_width();
        let s = &self.stream;
        // Per-request and per-row metadata.
        let mut anchor_rows = Vec::with_capacity(nreq * h);
        let (mut start, mut lo, mut anchors, mut temps, mut unif, mut bases) =
            (vec![], vec![], vec![], vec![], vec![], vec![]);
        let (mut pos, mut req) = (Vec::with_capacity(rows), Vec::with_capacity(rows));
        let mut max_keys = 0usize;
        for (i, q) in reqs.iter().enumerate() {
            let (slot, anchor, emb, t, u) =
                (q.slot, q.anchor, q.anchor_embed, q.temperature, q.uniforms);
            if emb.len() != h {
                return Err(format!(
                    "request {i}: anchor embedding has {} values",
                    emb.len()
                ));
            }
            if t > 0.0 && u.len() < dr {
                return Err(format!("request {i}: {} uniforms for {dr} drafts", u.len()));
            }
            anchor_rows.extend_from_slice(emb);
            start.push(slot.len as i64);
            lo.push(slot.lo as i64);
            anchors.push(anchor as i32);
            temps.push(t);
            if t > 0.0 {
                unif.extend_from_slice(&u[..dr]);
            } else {
                unif.extend(std::iter::repeat_n(0.0f32, dr));
            }
            bases.push(slot.ring.ptr::<u16>(0) as u64);
            for j in 0..bl {
                pos.push((slot.len + j) as i64);
                req.push(i as i32);
            }
            let klo = slot.len.saturating_sub(d.window - 1).max(slot.lo);
            max_keys = max_keys.max(slot.len + bl - klo);
        }
        let splits = max_keys.div_ceil(self.split_keys).max(1);
        let b = &mut self.buf;
        let ar = put(&mut b.anchor_rows, &anchor_rows, s)?;
        let st_d = put(&mut b.start, &start, s)?;
        let lo_d = put(&mut b.lo, &lo, s)?;
        let an_d = put(&mut b.anchors, &anchors, s)?;
        let te_d = put(&mut b.temps, &temps, s)?;
        let un_d = put(&mut b.unif, &unif, s)?;
        let ba_d = put(&mut b.bases, &bases, s)?;
        let pos_d = put(&mut b.meta_pos, &pos, s)?;
        let req_d = put(&mut b.meta_req, &req, s)?;
        let hb: *mut f32 = sp(&mut b.h, rows * h * 4)?;
        let xn: *mut f32 = sp(&mut b.xn, rows * h * 4)?;
        let xb: *mut u16 = sp(&mut b.xb, rows * h * 2)?;
        let wide: *mut u16 = sp(&mut b.wide_b, rows * d.inter * 2)?;
        let dynk: *mut f32 = sp(&mut b.dynk, rows * dw * 4)?;
        let qkv: *mut f32 = sp(&mut b.qkv, rows * qkvw * 4)?;
        let att: *mut u16 = sp(&mut b.att_b, rows * qw * 2)?;
        let a: *mut f32 = sp(&mut b.a, rows * h * 4)?;
        let gu: *mut f32 = sp(&mut b.gu, rows * 2 * d.inter * 4)?;
        // SAFETY: a pure size computation.
        let part_floats = unsafe {
            ffi::g53d_attention_partial_floats(nreq as i32, d.heads as i32, splits as i32)
        } as usize;
        let part: *mut f32 = sp(&mut b.part, part_floats * 4)?;
        let fin: *mut f32 = sp(&mut b.fin, rows * h * 4)?;
        let fin_b: *mut u16 = sp(&mut b.fin_b, rows * h * 2)?;
        let draft_b: *mut u16 = sp(&mut b.draft_b, drows * h * 2)?;
        let logits: *mut f32 = sp(&mut b.logits, drows * d.vocab * 4)?;
        let vals: *mut f32 = sp(&mut b.vals, drows * TOP_K * 4)?;
        let ids: *mut i32 = sp(&mut b.ids, drows * TOP_K * 4)?;
        let hp: *mut f32 = sp(&mut b.hp, drows * d.rank * 4)?;
        let tok: *mut i32 = sp(&mut b.tokens, drows * 4)?;
        let idx: *mut i32 = sp(&mut b.index, drows * 4)?;
        let sco: *mut f32 = sp(&mut b.scores, drows * TOP_K * 4)?;
        let qq: *mut f32 = sp(&mut b.q, drows * TOP_K * 4)?;
        let cf: *mut f32 = sp(&mut b.conf, drows * 4)?;
        let cs: *mut f32 = sp(&mut b.rope, rows * d.head_dim * 4)?;
        // SAFETY: a pure size computation.
        let ws_bytes = unsafe { ffi::g53d_topk16_workspace_bytes(drows as i32) } as usize;
        let ws: *mut u8 = sp(&mut b.topk_ws, ws_bytes)?;
        let tr: *mut f32 = if self.trace {
            sp(&mut b.trace, 2 * d.layers * rows * h * 4)?
        } else {
            core::ptr::null_mut()
        };

        let (ri, hi) = (i32c(rows, "rows")?, h as i32);
        let taps_side = d.conv_taps * d.groups();
        let null_f = core::ptr::null_mut::<f32>();
        let null_b = core::ptr::null_mut::<u16>();
        let raw = s.raw();
        let hp_ptr = self.head_ptr();
        // SAFETY (whole block): every pointer is a live device buffer sized above for `rows` block
        // rows and `drows` draft rows; the weights hold the checkpoint's shapes.
        unsafe {
            check(
                ffi::g53d_block_embed(
                    ar,
                    self.mask_row.ptr(0),
                    nreq as i32,
                    bl as i32,
                    hi,
                    hb,
                    raw,
                ),
                "block embed",
            )?;
            check(
                ffi::g53d_rope_table(pos_d, ri, self.inv_freq.ptr(0), cs, raw),
                "rope table",
            )?;
            for (l, lw) in self.layers.iter().enumerate() {
                // Attention site.
                check(
                    ffi::g53d_rmsnorm(
                        hb,
                        h as i64,
                        lw.input_ln.ptr(0),
                        ri,
                        hi,
                        d.eps,
                        xn,
                        h as i64,
                        xb,
                        h as i64,
                        raw,
                    ),
                    "input_layernorm",
                )?;
                self.gemm(rows, dw, h, xb, lw.attn_kp.ptr(0), dynk, dw)?;
                check(
                    ffi::g53d_dyn_conv(
                        xn,
                        h as i64,
                        dynk,
                        dw as i64,
                        lw.attn_base.ptr(0),
                        ri,
                        hi,
                        d.group_size as i32,
                        d.conv_taps as i32,
                        bl as i32,
                        null_f,
                        0,
                        xb,
                        h as i64,
                        null_f,
                        0,
                        raw,
                    ),
                    "attention_conv.prepare",
                )?;
                self.gemm(rows, qkvw, h, xb, lw.qkv.ptr(0), qkv, qkvw)?;
                check(
                    ffi::g53d_head_norm_rope(
                        qkv,
                        qkvw as i64,
                        ri,
                        d.heads as i32,
                        lw.q_norm.ptr(0),
                        cs,
                        d.eps,
                        null_b,
                        0,
                        raw,
                    ),
                    "q_norm + rope",
                )?;
                check(
                    ffi::g53d_head_norm_rope(
                        qkv.add(qw),
                        qkvw as i64,
                        ri,
                        d.kv_heads as i32,
                        lw.k_norm.ptr(0),
                        cs,
                        d.eps,
                        null_b,
                        0,
                        raw,
                    ),
                    "k_norm + rope",
                )?;
                check(
                    ffi::g53d_store_kv(
                        qkv.add(qw),
                        qkv.add(qw + kvw),
                        qkvw as i64,
                        ri,
                        kvw as i32,
                        req_d,
                        pos_d,
                        ba_d,
                        l as i32,
                        d.ring() as i32,
                        raw,
                    ),
                    "store block K/V",
                )?;
                check(
                    ffi::g53d_attention(
                        qkv,
                        qkvw as i64,
                        nreq as i32,
                        d.heads as i32,
                        d.kv_heads as i32,
                        st_d,
                        lo_d,
                        ba_d,
                        l as i32,
                        d.ring() as i32,
                        (d.window - 1) as i32,
                        1.0 / (d.head_dim as f32).sqrt(),
                        self.split_keys as i32,
                        splits as i32,
                        part,
                        att,
                        null_f,
                        qw as i64,
                        raw,
                    ),
                    "attention",
                )?;
                self.gemm(rows, h, qw, att, lw.o.ptr(0), a, h)?;
                check(
                    ffi::g53d_dyn_conv(
                        a,
                        h as i64,
                        dynk.add(taps_side),
                        dw as i64,
                        lw.attn_base.ptr::<u16>(d.conv_taps * h),
                        ri,
                        hi,
                        d.group_size as i32,
                        d.conv_taps as i32,
                        bl as i32,
                        null_f,
                        0,
                        null_b,
                        0,
                        hb,
                        h as i64,
                        raw,
                    ),
                    "attention_conv.finish",
                )?;
                if !tr.is_null() {
                    let at = tr.add((2 * l) * rows * h);
                    check(
                        crate::cuda::cudaMemcpyAsync(
                            at.cast(),
                            hb.cast(),
                            rows * h * 4,
                            crate::cuda::MEMCPY_D2D,
                            raw,
                        ),
                        "trace",
                    )?;
                }
                // MLP site.
                check(
                    ffi::g53d_rmsnorm(
                        hb,
                        h as i64,
                        lw.post_ln.ptr(0),
                        ri,
                        hi,
                        d.eps,
                        xn,
                        h as i64,
                        xb,
                        h as i64,
                        raw,
                    ),
                    "post_attention_layernorm",
                )?;
                self.gemm(rows, dw, h, xb, lw.mlp_kp.ptr(0), dynk, dw)?;
                check(
                    ffi::g53d_dyn_conv(
                        xn,
                        h as i64,
                        dynk,
                        dw as i64,
                        lw.mlp_base.ptr(0),
                        ri,
                        hi,
                        d.group_size as i32,
                        d.conv_taps as i32,
                        bl as i32,
                        null_f,
                        0,
                        xb,
                        h as i64,
                        null_f,
                        0,
                        raw,
                    ),
                    "mlp_conv.prepare",
                )?;
                self.gemm(rows, 2 * d.inter, h, xb, lw.gate_up.ptr(0), gu, 2 * d.inter)?;
                check(
                    ffi::g53d_silu_mul(
                        gu,
                        (2 * d.inter) as i64,
                        ri,
                        d.inter as i32,
                        wide,
                        d.inter as i64,
                        raw,
                    ),
                    "silu * up",
                )?;
                self.gemm(rows, h, d.inter, wide, lw.down.ptr(0), a, h)?;
                check(
                    ffi::g53d_dyn_conv(
                        a,
                        h as i64,
                        dynk.add(taps_side),
                        dw as i64,
                        lw.mlp_base.ptr::<u16>(d.conv_taps * h),
                        ri,
                        hi,
                        d.group_size as i32,
                        d.conv_taps as i32,
                        bl as i32,
                        null_f,
                        0,
                        null_b,
                        0,
                        hb,
                        h as i64,
                        raw,
                    ),
                    "mlp_conv.finish",
                )?;
                if !tr.is_null() {
                    let at = tr.add((2 * l + 1) * rows * h);
                    check(
                        crate::cuda::cudaMemcpyAsync(
                            at.cast(),
                            hb.cast(),
                            rows * h * 4,
                            crate::cuda::MEMCPY_D2D,
                            raw,
                        ),
                        "trace",
                    )?;
                }
            }
            check(
                ffi::g53d_rmsnorm(
                    hb,
                    h as i64,
                    self.norm.ptr(0),
                    ri,
                    hi,
                    d.eps,
                    fin,
                    h as i64,
                    fin_b,
                    h as i64,
                    raw,
                ),
                "norm",
            )?;
            check(
                ffi::g53d_gather_drafts(fin_b, h as i64, nreq as i32, bl as i32, hi, draft_b, raw),
                "gather drafts",
            )?;
            self.gemm(drows, d.vocab, h, draft_b, hp_ptr, logits, d.vocab)?;
            check(
                ffi::g53d_topk16(
                    logits,
                    d.vocab as i64,
                    drows as i32,
                    self.vocab_limit as i32,
                    ws.cast(),
                    vals,
                    ids,
                    raw,
                ),
                "top-16",
            )?;
            self.gemm(drows, d.rank, h, draft_b, self.hproj.ptr(0), hp, d.rank)?;
            check(
                ffi::g53d_select(
                    hp,
                    vals,
                    ids,
                    an_d,
                    self.pred.ptr(0),
                    self.succ.ptr(0),
                    d.rank as i32,
                    nreq as i32,
                    dr as i32,
                    te_d,
                    un_d,
                    tok,
                    idx,
                    sco,
                    qq,
                    cf,
                    raw,
                ),
                "selector",
            )?;
        }
        self.last_nreq = nreq;
        Ok(())
    }

    /// Wait for the last [`GpuDrafter::launch`] and collect its proposals.
    pub fn proposals(&mut self, nreq: usize) -> Result<Vec<Proposal>, String> {
        if nreq != self.last_nreq {
            return Err(format!(
                "proposals for {nreq} requests after a launch of {}",
                self.last_nreq
            ));
        }
        let d = self.dims;
        let dr = d.drafts();
        let s = &self.stream;
        let tok: Vec<i32> = self.buf.tokens.get(0)?.download(nreq * dr, s)?;
        let conf: Vec<f32> = self.buf.conf.get(0)?.download(nreq * dr, s)?;
        let ids: Vec<i32> = self.buf.ids.get(0)?.download(nreq * dr * TOP_K, s)?;
        let q: Vec<f32> = self.buf.q.get(0)?.download(nreq * dr * TOP_K, s)?;
        Ok((0..nreq)
            .map(|i| Proposal {
                tokens: tok[i * dr..(i + 1) * dr]
                    .iter()
                    .map(|&t| t as u32)
                    .collect(),
                conf: conf[i * dr..(i + 1) * dr].to_vec(),
                candidates: ids[i * dr * TOP_K..(i + 1) * dr * TOP_K]
                    .iter()
                    .map(|&t| t as u32)
                    .collect(),
                q: q[i * dr * TOP_K..(i + 1) * dr * TOP_K].to_vec(),
            })
            .collect())
    }

    /// With [`GpuDrafter::trace`] on: the residual stream of the last draft after layer `l`'s
    /// attention site (`mlp == false`) or MLP site, `[nreq * block][hidden]`.
    pub fn layer_trace(&mut self, l: usize, mlp: bool) -> Result<Vec<f32>, String> {
        let d = self.dims;
        let rows = self.last_nreq * d.block;
        let s = &self.stream;
        self.buf.trace.get(0)?.download_at(
            (2 * l + mlp as usize) * rows * d.hidden,
            rows * d.hidden,
            s,
        )
    }

    /// The last draft's intermediates (after [`Drafter::draft`] or [`GpuDrafter::proposals`]).
    pub fn outputs(&mut self) -> Result<Outputs, String> {
        let d = self.dims;
        let n = self.last_nreq;
        let (rows, drows) = (n * d.block, n * d.drafts());
        let s = &self.stream;
        Ok(Outputs {
            hidden: self.buf.fin.get(0)?.download(rows * d.hidden, s)?,
            logits: self.buf.logits.get(0)?.download(drows * d.vocab, s)?,
            vals: self.buf.vals.get(0)?.download(drows * TOP_K, s)?,
            ids: self.buf.ids.get(0)?.download(drows * TOP_K, s)?,
            hproj: self.buf.hp.get(0)?.download(drows * d.rank, s)?,
            scores: self.buf.scores.get(0)?.download(drows * TOP_K, s)?,
            index: self.buf.index.get(0)?.download(drows, s)?,
        })
    }
}

impl Drafter for GpuDrafter {
    type Slot = GpuSlot;

    fn dims(&self) -> Dims {
        self.dims
    }

    fn new_slot(&mut self) -> Result<GpuSlot, String> {
        GpuDrafter::new_slot(self)
    }

    fn len(&self, slot: &GpuSlot) -> usize {
        slot.len
    }

    fn append(&mut self, rows: &mut [Append<'_, GpuSlot>]) -> Result<(), String> {
        let mut items: Vec<(&mut GpuSlot, &[u16])> =
            rows.iter_mut().map(|a| (&mut *a.slot, a.taps)).collect();
        GpuDrafter::append(self, &mut items)
    }

    fn draft(&mut self, reqs: &[DraftRequest<'_, GpuSlot>]) -> Result<Vec<Proposal>, String> {
        self.launch(reqs)?;
        self.proposals(reqs.len())
    }

    fn rewind(&mut self, slot: &mut GpuSlot, len: usize) -> Result<(), String> {
        if len > slot.len {
            return Err(format!("rewind to {len} past the context's {}", slot.len));
        }
        // Positions >= len_old - window are intact (see reference::Context::rewind).
        slot.lo = slot
            .lo
            .max(slot.len.saturating_sub(self.dims.window))
            .min(len);
        slot.len = len;
        Ok(())
    }

    fn reset(&mut self, slot: &mut GpuSlot) {
        slot.len = 0;
        slot.lo = 0;
    }
}
