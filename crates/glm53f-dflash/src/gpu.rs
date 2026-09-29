//! The drafter on the GPU (feature `cuda`): BF16 weights on the device, cuBLAS GEMMs (BF16
//! inputs, f32 accumulation and outputs) and this crate's kernels for everything else.
//!
//! Numerics are the CPU reference's in [`crate::reference::Reference::bf16_io`] mode: every GEMM
//! input and the ring's keys and values are BF16; the residual stream, norms, RoPE, the
//! convolutions, attention, logits and the selector are f32.
//!
//! Per request a [`GpuSlot`] holds the context ring (`[layers][K, V][window + block][kv_width]`
//! BF16, 40.16 MiB at the checkpoint's shape) and the committed length. The ring is the slot's own
//! allocation ([`GpuDrafter::new_slot`]) or memory the caller owns ([`GpuSlot::external`]: the
//! target's KV pool keeps it with each request's fixed state). A drafter runs on its own stream,
//! or on the target forward's ([`GpuDrafter::with_borrowed_head_on`]), which orders its work with
//! the forward's.
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
//!
//! **The FP8 drafter** ([`GpuDrafter::new_fp8`], [`GpuDrafter::fp8_with_head_on`]). The GEMM
//! weights (`fc`, and per layer the fused QKV, `o_proj`, the fused gate/up, `down` and the two
//! convolutions' kernel projections) are quantized at load to FP8 E4M3 with one f32 scale per
//! 128 x 128 block, the checkpoint's own scheme for its FP8 weights (`glm53f-layers`'
//! `glm53f_fp8_quantize_weight`: per block `scale = amax / 448`, `q = e4m3(w / scale)`), and the
//! drafter gets its own FP8 copy of the LM head, quantized the same way from the target's; the
//! target's head is untouched. The selector's projection and codebooks, the norms and the
//! convolutions' base kernels stay BF16. The GEMMs are `glm53f-layers`' FP8 kernels, their outputs
//! f32 as before: up to 8 rows the decode GEMM (BF16 activations against the FP8 weights, every
//! product exact in f32), its K splits (at least two) summed in split order into f32
//! (`g53d_splitk_sum`); over 8 rows the tensor-core GEMM with the activations in E4M3 per row and
//! 128-group (W8A8, the checkpoint's dynamic scheme) and its f32 output. The drafter's numerics
//! change only which tokens it proposes: the target verifies every draft.

use glm53f_layers::ffi as lffi;

use crate::blas::Blas;
use crate::cuda::check;
use crate::device::{DeviceBuffer, Scratch, Stream};
use crate::ffi;
use crate::seam::{Append, DraftRequest, Drafter, Proposal};
use crate::weights::Weights;
use crate::{cpu, Dims};

const TOP_K: usize = 16;

/// A GEMM weight `[n][k]` on the device: BF16 bits, or FP8 E4M3 codes with f32 scales
/// `[ceil(n / 128)][k / 128]`.
enum Mat {
    Bf16(DeviceBuffer),
    Fp8 {
        w: DeviceBuffer,
        scales: DeviceBuffer,
    },
}

/// A view of a weight's rows from some row on, for one GEMM.
#[derive(Clone, Copy)]
enum W {
    Bf16(*const u16),
    Fp8 { w: *const u8, scales: *const f32 },
}

impl Mat {
    /// Rows `r0..` of this `[n][k]` weight (for FP8, `r0` a multiple of 128).
    fn rows_from(&self, r0: usize, k: usize) -> W {
        match self {
            Mat::Bf16(b) => W::Bf16(b.ptr(r0 * k)),
            Mat::Fp8 { w, scales } => {
                debug_assert!(r0.is_multiple_of(128));
                W::Fp8 {
                    w: w.ptr(r0 * k),
                    scales: scales.ptr((r0 / 128) * (k / 128)),
                }
            }
        }
    }

    fn all(&self) -> W {
        self.rows_from(0, 0)
    }

    fn bytes(&self) -> usize {
        match self {
            Mat::Bf16(b) => b.bytes(),
            Mat::Fp8 { w, scales } => w.bytes() + scales.bytes(),
        }
    }
}

/// `[n][k]` BF16 bits on the device quantized to FP8 E4M3 with 128 x 128 block scales, on `s`
/// (`glm53f-layers`' `glm53f_fp8_quantize_weight`, the checkpoint's scheme).
///
/// # Safety
///
/// `src` names `n * k` BF16 values on this device, 16-byte aligned, whose writes are ordered
/// before `s`'s work.
unsafe fn quantize(src: *const u16, n: usize, k: usize, s: &Stream) -> Result<Mat, String> {
    if !k.is_multiple_of(128) || !n.is_multiple_of(8) {
        return Err(format!(
            "an FP8 weight of {n} x {k}: needs n % 8 == 0 and k % 128 == 0"
        ));
    }
    let w = DeviceBuffer::alloc(n * k)?;
    let scales = DeviceBuffer::alloc(n.div_ceil(128) * (k / 128) * 4)?;
    // SAFETY: the caller's source; `w` holds n x k codes, `scales` the block scales.
    check(
        unsafe {
            lffi::glm53f_fp8_quantize_weight(
                src,
                i32c(n, "n")?,
                i32c(k, "k")?,
                w.ptr(0),
                scales.ptr(0),
                s.raw(),
            )
        },
        "glm53f_fp8_quantize_weight",
    )?;
    Ok(Mat::Fp8 { w, scales })
}

/// A host BF16 weight `[n][k]` on the device, as it is or quantized to FP8 (`fp8`).
fn mat(data: &[u16], n: usize, k: usize, fp8: bool, s: &Stream) -> Result<Mat, String> {
    if data.len() != n * k {
        return Err(format!(
            "a weight of {} values, expected {n} x {k}",
            data.len()
        ));
    }
    let b = up(data, s)?;
    if !fp8 {
        return Ok(Mat::Bf16(b));
    }
    // SAFETY: `b` holds n x k BF16 values, uploaded and synchronized; it is freed after the
    // quantization completes.
    let q = unsafe { quantize(b.ptr(0), n, k, s) }?;
    s.synchronize()?;
    Ok(q)
}

/// K splits of the FP8 decode GEMM for an `[n][k]` weight: its own choice
/// (`glm53f_layers::mlp::decode_ksplit`), at least 2 (the f32 outputs come from its split
/// partials).
pub fn fp8_ksplit(n: usize, k: usize) -> Result<usize, String> {
    let d = glm53f_layers::mlp::decode_ksplit(n, k);
    if d >= 2 {
        return Ok(d);
    }
    let blocks = k / 128;
    (2..=blocks)
        .find(|s| blocks.is_multiple_of(*s))
        .ok_or_else(|| format!("an FP8 GEMM with k = {k} has one 128-block: no f32 splits"))
}

/// `y [rows][n] (f32) = x [rows][k] (BF16) . w^T` for an FP8 weight (codes `w`, block scales
/// `scales`) on `raw`: up to 8 rows `glm53f-layers`' FP8 decode GEMM (BF16 activations, every
/// product exact in f32) into its split partials, summed in split order into `y`
/// (`g53d_splitk_sum`); over 8 rows the activations in E4M3 per row and 128-group
/// (`glm53f_act_quant`) and the tensor-core GEMM (`glm53f_fp8_gemm_prefill`, k32 partial sums
/// promoted to f32) writing `y` (and its BF16 copy into the scratch).
///
/// # Safety
///
/// Live device buffers of these shapes; `sc` grown for this GEMM (`Buffers::fp8`).
#[allow(clippy::too_many_arguments)]
unsafe fn fp8_gemm(
    raw: crate::cuda::RawStream,
    sc: &Fp8Scratch,
    rows: usize,
    n: usize,
    k: usize,
    x: *const u16,
    w: *const u8,
    scales: *const f32,
    y: *mut f32,
) -> Result<(), String> {
    if rows == 0 || n == 0 {
        return Ok(());
    }
    let (ri, ni, ki) = (i32c(rows, "rows")?, i32c(n, "n")?, i32c(k, "k")?);
    // SAFETY (both branches): the caller's buffers and scratch.
    unsafe {
        if rows <= 8 {
            let ksplit = fp8_ksplit(n, k)?;
            check(
                lffi::glm53f_fp8_gemm_decode(
                    x.cast(),
                    core::ptr::null(),
                    0,
                    w,
                    scales,
                    ri,
                    ni,
                    ki,
                    ksplit as i32,
                    sc.partials,
                    core::ptr::null_mut(),
                    raw,
                ),
                "glm53f_fp8_gemm_decode",
            )?;
            check(
                ffi::g53d_splitk_sum(sc.partials, ksplit as i32, ri, ni, y, raw),
                "g53d_splitk_sum",
            )
        } else {
            check(
                lffi::glm53f_act_quant(x, sc.xq, sc.xs, ri, ki, raw),
                "glm53f_act_quant",
            )?;
            check(
                lffi::glm53f_fp8_gemm_prefill(
                    sc.xq,
                    sc.xs,
                    w,
                    scales,
                    ri,
                    ni,
                    ki,
                    lffi::PREFILL_PROMOTE_K32,
                    sc.y16,
                    y,
                    raw,
                ),
                "glm53f_fp8_gemm_prefill",
            )
        }
    }
}

/// The FP8 drafter's GEMM on host data (tests): `w` `[n][k]` BF16 bits quantized on the device
/// as the FP8 drafter's weights are, times `x` `[rows][k]` BF16; returns `y` `[rows][n]` f32.
pub fn fp8_gemm_host(
    x: &[u16],
    rows: usize,
    w: &[u16],
    n: usize,
    k: usize,
) -> Result<Vec<f32>, String> {
    if x.len() != rows * k {
        return Err(format!("{} activations for {rows} x {k}", x.len()));
    }
    let s = Stream::new()?;
    let m = mat(w, n, k, true, &s)?;
    let W::Fp8 { w, scales } = m.all() else {
        unreachable!("an FP8 weight")
    };
    let mut buf = Buffers::default();
    let sc = buf.fp8(true, &[(rows, n, k)])?;
    let xd = up(x, &s)?;
    let y = DeviceBuffer::alloc(rows * n * 4)?;
    // SAFETY: the buffers above hold the shapes; the scratch was grown for this GEMM.
    unsafe { fp8_gemm(s.raw(), &sc, rows, n, k, xd.ptr(0), w, scales, y.ptr(0)) }?;
    y.download(rows * n, &s)
}

/// One layer's device weights.
struct Layer {
    /// `[q_width + 2 kv_width][hidden]`: q, then k, then v rows.
    qkv: Mat,
    o: Mat,
    /// `[2 inter][hidden]`: gate, then up rows.
    gate_up: Mat,
    down: Mat,
    attn_kp: Mat,
    mlp_kp: Mat,
    attn_base: DeviceBuffer,
    mlp_base: DeviceBuffer,
    input_ln: DeviceBuffer,
    post_ln: DeviceBuffer,
    q_norm: DeviceBuffer,
    k_norm: DeviceBuffer,
}

/// The LM head the drafter reads: its own upload, the target forward's (borrowed), or its own
/// FP8 copy (the FP8 drafter).
enum Head {
    Owned(DeviceBuffer),
    Borrowed(*const u16),
    Fp8(Mat),
}

/// Where a slot's ring lives.
enum Ring {
    /// Allocated by [`GpuDrafter::new_slot`].
    Owned(DeviceBuffer),
    /// Device memory the caller owns ([`GpuSlot::external`]).
    External(*mut u8),
}

/// One request's drafter state on the device.
pub struct GpuSlot {
    ring: Ring,
    /// The drafter's window (what [`GpuSlot::rewind`] keeps).
    window: usize,
    len: usize,
    lo: usize,
}

// SAFETY: the ring is device memory, usable from any host thread; the slot (or, for an external
// ring, the caller of `GpuSlot::external`) keeps it alive, and one thread at a time drives it.
unsafe impl Send for GpuSlot {}

impl GpuSlot {
    /// A slot with an empty context whose ring is caller-owned device memory: `dims.ring_bytes()`
    /// bytes at `ring` (the target's KV pool keeps one per request, with the rest of its fixed
    /// state).
    ///
    /// # Safety
    ///
    /// `ring` points at `dims.ring_bytes()` bytes on the drafter's device, 16-byte aligned, that
    /// outlive the slot and that nothing but the drafter this slot is used with writes.
    pub unsafe fn external(ring: *mut u8, dims: &Dims) -> GpuSlot {
        GpuSlot {
            ring: Ring::External(ring),
            window: dims.window,
            len: 0,
            lo: 0,
        }
    }

    fn ring_ptr(&self) -> *mut u16 {
        match &self.ring {
            Ring::Owned(b) => b.ptr(0),
            Ring::External(p) => p.cast(),
        }
    }

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

    /// Context rows a draft at the committed length reads: positions
    /// `max(lo, len - (window - 1)) .. len`.
    pub fn context_rows(&self) -> usize {
        self.len
            - self
                .lo
                .max(self.len.saturating_sub(self.window - 1))
                .min(self.len)
    }

    /// Return to an earlier committed length. Positions at or above `old_len - window` are
    /// intact (see `reference::Context::rewind`); a lower position a later draft would need is
    /// masked out (`lo`), never read stale.
    pub fn rewind(&mut self, len: usize) -> Result<(), String> {
        if len > self.len {
            return Err(format!("rewind to {len} past the context's {}", self.len));
        }
        self.lo = self.lo.max(self.len.saturating_sub(self.window)).min(len);
        self.len = len;
        Ok(())
    }

    /// Empty the context (a new request).
    pub fn reset(&mut self) {
        self.len = 0;
        self.lo = 0;
    }

    /// Continue at committed length `len` with no readable context (`lo = len`): for a request
    /// whose rows' taps are gone (restored from a host image). Drafts then read only the rows
    /// appended afterwards.
    pub fn restart(&mut self, len: usize) {
        self.len = len;
        self.lo = len;
    }

    /// Take `src`'s context at `len <= src.len()`, once the caller has copied `src`'s ring into
    /// this slot's: `src`'s state rewound to `len` (rows `src`'s ring no longer holds are masked
    /// out, as for [`GpuSlot::rewind`]).
    pub fn follow(&mut self, src: &GpuSlot, len: usize) -> Result<(), String> {
        if self.window != src.window {
            return Err("follow: slots of drafters with different windows".into());
        }
        self.len = src.len;
        self.lo = src.lo;
        self.rewind(len)
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
    /// The FP8 GEMMs': the decode GEMM's split partials; the tensor-core GEMM's E4M3 activations,
    /// their scales and its BF16 output (unused: the f32 one is read).
    f8_partials: Scratch,
    f8_xq: Scratch,
    f8_xs: Scratch,
    f8_y16: Scratch,
}

/// The FP8 GEMMs' scratch for one append pass or one draft (null for a BF16 drafter).
#[derive(Clone, Copy)]
struct Fp8Scratch {
    partials: *mut f32,
    xq: *mut u8,
    xs: *mut f32,
    y16: *mut u16,
}

impl Buffers {
    /// The FP8 GEMMs' scratch for the GEMMs `(rows, n, k)` of a pass (grown to the largest); none
    /// for a BF16 drafter (`fp8` false).
    fn fp8(&mut self, fp8: bool, gemms: &[(usize, usize, usize)]) -> Result<Fp8Scratch, String> {
        if !fp8 {
            return Ok(Fp8Scratch {
                partials: core::ptr::null_mut(),
                xq: core::ptr::null_mut(),
                xs: core::ptr::null_mut(),
                y16: core::ptr::null_mut(),
            });
        }
        let (mut p, mut xq, mut xs, mut y) = (0usize, 0usize, 0usize, 0usize);
        for &(rows, n, k) in gemms {
            if rows <= 8 {
                p = p.max(fp8_ksplit(n, k)? * rows * n * 4);
            } else {
                xq = xq.max(rows * k);
                xs = xs.max(rows * (k / 128) * 4);
                y = y.max(rows * n * 2);
            }
        }
        let get = |sc: &mut Scratch, b: usize| -> Result<*mut u8, String> {
            if b == 0 {
                Ok(core::ptr::null_mut())
            } else {
                sp(sc, b)
            }
        };
        Ok(Fp8Scratch {
            partials: get(&mut self.f8_partials, p)?.cast(),
            xq: get(&mut self.f8_xq, xq)?,
            xs: get(&mut self.f8_xs, xs)?.cast(),
            y16: get(&mut self.f8_y16, y)?.cast(),
        })
    }

    fn all(&self) -> [&Scratch; 42] {
        [
            &self.taps,
            &self.feat,
            &self.feat_b,
            &self.kv,
            &self.meta_pos,
            &self.meta_req,
            &self.bases,
            &self.h,
            &self.xn,
            &self.xb,
            &self.wide_b,
            &self.dynk,
            &self.qkv,
            &self.att_b,
            &self.a,
            &self.gu,
            &self.part,
            &self.fin,
            &self.fin_b,
            &self.draft_b,
            &self.logits,
            &self.vals,
            &self.ids,
            &self.hp,
            &self.anchor_rows,
            &self.start,
            &self.lo,
            &self.anchors,
            &self.temps,
            &self.unif,
            &self.tokens,
            &self.index,
            &self.scores,
            &self.q,
            &self.conf,
            &self.trace,
            &self.rope,
            &self.topk_ws,
            &self.f8_partials,
            &self.f8_xq,
            &self.f8_xs,
            &self.f8_y16,
        ]
    }
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
    fc: Mat,
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
    /// FP8 weights and LM head (module documentation).
    fp8: bool,
}

// SAFETY: the raw pointers a drafter holds are device addresses (a borrowed LM head) and stream
// handles, usable from any host thread; `&mut self` makes one thread at a time drive it.
unsafe impl Send for GpuDrafter {}

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
    /// is the target's embedding row of the mask token. The drafter gets a stream of its own.
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
        Self::build(w, h, mask_row, stream, false)
    }

    /// The FP8 drafter (module documentation) with its own stream: the weights quantized at
    /// load, and its own FP8 copy of the LM head `head` (`[vocab][hidden]` BF16 bits, uploaded,
    /// quantized, and freed).
    pub fn new_fp8(w: &Weights, head: &[u16], mask_row: &[u16]) -> Result<GpuDrafter, String> {
        let stream = Stream::new()?;
        let d = w.dims;
        let b = up(head, &stream)?;
        if head.len() != d.vocab * d.hidden {
            return Err(format!(
                "LM head has {} values, expected {} x {}",
                head.len(),
                d.vocab,
                d.hidden
            ));
        }
        // SAFETY: `b` holds the head, uploaded and synchronized.
        let q = unsafe { quantize(b.ptr(0), d.vocab, d.hidden, &stream) }?;
        stream.synchronize()?;
        drop(b);
        Self::build(w, Head::Fp8(q), mask_row, stream, true)
    }

    /// The FP8 drafter on `stream` (the target forward's), its FP8 copy of the LM head quantized
    /// from the target's `head` at load; the target's head is only read, once.
    ///
    /// # Safety
    ///
    /// `head` points at `[vocab][hidden]` BF16 values on this device, 16-byte aligned, written
    /// before `stream`'s work; `stream` is a live stream that outlives the drafter.
    pub unsafe fn fp8_with_head_on(
        w: &Weights,
        head: *const u16,
        mask_row: &[u16],
        stream: crate::cuda::RawStream,
    ) -> Result<GpuDrafter, String> {
        // SAFETY: the caller keeps the stream alive for the drafter's life.
        let stream = unsafe { Stream::borrowed(stream) };
        let d = w.dims;
        // SAFETY: the caller's head.
        let q = unsafe { quantize(head, d.vocab, d.hidden, &stream) }?;
        stream.synchronize()?;
        Self::build(w, Head::Fp8(q), mask_row, stream, true)
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
        Self::build(w, Head::Borrowed(head), mask_row, stream, false)
    }

    /// As [`GpuDrafter::with_borrowed_head`], on `stream`, a stream the caller owns (the target
    /// forward's): the drafter's work is then ordered with the forward's, so taps the forward
    /// wrote and rings it copied need no synchronization before an append or a draft.
    ///
    /// # Safety
    ///
    /// As for [`GpuDrafter::with_borrowed_head`]; and `stream` is a live stream that outlives the
    /// drafter.
    pub unsafe fn with_borrowed_head_on(
        w: &Weights,
        head: *const u16,
        mask_row: &[u16],
        stream: crate::cuda::RawStream,
    ) -> Result<GpuDrafter, String> {
        // SAFETY: the caller keeps the stream alive for the drafter's life.
        let stream = unsafe { Stream::borrowed(stream) };
        Self::build(w, Head::Borrowed(head), mask_row, stream, false)
    }

    fn build(
        w: &Weights,
        head: Head,
        mask_row: &[u16],
        stream: Stream,
        fp8: bool,
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
        let (h, qkvw, dw) = (d.hidden, d.q_width() + 2 * d.kv_width(), d.dyn_width());
        if fp8 && !d.q_width().is_multiple_of(128) {
            return Err(format!("an FP8 drafter needs q_width % 128 == 0: {d:?}"));
        }
        let mut layers = Vec::with_capacity(d.layers);
        for lw in &w.layers {
            layers.push(Layer {
                qkv: mat(&cat(&[&lw.q, &lw.k, &lw.v]), qkvw, h, fp8, s)?,
                o: mat(&lw.o, h, d.q_width(), fp8, s)?,
                gate_up: mat(&cat(&[&lw.gate, &lw.up]), 2 * d.inter, h, fp8, s)?,
                down: mat(&lw.down, h, d.inter, fp8, s)?,
                attn_kp: mat(&lw.attn_kp, dw, h, fp8, s)?,
                mlp_kp: mat(&lw.mlp_kp, dw, h, fp8, s)?,
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
            fc: mat(&w.fc, h, d.tap_width(), fp8, s)?,
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
            fp8,
            blas,
            stream,
        })
    }

    /// Whether the weights and the LM head the drafter reads are FP8 ([`GpuDrafter::new_fp8`]).
    pub fn is_fp8(&self) -> bool {
        self.fp8
    }

    pub fn stream(&self) -> &Stream {
        &self.stream
    }

    pub fn dims(&self) -> Dims {
        self.dims
    }

    /// Device bytes of the weights (and of the LM head when it is the drafter's own); the scratch
    /// grows with the batch on top of it.
    pub fn weight_bytes(&self) -> usize {
        let layers: usize = self
            .layers
            .iter()
            .map(|l| {
                [&l.qkv, &l.o, &l.gate_up, &l.down, &l.attn_kp, &l.mlp_kp]
                    .iter()
                    .map(|m| m.bytes())
                    .sum::<usize>()
                    + [
                        &l.attn_base,
                        &l.mlp_base,
                        &l.input_ln,
                        &l.post_ln,
                        &l.q_norm,
                        &l.k_norm,
                    ]
                    .iter()
                    .map(|b| b.bytes())
                    .sum::<usize>()
            })
            .sum();
        let own = self.fc.bytes()
            + [
                &self.hidden_norm,
                &self.norm,
                &self.hproj,
                &self.pred,
                &self.succ,
                &self.inv_freq,
                &self.mask_row,
            ]
            .iter()
            .map(|b| b.bytes())
            .sum::<usize>();
        let head = match &self.head {
            Head::Owned(b) => b.bytes(),
            Head::Borrowed(_) => 0,
            Head::Fp8(m) => m.bytes(),
        };
        layers + own + head
    }

    fn head_w(&self) -> W {
        match &self.head {
            Head::Owned(b) => W::Bf16(b.ptr::<u16>(0)),
            Head::Borrowed(p) => W::Bf16(*p),
            Head::Fp8(m) => m.all(),
        }
    }

    /// Device bytes of the working buffers appends and drafts have grown so far (weights and
    /// rings aside).
    pub fn scratch_bytes(&self) -> usize {
        self.buf.all().iter().map(|s| s.bytes()).sum()
    }

    /// Grow every working buffer to the size its largest use takes, so that later appends (any
    /// number of rows, passes of at most [`GpuDrafter::append_chunk`]) and drafts of up to `nreq`
    /// requests at any context allocate nothing: one append of a full window of zero taps into a
    /// scratch slot, then one greedy draft of `nreq` requests over it (the most attention splits
    /// a window gives); for the FP8 drafter also a draft of one request and an append of one
    /// block (the FP8 GEMMs' path up to 8 rows, which larger passes do not take). The scratch
    /// slot's ring is freed on return. Returns [`GpuDrafter::scratch_bytes`].
    pub fn reserve(&mut self, nreq: usize) -> Result<usize, String> {
        let d = self.dims;
        let mut slot = self.new_slot()?;
        let zeros = vec![0u16; d.window * d.tap_width()];
        // SAFETY: host taps only.
        unsafe { self.append_taps(&mut [(&mut slot, Taps::Host(&zeros))]) }?;
        let embed = vec![0u16; d.hidden];
        let reqs: Vec<DraftRequest<'_, GpuSlot>> = (0..nreq.max(1))
            .map(|_| DraftRequest {
                slot: &slot,
                anchor: 0,
                anchor_embed: &embed,
                temperature: 0.0,
                uniforms: &[],
            })
            .collect();
        self.launch(&reqs)?;
        self.proposals(reqs.len())?;
        if self.fp8 {
            self.launch(&reqs[..1])?;
            self.proposals(1)?;
            drop(reqs);
            let block = vec![0u16; d.block * d.tap_width()];
            // SAFETY: host taps only.
            unsafe { self.append_taps(&mut [(&mut slot, Taps::Host(&block))]) }?;
        }
        Ok(self.scratch_bytes())
    }

    /// A slot with an empty context (its ring allocated, not cleared).
    pub fn new_slot(&self) -> Result<GpuSlot, String> {
        Ok(GpuSlot {
            ring: Ring::Owned(DeviceBuffer::alloc(self.dims.ring_bytes())?),
            window: self.dims.window,
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
        if layer >= d.layers {
            return Err(format!("ring_row: layer {layer} of {}", d.layers));
        }
        let kvw = d.kv_width();
        let r = pos % d.ring();
        let read = |at: usize| -> Result<Vec<u16>, String> {
            let mut v = vec![0u16; kvw];
            // SAFETY: the ring holds [layers][K, V][ring][kv_width] BF16 values and `at + kvw`
            // lies inside it; v holds kvw values.
            check(
                unsafe {
                    crate::cuda::cudaMemcpyAsync(
                        v.as_mut_ptr().cast(),
                        slot.ring_ptr().add(at).cast(),
                        kvw * 2,
                        crate::cuda::MEMCPY_D2H,
                        self.stream.raw(),
                    )
                },
                "ring row",
            )?;
            self.stream.synchronize()?;
            Ok(v)
        };
        Ok((
            read(((layer * 2) * d.ring() + r) * kvw)?,
            read(((layer * 2 + 1) * d.ring() + r) * kvw)?,
        ))
    }

    /// `y [rows][n] (f32, row stride ldy) = x [rows][k] (BF16) . w^T`: cuBLAS for a BF16 weight;
    /// for an FP8 one (`ldy == n`), up to 8 rows the FP8 decode GEMM and its splits summed in f32,
    /// over 8 rows the W8A8 tensor-core GEMM's f32 output (module documentation).
    #[allow(clippy::too_many_arguments)]
    unsafe fn gemm(
        &self,
        sc: &Fp8Scratch,
        rows: usize,
        n: usize,
        k: usize,
        x: *const u16,
        w: W,
        y: *mut f32,
        ldy: usize,
    ) -> Result<(), String> {
        match w {
            // SAFETY: forwarded; the caller's buffers hold the shapes.
            W::Bf16(w) => unsafe { self.blas.gemm(rows, n, k, x, k, w, k, y, ldy) },
            W::Fp8 { w, scales } => {
                if ldy != n {
                    return Err(format!(
                        "an FP8 GEMM writes dense rows: ldy {ldy} for n {n}"
                    ));
                }
                // SAFETY: forwarded; the scratch was grown for this GEMM (`Buffers::fp8`).
                unsafe { fp8_gemm(self.stream.raw(), sc, rows, n, k, x, w, scales, y) }
            }
        }
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
                bases.push(items[i].0.ring_ptr() as u64);
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
        let sc = self.buf.fp8(self.fp8, &[(n, h, tw), (n, 2 * kvw, h)])?;
        let ni = i32c(n, "rows")?;
        let raw = s.raw();
        // SAFETY (whole block): every pointer is a live buffer sized above for n rows.
        unsafe {
            check(
                ffi::g53d_rope_table(pos_d, ni, self.inv_freq.ptr(0), cs, raw),
                "rope table",
            )?;
            self.gemm(&sc, n, h, tw, taps, self.fc.all(), feat, h)?;
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
                let wkv = lw.qkv.rows_from(d.q_width(), h);
                self.gemm(&sc, n, 2 * kvw, h, feat_b, wkv, kv, 2 * kvw)?;
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
            bases.push(slot.ring_ptr() as u64);
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
        let sc = b.fp8(
            self.fp8,
            &[
                (rows, dw, h),
                (rows, qkvw, h),
                (rows, h, qw),
                (rows, 2 * d.inter, h),
                (rows, h, d.inter),
                (drows, d.vocab, h),
            ],
        )?;

        let (ri, hi) = (i32c(rows, "rows")?, h as i32);
        let taps_side = d.conv_taps * d.groups();
        let null_f = core::ptr::null_mut::<f32>();
        let null_b = core::ptr::null_mut::<u16>();
        let raw = s.raw();
        let head = self.head_w();
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
                self.gemm(&sc, rows, dw, h, xb, lw.attn_kp.all(), dynk, dw)?;
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
                self.gemm(&sc, rows, qkvw, h, xb, lw.qkv.all(), qkv, qkvw)?;
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
                self.gemm(&sc, rows, h, qw, att, lw.o.all(), a, h)?;
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
                self.gemm(&sc, rows, dw, h, xb, lw.mlp_kp.all(), dynk, dw)?;
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
                self.gemm(
                    &sc,
                    rows,
                    2 * d.inter,
                    h,
                    xb,
                    lw.gate_up.all(),
                    gu,
                    2 * d.inter,
                )?;
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
                self.gemm(&sc, rows, h, d.inter, wide, lw.down.all(), a, h)?;
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
            self.gemm(&sc, drows, d.vocab, h, draft_b, head, logits, d.vocab)?;
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
            let hproj = W::Bf16(self.hproj.ptr(0));
            self.gemm(&sc, drows, d.rank, h, draft_b, hproj, hp, d.rank)?;
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
        slot.rewind(len)
    }

    fn reset(&mut self, slot: &mut GpuSlot) {
        slot.reset();
    }
}
