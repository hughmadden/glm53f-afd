//! Checked wrappers over the KDA kernels (feature `cuda`).
//!
//! Each launch checks on the host that every buffer is large enough for the geometry it is
//! given, that written regions do not overlap, and that the batch metadata is consistent, then
//! enqueues the kernel on the stream. Launches are asynchronous; buffers must outlive the
//! stream's work (dropping a [`DeviceBuffer`] calls `cudaFree`, which waits for the device).
//! The batch metadata is copied to the device synchronously when it is set.

use crate::cpu::{self, LayerParams};
use crate::device::{DeviceBuffer, Error, Stream};
use crate::ffi;
use crate::{bf16, channels, state_len, DK, DV, TAPS, WINDOW};

/// One layer's weights on the device.
pub struct Weights {
    pub heads: usize,
    /// bf16 `[C][TAPS]`.
    pub conv_w: DeviceBuffer,
    /// f32 `[H]`.
    pub a_log: DeviceBuffer,
    /// f32 `[H * DK]`.
    pub dt_bias: DeviceBuffer,
    /// bf16 `[DV]`.
    pub norm_w: DeviceBuffer,
    pub eps: f32,
    pub lower: f32,
}

impl Weights {
    /// Upload a layer's parameters (bfloat16 tensors are rounded, which is exact for values
    /// that came from bfloat16).
    pub fn upload(p: &LayerParams) -> Result<Self, Error> {
        if p.conv_w.len() != channels(p.heads) * TAPS
            || p.a_log.len() != p.heads
            || p.dt_bias.len() != p.heads * DK
            || p.norm_w.len() != DV
        {
            return Err(Error::Invalid(
                "layer parameters have the wrong sizes".into(),
            ));
        }
        Ok(Weights {
            heads: p.heads,
            conv_w: DeviceBuffer::from_slice(&bf16::encode(&p.conv_w))?,
            a_log: DeviceBuffer::from_slice(&p.a_log)?,
            dt_bias: DeviceBuffer::from_slice(&p.dt_bias)?,
            norm_w: DeviceBuffer::from_slice(&bf16::encode(&p.norm_w))?,
            eps: p.eps,
            lower: p.lower,
        })
    }
}

/// Rows in a device buffer: row `r` starts at element `offset + r * stride`.
#[derive(Clone, Copy)]
pub struct RowView<'a> {
    pub buf: &'a DeviceBuffer,
    pub offset: usize,
    pub stride: usize,
}

impl<'a> RowView<'a> {
    pub fn new(buf: &'a DeviceBuffer, offset: usize, stride: usize) -> Self {
        RowView {
            buf,
            offset,
            stride,
        }
    }

    /// Dense rows of `width` elements from the start of `buf`.
    pub fn dense(buf: &'a DeviceBuffer, width: usize) -> Self {
        RowView {
            buf,
            offset: 0,
            stride: width,
        }
    }

    /// Check that rows `first .. first + rows` have columns `[lo, hi)` inside the buffer
    /// (elements of `size` bytes).
    fn check(
        &self,
        what: &str,
        size: usize,
        first: usize,
        rows: usize,
        lo: i64,
        hi: i64,
    ) -> Result<(), Error> {
        if rows == 0 {
            return Ok(());
        }
        let len = (self.buf.bytes() / size) as i128;
        let start = self.offset as i128 + first as i128 * self.stride as i128;
        let min = start + lo as i128;
        let max = start + (rows as i128 - 1) * self.stride as i128 + hi as i128;
        if min < 0 || max > len {
            return Err(Error::Invalid(format!(
                "{what}: rows reach elements [{min}, {max}) of a buffer of {len}"
            )));
        }
        Ok(())
    }
}

fn check_region(
    what: &str,
    buf: &DeviceBuffer,
    size: usize,
    offset: usize,
    n: usize,
) -> Result<(), Error> {
    let len = buf.bytes() / size;
    if offset.checked_add(n).is_none_or(|end| end > len) {
        return Err(Error::Invalid(format!(
            "{what}: elements [{offset}, {offset} + {n}) exceed a buffer of {len}"
        )));
    }
    Ok(())
}

fn to_i32(x: usize, what: &str) -> Result<i32, Error> {
    i32::try_from(x).map_err(|_| Error::Invalid(format!("{what} = {x} does not fit in 32 bits")))
}

fn launched(code: i32, what: &str) -> Result<(), Error> {
    if code == 0 {
        Ok(())
    } else if code == 1 {
        Err(Error::Invalid(format!("{what}: rejected by the launcher")))
    } else {
        Err(Error::Cuda(code, what.to_string()))
    }
}

/// Per-row replay inputs on the device, for up to `rows` rows of `heads` heads (one layer).
pub struct Saves {
    pub heads: usize,
    pub rows: usize,
    /// f32 `[rows][H][DK]`.
    pub k: DeviceBuffer,
    /// bf16 `[rows][H][DV]`.
    pub v: DeviceBuffer,
    /// f32 `[rows][H][DK]`.
    pub g: DeviceBuffer,
    /// f32 `[rows][H]`.
    pub b: DeviceBuffer,
}

impl Saves {
    pub fn alloc(heads: usize, rows: usize) -> Result<Self, Error> {
        Ok(Saves {
            heads,
            rows,
            k: DeviceBuffer::zeroed(rows * heads * DK * 4)?,
            v: DeviceBuffer::zeroed(rows * heads * DV * 2)?,
            g: DeviceBuffer::zeroed(rows * heads * DK * 4)?,
            b: DeviceBuffer::zeroed(rows * heads * 4)?,
        })
    }

    /// The first `rows` rows, on the host.
    pub fn download(&self, rows: usize) -> Result<cpu::Saves, Error> {
        let h = self.heads;
        Ok(cpu::Saves {
            heads: h,
            rows,
            k: self.k.download(rows * h * DK)?,
            v: bf16::decode(&self.v.download::<u16>(rows * h * DV)?),
            g: self.g.download(rows * h * DK)?,
            beta: self.b.download(rows * h)?,
        })
    }

    fn ptrs(&self) -> (*mut f32, *mut u16, *mut f32, *mut f32) {
        (self.k.ptr(0), self.v.ptr(0), self.g.ptr(0), self.b.ptr(0))
    }
}

/// Where a chain writes the state after its last row.
#[derive(Clone, Copy)]
pub enum StateOut<'a> {
    /// Not written (a verify round followed by a replay).
    Skip,
    /// Over the input state.
    InPlace,
    /// To a buffer at an element offset (per request, the input's offsets are reused).
    To(&'a DeviceBuffer, usize),
}

/// How a batch launch's states are stored.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StateType {
    /// f32, as the reference keeps it.
    F32,
    /// bf16 (the `_bf16state` entry points, decision D8): the arithmetic stays f32, and the
    /// state is rounded to bf16 after every row (the chain and the replay) or every 16-row
    /// chunk (the prefill).
    Bf16,
}

impl StateType {
    /// Bytes of one state element.
    pub fn bytes(self) -> usize {
        match self {
            StateType::F32 => 4,
            StateType::Bf16 => 2,
        }
    }
}

fn same(a: &DeviceBuffer, b: &DeviceBuffer) -> bool {
    a.as_ptr() == b.as_ptr() && a.bytes() > 0
}

/// Regions `[a, a + n)` and `[b, b + n)` either coincide or are disjoint.
fn coincide_or_disjoint(a: usize, b: usize, n: usize) -> bool {
    a == b || a + n <= b || b + n <= a
}

/// One layer, one request: `rows` rows from the committed state (`glm53f_kda_chain`).
pub struct Chain<'a> {
    pub weights: &'a Weights,
    pub rows: usize,
    /// Row `r`: q | k | v in columns `[0, C)`, the beta logits in `[b_off, b_off + H)`.
    pub p: RowView<'a>,
    pub b_off: i64,
    /// Forget-gate projections, `[H * DK]` per row.
    pub a: RowView<'a>,
    /// Output-gate projections, `[H * DV]` per row.
    pub g: RowView<'a>,
    /// The conv window `[WINDOW][C]` at this element offset.
    pub conv: &'a DeviceBuffer,
    pub conv_offset: usize,
    /// The state `[H][DV][DK]` at this element offset.
    pub state: &'a DeviceBuffer,
    pub state_offset: usize,
    pub state_out: StateOut<'a>,
    /// Outputs, `[H * DV]` per row.
    pub out: RowView<'a>,
    pub saves: Option<&'a Saves>,
}

impl Chain<'_> {
    pub fn launch(&self, stream: &Stream) -> Result<(), Error> {
        let h = self.weights.heads;
        if self.rows == 0 {
            return Err(Error::Invalid("a chain needs at least one row".into()));
        }
        let c = channels(h) as i64;
        let (lo, hi) = (self.b_off.min(0), c.max(self.b_off + h as i64));
        self.p.check("p", 2, 0, self.rows, lo, hi)?;
        self.a.check("a", 2, 0, self.rows, 0, (h * DK) as i64)?;
        self.g.check("g", 2, 0, self.rows, 0, (h * DV) as i64)?;
        self.out.check("out", 2, 0, self.rows, 0, (h * DV) as i64)?;
        if self.rows > 1 && self.out.stride < h * DV {
            return Err(Error::Invalid("out rows overlap".into()));
        }
        check_region("conv", self.conv, 2, self.conv_offset, WINDOW * channels(h))?;
        check_region("state", self.state, 4, self.state_offset, state_len(h))?;
        let state_out = match self.state_out {
            StateOut::Skip => core::ptr::null_mut(),
            StateOut::InPlace => self.state.ptr::<f32>(self.state_offset),
            StateOut::To(buf, off) => {
                check_region("state_out", buf, 4, off, state_len(h))?;
                if same(buf, self.state)
                    && !coincide_or_disjoint(off, self.state_offset, state_len(h))
                {
                    return Err(Error::Invalid("state_out partly overlaps the state".into()));
                }
                buf.ptr::<f32>(off)
            }
        };
        let (ks, vs, gs, bs) = match self.saves {
            Some(s) => {
                if s.heads != h || s.rows < self.rows {
                    return Err(Error::Invalid(format!(
                        "saves hold {} rows of {} heads",
                        s.rows, s.heads
                    )));
                }
                s.ptrs()
            }
            None => (
                core::ptr::null_mut(),
                core::ptr::null_mut(),
                core::ptr::null_mut(),
                core::ptr::null_mut(),
            ),
        };
        let w = self.weights;
        // SAFETY: every region the kernel touches was checked above; the buffers are borrowed
        // for the call and outlive the stream's work (see the module notes).
        let code = unsafe {
            ffi::glm53f_kda_chain(
                to_i32(h, "heads")?,
                to_i32(self.rows, "rows")?,
                self.p.buf.ptr(self.p.offset),
                self.p.stride as i64,
                self.b_off,
                self.a.buf.ptr(self.a.offset),
                self.a.stride as i64,
                self.g.buf.ptr(self.g.offset),
                self.g.stride as i64,
                self.conv.ptr(self.conv_offset),
                w.conv_w.ptr(0),
                self.state.ptr(self.state_offset),
                state_out,
                w.a_log.ptr(0),
                w.dt_bias.ptr(0),
                w.norm_w.ptr(0),
                w.eps,
                w.lower,
                self.out.buf.ptr(self.out.offset),
                self.out.stride as i64,
                ks,
                vs,
                gs,
                bs,
                stream.raw(),
            )
        };
        launched(code, "glm53f_kda_chain")
    }
}

/// One layer, one request: the state after the first `rows` saved rows (`glm53f_kda_replay`).
/// `state_out` may be the same buffer and offset as `state` (in place).
pub fn replay(
    heads: usize,
    rows: usize,
    state: (&DeviceBuffer, usize),
    state_out: (&DeviceBuffer, usize),
    saves: &Saves,
    stream: &Stream,
) -> Result<(), Error> {
    check_region("state", state.0, 4, state.1, state_len(heads))?;
    check_region("state_out", state_out.0, 4, state_out.1, state_len(heads))?;
    if same(state.0, state_out.0) && !coincide_or_disjoint(state.1, state_out.1, state_len(heads)) {
        return Err(Error::Invalid("state_out partly overlaps the state".into()));
    }
    if saves.heads != heads || saves.rows < rows {
        return Err(Error::Invalid(format!(
            "saves hold {} rows of {} heads",
            saves.rows, saves.heads
        )));
    }
    let (ks, vs, gs, bs) = saves.ptrs();
    // SAFETY: regions checked above.
    let code = unsafe {
        ffi::glm53f_kda_replay(
            to_i32(heads, "heads")?,
            to_i32(rows, "rows")?,
            state.0.ptr(state.1),
            state_out.0.ptr(state_out.1),
            ks,
            vs,
            gs,
            bs,
            stream.raw(),
        )
    };
    launched(code, "glm53f_kda_replay")
}

/// Advance one conv window (`[WINDOW][C]` at `conv_offset`) past `keep` kept rows of `p`
/// (q | k | v in columns `[0, C)`), in place (`glm53f_kda_conv_shift`).
pub fn conv_shift(
    heads: usize,
    keep: usize,
    conv: &DeviceBuffer,
    conv_offset: usize,
    p: RowView<'_>,
    stream: &Stream,
) -> Result<(), Error> {
    let c = channels(heads);
    check_region("conv", conv, 2, conv_offset, WINDOW * c)?;
    // The shift reads new rows keep.saturating_sub(WINDOW) .. keep.
    let first = keep.saturating_sub(WINDOW);
    p.check("p", 2, first, keep - first, 0, c as i64)?;
    // SAFETY: regions checked above.
    let code = unsafe {
        ffi::glm53f_kda_conv_shift(
            to_i32(c, "channels")?,
            to_i32(keep, "keep")?,
            conv.ptr(conv_offset),
            p.buf.ptr(p.offset),
            p.stride as i64,
            stream.raw(),
        )
    };
    launched(code, "glm53f_kda_conv_shift")
}

/// One request of a batch.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Request {
    /// Rows of this request in the window (it owns the next `rows` rows of the shared buffers).
    pub rows: usize,
    /// Element offset of its conv window (per layer).
    pub conv_offset: usize,
    /// Element offset of its state (per layer).
    pub state_offset: usize,
}

/// Batch metadata on the device: row ranges, offsets and kept rows, with a host copy for the
/// checks.
pub struct BatchMeta {
    capacity: usize,
    requests: Vec<Request>,
    cu_rows: DeviceBuffer,
    keep_d: DeviceBuffer,
    conv_off: DeviceBuffer,
    state_off: DeviceBuffer,
}

impl BatchMeta {
    pub fn new(capacity: usize) -> Result<Self, Error> {
        Ok(BatchMeta {
            capacity,
            requests: Vec::new(),
            cu_rows: DeviceBuffer::zeroed((capacity + 1) * 4)?,
            keep_d: DeviceBuffer::zeroed(capacity * 4)?,
            conv_off: DeviceBuffer::zeroed(capacity * 8)?,
            state_off: DeviceBuffer::zeroed(capacity * 8)?,
        })
    }

    /// Set the requests and, for replays and conv shifts, the rows each keeps (none: all).
    /// Copies to the device synchronously (after the device's earlier work).
    pub fn set(&mut self, requests: &[Request], keep: Option<&[usize]>) -> Result<(), Error> {
        if requests.is_empty() || requests.len() > self.capacity {
            return Err(Error::Invalid(format!(
                "{} requests for a capacity of {}",
                requests.len(),
                self.capacity
            )));
        }
        let keep: Vec<usize> = match keep {
            Some(k) if k.len() == requests.len() => k.to_vec(),
            Some(_) => return Err(Error::Invalid("one keep per request".into())),
            None => requests.iter().map(|r| r.rows).collect(),
        };
        for (r, &k) in requests.iter().zip(&keep) {
            if k > r.rows {
                return Err(Error::Invalid(format!(
                    "keep {k} exceeds the request's {} rows",
                    r.rows
                )));
            }
        }
        let mut cu = vec![0i32; requests.len() + 1];
        for (i, r) in requests.iter().enumerate() {
            cu[i + 1] = cu[i]
                .checked_add(to_i32(r.rows, "rows")?)
                .ok_or_else(|| Error::Invalid("total rows do not fit in 32 bits".into()))?;
        }
        let keep32: Vec<i32> = keep
            .iter()
            .map(|&k| to_i32(k, "keep"))
            .collect::<Result<_, _>>()?;
        let conv: Vec<i64> = requests.iter().map(|r| r.conv_offset as i64).collect();
        let state: Vec<i64> = requests.iter().map(|r| r.state_offset as i64).collect();
        self.cu_rows.upload(&cu)?;
        self.keep_d.upload(&keep32)?;
        self.conv_off.upload(&conv)?;
        self.state_off.upload(&state)?;
        self.requests = requests.to_vec();
        Ok(())
    }

    pub fn requests(&self) -> &[Request] {
        &self.requests
    }

    /// The device arrays, for launches through [`crate::ffi`] (for example inside a captured
    /// graph): `cu_rows` (int32 `[batch + 1]`), `keep` (int32 `[batch]`), the conv and state
    /// offsets (int64 `[batch]`).
    pub fn device_arrays(&self) -> (*const i32, *const i32, *const i64, *const i64) {
        (
            self.cu_rows.ptr(0),
            self.keep_d.ptr(0),
            self.conv_off.ptr(0),
            self.state_off.ptr(0),
        )
    }

    pub fn total_rows(&self) -> usize {
        self.requests.iter().map(|r| r.rows).sum()
    }

    fn batch(&self) -> Result<i32, Error> {
        if self.requests.is_empty() {
            return Err(Error::Invalid("batch metadata not set".into()));
        }
        to_i32(self.requests.len(), "batch")
    }

    /// Each request's region of `n` elements at `offset(r)` inside `len` elements, pairwise
    /// disjoint when `disjoint`.
    #[allow(clippy::too_many_arguments)]
    fn check_regions(
        &self,
        what: &str,
        buf: &DeviceBuffer,
        size: usize,
        base: usize,
        n: usize,
        disjoint: bool,
        offset: impl Fn(&Request) -> usize,
    ) -> Result<(), Error> {
        let mut spans: Vec<usize> = Vec::with_capacity(self.requests.len());
        for r in &self.requests {
            check_region(what, buf, size, base + offset(r), n)?;
            spans.push(offset(r));
        }
        if disjoint {
            spans.sort_unstable();
            if spans.windows(2).any(|w| w[1] < w[0] + n) {
                return Err(Error::Invalid(format!("{what}: requests' regions overlap")));
            }
        }
        Ok(())
    }
}

/// One layer, a batch of requests (`glm53f_kda_chain_batch`). The requests' rows are
/// consecutive in `p`, `a`, `g`, `out` and the saves, in the order of the metadata.
pub struct ChainBatch<'a> {
    pub weights: &'a Weights,
    pub p: RowView<'a>,
    pub b_off: i64,
    pub a: RowView<'a>,
    pub g: RowView<'a>,
    /// Conv windows, at each request's `conv_offset`.
    pub conv: &'a DeviceBuffer,
    /// States, at each request's `state_offset`.
    pub state: &'a DeviceBuffer,
    /// `To(buf, _)` writes at each request's `state_offset` in `buf` (the offset is ignored).
    pub state_out: StateOut<'a>,
    pub out: RowView<'a>,
    pub saves: Option<&'a Saves>,
}

impl ChainBatch<'_> {
    pub fn launch(&self, meta: &BatchMeta, stream: &Stream) -> Result<(), Error> {
        self.launch_as(meta, stream, StateType::F32)
    }

    /// With bf16 states (`glm53f_kda_chain_batch_bf16state`): `state` and `state_out` hold bf16
    /// values, and the requests' `state_offset`s count bf16 elements.
    pub fn launch_bf16_state(&self, meta: &BatchMeta, stream: &Stream) -> Result<(), Error> {
        self.launch_as(meta, stream, StateType::Bf16)
    }

    fn launch_as(&self, meta: &BatchMeta, stream: &Stream, st: StateType) -> Result<(), Error> {
        let es = st.bytes();
        let h = self.weights.heads;
        let batch = meta.batch()?;
        let total = meta.total_rows();
        let c = channels(h) as i64;
        let (lo, hi) = (self.b_off.min(0), c.max(self.b_off + h as i64));
        self.p.check("p", 2, 0, total, lo, hi)?;
        self.a.check("a", 2, 0, total, 0, (h * DK) as i64)?;
        self.g.check("g", 2, 0, total, 0, (h * DV) as i64)?;
        self.out.check("out", 2, 0, total, 0, (h * DV) as i64)?;
        if total > 1 && self.out.stride < h * DV {
            return Err(Error::Invalid("out rows overlap".into()));
        }
        meta.check_regions("conv", self.conv, 2, 0, WINDOW * channels(h), false, |r| {
            r.conv_offset
        })?;
        meta.check_regions("state", self.state, es, 0, state_len(h), true, |r| {
            r.state_offset
        })?;
        let state_out = match self.state_out {
            StateOut::Skip => core::ptr::null_mut(),
            StateOut::InPlace => self.state.ptr::<u8>(0),
            StateOut::To(buf, _) => {
                meta.check_regions("state_out", buf, es, 0, state_len(h), true, |r| {
                    r.state_offset
                })?;
                if same(buf, self.state) {
                    return Err(Error::Invalid(
                        "state_out is the state buffer: use InPlace".into(),
                    ));
                }
                buf.ptr::<u8>(0)
            }
        };
        let (ks, vs, gs, bs) = match self.saves {
            Some(s) => {
                if s.heads != h || s.rows < total {
                    return Err(Error::Invalid(format!(
                        "saves hold {} rows of {} heads",
                        s.rows, s.heads
                    )));
                }
                s.ptrs()
            }
            None => (
                core::ptr::null_mut(),
                core::ptr::null_mut(),
                core::ptr::null_mut(),
                core::ptr::null_mut(),
            ),
        };
        let w = self.weights;
        let heads = to_i32(h, "heads")?;
        // SAFETY: every region was checked above against the metadata's host copy, which
        // matches what `BatchMeta::set` copied to the device.
        let code = unsafe {
            match st {
                StateType::F32 => ffi::glm53f_kda_chain_batch(
                    heads,
                    batch,
                    meta.cu_rows.ptr(0),
                    self.p.buf.ptr(self.p.offset),
                    self.p.stride as i64,
                    self.b_off,
                    self.a.buf.ptr(self.a.offset),
                    self.a.stride as i64,
                    self.g.buf.ptr(self.g.offset),
                    self.g.stride as i64,
                    self.conv.ptr(0),
                    meta.conv_off.ptr(0),
                    w.conv_w.ptr(0),
                    self.state.ptr(0),
                    state_out.cast(),
                    meta.state_off.ptr(0),
                    w.a_log.ptr(0),
                    w.dt_bias.ptr(0),
                    w.norm_w.ptr(0),
                    w.eps,
                    w.lower,
                    self.out.buf.ptr(self.out.offset),
                    self.out.stride as i64,
                    ks,
                    vs,
                    gs,
                    bs,
                    stream.raw(),
                ),
                StateType::Bf16 => ffi::glm53f_kda_chain_batch_bf16state(
                    heads,
                    batch,
                    meta.cu_rows.ptr(0),
                    self.p.buf.ptr(self.p.offset),
                    self.p.stride as i64,
                    self.b_off,
                    self.a.buf.ptr(self.a.offset),
                    self.a.stride as i64,
                    self.g.buf.ptr(self.g.offset),
                    self.g.stride as i64,
                    self.conv.ptr(0),
                    meta.conv_off.ptr(0),
                    w.conv_w.ptr(0),
                    self.state.ptr(0),
                    state_out.cast(),
                    meta.state_off.ptr(0),
                    w.a_log.ptr(0),
                    w.dt_bias.ptr(0),
                    w.norm_w.ptr(0),
                    w.eps,
                    w.lower,
                    self.out.buf.ptr(self.out.offset),
                    self.out.stride as i64,
                    ks,
                    vs,
                    gs,
                    bs,
                    stream.raw(),
                ),
            }
        };
        launched(code, "glm53f_kda_chain_batch")
    }
}

/// Saves for several layers in one allocation each (what a commit replays at once).
pub struct LayerSaves {
    pub layers: usize,
    pub heads: usize,
    pub rows: usize,
    pub k: DeviceBuffer,
    pub v: DeviceBuffer,
    pub g: DeviceBuffer,
    pub b: DeviceBuffer,
}

impl LayerSaves {
    pub fn alloc(layers: usize, heads: usize, rows: usize) -> Result<Self, Error> {
        Ok(LayerSaves {
            layers,
            heads,
            rows,
            k: DeviceBuffer::zeroed(layers * rows * heads * DK * 4)?,
            v: DeviceBuffer::zeroed(layers * rows * heads * DV * 2)?,
            g: DeviceBuffer::zeroed(layers * rows * heads * DK * 4)?,
            b: DeviceBuffer::zeroed(layers * rows * heads * 4)?,
        })
    }

    /// Elements between layers in `k`, `v` and `g`.
    pub fn kv_stride(&self) -> usize {
        self.rows * self.heads * DK
    }

    /// Elements between layers in `b`.
    pub fn b_stride(&self) -> usize {
        self.rows * self.heads
    }

    /// Copy one layer's saves (`rows` rows) into layer `layer`.
    pub fn fill_layer(&self, layer: usize, s: &Saves, rows: usize) -> Result<(), Error> {
        if layer >= self.layers || s.heads != self.heads || rows > self.rows.min(s.rows) {
            return Err(Error::Invalid("fill_layer: shape mismatch".into()));
        }
        let h = self.heads;
        self.k.upload_at(
            layer * self.kv_stride(),
            &s.k.download::<f32>(rows * h * DK)?,
        )?;
        self.v.upload_at(
            layer * self.kv_stride(),
            &s.v.download::<u16>(rows * h * DV)?,
        )?;
        self.g.upload_at(
            layer * self.kv_stride(),
            &s.g.download::<f32>(rows * h * DK)?,
        )?;
        self.b
            .upload_at(layer * self.b_stride(), &s.b.download::<f32>(rows * h)?)?;
        Ok(())
    }
}

/// Replay every layer of one request (`glm53f_kda_replay_layers`): layer `l`'s states at
/// `l * state_stride` in `state` and `state_out` (which may be the same buffer: in place).
pub fn replay_layers(
    rows: usize,
    state: &DeviceBuffer,
    state_out: &DeviceBuffer,
    state_stride: usize,
    saves: &LayerSaves,
    stream: &Stream,
) -> Result<(), Error> {
    let (h, l) = (saves.heads, saves.layers);
    if l == 0 || rows > saves.rows || state_stride < state_len(h) {
        return Err(Error::Invalid(
            "replay_layers: no layers, rows beyond the saves, or overlapping layers".into(),
        ));
    }
    for buf in [state, state_out] {
        check_region("state", buf, 4, (l - 1) * state_stride, state_len(h))?;
    }
    // SAFETY: regions checked above.
    let code = unsafe {
        ffi::glm53f_kda_replay_layers(
            to_i32(h, "heads")?,
            to_i32(l, "layers")?,
            to_i32(rows, "rows")?,
            state.ptr(0),
            state_out.ptr(0),
            state_stride as i64,
            saves.k.ptr(0),
            saves.v.ptr(0),
            saves.g.ptr(0),
            saves.b.ptr(0),
            saves.kv_stride() as i64,
            saves.b_stride() as i64,
            stream.raw(),
        )
    };
    launched(code, "glm53f_kda_replay_layers")
}

/// Replay every layer of every request of a batch (`glm53f_kda_replay_batch`): request `b`
/// keeps `keep[b]` of its rows (the metadata's). Layer `l`'s state of request `r` is at
/// `l * state_stride + r.state_offset`; `state_out` may be `state` (in place).
pub fn replay_batch(
    meta: &BatchMeta,
    state: &DeviceBuffer,
    state_out: &DeviceBuffer,
    state_stride: usize,
    saves: &LayerSaves,
    stream: &Stream,
) -> Result<(), Error> {
    replay_batch_as(
        meta,
        state,
        state_out,
        state_stride,
        saves,
        stream,
        StateType::F32,
    )
}

/// [`replay_batch`] with bf16 states (`glm53f_kda_replay_batch_bf16state`): strides and offsets
/// count bf16 elements.
pub fn replay_batch_bf16_state(
    meta: &BatchMeta,
    state: &DeviceBuffer,
    state_out: &DeviceBuffer,
    state_stride: usize,
    saves: &LayerSaves,
    stream: &Stream,
) -> Result<(), Error> {
    replay_batch_as(
        meta,
        state,
        state_out,
        state_stride,
        saves,
        stream,
        StateType::Bf16,
    )
}

fn replay_batch_as(
    meta: &BatchMeta,
    state: &DeviceBuffer,
    state_out: &DeviceBuffer,
    state_stride: usize,
    saves: &LayerSaves,
    stream: &Stream,
    st: StateType,
) -> Result<(), Error> {
    let (h, l) = (saves.heads, saves.layers);
    let batch = meta.batch()?;
    if l == 0 || meta.total_rows() > saves.rows {
        return Err(Error::Invalid(
            "replay_batch: no layers, or more rows than the saves hold".into(),
        ));
    }
    for buf in [state, state_out] {
        meta.check_regions(
            "state",
            buf,
            st.bytes(),
            (l - 1) * state_stride,
            state_len(h),
            true,
            |r| r.state_offset,
        )?;
    }
    if l > 1 {
        let max_off = meta
            .requests
            .iter()
            .map(|r| r.state_offset)
            .max()
            .unwrap_or(0);
        if state_stride < max_off + state_len(h) {
            return Err(Error::Invalid(
                "replay_batch: layers' regions overlap".into(),
            ));
        }
    }
    let (heads, layers) = (to_i32(h, "heads")?, to_i32(l, "layers")?);
    // SAFETY: regions checked above against the metadata's host copy.
    let code = unsafe {
        match st {
            StateType::F32 => ffi::glm53f_kda_replay_batch(
                heads,
                layers,
                batch,
                meta.cu_rows.ptr(0),
                meta.keep_d.ptr(0),
                state.ptr(0),
                state_out.ptr(0),
                state_stride as i64,
                meta.state_off.ptr(0),
                saves.k.ptr(0),
                saves.v.ptr(0),
                saves.g.ptr(0),
                saves.b.ptr(0),
                saves.kv_stride() as i64,
                saves.b_stride() as i64,
                stream.raw(),
            ),
            StateType::Bf16 => ffi::glm53f_kda_replay_batch_bf16state(
                heads,
                layers,
                batch,
                meta.cu_rows.ptr(0),
                meta.keep_d.ptr(0),
                state.ptr(0),
                state_out.ptr(0),
                state_stride as i64,
                meta.state_off.ptr(0),
                saves.k.ptr(0),
                saves.v.ptr(0),
                saves.g.ptr(0),
                saves.b.ptr(0),
                saves.kv_stride() as i64,
                saves.b_stride() as i64,
                stream.raw(),
            ),
        }
    };
    launched(code, "glm53f_kda_replay_batch")
}

/// Advance every layer's conv window of every request past its kept rows
/// (`glm53f_kda_conv_shift_batch`). Layer `l`'s window of request `r` is at
/// `l * conv_stride + r.conv_offset`; its new rows are in `p` at `l * p_layer_stride`, request
/// rows consecutive as in the chain.
#[allow(clippy::too_many_arguments)]
pub fn conv_shift_batch(
    heads: usize,
    layers: usize,
    meta: &BatchMeta,
    conv: &DeviceBuffer,
    conv_stride: usize,
    p: RowView<'_>,
    p_layer_stride: usize,
    stream: &Stream,
) -> Result<(), Error> {
    let c = channels(heads);
    let batch = meta.batch()?;
    if layers == 0 {
        return Err(Error::Invalid("conv_shift_batch: no layers".into()));
    }
    let max_off = meta
        .requests
        .iter()
        .map(|r| r.conv_offset)
        .max()
        .unwrap_or(0);
    if layers > 1 && conv_stride < max_off + WINDOW * c {
        return Err(Error::Invalid(
            "conv_shift_batch: layers' windows overlap".into(),
        ));
    }
    meta.check_regions(
        "conv",
        conv,
        2,
        (layers - 1) * conv_stride,
        WINDOW * c,
        true,
        |r| r.conv_offset,
    )?;
    for l in [0, layers - 1] {
        let view = RowView {
            buf: p.buf,
            offset: p.offset + l * p_layer_stride,
            stride: p.stride,
        };
        view.check("p", 2, 0, meta.total_rows(), 0, c as i64)?;
    }
    // SAFETY: regions checked above against the metadata's host copy.
    let code = unsafe {
        ffi::glm53f_kda_conv_shift_batch(
            to_i32(c, "channels")?,
            to_i32(layers, "layers")?,
            batch,
            meta.cu_rows.ptr(0),
            meta.keep_d.ptr(0),
            conv.ptr(0),
            conv_stride as i64,
            meta.conv_off.ptr(0),
            p.buf.ptr(p.offset),
            p_layer_stride as i64,
            p.stride as i64,
            stream.raw(),
        )
    };
    launched(code, "glm53f_kda_conv_shift_batch")
}

/// The gate lower bounds the prefill accepts: fifteen rows of decay must stay inside f32's
/// normal range in its in-chunk products.
pub const PREFILL_LOWER_MIN: f32 = -5.8;

fn check_lower(lower: f32) -> Result<(), Error> {
    if !(PREFILL_LOWER_MIN..=0.0).contains(&lower) {
        return Err(Error::Invalid(format!(
            "prefill: gate lower bound {lower} outside [{PREFILL_LOWER_MIN}, 0]"
        )));
    }
    Ok(())
}

/// Device workspace for the prefill: the per-chunk results passed from its first pass to its
/// second, for `rows_per_pass` rows of every request at a time (2.2 MB per 16 rows of a
/// 64-head request). Each pass costs a round trip of the state and a fixed overhead, so larger
/// passes are faster, with diminishing returns; see the README's prefill section.
pub struct PrefillWorkspace {
    pub heads: usize,
    pub batch: usize,
    pub rows_per_pass: usize,
    pub buf: DeviceBuffer,
}

impl PrefillWorkspace {
    /// Room for `rows_per_pass` rows (rounded up to 16) of `batch` requests of `heads` heads.
    pub fn new(heads: usize, batch: usize, rows_per_pass: usize) -> Result<Self, Error> {
        // SAFETY: a pure size computation.
        let bytes = unsafe {
            ffi::glm53f_kda_prefill_workspace_bytes(
                to_i32(heads, "heads")?,
                to_i32(batch, "batch")?,
                to_i32(rows_per_pass.max(1), "rows_per_pass")?,
            )
        };
        Ok(PrefillWorkspace {
            heads,
            batch,
            rows_per_pass: rows_per_pass.max(1).div_ceil(16) * 16,
            buf: DeviceBuffer::alloc(bytes as usize)?,
        })
    }

    /// The rows per pass that keep the workspace within `bytes` (at least one 16-row chunk).
    pub fn within(heads: usize, batch: usize, bytes: usize) -> Result<Self, Error> {
        // SAFETY: a pure size computation.
        let chunk = unsafe {
            ffi::glm53f_kda_prefill_workspace_bytes(
                to_i32(heads, "heads")?,
                to_i32(batch, "batch")?,
                16,
            )
        } as usize;
        Self::new(heads, batch, (bytes / chunk.max(1)).max(1) * 16)
    }

    fn args(&self, heads: usize, batch: usize) -> Result<(*mut f32, i64), Error> {
        if heads != self.heads || batch > self.batch {
            return Err(Error::Invalid(format!(
                "workspace for {} heads x {} requests used for {heads} x {batch}",
                self.heads, self.batch
            )));
        }
        // The kernels size their passes from the bytes they are given, for the launch's batch.
        Ok((self.buf.ptr(0), self.buf.bytes() as i64))
    }
}

fn check_value_blocks(value_blocks: usize) -> Result<i32, Error> {
    match value_blocks {
        0 | 1 | 2 | 4 => Ok(value_blocks as i32),
        _ => Err(Error::Invalid(format!(
            "value_blocks {value_blocks}: 0 (choose), 1, 2 or 4"
        ))),
    }
}

/// One layer, one request: a prompt segment through the chunked prefill
/// (`glm53f_kda_prefill`), from the state and conv window it is given. The conv window is
/// advanced past the rows in place; the state after the rows goes to `state_out`.
pub struct Prefill<'a> {
    pub weights: &'a Weights,
    pub rows: usize,
    /// Row `r`: q | k | v in columns `[0, C)`, the beta logits in `[b_off, b_off + H)`.
    pub p: RowView<'a>,
    pub b_off: i64,
    pub a: RowView<'a>,
    pub g: RowView<'a>,
    /// The conv window `[WINDOW][C]` at this element offset (updated in place).
    pub conv: &'a DeviceBuffer,
    pub conv_offset: usize,
    pub state: &'a DeviceBuffer,
    pub state_offset: usize,
    /// `InPlace` or `To`; the prefill always writes its state.
    pub state_out: StateOut<'a>,
    pub out: RowView<'a>,
    /// 1, 2 or 4 blocks per head, or 0 for the most that run as one wave on this GPU. The
    /// choice changes the order of some sums; pin it where bits must not depend on the GPU.
    pub value_blocks: usize,
    pub workspace: &'a PrefillWorkspace,
}

impl Prefill<'_> {
    pub fn launch(&self, stream: &Stream) -> Result<(), Error> {
        let h = self.weights.heads;
        let vb = check_value_blocks(self.value_blocks)?;
        check_lower(self.weights.lower)?;
        let c = channels(h) as i64;
        let (lo, hi) = (self.b_off.min(0), c.max(self.b_off + h as i64));
        self.p.check("p", 2, 0, self.rows, lo, hi)?;
        self.a.check("a", 2, 0, self.rows, 0, (h * DK) as i64)?;
        self.g.check("g", 2, 0, self.rows, 0, (h * DV) as i64)?;
        self.out.check("out", 2, 0, self.rows, 0, (h * DV) as i64)?;
        if self.rows > 1 && self.out.stride < h * DV {
            return Err(Error::Invalid("out rows overlap".into()));
        }
        check_region("conv", self.conv, 2, self.conv_offset, WINDOW * channels(h))?;
        check_region("state", self.state, 4, self.state_offset, state_len(h))?;
        let state_out = match self.state_out {
            StateOut::Skip => {
                return Err(Error::Invalid("a prefill always writes its state".into()));
            }
            StateOut::InPlace => self.state.ptr::<f32>(self.state_offset),
            StateOut::To(buf, off) => {
                check_region("state_out", buf, 4, off, state_len(h))?;
                if same(buf, self.state)
                    && !coincide_or_disjoint(off, self.state_offset, state_len(h))
                {
                    return Err(Error::Invalid("state_out partly overlaps the state".into()));
                }
                buf.ptr::<f32>(off)
            }
        };
        let (ws, ws_bytes) = self.workspace.args(h, 1)?;
        let w = self.weights;
        // SAFETY: every region the kernels touch was checked above.
        let code = unsafe {
            ffi::glm53f_kda_prefill(
                to_i32(h, "heads")?,
                to_i32(self.rows, "rows")?,
                self.p.buf.ptr(self.p.offset),
                self.p.stride as i64,
                self.b_off,
                self.a.buf.ptr(self.a.offset),
                self.a.stride as i64,
                self.g.buf.ptr(self.g.offset),
                self.g.stride as i64,
                self.conv.ptr(self.conv_offset),
                w.conv_w.ptr(0),
                self.state.ptr(self.state_offset),
                state_out,
                w.a_log.ptr(0),
                w.dt_bias.ptr(0),
                w.norm_w.ptr(0),
                w.eps,
                w.lower,
                self.out.buf.ptr(self.out.offset),
                self.out.stride as i64,
                vb,
                ws,
                ws_bytes,
                stream.raw(),
            )
        };
        launched(code, "glm53f_kda_prefill")
    }
}

/// One layer, a batch of prompt segments (`glm53f_kda_prefill_batch`): request rows consecutive
/// in `p`, `a`, `g` and `out` in the order of the metadata, each request with its own state and
/// conv window (advanced in place).
pub struct PrefillBatch<'a> {
    pub weights: &'a Weights,
    pub p: RowView<'a>,
    pub b_off: i64,
    pub a: RowView<'a>,
    pub g: RowView<'a>,
    pub conv: &'a DeviceBuffer,
    pub state: &'a DeviceBuffer,
    /// `InPlace`, or `To(buf, _)`: each request's state at its `state_offset` in `buf`.
    pub state_out: StateOut<'a>,
    pub out: RowView<'a>,
    /// As in [`Prefill`]; with 0 the choice depends on the batch size too.
    pub value_blocks: usize,
    pub workspace: &'a PrefillWorkspace,
}

impl PrefillBatch<'_> {
    pub fn launch(&self, meta: &BatchMeta, stream: &Stream) -> Result<(), Error> {
        self.launch_as(meta, stream, StateType::F32)
    }

    /// With bf16 states (`glm53f_kda_prefill_batch_bf16state`): `state` and `state_out` hold
    /// bf16 values, and the requests' `state_offset`s count bf16 elements.
    pub fn launch_bf16_state(&self, meta: &BatchMeta, stream: &Stream) -> Result<(), Error> {
        self.launch_as(meta, stream, StateType::Bf16)
    }

    fn launch_as(&self, meta: &BatchMeta, stream: &Stream, st: StateType) -> Result<(), Error> {
        let es = st.bytes();
        let h = self.weights.heads;
        let vb = check_value_blocks(self.value_blocks)?;
        check_lower(self.weights.lower)?;
        let batch = meta.batch()?;
        let total = meta.total_rows();
        let c = channels(h) as i64;
        let (lo, hi) = (self.b_off.min(0), c.max(self.b_off + h as i64));
        self.p.check("p", 2, 0, total, lo, hi)?;
        self.a.check("a", 2, 0, total, 0, (h * DK) as i64)?;
        self.g.check("g", 2, 0, total, 0, (h * DV) as i64)?;
        self.out.check("out", 2, 0, total, 0, (h * DV) as i64)?;
        if total > 1 && self.out.stride < h * DV {
            return Err(Error::Invalid("out rows overlap".into()));
        }
        // The windows are written: they must not overlap.
        meta.check_regions("conv", self.conv, 2, 0, WINDOW * channels(h), true, |r| {
            r.conv_offset
        })?;
        meta.check_regions("state", self.state, es, 0, state_len(h), true, |r| {
            r.state_offset
        })?;
        let state_out = match self.state_out {
            StateOut::Skip => {
                return Err(Error::Invalid("a prefill always writes its state".into()));
            }
            StateOut::InPlace => self.state.ptr::<u8>(0),
            StateOut::To(buf, _) => {
                meta.check_regions("state_out", buf, es, 0, state_len(h), true, |r| {
                    r.state_offset
                })?;
                if same(buf, self.state) {
                    return Err(Error::Invalid(
                        "state_out is the state buffer: use InPlace".into(),
                    ));
                }
                buf.ptr::<u8>(0)
            }
        };
        let (ws, ws_bytes) = self.workspace.args(h, batch as usize)?;
        let max_rows = meta.requests().iter().map(|r| r.rows).max().unwrap_or(0);
        let (cu_rows, _, conv_off, state_off) = meta.device_arrays();
        let w = self.weights;
        let (heads, max_rows) = (to_i32(h, "heads")?, to_i32(max_rows, "max_rows")?);
        // SAFETY: every region was checked above against the metadata's host copy.
        let code = unsafe {
            match st {
                StateType::F32 => ffi::glm53f_kda_prefill_batch(
                    heads,
                    batch,
                    cu_rows,
                    max_rows,
                    self.p.buf.ptr(self.p.offset),
                    self.p.stride as i64,
                    self.b_off,
                    self.a.buf.ptr(self.a.offset),
                    self.a.stride as i64,
                    self.g.buf.ptr(self.g.offset),
                    self.g.stride as i64,
                    self.conv.ptr(0),
                    conv_off,
                    w.conv_w.ptr(0),
                    self.state.ptr(0),
                    state_out.cast(),
                    state_off,
                    w.a_log.ptr(0),
                    w.dt_bias.ptr(0),
                    w.norm_w.ptr(0),
                    w.eps,
                    w.lower,
                    self.out.buf.ptr(self.out.offset),
                    self.out.stride as i64,
                    vb,
                    ws,
                    ws_bytes,
                    stream.raw(),
                ),
                StateType::Bf16 => ffi::glm53f_kda_prefill_batch_bf16state(
                    heads,
                    batch,
                    cu_rows,
                    max_rows,
                    self.p.buf.ptr(self.p.offset),
                    self.p.stride as i64,
                    self.b_off,
                    self.a.buf.ptr(self.a.offset),
                    self.a.stride as i64,
                    self.g.buf.ptr(self.g.offset),
                    self.g.stride as i64,
                    self.conv.ptr(0),
                    conv_off,
                    w.conv_w.ptr(0),
                    self.state.ptr(0),
                    state_out.cast(),
                    state_off,
                    w.a_log.ptr(0),
                    w.dt_bias.ptr(0),
                    w.norm_w.ptr(0),
                    w.eps,
                    w.lower,
                    self.out.buf.ptr(self.out.offset),
                    self.out.stride as i64,
                    vb,
                    ws,
                    ws_bytes,
                    stream.raw(),
                ),
            }
        };
        launched(code, "glm53f_kda_prefill_batch")
    }
}
