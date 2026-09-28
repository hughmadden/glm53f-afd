//! Routed experts on the four expert ranks (feature `coordinator`): [`RemoteExperts`], the
//! [`ExpertBackend`] the engine serves with, over the serving shell's wire client
//! (`glm53f_coordinator::wire::WireClient`, TCP or RDMA RC).
//!
//! One MoE layer's exchange:
//!
//! 1. [`ExpertBackend::submit`]: the FFN input (BF16, after `post_attention_layernorm`) is widened
//!    to f32 and quantized on the device into the wire's rows, E4M3 with one UE8M0 scale per 32
//!    values (`glm53f_coord_quant_scales`, `glm53f_coord_quantize_hidden`: bit for bit the wire
//!    client's host quantizer). The rows go to the four ranks with the routes, by one of three
//!    paths ([`FastPaths`]). The forward's `x_q` / `x_scales` are not used: their scales are per
//!    128 values in f32, not the wire's.
//! 2. The forward enqueues the shared expert on its stream: the GPU computes it while the ranks
//!    compute the routed experts.
//! 3. [`ExpertBackend::finish`]: waits for the ranks' returns (`WireClient::moe_recv_raw`) and
//!    writes the routed output into `call.out` on the stream, in the layout the exchange's return
//!    path gave (`WireClient::collected`):
//!    - **four planes** (every exchange by default; decode and verify windows always): each rank's
//!      BF16 partial of every row, added in rank order in f32 on the device
//!      (`glm53f_coord_rank_sum_bf16`, bit for bit the host `CoordinatorSum`) and rounded to BF16;
//!    - **row slices** (the prefill reduce-scatter, from `GLM53F_ROW_SHARDED_MIN_ROWS` rows; see
//!      `WireConfig::from_env`): the ranks added the four partials among themselves and each
//!      returns its quarter of the rows, final BF16, so each slice is copied to row `first` of
//!      `call.out` as it is: no sum, no f32. Within the bound of the four-plane sum that
//!      glm53f-rank's README derives, not its bits.
//!
//! **The routed scale.** The router's top-8 weights already carry `routed_scaling_factor` (2.5),
//! as the reference folds it, and they travel as the wire's gate weights. So the client's
//! `routed_scale` must be 1.0 ([`WireConfig::glm53_flash`]); [`RemoteExperts::new`] refuses
//! anything else rather than apply the scale twice (a row slice is used as it arrives).
//!
//! # The fast paths (mimo26f-afd perf resets P6 and P9)
//!
//! Each has an off switch ([`FastPaths::from_env`]); with both off, this is the host path the
//! engine had before them.
//!
//! - **Requests from the device** (`GLM53F_WIRE_DEVICE_ENCODE`, on by default). The wire client's
//!   request body is page-locked and mapped (`cudaHostRegister`, `WireClient::send_buffers`). The
//!   host encodes the row descriptors and route entries into the body while the GPU quantizes,
//!   then two DMA copies put the payload and the scales at the body's row pitch of 4,224 bytes
//!   (`WireClient::moe_send_mapped`, P6). Requests of at most [`FastPaths::fill_rows`] rows
//!   (`GLM53F_WIRE_FILL_ROWS`, none by default) are written instead by the frame-fill kernel,
//!   route entries (from the device routes) and rows, into the mapped body
//!   (`WireClient::moe_send_device`, P9). The source filled requests of up to 64 rows and copied
//!   larger ones (it measured the kernel's PCIe writes slower at prefill sizes). On the
//!   development GPU the kernel is slower at every size (`examples/wire_paths.rs`: 13 against
//!   8 us at 8 rows, 68 against 17 us at 64), and the forward brings the routes to the host
//!   anyway, so the copies are the default and the fill a switch for the target hardware. Off:
//!   the rows come to host memory, and the wire client encodes the frame and copies it into the
//!   body.
//! - **Returns read where they land** (`GLM53F_WIRE_ZERO_COPY`, on by default). The receive
//!   buffers (the RDMA rings, `WireClient::plane_buffers`) are page-locked and mapped: four-plane
//!   returns of at most [`FastPaths::mapped_rows`] rows (`GLM53F_WIRE_MAPPED_ROWS`, 64 by
//!   default, the source's) are summed in place, the rank-sum kernel reading the four planes from
//!   host memory; larger ones and row slices are DMA copies. Measured on the development GPU
//!   (`examples/wire_paths.rs`), in place is the fastest at decode sizes (18 against 28 us for
//!   8 rows) and level with the copies from about 512 rows; either way the host thread no longer
//!   waits for a pageable upload (4.1 ms for four planes of 2,048 rows, against 9 us to enqueue
//!   the copies). Off: pageable uploads.
//! - Either way the frames are the host encoder's bytes and the sums the same kernel's, so the
//!   paths change timings, never results.
//!
//! **When the returns may be overwritten.** The returns lie in the wire client's receive buffers
//! until the next send, which re-posts the RDMA slots to the next returns. `finish` enqueues the
//! reads (copies, or the in-place sum) and records an event after them; `submit` waits on it
//! before it sends. The forward's lane loop has already passed that point on the stream (it
//! waits for the routes first), so the wait costs nothing there.
//!
//! **Over TCP** (tests, and no fabric) the same paths run: the wire client stages the request
//! body and writes each rank's frame from it, and its receive buffers stay put once handed out.
//! They save nothing there, but a test on one machine exercises them end to end.
//!
//! **Overlap.** The wire client's `moe_recv_during` hook runs host work between the post and the
//! collection. The forward's shared expert is device work, enqueued between `submit` and
//! `finish`, so it already runs on the GPU while `finish` blocks on the returns; the hook has
//! nothing to add and is not used.
//!
//! **Two exchanges in flight** (the forward's two-lane prefill; mimo26f-afd perf reset R4). Over
//! RDMA the wire client takes a second request while the first is out (each rank pre-posts two
//! receive slots), so [`ExpertBackend::depth`] is 2 and `finish` collects the oldest. Over TCP it
//! is 1: a second multi-megabyte write can block against the first return. `GLM53F_WIRE_INFLIGHT=1`
//! holds an RDMA wire to one as well (the lanes then take turns on the wire, as over TCP). Each
//! exchange keeps its own return path (the wire client decides by its row count), and `finish`
//! checks the layout it collects against it. The device buffers are reused safely either way:
//! `submit` waits for its device work (the download, or the copies into the request body)
//! before it returns, and every `finish` enqueues its reads on the one stream, in order.
//!
//! **Numerics.** The ranks hold EXL3 4-bit experts and receive FP8 rows, so the routed output is
//! not the local FP8 experts' bits: the rank crate measures a cosine of at least 0.990 against the
//! reference (`glm53f-rank`, `tests/real_experts.rs`). A row's result does not depend on the
//! other rows of its pass, within one of the rank kernel's two configurations (up to 64 rows, and
//! above).
//!
//! A failed exchange leaves the ranks' connections in an unknown state (a rank closes its
//! connection after a failed request), so the backend refuses every later call with the first
//! error; the coordinator has to reconnect by restarting.

use std::collections::{BTreeMap, VecDeque};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use glm53f_coordinator::gpu as cgpu;
use glm53f_coordinator::wire::{Collected, ReturnPath, WireClient, WireConfig};

use crate::cuda;
use crate::device::{check, launched, DeviceBuffer, Event, Stream};
use crate::error::{invalid, Error, Result};
use crate::experts::{ExpertBackend, ExpertCall};
use crate::ffi;
use crate::shape::{HIDDEN, TOP_K};

/// Expert ranks (the wire's four TP4 shares).
pub const RANKS: usize = 4;
/// Rows of one exchange at most (the wire's request cap).
pub const MAX_ROWS: usize = 4096;
/// UE8M0 scale bytes per wire row (one per 32 values).
const SCALES: usize = HIDDEN / 32;
/// Bytes of one wire route entry (row, expert, weight).
const ROUTE_ENTRY: usize = 12;
/// The largest four-plane return summed in place by default: 64 rows (the source's
/// `ZERO_COPY_PLANE`, 512 KB of BF16 per rank): decode and verify windows.
pub const DEFAULT_MAPPED_ROWS: usize = 64;
/// The largest request written by the frame-fill kernel by default: none (the DMA copies at
/// every size). The source filled requests of up to 64 rows; `examples/wire_paths.rs` measures
/// the kernel's writes through the mapping slower than the copies at every size on the
/// development GPU, and the forward has the routes on the host anyway.
pub const DEFAULT_FILL_ROWS: usize = 0;

/// The CUDA runtime calls only the fast paths make.
mod rt {
    use core::ffi::{c_int, c_uint, c_void};

    use crate::cuda::{CudaError, RawStream};

    /// `cudaHostRegisterMapped`: page-locked, and mapped into the device's address space.
    pub const REGISTER_MAPPED: c_uint = 2;

    unsafe extern "C" {
        pub fn cudaHostRegister(ptr: *mut c_void, size: usize, flags: c_uint) -> CudaError;
        pub fn cudaHostUnregister(ptr: *mut c_void) -> CudaError;
        pub fn cudaMemcpy2DAsync(
            dst: *mut c_void,
            dpitch: usize,
            src: *const c_void,
            spitch: usize,
            width: usize,
            height: usize,
            kind: c_int,
            stream: RawStream,
        ) -> CudaError;
    }
}

/// Which of the fast paths an exchange takes (module documentation).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FastPaths {
    /// Requests built from the device into the page-locked request body (P6 copies, P9 fill);
    /// off: the host encode.
    pub device_encode: bool,
    /// Returns read from the page-locked receive buffers (in place, or DMA copies); off:
    /// pageable uploads.
    pub zero_copy: bool,
    /// Requests of at most this many rows are written by the frame-fill kernel (P9), larger ones
    /// by the DMA copies (P6). 0: always the copies.
    pub fill_rows: usize,
    /// Four-plane returns of at most this many rows are summed in place, larger ones copied
    /// first. 0: always the copies.
    pub mapped_rows: usize,
}

impl FastPaths {
    /// Both paths on, with the default thresholds.
    pub const ON: FastPaths = FastPaths {
        device_encode: true,
        zero_copy: true,
        fill_rows: DEFAULT_FILL_ROWS,
        mapped_rows: DEFAULT_MAPPED_ROWS,
    };
    /// The host paths only.
    pub const OFF: FastPaths = FastPaths {
        device_encode: false,
        zero_copy: false,
        fill_rows: DEFAULT_FILL_ROWS,
        mapped_rows: DEFAULT_MAPPED_ROWS,
    };

    /// [`FastPaths::ON`] with the environment's switches: `GLM53F_WIRE_DEVICE_ENCODE=0` (the
    /// host encode), `GLM53F_WIRE_ZERO_COPY=0` (pageable uploads), `GLM53F_WIRE_FILL_ROWS=N`
    /// and `GLM53F_WIRE_MAPPED_ROWS=N` (the thresholds).
    pub fn from_env() -> FastPaths {
        let on = |k: &str| std::env::var(k).map(|v| v.trim() != "0").unwrap_or(true);
        let rows = |k: &str, d: usize| {
            std::env::var(k)
                .ok()
                .and_then(|v| v.trim().parse().ok())
                .unwrap_or(d)
        };
        FastPaths {
            device_encode: on("GLM53F_WIRE_DEVICE_ENCODE"),
            zero_copy: on("GLM53F_WIRE_ZERO_COPY"),
            fill_rows: rows("GLM53F_WIRE_FILL_ROWS", DEFAULT_FILL_ROWS),
            mapped_rows: rows("GLM53F_WIRE_MAPPED_ROWS", DEFAULT_MAPPED_ROWS),
        }
    }
}

/// Wall-clock times of one kind of exchange (a layer and a row count), as the coordinator sees
/// them.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct ExchangeTimes {
    pub count: usize,
    /// Its return path: row slices (the reduce-scatter) rather than four planes.
    pub row_sharded: bool,
    /// From the request handed to the transport (the send call returned) to the last return
    /// read (the ranks' compute included), in milliseconds: total, smallest and largest.
    pub total_ms: f64,
    pub min_ms: f64,
    pub max_ms: f64,
    /// Of the total, the time `finish` blocked on the returns (after the shared expert was
    /// enqueued).
    pub wait_ms: f64,
    /// Host time outside the exchange: in `submit`, from the call to the request posted (the
    /// wire rows' quantization, their download or their copy into the request body, the frames'
    /// encode and post); in `finish`, after the returns were read (placing them: the uploads,
    /// the sum enqueued). Totals.
    pub send_ms: f64,
    pub upload_ms: f64,
    /// Of `send_ms`, the time `submit` waited for the device: the rows downloaded, or copied or
    /// filled into the request body.
    pub send_wait_ms: f64,
}

impl ExchangeTimes {
    pub fn mean_ms(&self) -> f64 {
        self.total_ms / self.count.max(1) as f64
    }

    fn add(&mut self, ms: f64, wait_ms: f64, s: &Sent, upload_ms: f64) {
        if self.count == 0 || ms < self.min_ms {
            self.min_ms = ms;
        }
        self.max_ms = self.max_ms.max(ms);
        self.count += 1;
        self.row_sharded = s.row_sharded;
        self.total_ms += ms;
        self.wait_ms += wait_ms;
        self.send_ms += s.send_ms;
        self.send_wait_ms += s.send_wait_ms;
        self.upload_ms += upload_ms;
    }
}

/// Exchange times by (layer, rows), shared with whoever holds [`RemoteExperts::times`].
pub type WireTimes = BTreeMap<(usize, usize), ExchangeTimes>;

/// How a request goes out.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum SendPath {
    /// Downloaded to host memory, encoded by the wire client.
    Host,
    /// Copied (DMA) into the page-locked request body (P6).
    Copy,
    /// Written with its route entries into the mapped request body by the frame fill (P9).
    Fill,
}

/// An exchange in flight: its layer, rows and return path, when it was handed to the
/// transport, and the host time `submit` spent getting it out.
struct Sent {
    layer: usize,
    rows: usize,
    row_sharded: bool,
    start: Instant,
    send_ms: f64,
    send_wait_ms: f64,
}

/// One exchange of a traced pass: (row-sharded, send, of it waiting for the device, finish
/// waiting, finish placing), in milliseconds.
type PassRec = (bool, f64, f64, f64, f64);

/// A host range page-locked and mapped for the device, unregistered when the backend drops.
struct Mapped {
    host: usize,
    bytes: usize,
    dev: usize,
}

/// Page-lock and map `bufs` (`cudaHostRegisterMapped`): all of them, or none.
fn register(bufs: Vec<(*mut u8, usize)>) -> Result<Vec<Mapped>> {
    let mut out: Vec<Mapped> = Vec::new();
    for (p, n) in bufs {
        // SAFETY: the wire client's live, page-aligned allocation of `n` bytes, which outlives
        // this backend's registration (unregistered in Drop, before the client is dropped).
        let r = check(
            unsafe { rt::cudaHostRegister(p.cast(), n, rt::REGISTER_MAPPED) },
            "cudaHostRegister",
        )
        .and_then(|()| {
            let mut dev = core::ptr::null_mut();
            // SAFETY: `p` was just registered, mapped.
            check(
                unsafe { cuda::cudaHostGetDevicePointer(&mut dev, p.cast(), 0) },
                "cudaHostGetDevicePointer",
            )
            .map(|()| dev as usize)
        });
        match r {
            Ok(dev) => out.push(Mapped {
                host: p as usize,
                bytes: n,
                dev,
            }),
            Err(e) => {
                // SAFETY: clear the error so later calls are unaffected.
                unsafe { cuda::cudaGetLastError() };
                unregister(&mut out);
                return Err(e);
            }
        }
    }
    Ok(out)
}

fn unregister(regions: &mut Vec<Mapped>) {
    for m in regions.drain(..) {
        // SAFETY: registered by `register`, unregistered once.
        unsafe { rt::cudaHostUnregister(m.host as *mut core::ffi::c_void) };
    }
}

/// The device address of the host bytes `[p, p + len)` when one of `regions` holds them.
fn dev_ptr(regions: &[Mapped], p: *const u8, len: usize) -> Option<*mut u8> {
    let a = p as usize;
    regions
        .iter()
        .find(|m| a >= m.host && a + len <= m.host + m.bytes)
        .map(|m| (m.dev + (a - m.host)) as *mut u8)
}

/// A host-to-device copy of `bytes` to `dst` on `stream`: asynchronous from page-locked memory;
/// from pageable memory the runtime stages the bytes before it returns.
fn upload(dst: *mut u8, bytes: &[u8], stream: &Stream) -> Result<()> {
    // SAFETY: `dst` holds `bytes.len()` bytes on the device (the callers' contracts).
    check(
        unsafe {
            cuda::cudaMemcpyAsync(
                dst.cast(),
                bytes.as_ptr().cast(),
                bytes.len(),
                cuda::MEMCPY_H2D,
                stream.raw(),
            )
        },
        "cudaMemcpyAsync H2D (returns)",
    )
}

/// The median of `v` (0 when empty).
fn median(mut v: Vec<f64>) -> f64 {
    if v.is_empty() {
        return 0.0;
    }
    v.sort_by(f64::total_cmp);
    v[v.len() / 2]
}

/// The routed experts on the expert ranks, through the wire client.
pub struct RemoteExperts {
    wire: WireClient,
    max_rows: usize,
    fast: FastPaths,
    /// f32 `[rows][4096]`: the widened FFN input.
    x32: DeviceBuffer,
    /// The wire rows: E4M3 `[rows][4096]`, then the UE8M0 scales `[rows][128]` right after the
    /// used payload rows (one download for both).
    q: DeviceBuffer,
    /// f32 `[rows][128]`: each scale's exact inverse.
    scale_inv: DeviceBuffer,
    /// BF16 `[4][rows][4096]`: the ranks' planes, and their f32 sum `[rows][4096]`.
    planes: DeviceBuffer,
    sum: DeviceBuffer,
    host_q: Vec<u8>,
    routes: Vec<(u32, f32)>,
    /// The wire client's request body and receive buffers, page-locked and mapped (empty when
    /// their path is off).
    body: Vec<Mapped>,
    rings: Vec<Mapped>,
    /// Recorded after each `finish`'s reads of the receive buffers (two: two exchanges may be
    /// collected between sends), and waited on before the next send.
    read_done: [Event; 2],
    reads: usize,
    /// Exchanges in flight, oldest first (at most [`ExpertBackend::depth`]), and the most a
    /// pipelined wire takes (`GLM53F_WIRE_INFLIGHT`, 2 by default).
    sent: VecDeque<Sent>,
    inflight: usize,
    failed: Option<String>,
    times: Arc<Mutex<WireTimes>>,
    /// The exchanges of the pass being traced ([`ExpertBackend::trace_begin`]).
    pass: Option<Vec<PassRec>>,
}

impl RemoteExperts {
    /// A backend over a connected wire client, for passes of up to `max_rows` rows (the forward's
    /// `max(max_rows, max_verify_rows)`), with the fast paths from the environment
    /// ([`FastPaths::from_env`]).
    pub fn new(wire: WireClient, max_rows: usize) -> Result<RemoteExperts> {
        RemoteExperts::with_paths(wire, max_rows, FastPaths::from_env())
    }

    /// [`RemoteExperts::new`] with the fast paths given. A buffer that cannot be page-locked
    /// turns its path off, with a line on stderr; another line names the paths in force.
    pub fn with_paths(wire: WireClient, max_rows: usize, fast: FastPaths) -> Result<RemoteExperts> {
        let cfg = *wire.config();
        if cfg.routed_scale != 1.0 {
            return Err(invalid!(
                "the router's weights carry the routed scale; the wire's routed_scale must be 1.0, not {}",
                cfg.routed_scale
            ));
        }
        if max_rows == 0 || max_rows > MAX_ROWS {
            return Err(invalid!("{max_rows} rows per exchange (1..={MAX_ROWS})"));
        }
        let r = max_rows;
        let mut me = RemoteExperts {
            wire,
            max_rows,
            fast,
            x32: DeviceBuffer::alloc(r * HIDDEN * 4)?,
            q: DeviceBuffer::alloc(r * (HIDDEN + SCALES))?,
            scale_inv: DeviceBuffer::alloc(r * SCALES * 4)?,
            planes: DeviceBuffer::alloc(RANKS * r * HIDDEN * 2)?,
            sum: DeviceBuffer::alloc(r * HIDDEN * 4)?,
            host_q: vec![0u8; r * (HIDDEN + SCALES)],
            routes: Vec::with_capacity(r * TOP_K),
            body: Vec::new(),
            rings: Vec::new(),
            read_done: [Event::new()?, Event::new()?],
            reads: 0,
            sent: VecDeque::new(),
            inflight: std::env::var("GLM53F_WIRE_INFLIGHT")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(2)
                .clamp(1, 2),
            failed: None,
            times: Arc::new(Mutex::new(WireTimes::default())),
            pass: None,
        };
        // Registered once the backend holds them, so Drop unregisters them before the client
        // frees them.
        if me.fast.device_encode {
            match register(me.wire.send_buffers()) {
                Ok(m) => me.body = m,
                Err(e) => {
                    eprintln!("[wire] the request body could not be page-locked ({e}): requests take the host encode");
                    me.fast.device_encode = false;
                }
            }
        }
        if me.fast.zero_copy {
            match register(me.wire.plane_buffers()) {
                Ok(m) => me.rings = m,
                Err(e) => {
                    eprintln!("[wire] the receive buffers could not be page-locked ({e}): returns take pageable uploads");
                    me.fast.zero_copy = false;
                }
            }
        }
        eprintln!("[wire] {}", me.describe());
        Ok(me)
    }

    /// The return path and the fast paths in force, one line.
    pub fn describe(&self) -> String {
        let ret = match self.wire.config().return_path {
            ReturnPath::FourPlaneSum => "four planes".to_string(),
            ReturnPath::RowSharded { min_rows, exchange } => format!(
                "row slices from {min_rows} rows ({} exchange between the ranks), four planes below",
                exchange.name()
            ),
        };
        let (f, m) = (self.fast.fill_rows, self.fast.mapped_rows);
        let req = match (self.fast.device_encode, f) {
            (false, _) => "host encode".to_string(),
            (true, 0) => "from the device (DMA copies into the page-locked body)".to_string(),
            (true, f) => format!("from the device (frame fill up to {f} rows, DMA copies above)"),
        };
        let rets = match (self.fast.zero_copy, m) {
            (false, _) => "pageable uploads".to_string(),
            (true, 0) => "page-locked (DMA copies)".to_string(),
            (true, m) => format!(
                "page-locked (four planes summed in place up to {m} rows, DMA copies above)"
            ),
        };
        format!("returns: {ret}; requests: {req}; receive buffers: {rets}")
    }

    /// The fast paths in force (a failed registration turns its path off).
    pub fn fast_paths(&self) -> FastPaths {
        self.fast
    }

    /// Device bytes the backend allocates for exchanges of up to `max_rows` rows (all at
    /// construction; an exchange allocates nothing). The fast paths add none: the page-locked
    /// memory is the wire client's own buffers.
    pub fn device_bytes(max_rows: usize) -> usize {
        max_rows * (HIDDEN * 4 + HIDDEN + SCALES + SCALES * 4 + RANKS * HIDDEN * 2 + HIDDEN * 4)
    }

    /// Connect to the four ranks (`addrs[r]` is rank `r`, `host:port`) with GLM-5.3-Flash's wire
    /// configuration, the return path from the environment (`WireConfig::from_env`:
    /// `GLM53F_ROW_SHARDED_MIN_ROWS`, `GLM53F_EXCHANGE_DTYPE`) and the fast paths from the
    /// environment ([`FastPaths::from_env`]). Over RDMA when `GLM53F_RDMA=1` (see
    /// `glm53f_coordinator::wire`).
    pub fn connect(addrs: &[String], max_rows: usize) -> Result<RemoteExperts> {
        let cfg = WireConfig::from_env().map_err(Error::Other)?;
        RemoteExperts::connect_with(addrs, max_rows, cfg, FastPaths::from_env())
    }

    /// [`RemoteExperts::connect`] with the wire configuration and the fast paths given.
    pub fn connect_with(
        addrs: &[String],
        max_rows: usize,
        cfg: WireConfig,
        fast: FastPaths,
    ) -> Result<RemoteExperts> {
        let wire = WireClient::connect(addrs, cfg).map_err(Error::Other)?;
        RemoteExperts::with_paths(wire, max_rows, fast)
    }

    /// The exchange times so far (a shared handle: take it before handing the backend to the
    /// forward). [`RemoteExperts::reset_times`] clears them.
    pub fn times(&self) -> Arc<Mutex<WireTimes>> {
        self.times.clone()
    }

    pub fn reset_times(&self) {
        *self.times.lock().unwrap_or_else(|p| p.into_inner()) = WireTimes::default();
    }

    fn live(&self) -> Result<()> {
        match &self.failed {
            Some(e) => Err(Error::Other(format!(
                "the expert ranks failed earlier ({e}); restart the coordinator"
            ))),
            None => Ok(()),
        }
    }

    /// Record a wire failure: this and every later call fail with it.
    fn fail(&mut self, e: String) -> Error {
        let msg = format!("expert wire: {e}");
        self.failed = Some(msg.clone());
        self.sent.clear();
        Error::Other(msg)
    }

    /// Wait until the collected returns have been read on the device: a send re-posts the
    /// receive buffers they lie in.
    fn reads_done(&self) -> Result<()> {
        for e in &self.read_done[..self.reads.min(2)] {
            e.synchronize()?;
        }
        Ok(())
    }

    /// Quantize `call.x` into the wire rows and send them with the call's routes.
    fn send(&mut self, call: &ExpertCall<'_>, stream: &Stream) -> Result<()> {
        let called = Instant::now();
        self.reads_done()?;
        let rows = call.rows;
        let st = stream.raw();
        let (x32, sinv) = (self.x32.ptr::<f32>(0), self.scale_inv.ptr::<f32>(0));
        let (payload, scales) = (self.q.byte_ptr(0), self.q.byte_ptr(rows * HIDDEN));
        // SAFETY: `call.x` holds rows x 4096 BF16 values on the device (the trait's contract);
        // the scratch holds max_rows >= rows rows.
        launched(
            unsafe {
                ffi::glm53f_fwd_bf16_to_f32(
                    call.x,
                    HIDDEN as i64,
                    x32,
                    HIDDEN as i64,
                    rows as i32,
                    HIDDEN as i32,
                    1.0,
                    st,
                )
            },
            "glm53f_fwd_bf16_to_f32 (wire rows)",
        )?;
        // SAFETY: as above; rows x 128 blocks of 32 values.
        check(
            unsafe {
                cgpu::glm53f_coord_quant_scales(x32, (rows * SCALES) as i64, scales, sinv, st)
            },
            "glm53f_coord_quant_scales",
        )?;
        check(
            unsafe {
                cgpu::glm53f_coord_quantize_hidden(x32, sinv, payload, (rows * HIDDEN) as i64, st)
            },
            "glm53f_coord_quantize_hidden",
        )?;
        let device_routes = !call.ids.is_null() && !call.weights.is_null();
        let path = if self.body.is_empty() {
            SendPath::Host
        } else if rows <= self.fast.fill_rows && device_routes {
            SendPath::Fill
        } else {
            SendPath::Copy
        };
        if path != SendPath::Fill {
            self.routes.clear();
            self.routes.extend(
                call.host_ids
                    .iter()
                    .zip(call.host_weights)
                    .map(|(&e, &w)| (e as u32, w)),
            );
        }
        let (payload, scales) = (payload as *const u8, scales as *const u8);
        let mut waited = 0f64;
        let sent = match path {
            SendPath::Host => {
                let n = rows * (HIDDEN + SCALES);
                let t = Instant::now();
                // Waits for the stream: the rows are the pass's latest work.
                self.q.download_bytes(stream, 0, &mut self.host_q[..n])?;
                waited = t.elapsed().as_secs_f64() * 1e3;
                let (p, s) = self.host_q[..n].split_at(rows * HIDDEN);
                self.wire
                    .moe_send_raw(call.layer as u32, p, s, &self.routes, TOP_K)
            }
            // The descriptors and route entries are encoded on the host while the GPU
            // quantizes; then the payload and the scales go by DMA to each row's place in the
            // page-locked body (pitch 4,224 bytes).
            SendPath::Copy => {
                self.wire
                    .moe_send_mapped(call.layer as u32, &self.routes, TOP_K, |dst, pitch| {
                        let t = Instant::now();
                        // SAFETY: `dst` is `rows` rows at `pitch` in the page-locked body (the
                        // wire client's contract); the sources hold rows x 4096 and rows x 128
                        // bytes on the device.
                        check(
                            unsafe {
                                rt::cudaMemcpy2DAsync(
                                    dst.cast(),
                                    pitch,
                                    payload.cast(),
                                    HIDDEN,
                                    HIDDEN,
                                    rows,
                                    cuda::MEMCPY_D2H,
                                    st,
                                )
                            },
                            "cudaMemcpy2DAsync (payload into the request body)",
                        )?;
                        check(
                            unsafe {
                                rt::cudaMemcpy2DAsync(
                                    dst.add(HIDDEN).cast(),
                                    pitch,
                                    scales.cast(),
                                    SCALES,
                                    SCALES,
                                    rows,
                                    cuda::MEMCPY_D2H,
                                    st,
                                )
                            },
                            "cudaMemcpy2DAsync (scales into the request body)",
                        )?;
                        stream.synchronize()?;
                        waited = t.elapsed().as_secs_f64() * 1e3;
                        Ok(())
                    })
            }
            // The kernel writes the route entries (from the device routes) and the rows into the
            // mapped body; only the descriptors and headers are host-written.
            SendPath::Fill => {
                let body = &self.body;
                self.wire.moe_send_device(
                    call.layer as u32,
                    rows,
                    TOP_K,
                    |routes, hidden, pitch| {
                        let t = Instant::now();
                        let r = dev_ptr(body, routes, rows * TOP_K * ROUTE_ENTRY)
                            .ok_or("the request body is not mapped")?;
                        let h = dev_ptr(body, hidden, rows * pitch)
                            .ok_or("the request body is not mapped")?;
                        // SAFETY: the device routes hold rows x 8 ids and weights (the trait's
                        // contract), the wire rows rows x (4096 + 128) bytes; `r` and `h` are the
                        // mapped body's route entries and rows (8-byte aligned: the body is
                        // page-aligned and each row's descriptor and routes take 136 bytes).
                        check(
                            unsafe {
                                cgpu::glm53f_coord_frame_fill(
                                    call.ids,
                                    call.weights,
                                    payload,
                                    scales,
                                    rows as i32,
                                    TOP_K as i32,
                                    HIDDEN as i32,
                                    r,
                                    h,
                                    pitch as i32,
                                    st,
                                )
                            },
                            "glm53f_coord_frame_fill",
                        )?;
                        stream.synchronize()?;
                        waited = t.elapsed().as_secs_f64() * 1e3;
                        Ok(())
                    },
                )
            }
        };
        if let Err(e) = sent {
            return Err(self.fail(e));
        }
        let start = Instant::now();
        self.sent.push_back(Sent {
            layer: call.layer,
            rows,
            row_sharded: self.wire.config().row_sharded(rows).is_some(),
            start,
            send_ms: (start - called).as_secs_f64() * 1e3,
            send_wait_ms: waited,
        });
        Ok(())
    }

    /// Collect the oldest exchange in flight and write its routed output into `call.out` on
    /// `stream`.
    fn collect(&mut self, call: &ExpertCall<'_>, stream: &Stream) -> Result<()> {
        let Some(sent) = self.sent.pop_front() else {
            return Err(invalid!("finish without a submitted exchange"));
        };
        if sent.layer != call.layer || sent.rows != call.rows {
            return Err(self.fail(format!(
                "finish of layer {} ({} rows) while layer {} ({} rows) is the oldest in flight",
                call.layer, call.rows, sent.layer, sent.rows
            )));
        }
        let wait = Instant::now();
        let got = self.wire.moe_recv_raw();
        let done = Instant::now();
        let (layer, rows) = match got {
            Ok(x) => x,
            Err(e) => return Err(self.fail(e)),
        };
        if layer as usize != sent.layer || rows != sent.rows {
            return Err(self.fail(format!(
                "collected layer {layer} ({rows} rows) for layer {} ({} rows)",
                sent.layer, sent.rows
            )));
        }
        if let Err(e) = self.place(call, rows, sent.row_sharded, stream) {
            return Err(self.fail(e));
        }
        let placed = Instant::now();
        let ms = |a: Instant, b: Instant| (b - a).as_secs_f64() * 1e3;
        let (wait_ms, place_ms) = (ms(wait, done), ms(done, placed));
        if let Some(p) = self.pass.as_mut() {
            p.push((
                sent.row_sharded,
                sent.send_ms,
                sent.send_wait_ms,
                wait_ms,
                place_ms,
            ));
        }
        self.times
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .entry((sent.layer, rows))
            .or_default()
            .add(ms(sent.start, done), wait_ms, &sent, place_ms);
        Ok(())
    }

    /// Write the collected returns of a `rows`-row exchange into `call.out` on `stream`, and
    /// record that the receive buffers have been read.
    fn place(
        &mut self,
        call: &ExpertCall<'_>,
        rows: usize,
        row_sharded: bool,
        stream: &Stream,
    ) -> std::result::Result<(), String> {
        let st = stream.raw();
        let plane = rows * HIDDEN * 2;
        match self.wire.collected() {
            Some(Collected::Planes(planes)) => {
                if row_sharded {
                    return Err(format!(
                        "{rows} rows came back as four planes, not row slices"
                    ));
                }
                let in_place = rows <= self.fast.mapped_rows;
                let mut p = [core::ptr::null::<u16>(); RANKS];
                for (r, bytes) in planes.iter().enumerate() {
                    let mapped = if in_place {
                        dev_ptr(&self.rings, bytes.as_ptr(), plane)
                    } else {
                        None
                    };
                    p[r] = match mapped {
                        // Read in place, from the mapped receive buffer.
                        Some(d) => d as *const u16,
                        None => {
                            upload(self.planes.byte_ptr(r * plane), bytes, stream)?;
                            self.planes.byte_ptr(r * plane) as *const u16
                        }
                    };
                }
                // SAFETY: four planes of rows x 4096 BF16 values (device memory, or mapped host
                // memory the receive buffers hold until the next send) and an f32 sum of the
                // same size; `out` holds rows x 4096 BF16 values (the trait's contract).
                check(
                    unsafe {
                        cgpu::glm53f_coord_rank_sum_bf16(
                            p[0],
                            p[1],
                            p[2],
                            p[3],
                            self.sum.ptr(0),
                            (rows * HIDDEN) as i64,
                            self.wire.routed_scale(),
                            st,
                        )
                    },
                    "glm53f_coord_rank_sum_bf16",
                )?;
                launched(
                    unsafe {
                        ffi::glm53f_fwd_f32_to_bf16(
                            self.sum.ptr(0),
                            HIDDEN as i64,
                            call.out,
                            HIDDEN as i64,
                            rows as i32,
                            HIDDEN as i32,
                            st,
                        )
                    },
                    "glm53f_fwd_f32_to_bf16 (routed sum)",
                )?;
            }
            Some(Collected::RowSlices(slices)) => {
                if !row_sharded {
                    return Err(format!(
                        "{rows} rows came back as row slices, not four planes"
                    ));
                }
                // Each rank's rows, summed over the ranks and final BF16 (the routed scale is 1),
                // go to their place in the output.
                for s in slices.iter() {
                    if s.first + s.rows > rows || s.bytes.len() != s.rows * HIDDEN * 2 {
                        return Err(format!("a row slice of {} rows at row {}", s.rows, s.first));
                    }
                    upload(
                        call.out.wrapping_add(s.first * HIDDEN).cast(),
                        s.bytes,
                        stream,
                    )?;
                }
            }
            None => return Err("no returns collected".into()),
        }
        self.read_done[self.reads % 2].record(stream)?;
        self.reads += 1;
        Ok(())
    }
}

impl Drop for RemoteExperts {
    fn drop(&mut self) {
        // Before the wire client (a field, dropped after this) frees the buffers, and after the
        // device's last reads of them.
        let _ = self.reads_done();
        unregister(&mut self.body);
        unregister(&mut self.rings);
    }
}

impl ExpertBackend for RemoteExperts {
    fn submit(&mut self, call: &ExpertCall<'_>, stream: &Stream) -> Result<()> {
        self.live()?;
        let rows = call.rows;
        if rows == 0 || rows > self.max_rows {
            return Err(invalid!(
                "{rows} rows for remote experts sized for {}",
                self.max_rows
            ));
        }
        if call.host_ids.len() != rows * TOP_K || call.host_weights.len() != rows * TOP_K {
            return Err(invalid!("routes for {rows} rows"));
        }
        if self.sent.len() >= self.depth() {
            return Err(invalid!(
                "{} exchanges already in flight (the wire takes {})",
                self.sent.len(),
                self.depth()
            ));
        }
        self.send(call, stream)
    }

    fn finish(&mut self, call: &ExpertCall<'_>, stream: &Stream) -> Result<()> {
        self.live()?;
        self.collect(call, stream)
    }

    fn depth(&self) -> usize {
        if self.wire.pipelined() {
            self.inflight
        } else {
            1
        }
    }

    fn trace_begin(&mut self) {
        self.pass = Some(Vec::new());
    }

    /// Per return path, the pass's exchanges and their medians: host time in `submit` (and of it
    /// waiting for the device), and in `finish` waiting for the returns, then placing them.
    fn trace_end(&mut self) -> Option<String> {
        let recs = self.pass.take()?;
        let modes: Vec<String> = [(true, "row slices"), (false, "four planes")]
            .iter()
            .map(|&(rs, name)| {
                let m: Vec<&PassRec> = recs.iter().filter(|r| r.0 == rs).collect();
                let med = |f: fn(&PassRec) -> f64| median(m.iter().map(|&r| f(r)).collect());
                if m.is_empty() {
                    format!("{name} 0")
                } else {
                    format!(
                        "{name} {}: send {:.2} (device {:.2}), finish wait {:.2} + place {:.2}",
                        m.len(),
                        med(|r| r.1),
                        med(|r| r.2),
                        med(|r| r.3),
                        med(|r| r.4)
                    )
                }
            })
            .collect();
        Some(format!(
            "wire ({} requests, {} returns), medians in ms: {}",
            if self.fast.device_encode {
                "device"
            } else {
                "host"
            },
            if self.fast.zero_copy {
                "page-locked"
            } else {
                "pageable"
            },
            modes.join("; ")
        ))
    }
}
