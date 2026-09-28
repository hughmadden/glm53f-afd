//! Routed experts on the four expert ranks (feature `coordinator`): [`RemoteExperts`], the
//! [`ExpertBackend`] the engine serves with, over the serving shell's wire client
//! (`glm53f_coordinator::wire::WireClient`, TCP or RDMA RC).
//!
//! One MoE layer's exchange:
//!
//! 1. [`ExpertBackend::submit`]: the FFN input (BF16, after `post_attention_layernorm`) is widened
//!    to f32 and quantized on the device into the wire's rows, E4M3 with one UE8M0 scale per 32
//!    values (`glm53f_coord_quant_scales`, `glm53f_coord_quantize_hidden`: bit for bit the wire
//!    client's host quantizer). The rows come to the host and go to the four ranks with the
//!    host routes (`WireClient::moe_send_raw`). The forward's `x_q` / `x_scales` are not used:
//!    their scales are per 128 values in f32, not the wire's.
//! 2. The forward enqueues the shared expert on its stream: the GPU computes it while the ranks
//!    compute the routed experts.
//! 3. [`ExpertBackend::finish`]: waits for the four ranks' BF16 partial planes
//!    (`WireClient::moe_recv_raw`), uploads them, adds them in rank order in f32 on the device
//!    (`glm53f_coord_rank_sum_bf16`, bit for bit the host `CoordinatorSum`), and rounds the sum to
//!    BF16 into `call.out`.
//!
//! **The routed scale.** The router's top-8 weights already carry `routed_scaling_factor` (2.5),
//! as the reference folds it, and they travel as the wire's gate weights. So the client's
//! `routed_scale` must be 1.0 ([`WireConfig::glm53_flash`]); [`RemoteExperts::new`] refuses
//! anything else rather than apply the scale twice.
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
//! holds an RDMA wire to one as well (the lanes then take turns on the wire, as over TCP). The
//! device buffers are reused safely either way: `submit` has the rows on the host before it
//! returns, and every `finish` uploads and sums on the one stream, in order.
//!
//! **Numerics.** The ranks hold EXL3 4-bit experts and receive FP8 rows, so the routed output is
//! not the local FP8 experts' bits: the rank crate measures a cosine of at least 0.990 against the
//! reference (`glm53f-rank`, `tests/real_experts.rs`). A row's result does not depend on the
//! other rows of its pass, within one of the rank kernel's two configurations (up to 64 rows, and
//! above).
//!
//! **Not ported yet** (mimo26f-afd's RDMA fast paths, perf resets P6 and P9): copying the rows
//! from the device straight into the registered request body, filling the route entries on the
//! device, and summing small returns in place from the mapped receive ring. Over RDMA this path
//! takes the wire client's host encode and a copy into the body; the fast paths need the fabric
//! to measure.
//!
//! A failed exchange leaves the ranks' connections in an unknown state (a rank closes its
//! connection after a failed request), so the backend refuses every later call with the first
//! error; the coordinator has to reconnect by restarting.

use std::collections::{BTreeMap, VecDeque};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use glm53f_coordinator::gpu as cgpu;
use glm53f_coordinator::wire::{ReturnPath, Returned, WireClient, WireConfig};

use crate::device::{check, launched, DeviceBuffer, Stream};
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

/// Wall-clock times of one kind of exchange (a layer and a row count), as the coordinator sees
/// them.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct ExchangeTimes {
    pub count: usize,
    /// From the first byte of the request written to the last byte of the returns read (the
    /// ranks' compute included), in milliseconds: total, smallest and largest.
    pub total_ms: f64,
    pub min_ms: f64,
    pub max_ms: f64,
    /// Of the total, the time `finish` blocked on the returns (after the shared expert was
    /// enqueued).
    pub wait_ms: f64,
    /// Host time outside the exchange: in `submit`, from the call to the request posted (the
    /// wire rows' quantization and download, the frames' encode and post); in `finish`, after
    /// the returns were read (the four planes' upload and the sum enqueued). Totals.
    pub send_ms: f64,
    pub upload_ms: f64,
}

impl ExchangeTimes {
    pub fn mean_ms(&self) -> f64 {
        self.total_ms / self.count.max(1) as f64
    }

    fn add(&mut self, ms: f64, wait_ms: f64, send_ms: f64, upload_ms: f64) {
        if self.count == 0 || ms < self.min_ms {
            self.min_ms = ms;
        }
        self.max_ms = self.max_ms.max(ms);
        self.count += 1;
        self.total_ms += ms;
        self.wait_ms += wait_ms;
        self.send_ms += send_ms;
        self.upload_ms += upload_ms;
    }
}

/// Exchange times by (layer, rows), shared with whoever holds [`RemoteExperts::times`].
pub type WireTimes = BTreeMap<(usize, usize), ExchangeTimes>;

/// An exchange in flight: its layer, rows, when its request started, and the host time `submit`
/// spent getting it out.
struct Sent {
    layer: usize,
    rows: usize,
    start: Instant,
    send_ms: f64,
}

/// The routed experts on the expert ranks, through the wire client.
pub struct RemoteExperts {
    wire: WireClient,
    max_rows: usize,
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
    /// Exchanges in flight, oldest first (at most [`ExpertBackend::depth`]), and the most a
    /// pipelined wire takes (`GLM53F_WIRE_INFLIGHT`, 2 by default).
    sent: VecDeque<Sent>,
    inflight: usize,
    failed: Option<String>,
    times: Arc<Mutex<WireTimes>>,
}

impl RemoteExperts {
    /// A backend over a connected wire client, for passes of up to `max_rows` rows (the forward's
    /// `max(max_rows, max_verify_rows)`).
    pub fn new(wire: WireClient, max_rows: usize) -> Result<RemoteExperts> {
        let cfg = *wire.config();
        if cfg.routed_scale != 1.0 {
            return Err(invalid!(
                "the router's weights carry the routed scale; the wire's routed_scale must be 1.0, not {}",
                cfg.routed_scale
            ));
        }
        if cfg.return_path != ReturnPath::FourPlaneSum {
            return Err(invalid!(
                "return path {:?} is not implemented",
                cfg.return_path
            ));
        }
        if max_rows == 0 || max_rows > MAX_ROWS {
            return Err(invalid!("{max_rows} rows per exchange (1..={MAX_ROWS})"));
        }
        let r = max_rows;
        Ok(RemoteExperts {
            wire,
            max_rows,
            x32: DeviceBuffer::alloc(r * HIDDEN * 4)?,
            q: DeviceBuffer::alloc(r * (HIDDEN + SCALES))?,
            scale_inv: DeviceBuffer::alloc(r * SCALES * 4)?,
            planes: DeviceBuffer::alloc(RANKS * r * HIDDEN * 2)?,
            sum: DeviceBuffer::alloc(r * HIDDEN * 4)?,
            host_q: vec![0u8; r * (HIDDEN + SCALES)],
            routes: Vec::with_capacity(r * TOP_K),
            sent: VecDeque::new(),
            inflight: std::env::var("GLM53F_WIRE_INFLIGHT")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(2)
                .clamp(1, 2),
            failed: None,
            times: Arc::new(Mutex::new(WireTimes::default())),
        })
    }

    /// Device bytes the backend allocates for exchanges of up to `max_rows` rows (all at
    /// construction; an exchange allocates nothing).
    pub fn device_bytes(max_rows: usize) -> usize {
        max_rows * (HIDDEN * 4 + HIDDEN + SCALES + SCALES * 4 + RANKS * HIDDEN * 2 + HIDDEN * 4)
    }

    /// Connect to the four ranks (`addrs[r]` is rank `r`, `host:port`) with GLM-5.3-Flash's wire
    /// configuration. Over RDMA when `GLM53F_RDMA=1` (see `glm53f_coordinator::wire`).
    pub fn connect(addrs: &[String], max_rows: usize) -> Result<RemoteExperts> {
        let wire = WireClient::connect(addrs, WireConfig::glm53_flash()).map_err(Error::Other)?;
        RemoteExperts::new(wire, max_rows)
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

    /// Quantize `call.x` into the wire rows and send them with the call's routes.
    fn send(&mut self, call: &ExpertCall<'_>, stream: &Stream) -> Result<()> {
        let called = Instant::now();
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
        let n = rows * (HIDDEN + SCALES);
        // Waits for the stream: the rows are the pass's latest work.
        self.q.download_bytes(stream, 0, &mut self.host_q[..n])?;
        self.routes.clear();
        self.routes.extend(
            call.host_ids
                .iter()
                .zip(call.host_weights)
                .map(|(&e, &w)| (e as u32, w)),
        );
        let start = Instant::now();
        let (p, s) = self.host_q[..n].split_at(rows * HIDDEN);
        if let Err(e) = self
            .wire
            .moe_send_raw(call.layer as u32, p, s, &self.routes, TOP_K)
        {
            return Err(self.fail(e));
        }
        self.sent.push_back(Sent {
            layer: call.layer,
            rows,
            start,
            send_ms: called.elapsed().as_secs_f64() * 1e3,
        });
        Ok(())
    }

    /// Collect the oldest exchange in flight and write its routed sum into `call.out` on
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
        let plane = rows * HIDDEN * 2;
        let Returned::Planes(planes) = self.wire.returned(rows);
        for (r, p) in planes.iter().enumerate() {
            // Pageable host memory: staged before the call returns, so the receive buffers may
            // take the next exchange.
            self.planes.upload_bytes_async(stream, r * plane, p)?;
        }
        let pl = |r: usize| self.planes.byte_ptr(r * plane) as *const u16;
        // SAFETY: four planes of rows x 4096 BF16 values and an f32 sum of the same size; `out`
        // holds rows x 4096 BF16 values (the trait's contract).
        check(
            unsafe {
                cgpu::glm53f_coord_rank_sum_bf16(
                    pl(0),
                    pl(1),
                    pl(2),
                    pl(3),
                    self.sum.ptr(0),
                    (rows * HIDDEN) as i64,
                    self.wire.routed_scale(),
                    stream.raw(),
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
                    stream.raw(),
                )
            },
            "glm53f_fwd_f32_to_bf16 (routed sum)",
        )?;
        let ms = |a: Instant, b: Instant| (b - a).as_secs_f64() * 1e3;
        self.times
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .entry((sent.layer, rows))
            .or_default()
            .add(
                ms(sent.start, done),
                ms(wait, done),
                sent.send_ms,
                ms(done, Instant::now()),
            );
        Ok(())
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
}
