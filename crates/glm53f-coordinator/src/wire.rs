//! Coordinator-side wire client: the expert ranks' client for one MoE layer at a time, over
//! TCP or RDMA RC (`GLM53F_RDMA=1`). It uses the `DS41RTE3` v3 codec and L4 ladder of
//! `glm53f-wire` (`StreamSender`/`StreamReceiver`) and the rank-ordered FP32 sum of the four
//! ranks' partials (`CoordinatorSum`).
//!
//! What changes for GLM-5.3-Flash:
//! - **Experts are a parameter** ([`WireConfig::experts`], 288): expert ids travel as `u32`
//!   route fields; the host paths refuse an id at or past it.
//! - **The routed scale is applied on the coordinator, never on a rank.** The reference folds
//!   `routed_scaling_factor` (2.5) into the router's top-8 weights, and `glm53f-layers`' router
//!   does the same, so the gate weights on the wire already carry it and the ranks' sum is used
//!   as it is ([`WireConfig::routed_scale`] 1.0, the default). A coordinator that sends the
//!   normalized weights instead sets it to 2.5, and the sum is multiplied by it (on the host path
//!   here; the device path passes it to `glm53f_coord_rank_sum_bf16`). Never both.
//! - **The shared expert runs during the remote wait** ([`WireClient::moe_recv_during`]): its
//!   hook runs after the request is posted and before the returns are collected.
//! - **The return handling sits behind [`ReturnPath`]**: by default every rank returns all rows
//!   and the coordinator sums the four BF16 planes; with [`ReturnPath::RowSharded`] the ranks
//!   reduce-scatter prefill-sized exchanges among themselves (`DS41RTE3` v4,
//!   `glm53f_wire::row_shard`) and each returns only its partition of the rows, summed, so the
//!   coordinator receives each row once. The choice is made per exchange, by its row count, and
//!   [`WireClient::collected`] gives each exchange's returns in their layout.
//! - **Request ids start at a random base per connection**, so a rank's peer mesh can never
//!   match a stale exchange frame of an earlier connection to a new exchange.

use std::io::{Read, Write};
use std::net::TcpStream;

use glm53f_wire::frame::{Frame, HiddenRow, RequestFrame, ReturnFrame, RouteEntry, RowDescriptor};
use glm53f_wire::l4::{CoordinatorSum, StreamReceiver, StreamSender};
use glm53f_wire::row_shard::{row_partition, ExchangeDtype};
use glm53f_wire::{SourceKind, WireNaive, HIDDEN, SPARKS};

/// How the ranks' returns of one exchange become its routed output.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ReturnPath {
    /// Every rank returns every row: its partial sum over its quarter of each expert, BF16. The
    /// coordinator adds the four planes in rank order, in FP32 (`CoordinatorSum` on the host,
    /// `glm53f_coord_rank_sum_bf16` on the device, bit-identical).
    FourPlaneSum,
    /// Exchanges of at least `min_rows` rows are reduce-scattered among the ranks, and each rank
    /// returns only its partition of the rows, already summed over the four ranks (no 4-to-1
    /// incast into the coordinator's port). The ranks exchange their rows as `exchange` (BF16 is
    /// the default: its error is close to the four-plane sum's). Smaller exchanges (decode and
    /// verify windows) keep the four-plane sum, whose latency is better. `min_rows` must be at
    /// least 4 (a row per rank); the design's threshold is 16 (`DEFAULT_ROW_SHARDED_MIN_ROWS`).
    /// The ranks need their peer mesh (`glm53f-rank serve --peers`).
    RowSharded { min_rows: usize, exchange: ExchangeDtype },
}

/// The design's threshold for [`ReturnPath::RowSharded`]: 16 rows and more.
pub const DEFAULT_ROW_SHARDED_MIN_ROWS: usize = 16;

/// The wire client's model parameters.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WireConfig {
    /// Routed experts per MoE layer (GLM-5.3-Flash: 288).
    pub experts: u32,
    /// Multiplies the sum of the ranks' partials: 1.0 when the gate weights carry the routed
    /// scale (see the module documentation).
    pub routed_scale: f32,
    pub return_path: ReturnPath,
}

impl WireConfig {
    /// GLM-5.3-Flash as the reference routes it: 288 experts, the routed scale 2.5 inside the
    /// gate weights (so 1.0 here), four-plane returns.
    pub fn glm53_flash() -> WireConfig {
        WireConfig { experts: 288, routed_scale: 1.0, return_path: ReturnPath::FourPlaneSum }
    }

    /// [`WireConfig::glm53_flash`] with the return path from the environment:
    /// `GLM53F_ROW_SHARDED_MIN_ROWS` (unset or 0: four-plane returns; otherwise
    /// [`ReturnPath::RowSharded`] from that many rows) and `GLM53F_EXCHANGE_DTYPE` (`bf16`, the
    /// default, or `fp8`).
    pub fn from_env() -> Result<WireConfig, String> {
        let var = |k: &str| std::env::var(k).ok().filter(|v| !v.is_empty());
        let mut cfg = WireConfig::glm53_flash();
        let min_rows: usize = match var("GLM53F_ROW_SHARDED_MIN_ROWS") {
            Some(v) => v.parse().map_err(|_| format!("GLM53F_ROW_SHARDED_MIN_ROWS={v}: not a row count"))?,
            None => 0,
        };
        if min_rows > 0 {
            let exchange = match var("GLM53F_EXCHANGE_DTYPE") {
                Some(v) => ExchangeDtype::parse(&v).ok_or(format!("GLM53F_EXCHANGE_DTYPE={v}: bf16 or fp8"))?,
                None => ExchangeDtype::Bf16,
            };
            cfg.return_path = ReturnPath::RowSharded { min_rows, exchange };
        }
        Ok(cfg)
    }

    /// The exchange dtype a `tokens`-row exchange is reduce-scattered with, or `None` for a
    /// four-plane return.
    pub fn row_sharded(&self, tokens: usize) -> Option<ExchangeDtype> {
        match self.return_path {
            ReturnPath::RowSharded { min_rows, exchange } if tokens >= min_rows.max(SPARKS) => Some(exchange),
            _ => None,
        }
    }
}

/// The planes one collected four-plane exchange left in the receive buffers
/// ([`WireClient::returned`]).
pub enum Returned<'a> {
    /// Rank `r`'s BF16 partial plane `[tokens * HIDDEN]` (little-endian bytes), ranks 0..3.
    Planes([&'a [u8]; SPARKS]),
}

/// One rank's part of a reduce-scattered exchange: rows `first..first + rows` of the routed
/// output, already summed over the four ranks, BF16 (little-endian bytes, `rows * HIDDEN * 2`).
#[derive(Clone, Copy, Debug)]
pub struct RowSlice<'a> {
    pub first: usize,
    pub rows: usize,
    pub bytes: &'a [u8],
}

/// The returns of the last collected exchange, as its return path laid them out
/// ([`WireClient::collected`]). They live in the receive buffers until the next send (an RDMA
/// slot is re-posted then), so a device caller copies them out first: four planes into its
/// staging buffer for the rank-order sum, or each row slice straight to row `first` of its
/// output (the slices cover the rows once and are final BF16 when the routed scale is 1).
pub enum Collected<'a> {
    /// A four-plane exchange: rank `r`'s BF16 partial plane `[tokens * HIDDEN]`, to be added in
    /// rank order (and multiplied by the routed scale).
    Planes([&'a [u8]; SPARKS]),
    /// A reduce-scattered exchange: rank `r`'s rows of the summed output; together they cover
    /// the rows once, in rank order (multiply by the routed scale, if it is not 1).
    RowSlices([RowSlice<'a>; SPARKS]),
}

/// An exchange sent and not yet collected.
#[derive(Clone, Copy, Debug)]
struct Inflight {
    request_id: u64,
    layer_id: u32,
    tokens: usize,
    /// Reduce-scattered (row-slice returns) rather than four planes.
    sharded: bool,
}

/// What the receive buffers hold after [`WireClient::moe_recv_raw`]: `(tokens, sharded)`.
type Last = (usize, bool);

/// The return a rank owes for an exchange: `(first row, rows)`, all rows for a four-plane one.
fn rank_rows(tokens: usize, rank: usize, sharded: bool) -> (usize, usize) {
    if sharded {
        row_partition(tokens, SPARKS, rank)
    } else {
        (0, tokens)
    }
}

/// A request-id base for a new connection: the ranks key their peer exchanges by request id and
/// layer, so ids must not repeat across connections (the time and the process id, mixed).
fn request_id_base() -> u64 {
    let t = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_nanos() as u64).unwrap_or(0);
    let mut z = t ^ (u64::from(std::process::id()) << 32) ^ 0x9E37_79B9_7F4A_7C15;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    // Keep far from wrapping around within a connection's lifetime.
    (z ^ (z >> 31)) >> 1
}

/// Wall-clock microsecond timestamp (CLOCK_REALTIME). Used by the cross-host
/// critical-path timeline (an offset-free four-timestamp method). The hosts are not
/// clock-synced, so only per-host deltas are meaningful:
/// The coordinator emits T1 (write start) and T4 (last return byte) per rank, and
/// the Spark emits T2/T3 from its own clock.
pub(crate) fn realtime_us() -> u128 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_micros())
        .unwrap_or(0)
}

/// Emit one `TL <event> [layer=<l>] [rank=<r>] us=<us>` line under
/// `GLM53F_TIMELINE=1`, for the cross-host timeline.
pub(crate) fn tl(event: &str, layer: Option<u32>, rank: Option<usize>) {
    if std::env::var_os("GLM53F_TIMELINE").is_some() {
        let us = realtime_us();
        match (layer, rank) {
            (Some(l), Some(r)) => eprintln!("TL {event} layer={l} rank={r} us={us}"),
            (Some(l), None) => eprintln!("TL {event} layer={l} us={us}"),
            (None, Some(r)) => eprintln!("TL {event} rank={r} us={us}"),
            (None, None) => eprintln!("TL {event} us={us}"),
        }
    }
}

/// `2^(127 - s)` for `s` in `0..=254` — the exact inverse of the UE8M0 scale
/// (a power of two, so the multiply by it is exact and bit-identical to the
/// division it replaces). Cached once.
fn scale_inv_table() -> &'static [f64] {
    use std::sync::OnceLock;
    static T: OnceLock<Vec<f64>> = OnceLock::new();
    T.get_or_init(|| (0..255u8).map(|s| 2f64.powi(127 - s as i32)).collect())
}

/// The wire-out activation quantizer (`Fp8E4m3Ue8m0K32`): one hidden row
/// `[HIDDEN]` f32 -> 4,096 E4M3 payload bytes + 128 UE8M0 K32 scale bytes
/// (one scale per 32 contiguous elements). The Spark decodes
/// `value[k] = decode_e4m3(payload[k]) * 2^(scales[k/32] - 127)`.
pub fn quantize_hidden(hidden: &[f32]) -> Result<HiddenRow, String> {
    if hidden.len() != HIDDEN {
        return Err(format!("hidden: {} elems, expected {HIDDEN}", hidden.len()));
    }
    const E4M3_MAX: f64 = crate::fp8::E4M3_MAX;
    let inv = scale_inv_table();
    let mut payload = vec![0u8; HIDDEN];
    let mut scales = vec![0u8; HIDDEN / 32];
    for b in 0..HIDDEN / 32 {
        let blk = &hidden[b * 32..(b + 1) * 32];
        let amax = blk.iter().fold(0.0f64, |m, &v| m.max((v as f64).abs()));
        // scale byte s: scale = 2^(s-127) >= amax/E4M3_MAX (round the exponent up).
        let s = if amax == 0.0 {
            0u8
        } else {
            let e = (amax / E4M3_MAX).log2().ceil() + 127.0;
            e.clamp(0.0, 254.0) as u8 // 255 reserved (T10)
        };
        scales[b] = s;
        let scale_inv = inv[s as usize]; // 2^(127-s), exact
        for (k, &v) in blk.iter().enumerate() {
            payload[b * 32 + k] = crate::fp8::encode_e4m3(v as f64 * scale_inv);
        }
    }
    Ok(HiddenRow { payload, scales })
}

/// Compute the per-K32-block scale bytes + the power-of-two inverse scales for the
/// whole hidden `[tokens * HIDDEN]` (CPU, bit-identical to [`quantize_hidden`]'s
/// scale step, and to the device's `glm53f_coord_quant_scales`).
pub fn quantize_hidden_scales(hidden: &[f32]) -> Result<(Vec<u8>, Vec<f32>), String> {
    if hidden.len() % HIDDEN != 0 {
        return Err(format!(
            "hidden: {} elems not a multiple of {HIDDEN}",
            hidden.len()
        ));
    }
    const E4M3_MAX: f64 = 448.0;
    let inv = scale_inv_table();
    let n_blocks = hidden.len() / 32;
    let mut scales = vec![0u8; n_blocks];
    let mut scale_inv = vec![0f32; n_blocks];
    for b in 0..n_blocks {
        let blk = &hidden[b * 32..(b + 1) * 32];
        let amax = blk.iter().fold(0.0f64, |m, &v| m.max((v as f64).abs()));
        let s = if amax == 0.0 {
            0u8
        } else {
            let e = (amax / E4M3_MAX).log2().ceil() + 127.0;
            e.clamp(0.0, 254.0) as u8
        };
        scales[b] = s;
        scale_inv[b] = inv[s as usize] as f32; // 2^(127-s), exact as f32
    }
    Ok((scales, scale_inv))
}

/// Quantize the whole hidden `[tokens * HIDDEN]` into per-token `HiddenRow`s on the
/// host (the device path is `glm53f_coord_quant_scales` + `glm53f_coord_quantize_hidden`,
/// bit-identical). Bit-identical to the per-token [`quantize_hidden`] loop.
pub fn quantize_hidden_batched(hidden: &[f32]) -> Result<Vec<HiddenRow>, String> {
    let tokens = hidden.len() / HIDDEN;
    if hidden.len() % HIDDEN != 0 {
        return Err(format!("hidden: {} elems not a multiple of {HIDDEN}", hidden.len()));
    }
    let (scales, scale_inv) = quantize_hidden_scales(hidden)?;
    let mut payload = vec![0u8; hidden.len()];
    for (i, &v) in hidden.iter().enumerate() {
        payload[i] = crate::fp8::encode_e4m3(v as f64 * f64::from(scale_inv[i >> 5]));
    }
    let mut rows = Vec::with_capacity(tokens);
    for t in 0..tokens {
        rows.push(HiddenRow {
            payload: payload[t * HIDDEN..(t + 1) * HIDDEN].to_vec(),
            scales: scales[t * (HIDDEN / 32)..(t + 1) * (HIDDEN / 32)].to_vec(),
        });
    }
    Ok(rows)
}

/// Largest request frame: 4,096 rows (the Spark B1 launch cap; perf reset P5,
/// was 2,048).
const REQ_MAX: usize = glm53f_wire::HEADER_LEN + 4096 * glm53f_wire::layout::REQUEST_ROW_BYTES;
/// One half of the double-buffered RDMA request body (perf reset P6), page-rounded.
const REQ_HALF: usize = REQ_MAX.div_ceil(4096) * 4096;
/// Largest return frame, rounded to a page: one RDMA receive slot.
const RET_SLOT: usize =
    (glm53f_wire::HEADER_LEN + 4096 * glm53f_wire::layout::RETURN_ROW_BYTES).div_ceil(4096) * 4096;

/// RDMA mode for one rank (perf reset R2 part 2): an RC queue pair with a
/// two-slot receive ring and a per-rank header buffer; the request body buffer is
/// shared by all ranks and owned by [`WireClient`].
struct RdmaConn {
    ep: glm53f_rdma::Endpoint,
    recv: glm53f_rdma::AlignedBuf,
    hdr: glm53f_rdma::AlignedBuf,
    /// Slots of returns already read, re-posted before the next request (two
    /// with the R4 two-lane prefill: both lanes' returns can be held at once).
    consumed: Vec<u32>,
    /// `(slot, len)` of the last return.
    last: Option<(u32, usize)>,
    /// Posted sends not yet reaped (perf reset P6: at most one per body half).
    sends_out: usize,
}

/// Check one return frame header against the request; returns its L4 sequence. A four-plane
/// exchange's return carries all `tokens` rows (version 3); a reduce-scattered exchange's
/// (`sharded`) carries the rank's partition, flagged as a row slice with `token_position` its
/// first row (version 4).
fn validate_return_header(h: &[u8], rank: usize, layer_id: u32, request_id: u64, tokens: usize, sharded: bool) -> Result<u64, String> {
    use glm53f_wire::frame::{FLAG_RETURN_REQUIRED, FLAG_ROW_SLICE};
    use glm53f_wire::layout::{hdr, KIND_RETURN, RETURN_ROW_BYTES, VERSION, VERSION_ROW_SHARD};
    let (first, count) = rank_rows(tokens, rank, sharded);
    let want = glm53f_wire::HEADER_LEN + count * RETURN_ROW_BYTES;
    let u16_at = |o: usize| u16::from_le_bytes([h[o], h[o + 1]]);
    let u32_at = |o: usize| u32::from_le_bytes(h[o..o + 4].try_into().unwrap());
    let u64_at = |o: usize| u64::from_le_bytes(h[o..o + 8].try_into().unwrap());
    if u32_at(hdr::STATUS) != 0 {
        return Err(format!("wire: Spark rank {rank} reported an error (layer {layer_id})"));
    }
    let (want_version, want_flags) =
        if sharded { (VERSION_ROW_SHARD, FLAG_RETURN_REQUIRED | FLAG_ROW_SLICE) } else { (VERSION, FLAG_RETURN_REQUIRED) };
    let (version, flags) = (u16_at(hdr::VERSION), u32_at(hdr::FLAGS) & (FLAG_RETURN_REQUIRED | FLAG_ROW_SLICE));
    let (kind, req, layer, rows, stride, dtype, exec, pos, wb) = (
        u16_at(hdr::KIND),
        u64_at(hdr::REQUEST_ID),
        u32_at(hdr::LAYER_ID),
        u32_at(hdr::ROW_COUNT) as usize,
        u32_at(hdr::ROW_STRIDE_BYTES) as usize,
        u16_at(hdr::PAYLOAD_DTYPE),
        u64_at(hdr::EXECUTOR_ID),
        u64_at(hdr::TOKEN_POSITION),
        u64_at(hdr::WIRE_BYTES) as usize,
    );
    if kind != KIND_RETURN || version != want_version || flags != want_flags || req != request_id || layer != layer_id
        || rows != count || stride != RETURN_ROW_BYTES || dtype != glm53f_wire::layout::Dtype::Bf16 as u16
        || exec != rank as u64 || pos != first as u64 || wb != want
    {
        return Err(format!(
            "wire: rank {rank} return header mismatch (kind {kind} version {version}/{want_version} flags {flags:#x}/{want_flags:#x} \
             req {req}/{request_id} layer {layer}/{layer_id} rows {rows}/{count} stride {stride} dtype {dtype} exec {exec} \
             pos {pos}/{first} bytes {wb}/{want})"
        ));
    }
    Ok(u64_at(hdr::SEQ))
}

/// One Spark connection: a TCP stream plus its per-connection L4 sequence state.
struct SparkConn {
    stream: TcpStream,
    tx: StreamSender,
    rx: StreamReceiver,
    rank: usize,
    /// Reused receive buffer of the zero-copy return path (perf reset R2).
    buf: Vec<u8>,
    /// RDMA mode (perf reset R2 part 2); `None` = TCP.
    rdma: Option<RdmaConn>,
}

impl SparkConn {
    fn connect(rank: usize, addr: &str) -> std::io::Result<Self> {
        let stream = TcpStream::connect(addr)?;
        // Inference runs only on the RDMA fabric, never the LAN/10G (API) path.
        match glm53f_rdma::fabric_port(stream.local_addr()?.ip()) {
            Ok(Some((dev, port, _, gbps))) => eprintln!("[wire] rank {rank} {addr}: fabric {dev} port {port} at {gbps} Gb/s"),
            Ok(None) => eprintln!("[wire] rank {rank} {addr}: loopback or GLM53F_WIRE_ALLOW_LAN=1, fabric check skipped"),
            Err(e) => return Err(std::io::Error::new(std::io::ErrorKind::PermissionDenied, format!("rank {rank} {addr}: {e}"))),
        }
        stream.set_nodelay(true)?;
        stream.set_nonblocking(true)?;
        Ok(Self {
            stream,
            tx: StreamSender::new(WireNaive::NONE),
            rx: StreamReceiver::new(WireNaive::NONE),
            rank,
            buf: Vec::new(),
            rdma: None,
        })
    }

    /// Serialize one request frame (advances this connection's L4 sequence).
    fn encode(&mut self, frame: &RequestFrame) -> Result<Vec<u8>, String> {
        self.tx.encode_request(frame).map_err(|e| e.to_string())
    }

    /// Blocking one-shot write of the serialized frame (the stream is non-blocking
    /// for recv; a 17.6 MB prefill request exceeds the socket send buffer). Under
    /// `GLM53F_PROFILE`, reports the number of write syscalls and total bytes so a
    /// frame split into many small writes (e.g. row-by-row) is visible.
    fn write_blocking(&mut self, layer_id: u32, bytes: &[u8]) -> Result<(), String> {
        self.stream
            .set_nonblocking(false)
            .map_err(|e| format!("spark set_blocking: {e}"))?;
        tl("wr_start", Some(layer_id), Some(self.rank));
        let mut written = 0usize;
        let mut n_write = 0usize;
        let mut last_err = None;
        while written < bytes.len() {
            match self.stream.write(&bytes[written..]) {
                Ok(0) => {
                    last_err = Some("spark write: zero-length write".to_string());
                    break;
                }
                Ok(n) => {
                    written += n;
                    n_write += 1;
                }
                Err(e) if e.kind() == std::io::ErrorKind::Interrupted => continue,
                Err(e) => {
                    last_err = Some(format!("spark write: {e}"));
                    break;
                }
            }
        }
        tl("wr_end", Some(layer_id), Some(self.rank));
        let _ = self.stream.set_nonblocking(true);
        if std::env::var_os("GLM53F_PROFILE").is_some() {
            eprintln!("PROFILE wire_write syscalls={n_write} bytes={written}");
        }
        match last_err {
            Some(e) => Err(e),
            None => Ok(()),
        }
    }

    /// Blocking read of one full return frame (header then body), used by the
    /// per-connection receive thread. A read timeout bounds a dropped rank.
    fn recv_frame_blocking(&mut self, layer_id: u32) -> Result<ReturnFrame, String> {
        self.stream
            .set_nonblocking(false)
            .map_err(|e| format!("spark set_blocking: {e}"))?;
        self.stream
            .set_read_timeout(Some(std::time::Duration::from_secs(120)))
            .map_err(|e| format!("spark set_read_timeout: {e}"))?;
        let r = self.recv_frame_full(layer_id);
        let _ = self.stream.set_read_timeout(None);
        let _ = self.stream.set_nonblocking(true);
        r
    }

    fn recv_frame_full(&mut self, layer_id: u32) -> Result<ReturnFrame, String> {
        // Under `GLM53F_PROFILE`, split the receive into the network wait
        // (`wire_recv`: header + body reads) and the frame decode
        // (`wire_deserialize`: `accept`/CRC verify). The parallel prefill prints
        // one pair per rank, so the slowest rank is visible alongside
        // `wire_collect` in `moe_layer`.
        let prof = std::env::var_os("GLM53F_PROFILE").is_some();
        let t_recv = std::time::Instant::now();
        let mut header = [0u8; glm53f_wire::HEADER_LEN];
        self.stream
            .read_exact(&mut header)
            .map_err(|e| format!("spark read header: {e}"))?;
        tl("first_byte", Some(layer_id), Some(self.rank));
        let wb = u64::from_le_bytes(
            header[76..84]
                .try_into()
                .map_err(|_| "wire: short header".to_string())?,
        ) as usize;
        if wb < glm53f_wire::HEADER_LEN {
            return Err(format!("wire: bad frame length {wb}"));
        }
        let mut bytes = vec![0u8; wb];
        bytes[..glm53f_wire::HEADER_LEN].copy_from_slice(&header);
        self.stream
            .read_exact(&mut bytes[glm53f_wire::HEADER_LEN..])
            .map_err(|e| format!("spark read body: {e}"))?;
        tl("last_byte", Some(layer_id), Some(self.rank));
        let t_decode = std::time::Instant::now();
        if prof {
            eprintln!(
                "PROFILE wire_recv {:.3}",
                (t_decode - t_recv).as_secs_f64() * 1e3
            );
        }
        let r = match self.rx.accept(&bytes).map_err(|e| e.to_string())? {
            Frame::Return(r) => Ok(r),
            Frame::Request(_) => Err("wire: unexpected request frame from Spark".to_string()),
        };
        tl("deser_done", Some(layer_id), Some(self.rank));
        if prof {
            eprintln!(
                "PROFILE wire_deserialize {:.3}",
                t_decode.elapsed().as_secs_f64() * 1e3
            );
        }
        r
    }
}

impl SparkConn {
    /// Zero-copy return receive (perf reset R2): read one frame into `buf`,
    /// validate the header against the request (kind, version and flags for the
    /// exchange's return path, request, layer, rank, rows and first row, BF16
    /// compact stride, status) and the L4 sequence, without decoding rows.
    /// With CRC on (the default) the frame also takes the full `decode_frame`
    /// check, so the fast path only ever skips work `GLM53F_WIRE_NOCRC=1` waived.
    fn recv_raw_blocking(&mut self, layer_id: u32, request_id: u64, tokens: usize, sharded: bool) -> Result<(), String> {
        self.stream.set_nonblocking(false).map_err(|e| format!("spark set_blocking: {e}"))?;
        self.stream
            .set_read_timeout(Some(std::time::Duration::from_secs(120)))
            .map_err(|e| format!("spark set_read_timeout: {e}"))?;
        let rows = rank_rows(tokens, self.rank, sharded).1;
        let want = glm53f_wire::HEADER_LEN + rows * glm53f_wire::layout::RETURN_ROW_BYTES;
        if self.buf.len() < want {
            self.buf.resize(want, 0);
        }
        let r = (|| {
            self.stream
                .read_exact(&mut self.buf[..glm53f_wire::HEADER_LEN])
                .map_err(|e| format!("spark read header: {e}"))?;
            let h = &self.buf[..glm53f_wire::HEADER_LEN];
            let seq = validate_return_header(h, self.rank, layer_id, request_id, tokens, sharded)?;
            self.stream
                .read_exact(&mut self.buf[glm53f_wire::HEADER_LEN..want])
                .map_err(|e| format!("spark read body: {e}"))?;
            if glm53f_wire::frame::crc_disabled() {
                self.rx.accept_seq(seq).map_err(|e| e.to_string())
            } else {
                match self.rx.accept(&self.buf[..want]).map_err(|e| e.to_string())? {
                    Frame::Return(_) => Ok(()),
                    Frame::Request(_) => Err("wire: unexpected request frame from Spark".to_string()),
                }
            }
        })();
        let _ = self.stream.set_read_timeout(None);
        let _ = self.stream.set_nonblocking(true);
        r
    }

    /// Switch this connection to RDMA: open an RC queue pair on the RoCE device
    /// that owns the TCP socket's local IPv4, exchange queue-pair coordinates over
    /// the socket, connect, and pre-post both receive slots.
    fn rdma_setup(&mut self, body: &mut glm53f_rdma::AlignedBuf) -> Result<(), String> {
        use glm53f_rdma::{AlignedBuf, Endpoint, Info, HANDSHAKE_LEN, HANDSHAKE_MAGIC};
        let ip = match self.stream.local_addr().map_err(|e| e.to_string())? {
            std::net::SocketAddr::V4(a) => *a.ip(),
            a => return Err(format!("rdma: IPv6 local address {a} unsupported")),
        };
        let (dev, port, gid, _) = match glm53f_rdma::fabric_port(std::net::IpAddr::V4(ip))? {
            Some(f) => f,
            None => {
                let (d, p, g) = glm53f_rdma::find_roce_v2(ip).ok_or(format!("rdma: no RoCE v2 GID for {ip}"))?;
                (d, p, g, 0)
            }
        };
        let mut recv = AlignedBuf::new(2 * RET_SLOT);
        let mut hdr = AlignedBuf::new(4096);
        let mut ep = Endpoint::open(&dev, port, gid, Some(body), &mut recv, 2, Some(&mut hdr))?;
        self.stream.set_nonblocking(false).map_err(|e| e.to_string())?;
        let mut msg = Vec::with_capacity(HANDSHAKE_LEN);
        msg.extend_from_slice(HANDSHAKE_MAGIC);
        msg.extend_from_slice(&ep.local_info().to_bytes());
        self.stream.write_all(&msg).map_err(|e| format!("rdma handshake write: {e}"))?;
        let mut reply = [0u8; HANDSHAKE_LEN];
        self.stream.read_exact(&mut reply).map_err(|e| format!("rdma handshake read: {e}"))?;
        if &reply[..8] != HANDSHAKE_MAGIC {
            return Err(format!("rdma: rank {} answered without the RDMA magic (TCP-only daemon?)", self.rank));
        }
        let remote = Info::from_bytes(&reply[8..]).ok_or("rdma: short handshake")?;
        ep.connect(&remote)?;
        ep.post_recv(0)?;
        ep.post_recv(1)?;
        self.stream.set_nonblocking(true).map_err(|e| e.to_string())?;
        eprintln!("[wire] rank {} RDMA RC on {dev} port {port} gid {gid} (qpn {} -> {})", self.rank,
            ep.local_info().qpn, remote.qpn);
        self.rdma = Some(RdmaConn { ep, recv, hdr, consumed: Vec::new(), last: None, sends_out: 0 });
        Ok(())
    }

    /// RDMA receive of one return: busy-poll the completion, validate the header.
    fn recv_rdma(&mut self, layer_id: u32, request_id: u64, tokens: usize, sharded: bool) -> Result<(), String> {
        let rank = self.rank;
        let rc = self.rdma.as_mut().ok_or("rdma: not set up")?;
        let (slot, len) = rc
            .ep
            .wait_recv(std::time::Duration::from_millis(20), Some(std::time::Duration::from_secs(120)))?
            .ok_or_else(|| format!("rdma: rank {rank} return timed out (layer {layer_id})"))?;
        rc.consumed.push(slot);
        let base = slot as usize * rc.ep.slot_len();
        let frame = &rc.recv.as_slice()[base..base + len];
        if len < glm53f_wire::HEADER_LEN {
            return Err(format!("rdma: short frame {len} from rank {rank}"));
        }
        let seq = validate_return_header(&frame[..glm53f_wire::HEADER_LEN], rank, layer_id, request_id, tokens, sharded)?;
        let rows = rank_rows(tokens, rank, sharded).1;
        if len != glm53f_wire::HEADER_LEN + rows * glm53f_wire::layout::RETURN_ROW_BYTES {
            return Err(format!("rdma: rank {rank} frame length {len}"));
        }
        rc.last = Some((slot, len));
        self.rx.accept_seq(seq).map_err(|e| e.to_string())
    }
}

/// The coordinator's four-Spark expert client (synchronous).
pub struct WireClient {
    conns: Vec<SparkConn>, // index = executor_id (rank)
    next_request_id: u64,
    /// RDMA mode: the request body shared by all ranks (dropped after `conns`,
    /// whose endpoints hold its registration).
    rdma_body: Option<glm53f_rdma::AlignedBuf>,
    /// Sent, not yet collected exchanges, oldest first (perf reset R4: two lanes in
    /// flight over RDMA). Each keeps its return path: the two lanes' exchanges may differ.
    inflight: std::collections::VecDeque<Inflight>,
    /// The exchange whose returns the receive buffers hold (the last one collected).
    last: Option<Last>,
    /// The body half the next RDMA request is built in (perf reset P6).
    send_half: usize,
    cfg: WireConfig,
}

impl WireClient {
    /// Connect to `SPARKS` expert ranks (rank `i` at `addrs[i]`).
    pub fn connect(addrs: &[String], cfg: WireConfig) -> Result<Self, String> {
        if addrs.len() != SPARKS {
            return Err(format!("wire: need {SPARKS} Spark addrs, got {}", addrs.len()));
        }
        if let ReturnPath::RowSharded { min_rows, .. } = cfg.return_path {
            if min_rows < SPARKS {
                return Err(format!("wire: the row-sharded return needs at least {SPARKS} rows (one per rank), not {min_rows}"));
            }
        }
        if cfg.experts == 0 || !cfg.routed_scale.is_finite() {
            return Err(format!("wire: bad configuration {cfg:?}"));
        }
        let mut conns = Vec::with_capacity(SPARKS);
        for (rank, a) in addrs.iter().enumerate() {
            conns.push(SparkConn::connect(rank, a).map_err(|e| format!("wire connect {a}: {e}"))?);
        }
        let mut rdma_body = None;
        if std::env::var("GLM53F_RDMA").map(|v| v == "1").unwrap_or(false) {
            if !glm53f_wire::frame::crc_disabled() {
                return Err("GLM53F_RDMA=1 needs GLM53F_WIRE_NOCRC=1 (one request body is shared by all ranks)".into());
            }
            // Two halves (perf reset P6): a request is posted without waiting
            // for its send completion; a half is reused only once reaped.
            let mut body = glm53f_rdma::AlignedBuf::new(2 * REQ_HALF);
            for c in conns.iter_mut() {
                c.rdma_setup(&mut body)?;
            }
            rdma_body = Some(body);
        }
        Ok(Self {
            conns,
            next_request_id: request_id_base(),
            rdma_body,
            inflight: Default::default(),
            last: None,
            send_half: 0,
            cfg,
        })
    }

    /// The request flags of a `tokens`-row exchange: a reduce-scatter with the configured
    /// exchange dtype from the row-sharded threshold, none below it.
    fn request_flags(&self, tokens: usize) -> u32 {
        self.cfg.row_sharded(tokens).map_or(0, |d| d.request_flags())
    }

    pub fn config(&self) -> &WireConfig {
        &self.cfg
    }

    /// The factor the ranks' summed partials are multiplied by (pass it to the device rank sum).
    pub fn routed_scale(&self) -> f32 {
        self.cfg.routed_scale
    }

    /// Refuse routes that are not whole top-`topk` rows or name an expert past the model's.
    fn check_routes(&self, routes: &[(u32, f32)], topk: usize) -> Result<(), String> {
        if topk == 0 || routes.is_empty() || routes.len() % topk != 0 {
            return Err(format!("wire: {} routes do not form whole top-{topk} rows", routes.len()));
        }
        if let Some(&(e, _)) = routes.iter().find(|&&(e, _)| e >= self.cfg.experts) {
            return Err(format!("wire: expert id {e} past the model's {} experts", self.cfg.experts));
        }
        Ok(())
    }

    /// The RDMA request body (both halves), for page-locking so the GPU can copy
    /// hidden rows straight into it ([`WireClient::moe_send_mapped`]).
    pub fn send_buffers(&mut self) -> Vec<(*mut u8, usize)> {
        self.rdma_body.as_mut().map(|b| vec![(b.as_mut_slice().as_mut_ptr(), b.len())]).unwrap_or_default()
    }

    /// Take the next request half, first reaping every rank's send that still
    /// reads it (RC completions are in order: with two halves, at most one older
    /// send per rank may stay outstanding).
    fn claim_half(&mut self) -> Result<usize, String> {
        let half = self.send_half;
        self.send_half ^= 1;
        for conn in self.conns.iter_mut() {
            let rc = conn.rdma.as_mut().ok_or("rdma: connection not set up")?;
            while rc.sends_out >= 2 {
                rc.ep.wait_send(Some(std::time::Duration::from_secs(30)))?;
                rc.sends_out -= 1;
            }
        }
        Ok(half)
    }

    /// Post the request in body half `half` (`blen` bytes after its header) to
    /// every rank: rank r's header copy carries executor id r and that
    /// connection's next L4 sequence. Returns without waiting for completions.
    fn post_half(&mut self, half: usize, header: &[u8], blen: usize) -> Result<(), String> {
        use glm53f_wire::layout::hdr;
        let h = glm53f_wire::HEADER_LEN;
        for (r, conn) in self.conns.iter_mut().enumerate() {
            let seq = conn.tx.take_seq();
            let rc = conn.rdma.as_mut().ok_or("rdma: connection not set up")?;
            let hb = &mut rc.hdr.as_mut_slice()[half * h..(half + 1) * h];
            hb.copy_from_slice(&header[..h]);
            hb[hdr::EXECUTOR_ID..hdr::EXECUTOR_ID + 8].copy_from_slice(&(r as u64).to_le_bytes());
            hb[hdr::SEQ..hdr::SEQ + 8].copy_from_slice(&seq.to_le_bytes());
            for slot in rc.consumed.drain(..) {
                rc.ep.post_recv(slot)?;
            }
            rc.ep.post_send(Some((half * h, h)), Some((half * REQ_HALF, blen)))?;
            rc.sends_out += 1;
        }
        Ok(())
    }

    /// RDMA fast path, device-filled (perf reset P9): as [`WireClient::moe_send_mapped`],
    /// but `fill(routes_dst, hidden_dst, pitch)` writes the `tokens * topk` 12-B
    /// route entries as well as the hidden rows, so the router output never
    /// comes to the host (`glm53f_coord_frame_fill`). Only the row descriptors and
    /// headers are host-written. The router must produce expert ids below
    /// [`WireConfig::experts`]; nothing on this path can check them.
    pub fn moe_send_device(
        &mut self,
        layer_id: u32,
        tokens: usize,
        topk: usize,
        fill: impl FnOnce(*mut u8, *mut u8, usize) -> Result<(), String>,
    ) -> Result<(), String> {
        if self.rdma_body.is_none() {
            return Err("wire: moe_send_device needs the RDMA transport".into());
        }
        if self.inflight.len() >= 2 {
            return Err(format!("wire: {} exchanges already in flight (limit 2)", self.inflight.len()));
        }
        let request_id = self.next_request_id;
        self.next_request_id += 1;
        let naive = self.conns[0].tx.naive();
        let flags = self.request_flags(tokens);
        // Sending re-posts the receive slots the last returns lie in.
        self.last = None;
        let half = self.claim_half()?;
        let body = self.rdma_body.as_mut().expect("rdma body");
        let dst = &mut body.as_mut_slice()[half * REQ_HALF..(half + 1) * REQ_HALF];
        let (mut header, routes_off, hidden_off, blen) =
            glm53f_wire::frame::encode_request_desc_into(dst, request_id, layer_id, 0, 0, tokens, topk, naive)
                .map_err(|e| format!("wire: {e}"))?;
        glm53f_wire::frame::set_request_flags(&mut header, flags).map_err(|e| format!("wire: {e}"))?;
        let base = dst.as_mut_ptr();
        // SAFETY: both offsets lie inside `dst` (checked by the encoder against blen).
        let (routes, hidden) = unsafe { (base.add(routes_off), base.add(hidden_off)) };
        fill(routes, hidden, glm53f_wire::layout::HIDDEN_ROW_BYTES)?;
        self.post_half(half, &header, blen)?;
        self.inflight.push_back(Inflight { request_id, layer_id, tokens, sharded: flags != 0 });
        Ok(())
    }

    /// RDMA fast path (perf reset P6): send one MoE exchange whose hidden rows
    /// are written straight into the registered request body. The descriptors
    /// and routes are encoded in place; `fill(dst, pitch)` must then write the
    /// `routes.len() / topk` hidden rows (payload then scales, `pitch` = 4,224 B
    /// per row) at `dst`, e.g. a device-to-host copy from the GPU. The frame is
    /// byte-identical to [`WireClient::moe_send_raw`]'s. Collect it with
    /// [`WireClient::moe_recv_raw`].
    pub fn moe_send_mapped(
        &mut self,
        layer_id: u32,
        routes: &[(u32, f32)],
        topk: usize,
        fill: impl FnOnce(*mut u8, usize) -> Result<(), String>,
    ) -> Result<(), String> {
        if self.rdma_body.is_none() {
            return Err("wire: moe_send_mapped needs the RDMA transport".into());
        }
        self.check_routes(routes, topk)?;
        let tokens = routes.len() / topk;
        if self.inflight.len() >= 2 {
            return Err(format!("wire: {} exchanges already in flight (limit 2)", self.inflight.len()));
        }
        let request_id = self.next_request_id;
        self.next_request_id += 1;
        let naive = self.conns[0].tx.naive();
        let flags = self.request_flags(tokens);
        // Sending re-posts the receive slots the last returns lie in.
        self.last = None;
        let half = self.claim_half()?;
        let body = self.rdma_body.as_mut().expect("rdma body");
        let dst = &mut body.as_mut_slice()[half * REQ_HALF..(half + 1) * REQ_HALF];
        let (mut header, hidden_off, blen) =
            glm53f_wire::frame::encode_request_meta_into(dst, request_id, layer_id, 0, 0, routes, topk, naive)
                .map_err(|e| format!("wire: {e}"))?;
        glm53f_wire::frame::set_request_flags(&mut header, flags).map_err(|e| format!("wire: {e}"))?;
        if hidden_off + tokens * glm53f_wire::layout::HIDDEN_ROW_BYTES != blen {
            return Err(format!("wire: mapped frame layout {hidden_off} + {tokens} rows != {blen}"));
        }
        fill(dst[hidden_off..].as_mut_ptr(), glm53f_wire::layout::HIDDEN_ROW_BYTES)?;
        self.post_half(half, &header, blen)?;
        self.inflight.push_back(Inflight { request_id, layer_id, tokens, sharded: flags != 0 });
        Ok(())
    }

    /// One MoE layer for `tokens` token rows. `hidden` is `[tokens, HIDDEN]`;
    /// `routes` is `tokens * topk` `(expert_id, gate_weight)` pairs, token-major
    /// then route order (the router's top-k output). Returns the FP32 rank sum
    /// `[tokens, HIDDEN]` (the R8 CoordinatorSum of the 4 Spark partials).
    pub fn moe_layer(
        &mut self,
        layer_id: u32,
        hidden: &[f32],
        routes: &[(u32, f32)],
        topk: usize,
    ) -> Result<Vec<f32>, String> {
        let tokens = hidden.len() / HIDDEN;
        if hidden.len() != tokens * HIDDEN || tokens == 0 {
            return Err(format!(
                "wire: hidden {} elems is not a positive multiple of {HIDDEN}",
                hidden.len()
            ));
        }
        let hidden_rows = quantize_hidden_batched(hidden)?;
        self.exchange(layer_id, tokens, hidden_rows, routes, topk)
    }

    /// [`moe_layer`] for rows already quantized on the device (perf reset R1):
    /// `payload` is `[tokens * HIDDEN]` E4M3 and `scales` `[tokens * HIDDEN / 32]`
    /// UE8M0, bit-identical to [`quantize_hidden_batched`].
    pub fn moe_layer_prequant(
        &mut self,
        layer_id: u32,
        payload: &[u8],
        scales: &[u8],
        routes: &[(u32, f32)],
        topk: usize,
    ) -> Result<Vec<f32>, String> {
        let tokens = payload.len() / HIDDEN;
        if payload.len() != tokens * HIDDEN || tokens == 0 || scales.len() != tokens * (HIDDEN / 32) {
            return Err(format!(
                "wire: prequant payload {} / scales {} bytes do not form whole {HIDDEN}-rows",
                payload.len(),
                scales.len()
            ));
        }
        let hidden_rows = (0..tokens)
            .map(|t| HiddenRow {
                payload: payload[t * HIDDEN..(t + 1) * HIDDEN].to_vec(),
                scales: scales[t * (HIDDEN / 32)..(t + 1) * (HIDDEN / 32)].to_vec(),
            })
            .collect();
        self.exchange(layer_id, tokens, hidden_rows, routes, topk)
    }

    /// [`moe_layer_prequant`] without the CPU sum (perf reset R2): the four ranks'
    /// BF16 return planes stay in the per-connection receive buffers, read them
    /// with [`WireClient::rank_plane`] (rank order 0..3) and sum on the device.
    /// Returns the token count.
    pub fn moe_layer_prequant_raw(
        &mut self,
        layer_id: u32,
        payload: &[u8],
        scales: &[u8],
        routes: &[(u32, f32)],
        topk: usize,
    ) -> Result<usize, String> {
        self.moe_send_raw(layer_id, payload, scales, routes, topk)?;
        self.moe_recv_raw().map(|(_, tokens)| tokens)
    }

    /// Whether more than one exchange may be in flight (perf reset R4). Only the
    /// RDMA transport: the Spark pre-posts two receive slots, so a second request
    /// lands while the first is computed. Over TCP a second multi-MB write can
    /// deadlock against the first return (both sides blocked in `write`).
    pub fn pipelined(&self) -> bool {
        self.rdma_body.is_some()
    }

    /// Send half of [`moe_layer_prequant_raw`]: quantized rows to the four ranks.
    /// Collect with [`WireClient::moe_recv_raw`], oldest first.
    pub fn moe_send_raw(
        &mut self,
        layer_id: u32,
        payload: &[u8],
        scales: &[u8],
        routes: &[(u32, f32)],
        topk: usize,
    ) -> Result<(), String> {
        let tokens = payload.len() / HIDDEN;
        if payload.len() != tokens * HIDDEN || tokens == 0 || scales.len() != tokens * (HIDDEN / 32) {
            return Err(format!(
                "wire: prequant payload {} / scales {} bytes do not form whole {HIDDEN}-rows",
                payload.len(),
                scales.len()
            ));
        }
        let limit = if self.pipelined() { 2 } else { 1 };
        if self.inflight.len() >= limit {
            return Err(format!("wire: {} exchanges already in flight (limit {limit})", self.inflight.len()));
        }
        let hidden_rows = (0..tokens)
            .map(|t| HiddenRow {
                payload: payload[t * HIDDEN..(t + 1) * HIDDEN].to_vec(),
                scales: scales[t * (HIDDEN / 32)..(t + 1) * (HIDDEN / 32)].to_vec(),
            })
            .collect();
        let request_id = self.send_layer(layer_id, tokens, hidden_rows, routes, topk)?;
        self.inflight.push_back(Inflight { request_id, layer_id, tokens, sharded: self.request_flags(tokens) != 0 });
        Ok(())
    }

    /// Receive half: the four returns of the oldest in-flight exchange, left in
    /// place for [`WireClient::collected`] (or [`WireClient::returned`] and
    /// [`WireClient::rank_plane`] for a four-plane exchange) until the next send.
    /// Returns `(layer_id, tokens)` of the exchange collected.
    pub fn moe_recv_raw(&mut self) -> Result<(u32, usize), String> {
        let Inflight { request_id, layer_id, tokens, sharded } = self.inflight.pop_front().ok_or("wire: no exchange in flight")?;
        self.last = None;
        let prof = std::env::var_os("GLM53F_PROFILE").is_some();
        let tc = std::time::Instant::now();
        if self.rdma_body.is_some() {
            // All four transfers proceed in hardware; poll each completion in turn.
            for conn in self.conns.iter_mut() {
                conn.recv_rdma(layer_id, request_id, tokens, sharded)?;
            }
        } else if tokens > 1 {
            let conns = std::mem::take(&mut self.conns);
            let handles: Vec<_> = conns
                .into_iter()
                .map(|mut conn| {
                    std::thread::spawn(move || {
                        let r = conn.recv_raw_blocking(layer_id, request_id, tokens, sharded);
                        (conn, r)
                    })
                })
                .collect();
            let mut new_conns = Vec::with_capacity(SPARKS);
            let mut first_err = None;
            for h in handles {
                let (conn, r) = h.join().map_err(|_| "wire recv thread panic".to_string())?;
                if let Err(e) = r {
                    first_err.get_or_insert(e);
                }
                new_conns.push(conn);
            }
            self.conns = new_conns;
            if let Some(e) = first_err {
                return Err(e);
            }
        } else {
            for conn in self.conns.iter_mut() {
                conn.recv_raw_blocking(layer_id, request_id, tokens, sharded)?;
            }
        }
        self.last = Some((tokens, sharded));
        tl("sum_done", Some(layer_id), None);
        if prof {
            eprintln!("PROFILE wire_collect_raw {:.3}", tc.elapsed().as_secs_f64() * 1e3);
        }
        Ok((layer_id, tokens))
    }

    /// [`WireClient::moe_recv_raw`] with work to overlap: `during` runs first, while the ranks
    /// compute the oldest exchange (the shared expert's MLP runs here, as the design has it), then
    /// the returns are collected. An error from `during` leaves the exchange in flight.
    pub fn moe_recv_during(&mut self, during: impl FnOnce() -> Result<(), String>) -> Result<(u32, usize), String> {
        if self.inflight.is_empty() {
            return Err("wire: no exchange in flight".into());
        }
        during()?;
        self.moe_recv_raw()
    }

    /// The last collected four-plane exchange's planes (the layout of [`ReturnPath::FourPlaneSum`],
    /// which every exchange below the row-sharded threshold keeps). [`WireClient::collected`]
    /// covers both layouts; this panics after a reduce-scattered exchange.
    pub fn returned(&self, tokens: usize) -> Returned<'_> {
        assert!(
            !matches!(self.last, Some((_, true))),
            "wire: the last exchange was reduce-scattered; read its row slices with collected()"
        );
        Returned::Planes(std::array::from_fn(|r| self.rank_plane(r, tokens)))
    }

    /// The last collected exchange's returns, in the layout its return path gave them: four
    /// planes to add, or four row slices that cover the rows once. `None` unless the last call
    /// was a [`WireClient::moe_recv_raw`] (a send re-posts the buffers; the host-sum paths leave
    /// nothing here).
    pub fn collected(&self) -> Option<Collected<'_>> {
        let (tokens, sharded) = self.last?;
        Some(if sharded {
            Collected::RowSlices(std::array::from_fn(|r| {
                let (first, rows) = row_partition(tokens, SPARKS, r);
                RowSlice { first, rows, bytes: self.return_rows(r, rows) }
            }))
        } else {
            Collected::Planes(std::array::from_fn(|r| self.rank_plane(r, tokens)))
        })
    }

    /// Rank `rank`'s BF16 return plane `[tokens * HIDDEN]` (LE bytes) from the
    /// last [`WireClient::moe_recv_raw`] of a four-plane exchange.
    pub fn rank_plane(&self, rank: usize, tokens: usize) -> &[u8] {
        self.return_rows(rank, tokens)
    }

    /// The first `rows` BF16 rows of rank `rank`'s last return, where they landed.
    fn return_rows(&self, rank: usize, rows: usize) -> &[u8] {
        let plane = rows * glm53f_wire::layout::RETURN_ROW_BYTES;
        let c = &self.conns[rank];
        if let Some(rc) = c.rdma.as_ref() {
            let (slot, _) = rc.last.expect("rank_plane before an RDMA return");
            let base = slot as usize * rc.ep.slot_len() + glm53f_wire::HEADER_LEN;
            return &rc.recv.as_slice()[base..base + plane];
        }
        &c.buf[glm53f_wire::HEADER_LEN..glm53f_wire::HEADER_LEN + plane]
    }

    /// The host receive buffers the return planes land in (the RDMA rings), so
    /// the caller can page-lock them for fast device uploads.
    pub fn plane_buffers(&self) -> Vec<(*mut u8, usize)> {
        self.conns.iter().filter_map(|c| c.rdma.as_ref().map(|r| (r.recv.as_ptr(), r.recv.len()))).collect()
    }

    /// Build, encode and write one quantized MoE layer to the four ranks;
    /// returns the request id the returns must carry.
    fn send_layer(
        &mut self,
        layer_id: u32,
        tokens: usize,
        hidden_rows: Vec<HiddenRow>,
        routes: &[(u32, f32)],
        topk: usize,
    ) -> Result<u64, String> {
        if routes.len() != tokens * topk {
            return Err(format!(
                "wire: {} routes for {tokens} tokens x topk {topk}",
                routes.len()
            ));
        }
        self.check_routes(routes, topk)?;

        let request_id = self.next_request_id;
        self.next_request_id += 1;
        // A reduce-scatter from the row-sharded threshold on (the same flags for every rank).
        let flags = self.request_flags(tokens);
        // Sending re-posts the receive slots the last returns lie in (RDMA).
        self.last = None;

        let prof = std::env::var_os("GLM53F_PROFILE").is_some();
        let t0 = std::time::Instant::now();
        let mut rows = Vec::with_capacity(tokens);
        let mut route_entries = Vec::with_capacity(tokens * topk);
        for t in 0..tokens {
            rows.push(RowDescriptor {
                row_id: t as u64,
                source_kind: SourceKind::Decode,
                source_request_id: request_id,
                token_position: t as u64,
                route_offset: (t * topk) as u32,
                route_count: topk as u32,
            });
            for k in 0..topk {
                let (expert_id, gate_weight) = routes[t * topk + k];
                route_entries.push(RouteEntry {
                    row_index: t as u32,
                    expert_id,
                    gate_weight,
                });
            }
        }
        if prof { eprintln!("PROFILE wire_build {:.3}", t0.elapsed().as_secs_f64() * 1e3); }

        if self.rdma_body.is_some() {
            // RDMA: encode once, share the body (one of two halves, perf reset
            // P6), patch the 128-B header per rank (executor id + that
            // connection's sequence); no wait for the send completions.
            let ts = std::time::Instant::now();
            let frame = RequestFrame {
                request_id,
                placement_version: 1,
                layer_id,
                executor_id: 0,
                source_kind: SourceKind::Decode,
                token_position: 0,
                flags,
                seq: 0,
                rows,
                routes: route_entries,
                hidden_rows,
            };
            let naive = self.conns[0].tx.naive();
            let bytes = glm53f_wire::frame::encode_request_seq(&frame, 0, naive).map_err(|e| e.to_string())?;
            let h = glm53f_wire::HEADER_LEN;
            let blen = bytes.len() - h;
            if blen > REQ_HALF {
                return Err(format!("rdma: request body {blen} B exceeds the {REQ_HALF} B half"));
            }
            let half = self.claim_half()?;
            let body = self.rdma_body.as_mut().expect("rdma body");
            body.as_mut_slice()[half * REQ_HALF..half * REQ_HALF + blen].copy_from_slice(&bytes[h..]);
            let t_encode = std::time::Instant::now();
            self.post_half(half, &bytes[..h], blen)?;
            if prof {
                eprintln!(
                    "PROFILE wire_encode {:.3} wire_write {:.3}",
                    (t_encode - ts).as_secs_f64() * 1e3,
                    t_encode.elapsed().as_secs_f64() * 1e3
                );
            }
            return Ok(request_id);
        }

        // Serialize the four rank frames (sequential L4 stamp). The prefill
        // (large frames) writes/reads in parallel threads so all four drive the
        // link at once; the decode (1 token) stays sequential because the thread
        // spawn/join overhead would dominate the tiny frames.
        let parallel = tokens > 1;
        let ts = std::time::Instant::now();
        let mut encoded: Vec<Vec<u8>> = Vec::with_capacity(SPARKS);
        for (exec, conn) in self.conns.iter_mut().enumerate() {
            let frame = RequestFrame {
                request_id,
                placement_version: 1,
                layer_id,
                executor_id: exec as u64,
                source_kind: SourceKind::Decode,
                token_position: 0,
                flags,
                seq: 0,
                rows: rows.clone(),
                routes: route_entries.clone(),
                hidden_rows: hidden_rows.clone(),
            };
            encoded.push(conn.encode(&frame)?);
        }
        tl("encode_done", Some(layer_id), None);
        let t_encode = std::time::Instant::now();
        if parallel {
            let conns = std::mem::take(&mut self.conns);
            let handles: Vec<_> = conns
                .into_iter()
                .zip(encoded)
                .map(|(mut conn, bytes)| {
                    std::thread::spawn(move || {
                        let r = conn.write_blocking(layer_id, &bytes);
                        (conn, r)
                    })
                })
                .collect();
            let mut new_conns = Vec::with_capacity(SPARKS);
            for h in handles {
                let (conn, r) = h.join().map_err(|_| "wire send thread panic".to_string())?;
                r?;
                new_conns.push(conn);
            }
            self.conns = new_conns;
        } else {
            for (conn, bytes) in self.conns.iter_mut().zip(encoded) {
                conn.write_blocking(layer_id, &bytes)?;
            }
        }
        if prof {
            eprintln!(
                "PROFILE wire_encode {:.3} wire_write {:.3}",
                (t_encode - ts).as_secs_f64() * 1e3,
                t_encode.elapsed().as_secs_f64() * 1e3
            );
        }
        Ok(request_id)
    }

    fn exchange(
        &mut self,
        layer_id: u32,
        tokens: usize,
        hidden_rows: Vec<HiddenRow>,
        routes: &[(u32, f32)],
        topk: usize,
    ) -> Result<Vec<f32>, String> {
        if self.rdma_body.is_some() {
            return Err("wire: the host-sum path does not run over RDMA; use moe_layer_prequant_raw".into());
        }
        if !self.inflight.is_empty() {
            return Err("wire: host-sum exchange while a raw exchange is in flight".into());
        }
        let request_id = self.send_layer(layer_id, tokens, hidden_rows, routes, topk)?;
        let parallel = tokens > 1;
        let prof = std::env::var_os("GLM53F_PROFILE").is_some();
        // Collect the 4 rank partials (or, reduce-scattered, the 4 row slices). Blocking
        // reads; the prefill uses one thread per connection, the decode reads the four
        // sequentially.
        let tc = std::time::Instant::now();
        let mut sum = if self.cfg.row_sharded(tokens).is_some() {
            CoordinatorSum::row_sharded(tokens, HIDDEN, WireNaive::NONE)
        } else {
            CoordinatorSum::new(tokens, HIDDEN, WireNaive::NONE)
        };
        let mut add = |frame: &ReturnFrame| -> Result<(), String> {
            if (frame.request_id, frame.layer_id) != (request_id, layer_id) {
                return Err(format!(
                    "wire: a return for request {} layer {} while collecting request {request_id} layer {layer_id}",
                    frame.request_id, frame.layer_id
                ));
            }
            sum.accumulate(frame).map_err(|e| e.to_string())
        };
        if parallel {
            let conns = std::mem::take(&mut self.conns);
            let handles: Vec<_> = conns
                .into_iter()
                .map(|mut conn| {
                    std::thread::spawn(move || {
                        let r = conn.recv_frame_blocking(layer_id);
                        (conn, r)
                    })
                })
                .collect();
            let mut new_conns = Vec::with_capacity(SPARKS);
            for h in handles {
                let (conn, r) = h.join().map_err(|_| "wire recv thread panic".to_string())?;
                let frame = r.map_err(|e| format!("wire: {e} (layer {layer_id})"))?;
                add(&frame)?;
                new_conns.push(conn);
            }
            self.conns = new_conns;
        } else {
            for conn in self.conns.iter_mut() {
                let frame = conn
                    .recv_frame_blocking(layer_id)
                    .map_err(|e| format!("wire: {e} (layer {layer_id})"))?;
                add(&frame)?;
            }
        }
        tl("sum_done", Some(layer_id), None);
        if prof { eprintln!("PROFILE wire_collect {:.3}", tc.elapsed().as_secs_f64() * 1e3); }
        // The routed scale, applied to the rank sum (1.0 leaves it bit for bit).
        let scale = self.cfg.routed_scale;
        sum.result().map(|s| s.iter().map(|&v| v * scale).collect()).map_err(|e| e.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Decode one block of the wire format inline (mirror of the Spark's
    /// `decode_hidden`), using the shared codec tables.
    fn decode_hidden_inline(h: &HiddenRow) -> Vec<f32> {
        let mut out = Vec::with_capacity(HIDDEN);
        for k in 0..HIDDEN {
            let v = crate::fp8::decode_e4m3(h.payload[k]);
            let sbyte = h.scales[k / 32];
            let s = 2f64.powi((sbyte.min(254) as i32) - 127);
            out.push((v * s) as f32);
        }
        out
    }

    #[test]
    fn quantize_round_trips_within_e4m3_error() {
        // Deterministic non-trivial values, kept inside one block's E4M3 dynamic
        // range (|x| in [0.05, 2]) so the scale never underflows a value — the
        // underflow region is a separate, expected E4M3 property, not this test.
        let mut hidden = vec![0.0f32; HIDDEN];
        for (i, v) in hidden.iter_mut().enumerate() {
            let mag = 2f64.powi(((i / 32) % 3) as i32 - 1);
            let sign = if (i / 4) % 2 == 0 { 1.0 } else { -1.0 };
            *v = (sign * ((i as f64 * 0.7).sin().abs() * 0.9 + 0.1) * mag) as f32;
        }
        let row = quantize_hidden(&hidden).expect("quantize");
        assert_eq!(row.payload.len(), HIDDEN);
        assert_eq!(row.scales.len(), HIDDEN / 32);
        let decoded = decode_hidden_inline(&row);
        // E4M3 is 3-bit mantissa => relative error ~2^-4 = 6.25% + scale rounding.
        let mut max_rel = 0.0f64;
        let mut worst = (0usize, 0.0f32, 0.0f32);
        for (i, (a, b)) in hidden.iter().zip(decoded.iter()).enumerate() {
            let denom = 1.0f64.max(a.abs() as f64);
            let rel = (a - b).abs() as f64 / denom;
            if rel > max_rel {
                max_rel = rel;
                worst = (i, *a, *b);
            }
        }
        assert!(
            max_rel < 0.07,
            "E4M3 K32 relative error {max_rel} too large at index {} (a={} b={} s={})",
            worst.0,
            worst.1,
            worst.2,
            row.scales[worst.0 / 32]
        );
    }

    #[test]
    fn quantize_zero_is_exact() {
        let row = quantize_hidden(&[0.0f32; HIDDEN]).expect("quantize");
        assert!(row.payload.iter().all(|&c| c == 0), "zero payload");
        assert!(row.scales.iter().all(|&s| s == 0), "zero scales");
    }

    /// Blocking read of one full DS41RTE3 frame (the mock Spark's server side).
    fn read_frame_blocking(s: &mut TcpStream) -> Vec<u8> {
        let mut hdr = vec![0u8; 128];
        s.read_exact(&mut hdr).expect("header");
        let wb = u64::from_le_bytes(hdr[76..84].try_into().unwrap()) as usize;
        let mut rest = vec![0u8; wb - 128];
        s.read_exact(&mut rest).expect("body");
        hdr.extend_from_slice(&rest);
        hdr
    }

    /// Four mock ranks on one listener: each answers `exchanges` requests. For a four-plane
    /// request rank `r` returns `value(r)` in every element of every row; for a reduce-scattered
    /// one it returns its partition of the rows holding `value(0) + ... + value(3)` (what the
    /// ranks' exchange sums), or, with `planes_always`, a full plane anyway (a rank that does
    /// not know the extension). Each request is reported on `seen` (rank, the request) before
    /// the rank waits for `go` (when given) and replies.
    fn mock_ranks_with(
        exchanges: usize,
        value: fn(usize) -> f32,
        seen: std::sync::mpsc::Sender<(usize, RequestFrame)>,
        go: Option<std::sync::mpsc::Receiver<()>>,
        planes_always: bool,
    ) -> (String, std::thread::JoinHandle<()>) {
        use std::net::TcpListener;
        let listener = TcpListener::bind("127.0.0.1:0").expect("bind");
        let addr = listener.local_addr().expect("addr").to_string();
        let server = std::thread::spawn(move || {
            let mut streams = Vec::new();
            for _ in 0..SPARKS {
                let (s, _) = listener.accept().expect("accept");
                streams.push(s);
            }
            let mut rxs: Vec<StreamReceiver> = (0..SPARKS).map(|_| StreamReceiver::new(WireNaive::NONE)).collect();
            let mut txs: Vec<StreamSender> = (0..SPARKS).map(|_| StreamSender::new(WireNaive::NONE)).collect();
            for _ in 0..exchanges {
                let mut reqs = Vec::new();
                for (rank, s) in streams.iter_mut().enumerate() {
                    let bytes = read_frame_blocking(s);
                    let req = match rxs[rank].accept(&bytes).expect("accept request") {
                        Frame::Request(r) => r,
                        Frame::Return(_) => panic!("expected request"),
                    };
                    seen.send((rank, req.clone())).expect("report");
                    reqs.push(req);
                }
                if let Some(go) = go.as_ref() {
                    go.recv().expect("go");
                }
                for (rank, (s, req)) in streams.iter_mut().zip(reqs).enumerate() {
                    let tokens = req.rows.len();
                    let sharded = req.flags & glm53f_wire::FLAG_REDUCE_SCATTER != 0 && !planes_always;
                    let (first, rows, v) = if sharded {
                        let (first, rows) = row_partition(tokens, SPARKS, rank);
                        (first, rows, (0..SPARKS).map(value).sum())
                    } else {
                        (0, tokens, value(rank))
                    };
                    let code = glm53f_wire::bf16::f32_to_bf16(v, WireNaive::NONE);
                    let ret = ReturnFrame {
                        request_id: req.request_id,
                        placement_version: req.placement_version,
                        layer_id: req.layer_id,
                        executor_id: req.executor_id,
                        token_position: first as u64,
                        status: glm53f_wire::Status::Ok,
                        flags: glm53f_wire::FLAG_RETURN_REQUIRED | if sharded { glm53f_wire::FLAG_ROW_SLICE } else { 0 },
                        route_count: 8,
                        seq: 0,
                        rows: (0..rows).map(|_| glm53f_wire::ReturnRow { codes: vec![code; HIDDEN] }).collect(),
                    };
                    let stamped = txs[rank].encode_return(&ret).expect("encode return");
                    s.write_all(&stamped).expect("send return");
                }
            }
        });
        (addr, server)
    }

    fn mock_ranks(
        exchanges: usize,
        value: fn(usize) -> f32,
        seen: std::sync::mpsc::Sender<(usize, RequestFrame)>,
        go: Option<std::sync::mpsc::Receiver<()>>,
    ) -> (String, std::thread::JoinHandle<()>) {
        mock_ranks_with(exchanges, value, seen, go, false)
    }

    fn config(scale: f32) -> WireConfig {
        WireConfig { experts: 288, routed_scale: scale, return_path: ReturnPath::FourPlaneSum }
    }

    /// End-to-end: 4 mock ranks each return a fixed 1.0 partial; the client's rank-ordered sum
    /// is exactly 4.0 per element, times the routed scale. GLM-5.3-Flash's expert ids past 255
    /// reach the ranks intact.
    #[test]
    fn wire_client_sums_four_rank_partials_over_tcp() {
        for scale in [1.0f32, 2.5] {
            let (seen_tx, seen) = std::sync::mpsc::channel();
            let (addr, server) = mock_ranks(1, |_| 1.0, seen_tx, None);
            let tokens = 3usize;
            let mut client = WireClient::connect(&vec![addr; SPARKS], config(scale)).expect("connect");
            let hidden = vec![1.0f32; tokens * HIDDEN];
            let routes: Vec<(u32, f32)> = (0..tokens * 8).map(|i| ((256 + (i + 24) % 32) as u32, 0.125)).collect();
            let sum = client.moe_layer(7, &hidden, &routes, 8).expect("moe_layer");
            assert_eq!(sum.len(), tokens * HIDDEN);
            assert!(sum.iter().all(|&v| v == 4.0 * scale), "rank sum must be 4.0 x {scale}");
            server.join().expect("server join");
            for _ in 0..SPARKS {
                let (_, got) = seen.recv().unwrap();
                let ids: Vec<u32> = got.routes.iter().map(|r| r.expert_id).collect();
                assert_eq!(ids, routes.iter().map(|r| r.0).collect::<Vec<_>>());
                assert!(ids.contains(&287) && ids.contains(&256));
                assert_eq!(got.flags, 0, "four-plane requests carry no reduce-scatter flag");
            }
        }
    }

    /// The shared-expert hook runs after the request reached every rank and before the returns
    /// are collected; the raw planes arrive in rank order through `returned`.
    #[test]
    fn the_shared_expert_hook_runs_during_the_remote_wait() {
        let (seen_tx, seen) = std::sync::mpsc::channel();
        let (go_tx, go) = std::sync::mpsc::channel();
        let (addr, server) = mock_ranks(1, |r| (r + 1) as f32, seen_tx, Some(go));
        let mut client = WireClient::connect(&vec![addr; SPARKS], config(1.0)).expect("connect");
        let tokens = 2usize;
        let (scales, scale_inv) = quantize_hidden_scales(&vec![0.5f32; tokens * HIDDEN]).unwrap();
        let payload: Vec<u8> = (0..tokens * HIDDEN)
            .map(|i| crate::fp8::encode_e4m3(0.5 * f64::from(scale_inv[i >> 5])))
            .collect();
        let routes: Vec<(u32, f32)> = (0..tokens * 8).map(|i| (i as u32 * 17 % 288, 0.1)).collect();
        client.moe_send_raw(3, &payload, &scales, &routes, 8).expect("send");
        let mut ran = false;
        let (layer, n) = client
            .moe_recv_during(|| {
                // Every rank has the request: the send completed before the hook.
                for _ in 0..SPARKS {
                    seen.recv().map_err(|e| e.to_string())?;
                }
                ran = true;
                go_tx.send(()).map_err(|e| e.to_string())
            })
            .expect("recv");
        assert!(ran);
        assert_eq!((layer, n), (3, tokens));
        let Returned::Planes(planes) = client.returned(n);
        for (r, p) in planes.iter().enumerate() {
            assert_eq!(p.len(), tokens * HIDDEN * 2);
            let v = f32::from_bits(u32::from(u16::from_le_bytes([p[0], p[1]])) << 16);
            assert_eq!(v, (r + 1) as f32, "rank {r}'s plane");
        }
        assert!(client.moe_recv_during(|| Ok(())).is_err(), "nothing left in flight");
        server.join().expect("server join");
    }

    fn sharded(min_rows: usize, exchange: ExchangeDtype, scale: f32) -> WireConfig {
        WireConfig { return_path: ReturnPath::RowSharded { min_rows, exchange }, ..config(scale) }
    }

    fn prequant(tokens: usize) -> (Vec<u8>, Vec<u8>, Vec<(u32, f32)>) {
        let (scales, scale_inv) = quantize_hidden_scales(&vec![0.5f32; tokens * HIDDEN]).unwrap();
        let payload: Vec<u8> = (0..tokens * HIDDEN).map(|i| crate::fp8::encode_e4m3(0.5 * f64::from(scale_inv[i >> 5]))).collect();
        let routes: Vec<(u32, f32)> = (0..tokens * 8).map(|i| (((i % 8) * 36 + i / 8 % 36) as u32, 0.1)).collect();
        (payload, scales, routes)
    }

    /// With the row-sharded return path, exchanges below the threshold stay four-plane and the
    /// others are reduce-scattered: their requests carry the flag (and the exchange dtype), the
    /// host path assembles the row slices, the raw path hands them out through `collected`.
    #[test]
    fn exchanges_are_row_sharded_from_the_threshold_on() {
        for (exchange, scale) in [(ExchangeDtype::Bf16, 1.0f32), (ExchangeDtype::Fp8RowScaled, 2.5)] {
            let (seen_tx, seen) = std::sync::mpsc::channel();
            let (addr, server) = mock_ranks(4, |r| (r + 1) as f32, seen_tx, None);
            let mut client = WireClient::connect(&vec![addr; SPARKS], sharded(16, exchange, scale)).expect("connect");
            // Host path: 15 rows are four planes, 18 rows are reduce-scattered; both sum to 10.
            for tokens in [15usize, 18] {
                let routes: Vec<(u32, f32)> = (0..tokens * 8).map(|i| ((i % 8 * 36) as u32, 0.125)).collect();
                let sum = client.moe_layer(5, &vec![1.0f32; tokens * HIDDEN], &routes, 8).expect("moe_layer");
                assert!(sum.len() == tokens * HIDDEN && sum.iter().all(|&v| v == 10.0 * scale), "{tokens} rows");
                for _ in 0..SPARKS {
                    let (_, req) = seen.recv().unwrap();
                    let want = if tokens >= 16 { exchange.request_flags() } else { 0 };
                    assert_eq!(req.flags, want, "{tokens} rows");
                }
            }
            assert!(client.collected().is_none(), "the host path leaves nothing to collect");
            // Raw path: the row slices cover the rows once, in rank order.
            for tokens in [8usize, 37] {
                let (payload, scales, routes) = prequant(tokens);
                client.moe_send_raw(9, &payload, &scales, &routes, 8).expect("send");
                assert_eq!(client.moe_recv_raw().expect("recv"), (9, tokens));
                for _ in 0..SPARKS {
                    seen.recv().unwrap();
                }
                match client.collected().expect("collected") {
                    Collected::Planes(p) => {
                        assert!(tokens < 16);
                        for (r, plane) in p.iter().enumerate() {
                            assert_eq!(plane.len(), tokens * HIDDEN * 2);
                            assert_eq!(f32::from_bits(u32::from(u16::from_le_bytes([plane[0], plane[1]])) << 16), (r + 1) as f32);
                        }
                        let Returned::Planes(q) = client.returned(tokens);
                        assert_eq!(q[3], p[3]);
                    }
                    Collected::RowSlices(s) => {
                        assert!(tokens >= 16);
                        let mut next = 0;
                        for (r, slice) in s.iter().enumerate() {
                            assert_eq!((slice.first, slice.rows), row_partition(tokens, SPARKS, r));
                            assert_eq!(slice.first, next);
                            next += slice.rows;
                            assert_eq!(slice.bytes.len(), slice.rows * HIDDEN * 2);
                            let v: Vec<f32> = slice.bytes.chunks_exact(2).map(|c| f32::from_bits(u32::from(u16::from_le_bytes([c[0], c[1]])) << 16)).collect();
                            assert!(v.iter().all(|&x| x == 10.0), "rank {r}'s rows are the summed rows");
                        }
                        assert_eq!(next, tokens);
                    }
                }
            }
            drop(client);
            server.join().expect("server join");
        }
    }

    /// A rank that answers a reduce-scattered request with a full plane (it does not know the
    /// extension) is refused, on the raw path and on the host path.
    #[test]
    fn a_plane_for_a_reduce_scattered_request_is_refused() {
        for raw in [true, false] {
            let (seen_tx, _seen) = std::sync::mpsc::channel();
            let (addr, server) = mock_ranks_with(1, |_| 1.0, seen_tx, None, true);
            let mut client = WireClient::connect(&vec![addr; SPARKS], sharded(16, ExchangeDtype::Bf16, 1.0)).expect("connect");
            let (payload, scales, routes) = prequant(20);
            let e = if raw {
                client.moe_send_raw(3, &payload, &scales, &routes, 8).expect("send");
                client.moe_recv_raw().unwrap_err()
            } else {
                client.moe_layer_prequant(3, &payload, &scales, &routes, 8).unwrap_err()
            };
            assert!(e.contains("mismatch") || e.contains("row_slice"), "{e}");
            drop(client);
            server.join().expect("server join");
        }
    }

    #[test]
    fn expert_ids_past_the_model_and_row_shards_smaller_than_a_row_per_rank_are_refused() {
        let (seen_tx, _seen) = std::sync::mpsc::channel();
        let (addr, server) = mock_ranks(0, |_| 1.0, seen_tx, None);
        let addrs = vec![addr; SPARKS];
        assert!(WireClient::connect(&addrs, sharded(3, ExchangeDtype::Bf16, 1.0)).is_err());
        let mut client = WireClient::connect(&addrs, config(1.0)).expect("connect");
        let hidden = vec![1.0f32; HIDDEN];
        let mut routes: Vec<(u32, f32)> = (0..8).map(|i| (280 + i, 0.125)).collect();
        assert!(client.check_routes(&routes, 8).is_ok());
        routes[3].0 = 288;
        let e = client.moe_layer(4, &hidden, &routes, 8).unwrap_err();
        assert!(e.contains("expert id 288"), "{e}");
        assert!(client.check_routes(&routes[..7], 8).is_err(), "a partial row");
        drop(client);
        server.join().expect("server join");
    }
}
