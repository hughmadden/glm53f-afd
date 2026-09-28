//! DS41RTE3 v3 frame codec — request rows (4,360 B) and compact returns
//! (8,192 B BF16), with the L4 CRC32C + sequence stamp per frame.
//!
//! Byte-level row formats derive from ds41rt `protocol_v2.rs`
//! (tpurtell/ds41rt v10 `ds41rt-transport`; row/route codecs
//! `:580-620`, request prefix `request.rs:364-410`) — byte-exact row
//! descriptor (40 B) and route entry (12 B) layouts. Deltas vs ds41rt v3:
//! unified 128-B header for both directions, L4 tail (seq @112, CRC32C @120
//! replacing the SHA-256 debug checksum), frame-level `source_kind`/
//! `token_position`/`executor_id`, and a bare BF16 return row (no descriptor —
//! the 8,192 B/token contract). Consumers preserve hidden payload and scale
//! bytes verbatim (ds41rt `Fp8E4m3Ue8m0K32` contract).
//!
//! Checksum convention: CRC32C over the whole frame with the CRC field
//! zeroed. The decode order is header parse (magic/version/kind/lengths) ->
//! CRC verify -> body parse -> geometry checks, so protocol mismatches fail
//! loud before corruption classification.
//!
//! Version 4 (the prefill reduce-scatter, [`crate::row_shard`]) is this codec
//! with three flags: a request with [`FLAG_REDUCE_SCATTER`] and a return with
//! [`FLAG_ROW_SLICE`] are written as version 4, everything else as version 3.
//! Decoding checks the pairing both ways after the CRC, and the word at byte
//! 124 must be zero in both versions.

use crate::bf16;
use crate::crc32c::{crc32c_with, crc32c_zeroed, family_for};
use crate::error::WireError;
use crate::layout::{self, desc, hdr, route, Dtype, SourceKind, Status};
use crate::naive::WireNaive;

/// ds41rt flag: Spark reduced its route partials before returning.
pub const FLAG_SPARK_REDUCTION: u32 = 1 << 9;
/// ds41rt flag: compact BF16 rank partials (the A6 return contract).
pub const FLAG_V41_COMPACT_BF16: u32 = 1 << 16;
/// Every return frame must carry both bits or decode fails loud: an
/// un-pre-summed return is the per-route FP32 wall (ADVISOR-I4 A6).
pub const FLAG_RETURN_REQUIRED: u32 = FLAG_SPARK_REDUCTION | FLAG_V41_COMPACT_BF16;
/// Version 4 (`crate::row_shard`): a return or exchange frame that carries one rank's
/// partition of the request's rows, `token_position` being its first row. A return's rows are
/// already summed over the four ranks.
pub const FLAG_ROW_SLICE: u32 = 1 << 17;
/// Version 4, requests: the ranks reduce-scatter the return among themselves, and each returns
/// only its partition of the rows, summed (a row-slice return).
pub const FLAG_REDUCE_SCATTER: u32 = 1 << 18;
/// Version 4, requests with [`FLAG_REDUCE_SCATTER`]: the ranks exchange their rows as FP8 E4M3
/// with one FP32 scale per row instead of BF16.
pub const FLAG_EXCHANGE_FP8: u32 = 1 << 19;
/// The flags only version 4 may carry.
pub const FLAGS_ROW_SHARD: u32 = FLAG_ROW_SLICE | FLAG_REDUCE_SCATTER | FLAG_EXCHANGE_FP8;

/// 40-B row descriptor (byte-exact ds41rt v3 layout).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RowDescriptor {
    pub row_id: u64,
    pub source_kind: SourceKind,
    pub source_request_id: u64,
    pub token_position: u64,
    pub route_offset: u32,
    pub route_count: u32,
}

/// 12-B route entry: row index, expert id, gate weight (byte-exact ds41rt v3).
#[derive(Debug, Clone, PartialEq)]
pub struct RouteEntry {
    pub row_index: u32,
    pub expert_id: u32,
    pub gate_weight: f32,
}

/// Hidden row as carried on the wire: E4M3 payload bytes followed by the raw
/// scale region (128 UE8M0 K32 bytes canonical). Preserved verbatim both ways.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HiddenRow {
    pub payload: Vec<u8>,
    pub scales: Vec<u8>,
}

/// Coordinator -> Spark request frame (one or more token rows).
#[derive(Debug, Clone, PartialEq)]
pub struct RequestFrame {
    pub request_id: u64,
    pub placement_version: u64,
    pub layer_id: u32,
    pub executor_id: u64,
    pub source_kind: SourceKind,
    pub token_position: u64,
    pub flags: u32,
    /// L4 sequence stamp (filled by `StreamSender`, checked by `StreamReceiver`).
    pub seq: u64,
    pub rows: Vec<RowDescriptor>,
    pub routes: Vec<RouteEntry>,
    pub hidden_rows: Vec<HiddenRow>,
}

/// One compact BF16 partial row (4,096 codes = 8,192 B on the wire).
#[derive(Debug, Clone, PartialEq)]
pub struct ReturnRow {
    pub codes: Vec<u16>,
}

/// Spark -> coordinator compact return frame: the Spark's pre-summed, weighted
/// route partials as BF16 rows (8,192 B per token).
#[derive(Debug, Clone, PartialEq)]
pub struct ReturnFrame {
    pub request_id: u64,
    pub placement_version: u64,
    pub layer_id: u32,
    pub executor_id: u64,
    pub token_position: u64,
    pub status: Status,
    pub flags: u32,
    /// Route partials pre-summed into each row (router top-8).
    pub route_count: usize,
    /// L4 sequence stamp (filled by `StreamSender`, checked by `StreamReceiver`).
    pub seq: u64,
    pub rows: Vec<ReturnRow>,
}

/// A decoded frame.
#[derive(Debug, Clone, PartialEq)]
pub enum Frame {
    Request(RequestFrame),
    Return(ReturnFrame),
}

impl Frame {
    pub fn seq(&self) -> u64 {
        match self {
            Frame::Request(r) => r.seq,
            Frame::Return(r) => r.seq,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct HeaderFields {
    pub(crate) version: u16,
    pub(crate) kind: u16,
    pub(crate) request_id: u64,
    pub(crate) placement_version: u64,
    pub(crate) layer_id: u32,
    pub(crate) row_count: u32,
    pub(crate) dim: u32,
    pub(crate) dtype: Dtype,
    pub(crate) source_kind: SourceKind,
    pub(crate) route_count: u32,
    pub(crate) row_descriptor_bytes: u32,
    pub(crate) route_bytes: u32,
    pub(crate) payload_bytes: u64,
    pub(crate) logical_payload_bytes: u64,
    pub(crate) wire_bytes: u64,
    pub(crate) flags: u32,
    pub(crate) row_stride_bytes: u32,
    pub(crate) status: Status,
    pub(crate) executor_id: u64,
    pub(crate) token_position: u64,
    pub(crate) seq: u64,
    /// The word at byte 124: zero, except in an exchange frame (`crate::row_shard`).
    pub(crate) reserved: u32,
}

/// The version a request with `flags` is written as: 4 for a reduce-scattered request, else 3.
fn request_version(flags: u32, rows: usize) -> Result<u16, WireError> {
    if flags & FLAG_ROW_SLICE != 0 || (flags & FLAG_EXCHANGE_FP8 != 0 && flags & FLAG_REDUCE_SCATTER == 0) {
        return Err(WireError::BadCode { field: "request flags", code: flags });
    }
    if flags & FLAG_REDUCE_SCATTER == 0 {
        return Ok(layout::VERSION);
    }
    // Every rank's partition must hold a row: the reduce-scatter needs at least one row per rank.
    if rows < layout::SPARKS {
        return Err(WireError::DimMismatch { field: "reduce_scatter_rows", want: layout::SPARKS, got: rows });
    }
    Ok(layout::VERSION_ROW_SHARD)
}

/// The version a return with `flags` is written as: 4 for a row slice, else 3.
fn return_version(flags: u32) -> Result<u16, WireError> {
    if flags & (FLAG_REDUCE_SCATTER | FLAG_EXCHANGE_FP8) != 0 {
        return Err(WireError::BadCode { field: "return flags", code: flags });
    }
    Ok(if flags & FLAG_ROW_SLICE != 0 { layout::VERSION_ROW_SHARD } else { layout::VERSION })
}

/// The version-4 rules for requests and returns, checked after the CRC (so a flipped bit is
/// corruption, not a protocol mismatch): version-4 flags only in version 4 and version 4 only
/// with them; FP8 exchange only with the reduce-scatter; at least one row per rank in a
/// reduce-scattered request; a row-slice flag on every successful version-4 return; the word at
/// byte 124 zero.
fn check_extension(h: &HeaderFields) -> Result<(), WireError> {
    if h.reserved != 0 {
        return Err(WireError::BadCode { field: "reserved", code: h.reserved });
    }
    let v4 = h.version == layout::VERSION_ROW_SHARD;
    let bad = |field: &'static str| Err(WireError::BadCode { field, code: h.flags });
    match h.kind {
        layout::KIND_REQUEST if !v4 && h.flags & FLAGS_ROW_SHARD != 0 => bad("version-3 request flags"),
        layout::KIND_REQUEST if v4 => {
            if request_version(h.flags, h.row_count as usize)? != layout::VERSION_ROW_SHARD {
                return bad("version-4 request flags");
            }
            Ok(())
        }
        layout::KIND_RETURN if !v4 && h.flags & FLAGS_ROW_SHARD != 0 => bad("version-3 return flags"),
        layout::KIND_RETURN if v4 => {
            if h.flags & (FLAG_REDUCE_SCATTER | FLAG_EXCHANGE_FP8) != 0
                || (h.status == Status::Ok && h.flags & FLAG_ROW_SLICE == 0)
            {
                return bad("version-4 return flags");
            }
            Ok(())
        }
        _ => Ok(()),
    }
}

// ---------------------------------------------------------------------------
// Primitive LE codec helpers
// ---------------------------------------------------------------------------

fn put_u16(out: &mut [u8], at: usize, v: u16) {
    out[at..at + 2].copy_from_slice(&v.to_le_bytes());
}

fn put_u32(out: &mut [u8], at: usize, v: u32) {
    out[at..at + 4].copy_from_slice(&v.to_le_bytes());
}

fn put_u64(out: &mut [u8], at: usize, v: u64) {
    out[at..at + 8].copy_from_slice(&v.to_le_bytes());
}

fn get_u16(bytes: &[u8], at: usize, _field: &'static str) -> Result<u16, WireError> {
    let end = at.checked_add(2).filter(|&e| e <= bytes.len()).ok_or(WireError::TooShort {
        need: at + 2,
        got: bytes.len(),
    })?;
    Ok(u16::from_le_bytes(bytes[at..end].try_into().unwrap()))
}

fn get_u32(bytes: &[u8], at: usize, _field: &'static str) -> Result<u32, WireError> {
    let end = at.checked_add(4).filter(|&e| e <= bytes.len()).ok_or(WireError::TooShort {
        need: at + 4,
        got: bytes.len(),
    })?;
    Ok(u32::from_le_bytes(bytes[at..end].try_into().unwrap()))
}

fn get_u64(bytes: &[u8], at: usize, _field: &'static str) -> Result<u64, WireError> {
    let end = at.checked_add(8).filter(|&e| e <= bytes.len()).ok_or(WireError::TooShort {
        need: at + 8,
        got: bytes.len(),
    })?;
    Ok(u64::from_le_bytes(bytes[at..end].try_into().unwrap()))
}

fn get_f32(bytes: &[u8], at: usize, _field: &'static str) -> Result<f32, WireError> {
    let end = at.checked_add(4).filter(|&e| e <= bytes.len()).ok_or(WireError::TooShort {
        need: at + 4,
        got: bytes.len(),
    })?;
    Ok(f32::from_le_bytes(bytes[at..end].try_into().unwrap()))
}

pub(crate) fn checked_mul(a: usize, b: usize, field: &'static str) -> Result<usize, WireError> {
    a.checked_mul(b).ok_or(WireError::DimMismatch { field, want: 0, got: usize::MAX })
}

pub(crate) fn as_u32(v: usize, field: &'static str) -> Result<u32, WireError> {
    u32::try_from(v).map_err(|_| WireError::DimMismatch { field, want: u32::MAX as usize, got: v })
}

// ---------------------------------------------------------------------------
// Row descriptor + route entry wire codecs (byte-exact ds41rt v3)
// ---------------------------------------------------------------------------

/// Encode a 40-B row descriptor.
pub fn row_descriptor_wire(row: &RowDescriptor, naive: WireNaive) -> [u8; layout::ROW_DESCRIPTOR_LEN] {
    let mut b = [0u8; layout::ROW_DESCRIPTOR_LEN];
    put_u64(&mut b, desc::ROW_ID, row.row_id);
    put_u16(&mut b, desc::SOURCE_KIND, row.source_kind.code());
    // The trap swaps the two 64-B identity fields (packing drift).
    let (id_at, pos_at) = if naive.has(WireNaive::DESC_FIELD_DRIFT) {
        (desc::TOKEN_POSITION, desc::SOURCE_REQUEST_ID)
    } else {
        (desc::SOURCE_REQUEST_ID, desc::TOKEN_POSITION)
    };
    put_u64(&mut b, id_at, row.source_request_id);
    put_u64(&mut b, pos_at, row.token_position);
    put_u32(&mut b, desc::ROUTE_OFFSET, row.route_offset);
    put_u32(&mut b, desc::ROUTE_COUNT, row.route_count);
    b
}

/// Decode a 40-B row descriptor at `offset`.
pub fn row_descriptor_from_wire(
    bytes: &[u8],
    offset: usize,
    naive: WireNaive,
) -> Result<RowDescriptor, WireError> {
    let end = offset
        .checked_add(layout::ROW_DESCRIPTOR_LEN)
        .filter(|&e| e <= bytes.len())
        .ok_or(WireError::TooShort {
            need: offset + layout::ROW_DESCRIPTOR_LEN,
            got: bytes.len(),
        })?;
    let b = &bytes[offset..end];
    let (id_at, pos_at) = if naive.has(WireNaive::DESC_FIELD_DRIFT) {
        (desc::TOKEN_POSITION, desc::SOURCE_REQUEST_ID)
    } else {
        (desc::SOURCE_REQUEST_ID, desc::TOKEN_POSITION)
    };
    Ok(RowDescriptor {
        row_id: get_u64(b, desc::ROW_ID, "row_id")?,
        source_kind: SourceKind::from_code(get_u16(b, desc::SOURCE_KIND, "source_kind")?)?,
        source_request_id: get_u64(b, id_at, "source_request_id")?,
        token_position: get_u64(b, pos_at, "token_position")?,
        route_offset: get_u32(b, desc::ROUTE_OFFSET, "route_offset")?,
        route_count: get_u32(b, desc::ROUTE_COUNT, "route_count")?,
    })
}

/// Encode a 12-B route entry (row index, expert id, gate weight).
pub fn route_entry_wire(r: &RouteEntry, naive: WireNaive) -> [u8; layout::ROUTE_ENTRY_LEN] {
    let mut b = [0u8; layout::ROUTE_ENTRY_LEN];
    put_u32(&mut b, route::ROW_INDEX, r.row_index);
    // The trap swaps id and weight (silent wrong routing + garbage weights).
    let (id_at, w_at) = if naive.has(WireNaive::ROUTE_FIELD_SWAP) {
        (route::GATE_WEIGHT, route::EXPERT_ID)
    } else {
        (route::EXPERT_ID, route::GATE_WEIGHT)
    };
    put_u32(&mut b, id_at, r.expert_id);
    b[w_at..w_at + 4].copy_from_slice(&r.gate_weight.to_le_bytes());
    b
}

/// Decode a 12-B route entry at `offset`.
pub fn route_entry_from_wire(
    bytes: &[u8],
    offset: usize,
    naive: WireNaive,
) -> Result<RouteEntry, WireError> {
    let end = offset
        .checked_add(layout::ROUTE_ENTRY_LEN)
        .filter(|&e| e <= bytes.len())
        .ok_or(WireError::TooShort {
            need: offset + layout::ROUTE_ENTRY_LEN,
            got: bytes.len(),
        })?;
    let b = &bytes[offset..end];
    let (id_at, w_at) = if naive.has(WireNaive::ROUTE_FIELD_SWAP) {
        (route::GATE_WEIGHT, route::EXPERT_ID)
    } else {
        (route::EXPERT_ID, route::GATE_WEIGHT)
    };
    Ok(RouteEntry {
        row_index: get_u32(b, route::ROW_INDEX, "row_index")?,
        expert_id: get_u32(b, id_at, "expert_id")?,
        gate_weight: get_f32(b, w_at, "gate_weight")?,
    })
}

fn validate_rows_routes(rows: &[RowDescriptor], routes: &[RouteEntry]) -> Result<(), WireError> {
    for (i, row) in rows.iter().enumerate() {
        let end = row.route_offset as usize + row.route_count as usize;
        if end > routes.len() {
            return Err(WireError::RouteRangeBad { row: i });
        }
    }
    for (j, r) in routes.iter().enumerate() {
        if r.row_index as usize >= rows.len() {
            return Err(WireError::RouteRowMismatch {
                entry: j,
                row_index: r.row_index,
                rows: rows.len() as u32,
            });
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Header codec
// ---------------------------------------------------------------------------

pub(crate) fn write_header(out: &mut Vec<u8>, h: &HeaderFields, naive: WireNaive) {
    let mut b = [0u8; layout::HEADER_LEN];
    b[hdr::MAGIC..hdr::MAGIC + 8].copy_from_slice(&layout::MAGIC);
    put_u16(&mut b, hdr::VERSION, h.version);
    put_u16(&mut b, hdr::KIND, h.kind);
    put_u32(&mut b, hdr::HEADER_LEN, layout::HEADER_LEN as u32);
    put_u64(&mut b, hdr::REQUEST_ID, h.request_id);
    put_u64(&mut b, hdr::PLACEMENT_VERSION, h.placement_version);
    put_u32(&mut b, hdr::LAYER_ID, h.layer_id);
    put_u32(&mut b, hdr::ROW_COUNT, h.row_count);
    put_u32(&mut b, hdr::DIM, h.dim);
    put_u16(&mut b, hdr::PAYLOAD_DTYPE, h.dtype.code(naive));
    put_u16(&mut b, hdr::SOURCE_KIND, h.source_kind.code());
    put_u32(&mut b, hdr::ROUTE_COUNT, h.route_count);
    put_u32(&mut b, hdr::ROW_DESCRIPTOR_BYTES, h.row_descriptor_bytes);
    put_u32(&mut b, hdr::ROUTE_BYTES, h.route_bytes);
    put_u64(&mut b, hdr::PAYLOAD_BYTES, h.payload_bytes);
    put_u64(&mut b, hdr::LOGICAL_PAYLOAD_BYTES, h.logical_payload_bytes);
    put_u64(&mut b, hdr::WIRE_BYTES, h.wire_bytes);
    put_u32(&mut b, hdr::FLAGS, h.flags);
    put_u32(&mut b, hdr::ROW_STRIDE_BYTES, h.row_stride_bytes);
    put_u32(&mut b, hdr::STATUS, h.status.code());
    put_u64(&mut b, hdr::EXECUTOR_ID, h.executor_id);
    put_u64(&mut b, hdr::TOKEN_POSITION, h.token_position);
    put_u64(&mut b, hdr::SEQ, h.seq);
    put_u32(&mut b, hdr::RESERVED, h.reserved);
    // The CRC32C field stays zero here; `seal` fills the checksum.
    out.extend_from_slice(&b);
}

/// Parse and check the header: magic, version (3, or 4 for the reduce-scatter extension), the
/// kinds that version has, header length and the frame length against the bytes.
pub(crate) fn parse_header(bytes: &[u8], naive: WireNaive) -> Result<HeaderFields, WireError> {
    if bytes.len() < layout::HEADER_LEN {
        return Err(WireError::TooShort { need: layout::HEADER_LEN, got: bytes.len() });
    }
    if bytes[hdr::MAGIC..hdr::MAGIC + 8] != layout::MAGIC {
        let mut m = [0u8; 8];
        m.copy_from_slice(&bytes[hdr::MAGIC..hdr::MAGIC + 8]);
        return Err(WireError::BadMagic(m));
    }
    let version = get_u16(bytes, hdr::VERSION, "version")?;
    if version != layout::VERSION && version != layout::VERSION_ROW_SHARD {
        return Err(WireError::BadVersion(version));
    }
    let kind = get_u16(bytes, hdr::KIND, "kind")?;
    let exchange = version == layout::VERSION_ROW_SHARD && kind == layout::KIND_EXCHANGE;
    if kind != layout::KIND_REQUEST && kind != layout::KIND_RETURN && !exchange {
        return Err(WireError::BadKind(kind));
    }
    let header_len = get_u32(bytes, hdr::HEADER_LEN, "header_len")?;
    if header_len as usize != layout::HEADER_LEN {
        return Err(WireError::BadHeaderLen(header_len));
    }
    let wire_bytes = get_u64(bytes, hdr::WIRE_BYTES, "wire_bytes")?;
    let declared = wire_bytes as usize;
    if bytes.len() < declared {
        return Err(WireError::Truncated { declared, got: bytes.len() });
    }
    if bytes.len() > declared {
        return Err(WireError::TrailingBytes { declared, got: bytes.len() });
    }
    Ok(HeaderFields {
        version,
        kind,
        request_id: get_u64(bytes, hdr::REQUEST_ID, "request_id")?,
        placement_version: get_u64(bytes, hdr::PLACEMENT_VERSION, "placement_version")?,
        layer_id: get_u32(bytes, hdr::LAYER_ID, "layer_id")?,
        row_count: get_u32(bytes, hdr::ROW_COUNT, "row_count")?,
        dim: get_u32(bytes, hdr::DIM, "dim")?,
        dtype: Dtype::from_code(get_u16(bytes, hdr::PAYLOAD_DTYPE, "payload_dtype")?, naive)?,
        source_kind: SourceKind::from_code(get_u16(bytes, hdr::SOURCE_KIND, "source_kind")?)?,
        route_count: get_u32(bytes, hdr::ROUTE_COUNT, "route_count")?,
        row_descriptor_bytes: get_u32(bytes, hdr::ROW_DESCRIPTOR_BYTES, "row_descriptor_bytes")?,
        route_bytes: get_u32(bytes, hdr::ROUTE_BYTES, "route_bytes")?,
        payload_bytes: get_u64(bytes, hdr::PAYLOAD_BYTES, "payload_bytes")?,
        logical_payload_bytes: get_u64(bytes, hdr::LOGICAL_PAYLOAD_BYTES, "logical_payload_bytes")?,
        wire_bytes,
        flags: get_u32(bytes, hdr::FLAGS, "flags")?,
        row_stride_bytes: get_u32(bytes, hdr::ROW_STRIDE_BYTES, "row_stride_bytes")?,
        status: Status::from_code(get_u32(bytes, hdr::STATUS, "status")?)?,
        executor_id: get_u64(bytes, hdr::EXECUTOR_ID, "executor_id")?,
        token_position: get_u64(bytes, hdr::TOKEN_POSITION, "token_position")?,
        seq: get_u64(bytes, hdr::SEQ, "seq")?,
        reserved: get_u32(bytes, hdr::RESERVED, "reserved")?,
    })
}

/// Best-effort sequence read (for diagnostics and resend bookkeeping).
pub fn frame_seq(bytes: &[u8]) -> Option<u64> {
    if bytes.len() < hdr::SEQ + 8 {
        return None;
    }
    get_u64(bytes, hdr::SEQ, "seq").ok()
}

// ---------------------------------------------------------------------------
// L4 checksum seal/verify
// ---------------------------------------------------------------------------

fn compute_crc(bytes: &[u8], naive: WireNaive) -> u32 {
    let family = family_for(naive);
    if naive.has(WireNaive::CRC_HEADER_EXEMPT) {
        // TRAP: body-only coverage — header fields ride unprotected.
        crc32c_with(family, &bytes[layout::HEADER_LEN..])
    } else {
        // Stream the checksum over the frame with the 4 CRC bytes zeroed, so no
        // 17.86 MB frame copy is paid on every seal and every verify.
        crc32c_zeroed(family, bytes, hdr::CRC32C)
    }
}

/// Process-level CRC switch (perf reset R2 fast path): `GLM53F_WIRE_NOCRC=1`
/// makes this process neither compute (`seal`) nor verify the L4 CRC32C, as the
/// reference engine runs with checksums off unless debugging. It is process
/// configuration, never a frame bit, so no corruption can switch the check off;
/// both peers must agree (a checking receiver refuses a zero-CRC frame loudly).
pub fn crc_disabled() -> bool {
    use std::sync::OnceLock;
    static OFF: OnceLock<bool> = OnceLock::new();
    *OFF.get_or_init(|| std::env::var("GLM53F_WIRE_NOCRC").map(|v| v == "1").unwrap_or(false))
}

fn seal(mut frame: Vec<u8>, naive: WireNaive) -> Vec<u8> {
    if crc_disabled() {
        return frame; // CRC field stays zero
    }
    // Header tail is written zeroed, so the CRC field is already zero here.
    let crc = compute_crc(&frame, naive);
    frame[hdr::CRC32C..hdr::CRC32C + 4].copy_from_slice(&crc.to_le_bytes());
    frame
}

pub(crate) fn verify_crc(bytes: &[u8], naive: WireNaive, seq: u64) -> Result<(), WireError> {
    if naive.has(WireNaive::CRC_UNCHECKED) {
        // TRAP: "trust the transport" — no checksum verification at all.
        return Ok(());
    }
    if crc_disabled() {
        return Ok(());
    }
    let want = compute_crc(bytes, naive);
    let got = get_u32(bytes, hdr::CRC32C, "crc32c")?;
    if want != got {
        return Err(WireError::Corrupt { seq, want, got });
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Request frame
// ---------------------------------------------------------------------------

/// Encode a request frame (header + descriptors + routes + hidden rows).
pub fn encode_request(frame: &RequestFrame, naive: WireNaive) -> Result<Vec<u8>, WireError> {
    encode_request_seq(frame, frame.seq, naive)
}

/// [`encode_request`] with an explicit L4 sequence stamp (so the sender can stamp
/// without cloning the whole frame — the clone was the hot path's 8.6 MB/layer).
pub fn encode_request_seq(
    frame: &RequestFrame,
    seq: u64,
    naive: WireNaive,
) -> Result<Vec<u8>, WireError> {
    if frame.rows.is_empty() {
        return Err(WireError::DimMismatch { field: "rows", want: 1, got: 0 });
    }
    if frame.hidden_rows.len() != frame.rows.len() {
        return Err(WireError::DimMismatch {
            field: "hidden_rows",
            want: frame.rows.len(),
            got: frame.hidden_rows.len(),
        });
    }
    validate_rows_routes(&frame.rows, &frame.routes)?;
    let hidden_row = layout::hidden_row_bytes(naive);
    let scales_len = hidden_row - layout::HIDDEN;
    for h in &frame.hidden_rows {
        if h.payload.len() != layout::HIDDEN {
            return Err(WireError::DimMismatch {
                field: "hidden.payload",
                want: layout::HIDDEN,
                got: h.payload.len(),
            });
        }
        if h.scales.len() != scales_len {
            return Err(WireError::DimMismatch {
                field: "hidden.scales",
                want: scales_len,
                got: h.scales.len(),
            });
        }
    }
    let rows = frame.rows.len();
    let routes = frame.routes.len();
    let version = request_version(frame.flags, rows)?;
    let row_bytes = layout::request_row_bytes(naive);
    let row_descriptor_bytes = checked_mul(rows, layout::ROW_DESCRIPTOR_LEN, "row_descriptor_bytes")?;
    let route_bytes = checked_mul(routes, layout::ROUTE_ENTRY_LEN, "route_bytes")?;
    let payload_bytes = checked_mul(rows, hidden_row, "payload_bytes")?;
    let logical_bytes = checked_mul(rows, row_bytes, "logical_payload_bytes")?;
    let wire_len = layout::HEADER_LEN
        .checked_add(row_descriptor_bytes + route_bytes + payload_bytes)
        .ok_or(WireError::DimMismatch { field: "wire_bytes", want: 0, got: usize::MAX })?;

    let dtype = if naive.has(WireNaive::ROW_SCALED_DTYPE) {
        Dtype::Fp8E4m3RowScaled
    } else {
        Dtype::Fp8E4m3Ue8m0K32
    };
    let mut out = Vec::with_capacity(wire_len);
    write_header(
        &mut out,
        &HeaderFields {
            version,
            kind: layout::KIND_REQUEST,
            request_id: frame.request_id,
            placement_version: frame.placement_version,
            layer_id: frame.layer_id,
            row_count: as_u32(rows, "row_count")?,
            dim: layout::HIDDEN as u32,
            dtype,
            source_kind: frame.source_kind,
            route_count: as_u32(routes, "route_count")?,
            row_descriptor_bytes: as_u32(row_descriptor_bytes, "row_descriptor_bytes")?,
            route_bytes: as_u32(route_bytes, "route_bytes")?,
            payload_bytes: payload_bytes as u64,
            logical_payload_bytes: logical_bytes as u64,
            wire_bytes: wire_len as u64,
            flags: frame.flags,
            row_stride_bytes: as_u32(hidden_row, "row_stride_bytes")?,
            status: Status::Ok,
            executor_id: frame.executor_id,
            token_position: frame.token_position,
            seq,
            reserved: 0,
        },
        naive,
    );
    for row in &frame.rows {
        out.extend_from_slice(&row_descriptor_wire(row, naive));
    }
    for r in &frame.routes {
        out.extend_from_slice(&route_entry_wire(r, naive));
    }
    for h in &frame.hidden_rows {
        out.extend_from_slice(&h.payload);
        out.extend_from_slice(&h.scales);
    }
    debug_assert_eq!(out.len(), wire_len);
    Ok(seal(out, naive))
}

/// In-place request encoding for the coordinator's RDMA fast path (perf reset
/// P6): the frame that [`encode_request_seq`] builds for `rows = routes.len() /
/// topk` decode-kind rows (row `t`: row id and token position `t`, routes
/// `t * topk..`), written without staging the hidden rows. The row descriptors
/// and route entries go into `body` (the frame after its header); the returned
/// header is the frame's first 128 bytes; the caller fills the hidden rows at
/// `body[hidden_offset..hidden_offset + rows * 4224]` (payload then scales per
/// row, e.g. straight from the GPU) before sending `header ++ body[..body_len]`.
/// The CRC field is left zero: callers use it only with the CRC disabled
/// ([`crc_disabled`]), or seal the assembled frame themselves.
///
/// Returns `(header, hidden_offset, body_len)`.
pub fn encode_request_meta_into(
    body: &mut [u8],
    request_id: u64,
    layer_id: u32,
    executor_id: u64,
    seq: u64,
    routes: &[(u32, f32)],
    topk: usize,
    naive: WireNaive,
) -> Result<([u8; layout::HEADER_LEN], usize, usize), WireError> {
    if topk == 0 || routes.is_empty() || routes.len() % topk != 0 {
        return Err(WireError::DimMismatch { field: "routes", want: topk, got: routes.len() });
    }
    let rows = routes.len() / topk;
    let hidden_row = layout::hidden_row_bytes(naive);
    let row_bytes = layout::request_row_bytes(naive);
    let row_descriptor_bytes = checked_mul(rows, layout::ROW_DESCRIPTOR_LEN, "row_descriptor_bytes")?;
    let route_bytes = checked_mul(routes.len(), layout::ROUTE_ENTRY_LEN, "route_bytes")?;
    let payload_bytes = checked_mul(rows, hidden_row, "payload_bytes")?;
    let logical_bytes = checked_mul(rows, row_bytes, "logical_payload_bytes")?;
    let body_len = row_descriptor_bytes + route_bytes + payload_bytes;
    if body.len() < body_len {
        return Err(WireError::TooShort { need: body_len, got: body.len() });
    }
    let mut row_list = Vec::with_capacity(rows);
    let mut route_list = Vec::with_capacity(routes.len());
    for t in 0..rows {
        let row = RowDescriptor {
            row_id: t as u64,
            source_kind: SourceKind::Decode,
            source_request_id: request_id,
            token_position: t as u64,
            route_offset: (t * topk) as u32,
            route_count: topk as u32,
        };
        body[t * layout::ROW_DESCRIPTOR_LEN..(t + 1) * layout::ROW_DESCRIPTOR_LEN]
            .copy_from_slice(&row_descriptor_wire(&row, naive));
        row_list.push(row);
        for k in 0..topk {
            let (expert_id, gate_weight) = routes[t * topk + k];
            let r = RouteEntry { row_index: t as u32, expert_id, gate_weight };
            let o = row_descriptor_bytes + (t * topk + k) * layout::ROUTE_ENTRY_LEN;
            body[o..o + layout::ROUTE_ENTRY_LEN].copy_from_slice(&route_entry_wire(&r, naive));
            route_list.push(r);
        }
    }
    validate_rows_routes(&row_list, &route_list)?;
    let dtype = if naive.has(WireNaive::ROW_SCALED_DTYPE) {
        Dtype::Fp8E4m3RowScaled
    } else {
        Dtype::Fp8E4m3Ue8m0K32
    };
    let mut out = Vec::with_capacity(layout::HEADER_LEN);
    write_header(
        &mut out,
        &HeaderFields {
            version: layout::VERSION,
            kind: layout::KIND_REQUEST,
            request_id,
            placement_version: 1,
            layer_id,
            row_count: as_u32(rows, "row_count")?,
            dim: layout::HIDDEN as u32,
            dtype,
            source_kind: SourceKind::Decode,
            route_count: as_u32(routes.len(), "route_count")?,
            row_descriptor_bytes: as_u32(row_descriptor_bytes, "row_descriptor_bytes")?,
            route_bytes: as_u32(route_bytes, "route_bytes")?,
            payload_bytes: payload_bytes as u64,
            logical_payload_bytes: logical_bytes as u64,
            wire_bytes: (layout::HEADER_LEN + body_len) as u64,
            flags: 0,
            row_stride_bytes: as_u32(hidden_row, "row_stride_bytes")?,
            status: Status::Ok,
            executor_id,
            token_position: 0,
            seq,
            reserved: 0,
        },
        naive,
    );
    let mut header = [0u8; layout::HEADER_LEN];
    header.copy_from_slice(&out);
    Ok((header, row_descriptor_bytes + route_bytes, body_len))
}

/// [`encode_request_meta_into`] without the route entries (perf reset P9): the
/// row descriptors of `rows` decode-kind rows with `topk` routes each go into
/// `body`; the caller fills the route entries at `body[routes_offset..]`
/// (`rows * topk` x 12 B, `route_entry_wire` layout) and the hidden rows at
/// `body[hidden_offset..]` itself, e.g. from the GPU. The header is the same
/// frame's. Returns `(header, routes_offset, hidden_offset, body_len)`.
pub fn encode_request_desc_into(
    body: &mut [u8],
    request_id: u64,
    layer_id: u32,
    executor_id: u64,
    seq: u64,
    rows: usize,
    topk: usize,
    naive: WireNaive,
) -> Result<([u8; layout::HEADER_LEN], usize, usize, usize), WireError> {
    if rows == 0 || topk == 0 || topk > layout::TOPK {
        return Err(WireError::DimMismatch { field: "rows/topk", want: layout::TOPK, got: topk });
    }
    let hidden_row = layout::hidden_row_bytes(naive);
    let row_bytes = layout::request_row_bytes(naive);
    let routes = checked_mul(rows, topk, "route_count")?;
    let row_descriptor_bytes = checked_mul(rows, layout::ROW_DESCRIPTOR_LEN, "row_descriptor_bytes")?;
    let route_bytes = checked_mul(routes, layout::ROUTE_ENTRY_LEN, "route_bytes")?;
    let payload_bytes = checked_mul(rows, hidden_row, "payload_bytes")?;
    let logical_bytes = checked_mul(rows, row_bytes, "logical_payload_bytes")?;
    let body_len = row_descriptor_bytes + route_bytes + payload_bytes;
    if body.len() < body_len {
        return Err(WireError::TooShort { need: body_len, got: body.len() });
    }
    for t in 0..rows {
        let row = RowDescriptor {
            row_id: t as u64,
            source_kind: SourceKind::Decode,
            source_request_id: request_id,
            token_position: t as u64,
            route_offset: (t * topk) as u32,
            route_count: topk as u32,
        };
        body[t * layout::ROW_DESCRIPTOR_LEN..(t + 1) * layout::ROW_DESCRIPTOR_LEN]
            .copy_from_slice(&row_descriptor_wire(&row, naive));
    }
    let dtype = if naive.has(WireNaive::ROW_SCALED_DTYPE) { Dtype::Fp8E4m3RowScaled } else { Dtype::Fp8E4m3Ue8m0K32 };
    let mut out = Vec::with_capacity(layout::HEADER_LEN);
    write_header(
        &mut out,
        &HeaderFields {
            version: layout::VERSION,
            kind: layout::KIND_REQUEST,
            request_id,
            placement_version: 1,
            layer_id,
            row_count: as_u32(rows, "row_count")?,
            dim: layout::HIDDEN as u32,
            dtype,
            source_kind: SourceKind::Decode,
            route_count: as_u32(routes, "route_count")?,
            row_descriptor_bytes: as_u32(row_descriptor_bytes, "row_descriptor_bytes")?,
            route_bytes: as_u32(route_bytes, "route_bytes")?,
            payload_bytes: payload_bytes as u64,
            logical_payload_bytes: logical_bytes as u64,
            wire_bytes: (layout::HEADER_LEN + body_len) as u64,
            flags: 0,
            row_stride_bytes: as_u32(hidden_row, "row_stride_bytes")?,
            status: Status::Ok,
            executor_id,
            token_position: 0,
            seq,
            reserved: 0,
        },
        naive,
    );
    let mut header = [0u8; layout::HEADER_LEN];
    header.copy_from_slice(&out);
    Ok((header, row_descriptor_bytes, row_descriptor_bytes + route_bytes, body_len))
}

/// Set the flags of a request header built in place ([`encode_request_meta_into`],
/// [`encode_request_desc_into`], both version 3 with no flags), and the version they need: a
/// reduce-scattered request ([`FLAG_REDUCE_SCATTER`]) is version 4 and needs at least one row per
/// rank. The CRC field is left as it is: like the in-place encoders, for frames sent with the CRC
/// disabled ([`crc_disabled`]) or sealed afterwards.
pub fn set_request_flags(header: &mut [u8], flags: u32) -> Result<(), WireError> {
    if header.len() < layout::HEADER_LEN {
        return Err(WireError::TooShort { need: layout::HEADER_LEN, got: header.len() });
    }
    if get_u16(header, hdr::KIND, "kind")? != layout::KIND_REQUEST {
        return Err(WireError::BadKind(get_u16(header, hdr::KIND, "kind")?));
    }
    let rows = get_u32(header, hdr::ROW_COUNT, "row_count")? as usize;
    let version = request_version(flags, rows)?;
    put_u16(header, hdr::VERSION, version);
    put_u32(header, hdr::FLAGS, flags);
    Ok(())
}

/// Env-default request encode (NEGATIVE tests).
pub fn encode_request_env(frame: &RequestFrame) -> Result<Vec<u8>, WireError> {
    encode_request(frame, crate::naive::naive_from_env())
}

fn decode_request(h: &HeaderFields, bytes: &[u8], naive: WireNaive) -> Result<RequestFrame, WireError> {
    if h.row_count == 0 {
        return Err(WireError::DimMismatch { field: "row_count", want: 1, got: 0 });
    }
    check_extension(h)?;
    let rows = h.row_count as usize;
    let routes = h.route_count as usize;
    let row_descriptor_bytes = checked_mul(rows, layout::ROW_DESCRIPTOR_LEN, "row_descriptor_bytes")?;
    let route_bytes = checked_mul(routes, layout::ROUTE_ENTRY_LEN, "route_bytes")?;
    if h.row_descriptor_bytes as usize != row_descriptor_bytes {
        return Err(WireError::DimMismatch {
            field: "row_descriptor_bytes",
            want: row_descriptor_bytes,
            got: h.row_descriptor_bytes as usize,
        });
    }
    if h.route_bytes as usize != route_bytes {
        return Err(WireError::DimMismatch {
            field: "route_bytes",
            want: route_bytes,
            got: h.route_bytes as usize,
        });
    }
    if h.dim as usize != layout::HIDDEN {
        return Err(WireError::DimMismatch {
            field: "hidden_dim",
            want: layout::HIDDEN,
            got: h.dim as usize,
        });
    }
    match h.dtype {
        Dtype::Fp8E4m3Ue8m0K32 | Dtype::Fp8E4m3RowScaled => {}
        other => {
            return Err(WireError::DimMismatch {
                field: "request_payload_dtype",
                want: Dtype::Fp8E4m3Ue8m0K32 as usize,
                got: other.code(naive) as usize,
            })
        }
    }
    let stride = h.row_stride_bytes as usize;
    if stride < layout::HIDDEN {
        return Err(WireError::DimMismatch {
            field: "row_stride_bytes",
            want: layout::HIDDEN,
            got: stride,
        });
    }
    let payload_bytes = checked_mul(rows, stride, "payload_bytes")?;
    if h.payload_bytes as usize != payload_bytes {
        return Err(WireError::DimMismatch {
            field: "payload_bytes",
            want: payload_bytes,
            got: h.payload_bytes as usize,
        });
    }
    let body = &bytes[layout::HEADER_LEN..];
    if body.len() != row_descriptor_bytes + route_bytes + payload_bytes {
        return Err(WireError::DimMismatch {
            field: "body_len",
            want: row_descriptor_bytes + route_bytes + payload_bytes,
            got: body.len(),
        });
    }

    let mut rows_vec = Vec::with_capacity(rows);
    for i in 0..rows {
        rows_vec.push(row_descriptor_from_wire(
            body,
            i * layout::ROW_DESCRIPTOR_LEN,
            naive,
        )?);
    }
    let mut routes_vec = Vec::with_capacity(routes);
    for j in 0..routes {
        routes_vec.push(route_entry_from_wire(
            body,
            row_descriptor_bytes + j * layout::ROUTE_ENTRY_LEN,
            naive,
        )?);
    }
    validate_rows_routes(&rows_vec, &routes_vec)?;

    let mut hidden_rows = Vec::with_capacity(rows);
    let hidden_base = row_descriptor_bytes + route_bytes;
    for i in 0..rows {
        let base = hidden_base + i * stride;
        hidden_rows.push(HiddenRow {
            payload: body[base..base + layout::HIDDEN].to_vec(),
            scales: body[base + layout::HIDDEN..base + stride].to_vec(),
        });
    }

    Ok(RequestFrame {
        request_id: h.request_id,
        placement_version: h.placement_version,
        layer_id: h.layer_id,
        executor_id: h.executor_id,
        source_kind: h.source_kind,
        token_position: h.token_position,
        flags: h.flags,
        seq: h.seq,
        rows: rows_vec,
        routes: routes_vec,
        hidden_rows,
    })
}

/// A request frame validated where it lies (perf reset: Spark zero-copy
/// receive). [`RequestView::parse`] runs exactly the checks [`decode_frame`]
/// runs on a request (header, CRC32C unless process-disabled, geometry, every
/// row descriptor and route entry) but copies nothing: descriptors, routes and
/// the hidden rows are read from `bytes` on demand.
pub struct RequestView<'a> {
    pub request_id: u64,
    pub placement_version: u64,
    pub layer_id: u32,
    pub executor_id: u64,
    pub source_kind: SourceKind,
    pub token_position: u64,
    pub flags: u32,
    pub seq: u64,
    /// Row count (> 0) and route count.
    pub rows: usize,
    pub routes: usize,
    /// Bytes per hidden row (payload then scales).
    pub row_stride: usize,
    naive: WireNaive,
    body: &'a [u8],
    route_base: usize,
    hidden_base: usize,
}

impl<'a> RequestView<'a> {
    /// Validate `bytes` as one request frame (no L4 sequence check: the caller
    /// runs `StreamReceiver::accept_seq(view.seq)`).
    pub fn parse(bytes: &'a [u8], naive: WireNaive) -> Result<Self, WireError> {
        let h = parse_header(bytes, naive)?;
        verify_crc(bytes, naive, h.seq)?;
        if h.kind != layout::KIND_REQUEST {
            return Err(WireError::BadKind(h.kind));
        }
        if h.row_count == 0 {
            return Err(WireError::DimMismatch { field: "row_count", want: 1, got: 0 });
        }
        check_extension(&h)?;
        let rows = h.row_count as usize;
        let routes = h.route_count as usize;
        let row_descriptor_bytes = checked_mul(rows, layout::ROW_DESCRIPTOR_LEN, "row_descriptor_bytes")?;
        let route_bytes = checked_mul(routes, layout::ROUTE_ENTRY_LEN, "route_bytes")?;
        if h.row_descriptor_bytes as usize != row_descriptor_bytes {
            return Err(WireError::DimMismatch {
                field: "row_descriptor_bytes",
                want: row_descriptor_bytes,
                got: h.row_descriptor_bytes as usize,
            });
        }
        if h.route_bytes as usize != route_bytes {
            return Err(WireError::DimMismatch { field: "route_bytes", want: route_bytes, got: h.route_bytes as usize });
        }
        if h.dim as usize != layout::HIDDEN {
            return Err(WireError::DimMismatch { field: "hidden_dim", want: layout::HIDDEN, got: h.dim as usize });
        }
        match h.dtype {
            Dtype::Fp8E4m3Ue8m0K32 | Dtype::Fp8E4m3RowScaled => {}
            other => {
                return Err(WireError::DimMismatch {
                    field: "request_payload_dtype",
                    want: Dtype::Fp8E4m3Ue8m0K32 as usize,
                    got: other.code(naive) as usize,
                })
            }
        }
        let stride = h.row_stride_bytes as usize;
        if stride < layout::HIDDEN {
            return Err(WireError::DimMismatch { field: "row_stride_bytes", want: layout::HIDDEN, got: stride });
        }
        let payload_bytes = checked_mul(rows, stride, "payload_bytes")?;
        if h.payload_bytes as usize != payload_bytes {
            return Err(WireError::DimMismatch {
                field: "payload_bytes",
                want: payload_bytes,
                got: h.payload_bytes as usize,
            });
        }
        let body = &bytes[layout::HEADER_LEN..];
        if body.len() != row_descriptor_bytes + route_bytes + payload_bytes {
            return Err(WireError::DimMismatch {
                field: "body_len",
                want: row_descriptor_bytes + route_bytes + payload_bytes,
                got: body.len(),
            });
        }
        let view = Self {
            request_id: h.request_id,
            placement_version: h.placement_version,
            layer_id: h.layer_id,
            executor_id: h.executor_id,
            source_kind: h.source_kind,
            token_position: h.token_position,
            flags: h.flags,
            seq: h.seq,
            rows,
            routes,
            row_stride: stride,
            naive,
            body,
            route_base: row_descriptor_bytes,
            hidden_base: row_descriptor_bytes + route_bytes,
        };
        // validate_rows_routes, entry by entry (each decode checks its codes).
        for i in 0..rows {
            let row = view.row(i)?;
            if row.route_offset as usize + row.route_count as usize > routes {
                return Err(WireError::RouteRangeBad { row: i });
            }
        }
        for j in 0..routes {
            let r = view.route(j)?;
            if r.row_index as usize >= rows {
                return Err(WireError::RouteRowMismatch { entry: j, row_index: r.row_index, rows: rows as u32 });
            }
        }
        Ok(view)
    }

    /// Row descriptor `i` (`i < rows`).
    pub fn row(&self, i: usize) -> Result<RowDescriptor, WireError> {
        row_descriptor_from_wire(self.body, i * layout::ROW_DESCRIPTOR_LEN, self.naive)
    }

    /// Route entry `j` (`j < routes`).
    pub fn route(&self, j: usize) -> Result<RouteEntry, WireError> {
        route_entry_from_wire(self.body, self.route_base + j * layout::ROUTE_ENTRY_LEN, self.naive)
    }

    /// The hidden rows, `rows * row_stride` bytes: per row the E4M3 payload
    /// (`HIDDEN` bytes) then its scales.
    pub fn hidden(&self) -> &'a [u8] {
        &self.body[self.hidden_base..]
    }
}

// ---------------------------------------------------------------------------
// Return frame
// ---------------------------------------------------------------------------

/// Encode a compact return frame (header + 8,192-B BF16 rows per token).
pub fn encode_return(frame: &ReturnFrame, naive: WireNaive) -> Result<Vec<u8>, WireError> {
    encode_return_seq(frame, frame.seq, naive)
}

/// The 128-B header [`encode_return_seq`] writes for a compact BF16 return of
/// `rows` rows (`frame.rows` is ignored): for a frame assembled in place, the body
/// (`rows` x 8,192 B BF16) written after it by the caller, then [`seal_in_place`]
/// (perf reset R2 zero-copy return).
pub fn return_header_seq(frame: &ReturnFrame, rows: usize, seq: u64, naive: WireNaive) -> Result<Vec<u8>, WireError> {
    if rows == 0 {
        return Err(WireError::DimMismatch { field: "rows", want: 1, got: 0 });
    }
    if frame.route_count == 0 {
        return Err(WireError::DimMismatch { field: "route_count", want: 1, got: 0 });
    }
    if naive.has(WireNaive::PER_ROUTE_RETURN) || naive.has(WireNaive::FP32_RETURN) {
        return Err(WireError::DimMismatch { field: "in_place_return_dtype", want: 0, got: 1 });
    }
    let version = return_version(frame.flags)?;
    let block = layout::return_row_bytes(frame.route_count, naive);
    let payload_bytes = checked_mul(rows, block, "payload_bytes")?;
    let wire_len = layout::HEADER_LEN
        .checked_add(payload_bytes)
        .ok_or(WireError::DimMismatch { field: "wire_bytes", want: 0, got: usize::MAX })?;
    let mut out = Vec::with_capacity(layout::HEADER_LEN);
    write_header(
        &mut out,
        &HeaderFields {
            version,
            kind: layout::KIND_RETURN,
            request_id: frame.request_id,
            placement_version: frame.placement_version,
            layer_id: frame.layer_id,
            row_count: as_u32(rows, "row_count")?,
            dim: layout::HIDDEN as u32,
            dtype: Dtype::Bf16,
            source_kind: SourceKind::Decode,
            route_count: as_u32(frame.route_count, "route_count")?,
            row_descriptor_bytes: 0,
            route_bytes: 0,
            payload_bytes: payload_bytes as u64,
            logical_payload_bytes: payload_bytes as u64,
            wire_bytes: wire_len as u64,
            flags: frame.flags,
            row_stride_bytes: as_u32(block, "row_stride_bytes")?,
            status: frame.status,
            executor_id: frame.executor_id,
            token_position: frame.token_position,
            seq,
            reserved: 0,
        },
        naive,
    );
    Ok(out)
}

/// Seal a frame assembled in place: write its CRC32C unless this process runs
/// with CRC disabled ([`crc_disabled`]); the header's CRC field must be zero.
pub fn seal_in_place(buf: &mut [u8], naive: WireNaive) {
    if crc_disabled() {
        return;
    }
    let crc = compute_crc(buf, naive);
    buf[hdr::CRC32C..hdr::CRC32C + 4].copy_from_slice(&crc.to_le_bytes());
}

/// [`encode_return`] with the L4 sequence given explicitly (no frame clone).
pub fn encode_return_seq(frame: &ReturnFrame, seq: u64, naive: WireNaive) -> Result<Vec<u8>, WireError> {
    if frame.rows.is_empty() {
        return Err(WireError::DimMismatch { field: "rows", want: 1, got: 0 });
    }
    if frame.route_count == 0 {
        return Err(WireError::DimMismatch { field: "route_count", want: 1, got: 0 });
    }
    for row in &frame.rows {
        if row.codes.len() != layout::HIDDEN {
            return Err(WireError::DimMismatch {
                field: "return_row_codes",
                want: layout::HIDDEN,
                got: row.codes.len(),
            });
        }
    }
    let per_route = naive.has(WireNaive::PER_ROUTE_RETURN);
    let fp32 = !per_route && naive.has(WireNaive::FP32_RETURN);
    let dtype = if fp32 { Dtype::F32 } else { Dtype::Bf16 };
    let version = return_version(frame.flags)?;
    let block = layout::return_row_bytes(frame.route_count, naive);
    let rows = frame.rows.len();
    let payload_bytes = checked_mul(rows, block, "payload_bytes")?;
    let wire_len = layout::HEADER_LEN
        .checked_add(payload_bytes)
        .ok_or(WireError::DimMismatch { field: "wire_bytes", want: 0, got: usize::MAX })?;

    let mut out = Vec::with_capacity(wire_len);
    write_header(
        &mut out,
        &HeaderFields {
            version,
            kind: layout::KIND_RETURN,
            request_id: frame.request_id,
            placement_version: frame.placement_version,
            layer_id: frame.layer_id,
            row_count: as_u32(rows, "row_count")?,
            dim: layout::HIDDEN as u32,
            dtype,
            source_kind: SourceKind::Decode,
            route_count: as_u32(frame.route_count, "route_count")?,
            row_descriptor_bytes: 0,
            route_bytes: 0,
            payload_bytes: payload_bytes as u64,
            logical_payload_bytes: payload_bytes as u64,
            wire_bytes: wire_len as u64,
            flags: frame.flags,
            row_stride_bytes: as_u32(block, "row_stride_bytes")?,
            status: frame.status,
            executor_id: frame.executor_id,
            token_position: frame.token_position,
            seq,
            reserved: 0,
        },
        naive,
    );
    for row in &frame.rows {
        if fp32 {
            for &code in &row.codes {
                out.extend_from_slice(&bf16::bf16_to_f32(code).to_le_bytes());
            }
        } else if per_route {
            for _ in 0..frame.route_count {
                for &code in &row.codes {
                    out.extend_from_slice(&code.to_le_bytes());
                }
            }
        } else if cfg!(target_endian = "little") {
            // SAFETY: a &[u16] is a valid &[u8] of twice the length, and on a
            // little-endian host its bytes are exactly the LE wire encoding.
            let bytes = unsafe { core::slice::from_raw_parts(row.codes.as_ptr() as *const u8, row.codes.len() * 2) };
            out.extend_from_slice(bytes);
        } else {
            for &code in &row.codes {
                out.extend_from_slice(&code.to_le_bytes());
            }
        }
    }
    debug_assert_eq!(out.len(), wire_len);
    Ok(seal(out, naive))
}

/// Env-default return encode (NEGATIVE tests).
pub fn encode_return_env(frame: &ReturnFrame) -> Result<Vec<u8>, WireError> {
    encode_return(frame, crate::naive::naive_from_env())
}

fn decode_return(h: &HeaderFields, bytes: &[u8], naive: WireNaive) -> Result<ReturnFrame, WireError> {
    // A Status::Error return carries no compact-BF16 reduction (flags 0); the
    // FLAG_RETURN_REQUIRED guard applies only to Ok returns. The coordinator
    // surfaces the Spark error from `status` instead of a decode failure.
    if h.status == Status::Ok && h.flags & FLAG_RETURN_REQUIRED != FLAG_RETURN_REQUIRED {
        return Err(WireError::UnpresummedReturn { flags: h.flags });
    }
    check_extension(h)?;
    if h.row_count == 0 {
        return Err(WireError::DimMismatch { field: "row_count", want: 1, got: 0 });
    }
    if h.dim as usize != layout::HIDDEN {
        return Err(WireError::DimMismatch {
            field: "output_dim",
            want: layout::HIDDEN,
            got: h.dim as usize,
        });
    }
    let rows = h.row_count as usize;
    let stride = h.row_stride_bytes as usize;
    let payload_bytes = checked_mul(rows, stride, "payload_bytes")?;
    if h.payload_bytes as usize != payload_bytes {
        return Err(WireError::DimMismatch {
            field: "payload_bytes",
            want: payload_bytes,
            got: h.payload_bytes as usize,
        });
    }
    if h.row_descriptor_bytes != 0 || h.route_bytes != 0 {
        return Err(WireError::DimMismatch {
            field: "return_row_descriptor_bytes",
            want: 0,
            got: (h.row_descriptor_bytes + h.route_bytes) as usize,
        });
    }
    let body = &bytes[layout::HEADER_LEN..];
    if body.len() != payload_bytes {
        return Err(WireError::DimMismatch {
            field: "body_len",
            want: payload_bytes,
            got: body.len(),
        });
    }
    if h.route_count == 0 {
        return Err(WireError::DimMismatch { field: "route_count", want: 1, got: 0 });
    }

    let mut rows_vec = Vec::with_capacity(rows);
    for i in 0..rows {
        let block = &body[i * stride..(i + 1) * stride];
        let codes = match h.dtype {
            Dtype::F32 => {
                if stride != layout::HIDDEN * 4 {
                    return Err(WireError::DimMismatch {
                        field: "f32_row_stride",
                        want: layout::HIDDEN * 4,
                        got: stride,
                    });
                }
                block
                    .chunks_exact(4)
                    .map(|c| {
                        let v = f32::from_le_bytes(c.try_into().unwrap());
                        bf16::f32_to_bf16(v, naive)
                    })
                    .collect()
            }
            Dtype::Bf16 => {
                let unit = layout::HIDDEN * 2;
                if stride % unit != 0 {
                    return Err(WireError::DimMismatch {
                        field: "bf16_row_stride",
                        want: unit,
                        got: stride,
                    });
                }
                let copies = stride / unit;
                if copies != 1 && copies != h.route_count as usize {
                    return Err(WireError::DimMismatch {
                        field: "bf16_row_copies",
                        want: 1,
                        got: copies,
                    });
                }
                block[..unit]
                    .chunks_exact(2)
                    .map(|c| u16::from_le_bytes(c.try_into().unwrap()))
                    .collect()
            }
            other => {
                return Err(WireError::DimMismatch {
                    field: "return_payload_dtype",
                    want: Dtype::Bf16 as usize,
                    got: other.code(naive) as usize,
                })
            }
        };
        rows_vec.push(ReturnRow { codes });
    }

    Ok(ReturnFrame {
        request_id: h.request_id,
        placement_version: h.placement_version,
        layer_id: h.layer_id,
        executor_id: h.executor_id,
        token_position: h.token_position,
        status: h.status,
        flags: h.flags,
        route_count: h.route_count as usize,
        seq: h.seq,
        rows: rows_vec,
    })
}

// ---------------------------------------------------------------------------
// Frame dispatch
// ---------------------------------------------------------------------------

/// Decode any frame: header parse (fail loud on protocol mismatch), CRC32C
/// verify (Retry-class `Corrupt`), then body parse + geometry checks.
pub fn decode_frame(bytes: &[u8], naive: WireNaive) -> Result<Frame, WireError> {
    let h = parse_header(bytes, naive)?;
    verify_crc(bytes, naive, h.seq)?;
    match h.kind {
        layout::KIND_REQUEST => Ok(Frame::Request(decode_request(&h, bytes, naive)?)),
        layout::KIND_RETURN => Ok(Frame::Return(decode_return(&h, bytes, naive)?)),
        other => Err(WireError::BadKind(other)),
    }
}

/// Env-default frame decode (NEGATIVE tests).
pub fn decode_frame_env(bytes: &[u8]) -> Result<Frame, WireError> {
    decode_frame(bytes, crate::naive::naive_from_env())
}
