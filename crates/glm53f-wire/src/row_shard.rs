//! The prefill reduce-scatter on the wire: version 4 of `DS41RTE3`.
//!
//! **Why.** In the version-3 return every rank sends back all the rows of its partial plane, so a
//! prefill-sized exchange brings four full planes into the coordinator's one port (4 x 32 MB at
//! 4,096 rows) and the coordinator adds them. In the reduce-scatter the ranks add the planes
//! among themselves: each rank sends each peer that peer's rows, adds the three peers' rows of its
//! own partition to its own, and returns only that partition. The coordinator receives each row
//! once. The design, its error bound and its traffic are in the rank crate's README ("Prefill
//! reduce-scatter").
//!
//! **Three frames**, all with version 3's 128-byte header and L4 tail (sequence and CRC32C):
//!
//! | Frame | Kind | Version 4 because | Fields |
//! |---|---|---|---|
//! | Request | 1 | [`FLAG_REDUCE_SCATTER`] (bit 18); [`FLAG_EXCHANGE_FP8`] (bit 19) picks FP8 over BF16 for the exchange | as version 3; at least one row per rank |
//! | Return (a row slice) | 2 | [`FLAG_ROW_SLICE`] (bit 17), with `FLAG_RETURN_REQUIRED` | `row_count`: the rank's rows; `token_position`: their first row; `executor_id`: the rank |
//! | Exchange | 3 | the kind exists only in version 4 | [`ExchangeHeader`] |
//!
//! The exchange frame (rank to rank): `executor_id` is the sending rank, `token_position` the
//! first row carried, `row_count` the rows carried, `payload_dtype` BF16 (8,192 bytes a row) or
//! FP8 E4M3 row-scaled (4,096 E4M3 bytes then one FP32 scale, 4,100 bytes a row), `route_count`
//! 8 (the rows are sums over the 8 routes), `flags` [`FLAG_ROW_SLICE`] only, `placement_version`
//! 0, and the word at byte 124 packs the receiving rank (bits 0-7), the world size (bits 8-15)
//! and the request's row count (bits 16-31). The rows carried are always the receiver's
//! partition, which [`ExchangeView::parse`] checks.
//!
//! **The partition** ([`row_partition`], glmrt's balanced partition): four contiguous ranges,
//! the first `rows % 4` ranks holding one row more.
//!
//! **Old and new peers.** A version-3 decoder checks `version == 3`, so it refuses every
//! version-4 frame by its version: an old rank never answers a reduce-scattered request with a
//! full plane, and an old coordinator never adds a row slice as a plane. Version-3 frames are
//! unchanged, so new coordinators and old ranks still exchange four-plane returns. This decoder
//! refuses a version-4 request or return without its flag, a version-3 one with a version-4
//! flag, an exchange frame in version 3, and a nonzero word at byte 124 in requests and returns.

use crate::error::WireError;
use crate::frame::{as_u32, checked_mul, parse_header, seal_in_place, verify_crc, write_header, HeaderFields};
pub use crate::frame::{FLAG_EXCHANGE_FP8, FLAG_REDUCE_SCATTER, FLAG_ROW_SLICE};
use crate::layout::{self, Dtype, SourceKind, Status, HIDDEN, SPARKS, TOPK};
use crate::naive::WireNaive;

/// The rows rank `rank` of `world` owns in an exchange of `rows` rows: `(first, count)`. Four
/// contiguous ranges in rank order; the first `rows % world` ranks hold one row more.
pub fn row_partition(rows: usize, world: usize, rank: usize) -> (usize, usize) {
    assert!(world > 0 && rank < world, "rank {rank} of {world}");
    let (base, extra) = (rows / world, rows % world);
    (rank * base + rank.min(extra), base + usize::from(rank < extra))
}

/// How a rank's rows travel to a peer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExchangeDtype {
    /// 4,096 BF16 values, 8,192 bytes a row: the proposed default (no more egress than a
    /// four-plane return, and an error close to it; the rank crate's README).
    Bf16,
    /// 4,096 E4M3 values of `v / scale` then the FP32 `scale = amax / 448`, 4,100 bytes a row
    /// (glmrt's exchange; about ten times the error of BF16).
    Fp8RowScaled,
}

impl ExchangeDtype {
    /// Bytes per row on the wire.
    pub fn row_bytes(self) -> usize {
        match self {
            ExchangeDtype::Bf16 => 2 * HIDDEN,
            ExchangeDtype::Fp8RowScaled => HIDDEN + 4,
        }
    }

    /// The payload dtype the exchange frame carries.
    pub fn wire(self) -> Dtype {
        match self {
            ExchangeDtype::Bf16 => Dtype::Bf16,
            ExchangeDtype::Fp8RowScaled => Dtype::Fp8E4m3RowScaled,
        }
    }

    /// The request flags that ask for a reduce-scatter with this exchange dtype.
    pub fn request_flags(self) -> u32 {
        match self {
            ExchangeDtype::Bf16 => FLAG_REDUCE_SCATTER,
            ExchangeDtype::Fp8RowScaled => FLAG_REDUCE_SCATTER | FLAG_EXCHANGE_FP8,
        }
    }

    /// The exchange dtype a request's flags ask for, or `None` for a four-plane request.
    pub fn from_request_flags(flags: u32) -> Option<ExchangeDtype> {
        if flags & FLAG_REDUCE_SCATTER == 0 {
            None
        } else if flags & FLAG_EXCHANGE_FP8 != 0 {
            Some(ExchangeDtype::Fp8RowScaled)
        } else {
            Some(ExchangeDtype::Bf16)
        }
    }

    /// `bf16` or `fp8` (configuration and trace lines).
    pub fn name(self) -> &'static str {
        match self {
            ExchangeDtype::Bf16 => "bf16",
            ExchangeDtype::Fp8RowScaled => "fp8",
        }
    }

    /// The dtype [`ExchangeDtype::name`] names.
    pub fn parse(s: &str) -> Option<ExchangeDtype> {
        match s {
            "bf16" => Some(ExchangeDtype::Bf16),
            "fp8" => Some(ExchangeDtype::Fp8RowScaled),
            _ => None,
        }
    }
}

/// What one exchange frame carries: rank `src`'s rows of rank `dst`'s partition, in the exchange
/// of request `request_id` (`rows` rows) for layer `layer_id`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ExchangeHeader {
    pub request_id: u64,
    pub layer_id: u32,
    pub src: usize,
    pub dst: usize,
    /// The request's row count.
    pub rows: usize,
    /// `dst`'s partition: its first row and its row count.
    pub first_row: usize,
    pub row_count: usize,
    pub dtype: ExchangeDtype,
}

impl ExchangeHeader {
    /// The frame `src` sends `dst`: `dst`'s partition of the request's `rows` rows.
    pub fn new(request_id: u64, layer_id: u32, rows: usize, src: usize, dst: usize, dtype: ExchangeDtype) -> Self {
        let (first_row, row_count) = row_partition(rows, SPARKS, dst);
        ExchangeHeader { request_id, layer_id, src, dst, rows, first_row, row_count, dtype }
    }

    /// Payload bytes: `row_count` rows.
    pub fn payload_len(&self) -> usize {
        self.row_count * self.dtype.row_bytes()
    }

    /// The whole frame's length.
    pub fn frame_len(&self) -> usize {
        layout::HEADER_LEN + self.payload_len()
    }

    /// Write the 128-byte header with L4 sequence `seq` into `out[..128]`, its CRC field zero:
    /// the caller writes the rows after it and seals the frame with
    /// [`crate::frame::seal_in_place`].
    pub fn write(&self, seq: u64, out: &mut [u8]) -> Result<(), WireError> {
        self.check()?;
        if out.len() < layout::HEADER_LEN {
            return Err(WireError::TooShort { need: layout::HEADER_LEN, got: out.len() });
        }
        let payload = checked_mul(self.row_count, self.dtype.row_bytes(), "payload_bytes")?;
        let mut h = Vec::with_capacity(layout::HEADER_LEN);
        write_header(
            &mut h,
            &HeaderFields {
                version: layout::VERSION_ROW_SHARD,
                kind: layout::KIND_EXCHANGE,
                request_id: self.request_id,
                placement_version: 0,
                layer_id: self.layer_id,
                row_count: as_u32(self.row_count, "row_count")?,
                dim: HIDDEN as u32,
                dtype: self.dtype.wire(),
                source_kind: SourceKind::Decode,
                route_count: TOPK as u32,
                row_descriptor_bytes: 0,
                route_bytes: 0,
                payload_bytes: payload as u64,
                logical_payload_bytes: payload as u64,
                wire_bytes: (layout::HEADER_LEN + payload) as u64,
                flags: FLAG_ROW_SLICE,
                row_stride_bytes: as_u32(self.dtype.row_bytes(), "row_stride_bytes")?,
                status: Status::Ok,
                executor_id: self.src as u64,
                token_position: self.first_row as u64,
                seq,
                reserved: self.dst as u32 | (SPARKS as u32) << 8 | (self.rows as u32) << 16,
            },
            WireNaive::NONE,
        );
        out[..layout::HEADER_LEN].copy_from_slice(&h);
        Ok(())
    }

    /// The header's invariants: ranks in the world and distinct, one row per rank at least, the
    /// row count within the word's 16 bits, and `dst`'s partition.
    fn check(&self) -> Result<(), WireError> {
        if self.src >= SPARKS || self.dst >= SPARKS || self.src == self.dst {
            return Err(WireError::DimMismatch { field: "exchange_ranks", want: SPARKS, got: self.src.max(self.dst) });
        }
        if self.rows < SPARKS || self.rows > u16::MAX as usize {
            return Err(WireError::DimMismatch { field: "exchange_rows", want: SPARKS, got: self.rows });
        }
        let (first, count) = row_partition(self.rows, SPARKS, self.dst);
        if self.first_row != first {
            return Err(WireError::DimMismatch { field: "exchange_first_row", want: first, got: self.first_row });
        }
        if self.row_count != count {
            return Err(WireError::DimMismatch { field: "exchange_row_count", want: count, got: self.row_count });
        }
        Ok(())
    }
}

/// Encode a whole exchange frame (header, `payload`, CRC unless disabled): for transports that
/// send owned buffers, and tests.
pub fn encode_exchange(h: &ExchangeHeader, payload: &[u8], seq: u64, naive: WireNaive) -> Result<Vec<u8>, WireError> {
    if payload.len() != h.payload_len() {
        return Err(WireError::DimMismatch { field: "exchange_payload", want: h.payload_len(), got: payload.len() });
    }
    let mut out = vec![0u8; h.frame_len()];
    h.write(seq, &mut out)?;
    out[layout::HEADER_LEN..].copy_from_slice(payload);
    seal_in_place(&mut out, naive);
    Ok(out)
}

/// An exchange frame validated where it lies: header, CRC (unless disabled for the process),
/// kind, flags, dtype and stride, the ranks, and that the rows are the receiver's partition.
/// The receiver still checks the request, layer, dtype and ranks against its own exchange, and
/// the L4 sequence against the link (`StreamReceiver::accept_seq(view.seq)`).
#[derive(Debug)]
pub struct ExchangeView<'a> {
    pub header: ExchangeHeader,
    pub seq: u64,
    /// `row_count` rows of `dtype.row_bytes()` bytes.
    pub payload: &'a [u8],
}

impl<'a> ExchangeView<'a> {
    pub fn parse(bytes: &'a [u8], naive: WireNaive) -> Result<Self, WireError> {
        let h = parse_header(bytes, naive)?;
        verify_crc(bytes, naive, h.seq)?;
        if h.kind != layout::KIND_EXCHANGE {
            return Err(WireError::BadKind(h.kind));
        }
        if h.status != Status::Ok {
            return Err(WireError::BadCode { field: "exchange status", code: h.status.code() });
        }
        if h.flags != FLAG_ROW_SLICE {
            return Err(WireError::BadCode { field: "exchange flags", code: h.flags });
        }
        let dtype = match h.dtype {
            Dtype::Bf16 => ExchangeDtype::Bf16,
            Dtype::Fp8E4m3RowScaled => ExchangeDtype::Fp8RowScaled,
            other => return Err(WireError::BadCode { field: "exchange payload dtype", code: other as u32 }),
        };
        let mismatch = |field: &'static str, want: usize, got: usize| Err(WireError::DimMismatch { field, want, got });
        if h.dim as usize != HIDDEN {
            return mismatch("exchange_dim", HIDDEN, h.dim as usize);
        }
        if h.row_stride_bytes as usize != dtype.row_bytes() {
            return mismatch("exchange_row_stride", dtype.row_bytes(), h.row_stride_bytes as usize);
        }
        if h.route_count as usize != TOPK || h.row_descriptor_bytes != 0 || h.route_bytes != 0 || h.placement_version != 0
        {
            return mismatch("exchange_header_fields", 0, 1);
        }
        let (dst, world, rows) = ((h.reserved & 0xff) as usize, ((h.reserved >> 8) & 0xff) as usize, (h.reserved >> 16) as usize);
        if world != SPARKS {
            return mismatch("exchange_world", SPARKS, world);
        }
        let src = h.executor_id;
        if src >= SPARKS as u64 || dst >= SPARKS || src as usize == dst {
            return mismatch("exchange_ranks", SPARKS, src.max(dst as u64) as usize);
        }
        let header = ExchangeHeader {
            request_id: h.request_id,
            layer_id: h.layer_id,
            src: src as usize,
            dst,
            rows,
            first_row: usize::try_from(h.token_position).unwrap_or(usize::MAX),
            row_count: h.row_count as usize,
            dtype,
        };
        header.check()?;
        let payload = checked_mul(header.row_count, dtype.row_bytes(), "payload_bytes")?;
        if h.payload_bytes as usize != payload || h.logical_payload_bytes as usize != payload {
            return mismatch("payload_bytes", payload, h.payload_bytes as usize);
        }
        let body = &bytes[layout::HEADER_LEN..];
        if body.len() != payload {
            return mismatch("body_len", payload, body.len());
        }
        Ok(ExchangeView { header, seq: h.seq, payload: body })
    }

    /// Row `i` of the rows carried, `dtype.row_bytes()` bytes.
    pub fn row(&self, i: usize) -> &'a [u8] {
        let rb = self.header.dtype.row_bytes();
        &self.payload[i * rb..(i + 1) * rb]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn partitions_cover_the_rows_once() {
        assert_eq!((0..4).map(|r| row_partition(1006, 4, r)).collect::<Vec<_>>(), vec![(0, 252), (252, 252), (504, 251), (755, 251)]);
        for rows in [4usize, 5, 16, 17, 18, 19, 33, 4096] {
            let mut next = 0;
            for r in 0..4 {
                let (s, c) = row_partition(rows, 4, r);
                assert_eq!(s, next);
                assert!(c >= 1);
                next += c;
            }
            assert_eq!(next, rows);
        }
    }

    #[test]
    fn exchange_dtypes_map_to_flags_and_back() {
        for d in [ExchangeDtype::Bf16, ExchangeDtype::Fp8RowScaled] {
            assert_eq!(ExchangeDtype::from_request_flags(d.request_flags()), Some(d));
            assert_eq!(ExchangeDtype::parse(d.name()), Some(d));
        }
        assert_eq!(ExchangeDtype::from_request_flags(0), None);
        assert_eq!(ExchangeDtype::from_request_flags(FLAG_EXCHANGE_FP8), None);
        assert_eq!((ExchangeDtype::Bf16.row_bytes(), ExchangeDtype::Fp8RowScaled.row_bytes()), (8192, 4100));
    }
}
