//! Shared fixtures + L4 session harness for the glm53f-wire two-run trap suite.
//!
//! Fixture values are small non-negative integers (`value(t, s, i) < 256`) so
//! every Spark partial is BF16-exact and the golden coordinator FP32 sums are
//! exact integers — any corruption that lands in a sum changes it. The session
//! harness drives `StreamReceiver` + `CoordinatorSum` the way I5 will (decode,
//! dispositions, bounded retry with the clean expected frame), so the
//! corruption tests exercise the real ladder, not a mock.
#![allow(dead_code)]

use glm53f_wire::error::{Disposition, WireError};
use glm53f_wire::frame::{self, HiddenRow, RequestFrame, ReturnFrame, ReturnRow, RouteEntry, RowDescriptor};
use glm53f_wire::layout::{self, SourceKind, Status};
use glm53f_wire::naive::WireNaive;
use glm53f_wire::{bf16, CoordinatorSum, StreamReceiver, StreamSender};

/// Deterministic partial value: BF16-exact integer in 0..251.
pub fn value(token: usize, spark: usize, i: usize) -> f32 {
    ((token * 7 + spark * 13 + i * 3) % 251) as f32
}

/// Golden coordinator FP32 sum row for one token (integers, exact).
pub fn golden_row(token: usize, hidden: usize) -> Vec<f32> {
    (0..hidden)
        .map(|i| (0..layout::SPARKS).map(|s| value(token, s, i)).sum())
        .collect()
}

/// One Spark's compact return frame (single token row, pre-summed partial).
pub fn spark_return_frame(
    request_id: u64,
    layer_id: u32,
    token_position: u64,
    executor_id: u64,
) -> ReturnFrame {
    let codes = (0..layout::HIDDEN)
        .map(|i| {
            bf16::f32_to_bf16_rne(value(token_position as usize, executor_id as usize, i))
        })
        .collect();
    ReturnFrame {
        request_id,
        placement_version: 1,
        layer_id,
        executor_id,
        token_position,
        status: Status::Ok,
        flags: frame::FLAG_RETURN_REQUIRED,
        route_count: layout::TOPK,
        seq: 0,
        rows: vec![ReturnRow { codes }],
    }
}

/// Deterministic hidden payload for a token row.
pub fn hidden_payload(token: usize) -> Vec<u8> {
    (0..layout::HIDDEN)
        .map(|i| ((token * 31 + i * 7 + 17) & 0xFF) as u8)
        .collect()
}

/// Canonical 128 UE8M0 K32 scale bytes for a token row.
pub fn canonical_scales(token: usize) -> Vec<u8> {
    (0..layout::HIDDEN / layout::K32)
        .map(|j| ((token * 13 + j * 5 + 3) & 0xFF) as u8)
        .collect()
}

/// Hidden row with the scale region the selected layout variant needs
/// (canonical 128 B; 256 duplicated bytes for the K-slip trap; 4-B FP32 row
/// scale for the `Fp8E4m3RowScaled` trap). Payload + scales travel verbatim.
pub fn hidden_row_fixture(token: usize, naive: WireNaive) -> HiddenRow {
    let payload = hidden_payload(token);
    // Precedence matches `layout::hidden_row_bytes` (ROW_SCALED first).
    let scales = if naive.has(WireNaive::ROW_SCALED_DTYPE) {
        1.0f32.to_le_bytes().to_vec()
    } else if naive.has(WireNaive::K16_SCALES) {
        canonical_scales(token).iter().flat_map(|&s| [s, s]).collect()
    } else {
        canonical_scales(token)
    };
    HiddenRow { payload, scales }
}

/// Route entry fixture (weight is a multiple of 1/8 — BF16-exact).
pub fn route_fixture(row: usize, e: usize) -> RouteEntry {
    RouteEntry {
        row_index: row as u32,
        expert_id: (1000 + row * layout::TOPK + e) as u32,
        gate_weight: ((e + 1) as f32) * 0.125,
    }
}

/// Pinned byte-layout fixture (used by the exact-byte trap negatives).
pub fn pinned_route() -> RouteEntry {
    RouteEntry { row_index: 0x0102_0304, expert_id: 0x1112_1314, gate_weight: 1.0f32 }
}

/// Pinned byte-layout fixture for the row descriptor.
pub fn pinned_row() -> RowDescriptor {
    RowDescriptor {
        row_id: 0x0102_0304_0506_0708,
        source_kind: SourceKind::Decode,
        source_request_id: 0x1112_1314_1516_1718,
        token_position: 0x2122_2324_2526_2728,
        route_offset: 0x3132_3334,
        route_count: 0x4142_4344,
    }
}

/// Multi-token request frame with grouped top-8 routes per row.
pub fn request_fixture(rows: usize, naive: WireNaive) -> RequestFrame {
    let mut descriptors = Vec::with_capacity(rows);
    let mut routes = Vec::with_capacity(rows * layout::TOPK);
    let mut hidden_rows = Vec::with_capacity(rows);
    for t in 0..rows {
        descriptors.push(RowDescriptor {
            row_id: 0xA000 + t as u64,
            source_kind: SourceKind::Decode,
            source_request_id: 77,
            token_position: t as u64,
            route_offset: (t * layout::TOPK) as u32,
            route_count: layout::TOPK as u32,
        });
        for e in 0..layout::TOPK {
            routes.push(route_fixture(t, e));
        }
        hidden_rows.push(hidden_row_fixture(t, naive));
    }
    RequestFrame {
        request_id: 5,
        placement_version: 1,
        layer_id: 3,
        executor_id: 2,
        source_kind: SourceKind::Decode,
        token_position: 0,
        flags: 0,
        seq: 0,
        rows: descriptors,
        routes,
        hidden_rows,
    }
}

/// Stamp a list of return frames into clean wire frames (seq = index).
pub fn session_wire(frames: &[ReturnFrame], naive: WireNaive) -> Vec<Vec<u8>> {
    let mut sender = StreamSender::new(naive);
    frames.iter().map(|f| sender.encode_return(f).expect("encode return")).collect()
}

/// Session configuration for [`run_return_session`].
pub struct SessionCfg {
    pub naive: WireNaive,
    pub rows: usize,
    pub hidden: usize,
    pub attempts_cap: usize,
}

/// What a delivery session produced.
#[derive(Debug)]
pub struct RunOutcome {
    /// Every error raised (receiver or coordinator slot layer) = detections.
    pub detected: Vec<WireError>,
    /// Final FP32 sum rows flattened (empty unless complete).
    pub sums: Vec<f32>,
    pub complete: bool,
    pub fail_loud: Option<WireError>,
    pub attempts: usize,
}

impl RunOutcome {
    /// Flattened golden for comparison.
    pub fn sums_match(&self, goldens: &[Vec<f32>]) -> bool {
        if !self.complete {
            return false;
        }
        let flat: Vec<f32> = goldens.iter().flatten().copied().collect();
        self.sums == flat
    }
}

/// Drive one return session: deliver `deliveries` (as the network produced
/// them — corrupt, truncated, duplicated, shuffled) against the receiver +
/// coordinator, retransmitting `clean[expected]` on Retry-class errors (the
/// bounded ladder of ADVISOR-I4 §3.2 item 6).
pub fn run_return_session(
    cfg: &SessionCfg,
    deliveries: &[Vec<u8>],
    clean: &[Vec<u8>],
) -> RunOutcome {
    let mut rx = StreamReceiver::new(cfg.naive);
    let mut sum = CoordinatorSum::new(cfg.rows, cfg.hidden, cfg.naive);
    let mut pending: Vec<Vec<u8>> = deliveries.to_vec();
    let mut detected: Vec<WireError> = Vec::new();
    let mut fail_loud: Option<WireError> = None;
    let mut attempts = 0usize;

    while attempts < cfg.attempts_cap {
        let bytes = match pending.first().cloned() {
            Some(b) => b,
            None => break,
        };
        pending.remove(0);
        attempts += 1;
        match rx.accept(&bytes) {
            Ok(frame::Frame::Return(f)) => {
                if let Err(e) = sum.accumulate(&f) {
                    detected.push(e.clone());
                    if e.disposition() == Disposition::FailLoud {
                        fail_loud = Some(e);
                        break;
                    }
                }
            }
            Ok(frame::Frame::Request(_)) => {
                fail_loud = Some(WireError::BadKind(layout::KIND_REQUEST));
                break;
            }
            Err(e) => {
                detected.push(e.clone());
                match e.disposition() {
                    Disposition::Retry => {
                        // Retransmit the expected frame first, then (only if
                        // the rejected frame still claims a FUTURE sequence)
                        // the rejected one — it may just have arrived early.
                        // Frames without a readable/later seq are dropped: the
                        // sender's retransmission replaces them.
                        let expected = rx.expected();
                        let requeue = frame::frame_seq(&bytes).filter(|&s| s > expected);
                        if requeue.is_some() {
                            pending.insert(0, bytes);
                        }
                        if (expected as usize) < clean.len() {
                            pending.insert(0, clean[expected as usize].clone());
                        }
                    }
                    Disposition::DropIdempotent => {}
                    Disposition::FailLoud => {
                        fail_loud = Some(e);
                        break;
                    }
                }
            }
        }
    }

    let complete = sum.is_complete();
    let sums = if complete {
        sum.result().map(|s| s.to_vec()).unwrap_or_default()
    } else {
        Vec::new()
    };
    RunOutcome { detected, sums, complete, fail_loud, attempts }
}

/// Flip one bit at byte `pos`.
pub fn flip_bit(bytes: &mut [u8], pos: usize) {
    bytes[pos] ^= 0x01;
}
