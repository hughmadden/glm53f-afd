//! I4 item 8 — non-blocking expert submit/collect API (ADVISOR-I4 §3.2 step 8).
//!
//! Classification: BOTH-RUNS (the API's non-blocking contract is not a trap
//! switch); the duplicate-refusal test reuses the L4 slot check, whose naive
//! double-count switch is exercised in `l4_integrity.rs`.

use std::collections::VecDeque;

use glm53f_wire::async_api::{ExpertClient, Ticket, Transport};
use glm53f_wire::frame::{HiddenRow, RequestFrame, ReturnFrame, ReturnRow, RouteEntry, RowDescriptor};
use glm53f_wire::layout::{SourceKind, Status, HIDDEN, SPARKS, TOPK};
use glm53f_wire::naive::WireNaive;
use glm53f_wire::WireError;

/// Test transport: a queue of return frames + counters proving non-blockingness.
#[derive(Default)]
struct QueueTransport {
    sent: Vec<RequestFrame>,
    inbox: VecDeque<ReturnFrame>,
    recv_calls: usize,
}

impl Transport for QueueTransport {
    fn send(&mut self, frame: &RequestFrame) -> Result<(), WireError> {
        self.sent.push(frame.clone());
        Ok(())
    }
    fn try_recv(&mut self) -> Option<ReturnFrame> {
        self.recv_calls += 1;
        self.inbox.pop_front()
    }
}

fn descs(rows: usize) -> Vec<RowDescriptor> {
    (0..rows)
        .map(|i| RowDescriptor {
            row_id: i as u64,
            source_kind: SourceKind::Decode,
            source_request_id: 1,
            token_position: i as u64,
            route_offset: (i * TOPK) as u32,
            route_count: TOPK as u32,
        })
        .collect()
}

fn routes(rows: usize) -> Vec<RouteEntry> {
    (0..rows * TOPK)
        .map(|i| RouteEntry {
            row_index: (i / TOPK) as u32,
            expert_id: (i % 256) as u32,
            gate_weight: 0.125,
        })
        .collect()
}

fn hidden_rows(rows: usize) -> Vec<HiddenRow> {
    (0..rows)
        .map(|_| HiddenRow { payload: vec![0u8; HIDDEN], scales: vec![127u8; HIDDEN / 32] })
        .collect()
}

fn partial(request_id: u64, layer_id: u32, executor: u64, rows: usize, value: u16) -> ReturnFrame {
    ReturnFrame {
        request_id,
        placement_version: 1,
        layer_id,
        executor_id: executor,
        token_position: 0,
        status: Status::Ok,
        flags: glm53f_wire::FLAG_SPARK_REDUCTION | glm53f_wire::FLAG_V41_COMPACT_BF16,
        route_count: TOPK,
        seq: 0,
        rows: (0..rows).map(|_| ReturnRow { codes: vec![value; HIDDEN] }).collect(),
    }
}

#[test]
fn submit_never_drains_the_receive_side() {
    let mut c = ExpertClient::new(QueueTransport::default(), HIDDEN, WireNaive::NONE);
    let t = c.submit(3, &descs(2), &routes(2), &hidden_rows(2)).unwrap();
    assert_eq!(t.sparks, SPARKS);
    assert_eq!(c.transport_mut().sent.len(), SPARKS, "one request frame per Spark");
    assert_eq!(
        c.transport_mut().recv_calls, 0,
        "submit must not touch the receive side (I5 overlap depends on it)"
    );
    assert_eq!(c.pending_len(), 1);
}

#[test]
fn collect_returns_none_until_every_spark_partial_arrives() {
    let mut c = ExpertClient::new(QueueTransport::default(), HIDDEN, WireNaive::NONE);
    let t = c.submit(0, &descs(1), &routes(1), &hidden_rows(1)).unwrap();
    for executor in 0..(SPARKS as u64 - 1) {
        c.transport_mut().inbox.push_back(partial(t.id, 0, executor, 1, 0x3F80)); // 1.0
    }
    assert_eq!(c.collect(&t).unwrap(), None, "3 of 4 partials is not complete");
    c.transport_mut().inbox.push_back(partial(t.id, 0, SPARKS as u64 - 1, 1, 0x3F80));
    let sum = c.collect(&t).unwrap().expect("complete after the 4th partial");
    assert_eq!(sum.len(), HIDDEN);
    assert!((sum[0] - SPARKS as f32).abs() < 1e-6, "FP32 sum of 4 x 1.0 = 4.0, got {}", sum[0]);
}

#[test]
fn two_batches_overlap_independently() {
    let mut c = ExpertClient::new(QueueTransport::default(), HIDDEN, WireNaive::NONE);
    let a = c.submit(0, &descs(1), &routes(1), &hidden_rows(1)).unwrap();
    let b = c.submit(1, &descs(1), &routes(1), &hidden_rows(1)).unwrap();
    assert_ne!(a.id, b.id);
    // B completes first (out-of-order across batches) — A must stay pending.
    for executor in 0..SPARKS as u64 {
        c.transport_mut().inbox.push_back(partial(b.id, 1, executor, 1, 0x4000)); // 2.0
    }
    let sb = c.collect(&b).unwrap().expect("B complete");
    assert!((sb[0] - 2.0 * SPARKS as f32).abs() < 1e-6);
    assert_eq!(c.collect(&a).unwrap(), None, "A must not be completed by B's frames");
    for executor in 0..SPARKS as u64 {
        c.transport_mut().inbox.push_back(partial(a.id, 0, executor, 1, 0x3F80));
    }
    let sa = c.collect(&a).unwrap().expect("A complete");
    assert!((sa[0] - SPARKS as f32).abs() < 1e-6);
    c.retire(&a);
    c.retire(&b);
    assert_eq!(c.pending_len(), 0);
}

#[test]
fn duplicate_partial_is_refused_not_double_counted() {
    let mut c = ExpertClient::new(QueueTransport::default(), HIDDEN, WireNaive::NONE);
    let t = c.submit(0, &descs(1), &routes(1), &hidden_rows(1)).unwrap();
    for executor in 0..SPARKS as u64 {
        c.transport_mut().inbox.push_back(partial(t.id, 0, executor, 1, 0x3F80));
    }
    // Replay executor 0's partial: the slot layer must refuse it.
    c.transport_mut().inbox.push_back(partial(t.id, 0, 0, 1, 0x3F80));
    let err = c.collect(&t).unwrap_err();
    assert!(matches!(err, WireError::SlotFilled { .. }), "got {err:?}");
}

#[test]
fn return_for_an_untracked_request_fails_loud() {
    let mut c = ExpertClient::new(QueueTransport::default(), HIDDEN, WireNaive::NONE);
    let t = c.submit(0, &descs(1), &routes(1), &hidden_rows(1)).unwrap();
    c.transport_mut().inbox.push_back(partial(t.id + 99, 0, 0, 1, 0x3F80));
    let err = c.collect(&t).unwrap_err();
    assert!(matches!(err, WireError::UnknownRequest { .. }), "got {err:?}");
}

#[test]
fn ticket_is_a_plain_handle() {
    let t = Ticket { id: 7, layer_id: 2, rows: 3, sparks: SPARKS };
    assert_eq!(t.clone(), t);
}
