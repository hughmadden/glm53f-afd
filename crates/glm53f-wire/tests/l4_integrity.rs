//! Two-run trap suite — L4 integrity ladder (I4 item 6).
//!
//! Injects **bitflip, reorder, truncation and duplicate** into return streams
//! and asserts every one is detected and retried or fails loud, and that NONE
//! produces a silent wrong sum: `n_corruption_never_lands_in_sum_undetected`
//! computes the coordinator FP32 sum for every injection and requires the sum
//! to be exactly the golden integer sum whenever it lands — a wrong sum is
//! only survivable together with a raised detection (and with the ladder the
//! retry then still delivers the golden sum).
//!
//! # Two-run classification (suite convention)
//!
//! NEGATIVES — FAIL under `GLM53F_SPIKE_NAIVE=1` (alias `GLM53F_WIRE_NAIVE=1`),
//! PASS on the correct impl (they call the env-default entry points):
//!   * `n_header_bitflip_detected` (kills body-only CRC / no-CRC)
//!   * `n_payload_bitflip_detected` (kills "trust the transport")
//!   * `n_seq_bitflip_detected` (kills CRC-without-seq-coverage + seq-ignored)
//!   * `n_reorder_never_misassigns_token_sums` (kills seq-ignored + arrival-
//!     order accumulation)
//!   * `n_duplicate_never_double_counts` (kills duplicate double-accumulation)
//!   * `n_corruption_never_lands_in_sum_undetected` (the item-3 umbrella)
//!
//! BOTH RUNS (explicit `naive=` flags):
//!   `bitflip_every_byte_position_detected`,
//!   `truncation_at_every_cut_rejected_without_panic`,
//!   `trailing_garbage_fails_loud`, `retry_recovers_after_corruption`,
//!   `permanent_corruption_fails_loud_after_retry_budget`,
//!   `duplicate_detected_at_seq_and_slot_layers`,
//!   `out_of_order_delivery_detected_before_accept`,
//!   `coordinator_fp32_sum_matches_golden`, `disposition_classification`
//!   (plus `tests/wire_layout.rs`, both runs).

mod common;

use common::*;
use glm53f_wire::error::{Disposition, WireError};
use glm53f_wire::frame::{self, Frame};
use glm53f_wire::layout::{self, hdr};
use glm53f_wire::naive::{naive_from_env, WireNaive};
use glm53f_wire::{retry_until, CoordinatorSum, SlotKey, StreamReceiver};

// ---------------------------------------------------------------------------
// NEGATIVES (flip behind the env-default naive flag)
// ---------------------------------------------------------------------------

/// Bitflip in a HEADER field must be detected (kills body-only checksums and
/// checksum-free "reliable transport" decoders).
#[test]
fn n_header_bitflip_detected() {
    let f = spark_return_frame(1, 2, 0, 0);
    let mut bytes = frame::encode_return_env(&f).expect("encode");
    bytes[hdr::LAYER_ID] ^= 0x01;
    let mut rx = StreamReceiver::new_env();
    assert!(rx.accept(&bytes).is_err(), "header bitflip must be detected");
}

/// Bitflip in the BF16 payload must be detected (kills CRC-unchecked decode).
#[test]
fn n_payload_bitflip_detected() {
    let f = spark_return_frame(1, 2, 0, 0);
    let mut bytes = frame::encode_return_env(&f).expect("encode");
    flip_bit(&mut bytes, layout::HEADER_LEN + 4000);
    let mut rx = StreamReceiver::new_env();
    assert!(rx.accept(&bytes).is_err(), "payload bitflip must be detected");
}

/// Bitflip in the sequence field must be detected (seq is inside the CRC
/// coverage; kills stale/replayed frame acceptance).
#[test]
fn n_seq_bitflip_detected() {
    let f = spark_return_frame(1, 2, 0, 0);
    let mut bytes = frame::encode_return_env(&f).expect("encode");
    flip_bit(&mut bytes, hdr::SEQ);
    let mut rx = StreamReceiver::new_env();
    assert!(rx.accept(&bytes).is_err(), "seq bitflip must be detected");
}

/// Reordered delivery must raise a detection AND land partials only in their
/// own token's FP32 sum (kills seq-ignored receivers + arrival-order
/// accumulation: token 1's partials silently summed as token 0).
#[test]
fn n_reorder_never_misassigns_token_sums() {
    let naive = naive_from_env();
    // Canonical sender order is token-major: [t0s0 t0s1 t0s2 t0s3 t1s0 ...].
    let frames: Vec<_> = (0..2u64)
        .flat_map(|t| (0..layout::SPARKS as u64).map(move |s| spark_return_frame(1, 2, t, s)))
        .collect();
    let clean = session_wire(&frames, naive);
    let mut deliveries = clean.clone();
    deliveries.rotate_left(layout::SPARKS); // network hands us token 1 first

    let cfg = SessionCfg { naive, rows: 2, hidden: layout::HIDDEN, attempts_cap: 512 };
    let outcome = run_return_session(&cfg, &deliveries, &clean);
    assert!(!outcome.detected.is_empty(), "reorder produced no detection");
    let gold = vec![golden_row(0, layout::HIDDEN), golden_row(1, layout::HIDDEN)];
    assert!(outcome.sums_match(&gold), "reorder misassigned token sums: wrong FP32 result");
}

/// A duplicate frame must be detected and must never be summed twice (kills
/// duplicate double-accumulation: 2x one Spark's partial silently lands).
#[test]
fn n_duplicate_never_double_counts() {
    let naive = naive_from_env();
    let frames: Vec<_> = (0..layout::SPARKS as u64)
        .map(|s| spark_return_frame(1, 2, 0, s))
        .collect();
    let mut clean = session_wire(&frames, naive);
    let mut deliveries = clean.clone();
    deliveries.push(clean[layout::SPARKS - 1].clone()); // duplicate of the last
    clean.push(vec![]); // no clean copy for the duplicate's "slot"

    let cfg = SessionCfg { naive, rows: 1, hidden: layout::HIDDEN, attempts_cap: 256 };
    let outcome = run_return_session(&cfg, &deliveries, &clean);
    assert!(!outcome.detected.is_empty(), "duplicate produced no detection");
    let gold = vec![golden_row(0, layout::HIDDEN)];
    assert!(outcome.sums_match(&gold), "duplicate double-counted into the FP32 sum");
}

/// The item-3 umbrella: for EVERY injected bitflip / truncation / duplicate /
/// reorder, the coordinator FP32 sum is either exactly the golden sum or a
/// detection was raised — corruption may never land in the result undetected.
#[test]
fn n_corruption_never_lands_in_sum_undetected() {
    let naive = naive_from_env();
    let frames: Vec<_> = (0..layout::SPARKS as u64)
        .map(|s| spark_return_frame(1, 2, 0, s))
        .collect();
    let clean = session_wire(&frames, naive);
    let gold = vec![golden_row(0, layout::HIDDEN)];
    let cfg = SessionCfg { naive, rows: 1, hidden: layout::HIDDEN, attempts_cap: 400 };

    let mut cases: Vec<(String, Vec<Vec<u8>>)> = Vec::new();

    // Bitflips: header identity fields, seq, and payload at spread positions.
    for k in 0..clean.len() {
        for pos in [
            hdr::LAYER_ID,
            hdr::TOKEN_POSITION,
            hdr::EXECUTOR_ID,
            hdr::SEQ,
            layout::HEADER_LEN + 7,
            layout::HEADER_LEN + 4000,
            layout::HEADER_LEN + 8191,
        ] {
            let mut d = clean.clone();
            flip_bit(&mut d[k], pos);
            cases.push((format!("bitflip f{k} @{pos}"), d));
        }
    }
    // Truncations: header cuts and payload cuts.
    for (k, c) in clean.iter().enumerate() {
        for cut in [1usize, 96, 127, 300, c.len() / 2, c.len() - 1] {
            let mut d = clean.clone();
            d[k] = c[..cut].to_vec();
            cases.push((format!("truncate f{k} @{cut}"), d));
        }
    }
    // Duplicates at every position.
    for k in 0..clean.len() {
        let mut d = clean.clone();
        d.insert(k, clean[k].clone());
        cases.push((format!("duplicate f{k}"), d));
    }
    // Reorders (frame-level permutations of the 4 Spark returns).
    cases.push(("reverse".to_string(), clean.iter().rev().cloned().collect()));
    let mut rotated = clean.clone();
    rotated.rotate_left(1);
    cases.push(("rotate-left-1".to_string(), rotated));
    let mut swapped = clean.clone();
    swapped.swap(0, 1);
    cases.push(("swap-0-1".to_string(), swapped));

    for (name, deliveries) in &cases {
        let outcome = run_return_session(&cfg, deliveries, &clean);
        let silent_wrong_sum =
            outcome.complete && !outcome.sums_match(&gold) && outcome.detected.is_empty();
        assert!(!silent_wrong_sum, "{name}: SILENT WRONG SUM landed in the FP32 result");
        assert!(!outcome.detected.is_empty(), "{name}: corruption produced no detection");
        assert!(
            !outcome.complete || outcome.sums_match(&gold),
            "{name}: wrong sum in the FP32 result after the ladder"
        );
    }
}

// ---------------------------------------------------------------------------
// BOTH RUNS (explicit flags)
// ---------------------------------------------------------------------------

/// Single-bit flips at EVERY byte position of a return frame and a request
/// frame are detected — kills weak checksums with blind spots.
#[test]
fn bitflip_every_byte_position_detected() {
    let naive = WireNaive::NONE;
    let ret = frame::encode_return(&spark_return_frame(1, 2, 0, 0), naive).expect("encode");
    let req = frame::encode_request(&request_fixture(1, naive), naive).expect("encode");
    for (tag, good) in [("return", &ret), ("request", &req)] {
        for pos in 0..good.len() {
            let mut b = good.clone();
            flip_bit(&mut b, pos);
            assert!(
                frame::decode_frame(&b, naive).is_err(),
                "{tag}: undetected bitflip at byte {pos}"
            );
        }
    }
}

/// Truncation at EVERY cut length is rejected without panicking — kills
/// length-trusting decoders (OOB reads / short-frame acceptance).
#[test]
fn truncation_at_every_cut_rejected_without_panic() {
    let naive = WireNaive::NONE;
    let good = frame::encode_return(&spark_return_frame(1, 2, 0, 0), naive).expect("encode");
    for cut in 0..good.len() {
        assert!(
            frame::decode_frame(&good[..cut], naive).is_err(),
            "undetected truncation at cut {cut}"
        );
    }
}

/// Trailing garbage after a frame is a stream-desync fail-loud, never a
/// silently accepted frame.
#[test]
fn trailing_garbage_fails_loud() {
    let naive = WireNaive::NONE;
    let mut good = frame::encode_return(&spark_return_frame(1, 2, 0, 0), naive).expect("encode");
    good.extend_from_slice(&[0xDE, 0xAD, 0xBE, 0xEF]);
    match frame::decode_frame(&good, naive) {
        Err(WireError::TrailingBytes { .. }) => {}
        other => panic!("expected TrailingBytes, got {other:?}"),
    }
}

/// "Detected AND retried": after a corrupt delivery the retransmission is
/// accepted and the coordinator sum is exactly golden.
#[test]
fn retry_recovers_after_corruption() {
    let naive = WireNaive::NONE;
    let frames: Vec<_> = (0..layout::SPARKS as u64)
        .map(|s| spark_return_frame(1, 2, 0, s))
        .collect();
    let clean = session_wire(&frames, naive);
    let mut deliveries = clean.clone();
    flip_bit(&mut deliveries[1], layout::HEADER_LEN + 4000);
    let cfg = SessionCfg { naive, rows: 1, hidden: layout::HIDDEN, attempts_cap: 64 };
    let outcome = run_return_session(&cfg, &deliveries, &clean);
    assert!(
        outcome.detected.iter().any(|e| matches!(e, WireError::Corrupt { .. })),
        "corruption not detected"
    );
    let gold = vec![golden_row(0, layout::HIDDEN)];
    assert!(outcome.sums_match(&gold), "retry did not recover the golden sum");
}

/// "…or fails loud": when every retransmission is still corrupt the retry
/// budget is exhausted and the ladder reports `RetryExhausted` (FailLoud).
#[test]
fn permanent_corruption_fails_loud_after_retry_budget() {
    let naive = WireNaive::NONE;
    let clean = session_wire(
        &(0..layout::SPARKS as u64).map(|s| spark_return_frame(1, 2, 0, s)).collect::<Vec<_>>(),
        naive,
    );
    let mut rx = StreamReceiver::new(naive);
    assert!(rx.accept(&clean[0]).is_ok());
    let mut corrupt = clean[1].clone();
    flip_bit(&mut corrupt, layout::HEADER_LEN + 100);
    let outcome = retry_until(4, || rx.accept(&corrupt));
    match outcome {
        Err(WireError::RetryExhausted { attempts, cause }) => {
            assert_eq!(attempts, 3);
            assert_eq!(cause.disposition(), Disposition::Retry);
        }
        other => panic!("expected RetryExhausted, got {other:?}"),
    }
    let e = retry_until(4, || rx.accept(&corrupt)).unwrap_err();
    assert_eq!(e.disposition(), Disposition::FailLoud, "exhaustion must fail loud");
}

/// Duplicates are detected at BOTH layers: the sequence policy (`Duplicate`)
/// and the coordinator slot (`SlotFilled`) — a double count cannot land even
/// if one layer were bypassed.
#[test]
fn duplicate_detected_at_seq_and_slot_layers() {
    let naive = WireNaive::NONE;
    let frames: Vec<_> = (0..layout::SPARKS as u64)
        .map(|s| spark_return_frame(1, 2, 0, s))
        .collect();
    let clean = session_wire(&frames, naive);
    let mut rx = StreamReceiver::new(naive);
    let mut accepted = Vec::new();
    for w in &clean {
        match rx.accept(w) {
            Ok(Frame::Return(f)) => accepted.push(f),
            other => panic!("expected accept, got {other:?}"),
        }
    }
    // Layer 1: sequence policy.
    match rx.accept(&clean[layout::SPARKS - 1]) {
        Err(WireError::Duplicate { seq, expected }) => {
            assert_eq!(seq, (layout::SPARKS - 1) as u64);
            assert_eq!(expected, layout::SPARKS as u64);
        }
        other => panic!("expected Duplicate, got {other:?}"),
    }
    // Layer 2: coordinator slot idempotence.
    let mut sum = CoordinatorSum::new(1, layout::HIDDEN, naive);
    for f in &accepted {
        sum.accumulate(f).expect("accumulate");
    }
    match sum.accumulate(&accepted[layout::SPARKS - 1]) {
        Err(WireError::SlotFilled { slot }) => {
            assert_eq!(slot.executor_id, (layout::SPARKS - 1) as u64);
        }
        other => panic!("expected SlotFilled, got {other:?}"),
    }
    let gold = golden_row(0, layout::HIDDEN);
    assert_eq!(sum.result().expect("complete"), gold.as_slice());
}

/// Out-of-order delivery is rejected BEFORE acceptance (`OutOfOrder`, Retry
/// class) — the expected sequence gates every frame.
#[test]
fn out_of_order_delivery_detected_before_accept() {
    let naive = WireNaive::NONE;
    let clean = session_wire(
        &(0..layout::SPARKS as u64).map(|s| spark_return_frame(1, 2, 0, s)).collect::<Vec<_>>(),
        naive,
    );
    let mut rx = StreamReceiver::new(naive);
    match rx.accept(&clean[1]) {
        Err(e @ WireError::OutOfOrder { expected: 0, got: 1 }) => {
            assert_eq!(e.disposition(), Disposition::Retry);
        }
        other => panic!("expected OutOfOrder(0,1), got {other:?}"),
    }
    assert!(rx.accept(&clean[0]).is_ok());
    assert!(rx.accept(&clean[1]).is_ok());
    assert_eq!(rx.expected(), 2);
}

/// The coordinator FP32 sum of 4 pre-summed BF16 partials equals the golden
/// integer sum exactly (two tokens).
#[test]
fn coordinator_fp32_sum_matches_golden() {
    let naive = WireNaive::NONE;
    let frames: Vec<_> = (0..2u64)
        .flat_map(|t| (0..layout::SPARKS as u64).map(move |s| spark_return_frame(1, 2, t, s)))
        .collect();
    let clean = session_wire(&frames, naive);
    let cfg = SessionCfg { naive, rows: 2, hidden: layout::HIDDEN, attempts_cap: 64 };
    let outcome = run_return_session(&cfg, &clean, &clean);
    let gold = vec![golden_row(0, layout::HIDDEN), golden_row(1, layout::HIDDEN)];
    assert!(outcome.complete, "window not complete");
    assert!(outcome.sums_match(&gold), "FP32 sum drifted from golden");
}

/// The disposition policy itself: Retry for recoverable corruption/gaps,
/// DropIdempotent for detected duplicates, FailLoud for protocol mismatch and
/// budget exhaustion.
#[test]
fn disposition_classification() {
    assert_eq!(
        WireError::Corrupt { seq: 0, want: 1, got: 2 }.disposition(),
        Disposition::Retry
    );
    assert_eq!(
        WireError::Truncated { declared: 10, got: 5 }.disposition(),
        Disposition::Retry
    );
    assert_eq!(
        WireError::TooShort { need: 128, got: 40 }.disposition(),
        Disposition::Retry
    );
    assert_eq!(
        WireError::OutOfOrder { expected: 1, got: 3 }.disposition(),
        Disposition::Retry
    );
    assert_eq!(
        WireError::Duplicate { seq: 1, expected: 2 }.disposition(),
        Disposition::DropIdempotent
    );
    assert_eq!(
        WireError::SlotFilled {
            slot: SlotKey { request_id: 1, layer_id: 2, token_position: 0, executor_id: 3 }
        }
        .disposition(),
        Disposition::DropIdempotent
    );
    assert_eq!(WireError::BadMagic([0; 8]).disposition(), Disposition::FailLoud);
    assert_eq!(WireError::BadVersion(2).disposition(), Disposition::FailLoud);
    assert_eq!(WireError::BadKind(9).disposition(), Disposition::FailLoud);
    assert_eq!(WireError::BadHeaderLen(96).disposition(), Disposition::FailLoud);
    assert_eq!(
        WireError::TrailingBytes { declared: 4, got: 8 }.disposition(),
        Disposition::FailLoud
    );
    assert_eq!(
        WireError::DimMismatch { field: "x", want: 1, got: 2 }.disposition(),
        Disposition::FailLoud
    );
    assert_eq!(
        WireError::UnpresummedReturn { flags: 0 }.disposition(),
        Disposition::FailLoud
    );
    assert_eq!(
        WireError::Incomplete { filled: 2, need: 4 }.disposition(),
        Disposition::FailLoud
    );
    assert_eq!(
        WireError::RetryExhausted {
            attempts: 3,
            cause: Box::new(WireError::Corrupt { seq: 0, want: 1, got: 2 })
        }
        .disposition(),
        Disposition::FailLoud
    );
}
