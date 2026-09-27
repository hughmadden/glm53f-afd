//! Two-run trap suite — wire layout + row accounting (I4 item 5).
//!
//! Pins the two ADVISOR-I4 §3.2 item 5 contracts in TESTS, not comments:
//! request row = **4,360 B** and compact return = **8,192 B** per token per
//! Spark (route-count independent — the A6 per-route FP32 wall must never
//! return), plus the byte-exact ds41rt v3 row descriptor / route entry / dtype
//! code wire surface.
//!
//! # Two-run classification (suite convention)
//!
//! NEGATIVES — FAIL under `GLM53F_SPIKE_NAIVE=1` (alias `GLM53F_WIRE_NAIVE=1`),
//! PASS on the correct impl (they call the env-default entry points):
//!   * `n_request_row_is_4360_bytes`
//!   * `n_return_row_is_8192_bytes`
//!   * `n_return_bytes_are_route_count_independent`
//!   * `n_row_descriptor_wire_bytes_pinned`
//!   * `n_route_entry_wire_bytes_pinned`
//!   * `n_dtype_wire_codes_pinned`
//!   * `n_crc32c_matches_rfc3720_vector`
//!   * `n_bf16_rounds_to_nearest_even`
//!
//! BOTH RUNS (explicit `naive=` flags / pure constants):
//!   `const_and_runtime_row_guards`, `k32_hidden_row_formula_and_multiples_of_32`,
//!   `request_body_is_4360_per_row`, `route_entries_map_by_descriptor_offsets`,
//!   `request_roundtrip_multirow_bit_exact`, `return_roundtrip_bit_exact`,
//!   `return_frame_requires_presummed_compact_flags`,
//!   `protocol_mismatch_fails_loud_before_crc`,
//!   `crc32c_table_matches_bitwise_reference`,
//!   `every_naive_bit_is_a_killable_wrong_impl`
//!   (plus `tests/l4_integrity.rs`, both runs).

mod common;

use common::*;
use glm53f_wire::bf16;
use glm53f_wire::crc32c;
use glm53f_wire::error::WireError;
use glm53f_wire::frame::{self, Frame};
use glm53f_wire::layout::{self, hdr, Dtype};
use glm53f_wire::naive::{naive_from_env, WireNaive};
use glm53f_wire::{StreamReceiver, StreamSender};

// ---------------------------------------------------------------------------
// NEGATIVES (flip behind the env-default naive flag)
// ---------------------------------------------------------------------------

/// Trap: request row drift (K16 scale slip -> 4,488; `Fp8E4m3RowScaled` ->
/// 4,236; descriptor/route width drift). Kills any wrong request row layout:
/// the body of a 1-token request frame must be exactly 4,360 B.
#[test]
fn n_request_row_is_4360_bytes() {
    let naive = naive_from_env();
    let f = request_fixture(1, naive);
    let bytes = frame::encode_request_env(&f).expect("encode request");
    assert_eq!(layout::request_row_bytes_env(), 4360, "request row drift");
    assert_eq!(bytes.len() - layout::HEADER_LEN, 4360, "request frame body drift");
}

/// Trap: FP32 compact return (16,384 B) or per-route return rows (A6 wall).
/// Kills any return that is not one 8,192-B BF16 row per token per Spark.
#[test]
fn n_return_row_is_8192_bytes() {
    let f = spark_return_frame(1, 2, 0, 0);
    let bytes = frame::encode_return_env(&f).expect("encode return");
    assert_eq!(layout::return_row_bytes_env(layout::TOPK), 8192, "return row drift");
    assert_eq!(bytes.len() - layout::HEADER_LEN, 8192, "return frame body drift");
}

/// Trap: per-route return accounting (return size scaling with route count).
/// Kills the 24.6 MB/token design (ADVISOR-I4 A6): 8 KB/Spark/token whether
/// one route or top-8 fired.
#[test]
fn n_return_bytes_are_route_count_independent() {
    assert_eq!(layout::return_row_bytes_env(1), 8192, "return bytes with 1 route");
    assert_eq!(layout::return_row_bytes_env(layout::TOPK), 8192, "return bytes with top-8");
}

/// Trap: row-descriptor packing drift / non-LE. Kills field swaps
/// (`source_request_id` <-> `token_position`) and offset drift against the
/// ds41rt v3 40-B descriptor.
#[test]
fn n_row_descriptor_wire_bytes_pinned() {
    let wire = frame::row_descriptor_wire(&pinned_row(), naive_from_env());
    #[rustfmt::skip]
    let expected: [u8; 40] = [
        0x08, 0x07, 0x06, 0x05, 0x04, 0x03, 0x02, 0x01, // row_id
        0x01, 0x00, // source_kind = Decode
        0x00, 0x00, // pad
        0x18, 0x17, 0x16, 0x15, 0x14, 0x13, 0x12, 0x11, // source_request_id
        0x28, 0x27, 0x26, 0x25, 0x24, 0x23, 0x22, 0x21, // token_position
        0x34, 0x33, 0x32, 0x31, // route_offset
        0x44, 0x43, 0x42, 0x41, // route_count
        0x00, 0x00, 0x00, 0x00, // pad
    ];
    assert_eq!(wire, expected, "row descriptor wire drift");
}

/// Trap: route entry field swap (`expert_id` <-> `gate_weight`). Kills silent
/// wrong routing + garbage weights against the ds41rt v3 12-B entry.
#[test]
fn n_route_entry_wire_bytes_pinned() {
    let wire = frame::route_entry_wire(&pinned_route(), naive_from_env());
    #[rustfmt::skip]
    let expected: [u8; 12] = [
        0x04, 0x03, 0x02, 0x01, // row_index
        0x14, 0x13, 0x12, 0x11, // expert_id
        0x00, 0x00, 0x80, 0x3F, // gate_weight = 1.0f32
    ];
    assert_eq!(wire, expected, "route entry wire drift");
}

/// Trap: dtype wire codes renumbered (enum-order drift recodes 5/7 and the
/// seam word-salads). Kills any deviation from the DS41RTE3 v3 codes.
#[test]
fn n_dtype_wire_codes_pinned() {
    let naive = naive_from_env();
    assert_eq!(Dtype::Bf16.code(naive), 1, "Bf16 code drift");
    assert_eq!(Dtype::F16.code(naive), 2, "F16 code drift");
    assert_eq!(Dtype::Fp8Debug.code(naive), 3, "Fp8Debug code drift");
    assert_eq!(Dtype::Nvfp4E2m1Fp8E4m3.code(naive), 4, "Nvfp4 code drift");
    assert_eq!(Dtype::Fp8E4m3RowScaled.code(naive), 5, "Fp8E4m3RowScaled code drift");
    assert_eq!(Dtype::F32.code(naive), 6, "F32 code drift");
    assert_eq!(Dtype::Fp8E4m3Ue8m0K32.code(naive), 7, "Fp8E4m3Ue8m0K32 code drift");
}

/// Trap: CRC-32 (IEEE) in place of CRC32C (Castagnoli). Kills the wrong
/// polynomial family via the RFC 3720 / iSCSI vector.
#[test]
fn n_crc32c_matches_rfc3720_vector() {
    assert_eq!(crc32c::crc32c_env(b"123456789"), 0xE306_9283, "CRC32C family drift");
}

/// Trap: f32 -> BF16 truncation (biases every Spark partial and the FP32 sum).
/// Kills anything but round-to-nearest-even (tie cases pinned).
#[test]
fn n_bf16_rounds_to_nearest_even() {
    // 0x3F818000: exact tie with kept LSB 1 -> rounds UP to 0x3F82.
    assert_eq!(bf16::f32_to_bf16_env(f32::from_bits(0x3F81_8000)), 0x3F82, "tie-up drift");
    // 0x3F808000: exact tie with kept LSB 0 -> rounds DOWN to 0x3F80.
    assert_eq!(bf16::f32_to_bf16_env(f32::from_bits(0x3F80_8000)), 0x3F80, "tie-down drift");
    assert_eq!(bf16::f32_to_bf16_env(1.0), 0x3F80, "1.0 drift");
}

// ---------------------------------------------------------------------------
// BOTH RUNS (explicit flags)
// ---------------------------------------------------------------------------

/// Runtime twins of the compile-time const guards in `src/layout.rs`.
#[test]
fn const_and_runtime_row_guards() {
    assert_eq!(layout::REQUEST_ROW_BYTES, 4360);
    assert_eq!(layout::RETURN_ROW_BYTES, 8192);
    assert_eq!(layout::HIDDEN_ROW_BYTES, 4224);
    assert_eq!(layout::request_row_bytes(WireNaive::NONE), 4360);
    assert_eq!(layout::return_row_bytes(layout::TOPK, WireNaive::NONE), 8192);
    assert_eq!(layout::hidden_row_bytes(WireNaive::NONE), 4224);
}

/// The `Fp8E4m3Ue8m0K32` row formula (elements + elements/32) and its
/// multiple-of-32 fail-loud guard (a truncated scale grid must never encode).
#[test]
fn k32_hidden_row_formula_and_multiples_of_32() {
    assert_eq!(layout::ue8m0_k32_row_bytes(4096).expect("4096 ok"), 4224);
    assert_eq!(layout::ue8m0_k32_row_bytes(32).expect("32 ok"), 33);
    assert!(layout::ue8m0_k32_row_bytes(0).is_err(), "zero width must fail loud");
    assert!(layout::ue8m0_k32_row_bytes(4097).is_err(), "non-multiple-of-32 must fail loud");
    assert_eq!(layout::scale_k(WireNaive::NONE), 32);
}

/// Multi-token request frames cost exactly 4,360 B per row (prefill rows are
/// N independent token rows).
#[test]
fn request_body_is_4360_per_row() {
    for rows in [1usize, 2, 8] {
        let f = request_fixture(rows, WireNaive::NONE);
        let bytes = frame::encode_request(&f, WireNaive::NONE).expect("encode");
        assert_eq!(bytes.len() - layout::HEADER_LEN, rows * 4360, "rows={rows}");
    }
}

/// Routes map to rows by descriptor `route_offset`/`route_count` + entry
/// `row_index`, never by table position. Kills positional route chunking.
#[test]
fn route_entries_map_by_descriptor_offsets() {
    let naive = WireNaive::NONE;
    let mut f = request_fixture(2, naive);
    // Rebuild with non-uniform, out-of-order route placement: row 1 owns
    // table slots 0..5, row 0 owns table slots 5..8.
    f.routes.clear();
    for e in 0..5 {
        f.routes.push(route_fixture(1, e)); // table slots 0..5 -> row 1
    }
    for e in 0..3 {
        f.routes.push(route_fixture(0, e)); // table slots 5..8 -> row 0
    }
    f.rows[0].route_offset = 5;
    f.rows[0].route_count = 3;
    f.rows[1].route_offset = 0;
    f.rows[1].route_count = 5;

    let bytes = frame::encode_request(&f, naive).expect("encode");
    let g = match frame::decode_frame(&bytes, naive).expect("decode") {
        Frame::Request(g) => g,
        _ => panic!("expected request"),
    };
    for (r, row) in g.rows.iter().enumerate() {
        let slice = &g.routes[row.route_offset as usize..(row.route_offset + row.route_count) as usize];
        assert_eq!(slice.len(), f.rows[r].route_count as usize);
        for entry in slice {
            assert_eq!(entry.row_index, r as u32, "entry mapped to wrong row");
        }
    }
    assert_eq!(g.rows[0].route_count, 3);
    assert_eq!(g.rows[1].route_count, 5);
    assert_eq!(g.routes, f.routes, "route table not preserved verbatim");
}

/// Request roundtrip: descriptors, routes (weights BIT-exact) and hidden
/// payload + scale bytes preserved verbatim ("consumers must preserve the
/// supplied payload and scales").
#[test]
fn request_roundtrip_multirow_bit_exact() {
    let naive = WireNaive::NONE;
    let f = request_fixture(2, naive);
    let bytes = frame::encode_request(&f, naive).expect("encode");
    let g = match frame::decode_frame(&bytes, naive).expect("decode") {
        Frame::Request(g) => g,
        _ => panic!("expected request"),
    };
    assert_eq!(g.rows, f.rows, "row descriptors drifted");
    assert_eq!(g.routes.len(), f.routes.len());
    for (a, b) in g.routes.iter().zip(&f.routes) {
        assert_eq!(a.row_index, b.row_index);
        assert_eq!(a.expert_id, b.expert_id);
        assert_eq!(a.gate_weight.to_bits(), b.gate_weight.to_bits(), "gate weight drifted");
    }
    assert_eq!(g.hidden_rows, f.hidden_rows, "payload/scales not verbatim");
    assert_eq!((g.request_id, g.layer_id, g.executor_id), (5, 3, 2));
}

/// Return roundtrip: 4,096 BF16 codes and identity fields preserved.
#[test]
fn return_roundtrip_bit_exact() {
    let naive = WireNaive::NONE;
    let f = spark_return_frame(9, 4, 1, 3);
    let bytes = frame::encode_return(&f, naive).expect("encode");
    assert_eq!(bytes.len(), layout::HEADER_LEN + 8192);
    let g = match frame::decode_frame(&bytes, naive).expect("decode") {
        Frame::Return(g) => g,
        _ => panic!("expected return"),
    };
    assert_eq!(g.rows, f.rows, "BF16 codes drifted");
    assert_eq!((g.request_id, g.layer_id, g.executor_id, g.token_position), (9, 4, 3, 1));
    assert_eq!(g.route_count, layout::TOPK);
}

/// Fail-loud guard for the A6 contract: a return frame not flagged
/// `SPARK_REDUCTION | V41_COMPACT_BF16` (an un-pre-summed / per-route return)
/// must be rejected, never summed.
#[test]
fn return_frame_requires_presummed_compact_flags() {
    let mut f = spark_return_frame(1, 2, 0, 0);
    f.flags = 0;
    let bytes = frame::encode_return(&f, WireNaive::NONE).expect("encode");
    match frame::decode_frame(&bytes, WireNaive::NONE) {
        Err(WireError::UnpresummedReturn { .. }) => {}
        other => panic!("expected UnpresummedReturn, got {other:?}"),
    }
}

/// Protocol mismatch (magic/version/kind/header_len) fails LOUD before the
/// checksum step — a foreign frame is never "retryable corruption".
#[test]
fn protocol_mismatch_fails_loud_before_crc() {
    let f = spark_return_frame(1, 2, 0, 0);
    let good = frame::encode_return(&f, WireNaive::NONE).expect("encode");

    let mut b = good.clone();
    b[0] ^= 0xFF;
    match frame::decode_frame(&b, WireNaive::NONE) {
        Err(e @ WireError::BadMagic(_)) => assert_eq!(e.disposition(), glm53f_wire::Disposition::FailLoud),
        other => panic!("expected BadMagic, got {other:?}"),
    }

    let mut b = good.clone();
    b[hdr::VERSION] = 2;
    assert!(matches!(frame::decode_frame(&b, WireNaive::NONE), Err(WireError::BadVersion(2))));

    let mut b = good.clone();
    b[hdr::KIND] = 3;
    b[hdr::KIND + 1] = 0;
    assert!(matches!(frame::decode_frame(&b, WireNaive::NONE), Err(WireError::BadKind(3))));

    let mut b = good.clone();
    b[hdr::HEADER_LEN] = 96;
    assert!(matches!(frame::decode_frame(&b, WireNaive::NONE), Err(WireError::BadHeaderLen(96))));
}

/// Table-driven CRC32C must equal the independent bit-by-bit reference —
/// kills table-generation errors the RFC vector alone could miss.
#[test]
fn crc32c_table_matches_bitwise_reference() {
    let mut long = Vec::with_capacity(1024);
    let mut x = 0x1234_5678u32;
    for _ in 0..1024 {
        x = x.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        long.push((x >> 24) as u8);
    }
    let samples: Vec<Vec<u8>> = vec![
        Vec::new(),
        b"a".to_vec(),
        b"123456789".to_vec(),
        vec![0u8; 300],
        long,
    ];
    for s in &samples {
        assert_eq!(crc32c::crc32c(s), crc32c::crc32c_bitwise(s), "CRC table mismatch");
    }
}

/// Meta-guard: every naive bit IS a killable wrong implementation (each one
/// violates the property its negative asserts). If a bit ever becomes a no-op,
/// its NEGATIVE stops failing the naive run — this catches that.
#[test]
fn every_naive_bit_is_a_killable_wrong_impl() {
    // Layout traps.
    assert_ne!(
        layout::request_row_bytes(WireNaive::of(WireNaive::K16_SCALES)),
        4360,
        "K16_SCALES must change the request row size"
    );
    assert_ne!(
        layout::request_row_bytes(WireNaive::of(WireNaive::ROW_SCALED_DTYPE)),
        4360,
        "ROW_SCALED_DTYPE must change the request row size"
    );
    assert_ne!(
        layout::return_row_bytes(layout::TOPK, WireNaive::of(WireNaive::FP32_RETURN)),
        8192,
        "FP32_RETURN must change the return row size"
    );
    assert_ne!(
        layout::return_row_bytes(layout::TOPK, WireNaive::of(WireNaive::PER_ROUTE_RETURN)),
        8192,
        "PER_ROUTE_RETURN must change the return row size"
    );
    assert_ne!(
        layout::return_row_bytes(1, WireNaive::of(WireNaive::PER_ROUTE_RETURN)),
        layout::return_row_bytes(layout::TOPK, WireNaive::of(WireNaive::PER_ROUTE_RETURN)),
        "PER_ROUTE_RETURN must scale with route count"
    );
    let desc_canon = frame::row_descriptor_wire(&pinned_row(), WireNaive::NONE);
    assert_ne!(
        frame::row_descriptor_wire(&pinned_row(), WireNaive::of(WireNaive::DESC_FIELD_DRIFT)),
        desc_canon,
        "DESC_FIELD_DRIFT must move descriptor bytes"
    );
    let route_canon = frame::route_entry_wire(&pinned_route(), WireNaive::NONE);
    assert_ne!(
        frame::route_entry_wire(&pinned_route(), WireNaive::of(WireNaive::ROUTE_FIELD_SWAP)),
        route_canon,
        "ROUTE_FIELD_SWAP must move route bytes"
    );
    assert_ne!(
        Dtype::Fp8E4m3Ue8m0K32.code(WireNaive::of(WireNaive::DTYPE_RECODE)),
        7,
        "DTYPE_RECODE must renumber wire codes"
    );
    assert_ne!(
        crc32c::crc32c_with(crc32c::family_for(WireNaive::of(WireNaive::CRC32_IEEE)), b"123456789"),
        0xE306_9283,
        "CRC32_IEEE must break the RFC 3720 vector"
    );
    assert_ne!(
        bf16::f32_to_bf16(f32::from_bits(0x3F81_8000), WireNaive::of(WireNaive::BF16_TRUNCATE)),
        0x3F82,
        "BF16_TRUNCATE must lose the tie-up rounding"
    );

    // Checksum coverage traps: corruption must PASS UNDETECTED for these.
    let f = spark_return_frame(1, 2, 0, 0);
    let naive = WireNaive::of(WireNaive::CRC_HEADER_EXEMPT);
    let mut b = frame::encode_return(&f, naive).expect("encode");
    b[hdr::LAYER_ID] ^= 0x01;
    assert!(
        frame::decode_frame(&b, naive).is_ok(),
        "CRC_HEADER_EXEMPT must miss header bitflips"
    );

    let naive = WireNaive::of(WireNaive::CRC_UNCHECKED);
    let mut b = frame::encode_return(&f, naive).expect("encode");
    b[layout::HEADER_LEN + 5] ^= 0x01;
    assert!(
        frame::decode_frame(&b, naive).is_ok(),
        "CRC_UNCHECKED must miss payload bitflips"
    );

    // Sequence traps: out-of-order and duplicates must PASS UNDETECTED.
    let naive = WireNaive::of(WireNaive::SEQ_IGNORED);
    let mut sender = StreamSender::new(naive);
    let w0 = sender.encode_return(&f).expect("encode 0");
    let w1 = sender.encode_return(&spark_return_frame(1, 2, 1, 0)).expect("encode 1");
    let mut rx = StreamReceiver::new(naive);
    assert!(rx.accept(&w1).is_ok(), "SEQ_IGNORED must accept seq 1 before 0");
    assert!(rx.accept(&w0).is_ok(), "SEQ_IGNORED must accept the late seq 0");

    let naive = WireNaive::of(WireNaive::DUP_DOUBLE_COUNT);
    let mut sum = glm53f_wire::CoordinatorSum::new(1, layout::HIDDEN, naive);
    for s in 0..layout::SPARKS {
        sum.accumulate(&spark_return_frame(1, 2, 0, s as u64)).expect("accumulate");
    }
    sum.accumulate(&spark_return_frame(1, 2, 0, 3)).expect("duplicate slips through");
    let got = sum.result().expect("complete").to_vec();
    assert_ne!(got, golden_row(0, layout::HIDDEN), "DUP_DOUBLE_COUNT must corrupt the sum");

    let naive = WireNaive::of(WireNaive::ACCUM_BY_ARRIVAL);
    let mut sum = glm53f_wire::CoordinatorSum::new(2, layout::HIDDEN, naive);
    for s in 0..layout::SPARKS {
        sum.accumulate(&spark_return_frame(1, 2, 1, s as u64)).expect("accumulate");
    }
    for s in 0..layout::SPARKS {
        sum.accumulate(&spark_return_frame(1, 2, 0, s as u64)).expect("accumulate");
    }
    let got = sum.result().expect("complete").to_vec();
    assert_ne!(
        got[..layout::HIDDEN],
        golden_row(0, layout::HIDDEN)[..],
        "ACCUM_BY_ARRIVAL must land token 1's partials in token 0's row"
    );
}
