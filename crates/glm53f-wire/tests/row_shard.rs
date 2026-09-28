//! Version 4, the prefill reduce-scatter (`row_shard`): the three frames' layouts pinned byte
//! for byte, round trips, the checks, how version-3 and version-4 peers refuse each other, and
//! the coordinator's scatter mode. Both runs (explicit `WireNaive::NONE`).

mod common;

use common::*;
use glm53f_wire::error::WireError;
use glm53f_wire::frame::{self, seal_in_place, Frame, RequestView, ReturnFrame, ReturnRow};
use glm53f_wire::layout::{self, hdr, Status};
use glm53f_wire::naive::WireNaive;
use glm53f_wire::row_shard::{encode_exchange, row_partition, ExchangeDtype, ExchangeHeader, ExchangeView};
use glm53f_wire::{CoordinatorSum, FLAG_EXCHANGE_FP8, FLAG_REDUCE_SCATTER, FLAG_RETURN_REQUIRED, FLAG_ROW_SLICE};

const NONE: WireNaive = WireNaive::NONE;

fn u16_at(b: &[u8], o: usize) -> u16 {
    u16::from_le_bytes([b[o], b[o + 1]])
}

fn u32_at(b: &[u8], o: usize) -> u32 {
    u32::from_le_bytes(b[o..o + 4].try_into().unwrap())
}

fn u64_at(b: &[u8], o: usize) -> u64 {
    u64::from_le_bytes(b[o..o + 8].try_into().unwrap())
}

/// Patch a header field and reseal, as a peer of another version (or a bug) would send it.
fn patched(bytes: &[u8], at: usize, value: &[u8]) -> Vec<u8> {
    let mut b = bytes.to_vec();
    b[at..at + value.len()].copy_from_slice(value);
    b[hdr::CRC32C..hdr::CRC32C + 4].fill(0);
    seal_in_place(&mut b, NONE);
    b
}

/// A row-slice return: rank `rank`'s partition of `rows` rows, row `i` holding `value(i, rank)`.
fn slice_return(request_id: u64, rows: usize, rank: usize) -> ReturnFrame {
    let (first, count) = row_partition(rows, layout::SPARKS, rank);
    ReturnFrame {
        request_id,
        placement_version: 1,
        layer_id: 3,
        executor_id: rank as u64,
        token_position: first as u64,
        status: Status::Ok,
        flags: FLAG_RETURN_REQUIRED | FLAG_ROW_SLICE,
        route_count: layout::TOPK,
        seq: 0,
        rows: (first..first + count)
            .map(|t| ReturnRow { codes: (0..layout::HIDDEN).map(|i| glm53f_wire::bf16::f32_to_bf16_rne(value(t, rank, i))).collect() })
            .collect(),
    }
}

#[test]
fn a_reduce_scattered_request_is_version_4_and_otherwise_unchanged() {
    for dtype in [ExchangeDtype::Bf16, ExchangeDtype::Fp8RowScaled] {
        let mut f = request_fixture(33, NONE);
        let v3 = frame::encode_request(&f, NONE).unwrap();
        f.flags = dtype.request_flags();
        let v4 = frame::encode_request(&f, NONE).unwrap();
        assert_eq!(u16_at(&v3, hdr::VERSION), 3);
        assert_eq!(u16_at(&v4, hdr::VERSION), 4, "a reduce-scattered request is version 4");
        assert_eq!(u32_at(&v4, hdr::FLAGS), dtype.request_flags());
        assert_eq!(u32_at(&v4, hdr::RESERVED), 0);
        // Only the version, the flags and the CRC differ: the body is the same bytes.
        for (i, (a, b)) in v3.iter().zip(&v4).enumerate() {
            let changed = (hdr::VERSION..hdr::VERSION + 2).contains(&i)
                || (hdr::FLAGS..hdr::FLAGS + 4).contains(&i)
                || (hdr::CRC32C..hdr::CRC32C + 4).contains(&i);
            assert!(a == b || changed, "byte {i} changed");
        }
        let Frame::Request(g) = frame::decode_frame(&v4, NONE).unwrap() else { panic!("a request") };
        assert_eq!(g, f);
        let view = RequestView::parse(&v4, NONE).unwrap();
        assert_eq!(ExchangeDtype::from_request_flags(view.flags), Some(dtype));
    }
    // FP8 exchange without the reduce-scatter, a return flag on a request, one row per rank.
    let mut f = request_fixture(33, NONE);
    f.flags = FLAG_EXCHANGE_FP8;
    assert!(frame::encode_request(&f, NONE).is_err());
    f.flags = FLAG_REDUCE_SCATTER | FLAG_ROW_SLICE;
    assert!(frame::encode_request(&f, NONE).is_err());
    let mut f = request_fixture(3, NONE);
    f.flags = FLAG_REDUCE_SCATTER;
    assert!(matches!(frame::encode_request(&f, NONE), Err(WireError::DimMismatch { field: "reduce_scatter_rows", .. })));
    let mut f4 = request_fixture(4, NONE);
    f4.flags = FLAG_REDUCE_SCATTER;
    assert!(frame::encode_request(&f4, NONE).is_ok(), "four rows: one per rank");
}

#[test]
fn a_row_slice_return_is_version_4_and_round_trips() {
    let r = slice_return(9, 33, 1);
    assert_eq!((r.token_position, r.rows.len()), (9, 8));
    let bytes = frame::encode_return(&r, NONE).unwrap();
    assert_eq!(bytes.len(), layout::HEADER_LEN + 8 * layout::RETURN_ROW_BYTES, "8,192 bytes a row, only the slice's rows");
    assert_eq!(u16_at(&bytes, hdr::VERSION), 4);
    assert_eq!(u32_at(&bytes, hdr::FLAGS), FLAG_RETURN_REQUIRED | FLAG_ROW_SLICE);
    assert_eq!((u32_at(&bytes, hdr::ROW_COUNT), u64_at(&bytes, hdr::TOKEN_POSITION)), (8, 9));
    assert_eq!(u32_at(&bytes, hdr::RESERVED), 0);
    let Frame::Return(g) = frame::decode_frame(&bytes, NONE).unwrap() else { panic!("a return") };
    assert_eq!(g, r);
    // The in-place header is the same bytes.
    let h = frame::return_header_seq(&r, 8, 0, NONE).unwrap();
    assert_eq!(h[..hdr::CRC32C], bytes[..hdr::CRC32C]);
    // A four-plane return stays version 3; a request flag on a return is refused.
    let plane = spark_return_frame(9, 3, 0, 1);
    assert_eq!(u16_at(&frame::encode_return(&plane, NONE).unwrap(), hdr::VERSION), 3);
    let mut bad = r.clone();
    bad.flags |= FLAG_REDUCE_SCATTER;
    assert!(frame::encode_return(&bad, NONE).is_err());
}

#[test]
fn the_exchange_frame_layout_is_pinned() {
    for (dtype, code, stride) in [(ExchangeDtype::Bf16, 1u16, 8192usize), (ExchangeDtype::Fp8RowScaled, 5, 4100)] {
        let h = ExchangeHeader::new(0x0102_0304_0506_0708, 44, 33, 2, 1, dtype);
        assert_eq!((h.first_row, h.row_count), (9, 8), "rank 1's partition of 33 rows");
        let mut b = vec![0u8; h.frame_len()];
        h.write(5, &mut b).unwrap();
        assert_eq!(&b[0..8], b"DS41RTE3");
        assert_eq!((u16_at(&b, hdr::VERSION), u16_at(&b, hdr::KIND), u32_at(&b, hdr::HEADER_LEN)), (4, 3, 128));
        assert_eq!(u64_at(&b, hdr::REQUEST_ID), 0x0102_0304_0506_0708);
        assert_eq!(u64_at(&b, hdr::PLACEMENT_VERSION), 0);
        assert_eq!((u32_at(&b, hdr::LAYER_ID), u32_at(&b, hdr::ROW_COUNT), u32_at(&b, hdr::DIM)), (44, 8, 4096));
        assert_eq!((u16_at(&b, hdr::PAYLOAD_DTYPE), u16_at(&b, hdr::SOURCE_KIND)), (code, 1));
        assert_eq!((u32_at(&b, hdr::ROUTE_COUNT), u32_at(&b, hdr::ROW_DESCRIPTOR_BYTES), u32_at(&b, hdr::ROUTE_BYTES)), (8, 0, 0));
        assert_eq!(u64_at(&b, hdr::PAYLOAD_BYTES), 8 * stride as u64);
        assert_eq!(u64_at(&b, hdr::LOGICAL_PAYLOAD_BYTES), 8 * stride as u64);
        assert_eq!(u64_at(&b, hdr::WIRE_BYTES), 128 + 8 * stride as u64);
        assert_eq!((u32_at(&b, hdr::FLAGS), u32_at(&b, hdr::ROW_STRIDE_BYTES), u32_at(&b, hdr::STATUS)), (1 << 17, stride as u32, 0));
        assert_eq!((u64_at(&b, hdr::EXECUTOR_ID), u64_at(&b, hdr::TOKEN_POSITION), u64_at(&b, hdr::SEQ)), (2, 9, 5));
        assert_eq!(u32_at(&b, hdr::CRC32C), 0, "sealed by the caller");
        assert_eq!(u32_at(&b, hdr::RESERVED), 1 | 4 << 8 | 33 << 16, "receiving rank, world, the request's rows");
    }
    // The header refuses what it cannot carry.
    let good = ExchangeHeader::new(1, 3, 33, 2, 1, ExchangeDtype::Bf16);
    let mut b = vec![0u8; good.frame_len()];
    for bad in [
        ExchangeHeader { src: 1, ..good },
        ExchangeHeader { dst: 4, ..good },
        ExchangeHeader { first_row: 8, ..good },
        ExchangeHeader { row_count: 9, ..good },
        ExchangeHeader { rows: 3, ..good },
    ] {
        assert!(bad.write(0, &mut b).is_err(), "{bad:?}");
    }
}

#[test]
fn exchange_frames_round_trip_and_are_checked() {
    for dtype in [ExchangeDtype::Bf16, ExchangeDtype::Fp8RowScaled] {
        let h = ExchangeHeader::new(77, 12, 37, 3, 0, dtype);
        let payload: Vec<u8> = (0..h.payload_len()).map(|i| (i * 7 + 1) as u8).collect();
        let bytes = encode_exchange(&h, &payload, 11, NONE).unwrap();
        let v = ExchangeView::parse(&bytes, NONE).unwrap();
        assert_eq!((v.header, v.seq, v.payload), (h, 11, &payload[..]));
        assert_eq!(v.row(1), &payload[dtype.row_bytes()..2 * dtype.row_bytes()]);
        // Truncated, and another frame kind through the exchange parser, and the reverse.
        assert!(ExchangeView::parse(&bytes[..bytes.len() - 1], NONE).is_err());
        assert!(matches!(frame::decode_frame(&bytes, NONE), Err(WireError::BadKind(3))));
        let req = frame::encode_request(&request_fixture(4, NONE), NONE).unwrap();
        assert!(matches!(ExchangeView::parse(&req, NONE), Err(WireError::BadKind(1))));
        if !frame::crc_disabled() {
            let mut flip = bytes.clone();
            flip[layout::HEADER_LEN + 100] ^= 0x10;
            assert!(matches!(ExchangeView::parse(&flip, NONE), Err(WireError::Corrupt { .. })));
        }
        // Fields that must hold: the partition, the ranks, the world, the flags, the stride.
        let word = |dst: u32, world: u32, rows: u32| (dst | world << 8 | rows << 16).to_le_bytes();
        for bad in [
            patched(&bytes, hdr::TOKEN_POSITION, &1u64.to_le_bytes()),
            patched(&bytes, hdr::EXECUTOR_ID, &0u64.to_le_bytes()),
            patched(&bytes, hdr::EXECUTOR_ID, &4u64.to_le_bytes()),
            patched(&bytes, hdr::RESERVED, &word(0, 8, 37)),
            patched(&bytes, hdr::RESERVED, &word(1, 4, 37)),
            // 41 rows: rank 0's partition would be 11 rows (38 would still give 10: the
            // receiver catches that one against its own request).
            patched(&bytes, hdr::RESERVED, &word(0, 4, 41)),
            patched(&bytes, hdr::FLAGS, &(FLAG_ROW_SLICE | FLAG_RETURN_REQUIRED).to_le_bytes()),
            patched(&bytes, hdr::STATUS, &1u32.to_le_bytes()),
            patched(&bytes, hdr::PLACEMENT_VERSION, &1u64.to_le_bytes()),
        ] {
            assert!(ExchangeView::parse(&bad, NONE).is_err());
        }
        // The kind exists only in version 4.
        assert!(matches!(ExchangeView::parse(&patched(&bytes, hdr::VERSION, &3u16.to_le_bytes()), NONE), Err(WireError::BadKind(3))));
    }
}

#[test]
fn version_3_and_version_4_peers_refuse_each_others_extension() {
    // A version-3 decoder accepts exactly `version == 3`: every version-4 frame carries 4.
    let mut req = request_fixture(33, NONE);
    req.flags = FLAG_REDUCE_SCATTER;
    let v4_request = frame::encode_request(&req, NONE).unwrap();
    let v4_return = frame::encode_return(&slice_return(1, 33, 2), NONE).unwrap();
    let mut ex = vec![0u8; ExchangeHeader::new(1, 3, 33, 0, 1, ExchangeDtype::Bf16).frame_len()];
    ExchangeHeader::new(1, 3, 33, 0, 1, ExchangeDtype::Bf16).write(0, &mut ex).unwrap();
    for f in [&v4_request, &v4_return, &ex] {
        assert_eq!(u16_at(f, hdr::VERSION), 4, "a version-3 decoder refuses it by version");
    }
    // This decoder: a version-4 flag in a version-3 frame, a version-4 frame without its flag.
    let as_v3 = patched(&v4_request, hdr::VERSION, &3u16.to_le_bytes());
    assert!(matches!(frame::decode_frame(&as_v3, NONE), Err(WireError::BadCode { .. })));
    assert!(RequestView::parse(&as_v3, NONE).is_err());
    let plain = frame::encode_request(&request_fixture(33, NONE), NONE).unwrap();
    let as_v4 = patched(&plain, hdr::VERSION, &4u16.to_le_bytes());
    assert!(matches!(frame::decode_frame(&as_v4, NONE), Err(WireError::BadCode { .. })));
    assert!(RequestView::parse(&as_v4, NONE).is_err());
    let slice_as_v3 = patched(&v4_return, hdr::VERSION, &3u16.to_le_bytes());
    assert!(matches!(frame::decode_frame(&slice_as_v3, NONE), Err(WireError::BadCode { .. })));
    let plane = frame::encode_return(&spark_return_frame(1, 3, 0, 0), NONE).unwrap();
    assert!(frame::decode_frame(&patched(&plane, hdr::VERSION, &4u16.to_le_bytes()), NONE).is_err());
    // Fewer rows than ranks in a version-4 request; a nonzero word at byte 124 in both versions.
    let mut small = request_fixture(4, NONE);
    small.flags = FLAG_REDUCE_SCATTER;
    let small = frame::encode_request(&small, NONE).unwrap();
    assert!(frame::decode_frame(&small, NONE).is_ok());
    assert!(matches!(
        frame::decode_frame(&patched(&small, hdr::ROW_COUNT, &3u32.to_le_bytes()), NONE),
        Err(WireError::DimMismatch { field: "reduce_scatter_rows", .. })
    ));
    for f in [&plain, &v4_request, &plane, &v4_return] {
        let bad = patched(f, hdr::RESERVED, &1u32.to_le_bytes());
        assert!(matches!(frame::decode_frame(&bad, NONE), Err(WireError::BadCode { field: "reserved", .. })));
    }
    // Unknown versions stay unknown.
    assert!(matches!(frame::decode_frame(&patched(&plain, hdr::VERSION, &5u16.to_le_bytes()), NONE), Err(WireError::BadVersion(5))));
}

#[test]
fn in_place_request_headers_take_the_reduce_scatter_flags() {
    let (rows, topk) = (18usize, layout::TOPK);
    let mut body = vec![0u8; rows * layout::REQUEST_ROW_BYTES];
    let (mut header, routes_off, hidden_off, blen) =
        frame::encode_request_desc_into(&mut body, 9, 3, 1, 0, rows, topk, NONE).unwrap();
    for t in 0..rows * topk {
        let e = frame::route_entry_wire(&glm53f_wire::RouteEntry { row_index: (t / topk) as u32, expert_id: t as u32 % 288, gate_weight: 0.25 }, NONE);
        body[routes_off + t * 12..routes_off + (t + 1) * 12].copy_from_slice(&e);
    }
    assert_eq!(hidden_off + rows * layout::HIDDEN_ROW_BYTES, blen);
    assert_eq!(u16_at(&header, hdr::VERSION), 3);
    frame::set_request_flags(&mut header, ExchangeDtype::Fp8RowScaled.request_flags()).unwrap();
    assert_eq!((u16_at(&header, hdr::VERSION), u32_at(&header, hdr::FLAGS)), (4, FLAG_REDUCE_SCATTER | FLAG_EXCHANGE_FP8));
    let mut f = header.to_vec();
    f.extend_from_slice(&body[..blen]);
    seal_in_place(&mut f, NONE);
    let view = RequestView::parse(&f, NONE).unwrap();
    assert_eq!(ExchangeDtype::from_request_flags(view.flags), Some(ExchangeDtype::Fp8RowScaled));
    // Back to four planes, and the refusals.
    frame::set_request_flags(&mut header, 0).unwrap();
    assert_eq!((u16_at(&header, hdr::VERSION), u32_at(&header, hdr::FLAGS)), (3, 0));
    assert!(frame::set_request_flags(&mut header, FLAG_ROW_SLICE).is_err());
    let (mut small, ..) = frame::encode_request_desc_into(&mut body, 9, 3, 1, 0, 3, topk, NONE).unwrap();
    assert!(frame::set_request_flags(&mut small, FLAG_REDUCE_SCATTER).is_err(), "3 rows for 4 ranks");
}

#[test]
fn the_coordinator_assembles_row_slices_once_each() {
    let rows = 18;
    let mut sum = CoordinatorSum::row_sharded(rows, layout::HIDDEN, NONE);
    for rank in [2usize, 0, 3] {
        sum.accumulate(&slice_return(4, rows, rank)).unwrap();
        assert!(!sum.is_complete());
    }
    assert!(matches!(sum.result(), Err(WireError::Incomplete { filled: 13, need: 18 })));
    // Rank 1 with another rank's partition, or twice.
    let mut wrong = slice_return(4, rows, 1);
    wrong.token_position = 0;
    assert!(sum.accumulate(&wrong).is_err());
    sum.accumulate(&slice_return(4, rows, 1)).unwrap();
    assert!(sum.is_complete());
    assert!(matches!(sum.accumulate(&slice_return(4, rows, 1)), Err(WireError::SlotFilled { .. })));
    let got = sum.result().unwrap();
    for t in 0..rows {
        let owner = (0..layout::SPARKS).find(|&r| {
            let (s, c) = row_partition(rows, layout::SPARKS, r);
            (s..s + c).contains(&t)
        });
        for i in [0usize, 1, 4095] {
            assert_eq!(got[t * layout::HIDDEN + i], value(t, owner.unwrap(), i), "row {t} element {i}: its owner's row, nothing added");
        }
    }
    // A plane is never written as a slice, nor a slice added as a plane.
    let mut scatter = CoordinatorSum::row_sharded(1, layout::HIDDEN, NONE);
    assert!(scatter.accumulate(&spark_return_frame(4, 3, 0, 0)).is_err());
    let mut planes = CoordinatorSum::new(rows, layout::HIDDEN, NONE);
    assert!(planes.accumulate(&slice_return(4, rows, 0)).is_err());
}
