//! Perf reset (Spark zero-copy receive): `RequestView::parse` accepts exactly the
//! request frames `decode_frame` accepts, with the same error for any frame whose
//! header says request (a frame of another kind is `BadKind`, where the decoder
//! dispatches it to the return decoder), and exposes the same header fields,
//! descriptors, routes and hidden bytes.

use glm53f_wire::frame::{
    decode_frame, encode_request_seq, seal_in_place, Frame, HiddenRow, RequestFrame, RequestView, RouteEntry,
    RowDescriptor,
};
use glm53f_wire::layout::{hdr, HEADER_LEN, ROUTE_ENTRY_LEN, ROW_DESCRIPTOR_LEN};
use glm53f_wire::{SourceKind, WireNaive, HIDDEN};

fn frame(rows: usize) -> RequestFrame {
    let mut descs = Vec::new();
    let mut routes = Vec::new();
    let mut hidden = Vec::new();
    for t in 0..rows {
        descs.push(RowDescriptor {
            row_id: t as u64,
            source_kind: SourceKind::Decode,
            source_request_id: 77,
            token_position: t as u64,
            route_offset: (t * 8) as u32,
            route_count: 8,
        });
        for k in 0..8u32 {
            routes.push(RouteEntry {
                row_index: t as u32,
                expert_id: (t as u32 * 37 + k * 29) % 256,
                gate_weight: 0.125 + k as f32 * 0.01,
            });
        }
        hidden.push(HiddenRow {
            payload: (0..HIDDEN).map(|i| ((t * 131 + i * 7) % 251) as u8).collect(),
            scales: (0..HIDDEN / 32).map(|i| (120 + (t + i) % 9) as u8).collect(),
        });
    }
    RequestFrame {
        request_id: 9,
        placement_version: 1,
        layer_id: 12,
        executor_id: 3,
        source_kind: SourceKind::Decode,
        token_position: 0,
        flags: 0,
        seq: 0,
        rows: descs,
        routes,
        hidden_rows: hidden,
    }
}

fn same_outcome(bytes: &[u8]) {
    let a = decode_frame(bytes, WireNaive::NONE);
    let b = RequestView::parse(bytes, WireNaive::NONE);
    match (&a, &b) {
        (Ok(_), Ok(_)) => {}
        (Err(x), Err(y)) => assert_eq!(format!("{x:?}"), format!("{y:?}"), "different errors ({x:?} vs {y:?})"),
        _ => panic!("decode_frame {:?} vs RequestView {:?}", a.as_ref().err(), b.as_ref().err()),
    }
}

#[test]
fn view_matches_decoder_on_valid_frames() {
    for rows in [1usize, 3, 17] {
        let f = frame(rows);
        let bytes = encode_request_seq(&f, 5, WireNaive::NONE).unwrap();
        let v = RequestView::parse(&bytes, WireNaive::NONE).unwrap();
        let d = match decode_frame(&bytes, WireNaive::NONE).unwrap() {
            Frame::Request(r) => r,
            Frame::Return(_) => panic!("not a request"),
        };
        assert_eq!(
            (v.request_id, v.placement_version, v.layer_id, v.executor_id, v.token_position, v.flags, v.seq),
            (d.request_id, d.placement_version, d.layer_id, d.executor_id, d.token_position, d.flags, d.seq)
        );
        assert_eq!(v.source_kind, d.source_kind);
        assert_eq!((v.rows, v.routes, v.row_stride), (d.rows.len(), d.routes.len(), HIDDEN + HIDDEN / 32));
        for (i, r) in d.rows.iter().enumerate() {
            assert_eq!(&v.row(i).unwrap(), r);
        }
        for (j, r) in d.routes.iter().enumerate() {
            assert_eq!(&v.route(j).unwrap(), r);
        }
        let hidden = v.hidden();
        for (i, h) in d.hidden_rows.iter().enumerate() {
            let row = &hidden[i * v.row_stride..(i + 1) * v.row_stride];
            assert_eq!(&row[..HIDDEN], &h.payload[..]);
            assert_eq!(&row[HIDDEN..], &h.scales[..]);
        }
    }
}

#[test]
fn view_rejects_what_the_decoder_rejects() {
    let rows = 4usize;
    let good = encode_request_seq(&frame(rows), 5, WireNaive::NONE).unwrap();
    let route_at = HEADER_LEN + rows * ROW_DESCRIPTOR_LEN;
    let mut cases: Vec<Vec<u8>> = Vec::new();
    // Structural corruptions, re-sealed so the CRC cannot be what catches them.
    let mut resealed = |edit: &dyn Fn(&mut Vec<u8>)| {
        let mut b = good.clone();
        edit(&mut b);
        seal_in_place(&mut b, WireNaive::NONE);
        cases.push(b);
    };
    resealed(&|b| b[route_at + 5 * ROUTE_ENTRY_LEN..route_at + 5 * ROUTE_ENTRY_LEN + 4]
        .copy_from_slice(&(rows as u32).to_le_bytes())); // route row_index out of range
    resealed(&|b| b[HEADER_LEN + 2 * ROW_DESCRIPTOR_LEN + 32..HEADER_LEN + 2 * ROW_DESCRIPTOR_LEN + 36]
        .copy_from_slice(&1000u32.to_le_bytes())); // row route_count past the routes
    resealed(&|b| b[HEADER_LEN + 8..HEADER_LEN + 10].copy_from_slice(&0xBEEFu16.to_le_bytes())); // row source kind
    resealed(&|b| b[hdr::DIM..hdr::DIM + 4].copy_from_slice(&2048u32.to_le_bytes()));
    resealed(&|b| b[hdr::ROW_COUNT..hdr::ROW_COUNT + 4].copy_from_slice(&0u32.to_le_bytes()));
    resealed(&|b| b[hdr::ROW_STRIDE_BYTES..hdr::ROW_STRIDE_BYTES + 4].copy_from_slice(&100u32.to_le_bytes()));
    // Raw corruptions: the CRC (on by default in tests) or the length check fires.
    let mut flipped = good.clone();
    flipped[HEADER_LEN + 3 * ROW_DESCRIPTOR_LEN + 1] ^= 0x10;
    cases.push(flipped);
    cases.push(good[..good.len() - 1].to_vec());
    let mut longer = good.clone();
    longer.push(0);
    cases.push(longer);
    for (i, c) in cases.iter().enumerate() {
        assert!(RequestView::parse(c, WireNaive::NONE).is_err(), "case {i} accepted");
        same_outcome(c);
    }
    same_outcome(&good);
    // A valid frame relabelled as a return: the view refuses the kind itself.
    let mut relabelled = good.clone();
    relabelled[hdr::KIND..hdr::KIND + 2].copy_from_slice(&2u16.to_le_bytes());
    seal_in_place(&mut relabelled, WireNaive::NONE);
    assert!(matches!(
        RequestView::parse(&relabelled, WireNaive::NONE),
        Err(glm53f_wire::WireError::BadKind(2))
    ));
    assert!(!matches!(decode_frame(&relabelled, WireNaive::NONE), Ok(Frame::Request(_))));
}
