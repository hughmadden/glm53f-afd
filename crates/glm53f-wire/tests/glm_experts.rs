//! GLM-5.3-Flash routes top-8 of 288 experts. Expert ids are `u32` route-entry
//! fields, so ids 256..=287 (beyond MiMo's 256) must cross the wire unchanged
//! through every request path: the frame encoder and decoder, the zero-copy
//! `RequestView`, and the in-place `encode_request_meta_into`.
//!
//! Classification: BOTH RUNS (explicit `WireNaive::NONE`).

use glm53f_wire::frame::{
    decode_frame, encode_request_meta_into, encode_request_seq, route_entry_wire, Frame, HiddenRow, RequestFrame,
    RequestView, RouteEntry, RowDescriptor,
};
use glm53f_wire::layout::{self, hdr, SourceKind};
use glm53f_wire::WireNaive;

/// GLM-5.3-Flash routed experts.
const EXPERTS: u32 = 288;

/// Row `t`'s top-8 expert ids, all distinct and all at or above 256; row 0 takes
/// the highest id, 287.
fn experts(t: usize) -> Vec<u32> {
    (0..layout::TOPK as u32).map(|k| EXPERTS - 1 - ((t as u32 * 3 + k * 4) % 32)).collect()
}

fn frame(rows: usize) -> RequestFrame {
    let mut descs = Vec::new();
    let mut routes = Vec::new();
    let mut hidden = Vec::new();
    for t in 0..rows {
        descs.push(RowDescriptor {
            row_id: t as u64,
            source_kind: SourceKind::Decode,
            source_request_id: 3,
            token_position: t as u64,
            route_offset: (t * layout::TOPK) as u32,
            route_count: layout::TOPK as u32,
        });
        for (k, e) in experts(t).into_iter().enumerate() {
            routes.push(RouteEntry { row_index: t as u32, expert_id: e, gate_weight: 0.125 * (k + 1) as f32 });
        }
        hidden.push(HiddenRow {
            payload: (0..layout::HIDDEN).map(|i| ((t * 17 + i * 5) % 251) as u8).collect(),
            scales: (0..layout::HIDDEN / layout::K32).map(|i| (118 + (t + i) % 11) as u8).collect(),
        });
    }
    RequestFrame {
        request_id: 3,
        placement_version: 1,
        layer_id: 44,
        executor_id: 1,
        source_kind: SourceKind::Decode,
        token_position: 0,
        flags: 0,
        seq: 0,
        rows: descs,
        routes,
        hidden_rows: hidden,
    }
}

#[test]
fn expert_ids_up_to_287_round_trip() {
    let f = frame(5);
    assert!(f.routes.iter().all(|r| (256..EXPERTS).contains(&r.expert_id)));
    assert!(f.routes.iter().any(|r| r.expert_id == EXPERTS - 1));
    let bytes = encode_request_seq(&f, 11, WireNaive::NONE).expect("encode");
    let d = match decode_frame(&bytes, WireNaive::NONE).expect("decode") {
        Frame::Request(r) => r,
        Frame::Return(_) => panic!("not a request"),
    };
    assert_eq!(d.routes, f.routes, "decoded routes differ");
    let v = RequestView::parse(&bytes, WireNaive::NONE).expect("view");
    for (j, r) in f.routes.iter().enumerate() {
        assert_eq!(&v.route(j).expect("route"), r, "view route {j} differs");
    }
}

#[test]
fn expert_287_route_entry_bytes() {
    let e = RouteEntry { row_index: 0, expert_id: EXPERTS - 1, gate_weight: 1.0 };
    #[rustfmt::skip]
    let expected: [u8; 12] = [
        0x00, 0x00, 0x00, 0x00, // row_index
        0x1F, 0x01, 0x00, 0x00, // expert_id = 287
        0x00, 0x00, 0x80, 0x3F, // gate_weight = 1.0f32
    ];
    assert_eq!(route_entry_wire(&e, WireNaive::NONE), expected);
}

#[test]
fn meta_into_carries_expert_ids_above_255() {
    let rows = 3;
    let f = frame(rows);
    let routes: Vec<(u32, f32)> = f.routes.iter().map(|r| (r.expert_id, r.gate_weight)).collect();
    let mut want = encode_request_seq(&f, 5, WireNaive::NONE).expect("encode");
    want[hdr::CRC32C..hdr::CRC32C + 4].fill(0);
    let mut body = vec![0u8; rows * layout::REQUEST_ROW_BYTES];
    let (header, hidden_off, body_len) =
        encode_request_meta_into(&mut body, 3, 44, 1, 5, &routes, layout::TOPK, WireNaive::NONE).expect("meta into");
    for (t, h) in f.hidden_rows.iter().enumerate() {
        let o = hidden_off + t * layout::HIDDEN_ROW_BYTES;
        body[o..o + layout::HIDDEN].copy_from_slice(&h.payload);
        body[o + layout::HIDDEN..o + layout::HIDDEN_ROW_BYTES].copy_from_slice(&h.scales);
    }
    let mut got = header.to_vec();
    got.extend_from_slice(&body[..body_len]);
    assert!(got == want, "in-place frame with expert ids above 255 differs from encode_request_seq");
}
