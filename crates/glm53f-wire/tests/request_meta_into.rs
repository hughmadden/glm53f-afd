//! `encode_request_meta_into` (perf reset P6): the in-place request encoding
//! The coordinator's RDMA fast path uses must produce exactly the frame
//! `encode_request_seq` builds for the same content (CRC field aside, which the
//! fast path leaves zero with the CRC disabled).

use glm53f_wire::frame::{
    encode_request_desc_into, encode_request_meta_into, encode_request_seq, route_entry_wire, HiddenRow, RequestFrame,
    RouteEntry, RowDescriptor,
};
use glm53f_wire::layout::{self, hdr, SourceKind};
use glm53f_wire::naive::WireNaive;

fn lcg(seed: &mut u64) -> u64 {
    *seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
    *seed >> 33
}

#[test]
fn meta_into_matches_encode_request_seq() {
    for &(rows, topk) in &[(1usize, 8usize), (3, 8), (64, 8), (257, 8)] {
        let mut seed = rows as u64 * 7919;
        let routes: Vec<(u32, f32)> =
            (0..rows * topk).map(|i| (((lcg(&mut seed) as usize + i) % 256) as u32, (lcg(&mut seed) % 1000) as f32 / 997.0)).collect();
        // Distinct experts within a row (the wire validator rejects duplicates).
        let routes: Vec<(u32, f32)> =
            routes.iter().enumerate().map(|(i, &(_, w))| ((((i / topk) * 31 + (i % topk) * 17) % 256) as u32, w)).collect();
        let hidden: Vec<HiddenRow> = (0..rows)
            .map(|_| HiddenRow {
                payload: (0..layout::HIDDEN).map(|_| (lcg(&mut seed) & 0x7e) as u8).collect(),
                scales: (0..layout::HIDDEN / 32).map(|_| 110 + (lcg(&mut seed) % 20) as u8).collect(),
            })
            .collect();
        let (request_id, layer_id, executor_id, seq) = (77u64, 13u32, 2u64, 991u64);
        let frame = RequestFrame {
            request_id,
            placement_version: 1,
            layer_id,
            executor_id,
            source_kind: SourceKind::Decode,
            token_position: 0,
            flags: 0,
            seq,
            rows: (0..rows)
                .map(|t| RowDescriptor {
                    row_id: t as u64,
                    source_kind: SourceKind::Decode,
                    source_request_id: request_id,
                    token_position: t as u64,
                    route_offset: (t * topk) as u32,
                    route_count: topk as u32,
                })
                .collect(),
            routes: routes
                .iter()
                .enumerate()
                .map(|(i, &(expert_id, gate_weight))| RouteEntry { row_index: (i / topk) as u32, expert_id, gate_weight })
                .collect(),
            hidden_rows: hidden.clone(),
        };
        let mut want = encode_request_seq(&frame, seq, WireNaive::NONE).expect("encode");
        want[hdr::CRC32C..hdr::CRC32C + 4].fill(0);

        let mut body = vec![0xa5u8; rows * layout::REQUEST_ROW_BYTES + 64];
        let (header, hidden_off, body_len) =
            encode_request_meta_into(&mut body, request_id, layer_id, executor_id, seq, &routes, topk, WireNaive::NONE)
                .expect("meta into");
        for (t, h) in hidden.iter().enumerate() {
            let o = hidden_off + t * layout::HIDDEN_ROW_BYTES;
            body[o..o + layout::HIDDEN].copy_from_slice(&h.payload);
            body[o + layout::HIDDEN..o + layout::HIDDEN_ROW_BYTES].copy_from_slice(&h.scales);
        }
        let mut got = header.to_vec();
        got.extend_from_slice(&body[..body_len]);
        assert_eq!(got.len(), want.len(), "rows {rows}");
        assert!(got == want, "rows {rows}: in-place frame differs from encode_request_seq");

        // Descriptor-only variant: the caller writes the route entries too.
        let mut body2 = vec![0x5au8; rows * layout::REQUEST_ROW_BYTES + 64];
        let (header2, routes_off, hidden_off2, body_len2) =
            encode_request_desc_into(&mut body2, request_id, layer_id, executor_id, seq, rows, topk, WireNaive::NONE)
                .expect("desc into");
        for (i, &(expert_id, gate_weight)) in routes.iter().enumerate() {
            let e = RouteEntry { row_index: (i / topk) as u32, expert_id, gate_weight };
            let o = routes_off + i * layout::ROUTE_ENTRY_LEN;
            body2[o..o + layout::ROUTE_ENTRY_LEN].copy_from_slice(&route_entry_wire(&e, WireNaive::NONE));
        }
        for (t, h) in hidden.iter().enumerate() {
            let o = hidden_off2 + t * layout::HIDDEN_ROW_BYTES;
            body2[o..o + layout::HIDDEN].copy_from_slice(&h.payload);
            body2[o + layout::HIDDEN..o + layout::HIDDEN_ROW_BYTES].copy_from_slice(&h.scales);
        }
        let mut got2 = header2.to_vec();
        got2.extend_from_slice(&body2[..body_len2]);
        assert!(got2 == want, "rows {rows}: descriptor-only frame differs from encode_request_seq");
    }
}

#[test]
fn meta_into_rejects_short_bodies_and_bad_route_counts() {
    let routes = vec![(1u32, 0.5f32); 8];
    let mut small = vec![0u8; 100];
    assert!(encode_request_meta_into(&mut small, 1, 0, 0, 0, &routes, 8, WireNaive::NONE).is_err());
    let mut body = vec![0u8; layout::REQUEST_ROW_BYTES];
    assert!(encode_request_meta_into(&mut body, 1, 0, 0, 0, &routes[..7], 8, WireNaive::NONE).is_err());
}
