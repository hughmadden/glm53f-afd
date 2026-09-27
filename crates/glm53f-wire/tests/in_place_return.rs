//! Perf reset R2 zero-copy return: a frame assembled in place (header from
//! `return_header_seq`, BF16 body written after it, `seal_in_place`) is
//! byte-identical to `encode_return_seq` of the same frame.

use glm53f_wire::frame::{encode_return_seq, return_header_seq, seal_in_place, ReturnFrame, ReturnRow, FLAG_RETURN_REQUIRED};
use glm53f_wire::layout::Status;
use glm53f_wire::{WireNaive, HIDDEN};

#[test]
fn in_place_return_matches_encoder() {
    let rows = 3;
    let codes: Vec<Vec<u16>> =
        (0..rows).map(|r| (0..HIDDEN).map(|i| ((r * 7919 + i * 31) & 0xffff) as u16).collect()).collect();
    let f = ReturnFrame {
        request_id: 42,
        placement_version: 1,
        layer_id: 5,
        executor_id: 2,
        token_position: 0,
        status: Status::Ok,
        flags: FLAG_RETURN_REQUIRED,
        route_count: 8,
        seq: 0,
        rows: codes.iter().map(|c| ReturnRow { codes: c.clone() }).collect(),
    };
    let want = encode_return_seq(&f, 17, WireNaive::NONE).unwrap();
    let mut meta = f.clone();
    meta.rows.clear();
    let mut buf = return_header_seq(&meta, rows, 17, WireNaive::NONE).unwrap();
    for c in &codes {
        for &x in c {
            buf.extend_from_slice(&x.to_le_bytes());
        }
    }
    seal_in_place(&mut buf, WireNaive::NONE);
    assert_eq!(buf.len(), want.len());
    assert!(buf == want, "in-place return differs from encode_return_seq");
}
