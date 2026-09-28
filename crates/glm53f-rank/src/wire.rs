//! Wire return path — the rank's pre-summed partial into a DS41RTE3 v3
//! compact return frame (8,192 B per token, BF16).
//!
//! The rank pre-sums its 8 weighted route partials into one 4,096-wide f32 row
//! per token, BF16-encodes it, and returns a frame carrying the required
//! `SPARK_REDUCTION | V41_COMPACT_BF16` flags (the A6 contract: never per-route
//! FP32). The live 8 KB-per-token assertion is enforced here, not just by the
//! codec's compile-time constants.

use glm53f_wire::bf16::f32_to_bf16;
use glm53f_wire::frame::{encode_return, ReturnFrame, ReturnRow, FLAG_RETURN_REQUIRED};
use glm53f_wire::{HIDDEN, RETURN_ROW_BYTES, Status, WireError, WireNaive};

/// Build the un-stamped compact return frame for a pre-summed rank partial
/// `[tokens, 4096]` f32 (the server stamps the L4 sequence before encoding).
pub fn rank_partial_to_return_frame(
    partial: &[f32],
    tokens: usize,
    request_id: u64,
    placement_version: u64,
    layer_id: u32,
    executor_id: u64,
    token_position: u64,
    naive: WireNaive,
) -> Result<ReturnFrame, WireError> {
    let want = tokens.checked_mul(HIDDEN).ok_or(WireError::DimMismatch {
        field: "partial",
        want: 0,
        got: usize::MAX,
    })?;
    if partial.len() != want {
        return Err(WireError::DimMismatch {
            field: "partial",
            want,
            got: partial.len(),
        });
    }
    let mut rows = Vec::with_capacity(tokens);
    for t in 0..tokens {
        let codes: Vec<u16> = partial[t * HIDDEN..(t + 1) * HIDDEN]
            .iter()
            .map(|&v| f32_to_bf16(v, naive))
            .collect();
        rows.push(ReturnRow { codes });
    }
    Ok(ReturnFrame {
        request_id,
        placement_version,
        layer_id,
        executor_id,
        token_position,
        status: Status::Ok,
        flags: FLAG_RETURN_REQUIRED,
        route_count: 8, // router top-8 pre-summed into each row
        seq: 0,
        rows,
    })
}

/// Build the un-stamped compact return frame from already-BF16 per-token rows
/// (the device route-reduce output). The BF16 codes are used verbatim — no
/// re-encode — so the GPU/CPU bitwise contract holds.
pub fn rank_bf16_to_return_frame(
    bf16: &[u16],
    tokens: usize,
    request_id: u64,
    placement_version: u64,
    layer_id: u32,
    executor_id: u64,
    token_position: u64,
) -> Result<ReturnFrame, WireError> {
    let want = tokens.checked_mul(HIDDEN).ok_or(WireError::DimMismatch {
        field: "bf16",
        want: 0,
        got: usize::MAX,
    })?;
    if bf16.len() != want {
        return Err(WireError::DimMismatch {
            field: "bf16",
            want,
            got: bf16.len(),
        });
    }
    let mut rows = Vec::with_capacity(tokens);
    for t in 0..tokens {
        rows.push(ReturnRow { codes: bf16[t * HIDDEN..(t + 1) * HIDDEN].to_vec() });
    }
    Ok(ReturnFrame {
        request_id,
        placement_version,
        layer_id,
        executor_id,
        token_position,
        status: Status::Ok,
        flags: FLAG_RETURN_REQUIRED,
        route_count: 8,
        seq: 0,
        rows,
    })
}

/// Encode one rank's pre-summed partial `[tokens, 4096]` f32 into a
/// compact return frame, asserting 8,192 B on the wire per token.
pub fn encode_rank_partial(
    partial: &[f32],
    tokens: usize,
    request_id: u64,
    placement_version: u64,
    layer_id: u32,
    executor_id: u64,
    token_position: u64,
    naive: WireNaive,
) -> Result<Vec<u8>, WireError> {
    let frame = rank_partial_to_return_frame(
        partial,
        tokens,
        request_id,
        placement_version,
        layer_id,
        executor_id,
        token_position,
        naive,
    )?;
    let bytes = encode_return(&frame, naive)?;
    // Live 8 KB-per-Spark-per-token assertion (the A6 return contract): the
    // header plus exactly RETURN_ROW_BYTES per token, route-count independent.
    assert_eq!(
        bytes.len(),
        glm53f_wire::HEADER_LEN + tokens * RETURN_ROW_BYTES,
        "return frame is not 8,192 B per token"
    );
    Ok(bytes)
}

#[cfg(test)]
mod tests {
    use super::*;
    use glm53f_wire::decode_frame;
    use glm53f_wire::frame::Frame;

    #[test]
    fn return_is_8192_bytes_per_token_and_round_trips() {
        let tokens = 3;
        let partial: Vec<f32> = (0..tokens * HIDDEN)
            .map(|i| (i as f32 * 0.5).sin())
            .collect();
        let bytes = encode_rank_partial(
            &partial,
            tokens,
            7,
            1,
            3,
            2,
            0,
            WireNaive::NONE,
        )
        .expect("encode");
        assert_eq!(bytes.len(), glm53f_wire::HEADER_LEN + tokens * RETURN_ROW_BYTES);
        assert_eq!(bytes.len(), 128 + tokens * 8192);

        // Round-trip: decode and compare BF16 codes (encode/decode are lossy
        // BF16, so compare the decoded codes against a fresh BF16 re-encode).
        match decode_frame(&bytes, WireNaive::NONE).expect("decode") {
            Frame::Return(frame) => {
                assert_eq!(frame.rows.len(), tokens);
                assert_eq!(frame.route_count, 8);
                assert_eq!(frame.flags & FLAG_RETURN_REQUIRED, FLAG_RETURN_REQUIRED);
                for (t, row) in frame.rows.iter().enumerate() {
                    assert_eq!(row.codes.len(), HIDDEN);
                    let want: Vec<u16> = partial[t * HIDDEN..(t + 1) * HIDDEN]
                        .iter()
                        .map(|&v| f32_to_bf16(v, WireNaive::NONE))
                        .collect();
                    assert_eq!(row.codes, want, "token {t} codes");
                }
            }
            Frame::Request(_) => panic!("expected a return frame"),
        }
    }

    #[test]
    fn rejects_wrong_extent() {
        let err = encode_rank_partial(
            &[0.0f32; HIDDEN + 1],
            1,
            1,
            1,
            1,
            0,
            0,
            WireNaive::NONE,
        )
        .unwrap_err();
        assert!(matches!(err, WireError::DimMismatch { .. }));
    }
}
