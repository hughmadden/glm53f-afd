//! R8 — rank-order `CoordinatorSum` (ADVISOR-I4:487; ADVISOR-I5 §6 kick-start).
//!
//! `CoordinatorSum` must buffer the four rank rows and sum them in fixed rank
//! order 0 → 3, so the FP32 result is bit-reproducible regardless of delivery
//! order. The trap example: `[2^24, 1, −2^24, 1]` sums to 1 in rank order but
//! to 2 if the middle two ranks arrive first (an arrival-order sum).
//!
//! * `r8_rank_order_bit_reproducible` — correct path (`WireNaive::NONE`): every
//!   one of the 24 arrival permutations yields the identical `[1.0]` sum.
//! * `r8_arrival_order_sum_is_the_trap` — naive path
//!   (`WireNaive::RANK_SUM_BY_ARRIVAL`): the same partials produce 1.0 in one
//!   arrival order and 2.0 in another — the non-reproducibility R8 forbids.
//! * `r8_executor_out_of_range_fails_loud` — a rank outside 0..SPARKS is a
//!   protocol error, never an indexed write into the rank buffer.

use glm53f_wire::frame::{ReturnFrame, ReturnRow, FLAG_RETURN_REQUIRED};
use glm53f_wire::layout::{Status, SPARKS, TOPK};
use glm53f_wire::naive::WireNaive;
use glm53f_wire::{bf16, CoordinatorSum};

/// One-rank partial frame for `executor_id` carrying a single BF16 value.
fn rank_frame(executor_id: u64, value: f32) -> ReturnFrame {
    ReturnFrame {
        request_id: 1,
        placement_version: 1,
        layer_id: 2,
        executor_id,
        token_position: 0,
        status: Status::Ok,
        flags: FLAG_RETURN_REQUIRED,
        route_count: TOPK,
        seq: 0,
        rows: vec![ReturnRow { codes: vec![bf16::f32_to_bf16_rne(value)] }],
    }
}

/// The R8 trap values: `[2^24, 1, −2^24, 1]` (rank 0..3).
fn r8_values() -> [f32; SPARKS] {
    let p = (1u32 << 24) as f32; // 2^24, exactly representable in BF16 and FP32
    [p, 1.0, -p, 1.0]
}

/// All 24 permutations of `0..4` (recursive).
fn permutations4() -> Vec<[u64; SPARKS]> {
    fn rec(prefix: &mut Vec<u64>, rem: &mut Vec<u64>, out: &mut Vec<[u64; SPARKS]>) {
        if rem.is_empty() {
            let mut a = [0u64; SPARKS];
            a.copy_from_slice(prefix);
            out.push(a);
            return;
        }
        for i in 0..rem.len() {
            let v = rem.remove(i);
            prefix.push(v);
            rec(prefix, rem, out);
            prefix.pop();
            rem.insert(i, v);
        }
    }
    let mut out = Vec::new();
    rec(&mut Vec::new(), &mut (0..SPARKS as u64).collect(), &mut out);
    out
}

/// Correct path: the FP32 sum is identical for every arrival order.
#[test]
fn r8_rank_order_bit_reproducible() {
    let values = r8_values();
    for order in permutations4() {
        let mut sum = CoordinatorSum::new(1, 1, WireNaive::NONE);
        for &rank in &order {
            sum.accumulate(&rank_frame(rank, values[rank as usize])).expect("accumulate");
        }
        let got = sum.result().expect("complete").to_vec();
        assert_eq!(got, vec![1.0f32], "arrival order {order:?} drifted from rank order");
    }
}

/// Naive path: an arrival-order sum is not reproducible — the trap R8 forbids.
#[test]
fn r8_arrival_order_sum_is_the_trap() {
    let values = r8_values();

    // Arrival order that coincides with rank order sums to 1.
    let mut ordered = CoordinatorSum::new(1, 1, WireNaive::of(WireNaive::RANK_SUM_BY_ARRIVAL));
    for rank in 0..SPARKS as u64 {
        ordered.accumulate(&rank_frame(rank, values[rank as usize])).expect("accumulate");
    }
    assert_eq!(ordered.result().expect("complete"), &[1.0f32]);

    // The same partials, middle two first, sum to 2 — non-reproducible.
    let mut scrambled = CoordinatorSum::new(1, 1, WireNaive::of(WireNaive::RANK_SUM_BY_ARRIVAL));
    for &rank in &[0u64, 2, 1, 3] {
        scrambled.accumulate(&rank_frame(rank, values[rank as usize])).expect("accumulate");
    }
    assert_eq!(scrambled.result().expect("complete"), &[2.0f32]);
}

/// A rank outside 0..SPARKS is a protocol error, not an indexed buffer write.
#[test]
fn r8_executor_out_of_range_fails_loud() {
    let mut sum = CoordinatorSum::new(1, 1, WireNaive::NONE);
    let err = sum.accumulate(&rank_frame(SPARKS as u64, 1.0)).unwrap_err();
    assert!(
        matches!(err, glm53f_wire::WireError::DimMismatch { field: "executor_id", .. }),
        "expected executor_id DimMismatch, got {err:?}"
    );
}
