//! The prefill reduce-scatter, simulated on the CPU: four ranks' FP32 partial
//! planes go through the protocol of `reduce_scatter` (row partition, peer
//! messages with checked headers, per-rank reduction in rank order, BF16), and
//! every returned row must equal the four-plane sum within the exchange
//! dtype's error bound.
//!
//! The bound, per element of a row owned by rank q:
//! - FP8 row-scaled exchange: `sum over the three peers r != q of
//!   quant_bound(P_r, scale_r(row))` (half an E4M3 step of each exchanged
//!   value);
//! - BF16 exchange: half a BF16 step of each exchanged value;
//!
//! plus half a BF16 step of the sum (the return's rounding) and the FP32
//! additions. The test also prints each path's RMS error next to today's
//! return (four BF16 planes summed by the coordinator).

use glm53f_rank::consts::{HIDDEN, WORLD};
use glm53f_rank::fp8::E4M3_MAX;
use glm53f_rank::reduce_scatter::{outgoing, partition, quant_bound, reduce, Exchange, PeerDtype, PeerHeader, MIN_ROWS};
use glm53f_rank::testkit::Rng;
use glm53f_wire::bf16::{bf16_to_f32, f32_to_bf16_rne};

/// Four partial planes like a rank's reduce output: independent across
/// ranks, mostly unit-scale with a few large channels and a few tiny rows.
fn planes(seed: u64, rows: usize) -> Vec<Vec<f32>> {
    let mut rng = Rng(seed);
    (0..WORLD)
        .map(|_| {
            (0..rows * HIDDEN)
                .map(|i| {
                    let (row, col) = (i / HIDDEN, i % HIDDEN);
                    let big = if col % 331 == 5 { 40.0 } else { 1.0 };
                    let tiny = if row % 13 == 7 { 1e-3 } else { 1.0 };
                    (rng.normal() * big * tiny) as f32
                })
                .collect()
        })
        .collect()
}

fn row_scale(row: &[f32]) -> f32 {
    let amax = row.iter().fold(0f32, |m, v| m.max(v.abs()));
    if amax > 0.0 {
        amax / E4M3_MAX
    } else {
        1.0
    }
}

fn exchange(rows: usize, dtype: PeerDtype) -> Exchange {
    Exchange { request_id: 77, layer: 12, seq: 3, rows, dtype }
}

/// Run the protocol: every rank sends, messages are delivered to their
/// destinations (in a scrambled order), every rank reduces.
fn simulate(p: &[Vec<f32>], x: &Exchange) -> Vec<(usize, Vec<u16>)> {
    let mut inbox: Vec<Vec<Vec<u8>>> = vec![Vec::new(); WORLD];
    for r in (0..WORLD).rev() {
        for m in outgoing(r, x, &p[r]) {
            let (h, _) = PeerHeader::decode(&m).unwrap();
            inbox[h.dst as usize].push(m);
        }
    }
    (0..WORLD).map(|q| reduce(q, x, &p[q], &inbox[q]).unwrap()).collect()
}

#[test]
fn reduced_rows_equal_the_four_plane_sum_within_the_bound() {
    for dtype in [PeerDtype::Fp8RowScaled, PeerDtype::Bf16] {
        for rows in [MIN_ROWS, 37, 256] {
            let p = planes(0x5CA7_0000 + rows as u64, rows);
            let out = simulate(&p, &exchange(rows, dtype));
            let mut covered = vec![0u8; rows];
            let (mut err_sq, mut ref_sq, mut direct_sq, mut worst) = (0f64, 0f64, 0f64, 0f64);
            for (q, (start, bf16)) in out.iter().enumerate() {
                let (s, c) = partition(rows, WORLD, q);
                assert_eq!((*start, bf16.len()), (s, c * HIDDEN), "rank {q} returns its partition");
                for i in 0..c {
                    let row = s + i;
                    covered[row] += 1;
                    let scales: Vec<f32> = (0..WORLD).map(|r| row_scale(&p[r][row * HIDDEN..(row + 1) * HIDDEN])).collect();
                    for h in 0..HIDDEN {
                        let e = row * HIDDEN + h;
                        let exact: f64 = (0..WORLD).map(|r| p[r][e] as f64).sum();
                        let got = bf16_to_f32(bf16[i * HIDDEN + h]) as f64;
                        let exchanged: f64 = (0..WORLD)
                            .filter(|&r| r != q)
                            .map(|r| match dtype {
                                PeerDtype::Fp8RowScaled => quant_bound(p[r][e], scales[r]) as f64,
                                PeerDtype::Bf16 => p[r][e].abs() as f64 * 2f64.powi(-8),
                            })
                            .sum();
                        let half_bf16 = (exact.abs() + exchanged) * 2f64.powi(-8);
                        let fp32: f64 = (0..WORLD).map(|r| p[r][e].abs() as f64).sum::<f64>() * 4.0 * 2f64.powi(-24);
                        let bound = exchanged + half_bf16 + fp32 + 1e-30;
                        assert!((got - exact).abs() <= bound, "{dtype:?} row {row} col {h}: {got} vs {exact} (bound {bound})");
                        worst = worst.max((got - exact).abs() / bound);
                        err_sq += (got - exact) * (got - exact);
                        ref_sq += exact * exact;
                        // Today's return: four BF16 planes summed by the coordinator.
                        let direct: f64 = (0..WORLD).map(|r| bf16_to_f32(f32_to_bf16_rne(p[r][e])) as f64).sum();
                        direct_sq += (direct - exact) * (direct - exact);
                    }
                }
            }
            assert!(covered.iter().all(|&c| c == 1), "every row returned exactly once");
            eprintln!(
                "{dtype:?} exchange, {rows} rows: rms error {:.2e} of the sum's rms (worst {:.2} of the bound); four BF16 planes to the coordinator {:.2e}",
                (err_sq / ref_sq).sqrt(),
                worst,
                (direct_sq / ref_sq).sqrt()
            );
        }
    }
}

#[test]
fn a_missing_duplicate_or_foreign_message_is_refused() {
    let rows = 20;
    let p = planes(0x5CA7_1000, rows);
    let x = Exchange { request_id: 5, layer: 3, seq: 0, rows, dtype: PeerDtype::Fp8RowScaled };
    let mut inbox: Vec<Vec<Vec<u8>>> = vec![Vec::new(); WORLD];
    for r in 0..WORLD {
        for m in outgoing(r, &x, &p[r]) {
            let (h, _) = PeerHeader::decode(&m).unwrap();
            inbox[h.dst as usize].push(m);
        }
    }
    // Missing: rank 1 without rank 3's message.
    let mut short = inbox[1].clone();
    short.retain(|m| PeerHeader::decode(m).unwrap().0.src != 3);
    assert!(reduce(1, &x, &p[1], &short).unwrap_err().contains("no message from rank 3"));
    // Duplicate.
    let mut dup = inbox[2].clone();
    dup.push(dup[0].clone());
    assert!(reduce(2, &x, &p[2], &dup).unwrap_err().contains("two messages"));
    // Another rank's partition, another request, another dtype.
    assert!(reduce(0, &x, &p[0], &inbox[1]).is_err());
    assert!(reduce(1, &Exchange { request_id: 6, ..x }, &p[1], &inbox[1]).is_err());
    assert!(reduce(1, &Exchange { dtype: PeerDtype::Bf16, ..x }, &p[1], &inbox[1]).is_err());
    // Corrupted payload.
    let mut bad = inbox[3].clone();
    let n = bad[0].len();
    bad[0][n - 7] ^= 0x10;
    assert!(reduce(3, &x, &p[3], &bad).unwrap_err().contains("CRC"));
    // Intact, it reduces.
    assert!(reduce(1, &x, &p[1], &inbox[1]).is_ok());
}
