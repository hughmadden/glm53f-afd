//! The TP4 split of an EXL3 expert, on the real trellis format.
//!
//! A synthetic expert at GLM-5.3-Flash's full shape (gate/up 4,096 -> 2,048,
//! down 2,048 -> 4,096) is cut into four rank blocks by `layout::slice_expert`
//! (which applies `glm53f_model::slicing`), and each rank's dequantized weights
//! must be exactly the rank's columns (gate, up) or rows (down) of the unsplit
//! dequantized weights, so the four partial FFN outputs add up to the unsplit
//! expert. A wrong split (tile rows instead of tile columns) must not.

use glm53f_rank::consts::{HIDDEN, INTERMEDIATE, RANK_WIDTH, WORLD};
use glm53f_rank::exl3::{self, MCG_MULT};
use glm53f_rank::half::f64_to_f16;
use glm53f_rank::layout::{self, parts, Exl3Expert, Exl3Linear, DOWN_TILES, EXPERT_BYTES, GATE_UP_TILES};
use glm53f_rank::reference::swiglu;

struct Rng(u64);

impl Rng {
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    fn unit(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }
    fn bytes(&mut self, n: usize) -> Vec<u8> {
        (0..n).map(|_| self.next_u64() as u8).collect()
    }
    /// FP16 scales with random signs and magnitudes in [lo, hi).
    fn scales(&mut self, n: usize, lo: f64, hi: f64) -> Vec<u8> {
        (0..n)
            .flat_map(|_| {
                let s = if self.next_u64() & 1 == 1 { -1.0 } else { 1.0 };
                f64_to_f16(s * (lo + (hi - lo) * self.unit())).to_le_bytes()
            })
            .collect()
    }
}

fn linear(rng: &mut Rng, k: usize, n: usize) -> Exl3Linear {
    Exl3Linear {
        trellis: rng.bytes(k * n / 2),
        suh: rng.scales(k, 0.01, 0.02),
        svh: rng.scales(n, 0.5, 2.0),
        mcg: MCG_MULT.to_le_bytes().to_vec(),
    }
}

fn expert(seed: u64) -> Exl3Expert {
    let mut rng = Rng(seed);
    Exl3Expert {
        gate: linear(&mut rng, HIDDEN, INTERMEDIATE),
        up: linear(&mut rng, HIDDEN, INTERMEDIATE),
        down: linear(&mut rng, INTERMEDIATE, HIDDEN),
    }
}

fn u16s(b: &[u8]) -> Vec<u16> {
    b.chunks_exact(2).map(|c| u16::from_le_bytes([c[0], c[1]])).collect()
}

fn dense(l: &Exl3Linear, k: usize, n: usize) -> Vec<f64> {
    exl3::dequantize(&l.trellis, k / 16, n / 16, &u16s(&l.suh), &u16s(&l.svh))
}

fn assert_close(a: f64, b: f64, what: &str) {
    assert!((a - b).abs() <= 1e-12 * b.abs().max(1e-3), "{what}: {a} vs {b}");
}

#[test]
fn rank_shares_are_exact_slices_of_the_unsplit_expert() {
    let full = expert(0x7E57_0001);
    let wg = dense(&full.gate, HIDDEN, INTERMEDIATE); // [4096][2048]
    let wu = dense(&full.up, HIDDEN, INTERMEDIATE);
    let wd = dense(&full.down, INTERMEDIATE, HIDDEN); // [2048][4096]

    // Two rows of input and the unsplit FFN (no roundings: exactness of the split).
    let mut rng = Rng(0x7E57_0002);
    let xs: Vec<Vec<f64>> = (0..2).map(|_| (0..HIDDEN).map(|_| 4.0 * rng.unit() - 2.0).collect()).collect();
    let ffn_full = |x: &[f64]| -> Vec<f64> {
        let mut act = vec![0f64; INTERMEDIATE];
        for (i, a) in act.iter_mut().enumerate() {
            let (mut g, mut u) = (0f64, 0f64);
            for k in 0..HIDDEN {
                g += x[k] * wg[k * INTERMEDIATE + i];
                u += x[k] * wu[k * INTERMEDIATE + i];
            }
            *a = swiglu(g as f32, u as f32, false) as f64;
        }
        (0..HIDDEN).map(|h| (0..INTERMEDIATE).map(|i| act[i] * wd[i * HIDDEN + h]).sum()).collect()
    };
    let want: Vec<Vec<f64>> = xs.iter().map(|x| ffn_full(x)).collect();
    let mut got = vec![vec![0f64; HIDDEN]; xs.len()];

    for rank in 0..WORLD {
        let mut block = vec![0u8; EXPERT_BYTES];
        layout::slice_expert(&full, rank, &mut block).unwrap();
        let p = parts(&block);
        let c0 = rank * RANK_WIDTH;
        let (kt, nt) = GATE_UP_TILES;
        let rg = exl3::dequantize(p.gate_trellis, kt, nt, &p.gate_suh, &p.gate_svh);
        let ru = exl3::dequantize(p.up_trellis, kt, nt, &p.up_suh, &p.up_svh);
        let (kt, nt) = DOWN_TILES;
        let rd = exl3::dequantize(p.down_trellis, kt, nt, &p.down_suh, &p.down_svh);
        for k in (0..HIDDEN).step_by(97) {
            for j in 0..RANK_WIDTH {
                assert_close(rg[k * RANK_WIDTH + j], wg[k * INTERMEDIATE + c0 + j], "gate");
                assert_close(ru[k * RANK_WIDTH + j], wu[k * INTERMEDIATE + c0 + j], "up");
            }
        }
        for j in 0..RANK_WIDTH {
            for h in (0..HIDDEN).step_by(89) {
                assert_close(rd[j * HIDDEN + h], wd[(c0 + j) * HIDDEN + h], "down");
            }
        }
        // The rank's partial FFN, accumulated over ranks.
        for (x, acc) in xs.iter().zip(got.iter_mut()) {
            let mut act = vec![0f64; RANK_WIDTH];
            for (j, a) in act.iter_mut().enumerate() {
                let (mut g, mut u) = (0f64, 0f64);
                for k in 0..HIDDEN {
                    g += x[k] * rg[k * RANK_WIDTH + j];
                    u += x[k] * ru[k * RANK_WIDTH + j];
                }
                *a = swiglu(g as f32, u as f32, false) as f64;
            }
            for (h, o) in acc.iter_mut().enumerate() {
                *o += (0..RANK_WIDTH).map(|j| act[j] * rd[j * HIDDEN + h]).sum::<f64>();
            }
        }
    }
    for (g, w) in got.iter().zip(&want) {
        let scale = w.iter().map(|v| v.abs()).fold(0f64, f64::max);
        for (a, b) in g.iter().zip(w) {
            assert!((a - b).abs() <= 1e-9 * scale, "sum of rank partials {a} vs unsplit {b}");
        }
    }
}

/// Negative control: the gate trellis cut by tile rows (the down rule) is not
/// the rank's output columns.
#[test]
fn the_wrong_axis_is_caught() {
    let full = expert(0x7E57_0003);
    let wg = dense(&full.gate, HIDDEN, INTERMEDIATE);
    // Tile rows [0, 64) of gate: 64 rows of 128 tiles = the first 1 MiB of the tensor.
    let mut wrong = full.clone();
    let rows = full.gate.trellis[..layout::TRELLIS_BYTES].to_vec();
    wrong.gate.trellis = vec![0u8; full.gate.trellis.len()];
    // Reinterpret those bytes as a [256][32] rank slice (what a wrong split hands the kernel).
    let mut block = vec![0u8; EXPERT_BYTES];
    layout::slice_expert(&wrong, 0, &mut block).unwrap();
    block[layout::GATE_TRELLIS..layout::GATE_TRELLIS + layout::TRELLIS_BYTES].copy_from_slice(&rows);
    let p = parts(&block);
    let (kt, nt) = GATE_UP_TILES;
    let rg = exl3::dequantize(p.gate_trellis, kt, nt, &p.gate_suh, &p.gate_svh);
    let mut max_rel = 0f64;
    for k in (0..HIDDEN).step_by(97) {
        for j in 0..RANK_WIDTH {
            let (a, b) = (rg[k * RANK_WIDTH + j], wg[k * INTERMEDIATE + j]);
            max_rel = max_rel.max((a - b).abs() / b.abs().max(1e-3));
        }
    }
    assert!(max_rel > 0.5, "a row-sliced gate trellis must not reproduce the rank's columns ({max_rel})");
}
