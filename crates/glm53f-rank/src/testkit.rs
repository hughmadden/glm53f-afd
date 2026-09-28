//! Synthetic inputs for tests and benchmarks: layer images with the real
//! checkpoint's scale magnitudes, FP8 wire rows, and top-8 routes.
//!
//! The scale vectors follow what the published `tr3-4bpw` checkpoint holds
//! (layer 3, read from its safetensors): gate/up `suh` about 0.016 in
//! magnitude, gate/up `svh` 0.7-2.0, down `suh` 0.010-0.022, down `svh` about
//! 1.0, signs about half negative. Trellis words are uniform random bits, which
//! decode to the codebook's own distribution.

use crate::consts::{EXPERTS, HIDDEN, RANK_WIDTH, TOPK};
use crate::fp8::{encode_row, SCALES_PER_ROW};
use crate::half::f64_to_f16;
use crate::layout::{
    DOWN_SUH, DOWN_SVH, EXPERT_BYTES, GATE_SUH, GATE_SVH, LAYER_BYTES, TRELLIS_BYTES, UP_SUH, UP_SVH,
};

/// SplitMix64.
#[derive(Clone, Debug)]
pub struct Rng(pub u64);

impl Rng {
    pub fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform in [0, 1).
    pub fn unit(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }

    /// Standard normal (Box-Muller).
    pub fn normal(&mut self) -> f64 {
        let u = self.unit().max(1e-300);
        let v = self.unit();
        (-2.0 * u.ln()).sqrt() * (2.0 * std::f64::consts::PI * v).cos()
    }

    pub fn fill(&mut self, out: &mut [u8]) {
        let mut chunks = out.chunks_exact_mut(8);
        for c in &mut chunks {
            c.copy_from_slice(&self.next_u64().to_le_bytes());
        }
        let rest = chunks.into_remainder();
        let last = self.next_u64().to_le_bytes();
        let n = rest.len();
        rest.copy_from_slice(&last[..n]);
    }
}

fn scales(rng: &mut Rng, out: &mut [u8], lo: f64, hi: f64) {
    for c in out.chunks_exact_mut(2) {
        let s = if rng.next_u64() & 1 == 1 { -1.0 } else { 1.0 };
        c.copy_from_slice(&f64_to_f16(s * (lo + (hi - lo) * rng.unit())).to_le_bytes());
    }
}

/// One synthetic expert block.
pub fn expert_block(seed: u64, block: &mut [u8]) {
    assert_eq!(block.len(), EXPERT_BYTES);
    let mut rng = Rng(seed);
    rng.fill(&mut block[..3 * TRELLIS_BYTES]);
    scales(&mut rng, &mut block[GATE_SUH..GATE_SUH + 2 * HIDDEN], 0.0153, 0.0167);
    scales(&mut rng, &mut block[UP_SUH..UP_SUH + 2 * HIDDEN], 0.0153, 0.0167);
    scales(&mut rng, &mut block[GATE_SVH..GATE_SVH + 2 * RANK_WIDTH], 0.75, 2.0);
    scales(&mut rng, &mut block[UP_SVH..UP_SVH + 2 * RANK_WIDTH], 0.75, 1.75);
    scales(&mut rng, &mut block[DOWN_SUH..DOWN_SUH + 2 * RANK_WIDTH], 0.0098, 0.0225);
    scales(&mut rng, &mut block[DOWN_SVH..DOWN_SVH + 2 * HIDDEN], 0.983, 1.014);
}

/// A synthetic layer image (all 288 experts).
pub fn layer_image(seed: u64) -> Vec<u8> {
    let mut image = vec![0u8; LAYER_BYTES];
    for (e, block) in image.chunks_exact_mut(EXPERT_BYTES).enumerate() {
        expert_block(seed ^ (e as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15), block);
    }
    image
}

/// Hidden rows in `f32`, like a post-norm MoE input: mostly unit normal with
/// a few large channels, then encoded as the wire's FP8 rows. Returns
/// (payload [rows * 4096], scales [rows * 128]).
pub fn wire_rows(seed: u64, rows: usize) -> (Vec<u8>, Vec<u8>) {
    let mut rng = Rng(seed);
    let mut payload = vec![0u8; rows * HIDDEN];
    let mut sc = vec![0u8; rows * SCALES_PER_ROW];
    let mut x = vec![0f32; HIDDEN];
    for r in 0..rows {
        for (i, v) in x.iter_mut().enumerate() {
            let big = if i % 509 == 17 { 12.0 } else { 1.0 };
            *v = (rng.normal() * big) as f32;
        }
        encode_row(&x, &mut payload[r * HIDDEN..(r + 1) * HIDDEN], &mut sc[r * SCALES_PER_ROW..(r + 1) * SCALES_PER_ROW]);
    }
    (payload, sc)
}

/// Top-8 routes: 8 distinct experts a row (from `pool`, or all 288 when
/// `pool` is 0) and positive weights summing to 2.5, the routed scale folded
/// in as the reference router does.
pub fn routes(seed: u64, rows: usize, pool: usize) -> (Vec<i32>, Vec<f32>) {
    let pool = if pool == 0 { EXPERTS } else { pool.clamp(TOPK, EXPERTS) };
    let mut rng = Rng(seed);
    let (mut ids, mut w) = (Vec::with_capacity(rows * TOPK), Vec::with_capacity(rows * TOPK));
    for _ in 0..rows {
        let mut pick: Vec<i32> = Vec::with_capacity(TOPK);
        while pick.len() < TOPK {
            let e = (rng.next_u64() % pool as u64) as i32;
            if !pick.contains(&e) {
                pick.push(e);
            }
        }
        let s: Vec<f64> = (0..TOPK).map(|_| 0.05 + rng.unit()).collect();
        let total: f64 = s.iter().sum();
        ids.extend_from_slice(&pick);
        w.extend(s.iter().map(|v| (v / total * 2.5) as f32));
    }
    (ids, w)
}
