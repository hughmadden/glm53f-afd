//! EXL3 weights, as GLM-5.3-Flash's `tr3-4bpw` checkpoint stores its routed
//! experts: the format and a CPU reference decoder. This is the definition the
//! CUDA kernel is checked against, and it is itself pinned bit for bit to
//! TensorFold's `families/glm5_next/cuda/exl3.py` (`tests/exl3_golden.rs`).
//!
//! The format is ExLlamaV3's (<https://github.com/turboderp-org/exllamav3>, MIT,
//! Copyright (c) 2025 Turboderp), 4 bits per weight with the "mcg" codebook, as
//! TensorFold describes it (MIT, Copyright (c) 2026 TensorFold contributors;
//! `LICENSE.tensorfold`). A linear layer with K inputs and N outputs is four
//! tensors:
//!
//! ```text
//! trellis  int16 [K/16, N/16, 64]  one 16x16 tile per (k tile, n tile), row-major tiles
//! suh      fp16  [K]               input scales (signs and magnitudes)
//! svh      fp16  [N]               output scales
//! mcg      int32 [1]               marker: the "mcg" codebook
//! ```
//!
//! A tile's 64 int16 words, read in pairs as little-endian 32-bit words, form a
//! circular 1,024-bit stream read from the most significant bit of each word.
//! Value `p` (0..256) is the 16 bits of the stream that end at bit `4 (p + 1)`,
//! as an unsigned state `s`, and decodes as
//!
//! ```text
//! x = s * 0xCBAC1FED mod 2^32
//! x = (x & 0x8FFF8FFF) ^ 0x3B603B60
//! value = fp16(x & 0xFFFF) + fp16(x >> 16)        one FP16 rounding, to nearest even
//! ```
//!
//! landing at row `2 (l % 4) + (j & 1) + 8 ((j >> 1) & 1)`, column
//! `l / 4 + 8 (j >> 2)` of its tile, with `l = p / 8` and `j = p % 8` (the
//! tensor-core fragment order). The tiles form `W_q` [K, N], the weight in the
//! rotated domain. With `H` the 128 x 128 Sylvester Hadamard matrix divided by
//! sqrt(128), applied to each block of 128 inputs or outputs:
//!
//! ```text
//! W = diag(suh) H_K W_q H_N diag(svh)        y = ((((x * suh) H_K) W_q) H_N) * svh
//! ```

use crate::half::{f16_to_f64, f64_to_f16};
use std::sync::OnceLock;

/// The "mcg" multiplier (also the value of the `.mcg` marker tensor).
pub const MCG_MULT: u32 = 0xCBAC_1FED;
pub const MCG_MASK: u32 = 0x8FFF_8FFF;
pub const MCG_FLIP: u32 = 0x3B60_3B60;
/// Bits per weight of the checkpoint (K4).
pub const BITS: usize = 4;
/// Tile edge.
pub const TILE: usize = 16;
/// Bytes of one 4-bit tile (64 int16).
pub const TILE_BYTES: usize = TILE * TILE * BITS / 8;
/// 32-bit words of one 4-bit tile.
pub const TILE_WORDS: usize = TILE_BYTES / 4;
/// Hadamard block size.
pub const HAD: usize = 128;
/// `1 / sqrt(128)` as the kernels use it (FP32; TensorFold's `HAD_SCALE`).
#[allow(clippy::excessive_precision)]
pub const HAD_SCALE_F32: f32 = 0.088_388_347_648_318_45;

/// The codebook value (FP16 bits) of a 16-bit state.
pub fn mcg_value(state: u16) -> u16 {
    let x = (state as u32).wrapping_mul(MCG_MULT);
    let x = (x & MCG_MASK) ^ MCG_FLIP;
    // Two FP16 values add exactly in f64; the cast is the one FP16 rounding.
    f64_to_f16(f16_to_f64((x & 0xFFFF) as u16) + f16_to_f64((x >> 16) as u16))
}

/// The whole codebook: FP16 bits of every state, `[65536]`.
pub fn codebook() -> &'static [u16] {
    static CB: OnceLock<Vec<u16>> = OnceLock::new();
    CB.get_or_init(|| (0..=0xFFFFu32).map(|s| mcg_value(s as u16)).collect())
}

/// `(row, column)` within its tile of stream value `p`.
pub fn tile_position(p: usize) -> (usize, usize) {
    let (l, j) = (p / 8, p % 8);
    (2 * (l % 4) + (j & 1) + 8 * ((j >> 1) & 1), l / 4 + 8 * (j >> 2))
}

/// A tile's 32 stream words from its 128 stored bytes (int16 pairs,
/// little-endian).
pub fn tile_words(tile: &[u8]) -> [u32; TILE_WORDS] {
    assert_eq!(tile.len(), TILE_BYTES);
    let mut w = [0u32; TILE_WORDS];
    for (i, c) in tile.chunks_exact(4).enumerate() {
        w[i] = u32::from_le_bytes([c[0], c[1], c[2], c[3]]);
    }
    w
}

/// The 16-bit state of value `p` of a 4-bit tile (exl3.py `states`).
pub fn tile_state(words: &[u32; TILE_WORDS], p: usize) -> u16 {
    let nw = TILE_WORDS;
    let first = p * BITS + BITS + 256 * BITS - 16; // non-negative form of p*bits + bits - 16
    let last = first + 16;
    let i0 = (first / 32) % nw;
    let i1 = ((last - 1) / 32) % nw;
    let shift = ((last - 1) / 32 + 1) * 32 - last;
    let (a, b) = (words[i0] as u64, words[i1] as u64);
    (((a << 32) | b) >> shift) as u16
}

/// Decode one tile (128 stored bytes) into a row-major 16 x 16 block of FP16
/// bits.
pub fn decode_tile(tile: &[u8], out: &mut [u16; TILE * TILE]) {
    let w = tile_words(tile);
    let cb = codebook();
    for p in 0..TILE * TILE {
        let (r, c) = tile_position(p);
        out[r * TILE + c] = cb[tile_state(&w, p) as usize];
    }
}

/// `W_q` [K, N] (FP16 bits, row-major) from a trellis of `kt x nt` tiles
/// (`[kt][nt][128 bytes]`), what ExLlamaV3's `reconstruct` writes.
pub fn unpack(trellis: &[u8], kt: usize, nt: usize) -> Vec<u16> {
    assert_eq!(trellis.len(), kt * nt * TILE_BYTES, "trellis extent");
    let n = nt * TILE;
    let mut w = vec![0u16; kt * TILE * n];
    let mut blk = [0u16; TILE * TILE];
    for a in 0..kt {
        for b in 0..nt {
            let off = (a * nt + b) * TILE_BYTES;
            decode_tile(&trellis[off..off + TILE_BYTES], &mut blk);
            for r in 0..TILE {
                let dst = (a * TILE + r) * n + b * TILE;
                w[dst..dst + TILE].copy_from_slice(&blk[r * TILE..(r + 1) * TILE]);
            }
        }
    }
    w
}

/// Unnormalized fast Walsh-Hadamard transform of 128 values, natural
/// (Sylvester) order, strides 1, 2, 4, ..., 64: `(lo, hi) -> (lo + hi, lo - hi)`.
/// In `f32` this is bit-identical to `fwht128` in TensorFold's `exl3.cu` (the
/// same butterflies, the same order, no fused multiply-add).
pub fn fwht128_f32(v: &mut [f32]) {
    assert_eq!(v.len(), HAD);
    let mut h = 1;
    while h < HAD {
        for i in (0..HAD).step_by(2 * h) {
            for j in i..i + h {
                let (lo, hi) = (v[j], v[j + h]);
                v[j] = lo + hi;
                v[j + h] = lo - hi;
            }
        }
        h *= 2;
    }
}

/// [`fwht128_f32`] in `f64`.
pub fn fwht128_f64(v: &mut [f64]) {
    assert_eq!(v.len(), HAD);
    let mut h = 1;
    while h < HAD {
        for i in (0..HAD).step_by(2 * h) {
            for j in i..i + h {
                let (lo, hi) = (v[j], v[j + h]);
                v[j] = lo + hi;
                v[j + h] = lo - hi;
            }
        }
        h *= 2;
    }
}

/// `x -> x H / sqrt(128)` on every 128-block of `x` (`f64`).
pub fn rotate_f64(x: &mut [f64]) {
    assert_eq!(x.len() % HAD, 0);
    let s = 1.0 / (HAD as f64).sqrt();
    for blk in x.chunks_exact_mut(HAD) {
        fwht128_f64(blk);
        for v in blk.iter_mut() {
            *v *= s;
        }
    }
}

/// The dense weight `W` [K, N] (`f64`, row-major): `diag(suh) H_K W_q H_N diag(svh)`.
/// `suh`, `svh` are FP16 bits.
pub fn dequantize(trellis: &[u8], kt: usize, nt: usize, suh: &[u16], svh: &[u16]) -> Vec<f64> {
    let (k, n) = (kt * TILE, nt * TILE);
    assert_eq!(suh.len(), k, "suh extent");
    assert_eq!(svh.len(), n, "svh extent");
    let wq = unpack(trellis, kt, nt);
    let mut w: Vec<f64> = wq.iter().map(|&h| f16_to_f64(h)).collect();
    // H_K from the left: rotate every column's 128-blocks.
    let mut col = vec![0f64; k];
    for c in 0..n {
        for r in 0..k {
            col[r] = w[r * n + c];
        }
        rotate_f64(&mut col);
        for r in 0..k {
            w[r * n + c] = col[r] * f16_to_f64(suh[r]);
        }
    }
    // H_N from the right: rotate every row's 128-blocks.
    for r in 0..k {
        let row = &mut w[r * n..(r + 1) * n];
        rotate_f64(row);
        for (c, v) in row.iter_mut().enumerate() {
            *v *= f16_to_f64(svh[c]);
        }
    }
    w
}

/// `y = x W` for `rows` rows of `x` [rows, K] (`f64`), computed the way the
/// kernels factor it: rotate the input, multiply by `W_q`, rotate the output.
pub fn forward(x: &[f64], rows: usize, trellis: &[u8], kt: usize, nt: usize, suh: &[u16], svh: &[u16]) -> Vec<f64> {
    let (k, n) = (kt * TILE, nt * TILE);
    assert_eq!(x.len(), rows * k);
    let wq: Vec<f64> = unpack(trellis, kt, nt).iter().map(|&h| f16_to_f64(h)).collect();
    let mut y = vec![0f64; rows * n];
    let mut xh = vec![0f64; k];
    for r in 0..rows {
        for i in 0..k {
            xh[i] = x[r * k + i] * f16_to_f64(suh[i]);
        }
        rotate_f64(&mut xh);
        let yr = &mut y[r * n..(r + 1) * n];
        for i in 0..k {
            let xv = xh[i];
            if xv != 0.0 {
                for (c, w) in yr.iter_mut().zip(&wq[i * n..(i + 1) * n]) {
                    *c += xv * w;
                }
            }
        }
        rotate_f64(yr);
        for (c, v) in yr.iter_mut().enumerate() {
            *v *= f16_to_f64(svh[c]);
        }
    }
    y
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tile_positions_are_a_permutation() {
        let mut seen = [false; 256];
        for p in 0..256 {
            let (r, c) = tile_position(p);
            assert!(!seen[r * 16 + c]);
            seen[r * 16 + c] = true;
        }
    }

    #[test]
    fn codebook_values_are_normal_and_bounded() {
        // (x & 0x8FFF) ^ 0x3B60 keeps each half's exponent in [-3, 0], so each
        // half is normal and below 2 in magnitude; the sum is below 4.
        for &h in codebook() {
            let v = f16_to_f64(h);
            assert!(v.is_finite() && v.abs() < 4.0);
        }
    }

    #[test]
    fn fwht_twice_is_128_times_identity() {
        let mut v: Vec<f64> = (0..128).map(|i| (i as f64 * 0.7).cos()).collect();
        let orig = v.clone();
        fwht128_f64(&mut v);
        fwht128_f64(&mut v);
        for (a, b) in v.iter().zip(&orig) {
            assert!((a - 128.0 * b).abs() < 1e-9);
        }
    }

    #[test]
    fn fwht_matches_the_sylvester_matrix() {
        let x: Vec<f64> = (0..128).map(|i| ((i * 37 % 101) as f64) - 50.0).collect();
        let mut v = x.clone();
        fwht128_f64(&mut v);
        for i in 0..128 {
            let want: f64 = (0..128usize).map(|j| if (i & j).count_ones() % 2 == 1 { -x[j] } else { x[j] }).sum();
            assert_eq!(v[i], want);
        }
    }
}
