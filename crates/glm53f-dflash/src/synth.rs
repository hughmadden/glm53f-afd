//! Synthetic context features, bit for bit as `oracle/golden_dflash.py` (`synth_taps`) makes them.
//!
//! The goldens do not run the target model, so their taps are a counter hash: element
//! `(pos, col)` of a `width`-wide row is `h = splitmix64(seed * 2^48 + pos * width + col)`
//! (wrapping 64-bit arithmetic), `v = (h >> 40) / 2^23 - 1` (uniform in [-1, 1), exact in f32),
//! rounded to bfloat16. The fixtures record the digest of what the script produced.

use crate::bf16;

/// splitmix64's finalizer (Steele, Lea and Flood, 2014).
pub fn splitmix64(x: u64) -> u64 {
    let mut z = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// Rows `start..start + rows` of the synthetic taps, as bfloat16 bits `[rows][width]`.
pub fn taps(seed: u64, start: usize, rows: usize, width: usize) -> Vec<u16> {
    let mut out = Vec::with_capacity(rows * width);
    for pos in start..start + rows {
        let base = (seed << 48).wrapping_add((pos as u64).wrapping_mul(width as u64));
        for col in 0..width {
            let h = splitmix64(base.wrapping_add(col as u64));
            let v = (h >> 40) as f64 / (1u64 << 23) as f64 - 1.0;
            out.push(bf16::from_f32(v as f32));
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn splitmix64_known_values() {
        // splitmix64 seeded with 0: the first output of the reference generator (state += golden
        // gamma, then the finalizer) is 0xe220a8397b1dcdaf.
        assert_eq!(splitmix64(0), 0xe220_a839_7b1d_cdaf);
    }

    #[test]
    fn taps_are_bf16_uniform_in_unit_range() {
        let t = taps(1, 0, 4, 64);
        assert_eq!(t.len(), 256);
        let v = bf16::decode(&t);
        assert!(v.iter().all(|&x| (-1.0..=1.0).contains(&x)));
        let mean = v.iter().sum::<f32>() / v.len() as f32;
        assert!(mean.abs() < 0.15, "{mean}");
        // Rows are addressed by position: a later start reproduces the same rows.
        assert_eq!(&taps(1, 2, 2, 64)[..], &t[128..]);
    }
}
