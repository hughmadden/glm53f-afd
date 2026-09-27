//! FP8 E4M3 (`e4m3fn`) codec and block scales.
//!
//! The encoder is round-to-nearest-even with saturation to +-448, the behaviour
//! of the PTX instruction `cvt.rn.satfinite.e4m3x2.f32`; the CUDA writers use
//! that instruction, so a record written on the GPU is byte-identical to the one
//! this module writes (checked by the GPU tests).

/// Largest finite E4M3 magnitude.
pub const E4M3_MAX: f32 = 448.0;

/// Decode one E4M3 byte (NaN is 0x7F / 0xFF; there is no infinity).
#[inline]
pub fn e4m3_decode(b: u8) -> f32 {
    E4M3_TABLE[b as usize]
}

const fn e4m3_value(b: u8) -> f32 {
    let neg = b & 0x80 != 0;
    let e = ((b >> 3) & 0xF) as i32;
    let m = (b & 7) as u32;
    if e == 15 && m == 7 {
        return f32::NAN;
    }
    // value = m * 2^-9 (subnormal) or (8 + m) * 2^(e - 10) (normal); both are exact f32.
    let (num, exp) = if e == 0 { (m, -9) } else { (8 + m, e - 10) };
    let mag = num as f32 * pow2(exp);
    if neg {
        -mag
    } else {
        mag
    }
}

const fn pow2(e: i32) -> f32 {
    // Normal range only (|e| <= 126 here).
    f32::from_bits(((e + 127) as u32) << 23)
}

const fn build_table() -> [f32; 256] {
    let mut t = [0.0f32; 256];
    let mut i = 0;
    while i < 256 {
        t[i] = e4m3_value(i as u8);
        i += 1;
    }
    t
}

/// Decoded value of every E4M3 byte.
pub static E4M3_TABLE: [f32; 256] = build_table();

/// Encode an f32 as E4M3: round to nearest even, saturate to +-448, NaN -> 0x7F.
pub fn e4m3_encode(x: f32) -> u8 {
    if x.is_nan() {
        return 0x7F;
    }
    let sign: u8 = if x.is_sign_negative() { 0x80 } else { 0 };
    let a = x.abs();
    if a >= E4M3_MAX {
        return sign | 0x7E;
    }
    let bits = a.to_bits();
    let exp = ((bits >> 23) & 0xFF) as i32 - 127;
    if exp < -6 {
        // Subnormal range: steps of 2^-9. `a * 512` is exact for normal f32 inputs;
        // f32 subnormals are far below half a step and round to zero.
        let q = crate::num::round_half_even(a * 512.0) as u8; // 0..=8; 8 is the min normal (code 0x08)
        return sign | q;
    }
    let mant = bits & 0x7F_FFFF;
    let mut m3 = mant >> 20;
    let rest = mant & 0xF_FFFF;
    let half = 0x8_0000;
    let mut e = exp + 7;
    if rest > half || (rest == half && (m3 & 1) == 1) {
        m3 += 1;
        if m3 == 8 {
            m3 = 0;
            e += 1;
        }
    }
    // a < 448 never rounds above 448 (e = 15, m = 6).
    debug_assert!(e <= 15 && !(e == 15 && m3 == 7));
    sign | ((e as u8) << 3) | m3 as u8
}

/// How a block scale is chosen.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ScaleMode {
    /// The smallest power of two `s` with `amax <= 448 s` (the default). Decoding
    /// `code * s` is then exact in BF16, F16 (in range) and f32.
    Pow2,
    /// `s = amax / 448` exactly (the arbitrary-f32 convention some kernels use).
    Amax,
}

/// Block scale for a group whose largest magnitude is `amax`.
pub fn block_scale(amax: f32, mode: ScaleMode) -> f32 {
    if !(amax > 0.0) || !amax.is_finite() {
        return 1.0;
    }
    match mode {
        ScaleMode::Amax => {
            let s = amax / E4M3_MAX;
            if s > 0.0 && s.is_normal() {
                s
            } else {
                pow2(-126)
            }
        }
        ScaleMode::Pow2 => pow2(pow2_scale_exponent(amax)),
    }
}

/// Exponent `k` of the smallest power of two with `amax <= 448 * 2^k`, clamped to
/// the normal f32 range. Pure integer logic on the f32 bits (no rounding).
pub fn pow2_scale_exponent(amax: f32) -> i32 {
    let bits = amax.to_bits() & 0x7FFF_FFFF;
    let biased = (bits >> 23) as i32;
    if biased == 0 {
        return -126;
    }
    let e = biased - 127; // amax = 1.m * 2^e
    let mant = bits & 0x7F_FFFF;
    // 448 = 1.75 * 2^8; 1.75 has mantissa bits 0x60_0000.
    let k = if mant <= 0x60_0000 { e - 8 } else { e - 7 };
    k.clamp(-126, 127)
}

/// Quantize `values` with one scale; returns the scale. Codes are written to `out`.
pub fn quantize_block(values: &[f32], out: &mut [u8], mode: ScaleMode) -> f32 {
    assert_eq!(values.len(), out.len());
    let amax = values.iter().fold(0.0f32, |m, v| m.max(v.abs()));
    let s = block_scale(amax, mode);
    match mode {
        ScaleMode::Pow2 => {
            // Multiplying by the exact reciprocal power of two matches the GPU writer.
            let inv = pow2(-pow2_scale_exponent(amax).clamp(-126, 126));
            let inv = if amax > 0.0 && amax.is_finite() { inv } else { 1.0 };
            for (o, v) in out.iter_mut().zip(values) {
                *o = e4m3_encode(v * inv);
            }
        }
        ScaleMode::Amax => {
            for (o, v) in out.iter_mut().zip(values) {
                *o = e4m3_encode(v / s);
            }
        }
    }
    s
}

/// Decode a block: `code * scale` in f32.
pub fn dequantize_block(codes: &[u8], scale: f32, out: &mut [f32]) {
    assert_eq!(codes.len(), out.len());
    for (o, c) in out.iter_mut().zip(codes) {
        *o = e4m3_decode(*c) * scale;
    }
}

/// Quantize and decode `values` in groups of `group` channels (one scale per
/// group): the values a cache of that granularity would return.
pub fn round_trip_grouped(values: &[f32], group: usize, mode: ScaleMode) -> Vec<f32> {
    let mut out = vec![0.0f32; values.len()];
    let mut codes = vec![0u8; group];
    for (v, o) in values.chunks(group).zip(out.chunks_mut(group)) {
        let s = quantize_block(v, &mut codes[..v.len()], mode);
        dequantize_block(&codes[..v.len()], s, o);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn finite_codes() -> Vec<(u8, f32)> {
        (0u16..256)
            .map(|b| (b as u8, e4m3_decode(b as u8)))
            .filter(|(_, v)| !v.is_nan())
            .collect()
    }

    #[test]
    fn decode_table() {
        assert_eq!(e4m3_decode(0x00), 0.0);
        assert_eq!(e4m3_decode(0x01), 2f32.powi(-9));
        assert_eq!(e4m3_decode(0x07), 7.0 * 2f32.powi(-9));
        assert_eq!(e4m3_decode(0x08), 2f32.powi(-6));
        assert_eq!(e4m3_decode(0x38), 1.0);
        assert_eq!(e4m3_decode(0x7E), 448.0);
        assert_eq!(e4m3_decode(0xFE), -448.0);
        assert!(e4m3_decode(0x7F).is_nan());
        assert!(e4m3_decode(0xFF).is_nan());
    }

    #[test]
    fn encode_round_trips_every_code() {
        for (b, v) in finite_codes() {
            let e = e4m3_encode(v);
            if v == 0.0 {
                assert_eq!(e & 0x7F, 0);
            } else {
                assert_eq!(e, b, "value {v}");
            }
        }
    }

    /// Nearest-even against a brute-force search at midpoints and just off them.
    #[test]
    fn encode_is_nearest_even() {
        let mut pos: Vec<(u8, f32)> = finite_codes().into_iter().filter(|(b, _)| b & 0x80 == 0).collect();
        pos.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        for w in pos.windows(2) {
            let (lo_b, lo) = w[0];
            let (hi_b, hi) = w[1];
            let mid = (lo + hi) / 2.0; // exact: both are small dyadic numbers
            let even = if lo_b & 1 == 0 { lo_b } else { hi_b };
            assert_eq!(e4m3_encode(mid), even, "midpoint of {lo} and {hi}");
            let below = f32::from_bits(mid.to_bits() - 1);
            let above = f32::from_bits(mid.to_bits() + 1);
            assert_eq!(e4m3_encode(below), lo_b);
            assert_eq!(e4m3_encode(above), hi_b);
            assert_eq!(e4m3_encode(-mid), even | 0x80);
        }
        assert_eq!(e4m3_encode(460.0), 0x7E);
        assert_eq!(e4m3_encode(1e9), 0x7E);
        assert_eq!(e4m3_encode(f32::INFINITY), 0x7E);
        assert_eq!(e4m3_encode(-f32::INFINITY), 0xFE);
        assert_eq!(e4m3_encode(2f32.powi(-10)), 0x00); // tie to even (0)
        assert_eq!(e4m3_encode(3.0 * 2f32.powi(-10)), 0x02); // 1.5 steps -> 2
        assert_eq!(e4m3_encode(1e-40), 0x00);
    }

    #[test]
    fn pow2_scale_is_minimal() {
        for &amax in &[1e-30f32, 0.001, 0.5, 1.0, 447.0, 448.0, 448.5, 449.0, 1000.0, 3.3e38] {
            let s = block_scale(amax, ScaleMode::Pow2);
            assert!(amax <= 448.0 * s, "amax {amax} scale {s}");
            if s > f32::MIN_POSITIVE {
                assert!(amax > 448.0 * (s / 2.0), "not minimal: amax {amax} scale {s}");
            }
            assert_eq!(s.to_bits() & 0x7F_FFFF, 0, "scale is a power of two");
        }
        assert_eq!(block_scale(0.0, ScaleMode::Pow2), 1.0);
        assert_eq!(block_scale(448.0, ScaleMode::Pow2), 1.0);
        assert_eq!(block_scale(448.0001, ScaleMode::Pow2), 2.0);
    }

    #[test]
    fn block_round_trip_error() {
        let v: Vec<f32> = (0..128).map(|i| ((i as f32) * 0.37).sin() * (1.0 + i as f32)).collect();
        for mode in [ScaleMode::Pow2, ScaleMode::Amax] {
            let mut codes = vec![0u8; 128];
            let s = quantize_block(&v, &mut codes, mode);
            let mut back = vec![0.0f32; 128];
            dequantize_block(&codes, s, &mut back);
            let amax = v.iter().fold(0.0f32, |m, x| m.max(x.abs()));
            for (a, b) in v.iter().zip(&back) {
                // Relative 2^-4 in the normal range; absolute half-step in the subnormal range.
                let tol = (a.abs() / 16.0).max(s * 2f32.powi(-10)) * 1.0001;
                assert!((a - b).abs() <= tol, "{mode:?}: {a} -> {b} (amax {amax}, s {s})");
            }
        }
    }
}
