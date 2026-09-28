//! The wire's hidden rows: FP8 E4M3 values with one UE8M0 scale per 32
//! (`Fp8E4m3Ue8m0K32`, 4,096 + 128 bytes per row), and the E4M3 row-scaled
//! format of the prefill reduce-scatter (4,096 bytes + one FP32 scale).
//!
//! E4M3 here is the OCP "FN" variant the coordinator emits: bias 7, no
//! infinities, `S.1111.111` is NaN, largest finite 448. A UE8M0 byte `b` is the
//! power of two `2^(b - 127)`; 255 is NaN.
//!
//! The expert kernel is W4A16: the rank widens each row exactly to FP32,
//! rotates it and rounds it once to FP16 for the tensor cores. See README.md,
//! "Wire rows are FP8, the kernel is W4A16".

use crate::consts::HIDDEN;

/// Scale block width of `Fp8E4m3Ue8m0K32`.
pub const K32: usize = 32;
/// Scale bytes per hidden row.
pub const SCALES_PER_ROW: usize = HIDDEN / K32;
/// Largest finite E4M3 magnitude.
pub const E4M3_MAX: f32 = 448.0;

/// Whether `code` is an E4M3 NaN (`0x7F`, `0xFF`).
pub fn e4m3_is_nan(code: u8) -> bool {
    code & 0x7F == 0x7F
}

/// E4M3 code to `f32` (exact; NaN codes give NaN).
pub fn e4m3_to_f32(code: u8) -> f32 {
    if e4m3_is_nan(code) {
        return f32::NAN;
    }
    let sign = if code & 0x80 != 0 { -1.0f32 } else { 1.0 };
    let e = ((code >> 3) & 0x0F) as i32;
    let m = (code & 0x07) as f32;
    let mag = if e == 0 {
        m * 2f32.powi(-9)
    } else {
        (8.0 + m) * 2f32.powi(e - 10)
    };
    sign * mag
}

/// UE8M0 scale byte to its power of two, `None` for the NaN byte 255.
/// Byte 0 is `2^-127`, an FP32 subnormal, still exact.
pub fn ue8m0_to_f32(b: u8) -> Option<f32> {
    if b == 0xFF {
        return None;
    }
    let e = b as i32 - 127;
    Some(if e >= -126 {
        f32::from_bits(((e + 127) as u32) << 23)
    } else {
        f32::from_bits(1u32 << (e + 149))
    })
}

/// `f32` to E4M3, round to nearest even, saturating to +-448 (the
/// `__nv_cvt_float_to_fp8(x, __NV_SATFINITE, __NV_E4M3)` rule). NaN stays NaN.
pub fn f32_to_e4m3(x: f32) -> u8 {
    let sign: u8 = if x.is_sign_negative() { 0x80 } else { 0 };
    if x.is_nan() {
        return 0x7F;
    }
    let a = x.abs();
    if a >= E4M3_MAX {
        return sign | 0x7E;
    }
    if a < 2f32.powi(-6) {
        // Subnormals are multiples of 2^-9; 8 of them is the smallest normal,
        // which is also code 8, so the encoding stays continuous.
        return sign | round_half_even(a as f64 * 512.0) as u8;
    }
    let e = ((a.to_bits() >> 23) & 0xFF) as i32 - 127; // -6..=8
    let r = round_half_even(a as f64 * 2f64.powi(3 - e)) as i32; // [8, 16]
    let (e, r) = if r == 16 { (e + 1, 8) } else { (e, r) };
    if e > 8 {
        return sign | 0x7E;
    }
    sign | (((e + 7) as u8) << 3) | (r - 8) as u8
}

/// Decode one wire row (4,096 E4M3 + 128 UE8M0) into `out`. NaN values and
/// NaN scales are refused: the rank never computes on a malformed row.
pub fn decode_row(payload: &[u8], scales: &[u8], out: &mut [f32]) -> Result<(), String> {
    if payload.len() != HIDDEN || scales.len() != SCALES_PER_ROW || out.len() != HIDDEN {
        return Err(format!(
            "hidden row: {} payload / {} scale bytes (want {HIDDEN} / {SCALES_PER_ROW})",
            payload.len(),
            scales.len()
        ));
    }
    for (b, &sb) in scales.iter().enumerate() {
        let s = ue8m0_to_f32(sb).ok_or_else(|| format!("hidden row: scale {b} is the UE8M0 NaN byte"))?;
        for k in 0..K32 {
            let c = payload[b * K32 + k];
            if e4m3_is_nan(c) {
                return Err(format!("hidden row: value {} is an E4M3 NaN", b * K32 + k));
            }
            out[b * K32 + k] = e4m3_to_f32(c) * s;
        }
    }
    Ok(())
}

/// Encode a row the way a coordinator does: per 32 values, the smallest power
/// of two scale with `amax / scale <= 448`, then E4M3 of `v / scale`. For
/// tests and benchmarks; the coordinator owns the production encoder.
pub fn encode_row(x: &[f32], payload: &mut [u8], scales: &mut [u8]) {
    assert_eq!(x.len(), HIDDEN);
    assert_eq!(payload.len(), HIDDEN);
    assert_eq!(scales.len(), SCALES_PER_ROW);
    for b in 0..SCALES_PER_ROW {
        let blk = &x[b * K32..(b + 1) * K32];
        let amax = blk.iter().fold(0f32, |m, v| m.max(v.abs()));
        let mut e = -127i32;
        if amax > 0.0 {
            e = (amax as f64 / E4M3_MAX as f64).log2().ceil() as i32;
            // Guard the log2 rounding at exact powers of two.
            while e > -127 && (amax as f64) <= E4M3_MAX as f64 * 2f64.powi(e - 1) {
                e -= 1;
            }
            while (amax as f64) > E4M3_MAX as f64 * 2f64.powi(e) {
                e += 1;
            }
            e = e.clamp(-127, 127);
        }
        scales[b] = (e + 127) as u8;
        let inv = 2f64.powi(-e);
        for k in 0..K32 {
            payload[b * K32 + k] = f32_to_e4m3((blk[k] as f64 * inv) as f32);
        }
    }
}

fn round_half_even(q: f64) -> f64 {
    let f = q.floor();
    let d = q - f;
    if d > 0.5 || (d == 0.5 && f % 2.0 != 0.0) {
        f + 1.0
    } else {
        f
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn e4m3_codes_round_trip() {
        for c in 0u8..=255 {
            if e4m3_is_nan(c) {
                assert!(e4m3_to_f32(c).is_nan());
                continue;
            }
            let v = e4m3_to_f32(c);
            let back = f32_to_e4m3(v);
            // +0 and -0 keep their sign bit.
            assert_eq!(back, c, "code {c:#04x} value {v}");
        }
        assert_eq!(e4m3_to_f32(0x38), 1.0);
        assert_eq!(e4m3_to_f32(0x7E), 448.0);
        assert_eq!(e4m3_to_f32(0x01), 2f32.powi(-9));
    }

    #[test]
    fn e4m3_rounds_to_nearest_even_and_saturates() {
        // 1.0625 is the tie between 1.0 (even) and 1.125.
        assert_eq!(f32_to_e4m3(1.0625), 0x38);
        assert_eq!(f32_to_e4m3(1.1875), 0x3A); // tie between 1.125 (odd) and 1.25 (even)
        assert_eq!(f32_to_e4m3(1000.0), 0x7E);
        assert_eq!(f32_to_e4m3(-1000.0), 0xFE);
        assert_eq!(f32_to_e4m3(f32::INFINITY), 0x7E);
        assert_eq!(f32_to_e4m3(2f32.powi(-10)), 0x00); // tie to even (0)
        assert_eq!(f32_to_e4m3(3.0 * 2f32.powi(-10)), 0x02);
    }

    #[test]
    fn ue8m0_is_a_power_of_two() {
        assert_eq!(ue8m0_to_f32(127), Some(1.0));
        assert_eq!(ue8m0_to_f32(0), Some(2f32.powi(-127)));
        assert_eq!(ue8m0_to_f32(254), Some(2f32.powi(127)));
        assert_eq!(ue8m0_to_f32(255), None);
    }

    #[test]
    fn encode_decode_is_within_half_an_e4m3_step() {
        let x: Vec<f32> = (0..HIDDEN).map(|i| ((i as f32) * 0.37).sin() * (1.0 + (i % 97) as f32)).collect();
        let (mut p, mut s) = (vec![0u8; HIDDEN], vec![0u8; SCALES_PER_ROW]);
        encode_row(&x, &mut p, &mut s);
        let mut y = vec![0f32; HIDDEN];
        decode_row(&p, &s, &mut y).unwrap();
        for b in 0..SCALES_PER_ROW {
            let scale = ue8m0_to_f32(s[b]).unwrap();
            let amax = x[b * 32..b * 32 + 32].iter().fold(0f32, |m, v| m.max(v.abs()));
            assert!(amax <= 448.0 * scale && amax > 224.0 * scale, "block {b}: scale not tight");
            for k in 0..32 {
                let (xv, yv) = (x[b * 32 + k], y[b * 32 + k]);
                let bound = (xv.abs() * 2f32.powi(-4)).max(scale * 2f32.powi(-10));
                assert!((xv - yv).abs() <= bound, "{xv} -> {yv}");
            }
        }
    }

    #[test]
    fn malformed_rows_are_refused() {
        let (p, mut s) = (vec![0x38u8; HIDDEN], vec![127u8; SCALES_PER_ROW]);
        let mut y = vec![0f32; HIDDEN];
        assert!(decode_row(&p, &s, &mut y).is_ok());
        s[5] = 255;
        assert!(decode_row(&p, &s, &mut y).is_err());
        let mut p2 = p.clone();
        p2[77] = 0xFF;
        assert!(decode_row(&p2, &[127u8; SCALES_PER_ROW], &mut y).is_err());
    }
}
