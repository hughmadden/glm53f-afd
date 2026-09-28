//! FP8 E4M3 codec for the wire quantizer: the host twin of the device encoder in
//! `kernels/wire.cu`. Copied from mimo26f-afd v1.2.0 `crates/mimo26-load/src/e4m3.rs` (the
//! two coding functions and the decode table; PROVENANCE.md).
//!
//! e4m3fn: byte `[s eeee mmm]`, bias 7, no infinities, `S.1111.111` = NaN. Encoding rounds to
//! the nearest representable magnitude, ties to the larger one, and saturates at 448.

pub const E4M3_MAX: f64 = 448.0;

/// mimo26f-afd `spike/quant.py:28-41` — decode one code byte. Table magnitudes are exact
/// in f32; the multiplies accumulate in f64 and cast to f32 at the end, mirroring
/// the numpy codec bit-for-bit (`decode_e4m3` there returns f64).
pub fn decode_e4m3(code: u8) -> f64 {
    if code == 0x7F || code == 0xFF {
        return f64::NAN; // S.1111.111
    }
    let s = if code & 0x80 != 0 { -1.0 } else { 1.0 };
    let e = i32::from((code >> 3) & 0x0F);
    let m = f64::from(code & 0x07);
    let val = if e == 0 {
        (m / 8.0) * 2f64.powi(-6) // subnormal
    } else {
        (1.0 + m / 8.0) * 2f64.powi(e - 7)
    };
    s * val
}

/// Full 256-code decode table (NaN for `0x7f`/`0xff`), cached once. The Spark
/// serve path decodes `[tokens, 4096]` hidden rows per request, so the per-element
/// `decode_e4m3` (which calls `2f64.powi` on every element) was the hot spot;
/// a table lookup is O(1) and bit-identical (computed with the same `decode_e4m3`).
pub fn decode_table() -> &'static [f64] {
    use std::sync::OnceLock;
    static T: OnceLock<Vec<f64>> = OnceLock::new();
    T.get_or_init(|| (0u16..=255).map(|c| decode_e4m3(c as u8)).collect())
}

/// mimo26f-afd `spike/quant.py:52-67` — nearest-representable encoding (search the decode
/// table's monotone magnitudes; ties go to the larger magnitude, as there).
/// O(1) bit-level: the E4M3 magnitudes are monotone in `code`, so the nearest
/// code is read directly from the f64's IEEE-754 exponent/mantissa bits with
/// round-half-up (which is exactly the linear scan's "tie prefers the larger
/// code"). No `log2`/`powi` in the hot path.
pub fn encode_e4m3(value: f64) -> u8 {
    let v = if value.is_nan() { 0.0 } else { value }; // nan_to_num(nan=0)
    let mag = v.abs().min(E4M3_MAX);
    let sign: u8 = if v < 0.0 { 0x80 } else { 0 };
    if mag == 0.0 {
        return sign; // +0/-0 encode to code 0
    }
    let bits = mag.to_bits();
    let exp = ((bits >> 52) & 0x7FF) as i32;
    let mant = bits & 0x000F_FFFF_FFFF_FFFF;
    // Subnormal region: mag < 2^-6 (exp < 1017). m = round(mag * 2^9); m == 0
    // encodes to zero (0x00), m in [1, 7] to the subnormal codes, and m == 8
    // rounds up to the FIRST normal code (e=1,m=0 == 2^-6) — the linear scan's
    // tie-break (larger magnitude) picks that code, not the subnormal clamp.
    if exp < 1017 {
        let m = (mag * 512.0).round() as u8;
        return if m >= 8 { sign | 0x08 } else { sign | m };
    }
    // Normal: mag = (1 + mant/2^52) * 2^(exp-1023). E4M3: e = exp-1016;
    // m = round(mant / 2^49) = (mant + 2^48) >> 49.
    let e = exp - 1016;
    let m = ((mant + 0x0001_0000_0000_0000u64) >> 49) as i32;
    let (e, m) = if m >= 8 {
        (e + 1, 0) // mantissa round-up carries into the exponent
    } else {
        (e, m)
    };
    let e = e.clamp(1, 15) as u8;
    // m in [0, 7]; e=15,m=7 is NaN but never arises (mag <= 448 == e=15,m=6).
    let m = m.clamp(0, 7) as u8;
    sign | (e << 3) | m
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every finite magnitude on a fine grid encodes to the nearest code, ties to the larger
    /// magnitude, saturating at 448; the sign is kept and NaN encodes to 0.
    #[test]
    fn encode_is_nearest_with_ties_to_the_larger_magnitude() {
        let table = decode_table();
        let mags: Vec<(u8, f64)> = (0u8..0x7F).map(|c| (c, table[c as usize])).collect();
        let mut x = 0.0f64;
        while x < 500.0 {
            for v in [x, x * 1.000_001, x * 0.999_999] {
                let want = mags
                    .iter()
                    .min_by(|a, b| {
                        let (da, db) = ((a.1 - v.min(E4M3_MAX)).abs(), (b.1 - v.min(E4M3_MAX)).abs());
                        da.total_cmp(&db).then(b.1.total_cmp(&a.1))
                    })
                    .unwrap()
                    .0;
                assert_eq!(encode_e4m3(v), want, "{v}");
                if v > 0.0 {
                    assert_eq!(encode_e4m3(-v), want | 0x80, "-{v}");
                }
            }
            x = if x < 0.02 { x + 1.0 / 4096.0 } else { x * 1.003 };
        }
        assert_eq!(encode_e4m3(f64::NAN), 0);
        assert_eq!(encode_e4m3(1e9), 0x7E);
        // A tie between two codes goes up: 1.0625 lies halfway between 1.0 and 1.125.
        assert_eq!(decode_e4m3(encode_e4m3(1.0625)), 1.125);
    }

    #[test]
    fn decode_round_trips_every_code() {
        for c in 0u8..=255 {
            let v = decode_e4m3(c);
            if v.is_nan() {
                assert!(c == 0x7F || c == 0xFF);
                continue;
            }
            // Negative zero (0x80) encodes as code 0, like positive zero.
            let want = if c == 0x80 { 0 } else { c };
            assert_eq!(encode_e4m3(v), want, "code {c:#04x} = {v}");
        }
    }
}
