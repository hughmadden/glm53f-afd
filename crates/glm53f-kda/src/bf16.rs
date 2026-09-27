//! bfloat16 conversions with the device's rounding (round to nearest, ties to even).

/// The bfloat16 nearest to `x` (ties to even), as bits. Infinities stay infinite, values
/// beyond the largest finite bfloat16 round to infinity, subnormals are kept, and NaN becomes
/// a quiet NaN.
pub fn from_f32(x: f32) -> u16 {
    let u = x.to_bits();
    if x.is_nan() {
        return ((u >> 16) as u16) | 0x0040;
    }
    let lsb = (u >> 16) & 1;
    (u.wrapping_add(0x7fff + lsb) >> 16) as u16
}

/// The f32 with the same value as bfloat16 `b` (exact).
pub fn to_f32(b: u16) -> f32 {
    f32::from_bits((b as u32) << 16)
}

/// `x` rounded to bfloat16 and widened back (`__bfloat162float(__float2bfloat16_rn(x))`).
pub fn round(x: f32) -> f32 {
    to_f32(from_f32(x))
}

/// Round every value to bfloat16 bits.
pub fn encode(xs: &[f32]) -> Vec<u16> {
    xs.iter().map(|&x| from_f32(x)).collect()
}

/// Widen bfloat16 bits to f32.
pub fn decode(bs: &[u16]) -> Vec<f32> {
    bs.iter().map(|&b| to_f32(b)).collect()
}

/// Distance in bfloat16 units in the last place between two finite values of the same sign
/// class (0 when equal). Values of opposite sign count through zero.
pub fn ulp_distance(a: u16, b: u16) -> u32 {
    fn ordered(x: u16) -> i32 {
        if x & 0x8000 != 0 {
            -((x & 0x7fff) as i32)
        } else {
            x as i32
        }
    }
    (ordered(a) - ordered(b)).unsigned_abs()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rounds_to_nearest_even() {
        assert_eq!(from_f32(1.0), 0x3f80);
        // 1 + 2^-8 is halfway between 1 and 1 + 2^-7: ties to even (1.0).
        assert_eq!(from_f32(1.0 + 1.0 / 256.0), 0x3f80);
        // 1 + 3 * 2^-8 is halfway between 1 + 2^-7 and 1 + 2^-6: ties to even (1 + 2^-6).
        assert_eq!(from_f32(1.0 + 3.0 / 256.0), 0x3f82);
        // Just above the tie rounds up.
        assert_eq!(from_f32(f32::from_bits(0x3f80_8001)), 0x3f81);
        assert_eq!(round(-2.5), -2.5);
        assert_eq!(to_f32(0xc020), -2.5);
    }

    #[test]
    fn specials() {
        assert_eq!(from_f32(f32::INFINITY), 0x7f80);
        assert_eq!(from_f32(f32::NEG_INFINITY), 0xff80);
        assert_eq!(from_f32(f32::MAX), 0x7f80);
        assert!(to_f32(from_f32(f32::NAN)).is_nan());
        assert_eq!(from_f32(0.0), 0x0000);
        assert_eq!(from_f32(-0.0), 0x8000);
        // A subnormal f32 keeps its top bits.
        assert_eq!(from_f32(f32::from_bits(0x0001_8000)), 0x0002);
        assert_eq!(ulp_distance(0x3f80, 0x3f81), 1);
        assert_eq!(ulp_distance(0x0001, 0x8001), 2);
    }
}
