//! BF16 values, carried as their bit patterns (`u16`).

/// Widen a BF16 bit pattern to f32. Exact.
#[inline]
pub fn to_f32(b: u16) -> f32 {
    f32::from_bits((b as u32) << 16)
}

/// Round an f32 to BF16, nearest-even: the rounding of `__float2bfloat16_rn` and of
/// `tensor.to(torch.bfloat16)`. Overflow rounds to infinity; NaN stays a quiet NaN.
#[inline]
pub fn from_f32(x: f32) -> u16 {
    let bits = x.to_bits();
    if x.is_nan() {
        return ((bits >> 16) | 0x0040) as u16;
    }
    let bias = 0x7FFF + ((bits >> 16) & 1);
    ((bits + bias) >> 16) as u16
}

/// Round an f32 to the nearest BF16 value, returned widened to f32.
#[inline]
pub fn round(x: f32) -> f32 {
    to_f32(from_f32(x))
}

/// Widen a slice of BF16 bit patterns.
pub fn widen(v: &[u16]) -> Vec<f32> {
    v.iter().map(|&b| to_f32(b)).collect()
}

/// Round a slice of f32 values to BF16 bit patterns.
pub fn narrow(v: &[f32]) -> Vec<u16> {
    v.iter().map(|&x| from_f32(x)).collect()
}

/// Little-endian bytes of BF16 bit patterns (the device and file layout).
pub fn to_le_bytes(v: &[u16]) -> Vec<u8> {
    v.iter().flat_map(|b| b.to_le_bytes()).collect()
}

/// BF16 bit patterns from little-endian bytes.
pub fn from_le_bytes(bytes: &[u8]) -> Vec<u16> {
    assert!(bytes.len() % 2 == 0, "BF16 byte length must be even");
    bytes
        .chunks_exact(2)
        .map(|c| u16::from_le_bytes([c[0], c[1]]))
        .collect()
}

/// Distance between two BF16 values in units in the last place (same-sign values only;
/// returns `u32::MAX` for a sign mismatch other than ±0, or for NaN).
pub fn ulp_distance(a: u16, b: u16) -> u32 {
    if to_f32(a).is_nan() || to_f32(b).is_nan() {
        return u32::MAX;
    }
    let key = |x: u16| -> i32 {
        if x & 0x8000 != 0 {
            -((x & 0x7FFF) as i32)
        } else {
            x as i32
        }
    };
    (key(a) - key(b)).unsigned_abs()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trip_and_ties() {
        assert_eq!(from_f32(1.0), 0x3F80);
        assert_eq!(to_f32(0x3F80), 1.0);
        // 1 + 2^-8 is exactly between 1 and 1 + 2^-7: ties to even (1.0).
        assert_eq!(from_f32(1.0 + 2f32.powi(-8)), 0x3F80);
        // 1 + 3*2^-8 is between 1 + 2^-7 and 1 + 2^-6: ties to even (1 + 2^-6).
        assert_eq!(from_f32(1.0 + 3.0 * 2f32.powi(-8)), 0x3F82);
        assert_eq!(from_f32(-0.0), 0x8000);
        assert_eq!(from_f32(f32::INFINITY), 0x7F80);
        assert_eq!(from_f32(f32::MAX), 0x7F80);
        assert!(to_f32(from_f32(f32::NAN)).is_nan());
        assert_eq!(ulp_distance(0x3F80, 0x3F81), 1);
        assert_eq!(ulp_distance(0x8000, 0x0000), 0);
        assert_eq!(ulp_distance(0x8001, 0x0001), 2);
    }
}
