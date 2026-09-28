//! bfloat16 conversions: round to nearest, ties to even (the device's `__float2bfloat16_rn` and
//! PyTorch's `.to(torch.bfloat16)`).

/// The bfloat16 nearest to `x` (ties to even), as bits. NaN stays a (quiet) NaN.
pub fn from_f32(x: f32) -> u16 {
    let u = x.to_bits();
    if x.is_nan() {
        return ((u >> 16) as u16) | 0x0040;
    }
    let lsb = (u >> 16) & 1;
    (u.wrapping_add(0x7fff + lsb) >> 16) as u16
}

/// The f32 with the value of bfloat16 `b` (exact).
#[inline]
pub fn to_f32(b: u16) -> f32 {
    f32::from_bits((b as u32) << 16)
}

/// `x` rounded to bfloat16 and widened back.
pub fn round(x: f32) -> f32 {
    to_f32(from_f32(x))
}

/// Widen bfloat16 bits to f32.
pub fn decode(bs: &[u16]) -> Vec<f32> {
    bs.iter().map(|&b| to_f32(b)).collect()
}

/// Round every value to bfloat16 bits.
pub fn encode(xs: &[f32]) -> Vec<u16> {
    xs.iter().map(|&x| from_f32(x)).collect()
}

/// Little-endian bfloat16 bytes (as safetensors stores them) to bits.
pub fn from_le_bytes(raw: &[u8]) -> Vec<u16> {
    raw.as_chunks::<2>()
        .0
        .iter()
        .map(|b| u16::from_le_bytes(*b))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rounding_is_nearest_even() {
        assert_eq!(from_f32(1.0), 0x3f80);
        assert_eq!(to_f32(0x3f80), 1.0);
        // 1 + 2^-8 is halfway between 1 and 1 + 2^-7: ties to even (1.0).
        assert_eq!(from_f32(1.0 + 1.0 / 256.0), 0x3f80);
        // 1 + 3 * 2^-8 is halfway between 1 + 2^-7 and 1 + 2^-6: ties to even (1 + 2^-6).
        assert_eq!(from_f32(1.0 + 3.0 / 256.0), 0x3f82);
        assert_eq!(from_f32(-2.5), 0xc020);
        assert!(to_f32(from_f32(f32::NAN)).is_nan());
        assert_eq!(from_f32(f32::INFINITY), 0x7f80);
        // 0.1 = 0x3dcc_cccd: the low half is above the midpoint, so it rounds up.
        assert_eq!(round(0.1), f32::from_bits(0x3dcd_0000));
    }
}
