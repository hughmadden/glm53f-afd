//! IEEE binary16 (FP16) and BF16 helpers, bit-level so they match CUDA's
//! `__float2half_rn`, `__half2float` and the EXL3 codebook's one FP16 rounding.
//!
//! Stable Rust has no `f16` type, so FP16 values travel as their `u16` bits.

/// FP16 bits to `f64` (exact).
pub fn f16_to_f64(h: u16) -> f64 {
    let sign = if h & 0x8000 != 0 { -1.0 } else { 1.0 };
    let e = ((h >> 10) & 0x1F) as i32;
    let m = (h & 0x3FF) as f64;
    match e {
        0 => sign * m * pow2(-24),
        31 if m == 0.0 => sign * f64::INFINITY,
        31 => f64::NAN,
        _ => sign * (1024.0 + m) * pow2(e - 25),
    }
}

/// FP16 bits to `f32` (exact).
pub fn f16_to_f32(h: u16) -> f32 {
    f16_to_f64(h) as f32
}

/// `f64` to FP16 bits, rounding to nearest, ties to even (one rounding, as
/// numpy's `astype(float16)` and IEEE FP16 arithmetic do). Overflow goes to
/// infinity, NaN to the canonical quiet NaN.
pub fn f64_to_f16(x: f64) -> u16 {
    let sign: u16 = if x.is_sign_negative() { 0x8000 } else { 0 };
    let a = x.abs();
    if a.is_nan() {
        return sign | 0x7E00;
    }
    // 65,504 is the largest finite value; 65,520 is the tie with 2^16, which
    // rounds to the even neighbour, infinity.
    if a >= 65520.0 {
        return sign | 0x7C00;
    }
    if a == 0.0 {
        return sign;
    }
    let e = ((a.to_bits() >> 52) & 0x7FF) as i32 - 1023;
    if e < -14 {
        // Subnormal: an integer multiple of 2^-24 (1,024 rounds up to the
        // smallest normal, whose bits are the next code).
        return sign | round_half_even(a * pow2(24)) as u16;
    }
    let r = round_half_even(a * pow2(10 - e)) as u32; // in [1024, 2048]
    let (e, r) = if r == 2048 { (e + 1, 1024) } else { (e, r) };
    sign | (((e + 15) as u16) << 10) | (r - 1024) as u16
}

/// `f32` to FP16 bits, round to nearest even (`__float2half_rn`). The widening
/// to `f64` is exact, so this is a single rounding.
pub fn f32_to_f16(x: f32) -> u16 {
    f64_to_f16(x as f64)
}

/// `f32` rounded to BF16 (nearest even) and widened back: the kernels'
/// `bf16r`, the precision the reference model runs its SwiGLU in.
pub fn bf16_round(x: f32) -> f32 {
    glm53f_wire::bf16::bf16_to_f32(glm53f_wire::bf16::f32_to_bf16_rne(x))
}

/// `2^e` for `e` in the normal `f64` range.
pub fn pow2(e: i32) -> f64 {
    debug_assert!((-1022..=1023).contains(&e));
    f64::from_bits(((e + 1023) as u64) << 52)
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
    fn every_f16_round_trips_through_f64() {
        for h in 0u16..=0xFFFF {
            let v = f16_to_f64(h);
            if v.is_nan() {
                continue;
            }
            assert_eq!(f64_to_f16(v), h, "{h:#06x}");
        }
    }

    #[test]
    fn rounding_is_nearest_even() {
        // 1 + 2^-11 is the tie between 1 and 1 + 2^-10: even mantissa (1) wins.
        assert_eq!(f64_to_f16(1.0 + pow2(-11)), 0x3C00);
        // 1 + 3 * 2^-11 ties between 1 + 2^-10 (odd) and 1 + 2^-9 (even).
        assert_eq!(f64_to_f16(1.0 + 3.0 * pow2(-11)), 0x3C02);
        // Just above the tie rounds up.
        assert_eq!(f64_to_f16(1.0 + pow2(-11) + pow2(-30)), 0x3C01);
        // Largest finite, overflow tie, subnormal tie.
        assert_eq!(f64_to_f16(65504.0), 0x7BFF);
        assert_eq!(f64_to_f16(65519.99), 0x7BFF);
        assert_eq!(f64_to_f16(65520.0), 0x7C00);
        assert_eq!(f64_to_f16(pow2(-25)), 0x0000); // half the smallest subnormal: tie to 0
        assert_eq!(f64_to_f16(3.0 * pow2(-25)), 0x0002); // tie between 1 and 2 subnormal steps: 2
        assert_eq!(f64_to_f16(-0.0), 0x8000);
        assert_eq!(f64_to_f16(pow2(-14) * (1.0 - pow2(-12))), 0x0400); // rounds up into the normals
    }

    #[test]
    fn bf16_round_is_rne() {
        assert_eq!(bf16_round(1.0 + 2f32.powi(-8)), 1.0); // tie to even
        assert_eq!(bf16_round(1.0 + 3.0 * 2f32.powi(-8)), 1.0 + 2f32.powi(-6));
    }
}
