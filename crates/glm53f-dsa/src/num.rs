//! Scalar helpers: BF16/F16 rounding, norms, dot products and matrix-vector
//! products in plain f32.
//!
//! The reference runs in f32 with sequential accumulation, so its rounding is
//! reproducible and its error against exact arithmetic is easy to bound.
//! `Rounding::Bf16Ref` adds the BF16 rounding points of the `transformers`
//! reference (which runs in BF16) where they matter for comparisons.

/// Where the reference rounds intermediate values.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Rounding {
    /// Keep every intermediate in f32.
    F32,
    /// Round to BF16 where the `transformers` reference stores a BF16 tensor
    /// (projection outputs, norm outputs, pooled keys, attention probabilities).
    Bf16Ref,
}

impl Rounding {
    /// Round `x` if this mode stores the value as BF16.
    #[inline]
    pub fn at(self, x: f32) -> f32 {
        match self {
            Rounding::F32 => x,
            Rounding::Bf16Ref => bf16_round(x),
        }
    }

    /// Round a slice in place (see [`Rounding::at`]).
    pub fn apply(self, xs: &mut [f32]) {
        if self == Rounding::Bf16Ref {
            for x in xs.iter_mut() {
                *x = bf16_round(*x);
            }
        }
    }
}

/// f32 -> BF16 bits, round to nearest even (NaN stays NaN).
#[inline]
pub fn f32_to_bf16_bits(x: f32) -> u16 {
    let b = x.to_bits();
    if x.is_nan() {
        return ((b >> 16) as u16) | 0x0040;
    }
    let lsb = (b >> 16) & 1;
    let rounded = b.wrapping_add(0x7FFF + lsb);
    (rounded >> 16) as u16
}

/// BF16 bits -> f32 (exact).
#[inline]
pub fn bf16_bits_to_f32(h: u16) -> f32 {
    f32::from_bits((h as u32) << 16)
}

/// Round an f32 to the nearest BF16 value (ties to even).
#[inline]
pub fn bf16_round(x: f32) -> f32 {
    bf16_bits_to_f32(f32_to_bf16_bits(x))
}

/// f32 -> IEEE half bits, round to nearest even; overflow saturates to infinity.
pub fn f32_to_f16_bits(x: f32) -> u16 {
    let b = x.to_bits();
    let sign = ((b >> 16) & 0x8000) as u16;
    let a = b & 0x7FFF_FFFF;
    if a > 0x7F80_0000 {
        return sign | 0x7E00; // NaN
    }
    if a >= 0x477F_F000 {
        // >= 65520 rounds to infinity (65504 is the largest finite half).
        return sign | 0x7C00;
    }
    let exp = (a >> 23) as i32 - 127;
    if exp >= -14 {
        // Normal half: 10 mantissa bits.
        let mant = a & 0x7F_FFFF;
        let mut h = (((exp + 15) as u32) << 10) | (mant >> 13);
        let rem = mant & 0x1FFF;
        if rem > 0x1000 || (rem == 0x1000 && (h & 1) == 1) {
            h += 1;
        }
        return sign | h as u16;
    }
    // Subnormal half: value = m * 2^-24.
    let v = f32::from_bits(a) * 16_777_216.0; // * 2^24, exact scaling
    let q = round_half_even(v) as u32;
    sign | q as u16
}

/// IEEE half bits -> f32 (exact).
pub fn f16_bits_to_f32(h: u16) -> f32 {
    let sign = if h & 0x8000 != 0 { -1.0f32 } else { 1.0 };
    let e = ((h >> 10) & 0x1F) as i32;
    let m = (h & 0x3FF) as f32;
    if e == 0 {
        sign * m * (2.0f32).powi(-24)
    } else if e == 31 {
        if m == 0.0 {
            sign * f32::INFINITY
        } else {
            f32::NAN
        }
    } else {
        sign * (1.0 + m / 1024.0) * (2.0f32).powi(e - 15)
    }
}

/// Round half to even for a non-negative float.
#[inline]
pub fn round_half_even(q: f32) -> f32 {
    let f = q.floor();
    let d = q - f;
    if d > 0.5 {
        f + 1.0
    } else if d < 0.5 {
        f
    } else if (f as u64) % 2 == 0 {
        f
    } else {
        f + 1.0
    }
}

/// Sequential f32 dot product.
#[inline]
pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let mut s = 0.0f32;
    for i in 0..a.len() {
        s += a[i] * b[i];
    }
    s
}

/// Sum of |a_i b_i| in f64 (the magnitude that bounds a dot product's rounding error).
pub fn dot_abs(a: &[f32], b: &[f32]) -> f64 {
    a.iter().zip(b).map(|(x, y)| (*x as f64 * *y as f64).abs()).sum()
}

/// `y = W x` for a row-major `W` of shape `[rows, cols]`.
pub fn matvec(w: &[f32], rows: usize, cols: usize, x: &[f32]) -> Vec<f32> {
    assert_eq!(w.len(), rows * cols, "matvec weight shape");
    assert_eq!(x.len(), cols, "matvec input width");
    (0..rows).map(|r| dot(&w[r * cols..(r + 1) * cols], x)).collect()
}

/// RMSNorm as in the reference: `w * (x * rsqrt(mean(x^2) + eps))`.
pub fn rms_norm(x: &[f32], w: &[f32], eps: f32, rounding: Rounding) -> Vec<f32> {
    assert_eq!(x.len(), w.len());
    let mut ss = 0.0f32;
    for v in x {
        ss += v * v;
    }
    let inv = 1.0 / (ss / x.len() as f32 + eps).sqrt();
    // The reference casts the normalized value to the input dtype before the weight.
    x.iter().zip(w).map(|(v, g)| rounding.at(g * rounding.at(v * inv))).collect()
}

/// LayerNorm with weight and bias (biased variance), as `torch.nn.LayerNorm`.
pub fn layer_norm(x: &[f32], w: &[f32], b: &[f32], eps: f32) -> Vec<f32> {
    assert_eq!(x.len(), w.len());
    assert_eq!(x.len(), b.len());
    let n = x.len() as f32;
    let mut mean = 0.0f32;
    for v in x {
        mean += v;
    }
    mean /= n;
    let mut var = 0.0f32;
    for v in x {
        let d = v - mean;
        var += d * d;
    }
    var /= n;
    let inv = 1.0 / (var + eps).sqrt();
    x.iter().zip(w.iter().zip(b)).map(|(v, (g, c))| (v - mean) * inv * g + c).collect()
}

/// Unit roundoff of f32.
pub const F32_U: f64 = 5.960_464_477_539_063e-8; // 2^-24

/// `gamma_n = n u / (1 - n u)`: the classical bound on relative rounding error
/// of an n-term floating-point sum of products.
pub fn gamma(n: usize) -> f64 {
    let nu = n as f64 * F32_U;
    nu / (1.0 - nu)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bf16_rounding() {
        assert_eq!(bf16_round(1.0), 1.0);
        // 1 + 2^-8 is exactly halfway between 1 and 1 + 2^-7: ties to even (1.0).
        assert_eq!(bf16_round(1.0 + 1.0 / 256.0), 1.0);
        assert_eq!(bf16_round(1.0 + 3.0 / 256.0), 1.0 + 4.0 / 256.0);
        assert!(bf16_round(f32::NAN).is_nan());
        assert_eq!(bf16_round(-2.5), -2.5);
    }

    #[test]
    fn f16_round_trip() {
        for h in 0u16..=0xFFFF {
            let e = (h >> 10) & 0x1F;
            if e == 31 {
                continue;
            }
            let x = f16_bits_to_f32(h);
            let back = f32_to_f16_bits(x);
            // +0/-0 both round-trip to themselves.
            assert_eq!(back, h, "half {h:#06x} -> {x} -> {back:#06x}");
        }
        assert_eq!(f32_to_f16_bits(65504.0), 0x7BFF);
        assert_eq!(f32_to_f16_bits(70000.0), 0x7C00);
        // Halfway between 1 and the next half rounds to even.
        assert_eq!(f32_to_f16_bits(1.0 + 1.0 / 2048.0), 0x3C00);
        assert_eq!(f32_to_f16_bits(1.0 + 3.0 / 2048.0), 0x3C02);
    }

    #[test]
    fn norms() {
        let x = [1.0f32, -2.0, 3.0, -4.0];
        let w = [1.0f32; 4];
        let y = rms_norm(&x, &w, 0.0, Rounding::F32);
        let rms = (30.0f32 / 4.0).sqrt();
        for i in 0..4 {
            assert!((y[i] - x[i] / rms).abs() < 1e-6);
        }
        let z = layer_norm(&x, &w, &[0.5; 4], 0.0);
        let mean: f32 = z.iter().sum::<f32>() / 4.0;
        assert!((mean - 0.5).abs() < 1e-6);
    }
}
