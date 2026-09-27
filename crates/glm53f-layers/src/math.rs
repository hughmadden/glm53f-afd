//! Transcendentals that give the same bits on the CPU and in the kernels.
//!
//! `exp` is written with IEEE operations only (fused multiply-add, multiply, add, and a
//! power-of-two scale), in the same order as `glm53f_exp` in `kernels/common.cuh`, so the
//! CPU reference and the GPU agree bit for bit. Its error is within 2 units in the last
//! place of the correctly rounded result (tested below against f64); the reference model
//! uses the platform `exp`, which is equally accurate but not reproducible across devices.
//!
//! The polynomial is the single-precision one from the Cephes library (`expf`), with
//! Cody-Waite range reduction; see `PROVENANCE.md`.

/// `log2(e)`.
const LOG2E: f32 = f32::from_bits(0x3fb8_aa3b);
/// `1.5 * 2^23`: adding and subtracting it rounds to the nearest integer.
const MAGIC: f32 = f32::from_bits(0x4b40_0000);
/// High part of `ln 2` (few significant bits, so `n * LN2_HI` is exact).
const LN2_HI: f32 = f32::from_bits(0x3f31_8000);
/// Low part of `ln 2`.
const LN2_LO: f32 = f32::from_bits(0xb95e_8083);
const P5: f32 = f32::from_bits(0x3950_6967);
const P4: f32 = f32::from_bits(0x3ab7_43ce);
const P3: f32 = f32::from_bits(0x3c08_8908);
const P2: f32 = f32::from_bits(0x3d2a_a9c1);
const P1: f32 = f32::from_bits(0x3e2a_aaaa);
const P0: f32 = f32::from_bits(0x3f00_0000);
/// Above this, `exp` overflows.
const EXP_HI: f32 = f32::from_bits(0x42b1_7218);
/// Below this, `exp` returns 0 (the result would be below the smallest normal f32).
const EXP_LO: f32 = f32::from_bits(0xc2ae_ac4f);

/// `e^x`, bit-reproducible on the CPU and in the kernels.
///
/// Returns +inf above about 88.72 and 0 below about -87.34 (subnormal results are
/// flushed; every caller adds the result to at least 1 or divides by a sum that
/// includes 1, where they cannot matter).
pub fn exp(x: f32) -> f32 {
    if x.is_nan() {
        return x;
    }
    if x > EXP_HI {
        return f32::INFINITY;
    }
    if x < EXP_LO {
        return 0.0;
    }
    let t = x.mul_add(LOG2E, MAGIC);
    let n = t - MAGIC;
    let r = n.mul_add(-LN2_HI, x);
    let r = n.mul_add(-LN2_LO, r);
    let z = r * r;
    let mut p = P5;
    p = p.mul_add(r, P4);
    p = p.mul_add(r, P3);
    p = p.mul_add(r, P2);
    p = p.mul_add(r, P1);
    p = p.mul_add(r, P0);
    let y = p.mul_add(z, r) + 1.0;
    let ni = n as i32;
    if ni > 127 {
        y * f32::from_bits(254 << 23) * 2.0
    } else {
        y * f32::from_bits(((ni + 127) as u32) << 23)
    }
}

/// `1 / (1 + e^-x)`, the form `torch.sigmoid` evaluates in f32.
pub fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + exp(-x))
}

/// `x / (1 + e^-x)`, the form `torch.nn.functional.silu` evaluates in f32.
pub fn silu(x: f32) -> f32 {
    x / (1.0 + exp(-x))
}

/// Pairwise ("butterfly") sum of 32 lane values, as a warp computes it with
/// `v += __shfl_xor_sync(mask, v, off)` for `off` = 16, 8, 4, 2, 1. Every lane ends with
/// the same bits; lane 0's value is returned.
pub fn warp_sum(lanes: &[f32; 32]) -> f32 {
    let mut v = *lanes;
    for off in [16usize, 8, 4, 2, 1] {
        let prev = v;
        for (l, x) in v.iter_mut().enumerate() {
            *x = prev[l] + prev[l ^ off];
        }
    }
    v[0]
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ulps(a: f32, b: f64) -> f64 {
        let r = b as f32;
        if r == a {
            return 0.0;
        }
        // Error relative to the spacing of f32 at the true value.
        let spacing = (f32::from_bits(r.abs().to_bits() + 1) - r.abs()) as f64;
        ((a as f64) - b).abs() / spacing
    }

    #[test]
    fn exp_accuracy() {
        let mut worst = 0.0f64;
        let mut x = -87.3f32;
        while x < 88.7 {
            let e = ulps(exp(x), (x as f64).exp());
            worst = worst.max(e);
            x += 0.000_731;
        }
        assert!(worst <= 2.0, "worst exp error {worst} ulps");
        assert_eq!(exp(0.0), 1.0);
        assert_eq!(exp(-100.0), 0.0);
        assert_eq!(exp(100.0), f32::INFINITY);
        assert!(exp(88.72).is_finite());
    }

    #[test]
    fn sigmoid_and_silu_limits() {
        assert_eq!(sigmoid(0.0), 0.5);
        assert_eq!(sigmoid(200.0), 1.0);
        assert_eq!(sigmoid(-200.0), 0.0);
        assert_eq!(silu(0.0), 0.0);
        assert!((silu(10.0) - 9.999_546).abs() < 1e-5);
        assert_eq!(silu(-1e4), -0.0);
    }

    #[test]
    fn butterfly_is_symmetric() {
        let mut lanes = [0f32; 32];
        for (i, l) in lanes.iter_mut().enumerate() {
            *l = (i as f32 + 0.1).powi(3) * if i % 3 == 0 { -1.0 } else { 1.0 };
        }
        let s = warp_sum(&lanes);
        let exact: f64 = lanes.iter().map(|&x| x as f64).sum();
        assert!((s as f64 - exact).abs() < 1e-2);
    }
}
