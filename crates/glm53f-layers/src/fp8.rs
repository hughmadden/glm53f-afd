//! FP8 E4M3 values, block-128 weights and per-row activation quantization.
//!
//! GLM-5.3-Flash's official checkpoint stores the dense MLPs, the shared experts, the
//! routed experts and the DSA projections as FP8 E4M3 (the `fn` variant: no infinities,
//! NaN = `S.1111.111`, largest finite 448) with an f32 `weight_scale_inv` per 128 x 128
//! block: the real weight is `e4m3(w[n][k]) * scale_inv[n / 128][k / 128]`. The scales are
//! arbitrary f32 values (`block_amax / 448`), not powers of two.
//!
//! The checkpoint's `quantization_config` declares `activation_scheme: dynamic`: at run time
//! each activation row is quantized per group of 128 values to E4M3 with the scale
//! `amax / 448` ([`quantize_rows`]), and the product is accumulated per 128-wide K block in
//! f32 and scaled by both block scales. [`ActScheme`] selects that (W8A8) or BF16
//! activations against the same FP8 weights (W8A16).

/// Weight block edge (both dimensions) and activation group width.
pub const BLOCK: usize = 128;
/// Largest finite E4M3 value.
pub const E4M3_MAX: f32 = 448.0;

/// How the activations meet the FP8 weights in a projection.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ActScheme {
    /// BF16 activations, FP8 weights (W8A16). Each product is exact in f32.
    Bf16,
    /// Activations quantized per row and 128-group to E4M3 with scale `amax / 448`
    /// (W8A8, the checkpoint's `activation_scheme: dynamic`).
    Fp8Dynamic128,
}

/// Decode an E4M3 byte to f32. Exact.
pub fn e4m3_to_f32(b: u8) -> f32 {
    let exp = (b >> 3) & 0x0F;
    let man = (b & 0x07) as f32;
    if exp == 0x0F && b & 0x07 == 0x07 {
        return f32::NAN;
    }
    let mag = if exp == 0 {
        man * f32::from_bits((127 - 9) << 23) // man * 2^-9
    } else {
        (1.0 + man * 0.125) * f32::from_bits(((exp as u32) + 120) << 23) // 2^(exp-7)
    };
    if b & 0x80 != 0 {
        -mag
    } else {
        mag
    }
}

/// Encode an f32 as E4M3 with round-to-nearest-even and saturation to ±448, as
/// `cvt.rn.satfinite.e4m3x2.f32` does. NaN encodes as `0x7F`; ±inf saturate.
pub fn f32_to_e4m3(x: f32) -> u8 {
    if x.is_nan() {
        return 0x7F;
    }
    let sign: u8 = if x.is_sign_negative() { 0x80 } else { 0 };
    let a = x.abs();
    if a >= E4M3_MAX {
        return sign | 0x7E;
    }
    let bits = a.to_bits();
    let e = (bits >> 23) as i32 - 127;
    if e < -6 {
        // E4M3 subnormal range: value = m * 2^-9, m in 0..=8 (8 is the smallest normal,
        // whose encoding 0x08 continues the subnormal codes).
        let m = (a * 512.0).round_ties_even() as u8;
        return sign | m;
    }
    let mant = bits & 0x7F_FFFF;
    let mut m3 = mant >> 20;
    let rest = mant & 0xF_FFFF;
    let half = 0x8_0000;
    if rest > half || (rest == half && (m3 & 1) == 1) {
        m3 += 1;
    }
    let mut e = e;
    if m3 == 8 {
        m3 = 0;
        e += 1;
    }
    // a < 448 rounds to at most 448 (e = 8, m3 = 6), so no saturation is needed here.
    debug_assert!(e <= 8 && !(e == 8 && m3 == 7));
    sign | (((e + 7) as u8) << 3) | m3 as u8
}

/// A 2-D FP8 E4M3 tensor `[rows][cols]` with f32 block scales
/// `[ceil(rows / 128)][ceil(cols / 128)]` (the checkpoint's `weight_scale_inv`).
#[derive(Clone, Debug)]
pub struct Fp8Matrix {
    pub rows: usize,
    pub cols: usize,
    pub data: Vec<u8>,
    pub scale_inv: Vec<f32>,
}

impl Fp8Matrix {
    pub fn new(rows: usize, cols: usize, data: Vec<u8>, scale_inv: Vec<f32>) -> Self {
        assert_eq!(data.len(), rows * cols, "FP8 data must be rows x cols");
        assert_eq!(
            scale_inv.len(),
            rows.div_ceil(BLOCK) * cols.div_ceil(BLOCK),
            "scale_inv must be ceil(rows/128) x ceil(cols/128)"
        );
        Fp8Matrix {
            rows,
            cols,
            data,
            scale_inv,
        }
    }

    /// Columns of the scale grid.
    pub fn scale_cols(&self) -> usize {
        self.cols.div_ceil(BLOCK)
    }

    /// The block scale that applies to element `(r, c)`.
    #[inline]
    pub fn scale(&self, r: usize, c: usize) -> f32 {
        self.scale_inv[(r / BLOCK) * self.scale_cols() + c / BLOCK]
    }

    /// The E4M3 value of element `(r, c)`, unscaled.
    #[inline]
    pub fn value(&self, r: usize, c: usize) -> f32 {
        e4m3_to_f32(self.data[r * self.cols + c])
    }

    /// Dequantize to f32: `e4m3(w) * scale_inv` (one f32 multiply per element).
    pub fn dequantize(&self) -> Vec<f32> {
        let mut out = Vec::with_capacity(self.rows * self.cols);
        for r in 0..self.rows {
            for c in 0..self.cols {
                out.push(self.value(r, c) * self.scale(r, c));
            }
        }
        out
    }

    /// Stack two matrices with the same column count and 128-aligned row counts (for
    /// example `gate_proj` over `up_proj`), keeping every block scale.
    pub fn stack(top: &Fp8Matrix, bottom: &Fp8Matrix) -> Fp8Matrix {
        assert_eq!(
            top.cols, bottom.cols,
            "stacked matrices need equal column counts"
        );
        assert_eq!(
            top.rows % BLOCK,
            0,
            "the top matrix must have whole 128-row blocks"
        );
        let mut data = top.data.clone();
        data.extend_from_slice(&bottom.data);
        let mut scale_inv = top.scale_inv.clone();
        scale_inv.extend_from_slice(&bottom.scale_inv);
        Fp8Matrix::new(top.rows + bottom.rows, top.cols, data, scale_inv)
    }
}

/// Activation rows quantized per 128-group: E4M3 codes `[rows][cols]` and f32 scales
/// `[rows][cols / 128]`.
#[derive(Clone, Debug)]
pub struct Fp8Rows {
    pub rows: usize,
    pub cols: usize,
    pub q: Vec<u8>,
    pub scale: Vec<f32>,
}

impl Fp8Rows {
    /// The quantized value times its group scale.
    pub fn dequantize(&self) -> Vec<f32> {
        let groups = self.cols / BLOCK;
        (0..self.rows * self.cols)
            .map(|i| {
                let (r, c) = (i / self.cols, i % self.cols);
                e4m3_to_f32(self.q[i]) * self.scale[r * groups + c / BLOCK]
            })
            .collect()
    }
}

/// Quantize one 128-group: `s = amax / 448` (1 if the group is all zero), `q = e4m3(x / s)`.
/// Matches `glm53f_quant_group128` in the kernels bit for bit.
pub fn quantize_group(x: &[f32], q: &mut [u8]) -> f32 {
    debug_assert_eq!(x.len(), q.len());
    let mut amax = 0.0f32;
    for &v in x {
        amax = amax.max(v.abs());
    }
    let s = if amax > 0.0 { amax / E4M3_MAX } else { 1.0 };
    for (qi, &v) in q.iter_mut().zip(x) {
        *qi = f32_to_e4m3(v / s);
    }
    s
}

/// Quantize activation rows `[rows][cols]` per 128-group (`cols` a multiple of 128): the
/// `activation_scheme: dynamic` quantization of a W8A8 block-FP8 linear.
pub fn quantize_rows(x: &[f32], rows: usize, cols: usize) -> Fp8Rows {
    assert_eq!(x.len(), rows * cols);
    assert_eq!(
        cols % BLOCK,
        0,
        "activation width must be a multiple of 128"
    );
    let groups = cols / BLOCK;
    let mut q = vec![0u8; rows * cols];
    let mut scale = vec![0f32; rows * groups];
    for r in 0..rows {
        for g in 0..groups {
            let at = r * cols + g * BLOCK;
            scale[r * groups + g] = quantize_group(&x[at..at + BLOCK], &mut q[at..at + BLOCK]);
        }
    }
    Fp8Rows {
        rows,
        cols,
        q,
        scale,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn known_codes() {
        assert_eq!(e4m3_to_f32(0x00), 0.0);
        assert_eq!(e4m3_to_f32(0x80), -0.0);
        assert!(e4m3_to_f32(0x80).is_sign_negative());
        assert_eq!(e4m3_to_f32(0x01), 2f32.powi(-9));
        assert_eq!(e4m3_to_f32(0x07), 7.0 * 2f32.powi(-9));
        assert_eq!(e4m3_to_f32(0x08), 2f32.powi(-6));
        assert_eq!(e4m3_to_f32(0x38), 1.0);
        assert_eq!(e4m3_to_f32(0x39), 1.125);
        assert_eq!(e4m3_to_f32(0xB8), -1.0);
        assert_eq!(e4m3_to_f32(0x7E), 448.0);
        assert_eq!(e4m3_to_f32(0xFE), -448.0);
        assert!(e4m3_to_f32(0x7F).is_nan());
        assert!(e4m3_to_f32(0xFF).is_nan());
    }

    #[test]
    fn encode_is_inverse_and_rounds_to_even() {
        for c in 0u8..=255 {
            if c & 0x7F == 0x7F {
                continue;
            }
            assert_eq!(f32_to_e4m3(e4m3_to_f32(c)), c, "code {c:#04x}");
        }
        // Halfway between 1.0 (0x38) and 1.125 (0x39): to even (0x38).
        assert_eq!(f32_to_e4m3(1.0625), 0x38);
        // Halfway between 1.125 and 1.25: to even (0x3A).
        assert_eq!(f32_to_e4m3(1.1875), 0x3A);
        // Subnormal halfway 0.5 * 2^-9: to even (0).
        assert_eq!(f32_to_e4m3(2f32.powi(-10)), 0x00);
        assert_eq!(f32_to_e4m3(3.0 * 2f32.powi(-10)), 0x02);
        // Saturation and specials.
        assert_eq!(f32_to_e4m3(460.0), 0x7E);
        assert_eq!(f32_to_e4m3(1e9), 0x7E);
        assert_eq!(f32_to_e4m3(f32::NEG_INFINITY), 0xFE);
        assert_eq!(f32_to_e4m3(f32::NAN), 0x7F);
        assert_eq!(f32_to_e4m3(-0.0), 0x80);
    }
}
