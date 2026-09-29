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
//!
//! Weights the checkpoint ships in BF16 (the KDA projections) can be quantized at load
//! ([`quantize_weight_bf16_as`]) with three kinds of scales ([`Fp8Scales`]): the checkpoint's own
//! (`amax / 448` per 128 x 128 block), powers of two per 128 x 128 block, or MXFP8 (the OCP
//! Microscaling layout: an E8M0 power of two per row and 32 values of K, [`ScaleLayout::Mx32`]).
//! A power-of-two scale only moves the exponent, so a weight with at most 3 significant mantissa
//! bits in range is kept exactly; with `amax / 448` nearly every weight is rounded to E4M3.

/// Weight block edge (both dimensions) and activation group width.
pub const BLOCK: usize = 128;
/// Values of K per MXFP8 scale.
pub const MX_BLOCK: usize = 32;
/// Largest finite E4M3 value.
pub const E4M3_MAX: f32 = 448.0;

/// Where an FP8 weight's scales sit.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum ScaleLayout {
    /// f32 per 128 x 128 block, `[ceil(rows / 128)][ceil(cols / 128)]` (the checkpoint's).
    #[default]
    Block128,
    /// MXFP8: an E8M0 power of two per row and 32 values of K, `[rows][cols / 32]` (one byte on
    /// the device; f32 values here).
    Mx32,
}

/// How a BF16 weight is scaled when it is quantized to FP8 at load.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum Fp8Scales {
    /// 128 x 128 blocks, `scale = amax / 448` (the checkpoint's own scheme).
    #[default]
    Block128,
    /// 128 x 128 blocks, `scale` = the smallest power of two >= `amax / 448` ([`pow2_scale`]).
    Block128Pow2,
    /// MXFP8: 1 x 32 blocks along K, power-of-two scales ([`pow2_scale`]) stored as E8M0 bytes.
    Mx32,
}

impl Fp8Scales {
    /// The layout its scales take.
    pub fn layout(self) -> ScaleLayout {
        match self {
            Fp8Scales::Block128 | Fp8Scales::Block128Pow2 => ScaleLayout::Block128,
            Fp8Scales::Mx32 => ScaleLayout::Mx32,
        }
    }

    /// A short name for logs.
    pub fn describe(self) -> &'static str {
        match self {
            Fp8Scales::Block128 => "FP8 block-128 (amax / 448 scales)",
            Fp8Scales::Block128Pow2 => "FP8 block-128 with power-of-two scales",
            Fp8Scales::Mx32 => "MXFP8 (E8M0 scales per 1 x 32)",
        }
    }
}

/// Bytes of an FP8 weight `[rows][cols]` and its scales on the device.
pub fn weight_bytes(rows: usize, cols: usize, layout: ScaleLayout) -> usize {
    rows * cols
        + match layout {
            ScaleLayout::Block128 => rows.div_ceil(BLOCK) * cols.div_ceil(BLOCK) * 4,
            ScaleLayout::Mx32 => rows * cols.div_ceil(MX_BLOCK),
        }
}

/// The smallest power of two `s >= amax / 448`, clamped to `[2^-126, 2^127]` (1 for an amax of 0
/// or NaN): an E8M0 scale rounded up, so a block's maximum never saturates. `glm53f-layers`'
/// `pow2_scale` in `kernels/common.cuh`, bit for bit.
pub fn pow2_scale(amax: f32) -> f32 {
    if !(amax > 0.0) {
        return 1.0;
    }
    let b = amax.to_bits();
    // amax = m 2^e with m in [1, 2): 2^(e - 8), or 2^(e - 7) when m > 1.75 (448 = 1.75 x 2^8).
    let x = ((b >> 23) as i32 - 135 + i32::from((b & 0x7F_FFFF) > 0x60_0000)).clamp(-126, 127);
    f32::from_bits(((x + 127) as u32) << 23)
}

/// The E8M0 byte of a power-of-two scale in `[2^-126, 2^127]` (its biased exponent).
pub fn e8m0_byte(scale: f32) -> u8 {
    let b = scale.to_bits();
    debug_assert!(b & 0x807F_FFFF == 0 && (1..255).contains(&(b >> 23)));
    (b >> 23) as u8
}

/// An E8M0 byte (1..254) as f32: `2^(b - 127)`.
pub fn e8m0_to_f32(b: u8) -> f32 {
    f32::from_bits((b as u32) << 23)
}

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
/// `[ceil(rows / 128)][ceil(cols / 128)]` (the checkpoint's `weight_scale_inv`), or with MXFP8
/// scales `[rows][cols / 32]` (powers of two, held as f32; [`Fp8Matrix::scale_bytes`] gives the
/// device's E8M0 bytes).
#[derive(Clone, Debug)]
pub struct Fp8Matrix {
    pub rows: usize,
    pub cols: usize,
    pub data: Vec<u8>,
    pub scale_inv: Vec<f32>,
    pub layout: ScaleLayout,
}

impl Fp8Matrix {
    pub fn new(rows: usize, cols: usize, data: Vec<u8>, scale_inv: Vec<f32>) -> Self {
        Self::with_layout(rows, cols, data, scale_inv, ScaleLayout::Block128)
    }

    /// A matrix whose scales take `layout`.
    pub fn with_layout(
        rows: usize,
        cols: usize,
        data: Vec<u8>,
        scale_inv: Vec<f32>,
        layout: ScaleLayout,
    ) -> Self {
        assert_eq!(data.len(), rows * cols, "FP8 data must be rows x cols");
        match layout {
            ScaleLayout::Block128 => assert_eq!(
                scale_inv.len(),
                rows.div_ceil(BLOCK) * cols.div_ceil(BLOCK),
                "scale_inv must be ceil(rows/128) x ceil(cols/128)"
            ),
            ScaleLayout::Mx32 => {
                assert_eq!(cols % MX_BLOCK, 0, "MXFP8 needs cols % 32 == 0");
                assert_eq!(
                    scale_inv.len(),
                    rows * cols / MX_BLOCK,
                    "MXFP8 scales must be rows x cols/32"
                );
                assert!(
                    scale_inv
                        .iter()
                        .all(|&s| e8m0_to_f32(e8m0_byte(s)) == s && s != 0.0),
                    "MXFP8 scales must be powers of two in [2^-126, 2^127]"
                );
            }
        }
        Fp8Matrix {
            rows,
            cols,
            data,
            scale_inv,
            layout,
        }
    }

    /// Columns of the scale grid.
    pub fn scale_cols(&self) -> usize {
        match self.layout {
            ScaleLayout::Block128 => self.cols.div_ceil(BLOCK),
            ScaleLayout::Mx32 => self.cols / MX_BLOCK,
        }
    }

    /// The block scale that applies to element `(r, c)`.
    #[inline]
    pub fn scale(&self, r: usize, c: usize) -> f32 {
        match self.layout {
            ScaleLayout::Block128 => self.scale_inv[(r / BLOCK) * self.scale_cols() + c / BLOCK],
            ScaleLayout::Mx32 => self.scale_inv[r * self.scale_cols() + c / MX_BLOCK],
        }
    }

    /// The scales as the device holds them: f32 (little-endian) per 128 x 128 block, or one E8M0
    /// byte per MXFP8 block.
    pub fn scale_bytes(&self) -> Vec<u8> {
        match self.layout {
            ScaleLayout::Block128 => self.scale_inv.iter().flat_map(|s| s.to_le_bytes()).collect(),
            ScaleLayout::Mx32 => self.scale_inv.iter().map(|&s| e8m0_byte(s)).collect(),
        }
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
        assert!(
            top.layout == ScaleLayout::Block128 && bottom.layout == ScaleLayout::Block128,
            "stacking takes block-128 matrices"
        );
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

/// Quantize a BF16 weight `[rows][cols]` (`cols` a multiple of 128) to E4M3 with one f32 scale
/// per 128 x 128 block: the scheme of the checkpoint's own FP8 weights (`weight_scale_inv =
/// block amax / 448`, 1 for an all-zero block; `q = e4m3(w / scale)` with IEEE division, round
/// to nearest even, saturating). A partial last block of rows has a scale of its own. For the
/// weights the checkpoint ships in BF16 (the KDA projections); `glm53f_fp8_quantize_weight`
/// computes the same bits on the GPU.
pub fn quantize_weight_bf16(w: &[u16], rows: usize, cols: usize) -> Fp8Matrix {
    quantize_weight_bf16_as(w, rows, cols, Fp8Scales::Block128)
}

/// Quantize a BF16 weight `[rows][cols]` (`cols` a multiple of 128) to E4M3 with `scales`:
/// [`Fp8Scales::Block128`] is [`quantize_weight_bf16`]; [`Fp8Scales::Block128Pow2`] the same
/// blocks with [`pow2_scale`] (`glm53f_fp8_quantize_weight_pow2`); [`Fp8Scales::Mx32`] a
/// [`pow2_scale`] per row and 32 values of K (`glm53f_fp8_quantize_weight_mx`). The codes are
/// `e4m3(w / scale)` in every case (exact quotients for powers of two). Bit for bit the kernels.
pub fn quantize_weight_bf16_as(
    w: &[u16],
    rows: usize,
    cols: usize,
    scales: Fp8Scales,
) -> Fp8Matrix {
    assert_eq!(w.len(), rows * cols, "the weight must be rows x cols");
    assert_eq!(
        cols % BLOCK,
        0,
        "the weight's width must be a multiple of 128"
    );
    // Blocks of `bh` rows by `bw` columns, their scales in row-major order.
    let (bh, bw) = match scales {
        Fp8Scales::Block128 | Fp8Scales::Block128Pow2 => (BLOCK, BLOCK),
        Fp8Scales::Mx32 => (1, MX_BLOCK),
    };
    let (sr, sc) = (rows.div_ceil(bh), cols / bw);
    let mut data = vec![0u8; rows * cols];
    let mut scale_inv = vec![0f32; sr * sc];
    for br in 0..sr {
        let rs = br * bh..((br + 1) * bh).min(rows);
        for bc in 0..sc {
            let cs = bc * bw..(bc + 1) * bw;
            let mut amax = 0.0f32;
            for r in rs.clone() {
                for &b in &w[r * cols + cs.start..r * cols + cs.end] {
                    amax = amax.max(crate::bf16::to_f32(b).abs());
                }
            }
            let s = match scales {
                Fp8Scales::Block128 => {
                    if amax > 0.0 {
                        amax / E4M3_MAX
                    } else {
                        1.0
                    }
                }
                Fp8Scales::Block128Pow2 | Fp8Scales::Mx32 => pow2_scale(amax),
            };
            scale_inv[br * sc + bc] = s;
            for r in rs.clone() {
                for c in cs.clone() {
                    data[r * cols + c] = f32_to_e4m3(crate::bf16::to_f32(w[r * cols + c]) / s);
                }
            }
        }
    }
    Fp8Matrix::with_layout(rows, cols, data, scale_inv, scales.layout())
}

/// Rows `row0 .. row0 + rows` of `m` as BF16: `bf16(e4m3(w) * scale)`, one f32 multiply
/// rounded to nearest even (`glm53f_fp8_dequant_bf16`).
pub fn dequant_bf16(m: &Fp8Matrix, row0: usize, rows: usize) -> Vec<u16> {
    assert!(row0 + rows <= m.rows, "rows past the weight");
    let mut out = Vec::with_capacity(rows * m.cols);
    for r in row0..row0 + rows {
        for c in 0..m.cols {
            out.push(crate::bf16::from_f32(m.value(r, c) * m.scale(r, c)));
        }
    }
    out
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
    fn weight_quantization_is_the_checkpoint_scheme() {
        use crate::bf16;
        // 200 x 256: two blocks of columns, a whole and a partial (72-row) block of rows. Block
        // (0, 1) is all zero; block (1, 0) holds an exact power-of-two maximum.
        let (rows, cols) = (200, 256);
        let mut w = vec![0u16; rows * cols];
        let mut s = 7u64;
        for r in 0..rows {
            for c in 0..cols {
                s = s
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                let v = ((s >> 40) as f32 / (1u64 << 24) as f32 - 0.5) * 0.08;
                if !(r < 128 && c >= 128) {
                    w[r * cols + c] = bf16::from_f32(v);
                }
            }
        }
        w[150 * cols + 3] = bf16::from_f32(0.25);
        let m = quantize_weight_bf16(&w, rows, cols);
        assert_eq!(m.scale_inv.len(), 2 * 2);
        assert_eq!(m.scale_inv[1], 1.0, "an all-zero block has scale 1");
        assert!(m.data[..128 * cols]
            .chunks(cols)
            .all(|row| row[128..].iter().all(|&q| q == 0)));
        assert_eq!(m.scale_inv[2], 0.25 / 448.0);
        assert_eq!(
            m.data[150 * cols + 3],
            0x7E,
            "the block maximum codes to 448"
        );
        // Every value within half an E4M3 step of its weight (relative 2^-4 in the normal
        // range, 2^-10 of the scale below it), and the codes equal a plain re-quantization.
        for r in 0..rows {
            for c in 0..cols {
                let x = bf16::to_f32(w[r * cols + c]);
                let sc = m.scale(r, c);
                let back = m.value(r, c) * sc;
                assert!(
                    (back - x).abs() <= x.abs() / 16.0 + sc / 1024.0,
                    "({r}, {c}): {x} -> {back}"
                );
                assert_eq!(m.data[r * cols + c], f32_to_e4m3(x / sc));
            }
        }
        // Dequantization: one rounded product.
        let d = dequant_bf16(&m, 128, 72);
        assert_eq!(d.len(), 72 * cols);
        assert_eq!(
            d[22 * cols + 3],
            bf16::from_f32(e4m3_to_f32(0x7E) * m.scale(150, 3))
        );
        assert_eq!(bf16::to_f32(d[22 * cols + 3]), bf16::round(0.25));
    }

    #[test]
    fn pow2_scales_round_up() {
        // The smallest power of two s with amax / s <= 448, against a search in f64.
        let mut s = 11u64;
        let mut cases: Vec<f32> = vec![448.0, 449.0, 1.0, 1.75, 1.7500001, 1.9999999, 0.25, 3e-3];
        for _ in 0..20_000 {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let e = ((s >> 40) % 60) as i32 - 30;
            cases.push((1.0 + (s >> 41) as f32 / (1u64 << 23) as f32) * 2f32.powi(e));
        }
        for a in cases {
            let p = pow2_scale(a);
            let want = (0..)
                .map(|i| 2f64.powi(i - 140))
                .find(|&x| a as f64 / x <= 448.0)
                .unwrap();
            assert_eq!(p as f64, want, "amax {a}");
            assert_eq!(e8m0_to_f32(e8m0_byte(p)), p);
        }
        assert_eq!(pow2_scale(0.0), 1.0);
        assert_eq!(pow2_scale(f32::NAN), 1.0);
        assert_eq!(pow2_scale(448.0), 1.0);
        assert_eq!(pow2_scale(449.0), 2.0);
        assert_eq!(pow2_scale(1.0), 2f32.powi(-8));
        // Below the normal range the scale stops at 2^-126 (E8M0 byte 1).
        assert_eq!(pow2_scale(1e-40), 2f32.powi(-126));
        assert_eq!(e8m0_byte(pow2_scale(1e-40)), 1);
    }

    #[test]
    fn power_of_two_scales_keep_three_bit_weights_exactly() {
        use crate::bf16;
        // 130 x 256 BF16 weights: most with at most 3 mantissa bits (E4M3's), the rest with 7.
        let (rows, cols) = (130, 256);
        let mut s = 5u64;
        let mut w = vec![0u16; rows * cols];
        for v in w.iter_mut() {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let e = ((s >> 40) % 12) as u16; // exponents 2^-10 .. 2^1
            let m = ((s >> 33) & 0x7F) as u16;
            let m = if (s >> 60) < 13 { m & 0x70 } else { m };
            *v = ((s >> 63) as u16) << 15 | (117 + e) << 7 | m;
        }
        // An all-zero MX group and a 128-block row group.
        for c in 0..32 {
            w[3 * cols + 64 + c] = 0;
        }
        let exact = |m: &Fp8Matrix| {
            (0..rows * cols)
                .filter(|&i| {
                    let (r, c) = (i / cols, i % cols);
                    m.value(r, c) * m.scale(r, c) == bf16::to_f32(w[i])
                })
                .count()
        };
        let three_bit = w.iter().filter(|&&v| v & 0x0F == 0).count();
        let (d2, p2, mx) = (
            quantize_weight_bf16_as(&w, rows, cols, Fp8Scales::Block128),
            quantize_weight_bf16_as(&w, rows, cols, Fp8Scales::Block128Pow2),
            quantize_weight_bf16_as(&w, rows, cols, Fp8Scales::Mx32),
        );
        assert_eq!(d2.data, quantize_weight_bf16(&w, rows, cols).data);
        assert_eq!((p2.layout, mx.layout), (ScaleLayout::Block128, ScaleLayout::Mx32));
        assert_eq!(mx.scale_inv.len(), rows * cols / 32);
        assert_eq!(mx.scale_inv[3 * 8 + 2], 1.0, "an all-zero group has scale 1");
        // Every 3-bit weight is kept by both power-of-two schemes (the range here keeps them
        // above E4M3's subnormals), and the others round to the nearest E4M3 value.
        assert_eq!(exact(&p2), three_bit);
        assert_eq!(exact(&mx), three_bit);
        assert!(exact(&d2) < three_bit / 4, "amax / 448 scales round most weights");
        for m in [&p2, &mx] {
            for i in 0..rows * cols {
                let (r, c) = (i / cols, i % cols);
                let x = bf16::to_f32(w[i]);
                assert!((m.value(r, c) * m.scale(r, c) - x).abs() <= x.abs() / 16.0);
                assert_eq!(m.data[i], f32_to_e4m3(x / m.scale(r, c)));
            }
        }
        // The device's scale bytes: E8M0 for MXFP8.
        let b = mx.scale_bytes();
        assert_eq!(b.len(), rows * cols / 32);
        assert!(b.iter().zip(&mx.scale_inv).all(|(&b, &s)| e8m0_to_f32(b) == s));
        assert_eq!(weight_bytes(rows, cols, ScaleLayout::Mx32), rows * cols + b.len());
        assert_eq!(
            weight_bytes(rows, cols, ScaleLayout::Block128),
            rows * cols + p2.scale_bytes().len()
        );
        // Dequantization of MXFP8 rows: exact (a power of two times 4 significant bits).
        let d = dequant_bf16(&mx, 1, 2);
        for (j, &v) in d.iter().enumerate() {
            let (r, c) = (1 + j / cols, j % cols);
            assert_eq!(bf16::to_f32(v), mx.value(r, c) * mx.scale(r, c));
        }
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
