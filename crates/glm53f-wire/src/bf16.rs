//! BF16 (truncated IEEE 754 binary32) conversions — the compact return dtype.
//!
//! f32 -> BF16 must round to nearest, even on ties; naive truncation biases
//! every Spark partial downward and the coordinator FP32 sum with it (a trap
//! with a negative test). BF16 -> f32 is exact (shift left 16).

use crate::naive::WireNaive;

/// BF16 code -> f32 (exact).
pub fn bf16_to_f32(bits: u16) -> f32 {
    f32::from_bits((bits as u32) << 16)
}

/// f32 -> BF16 code with naive-variant selection.
pub fn f32_to_bf16(value: f32, naive: WireNaive) -> u16 {
    if naive.has(WireNaive::BF16_TRUNCATE) {
        return (value.to_bits() >> 16) as u16;
    }
    f32_to_bf16_rne(value)
}

/// Env-default f32 -> BF16 (NEGATIVE tests; truncates in the naive run).
pub fn f32_to_bf16_env(value: f32) -> u16 {
    f32_to_bf16(value, crate::naive::naive_from_env())
}

/// f32 -> BF16, round-to-nearest-even on ties, NaN -> canonical BF16 qNaN.
pub fn f32_to_bf16_rne(value: f32) -> u16 {
    let bits = value.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let mag = bits & 0x7FFF_FFFF;
    if mag > 0x7F80_0000 {
        // NaN payloads collapse to the canonical BF16 quiet NaN.
        return sign | 0x7FFF;
    }
    let mut rounded = mag.wrapping_add(0x7FFF + ((mag >> 16) & 1));
    if rounded >= 0x7F80_0000 {
        // Rounded up to or past +inf: saturate at +inf.
        rounded = 0x7F80_0000;
    }
    sign | ((rounded >> 16) as u16)
}
