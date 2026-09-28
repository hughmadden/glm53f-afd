//! CPU references for one rank's share of the routed-expert FFN.
//!
//! **What the reference model computes.** `Glm5NextTextExperts.forward` in
//! transformers' `modeling_glm5_next.py` (read at main `7cd73d9df0`): for each
//! routed (row, slot) pair,
//!
//! ```text
//! gate, up = x W_gate^T, x W_up^T
//! gate     = gate.clamp(max = swiglu_limit)                    (no lower clamp)
//! up       = up.clamp(min = -swiglu_limit, max = swiglu_limit)
//! y        = (silu(gate) * up) W_down^T * top_k_weights[row, slot]
//! out[row] += y                                                (index_add over the row's slots)
//! ```
//!
//! and `top_k_weights` come from `Glm5NextTextTopkRouter.forward`: sigmoid
//! scores of the top-8 (chosen on score + `e_score_correction_bias`),
//! normalized to sum 1, then multiplied by `routed_scaling_factor` = 2.5. So
//! the gate weight multiplies each expert's down output before the sum over
//! slots, and the 2.5 is already inside it. The shared expert is added after,
//! on the coordinator.
//!
//! **What a rank computes.** Its 512 intermediate channels of every routed
//! expert: `g, u` for those channels, the clamped SwiGLU (channel-wise, so it
//! splits exactly), and a partial down projection over those 512 inputs. The
//! four ranks' partials add up to the expert's down output. A rank returns
//! `sum_slot w[slot] * partial[slot]` per row as BF16; the coordinator adds the
//! four ranks in FP32. The rank applies the wire's FP32 gate weights as they
//! come: if the coordinator sends normalized weights without the 2.5, it
//! multiplies the rank sum by 2.5 instead (the same linear map, README.md).
//!
//! Two references:
//!
//! - [`DenseSlice`] and [`expert_partial_dense`]: the rank's weights
//!   dequantized (`W = diag(suh) H W_q H diag(svh)`, in f64, stored as f32) and
//!   the FFN in plain matrix products, with an option for the reference
//!   model's BF16 roundings of the SwiGLU. This is the model's semantics.
//! - [`KernelSlice`] and [`expert_partial_kernel_order`]: the CUDA kernel's
//!   factorization and roundings step by step (rotate in FP32, round the
//!   rotated input to FP16, multiply by `W_q`, rotate back, TensorFold's BF16
//!   SwiGLU, FP16 down input), with the matrix products accumulated in f64,
//!   the one place where the GPU's tensor cores round differently.

use crate::consts::{HIDDEN, RANK_WIDTH, SWIGLU_LIMIT};
use crate::exl3::{self, fwht128_f32, HAD, HAD_SCALE_F32};
use crate::half::{bf16_round, f16_to_f32, f16_to_f64, f32_to_f16};
use crate::layout::{parts, DOWN_TILES, GATE_UP_TILES};

/// SwiGLU with GLM's limit. `bf16` rounds as the reference model does when it
/// runs in BF16 (gate and up are BF16 matmul outputs, `silu` and the product
/// are rounded to BF16); the kernel does the same, after TensorFold.
pub fn swiglu(g: f32, u: f32, bf16: bool) -> f32 {
    let l = SWIGLU_LIMIT;
    if bf16 {
        let g = bf16_round(g).min(l);
        let u = bf16_round(u).clamp(-l, l);
        bf16_round(bf16_round(g / (1.0 + (-g).exp())) * u)
    } else {
        let g = g.min(l);
        let u = u.clamp(-l, l);
        g / (1.0 + (-g).exp()) * u
    }
}

/// One rank's share of one expert, dequantized.
pub struct DenseSlice {
    /// `W_gate` restricted to the rank's channels, [4096][512] (input-major).
    pub wg: Vec<f32>,
    /// `W_up`, [4096][512].
    pub wu: Vec<f32>,
    /// `W_down` restricted to the rank's input channels, [512][4096].
    pub wd: Vec<f32>,
}

impl DenseSlice {
    /// Dequantize an expert block of a layer image.
    pub fn from_block(block: &[u8]) -> Self {
        let p = parts(block);
        let (kt, nt) = GATE_UP_TILES;
        let wg = exl3::dequantize(p.gate_trellis, kt, nt, &p.gate_suh, &p.gate_svh);
        let wu = exl3::dequantize(p.up_trellis, kt, nt, &p.up_suh, &p.up_svh);
        let (kt, nt) = DOWN_TILES;
        let wd = exl3::dequantize(p.down_trellis, kt, nt, &p.down_suh, &p.down_svh);
        let f = |v: Vec<f64>| v.into_iter().map(|x| x as f32).collect();
        Self { wg: f(wg), wu: f(wu), wd: f(wd) }
    }
}

/// `x` [K] times `w` [K][N] with f64 accumulation.
pub fn matvec(x: &[f32], w: &[f32], n: usize) -> Vec<f64> {
    let mut y = vec![0f64; n];
    for (k, &xv) in x.iter().enumerate() {
        if xv == 0.0 {
            continue;
        }
        let xv = xv as f64;
        for (acc, &wv) in y.iter_mut().zip(&w[k * n..(k + 1) * n]) {
            *acc += xv * wv as f64;
        }
    }
    y
}

/// The rank's partial of one expert's down output for one row (model
/// semantics, dequantized weights). `x` is the row as the rank received it.
pub fn expert_partial_dense(x: &[f32], d: &DenseSlice, bf16: bool) -> Vec<f32> {
    assert_eq!(x.len(), HIDDEN);
    let g = matvec(x, &d.wg, RANK_WIDTH);
    let u = matvec(x, &d.wu, RANK_WIDTH);
    let act: Vec<f32> = g.iter().zip(&u).map(|(&g, &u)| swiglu(g as f32, u as f32, bf16)).collect();
    matvec(&act, &d.wd, HIDDEN).into_iter().map(|v| v as f32).collect()
}

/// One rank's share of one expert, in the form the kernel reads it: `W_q`
/// widened from FP16 to f32 and the scale vectors as f32.
pub struct KernelSlice {
    pub wq_gate: Vec<f32>,
    pub wq_up: Vec<f32>,
    pub wq_down: Vec<f32>,
    pub gate_suh: Vec<f32>,
    pub up_suh: Vec<f32>,
    pub gate_svh: Vec<f32>,
    pub up_svh: Vec<f32>,
    pub down_suh: Vec<f32>,
    pub down_svh: Vec<f32>,
}

impl KernelSlice {
    pub fn from_block(block: &[u8]) -> Self {
        let p = parts(block);
        let wq = |t: &[u8], (kt, nt): (usize, usize)| -> Vec<f32> {
            exl3::unpack(t, kt, nt).into_iter().map(f16_to_f32).collect()
        };
        let f = |v: &[u16]| -> Vec<f32> { v.iter().map(|&h| f16_to_f32(h)).collect() };
        Self {
            wq_gate: wq(p.gate_trellis, GATE_UP_TILES),
            wq_up: wq(p.up_trellis, GATE_UP_TILES),
            wq_down: wq(p.down_trellis, DOWN_TILES),
            gate_suh: f(&p.gate_suh),
            up_suh: f(&p.up_suh),
            gate_svh: f(&p.gate_svh),
            up_svh: f(&p.up_svh),
            down_suh: f(&p.down_suh),
            down_svh: f(&p.down_svh),
        }
    }
}

/// Input rotation as the kernel does it, per 128-block: `v = x * suh` (f32),
/// butterflies in f32, times `HAD_SCALE`, rounded to FP16. Returns the FP16
/// values widened to f32.
pub fn rotate_in(x: &[f32], suh: &[f32]) -> Vec<f32> {
    let mut out = vec![0f32; x.len()];
    let mut v = [0f32; HAD];
    for b in 0..x.len() / HAD {
        for j in 0..HAD {
            v[j] = x[b * HAD + j] * suh[b * HAD + j];
        }
        fwht128_f32(&mut v);
        for j in 0..HAD {
            out[b * HAD + j] = f16_to_f32(f32_to_f16(v[j] * HAD_SCALE_F32));
        }
    }
    out
}

/// The gate/up epilogue of the kernel on one pair: the split sums `zg`, `zu`
/// (512 each), rotated back, scaled by `svh`, the SwiGLU (with the reference
/// model's BF16 roundings when `bf16`), the down input scale and rotation,
/// rounded to FP16. Returns the down GEMM's input (FP16 values as f32). The
/// multiplications keep the kernel's order.
pub fn gateup_epilogue(zg: &[f32], zu: &[f32], ks: &KernelSlice, bf16: bool) -> Vec<f32> {
    let l = SWIGLU_LIMIT;
    let mut out = vec![0f32; RANK_WIDTH];
    let (mut gv, mut uv, mut v) = ([0f32; HAD], [0f32; HAD], [0f32; HAD]);
    for b in 0..RANK_WIDTH / HAD {
        gv.copy_from_slice(&zg[b * HAD..(b + 1) * HAD]);
        uv.copy_from_slice(&zu[b * HAD..(b + 1) * HAD]);
        fwht128_f32(&mut gv);
        fwht128_f32(&mut uv);
        for j in 0..HAD {
            let n = b * HAD + j;
            let (g, u) = (gv[j] * HAD_SCALE_F32 * ks.gate_svh[n], uv[j] * HAD_SCALE_F32 * ks.up_svh[n]);
            let act = if bf16 {
                let gg = bf16_round(g).min(l);
                let uu = bf16_round(u).max(-l).min(l);
                bf16_round(bf16_round(gg / (1.0 + (-gg).exp())) * uu)
            } else {
                let gg = g.min(l);
                let uu = u.max(-l).min(l);
                gg / (1.0 + (-gg).exp()) * uu
            };
            v[j] = act * ks.down_suh[n];
        }
        fwht128_f32(&mut v);
        for j in 0..HAD {
            out[b * HAD + j] = f16_to_f32(f32_to_f16(v[j] * HAD_SCALE_F32));
        }
    }
    out
}

/// The down epilogue of the kernel on one pair: rotate each 128-block of the
/// (split-summed) down output back and scale by `svh`.
pub fn down_epilogue(zd: &[f32], ks: &KernelSlice) -> Vec<f32> {
    let mut y = vec![0f32; HIDDEN];
    let mut v = [0f32; HAD];
    for b in 0..HIDDEN / HAD {
        v.copy_from_slice(&zd[b * HAD..(b + 1) * HAD]);
        fwht128_f32(&mut v);
        for j in 0..HAD {
            y[b * HAD + j] = v[j] * HAD_SCALE_F32 * ks.down_svh[b * HAD + j];
        }
    }
    y
}

/// The rank's partial of one expert's down output for one row, computed in
/// the kernel's order (see the module docs). `x` is the decoded wire row;
/// `bf16` selects the kernel's default BF16 SwiGLU.
pub fn expert_partial_kernel_order(x: &[f32], ks: &KernelSlice, bf16: bool) -> Vec<f32> {
    assert_eq!(x.len(), HIDDEN);
    let ag = rotate_in(x, &ks.gate_suh);
    let au = rotate_in(x, &ks.up_suh);
    let zg: Vec<f32> = matvec(&ag, &ks.wq_gate, RANK_WIDTH).into_iter().map(|v| v as f32).collect();
    let zu: Vec<f32> = matvec(&au, &ks.wq_up, RANK_WIDTH).into_iter().map(|v| v as f32).collect();
    let xd = gateup_epilogue(&zg, &zu, ks, bf16);
    let zd: Vec<f32> = matvec(&xd, &ks.wq_down, HIDDEN).into_iter().map(|v| v as f32).collect();
    down_epilogue(&zd, ks)
}

/// A rank's row: `sum_slot w[slot] * y[slot]` accumulated in f32 in slot
/// order (a multiply then an add, as the kernel's final reduce), then BF16
/// (nearest even). Non-finite sums are an error, as on the device.
pub fn rank_row(ys: &[Vec<f32>], weights: &[f32], out: &mut [u16]) -> Result<(), String> {
    assert_eq!(ys.len(), weights.len());
    assert_eq!(out.len(), HIDDEN);
    for (h, o) in out.iter_mut().enumerate() {
        let mut acc = 0f32;
        for (y, &w) in ys.iter().zip(weights) {
            acc += w * y[h];
        }
        if !acc.is_finite() {
            return Err(format!("rank row: column {h} is not finite"));
        }
        *o = glm53f_wire::bf16::f32_to_bf16_rne(acc);
    }
    Ok(())
}

/// `f64` view of FP16 bits (for callers comparing against [`exl3::forward`]).
pub fn f16s_to_f64(v: &[u16]) -> Vec<f64> {
    v.iter().map(|&h| f16_to_f64(h)).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn swiglu_clamps_gate_above_and_up_both_ways() {
        // gate above the limit is clamped to 10: silu(10) = 10 / (1 + e^-10).
        let s10 = 10.0 / (1.0 + (-10f32).exp());
        assert!((swiglu(50.0, 1.0, false) - s10).abs() < 1e-6);
        // gate below: no clamp (silu of a large negative gate is ~0).
        assert!(swiglu(-50.0, 1.0, false).abs() < 1e-6);
        // up both ways.
        assert!((swiglu(1.0, 50.0, false) - swiglu(1.0, 10.0, false)).abs() < 1e-7);
        assert!((swiglu(1.0, -50.0, false) - swiglu(1.0, -10.0, false)).abs() < 1e-7);
        // BF16 roundings stay within a BF16 step of the exact value.
        for (g, u) in [(0.3f32, 0.7f32), (-2.0, 3.0), (7.5, -9.0)] {
            let (a, b) = (swiglu(g, u, true), swiglu(g, u, false));
            assert!((a - b).abs() <= 3.0 * b.abs() * 2f32.powi(-8) + 1e-6, "{g} {u}: {a} vs {b}");
        }
    }
}
