//! The MoE router: FP32 sigmoid scores, bias-corrected top-8 of 288, normalized and scaled.
//!
//! Reference: `Glm5NextTextTopkRouter` (`moe_router_dtype: float32`, `topk_method:
//! noaux_tc`, `n_group = topk_group = 1`, `norm_topk_prob`, `routed_scaling_factor = 2.5`):
//!
//! - `logits = x_f32 @ W_f32^T` (the router weight is BF16 in the checkpoint);
//! - `scores = sigmoid(logits)`;
//! - selection on `scores + e_score_correction_bias` (the bias only chooses; with one group
//!   the group mask admits every expert);
//! - weights are the **unbiased** scores of the chosen experts, divided by their sum plus
//!   `1e-20`, then multiplied by 2.5.
//!
//! **Determinism.** `torch.topk(sorted=False)` leaves both the order and the choice among
//! exact ties unspecified. Here the experts are chosen in descending order of corrected
//! score, and an exact tie goes to the lower expert index. The weights' sum runs in that
//! order. The logits are dot products in the kernels' order ([`logits_row`]); a row's
//! routing never depends on the other rows in the batch.

use crate::bf16;
use crate::math::{sigmoid, warp_sum};

/// Routed experts of GLM-5.3-Flash (`n_routed_experts`).
pub const EXPERTS: usize = 288;
/// Experts per token (`num_experts_per_tok`).
pub const TOP_K: usize = 8;
/// `routed_scaling_factor`.
pub const ROUTED_SCALE: f32 = 2.5;
/// Added to the weights' sum before dividing, as in the reference.
pub const NORM_EPS: f32 = 1e-20;
/// Hidden positions a warp covers per step: 32 lanes x 8 BF16 values.
pub const WARP_STEP: usize = 256;

/// Router logits of one token in the kernels' order: one warp per expert; lane `l` covers
/// `k = 8 * l + 256 * i + j` (`i` over the steps, `j` in 0..8) with
/// `fma(x[k], w[e][k], acc)`, `i` outer; lanes reduce by butterfly. Each product is exact
/// (BF16 x BF16), so only the sum order differs from an f32 matrix product.
/// `hidden % 256 == 0`.
pub fn logits_row(x: &[f32], weight: &[u16], experts: usize) -> Vec<f32> {
    let hidden = x.len();
    assert_eq!(hidden % WARP_STEP, 0, "hidden must be a multiple of 256");
    assert_eq!(weight.len(), experts * hidden);
    (0..experts)
        .map(|e| {
            let w = &weight[e * hidden..(e + 1) * hidden];
            let mut lanes = [0f32; 32];
            for (l, acc) in lanes.iter_mut().enumerate() {
                for i in 0..hidden / WARP_STEP {
                    for j in 0..8 {
                        let k = 8 * l + WARP_STEP * i + j;
                        *acc = x[k].mul_add(bf16::to_f32(w[k]), *acc);
                    }
                }
            }
            warp_sum(&lanes)
        })
        .collect()
}

/// One token's routing.
#[derive(Clone, Debug, PartialEq)]
pub struct Route {
    /// Chosen experts, in descending order of corrected score.
    pub ids: Vec<u32>,
    /// Their weights (normalized unbiased scores times the routed scale).
    pub weights: Vec<f32>,
}

/// Choose `top_k` experts from f32 logits.
pub fn select_row(logits: &[f32], bias: &[f32], top_k: usize, scale: f32) -> Route {
    assert_eq!(logits.len(), bias.len());
    assert!(top_k <= logits.len());
    let scores: Vec<f32> = logits.iter().map(|&l| sigmoid(l)).collect();
    let mut corrected: Vec<f32> = scores.iter().zip(bias).map(|(&s, &b)| s + b).collect();
    let mut ids = Vec::with_capacity(top_k);
    for _ in 0..top_k {
        let mut best = 0usize;
        for e in 1..corrected.len() {
            // Strictly greater: an exact tie keeps the lower index.
            if corrected[e] > corrected[best] {
                best = e;
            }
        }
        ids.push(best as u32);
        corrected[best] = f32::NEG_INFINITY;
    }
    let mut sum = 0f32;
    for &e in &ids {
        sum += scores[e as usize];
    }
    let denom = sum + NORM_EPS;
    let weights = ids
        .iter()
        .map(|&e| (scores[e as usize] / denom) * scale)
        .collect();
    Route { ids, weights }
}

/// Route `rows` BF16 tokens `[rows][hidden]` with the router weight `[experts][hidden]`
/// (BF16) and the f32 correction bias `[experts]`.
pub fn route(
    x: &[u16],
    rows: usize,
    weight: &[u16],
    bias: &[f32],
    top_k: usize,
    scale: f32,
) -> Vec<Route> {
    let experts = bias.len();
    let hidden = x.len() / rows;
    assert_eq!(x.len(), rows * hidden);
    x.chunks_exact(hidden)
        .map(|row| {
            select_row(
                &logits_row(&bf16::widen(row), weight, experts),
                bias,
                top_k,
                scale,
            )
        })
        .collect()
}
