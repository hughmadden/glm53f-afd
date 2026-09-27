//! Manifold-constrained hyper-connections (mHC): 4 residual streams per token.
//!
//! Reference: `Glm5NextTextHyperConnection` and `Glm5NextTextHyperHead` in `transformers`
//! `models/glm5_next`. For each sublayer (attention, then FFN) of every decoder layer:
//!
//! 1. The 4 BF16 streams `[4][4096]` of a token are flattened to 16,384 f32 values and
//!    normalized by an unweighted RMSNorm (`eps = rms_norm_eps = 1e-5`).
//! 2. A linear map `fn` (`[24][16384]`, BF16 in the checkpoint, used in f32) gives 24
//!    values: `pre_w` (4), `post_w` (4) and `comb_w` (16, a 4 x 4 matrix, row-major).
//! 3. `pre = sigmoid(pre_w * scale[0] + base[0..4]) + hc_eps`;
//!    `post = 2 * sigmoid(post_w * scale[1] + base[4..8])`;
//!    `comb = softmax(comb_w * scale[2] + base[8..24], over each row) + hc_eps`, then
//!    Sinkhorn: one column normalization, then 19 rounds of (row, column), each dividing
//!    by `sum + hc_eps`.
//! 4. Collapse: the sublayer input is `bf16(sum_j pre[j] * stream[j])` (f32 products and
//!    sum).
//! 5. Expand and mix: stream `i` becomes
//!    `bf16(bf16(bf16(post[i]) * out) + bf16(sum_j bf16(comb[j][i]) * stream[j]))`, where
//!    `out` is the sublayer's BF16 output. `comb[j][i]` weighs source stream `j` into
//!    destination stream `i` (the reference multiplies by `comb` transposed).
//!
//! The final collapse after the last layer is an unweighted mean of the 4 streams
//! ([`head_mean`]); DeepSeek-V4 weights it instead.
//!
//! **Order of operations.** These functions model the kernels in `kernels/hc.cu` exactly:
//! the projection is split into slices of 128 hidden positions (all 4 streams of each), a
//! slice is reduced by one warp per projection with the lane layout of [`project_slice`],
//! and the slices are summed in order. The RMS scale is applied after the projection
//! (`r * sum(x * fn)` rather than `sum((x * r) * fn)`), as the kernels do; everything else
//! keeps the reference's rounding points. `sigmoid`, `softmax` and the Sinkhorn divisions
//! use [`crate::math`], so the CPU and the GPU agree bit for bit.

use crate::bf16;
use crate::math::{exp, sigmoid, warp_sum};
use crate::norm;

/// Residual streams (`hc_mult`).
pub const HC_MULT: usize = 4;
/// Projection outputs: `(2 + hc_mult) * hc_mult`.
pub const HC_PROJ: usize = 24;
/// Sinkhorn iterations (`hc_sinkhorn_iters`).
pub const HC_SINKHORN_ITERS: usize = 20;
/// `hc_eps`.
pub const HC_EPS: f32 = 1e-6;
/// Hidden positions per projection slice.
pub const SLICE: usize = 128;
/// Values per slice partial: the 24 projections, then the sum of squares.
pub const PARTIAL: usize = HC_PROJ + 1;

/// One boundary's parameters: `hc_{attn,ffn}_fn` (BF16 `[24][4 * hidden]`),
/// `hc_{attn,ffn}_base` (f32 `[24]`) and `hc_{attn,ffn}_scale` (f32 `[3]`).
#[derive(Clone, Debug)]
pub struct HcParams {
    pub hidden: usize,
    pub fn_: Vec<u16>,
    pub base: [f32; HC_PROJ],
    pub scale: [f32; 3],
}

impl HcParams {
    pub fn new(hidden: usize, fn_: Vec<u16>, base: &[f32], scale: &[f32]) -> Self {
        assert_eq!(hidden % SLICE, 0, "hidden must be a multiple of 128");
        assert_eq!(
            fn_.len(),
            HC_PROJ * HC_MULT * hidden,
            "fn must be [24][4 * hidden]"
        );
        HcParams {
            hidden,
            fn_,
            base: base.try_into().expect("base must have 24 values"),
            scale: scale.try_into().expect("scale must have 3 values"),
        }
    }
}

/// The weights one boundary computes for one token.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HcMix {
    /// Collapse weights.
    pub pre: [f32; HC_MULT],
    /// Expansion weights.
    pub post: [f32; HC_MULT],
    /// Mixing matrix, row-major `[source][destination]`.
    pub comb: [f32; HC_MULT * HC_MULT],
}

/// One slice's partial sums: for the 32 lanes of a warp, lane `l` takes hidden positions
/// `d = 128 * slice + 4 * l + i` (`i` in 0..4) of each stream, and accumulates
/// `fma(x[st][d], fn[p][st * hidden + d], acc)` over `st` (outer) and `i` (inner); lanes
/// reduce by butterfly. The sum of squares uses the same layout. `streams` holds the
/// token's 4 streams `[4][hidden]` as f32 (BF16 values).
pub fn project_slice(streams: &[f32], fn_: &[u16], hidden: usize, slice: usize) -> [f32; PARTIAL] {
    let mut out = [0f32; PARTIAL];
    let d0 = slice * SLICE;
    for (p, o) in out.iter_mut().enumerate() {
        let mut lanes = [0f32; 32];
        for (l, acc) in lanes.iter_mut().enumerate() {
            for st in 0..HC_MULT {
                for i in 0..4 {
                    let d = d0 + 4 * l + i;
                    let x = streams[st * hidden + d];
                    let w = if p < HC_PROJ {
                        bf16::to_f32(fn_[p * HC_MULT * hidden + st * hidden + d])
                    } else {
                        x
                    };
                    *acc = x.mul_add(w, *acc);
                }
            }
        }
        *o = warp_sum(&lanes);
    }
    out
}

/// All slice partials of one token.
pub fn project(streams: &[f32], params: &HcParams) -> Vec<[f32; PARTIAL]> {
    (0..params.hidden / SLICE)
        .map(|s| project_slice(streams, &params.fn_, params.hidden, s))
        .collect()
}

/// Sinkhorn normalization of the comb logits, in the kernels' order: a row softmax, then a
/// column normalization, then 19 rounds of (row, column). Row sums are formed as
/// `(v[4i] + v[4i+1]) + (v[4i+2] + v[4i+3])` and column sums as
/// `(v[j] + v[4+j]) + (v[8+j] + v[12+j])` (the butterfly of 16 lanes).
pub fn sinkhorn(logits: &[f32; 16]) -> [f32; 16] {
    let xor_sum = |v: &[f32; 16], a: usize, b: usize| -> [f32; 16] {
        let mut s1 = [0f32; 16];
        for l in 0..16 {
            s1[l] = v[l] + v[l ^ a];
        }
        let mut s2 = [0f32; 16];
        for l in 0..16 {
            s2[l] = s1[l] + s1[l ^ b];
        }
        s2
    };
    let mut v = *logits;
    let mut row_max = [0f32; 16];
    for l in 0..16 {
        let m1 = v[l].max(v[l ^ 1]);
        let m1p = v[l ^ 2].max(v[l ^ 3]);
        row_max[l] = m1.max(m1p);
    }
    for l in 0..16 {
        v[l] = exp(v[l] - row_max[l]);
    }
    let s = xor_sum(&v, 1, 2);
    for l in 0..16 {
        v[l] = v[l] / s[l] + HC_EPS;
    }
    let s = xor_sum(&v, 4, 8);
    for l in 0..16 {
        v[l] /= s[l] + HC_EPS;
    }
    for _ in 1..HC_SINKHORN_ITERS {
        let s = xor_sum(&v, 1, 2);
        for l in 0..16 {
            v[l] /= s[l] + HC_EPS;
        }
        let s = xor_sum(&v, 4, 8);
        for l in 0..16 {
            v[l] /= s[l] + HC_EPS;
        }
    }
    v
}

/// The boundary weights from the slice partials: slices summed in order, the RMS scale
/// `1 / sqrt(sum_sq / (4 * hidden) + rms_eps)` applied to each projection, then the
/// reference's `pre`, `post` and `comb` formulas (each `w * scale + base` rounded twice,
/// as the reference's separate multiply and add).
pub fn finish(partials: &[[f32; PARTIAL]], params: &HcParams, rms_eps: f32) -> HcMix {
    let mut tot = [0f32; PARTIAL];
    for part in partials {
        for (t, &p) in tot.iter_mut().zip(part) {
            *t += p;
        }
    }
    let r = norm::rms_scale(tot[HC_PROJ], HC_MULT * params.hidden, rms_eps);
    let proj: Vec<f32> = tot[..HC_PROJ].iter().map(|&t| t * r).collect();
    let (s, b) = (&params.scale, &params.base);
    let mut pre = [0f32; 4];
    let mut post = [0f32; 4];
    for j in 0..4 {
        pre[j] = sigmoid(proj[j] * s[0] + b[j]) + HC_EPS;
        post[j] = 2.0 * sigmoid(proj[4 + j] * s[1] + b[4 + j]);
    }
    let mut logits = [0f32; 16];
    for l in 0..16 {
        logits[l] = proj[8 + l] * s[2] + b[8 + l];
    }
    HcMix {
        pre,
        post,
        comb: sinkhorn(&logits),
    }
}

/// The boundary weights of one token (projection and finish).
pub fn mix(streams: &[f32], params: &HcParams, rms_eps: f32) -> HcMix {
    finish(&project(streams, params), params, rms_eps)
}

/// Collapse: `bf16(((pre0 * s0 + pre1 * s1) + pre2 * s2) + pre3 * s3)`, each product and
/// sum rounded to f32 (the reference's elementwise product and sum over the stream axis).
pub fn collapse(streams: &[f32], pre: &[f32; 4], hidden: usize) -> Vec<u16> {
    (0..hidden)
        .map(|d| {
            let mut acc = 0f32;
            for j in 0..HC_MULT {
                acc += pre[j] * streams[j * hidden + d];
            }
            bf16::from_f32(acc)
        })
        .collect()
}

/// The sublayer output that enters the expansion: `out`, or `bf16(out + out2)` when the
/// output arrives in two parts (routed experts plus shared expert).
pub fn block_output(out: &[u16], out2: Option<&[u16]>) -> Vec<f32> {
    match out2 {
        None => bf16::widen(out),
        Some(o2) => out
            .iter()
            .zip(o2)
            .map(|(&a, &b)| bf16::round(bf16::to_f32(a) + bf16::to_f32(b)))
            .collect(),
    }
}

/// Expand and mix one token: stream `i` becomes
/// `bf16(bf16(bf16(post[i]) * h) + bf16(sum_j bf16(comb[j][i]) * residual[j]))`, the sum
/// over `j` in order from 0 (each product is exact in f32).
pub fn expand(
    h: &[f32],
    residual: &[f32],
    post: &[f32; 4],
    comb: &[f32; 16],
    hidden: usize,
) -> Vec<u16> {
    let mut out = vec![0u16; HC_MULT * hidden];
    for i in 0..HC_MULT {
        let pb = bf16::round(post[i]);
        let cb: [f32; 4] = std::array::from_fn(|j| bf16::round(comb[4 * j + i]));
        for d in 0..hidden {
            let e = bf16::round(pb * h[d]);
            let mut m = 0f32;
            for (j, &c) in cb.iter().enumerate() {
                m = c.mul_add(residual[j * hidden + d], m);
            }
            out[i * hidden + d] = bf16::from_f32(e + bf16::round(m));
        }
    }
    out
}

/// The final collapse (`Glm5NextTextHyperHead`): the mean of the 4 streams,
/// `bf16((((s0 + s1) + s2) + s3) * 0.25)`.
pub fn head_mean(streams: &[f32], hidden: usize) -> Vec<u16> {
    (0..hidden)
        .map(|d| {
            let s = ((streams[d] + streams[hidden + d]) + streams[2 * hidden + d])
                + streams[3 * hidden + d];
            bf16::from_f32(s * 0.25)
        })
        .collect()
}

/// Everything one boundary produces for one token.
#[derive(Clone, Debug)]
pub struct BoundaryOut {
    pub mix: HcMix,
    /// The collapsed sublayer input, before its RMSNorm.
    pub collapsed: Vec<u16>,
    /// The sublayer input after its weighted RMSNorm (`input_layernorm` or
    /// `post_attention_layernorm`).
    pub normed: Vec<u16>,
}

/// One boundary for one token: weights, collapse and the sublayer's input RMSNorm.
pub fn boundary(
    streams: &[u16],
    params: &HcParams,
    norm_weight: &[u16],
    rms_eps: f32,
) -> BoundaryOut {
    let s = bf16::widen(streams);
    let mix = mix(&s, params, rms_eps);
    let collapsed = collapse(&s, &mix.pre, params.hidden);
    let normed = norm::rms_norm_row(&collapsed, norm_weight, rms_eps);
    BoundaryOut {
        mix,
        collapsed,
        normed,
    }
}

/// The 4 identical streams a token starts with: its embedding row, repeated.
pub fn broadcast(embedding: &[u16]) -> Vec<u16> {
    let mut out = Vec::with_capacity(HC_MULT * embedding.len());
    for _ in 0..HC_MULT {
        out.extend_from_slice(embedding);
    }
    out
}
