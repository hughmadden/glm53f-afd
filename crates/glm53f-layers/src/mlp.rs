//! SwiGLU with the clamp, FP8 block-128 projections, and the dense and shared-expert MLPs.
//!
//! Reference: `Glm5NextTextMLP` (dense layers 0-2 with width 12,288; one shared expert of
//! width 2,048 in every MoE layer) and `Glm5NextTextExperts._apply_gate`:
//!
//! ```text
//! gate = gate_proj(x); up = up_proj(x)             # BF16 outputs
//! gate = gate.clamp(max=10)                        # swiglu_limit; no lower bound
//! up   = up.clamp(min=-10, max=10)
//! out  = down_proj(silu(gate) * up)                # silu -> BF16, product -> BF16
//! ```
//!
//! The FP8 projections have two accumulation orders here:
//!
//! - [`fp8_linear`]: the decode kernel's order (`fp8_gemm_decode` in
//!   `kernels/fp8_gemm.cu`), exact to the bit. Lane `l` of a warp takes 16 consecutive
//!   values at `k = k_lo + 16 * l + 512 * i`, forms their dot product with `fma` from 0,
//!   multiplies it by the block scale (`sw`, or `sw * sx` for W8A8; for MXFP8 weights the row's
//!   scale of the 32-block the 16 values lie in, `glm53f_fp8_gemm_decode_mx`) with `fma` into its
//!   accumulator; lanes reduce by butterfly; K splits (see [`decode_ksplit`]) add in order.
//! - [`fp8_linear_f64`]: exact block products in f64, the baseline for the tensor-core
//!   prefill kernel, whose in-block summation order is the hardware's.

use crate::bf16;
use crate::fp8::{self, e4m3_to_f32, ActScheme, Fp8Matrix, ScaleLayout, BLOCK, MX_BLOCK};
use crate::math::{silu, warp_sum};

/// `swiglu_limit`.
pub const SWIGLU_LIMIT: f32 = 10.0;
/// Width of the dense MLPs of layers 0-2 (`intermediate_size`).
pub const DENSE_WIDTH: usize = 12288;
/// Width of the shared expert (`moe_intermediate_size * n_shared_experts`).
pub const SHARED_WIDTH: usize = 2048;
/// Layers with a dense MLP (`first_k_dense_replace`).
pub const DENSE_LAYERS: usize = 3;
/// Output rows per decode CTA (4 warps x 2 rows).
pub const DECODE_ROWS_PER_CTA: usize = 8;
/// K positions a decode warp covers per step: 32 lanes x 16 values.
pub const DECODE_WARP_STEP: usize = 512;
/// Most activation rows the decode kernel takes.
pub const DECODE_MAX_ROWS: usize = 8;

/// SwiGLU of one element from BF16 `gate` and `up` (given as f32):
/// `bf16(bf16(silu(min(gate, 10))) * clamp(up, -10, 10))`. NaN passes through the clamps,
/// as `torch.clamp` passes it.
pub fn swiglu(gate: f32, up: f32) -> u16 {
    let g = if gate > SWIGLU_LIMIT {
        SWIGLU_LIMIT
    } else {
        gate
    };
    let u = if up > SWIGLU_LIMIT {
        SWIGLU_LIMIT
    } else if up < -SWIGLU_LIMIT {
        -SWIGLU_LIMIT
    } else {
        up
    };
    let s = bf16::round(silu(g));
    bf16::from_f32(s * u)
}

/// SwiGLU of rows laid out `[rows][2 * inter]`: gate in the first `inter` columns, up in
/// the rest (the output of a stacked `[gate; up]` projection).
pub fn swiglu_rows(gate_up: &[u16], rows: usize, inter: usize) -> Vec<u16> {
    assert_eq!(gate_up.len(), rows * 2 * inter);
    let mut out = Vec::with_capacity(rows * inter);
    for r in 0..rows {
        let row = &gate_up[r * 2 * inter..(r + 1) * 2 * inter];
        for i in 0..inter {
            out.push(swiglu(bf16::to_f32(row[i]), bf16::to_f32(row[inter + i])));
        }
    }
    out
}

/// K splits the decode kernel uses for an `[n][k]` weight: the fewest splits (each a whole
/// number of 128-blocks and at least 1,024 wide) that give at least 512 CTAs of 8 output
/// rows. A function of the shape only, never of the row count, so a row's result does not
/// depend on how many rows share the launch.
pub fn decode_ksplit(n: usize, k: usize) -> usize {
    assert_eq!(k % BLOCK, 0, "K must be a multiple of 128");
    let ctas = n.div_ceil(DECODE_ROWS_PER_CTA);
    let blocks = k / BLOCK;
    let mut best = 1;
    for d in 1..=blocks {
        if blocks % d != 0 || k / d < 1024 {
            continue;
        }
        best = d;
        if ctas * d >= 512 {
            break;
        }
    }
    best
}

/// Activations prepared for an FP8 projection: the values each product uses (BF16 values,
/// or E4M3 values for W8A8) and, for W8A8, the per-row, per-128-group scales.
pub struct PreparedActs {
    pub values: Vec<f32>,
    pub scales: Option<Vec<f32>>,
    pub cols: usize,
}

/// Prepare BF16 activation rows for a projection under `scheme`.
pub fn prepare_acts(x: &[u16], rows: usize, scheme: ActScheme) -> PreparedActs {
    let cols = x.len() / rows;
    assert_eq!(x.len(), rows * cols);
    match scheme {
        ActScheme::Bf16 => PreparedActs {
            values: bf16::widen(x),
            scales: None,
            cols,
        },
        ActScheme::Fp8Dynamic128 => {
            let q = fp8::quantize_rows(&bf16::widen(x), rows, cols);
            PreparedActs {
                values: q.q.iter().map(|&b| e4m3_to_f32(b)).collect(),
                scales: Some(q.scale),
                cols,
            }
        }
    }
}

/// `x @ W^T` in the decode kernel's order, f32 `[rows][n]`. `ksplit` must divide `k / 128`.
pub fn fp8_linear_prepared(
    a: &PreparedActs,
    rows: usize,
    w: &Fp8Matrix,
    ksplit: usize,
) -> Vec<f32> {
    let (n, k) = (w.rows, w.cols);
    assert_eq!(a.cols, k, "activation width must equal the weight's K");
    assert_eq!(k % BLOCK, 0);
    assert!(
        ksplit >= 1 && (k / BLOCK) % ksplit == 0,
        "ksplit must divide K / 128"
    );
    let kc = k / ksplit;
    let groups = k / BLOCK;
    let wv: Vec<f32> = w.data.iter().map(|&b| e4m3_to_f32(b)).collect();
    let mut out = vec![0f32; rows * n];
    for m in 0..rows {
        let x = &a.values[m * k..(m + 1) * k];
        for o in 0..n {
            let wr = &wv[o * k..(o + 1) * k];
            let mut total = 0f32;
            for split in 0..ksplit {
                let (lo, hi) = (split * kc, (split + 1) * kc);
                let mut lanes = [0f32; 32];
                for (l, acc) in lanes.iter_mut().enumerate() {
                    let mut k0 = lo + 16 * l;
                    while k0 < hi {
                        let mut d = 0f32;
                        for j in 0..16 {
                            d = x[k0 + j].mul_add(wr[k0 + j], d);
                        }
                        let sw = w.scale(o, k0);
                        let s = match &a.scales {
                            None => sw,
                            Some(sx) => sw * sx[m * groups + k0 / BLOCK],
                        };
                        *acc = d.mul_add(s, *acc);
                        k0 += DECODE_WARP_STEP;
                    }
                }
                let part = warp_sum(&lanes);
                total = if split == 0 { part } else { total + part };
            }
            out[m * n + o] = total;
        }
    }
    out
}

/// `x @ W^T` for BF16 rows `x` `[rows][k]` in the decode kernel's order.
pub fn fp8_linear(
    x: &[u16],
    rows: usize,
    w: &Fp8Matrix,
    scheme: ActScheme,
    ksplit: usize,
) -> Vec<f32> {
    fp8_linear_prepared(&prepare_acts(x, rows, scheme), rows, w, ksplit)
}

/// `x @ W^T` with each weight block's dot product and the scales exact in f64:
/// `sum_blocks (sx * sw) * sum_{k in block} x_k w_k`, the blocks 128 wide (32 for MXFP8
/// weights, whose scale changes every 32 values; the activation scale every 128). Also returns,
/// per output, the same sum over absolute values (an error scale for tolerance checks).
pub fn fp8_linear_f64(
    x: &[u16],
    rows: usize,
    w: &Fp8Matrix,
    scheme: ActScheme,
) -> (Vec<f64>, Vec<f64>) {
    let a = prepare_acts(x, rows, scheme);
    let (n, k) = (w.rows, w.cols);
    let groups = k / BLOCK;
    let wb = match w.layout {
        ScaleLayout::Block128 => BLOCK,
        ScaleLayout::Mx32 => MX_BLOCK,
    };
    let mut out = vec![0f64; rows * n];
    let mut mag = vec![0f64; rows * n];
    for m in 0..rows {
        for o in 0..n {
            let (mut acc, mut abs) = (0f64, 0f64);
            for b in 0..k / wb {
                let (mut d, mut da) = (0f64, 0f64);
                for kk in b * wb..(b + 1) * wb {
                    let p = a.values[m * k + kk] as f64 * w.value(o, kk) as f64;
                    d += p;
                    da += p.abs();
                }
                let mut s = w.scale(o, b * wb) as f64;
                if let Some(sx) = &a.scales {
                    s *= sx[m * groups + b * wb / BLOCK] as f64;
                }
                acc += d * s;
                abs += da * s.abs();
            }
            out[m * n + o] = acc;
            mag[m * n + o] = abs;
        }
    }
    (out, mag)
}

/// An FP8 MLP: `gate_proj` stacked over `up_proj` (`[2 * inter][hidden]`) and `down_proj`
/// (`[hidden][inter]`), with their block scales.
#[derive(Clone, Debug)]
pub struct Fp8Mlp {
    pub inter: usize,
    pub gate_up: Fp8Matrix,
    pub down: Fp8Matrix,
}

impl Fp8Mlp {
    pub fn new(gate: &Fp8Matrix, up: &Fp8Matrix, down: Fp8Matrix) -> Self {
        assert_eq!(gate.rows, up.rows);
        assert_eq!(down.cols, gate.rows);
        assert_eq!(down.rows, gate.cols);
        Fp8Mlp {
            inter: gate.rows,
            gate_up: Fp8Matrix::stack(gate, up),
            down,
        }
    }

    pub fn hidden(&self) -> usize {
        self.down.rows
    }
}

/// Everything an MLP forward produces, for checking each stage.
#[derive(Clone, Debug)]
pub struct MlpOut {
    /// `[rows][2 * inter]` BF16 gate and up.
    pub gate_up: Vec<u16>,
    /// `[rows][inter]` BF16 SwiGLU output (the down projection's input).
    pub act: Vec<u16>,
    /// `[rows][hidden]` BF16 output.
    pub out: Vec<u16>,
}

/// MLP forward of BF16 rows `[rows][hidden]` in the decode kernels' arithmetic.
pub fn mlp(x: &[u16], rows: usize, mlp: &Fp8Mlp, scheme: ActScheme) -> MlpOut {
    let gu = fp8_linear(
        x,
        rows,
        &mlp.gate_up,
        scheme,
        decode_ksplit(mlp.gate_up.rows, mlp.gate_up.cols),
    );
    let gate_up = bf16::narrow(&gu);
    let act = swiglu_rows(&gate_up, rows, mlp.inter);
    let o = fp8_linear(
        &act,
        rows,
        &mlp.down,
        scheme,
        decode_ksplit(mlp.down.rows, mlp.down.cols),
    );
    MlpOut {
        gate_up,
        act,
        out: bf16::narrow(&o),
    }
}

/// The reference's eager combination of routed experts (`Glm5NextTextExperts.forward`):
/// for each hit expert in ascending index order, `bf16(bf16(down(act)) * w)` is added to a
/// BF16 accumulator per token, rounding after every addition. The engine computes the
/// routed experts on the expert ranks; this is the coordinator-side reference for golden
/// checks, with `expert(e)` supplying expert `e`'s weights.
pub fn routed_experts_eager<'a>(
    x: &[u16],
    rows: usize,
    routes: &[crate::router::Route],
    expert: &mut dyn FnMut(u32) -> &'a Fp8Mlp,
    scheme: ActScheme,
) -> Vec<u16> {
    let hidden = x.len() / rows;
    let mut acc = vec![0f32; rows * hidden];
    let mut hit: Vec<u32> = routes.iter().flat_map(|r| r.ids.iter().copied()).collect();
    hit.sort_unstable();
    hit.dedup();
    for e in hit {
        let w = expert(e);
        for (t, r) in routes.iter().enumerate() {
            for (pos, &id) in r.ids.iter().enumerate() {
                if id != e {
                    continue;
                }
                let y = mlp(&x[t * hidden..(t + 1) * hidden], 1, w, scheme).out;
                for d in 0..hidden {
                    let c = bf16::round(bf16::to_f32(y[d]) * r.weights[pos]);
                    let a = &mut acc[t * hidden + d];
                    *a = bf16::round(*a + c);
                }
            }
        }
    }
    bf16::narrow(&acc)
}
