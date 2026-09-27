//! Checked launches of the layer kernels (`cuda` feature): each wrapper checks that every
//! buffer is large enough for the shapes before it calls the C ABI.

use core::ffi::c_void;
use core::ptr::{null, null_mut};

use crate::cuda::{check, DeviceBuffer, Stream};
use crate::ffi;

type R = Result<(), String>;

fn need(b: &DeviceBuffer, bytes: usize, what: &str) -> R {
    if b.bytes() < bytes {
        return Err(format!(
            "{what}: buffer of {} bytes, need {bytes}",
            b.bytes()
        ));
    }
    Ok(())
}

fn i(v: usize, what: &str) -> Result<i32, String> {
    i32::try_from(v).map_err(|_| format!("{what} = {v} does not fit in i32"))
}

/// Copy embeddings `[rows][hidden]` into 4 streams each.
pub fn hc_broadcast(
    embed: &DeviceBuffer,
    streams: &DeviceBuffer,
    rows: usize,
    hidden: usize,
    s: &Stream,
) -> R {
    need(embed, rows * hidden * 2, "embed")?;
    need(streams, rows * 4 * hidden * 2, "streams")?;
    check(
        unsafe {
            ffi::glm53f_hc_broadcast(
                embed.ptr(),
                streams.mut_ptr(),
                i(rows, "rows")?,
                i(hidden, "hidden")?,
                s.0,
            )
        },
        "glm53f_hc_broadcast",
    )
}

/// The previous sublayer's expansion, fused in front of a projection.
pub struct Expand<'a> {
    pub block_out: &'a DeviceBuffer,
    pub block_out2: Option<&'a DeviceBuffer>,
    pub post: &'a DeviceBuffer,
    pub comb: &'a DeviceBuffer,
    /// Where the expanded streams go (for `hc_project`).
    pub streams_out: Option<&'a DeviceBuffer>,
}

fn check_expand(e: &Expand<'_>, rows: usize, hidden: usize) -> R {
    need(e.block_out, rows * hidden * 2, "block_out")?;
    if let Some(b) = e.block_out2 {
        need(b, rows * hidden * 2, "block_out2")?;
    }
    need(e.post, rows * 16, "post")?;
    need(e.comb, rows * 64, "comb")
}

/// Projection partials `[rows][hidden/128][25]` (and the fused expansion, if any). With
/// `project = None` only the expansion runs.
pub fn hc_project(
    streams_in: &DeviceBuffer,
    expand: Option<&Expand<'_>>,
    project: Option<(&DeviceBuffer, &DeviceBuffer)>,
    rows: usize,
    hidden: usize,
    s: &Stream,
) -> R {
    need(streams_in, rows * 4 * hidden * 2, "streams_in")?;
    let (mut h1, mut h2, mut post, mut comb, mut out) =
        (null(), null(), null(), null(), null_mut());
    if let Some(e) = expand {
        check_expand(e, rows, hidden)?;
        let so = e.streams_out.ok_or("expansion needs streams_out")?;
        need(so, rows * 4 * hidden * 2, "streams_out")?;
        h1 = e.block_out.ptr();
        h2 = e.block_out2.map_or(null(), |b| b.ptr());
        post = e.post.ptr();
        comb = e.comb.ptr();
        out = so.mut_ptr();
    }
    let (mut fn_, mut partials) = (null(), null_mut());
    if let Some((f, p)) = project {
        need(f, 24 * 4 * hidden * 2, "fn")?;
        need(p, rows * (hidden / 128) * 25 * 4, "partials")?;
        fn_ = f.ptr();
        partials = p.mut_ptr();
    }
    check(
        unsafe {
            ffi::glm53f_hc_project(
                streams_in.ptr(),
                h1,
                h2,
                post,
                comb,
                out,
                fn_,
                partials,
                i(rows, "rows")?,
                i(hidden, "hidden")?,
                s.0,
            )
        },
        "glm53f_hc_project",
    )
}

/// Outputs of `hc_finish`; every one is optional.
#[derive(Default)]
pub struct FinishOut<'a> {
    pub pre: Option<&'a DeviceBuffer>,
    pub post: Option<&'a DeviceBuffer>,
    pub comb: Option<&'a DeviceBuffer>,
    pub collapsed: Option<&'a DeviceBuffer>,
    pub normed: Option<&'a DeviceBuffer>,
    /// E4M3 codes and scales of the normed row.
    pub quant: Option<(&'a DeviceBuffer, &'a DeviceBuffer)>,
}

fn opt_mut<T>(b: Option<&DeviceBuffer>) -> *mut T {
    b.map_or(null_mut(), |b| b.mut_ptr())
}

#[allow(clippy::too_many_arguments)]
pub fn hc_finish(
    partials: &DeviceBuffer,
    base: &DeviceBuffer,
    scale: &DeviceBuffer,
    streams: &DeviceBuffer,
    norm_weight: Option<&DeviceBuffer>,
    out: &FinishOut<'_>,
    rows: usize,
    hidden: usize,
    s: &Stream,
) -> R {
    need(partials, rows * (hidden / 128) * 25 * 4, "partials")?;
    need(base, 24 * 4, "base")?;
    need(scale, 3 * 4, "scale")?;
    need(streams, rows * 4 * hidden * 2, "streams")?;
    if let Some(w) = norm_weight {
        need(w, hidden * 2, "norm_weight")?;
    }
    for (b, n, what) in [
        (out.pre, rows * 16, "pre"),
        (out.post, rows * 16, "post"),
        (out.comb, rows * 64, "comb"),
    ] {
        if let Some(b) = b {
            need(b, n, what)?;
        }
    }
    for (b, what) in [(out.collapsed, "collapsed"), (out.normed, "normed")] {
        if let Some(b) = b {
            need(b, rows * hidden * 2, what)?;
        }
    }
    if let Some((q, qs)) = out.quant {
        need(q, rows * hidden, "normed_q")?;
        need(qs, rows * (hidden / 128) * 4, "normed_scales")?;
    }
    check(
        unsafe {
            ffi::glm53f_hc_finish(
                partials.ptr(),
                base.ptr(),
                scale.ptr(),
                streams.ptr(),
                norm_weight.map_or(null(), |b| b.ptr()),
                opt_mut(out.pre),
                opt_mut(out.post),
                opt_mut(out.comb),
                opt_mut(out.collapsed),
                opt_mut(out.normed),
                opt_mut(out.quant.map(|q| q.0)),
                opt_mut(out.quant.map(|q| q.1)),
                i(rows, "rows")?,
                i(hidden, "hidden")?,
                s.0,
            )
        },
        "glm53f_hc_finish",
    )
}

/// The final mean of the streams and `model.norm`, optionally after the last expansion.
pub fn hc_head(
    streams: &DeviceBuffer,
    expand: Option<&Expand<'_>>,
    norm_weight: &DeviceBuffer,
    out: &DeviceBuffer,
    rows: usize,
    hidden: usize,
    s: &Stream,
) -> R {
    need(streams, rows * 4 * hidden * 2, "streams")?;
    need(norm_weight, hidden * 2, "norm_weight")?;
    need(out, rows * hidden * 2, "out")?;
    let (mut h1, mut h2, mut post, mut comb) = (null(), null(), null(), null());
    if let Some(e) = expand {
        check_expand(e, rows, hidden)?;
        h1 = e.block_out.ptr();
        h2 = e.block_out2.map_or(null(), |b| b.ptr());
        post = e.post.ptr();
        comb = e.comb.ptr();
    }
    check(
        unsafe {
            ffi::glm53f_hc_head(
                streams.ptr(),
                h1,
                h2,
                post,
                comb,
                norm_weight.ptr(),
                out.mut_ptr(),
                i(rows, "rows")?,
                i(hidden, "hidden")?,
                s.0,
            )
        },
        "glm53f_hc_head",
    )
}

pub fn router_logits(
    x: &DeviceBuffer,
    weight: &DeviceBuffer,
    logits: &DeviceBuffer,
    rows: usize,
    experts: usize,
    hidden: usize,
    s: &Stream,
) -> R {
    need(x, rows * hidden * 2, "x")?;
    need(weight, experts * hidden * 2, "weight")?;
    need(logits, rows * experts * 4, "logits")?;
    check(
        unsafe {
            ffi::glm53f_router_logits(
                x.ptr(),
                weight.ptr(),
                logits.mut_ptr(),
                i(rows, "rows")?,
                i(experts, "experts")?,
                i(hidden, "hidden")?,
                s.0,
            )
        },
        "glm53f_router_logits",
    )
}

#[allow(clippy::too_many_arguments)]
pub fn router_select(
    logits: &DeviceBuffer,
    bias: &DeviceBuffer,
    ids: &DeviceBuffer,
    weights: &DeviceBuffer,
    rows: usize,
    experts: usize,
    top_k: usize,
    scale: f32,
    s: &Stream,
) -> R {
    need(logits, rows * experts * 4, "logits")?;
    need(bias, experts * 4, "bias")?;
    need(ids, rows * top_k * 4, "ids")?;
    need(weights, rows * top_k * 4, "weights")?;
    check(
        unsafe {
            ffi::glm53f_router_select(
                logits.ptr(),
                bias.ptr(),
                ids.mut_ptr(),
                weights.mut_ptr(),
                i(rows, "rows")?,
                i(experts, "experts")?,
                i(top_k, "top_k")?,
                scale,
                s.0,
            )
        },
        "glm53f_router_select",
    )
}

pub fn rmsnorm(
    x: &DeviceBuffer,
    weight: &DeviceBuffer,
    out: &DeviceBuffer,
    rows: usize,
    hidden: usize,
    s: &Stream,
) -> R {
    need(x, rows * hidden * 2, "x")?;
    need(weight, hidden * 2, "weight")?;
    need(out, rows * hidden * 2, "out")?;
    check(
        unsafe {
            ffi::glm53f_rmsnorm(
                x.ptr(),
                weight.ptr(),
                out.mut_ptr(),
                i(rows, "rows")?,
                i(hidden, "hidden")?,
                s.0,
            )
        },
        "glm53f_rmsnorm",
    )
}

pub fn act_quant(
    x: &DeviceBuffer,
    q: &DeviceBuffer,
    scales: &DeviceBuffer,
    rows: usize,
    cols: usize,
    s: &Stream,
) -> R {
    need(x, rows * cols * 2, "x")?;
    need(q, rows * cols, "q")?;
    need(scales, rows * (cols / 128) * 4, "scales")?;
    check(
        unsafe {
            ffi::glm53f_act_quant(
                x.ptr(),
                q.mut_ptr(),
                scales.mut_ptr(),
                i(rows, "rows")?,
                i(cols, "cols")?,
                s.0,
            )
        },
        "glm53f_act_quant",
    )
}

pub fn swiglu(
    gate_up: &DeviceBuffer,
    act: Option<&DeviceBuffer>,
    quant: Option<(&DeviceBuffer, &DeviceBuffer)>,
    rows: usize,
    inter: usize,
    s: &Stream,
) -> R {
    need(gate_up, rows * 2 * inter * 2, "gate_up")?;
    if let Some(a) = act {
        need(a, rows * inter * 2, "act")?;
    }
    if let Some((q, qs)) = quant {
        need(q, rows * inter, "q")?;
        need(qs, rows * (inter / 128) * 4, "scales")?;
    }
    check(
        unsafe {
            ffi::glm53f_swiglu(
                gate_up.ptr(),
                opt_mut(act),
                opt_mut(quant.map(|q| q.0)),
                opt_mut(quant.map(|q| q.1)),
                i(rows, "rows")?,
                i(inter, "inter")?,
                s.0,
            )
        },
        "glm53f_swiglu",
    )
}

/// The activations of a projection.
pub enum GemmInput<'a> {
    /// BF16 `[rows][k]` (W8A16).
    Bf16(&'a DeviceBuffer),
    /// E4M3 `[rows][k]` with scales `[rows][k/128]` (W8A8).
    Fp8 {
        q: &'a DeviceBuffer,
        scales: &'a DeviceBuffer,
    },
}

/// Where a decode projection writes.
pub enum GemmOutput<'a> {
    /// BF16 `[rows][n]` (requires `ksplit == 1`).
    Bf16(&'a DeviceBuffer),
    /// f32 partials `[ksplit][rows][n]`, for `splitk_reduce`.
    Partials(&'a DeviceBuffer),
}

#[allow(clippy::too_many_arguments)]
pub fn fp8_gemm_decode(
    x: &GemmInput<'_>,
    w: &DeviceBuffer,
    w_scales: &DeviceBuffer,
    rows: usize,
    n: usize,
    k: usize,
    ksplit: usize,
    out: &GemmOutput<'_>,
    s: &Stream,
) -> R {
    need(w, n * k, "w")?;
    need(w_scales, n.div_ceil(128) * k.div_ceil(128) * 4, "w_scales")?;
    let (xp, xs, a8): (*const c_void, *const f32, i32) = match x {
        GemmInput::Bf16(b) => {
            need(b, rows * k * 2, "x")?;
            (b.ptr(), null(), 0)
        }
        GemmInput::Fp8 { q, scales } => {
            need(q, rows * k, "x")?;
            need(scales, rows * (k / 128) * 4, "x_scales")?;
            (q.ptr(), scales.ptr(), 1)
        }
    };
    let (partials, o) = match out {
        GemmOutput::Bf16(b) => {
            if ksplit != 1 {
                return Err("a BF16 output needs ksplit == 1".into());
            }
            need(b, rows * n * 2, "out")?;
            (null_mut(), b.mut_ptr())
        }
        GemmOutput::Partials(p) => {
            need(p, ksplit * rows * n * 4, "partials")?;
            (p.mut_ptr(), null_mut())
        }
    };
    check(
        unsafe {
            ffi::glm53f_fp8_gemm_decode(
                xp,
                xs,
                a8,
                w.ptr(),
                w_scales.ptr(),
                i(rows, "rows")?,
                i(n, "n")?,
                i(k, "k")?,
                i(ksplit, "ksplit")?,
                partials,
                o,
                s.0,
            )
        },
        "glm53f_fp8_gemm_decode",
    )
}

pub fn splitk_reduce(
    partials: &DeviceBuffer,
    out: &DeviceBuffer,
    ksplit: usize,
    rows: usize,
    n: usize,
    s: &Stream,
) -> R {
    need(partials, ksplit * rows * n * 4, "partials")?;
    need(out, rows * n * 2, "out")?;
    check(
        unsafe {
            ffi::glm53f_splitk_reduce(
                partials.ptr(),
                out.mut_ptr(),
                i(ksplit, "ksplit")?,
                i(rows, "rows")?,
                i(n, "n")?,
                s.0,
            )
        },
        "glm53f_splitk_reduce",
    )
}

/// A projection in the decode kernels' order, reducing the K splits when there are several.
#[allow(clippy::too_many_arguments)]
pub fn fp8_linear_decode(
    x: &GemmInput<'_>,
    w: &DeviceBuffer,
    w_scales: &DeviceBuffer,
    rows: usize,
    n: usize,
    k: usize,
    partials: &DeviceBuffer,
    out: &DeviceBuffer,
    s: &Stream,
) -> R {
    let ksplit = crate::mlp::decode_ksplit(n, k);
    if ksplit == 1 {
        fp8_gemm_decode(x, w, w_scales, rows, n, k, 1, &GemmOutput::Bf16(out), s)
    } else {
        fp8_gemm_decode(
            x,
            w,
            w_scales,
            rows,
            n,
            k,
            ksplit,
            &GemmOutput::Partials(partials),
            s,
        )?;
        splitk_reduce(partials, out, ksplit, rows, n, s)
    }
}

/// How the prefill GEMM accumulates each 128-wide K block.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Promotion {
    /// The whole block in the tensor core, as the reference's block-FP8 kernels do.
    Block128,
    /// Every k32 product sum added to the block sum in f32 (more accurate, slower).
    K32,
}

#[allow(clippy::too_many_arguments)]
pub fn fp8_gemm_prefill(
    xq: &DeviceBuffer,
    x_scales: &DeviceBuffer,
    w: &DeviceBuffer,
    w_scales: &DeviceBuffer,
    rows: usize,
    n: usize,
    k: usize,
    promotion: Promotion,
    out: &DeviceBuffer,
    out_f32: Option<&DeviceBuffer>,
    s: &Stream,
) -> R {
    need(xq, rows * k, "xq")?;
    need(x_scales, rows * (k / 128) * 4, "x_scales")?;
    need(w, n * k, "w")?;
    need(w_scales, n.div_ceil(128) * k.div_ceil(128) * 4, "w_scales")?;
    need(out, rows * n * 2, "out")?;
    if let Some(o) = out_f32 {
        need(o, rows * n * 4, "out_f32")?;
    }
    check(
        unsafe {
            ffi::glm53f_fp8_gemm_prefill(
                xq.ptr(),
                x_scales.ptr(),
                w.ptr(),
                w_scales.ptr(),
                i(rows, "rows")?,
                i(n, "n")?,
                i(k, "k")?,
                match promotion {
                    Promotion::Block128 => 0,
                    Promotion::K32 => ffi::PREFILL_PROMOTE_K32,
                },
                out.mut_ptr(),
                opt_mut(out_f32),
                s.0,
            )
        },
        "glm53f_fp8_gemm_prefill",
    )
}
