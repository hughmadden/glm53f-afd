//! The decoder layer's stream flow around its two sublayers, and the model's ends.
//!
//! Reference: `Glm5NextTextDecoderLayer.forward` and `Glm5NextTextModel.forward`:
//!
//! ```text
//! streams = embed(ids) repeated over 4 streams                 # [T][4][D] BF16
//! for each layer:
//!     post, comb, x = attn_hc(streams)                         # collapse
//!     x = attn(input_layernorm(x))                             # KDA or DSA
//!     streams = post * x + comb^T @ streams                    # expand and mix
//!     post, comb, x = ffn_hc(streams)
//!     x = mlp(post_attention_layernorm(x))                     # dense MLP or MoE
//!     streams = post * x + comb^T @ streams
//! hidden = norm(mean over the 4 streams)                       # HyperHead, then RMSNorm
//! ```
//!
//! [`decoder_layer`] runs this flow with the sublayers as callbacks, so the attention and
//! FFN implementations (CPU references, kernels, or golden replays) plug in unchanged.

use crate::bf16;
use crate::fp8::ActScheme;
use crate::mhc::{self, BoundaryOut, HcParams, HC_MULT};
use crate::mlp::{self, Fp8Mlp};
use crate::norm;
use crate::router::{self, Route};

/// The parameters of one decoder layer that live outside its two sublayers.
#[derive(Clone, Debug)]
pub struct LayerParams {
    pub attn_hc: HcParams,
    pub ffn_hc: HcParams,
    /// `input_layernorm.weight` (BF16).
    pub input_norm: Vec<u16>,
    /// `post_attention_layernorm.weight` (BF16).
    pub post_attn_norm: Vec<u16>,
}

/// Everything one decoder layer produced, for checking each step against goldens.
#[derive(Clone, Debug)]
pub struct LayerTrace {
    /// Attention boundary, per token.
    pub attn: Vec<BoundaryOut>,
    /// Attention sublayer output `[T][D]`.
    pub attn_out: Vec<u16>,
    /// Streams after the attention sublayer `[T][4][D]`.
    pub mid: Vec<u16>,
    /// FFN boundary, per token.
    pub ffn: Vec<BoundaryOut>,
    /// FFN sublayer output `[T][D]`.
    pub ffn_out: Vec<u16>,
    /// Streams leaving the layer `[T][4][D]`.
    pub out: Vec<u16>,
}

/// A sublayer: maps the normalized inputs of all tokens `[T][D]` to its BF16 outputs `[T][D]`.
pub type Sublayer<'a> = dyn FnMut(&[u16]) -> Vec<u16> + 'a;

fn boundaries(
    streams: &[u16],
    rows: usize,
    hc: &HcParams,
    norm_w: &[u16],
    eps: f32,
) -> Vec<BoundaryOut> {
    let per = HC_MULT * hc.hidden;
    assert_eq!(streams.len(), rows * per);
    streams
        .chunks_exact(per)
        .map(|s| mhc::boundary(s, hc, norm_w, eps))
        .collect()
}

fn expand_all(out: &[u16], residual: &[u16], bs: &[BoundaryOut], hidden: usize) -> Vec<u16> {
    let per = HC_MULT * hidden;
    let mut next = Vec::with_capacity(residual.len());
    for (t, b) in bs.iter().enumerate() {
        let h = bf16::widen(&out[t * hidden..(t + 1) * hidden]);
        let res = bf16::widen(&residual[t * per..(t + 1) * per]);
        next.extend(mhc::expand(&h, &res, &b.mix.post, &b.mix.comb, hidden));
    }
    next
}

/// One decoder layer over `rows` tokens' streams `[T][4][D]`.
pub fn decoder_layer(
    streams: &[u16],
    rows: usize,
    p: &LayerParams,
    attn: &mut Sublayer<'_>,
    ffn: &mut Sublayer<'_>,
    rms_eps: f32,
) -> LayerTrace {
    let hidden = p.attn_hc.hidden;
    let a = boundaries(streams, rows, &p.attn_hc, &p.input_norm, rms_eps);
    let a_in: Vec<u16> = a.iter().flat_map(|b| b.normed.iter().copied()).collect();
    let attn_out = attn(&a_in);
    assert_eq!(
        attn_out.len(),
        rows * hidden,
        "attention output must be [T][D]"
    );
    let mid = expand_all(&attn_out, streams, &a, hidden);

    let f = boundaries(&mid, rows, &p.ffn_hc, &p.post_attn_norm, rms_eps);
    let f_in: Vec<u16> = f.iter().flat_map(|b| b.normed.iter().copied()).collect();
    let ffn_out = ffn(&f_in);
    assert_eq!(ffn_out.len(), rows * hidden, "FFN output must be [T][D]");
    let out = expand_all(&ffn_out, &mid, &f, hidden);
    LayerTrace {
        attn: a,
        attn_out,
        mid,
        ffn: f,
        ffn_out,
        out,
    }
}

/// The streams a batch of tokens starts with: each embedding row repeated 4 times.
pub fn embed_streams(embeddings: &[u16], rows: usize) -> Vec<u16> {
    let hidden = embeddings.len() / rows;
    embeddings
        .chunks_exact(hidden)
        .flat_map(mhc::broadcast)
        .collect()
}

/// The model's final hidden states: the mean of the 4 streams, then `model.norm`.
pub fn final_hidden(streams: &[u16], rows: usize, norm_weight: &[u16], rms_eps: f32) -> Vec<u16> {
    let hidden = norm_weight.len();
    let per = HC_MULT * hidden;
    assert_eq!(streams.len(), rows * per);
    streams
        .chunks_exact(per)
        .flat_map(|s| {
            norm::rms_norm_row(
                &mhc::head_mean(&bf16::widen(s), hidden),
                norm_weight,
                rms_eps,
            )
        })
        .collect()
}

/// An MoE layer's FFN on the coordinator's side: the routes, the shared expert, and the
/// sum with the routed experts' output, which `routed` supplies (the expert ranks in the
/// engine, or [`mlp::routed_experts_eager`] in a check).
pub struct MoeParams<'a> {
    /// `mlp.gate.weight` (BF16 `[experts][D]`).
    pub router_weight: &'a [u16],
    /// `mlp.gate.e_score_correction_bias` (f32 `[experts]`).
    pub router_bias: &'a [f32],
    pub shared: &'a Fp8Mlp,
}

/// Output of [`moe_ffn`].
#[derive(Clone, Debug)]
pub struct MoeOut {
    pub routes: Vec<Route>,
    pub shared: Vec<u16>,
    pub routed: Vec<u16>,
    /// `bf16(routed + shared)`, the sublayer output.
    pub out: Vec<u16>,
}

/// The MoE FFN of `rows` normalized tokens `x` `[T][D]`.
pub fn moe_ffn(
    x: &[u16],
    rows: usize,
    p: &MoeParams<'_>,
    routed: &mut dyn FnMut(&[u16], &[Route]) -> Vec<u16>,
    scheme: ActScheme,
) -> MoeOut {
    let routes = router::route(
        x,
        rows,
        p.router_weight,
        p.router_bias,
        router::TOP_K,
        router::ROUTED_SCALE,
    );
    let shared = mlp::mlp(x, rows, p.shared, scheme).out;
    let r = routed(x, &routes);
    assert_eq!(r.len(), shared.len());
    let out = r
        .iter()
        .zip(&shared)
        .map(|(&a, &b)| bf16::from_f32(bf16::to_f32(a) + bf16::to_f32(b)))
        .collect();
    MoeOut {
        routes,
        shared,
        routed: r,
        out,
    }
}
