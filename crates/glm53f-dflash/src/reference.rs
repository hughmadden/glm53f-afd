//! The DFlash2 drafter in f32 on the CPU, following the reference's semantics (z-lab/dflash
//! `dflash/model.py` @ `07ebd93`, the modules `DFlashDraftModel`, `Qwen3DFlashDecoderLayer`,
//! `Qwen3DFlashAttention`, `GroupedDynamicCausalConv`, `CandidateSelector`, driven as
//! `dflash_generate` drives them). Line numbers below are that file's.
//!
//! - **Context** ([`Reference::append`]): each committed row's taps `[taps * hidden]` become
//!   `feat = hidden_norm(fc(taps))` (line 584), then per layer `k = rope(k_norm(k_proj(feat)))`
//!   and `v = v_proj(feat)` (lines 384-393), at the row's position. Only committed rows enter the
//!   context: `dflash_generate` crops the draft cache back to the committed length after every
//!   draft (line 250).
//! - **Block** ([`Reference::block`]): the anchor's embedding and `block - 1` copies of the mask
//!   token's (line 238), at positions `P..P + block` where `P` is the committed length. Every
//!   layer (lines 433-475): `x = input_layernorm(h)`; `attention_conv.prepare(x)` convolves `x`
//!   and returns the finish kernel; attention over context and block (queries, keys and values
//!   of the block from the convolved `x`; keys from the context cache); `o_proj`;
//!   `attention_conv.finish`; the residual; the same around the MLP with `mlp_conv`. Then the
//!   final `norm` (line 597).
//! - **Attention**: non-causal inside the block, sliding window over the context: the query at
//!   position `p` sees keys at `q` with `|p - q| < window` (`_attention_mask`, lines 157-171).
//! - **Dynamic convolution** (lines 478-512): `out[l][c] = sum_o (base[side][o][c] +
//!   dyn[l][side][o][c / group]) * x[l - o][c]` over the block's rows (row `l - o < 0` is zero:
//!   the block's first row sees no predecessor), with `dyn = kernel_projection(prepare input)`
//!   `[side 2][tap][group]`.
//! - **Draft**: rows `1..block` of the final hidden (line 249) through the target LM head, top-k
//!   and the selector ([`crate::selector`]).
//!
//! A request's context is a ring of `window + block` rows per layer ([`Context`]), the same
//! layout the GPU forward uses, so the two can be compared row for row.

use std::collections::BTreeMap;

use crate::cpu::{self, matmul, rmsnorm};
use crate::selector::{self, Pick, Walk};
use crate::weights::Weights;
use crate::{bf16, Dims};

/// One request's drafter context: the keys and values of its committed rows, per layer, in a ring
/// of [`Dims::ring`] rows (row of position `p` at `p % ring`).
#[derive(Clone, Debug)]
pub struct Context {
    pub dims: Dims,
    len: usize,
    lo: usize,
    /// `[layers][ring][kv_width]`, after `k_norm` and RoPE.
    k: Vec<f32>,
    /// `[layers][ring][kv_width]`.
    v: Vec<f32>,
}

impl Context {
    pub fn new(dims: Dims) -> Context {
        let n = dims.layers * dims.ring() * dims.kv_width();
        Context {
            dims,
            len: 0,
            lo: 0,
            k: vec![0.0; n],
            v: vec![0.0; n],
        }
    }

    /// Committed rows: the next block's anchor sits at this position.
    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// The lowest position whose row is still in the ring (0 unless a rewind went further back
    /// than the ring can serve).
    pub fn lo(&self) -> usize {
        self.lo
    }

    /// Drop the rows at and after `len` (a return to an earlier committed length). The ring keeps
    /// positions `len - window..` only if nothing overwrote them: rows the ring no longer holds
    /// are masked out of later drafts (`lo`) instead of read stale.
    pub fn rewind(&mut self, len: usize) {
        assert!(len <= self.len, "rewind forward: {len} > {}", self.len);
        // A draft at the old length wrote its block rows into the slots of positions
        // len_old - ring .. len_old - ring + block; every append overwrote position p - ring.
        // Positions >= len_old - window are intact.
        self.lo = self
            .lo
            .max(self.len.saturating_sub(self.dims.window))
            .min(len);
        self.len = len;
    }

    fn slot(&self, layer: usize, pos: usize) -> usize {
        (layer * self.dims.ring() + pos % self.dims.ring()) * self.dims.kv_width()
    }

    /// The stored key and value rows of `layer` at `pos` (`[kv_width]` each).
    pub fn row(&self, layer: usize, pos: usize) -> (&[f32], &[f32]) {
        let (s, w) = (self.slot(layer, pos), self.dims.kv_width());
        (&self.k[s..s + w], &self.v[s..s + w])
    }
}

/// An LM head the caller supplies (the drafter has none of its own).
pub trait LmHead: Sync {
    /// Rows of the head (logits per row).
    fn vocab(&self) -> usize;
    /// `[rows][vocab]` logits of `hidden` (`[rows][hidden]`).
    fn logits(&self, hidden: &[f32], rows: usize) -> Vec<f32>;
}

/// A dense BF16 head `[vocab][hidden]` (the target's `lm_head.weight`).
pub struct Bf16Head<'a> {
    pub weight: &'a [u16],
    pub hidden: usize,
}

impl LmHead for Bf16Head<'_> {
    fn vocab(&self) -> usize {
        self.weight.len() / self.hidden
    }
    fn logits(&self, hidden: &[f32], rows: usize) -> Vec<f32> {
        matmul(hidden, rows, self.hidden, self.weight, self.vocab())
    }
}

/// Options of one draft.
#[derive(Clone, Copy, Debug)]
pub struct DraftOptions<'a> {
    /// Candidates come from logits `[..vocab_limit]`: [`crate::SAMPLE_VOCAB`] keeps the padding
    /// rows out; the head's full width is the reference's behaviour.
    pub vocab_limit: usize,
    pub pick: Pick<'a>,
}

/// A draft and what it was chosen from.
#[derive(Clone, Debug, Default)]
pub struct Draft {
    /// `norm(h)` for the block's rows `[block][hidden]`.
    pub hidden: Vec<f32>,
    /// Logits of rows `1..block` `[drafts][vocab]`.
    pub logits: Vec<f32>,
    /// Top-k per draft position, descending `[drafts][k]`.
    pub unary: Vec<f32>,
    pub candidates: Vec<u32>,
    /// `hidden_projection` of rows `1..block` `[drafts][rank]`.
    pub hproj: Vec<f32>,
    /// The path.
    pub walk: Walk,
}

/// Named intermediates of a block forward, keyed as the goldens name them (`L{i}.attn_norm`, ...).
pub type Trace = BTreeMap<String, Vec<f32>>;

/// The drafter on the CPU, borrowing its weights.
pub struct Reference<'w> {
    pub w: &'w Weights,
    pub inv_freq: Vec<f32>,
    /// Keys a query sees behind it: `window - 1` (the reference's `|p - q| < window`). A test knob.
    pub window_left: usize,
    /// `false`: f32 throughout (the reference run in FP32, the goldens' primary contract).
    /// `true`: the GPU forward's numerics: every GEMM input and the ring's keys and values rounded
    /// to BF16, everything else f32 (see `README.md`, "Numerics").
    pub bf16_io: bool,
}

impl<'w> Reference<'w> {
    pub fn new(w: &'w Weights) -> Reference<'w> {
        let d = w.dims;
        Reference {
            w,
            inv_freq: cpu::inv_freq(d.rope_theta, d.head_dim),
            window_left: d.window - 1,
            bf16_io: false,
        }
    }

    /// A GEMM input: rounded to BF16 in [`Reference::bf16_io`] mode.
    fn gin(&self, x: &[f32]) -> Vec<f32> {
        if self.bf16_io {
            x.iter().map(|&v| bf16::round(v)).collect()
        } else {
            x.to_vec()
        }
    }

    /// A value the ring stores: rounded to BF16 in [`Reference::bf16_io`] mode.
    fn stored(&self, x: &mut [f32]) {
        if self.bf16_io {
            for v in x {
                *v = bf16::round(*v);
            }
        }
    }

    fn dims(&self) -> Dims {
        self.w.dims
    }

    /// `hidden_norm(fc(taps))` for `taps` `[rows][taps * hidden]`.
    pub fn context_features(&self, taps: &[f32]) -> Vec<f32> {
        let d = self.dims();
        let rows = taps.len() / d.tap_width();
        let y = matmul(taps, rows, d.tap_width(), &self.w.fc, d.hidden);
        rmsnorm(&y, &self.w.hidden_norm, d.eps)
    }

    /// `k_norm` per head, then RoPE at `pos`, on one row of keys or queries.
    fn norm_rope(&self, x: &mut [f32], norm: &[u16], pos: usize) {
        let hd = self.dims().head_dim;
        for head in x.chunks_exact_mut(hd) {
            let n = rmsnorm(head, norm, self.dims().eps);
            head.copy_from_slice(&n);
            cpu::rope(head, pos, &self.inv_freq);
        }
    }

    /// Append rows of taps `[rows][taps * hidden]` at positions `ctx.len()..`.
    pub fn append(&self, ctx: &mut Context, taps: &[f32]) {
        let feats = self.context_features(taps);
        self.append_features(ctx, &feats);
    }

    /// Append rows of context features (`hidden_norm(fc(taps))`, `[rows][hidden]`).
    pub fn append_features(&self, ctx: &mut Context, feats: &[f32]) {
        let d = self.dims();
        let rows = feats.len() / d.hidden;
        let kvw = d.kv_width();
        let feats = self.gin(feats);
        for (l, lw) in self.w.layers.iter().enumerate() {
            let mut k = matmul(&feats, rows, d.hidden, &lw.k, kvw);
            let mut v = matmul(&feats, rows, d.hidden, &lw.v, kvw);
            self.stored(&mut v);
            for r in 0..rows {
                let pos = ctx.len + r;
                self.norm_rope(&mut k[r * kvw..(r + 1) * kvw], &lw.k_norm, pos);
                self.stored(&mut k[r * kvw..(r + 1) * kvw]);
                let s = ctx.slot(l, pos);
                ctx.k[s..s + kvw].copy_from_slice(&k[r * kvw..(r + 1) * kvw]);
                ctx.v[s..s + kvw].copy_from_slice(&v[r * kvw..(r + 1) * kvw]);
            }
        }
        ctx.len += rows;
    }

    /// The grouped dynamic convolution over one block (`x` `[block][hidden]`, `dynk` the
    /// kernel projection `[block][dyn_width]`, `base` `[2][taps][hidden]`): `side` 0 is
    /// `prepare`, 1 `finish`.
    pub fn conv(&self, x: &[f32], dynk: &[f32], base: &[u16], side: usize) -> Vec<f32> {
        let d = self.dims();
        let (h, taps, groups, gs) = (d.hidden, d.conv_taps, d.groups(), d.group_size);
        let dw = d.dyn_width();
        let mut out = vec![0f32; x.len()];
        for l in 0..d.block {
            for o in 0..taps.min(l + 1) {
                let src = &x[(l - o) * h..(l - o + 1) * h];
                let b = &base[(side * taps + o) * h..(side * taps + o + 1) * h];
                let dy = &dynk
                    [l * dw + (side * taps + o) * groups..l * dw + (side * taps + o + 1) * groups];
                for c in 0..h {
                    let acc = out[l * h + c] + bf16::to_f32(b[c]) * src[c];
                    out[l * h + c] = acc + dy[c / gs] * src[c];
                }
            }
        }
        out
    }

    /// Attention of the block's queries `[block][q_width]` over the context and the block's own
    /// keys and values `[block][kv_width]`. Returns `[block][q_width]`.
    fn attention(
        &self,
        ctx: &Context,
        layer: usize,
        q: &[f32],
        kb: &[f32],
        vb: &[f32],
    ) -> Vec<f32> {
        let d = self.dims();
        let (hd, kvw, p0) = (d.head_dim, d.kv_width(), ctx.len);
        let scale = 1.0 / (hd as f32).sqrt();
        let mut out = vec![0f32; d.block * d.q_width()];
        for j in 0..d.block {
            let qp = p0 + j;
            let first = qp.saturating_sub(self.window_left).max(ctx.lo);
            for h in 0..d.heads {
                let g = h / d.group();
                let qh = &q[j * d.q_width() + h * hd..j * d.q_width() + (h + 1) * hd];
                let mut keys: Vec<(&[f32], &[f32])> = (first..p0)
                    .map(|p| {
                        let (k, v) = ctx.row(layer, p);
                        (&k[g * hd..(g + 1) * hd], &v[g * hd..(g + 1) * hd])
                    })
                    .collect();
                for b in 0..d.block {
                    if b.abs_diff(j) <= self.window_left {
                        keys.push((
                            &kb[b * kvw + g * hd..b * kvw + (g + 1) * hd],
                            &vb[b * kvw + g * hd..b * kvw + (g + 1) * hd],
                        ));
                    }
                }
                let s: Vec<f32> = keys.iter().map(|(k, _)| cpu::dot(qh, k) * scale).collect();
                let m = s.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let e: Vec<f32> = s.iter().map(|&x| (x - m).exp()).collect();
                let z: f32 = e.iter().sum();
                let o = &mut out[j * d.q_width() + h * hd..j * d.q_width() + (h + 1) * hd];
                for ((_, v), &p) in keys.iter().zip(&e) {
                    let w = p / z;
                    for (oi, &vi) in o.iter_mut().zip(v.iter()) {
                        *oi += w * vi;
                    }
                }
            }
        }
        out
    }

    /// The block forward: `embeds` `[block][hidden]` (anchor, then mask rows) at positions
    /// `ctx.len()..`. Returns `norm(h)` `[block][hidden]`. The context is not changed.
    pub fn block(&self, ctx: &Context, embeds: &[f32], mut trace: Option<&mut Trace>) -> Vec<f32> {
        let d = self.dims();
        let (h, n) = (d.hidden, d.block);
        assert_eq!(embeds.len(), n * h);
        let mut put = |k: String, v: &[f32]| {
            if let Some(t) = trace.as_deref_mut() {
                t.insert(k, v.to_vec());
            }
        };
        let mut hid = embeds.to_vec();
        for (l, lw) in self.w.layers.iter().enumerate() {
            put(format!("L{l}.in"), &hid);
            // Attention site.
            let x = rmsnorm(&hid, &lw.input_ln, d.eps);
            let dynk = matmul(&self.gin(&x), n, h, &lw.attn_kp, d.dyn_width());
            let xc = self.gin(&self.conv(&x, &dynk, &lw.attn_base, 0));
            let mut q = matmul(&xc, n, h, &lw.q, d.q_width());
            let mut k = matmul(&xc, n, h, &lw.k, d.kv_width());
            let mut v = matmul(&xc, n, h, &lw.v, d.kv_width());
            for j in 0..n {
                let pos = ctx.len + j;
                self.norm_rope(
                    &mut q[j * d.q_width()..(j + 1) * d.q_width()],
                    &lw.q_norm,
                    pos,
                );
                self.norm_rope(
                    &mut k[j * d.kv_width()..(j + 1) * d.kv_width()],
                    &lw.k_norm,
                    pos,
                );
            }
            self.stored(&mut k);
            self.stored(&mut v);
            let att = self.gin(&self.attention(ctx, l, &q, &k, &v));
            let a = matmul(&att, n, d.q_width(), &lw.o, h);
            let ac = self.conv(&a, &dynk, &lw.attn_base, 1);
            for (x, y) in hid.iter_mut().zip(&ac) {
                *x += y;
            }
            put(format!("L{l}.mid"), &hid);
            put(format!("L{l}.attn_norm"), &x);
            put(format!("L{l}.attn_dyn"), &dynk);
            put(format!("L{l}.attn_conv_in"), &xc);
            put(format!("L{l}.k_block"), &k);
            put(format!("L{l}.v_block"), &v);
            put(format!("L{l}.attn_raw"), &a);
            put(format!("L{l}.attn_conv_out"), &ac);
            // MLP site.
            let x = rmsnorm(&hid, &lw.post_ln, d.eps);
            let dynk = matmul(&self.gin(&x), n, h, &lw.mlp_kp, d.dyn_width());
            let xc = self.gin(&self.conv(&x, &dynk, &lw.mlp_base, 0));
            let g = matmul(&xc, n, h, &lw.gate, d.inter);
            let u = matmul(&xc, n, h, &lw.up, d.inter);
            let act: Vec<f32> = g.iter().zip(&u).map(|(&g, &u)| cpu::silu(g) * u).collect();
            let m = matmul(&self.gin(&act), n, d.inter, &lw.down, h);
            let mc = self.conv(&m, &dynk, &lw.mlp_base, 1);
            for (x, y) in hid.iter_mut().zip(&mc) {
                *x += y;
            }
            put(format!("L{l}.mlp_norm"), &x);
            put(format!("L{l}.mlp_dyn"), &dynk);
            put(format!("L{l}.mlp_conv_in"), &xc);
            put(format!("L{l}.mlp_raw"), &m);
            put(format!("L{l}.mlp_conv_out"), &mc);
            put(format!("L{l}.out"), &hid);
        }
        let fin = rmsnorm(&hid, &self.w.norm, d.eps);
        put("final".to_string(), &fin);
        fin
    }

    /// Draft `block - 1` tokens after `anchor` (at position `ctx.len()`). `anchor_embed` and
    /// `mask_embed` are the target's embedding rows `[hidden]`.
    #[allow(clippy::too_many_arguments)]
    pub fn draft(
        &self,
        ctx: &Context,
        anchor: u32,
        anchor_embed: &[f32],
        mask_embed: &[f32],
        head: &dyn LmHead,
        opts: &DraftOptions<'_>,
        trace: Option<&mut Trace>,
    ) -> Draft {
        let d = self.dims();
        let mut embeds = anchor_embed.to_vec();
        for _ in 1..d.block {
            embeds.extend_from_slice(mask_embed);
        }
        let hidden = self.block(ctx, &embeds, trace);
        let logits = head.logits(&self.gin(&hidden[d.hidden..]), d.drafts());
        self.select(anchor, hidden, logits, opts)
    }

    /// Top-k and the selector's walk from the final hidden `[block][hidden]` and the logits of
    /// rows `1..block`.
    pub fn select(
        &self,
        anchor: u32,
        hidden: Vec<f32>,
        logits: Vec<f32>,
        opts: &DraftOptions<'_>,
    ) -> Draft {
        let d = self.dims();
        let vocab = logits.len() / d.drafts();
        let (mut unary, mut candidates) = (Vec::new(), Vec::new());
        for row in logits.chunks_exact(vocab) {
            let (v, i) = selector::top_k(row, d.top_k, opts.vocab_limit);
            unary.extend(v);
            candidates.extend(i);
        }
        let hproj = matmul(
            &self.gin(&hidden[d.hidden..]),
            d.drafts(),
            d.hidden,
            &self.w.hproj,
            d.rank,
        );
        let walk = selector::walk(
            &unary,
            &candidates,
            &hproj,
            anchor,
            &self.w.pred,
            &self.w.succ,
            d.rank,
            d.top_k,
            opts.pick,
        );
        Draft {
            hidden,
            logits,
            unary,
            candidates,
            hproj,
            walk,
        }
    }
}
