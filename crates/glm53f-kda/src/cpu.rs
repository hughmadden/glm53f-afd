//! f32 CPU reference of one KDA layer on the recurrent path (decode and verify).
//!
//! # Semantics
//!
//! For each row (token) and head, following the reference `Glm5NextTextLinearAttention` with a
//! cache (`causal_conv1d_update` + `recurrent_kimi_delta_attention`):
//!
//! 1. **Conv.** `x_c = SiLU(sum_tap w[c][tap] * u[c][t - 3 + tap])` over the q | k | v projection
//!    rows `u`, where the three rows before the window come from the conv window. Rounded to
//!    bfloat16 (see [`Rounding`]).
//! 2. **Gates.** Per key channel `g = lower * sigmoid(exp(A_log[h]) * (a + dt_bias))` (log
//!    space, `lower = -5`, so `g` lies in (-5, 0)); the state decays by `exp(g)`.
//!    `beta = bf16(sigmoid(b))`.
//! 3. **Norms.** `q <- q / sqrt(|q|^2 + 1e-6) * 128^-0.5`, `k <- k / sqrt(|k|^2 + 1e-6)`, f32.
//! 4. **Delta rule** on the f32 state `S` (key × value):
//!    `S <- S * exp(g)` (per key channel); `d = (v - S^T k) * beta`; `S <- S + k d^T`;
//!    `y = S^T q`, rounded to bfloat16.
//! 5. **Gated RMSNorm.** `o = bf16(w * (y / sqrt(mean(y^2) + eps)) * sigmoid(gate))`, the input
//!    of `o_proj`.
//!
//! # Order of operations
//!
//! [`chain`] evaluates these exactly as the kernels do: sums over 128 elements are 32 partial
//! sums of 4 consecutive elements followed by a butterfly ([`warp_sum`]); the norms multiply by
//! a reciprocal square root; no multiply-add is fused. With the transcendental functions as the
//! only exception (`exp` differs between the device and the host C library by an ulp or two),
//! it is a bit-level model of the kernels. [`update`] (the state update) involves no
//! transcendental function, so [`replay`] reproduces the device's replay bit for bit.
//!
//! [`literal`] evaluates the same semantics the way the reference writes them (state stored
//! key-major, sums in index order, the L2 norm as a division); it cross-checks [`chain`]
//! independently of its operation order.
//!
//! # Layouts
//!
//! - state: `[H][DV][DK]` (value row major), as the kernels store it; the reference stores
//!   `[H][DK][DV]` ([`transpose_state`] converts either way);
//! - conv window: `[WINDOW][C]`, oldest row first, `C = 3 * H * 128` (q | k | v);
//! - conv weight: `[C][TAPS]`, tap `TAPS - 1` multiplies the current row.
//!
//! # Relation to the chunked form
//!
//! For prefill the reference uses `chunk_kimi_delta_attention` (see [`crate::chunked`]): the
//! same recurrence regrouped into 64-row chunks. In exact arithmetic the two are identical; in
//! f32 they differ in rounding only (the chunk form sums in a different order and applies the
//! decays as cumulative products). The per-row recurrence here is the contract for decode and
//! verify; the chunked form is checked against it within a tolerance.

// Sums are written `acc = acc + x` to mirror the kernels' source line for line (the same as `+=`;
// Rust never contracts a multiply and an add into one fused operation).
#![allow(clippy::assign_op_pattern)]

use crate::bf16;
use crate::{channels, state_len, DK, DV, L2_EPS, TAPS, WINDOW};

/// Where an evaluation rounds to bfloat16.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Rounding {
    /// The model's bfloat16 activations, as the kernels evaluate them: conv and SiLU in f32 with
    /// one rounding (as the fused `causal_conv1d` kernel the reference uses when it is
    /// installed), beta, the read-out and the output each rounded once.
    #[default]
    Fused,
    /// As `Fused`, but the conv output is also rounded before the SiLU: the reference's
    /// pure-torch path (`F.conv1d` in bfloat16, then the activation).
    Unfused,
    /// No rounding anywhere: the reference evaluated in f32 throughout, as the oracle's primary
    /// goldens are. Not what the kernels compute; it isolates the formulas from the rounding.
    F32,
}

impl Rounding {
    /// `x` rounded to bfloat16, unless this is [`Rounding::F32`].
    pub fn round(self, x: f32) -> f32 {
        match self {
            Rounding::F32 => x,
            _ => bf16::round(x),
        }
    }
}

/// One KDA layer's parameters. bfloat16 tensors are widened to f32 exactly.
#[derive(Clone, Debug)]
pub struct LayerParams {
    pub heads: usize,
    /// `[C][TAPS]`: `q_conv1d | k_conv1d | v_conv1d` weights (bfloat16 values).
    pub conv_w: Vec<f32>,
    /// `[H]`, f32.
    pub a_log: Vec<f32>,
    /// `[H * DK]`, f32.
    pub dt_bias: Vec<f32>,
    /// `[DV]`: `o_norm.weight` (bfloat16 values).
    pub norm_w: Vec<f32>,
    /// `rms_norm_eps`.
    pub eps: f32,
    /// `gate_lower_bound`.
    pub lower: f32,
}

impl LayerParams {
    pub fn channels(&self) -> usize {
        channels(self.heads)
    }

    fn check(&self) {
        assert!(self.heads >= 1);
        assert_eq!(
            self.conv_w.len(),
            self.channels() * TAPS,
            "conv_w must be [C][TAPS]"
        );
        assert_eq!(self.a_log.len(), self.heads, "a_log must be [H]");
        assert_eq!(
            self.dt_bias.len(),
            self.heads * DK,
            "dt_bias must be [H * DK]"
        );
        assert_eq!(self.norm_w.len(), DV, "norm_w must be [DV]");
    }
}

/// Projection outputs for `rows` consecutive rows (bfloat16 values widened to f32).
#[derive(Clone, Debug, PartialEq)]
pub struct Rows {
    pub heads: usize,
    pub rows: usize,
    /// `[rows][C]`: q | k | v projections, the conv input.
    pub qkv: Vec<f32>,
    /// `[rows][H * DK]`: `f_b_proj(f_a_proj(x))`, before `dt_bias`.
    pub a: Vec<f32>,
    /// `[rows][H]`: `b_proj(x)`, before the sigmoid.
    pub b: Vec<f32>,
    /// `[rows][H * DV]`: `g_b_proj(g_a_proj(x))`.
    pub gate: Vec<f32>,
}

impl Rows {
    /// Rows `start .. start + n`.
    pub fn slice(&self, start: usize, n: usize) -> Rows {
        assert!(start + n <= self.rows);
        let (h, c) = (self.heads, channels(self.heads));
        Rows {
            heads: h,
            rows: n,
            qkv: self.qkv[start * c..(start + n) * c].to_vec(),
            a: self.a[start * h * DK..(start + n) * h * DK].to_vec(),
            b: self.b[start * h..(start + n) * h].to_vec(),
            gate: self.gate[start * h * DV..(start + n) * h * DV].to_vec(),
        }
    }

    fn check(&self) {
        let (h, r) = (self.heads, self.rows);
        assert_eq!(self.qkv.len(), r * channels(h), "qkv must be [rows][C]");
        assert_eq!(self.a.len(), r * h * DK, "a must be [rows][H * DK]");
        assert_eq!(self.b.len(), r * h, "b must be [rows][H]");
        assert_eq!(self.gate.len(), r * h * DV, "gate must be [rows][H * DV]");
    }
}

/// What a replay needs per row (the kernels' saves): L2-normalized k, v (bfloat16 values), the
/// decay multipliers `exp(g)` and beta.
#[derive(Clone, Debug, PartialEq)]
pub struct Saves {
    pub heads: usize,
    pub rows: usize,
    /// `[rows][H][DK]`.
    pub k: Vec<f32>,
    /// `[rows][H][DV]`.
    pub v: Vec<f32>,
    /// `[rows][H][DK]`.
    pub g: Vec<f32>,
    /// `[rows][H]`.
    pub beta: Vec<f32>,
}

impl Saves {
    pub fn zeros(heads: usize, rows: usize) -> Self {
        Saves {
            heads,
            rows,
            k: vec![0.0; rows * heads * DK],
            v: vec![0.0; rows * heads * DV],
            g: vec![0.0; rows * heads * DK],
            beta: vec![0.0; rows * heads],
        }
    }
}

/// A chain's results.
#[derive(Clone, Debug)]
pub struct ChainOut {
    /// `[rows][H * DV]`: the gated RMSNorm output (bfloat16 values).
    pub out: Vec<f32>,
    /// `[rows][H * DV]`: the delta rule's read-out before the norm (bfloat16 values).
    pub y: Vec<f32>,
    /// `[H][DV][DK]`: the state after the last row.
    pub state: Vec<f32>,
    pub saves: Saves,
}

/// `1 / (1 + exp(-x))`.
pub fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// `x / (1 + exp(-x))`.
pub fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

/// The sum of 32 lanes as a warp's `__shfl_xor_sync` butterfly adds them (offsets 16, 8, 4, 2,
/// 1). Every lane ends with the same value.
pub fn warp_sum(lanes: &[f32; 32]) -> f32 {
    let mut x = *lanes;
    let mut o = 16;
    while o > 0 {
        let prev = x;
        for (l, xl) in x.iter_mut().enumerate() {
            *xl = prev[l] + prev[l ^ o];
        }
        o >>= 1;
    }
    x[0]
}

/// `sum_i a[i] * b[i]` over 128 elements in the kernels' order: lane `l` sums elements
/// `4l .. 4l + 3` in order from zero, then [`warp_sum`].
pub fn dot128(a: &[f32], b: &[f32]) -> f32 {
    let mut lanes = [0.0f32; 32];
    for (l, lane) in lanes.iter_mut().enumerate() {
        let mut acc = 0.0f32;
        for i in 0..4 {
            acc = acc + a[4 * l + i] * b[4 * l + i];
        }
        *lane = acc;
    }
    warp_sum(&lanes)
}

/// The log-space decay of the reference's forget gate, `lower * sigmoid(exp(a_log) * (a + dt_bias))`.
pub fn log_decay(a: f32, dt_bias: f32, a_log: f32, lower: f32) -> f32 {
    lower * sigmoid(a_log.exp() * (a + dt_bias))
}

/// The decay multiplier `exp(g)` as the kernels compute it (`decay_rate = exp(a_log)`).
pub fn decay(a: f32, dt_bias: f32, decay_rate: f32, lower: f32) -> f32 {
    (lower * sigmoid(decay_rate * (a + dt_bias))).exp()
}

/// `bf16(sigmoid(b))`.
pub fn beta(b: f32) -> f32 {
    bf16::round(sigmoid(b))
}

/// One conv channel's output for row `r` of a window: `[conv window; rows]` over 4 taps.
fn conv_channel(
    p: &LayerParams,
    conv: &[f32],
    rows: &Rows,
    r: usize,
    c: usize,
    mode: Rounding,
) -> f32 {
    let cn = p.channels();
    let mut acc = 0.0f32;
    for tap in 0..TAPS {
        let at = r + tap;
        let x = if at < WINDOW {
            conv[at * cn + c]
        } else {
            rows.qkv[(at - WINDOW) * cn + c]
        };
        acc = acc + p.conv_w[c * TAPS + tap] * x;
    }
    match mode {
        Rounding::Fused => bf16::round(silu(acc)),
        Rounding::Unfused => bf16::round(silu(bf16::round(acc))),
        Rounding::F32 => silu(acc),
    }
}

/// Head `h`'s q, k and v after the conv for row `r`.
pub fn conv_qkv(
    p: &LayerParams,
    conv: &[f32],
    rows: &Rows,
    r: usize,
    h: usize,
    mode: Rounding,
) -> ([f32; DK], [f32; DK], [f32; DV]) {
    let hd = p.heads * DK;
    let mut q = [0.0f32; DK];
    let mut k = [0.0f32; DK];
    let mut v = [0.0f32; DV];
    for i in 0..DK {
        q[i] = conv_channel(p, conv, rows, r, h * DK + i, mode);
        k[i] = conv_channel(p, conv, rows, r, hd + h * DK + i, mode);
        v[i] = conv_channel(p, conv, rows, r, 2 * hd + h * DK + i, mode);
    }
    (q, k, v)
}

/// The kernels' L2 norm: `x * (1 / sqrt(|x|^2 + 1e-6))`, then times `scale` if given.
pub fn l2norm(x: &[f32; DK], scale: Option<f32>) -> [f32; DK] {
    let inv = 1.0 / (dot128(x, x) + L2_EPS).sqrt();
    let mut y = [0.0f32; DK];
    for i in 0..DK {
        y[i] = x[i] * inv;
        if let Some(s) = scale {
            y[i] = y[i] * s;
        }
    }
    y
}

/// The q scale, `1 / sqrt(DK)`.
pub fn q_scale() -> f32 {
    1.0 / (DK as f32).sqrt()
}

/// One delta-rule step on a head's state `s` (`[DV][DK]`), in the kernels' order: decay each
/// key column by `g`, read with `k`, correct toward `v` by `beta`.
pub fn update(s: &mut [f32], k: &[f32], g: &[f32], v: &[f32], beta: f32) {
    debug_assert_eq!(s.len(), DV * DK);
    for (row, &v_row) in v.iter().enumerate().take(DV) {
        let sr = &mut s[row * DK..(row + 1) * DK];
        let mut lanes = [0.0f32; 32];
        for (l, lane) in lanes.iter_mut().enumerate() {
            let mut kv = 0.0f32;
            for i in 0..4 {
                let c = 4 * l + i;
                sr[c] = sr[c] * g[c];
                kv = kv + sr[c] * k[c];
            }
            *lane = kv;
        }
        let kv = warp_sum(&lanes);
        let delta = (v_row - kv) * beta;
        for c in 0..DK {
            sr[c] = sr[c] + k[c] * delta;
        }
    }
}

/// The read-out `y = S^T q` of a head's state (`[DV][DK]`), in the kernels' order, not rounded.
pub fn read_out(s: &[f32], q: &[f32]) -> [f32; DV] {
    let mut y = [0.0f32; DV];
    for (row, yr) in y.iter_mut().enumerate() {
        *yr = dot128(&s[row * DK..(row + 1) * DK], q);
    }
    y
}

/// The gated RMSNorm in the kernels' order, rounded to bfloat16.
pub fn gated_rmsnorm(y: &[f32], norm_w: &[f32], gate: &[f32], eps: f32) -> [f32; DV] {
    gated_rmsnorm_as(y, norm_w, gate, eps, Rounding::Fused)
}

/// The gated RMSNorm in the kernels' order, rounded as `mode` says.
pub fn gated_rmsnorm_as(
    y: &[f32],
    norm_w: &[f32],
    gate: &[f32],
    eps: f32,
    mode: Rounding,
) -> [f32; DV] {
    let rinv = 1.0 / (dot128(y, y) / DV as f32 + eps).sqrt();
    let mut o = [0.0f32; DV];
    for t in 0..DV {
        let yn = y[t] * rinv;
        let yw = norm_w[t] * yn;
        o[t] = mode.round(yw * sigmoid(gate[t]));
    }
    o
}

/// One layer, `rows.rows` consecutive rows from `state` (`[H][DV][DK]`) with conv window `conv`
/// (`[WINDOW][C]`): the kernels' `chain`.
pub fn chain(
    p: &LayerParams,
    conv: &[f32],
    state: &[f32],
    rows: &Rows,
    mode: Rounding,
) -> ChainOut {
    chain_with(p, conv, state, rows, mode, false)
}

/// [`chain`] with the state stored in bfloat16 (the kernels' `_bf16state` chain, decision D8):
/// `state` holds bfloat16-exact values, every step is computed in f32, and the state is rounded
/// to bfloat16 after each row's read-out. A window of rows therefore equals serial single-row
/// calls, each of which stores its state in bfloat16.
pub fn chain_bf16_state(
    p: &LayerParams,
    conv: &[f32],
    state: &[f32],
    rows: &Rows,
    mode: Rounding,
) -> ChainOut {
    chain_with(p, conv, state, rows, mode, true)
}

fn round_state(s: &mut [f32]) {
    for x in s.iter_mut() {
        *x = bf16::round(*x);
    }
}

fn chain_with(
    p: &LayerParams,
    conv: &[f32],
    state: &[f32],
    rows: &Rows,
    mode: Rounding,
    bf16_state: bool,
) -> ChainOut {
    p.check();
    rows.check();
    let (hn, rn) = (p.heads, rows.rows);
    assert_eq!(rows.heads, hn);
    assert_eq!(
        conv.len(),
        WINDOW * p.channels(),
        "conv window must be [WINDOW][C]"
    );
    assert_eq!(state.len(), state_len(hn), "state must be [H][DV][DK]");
    let mut st = state.to_vec();
    let mut out = vec![0.0f32; rn * hn * DV];
    let mut ys = vec![0.0f32; rn * hn * DV];
    let mut saves = Saves::zeros(hn, rn);
    for h in 0..hn {
        let decay_rate = p.a_log[h].exp();
        let s = &mut st[h * DV * DK..(h + 1) * DV * DK];
        for r in 0..rn {
            let (q, k, v) = conv_qkv(p, conv, rows, r, h, mode);
            let mut g = [0.0f32; DK];
            for (i, gi) in g.iter_mut().enumerate() {
                *gi = decay(
                    rows.a[(r * hn + h) * DK + i],
                    p.dt_bias[h * DK + i],
                    decay_rate,
                    p.lower,
                );
            }
            let b = mode.round(sigmoid(rows.b[r * hn + h]));
            let qn = l2norm(&q, Some(q_scale()));
            let kn = l2norm(&k, None);
            update(s, &kn, &g, &v, b);
            let y = read_out(s, &qn).map(|x| mode.round(x));
            if bf16_state {
                round_state(s);
            }
            let gate = &rows.gate[(r * hn + h) * DV..(r * hn + h + 1) * DV];
            let o = gated_rmsnorm_as(&y, &p.norm_w, gate, p.eps, mode);
            let at = (r * hn + h) * DV;
            out[at..at + DV].copy_from_slice(&o);
            ys[at..at + DV].copy_from_slice(&y);
            saves.k[at..at + DK].copy_from_slice(&kn);
            saves.v[at..at + DV].copy_from_slice(&v);
            saves.g[at..at + DK].copy_from_slice(&g);
            saves.beta[r * hn + h] = b;
        }
    }
    ChainOut {
        out,
        y: ys,
        state: st,
        saves,
    }
}

/// The state after the first `keep` saved rows, from `state` (`[H][DV][DK]`): the kernels'
/// `replay`. Uses [`update`], as [`chain`] does, so a replayed prefix has the chain's bits.
pub fn replay(state: &[f32], saves: &Saves, keep: usize) -> Vec<f32> {
    replay_with(state, saves, keep, false)
}

/// [`replay`] of a bfloat16 state (the kernels' `_bf16state` replay): each row's update in f32,
/// then the state rounded to bfloat16, as [`chain_bf16_state`] does. The update involves no
/// transcendental function, so this is a bit-level model of the device.
pub fn replay_bf16_state(state: &[f32], saves: &Saves, keep: usize) -> Vec<f32> {
    replay_with(state, saves, keep, true)
}

fn replay_with(state: &[f32], saves: &Saves, keep: usize, bf16_state: bool) -> Vec<f32> {
    let hn = saves.heads;
    assert!(keep <= saves.rows);
    assert_eq!(state.len(), state_len(hn));
    let mut st = state.to_vec();
    for h in 0..hn {
        let s = &mut st[h * DV * DK..(h + 1) * DV * DK];
        for r in 0..keep {
            let at = (r * hn + h) * DK;
            update(
                s,
                &saves.k[at..at + DK],
                &saves.g[at..at + DK],
                &saves.v[at..at + DV],
                saves.beta[r * hn + h],
            );
            if bf16_state {
                round_state(s);
            }
        }
    }
    st
}

/// The conv window after keeping `keep` of `rows`: rows `keep .. keep + WINDOW` of
/// `[conv; rows.qkv]`.
pub fn conv_shift(conv: &[f32], rows: &Rows, keep: usize) -> Vec<f32> {
    let cn = channels(rows.heads);
    assert_eq!(conv.len(), WINDOW * cn);
    assert!(keep <= rows.rows);
    let mut w = vec![0.0f32; WINDOW * cn];
    for j in 0..WINDOW {
        let src = keep + j;
        let from = if src < WINDOW {
            &conv[src * cn..(src + 1) * cn]
        } else {
            &rows.qkv[(src - WINDOW) * cn..(src - WINDOW + 1) * cn]
        };
        w[j * cn..(j + 1) * cn].copy_from_slice(from);
    }
    w
}

/// `[H][A][B] -> [H][B][A]` for 128 × 128 heads: converts between the kernels' state layout
/// `[H][DV][DK]` and the reference's `[H][DK][DV]` (it is its own inverse).
pub fn transpose_state(s: &[f32]) -> Vec<f32> {
    assert_eq!(s.len() % (DK * DV), 0);
    let mut t = vec![0.0f32; s.len()];
    for (hs, ht) in s
        .as_chunks::<{ DK * DV }>()
        .0
        .iter()
        .zip(t.as_chunks_mut::<{ DK * DV }>().0)
    {
        for a in 0..DV {
            for b in 0..DK {
                ht[b * DV + a] = hs[a * DK + b];
            }
        }
    }
    t
}

/// The reference's formulation, evaluated as written: the state key-major (`[DK][DV]` per
/// head), sums in index order, the L2 norm as a division, `rsqrt` as `1 / sqrt`. Shares the
/// conv, the gates and the rounding points with [`chain`]; it differs only in the order of
/// operations.
pub mod literal {
    use super::*;

    /// `x / sqrt(sum(x * x) + 1e-6)`, the reference's `l2norm`.
    pub fn l2norm(x: &[f32]) -> [f32; DK] {
        let mut ss = 0.0f32;
        for &xi in x.iter().take(DK) {
            ss = ss + xi * xi;
        }
        let n = (ss + L2_EPS).sqrt();
        let mut y = [0.0f32; DK];
        for i in 0..DK {
            y[i] = x[i] / n;
        }
        y
    }

    /// One `recurrent_kimi_delta_attention` step for one head on state `s` (`[DK][DV]`): `q`,
    /// `k`, `v` after the conv, `g` in log space, `beta` after its rounding. Returns the
    /// read-out before rounding.
    pub fn step(s: &mut [f32], q: &[f32], k: &[f32], v: &[f32], g: &[f32], beta: f32) -> [f32; DV] {
        debug_assert_eq!(s.len(), DK * DV);
        let qn = l2norm(q);
        let kn = l2norm(k);
        let scale = q_scale();
        let mut qs = [0.0f32; DK];
        for i in 0..DK {
            qs[i] = qn[i] * scale;
        }
        for (kk, &gk) in g.iter().enumerate().take(DK) {
            let e = gk.exp();
            for vv in 0..DV {
                s[kk * DV + vv] = s[kk * DV + vv] * e;
            }
        }
        let mut delta = [0.0f32; DV];
        for vv in 0..DV {
            let mut kv = 0.0f32;
            for kk in 0..DK {
                kv = kv + s[kk * DV + vv] * kn[kk];
            }
            delta[vv] = (v[vv] - kv) * beta;
        }
        for kk in 0..DK {
            for vv in 0..DV {
                s[kk * DV + vv] = s[kk * DV + vv] + kn[kk] * delta[vv];
            }
        }
        let mut y = [0.0f32; DV];
        for (vv, yv) in y.iter_mut().enumerate() {
            let mut o = 0.0f32;
            for kk in 0..DK {
                o = o + s[kk * DV + vv] * qs[kk];
            }
            *yv = o;
        }
        y
    }

    /// `Glm5NextTextRMSNormGated`, rounded to bfloat16.
    pub fn gated_rmsnorm(y: &[f32], norm_w: &[f32], gate: &[f32], eps: f32) -> [f32; DV] {
        gated_rmsnorm_as(y, norm_w, gate, eps, Rounding::Fused)
    }

    /// `Glm5NextTextRMSNormGated`, rounded as `mode` says.
    pub fn gated_rmsnorm_as(
        y: &[f32],
        norm_w: &[f32],
        gate: &[f32],
        eps: f32,
        mode: Rounding,
    ) -> [f32; DV] {
        let mut ss = 0.0f32;
        for &yi in y.iter().take(DV) {
            ss = ss + yi * yi;
        }
        let r = 1.0 / (ss / DV as f32 + eps).sqrt();
        let mut o = [0.0f32; DV];
        for t in 0..DV {
            o[t] = mode.round(norm_w[t] * (y[t] * r) * sigmoid(gate[t]));
        }
        o
    }

    /// [`super::chain`] in the reference's formulation. The returned state is converted to the
    /// kernels' layout `[H][DV][DK]` so the two can be compared directly.
    pub fn chain(
        p: &LayerParams,
        conv: &[f32],
        state: &[f32],
        rows: &Rows,
        mode: Rounding,
    ) -> ChainOut {
        p.check();
        rows.check();
        let (hn, rn) = (p.heads, rows.rows);
        let mut st = transpose_state(state);
        let mut out = vec![0.0f32; rn * hn * DV];
        let mut ys = vec![0.0f32; rn * hn * DV];
        let mut saves = Saves::zeros(hn, rn);
        for h in 0..hn {
            let s = &mut st[h * DK * DV..(h + 1) * DK * DV];
            for r in 0..rn {
                let (q, k, v) = conv_qkv(p, conv, rows, r, h, mode);
                let mut g = [0.0f32; DK];
                for (i, gi) in g.iter_mut().enumerate() {
                    *gi = log_decay(
                        rows.a[(r * hn + h) * DK + i],
                        p.dt_bias[h * DK + i],
                        p.a_log[h],
                        p.lower,
                    );
                }
                let b = mode.round(sigmoid(rows.b[r * hn + h]));
                let y = step(s, &q, &k, &v, &g, b).map(|x| mode.round(x));
                let gate = &rows.gate[(r * hn + h) * DV..(r * hn + h + 1) * DV];
                let o = gated_rmsnorm_as(&y, &p.norm_w, gate, p.eps, mode);
                let at = (r * hn + h) * DV;
                out[at..at + DV].copy_from_slice(&o);
                ys[at..at + DV].copy_from_slice(&y);
                saves.k[at..at + DK].copy_from_slice(&l2norm(&k));
                saves.v[at..at + DV].copy_from_slice(&v);
                saves.g[at..at + DK].copy_from_slice(&g.map(f32::exp));
                saves.beta[r * hn + h] = b;
            }
        }
        ChainOut {
            out,
            y: ys,
            state: transpose_state(&st),
            saves,
        }
    }
}
