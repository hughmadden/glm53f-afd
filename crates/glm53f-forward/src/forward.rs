//! The forward (feature `cuda`): embedding, the mHC layer loop, the head, over batches of rows
//! from one or more requests, in three modes.
//!
//! | Mode | Rows | KDA | DSA tails | Afterwards |
//! |---|---|---|---|---|
//! | prefill | a segment per request (any length; cut into chunks of [`ForwardConfig::max_rows`]) | chain in place, conv shifted (or the chunked kernel: [`ForwardConfig::kda_chunked_prefill`]) | committed | `tokens += rows` |
//! | decode | one per request | chain in place, conv shifted | committed | `tokens += 1` |
//! | verify | a window of R <= 8 per request | chain without a state write, replay inputs saved | pending | `pending = R` |
//!
//! [`GlmForward::commit`] ends a verify round: the KDA states are replayed over the kept rows
//! from the saved inputs (bit for bit the state of serial steps), the conv windows shifted past
//! them, the DSA tails rewritten from the kept rows, and `tokens += keep`. The MLA latents and
//! pooled keys the rejected rows wrote stay past the committed length, where nothing reads
//! them and the next rows overwrite them.
//!
//! # Layer flow (`transformers` `Glm5NextTextDecoderLayer`)
//!
//! ```text
//! streams = embedding rows x 4                                   (host table, gathered on the GPU)
//! per layer: mHC attn boundary (expanding the previous FFN output) -> input_layernorm
//!            -> KDA or DSA attention
//!            mHC ffn boundary (expanding the attention output) -> post_attention_layernorm
//!            -> dense MLP (layers 0-2) or MoE: router, routed experts (backend), shared expert,
//!               bf16(routed + shared)
//! head: expand the last FFN output, mean of the 4 streams, final RMSNorm -> LM head -> argmax
//! ```
//!
//! Rows are row-independent in every kernel up to 8 rows (the GEMV, the FP8 decode GEMM, the
//! single-launch mHC boundary, the router, KDA, the DSA kernels with a fixed split plan), so a
//! verify window and a short prefill give the bits of serial decode steps. Larger passes use
//! tensor-core GEMMs (cuBLAS, the FP8 W8A8 GEMM) whose rounding depends on the row count.
//!
//! With a drafter attached ([`GlmForward::attach_drafter`], `crate::draft`), every pass through
//! the head also captures the drafter's taps at the entry of layers 6, 15, 25, 34 and 43, and
//! the committed rows (a prefill or decode pass's, a commit's kept rows) go to each slot's
//! drafter context.

use core::ffi::c_void;
use std::cell::RefCell;
use std::ops::Range;
use std::sync::Arc;

use glm53f_dsa::ffi::{self as dffi, DsaCache, DsaWindow};
use glm53f_kda::ffi as kffi;
use glm53f_layers::ffi as lffi;

use crate::device::{self, launched, DeviceBuffer, Event, Stream};
use crate::draft::{Dflash, DraftReq, LAYERS_NEEDED};
use crate::embed::HostEmbedding;
use crate::error::{invalid, Result};
use crate::experts::{ExpertBackend, ExpertCall};
use crate::ffi;
use crate::gemm::{act_quant, Fp8Input, Gemm, GemmPolicy};
use crate::kv::{GlmKv, KvPool};
use crate::kvplan::{KvLayout, LAYER_PAGE_BYTES, TAIL};
use crate::shape::*;
use crate::weights::{AttnW, DeviceModel, DsaW, FfnW, HcW, KdaW, MlpW};

/// Sizes and kernel choices of a forward.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ForwardConfig {
    /// Rows of one prefill pass (a longer segment runs in chunks of this many).
    pub max_rows: usize,
    /// Rows of one verify pass, all windows together.
    pub max_verify_rows: usize,
    /// Requests in one pass.
    pub max_requests: usize,
    pub policy: GemmPolicy,
    /// Sparse MLA splits for passes of at most 8 rows (fixed, so a row's result does not
    /// depend on the pass).
    pub decode_splits: usize,
    /// Sparse MLA head groups per block for passes of at most 8 rows, and for larger passes.
    pub decode_head_groups: usize,
    pub prefill_head_groups: usize,
    /// Prefill passes of more than 8 rows run KDA through the chunked kernel
    /// (`glm53f_kda_prefill_batch`) instead of the chain. It agrees with the chain to f32 rounding,
    /// not bit for bit: a numerics change, off until it passes the KL gate.
    pub kda_chunked_prefill: bool,
    /// Rows per request of each chunked-prefill pass (its workspace: 2.2 MB per 16 rows of a
    /// request), and its value-column blocks per head (fixed, so the bits do not depend on the GPU).
    pub kda_prefill_rows: usize,
    pub kda_prefill_value_blocks: i32,
}

impl Default for ForwardConfig {
    fn default() -> Self {
        ForwardConfig {
            max_rows: 256,
            max_verify_rows: 64,
            max_requests: 16,
            policy: GemmPolicy::default(),
            decode_splits: 16,
            decode_head_groups: 1,
            prefill_head_groups: 4,
            kda_chunked_prefill: false,
            kda_prefill_rows: 256,
            kda_prefill_value_blocks: 2,
        }
    }
}

/// What a pass does to its slots.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Mode {
    Prefill,
    Decode,
    Verify,
}

/// Where a pass's first streams come from.
pub enum Input<'a> {
    /// Token ids, one per row (embedding gather).
    Tokens(&'a [u32]),
    /// Token ids already on the device (i32, one per row), e.g. the previous step's argmax.
    DeviceIds(*const i32),
    /// The layer input streams, BF16 `[rows][4][4096]` (tests: a layer fed its golden input).
    Streams(&'a [u16]),
}

/// A point in the layer loop where a tap sees the scratch.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum TapPoint {
    /// After the attention sublayer: the attention boundary's outputs, the attention
    /// intermediates and the attention output are valid.
    AttnDone,
    /// After the FFN sublayer: the FFN boundary's outputs, the FFN intermediates and outputs.
    FfnDone,
    /// The layer's output streams (materialized for the tap).
    LayerOut,
}

/// Buffers a tap can read.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum TapBuf {
    /// The streams a boundary read: the layer input at `AttnDone`, the mid streams at
    /// `FfnDone`, the layer output at `LayerOut`. BF16 `[rows][4][4096]`.
    Streams,
    /// The boundary's `pre`, `post` f32 `[rows][4]`, `comb` f32 `[rows][16]` (`[src][dst]`).
    Pre,
    Post,
    Comb,
    /// The collapsed sublayer input and its RMSNorm, BF16 `[rows][4096]`.
    Collapsed,
    Normed,
    /// The sublayer output, BF16 `[rows][4096]` (for an MoE layer, the routed sum).
    Out,
    /// The MoE shared expert's output, BF16 `[rows][4096]`.
    SharedOut,
    /// KDA: q | k | v (before the conv) | beta logits, BF16 `[rows][24,640]`.
    KdaP,
    /// KDA: forget-gate | output-gate projections, BF16 `[rows][16,384]`.
    KdaGates,
    /// KDA: the gated RMSNorm output, BF16 `[rows][8192]`.
    KdaNormOut,
    /// DSA: `q_a_layernorm` output BF16 `[rows][1536]`; query f32 `[rows][64][256]`; `kv_a`
    /// BF16 `[rows][512]`; index query f32 `[rows][32][128]`; wk | gate | weights_proj
    /// BF16 `[rows][288]`; selected tokens i32 `[rows][2051]` and counts `[rows][2]`; per-head
    /// output before `o_proj` f32 `[rows][64][256]`.
    DsaQResid,
    DsaQ,
    DsaKvA,
    DsaIdxQ,
    DsaIdxProj,
    DsaTokens,
    DsaCounts,
    DsaHeads,
    /// Router logits f32 `[rows][288]`, chosen ids i32 and weights f32 `[rows][8]`.
    RouterLogits,
    RouterIds,
    RouterWeights,
}

/// What a tap is given: the layer, the pass's rows and a way to read the scratch.
pub struct Tap<'a> {
    pub layer: usize,
    pub point: TapPoint,
    pub mode: Mode,
    pub rows: usize,
    s: &'a Scratch,
}

impl Tap<'_> {
    fn buf(&self, b: TapBuf) -> (&DeviceBuffer, usize) {
        let s = self.s;
        match b {
            TapBuf::Streams => (&s.tap_streams, HC * HIDDEN * 2),
            TapBuf::Pre => (&s.pre, 16),
            TapBuf::Post => (
                if self.point == TapPoint::AttnDone {
                    &s.attn_post
                } else {
                    &s.ffn_post
                },
                16,
            ),
            TapBuf::Comb => (
                if self.point == TapPoint::AttnDone {
                    &s.attn_comb
                } else {
                    &s.ffn_comb
                },
                64,
            ),
            TapBuf::Collapsed => (&s.collapsed, HIDDEN * 2),
            TapBuf::Normed => (&s.normed, HIDDEN * 2),
            TapBuf::Out => (
                if self.point == TapPoint::AttnDone {
                    &s.attn_out
                } else {
                    &s.ffn_out
                },
                HIDDEN * 2,
            ),
            TapBuf::SharedOut => (&s.ffn_out2, HIDDEN * 2),
            TapBuf::KdaP => (&s.p, KDA_P_COLS * 2),
            TapBuf::KdaGates => (&s.ag, 2 * KDA_WIDTH * 2),
            TapBuf::KdaNormOut => (&s.kda_out, KDA_WIDTH * 2),
            TapBuf::DsaQResid => (&s.q_resid, Q_LORA * 2),
            TapBuf::DsaQ => (&s.q32, MLA_HEADS * QK_HEAD * 4),
            TapBuf::DsaKvA => (&s.kva, KV_LORA * 2),
            TapBuf::DsaIdxQ => (&s.idx_q32, INDEX_HEADS * INDEX_DIM * 4),
            TapBuf::DsaIdxProj => (&s.idx_p, IDX_PROJ_COLS * 2),
            TapBuf::DsaTokens => (&s.tokens, MAX_SELECTED * 4),
            TapBuf::DsaCounts => (&s.counts, 8),
            TapBuf::DsaHeads => (&s.o32, MLA_HEADS * V_HEAD * 4),
            TapBuf::RouterLogits => (&s.router_logits, EXPERTS * 4),
            TapBuf::RouterIds => (&s.ids, TOP_K * 4),
            TapBuf::RouterWeights => (&s.weights, TOP_K * 4),
        }
    }

    /// The first `rows` rows of a buffer as raw bytes.
    pub fn bytes(&self, b: TapBuf) -> Result<Vec<u8>> {
        let (buf, row) = self.buf(b);
        buf.download::<u8>(self.rows * row)
    }

    pub fn f32(&self, b: TapBuf) -> Result<Vec<f32>> {
        Ok(self
            .bytes(b)?
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| f32::from_le_bytes(*c))
            .collect())
    }

    pub fn bf16(&self, b: TapBuf) -> Result<Vec<u16>> {
        Ok(self
            .bytes(b)?
            .as_chunks::<2>()
            .0
            .iter()
            .map(|c| u16::from_le_bytes(*c))
            .collect())
    }

    pub fn i32(&self, b: TapBuf) -> Result<Vec<i32>> {
        Ok(self
            .bytes(b)?
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| i32::from_le_bytes(*c))
            .collect())
    }
}

/// A tap: called at every [`TapPoint`] of every layer while installed (with the stream
/// synchronized, so it may download what it needs).
pub type TapFn = dyn FnMut(&Tap<'_>) -> Result<()> + Send;

/// Per-stage GPU times of the last pass, when timing is on.
#[derive(Clone, Debug, Default)]
pub struct StageTimes {
    /// (layer, or `usize::MAX` for work outside the layers; stage; milliseconds), in order.
    pub stages: Vec<(usize, &'static str, f64)>,
}

impl StageTimes {
    /// Total milliseconds.
    pub fn total(&self) -> f64 {
        self.stages.iter().map(|s| s.2).sum()
    }

    /// Milliseconds of `stage` in `layer`.
    pub fn get(&self, layer: usize, stage: &str) -> f64 {
        self.stages
            .iter()
            .filter(|s| s.0 == layer && s.1 == stage)
            .map(|s| s.2)
            .sum()
    }
}

struct Timer {
    events: Vec<Event>,
    marks: Vec<(usize, &'static str)>,
}

impl Timer {
    fn mark(&mut self, stream: &Stream, layer: usize, stage: &'static str) -> Result<()> {
        let i = self.marks.len();
        if i == self.events.len() {
            self.events.push(Event::new()?);
        }
        self.events[i].record(stream)?;
        self.marks.push((layer, stage));
        Ok(())
    }

    fn collect(&mut self) -> Result<StageTimes> {
        let mut out = StageTimes::default();
        for i in 1..self.marks.len() {
            let ms = self.events[i].elapsed_ms_since(&self.events[i - 1])? as f64;
            out.stages.push((self.marks[i].0, self.marks[i].1, ms));
        }
        self.marks.clear();
        Ok(out)
    }
}

fn buf(bytes: usize) -> Result<DeviceBuffer> {
    DeviceBuffer::alloc(bytes.max(16))
}

/// Every per-pass buffer, sized for `rows` rows and `logits` logit rows.
pub(crate) struct Scratch {
    rows: usize,
    logit_rows: usize,
    streams: [DeviceBuffer; 2],
    tap_streams: DeviceBuffer,
    partials: DeviceBuffer,
    pre: DeviceBuffer,
    collapsed: DeviceBuffer,
    normed: DeviceBuffer,
    normed_q: DeviceBuffer,
    normed_s: DeviceBuffer,
    attn_post: DeviceBuffer,
    attn_comb: DeviceBuffer,
    ffn_post: DeviceBuffer,
    ffn_comb: DeviceBuffer,
    attn_out: DeviceBuffer,
    ffn_out: DeviceBuffer,
    ffn_out2: DeviceBuffer,
    sync: DeviceBuffer,
    // KDA
    p: DeviceBuffer,
    fga: DeviceBuffer,
    ag: DeviceBuffer,
    kda_out: DeviceBuffer,
    // DSA
    qa: DeviceBuffer,
    kva: DeviceBuffer,
    q_resid: DeviceBuffer,
    q_resid_q: DeviceBuffer,
    q_resid_s: DeviceBuffer,
    q16: DeviceBuffer,
    q32: DeviceBuffer,
    idx_q16: DeviceBuffer,
    idx_q32: DeviceBuffer,
    idx_p: DeviceBuffer,
    kva32: DeviceBuffer,
    k_raw: DeviceBuffer,
    gate: DeviceBuffer,
    w32: DeviceBuffer,
    q_abs: DeviceBuffer,
    tokens: DeviceBuffer,
    pools: DeviceBuffer,
    counts: DeviceBuffer,
    o_lat: DeviceBuffer,
    lse: DeviceBuffer,
    o32: DeviceBuffer,
    o16: DeviceBuffer,
    o_q: DeviceBuffer,
    o_s: DeviceBuffer,
    idx_ws: DeviceBuffer,
    kda_ws: DeviceBuffer,
    mla_ws: DeviceBuffer,
    // FFN
    gu: DeviceBuffer,
    act: DeviceBuffer,
    act_q: DeviceBuffer,
    act_s: DeviceBuffer,
    router_logits: DeviceBuffer,
    ids: DeviceBuffer,
    weights: DeviceBuffer,
    // Head
    head_out: DeviceBuffer,
    head_sel: DeviceBuffer,
    logits: DeviceBuffer,
    next: DeviceBuffer,
    // Per-pass metadata and the batch's page tables and tails.
    meta: DeviceBuffer,
    batch_table: DeviceBuffer,
    batch_tails: DeviceBuffer,
}

impl Scratch {
    fn new(
        rows: usize,
        logit_rows: usize,
        requests: usize,
        max_pages: usize,
        dsa_layers: usize,
    ) -> Result<Scratch> {
        let r = rows;
        Ok(Scratch {
            rows,
            logit_rows,
            streams: [buf(r * HC * HIDDEN * 2)?, buf(r * HC * HIDDEN * 2)?],
            tap_streams: buf(r * HC * HIDDEN * 2)?,
            partials: buf(r * (HIDDEN / 128) * 25 * 4)?,
            pre: buf(r * 16)?,
            collapsed: buf(r * HIDDEN * 2)?,
            normed: buf(r * HIDDEN * 2)?,
            normed_q: buf(r * HIDDEN)?,
            normed_s: buf(r * (HIDDEN / 128) * 4)?,
            attn_post: buf(r * 16)?,
            attn_comb: buf(r * 64)?,
            ffn_post: buf(r * 16)?,
            ffn_comb: buf(r * 64)?,
            attn_out: buf(r * HIDDEN * 2)?,
            ffn_out: buf(r * HIDDEN * 2)?,
            ffn_out2: buf(r * HIDDEN * 2)?,
            sync: DeviceBuffer::zeroed(r.max(64) * 4)?,
            p: buf(r * KDA_P_COLS * 2)?,
            fga: buf(r * 2 * KDA_DIM * 2)?,
            ag: buf(r * 2 * KDA_WIDTH * 2)?,
            kda_out: buf(r * KDA_WIDTH * 2)?,
            qa: buf(r * Q_LORA * 2)?,
            kva: buf(r * KV_LORA * 2)?,
            q_resid: buf(r * Q_LORA * 2)?,
            q_resid_q: buf(r * Q_LORA)?,
            q_resid_s: buf(r * (Q_LORA / 128) * 4)?,
            q16: buf(r * MLA_HEADS * QK_HEAD * 2)?,
            q32: buf(r * MLA_HEADS * QK_HEAD * 4)?,
            idx_q16: buf(r * INDEX_HEADS * INDEX_DIM * 2)?,
            idx_q32: buf(r * INDEX_HEADS * INDEX_DIM * 4)?,
            idx_p: buf(r * IDX_PROJ_COLS * 2)?,
            kva32: buf(r * KV_LORA * 4)?,
            k_raw: buf(r * INDEX_DIM * 4)?,
            gate: buf(r * INDEX_DIM * 4)?,
            w32: buf(r * INDEX_HEADS * 4)?,
            q_abs: buf(r * MLA_HEADS * KV_LORA * 2)?,
            tokens: buf(r * MAX_SELECTED * 4)?,
            pools: buf(r * TOP_POOLS * 4)?,
            counts: buf(r * 2 * 4)?,
            o_lat: buf(r * MLA_HEADS * KV_LORA * 4)?,
            lse: buf(r * MLA_HEADS * 4)?,
            o32: buf(r * MLA_HEADS * V_HEAD * 4)?,
            o16: buf(r * MLA_HEADS * V_HEAD * 2)?,
            o_q: buf(r * MLA_HEADS * V_HEAD)?,
            o_s: buf(r * (MLA_HEADS * V_HEAD / 128) * 4)?,
            idx_ws: buf(0)?,
            kda_ws: buf(0)?,
            mla_ws: buf(0)?,
            gu: buf(r * 2 * DENSE_INTER * 2)?,
            act: buf(r * DENSE_INTER * 2)?,
            act_q: buf(r * DENSE_INTER)?,
            act_s: buf(r * (DENSE_INTER / 128) * 4)?,
            router_logits: buf(r * EXPERTS * 4)?,
            ids: buf(r * TOP_K * 4)?,
            weights: buf(r * TOP_K * 4)?,
            head_out: buf(r * HIDDEN * 2)?,
            head_sel: buf(logit_rows * HIDDEN * 2)?,
            logits: buf(logit_rows * VOCAB * 4)?,
            next: buf(logit_rows * 4)?,
            meta: buf(64 * 1024 + r * 64 + requests * (128 + 16 * dsa_layers))?,
            batch_table: buf(requests * max_pages * 4)?,
            batch_tails: buf(dsa_layers * requests * TAIL)?,
        })
    }

    fn bytes(&self) -> usize {
        let v = [
            &self.streams[0],
            &self.streams[1],
            &self.tap_streams,
            &self.partials,
            &self.pre,
            &self.collapsed,
            &self.normed,
            &self.normed_q,
            &self.normed_s,
            &self.attn_post,
            &self.attn_comb,
            &self.ffn_post,
            &self.ffn_comb,
            &self.attn_out,
            &self.ffn_out,
            &self.ffn_out2,
            &self.sync,
            &self.p,
            &self.fga,
            &self.ag,
            &self.kda_out,
            &self.qa,
            &self.kva,
            &self.q_resid,
            &self.q_resid_q,
            &self.q_resid_s,
            &self.q16,
            &self.q32,
            &self.idx_q16,
            &self.idx_q32,
            &self.idx_p,
            &self.kva32,
            &self.k_raw,
            &self.gate,
            &self.w32,
            &self.q_abs,
            &self.tokens,
            &self.pools,
            &self.counts,
            &self.o_lat,
            &self.lse,
            &self.o32,
            &self.o16,
            &self.o_q,
            &self.o_s,
            &self.idx_ws,
            &self.mla_ws,
            &self.kda_ws,
            &self.gu,
            &self.act,
            &self.act_q,
            &self.act_s,
            &self.router_logits,
            &self.ids,
            &self.weights,
            &self.head_out,
            &self.head_sel,
            &self.logits,
            &self.next,
            &self.meta,
            &self.batch_table,
            &self.batch_tails,
        ];
        v.iter().map(|b| b.bytes()).sum()
    }
}

/// Buffers a verify round keeps until its commit: every KDA layer's projection rows (the conv
/// shift reads them) and replay inputs, every DSA layer's raw index keys and gates (the tail
/// commit reads them).
struct VerifyScratch {
    rows: usize,
    p: DeviceBuffer,
    k: DeviceBuffer,
    v: DeviceBuffer,
    g: DeviceBuffer,
    b: DeviceBuffer,
    k_raw: DeviceBuffer,
    gate: DeviceBuffer,
}

impl VerifyScratch {
    fn new(rows: usize, kda_layers: usize, dsa_layers: usize) -> Result<VerifyScratch> {
        let (r, kl, dl) = (rows, kda_layers, dsa_layers);
        Ok(VerifyScratch {
            rows,
            p: buf(kl * r * KDA_P_COLS * 2)?,
            k: buf(kl * r * KDA_WIDTH * 4)?,
            v: buf(kl * r * KDA_WIDTH * 2)?,
            g: buf(kl * r * KDA_WIDTH * 4)?,
            b: buf(kl * r * KDA_HEADS * 4)?,
            k_raw: buf(dl * r * INDEX_DIM * 4)?,
            gate: buf(dl * r * INDEX_DIM * 4)?,
        })
    }

    fn bytes(&self) -> usize {
        [
            &self.p,
            &self.k,
            &self.v,
            &self.g,
            &self.b,
            &self.k_raw,
            &self.gate,
        ]
        .iter()
        .map(|b| b.bytes())
        .sum()
    }
}

/// One request's rows in a pass.
#[derive(Clone, Copy, Debug)]
struct Req {
    slot: usize,
    start: usize,
    rows: usize,
    row0: usize,
    state_off: usize,
    conv_off: usize,
    tail_row: usize,
}

/// Device views of a pass's metadata.
#[derive(Clone, Copy)]
struct Meta {
    row_req: *const i32,
    row_pos: *const i32,
    ids: *const i32,
    cu_rows: *const i32,
    keep: *const i32,
    conv_off: *const i64,
    state_off: *const i64,
    windows: *const DsaWindow,
    tail_idx: *const i32,
    slots: *const i32,
    logit_rows: *const i32,
}

/// A verify round waiting for its commit.
struct Pending {
    reqs: Vec<Req>,
    rows: usize,
}

/// The GLM-5.3-Flash forward over decoder layers `0 .. model.shape.layers`.
pub struct GlmForward {
    pub model: Arc<DeviceModel>,
    pub embed: HostEmbedding,
    pub kv: KvPool,
    experts: Box<dyn ExpertBackend>,
    gemm: Gemm,
    stream: Arc<Stream>,
    pub cfg: ForwardConfig,
    s: Scratch,
    v: VerifyScratch,
    pending: Option<Pending>,
    timer: RefCell<Option<Timer>>,
    tap: Option<Box<TapFn>>,
    sms: i32,
    host_ids: Vec<i32>,
    host_weights: Vec<f32>,
    /// The DFlash2 drafter, when attached.
    draft: Option<Dflash>,
}

fn pack<T: Copy>(bytes: &mut Vec<u8>, v: &[T]) -> usize {
    while !bytes.len().is_multiple_of(16) {
        bytes.push(0);
    }
    let at = bytes.len();
    // SAFETY: T is plain data (i32, i64, DsaWindow); the byte view covers the slice.
    let b =
        unsafe { std::slice::from_raw_parts(v.as_ptr().cast::<u8>(), std::mem::size_of_val(v)) };
    bytes.extend_from_slice(b);
    at
}

impl GlmForward {
    /// A forward over `model`'s layers with the KV `kv` (whose layout must match) and `experts`
    /// for the routed experts. Every GPU operation runs on the KV pool's stream.
    pub fn new(
        model: DeviceModel,
        embed: HostEmbedding,
        kv: KvPool,
        experts: Box<dyn ExpertBackend>,
        cfg: ForwardConfig,
    ) -> Result<GlmForward> {
        let shape = &model.shape;
        let layout = KvLayout::new(shape, None);
        let kl = kv.config().layout;
        if kl.kda_layers != layout.kda_layers || kl.dsa_layers != layout.dsa_layers {
            return Err(invalid!("the KV pool is laid out for another model shape"));
        }
        let groups_ok = |g: usize| matches!(g, 1 | 2 | 4);
        if cfg.max_rows == 0
            || cfg.max_verify_rows == 0
            || cfg.max_requests == 0
            || !(1..=64).contains(&cfg.decode_splits)
            || !groups_ok(cfg.decode_head_groups)
            || !groups_ok(cfg.prefill_head_groups)
            || !matches!(cfg.kda_prefill_value_blocks, 1 | 2 | 4)
        {
            return Err(invalid!("bad forward config {cfg:?}"));
        }
        let stream = kv.stream().clone();
        // SAFETY: one-time kernel setup (shared-memory limits).
        device::launched(unsafe { dffi::glm53f_dsa_init() }, "glm53f_dsa_init")?;
        let rows = cfg.max_rows.max(cfg.max_verify_rows);
        let logit_rows = cfg.max_requests.max(cfg.max_verify_rows);
        let s = Scratch::new(
            rows,
            logit_rows,
            cfg.max_requests,
            kv.config().max_pages,
            shape.dsa_layers,
        )?;
        let v = VerifyScratch::new(cfg.max_verify_rows, shape.kda_layers, shape.dsa_layers)?;
        Ok(GlmForward {
            gemm: Gemm::new(&stream, cfg.policy)?,
            sms: device::sm_count()?,
            model: Arc::new(model),
            embed,
            kv,
            experts,
            stream,
            cfg,
            s,
            v,
            pending: None,
            timer: RefCell::new(None),
            tap: None,
            host_ids: Vec::new(),
            host_weights: Vec::new(),
            draft: None,
        })
    }

    pub fn shape(&self) -> &ModelShape {
        &self.model.shape
    }

    pub fn stream(&self) -> &Arc<Stream> {
        &self.stream
    }

    /// Device bytes of the forward's own scratch (not weights, not the KV pool).
    pub fn scratch_bytes(&self) -> usize {
        self.s.bytes() + self.v.bytes()
    }

    /// Replace the expert backend (returns the old one).
    pub fn set_experts(&mut self, experts: Box<dyn ExpertBackend>) -> Box<dyn ExpertBackend> {
        std::mem::replace(&mut self.experts, experts)
    }

    /// Install (or remove) a tap.
    pub fn set_tap(&mut self, tap: Option<Box<TapFn>>) {
        self.tap = tap;
    }

    /// Time every stage of later passes ([`GlmForward::take_times`]).
    pub fn set_timing(&mut self, on: bool) {
        *self.timer.borrow_mut() = if on {
            Some(Timer {
                events: Vec::new(),
                marks: Vec::new(),
            })
        } else {
            None
        };
    }

    /// The stage times of the last pass (timing on).
    pub fn take_times(&mut self) -> Result<StageTimes> {
        match self.timer.borrow_mut().as_mut() {
            Some(t) => t.collect(),
            None => Ok(StageTimes::default()),
        }
    }

    fn mark(&self, layer: usize, stage: &'static str) -> Result<()> {
        if let Some(t) = self.timer.borrow_mut().as_mut() {
            t.mark(&self.stream, layer, stage)?;
        }
        Ok(())
    }

    /// The logits of the last pass's logit rows, f32 `[rows][154,880]` (padding rows included).
    pub fn logits(&self, rows: usize) -> Result<Vec<f32>> {
        self.stream.synchronize()?;
        if rows > self.s.logit_rows {
            return Err(invalid!(
                "{rows} logit rows, the scratch holds {}",
                self.s.logit_rows
            ));
        }
        self.s.logits.download::<f32>(rows * VOCAB)
    }

    /// Device pointer to the last pass's greedy picks (i32 per logit row).
    pub fn device_picks(&self) -> *const i32 {
        self.s.next.ptr(0)
    }

    /// Device pointer to the last pass's logits, f32 `[rows][154,880]` (rows as for
    /// [`GlmForward::logits`]), for a device sampler. Valid until the next pass.
    pub fn device_logits(&self) -> *mut f32 {
        self.s.logits.ptr(0)
    }

    // ---- The drafter -------------------------------------------------------------------------

    /// Attach the DFlash2 drafter (made for this forward's model and stream: [`Dflash::new`]).
    /// From then on every pass through the head captures its taps and every committed row
    /// becomes drafter context. Needs decoder layers `0 ..= 43` and a KV pool with the drafter's
    /// rings (`KvLayout::new(shape, Some(drafter config))`); attach before the slots hold rows.
    pub fn attach_drafter(&mut self, mut d: Dflash) -> Result<()> {
        if !std::ptr::eq(d.head(), self.model.head.lm_head.buf.ptr::<u16>(0)) {
            return Err(invalid!("the drafter reads another model's LM head"));
        }
        if !Arc::ptr_eq(d.stream(), &self.stream) {
            return Err(invalid!(
                "the drafter runs on another stream than the forward"
            ));
        }
        if self.shape().layers < LAYERS_NEEDED {
            return Err(invalid!(
                "the drafter reads the outputs of layers 5 to 42: the forward runs {} layers, it needs {LAYERS_NEEDED}",
                self.shape().layers
            ));
        }
        if self.kv.config().layout.draft_kv_bytes == 0 {
            return Err(invalid!(
                "the KV pool has no drafter rings (lay it out with the drafter's config)"
            ));
        }
        d.alloc_taps(self.s.rows.max(self.v.rows))?;
        self.draft = Some(d);
        Ok(())
    }

    pub fn has_drafter(&self) -> bool {
        self.draft.is_some()
    }

    pub fn drafter(&self) -> Option<&Dflash> {
        self.draft.as_ref()
    }

    /// The drafter's proposals for `reqs` (each slot at its committed length), `block - 1` per
    /// request. Slots and the target's state do not change.
    pub fn draft(&mut self, reqs: &[DraftReq<'_>]) -> Result<Vec<glm53f_dflash::seam::Proposal>> {
        let d = self
            .draft
            .as_mut()
            .ok_or_else(|| invalid!("no drafter attached"))?;
        d.draft(reqs, &self.embed)
    }

    // ---- Public passes ---------------------------------------------------------------------

    /// Append and commit each segment's tokens; returns the greedy pick after each segment's
    /// last token. Segments that fit in one pass together run as one batch; others run one
    /// after another in chunks of `max_rows`.
    pub fn prefill(&mut self, segs: &mut [(&mut GlmKv, &[u32])]) -> Result<Vec<u32>> {
        self.check_idle(segs.iter().map(|(k, _)| &**k))?;
        let total: usize = segs.iter().map(|(_, t)| t.len()).sum();
        if segs.iter().any(|(_, t)| t.is_empty()) {
            return Err(invalid!("an empty prefill segment"));
        }
        if segs.len() <= self.cfg.max_requests && total <= self.cfg.max_rows {
            let tokens: Vec<u32> = segs.iter().flat_map(|(_, t)| t.iter().copied()).collect();
            let rows: Vec<usize> = segs.iter().map(|(_, t)| t.len()).collect();
            let mut kvs: Vec<&mut GlmKv> = segs.iter_mut().map(|(k, _)| &mut **k).collect();
            return self.pass(
                Mode::Prefill,
                &mut kvs,
                &rows,
                Input::Tokens(&tokens),
                0..self.shape().layers,
                true,
                true,
            );
        }
        let mut out = Vec::with_capacity(segs.len());
        for (kv, tokens) in segs.iter_mut() {
            let mut last = 0;
            for chunk in tokens.chunks(self.cfg.max_rows) {
                let is_last = chunk.as_ptr_range().end == tokens.as_ptr_range().end;
                let r = self.pass(
                    Mode::Prefill,
                    &mut [&mut **kv],
                    &[chunk.len()],
                    Input::Tokens(chunk),
                    0..self.shape().layers,
                    true,
                    is_last,
                )?;
                if is_last {
                    last = r[0];
                }
            }
            out.push(last);
        }
        Ok(out)
    }

    /// Append and commit one token per slot; returns the greedy pick after each.
    pub fn decode(&mut self, rows: &mut [(&mut GlmKv, u32)]) -> Result<Vec<u32>> {
        self.check_idle(rows.iter().map(|(k, _)| &**k))?;
        let tokens: Vec<u32> = rows.iter().map(|(_, t)| *t).collect();
        let mut kvs: Vec<&mut GlmKv> = rows.iter_mut().map(|(k, _)| &mut **k).collect();
        let n = vec![1; kvs.len()];
        self.pass(
            Mode::Decode,
            &mut kvs,
            &n,
            Input::Tokens(&tokens),
            0..self.shape().layers,
            true,
            true,
        )
    }

    /// One decode row per slot from token ids already on the device (i32, e.g. the previous
    /// pass's picks): no host round trip for the ids.
    pub fn decode_device(&mut self, kvs: &mut [&mut GlmKv], ids: *const i32) -> Result<Vec<u32>> {
        self.check_idle(kvs.iter().map(|k| &**k))?;
        let n = vec![1; kvs.len()];
        self.pass(
            Mode::Decode,
            kvs,
            &n,
            Input::DeviceIds(ids),
            0..self.shape().layers,
            true,
            true,
        )
    }

    /// Append each window as pending rows; returns every row's greedy pick (row j's pick is the
    /// token after `tokens[j]`).
    pub fn verify(&mut self, windows: &mut [(&mut GlmKv, &[u32])]) -> Result<Vec<Vec<u32>>> {
        self.check_idle(windows.iter().map(|(k, _)| &**k))?;
        let rows: Vec<usize> = windows.iter().map(|(_, t)| t.len()).collect();
        if rows.iter().any(|&r| r == 0 || r > 8) {
            return Err(invalid!("verify windows of 1..=8 rows, got {rows:?}"));
        }
        let tokens: Vec<u32> = windows
            .iter()
            .flat_map(|(_, t)| t.iter().copied())
            .collect();
        let mut kvs: Vec<&mut GlmKv> = windows.iter_mut().map(|(k, _)| &mut **k).collect();
        let flat = self.pass(
            Mode::Verify,
            &mut kvs,
            &rows,
            Input::Tokens(&tokens),
            0..self.shape().layers,
            true,
            true,
        )?;
        let mut out = Vec::with_capacity(rows.len());
        let mut at = 0;
        for r in rows {
            out.push(flat[at..at + r].to_vec());
            at += r;
        }
        Ok(out)
    }

    /// Keep the first `keep[i]` pending rows of `slots[i]` (the slots of the last verify, in
    /// its order).
    pub fn commit(&mut self, slots: &mut [&mut GlmKv], keep: &[usize]) -> Result<()> {
        let pending = self
            .pending
            .take()
            .ok_or_else(|| invalid!("commit without a verify"))?;
        if slots.len() != pending.reqs.len() || keep.len() != slots.len() {
            self.pending = Some(pending);
            return Err(invalid!(
                "commit of {} slots after a verify of {}",
                slots.len(),
                keep.len()
            ));
        }
        for ((kv, req), &k) in slots.iter().zip(&pending.reqs).zip(keep) {
            if kv.slot != req.slot
                || kv.pending != req.rows
                || kv.tokens != req.start
                || k == 0
                || k > req.rows
            {
                return Err(invalid!(
                    "commit of {k} rows does not match the pending verify of slot {}",
                    kv.slot
                ));
            }
        }
        self.mark(usize::MAX, "commit_start")?;
        let meta = self.upload_meta(&pending.reqs, pending.rows, None, Some(keep), &[])?;
        self.gather_batch(&pending.reqs, &meta)?;
        let shape = self.model.shape.clone();
        let st = self.stream.raw();
        let n_req = pending.reqs.len() as i32;
        let pool = self.kv.shared.clone();
        let (sn, cn) = (
            KvLayout::state_elems_per_layer(),
            KvLayout::conv_elems_per_layer(),
        );
        let vr = self.v.rows;
        if shape.kda_layers > 0 {
            // SAFETY (all launches): arenas and saves sized for the shape; offsets from the slots.
            launched(
                unsafe {
                    kffi::glm53f_kda_replay_batch(
                        KDA_HEADS as i32,
                        shape.kda_layers as i32,
                        n_req,
                        meta.cu_rows,
                        meta.keep,
                        pool.state.ptr(0),
                        pool.state.ptr(0),
                        sn as i64,
                        meta.state_off,
                        self.v.k.ptr(0),
                        self.v.v.ptr(0),
                        self.v.g.ptr(0),
                        self.v.b.ptr(0),
                        (vr * KDA_WIDTH) as i64,
                        (vr * KDA_HEADS) as i64,
                        st,
                    )
                },
                "glm53f_kda_replay_batch",
            )?;
            launched(
                unsafe {
                    kffi::glm53f_kda_conv_shift_batch(
                        KDA_QKV as i32,
                        shape.kda_layers as i32,
                        n_req,
                        meta.cu_rows,
                        meta.keep,
                        pool.conv.ptr(0),
                        cn as i64,
                        meta.conv_off,
                        self.v.p.ptr(0),
                        (vr * KDA_P_COLS) as i64,
                        KDA_P_COLS as i64,
                        st,
                    )
                },
                "glm53f_kda_conv_shift_batch",
            )?;
        }
        for (l, d) in shape.dsa_index.iter().enumerate() {
            let Some(j) = *d else { continue };
            let AttnW::Dsa(w) = &self.model.layers[l].attn else {
                unreachable!()
            };
            launched(
                unsafe {
                    dffi::glm53f_dsa_index_tail_commit(
                        self.v.k_raw.ptr::<f32>(j * vr * INDEX_DIM),
                        self.v.gate.ptr::<f32>(j * vr * INDEX_DIM),
                        w.k_norm_w.ptr(0),
                        w.k_norm_b.ptr(0),
                        INDEX_LN_EPS,
                        self.s.batch_tails.byte_ptr(j * pending.reqs.len() * TAIL),
                        meta.windows,
                        n_req,
                        st,
                    )
                },
                "glm53f_dsa_index_tail_commit",
            )?;
        }
        self.scatter_tails(&pending.reqs, &meta)?;
        for (kv, &k) in slots.iter_mut().zip(keep) {
            kv.tokens += k;
            kv.pending = 0;
        }
        self.mark(usize::MAX, "commit")?;
        // The kept rows (the anchor and the accepted drafts) become drafter context.
        if let Some(d) = self.draft.as_mut() {
            let rows: Vec<(usize, usize)> = pending
                .reqs
                .iter()
                .zip(keep)
                .map(|(r, &k)| (r.row0, k))
                .collect();
            d.append(slots, &rows)?;
            self.mark(usize::MAX, "draft_append")?;
        }
        Ok(())
    }

    /// Run layers `layers` over `rows[i]` rows of each slot from given input streams (BF16
    /// `[rows][4][4096]`), without the head; returns the output streams. The slots' states for
    /// the layers run are updated as in a prefill (tests: a layer fed its golden input).
    pub fn run_layers(
        &mut self,
        kvs: &mut [&mut GlmKv],
        rows: &[usize],
        input: &[u16],
        layers: Range<usize>,
    ) -> Result<Vec<u16>> {
        self.check_idle(kvs.iter().map(|k| &**k))?;
        let total: usize = rows.iter().sum();
        if input.len() != total * HC * HIDDEN {
            return Err(invalid!(
                "input streams for {} rows, expected {total}",
                input.len() / (HC * HIDDEN)
            ));
        }
        let mode = if total <= 8 && rows.iter().all(|&r| r == 1) {
            Mode::Decode
        } else {
            Mode::Prefill
        };
        self.pass(mode, kvs, rows, Input::Streams(input), layers, false, false)?;
        self.stream.synchronize()?;
        self.s.tap_streams.download::<u16>(total * HC * HIDDEN)
    }

    fn check_idle<'a>(&self, kvs: impl Iterator<Item = &'a GlmKv>) -> Result<()> {
        if self.pending.is_some() {
            return Err(invalid!("a verify round is waiting for its commit"));
        }
        for kv in kvs {
            if !Arc::ptr_eq(&kv.pool, &self.kv.shared) {
                return Err(invalid!("a slot from another KV pool"));
            }
            if kv.pending != 0 {
                return Err(invalid!("slot {} has pending rows", kv.slot));
            }
        }
        Ok(())
    }

    // ---- The pass ----------------------------------------------------------------------------

    /// Upload one pass's metadata. `keep` (commit) overrides the rows kept per request.
    fn upload_meta(
        &mut self,
        reqs: &[Req],
        rows: usize,
        tokens: Option<&[u32]>,
        keep: Option<&[usize]>,
        logit_rows: &[i32],
    ) -> Result<Meta> {
        let dl = self.model.shape.dsa_layers;
        let n = reqs.len();
        let mut row_req = vec![0i32; rows];
        let mut row_pos = vec![0i32; rows];
        let mut cu = vec![0i32; n + 1];
        let mut kp = vec![0i32; n];
        let mut conv = vec![0i64; n];
        let mut state = vec![0i64; n];
        let mut win = vec![DsaWindow::default(); n];
        let mut tail_idx = vec![0i32; dl * n];
        let slots: Vec<i32> = reqs.iter().map(|r| r.slot as i32).collect();
        for (b, r) in reqs.iter().enumerate() {
            for i in 0..r.rows {
                row_req[r.row0 + i] = b as i32;
                row_pos[r.row0 + i] = (r.start + i) as i32;
            }
            cu[b + 1] = (r.row0 + r.rows) as i32;
            let k = keep.map_or(r.rows, |k| k[b]);
            kp[b] = k as i32;
            conv[b] = r.conv_off as i64;
            state[b] = r.state_off as i64;
            win[b] = DsaWindow {
                first_row: r.row0 as i32,
                rows: r.rows as i32,
                start: r.start as i32,
                accepted: k as i32,
            };
            for j in 0..dl {
                tail_idx[j * n + b] = (r.tail_row + j) as i32;
            }
        }
        let ids: Vec<i32> = tokens.map_or_else(Vec::new, |t| t.iter().map(|&x| x as i32).collect());
        let mut bytes = Vec::with_capacity(4096);
        let o_req = pack(&mut bytes, &row_req);
        let o_pos = pack(&mut bytes, &row_pos);
        let o_ids = pack(&mut bytes, &ids);
        let o_cu = pack(&mut bytes, &cu);
        let o_keep = pack(&mut bytes, &kp);
        let o_conv = pack(&mut bytes, &conv);
        let o_state = pack(&mut bytes, &state);
        let o_win = pack(&mut bytes, &win);
        let o_tail = pack(&mut bytes, &tail_idx);
        let o_slots = pack(&mut bytes, &slots);
        let o_logit = pack(&mut bytes, logit_rows);
        if bytes.len() > self.s.meta.bytes() {
            return Err(invalid!("pass metadata of {} bytes", bytes.len()));
        }
        self.s.meta.upload_bytes_async(&self.stream, 0, &bytes)?;
        let m = &self.s.meta;
        Ok(Meta {
            row_req: m.byte_ptr(o_req).cast(),
            row_pos: m.byte_ptr(o_pos).cast(),
            ids: m.byte_ptr(o_ids).cast(),
            cu_rows: m.byte_ptr(o_cu).cast(),
            keep: m.byte_ptr(o_keep).cast(),
            conv_off: m.byte_ptr(o_conv).cast(),
            state_off: m.byte_ptr(o_state).cast(),
            windows: m.byte_ptr(o_win).cast(),
            tail_idx: m.byte_ptr(o_tail).cast(),
            slots: m.byte_ptr(o_slots).cast(),
            logit_rows: m.byte_ptr(o_logit).cast(),
        })
    }

    /// The batch's page tables and tails, gathered from the pool.
    fn gather_batch(&mut self, reqs: &[Req], meta: &Meta) -> Result<()> {
        let pool = &self.kv.shared;
        let mp = pool.cfg.max_pages;
        let st = self.stream.raw();
        // SAFETY: rows of max_pages i32 in both tables; slot indices < max_slots.
        launched(
            unsafe {
                ffi::glm53f_fwd_gather_rows(
                    pool.table.ptr(0),
                    (mp * 4) as i64,
                    meta.slots,
                    self.s.batch_table.ptr(0),
                    (mp * 4) as i64,
                    reqs.len() as i32,
                    (mp * 4) as i64,
                    st,
                )
            },
            "gather page tables",
        )?;
        let dl = self.model.shape.dsa_layers;
        if dl > 0 {
            // SAFETY: tail rows of 1,552 bytes; indices < max_slots x dsa_layers.
            launched(
                unsafe {
                    ffi::glm53f_fwd_gather_rows(
                        pool.tails.ptr(0),
                        TAIL as i64,
                        meta.tail_idx,
                        self.s.batch_tails.ptr(0),
                        TAIL as i64,
                        (dl * reqs.len()) as i32,
                        TAIL as i64,
                        st,
                    )
                },
                "gather tails",
            )?;
        }
        Ok(())
    }

    fn scatter_tails(&mut self, reqs: &[Req], meta: &Meta) -> Result<()> {
        let dl = self.model.shape.dsa_layers;
        if dl == 0 {
            return Ok(());
        }
        let pool = &self.kv.shared;
        // SAFETY: as in gather_batch.
        launched(
            unsafe {
                ffi::glm53f_fwd_scatter_rows(
                    self.s.batch_tails.ptr(0),
                    TAIL as i64,
                    meta.tail_idx,
                    pool.tails.ptr(0),
                    TAIL as i64,
                    (dl * reqs.len()) as i32,
                    TAIL as i64,
                    self.stream.raw(),
                )
            },
            "scatter tails",
        )
    }

    fn dsa_cache(&self, j: usize) -> DsaCache {
        let pool = &self.kv.shared;
        DsaCache {
            base: pool.pages.byte_ptr(j * LAYER_PAGE_BYTES),
            page_stride: pool.cfg.layout.page_bytes as i64,
            page_tables: self.s.batch_table.ptr(0),
            max_pages: pool.cfg.max_pages as i32,
            n_pages: pool.cfg.pages as i32,
        }
    }

    /// One pass over `layers` for `rows[i]` rows of `kvs[i]`. Returns the greedy picks of the
    /// logit rows (every row for decode and verify, each request's last row for prefill) when
    /// `head` and `logits`.
    #[allow(clippy::too_many_arguments)]
    fn pass(
        &mut self,
        mode: Mode,
        kvs: &mut [&mut GlmKv],
        rows: &[usize],
        input: Input<'_>,
        layers: Range<usize>,
        head: bool,
        logits: bool,
    ) -> Result<Vec<u32>> {
        let total: usize = rows.iter().sum();
        let cap = if mode == Mode::Verify {
            self.v.rows
        } else {
            self.s.rows
        };
        if kvs.is_empty() || kvs.len() > self.cfg.max_requests || total > cap || total == 0 {
            return Err(invalid!(
                "{} requests, {total} rows (capacity {cap})",
                kvs.len()
            ));
        }
        if layers.end > self.shape().layers || layers.start >= layers.end {
            return Err(invalid!(
                "layers {layers:?} of a {}-layer forward",
                self.shape().layers
            ));
        }
        self.mark(usize::MAX, "start")?;
        // Pages for the new rows (allocation and copy-on-write), then the request list.
        let mut reqs = Vec::with_capacity(kvs.len());
        let mut row0 = 0;
        for (kv, &r) in kvs.iter_mut().zip(rows) {
            kv.prepare_rows(r)?;
            reqs.push(Req {
                slot: kv.slot,
                start: kv.tokens,
                rows: r,
                row0,
                state_off: kv.state_elem_offset(),
                conv_off: kv.conv_elem_offset(),
                tail_row: kv.tail_row(0),
            });
            row0 += r;
        }
        let logit_rows: Vec<i32> = match mode {
            Mode::Prefill => reqs.iter().map(|r| (r.row0 + r.rows - 1) as i32).collect(),
            _ => (0..total as i32).collect(),
        };
        if head && logits && logit_rows.len() > self.s.logit_rows {
            return Err(invalid!(
                "{} logit rows, the scratch holds {}",
                logit_rows.len(),
                self.s.logit_rows
            ));
        }
        let tokens = match &input {
            Input::Tokens(t) => {
                if t.len() != total {
                    return Err(invalid!("{} tokens for {total} rows", t.len()));
                }
                Some(*t)
            }
            _ => None,
        };
        let meta = self.upload_meta(&reqs, total, tokens, None, &logit_rows)?;
        self.gather_batch(&reqs, &meta)?;
        let st = self.stream.clone();
        // Input streams.
        let mut cur = 0;
        match input {
            Input::Tokens(_) => unsafe {
                self.embed.gather(
                    meta.ids,
                    total,
                    self.s.streams[0].ptr(0),
                    core::ptr::null_mut(),
                    &st,
                )
            }?,
            Input::DeviceIds(ids) => unsafe {
                self.embed.gather(
                    ids,
                    total,
                    self.s.streams[0].ptr(0),
                    core::ptr::null_mut(),
                    &st,
                )
            }?,
            Input::Streams(x) => self.s.streams[0].upload_async(&st, 0, x)?,
        }
        self.mark(usize::MAX, "embed")?;
        // Every pass through the head captures the drafter's taps (`crate::draft`).
        let taps = head && self.draft.is_some();
        // prev: the previous sublayer's output to expand (block_out, block_out2, post, comb).
        let mut prev: Option<(*const u16, *const u16, *const f32, *const f32)> = None;
        for l in layers.clone() {
            // Attention boundary.
            let model = self.model.clone();
            let lw = &model.layers[l];
            let expanded = self.boundary(total, &lw.attn_hc, &lw.input_norm, cur, prev, true)?;
            if expanded {
                cur ^= 1;
            }
            if let Some(d) = self.draft.as_ref().filter(|_| taps) {
                // streams[cur]: this layer's input, the previous layer's completed output. At the
                // entry of layers 6, 15, 25, 34 and 43 their mean is a tap (glm53f-dflash
                // README step 1: SGLang captures before layer k + 1, contracted by the mean).
                d.capture(l, total, self.s.streams[cur].ptr(0), &self.stream)?;
            }
            self.mark(l, "attn_hc")?;
            match &lw.attn {
                AttnW::Kda(w) => self.kda(l, mode, total, &reqs, &meta, w)?,
                AttnW::Dsa(w) => self.dsa(l, mode, total, &reqs, &meta, w)?,
            }
            self.tap(l, TapPoint::AttnDone, mode, total, cur)?;
            // FFN boundary, expanding the attention output.
            let a: (*const u16, *const u16, *const f32, *const f32) = (
                self.s.attn_out.ptr(0),
                core::ptr::null(),
                self.s.attn_post.ptr(0),
                self.s.attn_comb.ptr(0),
            );
            self.boundary(total, &lw.ffn_hc, &lw.post_attn_norm, cur, Some(a), false)?;
            cur ^= 1;
            self.mark(l, "ffn_hc")?;
            let two = match &lw.ffn {
                FfnW::Dense(m) => {
                    self.mlp(total, m, self.s.ffn_out.ptr(0))?;
                    self.mark(l, "dense_mlp")?;
                    false
                }
                FfnW::Moe {
                    router,
                    bias,
                    shared,
                } => {
                    self.moe(l, total, router, bias, shared)?;
                    true
                }
            };
            prev = Some((
                self.s.ffn_out.ptr(0),
                if two {
                    self.s.ffn_out2.ptr(0)
                } else {
                    core::ptr::null()
                },
                self.s.ffn_post.ptr(0),
                self.s.ffn_comb.ptr(0),
            ));
            self.tap(l, TapPoint::FfnDone, mode, total, cur)?;
            if self.tap.is_some() {
                self.expand_to_tap(total, cur, prev.unwrap())?;
                self.tap(l, TapPoint::LayerOut, mode, total, cur)?;
            }
        }
        // Tails of prefill and decode passes were committed in the layers: back to the slots.
        if mode != Mode::Verify {
            self.scatter_tails(&reqs, &meta)?;
        }
        let mut picks = Vec::new();
        if head {
            let (bo, bo2, post, comb) = prev.unwrap();
            // SAFETY: scratch buffers sized for `total` rows.
            launched(
                unsafe {
                    lffi::glm53f_hc_head(
                        self.s.streams[cur].ptr(0),
                        bo,
                        bo2,
                        post,
                        comb,
                        self.model.head.norm.ptr(0),
                        self.s.head_out.ptr(0),
                        total as i32,
                        HIDDEN as i32,
                        st.raw().cast(),
                    )
                },
                "glm53f_hc_head",
            )?;
            self.mark(usize::MAX, "head_hc")?;
            if logits {
                let lr = logit_rows.len();
                let x: *const u16 = if lr == total {
                    self.s.head_out.ptr(0)
                } else {
                    // SAFETY: rows of the head output; indices < total.
                    launched(
                        unsafe {
                            ffi::glm53f_fwd_gather_rows(
                                self.s.head_out.ptr(0),
                                (HIDDEN * 2) as i64,
                                meta.logit_rows,
                                self.s.head_sel.ptr(0),
                                (HIDDEN * 2) as i64,
                                lr as i32,
                                (HIDDEN * 2) as i64,
                                st.raw(),
                            )
                        },
                        "gather logit rows",
                    )?;
                    self.s.head_sel.ptr(0)
                };
                let lm = self.model.head.lm_head.mat();
                unsafe {
                    self.gemm.bf16(
                        x,
                        HIDDEN,
                        0,
                        &lm,
                        lr,
                        self.s.logits.ptr::<c_void>(0),
                        VOCAB,
                        0,
                        true,
                        &st,
                    )
                }?;
                self.mark(usize::MAX, "lm_head")?;
                // SAFETY: logits [lr][VOCAB]; next holds lr ids.
                launched(
                    unsafe {
                        ffi::glm53f_fwd_argmax(
                            self.s.logits.ptr(0),
                            VOCAB as i64,
                            lr as i32,
                            SAMPLE_VOCAB as i32,
                            self.s.next.ptr(0),
                            core::ptr::null_mut(),
                            st.raw(),
                        )
                    },
                    "glm53f_fwd_argmax",
                )?;
                let mut ids = vec![0i32; lr];
                // SAFETY: a host vector of lr ids.
                let b = unsafe {
                    std::slice::from_raw_parts_mut(ids.as_mut_ptr().cast::<u8>(), lr * 4)
                };
                self.s.next.download_bytes(&st, 0, b)?;
                picks = ids.into_iter().map(|x| x as u32).collect();
                self.mark(usize::MAX, "argmax")?;
            }
        } else if !head && self.tap.is_none() {
            // The last layer's output streams, for run_layers.
            self.expand_to_tap(total, cur, prev.unwrap())?;
        }
        // Commit the positions.
        for (kv, &r) in kvs.iter_mut().zip(rows) {
            match mode {
                Mode::Verify => kv.pending = r,
                _ => kv.tokens += r,
            }
        }
        // Committed rows become drafter context (a verify window's rows wait for its commit).
        if taps && mode != Mode::Verify {
            let rows: Vec<(usize, usize)> = reqs.iter().map(|r| (r.row0, r.rows)).collect();
            if let Some(d) = self.draft.as_mut() {
                d.append(kvs, &rows)?;
            }
            self.mark(usize::MAX, "draft_append")?;
        }
        if mode == Mode::Verify {
            self.pending = Some(Pending { reqs, rows: total });
        }
        Ok(picks)
    }

    /// Materialize `prev`'s expansion of streams `cur` into the tap buffer.
    fn expand_to_tap(
        &mut self,
        rows: usize,
        cur: usize,
        prev: (*const u16, *const u16, *const f32, *const f32),
    ) -> Result<()> {
        let (bo, bo2, post, comb) = prev;
        // SAFETY: scratch sized for `rows`; expansion only (no projection).
        launched(
            unsafe {
                lffi::glm53f_hc_project(
                    self.s.streams[cur].ptr(0),
                    bo,
                    bo2,
                    post,
                    comb,
                    self.s.tap_streams.ptr(0),
                    core::ptr::null(),
                    core::ptr::null_mut(),
                    rows as i32,
                    HIDDEN as i32,
                    self.stream.raw().cast(),
                )
            },
            "glm53f_hc_project (expand)",
        )
    }

    fn tap(
        &mut self,
        layer: usize,
        point: TapPoint,
        mode: Mode,
        rows: usize,
        cur: usize,
    ) -> Result<()> {
        if self.tap.is_none() {
            return Ok(());
        }
        if point != TapPoint::LayerOut {
            // The streams the boundary read.
            self.s.tap_streams.copy_from(
                &self.stream,
                0,
                &self.s.streams[cur],
                0,
                rows * HC * HIDDEN * 2,
            )?;
        }
        self.stream.synchronize()?;
        let mut f = self.tap.take().unwrap();
        let r = f(&Tap {
            layer,
            point,
            mode,
            rows,
            s: &self.s,
        });
        self.tap = Some(f);
        r
    }

    /// One mHC boundary: optionally expand `prev` into the other stream buffer first, then the
    /// projection, weights, collapse, the sublayer's RMSNorm and its E4M3 form. Writes this
    /// site's `post`/`comb` (`attn` selects the site's buffers). Returns whether it expanded.
    fn boundary(
        &mut self,
        rows: usize,
        hc: &HcW,
        norm: &DeviceBuffer,
        cur: usize,
        prev: Option<(*const u16, *const u16, *const f32, *const f32)>,
        attn: bool,
    ) -> Result<bool> {
        let s = &self.s;
        let (post, comb) = if attn {
            (&s.attn_post, &s.attn_comb)
        } else {
            (&s.ffn_post, &s.ffn_comb)
        };
        let sin = s.streams[cur].ptr::<u16>(0);
        let (bo, bo2, pin, cin, sout) = match prev {
            Some((a, b, p, c)) => (a, b, p, c, s.streams[cur ^ 1].ptr::<u16>(0)),
            None => (
                core::ptr::null(),
                core::ptr::null(),
                core::ptr::null(),
                core::ptr::null(),
                core::ptr::null_mut(),
            ),
        };
        let st = self.stream.raw().cast();
        if rows <= 8 {
            // SAFETY: scratch sized for `rows`; `sync` zeroed.
            launched(
                unsafe {
                    lffi::glm53f_hc_boundary_decode(
                        sin,
                        bo,
                        bo2,
                        pin,
                        cin,
                        sout,
                        hc.fn_.ptr(0),
                        hc.base.ptr(0),
                        hc.scale.ptr(0),
                        norm.ptr(0),
                        s.partials.ptr(0),
                        s.sync.ptr(0),
                        s.pre.ptr(0),
                        post.ptr(0),
                        comb.ptr(0),
                        s.collapsed.ptr(0),
                        s.normed.ptr(0),
                        s.normed_q.ptr(0),
                        s.normed_s.ptr(0),
                        rows as i32,
                        HIDDEN as i32,
                        st,
                    )
                },
                "glm53f_hc_boundary_decode",
            )?;
        } else {
            let projected = if prev.is_some() {
                sout as *const u16
            } else {
                sin
            };
            // SAFETY: as above.
            launched(
                unsafe {
                    lffi::glm53f_hc_project(
                        sin,
                        bo,
                        bo2,
                        pin,
                        cin,
                        sout,
                        hc.fn_.ptr(0),
                        s.partials.ptr(0),
                        rows as i32,
                        HIDDEN as i32,
                        st,
                    )
                },
                "glm53f_hc_project",
            )?;
            launched(
                unsafe {
                    lffi::glm53f_hc_finish(
                        s.partials.ptr(0),
                        hc.base.ptr(0),
                        hc.scale.ptr(0),
                        projected,
                        norm.ptr(0),
                        s.pre.ptr(0),
                        post.ptr(0),
                        comb.ptr(0),
                        s.collapsed.ptr(0),
                        s.normed.ptr(0),
                        s.normed_q.ptr(0),
                        s.normed_s.ptr(0),
                        rows as i32,
                        HIDDEN as i32,
                        st,
                    )
                },
                "glm53f_hc_finish",
            )?;
        }
        Ok(prev.is_some())
    }

    fn normed_input(&self) -> Fp8Input {
        Fp8Input {
            bf16: self.s.normed.ptr(0),
            q: self.s.normed_q.ptr(0),
            scales: self.s.normed_s.ptr(0),
        }
    }

    // ---- Attention ---------------------------------------------------------------------------

    #[allow(clippy::too_many_arguments)]
    fn kda(
        &mut self,
        l: usize,
        mode: Mode,
        rows: usize,
        reqs: &[Req],
        meta: &Meta,
        w: &KdaW,
    ) -> Result<()> {
        let j = self.model.shape.kda_index[l].unwrap();
        let st = self.stream.clone();
        let verify = mode == Mode::Verify;
        let vr = self.v.rows;
        // Projection rows: per layer for a verify round (the commit's conv shift reads them).
        let p: *mut u16 = if verify {
            self.v.p.ptr(j * vr * KDA_P_COLS)
        } else {
            self.s.p.ptr(0)
        };
        let x: *const u16 = self.s.normed.ptr(0);
        unsafe {
            self.gemm.bf16(
                x,
                HIDDEN,
                0,
                &w.qkvb.mat(),
                rows,
                p.cast(),
                KDA_P_COLS,
                0,
                false,
                &st,
            )
        }?;
        unsafe {
            self.gemm.bf16(
                x,
                HIDDEN,
                0,
                &w.fga.mat(),
                rows,
                self.s.fga.ptr::<c_void>(0),
                2 * KDA_DIM,
                0,
                false,
                &st,
            )
        }?;
        unsafe {
            self.gemm.bf16(
                self.s.fga.ptr(0),
                2 * KDA_DIM,
                KDA_DIM,
                &w.fgb.mat(),
                rows,
                self.s.ag.ptr::<c_void>(0),
                2 * KDA_WIDTH,
                KDA_WIDTH,
                false,
                &st,
            )
        }?;
        if verify && self.tap.is_some() {
            // Taps read the projection rows from the scratch buffer.
            self.s.p.copy_from(
                &st,
                0,
                &self.v.p,
                j * vr * KDA_P_COLS * 2,
                rows * KDA_P_COLS * 2,
            )?;
        }
        self.mark(l, "kda_proj")?;
        let pool = self.kv.shared.clone();
        let (sn, cn) = (
            KvLayout::state_elems_per_layer(),
            KvLayout::conv_elems_per_layer(),
        );
        let state: *mut f32 = pool.state.ptr(j * sn);
        let conv: *mut u16 = pool.conv.ptr(j * cn);
        let (ks, vs, gs, bs): (*mut f32, *mut u16, *mut f32, *mut f32) = if verify {
            (
                self.v.k.ptr(j * vr * KDA_WIDTH),
                self.v.v.ptr(j * vr * KDA_WIDTH),
                self.v.g.ptr(j * vr * KDA_WIDTH),
                self.v.b.ptr(j * vr * KDA_HEADS),
            )
        } else {
            (
                core::ptr::null_mut(),
                core::ptr::null_mut(),
                core::ptr::null_mut(),
                core::ptr::null_mut(),
            )
        };
        let ag: *const u16 = self.s.ag.ptr(0);
        if mode == Mode::Prefill && self.cfg.kda_chunked_prefill && rows > 8 {
            // The chunked form: it advances the conv windows itself.
            let n = reqs.len() as i32;
            let per = self.cfg.kda_prefill_rows.max(16) as i32;
            // SAFETY: a host function.
            let need = unsafe { kffi::glm53f_kda_prefill_workspace_bytes(KDA_HEADS as i32, n, per) }
                as usize;
            if need > self.s.kda_ws.bytes() {
                self.stream.synchronize()?;
                self.s.kda_ws = DeviceBuffer::alloc(need)?;
            }
            let max_rows = reqs.iter().map(|r| r.rows).max().unwrap_or(0) as i32;
            // SAFETY: as for the chain below; the workspace was sized above.
            launched(
                unsafe {
                    kffi::glm53f_kda_prefill_batch(
                        KDA_HEADS as i32,
                        n,
                        meta.cu_rows,
                        max_rows,
                        p,
                        KDA_P_COLS as i64,
                        KDA_QKV as i64,
                        ag,
                        (2 * KDA_WIDTH) as i64,
                        ag.wrapping_add(KDA_WIDTH),
                        (2 * KDA_WIDTH) as i64,
                        conv,
                        meta.conv_off,
                        w.conv_w.ptr(0),
                        state,
                        state,
                        meta.state_off,
                        w.a_log.ptr(0),
                        w.dt_bias.ptr(0),
                        w.o_norm.ptr(0),
                        RMS_EPS,
                        KDA_LOWER,
                        self.s.kda_out.ptr(0),
                        KDA_WIDTH as i64,
                        self.cfg.kda_prefill_value_blocks,
                        self.s.kda_ws.ptr(0),
                        self.s.kda_ws.bytes() as i64,
                        st.raw(),
                    )
                },
                "glm53f_kda_prefill_batch",
            )?;
            self.mark(l, "kda_core")?;
            unsafe {
                self.gemm.bf16(
                    self.s.kda_out.ptr(0),
                    KDA_WIDTH,
                    0,
                    &w.o.mat(),
                    rows,
                    self.s.attn_out.ptr::<c_void>(0),
                    HIDDEN,
                    0,
                    false,
                    &st,
                )
            }?;
            self.mark(l, "kda_o")?;
            return Ok(());
        }
        // SAFETY: state and conv arenas hold every slot of the batch at its offsets; scratch
        // sized for `rows`; the saves for up to the verify capacity.
        launched(
            unsafe {
                kffi::glm53f_kda_chain_batch(
                    KDA_HEADS as i32,
                    reqs.len() as i32,
                    meta.cu_rows,
                    p,
                    KDA_P_COLS as i64,
                    KDA_QKV as i64,
                    ag,
                    (2 * KDA_WIDTH) as i64,
                    ag.wrapping_add(KDA_WIDTH),
                    (2 * KDA_WIDTH) as i64,
                    conv,
                    meta.conv_off,
                    w.conv_w.ptr(0),
                    state,
                    if verify { core::ptr::null_mut() } else { state },
                    meta.state_off,
                    w.a_log.ptr(0),
                    w.dt_bias.ptr(0),
                    w.o_norm.ptr(0),
                    RMS_EPS,
                    KDA_LOWER,
                    self.s.kda_out.ptr(0),
                    KDA_WIDTH as i64,
                    ks,
                    vs,
                    gs,
                    bs,
                    st.raw(),
                )
            },
            "glm53f_kda_chain_batch",
        )?;
        if !verify {
            // SAFETY: as above; keep = each request's rows.
            launched(
                unsafe {
                    kffi::glm53f_kda_conv_shift_batch(
                        KDA_QKV as i32,
                        1,
                        reqs.len() as i32,
                        meta.cu_rows,
                        meta.keep,
                        conv,
                        0,
                        meta.conv_off,
                        p,
                        0,
                        KDA_P_COLS as i64,
                        st.raw(),
                    )
                },
                "glm53f_kda_conv_shift_batch",
            )?;
        }
        self.mark(l, "kda_core")?;
        unsafe {
            self.gemm.bf16(
                self.s.kda_out.ptr(0),
                KDA_WIDTH,
                0,
                &w.o.mat(),
                rows,
                self.s.attn_out.ptr::<c_void>(0),
                HIDDEN,
                0,
                false,
                &st,
            )
        }?;
        self.mark(l, "kda_o")?;
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn dsa(
        &mut self,
        l: usize,
        mode: Mode,
        rows: usize,
        reqs: &[Req],
        meta: &Meta,
        w: &DsaW,
    ) -> Result<()> {
        let j = self.model.shape.dsa_index[l].unwrap();
        let st = self.stream.clone();
        let raw = st.raw();
        let s = &self.s;
        let quant = self.gemm.policy.fp8_needs_quant(rows);
        let x = self.normed_input();
        // Query and latent projections (FP8), the query latent's RMSNorm.
        unsafe { self.gemm.fp8(&x, &w.q_a.mat(), rows, s.qa.ptr(0), &st) }?;
        unsafe { self.gemm.fp8(&x, &w.kv_a.mat(), rows, s.kva.ptr(0), &st) }?;
        // SAFETY (all launches in this function): scratch sized for `rows`; weights of this
        // layer; the cache view covers the batch's page tables.
        launched(
            unsafe {
                lffi::glm53f_rmsnorm(
                    s.qa.ptr(0),
                    w.q_a_norm.ptr(0),
                    s.q_resid.ptr(0),
                    rows as i32,
                    Q_LORA as i32,
                    raw.cast(),
                )
            },
            "glm53f_rmsnorm",
        )?;
        if quant {
            unsafe {
                act_quant(
                    s.q_resid.ptr(0),
                    s.q_resid_q.ptr(0),
                    s.q_resid_s.ptr(0),
                    rows,
                    Q_LORA,
                    &st,
                )
            }?;
        }
        let qr = Fp8Input {
            bf16: s.q_resid.ptr(0),
            q: s.q_resid_q.ptr(0),
            scales: s.q_resid_s.ptr(0),
        };
        unsafe { self.gemm.fp8(&qr, &w.q_b.mat(), rows, s.q16.ptr(0), &st) }?;
        // Indexer projections (BF16).
        unsafe {
            self.gemm.bf16(
                s.q_resid.ptr(0),
                Q_LORA,
                0,
                &w.idx_wq_b.mat(),
                rows,
                s.idx_q16.ptr::<c_void>(0),
                INDEX_HEADS * INDEX_DIM,
                0,
                false,
                &st,
            )
        }?;
        unsafe {
            self.gemm.bf16(
                s.normed.ptr(0),
                HIDDEN,
                0,
                &w.idx_proj.mat(),
                rows,
                s.idx_p.ptr::<c_void>(0),
                IDX_PROJ_COLS,
                0,
                false,
                &st,
            )
        }?;
        self.mark(l, "dsa_proj")?;
        // f32 views the DSA kernels take.
        let widen = |src: *const u16,
                     ldi: usize,
                     dst: *mut f32,
                     ldo: usize,
                     cols: usize,
                     scale: f32|
         -> Result<()> {
            launched(
                unsafe {
                    ffi::glm53f_fwd_bf16_to_f32(
                        src,
                        ldi as i64,
                        dst,
                        ldo as i64,
                        rows as i32,
                        cols as i32,
                        scale,
                        raw,
                    )
                },
                "glm53f_fwd_bf16_to_f32",
            )
        };
        let verify = mode == Mode::Verify;
        let vr = self.v.rows;
        let (k_raw, gate): (*mut f32, *mut f32) = if verify {
            (
                self.v.k_raw.ptr(j * vr * INDEX_DIM),
                self.v.gate.ptr(j * vr * INDEX_DIM),
            )
        } else {
            (s.k_raw.ptr(0), s.gate.ptr(0))
        };
        widen(
            s.q16.ptr(0),
            MLA_HEADS * QK_HEAD,
            s.q32.ptr(0),
            MLA_HEADS * QK_HEAD,
            MLA_HEADS * QK_HEAD,
            1.0,
        )?;
        widen(
            s.idx_q16.ptr(0),
            INDEX_HEADS * INDEX_DIM,
            s.idx_q32.ptr(0),
            INDEX_HEADS * INDEX_DIM,
            INDEX_HEADS * INDEX_DIM,
            1.0,
        )?;
        widen(s.kva.ptr(0), KV_LORA, s.kva32.ptr(0), KV_LORA, KV_LORA, 1.0)?;
        let ip: *const u16 = s.idx_p.ptr(0);
        widen(ip, IDX_PROJ_COLS, k_raw, INDEX_DIM, INDEX_DIM, 1.0)?;
        widen(
            ip.wrapping_add(INDEX_DIM),
            IDX_PROJ_COLS,
            gate,
            INDEX_DIM,
            INDEX_DIM,
            1.0,
        )?;
        let wscale = (INDEX_HEADS as f64).powf(-0.5) as f32;
        widen(
            ip.wrapping_add(2 * INDEX_DIM),
            IDX_PROJ_COLS,
            s.w32.ptr(0),
            INDEX_HEADS,
            INDEX_HEADS,
            wscale,
        )?;
        let cache = self.dsa_cache(j);
        let tails: *mut u8 = s.batch_tails.byte_ptr(j * reqs.len() * TAIL);
        launched(
            unsafe {
                dffi::glm53f_dsa_mla_latent_write(
                    s.kva32.ptr(0),
                    w.kv_a_norm.ptr(0),
                    RMS_EPS,
                    meta.row_pos,
                    meta.row_req,
                    rows as i32,
                    cache,
                    raw,
                )
            },
            "glm53f_dsa_mla_latent_write",
        )?;
        launched(
            unsafe {
                dffi::glm53f_dsa_index_pool_write(
                    k_raw,
                    gate,
                    w.k_norm_w.ptr(0),
                    w.k_norm_b.ptr(0),
                    INDEX_LN_EPS,
                    w.ape.ptr(0),
                    tails,
                    meta.windows,
                    meta.row_req,
                    rows as i32,
                    cache,
                    raw,
                )
            },
            "glm53f_dsa_index_pool_write",
        )?;
        if !verify {
            launched(
                unsafe {
                    dffi::glm53f_dsa_index_tail_commit(
                        k_raw,
                        gate,
                        w.k_norm_w.ptr(0),
                        w.k_norm_b.ptr(0),
                        INDEX_LN_EPS,
                        tails,
                        meta.windows,
                        reqs.len() as i32,
                        raw,
                    )
                },
                "glm53f_dsa_index_tail_commit",
            )?;
        }
        self.mark(l, "dsa_cache")?;
        // Selection.
        let max_pos = reqs.iter().map(|r| r.start + r.rows - 1).max().unwrap();
        let max_pools = ((max_pos + 1) / 4).max(1) as i32;
        let (mut chunk_pools, mut chunks) = (0i32, 0i32);
        // SAFETY: a host function.
        unsafe {
            dffi::glm53f_dsa_index_plan(
                rows as i32,
                max_pools,
                self.sms,
                &mut chunk_pools,
                &mut chunks,
            )
        };
        // SAFETY: a host function.
        let need = unsafe { dffi::glm53f_dsa_index_workspace_bytes(rows as i32, chunks) } as usize;
        if need > self.s.idx_ws.bytes() {
            self.stream.synchronize()?;
            self.s.idx_ws = DeviceBuffer::alloc(need)?;
        }
        let small = rows <= 8;
        let (splits, groups) = if small {
            (self.cfg.decode_splits, self.cfg.decode_head_groups)
        } else {
            (1, self.cfg.prefill_head_groups)
        };
        // SAFETY: a host function.
        let mla_need =
            unsafe { dffi::glm53f_dsa_mla_workspace_bytes(rows as i32, splits as i32) } as usize;
        if mla_need > self.s.mla_ws.bytes() {
            self.stream.synchronize()?;
            self.s.mla_ws = DeviceBuffer::alloc(mla_need)?;
        }
        let s = &self.s;
        launched(
            unsafe {
                dffi::glm53f_dsa_index_select(
                    s.idx_q32.ptr(0),
                    s.w32.ptr(0),
                    (INDEX_DIM as f64).powf(-0.5) as f32,
                    meta.row_pos,
                    meta.row_req,
                    rows as i32,
                    max_pools,
                    cache,
                    chunk_pools,
                    chunks,
                    s.idx_ws.ptr(0),
                    s.idx_ws.bytes() as u64,
                    s.pools.ptr(0),
                    s.tokens.ptr(0),
                    s.counts.ptr(0),
                    core::ptr::null_mut(),
                    raw,
                )
            },
            "glm53f_dsa_index_select",
        )?;
        self.mark(l, "dsa_index")?;
        // Sparse MLA in latent space.
        launched(
            unsafe {
                dffi::glm53f_dsa_mla_absorb_q(
                    s.q32.ptr(0),
                    w.kv_b.ptr(0),
                    rows as i32,
                    s.q_abs.ptr(0),
                    core::ptr::null_mut(),
                    raw,
                )
            },
            "glm53f_dsa_mla_absorb_q",
        )?;
        launched(
            unsafe {
                dffi::glm53f_dsa_mla_sparse_attn(
                    s.q_abs.ptr(0),
                    s.tokens.ptr(0),
                    MAX_SELECTED as i32,
                    s.counts.ptr(0),
                    meta.row_req,
                    rows as i32,
                    MLA_SCALE,
                    cache,
                    splits as i32,
                    groups as i32,
                    s.mla_ws.ptr(0),
                    s.mla_ws.bytes() as u64,
                    s.o_lat.ptr(0),
                    s.lse.ptr(0),
                    raw,
                )
            },
            "glm53f_dsa_mla_sparse_attn",
        )?;
        launched(
            unsafe {
                dffi::glm53f_dsa_mla_unabsorb_v(
                    s.o_lat.ptr(0),
                    w.kv_b.ptr(0),
                    rows as i32,
                    s.o32.ptr(0),
                    raw,
                )
            },
            "glm53f_dsa_mla_unabsorb_v",
        )?;
        launched(
            unsafe {
                ffi::glm53f_fwd_f32_to_bf16(
                    s.o32.ptr(0),
                    (MLA_HEADS * V_HEAD) as i64,
                    s.o16.ptr(0),
                    (MLA_HEADS * V_HEAD) as i64,
                    rows as i32,
                    (MLA_HEADS * V_HEAD) as i32,
                    raw,
                )
            },
            "glm53f_fwd_f32_to_bf16",
        )?;
        if quant {
            unsafe {
                act_quant(
                    s.o16.ptr(0),
                    s.o_q.ptr(0),
                    s.o_s.ptr(0),
                    rows,
                    MLA_HEADS * V_HEAD,
                    &st,
                )
            }?;
        }
        self.mark(l, "dsa_attn")?;
        let o = Fp8Input {
            bf16: s.o16.ptr(0),
            q: s.o_q.ptr(0),
            scales: s.o_s.ptr(0),
        };
        unsafe { self.gemm.fp8(&o, &w.o.mat(), rows, s.attn_out.ptr(0), &st) }?;
        self.mark(l, "dsa_o")?;
        Ok(())
    }

    // ---- FFN ---------------------------------------------------------------------------------

    /// A SwiGLU MLP of the normed rows into `out`.
    fn mlp(&mut self, rows: usize, m: &MlpW, out: *mut u16) -> Result<()> {
        let st = self.stream.clone();
        let quant = self.gemm.policy.fp8_needs_quant(rows);
        let s = &self.s;
        unsafe {
            self.gemm.fp8(
                &self.normed_input(),
                &m.gate_up.mat(),
                rows,
                s.gu.ptr(0),
                &st,
            )
        }?;
        // SAFETY: gate_up [rows][2 inter]; act and its E4M3 form sized for the dense width.
        launched(
            unsafe {
                lffi::glm53f_swiglu(
                    s.gu.ptr(0),
                    s.act.ptr(0),
                    if quant {
                        s.act_q.ptr(0)
                    } else {
                        core::ptr::null_mut()
                    },
                    if quant {
                        s.act_s.ptr(0)
                    } else {
                        core::ptr::null_mut()
                    },
                    rows as i32,
                    m.inter as i32,
                    st.raw().cast(),
                )
            },
            "glm53f_swiglu",
        )?;
        let a = Fp8Input {
            bf16: s.act.ptr(0),
            q: s.act_q.ptr(0),
            scales: s.act_s.ptr(0),
        };
        unsafe { self.gemm.fp8(&a, &m.down.mat(), rows, out, &st) }
    }

    fn moe(
        &mut self,
        l: usize,
        rows: usize,
        router: &DeviceBuffer,
        bias: &DeviceBuffer,
        shared: &MlpW,
    ) -> Result<()> {
        let st = self.stream.clone();
        let s = &self.s;
        // SAFETY: normed [rows][4096]; router weight [288][4096]; outputs sized for `rows`.
        launched(
            unsafe {
                lffi::glm53f_router_fused(
                    s.normed.ptr(0),
                    router.ptr(0),
                    bias.ptr(0),
                    s.router_logits.ptr(0),
                    s.sync.ptr(0),
                    s.ids.ptr(0),
                    s.weights.ptr(0),
                    rows as i32,
                    EXPERTS as i32,
                    HIDDEN as i32,
                    TOP_K as i32,
                    ROUTED_SCALE,
                    st.raw().cast(),
                )
            },
            "glm53f_router_fused",
        )?;
        // The routes to the host: the step's one host round trip.
        let (mut h_ids, mut h_w) = (
            std::mem::take(&mut self.host_ids),
            std::mem::take(&mut self.host_weights),
        );
        h_ids.resize(rows * TOP_K, 0);
        h_w.resize(rows * TOP_K, 0.0);
        {
            // SAFETY: host vectors of rows x 8 values.
            let bi = unsafe {
                std::slice::from_raw_parts_mut(h_ids.as_mut_ptr().cast::<u8>(), rows * TOP_K * 4)
            };
            s.ids.download_bytes(&st, 0, bi)?;
            let bw = unsafe {
                std::slice::from_raw_parts_mut(h_w.as_mut_ptr().cast::<u8>(), rows * TOP_K * 4)
            };
            s.weights.download_bytes(&st, 0, bw)?;
        }
        let (x, x_q, x_scales): (*const u16, *const u8, *const f32) =
            (s.normed.ptr(0), s.normed_q.ptr(0), s.normed_s.ptr(0));
        let (ids, weights): (*const i32, *const f32) = (s.ids.ptr(0), s.weights.ptr(0));
        let out: *mut u16 = s.ffn_out.ptr(0);
        let shared_out: *mut u16 = s.ffn_out2.ptr(0);
        let call = |host_ids, host_weights| ExpertCall {
            layer: l,
            rows,
            x,
            x_q,
            x_scales,
            ids,
            weights,
            host_ids,
            host_weights,
            out,
        };
        let r = self
            .mark(l, "router")
            .and_then(|_| self.experts.submit(&call(&h_ids, &h_w), &st))
            .and_then(|_| self.mark(l, "routed"))
            // The shared expert overlaps a remote backend's exchange.
            .and_then(|_| self.mlp(rows, shared, shared_out))
            .and_then(|_| self.mark(l, "shared"))
            .and_then(|_| self.experts.finish(&call(&h_ids, &h_w), &st))
            .and_then(|_| self.mark(l, "routed_wait"));
        self.host_ids = h_ids;
        self.host_weights = h_w;
        r
    }
}
