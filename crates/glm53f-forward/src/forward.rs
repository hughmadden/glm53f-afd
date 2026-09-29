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
//!
//! # Lanes (prefill)
//!
//! A layer's routed experts run on the expert ranks while the coordinator waits, so a serial
//! prefill leaves one side idle at every MoE layer. With [`ForwardConfig::lanes`] N of 2 to
//! [`MAX_LANES`], a prefill pass cuts its rows into lanes (a request that straddles a cut is
//! split: the next lane continues it at the positions after the previous lane's part, and may hand
//! it on again) and runs them through each layer in turn, in round-robin order on the one stream
//! (lanes A and B are lanes 0 and 1):
//!
//! ```text
//! layer L:  lane 0: attention, router, submit its experts, shared expert
//!           lane 1: attention, router, submit its experts, shared expert  <- ranks: lane 0's
//!           ...
//!           lane N-1: attention, ...                                      <- ranks: lanes 0, 1..
//!           lane 0: finish its experts (L); attention (L + 1) ...
//! ```
//!
//! **The cut.** A pass of R rows runs in one lane per [`ForwardConfig::min_lane_rows`] rows, at
//! most N, and in at least as many as it needs to fit (lane 0 holds a one-lane pass, every lane
//! `max_rows / N` rows). In k lanes, lane i holds the pass's rows `ceil(i R / k) .. ceil((i + 1) R
//! / k)`: the lanes differ by a row at most (lane 0 is never the smaller). With N = 2 this is the
//! rule of the two lanes before: two from twice `min_lane_rows` rows (or more than one lane
//! holds), cut in the middle.
//!
//! Lane i's attention at layer L needs only what the lanes before it left at L: the KDA states
//! and conv windows (in the pool, updated in place), the MLA latents and pooled keys of their rows
//! (in the pages), and a split request's DSA tail (copied from lane i - 1's batch tails before
//! each DSA layer: a request across several lanes goes from one to the next). Within a request, a
//! lane's rows go through each layer before the next lane's, the causal order.
//!
//! **Calls in flight.** At most [`ExpertBackend::depth`] calls are out, the oldest finished
//! first: a lane's attention at the next layer waits for its own call, and a lane's submit for a
//! free place. Over RDMA the ranks queue the requests (four by default, one per receive slot they
//! post) and compute them in order; over TCP one call is out, so each lane's call is collected
//! before the next lane's is sent (the next lane's attention still overlaps it). The GPU keeps
//! busy while a lane's exchange takes at most the other N - 1 lanes' attention.
//!
//! Each lane has its own buffers (its `Scratch`, swapped in as the active one); the attention
//! workspaces are shared, since the lanes' kernels run one after another on the one stream.
//! `run_layers`, passes with a test tap ([`GlmForward::set_tap`]) and prefill passes too small
//! for two lanes run in one lane. Lanes change the row counts of the tensor-core GEMMs, so a
//! prefill's results move within rounding (deterministically: N lanes give the bits of the same
//! rows run as N passes, lane 0's, then lane 1's, and so on); decode's row independence is
//! untouched. [`GlmForward::set_lane_trace`] times each lane per layer.
//!
//! # Two lanes (decode and verify)
//!
//! A decode or verify pass of [`ForwardConfig::decode_lane_rows`] to
//! [`ForwardConfig::decode_lane_max_rows`] rows, over two requests or more, runs in the prefill's
//! first two lanes (whatever [`ForwardConfig::lanes`] is, from 2), cut between requests (the cut
//! that splits the rows most evenly, lane A the larger on a tie): a request's rows never straddle
//! the lanes, and requests do not read each other's state, so a lane is exactly a pass over its
//! own requests. Each lane runs its own head over its own rows (the final mean and RMSNorm, then
//! the LM head GEMM of the lane's rows alone), into its rows of lane A's logits, once its last
//! layer is complete: lane A's head runs while lane B's last routed experts are out. A verify
//! pass keeps each lane's replay inputs, projection rows and raw index keys at the lane's rows of
//! the verify buffers, so the commit reads the pass's rows as one pass wrote them; it appends
//! lane A's kept rows to the drafter's contexts, then lane B's. Two lanes give the bits of the two
//! passes over the lanes' requests (lane A's, then lane B's) in everything they leave: logits and
//! picks, KDA states, conv windows, pages, tails, the drafter's taps and rings
//! (`tests/decode_lanes.rs`).
//!
//! Whether it pays depends on the load: each lane reads the coordinator's weights once and the
//! ranks read the routed experts each lane's rows name, so two lanes of many rows read most
//! experts twice. It is off by default (`decode_lane_rows` 0).
//!
//! **The drafter in lanes** (prefill, decode and verify). Each lane captures the drafter's taps
//! for its own rows, into those rows of the tap buffer (a lane's rows are a contiguous slice of
//! the pass). After the pass the rows are committed and appended to the drafter's context lane by
//! lane, in lane order (a request split by a cut gets each lane's part in turn), with the calls
//! one-lane passes of the same rows would make, so the rings hold the same bits.
//!
//! # L2 prefetch (decode and verify)
//!
//! With [`ForwardConfig::l2_prefetch`] above 0, a decode or verify pass of one lane (the GPU
//! otherwise waits for the ranks) queues, on the forward's stream after each MoE layer's shared
//! expert, a prefetch of the first `l2_prefetch` bytes of the next layer's weights (after the last
//! layer, the head's; `crate::prefetch`). It writes nothing, so every bit is the same
//! (`tests/decode_lanes.rs`); the next layer's GEMVs find those bytes in L2. Passes of two lanes,
//! where the other lane's attention fills the exchange, and prefill passes never prefetch.
//!
//! # Scoring
//!
//! [`GlmForward::score`] feeds a teacher-forced sequence through prefill passes of a chosen size
//! and returns the logits of chosen rows (the KL gate, `docs/KL-GATE.md`): after each pass the
//! chosen rows' head outputs are gathered from the lanes and the LM head runs over them alone,
//! in the GEMV's groups of up to 8 rows. Nothing else in a pass changes.
//!
//! # Memory
//!
//! Every buffer a pass uses is allocated before the forward exists ([`ForwardBuffers`]): the
//! lanes' scratch for [`ForwardConfig::max_rows`], the verify scratch, and the attention
//! workspaces for the largest pass at any context. A drafter's tap buffer and working memory are
//! sized up front too (`crate::draft::Dflash::reserve`). A pass allocates no device memory, so a
//! server can size its KV page pool from what is left and never run out mid-pass.
//!
//! In a lane's scratch the buffers only a KDA attention uses, those only a DSA attention uses and
//! those only an FFN uses share one region (a layer's attention is one or the other, and its FFN
//! runs after it). A DSA layer's sparse MLA core (absorb, attention, un-absorb) runs in blocks of
//! [`ForwardConfig::mla_block_rows`] rows whose buffers the lanes share. About 280 KiB a row
//! remain, against 780 KiB with every buffer of its own (`docs/PERFORMANCE.md` §5a).
//!
//! # Op profile
//!
//! [`GlmForward::set_op_trace`] (`GLM53F_PROFILE_OPS=1` at construction) records a CUDA event
//! after every operation of each prefill lane's attention and shared expert, and each prefill
//! pass prints an `OPS` table of their median GPU times per layer kind (`crate::opprof`). The
//! hooks are the `op` calls in the layer's functions; off, each is one check.

use core::ffi::c_void;
use std::cell::RefCell;
use std::collections::VecDeque;
use std::ops::Range;
use std::sync::Arc;
use std::time::Instant;

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
use crate::opprof::{LayerKind, OpProfile, OpTrace, Segment};
use crate::prefetch::L2Prefetch;
use crate::shape::*;
use crate::weights::{AttnW, DeviceModel, DsaW, FfnW, HcW, KdaW, LayerW, MlpW, ProjW};

/// Lanes of a prefill pass at most ([`ForwardConfig::lanes`]).
pub const MAX_LANES: usize = 4;

/// Sizes and kernel choices of a forward.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ForwardConfig {
    /// Rows of one prefill pass, every lane's together (a longer segment runs in chunks of this
    /// many).
    pub max_rows: usize,
    /// Lanes of a prefill pass: 1, or 2 to [`MAX_LANES`] to pipeline it (module documentation,
    /// "Lanes (prefill)"). A lane holds `max_rows / lanes` rows (rounded up). Lowered after
    /// construction, passes run in fewer of the lanes built.
    pub lanes: usize,
    /// The fewest rows worth a lane of their own: a prefill pass runs in one lane per this many
    /// rows, at most `lanes`, and in as many as its rows need (a pass under twice this runs in
    /// one lane if it fits in one).
    pub min_lane_rows: usize,
    /// Decode and verify passes of at least this many rows, and of two requests or more, run in
    /// two lanes cut between requests (module documentation, "Two lanes (decode and verify)");
    /// 0 keeps them in one lane. Needs lane B's buffers (`lanes` 2 or more).
    pub decode_lane_rows: usize,
    /// Decode and verify passes of more rows than this run in one lane.
    pub decode_lane_max_rows: usize,
    /// Rows of one verify pass, all windows together.
    pub max_verify_rows: usize,
    /// Requests in one pass.
    pub max_requests: usize,
    pub policy: GemmPolicy,
    /// Sparse MLA splits for passes of at most 8 rows (fixed, so a row's result does not
    /// depend on the pass).
    pub decode_splits: usize,
    /// Sparse MLA head groups per block for passes of at most 8 rows, and, for larger passes, at
    /// most: a pass too small to fill the multiprocessors takes fewer (`mla_head_groups`).
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
    /// Rows of the blocks a DSA layer's sparse MLA core (absorb, attention, un-absorb) runs in:
    /// its buffers (the absorbed query and the latent output, 256 KiB a row) are sized for one
    /// block, shared by the lanes, instead of for a whole lane. The core is row-independent, so
    /// the blocks change no bit. At least 8 (a decode or verify window runs in one block). A cap:
    /// the forward rounds it down to a multiple of the GPU's multiprocessors ([`mla_block`]).
    pub mla_block_rows: usize,
    /// Decode and verify passes of one lane: bytes of the next layer's weights pulled into L2 while
    /// a MoE layer's routed experts are out (`crate::prefetch`); 0 turns it off. Changes no bit.
    pub l2_prefetch: usize,
}

impl ForwardConfig {
    /// Rows one lane holds.
    pub fn lane_rows(&self) -> usize {
        self.max_rows.div_ceil(self.lanes.max(1))
    }
}

impl Default for ForwardConfig {
    fn default() -> Self {
        ForwardConfig {
            max_rows: 256,
            lanes: 1,
            min_lane_rows: 64,
            decode_lane_rows: 0,
            decode_lane_max_rows: usize::MAX,
            max_verify_rows: 64,
            max_requests: 16,
            policy: GemmPolicy::default(),
            decode_splits: 16,
            decode_head_groups: 1,
            prefill_head_groups: 4,
            kda_chunked_prefill: false,
            kda_prefill_rows: 256,
            kda_prefill_value_blocks: 2,
            mla_block_rows: 512,
            l2_prefetch: 0,
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
    ws: &'a Workspaces,
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
            TapBuf::DsaQ => (&s.q16, MLA_HEADS * QK_HEAD * 2),
            TapBuf::DsaKvA => (&s.kva, KV_LORA * 2),
            TapBuf::DsaIdxQ => (&s.idx_q32, INDEX_HEADS * INDEX_DIM * 4),
            TapBuf::DsaIdxProj => (&s.idx_p, IDX_PROJ_COLS * 2),
            TapBuf::DsaTokens => (&s.tokens, MAX_SELECTED * 4),
            TapBuf::DsaCounts => (&s.counts, 8),
            TapBuf::DsaHeads => (&self.ws.o32, MLA_HEADS * V_HEAD * 4),
            TapBuf::RouterLogits => (&s.router_logits, EXPERTS * 4),
            TapBuf::RouterIds => (&s.ids, TOP_K * 4),
            TapBuf::RouterWeights => (&s.weights, TOP_K * 4),
        }
    }

    /// The first `rows` rows of a buffer as raw bytes.
    pub fn bytes(&self, b: TapBuf) -> Result<Vec<u8>> {
        if b == TapBuf::DsaHeads && self.rows > self.ws.mla_rows {
            return Err(invalid!(
                "a DsaHeads tap holds passes of up to {} rows (ForwardConfig::mla_block_rows)",
                self.ws.mla_rows
            ));
        }
        let (buf, row) = self.buf(b);
        let bytes = buf.download::<u8>(self.rows * row)?;
        if b == TapBuf::DsaQ {
            // The query stays in BF16 (the absorb reads it so); its f32 value is exact.
            return Ok(bytes
                .as_chunks::<2>()
                .0
                .iter()
                .flat_map(|h| f32::from_bits(u32::from(u16::from_le_bytes(*h)) << 16).to_le_bytes())
                .collect());
        }
        Ok(bytes)
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

/// Views laid out one after another from the start of a region, each at a 256-byte boundary.
struct Carve<'a> {
    region: &'a DeviceBuffer,
    at: usize,
}

impl Carve<'_> {
    /// Bytes a set of views of these sizes takes.
    fn size(bytes: &[usize]) -> usize {
        bytes.iter().map(|&b| b.max(16).next_multiple_of(256)).sum()
    }

    fn take(&mut self, bytes: usize) -> Result<DeviceBuffer> {
        let b = bytes.max(16);
        // SAFETY: every view is a field of the scratch that owns the region, dropped with it.
        let v = unsafe { self.region.view(self.at, b) }?;
        self.at += b.next_multiple_of(256);
        Ok(v)
    }
}

/// Every per-pass buffer of one lane, sized for `rows` rows and `logits` logit rows. The
/// attention workspaces are shared by the lanes ([`Workspaces`]).
///
/// A layer's attention is KDA or DSA, and its FFN runs after it, so the buffers only a KDA
/// attention uses, those only a DSA attention uses and those only an FFN uses are views of one
/// region (`shared`), the largest of the three sets, not three allocations. Nothing reads one
/// set's buffers after the next set's first write: taps read the attention's at `AttnDone` and
/// the FFN's outputs (not its intermediates) at `FfnDone`.
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
    /// The region the KDA, DSA and FFN views below share.
    shared: DeviceBuffer,
    // KDA (views)
    p: DeviceBuffer,
    fga: DeviceBuffer,
    ag: DeviceBuffer,
    kda_out: DeviceBuffer,
    // DSA (views); the sparse MLA core's buffers are per block, in the workspaces
    qa: DeviceBuffer,
    kva: DeviceBuffer,
    q_resid: DeviceBuffer,
    q_resid_q: DeviceBuffer,
    q_resid_s: DeviceBuffer,
    q16: DeviceBuffer,
    idx_q16: DeviceBuffer,
    idx_q32: DeviceBuffer,
    idx_p: DeviceBuffer,
    kva32: DeviceBuffer,
    k_raw: DeviceBuffer,
    gate: DeviceBuffer,
    w32: DeviceBuffer,
    tokens: DeviceBuffer,
    pools: DeviceBuffer,
    counts: DeviceBuffer,
    o16: DeviceBuffer,
    /// The E4M3 form of an attention's output rows (DSA's heads, or a KDA `o_proj`'s input): not
    /// a view, since either kind of layer may use it.
    o_q: DeviceBuffer,
    o_s: DeviceBuffer,
    // FFN (views)
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
    /// Bytes of the KDA, DSA and FFN views for `r` rows, in the order [`Scratch::new`] takes them.
    fn kda_views(r: usize) -> [usize; 4] {
        [
            r * KDA_P_COLS * 2,
            r * 2 * KDA_DIM * 2,
            r * 2 * KDA_WIDTH * 2,
            r * KDA_WIDTH * 2,
        ]
    }

    fn dsa_views(r: usize) -> [usize; 17] {
        [
            r * Q_LORA * 2,
            r * KV_LORA * 2,
            r * Q_LORA * 2,
            r * Q_LORA,
            r * (Q_LORA / 128) * 4,
            r * MLA_HEADS * QK_HEAD * 2,
            r * INDEX_HEADS * INDEX_DIM * 2,
            r * INDEX_HEADS * INDEX_DIM * 4,
            r * IDX_PROJ_COLS * 2,
            r * KV_LORA * 4,
            r * INDEX_DIM * 4,
            r * INDEX_DIM * 4,
            r * INDEX_HEADS * 4,
            r * MAX_SELECTED * 4,
            r * TOP_POOLS * 4,
            r * 2 * 4,
            r * MLA_HEADS * V_HEAD * 2,
        ]
    }

    fn ffn_views(r: usize) -> [usize; 4] {
        [
            r * 2 * DENSE_INTER * 2,
            r * DENSE_INTER * 2,
            r * DENSE_INTER,
            r * (DENSE_INTER / 128) * 4,
        ]
    }

    /// `taps`: the lane that taps read and `run_layers` returns from (lane 0); the other lanes'
    /// scratch has no tap buffer (passes with a tap run in one lane).
    fn new(
        rows: usize,
        logit_rows: usize,
        requests: usize,
        max_pages: usize,
        dsa_layers: usize,
        taps: bool,
    ) -> Result<Scratch> {
        let r = rows;
        let (kv, dv, fv) = (Self::kda_views(r), Self::dsa_views(r), Self::ffn_views(r));
        let shared = buf(Carve::size(&kv).max(Carve::size(&dv)).max(Carve::size(&fv)))?;
        let mut k = Carve {
            region: &shared,
            at: 0,
        };
        let (p, fga, ag, kda_out) = (
            k.take(kv[0])?,
            k.take(kv[1])?,
            k.take(kv[2])?,
            k.take(kv[3])?,
        );
        let mut d = Carve {
            region: &shared,
            at: 0,
        };
        let mut dsa = Vec::with_capacity(dv.len());
        for b in dv {
            dsa.push(d.take(b)?);
        }
        let mut dsa = dsa.into_iter();
        let mut next = || dsa.next().expect("a DSA view");
        let mut f = Carve {
            region: &shared,
            at: 0,
        };
        let (gu, act, act_q, act_s) = (
            f.take(fv[0])?,
            f.take(fv[1])?,
            f.take(fv[2])?,
            f.take(fv[3])?,
        );
        Ok(Scratch {
            rows,
            logit_rows,
            streams: [buf(r * HC * HIDDEN * 2)?, buf(r * HC * HIDDEN * 2)?],
            tap_streams: buf(if taps { r * HC * HIDDEN * 2 } else { 0 })?,
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
            p,
            fga,
            ag,
            kda_out,
            qa: next(),
            kva: next(),
            q_resid: next(),
            q_resid_q: next(),
            q_resid_s: next(),
            q16: next(),
            idx_q16: next(),
            idx_q32: next(),
            idx_p: next(),
            kva32: next(),
            k_raw: next(),
            gate: next(),
            w32: next(),
            tokens: next(),
            pools: next(),
            counts: next(),
            o16: next(),
            o_q: buf(r * MLA_HEADS * V_HEAD)?,
            o_s: buf(r * (MLA_HEADS * V_HEAD / 128) * 4)?,
            gu,
            act,
            act_q,
            act_s,
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
            shared,
        })
    }

    /// Device bytes the scratch allocated (the views take none of their own).
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
            &self.shared,
            &self.o_q,
            &self.o_s,
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
        v.iter().map(|b| b.allocated()).sum()
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

/// The attention kernels' workspaces. The lanes share them (their kernels run one after another
/// on the stream), and they are sized at construction for the largest pass at any context, so no
/// pass grows them.
struct Workspaces {
    /// The indexer's selection (`glm53f_dsa_index_workspace_bytes`).
    idx: DeviceBuffer,
    /// Sparse MLA's split partials (passes of at most 8 rows).
    mla: DeviceBuffer,
    /// The chunked KDA prefill kernel's (when [`ForwardConfig::kda_chunked_prefill`] is on at
    /// construction; switched on later, it grows on the first chunked pass).
    kda: DeviceBuffer,
    /// The sparse MLA core's buffers for one block of `mla_rows` rows
    /// ([`ForwardConfig::mla_block_rows`], at most a pass's): the absorbed query (BF16), the
    /// latent output and its log-sum-exp, and the per-head output in f32 (passes of up to 8
    /// rows, and taps).
    mla_rows: usize,
    q_abs: DeviceBuffer,
    o_lat: DeviceBuffer,
    lse: DeviceBuffer,
    o32: DeviceBuffer,
}

impl Workspaces {
    /// Bytes of each for attention calls of up to `rows` rows on a GPU of `sms` multiprocessors:
    /// the largest over every row count, whatever the context.
    fn plan(cfg: &ForwardConfig, rows: usize, sms: i32, dsa_layers: usize) -> [usize; 3] {
        let idx = if dsa_layers == 0 {
            0
        } else {
            // `glm53f_dsa_index_plan` splits the pools of a pass of r rows into at most
            // max(2 sms / r, 1) chunks per row whatever the context (a chunk holds at least
            // ceil(pools / per_row) pools), and the workspace grows with the chunks.
            (1..=rows)
                .map(|r| {
                    let chunks = ((2 * sms.max(1) as usize) / r).max(1) as i32;
                    // SAFETY: a host function.
                    unsafe { dffi::glm53f_dsa_index_workspace_bytes(r as i32, chunks) as usize }
                })
                .max()
                .unwrap_or(0)
        };
        // Split partials only for passes of at most 8 rows (larger passes run one split).
        let mla = if dsa_layers == 0 {
            0
        } else {
            // SAFETY: a host function.
            unsafe {
                dffi::glm53f_dsa_mla_workspace_bytes(rows.min(8) as i32, cfg.decode_splits as i32)
                    as usize
            }
        };
        let kda = if cfg.kda_chunked_prefill {
            // SAFETY: a host function.
            unsafe {
                kffi::glm53f_kda_prefill_workspace_bytes(
                    KDA_HEADS as i32,
                    cfg.max_requests as i32,
                    cfg.kda_prefill_rows.max(16) as i32,
                ) as usize
            }
        } else {
            0
        };
        [idx, mla, kda]
    }

    /// Rows of one sparse MLA block for passes of up to `rows` rows on `sms` multiprocessors.
    fn mla_rows(cfg: &ForwardConfig, rows: usize, dsa_layers: usize, sms: i32) -> usize {
        if dsa_layers == 0 {
            0
        } else {
            rows.min(mla_block(cfg.mla_block_rows, sms))
        }
    }

    fn new(sizes: [usize; 3], mla_rows: usize) -> Result<Workspaces> {
        let r = mla_rows;
        Ok(Workspaces {
            idx: buf(sizes[0])?,
            mla: buf(sizes[1])?,
            kda: buf(sizes[2])?,
            mla_rows,
            q_abs: buf(r * MLA_HEADS * KV_LORA * 2)?,
            o_lat: buf(r * MLA_HEADS * KV_LORA * 4)?,
            lse: buf(r * MLA_HEADS * 4)?,
            o32: buf(r * MLA_HEADS * V_HEAD * 4)?,
        })
    }

    /// The sparse MLA core's: split partials and the block's buffers.
    fn mla_bytes(&self) -> usize {
        self.mla.bytes()
            + self.q_abs.bytes()
            + self.o_lat.bytes()
            + self.lse.bytes()
            + self.o32.bytes()
    }

    fn bytes(&self) -> usize {
        self.idx.bytes() + self.mla_bytes() + self.kda.bytes()
    }
}

/// Device bytes of a forward's per-pass buffers, by use (for start-up logs).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct BufferBytes {
    /// Each lane's scratch (0 past the forward's lanes).
    pub lanes: [usize; MAX_LANES],
    /// The verify round's saved inputs.
    pub verify: usize,
    /// The indexer, sparse MLA (split partials and the core's block buffers) and chunked KDA
    /// workspaces.
    pub workspaces: [usize; 3],
    /// The GEMM engine's cuBLAS workspace and split-K scratch.
    pub gemm: usize,
}

impl BufferBytes {
    pub fn total(&self) -> usize {
        self.lanes.iter().sum::<usize>()
            + self.verify
            + self.workspaces.iter().sum::<usize>()
            + self.gemm
    }
}

/// Every device buffer a forward's passes use, allocated before the forward is built: a server
/// allocates them, then sizes its KV page pool from the memory left ([`GlmForward::with_buffers`]).
/// No pass allocates.
pub struct ForwardBuffers {
    cfg: ForwardConfig,
    max_pages: usize,
    kda_layers: usize,
    dsa_layers: usize,
    sms: i32,
    stream: Arc<Stream>,
    gemm: Gemm,
    /// Lane 0's scratch, and the other lanes' in order.
    s: Scratch,
    rest: Vec<Scratch>,
    v: VerifyScratch,
    ws: Workspaces,
}

/// Rows the largest pass holds: a prefill's in every lane (as many times the smallest lane's
/// rows), one lane's, or a verify pass's.
fn largest_pass(s: &Scratch, rest: &[Scratch], verify: usize) -> usize {
    let lane = rest.iter().map(|b| b.rows.min(s.rows)).min();
    lane.map_or(0, |r| (1 + rest.len()) * r)
        .max(s.rows)
        .max(verify)
}

impl ForwardBuffers {
    /// The buffers of a forward over `shape` with `cfg`, for page tables of `max_pages` pages,
    /// running on `stream` (the KV pool's): the lanes' scratch, the verify scratch, the
    /// attention workspaces, and the GEMM engine's (cuBLAS and split-K) scratch.
    pub fn new(
        cfg: &ForwardConfig,
        shape: &ModelShape,
        max_pages: usize,
        stream: &Arc<Stream>,
    ) -> Result<ForwardBuffers> {
        let groups_ok = |g: usize| matches!(g, 1 | 2 | 4);
        if cfg.max_rows == 0
            || !(1..=MAX_LANES).contains(&cfg.lanes)
            || cfg.min_lane_rows == 0
            || (cfg.decode_lane_rows > 0 && cfg.lanes < 2)
            || cfg.max_verify_rows == 0
            || cfg.max_requests == 0
            || !(1..=64).contains(&cfg.decode_splits)
            || !groups_ok(cfg.decode_head_groups)
            || !groups_ok(cfg.prefill_head_groups)
            || !matches!(cfg.kda_prefill_value_blocks, 1 | 2 | 4)
            || cfg.mla_block_rows < 8
        {
            return Err(invalid!("bad forward config {cfg:?}"));
        }
        let sms = device::sm_count()?;
        let lane = cfg.lane_rows();
        // Lane A also runs decode and verify passes, and every one-lane pass.
        let rows = lane.max(cfg.max_verify_rows);
        let logit_rows = cfg.max_requests.max(cfg.max_verify_rows);
        let dl = shape.dsa_layers;
        let s = Scratch::new(rows, logit_rows, cfg.max_requests, max_pages, dl, true)?;
        // The other lanes: prefill rows only (and lane 1 a decode lane's); their logit rows go to
        // lane 0's head buffers.
        let rest = (1..cfg.lanes)
            .map(|_| Scratch::new(lane, 0, cfg.max_requests, max_pages, dl, false))
            .collect::<Result<Vec<_>>>()?;
        let v = VerifyScratch::new(cfg.max_verify_rows, shape.kda_layers, dl)?;
        let ws = Workspaces::new(
            Workspaces::plan(cfg, rows, sms, dl),
            Workspaces::mla_rows(cfg, rows, dl, sms),
        )?;
        Ok(ForwardBuffers {
            cfg: *cfg,
            max_pages,
            kda_layers: shape.kda_layers,
            dsa_layers: dl,
            sms,
            stream: stream.clone(),
            gemm: Gemm::new(stream, cfg.policy)?,
            s,
            rest,
            v,
            ws,
        })
    }

    /// Rows the largest pass holds: a prefill's in every lane, one lane's, or a verify pass's (a
    /// drafter's tap buffer holds that many: `crate::draft::Dflash::reserve`).
    pub fn pass_rows(&self) -> usize {
        largest_pass(&self.s, &self.rest, self.v.rows)
    }

    pub fn bytes(&self) -> BufferBytes {
        let mut lanes = [0; MAX_LANES];
        for (b, s) in lanes
            .iter_mut()
            .zip(std::iter::once(&self.s).chain(&self.rest))
        {
            *b = s.bytes();
        }
        BufferBytes {
            lanes,
            verify: self.v.bytes(),
            workspaces: [
                self.ws.idx.bytes(),
                self.ws.mla_bytes(),
                self.ws.kda.bytes(),
            ],
            gemm: self.gemm.bytes(),
        }
    }
}

/// The previous sublayer's output a boundary expands: block_out, block_out2 (the shared
/// expert's output, or null), post, comb.
type Prev = (*const u16, *const u16, *const f32, *const f32);

/// One lane of a pass: its requests, metadata, and where it is in the layer loop.
struct Lane {
    /// The lane's requests; `row0` counts from the lane's first row.
    reqs: Vec<Req>,
    /// Where they are in the pass: the index of the first among the pass's requests (a lane's
    /// first is the previous lane's last when a cut splits it), and the lane's first row.
    first: usize,
    base: usize,
    rows: usize,
    meta: Meta,
    /// The stream buffer holding the lane's current streams, and the output to expand next.
    cur: usize,
    prev: Option<Prev>,
    /// `(k, n)` when its first request continues the previous lane's request `k` of `n` (its DSA
    /// tails come from that lane before each DSA layer).
    continues: Option<(usize, usize)>,
    /// Logit rows, from the lane's first row, and how many of the pass's logit rows come
    /// before them.
    logit_rows: Vec<i32>,
    logit_base: usize,
    /// The MoE call in flight: submitted to the backend, not finished.
    exchange: Option<Exchange>,
    host_ids: Vec<i32>,
    host_weights: Vec<f32>,
}

/// An MoE call in flight (its device pointers are in its lane's scratch, which may be swapped
/// out when it finishes), and what its lane's next boundary expands.
struct Exchange {
    layer: usize,
    x: *const u16,
    x_q: *const u8,
    x_scales: *const f32,
    ids: *const i32,
    weights: *const f32,
    out: *mut u16,
    next: Prev,
}

impl Exchange {
    fn call<'a>(&self, lane: &'a Lane) -> ExpertCall<'a> {
        ExpertCall {
            layer: self.layer,
            rows: lane.rows,
            x: self.x,
            x_q: self.x_q,
            x_scales: self.x_scales,
            ids: self.ids,
            weights: self.weights,
            host_ids: &lane.host_ids,
            host_weights: &lane.host_weights,
            out: self.out,
        }
    }
}

/// One lane's share of a pass ([`cut`]).
struct Part {
    /// The lane's requests (`row0` from the lane's first row), the index of its first among the
    /// pass's requests, and the pass's rows it holds.
    reqs: Vec<Req>,
    first: usize,
    span: Range<usize>,
    /// `(k, n)`: its first request continues the previous lane's request `k` of `n`.
    continues: Option<(usize, usize)>,
    /// Its last request goes on in the next lane: that request's index in this lane.
    goes_on: Option<usize>,
}

/// The pass's requests (`total` rows) cut into lanes at rows `at` (ascending, each inside the
/// pass): lane i holds the rows from `at[i - 1]` (0 for lane 0) to `at[i]` (`total` for the last).
/// A request across a cut is split: the next lane continues it at the positions after the
/// previous lane's part, and may hand it on again.
fn cut(reqs: &[Req], total: usize, at: &[usize]) -> Vec<Part> {
    let mut parts: Vec<Part> = Vec::with_capacity(at.len() + 1);
    let mut lo = 0;
    for hi in at.iter().copied().chain(std::iter::once(total)) {
        let mut p = Part {
            reqs: Vec::new(),
            first: 0,
            span: lo..hi,
            continues: None,
            goes_on: None,
        };
        for (i, r) in reqs.iter().enumerate() {
            let (a, b) = (r.row0.max(lo), (r.row0 + r.rows).min(hi));
            if a >= b {
                continue;
            }
            if p.reqs.is_empty() {
                p.first = i;
                if let Some(before) = parts.last().filter(|_| r.row0 < lo) {
                    p.continues = Some((before.reqs.len() - 1, before.reqs.len()));
                }
            }
            if r.row0 + r.rows > hi {
                p.goes_on = Some(p.reqs.len());
            }
            p.reqs.push(Req {
                start: r.start + (a - r.row0),
                rows: b - a,
                row0: a - lo,
                ..*r
            });
        }
        parts.push(p);
        lo = hi;
    }
    parts
}

/// The lanes a prefill pass of `total` rows runs in, of `n` it may use: one per `min_rows` rows,
/// and at least as many as its rows need when one lane holds `first` rows and each lane of
/// several `rows` (module documentation, "Lanes (prefill)").
fn lanes_for(total: usize, n: usize, min_rows: usize, first: usize, rows: usize) -> usize {
    let mut k = (total / min_rows.max(1)).clamp(1, n.max(1));
    while k < n && total > if k == 1 { first } else { k * rows } {
        k += 1;
    }
    k
}

/// Head groups per block of the sparse MLA kernel for a pass of `rows` rows in one split (a pass
/// of more than 8 rows), at most `cap`. The grid is `rows * 4 / groups` blocks and a block takes
/// over half a multiprocessor's shared memory, so blocks run one to a multiprocessor: at `cap`
/// groups a small pass leaves most of the `sms` multiprocessors idle. Fewer groups make more,
/// faster blocks (2,051 tokens on the RTX 4090: 316 us at 4 groups, 160 at 2, 109 at 1; see
/// `docs/PERFORMANCE.md` 5b), so the pass takes the fewest groups whose grid still fits one wave,
/// and a pass that fills the multiprocessors at `cap` groups keeps `cap`. A head group's heads
/// are computed the same way whichever groups share its block, so the choice moves no bit.
fn mla_head_groups(rows: usize, sms: i32, cap: usize) -> usize {
    (rows * (MLA_HEADS / 16))
        .div_ceil(sms.max(1) as usize)
        .next_power_of_two()
        .min(cap)
}

/// Rows of a sparse MLA block under a cap of `cap` rows on `sms` multiprocessors: the largest
/// multiple of `sms` up to `cap`, as the one-split kernels run a block's rows one to a
/// multiprocessor, in waves; `cap` itself when that would be under 8 rows. 512 on the RTX 4090
/// (128 multiprocessors); 510 on the RTX 5090 (170), where 512 took four waves, the last of 2 rows.
fn mla_block(cap: usize, sms: i32) -> usize {
    let sms = sms.max(1) as usize;
    let whole = cap / sms * sms;
    if whole >= 8 {
        whole
    } else {
        cap
    }
}

/// Pipeline timings of the last traced pass (with [`GlmForward::set_lane_trace`] on): per MoE
/// layer and lane, GPU time from events on the stream and host time from the wall clock.
#[derive(Clone, Debug)]
pub struct LaneTrace {
    pub mode: Mode,
    /// Rows and requests per lane, and the calls in flight at most (the backend's depth, at most
    /// one a lane).
    pub rows: Vec<usize>,
    pub requests: Vec<usize>,
    pub depth: usize,
    /// Host time of the layer loop.
    pub loop_ms: f64,
    pub layers: Vec<LayerTrace>,
    /// The expert backend's own record of the pass's calls (`ExpertBackend::trace_end`: the
    /// remote experts' return paths and host times), when it keeps one.
    pub wire: Option<String>,
    /// Decode and verify passes: the step around the pass.
    pub step: StepTimes,
}

/// Host times of the decode step around a traced decode or verify pass, in milliseconds.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct StepTimes {
    /// From the end of the forward's last call (a pass or a commit) to the step's first (its
    /// drafts, or the pass): the scheduler's and the sampler's work in between.
    pub gap_ms: f64,
    /// The drafter's proposals before a verify pass (0 before a decode pass).
    pub draft_ms: f64,
    /// The pass, from its call to the picks on the host (a decode pass's appends to the
    /// drafter's contexts included).
    pub pass_ms: f64,
    /// A verify round's commit (0 for a decode pass).
    pub commit_ms: f64,
}

/// One MoE layer of a traced pass; the vectors are per lane.
#[derive(Clone, Debug, Default)]
pub struct LayerTrace {
    pub layer: usize,
    /// Host time from lane 0 starting this layer to lane 0 starting the next (or the loop's end).
    pub wall_ms: f64,
    /// GPU time of the lane's attention sublayer with both boundaries and the router (its work
    /// before the routes go to the host), and of its shared expert.
    pub gpu_attn_ms: Vec<f64>,
    pub gpu_shared_ms: Vec<f64>,
    /// Host time blocked on the GPU for the routes, inside the backend's `submit`, and inside its
    /// `finish` (blocked on the exchange, then the output enqueued).
    pub routes_ms: Vec<f64>,
    pub submit_ms: Vec<f64>,
    pub finish_ms: Vec<f64>,
    /// Host time from `submit` returning to `finish` returning: the exchange's round trip as the
    /// coordinator sees it. About the other lane's work when the exchange hides behind it; the
    /// rank's compute plus the transfer when the coordinator waits (`finish`).
    pub out_ms: Vec<f64>,
}

fn median(mut v: Vec<f64>) -> f64 {
    if v.is_empty() {
        return 0.0;
    }
    v.sort_by(f64::total_cmp);
    v[v.len() / 2]
}

impl LaneTrace {
    /// Medians over the MoE layers, one line: per-layer wall time, each lane's GPU and host
    /// times, and how busy the GPU was.
    pub fn summary(&self) -> String {
        let n = self.rows.len();
        let med = |f: &dyn Fn(&LayerTrace) -> f64| median(self.layers.iter().map(f).collect());
        let per = |name: &str, f: &dyn Fn(&LayerTrace, usize) -> f64| {
            let v: Vec<String> = (0..n)
                .map(|x| format!("{:.2}", med(&|t| f(t, x))))
                .collect();
            format!("{name} {}", v.join(" + "))
        };
        let wall = med(&|t| t.wall_ms);
        let busy = med(&|t| (0..n).map(|x| t.gpu_attn_ms[x] + t.gpu_shared_ms[x]).sum());
        let rows: Vec<String> = self.rows.iter().map(|r| r.to_string()).collect();
        let line = format!(
            concat!(
                "lanes {} rows (depth {}): layer loop {:.1} ms; per MoE layer, median of {}: ",
                "wall {wall:.2} ms; GPU {}, {} (busy {busy:.2} ms, {:.0}%); host {}, {}, {}; {}"
            ),
            rows.join(" + "),
            self.depth,
            self.loop_ms,
            self.layers.len(),
            per("attention", &|t, x| t.gpu_attn_ms[x]),
            per("shared", &|t, x| t.gpu_shared_ms[x]),
            100.0 * busy / wall.max(1e-9),
            per("routes wait", &|t, x| t.routes_ms[x]),
            per("submit", &|t, x| t.submit_ms[x]),
            per("finish", &|t, x| t.finish_ms[x]),
            per("exchange out", &|t, x| t.out_ms[x]),
            wall = wall,
            busy = busy,
        );
        match &self.wire {
            Some(w) => format!("{line}; {w}"),
            None => line,
        }
    }

    /// A decode or verify pass's line (`STEP`): its requests, the step's host times, the MoE
    /// layers' totals (host wall, GPU busy, and the host blocked in `finish` waiting for the
    /// experts), then [`LaneTrace::summary`].
    pub fn step_summary(&self) -> String {
        let s = &self.step;
        let sum = |f: &dyn Fn(&LayerTrace) -> f64| self.layers.iter().map(f).sum::<f64>();
        let reqs: Vec<String> = self.requests.iter().map(|r| r.to_string()).collect();
        format!(
            concat!(
                "{:?}, {} requests ({}); step: gap {:.2} ms, draft {:.2} ms, pass {:.2} ms, ",
                "commit {:.2} ms; {} MoE layers: wall {:.1} ms, GPU busy {:.1} ms, host in finish ",
                "{:.1} ms; {}"
            ),
            self.mode,
            self.requests.iter().sum::<usize>(),
            reqs.join(" + "),
            s.gap_ms,
            s.draft_ms,
            s.pass_ms,
            s.commit_ms,
            self.layers.len(),
            sum(&|t| t.wall_ms),
            sum(&|t| t.gpu_attn_ms.iter().chain(&t.gpu_shared_ms).sum::<f64>()),
            sum(&|t| t.finish_ms.iter().sum::<f64>()),
            self.summary()
        )
    }
}

/// Where a lane's trace event is recorded in a layer.
#[derive(Clone, Copy)]
enum At {
    AttnStart = 0,
    RouterEnd = 1,
    SharedStart = 2,
    SharedEnd = 3,
}

/// One lane's layer while tracing: event indices and host times.
#[derive(Clone, Default)]
struct TraceRec {
    events: [Option<usize>; 4],
    routes_ms: f64,
    submit_ms: f64,
    finish_ms: f64,
    sent: Option<Instant>,
    out_ms: f64,
}

/// The trace being recorded, and the last pass's.
struct Tracer {
    /// Print each traced pass's summary to stderr (`PIPE` for prefill, `STEP` for decode and
    /// verify).
    print: bool,
    /// Recording the current pass, and its mode.
    active: bool,
    mode: Mode,
    events: Vec<Event>,
    used: usize,
    /// `[layer][lane]` records of the current pass.
    recs: Vec<Vec<TraceRec>>,
    /// When lane 0 started each layer, and the layer loop's bounds.
    starts: Vec<(usize, Instant)>,
    loop_start: Instant,
    loop_end: Instant,
    /// The last prefill pass's trace, and the last decode or verify pass's.
    last: Option<LaneTrace>,
    last_step: Option<LaneTrace>,
    /// The decode step being traced: its host times so far (the drafts' once drafted), and
    /// whether `last_step` is its verify pass waiting for the commit.
    step: Option<StepTimes>,
    verify_open: bool,
    /// When the forward's last pass or commit ended.
    idle_since: Option<Instant>,
}

impl Tracer {
    /// Milliseconds from the end of the forward's last call to `t`.
    fn gap_ms(&self, t: Instant) -> f64 {
        self.idle_since
            .map_or(0.0, |i| t.saturating_duration_since(i).as_secs_f64() * 1e3)
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
    /// Two decode lanes: lane A's requests (the commit appends each lane's rows to the drafter's
    /// contexts in a call of its own, as two passes would).
    split: Option<usize>,
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
    /// The active lane's buffers: lane 0's outside a pass.
    s: Scratch,
    /// The other lanes' (lane i's at `i - 1`, but lane 0's where the active lane's belong), and
    /// which lane `s` is ([`GlmForward::use_lane`]).
    rest: Vec<Scratch>,
    lane: usize,
    v: VerifyScratch,
    ws: Workspaces,
    pending: Option<Pending>,
    timer: RefCell<Option<Timer>>,
    tap: Option<Box<TapFn>>,
    trace: Option<Tracer>,
    /// The op profile (`GLM53F_PROFILE_OPS`, [`GlmForward::set_op_trace`]).
    ops: RefCell<Option<OpTrace>>,
    sms: i32,
    /// The DFlash2 drafter, when attached.
    draft: Option<Dflash>,
    /// The first row of each lane of the last pass (where [`GlmForward::score`] finds a row's
    /// head output).
    lane_bases: Vec<usize>,
    /// Decode and verify passes run in two lanes so far.
    decode_lane_passes: u64,
    /// The L2 prefetch's plan ([`ForwardConfig::l2_prefetch`]).
    prefetch: L2Prefetch,
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
        let bufs = ForwardBuffers::new(&cfg, &model.shape, kv.config().max_pages, kv.stream())?;
        GlmForward::with_buffers(model, embed, kv, experts, bufs)
    }

    /// [`GlmForward::new`] with buffers allocated beforehand ([`ForwardBuffers::new`], for the
    /// same shape and page-table width): the order a server takes to size its page pool from
    /// the memory the forward leaves.
    pub fn with_buffers(
        model: DeviceModel,
        embed: HostEmbedding,
        kv: KvPool,
        experts: Box<dyn ExpertBackend>,
        bufs: ForwardBuffers,
    ) -> Result<GlmForward> {
        let shape = &model.shape;
        let layout = KvLayout::new(shape, None);
        let kl = kv.config().layout;
        if kl.kda_layers != layout.kda_layers || kl.dsa_layers != layout.dsa_layers {
            return Err(invalid!("the KV pool is laid out for another model shape"));
        }
        if bufs.kda_layers != shape.kda_layers
            || bufs.dsa_layers != shape.dsa_layers
            || bufs.max_pages != kv.config().max_pages
            || !Arc::ptr_eq(&bufs.stream, kv.stream())
        {
            return Err(invalid!(
                "the forward's buffers were made for another shape, page-table width or stream"
            ));
        }
        // SAFETY: one-time kernel setup (shared-memory limits).
        device::launched(unsafe { dffi::glm53f_dsa_init() }, "glm53f_dsa_init")?;
        let prefetch = L2Prefetch::for_model(&model)?;
        let ForwardBuffers {
            cfg,
            sms,
            stream,
            gemm,
            s,
            rest,
            v,
            ws,
            ..
        } = bufs;
        let mut fwd = GlmForward {
            gemm,
            sms,
            model: Arc::new(model),
            embed,
            kv,
            experts,
            stream,
            cfg,
            s,
            rest,
            lane: 0,
            v,
            ws,
            pending: None,
            timer: RefCell::new(None),
            tap: None,
            trace: None,
            ops: RefCell::new(None),
            draft: None,
            lane_bases: Vec::new(),
            decode_lane_passes: 0,
            prefetch,
        };
        if std::env::var("GLM53F_PROFILE_OPS").is_ok_and(|v| !v.is_empty() && v != "0") {
            // Every prefill pass prints its `PIPE` line and its `OPS` table.
            fwd.set_op_trace(true, true);
        }
        Ok(fwd)
    }

    pub fn shape(&self) -> &ModelShape {
        &self.model.shape
    }

    pub fn stream(&self) -> &Arc<Stream> {
        &self.stream
    }

    /// Device bytes of the forward's own buffers (not weights, not the KV pool): every lane's
    /// scratch, the verify scratch, the attention workspaces and the GEMM engine's scratch.
    pub fn scratch_bytes(&self) -> usize {
        self.s.bytes()
            + self.rest.iter().map(Scratch::bytes).sum::<usize>()
            + self.v.bytes()
            + self.ws.bytes()
            + self.gemm.bytes()
    }

    /// Record [`LaneTrace`]s of later passes ([`GlmForward::take_lane_trace`] for prefill,
    /// [`GlmForward::take_step_trace`] for decode and verify), and with `print`, write each one's
    /// summary to stderr: `PIPE ...` per prefill pass, `STEP ...` per decode step (a decode pass,
    /// or a verify pass with its drafts and its commit).
    pub fn set_lane_trace(&mut self, on: bool, print: bool) {
        self.trace = on.then(|| Tracer {
            print,
            active: false,
            mode: Mode::Prefill,
            events: Vec::new(),
            used: 0,
            recs: Vec::new(),
            starts: Vec::new(),
            loop_start: Instant::now(),
            loop_end: Instant::now(),
            last: None,
            last_step: None,
            step: None,
            verify_open: false,
            idle_since: None,
        });
    }

    /// Decode and verify passes run in two lanes so far ([`ForwardConfig::decode_lane_rows`]).
    pub fn decode_lane_passes(&self) -> u64 {
        self.decode_lane_passes
    }

    /// The L2 prefetch ([`ForwardConfig::l2_prefetch`]): its plan and counter.
    pub fn l2_prefetch(&self) -> &L2Prefetch {
        &self.prefetch
    }

    /// The last traced prefill pass's timings.
    pub fn take_lane_trace(&mut self) -> Option<LaneTrace> {
        self.trace.as_mut().and_then(|t| t.last.take())
    }

    /// The last traced decode or verify pass's timings, with its step's (a verify pass's are
    /// complete after its commit).
    pub fn take_step_trace(&mut self) -> Option<LaneTrace> {
        self.trace.as_mut().and_then(|t| {
            t.verify_open = false;
            t.last_step.take()
        })
    }

    /// Record the op profile of later prefill passes (`crate::opprof`): the GPU time of every
    /// operation of each lane's attention and shared expert, per layer
    /// ([`GlmForward::take_op_profile`]); with `print`, each pass's `OPS` table goes to stderr,
    /// and the lane trace is turned on (printing) if it is off, for the `PIPE` line it adds up
    /// to. `GLM53F_PROFILE_OPS=1` does this at construction.
    pub fn set_op_trace(&mut self, on: bool, print: bool) {
        *self.ops.borrow_mut() = on.then(|| OpTrace::new(print));
        if on && print && self.trace.is_none() {
            self.set_lane_trace(true, true);
        }
    }

    /// The last prefill pass's op profile (op profile on).
    pub fn take_op_profile(&mut self) -> Option<OpProfile> {
        self.ops.borrow_mut().as_mut().and_then(|o| o.last.take())
    }

    /// The end of op `name` in the active lane's current segment (op profile on).
    fn op(&self, name: &'static str) -> Result<()> {
        match self.ops.borrow_mut().as_mut() {
            Some(o) => o.op(&self.stream, name),
            None => Ok(()),
        }
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
        if stage.starts_with("kda_") {
            // A KDA layer's stages (`kda_proj`, `kda_core`, `kda_o`) are its ops in the op
            // profile: its body carries no hooks of its own.
            self.op(stage)?;
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
        d.reserve_taps(self.pass_rows())?;
        self.draft = Some(d);
        Ok(())
    }

    /// Rows the largest pass holds: a prefill's in every lane, one lane's, or a verify pass's
    /// (the drafter's tap buffer holds that many).
    pub fn pass_rows(&self) -> usize {
        largest_pass(&self.s, &self.rest, self.v.rows)
    }

    pub fn has_drafter(&self) -> bool {
        self.draft.is_some()
    }

    pub fn drafter(&self) -> Option<&Dflash> {
        self.draft.as_ref()
    }

    /// The attached drafter, to reserve its memory ([`Dflash::reserve`]) or read its counters.
    pub fn drafter_mut(&mut self) -> Option<&mut Dflash> {
        self.draft.as_mut()
    }

    /// The drafter's proposals for `reqs` (each slot at its committed length), `block - 1` per
    /// request. Slots and the target's state do not change.
    pub fn draft(&mut self, reqs: &[DraftReq<'_>]) -> Result<Vec<glm53f_dflash::seam::Proposal>> {
        let t0 = Instant::now();
        let d = self
            .draft
            .as_mut()
            .ok_or_else(|| invalid!("no drafter attached"))?;
        let r = d.draft(reqs, &self.embed);
        if let Some(t) = self.trace.as_mut() {
            // A decode step with drafts starts here (the lane trace's `STEP` line).
            t.step = Some(StepTimes {
                gap_ms: t.gap_ms(t0),
                draft_ms: t0.elapsed().as_secs_f64() * 1e3,
                ..StepTimes::default()
            });
        }
        r
    }

    // ---- Public passes ---------------------------------------------------------------------

    /// Rows one prefill pass takes now: every lane's when passes run in several, else one
    /// lane's.
    pub fn prefill_rows(&self) -> usize {
        let one = self.s.rows.min(self.cfg.max_rows);
        let n = self.lanes_now();
        if n > 1 && self.tap.is_none() {
            self.cfg.max_rows.min(n * one.min(self.lane_rows()))
        } else {
            one
        }
    }

    /// Lanes a prefill pass may run in now: [`ForwardConfig::lanes`], at most the lanes built.
    fn lanes_now(&self) -> usize {
        self.cfg.lanes.clamp(1, 1 + self.rest.len())
    }

    /// Rows the smallest lane holds (lane 0 also holds a verify pass, or every row of a one-lane
    /// forward).
    fn lane_rows(&self) -> usize {
        self.rest
            .iter()
            .map(|b| b.rows)
            .fold(self.s.rows, usize::min)
    }

    /// Append and commit each segment's tokens; returns the greedy pick after each segment's
    /// last token. Segments that fit in one pass together run as one batch; others run one
    /// after another in chunks of [`GlmForward::prefill_rows`] (`max_rows`).
    pub fn prefill(&mut self, segs: &mut [(&mut GlmKv, &[u32])]) -> Result<Vec<u32>> {
        self.check_idle(segs.iter().map(|(k, _)| &**k))?;
        let total: usize = segs.iter().map(|(_, t)| t.len()).sum();
        if segs.iter().any(|(_, t)| t.is_empty()) {
            return Err(invalid!("an empty prefill segment"));
        }
        let cap = self.prefill_rows();
        if segs.len() <= self.cfg.max_requests && total <= cap {
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
            for chunk in tokens.chunks(cap) {
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
        let t0 = Instant::now();
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
                    if pool.cfg.layout.kda_state_bf16 {
                        kffi::glm53f_kda_replay_batch_bf16state(
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
                    } else {
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
                    }
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
        // The kept rows (the anchor and the accepted drafts) become drafter context; after a
        // verify in two lanes, lane A's requests and then lane B's, the calls the commits of two
        // passes make.
        if let Some(d) = self.draft.as_mut() {
            let rows: Vec<(usize, usize)> = pending
                .reqs
                .iter()
                .zip(keep)
                .map(|(r, &k)| (r.row0, k))
                .collect();
            let k = pending.split.unwrap_or(rows.len());
            let (a, b) = slots.split_at_mut(k);
            d.append(a, &rows[..k])?;
            if !b.is_empty() {
                d.append(b, &rows[k..])?;
            }
            self.mark(usize::MAX, "draft_append")?;
        }
        if let Some(t) = self.trace.as_mut() {
            let now = Instant::now();
            if std::mem::take(&mut t.verify_open) {
                if let Some(tr) = t.last_step.as_mut() {
                    tr.step.commit_ms = (now - t0).as_secs_f64() * 1e3;
                    if t.print {
                        eprintln!("STEP {}", tr.step_summary());
                    }
                }
            }
            t.idle_since = Some(now);
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

    // ---- Teacher-forced scoring ----------------------------------------------------------------

    /// Teacher-forced scoring: append `tokens` to `kv` (a fresh slot, for the KL gate) in passes
    /// of `pass_rows` rows and return the f32 logits `[rows.len()][VOCAB]` of `rows` (ascending
    /// indices into `tokens`: row r is the output at position r, predicting token r + 1),
    /// padding columns included.
    ///
    /// The passes are those [`GlmForward::prefill`] runs for one segment cut into chunks of
    /// `pass_rows` (at most [`GlmForward::prefill_rows`]): up to 8 rows run the row-independent
    /// decode kernels in one lane, whose bits equal serial decode steps; more rows the prefill
    /// kernels, in the forward's lanes. After each pass the requested rows of its
    /// chunk are gathered from their lanes' head outputs (the head's final norm, as for any logit
    /// row) and the LM head runs over them alone, in groups of up to the GEMV's 8 rows: the kernel
    /// a prefill's last row takes, so a row's logits do not depend on which other rows are scored
    /// and equal the forward's own logits for that row.
    pub fn score(
        &mut self,
        kv: &mut GlmKv,
        tokens: &[u32],
        rows: &[usize],
        pass_rows: usize,
    ) -> Result<Vec<f32>> {
        let mut out = Vec::with_capacity(rows.len() * VOCAB);
        self.score_each(kv, tokens, rows, pass_rows, |_, logits| {
            out.extend_from_slice(logits);
            Ok(())
        })?;
        Ok(out)
    }

    /// [`GlmForward::score`], handing each requested row's logits to `sink` (the row, its `VOCAB`
    /// logits) in order as soon as they are computed, instead of collecting them: a caller that
    /// writes them out holds one group of at most 8 rows.
    pub fn score_each(
        &mut self,
        kv: &mut GlmKv,
        tokens: &[u32],
        rows: &[usize],
        pass_rows: usize,
        mut sink: impl FnMut(usize, &[f32]) -> Result<()>,
    ) -> Result<()> {
        self.check_idle(std::iter::once(&*kv))?;
        let cap = self.prefill_rows();
        if tokens.is_empty() || pass_rows == 0 || pass_rows.min(tokens.len()) > cap {
            return Err(invalid!(
                "{} tokens in passes of {pass_rows} rows (this forward's prefill passes hold {cap})",
                tokens.len()
            ));
        }
        if rows.windows(2).any(|w| w[0] >= w[1]) || rows.last().is_some_and(|&r| r >= tokens.len())
        {
            return Err(invalid!(
                "the rows to score must ascend and index the {} tokens",
                tokens.len()
            ));
        }
        let st = self.stream.clone();
        let lm = self.model.head.lm_head.mat();
        let group = self.s.logit_rows.min(self.gemm.policy.gemv_max_rows).max(1);
        let mut host = vec![0f32; group * VOCAB];
        let mut next = 0;
        for (c, chunk) in tokens.chunks(pass_rows).enumerate() {
            let first = c * pass_rows;
            self.pass(
                Mode::Prefill,
                &mut [&mut *kv],
                &[chunk.len()],
                Input::Tokens(chunk),
                0..self.shape().layers,
                true,
                false,
            )?;
            let here = rows[next..].partition_point(|&r| r < first + chunk.len());
            for g in rows[next..next + here].chunks(group) {
                for (i, &r) in g.iter().enumerate() {
                    // Each lane's head output holds its rows of the pass.
                    let x = self.lane_bases.partition_point(|&b| b <= r - first) - 1;
                    let (src, at) = (&self.scratch(x).head_out, r - first - self.lane_bases[x]);
                    self.s.head_sel.copy_from(
                        &st,
                        i * HIDDEN * 2,
                        src,
                        at * HIDDEN * 2,
                        HIDDEN * 2,
                    )?;
                }
                let n = g.len();
                // SAFETY: head_sel holds n <= logit_rows rows of the head output; logits holds
                // n rows of VOCAB f32.
                unsafe {
                    self.gemm.bf16(
                        self.s.head_sel.ptr(0),
                        HIDDEN,
                        0,
                        &lm,
                        n,
                        self.s.logits.ptr::<c_void>(0),
                        VOCAB,
                        0,
                        true,
                        &st,
                    )
                }?;
                // SAFETY: a host vector of at least n rows of VOCAB f32.
                let b = unsafe {
                    std::slice::from_raw_parts_mut(host.as_mut_ptr().cast::<u8>(), n * VOCAB * 4)
                };
                self.s.logits.download_bytes(&st, 0, b)?;
                for (i, &r) in g.iter().enumerate() {
                    sink(r, &host[i * VOCAB..(i + 1) * VOCAB])?;
                }
            }
            next += here;
            self.mark(usize::MAX, "score_head")?;
        }
        Ok(())
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
        let t0 = Instant::now();
        let r = self.pass_lanes(mode, kvs, rows, input, layers, head, logits);
        // Between passes (a failed one too) lane 0's buffers are the active ones.
        self.use_lane(0);
        if let Some(t) = self.trace.as_mut() {
            t.active = false;
            let now = Instant::now();
            // A decode step's times: its drafts' (if drafted), then the pass; a verify pass's
            // line waits for its commit.
            let step = t.step.take();
            if let Some(tr) = t
                .last_step
                .as_mut()
                .filter(|_| r.is_ok() && head && mode != Mode::Prefill)
            {
                tr.step = StepTimes {
                    pass_ms: (now - t0).as_secs_f64() * 1e3,
                    ..step.unwrap_or(StepTimes {
                        gap_ms: t
                            .idle_since
                            .map_or(0.0, |i| t0.saturating_duration_since(i).as_secs_f64() * 1e3),
                        ..StepTimes::default()
                    })
                };
                t.verify_open = mode == Mode::Verify;
                if t.print && mode == Mode::Decode {
                    eprintln!("STEP {}", tr.step_summary());
                }
            }
            t.idle_since = Some(now);
        }
        r
    }

    /// The lanes a pass of `total` rows runs in (module documentation, "Lanes (prefill)"): a
    /// prefill of token ids without a tap runs in one lane per [`ForwardConfig::min_lane_rows`]
    /// rows, at most the lanes it may use now, and in at least as many as its rows need (one lane
    /// holds lane 0's rows, several the smallest lane's each); anything else in one lane.
    fn prefill_lanes(&self, mode: Mode, input: &Input<'_>, total: usize) -> usize {
        let n = self.lanes_now();
        if mode != Mode::Prefill
            || n == 1
            || self.tap.is_some()
            || !matches!(input, Input::Tokens(_))
        {
            return 1;
        }
        lanes_for(
            total,
            n,
            self.cfg.min_lane_rows,
            self.s.rows,
            self.lane_rows(),
        )
    }

    /// Where a decode or verify pass of `rows[i]` rows per request cuts its two lanes (module
    /// documentation, "Two lanes (decode and verify)"): the first row of lane B, the boundary
    /// between requests that splits the rows most evenly (lane A the larger on a tie). None runs
    /// the pass in one lane: a prefill; fewer than two requests; a pass outside
    /// [`ForwardConfig::decode_lane_rows`] to [`ForwardConfig::decode_lane_max_rows`] rows; no
    /// lane B; a tap; input streams; or a lane its buffers do not hold.
    fn decode_cut(&self, mode: Mode, input: &Input<'_>, rows: &[usize]) -> Option<usize> {
        let total: usize = rows.iter().sum();
        let lane_b = self.rest.first()?;
        if mode == Mode::Prefill
            || self.cfg.lanes < 2
            || self.cfg.decode_lane_rows == 0
            || !(self.cfg.decode_lane_rows..=self.cfg.decode_lane_max_rows).contains(&total)
            || rows.len() < 2
            || self.tap.is_some()
            || matches!(input, Input::Streams(_))
        {
            return None;
        }
        let off = |at: usize| (2 * at).abs_diff(total);
        let (mut at, mut best) = (0, 0);
        for &r in &rows[..rows.len() - 1] {
            at += r;
            if best == 0 || off(at) <= off(best) {
                best = at;
            }
        }
        (best <= self.s.rows && total - best <= lane_b.rows).then_some(best)
    }

    /// Make lane `i`'s buffers the active ones (`s`). The others wait in `rest`: lane j's at
    /// `j - 1`, but lane 0's at the active lane's place.
    fn use_lane(&mut self, i: usize) {
        if i == self.lane {
            return;
        }
        if self.lane > 0 {
            // Lane 0's back to `s`, the active lane's to its place.
            std::mem::swap(&mut self.s, &mut self.rest[self.lane - 1]);
        }
        if i > 0 {
            std::mem::swap(&mut self.s, &mut self.rest[i - 1]);
        }
        self.lane = i;
    }

    /// Lane `i`'s buffers, wherever they are ([`GlmForward::use_lane`]).
    fn scratch(&self, i: usize) -> &Scratch {
        if i == self.lane {
            &self.s
        } else if i == 0 {
            &self.rest[self.lane - 1]
        } else {
            &self.rest[i - 1]
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn pass_lanes(
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
        let n_lanes = self.prefill_lanes(mode, &input, total);
        // Decode and verify: two lanes of whole requests.
        let decode_at = self.decode_cut(mode, &input, rows);
        let cap = if mode == Mode::Verify {
            self.v.rows
        } else if n_lanes > 1 {
            n_lanes * self.lane_rows()
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
        let n_logits = if mode == Mode::Prefill {
            reqs.len()
        } else {
            total
        };
        if head && logits && n_logits > self.s.logit_rows {
            return Err(invalid!(
                "{n_logits} logit rows, the scratch holds {}",
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
        // The lanes: the rows in `n_lanes` even parts, lane i from row ceil(i total / n_lanes) (a
        // request across a cut is split, the next lane continuing it); a decode or verify pass's
        // requests in two (cut between requests); or one lane of every row.
        let at: Vec<usize> = match decode_at {
            Some(at) => vec![at],
            None => (1..n_lanes)
                .map(|i| (i * total).div_ceil(n_lanes))
                .collect(),
        };
        let parts = cut(&reqs, total, &at);
        let st = self.stream.clone();
        let mut lanes = Vec::with_capacity(parts.len());
        let mut logit_base = 0;
        for (i, part) in parts.into_iter().enumerate() {
            let Part {
                reqs: lreqs,
                first,
                span,
                continues,
                goes_on,
            } = part;
            self.use_lane(i);
            let n = span.len();
            // Prefill: each request's last row (not a lane's part of a request the next lane goes
            // on with). Decode and verify: every row.
            let logit_rows: Vec<i32> = match mode {
                Mode::Prefill => lreqs
                    .iter()
                    .enumerate()
                    .filter(|&(k, _)| goes_on != Some(k))
                    .map(|(_, r)| (r.row0 + r.rows - 1) as i32)
                    .collect(),
                _ => (0..n as i32).collect(),
            };
            let meta = self.upload_meta(
                &lreqs,
                n,
                tokens.map(|t| &t[span.clone()]),
                None,
                &logit_rows,
            )?;
            self.gather_batch(&lreqs, &meta)?;
            // Input streams.
            match &input {
                Input::Tokens(_) => unsafe {
                    self.embed.gather(
                        meta.ids,
                        n,
                        self.s.streams[0].ptr(0),
                        core::ptr::null_mut(),
                        &st,
                    )
                }?,
                Input::DeviceIds(ids) => unsafe {
                    self.embed.gather(
                        ids.wrapping_add(span.start),
                        n,
                        self.s.streams[0].ptr(0),
                        core::ptr::null_mut(),
                        &st,
                    )
                }?,
                Input::Streams(x) => self.s.streams[0].upload_async(&st, 0, x)?,
            }
            let nl = logit_rows.len();
            lanes.push(Lane {
                reqs: lreqs,
                first,
                base: span.start,
                rows: n,
                meta,
                cur: 0,
                prev: None,
                continues,
                logit_rows,
                logit_base,
                exchange: None,
                host_ids: Vec::new(),
                host_weights: Vec::new(),
            });
            logit_base += nl;
        }
        self.lane_bases = lanes.iter().map(|l| l.base).collect();
        self.mark(usize::MAX, "embed")?;
        // Every pass through the head captures the drafter's taps (`crate::draft`).
        let taps = head && self.draft.is_some();
        // Decode and verify lanes: each lane's head over its own rows, as soon as its last layer
        // is done (lane A's while lane B's last experts are out).
        let heads = head && logits && decode_at.is_some();
        self.run_lanes(mode, &mut lanes, layers, taps, heads)?;
        self.decode_lane_passes += u64::from(decode_at.is_some());
        // Tails of prefill and decode passes were committed in the layers: back to the slots,
        // lane by lane (a split request's final tail is its last lane's).
        if mode != Mode::Verify {
            for (i, lane) in lanes.iter().enumerate() {
                self.use_lane(i);
                self.scatter_tails(&lane.reqs, &lane.meta)?;
            }
        }
        let picks = if heads {
            self.argmax(total)?
        } else if head {
            self.head(&lanes, total, logits)?
        } else {
            if self.tap.is_none() {
                // The last layer's output streams, for run_layers (one lane).
                let l = &lanes[0];
                self.expand_to_tap(total, l.cur, l.prev.expect("the last layer's output"))?;
            }
            Vec::new()
        };
        self.finish_trace(&lanes)?;
        if let Some(o) = self.ops.get_mut().as_mut() {
            o.finish(&self.stream)?;
        }
        if mode == Mode::Verify {
            // Pending rows; they reach the drafter at their commit.
            for (kv, &r) in kvs.iter_mut().zip(rows) {
                kv.pending = r;
            }
            let split = decode_at.map(|_| lanes[0].reqs.len());
            self.pending = Some(Pending {
                reqs,
                rows: total,
                split,
            });
            return Ok(picks);
        }
        // Commit the positions, lane by lane, and with a drafter append each lane's rows to its
        // slots' contexts right after (in lane order: a request a cut split gets each lane's part
        // in turn), the calls one-lane passes of the same rows make.
        for lane in &lanes {
            let ks = &mut kvs[lane.first..lane.first + lane.reqs.len()];
            for (kv, r) in ks.iter_mut().zip(&lane.reqs) {
                kv.tokens += r.rows;
            }
            if let Some(d) = self.draft.as_mut().filter(|_| taps) {
                let rows: Vec<(usize, usize)> = lane
                    .reqs
                    .iter()
                    .map(|r| (lane.base + r.row0, r.rows))
                    .collect();
                d.append(ks, &rows)?;
            }
        }
        if taps {
            self.mark(usize::MAX, "draft_append")?;
        }
        Ok(picks)
    }

    /// Calls in flight at most over `lanes` lanes: the backend's depth, and one a lane.
    fn depth(&self, lanes: usize) -> usize {
        self.experts.depth().clamp(1, lanes.max(1))
    }

    /// Layers `layers` over the lanes: per layer, each lane's attention sublayer, then its FFN,
    /// the lanes in turn. An MoE FFN's routed experts go to the backend (`submit`) and are
    /// collected later (`finish`): a lane's next attention waits for its own, at most the
    /// backend's depth are out at once, and the oldest is collected first. With several lanes,
    /// a lane's attention runs while earlier lanes' experts are out. With `heads` (decode and
    /// verify lanes), each lane's head runs once its last layer is complete, lane A's while lane
    /// B's last call is out.
    fn run_lanes(
        &mut self,
        mode: Mode,
        lanes: &mut [Lane],
        layers: Range<usize>,
        taps: bool,
        heads: bool,
    ) -> Result<()> {
        let depth = self.depth(lanes.len());
        let mut flight: VecDeque<usize> = VecDeque::new();
        let model = self.model.clone();
        if let Some(o) = self.ops.get_mut().as_mut() {
            // Prefill passes only.
            o.begin_pass(
                mode == Mode::Prefill,
                lanes.iter().map(|l| l.rows).collect(),
            );
        }
        if let Some(t) = self.trace.as_mut() {
            t.active = true;
            t.mode = mode;
            t.used = 0;
            t.recs = vec![vec![TraceRec::default(); lanes.len()]; model.shape.layers];
            t.starts.clear();
            t.loop_start = Instant::now();
            // The backend's own record of the pass's calls (`LaneTrace::wire`).
            self.experts.trace_begin();
        }
        let end = layers.end;
        for l in layers {
            let lw = &model.layers[l];
            if let Some(t) = self.trace.as_mut().filter(|t| t.active) {
                t.starts.push((l, Instant::now()));
            }
            for x in 0..lanes.len() {
                while lanes[x].exchange.is_some() {
                    let y = flight.pop_front().expect("a call in flight");
                    self.finish_lane(&mut lanes[y], y, mode)?;
                }
                self.use_lane(x);
                self.trace_event(l, x, At::AttnStart)?;
                self.attention(l, mode, &mut lanes[x], lw, taps)?;
                match &lw.ffn {
                    FfnW::Dense(m) => {
                        let s = &self.s;
                        let (out, post, comb): (*mut u16, *const f32, *const f32) =
                            (s.ffn_out.ptr(0), s.ffn_post.ptr(0), s.ffn_comb.ptr(0));
                        self.mlp(lanes[x].rows, m, out)?;
                        self.mark(l, "dense_mlp")?;
                        lanes[x].prev = Some((out, core::ptr::null(), post, comb));
                        self.ffn_taps(l, mode, &lanes[x])?;
                    }
                    FfnW::Moe {
                        router,
                        bias,
                        shared,
                    } => {
                        self.moe_router(&lanes[x], router, bias)?;
                        self.trace_event(l, x, At::RouterEnd)?;
                        let t = Instant::now();
                        self.moe_routes(l, &mut lanes[x])?;
                        self.trace_host(l, x, |r| &mut r.routes_ms, t);
                        while flight.len() >= depth {
                            let y = flight.pop_front().expect("a call in flight");
                            self.finish_lane(&mut lanes[y], y, mode)?;
                        }
                        let t = Instant::now();
                        self.moe_submit(l, &mut lanes[x])?;
                        self.trace_host(l, x, |r| &mut r.submit_ms, t);
                        if let Some(t) = self.trace.as_mut().filter(|t| t.active) {
                            t.recs[l][x].sent = Some(Instant::now());
                        }
                        flight.push_back(x);
                        // The shared expert: the GPU runs it while the call is out.
                        let shared_out: *mut u16 = self.s.ffn_out2.ptr(0);
                        self.trace_event(l, x, At::SharedStart)?;
                        self.mlp(lanes[x].rows, shared, shared_out)?;
                        self.mark(l, "shared")?;
                        self.trace_event(l, x, At::SharedEnd)?;
                        // Decode and verify in one lane: the GPU waits for the ranks; the next
                        // layer's first weights go to L2 meanwhile, on this stream
                        // (`crate::prefetch`; the head's after the last layer).
                        if self.cfg.l2_prefetch > 0 && mode != Mode::Prefill && lanes.len() == 1 {
                            let next = if l + 1 < end {
                                l + 1
                            } else {
                                model.layers.len()
                            };
                            self.prefetch
                                .layer(&self.stream, next, self.cfg.l2_prefetch)?;
                        }
                    }
                }
            }
        }
        if heads {
            for x in 0..lanes.len() {
                while lanes[x].exchange.is_some() {
                    let y = flight.pop_front().expect("a call in flight");
                    self.finish_lane(&mut lanes[y], y, mode)?;
                }
                self.lane_head(&lanes[x], x)?;
            }
        }
        while let Some(y) = flight.pop_front() {
            self.finish_lane(&mut lanes[y], y, mode)?;
        }
        if let Some(t) = self.trace.as_mut() {
            t.loop_end = Instant::now();
        }
        Ok(())
    }

    /// A lane's attention sublayer: the attention boundary (expanding its last FFN output), the
    /// drafter's taps of the lane's rows (`taps`), KDA or DSA, then the FFN boundary.
    fn attention(
        &mut self,
        l: usize,
        mode: Mode,
        lane: &mut Lane,
        lw: &LayerW,
        taps: bool,
    ) -> Result<()> {
        if self.boundary(
            lane.rows,
            &lw.attn_hc,
            &lw.input_norm,
            lane.cur,
            lane.prev,
            true,
        )? {
            lane.cur ^= 1;
        }
        if let Some(d) = self.draft.as_ref().filter(|_| taps) {
            // streams[cur]: this layer's input, the previous layer's completed output. At the
            // entry of layers 6, 15, 25, 34 and 43 their mean is a tap (glm53f-dflash README
            // step 1: SGLang captures before layer k + 1, contracted by the mean), into the
            // lane's rows of the pass.
            d.capture(
                l,
                lane.base,
                lane.rows,
                self.s.streams[lane.cur].ptr(0),
                &self.stream,
            )?;
            self.op("draft_capture")?;
        }
        self.mark(l, "attn_hc")?;
        match &lw.attn {
            AttnW::Kda(w) => self.kda(l, mode, lane.rows, lane.base, &lane.reqs, &lane.meta, w)?,
            AttnW::Dsa(w) => {
                if let Some((k, n)) = lane.continues {
                    self.continue_tail(l, k, n, lane.reqs.len())?;
                    self.op("dsa_continue_tail")?;
                }
                self.dsa(l, mode, lane.rows, lane.base, &lane.reqs, &lane.meta, w)?
            }
        }
        self.tap(l, TapPoint::AttnDone, mode, lane.rows, lane.cur)?;
        // FFN boundary, expanding the attention output.
        let s = &self.s;
        let a: Prev = (
            s.attn_out.ptr(0),
            core::ptr::null(),
            s.attn_post.ptr(0),
            s.attn_comb.ptr(0),
        );
        self.boundary(
            lane.rows,
            &lw.ffn_hc,
            &lw.post_attn_norm,
            lane.cur,
            Some(a),
            false,
        )?;
        lane.cur ^= 1;
        self.mark(l, "ffn_hc")
    }

    /// The active lane's first request continues the previous lane's request `k` (of `n`): before
    /// DSA layer `l`, its tail is the one the previous lane's part left at that layer (that lane
    /// ran the layer first, on the same stream). `nb`: the active lane's requests.
    fn continue_tail(&mut self, l: usize, k: usize, n: usize, nb: usize) -> Result<()> {
        debug_assert!(self.lane > 0, "a lane after the first is active");
        let j = self.model.shape.dsa_index[l].expect("a DSA layer");
        let before = self.scratch(self.lane - 1);
        self.s.batch_tails.copy_from(
            &self.stream,
            j * nb * TAIL,
            &before.batch_tails,
            (j * n + k) * TAIL,
            TAIL,
        )
    }

    /// Collect lane `y`'s call in flight (its pointers are its own, so lane `y`'s buffers need
    /// not be the active ones): its FFN output is complete for later work on the stream.
    fn finish_lane(&mut self, lane: &mut Lane, y: usize, mode: Mode) -> Result<()> {
        let ex = lane.exchange.take().expect("a call in flight");
        let st = self.stream.clone();
        let t = Instant::now();
        self.experts.finish(&ex.call(lane), &st)?;
        self.trace_host(ex.layer, y, |r| &mut r.finish_ms, t);
        if let Some(t) = self.trace.as_mut().filter(|t| t.active) {
            let r = &mut t.recs[ex.layer][y];
            if let Some(s) = r.sent {
                r.out_ms = s.elapsed().as_secs_f64() * 1e3;
            }
        }
        self.mark(ex.layer, "routed_wait")?;
        lane.prev = Some(ex.next);
        self.ffn_taps(ex.layer, mode, lane)
    }

    /// Taps after a lane's FFN (only one-lane passes have a tap).
    fn ffn_taps(&mut self, l: usize, mode: Mode, lane: &Lane) -> Result<()> {
        if self.tap.is_none() {
            return Ok(());
        }
        self.tap(l, TapPoint::FfnDone, mode, lane.rows, lane.cur)?;
        self.expand_to_tap(lane.rows, lane.cur, lane.prev.expect("the FFN output"))?;
        self.tap(l, TapPoint::LayerOut, mode, lane.rows, lane.cur)
    }

    /// The head over each lane's last streams, then the LM head and the argmax over the pass's
    /// logit rows (gathered into lane 0's buffers, in request order). Returns the picks when
    /// `logits`.
    fn head(&mut self, lanes: &[Lane], total: usize, logits: bool) -> Result<Vec<u32>> {
        let st = self.stream.clone();
        let lr: usize = lanes.iter().map(|l| l.logit_rows.len()).sum();
        // Decode and verify: every row is a logit row, read in place.
        let whole = lanes.len() == 1 && lr == total;
        for (i, lane) in lanes.iter().enumerate() {
            self.use_lane(i);
            let (bo, bo2, post, comb) = lane.prev.expect("the last layer's output");
            // SAFETY: scratch buffers sized for the lane's rows.
            launched(
                unsafe {
                    lffi::glm53f_hc_head(
                        self.s.streams[lane.cur].ptr(0),
                        bo,
                        bo2,
                        post,
                        comb,
                        self.model.head.norm.ptr(0),
                        self.s.head_out.ptr(0),
                        lane.rows as i32,
                        HIDDEN as i32,
                        st.raw().cast(),
                    )
                },
                "glm53f_hc_head",
            )?;
            if logits && !whole && !lane.logit_rows.is_empty() {
                let dst: *mut u8 = self
                    .scratch(0)
                    .head_sel
                    .byte_ptr(lane.logit_base * HIDDEN * 2);
                // SAFETY: rows of the head output; indices < the lane's rows; lane 0's head
                // buffer holds every logit row of the pass.
                launched(
                    unsafe {
                        ffi::glm53f_fwd_gather_rows(
                            self.s.head_out.ptr(0),
                            (HIDDEN * 2) as i64,
                            lane.meta.logit_rows,
                            dst,
                            (HIDDEN * 2) as i64,
                            lane.logit_rows.len() as i32,
                            (HIDDEN * 2) as i64,
                            st.raw(),
                        )
                    },
                    "gather logit rows",
                )?;
            }
        }
        self.use_lane(0);
        self.mark(usize::MAX, "head_hc")?;
        if !logits {
            return Ok(Vec::new());
        }
        let x: *const u16 = if whole {
            self.s.head_out.ptr(0)
        } else {
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
        self.argmax(lr)
    }

    /// Decode and verify lanes: lane `i`'s head over its rows (every one a logit row), into its
    /// rows of lane A's logits: the final mean and RMSNorm, then the LM head over the lane's rows
    /// alone, as a pass over the lane's requests runs it.
    fn lane_head(&mut self, lane: &Lane, i: usize) -> Result<()> {
        self.use_lane(i);
        let st = self.stream.clone();
        let (bo, bo2, post, comb) = lane.prev.expect("the last layer's output");
        // SAFETY: scratch buffers sized for the lane's rows.
        launched(
            unsafe {
                lffi::glm53f_hc_head(
                    self.s.streams[lane.cur].ptr(0),
                    bo,
                    bo2,
                    post,
                    comb,
                    self.model.head.norm.ptr(0),
                    self.s.head_out.ptr(0),
                    lane.rows as i32,
                    HIDDEN as i32,
                    st.raw().cast(),
                )
            },
            "glm53f_hc_head",
        )?;
        self.mark(usize::MAX, "head_hc")?;
        // Lane A's logit rows hold every row of the pass (checked before the pass).
        let out: *mut f32 = self.scratch(0).logits.ptr(lane.logit_base * VOCAB);
        let lm = self.model.head.lm_head.mat();
        unsafe {
            self.gemm.bf16(
                self.s.head_out.ptr(0),
                HIDDEN,
                0,
                &lm,
                lane.rows,
                out.cast(),
                VOCAB,
                0,
                true,
                &st,
            )
        }?;
        self.mark(usize::MAX, "lm_head")
    }

    /// The greedy picks of the pass's first `lr` logit rows (lane 0's buffers, which the pass's
    /// end leaves active).
    fn argmax(&mut self, lr: usize) -> Result<Vec<u32>> {
        self.use_lane(0);
        let st = self.stream.clone();
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
        let b = unsafe { std::slice::from_raw_parts_mut(ids.as_mut_ptr().cast::<u8>(), lr * 4) };
        self.s.next.download_bytes(&st, 0, b)?;
        self.mark(usize::MAX, "argmax")?;
        Ok(ids.into_iter().map(|x| x as u32).collect())
    }

    // ---- Lane trace ----------------------------------------------------------------------------

    /// Record a trace event on the stream (tracing a pass). The op profile's segments start
    /// right after `AttnStart` and `SharedStart`, and end at `RouterEnd` and `SharedEnd`.
    fn trace_event(&mut self, layer: usize, lane: usize, at: At) -> Result<()> {
        if let Some(o) = self.ops.get_mut().as_mut() {
            match at {
                At::RouterEnd | At::SharedEnd => o.end(),
                At::AttnStart | At::SharedStart => {
                    let sh = &self.model.shape;
                    let kind = LayerKind::new(sh.dsa_index[layer].is_some(), sh.is_moe(layer));
                    let seg = if matches!(at, At::AttnStart) {
                        Segment::Attention
                    } else {
                        Segment::Shared
                    };
                    o.begin(&self.stream, layer, lane, kind, seg)?;
                }
            }
        }
        if let Some(t) = self.trace.as_mut().filter(|t| t.active) {
            if t.used == t.events.len() {
                t.events.push(Event::new()?);
            }
            t.events[t.used].record(&self.stream)?;
            t.recs[layer][lane].events[at as usize] = Some(t.used);
            t.used += 1;
        }
        Ok(())
    }

    /// Add the host time since `since` to a lane's layer record (tracing a pass).
    fn trace_host(
        &mut self,
        layer: usize,
        lane: usize,
        field: fn(&mut TraceRec) -> &mut f64,
        since: Instant,
    ) {
        if let Some(t) = self.trace.as_mut().filter(|t| t.active) {
            *field(&mut t.recs[layer][lane]) += since.elapsed().as_secs_f64() * 1e3;
        }
    }

    /// Close a traced pass: its per-layer record, from the events (waits for the stream).
    fn finish_trace(&mut self, lanes: &[Lane]) -> Result<()> {
        let n = lanes.len();
        let depth = self.depth(n);
        let Some(t) = self.trace.as_mut().filter(|t| t.active) else {
            return Ok(());
        };
        t.active = false;
        self.stream.synchronize()?;
        let mut layers = Vec::new();
        for (i, &(l, at)) in t.starts.iter().enumerate() {
            if !self.model.shape.is_moe(l) {
                continue;
            }
            let next = t.starts.get(i + 1).map_or(t.loop_end, |s| s.1);
            let rec = &t.recs[l];
            let gpu = |a: At, b: At| -> Result<Vec<f64>> {
                (0..n)
                    .map(
                        |x| match (rec[x].events[a as usize], rec[x].events[b as usize]) {
                            (Some(i), Some(j)) => {
                                Ok(t.events[j].elapsed_ms_since(&t.events[i])? as f64)
                            }
                            _ => Ok(0.0),
                        },
                    )
                    .collect()
            };
            layers.push(LayerTrace {
                layer: l,
                wall_ms: (next - at).as_secs_f64() * 1e3,
                gpu_attn_ms: gpu(At::AttnStart, At::RouterEnd)?,
                gpu_shared_ms: gpu(At::SharedStart, At::SharedEnd)?,
                routes_ms: (0..n).map(|x| rec[x].routes_ms).collect(),
                submit_ms: (0..n).map(|x| rec[x].submit_ms).collect(),
                finish_ms: (0..n).map(|x| rec[x].finish_ms).collect(),
                out_ms: (0..n).map(|x| rec[x].out_ms).collect(),
            });
        }
        let trace = LaneTrace {
            mode: t.mode,
            rows: lanes.iter().map(|l| l.rows).collect(),
            requests: lanes.iter().map(|l| l.reqs.len()).collect(),
            depth,
            loop_ms: (t.loop_end - t.loop_start).as_secs_f64() * 1e3,
            layers,
            wire: self.experts.trace_end(),
            step: StepTimes::default(),
        };
        // Decode and verify passes print their step's line once it is complete (`pass`,
        // `commit`).
        if t.mode == Mode::Prefill {
            if t.print {
                eprintln!("PIPE {}", trace.summary());
            }
            t.last = Some(trace);
        } else {
            t.last_step = Some(trace);
        }
        Ok(())
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
            ws: &self.ws,
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
        // The collapsed row is only a tap's (the sublayer reads its RMSNorm).
        let collapsed: *mut u16 = if self.tap.is_some() {
            s.collapsed.ptr(0)
        } else {
            core::ptr::null_mut()
        };
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
                        collapsed,
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
            self.op(if attn {
                "attn_hc_boundary"
            } else {
                "ffn_hc_boundary"
            })?;
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
            self.op(if attn {
                "attn_hc_project"
            } else {
                "ffn_hc_project"
            })?;
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
                        collapsed,
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
            self.op(if attn {
                "attn_hc_finish"
            } else {
                "ffn_hc_finish"
            })?;
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

    /// A KDA layer over `rows` rows. A verify pass keeps each layer's projection rows and replay
    /// inputs for the commit at the pass's rows `base ..` (a lane's rows).
    #[allow(clippy::too_many_arguments)]
    fn kda(
        &mut self,
        l: usize,
        mode: Mode,
        rows: usize,
        base: usize,
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
            self.v.p.ptr((j * vr + base) * KDA_P_COLS)
        } else {
            self.s.p.ptr(0)
        };
        let x: *const u16 = self.s.normed.ptr(0);
        match &w.qkvb {
            ProjW::Bf16(m) => unsafe {
                self.gemm.bf16(
                    x,
                    HIDDEN,
                    0,
                    &m.mat(),
                    rows,
                    p.cast(),
                    KDA_P_COLS,
                    0,
                    false,
                    &st,
                )
            }?,
            // FP8 (D2): the normed rows as the DSA projections take them (BF16, or their E4M3 form).
            ProjW::Fp8(m) => unsafe {
                self.gemm
                    .fp8_kda(&self.normed_input(), &m.mat(), rows, p, &st)
            }?,
        }
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
                (j * vr + base) * KDA_P_COLS * 2,
                rows * KDA_P_COLS * 2,
            )?;
        }
        self.mark(l, "kda_proj")?;
        let pool = self.kv.shared.clone();
        let (sn, cn) = (
            KvLayout::state_elems_per_layer(),
            KvLayout::conv_elems_per_layer(),
        );
        // The layer's states: f32, or bf16 (D8) for the kernels' `_bf16state` variants.
        let bf16_state = pool.cfg.layout.kda_state_bf16;
        let state: *mut f32 = pool.state.ptr(j * sn);
        let state16: *mut u16 = pool.state.ptr(j * sn);
        let conv: *mut u16 = pool.conv.ptr(j * cn);
        let (ks, vs, gs, bs): (*mut f32, *mut u16, *mut f32, *mut f32) = if verify {
            (
                self.v.k.ptr((j * vr + base) * KDA_WIDTH),
                self.v.v.ptr((j * vr + base) * KDA_WIDTH),
                self.v.g.ptr((j * vr + base) * KDA_WIDTH),
                self.v.b.ptr((j * vr + base) * KDA_HEADS),
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
            if need > self.ws.kda.bytes() {
                // Only when the chunked kernel was switched on after construction.
                self.stream.synchronize()?;
                self.ws.kda = DeviceBuffer::alloc(need)?;
            }
            let max_rows = reqs.iter().map(|r| r.rows).max().unwrap_or(0) as i32;
            // SAFETY: as for the chain below; the workspace was sized above.
            launched(
                unsafe {
                    if bf16_state {
                        kffi::glm53f_kda_prefill_batch_bf16state(
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
                            state16,
                            state16,
                            meta.state_off,
                            w.a_log.ptr(0),
                            w.dt_bias.ptr(0),
                            w.o_norm.ptr(0),
                            RMS_EPS,
                            KDA_LOWER,
                            self.s.kda_out.ptr(0),
                            KDA_WIDTH as i64,
                            self.cfg.kda_prefill_value_blocks,
                            self.ws.kda.ptr(0),
                            self.ws.kda.bytes() as i64,
                            st.raw(),
                        )
                    } else {
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
                            self.ws.kda.ptr(0),
                            self.ws.kda.bytes() as i64,
                            st.raw(),
                        )
                    }
                },
                "glm53f_kda_prefill_batch",
            )?;
            self.mark(l, "kda_core")?;
            self.kda_o(rows, &w.o)?;
            self.mark(l, "kda_o")?;
            return Ok(());
        }
        // SAFETY: state and conv arenas hold every slot of the batch at its offsets; scratch
        // sized for `rows`; the saves for up to the verify capacity.
        launched(
            unsafe {
                if bf16_state {
                    kffi::glm53f_kda_chain_batch_bf16state(
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
                        state16,
                        if verify {
                            core::ptr::null_mut()
                        } else {
                            state16
                        },
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
                } else {
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
                }
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
        self.kda_o(rows, &w.o)?;
        self.mark(l, "kda_o")?;
        Ok(())
    }

    /// KDA's `o_proj` of the gated norm's output into the attention output: BF16 as shipped, or
    /// FP8 (D2) with the output's E4M3 form, when the GEMM takes one, in the DSA output's
    /// buffers (the two attention kinds never share a pass of a lane).
    fn kda_o(&mut self, rows: usize, o: &ProjW) -> Result<()> {
        let st = self.stream.clone();
        let s = &self.s;
        match o {
            ProjW::Bf16(m) => unsafe {
                self.gemm.bf16(
                    s.kda_out.ptr(0),
                    KDA_WIDTH,
                    0,
                    &m.mat(),
                    rows,
                    s.attn_out.ptr::<c_void>(0),
                    HIDDEN,
                    0,
                    false,
                    &st,
                )
            },
            ProjW::Fp8(m) => {
                if self.gemm.policy.needs_quant(rows, true) {
                    // SAFETY: kda_out [rows][8192]; o_q and o_s hold rows of 16,384 codes and 128
                    // scales.
                    unsafe {
                        act_quant(
                            s.kda_out.ptr(0),
                            s.o_q.ptr(0),
                            s.o_s.ptr(0),
                            rows,
                            KDA_WIDTH,
                            &st,
                        )
                    }?;
                }
                let x = Fp8Input {
                    bf16: s.kda_out.ptr(0),
                    q: s.o_q.ptr(0),
                    scales: s.o_s.ptr(0),
                };
                unsafe {
                    self.gemm
                        .fp8_kda(&x, &m.mat(), rows, s.attn_out.ptr(0), &st)
                }
            }
        }
    }

    /// A DSA layer over `rows` rows. A verify pass keeps each layer's raw index keys and gates for
    /// the commit at the pass's rows `base ..` (a lane's rows).
    #[allow(clippy::too_many_arguments)]
    fn dsa(
        &mut self,
        l: usize,
        mode: Mode,
        rows: usize,
        base: usize,
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
        self.op("dsa_q_a")?;
        unsafe { self.gemm.fp8(&x, &w.kv_a.mat(), rows, s.kva.ptr(0), &st) }?;
        self.op("dsa_kv_a")?;
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
        self.op("dsa_q_norm")?;
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
            self.op("dsa_q_quant")?;
        }
        let qr = Fp8Input {
            bf16: s.q_resid.ptr(0),
            q: s.q_resid_q.ptr(0),
            scales: s.q_resid_s.ptr(0),
        };
        unsafe { self.gemm.fp8(&qr, &w.q_b.mat(), rows, s.q16.ptr(0), &st) }?;
        self.op("dsa_q_b")?;
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
        self.op("dsa_idx_q")?;
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
        self.op("dsa_idx_proj")?;
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
                self.v.k_raw.ptr((j * vr + base) * INDEX_DIM),
                self.v.gate.ptr((j * vr + base) * INDEX_DIM),
            )
        } else {
            (s.k_raw.ptr(0), s.gate.ptr(0))
        };
        // (The absorb reads the MLA query in BF16.)
        widen(
            s.idx_q16.ptr(0),
            INDEX_HEADS * INDEX_DIM,
            s.idx_q32.ptr(0),
            INDEX_HEADS * INDEX_DIM,
            INDEX_HEADS * INDEX_DIM,
            1.0,
        )?;
        self.op("dsa_widen_idx_q")?;
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
        self.op("dsa_widen_small")?;
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
        self.op("dsa_latent_write")?;
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
        self.op("dsa_pool_write")?;
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
            self.op("dsa_tail_commit")?;
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
        // The workspaces were sized at construction for any pass; growing here would be a
        // planning error (kept as a fallback rather than a failed pass).
        if need > self.ws.idx.bytes() {
            self.stream.synchronize()?;
            self.ws.idx = DeviceBuffer::alloc(need)?;
        }
        let small = rows <= 8;
        let (splits, groups) = if small {
            (self.cfg.decode_splits, self.cfg.decode_head_groups)
        } else {
            (
                1,
                mla_head_groups(rows, self.sms, self.cfg.prefill_head_groups),
            )
        };
        // SAFETY: a host function.
        let mla_need =
            unsafe { dffi::glm53f_dsa_mla_workspace_bytes(rows as i32, splits as i32) } as usize;
        if mla_need > self.ws.mla.bytes() {
            self.stream.synchronize()?;
            self.ws.mla = DeviceBuffer::alloc(mla_need)?;
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
                    self.ws.idx.ptr(0),
                    self.ws.idx.bytes() as u64,
                    s.pools.ptr(0),
                    s.tokens.ptr(0),
                    s.counts.ptr(0),
                    core::ptr::null_mut(),
                    raw,
                )
            },
            "glm53f_dsa_index_select",
        )?;
        self.op("dsa_index_select")?;
        self.mark(l, "dsa_index")?;
        // Sparse MLA in latent space, in blocks of the workspaces' rows (a whole pass up to
        // `ForwardConfig::mla_block_rows`): absorb from the BF16 query, attend, un-absorb into the
        // BF16 per-head output. Every kernel of the core computes each row alone, and the plan
        // (`splits`, `groups`) is the pass's, so the blocks change no bit. Up to 8 rows the
        // decode un-absorb writes f32 and a copy rounds it; beyond, the prefill un-absorb writes
        // BF16 directly (and f32 for a tap: one block when a tap reads it).
        let ws = &self.ws;
        let block = ws.mla_rows.max(1);
        let ld = MLA_HEADS * QK_HEAD;
        let tap_o32: *mut f32 = if self.tap.is_some() {
            ws.o32.ptr(0)
        } else {
            core::ptr::null_mut()
        };
        let mut b0 = 0;
        while b0 < rows {
            let n = block.min(rows - b0);
            launched(
                unsafe {
                    dffi::glm53f_dsa_mla_absorb_q_bf16(
                        s.q16.ptr::<u16>(b0 * ld),
                        ld as i64,
                        n as i32,
                        w.kv_b.ptr(0),
                        ws.q_abs.ptr(0),
                        core::ptr::null_mut(),
                        raw,
                    )
                },
                "glm53f_dsa_mla_absorb_q_bf16",
            )?;
            self.op("dsa_absorb_q")?;
            launched(
                unsafe {
                    dffi::glm53f_dsa_mla_sparse_attn(
                        ws.q_abs.ptr(0),
                        s.tokens.ptr::<i32>(b0 * MAX_SELECTED),
                        MAX_SELECTED as i32,
                        s.counts.ptr::<i32>(2 * b0),
                        meta.row_req.wrapping_add(b0),
                        n as i32,
                        MLA_SCALE,
                        cache,
                        splits as i32,
                        groups as i32,
                        ws.mla.ptr(0),
                        ws.mla.bytes() as u64,
                        ws.o_lat.ptr(0),
                        ws.lse.ptr(0),
                        raw,
                    )
                },
                "glm53f_dsa_mla_sparse_attn",
            )?;
            self.op("dsa_sparse_attn")?;
            let o16: *mut u16 = s.o16.ptr(b0 * MLA_HEADS * V_HEAD);
            if small {
                launched(
                    unsafe {
                        dffi::glm53f_dsa_mla_unabsorb_v(
                            ws.o_lat.ptr(0),
                            w.kv_b.ptr(0),
                            n as i32,
                            ws.o32.ptr(0),
                            raw,
                        )
                    },
                    "glm53f_dsa_mla_unabsorb_v",
                )?;
                self.op("dsa_unabsorb_v")?;
                launched(
                    unsafe {
                        ffi::glm53f_fwd_f32_to_bf16(
                            ws.o32.ptr(0),
                            (MLA_HEADS * V_HEAD) as i64,
                            o16,
                            (MLA_HEADS * V_HEAD) as i64,
                            n as i32,
                            (MLA_HEADS * V_HEAD) as i32,
                            raw,
                        )
                    },
                    "glm53f_fwd_f32_to_bf16",
                )?;
                self.op("dsa_o_bf16")?;
            } else {
                launched(
                    unsafe {
                        dffi::glm53f_dsa_mla_unabsorb_v_rows(
                            ws.o_lat.ptr(0),
                            w.kv_b.ptr(0),
                            n as i32,
                            o16,
                            (MLA_HEADS * V_HEAD) as i64,
                            tap_o32,
                            raw,
                        )
                    },
                    "glm53f_dsa_mla_unabsorb_v_rows",
                )?;
                self.op("dsa_unabsorb_v")?;
            }
            b0 += n;
        }
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
            self.op("dsa_o_quant")?;
        }
        self.mark(l, "dsa_attn")?;
        let o = Fp8Input {
            bf16: s.o16.ptr(0),
            q: s.o_q.ptr(0),
            scales: s.o_s.ptr(0),
        };
        unsafe { self.gemm.fp8(&o, &w.o.mat(), rows, s.attn_out.ptr(0), &st) }?;
        self.op("dsa_o")?;
        self.mark(l, "dsa_o")?;
        Ok(())
    }

    // ---- FFN ---------------------------------------------------------------------------------

    /// A SwiGLU MLP of the normed rows into `out`.
    fn mlp(&mut self, rows: usize, m: &MlpW, out: *mut u16) -> Result<()> {
        let st = self.stream.clone();
        let quant = self.gemm.policy.fp8_needs_quant(rows);
        let s = &self.s;
        let ops = if m.inter == DENSE_INTER {
            ["dense_gate_up", "dense_swiglu", "dense_down"]
        } else {
            ["shared_gate_up", "shared_swiglu", "shared_down"]
        };
        unsafe {
            self.gemm.fp8(
                &self.normed_input(),
                &m.gate_up.mat(),
                rows,
                s.gu.ptr(0),
                &st,
            )
        }?;
        self.op(ops[0])?;
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
        self.op(ops[1])?;
        let a = Fp8Input {
            bf16: s.act.ptr(0),
            q: s.act_q.ptr(0),
            scales: s.act_s.ptr(0),
        };
        unsafe { self.gemm.fp8(&a, &m.down.mat(), rows, out, &st) }?;
        self.op(ops[2])
    }

    /// The router over the lane's rows (top-8 of 288 with the bias for the choice; weights x 2.5).
    fn moe_router(
        &mut self,
        lane: &Lane,
        router: &DeviceBuffer,
        bias: &DeviceBuffer,
    ) -> Result<()> {
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
                    lane.rows as i32,
                    EXPERTS as i32,
                    HIDDEN as i32,
                    TOP_K as i32,
                    ROUTED_SCALE,
                    self.stream.raw().cast(),
                )
            },
            "glm53f_router_fused",
        )?;
        self.op("router")
    }

    /// The routes to the host: the step's one host round trip (the backend builds its frames
    /// from them).
    fn moe_routes(&mut self, l: usize, lane: &mut Lane) -> Result<()> {
        let n = lane.rows * TOP_K;
        lane.host_ids.resize(n, 0);
        lane.host_weights.resize(n, 0.0);
        let st = self.stream.clone();
        // SAFETY: host vectors of rows x 8 values.
        let bi = unsafe {
            std::slice::from_raw_parts_mut(lane.host_ids.as_mut_ptr().cast::<u8>(), n * 4)
        };
        self.s.ids.download_bytes(&st, 0, bi)?;
        let bw = unsafe {
            std::slice::from_raw_parts_mut(lane.host_weights.as_mut_ptr().cast::<u8>(), n * 4)
        };
        self.s.weights.download_bytes(&st, 0, bw)?;
        self.mark(l, "router")
    }

    /// Submit the lane's routed experts to the backend. The call stays in flight until
    /// [`Self::finish_lane`] collects it, and names the FFN outputs the lane's next boundary
    /// expands (the routed sum and the shared expert's output).
    fn moe_submit(&mut self, l: usize, lane: &mut Lane) -> Result<()> {
        let s = &self.s;
        let ex = Exchange {
            layer: l,
            x: s.normed.ptr(0),
            x_q: s.normed_q.ptr(0),
            x_scales: s.normed_s.ptr(0),
            ids: s.ids.ptr(0),
            weights: s.weights.ptr(0),
            out: s.ffn_out.ptr(0),
            next: (
                s.ffn_out.ptr(0),
                s.ffn_out2.ptr(0),
                s.ffn_post.ptr(0),
                s.ffn_comb.ptr(0),
            ),
        };
        let st = self.stream.clone();
        self.experts.submit(&ex.call(lane), &st)?;
        lane.exchange = Some(ex);
        self.mark(l, "routed")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn req(slot: usize, rows: usize, row0: usize) -> Req {
        Req {
            slot,
            start: 100 * slot,
            rows,
            row0,
            state_off: 0,
            conv_off: 0,
            tail_row: 0,
        }
    }

    /// A lane's first request among the pass's, rows, `continues` and `goes_on`.
    type Place = (usize, Range<usize>, Option<(usize, usize)>, Option<usize>);

    /// A part's place in the pass and its requests' (slot, start, rows, row0).
    fn part(p: &Part) -> (Place, Vec<(usize, usize, usize, usize)>) {
        (
            (p.first, p.span.clone(), p.continues, p.goes_on),
            p.reqs
                .iter()
                .map(|r| (r.slot, r.start, r.rows, r.row0))
                .collect(),
        )
    }

    /// Two requests of 150 and 211 rows in four lanes (cuts at 91, 181 and 271): the first split
    /// once, the second across three lanes, its middle part a lane of its own; then cut between
    /// the requests (the decode lanes' cut), and not at all.
    #[test]
    fn a_pass_cut_into_lanes() {
        let reqs = [req(0, 150, 0), req(1, 211, 150)];
        let parts: Vec<_> = cut(&reqs, 361, &[91, 181, 271]).iter().map(part).collect();
        assert_eq!(
            parts,
            vec![
                ((0, 0..91, None, Some(0)), vec![(0, 0, 91, 0)]),
                (
                    (0, 91..181, Some((0, 1)), Some(1)),
                    vec![(0, 91, 59, 0), (1, 100, 31, 59)]
                ),
                ((1, 181..271, Some((1, 2)), Some(0)), vec![(1, 131, 90, 0)]),
                ((1, 271..361, Some((0, 1)), None), vec![(1, 221, 90, 0)]),
            ]
        );
        let parts: Vec<_> = cut(&reqs, 361, &[150]).iter().map(part).collect();
        assert_eq!(
            parts,
            vec![
                ((0, 0..150, None, None), vec![(0, 0, 150, 0)]),
                ((1, 150..361, None, None), vec![(1, 100, 211, 0)]),
            ]
        );
        let parts: Vec<_> = cut(&reqs, 361, &[]).iter().map(part).collect();
        assert_eq!(
            parts,
            vec![(
                (0, 0..361, None, None),
                vec![(0, 0, 150, 0), (1, 100, 211, 150)]
            )]
        );
    }

    /// Two lanes follow the rule of the two lanes before: a second lane from twice `min_rows`
    /// rows, or when one lane does not hold the pass. More lanes: one per `min_rows` rows, and as
    /// many as the rows need.
    #[test]
    fn the_lanes_a_pass_runs_in() {
        for (min, first, rows) in [(8, 33, 33), (16, 64, 32), (64, 2048, 2048), (1, 5, 5)] {
            for total in 1..=2 * rows {
                let two = 1 + usize::from(total > first || total >= 2 * min);
                assert_eq!(lanes_for(total, 2, min, first, rows), two, "{total} rows");
            }
        }
        // Lanes of 2,048 rows, a lane per 64 rows.
        let four = |t: usize| lanes_for(t, 4, 64, 2048, 2048);
        assert_eq!(
            [1, 127, 128, 191, 192, 256, 8192].map(four),
            [1, 1, 2, 2, 3, 4, 4]
        );
        // Lane 0 holds a larger verify pass (256 rows), the others 100 rows each: 250 rows fit in
        // one lane, 300 need three lanes.
        let small = |t: usize| lanes_for(t, 4, 1000, 256, 100);
        assert_eq!([250, 257, 300, 301].map(small), [1, 3, 3, 4]);
        // Four lanes of 100 hold 400 rows; the caller refuses more.
        assert_eq!(small(401), 4);
    }

    /// Passes of more than 8 rows take the fewest head groups per block whose grid (`rows * 4 /
    /// groups` blocks) fits one wave of the multiprocessors, and never more than the configured
    /// number: 1 for up to a quarter as many rows as the GPU has multiprocessors, 2 up to half
    /// as many, then the prefill setting.
    #[test]
    fn the_mla_block_is_whole_waves() {
        assert_eq!(mla_block(512, 128), 512);
        assert_eq!(mla_block(512, 170), 510);
        assert_eq!(mla_block(340, 170), 340);
        assert_eq!(mla_block(512, 84), 504);
        assert_eq!(mla_block(100, 170), 100);
        assert_eq!(mla_block(8, 5), 8);
        for sms in [1, 5, 46, 84, 128, 132, 170, 188] {
            for cap in 8..=1024 {
                let b = mla_block(cap, sms);
                let s = sms as usize;
                assert!((8..=cap).contains(&b), "{cap} rows, {sms} SMs: {b}");
                // Whole waves, as many as fit, whenever a wave fits in the cap and is 8 rows or
                // more; the cap itself otherwise.
                if cap / s * s >= 8 {
                    assert!(b % s == 0 && b + s > cap, "{cap} rows, {sms} SMs: {b}");
                } else {
                    assert_eq!(b, cap);
                }
            }
        }
    }

    #[test]
    fn the_head_groups_of_a_pass() {
        let rows = [
            9, 32, 33, 42, 43, 64, 65, 85, 86, 128, 170, 171, 2048, 100_000,
        ];
        // 128 multiprocessors (the RTX 4090) and 170 (the RTX 5090).
        assert_eq!(
            rows.map(|r| mla_head_groups(r, 128, 4)),
            [1, 1, 2, 2, 2, 2, 4, 4, 4, 4, 4, 4, 4, 4]
        );
        assert_eq!(
            rows.map(|r| mla_head_groups(r, 170, 4)),
            [1, 1, 1, 1, 2, 2, 2, 2, 4, 4, 4, 4, 4, 4]
        );
        // A configured 2 or 1 is the cap.
        assert_eq!(
            rows.map(|r| mla_head_groups(r, 170, 2)),
            [1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]
        );
        assert_eq!(rows.map(|r| mla_head_groups(r, 170, 1)), [1; 14]);
        // Whatever the GPU: the grid fits one wave unless the configured groups make it, and one
        // group fewer would not fit.
        for sms in [1, 2, 24, 84, 128, 170, 188] {
            for cap in [1, 2, 4] {
                for rows in 9..600 {
                    let g = mla_head_groups(rows, sms, cap);
                    let fits = |g: usize| rows * 4 / g <= sms as usize;
                    let what = format!("{rows} rows, {sms} SMs, cap {cap}");
                    assert!([1, 2, 4].contains(&g) && g <= cap, "{what}");
                    assert!(g == cap || fits(g), "{what}");
                    assert!(g == 1 || !fits(g / 2), "{what}");
                }
            }
        }
    }
}
