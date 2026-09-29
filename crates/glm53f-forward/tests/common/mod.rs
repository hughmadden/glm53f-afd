//! Shared setup for the GPU tests: where the weights and goldens are, a small forward over
//! the first decoder layers, the golden-routing expert wrapper, and error statistics.
//!
//! - `GLM53F_CHECKPOINT_DIR` (or `GLM53F_CHECKPOINT`): the official checkpoint, or its
//!   coordinator subset (the non-expert tensors with their original names);
//! - `GLM53F_EXPERTS_DIR`: the routed experts of the MoE layers run (the official checkpoint,
//!   or a subset); defaults to the checkpoint directory;
//! - `GLM53F_GOLDENS`: the oracle's golden sets (default `oracle/goldens` in this repository);
//! - `GLM53F_TEST_NUMERICS`: numerics under test for every forward these helpers build, a
//!   comma-separated list of `kda-fp8`, `kda-fp8-pow2` (FP8 KDA projections with power-of-two
//!   block-128 scales), `kda-mxfp8` (with MXFP8 scales), `kda-state-bf16`, `prefill-w8a16` and
//!   `kda-prefill-w8a8` (default none), so a whole suite can run with an option on ([`numerics`]);
//!   and `l2-prefetch`, which changes no bit (`ForwardConfig::l2_prefetch` at
//!   [`TEST_L2_PREFETCH`]).
//!
//! Anything missing makes the tests print why and pass.
#![allow(dead_code)]

use std::collections::{HashMap, VecDeque};
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use glm53f_dsa::cache;
use glm53f_forward::device::{self, DeviceBuffer, Stream};
use glm53f_forward::embed::HostEmbedding;
use glm53f_forward::experts::{ExpertBackend, ExpertCall, LocalFp8Experts};
use glm53f_forward::forward::{ForwardConfig, GlmForward};
use glm53f_forward::gemm::Fp8Act;
use glm53f_forward::kv::{GlmKv, KvConfig, KvPool};
use glm53f_forward::kvplan::KvLayout;
use glm53f_forward::shape::{ModelShape, INDEX_DIM, TOP_K, VOCAB};
use glm53f_forward::weights::{open_checkpoint, DeviceModel, WeightOptions};
use glm53f_forward::{Fp8Scales, Result};
use glm53f_layers::testkit::goldens::{self, GoldenSet};
use glm53f_layers::testkit::json;

pub fn env_dir(keys: &[&str]) -> Option<PathBuf> {
    keys.iter().find_map(std::env::var_os).map(PathBuf::from)
}

pub fn checkpoint_dir() -> Option<PathBuf> {
    let d = env_dir(&["GLM53F_CHECKPOINT_DIR", "GLM53F_CHECKPOINT"]);
    if d.is_none() {
        eprintln!("skip: GLM53F_CHECKPOINT_DIR is not set");
    }
    d
}

pub fn experts_dir() -> Option<PathBuf> {
    env_dir(&["GLM53F_EXPERTS_DIR"]).or_else(checkpoint_dir)
}

/// The L2 prefetch the `l2-prefetch` option turns on (bytes).
pub const TEST_L2_PREFETCH: usize = 48 << 20;

/// Numerics under test for the forwards the tests build (`GLM53F_TEST_NUMERICS`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct TestNumerics {
    pub kda_fp8: bool,
    /// The FP8 KDA projections' scales (`kda-fp8`: the checkpoint's; `kda-fp8-pow2`,
    /// `kda-mxfp8`).
    pub kda_scales: Fp8Scales,
    pub kda_state_bf16: bool,
    pub prefill_w8a16: bool,
    pub kda_prefill_w8a8: bool,
    /// Decode and verify passes prefetch the next layer's weights into L2.
    pub l2_prefetch: bool,
}

impl TestNumerics {
    /// The target's arithmetic is the default one: `l2-prefetch` changes no bit of what the
    /// target computes.
    pub fn default_arithmetic(&self) -> bool {
        let target = TestNumerics {
            l2_prefetch: false,
            ..*self
        };
        target == TestNumerics::default()
    }

    /// The weights' load-time options.
    pub fn weights(&self) -> WeightOptions {
        WeightOptions {
            kda_fp8: self.kda_fp8,
            kda_scales: self.kda_scales,
        }
    }

    /// FP8 KDA projections with the checkpoint's `amax / 448` scales (D2 as first built), whose
    /// error the model-path tests bound more loosely.
    pub fn kda_fp8_amax(&self) -> bool {
        self.kda_fp8 && self.kda_scales == Fp8Scales::Block128
    }

    /// GiB of weights the FP8 KDA projections save over all 34 KDA layers (0 without them).
    pub fn kda_saved_gib(&self) -> f64 {
        match (self.kda_fp8, self.kda_scales) {
            (false, _) => 0.0,
            (true, Fp8Scales::Mx32) => 4.13,
            (true, _) => 4.26,
        }
    }

    /// `l` with the KDA states in BF16 when asked for.
    pub fn layout(&self, l: KvLayout) -> KvLayout {
        l.with_kda_state_bf16(self.kda_state_bf16)
    }

    /// `cfg` with the W8A16 prefill path (and the KDA projections' exception) and the L2
    /// prefetch when asked for.
    pub fn config(&self, mut cfg: ForwardConfig) -> ForwardConfig {
        cfg.policy.prefill_w8a16 |= self.prefill_w8a16;
        cfg.policy.kda_prefill_w8a8 |= self.kda_prefill_w8a8;
        if self.l2_prefetch {
            cfg.l2_prefetch = TEST_L2_PREFETCH;
        }
        cfg
    }
}

/// `GLM53F_TEST_NUMERICS` (panics on an unknown name, so a typo cannot pass silently).
pub fn numerics() -> TestNumerics {
    let mut n = TestNumerics::default();
    for name in std::env::var("GLM53F_TEST_NUMERICS")
        .unwrap_or_default()
        .split(',')
        .map(str::trim)
        .filter(|x| !x.is_empty())
    {
        match name {
            "kda-fp8" => n.kda_fp8 = true,
            "kda-fp8-pow2" => (n.kda_fp8, n.kda_scales) = (true, Fp8Scales::Block128Pow2),
            "kda-mxfp8" => (n.kda_fp8, n.kda_scales) = (true, Fp8Scales::Mx32),
            "kda-state-bf16" => n.kda_state_bf16 = true,
            "prefill-w8a16" => n.prefill_w8a16 = true,
            "kda-prefill-w8a8" => n.kda_prefill_w8a8 = true,
            "l2-prefetch" => n.l2_prefetch = true,
            other => panic!("GLM53F_TEST_NUMERICS: unknown option {other:?}"),
        }
    }
    if n != TestNumerics::default() {
        eprintln!("numerics under test: {n:?}");
    }
    n
}

/// Golden routes per MoE layer: (ids, weights) of every row, in order.
pub type RouteQueue = HashMap<usize, VecDeque<(Vec<i32>, Vec<f32>)>>;
/// Each MoE call's own routes: (layer, ids, weights).
pub type Seen = Arc<Mutex<Vec<(usize, Vec<i32>, Vec<f32>)>>>;

pub fn goldens_root() -> PathBuf {
    env_dir(&["GLM53F_GOLDENS"])
        .unwrap_or_else(|| PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../oracle/goldens"))
}

/// A GPU with at least `gib` GiB free, or None (printed).
pub fn gpu_with(gib: f64) -> bool {
    if device::device_count() == 0 {
        eprintln!("skip: no CUDA device");
        return false;
    }
    let (free, _) = device::mem_info().unwrap();
    let have = free as f64 / (1u64 << 30) as f64;
    if have < gib {
        eprintln!("skip: {have:.1} GiB free on the GPU, the test needs {gib:.1}");
        return false;
    }
    true
}

/// The oracle's golden sets, by name, when their payloads are present.
pub struct Goldens {
    pub sets: HashMap<String, GoldenSet>,
}

impl Goldens {
    pub fn load(names: &[&str]) -> Option<Goldens> {
        let root = goldens_root();
        let mut sets = HashMap::new();
        for d in goldens::discover(&root) {
            let name = d.file_name().unwrap().to_string_lossy().to_string();
            if !names.contains(&name.as_str()) {
                continue;
            }
            let set = GoldenSet::load(&d).ok()?;
            // The payloads are not in git: a set without its files is absent.
            if set.entries.iter().any(|e| !d.join(&e.file).is_file()) {
                eprintln!("skip: {name}: golden payloads missing (regenerate with the oracle)");
                return None;
            }
            sets.insert(name, set);
        }
        for n in names {
            if !sets.contains_key(*n) {
                eprintln!("skip: golden set {n} not found under {}", root.display());
                return None;
            }
        }
        Some(Goldens { sets })
    }

    fn entry(&self, set: &str, name: &str) -> (&GoldenSet, &goldens::GoldenEntry) {
        let s = &self.sets[set];
        let e = s
            .get(name)
            .unwrap_or_else(|| panic!("{set}: no tensor {name}"));
        (s, e)
    }

    pub fn f32(&self, set: &str, name: &str) -> Vec<f32> {
        let (s, e) = self.entry(set, name);
        s.read_f32(e).unwrap()
    }

    pub fn has(&self, set: &str, name: &str) -> bool {
        self.sets[set].get(name).is_some()
    }

    pub fn i64(&self, set: &str, name: &str) -> Vec<i64> {
        let (s, e) = self.entry(set, name);
        s.read_i64(e).unwrap()
    }

    pub fn shape(&self, set: &str, name: &str) -> Vec<usize> {
        self.entry(set, name).1.shape.clone()
    }

    /// The prompt and decode token ids from a set's manifest.
    pub fn token_ids(&self, set: &str) -> (Vec<u32>, Vec<u32>) {
        let path = self.sets[set].dir.join("manifest.json");
        let text = std::fs::read_to_string(&path).unwrap();
        let j = json::parse(&text).unwrap();
        let p = j
            .get("source")
            .and_then(|s| s.get("prompt"))
            .expect("manifest source.prompt");
        let ids = |k: &str| -> Vec<u32> {
            p.get(k)
                .and_then(|v| v.as_array())
                .unwrap_or_else(|| panic!("manifest source.prompt.{k}"))
                .iter()
                .map(|x| x.as_u64().unwrap() as u32)
                .collect()
        };
        (ids("token_ids"), ids("decode_token_ids"))
    }
}

/// Where a pass of `total` rows in `n` lanes cuts them (the forward's rule, `forward.rs` "Lanes
/// (prefill)"): lane i starts at row `ceil(i total / n)`.
pub fn lane_cuts(total: usize, n: usize) -> Vec<usize> {
    (1..n).map(|i| (i * total).div_ceil(n)).collect()
}

/// The rows of each lane of a pass of `total` rows in `n` lanes.
pub fn lane_rows(total: usize, n: usize) -> Vec<usize> {
    let mut at = lane_cuts(total, n);
    at.push(total);
    at.iter()
        .scan(0, |lo, &hi| {
            let r = hi - *lo;
            *lo = hi;
            Some(r)
        })
        .collect()
}

/// Prefill `prompts` into `kvs` (a slot each) as one-lane passes over their rows laid end to end,
/// cut at the pass rows `cuts` (ascending): each pass takes every prompt's rows in its range, as
/// the forward's lanes of one pass hold them. Returns each prompt's pick and the logits of its
/// last row, in prompt order.
pub fn prefill_in_passes(
    fwd: &mut GlmForward,
    kvs: &mut [GlmKv],
    prompts: &[&[u32]],
    cuts: &[usize],
) -> (Vec<u32>, Vec<f32>) {
    let total: usize = prompts.iter().map(|p| p.len()).sum();
    let mut picks = vec![0u32; prompts.len()];
    let mut logits = vec![0f32; prompts.len() * VOCAB];
    let mut lo = 0;
    for hi in cuts.iter().copied().chain(std::iter::once(total)) {
        // Each prompt's rows in [lo, hi), from its own first row.
        let (mut segs, mut row) = (Vec::new(), 0);
        for (i, p) in prompts.iter().enumerate() {
            let (a, b) = (lo.max(row), hi.min(row + p.len()));
            if a < b {
                segs.push((i, a - row..b - row));
            }
            row += p.len();
        }
        let first = segs[0].0;
        let got = {
            let mut pass: Vec<(&mut GlmKv, &[u32])> = kvs[first..first + segs.len()]
                .iter_mut()
                .zip(&segs)
                .map(|(k, (i, r))| (k, &prompts[*i][r.clone()]))
                .collect();
            fwd.prefill(&mut pass).unwrap()
        };
        let l = fwd.logits(segs.len()).unwrap();
        for (k, (i, r)) in segs.iter().enumerate() {
            if r.end == prompts[*i].len() {
                picks[*i] = got[k];
                logits[i * VOCAB..(i + 1) * VOCAB].copy_from_slice(&l[k * VOCAB..(k + 1) * VOCAB]);
            }
        }
        lo = hi;
    }
    (picks, logits)
}

/// What a slot keeps, as bytes: every KDA layer's state and conv window, every DSA layer's tail
/// (its valid tokens) and the committed rows of its pages (latent records, complete pools' keys
/// and scales).
pub fn kept(fwd: &GlmForward, kv: &GlmKv) -> Vec<u8> {
    let shape = fwd.shape();
    let mut out = Vec::new();
    for j in 0..shape.kda_layers {
        out.extend(
            kv.download_state(j)
                .unwrap()
                .iter()
                .flat_map(|x| x.to_le_bytes()),
        );
        out.extend(
            kv.download_conv(j)
                .unwrap()
                .iter()
                .flat_map(|x| x.to_le_bytes()),
        );
    }
    let t = kv.tokens();
    for j in 0..shape.dsa_layers {
        let tail = kv.download_tail(j).unwrap();
        let n = u32::from_le_bytes(tail[..4].try_into().unwrap()) as usize;
        out.extend(&tail[..16 + n.min(3) * cache::TAIL_TOKEN_BYTES]);
        for p in 0..t.div_ceil(cache::PAGE_TOKENS) {
            let b = kv.download_page_block(p, j).unwrap();
            let rows = (t - p * cache::PAGE_TOKENS).min(cache::PAGE_TOKENS);
            let pools = rows / 4;
            out.extend(&b[..rows * cache::LATENT_RECORD_BYTES]);
            out.extend(
                &b[cache::PAGE_POOL_CODES_OFFSET
                    ..cache::PAGE_POOL_CODES_OFFSET + pools * INDEX_DIM],
            );
            out.extend(
                &b[cache::PAGE_POOL_SCALES_OFFSET..cache::PAGE_POOL_SCALES_OFFSET + pools * 4],
            );
        }
    }
    out
}

/// Error of `got` against `want`.
#[derive(Clone, Copy, Debug, Default)]
pub struct Err {
    pub rel_rms: f64,
    pub max_abs: f64,
}

pub fn err(got: &[f32], want: &[f32]) -> Err {
    assert_eq!(got.len(), want.len(), "compared tensors differ in size");
    Err {
        rel_rms: glm53f_forward::reference::rel_rms(got, want),
        max_abs: glm53f_forward::reference::max_abs(got, want),
    }
}

pub fn widen(v: &[u16]) -> Vec<f32> {
    glm53f_layers::bf16::widen(v)
}

pub fn narrow(v: &[f32]) -> Vec<u16> {
    glm53f_layers::bf16::narrow(v)
}

/// Columns `[lo, hi)` of rows of `width`.
pub fn cols<T: Copy>(v: &[T], width: usize, lo: usize, hi: usize) -> Vec<T> {
    v.chunks_exact(width)
        .flat_map(|r| r[lo..hi].to_vec())
        .collect()
}

/// Rows `[lo, hi)` of rows of `width`.
pub fn rows<T: Copy>(v: &[T], width: usize, lo: usize, hi: usize) -> Vec<T> {
    v[lo * width..hi * width].to_vec()
}

/// The golden-routing wrapper: MoE calls take their routes from a queue of rows per layer (the
/// oracle README: routing sits on near-ties, so the expert path is checked with the golden
/// routes), each call the next `rows` rows whatever the passes' sizes, and every call's own
/// routes are recorded.
pub struct GoldenRoutes<B: ExpertBackend> {
    pub inner: B,
    /// Per layer: (ids, weights) of every row in order.
    pub queue: RouteQueue,
    pub seen: Seen,
    ids: DeviceBuffer,
    weights: DeviceBuffer,
}

impl<B: ExpertBackend> GoldenRoutes<B> {
    /// The next `rows` golden rows of `layer`.
    fn take(&mut self, layer: usize, rows: usize) -> Option<(Vec<i32>, Vec<f32>)> {
        let q = self.queue.get_mut(&layer)?;
        let (mut ids, mut w) = (Vec::new(), Vec::new());
        while ids.len() < rows * TOP_K {
            let (mut a, mut b) = q.pop_front().expect("golden routes ran out");
            let need = rows * TOP_K - ids.len();
            if a.len() > need {
                q.push_front((a.split_off(need), b.split_off(need)));
            }
            ids.extend(a);
            w.extend(b);
        }
        Some((ids, w))
    }
}

impl<B: ExpertBackend> GoldenRoutes<B> {
    pub fn new(inner: B, max_rows: usize) -> Self {
        GoldenRoutes {
            inner,
            queue: HashMap::new(),
            seen: Arc::new(Mutex::new(Vec::new())),
            ids: DeviceBuffer::alloc(max_rows * TOP_K * 4).unwrap(),
            weights: DeviceBuffer::alloc(max_rows * TOP_K * 4).unwrap(),
        }
    }
}

impl<B: ExpertBackend> ExpertBackend for GoldenRoutes<B> {
    fn submit(&mut self, call: &ExpertCall<'_>, stream: &Stream) -> Result<()> {
        self.seen.lock().unwrap().push((
            call.layer,
            call.host_ids.to_vec(),
            call.host_weights.to_vec(),
        ));
        let Some((ids, w)) = self.take(call.layer, call.rows) else {
            return self.inner.submit(call, stream);
        };
        self.ids.upload_async(stream, 0, &ids)?;
        self.weights.upload_async(stream, 0, &w)?;
        let c = ExpertCall {
            ids: self.ids.ptr(0),
            weights: self.weights.ptr(0),
            host_ids: &ids,
            host_weights: &w,
            ..*call
        };
        self.inner.submit(&c, stream)?;
        // The combine read the device copies; hold the call's routes until the stream passed it.
        stream.synchronize()
    }

    fn finish(&mut self, call: &ExpertCall<'_>, stream: &Stream) -> Result<()> {
        self.inner.finish(call, stream)
    }

    fn depth(&self) -> usize {
        self.inner.depth()
    }
}

/// A forward over decoder layers `0 .. layers` with local FP8 experts behind golden routing.
pub struct Setup {
    pub fwd: GlmForward,
    pub seen: Seen,
}

pub fn forward(
    layers: usize,
    cfg: ForwardConfig,
    golden_routes: RouteQueue,
    expert_budget: usize,
) -> Option<Setup> {
    let dir = checkpoint_dir()?;
    let edir = experts_dir()?;
    let (mcfg, ckpt) = match open_checkpoint(&dir) {
        Ok(x) => x,
        Err(e) => {
            eprintln!("skip: cannot open the checkpoint: {e}");
            return None;
        }
    };
    let shape = ModelShape::new(&mcfg.text, layers).unwrap();
    let t0 = std::time::Instant::now();
    let num = numerics();
    let model = DeviceModel::load_with(&ckpt, &shape, layers, num.weights()).unwrap();
    let embed = HostEmbedding::load(&ckpt).unwrap();
    eprintln!(
        "loaded layers 0..{layers} and the head: {:.2} GB on the GPU, embedding {:.2} GB in host RAM, {:.1} s",
        model.bytes as f64 / 1e9,
        embed.bytes() as f64 / 1e9,
        t0.elapsed().as_secs_f64()
    );
    let stream = Arc::new(Stream::new().unwrap());
    let layout = num.layout(KvLayout::new(&shape, None));
    let kv = KvPool::new(
        KvConfig {
            layout,
            max_slots: 6,
            // 64 pages of rows, and room for four snapshot marks (they take pool pages).
            pages: 64 + 4 * layout.mark_pages(),
            max_pages: 16,
            base_pages: 1,
        },
        stream.clone(),
    )
    .unwrap();
    // A budget of 0: routed outputs of zeros (tests of the coordinator path alone).
    let (experts, seen): (Box<dyn ExpertBackend>, _) = if expert_budget == 0 {
        (
            Box::new(glm53f_forward::experts::ZeroExperts),
            Arc::new(Mutex::new(Vec::new())),
        )
    } else {
        let local = LocalFp8Experts::new(
            &edir,
            expert_budget,
            cfg.max_rows.max(cfg.max_verify_rows),
            &stream,
            Fp8Act::Bf16,
        )
        .unwrap();
        let mut gr = GoldenRoutes::new(local, cfg.max_rows.max(cfg.max_verify_rows));
        gr.queue = golden_routes;
        let seen = gr.seen.clone();
        (Box::new(gr), seen)
    };
    let fwd = GlmForward::new(model, embed, kv, experts, num.config(cfg)).unwrap();
    Some(Setup { fwd, seen })
}

/// A forward over decoder layers `0 .. layers` with the experts `experts` makes (given the
/// forward's stream), a pool of `pages` pages (page tables of `max_pages`) and `slots` slots:
/// for tests that size the pool themselves.
pub fn forward_with(
    layers: usize,
    cfg: ForwardConfig,
    experts: impl FnOnce(&Arc<Stream>) -> Box<dyn ExpertBackend>,
    slots: usize,
    pages: usize,
    max_pages: usize,
) -> Option<GlmForward> {
    let dir = checkpoint_dir()?;
    let (mcfg, ckpt) = match open_checkpoint(&dir) {
        Ok(x) => x,
        Err(e) => {
            eprintln!("skip: cannot open the checkpoint: {e}");
            return None;
        }
    };
    let shape = ModelShape::new(&mcfg.text, layers).unwrap();
    let num = numerics();
    let model = DeviceModel::load_with(&ckpt, &shape, layers, num.weights()).unwrap();
    let embed = HostEmbedding::load(&ckpt).unwrap();
    let stream = Arc::new(Stream::new().unwrap());
    let kv = KvPool::new(
        KvConfig {
            layout: num.layout(KvLayout::new(&shape, None)),
            max_slots: slots,
            pages,
            max_pages,
            base_pages: 0,
        },
        stream.clone(),
    )
    .unwrap();
    let experts = experts(&stream);
    Some(GlmForward::new(model, embed, kv, experts, num.config(cfg)).unwrap())
}
