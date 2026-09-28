//! The scorer on the GPU (feature `cuda`): the model loaded and the experts connected as
//! `glm53f-serve` does it, the forward built as `glm53f-serve --prefill-rows R --prefill-lanes N`
//! builds it (one slot, no drafter), and one window scored into its file.

use std::path::Path;
use std::sync::Arc;
use std::time::Instant;

use glm53f_forward::cuda::{ATTR_CC_MAJOR, ATTR_CC_MINOR};
use glm53f_forward::device::{self, Stream};
use glm53f_forward::embed::HostEmbedding;
use glm53f_forward::experts::{ExpertBackend, ExpertCall, LocalFp8Experts, ZeroExperts};
use glm53f_forward::forward::{ForwardBuffers, ForwardConfig, GlmForward};
use glm53f_forward::gemm::{self, GemmPolicy};
use glm53f_forward::kv::{KvConfig, KvPool};
use glm53f_forward::kvplan::KvLayout;
use glm53f_forward::remote::RemoteExperts;
use glm53f_forward::shape::{ModelShape, EXPERTS, VOCAB};
use glm53f_forward::weights::{open_checkpoint, DeviceModel};
use glm53f_model::catalog::LAYERS;
use glm53f_model::safetensors::Checkpoint;

use crate::out::RowWriter;
use crate::plan::Window;
use crate::{engine_line, Build, Experts, Fp8Act, Options};

const GIB: f64 = (1u64 << 30) as f64;

fn s<T>(r: glm53f_forward::Result<T>) -> Result<T, String> {
    r.map_err(|e| e.to_string())
}

/// The forward's configuration: `glm53f-serve`'s with `--prefill-rows` = the pass rows and its
/// default 16 requests a pass (which size only per-pass metadata and the chunked KDA prefill's
/// workspace), and the numerics options.
pub fn forward_config(o: &Options) -> ForwardConfig {
    ForwardConfig {
        max_rows: o.pass_rows,
        lanes: o.lanes,
        max_requests: 16,
        kda_chunked_prefill: o.kda_chunked_prefill,
        policy: GemmPolicy {
            fp8_act: fp8_act(o.fp8_act),
            prefill_promote_k32: o.promote_k32,
            ..GemmPolicy::default()
        },
        ..ForwardConfig::default()
    }
}

fn fp8_act(a: Fp8Act) -> gemm::Fp8Act {
    match a {
        Fp8Act::Bf16 => gemm::Fp8Act::Bf16,
        Fp8Act::Dynamic => gemm::Fp8Act::Dynamic128,
    }
}

/// Local FP8 experts for the MoE layers the experts directory holds, routed outputs of zeros
/// for the others (a development run).
struct PartialExperts {
    local: LocalFp8Experts,
    /// Per decoder layer: its experts are in the directory.
    have: Vec<bool>,
}

impl ExpertBackend for PartialExperts {
    fn submit(&mut self, call: &ExpertCall<'_>, stream: &Stream) -> glm53f_forward::Result<()> {
        if self.have[call.layer] {
            self.local.submit(call, stream)
        } else {
            ZeroExperts.submit(call, stream)
        }
    }

    fn finish(&mut self, call: &ExpertCall<'_>, stream: &Stream) -> glm53f_forward::Result<()> {
        if self.have[call.layer] {
            self.local.finish(call, stream)
        } else {
            ZeroExperts.finish(call, stream)
        }
    }

    /// Both enqueue a call's work within `submit` and `finish`, on the stream.
    fn depth(&self) -> usize {
        2
    }
}

/// Per decoder layer of `shape`: whether `dir` holds every routed expert of it (an MoE layer
/// with some of its experts is an error).
fn expert_layers(dir: &Path, shape: &ModelShape) -> Result<Vec<bool>, String> {
    let ck = Checkpoint::open(dir).map_err(|e| format!("{}: {e}", dir.display()))?;
    let mut have = vec![false; shape.layers];
    for (l, h) in have.iter_mut().enumerate() {
        if !shape.is_moe(l) {
            continue;
        }
        let n = (0..EXPERTS)
            .filter(|e| {
                ["gate_proj", "up_proj", "down_proj"].iter().all(|p| {
                    ck.get(&format!("{LAYERS}{l}.mlp.experts.{e}.{p}.weight"))
                        .is_some()
                })
            })
            .count();
        if n > 0 && n < EXPERTS {
            return Err(format!(
                "{}: layer {l} holds {n} of its {EXPERTS} routed experts",
                dir.display()
            ));
        }
        *h = n == EXPERTS;
    }
    Ok(have)
}

/// The loaded scorer.
pub struct Engine {
    pub fwd: GlmForward,
    pub build: Build,
    /// The engine line every output carries.
    pub line: String,
    /// sha256 of the checkpoint's `config.json`.
    pub config_sha256: String,
}

/// Load the weights, connect the experts and build the forward for windows of up to
/// `max_tokens` tokens.
pub fn load(o: &Options, max_tokens: usize) -> Result<Engine, String> {
    let (free0, total) = s(device::mem_info())?;
    let (cfg, ckpt) = s(open_checkpoint(&o.checkpoint))?;
    let config = std::fs::read(o.checkpoint.join("config.json"))
        .map_err(|e| format!("{}: {e}", o.checkpoint.display()))?;
    let model_layers = cfg.text.num_hidden_layers as usize;
    let layers = o.dev_layers.unwrap_or(model_layers);
    let shape = s(ModelShape::new(&cfg.text, layers))?;
    let loaded = o.dev_load_layers.unwrap_or(layers).min(layers);
    let t0 = Instant::now();
    let model = s(DeviceModel::load_repeating(&ckpt, &shape, loaded))?;
    let embed = s(HostEmbedding::load(&ckpt))?;
    eprintln!(
        "[score] weights: decoder layers 0-{} ({} loaded), the head: {:.2} GB on the GPU, the \
         embedding {:.2} GB in host RAM, {:.1} s",
        layers - 1,
        loaded,
        model.bytes as f64 / 1e9,
        embed.bytes() as f64 / 1e9,
        t0.elapsed().as_secs_f64()
    );

    let stream = Arc::new(s(Stream::new())?);
    let fcfg = forward_config(o);
    // One lane's exchange, or a one-lane pass (as glm53f-serve sizes the experts).
    let rows = fcfg.lane_rows().max(fcfg.max_verify_rows);
    let moe = (0..layers).filter(|&l| shape.is_moe(l)).count();
    let (experts, zero): (Box<dyn ExpertBackend>, usize) = match &o.experts {
        Experts::Remote(addrs) => {
            eprintln!("[score] connecting the expert ranks {addrs:?}");
            (Box::new(s(RemoteExperts::connect(addrs, rows))?), 0)
        }
        Experts::Local { dir, gib } => {
            let have = expert_layers(dir, &shape)?;
            let missing: Vec<usize> = (0..layers)
                .filter(|&l| shape.is_moe(l) && !have[l])
                .collect();
            if !missing.is_empty() && !o.development() {
                return Err(format!(
                    "{}: no routed experts for MoE layers {missing:?} (a development run gives them \
                     zeros)",
                    dir.display()
                ));
            }
            eprintln!(
                "[score] routed experts on this GPU from {} ({gib} GiB, loaded on demand) for {} \
                 MoE layers, zeros for {}",
                dir.display(),
                moe - missing.len(),
                missing.len()
            );
            let local = s(LocalFp8Experts::new(
                dir,
                (gib * GIB) as usize,
                rows,
                &stream,
                fp8_act(o.fp8_act),
            ))?;
            (Box::new(PartialExperts { local, have }), missing.len())
        }
        Experts::Zero => (Box::new(ZeroExperts), moe),
    };

    // One slot of up to max_tokens tokens, and every buffer a pass uses.
    let max_pages = KvLayout::pages_for(max_tokens.max(1)).div_ceil(4) * 4;
    let bufs = s(ForwardBuffers::new(&fcfg, &shape, max_pages, &stream))?;
    let fb = bufs.bytes();
    let kv = s(KvPool::new(
        KvConfig {
            layout: KvLayout::new(&shape, None),
            max_slots: 1,
            pages: max_pages,
            max_pages,
            base_pages: 0,
        },
        stream.clone(),
    ))?;
    let fwd = s(GlmForward::with_buffers(model, embed, kv, experts, bufs))?;
    let (left, _) = s(device::mem_info())?;
    let gpu = format!(
        "sm_{}{} ({} SMs)",
        s(device::attribute(ATTR_CC_MAJOR))?,
        s(device::attribute(ATTR_CC_MINOR))?,
        s(device::sm_count())?
    );
    eprintln!(
        "[score] device memory: {:.2} GiB of {:.2} GiB free before the weights, {:.2} GiB after \
         everything; forward buffers {:.2} GiB (passes of {} rows in {} lane(s) of {}); one slot \
         of {} pages",
        free0 as f64 / GIB,
        total as f64 / GIB,
        left as f64 / GIB,
        fb.total() as f64 / GIB,
        fcfg.max_rows,
        fcfg.lanes,
        fcfg.lane_rows(),
        max_pages
    );
    let build = Build {
        layers,
        model_layers,
        loaded,
        zero_moe_layers: zero,
        moe_layers: moe,
        gpu,
    };
    let line = engine_line(o, &build);
    Ok(Engine {
        fwd,
        build,
        line,
        config_sha256: glm53f_dsa::sha256::sha256_hex(&config),
    })
}

/// One window's run.
#[derive(Clone, Debug)]
pub struct Scored {
    pub passes: usize,
    pub bytes: u64,
    /// Wall time of the window, and the part spent writing its file.
    pub seconds: f64,
    pub write_seconds: f64,
}

/// Score `w` in a fresh slot in passes of `pass_rows` rows and write its rows' logits to
/// `<out>/<window>.safetensors` with `metadata`.
pub fn score_window(
    fwd: &mut GlmForward,
    w: &Window,
    pass_rows: usize,
    out: &Path,
    metadata: &[(&str, &str)],
) -> Result<Scored, String> {
    let t0 = Instant::now();
    // A fresh slot: empty KV, zeroed KDA states and conv windows, empty DSA tails.
    let mut kv = s(fwd.kv.slot())?;
    let mut file = RowWriter::create(out, &w.window_id, &w.positions, VOCAB, metadata)
        .map_err(|e| format!("{}: {e}", out.display()))?;
    let mut write = 0f64;
    s(
        fwd.score_each(&mut kv, &w.tokens, &w.positions, pass_rows, |r, logits| {
            let t = Instant::now();
            file.push(r, logits).map_err(|e| {
                glm53f_forward::Error::Other(format!("{}: row {r}: {e}", w.window_id))
            })?;
            write += t.elapsed().as_secs_f64();
            Ok(())
        }),
    )?;
    let t = Instant::now();
    let bytes = file.finish().map_err(|e| format!("{}: {e}", w.window_id))?;
    write += t.elapsed().as_secs_f64();
    drop(kv);
    Ok(Scored {
        passes: w.tokens.len().div_ceil(pass_rows),
        bytes,
        seconds: t0.elapsed().as_secs_f64(),
        write_seconds: write,
    })
}
