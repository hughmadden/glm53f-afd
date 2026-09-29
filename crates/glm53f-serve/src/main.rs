//! `glm53f-serve`: the coordinator daemon. The crate documentation (`src/lib.rs`) lists the
//! options and the development mode.
//!
//! Built without the `cuda` feature it only says how to build it: the daemon runs the forward's
//! kernels.

use glm53f_serve::{Options, USAGE};

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.iter().any(|a| a == "--help" || a == "-h") {
        println!("{USAGE}");
        return;
    }
    let opts = match Options::parse(&args, &|k| std::env::var(k).ok()) {
        Ok(o) => o,
        Err(e) => {
            eprintln!("{USAGE}\nerror: {e}");
            std::process::exit(2);
        }
    };
    if let Err(e) = daemon::run(&opts) {
        eprintln!("glm53f-serve: {e}");
        std::process::exit(1);
    }
}

#[cfg(not(feature = "cuda"))]
mod daemon {
    pub fn run(_: &glm53f_serve::Options) -> Result<(), String> {
        Err("built without the `cuda` feature: cargo build --release -p glm53f-serve --features cuda".into())
    }
}

#[cfg(feature = "cuda")]
mod daemon {
    use std::sync::Arc;
    use std::time::Instant;

    use glm53f_api::dialect::GlmDialect;
    use glm53f_coordinator::{
        CoordinatorEngine, EngineConfig, GlmPrompts, HostCache, Queue, SchedulerConfig,
    };
    use glm53f_forward::device::{self, Stream};
    use glm53f_forward::draft::{Dflash, LAYERS_NEEDED};
    use glm53f_forward::embed::HostEmbedding;
    use glm53f_forward::experts::{ExpertBackend, LocalFp8Experts};
    use glm53f_forward::forward::{ForwardBuffers, ForwardConfig, GlmForward};
    use glm53f_forward::gemm::Fp8Act;
    use glm53f_forward::kv::{KvConfig, KvPool};
    use glm53f_forward::kvplan::{KvLayout, PAGE};
    use glm53f_forward::remote::RemoteExperts;
    use glm53f_forward::serve::{ServedForward, DRAFTS};
    use glm53f_forward::shape::{ModelShape, SAMPLE_VOCAB};
    use glm53f_forward::weights::{open_checkpoint, DeviceModel, WeightOptions};
    use glm53f_serve::{admission_line, dev_banner, kv_pages, verify_rows, Experts, Options};

    const GIB: f64 = (1u64 << 30) as f64;
    const MIB: f64 = (1u64 << 20) as f64;
    // The options' lane bound is the forward's.
    const _: () = assert!(glm53f_serve::MAX_PREFILL_LANES == glm53f_forward::forward::MAX_LANES);

    fn s<T, E: std::fmt::Display>(r: Result<T, E>) -> Result<T, String> {
        r.map_err(|e| e.to_string())
    }

    fn gib(b: usize) -> String {
        format!("{:.2} GiB", b as f64 / GIB)
    }

    fn mib(b: usize) -> String {
        format!("{:.1} MiB", b as f64 / MIB)
    }

    pub fn run(o: &Options) -> Result<(), String> {
        // The text side first (the scheduler's configuration feeds the memory plan below).
        let codec = GlmPrompts::load(&o.tokenizer, &o.chat_template)?;
        if codec.id_bound() != SAMPLE_VOCAB {
            return Err(format!(
                "the tokenizer's ids end at {}, the forward samples below {SAMPLE_VOCAB}",
                codec.id_bound()
            ));
        }
        let eos = codec.stop_ids()?;
        let mut sched = SchedulerConfig::from_env(eos.clone());
        sched.copy_windows = o.copy_windows;

        // 1. The coordinator's weights: the embedding in host RAM, the rest on the GPU.
        let (free0, total) = s(device::mem_info())?;
        let (cfg, ckpt) = s(open_checkpoint(&o.checkpoint))?;
        let n_layers = cfg.text.num_hidden_layers as usize;
        let layers = o.dev_layers.unwrap_or(n_layers);
        let banner = o.dev_layers.map(|n| dev_banner(n, n_layers));
        if let Some(b) = &banner {
            eprintln!("{b}");
        }
        let shape = s(ModelShape::new(&cfg.text, layers))?;
        let num = o.numerics;
        eprintln!("[coordinator] numerics: {}", num.describe());
        let t0 = Instant::now();
        let wopts = WeightOptions {
            kda_fp8: num.kda_fp8,
        };
        let model = s(DeviceModel::load_with(&ckpt, &shape, layers, wopts))?;
        let embed = s(HostEmbedding::load(&ckpt))?;
        eprintln!(
            "[coordinator] weights: decoder layers 0-{} and the head, {:.2} GB on the GPU (KDA \
             projections {}), the embedding {:.2} GB in host RAM, {:.1} s",
            layers - 1,
            model.bytes as f64 / 1e9,
            if num.kda_fp8 {
                "FP8 block-128, quantized at load"
            } else {
                "BF16"
            },
            embed.bytes() as f64 / 1e9,
            t0.elapsed().as_secs_f64()
        );

        // 2. The drafter's weights next to the model's.
        let stream = Arc::new(s(Stream::new())?);
        let mut drafter = match &o.drafter {
            Some(_) if layers < LAYERS_NEEDED => {
                eprintln!(
                    "[coordinator] drafter off: its taps are the outputs of layers 5 to 42 and this \
                     forward runs layers 0-{}",
                    layers - 1
                );
                None
            }
            Some(dir) => {
                let t0 = Instant::now();
                let d = s(Dflash::load(dir, &model, &embed, &stream))?;
                eprintln!(
                    "[coordinator] drafter: DFlash2 from {}, {:.2} GiB of weights on the GPU (the \
                     LM head is the forward's), {:.1} s",
                    dir.display(),
                    d.weight_bytes() as f64 / GIB,
                    t0.elapsed().as_secs_f64()
                );
                Some(d)
            }
            None => None,
        };

        // 3. The routed experts, with their buffers for one lane's exchange and up to a lane's
        //    exchange in flight each. With a drafter, one verify pass holds every slot's window,
        //    up to the step's row budget.
        let mut fcfg = ForwardConfig {
            max_rows: o.prefill_rows,
            lanes: o.prefill_lanes,
            decode_lane_rows: o.decode_lanes.0,
            decode_lane_max_rows: o.decode_lanes.1,
            max_requests: o.slots,
            ..ForwardConfig::default()
        };
        fcfg.policy.prefill_w8a16 = num.prefill_w8a16;
        fcfg.policy.kda_prefill_w8a8 = num.kda_prefill_w8a8;
        fcfg.kda_chunked_prefill = num.kda_chunked_prefill;
        if drafter.is_some() {
            fcfg.max_verify_rows =
                fcfg.max_verify_rows
                    .max(verify_rows(o.slots, DRAFTS + 1, sched.spec_max_rows));
        }
        let rows = fcfg.lane_rows().max(fcfg.max_verify_rows);
        // The expert wire's first failure, for `GET /health` (the forward refuses every call after
        // it until the coordinator restarts).
        let mut wire_failure = None;
        let (experts, experts_bytes, experts_what): (Box<dyn ExpertBackend>, usize, String) =
            match &o.experts {
                Experts::Remote(addrs) => {
                    eprintln!("[coordinator] connecting the expert ranks {addrs:?}");
                    let r = s(RemoteExperts::connect(addrs, rows, o.prefill_lanes))?;
                    wire_failure = Some(r.failure());
                    let (recv, body) = r.host_bytes();
                    let what = format!(
                        "the expert exchange, {rows} rows an exchange, {} in flight at most \
                         ({} lane(s); page-locked host buffers: receive {}, request body {}), \
                         device buffers",
                        glm53f_forward::experts::ExpertBackend::depth(&r),
                        o.prefill_lanes,
                        mib(recv),
                        mib(body)
                    );
                    (Box::new(r), RemoteExperts::device_bytes(rows), what)
                }
                Experts::Local { dir, gib: g } => {
                    eprintln!(
                        "[coordinator] routed experts on this GPU from {} ({g} GiB, loaded on \
                         demand)",
                        dir.display()
                    );
                    let budget = (g * GIB) as usize;
                    let e = s(LocalFp8Experts::new(
                        dir,
                        budget,
                        rows,
                        &stream,
                        Fp8Act::Bf16,
                    ))?;
                    // Loaded on demand: the budget is kept free for them.
                    (
                        Box::new(e),
                        budget,
                        "the local FP8 experts' cache".to_string(),
                    )
                }
            };

        // 4. Every buffer the forward's passes use, and the drafter's tap buffer and working
        //    memory, before the pool: no pass, append or draft allocates.
        let max_context = o
            .max_context
            .unwrap_or(usize::MAX)
            .min(cfg.text.max_position_embeddings as usize);
        let layout = KvLayout::new(&shape, drafter.as_ref().map(|d| d.config()))
            .with_kda_state_bf16(num.kda_state_bf16);
        let max_pages = KvLayout::pages_for(max_context).div_ceil(4) * 4;
        let bufs = s(ForwardBuffers::new(&fcfg, &shape, max_pages, &stream))?;
        let fb = bufs.bytes();
        let drafter_bytes = match drafter.as_mut() {
            Some(d) => s(d.reserve(bufs.pass_rows(), o.slots))?,
            None => (0, 0),
        };

        // 5. The KV: each slot's positional state (and drafter ring) and page table, then the page
        //    pool from what is left, less the reserve (and the local experts' budget, loaded
        //    later).
        let fixed = o.slots * (max_pages * 4 + layout.slot_fixed_bytes());
        let (free, _) = s(device::mem_info())?;
        let local = if matches!(o.experts, Experts::Local { .. }) {
            experts_bytes
        } else {
            0
        };
        let pages = kv_pages(
            o.kv_gib,
            free.saturating_sub(local),
            o.reserve_gib,
            fixed,
            layout.page_bytes,
        )?;
        let kv = s(KvPool::new(
            KvConfig {
                layout,
                max_slots: o.slots,
                pages,
                max_pages,
                base_pages: 0,
            },
            stream.clone(),
        ))?;
        let mut fwd = s(GlmForward::with_buffers(model, embed, kv, experts, bufs))?;
        let drafter_weights = drafter.as_ref().map_or(0, |d| d.weight_bytes());
        if let Some(d) = drafter {
            // Its tap buffer is already large enough: nothing is allocated here.
            s(fwd.attach_drafter(d))?;
        }
        let (left, _) = s(device::mem_info())?;
        let mark = layout.mark_pages() * layout.page_bytes;
        eprintln!(
            "[coordinator] device memory, {} in all, {} free before the weights:",
            gib(total),
            gib(free0)
        );
        eprintln!(
            "[coordinator]   weights {}; forward buffers {}: prefill {} lane(s) of {} rows ({}), \
             verify {} rows {}, workspaces {} (indexer {}, sparse MLA {}, KDA {}), GEMM {}",
            gib(fwd.model.bytes),
            gib(fb.total()),
            fcfg.lanes,
            fcfg.lane_rows(),
            fb.lanes
                .iter()
                .filter(|&&b| b > 0)
                .map(|&b| gib(b))
                .collect::<Vec<_>>()
                .join(" + "),
            fcfg.max_verify_rows,
            mib(fb.verify),
            mib(fb.workspaces.iter().sum()),
            mib(fb.workspaces[0]),
            mib(fb.workspaces[1]),
            mib(fb.workspaces[2]),
            mib(fb.gemm)
        );
        if fwd.has_drafter() {
            eprintln!(
                "[coordinator]   drafter: weights {}, tap buffer {} ({} rows), working memory {} \
                 (drafts of up to {} requests), a context ring of {} in each slot's state; up to \
                 {DRAFTS} drafts a step, verify passes of up to {} rows (the step's row budget, \
                 GLM53F_SPEC_MAX_ROWS: {})",
                gib(drafter_weights),
                mib(drafter_bytes.0),
                fwd.pass_rows(),
                mib(drafter_bytes.1),
                o.slots,
                mib(layout.draft_kv_bytes),
                fcfg.max_verify_rows,
                match sched.spec_max_rows {
                    0 => "none".to_string(),
                    n => n.to_string(),
                }
            );
        }
        eprintln!(
            "[coordinator]   decode and verify passes: {}",
            match o.decode_lanes {
                (0, _) => "one lane".to_string(),
                (a, usize::MAX) => format!("two lanes from {a} rows over two requests or more"),
                (a, b) => format!("two lanes from {a} to {b} rows over two requests or more"),
            }
        );
        eprintln!(
            "[coordinator]   {} {}; {} slots x {} of positional state (KDA states {}) and page \
             tables = {}",
            experts_what,
            mib(experts_bytes),
            o.slots,
            mib(fixed / o.slots),
            if layout.kda_state_bf16 {
                "BF16"
            } else {
                "FP32"
            },
            gib(fixed)
        );
        eprintln!(
            "[coordinator]   KV page pool {} ({pages} pages, {} tokens); up to {max_context} \
             tokens per request; snapshot marks take {} pages ({}) each from the pool, {}",
            gib(pages * layout.page_bytes),
            pages * PAGE,
            layout.mark_pages(),
            mib(mark),
            match sched.bank {
                0 => "as many as fit: they leave only when an incoming request or a new snapshot \
                      needs their pages, least recently used first, to RAM when the tier is on"
                    .to_string(),
                n => format!(
                    "at most {n} prompt + {n} turn (GLM53F_PREFIX_CACHE_ENTRIES), {} if full",
                    gib(2 * n * mark)
                ),
            }
        );
        eprintln!(
            "[coordinator]   {}",
            admission_line(pages, PAGE, layout.page_bytes, max_context)
        );
        eprintln!(
            "[coordinator]   left free: {} measured ({} reserved by --reserve-gib for kernel \
             modules, the sampler and allocator slack{})",
            gib(left),
            gib((o.reserve_gib * GIB) as usize),
            if local > 0 {
                ", plus the local experts' budget"
            } else {
                ""
            }
        );
        let mut model = ServedForward::new(fwd)?;
        model.sampled_walk = std::env::var("GLM53F_DFLASH_SAMPLED_WALK").map_or(true, |v| v != "0");
        if std::env::var_os("GLM53F_PROFILE").is_some() {
            // Per prefill pass and per decode step: each lane's GPU and host time per MoE layer
            // (`PIPE` and `STEP` lines).
            model.fwd.set_lane_trace(true, true);
        }
        let slots = (0..o.slots)
            .map(|_| model.fwd.kv.slot())
            .collect::<Result<Vec<_>, _>>()
            .map_err(|e| e.to_string())?;

        // 6. The engine and its scheduler, the API.
        let cache = HostCache::from_env(&slots[0])?;
        let queue = Queue::from_env(o.slots);
        let engine = CoordinatorEngine::start(
            codec,
            model,
            slots,
            cache,
            sched,
            queue,
            EngineConfig::new(eos, max_context),
        )?
        .with_health(move || match wire_failure.as_ref().and_then(|f| f.get()) {
            Some(e) => Err(format!("{e}; restart the coordinator")),
            None => Ok(()),
        });
        // The API listens only from here, once the engine is ready: before, a probe's connection
        // is refused, so `GET /health` never has to answer "not yet".
        eprintln!(
            "[coordinator] serving the API on {}{}",
            o.listen,
            if banner.is_some() {
                " (DEVELOPMENT MODE: the output is meaningless text)"
            } else {
                ""
            }
        );
        glm53f_api::serve(&o.listen, Arc::new(engine), Arc::new(GlmDialect))
            .map_err(|e| format!("serve on {}: {e}", o.listen))
    }
}
