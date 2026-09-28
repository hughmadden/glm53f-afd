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
    use glm53f_forward::forward::{ForwardConfig, GlmForward};
    use glm53f_forward::gemm::Fp8Act;
    use glm53f_forward::kv::{KvConfig, KvPool};
    use glm53f_forward::kvplan::{KvLayout, PAGE};
    use glm53f_forward::remote::RemoteExperts;
    use glm53f_forward::serve::{ServedForward, DRAFTS};
    use glm53f_forward::shape::{ModelShape, SAMPLE_VOCAB};
    use glm53f_forward::weights::{open_checkpoint, DeviceModel};
    use glm53f_serve::{dev_banner, kv_pages, Experts, Options};

    const GIB: f64 = (1u64 << 30) as f64;
    const MIB: f64 = (1u64 << 20) as f64;

    fn s<T, E: std::fmt::Display>(r: Result<T, E>) -> Result<T, String> {
        r.map_err(|e| e.to_string())
    }

    pub fn run(o: &Options) -> Result<(), String> {
        // 1. The coordinator's weights: the embedding in host RAM, the rest on the GPU.
        let (cfg, ckpt) = s(open_checkpoint(&o.checkpoint))?;
        let total = cfg.text.num_hidden_layers as usize;
        let layers = o.dev_layers.unwrap_or(total);
        let banner = o.dev_layers.map(|n| dev_banner(n, total));
        if let Some(b) = &banner {
            eprintln!("{b}");
        }
        let shape = s(ModelShape::new(&cfg.text, layers))?;
        let t0 = Instant::now();
        let model = s(DeviceModel::load(&ckpt, &shape))?;
        let embed = s(HostEmbedding::load(&ckpt))?;
        eprintln!(
            "[coordinator] weights: decoder layers 0-{} and the head, {:.2} GB on the GPU, the \
             embedding {:.2} GB in host RAM, {:.1} s",
            layers - 1,
            model.bytes as f64 / 1e9,
            embed.bytes() as f64 / 1e9,
            t0.elapsed().as_secs_f64()
        );

        // 2. The drafter's weights next to the model's, before the KV takes what is left.
        let stream = Arc::new(s(Stream::new())?);
        let drafter = match &o.drafter {
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

        // 3. The KV: each slot's positional state (and drafter ring) and page table, and the
        // shared page pool.
        let max_context = o
            .max_context
            .unwrap_or(usize::MAX)
            .min(cfg.text.max_position_embeddings as usize);
        let layout = KvLayout::new(&shape, drafter.as_ref().map(|d| d.config()));
        let max_pages = KvLayout::pages_for(max_context).div_ceil(4) * 4;
        let fixed = o.slots * (max_pages * 4 + layout.slot_fixed_bytes());
        let (free, _) = s(device::mem_info())?;
        let pages = kv_pages(o.kv_gib, free, o.reserve_gib, fixed, layout.page_bytes)?;
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
        eprintln!(
            "[coordinator] KV: {} slots, {:.0} MiB of positional state each{}; a page pool of \
             {:.2} GiB ({} tokens); up to {max_context} tokens per request",
            o.slots,
            layout.slot_fixed_bytes() as f64 / MIB,
            if layout.draft_kv_bytes > 0 {
                format!(
                    " (the drafter's ring {:.2} MiB of it)",
                    layout.draft_kv_bytes as f64 / MIB
                )
            } else {
                String::new()
            },
            (pages * layout.page_bytes) as f64 / GIB,
            pages * PAGE
        );

        // 4. The routed experts, and the forward. With a drafter, one verify pass holds every
        // slot's window.
        let mut fcfg = ForwardConfig {
            max_rows: o.prefill_rows,
            max_requests: o.slots,
            ..ForwardConfig::default()
        };
        if drafter.is_some() {
            fcfg.max_verify_rows = fcfg.max_verify_rows.max(o.slots * (DRAFTS + 1));
        }
        let rows = fcfg.max_rows.max(fcfg.max_verify_rows);
        let experts: Box<dyn ExpertBackend> = match &o.experts {
            Experts::Remote(addrs) => {
                eprintln!("[coordinator] connecting the expert ranks {addrs:?}");
                Box::new(s(RemoteExperts::connect(addrs, rows))?)
            }
            Experts::Local { dir, gib } => {
                eprintln!(
                    "[coordinator] routed experts on this GPU from {} ({gib} GiB, loaded on demand)",
                    dir.display()
                );
                Box::new(s(LocalFp8Experts::new(
                    dir,
                    (gib * GIB) as usize,
                    rows,
                    &stream,
                    Fp8Act::Bf16,
                ))?)
            }
        };
        let mut fwd = s(GlmForward::new(model, embed, kv, experts, fcfg))?;
        if let Some(d) = drafter {
            s(fwd.attach_drafter(d))?;
            eprintln!(
                "[coordinator] drafter attached: up to {DRAFTS} drafts a step, verify passes of up to \
                 {} rows; its tap buffer {:.1} MiB, the forward's scratch {:.0} MiB",
                fcfg.max_verify_rows,
                fwd.drafter().map_or(0, |d| d.tap_bytes()) as f64 / MIB,
                fwd.scratch_bytes() as f64 / MIB
            );
        }
        let mut model = ServedForward::new(fwd)?;
        model.sampled_walk = std::env::var("GLM53F_DFLASH_SAMPLED_WALK").map_or(true, |v| v != "0");
        let slots = (0..o.slots)
            .map(|_| model.fwd.kv.slot())
            .collect::<Result<Vec<_>, _>>()
            .map_err(|e| e.to_string())?;

        // 5. The text side, the engine and its scheduler, the API.
        let codec = GlmPrompts::load(&o.tokenizer, &o.chat_template)?;
        if codec.id_bound() != SAMPLE_VOCAB {
            return Err(format!(
                "the tokenizer's ids end at {}, the forward samples below {SAMPLE_VOCAB}",
                codec.id_bound()
            ));
        }
        let eos = codec.stop_ids()?;
        let cache = HostCache::from_env(&slots[0])?;
        let queue = Queue::from_env(o.slots);
        let engine = CoordinatorEngine::start(
            codec,
            model,
            slots,
            cache,
            SchedulerConfig::from_env(eos.clone()),
            queue,
            EngineConfig::new(eos, max_context),
        )?;
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
