//! `glm53f-score`: the KL gate's engine side. The crate documentation (`src/lib.rs`) lists the
//! options, the output and the development mode.
//!
//! Built without the `cuda` feature it only says how to build it: the scorer runs the forward's
//! kernels.

use glm53f_score::{Options, USAGE};

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
    if let Err(e) = scorer::run(&opts) {
        eprintln!("glm53f-score: {e}");
        std::process::exit(1);
    }
}

#[cfg(not(feature = "cuda"))]
mod scorer {
    pub fn run(_: &glm53f_score::Options) -> Result<(), String> {
        Err("built without the `cuda` feature: cargo build --release -p glm53f-score --features cuda".into())
    }
}

#[cfg(feature = "cuda")]
mod scorer {
    use std::fs;
    use std::time::{Instant, SystemTime, UNIX_EPOCH};

    use glm53f_forward::shape::{SAMPLE_VOCAB, VOCAB};
    use glm53f_model::json::Json;
    use glm53f_score::engine::{self, Engine};
    use glm53f_score::out::pretty;
    use glm53f_score::plan::{tokens_sha256, Plan};
    use glm53f_score::{Experts, Fp8Act, Options, CUDA_ARCH, REVISION};

    fn str(v: impl Into<String>) -> Json {
        Json::Str(v.into())
    }

    fn int(v: usize) -> Json {
        Json::Int(v as i64)
    }

    fn obj(pairs: Vec<(&str, Json)>) -> Json {
        Json::Object(pairs.into_iter().map(|(k, v)| (k.to_string(), v)).collect())
    }

    fn opt<T>(v: Option<T>, f: impl Fn(T) -> Json) -> Json {
        v.map_or(Json::Null, f)
    }

    /// `run.json`: the build, the options, the plan and every window scored so far.
    #[allow(clippy::too_many_arguments)]
    fn record(
        o: &Options,
        plan: &Plan,
        e: &Engine,
        started: u64,
        load: f64,
        windows: &[Json],
        total: f64,
        complete: bool,
    ) -> Json {
        let experts = match &o.experts {
            Experts::Remote(a) => format!("remote ({} ranks)", a.len()),
            Experts::Local { gib, .. } => format!("local ({gib} GiB)"),
            Experts::Zero => "zero".to_string(),
        };
        let wire: Vec<(&str, Json)> = ["GLM53F_RDMA", "GLM53F_WIRE_NOCRC", "GLM53F_WIRE_INFLIGHT"]
            .into_iter()
            .map(|k| (k, opt(std::env::var(k).ok(), str)))
            .collect();
        let b = &e.build;
        obj(vec![
            ("schema", str("glm53f-score-run.v1")),
            ("complete", Json::Bool(complete)),
            ("development", Json::Bool(o.development())),
            ("engine", str(e.line.clone())),
            ("revision", str(REVISION)),
            ("kernels", str(CUDA_ARCH)),
            ("gpu", str(b.gpu.clone())),
            ("started_unix", Json::Int(started as i64)),
            (
                "plan",
                obj(vec![
                    ("sha256", str(plan.sha256.clone())),
                    ("dataset_sha256", opt(plan.dataset_sha256.clone(), str)),
                    (
                        "teacher_model_revision",
                        opt(plan.teacher_model_revision.clone(), str),
                    ),
                    ("vocab", int(plan.vocab)),
                    ("windows", int(plan.windows.len())),
                ]),
            ),
            (
                "options",
                obj(vec![
                    ("pass_rows", int(o.pass_rows)),
                    ("prefill_lanes", int(o.lanes)),
                    (
                        "windows",
                        opt(o.windows.clone(), |w| {
                            Json::Array(w.into_iter().map(str).collect())
                        }),
                    ),
                    ("experts", str(experts)),
                    ("wire", obj(wire)),
                    ("kda_chunked_prefill", Json::Bool(o.kda_chunked_prefill)),
                    (
                        "fp8_act",
                        str(match o.fp8_act {
                            Fp8Act::Bf16 => "bf16",
                            Fp8Act::Dynamic => "dynamic",
                        }),
                    ),
                    ("promote_k32", Json::Bool(o.promote_k32)),
                    ("kda_fp8", Json::Bool(o.numerics.kda_fp8)),
                    ("kda_state_bf16", Json::Bool(o.numerics.kda_state_bf16)),
                    ("prefill_w8a16", Json::Bool(o.numerics.prefill_w8a16)),
                    ("kda_prefill_w8a8", Json::Bool(o.numerics.kda_prefill_w8a8)),
                    ("dev_layers", opt(o.dev_layers, int)),
                    ("dev_load_layers", opt(o.dev_load_layers, int)),
                ]),
            ),
            ("forward", str(format!("{:?}", e.fwd.cfg))),
            (
                "model",
                obj(vec![
                    ("layers", int(b.layers)),
                    ("model_layers", int(b.model_layers)),
                    ("loaded_layers", int(b.loaded)),
                    ("moe_layers", int(b.moe_layers)),
                    ("zero_moe_layers", int(b.zero_moe_layers)),
                    ("config_sha256", str(e.config_sha256.clone())),
                ]),
            ),
            ("load_seconds", Json::Num(load)),
            ("windows", Json::Array(windows.to_vec())),
            ("total_seconds", Json::Num(total)),
        ])
    }

    fn write_record(o: &Options, v: &Json) -> Result<(), String> {
        let path = o.out.join("run.json");
        let tmp = o.out.join("run.json.partial");
        fs::write(&tmp, pretty(v)).map_err(|e| format!("{}: {e}", tmp.display()))?;
        fs::rename(&tmp, &path).map_err(|e| format!("{}: {e}", path.display()))
    }

    pub fn run(o: &Options) -> Result<(), String> {
        let t0 = Instant::now();
        let started = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_or(0, |d| d.as_secs());
        let bytes = fs::read(&o.plan).map_err(|e| format!("{}: {e}", o.plan.display()))?;
        let plan = Plan::parse(&bytes, VOCAB, SAMPLE_VOCAB)?;
        let windows = plan.select(o.windows.as_deref())?;
        let max_tokens = windows.iter().map(|w| w.tokens.len()).max().unwrap_or(1);
        let rows: usize = windows.iter().map(|w| w.positions.len()).sum();
        eprintln!(
            "[score] plan {}: {} of its {} windows, {rows} rows to write ({:.2} GB of F32 logits)",
            plan.sha256,
            windows.len(),
            plan.windows.len(),
            (rows * VOCAB * 4) as f64 / 1e9
        );
        fs::create_dir_all(&o.out).map_err(|e| format!("{}: {e}", o.out.display()))?;
        let mut e = engine::load(o, max_tokens)?;
        let load = t0.elapsed().as_secs_f64();
        if o.development() {
            let rule = "=".repeat(78);
            eprintln!(
                "{rule}\nDEVELOPMENT RUN: the logits are meaningless by design (they exercise the \
                 plumbing, not the model).\n{rule}"
            );
        }
        eprintln!("[score] engine: {}", e.line);
        let mut done = Vec::new();
        for w in windows {
            let fed = tokens_sha256(&w.tokens);
            let meta = [
                ("window_id", w.window_id.as_str()),
                ("tokens_sha256", fed.as_str()),
                ("plan_sha256", plan.sha256.as_str()),
                ("engine", e.line.as_str()),
            ];
            let r = engine::score_window(&mut e.fwd, w, o.pass_rows, &o.out, &meta)?;
            let score = r.seconds - r.write_seconds;
            eprintln!(
                "[score] {}: {} tokens in {} passes of up to {} rows, {} rows written, {:.2} s \
                 ({:.0} tokens/s; writing {:.2} s, {:.1} MB)",
                w.window_id,
                w.tokens.len(),
                r.passes,
                o.pass_rows,
                w.positions.len(),
                r.seconds,
                w.tokens.len() as f64 / score.max(1e-9),
                r.write_seconds,
                r.bytes as f64 / 1e6
            );
            done.push(obj(vec![
                ("window_id", str(w.window_id.clone())),
                ("tokens", int(w.tokens.len())),
                ("tokens_sha256", str(fed)),
                ("rows", int(w.positions.len())),
                ("passes", int(r.passes)),
                ("seconds", Json::Num(r.seconds)),
                ("write_seconds", Json::Num(r.write_seconds)),
                ("bytes", Json::Int(r.bytes as i64)),
            ]));
            let rec = record(
                o,
                &plan,
                &e,
                started,
                load,
                &done,
                t0.elapsed().as_secs_f64(),
                false,
            );
            write_record(o, &rec)?;
        }
        let total = t0.elapsed().as_secs_f64();
        write_record(o, &record(o, &plan, &e, started, load, &done, total, true))?;
        eprintln!(
            "[score] {} windows, {rows} rows in {total:.1} s (load {load:.1} s) -> {}",
            done.len(),
            o.out.display()
        );
        Ok(())
    }
}
