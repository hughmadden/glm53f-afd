//! Print the GLM-5.3-Flash memory plan (docs/SIZING.md sections 2-6).
//!
//! cargo run -p glm53f-model --example plan -- --config <config.json>
//!     [--headers-dir <dir>] [--gpu-mib 32607] [--slots 16] [--runtime-gib 4]
//!     [--layout A|B|C|D] [--drafter-config <config.json> | --no-drafter]
//!     [--ranks 4] [--spark-free-gib 113]
//!
//! `--headers-dir` holds safetensors header bundles (`official.headers.json`,
//! `tr3.headers.json`, `nvfp4.headers.json`: shard file name -> header). Without
//! it, tensor bytes are derived from the config. The DFlash2 drafter defaults to
//! a copy of the published `incoai/GLM-5.3-Flash-DFlash2` config.

use std::path::PathBuf;
use std::process::ExitCode;

use glm53f_model::catalog::{Catalog, CheckpointFormat, Coverage, Group};
use glm53f_model::config::{DraftConfig, ModelConfig};
use glm53f_model::dtype::DType;
use glm53f_model::planner::*;
use glm53f_model::safetensors::parse_header_bundle;

const DFLASH2_CONFIG: &str = include_str!("../tests/data/incoai_GLM-5.3-Flash-DFlash2.config.json");
const USAGE: &str = "usage: plan --config <config.json> [--headers-dir <dir>] [--gpu-mib 32607] [--slots 16] \
[--runtime-gib 4] [--layout A|B|C|D] [--drafter-config <config.json> | --no-drafter] [--ranks 4] [--spark-free-gib 113]";

struct Args {
    config: PathBuf,
    headers: Option<PathBuf>,
    gpu_mib: u64,
    slots: u64,
    runtime_gib: f64,
    layout: Layout,
    drafter: Option<Option<PathBuf>>,
    ranks: u64,
    spark_free_gib: f64,
}

fn parse_args() -> Result<Args, String> {
    let mut a = Args {
        config: PathBuf::new(),
        headers: None,
        gpu_mib: 32_607,
        slots: 16,
        runtime_gib: 4.0,
        layout: Layout::B,
        drafter: Some(None),
        ranks: 4,
        spark_free_gib: 113.0,
    };
    let mut it = std::env::args().skip(1);
    let mut config = None;
    while let Some(flag) = it.next() {
        let mut val = || it.next().ok_or_else(|| format!("{flag} needs a value"));
        let num = |s: String| {
            s.parse::<f64>()
                .map_err(|_| format!("{flag}: not a number: {s}"))
        };
        match flag.as_str() {
            "--config" => config = Some(PathBuf::from(val()?)),
            "--headers-dir" => a.headers = Some(PathBuf::from(val()?)),
            "--gpu-mib" => a.gpu_mib = num(val()?)? as u64,
            "--slots" => a.slots = num(val()?)? as u64,
            "--runtime-gib" => a.runtime_gib = num(val()?)?,
            "--layout" => a.layout = Layout::parse(&val()?).ok_or("--layout takes A, B, C or D")?,
            "--drafter-config" => a.drafter = Some(Some(PathBuf::from(val()?))),
            "--no-drafter" => a.drafter = None,
            "--ranks" => a.ranks = num(val()?)? as u64,
            "--spark-free-gib" => a.spark_free_gib = num(val()?)?,
            "-h" | "--help" => return Err(USAGE.into()),
            other => return Err(format!("unknown argument {other}\n{USAGE}")),
        }
    }
    a.config = config.ok_or(USAGE)?;
    Ok(a)
}

struct Catalogs {
    official: Catalog,
    exl3: Catalog,
    nvfp4: Catalog,
    source: String,
}

fn catalogs(cfg: &ModelConfig, headers: Option<&PathBuf>) -> glm53f_model::Result<Catalogs> {
    let derived = |f| Catalog::from_config(cfg, f);
    let Some(dir) = headers else {
        return Ok(Catalogs {
            official: derived(CheckpointFormat::OfficialFp8),
            exl3: derived(CheckpointFormat::Exl3 { bits: 4 }),
            nvfp4: derived(CheckpointFormat::Nvfp4),
            source: "tensor shapes derived from config.json".into(),
        });
    };
    let mut used = Vec::new();
    let mut read = |name: &str, fmt: CheckpointFormat| -> glm53f_model::Result<Catalog> {
        let path = dir.join(format!("{name}.headers.json"));
        if !path.is_file() {
            return Ok(derived(fmt));
        }
        let text = std::fs::read_to_string(&path).map_err(|e| glm53f_model::Error::Io {
            path: path.display().to_string(),
            source: e,
        })?;
        used.push(name.to_string());
        Catalog::from_shards(
            cfg,
            &parse_header_bundle(&text)?,
            Some(fmt),
            Coverage::Complete,
        )
    };
    let official = read("official", CheckpointFormat::OfficialFp8)?;
    let exl3 = read("tr3", CheckpointFormat::Exl3 { bits: 4 })?;
    let nvfp4 = read("nvfp4", CheckpointFormat::Nvfp4)?;
    let source = match used.len() {
        0 => "tensor shapes derived from config.json (no header bundles found)".to_string(),
        3 => format!("safetensors headers ({})", used.join(", ")),
        _ => format!(
            "safetensors headers ({}), the other formats derived from config.json",
            used.join(", ")
        ),
    };
    Ok(Catalogs {
        official,
        exl3,
        nvfp4,
        source,
    })
}

fn dtypes(cat: &Catalog, groups: &[Group]) -> String {
    let mut by: Vec<(DType, u64)> = Vec::new();
    for &g in groups {
        for (d, b) in cat.group_dtypes(g) {
            match by.iter_mut().find(|(x, _)| *x == d) {
                Some(e) => e.1 += b,
                None => by.push((d, b)),
            }
        }
    }
    let total: u64 = by.iter().map(|(_, b)| b).sum();
    by.sort_by_key(|&(_, b)| std::cmp::Reverse(b));
    let names: Vec<&str> = by
        .iter()
        .filter(|(_, b)| *b * 100 >= total)
        .map(|(d, _)| {
            if *d == DType::F8E4M3 {
                "FP8"
            } else {
                d.as_str()
            }
        })
        .collect();
    names.join("+")
}

fn run(a: Args) -> glm53f_model::Result<()> {
    let cfg = ModelConfig::load(&a.config)?;
    let t = &cfg.text;
    let drafter = match &a.drafter {
        None => None,
        Some(None) => Some(DraftConfig::parse(DFLASH2_CONFIG)?),
        Some(Some(p)) => Some(DraftConfig::load(p)?),
    };
    if let Some(d) = &drafter {
        d.validate(&cfg)?;
    }
    let cats = catalogs(&cfg, a.headers.as_ref())?;
    let (kda, dsa, moe) = (
        t.kda_layer_ids().len(),
        t.dsa_layer_ids().len(),
        t.moe_layers().len(),
    );

    println!("GLM-5.3-Flash memory plan (docs/SIZING.md sections 2-6)");
    println!("bytes from: {}", cats.source);
    println!("GB = 1e9 bytes, GiB = 2^30 bytes\n");

    // Section 4.
    let resident: Vec<ResidentWeights> = Layout::ALL
        .iter()
        .map(|&l| {
            resident_weights(
                if l.policy().shipped_bf16 {
                    &cats.exl3
                } else {
                    &cats.official
                },
                l.policy(),
            )
        })
        .collect::<glm53f_model::Result<_>>()?;
    let first_dense = t.moe.first_k_dense_replace;
    let rows: Vec<(String, Vec<Group>)> = vec![
        (
            format!("KDA attention, {kda} layers"),
            vec![Group::KdaAttention],
        ),
        (
            format!("DSA attention and indexer, {dsa} layers"),
            vec![Group::DsaAttention, Group::DsaIndexer],
        ),
        (
            format!("Shared experts, {moe} layers"),
            vec![Group::SharedExperts],
        ),
        (
            format!("Dense MLP, layers 0-{}", first_dense - 1),
            vec![Group::DenseMlp],
        ),
        (
            "Routers, mHC, norms".into(),
            vec![Group::Routers, Group::Mhc, Group::Norms],
        ),
        ("LM head".into(), vec![Group::LmHead]),
        ("Embedding".into(), vec![Group::Embedding]),
    ];
    println!("Coordinator weights (section 4)");
    println!(
        "  {:<38} {:>16}   {:>9} {:>9} {:>9} {:>9}",
        "group", "official ckpt", "A", "B", "C", "D"
    );
    for (label, groups) in &rows {
        let official: u64 = groups.iter().map(|&g| cats.official.group_bytes(g)).sum();
        let cells: Vec<String> = resident
            .iter()
            .map(|r| {
                let b: u64 = groups.iter().map(|&g| r.group(g)).sum();
                if b == 0 {
                    "host".into()
                } else {
                    format!("{:.2} GB", gb(b))
                }
            })
            .collect();
        let od = format!("{:.2} GB {}", gb(official), dtypes(&cats.official, groups));
        println!(
            "  {label:<38} {od:>16}   {:>9} {:>9} {:>9} {:>9}",
            cells[0], cells[1], cells[2], cells[3]
        );
    }
    let totals: Vec<String> = resident
        .iter()
        .map(|r| format!("{:.2} GiB", gib(r.total)))
        .collect();
    println!(
        "  {:<38} {:>16}   {:>9} {:>9} {:>9} {:>9}",
        "total resident", "", totals[0], totals[1], totals[2], totals[3]
    );
    for l in Layout::ALL {
        println!("    {}: {}", l.letter(), l.title());
    }
    let drafter_bytes = drafter.as_ref().map_or(0, DraftConfig::weight_bytes);
    if drafter.is_some() {
        println!(
            "  {:<38} {:>16}   {:.2} GiB resident (shares the embedding and LM head)",
            "DFlash2 drafter",
            format!("{:.2} GB BF16", gb(drafter_bytes)),
            gib(drafter_bytes)
        );
    }
    println!(
        "  {:<38} {:>16}   optional second drafter; its experts sit on the expert ranks",
        "MTP layer",
        format!("{:.2} GB", gb(cats.official.group_bytes(Group::MtpLayer)))
    );
    println!(
        "  {:<38} {:>16}   optional, paged in per image request\n",
        "Vision tower",
        format!(
            "{:.2} GB BF16",
            gb(cats.official.group_bytes(Group::Vision))
        )
    );

    // Sections 2 and 3.
    let slot = SlotState::new(t, drafter.as_ref());
    println!("Per-slot state (section 3)");
    println!(
        "  KDA recurrent state, FP32      {:>8.2} MiB",
        mib(slot.kda_state)
    );
    println!(
        "  short-convolution state, BF16  {:>8.2} MiB",
        mib(slot.conv_state)
    );
    println!(
        "  drafter KV, BF16               {:>8.2} MiB",
        mib(slot.draft_kv)
    );
    println!(
        "  per slot                       {:>8.2} MiB; {} slots {:.2} GiB\n",
        mib(slot.total()),
        a.slots,
        gib(a.slots * slot.total())
    );
    println!("Context per token (section 2)");
    for (p, name) in [(KvPrecision::Fp8, "FP8"), (KvPrecision::Bf16, "BF16")] {
        let g = KvGeometry::new(t, p);
        println!(
            "  {name:<4} MLA {} B + indexer {} B = {} B per token; page of {PAGE_TOKENS} tokens {} B; 256K {:.2} GiB, 1M {:.2} GiB",
            thousands(g.mla_per_token()),
            g.index_per_pool() / g.kpool,
            thousands(g.per_token() as u64),
            thousands(g.page_bytes()),
            gib(g.bytes_for(TOKENS_256K)),
            gib(g.bytes_for(TOKENS_1M))
        );
    }
    println!();

    // Section 5.
    let budget_for = |layout: Layout| GpuBudget {
        device: a.gpu_mib * MIB,
        weights: resident[Layout::ALL.iter().position(|&l| l == layout).unwrap()].total,
        drafter: drafter_bytes,
        slots: a.slots,
        slot_state: slot,
        runtime: (a.runtime_gib * GIB as f64) as u64,
    };
    let p = plan_gpu(t, budget_for(a.layout))?;
    let b = p.budget;
    println!(
        "Coordinator GPU budget (section 5): layout {}, {}",
        a.layout.letter(),
        a.layout.title()
    );
    println!(
        "  device memory                {:>8.2} GiB ({} MiB)",
        gib(b.device),
        thousands(a.gpu_mib)
    );
    println!("  - resident weights           {:>8.2} GiB", gib(b.weights));
    println!("  - drafter weights            {:>8.2} GiB", gib(b.drafter));
    println!(
        "  - slot state, {:>2} slots       {:>8.2} GiB",
        b.slots,
        gib(b.slots * b.slot_state.total())
    );
    println!("  - runtime reserve            {:>8.2} GiB", gib(b.runtime));
    println!("  = KV pool                    {:>8.2} GiB", gib(p.kv_pool));
    println!("  {:<28} {:>12} {:>12}", "", "FP8 KV", "BF16 KV");
    println!(
        "  {:<28} {:>12} {:>12}",
        format!("pages of {PAGE_TOKENS} tokens"),
        thousands(p.fp8.pages),
        thousands(p.bf16.pages)
    );
    println!(
        "  {:<28} {:>12} {:>12}",
        "tokens",
        thousands(p.fp8.tokens),
        thousands(p.bf16.tokens)
    );
    println!(
        "  {:<28} {:>12} {:>12}",
        "256K requests at once", p.fp8.requests_256k, p.bf16.requests_256k
    );
    println!(
        "  {:<28} {:>12} {:>12}",
        "1M requests at once", p.fp8.requests_1m, p.bf16.requests_1m
    );
    println!(
        "  {:<28} {:>12} {:>12}\n",
        "longest single request",
        thousands(p.fp8.longest_request),
        thousands(p.bf16.longest_request)
    );
    println!(
        "All layouts at {:.1} GiB runtime and {} slots",
        a.runtime_gib, a.slots
    );
    println!(
        "  {:<7} {:>10} {:>12} {:>6} {:>4} {:>12}",
        "layout", "KV pool", "tokens FP8", "256K", "1M", "tokens BF16"
    );
    for l in Layout::ALL {
        match plan_gpu(t, budget_for(l)) {
            Ok(q) => println!(
                "  {:<7} {:>6.2} GiB {:>10.2} M {:>6} {:>4} {:>10.2} M",
                l.letter(),
                gib(q.kv_pool),
                mtok(q.fp8.tokens),
                q.fp8.requests_256k,
                q.fp8.requests_1m,
                mtok(q.bf16.tokens)
            ),
            Err(e) => println!("  {:<7} {e}", l.letter()),
        }
    }
    println!();

    // Section 6.
    println!(
        "Expert ranks (section 6): TP{} over the {}-wide expert intermediate",
        a.ranks, t.moe.moe_intermediate_size
    );
    let span = format!(
        "per rank, layers {}-{}",
        t.moe_layers()[0],
        t.num_hidden_layers - 1
    );
    let free = format!("left of {:.0} GiB", a.spark_free_gib);
    println!(
        "  {:<12} {span:>24} {:>12} {free:>16}   read per token at M1",
        "format", "MTP experts"
    );
    for cat in [&cats.exl3, &cats.nvfp4, &cats.official] {
        let s = plan_spark(cat, t, a.ranks)?;
        let left = a.spark_free_gib - gib(s.bytes + s.mtp_bytes);
        let per_rank = format!("{:.2} GB ({:.2} GiB)", gb(s.bytes), gib(s.bytes));
        let mtp = format!("{:.2} GB", gb(s.mtp_bytes));
        let read = format!(
            "{:.2} GB, {:.1} ms at 230 GB/s",
            gb(s.read_per_token),
            s.read_per_token as f64 / 230e6
        );
        println!(
            "  {:<12} {per_rank:>24} {mtp:>12} {left:>12.1} GiB   {read}",
            cat.format.label()
        );
    }
    Ok(())
}

/// 1234567 -> "1,234,567".
fn thousands(n: u64) -> String {
    let s = n.to_string();
    let mut out = String::new();
    for (i, c) in s.chars().enumerate() {
        if i > 0 && (s.len() - i).is_multiple_of(3) {
            out.push(',');
        }
        out.push(c);
    }
    out
}

fn main() -> ExitCode {
    let args = match parse_args() {
        Ok(a) => a,
        Err(m) => {
            eprintln!("{m}");
            return ExitCode::from(2);
        }
    };
    match run(args) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("error: {e}");
            ExitCode::FAILURE
        }
    }
}
