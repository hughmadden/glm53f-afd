//! One decode step of decoder layers 0-4 and the head, timed by stage, and a per-layer
//! extrapolation to the model's 45 layers.
//!
//! ```sh
//! GLM53F_CHECKPOINT_DIR=/path/to/checkpoint \
//!   cargo run --release -p glm53f-forward --features cuda --example decode_bench
//! ```
//!
//! Environment:
//! - `GLM53F_CHECKPOINT_DIR`: the official checkpoint or its coordinator subset (required);
//! - `GLM53F_BENCH_CONTEXT`: prompt tokens per request before timing (default 4096, so the DSA
//!   layer attends over its full 2,051-token selection);
//! - `GLM53F_BENCH_STEPS`: timed steps per batch size (default 40);
//! - `GLM53F_BENCH_EXPERTS=local` with `GLM53F_EXPERTS_DIR`: run the routed experts of layers 3
//!   and 4 on this GPU (default: zeros, the coordinator's own work only).
//!
//! The routed experts run on the expert ranks in the engine, so the extrapolation leaves them
//! out; the router's host copy of the routes (the step's host round trip) stays in.

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::Arc;

use glm53f_forward::device::{self, Event, Stream};
use glm53f_forward::embed::HostEmbedding;
use glm53f_forward::experts::{ExpertBackend, LocalFp8Experts, ZeroExperts};
use glm53f_forward::forward::{ForwardConfig, GlmForward, StageTimes};
use glm53f_forward::gemm::Fp8Act;
use glm53f_forward::kv::{GlmKv, KvConfig, KvPool};
use glm53f_forward::kvplan::KvLayout;
use glm53f_forward::shape::ModelShape;
use glm53f_forward::weights::{open_checkpoint, DeviceModel};

const LAYERS: usize = 5;

fn env_usize(k: &str, d: usize) -> usize {
    std::env::var(k)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(d)
}

fn ids(seed: u64, n: usize) -> Vec<u32> {
    let mut s = seed;
    (0..n)
        .map(|_| {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            1000 + ((s >> 33) % 150_000) as u32
        })
        .collect()
}

/// Mean milliseconds per (layer, stage) over several passes.
fn mean(times: &[StageTimes]) -> BTreeMap<(usize, &'static str), f64> {
    let mut m: BTreeMap<(usize, &'static str), f64> = BTreeMap::new();
    for t in times {
        for &(l, s, ms) in &t.stages {
            *m.entry((l, s)).or_default() += ms / times.len() as f64;
        }
    }
    m
}

fn get(m: &BTreeMap<(usize, &'static str), f64>, l: usize, stages: &[&str]) -> f64 {
    stages
        .iter()
        .map(|s| m.get(&(l, *s)).copied().unwrap_or(0.0))
        .sum()
}

const KDA: [&str; 4] = ["attn_hc", "kda_proj", "kda_core", "kda_o"];
const DSA: [&str; 6] = [
    "attn_hc",
    "dsa_proj",
    "dsa_cache",
    "dsa_index",
    "dsa_attn",
    "dsa_o",
];
const DENSE: [&str; 2] = ["ffn_hc", "dense_mlp"];
const MOE: [&str; 3] = ["ffn_hc", "router", "shared"];
const ROUTED: [&str; 2] = ["routed", "routed_wait"];
const HEAD: [&str; 3] = ["head_hc", "lm_head", "argmax"];

fn main() {
    let Some(dir) = std::env::var_os("GLM53F_CHECKPOINT_DIR").map(PathBuf::from) else {
        eprintln!("set GLM53F_CHECKPOINT_DIR");
        return;
    };
    let context = env_usize("GLM53F_BENCH_CONTEXT", 4096);
    let steps = env_usize("GLM53F_BENCH_STEPS", 40);
    let (cfg, ckpt) = open_checkpoint(&dir).unwrap();
    let shape = ModelShape::new(&cfg.text, LAYERS).unwrap();
    let model = DeviceModel::load(&ckpt, &shape).unwrap();
    let embed = HostEmbedding::load(&ckpt).unwrap();
    let stream = Arc::new(Stream::new().unwrap());
    let max_req = 8;
    // Every timed step appends a token: 3 batch sizes x (warm-up + two timed runs), and the
    // verify rounds.
    let extra = 3 * (3 + 2 * steps) + 20 * 8 + 64;
    let pages_per = KvLayout::pages_for(context + extra);
    let kv = KvPool::new(
        KvConfig {
            layout: KvLayout::new(&shape, None),
            max_slots: max_req + 1,
            pages: (max_req + 1) * pages_per,
            max_pages: pages_per.div_ceil(4) * 4,
            base_pages: 0,
        },
        stream.clone(),
    )
    .unwrap();
    let experts: Box<dyn ExpertBackend> =
        if std::env::var("GLM53F_BENCH_EXPERTS").as_deref() == Ok("local") {
            let edir = std::env::var_os("GLM53F_EXPERTS_DIR")
                .map(PathBuf::from)
                .unwrap_or(dir.clone());
            Box::new(LocalFp8Experts::new(&edir, 6 << 30, 64, &stream, Fp8Act::Bf16).unwrap())
        } else {
            Box::new(ZeroExperts)
        };
    let fcfg = ForwardConfig {
        max_rows: 256,
        max_verify_rows: 64,
        max_requests: max_req,
        ..ForwardConfig::default()
    };
    let mut fwd = GlmForward::new(model, embed, kv, experts, fcfg).unwrap();
    let (free, total) = device::mem_info().unwrap();
    println!(
        "RTX proxy: {} SMs, peak DRAM {:.0} GB/s; weights {:.2} GB; forward scratch {:.2} GB; {:.1}/{:.1} GiB free",
        device::sm_count().unwrap(),
        device::peak_bandwidth().unwrap() / 1e9,
        fwd.model.bytes as f64 / 1e9,
        fwd.scratch_bytes() as f64 / 1e9,
        free as f64 / (1u64 << 30) as f64,
        total as f64 / (1u64 << 30) as f64
    );
    // Requests with `context` tokens each; the first two prefills timed by stage, KDA through the
    // chain and through the chunked kernel.
    let mut kvs: Vec<GlmKv> = (0..max_req).map(|_| fwd.kv.slot().unwrap()).collect();
    let t0 = std::time::Instant::now();
    let mut prefill_times = Vec::new();
    for (i, kv) in kvs.iter_mut().enumerate() {
        kv.reserve(context + extra).unwrap();
        let timed = i < 2;
        fwd.cfg.kda_chunked_prefill = i == 1;
        fwd.set_timing(timed);
        fwd.prefill(&mut [(kv, &ids(i as u64, context)[..])])
            .unwrap();
        if timed {
            prefill_times.push(fwd.take_times().unwrap());
        }
    }
    fwd.set_timing(false);
    fwd.cfg.kda_chunked_prefill = false;
    fwd.stream().synchronize().unwrap();
    println!(
        "prefilled {max_req} requests of {context} tokens in {:.1} s",
        t0.elapsed().as_secs_f64()
    );
    for (t, what) in prefill_times.iter().zip(["chain", "chunked kernel"]) {
        let kda: f64 = t
            .stages
            .iter()
            .filter(|s| s.1 == "kda_core")
            .map(|s| s.2)
            .sum();
        let total = t.total();
        // Every pass of the prefill (chunks of max_rows) is in the times.
        println!(
            "prefill of {context} tokens in passes of {} rows (layers 0-4 + head), KDA through the {what}: {total:.1} ms ({:.0} tok/s), KDA recurrence {kda:.1} ms ({:.2} us per token per KDA layer)",
            fwd.cfg.max_rows,
            context as f64 / total * 1e3,
            kda * 1e3 / context as f64 / 4.0
        );
    }

    let (e0, e1) = (Event::new().unwrap(), Event::new().unwrap());
    let mut results = Vec::new();
    let mut order: Vec<(usize, &'static str)> = Vec::new();
    for batch in [1usize, 4, 8] {
        let toks = ids(100 + batch as u64, batch);
        // Warm up, then time whole steps (no stage events), then stages.
        for _ in 0..3 {
            let mut rows: Vec<(&mut GlmKv, u32)> = kvs
                .iter_mut()
                .take(batch)
                .zip(toks.iter().copied())
                .collect();
            fwd.decode(&mut rows).unwrap();
        }
        e0.record(fwd.stream()).unwrap();
        let wall = std::time::Instant::now();
        for _ in 0..steps {
            let mut rows: Vec<(&mut GlmKv, u32)> = kvs
                .iter_mut()
                .take(batch)
                .zip(toks.iter().copied())
                .collect();
            fwd.decode(&mut rows).unwrap();
        }
        e1.record(fwd.stream()).unwrap();
        let step_ms = e1.elapsed_ms_since(&e0).unwrap() as f64 / steps as f64;
        let wall_ms = wall.elapsed().as_secs_f64() * 1e3 / steps as f64;
        fwd.set_timing(true);
        let mut times = Vec::new();
        for _ in 0..steps {
            let mut rows: Vec<(&mut GlmKv, u32)> = kvs
                .iter_mut()
                .take(batch)
                .zip(toks.iter().copied())
                .collect();
            fwd.decode(&mut rows).unwrap();
            times.push(fwd.take_times().unwrap());
        }
        fwd.set_timing(false);
        if order.is_empty() {
            order = times[0].stages.iter().map(|s| (s.0, s.1)).collect();
        }
        results.push((batch, step_ms, wall_ms, mean(&times)));
    }
    // A verify window of 8 rows for one request, then its commit (keep 5).
    fwd.set_timing(true);
    let mut vt = Vec::new();
    for i in 0..steps.min(20) {
        let w = ids(200 + i as u64, 8);
        fwd.verify(&mut [(&mut kvs[0], &w[..])]).unwrap();
        let mut t = fwd.take_times().unwrap();
        fwd.commit(&mut [&mut kvs[0]], &[5]).unwrap();
        t.stages.extend(fwd.take_times().unwrap().stages);
        vt.push(t);
    }
    fwd.set_timing(false);
    let verify = mean(&vt);
    let verify_order: Vec<(usize, &'static str)> =
        vt[0].stages.iter().map(|s| (s.0, s.1)).collect();

    println!("\nOne decode step of layers 0-4 + head, ms per stage (mean of {steps} steps; context {context} tokens per request)");
    println!(
        "{:<34} {:>9} {:>9} {:>9} {:>11}",
        "stage", "B = 1", "B = 4", "B = 8", "verify R=8"
    );
    // Stages in pass order (the order the first timed decode step recorded them), then the
    // verify pass's own (the commit).
    let mut keys: Vec<(usize, &'static str)> = Vec::new();
    for t in order.iter().chain(&verify_order) {
        if !keys.contains(t) {
            keys.push(*t);
        }
    }
    for k in &keys {
        let name = if k.0 == usize::MAX {
            format!("pass: {}", k.1)
        } else {
            format!("L{}: {}", k.0, k.1)
        };
        print!("{name:<34}");
        for (_, _, _, m) in &results {
            print!(" {:>9.4}", m.get(k).copied().unwrap_or(0.0));
        }
        println!(" {:>11.4}", verify.get(k).copied().unwrap_or(0.0));
    }
    print!("{:<34}", "sum of stages");
    for (_, _, _, m) in &results {
        print!(" {:>9.3}", m.values().sum::<f64>());
    }
    println!(" {:>11.3}", verify.values().sum::<f64>());
    print!("{:<34}", "step, device time (no stage events)");
    for (_, s, _, _) in &results {
        print!(" {s:>9.3}");
    }
    println!();
    print!("{:<34}", "step, host wall time");
    for (_, _, w, _) in &results {
        print!(" {w:>9.3}");
    }
    println!();

    println!("\nExtrapolation to 45 layers (per-layer costs of layers 0-4 on this card; not a measurement)");
    println!(
        "{:<44} {:>9} {:>9} {:>9}",
        "part", "B = 1", "B = 4", "B = 8"
    );
    let kda_layers = [0usize, 1, 2, 4];
    for (label, f) in [
        ("KDA attention, per layer (mean of 0,1,2,4)", 0),
        ("DSA attention, per layer (layer 3)", 1),
        ("dense FFN, per layer (mean of 0-2)", 2),
        ("MoE FFN on the coordinator, per layer", 3),
        ("routed experts here (on the ranks in AFD)", 4),
        ("head + pass overhead", 5),
    ] {
        print!("{label:<44}");
        for (_, _, _, m) in &results {
            let v = match f {
                0 => kda_layers.iter().map(|&l| get(m, l, &KDA)).sum::<f64>() / 4.0,
                1 => get(m, 3, &DSA),
                2 => (0..3).map(|l| get(m, l, &DENSE)).sum::<f64>() / 3.0,
                3 => (3..5).map(|l| get(m, l, &MOE)).sum::<f64>() / 2.0,
                4 => (3..5).map(|l| get(m, l, &ROUTED)).sum::<f64>() / 2.0,
                _ => get(m, usize::MAX, &HEAD) + get(m, usize::MAX, &["embed"]),
            };
            print!(" {v:>9.3}");
        }
        println!();
    }
    print!("{:<44}", "45 layers: 34 KDA + 11 DSA + 3 dense + 42 MoE");
    for (_, _, _, m) in &results {
        let kda = kda_layers.iter().map(|&l| get(m, l, &KDA)).sum::<f64>() / 4.0;
        let dsa = get(m, 3, &DSA);
        let dense = (0..3).map(|l| get(m, l, &DENSE)).sum::<f64>() / 3.0;
        let moe = (3..5).map(|l| get(m, l, &MOE)).sum::<f64>() / 2.0;
        let head = get(m, usize::MAX, &HEAD) + get(m, usize::MAX, &["embed"]);
        print!(
            " {:>9.2}",
            34.0 * kda + 11.0 * dsa + 3.0 * dense + 42.0 * moe + head
        );
    }
    println!("  ms per step");
}
