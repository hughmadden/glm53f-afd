//! Prefill passes in lanes over all 45 decoder layers with the op profile: per operation, the
//! GPU time of each lane's attention sublayer and shared expert (`glm53f_forward::opprof`).
//!
//! ```sh
//! GLM53F_CHECKPOINT_DIR=/path/to/checkpoint \
//!   cargo run --release -p glm53f-forward --features cuda --example prefill_bench
//! ```
//!
//! The coordinator's weights of decoder layers 0-4 are loaded and every later layer runs on the
//! weights of the last loaded layer of its kinds (`DeviceModel::load_repeating`: layer 4 for the
//! KDA MoE layers, layer 3 for the DSA layers), so the pass has the model's 45 layers and its
//! per-layer state in the memory of five. The routed experts return zeros: they run on the
//! expert ranks in the engine, and their time is not the coordinator's.
//!
//! Environment:
//! - `GLM53F_CHECKPOINT_DIR`: the official checkpoint or its coordinator subset (required);
//! - `GLM53F_BENCH_LANE_ROWS`: rows per lane, comma-separated for several runs (default 2048);
//! - `GLM53F_BENCH_LANES`: lanes per pass, comma-separated for several runs (default 2; 1 to 4),
//!   so a pass holds lanes x lane rows;
//! - `GLM53F_BENCH_PASSES`: passes per run (default 4), one request whose prompt they cut (the
//!   later passes' DSA layers attend over more context);
//! - `GLM53F_BENCH_LOADED`: decoder layers loaded (default 5), `GLM53F_BENCH_LAYERS` run (45);
//! - `GLM53F_BENCH_OPS=0`: the same passes without the op profile (its overhead);
//! - `GLM53F_BENCH_KDA_CHUNKED=1`: KDA through the chunked prefill kernel instead of the chain
//!   (`ForwardConfig::kda_chunked_prefill`, a numerics change the KL gate has not passed);
//! - `GLM53F_BENCH_HEAD_GROUPS`, `GLM53F_BENCH_MLA_BLOCK`: `ForwardConfig::prefill_head_groups`
//!   and `mla_block_rows` (neither changes a bit);
//! - `GLM53F_BENCH_TABLES=all`: every pass's `OPS` table (default: the first and the last).

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

use glm53f_forward::device::{self, Stream};
use glm53f_forward::embed::HostEmbedding;
use glm53f_forward::experts::ZeroExperts;
use glm53f_forward::forward::{ForwardConfig, GlmForward};
use glm53f_forward::kv::{KvConfig, KvPool};
use glm53f_forward::kvplan::KvLayout;
use glm53f_forward::shape::ModelShape;
use glm53f_forward::weights::{open_checkpoint, DeviceModel};

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

fn main() {
    let Some(dir) = std::env::var_os("GLM53F_CHECKPOINT_DIR").map(PathBuf::from) else {
        eprintln!("set GLM53F_CHECKPOINT_DIR");
        return;
    };
    let list = |k: &str, d: &str| -> Vec<usize> {
        std::env::var(k)
            .unwrap_or_else(|_| d.into())
            .split(',')
            .filter_map(|v| v.trim().parse().ok())
            .collect()
    };
    let lane_rows = list("GLM53F_BENCH_LANE_ROWS", "2048");
    let lane_counts = list("GLM53F_BENCH_LANES", "2");
    let passes = env_usize("GLM53F_BENCH_PASSES", 4).max(1);
    let loaded = env_usize("GLM53F_BENCH_LOADED", 5);
    let ops_on = std::env::var("GLM53F_BENCH_OPS").map_or(true, |v| v != "0");
    let all_tables = std::env::var("GLM53F_BENCH_TABLES").is_ok_and(|v| v == "all");
    let (cfg, _) = open_checkpoint(&dir).unwrap();
    let layers = env_usize("GLM53F_BENCH_LAYERS", cfg.text.num_hidden_layers as usize);
    let shape = ModelShape::new(&cfg.text, layers).unwrap();
    println!(
        "device: {} SMs, peak DRAM {:.0} GB/s; {layers} decoder layers on the weights of layers \
         0-{} repeated",
        device::sm_count().unwrap(),
        device::peak_bandwidth().unwrap() / 1e9,
        loaded.min(layers) - 1,
    );
    for &lanes in &lane_counts {
        for &lane in &lane_rows {
            run(
                &dir,
                &shape,
                loaded.min(layers),
                lanes,
                lane,
                passes,
                ops_on,
                all_tables,
            );
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn run(
    dir: &std::path::Path,
    shape: &ModelShape,
    loaded: usize,
    lanes: usize,
    lane: usize,
    passes: usize,
    ops_on: bool,
    all_tables: bool,
) {
    let (_, ckpt) = open_checkpoint(dir).unwrap();
    let t0 = Instant::now();
    let model = DeviceModel::load_repeating(&ckpt, shape, loaded).unwrap();
    let embed = HostEmbedding::load(&ckpt).unwrap();
    let weights = model.bytes;
    let stream = Arc::new(Stream::new().unwrap());
    let pass = lanes * lane;
    let prompt = passes * pass;
    let pages = KvLayout::pages_for(prompt + 64);
    let kv = KvPool::new(
        KvConfig {
            layout: KvLayout::new(shape, None),
            max_slots: 1,
            pages,
            max_pages: pages.div_ceil(4) * 4,
            base_pages: 0,
        },
        stream.clone(),
    )
    .unwrap();
    let fcfg = ForwardConfig {
        max_rows: pass,
        lanes,
        max_requests: 1,
        max_verify_rows: 8,
        kda_chunked_prefill: std::env::var("GLM53F_BENCH_KDA_CHUNKED").is_ok_and(|v| v == "1"),
        prefill_head_groups: env_usize("GLM53F_BENCH_HEAD_GROUPS", 4),
        mla_block_rows: env_usize(
            "GLM53F_BENCH_MLA_BLOCK",
            ForwardConfig::default().mla_block_rows,
        ),
        ..ForwardConfig::default()
    };
    let mut fwd = GlmForward::new(model, embed, kv, Box::new(ZeroExperts), fcfg).unwrap();
    let (free, _) = device::mem_info().unwrap();
    let b = fwd.scratch_bytes();
    println!(
        "\n== {lanes} lane(s) of {lane} rows: {passes} passes of {pass} rows (a {prompt}-token \
         prompt); weights {:.2} GB ({:.1} s); forward buffers {:.2} GiB; {:.1} GiB free; op \
         profile {}; KDA through the {}",
        weights as f64 / 1e9,
        t0.elapsed().as_secs_f64(),
        b as f64 / (1u64 << 30) as f64,
        free as f64 / (1u64 << 30) as f64,
        if ops_on { "on" } else { "off" },
        if fcfg.kda_chunked_prefill {
            "chunked kernel"
        } else {
            "chain"
        }
    );
    fwd.set_lane_trace(true, false);
    fwd.set_op_trace(ops_on, false);
    let mut kv = fwd.kv.slot().unwrap();
    kv.reserve(prompt).unwrap();
    let toks = ids(lane as u64, prompt);
    // A warm-up pass on a second slot would need its own pages: the first pass is the warm-up
    // (cuBLAS picks its kernels, events are created); its table is printed too.
    let mut walls = Vec::new();
    for (p, chunk) in toks.chunks(pass).enumerate() {
        let t = Instant::now();
        fwd.prefill(&mut [(&mut kv, chunk)]).unwrap();
        fwd.stream().synchronize().unwrap();
        let wall = t.elapsed().as_secs_f64() * 1e3;
        walls.push(wall);
        let trace = fwd.take_lane_trace().unwrap();
        println!(
            "pass {p} (positions {}..{}): {wall:.1} ms host wall, {:.0} tok/s of coordinator work; \
             PIPE {}",
            p * pass,
            (p + 1) * pass,
            chunk.len() as f64 / wall * 1e3,
            trace.summary()
        );
        if let Some(prof) = fwd.take_op_profile() {
            if all_tables || p == 0 || p + 1 == passes {
                print!("{}", prof.table());
            }
        }
    }
    let mut w = walls[1.min(walls.len() - 1)..].to_vec();
    w.sort_by(f64::total_cmp);
    println!(
        "passes after the first: median {:.1} ms host wall ({} passes)",
        w[w.len() / 2],
        w.len()
    );
}
