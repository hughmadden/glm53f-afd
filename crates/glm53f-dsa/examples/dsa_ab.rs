//! Before/after timings of the decode-path kernels: the first implementations
//! (`glm53f_dsa_index_select_v1` with the first chunk plan,
//! `glm53f_dsa_mla_sparse_attn_v1`) against the current ones, for 1, 2, 4 and 8
//! query rows at 4K, 128K and 1M tokens of context.
//!
//!   cargo run --release --features cuda --example dsa_ab [-- quick]
//!
//! Three timings per case, all from CUDA events on one stream:
//! * eager: calls launched back to back from the host (includes host launch
//!   overhead whenever the GPU drains faster than the host launches);
//! * graph: 10 calls captured in one CUDA graph and replayed (how a serving loop
//!   runs steady shapes);
//! * cold: one call after reading 96 MiB has evicted the L2 cache, timed
//!   alone. The other two keep the inputs L2-resident across calls (one layer's
//!   pooled keys at 1M tokens are 35 MB, kv_b_proj is 32 MiB); in a decode step
//!   the other layers' data passes through the cache in between.
//!
//! Sparse attention runs every (splits, head groups) plan and reports the best,
//! and the current kernel also with `glm53f_dsa_mla_plan`'s plan.
//! Allocates about 0.8 GiB.
//!
//!   cargo run --release --features cuda --example dsa_ab -- mid [quick]
//!
//! runs only the sparse attention of the mid-sized passes (9 to 512 rows, one split, 1, 2 and
//! 4 head groups per block) at 128K and 1M tokens of context, graph and cold (each the best of
//! three; the fastest cold time of each row count in bold).

use std::ptr;

use glm53f_dsa::cache::{PAGE_LAYER_BYTES, PAGE_POOLS};
use glm53f_dsa::ffi::{self, CudaStream, DsaCache};
use glm53f_dsa::gpu::{self, check, DeviceBuffer, Stream};
use glm53f_dsa::rng::Rng;

const TOKENS: usize = 1 << 20;
const CONTEXTS: [(usize, &str); 3] = [(4096, "4K"), (131_072, "128K"), (1 << 20, "1M")];

struct Cache {
    _pages: DeviceBuffer,
    _table: DeviceBuffer,
    view: DsaCache,
}

fn build_cache(rng: &mut Rng) -> Cache {
    let pages = TOKENS / 64;
    let mut host = vec![0u8; pages * PAGE_LAYER_BYTES];
    let fill = |dst: &mut [u8], rng: &mut Rng| {
        for c in dst.chunks_mut(8) {
            let r = rng.next_u64().to_le_bytes();
            for (d, s) in c.iter_mut().zip(r) {
                *d = if s & 0x7F == 0x7F { s & 0xF0 } else { s };
            }
        }
    };
    for p in 0..pages {
        let page = &mut host[p * PAGE_LAYER_BYTES..(p + 1) * PAGE_LAYER_BYTES];
        for t in 0..64 {
            let rec = &mut page[t * 528..(t + 1) * 528];
            fill(&mut rec[..512], rng);
            for g in 0..4 {
                rec[512 + 4 * g..516 + 4 * g].copy_from_slice(&(2f32.powi(-7)).to_le_bytes());
            }
        }
        let codes = 64 * 528;
        fill(&mut page[codes..codes + PAGE_POOLS * 128], rng);
        for i in 0..PAGE_POOLS {
            let o = codes + PAGE_POOLS * 128 + 4 * i;
            page[o..o + 4].copy_from_slice(&(2f32.powi(-6)).to_le_bytes());
        }
    }
    let d_pages = DeviceBuffer::from_slice(&host).unwrap();
    let table: Vec<i32> = (0..pages as i32).collect();
    let d_table = DeviceBuffer::from_slice(&table).unwrap();
    let view = DsaCache {
        base: d_pages.as_mut_ptr(),
        page_stride: PAGE_LAYER_BYTES as i64,
        page_tables: d_table.as_ptr(),
        max_pages: pages as i32,
        n_pages: pages as i32,
    };
    Cache { _pages: d_pages, _table: d_table, view }
}

type SelectFn = unsafe extern "C" fn(
    *const f32,
    *const f32,
    f32,
    *const i32,
    *const i32,
    i32,
    i32,
    DsaCache,
    i32,
    i32,
    *mut core::ffi::c_void,
    u64,
    *mut i32,
    *mut i32,
    *mut i32,
    *mut f32,
    CudaStream,
) -> i32;

type AttnFn = unsafe extern "C" fn(
    *const u16,
    *const i32,
    i32,
    *const i32,
    *const i32,
    i32,
    f32,
    DsaCache,
    i32,
    i32,
    *mut core::ffi::c_void,
    u64,
    *mut f32,
    *mut f32,
    CudaStream,
) -> i32;

/// Evicts the L2 by reading 96 MiB: absorb and un-absorb over three kv_b_proj-sized
/// buffers (reads leave clean lines, so the timed call pays no write-backs).
struct Flusher {
    w: Vec<DeviceBuffer>,
    q: DeviceBuffer,
    qa: DeviceBuffer,
    ol: DeviceBuffer,
    o: DeviceBuffer,
}

impl Flusher {
    fn new() -> Result<Self, String> {
        let mut w = Vec::new();
        for _ in 0..3 {
            w.push(DeviceBuffer::zeroed(64 * 512 * 512 * 2)?);
        }
        Ok(Self {
            w,
            q: DeviceBuffer::zeroed(64 * 256 * 4)?,
            qa: DeviceBuffer::alloc(64 * 512 * 2)?,
            ol: DeviceBuffer::zeroed(64 * 512 * 4)?,
            o: DeviceBuffer::alloc(64 * 256 * 4)?,
        })
    }

    fn run(&self, s: CudaStream) -> Result<(), String> {
        for w in &self.w {
            check(unsafe { ffi::glm53f_dsa_mla_absorb_q(self.q.as_ptr(), w.as_ptr(), 1, self.qa.as_mut_ptr(), ptr::null_mut(), s) }, "flush")?;
            check(unsafe { ffi::glm53f_dsa_mla_unabsorb_v(self.ol.as_ptr(), w.as_ptr(), 1, self.o.as_mut_ptr(), s) }, "flush")?;
        }
        Ok(())
    }
}

struct Timing {
    eager: f64,
    graph: f64,
    cold: f64,
}

#[allow(clippy::too_many_arguments)]
fn time_select(
    stream: &Stream,
    cache: &Cache,
    flush: &Flusher,
    f: SelectFn,
    rows: usize,
    pools: usize,
    chunk_pools: i32,
    chunks: i32,
    rng: &mut Rng,
    zero_workspace: bool,
) -> Result<Timing, String> {
    let last = 4 * pools - 1;
    let pos: Vec<i32> = (0..rows).map(|r| (last + 1 - rows + r) as i32).collect();
    let ws_bytes = unsafe { ffi::glm53f_dsa_index_workspace_bytes(rows as i32, chunks) };
    let ws = if zero_workspace { DeviceBuffer::zeroed(ws_bytes as usize)? } else { DeviceBuffer::alloc(ws_bytes as usize)? };
    let q = DeviceBuffer::from_slice(&rng.normals(rows * 4096, 1.0))?;
    let w = DeviceBuffer::from_slice(&rng.normals(rows * 32, 0.18))?;
    let dpos = DeviceBuffer::from_slice(&pos)?;
    let dreq = DeviceBuffer::from_slice(&vec![0i32; rows])?;
    let pools_out = DeviceBuffer::alloc(rows * 512 * 4)?;
    let tokens_out = DeviceBuffer::alloc(rows * 2051 * 4)?;
    let counts = DeviceBuffer::alloc(rows * 8)?;
    let scale = (128f32).powf(-0.5);
    let mut call = |s: CudaStream| {
        check(
            unsafe {
                f(
                    q.as_ptr(), w.as_ptr(), scale, dpos.as_ptr(), dreq.as_ptr(), rows as i32, pools as i32, cache.view, chunk_pools, chunks,
                    ws.ptr(), ws_bytes, pools_out.as_mut_ptr(), tokens_out.as_mut_ptr(), counts.as_mut_ptr(), ptr::null_mut(), s,
                )
            },
            "index_select",
        )
    };
    let eager = gpu::time_stream_us(stream, 3, 20, &mut call)?;
    let graph = gpu::time_graph_us(stream, 10, 10, &mut call)?;
    let cold = gpu::time_cold_us(stream, 10, |s| flush.run(s), &mut call)?;
    Ok(Timing { eager, graph, cold })
}

/// The chunk plan of the first implementation (`glm53f_dsa_index_plan` before
/// the fused kernel).
fn old_plan(rows: usize, pools: usize, sms: i32) -> (i32, i32) {
    let per_row = ((2 * sms as usize) / rows).max(1);
    let mut cp = pools.div_ceil(per_row);
    cp = cp.div_ceil(64) * 64;
    cp = cp.max(256);
    (cp as i32, pools.div_ceil(cp) as i32)
}

struct AttnCase {
    q: DeviceBuffer,
    tokens: DeviceBuffer,
    counts: DeviceBuffer,
    req: DeviceBuffer,
    o: DeviceBuffer,
    lse: DeviceBuffer,
    ws: DeviceBuffer,
    rows: usize,
}

fn attn_case(rng: &mut Rng, rows: usize, context: usize, max_splits: i32) -> Result<AttnCase, String> {
    let mut toks = vec![-1i32; rows * 2051];
    let mut counts = vec![0i32; rows * 2];
    let pools = context / 4;
    for r in 0..rows {
        // 512 distinct pools of the context (ascending) plus a 3-token tail at its end.
        let mut all: Vec<u32> = (0..pools as u32 - 1).collect();
        for i in 0..512 {
            let j = i + rng.below(all.len() - i);
            all.swap(i, j);
        }
        let mut ps: Vec<u32> = all[..512].to_vec();
        ps.sort_unstable();
        let mut v: Vec<i32> = ps.iter().flat_map(|p| (0..4).map(move |i| (4 * p + i) as i32)).collect();
        v.extend([(context - 4) as i32, (context - 3) as i32, (context - 2) as i32]);
        toks[r * 2051..r * 2051 + 2051].copy_from_slice(&v);
        counts[2 * r] = 512;
        counts[2 * r + 1] = 2051;
    }
    let q: Vec<u16> = rng.normals(rows * 64 * 512, 0.5).iter().map(|v| glm53f_dsa::num::f32_to_bf16_bits(*v)).collect();
    let ws_bytes = unsafe { ffi::glm53f_dsa_mla_workspace_bytes(rows as i32, max_splits) } as usize;
    Ok(AttnCase {
        q: DeviceBuffer::from_slice(&q)?,
        tokens: DeviceBuffer::from_slice(&toks)?,
        counts: DeviceBuffer::from_slice(&counts)?,
        req: DeviceBuffer::from_slice(&vec![0i32; rows])?,
        o: DeviceBuffer::alloc(rows * 64 * 512 * 4)?,
        lse: DeviceBuffer::alloc(rows * 64 * 4)?,
        ws: DeviceBuffer::alloc(ws_bytes)?,
        rows,
    })
}

#[derive(Clone, Copy)]
enum Mode<'a> {
    Eager,
    Graph,
    Cold(&'a Flusher),
}

#[allow(clippy::too_many_arguments)]
fn time_attn(stream: &Stream, cache: &Cache, f: AttnFn, c: &AttnCase, splits: i32, groups: i32, mode: Mode) -> Result<f64, String> {
    let ws_bytes = unsafe { ffi::glm53f_dsa_mla_workspace_bytes(c.rows as i32, splits) };
    let mut call = |s: CudaStream| {
        check(
            unsafe {
                f(
                    c.q.as_ptr(), c.tokens.as_ptr(), 2051, c.counts.as_ptr(), c.req.as_ptr(), c.rows as i32, 0.0625, cache.view, splits, groups,
                    c.ws.ptr(), ws_bytes, c.o.as_mut_ptr(), c.lse.as_mut_ptr(), s,
                )
            },
            "sparse_attn",
        )
    };
    match mode {
        Mode::Eager => gpu::time_stream_us(stream, 3, 20, &mut call),
        Mode::Graph => gpu::time_graph_us(stream, 10, 10, &mut call),
        Mode::Cold(flush) => gpu::time_cold_us(stream, 10, |s| flush.run(s), &mut call),
    }
}

/// Best (time, splits, groups) over the plans.
fn best_attn(stream: &Stream, cache: &Cache, f: AttnFn, c: &AttnCase, mode: Mode, quick: bool) -> Result<(f64, i32, i32), String> {
    let splits: &[i32] = if quick { &[4, 16, 32] } else { &[1, 2, 4, 8, 12, 16, 24, 32, 40, 48, 64] };
    let mut best = (f64::INFINITY, 0, 0);
    for &s in splits {
        for g in [1, 2, 4] {
            let t = time_attn(stream, cache, f, c, s, g, mode)?;
            if t < best.0 {
                best = (t, s, g);
            }
        }
    }
    Ok(best)
}

fn main() -> Result<(), String> {
    let quick = std::env::args().any(|a| a == "quick");
    let (free, total) = gpu::mem_info()?;
    let (sms, maj, min) = gpu::device_info()?;
    println!("device: sm_{maj}{min}, {sms} SMs, {} MiB free of {} MiB", free >> 20, total >> 20);
    if free < (2usize << 30) {
        return Err("less than 2 GiB free; not running".into());
    }
    gpu::init()?;
    let stream = Stream::new()?;
    let mut rng = Rng::new(7);
    let cache = build_cache(&mut rng);
    let flush = Flusher::new()?;
    let rows_set: &[usize] = &[1, 2, 4, 8];

    if std::env::args().any(|a| a == "mid") {
        println!("\n## Sparse MLA of mid-sized passes: 2,051 selected tokens per row, one split, µs\n");
        println!("A block holds `groups` 16-head groups (128 x groups threads), so a pass runs rows x 4 / groups blocks on {sms} SMs.\n");
        println!("| rows | context | groups | blocks | graph | cold |");
        println!("|---:|---|---:|---:|---:|---:|");
        let mid: &[usize] = if quick { &[16, 64, 170] } else { &[9, 12, 16, 24, 32, 41, 48, 64, 96, 128, 170, 256, 512] };
        let v2 = ffi::glm53f_dsa_mla_sparse_attn as AttnFn;
        for &(ctx, label) in &CONTEXTS[1..] {
            for &rows in mid {
                let c = attn_case(&mut rng, rows, ctx, 1)?;
                let mut times = Vec::new();
                for groups in [1, 2, 4] {
                    // The best of three: another job on the GPU only slows a run.
                    let (mut graph, mut cold) = (f64::INFINITY, f64::INFINITY);
                    for _ in 0..3 {
                        graph = graph.min(time_attn(&stream, &cache, v2, &c, 1, groups, Mode::Graph)?);
                        cold = cold.min(time_attn(&stream, &cache, v2, &c, 1, groups, Mode::Cold(&flush))?);
                    }
                    times.push((groups, graph, cold));
                }
                let fastest = times.iter().map(|t| t.2).fold(f64::INFINITY, f64::min);
                for (groups, graph, cold) in times {
                    let bold = if cold == fastest { "**" } else { "" };
                    println!("| {rows} | {label} | {groups} | {} | {graph:.1} | {bold}{cold:.1}{bold} |", rows * 4 / groups as usize);
                }
            }
        }
        return Ok(());
    }

    if std::env::args().any(|a| a == "sweep") {
        println!("\n## Indexer chunk-plan sweep (graph replay), µs: chunks -> v1 / v2 prepared\n");
        for &(ctx, label) in &CONTEXTS {
            for &rows in rows_set {
                let pools = ctx / 4;
                let mut line = format!("rows {rows} {label}:");
                for chunks in [1usize, 2, 4, 8, 16, 32, 64, 128, 256, 512] {
                    let cp = (pools.div_ceil(chunks).div_ceil(64) * 64).max(64);
                    let ch = pools.div_ceil(cp);
                    if ch != chunks {
                        continue;
                    }
                    let v1 = time_select(&stream, &cache, &flush, ffi::glm53f_dsa_index_select_v1, rows, pools, cp as i32, ch as i32, &mut rng, false)?;
                    let v2 = time_select(&stream, &cache, &flush, ffi::glm53f_dsa_index_select_prepared, rows, pools, cp as i32, ch as i32, &mut rng, true)?;
                    line += &format!("  {ch}: {:.1}/{:.1}", v1.graph, v2.graph);
                }
                println!("{line}");
            }
        }
        return Ok(());
    }

    println!("\n## Indexer: score + top-512 (whole call), µs\n");
    println!("v1: scoring kernel + merge kernels + finalize kernel, with the first chunk plan. v2: one fused kernel");
    println!("(+ a counter memset; `prepared` skips it), with the current glm53f_dsa_index_plan. Plans: chunks x pools.\n");
    let mut idx = Vec::new();
    for &(ctx, label) in &CONTEXTS {
        for &rows in rows_set {
            let pools = ctx / 4;
            let (mut cp, mut chunks) = (0i32, 0i32);
            unsafe { ffi::glm53f_dsa_index_plan(rows as i32, pools as i32, sms, &mut cp, &mut chunks) };
            let (cp1, chunks1) = old_plan(rows, pools, sms);
            let v1 = time_select(&stream, &cache, &flush, ffi::glm53f_dsa_index_select_v1, rows, pools, cp1, chunks1, &mut rng, false)?;
            let v2 = time_select(&stream, &cache, &flush, ffi::glm53f_dsa_index_select, rows, pools, cp, chunks, &mut rng, false)?;
            let v2p = time_select(&stream, &cache, &flush, ffi::glm53f_dsa_index_select_prepared, rows, pools, cp, chunks, &mut rng, true)?;
            idx.push((rows, label, format!("{chunks1} x {cp1}"), format!("{chunks} x {cp}"), v1, v2, v2p));
        }
    }
    for (title, pick) in [
        ("CUDA graph replay (warm L2)", (|t: &Timing| t.graph) as fn(&Timing) -> f64),
        ("eager launches (warm L2)", |t: &Timing| t.eager),
        ("one call after an L2 flush", |t: &Timing| t.cold),
    ] {
        println!("{title}:\n");
        println!("| rows | context | v1 plan | v2 plan | v1 | v2 | v2 prepared |");
        println!("|---:|---|---|---|---:|---:|---:|");
        for (rows, label, p1, p2, v1, v2, v2p) in &idx {
            println!("| {rows} | {label} | {p1} | {p2} | {:.1} | {:.1} | {:.1} |", pick(v1), pick(v2), pick(v2p));
        }
        println!();
    }

    println!("## Absorb and un-absorb (each reads 16 MiB of kv_b_proj), µs\n");
    println!("| rows | absorb eager | absorb graph | absorb cold | un-absorb eager | un-absorb graph | un-absorb cold |");
    println!("|---:|---:|---:|---:|---:|---:|---:|");
    {
        let kv_b: Vec<u16> = rng.normals(64 * 512 * 512, 0.05).iter().map(|v| glm53f_dsa::num::f32_to_bf16_bits(*v)).collect();
        let d_kvb = DeviceBuffer::from_slice(&kv_b)?;
        for &rows in rows_set {
            let q = DeviceBuffer::from_slice(&rng.normals(rows * 64 * 256, 1.0))?;
            let qa = DeviceBuffer::alloc(rows * 64 * 512 * 2)?;
            let ol = DeviceBuffer::from_slice(&rng.normals(rows * 64 * 512, 1.0))?;
            let o = DeviceBuffer::alloc(rows * 64 * 256 * 4)?;
            let mut ab = |s: CudaStream| check(unsafe { ffi::glm53f_dsa_mla_absorb_q(q.as_ptr(), d_kvb.as_ptr(), rows as i32, qa.as_mut_ptr(), ptr::null_mut(), s) }, "absorb");
            let (ae, ag, ac) = (gpu::time_stream_us(&stream, 3, 20, &mut ab)?, gpu::time_graph_us(&stream, 10, 10, &mut ab)?, gpu::time_cold_us(&stream, 10, |s| flush.run(s), &mut ab)?);
            let mut un = |s: CudaStream| check(unsafe { ffi::glm53f_dsa_mla_unabsorb_v(ol.as_ptr(), d_kvb.as_ptr(), rows as i32, o.as_mut_ptr(), s) }, "unabsorb");
            let (ue, ug, uc) = (gpu::time_stream_us(&stream, 3, 20, &mut un)?, gpu::time_graph_us(&stream, 10, 10, &mut un)?, gpu::time_cold_us(&stream, 10, |s| flush.run(s), &mut un)?);
            println!("| {rows} | {ae:.1} | {ag:.1} | {ac:.1} | {ue:.1} | {ug:.1} | {uc:.1} |");
        }
    }

    println!("\n## Sparse MLA over 2,051 selected tokens, µs\n");
    println!("Best plan (splits x head groups) per kernel and mode; cold: the best graph plan, one call after an L2 flush;");
    println!("mla_plan: the current kernel with glm53f_dsa_mla_plan's plan.\n");
    println!("| rows | context | v1 eager | v2 eager | v1 graph | v2 graph | v2 graph, mla_plan | v1 cold | v2 cold |");
    println!("|---:|---|---:|---:|---:|---:|---:|---:|---:|");
    for &(ctx, label) in &CONTEXTS {
        for &rows in rows_set {
            let c = attn_case(&mut rng, rows, ctx, 64)?;
            let (v1, v2) = (ffi::glm53f_dsa_mla_sparse_attn_v1 as AttnFn, ffi::glm53f_dsa_mla_sparse_attn as AttnFn);
            let v1e = best_attn(&stream, &cache, v1, &c, Mode::Eager, quick)?;
            let v2e = best_attn(&stream, &cache, v2, &c, Mode::Eager, quick)?;
            let v1g = best_attn(&stream, &cache, v1, &c, Mode::Graph, quick)?;
            let v2g = best_attn(&stream, &cache, v2, &c, Mode::Graph, quick)?;
            let (mut plan_splits, mut plan_groups) = (0i32, 0i32);
            unsafe { ffi::glm53f_dsa_mla_plan(rows as i32, 2051, sms, &mut plan_splits, &mut plan_groups) };
            let v2p = time_attn(&stream, &cache, v2, &c, plan_splits, plan_groups, Mode::Graph)?;
            let v1c = time_attn(&stream, &cache, v1, &c, v1g.1, v1g.2, Mode::Cold(&flush))?;
            let v2c = time_attn(&stream, &cache, v2, &c, v2g.1, v2g.2, Mode::Cold(&flush))?;
            println!(
                "| {rows} | {label} | {:.1} ({}x{}) | {:.1} ({}x{}) | {:.1} ({}x{}) | {:.1} ({}x{}) | {v2p:.1} ({plan_splits}x{plan_groups}) | {v1c:.1} | {v2c:.1} |",
                v1e.0, v1e.1, v1e.2, v2e.0, v2e.1, v2e.2, v1g.0, v1g.1, v1g.2, v2g.0, v2g.1, v2g.2
            );
        }
    }
    Ok(())
}
