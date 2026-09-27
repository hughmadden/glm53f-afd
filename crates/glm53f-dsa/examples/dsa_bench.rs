//! Kernel timings (CUDA events, mean of 20 runs after 3 warm-ups).
//!
//!   cargo run --release --features cuda --example dsa_bench [-- quick]
//!
//! Allocates about 0.7 GiB: a 1M-token paged cache (35,904 B per 64 tokens) with
//! random FP8 latents and pooled keys, plus small buffers. Numbers depend on the
//! GPU; label them with the device they ran on.

use std::ptr;

use glm53f_dsa::cache::{PAGE_LAYER_BYTES, PAGE_POOLS};
use glm53f_dsa::ffi::{self, DsaCache, DsaWindow};
use glm53f_dsa::gpu::{self, check, DeviceBuffer};
use glm53f_dsa::rng::Rng;

const TOKENS: usize = 1 << 20;

struct Cache {
    _pages: DeviceBuffer,
    _table: DeviceBuffer,
    view: DsaCache,
}

fn build_cache(rng: &mut Rng) -> Cache {
    let pages = TOKENS / 64;
    let mut host = vec![0u8; pages * PAGE_LAYER_BYTES];
    for p in 0..pages {
        let page = &mut host[p * PAGE_LAYER_BYTES..(p + 1) * PAGE_LAYER_BYTES];
        // 64 latent records: 512 codes + 4 scales each.
        for t in 0..64 {
            let rec = &mut page[t * 528..(t + 1) * 528];
            for c in rec[..512].chunks_mut(8) {
                let r = rng.next_u64().to_le_bytes();
                for (d, s) in c.iter_mut().zip(r) {
                    *d = if s & 0x7F == 0x7F { s & 0xF0 } else { s };
                }
            }
            for g in 0..4 {
                rec[512 + 4 * g..516 + 4 * g].copy_from_slice(&(2f32.powi(-7)).to_le_bytes());
            }
        }
        let codes = 64 * 528;
        for c in page[codes..codes + PAGE_POOLS * 128].chunks_mut(8) {
            let r = rng.next_u64().to_le_bytes();
            for (d, s) in c.iter_mut().zip(r) {
                *d = if s & 0x7F == 0x7F { s & 0xF0 } else { s };
            }
        }
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

fn index_bench(cache: &Cache, rng: &mut Rng, rows: usize, pools: usize, sms: i32) -> Result<(f64, i32, i32), String> {
    // Rows at consecutive positions ending where `pools` pools are visible.
    let last = 4 * pools - 1;
    let pos: Vec<i32> = (0..rows).map(|r| (last + 1 - rows + r) as i32).collect();
    let max_pools = pools as i32;
    let (mut cp, mut chunks) = (0i32, 0i32);
    unsafe { ffi::glm53f_dsa_index_plan(rows as i32, max_pools, sms, &mut cp, &mut chunks) };
    let ws_bytes = unsafe { ffi::glm53f_dsa_index_workspace_bytes(rows as i32, chunks) };
    let ws = DeviceBuffer::alloc(ws_bytes as usize)?;
    let q = DeviceBuffer::from_slice(&rng.normals(rows * 4096, 1.0))?;
    let w = DeviceBuffer::from_slice(&rng.normals(rows * 32, 0.18))?;
    let dpos = DeviceBuffer::from_slice(&pos)?;
    let dreq = DeviceBuffer::from_slice(&vec![0i32; rows])?;
    let pools_out = DeviceBuffer::alloc(rows * 512 * 4)?;
    let tokens_out = DeviceBuffer::alloc(rows * 2051 * 4)?;
    let counts = DeviceBuffer::alloc(rows * 8)?;
    let scale = (128f32).powf(-0.5);
    let t = gpu::time_us(3, 20, || {
        check(
            unsafe {
                ffi::glm53f_dsa_index_select(
                    q.as_ptr(), w.as_ptr(), scale, dpos.as_ptr(), dreq.as_ptr(), rows as i32, max_pools, cache.view, cp, chunks,
                    ws.ptr(), ws_bytes, pools_out.as_mut_ptr(), tokens_out.as_mut_ptr(), counts.as_mut_ptr(), ptr::null_mut(), ptr::null_mut(),
                )
            },
            "index_select",
        )
    })?;
    Ok((t, chunks, cp))
}

fn attn_bench(cache: &Cache, rng: &mut Rng, rows: usize, n_tok: &dyn Fn(usize) -> usize, splits: i32, groups: i32) -> Result<f64, String> {
    let mut toks = vec![-1i32; rows * 2051];
    let mut counts = vec![0i32; rows * 2];
    for r in 0..rows {
        let n = n_tok(r);
        // 512 random pools (sorted) + 3 tail tokens, spread over the 1M context.
        let mut ps: Vec<i32> = (0..n / 4).map(|_| rng.below(TOKENS / 4 - 1) as i32).collect();
        ps.sort_unstable();
        ps.dedup();
        let mut v: Vec<i32> = ps.iter().flat_map(|p| (0..4).map(move |i| 4 * p + i)).collect();
        while v.len() < n {
            v.push((TOKENS - 4 + v.len() % 3) as i32);
        }
        v.truncate(n);
        v.sort_unstable();
        toks[r * 2051..r * 2051 + n].copy_from_slice(&v);
        counts[2 * r + 1] = n as i32;
    }
    let q: Vec<u16> = rng.normals(rows * 64 * 512, 0.5).iter().map(|v| glm53f_dsa::num::f32_to_bf16_bits(*v)).collect();
    let dq = DeviceBuffer::from_slice(&q)?;
    let dt = DeviceBuffer::from_slice(&toks)?;
    let dc = DeviceBuffer::from_slice(&counts)?;
    let dreq = DeviceBuffer::from_slice(&vec![0i32; rows])?;
    let ws_bytes = unsafe { ffi::glm53f_dsa_mla_workspace_bytes(rows as i32, splits) };
    let ws = DeviceBuffer::alloc(ws_bytes as usize)?;
    let o = DeviceBuffer::alloc(rows * 64 * 512 * 4)?;
    let lse = DeviceBuffer::alloc(rows * 64 * 4)?;
    gpu::time_us(3, 20, || {
        check(
            unsafe {
                ffi::glm53f_dsa_mla_sparse_attn(
                    dq.as_ptr(), dt.as_ptr(), 2051, dc.as_ptr(), dreq.as_ptr(), rows as i32, 0.0625, cache.view, splits, groups, ws.ptr(), ws_bytes,
                    o.as_mut_ptr(), lse.as_mut_ptr(), ptr::null_mut(),
                )
            },
            "sparse_attn",
        )
    })
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
    let mut rng = Rng::new(99);
    let cache = build_cache(&mut rng);

    println!("\n## Indexer: score + top-512 pools (index_select, whole call)\n");
    println!("| rows | context (tokens) | pools | chunks x pools | time (us) | score TFLOP/s | key GB/s |");
    println!("|---:|---:|---:|---|---:|---:|---:|");
    let rows_set: &[usize] = if quick { &[1, 8] } else { &[1, 2, 4, 8] };
    let pool_set: &[usize] = if quick { &[8192, 262_144] } else { &[1024, 8192, 32_768, 65_536, 262_144] };
    for &rows in rows_set {
        for &pools in pool_set {
            let (t, chunks, cp) = index_bench(&cache, &mut rng, rows, pools, sms)?;
            let flop = rows as f64 * pools as f64 * 32.0 * 128.0 * 2.0;
            let bytes = pools as f64 * 132.0;
            println!(
                "| {rows} | {} | {pools} | {chunks} x {cp} | {t:.1} | {:.1} | {:.0} |",
                pools * 4,
                flop / t / 1e6,
                bytes / t / 1e3
            );
        }
    }

    println!("\n## Sparse MLA decode/verify (2,051 selected FP8 latents per row, 64 heads)\n");
    println!("| rows | splits | head groups | time (us) | latent GB/s | TFLOP/s |");
    println!("|---:|---:|---:|---:|---:|---:|");
    let cfgs: &[(i32, i32)] = &[(1, 1), (1, 4), (4, 1), (4, 2), (4, 4), (8, 1), (8, 2), (8, 4), (16, 1), (16, 4), (32, 1), (32, 4)];
    for &rows in rows_set {
        let mut best = (f64::INFINITY, 0, 0);
        for &(splits, groups) in cfgs {
            let t = attn_bench(&cache, &mut rng, rows, &|_| 2051, splits, groups)?;
            let bytes = rows as f64 * 2051.0 * 528.0;
            let flop = rows as f64 * 2051.0 * 64.0 * 512.0 * 4.0;
            if !quick || t < best.0 {
                println!("| {rows} | {splits} | {groups} | {t:.1} | {:.0} | {:.1} |", bytes / t / 1e3, flop / t / 1e6);
            }
            if t < best.0 {
                best = (t, splits, groups);
            }
        }
        println!("| {rows} | **best: {} x {}** | | **{:.1}** | | |", best.1, best.2, best.0);
    }

    println!("\n## Small kernels (8 rows)\n");
    let rows = 8usize;
    let kv_b: Vec<u16> = rng.normals(64 * 512 * 512, 0.05).iter().map(|v| glm53f_dsa::num::f32_to_bf16_bits(*v)).collect();
    let d_kvb = DeviceBuffer::from_slice(&kv_b)?;
    for r in [1usize, 8] {
        let q = DeviceBuffer::from_slice(&rng.normals(r * 64 * 256, 1.0))?;
        let qa = DeviceBuffer::alloc(r * 64 * 512 * 2)?;
        let t = gpu::time_us(3, 20, || check(unsafe { ffi::glm53f_dsa_mla_absorb_q(q.as_ptr(), d_kvb.as_ptr(), r as i32, qa.as_mut_ptr(), ptr::null_mut(), ptr::null_mut()) }, "absorb"))?;
        println!("absorb_q, {r} row(s): {t:.1} us (reads the 16 MiB key half of kv_b_proj: {:.0} GB/s)", 16.0 * 1048576.0 / t / 1e3);
        let ol = DeviceBuffer::from_slice(&rng.normals(r * 64 * 512, 1.0))?;
        let o = DeviceBuffer::alloc(r * 64 * 256 * 4)?;
        let t = gpu::time_us(3, 20, || check(unsafe { ffi::glm53f_dsa_mla_unabsorb_v(ol.as_ptr(), d_kvb.as_ptr(), r as i32, o.as_mut_ptr(), ptr::null_mut()) }, "unabsorb"))?;
        println!("unabsorb_v, {r} row(s): {t:.1} us ({:.0} GB/s on the value half)", 16.0 * 1048576.0 / t / 1e3);
    }
    let lat = DeviceBuffer::from_slice(&rng.normals(rows * 512, 1.0))?;
    let norm = DeviceBuffer::from_slice(&vec![1.0f32; 512])?;
    let pos: Vec<i32> = (0..rows as i32).map(|i| 5000 + i).collect();
    let dpos = DeviceBuffer::from_slice(&pos)?;
    let dreq = DeviceBuffer::from_slice(&vec![0i32; rows])?;
    let t = gpu::time_us(3, 20, || {
        check(unsafe { ffi::glm53f_dsa_mla_latent_write(lat.as_ptr(), norm.as_ptr(), 1e-5, dpos.as_ptr(), dreq.as_ptr(), rows as i32, cache.view, ptr::null_mut()) }, "latent_write")
    })?;
    println!("latent_write (RMSNorm + FP8), 8 rows: {t:.1} us");
    let kr = DeviceBuffer::from_slice(&rng.normals(rows * 128, 1.0))?;
    let g = DeviceBuffer::from_slice(&rng.normals(rows * 128, 1.0))?;
    let lnw = DeviceBuffer::from_slice(&vec![1.0f32; 128])?;
    let lnb = DeviceBuffer::from_slice(&vec![0.0f32; 128])?;
    let ape = DeviceBuffer::from_slice(&rng.normals(512, 0.5))?;
    let tails = DeviceBuffer::zeroed(1552)?;
    let win = DeviceBuffer::from_slice(&[DsaWindow { first_row: 0, rows: 8, start: 5000, accepted: 8 }])?;
    let t = gpu::time_us(3, 20, || {
        check(
            unsafe {
                ffi::glm53f_dsa_index_pool_write(kr.as_ptr(), g.as_ptr(), lnw.as_ptr(), lnb.as_ptr(), 1e-6, ape.as_ptr(), tails.as_ptr(), win.as_ptr(), dreq.as_ptr(), 8, cache.view, ptr::null_mut())
            },
            "pool_write",
        )
    })?;
    println!("index_pool_write (LayerNorm + pool + FP8), 8 rows: {t:.1} us");

    if !quick {
        println!("\n## Prefill-shaped runs\n");
        println!("| kernel | rows | context | time (ms) | per row (us) |");
        println!("|---|---:|---:|---:|---:|");
        for (rows, pools) in [(1024usize, 8192usize), (4096, 8192), (1024, 65_536), (2048, 262_144)] {
            let (t, _, _) = index_bench(&cache, &mut rng, rows, pools, sms)?;
            println!("| index_select | {rows} | {} | {:.2} | {:.1} |", pools * 4, t / 1e3, t / rows as f64);
        }
        for rows in [1024usize, 4096] {
            let t = attn_bench(&cache, &mut rng, rows, &|_| 2051, 1, 4)?;
            println!("| sparse_attn (2,051 tokens, 4 head groups) | {rows} | any | {:.2} | {:.1} |", t / 1e3, t / rows as f64);
        }
        let t = attn_bench(&cache, &mut rng, 2048, &|r| r + 1, 1, 4)?;
        println!("| sparse_attn, dense causal start (row r sees r + 1 tokens) | 2048 | 2048 | {:.2} | {:.1} |", t / 1e3, t / 2048.0);
    }
    Ok(())
}
