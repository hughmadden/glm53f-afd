//! Time the GPU drafter: one draft block for 1, 4 and 16 requests, and the context appends a
//! verify commit and a prompt tail cost; the BF16 drafter, then the FP8 one (`gpu`, "The FP8
//! drafter"; `GLM53F_BENCH_DRAFTERS=bf16` or `fp8` for one of them).
//!
//! ```sh
//! GLM53F_DFLASH_DIR=<drafter> GLM53F_CHECKPOINT_DIR=<GLM-5.3-Flash with embed_tokens, lm_head> \
//!   cargo run -p glm53f-dflash --features cuda --release --example dflash_bench [context] [iterations]
//! ```
//!
//! Every request's context is `context` synthetic rows (default 2,100: past the 2,048 window, the
//! most keys a block query reads). Times are CUDA-event intervals around the queued work (uploads
//! of the per-request metadata included), after 3 warm-up runs.

use std::time::Instant;

use glm53f_dflash::device::{mem_info, DeviceBuffer, Event};
use glm53f_dflash::gpu::{GpuDrafter, GpuSlot, Taps};
use glm53f_dflash::seam::{DraftRequest, Drafter};
use glm53f_dflash::weights::{env_dir, Target, Weights};
use glm53f_dflash::{synth, Dims};

fn main() -> Result<(), String> {
    let args: Vec<String> = std::env::args().collect();
    let context: usize = args
        .get(1)
        .map_or(Ok(2100), |s| s.parse())
        .map_err(|e| format!("context: {e}"))?;
    let iters: usize = args
        .get(2)
        .map_or(Ok(20), |s| s.parse())
        .map_err(|e| format!("iterations: {e}"))?;
    let (Some(dd), Some(cd)) = (
        env_dir("GLM53F_DFLASH_DIR", "the drafter checkpoint directory"),
        env_dir(
            "GLM53F_CHECKPOINT_DIR",
            "a GLM-5.3-Flash directory with embed_tokens and lm_head",
        ),
    ) else {
        return Ok(());
    };
    let d = Dims::GLM53F;
    let t0 = Instant::now();
    let w = Weights::load(&dd, d)?;
    let target = Target::open(&cd, &d)?;
    let head = target.lm_head()?;
    let mask = target.embed_rows(&[d.mask_token])?;
    let which = std::env::var("GLM53F_BENCH_DRAFTERS").unwrap_or_default();
    for fp8 in [false, true] {
        if which == if fp8 { "bf16" } else { "fp8" } {
            continue;
        }
        let (free0, _) = mem_info()?;
        let g = if fp8 {
            GpuDrafter::new_fp8(&w, &head, &mask)?
        } else {
            GpuDrafter::new(&w, &head, &mask)?
        };
        let (free1, total) = mem_info()?;
        println!(
            "\n{} drafter loaded in {:.1} s; weights + LM head on the device: {:.2} GiB ({} bytes by \
             the drafter's count; device {:.1} GiB, {:.1} GiB free)",
            if fp8 { "FP8" } else { "BF16" },
            t0.elapsed().as_secs_f64(),
            free0.saturating_sub(free1) as f64 / (1u64 << 30) as f64,
            g.weight_bytes(),
            total as f64 / (1u64 << 30) as f64,
            free1 as f64 / (1u64 << 30) as f64
        );
        bench(g, &target, context, iters)?;
    }
    Ok(())
}

fn bench(mut g: GpuDrafter, target: &Target, context: usize, iters: usize) -> Result<(), String> {
    let d = g.dims();
    // The working memory `glm53f-serve` reserves up front at 16 and 48 slots.
    let mib = |b: usize| b as f64 / (1u64 << 20) as f64;
    let (w16, w48) = (g.reserve(16)?, g.reserve(48)?);
    println!(
        "working memory reserved for drafts of 16 requests {:.1} MiB, of 48 {:.1} MiB",
        mib(w16),
        mib(w48)
    );
    let n_max = 16;
    let mut slots: Vec<GpuSlot> = Vec::new();
    for i in 0..n_max {
        let mut s = g.new_slot()?;
        let t = synth::taps(100 + i as u64, 0, context, d.tap_width());
        g.append(&mut [(&mut s, &t[..])])?;
        slots.push(s);
    }
    let anchors: Vec<u32> = (0..n_max as u32).map(|i| 1000 + 37 * i).collect();
    let rows = target.embed_rows(&anchors)?;
    let ev = (Event::new()?, Event::new()?);
    println!(
        "GPU shared with other jobs: times are indicative. Context {context} rows per request."
    );
    for n in [1usize, 4, 16] {
        let reqs: Vec<DraftRequest<'_, GpuSlot>> = (0..n)
            .map(|i| DraftRequest {
                slot: &slots[i],
                anchor: anchors[i],
                anchor_embed: &rows[i * d.hidden..(i + 1) * d.hidden],
                temperature: 0.0,
                uniforms: &[],
            })
            .collect();
        for _ in 0..3 {
            g.draft(&reqs)?;
        }
        let mut ms = Vec::with_capacity(iters);
        for _ in 0..iters {
            ev.0.record(g.stream())?;
            g.launch(&reqs)?;
            ev.1.record(g.stream())?;
            ms.push(ev.1.elapsed_ms_since(&ev.0)? as f64);
        }
        let wall = Instant::now();
        let p = g.draft(&reqs)?;
        let wall = wall.elapsed().as_secs_f64() * 1e3;
        ms.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let mean = ms.iter().sum::<f64>() / ms.len() as f64;
        println!(
            "draft block, {n:>2} request(s) ({:>3} rows): mean {mean:.2} ms, median {:.2} ms, min {:.2} ms; one call with the download {wall:.2} ms; request 0 drafts {:?}",
            n * d.block,
            ms[ms.len() / 2],
            ms[0],
            p[0].tokens
        );
    }
    // Context appends: a verify commit (8 rows per request) and a prompt tail, from taps already on
    // the device (as the target forward leaves them).
    for (n, rows_each) in [(1usize, 8usize), (16, 8), (1, 2048)] {
        let t: Vec<u16> = (0..n)
            .flat_map(|i| synth::taps(500 + i as u64, context, rows_each, d.tap_width()))
            .collect();
        let td = DeviceBuffer::from_slice(&t, g.stream())?;
        let mut extra: Vec<GpuSlot> = (0..n).map(|_| g.new_slot()).collect::<Result<_, _>>()?;
        let mut ms = Vec::new();
        for it in 0..(iters.min(10) + 2) {
            for s in extra.iter_mut() {
                g.reset(s);
            }
            let mut items: Vec<(&mut GpuSlot, Taps<'_>)> = extra
                .iter_mut()
                .enumerate()
                .map(|(i, s)| {
                    (
                        s,
                        Taps::Device {
                            ptr: td.ptr::<u16>(i * rows_each * d.tap_width()),
                            rows: rows_each,
                        },
                    )
                })
                .collect();
            ev.0.record(g.stream())?;
            // SAFETY: td holds n x rows_each rows of taps, uploaded and synchronized above.
            unsafe { g.append_taps(&mut items)? };
            ev.1.record(g.stream())?;
            if it >= 2 {
                ms.push(ev.1.elapsed_ms_since(&ev.0)? as f64);
            }
        }
        ms.sort_by(|a, b| a.partial_cmp(b).unwrap());
        println!(
            "append {rows_each} row(s) to {n} request(s): median {:.2} ms, min {:.2} ms",
            ms[ms.len() / 2],
            ms[0]
        );
    }
    Ok(())
}
