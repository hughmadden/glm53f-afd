//! Throughput of the EXL3 expert kernel on one layer (FP8 rows, random top-8
//! routes over 288 experts), on synthetic weights at the real shape or on a
//! real layer image from a rank directory.
//!
//! ```text
//! cargo run -p glm53f-rank --release --features cuda --example exl3_bench -- [options]
//!   --dir <rank-dir> [--layer N]   a real layer image (default: the first layer of the manifest)
//!   --rows 1,8,2048                row counts (default 1,2,4,8,16,64,512,1024,2048,4096)
//!   --sweep                        also try a set of configurations of each kernel family
//!   --cfg mt=4,nt=1,...            also try this configuration (repeatable; keys as in exl3_cuda::Cfg,
//!                                  zero fields take the compiled-in defaults for the row count)
//!   --passes N                     passes over the routings (default 8 up to 64 rows, 3 above)
//!   --min                          report the fastest call's times instead of the medians (a GPU
//!                                  shared with other processes time-slices them into the events)
//! ```
//!
//! Decode sizes report GB/s: the bytes of the distinct experts' rank slices a
//! call reads (3,173,376 each: three 4,096 x 512 trellis slices at 4 bits
//! plus scale vectors) over the GPU time from the plan to the final reduce.
//! Prefill sizes report rows per second over the same GPU time, and the wall
//! time of the call (uploads and the download included). Times are medians
//! over the calls (per phase), or minima with `--min`. The "(default)" lines
//! run the kernel's policy with the environment's overrides
//! (`GLM53F_RANK_SMALL`, `_MID`, `_LARGE`, `_SMALL_MAX`, `_MID_MAX`; README
//! "The kernel").

use std::collections::BTreeSet;
use std::path::PathBuf;

use glm53f_rank::consts::{HIDDEN, TOPK};
use glm53f_rank::exl3_cuda::{self, Cfg, CudaKernel};
use glm53f_rank::kernel::{ExpertKernel, Rows};
use glm53f_rank::layout::EXPERT_BYTES;
use glm53f_rank::manifest::Manifest;
use glm53f_rank::testkit;

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[v.len() / 2]
}

fn cfg(s: &str) -> Cfg {
    Cfg::default().parse_over(s).expect("configuration")
}

/// The sweep. Up to 64 rows: the split kernels' split counts (which change a
/// decode row's bits: pick one configuration for every size up to 64), the
/// planning kernel against the plan in the gate/up blocks, unfused against
/// fused. Above: the kernels as they were before the large-M family (unfused,
/// planning kernel), then the large-M kernels' group rows (`mt`), gate/up warps
/// (`gw`) and rotation warps (`gp`), down chunk width (`nt`), keeping the
/// partial sums in L2 (`discard=1`) and the evict-first trellis loads (`l2=2`);
/// these never change a bit.
fn sweep(rows: usize) -> Vec<Cfg> {
    let list: &[&str] = if rows <= 64 {
        &[
            "big=1,mt=1,sk=8,skd=2,plan=1",
            "big=1,mt=1,sk=8,skd=2,plan=2,fuse=1",
            "big=1,mt=1,sk=4,skd=1",
            "big=1,mt=1,sk=4,skd=2",
            "big=1,mt=1,sk=8,skd=1",
            "big=1,mt=1,sk=16,skd=2",
            "big=1,mt=1,sk=16,skd=4",
            "big=1,mt=2,sk=8,skd=2",
        ]
    } else {
        &[
            "big=1,mt=2,sk=1,skd=1,plan=1,fuse=1,discard=1",
            "big=2,mt=2,gw=8,nt=1",
            "big=2,mt=2,gw=8,nt=2",
            "big=2,mt=2,gw=8,nt=4",
            "big=2,mt=2,gw=16,nt=2",
            "big=2,mt=2,gw=8,gp=2,nt=2",
            "big=2,mt=4,gw=16,nt=1",
            "big=2,mt=4,gw=16,nt=2",
            "big=2,mt=4,gw=16,nt=4",
            "big=2,mt=4,gw=8,nt=2",
            "big=2,mt=2,gw=8,nt=2,discard=1",
            "big=2,mt=2,gw=8,nt=2,l2=2",
            "big=2,mt=4,gw=16,nt=1,l2=2",
            "big=2,mt=4,gw=16,nt=2,l2=2",
        ]
    };
    list.iter().map(|s| cfg(s)).collect()
}

fn main() {
    let mut args = std::env::args().skip(1);
    let (mut do_sweep, mut dir, mut layer) = (false, None::<PathBuf>, None::<u32>);
    let (mut passes_arg, mut use_min) = (None::<usize>, false);
    let mut sizes: Vec<usize> = vec![1, 2, 4, 8, 16, 64, 512, 1024, 2048, 4096];
    let mut extra: Vec<Cfg> = Vec::new();
    while let Some(a) = args.next() {
        match a.as_str() {
            "--sweep" => do_sweep = true,
            "--dir" => dir = Some(PathBuf::from(args.next().expect("--dir DIR"))),
            "--layer" => layer = Some(args.next().expect("--layer N").parse().expect("--layer N")),
            "--rows" => sizes = args.next().expect("--rows LIST").split(',').map(|x| x.parse().expect("row count")).collect(),
            "--cfg" => extra.push(cfg(&args.next().expect("--cfg key=value,..."))),
            "--passes" => passes_arg = Some(args.next().expect("--passes N").parse().expect("--passes N")),
            "--min" => use_min = true,
            other => panic!("unknown argument {other}"),
        }
    }
    println!("device: {}", exl3_cuda::check_device().expect("device gate"));
    let image = match &dir {
        None => {
            println!("weights: synthetic layer");
            testkit::layer_image(0xBE7C_0001)
        }
        Some(d) => {
            let m = Manifest::read(d).expect("manifest");
            let l = layer.unwrap_or_else(|| m.layers.first().expect("an empty manifest").layer);
            let e = m.entry(l).unwrap_or_else(|| panic!("layer {l} is not in {}", d.display()));
            println!("weights: layer {l} of rank {} ({})", m.rank, e.file);
            std::fs::read(d.join(&e.file)).expect("layer image")
        }
    };
    let mut k = CudaKernel::new().expect("kernel (check the GLM53F_RANK_* configuration variables)");
    println!("{}", k.policy.summary());
    let layer = k.prepare_layer(&image).unwrap();
    drop(image);

    for &rows in &sizes {
        let (p, s) = testkit::wire_rows(0xBE7C_1000 + rows as u64, rows);
        let calls = if rows <= 64 { 16 } else { 4 };
        let routes: Vec<(Vec<i32>, Vec<f32>)> =
            (0..calls).map(|c| testkit::routes(0xBE7C_2000 + (rows * 100 + c) as u64, rows, 0)).collect();
        let mut out = vec![0u16; rows * HIDDEN];
        let mut cfgs: Vec<Option<Cfg>> = vec![None];
        if do_sweep {
            cfgs.extend(sweep(rows).into_iter().map(Some));
        }
        cfgs.extend(extra.iter().map(|&c| Some(c)));
        for c in &cfgs {
            k.cfg = *c;
            let used = match k.cfg_for(rows) {
                Ok(u) => u,
                Err(e) => {
                    println!("M{rows:<4} {e}");
                    continue;
                }
            };
            // The split partials grow with the split counts: skip configurations that would need
            // more than 1.5 GB of scratch (the GPU is shared).
            let pairs = (rows * TOPK) as f64;
            if pairs * 4.0 * (2.0 * 512.0 * used.sk as f64 + 4096.0 * used.skd as f64) > 1.5e9 {
                continue;
            }
            // Warm up (a configuration the kernel refuses is skipped), then time every routing once per pass.
            if let Err(e) = k.ffn(&layer, Rows::separate(&p, &s, rows).unwrap(), &routes[0].0, &routes[0].1, &mut out) {
                println!("M{rows:<4} {}: {e}", used.text());
                continue;
            }
            let (mut gpu, mut wall, mut gbs) = (Vec::new(), Vec::new(), Vec::new());
            let mut phases: [Vec<f64>; 5] = Default::default();
            let passes = passes_arg.unwrap_or(if rows <= 64 { 8 } else { 3 });
            for _ in 0..passes {
                for (ids, w) in &routes {
                    let t = std::time::Instant::now();
                    let st = k.ffn(&layer, Rows::separate(&p, &s, rows).unwrap(), ids, w, &mut out).unwrap();
                    wall.push(t.elapsed().as_secs_f64() * 1e3);
                    gpu.push(st.gpu_ms as f64);
                    for (p, &m) in phases.iter_mut().zip(&st.phase_ms) {
                        p.push(m as f64);
                    }
                    let distinct = ids.iter().collect::<BTreeSet<_>>().len();
                    gbs.push(distinct as f64 * EXPERT_BYTES as f64 / (st.gpu_ms as f64 * 1e-3) / 1e9);
                }
            }
            let gmin = gpu.iter().cloned().fold(f64::INFINITY, f64::min);
            let wmin = wall.iter().cloned().fold(f64::INFINITY, f64::min);
            let (g, wl) = if use_min { (gmin, wmin) } else { (median(gpu), median(wall)) };
            let phases = phases.map(|v| if use_min { v.iter().cloned().fold(f64::INFINITY, f64::min) } else { median(v) });
            let gbs = if use_min { gbs.iter().cloned().fold(0.0, f64::max) } else { median(gbs) };
            let distinct = routes[0].0.iter().collect::<BTreeSet<_>>().len();
            let tag = format!("{}{}", used.text(), if c.is_none() { " (default)" } else { "" });
            let ph = format!(
                "plan {:.3} gate/up {:.3} epi {:.3} down {:.3} reduce {:.3} ms",
                phases[0], phases[1], phases[2], phases[3], phases[4]
            );
            if rows <= 64 {
                println!("M{rows:<4} gpu {g:.3} ms  wall {wl:.3} ms  {gbs:.0} GB/s  (~{distinct} experts; {ph})  {tag}");
            } else {
                println!(
                    "M{rows:<4} gpu {g:.3} ms  wall {wl:.3} ms  {:.0} rows/s gpu, {:.0} rows/s wall  ({ph}; {} pairs)  {tag}",
                    rows as f64 / (g * 1e-3),
                    rows as f64 / (wl * 1e-3),
                    rows * TOPK
                );
            }
        }
    }
}
