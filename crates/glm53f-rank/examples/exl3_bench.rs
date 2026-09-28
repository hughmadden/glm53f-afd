//! Throughput of the EXL3 expert kernel on one layer (synthetic weights at
//! the real shape, FP8 rows, random top-8 routes over 288 experts).
//!
//! ```text
//! cargo run -p glm53f-rank --release --features cuda --example exl3_bench
//! cargo run -p glm53f-rank --release --features cuda --example exl3_bench -- --sweep   # also try split counts
//! ```
//!
//! Decode sizes report GB/s: the bytes of the distinct experts' rank slices a
//! call reads (3,173,376 each: three 4,096 x 512 trellis slices at 4 bits
//! plus scale vectors) over the GPU time from the plan to the final reduce.
//! Prefill sizes report rows per second over the same GPU time, and the wall
//! time of the call (uploads and the download included).

use std::collections::BTreeSet;

use glm53f_rank::consts::{HIDDEN, TOPK};
use glm53f_rank::exl3_cuda::{self, Cfg, CudaKernel};
use glm53f_rank::kernel::{ExpertKernel, Rows};
use glm53f_rank::layout::EXPERT_BYTES;
use glm53f_rank::testkit;

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[v.len() / 2]
}

fn main() {
    let sweep = std::env::args().any(|a| a == "--sweep");
    println!("device: {}", exl3_cuda::check_device().expect("device gate"));
    let image = testkit::layer_image(0xBE7C_0001);
    let mut k = CudaKernel::new().unwrap();
    let layer = k.prepare_layer(&image).unwrap();

    let mut cfgs: Vec<Option<Cfg>> = vec![None];
    if sweep {
        for (mt, sk, skd) in [(1, 4, 1), (1, 8, 1), (1, 8, 2), (1, 16, 2), (1, 16, 4), (1, 32, 4), (2, 1, 1), (2, 2, 1), (2, 4, 2)] {
            cfgs.push(Some(Cfg { mt, sk, skd, fp32_swiglu: 0 }));
        }
    }
    for rows in [1usize, 2, 4, 8, 16, 64, 512, 1024, 2048, 4096] {
        let (p, s) = testkit::wire_rows(0xBE7C_1000 + rows as u64, rows);
        let calls = if rows <= 64 { 16 } else { 4 };
        let routes: Vec<(Vec<i32>, Vec<f32>)> =
            (0..calls).map(|c| testkit::routes(0xBE7C_2000 + (rows * 100 + c) as u64, rows, 0)).collect();
        let mut out = vec![0u16; rows * HIDDEN];
        for cfg in &cfgs {
            if rows > 64 && cfg.is_some_and(|c| c.mt == 1) {
                continue;
            }
            k.cfg = *cfg;
            let used = cfg.unwrap_or_else(|| exl3_cuda::default_cfg(rows));
            // Warm up, then time every routing once per pass.
            k.ffn(&layer, Rows::separate(&p, &s, rows).unwrap(), &routes[0].0, &routes[0].1, &mut out).unwrap();
            let (mut gpu, mut wall, mut gbs) = (Vec::new(), Vec::new(), Vec::new());
            let mut phases = [0f64; 5];
            let passes = if rows <= 64 { 8 } else { 2 };
            for _ in 0..passes {
                for (ids, w) in &routes {
                    let t = std::time::Instant::now();
                    let st = k.ffn(&layer, Rows::separate(&p, &s, rows).unwrap(), ids, w, &mut out).unwrap();
                    wall.push(t.elapsed().as_secs_f64() * 1e3);
                    gpu.push(st.gpu_ms as f64);
                    for (p, &m) in phases.iter_mut().zip(&st.phase_ms) {
                        *p += m as f64;
                    }
                    let distinct = ids.iter().collect::<BTreeSet<_>>().len();
                    gbs.push(distinct as f64 * EXPERT_BYTES as f64 / (st.gpu_ms as f64 * 1e-3) / 1e9);
                }
            }
            let n = gpu.len() as f64;
            let (g, wl) = (median(gpu), median(wall));
            let distinct = routes[0].0.iter().collect::<BTreeSet<_>>().len();
            let tag = format!("mt {} sk {:>2} skd {}{}", used.mt, used.sk, used.skd, if cfg.is_none() { " (default)" } else { "" });
            if rows <= 64 {
                println!(
                    "M{rows:<4} {tag:<26} gpu {g:.3} ms  wall {wl:.3} ms  {:.0} GB/s  (~{distinct} experts; phases plan {:.3} gate/up {:.3} epi {:.3} down {:.3} reduce {:.3} ms)",
                    median(gbs),
                    phases[0] / n,
                    phases[1] / n,
                    phases[2] / n,
                    phases[3] / n,
                    phases[4] / n
                );
            } else {
                println!(
                    "M{rows:<4} {tag:<26} gpu {g:.3} ms  wall {wl:.3} ms  {:.0} rows/s gpu, {:.0} rows/s wall  (phases plan {:.3} gate/up {:.3} epi {:.3} down {:.3} reduce {:.3} ms; {} pairs)",
                    rows as f64 / (g * 1e-3),
                    rows as f64 / (wl * 1e-3),
                    phases[0] / n,
                    phases[1] / n,
                    phases[2] / n,
                    phases[3] / n,
                    phases[4] / n,
                    rows * TOPK
                );
            }
        }
    }
}
