//! What the L2 prefetch (`glm53f_forward::prefetch`) buys one KDA layer's decode GEMVs on this
//! GPU: the layer's BF16 weights (q|k|v|b `[24,640][4096]`, f_a|g_a, f_b|g_b, o_proj
//! `[4096][8192]`; random values, the shapes are what count) read by the forward's GEMV at 1 and
//! 8 rows, cold from DRAM, and after a prefetch of their first `budget` bytes.
//!
//! ```sh
//! cargo run --release -p glm53f-forward --features cuda --example l2_prefetch_bench
//! ```
//!
//! Each timed run, on one stream as in the forward: L2 flushed (a 512 MiB memset), the prefetch,
//! waited for, an idle gap (`GLM53F_BENCH_GAP_US`, default 300: the GPU waiting for the ranks),
//! then the layer's GEMVs, timed by events. Per budget, the median of `GLM53F_BENCH_ITERS` runs
//! (default 30) of: the layer at 1 and 8 rows; a probe, the GEMV of the prefetched bytes of
//! q|k|v|b alone (how much of them L2 served); and the prefetch kernel's own time (what an exchange
//! must outlast for the prefetch to hold nothing up). One table per prefetch mode
//! (`prefetch::Mode`) and stride. The GPU may be shared: times are indicative; compare the rows of
//! one run.

use std::time::{Duration, Instant};

use glm53f_forward::device::{self, DeviceBuffer, Event, Stream};
use glm53f_forward::gemm::{Bf16Mat, Gemm, GemmPolicy};
use glm53f_forward::prefetch::{prefix, L2Prefetch, Mode, Range};
use glm53f_forward::weights::Bf16W;

fn env_usize(k: &str, d: usize) -> usize {
    std::env::var(k)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(d)
}

/// Random BF16 values in about [-0.05, 0.05].
fn random_bf16(n: usize, seed: u64) -> Vec<u16> {
    let mut s = seed | 1;
    (0..n)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            let v = ((s >> 40) as f32 / (1u64 << 24) as f32 - 0.5) * 0.1;
            (v.to_bits() >> 16) as u16
        })
        .collect()
}

fn weight(n: usize, k: usize, groups: usize, seed: u64) -> Bf16W {
    let buf = DeviceBuffer::from_slice(&random_bf16(groups * n * k, seed)).unwrap();
    Bf16W { buf, n, k, groups }
}

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(f64::total_cmp);
    v[v.len() / 2]
}

fn spin(d: Duration) {
    let t = Instant::now();
    while t.elapsed() < d {
        std::hint::spin_loop();
    }
}

struct Layer {
    qkvb: Bf16W,
    fga: Bf16W,
    fgb: Bf16W,
    o: Bf16W,
}

impl Layer {
    /// The ranges in the forward's read order (q|k|v|b, f_a|g_a, f_b|g_b, o_proj).
    fn order(&self) -> Vec<Range> {
        [&self.qkvb, &self.fga, &self.fgb, &self.o]
            .iter()
            .map(|w| (w.buf.ptr::<u8>(0).cast_const(), w.buf.bytes()))
            .collect()
    }
}

#[allow(clippy::too_many_arguments)]
fn gemv(
    g: &Gemm,
    x: *const u16,
    ldx: usize,
    xg: usize,
    w: &Bf16Mat,
    rows: usize,
    y: &DeviceBuffer,
    ldo: usize,
    og: usize,
    st: &Stream,
) {
    // SAFETY: x holds 8 rows of 8,192 values, y 8 rows of 24,640 f32; the weights their shapes.
    unsafe { g.gemv(x, ldx, xg, w, rows, y.ptr(0), ldo, og, false, st) }.unwrap()
}

/// The layer's GEMVs over `rows` rows, as `glm53f_forward::forward`'s KDA layer runs them.
fn layer_gemvs(g: &Gemm, l: &Layer, x: &DeviceBuffer, y: &DeviceBuffer, rows: usize, st: &Stream) {
    let xp: *const u16 = x.ptr(0);
    gemv(g, xp, 4096, 0, &l.qkvb.mat(), rows, y, l.qkvb.n, 0, st);
    gemv(g, xp, 4096, 0, &l.fga.mat(), rows, y, 256, 0, st);
    gemv(g, xp, 256, 128, &l.fgb.mat(), rows, y, 16384, 8192, st);
    gemv(g, xp, 8192, 0, &l.o.mat(), rows, y, 4096, 0, st);
}

fn main() {
    if device::device_count() == 0 {
        eprintln!("no CUDA device");
        return;
    }
    let iters = env_usize("GLM53F_BENCH_ITERS", 30);
    let gap = Duration::from_micros(env_usize("GLM53F_BENCH_GAP_US", 300) as u64);
    let l2 = device::l2_bytes().unwrap();
    println!(
        "{} SMs, L2 {:.1} MiB, peak DRAM {:.0} GB/s; idle gap {} us; median of {iters}",
        device::sm_count().unwrap(),
        l2 as f64 / (1 << 20) as f64,
        device::peak_bandwidth().unwrap() / 1e9,
        gap.as_micros()
    );
    let st = Stream::new().unwrap();
    let layer = Layer {
        qkvb: weight(24_640, 4096, 1, 1),
        fga: weight(256, 4096, 1, 2),
        fgb: weight(8192, 128, 2, 3),
        o: weight(4096, 8192, 1, 4),
    };
    let x = DeviceBuffer::from_slice(&random_bf16(8 * 8192, 5)).unwrap();
    let y = DeviceBuffer::alloc(8 * 24_640 * 4).unwrap();
    let flush = DeviceBuffer::alloc(512 << 20).unwrap();
    let gemm = Gemm::new(&st, GemmPolicy::default()).unwrap();
    // The kernel without a model's plan: the ranges are this layer's.
    let mut pf = L2Prefetch::new().unwrap();
    let order = layer.order();
    let total: usize = order.iter().map(|r| r.1).sum();
    let (e0, e1, p0, p1) = (
        Event::new().unwrap(),
        Event::new().unwrap(),
        Event::new().unwrap(),
        Event::new().unwrap(),
    );
    // The probe: the GEMV of the first `budget` bytes of q|k|v|b (whole 8-row blocks).
    let probe_mat = |budget: usize| -> Bf16Mat {
        let n = (budget.min(layer.qkvb.buf.bytes()) / (4096 * 2)) / 8 * 8;
        Bf16Mat {
            n: n.max(8),
            ..layer.qkvb.mat()
        }
    };
    // One timed run: flush, prefetch `budget` bytes (0: none) and wait for it, spin the gap, then
    // `work`; returns the work's time and the prefetch's.
    let run = |pf: &mut L2Prefetch, budget: usize, work: &dyn Fn()| -> (f64, f64) {
        flush.zero_async(&st, 0, flush.bytes()).unwrap();
        let r = prefix(&order, budget);
        if !r.is_empty() {
            p0.record(&st).unwrap();
            pf.launch(&st, &r).unwrap();
            p1.record(&st).unwrap();
        }
        st.synchronize().unwrap();
        let pms = if r.is_empty() {
            0.0
        } else {
            p1.elapsed_ms_since(&p0).unwrap() as f64
        };
        spin(gap);
        e0.record(&st).unwrap();
        work();
        e1.record(&st).unwrap();
        (e1.elapsed_ms_since(&e0).unwrap() as f64, pms)
    };
    let one = || layer_gemvs(&gemm, &layer, &x, &y, 1, &st);
    let eight = || layer_gemvs(&gemm, &layer, &x, &y, 8, &st);
    for _ in 0..3 {
        run(&mut pf, 0, &one);
    }
    let mib = |b: usize| b as f64 / (1 << 20) as f64;
    println!(
        "one KDA layer's BF16 GEMVs ({:.1} MiB: q|k|v|b {:.1}, o_proj {:.1})",
        mib(total),
        mib(layer.qkvb.buf.bytes()),
        mib(layer.o.buf.bytes())
    );
    let budgets: Vec<usize> = [0usize, 16, 32, 48, 56, 64, 72, 96]
        .iter()
        .map(|m| m << 20)
        .collect();
    for (mode, stride) in [(Mode::Load, 128), (Mode::Load, 256), (Mode::Hint, 128)] {
        pf.mode = mode;
        pf.stride = stride;
        println!("\n{mode:?}, every {stride} B:");
        println!(
            "{:>7} {:>9} {:>8} {:>9} {:>8} {:>21} {:>9}",
            "MiB", "R=1 ms", "saved", "R=8 ms", "saved", "probe cold -> warm", "kernel"
        );
        let mut base = (0.0, 0.0);
        for &b in &budgets {
            let m1 = median((0..iters).map(|_| run(&mut pf, b, &one).0).collect());
            let m8 = median((0..iters).map(|_| run(&mut pf, b, &eight).0).collect());
            let pm = probe_mat(b.max(8 << 20));
            let probe = || gemv(&gemm, x.ptr(0), 4096, 0, &pm, 1, &y, pm.n, 0, &st);
            let cold = median((0..iters).map(|_| run(&mut pf, 0, &probe).0).collect());
            let warm: Vec<(f64, f64)> = (0..iters).map(|_| run(&mut pf, b, &probe)).collect();
            if b == 0 {
                base = (m1, m8);
            }
            println!(
                "{:>7.0} {:>9.4} {:>8.4} {:>9.4} {:>8.4} {:>10.4} -> {:>7.4} {:>9.4}",
                mib(b),
                m1,
                base.0 - m1,
                m8,
                base.1 - m8,
                cold,
                median(warm.iter().map(|w| w.0).collect()),
                median(warm.iter().map(|w| w.1).collect())
            );
        }
    }
}
