//! Timings of the layer kernels on the local GPU (`--features cuda`).
//!
//! ```text
//! cargo run -p glm53f-layers --features cuda --release --example layers_bench [-- SECTION...]
//! ```
//! Sections: `copy`, `decode`, `prefill`, `mhc`, `router`, `elementwise`, `fused` (the
//! single-launch variants against the pairs), `chain` (one token's 90 boundaries and 42
//! routers in one graph) (default: all).
//!
//! Each figure is the best of 5 launches of a CUDA graph holding many calls, divided by the
//! call count, so launch overhead is excluded. The decode GEMM cycles through enough weight
//! copies to exceed three times the L2 cache, so its weights stream from DRAM as they do in
//! a real decode step. Numbers from a development GPU are proxies for the coordinator's.

use glm53f_layers::cuda::{self, DeviceBuffer, Stream, Timer};
use glm53f_layers::mhc::PARTIAL;
use glm53f_layers::mlp::decode_ksplit;
use glm53f_layers::ops::{
    self, BoundaryDecode, Expand, FinishOut, GemmInput, GemmOutput, Promotion,
};
use glm53f_layers::router::{EXPERTS, ROUTED_SCALE, TOP_K};
use glm53f_layers::testkit::Rng;

const HIDDEN: usize = 4096;

type Res<T> = Result<T, String>;

fn dev<T: Copy>(v: &[T]) -> Res<DeviceBuffer> {
    DeviceBuffer::from_slice(v)
}
fn zeros(bytes: usize) -> Res<DeviceBuffer> {
    DeviceBuffer::zeroed(bytes)
}

struct Bench {
    s: Stream,
    t: Timer,
}

impl Bench {
    /// Microseconds per call: `reps` calls of `f(stream, i)` captured in one graph; best of 5.
    fn time(&self, reps: usize, f: &dyn Fn(&Stream, usize) -> Res<()>) -> Res<f64> {
        let g = self.s.capture(|s| (0..reps).try_for_each(|i| f(s, i)))?;
        g.launch(&self.s)?;
        self.s.sync()?;
        let mut best = f64::MAX;
        for _ in 0..5 {
            self.t.start(&self.s)?;
            g.launch(&self.s)?;
            best = best.min(self.t.stop_ms(&self.s)? as f64 * 1e3 / reps as f64);
        }
        Ok(best)
    }
}

fn section(name: &str, only: &[String]) -> bool {
    only.is_empty() || only.iter().any(|s| s == name)
}

fn main() -> Res<()> {
    let only: Vec<String> = std::env::args().skip(1).collect();
    let b = Bench {
        s: Stream::new()?,
        t: Timer::new()?,
    };
    let peak = cuda::peak_bandwidth()?;
    let l2 = cuda::l2_bytes()? as usize;
    let (free, total) = cuda::mem_info()?;
    println!(
        "device: {} SMs, peak DRAM {:.0} GB/s, L2 {} MiB, free {:.1} of {:.1} GiB",
        cuda::sm_count()?,
        peak / 1e9,
        l2 >> 20,
        free as f64 / (1u64 << 30) as f64,
        total as f64 / (1u64 << 30) as f64
    );
    let mut rng = Rng::new(7);

    if section("copy", &only) {
        let bytes = 256usize << 20;
        let (a, c) = (zeros(bytes)?, zeros(bytes)?);
        let us = b.time(10, &|s, _| cuda::copy_d2d(&c, &a, bytes, s))?;
        println!(
            "\n## Achievable bandwidth\ndevice-to-device copy of 256 MiB: {:.1} us, {:.0} GB/s (read + write), {:.1}% of peak",
            us,
            2.0 * bytes as f64 / us / 1e3,
            200.0 * bytes as f64 / (us * 1e-6) / peak
        );
    }

    if section("decode", &only) {
        println!("\n## FP8 decode GEMM (rows <= 8, CUDA cores)\n");
        println!("| projection | n x k | ksplit | rows | act | us | GB/s | % of peak |");
        println!("|---|---|---:|---:|---|---:|---:|---:|");
        for (name, n, k) in [
            ("shared gate+up", 4096usize, 4096usize),
            ("shared down", 4096, 2048),
            ("dense gate+up", 24576, 4096),
            ("dense down", 4096, 12288),
            ("DSA q_a", 1536, 4096),
            ("DSA o_proj", 4096, 16384),
        ] {
            let wbytes = n * k;
            let copies = (3 * l2).div_ceil(wbytes).clamp(2, 16);
            let w0 = rng.fp8_matrix(n, k);
            let ws: Vec<DeviceBuffer> = (0..copies).map(|_| dev(&w0.data)).collect::<Res<_>>()?;
            let sc = dev(&w0.scale_inv)?;
            let ksplit = decode_ksplit(n, k);
            let partials = zeros(ksplit * 8 * n * 4)?;
            let out = zeros(8 * n * 2)?;
            for rows in [1usize, 2, 4, 8] {
                let x = dev(&rng.bf16_vec(rows * k, 1.0))?;
                let (xq, xs) = (zeros(rows * k)?, zeros(rows * k / 128 * 4)?);
                ops::act_quant(&x, &xq, &xs, rows, k, &b.s)?;
                for a8 in [false, true] {
                    let input = if a8 {
                        GemmInput::Fp8 {
                            q: &xq,
                            scales: &xs,
                        }
                    } else {
                        GemmInput::Bf16(&x)
                    };
                    let us = b.time(copies * 4, &|s, i| {
                        let w = &ws[i % copies];
                        if ksplit == 1 {
                            ops::fp8_gemm_decode(
                                &input,
                                w,
                                &sc,
                                rows,
                                n,
                                k,
                                1,
                                &GemmOutput::Bf16(&out),
                                s,
                            )
                        } else {
                            ops::fp8_gemm_decode(
                                &input,
                                w,
                                &sc,
                                rows,
                                n,
                                k,
                                ksplit,
                                &GemmOutput::Partials(&partials),
                                s,
                            )?;
                            ops::splitk_reduce(&partials, &out, ksplit, rows, n, s)
                        }
                    })?;
                    let bytes = (wbytes
                        + (n / 128) * (k / 128) * 4
                        + rows * k * if a8 { 1 } else { 2 }
                        + rows * n * 2) as f64;
                    println!(
                        "| {name} | {n} x {k} | {ksplit} | {rows} | {} | {us:.2} | {:.0} | {:.1} |",
                        if a8 { "FP8" } else { "BF16" },
                        bytes / us / 1e3,
                        100.0 * bytes / (us * 1e-6) / peak
                    );
                }
            }
        }
    }

    if section("prefill", &only) {
        println!("\n## FP8 prefill GEMM (W8A8, tensor cores)\n");
        println!("| projection | rows x n x k | promotion | ms | TFLOPS |");
        println!("|---|---|---|---:|---:|");
        for (name, n, k) in [
            ("shared gate+up", 4096usize, 4096usize),
            ("dense gate+up", 24576, 4096),
            ("dense down", 4096, 12288),
        ] {
            let w = rng.fp8_matrix(n, k);
            let (dw, dws) = (dev(&w.data)?, dev(&w.scale_inv)?);
            for rows in [512usize, 2048, 4096] {
                let x = dev(&rng.bf16_vec(rows * k, 1.0))?;
                let (xq, xs) = (zeros(rows * k)?, zeros(rows * k / 128 * 4)?);
                ops::act_quant(&x, &xq, &xs, rows, k, &b.s)?;
                let out = zeros(rows * n * 2)?;
                for p in [Promotion::Block128, Promotion::K32] {
                    let us = b.time(5, &|s, _| {
                        ops::fp8_gemm_prefill(&xq, &xs, &dw, &dws, rows, n, k, p, &out, None, s)
                    })?;
                    println!(
                        "| {name} | {rows} x {n} x {k} | {p:?} | {:.3} | {:.0} |",
                        us / 1e3,
                        2.0 * (rows * n * k) as f64 / us / 1e6
                    );
                }
            }
        }
    }

    if section("mhc", &only) {
        println!("\n## mHC boundaries (hidden 4096, 4 streams)\n");
        println!("| step | rows | us | us per row |");
        println!("|---|---:|---:|---:|");
        let slices = HIDDEN / 128;
        let fn_ = dev(&rng.bf16_vec(24 * 4 * HIDDEN, 0.02))?;
        let base = dev(&rng.f32_vec(24, 0.5))?;
        let scale = dev(&[1.0f32, 1.0, 1.0])?;
        let nw = dev(&rng.bf16_vec(HIDDEN, 0.5))?;
        for rows in [1usize, 8, 512, 4096] {
            let st = dev(&rng.bf16_vec(rows * 4 * HIDDEN, 1.0))?;
            let st2 = zeros(rows * 4 * HIDDEN * 2)?;
            let h = dev(&rng.bf16_vec(rows * HIDDEN, 1.0))?;
            let post = dev(&vec![1.0f32; rows * 4])?;
            let comb = dev(&vec![0.25f32; rows * 16])?;
            let parts = zeros(rows * slices * PARTIAL * 4)?;
            let (pp, pc) = (zeros(rows * 16)?, zeros(rows * 64)?);
            let (normed, q, qs) = (
                zeros(rows * HIDDEN * 2)?,
                zeros(rows * HIDDEN)?,
                zeros(rows * slices * 4)?,
            );
            let out = zeros(rows * HIDDEN * 2)?;
            let fo = FinishOut {
                pre: Some(&pp),
                post: Some(&post),
                comb: Some(&pc),
                normed: Some(&normed),
                ..Default::default()
            };
            let fq = FinishOut {
                quant: Some((&q, &qs)),
                ..Default::default()
            };
            let reps = if rows > 512 { 5 } else { 50 };
            let t1 = b.time(reps, &|s, _| {
                ops::hc_project(&st, None, Some((&fn_, &parts)), rows, HIDDEN, s)
            })?;
            let t2 = b.time(reps, &|s, _| {
                ops::hc_finish(&parts, &base, &scale, &st, Some(&nw), &fo, rows, HIDDEN, s)
            })?;
            let e = Expand {
                block_out: &h,
                block_out2: None,
                post: &post,
                comb: &comb,
                streams_out: Some(&st2),
            };
            let t3 = b.time(reps, &|s, _| {
                ops::hc_project(&st, Some(&e), Some((&fn_, &parts)), rows, HIDDEN, s)
            })?;
            let t4 = b.time(reps, &|s, _| {
                ops::hc_finish(&parts, &base, &scale, &st2, Some(&nw), &fq, rows, HIDDEN, s)
            })?;
            let eh = Expand {
                streams_out: None,
                ..e
            };
            let t5 = b.time(reps, &|s, _| {
                ops::hc_head(&st, Some(&eh), &nw, &out, rows, HIDDEN, s)
            })?;
            for (step, us) in [
                ("project", t1),
                ("finish (collapse + norm)", t2),
                ("expand + project", t3),
                ("finish (collapse + norm + FP8 quant)", t4),
                ("boundary = expand + project + finish", t3 + t4),
                ("head (expand + mean + norm)", t5),
            ] {
                println!("| {step} | {rows} | {us:.2} | {:.3} |", us / rows as f64);
            }
        }
    }

    if section("router", &only) {
        println!("\n## Router (288 experts, top-8)\n");
        println!("| rows | logits us | select us | total us |");
        println!("|---:|---:|---:|---:|");
        let w = dev(&rng.bf16_vec(EXPERTS * HIDDEN, 0.02))?;
        let bias = dev(&rng.f32_vec(EXPERTS, 0.01))?;
        for rows in [1usize, 8, 512, 4096] {
            let x = dev(&rng.bf16_vec(rows * HIDDEN, 1.0))?;
            let (lg, ids, wt) = (
                zeros(rows * EXPERTS * 4)?,
                zeros(rows * TOP_K * 4)?,
                zeros(rows * TOP_K * 4)?,
            );
            let reps = if rows > 512 { 5 } else { 50 };
            let a = b.time(reps, &|s, _| {
                ops::router_logits(&x, &w, &lg, rows, EXPERTS, HIDDEN, s)
            })?;
            let c = b.time(reps, &|s, _| {
                ops::router_select(&lg, &bias, &ids, &wt, rows, EXPERTS, TOP_K, ROUTED_SCALE, s)
            })?;
            println!("| {rows} | {a:.2} | {c:.2} | {:.2} |", a + c);
        }
    }

    if section("elementwise", &only) {
        println!("\n## Elementwise\n");
        println!("| kernel | rows x width | us |");
        println!("|---|---|---:|");
        for rows in [8usize, 4096] {
            let reps = if rows > 512 { 10 } else { 50 };
            for inter in [2048usize, 12288] {
                let gu = dev(&rng.bf16_vec(rows * 2 * inter, 2.0))?;
                let (act, q, qs) = (
                    zeros(rows * inter * 2)?,
                    zeros(rows * inter)?,
                    zeros(rows * inter / 128 * 4)?,
                );
                let us = b.time(reps, &|s, _| {
                    ops::swiglu(&gu, None, Some((&q, &qs)), rows, inter, s)
                })?;
                println!("| SwiGLU -> FP8 down input | {rows} x {inter} | {us:.2} |");
                let us = b.time(reps, &|s, _| {
                    ops::swiglu(&gu, Some(&act), None, rows, inter, s)
                })?;
                println!("| SwiGLU -> BF16 | {rows} x {inter} | {us:.2} |");
            }
            let x = dev(&rng.bf16_vec(rows * HIDDEN, 1.0))?;
            let (q, qs) = (zeros(rows * HIDDEN)?, zeros(rows * HIDDEN / 128 * 4)?);
            let us = b.time(reps, &|s, _| ops::act_quant(&x, &q, &qs, rows, HIDDEN, s))?;
            println!("| activation quant | {rows} x {HIDDEN} | {us:.2} |");
            let (w, out) = (dev(&rng.bf16_vec(HIDDEN, 0.5))?, zeros(rows * HIDDEN * 2)?);
            let us = b.time(reps, &|s, _| ops::rmsnorm(&x, &w, &out, rows, HIDDEN, s))?;
            println!("| RMSNorm | {rows} x {HIDDEN} | {us:.2} |");
        }
    }

    if section("fused", &only) {
        println!("\n## Single-launch variants against the pairs they replace (decode)\n");
        // 90 distinct fn matrices (70.8 MB) and 42 distinct router weights (99 MB), cycled so
        // the weights stream from DRAM as they do across a token.
        let nb = 90usize;
        let fns: Vec<DeviceBuffer> = (0..nb)
            .map(|_| dev(&rng.bf16_vec(24 * 4 * HIDDEN, 0.02)))
            .collect::<Res<_>>()?;
        let base = dev(&rng.f32_vec(24, 0.5))?;
        let scale = dev(&[1.0f32, 1.0, 1.0])?;
        let nw = dev(&rng.bf16_vec(HIDDEN, 0.5))?;
        let slices = HIDDEN / 128;
        println!("FFN-site boundary (expansion, RMSNorm, BF16 and FP8 outputs), hidden 4096:\n");
        println!("| rows | pair (project + finish) us | one launch us | one launch, comb deferred us | glm53f_hc_comb alone us |");
        println!("|---:|---:|---:|---:|---:|");
        for rows in [1usize, 2, 4, 8] {
            let st = dev(&rng.bf16_vec(rows * 4 * HIDDEN, 1.0))?;
            let st2 = zeros(rows * 4 * HIDDEN * 2)?;
            let h = dev(&rng.bf16_vec(rows * HIDDEN, 1.0))?;
            let (pin, cin) = (
                dev(&vec![1.0f32; rows * 4])?,
                dev(&vec![0.25f32; rows * 16])?,
            );
            let parts = zeros(rows * slices * PARTIAL * 4)?;
            let sync = ops::sync_buffer(rows)?;
            let (pp, po, pc) = (zeros(rows * 16)?, zeros(rows * 16)?, zeros(rows * 64)?);
            let (normed, q, qs) = (
                zeros(rows * HIDDEN * 2)?,
                zeros(rows * HIDDEN)?,
                zeros(rows * slices * 4)?,
            );
            let fo = |comb: bool| FinishOut {
                pre: Some(&pp),
                post: Some(&po),
                comb: comb.then_some(&pc),
                normed: Some(&normed),
                quant: Some((&q, &qs)),
                ..Default::default()
            };
            let e = || Expand {
                block_out: &h,
                block_out2: None,
                post: &pin,
                comb: &cin,
                streams_out: Some(&st2),
            };
            let pair = b.time(nb, &|s, i| {
                ops::hc_project(&st, Some(&e()), Some((&fns[i], &parts)), rows, HIDDEN, s)?;
                ops::hc_finish(
                    &parts,
                    &base,
                    &scale,
                    &st2,
                    Some(&nw),
                    &fo(true),
                    rows,
                    HIDDEN,
                    s,
                )
            })?;
            let one = |comb: bool| {
                b.time(nb, &|s, i| {
                    let bd = BoundaryDecode {
                        streams_in: &st,
                        expand: Some(e()),
                        fn_: &fns[i],
                        base: &base,
                        scale: &scale,
                        norm_weight: Some(&nw),
                        partials: &parts,
                        sync: &sync,
                    };
                    ops::hc_boundary_decode(&bd, &fo(comb), rows, HIDDEN, s)
                })
            };
            let (inline, deferred) = (one(true)?, one(false)?);
            let comb = b.time(nb, &|s, _| {
                ops::hc_comb(&parts, &base, &scale, &pc, rows, HIDDEN, s)
            })?;
            println!("| {rows} | {pair:.2} | {inline:.2} | {deferred:.2} | {comb:.2} |");
        }

        println!("\nRouter (288 experts, top-8), 42 distinct weights:\n");
        println!("| rows | pair (logits + select) us | one launch us |");
        println!("|---:|---:|---:|");
        let nr = 42usize;
        let rws: Vec<DeviceBuffer> = (0..nr)
            .map(|_| dev(&rng.bf16_vec(EXPERTS * HIDDEN, 0.02)))
            .collect::<Res<_>>()?;
        let bias = dev(&rng.f32_vec(EXPERTS, 0.01))?;
        for rows in [1usize, 2, 4, 8] {
            let x = dev(&rng.bf16_vec(rows * HIDDEN, 1.0))?;
            let (lg, ids, wt) = (
                zeros(rows * EXPERTS * 4)?,
                zeros(rows * TOP_K * 4)?,
                zeros(rows * TOP_K * 4)?,
            );
            let sync = ops::sync_buffer(rows)?;
            let pair = b.time(nr, &|s, i| {
                ops::router_logits(&x, &rws[i], &lg, rows, EXPERTS, HIDDEN, s)?;
                ops::router_select(&lg, &bias, &ids, &wt, rows, EXPERTS, TOP_K, ROUTED_SCALE, s)
            })?;
            let one = b.time(nr, &|s, i| {
                ops::router_fused(
                    &x,
                    &rws[i],
                    &bias,
                    &lg,
                    &sync,
                    &ids,
                    &wt,
                    rows,
                    EXPERTS,
                    HIDDEN,
                    TOP_K,
                    ROUTED_SCALE,
                    s,
                )
            })?;
            println!("| {rows} | {pair:.2} | {one:.2} |");
        }

        println!("\nFP8 decode GEMM with K splits (weights cycled past the L2):\n");
        println!("| projection | n x k | ksplit | rows | GEMM + reduce us | one launch us |");
        println!("|---|---|---:|---:|---:|---:|");
        for (name, n, k) in [("DSA q_a", 1536usize, 4096usize), ("n 512", 512, 4096)] {
            let ksplit = decode_ksplit(n, k);
            let copies = (3 * l2).div_ceil(n * k).clamp(2, 64);
            let w0 = rng.fp8_matrix(n, k);
            let ws: Vec<DeviceBuffer> = (0..copies).map(|_| dev(&w0.data)).collect::<Res<_>>()?;
            let sc = dev(&w0.scale_inv)?;
            let sync = ops::sync_buffer(n / 8)?;
            for rows in [1usize, 8] {
                let x = dev(&rng.bf16_vec(rows * k, 1.0))?;
                let (p, out) = (zeros(ksplit * rows * n * 4)?, zeros(rows * n * 2)?);
                let pair = b.time(copies * 2, &|s, i| {
                    ops::fp8_gemm_decode(
                        &GemmInput::Bf16(&x),
                        &ws[i % copies],
                        &sc,
                        rows,
                        n,
                        k,
                        ksplit,
                        &GemmOutput::Partials(&p),
                        s,
                    )?;
                    ops::splitk_reduce(&p, &out, ksplit, rows, n, s)
                })?;
                let one = b.time(copies * 2, &|s, i| {
                    ops::fp8_gemm_decode_fused(
                        &GemmInput::Bf16(&x),
                        &ws[i % copies],
                        &sc,
                        rows,
                        n,
                        k,
                        ksplit,
                        Some(&p),
                        Some(&sync),
                        &out,
                        s,
                    )
                })?;
                println!("| {name} | {n} x {k} | {ksplit} | {rows} | {pair:.2} | {one:.2} |");
            }
        }
    }

    if section("chain", &only) {
        // One token's mHC boundaries and routers: 45 layers, each an attention boundary (the
        // expansion of the previous FFN output, except layer 0) and an FFN boundary (the
        // expansion of the attention output), and a router after the FFN boundary of layers
        // 3..44. Distinct weights per boundary and per router (170 MB, past the L2). The
        // sublayers are left out: their outputs are fixed buffers. "comb deferred" leaves the
        // Sinkhorn out of the chain: in a decode step glm53f_hc_comb computes it on a second
        // stream while the sublayer runs, off the critical path this chain measures.
        println!("\n## One token: 90 boundaries + 42 routers, in one graph\n");
        println!("| rows | pairs us | single launches, comb inline us | single launches, comb deferred us |");
        println!("|---:|---:|---:|---:|");
        let slices = HIDDEN / 128;
        let fns: Vec<DeviceBuffer> = (0..90)
            .map(|_| dev(&rng.bf16_vec(24 * 4 * HIDDEN, 0.02)))
            .collect::<Res<_>>()?;
        let bases: Vec<DeviceBuffer> = (0..90)
            .map(|_| dev(&rng.f32_vec(24, 0.5)))
            .collect::<Res<_>>()?;
        let scale = dev(&[1.0f32, 1.0, 1.0])?;
        let nws: Vec<DeviceBuffer> = (0..90)
            .map(|_| dev(&rng.bf16_vec(HIDDEN, 0.5)))
            .collect::<Res<_>>()?;
        let rws: Vec<DeviceBuffer> = (0..42)
            .map(|_| dev(&rng.bf16_vec(EXPERTS * HIDDEN, 0.02)))
            .collect::<Res<_>>()?;
        let bias = dev(&rng.f32_vec(EXPERTS, 0.01))?;
        for rows in [1usize, 8] {
            let s0 = dev(&rng.bf16_vec(rows * 4 * HIDDEN, 1.0))?;
            let st = [zeros(rows * 4 * HIDDEN * 2)?, zeros(rows * 4 * HIDDEN * 2)?];
            let (attn_out, ffn_out) = (
                dev(&rng.bf16_vec(rows * HIDDEN, 0.5))?,
                dev(&rng.bf16_vec(rows * HIDDEN, 0.5))?,
            );
            let post = [zeros(rows * 16)?, zeros(rows * 16)?];
            let comb = [zeros(rows * 64)?, zeros(rows * 64)?];
            let parts = zeros(rows * slices * PARTIAL * 4)?;
            let (normed, q, qs) = (
                zeros(rows * HIDDEN * 2)?,
                zeros(rows * HIDDEN)?,
                zeros(rows * slices * 4)?,
            );
            let (lg, ids, wt) = (
                zeros(rows * EXPERTS * 4)?,
                zeros(rows * TOP_K * 4)?,
                zeros(rows * TOP_K * 4)?,
            );
            let sync = ops::sync_buffer(rows)?;
            // Boundary b: residual streams, expansion input, weights in and out.
            let token = |s: &Stream, fused: bool, comb_inline: bool| -> Res<()> {
                for bnd in 0..90usize {
                    let layer = bnd / 2;
                    let ffn = bnd % 2 == 1;
                    // Boundary 0 collapses the input streams; boundary 1 expands onto them; later
                    // boundaries alternate between two stream buffers.
                    let res = if bnd <= 1 { &s0 } else { &st[(bnd + 1) % 2] };
                    let out = &st[bnd % 2];
                    let e = (bnd > 0).then(|| Expand {
                        block_out: if ffn { &attn_out } else { &ffn_out },
                        block_out2: None,
                        post: &post[(bnd + 1) % 2],
                        comb: &comb[(bnd + 1) % 2],
                        streams_out: Some(out),
                    });
                    let fo = FinishOut {
                        post: Some(&post[bnd % 2]),
                        comb: (comb_inline || !fused).then_some(&comb[bnd % 2]),
                        normed: Some(&normed),
                        quant: ffn.then_some((&q, &qs)),
                        ..Default::default()
                    };
                    let collapse_from = if bnd == 0 { &s0 } else { out };
                    if fused {
                        let bd = BoundaryDecode {
                            streams_in: res,
                            expand: e,
                            fn_: &fns[bnd],
                            base: &bases[bnd],
                            scale: &scale,
                            norm_weight: Some(&nws[bnd]),
                            partials: &parts,
                            sync: &sync,
                        };
                        ops::hc_boundary_decode(&bd, &fo, rows, HIDDEN, s)?;
                    } else {
                        ops::hc_project(
                            res,
                            e.as_ref(),
                            Some((&fns[bnd], &parts)),
                            rows,
                            HIDDEN,
                            s,
                        )?;
                        ops::hc_finish(
                            &parts,
                            &bases[bnd],
                            &scale,
                            collapse_from,
                            Some(&nws[bnd]),
                            &fo,
                            rows,
                            HIDDEN,
                            s,
                        )?;
                    }
                    if ffn && layer >= 3 {
                        let rw = &rws[layer - 3];
                        if fused {
                            ops::router_fused(
                                &normed,
                                rw,
                                &bias,
                                &lg,
                                &sync,
                                &ids,
                                &wt,
                                rows,
                                EXPERTS,
                                HIDDEN,
                                TOP_K,
                                ROUTED_SCALE,
                                s,
                            )?;
                        } else {
                            ops::router_logits(&normed, rw, &lg, rows, EXPERTS, HIDDEN, s)?;
                            ops::router_select(
                                &lg,
                                &bias,
                                &ids,
                                &wt,
                                rows,
                                EXPERTS,
                                TOP_K,
                                ROUTED_SCALE,
                                s,
                            )?;
                        }
                    }
                }
                Ok(())
            };
            let old = b.time(1, &|s, _| token(s, false, true))?;
            let new_inline = b.time(1, &|s, _| token(s, true, true))?;
            let new_deferred = b.time(1, &|s, _| token(s, true, false))?;
            println!("| {rows} | {old:.1} | {new_inline:.1} | {new_deferred:.1} |");
        }
    }
    Ok(())
}
