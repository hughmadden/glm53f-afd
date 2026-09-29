//! BF16 weight GEMMs at decode sizes (1-8 rows): this crate's GEMV against cuBLAS, for the
//! coordinator's BF16 weights (the KDA projections, the indexer's BF16 projections, the LM
//! head). Prints microseconds per call and the weight bandwidth reached. With the argument
//! `prefill`: the FP8 projections over 2,048 and 4,096 rows instead, W8A8 (the FP8 tensor-core
//! GEMM on E4M3 activations) against W8A16 (`GemmPolicy::prefill_w8a16`: BF16 tiles of the weight
//! and cuBLAS), with the KDA projections also as the BF16 cuBLAS GEMM they are today and as
//! MXFP8, and the totals over one pass of the model's 45 layers. With `fp8decode`: the KDA
//! projections at 1, 4 and 8 rows as the BF16 GEMV, the FP8 block-128 decode GEMM and the MXFP8
//! decode GEMM.
//!
//! ```sh
//! cargo run --release -p glm53f-forward --features cuda --example gemm_bench [-- prefill|fp8decode]
//! ```
//!
//! Each decode shape is given enough weight copies to exceed the L2 cache several times over,
//! and the calls rotate through them, so every call reads its weight from DRAM as a real step
//! does.

use core::ffi::c_void;

use glm53f_forward::device::{self, DeviceBuffer, Event, Stream};
use glm53f_forward::gemm::{act_quant, Bf16Mat, Fp8Input, Fp8Mat, Gemm, GemmPolicy};
use glm53f_forward::ScaleLayout;

struct Shape {
    name: &'static str,
    groups: usize,
    n: usize,
    k: usize,
    out_f32: bool,
}

const SHAPES: [Shape; 7] = [
    Shape {
        name: "KDA q|k|v|b   24640 x 4096",
        groups: 1,
        n: 24640,
        k: 4096,
        out_f32: false,
    },
    Shape {
        name: "KDA f_a|g_a     256 x 4096",
        groups: 1,
        n: 256,
        k: 4096,
        out_f32: false,
    },
    Shape {
        name: "KDA f_b,g_b 2 x 8192 x 128",
        groups: 2,
        n: 8192,
        k: 128,
        out_f32: false,
    },
    Shape {
        name: "KDA o          4096 x 8192",
        groups: 1,
        n: 4096,
        k: 8192,
        out_f32: false,
    },
    Shape {
        name: "idx proj        288 x 4096",
        groups: 1,
        n: 288,
        k: 4096,
        out_f32: false,
    },
    Shape {
        name: "idx wq_b       4096 x 1536",
        groups: 1,
        n: 4096,
        k: 1536,
        out_f32: false,
    },
    Shape {
        name: "LM head      154880 x 4096",
        groups: 1,
        n: 154880,
        k: 4096,
        out_f32: true,
    },
];

fn fill_random(buf: &DeviceBuffer, seed: u64) {
    // A 16 MiB random block, repeated: the values do not matter for speed, only that they are
    // not a compressible pattern.
    let block = 8 << 20;
    let mut s = seed;
    let host: Vec<u16> = (0..block)
        .map(|_| {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            // A BF16 in about [-0.03, 0.03]: sign, exponent 0x79..0x7a, random mantissa.
            let m = ((s >> 33) & 0x7f) as u16;
            let e = 0x3c00 + (((s >> 40) & 1) as u16) * 0x80;
            e | m | (((s >> 50) & 1) as u16) << 15
        })
        .collect();
    let total = buf.len::<u16>();
    let mut at = 0;
    while at < total {
        let n = block.min(total - at);
        buf.upload_at(at, &host[..n]).unwrap();
        at += n;
    }
}

/// The coordinator's FP8 projections: name, n, k, how many a pass of the model runs, and the
/// scales' layout (the KDA projections twice: block-128 as D2 loads them, and MXFP8).
const FP8_SHAPES: [(&str, usize, usize, usize, ScaleLayout); 12] = [
    ("DSA q_a", 1536, 4096, 11, ScaleLayout::Block128),
    ("DSA kv_a", 512, 4096, 11, ScaleLayout::Block128),
    ("DSA q_b", 16384, 1536, 11, ScaleLayout::Block128),
    ("DSA o", 4096, 16384, 11, ScaleLayout::Block128),
    ("shared gate+up", 4096, 4096, 42, ScaleLayout::Block128),
    ("shared down", 4096, 2048, 42, ScaleLayout::Block128),
    ("dense gate+up", 24576, 4096, 3, ScaleLayout::Block128),
    ("dense down", 4096, 12288, 3, ScaleLayout::Block128),
    ("KDA q|k|v|b (D2)", 24640, 4096, 34, ScaleLayout::Block128),
    ("KDA o (D2)", 4096, 8192, 34, ScaleLayout::Block128),
    ("KDA q|k|v|b (MX)", 24640, 4096, 34, ScaleLayout::Mx32),
    ("KDA o (MX)", 4096, 8192, 34, ScaleLayout::Mx32),
];

/// Milliseconds per call of `f`: 2 warm-up calls, then the mean of `reps`.
fn time_ms(stream: &Stream, reps: usize, f: &dyn Fn()) -> f64 {
    let (e0, e1) = (Event::new().unwrap(), Event::new().unwrap());
    for _ in 0..2 {
        f();
    }
    e0.record(stream).unwrap();
    for _ in 0..reps {
        f();
    }
    e1.record(stream).unwrap();
    e1.elapsed_ms_since(&e0).unwrap() as f64 / reps as f64
}

/// The FP8 projections over prefill-sized passes: W8A8 against W8A16 (and BF16 for the KDA
/// shapes), per call and summed over one pass of the model.
fn prefill(stream: &Stream) {
    let w8a8 = Gemm::new(stream, GemmPolicy::default()).unwrap();
    let w8a16 = Gemm::new(
        stream,
        GemmPolicy {
            prefill_w8a16: true,
            ..GemmPolicy::default()
        },
    )
    .unwrap();
    let max_k = 16384;
    let max_rows = 4096;
    let x = DeviceBuffer::alloc(max_rows * max_k * 2).unwrap();
    fill_random(&x, 9);
    let (xq, xs) = (
        DeviceBuffer::alloc(max_rows * max_k).unwrap(),
        DeviceBuffer::alloc(max_rows * max_k / 128 * 4).unwrap(),
    );
    let out = DeviceBuffer::alloc(max_rows * 24640 * 2).unwrap();
    println!(
        "{:<18} {:>22} {:>9} {:>9} {:>9} {:>9} {:>9}",
        "projection", "rows x n x k", "act_q ms", "W8A8 ms", "W8A16 ms", "BF16 ms", "W8A16 TF"
    );
    for rows in [2048usize, 4096] {
        // Per pass of the model: W8A8 (with its activation quantizations where the forward runs
        // one), W8A16, and the KDA projections in BF16; the MXFP8 KDA projections on their own.
        let (mut t8, mut t16, mut tq, mut kda8, mut kda16, mut kda_bf) =
            (0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
        let (mut mx8, mut mx16) = (0.0, 0.0);
        for &(name, n, k, per_pass, layout) in &FP8_SHAPES {
            let wq = DeviceBuffer::alloc(n * k).unwrap();
            fill_random(&wq, 11);
            // E4M3 codes with bits 0x7F/0xFF would be NaN: clear bit 6 of every byte (codes of
            // magnitude below 2).
            let codes: Vec<u8> = (0..n * k)
                .map(|i| ((i * 37 + 11) % 0x3F) as u8 | ((i as u8 & 1) << 7))
                .collect();
            wq.upload(&codes).unwrap();
            let mx = layout == ScaleLayout::Mx32;
            let scales = if mx {
                // E8M0 2^-7 per row and 32 values of K.
                DeviceBuffer::from_slice(&vec![120u8; n * (k / 32)]).unwrap()
            } else {
                DeviceBuffer::from_slice(&vec![0.01f32; n.div_ceil(128) * (k / 128)]).unwrap()
            };
            let mat = Fp8Mat {
                w: wq.ptr(0),
                scales: scales.ptr(0),
                n,
                k,
                layout,
            };
            let input = Fp8Input {
                bf16: x.ptr(0),
                q: xq.ptr(0),
                scales: xs.ptr(0),
            };
            let reps = if rows * n * k > 64 << 30 { 5 } else { 10 };
            let q = time_ms(stream, reps, &|| {
                unsafe { act_quant(x.ptr(0), xq.ptr(0), xs.ptr(0), rows, k, stream) }.unwrap()
            });
            let a = time_ms(stream, reps, &|| {
                unsafe { w8a8.fp8(&input, &mat, rows, out.ptr(0), stream) }.unwrap()
            });
            let b = time_ms(stream, reps, &|| {
                unsafe { w8a16.fp8(&input, &mat, rows, out.ptr(0), stream) }.unwrap()
            });
            let kda = name.starts_with("KDA");
            let bf = if kda && !mx {
                let wb = DeviceBuffer::alloc(n * k * 2).unwrap();
                fill_random(&wb, 13);
                let m = Bf16Mat {
                    ptr: wb.ptr(0),
                    n,
                    k,
                    ld: k,
                    groups: 1,
                    gstride: n * k,
                };
                time_ms(stream, reps, &|| {
                    unsafe { w8a8.cublas(x.ptr(0), k, 0, &m, rows, out.ptr(0), n, 0, false) }
                        .unwrap()
                })
            } else {
                0.0
            };
            // The forward quantizes the activations of q_b, the DSA and KDA o projections and
            // (fused into the SwiGLU) the down projections; the normed input's E4M3 form comes
            // with the mHC boundary.
            let quant = matches!(
                name,
                "DSA q_b" | "DSA o" | "KDA o (D2)" | "KDA o (MX)" | "shared down" | "dense down"
            );
            if mx {
                mx8 += per_pass as f64 * (a + if quant { q } else { 0.0 });
                mx16 += per_pass as f64 * b;
            } else if kda {
                kda8 += per_pass as f64 * (a + if quant { q } else { 0.0 });
                kda16 += per_pass as f64 * b;
                kda_bf += per_pass as f64 * bf;
            } else {
                t8 += per_pass as f64 * a;
                t16 += per_pass as f64 * b;
                if quant {
                    tq += per_pass as f64 * q;
                }
            }
            println!(
                "{:<18} {:>22} {:>9.3} {:>9.3} {:>9.3} {:>9} {:>9.0}",
                name,
                format!("{rows} x {n} x {k}"),
                q,
                a,
                b,
                if kda && !mx {
                    format!("{bf:.3}")
                } else {
                    "-".to_string()
                },
                2.0 * (rows * n * k) as f64 / b / 1e9
            );
        }
        println!(
            "one pass of {rows} rows, 45 layers: FP8 projections today W8A8 {:.1} ms (+ {:.1} ms of activation quantization), W8A16 {:.1} ms; KDA projections BF16 {:.1} ms, FP8 W8A8 {:.1} ms, FP8 W8A16 {:.1} ms; MXFP8 W8A8 {:.1} ms, MXFP8 W8A16 {:.1} ms",
            t8, tq, t16, kda_bf, kda8, kda16, mx8, mx16
        );
    }
}

/// The KDA projections at decode sizes: the BF16 GEMV (as shipped), the FP8 block-128 decode GEMM
/// (D2) and the MXFP8 decode GEMM, each over weight copies that exceed the L2 cache.
fn fp8_decode(stream: &Stream) {
    let gemm = Gemm::new(stream, GemmPolicy::default()).unwrap();
    let x = DeviceBuffer::alloc(8 * 8192 * 2).unwrap();
    fill_random(&x, 7);
    let out = DeviceBuffer::alloc(8 * 24640 * 2).unwrap();
    let (e0, e1) = (Event::new().unwrap(), Event::new().unwrap());
    println!(
        "{:<12} {:>4} {:>9} {:>9} {:>9} {:>9} {:>9} {:>9}",
        "shape", "rows", "BF16 us", "FP8 us", "MXFP8 us", "BF16 GB/s", "FP8 GB/s", "MX GB/s"
    );
    for (name, n, k) in [("KDA q|k|v|b", 24640usize, 4096usize), ("KDA o", 4096, 8192)] {
        let copies = ((256usize << 20) / (n * k)).clamp(1, 8);
        let bf: Vec<DeviceBuffer> = (0..copies)
            .map(|i| {
                let b = DeviceBuffer::alloc(n * k * 2).unwrap();
                fill_random(&b, 100 + i as u64);
                b
            })
            .collect();
        let codes: Vec<u8> = (0..n * k)
            .map(|i| ((i * 37 + 11) % 0x3F) as u8 | ((i as u8 & 1) << 7))
            .collect();
        let fp8: Vec<(DeviceBuffer, DeviceBuffer, DeviceBuffer)> = (0..copies)
            .map(|_| {
                (
                    DeviceBuffer::from_slice(&codes).unwrap(),
                    DeviceBuffer::from_slice(&vec![0.01f32; n.div_ceil(128) * (k / 128)]).unwrap(),
                    DeviceBuffer::from_slice(&vec![120u8; n * (k / 32)]).unwrap(),
                )
            })
            .collect();
        let mat = |i: usize, layout: ScaleLayout| {
            let (w, s128, smx) = &fp8[i % copies];
            Fp8Mat {
                w: w.ptr(0),
                scales: if layout == ScaleLayout::Mx32 {
                    smx.ptr(0)
                } else {
                    s128.ptr(0)
                },
                n,
                k,
                layout,
            }
        };
        let input = Fp8Input {
            bf16: x.ptr(0),
            q: core::ptr::null(),
            scales: core::ptr::null(),
        };
        for rows in [1usize, 4, 8] {
            let iters = 200;
            // 0: BF16 GEMV; 1: FP8 block-128; 2: MXFP8.
            let time = |which: usize| -> f64 {
                let run = |i: usize| match which {
                    0 => {
                        let w = Bf16Mat {
                            ptr: bf[i % copies].ptr(0),
                            n,
                            k,
                            ld: k,
                            groups: 1,
                            gstride: n * k,
                        };
                        unsafe {
                            gemm.gemv(x.ptr(0), k, 0, &w, rows, out.ptr(0), n, 0, false, stream)
                        }
                        .unwrap()
                    }
                    _ => {
                        let l = if which == 1 {
                            ScaleLayout::Block128
                        } else {
                            ScaleLayout::Mx32
                        };
                        unsafe { gemm.fp8(&input, &mat(i, l), rows, out.ptr(0), stream) }.unwrap()
                    }
                };
                for i in 0..5 {
                    run(i);
                }
                e0.record(stream).unwrap();
                for i in 0..iters {
                    run(i);
                }
                e1.record(stream).unwrap();
                e1.elapsed_ms_since(&e0).unwrap() as f64 * 1e3 / iters as f64
            };
            let (tb, t8, tm) = (time(0), time(1), time(2));
            let gbs = |bytes: usize, us: f64| bytes as f64 / us / 1e3;
            println!(
                "{:<12} {:>4} {:>9.2} {:>9.2} {:>9.2} {:>9.0} {:>9.0} {:>9.0}",
                name,
                rows,
                tb,
                t8,
                tm,
                gbs(n * k * 2, tb),
                gbs(mat(0, ScaleLayout::Block128).bytes(), t8),
                gbs(mat(0, ScaleLayout::Mx32).bytes(), tm)
            );
        }
    }
}

fn main() {
    if device::device_count() == 0 {
        eprintln!("no CUDA device");
        return;
    }
    let stream = Stream::new().unwrap();
    if std::env::args().any(|a| a == "prefill") {
        prefill(&stream);
        return;
    }
    if std::env::args().any(|a| a == "fp8decode") {
        fp8_decode(&stream);
        return;
    }
    let gemm = Gemm::new(&stream, GemmPolicy::default()).unwrap();
    let peak = device::peak_bandwidth().unwrap_or(0.0);
    let (free, _) = device::mem_info().unwrap();
    println!(
        "device: {} SMs, peak DRAM {:.0} GB/s, {:.1} GiB free",
        device::sm_count().unwrap(),
        peak / 1e9,
        free as f64 / (1u64 << 30) as f64
    );
    let x = DeviceBuffer::alloc(8 * 8192 * 2 * 2).unwrap();
    fill_random(&x, 7);
    let out = DeviceBuffer::alloc(8 * 154880 * 4).unwrap();
    let (e0, e1) = (Event::new().unwrap(), Event::new().unwrap());

    println!(
        "{:<28} {:>4} {:>9} {:>9} {:>8} {:>8}",
        "shape", "rows", "GEMV us", "cuBLAS us", "GEMV GB/s", "cuBLAS GB/s"
    );
    for sh in &SHAPES {
        let bytes = sh.groups * sh.n * sh.k * 2;
        // At least 256 MiB of weights in rotation (six times the 4090's L2), at most 8 copies.
        let copies = ((256usize << 20) / bytes).clamp(1, 8);
        let ws: Vec<DeviceBuffer> = (0..copies)
            .map(|i| {
                let b = DeviceBuffer::alloc(bytes).unwrap();
                fill_random(&b, 100 + i as u64);
                b
            })
            .collect();
        let mat = |b: &DeviceBuffer| Bf16Mat {
            ptr: b.ptr(0),
            n: sh.n,
            k: sh.k,
            ld: sh.k,
            groups: sh.groups,
            gstride: sh.n * sh.k,
        };
        let (ldx, xg) = if sh.groups > 1 {
            (sh.groups * sh.k, sh.k)
        } else {
            (sh.k, 0)
        };
        let (ldo, og) = if sh.groups > 1 {
            (sh.groups * sh.n, sh.n)
        } else {
            (sh.n, 0)
        };
        for rows in [1usize, 2, 4, 8] {
            let iters = if bytes > (512 << 20) { 20 } else { 200 };
            let time = |use_gemv: bool| -> f64 {
                let run = |i: usize| {
                    let w = mat(&ws[i % copies]);
                    let o: *mut c_void = out.ptr(0);
                    if use_gemv {
                        unsafe {
                            gemm.gemv(x.ptr(0), ldx, xg, &w, rows, o, ldo, og, sh.out_f32, &stream)
                        }
                        .unwrap();
                    } else {
                        unsafe { gemm.cublas(x.ptr(0), ldx, xg, &w, rows, o, ldo, og, sh.out_f32) }
                            .unwrap();
                    }
                };
                for i in 0..5 {
                    run(i);
                }
                e0.record(&stream).unwrap();
                for i in 0..iters {
                    run(i);
                }
                e1.record(&stream).unwrap();
                e1.elapsed_ms_since(&e0).unwrap() as f64 * 1e3 / iters as f64
            };
            let tg = time(true);
            let tc = time(false);
            println!(
                "{:<28} {:>4} {:>9.2} {:>9.2} {:>8.0} {:>8.0}",
                sh.name,
                rows,
                tg,
                tc,
                bytes as f64 / tg / 1e3,
                bytes as f64 / tc / 1e3
            );
        }
    }
}
