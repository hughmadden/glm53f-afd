//! BF16 weight GEMMs at decode sizes (1-8 rows): this crate's GEMV against cuBLAS, for the
//! coordinator's BF16 weights (the KDA projections, the indexer's BF16 projections, the LM
//! head). Prints microseconds per call and the weight bandwidth reached.
//!
//! ```sh
//! cargo run --release -p glm53f-forward --features cuda --example gemm_bench
//! ```
//!
//! Each shape is given enough weight copies to exceed the L2 cache several times over, and the
//! calls rotate through them, so every call reads its weight from DRAM as a real step does.

use core::ffi::c_void;

use glm53f_forward::device::{self, DeviceBuffer, Event, Stream};
use glm53f_forward::gemm::{Bf16Mat, Gemm, GemmPolicy};

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

fn main() {
    if device::device_count() == 0 {
        eprintln!("no CUDA device");
        return;
    }
    let stream = Stream::new().unwrap();
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
