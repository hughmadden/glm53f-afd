//! The FP8 block-128 GEMM dispatch (`Gemm::fp8`, feature `cuda`; skips without a device) against
//! exact products, over more than 8 rows: the W8A8 path (E4M3 activations per 128-group, the FP8
//! tensor cores) and the W8A16 path (`GemmPolicy::prefill_w8a16`: BF16 tiles of the weight and
//! cuBLAS). Shapes with a partial last block of rows (the fused KDA q|k|v|b projection's 64-row
//! beta block) and weights wider than one tile. Up to 8 rows the option changes nothing.
#![cfg(feature = "cuda")]

use glm53f_forward::device::{self, DeviceBuffer, Stream};
use glm53f_forward::gemm::{act_quant, Fp8Input, Fp8Mat, Gemm, GemmPolicy, DEQUANT_BYTES};
use glm53f_layers::bf16;
use glm53f_layers::fp8::{self, e4m3_to_f32};
use glm53f_layers::testkit::Rng;

fn gpu() -> Option<Stream> {
    if device::device_count() == 0 {
        eprintln!("skip: no CUDA device");
        return None;
    }
    Some(Stream::new().unwrap())
}

/// `Gemm::fp8` over `rows` rows of `x` with the engine `g`: BF16 outputs as f32.
fn run(g: &Gemm, s: &Stream, x: &[u16], rows: usize, w: &fp8::Fp8Matrix) -> Vec<f32> {
    let (n, k) = (w.rows, w.cols);
    let dx = DeviceBuffer::from_slice(x).unwrap();
    let (xq, xs) = (
        DeviceBuffer::alloc(rows * k).unwrap(),
        DeviceBuffer::alloc(rows * (k / 128) * 4).unwrap(),
    );
    if g.policy.fp8_needs_quant(rows) {
        unsafe { act_quant(dx.ptr(0), xq.ptr(0), xs.ptr(0), rows, k, s) }.unwrap();
    }
    let (dw, ds) = (
        DeviceBuffer::from_slice(&w.data).unwrap(),
        DeviceBuffer::from_slice(&w.scale_inv).unwrap(),
    );
    let out = DeviceBuffer::zeroed(rows * n * 2).unwrap();
    let mat = Fp8Mat {
        w: dw.ptr(0),
        scales: ds.ptr(0),
        n,
        k,
    };
    let input = Fp8Input {
        bf16: dx.ptr(0),
        q: xq.ptr(0),
        scales: xs.ptr(0),
    };
    unsafe { g.fp8(&input, &mat, rows, out.ptr(0), s) }.unwrap();
    s.synchronize().unwrap();
    bf16::widen(&out.download::<u16>(rows * n).unwrap())
}

/// Exact `x . W^T` with BF16 activations (W8A16: every product exact), in f64, and per output
/// the sum of |products| (an error scale).
fn exact(x: &[u16], rows: usize, w: &fp8::Fp8Matrix) -> (Vec<f64>, Vec<f64>) {
    let (n, k) = (w.rows, w.cols);
    let xf: Vec<f64> = x.iter().map(|&b| bf16::to_f32(b) as f64).collect();
    let wf: Vec<f64> = (0..n * k)
        .map(|i| e4m3_to_f32(w.data[i]) as f64 * w.scale(i / k, i % k) as f64)
        .collect();
    let mut out = vec![0f64; rows * n];
    let mut mag = vec![0f64; rows * n];
    for m in 0..rows {
        let xr = &xf[m * k..(m + 1) * k];
        for o in 0..n {
            let wr = &wf[o * k..(o + 1) * k];
            let (mut a, mut b) = (0f64, 0f64);
            for (p, q) in xr.iter().zip(wr) {
                a += p * q;
                b += (p * q).abs();
            }
            out[m * n + o] = a;
            mag[m * n + o] = b;
        }
    }
    (out, mag)
}

/// Mean and worst |got - exact| / mag.
fn errs(got: &[f32], exact: &[f64], mag: &[f64]) -> (f64, f64) {
    let (mut mean, mut worst) = (0f64, 0f64);
    for i in 0..got.len() {
        let e = (got[i] as f64 - exact[i]).abs() / mag[i].max(1e-30);
        mean += e;
        worst = worst.max(e);
    }
    (mean / got.len() as f64, worst)
}

#[test]
fn w8a16_prefill_is_the_exact_products_within_bf16_rounding() {
    let Some(s) = gpu() else { return };
    let w8a8 = Gemm::new(&s, GemmPolicy::default()).unwrap();
    let w8a16 = Gemm::new(
        &s,
        GemmPolicy {
            prefill_w8a16: true,
            ..GemmPolicy::default()
        },
    )
    .unwrap();
    assert_eq!(w8a16.bytes(), w8a8.bytes() + DEQUANT_BYTES);
    let mut rng = Rng::new(120);
    // (rows, n, k): a partial last block of rows (64 past the last whole block, as q|k|v|b);
    // a weight three tiles wide (k 16,384: tiles of 1,408 rows); a plain shape.
    for (rows, n, k) in [
        (100usize, 320usize, 512usize),
        (24, 4160, 16384),
        (200, 256, 4096),
    ] {
        assert!(Gemm::w8a16_tile_rows(n, k) * k * 2 <= DEQUANT_BYTES);
        let w = rng.fp8_matrix(n, k);
        let x = rng.bf16_vec(rows * k, 1.0);
        let (ex, mag) = exact(&x, rows, &w);
        let a8 = run(&w8a8, &s, &x, rows, &w);
        let a16 = run(&w8a16, &s, &x, rows, &w);
        let (m8, w8) = errs(&a8, &ex, &mag);
        let (m16, w16) = errs(&a16, &ex, &mag);
        eprintln!(
            "{rows} x {n} x {k} (tiles of {} rows): against exact W8A16, error / sum|x w|: W8A8 mean {m8:.2e} worst {w8:.2e}; W8A16 mean {m16:.2e} worst {w16:.2e}",
            Gemm::w8a16_tile_rows(n, k)
        );
        // W8A16: the BF16 tiles round each weight once (2^-9 relative) and the output is BF16
        // (2^-9 of the value, at most the magnitude): 2^-8 of sum|x w| bounds both.
        for i in 0..a16.len() {
            let e = (a16[i] as f64 - ex[i]).abs();
            assert!(
                e <= mag[i] * 2f64.powi(-8),
                "{rows} x {n} x {k} [{i}]: {} vs {} (sum|x w| {})",
                a16[i],
                ex[i],
                mag[i]
            );
        }
        // Far closer than the E4M3 activations.
        assert!(m16 * 4.0 < m8, "W8A16 {m16:.2e} vs W8A8 {m8:.2e}");
        // Up to 8 rows the option changes nothing: the same decode GEMM, bit for bit.
        for r in [1usize, 8] {
            let d8 = run(&w8a8, &s, &x[..r * k], r, &w);
            let d16 = run(&w8a16, &s, &x[..r * k], r, &w);
            assert_eq!(
                d8.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                d16.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                "{r} rows"
            );
        }
    }
}

/// `GemmPolicy::kda_prefill_w8a8`: with W8A16 on, the FP8 KDA projections (`Gemm::fp8_kda`)
/// keep the W8A8 GEMM, bit for bit, and the others take W8A16.
#[test]
fn kda_projections_can_keep_w8a8() {
    let Some(s) = gpu() else { return };
    let policy = |w8a16: bool, kda_w8a8: bool| GemmPolicy {
        prefill_w8a16: w8a16,
        kda_prefill_w8a8: kda_w8a8,
        ..GemmPolicy::default()
    };
    let (w8a8, w8a16, hybrid) = (
        Gemm::new(&s, policy(false, false)).unwrap(),
        Gemm::new(&s, policy(true, false)).unwrap(),
        Gemm::new(&s, policy(true, true)).unwrap(),
    );
    assert!(hybrid.policy.needs_quant(64, true) && !hybrid.policy.needs_quant(64, false));
    assert!(
        !hybrid.policy.needs_quant(8, true),
        "decode takes BF16 activations"
    );
    let mut rng = Rng::new(121);
    let (rows, n, k) = (40usize, 320usize, 1024usize);
    let w = rng.fp8_matrix(n, k);
    let x = rng.bf16_vec(rows * k, 1.0);
    let bits = |v: Vec<f32>| v.iter().map(|f| f.to_bits()).collect::<Vec<_>>();
    // Gemm::fp8 is the non-KDA projection; run_kda below the KDA one.
    assert_eq!(
        bits(run(&hybrid, &s, &x, rows, &w)),
        bits(run(&w8a16, &s, &x, rows, &w))
    );
    assert_eq!(
        bits(run_kda(&hybrid, &s, &x, rows, &w)),
        bits(run(&w8a8, &s, &x, rows, &w))
    );
    assert_eq!(
        bits(run_kda(&w8a16, &s, &x, rows, &w)),
        bits(run(&w8a16, &s, &x, rows, &w))
    );
}

/// `Gemm::fp8_kda` over `rows` rows, with the E4M3 activations when the policy needs them for a
/// KDA projection.
fn run_kda(g: &Gemm, s: &Stream, x: &[u16], rows: usize, w: &fp8::Fp8Matrix) -> Vec<f32> {
    let (n, k) = (w.rows, w.cols);
    let dx = DeviceBuffer::from_slice(x).unwrap();
    let (xq, xs) = (
        DeviceBuffer::alloc(rows * k).unwrap(),
        DeviceBuffer::alloc(rows * (k / 128) * 4).unwrap(),
    );
    if g.policy.needs_quant(rows, true) {
        unsafe { act_quant(dx.ptr(0), xq.ptr(0), xs.ptr(0), rows, k, s) }.unwrap();
    }
    let (dw, ds) = (
        DeviceBuffer::from_slice(&w.data).unwrap(),
        DeviceBuffer::from_slice(&w.scale_inv).unwrap(),
    );
    let out = DeviceBuffer::zeroed(rows * n * 2).unwrap();
    let mat = Fp8Mat {
        w: dw.ptr(0),
        scales: ds.ptr(0),
        n,
        k,
    };
    let input = Fp8Input {
        bf16: dx.ptr(0),
        q: xq.ptr(0),
        scales: xs.ptr(0),
    };
    unsafe { g.fp8_kda(&input, &mat, rows, out.ptr(0), s) }.unwrap();
    s.synchronize().unwrap();
    bf16::widen(&out.download::<u16>(rows * n).unwrap())
}

#[test]
fn tiles_cover_the_weight() {
    for (n, k) in [
        (24_640usize, 4096usize),
        (4096, 8192),
        (4096, 16_384),
        (24_576, 4096),
        (16_384, 1536),
        (320, 512),
    ] {
        let t = Gemm::w8a16_tile_rows(n, k);
        assert!(t % 128 == 0 && t * k * 2 <= DEQUANT_BYTES, "{n} x {k}: {t}");
        // As few tiles as the scratch allows (tiles of whole 128-row blocks).
        let most = DEQUANT_BYTES / (2 * k) / 128 * 128;
        let tiles = n.div_ceil(t);
        assert_eq!(tiles, n.div_ceil(most), "{n} x {k}");
        eprintln!("{n} x {k}: {tiles} tile(s) of {t} rows");
    }
}
