//! Kernels against the CPU reference on random data (`--features cuda`; needs a GPU).
//!
//! Every kernel is compared bit for bit with the CPU function that models its arithmetic,
//! except the tensor-core prefill GEMM, which is compared with exact f64 block products
//! within a bound. Each test also checks that a row's result does not depend on the other
//! rows of its launch.
#![cfg(feature = "cuda")]

use glm53f_layers::bf16;
use glm53f_layers::cuda::{DeviceBuffer, Stream};
use glm53f_layers::fp8::{self, ActScheme, Fp8Matrix, Fp8Scales, ScaleLayout};
use glm53f_layers::layer::{self, LayerParams};
use glm53f_layers::mhc::{self, HcParams, HC_MULT, HC_PROJ, PARTIAL};
use glm53f_layers::mlp::{self, Fp8Mlp};
use glm53f_layers::norm::{self, RMS_EPS};
use glm53f_layers::ops::{self, Expand, FinishOut, GemmInput, GemmOutput};
use glm53f_layers::router::{self, EXPERTS, ROUTED_SCALE, TOP_K};
use glm53f_layers::testkit::Rng;

fn up<T: Copy>(v: &[T]) -> DeviceBuffer {
    DeviceBuffer::from_slice(v).unwrap()
}
fn zeros(bytes: usize) -> DeviceBuffer {
    DeviceBuffer::zeroed(bytes).unwrap()
}
fn down<T: Copy + Default>(b: &DeviceBuffer, n: usize) -> Vec<T> {
    b.download(n).unwrap()
}

fn assert_bits_f32(gpu: &[f32], cpu: &[f32], what: &str) {
    assert_eq!(gpu.len(), cpu.len(), "{what}: length");
    for (i, (a, b)) in gpu.iter().zip(cpu).enumerate() {
        assert_eq!(a.to_bits(), b.to_bits(), "{what}[{i}]: gpu {a} cpu {b}");
    }
}

fn hc_params(rng: &mut Rng, hidden: usize) -> HcParams {
    HcParams::new(
        hidden,
        rng.bf16_vec(HC_PROJ * HC_MULT * hidden, 0.02),
        &rng.f32_vec(HC_PROJ, 0.5),
        &[0.9, 1.2, 1.1],
    )
}

// ---- Elementwise --------------------------------------------------------------------------

#[test]
fn act_quant_matches() {
    let s = Stream::new().unwrap();
    let mut rng = Rng::new(101);
    let (rows, cols) = (5, 1024);
    let mut x = rng.bf16_vec(rows * cols, 3.0);
    for v in &mut x[128..256] {
        *v = 0;
    }
    let (dx, dq, ds) = (up(&x), zeros(rows * cols), zeros(rows * cols / 128 * 4));
    ops::act_quant(&dx, &dq, &ds, rows, cols, &s).unwrap();
    let cpu = fp8::quantize_rows(&bf16::widen(&x), rows, cols);
    assert_eq!(down::<u8>(&dq, rows * cols), cpu.q);
    assert_bits_f32(&down::<f32>(&ds, rows * cols / 128), &cpu.scale, "scales");
}

#[test]
fn swiglu_matches() {
    let s = Stream::new().unwrap();
    let mut rng = Rng::new(102);
    let (rows, inter) = (3, 384);
    let gu = rng.bf16_vec(rows * 2 * inter, 8.0); // reaches both clamps
    let (dgu, dact, dq, ds) = (
        up(&gu),
        zeros(rows * inter * 2),
        zeros(rows * inter),
        zeros(rows * inter / 128 * 4),
    );
    ops::swiglu(&dgu, Some(&dact), Some((&dq, &ds)), rows, inter, &s).unwrap();
    let act = mlp::swiglu_rows(&gu, rows, inter);
    assert_eq!(down::<u16>(&dact, rows * inter), act);
    let q = fp8::quantize_rows(&bf16::widen(&act), rows, inter);
    assert_eq!(down::<u8>(&dq, rows * inter), q.q);
    assert_bits_f32(&down::<f32>(&ds, rows * inter / 128), &q.scale, "scales");
}

#[test]
fn rmsnorm_matches() {
    let s = Stream::new().unwrap();
    let mut rng = Rng::new(103);
    for hidden in [4096, 512, 8] {
        let rows = 3;
        let x = rng.bf16_vec(rows * hidden, 2.0);
        let w = rng.bf16_vec(hidden, 0.5);
        let (dx, dw, dout) = (up(&x), up(&w), zeros(rows * hidden * 2));
        ops::rmsnorm(&dx, &dw, &dout, rows, hidden, &s).unwrap();
        assert_eq!(
            down::<u16>(&dout, rows * hidden),
            norm::rms_norm(&x, &w, rows, RMS_EPS),
            "hidden {hidden}"
        );
    }
}

// ---- FP8 GEMMs ------------------------------------------------------------------------------

struct DevWeight {
    w: DeviceBuffer,
    s: DeviceBuffer,
}
fn dev_weight(m: &Fp8Matrix) -> DevWeight {
    DevWeight {
        w: up(&m.data),
        s: up(&m.scale_inv),
    }
}

/// The decode GEMM for `rows` BF16 rows, in the given scheme and split, via the device.
fn gpu_decode(
    x: &[u16],
    rows: usize,
    w: &Fp8Matrix,
    dw: &DevWeight,
    scheme: ActScheme,
    ksplit: usize,
    s: &Stream,
) -> Vec<u16> {
    let (n, k) = (w.rows, w.cols);
    let dx = up(x);
    let (dq, ds) = (zeros(rows * k), zeros(rows * k / 128 * 4));
    let input = match scheme {
        ActScheme::Bf16 => GemmInput::Bf16(&dx),
        ActScheme::Fp8Dynamic128 => {
            ops::act_quant(&dx, &dq, &ds, rows, k, s).unwrap();
            GemmInput::Fp8 {
                q: &dq,
                scales: &ds,
            }
        }
    };
    let out = zeros(rows * n * 2);
    if ksplit == 1 {
        ops::fp8_gemm_decode(
            &input,
            &dw.w,
            &dw.s,
            rows,
            n,
            k,
            1,
            &GemmOutput::Bf16(&out),
            s,
        )
        .unwrap();
    } else {
        let p = zeros(ksplit * rows * n * 4);
        ops::fp8_gemm_decode(
            &input,
            &dw.w,
            &dw.s,
            rows,
            n,
            k,
            ksplit,
            &GemmOutput::Partials(&p),
            s,
        )
        .unwrap();
        ops::splitk_reduce(&p, &out, ksplit, rows, n, s).unwrap();
    }
    down(&out, rows * n)
}

#[test]
fn decode_gemm_is_bitwise_and_row_independent() {
    let s = Stream::new().unwrap();
    let mut rng = Rng::new(104);
    // 320 and 136 rows end in a partial 128-row block (64 and 8 rows) with its own scales, as
    // the fused KDA q|k|v|b projection (24,640 rows) does.
    for (n, k, ksplits) in [
        (256, 1024, vec![1, 2, 4, 8]),
        (512, 4096, vec![1, 4]),
        (128, 1536, vec![1, 3]),
        (320, 2048, vec![1, 2]),
        (136, 512, vec![1]),
    ] {
        let w = rng.fp8_matrix(n, k);
        let dw = dev_weight(&w);
        let x = rng.bf16_vec(8 * k, 1.0);
        for scheme in [ActScheme::Bf16, ActScheme::Fp8Dynamic128] {
            for &ksplit in &ksplits {
                let cpu = bf16::narrow(&mlp::fp8_linear(&x, 8, &w, scheme, ksplit));
                let all = gpu_decode(&x, 8, &w, &dw, scheme, ksplit, &s);
                assert_eq!(all, cpu, "n={n} k={k} {scheme:?} ksplit={ksplit}");
                for rows in 1..8 {
                    let part = gpu_decode(&x[..rows * k], rows, &w, &dw, scheme, ksplit, &s);
                    assert_eq!(
                        &part[..],
                        &all[..rows * n],
                        "rows={rows} n={n} k={k} {scheme:?} ksplit={ksplit}"
                    );
                }
            }
        }
    }
}

#[test]
fn prefill_gemm_is_within_bound_of_exact() {
    // The FP8 tensor cores accumulate with fewer bits than f32, so the prefill GEMM is not
    // bitwise against any CPU order. Bound, per output, relative to m = sum |sx sw x w|:
    // Block128 worst 2^-10 m and mean 2^-15 m; K32 worst 2^-12 m and mean 2^-16 m (measured
    // on sm_89: Block128 worst 1.6e-4, mean 4e-6 to 1.4e-5; K32 worst 6.3e-5, mean 2e-6 to
    // 7e-6). The exact-integer test below pins the indexing bit for bit.
    let s = Stream::new().unwrap();
    let mut rng = Rng::new(105);
    for (rows, n, k) in [
        (1, 128, 128),
        (100, 256, 512),
        (300, 384, 1024),
        (256, 128, 4096),
        // A partial last block of rows (64 and 8 rows past the last whole block).
        (100, 320, 512),
        (130, 136, 1024),
    ] {
        let w = rng.fp8_matrix(n, k);
        let dw = dev_weight(&w);
        let x = rng.bf16_vec(rows * k, 1.0);
        let dx = up(&x);
        let (dq, ds) = (zeros(rows * k), zeros(rows * k / 128 * 4));
        ops::act_quant(&dx, &dq, &ds, rows, k, &s).unwrap();
        let (exact, mag) = mlp::fp8_linear_f64(&x, rows, &w, ActScheme::Fp8Dynamic128);
        let r = rows.min(8);
        let dec = gpu_decode(&x[..r * k], r, &w, &dw, ActScheme::Fp8Dynamic128, 1, &s);
        for (promotion, worst_bound, mean_bound) in [
            (ops::Promotion::Block128, 2f64.powi(-10), 2f64.powi(-15)),
            (ops::Promotion::K32, 2f64.powi(-12), 2f64.powi(-16)),
        ] {
            let (out, out32) = (zeros(rows * n * 2), zeros(rows * n * 4));
            ops::fp8_gemm_prefill(
                &dq,
                &ds,
                &dw.w,
                &dw.s,
                rows,
                n,
                k,
                promotion,
                &out,
                Some(&out32),
                &s,
            )
            .unwrap();
            let got32: Vec<f32> = down(&out32, rows * n);
            let got: Vec<u16> = down(&out, rows * n);
            let mut mean = 0f64;
            for i in 0..rows * n {
                let rel = (got32[i] as f64 - exact[i]).abs() / mag[i];
                mean += rel;
                assert!(
                    rel <= worst_bound,
                    "{promotion:?} rows={rows} n={n} k={k} [{i}]: {} vs {} (rel {rel:.2e})",
                    got32[i],
                    exact[i]
                );
                assert_eq!(
                    got[i],
                    bf16::from_f32(got32[i]),
                    "BF16 output is the rounded f32 value"
                );
            }
            mean /= (rows * n) as f64;
            assert!(
                mean <= mean_bound,
                "{promotion:?} rows={rows} n={n} k={k}: mean error {mean:.2e}"
            );
            // The decode kernel (f32 order) agrees within both errors and one BF16 rounding.
            for i in 0..r * n {
                let d = (bf16::to_f32(dec[i]) as f64 - got32[i] as f64).abs();
                assert!(
                    d <= got32[i].abs() as f64 * 2f64.powi(-8) + worst_bound * mag[i],
                    "decode vs prefill [{i}]"
                );
            }
        }
    }
}

#[test]
fn prefill_gemm_is_exact_on_integer_data() {
    // Small-integer E4M3 values with power-of-two scales: every partial sum is an exact
    // small integer times a power of two, so any correct accumulation gives the exact
    // result. A fragment-layout or swizzle error cannot hide in a tolerance here.
    let s = Stream::new().unwrap();
    let mut rng = Rng::new(110);
    let codes: Vec<u8> = [-3.0f32, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0]
        .iter()
        .map(|&v| fp8::f32_to_e4m3(v))
        .collect();
    for (rows, n, k) in [
        (1usize, 128usize, 128usize),
        (77, 256, 384),
        (130, 384, 1024),
        (77, 200, 384),
    ] {
        let pick = |rng: &mut Rng| codes[(rng.next_u64() % codes.len() as u64) as usize];
        let xq: Vec<u8> = (0..rows * k).map(|_| pick(&mut rng)).collect();
        let wq: Vec<u8> = (0..n * k).map(|_| pick(&mut rng)).collect();
        // Scales 2^-2 .. 2^2 keep every sum within 24 significant bits.
        let pow2 = |rng: &mut Rng| 2f32.powi((rng.next_u64() % 5) as i32 - 2);
        let xs: Vec<f32> = (0..rows * k / 128).map(|_| pow2(&mut rng)).collect();
        let ws: Vec<f32> = (0..n.div_ceil(128) * (k / 128))
            .map(|_| pow2(&mut rng))
            .collect();
        let (dq, dxs, dwq, dws) = (up(&xq), up(&xs), up(&wq), up(&ws));
        for promotion in [ops::Promotion::Block128, ops::Promotion::K32] {
            let (out, out32) = (zeros(rows * n * 2), zeros(rows * n * 4));
            ops::fp8_gemm_prefill(
                &dq,
                &dxs,
                &dwq,
                &dws,
                rows,
                n,
                k,
                promotion,
                &out,
                Some(&out32),
                &s,
            )
            .unwrap();
            let got: Vec<f32> = down(&out32, rows * n);
            for m in 0..rows {
                for o in 0..n {
                    let mut want = 0f64;
                    for kk in 0..k {
                        let sc = xs[m * (k / 128) + kk / 128] as f64
                            * ws[(o / 128) * (k / 128) + kk / 128] as f64;
                        want += fp8::e4m3_to_f32(xq[m * k + kk]) as f64
                            * fp8::e4m3_to_f32(wq[o * k + kk]) as f64
                            * sc;
                    }
                    assert_eq!(
                        got[m * n + o] as f64,
                        want,
                        "{promotion:?} rows={rows} n={n} k={k} ({m}, {o})"
                    );
                }
            }
        }
    }
}

#[test]
fn weight_quantization_and_dequantization_match_the_host() {
    // The load-time quantizer of BF16 weights and the W8A16 prefill's BF16 tiles, bit for bit
    // against src/fp8.rs, on shapes with a partial last block of rows (the fused KDA
    // q|k|v|b projection is 24,640 = 192 x 128 + 64 rows).
    let s = Stream::new().unwrap();
    let mut rng = Rng::new(111);
    for (n, k) in [(128usize, 128usize), (320, 1024), (200, 384), (24640, 256)] {
        let mut w = rng.bf16_vec(n * k, 0.02);
        // An all-zero block, a lone outlier, and a signed zero.
        for r in 0..n.min(128) {
            for c in 0..128 {
                w[r * k + c] = 0;
            }
        }
        w[(n - 1) * k + k - 1] = bf16::from_f32(-3.0);
        w[k + 1] = 0x8000;
        let host = fp8::quantize_weight_bf16(&w, n, k);
        let (dw, q, sc) = (up(&w), zeros(n * k), zeros(n.div_ceil(128) * (k / 128) * 4));
        ops::quantize_weight(&dw, n, k, &q, &sc, &s).unwrap();
        let (gq, gs): (Vec<u8>, Vec<f32>) =
            (down(&q, n * k), down(&sc, n.div_ceil(128) * (k / 128)));
        assert_eq!(gq, host.data, "codes n={n} k={k}");
        assert_bits_f32(&gs, &host.scale_inv, "scales");
        for (row0, rows) in [(0usize, n), (n / 2, n - n / 2), (n - 1, 1)] {
            let out = zeros(rows * k * 2);
            ops::dequant_bf16(&q, &sc, n, k, row0, rows, &out, &s).unwrap();
            let got: Vec<u16> = down(&out, rows * k);
            assert_eq!(
                got,
                fp8::dequant_bf16(&host, row0, rows),
                "dequant n={n} k={k} rows {row0}+{rows}"
            );
        }
    }
}

// ---- MXFP8 weights and power-of-two block-128 scales ------------------------------------------

/// An MXFP8 weight on the device: codes and E8M0 scale bytes.
fn dev_weight_mx(m: &Fp8Matrix) -> DevWeight {
    assert_eq!(m.layout, ScaleLayout::Mx32);
    DevWeight {
        w: up(&m.data),
        s: up(&m.scale_bytes()),
    }
}

/// The MXFP8 decode GEMM (K splits reduced in the launch) for `rows` BF16 rows.
fn gpu_decode_mx(
    x: &[u16],
    rows: usize,
    w: &Fp8Matrix,
    dw: &DevWeight,
    scheme: ActScheme,
    ksplit: usize,
    s: &Stream,
) -> Vec<u16> {
    let (n, k) = (w.rows, w.cols);
    let dx = up(x);
    let (dq, ds) = (zeros(rows * k), zeros(rows * k / 128 * 4));
    let input = match scheme {
        ActScheme::Bf16 => GemmInput::Bf16(&dx),
        ActScheme::Fp8Dynamic128 => {
            ops::act_quant(&dx, &dq, &ds, rows, k, s).unwrap();
            GemmInput::Fp8 {
                q: &dq,
                scales: &ds,
            }
        }
    };
    let out = zeros(rows * n * 2);
    let (p, sync) = (zeros(ksplit * rows * n * 4), ops::sync_buffer(n / 8).unwrap());
    ops::fp8_gemm_decode_mx(
        &input,
        &dw.w,
        &dw.s,
        rows,
        n,
        k,
        ksplit,
        Some(&p),
        Some(&sync),
        &out,
        s,
    )
    .unwrap();
    down(&out, rows * n)
}

#[test]
fn mx_decode_gemm_is_bitwise_and_row_independent() {
    // The MXFP8 decode GEMM against the CPU model of the decode order (each 16-value step scaled
    // by its row's 32-block scale), bit for bit, both activation schemes, every split; each row
    // the same bits whatever the launch's other rows. 320 and 136 rows: MX scales are per row,
    // so a partial last 128-row block needs nothing of its own.
    let s = Stream::new().unwrap();
    let mut rng = Rng::new(112);
    for (n, k, ksplits) in [
        (256, 1024, vec![1, 2, 4, 8]),
        (512, 4096, vec![1, 4]),
        (320, 2048, vec![1, 2]),
        (136, 512, vec![1]),
    ] {
        let w = rng.fp8_matrix_mx(n, k);
        let dw = dev_weight_mx(&w);
        let x = rng.bf16_vec(8 * k, 1.0);
        for scheme in [ActScheme::Bf16, ActScheme::Fp8Dynamic128] {
            for &ksplit in &ksplits {
                let cpu = bf16::narrow(&mlp::fp8_linear(&x, 8, &w, scheme, ksplit));
                let all = gpu_decode_mx(&x, 8, &w, &dw, scheme, ksplit, &s);
                assert_eq!(all, cpu, "n={n} k={k} {scheme:?} ksplit={ksplit}");
                for rows in 1..8 {
                    let part = gpu_decode_mx(&x[..rows * k], rows, &w, &dw, scheme, ksplit, &s);
                    assert_eq!(
                        &part[..],
                        &all[..rows * n],
                        "rows={rows} n={n} k={k} {scheme:?} ksplit={ksplit}"
                    );
                }
            }
        }
    }
}

#[test]
fn mx_prefill_gemm_is_within_bound_of_exact() {
    // The MXFP8 prefill GEMM (the k32 structure, each k32 sum scaled by its column's scale)
    // against exact f64 32-block products, within the K32 bounds of the block-128 kernel:
    // worst 2^-12 and mean 2^-16 of m = sum |sx sw x w|.
    let s = Stream::new().unwrap();
    let mut rng = Rng::new(113);
    assert_eq!(
        unsafe { glm53f_layers::ffi::glm53f_fp8_gemm_prefill_mx_smem_bytes() },
        101_376
    );
    for (rows, n, k) in [
        (1, 128, 128),
        (100, 256, 512),
        (300, 384, 1024),
        (256, 128, 4096),
        (100, 320, 512),
        (130, 136, 1024),
    ] {
        let w = rng.fp8_matrix_mx(n, k);
        let dw = dev_weight_mx(&w);
        let x = rng.bf16_vec(rows * k, 1.0);
        let dx = up(&x);
        let (dq, ds) = (zeros(rows * k), zeros(rows * k / 128 * 4));
        ops::act_quant(&dx, &dq, &ds, rows, k, &s).unwrap();
        let (exact, mag) = mlp::fp8_linear_f64(&x, rows, &w, ActScheme::Fp8Dynamic128);
        let (out, out32) = (zeros(rows * n * 2), zeros(rows * n * 4));
        ops::fp8_gemm_prefill_mx(&dq, &ds, &dw.w, &dw.s, rows, n, k, &out, Some(&out32), &s)
            .unwrap();
        let got32: Vec<f32> = down(&out32, rows * n);
        let got: Vec<u16> = down(&out, rows * n);
        let mut mean = 0f64;
        for i in 0..rows * n {
            let rel = (got32[i] as f64 - exact[i]).abs() / mag[i];
            mean += rel;
            assert!(
                rel <= 2f64.powi(-12),
                "rows={rows} n={n} k={k} [{i}]: {} vs {} (rel {rel:.2e})",
                got32[i],
                exact[i]
            );
            assert_eq!(got[i], bf16::from_f32(got32[i]));
        }
        mean /= (rows * n) as f64;
        assert!(mean <= 2f64.powi(-16), "rows={rows} n={n} k={k}: mean error {mean:.2e}");
        // The decode GEMM (f32 order) agrees within both errors and one BF16 rounding.
        let r = rows.min(8);
        let dec = gpu_decode_mx(&x[..r * k], r, &w, &dw, ActScheme::Fp8Dynamic128, 1, &s);
        for i in 0..r * n {
            let d = (bf16::to_f32(dec[i]) as f64 - got32[i] as f64).abs();
            assert!(
                d <= got32[i].abs() as f64 * 2f64.powi(-8) + 2f64.powi(-12) * mag[i],
                "decode vs prefill [{i}]"
            );
        }
    }
}

#[test]
fn mx_prefill_gemm_is_exact_on_integer_data() {
    // Small-integer E4M3 values with power-of-two scales per 128 (activations) and per 32
    // (weights, E8M0): every partial sum is exact, so any correct accumulation gives the exact
    // result and a wrong scale index, fragment or swizzle cannot hide in a tolerance.
    let s = Stream::new().unwrap();
    let mut rng = Rng::new(114);
    let codes: Vec<u8> = [-3.0f32, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0]
        .iter()
        .map(|&v| fp8::f32_to_e4m3(v))
        .collect();
    for (rows, n, k) in [
        (1usize, 128usize, 128usize),
        (77, 256, 384),
        (130, 384, 1024),
        (77, 200, 384),
    ] {
        let pick = |rng: &mut Rng| codes[(rng.next_u64() % codes.len() as u64) as usize];
        let xq: Vec<u8> = (0..rows * k).map(|_| pick(&mut rng)).collect();
        let wq: Vec<u8> = (0..n * k).map(|_| pick(&mut rng)).collect();
        let pow2 = |rng: &mut Rng| 2f32.powi((rng.next_u64() % 5) as i32 - 2);
        let xs: Vec<f32> = (0..rows * k / 128).map(|_| pow2(&mut rng)).collect();
        let ws: Vec<f32> = (0..n * k / 32).map(|_| pow2(&mut rng)).collect();
        let w = Fp8Matrix::with_layout(n, k, wq.clone(), ws.clone(), ScaleLayout::Mx32);
        let dw = dev_weight_mx(&w);
        let (dq, dxs) = (up(&xq), up(&xs));
        let (out, out32) = (zeros(rows * n * 2), zeros(rows * n * 4));
        ops::fp8_gemm_prefill_mx(&dq, &dxs, &dw.w, &dw.s, rows, n, k, &out, Some(&out32), &s)
            .unwrap();
        let got: Vec<f32> = down(&out32, rows * n);
        for m in 0..rows {
            for o in 0..n {
                let mut want = 0f64;
                for kk in 0..k {
                    let sc =
                        xs[m * (k / 128) + kk / 128] as f64 * ws[o * (k / 32) + kk / 32] as f64;
                    want += fp8::e4m3_to_f32(xq[m * k + kk]) as f64
                        * fp8::e4m3_to_f32(wq[o * k + kk]) as f64
                        * sc;
                }
                assert_eq!(got[m * n + o] as f64, want, "rows={rows} n={n} k={k} ({m}, {o})");
            }
        }
    }
}

#[test]
fn pow2_and_mx_weight_quantization_match_the_host() {
    // The load-time quantizers with power-of-two scales (block-128 and MXFP8) and the MXFP8
    // BF16 tiles, bit for bit against src/fp8.rs, on shapes with a partial last block of rows.
    let s = Stream::new().unwrap();
    let mut rng = Rng::new(115);
    for (n, k) in [(128usize, 128usize), (320, 1024), (200, 384), (24640, 256)] {
        let mut w = rng.bf16_vec(n * k, 0.02);
        for r in 0..n.min(128) {
            for c in 0..128 {
                w[r * k + c] = 0;
            }
        }
        w[(n - 1) * k + k - 1] = bf16::from_f32(-3.0);
        w[k + 1] = 0x8000;
        if n > 128 {
            // A block maximum of exactly 448: scale 1, code 0x7E, no saturation.
            w[(n - 2) * k] = bf16::from_f32(448.0);
        }
        let dw = up(&w);
        for scales in [Fp8Scales::Block128Pow2, Fp8Scales::Mx32] {
            let host = fp8::quantize_weight_bf16_as(&w, n, k, scales);
            let sb = host.scale_bytes();
            let (q, sc) = (zeros(n * k), zeros(sb.len()));
            ops::quantize_weight_as(&dw, n, k, scales, &q, &sc, &s).unwrap();
            assert_eq!(down::<u8>(&q, n * k), host.data, "codes {scales:?} n={n} k={k}");
            assert_eq!(down::<u8>(&sc, sb.len()), sb, "scales {scales:?} n={n} k={k}");
            for (row0, rows) in [(0usize, n), (n / 2, n - n / 2), (n - 1, 1)] {
                let out = zeros(rows * k * 2);
                match scales {
                    Fp8Scales::Mx32 => ops::dequant_bf16_mx(&q, &sc, n, k, row0, rows, &out, &s),
                    _ => ops::dequant_bf16(&q, &sc, n, k, row0, rows, &out, &s),
                }
                .unwrap();
                assert_eq!(
                    down::<u16>(&out, rows * k),
                    fp8::dequant_bf16(&host, row0, rows),
                    "dequant {scales:?} n={n} k={k} rows {row0}+{rows}"
                );
            }
        }
        // The checkpoint's scheme through the same entry: glm53f_fp8_quantize_weight's bits.
        let host = fp8::quantize_weight_bf16(&w, n, k);
        let (q, sc) = (zeros(n * k), zeros(host.scale_bytes().len()));
        ops::quantize_weight_as(&dw, n, k, Fp8Scales::Block128, &q, &sc, &s).unwrap();
        assert_eq!(down::<u8>(&q, n * k), host.data);
        assert_bits_f32(&down::<f32>(&sc, host.scale_inv.len()), &host.scale_inv, "scales");
    }
}

// ---- Router ---------------------------------------------------------------------------------

#[test]
fn router_matches_and_is_row_independent() {
    let s = Stream::new().unwrap();
    let mut rng = Rng::new(106);
    let hidden = 4096;
    let weight = rng.bf16_vec(EXPERTS * hidden, 0.02);
    let bias = rng.f32_vec(EXPERTS, 0.01);
    let (dw, db) = (up(&weight), up(&bias));
    let x = rng.bf16_vec(13 * hidden, 1.0);
    let mut first: Option<(Vec<f32>, Vec<i32>, Vec<f32>)> = None;
    for rows in [13usize, 8, 3, 1] {
        let dx = up(&x[..rows * hidden]);
        let (dl, di, dwt) = (
            zeros(rows * EXPERTS * 4),
            zeros(rows * TOP_K * 4),
            zeros(rows * TOP_K * 4),
        );
        ops::router_logits(&dx, &dw, &dl, rows, EXPERTS, hidden, &s).unwrap();
        ops::router_select(&dl, &db, &di, &dwt, rows, EXPERTS, TOP_K, ROUTED_SCALE, &s).unwrap();
        let logits: Vec<f32> = down(&dl, rows * EXPERTS);
        let ids: Vec<i32> = down(&di, rows * TOP_K);
        let wts: Vec<f32> = down(&dwt, rows * TOP_K);
        for t in 0..rows {
            let xt = bf16::widen(&x[t * hidden..(t + 1) * hidden]);
            let cl = router::logits_row(&xt, &weight, EXPERTS);
            assert_bits_f32(&logits[t * EXPERTS..(t + 1) * EXPERTS], &cl, "logits");
            let r = router::select_row(&cl, &bias, TOP_K, ROUTED_SCALE);
            let gi: Vec<u32> = ids[t * TOP_K..(t + 1) * TOP_K]
                .iter()
                .map(|&v| v as u32)
                .collect();
            assert_eq!(gi, r.ids, "row {t} ids");
            assert_bits_f32(&wts[t * TOP_K..(t + 1) * TOP_K], &r.weights, "weights");
        }
        match &first {
            None => first = Some((logits, ids, wts)),
            Some((l, i, w)) => {
                assert_bits_f32(&logits, &l[..rows * EXPERTS], "row-independent logits");
                assert_eq!(&ids[..], &i[..rows * TOP_K]);
                assert_bits_f32(&wts, &w[..rows * TOP_K], "row-independent weights");
            }
        }
    }
    // Exact ties go to the lower index on the device too.
    let flat = vec![0.5f32; 2 * EXPERTS];
    let (dl, di, dwt) = (up(&flat), zeros(2 * TOP_K * 4), zeros(2 * TOP_K * 4));
    let zb = up(&vec![0f32; EXPERTS]);
    ops::router_select(&dl, &zb, &di, &dwt, 2, EXPERTS, TOP_K, ROUTED_SCALE, &s).unwrap();
    let ids: Vec<i32> = down(&di, 2 * TOP_K);
    assert_eq!(&ids[..8], &[0, 1, 2, 3, 4, 5, 6, 7]);
}

// ---- mHC ------------------------------------------------------------------------------------

struct DevHc {
    fn_: DeviceBuffer,
    base: DeviceBuffer,
    scale: DeviceBuffer,
}
fn dev_hc(p: &HcParams) -> DevHc {
    DevHc {
        fn_: up(&p.fn_),
        base: up(&p.base),
        scale: up(&p.scale),
    }
}

#[test]
fn hc_boundary_matches() {
    let s = Stream::new().unwrap();
    let mut rng = Rng::new(107);
    for hidden in [4096usize, 512] {
        let rows = 5;
        let slices = hidden / 128;
        let p = hc_params(&mut rng, hidden);
        let dp = dev_hc(&p);
        let streams = rng.bf16_vec(rows * 4 * hidden, 1.5);
        let nw = rng.bf16_vec(hidden, 0.3);
        let (dst, dnw) = (up(&streams), up(&nw));
        let parts = zeros(rows * slices * PARTIAL * 4);
        ops::hc_project(&dst, None, Some((&dp.fn_, &parts)), rows, hidden, &s).unwrap();
        let (pre, post, comb) = (zeros(rows * 16), zeros(rows * 16), zeros(rows * 64));
        let (col, nrm, q, qs) = (
            zeros(rows * hidden * 2),
            zeros(rows * hidden * 2),
            zeros(rows * hidden),
            zeros(rows * slices * 4),
        );
        let out = FinishOut {
            pre: Some(&pre),
            post: Some(&post),
            comb: Some(&comb),
            collapsed: Some(&col),
            normed: Some(&nrm),
            quant: Some((&q, &qs)),
        };
        ops::hc_finish(
            &parts,
            &dp.base,
            &dp.scale,
            &dst,
            Some(&dnw),
            &out,
            rows,
            hidden,
            &s,
        )
        .unwrap();
        let g_parts: Vec<f32> = down(&parts, rows * slices * PARTIAL);
        let g_pre: Vec<f32> = down(&pre, rows * 4);
        let g_post: Vec<f32> = down(&post, rows * 4);
        let g_comb: Vec<f32> = down(&comb, rows * 16);
        let g_col: Vec<u16> = down(&col, rows * hidden);
        let g_nrm: Vec<u16> = down(&nrm, rows * hidden);
        let g_q: Vec<u8> = down(&q, rows * hidden);
        let g_qs: Vec<f32> = down(&qs, rows * slices);
        for t in 0..rows {
            let st = bf16::widen(&streams[t * 4 * hidden..(t + 1) * 4 * hidden]);
            let cp = mhc::project(&st, &p);
            let flat: Vec<f32> = cp.iter().flatten().copied().collect();
            assert_bits_f32(
                &g_parts[t * slices * PARTIAL..(t + 1) * slices * PARTIAL],
                &flat,
                "partials",
            );
            let m = mhc::finish(&cp, &p, RMS_EPS);
            assert_bits_f32(&g_pre[t * 4..t * 4 + 4], &m.pre, "pre");
            assert_bits_f32(&g_post[t * 4..t * 4 + 4], &m.post, "post");
            assert_bits_f32(&g_comb[t * 16..t * 16 + 16], &m.comb, "comb");
            let c = mhc::collapse(&st, &m.pre, hidden);
            assert_eq!(&g_col[t * hidden..(t + 1) * hidden], &c[..], "collapsed");
            let n = norm::rms_norm_row(&c, &nw, RMS_EPS);
            assert_eq!(&g_nrm[t * hidden..(t + 1) * hidden], &n[..], "normed");
            let qq = fp8::quantize_rows(&bf16::widen(&n), 1, hidden);
            assert_eq!(&g_q[t * hidden..(t + 1) * hidden], &qq.q[..], "normed q");
            assert_bits_f32(
                &g_qs[t * slices..(t + 1) * slices],
                &qq.scale,
                "normed scales",
            );
        }
        // One row alone gives the same bits as that row in the batch.
        let one = up(&streams[3 * 4 * hidden..4 * 4 * hidden]);
        let p1 = zeros(slices * PARTIAL * 4);
        ops::hc_project(&one, None, Some((&dp.fn_, &p1)), 1, hidden, &s).unwrap();
        let n1 = zeros(hidden * 2);
        let o1 = FinishOut {
            normed: Some(&n1),
            ..Default::default()
        };
        ops::hc_finish(
            &p1,
            &dp.base,
            &dp.scale,
            &one,
            Some(&dnw),
            &o1,
            1,
            hidden,
            &s,
        )
        .unwrap();
        assert_eq!(
            down::<u16>(&n1, hidden),
            g_nrm[3 * hidden..4 * hidden].to_vec()
        );
    }
}

#[test]
fn hc_fused_expand_and_head_match() {
    let s = Stream::new().unwrap();
    let mut rng = Rng::new(108);
    let (rows, hidden) = (7, 4096);
    let slices = hidden / 128;
    let p = hc_params(&mut rng, hidden);
    let dp = dev_hc(&p);
    let res = rng.bf16_vec(rows * 4 * hidden, 1.0);
    let h1 = rng.bf16_vec(rows * hidden, 0.5);
    let h2 = rng.bf16_vec(rows * hidden, 0.5);
    // Plausible previous-boundary weights.
    let mut post = Vec::new();
    let mut comb = Vec::new();
    for _ in 0..rows {
        let m = mhc::mix(&bf16::widen(&rng.bf16_vec(4 * hidden, 1.0)), &p, RMS_EPS);
        post.extend(m.post);
        comb.extend(m.comb);
    }
    let (dres, dh1, dh2, dpost, dcomb) = (up(&res), up(&h1), up(&h2), up(&post), up(&comb));
    for two in [false, true] {
        let so = zeros(rows * 4 * hidden * 2);
        let parts = zeros(rows * slices * PARTIAL * 4);
        let e = Expand {
            block_out: &dh1,
            block_out2: two.then_some(&dh2),
            post: &dpost,
            comb: &dcomb,
            streams_out: Some(&so),
        };
        ops::hc_project(&dres, Some(&e), Some((&dp.fn_, &parts)), rows, hidden, &s).unwrap();
        let g_so: Vec<u16> = down(&so, rows * 4 * hidden);
        let g_parts: Vec<f32> = down(&parts, rows * slices * PARTIAL);
        for t in 0..rows {
            let h = mhc::block_output(
                &h1[t * hidden..(t + 1) * hidden],
                two.then(|| &h2[t * hidden..(t + 1) * hidden]),
            );
            let pt: [f32; 4] = post[t * 4..t * 4 + 4].try_into().unwrap();
            let ct: [f32; 16] = comb[t * 16..t * 16 + 16].try_into().unwrap();
            let ns = mhc::expand(
                &h,
                &bf16::widen(&res[t * 4 * hidden..(t + 1) * 4 * hidden]),
                &pt,
                &ct,
                hidden,
            );
            assert_eq!(
                &g_so[t * 4 * hidden..(t + 1) * 4 * hidden],
                &ns[..],
                "expanded streams, two={two}"
            );
            let cp: Vec<f32> = mhc::project(&bf16::widen(&ns), &p)
                .iter()
                .flatten()
                .copied()
                .collect();
            assert_bits_f32(
                &g_parts[t * slices * PARTIAL..(t + 1) * slices * PARTIAL],
                &cp,
                "partials after expand",
            );
        }
        // Expansion alone gives the same streams.
        let so2 = zeros(rows * 4 * hidden * 2);
        let e2 = Expand {
            streams_out: Some(&so2),
            ..e
        };
        ops::hc_project(&dres, Some(&e2), None, rows, hidden, &s).unwrap();
        assert_eq!(down::<u16>(&so2, rows * 4 * hidden), g_so);
        // The head with the fused expansion equals the head of the expanded streams.
        let nw = rng.bf16_vec(hidden, 0.4);
        let dnw = up(&nw);
        let (o_fused, o_plain) = (zeros(rows * hidden * 2), zeros(rows * hidden * 2));
        let e3 = Expand {
            streams_out: None,
            ..e
        };
        ops::hc_head(&dres, Some(&e3), &dnw, &o_fused, rows, hidden, &s).unwrap();
        ops::hc_head(&so, None, &dnw, &o_plain, rows, hidden, &s).unwrap();
        let cpu = layer::final_hidden(&g_so, rows, &nw, RMS_EPS);
        assert_eq!(down::<u16>(&o_plain, rows * hidden), cpu, "head");
        assert_eq!(
            down::<u16>(&o_fused, rows * hidden),
            cpu,
            "head with fused expand"
        );
    }
    // Broadcast.
    let emb = rng.bf16_vec(3 * hidden, 1.0);
    let (de, ds) = (up(&emb), zeros(3 * 4 * hidden * 2));
    ops::hc_broadcast(&de, &ds, 3, hidden, &s).unwrap();
    assert_eq!(
        down::<u16>(&ds, 3 * 4 * hidden),
        layer::embed_streams(&emb, 3)
    );
}

// ---- A dense decoder layer, kernel chain against the CPU flow ------------------------------

#[test]
fn dense_layer_chain_matches_cpu_flow() {
    // embedding -> broadcast -> attention boundary -> (attention stand-in) -> expand + FFN
    // boundary with norm and W8A8 quantization -> dense MLP (gate/up GEMM, SwiGLU into the
    // down projection's input, down GEMM) -> expand + head. Bit for bit.
    let s = Stream::new().unwrap();
    let mut rng = Rng::new(109);
    let (rows, hidden, inter) = (4, 512, 1024);
    let slices = hidden / 128;
    let p = LayerParams {
        attn_hc: hc_params(&mut rng, hidden),
        ffn_hc: hc_params(&mut rng, hidden),
        input_norm: rng.bf16_vec(hidden, 0.5),
        post_attn_norm: rng.bf16_vec(hidden, 0.5),
    };
    let mlp_w = Fp8Mlp::new(
        &rng.fp8_matrix(inter, hidden),
        &rng.fp8_matrix(inter, hidden),
        rng.fp8_matrix(hidden, inter),
    );
    let final_norm = rng.bf16_vec(hidden, 0.5);
    let emb = rng.bf16_vec(rows * hidden, 1.0);
    let attn_map = |x: &[u16]| -> Vec<u16> {
        x.iter()
            .map(|&v| bf16::from_f32(bf16::to_f32(v) * 0.75 - 0.0625))
            .collect()
    };

    // CPU flow.
    let streams = layer::embed_streams(&emb, rows);
    let mut attn_cpu = |x: &[u16]| attn_map(x);
    let mut ffn_cpu = |x: &[u16]| mlp::mlp(x, rows, &mlp_w, ActScheme::Fp8Dynamic128).out;
    let tr = layer::decoder_layer(&streams, rows, &p, &mut attn_cpu, &mut ffn_cpu, RMS_EPS);
    let want = layer::final_hidden(&tr.out, rows, &final_norm, RMS_EPS);

    // Kernel chain.
    let (ah, fh) = (dev_hc(&p.attn_hc), dev_hc(&p.ffn_hc));
    let (in_norm, pa_norm, fin) = (up(&p.input_norm), up(&p.post_attn_norm), up(&final_norm));
    let (gu, dn) = (dev_weight(&mlp_w.gate_up), dev_weight(&mlp_w.down));
    let st0 = zeros(rows * 4 * hidden * 2);
    ops::hc_broadcast(&up(&emb), &st0, rows, hidden, &s).unwrap();
    let parts = zeros(rows * slices * PARTIAL * 4);
    let (post_a, comb_a, post_f, comb_f) = (
        zeros(rows * 16),
        zeros(rows * 64),
        zeros(rows * 16),
        zeros(rows * 64),
    );
    let normed = zeros(rows * hidden * 2);
    ops::hc_project(&st0, None, Some((&ah.fn_, &parts)), rows, hidden, &s).unwrap();
    let o = FinishOut {
        post: Some(&post_a),
        comb: Some(&comb_a),
        normed: Some(&normed),
        ..Default::default()
    };
    ops::hc_finish(
        &parts,
        &ah.base,
        &ah.scale,
        &st0,
        Some(&in_norm),
        &o,
        rows,
        hidden,
        &s,
    )
    .unwrap();
    // The attention stand-in runs on the host.
    let a_in: Vec<u16> = down(&normed, rows * hidden);
    let a_out = up(&attn_map(&a_in));
    let st1 = zeros(rows * 4 * hidden * 2);
    let e = Expand {
        block_out: &a_out,
        block_out2: None,
        post: &post_a,
        comb: &comb_a,
        streams_out: Some(&st1),
    };
    ops::hc_project(&st0, Some(&e), Some((&fh.fn_, &parts)), rows, hidden, &s).unwrap();
    let (xq, xs) = (zeros(rows * hidden), zeros(rows * slices * 4));
    let o = FinishOut {
        post: Some(&post_f),
        comb: Some(&comb_f),
        quant: Some((&xq, &xs)),
        ..Default::default()
    };
    ops::hc_finish(
        &parts,
        &fh.base,
        &fh.scale,
        &st1,
        Some(&pa_norm),
        &o,
        rows,
        hidden,
        &s,
    )
    .unwrap();
    let gate_up = zeros(rows * 2 * inter * 2);
    let partials = zeros(8 * rows * 2 * inter * 4);
    ops::fp8_linear_decode(
        &GemmInput::Fp8 {
            q: &xq,
            scales: &xs,
        },
        &gu.w,
        &gu.s,
        rows,
        2 * inter,
        hidden,
        &partials,
        &gate_up,
        &s,
    )
    .unwrap();
    let (aq, as_) = (zeros(rows * inter), zeros(rows * inter / 128 * 4));
    ops::swiglu(&gate_up, None, Some((&aq, &as_)), rows, inter, &s).unwrap();
    let mlp_out = zeros(rows * hidden * 2);
    ops::fp8_linear_decode(
        &GemmInput::Fp8 {
            q: &aq,
            scales: &as_,
        },
        &dn.w,
        &dn.s,
        rows,
        hidden,
        inter,
        &partials,
        &mlp_out,
        &s,
    )
    .unwrap();
    let out = zeros(rows * hidden * 2);
    let e = Expand {
        block_out: &mlp_out,
        block_out2: None,
        post: &post_f,
        comb: &comb_f,
        streams_out: None,
    };
    ops::hc_head(&st1, Some(&e), &fin, &out, rows, hidden, &s).unwrap();

    assert_eq!(
        down::<u16>(&st1, rows * 4 * hidden),
        tr.mid,
        "streams after attention"
    );
    assert_eq!(
        down::<u16>(&mlp_out, rows * hidden),
        tr.ffn_out,
        "MLP output"
    );
    assert_eq!(down::<u16>(&out, rows * hidden), want, "final hidden");
}
