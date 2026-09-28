//! The BF16 GEMV and the glue kernels against their host models (feature `cuda`; skips
//! without a device).
//!
//! - the GEMV is bitwise equal to its model for 1..8 rows, with and without K splits, grouped,
//!   BF16 and f32 outputs;
//! - a row's GEMV result does not depend on the other rows of the launch;
//! - cuBLAS agrees with it within f32 rounding (cuBLAS's own row independence is reported);
//! - the expert combine, the argmax (padding rows excluded, ties to the lower index), the
//!   row gather and scatter, and the conversions.
#![cfg(feature = "cuda")]

use core::ffi::c_void;

use glm53f_forward::device::{self, launched, DeviceBuffer, Stream};
use glm53f_forward::ffi;
use glm53f_forward::gemm::{Bf16Mat, Gemm, GemmPolicy};
use glm53f_forward::reference;
use glm53f_layers::bf16;

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    fn normal(&mut self) -> f32 {
        let u1 = ((self.next() >> 11) as f64 / (1u64 << 53) as f64).max(1e-12);
        let u2 = (self.next() >> 11) as f64 / (1u64 << 53) as f64;
        ((-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()) as f32
    }
    fn bf16s(&mut self, n: usize, sigma: f32) -> Vec<u16> {
        (0..n)
            .map(|_| bf16::from_f32(self.normal() * sigma))
            .collect()
    }
}

fn gpu() -> Option<Stream> {
    if device::device_count() == 0 {
        eprintln!("skip: no CUDA device");
        return None;
    }
    Some(Stream::new().unwrap())
}

/// Run the GEMV on `rows` rows and return f32 results `[groups][rows][n]`.
#[allow(clippy::too_many_arguments)]
fn run_gemv(
    gemm: &Gemm,
    s: &Stream,
    x: &[u16],
    ldx: usize,
    xg: usize,
    w: &DeviceBuffer,
    groups: usize,
    n: usize,
    k: usize,
    rows: usize,
    f32_out: bool,
) -> Vec<f32> {
    let dx = DeviceBuffer::from_slice(x).unwrap();
    let out = DeviceBuffer::zeroed(groups * rows * n * 4).unwrap();
    let mat = Bf16Mat {
        ptr: w.ptr(0),
        n,
        k,
        ld: k,
        groups,
        gstride: n * k,
    };
    unsafe {
        gemm.gemv(
            dx.ptr(0),
            ldx,
            xg,
            &mat,
            rows,
            out.ptr::<c_void>(0),
            n,
            rows * n,
            f32_out,
            s,
        )
    }
    .unwrap();
    s.synchronize().unwrap();
    if f32_out {
        out.download::<f32>(groups * rows * n).unwrap()
    } else {
        bf16::widen(&out.download::<u16>(groups * rows * n).unwrap())
    }
}

#[test]
fn gemv_matches_its_model_and_is_row_independent() {
    let Some(s) = gpu() else { return };
    let gemm = Gemm::new(&s, GemmPolicy::default()).unwrap();
    let mut rng = Rng(11);
    // (groups, n, k, f32 output): K splits, several splits, no split, grouped small K, odd n.
    for &(groups, n, k, f32_out) in &[
        (1usize, 64usize, 4096usize, false),
        (1, 288, 4096, true),
        (1, 1024, 1536, false),
        (2, 512, 128, false),
        (1, 4096, 4096, true),
    ] {
        let ksplit = Gemm::gemv_ksplit(&Bf16Mat {
            ptr: core::ptr::null(),
            n,
            k,
            ld: k,
            groups,
            gstride: n * k,
        });
        assert_eq!(ksplit, reference::gemv_ksplit(groups, n, k));
        let w = rng.bf16s(groups * n * k, 0.02);
        let dw = DeviceBuffer::from_slice(&w).unwrap();
        // Activations [8][groups * k]: group g of row m at m * ldx + g * k.
        let ldx = groups * k;
        let x = rng.bf16s(8 * ldx, 1.0);
        let all = run_gemv(&gemm, &s, &x, ldx, k, &dw, groups, n, k, 8, f32_out);
        for g in 0..groups {
            let xg: Vec<u16> = (0..8)
                .flat_map(|m| x[m * ldx + g * k..m * ldx + (g + 1) * k].to_vec())
                .collect();
            let model =
                reference::gemv_bf16(&xg, k, &w[g * n * k..(g + 1) * n * k], k, 8, n, k, ksplit);
            let model: Vec<f32> = if f32_out {
                model
            } else {
                model.iter().map(|&v| bf16::round(v)).collect()
            };
            let got = &all[g * 8 * n..(g + 1) * 8 * n];
            let bad = got
                .iter()
                .zip(&model)
                .filter(|(a, b)| a.to_bits() != b.to_bits())
                .count();
            assert_eq!(
                bad, 0,
                "groups {groups} n {n} k {k} (ksplit {ksplit}): {bad} values differ from the model"
            );
        }
        // Row independence: every row count, and a row alone.
        for rows in 1..=8 {
            let part = run_gemv(
                &gemm,
                &s,
                &x[..rows * ldx],
                ldx,
                k,
                &dw,
                groups,
                n,
                k,
                rows,
                f32_out,
            );
            for g in 0..groups {
                for m in 0..rows {
                    let a = &part[(g * rows + m) * n..(g * rows + m + 1) * n];
                    let b = &all[(g * 8 + m) * n..(g * 8 + m + 1) * n];
                    assert!(
                        a.iter().zip(b).all(|(p, q)| p.to_bits() == q.to_bits()),
                        "row {m} of {rows} differs from the same row of 8 (groups {groups} n {n} k {k})"
                    );
                }
            }
        }
        eprintln!("gemv groups {groups} n {n} k {k} ksplit {ksplit} f32 {f32_out}: bitwise with the model, row-independent");
    }
}

#[test]
fn cublas_agrees_within_rounding() {
    let Some(s) = gpu() else { return };
    let gemm = Gemm::new(&s, GemmPolicy::default()).unwrap();
    let mut rng = Rng(12);
    let (n, k) = (2048usize, 4096usize);
    let w = rng.bf16s(n * k, 0.02);
    let x = rng.bf16s(16 * k, 1.0);
    let dw = DeviceBuffer::from_slice(&w).unwrap();
    let dx = DeviceBuffer::from_slice(&x).unwrap();
    let mat = Bf16Mat {
        ptr: dw.ptr(0),
        n,
        k,
        ld: k,
        groups: 1,
        gstride: 0,
    };
    let run = |rows: usize, cublas: bool| -> Vec<f32> {
        let out = DeviceBuffer::zeroed(rows * n * 4).unwrap();
        if cublas {
            unsafe {
                gemm.cublas(
                    dx.ptr(0),
                    k,
                    0,
                    &mat,
                    rows,
                    out.ptr::<c_void>(0),
                    n,
                    0,
                    true,
                )
            }
            .unwrap();
        } else {
            unsafe {
                gemm.gemv(
                    dx.ptr(0),
                    k,
                    0,
                    &mat,
                    rows,
                    out.ptr::<c_void>(0),
                    n,
                    0,
                    true,
                    &s,
                )
            }
            .unwrap();
        }
        s.synchronize().unwrap();
        out.download::<f32>(rows * n).unwrap()
    };
    let g8 = run(8, false);
    let c8 = run(8, true);
    let rel = reference::rel_rms(&c8, &g8);
    eprintln!("cuBLAS vs GEMV, 8 rows: relative RMS {rel:.2e}");
    assert!(rel < 1e-5, "cuBLAS and the GEMV differ by {rel}");
    // Is cuBLAS row-independent? (Reported, not required: the forward uses the GEMV for <= 8 rows.)
    let c1 = run(1, true);
    let c16 = run(16, true);
    let same1 = c1
        .iter()
        .zip(&c8[..n])
        .all(|(a, b)| a.to_bits() == b.to_bits());
    let same16 = c16[..8 * n]
        .iter()
        .zip(&c8)
        .all(|(a, b)| a.to_bits() == b.to_bits());
    eprintln!(
        "cuBLAS row 0 alone == row 0 of 8: {same1}; rows of 8 == the same rows of 16: {same16}"
    );
}

#[test]
fn glue_kernels_match_their_models() {
    let Some(s) = gpu() else { return };
    let mut rng = Rng(13);
    let st = s.raw();

    // Expert combine: 5 rows, top-8, some ids repeated and one missing (-1).
    let (rows, top_k, hidden) = (5usize, 8usize, 256usize);
    let y = rng.bf16s(rows * top_k * hidden, 1.0);
    let mut ids: Vec<i32> = (0..rows * top_k)
        .map(|_| (rng.next() % 288) as i32)
        .collect();
    ids[3] = -1;
    ids[9] = ids[10];
    let weights: Vec<f32> = (0..rows * top_k)
        .map(|_| rng.normal().abs() * 0.3)
        .collect();
    let (dy, di, dwt) = (
        DeviceBuffer::from_slice(&y).unwrap(),
        DeviceBuffer::from_slice(&ids).unwrap(),
        DeviceBuffer::from_slice(&weights).unwrap(),
    );
    let dout = DeviceBuffer::zeroed(rows * hidden * 2).unwrap();
    launched(
        unsafe {
            ffi::glm53f_fwd_moe_combine(
                dy.ptr(0),
                di.ptr(0),
                dwt.ptr(0),
                rows as i32,
                top_k as i32,
                hidden as i32,
                dout.ptr(0),
                st,
            )
        },
        "combine",
    )
    .unwrap();
    s.synchronize().unwrap();
    assert_eq!(
        dout.download::<u16>(rows * hidden).unwrap(),
        reference::moe_combine(&y, &ids, &weights, rows, top_k, hidden)
    );

    // Argmax: padding excluded, ties to the lower index, NaN never wins.
    let (r, n_all, n_valid) = (3usize, 5000usize, 4990usize);
    let mut logits: Vec<f32> = (0..r * n_all).map(|_| rng.normal()).collect();
    logits[100] = 7.0;
    logits[200] = 7.0; // tie: 100 wins
    logits[4995] = 50.0; // padding: ignored
    logits[n_all + 17] = f32::NAN;
    logits[n_all + 4000] = 9.0;
    for v in &mut logits[2 * n_all..3 * n_all] {
        *v = -3.0; // all equal: index 0
    }
    let dl = DeviceBuffer::from_slice(&logits).unwrap();
    let dids = DeviceBuffer::zeroed(r * 4).unwrap();
    launched(
        unsafe {
            ffi::glm53f_fwd_argmax(
                dl.ptr(0),
                n_all as i64,
                r as i32,
                n_valid as i32,
                dids.ptr(0),
                core::ptr::null_mut(),
                st,
            )
        },
        "argmax",
    )
    .unwrap();
    s.synchronize().unwrap();
    assert_eq!(dids.download::<i32>(r).unwrap(), vec![100, 4000, 0]);

    // Gather and scatter of 48-byte rows between strided buffers.
    let src: Vec<u8> = (0..10 * 64).map(|i| (i * 7 % 251) as u8).collect();
    let idx = vec![7i32, 2, 9];
    let (ds, dix) = (
        DeviceBuffer::from_slice(&src).unwrap(),
        DeviceBuffer::from_slice(&idx).unwrap(),
    );
    let dd = DeviceBuffer::zeroed(3 * 48).unwrap();
    launched(
        unsafe { ffi::glm53f_fwd_gather_rows(ds.ptr(0), 64, dix.ptr(0), dd.ptr(0), 48, 3, 48, st) },
        "gather",
    )
    .unwrap();
    let back = DeviceBuffer::zeroed(10 * 64).unwrap();
    launched(
        unsafe {
            ffi::glm53f_fwd_scatter_rows(dd.ptr(0), 48, dix.ptr(0), back.ptr(0), 64, 3, 48, st)
        },
        "scatter",
    )
    .unwrap();
    s.synchronize().unwrap();
    let g = dd.download::<u8>(3 * 48).unwrap();
    let b = back.download::<u8>(10 * 64).unwrap();
    for (i, &row) in idx.iter().enumerate() {
        let row = row as usize;
        assert_eq!(&g[i * 48..(i + 1) * 48], &src[row * 64..row * 64 + 48]);
        assert_eq!(&b[row * 64..row * 64 + 48], &src[row * 64..row * 64 + 48]);
    }

    // Conversions.
    let xb = rng.bf16s(3 * 40, 2.0);
    let dxb = DeviceBuffer::from_slice(&xb).unwrap();
    let df = DeviceBuffer::zeroed(3 * 32 * 4).unwrap();
    launched(
        unsafe { ffi::glm53f_fwd_bf16_to_f32(dxb.ptr(0), 40, df.ptr(0), 32, 3, 32, 0.5, st) },
        "widen",
    )
    .unwrap();
    s.synchronize().unwrap();
    let f = df.download::<f32>(3 * 32).unwrap();
    for rr in 0..3 {
        for c in 0..32 {
            assert_eq!(f[rr * 32 + c], bf16::to_f32(xb[rr * 40 + c]) * 0.5);
        }
    }
}
