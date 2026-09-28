//! The CUDA kernel against the CPU references, on a synthetic layer (random
//! trellis bits, the real checkpoint's scale magnitudes, FP8 wire rows).
//!
//! One test function runs every case in turn, so the run holds one layer image
//! on the device (0.91 GB) and one scratch (about 0.7 GB at 4,096 rows): the
//! GPU is shared. Run with `--features cuda` (and `--release`).
//!
//! Cases:
//! - stage by stage at 1, 3, 8 and 200 rows (the default schedules, partial
//!   sums kept): products against f64, the epilogue and the reduce bit for bit;
//! - decode and verify windows, 1 to 8 rows: every row against the
//!   kernel-order reference (tight) and the dequantized FP32 model reference;
//! - batch invariance: a row's output is bit-identical alone and in windows of
//!   8 and 64 rows (the default configuration up to 64 rows), and alone and in
//!   64 rows under a fixed prefill configuration of either kernel family;
//! - schedules: both kernel families, their tilings, both plans, fused and
//!   unfused steps and the L2 discard give the unfused split kernels' output
//!   bit for bit at the same K splits (8 to 4,096 rows); so do the split
//!   kernels' trellis prefetch depth, L2 evict-first policy and block order,
//!   every combination at 1, 2, 4, 8, 16, 32 and 64 rows, and at other splits;
//! - prefill, 512 and 4,096 rows: sampled rows against the kernel-order
//!   reference;
//! - the FP32 output of the prefill reduce-scatter (`ffn_f32`), rounded to
//!   BF16, equals the default output bit for bit, 1 to 4,096 rows;
//! - faults: an E4M3 NaN, a UE8M0 NaN, a bad expert id, a duplicate expert and
//!   a negative weight are refused, and the kernel still serves afterwards.
#![cfg(feature = "cuda")]

use std::collections::BTreeMap;

use glm53f_rank::consts::{HIDDEN, RANK_WIDTH, TOPK};
use glm53f_rank::exl3_cuda::{self, Cfg, CudaKernel};
use glm53f_rank::fp8::{decode_row, SCALES_PER_ROW};
use glm53f_rank::kernel::{ExpertKernel, Rows};
use glm53f_rank::layout::expert_block;
use glm53f_rank::half::{f16_to_f32, f32_to_f16};
use glm53f_rank::reference::{
    down_epilogue, expert_partial_dense, expert_partial_kernel_order, gateup_epilogue, matvec, rank_row, rotate_in,
    DenseSlice, KernelSlice,
};
use glm53f_rank::testkit;
use glm53f_wire::bf16::bf16_to_f32;

#[derive(Clone, Copy, PartialEq)]
enum Mode {
    /// The kernel's order and roundings, BF16 SwiGLU (the kernel's default).
    Kernel,
    /// The kernel's order, FP32 SwiGLU (with `fp32_swiglu` on the device).
    KernelFp32,
    /// Dequantized weights, plain products, BF16 SwiGLU: the model.
    Dense,
}

/// Reference rows for `sample` (row indices) of a call, each expert decoded once.
fn reference(
    image: &[u8],
    payload: &[u8],
    scales: &[u8],
    ids: &[i32],
    weights: &[f32],
    sample: &[usize],
    mode: Mode,
) -> BTreeMap<usize, Vec<u16>> {
    // Pairs (row, slot) by expert.
    let mut by_expert: BTreeMap<usize, Vec<(usize, usize)>> = BTreeMap::new();
    for &r in sample {
        for s in 0..TOPK {
            by_expert.entry(ids[r * TOPK + s] as usize).or_default().push((r, s));
        }
    }
    let xs: BTreeMap<usize, Vec<f32>> = sample
        .iter()
        .map(|&r| {
            let mut x = vec![0f32; HIDDEN];
            decode_row(&payload[r * HIDDEN..(r + 1) * HIDDEN], &scales[r * SCALES_PER_ROW..(r + 1) * SCALES_PER_ROW], &mut x)
                .unwrap();
            (r, x)
        })
        .collect();
    let experts: Vec<(usize, Vec<(usize, usize)>)> = by_expert.into_iter().collect();
    let threads = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4).min(16);
    let mut ys: BTreeMap<(usize, usize), Vec<f32>> = BTreeMap::new();
    std::thread::scope(|sc| {
        let handles: Vec<_> = (0..threads)
            .map(|t| {
                let (experts, xs) = (&experts, &xs);
                sc.spawn(move || {
                    let mut out = Vec::new();
                    for (e, pairs) in experts.iter().skip(t).step_by(threads) {
                        let block = expert_block(image, *e);
                        if mode == Mode::Dense {
                            let d = DenseSlice::from_block(block);
                            for &(r, s) in pairs {
                                out.push(((r, s), expert_partial_dense(&xs[&r], &d, true)));
                            }
                        } else {
                            let k = KernelSlice::from_block(block);
                            for &(r, s) in pairs {
                                out.push(((r, s), expert_partial_kernel_order(&xs[&r], &k, mode == Mode::Kernel)));
                            }
                        }
                    }
                    out
                })
            })
            .collect();
        for h in handles {
            ys.extend(h.join().unwrap());
        }
    });
    sample
        .iter()
        .map(|&r| {
            let y: Vec<Vec<f32>> = (0..TOPK).map(|s| ys[&(r, s)].clone()).collect();
            let mut o = vec![0u16; HIDDEN];
            rank_row(&y, &weights[r * TOPK..(r + 1) * TOPK], &mut o).unwrap();
            (r, o)
        })
        .collect()
}

/// (max |diff| / rms(ref), rms(diff) / rms(ref), fraction of bit-equal values).
fn compare(got: &[u16], want: &[u16]) -> (f64, f64, f64) {
    let (mut sq, mut dsq, mut dmax, mut eq) = (0f64, 0f64, 0f64, 0usize);
    for (&g, &w) in got.iter().zip(want) {
        let (g, w) = (bf16_to_f32(g) as f64, bf16_to_f32(w) as f64);
        sq += w * w;
        dsq += (g - w) * (g - w);
        dmax = dmax.max((g - w).abs());
        eq += usize::from(g == w);
    }
    let rms = (sq / want.len() as f64).sqrt();
    (dmax / rms, (dsq / want.len() as f64).sqrt() / rms, eq as f64 / want.len() as f64)
}

fn run(k: &mut CudaKernel, l: &exl3_cuda::CudaLayer, p: &[u8], s: &[u8], ids: &[i32], w: &[f32], rows: usize) -> Vec<u16> {
    let mut out = vec![0u16; rows * HIDDEN];
    k.ffn(l, Rows::separate(p, s, rows).unwrap(), ids, w, &mut out).unwrap();
    out
}

/// Stage by stage, each stage fed the GPU's own input: gate/up products
/// against f64 (only the tensor cores' accumulation differs), the epilogue
/// and the final reduce bit for bit (the same FP32 operations in the same
/// order; the epilogue's `expf` may differ from the host's by an ulp), the
/// down products against f64.
fn stagewise(k: &mut CudaKernel, layer: &exl3_cuda::CudaLayer, image: &[u8], rows: usize, fp32: bool) {
    let (p, s) = testkit::wire_rows(0x5EED_7000 + rows as u64, rows);
    let (ids, w) = testkit::routes(0x5EED_7100 + rows as u64, rows, 0);
    // The default schedule for the row count, with its partial sums kept for the hook.
    k.cfg = Some(Cfg { fp32_swiglu: fp32 as i32, discard: 1, ..Cfg::default() });
    let got = run(k, layer, &p, &s, &ids, &w, rows);
    let im = k.intermediates(rows).unwrap();
    k.cfg = None;
    let (sk, skd, pn) = (im.cfg.sk as usize, im.cfg.skd as usize, im.routes);
    let w512 = RANK_WIDTH;
    let mut slices: BTreeMap<i32, KernelSlice> = BTreeMap::new();
    let mut zd_pair: BTreeMap<usize, Vec<f32>> = BTreeMap::new(); // route -> split-summed down output
    let (mut z_err, mut zd_err, mut xd_diff, mut xd_far) = (0f64, 0f64, 0usize, 0usize);
    let rel = |got: &[f32], want: &[f64]| -> f64 {
        let rms = (want.iter().map(|v| v * v).sum::<f64>() / want.len() as f64).sqrt();
        got.iter().zip(want).map(|(&g, &w)| (g as f64 - w).abs()).fold(0f64, f64::max) / rms
    };
    for gp in 0..pn {
        let r = im.pair_route[gp] as usize;
        let e = ids[r];
        let ks = slices.entry(e).or_insert_with(|| KernelSlice::from_block(expert_block(image, e as usize)));
        let row = r / TOPK;
        let mut x = vec![0f32; HIDDEN];
        decode_row(&p[row * HIDDEN..(row + 1) * HIDDEN], &s[row * SCALES_PER_ROW..(row + 1) * SCALES_PER_ROW], &mut x).unwrap();
        let split_sum = |mat: usize| -> Vec<f32> {
            (0..w512)
                .map(|n| {
                    let mut acc = 0f32;
                    for sp in 0..sk {
                        acc += im.z[((mat * sk + sp) * pn + gp) * w512 + n];
                    }
                    acc
                })
                .collect()
        };
        let (zg, zu) = (split_sum(0), split_sum(1));
        z_err = z_err.max(rel(&zg, &matvec(&rotate_in(&x, &ks.gate_suh), &ks.wq_gate, w512)));
        z_err = z_err.max(rel(&zu, &matvec(&rotate_in(&x, &ks.up_suh), &ks.wq_up, w512)));
        let xd_cpu = gateup_epilogue(&zg, &zu, ks, !fp32);
        let xd_bits = &im.xd[gp * w512..(gp + 1) * w512];
        let xd_gpu: Vec<f32> = xd_bits.iter().map(|&h| f16_to_f32(h)).collect();
        for (a, &b) in xd_cpu.iter().zip(xd_bits) {
            let a = f32_to_f16(*a);
            if a != b {
                xd_diff += 1;
                // At most one FP16 step apart (neighbouring codes of one sign).
                if (a ^ b) & 0x8000 != 0 || (a as i32 - b as i32).abs() > 1 {
                    xd_far += 1;
                }
            }
        }
        let zd: Vec<f32> = (0..HIDDEN)
            .map(|h| {
                let mut acc = 0f32;
                for sp in 0..skd {
                    acc += im.zd[(sp * pn + gp) * HIDDEN + h];
                }
                acc
            })
            .collect();
        zd_err = zd_err.max(rel(&zd, &matvec(&xd_gpu, &ks.wq_down, HIDDEN)));
        zd_pair.insert(r, zd);
    }
    // The final reduce from the GPU's down partials, bit for bit.
    let mut reduce_diff = 0usize;
    for row in 0..rows {
        let ys: Vec<Vec<f32>> =
            (0..TOPK).map(|sl| down_epilogue(&zd_pair[&(row * TOPK + sl)], &slices[&ids[row * TOPK + sl]])).collect();
        let mut o = vec![0u16; HIDDEN];
        rank_row(&ys, &w[row * TOPK..(row + 1) * TOPK], &mut o).unwrap();
        reduce_diff += o.iter().zip(&got[row * HIDDEN..(row + 1) * HIDDEN]).filter(|(a, b)| a != b).count();
    }
    eprintln!(
        "stages M{rows} {}: gate/up max rel {z_err:.1e}; epilogue {xd_diff} of {} FP16 values differ ({xd_far} by more than a step); down max rel {zd_err:.1e}; reduce {reduce_diff} BF16 values differ",
        if fp32 { "fp32 swiglu" } else { "bf16 swiglu" },
        pn * w512
    );
    // FP32 accumulation over K = 4,096 (one chain at prefill sizes, eight at
    // decode sizes) against f64: a few 1e-6 of the RMS, up to about 5e-5 at
    // the largest element of a prefill batch.
    assert!(z_err < 1e-4, "gate/up products");
    assert!(xd_far == 0 && xd_diff * 1000 <= pn * w512, "epilogue");
    assert!(zd_err < 1e-4, "down products");
    assert_eq!(reduce_diff, 0, "final reduce");
}

#[test]
fn kernel_matches_the_references() {
    eprintln!("device: {}", exl3_cuda::check_device().expect("device gate"));
    let image = testkit::layer_image(0xC0DA_0001);
    let mut k = CudaKernel::new().unwrap();
    let layer = k.prepare_layer(&image).unwrap();

    for rows in [1usize, 3, 8, 200] {
        for fp32 in [false, true] {
            stagewise(&mut k, &layer, &image, rows, fp32);
        }
    }

    // Decode and verify windows, end to end. The tensor cores round their FP32
    // accumulation differently from the reference's f64 products (about 1e-6
    // relative, see the stages above); a difference that size can tip a BF16
    // (default SwiGLU) or FP16 (down input) rounding by one step, which moves
    // single output values by up to a few percent of the row's RMS. The
    // stage-by-stage check is the tight one; these bound the whole path.
    for rows in 1..=8usize {
        let (p, s) = testkit::wire_rows(0x5EED_1000 + rows as u64, rows);
        let (ids, w) = testkit::routes(0x5EED_2000 + rows as u64, rows, 0);
        let got = run(&mut k, &layer, &p, &s, &ids, &w, rows);
        k.cfg = Some(Cfg { fp32_swiglu: 1, ..Cfg::default() });
        let got32 = run(&mut k, &layer, &p, &s, &ids, &w, rows);
        k.cfg = None;
        let all: Vec<usize> = (0..rows).collect();
        let kref = reference(&image, &p, &s, &ids, &w, &all, Mode::Kernel);
        let kref32 = reference(&image, &p, &s, &ids, &w, &all, Mode::KernelFp32);
        let dref = reference(&image, &p, &s, &ids, &w, &all, Mode::Dense);
        for r in 0..rows {
            let span = r * HIDDEN..(r + 1) * HIDDEN;
            let (kmax, krms, keq) = compare(&got[span.clone()], &kref[&r]);
            let (tmax, trms, teq) = compare(&got32[span.clone()], &kref32[&r]);
            let (dmax, drms, _) = compare(&got[span], &dref[&r]);
            eprintln!(
                "M{rows} row {r}: fp32 swiglu max {tmax:.1e} rms {trms:.1e} equal {teq:.4} | bf16 swiglu max {kmax:.1e} rms {krms:.1e} equal {keq:.3} | model max {dmax:.1e} rms {drms:.1e}"
            );
            assert!(trms < 2e-3 && tmax < 3e-2, "M{rows} row {r}: FP32 SwiGLU vs the kernel-order reference");
            assert!(krms < 3e-3 && kmax < 5e-2, "M{rows} row {r}: BF16 SwiGLU vs the kernel-order reference");
            assert!(drms < 6e-3 && dmax < 6e-2, "M{rows} row {r} vs the dequantized model reference");
        }
    }

    // Batch invariance under the default configuration (up to 64 rows).
    let (p, s) = testkit::wire_rows(0x5EED_3000, 64);
    let (ids, w) = testkit::routes(0x5EED_3001, 64, 0);
    let all64 = run(&mut k, &layer, &p, &s, &ids, &w, 64);
    let first8 = run(&mut k, &layer, &p[..8 * HIDDEN], &s[..8 * SCALES_PER_ROW], &ids[..64], &w[..64], 8);
    assert_eq!(&all64[..8 * HIDDEN], &first8[..], "rows 0-7: window of 64 vs window of 8");
    for r in [0usize, 5, 37, 63] {
        let one = run(
            &mut k,
            &layer,
            &p[r * HIDDEN..(r + 1) * HIDDEN],
            &s[r * SCALES_PER_ROW..(r + 1) * SCALES_PER_ROW],
            &ids[r * TOPK..(r + 1) * TOPK],
            &w[r * TOPK..(r + 1) * TOPK],
            1,
        );
        assert_eq!(&all64[r * HIDDEN..(r + 1) * HIDDEN], &one[..], "row {r}: window of 64 vs alone");
    }
    // A fixed configuration keeps a row's bits at any size, prefill included.
    for c in ["mt=2,sk=1,skd=1", "big=2,mt=4,sk=1,skd=1"] {
        k.cfg = Some(Cfg::default().parse_over(c).unwrap());
        let a = run(&mut k, &layer, &p, &s, &ids, &w, 64);
        let b = run(&mut k, &layer, &p[..HIDDEN], &s[..SCALES_PER_ROW], &ids[..8], &w[..8], 1);
        assert_eq!(&a[..HIDDEN], &b[..], "{c}: row 0 alone vs in 64");
    }
    k.cfg = None;

    // Only the K splits change a bit: the kernel family, the rows per group, the tilings, the plan in the gate/up
    // blocks, the fused epilogue and reduce and the L2 discard are schedules of the same products and sums. Each
    // against the planning kernel and the unfused split kernels at the same splits (the kernels as they were
    // before the large-M family).
    for (rows, sk, skd) in [(8usize, 8, 2), (64, 8, 2), (200, 1, 1), (4096, 1, 1), (512, 2, 2)] {
        let (p, s) = testkit::wire_rows(0x5EED_9000 + rows as u64, rows);
        let (ids, w) = testkit::routes(0x5EED_9100 + rows as u64, rows, 0);
        let base = format!("sk={sk},skd={skd}");
        k.cfg = Some(Cfg::default().parse_over(&format!("big=1,mt=2,{base},plan=1,fuse=1,discard=1")).unwrap());
        let want = run(&mut k, &layer, &p, &s, &ids, &w, rows);
        let mut schedules = vec![
            format!("big=1,mt=1,{base},plan=2,fuse=2,discard=2"),
            format!("big=1,mt=2,{base},plan=2,fuse=1,discard=1"),
            format!("big=1,mt=1,{base},plan=1,fuse=2,discard=1"),
        ];
        for mt in [2, 4] {
            for nt in [1, 2, 4] {
                schedules.push(format!("big=2,mt={mt},nt={nt},{base},fuse=2,discard=2"));
            }
            schedules.push(format!("big=2,mt={mt},nt=2,gw=16,{base},fuse=1,discard=1"));
            schedules.push(format!("big=2,mt={mt},nt=2,gw=8,gp={},{base}", if mt == 2 { 2 } else { -1 }));
        }
        for c in &schedules {
            k.cfg = Some(Cfg::default().parse_over(c).unwrap());
            let got = run(&mut k, &layer, &p, &s, &ids, &w, rows);
            let differ = got.iter().zip(&want).filter(|(a, b)| a != b).count();
            eprintln!("M{rows} {c}: {differ} values differ from the unfused split kernels");
            assert_eq!(differ, 0, "M{rows} {c}");
        }
    }
    k.cfg = None;

    // The split kernels' schedule knobs at every decode and verify size, at the small regime's K splits: the trellis
    // prefetch depth, the L2 evict-first policy, the block order and the L2 discard, every combination, against the
    // unfused split kernels with the planning kernel (the default schedule is one of them).
    for rows in [1usize, 2, 4, 8, 16, 32, 64] {
        let (p, s) = testkit::wire_rows(0x5EED_A000 + rows as u64, rows);
        let (ids, w) = testkit::routes(0x5EED_A100 + rows as u64, rows, 0);
        k.cfg = Some(Cfg::default().parse_over("big=1,mt=2,sk=8,skd=2,plan=1,fuse=1,discard=1").unwrap());
        let want = run(&mut k, &layer, &p, &s, &ids, &w, rows);
        let mut schedules = Vec::new();
        for pf in [1, 2, 4] {
            for l2 in [1, 2] {
                for ord in [1, 2] {
                    for discard in [1, 2] {
                        for pdl in [1, 2] {
                            schedules.push(format!(
                                "big=1,mt=1,sk=8,skd=2,plan=2,fuse=2,pf={pf},l2={l2},ord={ord},discard={discard},pdl={pdl}"
                            ));
                        }
                    }
                }
            }
        }
        schedules.push("big=1,mt=2,sk=8,skd=2,plan=2,fuse=2,discard=2,pf=2,l2=2,ord=2".into());
        schedules.push("big=1,mt=2,sk=8,skd=2,plan=2,fuse=2,discard=2,pf=4,ord=1,pdl=2".into());
        schedules.push("big=1,mt=2,sk=8,skd=2,plan=1,fuse=1,discard=1,pf=4,ord=2".into());
        schedules.push("big=1,mt=1,sk=8,skd=2,plan=1,fuse=2,discard=1,pf=4,l2=2,ord=2".into());
        for c in &schedules {
            k.cfg = Some(Cfg::default().parse_over(c).unwrap());
            let got = run(&mut k, &layer, &p, &s, &ids, &w, rows);
            let differ = got.iter().zip(&want).filter(|(a, b)| a != b).count();
            assert_eq!(differ, 0, "M{rows} {c}");
        }
        eprintln!("M{rows}: {} split-kernel schedules equal the unfused split kernels bit for bit", schedules.len());
    }
    // The same knobs at other K splits: a prefetch ring deeper than a down block's K range (skd 16 and 32: two k
    // tiles and one), the block order with one split per matrix, and a prefill size with the planning kernel.
    for (rows, sk, skd) in [(8usize, 32, 32), (8, 16, 16), (64, 4, 8), (200, 1, 1)] {
        let (p, s) = testkit::wire_rows(0x5EED_B000 + rows as u64, rows);
        let (ids, w) = testkit::routes(0x5EED_B100 + rows as u64, rows, 0);
        let base = format!("big=1,sk={sk},skd={skd}");
        k.cfg = Some(Cfg::default().parse_over(&format!("{base},mt=2,plan=1,fuse=1,discard=1")).unwrap());
        let want = run(&mut k, &layer, &p, &s, &ids, &w, rows);
        for c in [
            format!("{base},mt=1,pf=4,ord=2,l2=2,fuse=2,discard=2"),
            format!("{base},mt=1,pf=2,ord=1,l2=2,fuse=2,discard=1"),
            format!("{base},mt=2,pf=4,ord=2,l2=1,fuse=1,discard=1"),
            format!("{base},mt=2,pf=2,ord=2,l2=2,fuse=2,discard=2"),
            format!("{base},mt=1,pf=4,ord=2,l2=2,fuse=2,discard=2,plan=2,pdl=2"),
        ] {
            k.cfg = Some(Cfg::default().parse_over(&c).unwrap());
            let got = run(&mut k, &layer, &p, &s, &ids, &w, rows);
            let differ = got.iter().zip(&want).filter(|(a, b)| a != b).count();
            eprintln!("M{rows} {c}: {differ} values differ from the unfused split kernels");
            assert_eq!(differ, 0, "M{rows} {c}");
        }
    }
    // The knobs of the split kernels are refused with the large-M kernels, and the programmatic dependent launch
    // without the plan in the gate/up blocks or the fused steps.
    for c in ["big=2,pf=2", "big=2,ord=2", "big=2,pdl=2", "pf=3", "ord=3", "pdl=3"] {
        assert!(exl3_cuda::resolve_cfg(4096, Some(Cfg::default().parse_over(c).unwrap())).is_err(), "{c}");
    }
    for c in ["pdl=2,plan=1", "pdl=2,fuse=1"] {
        assert!(exl3_cuda::resolve_cfg(8, Some(Cfg::default().parse_over(c).unwrap())).is_err(), "{c}");
    }
    k.cfg = None;

    // Prefill.
    for rows in [512usize, 4096] {
        let (p, s) = testkit::wire_rows(0x5EED_4000 + rows as u64, rows);
        let (ids, w) = testkit::routes(0x5EED_5000 + rows as u64, rows, 0);
        let got = run(&mut k, &layer, &p, &s, &ids, &w, rows);
        let sample: Vec<usize> = (0..12).map(|i| i * (rows - 1) / 11).collect();
        k.cfg = Some(Cfg { fp32_swiglu: 1, ..Cfg::default() });
        let got32 = run(&mut k, &layer, &p, &s, &ids, &w, rows);
        k.cfg = None;
        let kref = reference(&image, &p, &s, &ids, &w, &sample, Mode::Kernel);
        let kref32 = reference(&image, &p, &s, &ids, &w, &sample, Mode::KernelFp32);
        for &r in &sample {
            let span = r * HIDDEN..(r + 1) * HIDDEN;
            let (kmax, krms, keq) = compare(&got[span.clone()], &kref[&r]);
            let (tmax, trms, teq) = compare(&got32[span], &kref32[&r]);
            eprintln!("prefill {rows} row {r}: fp32 swiglu max {tmax:.1e} rms {trms:.1e} equal {teq:.4} | bf16 swiglu max {kmax:.1e} rms {krms:.1e} equal {keq:.3}");
            assert!(trms < 2e-3 && tmax < 3e-2, "prefill {rows} row {r}: FP32 SwiGLU");
            assert!(krms < 3e-3 && kmax < 5e-2, "prefill {rows} row {r}: BF16 SwiGLU");
        }
    }

    // The FP32 output (the prefill reduce-scatter's partial) is the same arithmetic: rounded
    // to BF16 it is the default output bit for bit, in both configurations.
    for rows in [1usize, 8, 64, 512, 4096] {
        let (p, s) = testkit::wire_rows(0x5EED_8000 + rows as u64, rows);
        let (ids, w) = testkit::routes(0x5EED_8100 + rows as u64, rows, 0);
        let bf16 = run(&mut k, &layer, &p, &s, &ids, &w, rows);
        let mut f32s = vec![0f32; rows * HIDDEN];
        k.ffn_f32(&layer, Rows::separate(&p, &s, rows).unwrap(), &ids, &w, &mut f32s).unwrap();
        let rounded: Vec<u16> = f32s.iter().map(|&v| glm53f_wire::bf16::f32_to_bf16_rne(v)).collect();
        let differ = rounded.iter().zip(&bf16).filter(|(a, b)| a != b).count();
        eprintln!("M{rows}: FP32 output rounded to BF16 vs the BF16 output: {differ} values differ");
        assert_eq!(differ, 0, "M{rows}: the FP32 output is not the BF16 output's arithmetic");
    }

    // Faults are refused, and the kernel still serves afterwards.
    let (p, s) = testkit::wire_rows(0x5EED_6000, 4);
    let (ids, w) = testkit::routes(0x5EED_6001, 4, 0);
    let mut out = vec![0u16; 4 * HIDDEN];
    let mut bad_p = p.clone();
    bad_p[2 * HIDDEN + 100] = 0x7F;
    let e = k.ffn(&layer, Rows::separate(&bad_p, &s, 4).unwrap(), &ids, &w, &mut out).unwrap_err();
    assert!(e.contains("NaN in a wire row"), "{e}");
    let mut bad_s = s.clone();
    bad_s[SCALES_PER_ROW + 3] = 0xFF;
    let e = k.ffn(&layer, Rows::separate(&p, &bad_s, 4).unwrap(), &ids, &w, &mut out).unwrap_err();
    assert!(e.contains("NaN in a wire row"), "{e}");
    for (i, v) in [(5usize, 288i32), (6, -1)] {
        let mut bad = ids.clone();
        bad[i] = v;
        let e = k.ffn(&layer, Rows::separate(&p, &s, 4).unwrap(), &bad, &w, &mut out).unwrap_err();
        assert!(e.contains("invalid routes"), "{e}");
    }
    let mut dup = ids.clone();
    dup[9] = dup[8];
    assert!(k.ffn(&layer, Rows::separate(&p, &s, 4).unwrap(), &dup, &w, &mut out).is_err());
    let mut neg = w.clone();
    neg[3] = -0.1;
    assert!(k.ffn(&layer, Rows::separate(&p, &s, 4).unwrap(), &ids, &neg, &mut out).is_err());
    let good = run(&mut k, &layer, &p, &s, &ids, &w, 4);
    let kref = reference(&image, &p, &s, &ids, &w, &[0, 3], Mode::Kernel);
    assert!(compare(&good[3 * HIDDEN..], &kref[&3]).1 < 1e-3, "serves after faults");

    // A layer with a non-finite scale is refused at prepare time.
    let mut bad_image = image.clone();
    let off = 7 * glm53f_rank::layout::EXPERT_BYTES + glm53f_rank::layout::DOWN_SVH + 10;
    bad_image[off..off + 2].copy_from_slice(&0x7C00u16.to_le_bytes());
    drop(layer);
    assert!(k.prepare_layer(&bad_image).is_err());
}
