//! Real experts from the published EXL3 checkpoint (`tr3-4bpw`), when a copy
//! holding them is named by `GLM53F_EXL3_DIR`; every test here skips without
//! it. `GLM53F_FP8_DIR` may name a copy of the same experts from the official
//! FP8 checkpoint (the cross-check test skips without it).
//!
//! - `GLM53F_TEST_LAYERS` (default `3,4`) and `GLM53F_TEST_EXPERTS` (default
//!   `0,1,100,287`) choose what is read.
//!
//! What is checked:
//!
//! 1. Each rank's share, cut through `glm53f_model`'s slicing plan from the
//!    checkpoint, is byte-identical to the share cut from the whole tensors,
//!    and dequantizes to exactly the rank's columns (gate, up) or rows (down)
//!    of the whole expert.
//! 2. Against the official FP8 weights: the EXL3 expert computes the same
//!    function (the FFN outputs agree to the 4-bit quantization error), and
//!    its intermediate channels are a permutation of the official ones, the
//!    same permutation for gate, up and down (so the checkpoint is internally
//!    consistent and its TP4 split is exact, but a rank's 512 channels are not
//!    the official checkpoint's channels `[512 r, 512 r + 512)`).
//! 3. Against the oracle (`oracle/goldens`, when the payloads are present):
//!    the real MoE inputs of layers 3 and 4 (the prompt and the decode steps),
//!    sent as the wire's FP8 rows with the reference router's top-8 ids and
//!    weights, through all four ranks' shares (kernel-order CPU reference) and
//!    summed, reproduce the reference's routed-expert output (computed with
//!    the official FP8 weights) to the 4-bit quantization error. The same with
//!    the exact FP32 inputs gives the part of the error the FP8 wire adds.
//! 4. (`cuda`) A real layer cut by the slicer, verified by the boot readback,
//!    through the CUDA kernel against the CPU reference; and the kernel on all
//!    four ranks' shares against the oracle, as in 3.

use std::path::PathBuf;

use glm53f_model::catalog::{Catalog, Coverage};
use glm53f_model::config::ModelConfig;
use glm53f_model::safetensors::Checkpoint;
use glm53f_rank::consts::{HIDDEN, INTERMEDIATE, RANK_WIDTH, WORLD};
use glm53f_rank::exl3;
use glm53f_rank::fp8::e4m3_to_f32;
use glm53f_rank::layout::{self, parts, Exl3Expert, Exl3Linear, DOWN_TILES, EXPERT_BYTES, GATE_UP_TILES};
use glm53f_rank::reference::swiglu;
use glm53f_rank::testkit::Rng;

fn env_dir(key: &str) -> Option<PathBuf> {
    let d = PathBuf::from(std::env::var_os(key)?);
    d.is_dir().then_some(d)
}

fn env_list(key: &str, default: &[u64]) -> Vec<u64> {
    match std::env::var(key) {
        Ok(v) => v.split(',').map(|s| s.trim().parse().expect("a number")).collect(),
        Err(_) => default.to_vec(),
    }
}

struct Exl3Copy {
    cp: Checkpoint,
    cat: Catalog,
}

fn open_exl3() -> Option<Exl3Copy> {
    let Some(dir) = env_dir("GLM53F_EXL3_DIR") else {
        eprintln!("GLM53F_EXL3_DIR is not set: skipped");
        return None;
    };
    let cfg = ModelConfig::load(&dir.join("config.json")).expect("config.json");
    let cp = Checkpoint::open(&dir).expect("checkpoint");
    let cat = Catalog::from_shards(&cfg, &cp.shards, None, Coverage::Subset).expect("catalog");
    Some(Exl3Copy { cp, cat })
}

fn name(layer: u64, e: u64, proj: &str, part: &str) -> String {
    format!("model.language_model.layers.{layer}.mlp.experts.{e}.{proj}.{part}")
}

fn full_expert(x: &Exl3Copy, layer: u64, e: u64) -> Option<Exl3Expert> {
    let lin = |proj: &str| -> Option<Exl3Linear> {
        let r = |part: &str| x.cp.read_tensor(&name(layer, e, proj, part)).ok();
        Some(Exl3Linear { trellis: r("trellis")?, suh: r("suh")?, svh: r("svh")?, mcg: r("mcg")? })
    };
    Some(Exl3Expert { gate: lin("gate_proj")?, up: lin("up_proj")?, down: lin("down_proj")? })
}

fn u16s(b: &[u8]) -> Vec<u16> {
    b.chunks_exact(2).map(|c| u16::from_le_bytes([c[0], c[1]])).collect()
}

/// Dequantized whole linear, [in][out] (f64).
fn dense(l: &Exl3Linear, k: usize, n: usize) -> Vec<f64> {
    exl3::dequantize(&l.trellis, k / 16, n / 16, &u16s(&l.suh), &u16s(&l.svh))
}

/// The expert FFN (f64, no roundings) for one row, weights [in][out].
fn ffn(g: &[f64], u: &[f64], d: &[f64], x: &[f64]) -> Vec<f64> {
    let i = INTERMEDIATE;
    let mut gv = vec![0f64; i];
    let mut uv = vec![0f64; i];
    for k in 0..HIDDEN {
        let xv = x[k];
        for j in 0..i {
            gv[j] += xv * g[k * i + j];
            uv[j] += xv * u[k * i + j];
        }
    }
    let act: Vec<f64> = gv.iter().zip(&uv).map(|(&a, &b)| swiglu(a as f32, b as f32, false) as f64).collect();
    let mut y = vec![0f64; HIDDEN];
    for (j, &a) in act.iter().enumerate() {
        for (h, o) in y.iter_mut().enumerate() {
            *o += a * d[j * HIDDEN + h];
        }
    }
    y
}

fn cos(a: &[f64], b: &[f64]) -> f64 {
    let (mut ab, mut aa, mut bb) = (0f64, 0f64, 0f64);
    for (x, y) in a.iter().zip(b) {
        ab += x * y;
        aa += x * x;
        bb += y * y;
    }
    ab / (aa * bb).sqrt()
}

#[test]
fn rank_slices_of_real_experts_are_exact() {
    let Some(x) = open_exl3() else { return };
    for layer in env_list("GLM53F_TEST_LAYERS", &[3, 4]) {
        for e in env_list("GLM53F_TEST_EXPERTS", &[0, 1, 100, 287]) {
            let Some(full) = full_expert(&x, layer, e) else {
                eprintln!("layer {layer} expert {e}: not in this copy");
                continue;
            };
            let wg = dense(&full.gate, HIDDEN, INTERMEDIATE);
            let wd = dense(&full.down, INTERMEDIATE, HIDDEN);
            let rms = |w: &[f64]| (w.iter().map(|v| v * v).sum::<f64>() / w.len() as f64).sqrt();
            eprintln!("layer {layer} expert {e}: weight rms gate {:.5} down {:.5}", rms(&wg), rms(&wd));
            assert!(rms(&wg) > 1e-3 && rms(&wg) < 1.0, "an implausible weight scale: unfetched or corrupt data?");
            for rank in 0..WORLD {
                let mut a = vec![0u8; EXPERT_BYTES];
                let mut b = vec![0u8; EXPERT_BYTES];
                layout::slice_checkpoint_expert(&x.cp, &x.cat, layer as u32, e as usize, rank, &mut a).unwrap();
                layout::slice_expert(&full, rank, &mut b).unwrap();
                assert!(a == b, "layer {layer} expert {e} rank {rank}: the two slicing paths differ");
                let p = parts(&a);
                let (kt, nt) = GATE_UP_TILES;
                let rg = exl3::dequantize(p.gate_trellis, kt, nt, &p.gate_suh, &p.gate_svh);
                let (kt, nt) = DOWN_TILES;
                let rd = exl3::dequantize(p.down_trellis, kt, nt, &p.down_suh, &p.down_svh);
                let c0 = rank * RANK_WIDTH;
                for k in (0..HIDDEN).step_by(61) {
                    for j in 0..RANK_WIDTH {
                        let (s, w) = (rg[k * RANK_WIDTH + j], wg[k * INTERMEDIATE + c0 + j]);
                        assert!((s - w).abs() <= 1e-12 * w.abs().max(1e-6), "gate [{k}, {}]", c0 + j);
                    }
                }
                for j in (0..RANK_WIDTH).step_by(7) {
                    for h in 0..HIDDEN {
                        let (s, w) = (rd[j * HIDDEN + h], wd[(c0 + j) * HIDDEN + h]);
                        assert!((s - w).abs() <= 1e-12 * w.abs().max(1e-6), "down [{}, {h}]", c0 + j);
                    }
                }
            }
        }
    }
}

/// Official FP8 weight [out][in] with its 128 x 128 block scales, as [in][out] f64.
fn fp8_dense(cp: &Checkpoint, layer: u64, e: u64, proj: &str) -> Option<Vec<f64>> {
    let w = cp.read_tensor(&name(layer, e, proj, "weight")).ok()?;
    let s = cp.read_tensor(&name(layer, e, proj, "weight_scale_inv")).ok()?;
    let (out, inp) = if proj == "down_proj" { (HIDDEN, INTERMEDIATE) } else { (INTERMEDIATE, HIDDEN) };
    assert_eq!(w.len(), out * inp);
    let sc: Vec<f32> = s.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect();
    let mut d = vec![0f64; inp * out];
    for o in 0..out {
        for i in 0..inp {
            let v = e4m3_to_f32(w[o * inp + i]) as f64 * sc[(o / 128) * (inp / 128) + i / 128] as f64;
            d[i * out + o] = v;
        }
    }
    Some(d)
}

#[test]
fn real_experts_match_the_official_fp8_experts_up_to_a_channel_permutation() {
    let Some(x) = open_exl3() else { return };
    let Some(fdir) = env_dir("GLM53F_FP8_DIR") else {
        eprintln!("GLM53F_FP8_DIR is not set: skipped");
        return;
    };
    let fp = Checkpoint::open(&fdir).expect("FP8 copy");
    let mut rng = Rng(0xF8F8_0001);
    let mut checked = 0;
    for layer in env_list("GLM53F_TEST_LAYERS", &[3, 4]) {
        for e in env_list("GLM53F_TEST_EXPERTS", &[0, 1, 100, 287]) {
            let (Some(full), Some(g8)) = (full_expert(&x, layer, e), fp8_dense(&fp, layer, e, "gate_proj")) else {
                continue;
            };
            let (u8w, d8) = (fp8_dense(&fp, layer, e, "up_proj").unwrap(), fp8_dense(&fp, layer, e, "down_proj").unwrap());
            let (g, u, d) = (
                dense(&full.gate, HIDDEN, INTERMEDIATE),
                dense(&full.up, HIDDEN, INTERMEDIATE),
                dense(&full.down, INTERMEDIATE, HIDDEN),
            );
            // The same function: FFN outputs on random rows.
            let mut worst = 1f64;
            for _ in 0..4 {
                let xr: Vec<f64> = (0..HIDDEN).map(|_| rng.normal()).collect();
                worst = worst.min(cos(&ffn(&g, &u, &d, &xr), &ffn(&g8, &u8w, &d8, &xr)));
            }
            // A channel permutation: sampled EXL3 gate columns each match one
            // official gate column, the down rows follow the same map.
            let col = |w: &[f64], j: usize| -> Vec<f64> { (0..HIDDEN).map(|k| w[k * INTERMEDIATE + j]).collect() };
            let official: Vec<Vec<f64>> = (0..INTERMEDIATE).map(|j| col(&g8, j)).collect();
            let (mut matched, mut identity, mut min_gate, mut min_down) = (Vec::new(), 0usize, 1f64, 1f64);
            for s in 0..32 {
                let j = (s * 67 + 5) % INTERMEDIATE;
                let cj = col(&g, j);
                let (best, bc) = official
                    .iter()
                    .enumerate()
                    .map(|(m, c)| (m, cos(&cj, c)))
                    .max_by(|a, b| a.1.abs().partial_cmp(&b.1.abs()).unwrap())
                    .unwrap();
                min_gate = min_gate.min(bc);
                identity += usize::from(best == j);
                matched.push(best);
                min_down = min_down.min(cos(&d[j * HIDDEN..(j + 1) * HIDDEN], &d8[best * HIDDEN..(best + 1) * HIDDEN]));
            }
            matched.sort();
            matched.dedup();
            eprintln!(
                "layer {layer} expert {e}: FFN cosine vs FP8 >= {worst:.4}; 32 sampled channels match official channels at cos >= {min_gate:.4} (down rows {min_down:.4}), {identity} of 32 at the same index"
            );
            assert!(worst > 0.98, "the EXL3 expert does not compute the official expert's function");
            assert!(min_gate > 0.99 && min_down > 0.99 && matched.len() == 32, "no consistent channel permutation");
            checked += 1;
        }
    }
    assert!(checked > 0, "no expert present in both copies");
}

/// The oracle's goldens (the manifests are committed; the payloads are
/// regenerated locally, see `oracle/README.md`).
fn goldens() -> Option<PathBuf> {
    let d = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../oracle/goldens");
    d.join("layer03-prefill/prefill.ffn_norm.bin").is_file().then_some(d)
}

fn read_le<T: Copy>(path: &std::path::Path, conv: fn([u8; 4]) -> T) -> Vec<T> {
    let b = std::fs::read(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    b.chunks_exact(4).map(|c| conv([c[0], c[1], c[2], c[3]])).collect()
}

#[test]
fn real_layers_reproduce_the_oracle_routed_output() {
    use glm53f_rank::consts::TOPK;
    use glm53f_rank::fp8::{decode_row, encode_row, SCALES_PER_ROW};
    use glm53f_rank::reference::{expert_partial_kernel_order, rank_row, KernelSlice};
    use glm53f_wire::bf16::bf16_to_f32;
    use std::collections::BTreeMap;
    let Some(x) = open_exl3() else { return };
    let Some(g) = goldens() else {
        eprintln!("oracle golden payloads are not present: skipped");
        return;
    };
    let layers = env_list("GLM53F_TEST_LAYERS", &[3, 4]);
    let mut checked = 0;
    for (set, prefix) in [("prefill", "prefill"), ("decode", "decode")] {
        for &layer in &layers {
            let dir = g.join(format!("layer{layer:02}-{set}"));
            if !dir.join(format!("{prefix}.ffn_norm.bin")).is_file() || x.cat.expert(layer, 0, glm53f_model::catalog::Proj::Gate).is_none() {
                continue;
            }
            let xs = read_le(&dir.join(format!("{prefix}.ffn_norm.bin")), f32::from_le_bytes);
            let ids = read_le(&dir.join(format!("{prefix}.moe.topk_ids.bin")), i32::from_le_bytes);
            let ws = read_le(&dir.join(format!("{prefix}.moe.topk_weights.bin")), f32::from_le_bytes);
            let want = read_le(&dir.join(format!("{prefix}.moe.routed_out.bin")), f32::from_le_bytes);
            let rows = xs.len() / HIDDEN;
            // The rows as the wire carries them (FP8, UE8M0 per 32), decoded as the rank sees them.
            let mut wire = vec![0f32; rows * HIDDEN];
            for r in 0..rows {
                let (mut p, mut sc) = (vec![0u8; HIDDEN], vec![0u8; SCALES_PER_ROW]);
                encode_row(&xs[r * HIDDEN..(r + 1) * HIDDEN], &mut p, &mut sc);
                decode_row(&p, &sc, &mut wire[r * HIDDEN..(r + 1) * HIDDEN]).unwrap();
            }
            // Every (rank, expert) share once, in parallel; each computes its pairs for both inputs.
            let mut jobs: BTreeMap<(usize, i32), Vec<(usize, usize)>> = BTreeMap::new();
            for rank in 0..WORLD {
                for (i, &e) in ids.iter().enumerate() {
                    jobs.entry((rank, e)).or_default().push((i / TOPK, i % TOPK));
                }
            }
            let jobs: Vec<((usize, i32), Vec<(usize, usize)>)> = jobs.into_iter().collect();
            let threads = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4).min(16);
            let mut ys: BTreeMap<(usize, usize, usize), (Vec<f32>, Vec<f32>)> = BTreeMap::new();
            std::thread::scope(|sc| {
                let hs: Vec<_> = (0..threads)
                    .map(|t| {
                        let (jobs, x, xs, wire) = (&jobs, &x, &xs, &wire);
                        sc.spawn(move || {
                            let mut out = Vec::new();
                            for ((rank, e), pairs) in jobs.iter().skip(t).step_by(threads) {
                                let mut block = vec![0u8; EXPERT_BYTES];
                                layout::slice_checkpoint_expert(&x.cp, &x.cat, layer as u32, *e as usize, *rank, &mut block).unwrap();
                                let k = KernelSlice::from_block(&block);
                                for &(r, s) in pairs {
                                    let span = r * HIDDEN..(r + 1) * HIDDEN;
                                    out.push((
                                        (*rank, r, s),
                                        (expert_partial_kernel_order(&wire[span.clone()], &k, true), expert_partial_kernel_order(&xs[span], &k, true)),
                                    ));
                                }
                            }
                            out
                        })
                    })
                    .collect();
                for h in hs {
                    ys.extend(h.join().unwrap());
                }
            });
            // Each rank's BF16 row, the four ranks added in FP32 as the coordinator does.
            let (mut worst_cos, mut sum_rel, mut sum_rel_exact) = (1f64, 0f64, 0f64);
            for r in 0..rows {
                let (mut got, mut got_exact) = (vec![0f32; HIDDEN], vec![0f32; HIDDEN]);
                for rank in 0..WORLD {
                    for (variant, acc) in [(0, &mut got), (1, &mut got_exact)] {
                        let y: Vec<Vec<f32>> = (0..TOPK)
                            .map(|s| {
                                let (a, b) = &ys[&(rank, r, s)];
                                if variant == 0 { a.clone() } else { b.clone() }
                            })
                            .collect();
                        let mut o = vec![0u16; HIDDEN];
                        rank_row(&y, &ws[r * TOPK..(r + 1) * TOPK], &mut o).unwrap();
                        for (a, &c) in acc.iter_mut().zip(&o) {
                            *a += bf16_to_f32(c);
                        }
                    }
                }
                let w = &want[r * HIDDEN..(r + 1) * HIDDEN];
                let wd: Vec<f64> = w.iter().map(|&v| v as f64).collect();
                let rel = |v: &[f32]| -> f64 {
                    let d: f64 = v.iter().zip(w).map(|(&a, &b)| ((a - b) as f64).powi(2)).sum();
                    (d / w.iter().map(|&b| (b as f64).powi(2)).sum::<f64>()).sqrt()
                };
                let gd: Vec<f64> = got.iter().map(|&v| v as f64).collect();
                worst_cos = worst_cos.min(cos(&gd, &wd));
                sum_rel += rel(&got);
                sum_rel_exact += rel(&got_exact);
            }
            eprintln!(
                "layer {layer} {set} ({rows} rows): routed output vs the oracle (official FP8 weights): cosine >= {worst_cos:.4}, mean relative RMS {:.2}% (FP8 wire rows) / {:.2}% (exact FP32 rows)",
                100.0 * sum_rel / rows as f64,
                100.0 * sum_rel_exact / rows as f64
            );
            assert!(worst_cos > 0.97, "layer {layer} {set}: the four ranks do not reproduce the reference routed output");
            checked += 1;
        }
    }
    assert!(checked > 0, "no golden set matched the copy's layers");
}

#[cfg(feature = "cuda")]
#[test]
fn a_real_layer_through_the_slicer_boot_check_and_kernel() {
    use glm53f_rank::consts::TOPK;
    use glm53f_rank::exl3_cuda::CudaKernel;
    use glm53f_rank::kernel::{CpuKernel, ExpertKernel, Rows};
    use glm53f_rank::manifest::Expect;
    use glm53f_wire::bf16::bf16_to_f32;
    let Some(dir) = env_dir("GLM53F_EXL3_DIR") else {
        eprintln!("GLM53F_EXL3_DIR is not set: skipped");
        return;
    };
    let layer = env_list("GLM53F_TEST_LAYERS", &[3, 4])[0] as u32;
    let rank = 1;
    let out = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join(format!("glm53f-rank-real-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&out);
    let t = std::time::Instant::now();
    glm53f_rank::resident::write_rank_dir(&dir, rank, &[layer], &out, "test").unwrap();
    eprintln!("sliced layer {layer} for rank {rank} in {:.1} s", t.elapsed().as_secs_f64());
    let partial = Expect { all_layers: false, ..Expect::SERVING };
    glm53f_rank::boot::readback(&out, rank, &partial).unwrap();
    let image = glm53f_rank::resident::Resident::load_manifest(&out, rank, &partial).unwrap().layer_image(layer).unwrap();
    let mut gpu = CudaKernel::new().unwrap();
    let gl = gpu.prepare_layer(&image).unwrap();
    let mut cpu = CpuKernel;
    let cl = cpu.prepare_layer(&image).unwrap();
    let pool = env_list("GLM53F_TEST_EXPERTS", &[]);
    for rows in [1usize, 8, 256] {
        let (p, s) = glm53f_rank::testkit::wire_rows(0xEA1_0000 + rows as u64, rows);
        let (mut ids, w) = glm53f_rank::testkit::routes(0xEA1_1000 + rows as u64, rows, 0);
        if pool.len() >= TOPK {
            // A partial copy: route only to experts known to be present.
            for (r, id) in ids.iter_mut().enumerate() {
                *id = pool[(r % TOPK + r / TOPK) % pool.len()] as i32;
            }
        }
        let mut a = vec![0u16; rows * glm53f_rank::consts::HIDDEN];
        gpu.ffn(&gl, Rows::separate(&p, &s, rows).unwrap(), &ids, &w, &mut a).unwrap();
        let check = if rows > 8 { vec![0, rows / 2, rows - 1] } else { (0..rows).collect() };
        for r in check {
            let mut b = vec![0u16; glm53f_rank::consts::HIDDEN];
            let span = r * glm53f_rank::consts::HIDDEN..(r + 1) * glm53f_rank::consts::HIDDEN;
            let ps = &p[span.clone()];
            let ss = &s[r * 128..(r + 1) * 128];
            cpu.ffn(&cl, Rows::separate(ps, ss, 1).unwrap(), &ids[r * TOPK..(r + 1) * TOPK], &w[r * TOPK..(r + 1) * TOPK], &mut b)
                .unwrap();
            let (mut dd, mut rr) = (0f64, 0f64);
            for (&g, &c) in a[span].iter().zip(&b) {
                let (g, c) = (bf16_to_f32(g) as f64, bf16_to_f32(c) as f64);
                dd += (g - c) * (g - c);
                rr += c * c;
            }
            eprintln!("real layer {layer}, {rows} rows, row {r}: GPU vs CPU reference rms {:.2e}", (dd / rr).sqrt());
            assert!((dd / rr).sqrt() < 3e-3);
        }
    }
    std::fs::remove_dir_all(&out).ok();
}

/// The CUDA kernel on all four ranks' shares of real layers (one rank on the
/// GPU at a time), fed the oracle's MoE inputs as FP8 wire rows, reproduces
/// the reference routed output like the CPU path above.
#[cfg(feature = "cuda")]
#[test]
fn the_kernel_on_four_ranks_reproduces_the_oracle() {
    use glm53f_rank::consts::TOPK;
    use glm53f_rank::exl3_cuda::CudaKernel;
    use glm53f_rank::fp8::{encode_row, SCALES_PER_ROW};
    use glm53f_rank::kernel::{ExpertKernel, Rows};
    use glm53f_wire::bf16::bf16_to_f32;
    let Some(x) = open_exl3() else { return };
    let Some(g) = goldens() else {
        eprintln!("oracle golden payloads are not present: skipped");
        return;
    };
    let mut k = CudaKernel::new().unwrap();
    for layer in env_list("GLM53F_TEST_LAYERS", &[3, 4]) {
        let dir = g.join(format!("layer{layer:02}-prefill"));
        if !dir.join("prefill.ffn_norm.bin").is_file() {
            continue;
        }
        let xs = read_le(&dir.join("prefill.ffn_norm.bin"), f32::from_le_bytes);
        let ids = read_le(&dir.join("prefill.moe.topk_ids.bin"), i32::from_le_bytes);
        let ws = read_le(&dir.join("prefill.moe.topk_weights.bin"), f32::from_le_bytes);
        let want = read_le(&dir.join("prefill.moe.routed_out.bin"), f32::from_le_bytes);
        let rows = xs.len() / HIDDEN;
        let (mut p, mut sc) = (vec![0u8; rows * HIDDEN], vec![0u8; rows * SCALES_PER_ROW]);
        for r in 0..rows {
            encode_row(
                &xs[r * HIDDEN..(r + 1) * HIDDEN],
                &mut p[r * HIDDEN..(r + 1) * HIDDEN],
                &mut sc[r * SCALES_PER_ROW..(r + 1) * SCALES_PER_ROW],
            );
        }
        let mut sum = vec![0f32; rows * HIDDEN];
        for rank in 0..WORLD {
            let image = layout::slice_checkpoint_layer(&x.cp, &x.cat, layer as u32, rank).unwrap();
            let l = k.prepare_layer(&image).unwrap();
            let mut out = vec![0u16; rows * HIDDEN];
            k.ffn(&l, Rows::separate(&p, &sc, rows).unwrap(), &ids, &ws, &mut out).unwrap();
            for (s, &o) in sum.iter_mut().zip(&out) {
                *s += bf16_to_f32(o);
            }
        }
        assert_eq!(ids.len(), rows * TOPK);
        let (mut worst, mut mean) = (1f64, 0f64);
        for r in 0..rows {
            let a: Vec<f64> = sum[r * HIDDEN..(r + 1) * HIDDEN].iter().map(|&v| v as f64).collect();
            let b: Vec<f64> = want[r * HIDDEN..(r + 1) * HIDDEN].iter().map(|&v| v as f64).collect();
            worst = worst.min(cos(&a, &b));
            let d: f64 = a.iter().zip(&b).map(|(x, y)| (x - y) * (x - y)).sum();
            mean += (d / b.iter().map(|y| y * y).sum::<f64>()).sqrt();
        }
        eprintln!(
            "layer {layer} prefill ({rows} rows), GPU, four ranks: cosine vs the oracle >= {worst:.4}, mean relative RMS {:.2}%",
            100.0 * mean / rows as f64
        );
        assert!(worst > 0.97);
    }
}
