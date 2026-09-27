//! The TP expert split: that each format's rule is exact (rank-local
//! dequantization and arithmetic reproduce the unsplit layer), and that the
//! byte runs read from a real safetensors file are the right sub-tensors.

mod common;

use common::*;
use glm53f_model::catalog::*;
use glm53f_model::config::{AttnKind, MlpKind, ModelConfig};
use glm53f_model::dtype::numel;
use glm53f_model::safetensors::{self, Checkpoint, Runs};
use glm53f_model::slicing::*;

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    fn unit(&mut self) -> f64 {
        (self.next() >> 11) as f64 / (1u64 << 53) as f64 * 2.0 - 1.0
    }
    fn bytes(&mut self, n: usize) -> Vec<u8> {
        (0..n).map(|_| self.next() as u8).collect()
    }
}

/// Take a rank's share of a tensor stored as `esize`-byte elements.
fn take(full: &[u8], shape: &[u64], esize: u64, split: Split, tp: Tp) -> (Vec<u64>, Vec<u8>) {
    let (local, runs) = slice_tensor(shape, esize, split, tp).unwrap();
    assert!(runs.fits(full.len() as u64));
    (local, runs.gather(full))
}

fn f64s(bytes: &[u8]) -> Vec<f64> {
    bytes
        .chunks(8)
        .map(|c| f64::from_le_bytes(c.try_into().unwrap()))
        .collect()
}

fn to_bytes(v: &[f64]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

/// Orthonormal Walsh-Hadamard transform of each block of `block` values.
fn hadamard(v: &mut [f64], block: usize) {
    for b in v.chunks_mut(block) {
        let mut h = 1;
        while h < b.len() {
            for i in (0..b.len()).step_by(2 * h) {
                for j in i..i + h {
                    let (x, y) = (b[j], b[j + h]);
                    b[j] = x + y;
                    b[j + h] = x - y;
                }
            }
            h *= 2;
        }
        let s = 1.0 / (b.len() as f64).sqrt();
        b.iter_mut().for_each(|x| *x *= s);
    }
}

/// W_q [k, n] from EXL3-ordered tiles [k/16, n/16, 256] (row-major inside a tile).
fn untile(tiles: &[f64], k: usize, n: usize) -> Vec<f64> {
    let mut w = vec![0.0; k * n];
    for kt in 0..k / 16 {
        for nt in 0..n / 16 {
            for e in 0..256 {
                let t = tiles[(kt * (n / 16) + nt) * 256 + e];
                w[(kt * 16 + e / 16) * n + nt * 16 + e % 16] = t;
            }
        }
    }
    w
}

/// y = ((((x * suh) H_K) W_q) H_N) * svh, Hadamard blocks of `block`.
fn exl3_forward(x: &[f64], w: &[f64], suh: &[f64], svh: &[f64], block: usize) -> Vec<f64> {
    let (k, n) = (x.len(), svh.len());
    let mut xs: Vec<f64> = x.iter().zip(suh).map(|(a, b)| a * b).collect();
    hadamard(&mut xs, block.min(k));
    let mut y = vec![0.0; n];
    for i in 0..k {
        for j in 0..n {
            y[j] += xs[i] * w[i * n + j];
        }
    }
    hadamard(&mut y, block.min(n));
    y.iter().zip(svh).map(|(a, b)| a * b).collect()
}

fn max_diff(a: &[f64], b: &[f64]) -> f64 {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f64::max)
}

const H: usize = 256; // stands in for the hidden size
const I: usize = 512; // stands in for the expert width
const WORLD: u64 = 4; // I / WORLD = 128: one Hadamard block per rank

/// EXL3 gate/up: split outputs. Each rank takes its tile columns and its svh,
/// keeps all of suh, and computes exactly its slice of the unsplit output.
#[test]
fn exl3_output_split_is_exact() {
    let mut rng = Rng(7);
    let (k, n) = (H, I);
    let tiles: Vec<f64> = (0..k * n).map(|_| rng.unit()).collect();
    let suh: Vec<f64> = (0..k).map(|_| rng.unit()).collect();
    let svh: Vec<f64> = (0..n).map(|_| rng.unit()).collect();
    let x: Vec<f64> = (0..k).map(|_| rng.unit()).collect();
    let full = exl3_forward(&x, &untile(&tiles, k, n), &suh, &svh, 128);
    let q = Quant::Exl3 { bits: 4 };
    let tshape = [(k / 16) as u64, (n / 16) as u64, 256];
    let w = n / WORLD as usize;
    for rank in 0..WORLD {
        let tp = Tp { rank, world: WORLD };
        let (tl, tb) = take(
            &to_bytes(&tiles),
            &tshape,
            8,
            split_rule(q, Proj::Gate, Part::Exl3Trellis).unwrap(),
            tp,
        );
        assert_eq!(tl, vec![tshape[0], tshape[1] / WORLD, 256]);
        let (_, sv) = take(
            &to_bytes(&svh),
            &[n as u64],
            8,
            split_rule(q, Proj::Gate, Part::Exl3Svh).unwrap(),
            tp,
        );
        let (_, su) = take(
            &to_bytes(&suh),
            &[k as u64],
            8,
            split_rule(q, Proj::Gate, Part::Exl3Suh).unwrap(),
            tp,
        );
        let y = exl3_forward(&x, &untile(&f64s(&tb), k, w), &f64s(&su), &f64s(&sv), 128);
        let want = &full[rank as usize * w..(rank as usize + 1) * w];
        assert!(max_diff(&y, want) < 1e-12, "rank {rank}");
    }
}

/// EXL3 down: split inputs. Each rank takes its tile rows and its suh, keeps
/// all of svh, and the four partial outputs add up to the unsplit output.
#[test]
fn exl3_input_split_is_exact() {
    let mut rng = Rng(11);
    let (k, n) = (I, H);
    let tiles: Vec<f64> = (0..k * n).map(|_| rng.unit()).collect();
    let suh: Vec<f64> = (0..k).map(|_| rng.unit()).collect();
    let svh: Vec<f64> = (0..n).map(|_| rng.unit()).collect();
    let x: Vec<f64> = (0..k).map(|_| rng.unit()).collect();
    let full = exl3_forward(&x, &untile(&tiles, k, n), &suh, &svh, 128);
    let q = Quant::Exl3 { bits: 4 };
    let tshape = [(k / 16) as u64, (n / 16) as u64, 256];
    let w = k / WORLD as usize;
    let mut sum = vec![0.0; n];
    for rank in 0..WORLD {
        let tp = Tp { rank, world: WORLD };
        let (_, tb) = take(
            &to_bytes(&tiles),
            &tshape,
            8,
            split_rule(q, Proj::Down, Part::Exl3Trellis).unwrap(),
            tp,
        );
        let (_, su) = take(
            &to_bytes(&suh),
            &[k as u64],
            8,
            split_rule(q, Proj::Down, Part::Exl3Suh).unwrap(),
            tp,
        );
        let (_, sv) = take(
            &to_bytes(&svh),
            &[n as u64],
            8,
            split_rule(q, Proj::Down, Part::Exl3Svh).unwrap(),
            tp,
        );
        assert_eq!(f64s(&sv), svh, "svh is whole on every rank");
        let xr = &x[rank as usize * w..(rank as usize + 1) * w];
        let y = exl3_forward(xr, &untile(&f64s(&tb), w, n), &f64s(&su), &f64s(&sv), 128);
        sum.iter_mut().zip(&y).for_each(|(s, v)| *s += v);
    }
    assert!(max_diff(&sum, &full) < 1e-12);
}

/// Why 128: split the same layer so that a rank's 64 channels are half of a
/// Hadamard block, and the rank cannot reproduce its outputs; the slicer
/// refuses such a split.
#[test]
fn exl3_split_inside_a_hadamard_block_is_wrong_and_refused() {
    let mut rng = Rng(5);
    let (k, n, world) = (H, I, 8usize);
    let tiles: Vec<f64> = (0..k * n).map(|_| rng.unit()).collect();
    let suh: Vec<f64> = (0..k).map(|_| rng.unit()).collect();
    let svh: Vec<f64> = (0..n).map(|_| rng.unit()).collect();
    let x: Vec<f64> = (0..k).map(|_| rng.unit()).collect();
    let wq = untile(&tiles, k, n);
    let full = exl3_forward(&x, &wq, &suh, &svh, 128);
    let w = n / world;
    let cols: Vec<f64> = (0..k).flat_map(|i| wq[i * n..i * n + w].to_vec()).collect();
    let y = exl3_forward(&x, &cols, &suh, &svh[..w], 128);
    assert!(
        max_diff(&y, &full[..w]) > 1e-3,
        "a 64-channel share needs the other half of its block"
    );

    let cat = Catalog::from_config(config(), CheckpointFormat::Exl3 { bits: 4 });
    let lin = cat.expert(3, 0, Proj::Gate).unwrap();
    assert!(
        slice_expert(&cat, lin, Tp { rank: 0, world: 8 }).is_ok(),
        "2048 / 8 = 256 channels"
    );
    assert!(
        slice_expert(&cat, lin, Tp { rank: 0, world: 32 }).is_err(),
        "2048 / 32 = 64 channels"
    );
}

/// FP8: every 128 x 128 block keeps its own scale, so a rank's dequantized
/// share equals the matching rows (gate/up) or columns (down) of the whole.
#[test]
fn fp8_rank_dequantization_matches_the_whole() {
    let mut rng = Rng(3);
    let code = |b: u8| (b as f64 - 128.0) / 16.0;
    for (proj, out, inp) in [(Proj::Gate, I, H), (Proj::Down, H, I)] {
        let weight = rng.bytes(out * inp);
        let (sr, sc) = (out / 128, inp / 128);
        let scale: Vec<f64> = (0..sr * sc).map(|_| rng.unit()).collect();
        let deq = |w: &[u8], s: &[f64], cols: usize| -> Vec<f64> {
            let scols = cols / 128;
            (0..w.len())
                .map(|e| code(w[e]) * s[(e / cols / 128) * scols + (e % cols) / 128])
                .collect()
        };
        let whole = deq(&weight, &scale, inp);
        for rank in 0..WORLD {
            let tp = Tp { rank, world: WORLD };
            let (wl, wb) = take(
                &weight,
                &[out as u64, inp as u64],
                1,
                split_rule(Quant::Fp8Block, proj, Part::Weight).unwrap(),
                tp,
            );
            let (_, sb) = take(
                &to_bytes(&scale),
                &[sr as u64, sc as u64],
                8,
                split_rule(Quant::Fp8Block, proj, Part::Fp8ScaleInv).unwrap(),
                tp,
            );
            let local = deq(&wb, &f64s(&sb), wl[1] as usize);
            let r = rank as usize;
            let want: Vec<f64> = match proj {
                Proj::Down => (0..out)
                    .flat_map(|i| {
                        whole[i * inp + r * inp / 4..i * inp + (r + 1) * inp / 4].to_vec()
                    })
                    .collect(),
                _ => whole[r * out / 4 * inp..(r + 1) * out / 4 * inp].to_vec(),
            };
            assert_eq!(local, want, "{proj:?} rank {rank}");
        }
    }
}

/// NVFP4: two values per byte and one scale per 16 values, so a rank's share
/// of whole bytes and whole groups dequantizes to its slice of the whole.
#[test]
fn nvfp4_rank_dequantization_matches_the_whole() {
    let mut rng = Rng(9);
    const E2M1: [f64; 16] = [
        0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
    ];
    for (proj, out, inp) in [(Proj::Up, I, H), (Proj::Down, H, I)] {
        let packed = rng.bytes(out * inp / 2);
        let scales = rng.bytes(out * inp / 16);
        let global = 0.37;
        let deq = |p: &[u8], s: &[u8], cols: usize| -> Vec<f64> {
            (0..p.len() * 2)
                .map(|e| {
                    let (row, col) = (e / cols, e % cols);
                    let b = p[row * cols / 2 + col / 2];
                    let nib = if col % 2 == 0 { b & 15 } else { b >> 4 };
                    E2M1[nib as usize] * (s[row * cols / 16 + col / 16] as f64) * global
                })
                .collect()
        };
        let whole = deq(&packed, &scales, inp);
        for rank in 0..WORLD {
            let tp = Tp { rank, world: WORLD };
            let (wl, wb) = take(
                &packed,
                &[out as u64, inp as u64 / 2],
                1,
                split_rule(Quant::Nvfp4, proj, Part::Nvfp4Weight).unwrap(),
                tp,
            );
            let (_, sb) = take(
                &scales,
                &[out as u64, inp as u64 / 16],
                1,
                split_rule(Quant::Nvfp4, proj, Part::Nvfp4Scale).unwrap(),
                tp,
            );
            let local = deq(&wb, &sb, wl[1] as usize * 2);
            let r = rank as usize;
            let want: Vec<f64> = match proj {
                Proj::Down => (0..out)
                    .flat_map(|i| {
                        whole[i * inp + r * inp / 4..i * inp + (r + 1) * inp / 4].to_vec()
                    })
                    .collect(),
                _ => whole[r * out / 4 * inp..(r + 1) * out / 4 * inp].to_vec(),
            };
            assert_eq!(local, want, "{proj:?} rank {rank}");
        }
    }
}

/// A small model of the same architecture (hidden 256, expert width 512, two
/// experts, one DSA/MoE layer and the MTP layer), small enough to write out.
fn tiny_config() -> ModelConfig {
    let mut c = config().clone();
    let t = &mut c.text;
    t.hidden_size = H as u64;
    t.vocab_size = 64;
    t.num_hidden_layers = 4;
    t.layer_types = vec![AttnKind::Kda, AttnKind::Kda, AttnKind::Kda, AttnKind::Dsa];
    t.mlp_layer_types = vec![MlpKind::Dense, MlpKind::Dense, MlpKind::Dense, MlpKind::Moe];
    t.indexer_types = vec!["full".into(); 4];
    t.kda.num_heads = 2;
    t.kda.head_dim = 16;
    t.mla.num_attention_heads = 2;
    t.mla.q_lora_rank = 32;
    t.mla.kv_lora_rank = 32;
    t.mla.qk_nope_head_dim = 16;
    t.mla.qk_head_dim = 16;
    t.mla.v_head_dim = 16;
    t.indexer.n_heads = 2;
    t.indexer.head_dim = 16;
    t.moe.n_routed_experts = 2;
    t.moe.moe_intermediate_size = I as u64;
    t.moe.dense_intermediate_size = 128;
    let v = &mut c.vision;
    v.depth = 1;
    v.hidden_size = 32;
    v.num_heads = 2;
    v.intermediate_size = 64;
    v.out_hidden_size = H as u64;
    v.projection_intermediate_size = 64;
    v.patch_size = 2;
    c
}

/// Element-wise reference for a rank's share: keep the elements whose index
/// along the split axis is in the rank's range, in row-major order.
fn reference_share(full: &[u8], shape: &[u64], esize: u64, split: Split, tp: Tp) -> Vec<u8> {
    let axis = match split {
        Split::Replicate => return full.to_vec(),
        Split::Axis0 => 0,
        Split::Axis1 => 1,
    };
    let (d, inner) = (shape[axis], numel(&shape[axis + 1..]));
    let per = d / tp.world;
    let mut out = Vec::new();
    for e in 0..numel(shape) {
        let i = (e / inner) % d;
        if i / per == tp.rank {
            let a = (e * esize) as usize;
            out.extend_from_slice(&full[a..a + esize as usize]);
        }
    }
    out
}

/// Write a checkpoint of each format for the tiny model in two shards (with an
/// index), open it, catalogue it, and read every rank's expert shares back
/// through the byte runs: each must equal the element-wise reference.
#[test]
fn rank_shares_read_from_a_checkpoint_file_are_the_right_sub_tensors() {
    let cfg = tiny_config();
    for fmt in [
        CheckpointFormat::OfficialFp8,
        CheckpointFormat::Exl3 { bits: 4 },
        CheckpointFormat::Nvfp4,
    ] {
        let dir = TempDir::new(&format!("slices-{}", fmt.label().replace(' ', "-")));
        let derived = Catalog::from_config(&cfg, fmt);
        let mut rng = Rng(42);
        let data: Vec<Vec<u8>> = derived
            .tensors
            .iter()
            .map(|t| rng.bytes(t.bytes as usize))
            .collect();
        let half = derived.tensors.len() / 2;
        let mut index = Vec::new();
        for (s, range) in [(1, 0..half), (2, half..derived.tensors.len())] {
            let file = format!("model-0000{s}-of-00002.safetensors");
            let items: Vec<_> = range
                .map(|i| {
                    let t = &derived.tensors[i];
                    index.push(format!("\"{}\":\"{file}\"", t.name));
                    (
                        t.name.as_str(),
                        t.dtype,
                        t.shape.as_slice(),
                        data[i].as_slice(),
                    )
                })
                .collect();
            std::fs::write(
                dir.0.join(&file),
                safetensors::serialize(&items, &[("format", "pt")]),
            )
            .unwrap();
        }
        let idx = format!(
            "{{\"metadata\":{{}},\"weight_map\":{{{}}}}}",
            index.join(",")
        );
        std::fs::write(dir.0.join("model.safetensors.index.json"), idx).unwrap();

        let ckpt = Checkpoint::open(&dir.0).unwrap();
        let cat = Catalog::from_shards(&cfg, &ckpt.shards, None, Coverage::Complete).unwrap();
        assert_eq!(cat.format, fmt);
        for rank in 0..WORLD {
            let tp = Tp { rank, world: WORLD };
            let plan = rank_plan(&cat, tp).unwrap();
            assert_eq!(
                plan.experts.len(),
                2 * 2 * 3,
                "two experts in layer 3 and in the MTP layer"
            );
            for e in &plan.experts {
                for s in &e.tensors {
                    let t = &cat.tensors[s.tensor];
                    let full = &data[derived.find(&t.name).unwrap()];
                    let got = ckpt.read_runs(&t.name, &s.runs).unwrap();
                    assert_eq!(
                        got,
                        reference_share(full, &t.shape, t.dtype.size(), s.split, tp),
                        "{} rank {rank}",
                        t.name
                    );
                    assert_eq!(numel(&s.shape) * t.dtype.size(), got.len() as u64);
                }
            }
        }
        // Whole-tensor reads and absolute offsets agree with the data written.
        let name = &derived.tensors[3].name;
        assert_eq!(ckpt.read_tensor(name).unwrap(), data[3]);
        let (file, a, b) = ckpt.file_range(name).unwrap();
        let bytes = std::fs::read(dir.0.join(file)).unwrap();
        assert_eq!(&bytes[a as usize..b as usize], data[3].as_slice());
        assert_eq!(
            cat.get(name).unwrap().loc.as_ref().unwrap().file_range(),
            Some((a, b))
        );
        assert!(ckpt
            .read_runs(
                name,
                &Runs {
                    offset: 0,
                    len: 1,
                    stride: 1,
                    count: data[3].len() as u64 + 1
                }
            )
            .is_err());
    }
}
