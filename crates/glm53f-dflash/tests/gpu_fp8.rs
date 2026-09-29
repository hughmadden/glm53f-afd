//! The FP8 drafter (feature `cuda`; `glm53f_dflash::gpu`, "The FP8 drafter").
//!
//! - **Its GEMM.** Up to 8 rows, bit for bit `glm53f-layers`' CPU model of the FP8 decode GEMM in
//!   its split order (`mlp::fp8_linear` over the weight `fp8::quantize_weight_bf16` gives, the
//!   checkpoint's block scheme); over 8 rows (W8A8) within 2^-10 of the magnitude of exact block
//!   products (`mlp::fp8_linear_f64`).
//! - **Exact where FP8 is exact.** A random model whose GEMM weights and LM head lie on the FP8
//!   grid, block scales powers of two, so the load-time quantization keeps every value: on every
//!   path of up to 8 rows (appends of up to 8 rows, one request's draft), where each product is
//!   exact in f32, the FP8 drafter is the CPU reference as closely as `tests/gpu.rs` holds the BF16
//!   drafter to it: the ring's bound, the same paths (or a near tie), final rows and logits within
//!   1e-2 (twice the BF16 drafter's bound on this model; measured 4.9e-3, the BF16 drafter's 3.8e-3
//!   on the model's own weights). This pins the codes, the scales, the rows of the fused weights,
//!   the K splits' sum and the head copy.
//! - **What FP8 moves.** The FP8 drafter against the BF16 one on the same arbitrary weights and
//!   contexts, a random model and the checkpoint (`GLM53F_DFLASH_DIR`, `GLM53F_CHECKPOINT_DIR`;
//!   optional `GLM53F_GOLDENS`): how far its rings and logits move, how many of its candidates and
//!   drafts are the BF16 drafter's, its weight bytes, and that its working memory is reserved up
//!   front. The bounds are about twice the values measured on the development GPU. Its drafts only
//!   have to be likely: the target verifies every one. Acceptance on real prompts is a separate
//!   measurement (`examples/draft_replay.rs`).
//!
//! Every test passes with a message when no GPU (or no data) is present.
#![cfg(feature = "cuda")]

use glm53f_dflash::device::device_count;
use glm53f_dflash::goldens::{rel_rms, Set};
use glm53f_dflash::gpu::{fp8_gemm_host, fp8_ksplit, GpuDrafter, GpuSlot};
use glm53f_dflash::reference::{Bf16Head, Context, Draft, DraftOptions, Reference};
use glm53f_dflash::seam::{DraftRequest, Drafter, Proposal};
use glm53f_dflash::selector::Pick;
use glm53f_dflash::weights::{env_dir, Target, Weights};
use glm53f_dflash::{bf16, synth, Dims};
use glm53f_layers::fp8::{e4m3_to_f32, f32_to_e4m3, quantize_weight_bf16, ActScheme};
use glm53f_layers::mlp::{fp8_linear, fp8_linear_f64};

fn gpu_present() -> bool {
    if device_count() == 0 {
        eprintln!("skipped: no CUDA device");
        return false;
    }
    true
}

/// Deterministic values in [-1, 1).
fn rnd(seed: u64, n: usize) -> Vec<f32> {
    (0..n as u64)
        .map(|i| {
            (synth::splitmix64(seed.wrapping_mul(0x1_0000_0001).wrapping_add(i)) >> 40) as f32
                / 8_388_608.0
                - 1.0
        })
        .collect()
}

#[test]
fn fp8_gemm_is_the_cpu_model() {
    if !gpu_present() {
        return;
    }
    for (n, k) in [(264usize, 1024usize), (512, 2560), (2048, 4096)] {
        // Weights whose blocks differ in scale by up to 2^6, and activations with a massive one.
        let w: Vec<f32> = rnd(10 + n as u64, n * k)
            .iter()
            .enumerate()
            .map(|(i, v)| v * 0.02 * (1u32 << ((i / k / 128 + i % k / 128) % 7)) as f32)
            .collect();
        let wb = bf16::encode(&w);
        let m = quantize_weight_bf16(&wb, n, k);
        let ksplit = fp8_ksplit(n, k).unwrap();
        assert!(ksplit >= 2);
        let mut worst = 0f64;
        for rows in [1usize, 3, 7, 8, 9, 16, 40] {
            let mut x = rnd(20 + rows as u64, rows * k);
            x[3] = 900.0;
            let xb = bf16::encode(&x);
            let got = fp8_gemm_host(&xb, rows, &wb, n, k).unwrap();
            if rows <= 8 {
                let want = fp8_linear(&xb, rows, &m, ActScheme::Bf16, ksplit);
                let diff = got
                    .iter()
                    .zip(&want)
                    .filter(|(a, b)| a.to_bits() != b.to_bits())
                    .count();
                assert_eq!(
                    diff, 0,
                    "{rows} x {n} x {k}: {diff} values differ from the CPU model"
                );
            } else {
                let (exact, mag) = fp8_linear_f64(&xb, rows, &m, ActScheme::Fp8Dynamic128);
                let e = got
                    .iter()
                    .zip(exact.iter().zip(&mag))
                    .map(|(&g, (&e, &a))| (g as f64 - e).abs() / a.max(1e-30))
                    .fold(0f64, f64::max);
                // The tensor cores accumulate with fewer bits than f32 (`glm53f-layers`' own test
                // bounds its k32 promotion by 2^-12 on milder data); with the massive activation
                // and the spread of block scales here the worst measured on sm_89 is 4.3e-4.
                assert!(
                    e <= 2f64.powi(-10),
                    "{rows} x {n} x {k}: {e:.2e} of the magnitude from exact block products"
                );
                worst = worst.max(e);
            }
        }
        println!(
            "  {n} x {k}: 1-8 rows bit for bit (K splits {ksplit}), 9-40 rows within {worst:.2e} \
             of the magnitude"
        );
    }
}

fn tiny_dims() -> Dims {
    Dims {
        hidden: 512,
        layers: 2,
        heads: 8,
        kv_heads: 2,
        head_dim: 128,
        inter: 1024,
        vocab: 2048,
        taps: 5,
        group_size: 16,
        conv_taps: 2,
        rank: 64,
        top_k: 16,
        window: 40,
        block: 8,
        mask_token: 2000,
        eps: 1e-5,
        rope_theta: 10_000.0,
    }
}

/// The tiny model's LM head and embedding (as `tests/gpu.rs`).
fn head_and_embed(d: Dims) -> (Vec<u16>, Vec<u16>) {
    let scaled = |seed: u64, s: f32| {
        bf16::encode(
            &rnd(seed, d.vocab * d.hidden)
                .iter()
                .map(|v| s * v)
                .collect::<Vec<_>>(),
        )
    };
    (scaled(22, 0.1), scaled(23, 0.05))
}

/// `w` `[n][k]` moved onto the FP8 block-128 grid with power-of-two scales: in each 128 x 128
/// block (a partial last block of rows too) every value rounded to `e4m3(v / s) * s` for the
/// smallest power of two `s` at least the block's amax / 448, and the block's first value set to
/// `448 s`. The FP8 drafter's load-time scale (`amax / 448`) is then exactly `s` and its codes
/// hold every value exactly; the values are BF16 (at most 4 significant bits).
fn on_fp8_grid(w: &[u16], n: usize, k: usize) -> Vec<u16> {
    assert_eq!((w.len(), k % 128), (n * k, 0));
    let mut out = w.to_vec();
    for r0 in (0..n).step_by(128) {
        let rows = r0..(r0 + 128).min(n);
        for c0 in (0..k).step_by(128) {
            let at = |r: usize, c: usize| r * k + c;
            let amax = rows
                .clone()
                .flat_map(|r| (c0..c0 + 128).map(move |c| at(r, c)))
                .map(|i| bf16::to_f32(w[i]).abs())
                .fold(0f32, f32::max);
            let s = 2f32.powi((amax.max(1e-30) / 448.0).log2().ceil() as i32);
            for r in rows.clone() {
                for c in c0..c0 + 128 {
                    let q = f32_to_e4m3(bf16::to_f32(w[at(r, c)]) / s);
                    out[at(r, c)] = bf16::from_f32(e4m3_to_f32(q) * s);
                }
            }
            out[at(r0, c0)] = bf16::from_f32(448.0 * s);
        }
    }
    out
}

/// `w` with every weight the FP8 drafter quantizes on the FP8 grid ([`on_fp8_grid`]); the norms,
/// the convolutions' base kernels and the selector's tensors as they are.
fn fp8_exact(w: &Weights) -> Weights {
    let d = w.dims;
    let h = d.hidden;
    // The fused QKV and gate/up weights' blocks are their parts' blocks.
    assert!(d.q_width() % 128 == 0 && d.kv_width() % 128 == 0 && d.inter % 128 == 0);
    let mut x = w.clone();
    x.fc = on_fp8_grid(&w.fc, h, d.tap_width());
    for l in &mut x.layers {
        l.q = on_fp8_grid(&l.q, d.q_width(), h);
        l.k = on_fp8_grid(&l.k, d.kv_width(), h);
        l.v = on_fp8_grid(&l.v, d.kv_width(), h);
        l.o = on_fp8_grid(&l.o, h, d.q_width());
        l.gate = on_fp8_grid(&l.gate, d.inter, h);
        l.up = on_fp8_grid(&l.up, d.inter, h);
        l.down = on_fp8_grid(&l.down, h, d.inter);
        l.attn_kp = on_fp8_grid(&l.attn_kp, d.dyn_width(), h);
        l.mlp_kp = on_fp8_grid(&l.mlp_kp, d.dyn_width(), h);
    }
    x
}

/// One BF16 unit of `x`'s magnitude (`tests/gpu.rs`).
fn bf16_unit(x: f32) -> f32 {
    let e = (x.abs().max(f32::MIN_POSITIVE).to_bits() >> 23) as i32 - 127;
    2f32.powi(e - 7)
}

/// Where two walks part, the CPU's own scores must make it a near-tie (`tests/gpu.rs`).
fn agree_or_near_tie(what: &str, gpu: &[u32], cpu: &Draft, k: usize, tol: f32) {
    let n = gpu
        .iter()
        .zip(&cpu.walk.tokens)
        .take_while(|(a, b)| a == b)
        .count();
    if n < gpu.len() {
        let row = &cpu.walk.scores[n * k..(n + 1) * k];
        let best = row[cpu.walk.index[n] as usize];
        let alt = cpu.candidates[n * k..(n + 1) * k]
            .iter()
            .position(|&c| c == gpu[n])
            .map(|i| row[i])
            .unwrap_or_else(|| panic!("{what}: draft {n} is not among the CPU's candidates"));
        println!("  {what}: paths part at draft {n} ({best} against {alt})");
        assert!(
            best - alt <= tol,
            "{what}: not a near-tie ({best} vs {alt})"
        );
    }
}

#[test]
fn fp8_drafter_is_the_reference_on_fp8_exact_weights() {
    if !gpu_present() {
        return;
    }
    let d = tiny_dims();
    let w = fp8_exact(&Weights::random(d, 21));
    let (head, embed) = head_and_embed(d);
    let head = on_fp8_grid(&head, d.vocab, d.hidden);
    let row = |t: u32| &embed[t as usize * d.hidden..(t as usize + 1) * d.hidden];
    let mut g = GpuDrafter::new_fp8(&w, &head, row(d.mask_token)).unwrap();
    let mut r = Reference::new(&w);
    r.bf16_io = true;
    let bhead = Bf16Head {
        weight: &head,
        hidden: d.hidden,
    };
    let mask = bf16::decode(row(d.mask_token));
    // Three requests appended in pieces of at most 8 rows; A past the window (40) and the ring
    // (48).
    let tw = d.tap_width();
    let mut slots: Vec<GpuSlot> = (0..3).map(|_| g.new_slot().unwrap()).collect();
    let mut ctxs: Vec<Context> = (0..3).map(|_| Context::new(d)).collect();
    let plan: [&[usize]; 3] = [
        &[1, 7, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 4],
        &[8, 5],
        &[8, 8, 8, 8, 8, 5],
    ];
    for (i, pieces) in plan.iter().enumerate() {
        let mut at = 0;
        for &n in pieces.iter() {
            let t = synth::taps(30 + i as u64, at, n, tw);
            g.append(&mut [(&mut slots[i], &t[..])]).unwrap();
            r.append(&mut ctxs[i], &bf16::decode(&t));
            at += n;
        }
    }
    let (mut equal, mut total, mut beyond) = (0usize, 0usize, 0usize);
    for (i, c) in ctxs.iter().enumerate() {
        for l in 0..d.layers {
            for p in c.len().saturating_sub(d.window)..c.len() {
                let (gk, gv) = g.ring_row(&slots[i], l, p).unwrap();
                let (ck, cv) = c.row(l, p);
                for (got, want) in [(bf16::decode(&gk), ck), (bf16::decode(&gv), cv)] {
                    let rms = (want.iter().map(|v| v * v).sum::<f32>() / want.len() as f32).sqrt();
                    for (a, b) in got.iter().zip(want) {
                        equal += (a.to_bits() == b.to_bits()) as usize;
                        total += 1;
                        let bound = bf16_unit(a.abs().max(b.abs())) + rms / 4096.0;
                        beyond += ((a - b).abs() > bound) as usize;
                    }
                }
            }
        }
    }
    println!("  FP8-exact weights, rings against the reference: {equal} of {total} values equal");
    assert!(equal * 100 > total * 99, "too many differences");
    assert_eq!(
        beyond, 0,
        "ring values beyond one BF16 unit plus 1/4096 of the row RMS"
    );
    // Each request alone: a block of 8 rows and 7 draft rows, the decode GEMM throughout.
    let opts = DraftOptions {
        vocab_limit: d.vocab,
        pick: Pick::Greedy,
    };
    for (i, anchor) in [5u32, 17, 1999].into_iter().enumerate() {
        let req = DraftRequest {
            slot: &slots[i],
            anchor,
            anchor_embed: row(anchor),
            temperature: 0.0,
            uniforms: &[],
        };
        let p = g.draft(&[req]).unwrap();
        let out = g.outputs().unwrap();
        let cd = r.draft(
            &ctxs[i],
            anchor,
            &bf16::decode(row(anchor)),
            &mask,
            &bhead,
            &opts,
            None,
        );
        let (e, _) = rel_rms(&out.hidden, &cd.hidden);
        let (le, _) = rel_rms(&out.logits, &cd.logits);
        println!(
            "  request {i}: final {e:.2e}, logits {le:.2e}, path {:?} / CPU {:?}",
            p[0].tokens, cd.walk.tokens
        );
        assert!(e < 1e-2 && le < 1e-2, "final {e:.2e}, logits {le:.2e}");
        agree_or_near_tie(&format!("request {i}"), &p[0].tokens, &cd, d.top_k, 1e-2);
    }
}

/// Draft for `anchors[i]` over `slots[i]` (greedy), and the draft's logits.
fn drafts(
    g: &mut GpuDrafter,
    slots: &[&GpuSlot],
    anchors: &[u32],
    rows: &[Vec<u16>],
) -> (Vec<Proposal>, Vec<f32>) {
    let reqs: Vec<DraftRequest<'_, GpuSlot>> = (0..slots.len())
        .map(|i| DraftRequest {
            slot: slots[i],
            anchor: anchors[i],
            anchor_embed: &rows[i],
            temperature: 0.0,
            uniforms: &[],
        })
        .collect();
    let p = g.draft(&reqs).unwrap();
    (p, g.outputs().unwrap().logits)
}

/// Candidates in common (of 16 per draft) and drafts in common, request by request.
fn overlap(a: &[Proposal], b: &[Proposal]) -> (usize, usize, usize) {
    let (mut cands, mut toks, mut total) = (0, 0, 0);
    for (x, y) in a.iter().zip(b) {
        for p in 0..x.tokens.len() {
            let cx = &x.candidates[p * 16..(p + 1) * 16];
            cands += y.candidates[p * 16..(p + 1) * 16]
                .iter()
                .filter(|c| cx.contains(c))
                .count();
        }
        toks += x
            .tokens
            .iter()
            .zip(&y.tokens)
            .filter(|(s, t)| s == t)
            .count();
        total += x.tokens.len();
    }
    (cands, toks, total)
}

#[test]
fn fp8_drafter_on_a_random_model() {
    if !gpu_present() {
        return;
    }
    let d = tiny_dims();
    let w = Weights::random(d, 21);
    let (head, embed) = head_and_embed(d);
    let row = |t: u32| embed[t as usize * d.hidden..(t as usize + 1) * d.hidden].to_vec();
    let mut b = GpuDrafter::new(&w, &head, &row(d.mask_token)).unwrap();
    let mut f = GpuDrafter::new_fp8(&w, &head, &row(d.mask_token)).unwrap();
    assert!(f.is_fp8() && !b.is_fp8());
    // The GEMM weights and the head at a byte a value (plus scales), the rest as before.
    let (bb, fb) = (b.weight_bytes(), f.weight_bytes());
    println!("  weights and head: BF16 {bb} bytes, FP8 {fb} bytes");
    assert!(fb < bb * 6 / 10);
    let reserved = f.reserve(3).unwrap();
    // Three requests, one appended in pieces past the window; then a draft of all three (W8A8,
    // 24 rows) and of each alone (the decode GEMM, 8 rows).
    let tw = d.tap_width();
    let plan: [&[usize]; 3] = [&[1, 7, 60, 32], &[13], &[45]];
    let mut sb: Vec<GpuSlot> = (0..3).map(|_| b.new_slot().unwrap()).collect();
    let mut sf: Vec<GpuSlot> = (0..3).map(|_| f.new_slot().unwrap()).collect();
    for (i, pieces) in plan.iter().enumerate() {
        let mut at = 0;
        for &n in pieces.iter() {
            let t = synth::taps(30 + i as u64, at, n, tw);
            b.append(&mut [(&mut sb[i], &t[..])]).unwrap();
            f.append(&mut [(&mut sf[i], &t[..])]).unwrap();
            at += n;
        }
    }
    // The rings: keys and values within FP8's precision of the BF16 drafter's.
    let (mut num, mut den) = (0f64, 0f64);
    for i in 0..3 {
        for l in 0..d.layers {
            for p in sb[i].len().saturating_sub(d.window)..sb[i].len() {
                let (bk, bv) = b.ring_row(&sb[i], l, p).unwrap();
                let (fk, fv) = f.ring_row(&sf[i], l, p).unwrap();
                for (x, y) in [(bk, fk), (bv, fv)] {
                    for (u, v) in bf16::decode(&x).iter().zip(bf16::decode(&y)) {
                        num += ((u - v) as f64).powi(2);
                        den += (*u as f64).powi(2);
                    }
                }
            }
        }
    }
    let ring = (num / den).sqrt();
    let anchors = [5u32, 17, 1999];
    let rows: Vec<Vec<u16>> = anchors.iter().map(|&a| row(a)).collect();
    let (rb, rf): (Vec<&GpuSlot>, Vec<&GpuSlot>) = (sb.iter().collect(), sf.iter().collect());
    let (pb, lb) = drafts(&mut b, &rb, &anchors, &rows);
    let (pf, lf) = drafts(&mut f, &rf, &anchors, &rows);
    let mut logits = vec![rel_rms(&lf, &lb).0];
    let mut common = vec![overlap(&pf, &pb)];
    for i in 0..3 {
        let (pb1, lb1) = drafts(&mut b, &rb[i..i + 1], &anchors[i..i + 1], &rows[i..i + 1]);
        let (pf1, lf1) = drafts(&mut f, &rf[i..i + 1], &anchors[i..i + 1], &rows[i..i + 1]);
        logits.push(rel_rms(&lf1, &lb1).0);
        common.push(overlap(&pf1, &pb1));
    }
    let (c, t) = (common[0].0, 16 * common[0].2);
    println!(
        "  FP8 against BF16, random model: rings {ring:.2e} (relative RMS); three requests at once \
         (24 rows, W8A8): logits {:.2e}, candidates {c}/{t} in common, drafts {}/{}; each alone \
         (8 rows): logits {:.2e} {:.2e} {:.2e}, candidates {} {} {} of 112",
        logits[0],
        common[0].1,
        common[0].2,
        logits[1],
        logits[2],
        logits[3],
        common[1].0,
        common[2].0,
        common[3].0
    );
    // FP8 E4M3 keeps 3 mantissa bits: a few percent per GEMM, compounded over the layers. Measured on
    // the development GPU: rings 5.1e-2, logits 1.2e-1 to 1.7e-1, 82-87% of the candidates shared.
    assert!(ring < 0.1, "rings {ring:.2e}");
    for (e, (c, _, n)) in logits.iter().zip(&common) {
        assert!(*e < 0.3, "logits {e:.2e}");
        assert!(
            5 * c >= 3 * 16 * n,
            "{c} of {} candidates in common",
            16 * n
        );
    }
    assert_eq!(
        f.scratch_bytes(),
        reserved,
        "the FP8 drafter's working memory grew after reserve"
    );
}

#[test]
fn fp8_drafter_on_the_checkpoint() {
    if !gpu_present() {
        return;
    }
    let (Some(dd), Some(cd)) = (
        env_dir(
            "GLM53F_DFLASH_DIR",
            "the incoai/GLM-5.3-Flash-DFlash2 checkpoint directory",
        ),
        env_dir(
            "GLM53F_CHECKPOINT_DIR",
            "a GLM-5.3-Flash directory holding embed_tokens and lm_head",
        ),
    ) else {
        return;
    };
    let d = Dims::GLM53F;
    let w = Weights::load(&dd, d).unwrap();
    let target = Target::open(&cd, &d).unwrap();
    let head = target.lm_head().unwrap();
    let mask = target.embed_rows(&[d.mask_token]).unwrap();
    let mut b = GpuDrafter::new(&w, &head, &mask).unwrap();
    let mut f = GpuDrafter::new_fp8(&w, &head, &mask).unwrap();
    drop(head);
    let gib = |x: usize| x as f64 / (1u64 << 30) as f64;
    let (bb, fb) = (b.weight_bytes(), f.weight_bytes());
    let head_bf16 = d.vocab * d.hidden * 2;
    println!(
        "  device bytes: BF16 drafter {bb} ({:.3} GiB) and the LM head it borrows in the engine \
         ({head_bf16}, {:.3} GiB); FP8 drafter with its own FP8 head {fb} ({:.3} GiB)",
        gib(bb - head_bf16),
        gib(head_bf16),
        gib(fb)
    );
    // In the engine the FP8 drafter takes less device memory than the BF16 one, head copy and
    // all.
    assert!(fb < bb - head_bf16);
    // The golden cases' first steps (short, window) and a short third context.
    let cases: [(u64, usize, u32); 3] = [(1, 300, 29975), (3, 2100, 1136), (7, 37, 1007)];
    let mut sb = Vec::new();
    let mut sf = Vec::new();
    for &(seed, n, _) in &cases {
        let t = synth::taps(seed, 0, n, d.tap_width());
        let mut x = b.new_slot().unwrap();
        b.append(&mut [(&mut x, &t[..])]).unwrap();
        sb.push(x);
        let mut y = f.new_slot().unwrap();
        f.append(&mut [(&mut y, &t[..])]).unwrap();
        sf.push(y);
    }
    let anchors: Vec<u32> = cases.iter().map(|c| c.2).collect();
    let rows: Vec<Vec<u16>> = anchors
        .iter()
        .map(|&a| target.embed_rows(&[a]).unwrap())
        .collect();
    let goldens = std::env::var_os("GLM53F_GOLDENS").map(std::path::PathBuf::from);
    let cases_run: [(&str, &[usize]); 4] = [
        ("three requests at once (W8A8)", &[0, 1, 2]),
        ("request 0 alone", &[0]),
        ("request 1 alone", &[1]),
        ("request 2 alone", &[2]),
    ];
    for (what, idx) in cases_run {
        let sbi: Vec<&GpuSlot> = idx.iter().map(|&i| &sb[i]).collect();
        let sfi: Vec<&GpuSlot> = idx.iter().map(|&i| &sf[i]).collect();
        let an: Vec<u32> = idx.iter().map(|&i| anchors[i]).collect();
        let rw: Vec<Vec<u16>> = idx.iter().map(|&i| rows[i].clone()).collect();
        let (pb, lb) = drafts(&mut b, &sbi, &an, &rw);
        let (pf, lf) = drafts(&mut f, &sfi, &an, &rw);
        let (le, _) = rel_rms(&lf, &lb);
        let (cands, toks, total) = overlap(&pf, &pb);
        println!(
            "  {what}: FP8 against BF16 logits {le:.2e} (relative RMS), candidates {cands}/{} in \
             common, drafts {toks}/{total}; FP8 {:?} / BF16 {:?}",
            total * 16,
            pf.iter().map(|p| p.tokens.clone()).collect::<Vec<_>>(),
            pb.iter().map(|p| p.tokens.clone()).collect::<Vec<_>>()
        );
        if let (Some(root), [i]) = (&goldens, idx) {
            if *i < 2 {
                let set = Set::open(root, ["dflash-short", "dflash-window"][*i]).unwrap();
                let (gl, _) = rel_rms(&lf, &set.f32("s0.logits").unwrap());
                let path: Vec<u32> = set
                    .i32("s0.path")
                    .unwrap()
                    .iter()
                    .map(|&t| t as u32)
                    .collect();
                println!(
                    "    vs the FP32 golden: FP8 logits {gl:.2e}, path {:?} (golden {path:?})",
                    pf[0].tokens
                );
            }
        }
        // Synthetic taps give flat distributions, where FP8 moves the order most. Measured on the
        // development GPU: logits 1.3e-1 to 2.9e-1, 76-89% of the candidates shared.
        assert!(le < 0.5, "{what}: logits {le:.2e}");
        assert!(
            5 * cands >= 3 * 16 * total,
            "{what}: {cands} of {} candidates in common",
            16 * total
        );
    }
}
