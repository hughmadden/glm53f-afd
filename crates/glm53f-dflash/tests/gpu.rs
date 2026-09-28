//! The CUDA kernels and the GPU forward against the CPU reference (feature `cuda`).
//!
//! - Kernels on random data: RMSNorm, per-head norm + RoPE, the dynamic convolution and the
//!   selector walk are bit for bit the CPU functions (the kernels follow their order of
//!   operations); the top-16 is exact.
//! - The forward of a small random model (no checkpoint needed) against the CPU reference in its
//!   BF16-io mode: context rings, a batch of requests at different lengths (sliding window and
//!   ring wrap included), sampled walks and a rewind.
//! - The forward of the real drafter (GLM53F_DFLASH_DIR, GLM53F_CHECKPOINT_DIR; optional
//!   GLM53F_GOLDENS) against the CPU reference and the oracle's FP32 goldens.
//!
//! Every test passes with a message when no GPU (or no data) is present.
#![cfg(feature = "cuda")]

use glm53f_dflash::device::{device_count, DeviceBuffer, Stream};
use glm53f_dflash::goldens::{rel_rms, Set};
use glm53f_dflash::gpu::{GpuDrafter, GpuSlot};
use glm53f_dflash::reference::{Bf16Head, Context, Draft, DraftOptions, Reference};
use glm53f_dflash::seam::{DraftRequest, Drafter};
use glm53f_dflash::selector::{self, Pick};
use glm53f_dflash::weights::{env_dir, Target, Weights};
use glm53f_dflash::{bf16, cpu, ffi, synth, Dims};

fn gpu_present() -> bool {
    if device_count() == 0 {
        eprintln!("skipped: no CUDA device");
        return false;
    }
    true
}

fn req<'a>(
    slot: &'a GpuSlot,
    anchor: u32,
    anchor_embed: &'a [u16],
    temperature: f32,
    uniforms: &'a [f32],
) -> DraftRequest<'a, GpuSlot> {
    DraftRequest {
        slot,
        anchor,
        anchor_embed,
        temperature,
        uniforms,
    }
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

fn dev<T: glm53f_dflash::device::Pod>(v: &[T], s: &Stream) -> DeviceBuffer {
    DeviceBuffer::from_slice(v, s).unwrap()
}

fn same_bits(a: &[f32], b: &[f32]) -> usize {
    a.iter()
        .zip(b)
        .filter(|(x, y)| x.to_bits() != y.to_bits())
        .count()
}

#[test]
fn rmsnorm_kernel_is_the_cpu_function() {
    if !gpu_present() {
        return;
    }
    let s = Stream::new().unwrap();
    let (rows, n) = (5, 4096);
    let mut x = rnd(1, rows * n);
    x[7] = 900.0; // a massive activation, as the drafter's residual has
    let w = bf16::encode(&rnd(2, n));
    let (xd, wd) = (dev(&x, &s), dev(&w, &s));
    let yd = DeviceBuffer::alloc(rows * n * 4).unwrap();
    let bd = DeviceBuffer::alloc(rows * n * 2).unwrap();
    // SAFETY: live buffers of the sizes given.
    let rc = unsafe {
        ffi::g53d_rmsnorm(
            xd.ptr(0),
            n as i64,
            wd.ptr(0),
            rows as i32,
            n as i32,
            1e-5,
            yd.ptr(0),
            n as i64,
            bd.ptr(0),
            n as i64,
            s.raw(),
        )
    };
    assert_eq!(rc, 0);
    let y: Vec<f32> = yd.download(rows * n, &s).unwrap();
    let yb: Vec<u16> = bd.download(rows * n, &s).unwrap();
    let want = cpu::rmsnorm(&x, &w, 1e-5);
    assert_eq!(same_bits(&y, &want), 0);
    assert_eq!(yb, bf16::encode(&want));
}

#[test]
fn head_norm_rope_kernel_is_the_cpu_function() {
    if !gpu_present() {
        return;
    }
    let s = Stream::new().unwrap();
    let (rows, heads, ld) = (6usize, 3usize, 3 * 128 + 5);
    let x = rnd(3, rows * ld);
    let w = bf16::encode(
        &rnd(4, 128)
            .iter()
            .map(|v| 1.0 + 0.3 * v)
            .collect::<Vec<_>>(),
    );
    let pos: Vec<i64> = vec![0, 1, 5, 2047, 65_537, 1_000_000];
    let inv = cpu::inv_freq(10_000.0, 128);
    let (xd, wd, pd, id) = (dev(&x, &s), dev(&w, &s), dev(&pos, &s), dev(&inv, &s));
    let bd = DeviceBuffer::alloc(rows * heads * 128 * 2).unwrap();
    let cs = DeviceBuffer::alloc(rows * 128 * 4).unwrap();
    // SAFETY: live buffers of the sizes given.
    let rc = unsafe { ffi::g53d_rope_table(pd.ptr(0), rows as i32, id.ptr(0), cs.ptr(0), s.raw()) };
    assert_eq!(rc, 0);
    // SAFETY: live buffers of the sizes given.
    let rc = unsafe {
        ffi::g53d_head_norm_rope(
            xd.ptr(0),
            ld as i64,
            rows as i32,
            heads as i32,
            wd.ptr(0),
            cs.ptr(0),
            1e-5,
            bd.ptr(0),
            (heads * 128) as i64,
            s.raw(),
        )
    };
    assert_eq!(rc, 0);
    let y: Vec<f32> = xd.download(rows * ld, &s).unwrap();
    let yb: Vec<u16> = bd.download(rows * heads * 128, &s).unwrap();
    let mut want = x.clone();
    for r in 0..rows {
        for h in 0..heads {
            let head = &mut want[r * ld + h * 128..r * ld + (h + 1) * 128];
            let n = cpu::rmsnorm(head, &w, 1e-5);
            head.copy_from_slice(&n);
            cpu::rope(head, pos[r] as usize, &inv);
            assert_eq!(
                &yb[(r * heads + h) * 128..(r * heads + h + 1) * 128],
                &bf16::encode(head)[..]
            );
        }
    }
    assert_eq!(
        same_bits(&y, &want),
        0,
        "norm + rope differs from the CPU function"
    );
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

#[test]
fn dyn_conv_kernel_is_the_cpu_function() {
    if !gpu_present() {
        return;
    }
    let s = Stream::new().unwrap();
    let d = tiny_dims();
    let w = Weights::random(d, 9);
    let r = Reference::new(&w);
    let (h, dw, blocks) = (d.hidden, d.dyn_width(), 3usize);
    let rows = blocks * d.block;
    let x = rnd(5, rows * h);
    let dynk: Vec<f32> = rnd(6, rows * dw).iter().map(|v| 0.5 * v).collect();
    let base = &w.layers[0].attn_base;
    let (xd, dd, bd) = (dev(&x, &s), dev(&dynk, &s), dev(base, &s));
    let resid0 = rnd(7, rows * h);
    for side in 0..2 {
        let od = DeviceBuffer::alloc(rows * h * 4).unwrap();
        let rd = dev(&resid0, &s);
        // SAFETY: live buffers of the sizes given; the side's columns and base rows.
        let rc = unsafe {
            ffi::g53d_dyn_conv(
                xd.ptr(0),
                h as i64,
                dd.ptr::<f32>(side * d.conv_taps * d.groups()),
                dw as i64,
                bd.ptr::<u16>(side * d.conv_taps * h),
                rows as i32,
                h as i32,
                d.group_size as i32,
                d.conv_taps as i32,
                d.block as i32,
                od.ptr(0),
                h as i64,
                core::ptr::null_mut(),
                0,
                rd.ptr(0),
                h as i64,
                s.raw(),
            )
        };
        assert_eq!(rc, 0);
        let got: Vec<f32> = od.download(rows * h, &s).unwrap();
        let res: Vec<f32> = rd.download(rows * h, &s).unwrap();
        for b in 0..blocks {
            let span = b * d.block * h..(b + 1) * d.block * h;
            let want = r.conv(
                &x[span.clone()],
                &dynk[b * d.block * dw..(b + 1) * d.block * dw],
                base,
                side,
            );
            assert_eq!(
                same_bits(&got[span.clone()], &want),
                0,
                "side {side} block {b}"
            );
            let want_res: Vec<f32> = resid0[span.clone()]
                .iter()
                .zip(&want)
                .map(|(a, b)| a + b)
                .collect();
            assert_eq!(
                same_bits(&res[span], &want_res),
                0,
                "residual, side {side} block {b}"
            );
        }
    }
}

#[test]
fn topk16_kernel_is_exact() {
    if !gpu_present() {
        return;
    }
    let s = Stream::new().unwrap();
    let (rows, vocab, limit) = (5usize, 5000usize, 4990usize);
    let mut x = rnd(8, rows * vocab);
    // Ties: repeat values across ids, and put a large value past the limit.
    for r in 0..rows {
        x[r * vocab + 4995] = 9.0;
        for j in 0..20 {
            x[r * vocab + 100 + 37 * j] = 0.999;
        }
    }
    let xd = dev(&x, &s);
    let vd = DeviceBuffer::alloc(rows * 16 * 4).unwrap();
    let id = DeviceBuffer::alloc(rows * 16 * 4).unwrap();
    // SAFETY: a size computation, then live buffers of the sizes given.
    let ws = DeviceBuffer::alloc(unsafe { ffi::g53d_topk16_workspace_bytes(rows as i32) } as usize)
        .unwrap();
    let rc = unsafe {
        ffi::g53d_topk16(
            xd.ptr(0),
            vocab as i64,
            rows as i32,
            limit as i32,
            ws.ptr(0),
            vd.ptr(0),
            id.ptr(0),
            s.raw(),
        )
    };
    assert_eq!(rc, 0);
    let v: Vec<f32> = vd.download(rows * 16, &s).unwrap();
    let i: Vec<i32> = id.download(rows * 16, &s).unwrap();
    for r in 0..rows {
        let (wv, wi) = selector::top_k(&x[r * vocab..(r + 1) * vocab], 16, limit);
        assert_eq!(&v[r * 16..(r + 1) * 16], &wv[..]);
        assert_eq!(
            i[r * 16..(r + 1) * 16]
                .iter()
                .map(|&t| t as u32)
                .collect::<Vec<_>>(),
            wi
        );
    }
}

#[test]
fn select_kernel_is_the_cpu_walk() {
    if !gpu_present() {
        return;
    }
    let s = Stream::new().unwrap();
    for rank in [256usize, 64, 40] {
        let (nreq, slots, vocab, k) = (4usize, 7usize, 1000usize, 16usize);
        let pred = bf16::encode(&rnd(10, vocab * rank));
        let succ = bf16::encode(&rnd(11, vocab * rank));
        let hp: Vec<f32> = rnd(12, nreq * slots * rank);
        let mut vals = Vec::new();
        let mut ids = Vec::new();
        for row in 0..nreq * slots {
            let mut v: Vec<f32> = rnd(13 + row as u64, k).iter().map(|x| 3.0 * x).collect();
            v.sort_by(|a, b| b.partial_cmp(a).unwrap());
            vals.extend(v);
            ids.extend((0..k).map(|c| ((row * 37 + c * 53) % vocab) as i32));
        }
        let anchors: Vec<i32> = vec![3, 999, 500, 17];
        let temps: Vec<f32> = vec![0.0, 0.7, 1.3, 1e-9];
        let unif: Vec<f32> = rnd(40, nreq * slots)
            .iter()
            .map(|x| 0.5 * (x + 1.0))
            .collect();
        let (pd, sd, hd, vd, idd, ad, td, ud) = (
            dev(&pred, &s),
            dev(&succ, &s),
            dev(&hp, &s),
            dev(&vals, &s),
            dev(&ids, &s),
            dev(&anchors, &s),
            dev(&temps, &s),
            dev(&unif, &s),
        );
        let n = nreq * slots;
        let (tok, idx, sco, qq, cf) = (
            DeviceBuffer::alloc(n * 4).unwrap(),
            DeviceBuffer::alloc(n * 4).unwrap(),
            DeviceBuffer::alloc(n * k * 4).unwrap(),
            DeviceBuffer::alloc(n * k * 4).unwrap(),
            DeviceBuffer::alloc(n * 4).unwrap(),
        );
        // SAFETY: live buffers of the sizes given.
        let rc = unsafe {
            ffi::g53d_select(
                hd.ptr(0),
                vd.ptr(0),
                idd.ptr(0),
                ad.ptr(0),
                pd.ptr(0),
                sd.ptr(0),
                rank as i32,
                nreq as i32,
                slots as i32,
                td.ptr(0),
                ud.ptr(0),
                tok.ptr(0),
                idx.ptr(0),
                sco.ptr(0),
                qq.ptr(0),
                cf.ptr(0),
                s.raw(),
            )
        };
        assert_eq!(rc, 0);
        let tok: Vec<i32> = tok.download(n, &s).unwrap();
        let idx: Vec<i32> = idx.download(n, &s).unwrap();
        let sco: Vec<f32> = sco.download(n * k, &s).unwrap();
        let qq: Vec<f32> = qq.download(n * k, &s).unwrap();
        let cf: Vec<f32> = cf.download(n, &s).unwrap();
        for r in 0..nreq {
            let span = r * slots * k..(r + 1) * slots * k;
            let cands: Vec<u32> = ids[span.clone()].iter().map(|&t| t as u32).collect();
            let u = &unif[r * slots..(r + 1) * slots];
            let pick = if temps[r] > 0.0 {
                Pick::Sample {
                    temperature: temps[r],
                    uniforms: u,
                }
            } else {
                Pick::Greedy
            };
            let w = selector::walk(
                &vals[span.clone()],
                &cands,
                &hp[r * slots * rank..(r + 1) * slots * rank],
                anchors[r] as u32,
                &pred,
                &succ,
                rank,
                k,
                pick,
            );
            let got: Vec<u32> = tok[r * slots..(r + 1) * slots]
                .iter()
                .map(|&t| t as u32)
                .collect();
            assert_eq!(got, w.tokens, "rank {rank} request {r} tokens");
            assert_eq!(
                idx[r * slots..(r + 1) * slots]
                    .iter()
                    .map(|&t| t as u32)
                    .collect::<Vec<_>>(),
                w.index
            );
            assert_eq!(
                same_bits(&sco[span.clone()], &w.scores),
                0,
                "rank {rank} request {r}: scores not bitwise"
            );
            let (qe, _) = rel_rms(&qq[span.clone()], &w.q);
            let (ce, _) = rel_rms(&cf[r * slots..(r + 1) * slots], &w.conf);
            assert!(qe < 1e-6 && ce < 1e-6, "q {qe:.2e} conf {ce:.2e}");
        }
    }
}

/// Where two walks part, the CPU's own scores must make it a near-tie; returns the number of
/// leading tokens that agree.
fn agree_or_near_tie(what: &str, gpu: &[u32], cpu: &Draft, k: usize, tol: f32) -> usize {
    let n = gpu
        .iter()
        .zip(&cpu.walk.tokens)
        .take_while(|(a, b)| a == b)
        .count();
    if n < gpu.len() {
        let row = &cpu.walk.scores[n * k..(n + 1) * k];
        let cands = &cpu.candidates[n * k..(n + 1) * k];
        let best = row[cpu.walk.index[n] as usize];
        let alt = cands.iter().position(|&c| c == gpu[n]).map(|i| row[i]);
        println!(
            "  {what}: paths part at draft {n}: CPU {} ({best}), GPU {} ({alt:?})",
            cpu.walk.tokens[n], gpu[n]
        );
        let alt = alt.unwrap_or_else(|| {
            panic!("{what}: the GPU's draft {n} is not among the CPU's candidates")
        });
        assert!(
            best - alt <= tol,
            "{what}: not a near-tie ({best} vs {alt})"
        );
    }
    n
}

#[test]
fn forward_matches_the_reference_on_a_random_model() {
    if !gpu_present() {
        return;
    }
    let d = tiny_dims();
    let w = Weights::random(d, 21);
    let head = bf16::encode(
        &rnd(22, d.vocab * d.hidden)
            .iter()
            .map(|v| 0.1 * v)
            .collect::<Vec<_>>(),
    );
    let embed = bf16::encode(
        &rnd(23, d.vocab * d.hidden)
            .iter()
            .map(|v| 0.05 * v)
            .collect::<Vec<_>>(),
    );
    let row = |t: u32| &embed[t as usize * d.hidden..(t as usize + 1) * d.hidden];
    let mut g = GpuDrafter::new(&w, &head, row(d.mask_token)).unwrap();
    let mut r = Reference::new(&w);
    r.bf16_io = true;
    let bhead = Bf16Head {
        weight: &head,
        hidden: d.hidden,
    };
    let mask = bf16::decode(row(d.mask_token));

    // Three requests; A is appended in pieces, past the window (40) and the ring (48).
    let tw = d.tap_width();
    let mut slots: Vec<GpuSlot> = (0..3).map(|_| g.new_slot().unwrap()).collect();
    let mut ctxs: Vec<Context> = (0..3).map(|_| Context::new(d)).collect();
    let plan: [&[usize]; 3] = [&[1, 7, 60, 32], &[13], &[45]];
    for (i, pieces) in plan.iter().enumerate() {
        let mut at = 0;
        for &n in pieces.iter() {
            let t = synth::taps(30 + i as u64, at, n, tw);
            g.append(&mut [(&mut slots[i], &t[..])]).unwrap();
            r.append(&mut ctxs[i], &bf16::decode(&t));
            at += n;
        }
    }
    // The rings hold the reference's keys and values (BF16): equal but for GEMM rounding, which
    // moves a value by at most about one BF16 unit at the row's scale.
    let (mut equal, mut total, mut worst) = (0usize, 0usize, 0f32);
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
                        worst = worst.max((a - b).abs() / rms);
                    }
                }
            }
        }
    }
    println!("  ring vs reference: {equal} of {total} values equal; largest difference {worst:.2e} of the row's RMS");
    assert!(equal * 100 > total * 99, "too many differences");
    assert!(
        worst < 1.0 / 64.0,
        "a ring value is off by {worst:.2e} of its row's RMS"
    );

    let anchors = [5u32, 17, 1999];
    let reqs: Vec<DraftRequest<'_, GpuSlot>> = (0..3)
        .map(|i| req(&slots[i], anchors[i], row(anchors[i]), 0.0, &[]))
        .collect();
    let props = g.draft(&reqs).unwrap();
    let out = g.outputs().unwrap();
    for i in 0..3 {
        let opts = DraftOptions {
            vocab_limit: d.vocab,
            pick: Pick::Greedy,
        };
        let cd = r.draft(
            &ctxs[i],
            anchors[i],
            &bf16::decode(row(anchors[i])),
            &mask,
            &bhead,
            &opts,
            None,
        );
        let (e, _) = rel_rms(
            &out.hidden[i * d.block * d.hidden..(i + 1) * d.block * d.hidden],
            &cd.hidden,
        );
        let (le, _) = rel_rms(
            &out.logits[i * d.drafts() * d.vocab..(i + 1) * d.drafts() * d.vocab],
            &cd.logits,
        );
        println!(
            "  request {i} (context {}): final {e:.2e}, logits {le:.2e}, path {:?} / CPU {:?}",
            ctxs[i].len(),
            props[i].tokens,
            cd.walk.tokens
        );
        // Both round GEMM inputs to BF16, but where cuBLAS and the CPU accumulate a value
        // differently it can round the other way: a few elements move by one BF16 unit.
        assert!(e < 5e-3 && le < 5e-3);
        agree_or_near_tie(
            &format!("request {i}"),
            &props[i].tokens,
            &cd,
            d.top_k,
            1e-2,
        );
    }

    // Sampling: the same uniforms give the same draws.
    let u: Vec<f32> = rnd(50, d.drafts())
        .iter()
        .map(|x| 0.5 * (x + 1.0))
        .collect();
    let sp = g
        .draft(&[req(&slots[0], anchors[0], row(anchors[0]), 0.8, &u)])
        .unwrap();
    let opts = DraftOptions {
        vocab_limit: d.vocab,
        pick: Pick::Sample {
            temperature: 0.8,
            uniforms: &u,
        },
    };
    let cd = r.draft(
        &ctxs[0],
        anchors[0],
        &bf16::decode(row(anchors[0])),
        &mask,
        &bhead,
        &opts,
        None,
    );
    println!(
        "  sampled: GPU {:?} / CPU {:?}",
        sp[0].tokens, cd.walk.tokens
    );
    assert_eq!(sp[0].tokens, cd.walk.tokens);
    // Candidates whose logits nearly tie can come out in the other order, or the 16th and 17th
    // can trade places: compare q by token over the candidates both kept.
    let (mut got, mut want, mut missing) = (Vec::new(), Vec::new(), 0);
    for e in 0..d.drafts() {
        for c in 0..16 {
            let tok = cd.candidates[e * 16 + c];
            match sp[0].candidates[e * 16..(e + 1) * 16]
                .iter()
                .position(|&t| t == tok)
            {
                Some(j) => {
                    got.push(sp[0].q[e * 16 + j]);
                    want.push(cd.walk.q[e * 16 + c]);
                }
                None => missing += 1,
            }
        }
    }
    let (qe, _) = rel_rms(&got, &want);
    println!(
        "  sampled: q over the shared candidates {qe:.2e} ({missing} of 112 candidates not shared)"
    );
    assert!(missing <= 2);
    assert!(qe < 2e-2, "q {qe:.2e}");

    // A rewind within the ring, then more context and a draft: still the reference.
    Drafter::rewind(&mut g, &mut slots[0], 97).unwrap();
    ctxs[0].rewind(97);
    let t = synth::taps(99, 97, 2, tw);
    g.append(&mut [(&mut slots[0], &t[..])]).unwrap();
    r.append(&mut ctxs[0], &bf16::decode(&t));
    let p = g.draft(&[req(&slots[0], 7, row(7), 0.0, &[])]).unwrap();
    let cd = r.draft(
        &ctxs[0],
        7,
        &bf16::decode(row(7)),
        &mask,
        &bhead,
        &DraftOptions {
            vocab_limit: d.vocab,
            pick: Pick::Greedy,
        },
        None,
    );
    let (e, _) = rel_rms(&g.outputs().unwrap().hidden, &cd.hidden);
    println!(
        "  after a rewind: final {e:.2e}, lo {} / {}",
        slots[0].lo(),
        ctxs[0].lo()
    );
    assert!(e < 5e-3);
    assert_eq!(slots[0].lo(), ctxs[0].lo());
    agree_or_near_tie("after rewind", &p[0].tokens, &cd, d.top_k, 1e-2);
}

/// A slot over caller-owned memory drafts as an owned slot fed the same rows; a fork (the ring
/// copied, then `follow`) drafts as its source rewound to the fork's length; a cold restart
/// drafts as a rewind past what the ring holds: neither reads a row below `lo`.
#[test]
fn external_rings_forks_and_cold_restarts() {
    if !gpu_present() {
        return;
    }
    let d = tiny_dims();
    let w = Weights::random(d, 31);
    let head = bf16::encode(
        &rnd(32, d.vocab * d.hidden)
            .iter()
            .map(|v| 0.1 * v)
            .collect::<Vec<_>>(),
    );
    let embed = bf16::encode(
        &rnd(33, d.vocab * d.hidden)
            .iter()
            .map(|v| 0.05 * v)
            .collect::<Vec<_>>(),
    );
    let row = |t: u32| &embed[t as usize * d.hidden..(t as usize + 1) * d.hidden];
    let mut g = GpuDrafter::new(&w, &head, row(d.mask_token)).unwrap();
    let tw = d.tap_width();
    let one = |g: &mut GpuDrafter, s: &GpuSlot, anchor: u32| {
        g.draft(&[req(s, anchor, row(anchor), 0.0, &[])]).unwrap()
    };

    // Owned and external rings fed the same 30 rows.
    let arena = DeviceBuffer::alloc(2 * d.ring_bytes()).unwrap();
    let mut a = g.new_slot().unwrap();
    // SAFETY: two rings' worth of device memory, alive until the end of the test.
    let mut b = unsafe { GpuSlot::external(arena.ptr::<u8>(0), &d) };
    let t = synth::taps(40, 0, 30, tw);
    g.append(&mut [(&mut a, &t[..]), (&mut b, &t[..])]).unwrap();
    assert_eq!(one(&mut g, &a, 9), one(&mut g, &b, 9), "external ring");

    // A fork of `b` at 20: its ring copied into the second region, then `follow`.
    // SAFETY: as above.
    let mut c = unsafe { GpuSlot::external(arena.ptr::<u8>(d.ring_bytes()), &d) };
    // SAFETY: both regions lie inside the arena.
    let rc = unsafe {
        glm53f_dflash::cuda::cudaMemcpyAsync(
            arena.ptr::<u8>(d.ring_bytes()).cast(),
            arena.ptr::<u8>(0).cast(),
            d.ring_bytes(),
            glm53f_dflash::cuda::MEMCPY_D2D,
            g.stream().raw(),
        )
    };
    glm53f_dflash::cuda::check(rc, "ring copy").unwrap();
    c.follow(&b, 20).unwrap();
    Drafter::rewind(&mut g, &mut a, 20).unwrap();
    assert_eq!((c.len(), c.lo()), (20, 0));
    assert_eq!((a.len(), a.lo()), (20, 0));
    assert_eq!(one(&mut g, &c, 11), one(&mut g, &a, 11), "fork at 20");
    let t2 = synth::taps(41, 20, 3, tw);
    g.append(&mut [(&mut a, &t2[..])]).unwrap();
    g.append(&mut [(&mut c, &t2[..])]).unwrap();
    assert_eq!(
        one(&mut g, &c, 12),
        one(&mut g, &a, 12),
        "the fork continued"
    );

    // A cold restart at 63 against a rewind past the ring: 103 rows of another context, back
    // to 63 (lo = 103 - 40 = 63), then the same 5 rows appended to both.
    let mut e = g.new_slot().unwrap();
    e.restart(63);
    assert_eq!((e.len(), e.lo(), e.context_rows()), (63, 63, 0));
    let mut f = g.new_slot().unwrap();
    g.append(&mut [(&mut f, &synth::taps(42, 0, 103, tw)[..])])
        .unwrap();
    f.rewind(63).unwrap();
    assert_eq!((f.len(), f.lo()), (63, 63));
    let t3 = synth::taps(43, 63, 5, tw);
    g.append(&mut [(&mut e, &t3[..]), (&mut f, &t3[..])])
        .unwrap();
    assert_eq!((e.context_rows(), f.context_rows()), (5, 5));
    assert_eq!(one(&mut g, &e, 13), one(&mut g, &f, 13), "cold restart");
    println!("  external ring, fork at 20 and cold restart at 63: identical drafts");
}

struct RealEnv {
    w: Weights,
    head: Vec<u16>,
    target: Target,
}

fn real_env() -> Option<RealEnv> {
    let dd = env_dir(
        "GLM53F_DFLASH_DIR",
        "the incoai/GLM-5.3-Flash-DFlash2 checkpoint directory",
    )?;
    let cd = env_dir(
        "GLM53F_CHECKPOINT_DIR",
        "a GLM-5.3-Flash directory holding embed_tokens and lm_head",
    )?;
    let dims = Dims::GLM53F;
    let w = Weights::load(&dd, dims).expect("drafter");
    let target = Target::open(&cd, &dims).expect("target");
    let head = target.lm_head().expect("lm_head");
    Some(RealEnv { w, head, target })
}

#[test]
fn forward_matches_the_reference_on_the_checkpoint() {
    if !gpu_present() {
        return;
    }
    let Some(e) = real_env() else { return };
    let d = Dims::GLM53F;
    let mask = e.target.embed_rows(&[d.mask_token]).unwrap();
    let mut g = GpuDrafter::new(&e.w, &e.head, &mask).unwrap();
    g.trace = true;
    let goldens = std::env::var_os("GLM53F_GOLDENS").map(std::path::PathBuf::from);
    // The golden cases' first steps (short, window) and a short third context.
    let cases: [(u64, usize, u32); 3] = [(1, 300, 29975), (3, 2100, 1136), (7, 37, 1007)];
    let mut slots = Vec::new();
    let mut ctxs = Vec::new();
    let mut r = Reference::new(&e.w);
    r.bf16_io = true;
    for &(seed, n, _) in &cases {
        let t = synth::taps(seed, 0, n, d.tap_width());
        let mut s = g.new_slot().unwrap();
        g.append(&mut [(&mut s, &t[..])]).unwrap();
        let mut c = Context::new(d);
        r.append(&mut c, &bf16::decode(&t));
        slots.push(s);
        ctxs.push(c);
    }
    // The rings against the reference's (BF16) keys and values, request 0.
    let (mut equal, mut total, mut worst) = (0usize, 0usize, 0f32);
    for l in 0..d.layers {
        for p in 0..cases[0].1 {
            let (gk, gv) = g.ring_row(&slots[0], l, p).unwrap();
            let (ck, cv) = ctxs[0].row(l, p);
            for (got, want) in [(bf16::decode(&gk), ck), (bf16::decode(&gv), cv)] {
                let rms = (want.iter().map(|v| v * v).sum::<f32>() / want.len() as f32).sqrt();
                for (a, b) in got.iter().zip(want) {
                    equal += (a.to_bits() == b.to_bits()) as usize;
                    total += 1;
                    worst = worst.max((a - b).abs() / rms);
                }
            }
        }
    }
    println!("  ring (request 0): {equal} of {total} values equal ({:.2}% differ); largest difference {worst:.2e} of the row's RMS", 100.0 * (total - equal) as f64 / total as f64);
    assert!(equal * 100 > total * 95 && worst < 1.0 / 32.0);
    let rows: Vec<Vec<u16>> = cases
        .iter()
        .map(|c| e.target.embed_rows(&[c.2]).unwrap())
        .collect();
    let reqs: Vec<DraftRequest<'_, GpuSlot>> = (0..3)
        .map(|i| req(&slots[i], cases[i].2, &rows[i], 0.0, &[]))
        .collect();
    let props = g.draft(&reqs).unwrap();
    let out = g.outputs().unwrap();
    let bhead = Bf16Head {
        weight: &e.head,
        hidden: d.hidden,
    };
    let maskf = bf16::decode(&mask);
    for i in 0..3 {
        let opts = DraftOptions {
            vocab_limit: glm53f_dflash::SAMPLE_VOCAB,
            pick: Pick::Greedy,
        };
        let mut trace = glm53f_dflash::reference::Trace::new();
        let cd = r.draft(
            &ctxs[i],
            cases[i].2,
            &bf16::decode(&rows[i]),
            &maskf,
            &bhead,
            &opts,
            Some(&mut trace),
        );
        let span = i * d.block * d.hidden..(i + 1) * d.block * d.hidden;
        let per_layer: Vec<String> = (0..d.layers)
            .flat_map(|l| [(l, false, "mid"), (l, true, "out")])
            .map(|(l, mlp, k)| {
                format!(
                    "{:.1e}",
                    rel_rms(
                        &g.layer_trace(l, mlp).unwrap()[span.clone()],
                        &trace[&format!("L{l}.{k}")]
                    )
                    .0
                )
            })
            .collect();
        println!(
            "  request {i}: residual after each site, GPU vs CPU: {}",
            per_layer.join(" ")
        );
        let hid = &out.hidden[i * d.block * d.hidden..(i + 1) * d.block * d.hidden];
        let lg = &out.logits[i * d.drafts() * d.vocab..(i + 1) * d.drafts() * d.vocab];
        let (he, _) = rel_rms(hid, &cd.hidden);
        let (le, _) = rel_rms(lg, &cd.logits);
        let overlap: usize = (0..d.drafts())
            .map(|p| {
                let a = &props[i].candidates[p * 16..(p + 1) * 16];
                cd.candidates[p * 16..(p + 1) * 16]
                    .iter()
                    .filter(|c| a.contains(c))
                    .count()
            })
            .sum();
        println!(
            "  request {i} (context {}): GPU vs CPU (BF16-io) final {he:.2e}, logits {le:.2e}, top-16 overlap {overlap}/112; path {:?} / CPU {:?}",
            ctxs[i].len(),
            props[i].tokens,
            cd.walk.tokens
        );
        // Both paths round the same quantities to BF16, but a few percent of the ring's values
        // round the other way (the context GEMM sums 20,480 products in a different order), and
        // the model carries that to about 0.5% here: the scale of BF16 itself (the reference's
        // BF16-io mode is 0.9% from FP32).
        assert!(
            he < 1.5e-2 && le < 1.5e-2,
            "request {i}: final {he:.2e}, logits {le:.2e}"
        );
        agree_or_near_tie(&format!("request {i}"), &props[i].tokens, &cd, 16, 0.02);
        if let (Some(root), true) = (&goldens, i < 2) {
            let set = Set::open(root, ["dflash-short", "dflash-window"][i]).unwrap();
            let (ge, _) = rel_rms(hid, &set.f32("s0.final").unwrap());
            let (gl, _) = rel_rms(lg, &set.f32("s0.logits").unwrap());
            let path: Vec<u32> = set
                .i32("s0.path")
                .unwrap()
                .iter()
                .map(|&t| t as u32)
                .collect();
            println!("    vs the FP32 golden: final {ge:.2e}, logits {gl:.2e}, path {:?} (golden {path:?})", props[i].tokens);
            assert!(ge < 0.03 && gl < 0.03);
        }
    }

    // The golden cases' second steps: the anchor and the accepted drafts become context (the
    // goldens' synthetic taps), then a draft at the next position.
    let Some(root) = &goldens else { return };
    for (i, set) in ["dflash-short", "dflash-window"].iter().enumerate() {
        let g1 = Set::open(root, set).unwrap();
        let m = g1.manifest().unwrap();
        let notes = m.get("notes").unwrap();
        let case = notes.get("case").unwrap();
        let seed1 = case.get("seed1").and_then(|v| v.as_u64()).unwrap();
        let anchor1 = case.get("anchor1").and_then(|v| v.as_u64()).unwrap() as u32;
        let kept = notes
            .get("kept_rows")
            .and_then(|v| v.as_array())
            .unwrap()
            .len();
        let t = synth::taps(seed1, cases[i].1, kept, d.tap_width());
        g.append(&mut [(&mut slots[i], &t[..])]).unwrap();
        let emb = e.target.embed_rows(&[anchor1]).unwrap();
        let p = g.draft(&[req(&slots[i], anchor1, &emb, 0.0, &[])]).unwrap();
        let o = g.outputs().unwrap();
        let (ge, _) = rel_rms(&o.hidden, &g1.f32("s1.final").unwrap());
        let (gl, _) = rel_rms(&o.logits, &g1.f32("s1.logits").unwrap());
        let path: Vec<u32> = g1
            .i32("s1.path")
            .unwrap()
            .iter()
            .map(|&t| t as u32)
            .collect();
        println!("  {set} step 1 (context {}): vs the FP32 golden final {ge:.2e}, logits {gl:.2e}, path {:?} (golden {path:?})", slots[i].len(), p[0].tokens);
        assert!(ge < 0.03 && gl < 0.03);
    }
}
