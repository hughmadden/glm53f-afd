//! Invariants of the CPU reference and the seam on a small random model (no data needed): what
//! the context ring, the sliding window, the non-causal block, the convolution's causality, rewinds
//! and the candidate limit must do.

use glm53f_dflash::reference::{Bf16Head, Context, DraftOptions, Reference};
use glm53f_dflash::seam::{Append, CpuDrafter, DraftRequest, Drafter};
use glm53f_dflash::selector::Pick;
use glm53f_dflash::weights::Weights;
use glm53f_dflash::{bf16, synth, Dims};

fn dims() -> Dims {
    Dims {
        hidden: 128,
        layers: 2,
        heads: 4,
        kv_heads: 2,
        head_dim: 32,
        inter: 192,
        vocab: 600,
        taps: 5,
        group_size: 16,
        conv_taps: 2,
        rank: 24,
        top_k: 16,
        window: 24,
        block: 8,
        mask_token: 590,
        eps: 1e-5,
        rope_theta: 10_000.0,
    }
}

struct Model {
    w: Weights,
    head: Vec<u16>,
    embed: Vec<u16>,
}

fn model() -> Model {
    let d = dims();
    let v = |seed: u64, n: usize, scale: f32| -> Vec<u16> {
        (0..n as u64)
            .map(|i| {
                bf16::from_f32(
                    scale
                        * ((synth::splitmix64((seed << 40) + i) >> 40) as f32 / 8_388_608.0 - 1.0),
                )
            })
            .collect()
    };
    Model {
        w: Weights::random(d, 3),
        head: v(4, d.vocab * d.hidden, 0.2),
        embed: v(5, d.vocab * d.hidden, 0.1),
    }
}

impl Model {
    fn row(&self, t: u32) -> Vec<f32> {
        let h = self.w.dims.hidden;
        bf16::decode(&self.embed[t as usize * h..(t as usize + 1) * h])
    }
}

fn taps(seed: u64, start: usize, rows: usize) -> Vec<f32> {
    bf16::decode(&synth::taps(seed, start, rows, dims().tap_width()))
}

fn draft_hidden(m: &Model, r: &Reference<'_>, ctx: &Context, anchor: u32) -> Vec<f32> {
    let head = Bf16Head {
        weight: &m.head,
        hidden: dims().hidden,
    };
    let opts = DraftOptions {
        vocab_limit: dims().vocab,
        pick: Pick::Greedy,
    };
    r.draft(
        ctx,
        anchor,
        &m.row(anchor),
        &m.row(dims().mask_token),
        &head,
        &opts,
        None,
    )
    .hidden
}

#[test]
fn appending_in_pieces_is_appending_at_once() {
    let m = model();
    let r = Reference::new(&m.w);
    let t = taps(1, 0, 50);
    let tw = dims().tap_width();
    let (mut a, mut b) = (Context::new(dims()), Context::new(dims()));
    r.append(&mut a, &t);
    for (lo, hi) in [(0, 1), (1, 9), (9, 50)] {
        r.append(&mut b, &t[lo * tw..hi * tw]);
    }
    assert_eq!(a.len(), 50);
    for l in 0..dims().layers {
        for p in 50 - dims().window..50 {
            assert_eq!(a.row(l, p), b.row(l, p), "layer {l} position {p}");
        }
    }
    assert_eq!(draft_hidden(&m, &r, &a, 7), draft_hidden(&m, &r, &b, 7));
}

#[test]
fn the_window_decides_what_a_draft_reads() {
    let (m, d) = (model(), dims());
    let r = Reference::new(&m.w);
    let n = 60;
    let t = taps(2, 0, n);
    let tw = d.tap_width();
    let mut a = Context::new(d);
    r.append(&mut a, &t);
    // Rows before n - (window - 1) are outside every block query's window: changing them changes
    // nothing. The oldest row inside it changes the draft.
    let outside = n - (d.window - 1);
    let mut t2 = taps(3, 0, n);
    t2[outside * tw..].copy_from_slice(&t[outside * tw..]);
    let mut b = Context::new(d);
    r.append(&mut b, &t2);
    assert_eq!(draft_hidden(&m, &r, &a, 9), draft_hidden(&m, &r, &b, 9));
    let mut t3 = t.clone();
    t3[outside * tw..(outside + 1) * tw].copy_from_slice(&taps(4, outside, 1));
    let mut c = Context::new(d);
    r.append(&mut c, &t3);
    assert_ne!(draft_hidden(&m, &r, &a, 9), draft_hidden(&m, &r, &c, 9));
}

#[test]
fn the_block_is_non_causal_and_the_convolution_causal() {
    let (m, d) = (model(), dims());
    let r = Reference::new(&m.w);
    let mut ctx = Context::new(d);
    r.append(&mut ctx, &taps(5, 0, 10));
    // Row 0 (the anchor) attends to the mask rows after it: a different mask embedding moves it.
    let h = d.hidden;
    let mut e1 = m.row(11);
    for _ in 1..d.block {
        e1.extend(m.row(d.mask_token));
    }
    let mut e2 = e1.clone();
    for v in &mut e2[h..] {
        *v *= 1.5;
    }
    let (o1, o2) = (r.block(&ctx, &e1, None), r.block(&ctx, &e2, None));
    assert_ne!(&o1[..h], &o2[..h]);
    // The convolution: row l reads rows l and l - 1 only; row 0 has no predecessor.
    let dw = d.dyn_width();
    let x: Vec<f32> = (0..d.block * h)
        .map(|i| ((i * 37 % 101) as f32 - 50.0) / 50.0)
        .collect();
    let dy: Vec<f32> = (0..d.block * dw)
        .map(|i| ((i * 13 % 29) as f32 - 14.0) / 30.0)
        .collect();
    let base = &m.w.layers[0].attn_base;
    for side in 0..2 {
        let y = r.conv(&x, &dy, base, side);
        let mut x2 = x.clone();
        for v in &mut x2[3 * h..4 * h] {
            *v += 1.0;
        }
        let y2 = r.conv(&x2, &dy, base, side);
        for l in 0..d.block {
            let changed = y[l * h..(l + 1) * h] != y2[l * h..(l + 1) * h];
            assert_eq!(changed, l == 3 || l == 4, "side {side}: row {l}");
        }
        // Row 0 by hand: (base[side][0] + dyn[0][side][0][g]) * x[0].
        let (g0, taps) = (d.groups(), d.conv_taps);
        for c in 0..h {
            let b = bf16::to_f32(base[side * taps * h + c]);
            let want = 0.0 + b * x[c] + dy[side * taps * g0 + c / d.group_size] * x[c];
            assert_eq!(y[c], want);
        }
    }
}

#[test]
fn rewinds_keep_what_the_ring_holds() {
    let (m, d) = (model(), dims());
    let r = Reference::new(&m.w);
    let n = 70;
    let t = taps(6, 0, n);
    let tw = d.tap_width();
    let mut a = Context::new(d);
    r.append(&mut a, &t);
    let full = draft_hidden(&m, &r, &a, 12);
    // Back one row: the ring still holds the whole window of the draft at n - 1.
    let mut b = a.clone();
    b.rewind(n - 1);
    let mut fresh = Context::new(d);
    r.append(&mut fresh, &t[..(n - 1) * tw]);
    assert_eq!(
        draft_hidden(&m, &r, &b, 12),
        draft_hidden(&m, &r, &fresh, 12)
    );
    // Back further than the ring serves: positions below n - window are masked out ...
    let mut c = a.clone();
    c.rewind(50);
    assert_eq!(c.lo(), n - d.window);
    // ... until the rows are appended again, when every draft reads what it did before.
    r.append(&mut c, &t[50 * tw..]);
    assert_eq!(c.len(), n);
    assert_eq!(draft_hidden(&m, &r, &c, 12), full);
}

#[test]
fn candidates_stay_below_the_limit() {
    let (mut m, d) = (model(), dims());
    let h = d.hidden;
    let r = Reference::new(&m.w);
    let mut ctx = Context::new(d);
    r.append(&mut ctx, &taps(7, 0, 5));
    // Point the "padding" rows 580.. along the drafts' mean hidden state so they win everywhere.
    let first = draft_hidden(&m, &r, &ctx, 1);
    let mean: Vec<f32> = (0..h)
        .map(|c| (1..d.block).map(|j| first[j * h + c]).sum::<f32>())
        .collect();
    for row in 580..d.vocab {
        for (w, &mc) in m.head[row * h..(row + 1) * h].iter_mut().zip(&mean) {
            *w = bf16::from_f32(if mc >= 0.0 { 3.0 } else { -3.0 });
        }
    }
    let head = Bf16Head {
        weight: &m.head,
        hidden: h,
    };
    for limit in [d.vocab, 580] {
        let opts = DraftOptions {
            vocab_limit: limit,
            pick: Pick::Greedy,
        };
        let dr = r.draft(&ctx, 1, &m.row(1), &m.row(d.mask_token), &head, &opts, None);
        let above = dr.candidates.iter().filter(|&&c| c as usize >= 580).count();
        if limit == 580 {
            assert_eq!(above, 0);
            assert!(dr.walk.tokens.iter().all(|&t| t < 580));
        } else {
            assert!(above > 0, "the test needs the padding rows to rank");
        }
    }
}

#[test]
fn the_seam_on_the_cpu_reference() {
    let (m, d) = (model(), dims());
    let mut drafter = CpuDrafter {
        weights: m.w.clone(),
        lm_head: m.head.clone(),
        mask_embed: m.embed
            [d.mask_token as usize * d.hidden..(d.mask_token as usize + 1) * d.hidden]
            .to_vec(),
        vocab_limit: 590,
        bf16_io: true,
    };
    let (mut s1, mut s2) = (drafter.new_slot().unwrap(), drafter.new_slot().unwrap());
    let t1 = synth::taps(8, 0, 30, d.tap_width());
    let t2 = synth::taps(9, 0, 3, d.tap_width());
    drafter
        .append(&mut [
            Append {
                slot: &mut s1,
                taps: &t1,
            },
            Append {
                slot: &mut s2,
                taps: &t2,
            },
        ])
        .unwrap();
    assert_eq!((drafter.len(&s1), drafter.len(&s2)), (30, 3));
    let e = |t: u32| m.embed[t as usize * d.hidden..(t as usize + 1) * d.hidden].to_vec();
    let (e1, e2) = (e(4), e(5));
    let u = [0.3f32; 7];
    let reqs = [
        DraftRequest {
            slot: &s1,
            anchor: 4,
            anchor_embed: &e1,
            temperature: 0.0,
            uniforms: &[],
        },
        DraftRequest {
            slot: &s2,
            anchor: 5,
            anchor_embed: &e2,
            temperature: 0.9,
            uniforms: &u,
        },
    ];
    let p = drafter.draft(&reqs).unwrap();
    assert_eq!(p.len(), 2);
    for (i, prop) in p.iter().enumerate() {
        assert_eq!(prop.tokens.len(), d.drafts());
        assert_eq!(prop.candidates.len(), d.drafts() * d.top_k);
        assert!(prop.tokens.iter().all(|&t| (t as usize) < 590));
        assert!(prop.conf.iter().all(|&c| c > 0.0 && c <= 1.0));
        for (e, row) in prop.q.chunks(d.top_k).enumerate() {
            let sum: f32 = row.iter().sum();
            assert!(
                (sum - 1.0).abs() < 1e-5,
                "request {i} position {e}: q sums to {sum}"
            );
            // The draft is one of its position's candidates.
            assert!(prop.candidates[e * d.top_k..(e + 1) * d.top_k].contains(&prop.tokens[e]));
        }
    }
    // Greedy: q is one-hot on the chosen candidate.
    assert!(p[0].q.iter().all(|&v| v == 0.0 || v == 1.0));
    // Drafting leaves the slots alone: the same request again gives the same proposal.
    assert_eq!(drafter.draft(&reqs[..1]).unwrap()[0], p[0]);
    drafter.rewind(&mut s1, 29).unwrap();
    assert_eq!(drafter.len(&s1), 29);
    assert!(drafter.rewind(&mut s1, 31).is_err());
    drafter.reset(&mut s1);
    assert_eq!(drafter.len(&s1), 0);
}
