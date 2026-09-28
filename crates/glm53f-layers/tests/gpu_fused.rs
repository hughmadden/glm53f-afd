//! The single-launch kernels of the second revision against the kernel pairs they replace,
//! bit for bit (`--features cuda`; needs a GPU): the decode boundary against
//! `hc_project` + `hc_finish`, the fused router against `router_logits` + `router_select`,
//! and the fused decode GEMM against `fp8_gemm_decode` + `splitk_reduce`; and the E4M3
//! quantization's and the router's division-free arithmetic against division, exhaustively.
//! The pairs are themselves bitwise against the CPU reference (tests/gpu_kernels.rs). Every test also
//! checks that the counters are back to zero, and repeats launches (as graph replays do).
#![cfg(feature = "cuda")]

use glm53f_layers::cuda::{DeviceBuffer, Stream};
use glm53f_layers::fp8::ActScheme;
use glm53f_layers::mhc::{self, HcParams, HC_MULT, HC_PROJ, PARTIAL};
use glm53f_layers::norm::RMS_EPS;
use glm53f_layers::ops::{self, BoundaryDecode, Expand, FinishOut, GemmInput, GemmOutput};
use glm53f_layers::router::{EXPERTS, ROUTED_SCALE, TOP_K};
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
fn bits(v: &[f32]) -> Vec<u32> {
    v.iter().map(|x| x.to_bits()).collect()
}
fn assert_zero(sync: &DeviceBuffer, n: usize, what: &str) {
    assert!(
        down::<u32>(sync, n).iter().all(|&c| c == 0),
        "{what}: counters not re-armed"
    );
}

/// Every output a boundary can write.
struct Outs {
    pre: DeviceBuffer,
    post: DeviceBuffer,
    comb: DeviceBuffer,
    collapsed: DeviceBuffer,
    normed: DeviceBuffer,
    q: DeviceBuffer,
    qs: DeviceBuffer,
}

impl Outs {
    fn new(rows: usize, hidden: usize) -> Self {
        Outs {
            pre: zeros(rows * 16),
            post: zeros(rows * 16),
            comb: zeros(rows * 64),
            collapsed: zeros(rows * hidden * 2),
            normed: zeros(rows * hidden * 2),
            q: zeros(rows * hidden),
            qs: zeros(rows * hidden / 128 * 4),
        }
    }
    fn finish_out(&self, comb: bool) -> FinishOut<'_> {
        FinishOut {
            pre: Some(&self.pre),
            post: Some(&self.post),
            comb: comb.then_some(&self.comb),
            collapsed: Some(&self.collapsed),
            normed: Some(&self.normed),
            quant: Some((&self.q, &self.qs)),
        }
    }
    /// All outputs as one byte vector per output, for comparison.
    fn read(&self, rows: usize, hidden: usize) -> Vec<Vec<u32>> {
        vec![
            bits(&down::<f32>(&self.pre, rows * 4)),
            bits(&down::<f32>(&self.post, rows * 4)),
            bits(&down::<f32>(&self.comb, rows * 16)),
            down::<u16>(&self.collapsed, rows * hidden)
                .iter()
                .map(|&x| x as u32)
                .collect(),
            down::<u16>(&self.normed, rows * hidden)
                .iter()
                .map(|&x| x as u32)
                .collect(),
            down::<u8>(&self.q, rows * hidden)
                .iter()
                .map(|&x| x as u32)
                .collect(),
            bits(&down::<f32>(&self.qs, rows * hidden / 128)),
        ]
    }
}

const NAMES: [&str; 7] = [
    "pre",
    "post",
    "comb",
    "collapsed",
    "normed",
    "normed_q",
    "normed_scales",
];

#[test]
fn boundary_decode_is_the_two_kernels_bitwise() {
    let s = Stream::new().unwrap();
    let mut rng = Rng::new(201);
    // 768: 6 slices, so a row's 150 partials are neither a whole number of float4 nor 16-byte
    // aligned (the scalar staging path).
    for hidden in [4096usize, 512, 768] {
        let slices = hidden / 128;
        let p = HcParams::new(
            hidden,
            rng.bf16_vec(HC_PROJ * HC_MULT * hidden, 0.02),
            &rng.f32_vec(HC_PROJ, 0.5),
            &[0.9, 1.2, 1.1],
        );
        let (dfn, db, dsc, dnw) = (
            up(&p.fn_),
            up(&p.base),
            up(&p.scale),
            up(&rng.bf16_vec(hidden, 0.3)),
        );
        for rows in 1..=8usize {
            let streams = rng.bf16_vec(rows * 4 * hidden, 1.5);
            let h1 = rng.bf16_vec(rows * hidden, 0.5);
            let h2 = rng.bf16_vec(rows * hidden, 0.5);
            let mut post = Vec::new();
            let mut comb = Vec::new();
            for t in 0..rows {
                let m = mhc::mix(
                    &glm53f_layers::bf16::widen(&streams[t * 4 * hidden..(t + 1) * 4 * hidden]),
                    &p,
                    RMS_EPS,
                );
                post.extend(m.post);
                comb.extend(m.comb);
            }
            let (dst, dh1, dh2, dpost, dcomb) =
                (up(&streams), up(&h1), up(&h2), up(&post), up(&comb));
            // 0: no expansion; 1: expansion; 2: expansion of a two-part output.
            for variant in 0..3 {
                // The pair.
                let so_ref = zeros(rows * 4 * hidden * 2);
                let parts_ref = zeros(rows * slices * PARTIAL * 4);
                let e = (variant > 0).then(|| Expand {
                    block_out: &dh1,
                    block_out2: (variant == 2).then_some(&dh2),
                    post: &dpost,
                    comb: &dcomb,
                    streams_out: Some(&so_ref),
                });
                ops::hc_project(&dst, e.as_ref(), Some((&dfn, &parts_ref)), rows, hidden, &s)
                    .unwrap();
                let collapse_from = if variant > 0 { &so_ref } else { &dst };
                let r = Outs::new(rows, hidden);
                ops::hc_finish(
                    &parts_ref,
                    &db,
                    &dsc,
                    collapse_from,
                    Some(&dnw),
                    &r.finish_out(true),
                    rows,
                    hidden,
                    &s,
                )
                .unwrap();
                let want = r.read(rows, hidden);
                let want_streams: Vec<u16> = down(collapse_from, rows * 4 * hidden);
                let want_parts: Vec<f32> = down(&parts_ref, rows * slices * PARTIAL);

                // One launch, separate buffers; then in place with post/comb aliased; then with
                // comb left to glm53f_hc_comb. Twice each, to check the counters re-arm.
                let sync = ops::sync_buffer(rows).unwrap();
                for mode in 0..3 {
                    let st_io = up(&streams);
                    let (post_io, comb_io) = (up(&post), up(&comb));
                    let so = zeros(rows * 4 * hidden * 2);
                    let parts = zeros(rows * slices * PARTIAL * 4);
                    let o = Outs::new(rows, hidden);
                    for _ in 0..2 {
                        let (streams_in, streams_out) = if mode == 1 {
                            (&st_io, &st_io)
                        } else {
                            (&dst, &so)
                        };
                        let (pin, cin) = if mode == 1 {
                            (&post_io, &comb_io)
                        } else {
                            (&dpost, &dcomb)
                        };
                        if mode == 1 {
                            // In place: restore the inputs the previous launch overwrote.
                            st_io.upload(&streams).unwrap();
                            post_io.upload(&post).unwrap();
                            comb_io.upload(&comb).unwrap();
                        }
                        let b = BoundaryDecode {
                            streams_in,
                            expand: (variant > 0).then(|| Expand {
                                block_out: &dh1,
                                block_out2: (variant == 2).then_some(&dh2),
                                post: pin,
                                comb: cin,
                                streams_out: Some(streams_out),
                            }),
                            fn_: &dfn,
                            base: &db,
                            scale: &dsc,
                            norm_weight: Some(&dnw),
                            partials: &parts,
                            sync: &sync,
                        };
                        let fo = if mode == 1 {
                            FinishOut {
                                post: Some(&post_io),
                                comb: Some(&comb_io),
                                ..o.finish_out(false)
                            }
                        } else {
                            o.finish_out(mode == 0)
                        };
                        ops::hc_boundary_decode(&b, &fo, rows, hidden, &s).unwrap();
                        if mode == 2 {
                            ops::hc_comb(&parts, &db, &dsc, &o.comb, rows, hidden, &s).unwrap();
                        }
                        assert_zero(&sync, rows, "boundary");
                    }
                    let mut got = o.read(rows, hidden);
                    if mode == 1 {
                        got[1] = bits(&down::<f32>(&post_io, rows * 4));
                        got[2] = bits(&down::<f32>(&comb_io, rows * 16));
                    }
                    let tag = format!("hidden {hidden} rows {rows} variant {variant} mode {mode}");
                    for (k, name) in NAMES.iter().enumerate() {
                        assert_eq!(got[k], want[k], "{tag}: {name}");
                    }
                    let got_streams: Vec<u16> = if variant == 0 {
                        down(&dst, rows * 4 * hidden)
                    } else if mode == 1 {
                        down(&st_io, rows * 4 * hidden)
                    } else {
                        down(&so, rows * 4 * hidden)
                    };
                    assert_eq!(got_streams, want_streams, "{tag}: streams");
                    assert_eq!(
                        bits(&down::<f32>(&parts, rows * slices * PARTIAL)),
                        bits(&want_parts),
                        "{tag}: partials"
                    );
                }
            }
        }
    }
}

#[test]
fn router_fused_is_the_pair_bitwise() {
    let s = Stream::new().unwrap();
    let mut rng = Rng::new(202);
    for hidden in [4096usize, 1024] {
        let weight = rng.bf16_vec(EXPERTS * hidden, 0.02);
        let bias = rng.f32_vec(EXPERTS, 0.01);
        let (dw, db) = (up(&weight), up(&bias));
        for rows in [1usize, 2, 5, 8, 9, 13, 64, 100, 1027] {
            let x = up(&rng.bf16_vec(rows * hidden, 1.0));
            let (l1, i1, w1) = (
                zeros(rows * EXPERTS * 4),
                zeros(rows * TOP_K * 4),
                zeros(rows * TOP_K * 4),
            );
            ops::router_logits(&x, &dw, &l1, rows, EXPERTS, hidden, &s).unwrap();
            ops::router_select(&l1, &db, &i1, &w1, rows, EXPERTS, TOP_K, ROUTED_SCALE, &s).unwrap();
            let sync = ops::sync_buffer(rows).unwrap();
            for _ in 0..2 {
                let (l2, i2, w2) = (
                    zeros(rows * EXPERTS * 4),
                    zeros(rows * TOP_K * 4),
                    zeros(rows * TOP_K * 4),
                );
                ops::router_fused(
                    &x,
                    &dw,
                    &db,
                    &l2,
                    &sync,
                    &i2,
                    &w2,
                    rows,
                    EXPERTS,
                    hidden,
                    TOP_K,
                    ROUTED_SCALE,
                    &s,
                )
                .unwrap();
                let tag = format!("hidden {hidden} rows {rows}");
                assert_eq!(
                    bits(&down(&l2, rows * EXPERTS)),
                    bits(&down(&l1, rows * EXPERTS)),
                    "{tag}: logits"
                );
                assert_eq!(
                    down::<i32>(&i2, rows * TOP_K),
                    down::<i32>(&i1, rows * TOP_K),
                    "{tag}: ids"
                );
                assert_eq!(
                    bits(&down(&w2, rows * TOP_K)),
                    bits(&down(&w1, rows * TOP_K)),
                    "{tag}: weights"
                );
                assert_zero(&sync, rows, "router");
            }
        }
    }
    // Exact ties (zero weight: every logit 0) and ties made by the bias.
    let hidden = 256;
    let dw = up(&vec![0u16; EXPERTS * hidden]);
    let mut bias = vec![0f32; EXPERTS];
    bias[200] = 0.25;
    bias[9] = 0.25;
    let db = up(&bias);
    let rows = 3;
    let x = up(&rng.bf16_vec(rows * hidden, 1.0));
    let (l, i2, w2) = (
        zeros(rows * EXPERTS * 4),
        zeros(rows * TOP_K * 4),
        zeros(rows * TOP_K * 4),
    );
    let sync = ops::sync_buffer(rows).unwrap();
    ops::router_fused(
        &x,
        &dw,
        &db,
        &l,
        &sync,
        &i2,
        &w2,
        rows,
        EXPERTS,
        hidden,
        TOP_K,
        ROUTED_SCALE,
        &s,
    )
    .unwrap();
    let ids: Vec<i32> = down(&i2, rows * TOP_K);
    assert_eq!(&ids[..8], &[9, 200, 0, 1, 2, 3, 4, 5]);
}

#[test]
fn decode_gemm_fused_is_gemm_plus_reduce() {
    let s = Stream::new().unwrap();
    let mut rng = Rng::new(203);
    for (n, k, ksplits) in [
        (512usize, 4096usize, vec![4, 1]),
        (256, 1024, vec![2, 8]),
        (1536, 4096, vec![4]),
        (128, 1536, vec![3]),
    ] {
        let w = rng.fp8_matrix(n, k);
        let (dw, dws) = (up(&w.data), up(&w.scale_inv));
        let x = rng.bf16_vec(8 * k, 1.0);
        let dx = up(&x);
        let (xq, xs) = (zeros(8 * k), zeros(8 * k / 128 * 4));
        ops::act_quant(&dx, &xq, &xs, 8, k, &s).unwrap();
        let sync = ops::sync_buffer(n / 8).unwrap();
        for scheme in [ActScheme::Bf16, ActScheme::Fp8Dynamic128] {
            let input = match scheme {
                ActScheme::Bf16 => GemmInput::Bf16(&dx),
                ActScheme::Fp8Dynamic128 => GemmInput::Fp8 {
                    q: &xq,
                    scales: &xs,
                },
            };
            for &ksplit in &ksplits {
                for rows in 1..=8usize {
                    let want = zeros(rows * n * 2);
                    let p = zeros(ksplit * rows * n * 4);
                    if ksplit == 1 {
                        ops::fp8_gemm_decode(
                            &input,
                            &dw,
                            &dws,
                            rows,
                            n,
                            k,
                            1,
                            &GemmOutput::Bf16(&want),
                            &s,
                        )
                        .unwrap();
                    } else {
                        ops::fp8_gemm_decode(
                            &input,
                            &dw,
                            &dws,
                            rows,
                            n,
                            k,
                            ksplit,
                            &GemmOutput::Partials(&p),
                            &s,
                        )
                        .unwrap();
                        ops::splitk_reduce(&p, &want, ksplit, rows, n, &s).unwrap();
                    }
                    for _ in 0..2 {
                        let got = zeros(rows * n * 2);
                        let p2 = zeros(ksplit * rows * n * 4);
                        ops::fp8_gemm_decode_fused(
                            &input,
                            &dw,
                            &dws,
                            rows,
                            n,
                            k,
                            ksplit,
                            Some(&p2),
                            Some(&sync),
                            &got,
                            &s,
                        )
                        .unwrap();
                        assert_eq!(
                            down::<u16>(&got, rows * n),
                            down::<u16>(&want, rows * n),
                            "n {n} k {k} ksplit {ksplit} rows {rows} {scheme:?}"
                        );
                        assert_zero(&sync, n / 8, "gemm");
                    }
                }
            }
        }
    }
}

#[test]
fn fused_kernels_replay_in_a_graph() {
    // One graph with a boundary, a router and a split GEMM, launched three times: the
    // counters re-arm between replays and the outputs stay those of the pairs.
    let s = Stream::new().unwrap();
    let mut rng = Rng::new(204);
    let (rows, hidden) = (4usize, 4096usize);
    let slices = hidden / 128;
    let p = HcParams::new(
        hidden,
        rng.bf16_vec(HC_PROJ * HC_MULT * hidden, 0.02),
        &rng.f32_vec(HC_PROJ, 0.5),
        &[1.0, 1.0, 1.0],
    );
    let (dfn, db, dsc, dnw) = (
        up(&p.fn_),
        up(&p.base),
        up(&p.scale),
        up(&rng.bf16_vec(hidden, 0.3)),
    );
    let dst = up(&rng.bf16_vec(rows * 4 * hidden, 1.0));
    let parts = zeros(rows * slices * PARTIAL * 4);
    let (bsync, rsync, gsync) = (
        ops::sync_buffer(rows).unwrap(),
        ops::sync_buffer(rows).unwrap(),
        ops::sync_buffer(64).unwrap(),
    );
    let o = Outs::new(rows, hidden);
    let rw = up(&rng.bf16_vec(EXPERTS * hidden, 0.02));
    let rb = up(&rng.f32_vec(EXPERTS, 0.01));
    let (lg, ids, wt) = (
        zeros(rows * EXPERTS * 4),
        zeros(rows * TOP_K * 4),
        zeros(rows * TOP_K * 4),
    );
    let w = rng.fp8_matrix(512, 4096);
    let (gw, gws) = (up(&w.data), up(&w.scale_inv));
    let (gp, gout) = (zeros(4 * rows * 512 * 4), zeros(rows * 512 * 2));
    let graph = s
        .capture(|s| {
            let b = BoundaryDecode {
                streams_in: &dst,
                expand: None,
                fn_: &dfn,
                base: &db,
                scale: &dsc,
                norm_weight: Some(&dnw),
                partials: &parts,
                sync: &bsync,
            };
            ops::hc_boundary_decode(&b, &o.finish_out(true), rows, hidden, s)?;
            ops::router_fused(
                &o.normed,
                &rw,
                &rb,
                &lg,
                &rsync,
                &ids,
                &wt,
                rows,
                EXPERTS,
                hidden,
                TOP_K,
                ROUTED_SCALE,
                s,
            )?;
            ops::fp8_gemm_decode_fused(
                &GemmInput::Bf16(&o.normed),
                &gw,
                &gws,
                rows,
                512,
                4096,
                4,
                Some(&gp),
                Some(&gsync),
                &gout,
                s,
            )
        })
        .unwrap();
    let mut first = None;
    for _ in 0..3 {
        graph.launch(&s).unwrap();
        s.sync().unwrap();
        assert_zero(&bsync, rows, "boundary");
        assert_zero(&rsync, rows, "router");
        assert_zero(&gsync, 64, "gemm");
        let now = (
            o.read(rows, hidden),
            down::<i32>(&ids, rows * TOP_K),
            down::<u16>(&gout, rows * 512),
        );
        match &first {
            None => first = Some(now),
            Some(f) => assert!(f == &now, "replays differ"),
        }
    }
    // And the pairs give the same.
    let (want_o, want_ids, want_g) = first.unwrap();
    let r = Outs::new(rows, hidden);
    let parts2 = zeros(rows * slices * PARTIAL * 4);
    ops::hc_project(&dst, None, Some((&dfn, &parts2)), rows, hidden, &s).unwrap();
    ops::hc_finish(
        &parts2,
        &db,
        &dsc,
        &dst,
        Some(&dnw),
        &r.finish_out(true),
        rows,
        hidden,
        &s,
    )
    .unwrap();
    assert!(r.read(rows, hidden) == want_o);
    let (l1, i1, w1) = (
        zeros(rows * EXPERTS * 4),
        zeros(rows * TOP_K * 4),
        zeros(rows * TOP_K * 4),
    );
    ops::router_logits(&r.normed, &rw, &l1, rows, EXPERTS, hidden, &s).unwrap();
    ops::router_select(&l1, &rb, &i1, &w1, rows, EXPERTS, TOP_K, ROUTED_SCALE, &s).unwrap();
    assert_eq!(down::<i32>(&i1, rows * TOP_K), want_ids);
    let (p1, g1) = (zeros(4 * rows * 512 * 4), zeros(rows * 512 * 2));
    ops::fp8_gemm_decode(
        &GemmInput::Bf16(&r.normed),
        &gw,
        &gws,
        rows,
        512,
        4096,
        4,
        &GemmOutput::Partials(&p1),
        &s,
    )
    .unwrap();
    ops::splitk_reduce(&p1, &g1, 4, rows, 512, &s).unwrap();
    assert_eq!(down::<u16>(&g1, rows * 512), want_g);
}

#[test]
fn division_free_arithmetic_is_exact() {
    // Every BF16 value against every BF16 amax: the quantization's reciprocal-and-correction
    // quotients give the codes IEEE division gives; and every f32 in [1, 2^126): the router's
    // sigmoid reciprocal is IEEE 1 / d (the CPU reference divides in both).
    let s = Stream::new().unwrap();
    assert_eq!(ops::selfcheck_division_free(&s).unwrap(), [0, 0]);
}

#[test]
fn router_select_decides_special_values_as_the_reference() {
    // Rows of crafted logits and biases, against src/router.rs select_row: NaN at expert 0
    // (chosen first) and elsewhere (never chosen), -0 and +0, infinities, exact ties, and
    // fewer finite candidates than top_k (winners repeat as the reference's scan repeats
    // them). Weights compare bitwise, NaN with NaN.
    let s = Stream::new().unwrap();
    let mut rng = Rng::new(205);
    let base = rng.f32_vec(EXPERTS, 3.0);
    let mut rows: Vec<(Vec<f32>, Vec<f32>)> = Vec::new();
    let zero_bias = vec![0f32; EXPERTS];
    let mut l = base.clone();
    l[0] = f32::NAN;
    rows.push((l, rng.f32_vec(EXPERTS, 0.01)));
    let mut l = base.clone();
    l[5] = f32::NAN;
    l[100] = f32::NAN;
    l[287] = f32::NAN;
    rows.push((l, rng.f32_vec(EXPERTS, 0.01)));
    let mut b = zero_bias.clone();
    for (e, v) in b.iter_mut().enumerate() {
        if e % 3 == 0 {
            *v = -0.0;
        }
    }
    rows.push((vec![0.25f32; EXPERTS], b));
    let mut l = base.clone();
    l[7] = f32::INFINITY;
    l[9] = f32::INFINITY;
    l[33] = f32::NEG_INFINITY;
    rows.push((l, zero_bias.clone()));
    let mut b = vec![f32::NEG_INFINITY; EXPERTS];
    b[40] = 0.0;
    b[3] = 0.1;
    b[250] = -0.5;
    rows.push((base.clone(), b));
    let mut l = vec![f32::NEG_INFINITY; EXPERTS];
    l[31] = 0.0;
    l[32] = 0.0;
    rows.push((l, zero_bias.clone()));
    rows.push((base.clone(), rng.f32_vec(EXPERTS, 0.01)));
    let n = rows.len();
    let logits: Vec<f32> = rows.iter().flat_map(|r| r.0.iter().copied()).collect();
    let (dl, di, dwt) = (up(&logits), zeros(n * TOP_K * 4), zeros(n * TOP_K * 4));
    for (t, (l, b)) in rows.iter().enumerate() {
        // One bias per launch (the kernel takes one bias vector).
        let db = up(b);
        ops::router_select(&dl, &db, &di, &dwt, n, EXPERTS, TOP_K, ROUTED_SCALE, &s).unwrap();
        let ids: Vec<i32> = down(&di, n * TOP_K);
        let wts: Vec<f32> = down(&dwt, n * TOP_K);
        let r = glm53f_layers::router::select_row(l, b, TOP_K, ROUTED_SCALE);
        let got: Vec<u32> = ids[t * TOP_K..(t + 1) * TOP_K]
            .iter()
            .map(|&v| v as u32)
            .collect();
        assert_eq!(got, r.ids, "row {t} ids");
        for (k, (&g, &w)) in wts[t * TOP_K..(t + 1) * TOP_K]
            .iter()
            .zip(&r.weights)
            .enumerate()
        {
            assert!(
                g.to_bits() == w.to_bits() || (g.is_nan() && w.is_nan()),
                "row {t} weight {k}: {g} vs {w}"
            );
        }
    }
}
