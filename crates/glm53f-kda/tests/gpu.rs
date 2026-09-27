//! The kernels on a GPU (feature `cuda`): the port against the source kernels (bitwise), against
//! the CPU reference, the replay, batch, stride and in-place invariants (bitwise), and the
//! device recurrence on golden activations. Skipped, with a message, when no CUDA device is
//! present.
#![cfg(feature = "cuda")]

mod common;

use glm53f_kda::cpu::{self, ConvRounding, LayerParams, Rows};
use glm53f_kda::device::{self, DeviceBuffer, Error, Stream};
use glm53f_kda::ffi;
use glm53f_kda::goldens::{self, Comparison, Init, Role, Set};
use glm53f_kda::kernel::{
    self, BatchMeta, Chain, ChainBatch, LayerSaves, Request, RowView, Saves, StateOut, Weights,
};
use glm53f_kda::synth::{self, Rng};
use glm53f_kda::{bf16, channels, state_len, DK, DV, HEADS, WINDOW};

fn gpu() -> Option<Stream> {
    if device::device_count() == 0 {
        eprintln!("skipped: no CUDA device");
        return None;
    }
    Some(Stream::new().expect("stream"))
}

fn bits(xs: &[f32]) -> Vec<u32> {
    xs.iter().map(|x| x.to_bits()).collect()
}

const JUNK: usize = 256;

/// Rows staged on the device the way the engine lays them out: q | k | v and the beta logits in
/// one projection row with other columns between (or the logits first), padded strides, and
/// junk everywhere else.
struct Staged {
    p: DeviceBuffer,
    p_off: usize,
    p_stride: usize,
    b_off: i64,
    a: DeviceBuffer,
    a_stride: usize,
    g: DeviceBuffer,
    g_stride: usize,
}

impl Staged {
    fn new(rows: &Rows, pad: usize, b_first: bool, seed: u64) -> Staged {
        let (h, n, c) = (rows.heads, rows.rows, channels(rows.heads));
        let mut rng = Rng::new(seed);
        let p_stride = pad + c + JUNK + h + pad;
        let (q_col, b_col) = if b_first {
            (pad + h + JUNK, pad)
        } else {
            (pad, pad + c + JUNK)
        };
        let mut p = rng.fill_bf16(n.max(1) * p_stride, -9.0, 9.0);
        let (a_stride, g_stride) = (h * DK + pad, h * DV + pad);
        let mut a = rng.fill_bf16(n.max(1) * a_stride, -9.0, 9.0);
        let mut g = rng.fill_bf16(n.max(1) * g_stride, -9.0, 9.0);
        for r in 0..n {
            p[r * p_stride + q_col..][..c].copy_from_slice(&rows.qkv[r * c..(r + 1) * c]);
            p[r * p_stride + b_col..][..h].copy_from_slice(&rows.b[r * h..(r + 1) * h]);
            a[r * a_stride..][..h * DK].copy_from_slice(&rows.a[r * h * DK..(r + 1) * h * DK]);
            g[r * g_stride..][..h * DV].copy_from_slice(&rows.gate[r * h * DV..(r + 1) * h * DV]);
        }
        let up = |x: &[f32]| DeviceBuffer::from_slice(&bf16::encode(x)).unwrap();
        Staged {
            p: up(&p),
            p_off: q_col,
            p_stride,
            b_off: b_col as i64 - q_col as i64,
            a: up(&a),
            a_stride,
            g: up(&g),
            g_stride,
        }
    }

    fn p(&self) -> RowView<'_> {
        RowView::new(&self.p, self.p_off, self.p_stride)
    }
    fn a(&self) -> RowView<'_> {
        RowView::new(&self.a, 0, self.a_stride)
    }
    fn g(&self) -> RowView<'_> {
        RowView::new(&self.g, 0, self.g_stride)
    }
}

struct Run {
    out: Vec<f32>,
    state: Vec<f32>,
    saves: cpu::Saves,
}

/// One chain on the device, from host inputs. `pad` > 0 uses padded strides and checks that
/// the padding of `out` is left alone.
fn chain_gpu(
    stream: &Stream,
    p: &LayerParams,
    conv: &[f32],
    state: &[f32],
    rows: &Rows,
    pad: usize,
    b_first: bool,
) -> Run {
    let h = p.heads;
    let w = Weights::upload(p).unwrap();
    let st = Staged::new(rows, pad, b_first, 99);
    let conv_d = DeviceBuffer::from_slice(&bf16::encode(conv)).unwrap();
    let s_in = DeviceBuffer::from_slice(state).unwrap();
    let s_out = DeviceBuffer::zeroed(state.len() * 4).unwrap();
    let out_stride = h * DV + pad;
    let junk: Vec<u16> = (0..rows.rows * out_stride)
        .map(|i| 0x3f00 + (i % 97) as u16)
        .collect();
    let out = DeviceBuffer::from_slice(&junk).unwrap();
    let saves = Saves::alloc(h, rows.rows).unwrap();
    Chain {
        weights: &w,
        rows: rows.rows,
        p: st.p(),
        b_off: st.b_off,
        a: st.a(),
        g: st.g(),
        conv: &conv_d,
        conv_offset: 0,
        state: &s_in,
        state_offset: 0,
        state_out: StateOut::To(&s_out, 0),
        out: RowView::new(&out, 0, out_stride),
        saves: Some(&saves),
    }
    .launch(stream)
    .unwrap();
    stream.synchronize().unwrap();
    let raw: Vec<u16> = out.to_vec().unwrap();
    let mut o = Vec::with_capacity(rows.rows * h * DV);
    for r in 0..rows.rows {
        o.extend(bf16::decode(&raw[r * out_stride..r * out_stride + h * DV]));
        assert_eq!(
            &raw[r * out_stride + h * DV..(r + 1) * out_stride],
            &junk[r * out_stride + h * DV..(r + 1) * out_stride],
            "padding of out row {r} was written"
        );
    }
    // The input state is read only.
    assert_eq!(bits(&s_in.to_vec::<f32>().unwrap()), bits(state));
    Run {
        out: o,
        state: s_out.to_vec().unwrap(),
        saves: saves.download(rows.rows).unwrap(),
    }
}

fn case(heads: usize, rows: usize, seed: u64) -> (LayerParams, Vec<f32>, Vec<f32>, Rows) {
    (
        synth::layer(heads, seed),
        synth::conv_window(heads, seed),
        synth::state(heads, seed, 0.5),
        synth::rows(heads, rows, seed),
    )
}

/// The port and the source kernels, on the same inputs, give the same bits: outputs, states,
/// replay inputs, replays of every prefix, and the all-layers replay.
#[test]
fn port_matches_the_source_kernels_bitwise() {
    let Some(stream) = gpu() else { return };
    let h = HEADS;
    let c = channels(h);
    for r in 1..=8usize {
        let (p, conv, s0, rows) = case(h, r, 100 + r as u64);
        let ours = chain_gpu(&stream, &p, &conv, &s0, &rows, 0, false);
        // The source kernel, the source's dense layout ([q|k|v | f_a g_a | b]).
        let st = Staged::new(&rows, 0, false, 99);
        assert_eq!(st.p_stride, c + JUNK + h);
        let w = Weights::upload(&p).unwrap();
        let conv_d = DeviceBuffer::from_slice(&bf16::encode(&conv)).unwrap();
        let s_in = DeviceBuffer::from_slice(&s0).unwrap();
        let s_out = DeviceBuffer::zeroed(s0.len() * 4).unwrap();
        let out = DeviceBuffer::zeroed(r * h * DV * 2).unwrap();
        let saves = Saves::alloc(h, r).unwrap();
        // SAFETY: every buffer is sized for H = 64 and r rows in the source's dense layout.
        let code = unsafe {
            ffi::parity::glm53f_kda_parity_chain(
                h as i32,
                st.p.ptr(0),
                st.p_stride as i32,
                st.b_off as i32,
                st.a.ptr(0),
                st.a_stride as i32,
                st.g.ptr(0),
                st.g_stride as i32,
                conv_d.ptr(0),
                w.conv_w.ptr(0),
                s_in.ptr(0),
                w.a_log.ptr(0),
                w.dt_bias.ptr(0),
                w.norm_w.ptr(0),
                w.eps,
                w.lower,
                r as i32,
                out.ptr(0),
                s_out.ptr(0),
                saves.k.ptr(0),
                saves.v.ptr(0),
                saves.g.ptr(0),
                saves.b.ptr(0),
                stream.raw(),
            )
        };
        assert_eq!(code, 0);
        stream.synchronize().unwrap();
        let src_out = bf16::decode(&out.to_vec::<u16>().unwrap());
        let src_saves = saves.download(r).unwrap();
        assert_eq!(bits(&ours.out), bits(&src_out), "R={r}: outputs");
        assert_eq!(
            bits(&ours.state),
            bits(&s_out.to_vec::<f32>().unwrap()),
            "R={r}: state"
        );
        assert_eq!(ours.saves, src_saves, "R={r}: replay inputs");
        // Replays of every prefix: the port's replay against the source's.
        let ours_saves = Saves::alloc(h, r).unwrap();
        upload_saves(&ours_saves, &ours.saves);
        for keep in 0..=r {
            let a = DeviceBuffer::zeroed(s0.len() * 4).unwrap();
            let b = DeviceBuffer::zeroed(s0.len() * 4).unwrap();
            kernel::replay(h, keep, (&s_in, 0), (&a, 0), &ours_saves, &stream).unwrap();
            // SAFETY: sized for H heads and r >= keep rows.
            let code = unsafe {
                ffi::parity::glm53f_kda_parity_replay(
                    h as i32,
                    s_in.ptr(0),
                    saves.k.ptr(0),
                    saves.v.ptr(0),
                    saves.g.ptr(0),
                    saves.b.ptr(0),
                    keep as i32,
                    b.ptr(0),
                    stream.raw(),
                )
            };
            assert_eq!(code, 0);
            stream.synchronize().unwrap();
            assert_eq!(
                bits(&a.to_vec::<f32>().unwrap()),
                bits(&b.to_vec::<f32>().unwrap()),
                "R={r} keep={keep}: replay"
            );
        }
    }
    // The all-layers replay against the source's, three layers.
    let (layers, r, keep) = (3, 6, 4);
    let ls = LayerSaves::alloc(layers, h, r).unwrap();
    let mut states = Vec::new();
    for l in 0..layers {
        let (p, conv, s0, rows) = case(h, r, 200 + l as u64);
        let run = chain_gpu(&stream, &p, &conv, &s0, &rows, 0, false);
        let s = Saves::alloc(h, r).unwrap();
        upload_saves(&s, &run.saves);
        ls.fill_layer(l, &s, r).unwrap();
        states.extend(s0);
    }
    let s_in = DeviceBuffer::from_slice(&states).unwrap();
    let (a, b) = (
        DeviceBuffer::zeroed(states.len() * 4).unwrap(),
        DeviceBuffer::zeroed(states.len() * 4).unwrap(),
    );
    kernel::replay_layers(keep, &s_in, &a, state_len(h), &ls, &stream).unwrap();
    // SAFETY: sized for 3 layers of H heads and r >= keep rows.
    let code = unsafe {
        ffi::parity::glm53f_kda_parity_replay_layers(
            h as i32,
            s_in.ptr(0),
            state_len(h) as i64,
            ls.k.ptr(0),
            ls.v.ptr(0),
            ls.g.ptr(0),
            ls.b.ptr(0),
            ls.kv_stride() as i64,
            ls.b_stride() as i64,
            layers as i32,
            keep as i32,
            b.ptr(0),
            stream.raw(),
        )
    };
    assert_eq!(code, 0);
    stream.synchronize().unwrap();
    assert_eq!(
        bits(&a.to_vec::<f32>().unwrap()),
        bits(&b.to_vec::<f32>().unwrap()),
        "replay_layers"
    );
}

fn upload_saves(d: &Saves, s: &cpu::Saves) {
    d.k.upload(&s.k).unwrap();
    d.v.upload(&bf16::encode(&s.v)).unwrap();
    d.g.upload(&s.g).unwrap();
    d.b.upload(&s.beta).unwrap();
}

/// The kernel against the CPU reference, R = 1..8 at the model's 64 heads. The state update
/// has no transcendental function, so the CPU replay of the kernel's own replay inputs must
/// give the kernel's state bit for bit; the rest differs only through `exp` (device vs host
/// libm, an ulp or two), within the stated tolerances.
#[test]
fn chain_matches_the_cpu_reference() {
    let Some(stream) = gpu() else { return };
    let h = HEADS;
    let mut worst = (0.0f32, 0.0f32, 0.0f32, 0.0f32, 0usize, 0u32);
    for r in 1..=8usize {
        let (p, conv, s0, rows) = case(h, r, 300 + r as u64);
        let g = chain_gpu(&stream, &p, &conv, &s0, &rows, 0, false);
        // Bitwise: the recurrence given the kernel's replay inputs.
        assert_eq!(
            bits(&cpu::replay(&s0, &g.saves, r)),
            bits(&g.state),
            "R={r}: CPU replay of the kernel's saves"
        );
        let c = cpu::chain(&p, &conv, &s0, &rows, ConvRounding::Fused);
        // Decay multipliers: three chained exponentials (exp(A_log), the sigmoid, the decay); an
        // ulp or two in each moves exp(g) by up to a few 1e-6 relative.
        let eg = max_rel(&g.saves.g, &c.saves.g);
        // v and beta are bfloat16: equal, or one ulp apart where exp moved a rounding.
        let (uv, nv) = ulps(&g.saves.v, &c.saves.v);
        let (ub, nb) = ulps(&g.saves.beta, &c.saves.beta);
        // Normalized k: f32 of bfloat16 conv outputs; an ulp flip in the conv output shows as
        // a relative change up to 2^-8 in that element.
        let ek = max_abs(&g.saves.k, &c.saves.k);
        let es = max_abs(&g.state, &c.state) / max_abs(&c.state, &vec![0.0; c.state.len()]);
        // Outputs: bfloat16; near-zero elements can differ by several ulps of their own tiny
        // magnitude, so the bound is normwise (against the largest output).
        let (uo, no) = ulps(&g.out, &c.out);
        let eo = max_abs(&g.out, &c.out) / max_abs(&c.out, &vec![0.0; c.out.len()]);
        eprintln!(
            "R={r}: exp(g) {eg:.1e} rel; v {nv}/{} differ (max {uv} ulp); beta {nb}/{} (max {ub}); k {ek:.1e} abs; state {es:.1e} rel; out {eo:.1e} rel, {no}/{} differ (max {uo} bf16 ulp)",
            c.saves.v.len(),
            c.saves.beta.len(),
            c.out.len()
        );
        assert!(eg <= 4e-6, "R={r}: decay multipliers {eg}");
        assert!(
            uv <= 1 && ub <= 1,
            "R={r}: v or beta more than one bf16 ulp apart"
        );
        assert!(nv * 1000 <= c.saves.v.len() && nb * 100 <= c.saves.beta.len().max(100));
        assert!(ek <= 1.0 / 256.0, "R={r}: normalized k {ek}");
        assert!(es <= 1e-3, "R={r}: state {es}");
        assert!(
            eo <= 1.0 / 256.0 && no * 1000 <= c.out.len(),
            "R={r}: outputs {eo}, {no} differ"
        );
        worst = (
            worst.0.max(eg),
            worst.1.max(ek),
            worst.2.max(es),
            worst.3.max(eo),
            worst.4.max(no),
            worst.5.max(uo),
        );
    }
    eprintln!("worst over R=1..8: exp(g) {:.1e} rel, k {:.1e} abs, state {:.1e} rel, out {:.1e} rel ({} elements differ, max {} bf16 ulp)", worst.0, worst.1, worst.2, worst.3, worst.4, worst.5);
}

fn max_abs(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f32::max)
}

fn max_rel(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs() / y.abs().max(1e-30))
        .fold(0.0, f32::max)
}

fn ulps(a: &[f32], b: &[f32]) -> (u32, usize) {
    let mut worst = 0;
    let mut n = 0;
    for (x, y) in a.iter().zip(b) {
        let d = bf16::ulp_distance(bf16::from_f32(*x), bf16::from_f32(*y));
        worst = worst.max(d);
        n += (d > 0) as usize;
    }
    (worst, n)
}

/// A replayed prefix of a window has the bits of a chain over that prefix alone, and a window
/// has the bits of serial single-row steps (with the device conv shift between them).
#[test]
fn prefixes_and_serial_steps_are_bitwise() {
    let Some(stream) = gpu() else { return };
    let h = HEADS;
    let r = 8;
    let (p, conv, s0, rows) = case(h, r, 400);
    let full = chain_gpu(&stream, &p, &conv, &s0, &rows, 0, false);
    let saves = Saves::alloc(h, r).unwrap();
    upload_saves(&saves, &full.saves);
    let s_in = DeviceBuffer::from_slice(&s0).unwrap();
    for keep in 0..=r {
        let replayed = DeviceBuffer::zeroed(s0.len() * 4).unwrap();
        kernel::replay(h, keep, (&s_in, 0), (&replayed, 0), &saves, &stream).unwrap();
        stream.synchronize().unwrap();
        let want = if keep == 0 {
            s0.clone()
        } else {
            chain_gpu(&stream, &p, &conv, &s0, &rows.slice(0, keep), 0, false).state
        };
        assert_eq!(
            bits(&replayed.to_vec::<f32>().unwrap()),
            bits(&want),
            "keep={keep}"
        );
        if keep > 0 {
            let short = chain_gpu(&stream, &p, &conv, &s0, &rows.slice(0, keep), 0, false);
            assert_eq!(
                bits(&short.out),
                bits(&full.out[..keep * h * DV]),
                "keep={keep}: outputs"
            );
        }
    }
    // Serial steps, advancing the conv window on the device.
    let w = Weights::upload(&p).unwrap();
    let st = Staged::new(&rows, 0, false, 99);
    let conv_d = DeviceBuffer::from_slice(&bf16::encode(&conv)).unwrap();
    let state = DeviceBuffer::from_slice(&s0).unwrap();
    let out = DeviceBuffer::zeroed(r * h * DV * 2).unwrap();
    for i in 0..r {
        let row_p = RowView::new(&st.p, st.p_off + i * st.p_stride, st.p_stride);
        Chain {
            weights: &w,
            rows: 1,
            p: row_p,
            b_off: st.b_off,
            a: RowView::new(&st.a, i * st.a_stride, st.a_stride),
            g: RowView::new(&st.g, i * st.g_stride, st.g_stride),
            conv: &conv_d,
            conv_offset: 0,
            state: &state,
            state_offset: 0,
            state_out: StateOut::InPlace,
            out: RowView::new(&out, i * h * DV, h * DV),
            saves: None,
        }
        .launch(&stream)
        .unwrap();
        kernel::conv_shift(h, 1, &conv_d, 0, row_p, &stream).unwrap();
    }
    stream.synchronize().unwrap();
    assert_eq!(
        bits(&bf16::decode(&out.to_vec::<u16>().unwrap())),
        bits(&full.out),
        "serial outputs"
    );
    assert_eq!(
        bits(&state.to_vec::<f32>().unwrap()),
        bits(&full.state),
        "serial state"
    );
    let conv_after: Vec<u16> = conv_d.to_vec().unwrap();
    assert_eq!(
        bf16::decode(&conv_after),
        cpu::conv_shift(&conv, &rows, r),
        "conv window after the steps"
    );
}

/// Padded strides, the beta logits before q | k | v (negative offset), and in-place states give
/// the bits of the dense layout.
#[test]
fn strides_offsets_and_in_place() {
    let Some(stream) = gpu() else { return };
    let h = 8;
    let r = 5;
    let (p, conv, s0, rows) = case(h, r, 500);
    let dense = chain_gpu(&stream, &p, &conv, &s0, &rows, 0, false);
    for (pad, b_first) in [(24, false), (0, true), (40, true)] {
        let run = chain_gpu(&stream, &p, &conv, &s0, &rows, pad, b_first);
        assert_eq!(
            bits(&run.out),
            bits(&dense.out),
            "pad {pad}, b first {b_first}"
        );
        assert_eq!(bits(&run.state), bits(&dense.state));
        assert_eq!(run.saves, dense.saves);
    }
    // In place, with the state at an offset inside a larger buffer.
    let w = Weights::upload(&p).unwrap();
    let st = Staged::new(&rows, 0, false, 99);
    let conv_d = DeviceBuffer::from_slice(&bf16::encode(&conv)).unwrap();
    let off = 1000;
    let mut pool = vec![7.0f32; off + s0.len() + 50];
    pool[off..off + s0.len()].copy_from_slice(&s0);
    let pool_d = DeviceBuffer::from_slice(&pool).unwrap();
    let out = DeviceBuffer::zeroed(r * h * DV * 2).unwrap();
    Chain {
        weights: &w,
        rows: r,
        p: st.p(),
        b_off: st.b_off,
        a: st.a(),
        g: st.g(),
        conv: &conv_d,
        conv_offset: 0,
        state: &pool_d,
        state_offset: off,
        state_out: StateOut::InPlace,
        out: RowView::dense(&out, h * DV),
        saves: None,
    }
    .launch(&stream)
    .unwrap();
    stream.synchronize().unwrap();
    let after: Vec<f32> = pool_d.to_vec().unwrap();
    assert_eq!(
        bits(&after[off..off + s0.len()]),
        bits(&dense.state),
        "in place"
    );
    assert!(
        after[..off]
            .iter()
            .chain(&after[off + s0.len()..])
            .all(|&x| x == 7.0),
        "outside the state untouched"
    );
    assert_eq!(
        bits(&bf16::decode(&out.to_vec::<u16>().unwrap())),
        bits(&dense.out)
    );
    // Replay in place.
    let saves = Saves::alloc(h, r).unwrap();
    upload_saves(&saves, &dense.saves);
    let s = DeviceBuffer::from_slice(&s0).unwrap();
    kernel::replay(h, 3, (&s, 0), (&s, 0), &saves, &stream).unwrap();
    stream.synchronize().unwrap();
    assert_eq!(
        bits(&s.to_vec::<f32>().unwrap()),
        bits(&cpu::replay(&s0, &dense.saves, 3))
    );
}

/// A batch of requests in one launch equals one launch per request, bit for bit: the chain,
/// the all-layers commit replay with a different keep per request, and the conv shift.
#[test]
fn batch_equals_single_requests() {
    let Some(stream) = gpu() else { return };
    let h = 16;
    let c = channels(h);
    let n_rows = [3usize, 1, 5, 0];
    let keep = [2usize, 1, 5, 0];
    // Request i's state lives in slot slots[i] of a pool; conv windows likewise.
    let slots = [2usize, 0, 3, 1];
    let layers = 2;
    let p = synth::layer(h, 600);
    let w = Weights::upload(&p).unwrap();
    let total: usize = n_rows.iter().sum();
    let rows_all = synth::rows(h, total, 600);
    let st = Staged::new(&rows_all, 8, false, 601);
    let pool_states: Vec<f32> = (0..4).flat_map(|i| synth::state(h, 610 + i, 0.5)).collect();
    let pool_convs: Vec<f32> = (0..4)
        .flat_map(|i| synth::conv_window(h, 620 + i))
        .collect();
    let requests: Vec<Request> = (0..4)
        .map(|i| Request {
            rows: n_rows[i],
            conv_offset: slots[i] * WINDOW * c,
            state_offset: slots[i] * state_len(h),
        })
        .collect();
    let mut meta = BatchMeta::new(8).unwrap();
    meta.set(&requests, None).unwrap();

    // Batched chain into a second pool, with saves.
    let conv_d = DeviceBuffer::from_slice(&bf16::encode(&pool_convs)).unwrap();
    let s_in = DeviceBuffer::from_slice(&pool_states).unwrap();
    let s_out = DeviceBuffer::zeroed(pool_states.len() * 4).unwrap();
    let out = DeviceBuffer::zeroed(total * h * DV * 2).unwrap();
    let saves = Saves::alloc(h, total).unwrap();
    ChainBatch {
        weights: &w,
        p: st.p(),
        b_off: st.b_off,
        a: st.a(),
        g: st.g(),
        conv: &conv_d,
        state: &s_in,
        state_out: StateOut::To(&s_out, 0),
        out: RowView::dense(&out, h * DV),
        saves: Some(&saves),
    }
    .launch(&meta, &stream)
    .unwrap();
    stream.synchronize().unwrap();
    let out_b = bf16::decode(&out.to_vec::<u16>().unwrap());
    let s_out_b: Vec<f32> = s_out.to_vec().unwrap();
    let saves_b = saves.download(total).unwrap();

    // One request at a time.
    let mut row0 = 0;
    for i in 0..4 {
        let (n, slot) = (n_rows[i], slots[i]);
        let s0 = &pool_states[slot * state_len(h)..(slot + 1) * state_len(h)];
        let got_state = &s_out_b[slot * state_len(h)..(slot + 1) * state_len(h)];
        if n == 0 {
            assert_eq!(
                bits(got_state),
                bits(s0),
                "request {i}: no rows, state copied"
            );
            continue;
        }
        let conv = &pool_convs[slot * WINDOW * c..(slot + 1) * WINDOW * c];
        let one = chain_gpu(&stream, &p, conv, s0, &rows_all.slice(row0, n), 0, false);
        assert_eq!(
            bits(&out_b[row0 * h * DV..(row0 + n) * h * DV]),
            bits(&one.out),
            "request {i}: outputs"
        );
        assert_eq!(bits(got_state), bits(&one.state), "request {i}: state");
        assert_eq!(
            bits(&saves_b.k[row0 * h * DK..(row0 + n) * h * DK]),
            bits(&one.saves.k),
            "request {i}: saves"
        );
        assert_eq!(
            bits(&saves_b.beta[row0 * h..(row0 + n) * h]),
            bits(&one.saves.beta)
        );
        row0 += n;
    }

    // The commit: every layer of every request, each keeping its own prefix, in place. Layer 1
    // reuses the saves of layer 0 on different states.
    meta.set(&requests, Some(&keep)).unwrap();
    let ls = LayerSaves::alloc(layers, h, total).unwrap();
    for l in 0..layers {
        ls.fill_layer(l, &saves, total).unwrap();
    }
    let layer_pool: Vec<f32> = (0..layers)
        .flat_map(|l| {
            if l == 0 {
                pool_states.clone()
            } else {
                pool_states.iter().map(|x| -x).collect()
            }
        })
        .collect();
    let stride = 4 * state_len(h);
    let st_d = DeviceBuffer::from_slice(&layer_pool).unwrap();
    kernel::replay_batch(&meta, &st_d, &st_d, stride, &ls, &stream).unwrap();
    stream.synchronize().unwrap();
    let got: Vec<f32> = st_d.to_vec().unwrap();
    let mut row0 = 0;
    for i in 0..4 {
        let slot = slots[i];
        let sub = cpu::Saves {
            heads: h,
            rows: n_rows[i],
            k: saves_b.k[row0 * h * DK..(row0 + n_rows[i]) * h * DK].to_vec(),
            v: saves_b.v[row0 * h * DV..(row0 + n_rows[i]) * h * DV].to_vec(),
            g: saves_b.g[row0 * h * DK..(row0 + n_rows[i]) * h * DK].to_vec(),
            beta: saves_b.beta[row0 * h..(row0 + n_rows[i]) * h].to_vec(),
        };
        for l in 0..layers {
            let at = l * stride + slot * state_len(h);
            let s0 = &layer_pool[at..at + state_len(h)];
            // The CPU replay is bitwise with the device's (no transcendental functions).
            assert_eq!(
                bits(&got[at..at + state_len(h)]),
                bits(&cpu::replay(s0, &sub, keep[i])),
                "request {i} layer {l}"
            );
        }
        row0 += n_rows[i];
    }

    // The conv shift of every layer of every request (layer 1: the same rows, halved windows).
    let conv_layers: Vec<f32> = [
        pool_convs.clone(),
        pool_convs.iter().map(|x| x * 0.5).collect(),
    ]
    .concat();
    let conv_l = DeviceBuffer::from_slice(&bf16::encode(&conv_layers)).unwrap();
    let p_host: Vec<u16> = st.p.to_vec().unwrap();
    let p_layers = DeviceBuffer::from_slice(&[p_host.clone(), p_host].concat()).unwrap();
    let p_layer_stride = st.p.len::<u16>();
    kernel::conv_shift_batch(
        h,
        layers,
        &meta,
        &conv_l,
        4 * WINDOW * c,
        RowView::new(&p_layers, st.p_off, st.p_stride),
        p_layer_stride,
        &stream,
    )
    .unwrap();
    stream.synchronize().unwrap();
    let got: Vec<f32> = bf16::decode(&conv_l.to_vec::<u16>().unwrap());
    let mut row0 = 0;
    for i in 0..4 {
        let slot = slots[i];
        let rows_i = rows_all.slice(row0, n_rows[i]);
        for l in 0..layers {
            let at = l * 4 * WINDOW * c + slot * WINDOW * c;
            let before = &conv_layers[at..at + WINDOW * c];
            assert_eq!(
                &got[at..at + WINDOW * c],
                &cpu::conv_shift(before, &rows_i, keep[i])[..],
                "request {i} layer {l}"
            );
        }
        row0 += n_rows[i];
    }
}

/// The checked wrappers refuse geometry that would reach outside a buffer.
#[test]
fn wrappers_reject_bad_geometry() {
    let Some(stream) = gpu() else { return };
    let h = 4;
    let (p, conv, s0, rows) = case(h, 3, 700);
    let w = Weights::upload(&p).unwrap();
    let st = Staged::new(&rows, 0, false, 99);
    let conv_d = DeviceBuffer::from_slice(&bf16::encode(&conv)).unwrap();
    let s = DeviceBuffer::from_slice(&s0).unwrap();
    let out = DeviceBuffer::zeroed(3 * h * DV * 2).unwrap();
    let small = Saves::alloc(h, 2).unwrap();
    let chain = |n: usize, saves: Option<&Saves>, state_offset: usize| {
        Chain {
            weights: &w,
            rows: n,
            p: st.p(),
            b_off: st.b_off,
            a: st.a(),
            g: st.g(),
            conv: &conv_d,
            conv_offset: 0,
            state: &s,
            state_offset,
            state_out: StateOut::Skip,
            out: RowView::dense(&out, h * DV),
            saves,
        }
        .launch(&stream)
    };
    assert!(chain(3, None, 0).is_ok());
    assert!(
        matches!(chain(4, None, 0), Err(Error::Invalid(_))),
        "more rows than staged"
    );
    assert!(
        matches!(chain(3, Some(&small), 0), Err(Error::Invalid(_))),
        "saves too small"
    );
    assert!(
        matches!(chain(3, None, 1), Err(Error::Invalid(_))),
        "state past the end"
    );
    assert!(
        matches!(chain(0, None, 0), Err(Error::Invalid(_))),
        "no rows"
    );
    let mut meta = BatchMeta::new(2).unwrap();
    let r = Request {
        rows: 1,
        conv_offset: 0,
        state_offset: 0,
    };
    assert!(meta.set(&[r, r, r], None).is_err(), "over capacity");
    assert!(meta.set(&[r], Some(&[2])).is_err(), "keep beyond rows");
    meta.set(&[r, r], None).unwrap();
    let two = DeviceBuffer::zeroed(2 * state_len(h) * 4).unwrap();
    let ls = LayerSaves::alloc(1, h, 2).unwrap();
    assert!(
        matches!(
            kernel::replay_batch(&meta, &two, &two, state_len(h), &ls, &stream),
            Err(Error::Invalid(_))
        ),
        "two requests on one state"
    );
    stream.synchronize().unwrap();
}

/// The device recurrence (the replay kernel) on golden activations: a synthetic pair in the
/// oracle's layout, and the oracle's own sets when they exist. The replay inputs are built from
/// the golden's k, v, g and beta ([`goldens::replay_inputs`]); the result must equal the CPU
/// replay bit for bit and the golden state within 1e-2 (the kernels keep v in bfloat16, the f32
/// goldens do not).
#[test]
fn device_recurrence_on_golden_activations() {
    let Some(stream) = gpu() else { return };
    let root = common::scratch("gpu-oracle-layout");
    common::write_oracle_pair(&root, 8, 70, 8);
    let mut dirs = goldens::discover(&root);
    dirs.extend(goldens::discover(&goldens::default_root()));
    let mut checked = 0;
    for dir in dirs {
        let set = Set::load(&dir).unwrap();
        for lf in goldens::kda_layers(&set) {
            if !lf.has(&[Role::K, Role::V, Role::LogDecay, Role::Beta, Role::StateOut]) {
                continue;
            }
            let init = Init::from_prefill(&set, lf.layer).unwrap();
            let goldens::ReplayCase {
                saves,
                state: s0,
                after: Some(after),
            } = goldens::replay_inputs(&set, &lf, &init).unwrap()
            else {
                continue;
            };
            let (h, rows) = (saves.heads, saves.rows);
            let d = Saves::alloc(h, rows).unwrap();
            upload_saves(&d, &saves);
            let state = DeviceBuffer::from_slice(&s0).unwrap();
            kernel::replay(h, rows, (&state, 0), (&state, 0), &d, &stream).unwrap();
            stream.synchronize().unwrap();
            let got: Vec<f32> = state.to_vec().unwrap();
            assert_eq!(
                bits(&got),
                bits(&cpu::replay(&s0, &saves, rows)),
                "{}: layer {}",
                set.name(),
                lf.layer
            );
            let c = Comparison::compare(
                &format!(
                    "{}: layer {} device state after {rows} rows",
                    set.name(),
                    lf.layer
                ),
                &got,
                &after,
                1e-2,
            );
            eprintln!("{c}");
            assert!(c.passes(), "{c}");
            checked += 1;
        }
    }
    assert!(checked >= 2, "the synthetic prefill and decode sets");
}
