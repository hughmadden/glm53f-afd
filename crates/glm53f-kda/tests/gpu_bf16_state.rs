//! The bf16-state kernels on a GPU (feature `cuda`; decision D8): `glm53f_kda_chain_batch_bf16state`,
//! `glm53f_kda_replay_batch_bf16state` and `glm53f_kda_prefill_batch_bf16state` keep the recurrent
//! state in bf16 between rows and compute in f32.
//!
//! - **Bitwise:** a window of R rows against R serial single-row calls (outputs, state, replay
//!   inputs); the replay of every prefix against the serial steps and against the CPU model;
//!   verify windows (no state written); batched launches against one launch per request; the
//!   prefill's workspace size; one row against the f32-state chain (the same outputs, the state
//!   rounded once).
//! - **Within tolerance:** the chain against the CPU model (the device's and the host's `expf`
//!   differ by an ulp); the prefill (rounded every 16-row chunk) against the chain (every row).
//! - **Drift, measured and reported:** 8K rows against exact arithmetic and against the f32
//!   state, with gates across the range and with every channel decaying slowly; 64K rows in eight
//!   segments, each continuing from its own state.
//!
//! Run with `--release`; skipped, with a message, when no CUDA device is present.
#![cfg(feature = "cuda")]

use glm53f_kda::cpu::{self, LayerParams, Rounding, Rows};
use glm53f_kda::device::{self, DeviceBuffer, Stream};
use glm53f_kda::kernel::{
    self, BatchMeta, ChainBatch, LayerSaves, PrefillBatch, PrefillWorkspace, Request, RowView,
    Saves, StateOut, Weights,
};
use glm53f_kda::synth;
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

/// A state rounded to bf16 (what a bf16 state holds).
fn b16(xs: &[f32]) -> Vec<f32> {
    xs.iter().map(|&x| bf16::round(x)).collect()
}

/// Projection rows on the device in the engine's layout ([q|k|v | b] per row), dense a and g
/// rows, from host rows (bf16-exact), or `period` random rows tiled to `rows`.
struct Data {
    heads: usize,
    p: DeviceBuffer,
    a: DeviceBuffer,
    g: DeviceBuffer,
}

impl Data {
    fn p_stride(&self) -> usize {
        channels(self.heads) + self.heads
    }

    fn from_rows(r: &Rows) -> Data {
        let (h, c) = (r.heads, channels(r.heads));
        let mut p = vec![0.0f32; r.rows.max(1) * (c + h)];
        for i in 0..r.rows {
            p[i * (c + h)..i * (c + h) + c].copy_from_slice(&r.qkv[i * c..(i + 1) * c]);
            p[i * (c + h) + c..(i + 1) * (c + h)].copy_from_slice(&r.b[i * h..(i + 1) * h]);
        }
        let up = |x: &[f32]| DeviceBuffer::from_slice(&bf16::encode(x)).unwrap();
        let pad = |x: &[f32]| {
            if x.is_empty() {
                vec![0.0; 8]
            } else {
                x.to_vec()
            }
        };
        Data {
            heads: h,
            p: up(&p),
            a: up(&pad(&r.a)),
            g: up(&pad(&r.gate)),
        }
    }

    /// `rows` rows: `period` synthetic rows repeated.
    fn tiled(heads: usize, rows: usize, period: usize, seed: u64) -> Data {
        let r = synth::rows(heads, period.min(rows).max(1), seed);
        let one = Data::from_rows(&r);
        let per = r.rows;
        let tile = |src: &DeviceBuffer, width: usize| -> DeviceBuffer {
            let host: Vec<u16> = src.download(per * width).unwrap();
            let buf = DeviceBuffer::alloc(rows.max(1) * width * 2).unwrap();
            let mut at = 0;
            while at < rows {
                let k = per.min(rows - at);
                buf.upload_at(at * width, &host[..k * width]).unwrap();
                at += k;
            }
            buf
        };
        let ps = one.p_stride();
        Data {
            heads,
            p: tile(&one.p, ps),
            a: tile(&one.a, heads * DK),
            g: tile(&one.g, heads * DV),
        }
    }

    fn p(&self, first: usize) -> RowView<'_> {
        RowView::new(&self.p, first * self.p_stride(), self.p_stride())
    }
    fn a(&self, first: usize) -> RowView<'_> {
        RowView::new(&self.a, first * self.heads * DK, self.heads * DK)
    }
    fn g(&self, first: usize) -> RowView<'_> {
        RowView::new(&self.g, first * self.heads * DV, self.heads * DV)
    }
}

/// How a run stores its state.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Store {
    F32,
    Bf16,
}

fn upload_state(s: &[f32], store: Store) -> DeviceBuffer {
    match store {
        Store::F32 => DeviceBuffer::from_slice(s).unwrap(),
        Store::Bf16 => DeviceBuffer::from_slice(&bf16::encode(s)).unwrap(),
    }
}

fn download_state(b: &DeviceBuffer, n: usize, store: Store) -> Vec<f32> {
    match store {
        Store::F32 => b.download(n).unwrap(),
        Store::Bf16 => bf16::decode(&b.download::<u16>(n).unwrap()),
    }
}

/// Outputs (f32 values of the bf16 outputs), the state after the rows (as f32 values), the conv
/// window after the rows (conv-shifted past all of them), and the replay inputs.
struct Out {
    out: Vec<f32>,
    state: Vec<f32>,
    conv: Vec<f32>,
    saves: cpu::Saves,
}

/// The chain over rows `first .. first + n` of `data` as a one-request batch, from host copies
/// of a state and conv window. `write_state` false runs it as a verify window (no state
/// written; the returned state is the input's).
#[allow(clippy::too_many_arguments)]
fn chain(
    stream: &Stream,
    w: &Weights,
    data: &Data,
    first: usize,
    n: usize,
    state: &[f32],
    conv: &[f32],
    store: Store,
    write_state: bool,
) -> Out {
    let h = data.heads;
    let conv_d = DeviceBuffer::from_slice(&bf16::encode(conv)).unwrap();
    let s = upload_state(state, store);
    let out = DeviceBuffer::zeroed(n.max(1) * h * DV * 2).unwrap();
    let saves = Saves::alloc(h, n.max(1)).unwrap();
    let mut meta = BatchMeta::new(1).unwrap();
    meta.set(
        &[Request {
            rows: n,
            conv_offset: 0,
            state_offset: 0,
        }],
        None,
    )
    .unwrap();
    let c = ChainBatch {
        weights: w,
        p: data.p(first),
        b_off: channels(h) as i64,
        a: data.a(first),
        g: data.g(first),
        conv: &conv_d,
        state: &s,
        state_out: if write_state {
            StateOut::InPlace
        } else {
            StateOut::Skip
        },
        out: RowView::dense(&out, h * DV),
        saves: Some(&saves),
    };
    match store {
        Store::F32 => c.launch(&meta, stream).unwrap(),
        Store::Bf16 => c.launch_bf16_state(&meta, stream).unwrap(),
    }
    if n > 0 {
        kernel::conv_shift(h, n, &conv_d, 0, data.p(first), stream).unwrap();
    }
    stream.synchronize().unwrap();
    Out {
        out: bf16::decode(&out.download::<u16>(n * h * DV).unwrap()),
        state: download_state(&s, state.len(), store),
        conv: bf16::decode(&conv_d.to_vec::<u16>().unwrap()),
        saves: saves.download(n).unwrap(),
    }
}

/// Workspace bytes for the prefill tests: 32 MiB (240 rows per pass at 64 heads).
const WORKSPACE: usize = 32 << 20;

/// The prefill over rows `first .. first + n` as a one-request batch.
#[allow(clippy::too_many_arguments)]
fn prefill(
    stream: &Stream,
    w: &Weights,
    data: &Data,
    first: usize,
    n: usize,
    state: &[f32],
    conv: &[f32],
    store: Store,
    ws: &PrefillWorkspace,
    value_blocks: usize,
) -> Out {
    let h = data.heads;
    let conv_d = DeviceBuffer::from_slice(&bf16::encode(conv)).unwrap();
    let s = upload_state(state, store);
    let out = DeviceBuffer::zeroed(n.max(1) * h * DV * 2).unwrap();
    let mut meta = BatchMeta::new(1).unwrap();
    meta.set(
        &[Request {
            rows: n,
            conv_offset: 0,
            state_offset: 0,
        }],
        None,
    )
    .unwrap();
    let pf = PrefillBatch {
        weights: w,
        p: data.p(first),
        b_off: channels(h) as i64,
        a: data.a(first),
        g: data.g(first),
        conv: &conv_d,
        state: &s,
        state_out: StateOut::InPlace,
        out: RowView::dense(&out, h * DV),
        value_blocks,
        workspace: ws,
    };
    match store {
        Store::F32 => pf.launch(&meta, stream).unwrap(),
        Store::Bf16 => pf.launch_bf16_state(&meta, stream).unwrap(),
    }
    stream.synchronize().unwrap();
    Out {
        out: bf16::decode(&out.download::<u16>(n * h * DV).unwrap()),
        state: download_state(&s, state.len(), store),
        conv: bf16::decode(&conv_d.to_vec::<u16>().unwrap()),
        saves: cpu::Saves::zeros(h, 0),
    }
}

/// Replay `keep` rows of one layer's saves from `state` with the bf16-state replay.
fn replay(stream: &Stream, h: usize, state: &[f32], saves: &cpu::Saves, keep: usize) -> Vec<f32> {
    let rows = saves.rows.max(1);
    let ls = LayerSaves::alloc(1, h, rows).unwrap();
    ls.k.upload(&saves.k).unwrap();
    ls.v.upload(&bf16::encode(&saves.v)).unwrap();
    ls.g.upload(&saves.g).unwrap();
    ls.b.upload(&saves.beta).unwrap();
    let mut meta = BatchMeta::new(1).unwrap();
    meta.set(
        &[Request {
            rows: saves.rows,
            conv_offset: 0,
            state_offset: 0,
        }],
        Some(&[keep][..]),
    )
    .unwrap();
    let s = upload_state(state, Store::Bf16);
    kernel::replay_batch_bf16_state(&meta, &s, &s, state_len(h), &ls, stream).unwrap();
    stream.synchronize().unwrap();
    download_state(&s, state.len(), Store::Bf16)
}

fn case(
    heads: usize,
    rows: usize,
    seed: u64,
) -> (LayerParams, Weights, Data, Rows, Vec<f32>, Vec<f32>) {
    let p = synth::layer(heads, seed);
    let w = Weights::upload(&p).unwrap();
    let r = synth::rows(heads, rows, seed + 1);
    let data = Data::from_rows(&r);
    let conv = synth::conv_window(heads, seed + 2);
    let s0 = b16(&synth::state(heads, seed + 3, 0.5));
    (p, w, data, r, conv, s0)
}

/// Max |got - want| / max |want|, and how many values differ.
fn err(got: &[f32], want: &[f32]) -> (f64, usize) {
    assert_eq!(got.len(), want.len());
    let (mut d, mut m, mut n) = (0f64, 0f64, 0usize);
    for (g, w) in got.iter().zip(want) {
        d = d.max((*g as f64 - *w as f64).abs());
        m = m.max((*w as f64).abs());
        n += (g.to_bits() != w.to_bits()) as usize;
    }
    (d / m.max(1e-30), n)
}

/// Relative RMS of `got - want`.
fn rel_rms(got: &[f32], want: &[f32]) -> f64 {
    let (mut e, mut w2) = (0f64, 0f64);
    for (g, w) in got.iter().zip(want) {
        e += (*g as f64 - *w as f64).powi(2);
        w2 += (*w as f64).powi(2);
    }
    (e / w2.max(1e-300)).sqrt()
}

/// A window of R rows gives the bits of R serial single-row calls: outputs, state and replay
/// inputs; a verify window (no state written) gives the same outputs; the replay of every prefix
/// gives the serial state and equals the CPU model bit for bit.
#[test]
fn windows_serial_steps_and_replays_are_bitwise() {
    let Some(stream) = gpu() else { return };
    let h = HEADS;
    for (rows, seed) in [(1usize, 900u64), (2, 910), (5, 920), (8, 930)] {
        let (_, w, data, _, conv, s0) = case(h, rows, seed);
        let win = chain(&stream, &w, &data, 0, rows, &s0, &conv, Store::Bf16, true);
        assert_eq!(
            bits(&win.state),
            bits(&b16(&win.state)),
            "the state is bf16"
        );
        let (mut st, mut cv) = (s0.clone(), conv.clone());
        let mut outs = Vec::new();
        let mut states = vec![s0.clone()];
        for r in 0..rows {
            let step = chain(&stream, &w, &data, r, 1, &st, &cv, Store::Bf16, true);
            outs.extend_from_slice(&step.out);
            assert_eq!(
                bits(&step.saves.k),
                bits(&win.saves.k[r * h * DK..(r + 1) * h * DK]),
                "R={rows} row {r}: replay k"
            );
            st = step.state;
            cv = step.conv;
            states.push(st.clone());
        }
        assert_eq!(bits(&win.out), bits(&outs), "R={rows}: outputs");
        assert_eq!(bits(&win.state), bits(&st), "R={rows}: state");
        assert_eq!(win.conv, cv, "R={rows}: conv window");
        let verify = chain(&stream, &w, &data, 0, rows, &s0, &conv, Store::Bf16, false);
        assert_eq!(
            bits(&verify.out),
            bits(&win.out),
            "R={rows}: verify outputs"
        );
        assert_eq!(
            bits(&verify.state),
            bits(&s0),
            "R={rows}: a verify writes no state"
        );
        for (keep, want) in states.iter().enumerate() {
            let got = replay(&stream, h, &s0, &win.saves, keep);
            assert_eq!(bits(&got), bits(want), "R={rows}: replay of {keep} rows");
            let host = cpu::replay_bf16_state(&s0, &win.saves, keep);
            assert_eq!(
                bits(&got),
                bits(&host),
                "R={rows}: replay of {keep} rows vs the CPU"
            );
        }
    }
}

/// With one row the bf16 state changes nothing but the stored state: the same outputs and
/// replay inputs as the f32-state chain from the same (bf16-exact) state, and the f32 chain's
/// state rounded once.
#[test]
fn one_row_is_the_f32_chain_rounded_once() {
    let Some(stream) = gpu() else { return };
    let (_, w, data, _, conv, s0) = case(HEADS, 1, 940);
    let a = chain(&stream, &w, &data, 0, 1, &s0, &conv, Store::F32, true);
    let b = chain(&stream, &w, &data, 0, 1, &s0, &conv, Store::Bf16, true);
    assert_eq!(bits(&a.out), bits(&b.out));
    assert_eq!(bits(&a.saves.k), bits(&b.saves.k));
    assert_eq!(bits(&b16(&a.state)), bits(&b.state));
}

/// The device chain against the CPU model of the bf16 state (`cpu::chain_bf16_state`): the
/// same bounds as the f32 chain against `cpu::chain` (the only difference is the device's and
/// the host's `expf`, an ulp or two, which the state can carry).
#[test]
fn chain_matches_the_cpu_model() {
    let Some(stream) = gpu() else { return };
    for (rows, seed) in [(1usize, 950u64), (8, 960), (40, 970)] {
        let (p, w, data, r, conv, s0) = case(HEADS, rows, seed);
        let dev = chain(&stream, &w, &data, 0, rows, &s0, &conv, Store::Bf16, true);
        let host = cpu::chain_bf16_state(&p, &conv, &s0, &r, Rounding::Fused);
        let (se, _) = err(&dev.state, &host.state);
        let (oe, od) = err(&dev.out, &host.out);
        eprintln!(
            "{rows} rows: state {se:.2e} normwise, outputs {oe:.2e} normwise ({od} of {} differ)",
            dev.out.len()
        );
        assert!(se <= 1e-2, "state {se}");
        assert!(oe <= 1.0 / 256.0 && (od as f64) <= 0.001 * dev.out.len() as f64 + 8.0);
    }
}

/// Batched launches equal one launch per request, bit for bit: the chain, the replay with a
/// different `keep` per request, and the prefill; states in shuffled slots of one buffer,
/// requests with zero rows included.
#[test]
fn batched_equals_per_request() {
    let Some(stream) = gpu() else { return };
    let h = 16;
    let p = synth::layer(h, 980);
    let w = Weights::upload(&p).unwrap();
    let lens = [3usize, 0, 8, 1, 5];
    let slots = [2usize, 4, 0, 3, 1];
    let total: usize = lens.iter().sum();
    let data = Data::tiled(h, total, total, 981);
    let sl = state_len(h);
    let cl = WINDOW * channels(h);
    let states: Vec<Vec<f32>> = (0..5)
        .map(|i| b16(&synth::state(h, 982 + i, 0.5)))
        .collect();
    let convs: Vec<Vec<f32>> = (0..5).map(|i| synth::conv_window(h, 990 + i)).collect();
    let mut pool = vec![0.0f32; 5 * sl];
    let mut conv_pool = vec![0.0f32; 5 * cl];
    for i in 0..5 {
        pool[slots[i] * sl..(slots[i] + 1) * sl].copy_from_slice(&states[i]);
        conv_pool[slots[i] * cl..(slots[i] + 1) * cl].copy_from_slice(&convs[i]);
    }
    let reqs: Vec<Request> = (0..5)
        .map(|i| Request {
            rows: lens[i],
            conv_offset: slots[i] * cl,
            state_offset: slots[i] * sl,
        })
        .collect();
    let firsts: Vec<usize> = lens
        .iter()
        .scan(0, |a, &n| {
            let f = *a;
            *a += n;
            Some(f)
        })
        .collect();
    // The chain, in place, with saves.
    let s = upload_state(&pool, Store::Bf16);
    let conv_d = DeviceBuffer::from_slice(&bf16::encode(&conv_pool)).unwrap();
    let out = DeviceBuffer::zeroed(total * h * DV * 2).unwrap();
    let saves = Saves::alloc(h, total).unwrap();
    let mut meta = BatchMeta::new(5).unwrap();
    meta.set(&reqs, None).unwrap();
    ChainBatch {
        weights: &w,
        p: data.p(0),
        b_off: channels(h) as i64,
        a: data.a(0),
        g: data.g(0),
        conv: &conv_d,
        state: &s,
        state_out: StateOut::InPlace,
        out: RowView::dense(&out, h * DV),
        saves: Some(&saves),
    }
    .launch_bf16_state(&meta, &stream)
    .unwrap();
    stream.synchronize().unwrap();
    let got_states = download_state(&s, pool.len(), Store::Bf16);
    let got_out = bf16::decode(&out.download::<u16>(total * h * DV).unwrap());
    let got_saves = saves.download(total).unwrap();
    for i in 0..5 {
        let one = chain(
            &stream,
            &w,
            &data,
            firsts[i],
            lens[i],
            &states[i],
            &convs[i],
            Store::Bf16,
            true,
        );
        let st = &got_states[slots[i] * sl..(slots[i] + 1) * sl];
        assert_eq!(bits(st), bits(&one.state), "request {i}: state");
        let o = &got_out[firsts[i] * h * DV..(firsts[i] + lens[i]) * h * DV];
        assert_eq!(bits(o), bits(&one.out), "request {i}: outputs");
    }
    // The replay of the batch with a different keep per request, into a fresh pool.
    let keeps = [2usize, 0, 5, 1, 5];
    let ls = LayerSaves::alloc(1, h, total).unwrap();
    ls.k.upload(&got_saves.k).unwrap();
    ls.v.upload(&bf16::encode(&got_saves.v)).unwrap();
    ls.g.upload(&got_saves.g).unwrap();
    ls.b.upload(&got_saves.beta).unwrap();
    meta.set(&reqs, Some(&keeps[..])).unwrap();
    let s2 = upload_state(&pool, Store::Bf16);
    kernel::replay_batch_bf16_state(&meta, &s2, &s2, 5 * sl, &ls, &stream).unwrap();
    stream.synchronize().unwrap();
    let replayed = download_state(&s2, pool.len(), Store::Bf16);
    for i in 0..5 {
        let first = firsts[i];
        let sv = cpu::Saves {
            heads: h,
            rows: lens[i],
            k: got_saves.k[first * h * DK..(first + lens[i]) * h * DK].to_vec(),
            v: got_saves.v[first * h * DV..(first + lens[i]) * h * DV].to_vec(),
            g: got_saves.g[first * h * DK..(first + lens[i]) * h * DK].to_vec(),
            beta: got_saves.beta[first * h..(first + lens[i]) * h].to_vec(),
        };
        let want = replay(&stream, h, &states[i], &sv, keeps[i]);
        let st = &replayed[slots[i] * sl..(slots[i] + 1) * sl];
        assert_eq!(bits(st), bits(&want), "request {i}: replay of {}", keeps[i]);
    }
    // The prefill (longer rows: 37, 0, 100, 16, 1).
    let plens = [37usize, 0, 100, 16, 1];
    let ptotal: usize = plens.iter().sum();
    let pdata = Data::tiled(h, ptotal, ptotal, 999);
    let pfirsts: Vec<usize> = plens
        .iter()
        .scan(0, |a, &n| {
            let f = *a;
            *a += n;
            Some(f)
        })
        .collect();
    let preqs: Vec<Request> = (0..5)
        .map(|i| Request {
            rows: plens[i],
            ..reqs[i]
        })
        .collect();
    meta.set(&preqs, None).unwrap();
    let s3 = upload_state(&pool, Store::Bf16);
    let conv3 = DeviceBuffer::from_slice(&bf16::encode(&conv_pool)).unwrap();
    let out3 = DeviceBuffer::zeroed(ptotal * h * DV * 2).unwrap();
    let ws = PrefillWorkspace::within(h, 5, WORKSPACE).unwrap();
    PrefillBatch {
        weights: &w,
        p: pdata.p(0),
        b_off: channels(h) as i64,
        a: pdata.a(0),
        g: pdata.g(0),
        conv: &conv3,
        state: &s3,
        state_out: StateOut::InPlace,
        out: RowView::dense(&out3, h * DV),
        value_blocks: 1,
        workspace: &ws,
    }
    .launch_bf16_state(&meta, &stream)
    .unwrap();
    stream.synchronize().unwrap();
    let pst = download_state(&s3, pool.len(), Store::Bf16);
    let pout = bf16::decode(&out3.download::<u16>(ptotal * h * DV).unwrap());
    let ws1 = PrefillWorkspace::within(h, 1, WORKSPACE).unwrap();
    for i in 0..5 {
        let one = prefill(
            &stream,
            &w,
            &pdata,
            pfirsts[i],
            plens[i],
            &states[i],
            &convs[i],
            Store::Bf16,
            &ws1,
            1,
        );
        let st = &pst[slots[i] * sl..(slots[i] + 1) * sl];
        assert_eq!(bits(st), bits(&one.state), "prefill request {i}: state");
        let o = &pout[pfirsts[i] * h * DV..(pfirsts[i] + plens[i]) * h * DV];
        assert_eq!(bits(o), bits(&one.out), "prefill request {i}: outputs");
    }
}

/// The prefill with a bf16 state: its results do not depend on the workspace size (bitwise), and
/// it agrees with the bf16-state chain (rounded every row, where the prefill rounds every 16
/// rows) to bf16 rounding.
#[test]
fn prefill_with_a_bf16_state() {
    let Some(stream) = gpu() else { return };
    let h = HEADS;
    let p = synth::layer(h, 1000);
    let w = Weights::upload(&p).unwrap();
    let t = 1000;
    let data = Data::tiled(h, t, 1000, 1001);
    let conv = synth::conv_window(h, 1002);
    let s0 = b16(&synth::state(h, 1003, 0.5));
    let one_pass = PrefillWorkspace::new(h, 1, t).unwrap();
    for vb in [2usize, 4, 1] {
        let whole = prefill(
            &stream,
            &w,
            &data,
            0,
            t,
            &s0,
            &conv,
            Store::Bf16,
            &one_pass,
            vb,
        );
        for rows in [16usize, 48, 160] {
            let ws = PrefillWorkspace::new(h, 1, rows).unwrap();
            let r = prefill(&stream, &w, &data, 0, t, &s0, &conv, Store::Bf16, &ws, vb);
            assert_eq!(
                bits(&r.out),
                bits(&whole.out),
                "{vb} value blocks, {rows} rows per pass: outputs"
            );
            assert_eq!(
                bits(&r.state),
                bits(&whole.state),
                "{vb} value blocks, {rows} rows per pass: state"
            );
            assert_eq!(r.conv, whole.conv);
        }
    }
    let whole = prefill(
        &stream,
        &w,
        &data,
        0,
        t,
        &s0,
        &conv,
        Store::Bf16,
        &one_pass,
        1,
    );
    let c = chain(&stream, &w, &data, 0, t, &s0, &conv, Store::Bf16, true);
    let f = chain(&stream, &w, &data, 0, t, &s0, &conv, Store::F32, true);
    let (se, _) = err(&whole.state, &c.state);
    let (oe, od) = err(&whole.out, &c.out);
    let (fe, _) = err(&whole.state, &f.state);
    let (ce, _) = err(&c.state, &f.state);
    eprintln!(
        "{t} rows, 64 heads: prefill vs chain (both bf16 states): state {se:.2e}, outputs {oe:.2e} normwise ({od} of {} differ); against the f32-state chain: prefill {fe:.2e}, chain {ce:.2e}",
        c.out.len()
    );
    assert_eq!(whole.conv, c.conv);
    assert!(se <= 2e-2 && oe <= 1.0 / 32.0, "prefill vs chain");
}

/// Exact state of 8K rows: an f64 replay of the replay inputs (the same for every variant: k,
/// v, the decays and beta do not depend on the state).
fn exact_state(s0: &[f32], sv: &cpu::Saves) -> Vec<f64> {
    let h = sv.heads;
    let mut exact: Vec<f64> = s0.iter().map(|&x| x as f64).collect();
    for r in 0..sv.rows {
        for hh in 0..h {
            let (kk, dv) = ((r * h + hh) * DK, (r * h + hh) * DV);
            let beta = sv.beta[r * h + hh] as f64;
            let st = &mut exact[hh * DV * DK..(hh + 1) * DV * DK];
            for (row, sr) in st.as_chunks_mut::<DK>().0.iter_mut().enumerate() {
                let mut kv = 0.0f64;
                for (c, x) in sr.iter_mut().enumerate() {
                    *x *= sv.g[kk + c] as f64;
                    kv += *x * sv.k[kk + c] as f64;
                }
                let delta = (sv.v[dv + row] as f64 - kv) * beta;
                for (c, x) in sr.iter_mut().enumerate() {
                    *x += sv.k[kk + c] as f64 * delta;
                }
            }
        }
    }
    exact
}

/// Max |got - exact| / max |exact|.
fn rel_exact(got: &[f32], exact: &[f64]) -> f64 {
    let (mut d, mut m) = (0f64, 0f64);
    for (g, e) in got.iter().zip(exact) {
        d = d.max((*g as f64 - e).abs());
        m = m.max(e.abs());
    }
    d / m
}

/// Relative RMS of `got - exact`.
fn rms_exact(got: &[f32], exact: &[f64]) -> f64 {
    let (mut e, mut w2) = (0f64, 0f64);
    for (g, x) in got.iter().zip(exact) {
        e += (*g as f64 - x).powi(2);
        w2 += x.powi(2);
    }
    (e / w2).sqrt()
}

/// Drift of the bf16 state over 8K rows, against exact arithmetic and the f32 state, with gates
/// across the range and with every channel decaying slowly (multipliers of at least 0.9993,
/// where a bf16 rounding is several times a step's decay). Reported; bounded loosely (the KL gate
/// decides).
#[test]
fn drift_over_8k_rows_against_exact_arithmetic() {
    let Some(stream) = gpu() else { return };
    let (h, t) = (8, 8192);
    for slow in [false, true] {
        let mut p = synth::layer(h, 860);
        if slow {
            p.a_log.fill(0.0);
            p.dt_bias.fill(-13.0);
        }
        let w = Weights::upload(&p).unwrap();
        let data = Data::tiled(h, t, t, 895);
        let s0 = b16(&synth::state(h, 896, 0.5));
        let conv = synth::conv_window(h, 897);
        let f = chain(&stream, &w, &data, 0, t, &s0, &conv, Store::F32, true);
        let b = chain(&stream, &w, &data, 0, t, &s0, &conv, Store::Bf16, true);
        let ws = PrefillWorkspace::within(h, 1, WORKSPACE).unwrap();
        let pf = prefill(&stream, &w, &data, 0, t, &s0, &conv, Store::F32, &ws, 1);
        let pb = prefill(&stream, &w, &data, 0, t, &s0, &conv, Store::Bf16, &ws, 1);
        assert_eq!(
            bits(&f.saves.k),
            bits(&b.saves.k),
            "the replay inputs do not see the state"
        );
        let exact = exact_state(&s0, &f.saves);
        let name = if slow {
            "slow decay"
        } else {
            "gates across the range"
        };
        for (what, st) in [
            ("chain, f32 state", &f.state),
            ("chain, bf16 state (every row)", &b.state),
            ("prefill, f32 state", &pf.state),
            ("prefill, bf16 state (every 16 rows)", &pb.state),
        ] {
            eprintln!(
                "{name}, {t} rows, state against exact arithmetic: {what}: max {:.2e}, rms {:.2e}",
                rel_exact(st, &exact),
                rms_exact(st, &exact)
            );
        }
        let (_, bd) = err(&b.out, &f.out);
        let (_, pd) = err(&pb.out, &f.out);
        eprintln!(
            "{name}, {t} rows, outputs against the f32-state chain: bf16 chain rms {:.2e} ({:.1}% of bf16 outputs differ), bf16 prefill rms {:.2e} ({:.1}%)",
            rel_rms(&b.out, &f.out),
            100.0 * bd as f64 / f.out.len() as f64,
            rel_rms(&pb.out, &f.out),
            100.0 * pd as f64 / f.out.len() as f64
        );
        for st in [&b.state, &pb.state] {
            assert!(st.iter().all(|x| x.is_finite()));
            assert!(rel_exact(st, &exact) < 0.25, "bf16 state far from exact");
        }
    }
}

/// 64K rows in eight segments of 8K, each continuing from its own state, bf16 against f32: the
/// difference must stay bounded (reported per segment).
#[test]
fn drift_over_64k_rows_stays_bounded() {
    let Some(stream) = gpu() else { return };
    let h = 8;
    let seg = 8192;
    for slow in [false, true] {
        let mut p = synth::layer(h, 860);
        if slow {
            p.a_log.fill(0.0);
            p.dt_bias.fill(-13.0);
        }
        let w = Weights::upload(&p).unwrap();
        let s0 = b16(&synth::state(h, 861, 0.5));
        let conv0 = synth::conv_window(h, 862);
        let (mut sf, mut cf) = (s0.clone(), conv0.clone());
        let (mut sb, mut cb) = (s0.clone(), conv0.clone());
        let mut errs = Vec::new();
        for k in 0..8u64 {
            let data = Data::tiled(h, seg, seg, 870 + k);
            let f = chain(&stream, &w, &data, 0, seg, &sf, &cf, Store::F32, true);
            let b = chain(&stream, &w, &data, 0, seg, &sb, &cb, Store::Bf16, true);
            let se = rel_rms(&b.state, &f.state);
            let (sm, _) = err(&b.state, &f.state);
            let oe = rel_rms(&b.out, &f.out);
            eprintln!(
                "{} decay, rows {:>6}: bf16 against f32 state: rms {se:.2e}, max {sm:.2e} normwise; segment outputs rms {oe:.2e}",
                if slow { "slow" } else { "mixed" },
                (k as usize + 1) * seg
            );
            errs.push(se);
            assert_eq!(b.conv, f.conv);
            (sf, cf, sb, cb) = (f.state, f.conv, b.state, b.conv);
        }
        let max = errs.iter().cloned().fold(0.0f64, f64::max);
        assert!(max.is_finite() && max < 0.25, "state error {max}");
        assert!(errs[7] <= 4.0 * errs[0] + 1e-3, "error grows: {errs:?}");
    }
}
