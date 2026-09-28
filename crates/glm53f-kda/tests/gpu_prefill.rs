//! The chunked prefill on a GPU (feature `cuda`): against the serial chain at 1 to 16,384 rows,
//! across value blocks, batched against per request, segments that continue from a committed
//! state, the handoff to chain decode, error drift over 64K rows, the prefill and the chain
//! against exact arithmetic, and the oracle's real layers. Run with `--release`; skipped, with a
//! message, when no CUDA device is present.
#![cfg(feature = "cuda")]

use glm53f_kda::cpu::{self, LayerParams, Rounding};
use glm53f_kda::device::{self, DeviceBuffer, Error, Stream};
use glm53f_kda::kernel::{
    self, BatchMeta, Chain, Prefill, PrefillBatch, PrefillWorkspace, Request, RowView, StateOut,
    Weights,
};
use glm53f_kda::synth::{self, Rng};
use glm53f_kda::{bf16, channels, chunked, state_len, DK, DV, HEADS, WINDOW};

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

/// Projection rows on the device in the engine's layout ([q|k|v | f_a g_a | b]), dense a and g
/// rows. `period` random rows are tiled to `rows` (the kernels' cost and numerics do not care
/// that inputs repeat; it keeps host generation fast).
struct Data {
    heads: usize,
    p: DeviceBuffer,
    p_stride: usize,
    b_off: i64,
    a: DeviceBuffer,
    g: DeviceBuffer,
    /// The same rows on the host (only for small `rows`).
    host: Option<cpu::Rows>,
}

impl Data {
    fn new(heads: usize, rows: usize, period: usize, seed: u64) -> Data {
        let c = channels(heads);
        let p_stride = c + 256 + heads;
        let b_off = c + 256;
        let per = period.min(rows).max(1);
        let mut rng = Rng::new(seed);
        let mut p = vec![0u16; per * p_stride];
        for r in 0..per {
            for x in &mut p[r * p_stride..r * p_stride + c] {
                *x = bf16::from_f32(rng.uniform(-2.0, 2.0));
            }
            for x in &mut p[r * p_stride + b_off..r * p_stride + b_off + heads] {
                *x = bf16::from_f32(rng.uniform(-4.0, 4.0));
            }
        }
        let fill = |n: usize, rng: &mut Rng| -> Vec<u16> {
            (0..n)
                .map(|_| bf16::from_f32(rng.uniform(-4.0, 4.0)))
                .collect()
        };
        let a = fill(per * heads * DK, &mut rng);
        let g = fill(per * heads * DV, &mut rng);
        let tile = |block: &[u16], width: usize| -> DeviceBuffer {
            let buf = DeviceBuffer::alloc(rows.max(1) * width * 2).unwrap();
            let mut at = 0;
            while at < rows {
                let k = per.min(rows - at);
                buf.upload_at(at * width, &block[..k * width]).unwrap();
                at += k;
            }
            buf
        };
        let host = (rows <= per).then(|| cpu::Rows {
            heads,
            rows,
            qkv: (0..rows)
                .flat_map(|r| bf16::decode(&p[r * p_stride..r * p_stride + c]))
                .collect(),
            a: bf16::decode(&a[..rows * heads * DK]),
            b: (0..rows)
                .flat_map(|r| bf16::decode(&p[r * p_stride + b_off..r * p_stride + b_off + heads]))
                .collect(),
            gate: bf16::decode(&g[..rows * heads * DV]),
        });
        Data {
            heads,
            p: tile(&p, p_stride),
            p_stride,
            b_off: b_off as i64,
            a: tile(&a, heads * DK),
            g: tile(&g, heads * DV),
            host,
        }
    }

    fn p(&self, first: usize) -> RowView<'_> {
        RowView::new(&self.p, first * self.p_stride, self.p_stride)
    }
    fn a(&self, first: usize) -> RowView<'_> {
        RowView::new(&self.a, first * self.heads * DK, self.heads * DK)
    }
    fn g(&self, first: usize) -> RowView<'_> {
        RowView::new(&self.g, first * self.heads * DV, self.heads * DV)
    }
}

/// Outputs (as f32 values of the bf16 outputs), the state after the rows, and the conv window
/// after the rows.
struct Result3 {
    out: Vec<f32>,
    state: Vec<f32>,
    conv: Vec<f32>,
}

/// The chain over rows `first .. first + n` of `data`, from `state` and `conv` (host copies).
fn run_chain(
    stream: &Stream,
    w: &Weights,
    data: &Data,
    first: usize,
    n: usize,
    state: &[f32],
    conv: &[f32],
) -> Result3 {
    let h = data.heads;
    let conv_d = DeviceBuffer::from_slice(&bf16::encode(conv)).unwrap();
    let s = DeviceBuffer::from_slice(state).unwrap();
    let out = DeviceBuffer::zeroed(n.max(1) * h * DV * 2).unwrap();
    if n > 0 {
        Chain {
            weights: w,
            rows: n,
            p: data.p(first),
            b_off: data.b_off,
            a: data.a(first),
            g: data.g(first),
            conv: &conv_d,
            conv_offset: 0,
            state: &s,
            state_offset: 0,
            state_out: StateOut::InPlace,
            out: RowView::dense(&out, h * DV),
            saves: None,
        }
        .launch(stream)
        .unwrap();
        kernel::conv_shift(h, n, &conv_d, 0, data.p(first), stream).unwrap();
    }
    stream.synchronize().unwrap();
    Result3 {
        out: bf16::decode(&out.download::<u16>(n * h * DV).unwrap()),
        state: s.to_vec().unwrap(),
        conv: bf16::decode(&conv_d.to_vec::<u16>().unwrap()),
    }
}

/// Workspace bytes the tests give the prefill: 32 MiB, 240 rows per pass at 64 heads, so long
/// segments run many passes.
const WORKSPACE: usize = 32 << 20;

/// The prefill over rows `first .. first + n` of `data`.
#[allow(clippy::too_many_arguments)]
fn run_prefill(
    stream: &Stream,
    w: &Weights,
    data: &Data,
    first: usize,
    n: usize,
    state: &[f32],
    conv: &[f32],
    value_blocks: usize,
) -> Result3 {
    let ws = PrefillWorkspace::within(data.heads, 1, WORKSPACE).unwrap();
    run_prefill_ws(stream, w, data, first, n, state, conv, value_blocks, &ws)
}

#[allow(clippy::too_many_arguments)]
fn run_prefill_ws(
    stream: &Stream,
    w: &Weights,
    data: &Data,
    first: usize,
    n: usize,
    state: &[f32],
    conv: &[f32],
    value_blocks: usize,
    ws: &PrefillWorkspace,
) -> Result3 {
    let h = data.heads;
    let conv_d = DeviceBuffer::from_slice(&bf16::encode(conv)).unwrap();
    let s = DeviceBuffer::from_slice(state).unwrap();
    let out = DeviceBuffer::zeroed(n.max(1) * h * DV * 2).unwrap();
    Prefill {
        weights: w,
        rows: n,
        p: data.p(first),
        b_off: data.b_off,
        a: data.a(first),
        g: data.g(first),
        conv: &conv_d,
        conv_offset: 0,
        state: &s,
        state_offset: 0,
        state_out: StateOut::InPlace,
        out: RowView::dense(&out, h * DV),
        value_blocks,
        workspace: ws,
    }
    .launch(stream)
    .unwrap();
    stream.synchronize().unwrap();
    Result3 {
        out: bf16::decode(&out.download::<u16>(n * h * DV).unwrap()),
        state: s.to_vec().unwrap(),
        conv: bf16::decode(&conv_d.to_vec::<u16>().unwrap()),
    }
}

/// Error of `got` against `want`: max |diff|, max |diff| / max |want|, elements that differ.
fn err(got: &[f32], want: &[f32]) -> (f32, f32, usize) {
    assert_eq!(got.len(), want.len());
    let mut d = 0.0f32;
    let mut m = 0.0f32;
    let mut n = 0;
    for (g, w) in got.iter().zip(want) {
        d = d.max((g - w).abs());
        m = m.max(w.abs());
        n += (g.to_bits() != w.to_bits()) as usize;
    }
    (d, d / m.max(1e-30), n)
}

fn layer(heads: usize, seed: u64) -> (LayerParams, Weights) {
    let p = synth::layer(heads, seed);
    let w = Weights::upload(&p).unwrap();
    (p, w)
}

/// Bounds on the prefill against the chain, normwise: the state is f32 (the two orders of
/// summation differ by rounding); outputs are bfloat16, where an f32 difference occasionally
/// flips a rounding by an ulp.
const STATE_TOL: f32 = 1e-5;
const OUT_TOL: f32 = 1.0 / 128.0;
const OUT_DIFFER: f64 = 1e-3;

#[test]
fn prefill_matches_the_chain() {
    let Some(stream) = gpu() else { return };
    let h = HEADS;
    let (_, w) = layer(h, 800);
    for &t in &[1usize, 16, 17, 64, 1000, 1024, 4096, 16384] {
        let data = Data::new(h, t, 2048, 801);
        let conv = synth::conv_window(h, 802);
        let s0 = synth::state(h, 803, 0.5);
        let c = run_chain(&stream, &w, &data, 0, t, &s0, &conv);
        for nb in [1usize, 2, 4] {
            if t > 4096 && nb != 1 {
                continue;
            }
            let p = run_prefill(&stream, &w, &data, 0, t, &s0, &conv, nb);
            let (sd, sn, _) = err(&p.state, &c.state);
            let (od, on, ond) = err(&p.out, &c.out);
            eprintln!(
                "prefill vs chain, {t} rows, {nb} value block(s): state max |diff| {sd:.2e} ({sn:.2e} normwise); outputs max |diff| {od:.2e} ({on:.2e} normwise), {ond}/{} differ",
                c.out.len()
            );
            assert!(sn <= STATE_TOL, "{t} rows, nb {nb}: state {sn}");
            assert!(on <= OUT_TOL, "{t} rows, nb {nb}: outputs {on}");
            assert!(
                (ond as f64) <= OUT_DIFFER * c.out.len() as f64 + 1.0,
                "{t} rows, nb {nb}: {ond} outputs differ"
            );
            assert_eq!(p.conv, c.conv, "{t} rows: conv window after the segment");
        }
    }
}

/// The GPU prefill against its host model (`chunked::prefill`): the same algorithm, summed in a
/// different order.
#[test]
fn prefill_matches_its_host_model() {
    let Some(stream) = gpu() else { return };
    let h = 4;
    let (p, w) = layer(h, 810);
    for t in [5usize, 16, 70] {
        let data = Data::new(h, t, 2048, 811 + t as u64);
        let rows = data.host.clone().unwrap();
        let conv = synth::conv_window(h, 812);
        let s0 = synth::state(h, 813, 0.5);
        let g = run_prefill(&stream, &w, &data, 0, t, &s0, &conv, 1);
        let m = chunked::prefill(&p, &conv, &s0, &rows, Rounding::Fused);
        let (_, sn, _) = err(&g.state, &m.state);
        let (_, on, ond) = err(&g.out, &m.out);
        eprintln!("prefill vs host model, {t} rows: state {sn:.2e}, outputs {on:.2e} normwise, {ond} differ");
        assert!(sn <= 1e-6 && on <= OUT_TOL);
    }
}

#[test]
fn batched_equals_per_request_bitwise() {
    let Some(stream) = gpu() else { return };
    let h = 16;
    let c = channels(h);
    let (_, w) = layer(h, 820);
    let n_rows = [37usize, 0, 100, 16, 1];
    let slots = [3usize, 0, 4, 1, 2];
    let total: usize = n_rows.iter().sum();
    let data = Data::new(h, total, 2048, 821);
    let states: Vec<f32> = (0..5).flat_map(|i| synth::state(h, 830 + i, 0.5)).collect();
    let convs: Vec<f32> = (0..5)
        .flat_map(|i| synth::conv_window(h, 840 + i))
        .collect();
    let requests: Vec<Request> = (0..5)
        .map(|i| Request {
            rows: n_rows[i],
            conv_offset: slots[i] * WINDOW * c,
            state_offset: slots[i] * state_len(h),
        })
        .collect();
    let mut meta = BatchMeta::new(8).unwrap();
    meta.set(&requests, None).unwrap();
    let ws = PrefillWorkspace::within(h, 5, WORKSPACE).unwrap();
    for nb in [1usize, 2] {
        let conv_d = DeviceBuffer::from_slice(&bf16::encode(&convs)).unwrap();
        let s = DeviceBuffer::from_slice(&states).unwrap();
        let out = DeviceBuffer::zeroed(total * h * DV * 2).unwrap();
        PrefillBatch {
            weights: &w,
            p: data.p(0),
            b_off: data.b_off,
            a: data.a(0),
            g: data.g(0),
            conv: &conv_d,
            state: &s,
            state_out: StateOut::InPlace,
            out: RowView::dense(&out, h * DV),
            value_blocks: nb,
            workspace: &ws,
        }
        .launch(&meta, &stream)
        .unwrap();
        stream.synchronize().unwrap();
        let out_b = bf16::decode(&out.to_vec::<u16>().unwrap());
        let s_b: Vec<f32> = s.to_vec().unwrap();
        let conv_b = bf16::decode(&conv_d.to_vec::<u16>().unwrap());
        let mut row0 = 0;
        for i in 0..5 {
            let (sl, n) = (slots[i], n_rows[i]);
            let st = &states[sl * state_len(h)..(sl + 1) * state_len(h)];
            let cw = &convs[sl * WINDOW * c..(sl + 1) * WINDOW * c];
            let one = run_prefill(&stream, &w, &data, row0, n, st, cw, nb);
            assert_eq!(
                bits(&out_b[row0 * h * DV..(row0 + n) * h * DV]),
                bits(&one.out),
                "nb {nb}, request {i}: outputs"
            );
            assert_eq!(
                bits(&s_b[sl * state_len(h)..(sl + 1) * state_len(h)]),
                bits(&one.state),
                "nb {nb}, request {i}: state"
            );
            assert_eq!(
                &conv_b[sl * WINDOW * c..(sl + 1) * WINDOW * c],
                &one.conv[..],
                "nb {nb}, request {i}: conv window"
            );
            row0 += n;
        }
    }
}

/// A segment continues from a committed state and conv window: prefill of 600 then 400 rows
/// against one prefill of 1,000 and the chain; and the handoff to decode: prefill of N, then
/// eight chain steps, against the chain over N + 8.
#[test]
fn segments_and_the_handoff_to_decode() {
    let Some(stream) = gpu() else { return };
    let h = HEADS;
    let (_, w) = layer(h, 850);
    let data = Data::new(h, 1008, 2048, 851);
    let conv = synth::conv_window(h, 852);
    let s0 = synth::state(h, 853, 0.5);
    let chain = run_chain(&stream, &w, &data, 0, 1000, &s0, &conv);
    let first = run_prefill(&stream, &w, &data, 0, 600, &s0, &conv, 0);
    let second = run_prefill(&stream, &w, &data, 600, 400, &first.state, &first.conv, 0);
    let whole = run_prefill(&stream, &w, &data, 0, 1000, &s0, &conv, 0);
    let (_, sn, _) = err(&second.state, &chain.state);
    let (_, sw, _) = err(&second.state, &whole.state);
    let (_, on, ond) = err(&second.out, &chain.out[600 * h * DV..]);
    eprintln!("600 + 400 rows: state {sn:.2e} against the chain, {sw:.2e} against one prefill; outputs {on:.2e} normwise, {ond} differ");
    assert!(sn <= STATE_TOL && sw <= STATE_TOL && on <= OUT_TOL);
    assert_eq!(second.conv, chain.conv);
    // Handoff: prefill 1,000 rows, then chain decode of 8 single rows (each with its conv shift).
    let all = run_chain(&stream, &w, &data, 0, 1008, &s0, &conv);
    let (mut st, mut cv) = (whole.state.clone(), whole.conv.clone());
    let mut outs = Vec::new();
    for r in 1000..1008 {
        let step = run_chain(&stream, &w, &data, r, 1, &st, &cv);
        outs.extend_from_slice(&step.out);
        st = step.state;
        cv = step.conv;
    }
    let (_, dn, _) = err(&st, &all.state);
    let (_, don, dond) = err(&outs, &all.out[1000 * h * DV..]);
    eprintln!("prefill 1,000 + decode 8: state {dn:.2e}, decode outputs {don:.2e} normwise, {dond}/{} differ, against the chain over 1,008", outs.len());
    assert!(dn <= STATE_TOL && don <= OUT_TOL);
    assert_eq!(cv, all.conv);
}

/// Error over a long prompt: 64K rows in eight segments of 8K, the prefill and the chain each
/// continuing from their own state. The difference must not grow without bound. Twice: with
/// gates across the whole range, and with every channel decaying slowly (every multiplier at
/// least 0.9993, about e^-0.6 per segment), so that old rounding is forgotten mostly through
/// the delta rule itself.
#[test]
fn long_prompt_drift() {
    let Some(stream) = gpu() else { return };
    let h = 8;
    let seg = 8192;
    for slow in [false, true] {
        let mut p = synth::layer(h, 860);
        if slow {
            // exp(A_log) (a + dt_bias) <= -9 for a in [-4, 4]: sigmoid <= 1.2e-4.
            p.a_log.fill(0.0);
            p.dt_bias.fill(-13.0);
        }
        let w = Weights::upload(&p).unwrap();
        let s0 = synth::state(h, 861, 0.5);
        let conv0 = synth::conv_window(h, 862);
        let (mut sc, mut cc) = (s0.clone(), conv0.clone());
        let (mut sp, mut cp) = (s0, conv0);
        let mut errs = Vec::new();
        for k in 0..8 {
            let data = Data::new(h, seg, seg, 870 + k);
            let c = run_chain(&stream, &w, &data, 0, seg, &sc, &cc);
            let p = run_prefill(&stream, &w, &data, 0, seg, &sp, &cp, 0);
            let (_, sn, _) = err(&p.state, &c.state);
            let (_, on, ond) = err(&p.out, &c.out);
            eprintln!(
                "{} decay, rows {:>6}: state {sn:.2e} normwise, segment outputs {on:.2e} normwise, {ond}/{} differ",
                if slow { "slow" } else { "mixed" },
                (k as usize + 1) * seg,
                c.out.len()
            );
            errs.push(sn);
            assert_eq!(p.conv, c.conv);
            (sc, cc, sp, cp) = (c.state, c.conv, p.state, p.conv);
        }
        let max = errs.iter().cloned().fold(0.0f32, f32::max);
        assert!(max <= STATE_TOL, "state error {max} after up to 64K rows");
        assert!(errs[7] <= 4.0 * errs[0] + 1e-7, "error grows: {errs:?}");
    }
}

/// The chain and the prefill against exact arithmetic: both final states against an f64 replay
/// of the chain's own replay inputs (so all three see the same k, v, decay and beta), over 8K
/// rows with gates across the range and with slow decay. The prefill's error must stay within
/// 1e-6 and within twice the chain's. This pins the once-rounded decay products: with a plain
/// running product the prefill's slow-decay error was 3.9e-6, 2.5 times the chain's.
#[test]
fn accuracy_against_exact_arithmetic() {
    let Some(stream) = gpu() else { return };
    let (h, t) = (8, 8192);
    for slow in [false, true] {
        let mut p = synth::layer(h, 860);
        if slow {
            p.a_log.fill(0.0);
            p.dt_bias.fill(-13.0);
        }
        let w = Weights::upload(&p).unwrap();
        let data = Data::new(h, t, t, 895);
        let s0 = synth::state(h, 896, 0.5);
        let conv = synth::conv_window(h, 897);
        let conv_d = DeviceBuffer::from_slice(&bf16::encode(&conv)).unwrap();
        let s = DeviceBuffer::from_slice(&s0).unwrap();
        let out = DeviceBuffer::zeroed(t * h * DV * 2).unwrap();
        let saves = kernel::Saves::alloc(h, t).unwrap();
        Chain {
            weights: &w,
            rows: t,
            p: data.p(0),
            b_off: data.b_off,
            a: data.a(0),
            g: data.g(0),
            conv: &conv_d,
            conv_offset: 0,
            state: &s,
            state_offset: 0,
            state_out: StateOut::InPlace,
            out: RowView::dense(&out, h * DV),
            saves: Some(&saves),
        }
        .launch(&stream)
        .unwrap();
        stream.synchronize().unwrap();
        let chain: Vec<f32> = s.to_vec().unwrap();
        let pre = run_prefill(&stream, &w, &data, 0, t, &s0, &conv, 0).state;
        let sv = saves.download(t).unwrap();
        let mut exact: Vec<f64> = s0.iter().map(|&x| x as f64).collect();
        for r in 0..t {
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
        let rel = |got: &[f32]| {
            let (mut d, mut m) = (0.0f64, 0.0f64);
            for (g, e) in got.iter().zip(&exact) {
                d = d.max((*g as f64 - e).abs());
                m = m.max(e.abs());
            }
            d / m
        };
        let (ce, pe) = (rel(&chain), rel(&pre));
        eprintln!(
            "{} decay, {t} rows, state against exact arithmetic: chain {ce:.2e}, prefill {pe:.2e}",
            if slow { "slow" } else { "mixed" }
        );
        assert!(
            pe <= 1e-6 && pe <= 2.0 * ce + 1e-7,
            "prefill {pe}, chain {ce}"
        );
    }
}

#[test]
fn prefill_rejects_bad_arguments() {
    let Some(stream) = gpu() else { return };
    let h = 4;
    let data = Data::new(h, 8, 8, 880);
    let conv = DeviceBuffer::from_slice(&bf16::encode(&synth::conv_window(h, 881))).unwrap();
    let s = DeviceBuffer::from_slice(&synth::state(h, 882, 0.5)).unwrap();
    let out = DeviceBuffer::zeroed(8 * h * DV * 2).unwrap();
    let mut p = synth::layer(h, 883);
    let ws = PrefillWorkspace::within(h, 1, WORKSPACE).unwrap();
    let run = |w: &Weights, vb: usize, so: StateOut<'_>| {
        Prefill {
            weights: w,
            rows: 8,
            p: data.p(0),
            b_off: data.b_off,
            a: data.a(0),
            g: data.g(0),
            conv: &conv,
            conv_offset: 0,
            state: &s,
            state_offset: 0,
            state_out: so,
            out: RowView::dense(&out, h * DV),
            value_blocks: vb,
            workspace: &ws,
        }
        .launch(&stream)
    };
    let w = Weights::upload(&p).unwrap();
    assert!(run(&w, 0, StateOut::InPlace).is_ok());
    assert!(matches!(
        run(&w, 3, StateOut::InPlace),
        Err(Error::Invalid(_))
    ));
    assert!(matches!(run(&w, 1, StateOut::Skip), Err(Error::Invalid(_))));
    p.lower = -6.0;
    let w6 = Weights::upload(&p).unwrap();
    assert!(matches!(
        run(&w6, 1, StateOut::InPlace),
        Err(Error::Invalid(_))
    ));
    stream.synchronize().unwrap();
}

/// The workspace only sets how many rows each pass takes: every size gives the same bits, from one
/// 16-row chunk per pass to the whole segment in one.
#[test]
fn workspace_size_does_not_change_the_bits() {
    let Some(stream) = gpu() else { return };
    let h = 16;
    let (_, w) = layer(h, 890);
    let t = 300;
    let data = Data::new(h, t, 2048, 891);
    let conv = synth::conv_window(h, 892);
    let s0 = synth::state(h, 893, 0.5);
    let whole = run_prefill_ws(
        &stream,
        &w,
        &data,
        0,
        t,
        &s0,
        &conv,
        1,
        &PrefillWorkspace::new(h, 1, t).unwrap(),
    );
    for rows in [16usize, 48, 160] {
        let ws = PrefillWorkspace::new(h, 1, rows).unwrap();
        let r = run_prefill_ws(&stream, &w, &data, 0, t, &s0, &conv, 1, &ws);
        assert_eq!(
            bits(&r.out),
            bits(&whole.out),
            "{rows} rows per pass: outputs"
        );
        assert_eq!(
            bits(&r.state),
            bits(&whole.state),
            "{rows} rows per pass: state"
        );
        assert_eq!(r.conv, whole.conv);
    }
    // Too small a workspace is refused.
    let tiny = PrefillWorkspace {
        heads: h,
        batch: 1,
        rows_per_pass: 16,
        buf: DeviceBuffer::alloc(1024).unwrap(),
    };
    let conv_d = DeviceBuffer::from_slice(&bf16::encode(&conv)).unwrap();
    let s = DeviceBuffer::from_slice(&s0).unwrap();
    let out = DeviceBuffer::zeroed(t * h * DV * 2).unwrap();
    let r = Prefill {
        weights: &w,
        rows: t,
        p: data.p(0),
        b_off: data.b_off,
        a: data.a(0),
        g: data.g(0),
        conv: &conv_d,
        conv_offset: 0,
        state: &s,
        state_offset: 0,
        state_out: StateOut::InPlace,
        out: RowView::dense(&out, h * DV),
        value_blocks: 1,
        workspace: &tiny,
    }
    .launch(&stream);
    assert!(matches!(r, Err(Error::Invalid(_))), "{r:?}");
}

/// The kernels on the oracle's real layers (its weights, projections and states): the prefill
/// against the chain on the same inputs; the chain against the CPU reference; both against the
/// reference's f32 outputs, which the kernels' bfloat16 activations are compared to at bfloat16
/// scale. The projections are rounded to bfloat16, which is what the kernels take. Skipped
/// without the oracle's sets.
#[test]
fn kernels_on_oracle_layers() {
    use glm53f_kda::goldens::{self, Init, Set};
    let Some(stream) = gpu() else { return };
    let sets = goldens::discover(&goldens::default_root());
    let mut checked = 0;
    for dir in sets {
        let set = Set::load(&dir).unwrap();
        for lf in goldens::kda_layers(&set) {
            if !goldens::has_layer_inputs(&lf) {
                continue;
            }
            let init = Init::from_prefill(&set, lf.layer).unwrap();
            let case = goldens::layer_case(
                &set,
                &lf,
                &init,
                glm53f_kda::RMS_EPS,
                glm53f_kda::LOWER_BOUND,
            )
            .unwrap();
            let h = case.params.heads;
            let mut rows = case.rows.clone();
            for x in rows
                .qkv
                .iter_mut()
                .chain(&mut rows.a)
                .chain(&mut rows.b)
                .chain(&mut rows.gate)
            {
                *x = bf16::round(*x);
            }
            let t = rows.rows;
            // Stage the rows in the engine's layout.
            let c = channels(h);
            let p_stride = c + h;
            let mut p = vec![0.0f32; t * p_stride];
            for r in 0..t {
                p[r * p_stride..r * p_stride + c].copy_from_slice(&rows.qkv[r * c..(r + 1) * c]);
                p[r * p_stride + c..(r + 1) * p_stride]
                    .copy_from_slice(&rows.b[r * h..(r + 1) * h]);
            }
            let data = Data {
                heads: h,
                p: DeviceBuffer::from_slice(&bf16::encode(&p)).unwrap(),
                p_stride,
                b_off: c as i64,
                a: DeviceBuffer::from_slice(&bf16::encode(&rows.a)).unwrap(),
                g: DeviceBuffer::from_slice(&bf16::encode(&rows.gate)).unwrap(),
                host: None,
            };
            let w = Weights::upload(&case.params).unwrap();
            let conv = case
                .conv
                .iter()
                .map(|&x| bf16::round(x))
                .collect::<Vec<_>>();
            let chain = run_chain(&stream, &w, &data, 0, t, &case.state, &conv);
            let pre = run_prefill(&stream, &w, &data, 0, t, &case.state, &conv, 0);
            let cpu = cpu::chain(
                &case.params,
                &conv,
                &case.state,
                &rows,
                cpu::Rounding::Fused,
            );
            let (_, ps, _) = err(&pre.state, &chain.state);
            let (_, po, pod) = err(&pre.out, &chain.out);
            let (_, cs, _) = err(&chain.state, &cpu.state);
            let (_, co, cod) = err(&chain.out, &cpu.out);
            let (_, gs, _) = err(&chain.state, case.state_out.as_ref().unwrap());
            let (_, go, _) = err(&chain.out, &case.norm_out);
            eprintln!(
                "{} layer {} ({t} rows): prefill vs chain: state {ps:.1e}, output {po:.1e} ({pod} differ); chain vs CPU: state {cs:.1e}, output {co:.1e} ({cod} differ); chain vs the reference's f32: state {gs:.1e}, output {go:.1e}",
                set.name(),
                lf.layer
            );
            assert!(ps <= STATE_TOL && po <= OUT_TOL, "prefill vs chain");
            // Device and host exp differ by an ulp, which on real data can flip one bf16 conv output.
            assert!(cs <= 1e-4 && co <= OUT_TOL, "chain vs CPU");
            assert!(gs <= 2e-2 && go <= 2e-2, "chain vs the reference");
            checked += 1;
        }
    }
    if checked == 0 {
        eprintln!("skipped: no oracle sets with layer inputs");
    }
}
