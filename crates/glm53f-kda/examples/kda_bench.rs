//! Throughput of the KDA kernels at the model's geometry (64 heads).
//!
//! ```text
//! cargo run --release -p glm53f-kda --features cuda --example kda_bench [-- --quick]
//! ```
//!
//! 1. Long windows (the prefill question): one request, one layer, 1K / 4K / 16K rows, then
//!    2 and 3 requests of 4K rows in one launch (more blocks than one request's 64).
//! 2. One decode or verify step's recurrent part over all 34 layers, each layer with its own
//!    states (so nothing sits in L2 that would not in a real step), for 1 and 8 requests of
//!    R = 1, 2, 4, 8 rows; then the commit (replay of every layer) and the conv shift.
//!
//! Inputs are synthetic (the kernels' cost does not depend on the values). Launches go through
//! the raw C ABI after one checked launch, so host-side checks are not timed; section 2 is 34
//! launches per step and includes their launch overhead (an engine would capture a graph).

use glm53f_kda::device::{self, DeviceBuffer, Event, Stream};
use glm53f_kda::ffi;
use glm53f_kda::kernel::{
    self, BatchMeta, ChainBatch, LayerSaves, Request, RowView, StateOut, Weights,
};
use glm53f_kda::{bf16, channels, cuda, state_len, synth, DK, DV, HEADS, LAYERS, WINDOW};

const H: usize = HEADS;

/// `n` rows in the engine's projection layout (`[q|k|v | f_a g_a | b]`), plus the gate rows
/// and the outputs.
struct Rows {
    p: DeviceBuffer,
    p_stride: usize,
    b_off: i64,
    a: DeviceBuffer,
    g: DeviceBuffer,
    out: DeviceBuffer,
}

fn rows(n: usize, seed: u64) -> Rows {
    let c = channels(H);
    let p_stride = c + 256 + H;
    let mut rng = synth::Rng::new(seed);
    // One block of random rows, tiled.
    let tile = 64.min(n);
    let mut fill = |width: usize, lo: f32, hi: f32| -> DeviceBuffer {
        let block: Vec<u16> = bf16::encode(&rng.fill(tile * width, lo, hi));
        let buf = DeviceBuffer::alloc(n * width * 2).unwrap();
        let mut at = 0;
        while at < n {
            let k = tile.min(n - at);
            buf.upload_at(at * width, &block[..k * width]).unwrap();
            at += k;
        }
        buf
    };
    Rows {
        p: fill(p_stride, -2.0, 2.0),
        p_stride,
        b_off: (c + 256) as i64,
        a: fill(H * DK, -4.0, 4.0),
        g: fill(H * DV, -4.0, 4.0),
        out: DeviceBuffer::alloc(n * H * DV * 2).unwrap(),
    }
}

/// Milliseconds per call of `f`, averaged over `iters` calls after one warm-up call.
fn time(stream: &Stream, iters: usize, mut f: impl FnMut()) -> f64 {
    f();
    stream.synchronize().unwrap();
    let (a, b) = (Event::new().unwrap(), Event::new().unwrap());
    a.record(stream).unwrap();
    for _ in 0..iters {
        f();
    }
    b.record(stream).unwrap();
    b.elapsed_ms_since(&a).unwrap() as f64 / iters as f64
}

fn ok(code: i32) {
    assert_eq!(code, 0, "launch failed: {}", cuda::error_string(code));
}

/// A checked launch (validates the geometry once), then the same launch through the raw ABI.
struct ChainLaunch<'a> {
    checked: ChainBatch<'a>,
    meta: &'a BatchMeta,
    k_off: usize,
    b_off_saves: usize,
    saves: Option<&'a LayerSaves>,
}

impl ChainLaunch<'_> {
    fn raw(&self, stream: &Stream) {
        let c = &self.checked;
        let w = c.weights;
        let state_out: *mut f32 = match c.state_out {
            StateOut::Skip => core::ptr::null_mut(),
            StateOut::InPlace => c.state.ptr(0),
            StateOut::To(b, _) => b.ptr(0),
        };
        let (ks, vs, gs, bs) = match self.saves {
            Some(s) => (
                s.k.ptr::<f32>(self.k_off),
                s.v.ptr::<u16>(self.k_off),
                s.g.ptr::<f32>(self.k_off),
                s.b.ptr::<f32>(self.b_off_saves),
            ),
            None => (
                core::ptr::null_mut(),
                core::ptr::null_mut(),
                core::ptr::null_mut(),
                core::ptr::null_mut(),
            ),
        };
        let (cu_rows, _, conv_off, state_off) = self.meta.device_arrays();
        // SAFETY: the same arguments passed the checked launch before timing.
        ok(unsafe {
            ffi::glm53f_kda_chain_batch(
                H as i32,
                self.meta.requests().len() as i32,
                cu_rows,
                c.p.buf.ptr(c.p.offset),
                c.p.stride as i64,
                c.b_off,
                c.a.buf.ptr(c.a.offset),
                c.a.stride as i64,
                c.g.buf.ptr(c.g.offset),
                c.g.stride as i64,
                c.conv.ptr(0),
                conv_off,
                w.conv_w.ptr(0),
                c.state.ptr(0),
                state_out,
                state_off,
                w.a_log.ptr(0),
                w.dt_bias.ptr(0),
                w.norm_w.ptr(0),
                w.eps,
                w.lower,
                c.out.buf.ptr(c.out.offset),
                c.out.stride as i64,
                ks,
                vs,
                gs,
                bs,
                stream.raw(),
            )
        });
    }
}

fn main() {
    let quick = std::env::args().any(|a| a == "--quick");
    if device::device_count() == 0 {
        eprintln!("no CUDA device");
        std::process::exit(1);
    }
    let sms = device::attribute(cuda::ATTR_SM_COUNT).unwrap();
    let cc = (
        device::attribute(cuda::ATTR_CC_MAJOR).unwrap(),
        device::attribute(cuda::ATTR_CC_MINOR).unwrap(),
    );
    let (free, total) = device::mem_info().unwrap();
    println!(
        "device: sm_{}{}, {sms} SMs, {:.1} of {:.1} GiB free; kernels built for {}",
        cc.0,
        cc.1,
        free as f64 / (1u64 << 30) as f64,
        total as f64 / (1u64 << 30) as f64,
        env!("GLM53F_KDA_CUDA_ARCH")
    );
    let stream = Stream::new().unwrap();
    let w = Weights::upload(&synth::layer(H, 1)).unwrap();
    let c = channels(H);
    let slot = state_len(H);

    // 1. Long windows.
    println!(
        "\n1. chain over long windows: {H} heads, one layer, no replay inputs, state in place"
    );
    println!(
        "{:>9} {:>8} {:>10} {:>10} {:>18}",
        "requests", "rows", "ms", "us/row", "x34 layers us/tok"
    );
    let windows: &[(usize, usize)] = if quick {
        &[(1, 1024), (2, 1024)]
    } else {
        &[(1, 1024), (1, 4096), (1, 16384), (2, 4096), (3, 4096)]
    };
    for &(batch, n) in windows {
        let total_rows = batch * n;
        let r = rows(total_rows, 2);
        let conv = DeviceBuffer::zeroed(batch * WINDOW * c * 2).unwrap();
        let state = DeviceBuffer::zeroed(batch * slot * 4).unwrap();
        let requests: Vec<Request> = (0..batch)
            .map(|b| Request {
                rows: n,
                conv_offset: b * WINDOW * c,
                state_offset: b * slot,
            })
            .collect();
        let mut meta = BatchMeta::new(batch).unwrap();
        meta.set(&requests, None).unwrap();
        let launch = ChainLaunch {
            checked: ChainBatch {
                weights: &w,
                p: RowView::new(&r.p, 0, r.p_stride),
                b_off: r.b_off,
                a: RowView::dense(&r.a, H * DK),
                g: RowView::dense(&r.g, H * DV),
                conv: &conv,
                state: &state,
                state_out: StateOut::InPlace,
                out: RowView::dense(&r.out, H * DV),
                saves: None,
            },
            meta: &meta,
            k_off: 0,
            b_off_saves: 0,
            saves: None,
        };
        launch.checked.launch(&meta, &stream).unwrap();
        let ms = time(&stream, if n >= 16384 { 2 } else { 3 }, || {
            launch.raw(&stream)
        });
        let per_row = ms * 1e3 / total_rows as f64;
        println!(
            "{batch:>9} {n:>8} {ms:>10.2} {per_row:>10.3} {:>18.1}",
            per_row * LAYERS as f64
        );
    }

    // The recurrence alone: the replay over a long window (no conv, norms, gates, read-out or
    // gated norm). A serial kernel with this mapping cannot go below it.
    let n = if quick { 1024 } else { 4096 };
    let saves = kernel::Saves::alloc(H, n).unwrap();
    let state = DeviceBuffer::zeroed(slot * 4).unwrap();
    kernel::replay(H, n, (&state, 0), (&state, 0), &saves, &stream).unwrap();
    let ms = time(&stream, 3, || {
        kernel::replay(H, n, (&state, 0), (&state, 0), &saves, &stream).unwrap()
    });
    let per_row = ms * 1e3 / n as f64;
    println!(
        "{:>9} {n:>8} {ms:>10.2} {per_row:>10.3} {:>18.1}   (replay: the recurrence alone)",
        1,
        per_row * LAYERS as f64
    );
    drop((saves, state));

    // 2. Decode and verify steps.
    println!("\n2. one step's recurrent part, {LAYERS} layers x {H} heads, each layer its own states (ms per step)");
    println!(
        "{:>9} {:>5} {:>14} {:>14} {:>15} {:>11} {:>16}",
        "requests",
        "rows",
        "verify chain",
        "decode chain",
        "commit replay",
        "conv shift",
        "state traffic"
    );
    let shapes: &[(usize, usize)] = if quick {
        &[(1, 1), (8, 8)]
    } else {
        &[
            (1, 1),
            (1, 2),
            (1, 4),
            (1, 8),
            (8, 1),
            (8, 2),
            (8, 4),
            (8, 8),
        ]
    };
    for &(batch, r_n) in shapes {
        let total_rows = batch * r_n;
        let rs: Vec<Rows> = (0..LAYERS)
            .map(|l| rows(total_rows, 10 + l as u64))
            .collect();
        // Layer l's states and windows at l * batch slots.
        let pool = DeviceBuffer::zeroed(LAYERS * batch * slot * 4).unwrap();
        let convs = DeviceBuffer::zeroed(LAYERS * batch * WINDOW * c * 2).unwrap();
        let saves = LayerSaves::alloc(LAYERS, H, total_rows).unwrap();
        let metas: Vec<BatchMeta> = (0..LAYERS)
            .map(|l| {
                let reqs: Vec<Request> = (0..batch)
                    .map(|b| Request {
                        rows: r_n,
                        conv_offset: (l * batch + b) * WINDOW * c,
                        state_offset: (l * batch + b) * slot,
                    })
                    .collect();
                let mut m = BatchMeta::new(batch).unwrap();
                m.set(&reqs, None).unwrap();
                m
            })
            .collect();
        let launches = |write_state: bool, with_saves: bool| -> Vec<ChainLaunch<'_>> {
            (0..LAYERS)
                .map(|l| {
                    let r = &rs[l];
                    let state_out = if write_state {
                        StateOut::InPlace
                    } else {
                        StateOut::Skip
                    };
                    let checked = ChainBatch {
                        weights: &w,
                        p: RowView::new(&r.p, 0, r.p_stride),
                        b_off: r.b_off,
                        a: RowView::dense(&r.a, H * DK),
                        g: RowView::dense(&r.g, H * DV),
                        conv: &convs,
                        state: &pool,
                        state_out,
                        out: RowView::dense(&r.out, H * DV),
                        saves: None,
                    };
                    checked.launch(&metas[l], &stream).unwrap();
                    ChainLaunch {
                        checked,
                        meta: &metas[l],
                        k_off: l * saves.kv_stride(),
                        b_off_saves: l * saves.b_stride(),
                        saves: if with_saves { Some(&saves) } else { None },
                    }
                })
                .collect()
        };
        let verify = launches(false, true);
        let decode = launches(true, false);
        let iters = if quick { 3 } else { 10 };
        let t_verify = time(&stream, iters, || {
            verify.iter().for_each(|l| l.raw(&stream))
        });
        let t_decode = time(&stream, iters, || {
            decode.iter().for_each(|l| l.raw(&stream))
        });
        // The commit: every layer of every request replays its R rows (the most a verify round
        // keeps), in place; then the conv windows advance.
        let requests: Vec<Request> = (0..batch)
            .map(|b| Request {
                rows: r_n,
                conv_offset: b * WINDOW * c,
                state_offset: b * slot,
            })
            .collect();
        let mut meta = BatchMeta::new(batch).unwrap();
        meta.set(&requests, None).unwrap();
        let stride = batch * slot;
        kernel::replay_batch(&meta, &pool, &pool, stride, &saves, &stream).unwrap();
        let t_replay = time(&stream, iters, || {
            kernel::replay_batch(&meta, &pool, &pool, stride, &saves, &stream).unwrap()
        });
        // All layers' rows in one buffer for the shift: layer l's rows at l * total_rows rows.
        let p_all = DeviceBuffer::zeroed(LAYERS * total_rows * rs[0].p_stride * 2).unwrap();
        let t_shift = time(&stream, iters, || {
            kernel::conv_shift_batch(
                H,
                LAYERS,
                &meta,
                &convs,
                batch * WINDOW * c,
                RowView::new(&p_all, 0, rs[0].p_stride),
                total_rows * rs[0].p_stride,
                &stream,
            )
            .unwrap()
        });
        // State bytes: the decode chain reads and writes every state once.
        let bytes = 2.0 * (LAYERS * batch * slot * 4) as f64;
        println!(
            "{batch:>9} {r_n:>5} {t_verify:>14.3} {t_decode:>14.3} {t_replay:>15.3} {t_shift:>11.3} {:>11.0} GB/s",
            bytes / (t_decode * 1e-3) / 1e9
        );
    }
}
