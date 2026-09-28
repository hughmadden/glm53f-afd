//! `RemoteExperts`: the routed experts on four expert ranks over the wire (feature
//! `coordinator`).
//!
//! 1. **Against four mock ranks** (a GPU, no data), by every request and return path (the host
//!    paths, the fast paths, the frame fill, DMA copies only): the frames the ranks receive are
//!    the host encoder's byte for byte, with the wire client's host quantizer applied to the
//!    widened BF16 input and the call's routes, and the routed output is the BF16 rounding of
//!    the four planes added in rank order in f32.
//! 2. **Layers 0-4 on four real rank daemons** over TCP loopback: the EXL3 shares of layers 3
//!    and 4 cut by `glm53f-rank slice`, served by `glm53f-rank serve` on this GPU. Each MoE layer
//!    fed its golden input streams, with the golden routes, against the oracle's routed output
//!    and against `LocalFp8Experts` on the same inputs; then the chain (the 33-token prompt in
//!    one prefill, 8 decode steps through layers 0-4 and the head) against the golden logits.
//!    EXL3 4-bit experts are not the FP8 experts' bits: the rank crate measures a cosine of at
//!    least 0.990 per row against the reference (`glm53f-rank`, `tests/real_experts.rs`).
//!
//! 3. **Two lanes on the four rank daemons** (the serving path: the forward's own routing): the
//!    oracle's prompt, and two prompts batched across the lanes' cut, each as two lanes, as the
//!    same rows in two passes one after the other (bit for bit equal: over TCP the lanes take
//!    turns on the wire, one exchange in flight), and as one pass (the same tokens but on near
//!    ties, logits within the chain's bound); with the lane trace of each pass (a shared GPU, so
//!    it shows the schedule, not the overlap of separate machines).
//! 4. **Return paths and fast paths on the four rank daemons** (with their peer mesh): the
//!    row-sharded return against four planes, and the fast paths against the host paths, call
//!    by call on layers 3 and 4 (the oracle's MoE inputs) and through two-lane prefill of layers
//!    0-4 (the forward's own routing). The fast paths change no bit; row slices stay within the
//!    bound of the four-plane sum and pick the same tokens on every decided row; decode stays
//!    four planes, bit for bit.
//!
//! Tests 2 to 4 need, besides the variables of `tests/common/mod.rs`:
//!
//! - `GLM53F_RANK_BIN`: a `glm53f-rank` binary (`--features cuda` for GPU ranks);
//! - `GLM53F_RANK_DIRS`: the four rank directories, comma-separated in rank order, each cut
//!   with `glm53f-rank slice --layers 3-4` from the EXL3 checkpoint.
//!
//! It spawns the four daemons on loopback ports (`GLM53F_WIRE_ALLOW_LAN=1`, tests only) and
//! stops them when it ends. Anything missing makes it print why and pass. It needs about 16 GiB
//! of free GPU memory: the forward, the local experts, then the four ranks (about 2.1 GiB each).
//!
//! ```sh
//! GLM53F_CHECKPOINT_DIR=... GLM53F_EXPERTS_DIR=... GLM53F_RANK_BIN=... GLM53F_RANK_DIRS=a,b,c,d \
//!   cargo test --release -p glm53f-forward --features coordinator --test remote_experts \
//!   -- --nocapture --test-threads=1
//! ```
#![cfg(feature = "coordinator")]

mod common;

use std::collections::{HashMap, VecDeque};
use std::io::{BufRead, BufReader, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::path::PathBuf;
use std::process::{Child, ChildStdout, Command, Stdio};
use std::sync::{Arc, Mutex};

use common::*;
use glm53f_coordinator::wire::{Collected, ReturnPath, WireClient, WireConfig};
use glm53f_forward::device::{DeviceBuffer, Stream};
use glm53f_forward::experts::{ExpertBackend, ExpertCall, LocalFp8Experts, ZeroExperts};
use glm53f_forward::forward::{ForwardConfig, GlmForward, Tap, TapBuf, TapPoint};
use glm53f_forward::gemm::Fp8Act;
use glm53f_forward::remote::{FastPaths, RemoteExperts, WireTimes, RANKS};
use glm53f_forward::shape::{HC, HIDDEN, SAMPLE_VOCAB, TOP_K, VOCAB};
use glm53f_wire::frame::{
    Frame, HiddenRow, RequestFrame, ReturnFrame, ReturnRow, RouteEntry, RowDescriptor,
};
use glm53f_wire::l4::{StreamReceiver, StreamSender};
use glm53f_wire::row_shard::{row_partition, ExchangeDtype};
use glm53f_wire::WireNaive;

// ---- 1. Mock ranks ------------------------------------------------------------------------------

/// One request frame, read whole from a blocking stream.
fn read_frame(s: &mut TcpStream) -> Vec<u8> {
    use glm53f_wire::layout::hdr;
    let mut f = vec![0u8; glm53f_wire::HEADER_LEN];
    s.read_exact(&mut f).expect("header");
    let wb = u64::from_le_bytes(f[hdr::WIRE_BYTES..hdr::WIRE_BYTES + 8].try_into().unwrap());
    f.resize(wb as usize, 0);
    s.read_exact(&mut f[glm53f_wire::HEADER_LEN..])
        .expect("body");
    f
}

/// Four mock ranks on one listener (the client connects rank 0 to 3 in order). Each answers
/// `exchanges` requests, rank `r` returning `value(r, row, column)` rounded to BF16, and every
/// request is handed back when the server ends, decoded and as received, in (exchange, rank)
/// order.
fn mock_ranks(
    exchanges: usize,
    value: fn(usize, usize, usize) -> f32,
) -> (
    String,
    std::thread::JoinHandle<Vec<(RequestFrame, Vec<u8>)>>,
) {
    let listener = TcpListener::bind("127.0.0.1:0").expect("bind");
    let addr = listener.local_addr().unwrap().to_string();
    let server = std::thread::spawn(move || {
        let mut conns: Vec<TcpStream> = (0..RANKS).map(|_| listener.accept().unwrap().0).collect();
        let mut rx: Vec<StreamReceiver> = (0..RANKS)
            .map(|_| StreamReceiver::new(WireNaive::NONE))
            .collect();
        let mut tx: Vec<StreamSender> = (0..RANKS)
            .map(|_| StreamSender::new(WireNaive::NONE))
            .collect();
        let mut seen = Vec::new();
        for _ in 0..exchanges {
            for (r, s) in conns.iter_mut().enumerate() {
                let bytes = read_frame(s);
                let Frame::Request(req) = rx[r].accept(&bytes).expect("L4 accept") else {
                    panic!("rank {r}: expected a request frame");
                };
                let n = req.rows.len();
                let ret = ReturnFrame {
                    request_id: req.request_id,
                    placement_version: req.placement_version,
                    layer_id: req.layer_id,
                    executor_id: req.executor_id,
                    token_position: 0,
                    status: glm53f_wire::Status::Ok,
                    flags: glm53f_wire::FLAG_RETURN_REQUIRED,
                    route_count: TOP_K,
                    seq: 0,
                    rows: (0..n)
                        .map(|row| ReturnRow {
                            codes: (0..HIDDEN)
                                .map(|c| {
                                    glm53f_wire::bf16::f32_to_bf16(
                                        value(r, row, c),
                                        WireNaive::NONE,
                                    )
                                })
                                .collect(),
                        })
                        .collect(),
                };
                s.write_all(&tx[r].encode_return(&ret).expect("encode return"))
                    .expect("send return");
                seen.push((req, bytes));
            }
        }
        seen
    });
    (addr, server)
}

fn plane_value(rank: usize, row: usize, col: usize) -> f32 {
    let sign = if (col / 7 + rank).is_multiple_of(3) {
        -1.0
    } else {
        1.0
    };
    sign * ((rank + 1) as f32 * 0.37 + row as f32 * 0.11 + (col % 113) as f32 * 0.013)
}

/// BF16 input rows with a spread of magnitudes, an all-zero block of 32 and values past E4M3's
/// range after scaling.
fn input_rows(rows: usize) -> Vec<u16> {
    let v: Vec<f32> = (0..rows * HIDDEN)
        .map(|i| {
            let (r, c) = (i / HIDDEN, i % HIDDEN);
            if (64..96).contains(&c) {
                return 0.0;
            }
            let mag = 2f32.powi(((c / 32) % 9) as i32 - 4 + r as i32 % 3);
            let s = if (c * 7 + r) % 5 < 2 { -1.0 } else { 1.0 };
            s * mag * (0.05 + ((c * 31 + r * 17) % 101) as f32 / 101.0)
        })
        .collect();
    narrow(&v)
}

/// The frame the host encoder writes for `req` as received (its request id, executor and
/// sequence), with the wire rows `rows` and the routes `ids`, `w`: what every path must send.
fn host_frame(req: &RequestFrame, rows: Vec<HiddenRow>, ids: &[i32], w: &[f32]) -> Vec<u8> {
    let n = rows.len();
    let f = RequestFrame {
        request_id: req.request_id,
        placement_version: 1,
        layer_id: req.layer_id,
        executor_id: req.executor_id,
        source_kind: glm53f_wire::SourceKind::Decode,
        token_position: 0,
        flags: req.flags,
        seq: req.seq,
        rows: (0..n)
            .map(|t| RowDescriptor {
                row_id: t as u64,
                source_kind: glm53f_wire::SourceKind::Decode,
                source_request_id: req.request_id,
                token_position: t as u64,
                route_offset: (t * TOP_K) as u32,
                route_count: TOP_K as u32,
            })
            .collect(),
        routes: (0..ids.len())
            .map(|i| RouteEntry {
                row_index: (i / TOP_K) as u32,
                expert_id: ids[i] as u32,
                gate_weight: w[i],
            })
            .collect(),
        hidden_rows: rows,
    };
    glm53f_wire::frame::encode_request_seq(&f, req.seq, WireNaive::NONE).unwrap()
}

/// Every request and return form against four mock ranks: the host paths; the fast paths as
/// they default (the DMA copies into the request body, the in-place sum up to 64 rows and DMA
/// copies of the returns above); the frame fill up to 64 rows; and the DMA copies at every size.
/// What the ranks receive is the host encoder's frame byte for byte (the wire rows are the host
/// quantizer's), and the routed output the BF16 rounding of the four planes added in rank order
/// in f32, bit for bit.
#[test]
fn remote_experts_against_four_mock_ranks() {
    if !gpu_with(0.5) {
        return;
    }
    let passes = [1usize, 5, 16, 80];
    let max = 80;
    let variants = [
        ("host paths", FastPaths::OFF),
        ("fast paths", FastPaths::ON),
        (
            "frame fill up to 64 rows",
            FastPaths {
                fill_rows: 64,
                ..FastPaths::ON
            },
        ),
        (
            "DMA copies only",
            FastPaths {
                mapped_rows: 0,
                ..FastPaths::ON
            },
        ),
    ];
    let mut outputs: Vec<Vec<Vec<u16>>> = Vec::new();
    for (name, fast) in variants {
        let (addr, server) = mock_ranks(passes.len(), plane_value);
        let mut remote =
            RemoteExperts::connect_with(&vec![addr; RANKS], max, WireConfig::glm53_flash(), fast)
                .expect("connect");
        assert_eq!(
            remote.fast_paths(),
            fast,
            "{name}: every buffer page-locked"
        );
        let times = remote.times();
        let stream = Stream::new().unwrap();
        let mut sent: Vec<(Vec<u16>, Vec<i32>, Vec<f32>)> = Vec::new();
        let mut outs = Vec::new();
        for (k, &rows) in passes.iter().enumerate() {
            let x = input_rows(rows);
            // On the stream the backend runs on: a synchronous copy from pageable memory can
            // return before its data lands, and the stream does not wait for the legacy stream.
            let xd = DeviceBuffer::alloc(rows * HIDDEN * 2).unwrap();
            xd.upload_async(&stream, 0, &x).unwrap();
            let out = DeviceBuffer::alloc(rows * HIDDEN * 2).unwrap();
            // Distinct experts per row (37 is prime to 288), past 255 included.
            let ids: Vec<i32> = (0..rows * TOP_K)
                .map(|i| ((i * 37 + 250) % 288) as i32)
                .collect();
            let w: Vec<f32> = (0..rows * TOP_K)
                .map(|i| 2.5 * (1 + i % TOP_K) as f32 / 36.0)
                .collect();
            // The routes on the device too, as the forward's router leaves them (the frame fill
            // reads them there).
            let (di, dw) = (
                DeviceBuffer::alloc(rows * TOP_K * 4).unwrap(),
                DeviceBuffer::alloc(rows * TOP_K * 4).unwrap(),
            );
            di.upload_async(&stream, 0, &ids).unwrap();
            dw.upload_async(&stream, 0, &w).unwrap();
            let call = ExpertCall {
                layer: 3 + k,
                rows,
                x: xd.ptr(0),
                x_q: core::ptr::null(),
                x_scales: core::ptr::null(),
                ids: di.ptr(0),
                weights: dw.ptr(0),
                host_ids: &ids,
                host_weights: &w,
                out: out.ptr(0),
            };
            remote.submit(&call, &stream).expect("submit");
            remote.finish(&call, &stream).expect("finish");
            stream.synchronize().unwrap();
            let got: Vec<u16> = out.download(rows * HIDDEN).unwrap();
            let want: Vec<f32> = (0..rows * HIDDEN)
                .map(|i| {
                    let mut s = 0f32;
                    for r in 0..RANKS {
                        s += glm53f_layers::bf16::round(plane_value(r, i / HIDDEN, i % HIDDEN));
                    }
                    s
                })
                .collect();
            assert_eq!(got, narrow(&want), "{name}, {rows} rows: the rank sum");
            outs.push(got);
            sent.push((x, ids, w));
        }
        drop(remote);
        let frames = server.join().expect("mock ranks");
        assert_eq!(frames.len(), passes.len() * RANKS);
        for (f, (req, bytes)) in frames.iter().enumerate() {
            let (k, r) = (f / RANKS, f % RANKS);
            let (x, ids, w) = &sent[k];
            assert_eq!(
                (req.layer_id, req.executor_id, req.rows.len(), req.flags),
                ((3 + k) as u32, r as u64, passes[k], 0)
            );
            let rows = glm53f_coordinator::wire::quantize_hidden_batched(&widen(x)).unwrap();
            assert!(
                *bytes == host_frame(req, rows, ids, w),
                "{name}, pass {k} ({} rows), rank {r}: not the host encoder's frame",
                passes[k]
            );
        }
        let t = times.lock().unwrap();
        assert_eq!(t.len(), passes.len());
        assert!(t.values().all(|x| x.count == 1 && !x.row_sharded));
        eprintln!(
            "RemoteExperts against four mock ranks, {name}: {} passes of {passes:?} rows, the frames the host encoder's byte for byte, the rank sums exact",
            passes.len()
        );
        outputs.push(outs);
    }
    assert!(outputs.iter().all(|o| *o == outputs[0]));
}

// ---- 2. Real ranks --------------------------------------------------------------------------------

const PROMPT: usize = 33;
const STEPS: usize = 8;

/// The rank daemons, stopped when dropped.
struct Daemons {
    children: Vec<Child>,
    _out: Vec<BufReader<ChildStdout>>,
}

impl Drop for Daemons {
    fn drop(&mut self) {
        for c in &mut self.children {
            let _ = c.kill();
            let _ = c.wait();
        }
    }
}

/// `GLM53F_RANK_BIN` and the four `GLM53F_RANK_DIRS`, or None (printed).
fn rank_setup() -> Option<(PathBuf, Vec<PathBuf>)> {
    let Some(bin) = env_dir(&["GLM53F_RANK_BIN"]).filter(|p| p.is_file()) else {
        eprintln!("skip: GLM53F_RANK_BIN does not name a glm53f-rank binary");
        return None;
    };
    let dirs: Vec<PathBuf> = std::env::var("GLM53F_RANK_DIRS")
        .unwrap_or_default()
        .split(',')
        .filter(|s| !s.is_empty())
        .map(PathBuf::from)
        .collect();
    if dirs.len() != RANKS || !dirs.iter().all(|d| d.join("manifest.txt").is_file()) {
        eprintln!("skip: GLM53F_RANK_DIRS must name the four rank directories (with manifest.txt)");
        return None;
    }
    Some((bin, dirs))
}

/// Start the four daemons on loopback, with their peer mesh (for the row-sharded return);
/// returns them and their addresses in rank order.
fn spawn_ranks(bin: &PathBuf, dirs: &[PathBuf]) -> (Daemons, Vec<String>) {
    let mut d = Daemons {
        children: Vec::new(),
        _out: Vec::new(),
    };
    // Four free loopback ports for the mesh (bound, noted, released).
    let peers: Vec<String> = (0..RANKS)
        .map(|_| {
            TcpListener::bind("127.0.0.1:0")
                .unwrap()
                .local_addr()
                .unwrap()
                .to_string()
        })
        .collect();
    for (r, dir) in dirs.iter().enumerate() {
        let child = Command::new(bin)
            .args(["serve", "--rank", &r.to_string(), "--dir"])
            .arg(dir)
            .args(["--listen", "127.0.0.1:0", "--allow-partial"])
            .args(["--peers", &peers.join(",")])
            .env("GLM53F_WIRE_ALLOW_LAN", "1")
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .spawn()
            .expect("spawn glm53f-rank");
        d.children.push(child);
    }
    let mut addrs = Vec::new();
    for r in 0..RANKS {
        let mut out = BufReader::new(d.children[r].stdout.take().unwrap());
        let mut line = String::new();
        let addr = loop {
            line.clear();
            if out.read_line(&mut line).unwrap_or(0) == 0 {
                panic!("rank {r} exited before listening (see its log above)");
            }
            eprintln!("rank {r}: {}", line.trim_end());
            if let Some(rest) = line.strip_prefix("listening on ") {
                break rest.split(' ').next().unwrap().to_string();
            }
        };
        addrs.push(addr);
        d._out.push(out);
    }
    (d, addrs)
}

/// A backend shared between wrappers.
struct Shared<B: ExpertBackend>(Arc<Mutex<B>>);

impl<B: ExpertBackend> ExpertBackend for Shared<B> {
    fn submit(&mut self, call: &ExpertCall<'_>, stream: &Stream) -> glm53f_forward::Result<()> {
        self.0.lock().unwrap().submit(call, stream)
    }
    fn finish(&mut self, call: &ExpertCall<'_>, stream: &Stream) -> glm53f_forward::Result<()> {
        self.0.lock().unwrap().finish(call, stream)
    }
    fn depth(&self) -> usize {
        self.0.lock().unwrap().depth()
    }
    fn trace_begin(&mut self) {
        self.0.lock().unwrap().trace_begin()
    }
    fn trace_end(&mut self) -> Option<String> {
        self.0.lock().unwrap().trace_end()
    }
}

/// Golden routes of layers 3 and 4 for `runs` runs of the prompt then the 8 steps ([`run`] takes
/// two: each layer fed its golden input, then the chain).
fn golden_routes(g: &Goldens, runs: usize) -> RouteQueue {
    let mut q = HashMap::new();
    for l in [3usize, 4] {
        let (sp, sd) = (
            format!("layer{l:02}-prefill"),
            format!("layer{l:02}-decode"),
        );
        let ids =
            |s: &str, n: &str| -> Vec<i32> { g.i64(s, n).into_iter().map(|x| x as i32).collect() };
        let (pi, pw) = (
            ids(&sp, "prefill.moe.topk_ids"),
            g.f32(&sp, "prefill.moe.topk_weights"),
        );
        let (di, dw) = (
            ids(&sd, "decode.moe.topk_ids"),
            g.f32(&sd, "decode.moe.topk_weights"),
        );
        let mut d = VecDeque::new();
        for _ in 0..runs {
            d.push_back((pi.clone(), pw.clone()));
            for s in 0..STEPS {
                d.push_back((
                    di[s * TOP_K..(s + 1) * TOP_K].to_vec(),
                    dw[s * TOP_K..(s + 1) * TOP_K].to_vec(),
                ));
            }
        }
        q.insert(l, d);
    }
    q
}

/// What one backend produced: the routed output of layers 3 and 4 fed their golden inputs (the
/// prompt's rows, then the 8 steps) with the golden routes; the chain (the prompt in one prefill,
/// then the 8 steps, through layers 0-4 and the head) with the golden routes, and again with the
/// forward's own routing (the serving path).
struct Run {
    routed: HashMap<usize, Vec<f32>>,
    chain: Chain,
    own: Chain,
}

/// A chain's logits (the prompt's last row, then each step) and picks.
struct Chain {
    logits: Vec<f32>,
    picks: Vec<u32>,
}

fn chain(fwd: &mut GlmForward, prompt: &[u32], steps: &[u32]) -> Chain {
    let mut kv = fwd.kv.slot().unwrap();
    kv.reserve(PROMPT + STEPS).unwrap();
    let mut picks = fwd.prefill(&mut [(&mut kv, prompt)]).unwrap();
    let mut logits = fwd.logits(1).unwrap();
    for &t in steps {
        picks.extend(fwd.decode(&mut [(&mut kv, t)]).unwrap());
        logits.extend(fwd.logits(1).unwrap());
    }
    Chain { logits, picks }
}

fn run<B: ExpertBackend + 'static>(
    fwd: &mut GlmForward,
    backend: &Arc<Mutex<B>>,
    g: &Goldens,
    prompt: &[u32],
    steps: &[u32],
) -> Run {
    let max_rows = fwd.cfg.max_rows.max(fwd.cfg.max_verify_rows);
    let mut gr = GoldenRoutes::new(Shared(backend.clone()), max_rows);
    gr.queue = golden_routes(g, 2);
    fwd.set_experts(Box::new(gr));
    // Per layer: each MoE layer from its golden input streams, a fresh slot.
    let rec: Arc<Mutex<HashMap<usize, Vec<f32>>>> = Arc::new(Mutex::new(HashMap::new()));
    let r2 = rec.clone();
    fwd.set_tap(Some(Box::new(move |t: &Tap<'_>| {
        if t.point == TapPoint::FfnDone && (t.layer == 3 || t.layer == 4) {
            let v = widen(&t.bf16(TapBuf::Out)?);
            r2.lock().unwrap().entry(t.layer).or_default().extend(v);
        }
        Ok(())
    })));
    for l in [3usize, 4] {
        let mut kv = fwd.kv.slot().unwrap();
        kv.reserve(PROMPT + STEPS).unwrap();
        let ip = g.f32(&format!("layer{l:02}-prefill"), "prefill.in_streams");
        let id = g.f32(&format!("layer{l:02}-decode"), "decode.in_streams");
        fwd.run_layers(&mut [&mut kv], &[PROMPT], &narrow(&ip), l..l + 1)
            .unwrap();
        for s in 0..STEPS {
            let x = narrow(&rows(&id, HC * HIDDEN, s, s + 1));
            fwd.run_layers(&mut [&mut kv], &[1], &x, l..l + 1).unwrap();
        }
    }
    fwd.set_tap(None);
    let routed = rec.lock().unwrap().clone();
    let golden = chain(fwd, prompt, steps);
    // The serving path: the forward's own routes.
    fwd.set_experts(Box::new(Shared(backend.clone())));
    let own = chain(fwd, prompt, steps);
    fwd.set_experts(Box::new(ZeroExperts));
    Run {
        routed,
        chain: golden,
        own,
    }
}

fn cos(a: &[f32], b: &[f32]) -> f64 {
    let (mut ab, mut aa, mut bb) = (0f64, 0f64, 0f64);
    for (&x, &y) in a.iter().zip(b) {
        ab += x as f64 * y as f64;
        aa += (x as f64).powi(2);
        bb += (y as f64).powi(2);
    }
    ab / (aa.sqrt() * bb.sqrt()).max(1e-300)
}

/// Rows of 4096 of `got` against `want`: the smallest and the mean per-row cosine, the mean
/// per-row relative RMS (the rank crate's figure) and the relative RMS over every value.
struct Cmp {
    min_cos: f64,
    mean_cos: f64,
    row_rel: f64,
    rel: f64,
}

fn compare(got: &[f32], want: &[f32]) -> Cmp {
    assert_eq!(got.len(), want.len());
    let n = (got.len() / HIDDEN) as f64;
    let (mut min_cos, mut sum_cos, mut sum_rel) = (1f64, 0f64, 0f64);
    for (a, b) in got.chunks(HIDDEN).zip(want.chunks(HIDDEN)) {
        let c = cos(a, b);
        min_cos = min_cos.min(c);
        sum_cos += c;
        sum_rel += err(a, b).rel_rms;
    }
    Cmp {
        min_cos,
        mean_cos: sum_cos / n,
        row_rel: sum_rel / n,
        rel: err(got, want).rel_rms,
    }
}

impl std::fmt::Display for Cmp {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{:.4} / {:.4}, {:.2}% / {:.2}%",
            self.min_cos,
            self.mean_cos,
            100.0 * self.row_rel,
            100.0 * self.rel
        )
    }
}

fn argmax(v: &[f32]) -> usize {
    let mut b = 0;
    for i in 0..SAMPLE_VOCAB {
        if v[i] > v[b] {
            b = i;
        }
    }
    b
}

fn report_times(t: &WireTimes) {
    for ((l, rows), x) in t {
        eprintln!(
            "  layer {l}, {rows:>2} rows: {:>2} exchanges, {:.3} ms mean ({:.3} to {:.3}); {:.3} ms of it waiting in finish",
            x.count,
            x.mean_ms(),
            x.min_ms,
            x.max_ms,
            x.wait_ms / x.count.max(1) as f64
        );
    }
}

#[test]
fn layers_0_to_4_on_four_rank_daemons() {
    let names = [
        "layer03-prefill",
        "layer03-decode",
        "layer04-prefill",
        "layer04-decode",
        "layer00-prefill",
        "head",
    ];
    let Some(g) = Goldens::load(&names) else {
        return;
    };
    let Some((bin, dirs)) = rank_setup() else {
        return;
    };
    if !gpu_with(16.0) {
        return;
    }
    let cfg = ForwardConfig {
        max_rows: 64,
        max_verify_rows: 8,
        max_requests: 4,
        ..ForwardConfig::default()
    };
    let Some(su) = forward(5, cfg, HashMap::new(), 0) else {
        return;
    };
    let mut fwd = su.fwd;
    let (prompt, steps) = g.token_ids("layer00-prefill");
    assert_eq!((prompt.len(), steps.len()), (PROMPT, STEPS));
    let rows = cfg.max_rows.max(cfg.max_verify_rows);

    // The official FP8 experts on this GPU.
    let local = Arc::new(Mutex::new(
        LocalFp8Experts::new(
            &experts_dir().unwrap(),
            3 << 30,
            rows,
            fwd.stream(),
            Fp8Act::Bf16,
        )
        .unwrap(),
    ));
    let loc = run(&mut fwd, &local, &g, &prompt, &steps);
    drop(local);

    // The EXL3 experts on four rank daemons.
    let (daemons, addrs) = spawn_ranks(&bin, &dirs);
    let remote = RemoteExperts::connect(&addrs, rows).expect("connect to the ranks");
    let times = remote.times();
    let remote = Arc::new(Mutex::new(remote));
    let rem = run(&mut fwd, &remote, &g, &prompt, &steps);
    let wire = times.lock().unwrap().clone();
    drop(remote);
    drop(daemons);

    eprintln!("\nRouted output of layers 3 and 4, each fed its golden input, golden routes (per-row cosine smallest / mean, relative RMS mean per row / overall):");
    let mut worst = 1f64;
    for l in [3usize, 4] {
        let gp = g.f32(&format!("layer{l:02}-prefill"), "prefill.moe.routed_out");
        let gd = g.f32(&format!("layer{l:02}-decode"), "decode.moe.routed_out");
        let (r, lo) = (&rem.routed[&l], &loc.routed[&l]);
        let n = PROMPT * HIDDEN;
        for (what, span, want) in [("prompt", 0..n, &gp), ("decode", n..r.len(), &gd)] {
            let (a, b) = (&r[span.clone()], &lo[span]);
            let (rg, lg, rl) = (compare(a, want), compare(b, want), compare(a, b));
            eprintln!(
                "  L{l} {what:<6} ({:>2} rows): EXL3 ranks vs oracle {rg} | FP8 local vs oracle {lg} | ranks vs local {rl}",
                a.len() / HIDDEN
            );
            worst = worst.min(rg.min_cos).min(rl.min_cos);
            // The rank crate measures >= 0.990 per row on these layers (it asserts > 0.97). A
            // wrong channel order or a lost rank drops the cosine far below; a routed scale
            // applied twice leaves it at 1 and shows in the relative RMS.
            for (c, against) in [(&rg, "the oracle"), (&rl, "the local FP8 experts")] {
                assert!(
                    c.min_cos >= 0.985 && c.mean_cos >= 0.99 && c.rel < 0.15,
                    "L{l} {what}: the ranks against {against}: {c}"
                );
            }
            assert!(
                lg.min_cos >= 0.999,
                "L{l} {what}: the local FP8 experts against the oracle: {lg}"
            );
        }
    }

    let gl = g.f32("head", "head.logits");
    let agree = |c: &Chain| {
        (0..=STEPS)
            .filter(|&k| {
                argmax(&c.logits[k * VOCAB..(k + 1) * VOCAB])
                    == argmax(&gl[k * VOCAB..(k + 1) * VOCAB])
            })
            .count()
    };
    let same = |a: &Chain, b: &Chain| a.picks.iter().zip(&b.picks).filter(|(x, y)| x == y).count();
    for (what, r, l) in [
        ("golden routes", &rem.chain, &loc.chain),
        ("the forward's own routing", &rem.own, &loc.own),
    ] {
        eprintln!(
            "Chain, layers 0-4 and the head, {what}: logits relative RMS against the oracle {:.2}% (EXL3 ranks) / {:.2}% (FP8 local), ranks against local {:.2}%; argmax equal to the oracle's on {}/9 (ranks) and {}/9 (local); picks equal between the two on {}/9",
            100.0 * err(&r.logits, &gl).rel_rms,
            100.0 * err(&l.logits, &gl).rel_rms,
            100.0 * err(&r.logits, &l.logits).rel_rms,
            agree(r),
            agree(l),
            same(r, l)
        );
    }
    let er = err(&rem.chain.logits, &gl).rel_rms;
    let eo = err(&rem.own.logits, &gl).rel_rms;
    eprintln!("Wire exchanges over TCP loopback on a shared GPU (request written to returns read), by layer and rows:");
    report_times(&wire);
    // Measured 4.6% (the 4-bit experts' error, diluted by the residual streams); the local FP8
    // chain is held to 5% by tests/goldens_chain.rs.
    for (what, e, c) in [
        ("golden routes", er, &rem.chain),
        ("own routing", eo, &rem.own),
    ] {
        assert!(e < 0.08, "chain ({what}) logits against the oracle: {e:.4}");
        assert!(
            agree(c) >= 8,
            "the ranks' chain ({what}) agrees with the oracle's argmax on {}/9",
            agree(c)
        );
    }
    let exchanges: usize = wire.values().map(|t| t.count).sum();
    assert_eq!(
        exchanges,
        3 * 2 * (1 + STEPS),
        "every MoE call went over the wire"
    );
    eprintln!(
        "smallest per-row routed cosine (against the oracle and the local experts): {worst:.4}"
    );
}

// ---- 3. Two lanes on the rank daemons ----------------------------------------------------------

/// Logits rows of two runs: the relative RMS of every value and the largest per row, and the
/// rows (of those whose two best logits in `reference` are at least 0.25 apart) whose picks agree.
fn lane_cmp(a: &[f32], b: &[f32], pa: &[u32], pb: &[u32]) -> (f64, f64, usize, usize) {
    let rows = a.len() / VOCAB;
    let mut worst = 0f64;
    let (mut decided, mut agree) = (0, 0);
    for r in 0..rows {
        let (x, y) = (
            &a[r * VOCAB..(r + 1) * VOCAB],
            &b[r * VOCAB..(r + 1) * VOCAB],
        );
        worst = worst.max(err(x, y).rel_rms);
        let best = argmax(y);
        let second = (0..SAMPLE_VOCAB)
            .filter(|&i| i != best)
            .map(|i| y[i])
            .fold(f32::NEG_INFINITY, f32::max);
        if y[best] - second >= 0.25 {
            decided += 1;
            agree += usize::from(pa[r] == pb[r]);
        }
    }
    (err(a, b).rel_rms, worst, decided, agree)
}

fn bits(a: &[f32], b: &[f32]) -> bool {
    a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits())
}

/// Prompts in one prefill (with `cut`: the rows before it in one pass, the rest in a second),
/// then `steps` batched decode steps: every logit row (the prompts' last rows, then each step's)
/// and the picks. The steps take the tokens of `tokens` (a run's picks) when given, else the
/// run's own picks, so runs compared step by step see the same tokens.
fn run_prompts(
    fwd: &mut GlmForward,
    prompts: &[&[u32]],
    steps: usize,
    cut: Option<usize>,
    tokens: Option<&[u32]>,
) -> (Vec<f32>, Vec<u32>) {
    let mut kvs: Vec<_> = prompts
        .iter()
        .map(|p| {
            let mut kv = fwd.kv.slot().unwrap();
            kv.reserve(p.len() + steps).unwrap();
            kv
        })
        .collect();
    let (mut picks, mut logits) = match cut {
        None => {
            let mut segs: Vec<(&mut glm53f_forward::kv::GlmKv, &[u32])> =
                kvs.iter_mut().zip(prompts).map(|(k, p)| (k, *p)).collect();
            let picks = fwd.prefill(&mut segs).unwrap();
            (picks, fwd.logits(prompts.len()).unwrap())
        }
        Some(at) => {
            // The request the cut falls in is the last one of the first pass.
            let mut row = 0;
            let mut first: Vec<(usize, usize)> = Vec::new();
            for (i, p) in prompts.iter().enumerate() {
                if row < at {
                    first.push((i, p.len().min(at - row)));
                }
                row += p.len();
            }
            let (split, n) = *first.last().unwrap();
            let mut segs: Vec<(&mut glm53f_forward::kv::GlmKv, &[u32])> = kvs
                .iter_mut()
                .zip(prompts)
                .zip(&first)
                .map(|((k, p), &(_, n))| (k, &p[..n]))
                .collect();
            let mut picks = fwd.prefill(&mut segs).unwrap();
            let mut logits = fwd.logits(first.len()).unwrap();
            let rest: Vec<&[u32]> = std::iter::once(&prompts[split][n..])
                .chain(prompts[split + 1..].iter().copied())
                .collect();
            let mut segs: Vec<(&mut glm53f_forward::kv::GlmKv, &[u32])> = kvs[split..]
                .iter_mut()
                .zip(rest)
                .filter(|(_, p)| !p.is_empty())
                .collect();
            let skip = usize::from(n == prompts[split].len());
            let second = fwd.prefill(&mut segs).unwrap();
            let l2 = fwd.logits(second.len()).unwrap();
            // The first pass's row for the split request is not a prompt's last row.
            if skip == 0 {
                picks.pop();
                logits.truncate(logits.len() - VOCAB);
            }
            picks.extend(second);
            logits.extend(l2);
            (picks, logits)
        }
    };
    let n = prompts.len();
    let mut last = picks.clone();
    for s in 0..steps {
        let feed = tokens.map_or(last.clone(), |t| t[s * n..(s + 1) * n].to_vec());
        let mut rows: Vec<(&mut glm53f_forward::kv::GlmKv, u32)> =
            kvs.iter_mut().zip(&feed).map(|(k, &t)| (k, t)).collect();
        last = fwd.decode(&mut rows).unwrap();
        picks.extend(&last);
        logits.extend(fwd.logits(n).unwrap());
    }
    (logits, picks)
}

fn prompt_ids(seed: u64, n: usize) -> Vec<u32> {
    let mut s = seed;
    (0..n)
        .map(|_| {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            1000 + ((s >> 33) % 150_000) as u32
        })
        .collect()
}

#[test]
fn two_lanes_on_four_rank_daemons() {
    let Some(g) = Goldens::load(&["layer00-prefill", "head"]) else {
        return;
    };
    let Some((bin, dirs)) = rank_setup() else {
        return;
    };
    if !gpu_with(13.0) {
        return;
    }
    let (n1, n2) = (150usize, 211usize);
    let cfg = ForwardConfig {
        max_rows: 2 * (n1 + n2),
        lanes: 2,
        min_lane_rows: 8,
        max_verify_rows: 8,
        max_requests: 4,
        ..ForwardConfig::default()
    };
    let (daemons, addrs) = spawn_ranks(&bin, &dirs);
    let rows = cfg.lane_rows().max(cfg.max_verify_rows);
    let remote = RemoteExperts::connect(&addrs, rows).expect("connect to the ranks");
    let times = remote.times();
    let depth = remote.depth();
    let Some(mut fwd) = forward_with(5, cfg, |_| Box::new(remote), 8, 256, 16) else {
        return;
    };
    fwd.set_lane_trace(true, false);
    let (prompt, steps) = g.token_ids("layer00-prefill");
    let gl = g.f32("head", "head.logits");
    let prompts = [prompt_ids(1, n1), prompt_ids(2, n2)];
    let batch: Vec<&[u32]> = prompts.iter().map(|p| &p[..]).collect();
    let at = (n1 + n2).div_ceil(2);

    // The oracle's prompt: its fixed decode tokens after the prefill, the forward's own routing.
    let chain = |fwd: &mut GlmForward, cut: Option<usize>| -> (Vec<f32>, Vec<u32>) {
        let mut kv = fwd.kv.slot().unwrap();
        kv.reserve(PROMPT + STEPS).unwrap();
        if let Some(k) = cut {
            fwd.prefill(&mut [(&mut kv, &prompt[..k])]).unwrap();
        }
        let mut picks = fwd
            .prefill(&mut [(&mut kv, &prompt[cut.unwrap_or(0)..])])
            .unwrap();
        let mut logits = fwd.logits(1).unwrap();
        for &t in &steps {
            picks.extend(fwd.decode(&mut [(&mut kv, t)]).unwrap());
            logits.extend(fwd.logits(1).unwrap());
        }
        (logits, picks)
    };
    fwd.cfg.lanes = 1;
    let one = chain(&mut fwd, None);
    let seq = chain(&mut fwd, Some(17));
    let b_one = run_prompts(&mut fwd, &batch, 4, None, None);
    let tb_one = fwd.take_lane_trace().unwrap();
    // Every batch run's steps take the tokens the one-pass run's steps took (the prompts' picks,
    // then each step's).
    let feed = b_one.1[..8].to_vec();
    let b_seq = run_prompts(&mut fwd, &batch, 4, Some(at), Some(&feed));
    fwd.cfg.lanes = 2;
    let two = chain(&mut fwd, None);
    let t_two = fwd.take_lane_trace().unwrap();
    let b_two = run_prompts(&mut fwd, &batch, 4, None, Some(&feed));
    let tb_two = fwd.take_lane_trace().unwrap();
    let wire = times.lock().unwrap().clone();
    drop(fwd);
    drop(daemons);

    assert_eq!(t_two.rows, vec![17, 16]);
    assert_eq!(tb_two.rows, vec![at, n1 + n2 - at]);
    let (rel, worst, decided, agree) = lane_cmp(&two.0, &one.0, &two.1, &one.1);
    let e2 = err(&two.0, &gl).rel_rms;
    let a2 = (0..=STEPS)
        .filter(|&k| {
            argmax(&two.0[k * VOCAB..(k + 1) * VOCAB]) == argmax(&gl[k * VOCAB..(k + 1) * VOCAB])
        })
        .count();
    let (brel, bworst, bdecided, bagree) = lane_cmp(&b_two.0, &b_one.0, &b_two.1, &b_one.1);
    let (exact, bexact) = (
        bits(&two.0, &seq.0) && two.1 == seq.1,
        bits(&b_two.0, &b_seq.0) && b_two.1 == b_seq.1,
    );
    eprintln!(
        "two lanes on four rank daemons (TCP loopback, {depth} exchange in flight at most): \
         the oracle's prompt in lanes of 17 and 16 rows: against two passes bit for bit {exact}; \
         against the golden logits relative RMS {e2:.3e}, argmax {a2}/9; against one pass \
         {rel:.3e} (worst row {worst:.3e}), picks {agree}/{decided} of the rows decided by 0.25; \
         two prompts of {n1} and {n2} tokens in lanes of {at} and {} rows, 4 steps after: \
         against two passes bit for bit {bexact}; against one pass {brel:.3e} (worst row \
         {bworst:.3e}), picks {bagree}/{bdecided} decided",
        n1 + n2 - at
    );
    eprintln!(
        "lane trace (loopback, the ranks on this GPU), the oracle's prompt: {}",
        t_two.summary()
    );
    eprintln!(
        "lane trace (loopback), two prompts in two lanes: {}",
        tb_two.summary()
    );
    eprintln!(
        "lane trace (loopback), the same in one pass: {}",
        tb_one.summary()
    );
    eprintln!("wire exchanges (loopback), by layer and rows:");
    report_times(&wire);
    for ((l, r), x) in &wire {
        eprintln!(
            "  layer {l}, {r:>3} rows: host send {:.3} ms, upload {:.3} ms per exchange",
            x.send_ms / x.count.max(1) as f64,
            x.upload_ms / x.count.max(1) as f64
        );
    }
    assert!(
        exact && bexact,
        "two lanes differ from two passes of the same rows"
    );
    // The rank chain's bounds (test 2) against the oracle; one pass within the same.
    assert!(
        e2 < 0.08 && a2 >= 8,
        "two lanes against the golden: {e2:.3e}, {a2}/9"
    );
    assert!(
        worst < 0.08 && bworst < 0.08,
        "two lanes against one pass: {worst:.3e}, {bworst:.3e}"
    );
    assert_eq!(
        (agree, bagree),
        (decided, bdecided),
        "two lanes pick other tokens than one pass"
    );
    assert!(
        decided >= 6 && bdecided >= 6,
        "rows decided: {decided}, {bdecided}"
    );
}

// ---- 4. Return paths and fast paths on the rank daemons -------------------------------------

/// The four-plane configuration, and the row-sharded one (from 16 rows, BF16 exchange).
fn wire_configs() -> (WireConfig, WireConfig) {
    let four = WireConfig::glm53_flash();
    let rs = WireConfig {
        return_path: ReturnPath::RowSharded {
            min_rows: 16,
            exchange: ExchangeDtype::Bf16,
        },
        ..four
    };
    (four, rs)
}

fn bf16s(b: &[u8]) -> Vec<u16> {
    b.chunks_exact(2)
        .map(|c| u16::from_le_bytes([c[0], c[1]]))
        .collect()
}

/// Row slices against the four planes `planes` (BF16, rank order) of the same FP8 rows, whose
/// f32 sum is `s`: every element within the bound glm53f-rank's README derives for the BF16
/// exchange (for a row owned by rank q: half a BF16 step of q's own partial, which travels
/// unrounded in one path and rounded in the other; half a BF16 step of the row-sharded sum; the
/// f32 additions of both paths). Returns (RMS of the difference / RMS of the sum, the worst
/// difference / its bound).
fn within_bound(rs: &[f32], s: &[f32], planes: &[Vec<u16>]) -> (f64, f64) {
    let rows = rs.len() / HIDDEN;
    let (mut d2, mut s2, mut worst) = (0f64, 0f64, 0f64);
    for t in 0..rows {
        let q = (0..RANKS)
            .find(|&r| {
                let (f, c) = row_partition(rows, RANKS, r);
                (f..f + c).contains(&t)
            })
            .unwrap();
        for i in 0..HIDDEN {
            let e = t * HIDDEN + i;
            let b: Vec<f64> = planes.iter().map(|p| widen(&[p[e]])[0] as f64).collect();
            let sv = s[e] as f64;
            let own = b[q].abs() * 2f64.powi(-8);
            let fp32 = 8.0 * 2f64.powi(-24) * b.iter().map(|v| v.abs()).sum::<f64>();
            let bound = own + (sv.abs() + own) * 2f64.powi(-8) + fp32 + 1e-30;
            let d = (rs[e] as f64 - sv).abs();
            assert!(
                d <= bound,
                "row {t} element {i}: row-sharded {} against four planes {sv} (bound {bound:.3e})",
                rs[e]
            );
            worst = worst.max(d / bound);
            d2 += d * d;
            s2 += sv * sv;
        }
    }
    ((d2 / s2).sqrt(), worst)
}

/// One routed-experts call through `remote` on `stream`: its BF16 output.
fn call_remote(
    remote: &mut RemoteExperts,
    stream: &Stream,
    layer: usize,
    x: &[u16],
    ids: &[i32],
    w: &[f32],
) -> Vec<u16> {
    let rows = x.len() / HIDDEN;
    let xd = DeviceBuffer::alloc(x.len() * 2).unwrap();
    xd.upload_async(stream, 0, x).unwrap();
    let (di, dw) = (
        DeviceBuffer::alloc(ids.len() * 4).unwrap(),
        DeviceBuffer::alloc(w.len() * 4).unwrap(),
    );
    di.upload_async(stream, 0, ids).unwrap();
    dw.upload_async(stream, 0, w).unwrap();
    let out = DeviceBuffer::alloc(x.len() * 2).unwrap();
    let call = ExpertCall {
        layer,
        rows,
        x: xd.ptr(0),
        x_q: core::ptr::null(),
        x_scales: core::ptr::null(),
        ids: di.ptr(0),
        weights: dw.ptr(0),
        host_ids: ids,
        host_weights: w,
        out: out.ptr(0),
    };
    remote.submit(&call, stream).expect("submit");
    remote.finish(&call, stream).expect("finish");
    stream.synchronize().unwrap();
    out.download(rows * HIDDEN).unwrap()
}

fn report_modes(name: &str, t: &WireTimes) {
    for ((l, rows), x) in t {
        let n = x.count.max(1) as f64;
        eprintln!(
            "  {name}: layer {l}, {rows:>2} rows, {}: {} exchanges, {:.3} ms mean (waiting in finish {:.3}); host: send {:.3} (on the device {:.3}), placing the returns {:.3}",
            if x.row_sharded { "row slices" } else { "four planes" },
            x.count,
            x.mean_ms(),
            x.wait_ms / n,
            x.send_ms / n,
            x.send_wait_ms / n,
            x.upload_ms / n
        );
    }
}

/// The return paths and the fast paths through `RemoteExperts` on the four rank daemons (with
/// their peer mesh): layers 3 and 4, each fed the oracle's MoE input as the forward holds it
/// (BF16) with the golden routes, the prompt's 33 rows and the 8 decode rows, four ways (four
/// planes or row slices, host or fast paths).
///
/// - Four planes: the fast paths give the host paths' bits, and both are the BF16 rounding of the
///   planes a plain wire client collects for the same FP8 rows, added in rank order.
/// - Row slices: the prompt's rows within the bound of the four-plane sum, the fast paths giving
///   the host paths' bits; the decode rows stay four planes, the four-plane path's bits.
/// - Both against the oracle's routed output; the exchange times per return path (loopback, a
///   shared GPU: they show the paths, not the target hardware).
#[test]
fn return_paths_and_fast_paths_on_four_rank_daemons() {
    let names = [
        "layer03-prefill",
        "layer03-decode",
        "layer04-prefill",
        "layer04-decode",
    ];
    let Some(g) = Goldens::load(&names) else {
        return;
    };
    let Some((bin, dirs)) = rank_setup() else {
        return;
    };
    if !gpu_with(10.0) {
        return;
    }
    // Per layer, the prompt's rows then the decode rows: (layer, input BF16, ids, weights, the
    // oracle's routed output).
    type Input = (usize, Vec<u16>, Vec<i32>, Vec<f32>, Vec<f32>);
    let mut inputs: Vec<Input> = Vec::new();
    for l in [3usize, 4] {
        for (set, pre) in [
            (format!("layer{l:02}-prefill"), "prefill"),
            (format!("layer{l:02}-decode"), "decode"),
        ] {
            inputs.push((
                l,
                narrow(&g.f32(&set, &format!("{pre}.ffn_norm"))),
                g.i64(&set, &format!("{pre}.moe.topk_ids"))
                    .into_iter()
                    .map(|v| v as i32)
                    .collect(),
                g.f32(&set, &format!("{pre}.moe.topk_weights")),
                g.f32(&set, &format!("{pre}.moe.routed_out")),
            ));
        }
    }
    let (daemons, addrs) = spawn_ranks(&bin, &dirs);
    let (four, rs) = wire_configs();
    let stream = Stream::new().unwrap();
    // One backend at a time: a rank serves one coordinator connection at a time.
    let run = |cfg: WireConfig, fast: FastPaths| -> (Vec<Vec<u16>>, WireTimes) {
        let mut remote = RemoteExperts::connect_with(&addrs, 64, cfg, fast).expect("connect");
        assert_eq!(remote.fast_paths(), fast);
        let times = remote.times();
        let outs = inputs
            .iter()
            .map(|(l, x, ids, w, _)| call_remote(&mut remote, &stream, *l, x, ids, w))
            .collect();
        drop(remote);
        let t = times.lock().unwrap().clone();
        (outs, t)
    };
    // The four-plane fast paths with the frame fill (every call here has at most 64 rows), the
    // row-sharded ones as they default (the DMA copies).
    let fill = FastPaths {
        fill_rows: 64,
        ..FastPaths::ON
    };
    let runs = [
        ("four planes, host paths", run(four, FastPaths::OFF)),
        ("four planes, fast paths (frame fill)", run(four, fill)),
        ("row slices, host paths", run(rs, FastPaths::OFF)),
        ("row slices, fast paths", run(rs, FastPaths::ON)),
    ];
    let [a, b, c, d] = [&runs[0].1 .0, &runs[1].1 .0, &runs[2].1 .0, &runs[3].1 .0];
    let mut client = WireClient::connect(&addrs, four).expect("connect");
    for (k, (l, x, ids, w, want)) in inputs.iter().enumerate() {
        let rows = x.len() / HIDDEN;
        assert!(
            a[k] == b[k],
            "layer {l}, {rows} rows: the fast paths change the four-plane bits"
        );
        // The planes of the same FP8 rows, through a plain wire client.
        let q = glm53f_coordinator::wire::quantize_hidden_batched(&widen(x)).unwrap();
        let payload: Vec<u8> = q.iter().flat_map(|h| h.payload.iter().copied()).collect();
        let scales: Vec<u8> = q.iter().flat_map(|h| h.scales.iter().copied()).collect();
        let routes: Vec<(u32, f32)> = ids.iter().zip(w).map(|(&e, &g)| (e as u32, g)).collect();
        client
            .moe_send_raw(*l as u32, &payload, &scales, &routes, TOP_K)
            .unwrap();
        client.moe_recv_raw().unwrap();
        let Some(Collected::Planes(p)) = client.collected() else {
            panic!("four planes expected");
        };
        let planes: Vec<Vec<u16>> = p.iter().map(|b| bf16s(b)).collect();
        let s: Vec<f32> = (0..rows * HIDDEN)
            .map(|i| planes.iter().fold(0f32, |acc, pl| acc + widen(&[pl[i]])[0]))
            .collect();
        assert!(
            a[k] == narrow(&s),
            "layer {l}, {rows} rows: not the planes' sum"
        );
        let cos = |v: &[u16]| {
            let v = widen(v);
            (0..rows)
                .map(|r| {
                    let (x, y) = (
                        &v[r * HIDDEN..(r + 1) * HIDDEN],
                        &want[r * HIDDEN..(r + 1) * HIDDEN],
                    );
                    cos(x, y)
                })
                .fold(1f64, f64::min)
        };
        if rows < 16 {
            assert!(
                c[k] == a[k] && d[k] == a[k],
                "layer {l}, {rows} rows (decode): not the four-plane bits"
            );
            eprintln!(
                "layer {l}, {rows} rows (decode): four planes under every configuration, bit for bit; cosine against the oracle >= {:.4}",
                cos(&a[k])
            );
            continue;
        }
        assert!(
            c[k] == d[k],
            "layer {l}, {rows} rows: the fast paths change the row-slice bits"
        );
        let (rel, worst) = within_bound(&widen(&c[k]), &s, &planes);
        let (cf, cr) = (cos(&a[k]), cos(&c[k]));
        eprintln!(
            "layer {l}, {rows} rows (prompt): row slices against four planes: RMS {rel:.2e} of the sum's, worst {worst:.2} of the bound; cosine against the oracle >= {cf:.4} (four planes), {cr:.4} (row slices); the fast paths bit for bit both ways"
        );
        assert!(
            cf > 0.98 && cr > 0.98,
            "layer {l}: against the oracle's routed output"
        );
    }
    drop(client);
    drop(daemons);
    eprintln!("wire exchanges (TCP loopback, four daemons and the coordinator on one GPU):");
    for (name, (_, t)) in &runs {
        report_modes(name, t);
    }
    for (name, (_, t)) in &runs[2..] {
        let modes: Vec<(usize, bool)> = t.iter().map(|((_, r), x)| (*r, x.row_sharded)).collect();
        assert!(
            modes.iter().all(|&(r, s)| s == (r >= 16)),
            "{name}: row slices from 16 rows: {modes:?}"
        );
    }
}

/// Two-lane prefill (the serving path: the forward's own routing) on the four rank daemons, four
/// ways: four planes or row slices, host or fast paths. The oracle's prompt (lanes of 17 and 16
/// rows) and its 8 fixed decode tokens; two prompts batched across the cut (lanes of 181 and 180
/// rows), then 4 decode steps fed the same tokens.
///
/// - The fast paths give the host paths' bits, for four planes and for row slices, prefill and
///   decode.
/// - Row slices against four planes: the logits within the chain's bound, the same picks on every
///   row whose best two logits are 0.25 apart; the oracle's prompt against the golden logits as
///   the ranks' chain is held (`layers_0_to_4_on_four_rank_daemons`).
/// - The lane trace carries the wire's record: every MoE call of a prefill pass row-sharded under
///   the row-sharded configuration, none under the four-plane one.
#[test]
fn row_sharded_two_lane_prefill_on_four_rank_daemons() {
    let Some(g) = Goldens::load(&["layer00-prefill", "head"]) else {
        return;
    };
    let Some((bin, dirs)) = rank_setup() else {
        return;
    };
    if !gpu_with(13.0) {
        return;
    }
    let (n1, n2) = (150usize, 211usize);
    let cfg = ForwardConfig {
        max_rows: 2 * (n1 + n2),
        lanes: 2,
        min_lane_rows: 8,
        max_verify_rows: 8,
        max_requests: 4,
        ..ForwardConfig::default()
    };
    let (daemons, addrs) = spawn_ranks(&bin, &dirs);
    let rows = cfg.lane_rows().max(cfg.max_verify_rows);
    let (four, rs) = wire_configs();
    let connect = |c: WireConfig, fast: FastPaths| {
        RemoteExperts::connect_with(&addrs, rows, c, fast).expect("connect to the ranks")
    };
    let Some(mut fwd) = forward_with(5, cfg, |_| Box::new(ZeroExperts), 8, 256, 16) else {
        return;
    };
    fwd.set_lane_trace(true, false);
    let (prompt, steps) = g.token_ids("layer00-prefill");
    let gl = g.f32("head", "head.logits");
    let prompts = [prompt_ids(1, n1), prompt_ids(2, n2)];
    let batch: Vec<&[u32]> = prompts.iter().map(|p| &p[..]).collect();
    let chain = |fwd: &mut GlmForward| -> (Vec<f32>, Vec<u32>) {
        let mut kv = fwd.kv.slot().unwrap();
        kv.reserve(PROMPT + STEPS).unwrap();
        let mut picks = fwd.prefill(&mut [(&mut kv, &prompt[..])]).unwrap();
        let mut logits = fwd.logits(1).unwrap();
        for &t in &steps {
            picks.extend(fwd.decode(&mut [(&mut kv, t)]).unwrap());
            logits.extend(fwd.logits(1).unwrap());
        }
        (logits, picks)
    };
    let ways = [
        ("four planes, host paths", four, FastPaths::OFF),
        ("four planes, fast paths", four, FastPaths::ON),
        ("row slices, fast paths", rs, FastPaths::ON),
        ("row slices, host paths", rs, FastPaths::OFF),
    ];
    type Way = ((Vec<f32>, Vec<u32>), (Vec<f32>, Vec<u32>), String, String);
    let mut runs: Vec<Way> = Vec::new();
    let mut feed: Option<Vec<u32>> = None;
    for (name, c, fast) in ways {
        // One backend at a time: a rank serves one coordinator connection at a time.
        drop(fwd.set_experts(Box::new(ZeroExperts)));
        fwd.set_experts(Box::new(connect(c, fast)));
        let one = chain(&mut fwd);
        let t_one = fwd.take_lane_trace().unwrap();
        let two = run_prompts(&mut fwd, &batch, 4, None, feed.as_deref());
        let t_two = fwd.take_lane_trace().unwrap();
        feed.get_or_insert_with(|| two.1[..8].to_vec());
        assert_eq!(t_one.rows, vec![17, 16]);
        assert_eq!(t_two.rows, vec![(n1 + n2).div_ceil(2), (n1 + n2) / 2]);
        eprintln!(
            "{name}: lane trace (loopback, the ranks on this GPU), two prompts: {}",
            t_two.summary()
        );
        let wire = |t: &glm53f_forward::forward::LaneTrace| t.wire.clone().unwrap_or_default();
        runs.push((one, two, wire(&t_one), wire(&t_two)));
    }
    drop(fwd.set_experts(Box::new(ZeroExperts)));
    drop(fwd);
    drop(daemons);

    let same = |x: &(Vec<f32>, Vec<u32>), y: &(Vec<f32>, Vec<u32>)| bits(&x.0, &y.0) && x.1 == y.1;
    assert!(
        same(&runs[1].0, &runs[0].0) && same(&runs[1].1, &runs[0].1),
        "four planes: the fast paths change the chain"
    );
    assert!(
        same(&runs[3].0, &runs[2].0) && same(&runs[3].1, &runs[2].1),
        "row slices: the fast paths change the chain"
    );
    for (k, (_, _, w_one, w_two)) in runs.iter().enumerate() {
        let (want, none) = if k < 2 {
            ("four planes 4:", "row slices 0")
        } else {
            ("row slices 4:", "four planes 0")
        };
        for w in [w_one, w_two] {
            assert!(
                w.contains(want) && w.contains(none),
                "{}: the trace's wire record: {w}",
                ways[k].0
            );
        }
    }
    let (four_run, rs_run) = (&runs[0], &runs[2]);
    let (rel, worst, decided, agree) =
        lane_cmp(&rs_run.0 .0, &four_run.0 .0, &rs_run.0 .1, &four_run.0 .1);
    let (brel, bworst, bdecided, bagree) =
        lane_cmp(&rs_run.1 .0, &four_run.1 .0, &rs_run.1 .1, &four_run.1 .1);
    let e = err(&rs_run.0 .0, &gl).rel_rms;
    let a = (0..=STEPS)
        .filter(|&k| {
            argmax(&rs_run.0 .0[k * VOCAB..(k + 1) * VOCAB])
                == argmax(&gl[k * VOCAB..(k + 1) * VOCAB])
        })
        .count();
    eprintln!(
        "row slices against four planes, two lanes on four rank daemons: the oracle's prompt and 8 steps: \
         logits relative RMS {rel:.3e} (worst row {worst:.3e}), picks {agree}/{decided} of the rows decided by 0.25; \
         against the golden logits {e:.3e}, argmax {a}/9; two prompts of {n1} and {n2} tokens and 4 steps: \
         {brel:.3e} (worst row {bworst:.3e}), picks {bagree}/{bdecided} decided; the fast paths bit for bit both ways"
    );
    assert!(
        e < 0.08 && a >= 8,
        "row slices against the golden: {e:.3e}, {a}/9"
    );
    assert!(
        worst < 0.08 && bworst < 0.08,
        "row slices against four planes: {worst:.3e}, {bworst:.3e}"
    );
    assert_eq!(
        (agree, bagree),
        (decided, bdecided),
        "row slices pick other tokens than four planes"
    );
    assert!(
        decided >= 6 && bdecided >= 6,
        "rows decided: {decided}, {bdecided}"
    );
}
