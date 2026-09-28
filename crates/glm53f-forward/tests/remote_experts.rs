//! `RemoteExperts`: the routed experts on four expert ranks over the wire (feature
//! `coordinator`).
//!
//! 1. **Against four mock ranks** (a GPU, no data): the wire rows the ranks receive are the wire
//!    client's host quantizer applied to the widened BF16 input, the routes are the call's, and
//!    the routed output is the BF16 rounding of the four planes added in rank order in f32.
//! 2. **Layers 0-4 on four real rank daemons** over TCP loopback: the EXL3 shares of layers 3
//!    and 4 cut by `glm53f-rank slice`, served by `glm53f-rank serve` on this GPU. Each MoE layer
//!    fed its golden input streams, with the golden routes, against the oracle's routed output
//!    and against `LocalFp8Experts` on the same inputs; then the chain (the 33-token prompt in
//!    one prefill, 8 decode steps through layers 0-4 and the head) against the golden logits.
//!    EXL3 4-bit experts are not the FP8 experts' bits: the rank crate measures a cosine of at
//!    least 0.990 per row against the reference (`glm53f-rank`, `tests/real_experts.rs`).
//!
//! Test 2 needs, besides the variables of `tests/common/mod.rs`:
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
use glm53f_forward::device::{DeviceBuffer, Stream};
use glm53f_forward::experts::{ExpertBackend, ExpertCall, LocalFp8Experts, ZeroExperts};
use glm53f_forward::forward::{ForwardConfig, GlmForward, Tap, TapBuf, TapPoint};
use glm53f_forward::gemm::Fp8Act;
use glm53f_forward::remote::{RemoteExperts, WireTimes, RANKS};
use glm53f_forward::shape::{HC, HIDDEN, SAMPLE_VOCAB, TOP_K, VOCAB};
use glm53f_wire::frame::{Frame, RequestFrame, ReturnFrame, ReturnRow};
use glm53f_wire::l4::{StreamReceiver, StreamSender};
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
/// request is handed back when the server ends, in (exchange, rank) order.
fn mock_ranks(
    exchanges: usize,
    value: fn(usize, usize, usize) -> f32,
) -> (String, std::thread::JoinHandle<Vec<RequestFrame>>) {
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
                let Frame::Request(req) = rx[r].accept(&read_frame(s)).expect("L4 accept") else {
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
                seen.push(req);
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

#[test]
fn remote_experts_against_four_mock_ranks() {
    if !gpu_with(0.5) {
        return;
    }
    let passes = [1usize, 5, 16];
    let (addr, server) = mock_ranks(passes.len(), plane_value);
    let mut remote = RemoteExperts::connect(&vec![addr; RANKS], 16).expect("connect");
    let times = remote.times();
    let stream = Stream::new().unwrap();
    let mut sent: Vec<(Vec<u16>, Vec<i32>, Vec<f32>)> = Vec::new();
    for (k, &rows) in passes.iter().enumerate() {
        let x = input_rows(rows);
        // On the stream the backend runs on: a synchronous copy from pageable memory can return
        // before its data lands, and the stream does not wait for the legacy stream.
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
        let call = ExpertCall {
            layer: 3 + k,
            rows,
            x: xd.ptr(0),
            x_q: core::ptr::null(),
            x_scales: core::ptr::null(),
            ids: core::ptr::null(),
            weights: core::ptr::null(),
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
        assert_eq!(got, narrow(&want), "{rows} rows: the rank sum");
        sent.push((x, ids, w));
    }
    let frames = server.join().expect("mock ranks");
    assert_eq!(frames.len(), passes.len() * RANKS);
    for (f, req) in frames.iter().enumerate() {
        let (k, r) = (f / RANKS, f % RANKS);
        let (x, ids, w) = &sent[k];
        assert_eq!(
            (req.layer_id, req.executor_id, req.rows.len()),
            ((3 + k) as u32, r as u64, passes[k])
        );
        let want = glm53f_coordinator::wire::quantize_hidden_batched(&widen(x)).unwrap();
        assert_eq!(req.hidden_rows, want, "pass {k}, rank {r}: the wire rows");
        let routes: Vec<(u32, u32, f32)> = req
            .routes
            .iter()
            .map(|e| (e.row_index, e.expert_id, e.gate_weight))
            .collect();
        let expect: Vec<(u32, u32, f32)> = (0..ids.len())
            .map(|i| ((i / TOP_K) as u32, ids[i] as u32, w[i]))
            .collect();
        assert_eq!(routes, expect, "pass {k}, rank {r}: the routes");
    }
    let t = times.lock().unwrap();
    assert_eq!(t.len(), passes.len());
    assert!(t.values().all(|x| x.count == 1));
    eprintln!(
        "RemoteExperts against four mock ranks: {} passes of {passes:?} rows, wire rows, routes and rank sums exact",
        passes.len()
    );
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

/// Start the four daemons on loopback; returns them and their addresses in rank order.
fn spawn_ranks(bin: &PathBuf, dirs: &[PathBuf]) -> (Daemons, Vec<String>) {
    let mut d = Daemons {
        children: Vec::new(),
        _out: Vec::new(),
    };
    for (r, dir) in dirs.iter().enumerate() {
        let child = Command::new(bin)
            .args(["serve", "--rank", &r.to_string(), "--dir"])
            .arg(dir)
            .args(["--listen", "127.0.0.1:0", "--allow-partial"])
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
