//! The row-sharded return (the prefill reduce-scatter) end to end: four real rank daemons
//! (`glm53f-rank serve --peers ...`) on one GPU over loopback TCP, driven by the wire client.
//!
//! It needs:
//!
//! - `GLM53F_RANK_BIN`: a `glm53f-rank` binary built with `--features cuda`;
//! - `GLM53F_RANK_DIRS`: the four rank directories, comma-separated in rank order, each cut with
//!   `glm53f-rank slice --layers 3-4` from the EXL3 checkpoint;
//! - `GLM53F_GOLDENS` (default: this repository's `oracle/goldens`): the oracle's layer-3 and
//!   layer-4 sets.
//!
//! Anything missing makes it print why and pass. The four daemons take about 2.3 GiB of GPU
//! memory each (two layers, the CUDA context and the kernel's scratch at 2,048 rows).
//!
//! One test, one set of daemons, four phases:
//!
//! 1. **Both ways, and the oracle.** Layers 3 and 4, the oracle's prefill inputs (33 rows) and
//!    routes, sent three times: four planes (the version-3 return), row-sharded with the BF16
//!    exchange, row-sharded with the FP8 exchange. Every row-sharded element must be within the
//!    bound the rank crate's README derives, applied to the difference of the two paths (for a
//!    row owned by rank q, with `b_r` rank r's BF16 plane and `S` the four-plane sum):
//!    - BF16 exchange: `2^-8 |b_q|` (rank q's own partial travels unrounded in one path and
//!      rounded in the other) `+ 2^-8 (|S| + 2^-8 |b_q|)` (the row-sharded sum's BF16 rounding)
//!      `+ 8 * 2^-24 * sum_r |b_r|` (the FP32 additions of both paths);
//!    - FP8 exchange: the same, plus for each peer `r != q` half an E4M3 step of its value
//!      (`quant_bound`, with the row scale taken from `b_r`) and `2^-8 |b_r|`.
//!
//!    Each path is also compared with the oracle's routed output (the official FP8 experts):
//!    the 4-bit EXL3 experts give a cosine of at least 0.990 (`glm53f-rank`,
//!    `tests/real_experts.rs`).
//! 2. **Decode stays four-plane.** The decode inputs (8 rows) under the row-sharded
//!    configuration are below the threshold: four planes, bit for bit the four-plane client's.
//! 3. **Loopback timings** at 2,048 synthetic rows, both ways, with the ranks' exchange trace
//!    lines. Four daemons share one GPU and the loopback: the numbers are this machine's, not
//!    the target hardware's.
//! 4. **A dead peer.** Rank 3 is killed; ranks 0-2, sent a reduce-scattered request directly,
//!    each answer with an error return within the peer timeout: never a hang, never a sum.
//!
//! ```sh
//! GLM53F_RANK_BIN=... GLM53F_RANK_DIRS=a,b,c,d GLM53F_GOLDENS=... \
//!   cargo test --release -p glm53f-coordinator --test row_sharded -- --nocapture
//! ```

use std::io::{BufRead, BufReader, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::path::PathBuf;
use std::process::{Child, Command, Stdio};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use glm53f_coordinator::wire::{quantize_hidden_batched, Collected, ReturnPath, WireClient, WireConfig};
use glm53f_wire::bf16::bf16_to_f32;
use glm53f_wire::frame::{decode_frame, encode_request, Frame, RequestFrame, ReturnFrame, RouteEntry, RowDescriptor};
use glm53f_wire::row_shard::{row_partition, ExchangeDtype};
use glm53f_wire::{SourceKind, Status, WireNaive, HIDDEN, SPARKS};

const TOPK: usize = 8;
/// The daemons' exchange timeout.
const PEER_TIMEOUT_MS: u64 = 3000;
/// Rows of the timing phase.
const TIMING_ROWS: usize = 2048;

fn env_path(key: &str) -> Option<PathBuf> {
    std::env::var_os(key).map(PathBuf::from)
}

/// The oracle's goldens, or None (printed).
fn goldens() -> Option<PathBuf> {
    let d = env_path("GLM53F_GOLDENS").unwrap_or_else(|| PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../oracle/goldens"));
    if d.join("layer03-prefill/prefill.ffn_norm.bin").is_file() && d.join("layer04-prefill/prefill.moe.routed_out.bin").is_file() {
        Some(d)
    } else {
        eprintln!("skip: no layer-3/4 golden payloads under {} (GLM53F_GOLDENS)", d.display());
        None
    }
}

fn read4(path: &std::path::Path) -> Vec<[u8; 4]> {
    let b = std::fs::read(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    b.chunks_exact(4).map(|c| [c[0], c[1], c[2], c[3]]).collect()
}

/// One golden set: inputs `[rows][4096]`, top-8 ids and weights, the routed output.
struct Set {
    x: Vec<f32>,
    ids: Vec<i32>,
    w: Vec<f32>,
    want: Vec<f32>,
    rows: usize,
}

fn golden(g: &std::path::Path, layer: u32, kind: &str) -> Set {
    let d = g.join(format!("layer{layer:02}-{kind}"));
    let f = |n: &str| read4(&d.join(format!("{kind}.{n}.bin")));
    let x: Vec<f32> = f("ffn_norm").into_iter().map(f32::from_le_bytes).collect();
    let rows = x.len() / HIDDEN;
    Set {
        x,
        ids: f("moe.topk_ids").into_iter().map(i32::from_le_bytes).collect(),
        w: f("moe.topk_weights").into_iter().map(f32::from_le_bytes).collect(),
        want: f("moe.routed_out").into_iter().map(f32::from_le_bytes).collect(),
        rows,
    }
}

/// The rank daemons, killed when dropped. Their stderr is echoed and its `exchange` trace
/// lines kept.
struct Daemons {
    children: Vec<Option<Child>>,
    addrs: Vec<String>,
    trace: Arc<Mutex<Vec<(usize, String)>>>,
}

impl Daemons {
    fn kill(&mut self, r: usize) {
        if let Some(mut c) = self.children[r].take() {
            let _ = c.kill();
            let _ = c.wait();
        }
    }
}

impl Drop for Daemons {
    fn drop(&mut self) {
        for r in 0..self.children.len() {
            self.kill(r);
        }
    }
}

fn spawn(bin: &PathBuf, dirs: &[PathBuf]) -> Daemons {
    // Four free loopback ports for the peer mesh (bound, noted, released).
    let peers: Vec<String> = (0..SPARKS)
        .map(|_| TcpListener::bind("127.0.0.1:0").unwrap().local_addr().unwrap().to_string())
        .collect();
    let trace = Arc::new(Mutex::new(Vec::new()));
    let mut d = Daemons { children: Vec::new(), addrs: Vec::new(), trace: trace.clone() };
    for (r, dir) in dirs.iter().enumerate() {
        let mut child = Command::new(bin)
            .args(["serve", "--rank", &r.to_string(), "--dir"])
            .arg(dir)
            .args(["--listen", "127.0.0.1:0", "--allow-partial", "--peers", &peers.join(",")])
            .args(["--peer-timeout-ms", &PEER_TIMEOUT_MS.to_string()])
            .env("GLM53F_WIRE_ALLOW_LAN", "1")
            .env("GLM53F_RANK_TRACE", "1")
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .expect("spawn glm53f-rank");
        let err = BufReader::new(child.stderr.take().unwrap());
        let trace = trace.clone();
        std::thread::spawn(move || {
            for line in err.lines().map_while(Result::ok) {
                if line.starts_with("exchange ") {
                    trace.lock().unwrap().push((r, line));
                } else if !line.starts_with("timing ") && !line.starts_with("ffn ") {
                    eprintln!("rank {r}: {line}");
                }
            }
        });
        d.children.push(Some(child));
    }
    for r in 0..SPARKS {
        let mut out = BufReader::new(d.children[r].as_mut().unwrap().stdout.take().unwrap());
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
        d.addrs.push(addr);
        // Keep draining stdout so the daemon never blocks on it.
        std::thread::spawn(move || for _ in out.lines() {});
    }
    d
}

fn config(return_path: ReturnPath) -> WireConfig {
    WireConfig { return_path, ..WireConfig::glm53_flash() }
}

fn row_sharded(exchange: ExchangeDtype) -> WireConfig {
    config(ReturnPath::RowSharded { min_rows: 16, exchange })
}

/// One exchange's routed output `[rows][4096]` (four planes added in rank order in FP32, or the
/// row slices placed), the four BF16 planes when it was four-plane, and its wall time.
struct Got {
    sum: Vec<f32>,
    planes: Option<Vec<Vec<u16>>>,
    ms: f64,
}

fn bf16s(b: &[u8]) -> Vec<u16> {
    b.chunks_exact(2).map(|c| u16::from_le_bytes([c[0], c[1]])).collect()
}

fn exchange(client: &mut WireClient, layer: u32, x: &[f32], ids: &[i32], w: &[f32]) -> Got {
    let rows = x.len() / HIDDEN;
    let q = quantize_hidden_batched(x).unwrap();
    let payload: Vec<u8> = q.iter().flat_map(|h| h.payload.iter().copied()).collect();
    let scales: Vec<u8> = q.iter().flat_map(|h| h.scales.iter().copied()).collect();
    let routes: Vec<(u32, f32)> = ids.iter().zip(w).map(|(&e, &g)| (e as u32, g)).collect();
    let t = Instant::now();
    client.moe_send_raw(layer, &payload, &scales, &routes, TOPK).expect("send");
    let (l, n) = client.moe_recv_raw().expect("recv");
    let ms = t.elapsed().as_secs_f64() * 1e3;
    assert_eq!((l, n), (layer, rows));
    match client.collected().expect("collected") {
        Collected::Planes(p) => {
            let planes: Vec<Vec<u16>> = p.iter().map(|b| bf16s(b)).collect();
            let sum = (0..rows * HIDDEN).map(|i| planes.iter().fold(0f32, |s, pl| s + bf16_to_f32(pl[i]))).collect();
            Got { sum, planes: Some(planes), ms }
        }
        Collected::RowSlices(s) => {
            let mut sum = vec![f32::NAN; rows * HIDDEN];
            for slice in s.iter() {
                for (i, c) in bf16s(slice.bytes).into_iter().enumerate() {
                    sum[slice.first * HIDDEN + i] = bf16_to_f32(c);
                }
            }
            assert!(sum.iter().all(|v| !v.is_nan()), "the row slices cover every row");
            Got { sum, planes: None, ms }
        }
    }
}

/// Row-sharded against four planes within the bound (module docs). Returns (RMS of the
/// difference / RMS of the four-plane sum, the worst difference / bound).
fn within_bound(rs: &[f32], four: &Got, dtype: ExchangeDtype) -> (f64, f64) {
    let planes = four.planes.as_ref().expect("four planes");
    let rows = rs.len() / HIDDEN;
    let (mut d2, mut s2, mut worst) = (0f64, 0f64, 0f64);
    for t in 0..rows {
        let q = (0..SPARKS).find(|&r| {
            let (f, c) = row_partition(rows, SPARKS, r);
            (f..f + c).contains(&t)
        });
        let q = q.unwrap();
        let amax: Vec<f64> =
            planes.iter().map(|p| p[t * HIDDEN..(t + 1) * HIDDEN].iter().fold(0f64, |m, &c| m.max(bf16_to_f32(c).abs() as f64))).collect();
        for i in 0..HIDDEN {
            let e = t * HIDDEN + i;
            let b: Vec<f64> = planes.iter().map(|p| bf16_to_f32(p[e]) as f64).collect();
            let s = four.sum[e] as f64;
            let own = b[q].abs() * 2f64.powi(-8);
            let exch: f64 = match dtype {
                ExchangeDtype::Bf16 => 0.0,
                ExchangeDtype::Fp8RowScaled => (0..SPARKS)
                    .filter(|&r| r != q)
                    .map(|r| {
                        let v = b[r].abs() * (1.0 + 2f64.powi(-8));
                        let scale = amax[r] * (1.0 + 2f64.powi(-8)) / 448.0;
                        (v * 2f64.powi(-4)).max(scale * 2f64.powi(-10)) + v * 2f64.powi(-22) + b[r].abs() * 2f64.powi(-8)
                    })
                    .sum(),
            };
            let fp32 = 8.0 * 2f64.powi(-24) * b.iter().map(|v| v.abs()).sum::<f64>();
            let bound = own + exch + (s.abs() + own + exch) * 2f64.powi(-8) + fp32 + 1e-30;
            let d = (rs[e] as f64 - s).abs();
            assert!(d <= bound, "{dtype:?}: row {t} element {i}: row-sharded {} vs four planes {s} (bound {bound:.3e})", rs[e]);
            worst = worst.max(d / bound);
            d2 += d * d;
            s2 += s * s;
        }
    }
    ((d2 / s2).sqrt(), worst)
}

/// (worst row cosine, mean relative RMS) against the oracle's routed output.
fn vs_oracle(got: &[f32], want: &[f32]) -> (f64, f64) {
    let rows = want.len() / HIDDEN;
    let (mut worst, mut mean) = (1f64, 0f64);
    for r in 0..rows {
        let (a, b) = (&got[r * HIDDEN..(r + 1) * HIDDEN], &want[r * HIDDEN..(r + 1) * HIDDEN]);
        let (mut ab, mut aa, mut bb, mut dd) = (0f64, 0f64, 0f64, 0f64);
        for (&x, &y) in a.iter().zip(b) {
            let (x, y) = (x as f64, y as f64);
            ab += x * y;
            aa += x * x;
            bb += y * y;
            dd += (x - y) * (x - y);
        }
        worst = worst.min(ab / (aa * bb).sqrt());
        mean += (dd / bb).sqrt() / rows as f64;
    }
    (worst, mean)
}

/// Synthetic rows like a post-norm MoE input, and top-8 routes of distinct experts.
fn synthetic(rows: usize) -> (Vec<f32>, Vec<i32>, Vec<f32>) {
    let mut s = 0x9E37_79B9_7F4A_7C15u64;
    let mut next = move || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        (s >> 11) as f64 / (1u64 << 53) as f64
    };
    let x = (0..rows * HIDDEN).map(|i| ((next() - 0.5) * 3.4 * if i % 509 == 17 { 12.0 } else { 1.0 }) as f32).collect();
    let ids = (0..rows * TOPK).map(|i| ((i % TOPK) * 36 + (i / TOPK * 7) % 36) as i32).collect();
    (x, ids, vec![2.5 / TOPK as f32; rows * TOPK])
}

/// A request sent straight to one rank on a new connection, and its return (bounded wait).
fn raw_exchange(addr: &str, rank: usize, set: &Set, flags: u32) -> Result<(ReturnFrame, Duration), String> {
    let f = RequestFrame {
        request_id: 0x0DEA_D000 + rank as u64,
        placement_version: 1,
        layer_id: 3,
        executor_id: rank as u64,
        source_kind: SourceKind::Decode,
        token_position: 0,
        flags,
        seq: 0,
        rows: (0..set.rows)
            .map(|t| RowDescriptor {
                row_id: t as u64,
                source_kind: SourceKind::Decode,
                source_request_id: 0,
                token_position: t as u64,
                route_offset: (t * TOPK) as u32,
                route_count: TOPK as u32,
            })
            .collect(),
        routes: (0..set.rows * TOPK)
            .map(|i| RouteEntry { row_index: (i / TOPK) as u32, expert_id: set.ids[i] as u32, gate_weight: set.w[i] })
            .collect(),
        hidden_rows: quantize_hidden_batched(&set.x).unwrap(),
    };
    let bytes = encode_request(&f, WireNaive::NONE).map_err(|e| e.to_string())?;
    let t = Instant::now();
    let mut s = TcpStream::connect(addr).map_err(|e| e.to_string())?;
    s.set_read_timeout(Some(Duration::from_secs(60))).map_err(|e| e.to_string())?;
    s.write_all(&bytes).map_err(|e| e.to_string())?;
    let mut h = vec![0u8; glm53f_wire::HEADER_LEN];
    s.read_exact(&mut h).map_err(|e| format!("rank {rank}: {e}"))?;
    let wb = u64::from_le_bytes(h[76..84].try_into().unwrap()) as usize;
    h.resize(wb, 0);
    s.read_exact(&mut h[glm53f_wire::HEADER_LEN..]).map_err(|e| format!("rank {rank}: {e}"))?;
    match decode_frame(&h, WireNaive::NONE).map_err(|e| e.to_string())? {
        Frame::Return(r) => Ok((r, t.elapsed())),
        Frame::Request(_) => Err("a request frame from a rank".into()),
    }
}

#[test]
fn row_sharded_returns_on_four_rank_daemons() {
    let Some(bin) = env_path("GLM53F_RANK_BIN").filter(|p| p.is_file()) else {
        eprintln!("skip: GLM53F_RANK_BIN does not name a glm53f-rank binary");
        return;
    };
    let dirs: Vec<PathBuf> = std::env::var("GLM53F_RANK_DIRS").unwrap_or_default().split(',').filter(|s| !s.is_empty()).map(PathBuf::from).collect();
    if dirs.len() != SPARKS || !dirs.iter().all(|d| d.join("manifest.txt").is_file()) {
        eprintln!("skip: GLM53F_RANK_DIRS must name the four rank directories (with manifest.txt)");
        return;
    }
    let Some(g) = goldens() else { return };
    let mut daemons = spawn(&bin, &dirs);

    // 1. Both ways, and the oracle.
    let sets: Vec<(u32, Set)> = [3u32, 4].iter().map(|&l| (l, golden(&g, l, "prefill"))).collect();
    let mut four = Vec::new();
    {
        let mut c = WireClient::connect(&daemons.addrs, config(ReturnPath::FourPlaneSum)).expect("connect");
        for (l, s) in &sets {
            four.push(exchange(&mut c, *l, &s.x, &s.ids, &s.w));
        }
    }
    for dtype in [ExchangeDtype::Bf16, ExchangeDtype::Fp8RowScaled] {
        let mut c = WireClient::connect(&daemons.addrs, row_sharded(dtype)).expect("connect");
        for ((l, s), fp) in sets.iter().zip(&four) {
            let rs = exchange(&mut c, *l, &s.x, &s.ids, &s.w);
            assert!(rs.planes.is_none(), "{} rows are reduce-scattered", s.rows);
            let (rel, worst) = within_bound(&rs.sum, fp, dtype);
            let (cos_fp, rms_fp) = vs_oracle(&fp.sum, &s.want);
            let (cos_rs, rms_rs) = vs_oracle(&rs.sum, &s.want);
            eprintln!(
                "layer {l}, {} rows, {} exchange: row-sharded vs four planes: RMS {rel:.2e} of the sum's RMS, worst {worst:.2} of the bound; \
                 vs the oracle: four planes cosine >= {cos_fp:.4}, mean rel RMS {:.2}%; row-sharded cosine >= {cos_rs:.4}, mean rel RMS {:.2}%",
                s.rows,
                dtype.name(),
                100.0 * rms_fp,
                100.0 * rms_rs
            );
            assert!(cos_fp > 0.98 && cos_rs > 0.98, "layer {l}: the oracle's routed output");
        }
    }

    // 2. Decode stays four-plane under the row-sharded configuration, bit for bit.
    let dec = golden(&g, 3, "decode");
    let a = {
        let mut c = WireClient::connect(&daemons.addrs, config(ReturnPath::FourPlaneSum)).expect("connect");
        exchange(&mut c, 3, &dec.x, &dec.ids, &dec.w)
    };
    let b = {
        let mut c = WireClient::connect(&daemons.addrs, row_sharded(ExchangeDtype::Bf16)).expect("connect");
        exchange(&mut c, 3, &dec.x, &dec.ids, &dec.w)
    };
    assert!(b.planes.is_some(), "8 rows stay four-plane");
    assert!(a.sum == b.sum && a.planes == b.planes, "decode: the same planes under both configurations");
    eprintln!("layer 3 decode, {} rows: four-plane under the row-sharded configuration, bit for bit", dec.rows);

    // 3. Loopback timings (four daemons on one GPU; not the target hardware).
    let (x, ids, w) = synthetic(TIMING_ROWS);
    for (name, cfg) in [("four planes", config(ReturnPath::FourPlaneSum)), ("row-sharded bf16", row_sharded(ExchangeDtype::Bf16))] {
        daemons.trace.lock().unwrap().clear();
        let mut c = WireClient::connect(&daemons.addrs, cfg).expect("connect");
        exchange(&mut c, 3, &x, &ids, &w); // warm-up: scratch growth, the mesh's first frames
        let ms: Vec<f64> = (0..5).map(|_| exchange(&mut c, 3, &x, &ids, &w).ms).collect();
        let into = if name == "four planes" { SPARKS * TIMING_ROWS * 8192 } else { TIMING_ROWS * 8192 } + SPARKS * 128;
        eprintln!(
            "loopback, {TIMING_ROWS} rows, {name}: exchange wall {:.1} ms mean, {:.1} ms min over 5; {:.1} MB into the coordinator",
            ms.iter().sum::<f64>() / 5.0,
            ms.iter().cloned().fold(f64::MAX, f64::min),
            into as f64 / 1e6
        );
        std::thread::sleep(Duration::from_millis(200));
        let trace = daemons.trace.lock().unwrap();
        for r in 0..SPARKS {
            let lines: Vec<&String> = trace.iter().filter(|(q, l)| *q == r && l.contains(&format!("rows={TIMING_ROWS}"))).map(|(_, l)| l).collect();
            if let Some(last) = lines.last() {
                eprintln!("  rank {r} ({} exchanges), last: {last}", lines.len());
            }
        }
    }

    // 4. A dead peer: rank 3 is killed; ranks 0-2 answer with an error return within the timeout.
    daemons.kill(3);
    let set = &sets[0].1;
    let results: Vec<Result<(ReturnFrame, Duration), String>> = std::thread::scope(|s| {
        let hs: Vec<_> = (0..3)
            .map(|r| {
                let addr = daemons.addrs[r].clone();
                s.spawn(move || raw_exchange(&addr, r, set, ExchangeDtype::Bf16.request_flags()))
            })
            .collect();
        hs.into_iter().map(|h| h.join().unwrap()).collect()
    });
    for (r, res) in results.into_iter().enumerate() {
        let (ret, t) = res.unwrap_or_else(|e| panic!("rank {r}: no answer: {e}"));
        eprintln!("dead peer: rank {r} answered status {:?} after {:.2} s", ret.status, t.as_secs_f64());
        assert_eq!(ret.status, Status::Error, "rank {r} must fail the request, not sum without rank 3");
        assert!(t < Duration::from_millis(PEER_TIMEOUT_MS) + Duration::from_secs(10), "rank {r} took {t:?}");
    }
}
