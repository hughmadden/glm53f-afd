//! The expert exchange alone, four-plane and row-sharded (the prefill reduce-scatter), against
//! four running rank daemons: per-exchange wall time from the first byte written to the last
//! return read, and the bytes the coordinator receives, on the same synthetic rows both ways.
//! It measures the return path before the forward uses it: no model weights on this side.
//!
//! ```text
//! wire_bench --ranks A,B,C,D [--rows 4096] [--layer 3] [--iters 20] [--min-rows 16]
//!            [--exchange bf16|fp8] [--mode both|four|sharded] [--inflight 1..4]
//! ```
//!
//! `--ranks` falls back to `GLM53F_SPARK_ADDRS`. Over RDMA with `GLM53F_RDMA=1` and
//! `GLM53F_WIRE_NOCRC=1` (an `rdma` build), as the coordinator runs; `--inflight N` keeps N
//! exchanges in flight (N prefill lanes; RDMA only, at most what the ranks queue: their
//! `--recv-slots`). The ranks need `--peers` for the
//! row-sharded mode. With both modes it also prints the RMS difference of the two outputs of
//! the first exchange (the row-sharded return's error against the four-plane sum).

use std::collections::VecDeque;
use std::time::Instant;

use glm53f_coordinator::wire::{quantize_hidden_batched, Collected, ReturnPath, WireClient, WireConfig};
use glm53f_wire::bf16::bf16_to_f32;
use glm53f_wire::row_shard::ExchangeDtype;
use glm53f_wire::{HIDDEN, SPARKS};

const TOPK: usize = 8;

struct Args {
    ranks: Vec<String>,
    rows: usize,
    layer: u32,
    iters: usize,
    min_rows: usize,
    exchange: ExchangeDtype,
    modes: Vec<bool>,
    inflight: usize,
}

fn parse() -> Result<Args, String> {
    let mut a = Args {
        ranks: std::env::var("GLM53F_SPARK_ADDRS").unwrap_or_default().split(',').filter(|s| !s.is_empty()).map(String::from).collect(),
        rows: 4096,
        layer: 3,
        iters: 20,
        min_rows: 16,
        exchange: ExchangeDtype::Bf16,
        modes: vec![false, true],
        inflight: 1,
    };
    let mut it = std::env::args().skip(1);
    while let Some(k) = it.next() {
        let v = it.next().ok_or(format!("{k} needs a value"))?;
        let num = |v: &str| v.parse::<usize>().map_err(|_| format!("{k}: {v} is not a number"));
        match k.as_str() {
            "--ranks" => a.ranks = v.split(',').map(String::from).collect(),
            "--rows" => a.rows = num(&v)?,
            "--layer" => a.layer = num(&v)? as u32,
            "--iters" => a.iters = num(&v)?,
            "--min-rows" => a.min_rows = num(&v)?,
            "--exchange" => a.exchange = ExchangeDtype::parse(&v).ok_or("--exchange: bf16 or fp8")?,
            "--mode" => {
                a.modes = match v.as_str() {
                    "both" => vec![false, true],
                    "four" => vec![false],
                    "sharded" => vec![true],
                    _ => return Err("--mode: both, four or sharded".into()),
                }
            }
            "--inflight" => a.inflight = num(&v)?.clamp(1, glm53f_coordinator::wire::MAX_DEPTH),
            _ => return Err(format!("unknown argument {k}")),
        }
    }
    if a.ranks.len() != SPARKS || a.rows == 0 || a.rows > 4096 || a.iters == 0 {
        return Err("need --ranks A,B,C,D (or GLM53F_SPARK_ADDRS), 1..=4096 --rows and --iters > 0".into());
    }
    Ok(a)
}

/// Synthetic rows like a post-norm MoE input (FP8 payload and scales) and top-8 routes of
/// distinct experts.
fn inputs(rows: usize) -> (Vec<u8>, Vec<u8>, Vec<(u32, f32)>) {
    let mut s = 0x9E37_79B9_7F4A_7C15u64;
    let mut next = move || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        (s >> 11) as f64 / (1u64 << 53) as f64
    };
    let x: Vec<f32> = (0..rows * HIDDEN).map(|i| ((next() - 0.5) * 3.4 * if i % 509 == 17 { 12.0 } else { 1.0 }) as f32).collect();
    let q = quantize_hidden_batched(&x).expect("quantize");
    let routes = (0..rows * TOPK).map(|i| (((i % TOPK) * 36 + (i / TOPK * 7) % 36) as u32, 2.5 / TOPK as f32)).collect();
    (q.iter().flat_map(|h| h.payload.clone()).collect(), q.iter().flat_map(|h| h.scales.clone()).collect(), routes)
}

/// The routed output of the last collected exchange, `[rows][4096]`.
fn output(c: &WireClient, rows: usize) -> Vec<f32> {
    let mut out = vec![0f32; rows * HIDDEN];
    match c.collected().expect("an exchange was collected") {
        Collected::Planes(p) => {
            for plane in p.iter() {
                for (o, b) in out.iter_mut().zip(plane.chunks_exact(2)) {
                    *o += bf16_to_f32(u16::from_le_bytes([b[0], b[1]]));
                }
            }
        }
        Collected::RowSlices(s) => {
            for slice in s.iter() {
                for (o, b) in out[slice.first * HIDDEN..].iter_mut().zip(slice.bytes.chunks_exact(2)) {
                    *o = bf16_to_f32(u16::from_le_bytes([b[0], b[1]]));
                }
            }
        }
    }
    out
}

fn main() {
    let a = parse().unwrap_or_else(|e| {
        eprintln!("wire_bench: {e}");
        std::process::exit(2);
    });
    let (payload, scales, routes) = inputs(a.rows);
    let mut first: Vec<Vec<f32>> = Vec::new();
    for &sharded in &a.modes {
        let return_path = if sharded { ReturnPath::RowSharded { min_rows: a.min_rows, exchange: a.exchange } } else { ReturnPath::FourPlaneSum };
        let cfg = WireConfig { return_path, depth: a.inflight, ..WireConfig::glm53_flash() };
        let mut c = WireClient::connect(&a.ranks, cfg).unwrap_or_else(|e| {
            eprintln!("wire_bench: {e}");
            std::process::exit(1);
        });
        let inflight = c.depth();
        let name = if sharded && c.config().row_sharded(a.rows).is_some() { format!("row-sharded {}", a.exchange.name()) } else { "four planes".into() };
        // One exchange collected (warms the kernels' scratch and the peer mesh), kept for the comparison.
        c.moe_send_raw(a.layer, &payload, &scales, &routes, TOPK).expect("send");
        c.moe_recv_raw().expect("recv");
        first.push(output(&c, a.rows));
        let (mut ms, mut sent) = (Vec::with_capacity(a.iters), VecDeque::new());
        let t0 = Instant::now();
        for _ in 0..a.iters {
            while sent.len() < inflight {
                c.moe_send_raw(a.layer, &payload, &scales, &routes, TOPK).expect("send");
                sent.push_back(Instant::now());
            }
            c.moe_recv_raw().expect("recv");
            ms.push(sent.pop_front().expect("in flight").elapsed().as_secs_f64() * 1e3);
        }
        while sent.pop_front().is_some() {
            c.moe_recv_raw().expect("recv");
        }
        let wall = t0.elapsed().as_secs_f64();
        ms.sort_by(|x, y| x.total_cmp(y));
        let into = SPARKS * glm53f_wire::HEADER_LEN
            + if c.config().row_sharded(a.rows).is_some() { 1 } else { SPARKS } * a.rows * glm53f_wire::RETURN_ROW_BYTES;
        println!(
            "{name}: {} rows x {} exchanges ({inflight} in flight): per exchange median {:.2} ms, min {:.2}, p90 {:.2}, max {:.2}; \
             {:.0} rows/s; {:.2} MB into the coordinator per exchange ({:.0} bytes a row)",
            a.rows,
            a.iters,
            ms[ms.len() / 2],
            ms[0],
            ms[ms.len() * 9 / 10],
            ms[ms.len() - 1],
            (a.iters * a.rows) as f64 / wall,
            into as f64 / 1e6,
            into as f64 / a.rows as f64
        );
    }
    if let [four, rs] = &first[..] {
        let (mut d, mut s) = (0f64, 0f64);
        for (x, y) in rs.iter().zip(four) {
            d += ((x - y) as f64).powi(2);
            s += (*y as f64).powi(2);
        }
        println!("row-sharded vs four planes (first exchange): RMS difference {:.2e} of the four-plane sum's RMS", (d / s).sqrt());
    }
}
