//! Layers 0-4 of GLM-5.3-Flash against the oracle's goldens (feature `cuda`).
//!
//! The engine runs in its production dtypes (BF16 activations, BF16 KDA and indexer
//! projections, FP8 DSA / dense / shared / routed projections, the FP8 MLA latent and pooled
//! keys) against the oracle's FP32 contract, and the oracle's own BF16 run (`native`) is the
//! yardstick: it is the reference's rounding alone.
//!
//! 1. **Chain**: the 33-token prompt as one prefill, then the 8 fixed decode tokens one step
//!    each, through layers 0-4 and the head, as a serving forward runs them (errors accumulate
//!    from layer to layer). Taps record every layer's boundaries and sublayer internals.
//! 2. **Per layer**: each recorded layer (0, 3, 4) fed its golden FP32 input streams cast to
//!    BF16, with a fresh cache: the same setup as the native set.
//!
//! MoE layers use the golden routing for the expert path (the oracle README: the 8th and 9th
//! routing scores are a median 0.002 apart); the engine's own routing is compared with a
//! near-tie allowance.
//!
//! ```sh
//! GLM53F_CHECKPOINT_DIR=... GLM53F_EXPERTS_DIR=... \
//!   cargo test --release -p glm53f-forward --features cuda --test goldens_chain -- --nocapture
//! ```
#![cfg(feature = "cuda")]

mod common;

use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex};

use common::*;
use glm53f_dsa::cache::{
    decode_index_key, decode_latent, LATENT_RECORD_BYTES, PAGE_POOL_CODES_OFFSET,
    PAGE_POOL_SCALES_OFFSET,
};
use glm53f_forward::forward::{ForwardConfig, Tap, TapBuf, TapPoint};
use glm53f_forward::shape::*;

const LAYERS: usize = 5;
const PROMPT: usize = 33;
const STEPS: usize = 8;

/// Tensors recorded by the tap, per (layer, point, buffer): one entry per pass.
type Record = HashMap<(usize, TapPoint, TapBuf), Vec<Vec<f32>>>;

fn record_tap(
    rec: Arc<Mutex<Record>>,
    layers: &'static [usize],
) -> Box<glm53f_forward::forward::TapFn> {
    Box::new(move |t: &Tap<'_>| {
        if !layers.contains(&t.layer) {
            return Ok(());
        }
        let kda = t.layer % 4 != 3;
        let moe = t.layer >= 3;
        let mut bufs: Vec<TapBuf> = Vec::new();
        match t.point {
            TapPoint::AttnDone => {
                bufs.extend([
                    TapBuf::Streams,
                    TapBuf::Pre,
                    TapBuf::Post,
                    TapBuf::Comb,
                    TapBuf::Collapsed,
                    TapBuf::Normed,
                    TapBuf::Out,
                ]);
                if kda {
                    bufs.extend([TapBuf::KdaP, TapBuf::KdaGates, TapBuf::KdaNormOut]);
                } else {
                    bufs.extend([
                        TapBuf::DsaQResid,
                        TapBuf::DsaQ,
                        TapBuf::DsaKvA,
                        TapBuf::DsaIdxQ,
                        TapBuf::DsaIdxProj,
                        TapBuf::DsaTokens,
                        TapBuf::DsaCounts,
                        TapBuf::DsaHeads,
                    ]);
                }
            }
            TapPoint::FfnDone => {
                bufs.extend([
                    TapBuf::Streams,
                    TapBuf::Pre,
                    TapBuf::Post,
                    TapBuf::Comb,
                    TapBuf::Collapsed,
                    TapBuf::Normed,
                    TapBuf::Out,
                ]);
                if moe {
                    bufs.extend([
                        TapBuf::SharedOut,
                        TapBuf::RouterLogits,
                        TapBuf::RouterIds,
                        TapBuf::RouterWeights,
                    ]);
                }
            }
            TapPoint::LayerOut => bufs.push(TapBuf::Streams),
        }
        let mut r = rec.lock().unwrap();
        for b in bufs {
            let v: Vec<f32> = match b {
                TapBuf::Pre
                | TapBuf::Post
                | TapBuf::Comb
                | TapBuf::DsaQ
                | TapBuf::DsaIdxQ
                | TapBuf::DsaHeads
                | TapBuf::RouterLogits
                | TapBuf::RouterWeights => t.f32(b)?,
                TapBuf::DsaTokens | TapBuf::DsaCounts | TapBuf::RouterIds => {
                    t.i32(b)?.into_iter().map(|x| x as f32).collect()
                }
                _ => widen(&t.bf16(b)?),
            };
            r.entry((t.layer, t.point, b)).or_default().push(v);
        }
        Ok(())
    })
}

/// The recorded passes of one buffer, concatenated (prefill rows, then each decode row).
fn all(rec: &Record, layer: usize, p: TapPoint, b: TapBuf) -> Vec<f32> {
    rec[&(layer, p, b)].concat()
}

struct Table {
    rows: Vec<String>,
}

impl Table {
    fn new() -> Table {
        Table { rows: Vec::new() }
    }
    fn add(&mut self, what: &str, prefill: Err, decode: Err, native: Option<(f64, f64)>) {
        let nat = native.map_or("        -          -".to_string(), |(p, d)| {
            format!("{p:>9.2e} {d:>10.2e}")
        });
        self.rows.push(format!(
            "{what:<34} {:>9.2e} {:>10.2e}  {nat}   {:>9.2e}",
            prefill.rel_rms,
            decode.rel_rms,
            prefill.max_abs.max(decode.max_abs)
        ));
    }
    fn print(&self, title: &str) {
        eprintln!("\n{title}");
        eprintln!(
            "{:<34} {:>9} {:>10}  {:>9} {:>10}   {:>9}",
            "tensor (relative RMS vs FP32)",
            "prefill",
            "decode",
            "native pf",
            "native dec",
            "max abs"
        );
        for r in &self.rows {
            eprintln!("{r}");
        }
    }
}

/// Golden `name` of a layer's prefill and decode sets.
fn gold(g: &Goldens, layer: usize, name: &str) -> (Vec<f32>, Vec<f32>) {
    (
        g.f32(
            &format!("layer{layer:02}-prefill"),
            &format!("prefill.{name}"),
        ),
        g.f32(
            &format!("layer{layer:02}-decode"),
            &format!("decode.{name}"),
        ),
    )
}

/// Native rows `[0, 33)` and `[33, 41)` of `LNN.<name>`.
fn native(g: &Goldens, layer: usize, name: &str) -> Option<(Vec<f32>, Vec<f32>)> {
    let key = format!("L{layer:02}.{name}");
    if !g.has("native", &key) {
        return None;
    }
    let v = g.f32("native", &key);
    let w = v.len() / (PROMPT + STEPS);
    Some((v[..PROMPT * w].to_vec(), v[PROMPT * w..].to_vec()))
}

fn native_err(g: &Goldens, layer: usize, name: &str, golden_name: &str) -> Option<(f64, f64)> {
    let (np, nd) = native(g, layer, name)?;
    let (gp, gd) = gold(g, layer, golden_name);
    Some((err(&np, &gp).rel_rms, err(&nd, &gd).rel_rms))
}

/// Selected-token sets per row: the valid entries of each row, sorted.
fn token_sets(v: &[f32], width: usize, counts: Option<&[f32]>) -> Vec<Vec<i64>> {
    v.chunks_exact(width)
        .enumerate()
        .map(|(r, row)| {
            let n = counts.map_or(width, |c| c[2 * r + 1] as usize);
            let mut s: Vec<i64> = row[..n]
                .iter()
                .map(|&x| x as i64)
                .filter(|&x| x >= 0)
                .collect();
            s.sort_unstable();
            s
        })
        .collect()
}

fn golden_routes(g: &Goldens, layers: &[usize]) -> RouteQueue {
    let mut q = HashMap::new();
    for &l in layers {
        let mut d = VecDeque::new();
        let (sp, sd) = (
            format!("layer{l:02}-prefill"),
            format!("layer{l:02}-decode"),
        );
        let ids =
            |s: &str, n: &str| -> Vec<i32> { g.i64(s, n).into_iter().map(|x| x as i32).collect() };
        d.push_back((
            ids(&sp, "prefill.moe.topk_ids"),
            g.f32(&sp, "prefill.moe.topk_weights"),
        ));
        let (di, dw) = (
            ids(&sd, "decode.moe.topk_ids"),
            g.f32(&sd, "decode.moe.topk_weights"),
        );
        for s in 0..STEPS {
            d.push_back((
                di[s * TOP_K..(s + 1) * TOP_K].to_vec(),
                dw[s * TOP_K..(s + 1) * TOP_K].to_vec(),
            ));
        }
        q.insert(l, d);
    }
    q
}

/// How the engine's routing compares with the golden one: rows choosing the same experts, rows
/// that differ, and the largest gap, under the golden scores (sigmoid + correction bias),
/// between a swapped expert and the golden 8th choice (a near-tie has a small gap).
fn routing_agreement(
    ours: &[i32],
    golden_ids: &[i64],
    golden_logits: &[f32],
    bias: &[f32],
) -> (usize, usize, f32) {
    let rows = ours.len() / TOP_K;
    let (mut same, mut differ, mut gap) = (0, 0, 0f32);
    for r in 0..rows {
        let mut a: Vec<i32> = ours[r * TOP_K..(r + 1) * TOP_K].to_vec();
        let mut b: Vec<i32> = golden_ids[r * TOP_K..(r + 1) * TOP_K]
            .iter()
            .map(|&x| x as i32)
            .collect();
        a.sort_unstable();
        b.sort_unstable();
        if a == b {
            same += 1;
            continue;
        }
        let score = |e: i32| -> f32 {
            let l = golden_logits[r * EXPERTS + e as usize];
            1.0 / (1.0 + (-l).exp()) + bias[e as usize]
        };
        let eighth = b.iter().map(|&e| score(e)).fold(f32::INFINITY, f32::min);
        for &e in a
            .iter()
            .filter(|e| !b.contains(e))
            .chain(b.iter().filter(|e| !a.contains(e)))
        {
            gap = gap.max((score(e) - eighth).abs());
        }
        differ += 1;
    }
    (same, differ, gap)
}

/// What one run of layers 0, 3 and 4 produced: the tap records, and the KV after the prompt
/// and after the last step (KDA states and conv windows of layers 0 and 4, layer 3's page).
struct Run {
    rec: Record,
    state: [[Vec<f32>; 2]; 2],
    conv: [[Vec<u16>; 2]; 2],
    page: [Vec<u8>; 2],
    /// Logits of the last prompt row and each step (chain runs).
    logits: Option<(Vec<f32>, Vec<u32>)>,
}

/// The chain: the prompt as one prefill (in chunks of `chunk` rows), then the 8 steps.
fn run_chain(su: &mut Setup, prompt: &[u32], steps: &[u32], chunk: usize) -> Run {
    let rec: Arc<Mutex<Record>> = Arc::new(Mutex::new(HashMap::new()));
    su.fwd.cfg.max_rows = chunk;
    su.fwd.set_tap(Some(record_tap(rec.clone(), &[0, 3, 4])));
    let mut kv = su.fwd.kv.slot().unwrap();
    kv.reserve(PROMPT + STEPS).unwrap();
    let mut logits = Vec::new();
    let mut picks = su.fwd.prefill(&mut [(&mut kv, prompt)]).unwrap();
    logits.extend(su.fwd.logits(1).unwrap());
    let snap = |kv: &glm53f_forward::kv::GlmKv| {
        (
            [kv.download_state(0).unwrap(), kv.download_state(3).unwrap()],
            [kv.download_conv(0).unwrap(), kv.download_conv(3).unwrap()],
            kv.download_page_block(0, 0).unwrap(),
        )
    };
    let (s0, c0, p0) = snap(&kv);
    for &t in steps {
        picks.extend(su.fwd.decode(&mut [(&mut kv, t)]).unwrap());
        logits.extend(su.fwd.logits(1).unwrap());
    }
    su.fwd.set_tap(None);
    su.fwd.cfg.max_rows = 64;
    assert_eq!(kv.tokens(), PROMPT + STEPS);
    let (s1, c1, p1) = snap(&kv);
    let rec = rec.lock().unwrap().clone();
    Run {
        rec,
        state: [s0, s1],
        conv: [c0, c1],
        page: [p0, p1],
        logits: Some((logits, picks)),
    }
}

/// Each layer from its golden input streams (cast to BF16) with a fresh cache: the prompt,
/// then each step's golden input.
fn run_per_layer(su: &mut Setup, g: &Goldens, chunk: usize) -> Run {
    let rec: Arc<Mutex<Record>> = Arc::new(Mutex::new(HashMap::new()));
    su.fwd.set_tap(Some(record_tap(rec.clone(), &[0, 3, 4])));
    let mut state: [[Vec<f32>; 2]; 2] = Default::default();
    let mut conv: [[Vec<u16>; 2]; 2] = Default::default();
    let mut page: [Vec<u8>; 2] = Default::default();
    for &l in &[0usize, 3, 4] {
        let mut kv = su.fwd.kv.slot().unwrap();
        kv.reserve(PROMPT + STEPS).unwrap();
        let (ip, id) = gold(g, l, "in_streams");
        for c in (0..PROMPT).step_by(chunk) {
            let n = chunk.min(PROMPT - c);
            su.fwd
                .run_layers(
                    &mut [&mut kv],
                    &[n],
                    &narrow(&rows(&ip, HC * HIDDEN, c, c + n)),
                    l..l + 1,
                )
                .unwrap();
        }
        let mut snap = |kv: &glm53f_forward::kv::GlmKv, phase: usize| match l {
            0 | 4 => {
                let (i, j) = if l == 0 { (0, 0) } else { (1, 3) };
                state[phase][i] = kv.download_state(j).unwrap();
                conv[phase][i] = kv.download_conv(j).unwrap();
            }
            _ => page[phase] = kv.download_page_block(0, 0).unwrap(),
        };
        snap(&kv, 0);
        for s in 0..STEPS {
            let x = narrow(&rows(&id, HC * HIDDEN, s, s + 1));
            su.fwd
                .run_layers(&mut [&mut kv], &[1], &x, l..l + 1)
                .unwrap();
        }
        snap(&kv, 1);
    }
    su.fwd.set_tap(None);
    let rec = rec.lock().unwrap().clone();
    Run {
        rec,
        state,
        conv,
        page,
        logits: None,
    }
}

/// Bounds a report checks.
struct Summary {
    out_streams: Vec<(usize, f64)>,
    routing_gap: Vec<(usize, f32)>,
    logits: Option<(f64, f64, usize, usize)>,
}

fn report(g: &Goldens, run: &Run, title: &str) -> Summary {
    let rec = &run.rec;
    let mut table = Table::new();
    let split = |v: Vec<f32>, w: usize| -> (Vec<f32>, Vec<f32>) {
        (v[..PROMPT * w].to_vec(), v[PROMPT * w..].to_vec())
    };
    let cmp = |t: &mut Table,
               what: &str,
               ours: Vec<f32>,
               w: usize,
               golden: (Vec<f32>, Vec<f32>),
               nat: Option<(f64, f64)>|
     -> (Err, Err) {
        let (op, od) = split(ours, w);
        let (ep, ed) = (err(&op, &golden.0), err(&od, &golden.1));
        t.add(what, ep, ed, nat);
        (ep, ed)
    };
    let hs = HC * HIDDEN;
    let mut sum = Summary {
        out_streams: Vec::new(),
        routing_gap: Vec::new(),
        logits: None,
    };
    for &l in &[0usize, 3, 4] {
        let kda = l % 4 != 3;
        let a = TapPoint::AttnDone;
        let f = TapPoint::FfnDone;
        cmp(
            &mut table,
            &format!("L{l} in_streams"),
            all(rec, l, a, TapBuf::Streams),
            hs,
            gold(g, l, "in_streams"),
            None,
        );
        for (b, n, w) in [
            (TapBuf::Pre, "pre", 4),
            (TapBuf::Post, "post", 4),
            (TapBuf::Comb, "comb", 16),
        ] {
            cmp(
                &mut table,
                &format!("L{l} attn_hc.{n}"),
                all(rec, l, a, b),
                w,
                gold(g, l, &format!("attn_hc.{n}")),
                None,
            );
        }
        cmp(
            &mut table,
            &format!("L{l} attn_hc.collapsed"),
            all(rec, l, a, TapBuf::Collapsed),
            HIDDEN,
            gold(g, l, "attn_hc.collapsed"),
            None,
        );
        cmp(
            &mut table,
            &format!("L{l} attn_norm"),
            all(rec, l, a, TapBuf::Normed),
            HIDDEN,
            gold(g, l, "attn_norm"),
            None,
        );
        if kda {
            let p = all(rec, l, a, TapBuf::KdaP);
            cmp(
                &mut table,
                &format!("L{l} kda.qkv_preconv"),
                cols(&p, KDA_P_COLS, 0, KDA_QKV),
                KDA_QKV,
                gold(g, l, "kda.qkv_preconv"),
                native_err(g, l, "kda.qkv_preconv", "kda.qkv_preconv"),
            );
            cmp(
                &mut table,
                &format!("L{l} kda.b_logits"),
                cols(&p, KDA_P_COLS, KDA_QKV, KDA_P_COLS),
                KDA_HEADS,
                gold(g, l, "kda.b_logits"),
                native_err(g, l, "kda.b_logits", "kda.b_logits"),
            );
            let gates = all(rec, l, a, TapBuf::KdaGates);
            cmp(
                &mut table,
                &format!("L{l} kda.f_proj"),
                cols(&gates, 2 * KDA_WIDTH, 0, KDA_WIDTH),
                KDA_WIDTH,
                gold(g, l, "kda.f_proj"),
                native_err(g, l, "kda.f_proj", "kda.f_proj"),
            );
            cmp(
                &mut table,
                &format!("L{l} kda.gate"),
                cols(&gates, 2 * KDA_WIDTH, KDA_WIDTH, 2 * KDA_WIDTH),
                KDA_WIDTH,
                gold(g, l, "kda.gate"),
                native_err(g, l, "kda.gate", "kda.gate"),
            );
            cmp(
                &mut table,
                &format!("L{l} kda.norm_out"),
                all(rec, l, a, TapBuf::KdaNormOut),
                KDA_WIDTH,
                gold(g, l, "kda.norm_out"),
                None,
            );
            // KDA state (ours [H][v][k], the reference's [H][k][v]) and the conv window (ours
            // [3][C] time-major, the reference's [C][4] with the last 3 used).
            let i = if l == 0 { 0 } else { 1 };
            let tr = |s: &[f32]| glm53f_kda::cpu::transpose_state(s);
            let (sp, sd) = (
                g.f32(&format!("layer{l:02}-prefill"), "prefill.kda.state"),
                g.f32(&format!("layer{l:02}-decode"), "decode.kda.state_final"),
            );
            table.add(
                &format!("L{l} kda.state"),
                err(&tr(&run.state[0][i]), &sp),
                err(&tr(&run.state[1][i]), &sd),
                None,
            );
            let last3 = |c: &[f32]| -> Vec<f32> {
                (0..3)
                    .flat_map(|t| (0..KDA_QKV).map(move |ch| c[ch * 4 + 1 + t]))
                    .collect::<Vec<f32>>()
            };
            let (cp, cd) = (
                g.f32(&format!("layer{l:02}-prefill"), "prefill.kda.conv_state"),
                g.f32(
                    &format!("layer{l:02}-decode"),
                    "decode.kda.conv_state_final",
                ),
            );
            table.add(
                &format!("L{l} kda.conv_state"),
                err(&widen(&run.conv[0][i]), &last3(&cp)),
                err(&widen(&run.conv[1][i]), &last3(&cd)),
                None,
            );
        } else {
            cmp(
                &mut table,
                &format!("L{l} mla.q_resid"),
                all(rec, l, a, TapBuf::DsaQResid),
                Q_LORA,
                gold(g, l, "mla.q_resid"),
                None,
            );
            cmp(
                &mut table,
                &format!("L{l} mla.q"),
                all(rec, l, a, TapBuf::DsaQ),
                MLA_HEADS * QK_HEAD,
                gold(g, l, "mla.q"),
                None,
            );
            cmp(
                &mut table,
                &format!("L{l} mla.kv_a"),
                all(rec, l, a, TapBuf::DsaKvA),
                KV_LORA,
                gold(g, l, "mla.kv_a"),
                None,
            );
            cmp(
                &mut table,
                &format!("L{l} idx.q"),
                all(rec, l, a, TapBuf::DsaIdxQ),
                INDEX_HEADS * INDEX_DIM,
                gold(g, l, "idx.q"),
                None,
            );
            let ip = all(rec, l, a, TapBuf::DsaIdxProj);
            cmp(
                &mut table,
                &format!("L{l} idx.gate_scores"),
                cols(&ip, IDX_PROJ_COLS, INDEX_DIM, 2 * INDEX_DIM),
                INDEX_DIM,
                gold(g, l, "idx.gate_scores"),
                None,
            );
            cmp(
                &mut table,
                &format!("L{l} idx.weights"),
                cols(&ip, IDX_PROJ_COLS, 2 * INDEX_DIM, IDX_PROJ_COLS),
                INDEX_HEADS,
                gold(g, l, "idx.weights"),
                None,
            );
            // Latents as stored (FP8 records) against the golden latent.
            let page = &run.page[1];
            let lat: Vec<f32> = (0..PROMPT + STEPS)
                .flat_map(|t| {
                    decode_latent(&page[t * LATENT_RECORD_BYTES..(t + 1) * LATENT_RECORD_BYTES])
                })
                .collect();
            let nat = native(g, l, "mla.latent").map(|(p, d)| {
                let (gp, gd) = gold(g, l, "mla.latent");
                (err(&p, &gp).rel_rms, err(&d, &gd).rel_rms)
            });
            cmp(
                &mut table,
                &format!("L{l} mla.latent (FP8 cache)"),
                lat,
                KV_LORA,
                gold(g, l, "mla.latent"),
                nat,
            );
            // Pooled keys (FP8) after the prompt and after the last step.
            let pk = |b: &[u8], n: usize| -> Vec<f32> {
                (0..n)
                    .flat_map(|p| {
                        let c = &b[PAGE_POOL_CODES_OFFSET + p * 128
                            ..PAGE_POOL_CODES_OFFSET + (p + 1) * 128];
                        let o = PAGE_POOL_SCALES_OFFSET + p * 4;
                        decode_index_key(
                            c,
                            f32::from_le_bytes([b[o], b[o + 1], b[o + 2], b[o + 3]]),
                        )
                    })
                    .collect()
            };
            let (gpk, gdk) = (
                g.f32(&format!("layer{l:02}-prefill"), "prefill.idx.pool_keys"),
                g.f32(&format!("layer{l:02}-decode"), "decode.s7.idx.pool_keys"),
            );
            table.add(
                &format!("L{l} idx.pool_keys (FP8 cache)"),
                err(&pk(&run.page[0], 8), &gpk),
                err(&pk(&run.page[1], 10), &gdk),
                None,
            );
            cmp(
                &mut table,
                &format!("L{l} mla.out (per head)"),
                all(rec, l, a, TapBuf::DsaHeads),
                MLA_HEADS * V_HEAD,
                gold(g, l, "mla.out"),
                None,
            );
            // Selections: the same token sets (33 prompt tokens: every complete pool is kept).
            let toks = all(rec, l, a, TapBuf::DsaTokens);
            let counts = all(rec, l, a, TapBuf::DsaCounts);
            let ours = token_sets(&toks, MAX_SELECTED, Some(&counts));
            let mut gt: Vec<f32> = g
                .i64(&format!("layer{l:02}-prefill"), "prefill.idx.topk")
                .into_iter()
                .map(|x| x as f32)
                .collect();
            gt.extend(
                g.i64(&format!("layer{l:02}-decode"), "decode.idx.topk")
                    .into_iter()
                    .map(|x| x as f32),
            );
            let gs = token_sets(&gt, MAX_SELECTED, None);
            let same = ours.iter().zip(&gs).filter(|(a, b)| a == b).count();
            eprintln!(
                "  L{l} indexer: {same}/{} rows select the golden token set",
                gs.len()
            );
            assert_eq!(same, gs.len(), "L{l}: selections differ from the goldens");
        }
        cmp(
            &mut table,
            &format!("L{l} attn_out"),
            all(rec, l, a, TapBuf::Out),
            HIDDEN,
            gold(g, l, "attn_out"),
            native_err(g, l, "attn_out", "attn_out"),
        );
        cmp(
            &mut table,
            &format!("L{l} mid_streams"),
            all(rec, l, f, TapBuf::Streams),
            hs,
            gold(g, l, "mid_streams"),
            None,
        );
        for (b, n, w) in [
            (TapBuf::Pre, "pre", 4),
            (TapBuf::Post, "post", 4),
            (TapBuf::Comb, "comb", 16),
        ] {
            cmp(
                &mut table,
                &format!("L{l} ffn_hc.{n}"),
                all(rec, l, f, b),
                w,
                gold(g, l, &format!("ffn_hc.{n}")),
                None,
            );
        }
        cmp(
            &mut table,
            &format!("L{l} ffn_norm"),
            all(rec, l, f, TapBuf::Normed),
            HIDDEN,
            gold(g, l, "ffn_norm"),
            None,
        );
        if l >= 3 {
            cmp(
                &mut table,
                &format!("L{l} moe.router_logits"),
                all(rec, l, f, TapBuf::RouterLogits),
                EXPERTS,
                gold(g, l, "moe.router_logits"),
                None,
            );
            cmp(
                &mut table,
                &format!("L{l} moe.shared_out"),
                all(rec, l, f, TapBuf::SharedOut),
                HIDDEN,
                gold(g, l, "moe.shared_out"),
                None,
            );
            cmp(
                &mut table,
                &format!("L{l} moe.routed_out (golden routes)"),
                all(rec, l, f, TapBuf::Out),
                HIDDEN,
                gold(g, l, "moe.routed_out"),
                native_err(g, l, "moe.routed_out", "moe.routed_out"),
            );
            let mlp: Vec<f32> = all(rec, l, f, TapBuf::Out)
                .iter()
                .zip(all(rec, l, f, TapBuf::SharedOut))
                .map(|(&r, s)| glm53f_layers::bf16::round(r + s))
                .collect();
            cmp(
                &mut table,
                &format!("L{l} mlp_out"),
                mlp,
                HIDDEN,
                gold(g, l, "mlp_out"),
                native_err(g, l, "mlp_out", "mlp_out"),
            );
            // The engine's own routing against the golden one, prompt rows and decode rows.
            let ours: Vec<i32> = all(rec, l, f, TapBuf::RouterIds)
                .iter()
                .map(|&x| x as i32)
                .collect();
            let (gp, gd) = (
                format!("layer{l:02}-prefill"),
                format!("layer{l:02}-decode"),
            );
            let mut gi = g.i64(&gp, "prefill.moe.topk_ids");
            gi.extend(g.i64(&gd, "decode.moe.topk_ids"));
            let mut gl = g.f32(&gp, "prefill.moe.router_logits");
            gl.extend(g.f32(&gd, "decode.moe.router_logits"));
            if let Some(bias) = router_bias(l) {
                let k = PROMPT * TOP_K;
                let (sp, dp, gpf) =
                    routing_agreement(&ours[..k], &gi[..k], &gl[..PROMPT * EXPERTS], &bias);
                let (sd, dd, gdc) =
                    routing_agreement(&ours[k..], &gi[k..], &gl[PROMPT * EXPERTS..], &bias);
                eprintln!(
                    "  L{l} routing: prompt rows {sp}/33 the golden experts ({dp} swap, largest golden-score gap {gpf:.1e}); decode rows {sd}/8 ({dd} swap, gap {gdc:.1e})"
                );
                sum.routing_gap.push((l, gpf.max(gdc)));
            }
        } else {
            cmp(
                &mut table,
                &format!("L{l} mlp_out"),
                all(rec, l, f, TapBuf::Out),
                HIDDEN,
                gold(g, l, "mlp_out"),
                native_err(g, l, "mlp_out", "mlp_out"),
            );
        }
        let (ep, ed) = cmp(
            &mut table,
            &format!("L{l} out_streams"),
            all(rec, l, TapPoint::LayerOut, TapBuf::Streams),
            hs,
            gold(g, l, "out_streams"),
            native_err(g, l, "out_streams", "out_streams"),
        );
        sum.out_streams.push((l, ep.rel_rms.max(ed.rel_rms)));
    }
    if let Some((logits, picks)) = &run.logits {
        let gl = g.f32("head", "head.logits");
        let nl = g.f32("native", "head.logits");
        let (lp, ld) = (
            err(&logits[..VOCAB], &gl[..VOCAB]),
            err(&logits[VOCAB..], &gl[VOCAB..]),
        );
        table.add(
            "head logits",
            lp,
            ld,
            Some((
                err(&nl[..VOCAB], &gl[..VOCAB]).rel_rms,
                err(&nl[VOCAB..], &gl[VOCAB..]).rel_rms,
            )),
        );
        let argmax = |v: &[f32]| -> usize {
            let mut b = 0;
            for i in 0..SAMPLE_VOCAB {
                if v[i] > v[b] {
                    b = i;
                }
            }
            b
        };
        let agree = (0..=STEPS)
            .filter(|&r| {
                argmax(&logits[r * VOCAB..(r + 1) * VOCAB])
                    == argmax(&gl[r * VOCAB..(r + 1) * VOCAB])
            })
            .count();
        let picks_ok = (0..=STEPS)
            .filter(|&r| picks[r] as usize == argmax(&logits[r * VOCAB..(r + 1) * VOCAB]))
            .count();
        sum.logits = Some((lp.rel_rms, ld.rel_rms, agree, picks_ok));
    }
    table.print(title);
    if let Some((_, _, agree, picks_ok)) = sum.logits {
        eprintln!("  head: argmax equal to the FP32 golden's on {agree}/9 rows; device picks equal the host argmax on {picks_ok}/9");
    }
    sum
}

#[test]
fn layers_0_to_4_against_the_goldens() {
    let names = [
        "layer00-prefill",
        "layer00-decode",
        "layer03-prefill",
        "layer03-decode",
        "layer04-prefill",
        "layer04-decode",
        "head",
        "native",
    ];
    let Some(g) = Goldens::load(&names) else {
        return;
    };
    if !gpu_with(9.0) {
        return;
    }
    let cfg = ForwardConfig {
        max_rows: 64,
        max_verify_rows: 8,
        max_requests: 4,
        ..ForwardConfig::default()
    };
    let Some(mut su) = forward(LAYERS, cfg, HashMap::new(), 3 << 30) else {
        return;
    };
    let (prompt, steps) = g.token_ids("layer00-prefill");
    assert_eq!((prompt.len(), steps.len()), (PROMPT, STEPS));

    // 1. The chain, the prompt in one prefill pass (FP8 projections W8A8 on tensor cores).
    su.fwd.set_experts(golden_experts(&g, su.fwd.stream()));
    let chain = run_chain(&mut su, &prompt, &steps, 64);
    let s1 = report(&g, &chain, "Chain, layers 0-4, the prompt in one pass (relative RMS against the FP32 goldens; native = the reference in BF16)");
    // 2. The chain, the prompt in passes of 8 rows (the decode kernels: FP8 projections W8A16).
    su.fwd.set_experts(golden_experts(&g, su.fwd.stream()));
    let chain8 = run_chain(&mut su, &prompt, &steps, 8);
    let s2 = report(
        &g,
        &chain8,
        "Chain, layers 0-4, the prompt in passes of 8 rows",
    );
    // 3. Each layer fed its golden input (cast to BF16), as the native set was made: the prompt
    //    in passes of 8 rows (the decode kernels), then in one pass.
    su.fwd.set_experts(golden_experts(&g, su.fwd.stream()));
    let per8 = run_per_layer(&mut su, &g, 8);
    let s3 = report(&g, &per8, "Per layer: each layer fed its golden FP32 input cast to BF16, fresh cache (the native set's setup), the prompt in passes of 8 rows");
    su.fwd.set_experts(golden_experts(&g, su.fwd.stream()));
    let per = run_per_layer(&mut su, &g, PROMPT);
    let s4 = report(&g, &per, "Per layer, the prompt in one pass");
    // 4. The chain with the prompt's KDA through the chunked prefill kernel (a numerics change
    //    against the chain: f32 rounding, not bit for bit).
    su.fwd.set_experts(golden_experts(&g, su.fwd.stream()));
    su.fwd.cfg.kda_chunked_prefill = true;
    let chunked = run_chain(&mut su, &prompt, &steps, 64);
    su.fwd.cfg.kda_chunked_prefill = false;
    let s5 = report(
        &g,
        &chunked,
        "Chain, the prompt in one pass with KDA through the chunked prefill kernel",
    );
    // The chunked kernel against the chain on the same layer-0 input: close, and not the same bits.
    let (sc, sk) = (&chain.state[0][0], &chunked.state[0][0]);
    let st_diff = err(sk, sc).rel_rms;
    let no = |r: &Run| {
        all(&r.rec, 0, TapPoint::AttnDone, TapBuf::KdaNormOut)[..PROMPT * KDA_WIDTH].to_vec()
    };
    let (nc, nk) = (no(&chain), no(&chunked));
    let flips = nc.iter().zip(&nk).filter(|(a, b)| a != b).count();
    eprintln!(
        "  chunked vs chain, layer 0 after the prompt: state relative RMS {st_diff:.2e}; {flips} of {} gated-norm outputs differ (BF16)",
        nc.len()
    );
    assert!(
        st_diff > 0.0 && st_diff < 1e-4,
        "chunked KDA prefill against the chain: {st_diff:.3e}"
    );

    for (s, what, bound) in [
        (&s1, "chain", 5e-2),
        (&s2, "chain, 8-row passes", 2e-2),
        (&s5, "chain, chunked KDA prefill", 5e-2),
    ] {
        let (lp, ld, agree, picks_ok) = s.logits.unwrap();
        assert_eq!(
            picks_ok, 9,
            "{what}: the device argmax disagrees with the host"
        );
        assert!(agree >= 8, "{what}: argmax agrees on only {agree}/9 rows");
        assert!(
            lp.max(ld) < bound,
            "{what}: logits relative RMS {lp:.3e} / {ld:.3e}"
        );
        for (l, e) in &s.out_streams {
            assert!(*e < bound, "{what}: L{l} out_streams relative RMS {e:.3e}");
        }
    }
    for (s, what, bound) in [
        (&s3, "per layer, 8-row passes", 1e-2),
        (&s4, "per layer, one pass", 5e-2),
    ] {
        for (l, e) in &s.out_streams {
            assert!(*e < bound, "{what}: L{l} out_streams relative RMS {e:.3e}");
        }
    }
    for s in [&s1, &s2, &s3, &s4, &s5] {
        for (l, gap) in &s.routing_gap {
            assert!(
                *gap < 2e-2,
                "L{l}: a routing difference is not a near-tie (golden-score gap {gap:.3e})"
            );
        }
    }
}

/// The router's correction bias of `layer`, read from the checkpoint (for the near-tie check).
fn router_bias(layer: usize) -> Option<Vec<f32>> {
    let dir = checkpoint_dir()?;
    let ck = glm53f_model::safetensors::Checkpoint::open(&dir).ok()?;
    let b = ck
        .read_tensor(&format!(
            "model.language_model.layers.{layer}.mlp.gate.e_score_correction_bias"
        ))
        .ok()?;
    Some(
        b.as_chunks::<4>()
            .0
            .iter()
            .map(|c| f32::from_le_bytes(*c))
            .collect(),
    )
}

/// Local FP8 experts behind a fresh queue of golden routes (each MoE layer: the prompt, then
/// the 8 steps).
fn golden_experts(
    g: &Goldens,
    stream: &glm53f_forward::device::Stream,
) -> Box<dyn glm53f_forward::experts::ExpertBackend> {
    let local = glm53f_forward::experts::LocalFp8Experts::new(
        &experts_dir().unwrap(),
        3 << 30,
        64,
        stream,
        glm53f_forward::gemm::Fp8Act::Bf16,
    )
    .unwrap();
    let mut gr = GoldenRoutes::new(local, 64);
    gr.queue = golden_routes(g, &[3, 4]);
    Box::new(gr)
}
