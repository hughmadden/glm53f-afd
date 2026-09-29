//! The drafter's acceptance on real prompts, the BF16 drafter against the FP8 one
//! (`glm53f_dflash::gpu`, "The FP8 drafter"), replayed from recordings of the whole target
//! (`glm53f-forward`'s `examples/draft_record.rs`: each case's tokens, every row's taps and the
//! target's greedy pick after it).
//!
//! ```sh
//! GLM53F_DFLASH_DIR=<drafter> GLM53F_CHECKPOINT_DIR=<GLM-5.3-Flash with embed_tokens, lm_head> \
//!   cargo run --release -p glm53f-dflash --features cuda --example draft_replay -- <recordings>
//! ```
//!
//! **A round at every position.** For each position `p` of a case's reply, a greedy round as a
//! serving slot runs it: the slot's context holds rows `0 .. p` (the prompt appended in one call,
//! as a prefill appends it, then one row per position, as commits do), the anchor is token `p`,
//! and the drafter proposes 7 tokens. The target accepts draft `j` when it equals the target's
//! greedy pick after row `p + j` and every earlier draft was accepted; the recording holds that
//! pick for the reply's own tokens, so a draft is scored while the accepted drafts are the
//! reply's tokens (teacher forcing). Where the reply is the target's own greedy text (counting,
//! most of the structured case) this is the serving result exactly; where the target would have
//! written something else, a kept draft that leaves the reply ends the count, a lower bound. Both
//! drafters are scored on the same positions, contexts and picks.
//!
//! Per case and drafter, with every draft verified (7) and with the serving chain cut
//! (`GLM53F_SPEC_TAU`, 0.7: drafts verified while the product of the drafter's confidences stays
//! at or above it): drafts kept per round, drafts kept of those verified, tokens per round. The
//! FP8 drafter's difference is given per position with its standard error. Then all cases
//! drafted in one batch per step (their rows over 8: the FP8 GEMMs' W8A8 path).

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::time::Instant;

use glm53f_dflash::device::Event;
use glm53f_dflash::gpu::{GpuDrafter, GpuSlot};
use glm53f_dflash::seam::{DraftRequest, Proposal};
use glm53f_dflash::weights::{env_dir, Target, Weights};
use glm53f_dflash::Dims;

/// A recording (`draft_record.rs`).
struct Rec {
    name: String,
    prompt: usize,
    tokens: Vec<u32>,
    picks: Vec<u32>,
    /// `[n][taps * hidden]` BF16.
    taps: Vec<u16>,
}

fn read_rec(path: &Path, tw: usize) -> Result<Rec, String> {
    let b = std::fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
    if b.len() < 16 || &b[..8] != b"G53REC01" {
        return Err(format!("{}: not a recording", path.display()));
    }
    let u = |i: usize| u32::from_le_bytes(b[i..i + 4].try_into().unwrap());
    let (n, prompt) = (u(8) as usize, u(12) as usize);
    let at = 16;
    let want = at + 8 * n + 2 * n * tw;
    if b.len() != want {
        return Err(format!(
            "{}: {} bytes, expected {want}",
            path.display(),
            b.len()
        ));
    }
    let words = |o: usize| (0..n).map(|i| u(o + 4 * i)).collect::<Vec<u32>>();
    let taps = b[at + 8 * n..]
        .chunks_exact(2)
        .map(|c| u16::from_le_bytes([c[0], c[1]]))
        .collect();
    Ok(Rec {
        name: path.file_stem().unwrap().to_string_lossy().into_owned(),
        prompt,
        tokens: words(at),
        picks: words(at + 4 * n),
        taps,
    })
}

/// Drafts the target keeps of `d` proposed at position `p`: while each equals the target's
/// pick after its row and the drafts before it are the reply's tokens.
fn kept(r: &Rec, p: usize, d: &[u32]) -> usize {
    let n = r.tokens.len();
    let mut k = 0;
    while k < d.len() && p + k < n {
        if k > 0 && r.picks[p + k - 1] != r.tokens[p + k] {
            break;
        }
        if d[k] != r.picks[p + k] {
            break;
        }
        k += 1;
    }
    k
}

/// The chain cut: drafts verified while the product of the confidences stays at or above `tau`.
fn chain(conf: &[f32], tau: f64) -> usize {
    let mut q = 1.0f64;
    for (j, &c) in conf.iter().enumerate() {
        q *= f64::from(c).clamp(0.0, 1.0);
        if q < tau {
            return j;
        }
    }
    conf.len()
}

/// One drafter's rounds over the cases: per case, per position `(kept of 7, verified at tau,
/// kept at tau)`, and the median draft time (B = 1) in milliseconds.
#[derive(Default, Clone)]
struct Score {
    rounds: Vec<(usize, usize, usize)>,
    draft_ms: Vec<f64>,
}

fn embed_of(rows: &HashMap<u32, Vec<u16>>, t: u32) -> &[u16] {
    &rows[&t]
}

/// Every case alone: the prompt appended in one call, then a draft and a one-row append per
/// position.
fn replay_each(
    g: &mut GpuDrafter,
    recs: &[Rec],
    rows: &HashMap<u32, Vec<u16>>,
    tau: f64,
) -> Result<Vec<Score>, String> {
    let d = g.dims();
    let tw = d.tap_width();
    let ev = (Event::new()?, Event::new()?);
    let mut out = Vec::new();
    for r in recs {
        let mut s = Score::default();
        let mut slot = g.new_slot()?;
        g.append(&mut [(&mut slot, &r.taps[..r.prompt * tw])])?;
        for p in r.prompt..r.tokens.len() - 1 {
            let req = DraftRequest {
                slot: &slot,
                anchor: r.tokens[p],
                anchor_embed: embed_of(rows, r.tokens[p]),
                temperature: 0.0,
                uniforms: &[],
            };
            ev.0.record(g.stream())?;
            g.launch(std::slice::from_ref(&req))?;
            ev.1.record(g.stream())?;
            let prop = g.proposals(1)?.remove(0);
            s.draft_ms.push(ev.1.elapsed_ms_since(&ev.0)? as f64);
            s.rounds.push(score(r, p, &prop, tau));
            g.append(&mut [(&mut slot, &r.taps[p * tw..(p + 1) * tw])])?;
        }
        out.push(s);
    }
    Ok(out)
}

fn score(r: &Rec, p: usize, prop: &Proposal, tau: f64) -> (usize, usize, usize) {
    let k = kept(r, p, &prop.tokens);
    let v = chain(&prop.conf, tau);
    (k, v, k.min(v))
}

/// All cases drafted together, step by step from each case's reply (a case whose reply ended
/// drops out); each step one draft call and one append call for every case still going.
fn replay_batched(
    g: &mut GpuDrafter,
    recs: &[Rec],
    rows: &HashMap<u32, Vec<u16>>,
    tau: f64,
) -> Result<Vec<Score>, String> {
    let tw = g.dims().tap_width();
    let mut slots: Vec<GpuSlot> = recs
        .iter()
        .map(|_| g.new_slot())
        .collect::<Result<_, _>>()?;
    {
        let mut items: Vec<(&mut GpuSlot, &[u16])> = slots
            .iter_mut()
            .zip(recs)
            .map(|(s, r)| (s, &r.taps[..r.prompt * tw]))
            .collect();
        g.append(&mut items)?;
    }
    let mut out = vec![Score::default(); recs.len()];
    let steps = recs
        .iter()
        .map(|r| r.tokens.len() - 1 - r.prompt)
        .max()
        .unwrap_or(0);
    for i in 0..steps {
        let live: Vec<usize> = (0..recs.len())
            .filter(|&c| recs[c].prompt + i < recs[c].tokens.len() - 1)
            .collect();
        let props = {
            let reqs: Vec<DraftRequest<'_, GpuSlot>> = live
                .iter()
                .map(|&c| {
                    let t = recs[c].tokens[recs[c].prompt + i];
                    DraftRequest {
                        slot: &slots[c],
                        anchor: t,
                        anchor_embed: embed_of(rows, t),
                        temperature: 0.0,
                        uniforms: &[],
                    }
                })
                .collect();
            g.launch(&reqs)?;
            g.proposals(reqs.len())?
        };
        for (&c, prop) in live.iter().zip(&props) {
            out[c]
                .rounds
                .push(score(&recs[c], recs[c].prompt + i, prop, tau));
        }
        let mut items: Vec<(&mut GpuSlot, &[u16])> = slots
            .iter_mut()
            .enumerate()
            .filter(|(c, _)| live.contains(c))
            .map(|(c, s)| {
                let p = recs[c].prompt + i;
                (s, &recs[c].taps[p * tw..(p + 1) * tw])
            })
            .collect();
        g.append(&mut items)?;
    }
    Ok(out)
}

/// (kept per round, kept of verified, tokens per round) at 7 drafts and at the chain cut.
fn summary(rounds: &[(usize, usize, usize)]) -> [f64; 5] {
    let n = rounds.len().max(1) as f64;
    let k7: usize = rounds.iter().map(|x| x.0).sum();
    let v: usize = rounds.iter().map(|x| x.1).sum();
    let kt: usize = rounds.iter().map(|x| x.2).sum();
    [
        k7 as f64 / n,
        k7 as f64 / (7.0 * n),
        kt as f64 / v.max(1) as f64,
        1.0 + kt as f64 / n,
        v as f64 / n,
    ]
}

/// Mean and standard error of the per-position difference of two drafters' kept drafts.
fn paired(
    a: &[(usize, usize, usize)],
    b: &[(usize, usize, usize)],
    f: fn(&(usize, usize, usize)) -> usize,
) -> (f64, f64) {
    let d: Vec<f64> = a
        .iter()
        .zip(b)
        .map(|(x, y)| f(y) as f64 - f(x) as f64)
        .collect();
    let n = d.len().max(2) as f64;
    let m = d.iter().sum::<f64>() / n;
    let var = d.iter().map(|x| (x - m).powi(2)).sum::<f64>() / (n - 1.0);
    (m, (var / n).sqrt())
}

fn main() -> Result<(), String> {
    let Some(dir) = std::env::args().nth(1).map(PathBuf::from) else {
        eprintln!("usage: draft_replay <directory of .rec recordings>");
        return Ok(());
    };
    let (Some(dd), Some(cd)) = (
        env_dir("GLM53F_DFLASH_DIR", "the drafter checkpoint directory"),
        env_dir(
            "GLM53F_CHECKPOINT_DIR",
            "a GLM-5.3-Flash directory with embed_tokens and lm_head",
        ),
    ) else {
        return Ok(());
    };
    let tau: f64 = std::env::var("GLM53F_SPEC_TAU")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(0.7);
    let d = Dims::GLM53F;
    let mut paths: Vec<PathBuf> = std::fs::read_dir(&dir)
        .map_err(|e| format!("{}: {e}", dir.display()))?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|x| x == "rec"))
        .collect();
    paths.sort();
    let recs: Vec<Rec> = paths
        .iter()
        .map(|p| read_rec(p, d.tap_width()))
        .collect::<Result<_, _>>()?;
    let w = Weights::load(&dd, d)?;
    let target = Target::open(&cd, &d)?;
    let head = target.lm_head()?;
    let mask = target.embed_rows(&[d.mask_token])?;
    let mut ids: Vec<u32> = recs.iter().flat_map(|r| r.tokens.iter().copied()).collect();
    ids.sort_unstable();
    ids.dedup();
    let table = target.embed_rows(&ids)?;
    let rows: HashMap<u32, Vec<u16>> = ids
        .iter()
        .enumerate()
        .map(|(i, &t)| (t, table[i * d.hidden..(i + 1) * d.hidden].to_vec()))
        .collect();
    println!(
        "{} recordings; chain cut at tau {tau}; a round at every reply position",
        recs.len()
    );
    let mut runs: Vec<(&str, Vec<Score>, Vec<Score>, usize)> = Vec::new();
    for fp8 in [false, true] {
        let t = Instant::now();
        let mut g = if fp8 {
            GpuDrafter::new_fp8(&w, &head, &mask)?
        } else {
            GpuDrafter::new(&w, &head, &mask)?
        };
        let bytes = g.weight_bytes();
        let each = replay_each(&mut g, &recs, &rows, tau)?;
        let batched = replay_batched(&mut g, &recs, &rows, tau)?;
        println!(
            "{} drafter: {:.3} GiB of weights and head; replayed in {:.1} s",
            if fp8 { "FP8" } else { "BF16" },
            bytes as f64 / (1u64 << 30) as f64,
            t.elapsed().as_secs_f64()
        );
        runs.push((if fp8 { "FP8" } else { "BF16" }, each, batched, bytes));
    }
    println!(
        "\n{:<11} {:>6} | {:>28} | {:>28} | {:>21}",
        "case",
        "rounds",
        "7 drafts: kept, % of drafts",
        "chain: kept %, tokens/round",
        "FP8 - BF16 kept (SE)"
    );
    let all = |v: &[Score]| -> Vec<(usize, usize, usize)> {
        v.iter().flat_map(|s| s.rounds.iter().copied()).collect()
    };
    for (label, pick) in [("alone (B = 1)", 0usize), ("batched (B = all)", 1)] {
        println!("{label}:");
        let get = |run: &(&str, Vec<Score>, Vec<Score>, usize)| -> Vec<Score> {
            if pick == 0 {
                run.1.clone()
            } else {
                run.2.clone()
            }
        };
        let (b, f) = (get(&runs[0]), get(&runs[1]));
        let mut lines: Vec<(
            String,
            Vec<(usize, usize, usize)>,
            Vec<(usize, usize, usize)>,
        )> = recs
            .iter()
            .enumerate()
            .map(|(i, r)| (r.name.clone(), b[i].rounds.clone(), f[i].rounds.clone()))
            .collect();
        lines.push(("all".into(), all(&b), all(&f)));
        for (name, rb, rf) in lines {
            let (sb, sf) = (summary(&rb), summary(&rf));
            let (m7, se7) = paired(&rb, &rf, |x| x.0);
            let (mt, set) = paired(&rb, &rf, |x| x.2);
            println!(
                "{name:<11} {:>6} | BF16 {:.2} {:>5.1}%  FP8 {:.2} {:>5.1}% | BF16 {:>5.1}% {:.2}  FP8 {:>5.1}% {:.2} | 7: {m7:+.3} ({se7:.3}); chain: {mt:+.3} ({set:.3})",
                rb.len(),
                sb[0],
                100.0 * sb[1],
                sf[0],
                100.0 * sf[1],
                100.0 * sb[2],
                sb[3],
                100.0 * sf[2],
                sf[3],
            );
        }
    }
    let med = |v: &[Score]| -> f64 {
        let mut t: Vec<f64> = v.iter().flat_map(|s| s.draft_ms.iter().copied()).collect();
        t.sort_by(f64::total_cmp);
        t[t.len() / 2]
    };
    println!(
        "\ndraft time on real contexts (one request, CUDA events, median): BF16 {:.3} ms, FP8 {:.3} ms",
        med(&runs[0].1),
        med(&runs[1].1)
    );
    Ok(())
}
