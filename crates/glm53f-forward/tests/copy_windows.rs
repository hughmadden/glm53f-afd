//! Copy windows (`glm53f_coordinator::copy`) on the forward (feature `coordinator`; the
//! coordinator's weights, the experts of layers 3 and 4, the drafter's checkpoint and
//! `GLM53F_TOKENIZER`; skips without them). The shell's scheduler runs chat requests rendered with
//! the official template over a forward of all 45 decoder layers with the DFlash2 drafter
//! (`drafting`) three ways: plain (the model's block hidden, one token a step), speculative, and
//! speculative with copy windows. The token sequences must be equal:
//!
//! - greedy requests one at a time: a file to write back with a method renamed, a function to
//!   quote, and code to write from scratch;
//! - two greedy requests at once, 3 drafts a step (verify passes of at most 8 rows): the rename,
//!   which copies, next to the fresh code;
//! - a sampled request with a fixed seed, which never copies.
//!
//! A copied window is never drafted for (the drafter is asked for fewer requests than there are
//! windows, by at least the copied ones), and a request whose steps copied drafts again from a warm
//! context (none cold). The test runs with one-lane prefill, and with two-lane prefill and decode.
//!
//! **Which model copies.** Copies need a forward that repeats its context. The development default
//! (`GLM53F_DRAFT_TEST_LAYERS` 5: layers 0-4, the rest repeating them) does not: its greedy text
//! repeated no 8-token span in 400 tokens, and its runs verified no copy, so the tests skip it.
//! With every layer loaded (`GLM53F_DRAFT_TEST_LAYERS=45`; `GLM53F_TEST_NUMERICS=kda-fp8` fits a
//! 24 GB GPU), the forward repeats the prompt it is given, and copies are verified and kept (the
//! test requires it). Routed experts run for layers 3 and 4 only (zeros elsewhere), so the text is
//! still not the model's.
//!
//! `copy_windows_tokens_per_round` (ignored; run it with `--ignored`) measures, with every layer
//! loaded and the default verify policy, the tokens a verify round delivers and the drafter's time
//! per step, with copy windows off and on, on copy-heavy requests and on controls.
#![cfg(feature = "coordinator")]

mod common;
mod drafting;

use std::sync::mpsc;

use drafting::*;
use glm53f_api::engine::PromptOptions;
use glm53f_api::json::Json;
use glm53f_api::types::ChatRequest;
use glm53f_coordinator::engine::PromptCodec;
use glm53f_coordinator::model::{
    DecodeRow, Draft, DraftRow, Limits, ModelForward, Pick, Segment, SegmentOut, Token, Window,
};
use glm53f_coordinator::sampling::Sampling;
use glm53f_coordinator::scheduler::{Job, SchedStats, Scheduler, SchedulerConfig};
use glm53f_coordinator::spec::SpecPolicy;
use glm53f_coordinator::GlmPrompts;
use glm53f_forward::forward::ForwardConfig;
use glm53f_forward::kv::GlmKv;
use glm53f_forward::serve::{DraftStats, ServedForward};

const QUEUE: &str = include_str!("../../glm53f-coordinator/src/queue.rs");
const STREAMING: &str = include_str!("../../glm53f-coordinator/src/streaming.rs");

/// The served forward, with its block hidden (plain decoding) or not.
struct Lens<'a> {
    m: &'a mut ServedForward,
    spec: bool,
}

impl ModelForward for Lens<'_> {
    type Slot = GlmKv;

    fn limits(&self) -> Limits {
        let mut l = self.m.limits();
        if !self.spec {
            l.block = 0;
        }
        l
    }
    fn free_bytes(&self) -> Result<usize, String> {
        self.m.free_bytes()
    }
    fn prefill(&mut self, segs: &mut [Segment<'_, GlmKv>]) -> Result<Vec<SegmentOut>, String> {
        self.m.prefill(segs)
    }
    fn decode(&mut self, rows: &mut [DecodeRow<'_, GlmKv>]) -> Result<Vec<Token>, String> {
        self.m.decode(rows)
    }
    fn draft(&mut self, rows: &mut [DraftRow<'_, GlmKv>]) -> Result<Vec<Draft>, String> {
        self.m.draft(rows)
    }
    fn verify(&mut self, windows: &mut [Window<'_, GlmKv>]) -> Result<Vec<Vec<Token>>, String> {
        self.m.verify(windows)
    }
    fn commit(&mut self, slots: &mut [&mut GlmKv], keep: &[usize]) -> Result<(), String> {
        self.m.commit(slots, keep)
    }
    fn select_host(&mut self, logits: &[f32], pick: &Pick) -> Result<Token, String> {
        self.m.select_host(logits, pick)
    }
}

/// A request: prompt, token budget, sampling.
type Req = (Vec<Token>, usize, Option<Sampling>);

/// How a run decodes.
#[derive(Clone, Copy, PartialEq)]
enum Mode {
    Plain,
    Drafts,
    Copies,
}

fn delta(a: DraftStats, b: DraftStats) -> DraftStats {
    DraftStats {
        drafted: b.drafted - a.drafted,
        cold: b.cold - a.cold,
        proposed: b.proposed - a.proposed,
        rounds: b.rounds - a.rounds,
        windows: b.windows - a.windows,
        verified: b.verified - a.verified,
        kept: b.kept - a.kept,
    }
}

/// What a run gave: every request's tokens, the scheduler's and the drafter's counters, and with
/// the lane trace on, the speculative steps' drafter and pass milliseconds summed.
struct Ran {
    out: Vec<Vec<Token>>,
    st: SchedStats,
    ds: DraftStats,
    draft_ms: f64,
    pass_ms: f64,
    secs: f64,
}

/// Run `wave` (submitted together, run until idle) through a scheduler over `m` under `policy`.
fn run(m: &mut ServedForward, mode: Mode, policy: SpecPolicy, wave: &[Req]) -> Ran {
    let before = m.stats;
    let slots: Vec<GlmKv> = (0..3).map(|_| m.fwd.kv.slot().unwrap()).collect();
    let mut cfg = SchedulerConfig::new(Vec::new());
    cfg.min_retain = usize::MAX;
    cfg.out_min = 16;
    cfg.policy = policy;
    cfg.copy_windows = mode == Mode::Copies;
    let (tx, rx) = mpsc::channel();
    let spec = mode != Mode::Plain;
    let t0 = std::time::Instant::now();
    let mut sched = Scheduler::new(Lens { m: &mut *m, spec }, slots, None, cfg, rx);
    let rxs: Vec<_> = wave
        .iter()
        .map(|(p, max, s)| {
            let (job, r) = Job::new(p.clone(), *max, *s);
            tx.send(job).unwrap();
            r
        })
        .collect();
    let (mut idle, mut draft_ms, mut pass_ms) = (0, 0.0, 0.0);
    for _ in 0..100_000 {
        assert!(sched.step(false));
        if let Some(t) = sched.model_mut().m.fwd.take_step_trace() {
            draft_ms += t.step.draft_ms;
            pass_ms += t.step.pass_ms + t.step.commit_ms;
        }
        idle = if sched.is_idle() { idle + 1 } else { 0 };
        if idle >= 3 {
            break;
        }
    }
    assert!(idle >= 3, "the scheduler did not go idle");
    let out = rxs.iter().map(|r| r.try_iter().collect::<Result<Vec<Token>, String>>().unwrap()).collect();
    let st = sched.stats;
    drop(sched);
    Ran { out, st, ds: delta(before, m.stats), draft_ms, pass_ms, secs: t0.elapsed().as_secs_f64() }
}

/// Plain, speculative, and speculative with copy windows (every draft verified): equal tokens.
/// Returns the copying run.
fn same(m: &mut ServedForward, what: &str, wave: &[Req]) -> Ran {
    let plain = run(m, Mode::Plain, SpecPolicy::Fixed, wave);
    let drafts = run(m, Mode::Drafts, SpecPolicy::Fixed, wave);
    let copies = run(m, Mode::Copies, SpecPolicy::Fixed, wave);
    for (i, ((p, d), c)) in plain.out.iter().zip(&drafts.out).zip(&copies.out).enumerate() {
        assert_eq!(p, d, "{what}: request {i}: speculative decoding changed the tokens");
        assert_eq!(p, c, "{what}: request {i}: copy windows changed the tokens");
    }
    let (sc, dc) = (copies.st, copies.ds);
    assert_eq!(drafts.st.copy_windows, 0);
    // Every copied window went undrafted; the drafter's context stayed warm.
    assert_eq!(dc.windows, sc.windows);
    assert!(dc.windows - dc.drafted - dc.cold >= sc.copy_windows, "{what}: {sc:?} {dc:?}");
    assert_eq!(dc.cold, 0, "{what}: a request's drafter context went cold");
    let n: usize = plain.out.iter().map(Vec::len).sum();
    eprintln!(
        "{what}: {n} tokens equal; plain {:.1} s; drafts {} steps, {:.1} s, {} requests drafted; copies {} steps, \
         {:.1} s, {} requests drafted, {} of {} windows copied, {} of {} copied tokens kept",
        plain.secs,
        drafts.st.spec_steps,
        drafts.secs,
        drafts.ds.drafted,
        sc.spec_steps,
        copies.secs,
        dc.drafted,
        sc.copy_windows,
        sc.windows,
        sc.copies_accepted,
        sc.copies_verified
    );
    copies
}

fn s(x: &str) -> Json {
    Json::Str(x.to_string())
}

fn obj(kv: Vec<(&str, Json)>) -> Json {
    Json::Object(kv.into_iter().map(|(k, v)| (k.to_string(), v)).collect())
}

fn user(text: String) -> Json {
    obj(vec![("role", s("user")), ("content", Json::Str(text))])
}

/// The chat requests of the tests: (name, copy-heavy, request), rendered with thinking off.
fn requests() -> Vec<(&'static str, bool, Json)> {
    let req = |messages: Vec<Json>, tools: Option<Json>| {
        let mut kv = vec![("model", s("glm-5.3-flash")), ("messages", Json::Array(messages))];
        kv.extend(tools.map(|t| ("tools", t)));
        obj(kv)
    };
    let fenced = |code: &str| format!("```rust\n{code}```");
    let tool = |name: &str, desc: &str, params: &[&str]| {
        obj(vec![
            ("type", s("function")),
            ("function", obj(vec![
                ("name", s(name)),
                ("description", s(desc)),
                ("parameters", obj(vec![
                    ("type", s("object")),
                    ("properties", obj(params.iter().map(|p| (*p, obj(vec![("type", s("string"))]))).collect())),
                    ("required", Json::Array(params.iter().map(|p| s(p)).collect())),
                ])),
            ])),
        ])
    };
    let path = "crates/glm53f-coordinator/src/streaming.rs";
    let tools = Json::Array(vec![
        tool("read_file", "Read a file.", &["path"]),
        tool("edit_file", "Replace an exact string in a file.", &["path", "old_string", "new_string"]),
    ]);
    vec![
        (
            "rename, the whole file back",
            true,
            req(vec![user(format!(
                "Rename the method `queued` to `in_queue` everywhere in this file. Reply with the complete updated file \
                 only, no commentary.\n\n{}",
                fenced(QUEUE)
            ))], None),
        ),
        (
            "edit, the whole file back",
            true,
            req(vec![user(format!(
                "In this file, make the queue's default wait 30,000 ms instead of 25,000, in the code and in the \
                 comments that state it. Reply with the whole updated file.\n\n{}",
                fenced(QUEUE)
            ))], None),
        ),
        (
            "an edit_file call",
            true,
            req(vec![
                user(format!(
                    "In {path}, use `Option::is_none_or` in `flush_pending` where it compares stop positions. Use \
                     edit_file and replace the whole function in one call."
                )),
                obj(vec![
                    ("role", s("assistant")),
                    ("content", s("")),
                    ("tool_calls", Json::Array(vec![obj(vec![
                        ("id", s("call_1")),
                        ("type", s("function")),
                        ("function", obj(vec![
                            ("name", s("read_file")),
                            ("arguments", Json::Str(format!("{{\"path\": \"{path}\"}}"))),
                        ])),
                    ])])),
                ]),
                obj(vec![("role", s("tool")), ("tool_call_id", s("call_1")), ("content", s(STREAMING))]),
            ], Some(tools)),
        ),
        (
            "quote a function",
            true,
            req(vec![user(format!(
                "Copy the function `flush_pending` from this file exactly as written (verbatim, in a rust code block), \
                 then explain in two sentences what it returns.\n\n{}",
                fenced(STREAMING)
            ))], None),
        ),
        (
            "fresh code",
            false,
            req(vec![user(
                "Write a Rust module implementing a least-recently-used cache with get, put and resize, with doc \
                 comments and tests."
                    .to_string(),
            )], None),
        ),
        (
            "fresh prose",
            false,
            req(vec![user(
                "Write a detailed essay about the history of lighthouses, their engineering and the lives of their \
                 keepers."
                    .to_string(),
            )], None),
        ),
    ]
}

/// The prompts of [`requests`], tokenized (`GLM53F_TOKENIZER`, the template beside the
/// checkpoint's weights), and the codec.
fn prompts() -> Option<(GlmPrompts, Vec<(&'static str, bool, Vec<Token>)>)> {
    let (Some(tok), Some(dir)) = (common::env_dir(&["GLM53F_TOKENIZER"]), common::checkpoint_dir()) else {
        eprintln!("skip: GLM53F_TOKENIZER is not set");
        return None;
    };
    let codec = GlmPrompts::load(&tok, &dir.join("chat_template.jinja")).unwrap();
    let opts = PromptOptions { thinking: false, reasoning_effort: None, clear_thinking: None };
    let none = |_: &str| -> Result<glm53f_api::engine::ImageInput, String> { Err("no images".into()) };
    let list = requests()
        .into_iter()
        .map(|(name, heavy, body)| {
            let r = ChatRequest::parse(&body, &none).unwrap();
            (name, heavy, codec.encode(&codec.render(&r.messages, &r.tools, &opts).unwrap()).unwrap())
        })
        .collect();
    Some((codec, list))
}

/// Decoder layers the drafting forward loads.
fn loaded_layers() -> usize {
    std::env::var("GLM53F_DRAFT_TEST_LAYERS").ok().and_then(|v| v.parse().ok()).unwrap_or(5)
}

/// The drafting forward for these prompts (up to 4,096 tokens a slot), every layer loaded; the local
/// experts get 0.5 GiB, so the forward fits a 24 GB GPU.
fn forward(cfg: ForwardConfig) -> Option<ServedForward> {
    let fwd = drafted_forward(cfg, 3, 64, 192, 0.5)?;
    let m = ServedForward::new(fwd).unwrap();
    assert_eq!(m.limits().block, 8);
    Some(m)
}

#[test]
fn copy_windows_change_no_token() {
    lossless(ForwardConfig {
        max_rows: 256,
        max_verify_rows: 16,
        max_requests: 3,
        ..ForwardConfig::default()
    });
}

/// The same with two-lane prefill and decode: the pair decodes and verifies one request a lane.
#[test]
fn copy_windows_change_no_token_with_two_lane_decode() {
    lossless(ForwardConfig {
        max_rows: 256,
        lanes: 2,
        min_lane_rows: 8,
        decode_lane_rows: 2,
        max_verify_rows: 16,
        max_requests: 3,
        ..ForwardConfig::default()
    });
}

fn lossless(cfg: ForwardConfig) {
    if loaded_layers() < 45 {
        eprintln!("skip: set GLM53F_DRAFT_TEST_LAYERS=45 (a forward that repeats its context)");
        return;
    }
    let Some((_, prompts)) = prompts() else {
        return;
    };
    let Some(mut m) = forward(cfg) else {
        return;
    };
    let get = |name: &str| prompts.iter().find(|p| p.0 == name).unwrap().2.clone();
    let (rename, quote, fresh) = (get("rename, the whole file back"), get("quote a function"), get("fresh code"));

    // One at a time.
    let mut copied = SchedStats::default();
    for (what, prompt) in [("rename", &rename), ("quote", &quote), ("fresh code", &fresh)] {
        let st = same(&mut m, what, &[(prompt.clone(), 40, None)]).st;
        copied.copy_windows += st.copy_windows;
        copied.copies_accepted += st.copies_accepted;
    }
    // Two at once, 3 drafts a step.
    m.max_drafts = 3;
    let pair = [(rename.clone(), 32, None), (fresh, 32, None)];
    copied.copy_windows += same(&mut m, "rename and fresh code at once, 3 drafts a step", &pair).st.copy_windows;
    m.max_drafts = 7;
    // A sampled request never copies.
    let s = Sampling::new(0.9, 0.95, 0, 0.0, Some(5)).unwrap();
    assert_eq!(same(&mut m, "rename sampled, seed 5", &[(rename, 32, s)]).st.copy_windows, 0);

    assert!(copied.copy_windows > 0 && copied.copies_accepted > 0, "the forward copied nothing: {copied:?}");
}

/// Tokens a verify round delivers, with copy windows off and on, under the default verify policy
/// (the chain cut at 0.7), and the drafter's and the passes' time per step (the lane trace, on this
/// GPU). Every layer loaded; `GLM53F_COPY_TOKENS` tokens a request (128).
#[test]
#[ignore]
fn copy_windows_tokens_per_round() {
    if loaded_layers() < 45 {
        eprintln!("skip: set GLM53F_DRAFT_TEST_LAYERS=45 (a forward that repeats its context)");
        return;
    }
    let Some((codec, prompts)) = prompts() else {
        return;
    };
    let cfg = ForwardConfig { max_rows: 256, max_verify_rows: 16, max_requests: 3, ..ForwardConfig::default() };
    let Some(mut m) = forward(cfg) else {
        return;
    };
    let max: usize = std::env::var("GLM53F_COPY_TOKENS").ok().and_then(|v| v.parse().ok()).unwrap_or(128);
    m.fwd.set_lane_trace(true, false);
    for (name, heavy, prompt) in &prompts {
        eprintln!("{name} ({}), prompt {} tokens:", if *heavy { "copy-heavy" } else { "control" }, prompt.len());
        let mut outs = Vec::new();
        for mode in [Mode::Drafts, Mode::Copies] {
            let r = run(&mut m, mode, SpecPolicy::default(), &[(prompt.clone(), max, None)]);
            let n = r.out[0].len();
            let steps = r.st.spec_steps.max(1) as f64;
            eprintln!(
                "  {}: {:.2} tokens and {:.2} rows a round ({n} tokens, {} rounds; {} copied windows, {} of {} copied \
                 tokens kept; {} drafted); a step: draft {:.2} ms, verify and commit {:.1} ms; {:.1} ms a token",
                if mode == Mode::Copies { "copies on " } else { "copies off" },
                (n - 1) as f64 / steps,
                (r.st.drafts_verified + r.st.windows) as f64 / steps,
                r.st.spec_steps,
                r.st.copy_windows,
                r.st.copies_accepted,
                r.st.copies_verified,
                r.ds.drafted,
                r.draft_ms / steps,
                r.pass_ms / steps,
                (r.draft_ms + r.pass_ms) / (n - 1) as f64
            );
            outs.push(r.out);
        }
        assert_eq!(outs[0], outs[1], "{name}: copy windows changed the tokens");
        let text = String::from_utf8_lossy(&codec.decode_bytes(&outs[0][0])).into_owned();
        eprintln!("  text: {:?}", text.chars().take(240).collect::<String>());
    }
}
