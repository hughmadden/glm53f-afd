//! Real prompts for the drafter's acceptance replay (`glm53f-dflash`'s `examples/draft_replay.rs`):
//! each case, a chat request rendered with the official template (thinking off) and a reference
//! reply, is run teacher-forced through the whole model on this GPU (all 45 decoder layers and
//! the head, the official FP8 routed experts loaded on demand) with the drafter attached, in
//! prefill passes: every row's taps (what a slot's drafter context holds after the row is
//! committed) and the target's greedy pick after the row are written out.
//!
//! ```sh
//! GLM53F_CHECKPOINT_DIR=<coordinator tensors> GLM53F_EXPERTS_DIR=<the official checkpoint> \
//! GLM53F_DFLASH_DIR=<drafter> GLM53F_TOKENIZER=<tokenizer.json> GLM53F_RECORD_OUT=<dir> \
//!   cargo run --release -p glm53f-forward --features coordinator --example draft_record
//! ```
//!
//! - `GLM53F_RECORD_KDA_FP8=1`: the KDA projections in FP8 (decision D2), so the whole model and
//!   the drafter fit a 24 GB GPU next to other work; `GLM53F_RECORD_EXPERT_GIB` (default 2): the
//!   device cache of routed experts (a pass loads each layer's experts once);
//!   `GLM53F_RECORD_CASES`: some of the cases, comma-separated (`code`, `prose`, `counting`,
//!   `structured`, `rewrite`; default all).
//! - Every case fits one prefill pass (at most [`PASS`] rows), whose taps are the ones a serving
//!   prefill and 8-row verify windows give a greedy request's committed rows up to rounding (a
//!   window of up to 8 rows runs the decode kernels, a pass of more rows the tensor-core GEMMs).
//! - Output, per case, `<out>/<case>.rec`: `G53REC01`, then u32 little-endian `n` (rows),
//!   `prompt` (the prompt's rows), then `tokens[n]` u32, `picks[n]` u32 (the target's greedy pick
//!   after each row: the first maximum below 154,856), `taps[n][20480]` BF16.
//!
//! Teacher forcing makes the replay exact where the reply is the target's own greedy text and a
//! lower bound elsewhere (`draft_replay.rs`).

use std::io::Write;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

use glm53f_api::engine::PromptOptions;
use glm53f_api::json::Json;
use glm53f_api::types::ChatRequest;
use glm53f_coordinator::engine::PromptCodec;
use glm53f_coordinator::GlmPrompts;
use glm53f_forward::device::{self, Stream};
use glm53f_forward::draft::{Dflash, TAP_WIDTH};
use glm53f_forward::embed::HostEmbedding;
use glm53f_forward::experts::LocalFp8Experts;
use glm53f_forward::forward::{ForwardConfig, GlmForward};
use glm53f_forward::gemm::Fp8Act;
use glm53f_forward::kv::{KvConfig, KvPool};
use glm53f_forward::kvplan::KvLayout;
use glm53f_forward::shape::{ModelShape, SAMPLE_VOCAB, VOCAB};
use glm53f_forward::weights::{open_checkpoint, DeviceModel, WeightOptions};

/// Rows of the prefill pass (a case's prompt and reply together at most).
const PASS: usize = 2048;

const SPEC: &str = include_str!("../../glm53f-coordinator/src/spec.rs");
const QUEUE: &str = include_str!("../../glm53f-coordinator/src/queue.rs");
const DESIGN: &str = include_str!("../../../docs/DESIGN.md");

fn s(x: &str) -> Json {
    Json::Str(x.to_string())
}

fn request(text: String) -> Json {
    Json::Object(vec![
        ("model".to_string(), s("glm-5.3-flash")),
        (
            "messages".to_string(),
            Json::Array(vec![Json::Object(vec![
                ("role".to_string(), s("user")),
                ("content".to_string(), Json::Str(text)),
            ])]),
        ),
    ])
}

/// The first `n` lines of `text`.
fn lines(text: &str, n: usize) -> String {
    text.lines().take(n).collect::<Vec<_>>().join("\n")
}

/// (name, request, reference reply).
fn cases() -> Vec<(&'static str, Json, String)> {
    let counting: Vec<String> = (1..=400).map(|i| i.to_string()).collect();
    let rows: Vec<(u32, String, u32)> = (1..=40)
        .map(|i| (i, format!("item-{:03}", i * 7 % 101), (i * 37) % 1000))
        .collect();
    let csv: String = std::iter::once("id,name,value".to_string())
        .chain(rows.iter().map(|(i, n, v)| format!("{i},{n},{v}")))
        .collect::<Vec<_>>()
        .join("\n");
    let json: String = format!(
        "```json\n[\n{}\n]\n```",
        rows.iter()
            .map(|(i, n, v)| format!("  {{\"id\": {i}, \"name\": \"{n}\", \"value\": {v}}}"))
            .collect::<Vec<_>>()
            .join(",\n")
    );
    vec![
        (
            "code",
            request(
                "Write a Rust module that decides how many speculative drafts each request \
                 verifies per step: a fixed policy, an adaptive one from the drafter's \
                 probabilities under a cost model, and a chain cut at a threshold, plus a row \
                 budget that drops the least likely drafts first. Reply with the code only."
                    .into(),
            ),
            format!("```rust\n{}\n```", lines(SPEC, 150)),
        ),
        (
            "prose",
            request(
                "Explain the design of a GLM-5.3-Flash inference engine that splits attention \
                 and experts across machines, in a few sections of prose."
                    .into(),
            ),
            lines(DESIGN, 90),
        ),
        (
            "counting",
            request("Count from 1 to 400, separated by commas and spaces. No other text.".into()),
            counting.join(", "),
        ),
        (
            "structured",
            request(format!(
                "Convert this CSV to a JSON array of objects with the keys id, name and value \
                 (numbers as numbers). Reply with the JSON only.\n\n{csv}"
            )),
            json,
        ),
        (
            "rewrite",
            request(format!(
                "In this file, make the queue's default wait 30,000 ms instead of 25,000, in the \
                 code and in the comments that state it. Reply with the whole updated file.\n\n\
                 ```rust\n{}\n```",
                lines(QUEUE, 60)
            )),
            format!(
                "```rust\n{}\n```",
                lines(QUEUE, 60)
                    .replace("25,000", "30,000")
                    .replace("25_000", "30_000")
            ),
        ),
    ]
}

fn env_path(k: &str) -> Option<PathBuf> {
    std::env::var_os(k)
        .filter(|v| !v.is_empty())
        .map(PathBuf::from)
}

fn main() {
    let (Some(dir), Some(edir), Some(ddir), Some(tok), Some(out)) = (
        env_path("GLM53F_CHECKPOINT_DIR"),
        env_path("GLM53F_EXPERTS_DIR"),
        env_path("GLM53F_DFLASH_DIR"),
        env_path("GLM53F_TOKENIZER"),
        env_path("GLM53F_RECORD_OUT"),
    ) else {
        eprintln!(
            "set GLM53F_CHECKPOINT_DIR, GLM53F_EXPERTS_DIR (every layer's routed experts), \
             GLM53F_DFLASH_DIR, GLM53F_TOKENIZER and GLM53F_RECORD_OUT"
        );
        return;
    };
    let on = |k: &str| std::env::var(k).is_ok_and(|v| v != "0");
    let kda_fp8 = on("GLM53F_RECORD_KDA_FP8");
    let expert_gib: f64 = std::env::var("GLM53F_RECORD_EXPERT_GIB")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(2.0);
    std::fs::create_dir_all(&out).unwrap();
    let codec = GlmPrompts::load(&tok, &dir.join("chat_template.jinja")).unwrap();
    let stops = codec.stop_ids().unwrap();
    let opts = PromptOptions {
        thinking: false,
        reasoning_effort: None,
        clear_thinking: None,
    };
    let no_images =
        |_: &str| -> Result<glm53f_api::engine::ImageInput, String> { Err("no images".into()) };
    let t0 = Instant::now();
    let (cfg, ckpt) = open_checkpoint(&dir).unwrap();
    let shape = ModelShape::full(&cfg.text).unwrap();
    let wopts = WeightOptions {
        kda_fp8,
        ..WeightOptions::default()
    };
    let model = DeviceModel::load_with(&ckpt, &shape, shape.layers, wopts).unwrap();
    let embed = HostEmbedding::load(&ckpt).unwrap();
    let stream = Arc::new(Stream::new().unwrap());
    let d = Dflash::load(&ddir, &model, &embed, &stream).unwrap();
    let gib = |b: usize| b as f64 / (1u64 << 30) as f64;
    eprintln!(
        "weights {:.2} GiB, the drafter's {:.2} GiB; {:.1} GiB free",
        gib(model.bytes),
        gib(d.weight_bytes()),
        gib(device::mem_info().unwrap().0)
    );
    let layout = KvLayout::new(&shape, Some(d.config())).with_kda_state_bf16(true);
    let pages = KvLayout::pages_for(PASS);
    let kv = KvPool::new(
        KvConfig {
            layout,
            max_slots: 1,
            pages,
            max_pages: pages.div_ceil(4) * 4,
            base_pages: 0,
        },
        stream.clone(),
    )
    .unwrap();
    let experts = LocalFp8Experts::new(
        &edir,
        (expert_gib * (1u64 << 30) as f64) as usize,
        PASS,
        &stream,
        Fp8Act::Bf16,
    )
    .unwrap();
    let fcfg = ForwardConfig {
        max_rows: PASS,
        max_verify_rows: 8,
        max_requests: 1,
        ..ForwardConfig::default()
    };
    let mut fwd = GlmForward::new(model, embed, kv, Box::new(experts), fcfg).unwrap();
    fwd.attach_drafter(d).unwrap();
    let (free, _) = device::mem_info().unwrap();
    eprintln!(
        "the whole model (KDA projections {}) and the drafter loaded in {:.1} s; forward buffers \
         {:.2} GiB; {:.1} GiB free",
        if kda_fp8 { "FP8" } else { "BF16" },
        t0.elapsed().as_secs_f64(),
        gib(fwd.scratch_bytes()),
        gib(free)
    );
    let only = std::env::var("GLM53F_RECORD_CASES").ok();
    for (name, req, reply) in cases() {
        if only
            .as_ref()
            .is_some_and(|o| !o.split(',').any(|c| c.trim() == name))
        {
            continue;
        }
        let r = ChatRequest::parse(&req, &no_images).unwrap();
        let prompt = codec
            .encode(&codec.render(&r.messages, &r.tools, &opts).unwrap())
            .unwrap();
        let mut tokens = prompt.clone();
        tokens.extend(codec.tokenizer().encode(&reply));
        tokens.push(stops[1]);
        tokens.truncate(PASS);
        let n = tokens.len();
        let t = Instant::now();
        let mut kv = fwd.kv.slot().unwrap();
        kv.reserve(n).unwrap();
        let mut picks = vec![0u32; n];
        let rows: Vec<usize> = (0..n).collect();
        fwd.score_each(&mut kv, &tokens, &rows, n, |r, logits: &[f32]| {
            let mut b = 0;
            for (i, &v) in logits[..SAMPLE_VOCAB].iter().enumerate() {
                if v > logits[b] {
                    b = i;
                }
            }
            debug_assert_eq!(logits.len(), VOCAB);
            picks[r] = b as u32;
            Ok(())
        })
        .unwrap();
        let taps = fwd.drafter().unwrap().taps(n).unwrap();
        drop(kv);
        let agree = (prompt.len()..n - 1)
            .filter(|&p| picks[p] == tokens[p + 1])
            .count();
        let mut f = std::fs::File::create(out.join(format!("{name}.rec"))).unwrap();
        f.write_all(b"G53REC01").unwrap();
        for v in [n as u32, prompt.len() as u32] {
            f.write_all(&v.to_le_bytes()).unwrap();
        }
        let le = |v: &[u32]| v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>();
        f.write_all(&le(&tokens)).unwrap();
        f.write_all(&le(&picks)).unwrap();
        let tb: Vec<u8> = taps.iter().flat_map(|x| x.to_le_bytes()).collect();
        assert_eq!(tb.len(), n * TAP_WIDTH * 2);
        f.write_all(&tb).unwrap();
        eprintln!(
            "{name}: prompt {} + reply {} rows in {:.1} s; the target's greedy pick is the \
             reply's next token at {agree} of {} reply positions",
            prompt.len(),
            n - prompt.len(),
            t.elapsed().as_secs_f64(),
            n - 1 - prompt.len()
        );
    }
}
