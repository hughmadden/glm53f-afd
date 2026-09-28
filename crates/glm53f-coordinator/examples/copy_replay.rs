//! Copy windows on real text, replayed (`glm53f_coordinator::copy`): how often a copy is found and
//! how many of its tokens a greedy target accepts, when the target's reply is known.
//!
//! ```text
//! GLM53F_TOKENIZER=<dir>/tokenizer.json GLM53F_CHECKPOINT_DIR=<dir> \
//!     cargo run --release -p glm53f-coordinator --example copy_replay
//! ```
//!
//! No model runs. Each case is a chat request, parsed by the API and rendered with GLM-5.3-Flash's
//! chat template (thinking off; the template from `GLM53F_CHECKPOINT_DIR`) and tokenizer, and a
//! **reference reply**: the text a model following the instruction writes, built from this
//! repository's files (the file with the rename or edit applied, the quoted function, the tool
//! call), or for the controls text that is not in the prompt (a source file for fresh code, the
//! design document for fresh prose). A greedy target's verify pass accepts a copied token exactly
//! when it equals the reply's token there, so with the reply known a copy's accepted length is
//! exact. What the drafter would have proposed is not: that needs the model on the target hardware
//! (`docs/RUNNING.md`, copy windows).
//!
//! Two measures per case:
//!
//! - **Rounds, copy or decode.** A verify round starts after each delivered run: with a copy, it
//!   delivers the accepted copied tokens and the target's next one; without, one token (a plain
//!   decode step). Tokens per round against 1.0 is what copies alone deliver.
//! - **Every position.** A copy looked up at every position of the reply, as if a round started
//!   there: the share of positions with one, and by the match behind it (8-15, 16-31, 32-63, 64
//!   tokens) how often its first token is right and how many tokens it delivers.

use glm53f_api::engine::PromptOptions;
use glm53f_api::json::Json;
use glm53f_api::types::ChatRequest;
use glm53f_coordinator::copy::{CopyIndex, ENTRY, MATCH};
use glm53f_coordinator::engine::PromptCodec;
use glm53f_coordinator::{GlmPrompts, Token};

const SPEC: &str = include_str!("../src/spec.rs");
const QUEUE: &str = include_str!("../src/queue.rs");
const STREAMING: &str = include_str!("../src/streaming.rs");
const RADIX: &str = include_str!("../src/radix.rs");
const DESIGN: &str = include_str!("../../../docs/DESIGN.md");
/// The verify window's drafts at most.
const DRAFTS: usize = 7;

fn s(x: &str) -> Json {
    Json::Str(x.to_string())
}

fn obj(kv: Vec<(&str, Json)>) -> Json {
    Json::Object(kv.into_iter().map(|(k, v)| (k.to_string(), v)).collect())
}

fn user(text: String) -> Json {
    obj(vec![("role", s("user")), ("content", Json::Str(text))])
}

fn fenced(lang: &str, code: &str) -> String {
    format!("```{lang}\n{code}```")
}

/// The function `name` in `src`: its doc comment and body, up to the closing brace at column 0.
fn function(src: &str, name: &str) -> String {
    let at = src.find(&format!("pub fn {name}(")).expect("the function");
    let start = src[..at].rfind("\n\n").map_or(0, |i| i + 2);
    let end = at + src[at..].find("\n}\n").expect("its end") + 3;
    src[start..end].to_string()
}

/// A case: its name, whether its reply copies the prompt, the request, and the reference reply.
struct Case {
    name: &'static str,
    copy_heavy: bool,
    request: Json,
    reply: String,
}

fn cases() -> Vec<Case> {
    let tools = Json::Array(vec![
        obj(vec![
            ("type", s("function")),
            ("function", obj(vec![
                ("name", s("read_file")),
                ("description", s("Read a file.")),
                ("parameters", obj(vec![
                    ("type", s("object")),
                    ("properties", obj(vec![("path", obj(vec![("type", s("string"))]))])),
                    ("required", Json::Array(vec![s("path")])),
                ])),
            ])),
        ]),
        obj(vec![
            ("type", s("function")),
            ("function", obj(vec![
                ("name", s("edit_file")),
                ("description", s("Replace an exact string in a file. old_string must match the file exactly.")),
                ("parameters", obj(vec![
                    ("type", s("object")),
                    ("properties", obj(vec![
                        ("path", obj(vec![("type", s("string"))])),
                        ("old_string", obj(vec![("type", s("string"))])),
                        ("new_string", obj(vec![("type", s("string"))])),
                    ])),
                    ("required", Json::Array(vec![s("path"), s("old_string"), s("new_string")])),
                ])),
            ])),
        ]),
    ]);
    let path = "crates/glm53f-coordinator/src/streaming.rs";
    let old_fn = function(STREAMING, "flush_pending");
    let new_fn = old_fn.replace("stop_at.map_or(true, |p| pos < p)", "stop_at.is_none_or(|p| pos < p)");
    assert_ne!(old_fn, new_fn);
    let quoted = function(SPEC, "budget");
    let req = |messages: Vec<Json>, with_tools: bool| {
        let mut kv = vec![("model", s("glm-5.3-flash")), ("messages", Json::Array(messages))];
        if with_tools {
            kv.push(("tools", tools.clone()));
        }
        obj(kv)
    };
    let three = format!(
        "Here are three files of the project.\n\nsrc/spec.rs:\n{}\n\nsrc/queue.rs:\n{}\n\nsrc/streaming.rs:\n{}",
        fenced("rust", SPEC),
        fenced("rust", QUEUE),
        fenced("rust", STREAMING)
    );
    vec![
        Case {
            name: "rewrite: a file with one function renamed",
            copy_heavy: true,
            request: req(vec![user(format!(
                "Rename the function `chain_length` to `chain_cut` everywhere in this file (its definition and every \
                 call). Reply with the complete updated file only, no commentary.\n\n{}",
                fenced("rust", SPEC)
            ))], false),
            reply: fenced("rust", &SPEC.replace("chain_length", "chain_cut")),
        },
        Case {
            name: "edit, then the whole file",
            copy_heavy: true,
            request: req(vec![user(format!(
                "In this file, make the queue's default wait 30,000 ms instead of 25,000, in the code and in the \
                 comments that state it. Reply with the whole updated file.\n\n{}",
                fenced("rust", QUEUE)
            ))], false),
            reply: fenced("rust", &QUEUE.replace("25,000", "30,000").replace("25_000", "30_000")),
        },
        Case {
            name: "edit_file call quoting a function",
            copy_heavy: true,
            request: req(vec![
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
            ], true),
            reply: format!(
                "<tool_call>edit_file<arg_key>path</arg_key><arg_value>{path}</arg_value><arg_key>old_string</arg_key>\
                 <arg_value>{old_fn}</arg_value><arg_key>new_string</arg_key><arg_value>{new_fn}</arg_value></tool_call>"
            ),
        },
        Case {
            name: "quote a function, then two sentences",
            copy_heavy: true,
            request: req(vec![user(format!(
                "Copy the function `budget` from this file exactly as written (verbatim, in a rust code block), then \
                 explain in two sentences what it returns.\n\n{}",
                fenced("rust", SPEC)
            ))], false),
            reply: format!(
                "{}\n\nIt returns the number of drafts to verify for each request, the lengths the policy chose cut so \
                 that all the windows together fit the row budget. Drafts are dropped from the least likely up, and \
                 every request keeps at least its first row.",
                fenced("rust", &quoted)
            ),
        },
        Case {
            name: "three files in context, one rewritten with a rename",
            copy_heavy: true,
            request: req(vec![user(format!(
                "{three}\n\nIn src/queue.rs, rename the method `queued` to `in_queue` everywhere. Reply with the \
                 complete updated src/queue.rs only."
            ))], false),
            reply: fenced("rust", &QUEUE.replace("fn queued(", "fn in_queue(").replace(".queued()", ".in_queue()")),
        },
        Case {
            name: "control: fresh code",
            copy_heavy: false,
            request: req(vec![user(
                "Write a Rust module implementing a compressed trie over token ids that finds the longest stored \
                 prefix of a query and counts shared prefixes, with tests."
                    .to_string(),
            )], false),
            reply: fenced("rust", RADIX),
        },
        Case {
            name: "control: fresh prose",
            copy_heavy: false,
            request: req(vec![user(
                "Write a design document for an inference engine that serves a mixture-of-experts model from one GPU \
                 and four expert servers."
                    .to_string(),
            )], false),
            reply: DESIGN.to_string(),
        },
    ]
}

/// Copies by the match behind them.
fn bucket(m: usize) -> usize {
    match m {
        0..=15 => 0,
        16..=23 => 1,
        24..=31 => 2,
        32..=63 => 3,
        _ => 4,
    }
}

const BUCKETS: [&str; 5] = ["8-15", "16-23", "24-31", "32-63", "64"];

/// Tokens of `copy` equal to `reply` from `at`.
fn accepted(copy: &[Token], reply: &[Token], at: usize) -> usize {
    copy.iter().zip(&reply[at..]).take_while(|(a, b)| a == b).count()
}

/// Rounds over the reply: at each round's start an index whose copies need `entry` matching tokens
/// proposes; a copy delivers its accepted tokens and the target's next one, a round without one
/// delivers `d` tokens (1: a plain decode step; more: a stand-in for a drafter's round). The first
/// token comes from the prefill. Returns (rounds, copy rounds, copied tokens verified, copied
/// tokens kept, tokens the copy rounds delivered).
fn rounds(prompt: &[Token], reply: &[Token], bound: usize, entry: usize, d: usize) -> [usize; 5] {
    let n = reply.len();
    let mut hist = prompt.to_vec();
    hist.push(reply[0]);
    let mut idx = CopyIndex::with_entry(entry);
    let (mut at, mut r) = (1usize, [0usize; 5]);
    while at < n {
        let copy = idx.propose(&hist, (n - at - 1).min(DRAFTS), bound);
        let run = if copy.is_empty() {
            d.min(n - at)
        } else {
            let a = accepted(&copy, reply, at);
            r[1] += 1;
            r[2] += copy.len();
            r[3] += a;
            r[4] += a + 1;
            a + 1
        };
        hist.extend(&reply[at..at + run]);
        at += run;
        r[0] += 1;
    }
    r
}

fn main() {
    let tok = std::env::var("GLM53F_TOKENIZER");
    let dir = std::env::var("GLM53F_CHECKPOINT_DIR");
    let (Ok(tok), Ok(dir)) = (tok, dir) else {
        println!("skip: set GLM53F_TOKENIZER (tokenizer.json) and GLM53F_CHECKPOINT_DIR (chat_template.jinja)");
        return;
    };
    let codec = GlmPrompts::load(tok.as_ref(), &std::path::Path::new(&dir).join("chat_template.jinja")).unwrap();
    let bound = codec.id_bound();
    let stops = codec.stop_ids().unwrap();
    let opts = PromptOptions { thinking: false, reasoning_effort: None, clear_thinking: None };
    let no_images = |_: &str| -> Result<glm53f_api::engine::ImageInput, String> { Err("no images".into()) };
    println!(
        "copy windows replayed: up to {DRAFTS} copied tokens a round, an entry of {MATCH} (mimo26f-afd) or {ENTRY} (this \
         engine) matching tokens; a greedy target"
    );
    for c in cases() {
        let r = ChatRequest::parse(&c.request, &no_images).unwrap();
        let prompt = codec.encode(&codec.render(&r.messages, &r.tools, &opts).unwrap()).unwrap();
        let mut reply = codec.tokenizer().encode(&c.reply);
        // The turn ends: `<|observation|>` after a tool call, else `<|user|>`.
        reply.push(if c.reply.starts_with("<tool_call>") { stops[2] } else { stops[1] });
        let n = reply.len();
        println!(
            "\n{} ({}): prompt {} tokens, reply {n} tokens",
            c.name,
            if c.copy_heavy { "copy-heavy" } else { "control" },
            prompt.len()
        );
        for entry in [MATCH, ENTRY] {
            let [all, cr, ver, kept, ctok] = rounds(&prompt, &reply, bound, entry, 1);
            let [_, cr3, ver3, kept3, ctok3] = rounds(&prompt, &reply, bound, entry, 3);
            println!(
                "  entry {entry:2}, copy or decode: {:.2} tokens a round; {cr} copy rounds of {all}, {:.2} tokens and {:.2} \
                 rows each, {kept} of {ver} copied tokens kept ({:.1}%), {:.1}% of the reply",
                (n - 1) as f64 / all as f64,
                ctok as f64 / cr.max(1) as f64,
                (ver + cr) as f64 / cr.max(1) as f64,
                100.0 * kept as f64 / ver.max(1) as f64,
                100.0 * ctok as f64 / (n - 1) as f64
            );
            println!(
                "            other rounds at 3 tokens: {cr3} copy rounds, {:.2} tokens each, {} copied rows wasted, \
                 {:.1}% of the reply",
                ctok3 as f64 / cr3.max(1) as f64,
                ver3 - kept3,
                100.0 * ctok3 as f64 / (n - 1) as f64
            );
        }
        // Every position, every match of MATCH tokens or more.
        let mut hist = prompt.clone();
        let mut idx = CopyIndex::with_entry(MATCH);
        let (mut found, mut by) = (0usize, [(0usize, 0usize, 0usize, 0usize); 5]);
        for p in 1..n - 1 {
            hist.push(reply[p - 1]);
            let copy = idx.propose(&hist, (n - p - 1).min(DRAFTS), bound);
            if copy.is_empty() {
                continue;
            }
            found += 1;
            let a = accepted(&copy, &reply, p);
            let b = &mut by[bucket(idx.matched())];
            *b = (b.0 + 1, b.1 + usize::from(a > 0), b.2 + a + 1, b.3 + usize::from(a == copy.len()));
        }
        println!(
            "  every position: a copy of {MATCH}+ matching tokens at {found} of {} ({:.1}%)",
            n - 2,
            100.0 * found as f64 / (n - 2) as f64
        );
        for (name, (m, right, tokens, full)) in BUCKETS.iter().zip(by) {
            if m > 0 {
                println!(
                    "    match {name:>5}: {m:5} copies, first token right {:5.1}%, {:.2} tokens a copy, whole copy kept {:5.1}%",
                    100.0 * right as f64 / m as f64,
                    tokens as f64 / m as f64,
                    100.0 * full as f64 / m as f64
                );
            }
        }
    }
}
