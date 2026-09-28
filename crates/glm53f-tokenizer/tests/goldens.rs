//! The tokenizer and the chat template against the reference goldens in oracle/goldens/
//! (written by oracle/tokenizer_goldens.py and oracle/template_goldens.py from the reference
//! `tokenizers` and `transformers`).
//!
//! Always run (the goldens are committed): the pre-tokenizer pieces, the character tables and
//! contraction folds, and the chat-template renders.
//!
//! Run when `GLM53F_TOKENIZER` names the checkpoint's tokenizer.json, else skipped with a note:
//! encode, decode (both skip settings), streaming decode, the stop ids, and the token ids of
//! every rendered template.

use std::path::{Path, PathBuf};
use std::sync::OnceLock;

use glm53f_tokenizer::json::{self, Value};
use glm53f_tokenizer::template::{self, Content, Message, Options, Part, ToolCall};
use glm53f_tokenizer::{pretok, unicode, StreamDecoder, Tokenizer, STOP_IDS, STOP_TOKENS};

fn golden_dir(sub: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../oracle/goldens").join(sub)
}

fn load(path: &Path) -> Value {
    let text = std::fs::read_to_string(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    json::parse(&text).unwrap_or_else(|e| panic!("{}: {e}", path.display()))
}

fn tokenizer() -> Option<&'static Tokenizer> {
    static TOK: OnceLock<Option<Tokenizer>> = OnceLock::new();
    TOK.get_or_init(|| match std::env::var("GLM53F_TOKENIZER") {
        Ok(p) if Path::new(&p).is_file() => Some(Tokenizer::from_file(&p).expect("load GLM53F_TOKENIZER")),
        _ => None,
    })
    .as_ref()
    .or_else(|| {
        eprintln!("skipped: set GLM53F_TOKENIZER to the checkpoint's tokenizer.json");
        None
    })
}

fn str_of<'a>(v: &'a Value, key: &str) -> &'a str {
    v.get(key).and_then(|s| s.as_str()).unwrap_or_else(|| panic!("no string {key}"))
}

fn ids_of(v: &Value, key: &str) -> Vec<u32> {
    v.get(key)
        .and_then(|a| a.as_array())
        .unwrap_or_else(|| panic!("no array {key}"))
        .iter()
        .map(|x| x.as_u64().expect("id") as u32)
        .collect()
}

/// (id, text) of every tokenizer case, with its reference outputs.
fn tokenizer_cases() -> Vec<(String, String, Value)> {
    let dir = golden_dir("tokenizer");
    let cases = load(&dir.join("cases.json"));
    let outs = load(&dir.join("ids.json"));
    let cases = cases.as_array().expect("cases");
    let outs = outs.as_array().expect("ids");
    assert_eq!(cases.len(), outs.len());
    assert!(cases.len() >= 100, "{} cases", cases.len());
    cases
        .iter()
        .zip(outs)
        .map(|(c, o)| {
            assert_eq!(str_of(c, "id"), str_of(o, "id"));
            (str_of(c, "id").to_string(), str_of(c, "text").to_string(), o.clone())
        })
        .collect()
}

fn report(what: &str, fails: &[String], total: usize) {
    assert!(fails.is_empty(), "{what}: {} of {total} differ; first:\n{}", fails.len(), fails.iter().take(5).cloned().collect::<Vec<_>>().join("\n"));
    eprintln!("{what}: {total} of {total} match");
}

#[test]
fn pretokenizer_pieces_match_the_reference() {
    let cases = tokenizer_cases();
    let mut fails = Vec::new();
    for (id, text, out) in &cases {
        let want: Vec<&str> = out.get("pieces").and_then(|p| p.as_array()).expect("pieces").iter().map(|p| p.as_str().unwrap()).collect();
        let got = pretok::split_str(text);
        if got != want {
            let at = got.iter().zip(&want).position(|(a, b)| a != b).unwrap_or(got.len().min(want.len()));
            fails.push(format!("{id}: piece {at}: got {:?}, want {:?}", got.get(at), want.get(at)));
        }
    }
    report("pre-tokenizer cases", &fails, cases.len());
}

#[test]
fn character_tables_match_the_reference_probe() {
    let doc = load(&golden_dir("tokenizer").join("unicode_classes.json"));
    for (key, table) in unicode::class_tables() {
        let want: Vec<(u32, u32)> = doc
            .get(key)
            .and_then(|r| r.as_array())
            .expect("ranges")
            .iter()
            .map(|r| {
                let r = r.as_array().unwrap();
                (r[0].as_u64().unwrap() as u32, r[1].as_u64().unwrap() as u32)
            })
            .collect();
        assert_eq!(table, want.as_slice(), "class {key}");
    }
    let folds = doc.get("contraction_folds").expect("contraction_folds");
    for (letter, cps) in pretok::contraction_folds() {
        let want: Vec<u32> = folds.get(&letter.to_string()).and_then(|a| a.as_array()).expect("fold").iter().map(|x| x.as_u64().unwrap() as u32).collect();
        assert_eq!(cps, want.as_slice(), "fold {letter}");
    }
    for pair in ["re", "ve", "ll"] {
        assert_eq!(folds.get(pair).and_then(|a| a.as_array()).map(|a| a.len()), Some(0), "no single letter matches {pair}");
    }
}

#[test]
fn encode_matches_the_reference() {
    let Some(tok) = tokenizer() else { return };
    let cases = tokenizer_cases();
    let mut fails = Vec::new();
    for (id, text, out) in &cases {
        let want = ids_of(out, "ids");
        let got = tok.encode(text);
        if got != want {
            let at = got.iter().zip(&want).position(|(a, b)| a != b).unwrap_or(got.len().min(want.len()));
            fails.push(format!("{id}: token {at}: got {:?}, want {:?}", got.get(at), want.get(at)));
        }
    }
    report("encode cases", &fails, cases.len());
}

/// Every id sequence with reference decodes: the tokenizer cases and the random sequences.
fn decode_cases() -> Vec<(String, Vec<u32>, String, String)> {
    let mut all = Vec::new();
    for (id, _, out) in tokenizer_cases() {
        all.push((id, ids_of(&out, "ids"), str_of(&out, "decoded").to_string(), str_of(&out, "decoded_skip_special").to_string()));
    }
    for d in load(&golden_dir("tokenizer").join("decode.json")).as_array().expect("decode cases") {
        all.push((str_of(d, "id").to_string(), ids_of(d, "ids"), str_of(d, "decoded").to_string(), str_of(d, "decoded_skip_special").to_string()));
    }
    all
}

#[test]
fn decode_matches_the_reference() {
    let Some(tok) = tokenizer() else { return };
    let cases = decode_cases();
    let mut fails = Vec::new();
    for (id, ids, want, want_skip) in &cases {
        if &tok.decode(ids, false) != want {
            fails.push(format!("{id}: decode differs"));
        }
        if &tok.decode(ids, true) != want_skip {
            fails.push(format!("{id}: decode(skip_special) differs"));
        }
    }
    report("decode cases (x2 skip settings)", &fails, cases.len());
}

#[test]
fn streaming_decode_equals_the_reference_decode() {
    let Some(tok) = tokenizer() else { return };
    let cases = decode_cases();
    let mut fails = Vec::new();
    for (id, ids, want, want_skip) in &cases {
        for (skip, want) in [(false, want), (true, want_skip)] {
            let mut s = StreamDecoder::new(tok, skip);
            let mut text = String::new();
            for &t in ids {
                text.push_str(&s.push(t));
            }
            text.push_str(&s.finish());
            if &text != want {
                fails.push(format!("{id} (skip {skip}): streamed {text:?}, want {want:?}"));
            }
        }
    }
    report("streaming decode cases (x2 skip settings)", &fails, cases.len());
}

/// The checkpoint's chat template (next to its tokenizer.json) is the one the renderer
/// reproduces; any other text fails the drift guard.
#[test]
fn chat_template_is_the_one_rendered() {
    let Ok(path) = std::env::var("GLM53F_TOKENIZER") else {
        eprintln!("skipped: set GLM53F_TOKENIZER to the checkpoint's tokenizer.json");
        return;
    };
    let template_path = Path::new(&path).with_file_name("chat_template.jinja");
    let Ok(text) = std::fs::read_to_string(&template_path) else {
        eprintln!("skipped: no chat_template.jinja next to GLM53F_TOKENIZER");
        return;
    };
    template::check_template(&text).unwrap();
    // One changed letter (same length), one added space, nothing.
    let changed = text.replacen("Reasoning", "reasoning", 1);
    assert_ne!(changed, text);
    assert!(template::check_template(&changed).is_err());
    assert!(template::check_template(&format!("{text} ")).is_err());
    assert!(template::check_template("").is_err());
}

#[test]
fn stop_tokens_and_shape() {
    let Some(tok) = tokenizer() else { return };
    assert_eq!(tok.stop_ids().unwrap(), STOP_IDS);
    for (id, text) in STOP_IDS.iter().zip(STOP_TOKENS) {
        assert!(tok.is_special(*id));
        assert_eq!(tok.decode(&[*id], false), text);
        assert_eq!(tok.decode(&[*id], true), "");
    }
    assert_eq!(tok.vocab_len(), 154_820);
    assert_eq!(tok.merges_len(), 321_649);
    assert_eq!(tok.added_tokens().len(), 36);
    assert_eq!(tok.id_bound(), 154_856);
    // The think and tool-call tags are added tokens but not special: they survive skip_special.
    let think = concat!("<", "think", ">");
    let id = tok.token_to_id(think).expect("think tag");
    assert_eq!(id, 154_841);
    assert_eq!(tok.decode(&[id], true), think);
}

// --- chat template -------------------------------------------------------------------------

fn part(v: &Value) -> Option<Part> {
    if let Some(s) = v.as_str() {
        return Some(Part::Text(s.to_string()));
    }
    match v.get("type").and_then(|t| t.as_str()) {
        Some("text") => Some(Part::Text(v.get("text").and_then(|t| t.as_str()).unwrap_or("").to_string())),
        Some("image" | "image_url") => Some(Part::Image(None)),
        Some("video" | "video_url") => Some(Part::Video),
        Some("audio" | "audio_url" | "input_audio") => Some(Part::Audio),
        _ => None,
    }
}

fn content(v: Option<&Value>) -> Content {
    match v {
        None | Some(Value::Null) => Content::Null,
        Some(Value::Str(s)) => Content::Text(s.clone()),
        Some(Value::Array(parts)) => Content::Parts(parts.iter().filter_map(part).collect()),
        Some(other) => panic!("unexpected content {other:?}"),
    }
}

fn message(v: &Value) -> Message {
    let calls = v.get("tool_calls").and_then(|t| t.as_array()).unwrap_or(&[]);
    Message {
        role: str_of(v, "role").to_string(),
        content: content(v.get("content")),
        reasoning_content: v.get("reasoning_content").and_then(|r| r.as_str()).map(str::to_string),
        tool_calls: calls
            .iter()
            .map(|c| {
                let f = c.get("function").expect("function");
                ToolCall {
                    id: c.get("id").and_then(|i| i.as_str()).map(str::to_string),
                    name: str_of(f, "name").to_string(),
                    arguments: f.get("arguments").cloned().expect("arguments"),
                }
            })
            .collect(),
        tool_call_id: v.get("tool_call_id").and_then(|i| i.as_str()).map(str::to_string),
    }
}

struct TemplateCase {
    name: String,
    messages: Vec<Message>,
    tools: Vec<Value>,
    opts: Options,
    rendered: String,
    ids: Vec<u32>,
    with_answer: Option<String>,
    kwargs: Value,
}

fn template_cases() -> Vec<TemplateCase> {
    let dir = golden_dir("template");
    let manifest = load(&dir.join("manifest.json"));
    let names = manifest.get("cases").and_then(|c| c.as_array()).expect("case list");
    assert!(names.len() >= 40, "{} template cases", names.len());
    names
        .iter()
        .map(|n| {
            let name = n.as_str().unwrap().to_string();
            let g = load(&dir.join(format!("{name}.json")));
            let kwargs = g.get("kwargs").cloned().unwrap_or(Value::Object(Vec::new()));
            let opts = Options {
                add_generation_prompt: g.get("add_generation_prompt").and_then(|b| b.as_bool()).expect("flag"),
                thinking: true,
                reasoning_effort: kwargs.get("reasoning_effort").and_then(|r| r.as_str()).map(str::to_string),
                clear_thinking: kwargs.get("clear_thinking").and_then(|c| c.as_bool()).unwrap_or(false),
            };
            TemplateCase {
                messages: g.get("messages").and_then(|m| m.as_array()).expect("messages").iter().map(message).collect(),
                tools: g.get("tools").and_then(|t| t.as_array()).map(|t| t.to_vec()).unwrap_or_default(),
                opts,
                rendered: str_of(&g, "rendered").to_string(),
                ids: ids_of(&g, "ids"),
                with_answer: g.get("rendered_with_answer").and_then(|r| r.as_str()).map(str::to_string),
                kwargs,
                name,
            }
        })
        .collect()
}

fn first_difference(a: &str, b: &str) -> String {
    let at = a.bytes().zip(b.bytes()).position(|(x, y)| x != y).unwrap_or(a.len().min(b.len()));
    let from = at.saturating_sub(40);
    let clip = |s: &str| s.get(from..(at + 40).min(s.len())).map(str::to_string).unwrap_or_else(|| "(not at a char boundary)".into());
    format!("byte {at}: got ...{:?}... want ...{:?}...", clip(a), clip(b))
}

#[test]
fn template_renders_match_the_reference() {
    let cases = template_cases();
    let mut fails = Vec::new();
    for c in &cases {
        match template::render(&c.messages, &c.tools, &c.opts) {
            Ok(got) if got == c.rendered => {}
            Ok(got) => fails.push(format!("{}: {}", c.name, first_difference(&got, &c.rendered))),
            Err(e) => fails.push(format!("{}: error {e}", c.name)),
        }
    }
    report("template renders", &fails, cases.len());
}

/// Thinking off (not a template feature): the prompt must be exactly the text the template
/// writes before the content of a reasoning-free assistant turn.
#[test]
fn thinking_off_prompt_is_the_templates_reasoning_free_turn() {
    let cases = template_cases();
    let mut checked = 0;
    let mut fails = Vec::new();
    for c in cases.iter().filter(|c| c.opts.add_generation_prompt) {
        let Some(want) = &c.with_answer else { continue };
        let off = Options { thinking: false, ..c.opts.clone() };
        let got = template::render(&c.messages, &c.tools, &off).expect("render") + "ANSWER";
        if &got != want {
            fails.push(format!("{}: {}", c.name, first_difference(&got, want)));
        }
        checked += 1;
    }
    report("thinking-off prompts", &fails, checked);
    assert!(checked >= 30);
}

/// The reference ignores `enable_thinking`: a render with it false equals the default render.
#[test]
fn the_reference_ignores_enable_thinking() {
    let cases = template_cases();
    let c = cases.iter().find(|c| c.name == "enable_thinking_false_is_ignored").expect("case");
    assert_eq!(c.kwargs.get("enable_thinking"), Some(&Value::Bool(false)));
    let single = cases.iter().find(|c| c.name == "effort_max").expect("case");
    assert_eq!(c.rendered, single.rendered);
    assert!(c.rendered.ends_with(concat!("<", "think", ">")));
}

#[test]
fn template_token_ids_match_the_reference() {
    let Some(tok) = tokenizer() else { return };
    let cases = template_cases();
    let mut fails = Vec::new();
    for c in &cases {
        let text = template::render(&c.messages, &c.tools, &c.opts).expect("render");
        if tok.encode(&text) != c.ids {
            fails.push(format!("{}: ids differ", c.name));
        }
    }
    report("template token ids", &fails, cases.len());
}

/// Images as markers: the engine's marker expands to the image's tokens, and with `<|image|>`
/// as the expansion the ids equal the reference's for the template's placeholder text.
#[test]
fn image_markers_expand_where_the_placeholder_was() {
    let Some(tok) = tokenizer() else { return };
    let cases = template_cases();
    let c = cases.iter().find(|c| c.name == "images_as_markers").expect("case");
    let (open, close) = (char::from_u32(0xFDD0).unwrap(), char::from_u32(0xFDD1).unwrap());
    let marker = format!("{open}0123456789abcdef:1{close}");
    let marked: Vec<Message> = c
        .messages
        .iter()
        .map(|m| {
            let mut m = m.clone();
            if let Content::Parts(parts) = &mut m.content {
                for p in parts.iter_mut() {
                    if let Part::Image(slot) = p {
                        *slot = Some(marker.clone());
                    }
                }
            }
            m
        })
        .collect();
    let text = template::render(&marked, &c.tools, &c.opts).unwrap();
    let image_id = tok.token_to_id(template::IMAGE).expect("image token");
    let ids = tok
        .encode_with_markers(&text, open, close, &mut |inner| {
            let n: usize = inner.split(':').nth(1).ok_or("marker")?.parse().map_err(|_| "count")?;
            Ok(vec![image_id; n])
        })
        .unwrap();
    assert_eq!(ids, c.ids);
}
