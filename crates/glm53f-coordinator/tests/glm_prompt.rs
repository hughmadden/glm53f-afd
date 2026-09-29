//! GLM-5.3-Flash's prompt codec against the reference renders in `oracle/goldens/template/`: each
//! golden conversation goes in as a client sends it (tool-call arguments as JSON text), through the
//! API's own request parser, then `to_template` and the renderer; the text must equal the render
//! of `transformers`' `apply_chat_template`.
//!
//! One golden is expected to differ, and the test pins that it differs only there: the API keeps a
//! request's tools as its own JSON values, whose numbers are `f64`, so a tool schema's integral
//! float (`1.0`) renders as `1` (glm53f-tokenizer's template documentation notes it).
//!
//! With `GLM53F_TOKENIZER` naming the official checkpoint's tokenizer.json (its
//! chat_template.jinja beside it), the codec is also loaded as the engine loads it and its token
//! ids are checked against the goldens' ids; otherwise that part is skipped with a note.

use std::path::{Path, PathBuf};

use glm53f_api::engine::PromptOptions;
use glm53f_api::json::{self, Json};
use glm53f_api::types::ChatRequest;
use glm53f_coordinator::engine::PromptCodec;
use glm53f_coordinator::glm_prompt::{to_template, GlmPrompts};
use glm53f_tokenizer::json::{self as tjson, Value};
use glm53f_tokenizer::template;

/// Goldens whose tools carry an integral float the API's JSON cannot keep.
const API_NUMBER_LOSS: [&str; 1] = ["tools_strict_and_deferred"];

fn golden_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../oracle/goldens/template")
}

fn read(path: &Path) -> String {
    std::fs::read_to_string(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()))
}

fn load(path: &Path) -> Json {
    json::parse(&read(path)).unwrap_or_else(|e| panic!("{}: {e}", path.display()))
}

/// Tool-call arguments as a client sends them: JSON text (written from the lossless reading of
/// the golden, so `1.0` stays `1.0`).
fn arguments_as_text(v: &Value) -> Value {
    match v {
        Value::Object(pairs) => Value::Object(
            pairs
                .iter()
                .map(|(k, x)| match (k.as_str(), x) {
                    ("arguments", Value::Object(_)) => (k.clone(), Value::Str(tjson::to_python_json(x))),
                    _ => (k.clone(), arguments_as_text(x)),
                })
                .collect(),
        ),
        Value::Array(xs) => Value::Array(xs.iter().map(arguments_as_text).collect()),
        other => other.clone(),
    }
}

struct Case {
    name: String,
    req: ChatRequest,
    opts: PromptOptions,
    rendered: String,
    ids: Vec<u32>,
}

/// The golden cases a server can receive: a generation prompt, no media parts.
fn cases() -> Vec<Case> {
    let manifest = load(&golden_dir().join("manifest.json"));
    let names = manifest.get("cases").and_then(|c| c.as_array()).expect("case list").to_vec();
    let mut out = Vec::new();
    for n in names {
        let name = n.as_str().expect("name").to_string();
        let g = load(&golden_dir().join(format!("{name}.json")));
        if g.get("add_generation_prompt") != Some(&Json::Bool(true)) {
            continue;
        }
        let kwargs = g.get("kwargs").cloned().unwrap_or(Json::Object(Vec::new()));
        // The request body as a client would send it, then the API's own parser.
        let lossless = tjson::parse(&read(&golden_dir().join(format!("{name}.json")))).expect("golden");
        let mut body = vec![("model".to_string(), Value::Str("glm-5.3-flash".into()))];
        body.push(("messages".into(), arguments_as_text(lossless.get("messages").expect("messages"))));
        if let Some(t) = lossless.get("tools") {
            body.push(("tools".into(), t.clone()));
        }
        let body = json::parse(&tjson::to_python_json(&Value::Object(body))).expect("request body");
        // Media parts are refused by the API (and images need an encoder): not a server's input.
        let Ok(req) = ChatRequest::parse(&body, &|_| Err("no image encoder".into())) else { continue };
        let opts = PromptOptions {
            thinking: true,
            reasoning_effort: kwargs.get("reasoning_effort").and_then(|r| r.as_str()).map(str::to_string),
            clear_thinking: kwargs.get("clear_thinking").and_then(|c| if let Json::Bool(b) = c { Some(*b) } else { None }),
        };
        let ids = g.get("ids").and_then(|a| a.as_array()).map(|a| a.iter().map(|x| x.as_f64().unwrap() as u32).collect());
        out.push(Case { name, req, opts, rendered: g.get("rendered").and_then(|r| r.as_str()).unwrap().to_string(),
            ids: ids.unwrap_or_default() });
    }
    out
}

#[test]
fn api_requests_render_as_the_reference_template() {
    let cases = cases();
    assert!(cases.len() >= 30, "{} server-shaped golden cases", cases.len());
    let mut fails = Vec::new();
    for c in &cases {
        let (m, t, o) = to_template(&c.req.messages, &c.req.tools, &c.opts).unwrap_or_else(|e| panic!("{}: {e}", c.name));
        let got = template::render(&m, &t, &o).unwrap_or_else(|e| panic!("{}: {e}", c.name));
        if API_NUMBER_LOSS.contains(&c.name.as_str()) {
            // Only numbers inside the tools block differ (the case is a number-formatting torture
            // test: 1e+16, -0.0, 1.7976931348623157e+308, 12345678901234567890, ...).
            assert_ne!(got, c.rendered, "{}: the API's number loss is fixed; drop the exception", c.name);
            let head = |s: &str| s.split("\"minimum\"").next().map(str::to_string);
            let tail = |s: &str| s.rsplit("</tools>").next().map(str::to_string);
            assert_eq!(head(&got), head(&c.rendered), "{}", c.name);
            assert_eq!(tail(&got), tail(&c.rendered), "{}", c.name);
            continue;
        }
        if got != c.rendered {
            let at = got.bytes().zip(c.rendered.bytes()).position(|(a, b)| a != b).unwrap_or(got.len().min(c.rendered.len()));
            let clip = |s: &str| s.get(at.saturating_sub(30)..(at + 30).min(s.len())).unwrap_or("").to_string();
            fails.push(format!("{}: differs at byte {at}: got {:?} want {:?}", c.name, clip(&got), clip(&c.rendered)));
        }
    }
    assert!(fails.is_empty(), "{} of {} cases: {fails:#?}", fails.len(), cases.len());
}

#[test]
fn tool_call_arguments_that_are_not_an_object_are_an_error() {
    let body = json::parse(r#"{"messages": [{"role": "assistant", "content": "",
        "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "f", "arguments": "[1, 2]"}}]}]}"#).unwrap();
    let req = ChatRequest::parse(&body, &|_| Err(String::new())).unwrap();
    let e = to_template(&req.messages, &req.tools, &PromptOptions::default()).unwrap_err();
    assert!(e.contains("messages[0]") && e.contains("not a JSON object"), "{e}");
}

/// The effort requests end to end, through the API's handler and the real chat template: thinking
/// off, and a `reasoning_effort` of "none" or "minimal" (top level or in `chat_template_kwargs`,
/// with thinking switched on too), render the same prompt as "low": the Low effort with the think
/// block open, never an empty one. "high" renders High; "max", "medium" and a request that says
/// nothing render Max, as the template does.
#[test]
fn effort_requests_render_the_templates_efforts() {
    use std::sync::{Arc, Mutex};

    use glm53f_api::dialect::GlmDialect;
    use glm53f_api::engine::{Engine, GenerateOutcome, GenerateParams};
    use glm53f_api::types::{ChatMessage, Tool};

    /// Renders each prompt with the real template and keeps it.
    struct Renders(Mutex<Vec<String>>);
    impl Engine for Renders {
        fn tokenize(&self, _: &[ChatMessage], _: &[Tool], _: bool) -> usize {
            1
        }
        fn render_chat(&self, m: &[ChatMessage], t: &[Tool], thinking: bool) -> String {
            self.render_prompt(m, t, &PromptOptions { thinking, ..Default::default() })
        }
        fn render_prompt(&self, m: &[ChatMessage], t: &[Tool], opts: &PromptOptions) -> String {
            let (m, t, o) = to_template(m, t, opts).unwrap();
            let text = template::render(&m, &t, &o).unwrap();
            self.0.lock().unwrap().push(text.clone());
            text
        }
        fn generate(&self, _: &str, _: &GenerateParams, _: &mut dyn FnMut(&str)) -> Result<GenerateOutcome, String> {
            Ok(GenerateOutcome { text: "ok".into(), finish_reason: "stop".into(), completion_tokens: 1 })
        }
    }

    let engine = Arc::new(Renders(Mutex::new(Vec::new())));
    let dialect: Arc<dyn glm53f_api::Dialect> = Arc::new(GlmDialect);
    let served = |extra: &str| {
        let sep = if extra.is_empty() { "" } else { "," };
        let body = json::parse(&format!(r#"{{"messages":[{{"role":"user","content":"hi"}}]{sep}{extra}}}"#)).unwrap();
        glm53f_api::chat::handle(engine.clone(), dialect.clone(), &body).unwrap_or_else(|e| panic!("{extra}: {e:?}"));
        engine.0.lock().unwrap().pop().expect("the prompt was rendered")
    };
    let rendered = |thinking: bool, effort: Option<&str>| {
        let msgs = [ChatMessage { role: "user".into(), content: "hi".into(), tool_calls: Vec::new(), reasoning_content: None,
            tool_call_id: None }];
        let opts = PromptOptions { thinking, reasoning_effort: effort.map(str::to_string), clear_thinking: None };
        let (m, t, o) = to_template(&msgs, &[], &opts).unwrap();
        template::render(&m, &t, &o).unwrap()
    };

    let low = rendered(true, Some("low"));
    assert!(low.contains("Reasoning Effort: Low") && low.ends_with(template::THINK), "{low}");
    for extra in [
        r#""reasoning_effort":"low""#,
        r#""reasoning_effort":"none""#,
        r#""reasoning_effort":"minimal""#,
        r#""chat_template_kwargs":{"reasoning_effort":"none"}"#,
        r#""chat_template_kwargs":{"reasoning_effort":"minimal"}"#,
        r#""chat_template_kwargs":{"enable_thinking":false}"#,
        r#""thinking":{"type":"disabled"}"#,
        r#""chat_template_kwargs":{"enable_thinking":true},"reasoning_effort":"none""#,
        r#""thinking":{"type":"enabled"},"reasoning_effort":"minimal""#,
    ] {
        assert_eq!(served(extra), low, "{extra}");
    }
    let high = rendered(true, Some("high"));
    assert!(high.contains("Reasoning Effort: High") && high.ends_with(template::THINK), "{high}");
    assert_eq!(served(r#""reasoning_effort":"high""#), high);
    let max = rendered(true, None);
    assert!(max.contains("Reasoning Effort: Max") && max.ends_with(template::THINK), "{max}");
    for extra in ["", r#""reasoning_effort":"max""#, r#""reasoning_effort":"medium""#, r#""chat_template_kwargs":{"enable_thinking":true}"#] {
        assert_eq!(served(extra), max, "{extra:?}");
    }
}

#[test]
fn the_official_tokenizer_encodes_the_goldens_ids() {
    let Ok(path) = std::env::var("GLM53F_TOKENIZER") else {
        eprintln!("skipped: set GLM53F_TOKENIZER to the official checkpoint's tokenizer.json");
        return;
    };
    let template_path = Path::new(&path).with_file_name("chat_template.jinja");
    let codec = GlmPrompts::load(Path::new(&path), &template_path).expect("the official tokenizer and template");
    assert_eq!(codec.stop_ids().unwrap(), vec![154_820, 154_827, 154_829]);
    assert_eq!(codec.id_bound(), 154_856);
    let mut checked = 0;
    for c in cases().iter().filter(|c| !c.ids.is_empty() && !API_NUMBER_LOSS.contains(&c.name.as_str())) {
        let text = codec.render(&c.req.messages, &c.req.tools, &c.opts).unwrap();
        assert_eq!(codec.encode(&text).unwrap(), c.ids, "{}", c.name);
        checked += 1;
    }
    assert!(checked >= 30, "{checked} cases with ids");
    // A template that is not the official one is refused.
    let copy = std::env::temp_dir().join(format!("glm53f-coordinator-template-{}.jinja", std::process::id()));
    let official = std::fs::read_to_string(&template_path).unwrap();
    std::fs::write(&copy, official.replacen("Reasoning", "reasoning", 1)).unwrap();
    let refused = GlmPrompts::load(Path::new(&path), &copy);
    let _ = std::fs::remove_file(&copy);
    assert!(refused.is_err());
}
