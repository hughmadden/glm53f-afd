//! Acceptance tests for `glm53f-api` (A8): the HTTP surface against a scripted
//! stub Engine speaking the MiMo reference dialect, the MiMo tool-call parser
//! goldens (T24/T29), and the L5 ladder runner driven against the loopback
//! server. The ladder comes from the harness directory (`harness/`, or
//! `GLM53F_HARNESS`); the tests that need it are skipped with a message while it
//! is absent. The last section serves the GLM dialect over a
//! scripted stand-in: one reasoning field in both modes, the chunk head on every
//! chunk, and the API contract rows (`harness/api_contract.py`).

use std::io::Read;
use std::net::TcpListener;
use std::path::PathBuf;
use std::process::Command;
use std::sync::Arc;

use glm53f_api::dialect::{Dialect, GlmDialect, MimoDialect};
use glm53f_api::engine::{Engine, GenerateOutcome, GenerateParams, PromptOptions};
use glm53f_api::types::{ChatMessage, Tool};
use glm53f_api::{self, MODEL_ID};

// ---------------------------------------------------------------------------
// Stub engine: a scripted model that answers the ladder/tool prompts correctly.
// ---------------------------------------------------------------------------

const PROSE: &str = "A refrigerator moves heat out of the box using a refrigerant that evaporates and condenses.";

struct Stub;

fn last_content(messages: &[ChatMessage]) -> String {
    messages.last().map(|m| m.content.clone()).unwrap_or_default()
}

fn scripted_answer(prompt: &str) -> String {
    if prompt.starts_with("Reply exactly APPLE") {
        return "APPLE".into();
    }
    if prompt.starts_with("Calculate 17 times 23") {
        return "391".into();
    }
    if prompt.starts_with("Reply with exactly this JSON") {
        return r#"{"ok":true,"n":3}"#.into();
    }
    if let Some(rest) = prompt.strip_prefix("Count down from ") {
        if let Some((a, b)) = rest.split_once(" to ") {
            let a: i64 = a.trim().parse().unwrap_or(0);
            let b: i64 = b.split(',').next().unwrap_or("").trim().parse().unwrap_or(0);
            return (b..=a).rev().map(|x| x.to_string()).collect::<Vec<_>>().join(", ");
        }
    }
    if let Some(rest) = prompt.strip_prefix("Count from ") {
        if let Some((a, b)) = rest.split_once(" to ") {
            let a: i64 = a.trim().parse().unwrap_or(0);
            let b: i64 = b.split(',').next().unwrap_or("").trim().parse().unwrap_or(0);
            return (a..=b).map(|x| x.to_string()).collect::<Vec<_>>().join(", ");
        }
    }
    if let Some(rest) = prompt.strip_prefix("List the even numbers from ") {
        if let Some((a, b)) = rest.split_once(" to ") {
            let a: i64 = a.trim().parse().unwrap_or(0);
            let b: i64 = b.split(',').next().unwrap_or("").trim().parse().unwrap_or(0);
            return (a..=b).step_by(2).map(|x| x.to_string()).collect::<Vec<_>>().join(", ");
        }
    }
    const NEEDLE: &str = "The secret vault code is ";
    if let Some(pos) = prompt.find(NEEDLE) {
        let code = &prompt[pos + NEEDLE.len()..];
        let digits: String = code.chars().take_while(|c| c.is_ascii_digit()).collect();
        return digits;
    }
    // Three labelled needles: "The vault code for X is NNNNNN" -> "X: NNNNNN".
    if prompt.contains("The vault code for ") {
        let mut lines = Vec::new();
        let mut rest = prompt;
        while let Some(pos) = rest.find("The vault code for ") {
            let after = &rest[pos + "The vault code for ".len()..];
            let label: String = after.chars().take_while(|c| c.is_alphanumeric()).collect();
            if let Some(n) = after.find(" is ") {
                let code = &after[n + 4..];
                let digits: String = code.chars().take_while(|c| c.is_ascii_digit()).collect();
                if !digits.is_empty() {
                    lines.push(format!("{label}: {digits}"));
                }
            }
            rest = &after[after.find('.').map(|p| p + 1).unwrap_or(after.len())..];
        }
        if !lines.is_empty() {
            return lines.join("\n");
        }
    }
    PROSE.into()
}

impl Engine for Stub {
    fn tokenize(&self, messages: &[ChatMessage], _tools: &[Tool], _thinking: bool) -> usize {
        messages.iter().map(|m| m.content.len()).sum::<usize>() / 4
    }
    fn render_chat(&self, messages: &[ChatMessage], tools: &[Tool], _thinking: bool) -> String {
        if tools.is_empty() {
            last_content(messages)
        } else {
            let names = tools.iter().map(|t| t.function.name.clone()).collect::<Vec<_>>().join(",");
            format!("__TOOLS__:{names}\n{}", last_content(messages))
        }
    }
    fn generate(
        &self,
        prompt: &str,
        _params: &GenerateParams,
        on_delta: &mut dyn FnMut(&str),
    ) -> Result<GenerateOutcome, String> {
        let text = if let Some(rest) = prompt.strip_prefix("__TOOLS__:") {
            // Emit one <tool_call><function=NAME></function></tool_call> per tool.
            let names: Vec<&str> = rest.lines().next().unwrap_or("").split(',').filter(|s| !s.is_empty()).collect();
            let mut out = String::new();
            for n in names {
                out.push_str("\u{3c}tool_call\u{3e}");
                out.push_str("\u{3c}function\u{3d}");
                out.push_str(n);
                out.push_str("\u{3e}");
                out.push_str("\u{3c}\u{2f}function\u{3e}");
                out.push_str("\u{3c}\u{2f}tool_call\u{3e}");
            }
            out
        } else {
            scripted_answer(prompt)
        };
        on_delta(&text);
        Ok(GenerateOutcome { text: text.clone(), finish_reason: "stop".into(), completion_tokens: text.len() / 4 + 1 })
    }
}

/// A slow stub: 10 tokens at 50 ms each, so real streaming is distinguishable
/// from the old whole-response buffering (TTFT must be the first token, not the
/// end of the response).
struct SlowStub;

impl Engine for SlowStub {
    fn tokenize(&self, messages: &[ChatMessage], _tools: &[Tool], _thinking: bool) -> usize {
        messages.iter().map(|m| m.content.len()).sum::<usize>() / 4
    }
    fn render_chat(&self, messages: &[ChatMessage], _tools: &[Tool], _thinking: bool) -> String {
        last_content(messages)
    }
    fn generate(
        &self,
        _prompt: &str,
        _params: &GenerateParams,
        on_delta: &mut dyn FnMut(&str),
    ) -> Result<GenerateOutcome, String> {
        const N: usize = 10;
        let mut text = String::new();
        for i in 0..N {
            std::thread::sleep(std::time::Duration::from_millis(50));
            let tok = format!("tok{i} ");
            text.push_str(&tok);
            on_delta(&tok);
        }
        Ok(GenerateOutcome { text, finish_reason: "stop".into(), completion_tokens: N })
    }
}

/// A stub that echoes the rendered prompt back as the completion (for the D6
/// end-to-end test: the decoded request content must reach the model unchanged).
struct EchoStub;

impl Engine for EchoStub {
    fn tokenize(&self, messages: &[ChatMessage], _tools: &[Tool], _thinking: bool) -> usize {
        messages.iter().map(|m| m.content.len()).sum::<usize>() / 4
    }
    fn render_chat(&self, messages: &[ChatMessage], _tools: &[Tool], _thinking: bool) -> String {
        last_content(messages)
    }
    fn generate(
        &self,
        prompt: &str,
        _params: &GenerateParams,
        on_delta: &mut dyn FnMut(&str),
    ) -> Result<GenerateOutcome, String> {
        on_delta(prompt);
        Ok(GenerateOutcome { text: prompt.to_string(), finish_reason: "stop".into(), completion_tokens: prompt.len() / 4 + 1 })
    }
}

/// A stub that emits one MiMo tool call token by token (so the streaming
/// holdback is exercised across delta boundaries).
struct TokenToolStub;

impl Engine for TokenToolStub {
    fn tokenize(&self, messages: &[ChatMessage], _tools: &[Tool], _thinking: bool) -> usize {
        messages.iter().map(|m| m.content.len()).sum::<usize>() / 4
    }
    fn render_chat(&self, messages: &[ChatMessage], tools: &[Tool], _thinking: bool) -> String {
        if tools.is_empty() {
            last_content(messages)
        } else {
            format!("__TOOLS__:tool0\n{}", last_content(messages))
        }
    }
    fn generate(
        &self,
        prompt: &str,
        _params: &GenerateParams,
        on_delta: &mut dyn FnMut(&str),
    ) -> Result<GenerateOutcome, String> {
        let text = if let Some(_) = prompt.strip_prefix("__TOOLS__:") {
            // The full markup `<tool_call><function=tool0></function></tool_call>`,
            // emitted in pieces that split the open tag across two deltas.
            let tokens: [&str; 5] = [
                "\u{3c}tool",                                 // <tool
                "_call\u{3e}\u{3c}function\u{3d}tool0\u{3e}", // _call><function=tool0>
                "\u{3c}\u{2f}function\u{3e}",                 // </function>
                "\u{3c}\u{2f}tool_call\u{3e}",                // </tool_call>
                "",
            ];
            let mut out = String::new();
            for t in tokens {
                on_delta(t);
                out.push_str(t);
            }
            out
        } else {
            let t = scripted_answer(prompt);
            on_delta(&t);
            t
        };
        Ok(GenerateOutcome { text: text.clone(), finish_reason: "stop".into(), completion_tokens: 6 })
    }
}

/// A stub whose content deltas contain multi-byte characters ending at delta
/// boundaries (a curly quote, CJK, a 4-byte emoji) and a delta ending mid-way
/// through the tool-call open tag after a multi-byte character, so the D4
/// holdback must step over char boundaries, not bytes.
struct MultiByteToolStub;

impl Engine for MultiByteToolStub {
    fn tokenize(&self, messages: &[ChatMessage], _tools: &[Tool], _thinking: bool) -> usize {
        messages.iter().map(|m| m.content.len()).sum::<usize>() / 4
    }
    fn render_chat(&self, messages: &[ChatMessage], tools: &[Tool], _thinking: bool) -> String {
        if tools.is_empty() {
            last_content(messages)
        } else {
            format!("__TOOLS__:tool0\n{}", last_content(messages))
        }
    }
    fn generate(
        &self,
        prompt: &str,
        _params: &GenerateParams,
        on_delta: &mut dyn FnMut(&str),
    ) -> Result<GenerateOutcome, String> {
        let text = if prompt.starts_with("__TOOLS__:") {
            // Deltas: ASCII, then a lone curly quote (3 bytes), CJK, a 4-byte
            // emoji, then " <CJK><too" (a multi-byte char immediately before the
            // open-tag prefix), then the rest of the tag + call.
            let deltas: [&str; 6] = [
                "Hello ",
                "\u{2019}",           // ’ — a lone 3-byte char ends the delta
                "\u{4e16}\u{754c}",   // 世界 — CJK
                "\u{1f600}",          // 😀 — a 4-byte emoji
                " \u{6d4b}\u{8bd5}\u{3c}too", // " 测试<too" — multibyte then open-tag prefix
                "l_call\u{3e}\u{3c}function\u{3d}tool0\u{3e}\u{3c}\u{2f}function\u{3e}\u{3c}\u{2f}tool_call\u{3e}",
            ];
            let mut out = String::new();
            for d in deltas {
                on_delta(d);
                out.push_str(d);
            }
            out
        } else {
            let t = scripted_answer(prompt);
            on_delta(&t);
            t
        };
        Ok(GenerateOutcome { text: text.clone(), finish_reason: "stop".into(), completion_tokens: 8 })
    }
}

// ---------------------------------------------------------------------------
// Loopback server helper.
// ---------------------------------------------------------------------------

struct Server {
    base: String,
    _handle: std::thread::JoinHandle<()>,
}

fn start() -> Server {
    start_engine(Stub)
}

fn start_engine<E: Engine + Send + Sync + 'static>(engine: E) -> Server {
    // The stub engines write MiMo markup, so they are served with the MiMo dialect.
    start_engine_with(engine, Arc::new(MimoDialect))
}

fn start_engine_with<E: Engine + Send + Sync + 'static>(engine: E, dialect: Arc<dyn Dialect>) -> Server {
    let engine = std::sync::Arc::new(engine);
    let listener = TcpListener::bind("127.0.0.1:0").expect("bind");
    let port = listener.local_addr().expect("addr").port();
    let handle = std::thread::spawn(move || {
        let _ = glm53f_api::http::serve_listener(listener, move |req| {
            let path = req.path.split('?').next().unwrap_or("").to_string();
            match (req.method.as_str(), path.as_str()) {
                ("GET", "/v1/models") => glm53f_api::models::handle(),
                ("POST", "/v1/chat/completions") => {
                    let text = String::from_utf8_lossy(&req.body);
                    match glm53f_api::json::parse(&text) {
                        Ok(body) => match glm53f_api::chat::handle(engine.clone(), dialect.clone(), &body) {
                            Ok(resp) => resp,
                            Err(e) => glm53f_api::http::json_response(e.status, &glm53f_api::json::serialize(&e.body())),
                        },
                        Err(e) => {
                            let err = glm53f_api::ApiError::bad_request(format!("invalid JSON: {e}"));
                            glm53f_api::http::json_response(400, &glm53f_api::json::serialize(&err.body()))
                        }
                    }
                }
                _ => {
                    let err = glm53f_api::ApiError::not_found("not found");
                    glm53f_api::http::json_response(404, &glm53f_api::json::serialize(&err.body()))
                }
            }
        });
    });
    Server { base: format!("http://127.0.0.1:{port}"), _handle: handle }
}

fn http_post(url: &str, body: &str) -> (u16, String) {
    let mut parts = url.trim_start_matches("http://").splitn(2, '/');
    let hostport = parts.next().unwrap();
    let path = format!("/{}", parts.next().unwrap_or(""));
    let stream = std::net::TcpStream::connect(hostport).expect("connect");
    let mut stream = stream;
    use std::io::Write;
    write!(stream, "POST {path} HTTP/1.1\r\nHost: {hostport}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}", body.len(), body).unwrap();
    stream.flush().unwrap();
    let mut resp = String::new();
    let mut stream = stream;
    stream.read_to_string(&mut resp).unwrap();
    let status: u16 = resp.lines().next().and_then(|l| l.split(' ').nth(1)).and_then(|s| s.parse().ok()).unwrap_or(0);
    let body = resp.split("\r\n\r\n").nth(1).unwrap_or("").to_string();
    (status, body)
}

fn http_get(url: &str) -> (u16, String) {
    let mut parts = url.trim_start_matches("http://").splitn(2, '/');
    let hostport = parts.next().unwrap();
    let path = format!("/{}", parts.next().unwrap_or(""));
    let mut stream = std::net::TcpStream::connect(hostport).expect("connect");
    use std::io::Write;
    write!(stream, "GET {path} HTTP/1.1\r\nHost: {hostport}\r\nConnection: close\r\n\r\n").unwrap();
    stream.flush().unwrap();
    let mut resp = String::new();
    stream.read_to_string(&mut resp).unwrap();
    let status: u16 = resp.lines().next().and_then(|l| l.split(' ').nth(1)).and_then(|s| s.parse().ok()).unwrap_or(0);
    let body = resp.split("\r\n\r\n").nth(1).unwrap_or("").to_string();
    (status, body)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[test]
fn models_lists_the_model_id() {
    let srv = start();
    let (status, body) = http_get(&format!("{}/v1/models", srv.base));
    assert_eq!(status, 200);
    let v = glm53f_api::json::parse(&body).unwrap();
    let ids: Vec<&str> = v.get("data").and_then(|d| d.as_array()).unwrap().iter()
        .filter_map(|m| m.get("id").and_then(|i| i.as_str())).collect();
    assert_eq!(ids, vec![MODEL_ID]);
}

#[test]
fn chat_non_stream_returns_content_and_usage() {
    let srv = start();
    let body = r#"{"model":"glm-5.3-flash","messages":[{"role":"user","content":"Calculate 17 times 23. Reply with only the integer answer."}],"temperature":0}"#;
    let (status, resp) = http_post(&format!("{}/v1/chat/completions", srv.base), body);
    assert_eq!(status, 200, "{resp}");
    let v = glm53f_api::json::parse(&resp).unwrap();
    let c0 = &v.get("choices").and_then(|c| c.as_array()).unwrap()[0];
    assert_eq!(c0.get("message").and_then(|m| m.get("content")).and_then(|c| c.as_str()), Some("391"));
    assert_eq!(c0.get("finish_reason").and_then(|f| f.as_str()), Some("stop"));
    assert!(v.get("usage").is_some());
}

#[test]
fn chat_stream_has_content_deltas_and_usage() {
    let srv = start();
    let body = r#"{"model":"glm-5.3-flash","messages":[{"role":"user","content":"hi"}],"temperature":0,"stream":true,"stream_options":{"include_usage":true}}"#;
    let (status, resp) = http_post(&format!("{}/v1/chat/completions", srv.base), body);
    assert_eq!(status, 200);
    assert!(resp.contains("data: [DONE]"), "{resp}");
    assert!(resp.contains("\"content\""), "{resp}");
    assert!(resp.contains("\"usage\""), "{resp}");
}

/// The streaming defect the A8 ladder exposed: with a slow engine, TTFT must be
/// the first token (~50 ms), not the whole response (~500 ms). The old code
/// buffered the entire SSE body into one response, so TTFT == total.
#[test]
fn streaming_ttft_is_first_token_not_total() {
    let py = python();
    if Command::new(&py).arg("--version").output().is_err() {
        eprintln!("SKIP: python3 not available");
        return;
    }
    let srv = start_engine(SlowStub);
    // A streaming client in the standard library: the time to the first content delta, and to
    // the end of the stream.
    let driver = r#"
import json, sys, time, urllib.request
body = json.dumps({"model": "glm-5.3-flash", "stream": True, "max_tokens": 20,
                   "messages": [{"role": "user", "content": sys.argv[2]}]}).encode()
req = urllib.request.Request(sys.argv[1] + "/chat/completions", data=body, headers={"Content-Type": "application/json"})
t0, first = time.time(), None
with urllib.request.urlopen(req) as r:
    for raw in r:
        line = raw.decode().strip()
        if not line.startswith("data:") or line == "data: [DONE]":
            continue
        for c in json.loads(line[5:]).get("choices", []):
            if first is None and (c.get("delta") or {}).get("content"):
                first = time.time() - t0
print(json.dumps({"ttft_s": first, "total_s": time.time() - t0}))
"#;
    let driver_file = std::env::temp_dir().join("glm53f-api-stream-slow-driver.py");
    std::fs::write(&driver_file, driver).unwrap();
    let out = Command::new(&py)
        .arg(&driver_file)
        .arg(&format!("{}/v1", srv.base))
        .arg("hi")
        .output()
        .expect("run the streaming driver");
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(out.status.success(), "{stdout}\n{}", String::from_utf8_lossy(&out.stderr));
    let r = serde_json_ish::parse(&stdout);
    let ttft = r["ttft_s"];
    let total = r["total_s"];
    assert!(ttft < 0.2, "TTFT {ttft} should be < 0.2 s (first token at ~50 ms); got {stdout}");
    assert!(total >= 0.5, "total {total} should be >= 0.5 s (10 tokens at 50 ms); got {stdout}");
}

#[test]
fn media_content_part_returns_400() {
    let srv = start();
    let body = r#"{"model":"glm-5.3-flash","messages":[{"role":"user","content":[{"type":"text","text":"hi"},{"type":"image_url","image_url":{"url":"http://x"}}]}]}"#;
    let (status, resp) = http_post(&format!("{}/v1/chat/completions", srv.base), body);
    assert_eq!(status, 400, "{resp}");
    assert!(resp.contains("error"), "{resp}");
}

#[test]
fn tool_calls_round_trip_non_stream_and_stream() {
    let srv = start();
    // Non-stream.
    let body = r#"{"model":"glm-5.3-flash","messages":[{"role":"user","content":"use the tools"}],"tools":[{"type":"function","function":{"name":"tool0","parameters":{}}},{"type":"function","function":{"name":"tool1","parameters":{}}}],"stream":false}"#;
    let (status, resp) = http_post(&format!("{}/v1/chat/completions", srv.base), body);
    assert_eq!(status, 200, "{resp}");
    let v = glm53f_api::json::parse(&resp).unwrap();
    let c0 = &v.get("choices").and_then(|c| c.as_array()).unwrap()[0];
    let calls = c0.get("message").and_then(|m| m.get("tool_calls")).and_then(|t| t.as_array()).unwrap();
    assert_eq!(calls.len(), 2);
    assert_eq!(c0.get("finish_reason").and_then(|f| f.as_str()), Some("tool_calls"));

    // Stream.
    let body = r#"{"model":"glm-5.3-flash","messages":[{"role":"user","content":"use the tools"}],"tools":[{"type":"function","function":{"name":"tool0","parameters":{}}}],"stream":true,"stream_options":{"include_usage":true}}"#;
    let (status, resp) = http_post(&format!("{}/v1/chat/completions", srv.base), body);
    assert_eq!(status, 200, "{resp}");
    assert!(resp.contains("\"tool_calls\""), "{resp}");
    assert!(resp.contains("data: [DONE]"), "{resp}");
}

/// Extract the `data:` payloads from a (possibly chunked) SSE response body.
fn sse_data_payloads(raw: &str) -> Vec<String> {
    raw.lines()
        .map(|l| l.trim())
        .filter_map(|l| l.strip_prefix("data: "))
        .map(|s| s.to_string())
        .collect()
}

/// The D3 defect: streaming must not leak the raw tool-call markup into content
/// deltas. A token-by-token tool call must come back as tool_calls deltas only.
#[test]
fn streaming_tool_call_does_not_leak_markup() {
    let srv = start_engine(TokenToolStub);
    let body = r#"{"model":"glm-5.3-flash","messages":[{"role":"user","content":"use the tools"}],"tools":[{"type":"function","function":{"name":"tool0","parameters":{}}}],"stream":true}"#;
    let (status, resp) = http_post(&format!("{}/v1/chat/completions", srv.base), body);
    assert_eq!(status, 200, "{resp}");

    let mut name: Option<String> = None;
    let mut args = String::new();
    let mut finish: Option<String> = None;
    for payload in sse_data_payloads(&resp) {
        if payload == "[DONE]" {
            continue;
        }
        let v = glm53f_api::json::parse(&payload).unwrap_or_else(|e| panic!("bad event {payload}: {e}"));
        let Some(choices) = v.get("choices").and_then(|c| c.as_array()) else { continue };
        for ch in choices {
            let Some(delta) = ch.get("delta") else { continue };
            // No content delta may carry any part of the markup.
            if let Some(content) = delta.get("content").and_then(|c| c.as_str()) {
                assert!(
                    !content.contains('\u{3c}') && !content.contains('\u{3e}'),
                    "content delta leaked markup: {content}"
                );
            }
            // Collect tool_calls deltas (name from the header, arguments from args).
            if let Some(tcs) = delta.get("tool_calls").and_then(|t| t.as_array()) {
                for tc in tcs {
                    if let Some(f) = tc.get("function") {
                        if let Some(n) = f.get("name").and_then(|x| x.as_str()) {
                            name = Some(n.to_string());
                        }
                        if let Some(a) = f.get("arguments").and_then(|x| x.as_str()) {
                            args.push_str(a);
                        }
                    }
                }
            }
            if let Some(fr) = ch.get("finish_reason").and_then(|f| f.as_str()) {
                finish = Some(fr.to_string());
            }
        }
    }
    assert_eq!(name.as_deref(), Some("tool0"), "{resp}");
    assert_eq!(args, "{}", "tool_calls deltas must reassemble to the parsed call: {resp}");
    assert_eq!(finish.as_deref(), Some("tool_calls"), "{resp}");
}

/// The D4 defect: the holdback sliced `hold[hold.len() - k..]` by byte, so any
/// streamed text with a multi-byte character at the tail panics (curly quote,
/// CJK, emoji). The stream must not panic, content must equal the concatenation,
/// and the tool path must still work.
#[test]
fn streaming_multibyte_text_does_not_panic_in_holdback() {
    let srv = start_engine(MultiByteToolStub);
    let body = r#"{"model":"glm-5.3-flash","messages":[{"role":"user","content":"use the tools"}],"tools":[{"type":"function","function":{"name":"tool0","parameters":{}}}],"stream":true}"#;
    let (status, resp) = http_post(&format!("{}/v1/chat/completions", srv.base), body);
    assert_eq!(status, 200, "panic in the holdback should not drop the stream: {resp}");

    // Content is exactly the concatenation of the pre-tool deltas, every fragment verbatim and in
    // order (the raw stream body is UTF-8-decoded by read_to_string). The space that ends the
    // first fragment is held until the next text arrives, and goes out with it.
    let content: String = sse_events(&resp)
        .iter()
        .flat_map(|ev| ev.get("choices").and_then(|c| c.as_array()).unwrap_or(&[]).iter())
        .filter_map(|ch| ch.get("delta").and_then(|d| d.get("content")).and_then(|c| c.as_str()))
        .collect();
    assert_eq!(content, "Hello \u{2019}\u{4e16}\u{754c}\u{1f600} \u{6d4b}\u{8bd5}", "{resp}");

    // The tool path still works: name + finish_reason (ASCII) come back parsed.
    let mut name: Option<String> = None;
    let mut finish: Option<String> = None;
    for payload in sse_data_payloads(&resp) {
        if payload == "[DONE]" {
            continue;
        }
        let v = glm53f_api::json::parse(&payload).unwrap_or_else(|e| panic!("bad event {payload}: {e}"));
        let Some(choices) = v.get("choices").and_then(|c| c.as_array()) else { continue };
        for ch in choices {
            if let Some(delta) = ch.get("delta") {
                if let Some(tcs) = delta.get("tool_calls").and_then(|t| t.as_array()) {
                    for tc in tcs {
                        if let Some(n) = tc.get("function").and_then(|f| f.get("name")).and_then(|x| x.as_str()) {
                            name = Some(n.to_string());
                        }
                    }
                }
            }
            if let Some(fr) = ch.get("finish_reason").and_then(|f| f.as_str()) {
                finish = Some(fr.to_string());
            }
        }
    }
    assert_eq!(name.as_deref(), Some("tool0"), "tool path must still work: {resp}");
    assert_eq!(finish.as_deref(), Some("tool_calls"), "{resp}");
}

/// D6 end-to-end: a request whose content is raw non-ASCII (CJK, accents, emoji,
/// curly quotes, unescaped) must reach the engine unchanged and come back
/// verbatim — no Latin-1 mojibake, no 400.
#[test]
fn raw_utf8_request_content_round_trips_unchanged() {
    let srv = start_engine(EchoStub);
    let content = "東京 café \u{1f600} \u{2019}curl\u{2019}";
    let body = format!(
        r#"{{"model":"glm-5.3-flash","messages":[{{"role":"user","content":"{content}"}}],"stream":false}}"#
    );
    let (status, resp) = http_post(&format!("{}/v1/chat/completions", srv.base), &body);
    assert_eq!(status, 200, "{resp}");
    // The echoed completion must contain the exact decoded content.
    assert!(resp.contains(content), "decoded content must echo unchanged: {resp}");

    // An escaped surrogate pair for an emoji must also decode and echo (the
    // Python json.dumps default form), not 400.
    let body = r#"{"model":"glm-5.3-flash","messages":[{"role":"user","content":"hi \uD83D\uDE00"}]}"#;
    let (status, resp) = http_post(&format!("{}/v1/chat/completions", srv.base), body);
    assert_eq!(status, 200, "escaped surrogate pair must not 400: {resp}");
    assert!(resp.contains("\u{1f600}"), "surrogate pair must decode to the emoji: {resp}");

    // A lone high surrogate must be a 400 (never a silent decode).
    let body = r#"{"model":"glm-5.3-flash","messages":[{"role":"user","content":"hi \uD83D"}]}"#;
    let (status, resp) = http_post(&format!("{}/v1/chat/completions", srv.base), body);
    assert_eq!(status, 400, "lone high surrogate must 400: {resp}");
}

#[test]
fn nameless_tool_call_is_rejected_400() {
    // A request whose stub output would be a nameless call is not reachable via
    // the scripted stub, so exercise the parser directly for the T24/T29 rule.
    let r = glm53f_api::dialect::mimo::parse("\u{3c}tool_call\u{3e}\u{3c}parameter\u{3d}url\u{3e}x\u{3c}\u{2f}parameter\u{3e}\u{3c}\u{2f}tool_call\u{3e}", &[], 6);
    assert_eq!(r.error.as_deref(), Some("nameless tool call"));
    assert!(r.calls.is_empty());
}

/// Order-insensitive JSON equality (object key order is not significant).
fn json_eq(a: &glm53f_api::json::Json, b: &glm53f_api::json::Json) -> bool {
    use glm53f_api::json::Json;
    match (a, b) {
        (Json::Object(pa), Json::Object(pb)) => pa.len() == pb.len()
            && pa.iter().all(|(k, va)| pb.iter().find(|(k2, _)| k2 == k).map(|(_, vb)| json_eq(va, vb)).unwrap_or(false)),
        (Json::Array(aa), Json::Array(ab)) => aa.len() == ab.len() && aa.iter().zip(ab).all(|(x, y)| json_eq(x, y)),
        (x, y) => x == y,
    }
}

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).parent().unwrap().parent().unwrap().to_path_buf()
}

fn python() -> String {
    std::env::var("GLM53F_PYTHON").unwrap_or_else(|_| "python3".into())
}

/// The harness directory: `GLM53F_HARNESS`, else `harness/` at the repository root.
fn harness() -> PathBuf {
    std::env::var_os("GLM53F_HARNESS").map(PathBuf::from).unwrap_or_else(|| repo_root().join("harness"))
}

/// A harness file, or `None` (with a SKIP line) while the harness does not carry it.
fn harness_file(rel: &str) -> Option<PathBuf> {
    let p = harness().join(rel);
    if p.is_file() {
        Some(p)
    } else {
        eprintln!("SKIP: {} not present (GLM53F_HARNESS names a harness directory)", p.display());
        None
    }
}

// A tiny JSON reader for the streaming driver's output (avoids a serde dep).
mod serde_json_ish {
    pub type Map = std::collections::BTreeMap<String, f64>;
    pub fn parse(text: &str) -> Map {
        let mut m = Map::new();
        for pair in text.trim().trim_start_matches('{').trim_end_matches('}').split(',') {
            let pair = pair.trim();
            if let Some((k, v)) = pair.split_once(':') {
                let k = k.trim().trim_matches('"').to_string();
                if let Ok(n) = v.trim().parse::<f64>() {
                    m.insert(k, n);
                }
            }
        }
        m
    }
}

#[test]
fn t29_parser_goldens_pass() {
    // Load the committed T27/T29 parser corpus (the MiMo dialect's goldens, kept
    // with this crate) and run the parser against every case, checking the parsed
    // calls and the must_report losses.
    let goldens = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/mimo/t29_parser_goldens.json");
    let text = std::fs::read_to_string(goldens).expect("read goldens");
    let doc = glm53f_api::json::parse(&text).expect("parse goldens");
    let cases = doc.get("cases").and_then(|c| c.as_array()).expect("cases");
    assert!(!cases.is_empty());
    for case in cases {
        let id = case.get("id").and_then(|i| i.as_str()).unwrap_or("?");
        let output = case.get("output").and_then(|o| o.as_str()).expect("output");
        let expected = case.get("expected").expect("expected");
        let r = glm53f_api::dialect::mimo::parse(output, &[], 0);

        if let Some(e) = expected.get("error").and_then(|e| e.as_str()) {
            assert_eq!(r.error.as_deref(), Some(e), "case {id}");
        } else {
            assert_eq!(r.error, None, "case {id}");
        }
        let exp_calls = expected.get("calls").and_then(|c| c.as_array()).map(|a| a.to_vec()).unwrap_or_default();
        assert_eq!(r.calls.len(), exp_calls.len(), "case {id}");
        for (i, ec) in exp_calls.iter().enumerate() {
            let name = ec.get("name").and_then(|n| n.as_str()).unwrap();
            assert_eq!(r.calls[i].name, name, "case {id}");
            assert!(json_eq(&r.calls[i].arguments, ec.get("arguments").unwrap()), "case {id}");
        }
        if let Some(reps) = expected.get("must_report").and_then(|m| m.as_array()) {
            for rep in reps {
                let rep = rep.as_str().unwrap();
                assert!(r.reports.iter().any(|x| x.contains(rep)), "case {id} missing report {rep}; got {:?}", r.reports);
            }
        }
    }
}

#[test]
fn l5_ladder_cell_passes_against_the_stub() {
    let py = python();
    if Command::new(&py).arg("--version").output().is_err() {
        eprintln!("SKIP: python3 not available");
        return;
    }
    let Some(ladder) = harness_file("l5_ladder.py") else { return };
    let srv = start();
    let out_dir = std::env::temp_dir().join(format!("glm53f-api-l5-{}", std::process::id()));
    let out = Command::new(&py)
        .arg(&ladder)
        .arg("--base").arg(format!("{}/v1", srv.base))
        .arg("--out").arg(&out_dir)
        .arg("--cell").arg("ladder")
        .arg("--needle-targets").arg("400,800")
        .output().expect("run ladder");
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(out.status.success(), "ladder FAILED:\n{stdout}\n{}", String::from_utf8_lossy(&out.stderr));
    assert!(stdout.contains("RESULT: PASS L5 ladder"), "{stdout}");
}

/// A thinking completion, `<think>Let me think.</think>The answer is 42 <`,
/// emitted with both think tags split across deltas and a bare `<` at the end.
struct ThinkStub;

const THINK_TOKENS: [&str; 6] = [
    "\u{3c}thi",                  // <thi
    "nk\u{3e}Let me ",            // nk>Let me
    "think.\u{3c}/th",            // think.</th
    "ink\u{3e}The ans",           // ink>The ans
    "wer is 42 \u{3c}",           // wer is 42 <
    "",
];

impl Engine for ThinkStub {
    fn tokenize(&self, messages: &[ChatMessage], _tools: &[Tool], _thinking: bool) -> usize {
        messages.iter().map(|m| m.content.len()).sum::<usize>() / 4
    }
    fn render_chat(&self, messages: &[ChatMessage], _tools: &[Tool], _thinking: bool) -> String {
        last_content(messages)
    }
    fn generate(
        &self,
        _prompt: &str,
        _params: &GenerateParams,
        on_delta: &mut dyn FnMut(&str),
    ) -> Result<GenerateOutcome, String> {
        let mut out = String::new();
        for t in THINK_TOKENS {
            on_delta(t);
            out.push_str(t);
        }
        Ok(GenerateOutcome { text: out, finish_reason: "stop".into(), completion_tokens: 6 })
    }
}

/// Streamed thinking (found wiring the engine into an agent client, 2026-09-26): the think
/// block used to stream as content, markup and all, and then again as a
/// reasoning delta. The stream must split exactly as the non-stream parse does:
/// reasoning once, content without markup, and a trailing `<` kept.
#[test]
fn streaming_think_block_is_reasoning_only_and_matches_non_stream() {
    let srv = start_engine(ThinkStub);
    let body = r#"{"model":"glm-5.3-flash","messages":[{"role":"user","content":"think"}],"chat_template_kwargs":{"enable_thinking":true},"stream":true}"#;
    let (status, resp) = http_post(&format!("{}/v1/chat/completions", srv.base), body);
    assert_eq!(status, 200, "{resp}");
    let (mut content, mut reasoning) = (String::new(), String::new());
    for payload in sse_data_payloads(&resp) {
        if payload == "[DONE]" {
            continue;
        }
        let v = glm53f_api::json::parse(&payload).unwrap_or_else(|e| panic!("bad event {payload}: {e}"));
        let Some(choices) = v.get("choices").and_then(|c| c.as_array()) else { continue };
        for ch in choices {
            let Some(delta) = ch.get("delta") else { continue };
            if let Some(c) = delta.get("content").and_then(|c| c.as_str()) {
                assert!(!c.contains("think"), "content delta leaked think markup: {c:?}");
                content.push_str(c);
            }
            assert!(delta.get("reasoning").is_none(), "reasoning under a second name: {payload}");
            if let Some(r) = delta.get("reasoning_content").and_then(|r| r.as_str()) {
                reasoning.push_str(r);
            }
        }
    }
    assert_eq!(reasoning, "Let me think.", "reasoning streamed once, without markup: {resp}");
    assert_eq!(content, "The answer is 42 \u{3c}", "content without markup, trailing tag prefix kept: {resp}");

    let body = r#"{"model":"glm-5.3-flash","messages":[{"role":"user","content":"think"}],"chat_template_kwargs":{"enable_thinking":true}}"#;
    let (status, resp) = http_post(&format!("{}/v1/chat/completions", srv.base), body);
    assert_eq!(status, 200, "{resp}");
    let v = glm53f_api::json::parse(&resp).unwrap();
    let msg = v.get("choices").and_then(|c| c.as_array()).and_then(|a| a.first()).and_then(|c| c.get("message")).unwrap();
    assert_eq!(msg.get("content").and_then(|c| c.as_str()), Some(content.as_str()), "stream content = non-stream: {resp}");
    assert_eq!(msg.get("reasoning_content").and_then(|c| c.as_str()), Some(reasoning.as_str()), "{resp}");
}

/// An 8x8 red PNG (64x64 after MiMo's smart_resize: 4 image tokens).
const RED8_PNG: &str = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAgAAAAICAIAAABLbSncAAAAFElEQVR4nGM8ISfHgA0wYRUdtBIA0MoBFD5jqJkAAAAASUVORK5CYII=";

/// A vision engine stub: replies `<images passed>|<markers in the prompt>|<omitted notes>`, and
/// the whole rendered text after a `#`. Its decoder stands in for a vision tower's: it takes any
/// PNG data URL to be [`RED8_PNG`] (8x8 red, 4 tokens) without decoding it; the real decoding
/// belongs to the image crate and its own tests.
struct VisionStub;

impl Engine for VisionStub {
    fn tokenize(&self, messages: &[ChatMessage], _tools: &[Tool], _thinking: bool) -> usize {
        messages.iter().map(|m| m.content.len()).sum::<usize>() / 4
    }
    fn render_chat(&self, messages: &[ChatMessage], _tools: &[Tool], _thinking: bool) -> String {
        messages.iter().map(|m| m.content.clone()).collect::<Vec<_>>().join("\n")
    }
    fn vision(&self) -> bool {
        true
    }
    fn decode_image(&self, data_url: &str) -> Result<glm53f_api::engine::ImageInput, String> {
        if !data_url.starts_with("data:image/png;base64,") {
            return Err("the stub decodes PNG data URLs only".into());
        }
        Ok(glm53f_api::engine::ImageInput { hash: 0, tokens: 4, width: 8, height: 8, rgb: [255, 0, 0].repeat(64) })
    }
    fn generate(
        &self,
        prompt: &str,
        params: &GenerateParams,
        on_delta: &mut dyn FnMut(&str),
    ) -> Result<GenerateOutcome, String> {
        let markers = prompt.chars().filter(|&c| c == glm53f_api::engine::IMAGE_OPEN).count();
        let notes = prompt.matches("[image omitted").count();
        let text = format!("{}|{markers}|{notes}#{prompt}", params.images.len());
        on_delta(&text);
        Ok(GenerateOutcome { text, finish_reason: "stop".into(), completion_tokens: 1 })
    }
}

fn vision_reply(parts: &str) -> (u16, String) {
    let srv = start_engine(VisionStub);
    let body = format!(r#"{{"model":"glm-5.3-flash","messages":[{{"role":"user","content":[{parts}]}}]}}"#);
    let (status, resp) = http_post(&format!("{}/v1/chat/completions", srv.base), &body);
    if status != 200 {
        return (status, resp);
    }
    let v = glm53f_api::json::parse(&resp).unwrap();
    let msg = v.get("choices").and_then(|c| c.as_array()).and_then(|a| a.first()).and_then(|c| c.get("message")).unwrap();
    (status, msg.get("content").and_then(|c| c.as_str()).unwrap_or("").to_string())
}

/// Perf reset V2: an image part becomes one image marker in the text and one decoded image in the
/// request, in order; the marker carries the image's token count (the stub decoder's 4 for the
/// 8x8 PNG).
#[test]
fn image_part_becomes_a_marker_and_an_image() {
    let img = format!(r#"{{"type":"image_url","image_url":{{"url":"{RED8_PNG}"}}}}"#);
    let (status, reply) = vision_reply(&format!(r#"{{"type":"text","text":"what colour? "}},{img}"#));
    assert_eq!(status, 200, "{reply}");
    let (head, text) = reply.split_once('#').unwrap();
    assert_eq!(head, "1|1|0", "{reply}");
    assert!(text.starts_with("what colour? \u{FDD0}") && text.ends_with(":4\u{FDD1}"), "{text:?}");
}

/// Only the newest MAX_IMAGES images are sent; older ones become a note (ADVISOR-I3 §10.2 item 6).
#[test]
fn only_the_newest_sixteen_images_are_kept() {
    let img = format!(r#"{{"type":"image_url","image_url":{{"url":"{RED8_PNG}"}}}}"#);
    let parts = vec![img; 17].join(",");
    let (status, reply) = vision_reply(&parts);
    assert_eq!(status, 200, "{reply}");
    assert!(reply.starts_with("16|16|1#[image omitted"), "{reply}");
}

/// The marker characters cannot come from a client: they are stripped from text.
#[test]
fn marker_characters_in_text_are_stripped() {
    let (status, reply) = vision_reply(r#"{"type":"text","text":"a﷐b﷑c"}"#);
    assert_eq!(status, 200, "{reply}");
    assert_eq!(reply, "0|0|0#abc");
}

/// A remote image URL is refused (the server does not fetch), and an engine without an encoder
/// refuses image parts.
#[test]
fn remote_images_and_images_without_an_encoder_are_400() {
    let (status, resp) = vision_reply(r#"{"type":"image_url","image_url":{"url":"https://example.com/a.png"}}"#);
    assert_eq!(status, 400, "{resp}");
    assert!(resp.contains("data URL"), "{resp}");
    let srv = start();
    let body = format!(r#"{{"model":"glm-5.3-flash","messages":[{{"role":"user","content":[{{"type":"image_url","image_url":{{"url":"{RED8_PNG}"}}}}]}}]}}"#);
    let (status, resp) = http_post(&format!("{}/v1/chat/completions", srv.base), &body);
    assert_eq!(status, 400, "{resp}");
    assert!(resp.contains("image encoder"), "{resp}");
}

/// Perf reset V3: an engine stub that replies with the sampling parameters it was handed.
struct ParamsStub;

impl Engine for ParamsStub {
    fn tokenize(&self, messages: &[ChatMessage], _tools: &[Tool], _thinking: bool) -> usize {
        messages.iter().map(|m| m.content.len()).sum::<usize>() / 4
    }
    fn render_chat(&self, messages: &[ChatMessage], _tools: &[Tool], _thinking: bool) -> String {
        last_content(messages)
    }
    fn generate(
        &self,
        _prompt: &str,
        params: &GenerateParams,
        on_delta: &mut dyn FnMut(&str),
    ) -> Result<GenerateOutcome, String> {
        let text = format!("T={} top_p={} top_k={} min_p={} seed={:?}", params.temperature, params.top_p, params.top_k,
            params.min_p, params.seed);
        on_delta(&text);
        Ok(GenerateOutcome { text, finish_reason: "stop".into(), completion_tokens: 1 })
    }
}

fn params_reply(extra: &str) -> (u16, String) {
    let srv = start_engine(ParamsStub);
    let body = format!(r#"{{"model":"glm-5.3-flash","messages":[{{"role":"user","content":"hi"}}]{extra}}}"#);
    let (status, resp) = http_post(&format!("{}/v1/chat/completions", srv.base), &body);
    if status != 200 {
        return (status, resp);
    }
    let v = glm53f_api::json::parse(&resp).unwrap();
    let msg = v.get("choices").and_then(|c| c.as_array()).and_then(|a| a.first()).and_then(|c| c.get("message")).unwrap();
    (status, msg.get("content").and_then(|c| c.as_str()).unwrap_or("").to_string())
}

/// Perf reset V3 (DS41RT v15's contract): the sampling parameters reach the engine; without them a
/// request is greedy with every filter off; top_k 0 and -1 are off; a negative seed is its two's
/// complement.
#[test]
fn sampling_parameters_reach_the_engine() {
    let (status, reply) = params_reply(r#","temperature":0.7,"top_p":0.9,"top_k":40,"min_p":0.05,"seed":-1"#);
    assert_eq!(status, 200, "{reply}");
    assert_eq!(reply, "T=0.7 top_p=0.9 top_k=40 min_p=0.05 seed=Some(18446744073709551615)");
    assert_eq!(params_reply("").1, "T=0 top_p=1 top_k=0 min_p=0 seed=None");
    assert_eq!(params_reply(r#","temperature":null,"top_k":-1,"seed":42"#).1, "T=0 top_p=1 top_k=0 min_p=0 seed=Some(42)");
    assert_eq!(params_reply(r#","temperature":1,"top_k":0"#).1, "T=1 top_p=1 top_k=0 min_p=0 seed=None");
}

/// Out-of-range or malformed sampling parameters are refused, naming the parameter.
#[test]
fn invalid_sampling_parameters_are_400() {
    for (extra, name) in [
        (r#","temperature":2.5"#, "temperature"),
        (r#","temperature":-0.1"#, "temperature"),
        (r#","temperature":"hot""#, "temperature"),
        (r#","top_p":0"#, "top_p"),
        (r#","top_p":1.2"#, "top_p"),
        (r#","min_p":-0.1"#, "min_p"),
        (r#","min_p":1.5"#, "min_p"),
        (r#","top_k":1.5"#, "top_k"),
        (r#","top_k":-2"#, "top_k"),
        (r#","seed":1.5"#, "seed"),
    ] {
        let (status, resp) = params_reply(extra);
        assert_eq!(status, 400, "{extra}: {resp}");
        assert!(resp.contains(name), "{extra}: {resp}");
    }
}

/// Perf reset V3: an engine whose queue is full; its `admit` refuses.
struct BusyStub;

impl Engine for BusyStub {
    fn tokenize(&self, messages: &[ChatMessage], _tools: &[Tool], _thinking: bool) -> usize {
        messages.iter().map(|m| m.content.len()).sum::<usize>() / 4
    }
    fn render_chat(&self, messages: &[ChatMessage], _tools: &[Tool], _thinking: bool) -> String {
        last_content(messages)
    }
    fn admit(&self) -> Result<Option<glm53f_api::engine::QueuePlace>, String> {
        Err("request queue is full or its wait budget expired".into())
    }
    fn generate(&self, _: &str, _: &GenerateParams, _: &mut dyn FnMut(&str)) -> Result<GenerateOutcome, String> {
        panic!("a refused request must not generate");
    }
}

/// A full queue is a 429 with `Retry-After: 1`, streamed or not (the refusal comes before the
/// response starts), so a client or proxy retries instead of failing the request.
#[test]
fn a_full_queue_is_429_with_retry_after() {
    let srv = start_engine(BusyStub);
    let hostport = srv.base.trim_start_matches("http://").to_string();
    for stream in [false, true] {
        let body = format!(r#"{{"model":"glm-5.3-flash","stream":{stream},"messages":[{{"role":"user","content":"hi"}}]}}"#);
        let mut s = std::net::TcpStream::connect(&hostport).expect("connect");
        use std::io::Write;
        write!(s, "POST /v1/chat/completions HTTP/1.1\r\nHost: {hostport}\r\nContent-Type: application/json\r\n\
            Content-Length: {}\r\nConnection: close\r\n\r\n{body}", body.len()).unwrap();
        let mut resp = String::new();
        s.read_to_string(&mut resp).unwrap();
        assert!(resp.starts_with("HTTP/1.1 429 Too Many Requests\r\n"), "{resp}");
        assert!(resp.contains("\r\nRetry-After: 1\r\n"), "{resp}");
        assert!(resp.contains("rate_limit_exceeded") && resp.contains("queue is full"), "{resp}");
    }
}

// ---------------------------------------------------------------------------
// The dialect seam (added in glm53f-afd).
// ---------------------------------------------------------------------------

/// A dialect whose chat template opens the reasoning block in the prompt when thinking is on,
/// as GLM-5.3-Flash's does: the MiMo markup, with the completion read as if it began with the
/// think tag.
struct PromptOpensThink;

impl Dialect for PromptOpensThink {
    fn parse(&self, text: &str, tools: &[Tool], thinking: bool, cap: usize) -> glm53f_api::ParseResult {
        if thinking {
            let open = self.stream_tags().think_open;
            glm53f_api::dialect::mimo::parse(&format!("{open}{text}"), tools, cap)
        } else {
            glm53f_api::dialect::mimo::parse(text, tools, cap)
        }
    }
    fn stream_tags(&self) -> glm53f_api::StreamTags {
        MimoDialect.stream_tags()
    }
    fn reasoning_first(&self, thinking: bool) -> bool {
        thinking
    }
}

/// A completion that starts inside the reasoning block its prompt opened, with the closing tag
/// split across deltas: `Let me think.</think>The answer.`
struct OpenThinkStub;

const OPEN_THINK_TOKENS: [&str; 4] = ["Let me ", "think.\u{3c}/th", "ink\u{3e}The ans", "wer."];

impl Engine for OpenThinkStub {
    fn tokenize(&self, messages: &[ChatMessage], _tools: &[Tool], _thinking: bool) -> usize {
        messages.iter().map(|m| m.content.len()).sum::<usize>() / 4
    }
    fn render_chat(&self, messages: &[ChatMessage], _tools: &[Tool], _thinking: bool) -> String {
        last_content(messages)
    }
    fn generate(
        &self,
        _prompt: &str,
        _params: &GenerateParams,
        on_delta: &mut dyn FnMut(&str),
    ) -> Result<GenerateOutcome, String> {
        let mut out = String::new();
        for t in OPEN_THINK_TOKENS {
            on_delta(t);
            out.push_str(t);
        }
        Ok(GenerateOutcome { text: out, finish_reason: "stop".into(), completion_tokens: 4 })
    }
}

/// `Dialect::reasoning_first`: with thinking on, the stream starts in reasoning, sends the
/// reasoning once and the content without markup, and agrees with the non-stream parse.
#[test]
fn reasoning_opened_by_the_prompt_streams_as_reasoning() {
    let srv = start_engine_with(OpenThinkStub, Arc::new(PromptOpensThink));
    let body = r#"{"model":"glm-5.3-flash","messages":[{"role":"user","content":"think"}],"chat_template_kwargs":{"enable_thinking":true},"stream":true}"#;
    let (status, resp) = http_post(&format!("{}/v1/chat/completions", srv.base), body);
    assert_eq!(status, 200, "{resp}");
    let (mut content, mut reasoning) = (String::new(), String::new());
    for payload in sse_data_payloads(&resp) {
        if payload == "[DONE]" {
            continue;
        }
        let v = glm53f_api::json::parse(&payload).unwrap_or_else(|e| panic!("bad event {payload}: {e}"));
        let Some(choices) = v.get("choices").and_then(|c| c.as_array()) else { continue };
        for ch in choices {
            let Some(delta) = ch.get("delta") else { continue };
            if let Some(c) = delta.get("content").and_then(|c| c.as_str()) {
                content.push_str(c);
            }
            assert!(delta.get("reasoning").is_none(), "reasoning under a second name: {payload}");
            if let Some(r) = delta.get("reasoning_content").and_then(|r| r.as_str()) {
                reasoning.push_str(r);
            }
        }
    }
    assert_eq!(reasoning, "Let me think.", "reasoning streamed once, without markup: {resp}");
    assert_eq!(content, "The answer.", "content without markup: {resp}");

    let body = r#"{"model":"glm-5.3-flash","messages":[{"role":"user","content":"think"}],"chat_template_kwargs":{"enable_thinking":true}}"#;
    let (status, resp) = http_post(&format!("{}/v1/chat/completions", srv.base), body);
    assert_eq!(status, 200, "{resp}");
    let v = glm53f_api::json::parse(&resp).unwrap();
    let msg = v.get("choices").and_then(|c| c.as_array()).and_then(|a| a.first()).and_then(|c| c.get("message")).unwrap();
    assert_eq!(msg.get("content").and_then(|c| c.as_str()), Some(content.as_str()), "stream content = non-stream: {resp}");
    assert_eq!(msg.get("reasoning_content").and_then(|c| c.as_str()), Some(reasoning.as_str()), "{resp}");
}

// ---------------------------------------------------------------------------
// The response contract with the GLM dialect (added in glm53f-afd): one reasoning field in both
// modes, the chunk head on every chunk, and harness/api_contract.py against a scripted stand-in.
// ---------------------------------------------------------------------------

const GLM_THINK: &str = concat!("<", "think", ">");
const GLM_THINK_END: &str = concat!("<", "/think", ">");

/// GLM-5.3-Flash's markup for one tool call with one argument.
fn glm_call(name: &str, key: &str, value: &str) -> String {
    format!(
        concat!("<", "tool_call", ">{}<", "arg_key", ">{}<", "/arg_key", "><", "arg_value", ">{}<", "/arg_value", "><", "/tool_call", ">"),
        name, key, value
    )
}

/// A scripted stand-in for GLM-5.3-Flash behind the GLM dialect. Its prompt is a small GLM-like
/// rendering of the request: the first tool's name and parameter, each message by role, an
/// assistant turn's reasoning unless `clear_thinking` drops it, and a generation prompt that
/// opens the think block (or, with thinking off, closes it). A prompt's token count is its
/// length in characters, so `usage.prompt_tokens` shows the switches and the decoded text. It
/// answers the contract harness's prompts correctly, starts inside the think block when the
/// prompt opened it, and streams three characters per delta, so tags and multi-byte characters
/// split across deltas.
struct GlmScript;

fn glm_script_prompt(messages: &[ChatMessage], tools: &[Tool], opts: &PromptOptions) -> String {
    // The template's effort line: low or high when asked for, else max (same length each).
    let effort = match opts.reasoning_effort.as_deref() {
        Some("low") => "low",
        Some("high") => "hig",
        _ => "max",
    };
    let mut s = format!("[effort:{effort}]");
    if let Some(t) = tools.first() {
        let props = t.function.parameters.as_ref().and_then(|p| p.get("properties")).and_then(|p| p.as_object());
        let param = props.and_then(|p| p.first()).map_or("", |(k, _)| k.as_str());
        s.push_str(&format!("[tools]{}({param})", t.function.name));
    }
    let last_user = messages.iter().rposition(|m| m.role == "user");
    for (i, m) in messages.iter().enumerate() {
        s.push_str(&format!("[{}]", m.role));
        if m.role == "assistant" {
            let keep = !opts.clear_thinking.unwrap_or(false) || last_user.is_none_or(|u| i > u);
            s.push_str(GLM_THINK);
            if keep {
                s.push_str(m.reasoning_content.as_deref().unwrap_or(""));
            }
            s.push_str(GLM_THINK_END);
        }
        s.push_str(&m.content);
    }
    s.push_str("[assistant]");
    s.push_str(GLM_THINK);
    if !opts.thinking {
        s.push_str(GLM_THINK_END);
    }
    s
}

/// The script's completion for a prompt: reasoning first when the prompt opened the think block.
fn glm_script_completion(prompt: &str) -> String {
    let low = prompt.starts_with("[effort:low]");
    let prompt = prompt.split_once(']').map_or(prompt, |(_, rest)| rest); // past the effort tag
    let question = prompt.rsplit("[user]").next().and_then(|q| q.rsplit_once("[assistant]")).map_or("", |(q, _)| q);
    let tool = prompt.strip_prefix("[tools]").and_then(|t| t.split_once('(')).map(|(n, rest)| (n, rest.split(')').next().unwrap_or("")));
    let (reasoning, answer) = if let Some((name, param)) = tool {
        ("The file has to be read first.", format!("I will read it.{}", glm_call(name, param, "src/theme/palette.js")))
    } else if let Some(text) = question.strip_prefix("Repeat exactly this text and nothing else: ") {
        ("An echo.", text.to_string())
    } else if question.starts_with("Reply exactly: OK") || question.starts_with("The secret word is") {
        ("A fixed reply.", "OK".to_string())
    } else if question.starts_with("What is the secret word") {
        ("There is no earlier message.", "NONE".to_string())
    } else if question.contains("2 + 2") {
        (if low { "2+2." } else { "Two and two." }, "4".to_string())
    } else {
        ("Let me think about it.", "The answer is 42.".to_string())
    };
    if prompt.ends_with(GLM_THINK) {
        format!("{reasoning}{GLM_THINK_END}{answer}")
    } else {
        answer
    }
}

impl Engine for GlmScript {
    fn tokenize(&self, messages: &[ChatMessage], tools: &[Tool], thinking: bool) -> usize {
        self.tokenize_prompt(messages, tools, &PromptOptions { thinking, ..Default::default() })
    }
    fn render_chat(&self, messages: &[ChatMessage], tools: &[Tool], thinking: bool) -> String {
        self.render_prompt(messages, tools, &PromptOptions { thinking, ..Default::default() })
    }
    fn tokenize_prompt(&self, messages: &[ChatMessage], tools: &[Tool], opts: &PromptOptions) -> usize {
        glm_script_prompt(messages, tools, opts).chars().count()
    }
    fn render_prompt(&self, messages: &[ChatMessage], tools: &[Tool], opts: &PromptOptions) -> String {
        glm_script_prompt(messages, tools, opts)
    }
    fn generate(
        &self,
        prompt: &str,
        params: &GenerateParams,
        on_delta: &mut dyn FnMut(&str),
    ) -> Result<GenerateOutcome, String> {
        let chars: Vec<char> = glm_script_completion(prompt).chars().collect();
        let pieces: Vec<String> = chars.chunks(3).take(params.max_tokens).map(|c| c.iter().collect()).collect();
        for p in &pieces {
            on_delta(p);
        }
        let finish = if pieces.len() * 3 < chars.len() { "length" } else { "stop" };
        Ok(GenerateOutcome { text: pieces.concat(), finish_reason: finish.into(), completion_tokens: pieces.len() })
    }
}

/// The data events of an SSE body, parsed (without `[DONE]`).
fn sse_events(raw: &str) -> Vec<glm53f_api::json::Json> {
    sse_data_payloads(raw)
        .iter()
        .filter(|p| p.as_str() != "[DONE]")
        .map(|p| glm53f_api::json::parse(p).unwrap_or_else(|e| panic!("bad event {p}: {e}")))
        .collect()
}

/// Reasoning is `reasoning_content` in both modes: streamed live in pieces and whole, never under
/// `reasoning`, never in content. The source streamed `delta.reasoning` beside a non-streamed
/// `reasoning_content`.
#[test]
fn reasoning_is_reasoning_content_whole_and_streamed() {
    let srv = start_engine_with(GlmScript, Arc::new(GlmDialect));
    let body = r#"{"model":"glm-5.3-flash","messages":[{"role":"user","content":"What is 17 + 25?"}],"stream":true}"#;
    let (status, resp) = http_post(&format!("{}/v1/chat/completions", srv.base), body);
    assert_eq!(status, 200, "{resp}");
    let (mut reasoning, mut content, mut pieces) = (String::new(), String::new(), 0);
    for ev in sse_events(&resp) {
        for ch in ev.get("choices").and_then(|c| c.as_array()).unwrap_or(&[]) {
            let Some(delta) = ch.get("delta") else { continue };
            assert!(delta.get("reasoning").is_none(), "reasoning under a second name: {resp}");
            if let Some(r) = delta.get("reasoning_content").and_then(|r| r.as_str()) {
                reasoning.push_str(r);
                pieces += 1;
            }
            if let Some(c) = delta.get("content").and_then(|c| c.as_str()) {
                content.push_str(c);
            }
        }
    }
    assert_eq!((reasoning.as_str(), content.as_str()), ("Let me think about it.", "The answer is 42."), "{resp}");
    assert!(pieces > 1, "reasoning streams live, in pieces: {resp}");

    let (status, resp) = http_post(&format!("{}/v1/chat/completions", srv.base), &body.replace(r#","stream":true"#, ""));
    assert_eq!(status, 200, "{resp}");
    let v = glm53f_api::json::parse(&resp).unwrap();
    let msg = v.get("choices").and_then(|c| c.as_array()).and_then(|a| a.first()).and_then(|c| c.get("message")).unwrap();
    assert_eq!(msg.get("reasoning_content").and_then(|r| r.as_str()), Some(reasoning.as_str()), "{resp}");
    assert_eq!(msg.get("content").and_then(|c| c.as_str()), Some(content.as_str()), "{resp}");
    assert!(msg.get("reasoning").is_none(), "{resp}");
}

/// Every streamed chunk carries the completion's `id`, `object`, `created` and `model`, with one
/// `id` and one `created` throughout: the role, reasoning, content, tool-call, finish and usage
/// chunks alike (the source sent them on the role chunk only). `created` is integer Unix
/// seconds, as in the non-streamed reply.
#[test]
fn every_chunk_carries_the_completion_head() {
    let srv = start_engine_with(GlmScript, Arc::new(GlmDialect));
    let now = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_secs_f64();
    let body = r#"{"model":"glm-5.3-flash","messages":[{"role":"user","content":"Read src/theme/palette.js."}],"tools":[{"type":"function","function":{"name":"read","parameters":{"type":"object","properties":{"file_path":{"type":"string"}}}}}],"stream":true,"stream_options":{"include_usage":true}}"#;
    let (status, resp) = http_post(&format!("{}/v1/chat/completions", srv.base), body);
    assert_eq!(status, 200, "{resp}");
    let events = sse_events(&resp);
    let delta_has = |k: &str| {
        events.iter().any(|ev| {
            ev.get("choices").and_then(|c| c.as_array()).and_then(|c| c.first()).and_then(|c| c.get("delta")).is_some_and(|d| d.get(k).is_some())
        })
    };
    for k in ["role", "reasoning_content", "content", "tool_calls"] {
        assert!(delta_has(k), "no {k} chunk: {resp}");
    }
    let usage = events.last().unwrap();
    assert!(usage.get("usage").is_some() && usage.get("choices").and_then(|c| c.as_array()).is_some_and(|c| c.is_empty()), "{resp}");
    let id = events[0].get("id").and_then(|x| x.as_str()).unwrap().to_string();
    let created = events[0].get("created").and_then(|x| x.as_f64()).unwrap();
    assert!(created.fract() == 0.0 && (created - now).abs() < 60.0, "created {created}, now {now}");
    for ev in &events {
        assert_eq!(ev.get("id").and_then(|x| x.as_str()), Some(id.as_str()), "{resp}");
        assert_eq!(ev.get("object").and_then(|x| x.as_str()), Some("chat.completion.chunk"), "{resp}");
        assert_eq!(ev.get("created").and_then(|x| x.as_f64()), Some(created), "{resp}");
        assert_eq!(ev.get("model").and_then(|x| x.as_str()), Some(MODEL_ID), "{resp}");
    }
    // On the wire, `created` is an integer.
    for p in sse_data_payloads(&resp).iter().filter(|p| p.as_str() != "[DONE]") {
        assert!(p.contains(&format!(r#""created":{created},"#)), "{p}");
    }

    let (status, resp) = http_post(&format!("{}/v1/chat/completions", srv.base), &body.replace(r#","stream":true"#, ""));
    assert_eq!(status, 200, "{resp}");
    let v = glm53f_api::json::parse(&resp).unwrap();
    assert_eq!(v.get("object").and_then(|x| x.as_str()), Some("chat.completion"), "{resp}");
    assert!(v.get("id").and_then(|x| x.as_str()).is_some_and(|i| i.starts_with("chatcmpl-")), "{resp}");
    assert_eq!(v.get("model").and_then(|x| x.as_str()), Some(MODEL_ID), "{resp}");
    let c = v.get("created").and_then(|x| x.as_f64()).unwrap();
    assert!(c.fract() == 0.0 && (c - now).abs() < 60.0 && resp.contains(&format!(r#""created":{c},"#)), "{resp}");
}

/// harness/api_contract.py, every row, against the API serving the GLM dialect over the scripted
/// stand-in: all PASS (the script answers the model rows correctly too).
#[test]
fn api_contract_harness_passes_against_the_glm_script() {
    let py = python();
    if Command::new(&py).arg("--version").output().is_err() {
        eprintln!("SKIP: python3 not available");
        return;
    }
    let Some(harness) = harness_file("api_contract.py") else { return };
    let srv = start_engine_with(GlmScript, Arc::new(GlmDialect));
    let out = Command::new(&py)
        .arg(&harness)
        .arg("--base").arg(&srv.base)
        .arg("--timeout").arg("60")
        .output().expect("run api_contract.py");
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(out.status.success(), "api_contract FAILED:\n{stdout}\n{}", String::from_utf8_lossy(&out.stderr));
    let rows = ["LIVE", "CREATED", "JSON", "STREAM-MARKUP", "USAGE", "UTF8-esc", "UTF8-raw", "TOOLS-json", "TOOLS-stream",
        "THINK-OFF", "CLEAR-THINKING", "ISO"];
    for row in rows {
        assert!(stdout.lines().any(|l| l.split_whitespace().take(2).eq([row, "PASS"])), "{row} did not pass:\n{stdout}");
    }
    assert!(stdout.contains("RESULT: PASS api contract"), "{stdout}");
}

/// Records the prompt options each request renders with, and answers with a short think block
/// then "OK" (the prompt opened the block when thinking is on).
struct OptionsStub(Arc<std::sync::Mutex<Vec<(bool, Option<String>)>>>);

impl Engine for OptionsStub {
    fn tokenize(&self, messages: &[ChatMessage], _tools: &[Tool], _thinking: bool) -> usize {
        messages.iter().map(|m| m.content.len()).sum::<usize>() / 4
    }
    fn render_chat(&self, messages: &[ChatMessage], _tools: &[Tool], _thinking: bool) -> String {
        last_content(messages)
    }
    fn render_prompt(&self, messages: &[ChatMessage], _tools: &[Tool], opts: &PromptOptions) -> String {
        self.0.lock().unwrap().push((opts.thinking, opts.reasoning_effort.clone()));
        last_content(messages)
    }
    fn generate(
        &self,
        _prompt: &str,
        params: &GenerateParams,
        on_delta: &mut dyn FnMut(&str),
    ) -> Result<GenerateOutcome, String> {
        // With thinking on the prompt opened the block: a short plan, its closing tag, the answer.
        let text = if params.thinking { "Answer briefly.\u{3c}/think\u{3e}OK" } else { "OK" };
        on_delta(text);
        Ok(GenerateOutcome { text: text.into(), finish_reason: "stop".into(), completion_tokens: 3 })
    }
}

/// GLM-5.3-Flash's chat template has no thinking-off mode: a request that turns thinking off
/// (`enable_thinking: false` either way, or `thinking.type: "disabled"`) renders with the
/// template's Low effort and thinking on, and its short reasoning comes back as reasoning, not
/// content. `reasoning_effort: "none"` still renders with thinking off (no reasoning at all), and
/// a request that sets no switch keeps the template's default (thinking on, its own effort).
#[test]
fn glm_thinking_off_is_low_effort() {
    let seen = Arc::new(std::sync::Mutex::new(Vec::new()));
    let srv = start_engine_with(OptionsStub(seen.clone()), Arc::new(GlmDialect));
    let cases = [
        (r#""chat_template_kwargs":{"enable_thinking":false}"#, (true, Some("low")), true),
        (r#""chat_template_kwargs":{"thinking":false}"#, (true, Some("low")), true),
        (r#""enable_thinking":false"#, (true, Some("low")), true),
        (r#""thinking":{"type":"disabled"}"#, (true, Some("low")), true),
        (r#""chat_template_kwargs":{"enable_thinking":false},"reasoning_effort":"high""#, (true, Some("low")), true),
        (r#""reasoning_effort":"none""#, (false, Some("none")), false),
        (r#""reasoning_effort":"low""#, (true, Some("low")), true),
        (r#""chat_template_kwargs":{"enable_thinking":true}"#, (true, None), true),
    ];
    for (extra, want, reasons) in cases {
        let body = format!(r#"{{"model":"glm-5.3-flash","messages":[{{"role":"user","content":"hi"}}],{extra}}}"#);
        let (status, resp) = http_post(&format!("{}/v1/chat/completions", srv.base), &body);
        assert_eq!(status, 200, "{extra}: {resp}");
        let got = seen.lock().unwrap().pop().expect("the prompt was rendered");
        assert_eq!((got.0, got.1.as_deref()), want, "{extra}");
        let v = glm53f_api::json::parse(&resp).unwrap();
        let msg = v.get("choices").and_then(|c| c.as_array()).and_then(|a| a.first()).and_then(|c| c.get("message")).unwrap();
        assert_eq!(msg.get("content").and_then(|c| c.as_str()), Some("OK"), "{extra}: {resp}");
        let reasoning = msg.get("reasoning_content").and_then(|c| c.as_str()).unwrap_or("");
        assert_eq!(reasoning.is_empty(), !reasons, "{extra}: {resp}");
    }
}

// ---------------------------------------------------------------------------
// Tool calls as GLM-5.3-Flash writes them, well formed or not (added in glm53f-afd): what the
// client receives, whole and streamed delta by delta, and what the server logs.
// ---------------------------------------------------------------------------

const GLM_TC: &str = concat!("<", "tool_call", ">");
const GLM_TC_END: &str = concat!("<", "/tool_call", ">");
const GLM_AK: &str = concat!("<", "arg_key", ">");
const GLM_AK_END: &str = concat!("<", "/arg_key", ">");
const GLM_AV: &str = concat!("<", "arg_value", ">");
const GLM_AV_END: &str = concat!("<", "/arg_value", ">");

/// A tool call in GLM-5.3-Flash's markup, its values written as the chat template writes them (a
/// string as is, anything else as JSON).
fn glm_tool_call(name: &str, args: &[(&str, &str)]) -> String {
    let mut s = format!("{GLM_TC}{name}");
    for (k, v) in args {
        s.push_str(&format!("{GLM_AK}{k}{GLM_AK_END}{GLM_AV}{v}{GLM_AV_END}"));
    }
    s + GLM_TC_END
}

/// The tools each request below offers: a shell whose `timeout` is an integer, and a file reader.
const AGENT_TOOLS: &str = r#"[{"type":"function","function":{"name":"bash","parameters":{"type":"object","properties":{"command":{"type":"string"},"timeout":{"type":"integer"}},"required":["command"]}}},{"type":"function","function":{"name":"read","parameters":{"type":"object","properties":{"file_path":{"type":"string"}},"required":["file_path"]}}}]"#;

/// The reasoning each completion below starts with (thinking is on: the prompt opened the block).
const AGENT_REASONING: &str = "The user wants the files listed.";

/// A completion and what the client must receive for it.
struct ToolCase {
    name: &'static str,
    /// The completion after the reasoning and its closing tag.
    text: String,
    /// `message.content` (`None`: null); streamed, the content deltas joined (`None`: none sent).
    content: Option<String>,
    /// Each call's name and arguments (a JSON object's text), in order.
    calls: Vec<(&'static str, &'static str)>,
    finish: &'static str,
    /// How each report of the parse, one logged line each, begins.
    reports: Vec<&'static str>,
}

fn tool_cases() -> Vec<ToolCase> {
    let unclosed = format!("{GLM_TC}bash{GLM_AK}command{GLM_AK_END}{GLM_AV}ls{GLM_AV_END}");
    let nameless = format!("{GLM_TC}{GLM_AK}command{GLM_AK_END}{GLM_AV}ls{GLM_AV_END}{GLM_TC_END}");
    let after_name = |name: &str, key: &str, value: &str| {
        format!("{GLM_TC}{name}{GLM_AK_END}{GLM_AK}{key}{GLM_AK_END}{GLM_AV}{value}{GLM_AV_END}{GLM_TC_END}")
    };
    vec![
        ToolCase {
            name: "good",
            text: glm_tool_call("bash", &[("command", "ls -la")]),
            content: None,
            calls: vec![("bash", r#"{"command":"ls -la"}"#)],
            finish: "tool_calls",
            reports: vec![],
        },
        ToolCase {
            name: "two args",
            text: glm_tool_call("bash", &[("command", "sleep 2 && echo done"), ("timeout", "30")]),
            content: None,
            calls: vec![("bash", r#"{"command":"sleep 2 && echo done","timeout":30}"#)],
            finish: "tool_calls",
            reports: vec![],
        },
        // The second call has a key outside its schema. Calls are not validated: both pass, and
        // nothing is reported.
        ToolCase {
            name: "good then bad",
            text: format!("{}\n{}", glm_tool_call("read", &[("file_path", "src/main.rs")]),
                glm_tool_call("bash", &[("command", "cargo test"), ("cwd", "crates/api")])),
            content: None,
            calls: vec![("read", r#"{"file_path":"src/main.rs"}"#), ("bash", r#"{"command":"cargo test","cwd":"crates/api"}"#)],
            finish: "tool_calls",
            reports: vec![],
        },
        // The name runs into a stray closing tag: recovered as the offered tool, and reported.
        ToolCase {
            name: "markup after the name",
            text: after_name("bash", "command", "git status"),
            content: None,
            calls: vec![("bash", r#"{"command":"git status"}"#)],
            finish: "tool_calls",
            reports: vec![r#"recovered call "bash" (markup after the name)"#],
        },
        ToolCase {
            name: "key not in schema",
            text: glm_tool_call("read", &[("path", "README.md")]),
            content: None,
            calls: vec![("read", r#"{"path":"README.md"}"#)],
            finish: "tool_calls",
            reports: vec![],
        },
        ToolCase {
            name: "tool not offered",
            text: glm_tool_call("grep", &[("pattern", "TODO"), ("max_count", "5")]),
            content: None,
            calls: vec![("grep", r#"{"pattern":"TODO","max_count":5}"#)],
            finish: "tool_calls",
            reports: vec![],
        },
        // Lost calls: the text reaches the client as content, after the text before it and
        // beside a call that parsed.
        ToolCase {
            name: "closing tag missing",
            text: format!("Let me look.{unclosed}"),
            content: Some(format!("Let me look.{unclosed}")),
            calls: vec![],
            finish: "stop",
            reports: vec![r#"lost call "bash" (closing tag missing)"#],
        },
        ToolCase {
            name: "a call opened inside a call",
            text: format!("{unclosed}{}", glm_tool_call("read", &[("file_path", "Cargo.toml")])),
            content: Some(unclosed.clone()),
            calls: vec![("read", r#"{"file_path":"Cargo.toml"}"#)],
            finish: "tool_calls",
            reports: vec![r#"lost call "bash" (closing tag missing before the next call)"#],
        },
        ToolCase {
            name: "markup after the name of a tool not offered",
            text: after_name("grep", "pattern", "TODO"),
            content: Some(after_name("grep", "pattern", "TODO")),
            calls: vec![],
            finish: "stop",
            reports: vec!["lost call (markup in the name)"],
        },
        // Arguments without a name are a lost call like the others: no error, no other finish
        // reason.
        ToolCase {
            name: "arguments without a name",
            text: nameless.clone(),
            content: Some(nameless.clone()),
            calls: vec![],
            finish: "stop",
            reports: vec!["lost call (arguments without a name)"],
        },
        ToolCase {
            name: "arguments without a name, then a call",
            text: format!("Let me look.\n{nameless}{}", glm_tool_call("read", &[("file_path", "Cargo.toml")])),
            content: Some(format!("Let me look.\n{nameless}")),
            calls: vec![("read", r#"{"file_path":"Cargo.toml"}"#)],
            finish: "tool_calls",
            reports: vec!["lost call (arguments without a name)"],
        },
        // The text before the first call is the reply's content (the whitespace that ends it is
        // dropped, none is null), and the text after it is not.
        ToolCase {
            name: "text before a call",
            text: format!("Let me look.\n{}", glm_tool_call("read", &[("file_path", "src/main.rs")])),
            content: Some("Let me look.".into()),
            calls: vec![("read", r#"{"file_path":"src/main.rs"}"#)],
            finish: "tool_calls",
            reports: vec![],
        },
        ToolCase {
            name: "whitespace before a call",
            text: format!("\n\n {}", glm_tool_call("read", &[("file_path", "src/main.rs")])),
            content: None,
            calls: vec![("read", r#"{"file_path":"src/main.rs"}"#)],
            finish: "tool_calls",
            reports: vec![],
        },
        ToolCase {
            name: "text before a call, whitespace in front of it kept",
            text: format!("\n\nLet me look. \n{}", glm_tool_call("read", &[("file_path", "src/main.rs")])),
            content: Some("\n\nLet me look.".into()),
            calls: vec![("read", r#"{"file_path":"src/main.rs"}"#)],
            finish: "tool_calls",
            reports: vec![],
        },
        ToolCase {
            name: "text between and after calls",
            text: format!(
                "First.\n{}\nThen this.\n{}\nDone.",
                glm_tool_call("read", &[("file_path", "a.rs")]),
                glm_tool_call("bash", &[("command", "cargo test")])
            ),
            content: Some("First.".into()),
            calls: vec![("read", r#"{"file_path":"a.rs"}"#), ("bash", r#"{"command":"cargo test"}"#)],
            finish: "tool_calls",
            reports: vec![],
        },
        ToolCase {
            name: "multi-byte text before a call",
            text: format!("\u{6771}\u{4eac}\u{306f}\u{6674}\u{308c}\u{3002} caf\u{e9}\n{}", glm_tool_call("read", &[("file_path", "a.rs")])),
            content: Some("\u{6771}\u{4eac}\u{306f}\u{6674}\u{308c}\u{3002} caf\u{e9}".into()),
            calls: vec![("read", r#"{"file_path":"a.rs"}"#)],
            finish: "tool_calls",
            reports: vec![],
        },
        // A reply without calls is untouched: its whitespace stays, streamed as it is written.
        ToolCase {
            name: "a plain reply",
            text: "  Nothing to call.\n\n".into(),
            content: Some("  Nothing to call.\n\n".into()),
            calls: vec![],
            finish: "stop",
            reports: vec![],
        },
        // Multi-byte characters before and inside a lost call.
        ToolCase {
            name: "multi-byte text around a lost call",
            text: format!("caf\u{e9} \u{6771}\u{4eac}\u{3002}{GLM_TC}bash{GLM_AK}command{GLM_AK_END}{GLM_AV}echo \u{6771}\u{4eac} caf\u{e9}{GLM_AV_END}"),
            content: Some(format!("caf\u{e9} \u{6771}\u{4eac}\u{3002}{GLM_TC}bash{GLM_AK}command{GLM_AK_END}{GLM_AV}echo \u{6771}\u{4eac} caf\u{e9}{GLM_AV_END}")),
            calls: vec![],
            finish: "stop",
            reports: vec![r#"lost call "bash" (closing tag missing)"#],
        },
    ]
}

/// One delta per character: every tag split across deltas.
fn by_char(text: &str) -> Vec<String> {
    text.chars().map(String::from).collect()
}

/// One delta per token, roughly as GLM-5.3-Flash's tokenizer splits the text: each tag one token
/// (they are added tokens of its vocabulary), other text in pieces of up to three characters.
fn by_token(text: &str) -> Vec<String> {
    const TAGS: [&str; 8] = [GLM_THINK, GLM_THINK_END, GLM_TC, GLM_TC_END, GLM_AK, GLM_AK_END, GLM_AV, GLM_AV_END];
    let (mut out, mut rest) = (Vec::new(), text);
    while !rest.is_empty() {
        let end = match TAGS.iter().find(|t| rest.starts_with(**t)) {
            Some(t) => t.len(),
            None => {
                let next_tag = TAGS.iter().filter_map(|t| rest.find(t)).min().unwrap_or(rest.len());
                rest.char_indices().nth(3).map_or(rest.len(), |(i, _)| i).min(next_tag)
            }
        };
        out.push(rest[..end].to_string());
        rest = &rest[end..];
    }
    out
}

/// GLM-5.3-Flash replaying fixed completions: the last user message names a [`ToolCase`], whose
/// completion (the reasoning, its closing tag, the case's text) goes out in the deltas `split`
/// makes.
struct ToolReplay {
    split: fn(&str) -> Vec<String>,
}

impl Engine for ToolReplay {
    fn tokenize(&self, messages: &[ChatMessage], _tools: &[Tool], _thinking: bool) -> usize {
        messages.iter().map(|m| m.content.len()).sum::<usize>() / 4 + 1
    }
    fn render_chat(&self, messages: &[ChatMessage], _tools: &[Tool], _thinking: bool) -> String {
        last_content(messages)
    }
    fn generate(
        &self,
        prompt: &str,
        _params: &GenerateParams,
        on_delta: &mut dyn FnMut(&str),
    ) -> Result<GenerateOutcome, String> {
        let case = tool_cases().into_iter().find(|c| c.name == prompt).ok_or("no such case")?;
        let deltas = (self.split)(&format!("{AGENT_REASONING}{GLM_THINK_END}{}", case.text));
        for d in &deltas {
            on_delta(d);
        }
        Ok(GenerateOutcome { text: deltas.concat(), finish_reason: "stop".into(), completion_tokens: deltas.len() })
    }
}

fn tool_case_body(case: &str, stream: bool) -> String {
    format!(r#"{{"model":"glm-5.3-flash","messages":[{{"role":"user","content":"{case}"}}],"tools":{AGENT_TOOLS},"stream":{stream}}}"#)
}

/// A streamed reply as a client gathers it: reasoning joined, the content deltas, and tool calls
/// assembled by index (the name from the first delta, the arguments joined).
struct Gathered {
    reasoning: String,
    content: Vec<String>,
    calls: Vec<(String, String)>,
    finish: Option<String>,
}

fn gather(raw: &str) -> Gathered {
    let mut g = Gathered { reasoning: String::new(), content: Vec::new(), calls: Vec::new(), finish: None };
    for ev in sse_events(raw) {
        for ch in ev.get("choices").and_then(|c| c.as_array()).unwrap_or(&[]) {
            if let Some(f) = ch.get("finish_reason").and_then(|f| f.as_str()) {
                g.finish = Some(f.to_string());
            }
            let Some(d) = ch.get("delta") else { continue };
            if let Some(r) = d.get("reasoning_content").and_then(|r| r.as_str()) {
                g.reasoning.push_str(r);
            }
            if let Some(c) = d.get("content").and_then(|c| c.as_str()) {
                g.content.push(c.to_string());
            }
            for tc in d.get("tool_calls").and_then(|t| t.as_array()).unwrap_or(&[]) {
                let i = tc.get("index").and_then(|i| i.as_f64()).expect("a tool-call index") as usize;
                if i == g.calls.len() {
                    g.calls.push((String::new(), String::new()));
                }
                let f = tc.get("function").expect("a function");
                g.calls[i].0.push_str(f.get("name").and_then(|n| n.as_str()).unwrap_or(""));
                g.calls[i].1.push_str(f.get("arguments").and_then(|a| a.as_str()).unwrap_or(""));
            }
        }
    }
    g
}

/// Six tool calls as GLM-5.3-Flash writes them, lost calls (never closed, opened inside another,
/// markup in the name, arguments without a name) and the text around calls, each whole and
/// streamed a character and a token at a time. Either way the client gets the same content,
/// calls, finish reason and reasoning:
/// - a lost call's text comes back as content, after the text before it and beside a call that
///   parsed, never as an empty turn or an error; finish_reason stays `stop` unless a call parsed;
/// - the text before the first call is the content, without the whitespace that ends it (null
///   when nothing is left); the text after it is dropped; a reply without calls is untouched;
/// - "markup after the name" is recovered as the offered tool;
/// - a key outside the schema and a tool not offered pass as written (calls are not validated).
///
/// The parse's reports, one logged line each, are checked here too.
#[test]
fn glm_tool_calls_as_the_model_writes_them() {
    let body = tool_case_body("", false);
    let tools = glm53f_api::types::ChatRequest::parse(&glm53f_api::json::parse(&body).unwrap(), &|_: &str| Err("no images".into()))
        .unwrap()
        .tools;
    let whole = start_engine_with(ToolReplay { split: |t| vec![t.to_string()] }, Arc::new(GlmDialect));
    let streamed = [
        ("by char", start_engine_with(ToolReplay { split: by_char }, Arc::new(GlmDialect))),
        ("by token", start_engine_with(ToolReplay { split: by_token }, Arc::new(GlmDialect))),
    ];
    for case in tool_cases() {
        let name = case.name;
        let want_calls: Vec<(String, String)> = case.calls.iter().map(|(n, a)| (n.to_string(), a.to_string())).collect();
        let parsed = glm53f_api::dialect::glm::parse(&format!("{AGENT_REASONING}{GLM_THINK_END}{}", case.text), &tools, true, 0);
        assert_eq!(parsed.reports.len(), case.reports.len(), "{name}: {:?}", parsed.reports);
        for (got, want) in parsed.reports.iter().zip(&case.reports) {
            assert!(got.starts_with(want), "{name}: report {got:?}");
        }

        let (status, resp) = http_post(&format!("{}/v1/chat/completions", whole.base), &tool_case_body(name, false));
        assert_eq!(status, 200, "{name}: {resp}");
        let v = glm53f_api::json::parse(&resp).unwrap();
        let choice = v.get("choices").and_then(|c| c.as_array()).and_then(|c| c.first()).unwrap();
        let msg = choice.get("message").unwrap();
        assert_eq!(msg.get("content").and_then(|c| c.as_str()), case.content.as_deref(), "{name}: {resp}");
        let calls: Vec<(String, String)> = msg.get("tool_calls").and_then(|t| t.as_array()).unwrap_or(&[]).iter()
            .map(|t| {
                let f = t.get("function").unwrap();
                (f.get("name").and_then(|n| n.as_str()).unwrap().to_string(), f.get("arguments").and_then(|a| a.as_str()).unwrap().to_string())
            })
            .collect();
        assert_eq!(calls, want_calls, "{name}: {resp}");
        assert_eq!(choice.get("finish_reason").and_then(|f| f.as_str()), Some(case.finish), "{name}: {resp}");
        assert_eq!(msg.get("reasoning_content").and_then(|r| r.as_str()), Some(AGENT_REASONING), "{name}: {resp}");
        let whole_reply = (msg.get("content").and_then(|c| c.as_str()).map(String::from), calls.clone(), case.finish, AGENT_REASONING);

        for (how, srv) in &streamed {
            let (status, resp) = http_post(&format!("{}/v1/chat/completions", srv.base), &tool_case_body(name, true));
            assert_eq!(status, 200, "{name} {how}: {resp}");
            let g = gather(&resp);
            let content = (!g.content.is_empty()).then(|| g.content.concat());
            assert_eq!(content, case.content, "{name} {how}: {resp}");
            // Markup reaches content only as a lost call's text, sent once after generation.
            let live = g.content.len() - usize::from(case.content.as_deref().is_some_and(|c| c.contains('<')));
            assert!(g.content[..live].iter().all(|c| !c.contains('<')), "{name} {how}: markup in a live delta: {resp}");
            assert_eq!(g.calls, want_calls, "{name} {how}: {resp}");
            assert_eq!(g.finish.as_deref(), Some(case.finish), "{name} {how}: {resp}");
            assert_eq!(g.reasoning, AGENT_REASONING, "{name} {how}: {resp}");
            // The streamed reply is the whole one.
            let finish = g.finish.as_deref().unwrap_or("");
            assert_eq!((content, g.calls.clone(), finish, g.reasoning.as_str()), whole_reply, "{name} {how}: streamed and whole replies differ");
        }
    }
}

/// GLM-5.3-Flash replaying one fixed completion (after the reasoning the prompt opened), in the
/// deltas `split` makes.
struct FixedCompletion {
    text: String,
    split: fn(&str) -> Vec<String>,
}

impl Engine for FixedCompletion {
    fn tokenize(&self, _messages: &[ChatMessage], _tools: &[Tool], _thinking: bool) -> usize {
        4
    }
    fn render_chat(&self, _messages: &[ChatMessage], _tools: &[Tool], _thinking: bool) -> String {
        String::new()
    }
    fn generate(
        &self,
        _prompt: &str,
        _params: &GenerateParams,
        on_delta: &mut dyn FnMut(&str),
    ) -> Result<GenerateOutcome, String> {
        let deltas = (self.split)(&format!("{AGENT_REASONING}{GLM_THINK_END}{}", self.text));
        for d in &deltas {
            on_delta(d);
        }
        Ok(GenerateOutcome { text: deltas.concat(), finish_reason: "stop".into(), completion_tokens: deltas.len() })
    }
}

/// A think block inside the text before a call is reasoning, not content, and the whitespace
/// around it is content as written; only the whitespace that ends the text goes. Whole and
/// streamed a character and a token at a time, the client gets the same content and reasoning.
#[test]
fn a_think_block_inside_the_text_before_a_call_leaves_whole_and_streamed_alike() {
    let text = format!("First. {GLM_THINK}hmm{GLM_THINK_END} Second. \n{}", glm_tool_call("read", &[("file_path", "a.rs")]));
    let body = |stream: bool| tool_case_body("", stream);
    let reasoning = format!("{AGENT_REASONING}hmm");
    let hows: [(&str, fn(&str) -> Vec<String>); 3] = [("whole", |t| vec![t.to_string()]), ("by char", by_char), ("by token", by_token)];
    for (how, split) in hows {
        let srv = start_engine_with(FixedCompletion { text: text.clone(), split }, Arc::new(GlmDialect));
        let (status, resp) = http_post(&format!("{}/v1/chat/completions", srv.base), &body(how != "whole"));
        assert_eq!(status, 200, "{how}: {resp}");
        let (content, calls, reasoning_got) = if how == "whole" {
            let v = glm53f_api::json::parse(&resp).unwrap();
            let msg = v.get("choices").and_then(|c| c.as_array()).and_then(|c| c.first()).and_then(|c| c.get("message")).unwrap();
            let calls = msg.get("tool_calls").and_then(|t| t.as_array()).map_or(0, |t| t.len());
            (msg.get("content").and_then(|c| c.as_str()).map(String::from), calls, msg.get("reasoning_content").and_then(|r| r.as_str()).unwrap_or("").to_string())
        } else {
            let g = gather(&resp);
            ((!g.content.is_empty()).then(|| g.content.concat()), g.calls.len(), g.reasoning)
        };
        assert_eq!(content.as_deref(), Some("First.  Second."), "{how}: {resp}");
        assert_eq!((calls, reasoning_got.as_str()), (1, reasoning.as_str()), "{how}: {resp}");
    }
}

/// Each report of the parse is logged on stderr, one line, under the completion's id, whole and
/// streamed. The server logs from its own threads, so `lost_call_log_child` serves the requests
/// in a child process of this test binary, and this test reads the child's output.
#[test]
fn parse_reports_are_logged_under_the_completion_id() {
    let out = Command::new(std::env::current_exe().unwrap())
        .args(["lost_call_log_child", "--exact", "--ignored", "--nocapture", "--test-threads=1"])
        .env("GLM53F_API_LOG_CHILD", "1")
        .output()
        .expect("run the child");
    let (stdout, stderr) = (String::from_utf8_lossy(&out.stdout), String::from_utf8_lossy(&out.stderr));
    assert!(out.status.success(), "{stdout}\n{stderr}");
    // The first id shares its line with the test runner's "test ... " prefix.
    let ids: Vec<&str> = stdout.lines().filter_map(|l| l.split_once("completion id: ").map(|(_, id)| id)).collect();
    assert_eq!(ids.len(), 4, "{stdout}");
    for (i, id) in ids.iter().enumerate() {
        let lines: Vec<&str> = stderr.lines().filter(|l| l.starts_with(&format!("[api] {id}: "))).collect();
        let want = if i < 2 { r#"lost call "bash" (closing tag missing before the next call)"# } else { r#"recovered call "bash""# };
        assert_eq!(lines.len(), 1, "{id}: {stderr}");
        assert!(lines[0].contains(want), "{}", lines[0]);
    }
}

/// Serves a lost call and a recovered one, whole and streamed, printing each reply's completion
/// id (for `parse_reports_are_logged_under_the_completion_id`).
#[test]
#[ignore = "run by parse_reports_are_logged_under_the_completion_id"]
fn lost_call_log_child() {
    if std::env::var_os("GLM53F_API_LOG_CHILD").is_none() {
        return;
    }
    let srv = start_engine_with(ToolReplay { split: by_token }, Arc::new(GlmDialect));
    for (case, stream) in [("a call opened inside a call", false), ("a call opened inside a call", true),
        ("markup after the name", false), ("markup after the name", true)] {
        let (status, resp) = http_post(&format!("{}/v1/chat/completions", srv.base), &tool_case_body(case, stream));
        assert_eq!(status, 200, "{resp}");
        let head = if stream { sse_events(&resp).remove(0) } else { glm53f_api::json::parse(&resp).unwrap() };
        println!("completion id: {}", head.get("id").and_then(|i| i.as_str()).unwrap());
    }
}
