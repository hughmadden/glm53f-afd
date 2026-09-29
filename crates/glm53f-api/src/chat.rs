//! The `POST /v1/chat/completions` handler: decode the request, run the engine,
//! parse tool calls, and render the OpenAI non-stream or SSE response.

use std::net::TcpStream;
use std::sync::Arc;
use std::time::{Duration, Instant};

use crate::dialect::{Dialect, ParseResult, ParsedCall, StreamTags};
use crate::engine::{Engine, GenerateOutcome, GenerateParams, PromptOptions};
use crate::http::{self, json_response, Response};
use crate::json::{self, Json};
use crate::types::{self, ApiError, ChatRequest, Tool, ToolCall, MODEL_ID};

/// The response field that carries reasoning, in both modes: the non-streamed
/// `message.reasoning_content` and every streamed `delta.reasoning_content`. This is the DeepSeek
/// convention, which vLLM, SGLang and LiteLLM read and clients send back in the history. The
/// source streamed `delta.reasoning` beside a non-streamed `reasoning_content`, so a client that
/// reads one name lost the reasoning of the other mode.
pub const REASONING_FIELD: &str = "reasoning_content";

pub fn handle<E: Engine + Send + Sync + 'static>(
    engine: Arc<E>,
    dialect: Arc<dyn Dialect>,
    body: &Json,
) -> Result<Response, ApiError> {
    let req = ChatRequest::parse(body, &|url: &str| engine.decode_image(url))?;
    if !req.images.is_empty() && !engine.vision() {
        return Err(ApiError::bad_request("this server has no image encoder; image parts are rejected"));
    }

    // The thinking switch: the request's, else the dialect's default (the chat template's). A
    // template with no off mode maps "off" to its lowest effort, thinking on
    // (`Dialect::thinking_off_effort`), and so does a `reasoning_effort` that names the lowest
    // effort ("none", "minimal"), whatever the switch says: the prompt is never an empty block.
    let requested = req.enable_thinking.unwrap_or_else(|| dialect.default_thinking());
    let lowest = types::lowest_effort(req.reasoning_effort.as_deref());
    let (thinking, reasoning_effort) = match dialect.thinking_off_effort() {
        Some(low) if !requested || lowest => (true, Some(low.to_string())),
        _ => (requested, req.reasoning_effort.clone()),
    };
    let opts = PromptOptions { thinking, reasoning_effort, clear_thinking: req.clear_thinking };
    let prompt_tokens = engine.tokenize_prompt(&req.messages, &req.tools, &opts);
    if let Some(max) = engine.max_context() {
        if prompt_tokens >= max {
            return Err(ApiError::bad_request(format!(
                "prompt is {prompt_tokens} tokens; this deployment's maximum context is {max} tokens"
            )));
        }
    }
    let prompt = engine.render_prompt(&req.messages, &req.tools, &opts);
    let params = GenerateParams {
        max_tokens: req.max_tokens.unwrap_or(65_536) as usize,
        temperature: req.temperature.unwrap_or(0.0),
        top_p: req.top_p.unwrap_or(1.0),
        top_k: req.top_k.unwrap_or(0),
        min_p: req.min_p.unwrap_or(0.0),
        seed: req.seed,
        stop: req.stop.clone(),
        thinking,
        cancel: Some(std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false))),
        images: req.images.clone(),
        place: Default::default(),
    };
    // A place in the engine's queue before the response starts (perf reset V3): a full queue is a
    // 429 the client can retry, not a stream that fails.
    *params.place.lock().unwrap_or_else(|p| p.into_inner()) = engine.admit().map_err(ApiError::too_many_requests)?;

    let id = format!("chatcmpl-{}", now_nanos());
    let created = now_secs();

    if req.stream {
        // Streaming: send the head first (chunked), then each SSE event as its
        // own chunk, so the first content delta reaches the client immediately.
        let tools = req.tools.clone();
        let include_usage = req.include_usage;
        let engine = engine.clone();
        let dialect = dialect.clone();
        let body = Box::new(move |stream: &mut TcpStream| -> std::io::Result<()> {
            stream_events(
                engine.as_ref(),
                dialect.as_ref(),
                stream,
                &prompt,
                &params,
                &tools,
                include_usage,
                prompt_tokens,
                &id,
                created,
            )
        });
        Ok(http::sse_stream_response(body))
    } else {
        let outcome: GenerateOutcome = engine
            .generate(&prompt, &params, &mut |_d: &str| {})
            .map_err(ApiError::internal)?;

        // Parse the completion for think blocks and tool calls. The T24 tool-call
        // cap is a coordinator policy, off in the API by default (a request may
        // legitimately call more than the storm cap, e.g. the 7-tool replay_exact).
        let cap = 0;
        let parsed = dialect.parse(&outcome.text, &req.tools, thinking, cap);
        log_reports(&id, &parsed);
        if let Some(e) = parsed.error {
            return Err(ApiError::bad_request(format!("tool call parse error: {e}")));
        }

        let finish_reason = if parsed.capped || !parsed.calls.is_empty() {
            "tool_calls".to_string()
        } else {
            outcome.finish_reason.clone()
        };

        Ok(json_response(200, &json::serialize(&non_stream_json(&parsed, &finish_reason, &id, created, prompt_tokens, outcome.completion_tokens))))
    }
}

fn now_secs() -> u64 {
    std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0)
}
fn now_nanos() -> u128 {
    std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_nanos()).unwrap_or(0)
}

/// Every parse report goes to the log, one line each, under the completion's id: a lost or
/// recovered tool call is visible on the server as well as to the client.
fn log_reports(id: &str, parsed: &ParseResult) {
    for r in &parsed.reports {
        eprintln!("[api] {id}: {r}");
    }
}

/// The text of a reply, whole or streamed: the parsed content as it is, but for a reply with tool
/// calls, without the whitespace that ends it (it only separated the text from the first call).
fn reply_content(parsed: &ParseResult) -> &str {
    if parsed.calls.is_empty() {
        &parsed.content
    } else {
        parsed.content.trim_end()
    }
}

/// The assistant message content for a response: the reply's text, and `None` for a tool-call
/// turn that has none.
fn message_content(parsed: &ParseResult) -> Json {
    let text = reply_content(parsed);
    if text.is_empty() && !parsed.calls.is_empty() {
        Json::Null
    } else {
        Json::Str(text.to_string())
    }
}

fn tool_calls_json(calls: &[ParsedCall]) -> Vec<Json> {
    calls.iter().enumerate().map(|(i, c)| {
        let args = json::serialize(&c.arguments);
        ToolCall { id: format!("call_{i}"), r#type: "function".into(), name: c.name.clone(), arguments: args }.to_json()
    }).collect()
}

fn non_stream_json(
    parsed: &ParseResult,
    finish_reason: &str,
    id: &str,
    created: u64,
    prompt_tokens: usize,
    completion_tokens: usize,
) -> Json {
    let mut message = vec![("role".to_string(), Json::Str("assistant".to_string()))];
    message.push(("content".to_string(), message_content(parsed)));
    if !parsed.reasoning.is_empty() {
        message.push((REASONING_FIELD.to_string(), Json::Str(parsed.reasoning.join(""))));
    }
    if !parsed.calls.is_empty() {
        message.push(("tool_calls".to_string(), Json::Array(tool_calls_json(&parsed.calls))));
    }
    types::obj(vec![
        ("id", types::s(id)),
        ("object", types::s("chat.completion")),
        ("created", Json::Num(created as f64)),
        ("model", types::s(MODEL_ID)),
        ("choices", types::arr(vec![types::obj(vec![
            ("index", Json::Num(0.0)),
            ("message", Json::Object(message)),
            ("finish_reason", types::s(finish_reason)),
        ])])),
        ("usage", types::usage_json(prompt_tokens as u64, completion_tokens as u64)),
    ])
}

/// Render one SSE event (`data: {json}\n\n`).
fn sse_event(obj: &Json) -> String {
    let mut s = String::from("data: ");
    s.push_str(&json::serialize(obj));
    s.push_str("\n\n");
    s
}

/// What every streamed chunk carries besides its choices, as OpenAI's chunk objects do: the
/// completion's `id`, `object`, `created` and `model`. The source sent them on the role chunk
/// only, so a client that decodes each chunk into a type with these fields required failed on
/// the second chunk.
#[derive(Debug, Clone)]
struct ChunkHead {
    id: String,
    created: u64,
}

impl ChunkHead {
    /// One chunk: the head fields, then `rest` (the choices, and the usage on the usage chunk).
    fn event(&self, rest: Vec<(&str, Json)>) -> String {
        let mut pairs = vec![
            ("id", types::s(&self.id)),
            ("object", types::s("chat.completion.chunk")),
            ("created", Json::Num(self.created as f64)),
            ("model", types::s(MODEL_ID)),
        ];
        pairs.extend(rest);
        sse_event(&types::obj(pairs))
    }

    /// A chunk of the one choice: its `delta` and `finish_reason`.
    fn choice_event(&self, delta: Json, finish_reason: Json) -> String {
        self.event(vec![(
            "choices",
            types::arr(vec![types::obj(vec![
                ("index", Json::Num(0.0)),
                ("delta", delta),
                ("finish_reason", finish_reason),
            ])]),
        )])
    }

    fn role_event(&self) -> String {
        self.choice_event(types::obj(vec![("role", types::s("assistant"))]), Json::Null)
    }

    fn reasoning_event(&self, r: &str) -> String {
        self.choice_event(types::obj(vec![(REASONING_FIELD, types::s(r))]), Json::Null)
    }

    fn content_event(&self, d: &str) -> String {
        self.choice_event(types::obj(vec![("content", types::s(d))]), Json::Null)
    }

    fn tool_call_header_event(&self, i: usize, c: &ParsedCall) -> String {
        let call = types::obj(vec![
            ("index", Json::Num(i as f64)),
            ("id", types::s(&format!("call_{i}"))),
            ("type", types::s("function")),
            ("function", types::obj(vec![("name", types::s(&c.name)), ("arguments", types::s(""))])),
        ]);
        self.choice_event(types::obj(vec![("tool_calls", types::arr(vec![call]))]), Json::Null)
    }

    fn tool_call_args_event(&self, i: usize, c: &ParsedCall) -> String {
        let args = json::serialize(&c.arguments);
        let call = types::obj(vec![
            ("index", Json::Num(i as f64)),
            ("function", types::obj(vec![("arguments", types::s(&args))])),
        ]);
        self.choice_event(types::obj(vec![("tool_calls", types::arr(vec![call]))]), Json::Null)
    }

    fn finish_event(&self, finish_reason: &str) -> String {
        self.choice_event(types::obj(vec![]), types::s(finish_reason))
    }

    /// The `include_usage` chunk: no choices, and the usage block.
    fn usage_event(&self, prompt_tokens: usize, completion_tokens: usize) -> String {
        self.event(vec![
            ("choices", Json::Array(vec![])),
            ("usage", types::usage_json(prompt_tokens as u64, completion_tokens as u64)),
        ])
    }
}

/// Where the streamed completion currently is, in the dialect parser's terms.
#[derive(Clone, Copy, PartialEq)]
enum Span {
    Content,
    Think,
    Tool,
}

/// Split the streamed completion the way the dialect's `parse` splits the whole
/// text: content deltas outside blocks, reasoning deltas inside a `<think>` block
/// (reasoning up to the first `</think>`), and everything from a `<tool_call>`
/// on held back (re-emitted as parsed `tool_calls` deltas after generation, and a
/// lost call as one content delta). A tag split across deltas is held until it
/// resolves, so no markup reaches a live content or reasoning delta. Whitespace
/// that ends the text so far is held too: a tool call that follows drops it (a
/// reply with calls carries its text without it, see [`reply_content`]), any other
/// text sends it first. Before this, think blocks streamed as content and were
/// then repeated as reasoning after generation. The tags are the dialect's
/// ([`StreamTags`]); a dialect whose prompt opens the reasoning block starts the
/// split inside it. What it holds back is not written, so it also keeps the time of
/// its last write: a stream that has been quiet for [`Engine::keepalive`] while the model
/// writes a tool call gets a keepalive comment ([`StreamSplit::keepalive_if_idle`]).
struct StreamSplit {
    hold: String,
    span: Span,
    /// Think blocks already streamed live; the post-parse pass sends the rest.
    think_blocks: usize,
    /// Bytes of content streamed live: the start of the parsed content not yet sent.
    content_bytes: usize,
    tags: StreamTags,
    head: ChunkHead,
    /// When the last chunk went out (the role chunk, at first).
    last_write: Instant,
}

impl StreamSplit {
    fn new(tags: StreamTags, reasoning_first: bool, head: ChunkHead) -> Self {
        let last_write = Instant::now();
        if reasoning_first {
            StreamSplit { hold: String::new(), span: Span::Think, think_blocks: 1, content_bytes: 0, tags, head, last_write }
        } else {
            StreamSplit { hold: String::new(), span: Span::Content, think_blocks: 0, content_bytes: 0, tags, head, last_write }
        }
    }

    /// Write one chunk, and note when.
    fn send(&mut self, stream: &mut TcpStream, event: &str) -> std::io::Result<()> {
        http::write_chunk(stream, event.as_bytes())?;
        self.last_write = Instant::now();
        Ok(())
    }

    /// The comment that keeps a quiet stream open: SSE comments carry no data, and a client or
    /// proxy with an idle timeout sees bytes.
    fn keepalive(&mut self, stream: &mut TcpStream) -> std::io::Result<()> {
        self.send(stream, ": keepalive\n\n")
    }

    /// A keepalive if nothing has been written for `idle`. A tool call is held back until it is
    /// complete, so while the model writes one the stream writes nothing, however long it takes.
    fn keepalive_if_idle(&mut self, stream: &mut TcpStream, idle: Duration) -> std::io::Result<()> {
        if self.last_write.elapsed() >= idle {
            self.keepalive(stream)
        } else {
            Ok(())
        }
    }

    fn push(&mut self, stream: &mut TcpStream, delta: &str) -> std::io::Result<()> {
        let StreamTags { think_open, think_close, tool_open } = self.tags;
        self.hold.push_str(delta);
        loop {
            match self.span {
                Span::Tool => return Ok(()),
                Span::Think => {
                    if let Some(p) = self.hold.find(think_close) {
                        if p > 0 {
                            let event = self.head.reasoning_event(&self.hold[..p]);
                            self.send(stream, &event)?;
                        }
                        self.hold.drain(..p + think_close.len());
                        self.span = Span::Content;
                    } else {
                        let flush = self.hold.len() - held_suffix(&self.hold, &[think_close]);
                        if flush > 0 {
                            let event = self.head.reasoning_event(&self.hold[..flush]);
                            self.send(stream, &event)?;
                            self.hold.drain(..flush);
                        }
                        return Ok(());
                    }
                }
                Span::Content => {
                    let think = self.hold.find(think_open);
                    let tool = self.hold.find(tool_open);
                    match (think, tool) {
                        (Some(p), t) if t.is_none_or(|t| p < t) => {
                            if p > 0 {
                                let event = self.head.content_event(&self.hold[..p]);
                                self.send(stream, &event)?;
                                self.content_bytes += p;
                            }
                            self.hold.drain(..p + think_open.len());
                            self.span = Span::Think;
                            self.think_blocks += 1;
                        }
                        (_, Some(p)) => {
                            // A tool call starts here: stream the content before it, less the
                            // whitespace that ends it, then hold the markup (and anything after)
                            // back.
                            let text = self.hold[..p].trim_end();
                            if !text.is_empty() {
                                let (event, len) = (self.head.content_event(text), text.len());
                                self.send(stream, &event)?;
                                self.content_bytes += len;
                            }
                            self.hold.drain(..p);
                            self.span = Span::Tool;
                        }
                        _ => {
                            // No tag yet (the first arm takes a think tag with no tool tag). Hold
                            // what may still become a tag, and the whitespace before it: a tool
                            // call may follow.
                            let unheld = self.hold.len() - held_suffix(&self.hold, &[think_open, tool_open]);
                            let flush = self.hold[..unheld].trim_end().len();
                            if flush > 0 {
                                let event = self.head.content_event(&self.hold[..flush]);
                                self.send(stream, &event)?;
                                self.content_bytes += flush;
                                self.hold.drain(..flush);
                            }
                            return Ok(());
                        }
                    }
                }
            }
        }
    }

    /// After generation, text still held is what the parser reads there: plain
    /// content (an unfinished tag is text) or the rest of an unclosed think block.
    /// Before this, a completion ending in a tag prefix (a bare `<`) lost it.
    fn finish(&mut self, stream: &mut TcpStream) -> std::io::Result<()> {
        if !self.hold.is_empty() {
            match self.span {
                Span::Content => {
                    http::write_chunk(stream, self.head.content_event(&self.hold).as_bytes())?;
                    self.content_bytes += self.hold.len();
                }
                Span::Think => http::write_chunk(stream, self.head.reasoning_event(&self.hold).as_bytes())?,
                Span::Tool => {}
            }
        }
        if self.span != Span::Tool {
            self.hold.clear();
        }
        Ok(())
    }
}

/// The longest suffix of `hold` that may still become one of `tags` (a prefix
/// of it). Steps only over char boundaries: the tags are ASCII, so a suffix that
/// starts mid-character can never be a prefix, and slicing mid-character would
/// panic on multi-byte text (D4).
fn held_suffix(hold: &str, tags: &[&str]) -> usize {
    let mut keep = 0;
    for tag in tags {
        for k in 1..=tag.len().min(hold.len()) {
            let start = hold.len() - k;
            if hold.is_char_boundary(start) && tag.starts_with(&hold[start..]) {
                keep = keep.max(k);
            }
        }
    }
    keep
}

/// Stream the SSE response directly to the socket: the role first, then each
/// content or reasoning delta as the engine produces it (flushed per event, with
/// the markup held back), then the post-parse events (reasoning behind a tool
/// call, a lost call's text, tool calls, finish, usage) and `data: [DONE]`. The
/// parse reports are logged under the completion's id. Every chunk carries the
/// completion's id, `created` and model ([`ChunkHead`]).
#[allow(clippy::too_many_arguments)]
fn stream_events<E: Engine>(
    engine: &E,
    dialect: &dyn Dialect,
    stream: &mut TcpStream,
    prompt: &str,
    params: &GenerateParams,
    tools: &[Tool],
    include_usage: bool,
    prompt_tokens: usize,
    id: &str,
    created: u64,
) -> std::io::Result<()> {
    let head = ChunkHead { id: id.to_string(), created };
    // 1. role delta, sent before generation so the client sees the stream start.
    http::write_chunk(stream, head.role_event().as_bytes())?;

    // 2. generate, streaming content and reasoning deltas (markup held back).
    let mut split = StreamSplit::new(dialect.stream_tags(), dialect.reasoning_first(params.thinking), head.clone());
    // An empty delta is the engine's keepalive while a long prompt prefills: an
    // SSE comment keeps the client and any proxy from timing out. So is one whenever the
    // stream has written nothing for the engine's keepalive interval while it generates: a
    // tool call is held back until it is complete. A failed write means the client is gone:
    // tell the engine to stop (perf reset Q2).
    let cancel = params.cancel.clone();
    let keepalive = engine.keepalive();
    let outcome: GenerateOutcome = match engine.generate(prompt, params, &mut |d: &str| {
        let wrote = if d.is_empty() {
            split.keepalive(stream)
        } else {
            split.push(stream, d).and_then(|()| split.keepalive_if_idle(stream, keepalive))
        };
        if wrote.is_err() {
            if let Some(c) = &cancel {
                c.store(true, std::sync::atomic::Ordering::Relaxed);
            }
        }
    }) {
        Ok(o) => o,
        Err(_) => {
            // The head is already sent; end the stream cleanly on engine error.
            http::write_chunk(stream, head.finish_event("error").as_bytes())?;
            http::write_chunk(stream, b"data: [DONE]\n\n")?;
            return Ok(());
        }
    };

    split.finish(stream)?;

    // 3. parse the full completion for think blocks and tool calls.
    let cap = 0;
    let parsed = dialect.parse(&outcome.text, tools, params.thinking, cap);
    log_reports(id, &parsed);
    if parsed.error.is_some() {
        http::write_chunk(stream, head.finish_event("error").as_bytes())?;
        http::write_chunk(stream, b"data: [DONE]\n\n")?;
        return Ok(());
    }
    let finish_reason = if parsed.capped || !parsed.calls.is_empty() {
        "tool_calls".to_string()
    } else {
        outcome.finish_reason.clone()
    };

    // 4. reasoning deltas for think blocks not streamed live (after a tool call).
    for r in parsed.reasoning.iter().skip(split.think_blocks) {
        http::write_chunk(stream, head.reasoning_event(r).as_bytes())?;
    }

    // 5. the reply's text not streamed live: what was held back from a lost call's opening tag
    //    on, and the whitespace before it. With what went out live it adds up to the
    //    non-streamed reply's content.
    if let Some(rest) = reply_content(&parsed).get(split.content_bytes..).filter(|r| !r.is_empty()) {
        http::write_chunk(stream, head.content_event(rest).as_bytes())?;
    }

    // 6. tool-call deltas (header then arguments).
    for (i, c) in parsed.calls.iter().enumerate() {
        http::write_chunk(stream, head.tool_call_header_event(i, c).as_bytes())?;
        http::write_chunk(stream, head.tool_call_args_event(i, c).as_bytes())?;
    }

    // 7. finish.
    http::write_chunk(stream, head.finish_event(&finish_reason).as_bytes())?;

    // 8. usage (only when requested).
    if include_usage {
        http::write_chunk(stream, head.usage_event(prompt_tokens, outcome.completion_tokens).as_bytes())?;
    }

    // 9. done.
    http::write_chunk(stream, b"data: [DONE]\n\n")?;
    Ok(())
}
