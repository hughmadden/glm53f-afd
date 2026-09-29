//! `glm53f-api` — the OpenAI-compatible HTTP/1.1 API (A8, I5-R8).
//!
//! std-only (no external crates). Serves `GET /v1/models` and
//! `POST /v1/chat/completions` (non-stream + SSE with the `include_usage` usage
//! block), splits completions into content, reasoning and tool calls through
//! the model's [`Dialect`] (the MiMo dialect, COHERENCE-TRAPS T27/T29 and cap
//! T24, is the reference), and rejects media content parts and constrained
//! `tool_choice` with a 400. The model itself is behind the [`Engine`] trait,
//! which the coordinator implements; the tests drive the server with a stub
//! engine.
//!
//! # Reasoning and the thinking switch
//!
//! Reasoning goes out under one name in both modes, [`chat::REASONING_FIELD`]
//! (`reasoning_content`): the non-streamed `message.reasoning_content` and each
//! streamed `delta.reasoning_content`, never in `content`. Every streamed chunk
//! carries the completion's `id`, `object`, `created` (Unix seconds) and `model`.
//!
//! A request sets thinking with the first of these it carries:
//!
//! 1. `chat_template_kwargs.enable_thinking` (vLLM and SGLang);
//! 2. `chat_template_kwargs.thinking`, a boolean (the spelling some clients and chat templates
//!    use);
//! 3. a top-level `enable_thinking`;
//! 4. `thinking.type` (GLM and Anthropic): `"disabled"` is off, any other type on;
//! 5. `reasoning_effort` (top level, else in `chat_template_kwargs`) of `"none"` or `"minimal"`,
//!    the names OpenAI clients send for the lowest effort: off.
//!
//! Otherwise the dialect's default applies ([`Dialect::default_thinking`]; on for
//! GLM-5.3-Flash). A dialect whose chat template has no off mode maps "off" to its lowest
//! effort with thinking on ([`Dialect::thinking_off_effort`]; GLM-5.3-Flash: `"low"`), and so
//! does a `reasoning_effort` of `"none"` or `"minimal"`, whatever the switch says: `"none"` is
//! exactly thinking off, and the API renders only the efforts the template has (GLM-5.3-Flash:
//! never an empty think block). Every other `reasoning_effort` goes to the template as sent,
//! which decides what a value means (GLM-5.3-Flash: exactly `"low"` and `"high"` are Low and
//! High; anything else, `"medium"` and `"max"` too, is Max). The engine gets the switch with
//! `reasoning_effort` and `clear_thinking` (`chat_template_kwargs`, else `thinking`; off unless
//! sent), which drops the reasoning of assistant turns before the last user
//! message. An assistant turn's reasoning in the history is read from
//! `reasoning_content`, else `reasoning`. `harness/api_contract.py` checks all of
//! this against a running server.
//!
//! # Tool calls
//!
//! A reply with tool calls carries the text the model wrote before them as `content`, streamed
//! or not (`null`, or no content delta, when there is none). The whitespace that ends that text
//! is dropped; text after the first call is dropped.
//!
//! A call the GLM dialect cannot parse is never dropped: its text goes to the client as
//! `content`, beside any calls that did parse, streamed or not, and `finish_reason` is
//! `tool_calls` only when a call parsed. That includes a call with arguments but no name: a
//! reply never fails, or ends with another finish reason, for what the model wrote. (The MiMo
//! reference dialect reports its lost calls without keeping their text, and fails the request
//! on a nameless one.) One malformed shape is recovered: a name followed by a stray closing tag,
//! when what is left is a tool the request offered. A call that parses is passed on as written:
//! its name is not checked against the request's tools, nor its arguments against their schema
//! (which only types them). Every report of the parse (a lost call, a dropped argument, a
//! recovered name) is logged to stderr, one line each, under the completion's id.

pub mod chat;
pub mod dialect;
pub mod engine;
pub mod http;
pub mod json;
pub mod models;
pub mod types;

use std::io;
use std::sync::Arc;

pub use dialect::{Dialect, ParseResult, ParsedCall, StreamTags};
pub use engine::{Engine, GenerateOutcome, GenerateParams};
pub use types::{ApiError, ChatMessage, ChatRequest, Tool, ToolCall, MODEL_ID};

use http::Request;

/// Run the API on `addr` (e.g. `0.0.0.0:8000`) with the given engine and the
/// model's completion dialect.
pub fn serve<E: Engine + Send + Sync + 'static>(addr: &str, engine: Arc<E>, dialect: Arc<dyn Dialect>) -> io::Result<()> {
    http::serve(addr, move |req| route(engine.clone(), dialect.clone(), req))
}

fn route<E: Engine + Send + Sync + 'static>(engine: Arc<E>, dialect: Arc<dyn Dialect>, req: Request) -> http::Response {
    // Strip a query string for routing (the tools use exact paths anyway).
    let path = req.path.split('?').next().unwrap_or("");
    match (req.method.as_str(), path) {
        ("GET", "/v1/models") => models::handle(),
        ("POST", "/v1/chat/completions") => {
            match json::parse_bytes(&req.body) {
                Ok(body) => match chat::handle(engine, dialect, &body) {
                    Ok(resp) => resp,
                    Err(e) => http::json_response(e.status, &json::serialize(&e.body())),
                },
                Err(e) => {
                    let err = ApiError::bad_request(format!("invalid JSON body: {e}"));
                    http::json_response(400, &json::serialize(&err.body()))
                }
            }
        }
        _ => {
            let err = ApiError::not_found("not found");
            http::json_response(404, &json::serialize(&err.body()))
        }
    }
}
