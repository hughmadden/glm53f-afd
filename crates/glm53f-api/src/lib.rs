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
