//! `GET /health` — whether the engine can serve: 200 `{"status":"ok"}`, or 503
//! `{"status":"unavailable","reason":...}` with the engine's reason ([`Engine::health`]). It
//! answers from the engine's state, never through its request queue or its model, so a probe gets
//! its answer at once while requests run or wait.

use crate::engine::Engine;
use crate::http::{json_response, Response};
use crate::json::{self, Json};

pub fn handle<E: Engine + ?Sized>(engine: &E) -> Response {
    let s = |v: &str| Json::Str(v.to_string());
    let (status, body) = match engine.health() {
        Ok(()) => (200, vec![("status".to_string(), s("ok"))]),
        Err(reason) => (
            503,
            vec![("status".to_string(), s("unavailable")), ("reason".to_string(), Json::Str(reason))],
        ),
    };
    json_response(status, &json::serialize(&Json::Object(body)))
}
