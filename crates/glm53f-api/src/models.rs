//! `GET /v1/models` — the model list (A8 readiness; id `glm-5.3-flash`).

use crate::http::{json_response, Response};
use crate::json::{self, Json};
use crate::types::MODEL_ID;

pub fn handle() -> Response {
    let body = json::serialize(&Json::Object(vec![
        ("object".to_string(), Json::Str("list".to_string())),
        ("data".to_string(), Json::Array(vec![Json::Object(vec![
            ("id".to_string(), Json::Str(MODEL_ID.to_string())),
            ("object".to_string(), Json::Str("model".to_string())),
        ])])),
    ]));
    json_response(200, &body)
}
