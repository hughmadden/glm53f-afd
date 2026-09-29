//! The API key. With one, every `/v1/*` request must carry it as `Authorization: Bearer <key>`,
//! else it is refused with a 401 in OpenAI's error shape (type `invalid_request_error`, code
//! `invalid_api_key`); `GET /health` and paths outside `/v1` need none. The key is compared in
//! constant time and never logged: the API logs no request, and [`ApiKey`]'s `Debug` shows
//! nothing of it.

use crate::http::Request;
use crate::types::ApiError;

/// The key a request must carry.
pub struct ApiKey(Vec<u8>);

impl ApiKey {
    /// A key of `key`, which must not be empty.
    pub fn new(key: &str) -> Result<ApiKey, String> {
        if key.is_empty() {
            return Err("the API key is empty".into());
        }
        Ok(ApiKey(key.as_bytes().to_vec()))
    }

    /// `Ok` when `req` is not for `/v1/...` or carries this key; else the 401 to answer with.
    pub fn check(&self, req: &Request) -> Result<(), ApiError> {
        // The path as the router reads it: without a query string.
        let path = req.path.split('?').next().unwrap_or("");
        if !path.starts_with("/v1") {
            return Ok(());
        }
        let sent = req.header("authorization").and_then(bearer);
        // Compared whatever was sent (nothing counts as empty), so the time it takes says
        // nothing of the key.
        if constant_time_eq(&self.0, sent.unwrap_or("").as_bytes()) {
            return Ok(());
        }
        Err(ApiError::unauthorized(match sent {
            None => "no API key: send `Authorization: Bearer <key>`",
            Some(_) => "the API key is not valid",
        }))
    }
}

impl std::fmt::Debug for ApiKey {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("ApiKey(<redacted>)")
    }
}

/// The token of an `Authorization: Bearer <token>` value (the scheme in any case).
fn bearer(value: &str) -> Option<&str> {
    let (scheme, token) = value.split_once(' ')?;
    scheme
        .eq_ignore_ascii_case("bearer")
        .then(|| token.trim_start())
}

/// `a == b`, in a time that depends on `a`'s length (the key's) only: not on where the two
/// differ, nor on `b`'s length.
fn constant_time_eq(a: &[u8], b: &[u8]) -> bool {
    let mut diff = a.len() ^ b.len();
    for (i, x) in a.iter().enumerate() {
        diff |= usize::from(x ^ b.get(i).copied().unwrap_or(0));
    }
    std::hint::black_box(diff) == 0
}

#[cfg(test)]
mod tests {
    use super::*;

    fn request(path: &str, authorization: Option<&str>) -> Request {
        Request {
            method: "GET".into(),
            path: path.into(),
            headers: authorization
                .map(|v| ("Authorization".to_string(), v.to_string()))
                .into_iter()
                .collect(),
            body: Vec::new(),
        }
    }

    #[test]
    fn keys_compare_whole() {
        assert!(constant_time_eq(b"key", b"key"));
        assert!(constant_time_eq(b"", b""));
        for other in ["", "ke", "keyy", "kez", "Key", "key ", "\0ey"] {
            assert!(!constant_time_eq(b"key", other.as_bytes()), "{other:?}");
        }
        // Bytes past the key's length count: the key with a tail is another key.
        assert!(!constant_time_eq(b"key", b"key\0"));
        assert!(!constant_time_eq(b"", b"key"));
    }

    #[test]
    fn the_bearer_token_is_read_with_the_scheme_in_any_case() {
        assert_eq!(bearer("Bearer abc"), Some("abc"));
        assert_eq!(bearer("bearer abc"), Some("abc"));
        assert_eq!(bearer("BEARER   abc"), Some("abc"));
        assert_eq!(bearer("Bearer a b"), Some("a b"));
        assert_eq!(bearer("Bearer "), Some(""));
        for other in ["Basic abc", "Bearerabc", "Bearer", "abc", ""] {
            assert_eq!(bearer(other), None, "{other:?}");
        }
    }

    #[test]
    fn only_v1_needs_the_key_and_only_the_right_key_passes() {
        let key = ApiKey::new("test-key-1").unwrap();
        for path in [
            "/v1/models",
            "/v1/models?x=1",
            "/v1/chat/completions",
            "/v1/nothing",
            "/v1",
        ] {
            assert!(
                key.check(&request(path, Some("Bearer test-key-1"))).is_ok(),
                "{path}"
            );
            assert!(
                key.check(&request(path, Some("bearer  test-key-1")))
                    .is_ok(),
                "{path}"
            );
            let e = key.check(&request(path, None)).unwrap_err();
            assert_eq!(
                (e.status, e.code.as_str()),
                (401, "invalid_api_key"),
                "{path}"
            );
            assert!(e.message.starts_with("no API key"), "{}", e.message);
            for wrong in [
                "Bearer test-key-2",
                "Bearer test-key-",
                "Bearer test-key-11",
                "Bearer",
                "Basic test-key-1",
                "test-key-1",
            ] {
                let e = key.check(&request(path, Some(wrong))).unwrap_err();
                assert_eq!(e.status, 401, "{path} {wrong}");
                let (want, not) = if wrong.starts_with("Bearer ") {
                    ("the API key is not valid", "no API key")
                } else {
                    ("no API key", "not valid")
                };
                assert!(
                    e.message.starts_with(want) && !e.message.contains(not),
                    "{wrong}: {}",
                    e.message
                );
            }
        }
        // Everything else is open, whatever it carries.
        for path in ["/health", "/health?probe=1", "/", "/nothing", "/metrics"] {
            assert!(key.check(&request(path, None)).is_ok(), "{path}");
            assert!(
                key.check(&request(path, Some("Bearer wrong"))).is_ok(),
                "{path}"
            );
        }
    }

    #[test]
    fn a_refusal_names_no_key_and_the_key_is_not_debug_printed() {
        let key = ApiKey::new("test-key-1").unwrap();
        for sent in [None, Some("Bearer test-key-2")] {
            let e = key.check(&request("/v1/models", sent)).unwrap_err();
            let body = crate::json::serialize(&e.body());
            assert!(!body.contains("test-key"), "{body}");
            assert!(
                body.contains(r#""type":"invalid_request_error""#)
                    && body.contains(r#""code":"invalid_api_key""#),
                "{body}"
            );
        }
        assert_eq!(format!("{key:?}"), "ApiKey(<redacted>)");
        assert!(ApiKey::new("").is_err());
    }
}
