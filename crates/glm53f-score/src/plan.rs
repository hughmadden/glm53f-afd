//! The plan `harness/klgate.py plan` writes (schema `glm53f-kl-plan.v1`), read and checked: the
//! vocabulary, every window's token ids against the tokenizer's bound and their digest, the rows
//! to write.

use glm53f_dsa::sha256::{sha256_hex, Sha256};
use glm53f_model::json::{self, Json};

/// The plan's schema.
pub const SCHEMA: &str = "glm53f-kl-plan.v1";

/// One window: the ids to feed and the rows to write.
#[derive(Clone, Debug, PartialEq)]
pub struct Window {
    pub window_id: String,
    pub tokens: Vec<u32>,
    /// sha256 of the ids as little-endian u32 (checked against `tokens`).
    pub tokens_sha256: String,
    /// Row r is the output at input position r, predicting token r + 1: ascending, distinct,
    /// below `tokens.len() - 1`.
    pub positions: Vec<usize>,
}

/// A checked plan.
#[derive(Clone, Debug, PartialEq)]
pub struct Plan {
    /// sha256 of the plan file.
    pub sha256: String,
    pub vocab: usize,
    /// The teacher panel's identity, as the plan records it.
    pub dataset_sha256: Option<String>,
    pub teacher_model_revision: Option<String>,
    pub windows: Vec<Window>,
}

/// sha256 of token ids as little-endian u32: the digest the plan and the outputs carry.
pub fn tokens_sha256(tokens: &[u32]) -> String {
    let mut h = Sha256::new();
    for t in tokens {
        h.update(&t.to_le_bytes());
    }
    h.finish().iter().map(|b| format!("{b:02x}")).collect()
}

/// A window id usable as a file name: letters, digits, `.`, `_`, `-`, not starting with `.`.
fn file_safe(id: &str) -> bool {
    !id.is_empty()
        && !id.starts_with('.')
        && id
            .bytes()
            .all(|c| c.is_ascii_alphanumeric() || matches!(c, b'.' | b'_' | b'-'))
}

fn uint(v: &Json, what: &str) -> Result<u64, String> {
    v.as_u64()
        .ok_or_else(|| format!("{what}: {v:?} is not a non-negative integer"))
}

impl Plan {
    /// Parse and check a plan file's bytes: `vocab` columns (the LM head's), every id below
    /// `id_bound` (the tokenizer's).
    pub fn parse(bytes: &[u8], vocab: usize, id_bound: usize) -> Result<Plan, String> {
        let v = json::parse_bytes(bytes).map_err(|e| format!("plan: {e}"))?;
        let s = |k: &str| v.get(k).and_then(Json::as_str).map(str::to_string);
        if s("schema").as_deref() != Some(SCHEMA) {
            return Err(format!(
                "plan: schema {:?}, expected {SCHEMA}",
                s("schema").unwrap_or_default()
            ));
        }
        let pv = uint(v.get("vocab").unwrap_or(&Json::Null), "plan: vocab")? as usize;
        if pv != vocab {
            return Err(format!(
                "plan: vocab {pv}, the LM head writes {vocab} columns"
            ));
        }
        let list = v
            .get("windows")
            .and_then(Json::as_array)
            .filter(|w| !w.is_empty())
            .ok_or("plan: no windows")?;
        let mut windows: Vec<Window> = Vec::with_capacity(list.len());
        for w in list {
            let id = w
                .get("window_id")
                .and_then(Json::as_str)
                .ok_or("plan: a window without a window_id")?
                .to_string();
            if !file_safe(&id) {
                return Err(format!("plan: window id {id:?} is not a plain file name"));
            }
            if windows.iter().any(|x| x.window_id == id) {
                return Err(format!("plan: window {id} twice"));
            }
            let arr = |k: &str| {
                w.get(k)
                    .and_then(Json::as_array)
                    .ok_or_else(|| format!("plan: {id}: no {k}"))
            };
            let mut tokens = Vec::new();
            for t in arr("tokens")? {
                let t = uint(t, &format!("plan: {id}: a token id"))?;
                if t >= id_bound as u64 {
                    return Err(format!(
                        "plan: {id}: token id {t} is not below {id_bound} (the tokenizer's ids)"
                    ));
                }
                tokens.push(t as u32);
            }
            if tokens.len() < 2 {
                return Err(format!("plan: {id}: {} tokens", tokens.len()));
            }
            let want = w
                .get("tokens_sha256")
                .and_then(Json::as_str)
                .ok_or_else(|| format!("plan: {id}: no tokens_sha256"))?;
            let have = tokens_sha256(&tokens);
            if have != want {
                return Err(format!(
                    "plan: {id}: the ids' sha256 is {have}, the plan says {want}"
                ));
            }
            let mut positions = Vec::new();
            for p in arr("positions")? {
                positions.push(uint(p, &format!("plan: {id}: a position"))? as usize);
            }
            if positions.is_empty()
                || positions.windows(2).any(|p| p[0] >= p[1])
                || *positions.last().unwrap() + 1 >= tokens.len()
            {
                return Err(format!(
                    "plan: {id}: the positions must be distinct, ascending rows below {} (a row \
                     predicts the token after it)",
                    tokens.len() - 1
                ));
            }
            windows.push(Window {
                window_id: id,
                tokens,
                tokens_sha256: have,
                positions,
            });
        }
        Ok(Plan {
            sha256: sha256_hex(bytes),
            vocab,
            dataset_sha256: s("dataset_sha256"),
            teacher_model_revision: s("teacher_model_revision"),
            windows,
        })
    }

    /// The windows named by `ids` (in the plan's order), or every window.
    pub fn select(&self, ids: Option<&[String]>) -> Result<Vec<&Window>, String> {
        let Some(ids) = ids else {
            return Ok(self.windows.iter().collect());
        };
        if let Some(bad) = ids
            .iter()
            .find(|i| !self.windows.iter().any(|w| &w.window_id == *i))
        {
            return Err(format!("--windows: the plan has no window {bad}"));
        }
        Ok(self
            .windows
            .iter()
            .filter(|w| ids.contains(&w.window_id))
            .collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn plan_text(windows: &[(&str, &[u32], &[usize])], vocab: usize) -> String {
        let ws: Vec<String> = windows
            .iter()
            .map(|(id, t, p)| {
                format!(
                    r#"{{"window_id":"{id}","tokens_sha256":"{}","tokens":{t:?},"positions":{p:?}}}"#,
                    tokens_sha256(t)
                )
            })
            .collect();
        format!(
            r#"{{"schema":"{SCHEMA}","dataset_sha256":"d","teacher_model_revision":"r","vocab":{vocab},"windows":[{}]}}"#,
            ws.join(",")
        )
    }

    #[test]
    fn the_digest_is_klgates() {
        // klgate.py: hashlib.sha256(array("I", tokens).tobytes()) on a little-endian host.
        assert_eq!(
            tokens_sha256(&[]),
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
        let mut bytes = Vec::new();
        for t in [1u32, 154_855, 70_000] {
            bytes.extend_from_slice(&t.to_le_bytes());
        }
        assert_eq!(tokens_sha256(&[1, 154_855, 70_000]), sha256_hex(&bytes));
    }

    #[test]
    fn a_good_plan() {
        let text = plan_text(
            &[
                ("final-0000", &[5, 6, 7, 8], &[0, 2]),
                ("final-0001", &[9, 10, 11], &[1]),
            ],
            100,
        );
        let p = Plan::parse(text.as_bytes(), 100, 50).unwrap();
        assert_eq!(p.sha256, sha256_hex(text.as_bytes()));
        assert_eq!(p.windows.len(), 2);
        assert_eq!(p.windows[0].tokens, vec![5, 6, 7, 8]);
        assert_eq!(p.windows[0].positions, vec![0, 2]);
        assert_eq!(p.dataset_sha256.as_deref(), Some("d"));
        assert_eq!(p.select(None).unwrap().len(), 2);
        let sel = p.select(Some(&["final-0001".to_string()])).unwrap();
        assert_eq!(sel[0].window_id, "final-0001");
        assert!(p.select(Some(&["final-0009".to_string()])).is_err());
    }

    /// What `klgate.py plan` writes for a panel of several roles: the other roles' windows (ids
    /// with hyphens), each window's role, and the panel's identity beside the plan's own fields.
    #[test]
    fn a_plan_of_a_larger_panel() {
        let windows: [(&str, &[u32], &[usize]); 3] = [
            ("final-0024", &[5, 6, 7, 8], &[0, 2]),
            ("confirmation-0000", &[9, 10, 11], &[1]),
            ("conditional-fit-0012", &[12, 13, 14, 15], &[0, 1, 2]),
        ];
        let ws: Vec<String> = windows
            .iter()
            .map(|(id, t, p)| {
                let role = id.rsplit_once('-').unwrap().0;
                format!(
                    r#"{{"window_id":"{id}","role":"{role}","tokens_sha256":"{}","token_ids_npy_sha256":"{}","tokens":{t:?},"positions":{p:?}}}"#,
                    tokens_sha256(t),
                    "a".repeat(64)
                )
            })
            .collect();
        let text = format!(
            r#"{{"schema":"{SCHEMA}","row_semantics":"row r = logits after tokens[0..r]","dataset_sha256":"d","dataset_manifest_file_sha256":"m","full_panel_manifest_sha256":"f","full_panel_manifest_file_sha256":"g","teacher_model_revision":"r","panel":{{"windows":3,"window_ids_sha256":"i","roles":{{"final":1,"confirmation":1,"conditional-fit":1}}}},"vocab":100,"windows":[{}]}}"#,
            ws.join(",")
        );
        let p = Plan::parse(text.as_bytes(), 100, 50).unwrap();
        let ids: Vec<&str> = p.windows.iter().map(|w| w.window_id.as_str()).collect();
        assert_eq!(
            ids,
            ["final-0024", "confirmation-0000", "conditional-fit-0012"]
        );
        assert_eq!(p.dataset_sha256.as_deref(), Some("d"));
        assert_eq!(p.windows[2].positions, vec![0, 1, 2]);
    }

    #[test]
    fn bad_plans_are_refused() {
        let good: &[u32] = &[5, 6, 7, 8];
        let cases: Vec<(String, &str)> = vec![
            (plan_text(&[("w", good, &[0])], 99), "vocab"),
            (plan_text(&[("w", &[5, 60, 7], &[0])], 100), "id bound"),
            (plan_text(&[("w", good, &[2, 1])], 100), "descending"),
            (plan_text(&[("w", good, &[1, 1])], 100), "repeated"),
            (plan_text(&[("w", good, &[3])], 100), "the last token's row"),
            (plan_text(&[("w", good, &[])], 100), "no positions"),
            (plan_text(&[("w", &[5], &[0])], 100), "one token"),
            (plan_text(&[("../w", good, &[0])], 100), "a path"),
            (
                plan_text(&[("w", good, &[0]), ("w", good, &[0])], 100),
                "twice",
            ),
            (plan_text(&[], 100), "no windows"),
            (
                plan_text(&[("w", good, &[0])], 100).replace(SCHEMA, "other"),
                "schema",
            ),
            (
                plan_text(&[("w", good, &[0])], 100).replace(&tokens_sha256(good), &"0".repeat(64)),
                "digest",
            ),
            ("{".to_string(), "json"),
        ];
        for (text, what) in cases {
            assert!(
                Plan::parse(text.as_bytes(), 100, 50).is_err(),
                "{what}: accepted"
            );
        }
    }
}
