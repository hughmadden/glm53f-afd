//! The MiMo tool-call markup parser (COHERENCE-TRAPS T27/T29, tool-cap T24),
//! and [`MimoDialect`], the reference [`Dialect`] for MiMo-V2.6-Flash.
//!
//! The model emits tool calls in a Qwen3-style XML markup with no escaping:
//! `<think>reasoning</think>` and
//! `<tool_call><function=NAME><parameter=K>V</parameter>…</function></tool_call>`.
//! This module parses a completion string into reasoning, content and a list of
//! calls, applying the vLLM `mimo` semantics pinned by ADVISOR-I3 §10.2.4:
//!   - split on the FIRST closing `</parameter>` (no escaping), and report the
//!     unparsed tail rather than silently dropping it (T29);
//!   - `im_end` inside an argument is plain text, not end-of-message (T29);
//!   - a call missing its closing `</tool_call>` is LOST and reported (T27);
//!   - a call with no `function=` name is an ERROR, never a persisted nameless
//!     call (T24/T29);
//!   - argument values lose one leading/trailing newline and are coerced to the
//!     tool schema recursively (T27);
//!   - the tool-call cap stops generation at call `cap+1` (T24).
//!
//! Every tag literal is built from `\u{3c}`/`\u{3e}` escapes so no literal
//! `<tool_call>` appears in this source (T29 hygiene for readers).

use super::{Dialect, ParseResult, ParsedCall, StreamTags};
use crate::json::{self, Json};
use crate::types::Tool;

const GT: &str = "\u{3e}";

const TC: &str = "\u{3c}tool_call\u{3e}";
const TC_END: &str = "\u{3c}\u{2f}tool_call\u{3e}";
const FN: &str = "\u{3c}function\u{3d}";
const FN_END: &str = "\u{3c}\u{2f}function\u{3e}";
const PA: &str = "\u{3c}parameter\u{3d}";
const PA_END: &str = "\u{3c}\u{2f}parameter\u{3e}";
const TH: &str = "\u{3c}think\u{3e}";
const TH_END: &str = "\u{3c}\u{2f}think\u{3e}";

/// Default tool-call cap (T24).
pub const TOOL_CALL_CAP: usize = 6;

/// MiMo-V2.6-Flash's completion markup: `<think>` blocks and Qwen3-style XML
/// tool calls. Its template leaves the opening think tag to the model (with
/// thinking off, the prompt closes an empty block), so a completion never starts
/// inside reasoning and `thinking` does not change the parse.
#[derive(Debug, Clone, Copy, Default)]
pub struct MimoDialect;

impl Dialect for MimoDialect {
    fn parse(&self, text: &str, tools: &[Tool], _thinking: bool, cap: usize) -> ParseResult {
        parse(text, tools, cap)
    }

    fn stream_tags(&self) -> StreamTags {
        StreamTags { think_open: TH, think_close: TH_END, tool_open: TC }
    }
}

/// Parse a completion string. `tools` supplies schema types for coercion (may be
/// empty — then values are coerced by the `json-or-raw` rule only).
pub fn parse(text: &str, tools: &[Tool], cap: usize) -> ParseResult {
    let mut r = ParseResult::default();
    let mut content = String::new();
    let mut i = 0usize;
    let n = text.len();
    while i < n {
        let rest = &text[i..];
        if rest.starts_with(TH) {
            // A think block: reasoning until the first closing tag.
            if let Some(j) = rest.find(TH_END) {
                r.reasoning.push(rest[TH.len()..j].to_string());
                i += j + TH_END.len();
            } else {
                // Unclosed think: the remainder is reasoning.
                r.reasoning.push(rest[TH.len()..].to_string());
                i = n;
            }
        } else if rest.starts_with(TC) {
            // A tool-call block; the cap stops generation before call cap+1.
            if cap != 0 && r.calls.len() >= cap {
                r.capped = true;
                break;
            }
            let (call, consumed, reports, err) = parse_call(&rest[TC.len()..], tools);
            r.reports.extend(reports);
            if let Some(e) = err {
                r.error = Some(e);
                r.calls.clear();
                return r;
            }
            if let Some(c) = call {
                r.calls.push(c);
            }
            i += TC.len() + consumed;
        } else {
            // Plain content: advance one char.
            let ch = rest.chars().next().unwrap();
            content.push(ch);
            i += ch.len_utf8();
        }
    }
    r.content = content;
    r
}

/// Parse one `<tool_call>` block, given the text after `<tool_call>`.
/// Returns (call, bytes-consumed, reports, error).
fn parse_call(s: &str, tools: &[Tool]) -> (Option<ParsedCall>, usize, Vec<String>, Option<String>) {
    let mut reports = Vec::new();
    // Find the function name, if any.
    let (name, after_fn) = match s.find(FN) {
        Some(p) => {
            let name_start = p + FN.len();
            let name_end = match s[name_start..].find(GT) {
                Some(q) => name_start + q,
                None => {
                    reports.push("lost call (closing tag missing)".to_string());
                    return (None, s.len(), reports, None);
                }
            };
            let name = s[name_start..name_end].to_string();
            (Some(name), name_end + GT.len())
        }
        None => {
            // No <function= . If there are parameters, this is a nameless call.
            if s.contains(PA) {
                return (None, s.len(), Vec::new(), Some("nameless tool call".to_string()));
            }
            // An empty <tool_call></tool_call> — no call.
            return (None, s.len(), Vec::new(), None);
        }
    };
    let name = match name {
        Some(nm) if !nm.is_empty() => nm,
        _ => return (None, s.len(), Vec::new(), Some("nameless tool call".to_string())),
    };
    let tool = tools.iter().find(|t| t.function.name == name);

    // Parse <parameter=K>V</parameter> pairs until </function> or end.
    let mut args: Vec<(String, Json)> = Vec::new();
    let mut cursor = after_fn;
    loop {
        let rest = &s[cursor..];
        if rest.starts_with(FN_END) {
            cursor += FN_END.len();
            break;
        }
        if rest.starts_with(TC_END) || rest.is_empty() {
            break;
        }
        if rest.starts_with(PA) {
            // A <parameter=K> opener; the value runs to the FIRST </parameter>.
            let key_start = PA.len();
            let key_end = match rest[key_start..].find(GT) {
                Some(q) => key_start + q,
                None => {
                    reports.push("unparsed tail after the embedded closing tag".to_string());
                    cursor += rest.len();
                    break;
                }
            };
            let key = rest[key_start..key_end].to_string();
            let val_start = key_end + GT.len();
            let tail = &rest[val_start..];
            match tail.find(PA_END) {
                Some(q) => {
                    let raw = &tail[..q];
                    args.push((key.clone(), coerce(raw, tool, &key)));
                    cursor += val_start + q + PA_END.len();
                }
                None => {
                    reports.push("lost call (closing tag missing)".to_string());
                    return (None, s.len(), reports, None);
                }
            }
        } else if rest.starts_with(PA_END) {
            // A stray closing parameter tag (the embedded-tag tail): skip it.
            cursor += PA_END.len();
        } else {
            // Stray text after a parameter's first closing tag: report it, then
            // resync at the next tag.
            reports.push("unparsed tail after the embedded closing tag".to_string());
            match rest.find('\u{3c}') {
                Some(rel) => cursor += rel,
                None => cursor += rest.len(),
            }
        }
    }

    // A well-formed call needs its closing </tool_call>; without it, it is lost.
    if !s[cursor..].starts_with(TC_END) {
        reports.push("lost call (closing tag missing)".to_string());
        return (None, s.len(), reports, None);
    }
    cursor += TC_END.len();

    let arguments = Json::Object(args);
    (Some(ParsedCall { name, arguments }), cursor, reports, None)
}

/// Coerce a raw argument value to its schema type (T27). Without a schema the
/// value is parsed as JSON when it is a well-formed literal, else kept as text.
fn coerce(raw: &str, tool: Option<&Tool>, key: &str) -> Json {
    // Lose one leading and one trailing newline.
    let raw = raw.strip_prefix('\n').unwrap_or(raw);
    let raw = raw.strip_suffix('\n').unwrap_or(raw);
    let schema = tool.and_then(|t| t.property_schema(key));
    // json-or-raw: a well-formed literal parses; anything else stays text.
    if let Ok(v) = json::parse(raw) {
        return match schema.as_deref() {
            Some("integer") if matches!(v, Json::Str(_)) => {
                if let Ok(n) = raw.parse::<f64>() { Json::Num(n) } else { v }
            }
            Some("string") if matches!(v, Json::Num(_)) => {
                Json::Str(json::serialize(&v))
            }
            _ => v,
        };
    }
    // Not a literal: a plain string, unless the schema asks for a number.
    match schema.as_deref() {
        Some("integer") | Some("number") => {
            if let Ok(n) = raw.parse::<f64>() { Json::Num(n) } else { Json::Str(raw.to_string()) }
        }
        _ => Json::Str(raw.to_string()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn call(name: &str, pairs: &[(&str, Json)]) -> ParsedCall {
        ParsedCall {
            name: name.to_string(),
            arguments: Json::Object(pairs.iter().map(|(k, v)| (k.to_string(), v.clone())).collect()),
        }
    }
    fn num(v: f64) -> Json {
        Json::Num(v)
    }
    fn str_(v: &str) -> Json {
        Json::Str(v.to_string())
    }

    fn parse_all(text: &str) -> ParseResult {
        parse(text, &[], TOOL_CALL_CAP)
    }

    #[test]
    fn simple_two_params_schema_coercion() {
        let out = format!("{TC}\u{3c}function=fetch{GT}{PA}url{GT}https://x/y{PA_END}{PA}retries{GT}5{PA_END}{FN_END}{TC_END}");
        let r = parse_all(&out);
        assert_eq!(r.error, None);
        assert_eq!(r.calls, vec![call("fetch", &[("url", str_("https://x/y")), ("retries", num(5.0))])]);
    }

    #[test]
    fn value_contains_closing_parameter_tag() {
        let out = format!("{TC}\u{3c}function=echo{GT}{PA}v{GT}abc{PA_END}def{PA_END}{FN_END}{TC_END}");
        let r = parse_all(&out);
        assert_eq!(r.error, None);
        assert_eq!(r.calls, vec![call("echo", &[("v", str_("abc"))])]);
        assert!(r.reports.iter().any(|x| x.contains("unparsed tail")), "{:?}", r.reports);
    }

    #[test]
    fn value_contains_chat_eos_special_token() {
        let eos = "\u{3c}\u{7c}im_end\u{7c}\u{3e}";
        let val = format!("abc{eos}def");
        let out = format!("{TC}\u{3c}function=echo{GT}{PA}v{GT}{val}{PA_END}{FN_END}{TC_END}");
        let r = parse_all(&out);
        assert_eq!(r.error, None);
        assert_eq!(r.calls, vec![call("echo", &[("v", str_(&val))])]);
    }

    #[test]
    fn missing_closing_tool_call_tag_is_lost() {
        let out = format!("{TC}\u{3c}function=fetch{GT}{PA}url{GT}https://x/y{PA_END}{FN_END}");
        let r = parse_all(&out);
        assert_eq!(r.error, None);
        assert_eq!(r.calls, vec![]);
        assert!(r.reports.iter().any(|x| x.contains("lost call")), "{:?}", r.reports);
    }

    #[test]
    fn nameless_call_is_an_error() {
        let out = format!("{TC}{PA}url{GT}x{PA_END}{TC_END}");
        let r = parse_all(&out);
        assert_eq!(r.error, Some("nameless tool call".to_string()));
        assert_eq!(r.calls, vec![]);
    }

    #[test]
    fn non_string_renders_with_tojson() {
        let quoted = format!("\u{22}5\u{22}"); // "5"
        let out = format!("{TC}\u{3c}function=set{GT}{PA}n{GT}5{PA_END}{PA}s{GT}{quoted}{PA_END}{FN_END}{TC_END}");
        let r = parse_all(&out);
        assert_eq!(r.calls, vec![call("set", &[("n", num(5.0)), ("s", str_("5"))])]);
    }

    #[test]
    fn think_block_then_call() {
        let out = format!("{TH}reasoning{TH_END}{TC}\u{3c}function=fetch{GT}{PA}url{GT}https://x/y{PA_END}{FN_END}{TC_END}");
        let r = parse_all(&out);
        assert_eq!(r.reasoning, vec!["reasoning".to_string()]);
        assert_eq!(r.calls, vec![call("fetch", &[("url", str_("https://x/y"))])]);
    }

    #[test]
    fn two_calls_one_message() {
        let out = format!(
            "{TC}\u{3c}function=a{GT}{PA}k{GT}1{PA_END}{FN_END}{TC_END}{TC}\u{3c}function=b{GT}{PA}k{GT}2{PA_END}{FN_END}{TC_END}"
        );
        let r = parse_all(&out);
        assert_eq!(r.calls, vec![call("a", &[("k", num(1.0))]), call("b", &[("k", num(2.0))])]);
    }

    #[test]
    fn cap_stops_before_call_cap_plus_one() {
        let mut out = String::new();
        for k in 0..8 {
            out.push_str(&format!("{TC}\u{3c}function=t{k}{GT}{FN_END}{TC_END}"));
        }
        let r = parse(&out, &[], 6);
        assert!(r.capped);
        assert_eq!(r.calls.len(), 6);
    }
}
