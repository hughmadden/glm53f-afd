//! GLM-5.3-Flash's completion markup, and [`GlmDialect`].
//!
//! GLM-5.3-Flash's chat template (the `glm53f-tokenizer` crate's `template` module) fixes what
//! the model writes:
//!
//! - **Reasoning.** With thinking on, the prompt ends by opening the think block, so a
//!   completion starts inside it: the text up to the first &lt;/think&gt; is reasoning
//!   ([`GlmDialect::reasoning_first`] is the thinking switch). With thinking off the prompt ends
//!   with an empty think block and the completion starts in content. A think block the model
//!   opens later is reasoning too, up to its first closing tag; an unclosed block runs to the end.
//! - **Tool calls.** &lt;tool_call&gt;NAME, then per argument
//!   &lt;arg_key&gt;KEY&lt;/arg_key&gt;&lt;arg_value&gt;VALUE&lt;/arg_value&gt;, then
//!   &lt;/tool_call&gt;. Calls follow each other directly. Nothing is escaped: the template
//!   writes a string argument as is and any other value as JSON.
//!
//! [`parse`] reads a completion the way the API's streaming splitter walks it, so streamed and
//! whole responses agree. When the prompt opened the reasoning block, reasoning comes first;
//! then content, think blocks and tool calls in order. Tags inside reasoning or inside an
//! argument value are text. Its rules:
//!
//! - **Argument values.** A value runs to the first &lt;/arg_value&gt;. Its type comes from the
//!   tool's JSON schema, by inverting the template:
//!   - a property declared only as `string` keeps the value text exactly, with no trimming;
//!   - otherwise the text is parsed as JSON. It is kept as that value when it is a number,
//!     boolean, null, array or object the schema allows (any type, without a schema), and as
//!     text when it is not;
//!   - a JSON string literal stays text, quotes included, because the template never writes a
//!     string as JSON.
//! - **Whitespace** around the name, between arguments and before the closing tag is ignored.
//! - **Nothing is dropped silently.** Every loss goes into `reports`:
//!   - a call that ends before its closing tag is lost;
//!   - a new call opening inside one loses the first call, and parsing resumes at the new call;
//!   - stray text inside a call, and an argument without a value, are dropped from that call;
//!   - of two values for one key, the first is dropped;
//!   - a call with markup in its name is lost;
//!   - an empty call is noted.
//!
//!   A call with arguments but no name is an `error`, never a nameless call, as in the MiMo
//!   reference dialect.
//! - **The tool-call cap:** with `cap` > 0, parsing stops before call `cap + 1` and sets `capped`.
//!
//! Every tag literal here is assembled from pieces, so tools that scan sources for markup do not
//! misread this file.

use super::{Dialect, ParseResult, ParsedCall, StreamTags};
use crate::json::{self, Json};
use crate::types::Tool;

const TH: &str = concat!("<", "think", ">");
const TH_END: &str = concat!("<", "/think", ">");
const TC: &str = concat!("<", "tool_call", ">");
const TC_END: &str = concat!("<", "/tool_call", ">");
const AK: &str = concat!("<", "arg_key", ">");
const AK_END: &str = concat!("<", "/arg_key", ">");
const AV: &str = concat!("<", "arg_value", ">");
const AV_END: &str = concat!("<", "/arg_value", ">");

/// GLM-5.3-Flash's completion markup (see the module doc). Its chat template always thinks
/// unless a request turns thinking off, so thinking is on by default.
#[derive(Debug, Clone, Copy, Default)]
pub struct GlmDialect;

impl Dialect for GlmDialect {
    fn parse(&self, text: &str, tools: &[Tool], thinking: bool, cap: usize) -> ParseResult {
        parse(text, tools, thinking, cap)
    }

    fn stream_tags(&self) -> StreamTags {
        StreamTags { think_open: TH, think_close: TH_END, tool_open: TC }
    }

    /// The prompt opens the think block exactly when thinking is on.
    fn reasoning_first(&self, thinking: bool) -> bool {
        thinking
    }

    fn default_thinking(&self) -> bool {
        true
    }
}

/// Split a completion into content, reasoning blocks and tool calls. `thinking`: the prompt
/// opened the think block. `cap` > 0 stops before call `cap + 1`.
pub fn parse(text: &str, tools: &[Tool], thinking: bool, cap: usize) -> ParseResult {
    let mut r = ParseResult::default();
    let mut rest = text;
    if thinking {
        rest = take_reasoning(rest, &mut r);
    }
    while !rest.is_empty() {
        match (rest.find(TH), rest.find(TC)) {
            (Some(p), c) if c.is_none_or(|c| p < c) => {
                r.content.push_str(&rest[..p]);
                rest = take_reasoning(&rest[p + TH.len()..], &mut r);
            }
            (_, Some(p)) => {
                r.content.push_str(&rest[..p]);
                if cap != 0 && r.calls.len() >= cap {
                    r.capped = true;
                    break;
                }
                let body = &rest[p + TC.len()..];
                match parse_call(body, tools, &mut r.reports) {
                    Call::Parsed(call, used) => {
                        r.calls.push(call);
                        rest = &body[used..];
                    }
                    Call::Lost(used) => rest = &body[used..],
                    Call::Nameless => {
                        r.error = Some("nameless tool call".to_string());
                        r.calls.clear();
                        return r;
                    }
                }
            }
            _ => {
                r.content.push_str(rest);
                break;
            }
        }
    }
    r
}

/// One reasoning block from `s` (after its opening): up to the first closing tag, or all of it.
/// Returns the text after the block.
fn take_reasoning<'a>(s: &'a str, r: &mut ParseResult) -> &'a str {
    match s.find(TH_END) {
        Some(q) => {
            r.reasoning.push(s[..q].to_string());
            &s[q + TH_END.len()..]
        }
        None => {
            r.reasoning.push(s.to_string());
            ""
        }
    }
}

enum Call {
    /// A call, and the bytes it used.
    Parsed(ParsedCall, usize),
    /// No call (reported); resume after the bytes used.
    Lost(usize),
    /// Arguments without a name: a fatal error.
    Nameless,
}

/// The earliest of `tags` in `s`: (position, tag).
fn first_of(s: &str, tags: &[&'static str]) -> Option<(usize, &'static str)> {
    tags.iter().filter_map(|&t| s.find(t).map(|p| (p, t))).min_by_key(|&(p, _)| p)
}

/// A short excerpt of lost text for a report.
fn excerpt(s: &str) -> String {
    const MAX: usize = 80;
    match s.char_indices().nth(MAX) {
        Some((cut, _)) => format!("{:?}...", &s[..cut]),
        None => format!("{s:?}"),
    }
}

/// Where a malformed call ends: after its closing tag, or at a new call (resume there), or at
/// the end.
fn skip_call(s: &str, from: usize) -> usize {
    match first_of(&s[from..], &[TC_END, TC]) {
        Some((p, t)) if t == TC_END => from + p + TC_END.len(),
        Some((p, _)) => from + p,
        None => s.len(),
    }
}

/// Parse one call from `s`, the text after its opening tag.
fn parse_call(s: &str, tools: &[Tool], reports: &mut Vec<String>) -> Call {
    // The name runs to the first argument or the closing tag; a new call first means this one
    // never closed.
    let Some((end, tag)) = first_of(s, &[AK, TC_END, TC]) else {
        reports.push(format!("lost call (closing tag missing): {}", excerpt(s)));
        return Call::Lost(s.len());
    };
    if tag == TC {
        reports.push(format!("lost call (closing tag missing before the next call): {}", excerpt(&s[..end])));
        return Call::Lost(end);
    }
    let name = s[..end].trim();
    if name.is_empty() {
        if tag == AK {
            return Call::Nameless;
        }
        reports.push("empty tool call".to_string());
        return Call::Lost(end + TC_END.len());
    }
    if name.contains(['<', '>']) {
        let used = skip_call(s, end);
        reports.push(format!("lost call (markup in the name): {}", excerpt(&s[..used])));
        return Call::Lost(used);
    }
    let tool = tools.iter().find(|t| t.function.name == name);
    let mut args: Vec<(String, Json)> = Vec::new();
    let mut i = end;
    loop {
        let rest = s[i..].trim_start();
        i = s.len() - rest.len();
        if let Some(after) = rest.strip_prefix(TC_END) {
            let used = s.len() - after.len();
            return Call::Parsed(ParsedCall { name: name.to_string(), arguments: Json::Object(args) }, used);
        }
        if rest.is_empty() {
            reports.push(format!("lost call {name:?} (closing tag missing)"));
            return Call::Lost(s.len());
        }
        if rest.starts_with(TC) {
            reports.push(format!("lost call {name:?} (closing tag missing before the next call)"));
            return Call::Lost(i);
        }
        if let Some(k) = rest.strip_prefix(AK) {
            let Some(ke) = k.find(AK_END) else {
                reports.push(format!("lost call {name:?} (argument key not closed): {}", excerpt(k)));
                return Call::Lost(s.len());
            };
            let key = k[..ke].trim();
            let after_key = &k[ke + AK_END.len()..];
            if key.contains(['<', '>']) {
                reports.push(format!("argument with markup in its key dropped from {name:?}: {}", excerpt(key)));
                i = s.len() - after_key.len();
                continue;
            }
            let Some(v) = after_key.trim_start().strip_prefix(AV) else {
                reports.push(format!("argument {key:?} of {name:?} has no value; dropped"));
                i = s.len() - after_key.len();
                continue;
            };
            let Some(ve) = v.find(AV_END) else {
                reports.push(format!("lost call {name:?} (value of {key:?} not closed): {}", excerpt(v)));
                return Call::Lost(s.len());
            };
            let value = coerce(&v[..ve], tool.and_then(|t| property(t, key)));
            match args.iter_mut().find(|(k, _)| k == key) {
                Some(slot) => {
                    reports.push(format!("duplicate argument {key:?} in {name:?}: the first value was dropped"));
                    slot.1 = value;
                }
                None => args.push((key.to_string(), value)),
            }
            i = s.len() - v[ve + AV_END.len()..].len();
            continue;
        }
        // Stray text, such as the rest of a value that held its own closing tag: report it and
        // resume at the next argument, closing tag or call.
        let skip = first_of(rest, &[AK, TC_END, TC]).map_or(rest.len(), |(p, _)| p);
        reports.push(format!("unparsed text in tool call {name:?}: {}", excerpt(&rest[..skip])));
        i += skip;
    }
}

/// A property's schema in a tool's parameters.
fn property<'a>(tool: &'a Tool, key: &str) -> Option<&'a Json> {
    tool.function.parameters.as_ref()?.get("properties")?.get(key)
}

fn json_type(v: &Json) -> &'static str {
    match v {
        Json::Null => "null",
        Json::Bool(_) => "boolean",
        Json::Num(_) => "number",
        Json::Str(_) => "string",
        Json::Array(_) => "array",
        Json::Object(_) => "object",
    }
}

/// The JSON types a schema allows: its `type` (a name or a list), the types of its `anyOf` and
/// `oneOf` alternatives, and, without a `type`, the types of its `enum` or `const` values.
/// Empty when the schema does not say.
fn schema_types(schema: &Json, out: &mut Vec<String>, depth: usize) {
    if depth > 16 {
        return;
    }
    match schema.get("type") {
        Some(Json::Str(t)) => out.push(t.clone()),
        Some(Json::Array(ts)) => out.extend(ts.iter().filter_map(|t| t.as_str()).map(str::to_string)),
        _ => {
            let values = schema.get("enum").and_then(|e| e.as_array()).unwrap_or(&[]);
            out.extend(values.iter().chain(schema.get("const")).map(|v| json_type(v).to_string()));
        }
    }
    for key in ["anyOf", "oneOf"] {
        for alt in schema.get(key).and_then(|a| a.as_array()).unwrap_or(&[]) {
            schema_types(alt, out, depth + 1);
        }
    }
}

fn allows(types: &[String], v: &Json) -> bool {
    let has = |t: &str| types.iter().any(|x| x == t);
    match v {
        Json::Num(n) => has("number") || (has("integer") && n.fract() == 0.0),
        other => has(json_type(other)),
    }
}

/// An argument value's text read back to the type the template wrote it from.
fn coerce(raw: &str, schema: Option<&Json>) -> Json {
    let mut types = Vec::new();
    if let Some(s) = schema {
        schema_types(s, &mut types, 0);
    }
    if !types.is_empty() && types.iter().all(|t| t == "string") {
        return Json::Str(raw.to_string());
    }
    match json::parse(raw.trim()) {
        Ok(v) if !matches!(v, Json::Str(_)) && (types.is_empty() || allows(&types, &v)) => v,
        _ => Json::Str(raw.to_string()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::{Engine, GenerateOutcome, GenerateParams, PromptOptions};
    use crate::types::{ChatMessage, ChatRequest};
    use std::sync::{Arc, Mutex};

    fn tool(name: &str, params: &str) -> Tool {
        let raw = json::parse(&format!(r#"{{"type":"function","function":{{"name":"{name}","parameters":{params}}}}}"#)).unwrap();
        let f = raw.get("function").unwrap();
        Tool {
            r#type: "function".into(),
            function: crate::types::ToolFunction {
                name: name.into(),
                description: None,
                parameters: f.get("parameters").cloned(),
            },
            raw: raw.clone(),
        }
    }

    fn weather() -> Tool {
        tool(
            "get_weather",
            r#"{"type":"object","properties":{"city":{"type":"string"},"days":{"type":"integer"},"metric":{"type":"boolean"},"ratio":{"type":"number"},"note":{"type":["string","null"]},"where":{"type":"object"},"tags":{"type":"array"},"mode":{"enum":["fast","slow"]}}}"#,
        )
    }

    /// A tool call in the template's form. Values are written as the template writes them: a
    /// string as is, anything else as JSON.
    fn call_text(name: &str, args: &[(&str, &str)]) -> String {
        let mut s = format!("{TC}{name}");
        for (k, v) in args {
            s.push_str(&format!("{AK}{k}{AK_END}{AV}{v}{AV_END}"));
        }
        s.push_str(TC_END);
        s
    }

    fn obj(pairs: &[(&str, Json)]) -> Json {
        Json::Object(pairs.iter().map(|(k, v)| (k.to_string(), v.clone())).collect())
    }

    fn st(s: &str) -> Json {
        Json::Str(s.to_string())
    }

    fn calls(r: &ParseResult) -> Vec<(String, String)> {
        r.calls.iter().map(|c| (c.name.clone(), json::serialize(&c.arguments))).collect()
    }

    #[test]
    fn reasoning_then_content() {
        let r = parse(&format!("Let me think.{TH_END}The answer."), &[], true, 0);
        assert_eq!(r.reasoning, ["Let me think."]);
        assert_eq!(r.content, "The answer.");
        assert!(r.calls.is_empty() && r.reports.is_empty() && r.error.is_none());
        // Unclosed: all reasoning, no content (the budget ran out while thinking).
        let r = parse("still thinking", &[], true, 0);
        assert_eq!(r.reasoning, ["still thinking"]);
        assert_eq!(r.content, "");
        // An immediately closed block is an empty reasoning block.
        let r = parse(&format!("{TH_END}Hi"), &[], true, 0);
        assert_eq!((r.reasoning.as_slice(), r.content.as_str()), (&[String::new()][..], "Hi"));
    }

    #[test]
    fn content_without_reasoning_when_thinking_is_off() {
        let r = parse("Plain answer.", &[], false, 0);
        assert!(r.reasoning.is_empty());
        assert_eq!(r.content, "Plain answer.");
        // A stray closing tag is content: only the prompt could have opened a block.
        let text = format!("a{TH_END}b");
        let r = parse(&text, &[], false, 0);
        assert_eq!((r.content.as_str(), r.reasoning.len()), (text.as_str(), 0));
        // A block the model opens itself is reasoning.
        let r = parse(&format!("a{TH}r{TH_END}b"), &[], false, 0);
        assert_eq!((r.content.as_str(), r.reasoning.as_slice()), ("ab", &["r".to_string()][..]));
    }

    #[test]
    fn schema_types_the_values() {
        let text = call_text(
            "get_weather",
            &[("city", "Paris"), ("days", "3"), ("metric", "true"), ("ratio", "0.5"), ("note", "null"), ("mode", "fast")],
        );
        let r = parse(&text, &[weather()], false, 0);
        assert!(r.reports.is_empty(), "{:?}", r.reports);
        assert_eq!(
            r.calls[0].arguments,
            obj(&[("city", st("Paris")), ("days", Json::Num(3.0)), ("metric", Json::Bool(true)), ("ratio", Json::Num(0.5)),
                ("note", Json::Null), ("mode", st("fast"))])
        );
    }

    #[test]
    fn string_values_are_kept_exactly() {
        let code = "fn main() {\n    println!(\"hi\");\n}\n";
        let text = call_text("get_weather", &[("city", " 5 \n"), ("mode", code)]);
        let r = parse(&text, &[weather()], false, 0);
        assert_eq!(r.calls[0].arguments, obj(&[("city", st(" 5 \n")), ("mode", st(code))]));
        // Text that looks like JSON stays text for a string property.
        let r = parse(&call_text("get_weather", &[("city", "[1, 2]")]), &[weather()], false, 0);
        assert_eq!(r.calls[0].arguments, obj(&[("city", st("[1, 2]"))]));
    }

    #[test]
    fn values_that_do_not_fit_the_schema_stay_text() {
        let text = call_text("get_weather", &[("days", "five"), ("ratio", "true"), ("note", "5"), ("metric", "")]);
        let r = parse(&text, &[weather()], false, 0);
        assert_eq!(r.calls[0].arguments, obj(&[("days", st("five")), ("ratio", st("true")), ("note", st("5")), ("metric", st(""))]));
        // An integer property takes only whole numbers.
        let r = parse(&call_text("get_weather", &[("days", "2.5")]), &[weather()], false, 0);
        assert_eq!(r.calls[0].arguments, obj(&[("days", st("2.5"))]));
    }

    #[test]
    fn without_a_schema_values_are_json_or_text() {
        let quoted = "\"quoted\"";
        let text = call_text("unknown", &[("n", " 5 "), ("b", "true"), ("z", "null"), ("s", "hello"), ("q", quoted), ("a", "[1,2]")]);
        let r = parse(&text, &[], false, 0);
        assert_eq!(
            r.calls[0].arguments,
            obj(&[("n", Json::Num(5.0)), ("b", Json::Bool(true)), ("z", Json::Null), ("s", st("hello")), ("q", st(quoted)),
                ("a", Json::Array(vec![Json::Num(1.0), Json::Num(2.0)]))])
        );
    }

    #[test]
    fn nested_json_values() {
        let nested = r#"{"lat": 48.85, "tags": ["a", {"b": [true, null]}], "s": "x<y"}"#;
        let text = call_text("get_weather", &[("where", nested), ("tags", "[[1], [2, [3]], {}]")]);
        let r = parse(&text, &[weather()], false, 0);
        assert!(r.reports.is_empty(), "{:?}", r.reports);
        assert_eq!(calls(&r), [("get_weather".to_string(),
            r#"{"where":{"lat":48.85,"tags":["a",{"b":[true,null]}],"s":"x<y"},"tags":[[1],[2,[3]],{}]}"#.to_string())]);
    }

    #[test]
    fn multiple_parallel_calls() {
        let a = call_text("get_weather", &[("city", "Paris")]);
        let b = call_text("get_weather", &[("city", "Rome")]);
        let c = call_text("lookup", &[]);
        for sep in ["", "\n", "\n\n  "] {
            let text = format!("Checking.{sep}{a}{sep}{b}{sep}{c}{sep}");
            let r = parse(&text, &[weather()], false, 0);
            assert_eq!(
                calls(&r),
                [("get_weather".into(), r#"{"city":"Paris"}"#.into()), ("get_weather".into(), r#"{"city":"Rome"}"#.into()),
                    ("lookup".into(), "{}".into())],
                "{sep:?}"
            );
            assert_eq!(r.content.trim(), "Checking.");
            assert!(r.reports.is_empty() && r.error.is_none());
        }
        // Reasoning, then parallel calls: one reasoning block, then the calls.
        let r = parse(&format!("plan{TH_END}{a}{b}"), &[weather()], true, 0);
        assert_eq!((r.reasoning.len(), r.calls.len(), r.content.as_str()), (1, 2, ""));
    }

    #[test]
    fn strings_containing_tags() {
        let value = format!("use {TC} and {TH}x{TH_END} or {AK}k{AK_END}{AV} in text {TC_END}!");
        let text = call_text("get_weather", &[("city", &value)]);
        let r = parse(&text, &[weather()], false, 0);
        assert!(r.reports.is_empty(), "{:?}", r.reports);
        assert_eq!(r.calls[0].arguments, obj(&[("city", st(&value))]));
        // Tags inside reasoning are reasoning text.
        let r = parse(&format!("maybe {TC}f{TC_END}?{TH_END}ok"), &[], true, 0);
        assert_eq!((r.reasoning[0].as_str(), r.calls.len(), r.content.as_str()), (format!("maybe {TC}f{TC_END}?").as_str(), 0, "ok"));
    }

    #[test]
    fn a_value_holding_its_own_closing_tag_splits_and_reports_the_tail() {
        let value = format!("abc{AV_END}def");
        let r = parse(&call_text("get_weather", &[("city", &value), ("days", "2")]), &[weather()], false, 0);
        assert_eq!(r.calls[0].arguments, obj(&[("city", st("abc")), ("days", Json::Num(2.0))]));
        assert!(r.reports.iter().any(|x| x.contains("unparsed text") && x.contains("def")), "{:?}", r.reports);
    }

    #[test]
    fn malformed_calls_are_reported_not_dropped() {
        let ok = call_text("g", &[("k", "1")]);
        // Truncated: the call never closes.
        let r = parse(&format!("{TC}f{AK}k{AK_END}{AV}v"), &[], false, 0);
        assert!(r.calls.is_empty() && r.reports.iter().any(|x| x.contains("lost call")), "{:?}", r.reports);
        let r = parse(&format!("{TC}get_wea"), &[], false, 0);
        assert!(r.calls.is_empty() && r.reports.iter().any(|x| x.contains("lost call")), "{:?}", r.reports);
        // A new call inside an open one: the first is lost and reported, the second parses.
        let r = parse(&format!("{TC}f{AK}k{AK_END}{AV}v{AV_END}{ok}"), &[], false, 0);
        assert_eq!(calls(&r), [("g".to_string(), r#"{"k":1}"#.to_string())]);
        assert!(r.reports.iter().any(|x| x.contains("lost call \"f\"")), "{:?}", r.reports);
        // An argument without a value, and stray text: the call survives, the losses are reported.
        let r = parse(&format!("{TC}f{AK}k{AK_END}junk{AK}j{AK_END}{AV}2{AV_END}{TC_END}"), &[], false, 0);
        assert_eq!(calls(&r), [("f".to_string(), r#"{"j":2}"#.to_string())]);
        assert!(r.reports.iter().any(|x| x.contains("no value")), "{:?}", r.reports);
        assert!(r.reports.iter().any(|x| x.contains("unparsed text") && x.contains("junk")), "{:?}", r.reports);
        // A duplicate key keeps the last value and reports the first.
        let r = parse(&call_text("f", &[("k", "1"), ("k", "2")]), &[], false, 0);
        assert_eq!(calls(&r), [("f".to_string(), r#"{"k":2}"#.to_string())]);
        assert!(r.reports.iter().any(|x| x.contains("duplicate")), "{:?}", r.reports);
        // Markup in the name: lost and reported; the next call still parses.
        let r = parse(&format!("{TC}f{AV}1{AV_END}{TC_END}{ok}"), &[], false, 0);
        assert_eq!(r.calls.len(), 1);
        assert!(r.reports.iter().any(|x| x.contains("markup in the name")), "{:?}", r.reports);
        // An empty call is noted.
        let r = parse(&format!("{TC}{TC_END}"), &[], false, 0);
        assert!(r.calls.is_empty() && r.reports == ["empty tool call"]);
        // Arguments without a name are an error, never a nameless call.
        let r = parse(&format!("{TC}{AK}k{AK_END}{AV}v{AV_END}{TC_END}"), &[], false, 0);
        assert_eq!(r.error.as_deref(), Some("nameless tool call"));
        assert!(r.calls.is_empty());
    }

    #[test]
    fn the_cap_stops_before_call_cap_plus_one() {
        let text: String = (0..8).map(|k| call_text(&format!("t{k}"), &[])).collect();
        let r = parse(&text, &[], false, 6);
        assert!(r.capped);
        assert_eq!(r.calls.len(), 6);
        let r = parse(&text, &[], false, 0);
        assert!(!r.capped);
        assert_eq!(r.calls.len(), 8);
    }

    /// The template writes arguments; the dialect reads them back to the same values.
    #[test]
    fn round_trips_the_templates_arguments() {
        let args = json::parse(r#"{"city": "Oslo", "days": 7, "metric": false, "ratio": 0.25, "note": null,
            "where": {"a": [1, "two", {"b": null}]}, "tags": ["x", "y"], "mode": "slow"}"#).unwrap();
        let mut text = format!("{TC}get_weather");
        for (k, v) in args.as_object().unwrap() {
            let written = match v {
                Json::Str(s) => s.clone(),
                other => json::serialize(other),
            };
            text.push_str(&format!("{AK}{k}{AK_END}{AV}{written}{AV_END}"));
        }
        text.push_str(TC_END);
        let r = parse(&text, &[weather()], false, 0);
        assert!(r.reports.is_empty(), "{:?}", r.reports);
        assert_eq!(r.calls[0].arguments, args);
    }

    #[test]
    fn stream_tags_and_defaults() {
        let d = GlmDialect;
        assert_eq!(d.stream_tags(), StreamTags { think_open: TH, think_close: TH_END, tool_open: TC });
        assert!(d.reasoning_first(true) && !d.reasoning_first(false));
        assert!(d.default_thinking());
        assert_eq!(TH.len() + TC.len(), 7 + 11);
    }

    /// The request fields that set the thinking switch, reasoning effort and clear_thinking.
    #[test]
    fn request_fields_map_to_the_template_switches() {
        let parse_req = |extra: &str| {
            let body = format!(r#"{{"messages":[{{"role":"user","content":"hi"}}]{extra}}}"#);
            ChatRequest::parse(&json::parse(&body).unwrap(), &|_: &str| Err("no images".to_string())).unwrap()
        };
        let r = parse_req("");
        assert_eq!((r.enable_thinking, r.reasoning_effort.as_deref(), r.clear_thinking), (None, None, None));
        assert_eq!(parse_req(r#","chat_template_kwargs":{"enable_thinking":false}"#).enable_thinking, Some(false));
        assert_eq!(parse_req(r#","enable_thinking":true"#).enable_thinking, Some(true));
        assert_eq!(parse_req(r#","thinking":{"type":"disabled"}"#).enable_thinking, Some(false));
        assert_eq!(parse_req(r#","thinking":{"type":"enabled","clear_thinking":true}"#).enable_thinking, Some(true));
        assert_eq!(parse_req(r#","thinking":{"type":"enabled","clear_thinking":true}"#).clear_thinking, Some(true));
        assert_eq!(parse_req(r#","reasoning_effort":"none""#).enable_thinking, Some(false));
        let r = parse_req(r#","reasoning_effort":"low""#);
        assert_eq!((r.enable_thinking, r.reasoning_effort.as_deref()), (None, Some("low")));
        let r = parse_req(r#","chat_template_kwargs":{"reasoning_effort":"high","clear_thinking":false}"#);
        assert_eq!((r.reasoning_effort.as_deref(), r.clear_thinking), (Some("high"), Some(false)));
        // The template's own switch wins over the others.
        let r = parse_req(r#","chat_template_kwargs":{"enable_thinking":true},"enable_thinking":false,"reasoning_effort":"none""#);
        assert_eq!(r.enable_thinking, Some(true));
        // History fields for the template.
        let body = r#"{"messages":[{"role":"assistant","content":"a","reasoning_content":"r1"},{"role":"assistant","content":"b","reasoning":"r2"},{"role":"tool","tool_call_id":"c1","content":"x"}]}"#;
        let req = ChatRequest::parse(&json::parse(body).unwrap(), &|_: &str| Err("no images".to_string())).unwrap();
        let m: &[ChatMessage] = &req.messages;
        assert_eq!((m[0].reasoning_content.as_deref(), m[1].reasoning_content.as_deref()), (Some("r1"), Some("r2")));
        assert_eq!((m[2].tool_call_id.as_deref(), m[0].tool_call_id.as_deref()), (Some("c1"), None));
    }

    // --- through the HTTP server --------------------------------------------------------------

    /// A stub engine: records the prompt options it is given (to render the prompt, and to count
    /// its tokens) and streams `text` in `chunk`-byte deltas.
    struct Stub {
        text: String,
        chunk: usize,
        seen: Mutex<Vec<PromptOptions>>,
        counted: Mutex<Vec<PromptOptions>>,
    }

    fn stub(text: &str, chunk: usize) -> Arc<Stub> {
        Arc::new(Stub { text: text.into(), chunk, seen: Mutex::new(Vec::new()), counted: Mutex::new(Vec::new()) })
    }

    impl Engine for Stub {
        fn tokenize(&self, _: &[ChatMessage], _: &[Tool], _: bool) -> usize {
            8
        }
        fn render_chat(&self, _: &[ChatMessage], _: &[Tool], _: bool) -> String {
            "prompt".into()
        }
        fn tokenize_prompt(&self, _: &[ChatMessage], _: &[Tool], opts: &PromptOptions) -> usize {
            self.counted.lock().unwrap().push(opts.clone());
            8
        }
        fn render_prompt(&self, _: &[ChatMessage], _: &[Tool], opts: &PromptOptions) -> String {
            self.seen.lock().unwrap().push(opts.clone());
            "prompt".into()
        }
        fn generate(&self, _: &str, _: &GenerateParams, on_delta: &mut dyn FnMut(&str)) -> Result<GenerateOutcome, String> {
            for piece in self.text.as_bytes().chunks(self.chunk) {
                on_delta(std::str::from_utf8(piece).expect("ASCII test text"));
            }
            Ok(GenerateOutcome { text: self.text.clone(), finish_reason: "stop".into(), completion_tokens: 9 })
        }
    }

    fn serve(stub: Arc<Stub>) -> String {
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let base = format!("127.0.0.1:{}", listener.local_addr().unwrap().port());
        std::thread::spawn(move || {
            let dialect: Arc<dyn Dialect> = Arc::new(GlmDialect);
            let _ = crate::http::serve_listener(listener, move |req| {
                let body = json::parse_bytes(&req.body).unwrap();
                match crate::chat::handle(stub.clone(), dialect.clone(), &body) {
                    Ok(resp) => resp,
                    Err(e) => crate::http::json_response(e.status, &json::serialize(&e.body())),
                }
            });
        });
        base
    }

    fn post(base: &str, body: &str) -> String {
        use std::io::{Read, Write};
        let mut s = std::net::TcpStream::connect(base).unwrap();
        write!(s, "POST /v1/chat/completions HTTP/1.1\r\nHost: {base}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}", body.len()).unwrap();
        let mut out = String::new();
        s.read_to_string(&mut out).unwrap();
        assert!(out.starts_with("HTTP/1.1 200"), "{out}");
        out.split_once("\r\n\r\n").unwrap().1.to_string()
    }

    const TOOLS: &str = r#""tools":[{"type":"function","function":{"name":"get_weather","parameters":{"type":"object","properties":{"city":{"type":"string"},"days":{"type":"integer"}}}}}]"#;

    /// A thinking completion with two calls, streamed in small deltas that split every tag:
    /// reasoning streams live, content carries no markup, and the calls equal the non-stream
    /// response's.
    #[test]
    fn a_call_split_across_stream_deltas_matches_the_whole_response() {
        let text = format!(
            "Checking the weather.{TH_END}I'll look it up.{}{}",
            call_text("get_weather", &[("city", "Paris"), ("days", "2")]),
            call_text("get_weather", &[("city", "Rome")])
        );
        for chunk in [1, 3, 5, 7] {
            let stub = stub(&text, chunk);
            let base = serve(stub.clone());
            let body = format!(r#"{{"messages":[{{"role":"user","content":"weather?"}}],{TOOLS},"stream":true}}"#);
            let raw = post(&base, &body);
            let (mut reasoning, mut content, mut names, mut args, mut finish) = (String::new(), String::new(), Vec::new(), Vec::new(), None);
            for line in raw.lines().filter_map(|l| l.trim().strip_prefix("data: ")) {
                if line == "[DONE]" {
                    continue;
                }
                let v = json::parse(line).unwrap();
                let Some(ch) = v.get("choices").and_then(|c| c.as_array()).and_then(|c| c.first()) else { continue };
                if let Some(f) = ch.get("finish_reason").and_then(|f| f.as_str()) {
                    finish = Some(f.to_string());
                }
                let Some(d) = ch.get("delta") else { continue };
                assert!(d.get("reasoning").is_none(), "reasoning under a second name: {line}");
                if let Some(r) = d.get("reasoning_content").and_then(|r| r.as_str()) {
                    reasoning.push_str(r);
                }
                if let Some(c) = d.get("content").and_then(|c| c.as_str()) {
                    assert!(!c.contains('<') && !c.contains('>'), "markup in a content delta: {c:?}");
                    content.push_str(c);
                }
                for tc in d.get("tool_calls").and_then(|t| t.as_array()).unwrap_or(&[]) {
                    let f = tc.get("function").unwrap();
                    if let Some(n) = f.get("name").and_then(|n| n.as_str()) {
                        names.push(n.to_string());
                    }
                    if let Some(a) = f.get("arguments").and_then(|a| a.as_str()).filter(|a| !a.is_empty()) {
                        args.push(a.to_string());
                    }
                }
            }
            assert_eq!(reasoning, "Checking the weather.", "chunk {chunk}");
            assert_eq!(content, "I'll look it up.", "chunk {chunk}");
            assert_eq!(finish.as_deref(), Some("tool_calls"));
            // The same request, whole.
            let whole = json::parse(&post(&base, &body.replace(r#","stream":true"#, ""))).unwrap();
            let msg = whole.get("choices").and_then(|c| c.as_array()).and_then(|c| c.first()).and_then(|c| c.get("message")).unwrap();
            assert_eq!(msg.get("reasoning_content").and_then(|r| r.as_str()), Some(reasoning.as_str()));
            assert!(msg.get("reasoning").is_none());
            let whole_calls: Vec<(String, String)> = msg.get("tool_calls").and_then(|t| t.as_array()).unwrap().iter()
                .map(|t| {
                    let f = t.get("function").unwrap();
                    (f.get("name").and_then(|n| n.as_str()).unwrap().to_string(), f.get("arguments").and_then(|a| a.as_str()).unwrap().to_string())
                })
                .collect();
            let streamed: Vec<(String, String)> = names.into_iter().zip(args).collect();
            assert_eq!(streamed, whole_calls, "chunk {chunk}");
            assert_eq!(whole_calls, [("get_weather".to_string(), r#"{"city":"Paris","days":2}"#.to_string()),
                ("get_weather".to_string(), r#"{"city":"Rome"}"#.to_string())]);
            // No thinking field in the request: GLM's default (on) reached the engine.
            assert!(stub.seen.lock().unwrap().iter().all(|o| o.thinking));
        }
    }

    /// A streamed response's deltas, joined: (reasoning, content). Reasoning must come under
    /// `reasoning_content` only.
    fn streamed_text(raw: &str) -> (String, String) {
        let (mut reasoning, mut content) = (String::new(), String::new());
        for line in raw.lines().filter_map(|l| l.trim().strip_prefix("data: ")) {
            if line == "[DONE]" {
                continue;
            }
            let v = json::parse(line).unwrap();
            for ch in v.get("choices").and_then(|c| c.as_array()).unwrap_or(&[]) {
                let Some(d) = ch.get("delta") else { continue };
                assert!(d.get("reasoning").is_none(), "reasoning under a second name: {line}");
                if let Some(r) = d.get("reasoning_content").and_then(|r| r.as_str()) {
                    reasoning.push_str(r);
                }
                if let Some(c) = d.get("content").and_then(|c| c.as_str()) {
                    content.push_str(c);
                }
            }
        }
        (reasoning, content)
    }

    /// Thinking off end to end: the engine is told (to render the prompt and to count it), and
    /// the completion is read as content, whole and streamed.
    #[test]
    fn thinking_off_reaches_the_engine_and_the_parse() {
        let stub = stub("Just the answer.", 4);
        let base = serve(stub.clone());
        let body = r#"{"messages":[{"role":"user","content":"hi"}],"thinking":{"type":"disabled"},"reasoning_effort":"low"}"#;
        let v = json::parse(&post(&base, body)).unwrap();
        let msg = v.get("choices").and_then(|c| c.as_array()).and_then(|c| c.first()).and_then(|c| c.get("message")).unwrap();
        assert_eq!(msg.get("content").and_then(|c| c.as_str()), Some("Just the answer."));
        assert!(msg.get("reasoning_content").is_none() && msg.get("reasoning").is_none());
        let streamed = body.replace(r#""reasoning_effort":"low""#, r#""reasoning_effort":"low","stream":true"#);
        assert_eq!(streamed_text(&post(&base, &streamed)), (String::new(), "Just the answer.".to_string()));
        let off = PromptOptions { thinking: false, reasoning_effort: Some("low".into()), clear_thinking: None };
        assert_eq!(stub.seen.lock().unwrap().as_slice(), [off.clone(), off.clone()]);
        assert_eq!(stub.counted.lock().unwrap().as_slice(), [off.clone(), off]);
    }

    /// Every form of the thinking switch, and their precedence, reaches the engine: the same
    /// options to render the prompt and to count its tokens. `reasoning_effort` and
    /// `clear_thinking` travel with it; `clear_thinking` is off unless a request sends it.
    #[test]
    fn the_thinking_switch_forms_and_their_precedence_reach_the_engine() {
        let opts = |thinking: bool, effort: Option<&str>, clear: Option<bool>| PromptOptions {
            thinking,
            reasoning_effort: effort.map(str::to_string),
            clear_thinking: clear,
        };
        let cases = [
            // No switch: GLM-5.3-Flash's template thinks.
            ("", opts(true, None, None)),
            // vLLM and SGLang.
            (r#""chat_template_kwargs":{"enable_thinking":false}"#, opts(false, None, None)),
            (r#""chat_template_kwargs":{"enable_thinking":true}"#, opts(true, None, None)),
            (r#""enable_thinking":false"#, opts(false, None, None)),
            // GLM and Anthropic.
            (r#""thinking":{"type":"disabled"}"#, opts(false, None, None)),
            (r#""thinking":{"type":"enabled","budget_tokens":1024}"#, opts(true, None, None)),
            // OpenAI's effort: "none" turns thinking off; every value goes to the template as sent.
            (r#""reasoning_effort":"none""#, opts(false, Some("none"), None)),
            (r#""chat_template_kwargs":{"reasoning_effort":"none"}"#, opts(false, Some("none"), None)),
            (r#""reasoning_effort":"low""#, opts(true, Some("low"), None)),
            (r#""reasoning_effort":"low","chat_template_kwargs":{"reasoning_effort":"high"}"#, opts(true, Some("low"), None)),
            // Precedence: chat_template_kwargs, top-level enable_thinking, thinking.type, effort "none".
            (r#""chat_template_kwargs":{"enable_thinking":true},"thinking":{"type":"disabled"},"reasoning_effort":"none""#,
                opts(true, Some("none"), None)),
            (r#""enable_thinking":true,"thinking":{"type":"disabled"}"#, opts(true, None, None)),
            (r#""thinking":{"type":"enabled"},"reasoning_effort":"none""#, opts(true, Some("none"), None)),
            (r#""thinking":{"type":"disabled"},"reasoning_effort":"high""#, opts(false, Some("high"), None)),
            // clear_thinking: chat_template_kwargs, then thinking.
            (r#""chat_template_kwargs":{"clear_thinking":true}"#, opts(true, None, Some(true))),
            (r#""thinking":{"type":"enabled","clear_thinking":true}"#, opts(true, None, Some(true))),
            (r#""chat_template_kwargs":{"clear_thinking":false},"thinking":{"type":"enabled","clear_thinking":true}"#,
                opts(true, None, Some(false))),
        ];
        let stub = stub("Answer.", 4);
        let base = serve(stub.clone());
        for (extra, want) in cases {
            let sep = if extra.is_empty() { "" } else { "," };
            post(&base, &format!(r#"{{"messages":[{{"role":"user","content":"hi"}}]{sep}{extra}}}"#));
            let got = (stub.seen.lock().unwrap().pop(), stub.counted.lock().unwrap().pop());
            assert_eq!(got, (Some(want.clone()), Some(want)), "{extra}");
        }
    }

    /// Reasoning goes out under `reasoning_content` from both places the stream sends it (live, and
    /// after generation for a think block behind a tool call) and adds up to the whole response's.
    #[test]
    fn reasoning_after_a_tool_call_streams_under_the_same_field() {
        let text = format!("Plan.{TH_END}Checking.{}{TH}Recheck.{TH_END}", call_text("get_weather", &[("city", "Paris")]));
        let base = serve(stub(&text, 3));
        let body = format!(r#"{{"messages":[{{"role":"user","content":"weather?"}}],{TOOLS},"stream":true}}"#);
        let (reasoning, content) = streamed_text(&post(&base, &body));
        assert_eq!((reasoning.as_str(), content.as_str()), ("Plan.Recheck.", "Checking."));
        let whole = json::parse(&post(&base, &body.replace(r#","stream":true"#, ""))).unwrap();
        let msg = whole.get("choices").and_then(|c| c.as_array()).and_then(|c| c.first()).and_then(|c| c.get("message")).unwrap();
        assert_eq!(msg.get("reasoning_content").and_then(|r| r.as_str()), Some("Plan.Recheck."));
        assert!(msg.get("reasoning").is_none());
    }

    /// This file spells no complete think or tool-call tag, and keeps to ASCII, so a decoded
    /// escape (an editing tool writing the character in place of its escape) cannot hide here.
    #[test]
    fn source_hygiene() {
        let src = include_str!("glm.rs");
        assert!(src.is_ascii());
        for t in [TH, TH_END, TC, TC_END, AK, AK_END, AV, AV_END] {
            assert!(!src.contains(t), "literal {t} in glm.rs");
        }
    }
}
