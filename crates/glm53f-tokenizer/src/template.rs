//! GLM-5.3-Flash's chat template, hand-written.
//!
//! [`render`] produces the text `transformers`' `apply_chat_template` renders from the
//! checkpoint's chat_template.jinja, byte for byte, for what the API sends: system, user,
//! assistant and tool messages; text and image content; tools; assistant tool calls and
//! reasoning; tool results. tests/goldens.rs checks it against the reference renders in
//! oracle/goldens/template/. The template's rules, in the order it applies them:
//!
//! - **Prefix:** `[gMASK]<sop>`, then `<|system|>Reasoning Effort: X`, where X is `Low` or `High`
//!   when `reasoning_effort` is exactly `low` or `high`, and `Max` otherwise (unset,
//!   `medium`, `none` and `High` all mean Max).
//! - **Tools** (when the list is not empty): a system block listing each tool's function object
//!   as one JSON line (`tojson` of each value; the `strict` and `defer_loading` keys are left out;
//!   a tool whose `defer_loading` is true is not listed), then the call format.
//! - **Messages:** `<|system|>`, `<|user|>` + the text, as given (no trimming). Roles other than
//!   system, user, assistant and tool render nothing.
//! - **Assistant:** `<|assistant|>`, then &lt;think&gt;R&lt;/think&gt; when the turn has
//!   reasoning R, else an empty think block; then the content, stripped as Python's `str.strip`
//!   strips; then each tool call as &lt;tool_call&gt;NAME, one
//!   &lt;arg_key&gt;K&lt;/arg_key&gt;&lt;arg_value&gt;V&lt;/arg_value&gt; per argument, and
//!   &lt;/tool_call&gt;, where V is a string argument as is and any other value as `tojson`.
//!   The reasoning is `reasoning_content` when it is a string; otherwise, when the content holds
//!   a closing think tag, the text before the first closing tag (after the last opening tag in
//!   it), and the content becomes the text after the last closing tag. With `clear_thinking`,
//!   only turns after the last user message keep their reasoning.
//! - **Tool results:** a run of tool messages renders once, at its first message:
//!   `<|observation|>`, then &lt;tool_response&gt;TEXT&lt;/tool_response&gt; per result. The
//!   results follow the order of the preceding assistant turn's calls when every result and call
//!   has an id, the ids are unique on both sides and every result answers one of the calls;
//!   otherwise message order.
//! - **Generation prompt:** `<|assistant|>` and an opening think tag: the prompt opens the
//!   reasoning block.
//!
//! **Thinking off is not in the template.** The template always opens the think block; it has
//! no `enable_thinking` variable (the reference ignores one). [`Options::thinking`] = false ends
//! the prompt with `<|assistant|>` and an empty think block instead: the form the template itself
//! writes for an assistant turn without reasoning, so the model continues exactly as after such
//! a turn. The goldens pin that form (`rendered_with_answer`). Every other output equals the
//! reference.
//!
//! **Images** render as `<|begin_of_image|>` + marker + `<|end_of_image|>`. With no marker the
//! marker is `<|image|>`, as in the template; an engine puts its own marker there and expands it
//! when it tokenizes ([`crate::Tokenizer::encode_with_markers`]).
//!
//! **Not supported:** tool results given as lists of `output` entries or `tool_reference`
//! parts, and a tool message's `id` standing in for `tool_call_id`. The API never produces them.
//!
//! **From the API's request types** (`glm53f-api`), an engine builds the inputs as follows:
//! - a message's content: [`Content::from_marked_text`] with the API's image-marker characters;
//! - `reasoning_content` and `tool_call_id`: as they are;
//! - each tool call: [`ToolCall::from_openai`] (its `arguments` is JSON text);
//! - each tool: [`crate::json::parse`] of the API's serialization of the tool's `raw` object,
//!   which keeps the client's key order. The API stores numbers as `f64`, so a schema number
//!   written as an integral float (`1.0`, `1e3`) renders as an integer, and integers beyond
//!   2^53 lose precision. Integers and other floats render exactly;
//! - the options: the request's thinking switch, `reasoning_effort` and `clear_thinking`.
//!
//! Every tag literal in this file is assembled from pieces, so tools that scan sources for
//! markup do not misread it.

use crate::json::{write_python_json, Value};

const GMASK: &str = concat!("[", "gMASK", "]");
const SOP: &str = concat!("<", "sop", ">");
const SYSTEM: &str = concat!("<", "|system|", ">");
const USER: &str = concat!("<", "|user|", ">");
const ASSISTANT: &str = concat!("<", "|assistant|", ">");
const OBSERVATION: &str = concat!("<", "|observation|", ">");
/// Opens a reasoning block.
pub const THINK: &str = concat!("<", "think", ">");
/// Closes a reasoning block.
pub const THINK_END: &str = concat!("<", "/think", ">");
const TOOL_CALL: &str = concat!("<", "tool_call", ">");
const TOOL_CALL_END: &str = concat!("<", "/tool_call", ">");
const ARG_KEY: &str = concat!("<", "arg_key", ">");
const ARG_KEY_END: &str = concat!("<", "/arg_key", ">");
const ARG_VALUE: &str = concat!("<", "arg_value", ">");
const ARG_VALUE_END: &str = concat!("<", "/arg_value", ">");
const TOOL_RESPONSE: &str = concat!("<", "tool_response", ">");
const TOOL_RESPONSE_END: &str = concat!("<", "/tool_response", ">");
const TOOLS: &str = concat!("<", "tools", ">");
const TOOLS_END: &str = concat!("<", "/tools", ">");
const BEGIN_IMAGE: &str = concat!("<", "|begin_of_image|", ">");
/// The template's image placeholder token.
pub const IMAGE: &str = concat!("<", "|image|", ">");
const END_IMAGE: &str = concat!("<", "|end_of_image|", ">");
const BEGIN_VIDEO: &str = concat!("<", "|begin_of_video|", ">");
const VIDEO: &str = concat!("<", "|video|", ">");
const END_VIDEO: &str = concat!("<", "|end_of_video|", ">");
const BEGIN_AUDIO: &str = concat!("<", "|begin_of_audio|", ">");
const END_AUDIO: &str = concat!("<", "|end_of_audio|", ">");

/// The length of the chat_template.jinja this renderer reproduces (sha256
/// `0c4099f3382d6c92700dfb99725025360966fd73032f0ecf32377c0d9e6309c5`).
pub const TEMPLATE_LEN: usize = 10_950;
/// Its FNV-1a (64-bit) digest.
pub const TEMPLATE_FNV1A64: u64 = 0x1f19_9a1d_d56b_7ce5;

/// Guard against chat-template drift: `Ok` when `template` is the chat_template.jinja this renderer
/// was written for and checked against. An engine should refuse to serve a checkpoint whose
/// template differs, because its prompts would silently diverge from the model's own.
pub fn check_template(template: &str) -> Result<(), String> {
    let digest = template.bytes().fold(0xcbf2_9ce4_8422_2325u64, |h, b| (h ^ b as u64).wrapping_mul(0x0000_0100_0000_01b3));
    if template.len() == TEMPLATE_LEN && digest == TEMPLATE_FNV1A64 {
        Ok(())
    } else {
        Err(format!(
            "chat template differs from the one this renderer reproduces ({} bytes, FNV-1a {digest:016x}; expected {TEMPLATE_LEN} bytes, {TEMPLATE_FNV1A64:016x})",
            template.len()
        ))
    }
}

/// One part of a message's content.
#[derive(Debug, Clone, PartialEq)]
pub enum Part {
    Text(String),
    /// An image: `<|begin_of_image|>` + marker + `<|end_of_image|>`, the marker defaulting to
    /// `<|image|>`.
    Image(Option<String>),
    Video,
    Audio,
}

/// A message's content, as the template reads it.
#[derive(Debug, Clone, PartialEq, Default)]
pub enum Content {
    /// `null` or absent.
    #[default]
    Null,
    Text(String),
    Parts(Vec<Part>),
}

impl Content {
    /// Split text in which images stand as markers `open ... close` (the API's image markers)
    /// into text and image parts; each image part keeps its whole marker.
    pub fn from_marked_text(text: &str, open: char, close: char) -> Content {
        if !text.contains(open) {
            return Content::Text(text.to_string());
        }
        let mut parts = Vec::new();
        let mut rest = text;
        while let Some(p) = rest.find(open) {
            let Some(q) = rest[p..].find(close) else { break };
            if p > 0 {
                parts.push(Part::Text(rest[..p].to_string()));
            }
            let end = p + q + close.len_utf8();
            parts.push(Part::Image(Some(rest[p..end].to_string())));
            rest = &rest[end..];
        }
        if !rest.is_empty() {
            parts.push(Part::Text(rest.to_string()));
        }
        Content::Parts(parts)
    }
}

impl From<&str> for Content {
    fn from(s: &str) -> Self {
        Content::Text(s.to_string())
    }
}

/// An assistant turn's tool call.
#[derive(Debug, Clone, PartialEq)]
pub struct ToolCall {
    pub id: Option<String>,
    pub name: String,
    /// A JSON object (the OpenAI wire form's `arguments` string, parsed).
    pub arguments: Value,
}

impl ToolCall {
    /// A call in the OpenAI wire form: `arguments` is JSON text, and empty text means no
    /// arguments (as servers pass it to templates). Anything but a JSON object is an error: the
    /// template cannot render it.
    pub fn from_openai(id: &str, name: &str, arguments: &str) -> Result<ToolCall, String> {
        let arguments = if arguments.trim().is_empty() {
            Value::Object(Vec::new())
        } else {
            crate::json::parse(arguments).map_err(|e| format!("arguments of tool call {name:?}: {e}"))?
        };
        if arguments.as_object().is_none() {
            return Err(format!("arguments of tool call {name:?} are not a JSON object"));
        }
        Ok(ToolCall { id: (!id.is_empty()).then(|| id.to_string()), name: name.to_string(), arguments })
    }
}

/// One chat message.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct Message {
    pub role: String,
    pub content: Content,
    /// An assistant turn's reasoning, carried back by the client.
    pub reasoning_content: Option<String>,
    pub tool_calls: Vec<ToolCall>,
    /// A tool result's call id.
    pub tool_call_id: Option<String>,
}

impl Message {
    /// A message with text content.
    pub fn new(role: &str, content: &str) -> Message {
        Message { role: role.to_string(), content: Content::Text(content.to_string()), ..Default::default() }
    }
}

/// The template's switches.
#[derive(Debug, Clone, PartialEq)]
pub struct Options {
    pub add_generation_prompt: bool,
    /// On: the prompt opens the think block (the template's only form). Off: it ends with an
    /// empty think block (see the module doc).
    pub thinking: bool,
    /// As the client sent it; the template decides what it means ([`effort_label`]).
    pub reasoning_effort: Option<String>,
    pub clear_thinking: bool,
}

impl Default for Options {
    fn default() -> Self {
        Options { add_generation_prompt: true, thinking: true, reasoning_effort: None, clear_thinking: false }
    }
}

/// The template's reasoning-effort word: `low` -> `Low`, `high` -> `High`, anything else (or
/// nothing) -> `Max`.
pub fn effort_label(reasoning_effort: Option<&str>) -> &'static str {
    match reasoning_effort {
        Some("low") => "Low",
        Some("high") => "High",
        _ => "Max",
    }
}

/// Python's `str.isspace` for one character: Unicode White_Space plus U+001C..U+001F.
fn py_isspace(c: char) -> bool {
    c.is_whitespace() || (0x1C..=0x1F).contains(&(c as u32))
}

/// Python's `str.strip()`.
fn py_strip(s: &str) -> &str {
    s.trim_matches(py_isspace)
}

/// The template's `visible_text`: text parts joined, media as their markup.
fn visible_text(content: &Content) -> String {
    match content {
        Content::Null => String::new(),
        Content::Text(s) => s.clone(),
        Content::Parts(parts) => {
            let mut out = String::new();
            for p in parts {
                match p {
                    Part::Text(t) => out.push_str(t),
                    Part::Image(marker) => {
                        out.push_str(BEGIN_IMAGE);
                        out.push_str(marker.as_deref().unwrap_or(IMAGE));
                        out.push_str(END_IMAGE);
                    }
                    Part::Video => {
                        out.push_str(BEGIN_VIDEO);
                        out.push_str(VIDEO);
                        out.push_str(END_VIDEO);
                    }
                    Part::Audio => {
                        out.push_str(BEGIN_AUDIO);
                        out.push_str(END_AUDIO);
                    }
                }
            }
            out
        }
    }
}

/// The template's `tool_to_json`: `{"k": tojson(v), ...}` over the function object's keys except
/// `defer_loading` and `strict`. Keys are written raw between quotes, as the template does.
fn tool_to_json(out: &mut String, function: &Value) -> Result<(), String> {
    let pairs = function.as_object().ok_or("a tool's function is not an object")?;
    out.push('{');
    let mut first = true;
    for (k, v) in pairs {
        if k == "defer_loading" || k == "strict" {
            continue;
        }
        if !first {
            out.push_str(", ");
        }
        first = false;
        out.push('"');
        out.push_str(k);
        out.push_str("\": ");
        write_python_json(out, v);
    }
    out.push('}');
    Ok(())
}

fn render_tools(out: &mut String, tools: &[Value]) -> Result<(), String> {
    out.push_str(SYSTEM);
    out.push_str("\n# Tools\n\nYou may call one or more functions to assist with the user query.\n\n");
    out.push_str("You are provided with function signatures within ");
    out.push_str(TOOLS);
    out.push_str(TOOLS_END);
    out.push_str(" XML tags:\n");
    out.push_str(TOOLS);
    out.push('\n');
    for (i, tool) in tools.iter().enumerate() {
        if tool.as_object().is_none() {
            return Err(format!("tools[{i}] is not an object"));
        }
        // `'function' in tool` then `tool = tool['function']`.
        let function = tool.get("function").unwrap_or(tool);
        if function.get("defer_loading").is_some_and(|d| d.truthy()) {
            continue;
        }
        tool_to_json(out, function).map_err(|e| format!("tools[{i}]: {e}"))?;
        out.push('\n');
    }
    out.push_str(TOOLS_END);
    out.push_str("\n\nFor each function call, output the function name and arguments within the following XML format:\n");
    out.push_str(TOOL_CALL);
    out.push_str("{function-name}");
    for n in 1..=2 {
        out.push_str(ARG_KEY);
        out.push_str(&format!("{{arg-key-{n}}}"));
        out.push_str(ARG_KEY_END);
        out.push_str(ARG_VALUE);
        out.push_str(&format!("{{arg-value-{n}}}"));
        out.push_str(ARG_VALUE_END);
    }
    out.push_str("...");
    out.push_str(TOOL_CALL_END);
    Ok(())
}

/// One tool call as the template writes it in an assistant turn.
pub fn render_tool_call(out: &mut String, call: &ToolCall) -> Result<(), String> {
    let args = call.arguments.as_object().ok_or_else(|| format!("arguments of tool call {:?} are not a JSON object", call.name))?;
    out.push_str(TOOL_CALL);
    out.push_str(&call.name);
    for (k, v) in args {
        out.push_str(ARG_KEY);
        out.push_str(k);
        out.push_str(ARG_KEY_END);
        out.push_str(ARG_VALUE);
        match v {
            Value::Str(s) => out.push_str(s),
            other => write_python_json(out, other),
        }
        out.push_str(ARG_VALUE_END);
    }
    out.push_str(TOOL_CALL_END);
    Ok(())
}

fn render_assistant(out: &mut String, m: &Message, keep_reasoning: bool) -> Result<(), String> {
    out.push_str(ASSISTANT);
    let mut content = visible_text(&m.content);
    let mut reasoning: Option<String> = m.reasoning_content.clone();
    if reasoning.is_none() {
        if let Some(first) = content.find(THINK_END) {
            let before = &content[..first];
            let r = match before.rfind(THINK) {
                Some(p) => &before[p + THINK.len()..],
                None => before,
            };
            let last = content.rfind(THINK_END).expect("found above");
            let after = content[last + THINK_END.len()..].to_string();
            reasoning = Some(r.to_string());
            content = after;
        }
    }
    out.push_str(THINK);
    if keep_reasoning {
        if let Some(r) = &reasoning {
            out.push_str(r);
        }
    }
    out.push_str(THINK_END);
    out.push_str(py_strip(&content));
    for call in &m.tool_calls {
        render_tool_call(out, call)?;
    }
    Ok(())
}

fn tool_response(out: &mut String, m: &Message) {
    out.push_str(TOOL_RESPONSE);
    out.push_str(&visible_text(&m.content));
    out.push_str(TOOL_RESPONSE_END);
}

/// A run of tool messages `block`, answering `calls` (the preceding assistant turn's, if any).
fn render_tool_block(out: &mut String, block: &[Message], calls: Option<&[ToolCall]>) {
    out.push_str(OBSERVATION);
    let result_id = |m: &Message| m.tool_call_id.clone().unwrap_or_default();
    let call_id = |c: &ToolCall| c.id.clone().unwrap_or_default();
    let sortable = calls.is_some_and(|calls| {
        let results_ok = block.iter().all(|m| {
            let id = result_id(m);
            !id.is_empty()
                && block.iter().filter(|x| result_id(x) == id).count() == 1
                && calls.iter().any(|c| call_id(c) == id)
        });
        let calls_ok = calls.iter().enumerate().all(|(i, c)| {
            let id = call_id(c);
            !id.is_empty() && !calls[i + 1..].iter().any(|d| call_id(d) == id)
        });
        results_ok && calls_ok
    });
    match calls {
        Some(calls) if sortable => {
            for c in calls {
                let id = call_id(c);
                for m in block.iter().filter(|m| result_id(m) == id) {
                    tool_response(out, m);
                }
            }
        }
        _ => {
            for m in block {
                tool_response(out, m);
            }
        }
    }
}

/// Render a conversation. `tools` are the request's tool objects (`{"type": "function",
/// "function": {...}}`, or a bare function object), in the client's key order.
pub fn render(messages: &[Message], tools: &[Value], opts: &Options) -> Result<String, String> {
    let mut out = String::new();
    out.push_str(GMASK);
    out.push_str(SOP);
    out.push_str(SYSTEM);
    out.push_str("Reasoning Effort: ");
    out.push_str(effort_label(opts.reasoning_effort.as_deref()));
    if !tools.is_empty() {
        render_tools(&mut out, tools)?;
    }
    let last_user = messages.iter().rposition(|m| m.role == "user");
    for (i, m) in messages.iter().enumerate() {
        match m.role.as_str() {
            "user" => {
                out.push_str(USER);
                out.push_str(&visible_text(&m.content));
            }
            "system" => {
                out.push_str(SYSTEM);
                out.push_str(&visible_text(&m.content));
            }
            "assistant" => {
                let keep = !opts.clear_thinking || last_user.is_none_or(|u| i > u);
                render_assistant(&mut out, m, keep).map_err(|e| format!("messages[{i}]: {e}"))?;
            }
            // A run of tool results renders once, at its first message; the rest render nothing.
            "tool" if i == 0 || messages[i - 1].role != "tool" => {
                let end = messages[i..].iter().take_while(|x| x.role == "tool").count() + i;
                let calls = (i > 0 && messages[i - 1].role == "assistant" && !messages[i - 1].tool_calls.is_empty())
                    .then(|| messages[i - 1].tool_calls.as_slice());
                render_tool_block(&mut out, &messages[i..end], calls);
            }
            _ => {}
        }
    }
    if opts.add_generation_prompt {
        out.push_str(ASSISTANT);
        out.push_str(THINK);
        if !opts.thinking {
            out.push_str(THINK_END);
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::json;

    #[test]
    fn prompt_opens_or_closes_the_think_block() {
        let msgs = [Message::new("user", "hi")];
        let on = render(&msgs, &[], &Options::default()).unwrap();
        let prefix = format!("{GMASK}{SOP}{SYSTEM}Reasoning Effort: Max{USER}hi{ASSISTANT}");
        assert_eq!(on, format!("{prefix}{THINK}"));
        let off = render(&msgs, &[], &Options { thinking: false, ..Options::default() }).unwrap();
        assert_eq!(off, format!("{prefix}{THINK}{THINK_END}"));
    }

    #[test]
    fn effort_words() {
        assert_eq!(effort_label(None), "Max");
        assert_eq!(effort_label(Some("low")), "Low");
        assert_eq!(effort_label(Some("high")), "High");
        for other in ["max", "medium", "none", "High", "LOW", ""] {
            assert_eq!(effort_label(Some(other)), "Max", "{other}");
        }
    }

    #[test]
    fn strip_is_pythons() {
        let t: String = [0x1C, 0x3000, 0x78, 0xA0, 0x2029, 0x1F].iter().map(|&c| char::from_u32(c).unwrap()).collect();
        assert_eq!(py_strip(&t), "x");
        let zwsp: String = [0x200B, 0x78].iter().map(|&c| char::from_u32(c).unwrap()).collect();
        assert_eq!(py_strip(&zwsp), zwsp);
    }

    #[test]
    fn marked_text_splits_into_image_parts() {
        let (open, close) = (char::from_u32(0xFDD0).unwrap(), char::from_u32(0xFDD1).unwrap());
        let text = format!("a{open}00ff:4{close}b{open}01:9{close}");
        let c = Content::from_marked_text(&text, open, close);
        assert_eq!(
            c,
            Content::Parts(vec![
                Part::Text("a".into()),
                Part::Image(Some(format!("{open}00ff:4{close}"))),
                Part::Text("b".into()),
                Part::Image(Some(format!("{open}01:9{close}"))),
            ])
        );
        assert_eq!(visible_text(&c), format!("a{BEGIN_IMAGE}{open}00ff:4{close}{END_IMAGE}b{BEGIN_IMAGE}{open}01:9{close}{END_IMAGE}"));
        assert_eq!(Content::from_marked_text("plain", open, close), Content::Text("plain".into()));
    }

    /// The wire form's arguments text keeps its key order and number forms (`1.0` stays a float,
    /// as in Python), and renders as the template renders it.
    #[test]
    fn openai_tool_calls() {
        let c = ToolCall::from_openai("c1", "f", r#"{"b": 1.0, "a": "x", "n": [1, null]}"#).unwrap();
        assert_eq!(c.id.as_deref(), Some("c1"));
        let mut out = String::new();
        render_tool_call(&mut out, &c).unwrap();
        let arg = |k: &str, v: &str| format!("{ARG_KEY}{k}{ARG_KEY_END}{ARG_VALUE}{v}{ARG_VALUE_END}");
        assert_eq!(out, format!("{TOOL_CALL}f{}{}{}{TOOL_CALL_END}", arg("b", "1.0"), arg("a", "x"), arg("n", "[1, null]")));
        let empty = ToolCall::from_openai("", "f", "  ").unwrap();
        assert_eq!((empty.id, empty.arguments), (None, Value::Object(Vec::new())));
        assert!(ToolCall::from_openai("x", "f", "[1]").is_err());
        assert!(ToolCall::from_openai("x", "f", "{").is_err());
    }

    #[test]
    fn non_object_arguments_are_an_error() {
        let mut m = Message::new("assistant", "");
        m.tool_calls.push(ToolCall { id: None, name: "f".into(), arguments: json::parse("[1]").unwrap() });
        assert!(render(&[m], &[], &Options::default()).unwrap_err().contains("not a JSON object"));
    }
}
