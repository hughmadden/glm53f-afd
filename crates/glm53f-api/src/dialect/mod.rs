//! The model boundary of the API: how a model's completion text splits into
//! content, reasoning and tool calls.
//!
//! The API owns everything that does not depend on the model: HTTP, JSON,
//! request validation, SSE framing and the streaming holdback. A [`Dialect`]
//! owns the markup the model writes, which its chat template defines: the
//! reasoning tags, the tool-call envelope and how arguments are typed. The
//! engine's `render_chat` and the dialect must describe the same template.
//!
//! [`mimo::MimoDialect`] is the reference implementation (MiMo-V2.6-Flash, from
//! mimo26f-afd v1.2.0). [`glm::GlmDialect`] is GLM-5.3-Flash's, written against
//! this trait and passed to [`crate::serve`] in its place.

pub mod glm;
pub mod mimo;

pub use glm::GlmDialect;
pub use mimo::MimoDialect;

use crate::json::Json;
use crate::types::Tool;

#[derive(Debug, Clone, PartialEq)]
pub struct ParsedCall {
    pub name: String,
    pub arguments: Json,
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct ParseResult {
    /// Text outside `<think>` and `<tool_call>` blocks.
    pub content: String,
    /// Reasoning blocks, in order.
    pub reasoning: Vec<String>,
    /// Parsed calls, in order.
    pub calls: Vec<ParsedCall>,
    /// Losses that must be surfaced (never silently dropped).
    pub reports: Vec<String>,
    /// A fatal parse error (e.g. a nameless call).
    pub error: Option<String>,
    /// True when the tool-call cap fired (finish_reason `tool_calls`).
    pub capped: bool,
}

/// The literal tags the SSE stream watches while it streams: text inside a
/// reasoning block goes out as reasoning deltas, and everything from a tool-call
/// opening on is held back until the whole completion is parsed. A tag split
/// across deltas is held until it resolves.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StreamTags {
    /// Opens a reasoning block (`<think>`).
    pub think_open: &'static str,
    /// Closes a reasoning block (`</think>`).
    pub think_close: &'static str,
    /// Opens a tool call (`<tool_call>`).
    pub tool_open: &'static str,
}

/// A model's completion markup.
pub trait Dialect: Send + Sync {
    /// Split a whole completion into content, reasoning blocks and tool calls.
    /// `tools` supplies the schema types arguments are coerced to; `thinking` is
    /// the request's thinking switch; `cap` > 0 stops before call `cap + 1` (T24),
    /// and 0 is no cap.
    fn parse(&self, text: &str, tools: &[Tool], thinking: bool, cap: usize) -> ParseResult;

    /// The tags the streaming splitter holds back.
    fn stream_tags(&self) -> StreamTags;

    /// Whether a completion starts inside a reasoning block because the chat
    /// template already opened it in the prompt. The streaming splitter then
    /// starts in reasoning, and [`Dialect::parse`] must read the text before the
    /// first closing tag as reasoning too. Default: no.
    fn reasoning_first(&self, thinking: bool) -> bool {
        let _ = thinking;
        false
    }

    /// The thinking switch when a request does not set one (its chat template's default).
    /// Default: off.
    fn default_thinking(&self) -> bool {
        false
    }

    /// What a request that turns thinking off gets, for a chat template with no off mode: the
    /// template's own lowest reasoning effort, with thinking on (the prompt opens the reasoning
    /// block and the parser reads it as reasoning). `None`: thinking off is off. Either way,
    /// `reasoning_effort: "none"` asks for no reasoning at all.
    fn thinking_off_effort(&self) -> Option<&'static str> {
        None
    }
}
