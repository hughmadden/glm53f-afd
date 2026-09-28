//! GLM-5.3-Flash's text format for the engine: `glm53f-tokenizer`'s tokenizer and chat template
//! behind [`PromptCodec`].
//!
//! The chat template must be the **official** checkpoint's `chat_template.jinja`
//! (`zai-org/GLM-5.3-Flash`): [`GlmPrompts::load`] refuses any other text, because the renderer
//! reproduces that template byte for byte and a different one (the EXL3 checkpoint ships an older
//! template) would silently change every prompt.
//!
//! The request's messages are converted as `glm53f_tokenizer::template` documents: content with
//! the API's image markers split into text and image parts, tool calls parsed from their JSON
//! argument text, tools re-read from the client's raw objects (their key order kept), and the
//! request's thinking switch, `reasoning_effort` and `clear_thinking` as the template's options.
//! Decoding skips special tokens, as the reference server does; the reasoning and tool-call tags
//! are not special and reach the API, which splits the completion with the GLM dialect.

use std::path::Path;

use glm53f_api::engine::{PromptOptions, IMAGE_CLOSE, IMAGE_OPEN};
use glm53f_api::types::{ChatMessage, Tool};
use glm53f_tokenizer::json::Value;
use glm53f_tokenizer::{template, Tokenizer};

use crate::engine::{expand_image_marker, PromptCodec};
use crate::model::Token;

/// GLM-5.3-Flash's tokenizer and chat template.
pub struct GlmPrompts {
    tok: Tokenizer,
}

impl GlmPrompts {
    /// Load `tokenizer.json` and check `chat_template.jinja`, both from the official checkpoint.
    pub fn load(tokenizer_json: &Path, chat_template: &Path) -> Result<Self, String> {
        let text = std::fs::read_to_string(chat_template).map_err(|e| format!("{}: {e}", chat_template.display()))?;
        template::check_template(&text).map_err(|e| format!("{}: {e}", chat_template.display()))?;
        Ok(GlmPrompts { tok: Tokenizer::from_file(tokenizer_json)? })
    }

    /// From a loaded tokenizer; the caller has checked the chat template.
    pub fn from_tokenizer(tok: Tokenizer) -> Self {
        GlmPrompts { tok }
    }

    pub fn tokenizer(&self) -> &Tokenizer {
        &self.tok
    }

    /// The tokens that end a completion: `<|endoftext|>` (154,820), `<|user|>` (154,827) and
    /// `<|observation|>` (154,829).
    pub fn stop_ids(&self) -> Result<Vec<Token>, String> {
        self.tok.stop_ids().map(|s| s.to_vec())
    }

    /// One past the largest token id (154,856): the model's `Limits::sample_vocab`, below the LM
    /// head's 154,880 rows.
    pub fn id_bound(&self) -> usize {
        self.tok.id_bound()
    }
}

/// A request's conversation in the chat template's terms.
pub fn to_template(messages: &[ChatMessage], tools: &[Tool], opts: &PromptOptions)
    -> Result<(Vec<template::Message>, Vec<Value>, template::Options), String> {
    let msgs = messages
        .iter()
        .enumerate()
        .map(|(i, m)| {
            let tool_calls = m
                .tool_calls
                .iter()
                .map(|tc| template::ToolCall::from_openai(&tc.id, &tc.name, &tc.arguments))
                .collect::<Result<Vec<_>, _>>()
                .map_err(|e| format!("messages[{i}]: {e}"))?;
            Ok(template::Message {
                role: m.role.clone(),
                content: template::Content::from_marked_text(&m.content, IMAGE_OPEN, IMAGE_CLOSE),
                reasoning_content: m.reasoning_content.clone(),
                tool_calls,
                tool_call_id: m.tool_call_id.clone(),
            })
        })
        .collect::<Result<Vec<_>, String>>()?;
    let tools = tools
        .iter()
        .map(|t| glm53f_tokenizer::json::parse(&glm53f_api::json::serialize(&t.raw)))
        .collect::<Result<Vec<_>, _>>()
        .map_err(|e| format!("tools: {e}"))?;
    let opts = template::Options {
        add_generation_prompt: true,
        thinking: opts.thinking,
        reasoning_effort: opts.reasoning_effort.clone(),
        clear_thinking: opts.clear_thinking.unwrap_or(false),
    };
    Ok((msgs, tools, opts))
}

impl PromptCodec for GlmPrompts {
    fn render(&self, messages: &[ChatMessage], tools: &[Tool], opts: &PromptOptions) -> Result<String, String> {
        let (m, t, o) = to_template(messages, tools, opts)?;
        template::render(&m, &t, &o)
    }

    fn encode(&self, prompt: &str) -> Result<Vec<Token>, String> {
        self.tok.encode_with_markers(prompt, IMAGE_OPEN, IMAGE_CLOSE, &mut |inner| expand_image_marker(inner))
    }

    fn decode_bytes(&self, ids: &[Token]) -> Vec<u8> {
        self.tok.decode_bytes(ids, true)
    }
}
