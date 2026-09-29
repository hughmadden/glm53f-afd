//! The `Engine` trait — the seam the coordinator implements and the API tests
//! stub. The API owns the HTTP surface and the tool-call parsing (through the
//! model's [`crate::Dialect`]); the engine owns tokenization, the chat template,
//! image decoding and generation.

use std::time::Duration;

use crate::types::{ChatMessage, Tool};

/// Sampling/control parameters handed to [`Engine::generate`].
#[derive(Debug, Clone)]
pub struct GenerateParams {
    pub max_tokens: usize,
    /// 0 (the default) is greedy; a positive temperature samples (DS41RT v15's contract: filters
    /// left at `top_p` 1, `top_k` 0 and `min_p` 0 are off).
    pub temperature: f64,
    pub top_p: f64,
    pub top_k: usize,
    pub min_p: f64,
    /// The request's seed; none: the engine draws one.
    pub seed: Option<u64>,
    pub stop: Vec<String>,
    pub thinking: bool,
    /// Set by the API once the client is gone (a failed write): the engine stops
    /// generating and frees the request (perf reset Q2).
    pub cancel: Option<std::sync::Arc<std::sync::atomic::AtomicBool>>,
    /// The request's images, in prompt order (perf reset V2). Each stands in the rendered prompt
    /// as an [`image_marker`].
    pub images: Vec<std::sync::Arc<ImageInput>>,
    /// The request's place in the engine's queue ([`Engine::admit`]), taken by the engine when it
    /// hands the request on.
    pub place: std::sync::Arc<std::sync::Mutex<Option<QueuePlace>>>,
}

/// A place in an engine's bounded request queue (perf reset V3, DS41RT v15's admission), held
/// from before the response starts until the engine takes the request; dropping it gives the
/// place back.
pub struct QueuePlace(Option<Box<dyn FnOnce() + Send>>);

impl QueuePlace {
    pub fn new(release: impl FnOnce() + Send + 'static) -> Self {
        QueuePlace(Some(Box::new(release)))
    }
}

impl Drop for QueuePlace {
    fn drop(&mut self) {
        if let Some(release) = self.0.take() {
            release();
        }
    }
}

impl std::fmt::Debug for QueuePlace {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("QueuePlace")
    }
}

impl Default for GenerateParams {
    fn default() -> Self {
        GenerateParams { max_tokens: 65_536, temperature: 0.0, top_p: 1.0, top_k: 0, min_p: 0.0, seed: None,
            stop: Vec::new(), thinking: false, cancel: None, images: Vec::new(), place: Default::default() }
    }
}

/// A decoded image of a chat request (perf reset V2): RGB8 pixels, the number of language-model
/// positions it takes (`tokens`, its merged 2x2 patch grid) and a hash of its encoded bytes.
#[derive(Debug)]
pub struct ImageInput {
    pub hash: u64,
    pub tokens: usize,
    pub width: u32,
    pub height: u32,
    pub rgb: Vec<u8>,
}

/// The reserved characters that delimit an image in rendered message text. U+FDD0 and U+FDD1 are
/// Unicode noncharacters; the API strips them from client text, so only the API can place one.
pub const IMAGE_OPEN: char = '\u{FDD0}';
pub const IMAGE_CLOSE: char = '\u{FDD1}';

/// The text an image stands as in a message (where the chat template renders the model's image
/// placeholder tokens): its hash and token count, which the engine's tokenizer turns into the
/// image's token span.
pub fn image_marker(img: &ImageInput) -> String {
    format!("{IMAGE_OPEN}{:016x}:{}{IMAGE_CLOSE}", img.hash, img.tokens)
}

/// The result of one generation.
#[derive(Debug, Clone)]
pub struct GenerateOutcome {
    /// The full completion text (which the API parses for tool calls and think
    /// blocks).
    pub text: String,
    /// `stop` (a stop sequence or EOS), `length` (max_tokens), or `tool_calls`
    /// (the tool-call cap fired).
    pub finish_reason: String,
    /// Completion token count (for `usage.completion_tokens`).
    pub completion_tokens: usize,
}

/// A request's chat-template switches beyond its messages and tools, for
/// [`Engine::render_prompt`] and [`Engine::tokenize_prompt`].
#[derive(Debug, Clone, Default, PartialEq)]
pub struct PromptOptions {
    /// The thinking switch: the request's, else the dialect's default.
    pub thinking: bool,
    /// `reasoning_effort` as the client sent it; the chat template decides what it means. The
    /// dialect's lowest effort ([`crate::Dialect::thinking_off_effort`]) replaces it when the
    /// request turns thinking off or names the lowest effort ("none", "minimal").
    pub reasoning_effort: Option<String>,
    /// `clear_thinking`: drop the reasoning of assistant turns before the last user message.
    pub clear_thinking: Option<bool>,
}

/// A model backend. All methods are `&self` so a single engine serves many
/// concurrent connections without interior synchronization.
pub trait Engine {
    /// Token count of the prompt (for `usage.prompt_tokens`).
    fn tokenize(&self, messages: &[ChatMessage], tools: &[Tool], thinking: bool) -> usize;

    /// Render the chat into the model's native input (chat template).
    fn render_chat(&self, messages: &[ChatMessage], tools: &[Tool], thinking: bool) -> String;

    /// Token count of the prompt rendered with all of the request's template options (the API
    /// calls this). Default: [`Engine::tokenize`] with the thinking switch alone.
    fn tokenize_prompt(&self, messages: &[ChatMessage], tools: &[Tool], opts: &PromptOptions) -> usize {
        self.tokenize(messages, tools, opts.thinking)
    }

    /// The prompt rendered with all of the request's template options (the API calls this).
    /// Default: [`Engine::render_chat`] with the thinking switch alone.
    fn render_prompt(&self, messages: &[ChatMessage], tools: &[Tool], opts: &PromptOptions) -> String {
        self.render_chat(messages, tools, opts.thinking)
    }

    /// The longest request (prompt plus output) this deployment can hold, when it
    /// is bounded (the coordinator's KV pool); prompts at or over it are refused
    /// with 400 before generation starts.
    fn max_context(&self) -> Option<usize> {
        None
    }

    /// Whether this engine encodes images (perf reset V2). Without it the API refuses image
    /// parts with a 400.
    fn vision(&self) -> bool {
        false
    }

    /// Decode one inline image (the payload of a `data:` URL) for this engine's vision tower:
    /// its RGB8 pixels and the language-model positions it takes. The token count depends on
    /// the tower's patch geometry, so decoding lives with the engine. The API sets `hash`
    /// (FNV-1a of the URL) itself, so every engine keys images the same way. A refusal is
    /// answered 400. The default refuses: an engine without an image encoder.
    fn decode_image(&self, data_url: &str) -> Result<ImageInput, String> {
        let _ = data_url;
        Err("this server has no image encoder; image parts are rejected".into())
    }

    /// A place in the engine's request queue (perf reset V3, DS41RT v15's bounded admission),
    /// taken before the response starts: waits up to the engine's budget while the queue is full;
    /// `Err` (the queue and its waiters full, or the wait expired) is answered 429 with
    /// `Retry-After`. An engine without a queue admits everything.
    fn admit(&self) -> Result<Option<QueuePlace>, String> {
        Ok(None)
    }

    /// Whether the engine can serve requests now (`GET /health`), answered from its state: it
    /// queues nothing and waits for nothing. `Err` says why not (answered 503). Default: it can.
    fn health(&self) -> Result<(), String> {
        Ok(())
    }

    /// How long a streamed response may go without a write before the API sends an SSE comment
    /// (`: keepalive`), so that a client or proxy with an idle timeout does not cut it. Default: 15 s.
    /// The API counts from its last write to the response, whatever the model is doing: a tool call
    /// is held back until it is complete, so while the model writes one the stream would otherwise
    /// be silent. An engine that itself waits this long for the model (a long prefill) sends an
    /// empty delta ([`Engine::generate`]).
    fn keepalive(&self) -> Duration {
        Duration::from_secs(15)
    }

    /// Generate the completion. `on_delta` is called with each incremental text
    /// delta (the API forwards it as an SSE content delta); the returned text is
    /// the full completion the API parses for tool calls and think blocks. An empty
    /// delta is a keepalive, for a wait of [`Engine::keepalive`] on the model: the API
    /// writes an SSE comment for it, no event.
    fn generate(
        &self,
        prompt: &str,
        params: &GenerateParams,
        on_delta: &mut dyn FnMut(&str),
    ) -> Result<GenerateOutcome, String>;
}
