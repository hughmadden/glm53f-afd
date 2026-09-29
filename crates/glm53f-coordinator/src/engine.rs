//! The [`glm53f_api::Engine`] implementation: the API's requests become scheduler jobs.
//!
//! The engine owns the text side and the queue; the scheduler (on its own thread) owns the model.
//! What depends on the model's text format sits behind [`PromptCodec`]: the chat template, the
//! tokenizer, token decoding, and image decoding when there is a vision tower
//! (`crate::glm_prompt::GlmPrompts` is GLM-5.3-Flash's).
//!
//! `generate` turns a rendered prompt into token ids, sends a [`Job`] and streams the tokens it
//! gets back as text: stop strings are matched in the engine, and text that could still become
//! a stop string (or an incomplete UTF-8 character) is held back until it resolves
//! ([`crate::streaming::flush_pending`]). While a token is slow to come (a long prefill) an empty
//! delta every 15 s keeps the client's connection alive; a client that left ends the request,
//! which frees its slot and keeps its prefill as a snapshot for a retry.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{mpsc, Arc, Mutex};
use std::time::Duration;

use glm53f_api::engine::{
    Engine, GenerateOutcome, GenerateParams, ImageInput, PromptOptions, QueuePlace, IMAGE_CLOSE, IMAGE_OPEN,
};
use glm53f_api::types::{ChatMessage, Tool};

use crate::hostcache::HostCache;
use crate::model::{ImageSpan, KvSlot, ModelForward, Token, IMAGE_ID_BASE};
use crate::queue::Queue;
use crate::sampling::Sampling;
use crate::scheduler::{Job, Scheduler, SchedulerConfig};
use crate::streaming::flush_pending;

/// A model's text format: chat template, tokenizer and (optionally) image decoding.
pub trait PromptCodec: Send + Sync + 'static {
    /// The conversation rendered with the model's chat template, ready to generate from.
    fn render(&self, messages: &[ChatMessage], tools: &[Tool], opts: &PromptOptions) -> Result<String, String>;

    /// Token ids of a rendered prompt. Images stand in it as the API's markers
    /// (`IMAGE_OPEN` hash `:` tokens `IMAGE_CLOSE`); each becomes its span of image ids
    /// ([`expand_image_marker`]).
    fn encode(&self, prompt: &str) -> Result<Vec<Token>, String>;

    /// The bytes generated ids decode to, as the client sees them.
    fn decode_bytes(&self, ids: &[Token]) -> Vec<u8>;

    /// Whether this deployment encodes images. Default: no.
    fn vision(&self) -> bool {
        false
    }

    /// Decode one inline image (see [`Engine::decode_image`]). Default: refused.
    fn decode_image(&self, data_url: &str) -> Result<ImageInput, String> {
        let _ = data_url;
        Err("this server has no image encoder; image parts are rejected".into())
    }
}

/// The token id standing for row `i` of the image with hash `hash`: past any vocabulary (bit 31
/// set, [`IMAGE_ID_BASE`]), and a function of the image's bytes, so the prefix index, which
/// compares token ids, shares a prompt only when its images are the same.
pub fn image_token_id(hash: u64, i: usize) -> Token {
    let mut x = hash ^ (i as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);
    x ^= x >> 33;
    x = x.wrapping_mul(0xff51_afd7_ed55_8ccd);
    x ^= x >> 33;
    x = x.wrapping_mul(0xc4ce_b9fe_1a85_ec53);
    x ^= x >> 33;
    IMAGE_ID_BASE | (x as u32 & 0x7fff_ffff)
}

/// The image ids an image marker's inner text (`hash:tokens`, hexadecimal hash) stands for.
pub fn expand_image_marker(inner: &str) -> Result<Vec<Token>, String> {
    let (h, n) = inner.split_once(':').ok_or_else(|| format!("malformed image marker {inner:?}"))?;
    let hash = u64::from_str_radix(h, 16).map_err(|_| format!("malformed image marker {inner:?}"))?;
    let n: usize = n.parse().map_err(|_| format!("malformed image marker {inner:?}"))?;
    Ok((0..n).map(|k| image_token_id(hash, k)).collect())
}

/// Where each image of a request sits in its prompt: the runs of image ids, matched in order
/// against the request's images by length and first id.
pub fn image_spans(ids: &[Token], images: &[Arc<ImageInput>]) -> Result<Vec<ImageSpan>, String> {
    let mut spans = Vec::new();
    let mut i = 0;
    while i < ids.len() {
        if ids[i] < IMAGE_ID_BASE {
            i += 1;
            continue;
        }
        let start = i;
        while i < ids.len() && ids[i] >= IMAGE_ID_BASE {
            i += 1;
        }
        let Some(img) = images.get(spans.len()) else { return Err("the prompt has more images than the request".into()) };
        if i - start != img.tokens || ids[start] != image_token_id(img.hash, 0) {
            return Err(format!("image {} does not match its place in the prompt", spans.len()));
        }
        spans.push(ImageSpan { start, image: img.clone() });
    }
    if spans.len() != images.len() {
        return Err(format!("{} images in the request, {} in the prompt", images.len(), spans.len()));
    }
    Ok(spans)
}

/// Call the API delta callback, catching a panic (the SSE write path can panic on bad input)
/// so the request ends cleanly instead of unwinding through held locks and poisoning them.
fn emit_delta(on_delta: &mut dyn FnMut(&str), text: &str) -> Result<(), String> {
    let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| on_delta(text)));
    r.map_err(|_| "on_delta panicked".to_string())
}

/// Sets a job's cancel flag when the caller stops reading (stop string, client gone, error), so
/// the scheduler frees its slot at the next step.
struct CancelOnDrop(Arc<AtomicBool>);

impl Drop for CancelOnDrop {
    fn drop(&mut self) {
        self.0.store(true, Ordering::Relaxed);
    }
}

/// What [`Engine::render_prompt`] returns when the chat template refuses a conversation (it cannot
/// return an error): this tag, then the reason. `generate` answers it with that error. The tag is
/// written with the API's image-marker noncharacters, which never reach a prompt from client text.
fn render_error(e: &str) -> String {
    format!("{IMAGE_OPEN}render error{IMAGE_CLOSE}{e}")
}

fn render_error_of(prompt: &str) -> Option<&str> {
    prompt.strip_prefix(IMAGE_OPEN)?.strip_prefix("render error")?.strip_prefix(IMAGE_CLOSE)
}

thread_local! {
    /// The prompt this connection's thread last tokenized to count it: the API counts a prompt's
    /// tokens and then generates it on the same thread, so generation reuses the ids instead of
    /// encoding again (about 0.35 s at 512K tokens in the source).
    static LAST_ENCODE: std::cell::RefCell<Option<(String, Vec<Token>)>> = const { std::cell::RefCell::new(None) };
}

/// The engine's knobs.
#[derive(Clone, Debug)]
pub struct EngineConfig {
    /// Tokens that end a completion (`finish_reason` "stop").
    pub eos: Vec<Token>,
    /// The largest `max_tokens` a request gets.
    pub max_tokens: usize,
    /// The model's context limit (GLM-5.3-Flash: 1,048,576).
    pub model_max_context: usize,
    /// How long a caller waits for a token before sending a keep-alive delta. It is also how long
    /// a stream may write nothing before the API writes a keepalive of its own
    /// ([`Engine::keepalive`]).
    pub keepalive: Duration,
}

impl EngineConfig {
    pub fn new(eos: Vec<Token>, model_max_context: usize) -> Self {
        EngineConfig { eos, max_tokens: 65_536, model_max_context, keepalive: Duration::from_secs(15) }
    }
}

/// The longest request (prompt plus the admission's output allowance `reserve`) a slot can hold
/// with the pool idle: the largest `t` with `need_bytes(t)` within the free memory, less
/// `reserve`, at most `cap`. None when the free memory is unknown.
pub fn max_context<M: ModelForward>(model: &M, slot: &M::Slot, reserve: usize, cap: usize) -> Option<usize> {
    let free = model.free_bytes().ok()?;
    let top = cap.saturating_add(reserve);
    if slot.need_bytes(top) <= free {
        return Some(cap);
    }
    let (mut lo, mut hi) = (0usize, top);
    while lo < hi {
        let mid = lo + (hi - lo).div_ceil(2);
        if slot.need_bytes(mid) <= free {
            lo = mid;
        } else {
            hi = mid - 1;
        }
    }
    Some(lo.saturating_sub(reserve).min(cap))
}

/// The coordinator engine: a prompt codec, the bounded queue and the scheduler's job channel.
pub struct CoordinatorEngine<C: PromptCodec> {
    codec: C,
    jobs: Mutex<mpsc::Sender<Job>>,
    queue: Arc<Queue>,
    max_context: Option<usize>,
    cfg: EngineConfig,
    /// The scheduler's thread ([`CoordinatorEngine::start`]; none over a caller's channel).
    scheduler: Option<std::thread::JoinHandle<()>>,
    /// What else [`Engine::health`] asks ([`CoordinatorEngine::with_health`]).
    health: Option<HealthCheck>,
}

/// A check of state the scheduler's model holds (the expert wire's), read from another thread.
type HealthCheck = Box<dyn Fn() -> Result<(), String> + Send + Sync>;

impl<C: PromptCodec> CoordinatorEngine<C> {
    /// Start the scheduler on its own thread over `model` and its `slots` (all empty), with an
    /// optional RAM tier. The maximum context is measured first, from the first slot and the
    /// free device memory.
    pub fn start<M>(codec: C, model: M, slots: Vec<M::Slot>, cache: Option<HostCache>, sched: SchedulerConfig,
        queue: Arc<Queue>, cfg: EngineConfig) -> Result<Self, String>
    where
        M: ModelForward + Send + 'static,
        M::Slot: Send + 'static,
        <M::Slot as KvSlot>::Mark: Send + 'static,
    {
        let first = slots.first().ok_or("no KV slots")?;
        let reserve = sched.out_max + sched.out_slack;
        let max_context = max_context(&model, first, reserve, cfg.model_max_context);
        eprintln!("[coordinator] batching scheduler: {} slots; max context {max_context:?} tokens per request; queue \
            depth {}", slots.len(), queue.depth());
        let (tx, rx) = mpsc::channel();
        let scheduler = std::thread::Builder::new()
            .name("glm53f-scheduler".into())
            .spawn(move || Scheduler::new(model, slots, cache, sched, rx).run())
            .map_err(|e| format!("spawn scheduler: {e}"))?;
        Ok(CoordinatorEngine { scheduler: Some(scheduler), ..Self::with_channel(codec, tx, queue, max_context, cfg) })
    }

    /// An engine over an existing job channel (a scheduler the caller runs).
    pub fn with_channel(codec: C, jobs: mpsc::Sender<Job>, queue: Arc<Queue>, max_context: Option<usize>,
        cfg: EngineConfig) -> Self {
        CoordinatorEngine { codec, jobs: Mutex::new(jobs), queue, max_context, cfg, scheduler: None, health: None }
    }

    /// Adds `check` to [`Engine::health`]: state the model holds on the scheduler's thread and
    /// shares for reading (`glm53f-serve`: whether the expert wire has failed). It must not wait.
    pub fn with_health(mut self, check: impl Fn() -> Result<(), String> + Send + Sync + 'static) -> Self {
        self.health = Some(Box::new(check));
        self
    }

    pub fn codec(&self) -> &C {
        &self.codec
    }

    /// `prompt`'s token ids, from [`LAST_ENCODE`] when it is the prompt just counted.
    fn encode_prompt(&self, prompt: &str) -> Result<Vec<Token>, String> {
        let hit = LAST_ENCODE.with(|c| c.borrow_mut().take().filter(|(p, _)| p == prompt).map(|(_, ids)| ids));
        match hit {
            Some(ids) => Ok(ids),
            None => self.codec.encode(prompt),
        }
    }
}

impl<C: PromptCodec> Engine for CoordinatorEngine<C> {
    fn tokenize(&self, messages: &[ChatMessage], tools: &[Tool], thinking: bool) -> usize {
        self.tokenize_prompt(messages, tools, &PromptOptions { thinking, ..Default::default() })
    }

    fn render_chat(&self, messages: &[ChatMessage], tools: &[Tool], thinking: bool) -> String {
        self.render_prompt(messages, tools, &PromptOptions { thinking, ..Default::default() })
    }

    fn tokenize_prompt(&self, messages: &[ChatMessage], tools: &[Tool], opts: &PromptOptions) -> usize {
        let encoded = self.codec.render(messages, tools, opts).and_then(|r| self.codec.encode(&r).map(|ids| (r, ids)));
        match encoded {
            Ok((rendered, ids)) => {
                let n = ids.len();
                LAST_ENCODE.with(|c| *c.borrow_mut() = Some((rendered, ids)));
                n
            }
            // The error comes back from `generate`.
            Err(_) => 0,
        }
    }

    fn render_prompt(&self, messages: &[ChatMessage], tools: &[Tool], opts: &PromptOptions) -> String {
        match self.codec.render(messages, tools, opts) {
            Ok(s) => s,
            Err(e) => render_error(&e),
        }
    }

    fn max_context(&self) -> Option<usize> {
        self.max_context
    }

    fn vision(&self) -> bool {
        self.codec.vision()
    }

    fn decode_image(&self, data_url: &str) -> Result<ImageInput, String> {
        self.codec.decode_image(data_url)
    }

    fn admit(&self) -> Result<Option<QueuePlace>, String> {
        self.queue.admit().map(Some)
    }

    /// Serving while the scheduler's thread runs (it ends only by a panic: the engine holds its
    /// job channel open) and the added check passes; the queue and the model are not touched.
    fn health(&self) -> Result<(), String> {
        if self.scheduler.as_ref().is_some_and(|t| t.is_finished()) {
            return Err("the scheduler stopped (see the log); restart the coordinator".into());
        }
        self.health.as_ref().map_or(Ok(()), |check| check())
    }

    /// The wait for a token after which `generate` sends an empty delta, which is also how long a
    /// stream may be silent before the API writes its own keepalive.
    fn keepalive(&self) -> Duration {
        self.cfg.keepalive
    }

    fn generate(&self, prompt: &str, params: &GenerateParams, on_delta: &mut dyn FnMut(&str))
        -> Result<GenerateOutcome, String> {
        if let Some(e) = render_error_of(prompt) {
            return Err(format!("chat template: {e}"));
        }
        let ids = self.encode_prompt(prompt)?;
        let cancel = params.cancel.clone().unwrap_or_else(|| Arc::new(AtomicBool::new(false)));
        let sampling = Sampling::new(params.temperature as f32, params.top_p as f32, params.top_k, params.min_p as f32,
            params.seed)?;
        let max = params.max_tokens.min(self.cfg.max_tokens).max(1);
        let images = image_spans(&ids, &params.images)?;
        let (tx, rx) = mpsc::channel();
        let place = params.place.lock().unwrap_or_else(|p| p.into_inner()).take();
        self.jobs
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .send(Job { ids, max_tokens: max, tx, cancel: cancel.clone(), images, sampling, place })
            .map_err(|_| "scheduler stopped".to_string())?;
        let _cancel = CancelOnDrop(cancel.clone());
        let keepalive = self.cfg.keepalive;
        let next_token = |on_delta: &mut dyn FnMut(&str)| -> Result<Token, String> {
            loop {
                match rx.recv_timeout(keepalive) {
                    Ok(r) => return r,
                    Err(mpsc::RecvTimeoutError::Timeout) => {
                        emit_delta(on_delta, "")?;
                        if cancel.load(Ordering::Relaxed) {
                            return Err("client gone".to_string());
                        }
                    }
                    Err(mpsc::RecvTimeoutError::Disconnected) => return Err("scheduler stopped".to_string()),
                }
            }
        };
        let decode = |ids: &[u32]| self.codec.decode_bytes(ids);
        let mut next = next_token(on_delta)?;
        let mut generated = 1usize;
        // `pending` holds token ids whose text could still be a stop prefix, so a stop sequence is
        // never emitted early.
        let mut full_text = String::new();
        let mut pending: Vec<u32> = Vec::new();
        let mut finish_reason = "length";
        loop {
            if self.cfg.eos.contains(&next) {
                finish_reason = "stop";
                break;
            }
            pending.push(next);
            let (emit, stopped) = flush_pending(&decode, &params.stop, &mut pending);
            if !emit.is_empty() {
                emit_delta(on_delta, &emit)?;
                full_text.push_str(&emit);
            }
            if stopped {
                finish_reason = "stop";
                break;
            }
            if generated >= max {
                break;
            }
            if cancel.load(Ordering::Relaxed) {
                return Err("client gone".to_string());
            }
            next = next_token(on_delta)?;
            generated += 1;
        }
        // Flush a held-back tail (a partial stop that never completed, or the last token before
        // the end).
        let tail = String::from_utf8_lossy(&decode(&pending)).into_owned();
        if !tail.is_empty() {
            emit_delta(on_delta, &tail)?;
            full_text.push_str(&tail);
        }
        Ok(GenerateOutcome { text: full_text, finish_reason: finish_reason.to_string(), completion_tokens: generated })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A panicking on_delta must be caught (return Err), not unwind through held locks.
    #[test]
    fn emit_catches_a_panicking_callback() {
        let r = emit_delta(&mut |_| panic!("boom"), "x");
        assert_eq!(r, Err("on_delta panicked".to_string()));
        assert!(emit_delta(&mut |_| {}, "ok").is_ok());
    }

    /// A poisoned mutex guard must be recoverable via into_inner.
    #[test]
    fn poisoned_guard_is_recovered() {
        let m = std::sync::Mutex::new(0i32);
        let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _g = m.lock().unwrap();
            panic!("poison");
        }));
        assert!(m.is_poisoned());
        let g = m.lock().unwrap_or_else(|p| p.into_inner());
        assert_eq!(*g, 0);
    }

    #[test]
    fn render_errors_round_trip_and_never_match_a_marker() {
        let p = render_error("arguments of tool call \"f\" are not a JSON object");
        assert_eq!(render_error_of(&p), Some("arguments of tool call \"f\" are not a JSON object"));
        let marker = format!("{IMAGE_OPEN}{:016x}:{}{IMAGE_CLOSE}", 7u64, 4);
        assert_eq!(render_error_of(&marker), None);
        assert_eq!(render_error_of("[gMASK]<sop>hello"), None);
    }

    fn img(hash: u64, tokens: usize) -> Arc<ImageInput> {
        Arc::new(ImageInput { hash, tokens, width: 64, height: 64, rgb: Vec::new() })
    }

    #[test]
    fn image_token_ids_are_past_the_vocabulary_and_follow_the_image() {
        let a: Vec<u32> = (0..64).map(|i| image_token_id(7, i)).collect();
        let b: Vec<u32> = (0..64).map(|i| image_token_id(8, i)).collect();
        assert!(a.iter().chain(&b).all(|&t| t >= IMAGE_ID_BASE));
        assert_ne!(a, b, "different images, different ids");
        assert_eq!(a, (0..64).map(|i| image_token_id(7, i)).collect::<Vec<_>>(), "same image, same ids");
        assert_eq!(expand_image_marker(&format!("{:x}:64", 7u64)).unwrap(), a);
        assert!(expand_image_marker("zz:3").is_err());
        assert!(expand_image_marker("7").is_err());
    }

    #[test]
    fn image_spans_locate_each_image_in_order() {
        let (x, y) = (img(1, 4), img(2, 3));
        let mut ids: Vec<u32> = vec![10, 11, 151_652];
        ids.extend((0..4).map(|i| image_token_id(1, i)));
        ids.extend([151_653, 12, 151_652]);
        ids.extend((0..3).map(|i| image_token_id(2, i)));
        ids.extend([151_653, 13]);
        let spans = image_spans(&ids, &[x.clone(), y.clone()]).unwrap();
        assert_eq!(spans.iter().map(|s| s.start).collect::<Vec<_>>(), vec![3, 10]);
        // Order, count and length mismatches are errors, never a silent misplacement.
        assert!(image_spans(&ids, &[y.clone(), x.clone()]).is_err());
        assert!(image_spans(&ids, &[x.clone()]).is_err());
        assert!(image_spans(&ids, &[x, img(2, 4)]).is_err());
    }
}
