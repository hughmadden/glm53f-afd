//! The `glm53f_api::Engine` implementation over the toy model, with the scheduler on its own
//! thread: generation, streaming, stop strings held back, end-of-sequence, the bounded queue,
//! health and chat-template errors.

mod common;

use std::sync::{Arc, OnceLock};
use std::time::{Duration, Instant};

use common::*;
use glm53f_api::engine::{Engine, GenerateParams, PromptOptions};
use glm53f_api::types::{ChatMessage, Tool};
use glm53f_coordinator::{CoordinatorEngine, EngineConfig, PromptCodec, Queue, Token};

/// Letters as tokens: 'a'..'z' are ids 0..25; ids 26..31 print as '#'.
struct Letters;

impl PromptCodec for Letters {
    fn render(&self, messages: &[ChatMessage], _tools: &[Tool], opts: &PromptOptions) -> Result<String, String> {
        if let Some(m) = messages.iter().find(|m| m.role == "bad") {
            return Err(format!("no role {:?} in this template", m.role));
        }
        let body: String = messages.iter().map(|m| m.content.as_str()).collect();
        Ok(if opts.thinking { format!("t{body}") } else { body })
    }

    fn encode(&self, prompt: &str) -> Result<Vec<Token>, String> {
        Ok(prompt.bytes().map(|b| if b.is_ascii_lowercase() { (b - b'a') as Token } else { 26 }).collect())
    }

    fn decode_bytes(&self, ids: &[Token]) -> Vec<u8> {
        ids.iter().map(|&i| if i < 26 { b'a' + i as u8 } else { b'#' }).collect()
    }
}

fn text(ids: &[Token]) -> String {
    String::from_utf8(Letters.decode_bytes(ids)).unwrap()
}

fn engine(eos: Vec<Token>, queue: Arc<Queue>) -> CoordinatorEngine<Letters> {
    let dev = device(10_000_000);
    let model = MockModel::new(&dev, 8);
    let slots: Vec<MockSlot> = (0..4).map(|i| MockSlot::new(i, &dev, 64)).collect();
    let sched = test_config(eos.clone(), glm53f_coordinator::scheduler::wall_clock());
    CoordinatorEngine::start(Letters, model, slots, None, sched, queue, EngineConfig::new(eos, 100_000)).unwrap()
}

fn user(content: &str) -> Vec<ChatMessage> {
    vec![ChatMessage { role: "user".into(), content: content.into(), tool_calls: Vec::new(), reasoning_content: None,
        tool_call_id: None }]
}

fn params(max: usize, stop: Vec<String>) -> GenerateParams {
    GenerateParams { max_tokens: max, stop, ..GenerateParams::default() }
}

#[test]
fn generate_streams_the_serial_completion() {
    let e = engine(Vec::new(), Queue::new(16, Duration::from_secs(1)));
    let msgs = user("helloworld");
    let opts = PromptOptions { thinking: true, ..Default::default() };
    // The API counts the prompt, renders it, then generates it on the same thread.
    assert_eq!(e.tokenize_prompt(&msgs, &[], &opts), 11);
    let prompt = e.render_prompt(&msgs, &[], &opts);
    assert_eq!(prompt, "thelloworld");
    let mut deltas = Vec::new();
    let out = e.generate(&prompt, &params(24, Vec::new()), &mut |d| deltas.push(d.to_string())).unwrap();
    let want = reference(&Letters.encode(&prompt).unwrap(), 24, None, &[]);
    assert_eq!(out.text, text(&want));
    assert_eq!(deltas.concat(), out.text);
    assert_eq!((out.finish_reason.as_str(), out.completion_tokens), ("length", 24));
    // Again: the snapshot of the same prompt serves it, with the same text.
    let again = e.generate(&prompt, &params(24, Vec::new()), &mut |_| {}).unwrap();
    assert_eq!(again.text, out.text);
    // Sampled, seeded: reproducible.
    let sampled = GenerateParams { temperature: 0.9, seed: Some(5), ..params(20, Vec::new()) };
    let a = e.generate(&prompt, &sampled, &mut |_| {}).unwrap();
    let b = e.generate(&prompt, &sampled, &mut |_| {}).unwrap();
    assert_eq!(a.text, b.text);
    let s = glm53f_coordinator::Sampling::new(0.9, 1.0, 0, 0.0, Some(5)).unwrap();
    assert_eq!(a.text, text(&reference(&Letters.encode(&prompt).unwrap(), 20, s, &[])));
}

#[test]
fn stop_strings_end_the_completion_and_are_never_streamed() {
    let e = engine(Vec::new(), Queue::new(16, Duration::from_secs(1)));
    let prompt = "abcabcabc".to_string();
    let full = text(&reference(&Letters.encode(&prompt).unwrap(), 30, None, &[]));
    // A stop string from inside the completion.
    let stop = full[6..9].to_string();
    let cut = full.find(&stop).unwrap();
    let mut deltas = Vec::new();
    let out = e.generate(&prompt, &params(30, vec![stop.clone()]), &mut |d| deltas.push(d.to_string())).unwrap();
    assert_eq!(out.text, full[..cut]);
    assert_eq!(out.finish_reason, "stop");
    assert!(!deltas.concat().contains(&stop));
    assert_eq!(deltas.concat(), out.text);
}

#[test]
fn an_end_of_sequence_token_stops_with_finish_reason_stop() {
    let prompt = "zyxwvu".to_string();
    let ids = Letters.encode(&prompt).unwrap();
    let full = reference(&ids, 40, None, &[]);
    let eos = full[5];
    let e = engine(vec![eos], Queue::new(16, Duration::from_secs(1)));
    let out = e.generate(&prompt, &params(40, Vec::new()), &mut |_| {}).unwrap();
    let first = full.iter().position(|&t| t == eos).unwrap();
    assert_eq!(out.text, text(&full[..first]));
    assert_eq!((out.finish_reason.as_str(), out.completion_tokens), ("stop", first + 1));
}

#[test]
fn a_full_queue_is_refused_before_the_response_starts() {
    let e = engine(Vec::new(), Queue::new(1, Duration::from_millis(0)));
    let place = e.admit().unwrap();
    assert!(place.is_some());
    assert!(e.admit().is_err(), "the API answers this with 429");
    drop(place);
    assert!(e.admit().is_ok());
}

/// `Engine::health` (the API's `GET /health`): serving while the scheduler runs and the check the
/// daemon adds passes (the expert wire's state), with the queue full or not; failing, with the
/// reason, once the check fails, or once the scheduler's thread has ended (a pass panicked).
#[test]
fn health_follows_the_scheduler_and_the_added_check() {
    let wire: Arc<OnceLock<String>> = Arc::new(OnceLock::new());
    let w = wire.clone();
    let e = engine(Vec::new(), Queue::new(1, Duration::from_millis(0)))
        .with_health(move || w.get().map_or(Ok(()), |m| Err(m.clone())));
    let place = e.admit().unwrap();
    assert!(e.admit().is_err(), "the queue is full");
    assert_eq!(e.health(), Ok(()));
    drop(place);
    wire.set("expert wire: rank 1 closed its connection".to_string()).unwrap();
    assert_eq!(e.health(), Err("expert wire: rank 1 closed its connection".to_string()));

    let dev = device(10_000_000);
    let mut model = MockModel::new(&dev, 8);
    model.panic_in_prefill = true;
    let slots: Vec<MockSlot> = (0..4).map(|i| MockSlot::new(i, &dev, 64)).collect();
    let sched = test_config(Vec::new(), glm53f_coordinator::scheduler::wall_clock());
    let queue = Queue::new(16, Duration::from_secs(1));
    let e = CoordinatorEngine::start(Letters, model, slots, None, sched, queue, EngineConfig::new(Vec::new(), 100_000))
        .unwrap();
    assert_eq!(e.health(), Ok(()));
    let err = e.generate("abc", &params(4, Vec::new()), &mut |_| {}).unwrap_err();
    assert_eq!(err, "scheduler stopped");
    let t0 = Instant::now();
    while e.health().is_ok() {
        assert!(t0.elapsed() < Duration::from_secs(10), "the scheduler's thread did not end");
        std::thread::sleep(Duration::from_millis(5));
    }
    let why = e.health().unwrap_err();
    assert!(why.starts_with("the scheduler stopped"), "{why}");
}

#[test]
fn template_errors_come_back_from_generate() {
    let e = engine(Vec::new(), Queue::new(16, Duration::from_secs(1)));
    let mut msgs = user("abc");
    msgs[0].role = "bad".into();
    let opts = PromptOptions::default();
    assert_eq!(e.tokenize_prompt(&msgs, &[], &opts), 0);
    let prompt = e.render_prompt(&msgs, &[], &opts);
    let err = e.generate(&prompt, &params(5, Vec::new()), &mut |_| {}).unwrap_err();
    assert!(err.starts_with("chat template: no role"), "{err}");
}

#[test]
fn max_context_is_what_one_slot_can_grow_to() {
    let dev = device(1_000_000);
    let model = MockModel::new(&dev, 0);
    let slot = MockSlot::new(0, &dev, 64);
    // 1,000,000 B less the slot's 6,400: 9,936 more tokens, rounded down to whole 64s (9,920) plus
    // the 64 the slot holds, less the admission's 32 + 8 allowance.
    let got = glm53f_coordinator::engine::max_context(&model, &slot, 40, 1 << 20);
    assert_eq!(got, Some(9_984 - 40));
    assert_eq!(glm53f_coordinator::engine::max_context(&model, &slot, 40, 1000), Some(1000));
}

/// The engine plugs into `glm53f_api::serve` (a compile-time check: it is an `Engine` that can be
/// shared between the server's connection threads), with GLM-5.3-Flash's codec as with any other.
#[test]
fn the_engine_can_be_served() {
    fn servable<E: Engine + Send + Sync + 'static>() {}
    servable::<CoordinatorEngine<Letters>>();
    servable::<CoordinatorEngine<glm53f_coordinator::GlmPrompts>>();
    let _serve = |e: Arc<CoordinatorEngine<glm53f_coordinator::GlmPrompts>>| {
        glm53f_api::serve("127.0.0.1:0", e, Arc::new(glm53f_api::dialect::GlmDialect))
    };
}
