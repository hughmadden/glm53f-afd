//! The batching scheduler (mimo26f-afd perf reset W4): owns the forward and every slot on one
//! thread. Between steps it admits queued requests (each prefilled on its own slot, in segments),
//! then runs one step for every running request: a batched decode, or with a drafter a
//! speculative step that drafts, verifies every request's window in one pass and emits the
//! accepted run plus the target's next token.
//!
//! **Admission** reserves a request's prompt plus an output allowance
//! (`prompt + clamp(max_tokens, 1,024, 8,192) + 64` tokens), not the model's maximum context.
//! A request that does not fit while others run waits for their memory, first in first out;
//! one that does not fit on an idle device is refused. The bounded queue in front of the
//! scheduler (`crate::queue`) answers 429 before a response starts.
//!
//! **Prefill** (perf reset Q1): segments round robin over the prompts in flight until a round
//! has spent about `segment_ms` (2 s), then one step for the running requests. A burst of short
//! prompts prefills in one pass (perf reset B1); a long prompt gets one segment per round, in
//! multiples of `seg_quantum` tokens sized from the measured rate.
//!
//! **KV reuse** (perf reset K3, `crate::pool`): every prompt end and completion end of 512+
//! tokens is kept on the device as a snapshot point; a new prompt resumes at its longest exact
//! point, on the device (in place or forked) or restored from RAM, else prefills cold. Only
//! pressure (a bank over its size, no free slot, or not enough device memory) copies snapshots
//! to RAM. A prompt that extends one still prefilling waits for it and then forks its point.
//!
//! **Speculation** (perf reset S1 and S2, DS41RT's sample-and-match): drafts are verified in one
//! target pass per step; a draft is accepted while it equals the target's own pick at that
//! row (the argmax, or the request's seeded draw at that row's emitted position), and the first
//! mismatch emits the target's pick. The slot then commits exactly the rows of the tokens it
//! delivered ([`crate::model::ModelForward::commit`]). Each request's drafts are cut by the
//! policy ([`SpecPolicy`]), and the step's rows are held to a budget
//! ([`SchedulerConfig::spec_max_rows`], [`crate::spec::budget`]): under load the rows go to the
//! drafts most likely to be kept.
//!
//! Every row type joins the same step: greedy, sampled and (with masks) constrained rows carry
//! their own [`Pick`].

use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{mpsc, Arc};

use glm53f_api::engine::QueuePlace;

use crate::hostcache::{HostCache, Kind};
use crate::model::{DecodeRow, DraftRow, ImageSpan, KvSlot, Limits, ModelForward, Pick, Segment, Token, Window};
use crate::pool::{Active, Pool, PoolStats, Prefilling};
use crate::sampling::{After, Sampling};
use crate::spec::SpecPolicy;

/// A monotonic clock in seconds (injectable, so tests can run on simulated time).
pub type Clock = Arc<dyn Fn() -> f64 + Send + Sync>;

/// The wall clock.
pub fn wall_clock() -> Clock {
    let t0 = std::time::Instant::now();
    Arc::new(move || t0.elapsed().as_secs_f64())
}

/// One generation for the scheduler: the prompt, the token budget, the channel each sampled
/// token (or an error) goes back on, and the caller's cancel flag.
pub struct Job {
    pub ids: Vec<Token>,
    pub max_tokens: usize,
    pub tx: mpsc::Sender<Result<Token, String>>,
    pub cancel: Arc<AtomicBool>,
    pub images: Vec<ImageSpan>,
    /// None: greedy.
    pub sampling: Option<Sampling>,
    /// Its place in the bounded queue, given back when the scheduler takes it.
    pub place: Option<QueuePlace>,
}

impl Job {
    /// A job without images or queue place, and the receiver its tokens arrive on.
    pub fn new(ids: Vec<Token>, max_tokens: usize, sampling: Option<Sampling>)
        -> (Job, mpsc::Receiver<Result<Token, String>>) {
        let (tx, rx) = mpsc::channel();
        let job = Job { ids, max_tokens, tx, cancel: Arc::new(AtomicBool::new(false)), images: Vec::new(), sampling, place: None };
        (job, rx)
    }
}

/// The scheduler's knobs.
#[derive(Clone)]
pub struct SchedulerConfig {
    /// Tokens that end a request (GLM-5.3-Flash: 154,820, 154,827, 154,829).
    pub eos: Vec<Token>,
    /// Snapshots shorter than this are neither retained nor stored (the design's
    /// `--host-cache-min-tokens`).
    pub min_retain: usize,
    /// Device points per bank, prompt and turn (the design's `--prefix-cache-entries`).
    pub bank: usize,
    /// The longest a running request waits for its next step while prompts prefill.
    pub segment_ms: f64,
    /// Prefill segments are whole multiples of this (8,192: two 4,096-row chunks in MiMo).
    pub seg_quantum: usize,
    /// The longest prefill segment.
    pub seg_max: usize,
    /// The output allowance admission reserves: `clamp(max_tokens, out_min, out_max) + out_slack`.
    pub out_min: usize,
    pub out_max: usize,
    pub out_slack: usize,
    /// Speculate when the model has a drafter.
    pub spec: bool,
    pub policy: SpecPolicy,
    /// The most verify rows a speculative step holds, every request's window together
    /// (`crate::spec::budget`: past it the least likely drafts are dropped); 0 for no budget. A
    /// forward's verify pass must hold this many (or the slots' windows, if fewer).
    pub spec_max_rows: usize,
    /// Granularity of the prefix index's shared-prefix count (4: GLM-5.3-Flash's indexer pool).
    pub granularity: usize,
    pub clock: Clock,
}

impl SchedulerConfig {
    /// The defaults: the source's values.
    pub fn new(eos: Vec<Token>) -> Self {
        SchedulerConfig {
            eos,
            min_retain: 512,
            bank: 24,
            segment_ms: 2000.0,
            seg_quantum: 8192,
            seg_max: 65536,
            out_min: 1024,
            out_max: 8192,
            out_slack: 64,
            spec: true,
            policy: SpecPolicy::default(),
            spec_max_rows: crate::spec::MAX_VERIFY_ROWS,
            granularity: 4,
            clock: wall_clock(),
        }
    }

    /// [`Self::new`] with `GLM53F_PREFILL_SEGMENT_MS`, `GLM53F_SPEC` (0: off),
    /// `GLM53F_PREFIX_CACHE_ENTRIES`, the speculation policy ([`SpecPolicy::from_env`]) and its
    /// row budget (`GLM53F_SPEC_MAX_ROWS`, 0 for none).
    pub fn from_env(eos: Vec<Token>) -> Self {
        let mut c = Self::new(eos);
        let var = |k: &str| std::env::var(k).ok();
        if let Some(v) = var("GLM53F_PREFILL_SEGMENT_MS").and_then(|v| v.parse().ok()) {
            c.segment_ms = v;
        }
        c.spec = var("GLM53F_SPEC").map(|v| v != "0").unwrap_or(true);
        if let Some(v) = var("GLM53F_PREFIX_CACHE_ENTRIES").and_then(|v| v.parse().ok()) {
            c.bank = v;
        }
        c.policy = SpecPolicy::from_env();
        if let Some(v) = var("GLM53F_SPEC_MAX_ROWS").and_then(|v| v.parse().ok()) {
            c.spec_max_rows = v;
        }
        c
    }

    /// Tokens admission reserves for a request (the source's `admit_rows`): the prompt, at
    /// least `out_min` output tokens (so a short follow-up turn resumes in its retained slot
    /// without growing it), at most `out_max` (it grows per step after that), and a verify block.
    pub fn admit_rows(&self, prompt: usize, max_tokens: usize) -> usize {
        prompt + max_tokens.clamp(self.out_min, self.out_max) + self.out_slack
    }
}

/// Step counters for logs and tests.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SchedStats {
    /// Prompt tokens run through prefill passes.
    pub prefill_tokens: u64,
    /// Prefill passes (a batched pass counts once).
    pub prefill_passes: u64,
    /// Decode passes and the rows they carried.
    pub decode_steps: u64,
    pub decode_rows: u64,
    /// Speculative steps, drafts verified and drafts accepted.
    pub spec_steps: u64,
    pub drafts_verified: u64,
    pub drafts_accepted: u64,
    /// Requests refused at admission and requests that waited for memory.
    pub refused: u64,
    pub stalls: u64,
    /// Prompts deferred behind an identical prefill.
    pub deferred: u64,
}

/// Whether prompt `ids` starts with the in-flight prompt `inflight` (worth waiting for: its
/// snapshot will cover `inflight.len()` tokens of `ids`).
fn extends(ids: &[Token], inflight: &[Token], min_retain: usize) -> bool {
    inflight.len() >= min_retain && ids.len() >= inflight.len() && ids[..inflight.len()] == inflight[..]
}

/// A prefilled request starts decoding with its first token `next`; its prompt-end snapshot
/// (with `p.after`) joins the device bank (a device copy, no RAM traffic).
fn start<S: KvSlot>(pool: &mut Pool<S>, active: &mut Vec<Active<S>>, p: Prefilling<S>, next: Token, eos: &[Token]) {
    let Prefilling { mut slot, ids, mut points, max, tx, cancel, after, sampling, .. } = p;
    // The prefill is done: its working set goes back (outside the forward).
    slot.end_prefill();
    let plen = ids.len();
    if let Some(after) = after.filter(|_| plen >= pool.min_retain && points.iter().all(|x| x.len != plen)) {
        let now = pool.now();
        points.extend(pool.save_point(&slot, &ids, after, Kind::Prompt, now));
    }
    let done = eos.contains(&next) || max <= 1;
    let mut hist = ids;
    hist.push(next);
    let a = Active { slot, last: next, generated: 1, max, tx, cancel, hist, points, sampling };
    if a.tx.send(Ok(next)).is_err() || done {
        pool.retire(a);
    } else {
        active.push(a);
    }
}

/// The scheduler: the forward, the slot pool and every request in flight.
pub struct Scheduler<M: ModelForward> {
    model: M,
    cfg: SchedulerConfig,
    limits: Limits,
    pool: Pool<M::Slot>,
    active: Vec<Active<M::Slot>>,
    prefilling: VecDeque<Prefilling<M::Slot>>,
    deferred: VecDeque<Job>,
    /// A job that did not fit while others ran: it waits for their memory, first in first out,
    /// instead of being refused.
    stalled: Option<Job>,
    rx: mpsc::Receiver<Job>,
    /// The prefill rate at the current context length, which sizes the next segment.
    sec_per_token: f64,
    spec: bool,
    pub stats: SchedStats,
}

impl<M: ModelForward> Scheduler<M> {
    /// A scheduler over `model` with `slots` (the pool's slots, all empty) and an optional RAM
    /// tier, taking jobs from `rx`.
    pub fn new(model: M, slots: Vec<M::Slot>, cache: Option<HostCache>, cfg: SchedulerConfig, rx: mpsc::Receiver<Job>)
        -> Self {
        let limits = model.limits();
        let spec = cfg.spec && limits.block > 1;
        let pool = Pool::new(slots, cache, cfg.bank, cfg.min_retain, cfg.granularity);
        eprintln!("[coordinator] decode: {}; device snapshot banks {} prompt + {} turn",
            if spec { format!("speculative (block {})", limits.block) } else { "one token per step".to_string() },
            cfg.bank, cfg.bank);
        Scheduler {
            model,
            cfg,
            limits,
            pool,
            active: Vec::new(),
            prefilling: VecDeque::new(),
            deferred: VecDeque::new(),
            stalled: None,
            rx,
            sec_per_token: 1.0 / 4000.0,
            spec,
            stats: SchedStats::default(),
        }
    }

    /// Run until the job channel closes.
    pub fn run(mut self) {
        while self.step(true) {}
    }

    pub fn model(&self) -> &M {
        &self.model
    }

    pub fn model_mut(&mut self) -> &mut M {
        &mut self.model
    }

    pub fn pool_stats(&self) -> PoolStats {
        self.pool.stats
    }

    pub fn host_cache(&self) -> Option<&HostCache> {
        self.pool.cache.as_ref()
    }

    /// Requests decoding, prefilling, deferred and stalled.
    pub fn in_flight(&self) -> (usize, usize, usize, usize) {
        (self.active.len(), self.prefilling.len(), self.deferred.len(), usize::from(self.stalled.is_some()))
    }

    /// Free and retained slots.
    pub fn slots(&self) -> (usize, usize) {
        (self.pool.free.len(), self.pool.retained.len())
    }

    /// Snapshot points indexed on the device (running and retained).
    pub fn device_points(&self) -> usize {
        self.pool.indexed()
    }

    /// Nothing in flight.
    pub fn is_idle(&self) -> bool {
        self.active.is_empty() && self.prefilling.is_empty() && self.deferred.is_empty() && self.stalled.is_none()
    }

    fn now(&self) -> f64 {
        (self.cfg.clock)()
    }

    /// One pass of the loop: admission, one prefill round, one step for the running requests.
    /// With `wait`, blocks for a job while nothing is in flight. False once the job channel is
    /// closed (the engine is gone).
    pub fn step(&mut self, wait: bool) -> bool {
        if !self.admit(wait) {
            return false;
        }
        self.prefill_round();
        // Bank overflow from the last step's retirements goes to RAM.
        self.pool.enforce_banks(&mut self.active);
        // Retire cancelled requests before spending a step on them.
        let mut i = 0;
        while i < self.active.len() {
            if self.active[i].cancel.load(Ordering::Relaxed) {
                let a = self.active.swap_remove(i);
                eprintln!("[coordinator] client gone: ending a request after {} of {} tokens", a.generated, a.max);
                self.pool.retire(a);
            } else {
                i += 1;
            }
        }
        if self.active.is_empty() {
            return true;
        }
        let drafts = if self.spec { self.limits.drafts() } else { 0 };
        self.pool.grow_active(&self.model, &mut self.active, drafts);
        if self.spec {
            self.spec_step();
        } else {
            self.decode_step();
        }
        true
    }

    /// Admit: block while idle (with `wait`), otherwise take what is queued while slots last (a
    /// retained slot counts as available: admission evicts it if needed). False when the job
    /// channel closed.
    fn admit(&mut self, wait: bool) -> bool {
        while !self.pool.free.is_empty() || !self.pool.retained.is_empty() {
            let min_retain = self.cfg.min_retain;
            // A job deferred behind an identical prefill goes first once that prefill is done
            // (it then forks its snapshot).
            let ready = self.deferred.iter().position(|j| !self.prefilling.iter().any(|p| extends(&j.ids, &p.ids, min_retain)));
            let was_stalled = self.stalled.is_some();
            let mut job = if let Some(j) = self.stalled.take() {
                j
            } else if let Some(i) = ready {
                self.deferred.remove(i).expect("deferred job")
            } else if wait && self.active.is_empty() && self.prefilling.is_empty() {
                match self.rx.recv() {
                    Ok(j) => j,
                    Err(_) => return false, // engine dropped
                }
            } else {
                match self.rx.try_recv() {
                    Ok(j) => j,
                    Err(mpsc::TryRecvError::Empty) => break,
                    Err(mpsc::TryRecvError::Disconnected) => return false,
                }
            };
            // Out of the queue: its place goes back.
            drop(job.place.take());
            if job.cancel.load(Ordering::Relaxed) {
                continue;
            }
            // A prompt that extends one still prefilling waits for it and then forks its snapshot,
            // instead of prefilling the same tokens again (n parallel samples, identical subagent
            // prompts).
            if self.prefilling.iter().any(|p| extends(&job.ids, &p.ids, min_retain)) {
                self.stats.deferred += 1;
                self.deferred.push_back(job);
                continue;
            }
            let rows = self.cfg.admit_rows(job.ids.len(), job.max_tokens);
            let admitted = self.pool.admit(&self.model, &self.active, &self.prefilling, &job.ids, rows, job.sampling.is_some());
            let (slot, points, resume) = match admitted {
                Ok(x) => x,
                // Others hold the memory: wait for them (nothing else is admitted meanwhile).
                Err(e) if !self.active.is_empty() || !self.prefilling.is_empty() => {
                    if !was_stalled {
                        eprintln!("[coordinator] a {}-token prompt waits for running requests' memory: {e}", job.ids.len());
                        self.stats.stalls += 1;
                    }
                    self.stalled = Some(job);
                    break;
                }
                Err(e) => {
                    eprintln!("[coordinator] refused a {}-token prompt: {e}", job.ids.len());
                    self.stats.refused += 1;
                    let _ = job.tx.send(Err(e));
                    continue;
                }
            };
            match resume {
                // An exact snapshot (one that serves this request): the first token without a
                // forward, the argmax or, for a sampled request, a draw from the kept logits.
                Some((n, after)) if n == job.ids.len() => {
                    let first = match (job.sampling, after.logits.as_ref()) {
                        (Some(s), Some(l)) => self.model.select_host(l, &Pick::at(Some(s), 0)),
                        _ => after.greedy.map(|t| t as Token).ok_or_else(|| "a snapshot without its next token".to_string()),
                    };
                    let p = Prefilling { slot, ids: job.ids, done: n, after: Some(after), points, max: job.max_tokens,
                        tx: job.tx, cancel: job.cancel, images: Vec::new(), sampling: job.sampling };
                    match first {
                        Ok(first) => start(&mut self.pool, &mut self.active, p, first, &self.cfg.eos),
                        Err(e) => {
                            eprintln!("[coordinator] first token from a snapshot failed: {e}");
                            let _ = p.tx.send(Err(e));
                            self.pool.park(p);
                        }
                    }
                }
                _ => {
                    // The prefill working set now, before any pipelined forward.
                    let mut slot = slot;
                    if let Err(e) = slot.begin_prefill() {
                        eprintln!("[coordinator] refused a {}-token prompt: {e}", job.ids.len());
                        let _ = job.tx.send(Err(e));
                        self.pool.discard(slot, &job.ids, points);
                        continue;
                    }
                    let done = resume.as_ref().map_or(0, |r| r.0);
                    self.prefilling.push_back(Prefilling { slot, done, after: None, ids: job.ids, points,
                        max: job.max_tokens, tx: job.tx, cancel: job.cancel, images: job.images, sampling: job.sampling });
                }
            }
            self.pool.enforce_banks(&mut self.active);
        }
        true
    }

    /// Prefill (perf reset Q1): segments round robin over the prompts in flight until this round
    /// has spent about `segment_ms`. A burst of short prompts prefills in one round (they start
    /// decoding together); a long prompt gets one segment per round.
    fn prefill_round(&mut self) {
        let seg_target = self.cfg.segment_ms / 1000.0;
        let round = self.now();
        let min_retain = self.cfg.min_retain;
        let bound = self.limits.sample_vocab;
        // Perf reset B1: the short prompts in flight prefill together, one pass over up to
        // `batch_rows` rows instead of one pass each.
        if self.prefilling.len() >= 2 {
            let mut batch: Vec<Prefilling<M::Slot>> = Vec::new();
            let mut rows = 0;
            let mut i = 0;
            while i < self.prefilling.len() {
                let p = &self.prefilling[i];
                let r = p.ids.len() - p.done;
                if !p.cancel.load(Ordering::Relaxed) && p.images.is_empty() && rows + r <= self.limits.batch_rows {
                    rows += r;
                    batch.push(self.prefilling.remove(i).expect("prefilling entry"));
                } else {
                    i += 1;
                }
            }
            if batch.len() >= 2 {
                let result = {
                    let mut segs: Vec<Segment<'_, M::Slot>> = batch
                        .iter_mut()
                        .map(|p| Segment {
                            slot: &mut p.slot,
                            tokens: &p.ids[p.done..],
                            images: &[],
                            pick: Pick::at(p.sampling, 0),
                            // The last logit row of a prompt that gets a snapshot.
                            keep_logits: p.ids.len() >= min_retain,
                        })
                        .collect();
                    self.model.prefill(&mut segs)
                };
                self.stats.prefill_passes += 1;
                self.stats.prefill_tokens += rows as u64;
                match result {
                    Ok(outs) if outs.len() == batch.len() => {
                        for (mut p, out) in batch.into_iter().zip(outs) {
                            p.done = p.ids.len();
                            p.after = Some(match out.logits {
                                Some(l) => After::from_logits(&l, bound, true),
                                None => After { greedy: p.sampling.is_none().then_some(out.next as usize), logits: None },
                            });
                            start(&mut self.pool, &mut self.active, p, out.next, &self.cfg.eos);
                        }
                    }
                    other => {
                        let e = match other {
                            Err(e) => e,
                            Ok(o) => format!("prefill returned {} results for {} prompts", o.len(), batch.len()),
                        };
                        eprintln!("[coordinator] batched prefill of {} prompts failed: {e}", batch.len());
                        for p in batch {
                            let _ = p.tx.send(Err(e.clone()));
                            self.pool.discard(p.slot, &p.ids, p.points);
                        }
                    }
                }
            } else {
                for p in batch.into_iter().rev() {
                    self.prefilling.push_front(p);
                }
            }
        }
        while let Some(mut p) = self.prefilling.pop_front() {
            if p.cancel.load(Ordering::Relaxed) {
                eprintln!("[coordinator] client gone: parking a prefill at {} of {} tokens", p.done, p.ids.len());
                self.pool.park(p);
                continue;
            }
            // Alone, a longer segment (fewer pipeline restarts); a new arrival still waits at most
            // about four targets.
            let target = if self.active.is_empty() && self.prefilling.is_empty() { 4.0 * seg_target } else { seg_target };
            let q = self.cfg.seg_quantum.max(1);
            let len = ((target / self.sec_per_token) as usize / q * q).clamp(q, self.cfg.seg_max.max(q));
            let end = (p.done + len).min(p.ids.len());
            let t0 = self.now();
            // The images this segment reaches: room for the model's encoder first.
            let spans: Vec<ImageSpan> =
                p.images.iter().filter(|s| s.start < end && s.start + s.image.tokens > p.done).cloned().collect();
            if !spans.is_empty() {
                let need = self.model.image_bytes(&spans);
                if let Err(e) = self.pool.make_room_bytes(&self.model, need) {
                    eprintln!("[coordinator] images of a {}-token prompt: {e}", p.ids.len());
                    let _ = p.tx.send(Err(e));
                    self.pool.discard(p.slot, &p.ids, p.points);
                    continue;
                }
            }
            let last = end == p.ids.len();
            let result = {
                let mut segs = [Segment {
                    slot: &mut p.slot,
                    tokens: &p.ids[p.done..end],
                    images: &spans,
                    // The first token's pick on the last segment; the argmax at every other end
                    // (a parked prefill's snapshot serves greedy repeats with it).
                    pick: if last { Pick::at(p.sampling, 0) } else { Pick::greedy() },
                    keep_logits: end >= min_retain,
                }];
                self.model.prefill(&mut segs)
            };
            self.stats.prefill_passes += 1;
            self.stats.prefill_tokens += (end - p.done) as u64;
            match result.and_then(|mut o| o.pop().ok_or_else(|| "prefill returned no result".to_string())) {
                Ok(out) => {
                    // The rate at this context length sizes the next segment.
                    if end - p.done >= q {
                        self.sec_per_token = (self.now() - t0) / (end - p.done) as f64;
                    }
                    p.done = end;
                    let after = match out.logits {
                        Some(l) => After::from_logits(&l, bound, true),
                        None if !last || p.sampling.is_none() => After { greedy: Some(out.next as usize), logits: None },
                        None => After::default(),
                    };
                    p.after = Some(after);
                    if !last {
                        self.prefilling.push_back(p);
                    } else {
                        start(&mut self.pool, &mut self.active, p, out.next, &self.cfg.eos);
                    }
                }
                Err(e) => {
                    eprintln!("[coordinator] prefill of a {}-token prompt failed at {}: {e}", p.ids.len(), p.done);
                    let _ = p.tx.send(Err(e));
                    self.pool.discard(p.slot, &p.ids, p.points);
                }
            }
            // Back to admission and the decode step once the round's budget is spent.
            if self.now() - round >= seg_target {
                break;
            }
        }
    }

    /// Fail every running request with `e` and give its slot back.
    fn fail_active(&mut self, what: &str, e: String) {
        eprintln!("[coordinator] {what} failed for {} requests: {e}", self.active.len());
        for a in std::mem::take(&mut self.active) {
            let _ = a.tx.send(Err(e.clone()));
            self.pool.discard(a.slot, &a.hist, a.points);
        }
    }

    /// Deliver `next` to request `a`: its history, budget and end-of-sequence. True when the
    /// request is done (or its client gone).
    fn deliver(a: &mut Active<M::Slot>, next: Token, eos: &[Token]) -> bool {
        a.last = next;
        a.generated += 1;
        a.hist.push(next);
        let done = eos.contains(&next) || a.generated >= a.max;
        a.tx.send(Ok(next)).is_err() || done
    }

    /// One decode step for every running request: its last token in, its next token out.
    fn decode_step(&mut self) {
        let result = {
            let mut rows: Vec<DecodeRow<'_, M::Slot>> = self
                .active
                .iter_mut()
                .map(|a| DecodeRow { slot: &mut a.slot, token: a.last, pick: Pick::at(a.sampling, a.generated as u64) })
                .collect();
            self.model.decode(&mut rows)
        };
        self.stats.decode_steps += 1;
        self.stats.decode_rows += self.active.len() as u64;
        match result {
            Ok(nexts) if nexts.len() == self.active.len() => {
                let mut keep = Vec::with_capacity(self.active.len());
                for (mut a, next) in std::mem::take(&mut self.active).into_iter().zip(nexts) {
                    if Self::deliver(&mut a, next, &self.cfg.eos) {
                        self.pool.retire(a);
                    } else {
                        keep.push(a);
                    }
                }
                self.active = keep;
            }
            Ok(nexts) => self.fail_active("decode step", format!("decode returned {} tokens for {} rows", nexts.len(),
                self.active.len())),
            Err(e) => self.fail_active("decode step", e),
        }
    }

    /// One speculative step (perf reset S1): draft, choose each request's verify length, verify
    /// every window in one pass, accept the longest run the target agrees with, deliver it plus
    /// the target's own next token, and commit exactly the delivered rows.
    fn spec_step(&mut self) {
        let lim = self.limits;
        // Never draft past the token budget: a step emits at most k + 1.
        let caps: Vec<usize> = self.active.iter().map(|a| (a.max - a.generated - 1).min(lim.drafts())).collect();
        let drafts = {
            let mut rows: Vec<DraftRow<'_, M::Slot>> = self
                .active
                .iter_mut()
                .zip(&caps)
                .map(|(a, &k)| DraftRow { slot: &mut a.slot, last: a.last, max: k, pick: Pick::at(a.sampling, a.generated as u64) })
                .collect();
            self.model.draft(&mut rows)
        };
        let drafts = match drafts {
            Ok(d) if d.len() == self.active.len() && d.iter().all(|d| d.probs.len() == d.tokens.len()) => d,
            Ok(_) => return self.fail_active("draft", "the drafter returned a malformed draft".into()),
            Err(e) => return self.fail_active("draft", e),
        };
        let probs: Vec<Vec<f32>> = drafts.iter().map(|d| d.probs.clone()).collect();
        // The policy's lengths, then the step's row budget (the most likely drafts first).
        let ks = crate::spec::budget(&probs, &self.cfg.policy.lengths(&probs, &caps), self.cfg.spec_max_rows);
        let blocks: Vec<Vec<Token>> = self
            .active
            .iter()
            .zip(&drafts)
            .zip(&ks)
            .map(|((a, d), &k)| std::iter::once(a.last).chain(d.tokens[..k].iter().copied()).collect())
            .collect();
        let picks: Vec<Vec<Pick>> = self
            .active
            .iter()
            .zip(&blocks)
            .map(|(a, b)| (0..b.len()).map(|j| Pick::at(a.sampling, (a.generated + j) as u64)).collect())
            .collect();
        let sel = {
            let mut windows: Vec<Window<'_, M::Slot>> = self
                .active
                .iter_mut()
                .zip(&blocks)
                .zip(&picks)
                .map(|((a, b), p)| Window { slot: &mut a.slot, tokens: b, picks: p })
                .collect();
            self.model.verify(&mut windows)
        };
        self.stats.spec_steps += 1;
        let sel = match sel {
            Ok(s) if s.len() == blocks.len() && s.iter().zip(&blocks).all(|(g, b)| g.len() == b.len()) => s,
            Ok(_) => return self.fail_active("speculative step", "verify returned a malformed selection".into()),
            Err(e) => return self.fail_active("speculative step", e),
        };
        // Accept, deliver, and count the rows each slot keeps: one per delivered token (the
        // window's first row is `last`, whose row the slot did not hold yet).
        let mut keeps = Vec::with_capacity(blocks.len());
        let mut done = Vec::with_capacity(blocks.len());
        for ((a, b), g) in self.active.iter_mut().zip(&blocks).zip(&sel) {
            let mut acc = 0;
            while acc + 1 < b.len() && b[acc + 1] == g[acc] {
                acc += 1;
            }
            self.stats.drafts_verified += (b.len() - 1) as u64;
            self.stats.drafts_accepted += acc as u64;
            let mut delivered = 0;
            let mut finished = false;
            for &next in b[1..=acc].iter().chain(std::iter::once(&g[acc])) {
                delivered += 1;
                if Self::deliver(a, next, &self.cfg.eos) {
                    finished = true;
                    break;
                }
            }
            keeps.push(delivered);
            done.push(finished);
        }
        let committed = {
            let mut slots: Vec<&mut M::Slot> = self.active.iter_mut().map(|a| &mut a.slot).collect();
            self.model.commit(&mut slots, &keeps)
        };
        if let Err(e) = committed {
            return self.fail_active("commit", e);
        }
        let mut keep = Vec::with_capacity(self.active.len());
        for (a, finished) in std::mem::take(&mut self.active).into_iter().zip(done) {
            if finished {
                self.pool.retire(a);
            } else {
                keep.push(a);
            }
        }
        self.active = keep;
    }
}
