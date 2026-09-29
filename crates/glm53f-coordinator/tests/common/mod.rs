//! A toy deterministic model behind `KvSlot` and `ModelForward`, for the scheduler tests.
//!
//! The model's next-token logits are a function of a hash of the whole token prefix. The slot
//! keeps that hash as its *positional* state (like KDA: one value for the current position, no
//! way back except a saved mark) and the token ids as its *appendable* rows (like MLA pages).
//! Every operation checks that the state is the hash of the rows, so a scheduler that resumes
//! from the wrong point, rewinds to a mark of another history, forgets a commit or restores the
//! wrong snapshot fails loudly, and any output differs from the serial reference
//! ([`reference`]).
//!
//! Memory is a shared budget: slot capacity (per token) and marks draw from it, so admission,
//! waiting and eviction under pressure are real. Time is a simulated clock the forward advances,
//! so segment sizing is deterministic. Every pass is logged ([`Call`]).

#![allow(dead_code)]

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{mpsc, Arc, Mutex};

use glm53f_coordinator::model::{
    DecodeRow, Draft, DraftRow, KvSlot, Limits, ModelForward, Pick, Segment, SegmentOut, Token, Window,
};
use glm53f_coordinator::sampling::{select_pick, Sampling};
use glm53f_coordinator::scheduler::{Clock, Job, Scheduler, SchedulerConfig};

/// Logit row length (the "LM head" rows).
pub const LD: usize = 40;
/// Token ids that exist; rows 32..39 are padding with the largest logits of all.
pub const BOUND: usize = 32;
/// Tokens per host page.
pub const PAGE: usize = 8;

pub fn splitmix(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9e37_79b9_7f4a_7c15);
    x = (x ^ (x >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    x ^ (x >> 31)
}

/// The state of the empty prefix.
pub const H0: u64 = 0x5eed_0000_1234_5678;

/// The state after one more token.
pub fn step(h: u64, t: Token) -> u64 {
    splitmix(h ^ (u64::from(t) + 1).wrapping_mul(0x9e37_79b9_7f4a_7c15))
}

pub fn prefix_hash(toks: &[Token]) -> u64 {
    toks.iter().fold(H0, |h, &t| step(h, t))
}

/// The logits after a prefix whose state is `h`: values in [0, 6) over the real ids, and 100 on
/// the padding rows (which must never be picked).
pub fn logits(h: u64) -> Vec<f32> {
    (0..LD)
        .map(|i| if i >= BOUND { 100.0 } else { (splitmix(h ^ (i as u64).wrapping_mul(0xabcd_ef01)) >> 40) as f32 / 16_777_216.0 * 6.0 })
        .collect()
}

/// Context tokens a copying model's induction matches ([`copying_logits`]).
pub const INDUCTION: usize = 8;

/// A copying model's logits after `ctx` (`MockModel::copying`): the hash model's, with the token
/// that followed the most recent earlier occurrence of the last [`INDUCTION`] tokens (one of them
/// may differ) raised above every other: an induction head that goes on copying past a
/// substituted token. After one prefix in forty the copy is left for the hash model's pick (an
/// edit).
pub fn copying_logits(ctx: &[Token]) -> Vec<f32> {
    let h = prefix_hash(ctx);
    let mut l = logits(h);
    let n = ctx.len();
    if n <= INDUCTION || splitmix(h ^ 0xed17).is_multiple_of(40) {
        return l;
    }
    let tail = &ctx[n - INDUCTION..];
    let close = |e: usize| ctx[e - INDUCTION..e].iter().zip(tail).filter(|(a, b)| a != b).count() <= 1;
    if let Some(e) = (INDUCTION..n).rev().find(|&e| close(e)) {
        l[ctx[e] as usize] = 7.0;
    }
    l
}

/// Serial generation by the copying model: what every request must produce with
/// `MockModel::copying`.
pub fn copying_reference(prompt: &[Token], max: usize, sampling: Option<Sampling>, eos: &[Token]) -> Vec<Token> {
    let mut ctx = prompt.to_vec();
    let mut out = Vec::new();
    loop {
        let t = select_pick(&copying_logits(&ctx), BOUND, &Pick::at(sampling, out.len() as u64));
        out.push(t);
        if eos.contains(&t) || out.len() >= max {
            return out;
        }
        ctx.push(t);
    }
}

/// Serial generation: what every request must produce, whatever the batching, speculation or
/// caching.
pub fn reference(prompt: &[Token], max: usize, sampling: Option<Sampling>, eos: &[Token]) -> Vec<Token> {
    let mut h = prefix_hash(prompt);
    let mut out = Vec::new();
    loop {
        let t = select_pick(&logits(h), BOUND, &Pick::at(sampling, out.len() as u64));
        out.push(t);
        if eos.contains(&t) || out.len() >= max {
            return out;
        }
        h = step(h, t);
    }
}

/// The shared device memory.
pub struct Mem {
    pub budget: usize,
    pub used: usize,
}

pub type Dev = Arc<Mutex<Mem>>;

pub fn device(budget: usize) -> Dev {
    Arc::new(Mutex::new(Mem { budget, used: 0 }))
}

fn take(dev: &Dev, bytes: usize) -> Result<(), String> {
    let mut m = dev.lock().unwrap();
    if m.used + bytes > m.budget {
        return Err(format!("device: {bytes} B wanted, {} B free", m.budget - m.used));
    }
    m.used += bytes;
    Ok(())
}

fn give(dev: &Dev, bytes: usize) {
    dev.lock().unwrap().used -= bytes;
}

/// Device bytes per token of slot capacity, and per mark.
pub const TOKEN_BYTES: usize = 100;
pub const MARK_BYTES: usize = 1000;
/// Capacity grows in steps of this many tokens.
pub const GROW: usize = 64;

pub struct MockMark {
    pub len: usize,
    pub state: u64,
    dev: Dev,
}

impl Drop for MockMark {
    fn drop(&mut self) {
        give(&self.dev, MARK_BYTES);
    }
}

pub struct MockSlot {
    pub id: usize,
    dev: Dev,
    /// Committed rows: the token ids.
    pub toks: Vec<Token>,
    /// The positional state: `prefix_hash(toks)`.
    pub state: u64,
    pending: Vec<Token>,
    cap: usize,
    base: usize,
}

impl MockSlot {
    pub fn new(id: usize, dev: &Dev, base: usize) -> MockSlot {
        take(dev, base * TOKEN_BYTES).expect("room for the base slots");
        MockSlot { id, dev: dev.clone(), toks: Vec::new(), state: H0, pending: Vec::new(), cap: base, base }
    }

    /// The invariant: the positional state is the rows' state.
    pub fn check(&self) -> Result<(), String> {
        if prefix_hash(&self.toks) != self.state {
            return Err(format!("slot {}: positional state does not match its {} rows", self.id, self.toks.len()));
        }
        Ok(())
    }

    /// Append committed tokens directly (reserving room): a slot that has run them.
    pub fn fill(&mut self, toks: &[Token]) {
        self.reserve(self.toks.len() + toks.len()).expect("room");
        for &t in toks {
            self.append(t).expect("append");
        }
    }

    fn append(&mut self, t: Token) -> Result<(), String> {
        if self.toks.len() + self.pending.len() + 1 > self.cap {
            return Err(format!("slot {}: over its capacity of {} tokens", self.id, self.cap));
        }
        self.toks.push(t);
        self.state = step(self.state, t);
        Ok(())
    }
}

impl KvSlot for MockSlot {
    type Mark = MockMark;

    fn tokens(&self) -> usize {
        self.toks.len()
    }

    fn pending(&self) -> usize {
        self.pending.len()
    }

    fn capacity(&self) -> usize {
        self.cap
    }

    fn need_bytes(&self, tokens: usize) -> usize {
        if tokens > self.cap { (tokens.next_multiple_of(GROW) - self.cap) * TOKEN_BYTES } else { 0 }
    }

    fn reserve(&mut self, tokens: usize) -> Result<(), String> {
        if tokens > self.cap {
            let to = tokens.next_multiple_of(GROW);
            take(&self.dev, (to - self.cap) * TOKEN_BYTES)?;
            self.cap = to;
        }
        Ok(())
    }

    fn bytes(&self) -> usize {
        self.cap * TOKEN_BYTES
    }

    fn reset(&mut self) {
        self.toks.clear();
        self.pending.clear();
        self.state = H0;
    }

    fn release(&mut self) {
        self.reset();
        if self.cap > self.base {
            give(&self.dev, (self.cap - self.base) * TOKEN_BYTES);
            self.cap = self.base;
        }
    }

    fn mark(&self) -> Result<MockMark, String> {
        if !self.pending.is_empty() {
            return Err("mark with pending rows".into());
        }
        self.check()?;
        take(&self.dev, MARK_BYTES)?;
        Ok(MockMark { len: self.toks.len(), state: self.state, dev: self.dev.clone() })
    }

    fn rewind(&mut self, to: usize, mark: &MockMark) -> Result<(), String> {
        if to > self.toks.len() || mark.len != to || !self.pending.is_empty() {
            return Err(format!("slot {}: rewind to {to} (mark at {}) from {}", self.id, mark.len, self.toks.len()));
        }
        if prefix_hash(&self.toks[..to]) != mark.state {
            return Err(format!("slot {}: rewind to a mark of another history", self.id));
        }
        self.toks.truncate(to);
        self.state = mark.state;
        Ok(())
    }

    fn fork(&mut self, src: &MockSlot, to: usize, mark: &MockMark) -> Result<(), String> {
        if !self.toks.is_empty() || to > src.toks.len() || mark.len != to || to > self.cap {
            return Err(format!("slot {}: fork {to} of slot {}", self.id, src.id));
        }
        if prefix_hash(&src.toks[..to]) != mark.state {
            return Err(format!("slot {}: fork from a mark of another history", self.id));
        }
        self.toks = src.toks[..to].to_vec();
        self.state = mark.state;
        Ok(())
    }

    fn page_tokens(&self) -> usize {
        PAGE
    }

    fn page_bytes(&self) -> usize {
        PAGE * 4
    }

    fn state_bytes(&self) -> usize {
        16
    }

    fn export_page(&self, first: usize, n: usize, dst: &mut [u8]) -> Result<(), String> {
        if !first.is_multiple_of(PAGE) || n > PAGE || first + n > self.toks.len() {
            return Err(format!("export_page {first}+{n} of {}", self.toks.len()));
        }
        for (k, t) in self.toks[first..first + n].iter().enumerate() {
            dst[4 * k..4 * k + 4].copy_from_slice(&t.to_le_bytes());
        }
        Ok(())
    }

    fn export_state(&self, mark: &MockMark, dst: &mut [u8]) -> Result<(), String> {
        dst[..8].copy_from_slice(&mark.state.to_le_bytes());
        dst[8..16].copy_from_slice(&(mark.len as u64).to_le_bytes());
        Ok(())
    }

    fn import_page(&mut self, n: usize, src: &[u8]) -> Result<(), String> {
        if self.toks.len() + n > self.cap {
            return Err("import past the reservation".into());
        }
        for k in 0..n {
            self.toks.push(u32::from_le_bytes(src[4 * k..4 * k + 4].try_into().unwrap()));
        }
        Ok(())
    }

    fn import_state(&mut self, tokens: usize, src: &[u8]) -> Result<(), String> {
        let state = u64::from_le_bytes(src[..8].try_into().unwrap());
        let len = u64::from_le_bytes(src[8..16].try_into().unwrap()) as usize;
        if len != tokens || self.toks.len() != tokens {
            return Err(format!("import_state: {tokens} tokens, image of {len}, rows {}", self.toks.len()));
        }
        self.state = state;
        self.check()
    }
}

/// One logged pass: the slots and row counts it carried.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Call {
    Prefill(Vec<(usize, usize)>),
    Decode(Vec<usize>),
    Draft(Vec<usize>),
    Verify(Vec<(usize, usize)>),
    Commit(Vec<(usize, usize)>),
}

pub struct MockModel {
    pub dev: Dev,
    pub log: Arc<Mutex<Vec<Call>>>,
    /// Simulated time, nanoseconds.
    pub clock: Arc<AtomicU64>,
    /// Verify window rows (0: no drafter).
    pub block: usize,
    pub batch_rows: usize,
    /// Simulated cost of a pass and of each row.
    pub pass_ns: u64,
    pub row_ns: u64,
    /// The copying model ([`copying_logits`]) instead of the hash model.
    pub copying: bool,
    /// Prefill panics: a bug in a pass, which ends the scheduler's thread.
    pub panic_in_prefill: bool,
}

impl MockModel {
    pub fn new(dev: &Dev, block: usize) -> MockModel {
        MockModel {
            dev: dev.clone(),
            log: Arc::new(Mutex::new(Vec::new())),
            clock: Arc::new(AtomicU64::new(0)),
            block,
            batch_rows: 64,
            pass_ns: 1_000_000,
            row_ns: 250_000,
            copying: false,
            panic_in_prefill: false,
        }
    }

    /// The logits after `toks`, whose state is `h`.
    fn next_logits(&self, toks: &[Token], h: u64) -> Vec<f32> {
        if self.copying { copying_logits(toks) } else { logits(h) }
    }

    fn cost(&self, rows: usize) {
        self.clock.fetch_add(self.pass_ns + self.row_ns * rows as u64, Ordering::Relaxed);
    }

    fn record(&self, c: Call) {
        self.log.lock().unwrap().push(c);
    }

    /// The scheduler's clock over this model's simulated time.
    pub fn clock_fn(&self) -> Clock {
        let c = self.clock.clone();
        Arc::new(move || c.load(Ordering::Relaxed) as f64 * 1e-9)
    }

    /// Whether draft `j` proposed after state `h` is deliberately wrong (about one in four).
    fn corrupt(h: u64, j: usize) -> bool {
        splitmix(h ^ (j as u64 + 7)).is_multiple_of(4)
    }
}

impl ModelForward for MockModel {
    type Slot = MockSlot;

    fn limits(&self) -> Limits {
        Limits { vocab: LD, sample_vocab: BOUND, batch_rows: self.batch_rows, block: self.block }
    }

    fn free_bytes(&self) -> Result<usize, String> {
        let m = self.dev.lock().unwrap();
        Ok(m.budget - m.used)
    }

    fn prefill(&mut self, segs: &mut [Segment<'_, MockSlot>]) -> Result<Vec<SegmentOut>, String> {
        assert!(!self.panic_in_prefill, "the toy model's prefill panics, as asked");
        self.record(Call::Prefill(segs.iter().map(|s| (s.slot.id, s.tokens.len())).collect()));
        self.cost(segs.iter().map(|s| s.tokens.len()).sum());
        let mut out = Vec::new();
        for s in segs.iter_mut() {
            if s.slot.pending() != 0 {
                return Err("prefill with pending rows".into());
            }
            s.slot.check()?;
            for &t in s.tokens {
                if t as usize >= BOUND {
                    return Err(format!("token {t} past the vocabulary"));
                }
                s.slot.append(t)?;
            }
            let l = self.next_logits(&s.slot.toks, s.slot.state);
            out.push(SegmentOut { next: select_pick(&l, BOUND, &s.pick), logits: s.keep_logits.then_some(l) });
        }
        Ok(out)
    }

    fn decode(&mut self, rows: &mut [DecodeRow<'_, MockSlot>]) -> Result<Vec<Token>, String> {
        self.record(Call::Decode(rows.iter().map(|r| r.slot.id).collect()));
        self.cost(rows.len());
        rows.iter_mut()
            .map(|r| {
                r.slot.check()?;
                r.slot.append(r.token)?;
                Ok(select_pick(&self.next_logits(&r.slot.toks, r.slot.state), BOUND, &r.pick))
            })
            .collect()
    }

    fn draft(&mut self, rows: &mut [DraftRow<'_, MockSlot>]) -> Result<Vec<Draft>, String> {
        if self.block < 2 {
            return Err("no drafter".into());
        }
        self.record(Call::Draft(rows.iter().map(|r| r.slot.id).collect()));
        Ok(rows
            .iter()
            .map(|r| {
                // The model's own greedy chain after `last`, with some drafts deliberately wrong.
                let mut h = step(r.slot.state, r.last);
                let mut tokens = Vec::new();
                for j in 0..r.max.min(self.block - 1) {
                    let mut d = select_pick(&logits(h), BOUND, &Pick::greedy());
                    if Self::corrupt(h, j) {
                        d = (d + 1) % BOUND as Token;
                    }
                    tokens.push(d);
                    h = step(h, d);
                }
                Draft { probs: vec![0.9; tokens.len()], tokens }
            })
            .collect())
    }

    fn verify(&mut self, windows: &mut [Window<'_, MockSlot>]) -> Result<Vec<Vec<Token>>, String> {
        self.record(Call::Verify(windows.iter().map(|w| (w.slot.id, w.tokens.len())).collect()));
        self.cost(windows.iter().map(|w| w.tokens.len()).sum());
        let mut out = Vec::new();
        for w in windows.iter_mut() {
            let slot = &mut *w.slot;
            if !slot.pending.is_empty() || w.picks.len() != w.tokens.len() {
                return Err("verify: pending rows or a pick per row missing".into());
            }
            slot.check()?;
            if slot.toks.len() + w.tokens.len() > slot.cap {
                return Err(format!("slot {}: window over its capacity", slot.id));
            }
            // The positional state stays at the committed position; the rows go pending.
            let mut h = slot.state;
            let mut ctx = if self.copying { slot.toks.clone() } else { Vec::new() };
            let mut picks = Vec::new();
            for (t, p) in w.tokens.iter().zip(w.picks) {
                if *t as usize >= BOUND {
                    return Err(format!("verify: token {t} past the vocabulary"));
                }
                h = step(h, *t);
                ctx.push(*t);
                picks.push(select_pick(&self.next_logits(&ctx, h), BOUND, p));
            }
            slot.pending = w.tokens.to_vec();
            out.push(picks);
        }
        Ok(out)
    }

    fn commit(&mut self, slots: &mut [&mut MockSlot], keep: &[usize]) -> Result<(), String> {
        self.record(Call::Commit(slots.iter().zip(keep).map(|(s, &k)| (s.id, k)).collect()));
        for (s, &k) in slots.iter_mut().zip(keep) {
            if k == 0 || k > s.pending.len() {
                return Err(format!("slot {}: keep {k} of {} pending rows", s.id, s.pending.len()));
            }
            // Replay the kept rows from the committed state (KDA's replay).
            let kept: Vec<Token> = s.pending.drain(..).take(k).collect();
            for t in kept {
                s.toks.push(t);
                s.state = step(s.state, t);
            }
            s.check()?;
        }
        Ok(())
    }
}

/// A scheduler over the mock, driven step by step.
pub struct Harness {
    pub sched: Scheduler<MockModel>,
    pub jobs: mpsc::Sender<Job>,
    pub log: Arc<Mutex<Vec<Call>>>,
    pub dev: Dev,
    pub eos: Vec<Token>,
}

pub struct Setup {
    pub slots: usize,
    pub base: usize,
    pub budget: usize,
    pub block: usize,
    pub host: Option<glm53f_coordinator::HostTierConfig>,
    pub eos: Vec<Token>,
    pub tweak: fn(&mut SchedulerConfig),
    /// The copying model ([`copying_logits`]).
    pub copying: bool,
}

impl Default for Setup {
    fn default() -> Self {
        Setup { slots: 4, base: 64, budget: 10_000_000, block: 0, host: None, eos: Vec::new(), tweak: |_| {}, copying: false }
    }
}

/// Small-scale knobs: snapshots from 16 tokens, 16-token segment quanta, 20 ms segments at the
/// mock's 0.25 ms per row, admission allowances of 8..32 tokens. `GLM53F_PREFIX_CACHE_ENTRIES=N`
/// runs every test that does not set its own under a bank cap of N (the daemon reads it too):
/// `GLM53F_PREFIX_CACHE_ENTRIES=24` is the source's configuration.
pub fn test_config(eos: Vec<Token>, clock: Clock) -> SchedulerConfig {
    let mut c = SchedulerConfig::new(eos);
    c.min_retain = 16;
    c.seg_quantum = 16;
    c.seg_max = 256;
    c.segment_ms = 20.0;
    c.out_min = 8;
    c.out_max = 32;
    c.out_slack = 8;
    c.granularity = 4;
    c.clock = clock;
    if let Some(n) = std::env::var("GLM53F_PREFIX_CACHE_ENTRIES").ok().and_then(|v| v.parse().ok()) {
        c.bank = n;
    }
    c
}

pub fn harness(s: Setup) -> Harness {
    let dev = device(s.budget);
    let mut model = MockModel::new(&dev, s.block);
    model.copying = s.copying;
    let log = model.log.clone();
    let slots: Vec<MockSlot> = (0..s.slots).map(|i| MockSlot::new(i, &dev, s.base)).collect();
    let mut cfg = test_config(s.eos.clone(), model.clock_fn());
    (s.tweak)(&mut cfg);
    let cache = s.host.map(|h| glm53f_coordinator::HostCache::new(h).expect("host tier"));
    let (tx, rx) = mpsc::channel();
    Harness { sched: Scheduler::new(model, slots, cache, cfg, rx), jobs: tx, log, dev, eos: s.eos }
}

impl Harness {
    /// Submit a prompt; the receiver gets its tokens.
    pub fn submit(&self, prompt: &[Token], max: usize, sampling: Option<Sampling>)
        -> mpsc::Receiver<Result<Token, String>> {
        let (job, rx) = Job::new(prompt.to_vec(), max, sampling);
        self.jobs.send(job).expect("scheduler alive");
        rx
    }

    /// Step until nothing is in flight (and the channel is drained).
    pub fn run(&mut self) {
        let mut idle = 0;
        for _ in 0..100_000 {
            assert!(self.sched.step(false));
            if self.sched.is_idle() {
                idle += 1;
                if idle >= 3 {
                    return;
                }
            } else {
                idle = 0;
            }
        }
        panic!("the scheduler did not go idle");
    }

    pub fn calls(&self) -> Vec<Call> {
        self.log.lock().unwrap().clone()
    }

    pub fn clear_log(&self) {
        self.log.lock().unwrap().clear();
    }

    /// Prompt rows prefilled since the log was cleared.
    pub fn prefilled(&self) -> usize {
        self.calls().iter().map(|c| if let Call::Prefill(v) = c { v.iter().map(|x| x.1).sum() } else { 0 }).sum()
    }
}

/// Every token a request produced, or its error.
pub fn tokens(rx: &mpsc::Receiver<Result<Token, String>>) -> Result<Vec<Token>, String> {
    rx.try_iter().collect()
}

/// A prompt of `n` tokens from a seed.
pub fn prompt(seed: u64, n: usize) -> Vec<Token> {
    (0..n).map(|i| (splitmix(seed.wrapping_mul(1000) + i as u64) % BOUND as u64) as Token).collect()
}
