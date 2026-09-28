//! The host RAM tier (mimo26f-afd perf reset K3; the DS41RT host snapshot cache design in its
//! `on-evict` store mode): the RAM home of the snapshots the device had to evict.
//!
//! Retained snapshots live on the device first (`crate::pool`: a snapshot is a *point* in a
//! slot's history, with a [`KvSlot::Mark`] of its positional state): with no pressure nothing
//! is copied to RAM, however many there are. When the device evicts a point (an incoming request
//! needs its memory or its slot, least recently used first, a running request's own included;
//! or a bank over an optional cap), the scheduler stores it here, then frees it on the device; a
//! returning conversation that misses the device restores from here instead of prefilling. RAM
//! eviction deletes the least recently used snapshot first; at equal use a prompt snapshot goes
//! before a turn snapshot (`victim`, the device's order too).
//!
//! That departs from an older order, every prompt snapshot before any turn snapshot, which
//! mimo26f-afd v1.1.1 fixed. A prompt snapshot shares its pages with its conversation's turn
//! snapshot, so deleting it frees a state slot but no pages. Under page pressure that order
//! deleted every prompt snapshot, the fresh ones included, before the first stale turn
//! snapshot, and an exact repeat of a recent prompt (a retry, a regenerate, identical subagent
//! prompts) prefilled from cold: in the source's measurements, 37-41 s instead of about 0.2 s.
//!
//! A snapshot is a request's KV state at an exact position:
//! - the appendable rows, as host pages of [`KvSlot::page_tokens`] tokens
//!   ([`KvSlot::export_page`]; 64 tokens of 6,171 B for GLM-5.3-Flash with an FP8 cache),
//!   shared between snapshots. A page's content depends only on the tokens up to its end, so a
//!   page is identified by a hash chain over the token prefix;
//! - the rows past the last full page, in a page slot of their own;
//! - the positional state ([`KvSlot::export_state`] of the point's mark; about 141 MiB for
//!   GLM-5.3-Flash: KDA recurrent state plus convolution windows);
//! - what follows ([`After`]): the argmax there when known and, for a prompt snapshot, the last
//!   logit row, so the first token after an exact restore needs no forward, greedy or sampled.
//!   An exact match that cannot serve a request gives way to the longest shorter one
//!   ([`HostCache::lookup_for`]).
//!
//! A prompt is restored only from a snapshot whose tokens are a prefix of it; the rest is
//! prefilled. Snapshots are found through a [`RadixIndex`] over their tokens. Everything runs
//! on the scheduler thread. With the `cuda` feature the arenas are page-locked, so page and
//! state copies run at full PCIe rate. `GLM53F_HOST_CACHE_GB=0` turns the tier off (nothing is
//! allocated or copied).

use std::collections::{BTreeMap, HashMap};
use std::hash::{Hash, Hasher};

use crate::model::{KvSlot, Token};
use crate::radix::RadixIndex;
use crate::sampling::After;

/// Which retention bank a snapshot came from.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    /// At prompt end (the multi-turn fallback point).
    Prompt,
    /// At completion end.
    Turn,
}

/// Page-locked slab size (whole slots are carved from each).
const SLAB_BYTES: usize = 1 << 30;

/// The token count of the reference conversation [`HostTierConfig::balanced`] sizes for.
pub const BALANCE_TOKENS: usize = 65_536;

/// Fixed-size slots carved from (optionally page-locked) slabs.
struct Arena {
    slabs: Vec<Vec<u8>>,
    slot_bytes: usize,
    free: Vec<(usize, usize)>,
    pinned: bool,
}

impl Arena {
    fn new(slot_bytes: usize, slots: usize, pin: bool) -> Result<Self, String> {
        let per_slab = (SLAB_BYTES / slot_bytes.max(1)).max(1);
        let mut me = Self { slabs: Vec::new(), slot_bytes, free: Vec::with_capacity(slots), pinned: false };
        let mut left = slots;
        while left > 0 {
            let n = left.min(per_slab);
            #[cfg_attr(not(feature = "cuda"), allow(unused_mut))]
            let mut slab = vec![0u8; n * slot_bytes];
            #[cfg(feature = "cuda")]
            if pin {
                // Registered once for the arena's lifetime (unregistered in Drop).
                crate::gpu::host_register(slab.as_mut_ptr(), slab.len())
                    .map_err(|e| format!("hostcache: page-locking {} B: {e}", slab.len()))?;
                me.pinned = true;
            }
            #[cfg(not(feature = "cuda"))]
            let _ = pin;
            let s = me.slabs.len();
            me.free.extend((0..n).rev().map(|i| (s, i * slot_bytes)));
            me.slabs.push(slab);
            left -= n;
        }
        Ok(me)
    }

    fn get(&self, slot: (usize, usize)) -> &[u8] {
        &self.slabs[slot.0][slot.1..slot.1 + self.slot_bytes]
    }

    fn get_mut(&mut self, slot: (usize, usize)) -> &mut [u8] {
        let n = self.slot_bytes;
        &mut self.slabs[slot.0][slot.1..slot.1 + n]
    }

    fn slots(&self) -> usize {
        self.slabs.iter().map(|s| s.len() / self.slot_bytes.max(1)).sum()
    }
}

impl Drop for Arena {
    fn drop(&mut self) {
        #[cfg(feature = "cuda")]
        if self.pinned {
            for s in &mut self.slabs {
                crate::gpu::host_unregister(s.as_mut_ptr());
            }
        }
    }
}

struct Page {
    slot: (usize, usize),
    refs: u32,
}

struct Snapshot {
    tokens: Vec<Token>,
    /// Hash-chain ids of the full pages, in order.
    pages: Vec<u128>,
    /// The rows past the last full page (fewer than a page), in a page slot.
    tail: Option<(usize, usize)>,
    /// The positional state.
    state: (usize, usize),
    after: After,
    kind: Kind,
    last_use: u64,
}

/// Hash chain over `page_tokens`-token blocks: `ids[i]` identifies the prefix
/// `tokens[..page_tokens * (i + 1)]` (two SipHash lanes with fixed seeds, 128 bits).
fn page_chain(tokens: &[Token], page_tokens: usize) -> Vec<u128> {
    let mut prev = 0u128;
    tokens
        .chunks_exact(page_tokens)
        .map(|block| {
            let lane = |seed: u64| {
                let mut h = std::collections::hash_map::DefaultHasher::new();
                seed.hash(&mut h);
                prev.hash(&mut h);
                block.hash(&mut h);
                h.finish()
            };
            prev = (u128::from(lane(0x6d69_6d6f)) << 64) | u128::from(lane(0x3236_6b76));
            prev
        })
        .collect()
}

/// Counters for the log line and receipts.
#[derive(Default, Debug, Clone, Copy, PartialEq, Eq)]
pub struct Stats {
    pub captures: u64,
    pub restores: u64,
    pub restored_tokens: u64,
    pub evicted: u64,
    pub pages_written: u64,
}

/// The tier's shape: slot counts and sizes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct HostTierConfig {
    /// Page slots (full pages and snapshot tails).
    pub pages: usize,
    /// Positional-state slots (one per snapshot).
    pub states: usize,
    pub page_tokens: usize,
    pub page_bytes: usize,
    pub state_bytes: usize,
    /// Shortest snapshot worth storing (the design's `--host-cache-min-tokens`).
    pub min_tokens: usize,
    /// Page-lock the arenas (feature `cuda`; ignored without it).
    pub pin: bool,
}

impl HostTierConfig {
    /// Split `budget` bytes so that conversations of [`BALANCE_TOKENS`] tokens, each keeping two
    /// snapshots (prompt end and turn end, which share their pages), fill the page and the state
    /// slots at the same rate: states get `2 S / (2 S + P)` of the budget, `S` a state's bytes
    /// and `P` the page bytes of one such conversation. At least 4 and at most 256 states; at
    /// least 16 pages.
    ///
    /// GLM-5.3-Flash (FP8 cache, 141 MiB states): a 42% share; 64 GiB then holds 196 states and
    /// 100,624 pages (6.4 M tokens), 98 retained 64K conversations, as docs/SIZING.md section 7
    /// counts. With MiMo's sizes the share is 8.5%, near the source's fixed 10%.
    pub fn balanced(budget: usize, page_tokens: usize, page_bytes: usize, state_bytes: usize) -> HostTierConfig {
        let conv_pages = BALANCE_TOKENS.div_ceil(page_tokens.max(1)) as f64 * page_bytes as f64;
        let share = 2.0 * state_bytes as f64 / (2.0 * state_bytes as f64 + conv_pages);
        let states = ((budget as f64 * share) as usize / state_bytes.max(1)).clamp(4, 256);
        let pages = (budget.saturating_sub(states * state_bytes) / page_bytes.max(1)).max(16);
        HostTierConfig { pages, states, page_tokens, page_bytes, state_bytes, min_tokens: 512, pin: true }
    }
}

/// The host KV tier: page and state arenas, the shared page map, the snapshots.
pub struct HostCache {
    cfg: HostTierConfig,
    pages: Arena,
    states: Arena,
    page_map: HashMap<u128, Page>,
    snaps: BTreeMap<u64, Snapshot>,
    index: RadixIndex<u64>,
    next_id: u64,
    clock: u64,
    pub stats: Stats,
}

impl HostCache {
    /// A tier of the given shape (every slot allocated now).
    pub fn new(cfg: HostTierConfig) -> Result<Self, String> {
        if cfg.page_tokens == 0 || cfg.page_bytes == 0 || cfg.state_bytes == 0 {
            return Err(format!("hostcache: empty page or state size in {cfg:?}"));
        }
        let t0 = std::time::Instant::now();
        let me = Self {
            pages: Arena::new(cfg.page_bytes, cfg.pages, cfg.pin)?,
            states: Arena::new(cfg.state_bytes, cfg.states, cfg.pin)?,
            index: RadixIndex::new(cfg.page_tokens),
            cfg,
            page_map: HashMap::new(),
            snaps: BTreeMap::new(),
            next_id: 0,
            clock: 0,
            stats: Stats::default(),
        };
        if cfg.pages * cfg.page_bytes + cfg.states * cfg.state_bytes >= 1 << 30 {
            eprintln!(
                "[hostcache] {} pages of {} B ({} tokens) + {} states of {} B, {:.1} GiB{} in {:.1} s",
                cfg.pages,
                cfg.page_bytes,
                cfg.pages * cfg.page_tokens,
                cfg.states,
                cfg.state_bytes,
                (cfg.pages * cfg.page_bytes + cfg.states * cfg.state_bytes) as f64 / (1u64 << 30) as f64,
                if me.pages.pinned { " page-locked" } else { "" },
                t0.elapsed().as_secs_f64()
            );
        }
        Ok(me)
    }

    /// `GLM53F_HOST_CACHE_GB` of RAM (0 = off) shaped for `slot`'s pages and states
    /// ([`HostTierConfig::balanced`]). Unset, the default adapts to the host: min(32, 40% of
    /// `MemAvailable` at boot) GiB, so the recipe carries to coordinator hosts with less RAM.
    pub fn from_env<S: KvSlot>(slot: &S) -> Result<Option<Self>, String> {
        let gb: f64 = match std::env::var("GLM53F_HOST_CACHE_GB").ok().and_then(|v| v.parse().ok()) {
            Some(g) => g,
            None => {
                let avail_kb: f64 = std::fs::read_to_string("/proc/meminfo")
                    .ok()
                    .and_then(|m| m.lines().find(|l| l.starts_with("MemAvailable:")).map(str::to_string))
                    .and_then(|l| l.split_whitespace().nth(1).and_then(|v| v.parse().ok()))
                    .unwrap_or(0.0);
                (0.4 * avail_kb / (1u64 << 20) as f64).min(32.0)
            }
        };
        if gb <= 0.0 {
            return Ok(None);
        }
        let budget = (gb * (1u64 << 30) as f64) as usize;
        let cfg = HostTierConfig::balanced(budget, slot.page_tokens(), slot.page_bytes(), slot.state_bytes());
        Self::new(cfg).map(Some)
    }

    pub fn config(&self) -> &HostTierConfig {
        &self.cfg
    }

    /// Snapshots held.
    pub fn len(&self) -> usize {
        self.snaps.len()
    }

    pub fn is_empty(&self) -> bool {
        self.snaps.is_empty()
    }

    /// Distinct full pages held.
    pub fn pages_held(&self) -> usize {
        self.page_map.len()
    }

    /// Free page and state slots.
    pub fn free_slots(&self) -> (usize, usize) {
        (self.pages.free.len(), self.states.free.len())
    }

    /// Page and state slots in all.
    pub fn slots(&self) -> (usize, usize) {
        (self.pages.slots(), self.states.slots())
    }

    /// `(tokens, kind)` of every snapshot held, oldest first.
    pub fn snapshots(&self) -> Vec<(Vec<Token>, Kind)> {
        self.snaps.values().map(|s| (s.tokens.clone(), s.kind)).collect()
    }

    /// Evict one snapshot ([`victim`]); false when none is left.
    fn evict_one(&mut self) -> bool {
        let Some(i) = victim(self.snaps.values().map(|s| (s.kind, s.last_use))) else {
            return false;
        };
        let id = *self.snaps.keys().nth(i).expect("victim index");
        let s = self.snaps.remove(&id).expect("victim");
        self.index.remove(&s.tokens, |&x| x == id);
        self.release(s);
        self.stats.evicted += 1;
        true
    }

    fn release(&mut self, s: Snapshot) {
        self.states.free.push(s.state);
        if let Some(t) = s.tail {
            self.pages.free.push(t);
        }
        self.unref(s.pages);
    }

    /// Drop one reference to each page; a page left without one goes free.
    fn unref(&mut self, ids: Vec<u128>) {
        for id in ids {
            if let Some(p) = self.page_map.get_mut(&id) {
                p.refs -= 1;
                if p.refs == 0 {
                    let slot = p.slot;
                    self.page_map.remove(&id);
                    self.pages.free.push(slot);
                }
            }
        }
    }

    fn alloc_page(&mut self) -> Option<(usize, usize)> {
        loop {
            if let Some(s) = self.pages.free.pop() {
                return Some(s);
            }
            if !self.evict_one() {
                return None;
            }
        }
    }

    fn alloc_state(&mut self) -> Option<(usize, usize)> {
        loop {
            if let Some(s) = self.states.free.pop() {
                return Some(s);
            }
            if !self.evict_one() {
                return None;
            }
        }
    }

    /// Store a snapshot of `tokens` with `after` for the token after it: the appendable rows
    /// `[0, tokens.len())` from `slot` (which holds at least that many, and these tokens), and
    /// the positional state of `mark` (taken in `slot` at `tokens.len()`). Pages already held
    /// are shared, not copied. On exhaustion the capture is dropped (the request is unaffected);
    /// an identical snapshot already held is refreshed instead.
    pub fn capture<S: KvSlot>(&mut self, slot: &S, tokens: &[Token], after: &After, kind: Kind, mark: &S::Mark)
        -> Result<(), String> {
        self.capture_for(slot, tokens, after, kind, mark, "")
    }

    /// [`Self::capture`], its log line saying why the snapshot is stored (`why`; empty: nothing).
    pub(crate) fn capture_for<S: KvSlot>(
        &mut self,
        slot: &S,
        tokens: &[Token],
        after: &After,
        kind: Kind,
        mark: &S::Mark,
        why: &str,
    ) -> Result<(), String> {
        let n = tokens.len();
        if n < self.cfg.min_tokens || slot.tokens() < n {
            return Ok(());
        }
        self.clock += 1;
        if let Some(&id) = self.index.get(tokens).first() {
            let s = self.snaps.get_mut(&id).expect("indexed snapshot");
            s.last_use = self.clock;
            s.after.greedy = s.after.greedy.or(after.greedy);
            if s.after.logits.is_none() {
                s.after.logits = after.logits.clone();
            }
            return Ok(());
        }
        let t0 = std::time::Instant::now();
        let pt = self.cfg.page_tokens;
        let chain = page_chain(tokens, pt);
        let mut held: Vec<u128> = Vec::with_capacity(chain.len());
        let mut written = 0u64;
        for (i, &id) in chain.iter().enumerate() {
            if let Some(p) = self.page_map.get_mut(&id) {
                p.refs += 1;
                held.push(id);
                continue;
            }
            let Some(ps) = self.alloc_page() else {
                self.unref(held);
                return Ok(());
            };
            if let Err(e) = slot.export_page(i * pt, pt, self.pages.get_mut(ps)) {
                self.pages.free.push(ps);
                self.unref(held);
                return Err(e);
            }
            self.page_map.insert(id, Page { slot: ps, refs: 1 });
            held.push(id);
            written += 1;
        }
        let rem = n % pt;
        let tail = if rem > 0 {
            let Some(ps) = self.alloc_page() else {
                self.unref(held);
                return Ok(());
            };
            if let Err(e) = slot.export_page(n - rem, rem, self.pages.get_mut(ps)) {
                self.pages.free.push(ps);
                self.unref(held);
                return Err(e);
            }
            Some(ps)
        } else {
            None
        };
        let give_back = |me: &mut Self, held: Vec<u128>| {
            if let Some(t) = tail {
                me.pages.free.push(t);
            }
            me.unref(held);
        };
        if let Err(e) = slot.sync() {
            give_back(self, held);
            return Err(e);
        }
        let Some(ss) = self.alloc_state() else {
            give_back(self, held);
            return Ok(());
        };
        if let Err(e) = slot.export_state(mark, self.states.get_mut(ss)) {
            self.states.free.push(ss);
            give_back(self, held);
            return Err(e);
        }
        let id = self.next_id;
        self.next_id += 1;
        self.index.insert(tokens, id);
        self.snaps.insert(id, Snapshot {
            tokens: tokens.to_vec(),
            pages: held,
            tail,
            state: ss,
            after: after.clone(),
            kind,
            last_use: self.clock,
        });
        self.stats.captures += 1;
        self.stats.pages_written += written;
        let why = if why.is_empty() { String::new() } else { format!(" ({why})") };
        eprintln!("[hostcache] store {kind:?} snapshot {n} tokens{why}: {written} new of {} pages, {:.1} ms ({} \
            snapshots, {} pages held)", chain.len(), t0.elapsed().as_secs_f64() * 1e3, self.snaps.len(),
            self.page_map.len());
        Ok(())
    }

    /// The longest snapshot whose tokens are a prefix of `prompt`: `(id, tokens)`.
    pub fn lookup(&self, prompt: &[Token]) -> Option<(u64, usize)> {
        let m = self.index.lookup(prompt, |_, _| true);
        m.hit.map(|(n, ids)| (*ids[0], n))
    }

    /// [`Self::lookup`] for a request that is `sampled` or not: a snapshot of the whole prompt
    /// that cannot give it its first token (no kept logits for a sampled request, no argmax for
    /// a greedy one) gives way to the longest shorter one.
    pub fn lookup_for(&self, prompt: &[Token], sampled: bool) -> Option<(u64, usize)> {
        let m = self.index.lookup(prompt, |len, id| len < prompt.len() || self.snaps[id].after.serves(sampled));
        m.hit.map(|(n, ids)| (*ids[0], n))
    }

    /// How much of `prompt` the held snapshots share, in whole pages (for statistics).
    pub fn shared(&self, prompt: &[Token]) -> usize {
        self.index.lookup(prompt, |_, _| false).shared
    }

    /// Load snapshot `id` into `slot` (reset here and reserved for the snapshot). Returns
    /// `(tokens restored, what follows the snapshot, its bank)`.
    pub fn restore<S: KvSlot>(&mut self, id: u64, slot: &mut S) -> Result<(usize, After, Kind), String> {
        let t0 = std::time::Instant::now();
        self.clock += 1;
        let pt = self.cfg.page_tokens;
        let s = self.snaps.get_mut(&id).ok_or("hostcache: no such snapshot")?;
        s.last_use = self.clock;
        let n = s.tokens.len();
        slot.reset();
        slot.reserve(n)?;
        for pid in &s.pages {
            let p = self.page_map.get(pid).ok_or("hostcache: snapshot page missing")?;
            slot.import_page(pt, self.pages.get(p.slot))?;
        }
        if let Some(t) = s.tail {
            slot.import_page(n % pt, self.pages.get(t))?;
        }
        slot.import_state(n, self.states.get(s.state))?;
        let (after, kind) = (s.after.clone(), s.kind);
        self.stats.restores += 1;
        self.stats.restored_tokens += n as u64;
        eprintln!("[hostcache] restore {n} tokens in {:.1} ms ({} restores, {} tokens total)",
            t0.elapsed().as_secs_f64() * 1e3, self.stats.restores, self.stats.restored_tokens);
        Ok((n, after, kind))
    }
}

/// The snapshot RAM deletes next: the least recently used; at equal use a prompt snapshot before a
/// turn snapshot; at a full tie the first. (A conversation's prompt snapshot is stored just before
/// its turn snapshot, so the pair goes prompt first; across conversations age decides.) The device
/// evicts its points by the same rule (`crate::pool`).
pub(crate) fn victim(snaps: impl Iterator<Item = (Kind, u64)>) -> Option<usize> {
    snaps.enumerate().min_by_key(|&(_, (kind, last))| (last, kind == Kind::Turn)).map(|(i, _)| i)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn eviction_is_least_recently_used_first() {
        // An old conversation's turn snapshot goes before a fresh conversation's prompt snapshot.
        let snaps = [(Kind::Turn, 3), (Kind::Prompt, 9), (Kind::Turn, 10)];
        assert_eq!(victim(snaps.into_iter()), Some(0));
        // At equal use the prompt snapshot goes first.
        let snaps = [(Kind::Turn, 5), (Kind::Prompt, 5)];
        assert_eq!(victim(snaps.into_iter()), Some(1));
        assert_eq!(victim(std::iter::empty()), None);
    }

    #[test]
    fn page_chain_identifies_prefixes() {
        for pt in [256usize, 64] {
            let a: Vec<Token> = (0..(4 * pt - 24) as Token).collect();
            let mut b = a.clone();
            b[2 * pt + 10] = 5;
            let (ca, cb) = (page_chain(&a, pt), page_chain(&b, pt));
            assert_eq!(ca.len(), 3);
            assert_eq!(ca[..2], cb[..2]);
            assert_ne!(ca[2], cb[2]);
            // A page's id depends on everything before it, not just its own block.
            let mut c = a.clone();
            c[3] = 9;
            let cc = page_chain(&c, pt);
            assert!(cc.iter().zip(&ca).all(|(x, y)| x != y));
            assert_eq!(page_chain(&a[..pt - 1], pt).len(), 0);
        }
    }

    #[test]
    fn balanced_split_matches_the_sizing_numbers() {
        const GIB: usize = 1 << 30;
        // GLM-5.3-Flash, FP8 cache: 64-token pages of 6,171 B per token; 141 MiB states.
        let page = 64 * 6_171;
        let state = 141 << 20;
        let c = HostTierConfig::balanced(64 * GIB, 64, page, state);
        assert_eq!((c.states, c.pages), (196, 100_624));
        let conversations = (c.pages / (65_536 / 64)).min(c.states / 2);
        assert_eq!(conversations, 98, "docs/SIZING.md section 7: about 98 retained 64K conversations");
        // MiMo's shape: 256-token pages of 11,520 B per token, 35 MB states: 8.5%, near its 10%.
        let m = HostTierConfig::balanced(32 * GIB, 256, 256 * 11_520, 35_000_000);
        let share = (m.states * 35_000_000) as f64 / (32 * GIB) as f64;
        assert!((0.08..0.09).contains(&share), "MiMo state share {share}");
        // Tiny budgets keep the floors.
        let t = HostTierConfig::balanced(1024, 4, 64, 256);
        assert_eq!((t.states, t.pages), (4, 16));
    }
}
