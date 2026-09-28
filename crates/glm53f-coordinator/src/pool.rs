//! The KV slots and snapshot tiers (mimo26f-afd perf reset K3, the DS41RT host cache design in
//! its `on-evict` store mode): free slots; retained slots holding snapshot points (the device
//! tier: no copies while nothing is under pressure); the RAM tier behind them. A point goes to
//! RAM only when the device evicts it: its bank over `bank` points, or its slot or the slot's
//! memory needed by a request.
//!
//! A point is an exact position in a slot's token history: the slot's appendable rows
//! `[0, len)` are the point's (append-only, so they stay valid while the slot runs on), and its
//! positional state at `len` is a [`KvSlot::Mark`]. Saving one is a device-side copy, never RAM
//! traffic.
//!
//! Every point is indexed by its tokens in a [`RadixIndex`] (the source scanned every retained
//! slot's history). Points of running requests are indexed too: a new prompt that extends a
//! running request's prompt forks that point into a free slot (the source only resumed from
//! retained slots, so a prompt deferred behind an identical prefill prefilled again once the
//! first request was running).

use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{mpsc, Arc};

use crate::hostcache::{HostCache, Kind};
use crate::model::{ImageSpan, KvSlot, ModelForward, Token};
use crate::radix::RadixIndex;
use crate::sampling::{After, Sampling};

/// A retained snapshot on the device: an exact position in a slot's token history.
pub(crate) struct Point<M> {
    pub(crate) id: u64,
    pub(crate) len: usize,
    /// The token after the snapshot (an exact hit that it serves needs no forward).
    pub(crate) after: After,
    pub(crate) kind: Kind,
    pub(crate) mark: M,
    pub(crate) last_use: u64,
}

/// A finished request's slot kept on the device for its points (no copies).
pub(crate) struct Retained<S: KvSlot> {
    pub(crate) slot: S,
    /// The tokens of the slot's rows (at least up to its last point).
    pub(crate) hist: Vec<Token>,
    pub(crate) points: Vec<Point<S::Mark>>,
}

impl<S: KvSlot> Retained<S> {
    fn last_use(&self) -> u64 {
        self.points.iter().map(|p| p.last_use).max().unwrap_or(0)
    }

    fn top(&self) -> usize {
        self.points.iter().map(|p| p.len).max().unwrap_or(0)
    }
}

/// A request being decoded: its slot and progress.
pub(crate) struct Active<S: KvSlot> {
    pub(crate) slot: S,
    pub(crate) last: Token,
    pub(crate) generated: usize,
    pub(crate) max: usize,
    pub(crate) tx: mpsc::Sender<Result<Token, String>>,
    pub(crate) cancel: Arc<AtomicBool>,
    /// The prompt then every token sent (the slot holds all but the last).
    pub(crate) hist: Vec<Token>,
    /// Snapshot points inside this slot's history.
    pub(crate) points: Vec<Point<S::Mark>>,
    pub(crate) sampling: Option<Sampling>,
}

/// A request whose prompt is still being prefilled (mimo26f-afd perf reset Q1): one segment per
/// scheduler round, a decode step for the running requests between segments, so a long prompt
/// does not stall every other stream.
pub(crate) struct Prefilling<S: KvSlot> {
    pub(crate) slot: S,
    pub(crate) ids: Vec<Token>,
    /// Prompt tokens in the slot so far.
    pub(crate) done: usize,
    /// What is known about the token after `done` (from the last segment).
    pub(crate) after: Option<After>,
    pub(crate) points: Vec<Point<S::Mark>>,
    pub(crate) max: usize,
    pub(crate) tx: mpsc::Sender<Result<Token, String>>,
    pub(crate) cancel: Arc<AtomicBool>,
    /// The prompt's images.
    pub(crate) images: Vec<ImageSpan>,
    pub(crate) sampling: Option<Sampling>,
}

/// An admitted request's slot, the points it carries, and where it resumes: `(tokens, what
/// follows)`, or none for a cold prefill.
pub(crate) type Admitted<S> = (S, Vec<Point<<S as KvSlot>::Mark>>, Option<(usize, After)>);

/// Where a device hit lives.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Owner {
    /// A retained slot: used in place, or forked.
    Retained(usize),
    /// A running request: forked only.
    Active(usize),
    /// A request still prefilling (a point it resumed from): forked only.
    Prefilling(usize),
}

/// Counters for logs and tests.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PoolStats {
    /// Admissions resumed from a device point.
    pub device_hits: u64,
    /// Of those, forked into a fresh slot.
    pub forks: u64,
    /// Admissions resumed from the RAM tier.
    pub host_restores: u64,
    /// Prompt tokens not prefilled thanks to a resume.
    pub resumed_tokens: u64,
    /// Prompt tokens whose appendable KV existed (in whole granules) past the resume point:
    /// prefill a periodic checkpoint would have saved.
    pub branch_gap_tokens: u64,
    /// Retained slots evicted under pressure.
    pub evictions: u64,
    /// Points saved.
    pub points: u64,
}

/// The slots, retained points and the RAM tier.
pub(crate) struct Pool<S: KvSlot> {
    pub(crate) free: Vec<S>,
    pub(crate) retained: Vec<Retained<S>>,
    pub(crate) cache: Option<HostCache>,
    clock: u64,
    /// Points kept on the device per bank (the design's `--prefix-cache-entries`).
    bank: usize,
    /// Snapshots shorter than this are neither retained nor stored.
    pub(crate) min_retain: usize,
    /// Every live point by its tokens.
    index: RadixIndex<u64>,
    next_point: u64,
    pub(crate) stats: PoolStats,
}

/// Can `slot` take a request of `rows` tokens now?
pub(crate) fn admit_check<M: ModelForward>(model: &M, slot: &M::Slot, rows: usize) -> Result<(), String> {
    let need = slot.need_bytes(rows);
    let free = model.free_bytes()?;
    if need > free {
        return Err(format!(
            "KV pool: a {rows}-token request needs {:.2} GiB of GPU memory, {:.2} GiB is free; retry when other \
             requests finish",
            need as f64 / (1u64 << 30) as f64,
            free as f64 / (1u64 << 30) as f64
        ));
    }
    Ok(())
}

impl<S: KvSlot> Pool<S> {
    pub(crate) fn new(free: Vec<S>, cache: Option<HostCache>, bank: usize, min_retain: usize, granularity: usize) -> Self {
        Pool {
            free,
            retained: Vec::new(),
            cache,
            clock: 0,
            bank,
            min_retain,
            index: RadixIndex::new(granularity),
            next_point: 0,
            stats: PoolStats::default(),
        }
    }

    pub(crate) fn now(&mut self) -> u64 {
        self.clock += 1;
        self.clock
    }

    /// Points indexed (running, prefilling and retained).
    pub(crate) fn indexed(&self) -> usize {
        self.index.len()
    }

    /// A point at `slot`'s current position, indexed under `hist[..slot.tokens()]`; none when the
    /// device has no room for its mark.
    pub(crate) fn save_point(&mut self, slot: &S, hist: &[Token], after: After, kind: Kind, now: u64)
        -> Option<Point<S::Mark>> {
        let len = slot.tokens();
        match slot.mark() {
            Ok(mark) => {
                let id = self.next_point;
                self.next_point += 1;
                self.index.insert(&hist[..len], id);
                self.stats.points += 1;
                Some(Point { id, len, after, kind, mark, last_use: now })
            }
            Err(e) => {
                eprintln!("[coordinator] snapshot save failed: {e}");
                None
            }
        }
    }

    /// Drop points (their slot's history `hist`) from the index.
    pub(crate) fn forget(&mut self, hist: &[Token], points: &[Point<S::Mark>]) {
        for p in points {
            let removed = self.index.remove(&hist[..p.len], |&x| x == p.id);
            debug_assert!(removed.is_some(), "point {} not indexed", p.id);
        }
    }

    /// Store point `p` of `slot` (holding `hist`) to RAM, when the RAM tier is on.
    fn store(cache: &mut Option<HostCache>, slot: &S, hist: &[Token], p: &Point<S::Mark>) {
        if let Some(c) = cache.as_mut() {
            if let Err(e) = c.capture(slot, &hist[..p.len], &p.after, p.kind, &p.mark) {
                eprintln!("[hostcache] store failed: {e}");
            }
        }
    }

    /// Give a slot back: every row dropped, memory grown past its base returned.
    pub(crate) fn release(&mut self, mut slot: S) {
        slot.release();
        self.free.push(slot);
    }

    /// Fail-path release: the request's points leave the index, the slot goes back.
    pub(crate) fn discard(&mut self, slot: S, hist: &[Token], points: Vec<Point<S::Mark>>) {
        self.forget(hist, &points);
        drop(points);
        self.release(slot);
    }

    /// Keep `r` on the device.
    fn keep(&mut self, r: Retained<S>) {
        self.retained.push(r);
    }

    /// Evict the least recently used retained slot: its points to RAM, its slot back to the
    /// free list. False when none is left.
    pub(crate) fn evict_lru(&mut self) -> bool {
        let Some(i) = (0..self.retained.len()).min_by_key(|&i| self.retained[i].last_use()) else { return false };
        let r = self.retained.swap_remove(i);
        eprintln!("[coordinator] device pressure: evicting a retained {}-token slot ({} snapshots) to RAM",
            r.slot.tokens(), r.points.len());
        for p in &r.points {
            Self::store(&mut self.cache, &r.slot, &r.hist, p);
        }
        self.forget(&r.hist, &r.points);
        self.stats.evictions += 1;
        self.release(r.slot);
        true
    }

    /// A clean slot: a free one, else the least recently used retained one.
    pub(crate) fn take_free(&mut self) -> Option<S> {
        if self.free.is_empty() {
            self.evict_lru();
        }
        self.free.pop().map(|mut s| {
            s.reset();
            s
        })
    }

    /// Room on the device for `slot` to hold `rows` tokens, evicting retained slots (least
    /// recently used first) while it does not fit.
    pub(crate) fn make_room<M: ModelForward<Slot = S>>(&mut self, model: &M, slot: &S, rows: usize) -> Result<(), String> {
        loop {
            match admit_check(model, slot, rows) {
                Ok(()) => return Ok(()),
                Err(e) => {
                    if !self.evict_lru() {
                        return Err(e);
                    }
                }
            }
        }
    }

    /// `bytes` of free device memory, evicting retained slots (least recently used first)
    /// while there is not (mimo26f-afd perf reset V2: an image encoder's transient buffers).
    pub(crate) fn make_room_bytes<M: ModelForward<Slot = S>>(&mut self, model: &M, bytes: usize) -> Result<(), String> {
        loop {
            let free = model.free_bytes()?;
            if free >= bytes {
                return Ok(());
            }
            if !self.evict_lru() {
                return Err(format!("the image encoder needs {} MiB of GPU memory; {} MiB is free", bytes >> 20, free >> 20));
            }
        }
    }

    /// Where point `id` lives.
    fn locate(&self, active: &[Active<S>], prefilling: &VecDeque<Prefilling<S>>, id: u64) -> Option<(Owner, usize)> {
        let find = |points: &[Point<S::Mark>]| points.iter().position(|p| p.id == id);
        self.retained.iter().enumerate().find_map(|(i, r)| find(&r.points).map(|pi| (Owner::Retained(i), pi)))
            .or_else(|| active.iter().enumerate().find_map(|(i, a)| find(&a.points).map(|pi| (Owner::Active(i), pi))))
            .or_else(|| prefilling.iter().enumerate().find_map(|(i, p)| find(&p.points).map(|pi| (Owner::Prefilling(i), pi))))
    }

    fn point<'a>(&'a self, active: &'a [Active<S>], prefilling: &'a VecDeque<Prefilling<S>>, at: (Owner, usize))
        -> &'a Point<S::Mark> {
        match at.0 {
            Owner::Retained(i) => &self.retained[i].points[at.1],
            Owner::Active(i) => &active[i].points[at.1],
            Owner::Prefilling(i) => &prefilling[i].points[at.1],
        }
    }

    /// The longest point whose tokens prefix `ids` and that can serve a request that is
    /// `sampled` or not (a point at the prompt's full length must give this request its first
    /// token: kept logits for a sampled request, the argmax for a greedy one; otherwise the
    /// longest shorter point is used). Among points of that length: a retained slot's last point
    /// (used in place), then any retained point, then a running request's. Also returns the
    /// shared prefix ([`crate::radix::Match::shared`]).
    fn device_hit(&self, active: &[Active<S>], prefilling: &VecDeque<Prefilling<S>>, ids: &[Token], sampled: bool)
        -> (Option<(Owner, usize)>, usize) {
        let m = self.index.lookup(ids, |len, &id| {
            len < ids.len()
                || self.locate(active, prefilling, id).is_some_and(|at| self.point(active, prefilling, at).after.serves(sampled))
        });
        let shared = m.shared;
        let Some((len, ids_at)) = m.hit else { return (None, shared) };
        let rank = |at: (Owner, usize)| match at.0 {
            Owner::Retained(i) if self.retained[i].top() == len => 0,
            Owner::Retained(_) => 1,
            Owner::Active(_) | Owner::Prefilling(_) => 2,
        };
        let best = ids_at
            .into_iter()
            .filter_map(|&id| self.locate(active, prefilling, id))
            .filter(|&at| len < ids.len() || self.point(active, prefilling, at).after.serves(sampled))
            .min_by_key(|&at| rank(at));
        (best, shared)
    }

    /// A slot for a new request of `rows` tokens, resumed at the longest exact snapshot of
    /// `ids` on the device or in RAM (equal lengths: the device). Returns the slot, the points
    /// it carries and where it resumes, `(tokens, what follows)` (none: cold). An exact resume
    /// always serves the request (`sampled` or not).
    ///
    /// A hit on a retained slot's last point continues in that slot. A hit on an earlier point,
    /// or on a running request's point, forks it into a free slot when one is free and fits
    /// without evicting anything (the source's copy-on-write sharing; [`KvSlot::fork`]);
    /// otherwise a retained slot rewinds to the point ([`KvSlot::rewind`]) and its later points
    /// go to RAM, and a running request's point is not used.
    pub(crate) fn admit<M: ModelForward<Slot = S>>(
        &mut self,
        model: &M,
        active: &[Active<S>],
        prefilling: &VecDeque<Prefilling<S>>,
        ids: &[Token],
        rows: usize,
        sampled: bool,
    ) -> Result<Admitted<S>, String> {
        let now = self.now();
        let (dev, shared_dev) = self.device_hit(active, prefilling, ids, sampled);
        let shared = shared_dev.max(self.cache.as_ref().map_or(0, |c| c.shared(ids)));
        let dev_len = dev.map_or(0, |at| self.point(active, prefilling, at).len);
        let host_longer = dev_len < ids.len()
            && self.cache.as_ref().and_then(|c| c.lookup_for(ids, sampled)).is_some_and(|(_, n)| n > dev_len);
        let resumed = |me: &mut Self, n: usize| {
            me.stats.resumed_tokens += n as u64;
            me.stats.branch_gap_tokens += shared.saturating_sub(n) as u64;
        };
        if let (Some(at), false) = (dev, host_longer) {
            let in_place = matches!(at.0, Owner::Retained(i) if self.retained[i].top() == dev_len);
            // Fork: the point's rows shared (or copied) into a free slot that fits as it is.
            if !in_place {
                if let Some(mut slot) = self.free.pop() {
                    slot.reset();
                    let fits = admit_check(model, &slot, rows).is_ok();
                    let forked = fits && {
                        let (src, p) = match at.0 {
                            Owner::Retained(i) => (&self.retained[i].slot, &self.retained[i].points[at.1]),
                            Owner::Active(i) => (&active[i].slot, &active[i].points[at.1]),
                            Owner::Prefilling(i) => (&prefilling[i].slot, &prefilling[i].points[at.1]),
                        };
                        slot.reserve(rows)
                            .and_then(|_| slot.fork(src, dev_len, &p.mark))
                            .map_err(|e| eprintln!("[coordinator] snapshot fork failed: {e}"))
                            .is_ok()
                    };
                    if forked {
                        let after = match at.0 {
                            Owner::Retained(i) => {
                                let p = &mut self.retained[i].points[at.1];
                                p.last_use = now;
                                eprintln!("[coordinator] device hit: {dev_len} of {} prompt tokens ({:?} snapshot, forked)",
                                    ids.len(), p.kind);
                                p.after.clone()
                            }
                            _ => {
                                let p = self.point(active, prefilling, at);
                                eprintln!("[coordinator] device hit: {dev_len} of {} prompt tokens (a running request's \
                                    {:?} snapshot, forked)", ids.len(), p.kind);
                                p.after.clone()
                            }
                        };
                        self.stats.device_hits += 1;
                        self.stats.forks += 1;
                        resumed(self, dev_len);
                        return Ok((slot, Vec::new(), Some((dev_len, after))));
                    }
                    self.release(slot);
                }
            }
            if let Owner::Retained(si) = at.0 {
                // In place: the slot rewinds to the point; later points go to RAM.
                let mut r = self.retained.swap_remove(si);
                let x = &mut r.points[at.1];
                x.last_use = now;
                let (after, kind) = (x.after.clone(), x.kind);
                let (keep, later): (Vec<_>, Vec<_>) = r.points.into_iter().partition(|p| p.len <= dev_len);
                for p in &later {
                    Self::store(&mut self.cache, &r.slot, &r.hist, p);
                }
                self.forget(&r.hist, &later);
                drop(later);
                let mut slot = r.slot;
                let placed = self
                    .make_room(model, &slot, rows)
                    .and_then(|_| slot.reserve(rows))
                    .and_then(|_| {
                        let p = keep.iter().find(|p| p.len == dev_len).expect("the hit point");
                        slot.rewind(dev_len, &p.mark)
                    });
                match placed {
                    Ok(()) => {
                        eprintln!("[coordinator] device hit: {dev_len} of {} prompt tokens ({kind:?} snapshot, in place)",
                            ids.len());
                        self.stats.device_hits += 1;
                        resumed(self, dev_len);
                        return Ok((slot, keep, Some((dev_len, after))));
                    }
                    // Too little memory to grow the slot in place: with the RAM tier on, move it
                    // through RAM into a fresh slot of exactly the request's size (below: store,
                    // free, restore).
                    Err(e) if self.cache.is_some() => {
                        eprintln!("[coordinator] cannot grow a retained {}-token slot in place ({e}); relocating it \
                            through RAM", slot.tokens());
                        for p in &keep {
                            Self::store(&mut self.cache, &slot, &r.hist, p);
                        }
                        self.forget(&r.hist, &keep);
                        drop(keep);
                        self.release(slot);
                    }
                    Err(e) => {
                        // The slot keeps its points; the request is refused.
                        r.hist.truncate(dev_len);
                        self.keep(Retained { slot, hist: r.hist, points: keep });
                        return Err(e);
                    }
                }
            }
        }
        // A clean slot (evicting the least recently used retained slot if none is free), then the
        // RAM tier: looked up again, since making room stores to it.
        let mut slot = self.take_free().ok_or("no KV slot")?;
        if let Err(e) = self.make_room(model, &slot, rows).and_then(|_| slot.reserve(rows)) {
            self.release(slot);
            return Err(e);
        }
        let hit = self.cache.as_ref().and_then(|c| c.lookup_for(ids, sampled));
        if let Some((id, _)) = hit {
            let restored = self.cache.as_mut().expect("RAM tier").restore(id, &mut slot);
            match restored {
                Ok((n, after, kind)) => {
                    // The rebuilt snapshot joins the device bank (the design's restore).
                    let points = self.save_point(&slot, ids, after.clone(), kind, now).into_iter().collect();
                    self.stats.host_restores += 1;
                    resumed(self, n);
                    return Ok((slot, points, Some((n, after))));
                }
                Err(e) => {
                    eprintln!("[hostcache] restore failed, prefilling cold: {e}");
                    slot.reset();
                }
            }
        }
        resumed(self, 0);
        Ok((slot, Vec::new(), None))
    }

    /// Retire a finished request: its slot stays on the device with its points plus a
    /// completion-end turn point (device copies only), or is freed when it has none.
    pub(crate) fn retire(&mut self, mut a: Active<S>) {
        let n = a.hist.len() - 1;
        let cancelled = a.cancel.load(Ordering::Relaxed);
        // The slot holds the history but its last token: the scheduler commits exactly the rows
        // of the tokens it delivered, so there is nothing to truncate (a KDA state cannot be).
        if a.slot.tokens() != n || a.slot.pending() != 0 {
            eprintln!("[coordinator] a finished request's slot holds {} tokens (+{} pending), its history {n}; \
                freed without retaining it", a.slot.tokens(), a.slot.pending());
            self.discard(a.slot, &a.hist, a.points);
            return;
        }
        if !cancelled && n >= self.min_retain && a.points.iter().all(|p| p.len < n) {
            let now = self.now();
            // An identical snapshot already retained is refreshed, not kept twice (the design's
            // radix bank holds one entry per key).
            let dup = self.index.get(&a.hist[..n]).iter().find_map(|&id| {
                self.retained.iter().enumerate().find_map(|(si, r)| r.points.iter().position(|p| p.id == id).map(|pi| (si, pi)))
            });
            if let Some((si, pi)) = dup {
                self.retained[si].points[pi].last_use = now;
            } else {
                // A sampled request's last token is a draw, not the argmax there (perf reset V3).
                let after = After { greedy: a.sampling.is_none().then_some(a.hist[n] as usize), logits: None };
                if let Some(p) = self.save_point(&a.slot, &a.hist, after, Kind::Turn, now) {
                    a.points.push(p);
                }
            }
        }
        if a.points.is_empty() {
            self.release(a.slot);
            return;
        }
        a.hist.truncate(n);
        self.keep(Retained { slot: a.slot, hist: a.hist, points: a.points });
    }

    /// A prefill abandoned at `done` tokens (client gone, or an error after the forward): its
    /// position is kept as a snapshot, so a retry of the same prompt resumes there instead of
    /// prefilling again.
    pub(crate) fn park(&mut self, p: Prefilling<S>) {
        let Prefilling { mut slot, mut ids, done, after, mut points, .. } = p;
        slot.end_prefill();
        if let Some(after) = after.filter(|_| slot.tokens() == done && done >= self.min_retain) {
            if points.iter().all(|x| x.len < done) {
                let now = self.now();
                points.extend(self.save_point(&slot, &ids, after, Kind::Prompt, now));
            }
        }
        if points.is_empty() {
            self.release(slot);
            return;
        }
        ids.truncate(done);
        self.keep(Retained { slot, hist: ids, points });
    }

    /// Keep each bank within `bank` points on the device: the oldest point of an overflowing
    /// bank goes to RAM (the design's bank overflow). A retained slot left without points is
    /// freed.
    pub(crate) fn enforce_banks(&mut self, active: &mut [Active<S>]) {
        for kind in [Kind::Prompt, Kind::Turn] {
            loop {
                let mut count = 0;
                // (last use, in a running request, owner, point)
                let mut oldest: Option<(u64, bool, usize, usize)> = None;
                let owners = active.iter().map(|a| &a.points).chain(self.retained.iter().map(|r| &r.points));
                for (oi, points) in owners.enumerate() {
                    for (pi, p) in points.iter().enumerate().filter(|(_, p)| p.kind == kind) {
                        count += 1;
                        if oldest.is_none_or(|o| p.last_use < o.0) {
                            oldest = Some((p.last_use, oi < active.len(), oi, pi));
                        }
                    }
                }
                if count <= self.bank {
                    break;
                }
                let (_, running, oi, pi) = oldest.expect("a point to evict");
                if running {
                    let a = &mut active[oi];
                    let p = a.points.swap_remove(pi);
                    Self::store(&mut self.cache, &a.slot, &a.hist, &p);
                    self.forget(&a.hist, std::slice::from_ref(&p));
                } else {
                    let ri = oi - active.len();
                    let p = self.retained[ri].points.swap_remove(pi);
                    let r = &self.retained[ri];
                    Self::store(&mut self.cache, &r.slot, &r.hist, &p);
                    let hist = std::mem::take(&mut self.retained[ri].hist);
                    self.forget(&hist, std::slice::from_ref(&p));
                    self.retained[ri].hist = hist;
                    if self.retained[ri].points.is_empty() {
                        let r = self.retained.swap_remove(ri);
                        self.release(r.slot);
                    }
                }
            }
        }
    }

    /// Before a decode step: a request about to outgrow its reservation grows now, with room
    /// made first (a running request is not failed for pool pressure while retained slots can be
    /// evicted). `drafts`: the most rows a step may add beyond one.
    pub(crate) fn grow_active<M: ModelForward<Slot = S>>(&mut self, model: &M, active: &mut [Active<S>], drafts: usize) {
        for a in active.iter_mut() {
            let cap = a.slot.capacity();
            if a.slot.tokens() + 2 * drafts + 2 < cap {
                continue;
            }
            let target = cap + (cap / 8).max(4096);
            if let Err(e) = self.make_room(model, &a.slot, target).and_then(|_| a.slot.reserve(target)) {
                eprintln!("[coordinator] KV growth to {target} tokens failed: {e}");
            }
        }
    }
}
