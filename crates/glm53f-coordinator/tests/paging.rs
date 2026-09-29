//! KV snapshots and the RAM tier under the rule "no tax unless loaded": a snapshot point stays on
//! the device, uncopied, while nothing needs its memory; when an incoming request, or a new
//! point's mark, needs memory the device lacks, points are evicted least recently used first
//! wherever they live (retained slots, running and prefilling requests), each stored to RAM first
//! when the tier is on, until it fits and no further. A running request whose point is evicted
//! runs on.
//!
//! Against the toy model in `common`: device memory is 100 B per token of slot capacity (grown
//! in steps of 64 tokens from a base of 64) and 1,000 B per mark, so every eviction frees a known
//! amount. Every request's tokens must equal serial decoding of its prompt.

mod common;

use common::*;
use glm53f_coordinator::pool::PoolStats;
use glm53f_coordinator::{HostTierConfig, Kind, SchedulerConfig, Token};

fn tier(pages: usize, states: usize) -> HostTierConfig {
    let (page_tokens, page_bytes) = (PAGE, PAGE * 4);
    HostTierConfig { pages, states, page_tokens, page_bytes, state_bytes: 16, min_tokens: 16, pin: false }
}

/// No bank cap (the default; set here so `GLM53F_PREFIX_CACHE_ENTRIES` cannot change these tests).
fn no_cap(c: &mut SchedulerConfig) {
    c.bank = 0;
}

fn serial(p: &[Token], max: usize) -> Vec<Token> {
    reference(p, max, None, &[])
}

/// Lengths and banks of the RAM tier's snapshots, oldest first.
fn held(h: &Harness) -> Vec<(usize, Kind)> {
    h.sched.host_cache().expect("RAM tier").snapshots().into_iter().map(|(t, k)| (t.len(), k)).collect()
}

/// Tighten the device so that a reservation of `need` bytes lacks `short` bytes now.
fn tighten(h: &Harness, need: usize, short: usize) {
    let mut m = h.dev.lock().unwrap();
    m.budget = m.used + need - short;
}

/// No load, no tax: memory and slots to spare, no cap. 100 distinct prompts of 40-48 tokens keep
/// their 200 points on the device and store nothing to RAM; each repeat is a device hit (a fork,
/// nothing prefilled). The source's cap of 24 per bank moved 152 of those points to RAM.
#[test]
fn no_load_no_ram_traffic_and_every_repeat_is_a_device_hit() {
    let prompts: Vec<Vec<Token>> = (0..100u64).map(|i| prompt(100 + i, 40 + (i % 9) as usize)).collect();
    let first_pass = |tweak: fn(&mut SchedulerConfig)| {
        let mut h = harness(Setup { slots: 210, host: Some(tier(4096, 256)), tweak, ..Setup::default() });
        let outs: Vec<Vec<Token>> = prompts
            .iter()
            .map(|p| {
                let rx = h.submit(p, 4, None);
                h.run();
                tokens(&rx).unwrap()
            })
            .collect();
        for (p, out) in prompts.iter().zip(&outs) {
            assert_eq!(out, &serial(p, 4));
        }
        (h, outs)
    };

    let (mut h, outs) = first_pass(no_cap);
    let none = PoolStats::default();
    let quiet = |st: PoolStats| (st.evictions, st.evicted_points, st.evicted_in_flight, st.bank_overflows);
    assert_eq!(quiet(h.sched.pool_stats()), quiet(none), "nothing evicted");
    assert_eq!(h.sched.host_cache().unwrap().stats.captures, 0, "nothing stored to RAM");
    assert_eq!(h.sched.device_points(), 200, "a prompt and a turn point per conversation, on the device");
    assert_eq!(h.sched.slots(), (110, 100));

    // Every repeat resumes from its device point: the first token from the snapshot, no prefill.
    h.clear_log();
    for (p, out) in prompts.iter().zip(&outs) {
        let rx = h.submit(p, 4, None);
        h.run();
        assert_eq!(&tokens(&rx).unwrap(), out);
    }
    assert_eq!(h.prefilled(), 0, "every repeat served from a device point");
    let st = h.sched.pool_stats();
    assert_eq!((st.device_hits, st.forks, st.host_restores), (100, 100, 0));
    assert_eq!(quiet(st), quiet(none), "nothing evicted");
    let hc = h.sched.host_cache().unwrap();
    assert_eq!((hc.stats.captures, hc.stats.restores, hc.len()), (0, 0, 0), "no RAM traffic");

    // The source's cap: past 24 points per bank the oldest went to RAM, loaded or not.
    let (h24, _) = first_pass(|c| c.bank = 24);
    let st = h24.sched.pool_stats();
    assert_eq!((st.bank_overflows, st.evicted_points), (152, 0));
    assert_eq!(h24.sched.host_cache().unwrap().stats.captures, 152);
    assert_eq!(h24.sched.device_points(), 48);
    eprintln!(
        "100 distinct prompts, no cap: 0 RAM stores, 0 evictions, 200 device points; 100 repeats: 100 \
         device hits, 0 prefilled. With a cap of 24: 152 RAM stores"
    );
}

/// The pressure scene's incoming prompt: 176 tokens, 4 to generate, reserves 192 tokens, 128 of
/// capacity grown past the slot's base of 64 (`MockSlot::need_bytes`).
const X_LEN: usize = 176;
const X_NEED: usize = (192 - 64) * TOKEN_BYTES;

struct Scene {
    h: Harness,
    convs: Vec<Vec<Token>>,
    runs: Vec<Vec<Token>>,
    /// Tokens of the conversations, the two long requests and the incoming prompt, in that order.
    outs: Vec<Vec<Token>>,
    /// The pool's counters after the step that admitted the incoming prompt.
    at_admission: PoolStats,
}

/// Three finished conversations, one after another (retained, a prompt and a turn point each: the
/// six oldest points), then two long requests decoding (a prompt mark each, the newest), then an
/// incoming 176-token prompt. With `short`, the device is first tightened so that the prompt's
/// reservation lacks `short` bytes; without, memory is ample.
fn scene(host: bool, short: Option<usize>) -> Scene {
    let mut h = harness(Setup { slots: 8, host: host.then(|| tier(512, 32)), tweak: no_cap, ..Setup::default() });
    let convs: Vec<Vec<Token>> = (0..3u64).map(|i| prompt(300 + i, 40)).collect();
    let mut outs = Vec::new();
    for c in &convs {
        let rx = h.submit(c, 4, None);
        h.run();
        outs.push(tokens(&rx).unwrap());
    }
    let runs: Vec<Vec<Token>> = (0..2u64).map(|i| prompt(310 + i, 40)).collect();
    let rxs: Vec<_> = runs.iter().map(|p| h.submit(p, 60, None)).collect();
    for _ in 0..4 {
        h.sched.step(false);
    }
    assert_eq!(h.sched.in_flight().0, 2, "both long requests decoding");
    assert_eq!(h.sched.device_points(), 8, "six retained points and two running requests' marks");
    if let Some(short) = short {
        tighten(&h, X_NEED, short);
    }
    let x = prompt(320, X_LEN);
    let rx = h.submit(&x, 4, None);
    h.sched.step(false);
    let at_admission = h.sched.pool_stats();
    h.run();
    for rx in rxs.iter().chain([&rx]) {
        outs.push(tokens(rx).unwrap());
    }
    for (p, (out, max)) in convs.iter().chain(&runs).chain([&x]).zip(outs.iter().zip([4, 4, 4, 60, 60, 4])) {
        assert_eq!(out, &serial(p, max));
    }
    Scene { h, convs, runs, outs, at_admission }
}

/// Pressure only on demand. The incoming prompt lacks 6,500 B: exactly seven points go, least
/// recently used first (the three conversations' prompt and turn points, then the older running
/// request's mark; 7,000 B), each stored to RAM, and the newer running request's mark stays. The
/// running requests go on with the tokens of a run without pressure, and a later repeat of an
/// evicted conversation, and of the running request's prompt, restores from RAM. The points saved
/// afterwards make room the same way, one eviction each: the incoming prompt's own prompt point
/// takes the newer running request's mark; its turn point finds nothing else to evict (its prompt
/// point is its own) and is skipped; the first long request's turn point then takes that prompt
/// point, and its slot.
#[test]
fn an_incoming_prompt_evicts_exactly_enough_least_recently_used_points() {
    let calm = scene(true, None);
    assert_eq!(calm.at_admission.evicted_points, 0);
    assert_eq!(calm.h.sched.host_cache().unwrap().stats.captures, 0, "no pressure, no RAM traffic");

    let mut s = scene(true, Some(6_500));
    assert_eq!(s.outs, calm.outs, "the tokens of the run without pressure");
    let st = s.at_admission;
    assert_eq!((st.evicted_points, st.evicted_in_flight, st.evictions), (7, 1, 3), "{st:?}");
    // In eviction order: the conversations' points, oldest first, then the first request's mark;
    // then the two evicted for later points.
    use Kind::{Prompt, Turn};
    let order = [(40, Prompt), (43, Turn), (40, Prompt), (43, Turn), (40, Prompt), (43, Turn), (40, Prompt)];
    let held = held(&s.h);
    assert_eq!(held[..7], order);
    let stored: Vec<Vec<Token>> = s.h.sched.host_cache().unwrap().snapshots().into_iter().map(|(t, _)| t).collect();
    assert_eq!(stored[0], s.convs[0]);
    assert_eq!(stored[6], s.runs[0], "the older running request's mark; the newer one's stayed");
    assert_eq!(held[7..], [(40, Prompt), (X_LEN, Prompt)]);
    assert_eq!(stored[7], s.runs[1], "the newer running request's mark, for the incoming prompt's point");
    assert_eq!(s.h.sched.pool_stats().evicted_points, 9, "one eviction for each point that did not fit");

    // The first conversation and the first long request's prompt again: restored from RAM.
    s.h.clear_log();
    let r0 = s.h.submit(&s.convs[0], 4, None);
    s.h.run();
    let r1 = s.h.submit(&s.runs[0], 60, None);
    s.h.run();
    assert_eq!(tokens(&r0).unwrap(), s.outs[0]);
    assert_eq!(tokens(&r1).unwrap(), s.outs[3]);
    assert_eq!(s.h.prefilled(), 0, "both restored");
    assert_eq!(s.h.sched.pool_stats().host_restores, 2);
    eprintln!(
        "pressure: 7 points evicted to RAM for a 6,500 B shortfall (6 retained, 1 running mark), 2 more for \
         later points' marks; outputs unchanged; 2 restores"
    );
}

/// A running request's prompt mark is its only point; an incoming prompt lacks 500 B. The mark
/// goes to RAM, the request runs on with serial decoding's tokens, and its prompt later restores
/// from RAM (room for the restore evicts the request's own retained turn point, by then the least
/// recently used).
#[test]
fn a_running_request_s_mark_goes_to_ram_and_the_request_runs_on() {
    let mut h = harness(Setup { host: Some(tier(512, 32)), tweak: no_cap, ..Setup::default() });
    let r = prompt(400, 40);
    let rr = h.submit(&r, 60, None);
    for _ in 0..3 {
        h.sched.step(false);
    }
    assert_eq!((h.sched.in_flight().0, h.sched.device_points()), (1, 1));
    // 112 tokens reserve 128: 64 past the base.
    tighten(&h, 64 * TOKEN_BYTES, 500);
    let x = prompt(401, 112);
    let rx = h.submit(&x, 4, None);
    h.sched.step(false);
    let st = h.sched.pool_stats();
    assert_eq!((st.evicted_points, st.evicted_in_flight, st.evictions), (1, 1, 0));
    assert_eq!(held(&h), [(40, Kind::Prompt)]);
    assert_eq!(h.sched.host_cache().unwrap().snapshots()[0].0, r);
    h.run();
    assert_eq!(tokens(&rr).unwrap(), serial(&r, 60), "the running request's tokens");
    assert_eq!(tokens(&rx).unwrap(), serial(&x, 4));

    h.clear_log();
    let again = h.submit(&r, 60, None);
    h.run();
    assert_eq!(tokens(&again).unwrap(), serial(&r, 60));
    assert_eq!(h.prefilled(), 0, "restored from RAM");
    assert_eq!(h.sched.pool_stats().host_restores, 1);
}

/// A follow-up turn resumes its conversation's slot in place, carrying the prompt and turn points,
/// and is still prefilling its new message when another prompt, lacking 500 B, is admitted: the
/// conversation's prompt point (the least recently used; the turn point was just used) goes to RAM
/// from the prefilling request. Both requests give serial decoding's tokens, and the conversation's
/// first prompt later restores from RAM.
#[test]
fn a_prefilling_request_s_points_are_evicted_too() {
    let mut h = harness(Setup { host: Some(tier(512, 32)), tweak: no_cap, ..Setup::default() });
    let c = prompt(500, 40);
    let rc = h.submit(&c, 4, None);
    h.run();
    let out_c = tokens(&rc).unwrap();
    let mut f = c.clone();
    f.extend(&out_c[..3]);
    f.extend(prompt(501, 200));
    let x = prompt(502, 112);
    // The follow-up reserves 259 tokens (320 of capacity: 256 past the base) and fits; then the
    // other prompt's 64 past the base lack 500 B.
    tighten(&h, (256 + 64) * TOKEN_BYTES, 500);
    let rf = h.submit(&f, 4, None);
    let rx = h.submit(&x, 4, None);
    h.sched.step(false);
    let st = h.sched.pool_stats();
    assert_eq!((st.device_hits, st.evicted_points, st.evicted_in_flight), (1, 1, 1));
    assert_eq!(held(&h), [(40, Kind::Prompt)]);
    h.run();
    assert_eq!(tokens(&rf).unwrap(), serial(&f, 4));
    assert_eq!(tokens(&rx).unwrap(), serial(&x, 4));

    h.clear_log();
    let again = h.submit(&c, 4, None);
    h.run();
    assert_eq!(tokens(&again).unwrap(), out_c);
    assert_eq!(h.prefilled(), 0, "restored from RAM");
    assert_eq!(h.sched.pool_stats().host_restores, 1);
}

/// A running request outgrowing its reservation makes room the same way. Two finished
/// conversations (four points), then a request that must grow from 128 to 4,224 tokens of capacity
/// while the device lacks 4,500 B for it: the five least recently used points go to RAM, the last
/// of them its own prompt mark, and it runs on to serial decoding's tokens.
#[test]
fn a_growing_request_makes_room_the_same_way() {
    let mut h = harness(Setup { host: Some(tier(512, 32)), tweak: no_cap, ..Setup::default() });
    for i in 0..2u64 {
        let c = prompt(700 + i, 40);
        let rx = h.submit(&c, 4, None);
        h.run();
        assert_eq!(tokens(&rx).unwrap(), serial(&c, 4));
    }
    let r = prompt(710, 40);
    let rr = h.submit(&r, 100, None);
    for _ in 0..3 {
        h.sched.step(false);
    }
    // 40 + 32 + 8 tokens reserved (128 of capacity); past 126 it grows by 4,096.
    tighten(&h, 4096 * TOKEN_BYTES, 4_500);
    h.run();
    assert_eq!(tokens(&rr).unwrap(), serial(&r, 100), "the growing request's tokens");
    let st = h.sched.pool_stats();
    assert_eq!((st.evicted_points, st.evicted_in_flight, st.evictions), (5, 1, 2), "{st:?}");
    use Kind::{Prompt, Turn};
    assert_eq!(held(&h), [(40, Prompt), (43, Turn), (40, Prompt), (43, Turn), (40, Prompt)]);
    assert_eq!(h.sched.host_cache().unwrap().snapshots()[4].0, r, "its own prompt mark, last");
}

/// A long prompt's snapshots on a device its reservation filled (a mark that found no room used to
/// be skipped, so an identical repeat prefilled the whole prompt again). Twenty short conversations
/// are retained (a prompt and a turn point each), then a 600-token prompt's reservation takes the
/// device's last free bytes. Its prompt and turn marks make room as an admission does: the least
/// recently used points go, the first conversation's prompt and turn points (its slot freed), each
/// stored to RAM first, and nothing more. A repeat of the long prompt resumes from its prompt point
/// on the device, nothing prefilled, and the first conversation restores from RAM. With the tier
/// off the same two points are dropped.
#[test]
fn a_snapshot_without_room_makes_room_like_an_admission() {
    for host in [true, false] {
        let mut h = harness(Setup { slots: 24, host: host.then(|| tier(1024, 64)), tweak: no_cap, ..Setup::default() });
        let convs: Vec<Vec<Token>> = (0..20u64).map(|i| prompt(800 + i, 40)).collect();
        let outs: Vec<Vec<Token>> = convs
            .iter()
            .map(|c| {
                let rx = h.submit(c, 4, None);
                h.run();
                tokens(&rx).unwrap()
            })
            .collect();
        assert_eq!(h.sched.device_points(), 40, "a prompt and a turn point per conversation");
        // 600 tokens reserve 616: 640 of capacity, 576 past the slot's base. Nothing is left for a mark.
        tighten(&h, (640 - 64) * TOKEN_BYTES, 0);
        let long = prompt(900, 600);
        let r1 = h.submit(&long, 4, None);
        h.run();
        let out = tokens(&r1).unwrap();
        assert_eq!(out, serial(&long, 4));
        let st = h.sched.pool_stats();
        let stored = if host { held(&h) } else { Vec::new() };

        h.clear_log();
        let r2 = h.submit(&long, 4, None);
        h.run();
        assert_eq!(tokens(&r2).unwrap(), out);
        assert_eq!(h.prefilled(), 0, "the repeat prefilled again: the long prompt's snapshot was not kept");
        assert_eq!(h.sched.pool_stats().device_hits, 1);
        // Room for its two marks: the first conversation's two points, and no more.
        assert_eq!((st.evicted_points, st.evicted_in_flight, st.evictions), (2, 0, 1), "{st:?}");
        if host {
            assert_eq!(stored, [(40, Kind::Prompt), (43, Kind::Turn)]);
        }

        // The first conversation again: restored from RAM; with the tier off, prefilled again.
        h.clear_log();
        let r3 = h.submit(&convs[0], 4, None);
        h.run();
        assert_eq!(tokens(&r3).unwrap(), outs[0]);
        assert_eq!(h.prefilled(), if host { 0 } else { 40 });
        assert_eq!(h.sched.pool_stats().host_restores, u64::from(host));
    }
}

/// With the RAM tier off (`GLM53F_HOST_CACHE_GB=0`) the same seven points are evicted and dropped:
/// no RAM traffic, the tokens unchanged, and the evicted conversation prefills again.
#[test]
fn with_the_tier_off_evicted_points_are_dropped() {
    let calm = scene(false, None);
    let mut s = scene(false, Some(6_500));
    assert!(s.h.sched.host_cache().is_none());
    assert_eq!(s.outs, calm.outs);
    let st = s.at_admission;
    assert_eq!((st.evicted_points, st.evicted_in_flight, st.evictions), (7, 1, 3), "{st:?}");
    s.h.clear_log();
    let r0 = s.h.submit(&s.convs[0], 4, None);
    s.h.run();
    assert_eq!(tokens(&r0).unwrap(), s.outs[0]);
    assert_eq!(s.h.prefilled(), 40, "dropped, so prefilled again");
    assert_eq!(s.h.sched.pool_stats().host_restores, 0);
}

/// A bank cap (`GLM53F_PREFIX_CACHE_ENTRIES`) keeps the source's behaviour: 30 conversations with
/// room for all, a cap of 24: the six oldest prompt and turn points go to RAM, oldest first, and
/// their slots are freed. No cap is the default, and keeps all 60 on the device.
#[test]
fn a_bank_cap_moves_the_oldest_points_to_ram_as_before() {
    assert_eq!(SchedulerConfig::new(Vec::new()).bank, 0, "no cap by default");
    let convs: Vec<Vec<Token>> = (0..30u64).map(|i| prompt(600 + i, 40)).collect();
    let run = |tweak: fn(&mut SchedulerConfig)| {
        let mut h = harness(Setup { slots: 40, host: Some(tier(4096, 256)), tweak, ..Setup::default() });
        for c in &convs {
            let rx = h.submit(c, 4, None);
            h.run();
            assert_eq!(tokens(&rx).unwrap(), serial(c, 4));
        }
        h
    };
    let h = run(|c| c.bank = 24);
    let st = h.sched.pool_stats();
    assert_eq!((st.bank_overflows, st.evicted_points, st.evictions), (12, 0, 0));
    assert_eq!(h.sched.device_points(), 48);
    assert_eq!(h.sched.slots(), (16, 24));
    use Kind::{Prompt, Turn};
    assert_eq!(held(&h), [(40, Prompt), (43, Turn)].repeat(6));
    let stored: Vec<Vec<Token>> = h.sched.host_cache().unwrap().snapshots().into_iter().map(|(t, _)| t).collect();
    for (i, c) in convs[..6].iter().enumerate() {
        assert_eq!(&stored[2 * i], c, "conversation {i}'s prompt point");
    }

    let h = run(no_cap);
    assert_eq!(h.sched.pool_stats().bank_overflows, 0);
    assert_eq!(h.sched.host_cache().unwrap().stats.captures, 0);
    assert_eq!(h.sched.device_points(), 60);
    assert_eq!(h.sched.slots(), (10, 30));
}
