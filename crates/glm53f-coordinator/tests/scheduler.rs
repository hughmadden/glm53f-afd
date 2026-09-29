//! The scheduler, slot pool, prefix index and host tier against the toy model in `common`: every
//! request's tokens must equal serial decoding of its prompt ([`common::reference`]), whatever
//! the batching, segmenting, speculation, resumption or eviction; the passes the model saw
//! (`Call`) show how the work was scheduled.

mod common;

use common::*;
use glm53f_coordinator::sampling::Sampling;
use glm53f_coordinator::{HostTierConfig, Kind, Token};

fn greedy(h: &Harness, p: &[Token], max: usize) -> Vec<Token> {
    reference(p, max, None, &h.eos)
}

#[test]
fn greedy_requests_decode_together_and_match_serial_decoding() {
    let mut h = harness(Setup::default());
    let prompts = [prompt(1, 20), prompt(2, 30), prompt(3, 7)];
    let rxs: Vec<_> = prompts.iter().map(|p| h.submit(p, 12, None)).collect();
    h.run();
    for (p, rx) in prompts.iter().zip(&rxs) {
        assert_eq!(tokens(rx).unwrap(), greedy(&h, p, 12));
    }
    // The three short prompts prefilled in one pass, then decoded as one batch per step.
    let calls = h.calls();
    assert!(calls.iter().any(|c| matches!(c, Call::Prefill(v) if v.len() == 3)), "{calls:?}");
    let decodes: Vec<usize> = calls.iter().filter_map(|c| if let Call::Decode(v) = c { Some(v.len()) } else { None }).collect();
    assert_eq!(decodes.len(), 11, "one step per token after the first: {decodes:?}");
    assert!(decodes.iter().all(|&n| n == 3));
}

#[test]
fn greedy_sampled_and_seeded_rows_share_every_step() {
    let mut h = harness(Setup::default());
    let s1 = Sampling::new(0.8, 0.95, 0, 0.0, Some(11)).unwrap();
    let s2 = Sampling::new(1.3, 1.0, 5, 0.05, Some(12)).unwrap();
    let reqs = [(prompt(4, 30), None), (prompt(5, 25), s1), (prompt(6, 18), s2), (prompt(7, 22), None)];
    let rxs: Vec<_> = reqs.iter().map(|(p, s)| h.submit(p, 16, *s)).collect();
    h.run();
    for ((p, s), rx) in reqs.iter().zip(&rxs) {
        assert_eq!(tokens(rx).unwrap(), reference(p, 16, *s, &[]));
    }
    assert!(h.calls().iter().any(|c| matches!(c, Call::Decode(v) if v.len() == 4)), "mixed rows in one step");
}

#[test]
fn admission_waits_for_memory_and_refuses_what_never_fits() {
    // Two slots of 64 tokens; room to grow one request to 128 tokens (plus a few marks).
    let base = 2 * 64 * TOKEN_BYTES;
    let mut h = harness(Setup { slots: 2, budget: base + 64 * TOKEN_BYTES + 4 * MARK_BYTES, ..Setup::default() });
    let (a, b, c) = (prompt(8, 100), prompt(9, 100), prompt(10, 400));
    let ra = h.submit(&a, 20, None);
    let rb = h.submit(&b, 20, None);
    let rc = h.submit(&c, 20, None);
    h.run();
    assert_eq!(tokens(&ra).unwrap(), greedy(&h, &a, 20));
    // B waited for A's memory (A's retained slot was evicted to make room), then ran.
    assert_eq!(tokens(&rb).unwrap(), greedy(&h, &b, 20));
    assert!(h.sched.stats.stalls >= 1);
    // C cannot fit even on an idle device: refused, not left waiting.
    let e = tokens(&rc).unwrap_err();
    assert!(e.contains("KV pool"), "{e}");
    assert_eq!(h.sched.stats.refused, 1);
    assert!(h.sched.pool_stats().evictions >= 1);
    // Everything given back but the slots' base capacity and the retained snapshot's marks.
    let used = h.dev.lock().unwrap().used;
    assert!(used <= base + 128 * TOKEN_BYTES + 2 * MARK_BYTES, "used {used}");
}

#[test]
fn long_prompts_prefill_in_segments_between_decode_steps() {
    let mut h = harness(Setup::default());
    let r = prompt(11, 20);
    let rr = h.submit(&r, 200, None);
    for _ in 0..3 {
        h.sched.step(false);
    }
    let l = prompt(12, 600);
    let rl = h.submit(&l, 4, None);
    h.clear_log();
    h.run();
    assert_eq!(tokens(&rr).unwrap(), greedy(&h, &r, 200));
    assert_eq!(tokens(&rl).unwrap(), greedy(&h, &l, 4));
    // The long prompt went in segments of whole quanta, sized from the measured rate, and the
    // running request kept stepping: between two of its decode steps, one round of prefill, which
    // runs one segment of each prompt at most.
    let calls = h.calls();
    let segs: Vec<usize> = calls
        .iter()
        .filter_map(|c| match c {
            Call::Prefill(v) if v.len() == 1 && v[0].1 > 1 => Some(v[0].1),
            _ => None,
        })
        .collect();
    assert!(segs.len() >= 5, "600 tokens in segments of about 64-80: {segs:?}");
    assert_eq!(segs.iter().sum::<usize>(), 600);
    assert!(segs[..segs.len() - 1].iter().all(|&n| n % 16 == 0 && n <= 256), "whole quanta: {segs:?}");
    let (mut rows, mut worst, mut gaps) = (0usize, 0usize, 0usize);
    for c in &calls {
        match c {
            Call::Prefill(v) if v.len() == 1 && v[0].1 > 1 => rows += v[0].1,
            Call::Decode(_) => {
                gaps += usize::from(rows > 0);
                worst = worst.max(rows);
                rows = 0;
            }
            _ => {}
        }
    }
    // 20 ms at 0.25 ms per row plus 1 ms per pass: segments of 64 rows (17 ms), one a round, and a
    // decode step after every segment.
    assert!(worst <= 80, "a decode step waited behind {worst} prefill rows");
    assert_eq!(gaps, segs.len(), "decode steps between segments: {gaps} for {} segments", segs.len());
}

/// While a long prompt prefills, a running request keeps its share of the time
/// (`SchedulerConfig::decode_share`): after each prefill round it steps for `share / (1 - share)`
/// of the round's time (one step at least), greedy and speculative alike. The prompt's segments
/// stay the very same (the share moves only when the steps run), no step waits behind more than
/// one segment, the prompt completes, and every token is serial decoding's. With nothing running,
/// the prompt prefills without a pause.
#[test]
fn running_requests_keep_their_share_of_the_time_while_a_long_prompt_prefills() {
    type Tweak = fn(&mut glm53f_coordinator::SchedulerConfig);
    // The mock's simulated cost of a call: 1 ms a pass and 0.25 ms a row (drafts and commits free).
    let ms = |c: &Call| -> f64 {
        let rows: usize = match c {
            Call::Prefill(v) | Call::Verify(v) => v.iter().map(|x| x.1).sum(),
            Call::Decode(v) => v.len(),
            Call::Draft(_) | Call::Commit(_) => return 0.0,
        };
        1.0 + 0.25 * rows as f64
    };
    let (g, l) = (prompt(50, 20), prompt(51, 2000));
    let shares: [(f64, Tweak); 3] =
        [(0.0, |c| c.decode_share = 0.0), (0.2, |c| c.decode_share = 0.2), (0.5, |c| c.decode_share = 0.5)];
    for block in [0, 8] {
        let mut segments = Vec::new();
        for (share, tweak) in shares {
            let mut h = harness(Setup { block, tweak, ..Setup::default() });
            let rg = h.submit(&g, 1500, None);
            for _ in 0..3 {
                h.sched.step(false);
            }
            let rl = h.submit(&l, 4, None);
            h.clear_log();
            h.run();
            assert_eq!(tokens(&rg).unwrap(), greedy(&h, &g, 1500), "block {block}, share {share}");
            assert_eq!(tokens(&rl).unwrap(), greedy(&h, &l, 4), "block {block}, share {share}");
            // Every prefill pass is the long prompt's (the running request prefilled before).
            let calls = h.calls();
            let pre: Vec<usize> = (0..calls.len()).filter(|&i| matches!(calls[i], Call::Prefill(_))).collect();
            let rows = |i: usize| match &calls[i] {
                Call::Prefill(v) => v[0].1,
                _ => 0,
            };
            segments.push(pre.iter().map(|&i| rows(i)).collect::<Vec<_>>());
            // The steps between consecutive segments, and their share of the time over every round
            // but the last (the prompt then starts; nothing waits for another round).
            let (mut prefill, mut decode) = (0.0, 0.0);
            for w in pre.windows(2) {
                let steps = &calls[w[0] + 1..w[1]];
                let n = steps.iter().filter(|c| matches!(c, Call::Decode(_) | Call::Verify(_))).count();
                assert!(n >= 1, "block {block}, share {share}: no step between segments at calls {w:?}");
                if share == 0.0 {
                    assert_eq!(n, 1, "block {block}: one step a round without a share");
                }
                prefill += ms(&calls[w[0]]);
                decode += steps.iter().map(ms).sum::<f64>();
            }
            let kept = decode / (prefill + decode);
            eprintln!(
                "block {block}, share {share}: {} segments, the running request kept {:.1}% of the time",
                pre.len(),
                100.0 * kept
            );
            if share > 0.0 {
                // At least the share; at most one step a round more.
                assert!(kept >= share && kept < share + 0.1, "block {block}, share {share}: kept {kept:.3}");
            }
            // One segment at most between two steps (64 rows: 17 ms of the 20 ms target).
            assert!(pre.iter().all(|&i| rows(i) <= 64), "block {block}, share {share}: {:?}", segments.last());
        }
        assert!(segments.iter().all(|s| *s == segments[0]), "block {block}: the segments moved: {segments:?}");
        assert!(segments[0].len() >= 30 && segments[0].iter().sum::<usize>() == 2000, "{:?}", segments[0]);
    }
    // Nothing running: the prompt prefills without a step between its segments.
    let mut h = harness(Setup { tweak: |c| c.decode_share = 0.5, ..Setup::default() });
    let rl = h.submit(&l, 4, None);
    h.run();
    assert_eq!(tokens(&rl).unwrap(), greedy(&h, &l, 4));
    let calls = h.calls();
    let first_step = calls.iter().position(|c| matches!(c, Call::Decode(_))).unwrap();
    assert!(calls[..first_step].iter().all(|c| matches!(c, Call::Prefill(_))));
    assert_eq!(calls[..first_step].iter().map(|c| if let Call::Prefill(v) = c { v[0].1 } else { 0 }).sum::<usize>(), 2000);
}

#[test]
fn verify_windows_with_partial_accepts_equal_serial_decoding() {
    let eos = vec![5];
    let mut h = harness(Setup { block: 8, eos: eos.clone(), ..Setup::default() });
    let s = Sampling::new(0.9, 1.0, 0, 0.0, Some(99)).unwrap();
    let reqs = [(prompt(13, 30), None, 60), (prompt(14, 50), None, 37), (prompt(15, 20), s, 45), (prompt(16, 26), None, 3)];
    let rxs: Vec<_> = reqs.iter().map(|(p, s, m)| h.submit(p, *m, *s)).collect();
    h.run();
    for ((p, s, m), rx) in reqs.iter().zip(&rxs) {
        assert_eq!(tokens(rx).unwrap(), reference(p, *m, *s, &eos), "prompt {p:?}");
    }
    let calls = h.calls();
    let windows: Vec<(usize, usize)> = calls.iter().filter_map(|c| if let Call::Verify(v) = c { Some(v.clone()) } else { None })
        .flatten().collect();
    let commits: Vec<(usize, usize)> = calls.iter().filter_map(|c| if let Call::Commit(v) = c { Some(v.clone()) } else { None })
        .flatten().collect();
    assert_eq!(windows.len(), commits.len());
    assert!(windows.iter().all(|w| w.1 >= 1 && w.1 <= 8));
    let partial = windows.iter().zip(&commits).filter(|(w, c)| c.1 < w.1).count();
    let full = windows.iter().zip(&commits).filter(|(w, c)| w.1 > 1 && c.1 == w.1).count();
    assert!(partial > 0 && full > 0, "partial {partial}, full {full}");
    let st = h.sched.stats;
    assert!(st.drafts_accepted > 0 && st.drafts_accepted < st.drafts_verified, "{st:?}");
    assert_eq!(st.decode_steps, 0, "every step speculative");
}

/// The step's verify-row budget (`SchedulerConfig::spec_max_rows`): four requests draft 3 each a
/// step (the mock's drafts at probability 0.9 under the chain cut: 0.9, 0.81, 0.73), 16 rows. A
/// budget of 10 holds every verify pass to 10 rows, dropping the least likely drafts (the third
/// ones, then the later requests' second ones), and the tokens stay serial decoding's; a budget of
/// 16, or none, runs the very same passes.
#[test]
fn the_verify_row_budget_bounds_every_step() {
    let reqs = [prompt(40, 30), prompt(41, 25), prompt(42, 20), prompt(43, 28)];
    let run = |tweak: fn(&mut glm53f_coordinator::SchedulerConfig)| {
        let mut h = harness(Setup { block: 8, tweak, ..Setup::default() });
        let rxs: Vec<_> = reqs.iter().map(|p| h.submit(p, 24, None)).collect();
        h.run();
        for (p, rx) in reqs.iter().zip(&rxs) {
            assert_eq!(tokens(rx).unwrap(), greedy(&h, p, 24));
        }
        let verifies: Vec<Vec<(usize, usize)>> =
            h.calls().iter().filter_map(|c| if let Call::Verify(v) = c { Some(v.clone()) } else { None }).collect();
        (verifies, h.sched.stats)
    };
    let (free, _) = run(|c| c.spec_max_rows = 0);
    let (at16, _) = run(|c| c.spec_max_rows = 16);
    let (at10, st10) = run(|c| c.spec_max_rows = 10);
    let rows = |v: &[(usize, usize)]| v.iter().map(|w| w.1).sum::<usize>();
    assert_eq!(free, at16, "a budget the steps fit changed them");
    assert!(free.iter().any(|v| v.len() == 4 && rows(v) == 16), "{free:?}");
    assert!(at10.iter().all(|v| rows(v) <= 10), "{at10:?}");
    // Four requests at once: the budget's windows, the least likely drafts dropped.
    let full: Vec<&Vec<(usize, usize)>> = at10.iter().filter(|v| v.len() == 4).collect();
    assert!(!full.is_empty());
    assert!(full.iter().any(|v| v.iter().map(|w| w.1).collect::<Vec<_>>() == [3, 3, 2, 2]), "{full:?}");
    assert!(st10.spec_steps > 0 && st10.drafts_accepted > 0);
}

/// A request that stops inside an accepted run (its budget, or an end-of-sequence token) commits
/// only the rows of the tokens it delivered, so its slot holds exactly its history: the next turn
/// of the conversation resumes at the turn point in place, with no prefill of the old tokens.
#[test]
fn a_stop_inside_an_accepted_run_commits_only_what_was_delivered() {
    let mut h = harness(Setup { block: 8, ..Setup::default() });
    let p = prompt(17, 40);
    for max in [3usize, 5, 9, 13] {
        let rx = h.submit(&p, max, None);
        h.run();
        let out = tokens(&rx).unwrap();
        assert_eq!(out, greedy(&h, &p, max));
        // The follow-up turn: the history (all but the last token) plus a new message.
        let mut next = p.clone();
        next.extend(&out[..max - 1]);
        next.extend(prompt(18 + max as u64, 6));
        h.clear_log();
        let rx2 = h.submit(&next, 4, None);
        h.run();
        assert_eq!(tokens(&rx2).unwrap(), greedy(&h, &next, 4));
        assert_eq!(h.prefilled(), 6, "max {max}: only the new message prefilled");
    }
}

#[test]
fn radix_reuse_exact_extending_and_divergent_prompts() {
    let mut h = harness(Setup::default());
    let p = prompt(20, 40);
    let ra = h.submit(&p, 6, None);
    h.run();
    let out_a = tokens(&ra).unwrap();
    assert_eq!(out_a, greedy(&h, &p, 6));

    // Exact: the prompt point serves the first token; nothing is prefilled.
    h.clear_log();
    let rb = h.submit(&p, 6, None);
    h.run();
    assert_eq!(tokens(&rb).unwrap(), out_a);
    assert_eq!(h.prefilled(), 0, "exact repeat: no prefill");

    // Extending: the conversation's next turn resumes at the turn point (40 + 5 tokens).
    let mut c = p.clone();
    c.extend(&out_a[..5]);
    c.extend(prompt(21, 10));
    h.clear_log();
    let before = h.sched.pool_stats();
    let rc = h.submit(&c, 5, None);
    h.run();
    assert_eq!(tokens(&rc).unwrap(), greedy(&h, &c, 5));
    assert_eq!(h.prefilled(), 10);
    assert_eq!(h.sched.pool_stats().resumed_tokens - before.resumed_tokens, 45);

    // Divergent at the first generated token: back to the prompt point, the rest prefilled.
    let mut d = p.clone();
    d.push((out_a[0] + 1) % BOUND as Token);
    d.extend(prompt(22, 7));
    h.clear_log();
    let rd = h.submit(&d, 5, None);
    h.run();
    assert_eq!(tokens(&rd).unwrap(), greedy(&h, &d, 5));
    assert_eq!(h.prefilled(), 8);

    // Divergent inside the turn: 4 tokens past the prompt point exist (a whole pool) but no point
    // sits there; they are prefilled again and counted as the branch gap.
    let mut e = p.clone();
    e.extend(&out_a[..4]);
    e.push((out_a[4] + 3) % BOUND as Token);
    h.clear_log();
    let before = h.sched.pool_stats();
    let re = h.submit(&e, 5, None);
    h.run();
    assert_eq!(tokens(&re).unwrap(), greedy(&h, &e, 5));
    assert_eq!(h.prefilled(), 5);
    assert_eq!(h.sched.pool_stats().branch_gap_tokens - before.branch_gap_tokens, 4);
}

/// Identical prompts arriving together: the second waits for the first one's prefill, then forks
/// the running request's prompt point instead of prefilling the same tokens again.
#[test]
fn a_prompt_deferred_behind_an_identical_prefill_forks_its_snapshot() {
    let mut h = harness(Setup::default());
    let p = prompt(23, 100);
    let ra = h.submit(&p, 30, None);
    let rb = h.submit(&p, 30, None);
    let s = Sampling::new(1.0, 1.0, 0, 0.0, Some(3)).unwrap();
    let rc = h.submit(&p, 30, s);
    h.run();
    assert_eq!(tokens(&ra).unwrap(), greedy(&h, &p, 30));
    assert_eq!(tokens(&rb).unwrap(), greedy(&h, &p, 30));
    assert_eq!(tokens(&rc).unwrap(), reference(&p, 30, s, &[]));
    assert_eq!(h.prefilled(), 100, "the prompt prefilled once");
    assert_eq!(h.sched.stats.deferred, 2);
    assert_eq!(h.sched.pool_stats().forks, 2);
}

/// Bank overflow moves the oldest points to RAM; a returning prompt restores from there, and its
/// first token comes from the stored snapshot without a forward.
#[test]
fn snapshots_overflow_to_ram_and_restore_exactly() {
    let host = HostTierConfig { pages: 64, states: 8, page_tokens: PAGE, page_bytes: PAGE * 4, state_bytes: 16, min_tokens: 16, pin: false };
    let mut h = harness(Setup { host: Some(host), tweak: |c| c.bank = 1, ..Setup::default() });
    let (p1, p2) = (prompt(24, 43), prompt(25, 48));
    let r1 = h.submit(&p1, 4, None);
    h.run();
    let out1 = tokens(&r1).unwrap();
    let r2 = h.submit(&p2, 4, None);
    h.run();
    assert_eq!(tokens(&r2).unwrap(), greedy(&h, &p2, 4));
    let hc = h.sched.host_cache().unwrap();
    let held: Vec<(usize, Kind)> = hc.snapshots().into_iter().map(|(t, k)| (t.len(), k)).collect();
    assert!(held.contains(&(43, Kind::Prompt)) && held.contains(&(46, Kind::Turn)), "{held:?}");
    // The first conversation's prompt again: restored from RAM, first token without a forward.
    h.clear_log();
    let r3 = h.submit(&p1, 4, None);
    h.run();
    assert_eq!(tokens(&r3).unwrap(), out1);
    assert_eq!(h.prefilled(), 0);
    assert_eq!(h.sched.pool_stats().host_restores, 1);
    // And its next turn, from the turn snapshot (43 + 3 tokens), prefilling only the new tokens.
    let mut next = p1.clone();
    next.extend(&out1[..3]);
    next.extend(prompt(26, 9));
    h.clear_log();
    let r4 = h.submit(&next, 4, None);
    h.run();
    assert_eq!(tokens(&r4).unwrap(), greedy(&h, &next, 4));
    assert_eq!(h.prefilled(), 9);
}

/// Device memory pressure evicts a retained slot's points to RAM; the conversation comes back
/// through RAM.
#[test]
fn device_pressure_evicts_retained_slots_through_ram() {
    let host = HostTierConfig { pages: 256, states: 16, page_tokens: PAGE, page_bytes: PAGE * 4, state_bytes: 16, min_tokens: 16, pin: false };
    let base = 2 * 64 * TOKEN_BYTES;
    let mut h = harness(Setup { slots: 2, host: Some(host), budget: base + 200 * TOKEN_BYTES + 4 * MARK_BYTES, ..Setup::default() });
    let (a, b) = (prompt(27, 90), prompt(28, 200));
    let ra = h.submit(&a, 10, None);
    h.run();
    let out_a = tokens(&ra).unwrap();
    // B needs the device: A's retained slot goes to RAM.
    let rb = h.submit(&b, 10, None);
    h.run();
    assert_eq!(tokens(&rb).unwrap(), greedy(&h, &b, 10));
    assert!(h.sched.pool_stats().evictions >= 1);
    // A's next turn restores from RAM.
    let mut next = a.clone();
    next.extend(&out_a[..9]);
    next.extend(prompt(29, 12));
    h.clear_log();
    let rn = h.submit(&next, 6, None);
    h.run();
    assert_eq!(tokens(&rn).unwrap(), greedy(&h, &next, 6));
    assert_eq!(h.prefilled(), 12);
    assert!(h.sched.pool_stats().host_restores >= 1);
}

/// A client that leaves mid-prefill: the prefix is parked as a snapshot, and a retry resumes it.
#[test]
fn a_client_gone_mid_prefill_parks_its_prefix_for_a_retry() {
    let mut h = harness(Setup::default());
    let p = prompt(30, 400);
    let (job, rx) = glm53f_coordinator::Job::new(p.clone(), 5, None);
    let cancel = job.cancel.clone();
    h.jobs.send(job).unwrap();
    h.sched.step(false);
    cancel.store(true, std::sync::atomic::Ordering::Relaxed);
    h.run();
    drop(rx);
    let done = h.prefilled();
    assert!(done > 0 && done < 400, "cancelled mid-prefill after {done} tokens");
    h.clear_log();
    let r2 = h.submit(&p, 5, None);
    h.run();
    assert_eq!(tokens(&r2).unwrap(), greedy(&h, &p, 5));
    assert_eq!(h.prefilled(), 400 - done, "the retry resumed where the first attempt stopped");
}
