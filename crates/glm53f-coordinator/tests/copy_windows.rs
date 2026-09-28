//! Copy windows (`glm53f_coordinator::copy`) in the scheduler, against the copying toy model in
//! `common` (its pick repeats what followed the last earlier occurrence of its context's last 4
//! tokens, except where it edits): every request's tokens equal serial decoding
//! ([`common::copying_reference`]) with copy windows on or off, whatever is copied, accepted,
//! cut by the row budget or stopped inside; a request verifying a copy is not drafted for; sampled
//! requests never copy.

mod common;

use common::*;
use glm53f_coordinator::sampling::Sampling;
use glm53f_coordinator::{SchedStats, SchedulerConfig, Token};

/// A copy-heavy prompt: a span, other tokens, then the span's first 30 tokens again (past the
/// copy entry: the model goes on with the span, with an edit now and then).
fn copy_heavy(seed: u64) -> Vec<Token> {
    let span = prompt(seed, 90);
    let mut p = span.clone();
    p.extend(prompt(seed + 1000, 16));
    p.extend(&span[..30]);
    p
}

fn copies_on(c: &mut SchedulerConfig) {
    c.copy_windows = true;
}

fn copies_off(_: &mut SchedulerConfig) {}

/// A speculative step: the slots drafted for, and the verify windows (slot, rows).
type Step = (Vec<usize>, Vec<(usize, usize)>);

/// The speculative steps in a call log.
fn steps(calls: &[Call]) -> Vec<Step> {
    let mut out = Vec::new();
    let mut drafted = Vec::new();
    for c in calls {
        match c {
            Call::Draft(v) => drafted = v.clone(),
            Call::Verify(v) => out.push((std::mem::take(&mut drafted), v.clone())),
            _ => {}
        }
    }
    out
}

/// Run greedy requests `(prompt, max)` together on the copying model: every request's tokens
/// against the serial reference; the call log and the counters.
fn run(reqs: &[(Vec<Token>, usize, Option<Sampling>)], tweak: fn(&mut SchedulerConfig), eos: &[Token])
    -> (Vec<Call>, SchedStats) {
    let mut h = harness(Setup { block: 8, copying: true, tweak, eos: eos.to_vec(), ..Setup::default() });
    let rxs: Vec<_> = reqs.iter().map(|(p, max, s)| h.submit(p, *max, *s)).collect();
    h.run();
    for ((p, max, s), rx) in reqs.iter().zip(&rxs) {
        assert_eq!(tokens(rx).unwrap(), copying_reference(p, *max, *s, eos), "prompt {p:?}");
    }
    (h.calls(), h.sched.stats)
}

#[test]
fn copies_change_no_token_and_the_drafter_skips_copying_requests() {
    let s = Sampling::new(0.8, 1.0, 0, 0.0, Some(11)).unwrap();
    let reqs = [(copy_heavy(60), 48, None), (copy_heavy(61), 40, None), (prompt(62, 30), 40, None), (copy_heavy(63), 40, s)];
    let (calls, on) = run(&reqs, copies_on, &[]);
    let (_, off) = run(&reqs, copies_off, &[]);
    assert_eq!((off.copy_windows, off.copies_verified), (0, 0));
    // Copies were verified, some kept whole, some cut by the model's edits.
    assert!(on.copy_windows > 0 && on.copies_accepted > 0, "{on:?}");
    assert!(on.copies_accepted < on.copies_verified, "{on:?}");
    // Every window is either copied or drafted for, never both: in each step, the slots the drafter
    // saw and the copying slots (verified but not drafted for) split the verified slots.
    let mut copied = 0;
    for (drafted, windows) in steps(&calls) {
        assert!(drafted.iter().all(|d| windows.iter().any(|w| w.0 == *d)), "{drafted:?} {windows:?}");
        copied += windows.iter().filter(|w| !drafted.contains(&w.0)).count() as u64;
    }
    assert_eq!(copied, on.copy_windows);
    // The drafts of this toy never follow a copy, so copied windows carry more tokens.
    let (per_copy, per_draft) = on.tokens_per_window();
    assert!(per_copy > per_draft && on.spec_steps < off.spec_steps, "{on:?} {off:?}");
    eprintln!(
        "copy windows: {} of {} windows copied, {} of {} copied tokens kept; {per_copy:.2} tokens a copied \
         window, {per_draft:.2} a drafted one; {} steps with copies, {} without",
        on.copy_windows, on.windows, on.copies_accepted, on.copies_verified, on.spec_steps, off.spec_steps
    );
}

#[test]
fn sampled_requests_never_copy() {
    let s = Sampling::new(0.8, 1.0, 0, 0.0, Some(12)).unwrap();
    let (_, st) = run(&[(copy_heavy(64), 40, s)], copies_on, &[]);
    assert!(st.spec_steps > 0);
    assert_eq!(st.copy_windows, 0);
}

/// A request that stops inside an accepted copy (its budget, or an end-of-sequence token it
/// copied) commits only the rows of the tokens it delivered: its next turn resumes at the turn
/// point in place, prefilling only the new message.
#[test]
fn a_stop_inside_a_copied_run_commits_only_what_was_delivered() {
    let p = copy_heavy(65);
    // A token of the copied part of the span ends the sequence.
    let eos = vec![p[35]];
    let mut h = harness(Setup { block: 8, copying: true, tweak: copies_on, eos: eos.clone(), ..Setup::default() });
    for max in [6usize, 9, 40] {
        let rx = h.submit(&p, max, None);
        h.run();
        let out = tokens(&rx).unwrap();
        assert_eq!(out, copying_reference(&p, max, None, &eos));
        let n = out.len();
        let mut next = p.clone();
        next.extend(&out[..n - 1]);
        next.extend(prompt(70 + max as u64, 6));
        h.clear_log();
        let rx2 = h.submit(&next, 4, None);
        h.run();
        assert_eq!(tokens(&rx2).unwrap(), copying_reference(&next, 4, None, &eos));
        assert_eq!(h.prefilled(), 6, "max {max}: only the new message prefilled");
    }
    assert!(h.sched.stats.copy_windows > 0);
}

/// Four copying requests of up to 8 rows each, 12 rows a step: the budget drops copied tokens from
/// the back of the windows, and the tokens stay serial decoding's.
#[test]
fn copies_are_held_to_the_row_budget() {
    let reqs: Vec<_> = (0..4).map(|i| (copy_heavy(80 + i), 32, None)).collect();
    let (calls, st) = run(&reqs, |c| {
        c.copy_windows = true;
        c.spec_max_rows = 12;
    }, &[]);
    let passes: Vec<usize> = steps(&calls).iter().map(|(_, w)| w.iter().map(|x| x.1).sum()).collect();
    assert!(passes.iter().all(|&r| r <= 12), "{passes:?}");
    assert!(passes.contains(&12) && st.copy_windows > 0, "{passes:?} {st:?}");
}
