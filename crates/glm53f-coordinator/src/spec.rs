//! How many drafts each speculative step verifies (mimo26f-afd perf reset S2). Model-agnostic:
//! it reads only the drafter's per-draft probabilities ([`crate::model::Draft::probs`]).

/// Verify-length policy for speculative steps.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum SpecPolicy {
    /// Verify every draft the budget allows.
    Fixed,
    /// Adaptive (`GLM53F_SPEC_POLICY=conf`): estimate P(draft j accepted) as the product of the
    /// drafter's probabilities of drafts 1..j, then add verify rows best-first while a row's
    /// expected tokens per ms beat the step's average, under the cost model `a_ms + b_ms * rows`
    /// (`GLM53F_SPEC_COST_A` / `_B`).
    Confidence { a_ms: f64, b_ms: f64 },
    /// Chain cut (the default): verify drafts while the product of the drafter's probabilities
    /// stays at or above `tau` (`GLM53F_SPEC_TAU`, default 0.7: on GLM-5.3-Flash with DFlash2, 0.5 and
    /// 0.7 were equal or faster than MiMo's 0.3 on code, prose and counting, and verify fewer drafts).
    Chain { tau: f64 },
}

impl Default for SpecPolicy {
    fn default() -> Self {
        SpecPolicy::Chain { tau: 0.7 }
    }
}

impl SpecPolicy {
    /// `GLM53F_SPEC_POLICY` (`fixed`, `conf`, else the chain cut) and its parameters.
    pub fn from_env() -> SpecPolicy {
        let num = |k: &str, d: f64| std::env::var(k).ok().and_then(|v| v.parse().ok()).unwrap_or(d);
        match std::env::var("GLM53F_SPEC_POLICY").as_deref() {
            Ok("fixed") => SpecPolicy::Fixed,
            Ok("conf") => SpecPolicy::Confidence { a_ms: num("GLM53F_SPEC_COST_A", 25.0), b_ms: num("GLM53F_SPEC_COST_B", 3.7) },
            _ => SpecPolicy::Chain { tau: num("GLM53F_SPEC_TAU", 0.7) },
        }
    }

    /// Drafts to verify per request: at most `caps[i]` and at most the drafts it has.
    pub fn lengths(&self, probs: &[Vec<f32>], caps: &[usize]) -> Vec<usize> {
        let caps: Vec<usize> = caps.iter().zip(probs).map(|(&c, p)| c.min(p.len())).collect();
        match *self {
            SpecPolicy::Fixed => caps,
            SpecPolicy::Confidence { a_ms, b_ms } => verify_lengths(probs, &caps, a_ms, b_ms),
            SpecPolicy::Chain { tau } => probs.iter().zip(&caps).map(|(p, &k)| chain_length(p, k, tau)).collect(),
        }
    }
}

/// The default of [`crate::scheduler::SchedulerConfig::spec_max_rows`] (`GLM53F_SPEC_MAX_ROWS`):
/// the most verify rows one step holds, every request's window together.
///
/// Why 256. The expert ranks' cost stops growing with the rows at about 128: a rank reads each
/// expert its rows name once per group of up to 32 of its rows (the measured 0.87 ms for 8 rows
/// over 58 experts and 16.9 ms for 4,096 rows both come to about 15 us per group, the second
/// counting uniform routes), and by 128 rows most of the 288 experts are named, each by fewer
/// than 32 rows, as at 256. Past that a row costs the coordinator's and the wire's per-row work,
/// about 10 us per MoE layer (the prefill trace: 34.6 ms of coordinator work and 8.6 ms of
/// transfer per 4,096 rows), about 0.45 ms a step, against at least 0.7 expected tokens for a
/// draft the chain cut keeps. So the budget should not bind below a few hundred rows: at 16 and
/// 32 slots it never does (8 rows a slot at most); at 48 it caps the pass at 256 of 384 rows, and
/// with it the verify buffers (about 4.3 MiB of saved inputs and 0.6 MiB of logits a row) at
/// 1.2 GiB instead of 1.8 GiB of the KV pool's memory.
pub const MAX_VERIFY_ROWS: usize = 256;

/// At most `max_rows` verify rows in all (each request's window is its last token plus its
/// drafts): when the policy's lengths `ks` (drafts per request) come to more, drafts are dropped
/// from the least likely up, likelihood being the product of the drafter's probabilities through
/// the draft (the chain cut's own estimate that it is kept), so the rows go to the drafts most
/// likely to be kept. A request always keeps its first row, and its rows stay a prefix. At or
/// under the budget (and with `max_rows` 0) the lengths are unchanged.
pub fn budget(probs: &[Vec<f32>], ks: &[usize], max_rows: usize) -> Vec<usize> {
    let rows: usize = ks.iter().map(|k| k + 1).sum();
    if max_rows == 0 || rows <= max_rows {
        return ks.to_vec();
    }
    // Every draft the lengths verify: (P(kept through it), depth, request).
    let mut cand: Vec<(f64, usize, usize)> = Vec::new();
    for (i, &k) in ks.iter().enumerate() {
        let mut q = 1.0f64;
        for j in 0..k {
            let pj = probs.get(i).and_then(|p| p.get(j)).copied().unwrap_or(0.0);
            q *= f64::from(pj).clamp(0.0, 1.0);
            cand.push((q, j + 1, i));
        }
    }
    // The least likely first; on a tie the deeper draft, then the later request. A request's
    // own drafts come deepest first (the product never grows with depth), so each drop is the
    // last row of its window.
    cand.sort_by(|x, y| x.0.total_cmp(&y.0).then(y.1.cmp(&x.1)).then(y.2.cmp(&x.2)));
    let mut out = ks.to_vec();
    let mut over = rows - max_rows.max(ks.len());
    for (_, depth, i) in cand {
        if over == 0 {
            break;
        }
        if depth == out[i] {
            out[i] -= 1;
            over -= 1;
        }
    }
    out
}

/// [`SpecPolicy::Chain`]'s choice for one request (at most `cap` drafts).
pub fn chain_length(p: &[f32], cap: usize, tau: f64) -> usize {
    let cap = cap.min(p.len());
    let mut q = 1.0f64;
    for (j, &pj) in p.iter().enumerate().take(cap) {
        q *= f64::from(pj).clamp(0.0, 1.0);
        if q < tau {
            return j;
        }
    }
    cap
}

/// [`SpecPolicy::Confidence`]'s choice: per request, how many of its drafts to verify (each at
/// most `caps[i]`). Every request always gets its bonus row.
pub fn verify_lengths(probs: &[Vec<f32>], caps: &[usize], a_ms: f64, b_ms: f64) -> Vec<usize> {
    let n = probs.len();
    // Candidate rows (request, depth, P(accepted through depth)), best first.
    let mut cand: Vec<(usize, usize, f64)> = Vec::new();
    for (i, p) in probs.iter().enumerate() {
        let mut q = 1.0f64;
        for (j, &pj) in p.iter().enumerate().take(caps[i].min(p.len())) {
            q *= f64::from(pj).clamp(0.0, 1.0);
            cand.push((i, j + 1, q));
        }
    }
    cand.sort_by(|x, y| y.2.total_cmp(&x.2));
    let mut ks = vec![0usize; n];
    let (mut tokens, mut rows) = (n as f64, n as f64);
    for (i, depth, q) in cand {
        // Rows of one request must stay a prefix: the chain is monotone in depth, so a deeper row
        // never outranks a shallower one of the same request.
        if depth != ks[i] + 1 {
            continue;
        }
        if q / b_ms < tokens / (a_ms + b_ms * rows) {
            break;
        }
        ks[i] = depth;
        tokens += q;
        rows += 1.0;
    }
    ks
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn verify_lengths_follow_confidence_and_caps() {
        let hi = vec![0.99f32, 0.98, 0.97, 0.96, 0.95, 0.94, 0.93];
        let lo = vec![0.30f32, 0.20, 0.10, 0.05, 0.05, 0.05, 0.05];
        // Confident drafts are all verified; unconfident ones barely at all.
        assert_eq!(verify_lengths(std::slice::from_ref(&hi), &[7], 25.0, 3.7), vec![7]);
        assert!(verify_lengths(std::slice::from_ref(&lo), &[7], 25.0, 3.7)[0] <= 1);
        // Caps (the token budget) bind.
        assert_eq!(verify_lengths(std::slice::from_ref(&hi), &[3], 25.0, 3.7), vec![3]);
        assert_eq!(verify_lengths(&[hi.clone(), hi.clone()], &[0, 7], 25.0, 3.7), vec![0, 7]);
        // Mixed batch: rows go to the confident request first.
        let ks = verify_lengths(&[lo.clone(), hi.clone()], &[7, 7], 25.0, 3.7);
        assert!(ks[1] == 7 && ks[0] <= 2, "{ks:?}");
        // Zero per-row cost verifies everything the caps allow.
        assert_eq!(verify_lengths(&[lo], &[7], 25.0, 0.0), vec![7]);
    }

    #[test]
    fn chain_cuts_where_the_product_drops_below_tau() {
        let p = [0.9f32, 0.8, 0.5, 0.9];
        // 0.9, 0.72, 0.36, 0.324: at tau 0.3 all four; at 0.4 the third falls below.
        assert_eq!(chain_length(&p, 7, 0.3), 4);
        assert_eq!(chain_length(&p, 7, 0.4), 2);
        assert_eq!(chain_length(&p, 1, 0.3), 1, "the cap binds");
        assert_eq!(chain_length(&[], 7, 0.3), 0);
        // The policy never verifies more drafts than there are.
        let ks = SpecPolicy::Fixed.lengths(&[vec![0.5; 3], vec![0.5; 7]], &[7, 5]);
        assert_eq!(ks, vec![3, 5]);
        // The default tau (0.7) keeps the first two (0.9, 0.72).
        assert_eq!(SpecPolicy::default().lengths(&[p.to_vec()], &[7]), vec![2]);
    }

    #[test]
    fn the_budget_keeps_the_most_likely_drafts() {
        let hi = vec![0.99f32, 0.98, 0.97, 0.96, 0.95, 0.94, 0.93];
        let mid = vec![0.9f32, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9];
        let lo = vec![0.8f32, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5];
        let probs = [hi.clone(), mid.clone(), lo.clone()];
        let ks = [7usize, 7, 7];
        // At or under the budget, and without one, nothing changes.
        assert_eq!(budget(&probs, &ks, 24), ks.to_vec());
        assert_eq!(budget(&probs, &ks, 100), ks.to_vec());
        assert_eq!(budget(&probs, &ks, 0), ks.to_vec());
        // One row over: the least likely draft goes (lo's 7th, 0.8 x 0.5^6).
        assert_eq!(budget(&probs, &ks, 23), vec![7, 7, 6]);
        // lo's drafts after its first are the least likely (0.4 and below), then mid's from the
        // back (0.9^7 = 0.48 < 0.9^6 < ...), while hi keeps all 7 (0.99 ... 0.83 through 7).
        assert_eq!(budget(&probs, &ks, 18), vec![7, 7, 1]);
        assert_eq!(budget(&probs, &ks, 16), vec![7, 5, 1]);
        // Down to the anchors: every request keeps its first row, however small the budget.
        assert_eq!(budget(&probs, &ks, 3), vec![0, 0, 0]);
        assert_eq!(budget(&probs, &ks, 1), vec![0, 0, 0]);
        // Each request's rows stay a prefix, and the lengths never grow.
        for max in 1..=24 {
            let b = budget(&probs, &ks, max);
            let rows: usize = b.iter().map(|k| k + 1).sum();
            assert!(rows <= max.max(3) && b.iter().zip(&ks).all(|(x, y)| x <= y), "{max}: {b:?}");
        }
        // Ties: the deeper draft goes first, then the later request.
        assert_eq!(budget(&[mid.clone(), mid.clone()], &[2, 2], 5), vec![2, 1]);
        assert_eq!(budget(&[vec![1.0; 3], vec![1.0; 3]], &[3, 3], 6), vec![2, 2]);
        // The chain cut, then the budget: a light step is the chain's own.
        let chain = SpecPolicy::default().lengths(&probs, &ks);
        assert_eq!(chain, vec![7, 3, 1]);
        assert_eq!(budget(&probs, &chain, MAX_VERIFY_ROWS), chain);
        // 14 rows into 8: mid's 3rd (0.729), hi's 7th (0.750), lo's 1st (0.800), hi's 6th
        // (0.807), mid's 2nd (0.810) and hi's 5th (0.858) go.
        assert_eq!(budget(&probs, &chain, 8), vec![4, 1, 0]);
    }
}
