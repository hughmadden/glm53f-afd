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
    /// stays at or above `tau` (`GLM53F_SPEC_TAU`, default 0.3).
    Chain { tau: f64 },
}

impl Default for SpecPolicy {
    fn default() -> Self {
        SpecPolicy::Chain { tau: 0.3 }
    }
}

impl SpecPolicy {
    /// `GLM53F_SPEC_POLICY` (`fixed`, `conf`, else the chain cut) and its parameters.
    pub fn from_env() -> SpecPolicy {
        let num = |k: &str, d: f64| std::env::var(k).ok().and_then(|v| v.parse().ok()).unwrap_or(d);
        match std::env::var("GLM53F_SPEC_POLICY").as_deref() {
            Ok("fixed") => SpecPolicy::Fixed,
            Ok("conf") => SpecPolicy::Confidence { a_ms: num("GLM53F_SPEC_COST_A", 25.0), b_ms: num("GLM53F_SPEC_COST_B", 3.7) },
            _ => SpecPolicy::Chain { tau: num("GLM53F_SPEC_TAU", 0.3) },
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
        assert_eq!(SpecPolicy::default().lengths(&[p.to_vec()], &[7]), vec![4]);
    }
}
