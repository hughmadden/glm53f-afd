//! Shared by the test binaries: synthetic golden sets in the oracle's layout.
#![allow(dead_code)]

use std::path::{Path, PathBuf};

use glm53f_kda::goldens::{self, DType, NewTensor};
use glm53f_kda::{channels, chunked, cpu, synth, DK, DV, TAPS};

/// A fresh directory under the test binary's scratch space.
pub fn scratch(name: &str) -> PathBuf {
    let dir = Path::new(env!("CARGO_TARGET_TMPDIR")).join(name);
    let _ = std::fs::remove_dir_all(&dir);
    dir
}

/// Random f32 core inputs for `t` rows of `heads` heads: q, k, v (conv outputs), g (log decay),
/// beta.
pub struct Core {
    pub q: Vec<f32>,
    pub k: Vec<f32>,
    pub v: Vec<f32>,
    pub g: Vec<f32>,
    pub beta: Vec<f32>,
}

pub fn core(heads: usize, t: usize, seed: u64) -> Core {
    let mut rng = synth::Rng::new(seed);
    Core {
        q: rng.fill(t * heads * DK, -1.0, 1.0),
        k: rng.fill(t * heads * DK, -1.0, 1.0),
        v: rng.fill(t * heads * DV, -1.0, 1.0),
        g: (0..t * heads * DK)
            .map(|_| cpu::log_decay(rng.uniform(-4.0, 4.0), 0.0, 0.0, -5.0))
            .collect(),
        beta: (0..t * heads)
            .map(|_| cpu::sigmoid(rng.uniform(-4.0, 4.0)))
            .collect(),
    }
}

/// The recurrence (reference formulation) over `t` rows from `state` (`[H][DK][DV]`): the
/// read-out, and the states of `kept` heads after each row.
pub fn recur(
    c: &Core,
    heads: usize,
    t: usize,
    state: &mut [f32],
    kept: &[usize],
) -> (Vec<f32>, Vec<f32>) {
    let mut y = vec![0.0f32; t * heads * DV];
    let mut steps = Vec::new();
    for r in 0..t {
        for h in 0..heads {
            let i = r * heads + h;
            let o = cpu::literal::step(
                &mut state[h * DK * DV..(h + 1) * DK * DV],
                &c.q[i * DK..(i + 1) * DK],
                &c.k[i * DK..(i + 1) * DK],
                &c.v[i * DV..(i + 1) * DV],
                &c.g[i * DK..(i + 1) * DK],
                c.beta[i],
            );
            y[i * DV..(i + 1) * DV].copy_from_slice(&o);
        }
        for &h in kept {
            steps.extend_from_slice(&state[h * DK * DV..(h + 1) * DK * DV]);
        }
    }
    (y, steps)
}

/// The reference conv cache `[C][4]` after `rows` (`[t][C]`) follow `before`.
pub fn conv_cache(before: &[f32], rows: &[f32], c: usize) -> Vec<f32> {
    let t = rows.len() / c;
    let mut after = vec![0.0f32; c * TAPS];
    for ch in 0..c {
        for j in 0..TAPS {
            let s = t + j;
            after[ch * TAPS + j] = if s < TAPS {
                before[ch * TAPS + s]
            } else {
                rows[(s - TAPS) * c + ch]
            };
        }
    }
    after
}

pub type Owned = (String, DType, Vec<usize>, Vec<f32>);

fn f32t(name: &str, shape: Vec<usize>, values: Vec<f32>) -> Owned {
    (name.to_string(), DType::F32, shape, values)
}

pub fn write(dir: &Path, tensors: &[Owned], notes: &str) {
    let refs: Vec<NewTensor<'_>> = tensors
        .iter()
        .map(|(n, d, s, v)| (n.as_str(), *d, s.clone(), v.clone()))
        .collect();
    goldens::write_set(dir, &refs, notes).unwrap();
}

/// A prefill set and its decode set for layer 4, in the oracle's layout and f32 like its
/// primary contract: the prefill by the chunked form (as the reference's prefill runs), then
/// the decode steps by the recurrence. Decode records the per-row states of heads 0, 1 and
/// `heads - 1`.
pub fn write_oracle_pair(root: &Path, heads: usize, prompt: usize, steps: usize) {
    let c = channels(heads);
    let thd = |t: usize| vec![t, heads, DK];
    let pc = core(heads, prompt, 41);
    let mut rng = synth::Rng::new(42);
    let qkv_p = rng.fill(prompt * c, -2.0, 2.0);
    let mut state = vec![0.0f32; heads * DK * DV];
    let y_p = chunked::layer(heads, &pc.q, &pc.k, &pc.v, &pc.g, &pc.beta, &mut state, 64);
    let conv_p = conv_cache(&vec![0.0; c * TAPS], &qkv_p, c);
    let prefill = vec![
        f32t("prefill.kda.qkv_preconv", vec![prompt, c], qkv_p),
        f32t("prefill.kda.q", thd(prompt), pc.q),
        f32t("prefill.kda.k", thd(prompt), pc.k),
        f32t("prefill.kda.v", thd(prompt), pc.v),
        f32t("prefill.kda.g", thd(prompt), pc.g),
        f32t("prefill.kda.beta", vec![prompt, heads], pc.beta),
        f32t("prefill.kda.core_out", thd(prompt), y_p),
        f32t("prefill.kda.state", vec![heads, DK, DV], state.clone()),
        f32t("prefill.kda.conv_state", vec![c, TAPS], conv_p.clone()),
        ("prefill.kda.path".into(), DType::I32, vec![1], vec![0.0]),
        // Not a KDA tensor: skipped.
        f32t("prefill.attn_norm", vec![prompt, 8], vec![0.5; prompt * 8]),
    ];
    write(
        &root.join("layer04-prefill"),
        &prefill,
        r#"{"layer": 4, "attention": "kda", "phase": "prefill"}"#,
    );
    let kept = [0usize, 1, heads - 1];
    let dc = core(heads, steps, 43);
    let qkv_d = rng.fill(steps * c, -2.0, 2.0);
    let (y_d, per_step) = recur(&dc, heads, steps, &mut state, &kept);
    let decode = vec![
        f32t("decode.kda.qkv_preconv", vec![steps, c], qkv_d.clone()),
        f32t("decode.kda.q", thd(steps), dc.q),
        f32t("decode.kda.k", thd(steps), dc.k),
        f32t("decode.kda.v", thd(steps), dc.v),
        f32t("decode.kda.g", thd(steps), dc.g),
        f32t("decode.kda.beta", vec![steps, heads], dc.beta),
        f32t("decode.kda.core_out", thd(steps), y_d),
        f32t(
            "decode.kda.state_heads",
            vec![steps, kept.len(), DK, DV],
            per_step,
        ),
        f32t("decode.kda.state_final", vec![heads, DK, DV], state),
        f32t(
            "decode.kda.conv_state_final",
            vec![c, TAPS],
            conv_cache(&conv_p, &qkv_d, c),
        ),
        (
            "decode.kda.path".into(),
            DType::I32,
            vec![steps, 1],
            vec![1.0; steps],
        ),
        // A per-step tensor: skipped.
        f32t("decode.s0.idx.scores", vec![2], vec![1.0, 2.0]),
    ];
    let heads_list: Vec<String> = kept.iter().map(|h| h.to_string()).collect();
    let notes = format!(
        r#"{{"layer": 4, "attention": "kda", "phase": "decode", "state_heads": [{}]}}"#,
        heads_list.join(", ")
    );
    write(&root.join("layer04-decode"), &decode, &notes);
}
