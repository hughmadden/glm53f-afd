//! Golden fixtures: the oracle's sets when they exist (skipped cleanly otherwise), and
//! synthetic sets, in the oracle's layout and in checkpoint naming, that exercise the loader
//! and every check.

mod common;

use std::path::Path;

use common::{core, scratch, write, write_oracle_pair, Owned};
use glm53f_kda::cpu::{self, Rounding};
use glm53f_kda::goldens::{self, Comparison, DType, Init, Role, Set};
use glm53f_kda::{channels, synth, DK, DV, LOWER_BOUND, RMS_EPS, TAPS, WINDOW};

/// Every check a set's KDA tensors allow, judged by `goldens::verdict`.
fn check_set(dir: &Path) -> Vec<Comparison> {
    let set = Set::load(dir).unwrap_or_else(|e| panic!("{}: {e}", dir.display()));
    set.verify_all()
        .unwrap_or_else(|e| panic!("{}: {e}", dir.display()));
    let mut all = Vec::new();
    for lf in goldens::kda_layers(&set) {
        let init = Init::from_prefill(&set, lf.layer).unwrap();
        let mut results = Vec::new();
        if lf.has(goldens::CORE_ROLES) {
            results.extend(goldens::check_core(&set, &lf, &init).unwrap());
        }
        if let Some(c) = goldens::check_conv_cache(&set, &lf, &init).unwrap() {
            results.push(c);
        }
        if goldens::has_layer_inputs(&lf) {
            results.extend(goldens::check_layer(&set, &lf, &init, RMS_EPS, LOWER_BOUND).unwrap());
        }
        if results.is_empty() {
            let roles: Vec<Role> = lf.tensors.iter().map(|(r, _)| *r).collect();
            eprintln!(
                "{}: layer {}: not enough tensors for a check ({roles:?})",
                set.name(),
                lf.layer
            );
        }
        for c in &results {
            eprintln!("{}: {c}", set.name());
        }
        assert!(
            goldens::verdict(&results),
            "{}: layer {} outside tolerance",
            set.name(),
            lf.layer
        );
        all.extend(results);
    }
    all
}

/// The oracle's sets under `oracle/goldens/` (or `GLM53F_GOLDENS`).
#[test]
fn oracle_goldens() {
    let root = goldens::default_root();
    let sets = goldens::discover(&root);
    if sets.is_empty() {
        eprintln!("skipped: no golden sets under {}", root.display());
        return;
    }
    let n: usize = sets.iter().map(|d| check_set(d).len()).sum();
    eprintln!("golden comparisons: {n}");
}

#[test]
fn oracle_layout_sets_pass_every_check() {
    let root = scratch("oracle-layout");
    let heads = 4;
    write_oracle_pair(&root, heads, 70, 8);
    let sets = goldens::discover(&root);
    assert_eq!(sets.len(), 2);
    // The decode set's first state comes from the prefill set next to it.
    let decode = Set::load(&root.join("layer04-decode")).unwrap();
    assert_eq!(decode.layer_phase(), Some((4, goldens::Phase::Decode)));
    assert_eq!(decode.state_heads(), Some(vec![0, 1, heads - 1]));
    let init = Init::from_prefill(&decode, 4).unwrap();
    assert!(init.state.is_some() && init.conv.is_some());
    let prefill = check_set(&root.join("layer04-prefill"));
    let decode = check_set(&root.join("layer04-decode"));
    // Prefill: the chunked form against the recurrence (f32 rounding only). Decode: exact.
    assert_eq!(prefill.len(), 3, "core output, state, conv cache");
    assert!(prefill
        .iter()
        .all(|c| c.max_abs <= 1e-5 * c.max_ref.max(1e-3)));
    assert_eq!(
        decode.len(),
        4,
        "core output, state, per-row states, conv cache"
    );
    assert!(decode.iter().all(|c| c.max_abs == 0.0), "{decode:?}");
    // Without its prefill set, a decode set starts from zero and fails.
    std::fs::remove_dir_all(root.join("layer04-prefill")).unwrap();
    let set = Set::load(&root.join("layer04-decode")).unwrap();
    let lf = &goldens::kda_layers(&set)[0];
    let r = goldens::check_core(&set, lf, &Init::default()).unwrap();
    assert!(!r[0].passes());
}

/// Full-layer tensors for layer `layer` under the checkpoint's module names, with the
/// reference's layouts, and outputs from the fused-conv reference.
fn layer_tensors(layer: usize, heads: usize, t: usize, seed: u64) -> Vec<Owned> {
    let p = synth::layer(heads, seed);
    let rows = synth::rows(heads, t, seed);
    let conv = synth::conv_window(heads, seed);
    let s0 = synth::state(heads, seed, 0.5);
    let r = cpu::chain(&p, &conv, &s0, &rows, Rounding::Fused);
    let c = channels(heads);
    // The reference's conv cache: [C][4], oldest first; its first column is not used.
    let mut cache = vec![0.0f32; c * TAPS];
    for ch in 0..c {
        cache[ch * TAPS] = 7.0;
        for j in 0..WINDOW {
            cache[ch * TAPS + 1 + j] = conv[j * c + ch];
        }
    }
    let hd = heads * DK;
    let pre = format!("model.language_model.layers.{layer}.self_attn.");
    let bf = DType::BF16;
    vec![
        (
            format!("{pre}q_conv1d.weight"),
            bf,
            vec![hd, 1, TAPS],
            p.conv_w[..hd * TAPS].to_vec(),
        ),
        (
            format!("{pre}k_conv1d.weight"),
            bf,
            vec![hd, 1, TAPS],
            p.conv_w[hd * TAPS..2 * hd * TAPS].to_vec(),
        ),
        (
            format!("{pre}v_conv1d.weight"),
            bf,
            vec![hd, 1, TAPS],
            p.conv_w[2 * hd * TAPS..].to_vec(),
        ),
        (
            format!("{pre}forget_gate.A_log"),
            DType::F32,
            vec![heads],
            p.a_log.clone(),
        ),
        (
            format!("{pre}forget_gate.dt_bias"),
            DType::F32,
            vec![hd],
            p.dt_bias.clone(),
        ),
        (
            format!("{pre}o_norm.weight"),
            bf,
            vec![DV],
            p.norm_w.clone(),
        ),
        (
            format!("{pre}mixed_qkv"),
            bf,
            vec![1, t, c],
            rows.qkv.clone(),
        ),
        (format!("{pre}conv_state_in"), bf, vec![1, c, TAPS], cache),
        (format!("{pre}f_proj"), bf, vec![1, t, hd], rows.a.clone()),
        (
            format!("{pre}b_proj"),
            bf,
            vec![1, t, heads],
            rows.b.clone(),
        ),
        (
            format!("{pre}gate"),
            bf,
            vec![1, t, heads * DV],
            rows.gate.clone(),
        ),
        (
            format!("{pre}initial_state"),
            DType::F32,
            vec![1, heads, DK, DV],
            cpu::transpose_state(&s0),
        ),
        (
            format!("{pre}o_norm_out"),
            bf,
            vec![1, t, heads * DV],
            r.out,
        ),
        (
            format!("{pre}final_state"),
            DType::F32,
            vec![1, heads, DK, DV],
            cpu::transpose_state(&r.state),
        ),
    ]
}

#[test]
fn checkpoint_named_layer_set_passes_the_layer_check() {
    let heads = 2;
    let dir = scratch("checkpoint-named");
    let mut tensors = layer_tensors(4, heads, 3, 23);
    tensors.push(("logits".into(), DType::F32, vec![4], vec![0.0; 4]));
    write(&dir, &tensors, "{}");
    let set = Set::load(&dir).unwrap();
    let layers = goldens::kda_layers(&set);
    assert_eq!(layers.len(), 1);
    let lf = &layers[0];
    assert!(goldens::has_layer_inputs(lf) && !lf.has(goldens::CORE_ROLES));
    let results = goldens::check_layer(&set, lf, &Init::default(), RMS_EPS, LOWER_BOUND).unwrap();
    for c in &results {
        eprintln!("{c}");
        assert!(c.passes(), "{c}");
        if !c.what.contains("unfused") {
            assert_eq!(c.max_abs, 0.0, "{c}");
        }
    }
    // The unfused conv rounding is a different computation.
    assert!(results
        .iter()
        .any(|c| c.what.contains("unfused") && c.differing > 0));
}

#[test]
fn corrupted_fixture_is_rejected() {
    let dir = scratch("corrupted-goldens");
    let c = core(1, 1, 31);
    let tensors = vec![
        (
            "layers.0.kda.v".to_string(),
            DType::BF16,
            vec![1, 1, DV],
            c.v,
        ),
        (
            "layers.0.kda.beta".to_string(),
            DType::F32,
            vec![1, 1],
            c.beta,
        ),
    ];
    write(&dir, &tensors, "{}");
    let set = Set::load(&dir).unwrap();
    let e = set.entry("layers.0.kda.v").unwrap().clone();
    let path = dir.join(&e.file);
    let mut bytes = std::fs::read(&path).unwrap();
    bytes[3] ^= 1;
    std::fs::write(&path, &bytes).unwrap();
    let err = set.verify_all().unwrap_err().to_string();
    assert!(err.contains("sha256"), "{err}");
    bytes.pop();
    std::fs::write(&path, &bytes).unwrap();
    let err = set.read("layers.0.kda.v").unwrap_err().to_string();
    assert!(err.contains("bytes"), "{err}");
}

#[test]
fn flat_manifest_is_accepted() {
    let dir = scratch("flat-manifest");
    std::fs::create_dir_all(&dir).unwrap();
    let data: Vec<u8> = [1.5f32, -2.0]
        .iter()
        .flat_map(|x| x.to_le_bytes())
        .collect();
    std::fs::write(dir.join("a.bin"), &data).unwrap();
    let manifest = format!(
        r#"{{"layers.1.kda.A_log": {{"file": "a.bin", "dtype": "float32", "shape": [2], "sha256": "{}"}}}}"#,
        glm53f_kda::sha256::hex(&data)
    );
    std::fs::write(dir.join("manifest.json"), manifest).unwrap();
    let set = Set::load(&dir).unwrap();
    assert_eq!(set.read("layers.1.kda.A_log").unwrap().data, [1.5, -2.0]);
    assert_eq!(
        goldens::kda_layers(&set)[0].name(Role::ALog),
        Some("layers.1.kda.A_log")
    );
    assert!(set.notes.is_none() && set.layer_phase().is_none());
}
