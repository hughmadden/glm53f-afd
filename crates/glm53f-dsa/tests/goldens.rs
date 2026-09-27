//! Oracle fixtures for layer 3 (a DSA layer): `oracle/goldens/layer03-prefill`
//! and `layer03-decode` (see oracle/README.md for names and layouts).
//!
//! * Without weights: the indexer's scoring and selection are checked from the
//!   fixtures alone (index queries, head weights and pooled keys in; scores and
//!   top-k rows out), including the `index_topk = 16` variant that drops pools.
//! * With `GLM53F_CHECKPOINT` (and `--release`): the whole layer from its input
//!   (`attn_norm`) in both MLA forms, for the prompt and the eight decode steps.
//!
//! Every test skips when its fixtures are missing. `synthetic_fixture_round_trip`
//! runs by default: it writes a fixture set in the oracle's format from this
//! crate's reference (tiny dimensions) and checks it with the same code, so the
//! reader and the comparisons are exercised before real fixtures exist.

mod common;

use common::{write_phase, Writer};

use glm53f_dsa::config::DsaConfig;
use glm53f_dsa::golden::GoldenSet;
use glm53f_dsa::indexer::index_score;
use glm53f_dsa::layer::{DsaLayerWeights, DsaState, LayerOptions, MlaForm, RowTrace};
use glm53f_dsa::rng::Rng;
use glm53f_dsa::select::{self, Selection};

fn rel_l2(a: &[f32], b: &[f32]) -> f64 {
    let num: f64 = a.iter().zip(b).map(|(x, y)| ((x - y) as f64).powi(2)).sum();
    let den: f64 = a.iter().map(|x| (*x as f64).powi(2)).sum();
    (num / den.max(1e-300)).sqrt()
}

fn find_set(part: &str) -> Option<GoldenSet> {
    GoldenSet::discover().into_iter().find(|s| s.dir.file_name().map(|n| n.to_string_lossy().contains(part)).unwrap_or(false))
}

fn get(set: &GoldenSet, name: &str) -> Option<(Vec<f32>, Vec<usize>)> {
    let t = set.get(name)?;
    Some((set.load_f32(t).unwrap_or_else(|e| panic!("{e}")), t.shape.clone()))
}

/// Indexer checks that need only the fixtures. `prefix` is `prefill.` or `prefill.k16.`.
/// Returns the number of rows checked.
fn check_indexer_rows(cfg: &DsaConfig, set: &GoldenSet, scores_prefix: &str, topk_prefix: &str) -> usize {
    let (Some((q, qs)), Some((w, _)), Some((keys, ks)), Some((scores, ss)), Some((topk, ts))) = (
        get(set, &format!("{scores_prefix}idx.q")),
        get(set, &format!("{scores_prefix}idx.weights")),
        get(set, &format!("{scores_prefix}idx.pool_keys")),
        get(set, &format!("{scores_prefix}idx.scores")),
        get(set, &format!("{topk_prefix}idx.topk")),
    ) else {
        return 0;
    };
    let (d, nh) = (cfg.index_head_dim, cfg.index_n_heads);
    let rows = qs[0];
    let pools = ks[0];
    assert_eq!(ss, vec![rows, pools], "scores shape");
    assert_eq!(ts, vec![rows, cfg.selection_width()], "topk shape");
    if let Some((pi, _)) = get(set, &format!("{scores_prefix}idx.pool_indices")) {
        for p in 0..pools {
            for i in 0..4 {
                assert_eq!(pi[p * 4 + i] as usize, 4 * p + i, "pool {p} token {i}");
            }
        }
    }
    let ws = cfg.index_weight_scale();
    let select_k = cfg.topk_pools().min(pools);
    for r in 0..rows {
        let qr = &q[r * nh * d..(r + 1) * nh * d];
        let wr: Vec<f32> = w[r * nh..(r + 1) * nh].iter().map(|v| v * ws).collect();
        let vis = (r + 1) / 4;
        let gold = &scores[r * pools..(r + 1) * pools];
        for p in 0..pools {
            if p < vis {
                let ours = index_score(cfg, qr, &wr, &keys[p * d..(p + 1) * d]);
                let tol = 1e-5 * (1.0 + gold[p].abs());
                assert!((ours - gold[p]).abs() <= tol, "row {r} pool {p}: score {ours} vs {}", gold[p]);
            } else {
                assert_eq!(gold[p], f32::MIN, "row {r} pool {p} is not visible");
            }
        }
        // Selection from the fixture's own scores; ties may be ordered differently.
        let sel = Selection::from_scores(r, cfg.index_kpool, cfg.topk_pools(), cfg.always_select_tail, &gold[..vis]);
        let ours = sel.reference_row(cfg.index_kpool, select_k, cfg.always_select_tail, cfg.selection_width());
        let theirs = &topk[r * cfg.selection_width()..(r + 1) * cfg.selection_width()];
        let tail_at = select_k * cfg.index_kpool;
        let theirs_i: Vec<i32> = theirs.iter().map(|v| *v as i32).collect();
        assert_eq!(&ours[tail_at..], &theirs_i[tail_at..], "row {r}: tail and padding");
        let blocks = |row: &[i32]| -> Vec<u32> { row[..tail_at].chunks(4).filter(|c| c[0] >= 0).map(|c| c[0] as u32 / 4).collect() };
        let (a, b) = (blocks(&ours), blocks(&theirs_i));
        let bad = select::boundary_mismatches(&a, &b, gold, 0.0);
        assert!(bad.is_empty(), "row {r}: pools {bad:?} differ beyond exact ties");
        for c in theirs_i[..tail_at].chunks(4) {
            assert!(c[0] < 0 || (c[1] == c[0] + 1 && c[2] == c[0] + 2 && c[3] == c[0] + 3 && c[0] % 4 == 0), "row {r}: pool block {c:?}");
        }
    }
    rows
}

/// Compare every recorded intermediate of one phase with the reference traces.
fn check_traces(cfg: &DsaConfig, set: &GoldenSet, prefix: &str, tr: &[RowTrace], tol: f64) {
    let rows = tr.len();
    let cmp = |name: &str, ours: Vec<f32>| {
        if let Some((g, _)) = get(set, &format!("{prefix}{name}")) {
            assert_eq!(g.len(), ours.len(), "{prefix}{name}: length");
            let e = rel_l2(&g, &ours);
            eprintln!("  {prefix}{name:18} rel L2 {e:.2e}");
            assert!(e <= tol, "{prefix}{name}: rel L2 {e} > {tol}");
        }
    };
    let ws = cfg.index_weight_scale();
    cmp("mla.q_resid", tr.iter().flat_map(|t| t.proj.q_resid.clone()).collect());
    cmp("mla.q", tr.iter().flat_map(|t| t.proj.q.clone()).collect());
    cmp("mla.latent", tr.iter().flat_map(|t| t.proj.latent.clone()).collect());
    cmp("idx.q", tr.iter().flat_map(|t| t.proj.idx.q.clone()).collect());
    cmp("idx.k", tr.iter().flat_map(|t| t.proj.idx.k.clone()).collect());
    cmp("idx.gate_scores", tr.iter().flat_map(|t| t.proj.idx.gate.clone()).collect());
    cmp("idx.weights", tr.iter().flat_map(|t| t.proj.idx.w.iter().map(|v| v / ws).collect::<Vec<_>>()).collect());
    cmp("mla.out", tr.iter().flat_map(|t| t.attn.out.clone()).collect());
    cmp("attn_out", tr.iter().flat_map(|t| t.out.clone()).collect());
    // Selections as token sets.
    if let Some((g, _)) = get(set, &format!("{prefix}idx.topk")) {
        let w = cfg.selection_width();
        for (i, t) in tr.iter().enumerate() {
            let mut theirs: Vec<u32> = g[i * w..(i + 1) * w].iter().filter(|v| **v >= 0.0).map(|v| *v as u32).collect();
            theirs.sort_unstable();
            assert_eq!(t.selection.tokens(cfg.index_kpool), theirs, "{prefix}idx.topk row {i}");
        }
        eprintln!("  {prefix}idx.topk          {rows} rows: identical token sets");
    }
}

/// Run a layer set (prefill, then decode) through the reference and compare.
fn check_layer(cfg: &DsaConfig, w: &DsaLayerWeights, pre: &GoldenSet, dec: Option<&GoldenSet>, tol: f64) {
    let (x, xs) = get(pre, "prefill.attn_norm").expect("prefill.attn_norm");
    let rows: Vec<Vec<f32>> = x.chunks(xs[1]).map(|c| c.to_vec()).collect();
    for form in [MlaForm::Expanded, MlaForm::Absorbed] {
        let mut opts = LayerOptions::f32_absorbed();
        opts.form = form;
        eprintln!("layer ({form:?} form):");
        let mut st = DsaState::new(opts);
        let tr = st.forward(cfg, w, &rows);
        check_traces(cfg, pre, "prefill.", &tr, tol);
        if let Some(pk) = get(pre, "prefill.idx.pool_keys") {
            let ours: Vec<f32> = st.index.pool_keys.iter().flatten().copied().collect();
            let e = rel_l2(&pk.0, &ours);
            eprintln!("  prefill.idx.pool_keys     rel L2 {e:.2e}");
            assert!(e <= tol);
        }
        if let Some(dec) = dec {
            if let Some((dx, dxs)) = get(dec, "decode.attn_norm") {
                let mut all = Vec::new();
                for step in dx.chunks(dxs[1]) {
                    all.extend(st.forward(cfg, w, &[step.to_vec()]));
                }
                check_traces(cfg, dec, "decode.", &all, tol);
            }
        }
        // index_topk = 16 variant (pool dropping), same weights and inputs.
        if pre.get("prefill.k16.attn_out").is_some() {
            let mut c16 = cfg.clone();
            c16.index_topk = 16;
            let mut st = DsaState::new(opts);
            let tr = st.forward(&c16, w, &rows);
            check_traces(&c16, pre, "prefill.k16.", &tr, tol);
        }
    }
}

#[test]
fn layer3_indexer_from_fixtures() {
    let Some(set) = find_set("layer03-prefill") else {
        eprintln!("skipping: no layer03-prefill fixtures under {}", GoldenSet::root().display());
        return;
    };
    let cfg = DsaConfig::glm53_flash();
    let n = check_indexer_rows(&cfg, &set, "prefill.", "prefill.");
    assert!(n > 0, "layer03-prefill lacks the indexer tensors");
    let mut c16 = cfg.clone();
    c16.index_topk = 16;
    let n16 = check_indexer_rows(&c16, &set, "prefill.", "prefill.k16.");
    eprintln!("indexer fixtures: {n} rows (index_topk 2048), {n16} rows (index_topk 16) agree");
}

#[test]
fn layer3_whole_path_from_fixtures() {
    let Some(pre) = find_set("layer03-prefill") else {
        eprintln!("skipping: no layer03-prefill fixtures");
        return;
    };
    if cfg!(debug_assertions) {
        eprintln!("skipping: needs --release");
        return;
    }
    let Some(ck) = glm53f_dsa::weights::checkpoint_from_env() else {
        eprintln!("skipping: GLM53F_CHECKPOINT not set");
        return;
    };
    let cfg = DsaConfig::glm53_flash();
    let w = glm53f_dsa::weights::load_dsa_layer(&ck, &cfg, 3).expect("layer 3 weights");
    let dec = find_set("layer03-decode");
    check_layer(&cfg, &w, &pre, dec.as_ref(), 1e-4);
}

#[test]
fn synthetic_fixture_round_trip() {
    let cfg = DsaConfig::tiny();
    let mut rng = Rng::new(91);
    let w = DsaLayerWeights::random(&cfg, &mut rng);
    let (t_prompt, t_decode) = (45usize, 5usize);
    let x: Vec<Vec<f32>> = (0..t_prompt + t_decode).map(|_| rng.normals(cfg.hidden, 1.0)).collect();
    let mut opts = LayerOptions::f32_absorbed();
    opts.form = MlaForm::Expanded;
    let root = std::env::temp_dir().join(format!("glm53f-dsa-fixtures-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&root);
    // Prefill set (with the index_topk 16 variant) and decode set.
    let mut st = DsaState::new(opts);
    let tr = st.forward(&cfg, &w, &x[..t_prompt]);
    let mut pw = Writer::new(&root.join("layer03-prefill"));
    pw.add_f32("prefill.attn_norm", &[t_prompt, cfg.hidden], &x[..t_prompt].concat());
    write_phase(&mut pw, &cfg, "prefill.", &tr, &st, true);
    let mut c16 = cfg.clone();
    c16.index_topk = 16;
    let mut st16 = DsaState::new(opts);
    let tr16 = st16.forward(&c16, &w, &x[..t_prompt]);
    write_phase(&mut pw, &c16, "prefill.k16.", &tr16, &st16, false);
    pw.close();
    let mut dec = Vec::new();
    for row in &x[t_prompt..] {
        dec.extend(st.forward(&cfg, &w, std::slice::from_ref(row)));
    }
    let mut dw = Writer::new(&root.join("layer03-decode"));
    dw.add_f32("decode.attn_norm", &[t_decode, cfg.hidden], &x[t_prompt..].concat());
    write_phase(&mut dw, &cfg, "decode.", &dec, &st, false);
    dw.close();

    let pre = GoldenSet::open(&root.join("layer03-prefill")).unwrap();
    let decs = GoldenSet::open(&root.join("layer03-decode")).unwrap();
    assert_eq!(check_indexer_rows(&cfg, &pre, "prefill.", "prefill."), t_prompt);
    assert_eq!(check_indexer_rows(&c16, &pre, "prefill.", "prefill.k16."), t_prompt);
    check_layer(&cfg, &w, &pre, Some(&decs), 1e-4);
    // A corrupted file is caught by its digest.
    let f = root.join("layer03-prefill").join("prefill.mla.q.bin");
    let mut b = std::fs::read(&f).unwrap();
    b[0] ^= 1;
    std::fs::write(&f, b).unwrap();
    let t = pre.get("prefill.mla.q").unwrap();
    assert!(pre.load_f32(t).unwrap_err().contains("sha256"));
    let _ = std::fs::remove_dir_all(&root);
}
