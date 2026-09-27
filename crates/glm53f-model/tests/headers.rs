//! The catalog and the TP4 expert split, checked against the published
//! checkpoints' real safetensors headers. Set `GLM53F_TEST_HEADERS_DIR` to a
//! directory holding `official.headers.json`, `tr3.headers.json` and
//! `nvfp4.headers.json`; without it these tests are skipped.

mod common;

use common::*;
use glm53f_model::catalog::*;
use glm53f_model::slicing::{rank_plan, slice_expert, Split, Tp};

fn catalog(which: &'static str) -> Option<Catalog> {
    let shards = bundle(which)?;
    Some(
        Catalog::from_shards(config(), shards, None, Coverage::Complete)
            .expect("headers match the config"),
    )
}

#[test]
fn official_headers_match_the_derived_catalog() {
    let Some(cat) = catalog("official") else {
        return;
    };
    assert_eq!(cat.format, CheckpointFormat::OfficialFp8);
    assert_eq!(bundle("official").unwrap().len(), OFFICIAL_SHARDS);
    assert_eq!(cat.tensors.len(), OFFICIAL_TENSORS);
    let got: Vec<(Group, u64)> = Group::ALL
        .iter()
        .map(|&g| (g, cat.group_bytes(g)))
        .collect();
    assert_eq!(got, expected_groups("official"));
    // KDA attention is BF16 apart from A_log and dt_bias.
    let kda = cat.group_dtypes(Group::KdaAttention);
    assert_eq!(
        kda.keys().map(|d| d.as_str()).collect::<Vec<_>>(),
        ["BF16", "F32"]
    );
    assert_eq!(kda[&glm53f_model::dtype::DType::F32], 34 * (64 + 8192) * 4);
    // Every FP8 weight carries its 128 x 128 block scales.
    let fp8 = cat
        .linears
        .iter()
        .filter(|l| l.spec.quant == Quant::Fp8Block)
        .count();
    assert_eq!(fp8, 12_384 * 3 + 43 * 3 + 3 * 3 + 12 * 4);
    assert!(cat.tensors.iter().all(|t| t.loc.is_some()));
}

#[test]
fn exl3_headers_match_the_derived_catalog() {
    let Some(cat) = catalog("tr3") else { return };
    assert_eq!(cat.format, CheckpointFormat::Exl3 { bits: 4 });
    let shards = bundle("tr3").unwrap();
    assert_eq!(shards.len(), EXL3_SHARDS);
    assert!(shards[0]
        .header
        .metadata
        .iter()
        .any(|(k, v)| k == "codec" && v == "exl3-mcg"));
    assert_eq!(cat.tensors.len(), EXL3_TENSORS);
    let got: Vec<(Group, u64)> = Group::ALL
        .iter()
        .map(|&g| (g, cat.group_bytes(g)))
        .collect();
    assert_eq!(got, expected_groups("exl3"));
}

#[test]
fn nvfp4_headers_match_the_derived_catalog() {
    let Some(cat) = catalog("nvfp4") else { return };
    assert_eq!(cat.format, CheckpointFormat::Nvfp4);
    assert_eq!(bundle("nvfp4").unwrap().len(), NVFP4_SHARDS);
    assert_eq!(cat.tensors.len(), NVFP4_TENSORS);
    let got: Vec<(Group, u64)> = Group::ALL
        .iter()
        .map(|&g| (g, cat.group_bytes(g)))
        .collect();
    assert_eq!(got, expected_groups("nvfp4"));
    // The input scales ship in their own file.
    let t = cat
        .get("model.language_model.layers.10.mlp.experts.0.down_proj.input_scale")
        .unwrap();
    assert_eq!(
        t.loc.as_ref().unwrap().file,
        "model-input-scales.safetensors"
    );
}

/// Per-rank TP4 bytes for decoder layers 3-44 and, separately, the MTP layer.
/// Exact values; docs/SIZING.md section 6 rounds them to 38.2 / 42.8 / 76.1 GB
/// (see the EXL3 note below).
const TP4: [(&str, u64, u64, u64); 3] = [
    // (bundle, per rank, MTP per rank, replicated per rank incl. MTP)
    // EXL3 replicates suh (gate, up), svh (down) and mcg: 24,588 B per expert.
    ("tr3", 38_385_301_248, 913_935_744, 43 * 288 * 24_588),
    // NVFP4 replicates weight_scale_2 and input_scale: 24 B per expert.
    ("nvfp4", 42_807_356_928, 1_019_222_784, 43 * 288 * 24),
    ("official", 76_120_031_232, 1_812_381_696, 0),
];

#[test]
fn tp4_rank_bytes_from_the_real_headers() {
    for (which, bytes, mtp, replicated) in TP4 {
        let Some(cat) = catalog(which) else { return };
        let mut total = 0;
        for rank in 0..4 {
            let p = rank_plan(&cat, Tp { rank, world: 4 }).unwrap();
            assert_eq!(
                (p.bytes, p.mtp_bytes, p.replicated_bytes),
                (bytes, mtp, replicated),
                "{which} rank {rank}"
            );
            assert_eq!(p.experts.len(), 43 * 288 * 3);
            total += p.bytes + p.mtp_bytes;
        }
        // Four quarters plus three extra copies of what every rank replicates.
        let all = cat.group_bytes(Group::RoutedExperts) + cat.group_bytes(Group::MtpRoutedExperts);
        assert_eq!(total, all + 3 * replicated, "{which}");
    }
}

/// The EXL3 figure in docs/SIZING.md (38.2 GB) is a quarter of the routed
/// bytes. The exact per-rank load is higher: `suh` of gate and up and `svh` of
/// down (4,096 F16 values each) must be whole on every rank.
#[test]
fn exl3_rank_share_exceeds_a_quarter_by_the_replicated_rotations() {
    let Some(cat) = catalog("tr3") else { return };
    let quarter = cat.group_bytes(Group::RoutedExperts) / 4;
    assert_rounds_to(
        quarter as f64 / 1e9,
        "38.2",
        "a quarter of the EXL3 routed experts",
    );
    let p = rank_plan(&cat, Tp { rank: 0, world: 4 }).unwrap();
    // 42 layers x 288 experts x 3 x 4,096 F16 x 3/4 (the part beyond a quarter).
    assert_eq!(
        p.bytes - quarter,
        42 * 288 * 3 * 4096 * 2 * 3 / 4 + 42 * 288 * 3 * 4 * 3 / 4
    );
    assert_rounds_to(p.bytes as f64 / 1e9, "38.39", "exact EXL3 bytes per rank");
}

#[test]
fn rank_slices_of_one_expert_tile_the_tensors() {
    for which in ["official", "tr3", "nvfp4"] {
        let Some(cat) = catalog(which) else { return };
        for proj in Proj::ALL {
            let lin = cat.expert(20, 131, proj).unwrap();
            let per_rank: Vec<_> = (0..4)
                .map(|rank| slice_expert(&cat, lin, Tp { rank, world: 4 }).unwrap())
                .collect();
            for (k, &(part, i)) in lin.parts.iter().enumerate() {
                let t = &cat.tensors[i];
                let mut hits = vec![0u8; t.bytes as usize];
                for s in &per_rank {
                    let sl = &s.tensors[k];
                    assert_eq!(sl.part, part);
                    assert!(sl.runs.fits(t.bytes));
                    for r in 0..sl.runs.count {
                        let a = sl.runs.offset + r * sl.runs.stride;
                        for b in a..a + sl.runs.len {
                            hits[b as usize] += 1;
                        }
                    }
                }
                let want = if per_rank[0].tensors[k].split == Split::Replicate {
                    4
                } else {
                    1
                };
                assert!(hits.iter().all(|&h| h == want), "{which} {proj:?} {part:?}");
            }
        }
    }
}
