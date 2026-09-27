//! The catalog derived from config.json, the name classifier, and matching
//! against (synthetic) checkpoint headers.

mod common;

use std::collections::BTreeMap;

use common::*;
use glm53f_model::catalog::*;
use glm53f_model::dtype::DType;
use glm53f_model::safetensors::{Header, Shard, TensorEntry};

const FORMATS: [(&str, CheckpointFormat); 3] = [
    ("official", CheckpointFormat::OfficialFp8),
    ("exl3", CheckpointFormat::Exl3 { bits: 4 }),
    ("nvfp4", CheckpointFormat::Nvfp4),
];

#[test]
fn derived_catalogs_have_the_published_counts_and_bytes() {
    for (which, fmt) in FORMATS {
        let cat = Catalog::from_config(config(), fmt);
        let n = match which {
            "official" => OFFICIAL_TENSORS,
            "exl3" => EXL3_TENSORS,
            _ => NVFP4_TENSORS,
        };
        assert_eq!(cat.tensors.len(), n, "{which}");
        let got: Vec<(Group, u64)> = Group::ALL
            .iter()
            .map(|&g| (g, cat.group_bytes(g)))
            .collect();
        assert_eq!(got, expected_groups(which), "{which}");
        assert!(cat.tensors.windows(2).all(|w| w[0].name < w[1].name));
    }
}

#[test]
fn classifier_agrees_with_the_derived_catalog_on_every_tensor() {
    let text = &config().text;
    for (which, fmt) in FORMATS {
        let cat = Catalog::from_config(config(), fmt);
        for t in &cat.tensors {
            assert_eq!(
                classify(&t.name, text, fmt),
                Some(t.role),
                "{which}: {}",
                t.name
            );
        }
    }
}

#[test]
fn classifier_rejects_names_the_model_does_not_have() {
    let text = &config().text;
    let fmt = CheckpointFormat::OfficialFp8;
    for name in [
        "model.language_model.layers.46.input_layernorm.weight",
        "model.language_model.layers.03.input_layernorm.weight",
        "model.language_model.layers.3.mlp.experts.288.up_proj.weight",
        "model.language_model.layers.3.mlp.experts.01.up_proj.weight",
        "model.language_model.layers.3.mlp.up_proj.weight",
        "model.language_model.layers.0.self_attn.indexer.wk.weight",
        "model.language_model.layers.0.self_attn.q_a_proj.weight",
        "model.language_model.layers.3.self_attn.q_proj.weight",
        "model.language_model.layers.3.self_attn.q_a_layernorm.weight_scale_inv",
        "model.language_model.layers.45.hc_attn_fn",
        "model.language_model.layers.3.hc_attn_bias",
        "model.language_model.layers.3.mlp.experts.0.up_proj.trellis",
        "model.language_model.embed_tokens.bias",
        "model.visual.blocks.0.attn.qkv",
        "lm_head.bias",
    ] {
        assert_eq!(classify(name, text, fmt), None, "{name}");
    }
    let r = classify(
        "model.language_model.layers.44.mlp.experts.287.down_proj.weight_scale_inv",
        text,
        fmt,
    )
    .unwrap();
    assert_eq!(r.layer, Some(44));
    assert_eq!(
        r.component,
        Component::RoutedExpert {
            expert: 287,
            proj: Proj::Down
        }
    );
    assert_eq!(r.part, Part::Fp8ScaleInv);
    let r = classify(
        "model.language_model.layers.44.mlp.experts.287.down_proj.weight",
        text,
        CheckpointFormat::Nvfp4,
    )
    .unwrap();
    assert_eq!(r.part, Part::Nvfp4Weight);
}

#[test]
fn quantized_linears_are_paired_with_their_scales() {
    let official = Catalog::from_config(config(), CheckpointFormat::OfficialFp8);
    let shape = |c: &Catalog, i: usize| c.tensors[i].shape.clone();
    let q_a = official
        .linear("model.language_model.layers.3.self_attn.q_a_proj")
        .unwrap();
    assert_eq!(q_a.spec.quant, Quant::Fp8Block);
    assert_eq!(
        shape(&official, q_a.tensor(Part::Weight).unwrap()),
        vec![1536, 4096]
    );
    assert_eq!(
        shape(&official, q_a.tensor(Part::Fp8ScaleInv).unwrap()),
        vec![12, 32]
    );
    let kv_b = official
        .linear("model.language_model.layers.3.self_attn.kv_b_proj")
        .unwrap();
    assert_eq!(kv_b.spec.quant, Quant::Bf16);
    assert_eq!(kv_b.parts.len(), 1);
    let kda_q = official
        .linear("model.language_model.layers.0.self_attn.q_proj")
        .unwrap();
    assert_eq!(
        (
            kda_q.spec.quant,
            kda_q.spec.out_features,
            kda_q.spec.in_features
        ),
        (Quant::Bf16, 8192, 4096)
    );
    let e = official.expert(44, 287, Proj::Down).unwrap();
    assert_eq!(
        shape(&official, e.tensor(Part::Fp8ScaleInv).unwrap()),
        vec![32, 16]
    );
    assert!(official.expert(45, 0, Proj::Gate).is_some(), "MTP experts");
    assert!(official.expert(2, 0, Proj::Gate).is_none(), "dense layer");

    let exl3 = Catalog::from_config(config(), CheckpointFormat::Exl3 { bits: 4 });
    let g = exl3.expert(3, 0, Proj::Gate).unwrap();
    assert_eq!(
        shape(&exl3, g.tensor(Part::Exl3Trellis).unwrap()),
        vec![256, 128, 64]
    );
    assert_eq!(shape(&exl3, g.tensor(Part::Exl3Suh).unwrap()), vec![4096]);
    assert_eq!(shape(&exl3, g.tensor(Part::Exl3Svh).unwrap()), vec![2048]);
    let d = exl3.expert(3, 0, Proj::Down).unwrap();
    assert_eq!(
        shape(&exl3, d.tensor(Part::Exl3Trellis).unwrap()),
        vec![128, 256, 64]
    );
    assert_eq!(
        exl3.linear("model.language_model.layers.3.self_attn.q_a_proj")
            .unwrap()
            .spec
            .quant,
        Quant::Bf16
    );

    let nv = Catalog::from_config(config(), CheckpointFormat::Nvfp4);
    let d = nv.expert(3, 0, Proj::Down).unwrap();
    assert_eq!(
        shape(&nv, d.tensor(Part::Nvfp4Weight).unwrap()),
        vec![4096, 1024]
    );
    assert_eq!(
        shape(&nv, d.tensor(Part::Nvfp4Scale).unwrap()),
        vec![4096, 128]
    );
    assert_eq!(
        shape(&nv, d.tensor(Part::Nvfp4Scale2).unwrap()),
        Vec::<u64>::new()
    );
}

/// Build shards from a derived catalog, with contiguous offsets, as a
/// checkpoint of that format would have (`per_shard` tensors per file).
fn synthetic_shards(cat: &Catalog, per_shard: usize) -> Vec<Shard> {
    cat.tensors
        .chunks(per_shard)
        .enumerate()
        .map(|(i, chunk)| {
            let mut tensors = BTreeMap::new();
            let mut at = 0;
            for t in chunk {
                tensors.insert(
                    t.name.clone(),
                    TensorEntry {
                        dtype: t.dtype,
                        shape: t.shape.clone(),
                        begin: at,
                        end: at + t.bytes,
                    },
                );
                at += t.bytes;
            }
            Shard {
                file: format!("model-{:05}.safetensors", i + 1),
                header: Header {
                    tensors,
                    metadata: vec![],
                },
                data_start: None,
            }
        })
        .collect()
}

#[test]
fn headers_that_match_the_config_are_catalogued_with_locations() {
    for (which, fmt) in FORMATS {
        let derived = Catalog::from_config(config(), fmt);
        let shards = synthetic_shards(&derived, 5000);
        assert_eq!(detect_format(&shards).unwrap(), fmt, "{which}");
        let cat = Catalog::from_shards(config(), &shards, None, Coverage::Complete).unwrap();
        assert!(cat.complete);
        assert_eq!(cat.tensors.len(), derived.tensors.len());
        assert_eq!(cat.linears.len(), derived.linears.len());
        assert_eq!(cat.bytes_by_group(), derived.bytes_by_group());
        let t = cat.get(LM_HEAD).unwrap();
        let loc = t.loc.as_ref().unwrap();
        assert_eq!(loc.end - loc.begin, t.bytes);
        assert_eq!(
            loc.file_range(),
            None,
            "a header bundle has no file offsets"
        );
    }
}

#[test]
fn mismatched_headers_are_refused() {
    let derived = Catalog::from_config(config(), CheckpointFormat::OfficialFp8);
    let base = synthetic_shards(&derived, 100_000);
    let edit = |f: &dyn Fn(&mut BTreeMap<String, TensorEntry>)| {
        let mut s = base.clone();
        f(&mut s[0].header.tensors);
        s
    };
    let err = |s: &[Shard], cov| {
        Catalog::from_shards(config(), s, Some(CheckpointFormat::OfficialFp8), cov)
            .unwrap_err()
            .to_string()
    };

    let missing = edit(&|t| {
        t.remove("model.language_model.layers.7.self_attn.indexer.wk.weight");
    });
    assert!(err(&missing, Coverage::Complete).contains("1 missing"));
    // As a subset the same headers are fine.
    let sub = Catalog::from_shards(config(), &missing, None, Coverage::Subset).unwrap();
    assert!(!sub.complete);

    let extra = edit(&|t| {
        t.insert(
            "model.language_model.layers.7.self_attn.rotary.inv_freq".into(),
            TensorEntry {
                dtype: DType::F32,
                shape: vec![1],
                begin: 0,
                end: 4,
            },
        );
    });
    assert!(err(&extra, Coverage::Subset).contains("1 unexpected"));

    let wrong = edit(&|t| {
        t.get_mut("model.language_model.layers.0.self_attn.A_log")
            .unwrap()
            .dtype = DType::BF16;
    });
    assert!(err(&wrong, Coverage::Complete).contains("wrong dtype or shape"));

    // An FP8 weight without its scales is refused even as a subset.
    let torn = edit(&|t| {
        t.remove("model.language_model.layers.3.self_attn.o_proj.weight_scale_inv");
    });
    let e = err(&torn, Coverage::Subset);
    assert!(
        e.contains("o_proj is missing its [\"weight_scale_inv\"]"),
        "{e}"
    );

    // A tensor in two shards.
    let mut twice = base.clone();
    let mut copy = twice[0].clone();
    copy.file = "model-99999.safetensors".into();
    copy.header.tensors.retain(|k, _| k == LM_HEAD);
    twice.push(copy);
    assert!(err(&twice, Coverage::Complete).contains("is in both"));
}

#[test]
fn format_detection_needs_one_expert_format() {
    let official = synthetic_shards(
        &Catalog::from_config(config(), CheckpointFormat::OfficialFp8),
        100_000,
    );
    let nvfp4 = synthetic_shards(
        &Catalog::from_config(config(), CheckpointFormat::Nvfp4),
        100_000,
    );
    let mut mixed = official.clone();
    mixed.extend(nvfp4);
    assert!(detect_format(&mixed).is_err());
    let mut no_experts = official;
    no_experts[0]
        .header
        .tensors
        .retain(|k, _| !k.contains(".mlp.experts."));
    assert!(detect_format(&no_experts).is_err());
}
