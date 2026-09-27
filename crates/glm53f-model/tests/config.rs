//! config.json: typed parsing and the GLM-5.3-Flash invariants.

mod common;

use common::*;
use glm53f_model::catalog::{spec, CheckpointFormat, Quant};
use glm53f_model::config::{AttnKind, DraftConfig, MlpKind, ModelConfig, Quantization};
use glm53f_model::json::{self, Json};
use glm53f_model::Error;

fn read(name: &str) -> String {
    std::fs::read_to_string(data(name)).unwrap()
}

#[test]
fn official_config_has_the_shape_the_engine_relies_on() {
    let c = config();
    let t = &c.text;
    assert_eq!(t.num_hidden_layers, 45);
    assert_eq!(t.kda_layer_ids().len(), 34);
    assert_eq!(
        t.dsa_layer_ids(),
        vec![3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43]
    );
    assert_eq!(t.hidden_size, 4096);
    assert_eq!(
        (
            t.kda.num_heads,
            t.kda.head_dim,
            t.kda.short_conv_kernel_size
        ),
        (64, 128, 4)
    );
    assert_eq!(t.kda_dim(), 8192);
    let m = &t.mla;
    assert_eq!(
        (
            m.q_lora_rank,
            m.kv_lora_rank,
            m.qk_nope_head_dim,
            m.qk_rope_head_dim,
            m.v_head_dim
        ),
        (1536, 512, 256, 0, 256)
    );
    let ix = &t.indexer;
    assert_eq!(
        (ix.n_heads, ix.head_dim, ix.topk, ix.kpool),
        (32, 128, 2048, 4)
    );
    assert!(ix.always_select_tail && ix.kpool_compress);
    assert_eq!((t.mhc.hc_mult, t.mhc.sinkhorn_iters), (4, 20));
    assert_eq!(t.hc_mix(), 24);
    let e = &t.moe;
    assert_eq!(
        (
            e.n_routed_experts,
            e.num_experts_per_tok,
            e.moe_intermediate_size,
            e.n_shared_experts
        ),
        (288, 8, 2048, 1)
    );
    assert_eq!(
        (e.first_k_dense_replace, e.dense_intermediate_size),
        (3, 12_288)
    );
    assert_eq!(
        (e.scoring_func.as_str(), e.topk_method.as_str()),
        ("sigmoid", "noaux_tc")
    );
    assert_eq!((e.routed_scaling_factor, e.swiglu_limit), (2.5, 10.0));
    assert_eq!(
        t.mlp_layer_types[..4],
        [MlpKind::Dense, MlpKind::Dense, MlpKind::Dense, MlpKind::Moe]
    );
    assert_eq!(t.moe_layers(), (3..45).collect::<Vec<_>>());
    assert_eq!(
        (t.vocab_size, t.max_position_embeddings),
        (154_880, 1_048_576)
    );
    assert_eq!(t.num_nextn_predict_layers, 1);
    assert_eq!(t.mtp_layers(), 45..46);
    assert_eq!(t.attn_kind(45), AttnKind::Dsa, "the MTP layer uses DSA");
    assert!(t.is_moe(45));
    assert_eq!(t.eos_token_ids, vec![154_820, 154_827, 154_829]);
    let v = &c.vision;
    assert_eq!(
        (
            v.depth,
            v.hidden_size,
            v.patch_size,
            v.spatial_merge_size,
            v.out_hidden_size
        ),
        (24, 1024, 14, 2, 4096)
    );
    assert_eq!(
        c.quantization,
        Quantization::Fp8Block {
            fmt: "e4m3".into(),
            block: [128, 128],
            activation_scheme: "dynamic".into()
        }
    );
    assert_eq!(
        CheckpointFormat::from_config(c),
        Some(CheckpointFormat::OfficialFp8)
    );
}

#[test]
fn quantized_checkpoints_share_the_model_config() {
    let official = config();
    for (name, fmt) in [
        (EXL3_CONFIG, CheckpointFormat::Exl3 { bits: 4 }),
        (NVFP4_CONFIG, CheckpointFormat::Nvfp4),
    ] {
        let c = ModelConfig::parse(&read(name)).unwrap();
        c.validate().unwrap();
        assert_eq!(c.text, official.text, "{name}");
        assert_eq!(c.vision, official.vision, "{name}");
        assert_eq!(CheckpointFormat::from_config(&c), Some(fmt), "{name}");
    }
}

/// The official checkpoint's dtype policy in the catalog agrees with the
/// config's own `modules_to_not_convert`: a text linear is FP8 exactly when the
/// config does not exempt it.
#[test]
fn official_fp8_policy_matches_modules_to_not_convert() {
    let v = json::parse(&read(OFFICIAL_CONFIG)).unwrap();
    let exempt: std::collections::BTreeSet<String> = v
        .get("quantization_config")
        .and_then(|q| q.get("modules_to_not_convert"))
        .and_then(Json::as_array)
        .unwrap()
        .iter()
        .map(|x| x.as_str().unwrap().to_string())
        .collect();
    let (_, linears) = spec(config(), CheckpointFormat::OfficialFp8);
    let (mut fp8, mut bf16) = (0, 0);
    for l in &linears {
        // The config names modules without the `language_model.` level.
        let module = l.module.replace("model.language_model.", "model.");
        let exempted = exempt.contains(&module);
        match l.quant {
            Quant::Fp8Block => {
                assert!(
                    !exempted,
                    "{module} is FP8 in the catalog but exempt in the config"
                );
                fp8 += 1;
            }
            Quant::Bf16 => {
                assert!(
                    exempted,
                    "{module} is BF16 in the catalog but not exempt in the config"
                );
                bf16 += 1;
            }
            q => panic!("{module}: {q:?} in the official format"),
        }
    }
    // 12,384 routed experts x 3, 43 shared and 3 dense MLPs x 3, 12 DSA layers x 4.
    assert_eq!(fp8, 12_384 * 3 + 43 * 3 + 3 * 3 + 12 * 4);
    assert!(bf16 > 0);
}

/// Replace `"key": <old>` inside the text_config (first occurrence after
/// `"text_config"`) and parse.
fn mutated(key: &str, old: &str, new: &str) -> ModelConfig {
    let text = read(OFFICIAL_CONFIG);
    let at = text.find("\"text_config\"").unwrap();
    let needle = format!("\"{key}\": {old}");
    let pos = at
        + text[at..]
            .find(&needle)
            .unwrap_or_else(|| panic!("{needle} not found"));
    let out = format!(
        "{}\"{key}\": {new}{}",
        &text[..pos],
        &text[pos + needle.len()..]
    );
    ModelConfig::parse(&out).unwrap()
}

fn violations(c: &ModelConfig) -> Vec<String> {
    match c.validate() {
        Err(Error::Invariant(v)) => v,
        other => panic!("expected invariant violations, got {other:?}"),
    }
}

#[test]
fn broken_invariants_are_all_reported() {
    let v = violations(&mutated("hidden_size", "4096", "5120"));
    assert!(
        v.iter()
            .any(|m| m.starts_with("text_config.hidden_size: expected 4096, got 5120")),
        "{v:?}"
    );
    // hidden 5120 also breaks the vision tower's output width.
    assert!(
        v.iter()
            .any(|m| m.starts_with("vision_config.out_hidden_size")),
        "{v:?}"
    );

    let v = violations(&mutated("swiglu_limit", "10.0", "7.0"));
    assert_eq!(
        v,
        vec!["text_config.swiglu_limit: expected 10, got 7".to_string()]
    );

    let v = violations(&mutated("index_kpool", "4", "8"));
    assert_eq!(v.len(), 1, "{v:?}");

    let v = violations(&mutated("n_routed_experts", "288", "256"));
    assert_eq!(v.len(), 1, "{v:?}");
    // A layer turned from DSA into KDA: counts and the explicit lists disagree.
    let text =
        read(OFFICIAL_CONFIG).replacen("\"deepseek_sparse_attention\"", "\"linear_attention\"", 1);
    let v = violations(&ModelConfig::parse(&text).unwrap());
    assert!(
        v.iter()
            .any(|m| m.contains("linear_attention layers: expected 34, got 35")),
        "{v:?}"
    );
    assert!(v.iter().any(|m| m.contains("full_attn_layers")), "{v:?}");
    // A DSA layer that would share its indexer.
    let mut j = json::parse(&read(OFFICIAL_CONFIG)).unwrap();
    if let Json::Object(top) = &mut j {
        let (_, tc) = top.iter_mut().find(|(k, _)| k == "text_config").unwrap();
        if let Json::Object(tc) = tc {
            let (_, it) = tc.iter_mut().find(|(k, _)| k == "indexer_types").unwrap();
            if let Json::Array(a) = it {
                a[7] = Json::Str("shared".into());
            }
        }
    }
    let v = violations(&ModelConfig::from_json(&j).unwrap());
    assert_eq!(
        v,
        vec!["DSA layers without their own indexer: expected [], got [7]".to_string()]
    );
}

#[test]
fn malformed_configs_fail_to_parse() {
    let text = read(OFFICIAL_CONFIG);
    // An integer written as a float.
    let e =
        ModelConfig::parse(&text.replacen("\"hidden_size\": 4096", "\"hidden_size\": 4096.0", 1))
            .unwrap_err();
    assert!(
        e.to_string()
            .contains("text_config.hidden_size: expected a non-negative integer"),
        "{e}"
    );
    // A missing key.
    let e = ModelConfig::parse(&text.replacen("\"index_topk\"", "\"index_topk_renamed\"", 1))
        .unwrap_err();
    assert!(
        e.to_string().contains("text_config.index_topk: missing"),
        "{e}"
    );
    // A duplicate key.
    let dup = text.replacen(
        "\"hidden_size\": 4096",
        "\"hidden_size\": 4096, \"hidden_size\": 4096",
        1,
    );
    assert!(ModelConfig::parse(&dup).is_err());
    // An unknown layer type.
    let e = ModelConfig::parse(&text.replacen("\"linear_attention\"", "\"sliding_attention\"", 1))
        .unwrap_err();
    assert!(e.to_string().contains("unknown type"), "{e}");
}

#[test]
fn drafter_config_fits_the_target() {
    let d = drafter();
    d.validate(config()).unwrap();
    assert_eq!(
        (
            d.num_hidden_layers,
            d.num_attention_heads,
            d.num_key_value_heads,
            d.head_dim
        ),
        (5, 32, 8, 128)
    );
    assert_eq!(
        (d.block_size, d.sliding_window, d.kv_window_tokens()),
        (8, 2048, 2056)
    );
    assert_eq!(d.target_layer_ids, vec![5, 14, 24, 33, 42]);
    // 5 layers x 8 KV heads x 128 x (K, V) x BF16 x 2,056 tokens.
    assert_eq!(d.kv_bytes_per_slot(), 42_106_880);
    // Tensor bytes of the published checkpoint (bf582e4e): 2,342,160,896.
    assert_eq!(d.weight_bytes(), 2_342_160_896);
    assert_eq!(d.tensors().len(), 6 + 5 * 15);

    let mut bad = d.clone();
    bad.target_layer_ids.push(45);
    bad.hidden_size = 5120;
    match bad.validate(config()) {
        // hidden size, and a tap beyond the target's 45 layers
        Err(Error::Invariant(v)) => assert_eq!(v.len(), 2, "{v:?}"),
        other => panic!("{other:?}"),
    }
    assert!(DraftConfig::parse("{}").is_err());
}
