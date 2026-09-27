//! Shared test helpers and the small expectations derived from the published
//! checkpoints' safetensors headers (tensor counts and bytes per group).
//!
//! Tests that need the full headers read them from the directory named by
//! `GLM53F_TEST_HEADERS_DIR` (`official.headers.json`, `tr3.headers.json`,
//! `nvfp4.headers.json`: each a JSON object mapping shard file name to that
//! shard's header) and skip when it is not set. The numbers below were taken
//! from those headers:
//! - official: `zai-org/GLM-5.3-Flash` @ eb9eb208;
//! - EXL3: `brandonmusic/GLM-5.3-Flash-tr3-4bpw` (the same weights as
//!   `Mia-AiLab/GLM-5.3-Flash-EXL3-TR3-4bpw`);
//! - NVFP4: `LibertAIDAI/GLM-5.3-Flash-NVFP4`.
#![allow(dead_code)]

use std::path::PathBuf;
use std::sync::OnceLock;

use glm53f_model::catalog::Group;
use glm53f_model::config::{DraftConfig, ModelConfig};
use glm53f_model::safetensors::{parse_header_bundle, Shard};

pub fn data(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/data")
        .join(name)
}

pub const OFFICIAL_CONFIG: &str = "zai-org_GLM-5.3-Flash.config.json";
pub const EXL3_CONFIG: &str = "brandonmusic_GLM-5.3-Flash-tr3-4bpw.config.json";
pub const NVFP4_CONFIG: &str = "LibertAIDAI_GLM-5.3-Flash-NVFP4.config.json";
pub const DRAFTER_CONFIG: &str = "incoai_GLM-5.3-Flash-DFlash2.config.json";

pub fn config() -> &'static ModelConfig {
    static C: OnceLock<ModelConfig> = OnceLock::new();
    C.get_or_init(|| ModelConfig::load(&data(OFFICIAL_CONFIG)).expect("official config loads"))
}

pub fn drafter() -> DraftConfig {
    DraftConfig::load(&data(DRAFTER_CONFIG)).expect("drafter config loads")
}

/// A directory named by an environment variable, or `None` (with a note that
/// the test is skipped).
pub fn env_dir(var: &str) -> Option<PathBuf> {
    match std::env::var_os(var) {
        Some(v) if !v.is_empty() => Some(PathBuf::from(v)),
        _ => {
            eprintln!("skipped: set {var} to run this test");
            None
        }
    }
}

type Bundles = std::sync::Mutex<Vec<(&'static str, &'static [Shard])>>;

/// One header bundle from `GLM53F_TEST_HEADERS_DIR`, parsed once per test binary.
pub fn bundle(which: &'static str) -> Option<&'static [Shard]> {
    static BUNDLES: OnceLock<Bundles> = OnceLock::new();
    let dir = env_dir("GLM53F_TEST_HEADERS_DIR")?;
    let cache = BUNDLES.get_or_init(Default::default);
    let mut cache = cache.lock().unwrap();
    if let Some((_, s)) = cache.iter().find(|(w, _)| *w == which) {
        return Some(s);
    }
    let path = dir.join(format!("{which}.headers.json"));
    let text = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    let shards: &'static [Shard] = Box::leak(
        parse_header_bundle(&text)
            .expect("bundle parses")
            .into_boxed_slice(),
    );
    cache.push((which, shards));
    Some(shards)
}

/// Assert `value` prints as `doc` at `doc`'s number of decimals.
pub fn assert_rounds_to(value: f64, doc: &str, what: &str) {
    let decimals = doc.split_once('.').map_or(0, |(_, d)| d.len());
    let got = format!("{value:.decimals$}");
    assert_eq!(
        got, doc,
        "{what}: {value} does not round to the documented {doc}"
    );
}

/// Tensors and shards per checkpoint, from the published headers.
pub const OFFICIAL_TENSORS: usize = 76_108;
pub const OFFICIAL_SHARDS: usize = 62;
pub const EXL3_TENSORS: usize = 150_226;
pub const EXL3_SHARDS: usize = 120;
pub const NVFP4_TENSORS: usize = 150_226;
pub const NVFP4_SHARDS: usize = 121;

/// Bytes per group in the official FP8 checkpoint.
pub const OFFICIAL_GROUPS: &[(Group, u64)] = &[
    (Group::KdaAttention, 9_366_356_992),
    (Group::DsaAttention, 1_476_710_400),
    (Group::DsaIndexer, 164_381_184),
    (Group::SharedExperts, 1_057_222_656),
    (Group::DenseMlp, 453_095_424),
    (Group::Routers, 99_138_816),
    (Group::Mhc, 70_788_600),
    (Group::Norms, 745_472),
    (Group::LmHead, 1_268_776_960),
    (Group::Embedding, 1_268_776_960),
    (Group::RoutedExperts, 304_480_124_928),
    (Group::MtpLayer, 243_872_384),
    (Group::MtpRoutedExperts, 7_249_526_784),
    (Group::Vision, 1_127_254_016),
];

/// Bytes per group that differ in the EXL3 checkpoint (the rest as official):
/// every non-expert tensor is BF16.
pub const SHIPPED_BF16_GROUPS: &[(Group, u64)] = &[
    (Group::DsaAttention, 2_583_736_320),
    (Group::SharedExperts, 2_113_929_216),
    (Group::DenseMlp, 905_969_664),
    (Group::MtpLayer, 369_670_784),
];

pub const EXL3_EXPERTS: (u64, u64) = (152_648_955_648, 3_634_498_944);
pub const NVFP4_EXPERTS: (u64, u64) = (171_228_556_800, 4_076_870_400);

/// Expected bytes per group for a format: "official", "exl3" or "nvfp4".
pub fn expected_groups(which: &str) -> Vec<(Group, u64)> {
    let mut out: Vec<(Group, u64)> = OFFICIAL_GROUPS.to_vec();
    if which == "official" {
        return out;
    }
    let (routed, mtp) = if which == "exl3" {
        EXL3_EXPERTS
    } else {
        NVFP4_EXPERTS
    };
    for (g, b) in out.iter_mut() {
        if let Some(&(_, x)) = SHIPPED_BF16_GROUPS.iter().find(|(h, _)| h == g) {
            *b = x;
        }
        match g {
            Group::RoutedExperts => *b = routed,
            Group::MtpRoutedExperts => *b = mtp,
            _ => {}
        }
    }
    out
}

/// A unique scratch directory under the system temp dir, removed on drop.
pub struct TempDir(pub PathBuf);

impl TempDir {
    pub fn new(tag: &str) -> TempDir {
        let p = std::env::temp_dir().join(format!("glm53f-model-{tag}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&p);
        std::fs::create_dir_all(&p).unwrap();
        TempDir(p)
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}
