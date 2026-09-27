//! Reading safetensors checkpoints from disk: synthetic files always, and real
//! checkpoints when their directories are given:
//! - `GLM53F_TEST_DRAFTER_DIR`: the DFlash2 drafter (`model.safetensors`);
//! - `GLM53F_TEST_CHECKPOINT_DIR`: official-checkpoint shards, all of them or
//!   a subset (for example the coordinator's non-expert tensors).

mod common;

use common::*;
use glm53f_model::catalog::{Catalog, CheckpointFormat, Coverage, Group};
use glm53f_model::dtype::DType;
use glm53f_model::safetensors::{read_file_header, serialize, Checkpoint, Runs};

fn bf16(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks(2)
        .map(|c| f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16))
        .collect()
}

#[test]
fn synthetic_checkpoint_reads_back() {
    let dir = TempDir::new("ckpt");
    let a: Vec<u8> = (0..24).collect();
    let b = vec![7u8; 8];
    let one = dir.0.join("x-1.safetensors");
    std::fs::write(&one, serialize(&[("a", DType::BF16, &[3, 4], &a)], &[])).unwrap();
    std::fs::write(
        dir.0.join("x-2.safetensors"),
        serialize(&[("b", DType::F32, &[2], &b)], &[("format", "pt")]),
    )
    .unwrap();
    std::fs::write(dir.0.join("notes.txt"), "not a shard").unwrap();

    let c = Checkpoint::open(&dir.0).unwrap();
    assert_eq!(c.names().collect::<Vec<_>>(), ["a", "b"]);
    assert_eq!(
        c.shards[1].header.metadata,
        vec![("format".to_string(), "pt".to_string())]
    );
    assert_eq!(c.read_tensor("a").unwrap(), a);
    assert_eq!(c.read_tensor("b").unwrap(), b);
    // Column 1 of the 3 x 4 BF16 tensor: one 2-byte run per row.
    let col = Runs {
        offset: 2,
        len: 2,
        stride: 8,
        count: 3,
    };
    assert_eq!(c.read_runs("a", &col).unwrap(), [2, 3, 10, 11, 18, 19]);
    let (file, begin, end) = c.file_range("b").unwrap();
    let raw = std::fs::read(dir.0.join(&file)).unwrap();
    assert_eq!(&raw[begin as usize..end as usize], b.as_slice());
    assert!(c.read_tensor("c").is_err());

    // The index must agree with the files.
    let index = dir.0.join("model.safetensors.index.json");
    std::fs::write(
        &index,
        r#"{"weight_map":{"a":"x-1.safetensors","b":"x-2.safetensors"}}"#,
    )
    .unwrap();
    Checkpoint::open(&dir.0).unwrap();
    std::fs::write(
        &index,
        r#"{"weight_map":{"a":"x-2.safetensors","b":"x-2.safetensors"}}"#,
    )
    .unwrap();
    assert!(
        Checkpoint::open(&dir.0).is_err(),
        "a tensor indexed in the wrong file"
    );
    std::fs::write(&index, r#"{"weight_map":{"a":"x-1.safetensors"}}"#).unwrap();
    assert!(
        Checkpoint::open(&dir.0).is_err(),
        "a tensor missing from the index"
    );
    std::fs::remove_file(&index).unwrap();

    // A tensor in two files.
    std::fs::write(
        dir.0.join("x-3.safetensors"),
        serialize(&[("a", DType::BF16, &[3, 4], &a)], &[]),
    )
    .unwrap();
    assert!(Checkpoint::open(&dir.0).is_err());
    std::fs::remove_file(dir.0.join("x-3.safetensors")).unwrap();

    // A truncated or padded file.
    let mut bytes = std::fs::read(&one).unwrap();
    bytes.pop();
    std::fs::write(&one, &bytes).unwrap();
    assert!(read_file_header(&one).is_err());
    bytes.extend_from_slice(&[0, 0]);
    std::fs::write(&one, &bytes).unwrap();
    assert!(read_file_header(&one).is_err());
    // A header length beyond the file.
    std::fs::write(&one, 1_000u64.to_le_bytes()).unwrap();
    assert!(read_file_header(&one).is_err());
}

#[test]
fn drafter_checkpoint_matches_its_config() {
    let Some(dir) = env_dir("GLM53F_TEST_DRAFTER_DIR") else {
        return;
    };
    let d = drafter();
    let c = Checkpoint::open(&dir).unwrap();
    let got: Vec<(String, DType, Vec<u64>)> = c
        .names()
        .map(|n| {
            let (_, e) = c.get(n).unwrap();
            (n.to_string(), e.dtype, e.shape.clone())
        })
        .collect();
    assert_eq!(got, d.tensors());
    let bytes: u64 = c.shards.iter().map(|s| s.header.data_len()).sum();
    assert_eq!(bytes, d.weight_bytes());
    assert_eq!(bytes, 2_342_160_896);
    let norm = bf16(&c.read_tensor("layers.0.self_attn.q_norm.weight").unwrap());
    assert_eq!(norm.len(), 128);
    assert!(norm.iter().all(|x| x.is_finite()) && norm.iter().any(|&x| x != 0.0));
}

#[test]
fn official_checkpoint_dir_matches_the_catalog() {
    let Some(dir) = env_dir("GLM53F_TEST_CHECKPOINT_DIR") else {
        return;
    };
    let c = Checkpoint::open(&dir).unwrap();
    let cat = Catalog::from_shards(
        config(),
        &c.shards,
        Some(CheckpointFormat::OfficialFp8),
        Coverage::Subset,
    )
    .unwrap();
    assert_eq!(cat.tensors.len(), c.names().count());
    for t in &cat.tensors {
        let (file, a, b) = c.file_range(&t.name).unwrap();
        let loc = t.loc.as_ref().unwrap();
        assert_eq!(
            (loc.file.as_str(), loc.file_range()),
            (file.as_str(), Some((a, b)))
        );
    }
    for (g, b) in cat.bytes_by_group() {
        eprintln!(
            "{:<24} {:>8} tensors {:>16} bytes",
            g.label(),
            cat.tensors.iter().filter(|t| t.group == g).count(),
            b
        );
    }
    if cat
        .get("model.language_model.layers.0.self_attn.A_log")
        .is_some()
    {
        let raw = c
            .read_tensor("model.language_model.layers.0.self_attn.A_log")
            .unwrap();
        let v: Vec<f32> = raw
            .chunks(4)
            .map(|x| f32::from_le_bytes(x.try_into().unwrap()))
            .collect();
        assert_eq!(v.len(), 64);
        assert!(v.iter().all(|x| x.is_finite()));
    }
    if cat.complete {
        assert_eq!(cat.group_bytes(Group::RoutedExperts), 304_480_124_928);
    }
}
