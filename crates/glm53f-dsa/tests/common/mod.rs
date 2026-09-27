//! Writes fixture sets in the oracle's format (manifest.json + raw .bin files)
//! from this crate's reference, for self-tests of the fixture readers.
#![allow(dead_code)]

use std::path::{Path, PathBuf};

use glm53f_dsa::config::DsaConfig;
use glm53f_dsa::layer::{DsaState, RowTrace};
use glm53f_dsa::sha256::sha256_hex;

pub struct Writer {
    dir: PathBuf,
    entries: Vec<String>,
}

impl Writer {
    pub fn new(dir: &Path) -> Self {
        std::fs::create_dir_all(dir).unwrap();
        Self { dir: dir.to_path_buf(), entries: Vec::new() }
    }

    pub fn add_f32(&mut self, name: &str, shape: &[usize], v: &[f32]) {
        let bytes: Vec<u8> = v.iter().flat_map(|x| x.to_le_bytes()).collect();
        self.add(name, "f32", shape, bytes);
    }

    pub fn add_i32(&mut self, name: &str, shape: &[usize], v: &[i32]) {
        let bytes: Vec<u8> = v.iter().flat_map(|x| x.to_le_bytes()).collect();
        self.add(name, "i32", shape, bytes);
    }

    fn add(&mut self, name: &str, dtype: &str, shape: &[usize], bytes: Vec<u8>) {
        assert_eq!(bytes.len(), shape.iter().product::<usize>() * 4);
        let file = format!("{name}.bin");
        std::fs::write(self.dir.join(&file), &bytes).unwrap();
        let shape_s: Vec<String> = shape.iter().map(|s| s.to_string()).collect();
        self.entries.push(format!(
            "\"{name}\": {{\"file\": \"{file}\", \"dtype\": \"{dtype}\", \"shape\": [{}], \"sha256\": \"{}\", \"desc\": \"synthetic\"}}",
            shape_s.join(", "),
            sha256_hex(&bytes)
        ));
    }

    pub fn close(self) {
        let m = format!("{{\"tensors\": {{{}}}, \"source\": {{\"note\": \"synthetic self-test\"}}}}", self.entries.join(",\n"));
        std::fs::write(self.dir.join("manifest.json"), m).unwrap();
    }
}

pub fn write_phase(w: &mut Writer, cfg: &DsaConfig, prefix: &str, tr: &[RowTrace], st: &DsaState, with_pools: bool) {
    let n = tr.len();
    let flat = |f: &dyn Fn(&RowTrace) -> Vec<f32>| -> Vec<f32> { tr.iter().flat_map(f).collect() };
    let ws = cfg.index_weight_scale();
    w.add_f32(&format!("{prefix}mla.q_resid"), &[n, cfg.q_lora_rank], &flat(&|t| t.proj.q_resid.clone()));
    w.add_f32(&format!("{prefix}mla.q"), &[n, cfg.n_heads, cfg.qk_nope_head_dim], &flat(&|t| t.proj.q.clone()));
    w.add_f32(&format!("{prefix}mla.latent"), &[n, cfg.kv_lora_rank], &flat(&|t| t.proj.latent.clone()));
    w.add_f32(&format!("{prefix}idx.q"), &[n, cfg.index_n_heads, cfg.index_head_dim], &flat(&|t| t.proj.idx.q.clone()));
    w.add_f32(&format!("{prefix}idx.k"), &[n, cfg.index_head_dim], &flat(&|t| t.proj.idx.k.clone()));
    w.add_f32(&format!("{prefix}idx.gate_scores"), &[n, cfg.index_head_dim], &flat(&|t| t.proj.idx.gate.clone()));
    w.add_f32(&format!("{prefix}idx.weights"), &[n, cfg.index_n_heads], &flat(&|t| t.proj.idx.w.iter().map(|v| v / ws).collect()));
    w.add_f32(&format!("{prefix}mla.out"), &[n, cfg.n_heads, cfg.v_head_dim], &flat(&|t| t.attn.out.clone()));
    w.add_f32(&format!("{prefix}attn_out"), &[n, cfg.hidden], &flat(&|t| t.out.clone()));
    let width = cfg.selection_width();
    let pools = st.index.pool_keys.len();
    let select_k = cfg.topk_pools().min(pools);
    let topk: Vec<i32> = tr.iter().flat_map(|t| t.selection.reference_row(cfg.index_kpool, select_k, true, width)).collect();
    w.add_i32(&format!("{prefix}idx.topk"), &[n, width], &topk);
    if with_pools {
        let keys: Vec<f32> = st.index.pool_keys.iter().flatten().copied().collect();
        w.add_f32(&format!("{prefix}idx.pool_keys"), &[pools, cfg.index_head_dim], &keys);
        let idx: Vec<i32> = (0..pools as i32 * 4).collect();
        w.add_i32(&format!("{prefix}idx.pool_indices"), &[pools, 4], &idx);
        let scores: Vec<f32> =
            tr.iter().flat_map(|t| (0..pools).map(|p| if p < t.scores.len() { t.scores[p] } else { f32::MIN }).collect::<Vec<_>>()).collect();
        w.add_f32(&format!("{prefix}idx.scores"), &[n, pools], &scores);
    }
}

