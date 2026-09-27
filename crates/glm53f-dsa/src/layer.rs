//! The whole DSA layer path (f32 reference): projections, indexer with k-pool
//! compression and the tail, top-k selection, and sparse MLA in either form.
//!
//! [`DsaState`] is one request's cache for one layer. [`DsaState::forward`]
//! appends a window of rows (a prefill chunk, a decode step, or a verify window)
//! and returns every intermediate. Processing a sequence in one window or in
//! many gives bit-identical results: a pool's key depends only on its own
//! tokens, and a row only sees pools that end at or before it.

use crate::config::DsaConfig;
use crate::fp8::{self, ScaleMode};
use crate::indexer::{self, IndexerCache, IndexerToken, IndexerWeights, KeyFormat};
use crate::mla::{self, AttnRow, MlaWeights};
use crate::num::{bf16_round, Rounding};
use crate::rng::Rng;
use crate::select::Selection;

/// All weights of one DSA layer.
#[derive(Clone, Debug)]
pub struct DsaLayerWeights {
    pub mla: MlaWeights,
    pub idx: IndexerWeights,
}

impl DsaLayerWeights {
    pub fn random(cfg: &DsaConfig, rng: &mut Rng) -> Self {
        Self { mla: MlaWeights::random(cfg, rng), idx: IndexerWeights::random(cfg, rng) }
    }

    pub fn check(&self, cfg: &DsaConfig) -> Result<(), String> {
        self.mla.check(cfg)?;
        self.idx.check(cfg)
    }
}

/// Which attention form to evaluate.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MlaForm {
    /// Expand every attended latent through `kv_b_proj` (the reference form).
    Expanded,
    /// Fold `kv_b_proj` into the query and the output (the engine's form).
    Absorbed,
}

/// How cached latents are stored.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LatentFormat {
    F32,
    Bf16,
    /// FP8 E4M3 with one scale per 128 channels (the 528-byte record at 512 channels).
    Fp8(ScaleMode),
}

impl LatentFormat {
    /// The latent as attention will read it.
    pub fn store(self, v: &[f32]) -> Vec<f32> {
        match self {
            LatentFormat::F32 => v.to_vec(),
            LatentFormat::Bf16 => v.iter().map(|x| bf16_round(*x)).collect(),
            LatentFormat::Fp8(mode) => {
                let mut out = vec![0.0f32; v.len()];
                for (chunk, o) in v.chunks(crate::cache::LATENT_GROUP).zip(out.chunks_mut(crate::cache::LATENT_GROUP)) {
                    let mut codes = vec![0u8; chunk.len()];
                    let s = fp8::quantize_block(chunk, &mut codes, mode);
                    fp8::dequantize_block(&codes, s, o);
                }
                out
            }
        }
    }
}

/// Options of the reference forward.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LayerOptions {
    pub rounding: Rounding,
    pub form: MlaForm,
    pub latent: LatentFormat,
    pub index_key: KeyFormat,
}

impl LayerOptions {
    /// f32 everywhere, absorbed form, no cache quantization.
    pub fn f32_absorbed() -> Self {
        Self { rounding: Rounding::F32, form: MlaForm::Absorbed, latent: LatentFormat::F32, index_key: KeyFormat::F32 }
    }

    /// The engine's storage: FP8 latents and FP8 pooled keys (power-of-two scales).
    pub fn engine() -> Self {
        Self {
            rounding: Rounding::F32,
            form: MlaForm::Absorbed,
            latent: LatentFormat::Fp8(ScaleMode::Pow2),
            index_key: KeyFormat::Fp8(ScaleMode::Pow2),
        }
    }
}

/// A row's projected inputs (what the layer computes before attention).
#[derive(Clone, Debug)]
pub struct ProjectedRow {
    /// `q_a_layernorm(q_a_proj x)`.
    pub q_resid: Vec<f32>,
    /// `q_b_proj q_resid`, `[n_heads][qk_nope_head_dim]`.
    pub q: Vec<f32>,
    /// `kv_a_layernorm(kv_a_proj_with_mqa x)`, `[kv_lora_rank]`.
    pub latent: Vec<f32>,
    /// The indexer's per-token values.
    pub idx: IndexerToken,
}

/// Project one row from the layer's (normalized) input.
pub fn project_row(cfg: &DsaConfig, w: &DsaLayerWeights, hidden: &[f32], rounding: Rounding) -> ProjectedRow {
    assert_eq!(hidden.len(), cfg.hidden, "hidden width");
    let (q_resid, q) = mla::project_q(cfg, &w.mla, hidden, rounding);
    let latent = mla::project_latent(cfg, &w.mla, hidden, rounding);
    let idx = indexer::project_token(cfg, &w.idx, hidden, &q_resid, rounding);
    ProjectedRow { q_resid, q, latent, idx }
}

/// Every intermediate of one processed row.
#[derive(Clone, Debug)]
pub struct RowTrace {
    pub position: usize,
    pub proj: ProjectedRow,
    /// Index scores of the pools visible to this row.
    pub scores: Vec<f32>,
    pub selection: Selection,
    /// Absorbed query `[n_heads][kv_lora_rank]` (absorbed form only).
    pub q_abs: Option<Vec<f32>>,
    /// Attention in latent space `[n_heads][kv_lora_rank]` (absorbed form only).
    pub o_latent: Option<Vec<f32>>,
    /// Per-head attention output before `o_proj`, and the log-sum-exp.
    pub attn: AttnRow,
    /// Layer output (`o_proj`).
    pub out: Vec<f32>,
}

/// One request's DSA cache for one layer.
#[derive(Clone, Debug)]
pub struct DsaState {
    /// Latents as stored (after [`LatentFormat::store`]).
    pub latents: Vec<Vec<f32>>,
    /// Latents before storage rounding.
    pub latents_exact: Vec<Vec<f32>>,
    /// Expanded keys and values per stored latent (expanded form only).
    pub expanded: Vec<(Vec<f32>, Vec<f32>)>,
    pub index: IndexerCache,
    pub opts: LayerOptions,
}

impl DsaState {
    pub fn new(opts: LayerOptions) -> Self {
        Self {
            latents: Vec::new(),
            latents_exact: Vec::new(),
            expanded: Vec::new(),
            index: IndexerCache::new(opts.index_key, opts.rounding),
            opts,
        }
    }

    /// Tokens in the cache.
    pub fn len(&self) -> usize {
        self.latents.len()
    }

    pub fn is_empty(&self) -> bool {
        self.latents.is_empty()
    }

    /// Append rows from the layer input and run the layer on them.
    pub fn forward(&mut self, cfg: &DsaConfig, w: &DsaLayerWeights, hidden: &[Vec<f32>]) -> Vec<RowTrace> {
        let rows: Vec<ProjectedRow> = hidden.iter().map(|h| project_row(cfg, w, h, self.opts.rounding)).collect();
        self.forward_projected(cfg, w, rows)
    }

    /// Append already-projected rows and run the layer on them.
    ///
    /// All rows are written to the cache first (latents, pooled keys completed
    /// inside the window, the tail); then each row selects among the pools
    /// that end at or before it and attends over its selection.
    pub fn forward_projected(&mut self, cfg: &DsaConfig, w: &DsaLayerWeights, rows: Vec<ProjectedRow>) -> Vec<RowTrace> {
        let start = self.len();
        for r in &rows {
            let stored = self.opts.latent.store(&r.latent);
            if self.opts.form == MlaForm::Expanded {
                self.expanded.push(mla::expand_kv(cfg, &w.mla, &stored, self.opts.rounding));
            }
            self.latents.push(stored);
            self.latents_exact.push(r.latent.clone());
            self.index.push(cfg, &w.idx.ape, &r.idx.k, &r.idx.gate);
        }
        rows.into_iter()
            .enumerate()
            .map(|(i, proj)| {
                let position = start + i;
                let (selection, scores) = self.index.select(cfg, position, &proj.idx.q, &proj.idx.w);
                let tokens = selection.tokens(cfg.index_kpool);
                let (attn, q_abs, o_latent) = match self.opts.form {
                    MlaForm::Expanded => {
                        let keys: Vec<Vec<f32>> = tokens.iter().map(|t| self.expanded[*t as usize].0.clone()).collect();
                        let vals: Vec<Vec<f32>> = tokens.iter().map(|t| self.expanded[*t as usize].1.clone()).collect();
                        (mla::attend_expanded(cfg, &proj.q, &keys, &vals, self.opts.rounding), None, None)
                    }
                    MlaForm::Absorbed => {
                        let lat: Vec<&[f32]> = tokens.iter().map(|t| self.latents[*t as usize].as_slice()).collect();
                        let qa = mla::absorb_q(cfg, &w.mla, &proj.q);
                        let (ol, lse) = mla::attend_absorbed_latent(cfg, &qa, &lat);
                        let out = mla::unabsorb_v(cfg, &w.mla, &ol);
                        (AttnRow { out, lse }, Some(qa), Some(ol))
                    }
                };
                let out = mla::o_proj(cfg, &w.mla, &attn.out);
                RowTrace { position, proj, scores, selection, q_abs, o_latent, attn, out }
            })
            .collect()
    }
}

/// Run a whole prompt through a fresh cache in one window.
pub fn prefill(cfg: &DsaConfig, w: &DsaLayerWeights, hidden: &[Vec<f32>], opts: LayerOptions) -> (DsaState, Vec<RowTrace>) {
    let mut st = DsaState::new(opts);
    let tr = st.forward(cfg, w, hidden);
    (st, tr)
}
