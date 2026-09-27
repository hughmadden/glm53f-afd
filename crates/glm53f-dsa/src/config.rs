//! Model dimensions of the DSA layers.
//!
//! Values for GLM-5.3-Flash come from the checkpoint's `config.json`
//! (`text_config`): 45 layers, of which 11 are DSA layers (3, 7, ..., 43). Every
//! DSA layer runs its own indexer (`indexer_types` are all `"full"`).
//!
//! The CPU reference is generic over these dimensions so the tests can run small
//! shapes quickly; the CUDA kernels are specialised to the GLM-5.3-Flash values.

/// The DSA layer indices of GLM-5.3-Flash (`layer_types[i] == "deepseek_sparse_attention"`).
pub const DSA_LAYERS: [usize; 11] = [3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43];

/// Dimensions and constants of one DSA layer.
#[derive(Clone, Debug, PartialEq)]
pub struct DsaConfig {
    /// Residual width (`hidden_size`).
    pub hidden: usize,
    /// Attention heads (`num_attention_heads`; `num_key_value_heads` is equal).
    pub n_heads: usize,
    /// `q_lora_rank`: width of the query latent after `q_a_proj`.
    pub q_lora_rank: usize,
    /// `kv_lora_rank`: width of the cached MLA latent.
    pub kv_lora_rank: usize,
    /// `qk_nope_head_dim` (the whole query/key head: `qk_rope_head_dim` is 0).
    pub qk_nope_head_dim: usize,
    /// `v_head_dim`.
    pub v_head_dim: usize,
    /// `index_n_heads`.
    pub index_n_heads: usize,
    /// `index_head_dim`.
    pub index_head_dim: usize,
    /// `index_kpool`: tokens per pooled index key.
    pub index_kpool: usize,
    /// `index_topk`: selection budget in tokens (the pool budget is `index_topk / index_kpool`).
    pub index_topk: usize,
    /// `index_kpool_always_select_tail`.
    pub always_select_tail: bool,
    /// `rms_norm_eps` (q_a_layernorm, kv_a_layernorm).
    pub rms_norm_eps: f32,
    /// The indexer's `k_norm` LayerNorm epsilon (hard-coded 1e-6 in the reference).
    pub index_k_norm_eps: f32,
}

impl DsaConfig {
    /// GLM-5.3-Flash (`zai-org/GLM-5.3-Flash`, architecture `glm5_next`).
    pub fn glm53_flash() -> Self {
        Self {
            hidden: 4096,
            n_heads: 64,
            q_lora_rank: 1536,
            kv_lora_rank: 512,
            qk_nope_head_dim: 256,
            v_head_dim: 256,
            index_n_heads: 32,
            index_head_dim: 128,
            index_kpool: 4,
            index_topk: 2048,
            always_select_tail: true,
            rms_norm_eps: 1e-5,
            index_k_norm_eps: 1e-6,
        }
    }

    /// A small configuration with the same structure (k-pool 4, a tail, top-k
    /// budget in pools) for fast CPU tests.
    pub fn tiny() -> Self {
        Self {
            hidden: 48,
            n_heads: 4,
            q_lora_rank: 24,
            kv_lora_rank: 32,
            qk_nope_head_dim: 16,
            v_head_dim: 12,
            index_n_heads: 4,
            index_head_dim: 16,
            index_kpool: 4,
            index_topk: 32,
            always_select_tail: true,
            rms_norm_eps: 1e-5,
            index_k_norm_eps: 1e-6,
        }
    }

    /// Pools kept per query row: `index_topk / index_kpool` (512 for GLM-5.3-Flash).
    pub fn topk_pools(&self) -> usize {
        self.index_topk / self.index_kpool
    }

    /// Width of the reference's index output: `index_topk` plus the tail (`index_kpool - 1`).
    pub fn selection_width(&self) -> usize {
        self.index_topk + if self.always_select_tail { self.index_kpool - 1 } else { 0 }
    }

    /// Attention softmax scale: `qk_head_dim^-0.5` with `qk_head_dim = qk_nope_head_dim`.
    pub fn attn_scale(&self) -> f32 {
        (self.qk_nope_head_dim as f64).powf(-0.5) as f32
    }

    /// Indexer score scale: `index_head_dim^-0.5`.
    pub fn index_scale(&self) -> f32 {
        (self.index_head_dim as f64).powf(-0.5) as f32
    }

    /// Indexer head-weight scale: `index_n_heads^-0.5`.
    pub fn index_weight_scale(&self) -> f32 {
        (self.index_n_heads as f64).powf(-0.5) as f32
    }

    /// Complete pools visible to a query at absolute position `pos` (0-based),
    /// for a request whose first valid token is position 0.
    pub fn visible_pools(&self, pos: usize) -> usize {
        (pos + 1) / self.index_kpool
    }

    /// Checks the structural assumptions the reference relies on.
    pub fn validate(&self) -> Result<(), String> {
        if self.index_kpool == 0 || self.index_topk % self.index_kpool != 0 {
            return Err(format!(
                "index_topk ({}) must be a positive multiple of index_kpool ({})",
                self.index_topk, self.index_kpool
            ));
        }
        if self.qk_nope_head_dim == 0 || self.v_head_dim == 0 || self.kv_lora_rank == 0 {
            return Err("empty head".into());
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn glm53_constants() {
        let c = DsaConfig::glm53_flash();
        c.validate().unwrap();
        assert_eq!(c.topk_pools(), 512);
        assert_eq!(c.selection_width(), 2051);
        assert_eq!(c.attn_scale(), 0.0625);
        assert_eq!(c.index_scale(), (128f64).powf(-0.5) as f32);
        assert_eq!(c.visible_pools(0), 0);
        assert_eq!(c.visible_pools(3), 1);
        assert_eq!(c.visible_pools(2050), 512);
        assert_eq!(c.visible_pools(2051), 513);
        assert_eq!(DSA_LAYERS.len(), 11);
        assert!(DSA_LAYERS.iter().all(|l| l % 4 == 3));
    }
}
