//! `glm53f-dflash`: the DFlash2 speculative drafter of GLM-5.3-Flash
//! (`incoai/GLM-5.3-Flash-DFlash2`).
//!
//! The drafter proposes seven tokens per step from a block of eight rows: the last verified token
//! (the *anchor*) followed by seven copies of the mask token. It has no embedding or LM head of its
//! own; it reads the target's. Its context is the target's own hidden states: after every
//! committed row, five of them (the mean of the four mHC streams after target layers 5, 14, 24,
//! 33 and 42, concatenated: the *taps*) are projected to 4,096 values and, per draft layer, to
//! keys and values that the block attends to over a 2,048-token window. The computation step by
//! step, with citations of the reference, is in `README.md`.
//!
//! What this crate holds:
//!
//! - [`reference`](mod@reference): the drafter in f32 on the CPU, following the reference's
//!   semantics: context append, the block forward, logits through a caller-supplied LM head, and
//!   the selector.
//! - [`selector`]: the top-k candidates and the path walk (greedy and sampled), shared by the
//!   reference and the tests of the GPU walk.
//! - [`weights`]: the drafter's tensors and the target rows it borrows, read from safetensors.
//! - [`seam`]: the calls the target forward and the scheduler will make ([`seam::Drafter`]).
//! - [`goldens`], [`synth`]: the oracle's fixtures and the synthetic context features they use.
//! - With the `cuda` feature: `gpu` (the batched forward) over the kernels in `kernels/`.
//!
//! Dimensions are runtime values ([`Dims`]) so the tests can run small random models; the
//! checkpoint's are [`Dims::GLM53F`]. The CUDA kernels fix the head size (128), the block (8),
//! the convolution taps (2) and the candidates (16).

pub mod bf16;
pub mod cpu;
pub mod goldens;
pub mod reference;
pub mod seam;
pub mod selector;
pub mod sha256;
pub mod synth;
pub mod weights;

#[cfg(feature = "cuda")]
pub mod blas;
#[cfg(feature = "cuda")]
pub mod cuda;
#[cfg(feature = "cuda")]
pub mod device;
#[cfg(feature = "cuda")]
pub mod ffi;
#[cfg(feature = "cuda")]
pub mod gpu;

/// Target layers whose outputs feed the drafter, in the order the taps are concatenated
/// (`dflash_config.target_layer_ids`).
pub const TARGET_LAYERS: [usize; 5] = [5, 14, 24, 33, 42];

/// The drafter's shape. Everything the reference and the forward need is here; the published
/// checkpoint is [`Dims::GLM53F`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Dims {
    /// Width of the residual stream (and of each target tap).
    pub hidden: usize,
    /// Decoder layers of the drafter.
    pub layers: usize,
    /// Query heads.
    pub heads: usize,
    /// Key/value heads (grouped-query attention: query head `h` reads KV head `h / (heads / kv_heads)`).
    pub kv_heads: usize,
    /// Size of every head.
    pub head_dim: usize,
    /// SwiGLU MLP width.
    pub inter: usize,
    /// Rows of the target's embedding and LM head, padding included.
    pub vocab: usize,
    /// Target hidden states concatenated per context row.
    pub taps: usize,
    /// Channels that share one dynamic convolution coefficient.
    pub group_size: usize,
    /// Taps of each dynamic convolution (the current row and the previous one).
    pub conv_taps: usize,
    /// Rank of the selector's codebooks.
    pub rank: usize,
    /// Candidates kept per draft position.
    pub top_k: usize,
    /// Sliding window: a query at position `p` sees keys at positions `q` with `|p - q| < window`.
    pub window: usize,
    /// Rows per draft block: the anchor and `block - 1` mask rows.
    pub block: usize,
    /// The mask token whose embedding fills rows 1.. of a block.
    pub mask_token: u32,
    /// RMSNorm epsilon (every norm of the drafter).
    pub eps: f32,
    /// RoPE base.
    pub rope_theta: f64,
}

impl Dims {
    /// `incoai/GLM-5.3-Flash-DFlash2` at `bf582e4e` for `zai-org/GLM-5.3-Flash`.
    pub const GLM53F: Dims = Dims {
        hidden: 4096,
        layers: 5,
        heads: 32,
        kv_heads: 8,
        head_dim: 128,
        inter: 12288,
        vocab: 154_880,
        taps: 5,
        group_size: 16,
        conv_taps: 2,
        rank: 256,
        top_k: 16,
        window: 2048,
        block: 8,
        mask_token: 154_856,
        eps: 1e-5,
        rope_theta: 10_000.0,
    };

    /// Width of one context row as the target hands it over: `taps * hidden` (20,480).
    pub fn tap_width(&self) -> usize {
        self.taps * self.hidden
    }
    /// Query width `heads * head_dim`.
    pub fn q_width(&self) -> usize {
        self.heads * self.head_dim
    }
    /// Key (or value) width `kv_heads * head_dim`.
    pub fn kv_width(&self) -> usize {
        self.kv_heads * self.head_dim
    }
    /// Convolution groups per row: `hidden / group_size`.
    pub fn groups(&self) -> usize {
        self.hidden / self.group_size
    }
    /// Outputs of a `kernel_projection`: two sides (prepare, finish) x taps x groups.
    pub fn dyn_width(&self) -> usize {
        2 * self.conv_taps * self.groups()
    }
    /// Drafts per block: `block - 1`.
    pub fn drafts(&self) -> usize {
        self.block - 1
    }
    /// Rows of a request's context ring: the window plus one block (2,056).
    ///
    /// A block query at position `P + j` (`P` the anchor's position) sees context positions
    /// `P + j - (window - 1) ..= P - 1`, at most `window - 1` rows, and the block's own 8 rows,
    /// written at their positions for the duration of the draft. Position `p` lives in row
    /// `p % ring`; `window - 1 + block <= ring`, so nothing a query reads is overwritten.
    pub fn ring(&self) -> usize {
        self.window + self.block
    }
    /// Device bytes of one request's ring: layers x (K, V) x ring x KV width x BF16.
    pub fn ring_bytes(&self) -> usize {
        self.layers * 2 * self.ring() * self.kv_width() * 2
    }
    /// Query heads per KV head.
    pub fn group(&self) -> usize {
        self.heads / self.kv_heads
    }

    /// Check the invariants the code relies on.
    pub fn validate(&self) -> Result<(), String> {
        let mut errs = Vec::new();
        if !self.hidden.is_multiple_of(self.group_size) {
            errs.push("hidden % group_size != 0".to_string());
        }
        if !self.heads.is_multiple_of(self.kv_heads) {
            errs.push("heads % kv_heads != 0".to_string());
        }
        if !self.head_dim.is_multiple_of(2) {
            errs.push("head_dim is odd".to_string());
        }
        if self.block < 2 || self.conv_taps < 1 || self.conv_taps > self.block {
            errs.push(format!(
                "block {} / conv_taps {}",
                self.block, self.conv_taps
            ));
        }
        if self.top_k == 0 || self.top_k > self.vocab {
            errs.push(format!("top_k {}", self.top_k));
        }
        if self.window == 0 {
            errs.push("window 0".to_string());
        }
        if (self.mask_token as usize) >= self.vocab {
            errs.push("mask_token beyond the vocabulary".to_string());
        }
        if errs.is_empty() {
            Ok(())
        } else {
            Err(errs.join("; "))
        }
    }

    /// The drafter config the checkpoint ships, checked against these dimensions.
    pub fn check_config(&self, c: &glm53f_model::config::DraftConfig) -> Result<(), String> {
        let mut errs = Vec::new();
        let mut eq = |what: &str, got: u64, want: usize| {
            if got != want as u64 {
                errs.push(format!("{what}: config {got}, expected {want}"));
            }
        };
        eq("hidden_size", c.hidden_size, self.hidden);
        eq("num_hidden_layers", c.num_hidden_layers, self.layers);
        eq("num_attention_heads", c.num_attention_heads, self.heads);
        eq("num_key_value_heads", c.num_key_value_heads, self.kv_heads);
        eq("head_dim", c.head_dim, self.head_dim);
        eq("intermediate_size", c.intermediate_size, self.inter);
        eq("vocab_size", c.vocab_size, self.vocab);
        eq("sliding_window", c.sliding_window, self.window);
        eq("block_size", c.block_size, self.block);
        eq("conv_group_size", c.conv_group_size, self.group_size);
        eq("conv_kernel_size", c.conv_kernel_size, self.conv_taps);
        eq("selector_rank", c.selector_rank, self.rank);
        eq("selector_top_k", c.selector_top_k, self.top_k);
        eq("mask_token_id", c.mask_token_id, self.mask_token as usize);
        eq(
            "target_layer_ids (count)",
            c.target_layer_ids.len() as u64,
            self.taps,
        );
        if c.layer_types.iter().any(|t| t != "sliding_attention") {
            errs.push(format!(
                "layer_types {:?}: expected sliding_attention only",
                c.layer_types
            ));
        }
        if errs.is_empty() {
            Ok(())
        } else {
            Err(errs.join("; "))
        }
    }
}

/// Token ids a draft may propose: one past the tokenizer's largest id. The LM head's rows at and
/// above it (154,856..154,879, including the mask token) are padding. The reference's top-k runs
/// over all 154,880 rows; this crate excludes the padding rows (see `README.md`).
pub const SAMPLE_VOCAB: usize = 154_856;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn glm_dims_and_ring_size() {
        let d = Dims::GLM53F;
        d.validate().unwrap();
        assert_eq!(d.tap_width(), 20_480);
        assert_eq!(d.dyn_width(), 1024);
        assert_eq!(d.ring(), 2056);
        // 5 layers x (K, V) x 2,056 rows x 8 x 128 x 2 bytes = 40.16 MiB per request.
        assert_eq!(d.ring_bytes(), 42_106_880);
        assert!((d.ring_bytes() as f64 / (1 << 20) as f64 - 40.16).abs() < 0.01);
        assert_eq!(SAMPLE_VOCAB, d.mask_token as usize);
    }
}
