//! `glm53f-forward`: the GLM-5.3-Flash model path on the coordinator GPU.
//!
//! This crate assembles the kernel crates (`glm53f-kda`, `glm53f-dsa`, `glm53f-layers`) into the
//! model: device weights, the BF16 weight GEMMs, the per-request device KV, and the forward over
//! batches of rows for prefill, decode and verify windows with their commit. Routed experts go
//! through a trait ([`experts::ExpertBackend`]): the expert ranks in the engine, or this GPU in
//! tests.
//!
//! # Modules
//!
//! | Module | What |
//! |---|---|
//! | [`shape`] | GLM-5.3-Flash's dimensions; which decoder layers a forward runs (a prefix, or all 45) |
//! | [`kvplan`] | The KV's byte layout and accounting (the planner's figures), page reference counts and copy-on-write (host only) |
//! | [`reference`] | Host models of this crate's kernels, for tests |
//! | `weights` | The coordinator's tensors from the official FP8 checkpoint, on the device |
//! | `embed` | The embedding table in page-locked host RAM, gathered by row on the GPU |
//! | `gemm` | BF16 GEMMs (a row-independent GEMV for 1-8 rows, cuBLAS beyond) and the FP8 GEMM dispatch |
//! | `kv` | The pool (pages, page tables, KDA states, conv windows, DSA tails) and [`kv::GlmKv`], one request's view |
//! | `experts` | [`experts::ExpertBackend`], `LocalFp8Experts`, `ZeroExperts` |
//! | `remote` | `RemoteExperts`: the routed experts on the four expert ranks over the shell's wire client (feature `coordinator`) |
//! | `forward` | [`forward::GlmForward`]: the layer loop in one lane or up to four (a prefill's lanes overlap one lane's attention with the others' routed experts; decode and verify in two), the head, prefill / decode / verify / commit, taps, stage timing and the lane trace; `ForwardBuffers`, every buffer a pass uses, allocated up front |
//! | `opprof` | The op profile of prefill passes (`GLM53F_PROFILE_OPS=1`): each operation's GPU time in each lane's attention and shared expert, per layer, and the `OPS` table |
//! | `draft` | The DFlash2 drafter in the forward (`glm53f-dflash`): its taps (the mean of the four streams after layers 5, 14, 24, 33 and 42), the committed rows appended to each slot's ring, the drafts |
//! | `serve` | The serving shell's `KvSlot` and `ModelForward` (feature `coordinator`) |
//! | `device`, `cuda`, `cublas`, `ffi` | Device memory, streams and events; the runtime, cuBLAS and kernel bindings |
//!
//! Modules other than `shape`, `kvplan`, `reference` and `error` need the `cuda` feature.
//!
//! # The forward
//!
//! Per layer, following `transformers` `Glm5NextTextDecoderLayer` (and `glm53f-layers`' CPU
//! flow in `src/layer.rs`):
//!
//! ```text
//! streams [rows][4][4096] BF16 = the embedding row, 4 times
//! for each layer:
//!   mHC attention boundary: expand the previous FFN output into the streams, project, pre /
//!     post / comb (Sinkhorn 20), collapse, input_layernorm (and its E4M3 form)
//!   attention:
//!     KDA: [q|k|v|b] = x W1^T (BF16 [24,640][4096]); [f_a|g_a] = x W2^T; f_b, g_b (two GEMV
//!          groups); the fused chain (conv, norms, decay, beta, delta rule, gated RMSNorm);
//!          o_proj (BF16)
//!     DSA: q_a, kv_a (FP8); q_a_layernorm; q_b (FP8); the indexer's wq_b and [wk|gate|weights]
//!          (BF16); the FP8 latent record and pooled keys written to the pages; top-512 pools
//!          plus the tail; absorbed sparse MLA; o_proj (FP8)
//!   mHC FFN boundary: expand the attention output, project, collapse, post_attention_layernorm
//!   FFN: dense SwiGLU (layers 0-2, FP8) or MoE: router (sigmoid top-8 of 288, bias for the
//!        choice, weights x 2.5), the routes to the host, routed experts (backend), the shared
//!        expert (FP8), and bf16(routed + shared) in the next expansion
//! head: expand the last FFN output, mean of the 4 streams, final RMSNorm, LM head (BF16) ->
//!       f32 logits -> argmax below 154,856 (the padding rows are never picked)
//! ```
//!
//! **Row independence.** Up to 8 rows, every kernel computes a row with the same operations
//! whatever the rest of the pass holds: the GEMV (fixed lane order, K splits chosen from the
//! weight's shape), the FP8 decode GEMM, the single-launch mHC boundary, the router, KDA (the
//! chain, and the replay of a kept prefix), and the DSA kernels with a fixed split plan. So a
//! verify window committed at k rows, a decode batch and a short prefill give the bits of
//! serial one-row steps (`tests/verify_commit.rs`). Passes over more rows use tensor-core GEMMs
//! (cuBLAS; the FP8 GEMM with E4M3 activations), whose rounding depends on the row count.
//!
//! **Precision.** BF16 activations between modules, as the reference in its own dtypes. FP8
//! projections take BF16 activations up to 8 rows (W8A16, every product exact) and E4M3
//! activations per 128-group beyond (W8A8, the checkpoint's dynamic scheme; the only
//! tensor-core FP8 kernel). The MLA latent and the pooled index keys are stored in FP8
//! (decision D1). `tests/goldens_chain.rs` reports every stage against the oracle.
//!
//! **Numerics under test**, each off by default (`docs/SIZING.md` §10): the KDA projections
//! quantized to FP8 at load (`weights::WeightOptions::kda_fp8`, decision D2) with `amax / 448`,
//! power-of-two or MXFP8 scales (`WeightOptions::kda_scales`, [`Fp8Scales`]); the KDA
//! states stored in BF16 (`kvplan::KvLayout::with_kda_state_bf16`, decision D8; the state is
//! rounded after every row, so verify and commit keep the bits of serial steps); FP8 projections
//! over 8 rows with BF16 activations (`gemm::GemmPolicy::prefill_w8a16`, with
//! `kda_prefill_w8a8` keeping the FP8 KDA projections at E4M3). The model-path tests run with any
//! of them on through `GLM53F_TEST_NUMERICS`.
//!
//! # The KV
//!
//! A pool of pages (64 tokens of every DSA layer: 35,904 B per layer, 394,944 B over 11 layers),
//! one page-table row per slot, and per slot the FP32 KDA states (136 MiB for 34 layers), the
//! BF16 conv windows (4.8 MiB) and the DSA tails (17 KB). [`kv::GlmKv`] is one request: its
//! committed length and pending verify rows, its pages, `reserve`, marks of the positional state,
//! `rewind` to a mark, `fork` from a mark of another slot (full pages shared copy-on-write, the
//! partial page copied), and host images of pages and marks. A mark is held in pages of the pool
//! (376 of them), so admission, which counts free pages, counts marks too. Byte counts equal the memory
//! planner's (`glm53f-model`'s `KvGeometry` and `SlotState`) plus the tails, which the planner
//! does not count (`tests/kv_plan.rs`). With a drafter, each slot also holds the drafter's context
//! ring (40.16 MiB), kept at the committed length (`crate::draft`).
//!
//! # Tests and tools
//!
//! ```sh
//! cargo test -p glm53f-forward                                  # CPU: accounting and pages
//! cargo test --release -p glm53f-forward --features cuda        # kernels; with the weights below, the model path
//! cargo test --release -p glm53f-forward --features coordinator --test serve
//! cargo test --release -p glm53f-forward --features coordinator --test draft           # the drafter's taps and context
//! cargo test --release -p glm53f-forward --features coordinator --test draft_lossless  # speculation changes no token
//! cargo test --release -p glm53f-forward --features coordinator --test lanes --test admission   # prefill in lanes; memory
//! cargo run  --release -p glm53f-forward --features cuda --example gemm_bench
//! cargo run  --release -p glm53f-forward --features cuda --example decode_bench
//! ```
//!
//! The model-path tests read `GLM53F_CHECKPOINT_DIR` (the official checkpoint or its coordinator
//! subset), `GLM53F_EXPERTS_DIR` (the routed experts of layers 3 and 4, default the checkpoint)
//! and `GLM53F_GOLDENS` (default `oracle/goldens`), and skip cleanly without them; the drafter's
//! tests also `GLM53F_DFLASH_DIR` (the DFlash2 checkpoint) and run all 45 layers on the weights of
//! the first `GLM53F_DRAFT_TEST_LAYERS` (default 5), repeated. The build
//! uses `GLM53F_NVCC`, `GLM53F_CUDA_ARCH` (default `sm_89`) and `GLM53F_CUDA_LIB`, as the other
//! kernel crates do.

pub mod error;
pub mod kvplan;
pub mod reference;
pub mod shape;

#[cfg(feature = "cuda")]
pub mod cublas;
#[cfg(feature = "cuda")]
pub mod cuda;
#[cfg(feature = "cuda")]
pub mod device;
#[cfg(feature = "cuda")]
pub mod draft;
#[cfg(feature = "cuda")]
pub mod embed;
#[cfg(feature = "cuda")]
pub mod experts;
#[cfg(feature = "cuda")]
pub mod ffi;
#[cfg(feature = "cuda")]
pub mod forward;
#[cfg(feature = "cuda")]
pub mod gemm;
#[cfg(feature = "cuda")]
pub mod kv;
#[cfg(feature = "cuda")]
pub mod opprof;
#[cfg(feature = "coordinator")]
pub mod remote;
#[cfg(feature = "coordinator")]
pub mod serve;
#[cfg(feature = "cuda")]
pub mod weights;

pub use error::{Error, Result};
/// The scales of FP8 weights quantized at load, and their layouts (`weights::WeightOptions`).
pub use glm53f_layers::fp8::{Fp8Scales, ScaleLayout};

#[cfg(all(test, feature = "cuda"))]
mod tests {
    fn send<T: Send>() {}

    #[test]
    fn the_forward_and_its_slots_move_to_the_scheduler_thread() {
        send::<crate::forward::GlmForward>();
        send::<crate::kv::GlmKv>();
        send::<crate::kv::KvMark>();
    }

    #[cfg(feature = "coordinator")]
    #[test]
    fn the_served_forward_and_the_remote_experts_move_too() {
        send::<crate::serve::ServedForward>();
        send::<crate::remote::RemoteExperts>();
    }
}
