//! `glm53f-layers`: GLM-5.3-Flash's coordinator layers other than attention.
//!
//! - [`mhc`]: the manifold-constrained hyper-connection boundaries (4 residual streams of
//!   4,096; RMS-normalized projection to 24 weights; sigmoid `pre` and `post`; Sinkhorn
//!   `comb`), the collapse, the expand-and-mix, and the final unweighted mean.
//! - [`router`]: FP32 sigmoid scores, bias-corrected top-8 of 288, normalized and scaled
//!   by 2.5, with a fixed tie order.
//! - [`mlp`]: SwiGLU clamped at 10, FP8 block-128 projections (W8A16, or W8A8 with the
//!   checkpoint's dynamic per-128 activation quantization), the dense MLPs of layers 0-2
//!   and the shared expert.
//! - [`fp8`]: E4M3 coding, block-scaled weights, activation quantization, and the
//!   quantization of a BF16 weight in the checkpoint's block-128 scheme (with its BF16
//!   dequantization, for the W8A16 prefill path).
//! - [`norm`]: the weighted RMSNorm.
//! - [`layer`]: the decoder layer's stream flow with the sublayers as callbacks, the
//!   embedding broadcast and the final hidden state.
//! - `kernels/`: the CUDA kernels behind a C ABI (`kernels/glm53f_layers.h`); with the
//!   `cuda` feature they are compiled by `build.rs` and exposed as `ffi` (raw),
//!   `cuda` (device memory) and `ops` (checked launches). For decode, the boundary, the
//!   router and the split-K GEMM also come as single launches whose last CTA finishes the
//!   work (the second revision), bit-identical to the kernel pairs.
//! - [`testkit`]: fixture loading and test data.
//!
//! # Numerics contract
//!
//! The reference is `transformers` `models/glm5_next` (`modeling_glm5_next.py`). The CPU
//! functions here keep every rounding point of that reference (BF16 activations between
//! modules; f32 inside the mHC weights and the router; BF16 products where the reference
//! multiplies BF16 tensors) and, where the reference leaves an order of operations to its
//! backend (the sums inside a matrix product, `topk` among ties, `exp`), they fix the order
//! the kernels use. So:
//!
//! - **kernel versus CPU** is bitwise for every kernel except the tensor-core prefill GEMM
//!   (the in-block summation order of `mma` is the hardware's; that kernel is checked
//!   against exact f64 block products within a bound);
//! - **CPU versus the oracle** agrees within a stated tolerance, because the oracle's own
//!   summation orders and `exp` differ from these in the last bits. On the oracle's BF16 run
//!   the decoder layer's output streams match bit for bit, except on tokens where one f32 mHC
//!   weight sits within a few dozen f32 ulps of a BF16 rounding boundary and the oracle's
//!   value rounds the other way (`tests/goldens.rs`).
//!
//! Some kernels replace a division per value by cheaper arithmetic that gives the same bits
//! on every input they can receive (the E4M3 quantization's quotients, the router's
//! sigmoid); `glm53f_selfcheck_division_free` proves it exhaustively on the running GPU.
//!
//! The deliberate differences from the reference's arithmetic are: the mHC RMS scale is
//! applied after the projection (`r * sum(x * fn)`); sums run in the kernels' fixed orders;
//! `exp` is [`math::exp`] (within 2 ulp, reproducible on every device); and the router
//! chooses ties by lower expert index. None of these changes a rounding point.
//!
//! A row's result never depends on the other rows in its batch: every kernel computes each
//! row with the same operations whatever the row count, so a verify pass over 8 drafted rows
//! gives the bits of 8 serial one-row steps (the decode GEMM's K split depends only on the
//! weight's shape, see [`mlp::decode_ksplit`]).

pub mod bf16;
pub mod fp8;
pub mod layer;
pub mod math;
pub mod mhc;
pub mod mlp;
pub mod norm;
pub mod router;
pub mod testkit;

#[cfg(feature = "cuda")]
pub mod cuda;
#[cfg(feature = "cuda")]
pub mod ffi;
#[cfg(feature = "cuda")]
pub mod ops;

/// Hidden size of GLM-5.3-Flash.
pub const HIDDEN: usize = 4096;
/// Decoder layers.
pub const LAYERS: usize = 45;
