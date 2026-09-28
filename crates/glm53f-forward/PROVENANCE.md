# Provenance: glm53f-forward

Rows in the format of [docs/REUSE.md](../../docs/REUSE.md). All dates are 28 September 2026.
No code is copied from outside this repository. The forward is assembled from this repository's
kernel crates (called through their C ABIs, not copied) and written from the reference's
semantics; the units below record what each part was written from.

Sources:

- **R**: `huggingface/transformers` @ `7cd73d9df0c14b151c684b708a9f27d8d0349dfe`,
  `src/transformers/models/glm5_next/modeling_glm5_next.py`, Apache-2.0. Semantics only; no code
  copied. (The oracle runs the same file from the 5.17.0 wheel; see `oracle/README.md`.)
- **L**: `crates/glm53f-layers` (this repository), MIT.
- **K**: `crates/glm53f-kda` (this repository), MIT.
- **Q**: `crates/glm53f-dsa` (this repository), MIT.
- **C**: `crates/glm53f-coordinator` (this repository), MIT. Interface only.
- **N**: NVIDIA CUDA runtime and cuBLAS 12.8 public headers (`cuda_runtime_api.h`,
  `driver_types.h`, `library_types.h`, `cublas_api.h`). Function signatures and enumeration
  values only, re-declared in Rust; the libraries are linked from the toolkit.

| Unit | Source (repo @ commit : path) | sha256 (source file) | Here | Delta | Pinned by | Date |
|---|---|---|---|---|---|---|
| Decoder-layer flow, KDA layer (projections, gates, gated norm, o_proj), DSA layer (MLA query and latent paths, the indexer's projections and head-weight scale), MoE (routed plus shared, routed scale in the weights), the head (mean of the streams, final norm, LM head) | R : `Glm5NextTextDecoderLayer`, `Glm5NextTextLinearAttention`, `Glm5NextTextForgetGate`, `Glm5NextTextAttention`, `Glm5NextTextIndexer`, `Glm5NextTextMoE`, `Glm5NextTextHyperHead` | `4fe6ed7703e4f8f1dc7e3995af2b619058be8fec1f5157b7f512fc6f6150503f` | `src/forward.rs`, `src/weights.rs` | **Reimplemented** as a sequence of this repository's kernels. BF16 activations between modules as the reference in its own dtypes; the latent and pooled keys in FP8 (D1); FP8 projections W8A16 up to 8 rows and W8A8 beyond. | `tests/goldens_chain.rs` (layers 0-4 against every golden set, and the native set as the yardstick) | 2026-09-28 |
| Routed-expert combine (per row, ascending expert id, `bf16(acc + bf16(y * w))`) | R : `Glm5NextTextExperts.forward`; L : `src/mlp.rs` `routed_experts_eager` (its CPU model of the same loop) | R as above; L @ `7f6ffa3e9210868e8056b346f98d0323b6da21f9` | `kernels/glue.cu` (`moe_combine_kernel`), `src/reference.rs` (`moe_combine`) | **Reimplemented** on the GPU. | `tests/gemv.rs` (`glue_kernels_match_their_models`, bitwise) | 2026-09-28 |
| BF16 weight GEMV for 1-8 rows (a CTA of 4 warps, 2 outputs per warp, lanes streaming 16-byte weight chunks, butterfly reduction, K splits chosen from the weight's shape and added in split order by the last CTA of each block) | L : `kernels/fp8_gemm.cu` (`fp8_gemm_decode_kernel`, its fused split-K variant) and `kernels/common.cuh` (`ld_stream`, `ld_cached`, `unpack_bf16x8`, `warp_sum`), as of the second revision in the working tree | committed `92674b4d2c0cc247bb20f24d71efe892f056c5af84c8b6d26afbf63514582c60`, `46d7683545721508c08317fedda1bdd88ce2b1d05a1c86cf0971b4fd666817d5` (the second revision was uncommitted when read; it landed in `0e379e2`) | `kernels/gemv_bf16.cu`, `kernels/common.cuh` | Design reference, rewritten: BF16 weights (8 values per 16-byte load, K step 256), f32 `fma` accumulation per lane with no block scale, grouped launches (grid.z) for the forget and output gates, BF16 or f32 output, any `n % 8 == 0`; the split rule targets 512 CTAs. The load helpers and the butterfly are the same few lines. | `tests/gemv.rs` (bitwise against `src/reference.rs`, row independence for 1..8 rows), `examples/gemm_bench.rs` | 2026-09-28 |
| Device buffers, streams, events | K : `src/device.rs` | `4a3d38637adfd837b8e9ba95dcba35452d4cd7e1daa0e95bf216487f4020489a` @ `7cee788a37431c4f86fec36ce4cfa2eb23741c18` | `src/device.rs` | Pattern, rewritten: async copies on a stream, byte-range checks, page-locked mapped host memory (`PinnedBuffer`), the crate's error type. | every GPU test | 2026-09-28 |
| CUDA runtime and cuBLAS declarations | N | — | `src/cuda.rs`, `src/cublas.rs` | Signatures and enumeration values (`CUDA_R_16BF`, `CUBLAS_COMPUTE_32F`, `CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION`, `cudaHostAllocMapped`, ...) re-declared; the row-major-to-column-major mapping of `cublasGemmEx` written here. | `tests/gemv.rs` (`cublas_agrees_within_rounding`) | 2026-09-28 |
| Paged cache view, tails, window metadata (the kernels' contracts) | Q : `kernels/include/glm53f_dsa.h`, `src/cache.rs` | as committed at `5dd4dae77bd0e3469ce0e3a20b41c4ff554b253d` (working-tree revision read; it landed in `687e381`) | `src/kv.rs`, `src/kvplan.rs`, `src/forward.rs` | Used as specified: one physical page holds every DSA layer's 35,904-byte block (`page_stride` = the page), batch-local request indices with the page tables and tails gathered per pass. | `tests/verify_commit.rs`, `tests/snapshot.rs` | 2026-09-28 |
| KDA batch launches, replay commit, conv shift | K : `kernels/glm53f_kda.h` | as committed at `7cee788a37431c4f86fec36ce4cfa2eb23741c18` (working-tree revision read; it landed in `aec0083`) | `src/forward.rs` | Used as specified: per-slot state and conv arenas addressed by offsets; verify rounds keep each KDA layer's projection rows and replay inputs until the commit. | `tests/verify_commit.rs` (bitwise against serial steps) | 2026-09-28 |
| `KvSlot` and `ModelForward` | C : `src/model.rs` | (untracked when read; committed in `6d6a0dc`) | `src/serve.rs` | Implemented for `GlmKv` and `ServedForward`; the shell's GPU sampler applies draws and masks. | `tests/serve.rs` | 2026-09-28 |

## Test data

None is carried. The tests read the oracle's golden sets (`oracle/goldens`, or
`GLM53F_GOLDENS`), whose payloads are regenerated by `oracle/golden_layers.py`, and the
checkpoint named by `GLM53F_CHECKPOINT_DIR` / `GLM53F_EXPERTS_DIR`. `tests/kv_plan.rs` reads the
published configs committed in `crates/glm53f-model/tests/data`.
