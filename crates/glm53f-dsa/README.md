# glm53f-dsa

The DeepSeek sparse attention (DSA) layers of GLM-5.3-Flash: an f32 CPU
reference of the whole layer path, the cache formats, and CUDA kernels for the
indexer's top-512 selection and for sparse MLA over the FP8 latent.

GLM-5.3-Flash has 11 DSA layers (3, 7, …, 43). Each is MLA with a 512-dim
latent and **no RoPE** (64 heads of 256), plus its own indexer (32 heads of 128)
that pools index keys 4 tokens to 1, keeps the best 512 pools for each query
and adds the incomplete tail pool. Sparse attention therefore reads at most
2,051 cached latents per query and layer.

| Part | Where |
|---|---|
| Dimensions and constants | `src/config.rs` |
| Indexer: projections, k-pool compression, scores, per-request cache | `src/indexer.rs` |
| Top-k with a deterministic tie rule; expansion and the tail | `src/select.rs` |
| MLA, expanded (reference) and absorbed (engine) forms, error bound | `src/mla.rs` |
| The whole layer over a per-request cache, with every intermediate | `src/layer.rs` |
| FP8 E4M3 codec, block scales | `src/fp8.rs` |
| Cache formats and the page layout | `src/cache.rs` |
| Checkpoint and fixture readers (safetensors, JSON, SHA-256) | `src/safetensors.rs`, `src/weights.rs`, `src/golden.rs`, `src/json.rs`, `src/sha256.rs` |
| CUDA: indexer (pooled-key write, tail commit, score + top-512) | `kernels/dsa_index.cu` |
| CUDA: latent write, absorb, sparse attention, un-absorb | `kernels/dsa_mla.cu` |
| C ABI | `kernels/include/glm53f_dsa.h` |
| Rust FFI, device buffers, timing (feature `cuda`) | `src/ffi.rs`, `src/gpu.rs` |
| Benchmarks | `examples/dsa_bench.rs` |

## Semantics

The reference is `Glm5NextTextIndexer` and `Glm5NextTextAttention` in
`transformers` (`models/glm5_next/modeling_glm5_next.py`; see PROVENANCE.md).
For a request whose tokens start at position 0:

- **Per token:** index query `q = wq_b(q_resid)` (32 × 128, from the MLA query
  latent), index key `k = LayerNorm(wk(x))` (128, ε 1e-6), pool gate
  `g = compress_gate(x)` (128: **one gate per key channel**), head weights
  `w = weights_proj(x) · 32^-0.5`, and the MLA latent
  `c = RMSNorm(kv_a_proj_with_mqa(x))` (512, cached).
- **Pools:** tokens `4p … 4p+3` form pool `p`. Its key is, per channel `d`,
  `Σ_i softmax_i(g_i[d] + ape[i][d]) · k_i[d]`. A pooled key depends only on its
  own 4 tokens, so it is computed once, when its last token arrives. With left
  padding the reference aligns pools to the first real token; the engine never
  pads.
- **Visibility:** a query at position `p` sees complete pools `0 … n-1` with
  `n = (p+1)/4` (a pool is a candidate only when its last token is visible).
- **Score:** `s = Σ_h w_h · relu(128^-0.5 · q_h · key)` (f32).
- **Selection:** the best `min(512, n)` pools. Order: higher score first, equal
  scores (including +0/−0) to the lower pool; NaN never selected.
  `torch.topk` leaves ties unspecified, so the reference may differ only among
  exactly equal scores. Rows with `n ≤ 512` (positions ≤ 2,050) keep every pool:
  attention is dense causal.
- **Tail:** tokens `4n … p` (0 to 3 of them) are always attended.
- **Attention:** over the ascending union of the kept pools' tokens and the
  tail, softmax scale `256^-0.5`.

The reference returns each row as `[min(512, P) pools × 4 tokens in score
order (-1 for pools not visible to the row)] [3 tail slots] [-1 padding]` to
width 2,051, where `P` is the number of complete pools in the whole call.
`Selection::reference_row` reproduces that layout; `reference_topk_indices`
transcribes the reference's index arithmetic, including left padding.

## Absorbed versus expanded MLA

The reference expands every attended latent through `kv_b_proj`
(`k_h = W^K_h c`, `v_h = W^V_h c`). The engine folds the key half into the query,
`q'_h = (W^K_h)ᵀ q_h` (512 wide), attends in latent space
(`o'_h = Σ_j softmax_j(256^-0.5 · q'_h · c_j) c_j`) and applies the value half
afterwards (`o_h = W^V_h o'_h`). The two are equal in exact arithmetic.

**f32 bound.** Computing a score both ways with sequential f32 sums, each form is
a two-stage product (a 512-term then a 256-term sum, or the reverse), so
`|s_absorbed − s_expanded| ≤ 2 γ_{769} Σ_{r,l} |q_r| |W_rl| |c_l|` with
`γ_n = n·2^-24 / (1 − n·2^-24)` (`mla::score_error_bound`). Measured at the real
dimensions with checkpoint-scale weights (`tests/reference.rs`): the worst score
difference is 1.8·10⁻⁴ of that bound; per-head outputs differ by 1.0·10⁻⁶
relative L2.

**Against the BF16 reference.** Run in its native dtype, the reference rounds
`k` and `v` (the `kv_b_proj` output), the scores and the probabilities to BF16
(relative 2⁻⁹ each). The absorbed form never materializes `k` or `v`; the GPU
kernel rounds `q'` and the probabilities to BF16 instead. Both sides therefore
carry errors of the same order (≈2⁻⁹ relative per operand). The oracle's
contract is f32, where the CPU reference matches to about 10⁻⁶.

## Cache formats

Per token and DSA layer: one **latent record**. Per complete pool: one **pooled
index key**. Per request and layer: the **tail**.

```text
Latent record, 528 bytes (one token, one layer):
  [  0, 512)  512 x E4M3 codes (latent channel order)
  [512, 528)  4 x f32 LE scales, one per 128-channel group
  value[i] = e4m3(code[i]) * scale[i / 128]

Pooled index key, 132 bytes (one pool, one layer):
  128 x E4M3 codes + 1 x f32 scale; value[i] = e4m3(code[i]) * scale
  stored structure-of-arrays in the page: 16 x 128 code bytes, then 16 x f32

Tail, 1,552 bytes (one request, one layer):
  [ 0,  4) u32 count (0..=3)    [ 4, 16) reserved, zero
  [16, 16 + 512 c)  token c: 128 x BF16 key (after k_norm), 128 x BF16 gate

Page (64 tokens = 16 pools), per DSA layer, 35,904 bytes:
  [     0, 33,792)  64 latent records (token slot t at 528 t)
  [33,792, 35,840)  16 x 128 pooled-key codes
  [35,840, 35,904)  16 x f32 pooled-key scales
```

Over 11 layers that is 6,171 bytes per token, the figure in `docs/SIZING.md`.

**Why these choices.**

- **E4M3 with power-of-two scales.** The writer picks the smallest `2^k` with
  `amax ≤ 448·2^k`. E4M3 has the same relative spacing in every binade, so
  multiplying by a power of two changes no rounding except in the subnormal
  range, and `code × scale` is exact in BF16 and f32: the GPU decodes latents
  into its BF16 tensor-core operands without rounding, and the CPU and GPU
  writers produce identical bytes (tested). Readers accept any f32 scale.
- **One scale per 128 channels.** With power-of-two scales the granularity does
  not change the error measurably: on layer 3's real weights the stored-latent
  error is 2.651·10⁻² relative L2 for groups of 32, 128 and 512 alike (the
  scale only has to prevent overflow). 4 × f32 per 128 keeps the record 528
  bytes, a multiple of 16 (every record stays 16-byte aligned), and is the
  layout of the existing sm_120 GLM-5.3-Flash sparse-MLA readers (the 528-byte
  record in the SparkInfer fork's b12x), so those kernels can read this cache.
- **Pooled keys in FP8 with one f32 scale**, not BF16: 132 instead of 256 bytes
  per pool. Its effect on selection is measured below. The page stores codes
  and scales as separate arrays so each key's 128 codes stay 16-byte aligned for
  vector loads.
- **Tail in BF16:** the reference caches the index key and gate in the model's
  dtype; pooling from the same BF16 values keeps the pooled key's inputs
  identical, and the tail is small (1.5 KiB per request and layer).

**Measured error (layer 3, real weights).** `tests/real_data.rs` loads layer 3
from the official checkpoint. The inputs are **proxies**: 2,400 token-embedding
rows (pseudo-random ids) through layer 3's input RMSNorm, so they carry the
checkpoint's per-channel structure but are not real layer-3 activations. The
last 8 positions (598–600 visible pools) are the queries.

| Quantity | BF16 | FP8 E4M3, pow2 scale (default) |
|---|---:|---:|
| Stored latent, rel. L2 | 1.66e-3 | 2.65e-2 (amax/448 scale: 2.57e-2) |
| Stored pooled key, rel. L2 | 1.66e-3 | 2.66e-2 |
| Index scores, rel. L2 | 2.21e-3 | 3.59e-2 |
| Kept pools shared with f32 keys (of 512) | 100% | 99.4% |
| Per-head attention output, rel. L2 (same selection) | 6.2e-4 | 9.7e-3 |
| Layer output after `o_proj`, rel. L2 (same selection) | 8.5e-4 | 1.3e-2 |
| Layer output, FP8 keys (own selection) + FP8 latents | | 2.5e-2 |

These are per-layer tensor errors, not a quality metric; the design's quality
gate is end-to-end KL against BF16 logits (`docs/DESIGN.md` §5, D1). On the same
layer, the GPU path (FP8 records, BF16 operands) is within 1.3·10⁻³ of the CPU on
identical FP8 latents.

## CUDA kernels

All entry points are in `kernels/include/glm53f_dsa.h`; they validate their
arguments, never synchronize, and address the cache through per-request page
tables. Shapes are fixed to GLM-5.3-Flash.

### (a) Indexer: score and top-512 over up to 256K pools

`glm53f_dsa_index_select` never materializes a score matrix. It runs three
kernels:

1. **`index_tiles`**, grid (chunks, rows). A block owns one query row and a
   contiguous chunk of pools. Its 8 warps stream 8-pool tiles:
   - each lane loads 32 bytes of a pooled key (two 16-byte loads) and the
     pools' scales; the next tile's loads are issued before the current tile's
     products (software pipelining);
   - `mma.sync.m16n8k16` with f16 operands and f32 accumulation: the row's
     32 × 128 query sits in registers as A fragments, E4M3 codes become f16
     exactly (`cvt.rn.f16x2.e4m3x2`). The 128 key channels are permuted so a
     lane's B operand for all 8 k-steps is one contiguous 32-byte slice; the
     query uses the same permutation;
   - the query is scaled per head by a power of two into f16's range (exact)
     and the inverse folded into the head weight; ReLU and head weights in f32;
     the head sum by shuffles; `× key scale`;
   - each score becomes a 64-bit key, `ordered(score) << 32 | ~pool`; keys above
     the block's running threshold go to a 2,048-entry shared buffer. When it
     fills, a block radix select keeps the best 512 and raises the threshold.
     The block writes at most 512 keys.
2. **`index_merge`**, groups of 8 lists to 1 (radix select), repeated.
3. **`index_finalize`**: kept pools ascending (bitonic sort), expanded tokens
   plus the tail, counts. Rows with `n ≤ 512` skip scoring and keep everything.

Keys are unique and totally ordered, so the result does not depend on the chunk
plan, and it equals a CPU selection over the same scores exactly (integer-valued
data makes the scores bit-identical and exercises many ties). The workspace is
`rows × chunks × 4 KiB` (plus a merge level), independent of context length for
a fixed plan. `glm53f_dsa_index_plan` picks chunks of at least 256 pools and
about two blocks per multiprocessor.

**Tail and pooled-key writes.** `glm53f_dsa_index_pool_write` computes every pool
a window completes (LayerNorm of the raw key, BF16 key and gate, per-channel
softmax pooling, FP8) with earlier tokens taken from the tail record.
`glm53f_dsa_index_tail_commit` rewrites the tail after a verify round from the
old tail and the accepted rows. Pools completed by rejected rows are written
beyond the committed length and overwritten later; no reader sees them.

### (b) Sparse MLA, decode and verify

`glm53f_dsa_mla_absorb_q` (bit-exact with the CPU), then
`glm53f_dsa_mla_sparse_attn`, then `glm53f_dsa_mla_unabsorb_v`. The attention
kernel, grid (rows, 4 / head_groups, splits), for each 64-token tile:

- resolves the tokens' records through the page table and decodes them once
  into a padded BF16 tile in shared memory (16-byte loads, one record per warp
  step), shared by 1, 2 or 4 groups of 16 heads;
- scores 16 tokens × 16 heads per warp with BF16 WMMA over the 512 channels;
- keeps an online softmax in f32 and rescales the running output in registers
  through a factor fragment loaded in the accumulator's own layout;
- accumulates `P (BF16) × V` in f32, each warp owning 128 output channels.

With `splits > 1` each row's tokens are split across blocks and a second kernel
merges the partials by their (max, sum). The structure follows ds41rt's
DeepSeek-V4.1 sparse attention (see PROVENANCE.md).

### (c) Prefill

The same entry points take prefill-sized row counts. `index_select` puts one
block per row and chunk (the plan gives one chunk per row at 4,096 rows);
`sparse_attn` uses 4 head groups per block so all 64 heads share each decoded
tile. Measured costs are below. **Design for the next step:**

- **Indexer, row blocks.** A block holds 2–4 consecutive rows' queries
  (64–128 MMA rows) and streams each key tile once for all of them, cutting L2
  traffic by the block height; the 1M-token prompt is otherwise dominated by
  `Σ_p p/4 × 8,192` flops (≈1.1·10¹⁵ per layer). An FP8 × FP8 variant
  (`mma.m16n8k32.e4m3`, query quantized per head) doubles the rate but changes
  numerics and needs the KL gate.
- **Sparse MLA, shared selections.** Consecutive prompt rows' selections overlap
  heavily. A block over R consecutive rows attends over the union of their
  tokens with per-row masks, decoding each latent tile once for R × 64 heads.
- **Dense start.** Rows at positions ≤ 2,050 attend to every earlier token; a
  causal flash-attention kernel (MQA: one latent shared by 64 heads) reuses
  each tile across a block of rows instead of re-reading it per row.
- **Pools and latents for a chunk** are written by the same `pool_write` and
  `latent_write` kernels (any row count).

## Tests

```sh
# CPU (default, no CUDA toolkit needed)
cargo test -p glm53f-dsa

# CUDA kernels against the CPU reference on the local GPU (skips if < 2 GiB free)
cargo test --release -p glm53f-dsa --features cuda --test gpu

# Real weights (layer 3 of the official checkpoint; skips without the variable)
GLM53F_CHECKPOINT=/path/to/checkpoint cargo test --release -p glm53f-dsa --test real_data -- --nocapture

# Oracle fixtures (oracle/goldens/layer03-*; skip when absent)
GLM53F_CHECKPOINT=/path/to/checkpoint cargo test --release -p glm53f-dsa --test goldens -- --nocapture

# Kernel timings
cargo run --release -p glm53f-dsa --features cuda --example dsa_bench
```

`GLM53F_NVCC` (default `/usr/local/cuda/bin/nvcc`), `GLM53F_CUDA_ARCH` (default
`sm_89`; `sm_120` for the RTX 5090) and `GLM53F_CUDA_LIB` configure the build;
`GLM53F_GOLDENS` moves the fixture root.

| Suite | Checks |
|---|---|
| unit (`src/`) | E4M3 round-to-nearest-even against brute force at every midpoint, power-of-two scale minimality, BF16/F16 rounding, record and page layout sizes, tail round trip, key order, SHA-256 vectors |
| `tests/reference.rs` | absorbed = expanded within the f32 bound at real dimensions; whole-layer forms agree; window size (1, 3, 8, all) gives bit-identical results; k-pool and tail edges (lengths 1–64 around pool boundaries, dense and sparse rows); the reference output layout, with and without left padding; deterministic ties under any chunking; the 512-pool budget over 3,000 pools |
| `tests/gpu.rs` | latent write byte-exact; pooled-key write and tail commit; selection exact on integer data with ties (three chunk plans, three merge levels); random data within an f16 bound; 1M tokens (262,144 pools) at 1 and 8 rows; sparse attention for 2,051 / 2,048 / 777 / 64 / 1 tokens, split and unsplit, 1/2/4 head groups; absorb bit-exact and the absorbed GPU path against the expanded CPU; layer 3 on real weights against fixtures (or a stand-in) |
| `tests/real_data.rs` | FP8 against BF16 on layer 3's real weights (table above) |
| `tests/goldens.rs` | the indexer from fixtures alone (scores, selections, the `index_topk = 16` variant); the whole layer from `attn_norm` with weights, prompt and decode steps; a synthetic fixture round trip that runs by default |

## Timings (RTX 4090, sm_89; a development proxy for the RTX 5090)

CUDA events, mean of 20 runs after 3 warm-ups, random data, one request.

**Indexer, whole call** (score + top-512 + merges + finalize), µs:

| rows \ context | 4K | 32K | 128K | 256K | 1M |
|---:|---:|---:|---:|---:|---:|
| 1 | 17 | 25 | 34 | 38 | 59 |
| 2 | 17 | 26 | 36 | 45 | 77 |
| 4 | 18 | 26 | 40 | 51 | 114 |
| 8 | 18 | 28 | 49 | 68 | 190 |

At 8 rows × 1M the tile kernel runs at about 90 TFLOP/s of score arithmetic
(f16 products, f32 accumulation). At 1 row the call is latency-bound: the tile
kernel reads the 34.6 MB of pooled keys at about 0.6 TB/s and the merge and
finalize launches add about 15 µs.

**Sparse MLA** (2,051 tokens per row, best split plan), µs: 1 row 22 (32 splits);
2 rows 30; 4 rows 42; 8 rows 51. Absorb 23 µs and un-absorb 35–39 µs (each
reads 16 MiB of `kv_b_proj`). Latent write 2 µs, pooled-key write 4 µs
(8 rows).

**Prefill-shaped:** `index_select` 4,096 rows at 32K context 2.3 ms; 2,048 rows
at 1M context 36.3 ms (121 TFLOP/s); `sparse_attn` 4.0 µs per row at 2,051
tokens (4 head groups, no split); 2.2 µs per row for a dense causal start of
2,048 rows.

## Open issues

- **Decode attention is latency-bound** at 1–8 rows (22–51 µs against a few µs
  of tensor work): each block walks a short serial chain (page lookups, tile
  decode, scores, softmax, P·V, partial store). Candidates: cp.async double
  buffering of the raw FP8 records, keeping the query in shared memory, BF16
  partials, and folding the merge into the last block.
- **Indexer at one row:** the merge levels and finalize cost ≈15 µs of the
  59 µs at 1M; fusing them into the tile kernel's last block would remove most
  of it.
- **Absorb/un-absorb read 32 MiB of BF16 `kv_b_proj` per layer and step**
  (about 0.2 ms per step over 11 layers on a 5090). FP8 would halve it; the
  checkpoint keeps this tensor in BF16, so that needs the KL gate.
- **Precondition:** `index_select`'s `max_pools` must cover every row's visible
  pools; the host knows the positions, the kernel does not check them.
- **sm_120 numbers** are not measured yet (the kernels use only sm_80/sm_89
  instructions and build for `sm_120` unchanged).
- **Fixtures:** the oracle's layer-3 sets did not exist when this crate was
  written; `tests/goldens.rs` was exercised on a synthetic set and on a
  stand-in made from the CPU reference on real weights.
