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
| Benchmarks | `examples/dsa_bench.rs`; before/after of the decode path: `examples/dsa_ab.rs` |

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
kernel rounds `q'` to BF16 and the probabilities to F16 (2⁻¹¹; the first kernel
rounded them to BF16) instead. Both sides therefore carry errors of the same
order or smaller. The oracle's contract is f32, where the CPU reference matches
to about 10⁻⁶.

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
  range, and `code × scale` is exact in BF16 and f32: the GPU feeds the codes to
  its F16 tensor-core operands exactly and applies the scales in f32 (the first
  kernel decoded `code × scale` into BF16, also exactly), and the CPU and GPU
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
gate is end-to-end KL against BF16 logits (`docs/DESIGN.md` §5, D1). On the
oracle's layer-3 fixtures (33 prefill rows), the GPU path (FP8 records, F16
operands) is within 5.6·10⁻⁴ (worst row, relative L2) of the CPU on identical
FP8 latents, and 3.2·10⁻² of the f32 fixture.

## CUDA kernels

All entry points are in `kernels/include/glm53f_dsa.h`; they validate their
arguments, never synchronize, and address the cache through per-request page
tables. Shapes are fixed to GLM-5.3-Flash. Call `glm53f_dsa_init` once per
process and device before any launch (it sets shared-memory limits).

### (a) Indexer: score and top-512 over up to 256K pools

`glm53f_dsa_index_select` never materializes a score matrix. After a small
`cudaMemsetAsync` of its arrival counters it runs one kernel, grid
(chunks, rows):

1. **Scoring.** A block owns one query row and a contiguous chunk of pools. It
   resolves the chunk's page ids into shared memory once; its 8 warps then
   stream 8-pool tiles, loads issued two tiles ahead:
   - each lane loads 32 bytes of a pooled key (two 16-byte loads) and the
     pools' scales;
   - `mma.sync.m16n8k16` with f16 operands and f32 accumulation: the row's
     32 × 128 query sits in registers as A fragments, E4M3 codes become f16
     exactly (`cvt.rn.f16x2.e4m3x2`). The 128 key channels are permuted so a
     lane's B operand for all 8 k-steps is one contiguous 32-byte slice; the
     query uses the same permutation;
   - the query is scaled per head by a power of two into f16's range (exact)
     and the inverse folded into the head weight; ReLU and head weights in f32;
     the head sum by shuffles; `× key scale`;
   - each score becomes a 64-bit key, `ordered(score) << 32 | ~pool`; keys
     above the block's running threshold go to a 2,048-entry shared buffer.
     When it fills, a radix select keeps the best 512 and raises the threshold.
2. **Merge tree.** Each block publishes its kept keys (at most 512) and arrives
   at its parent node of a fan-in-8 tree: `__threadfence`, then one `atomicAdd`
   on the node's counter in the workspace. The last block to arrive resets the
   counter, gathers the node's lists (every child slot and count in one round
   trip), keeps the best 512 and climbs to the next level. Merges of finished
   subtrees overlap the scoring of blocks still running; there are no further
   launches.
3. **Finalize.** The block that completes a row's root writes the kept pools
   ascending, their tokens plus the tail, and the counts. The order comes from a
   bitmap over the visible pools: each thread owns 32 bitmap words and publishes
   prefix counts of its four 8-word groups, so a pool's position costs at most
   seven popcounts after one scan. Rows with `n ≤ 512` keep everything without
   scoring.

At these sizes the block-wide steps are bound by latency (barriers and
shared-memory round trips), not arithmetic; per-phase clock stamps showed it.
Each step is therefore arranged to need few of them: the radix select takes
8-bit digits from the keys' highest differing bit (merged lists hold similar
scores, so their leading digits are shared), double-buffers its histogram (two
barriers a pass) and works on 32-bit values while the digit lies in the score
half; the compaction after it is one block-wide scan; the f16 query in shared
memory is padded and permuted so its fragment loads are free of bank conflicts.

Keys are unique and totally ordered, so the result does not depend on the chunk
plan, and it equals a CPU selection over the same scores exactly (integer-valued
data makes the scores bit-identical and exercises many ties). The workspace is
about `rows × chunks × 4 KiB` (`glm53f_dsa_index_workspace_bytes`), independent
of context length for a fixed plan. `glm53f_dsa_index_plan` gives about one
block per multiprocessor for one or two rows up to 64K pools and two per
multiprocessor otherwise, with chunks of at least `max_pools / 64` pools (64 to
256); this was measured on the RTX 4090. The fused kernel takes up to 262,144
pools (1M tokens).

**Variants.** `glm53f_dsa_index_select_prepared` skips the memset: zero-fill
the workspace once and use it only for calls with the same `rows` and `chunks`
(every completed call leaves the counters zero). `glm53f_dsa_index_select_v1`
is the first implementation (a scoring kernel, merge kernels, a finalize
kernel), kept for comparison. All three return identical scores and outputs
(tested up to 1M tokens).

**Tail and pooled-key writes.** `glm53f_dsa_index_pool_write` computes every pool
a window completes (LayerNorm of the raw key, BF16 key and gate, per-channel
softmax pooling, FP8) with earlier tokens taken from the tail record.
`glm53f_dsa_index_tail_commit` rewrites the tail after a verify round from the
old tail and the accepted rows. Pools completed by rejected rows are written
beyond the committed length and overwritten later; no reader sees them.

### (b) Sparse MLA, decode and verify

`glm53f_dsa_mla_absorb_q` (bit-exact with the CPU), then
`glm53f_dsa_mla_sparse_attn`, then `glm53f_dsa_mla_unabsorb_v`.

**Absorb and un-absorb** each read one 16 MiB half of `kv_b_proj`. A block owns
one head and up to 8 rows (templates for 1, 2, 4 and 8 rows per block), so each
weight is read once per row block, and every thread has eight weight rows
(absorb) or two value rows (un-absorb) in flight before its multiply-adds.
Absorb keeps the CPU's sequential summation order and stays bit-exact.

**Attention**, grid (rows, 4 / head_groups, splits), 128 threads per 16-head
group, over tiles of 32 tokens:

- the FP8 records of the next tile are loaded into registers while the current
  tile is computed; the block resolves its tokens through the page table once;
- the codes are converted to F16 (`cvt.rn.f16x2.e4m3x2`, exact) into one tile
  shared by the head groups, its 16-byte units XOR-swizzled so the conversion
  stores and the `ldmatrix` reads avoid bank conflicts; each token's four f32
  scales stay beside it;
- scores: warp w of a head group multiplies the group's 16 query rows (F16,
  scaled per head by a power of two so the BF16 query converts exactly; in
  registers for the whole block) by the codes of its 128 channels
  (`mma.m16n8k16`, B fragments by `ldmatrix`) and applies the tokens' group-w
  scales in f32; the four partial score tiles are summed through shared memory;
- every warp of the group runs the same online softmax in f32 registers, forms
  P × (the tokens' scales) directly as F16 A fragments and accumulates `P × V`
  in f32 against the codes of its 128 output channels (`ldmatrix.trans`).

With `splits > 1` each block takes an equal run of tiles and writes its partial
output with (max, sum); a second kernel merges them, with the split weights
computed once per head. `glm53f_dsa_mla_plan` recommends (splits, head groups):
about one block per multiprocessor, one head group per block up to 4 rows and
two beyond.

The first implementation, `glm53f_dsa_mla_sparse_attn_v1` (64-token tiles of
`code × scale` decoded into BF16, BF16 WMMA; see PROVENANCE.md), stays
available, and `glm53f_dsa_mla_sparse_attn` falls back to it when
`glm53f_dsa_init` was not called. The F16 path skips the E4M3 → f32 → BF16
conversion (sm_89 has no direct E4M3 → BF16) and rounds the probabilities to
F16 rather than BF16: the worst per-head error against the CPU drops from
1.9·10⁻³ to 2.2·10⁻⁴ relative L2.

**Design notes.** The plan in the previous revision (raw FP8 tiles in shared
memory with `cp.async` double buffering, conversion inside the fragment loads,
all 64 heads per block) changed where measurements disagreed:

- a register prefetch of the next tile's records overlaps its loads with the
  current tile's products, as a second `cp.async` buffer would, without the
  shared memory;
- converting once per tile into a shared F16 tile costs about 400 cycles per
  tile once swizzled; converting inside the fragment loads would repeat it for
  every head group and for both products (`P × V` reads the tile transposed);
- at 1–4 rows one 16-head group per block with more splits beats all 64 heads
  per block (more blocks in flight), and four groups per block (512 threads)
  spill registers.

### (c) Prefill

The same entry points take prefill-sized row counts. `index_select` puts one
block per row and chunk (the plan gives one chunk per row at 4,096 rows);
`sparse_attn` runs unsplit with 2 or 4 head groups per block, so each decoded
tile serves 32 or 64 heads. Unsplit, 1, 2 and 4 head groups give the same bits
(`tests/gpu.rs`, `sparse_attn_head_groups_are_bitwise`). Two entry points are
for prefill-sized passes, each with the bits of the one it replaces
(`prefill_absorb_and_unabsorb_match_bitwise`, 1 to 257 rows):

- `glm53f_dsa_mla_absorb_q_bf16` reads the BF16 query in place, with a row
  stride (its f32 value is exact), so the forward no longer widens it to f32
  first;
- `glm53f_dsa_mla_unabsorb_v_rows` writes the BF16 output the forward rounds
  anyway (and the f32 one only for a test's tap). A block stages 16 rows of the
  latent output in shared memory. Each warp sums 32 outputs at once (8 rows × 4
  value rows), each lane in `unabsorb_kernel`'s order (pairs of latent columns
  as `fma(w0, o0, w1 * o1)`, the product nvcc contracts there), then reduces
  them by recursive halving, which adds the same subtree sums as the butterfly.
  On the RTX 4090, in the forward: 1.53 ms for 2,048 rows against 2.64 ms
  (3.29 against 5.67 at 4,096).

Measured costs are below. **Design for the next step:**

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
GLM53F_CHECKPOINT_DIR=/path/to/checkpoint cargo test --release -p glm53f-dsa --test real_data -- --nocapture

# Oracle fixtures (oracle/goldens/layer03-*; skip when absent)
GLM53F_CHECKPOINT_DIR=/path/to/checkpoint cargo test --release -p glm53f-dsa --test goldens -- --nocapture

# Kernel timings; before/after of the decode path
cargo run --release -p glm53f-dsa --features cuda --example dsa_bench
cargo run --release -p glm53f-dsa --features cuda --example dsa_ab
```

`GLM53F_NVCC` (default `/usr/local/cuda/bin/nvcc`), `GLM53F_CUDA_ARCH` (default
`sm_89`; `sm_120` for the RTX 5090) and `GLM53F_CUDA_LIB` configure the build;
`GLM53F_GOLDENS` moves the fixture root.

| Suite | Checks |
|---|---|
| unit (`src/`) | E4M3 round-to-nearest-even against brute force at every midpoint, power-of-two scale minimality, BF16/F16 rounding, record and page layout sizes, tail round trip, key order, SHA-256 vectors |
| `tests/reference.rs` | absorbed = expanded within the f32 bound at real dimensions; whole-layer forms agree; window size (1, 3, 8, all) gives bit-identical results; k-pool and tail edges (lengths 1–64 around pool boundaries, dense and sparse rows); the reference output layout, with and without left padding; deterministic ties under any chunking; the 512-pool budget over 3,000 pools |
| `tests/gpu.rs` | latent write byte-exact; pooled-key write and tail commit; selection exact on integer data with ties (three chunk plans, three merge levels); random data within an f16 bound; the fused selection, its prepared variant and the first implementation identical (scores, pools, tokens) for three plans and at 1M tokens (262,144 pools, 1 and 8 rows); one workspace reused on garbage, after one zero-fill, interleaved and inside replayed CUDA graphs; sparse attention for 2,051 / 2,048 / 777 / 64 / 1 tokens, split and unsplit, 1/2/4 head groups and the recommended plan, current and first kernel each against the CPU and against each other; absorb bit-exact (and its BF16 output) for 1–17 rows, un-absorb within its summation bound; the absorbed GPU path against the expanded CPU; layer 3 on real weights against the oracle fixtures |
| `tests/real_data.rs` | FP8 against BF16 on layer 3's real weights (table above) |
| `tests/goldens.rs` | the indexer from fixtures alone (scores, selections, the `index_topk = 16` variant); the whole layer from `attn_norm` with weights, prompt and decode steps; a synthetic fixture round trip that runs by default |

## Timings (RTX 4090, sm_89; a development proxy for the RTX 5090)

One request, random data, µs per call, CUDA events on one stream. "Before" is
the first implementation (`index_select_v1` with its chunk plan,
`sparse_attn_v1`, and the first absorb/un-absorb kernels, timed from the commit
that introduced them); "after" is the current default. From
`examples/dsa_ab.rs`:

- **graph**: 10 calls captured in a CUDA graph and replayed (a steady serving
  loop). The inputs stay L2-resident across calls.
- **cold**: one call after 96 MiB of reads evicted the L2 (72 MiB), timed
  alone. In a decode step each DSA layer's pooled keys (35 MB at 1M tokens),
  latents and `kv_b_proj` (32 MiB) come from DRAM, so this is the realistic
  column.

The GPU is shared with other work; repeated runs move individual cells by up to
about 10%.

**Indexer** (score + top-512, whole call; after = `index_select` with its
counter memset, prepared = `index_select_prepared`):

| rows | context | before graph | after graph | prepared graph | before cold | after cold |
|---:|---|---:|---:|---:|---:|---:|
| 1 | 4K | 15.3 | 9.5 | 8.8 | 20.1 | 12.3 |
| 2 | 4K | 15.5 | 9.7 | 8.9 | 20.9 | 13.4 |
| 4 | 4K | 16.1 | 9.8 | 9.0 | 21.3 | 13.1 |
| 8 | 4K | 16.1 | 10.9 | 9.3 | 21.2 | 13.3 |
| 1 | 128K | 31.1 | 19.9 | 19.1 | 38.5 | 24.6 |
| 2 | 128K | 32.8 | 22.1 | 20.7 | 39.5 | 25.1 |
| 4 | 128K | 37.5 | 27.5 | 25.7 | 42.9 | 29.9 |
| 8 | 128K | 45.2 | 37.5 | 36.1 | 50.7 | 39.6 |
| 1 | 1M | 53.6 | 42.9 | 40.3 | 79.7 | 66.2 |
| 2 | 1M | 70.4 | 56.3 | 54.8 | 78.4 | 65.2 |
| 4 | 1M | 103.4 | 91.6 | 88.3 | 107.5 | 92.6 |
| 8 | 1M | 170.9 | 159.9 | 158.4 | 173.5 | 162.6 |

Fusing the merges and finalize into the scoring kernel removed the launches,
but the ~15 µs those stages cost at one row was mostly serial device work (radix
passes, gathers, sorting), not launch gaps; the gain came from shortening that
work. At 1M tokens and 4–8 rows scoring dominates: it runs at about 90–100
TFLOP/s of f16 products with f32 accumulation, near the 4090's rate for that
mode. At one row, cold, it streams 35 MB of pooled keys at about 0.5 TB/s.

**Sparse MLA** over 2,051 selected tokens (best splits × head groups per
kernel; cold uses the best graph plan). `glm53f_dsa_mla_plan`'s plan is the best
one from 2 rows up and within 0.6 µs of it at one row.

| rows | context | before graph | after graph | before cold | after cold |
|---:|---|---:|---:|---:|---:|
| 1 | 4K | 14.8 (40×2) | 10.1 (24×1) | 18.7 | 13.8 |
| 2 | 4K | 22.5 (24×2) | 13.0 (16×1) | 27.8 | 16.4 |
| 4 | 4K | 31.9 (12×2) | 19.0 (8×1) | 36.8 | 23.1 |
| 8 | 4K | 50.1 (8×2) | 28.4 (8×2) | 56.3 | 31.8 |
| 1 | 128K | 14.8 (40×2) | 10.2 (24×1) | 18.7 | 13.7 |
| 2 | 128K | 22.5 (24×2) | 13.1 (16×1) | 28.4 | 16.4 |
| 4 | 128K | 32.0 (12×2) | 19.1 (8×1) | 39.4 | 23.3 |
| 8 | 128K | 50.1 (8×2) | 28.5 (8×2) | 60.9 | 32.5 |
| 1 | 1M | 14.8 (40×2) | 10.2 (24×1) | 18.8 | 14.3 |
| 2 | 1M | 22.6 (24×2) | 13.1 (16×1) | 28.7 | 16.8 |
| 4 | 1M | 32.1 (12×2) | 19.2 (8×1) | 39.6 | 23.6 |
| 8 | 1M | 50.2 (8×2) | 28.5 (8×2) | 61.6 | 32.5 |

**Absorb and un-absorb** (each reads 16 MiB of `kv_b_proj`):

| rows | absorb before graph | absorb after graph | absorb before cold | absorb after cold | un-absorb before graph | un-absorb after graph | un-absorb before cold | un-absorb after cold |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 24.2 | 4.6 | 45.8 | 21.0 | 37.5 | 4.7 | 48.2 | 21.8 |
| 2 | 24.2 | 6.5 | 45.9 | 21.5 | 38.3 | 4.9 | 49.0 | 21.6 |
| 4 | 24.2 | 10.1 | 46.1 | 26.1 | 39.8 | 7.2 | 49.9 | 22.5 |
| 8 | 24.4 | 13.0 | 47.2 | 30.1 | 42.3 | 10.8 | 53.1 | 23.2 |

Cold, one 16 MiB half is read at about 0.8 TB/s, close to what the 4090's
DRAM delivers.

**Decode path of one DSA layer**, one row at 128K (indexer + absorb +
attention + un-absorb): before 38.5 + 45.8 + 18.7 + 48.2 = 151 µs cold
(108 µs warm); after 24.6 + 21.0 + 13.7 + 21.8 = 81 µs cold (40 µs warm). At
8 rows: 212 → 125 µs cold.

**Other kernels** (`examples/dsa_bench.rs`, eager): latent write 2 µs,
pooled-key write 4 µs (8 rows). **Prefill-shaped:** `index_select` 4,096 rows at
32K context 2.2 ms; 2,048 rows at 1M context 37.0 ms (119 TFLOP/s, as before);
`sparse_attn` 2.5–2.8 µs per row at 2,051 tokens (4 head groups, unsplit; the
first kernel 3.8–4.0 µs); 1.4 µs per row for a dense causal start of 2,048 rows (first
kernel 2.2 µs).

## Open issues

- **Indexer merge levels.** Each level of the fan-in-8 tree costs about
  2–3 µs of block-wide latency (the gather round trip, two or three radix
  passes, the compaction); at 128K and one row there are three levels, roughly
  half of the call. Wider levels need more shared memory per block; a threshold
  estimated from per-chunk maxima could let most chunks publish far fewer
  than 512 keys, but needs a fallback for adversarial score layouts.
- **Indexer scoring at 1M.** f16 × f16 with f32 accumulation runs at half rate
  on GeForce parts; one row is 2.1 GFLOP (≥ 13 µs on the 4090) and cold it is
  bound by DRAM (35 MB). FP8 × FP8 doubles the rate but changes numerics (KL
  gate).
- **Sparse MLA at one row** walks about three 32-token tiles per block; the
  split merge is a second kernel (about 2.6 µs of the 10 µs), and the partials
  are f32 (32 KiB per block).
- **Absorb/un-absorb read 32 MiB of BF16 `kv_b_proj` per layer and step**
  (about 43 µs cold on the 4090, over 11 layers about 0.5 ms per step). FP8
  would halve it; the checkpoint keeps this tensor in BF16, so that needs the
  KL gate.
- **`attn_v2_kernel<4>`** (4 head groups, 512 threads) spills registers (124
  bytes per thread on sm_89, 204 on sm_120); `glm53f_dsa_mla_plan` never picks
  it, and at prefill sizes it is no faster than 2 groups on the 4090.
- **Plans** (`glm53f_dsa_index_plan`, `glm53f_dsa_mla_plan`) are measured on the
  RTX 4090 only; any plan gives the same selection (and the same attention up to
  rounding), so callers may override them.
- **Precondition:** `index_select`'s `max_pools` must cover every row's visible
  pools; the host knows the positions, the kernel does not check them.
- **sm_120 numbers** are not measured yet (the kernels use only sm_80/sm_89
  instructions and build for `sm_120` unchanged). *(29 September 2026: the
  kernels run on the RTX 5090 in the served model; a DSA layer's attention per
  2,048-row prefill lane there, 9.66 ms, is in `docs/PERFORMANCE.md` §0. The
  kernels' own benches have not been recorded there.)*
