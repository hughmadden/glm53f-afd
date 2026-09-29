# glm53f-dflash

The DFlash2 speculative drafter of GLM-5.3-Flash (`incoai/GLM-5.3-Flash-DFlash2` at
`bf582e4e`): an f32 CPU reference, a CUDA forward that drafts for a batch of requests on the
coordinator GPU, and the seam the target forward and the scheduler call. The drafter proposes
seven tokens per step; the target verifies them in one window.

## Contents

| Path | What |
|---|---|
| `src/reference.rs` | The drafter in f32 on the CPU: context append, the block forward, logits through a caller-supplied LM head, the selector. A `bf16_io` mode reproduces the GPU forward's roundings |
| `src/selector.rs` | Top-k and the selector's walk (greedy and sampled), with the GPU's summation order |
| `src/cpu.rs` | Threaded GEMM against BF16 weights, RMSNorm, RoPE (the reference's own `inv_freq` rounding), SiLU |
| `src/weights.rs` | The drafter's tensors and the target rows it borrows, read with `glm53f-model`'s safetensors reader |
| `src/seam.rs` | The integration seam: `Drafter`, `Append`, `DraftRequest`, `Proposal`; `CpuDrafter` implements it |
| `src/gpu.rs` | `GpuDrafter` (feature `cuda`): device weights, per-request context rings (its own, or caller-owned: `GpuSlot::external`), `append_taps`, `launch` / `proposals`, rewinds, forks (`GpuSlot::follow`), cold restarts (`GpuSlot::restart`); its own stream or the target forward's (`with_borrowed_head_on`); the FP8 drafter (`new_fp8`, `fp8_with_head_on`: [The FP8 drafter](#the-fp8-drafter)) |
| `kernels/glm53f_dflash.h`, `kernels/dflash.cu` | The kernels behind a C ABI: RMSNorm, RoPE table, per-head norm + RoPE, ring stores, the dynamic convolution, split-K attention over the ring, SiLU x up, top-16, the selector walk |
| `src/blas.rs`, `src/device.rs`, `src/cuda.rs`, `src/ffi.rs` | cuBLAS (BF16 in, f32 accumulate and out), device buffers, runtime and kernel bindings |
| `src/goldens.rs`, `src/synth.rs`, `src/sha256.rs`, `src/bf16.rs` | Golden-set reader (digests checked), the goldens' synthetic taps, SHA-256, bfloat16 |
| `tests/reference.rs` | Invariants on a small random model (no data): pieces vs one append, the window, non-causal block and causal convolution, rewinds, the candidate limit, the seam |
| `tests/goldens.rs` | The CPU reference against the oracle's FP32 goldens (needs data) |
| `tests/gpu.rs` | Kernels bit for bit against the CPU functions; the forward against the reference on a random model and on the checkpoint (feature `cuda`) |
| `tests/gpu_fp8.rs` | The FP8 drafter: its GEMM against `glm53f-layers`' CPU model, the drafter against the reference on weights FP8 holds exactly, FP8 against BF16 (feature `cuda`) |
| `examples/dflash_bench.rs` | Draft-block and append timings, BF16 and FP8 |
| `examples/draft_replay.rs` | Acceptance of both drafters on recordings of the whole target (`glm53f-forward`'s `examples/draft_record.rs`) |
| `../../oracle/golden_dflash.py` | The golden capture: the pinned reference run in the oracle image |

```sh
cargo test -p glm53f-dflash                                  # CPU; data tests skip
GLM53F_DFLASH_DIR=<drafter> GLM53F_CHECKPOINT_DIR=<GLM-5.3-Flash with embed_tokens and lm_head> \
GLM53F_GOLDENS=oracle/goldens cargo test -p glm53f-dflash --release
GLM53F_NVCC=<nvcc> GLM53F_CUDA_LIB=<cuda lib64> ... cargo test -p glm53f-dflash --release --features cuda
```

## Sources

Every statement below cites one of these, at the given commit (digests in `PROVENANCE.md`):

- **D**: z-lab/dflash @ `07ebd93db9f4`, `dflash/model.py` — the reference drafter
  (`DFlash2DraftModel`, `CandidateSelector`, `GroupedDynamicCausalConv`) and its drafting loop
  `dflash_generate`.
- **S**: sgl-project/sglang at the head of PR #36708 (`2d4b6acea720`), the build the checkpoint's
  README says to serve it with: the GLM-5.3-Flash capture (`srt/models/glm5_next.py`,
  `kernels/ops/layernorm/mhc.py`), the DFLASH draft model (`srt/models/dflash.py`) and worker
  (`srt/speculative/dflash_worker_v2.py`, `dflash_utils.py`, `kernels/ops/speculative/dflash.py`).
- **S'**: sgl-project/sglang @ `926968b3777d`, the head of PR #36507 as merged into main (it carries
  #36708): `srt/models/glm5_next.py`, where the capture is final (see step 1).

## The computation, step by step

Shapes are the checkpoint's: hidden 4,096; 5 layers; 32 query heads and 8 KV heads of 128; SwiGLU
12,288; RMSNorm eps 1e-5; block 8; window 2,048; 16 candidates; selector rank 256; vocabulary
154,880 (ids 154,856.. are padding; the mask token is 154,856).

**1. The taps (what the target hands over).** For every token row the target has *committed*, the
drafter needs five of the target's hidden states, concatenated in this order: after target layers
5, 14, 24, 33 and 42 (`dflash_config.target_layer_ids`), a 20,480-wide row, BF16.

- *Which state.* The completed output of layer `k`: the four mHC residual streams after layer
  `k`'s FFN-site update (the value layer `k + 1` receives), contracted to one 4,096-vector by
  their **mean**. S captures "before layer `k + 1`" (`glm5_next.py:1403-1415`,
  `self.model.layers_to_capture = [val + 1 for val in layer_ids]`, with the comment "Capturing
  before layer k + 1 gives the completed output of layer k") and contracts with `hc_contract` when `dflash_capture` and
  `config.mhc` (`glm5_next.py:1078-1084`), which is `x.unflatten(-1, (4, -1)).mean(dim=-2)`
  (`mhc.py:1571-1573`); the unit test `test_glm5_next_dflash_capture.py` pins the mean. Between
  mHC layers the streams travel as `hidden_states` with `residual = None`
  (`communicator_mhc.py:299-322`); S' makes the capture use them directly ("mHC folds the residual
  into widened hidden state, so residual remains None", `glm5_next.py:953-963` at S'). It is not a
  single stream, not the HyperHead's weighted collapse and not the final norm. D's HF path agrees
  on the point (`hidden_states[layer_id + 1]`, the output of layer `layer_id`, `model.py:37-43`)
  but has no GLM-5.3-Flash support; the mean is S's.
- *Order.* The captured list is concatenated along the last axis (`aux_hidden_states.py:55-58`),
  layer 5 first, and must match `fc.weight`'s 20,480 input columns.

**2. Context features.** `feat = hidden_norm(fc(taps))`, one 4,096-vector per context row
(D `model.py:584`; S `DFlashDraftModel.project_target_hidden`, `dflash.py:663-680`).

**3. Context keys and values, per layer.** `k = RoPE(k_norm(k_proj(feat)), pos)`,
`v = v_proj(feat)`, with the same `k_proj`, `v_proj` and `k_norm` the block uses; no
`input_layernorm` and no convolution touch the context (D `model.py:384-393`: `k_ctx =
k_proj(target_hidden)`, concatenated with the block's keys before `k_norm` and RoPE; S
`kv_proj_only`, `apply_k_norm`, `apply_k_rope`, `dflash.py:320-354`, written to the draft KV pool by
`_append_target_hidden_to_draft_kv_by_loc`, `dflash_worker_v2.py:1293`).

**4. Which rows become context, and when.** Only committed rows. After a prefill, every prompt row
(S `dflash_worker_v2.py:1669-1716`). After a verify, the rows the target kept: the anchor and the
accepted drafts, `commit_len = accept + 1`, from the verify pass's own taps
(`dflash_worker_v2.py:2145-2159`, prefix-valid writes of `commit_lens` rows). D does the same by
cropping its draft cache back to the committed length after every draft (`model.py:250`) and
feeding the verify's first `produced` rows next time (`model.py:307-308`). Nothing is ever rolled
back after a verify: rejected rows never reach the drafter.

**5. The block.** `[anchor, mask x 7]`: the anchor is the last verified token (the previous
verify's bonus token, or the first token sampled after prefill) at position `P` = the committed
length, then seven mask tokens at `P + 1 .. P + 7` (D `model.py:197-199, 235-242`: `output_ids` is
filled with `mask_token_id`; S `kernels/ops/speculative/dflash.py:144-191`). The block's input is
the target's embedding of those ids (the mask token's row, 154,856, is a padding row of the
target's table: RMS about 1e-5), times `input_embedding_scale` (absent from the config: 1.0).

**6. Positions and RoPE.** Default RoPE, theta 10,000, all 128 dims, `rotate_half` pairs
`(i, i + 64)`; context rows at their absolute positions, block rows at `P + j` (D `model.py:246,
331-337`). The inverse frequencies are `1 / 10000^(2i/128)` with the reference's roundings (the
power in f64 rounded to f32, the reciprocal in f32): all 64 match the reference bit for bit.

**7. Attention.** Queries from the block (after `q_norm`), keys and values from the context rows
and the block's own rows; GQA 4 (query head `h` reads KV head `h / 4`); scale `1/sqrt(128)`;
**non-causal** (`is_causal: false`, D `model.py:365-367`; S `ENCODER_ONLY`, `dflash.py:104-139`);
**sliding window**: key position `q` is visible to query position `p` iff `|p - q| < 2048` (D
`_attention_mask`, `model.py:157-171`; S `window_left = sliding_window - 1`,
`dflash_utils.py:407-424`, and window `(2047, 2047)` for encoder-only layers,
`flashattention_backend.py:1331-1345`). So block row `j` reads context positions
`P + j - 2047 .. P - 1` and all eight block rows.

**8. A decoder layer** (D `model.py:433-475`; S `dflash.py:502-542`):

```text
x = input_layernorm(h);   (xc, k_att) = attention_conv.prepare(x)
a = o_proj(attention(xc));  h = h + attention_conv.finish(a, k_att)
x = post_attention_layernorm(h);   (xc, k_mlp) = mlp_conv.prepare(x)
m = down(silu(gate(xc)) * up(xc));  h = h + mlp_conv.finish(m, k_mlp)
```

**9. The dynamic convolution** (D `model.py:478-512`; S `_grouped_conv`, `dflash.py:396-466`).
`dyn = kernel_projection(x)`, viewed `[2 sides][2 taps][256 groups]` (1,024 values per row); both
sides come from `prepare`'s input. Side `s` of the convolution of `y`:

```text
out[l][c] = sum over o in {0, 1}, o <= l:  (base_kernel[s][o][c] + dyn[l][s][o][c / 16]) * y[l - o][c]
```

over the block's rows only: row 0 (the anchor) has no predecessor (S masks
`position % block >= tap`). `prepare` convolves the norm's output with side 0; `finish` convolves
the attention (or MLP) output with side 1. These are what make the drafter's residual stream carry
massive activations (up to about 1.8e6 in a few channels after layer 0 on the goldens).

**10. Draft rows and logits.** The final `norm`, then rows 1..7 (D `model.py:249`; S
`dflash_worker_v2.py:257, 972`) through the target's `lm_head` (`output_multiplier` 1, no
softcap: D `model.py:599-605`).

**11. The selector** (D `CandidateSelector.select`, `model.py:515-547`; S `_score_edges`,
`CandidateSelector`, `dflash.py:910-1063`, and `_selector_walk_kernel`,
`kernels/ops/speculative/dflash.py:250-288`). Per draft row `e`: the 16 largest logits (the
*unary* scores and candidate ids) and `h[e] = hidden_projection(row)` (256). The walk starts from
the anchor: `score(c) = unary[e][c] + sum_r P[prev][r] * h[e][r] * S[cand_c][r]`
(`predecessor_codebook` P, `successor_codebook` S, both `[154880][256]`), then `prev` becomes the
chosen candidate. Greedy takes the first maximum. Sampling draws from `softmax(score / T)`: D with
`torch.multinomial`, S by inverse CDF with one uniform per position (`index = #{c : u >=
cumsum_c}`, capped at 15); `q` rows are the distributions drawn from (one-hot for greedy rows in S).

**12. Verify (for the integration).** Greedy: accept while `draft[j] == argmax(target row j - 1)`,
then the target's token at the first mismatch is the bonus (S `dflash_utils.py:774-827`; D
`model.py:290-294`). Sampled: rejection sampling of each draft against `q` over its candidates
(D `_rejection_sample`, `model.py:94-124`; S `_selector_sampling_accept`).

## Per-request state

A request's drafter state is a ring of `window + block` = **2,056 rows** per layer, keys (after
`k_norm` and RoPE) and values, 8 x 128 BF16 each: 5 x 2 x 2,056 x 1,024 x 2 B = 42,106,880 B =
**40.16 MiB**. The engine's sizing (40.2 MiB, `DraftConfig::kv_window_tokens` = 2,056) is
confirmed. The least that works is 2,055 rows: a block reads at most 2,047 context rows (for its
row 0) plus its own 8.

- Position `p` lives in row `p % 2056`. During a draft the block's keys and values are written at
  `P .. P + 7`, into the rows of positions `P - 2056 .. P - 2049`, which no later query reads
  (the oldest one still needed is `P - 2047`), so drafting leaves the context intact.
- An append of more than 2,048 rows computes only the last 2,048 (the rest can never be read), so
  a long prompt appended in one call costs `fc` over 2,048 rows, not over the whole prompt.
- `rewind(len)` keeps what the ring still holds: positions at or above `old_len - 2048` are
  intact; any lower position a later draft would need is masked out (`lo`), never read stale.
  Returning by one row loses nothing; appending the rows again restores the full context.

## Numerics and tests

**CPU reference = the reference in FP32.** f32 throughout (the goldens' primary contract). Against
the oracle's FP32 goldens (`tests/goldens.rs`, 5 tests), relative RMS:

| Quantity | short, step 0 | short, step 1 | window, step 0 | window, step 1 |
|---|---:|---:|---:|---:|
| context features `hidden_norm(fc(taps))` | 3.9e-7 | 3.9e-7 | — | — |
| context keys / values, all layers | 3.9e-7 / 4.3e-7 | 4.0e-7 / 4.3e-7 | — | — |
| every layer's output | ≤ 1.5e-6 | — | ≤ 1.9e-6 | — |
| final hidden | 3.2e-6 | 3.0e-6 | 3.5e-6 | 5.4e-6 |
| logits | 2.9e-6 | 2.6e-6 | 3.2e-6 | 4.5e-6 |
| top-16 ids; path | equal; equal | equal; equal | equal; equal | equal; equal |

Every recorded intermediate of every layer (norms, dynamic coefficients, convolutions, block keys
and values, attention and MLP outputs) is within 7.8e-6; the selector's scores, lattice row and
sampled `q` within 2e-6. The window case pins the window: with 2,046 or 2,048 keys on the left
instead of 2,047 the final hidden moves by 3.4e-2 or 9.0e-3 against 3.5e-6. The oracle also
records the reference's native BF16 run: it is 3.5% (final) and 3.1% (logits) from FP32; it keeps
the FP32 path at step 0 but changes 3 of the 7 drafts at step 1.

**GPU forward.** BF16 weights; every GEMM input and the ring's keys and values rounded to BF16;
f32 accumulation (cuBLAS `COMPUTE_32F`), f32 residual stream, norms, RoPE, convolutions,
attention, logits and selector. The reference's `bf16_io` mode is exactly this rounding recipe; it
is 0.86% (final) and 0.81% (logits) from FP32, against 3.5% for the reference's own BF16.

- Kernels (`tests/gpu.rs`): RMSNorm, per-head norm + RoPE (positions up to 1,000,000), the dynamic
  convolution (with the residual add) and the selector walk (scores, choices; rank 256, 64, 40;
  greedy and sampled) are **bit for bit** the CPU functions; the top-16 is exact, ties included.
  The kernels spell out the reference's order of operations and are built with `--fmad=false`.
  Attention uses fused multiply-adds and an online softmax: it is not bit-compared.
- Forward vs the CPU reference in `bf16_io` mode, on the checkpoint, three requests in one batch
  (contexts 300, 2,100, 37): final hidden 4.6e-3 / 6.4e-3 / 9.1e-3, logits 4.5e-3 / 6.4e-3 /
  8.6e-3, top-16 overlap 109, 112, 111 of 112, **identical paths**. Why not closer: both round the
  same quantities to BF16, but where cuBLAS and the CPU accumulate a value in a different order it
  can round the other way (2.9% of the ring's values differ, by at most 2.4% of their row's RMS:
  one BF16 unit; the context GEMM sums 20,480 products), and the model carries that to about half
  a percent. The residual stream is already 6e-4 apart after layer 0's attention and stays near
  1e-3 through the five layers.
- Forward vs the FP32 goldens: final 8.8e-3 / 1.0e-2 / 1.0e-2 / 2.0e-2 and logits 8.2e-3 / 1.1e-2
  / 8.0e-3 / 1.5e-2 (short and window, steps 0 and 1); **all four paths equal the reference's**.
- On a small random model (`tests/gpu.rs`, no data needed): context appended in pieces, requests at
  lengths 100, 13 and 45 against a 40-row window, a sampled walk (same uniforms, same draws, `q`
  within 2.6e-3 over the shared candidates) and a rewind (same `lo` as the reference).

Tolerances: CPU vs goldens 1e-5 (observed 3e-6: summation order only); GPU vs CPU 1.5e-2 on the
checkpoint (observed at most 9.1e-3) and 5e-3 on the random model; GPU vs FP32 3e-2 (observed at
most 2.0e-2); paths equal except at a near-tie of the CPU's own scores (none occurred).

## Performance

RTX 4090 (sm_89), **shared with other jobs: indicative**. CUDA-event time of one draft block
(`examples/dflash_bench.rs`, 30 iterations after warm-up, median):

| Requests (rows) | context 2,100 (full window) | context 300 |
|---|---:|---:|
| 1 (8) | 4.26 ms | 4.24 ms |
| 4 (32) | 4.60 ms | 4.54 ms |
| 16 (128) | 6.37 ms | 5.45 ms |

Appends from taps already on the device: 8 rows (a verify commit) 0.35 ms for one request, 0.44 ms
for 16; a 2,048-row prompt tail 3.53 ms. Weights and LM head take 3.41 GiB on the device.

Where the time goes (nsys, context 2,100): at one request the 32 GEMMs read the drafter's 2.0 GB
and the LM head's 1.27 GB in 3.69 ms (0.89 TB/s); attention takes 0.24 ms and everything else
0.27 ms. At 16 requests the GEMMs are compute-bound (the LM head's 142 GFLOP in 1.57 ms) and
attention reads 40 MiB of ring per request per draft (1.22 ms). An estimate from bandwidth alone,
not a measurement: the 5090's 1.8 TB/s would bring the single-request block near 2 ms.

## The FP8 drafter

`GpuDrafter::new_fp8` (its own stream; the head uploaded, quantized, freed) and
`GpuDrafter::fp8_with_head_on` (the target forward's stream; the forward's head read once);
`glm53f-serve --drafter-fp8`. Off by default until its acceptance is measured on the target
hardware. The idea comes from two public four-Spark recipes (`NOTICE.md`): block-FP8 drafter
weights with an FP8 draft head, reported with acceptance unchanged, and NVFP4 drafter weights.

- **What changes.** The GEMM weights (`fc`, and per layer the fused QKV, `o_proj`, the fused
  gate/up, `down` and the two convolutions' kernel projections) are quantized at load to FP8 E4M3
  with one f32 scale per 128 x 128 block, the checkpoint's own scheme for its FP8 weights
  (`glm53f-layers`' quantizer: `scale = amax / 448`, `q = e4m3(w / scale)`), and the drafter gets
  its own FP8 copy of the LM head, quantized the same way; the target's head is not touched. The
  norms, the convolutions' base kernels and the selector's tensors stay BF16. Up to 8 rows (one
  request's block or draft rows, a commit of up to 8 rows) the GEMMs are `glm53f-layers`' FP8
  decode GEMM with BF16 activations, every product exact in f32, its K splits (at least two)
  summed in split order into f32 (`g53d_splitk_sum`); over 8 rows (several requests, a prompt's
  context) its W8A8 tensor-core GEMM (E4M3 activations per row and 128-group, k32 promotion).
- **What it cannot change.** The verify pass keeps a draft only where it equals the target's own
  pick (greedy, or the target's seeded sample), so the output is the same token for token;
  `glm53f-forward`'s `tests/draft_lossless.rs` and `tests/copy_windows.rs` pass with
  `GLM53F_TEST_NUMERICS=drafter-fp8`.
- **Memory.** 1.76 GiB (1.17 GiB of weights and scales, the head's copy 0.59 GiB) against the BF16
  drafter's 2.18 GiB, which reads the target's head in place: 0.42 GiB less.
- **Tests** (`tests/gpu_fp8.rs`):
  - the GEMM, up to 8 rows, bit for bit `glm53f-layers`' CPU model in its split order; over 8 rows
    within 2^-10 of the magnitude of exact block products (worst measured 4.3e-4);
  - on a random model whose weights and head FP8 holds exactly (block scales powers of two), the
    FP8 drafter against the CPU reference on every path of up to 8 rows, as closely as the BF16
    drafter: 99.9% of the ring values equal and the rest within one BF16 unit, final rows and
    logits within 4.9e-3 (the BF16 drafter 3.8e-3 on the model's own weights; bound 1e-2), the
    same paths;
  - on arbitrary weights, FP8 against BF16 (what FP8 moves): on the random model rings 5.1e-2,
    logits 1.2e-1 to 1.7e-1 (relative RMS), 82-87% of the top-16 candidates shared; on the
    checkpoint with synthetic taps (flat distributions, where the order moves most) logits 1.3e-1
    to 2.9e-1, 76-89% of the candidates shared.
- **Acceptance on real prompts** (`examples/draft_replay.rs` over recordings of the whole target
  by `glm53f-forward`'s `examples/draft_record.rs`: five chat cases, thinking off, all 45 layers
  with the official FP8 experts and FP8 KDA projections, teacher-forced on a reference reply; a
  round at every reply position, both drafters on the same contexts; RTX 4090). Drafts kept per
  round with all 7 verified, and tokens per round with the serving chain cut (τ 0.7):

  | Case (rounds; the target's greedy pick is the reply's next token at) | BF16: kept, tokens | FP8: kept, tokens | FP8 − BF16 kept per round (standard error) |
  |---|---:|---:|---:|
  | code (1,974; 64%) | 1.66, 2.41 | 1.65, 2.40 | −0.003 (0.006) |
  | prose (1,801; 48%) | 0.98, 1.77 | 0.98, 1.77 | +0.000 (0.003) |
  | counting (1,348; 99%) | 6.48, 7.41 | 6.46, 7.40 | −0.025 (0.009) |
  | structured (908; 100%) | 6.96, 7.95 | 6.96, 7.96 | +0.007 (0.005) |
  | rewrite (726; 100%) | 6.67, 7.64 | 6.67, 7.64 | −0.001 (0.013) |
  | all (6,757) | 3.69, 4.54 | 3.69, 4.54 | −0.005 (0.003) |

  All five drafted at once (the W8A8 path): −0.002 (0.003) in all. Where the reply is not the
  target's own greedy text (code, prose) a kept draft that leaves the reply ends the count, so
  those rows are lower bounds for both drafters alike. The one change beyond two standard
  errors, counting's −0.025 drafts a round, is 0.01 tokens a round with the chain cut.
- **Speed** (`examples/dflash_bench.rs`, RTX 4090, contexts of 2,100 rows): one draft block 4.26
  → 2.46 ms at one request, 4.66 → 4.76 ms at 4, 7.02 → 6.27 ms at 16; on the recorded contexts
  4.23 → 2.56 ms at one request. Context appends of 8 rows 0.39 → 0.21 ms for one request, but
  0.50 → 0.78 ms for 16 requests at once; a 2,048-row prompt 3.54 → 3.24 ms. Over 8 rows the
  W8A8 GEMM's 128 × 128 tiles give the 4,096-wide projections 32 blocks, too few to fill the
  GPU: several requests at once gain little, and a commit of many requests' rows costs more than
  cuBLAS. A W8A8 GEMM for few rows (K split across blocks) would remove that (not built).
- **Working memory** (reserved for drafts of 16 / 48 requests): 151.8 → 203.5 MiB / 382.8 →
  500.7 MiB, mostly the BF16 output the tensor-core GEMM writes and the drafter does not read
  (33 / 99 MiB).

## Integration

The target forward and the scheduler drive the drafter through `seam::Drafter` (implemented by
`GpuDrafter` and `CpuDrafter`); `GpuDrafter::append_taps` takes taps already on the device.
`crates/glm53f-forward` wires it in (`src/draft.rs`, `src/kv.rs`, `src/serve.rs`): the drafter
runs on the forward's stream, each slot's ring lives in the KV pool (`GpuSlot::external`), and
the calls below are the ones it makes.

What the **target forward** must provide:

- **Taps**, per committed row: at the entry of layers 6, 15, 25, 34 and 43 (the outputs of 5, 14,
  24, 33, 42), the mean of the four streams, rounded to BF16, written into a `[rows][20480]`
  capture buffer (layer 5's 4,096 first). Capture them for every row of prefill segments, decode
  rows and verify windows; for a window the drafter keeps only the committed prefix.
- **Embedding rows**: the anchor's row per draft (the forward gathers it for the verify window
  anyway) and, once at load, the mask token's row (154,856).
- **The LM head**: `GpuDrafter::with_borrowed_head` reads the forward's device copy (1.27 GB);
  `GpuDrafter::new` uploads its own.

The calls, mapped to `glm53f_coordinator::model`:

| Shell pass | Drafter call |
|---|---|
| `prefill` (a segment) | `append_taps(slot, taps of the segment's rows)`; only the prompt's last 2,047 rows are ever read |
| `decode` (one row) | `append_taps(slot, 1 row)` |
| `draft(rows)` | `Drafter::draft` (or `launch`, then `proposals`) with `anchor = row.last` at `slot.len()`; `Draft { tokens: tokens[..max], probs: conf[..max] }` (see below on `conf`) |
| `verify(windows)` | nothing (the window's taps are captured) |
| `commit(slots, keep)` | `append_taps(slot, the window's first keep rows)`: the anchor and the accepted drafts |
| `KvSlot::rewind` to a mark | `GpuSlot::rewind(len)` (`Drafter::rewind`) |
| `KvSlot::fork` from a mark of another slot | copy the source's ring, then `GpuSlot::follow(src, len)`: the source's state rewound to `len` |
| `KvSlot::reset` / release | `GpuSlot::reset` (`Drafter::reset`); the ring (40.16 MiB) is the slot's fixed state |
| host tier export / import | the ring is positional state: save it with the mark (one 40.16 MiB allocation per slot), or restart the context cold at that length (`GpuSlot::restart`: `len = lo = L`, drafts see only rows appended afterwards). The forward restarts cold |

For sampled requests the proposal carries `candidates` and `q` per position, which a
rejection-sampling verify needs; an exact-match verify (the shell's current one: a draft is
accepted while it equals the target's own pick for its row) needs only the tokens. The caller
draws the uniforms. `conf` is a softmax over a position's 16 candidates only, so it reads higher
than a full-vocabulary probability; the shell's chain-cut threshold (`GLM53F_SPEC_TAU`, carried
over from MiMo, whose drafter reported full-vocabulary probabilities) will want re-tuning against
measured acceptance.

## Choices and differences from the reference

- **Padding ids are never proposed**: candidates come from ids `< 154,856`
  (`SAMPLE_VOCAB`). D and S take the top-16 over all 154,880 rows (S's `lm_head.org_vocab_size`
  is `config.vocab_size`). In every golden case no padding id reaches the top 16, so the
  results are the same; the reference behaviour is `vocab_limit = 154,880`.
- Candidates are ordered by logit, ties to the lower id (D: `torch.topk(sorted=False)`; S: a sorted
  top-k). Greedy takes the first maximum; sampling is S's inverse CDF.
- `conf` (the chosen candidate's softmax probability at temperature 1) is this crate's addition,
  for the shell's verify-length policy.
- The GPU forward is more precise than S's all-BF16 serving (f32 residual stream and elementwise
  work); S itself is BF16.

## Limits

- The goldens use synthetic taps (the target is not run: its experts for layers beyond 4 are not
  in the oracle's subsets). The tap definition (step 1) comes from reading S, not from a numeric
  comparison with a running target. *(29 September 2026: on the whole model the drafter keeps
  51–71% of its verified drafts; `docs/PERFORMANCE.md` §0.)*
- One CUDA stream (its own, or the target forward's), no graphs, no fused kernels; the GEMMs are
  cuBLAS (the FP8 drafter's: `glm53f-layers`' FP8 kernels). Built and measured for
  sm_89 only; sm_120 is untested. *(29 September 2026: it also runs on the RTX 5090, `sm_120`, in
  every drafted measurement of `docs/PERFORMANCE.md` §0; its kernels' own benches there are not
  recorded.)*
- The attention kernel supports up to 4 query heads per KV head; the kernels fix head size 128,
  block 8, 16 candidates and rank at most 1,024.
