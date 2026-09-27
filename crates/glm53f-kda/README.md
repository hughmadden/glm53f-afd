# glm53f-kda

GLM-5.3-Flash's 34 KDA (Kimi delta attention) layers on the coordinator: the fused decode and
verify kernels behind a C ABI, an f32 CPU reference of the recurrence, and the numerics
reference for chunked prefill.

Everything outside the recurrent core (the q/k/v, gate and beta projections before it, and
`o_proj` after it) is a GEMM and lives elsewhere. This crate takes the projection outputs and
returns the gated RMSNorm's output, which is `o_proj`'s input.

## Contents

| Path | What |
|---|---|
| `kernels/glm53f_kda.h` | The C ABI: layouts, numerics and the contract of every entry point |
| `kernels/kda.cu` | The kernels, ported from TensorFold (see [PROVENANCE.md](PROVENANCE.md)) |
| `kernels/parity/tensorfold_kda.cu` | The source kernels, verbatim, linked only by the parity test |
| `src/cpu.rs` | f32 reference of the recurrent path, in the kernels' order of operations, plus the reference's own formulation (`cpu::literal`) |
| `src/chunked.rs` | The chunked (WY) form, as the reference evaluates prefill |
| `src/kernel.rs`, `src/device.rs` | Checked Rust wrappers and minimal device memory (feature `cuda`) |
| `src/ffi.rs`, `src/cuda.rs` | Raw bindings to the ABI and the few CUDA runtime calls used (feature `cuda`) |
| `src/goldens.rs` | Loader and checks for the oracle's golden fixtures |
| `examples/kda_bench.rs` | Throughput at the model's geometry |

The ABI:

| Function | Grid | What |
|---|---|---|
| `glm53f_kda_chain` | H blocks of 1,024 threads | One layer, one request, R rows from the committed state: conv and SiLU, L2 norms, decay, beta, delta rule, gated RMSNorm. Writes the outputs, optionally the state after the last row, and optionally the replay inputs |
| `glm53f_kda_chain_batch` | H × B | The same for B requests in one launch: each owns a row range and its own state and conv window |
| `glm53f_kda_replay` | H | The state after the first `keep` saved rows of a chain |
| `glm53f_kda_replay_layers` | L·H | The same for every layer of a request |
| `glm53f_kda_replay_batch` | L·H × B | Every layer of every request, each with its own `keep`: the commit after a verify round |
| `glm53f_kda_conv_shift`, `_batch` | C/256 (× L × B) | Advance conv windows past the kept rows, in place |

States may be updated in place (`state_out == state_in`) or not written at all.

## Build and test

```sh
cargo test -p glm53f-kda                                  # CPU only: no CUDA toolkit needed
cargo test -p glm53f-kda --features cuda --release        # plus the kernels on a GPU
cargo run  -p glm53f-kda --features cuda --release --example kda_bench
```

With `--features cuda`, `build.rs` compiles `kernels/*.cu` with nvcc:

- `GLM53F_NVCC` is the compiler (default `/usr/local/cuda/bin/nvcc`);
- `GLM53F_CUDA_ARCH` is the target (default `sm_89`, an RTX 4090 used as a development
  proxy; the RTX 5090 is `sm_120`);
- `GLM53F_CUDA_LIB` is where `libcudart` lives (default `/usr/local/cuda/lib64`).

The GPU tests skip, with a message, when no device is present. The golden test skips when
`oracle/goldens/` holds no set. Set `GLM53F_GOLDENS` to use another directory.

## Numerics contract

The recurrent, per-token path is the contract for decode and verify. For every row and head
(see `src/cpu.rs` for the exact order of operations):

| Step | Computation | Precision |
|---|---|---|
| Conv | `SiLU(Σ_tap w[c][tap] · u[c][t-3+tap])` over the q \| k \| v projections | f32, **one** bfloat16 rounding |
| Norms | `q ← q · 1/√(‖q‖² + 1e-6) · 128^-½`, `k ← k · 1/√(‖k‖² + 1e-6)` | f32 |
| Decay | `exp(lower · sigmoid(exp(A_log) · (a + dt_bias)))` per key channel, `lower = -5` | f32 |
| Beta | `bf16(sigmoid(b))` | bfloat16 |
| Delta rule | `S ← S·diag(decay)`; `d = (v − Sᵀk)·β`; `S ← S + k dᵀ`; `y = bf16(Sᵀq)` | f32 state, no fused multiply-add |
| Gated norm | `bf16(w · (y · 1/√(mean(y²) + eps)) · sigmoid(gate))` | f32, one rounding |

**Where this differs from the reference implementation** (`transformers` `glm5_next`):

1. **Conv rounding.** The kernels round once after conv and SiLU, as the fused `causal_conv1d`
   kernel does. The reference's pure-torch fallback rounds the conv output to bfloat16 before
   the SiLU as well. The two disagree on about a quarter of the conv outputs, by one or two
   bfloat16 ulps. That moves a layer's state by about 0.5% and changes most bfloat16 outputs
   by an ulp or more. `cpu::ConvRounding` provides both, and the layer check reports both.
   The oracle's f32 goldens round nowhere. Its native bfloat16 set ran the pure-torch path,
   which is the unfused rounding (`oracle/README.md`).
2. **Norms.** The kernels multiply by a reciprocal square root where the reference divides by
   the norm, and use `1/sqrt` where it uses `rsqrt`. This is ulp-level; in the tests it has not
   changed a single bfloat16 output.
3. **Layouts.** The state is stored `[H][DV][DK]`, the transpose of the reference's
   `[H][DK][DV]` (`cpu::transpose_state` converts). The conv window is `[3][C]`, time-major.
   The reference caches `[C][4]` and uses only the last three columns.
4. **Scope.** The kernels assume what GLM-5.3-Flash uses: `DK = DV = 128`, 4 taps, no conv
   bias, the lower-bound form of the forget gate (not the softplus form), bfloat16 conv and norm
   weights, and f32 `A_log` and `dt_bias`. This matches the official checkpoint.

**Verified bit for bit** (on an RTX 4090, `tests/gpu.rs`):

- The port against the source kernels compiled verbatim, at 64 heads and R = 1..8: outputs,
  final states, replay inputs, the replay of every prefix, and the three-layer replay.
- The replay of any prefix k ≤ R against a chain over those k rows alone. The first k
  outputs of a window also match a chain over k rows.
- R serial single-row steps, with the device conv shift between them, against one R-row
  window: outputs, state and the final conv window.
- A batched launch against one launch per request: the chain, the all-layer commit replay
  with a different `keep` per request, and the conv shift. Requests with zero rows are
  included.
- Padded strides, beta logits placed before q \| k \| v (a negative offset), and in-place
  states against the dense layout.
- The CPU replay of the kernel's own replay inputs against the kernel's state. The state
  update involves no transcendental function, so `cpu::update` is an exact model of it.

**Verified within tolerance:** the kernel against the CPU reference at 64 heads, R = 1..8,
on random inputs spanning every gate's range. The only difference between the two is the
device `expf` against the host's (an ulp or two):

| Quantity | Measured worst case | Test bound |
|---|---|---|
| decay multipliers | 1.1e-6 relative | 4e-6 |
| v, beta, normalized k | identical | ≤ 1 bfloat16 ulp; 2^-8 absolute for k |
| state after the window | 1.7e-7 relative (normwise) | 1e-3 |
| outputs | 2.9e-4 normwise; at most 8 of 49,152 elements differ | 2^-8 normwise, ≤ 0.1% of elements differ |

The CPU reference itself is checked against:

- hand-computed values: gates, conv, norms, the butterfly sum order, the delta rule's
  write-then-read;
- invariants: a saturated gate with beta = 0 decays the state by exactly `exp(-5)` per row
  down to zero; with no decay and beta = 0 the state is unchanged; replayed prefixes and serial
  steps are bitwise;
- its literal formulation: state 1.9e-7, no bfloat16 output differs.

**Relation to the chunked form.** In exact arithmetic the chunked (WY) form of
`chunk_kimi_delta_attention` equals the recurrence. `src/chunked.rs` follows that form step
by step. In f32 it agrees with the recurrence to ≤ 1.1e-6 on the read-out and ≤ 7.7e-6 on the
state (normwise, T up to 150, chunks of 64 and 16). Rounded to bfloat16, about 0.2% of
read-out elements differ, some by many ulps where the value is tiny.

## Golden fixtures

`tests/goldens.rs` checks every set under `oracle/goldens/` (see `oracle/README.md`), or under
`GLM53F_GOLDENS`, and skips cleanly when there is none. A set is a `manifest.json` (`tensors`:
`{name: {file, dtype, shape, sha256}}`, plus `notes`) next to raw little-endian `.bin` files.
Every digest is checked first.

**Names.** The oracle names its sets `layerNN-prefill` and `layerNN-decode`, and their tensors
`prefill.kda.q`, `decode.kda.core_out` and so on. Decode tensors are stacked over the steps.
A tensor is a KDA tensor when its layer is a KDA layer. The layer comes from a component of
the name (`layers.4.`, `L04.`) or, failing that, from the set's name. Its role is what is
left after the phase prefix, the layer and the module names (`kda`, `self_attn`,
`linear_attn`, `forget_gate`). Per-step names (`decode.sN.`) are skipped. `goldens::ROLES`
lists the roles and the names each accepts, so checkpoint names such as
`model.language_model.layers.4.self_attn.forget_gate.A_log` work too.

**Checks.**

- **Core** (`check_core`): `q`, `k`, `v` (after the conv), `g` (log decay) and `beta` run
  through the recurrence. The results are compared with `core_out`, the state after the rows
  (`state`, `state_final`) and, where `notes.state_heads` names them, the per-step states of
  those heads (`state_heads`).
  - The state before the rows is the set's `initial_state` if it has one. For a decode set it
    is otherwise the matching prefill set's final state; for a prefill set, zero.
- **Conv cache** (`check_conv_cache`): the cache after the rows (`conv_state`,
  `conv_state_final`, reference layout `[C, 4]`, oldest first) must be exactly the last four
  inputs (`qkv_preconv`). This pins the layout.
- **Layer** (`check_layer`): the whole layer, from the projections (`qkv_preconv`, `f_proj`,
  `b_logits`, `gate`) and the layer's KDA weights (conv weight, `A_log`, `dt_bias`,
  `o_norm.weight`), against `norm_out`. It runs both conv roundings. The oracle's sets record
  activations, not weights, so this check runs only on sets that also carry the weights.
- **Device recurrence** (`tests/gpu.rs`, feature `cuda`): the replay kernel on the golden's
  k, v, g and beta, against the golden state (within 1e-2, since the kernels keep v in
  bfloat16) and against the CPU replay (bit for bit).

**Tolerances**, normwise against the largest golden value:

| Golden | Bound | Why |
|---|---:|---|
| f32 (the oracle's primary contract) | 1e-4 | The same f32 recurrence, or its chunked form: rounding only (about 1e-5, see above) |
| bfloat16, per-token path | 1e-2 | A bfloat16 rounding flip |
| bfloat16, chunked path | 5e-2 | A fused library kernel may keep bfloat16 intermediates |

The path comes from the set's `path` tensor (0 chunked, 1 recurrent) when present.

Synthetic sets written by `goldens::write_set` exercise all of this (`tests/common`):

- a prefill and decode pair in the oracle's layout, 4–8 heads, 70 prompt rows and 8 steps,
  with the prefill made by the chunked form. Prefill agrees to about 2e-6; decode, the
  per-step states and the conv caches match exactly;
- a layer in checkpoint naming, with its weights: exact for the fused conv rounding.

## Engine notes

- **State after a verify round.** Two ways to keep a prefix of R rows:
  - *Double buffer*, as in the source: the chain writes the new state to a second buffer.
    Keeping all R rows flips the buffers; keeping fewer replays the kept rows into it. This
    costs 272 MiB of state per slot.
  - *One buffer*, as in DESIGN §6: the chain runs with no state output
    (`StateOut::Skip`), and the commit replays the kept rows in place. Each round reads the
    state twice and writes it once. The double buffer reads and writes it once when all rows
    are kept, and twice each when fewer are. Both are measured below.

  A plain decode step (R = 1) runs the chain in place.
- **Replay inputs.** They cost 1,284 bytes per row, head and layer:
  - normalized k, f32;
  - v, bfloat16;
  - `exp(g)`, f32;
  - beta, f32.

  A lane of 8 requests × 8 rows × 34 layers therefore needs 179 MB. That is about 22 MB per
  slot, against 136 MiB for a second state. The replay reads them back exactly, which is
  what makes it bitwise.
- **The conv window** advances only at commit (`conv_shift*`). The chain reads the rows it
  needs from the projection buffer, so the projection rows of a round must stay in place
  until its commit.
- **Launch metadata.** The batch entry points read row ranges and offsets from device arrays,
  so a captured graph can be replayed with new metadata. `kernel::BatchMeta` checks the
  metadata and uploads it; `BatchMeta::device_arrays` exposes the arrays for raw launches.
- **Possible speed-up, not taken.** The state loads and stores are scalar. With 16-byte
  aligned state offsets they could be vectorized without changing any arithmetic.

## Measured throughput (RTX 4090, sm_89, 128 SMs)

`examples/kda_bench.rs`, 64 heads, synthetic inputs, 28 September 2026. The GPU was shared with
other work. Where two runs differ, both are given; they vary by up to about 10%.

**Long windows** (one layer; state in place; no replay inputs):

| Requests × rows | ms | µs per row | × 34 layers, µs per token |
|---|---:|---:|---:|
| 1 × 1,024 | 2.82 | 2.75 | 93.6 |
| 1 × 4,096 | 11.24–12.20 | 2.74–2.98 | 93–101 |
| 1 × 16,384 | 41.55–41.66 | 2.54 | 86.2–86.5 |
| 2 × 4,096 | 10.45–10.47 | 1.28 | 43.4–43.5 |
| 3 × 4,096 | 20.72–20.74 | 1.69 | 57.3–57.4 |
| Replay, 1 × 4,096: the recurrence alone | 5.49 | 1.34 | 45.6 |

The chain is latency-bound. Each row is a serial step of about 2.6 µs per head: conv, norms
and gates, 8 warp reductions and 5 block barriers. Each head occupies a whole SM for the whole
window. It does about 7 flops per state element per row (115K per head), about 7% of the SM's
f32 rate. A second request in the same launch is free until the SMs run out (2 × 64 blocks
on 128 SMs); a third runs as a second wave. The replay row isolates the delta rule, the part no
serial design can move off the critical path. Even with everything else hoisted into a
parallel pass, the serial step would stay above 1.3 µs per row.

**One decode or verify step**, recurrent part only: 34 layers, each with its own states, so
nothing sits in L2 that would not in a real step. Times are ms per step. Each step is 34
launches, including launch overhead.

| Requests | Rows | Verify chain (saves, no state out) | Decode chain (state in place) | Commit replay (keep = R) | Conv shift | State traffic of the decode chain |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1 | 0.312–0.324 | 0.563–0.567 | 0.316 | 0.005 | 503–507 GB/s |
| 1 | 2 | 0.406 | 0.647–0.648 | 0.323–0.324 | 0.005 | 440–441 GB/s |
| 1 | 4 | 0.592 | 0.821–0.822 | 0.343–0.349 | 0.005 | 347–348 GB/s |
| 1 | 8 | 0.963–0.964 | 1.164 | 0.426–0.428 | 0.005 | 245 GB/s |
| 8 | 1 | 1.602–1.609 | 3.040–3.525 | 2.512–3.130 | 0.025 | 647–751 GB/s |
| 8 | 2 | 1.967–1.977 | 3.419–3.473 | 2.538–2.539 | 0.025 | 657–667 GB/s |
| 8 | 4 | 2.718–2.724 | 4.089–4.095 | 2.592 | 0.057–0.058 | 557–558 GB/s |
| 8 | 8 | 4.207–4.805 | 5.515–5.962 | 2.965–2.968 | 0.059 | 383–414 GB/s |

A single-buffer verify round of 8 requests × 8 rows costs 4.2–4.8 + 3.0, about 7.2–7.8 ms, on
this card. A plain decode step of 8 requests costs 3.0–3.5 ms. The state traffic reaches
50–75% of the card's DRAM bandwidth (1,008 GB/s). PERFORMANCE.md §4 assumed full bandwidth
for the C16 state term, about 2.5 ms per step. Scaled by the measured efficiency, expect
nearer 4 ms on the RTX 5090.

## Prefill: is a chunked kernel needed?

**Yes.** Serially, the chain spends 86–101 µs per prompt token on the 34 KDA layers of a
single request. Even a serial kernel that moved everything but the delta rule into a parallel
pass would spend at least 46 µs (the replay row above). The coordinator's whole budget is
about 200 µs per token (DESIGN §12, question 4), and the projections alone need 70–140 µs of
it (PERFORMANCE.md §5). The chain could run 2–3 layers concurrently in the chunk × layer
wavefront; the measured 2 × 4,096 case shows two chains overlap perfectly. Even then its SM
time is fixed: 34 × 64 blocks × about 2.6 µs = about 5,700 SM·µs per token. That is 33 µs of
a fully occupied 170-SM RTX 5090, spent at under a tenth of the SMs' arithmetic rate, and
taken from the projections running beside it.

The chunked form does the same work in about 4.5M multiply-adds per 64-row chunk and head.
That is about 0.3 GFLOP per token over 34 layers, most of it as small matrix products. At
30–50% of the RTX 5090's f32 rate (about 105 TFLOPS), that is 6–10 µs per token of
arithmetic. It also parallelizes over chunks, not just heads.

**What it would look like.** This is the WY/UT form of the gated delta rule (flash-linear-attention
`fla/ops/kda`, and the reference's torch fallback). Take chunks of C = 64 rows and let
`G` be the within-chunk cumulative log decay per key channel:

1. **Prologue, parallel over rows.** Conv and SiLU, the L2 norms, decay and beta: the chain's
   per-row code without the recurrence. Then the within-chunk cumulative sum of the log
   decay. FLA fuses the gate with the cumulative sum and works in base 2.
2. **Intra-chunk, parallel over (chunk, head).**
   - Compute `A = −β_i Σ_d k_i k_j e^{G_i−G_j}` (strictly lower) and
     `M = Σ_d q_i k_j e^{G_i−G_j}` (lower, with the diagonal).
   - Forward substitution gives `T = (I − A)^-1`.
   - Form `W = T(β K e^G)`, `U = T(β V)`, `Q e^G` and `K e^{G_last − G}`.

   FLA splits each chunk into 16-row sub-chunks. With the lower bound on the gate (−5 per
   row), `e^{G_i − G_r}` against a mid-sub-chunk reference row stays within `e^{±40}`. So both
   products factor into tensor-core GEMMs, with exponentials only per row and channel instead
   of per pair. That factoring is valid only because of the lower bound, which GLM-5.3-Flash
   has.
3. **Inter-chunk, sequential over chunks, parallel over (head, value-column block).**
   - `V' = U − W S`;
   - `O = (Q e^G) S + M V'`;
   - `S ← e^{G_last} S + (K e^{G_last − G})ᵀ V'`.

   Every value column's recurrence is independent, so 64 heads × 4 blocks of 32 columns
   gives 256 blocks. Each block keeps its 128 × 32 f32 slice of the state on chip for the
   whole prompt.
4. **Epilogue.** Round `O` to bfloat16, then apply the gated RMSNorm per (row, head).

   Memory scales with the segment, not the prompt. The per-chunk products (W, U, M, `Q e^G`,
   `K e^{G_last − G}`) take about 144 KB per chunk per head, so a 2K-token segment needs about
   0.3 GB. The design's roughly 2 s prefill segments fit.

   Memory traffic matters as much as arithmetic. Step 3 re-reads W, M, `Q e^G` and
   `K e^{G_last − G}` once per value-column block. In f32 with 4 blocks that is about 16 MB per
   token over 34 layers, about 9 µs at the RTX 5090's 1.79 TB/s. The block count, the storage
   precision, and fusing steps 2 and 3 so the products stay on chip are the main levers.

**Precision, and how to check it.** FLA stores `A`, `W`, `U`, the per-chunk states and `V'` in
the input dtype (bfloat16). It casts the state to bfloat16 for its tensor-core products, and
accumulates in f32. The error that leaves is at the bfloat16 level, about 1e-3 relative, not
the recurrence's f32 level. A handwritten kernel can keep f32 storage and use f32 or 3×TF32
products. It then stays close to the `src/chunked.rs` measurement: about 1e-6 on the read-out
and 1e-5 on the state, normwise. Bit-level agreement with the serial chain is not possible,
because the chunked form regroups the sums. So, per DESIGN §6, the serial chain stays the
reference, and the chunked kernel is gated like any other numerics change:

1. Per layer, against the chain on the same inputs: normwise error of the state and read-out,
   and the bfloat16 mismatch rate of the outputs. Use random inputs across the gate range
   (tests like `tests/gpu.rs`) and captured activations. `src/chunked.rs` is the host model of
   the kernel's arithmetic.
2. Long prompts, 16K to 1M tokens: the state error must not grow with length. The decay
   contracts it, but that needs checking with slowly decaying channels.
3. The handoff to decode: the chunked prefill's final state, continued by the chain, gives the
   same greedy tokens as an all-chain prefill over a prompt corpus.
4. The model-level KL gate against the oracle, as for any numerics change (PLAN.md).

**Until the chunked kernel exists,** prefill can use the chain with the chunk × layer
wavefront. Two or three layers in flight bring the KDA term to about 30–45 µs per token on the
RTX 5090, provided the SMs it holds can be spared.

`src/chunked.rs` is the reference for the chunked kernel's arithmetic, and
`fla/ops/kda/chunk_intra.py`, `wy_fast.py` and `fla/ops/common/chunk_delta_h.py` (MIT) are the
algorithm references (see [PROVENANCE.md](PROVENANCE.md)). This crate does not implement the
chunked kernel yet.
