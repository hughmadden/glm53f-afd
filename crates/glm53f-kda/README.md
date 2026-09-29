# glm53f-kda

GLM-5.3-Flash's 34 KDA (Kimi delta attention) layers on the coordinator: the fused decode and
verify kernels and a chunked prefill kernel behind a C ABI, and an f32 CPU reference of both
forms of the recurrence.

Everything outside the recurrent core (the q/k/v, gate and beta projections before it, and
`o_proj` after it) is a GEMM and lives elsewhere. This crate takes the projection outputs and
returns the gated RMSNorm's output, which is `o_proj`'s input.

## Contents

| Path | What |
|---|---|
| `kernels/glm53f_kda.h` | The C ABI: layouts, numerics and the contract of every entry point |
| `kernels/kda.cu` | The decode and verify kernels, ported from TensorFold (see [PROVENANCE.md](PROVENANCE.md)) |
| `kernels/kda_prefill.cu` | The chunked prefill kernel (new code) |
| `kernels/parity/tensorfold_kda.cu` | The source kernels, verbatim, linked only by the parity test |
| `src/cpu.rs` | f32 reference of the recurrent path, in the kernels' order of operations, plus the reference's own formulation (`cpu::literal`) |
| `src/chunked.rs` | The chunked (WY) form: as the reference evaluates prefill (`chunked::head`, `chunked::layer`), and as the prefill kernel does (`chunked::prefill`, its host model) |
| `src/kernel.rs`, `src/device.rs` | Checked Rust wrappers and minimal device memory (feature `cuda`) |
| `src/ffi.rs`, `src/cuda.rs` | Raw bindings to the ABI and the few CUDA runtime calls used (feature `cuda`) |
| `src/goldens.rs` | Loader and checks for the oracle's golden fixtures |
| `tests/` | CPU only: `cpu_reference.rs`, `goldens.rs`, `provenance.rs`. Feature `cuda`: `gpu.rs` (decode and verify kernels), `gpu_prefill.rs` (prefill), `gpu_bf16_state.rs` (the BF16-state variants) |
| `examples/kda_bench.rs` | Throughput at the model's geometry (`--bf16-state`: the BF16-state variants) |

The ABI:

| Function | Grid | What |
|---|---|---|
| `glm53f_kda_chain` | H blocks of 1,024 threads | One layer, one request, R rows from the committed state: conv and SiLU, L2 norms, decay, beta, delta rule, gated RMSNorm. Writes the outputs, optionally the state after the last row, and optionally the replay inputs |
| `glm53f_kda_chain_batch` | H × B | The same for B requests in one launch: each owns a row range and its own state and conv window |
| `glm53f_kda_replay` | H | The state after the first `keep` saved rows of a chain |
| `glm53f_kda_replay_layers` | L·H | The same for every layer of a request |
| `glm53f_kda_replay_batch` | L·H × B | Every layer of every request, each with its own `keep`: the commit after a verify round |
| `glm53f_kda_conv_shift`, `_batch` | C/256 (× L × B) | Advance conv windows past the kept rows, in place |
| `glm53f_kda_prefill` | per pass: chunks × H, then H × value blocks | One layer, one request, a prompt segment of any length through the chunked form: the chain's outputs and final state, from any committed state and conv window. Advances the conv window past the rows |
| `glm53f_kda_prefill_batch` | the same × B | The same for B requests of different lengths, each with its own state and conv window |
| `glm53f_kda_prefill_workspace_bytes` | — | The workspace for a given number of rows per pass |
| `glm53f_kda_chain_batch_bf16state`, `glm53f_kda_replay_batch_bf16state`, `glm53f_kda_prefill_batch_bf16state` | as above | The batch entry points with the states stored in BF16 ([BF16 state](#bf16-state-decision-d8)) |

The chain and the replays may update states in place (`state_out == state_in`) or not write
them at all. The prefill always writes its state, in place or not.

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

Run the GPU tests with `--release`: the prefill tests compare against the chain over 16K-row
segments and against an f64 replay of 8K rows. The GPU tests skip, with a message, when no
device is present. The golden tests skip when `oracle/goldens/` holds no set. Set
`GLM53F_GOLDENS` to use another directory.

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

The prefill kernel computes the same per-row steps with the chain's bits, and the delta rule
in the chunked form. It agrees with the chain to f32 rounding, not bit for bit (see
[Prefill](#prefill)).

**Where this differs from the reference implementation** (`transformers` `glm5_next`):

1. **Conv rounding.** The kernels round once after conv and SiLU, as the fused `causal_conv1d`
   kernel does. The reference's pure-torch fallback rounds the conv output to bfloat16 before
   the SiLU as well. The two disagree on about a quarter of the conv outputs, by one or two
   bfloat16 ulps. That moves a layer's state by about 0.5% and changes most bfloat16 outputs
   by an ulp or more. `cpu::Rounding` provides both (`Fused`, `Unfused`), and `F32`, which
   rounds nowhere, as the oracle's f32 goldens do. The oracle's native bfloat16 set ran the
   pure-torch path, which is the unfused rounding (`oracle/README.md`).
2. **Norms.** The kernels multiply by a reciprocal square root where the reference divides by
   the norm, and use `1/sqrt` where it uses `rsqrt`. This is ulp-level; in the tests it has not
   changed a single bfloat16 output.
3. **Layouts.** The state is stored `[H][DV][DK]`, the transpose of the reference's
   `[H][DK][DV]` (`cpu::transpose_state` converts). The conv window is `[3][C]`, time-major.
   The reference caches `[C][4]` and uses only the last three columns.
4. **Scope.** The kernels assume what GLM-5.3-Flash uses: `DK = DV = 128`, 4 taps, no conv
   bias, the lower-bound form of the forget gate (not the softplus form), bfloat16 conv and norm
   weights, and f32 `A_log` and `dt_bias`. This matches the official checkpoint. The prefill
   also requires `lower` in [−5.8, 0].

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

**Relation to the chunked form.** In exact arithmetic the chunked (WY) form equals the
recurrence. `src/chunked.rs` holds it twice, in f32:

- `chunked::head` and `chunked::layer` follow the reference's `chunk_kimi_delta_attention`
  step by step, with decays from cumulative log sums. Against the recurrence: ≤ 1.1e-6 on the
  read-out and ≤ 7.7e-6 on the state (normwise, T up to 150, chunks of 64 and 16). Rounded to
  bfloat16, about 0.2% of read-out elements differ, some by many ulps where the value is tiny.
- `chunked::prefill` is the prefill kernel's form. Against the chain: ≤ 1.6e-7 on the state;
  at most 7 of 25,600 bfloat16 outputs differ (T up to 100, gates across the range and at
  the strongest).

## Golden fixtures

`tests/goldens.rs` checks every set under `oracle/goldens/` (see `oracle/README.md`), or under
`GLM53F_GOLDENS`, and skips cleanly when there is none. A set is a `manifest.json` (`tensors`:
`{name: {file, dtype, shape, sha256}}`, plus `notes`) next to raw little-endian `.bin` files.
Every digest is checked first.

**Names.** The oracle names its sets `layerNN-prefill` and `layerNN-decode`, and their tensors
`prefill.kda.q`, `decode.kda.core_out` and so on. Decode tensors are stacked over the steps.
The layer's KDA weights come as `weights.self_attn.*`. A tensor is a KDA tensor when its
layer is a KDA layer. The layer comes from a component of the name (`layers.4.`, `L04.`) or,
failing that, from the set's name. Its role is what is left after the phase prefix, the layer
and the module names (`weights`, `kda`, `self_attn`, `linear_attn`, `forget_gate`). Per-step
names (`decode.sN.`) are skipped. `goldens::ROLES` lists the roles and the names each accepts,
so checkpoint names such as `model.language_model.layers.4.self_attn.forget_gate.A_log` work
too.

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
  `o_norm.weight`), against `norm_out` and the state after the rows.
  - An f32 golden (the oracle's primary contract) must match the reference evaluated in f32
    with no rounding at all (`Rounding::F32`). The kernels' bfloat16 roundings are reported
    beside it (`Weight::Informational`).
  - A bfloat16 golden must match one of the two conv roundings (`Weight::OneOf`).
  - `goldens::verdict` applies these weights.
- **Device recurrence** (`tests/gpu.rs`, feature `cuda`): the replay kernel on the golden's
  k, v, g and beta, against the golden state (within 1e-2, since the kernels keep v in
  bfloat16) and against the CPU replay (bit for bit).
- **Kernels on the real layers** (`tests/gpu_prefill.rs`, `kernels_on_oracle_layers`): the
  chain and the prefill on each layer set's projections (rounded to bfloat16, as the kernels
  take them), weights and initial state.

**Tolerances**, normwise against the largest golden value:

| Golden | Bound | Why |
|---|---:|---|
| f32 (the oracle's primary contract) | 1e-4 | The same f32 recurrence, or its chunked form: rounding only (about 1e-5, see above) |
| bfloat16, per-token path | 1e-2 | A bfloat16 rounding flip |
| bfloat16, chunked path | 5e-2 | A fused library kernel may keep bfloat16 intermediates |

The path comes from the set's `path` tensor (0 chunked, 1 recurrent) when present.

**Results on the oracle's sets** (layers 0 and 4; prefill sets of 33 rows, decode sets of 8
steps; 28 September 2026):

| Check | Measured |
|---|---|
| Core: outputs | ≤ 7.5e-9 absolute (largest values 1.7e-2 to 3.7e-2) |
| Core: state after the rows | ≤ 1.8e-7 absolute (largest values 0.23 to 0.94) |
| Core: per-step states of 4 heads | ≤ 9.3e-9 absolute |
| Conv caches | exact |
| Core, native bfloat16 set (41 rows) | ≤ 3.8e-6 absolute; 76 of 335,872 outputs differ per layer |
| Layer, f32 evaluation: outputs | ≤ 9.4e-8 absolute (4e-7 relative) |
| Layer, f32 evaluation: state | ≤ 1.8e-7 absolute |
| Layer, the kernels' bfloat16 roundings (reported only) | outputs 3.4e-3 to 3.9e-3 relative, state 2.9e-4 to 1.3e-3 relative: the bfloat16 level |
| Device: prefill against the chain | state 2.4e-8 to 8.1e-8, outputs ≤ 1.3e-4 relative |
| Device: chain against the CPU reference | state ≤ 8.6e-6, outputs ≤ 5.3e-4 relative |
| Device: chain against the reference's f32 outputs | state 7.1e-4 to 2.0e-3, outputs 2.9e-3 to 5.1e-3 relative |

On one real layer the chain's state differs from the CPU reference's by 8.6e-6. The device's
and the host's `expf` differ by an ulp, which on real data can flip a bfloat16 conv output;
the state then carries the difference.

Synthetic sets written by `goldens::write_set` exercise all of this (`tests/common`):

- a prefill and decode pair in the oracle's layout, 4–8 heads, 70 prompt rows and 8 steps,
  with the prefill made by the chunked form. Prefill agrees to about 2e-6; decode, the
  per-step states and the conv caches match exactly;
- a layer in checkpoint naming, with its weights: exact for the fused conv rounding.

## Prefill

`glm53f_kda_prefill` and `glm53f_kda_prefill_batch` (`kernels/kda_prefill.cu`) run a prompt
segment of any length through the chunked form of the delta rule. They take the chain's inputs
and return its outputs and final state. A segment starts from any committed state and conv
window, such as a restored snapshot, and leaves the conv window advanced, so the next segment
or the chain's decode continues from it. Batched requests may have different lengths,
including zero.

### Why a chunked kernel

The chain is serial in the rows. It spends 2.5–2.9 µs per row on a layer: 86–98 µs per prompt
token over the 34 KDA layers of one request (measured below). Even a serial kernel that moved
everything but the delta rule into a parallel pass would spend at least 46 µs (the replay
row). The coordinator's whole budget is about 200 µs per token (DESIGN §12, question 4), and
the projections alone need 70–140 µs of it (PERFORMANCE.md §5). The chunked form does about
the same arithmetic as the chain, about 115K flops per row and head, but in parallel over
chunks and value columns.

### How it works

Per chunk of 16 rows (fewer at the end of a segment):

1. **Prologue.** The chain's own per-row arithmetic, operation for operation and compiled with
   the same flags: conv and SiLU (one bfloat16 rounding), the q and k L2 norms, the decay
   multipliers `d`, and beta. These have the chain's bits.
2. **Decay products.** `Lf_i = d_1 ⋯ d_i`, from the chunk's first row. Then
   `A = k ⊙ Lf`, `A' = q ⊙ Lf` and `B = k ⊘ Lf`, so that `e^{G_i − G_j} = Lf_i / Lf_j`.
3. **Intra-chunk products.** `L[i][j] = β_i A_i·B_j` for `j < i`, and `M[i][j] = A'_i·B_j`
   for `j ≤ i`. Forward substitution gives `T' = (I + L)^-1 diag(β)`. Then `W = d_0 ⊙ T'A`,
   `U = T'V` and `Qg = d_0 ⊙ A'`.
4. **Inter-chunk step,** from the state `S` entering the chunk: `D = U − W S`;
   `Y = Qg S + M D`; `S ← (d_0 Lf_last) ⊙ S + Lf_last ⊙ (Bᵀ D)`.
5. **Epilogue.** `y = bf16(Y)`, then the chain's gated RMSNorm.

The kernels:

- `intra_kernel` runs steps 1–3 with one block per (chunk, head, request). Nothing there needs
  the state, so every chunk of a pass runs in parallel. It writes 34,816 bytes per chunk and
  head to a workspace.
- `inter_kernel` runs steps 4–5 with one block per (head, value-column block, request). Each
  block keeps its slice of the state in registers across the pass's chunks. It stages each
  chunk's workspace into shared memory one chunk ahead (`cp.async`). With one block per head
  it also applies the gated norm. With 2 or 4 (`value_blocks`), a separate `norm_kernel` does,
  since the norm needs all 128 value columns of a row.
- The host repeats the two over sub-segments as long as the workspace allows, then
  `conv_advance_kernel` advances the conv window.

**Design choices**, and where they depart from the design note that preceded the kernel:

- **16-row chunks, not 64.** Decays are products of the chain's own multipliers, taken from the
  chunk's first row. With the −5 bound, 16 rows keep every product inside f32's normal range
  (`e^±75`), so there are no per-pair exponentials and no sub-chunk reference rows. A 64-row
  chunk would need one or the other. Exponentials of cumulative log sums, as the reference
  and flash-linear-attention use them, lose about `ulp(G)`: some 3e-5 relative at
  `G ≈ −300`. The launcher requires `lower` in [−5.8, 0], so that 15 rows of decay stay
  normal.
- **Every decay product is rounded once.** The running product carries its own rounding error
  (an exact two-product with `fmaf`), and each `Lf_i` is rounded once from it. Products of
  multipliers just below 1 tend to round the same way, and the chunk's total decay reaches
  the state at every chunk. On slowly decaying channels a plain running product put the
  state's error at 3.9e-6 against exact arithmetic, 2.5 times the chain's; now it is 3.8e-7,
  a quarter of the chain's (below).
- **f32 on the CUDA cores, not 3×TF32.** On the RTX 4090 and 5090 dense TF32 tensor
  throughput equals the f32 rate, so 3×TF32 peaks at a third of it. The kernel runs at
  13–16% of the f32 peak (below), limited by shared-memory loads rather than by
  multiply-adds, so tensor cores would at best match it. Products use explicit `fmaf`.
- **Two kernels and a workspace, not one fused kernel.** A fused kernel, one block per head
  doing every step chunk by chunk, measured about 1.5 µs per row, because the intra-chunk
  work then sits on each head's serial path. Split, it runs in parallel over the chunks. The
  workspace is written once and read once, about 280 KB per row and layer (9.5 MB per token
  over 34 layers). As measured below, larger passes are faster even when the workspace spills
  from L2 to DRAM.

### Accuracy

Against the chain on the same inputs (`tests/gpu_prefill.rs`, RTX 4090). Errors are
max |difference|, and relative to the largest reference value:

| Check | Measured | Bound |
|---|---|---|
| 1, 16, 17, 64, 1,000, 1,024, 4,096 and 16,384 rows; 64 heads; 1, 2 and 4 value blocks (16,384: 1 only) | State ≤ 1.8e-7 (≤ 1.4e-7 relative). Outputs: max \|difference\| 1.6e-2 (1.8e-3 relative); at most 0.013% differ (15,833 of 134M at 16,384 rows). Conv windows exact | state 1e-5 relative; outputs 2^-7 relative, ≤ 0.1% differ |
| Against its host model `chunked::prefill` (5, 16 and 70 rows) | state ≤ 1.5e-7; at most 5 outputs differ | state 1e-6 |
| A segment continued from a committed state: 600 then 400 rows | state 1.0e-7 against the chain over 1,000, 1.5e-7 against one prefill of 1,000; conv window exact | 1e-5 |
| The handoff: prefill 1,000 rows, then 8 chain decode steps, against the chain over 1,008 | state 2.9e-8; all 65,536 decode outputs identical; conv window exact | 1e-5 |
| Batched against per request (16 heads; 5 requests of 37, 0, 100, 16 and 1 rows, states in shuffled slots; 1 and 2 value blocks) | bit for bit: outputs, states, conv windows | exact |
| Workspace size (16, 48, 160 or all 300 rows per pass) | bit for bit | exact |
| 64K rows (8 heads) in 8 segments of 8K, each continuing from its own state; gates across the range | state 1.3e-7 to 2.3e-7 at every segment, not growing | 1e-5, and the last ≤ 4 × the first |
| The same with every channel decaying slowly (multipliers ≥ 0.9993) | state 1.2e-6 to 2.5e-6, not growing; 0.07% of outputs differ | as above |
| Against exact arithmetic: an f64 replay of the chain's own replay inputs, 8 heads, 8K rows | gates across the range: chain 9.1e-8, prefill 1.3e-7. Slow decay: chain 1.6e-6, prefill 3.8e-7 | prefill ≤ 1e-6 and ≤ 2 × the chain + 1e-7 |
| The oracle's real layers (above) | state 2.4e-8 to 8.1e-8 | 1e-5 |

Bad arguments are rejected: 3 value blocks, no state output, a lower bound of −6, too small a
workspace.

The target was ≲ 1e-5 against the chain. The state agrees to f32 rounding: about 1e-7, and
about 2e-6 on slowly decaying channels. There the difference is mostly the chain's own
rounding; against exact arithmetic the prefill is four times closer. Outputs differ only
through bfloat16 roundings that an f32 difference flips. For comparison,
flash-linear-attention's chunked kernel stores its intermediates in bfloat16, which leaves
errors at the bfloat16 level, about 1e-3 (not measured here).

### Throughput (RTX 4090, sm_89, 128 SMs)

`examples/kda_bench.rs`, 64 heads, one layer, state in place, conv window advanced, 28
September 2026. The GPU was shared; each figure is the faster of two runs, which agree within
1% unless another job interfered. µs per row is per row of the whole batch; µs per token
multiplies it by the 34 layers.

At 1,024 rows of every request per pass (143 MB of workspace per request), µs per row (µs per
token):

| Requests × rows | 1 value block | 2 | 4 |
|---|---:|---:|---:|
| 1 × 1,024 | 0.848 (28.8) | **0.667 (22.7)** | 0.843 (28.7) |
| 1 × 4,096 | 0.843 (28.6) | **0.668 (22.7)** | 0.842 (28.6) |
| 1 × 16,384 | 0.844 (28.7) | **0.668 (22.7)** | 0.844 (28.7) |
| 2 × 4,096 | **0.564 (19.2)** | 0.668 (22.7) | |
| 4 × 4,096 | **0.561 (19.1)** | | |
| 8 × 1,024 | **0.558 (19.0)** | | |

Bold is what `value_blocks = 0` chooses on this card: the most blocks per head that still run
as one wave (`inter_kernel` fits one block per SM). One request's 64 heads take 2 blocks each
(128 blocks); two or more requests take 1.

One request's prefill is 3.8–4.3 times faster than the chain (22.7 against 86–98 µs per
token). Batched, it reaches 19 µs per token, against the chain's 43 µs for 2 × 4,096 rows.
Counted from the algorithm, the work is about 250 Mflop per token over 34 layers: 11–13
Tflop/s of useful f32 work, 13–16% of the card's peak.

**Rows per pass** (the workspace size), µs per row, with the workspace in brackets:

| Requests × rows, value blocks | 64 | 256 | 1,024 | 2,048 | 4,096 |
|---|---:|---:|---:|---:|---:|
| 1 × 4,096, 2 | 1.014 (8.9 MB) | 0.716 (36 MB) | 0.668 (143 MB) | | 0.664 (570 MB) |
| 4 × 4,096, 1 | | 0.656 (143 MB) | 0.561 (570 MB) | 0.548 (1.1 GB) | |
| 8 × 1,024, 1 | 1.018 (71 MB) | 0.641 (285 MB) | 0.558 (1.1 GB) | | |

Each pass costs `inter_kernel` about 22 µs beyond its work per row (fitted from the profile
below): loading and storing the state, and starting the pipeline. So larger passes are
faster, with diminishing returns: 256 rows per request are 7–17% slower than 1,024, and
1,024 are within 2.5% of the largest tried. On this card, keeping the workspace inside the
72 MB L2 does not pay for the extra passes.

**Where the time goes** (`nsys` kernel durations, 5 repetitions; the profiler adds 4–8% to
the totals):

| Case | `intra_kernel` | `inter_kernel` | `norm_kernel` | Total, µs per row |
|---|---:|---:|---:|---:|
| 1 × 4,096, 2 blocks, 256 rows per pass | 58 µs per pass, 0.227 µs per row | 124 µs per pass, 0.485 | 0.055 | 0.775 |
| 1 × 4,096, 2 blocks, 1,024 rows per pass | 240 µs per pass, 0.235 | 430 µs per pass, 0.420 | 0.051 | 0.707 |
| 2 × 4,096, 1 block, 1,024 rows per pass | 483 µs per pass, 0.236 | 720 µs per pass, 0.352 | fused | 0.589 |

`conv_advance_kernel` takes 2 µs per segment. `inter_kernel` holds 60% of the time and issues
24–28 multiply-adds per clock per SM, about a fifth of the peak. Instrumented with clock
counters during development (the GPU's performance counters were not available), it waited
mostly on shared-memory loads.

### RTX 5090: a projection, not a measurement

No RTX 5090 was available. The projection scales the profiled RTX 4090 figures, and assumes
this RTX 4090's 2.58 GHz and the RTX 5090's 2.41 GHz boost clock:

- `intra_kernel` spreads over all SMs, so it scales with SMs × clock: 128 SMs at 2.58 GHz
  against 170 at 2.41 GHz, a factor of 0.81. (If DRAM limits it, the factor is 0.56.)
- `inter_kernel` fits one block per SM on both cards. One request is 128 blocks, a single
  wave on both, so the RTX 5090's extra SMs sit idle and only the clock counts: a factor of
  1.07. Batches of 4 and 8 requests (256 and 512 blocks) also take 2 and 4 waves on both
  cards. 5 requests (320 blocks) take 3 waves on the RTX 4090 and 2 on the RTX 5090.
- `norm_kernel` streams the outputs, so DRAM bandwidth counts: 1.01 against 1.79 TB/s, a
  factor of 0.56.

| Case | RTX 4090, measured | RTX 5090, projected |
|---|---:|---:|
| 1 request | 0.668 µs per row, 22.7 µs per token | about 0.63 µs per row, **21 µs per token** |
| 2 requests | 0.564, 19.2 | about 0.54, **18.5** |
| 4 or 8 requests | 0.558–0.561, 19.0–19.1 | about 0.54, **18** |
| 5 requests | not measured | about 0.47, **16** |

The RTX 5090 gains little here, because the inter-chunk pass is a single wave of
latency-bound blocks. The next steps below would put its idle SMs to use.

### Using it in the engine

- **Workspace.** 2.2 MB per 16 rows of a 64-head request per pass
  (`glm53f_kda_prefill_workspace_bytes`, `kernel::PrefillWorkspace`). Use at least 256 rows
  per request per pass, and 1,024 where 143 MB per request can be spared. A workspace sized
  for a batch serves any smaller batch, with proportionally more rows per pass. Layers run in
  stream order, so one workspace serves every layer.
- **Value blocks.** Pass 0 to let the launcher choose for the GPU and the batch. The choice
  changes the order of some sums, so results differ at f32 rounding between choices. Pass 1, 2
  or 4 where results must not depend on the batch composition or the GPU. For a given value
  block count, results do not depend on batching or on the workspace size.
- **States and conv windows.** The prefill writes the state after the segment (in place or to
  another buffer) and advances the conv window in place. The chain, or another segment,
  continues from both. The prefill writes no replay inputs: a prompt segment is committed
  whole.
- **Launches.** Each layer takes 3 launches per segment (4 with 2 or 4 value blocks) when one
  pass covers the segment, and 2 more per extra pass. Capture them in a graph.
- **Still to check at the model level** (DESIGN §6). The kernel-level checks above cover
  per-layer agreement with the chain, long prompts and the handoff to decode. Still to run:
  greedy tokens after a chunked prefill against an all-chain prefill over a prompt corpus,
  and the KL gate against the oracle (PLAN.md). *(29 September 2026: the KL gate has run on
  the target hardware. The chunked prefill alone raised the mean KL by 0.0021 nats and failed
  the gate's margin; with W8A16 projections the pair lowered it, but 25 windows could not yet
  show it non-inferior. It stays opt-in, `glm53f-serve --kda-chunked-prefill`;
  `docs/KL-GATE.md` §6b.)* *(Later on 29 September: on the 125-window panel the pair passed,
  upper bound +0.0002 nats against the 0.002 margin, and prefilled 21-28% faster, so it is
  `glm53f-serve`'s default; `--kda-chain-prefill --prefill-w8a8` restores the chain and W8A8;
  `docs/KL-GATE.md` §6d.)*

### Open, and possible next steps

- **Overlap the two passes.** With one request, `inter_kernel` leaves 42 of the RTX 5090's
  SMs idle. Running the next sub-segment's `intra_kernel` beside it (on a second stream, or by
  interleaving layers of different requests) would hide most of the intra-chunk pass: about
  15 µs per token for one request on the RTX 5090 (projection).
- **Fuse the norm for 2 value blocks** (7% of the time). Thread block clusters, available on
  sm_120 but not on sm_89, would let a head's two blocks share the sum of squares.
- **Cut the fixed cost per pass** (about 22 µs, 5% of `inter_kernel`'s time at 1,024 rows per
  pass).
- **TF32 for the output products only** (`Qg S` and `M D` feed a bfloat16 output). The state
  update must stay f32.
- **Wave quantization on the RTX 5090.** A batch whose block count just exceeds a multiple of
  170 (8 requests: 512 blocks) wastes most of a wave. The engine can group prefill requests
  to avoid it.

## BF16 state (decision D8)

The `_bf16state` entry points store the recurrent state in BF16 (68 MiB per request over 34
layers instead of 136) and compute exactly as the f32 kernels do, from the state widened to f32.
They are the f32 kernels' own code, instantiated for a second state type (`StateIO` in
`kernels/kda.cu` and `kda_prefill.cu`); the f32 instantiation's arithmetic is unchanged (the
parity test against the source kernels still passes bit for bit). Checked wrappers:
`ChainBatch::launch_bf16_state`, `replay_batch_bf16_state`, `PrefillBatch::launch_bf16_state`;
CPU models: `cpu::chain_bf16_state`, `cpu::replay_bf16_state`. The engine turns them on with
`glm53f-serve --kda-state-bf16`, the default since 29 September 2026, when they passed the KL
gate (`docs/KL-GATE.md` §6b); `--kda-state-f32` turns them off.

**Where the state is rounded** (round to nearest even):

- **The chain and the replay: after every row**, after that row's read-out. A window of R rows
  therefore gives the bits of R serial single-row calls, each of which stores its state in BF16,
  and a verify round committed at k rows gives the bits of k serial decode steps, as with an f32
  state. With one row the outputs equal the f32 chain's and the state is its state rounded once.
- **The chunked prefill: after every chunk of 16 rows** (chunks count from each request's first
  row). The state is never materialized per row there. Stores between sub-segments are then
  exact, so the results still do not depend on the workspace size. The prefill rounds 16 times
  less often than the chain: the two agree to BF16 rounding, not bit for bit.

**Verified** (`tests/gpu_bf16_state.rs`, RTX 4090, 64 heads unless stated):

| Check | Result |
|---|---|
| A window of R = 1, 2, 5, 8 rows against R serial single-row calls (with the conv shift between them) | outputs, state, replay inputs and conv window bit for bit |
| A verify window (no state written) | the window's outputs bit for bit; the state untouched |
| The replay of every prefix k ≤ R | the state after k serial steps bit for bit, and `cpu::replay_bf16_state` bit for bit |
| One row against the f32-state chain | outputs and replay inputs bit for bit; the state the f32 state rounded once |
| The chain against `cpu::chain_bf16_state` (1, 8, 40 rows) | state ≤ 9.3e-4 normwise, outputs ≤ 4.7e-4 (≤ 26 of 327,680 BF16 outputs differ): the device's and the host's `expf` |
| Batched against per request (16 heads; 5 requests of 3, 0, 8, 1, 5 rows in shuffled slots; the replay with a different `keep` each; the prefill with 37, 0, 100, 16, 1 rows) | bit for bit |
| The prefill with 16, 48 and 160 rows per pass against one pass (1,000 rows; 1, 2 and 4 value blocks) | bit for bit |
| The prefill against the chain, both BF16 (1,000 rows) | state 3.6e-3, outputs 6.5e-3 normwise |

**Drift.** Rounding the state every row is the risk: when a channel's decay multiplier lies
within half a BF16 ulp of 1 (above about 0.998), rounding undoes the step's decay unless the
delta-rule update moves the value, and the state then forgets more slowly than it should. The same
data as `accuracy_against_exact_arithmetic` and `long_prompt_drift` (8 heads; the f64 replay of
the replay inputs, which do not depend on the state, is exact arithmetic):

| 8,192 rows, state against exact arithmetic (max normwise, relative RMS) | Gates across the range | Every multiplier ≥ 0.9993 |
|---|---:|---:|
| Chain, f32 state | 1.0e-7, 5.6e-8 | 1.4e-6, 6.0e-7 |
| Chain, BF16 state (every row) | 5.3e-3, 2.1e-3 | **7.8e-2, 2.6e-2** |
| Prefill, f32 state | 1.2e-7, 6.6e-8 | 3.4e-7, 2.1e-7 |
| Prefill, BF16 state (every 16 rows) | 2.9e-3, 1.7e-3 | 1.2e-2, 6.8e-3 |
| Outputs against the f32 chain, relative RMS (BF16 outputs that differ) | chain 1.9e-3 (25%), prefill 6.0e-4 (3.4%) | chain 2.3e-2 (92%), prefill 6.6e-3 (72%) |

Over 64K rows in eight segments of 8K, each continuing from its own state, the BF16 chain's state
stays 1.8e-3 to 2.1e-3 (relative RMS) from the f32 chain's at every segment with gates across the
range, and 2.6e-2 with slow decay: bounded, not growing, but the slowly decaying channels carry a
few percent. On the oracle's real prompt 5.3–6.1% of the (row, key channel) decay multipliers of
layers 0 and 4 exceed 0.998 and 3.3–3.8% exceed 0.9993. The KL gate on the target hardware
decides. For comparison, an FP16 state (the same bytes, three more mantissa bits; not built) drifts
7.7 times less on the same data with the CPU model: 2.7e-4 and 3.3e-3.

**Speed** (`kda_bench --bf16-state`, section 2: 34 layers, 64 heads, ms per step, best of two
runs on a shared RTX 4090):

| Requests × rows | Verify chain | Decode chain | Commit replay |
|---|---:|---:|---:|
| 1 × 1 | 0.350 → 0.237 | 0.598 → 0.343 | 0.346 → 0.139 |
| 1 × 8 | 1.001 → 0.975 | 1.262 → 1.023 | 0.481 → 0.335 |
| 8 × 1 | 1.791 → 1.194 | 3.336 → 2.022 | 2.836 → 1.393 |
| 8 × 8 | 4.839 → 4.151 | 6.322 → 4.850 | 3.394 → 2.561 |

## Engine notes (decode and verify)

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

## Measured throughput of the chain (RTX 4090, sm_89, 128 SMs)

`examples/kda_bench.rs`, 64 heads, synthetic inputs, 28 September 2026. The GPU was shared with
other work; the ranges span four runs.

**Long windows** (one layer; state in place; no replay inputs):

| Requests × rows | µs per row | × 34 layers, µs per token |
|---|---:|---:|
| 1 × 1,024 | 2.55–2.75 | 86.9–93.6 |
| 1 × 4,096 | 2.56–2.75 | 87.0–93.4 |
| 1 × 16,384 | 2.54–2.87 | 86.2–97.6 |
| 2 × 4,096 | 1.28 | 43.3–43.5 |
| 3 × 4,096 | 1.69–1.87 | 57.3–63.5 |
| Replay, 1 × 4,096: the recurrence alone | 1.34–1.37 | 45.5–46.7 |

The chain is latency-bound. Each row is a serial step of about 2.6 µs per head: conv, norms
and gates, 8 warp reductions and 5 block barriers. Each head occupies a whole SM for the whole
window. It does about 7 flops per state element per row (115K per head), about 7% of the SM's
f32 rate. A second request in the same launch is free until the SMs run out (2 × 64 blocks
on 128 SMs); a third runs as a second wave. The replay row isolates the delta rule, the part no
serial design can move off the critical path.

**One decode or verify step**, recurrent part only: 34 layers, each with its own states, so
nothing sits in L2 that would not in a real step. Times are ms per step. Each step is 34
launches, including launch overhead. Measured before the prefill work, on the same kernels;
later runs agree within a few percent except where another job interfered.

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
