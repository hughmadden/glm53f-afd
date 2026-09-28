# glm53f-rank

The expert rank of glm53f-afd. Each of the four DGX Sparks runs one
`glm53f-rank` daemon. It holds a quarter of every routed expert of
GLM-5.3-Flash: intermediate channels `[512 r, 512 r + 512)` of each of the 288
experts, in all 42 MoE layers (3–44). The weights are stored as EXL3 K4, the
`tr3-4bpw` checkpoint's format. For every MoE layer of every forward pass the
coordinator sends the routed rows. For each row, the rank returns the BF16 sum
over the row's 8 experts of `gate_weight × expert_partial`. The coordinator
adds the four ranks; for prefill-sized requests the ranks can add among
themselves instead, and each returns a quarter of the rows
([Prefill reduce-scatter](#prefill-reduce-scatter)).

**Status (28 September 2026):**

- The EXL3 decoder matches TensorFold's reference bit for bit.
- The TP4 split is exact, on synthetic experts at full size and on real
  experts from layers 3 and 4 of the published checkpoint.
- The CUDA kernel matches the CPU reference stage by stage, on the development
  GPU (an RTX 4090, `sm_89`), on synthetic layers and on a real one.
- The four ranks' shares of real layers 3 and 4, fed the reference oracle's
  MoE inputs as FP8 wire rows, reproduce its routed-expert output. The cosine
  is at least 0.990 and the relative RMS error 5–11%, the 4-bit quantization
  error. This holds on the GPU and on the CPU.
- The daemon serves end to end over TCP.
- The prefill reduce-scatter runs end to end: four daemons with their peer
  mesh, on the development GPU over loopback TCP, return the oracle's layers 3
  and 4 row-sharded, within the derived bound of the four-plane return
  ([Prefill reduce-scatter](#prefill-reduce-scatter)).
- The forward uses it (glm53f-forward's `RemoteExperts`, from
  `GLM53F_ROW_SHARDED_MIN_ROWS` rows): two-lane prefill of layers 0-4 on the
  four daemons over loopback picks the four-plane path's tokens on every
  decided row, and four lanes give the bits of four passes of the same rows
  (glm53f-forward `tests/remote_experts.rs`).
- The expert kernel has a large-M family for prefill and a fused epilogue,
  reduce and plan (28 September 2026): 1.16–1.32× on the development GPU at
  512–4,096 rows and at one or two rows, the same bits
  ([Measured on the development GPU](#measured-on-the-development-gpu)).
- The split kernels, which serve every decode and verify window, have
  schedule options for GB10's memory (29 September 2026): the trellis words
  loaded up to 4 k tiles ahead (`pf`), evict-first (`l2`), a block order that
  keeps partial sums in L2 (`ord`), and the down kernel started before the
  gate/up kernel ends (`pdl`, `sm_90` and later). Fewer fixed costs per call
  and the new block order make the defaults 1.04–1.12× faster on the
  development GPU at 1–64 rows, the same bits
  ([Decode sizes](#decode-sizes-29-september-2026)).
- Not yet done:
  - the new kernels on a Spark (`sm_121`; the earlier kernel's GB10 numbers
    are in `docs/PERFORMANCE.md`), and the split kernels' options swept there
    ([On a Spark](#on-a-spark));
  - the RDMA path, for the coordinator's link and the peer mesh (built, but
    untested without a fabric; the receive slots that queue the prefill
    lanes' requests are tested on the CPU against a model of the receive
    queue).
- **Update (29 September 2026).** Both items above have since run on the
  target hardware, except the peer mesh over RDMA:
  - the kernels run on four Sparks, the kernel test passes there unchanged,
    and the split kernels' options were swept there; `l2=2,pf=2,pdl=2` is now
    the default of GB10 builds ([Measured on GB10](#measured-on-gb10));
  - the coordinator's link runs over RDMA, with the whole model served by four
    ranks (`docs/PERFORMANCE.md` §0);
  - the peer mesh over RDMA is still built but untested (the reduce-scatter
    stays off; [Open issues](#open-issues), item 5).

Contents:

1. [Build and test](#build-and-test)
2. [The kernel boundary](#the-kernel-boundary)
3. [EXL3 K4](#exl3-k4)
4. [The TP4 split](#the-tp4-split)
5. [The kernel](#the-kernel)
6. [Measured on the development GPU](#measured-on-the-development-gpu)
7. [On a Spark](#on-a-spark)
8. [The rank directory](#the-rank-directory)
9. [The daemon](#the-daemon)
10. [Prefill reduce-scatter](#prefill-reduce-scatter)
11. [Open issues](#open-issues)

## Build and test

```sh
# CPU gate: no CUDA toolkit or GPU needed.
cargo test -p glm53f-rank --release

# The kernels, on a CUDA GPU (default target sm_89; use at most a few GB).
cargo test -p glm53f-rank --release --features cuda -- --test-threads=1
cargo run  -p glm53f-rank --release --features cuda --example exl3_bench -- [--dir <rank-dir> [--layer 3,4]] [--sweep] [--min]

# Real experts (the tests skip without these; see tests/real_experts.rs).
GLM53F_EXL3_DIR=<copy holding layers 3-4 of the EXL3 checkpoint> \
GLM53F_FP8_DIR=<copy of the same experts from the official FP8 checkpoint> \
  cargo test -p glm53f-rank --release [--features cuda] --test real_experts -- --test-threads=1
```

**Build variables:**

- `GLM53F_CUDA_ARCH`: the target GPU architecture; `sm_89` by default, `sm_121`
  for a DGX Spark. The architecture is baked into the binary, and the daemon
  refuses a device of another architecture.
- `GLM53F_NVCC`: the `nvcc` to use.
- `GLM53F_CUDA_LIB`: the directory holding `libcudart`.

**Features:**

- `cuda` builds the kernels.
- `rdma` builds the RDMA shim (`glm53f-rdma`), for the coordinator's link and
  the peer mesh. A Spark's binary uses `--features cuda,rdma`.

The workspace takes no external crates.

## The kernel boundary

The boundary keeps the shape of mimo26f-afd's B1 path. It is the
`kernel::ExpertKernel` trait:

- `prepare_layer(image)` takes one layer's resident weights, laid out as
  described in [The rank directory](#the-rank-directory).
- `ffn(rows, ids, weights, out)` takes three things:
  - the wire rows as received: FP8 E4M3 with UE8M0 scales per 32 values, in
    separate arrays or read in place from a request frame;
  - the top-8 expert ids per row;
  - FP32 gate weights per row.

  It writes one BF16 row per input row, each already summed over the 8 routes.

There are two backends:

- `exl3_cuda::CudaKernel`, the kernels in `kernels/exl3_rank.cu`;
- `kernel::CpuKernel`, the kernel-order CPU reference, used for the CPU gate
  and for serving tests.

**Wire rows are FP8; the kernel is W4A16.** The wire carries FP8 rows, but
EXL3's tensor-core path multiplies FP16 activations by the decoded FP16
weights. The rank handles this in three steps:

1. It widens each FP8 value exactly to FP32 (`e4m3 × 2^(scale − 127)`).
2. It multiplies by `suh` and applies the Hadamard rotation in FP32.
3. It rounds once to FP16 for the tensor cores.

So the only precision the rank adds is that one FP16 rounding, after the
rotation, which is the same point where TensorFold rounds its BF16 input.

The loss the reference does not have is the coordinator's FP8 quantization of
a BF16 hidden row. It is a numerics choice of the wire format, and it **needs
the KL gate**. An optional BF16 row kind (8,192 bytes a row instead of 4,224)
would remove it, but that is a later protocol change.

Measured on real layers against the oracle's routed output, the FP8 rows add
little to the 4-bit weight error (`tests/real_experts.rs`):

| Relative RMS of the routed output | Exact FP32 rows | FP8 wire rows |
|---|---:|---:|
| Layer 3 prompt | 5.05% | 5.24% |
| Layer 3 decode | 5.85% | 6.00% |
| Layer 4 prompt | 9.91% | 10.38% |
| Layer 4 decode | 10.92% | 11.30% |

The rank refuses malformed input. Its fault word reports each case:

- E4M3 NaN codes and the UE8M0 NaN byte in a wire row;
- expert ids outside 0–287;
- negative or non-finite gate weights;
- an expert named twice in one row.

**Where the gate weights and the routed scale apply.** Read in
`modeling_glm5_next.py` (transformers @ `7cd73d9df0`):

- **Router.** `Glm5NextTextTopkRouter.forward` takes the sigmoid scores of the
  top 8 (chosen on score plus correction bias). It normalizes them to sum to 1
  (`norm_topk_prob`) and then multiplies them by `routed_scaling_factor` = 2.5.
  The 2.5 is therefore inside `top_k_weights`.
- **Experts.** `Glm5NextTextExperts.forward` works per routed (row, slot) pair:
  - `gate = x Wg^T`, clamped above at 10 (there is no lower clamp);
  - `up = x Wu^T`, clamped to [−10, 10];
  - `y = (silu(gate) × up) Wd^T × top_k_weights[row, slot]`;
  - `index_add` then sums `y` over the row's slots.
- **Shared expert.** It is added after the routed experts, on the coordinator.

The rank applies the wire's FP32 gate weights as they arrive. The coordinator
can either send weights with the 2.5 folded in, as the reference does, or send
normalized weights and multiply the four-rank sum by 2.5. The two give the
same linear map. The first costs nothing on the rank and keeps the BF16
return's relative precision unchanged.

## EXL3 K4

The format is ExLlamaV3's. [TensorFold](https://github.com/ashhart/TensorFold)'s
`exl3.py` describes it; `src/exl3.rs` repeats the description.

Each linear layer is stored as four tensors:

- `trellis`: I16 `[in/16, out/16, 64]`, one 16×16 tile per (k tile, n tile);
- `suh`: F16 `[in]`;
- `svh`: F16 `[out]`;
- `mcg`: the codebook marker, `0xCBAC1FED`.

**Decoding a tile.**

- The 64 words of a tile form a 1,024-bit circular stream.
- Value `p` is the 16-bit state that ends at bit `4 (p + 1)`.
- The state decodes through the "mcg" codebook:
  `x = s × 0xCBAC1FED`, `(x & 0x8FFF8FFF) ^ 0x3B603B60`, then two FP16 halves
  added with one rounding.
- Values land in tensor-core fragment order.

**The weight.**

```text
W = diag(suh) · H · W_q · H · diag(svh)
```

`H` is the 128-point Sylvester Hadamard matrix divided by √128, applied to each
block of 128 inputs or outputs.

**Pinned by two tests:**

- `tests/exl3_golden.rs` checks bit for bit against TensorFold's `exl3.py`, run
  unmodified: the codebook (all 65,536 states), the tile order, the states and
  the unpacked `W_q` at the rank's gate/up and down shapes. It also checks the
  float64 dequantization and forward pass, to 1e-12.
- `tests/real_experts.rs` checks against the official FP8 checkpoint and the
  reference oracle:
  - For experts 0, 1, 100 and 287 of layers 3 and 4, the EXL3 expert computes
    the official expert's function: FFN output cosine ≥ 0.987 on random rows,
    the 4-bit error.
  - The four ranks together reproduce the oracle's routed-expert output on
    real activations: cosine ≥ 0.990 (see
    [The kernel boundary](#the-kernel-boundary)).

**The EXL3 checkpoint reorders the intermediate channels.**
`tests/real_experts.rs` found that the EXL3 checkpoint stores each expert's
2,048 intermediate channels in an order different from the official
checkpoint's:

- Every EXL3 gate column matches exactly one official gate column, with cosine
  ≥ 0.996.
- The up columns follow the same permutation.
- So do the down rows (cosine ≥ 0.99).
- Only 1 of 32 sampled channels sits at the same index.

The expert is internally consistent, so its TP4 split is exact (next section).
But rank `r`'s 512 channels are not the official checkpoint's channels
`[512 r, 512 r + 512)`. **Never mix expert formats across ranks.**

## The TP4 split

Rank `r` owns intermediate channels `[512 r, 512 r + 512)`. These are the
output rows of gate and up and the input columns of down, in the `[out, in]`
convention of the FP8 and BF16 checkpoints. EXL3 stores the transposed layout
`[in/16, out/16, 64]`, so on disk they are gate/up tile columns and down tile
rows. The split is `glm53f_model::slicing`, and `src/layout.rs` places the
pieces:

| Tensor | gate / up (the channels are outputs) | down (the channels are inputs) |
|---|---|---|
| `trellis` | axis 1: tile columns `[32 r, 32 r + 32)` of `[256, 128, 64]` | axis 0: tile rows `[32 r, 32 r + 32)` of `[128, 256, 64]` |
| `suh` | whole, 4,096 | `[512 r, 512 r + 512)` |
| `svh` | `[512 r, 512 r + 512)` | whole, 4,096 |
| `mcg` | checked, not stored | checked, not stored |

**Why it is exact.**

- **The Hadamard transforms are block-diagonal.** Their 128-point blocks and
  the 16×16 tiles are independent units. A rank's 512 channels are exactly 4
  Hadamard blocks and 32 tiles, so no block or tile straddles two ranks.
- **Gate and up (outputs).** Output block `j` of `y = ((x · suh) H) W_q H · svh`
  needs all the inputs (so all of `suh` and all tile rows), the tile columns of
  block `j`, and `svh` of block `j`.
- **Down (inputs).** The input rotation of the rank's 512 activations needs
  only their `suh` and gives a partial `z_r` from the rank's tile rows. Because
  `H` and `svh` on the output side are linear,
  `Σ_r (z_r H) · svh = (Σ_r z_r) H · svh`. So each rank applies the full output
  transform to its own partial, and the four partials add up.
- **Only rounding changes.** The only numerical change from an unsplit expert
  is the order of the final sum over ranks.

**Evidence:**

- **Synthetic experts at full size.** `tests/tp4_slicing.rs` cuts an expert
  with random trellis bits at the real shape. Each rank's dequantized share
  equals its columns or rows of the unsplit weights to 1e-12. The four partial
  FFN outputs add up to the unsplit output to 1e-9. A share cut on the wrong
  axis is caught.
- **Real experts.** In `tests/real_experts.rs` (layers 3 and 4, experts 0, 1,
  100 and 287):
  - the share cut through the checkpoint's slicing plan (strided byte runs of
    the safetensors) is byte-identical to the share cut from the whole tensors;
  - the share dequantizes exactly to the whole expert's columns and rows.
- **glmrt** runs the same rule at TP4 for full GLM-5.3, which has the same
  2,048 intermediate width. It is v9, `glmrt-loader/src/exl3_format.rs` (with
  the `% (4 × 128)` alignment check) and `sparse_mlp/route.rs`
  (`queue_route_cuda_exl3_trellis_tp4`):
  - gate/up rows take `rank × local_row_bytes` of every tile row;
  - down is one contiguous `rank × local_bytes` range;
  - `svh` windows are at `rank × 512`, and the hidden-side vectors are whole.
- **TensorFold** runs the same rule at TP2: `split.py`, `EXL3_RULES`.

## The kernel

`kernels/exl3_rank.cu` has two C entry points, `g53r_ffn` (BF16 rows) and
`g53r_ffn_f32` (the same sums before their BF16 rounding, for the
reduce-scatter). A call uploads the rows, runs its kernels on one stream and
synchronizes once. It records a timing event only where a phase that ran a
kernel of its own ends: every event is a command on the stream, about a
microsecond of GPU time between two kernels. The stages:

1. **Plan:** it checks the routes, counts the (row, slot) pairs per expert,
   places them in ascending expert order and cuts each expert's pairs into
   groups of at most 16·MT pairs. Either one 1,024-thread CTA (after MiMo's
   `plan_parallel`; the order of an expert's pairs follows its atomics) or,
   at decode sizes (up to 512 routes), every gate/up block itself
   (`self_plan`: each expert's pairs in route order, so every block agrees,
   and the group's first block writes the plan for the down kernel). The
   launch sizes the grids with the bound
   `min(pairs, pairs / (16·MT) + min(288, pairs))` (`route::max_groups`,
   tested on the CPU twin); blocks past the actual group count exit, so the
   host never waits for the plan.
2. **Gate and up:** for each 128-wide K block the group's rows are rotated
   straight from the FP8 wire rows into shared memory (decode, × scale,
   × `suh`, a 128-point butterfly in FP32, × 1/√128, FP16; no rotated input
   is ever written to memory); the warps then decode the trellis tiles into B
   fragments and multiply with `mma.sync.m16n8k16` (FP16, FP32 accumulate).
   FP32 partial sums per K split.
3. **Epilogue:** per (pair, 128 block) the splits are summed in order and
   rotated back, × `svh`, GLM's clamped SwiGLU with the BF16 roundings the
   reference model makes in BF16 (TensorFold's choice), × down `suh`, the
   rotation and FP16: the down input.
4. **Down:** the same tile decoding and multiply over K = 512.
5. **Reduce:** per (row, 128 block), for each slot in order the splits are
   summed, rotated back and × `svh`, multiplied by the gate weight and added
   in FP32 (a multiply then an add); the sum is rounded to BF16 (or kept in
   FP32), and non-finite sums are faulted. Every slot's inputs are loaded
   before the first is used.

The rotation, the epilogue and the reduce are each one device function
(`row_rotate`, `epilogue_item`, `reduce_item`) that every path calls.

**Two kernel families** do the matrix products:

- **Split kernels** (`gateup_kernel`, `down_kernel`), for decode and verify
  windows. A block is one (matrix, group, K split) of 16 or 32 rows; its 4
  warps cover the slice's four 128-column Hadamard blocks. K is split into
  partial sums (8 for gate/up and 2 for down by default) so that enough
  blocks stream the few experts' weights:
  - the trellis words are loaded `pf` k tiles ahead (1, 2 or 4), in a ring
    of registers where each word's register is refilled as soon as the word
    is decoded (the loop has no branch: past the split's end it reloads its
    last tile);
  - with `l2` 2 they are loaded with an L2 evict-first policy (a template
    parameter: a runtime choice costs a second load instruction per word,
    which the one-row case showed);
  - the down kernel stages its group's down input for its K range in shared
    memory with `cp.async` when it starts, instead of loading the A
    fragments from memory every k tile;
  - memory the blocks would otherwise wait for is fetched into L2 when they
    start: each gate/up block's `suh` values of its split's later K blocks
    (only that block reads them, one DRAM round trip per K block otherwise),
    and the scale vectors the fused epilogue and reduce read.
- **Large-M kernels** (`gateup_big`, `down_big`), for prefill. Persistent
  blocks take work items from a counter in grid order:
  - gate/up: an item is one (matrix, group) of up to 32 or 64 rows (`mt` 2 or
    4) and all 512 columns; 8 or 16 MMA warps (`gw`) each own 64 or 32
    columns of every row, so each decoded tile feeds up to 8 MMAs (twice the
    split kernel's). The rotated rows are double-buffered in shared memory:
    during the first k tiles of a K block every warp rotates one row of the
    next K block (its inputs loaded a k tile ahead), with one barrier per K
    block; optionally two extra warps do all the rotation (`gp`). The trellis
    words are loaded 2–4 k tiles ahead, the A fragments come from `ldmatrix`
    and the MMAs of 16-row tiles past the group's rows are skipped;
  - down: an item is one (group, chunk of 128·`nt` output columns); the
    group's down input is staged in shared memory with `cp.async`,
    double-buffered. Items run chunk by chunk (the group varies fastest).

**Fusion.** By default the epilogue and the reduce are not kernels of their
own: the last block to finish a group's gate/up (all its matrices and splits)
runs the epilogue for the group, and the last block to finish a (row, output
chunk) of the down runs its reduce. They find out with an atomic counter per
group and per (row, chunk), after a memory fence; the last arrival clears the
counter for the next call (the counters are zeroed when allocated). The
partial sums the fused step reads were just written by the other blocks, so
they come from L2; with `discard` the step then drops those lines from L2
(`discard.global.L2`) instead of letting them be written back to memory.
Because the down items run chunk by chunk, a chunk's partial sums live in L2
for about one chunk's pass; `nt` sets how wide that is. The split kernels'
blocks run in grid order, with the split slowest (`ord` 1): the partial sums
of the first splits were written a whole kernel before the fused step reads
them. With `ord` 2 (the default up to 64 rows) a group's 2·`sk` gate/up blocks
are adjacent, and the down blocks run chunk by chunk with the splits fastest,
so their partial sums are read, and with `discard` dropped, while still in L2.

**Programmatic dependent launch** (`pdl` 2, `sm_90` and later). The down
kernel is launched with programmatic stream serialization. Every gate/up block
first allows it to be scheduled (`griddepcontrol.launch_dependents`); the down
blocks, which then start on free multiprocessors while the gate/up kernel
ends, plan the call themselves from the routes (the plan in the gate/up
blocks: the same plan), load their first trellis words, and wait until the
gate/up grid has completed and its memory is visible (`griddepcontrol.wait`)
before they read its down input. Nothing may stand between the two kernels on
the stream, so the call's gate/up phase then times both products. On an older
device the same kernels run in stream order: the development GPU tests this
path, not the overlap.

**What changes a bit.** Every row's arithmetic is independent of the other
rows: the tensor-core rows are independent, and every sum runs in a fixed
order. Only the K splits (`sk`, `skd`) and `fp32_swiglu` change the result;
the kernel family, the rows per group, the tilings, the plan, the fusion, the
L2 policies, the trellis prefetch depth, the block order and the dependent
launch are schedules of the same products and sums, bit for bit
(`tests/cuda_kernel.rs` compares 13 schedules at 5 row counts, and 52
split-kernel schedules at 1, 2, 4, 8, 16, 32 and 64 rows). So **under one
configuration a row gets the same bits whatever else is in the batch.**

**Configurations** (`g53r_cfg`, `exl3_cuda::Cfg`; zero fields take the
defaults for the row count):

| Field | Values | Meaning |
|---|---|---|
| `mt` | split: 1, 2; large-M: 2, 4 | 16-row tiles per group |
| `sk`, `skd` | divide 32 (large-M `skd` ≤ 4) | gate/up and down K splits |
| `fp32_swiglu` | 0, 1 | the SwiGLU in FP32 instead of the reference's BF16 roundings |
| `big` | 1, 2 | split kernels, large-M kernels |
| `nt` | 1, 2, 4 | large-M down: output chunk of 128·`nt` columns |
| `gw` | 8, 16 | large-M gate/up: MMA warps per block |
| `gp` | 2, or −1 for none | large-M gate/up: rotation warps (with `gw` 8, `mt` 2) |
| `plan` | 1, 2 | the planning kernel, or the gate/up blocks plan (split kernels, up to 512 routes) |
| `fuse` | 1, 2 | separate epilogue and reduce kernels, or fused |
| `discard` | 1, 2 | keep the fused steps' partial sums, or drop them from L2 |
| `l2` | 1, 2 | default caching of the trellis words, or an L2 evict-first policy (both families) |
| `pf` | 1, 2, 4 (split only) | split kernels: trellis words loaded `pf` k tiles ahead |
| `ord` | 1, 2 (split only) | split kernels' block order: the split slowest, or a group's (a down chunk's) blocks together |
| `pdl` | 1, 2 (split, `plan` 2, `fuse` 2) | the down kernel as a programmatic dependent launch (`sm_90` and later) |

**Which configuration a call runs** (`exl3_cuda::Policy`, the daemon prints
it at start-up):

| Rows | Default | Environment |
|---|---|---|
| 1 – `GLM53F_RANK_SMALL_MAX` (64) | split: `mt=1,sk=8,skd=2,plan=2,fuse=2,discard=2,ord=2` (`pf=1,l2=1,pdl=1`) | `GLM53F_RANK_SMALL` |
| to `GLM53F_RANK_MID_MAX` (2,048) | large-M: `mt=2,gw=8,nt=2,fuse=2,discard=2,l2=2` | `GLM53F_RANK_MID` |
| above | large-M: `mt=4,gw=16,nt=4,fuse=2,discard=2` | `GLM53F_RANK_LARGE` |

The variables hold `key=value` pairs over the regime's defaults (for example
`GLM53F_RANK_LARGE=nt=1,l2=2`). One configuration serves every decode and
verify window, so a row's bits do not depend on the window size; the middle
and large regimes must share `sk`, `skd` and `fp32_swiglu` (the policy
refuses otherwise), so a prefill row's bits do not depend on the row count
either. Decode-size and prefill-size calls use different K splits, so they are
not bitwise equal to each other; that is the boundary. Configurations can be
changed without rebuilding; the kernels are compiled for every value above.

**Build flags.** The kernels build with `--fmad=false`, so the rotations and
epilogues use the same separately rounded operations as the CPU reference.

**Memory per call** (grow-only scratch):

- at 4,096 rows, about 0.76 GB: the down partials are 0.54 GB (32,768 pairs ×
  4,096 × FP32). With `discard` most of them never reach memory, but the
  buffer is still their address range;
- at decode sizes, a few MB.

The layer images use 0.91 GB each, 38.4 GB for 42 layers.

**Correctness** (`tests/cuda_kernel.rs`, one test function holding one layer
on the GPU):

- **Stage by stage, each stage fed the GPU's own input** (the default
  schedules, with the partial sums kept):
  - gate/up and down products against f64: at most 5e-6 of the RMS at decode
    sizes and 4e-5 at prefill sizes. This is FP32 accumulation over K = 4,096.
  - the epilogue: bit for bit with the BF16 SwiGLU. With the FP32 SwiGLU up to
    4 FP16 values in 10^4 differ by one step, because `expf` differs by an
    ulp.
  - the final reduce: bit for bit.
- **Schedules:** at 8, 64, 200, 512 and 4,096 rows, 13 schedules (both
  families, `mt`, `nt`, `gw`, `gp`, both plans, fused and not, with and
  without `discard`) give the unfused split kernels' output bit for bit.
- **The split kernels' options:** at 1, 2, 4, 8, 16, 32 and 64 rows, every
  combination of `pf` 1, 2, 4, `l2`, `ord`, `discard` and `pdl` (48), and four
  with `mt` 2 or the planning kernel, give the unfused split kernels' output
  bit for bit at the small regime's splits; so do four at splits where the
  prefetch ring is longer than a down block's K range (`skd` 16 and 32), at
  `sk=4,skd=8` and at 200 rows. The options are refused with the large-M
  kernels, and `pdl` without the plan in the gate/up blocks or the fused
  steps.
- **End to end against the kernel-order reference:** RMS ≤ 2e-3 of the row's
  RMS at 1–8 rows and at 512 and 4,096 rows. Those one-ulp accumulation
  differences occasionally tip a BF16 rounding.
- **Against the dequantized FP32 model reference** (model semantics, BF16
  SwiGLU): RMS about 2e-3 at 1–8 rows.
- **Batch invariance:** a row alone, in a window of 8 and in a window of 64
  gives the same bits; so does a row alone and in 64 under a fixed prefill
  configuration of either family.
- **FP32 output:** rounded to BF16 it equals the BF16 output at 1, 8, 64, 512
  and 4,096 rows.
- **Faults:** all five refused (see [The kernel boundary](#the-kernel-boundary)),
  and the kernel serves again afterwards.
- **Real layer:** layer 3 was cut by the slicer and verified by the boot
  readback. Against the CPU reference it gives RMS ≤ 1e-3 (1, 8 and 256 rows).
- **Four ranks on real layers:** all four shares of layers 3 and 4 on the GPU,
  one at a time, reproduce the oracle's routed output: cosine ≥ 0.9931
  (layer 3) and ≥ 0.9903 (layer 4). The numbers equal the CPU path's.

An `fp32_swiglu` option computes the SwiGLU in FP32. It is used for the tight
test and is a KL-gate candidate.

## Measured on the development GPU

These are **proxy numbers**: an RTX 4090 (`sm_89`, 128 SMs, about 1 TB/s,
72 MB L2). A Spark has 48 SMs, unified LPDDR5X at about 270 GB/s and a
smaller L2, so the balance between compute and memory differs; its numbers
are the ones that count ([On a Spark](#on-a-spark)).

Method (28 September 2026): `exl3_bench --dir <rank-0 directory> --layer 3
--passes 8 --min` on a real layer slice, before (the kernel as of the previous
commit, with the same bench) and after, three runs each, interleaved. Routes
are uniform over the 288 experts, four routings per size. The GPU is shared
with other processes, which time-slice into the events, so the table gives
the fastest call's GPU time (plan to reduce) of the 96 per size and variant.

| Rows | Before | After | Speedup | After, by phase (plan, gate/up, epilogue, down, reduce) |
|---:|---:|---:|---:|---|
| 1 | 0.044 ms | 0.038 ms | 1.16× | 0.001, 0.020, –, 0.013, – (fused; in-block plan) |
| 2 | 0.056 ms | 0.048 ms | 1.17× | |
| 4 | 0.122 ms | 0.116 ms | 1.05× | |
| 8 | 0.214 ms | 0.207 ms | 1.03× | 0.001, 0.137, –, 0.070, – |
| 16 | 0.400 ms | 0.389 ms | 1.03× | |
| 64 | 0.900 ms | 0.888 ms | 1.01× | |
| 512 | 1.322 ms | 1.143 ms | 1.16× | 0.010, 0.768, –, 0.364, – |
| 1,024 | 1.651 ms | 1.407 ms | 1.17× | 0.018, 0.935, –, 0.452, – |
| 2,048 | 2.897 ms | 2.269 ms | 1.28× | 0.036, 1.447, –, 0.779, – |
| 4,096 | 5.319 ms | 4.026 ms | 1.32× | 0.066, 2.600, –, 1.368, – |

Before, by phase (same method): 1 row 0.003, 0.018, 0.004, 0.010, 0.008 ms;
2,048 rows 0.035, 1.549, 0.091, 0.893, 0.334 ms; 4,096 rows 0.068, 2.695,
0.190, 1.709, 0.648 ms.

What the profile showed (ablations of the kernels and a microbenchmark of the
MMA loop, since the GPU's performance counters are not available here):

- **Gate/up is tensor-bound at prefill sizes** on this card: 4,096 rows issue
  about 309 GFLOP of `m16n8k16` (FP16, FP32 accumulate) including the rows
  that pad 16-row tiles (about 11%). The card sustains 165–180 TFLOPS on that
  instruction in a long loop, and about 141 TFLOPS in blocks as short as the
  kernel's (256 k tiles, 4.5 waves). The trellis decode costs little at 32 or
  more rows a tile (164 against 168 TFLOPS in the loop) and drops the rate to
  117 TFLOPS at 16 rows. What is left over the MMAs is the input rotation (about 0.4 ms at
  4,096 rows), weight streaming when nothing hides it (576 MB), and the
  per-block start and end. After: 2.60 ms, about 119 TFLOPS issued.
- **Down and reduce were memory-bound:** the down kernel loaded its A
  fragments from memory per k tile, and it wrote 0.54 GB of FP32 partials at
  4,096 rows that the reduce read back. Staging A in shared memory, fusing the
  reduce and dropping the consumed partials from L2 took down + reduce from
  2.36 to 1.37 ms at 4,096 rows (1.7×) and from 1.23 to 0.78 ms at 2,048.
- **The fixed phases at one row** (plan, epilogue, reduce: 15 µs of 44) are
  gone as kernels: the plan runs inside the gate/up blocks and the epilogue
  and reduce in the last blocks. One row now takes 38 µs for 25.4 MB of
  weights.
- **Choices by size:** 64-row groups win above 2,048 rows and lose below
  (their blocks hold 16 warps, one per SM, and at 512–1,024 rows most groups
  have fewer than 32 rows). Rotation warps and a deeper weight prefetch for
  the 32-row kernel did not help (within noise or slower); the evict-first
  policy and persistent blocks against a plain grid were within noise.
- **Wall time** includes the PCIe copies of the rows in and the output out,
  about 25 µs at one row. On a Spark, whose memory is unified, those copies
  are memory to memory.

### Decode sizes (29 September 2026)

Method: `exl3_bench --dir <rank-0 directory> --layer 3,4 --rows
1,2,4,8,16,32,64 --passes 8 --min`. The calls take layers 3 and 4 in turn, as
a serving rank reads another layer's image on every call; 16 routings per
size. Before (the kernel as of the previous commit, with the same bench) and
after, five runs each, interleaved; the table gives the fastest call's GPU
time of the 640 per size and build. Every line's `bits` digest is the same in
both builds and in every configuration below, at every size.

| Rows | Before | After | Speedup | Bench GB/s, before → after |
|---:|---:|---:|---:|---:|
| 1 | 0.037 ms | 0.033 ms | 1.12× | 689 → 771 |
| 2 | 0.047 ms | 0.045 ms | 1.04× | 1,078 → 1,127 |
| 4 | 0.117 ms | 0.111 ms | 1.05× | 795 → 843 |
| 8 | 0.208 ms | 0.198 ms | 1.05× | 840 → 883 |
| 16 | 0.390 ms | 0.364 ms | 1.07× | 852 → 903 |
| 32 | 0.601 ms | 0.563 ms | 1.07× | 854 → 918 |
| 64 | 0.885 ms | 0.804 ms | 1.10× | 836 → 917 |

The bandwidth column is the bench's (the fastest call's weight bytes over its
time); at two rows it exceeds the card's memory, because part of the weights
are still in its 72 MB L2 from two calls before. At 128–4,096 rows (the
large-M kernels, three runs each) the times are within 1% of before, the same
bits.

What made the difference, each change the same bits:

- **Fewer events.** Three of a call's six timing events closed empty phases
  (the plan in the gate/up blocks, the fused epilogue and reduce), each about
  a microsecond of GPU time. Under nsys (one row, minimum; nsys adds its own
  launch cost) the gap from the end of the gate/up kernel to the start of the
  down kernel fell from 4.6 to 2.6 µs, and the span from the start of the one
  to the end of the other by about 3 µs.
- **The block order with `discard`** (`ord=2,discard=2`, now the default):
  against `ord=1,discard=1` in the same build, 64 rows 0.808 against 0.885
  ms, 32 rows 0.563 against 0.601, 16 rows 0.368 against 0.384, 8 rows 0.198
  against 0.205, 4 rows 0.111 against 0.115; at one and two rows within the
  noise (±2 µs; the discards run in the last blocks' tails). Either alone
  gains little (64 rows: 0.866 with `ord=2`, 0.871 with `discard=2`).
- The down kernel's down input in shared memory and its group, pair rows
  and first trellis words loaded together; the scale vectors of the fused
  steps and the `suh` values of later K blocks fetched ahead (for GB10's
  memory latency: here its L2 holds them anyway).

The options against the new default, in the same build (three runs each,
fastest call, ms):

| Rows | Default | `ord=1,discard=1` | `pf=2` | `pf=4` | `l2=2` | `pf=2,l2=2` | `pdl=2` |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.034 | 0.032 | 0.036 | 0.036 | 0.038 | 0.036 | 0.033 |
| 2 | 0.044 | 0.044 | 0.046 | 0.047 | 0.049 | 0.047 | 0.044 |
| 4 | 0.111 | 0.115 | 0.115 | 0.125 | 0.088 | 0.113 | 0.111 |
| 8 | 0.198 | 0.205 | 0.201 | 0.207 | 0.193 | 0.200 | 0.202 |
| 16 | 0.368 | 0.384 | 0.371 | 0.366 | 0.353 | 0.370 | 0.371 |
| 32 | 0.563 | 0.601 | 0.572 | 0.570 | 0.558 | 0.564 | 0.566 |
| 64 | 0.808 | 0.885 | 0.816 | 0.810 | 0.793 | 0.795 | 0.820 |

- **`pf`** gains nothing on this card: at one row the gate/up kernel already
  streams its 16.8 MB in 19 µs, near the card's memory speed, with 512 KB in
  flight. The gate/up kernel with `pf=4` takes 167 registers (three blocks
  a multiprocessor instead of four) and loses at 4–8 rows.
- **`l2=2`** is faster from 4 rows (the 4-row minimum varies between runs,
  0.088–0.115 ms) and 10–12% slower at one and two rows, where each
  multiprocessor runs one block and the hinted loads' latency shows.
- **`pdl=2`** runs here in stream order: the down blocks plan themselves,
  equal at 1–4 rows and 1–2% slower at 8–64. Its overlap needs `sm_90` or
  later.
- The large-M kernels at the small regime's splits
  (`big=2,mt=2,gw=8,nt=2,sk=8,skd=2,l2=2`, the same bits) were 5–70% slower
  than the split kernels' schedule before at 1–32 rows, and even at 64 rows
  (0.873 against 0.883 ms): the planning kernel, and a persistent grid of
  256-thread blocks for few groups.

## On a Spark

The GB10 numbers of the kernel before these changes are in
`docs/PERFORMANCE.md` (1 row 0.145 ms, 4,096 rows 16.9 ms). The kernels are
unchanged in the arithmetic; every schedule knob is read from the environment
at start-up, so the sweep needs no rebuild.

```sh
# On the Spark (arm64, CUDA 13). The daemon (cuda,rdma) and the kernel test and bench (cuda) in separate
# target directories: a build with other features relinks target/release/glm53f-rank without them.
export GLM53F_CUDA_ARCH=sm_121        # GLM53F_CUDA_LIB if libcudart is not in /usr/local/cuda/lib64
CARGO_TARGET_DIR=target-rank  cargo build -p glm53f-rank --release --features cuda,rdma
CARGO_TARGET_DIR=target-bench cargo test  -p glm53f-rank --release --features cuda --test cuda_kernel -- --nocapture --test-threads=1
CARGO_TARGET_DIR=target-bench cargo build -p glm53f-rank --release --features cuda --example exl3_bench
B=target-bench/release/examples/exl3_bench
$B --dir <rank-dir> --layer 3,4 --min --passes 4
$B --dir <rank-dir> --layer 3,4 --min --sweep --rows 1,2,4,8,16,32,64,512,1024,2048,4096

# Decode: the split kernels' options (each keeps the default's bits: compare the lines' `bits` digests).
$B --dir <rank-dir> --layer 3,4 --rows 1,2,4,8,16,32,64 --min --passes 8 \
  --cfg ord=1,discard=1 --cfg pf=2 --cfg pf=4 --cfg l2=2 --cfg pf=2,l2=2 --cfg pf=4,l2=2 \
  --cfg pdl=2 --cfg pdl=2,pf=2 --cfg pdl=2,pf=4 --cfg pdl=2,pf=2,l2=2 --cfg pdl=2,pf=4,l2=2

# The sweep's winners become the defaults without a rebuild, for example:
GLM53F_RANK_SMALL=pf=4,l2=2,pdl=2 GLM53F_RANK_MID=nt=1,l2=2 GLM53F_RANK_LARGE=mt=4,gw=16,nt=1 \
  $B --dir <rank-dir> --layer 3,4 --min

# Once per rank: cut its share and serve it. --peers lists the four ranks' peer-mesh
# addresses in rank order (the same list on every rank) for the prefill reduce-scatter.
# GLM53F_WIRE_NOCRC=1 as on the coordinator (its RDMA transport requires it); GLM53F_RDMA=1
# here would also put the peer mesh on RDMA (built, untested).
target-rank/release/glm53f-rank slice --checkpoint <exl3-checkpoint> --rank R --out <rank-dir> --source "<repo>@<revision>"
GLM53F_WIRE_NOCRC=1 target-rank/release/glm53f-rank serve --rank R --dir <rank-dir> --listen <fabric-address>:8600 \
  --peers <fabric-0>:8601,<fabric-1>:8601,<fabric-2>:8601,<fabric-3>:8601
```

**What to compare:**

- **Kernel test:** it should pass unchanged; the schedule comparisons are bit
  for bit on any GPU, and only the tensor cores' accumulation rounding could
  differ from the references.
- **Decode (1–64 rows):** the default against `ord=1,discard=1` (the block
  order before), then the bytes in flight (`pf` 2 and 4: expected to matter
  most at 1–3 rows, see [Measured on GB10](#measured-on-gb10)), `l2=2`, and
  `pdl=2` (the down kernel's start overlapped with the gate/up kernel's end,
  which the development GPU cannot show), and their combinations; `plan=1`
  and `fuse=1` as before. The small regime is one configuration for every
  size up to 64 rows, so the choice weighs one-row decode against 16–64-row
  verify windows. The target is 230 GB/s at 1–64 rows.
- **Prefill (512–4,096 rows):** the default against
  `big=1,mt=2,sk=1,skd=1,plan=1,fuse=1,discard=1,ord=1` (the kernels before), then
  `mt` and `gw` (32- or 64-row groups; where the crossover sits on 48 SMs),
  `nt` 1, 2, 4 with `discard` 2 against 1, and `l2=2`. On GB10 the down
  partials and the down input compete for a smaller L2 than on the 4090: a
  narrow chunk (`nt=1`) keeps a chunk's partials in L2 but reads the down
  input once per chunk, a wide one the reverse, and `l2=2` gives way to both
  over the streamed weights. The fused steps' benefit over `fuse=1` should be
  larger than here, since the partials cost over 1 GB of memory traffic per
  layer at 4,096 rows without them.
- **Set the defaults** from the sweep in `g53r_default_cfg` (or in the
  environment of the daemon), keeping one configuration up to 64 rows and the
  same `sk`, `skd` above.

## Measured on GB10

`exl3_bench` on one Spark: layer 3 of rank 0, CUDA 13, `--min`. The kernel
test passes there unchanged.

| Rows | Before the tuning | Change | After |
|---:|---:|---|---:|
| 1 | 0.131 ms (194 GB/s) | — | — |
| 16 | 1.530 ms (221 GB/s, about 103 experts) | — | — |
| 64 | 3.522 ms (216 GB/s, about 230 experts) | — | — |
| 2,048 | 7.869 ms | `l2=2` | 7.087 ms |
| 4,096 | 14.218 ms | `nt=4` | 13.283 ms |

The two changes are now the defaults of the middle and large regimes. Neither
changes a bit.

**Served** (four ranks, two prefill lanes, the coordinator on an RTX 5090):
- Prefill improved 1.5–2% with the tuned defaults (lanes of 2,048 rows). The
  kernel itself was 6–8% faster than the one before it.
- The rank's trace gives the weight bandwidth of decode-size calls: the
  distinct expert blocks it read times 3,173,376 bytes, over the GPU time.

| Rows per call | Expert blocks read | GPU time | Bandwidth |
|---:|---:|---:|---:|
| 1 | 8 | 0.134 ms | 189 GB/s |
| 2–3 | 15 | 0.254 ms | 187 GB/s |
| 4–12 | 32 | 0.495 ms | 205 GB/s |
| 13–32 | 67 | 1.043 ms | 204 GB/s |
| 33–64 | 103–110 | 1.62–1.74 ms | 200–201 GB/s |
| 65–128 (large-M, `l2=2`) | 142 | 1.929 ms | 234 GB/s |

- GB10's memory peaks at 273 GB/s, so the split kernels, which serve every
  decode and verify window, reach 68–75% of it and the large-M kernel 86%.

**What the table says about the split kernels** (arithmetic on the table;
the conclusions are inferences until the options are swept on a Spark, which
the next table does):

- **Their cost is per expert, not per call.** One row reads 8 expert blocks
  in 0.134 ms and two or three rows 15 in 0.254 ms: 17.1 µs an expert and no
  fixed cost left over (0.134 = F + 8x and 0.254 = F + 15x give x = 17.1 µs,
  F = −3 µs). From 4 rows the kernels take 15.5–15.8 µs an expert, the
  large-M kernels 13.6. The split kernels stream slower; their plan,
  epilogue and reduce are not what is missing.
- **They look latency-bound.** At one row the gate/up kernel runs 128 blocks
  of 4 warps, each warp with one k tile of trellis words (1 KB) in flight:
  512 KB in flight. At 189 GB/s, Little's law (bytes in flight = bandwidth ×
  latency) puts the memory's latency at about 2.7 µs; the large-M kernels
  (96 blocks of 8 warps, 1 KB each: 768 KB) at 234 GB/s give 3.3 µs. At that
  latency 273 GB/s needs about 750 KB in flight before any queueing. `pf=2`
  and `pf=4` put two and four times the bytes in flight (1 and 2 MB at one
  row).
- **Larger windows' partial sums can leave L2.** With the split slowest
  (`ord=1`), the gate/up partial sums of the first splits (2 × 8 × pairs ×
  2 KB: 4 MB at 16 rows, 17 MB at 64) and the down partial sums (2 × pairs ×
  16 KB: the same) are read a whole kernel after they are written. In an L2
  smaller than the 4090's they are written back and read again: up to about
  67 MB beside 730 MB of weights at 64 rows. `ord=2` with `discard=2`
  (the default now) reads them while they are in L2 and never writes them
  back; `l2=2` protects them from the other side.
- **Dependent round trips cost more here.** Each is a latency of the order
  above: the down kernel's start (the group, then its first trellis words)
  and the gate/up kernel's tail (the epilogue) sit between the two weight
  streams of every call. `pdl=2` overlaps the down kernel's start with the
  gate/up kernel's end; the scale vectors are now fetched into L2 before they
  are needed.

**The split kernels swept on GB10** (29 September 2026):
- Layers 3 and 4 alternating (`--layer 3,4`), `--passes 8`, best of two runs.
- Every configuration below printed the same output digest at every row count.

| Rows | Before the sweep | `ord=1,discard=1` | `ord=2,discard=2` | `pf=2,pdl=2` | **`l2=2,pf=2,pdl=2`** |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.131 ms | 0.086 | 0.113 | **0.080** | 0.084 |
| 2 | 0.248 | 0.233 | 0.228 | 0.220 | **0.195** |
| 4 | 0.454 | 0.410 | 0.401 | 0.394 | **0.379** |
| 8 | 0.852 | 0.766 | 0.736 | 0.732 | **0.719** |
| 16 | 1.530 | 1.430 | 1.360 | 1.358 | **1.336** |
| 32 | — | 2.317 | 2.199 | 2.199 | **2.150** |
| 64 | 3.522 | 3.374 | 3.173 | 3.174 | **3.087** |

- `l2=2,pf=2,pdl=2` is the fastest or within 5% of it at every size. On GB10 builds (sm_121) it is the small regime's default; other GPUs keep `l2=1,pf=1,pdl=1`, since on the development GPU `l2=2` was 10–12% slower at 1–2 rows.
- Through the policy it reads 221–237 GB/s at 1–64 rows (234–237 from 8 rows). The one-row figure is noisy between runs (0.084–0.115 ms).
- `pf=4` never helped (fewer blocks per SM). The prefill sizes with the middle and large regimes' defaults: 2,048 rows 6.755 ms, 4,096 rows 12.709 ms.
- **Served** (the same four ranks, the rank trace):

  | Rows per call | 1 | 2–3 | 4–12 | 13–32 | 33–64 | 65–128 |
  |---|---:|---:|---:|---:|---:|---:|
  | GB/s | 198 | 198 | 220 | 226 | 226 | 233 |

  That is 5–13% over the table above for 1–64 rows. End to end, single-stream decode rose 3–4% and 16 streams 6%.

## The rank directory

A rank serves a directory with one image per MoE layer, named
`L03.r0.exl3` … `L44.r0.exl3`. Each image is 913,932,288 bytes, layout
`glm53f-exl3-k4-tp4-e1`.

**The image.** It holds the 288 expert blocks in expert order. Each block is
3,173,376 bytes: the gate, up and down trellis shares (1 MiB each), then the
`suh`/`svh` vectors (`src/layout.rs` has the table). The image is the kernel's
input as is. `prepare_layer` uploads it and checks that every scale vector is
finite.

**The manifest.** `manifest.txt` lists, per layer, the file, its size and its
SHA-256. It also records the layout, the rank, the world size and the source
checkpoint.

**The boot readback** (`boot.rs`, after MiMo) checks every image and names
every failure. It refuses to serve on:

- a size or SHA-256 mismatch;
- a missing image;
- an image the manifest does not list;
- another rank's manifest, a different layout or a different world size;
- a missing decoder layer.

Images are hashed in parallel.

**`glm53f-rank slice`** cuts a rank's share from an EXL3 checkpoint. The
checkpoint can be a whole copy or a subset that holds the wanted layers. The
slicer:

1. reads the config and the safetensors headers through `glm53f-model`;
2. reads each expert's strided byte runs from the shards;
3. checks the `mcg` markers and the split rule;
4. writes each layer through a `.part` file;
5. writes the manifest last.

Layer 45, the MTP layer's experts, is optional (`--mtp`).

## The daemon

**`glm53f-rank serve`** is mimo26f-afd's rank daemon with its kernel path
replaced:

1. **Boot readback.**
2. **Device gate.** The kernels must have been compiled for this GPU.
3. **Prepare every layer** before listening. `--lazy` prepares each on first
   use instead. `--allow-partial` serves a directory with only some layers
   (bring-up and tests); a request for a missing layer fails.
4. **Join the peer mesh** when `--peers` is given (see
   [The peer mesh](#the-peer-mesh)): only then does the rank serve
   reduce-scattered requests.
5. **Accept connections.** Connections that do not arrive on the RDMA fabric
   are refused (`glm53f_rdma::fabric_port`; `GLM53F_WIRE_ALLOW_LAN=1` for
   tests). A coordinator that opens with the RDMA handshake gets an RC queue
   pair; anything else stays on TCP.

The handshake value `M26RDMA1` is kept as a protocol constant shared with the
coordinator's `glm53f-rdma`. The frames are `DS41RTE3` v3 with the L4 ladder
(CRC32C and a per-connection sequence).

**Over RDMA:**

- a request is served where the NIC landed it (the receive ring is
  page-locked for the GPU copies);
- the BF16 return is written straight into the registered send buffer;
- the slot is re-posted before the return goes out.

**Queued requests.** The coordinator keeps up to one exchange in flight per
prefill lane (`glm53f-serve --prefill-lanes`, up to four). Over RDMA a
request lands in a receive slot the rank posted beforehand, so a rank takes
as many requests ahead of the one it computes as it posts slots:
`--recv-slots` (4 by default). Every slot is posted when the connection
opens; a queue pair's receive queue is first in, first out, so the requests
land in the oldest posted slots in the order they were sent and are served in
that order; each slot is posted again, at the back of the queue, once its
request is on the device (`transport::RecvSlots`, whose bookkeeping
`transport.rs` tests on the CPU against a model of the receive queue: up to N
requests in flight on N slots are never refused and are served in order).
The rank names its slots in the RDMA handshake (`glm53f_rdma::Info::recv_slots`,
a word that was reserved and zero), and the coordinator keeps no more
exchanges in flight than every rank queues. The frames are unchanged
(`DS41RTE3` v3 and v4); a rank built before the field sends zero, which the
coordinator reads as two slots, and ignores the coordinator's word. A request
with no slot posted is not lost: the NIC retries it (receiver-not-ready
retries without limit) until a slot is posted again. The peer mesh keeps two
slots per link: its frames are copied out as soon as they are polled.

**Over TCP** the same code runs on owned buffers.

**Failures.** A failed request gets a `Status::Error` return and the
connection closes. The daemon keeps listening: it fails the request, not the
rank.

**Flags and environment variables:**

- `--recv-slots N` (`GLM53F_RANK_RECV_SLOTS`, 1 to 16, default 4): RDMA
  receive slots per coordinator connection, the requests it queues (17 MiB
  each, page-locked);
- `--peers A0,A1,A2,A3` (`GLM53F_RANK_PEERS`): the four ranks' peer-mesh
  addresses in rank order; this rank listens on its own;
- `--peer-timeout-ms N` (`GLM53F_RANK_PEER_TIMEOUT_MS`, default 10,000): how
  long an exchange waits for a peer's link or frame before it fails;
- `GLM53F_RDMA=1`: the peer links this rank dials are RDMA RC (an `rdma` build);
- `GLM53F_RANK_TRACE=1`: a timing line per request, and an `exchange` line per
  reduce-scattered request;
- `GLM53F_TIMELINE=1`: cross-host timeline events;
- `GLM53F_RANK_DUMP_FRAME=<path>` (with `GLM53F_RANK_DUMP_LAYER`, default 3):
  the first request frame of that layer is written to `<path>`, for offline
  replay.

`tests/daemon.rs` runs the binary: two requests, a request for a missing layer
(error return, connection closed) and a new connection, over TCP. The
reduce-scatter's daemon test is glm53f-coordinator's `tests/row_sharded.rs`.

## Prefill reduce-scatter

**The problem.** In the four-plane return every rank returns every row:
`rows × 8,192` bytes each, four partial planes converging on the coordinator's
port. At 4,096 rows that is 4 × 32 MB per layer into one 200 Gb/s port, about
5 ms per layer. It is a 4→1 incast that drops packets on switches without
priority flow control. For prefill-sized requests the ranks reduce among
themselves and return each row once.

It follows glmrt v9 (`rdma_reduction.rs`, `intermediate_sharding.rs`),
reimplemented: `src/reduce_scatter.rs` (the arithmetic and bookkeeping),
`src/mesh.rs` (the links between the ranks) and version 4 of the wire
(`glm53f-wire`, `src/row_shard.rs`).

### Protocol

1. **When.** The coordinator decides per exchange, by its row count
   (`ReturnPath::RowSharded` in glm53f-coordinator's wire client: 16 rows and
   more by default; decode and verify windows keep the four-plane return,
   whose latency is better). It sets `FLAG_REDUCE_SCATTER` (bit 18) on the
   request to all four ranks, and `FLAG_EXCHANGE_FP8` (bit 19) for the FP8
   exchange. Such a request is `DS41RTE3` version 4 and has at least one row
   per rank.
2. **Partition.** The rows split into four contiguous ranges, and the first
   `rows % 4` ranks get one extra row (`row_partition`, glmrt's
   `balanced_row_partition`).
3. **Exchange.** Each rank computes its full FP32 partial: the kernel's
   reduce output before its BF16 rounding (`ExpertKernel::ffn_f32`; rounded,
   it is the four-plane return's rows bit for bit). It sends each peer that
   peer's rows in an exchange frame: version 4, kind 3, with the header and L4
   tail of every frame (a sequence per link and direction, the CRC32C unless
   disabled). The frame's `executor_id` is the sender, `token_position` the
   first row, and the word at byte 124 packs the receiver, the world size and
   the request's row count.
4. **Reduce.** For each of its rows, a rank adds per element, in rank order
   0–3 and in FP32, its own row and the three decoded peer rows. It rounds the
   sum to BF16. A missing, duplicate, corrupted or mismatched frame fails the
   request.
5. **Return.** A row-slice return carries only the rank's rows: version 4,
   `FLAG_ROW_SLICE` (bit 17) with the usual `SPARK_REDUCTION |
   V41_COMPACT_BF16`, `row_count` the partition's rows and `token_position` its
   first row. The coordinator checks each rank's slice against the partition
   and places the rows: each row once, nothing added
   (`WireClient::collected`, or `CoordinatorSum::row_sharded` on the host
   path).

**Old and new peers.** Only the frames that use the extension are version 4. A
version-3 decoder accepts version 3 only, so an old rank refuses a
reduce-scattered request (and closes the connection) instead of answering with
a full plane, and an old coordinator never adds a row slice as a plane.
Version-3 frames are unchanged, so old ranks still serve four-plane requests.
The version-4 decoder refuses a version-4 request or return without its flag,
a version-3 one with a version-4 flag, and a nonzero word at byte 124 in
requests and returns (`glm53f-wire`, `tests/row_shard.rs`).

### The peer mesh

- **Links.** One link per pair of ranks, six in all. The lower rank dials and
  the higher one accepts. Each rank listens on its own entry of `--peers` (the
  four ranks' addresses in rank order, on the fabric) and dials the ranks
  above it. The daemon's fabric guard applies to these connections too
  (loopback is exempt; `GLM53F_WIRE_ALLOW_LAN=1` for tests).
- **Hello.** A 16-byte hello each way (magic `G53RMESH`, wire version 4, world
  size, source and destination ranks, transport) refuses a peer at the wrong
  address, of another version or of another world before any frame.
- **Transport.** RDMA RC with `GLM53F_RDMA=1` in an `rdma` build: the
  coordinator link's transport, sized for exchange frames (two receive slots
  and two send halves of 8.4 MB per link). The dialling side opens it and the
  accepting side follows the hello. Otherwise TCP: a reader thread per link
  validates each frame and hands it to the serving thread, so two ranks
  sending each other megabytes at once cannot block each other. Over RDMA the
  serving thread polls the receive queue, as the RDMA design polls
  completions.
- **Keys.** An exchange is keyed by request id and layer. Frames of another
  exchange (another of the coordinator's prefill lanes, which a faster peer
  may already be sending) wait until their exchange claims them. The coordinator
  starts each connection's request ids at a random base, so a stale frame
  never matches a later exchange; frames nobody claims are dropped after twice
  the timeout.
- **Failure.** An exchange fails and never hangs:
  - a peer without a link, or whose frame has not arrived, within
    `--peer-timeout-ms` (10 s by default);
  - a link lost while its frame is awaited: at once;
  - a frame that names the exchange but not its rows, dtype or ranks, and a
    duplicate.

  The daemon sends an error return and closes the coordinator's connection,
  as for any failed request. The lower rank of a lost link dials it again.
- **Trace.** Under `GLM53F_RANK_TRACE=1`, one line per exchange:
  `exchange ... part=<first>+<rows> dtype=... send=... wait=... sum=...
  return=... ms`. They are the encoding and posting of the three frames, the
  wait for the peers, the sum and its BF16 rounding, and the return.

**Tests.** `tests/reduce_scatter.rs` (all four ranks on the CPU against the
exact sum), `tests/mesh.rs` (four meshes over loopback: every partition bit for
bit against the in-process protocol, two exchanges in flight, a peer that never
sends, one that dies during the exchange, one gone before it, a wrong peer
list), and glm53f-coordinator's `tests/row_sharded.rs` (four daemons on the
GPU: the oracle's layers both ways, decode staying four-plane, timings, a
killed peer).

### Error and dtype: BF16 exchange by default

The simulation takes four independent partial planes. They are mostly unit
normal, with a few channels 40× larger and a few rows 1,000× smaller. It
checks every returned element against the four-plane sum, within a bound made
of half a step of each exchanged value, half a BF16 step of the result and the
FP32 additions. It also measures the RMS error:

| Return path | RMS error / RMS of the sum |
|---|---:|
| Four BF16 planes, added in FP32 by the coordinator | 1.7e-3 |
| Reduce-scatter, **BF16** exchange | 2.2e-3 |
| Reduce-scatter, FP8 E4M3 exchange, one FP32 scale per row (glmrt's) | 1.9e-2 |

FP8 exchange is ten times less precise. E4M3 has 3 mantissa bits, so each
exchanged value is off by up to 6%. It is a numerics change that would have to
pass the KL gate.

**On real layers** (glm53f-coordinator `tests/row_sharded.rs`: four daemons on
the development GPU, the oracle's layer-3 and layer-4 prefill inputs, 33 rows,
the same request both ways). Every row-sharded element is within the bound
above applied to the difference of the two paths: half a BF16 step of the
owning rank's own partial (unrounded in one path, rounded in the other), half a
BF16 step of the row-sharded sum, the FP32 additions of both paths, and for
the FP8 exchange half an E4M3 step of each peer's value.

| Layer | Exchange | Row-sharded vs four planes: RMS / RMS of the sum | Worst element / bound | Cosine vs the oracle: four planes, row-sharded | Relative RMS vs the oracle: four planes, row-sharded |
|---|---|---:|---:|---|---|
| 3 | BF16 | 1.98e-3 | 0.97 | 0.9931, 0.9931 | 5.24%, 5.24% |
| 4 | BF16 | 1.79e-3 | 0.97 | 0.9903, 0.9903 | 10.38%, 10.38% |
| 3 | FP8 | 1.91e-2 | 0.83 | 0.9931, 0.9929 | 5.24%, 5.67% |
| 4 | FP8 | 2.20e-2 | 0.82 | 0.9903, 0.9900 | 10.38%, 10.61% |

Against the oracle, the BF16 exchange changes nothing visible; the FP8
exchange adds 0.2–0.4 points of relative RMS.

BF16 exchange costs no extra egress. Per rank, per row of the request:

| | To peers | To the coordinator | Total out | Into the coordinator (all ranks) |
|---|---:|---:|---:|---:|
| Four planes | 0 | 8 KB | 8 KB | 32 KB |
| Reduce-scatter, BF16 | 6 KB | 2 KB | **8 KB** | **8 KB** |
| Reduce-scatter, FP8 | 3 KB | 2 KB | 5 KB | 8 KB |

**BF16 is the default exchange.** FP8 is worth it only if Spark-to-Spark
bandwidth turns out to be the limit and the KL gate allows it.

### Remaining incast and timing

- **Incast.** Each Spark still receives from three peers at once, but each
  burst is a quarter of the four-plane return's per-rank volume. If that still
  drops packets, a ring schedule (three steps, each rank sending to one
  neighbour) removes incast altogether. It costs two extra hops of latency.
- **Measured on the development GPU only** (four daemons sharing one RTX 4090,
  loopback TCP, 2,048 rows, glm53f-coordinator `examples/wire_bench.rs`, median
  of 10): four planes 23.7 ms and row-sharded (BF16) 33.9 ms per exchange,
  with 67.1 MB and 16.8 MB into the coordinator. Loopback has no incast to
  remove, and the reduce-scatter adds the ranks' work: encoding the frames
  (5–11 ms for 1,536 rows) and the sum (1.4–3.1 ms), both on the CPU, spread
  over threads. The target hardware's numbers are the ones that count.
- **Next, if the trace shows them:**
  - the BF16 frames are the kernel's BF16 output rows bit for bit, so the
    kernel could write the peers' rows as BF16 and only the rank's own rows as
    FP32, and the encoding would disappear;
  - the sum could run on the GPU;
  - exchange and reduction can run chunk by chunk, overlapped with the next
    chunk's kernels; row chunks of the prefill wavefront already fit that
    pattern.

## Open issues

1. **Spark numbers.** The large-M kernels, the fused epilogue and reduce, the
   plan in the gate/up blocks and the split kernels' options (`pf`, `l2`,
   `ord`, `pdl`) are measured on the 4090 only, and `pdl`'s overlap not at
   all; the kernel test, the sweep and the defaults on GB10 are listed in
   [On a Spark](#on-a-spark). The 4090 is a proxy. *(29 September 2026: the
   kernels have since run on GB10 and the split kernels' options were swept
   there, `pdl` included in the timings though not measured apart: see
   [Measured on GB10](#measured-on-gb10).)*
2. **KL gates.** Three numerics choices need them:
   - the wire's FP8 rows, where BF16 would be a protocol change (the engine's
     gate runs on the target hardware include them and pass,
     `docs/KL-GATE.md` §6a; no paired run against BF16 rows exists);
   - the BF16-rounded SwiGLU versus FP32 (`fp32_swiglu`);
   - the reduce-scatter's exchange dtype.
3. **Channel order.** The EXL3 checkpoint reorders each expert's intermediate
   channels relative to the official checkpoint. A rank's share is correct for
   that checkpoint only; formats must never be mixed across ranks.
4. **Batch invariance.** Prefill-size calls (above 64 rows) use another K
   split than decode-size calls. A row is bitwise batch-invariant within each
   regime, not across them. The schedule knobs do not move this boundary; the
   K splits of the regimes do (`GLM53F_RANK_SMALL_MAX`).
5. **The reduce-scatter** runs over TCP on one machine, through the forward
   too (glm53f-forward's `RemoteExperts`). On the target hardware over the TCP
   mesh it was slower than the four-plane return (28 September 2026): the ranks
   encode their peers' rows and sum them on the CPU, which costs more than the
   transfer it saves. Still to do: the exchange rows and the sum on the GPU,
   and the peer mesh over RDMA (built, untested).
6. **Performance.** Candidates:
   - the input rotation of the large-M gate/up (about 15% of it on the 4090):
     it is recomputed per (row, matrix), and its shuffle chains delay the
     MMAs of the warp that runs it;
   - 16-row tiles pad each group by 11% at uniform routing, more with skewed
     routing; 8-row granularity needs the weights as the A operand (the
     decoded fragments are already that operand's layout);
   - graphs for the fixed decode shapes (the dependent launch between the two
     products is `pdl`, not yet measured where it applies);
   - the split gate/up kernel's rotation at each K block boundary: two
     barriers and a round trip for the rows' inputs (an L2 hit now that the
     `suh` values are fetched ahead). A double-buffered rotation from inputs
     loaded a K block ahead (as in the large-M kernel) removes both, but a
     first version took 147 registers at 16 rows against 128 (three blocks a
     multiprocessor instead of four); staging the inputs in shared memory
     with `cp.async` would not;
   - the uploads and downloads of a call (about 25 µs of wall time at one row
     on the 4090): one staging copy each way, or the output written straight
     into the send buffer through mapped memory.
7. **Layer 45** (MTP experts) is supported by the slicer, the manifest and the
   daemon, but has not been exercised with real data.
