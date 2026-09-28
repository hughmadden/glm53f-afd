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
  decided row (glm53f-forward `tests/remote_experts.rs`).
- Not yet done:
  - running on a Spark (`sm_121`);
  - the RDMA path, for the coordinator's link and the peer mesh (built, but
    untested without a fabric).

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
cargo run  -p glm53f-rank --release --features cuda --example exl3_bench [-- --sweep]

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

`kernels/exl3_rank.cu` has one C entry point, `g53r_ffn`. A call runs five
kernels on one stream and synchronizes once:

1. **Plan** (one 1,024-thread CTA, after MiMo's `plan_parallel`):
   - it checks the routes;
   - it counts the (row, slot) pairs per expert;
   - it places them in ascending expert order;
   - it cuts each expert's pairs into groups of at most 16·MT pairs.

   The launch sizes the grids with the bound
   `min(pairs, pairs / (16·MT) + min(288, pairs))` (`route::max_groups`, tested
   on the CPU twin). Blocks past the actual group count exit, so the host never
   waits for the plan.
2. **Gate and up.** Each block is one (group, matrix, K split). Its 4 warps
   cover the slice's four 128-column Hadamard blocks.
   - **Rotation.** For each 128-wide K block, the warps first rotate the
     group's rows straight from the FP8 wire rows into shared memory: decode,
     × scale, × `suh`, a 128-point butterfly in FP32, × 1/√128, FP16. One warp
     handles each row. The rotation is computed once per block, and no rotated
     input is ever written to memory.
   - **Multiply.** Each warp then decodes the tiles into B fragments and
     multiplies with `mma.sync.m16n8k16` (FP16, FP32 accumulate). It loads the
     next k tile's 8 trellis words while the current one is multiplied.
   - **Output.** FP32 partial sums per K split.
3. **Epilogue.** One warp per (pair, 128 block):
   - the splits are summed in order and rotated back, then × `svh`;
   - GLM's clamped SwiGLU follows, with the BF16 roundings the reference model
     makes when it runs in BF16 (TensorFold's choice);
   - then × down `suh`, the rotation and FP16: the down input.
4. **Down.** Each block is one (group, 512-column chunk of the output, K
   split), with the same tile decoding and multiply.
5. **Reduce.** One warp per (row, 128 block):
   - for each slot in order, the splits are summed, rotated back and × `svh`;
   - the result is multiplied by the gate weight and added in FP32 (a multiply
     then an add);
   - the sum is rounded to BF16, and non-finite sums are faulted.

**Sources.** The tile decoder, the MMA wrapper and the warp butterfly are
TensorFold's (`exl3.cu`, verbatim). The block structure changes for TP4 and
for prefill; see the file header and PROVENANCE.md.

**Configurations.** A configuration is (MT 16-row tiles per group, gate/up K
splits, down K splits):

- `(1, 8, 2)` up to 64 rows (decode and verify windows);
- `(2, 1, 1)` above 64 rows.

Every row's arithmetic is independent of the other rows: the tensor-core rows
are independent, and every sum runs in a fixed order. **Under one
configuration a row gets the same bits whatever else is in the batch.** The
test checks one row alone, in a window of 8 and in a window of 64. Prefill-size
calls use a different K split, so they are not bitwise equal to decode-size
calls.

**Build flags.** The kernels build with `--fmad=false`, so the rotations and
epilogues use the same separately rounded operations as the CPU reference.

**Memory per call** (grow-only scratch):

- at 4,096 rows, about 0.76 GB: the down partials are 0.54 GB (32,768 pairs ×
  4,096 × FP32);
- at decode sizes, a few MB.

The layer images use 0.91 GB each, 38.4 GB for 42 layers.

**Correctness** (`tests/cuda_kernel.rs`, one test function holding one layer
on the GPU):

- **Stage by stage, each stage fed the GPU's own input:**
  - gate/up and down products against f64: at most 5e-6 of the RMS at decode
    sizes and 4e-5 at prefill sizes. This is FP32 accumulation over K = 4,096.
  - the epilogue: bit for bit with the BF16 SwiGLU. With the FP32 SwiGLU up to
    4 FP16 values in 10^4 differ by one step, because `expf` differs by an
    ulp.
  - the final reduce: bit for bit.
- **End to end against the kernel-order reference:** RMS ≤ 2e-3 of the row's
  RMS at 1–8 rows and at 512 and 4,096 rows. Those one-ulp accumulation
  differences occasionally tip a BF16 rounding.
- **Against the dequantized FP32 model reference** (model semantics, BF16
  SwiGLU): RMS about 2e-3 at 1–8 rows.
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

RTX 4090 (`sm_89`, 128 SMs, about 1 TB/s), shared, 28 September 2026.
Command: `cargo run -p glm53f-rank --release --features cuda --example exl3_bench -- --sweep`.

Weights are synthetic, at the real shape. Routes are uniform over the 288
experts. GB/s counts the bytes of the distinct experts' rank shares
(3,173,376 each) over the GPU time from plan to reduce.

| Rows | Distinct experts | GPU time | Rate | Phases (plan, gate/up, epilogue, down, reduce) |
|---:|---:|---:|---:|---|
| 1 | 8 | 0.056 ms | **451 GB/s** | 0.004, 0.024, 0.005, 0.014, 0.010 ms |
| 2 | 16 | 0.079 ms | 620 GB/s | |
| 4 | 30 | 0.132 ms | 726 GB/s | |
| 8 | 58 | 0.232 ms | **802 GB/s** | 0.004, 0.143, 0.006, 0.070, 0.010 ms |
| 64 | 230 | 0.918 ms | 824 GB/s | |
| 512 | | 1.33 ms | 385 K rows/s (wall 282 K) | |
| 1,024 | | 1.67 ms | 615 K rows/s (wall 401 K) | |
| 4,096 | | 5.33 ms | **768 K rows/s** (wall 455 K) | 0.072, 2.71, 0.19, 1.71, 0.65 ms |

- **The planner** takes 4 µs at decode sizes and 72 µs at 4,096 rows. It
  sorts on the GPU in one CTA, with a warp-shuffle scan over the 288 experts.
  A serial scan had cost 16 µs.
- **The 4096-row call** is 412 GFLOP of tensor-core work in 4.4 ms, about 93
  TFLOPS.
- **Wall time** includes the PCIe copies of the rows in and the BF16 out. On a
  Spark, whose memory is unified, those copies are memory to memory.
- **The sweep** found `sk 4, skd 1` up to 1.05× faster than the default at 4–64
  rows on this 128-SM card. A Spark (48 SMs) needs its own sweep.

## On a Spark

The Sparks were busy during this work, so nothing here has run on GB10 yet.
When one is free:

```sh
# On the Spark (arm64, CUDA 13).
export GLM53F_CUDA_ARCH=sm_121        # GLM53F_CUDA_LIB if libcudart is not in /usr/local/cuda/lib64
cargo test  -p glm53f-rank --release --features cuda --test cuda_kernel -- --nocapture --test-threads=1
cargo run   -p glm53f-rank --release --features cuda --example exl3_bench -- --sweep
cargo build -p glm53f-rank --release --features cuda,rdma

# Once per rank: cut its share and serve it. --peers lists the four ranks' peer-mesh
# addresses in rank order (the same list on every rank) for the prefill reduce-scatter.
target/release/glm53f-rank slice --checkpoint <exl3-checkpoint> --rank R --out <rank-dir> --source "<repo>@<revision>"
GLM53F_RDMA=1 target/release/glm53f-rank serve --rank R --dir <rank-dir> --listen <fabric-address>:8600 \
  --peers <fabric-0>:8601,<fabric-1>:8601,<fabric-2>:8601,<fabric-3>:8601
```

**Expected results:**

- **Kernel test.** It should pass with the same tolerances: the arithmetic is
  the same, and only the tensor cores' accumulation rounding could differ.
- **Decode read rate.** The phase-1 gate is at least 200 GB/s at 1–8 rows.
  TensorFold's kernel, with the same tile decoder, reads 208–220 GB/s on GB10.
  At 230 GB/s:
  - one row reads 25.4 MB a layer, about 0.11 ms plus the fixed cost, about
    20–30 µs on the 4090;
  - 8 rows read about 184 MB, about 0.8 ms.
- **Prefill.**
  - Work: about 412 GFLOP per rank per layer at 4,096 rows.
  - Speed: GB10's sustained FP16 tensor-core rate (FP32 accumulate) is not
    measured here. At 60–100 TFLOPS, 4,096 rows take 4–7 ms a layer.
  - Throughput: a 4,096-token chunk then clears 42 layers in 0.17–0.3 s, above
    the design's Spark-bound target of about 5K tokens/s.
- **Split counts.** Sweep `sk` and `skd` at 48 SMs, and set the defaults from
  the sweep (`g53r_default_cfg`).

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

**Over TCP** the same code runs on owned buffers.

**Failures.** A failed request gets a `Status::Error` return and the
connection closes. The daemon keeps listening: it fails the request, not the
rank.

**Flags and environment variables:**

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
  exchange (the coordinator's other prefill lane, which a faster peer may
  already be sending) wait until their exchange claims them. The coordinator
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

1. **Spark numbers.** The Spark numbers are unmeasured: kernel test, read rate,
   prefill and split sweep, as listed in [On a Spark](#on-a-spark). The 4090
   is a proxy.
2. **KL gates.** Three numerics choices need them:
   - the wire's FP8 rows, where BF16 would be a protocol change;
   - the BF16-rounded SwiGLU versus FP32 (`fp32_swiglu`);
   - the reduce-scatter's exchange dtype.
3. **Channel order.** The EXL3 checkpoint reorders each expert's intermediate
   channels relative to the official checkpoint. A rank's share is correct for
   that checkpoint only; formats must never be mixed across ranks.
4. **Batch invariance.** Prefill-size calls (above 64 rows) use another K
   split than decode-size calls. A row is bitwise batch-invariant within each
   regime, not across them.
5. **The reduce-scatter** runs over TCP on one machine, through the forward
   too (glm53f-forward's `RemoteExperts`). Still to do: the peer mesh over
   RDMA on the target hardware (built, untested), and its timings there
   against the four-plane return.
6. **Performance** is not tuned beyond the split sweep. Candidates:
   - graphs for the fixed decode shapes;
   - double-buffered rotation tiles;
   - writing the output straight into the send buffer through mapped memory;
   - a persistent decode kernel.
7. **Layer 45** (MTP experts) is supported by the slicer, the manifest and the
   daemon, but has not been exercised with real data.
