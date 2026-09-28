# glm53f-rank: provenance

Every unit this crate copied, ported, adapted or reimplemented from another
project. Rows are consolidated into `docs/REUSE.md`. The sha256 column is the
digest of the source file at the named commit.

**Sources**

- **mimo26f-afd:** [hughmadden/mimo26f-afd](https://github.com/hughmadden/mimo26f-afd)
  v1.2.0, commit `bab9fa2f2fc1e22ae67b56fbc1c209278f6a9d79` (MIT, the same
  licence and holder as this repository; `LICENSE.mimo26f-afd`).
- **TensorFold:** [ashhart/TensorFold](https://github.com/ashhart/TensorFold)
  v0.3.4.1, commit `bb4b4a35863af562fc4ccb2586300d8f94b5d6de` (MIT, Copyright
  (c) 2026 TensorFold contributors; `LICENSE.tensorfold`). TensorFold's EXL3
  code reads the format of [ExLlamaV3](https://github.com/turboderp-org/exllamav3)
  (MIT, Copyright (c) 2025 Turboderp): its trellis layout, its "mcg" codebook
  and its tensor-core fragment order. No ExLlamaV3 code is copied here.
- **glmrt:** [tpurtell/glmrt-5.3-1rtx-4spark](https://github.com/tpurtell/glmrt-5.3-1rtx-4spark)
  v9, commit `dc6d9b8e1600e001cb1d4228bd911f4df8091f99` (MIT). Read for
  designs only; no code copied.
- **transformers:** `src/transformers/models/glm5_next/modeling_glm5_next.py`
  at `7cd73d9df0` (Apache-2.0). Read for semantics only.

**Renames applied to every file ported from mimo26f-afd** (not repeated in the
Delta column):

- `mimo26_wire` → `glm53f_wire` and `mimo26_rdma` → `glm53f_rdma`;
- the environment prefix `MIMO26_` → `GLM53F_`. The trace switch becomes
  `GLM53F_RANK_TRACE` and the timeline switch `GLM53F_TIMELINE`.

"Spark" in names became "rank" where it named the role (`SparkServer` →
`RankServer`). **Kept on purpose:** the RDMA handshake magic `M26RDMA1` (in
`glm53f-rdma`), a protocol value shared with the coordinator.

## Ported from mimo26f-afd

All from `crates/mimo26-spark/` unless the path says otherwise.

| Unit | Source path | sha256 (source file) | Here | Delta | Pinned by |
|---|---|---|---|---|---|
| Byte transport, loopback, fault injector, TCP and RDMA RC transports | `src/transport.rs` | `bef50c4ec339bb168433cb5f251eb348c2c56a0e2cd2c58786835afee7dd32b4` | `src/transport.rs` | Renames; two doc comments point to the source project instead of its internal design notes; an unused test variable prefixed with `_`. For the peer mesh: the RDMA transport's slot sizes are parameters (`accept_sized`), it has the connecting side of the handshake (`connect_sized`, after the coordinator's `rdma_setup` in `crates/mimo26-coordinator/src/wire.rs`) and a non-blocking `try_recv`. Queued requests (written here, 2026-09-29): the RDMA transport posts `slots` receive slots instead of two (`accept`, `accept_sized`, `connect_sized` take the count; `RECV_SLOTS`, 4, for the coordinator's connection), keeps their bookkeeping in `RecvSlots` (posted oldest first, the one held, a completion checked against the oldest posted slot, a released slot posted again at the back) and names its slots in the handshake reply (`glm53f_rdma::Info::recv_slots`) | `transport::tests` (11, with `requests_queue_in_the_receive_slots_and_are_served_in_order` and `completions_out_of_order_or_while_a_slot_is_held_are_refused`); `--features rdma` builds |
| Wire server loop | `src/server.rs` | `40f4bb6ffd848d4e796ed2dca18f7fb868dd30b3c9932a979a1f335b5da51e75` | `src/server.rs` | `SparkServer` → `RankServer`. `step` takes an `ExpertKernel` and the prepared `Layers` instead of an FFN closure and one grouped image. A missing layer is a serve error. Tests use a stand-in kernel. | `server::tests` (2) |
| Serve paths (owned, in-place return, zero-copy view) | `src/serve.rs` | `63b39227572267dc85c8b65b71547472a02be91981f3ace5a456cf13d1659c66` | `src/serve.rs` | The B1 paths (`serve_b1_into`, `serve_b1_view`, `serve_return_b1`) become `serve_into`, `serve_view` and `serve_return` over `ExpertKernel`. The host FP8 decode and the MXFP4 CPU oracle are dropped. Adds `Layers`. `return_meta`/`return_meta_view` are kept. `serve_view_f32` serves the FP32 rows of the reduce-scatter. | `serve::tests` (3), `tests/daemon.rs` |
| Route table | `src/route.rs`, and `serve.rs` (`b1_inputs`, the route loop of `serve_b1_view`) | `c9ddbad53c4ba76e6f8309d2091c14bfc5cd793575463a05e7ddc3429e2732ee` (route.rs), `63b39227…` (serve.rs) | `src/route.rs` | Rewritten. The MXFP4 per-M tiling (`tile_m`, padded replicate/reduce) is gone. Kept: the route-shape checks (8 per row, row index = row) as `request_routes`/`view_routes`. New: `GroupPlan`, the CPU twin of the device planner, and `max_groups`. | `route::tests` (2) |
| Boot identity readback | `src/boot.rs`, with the checks of `crates/mimo26-repack/src/identity.rs` | `df64dfecea50a7e964f81f155b984ddf5294a1bdf0af051368d47a69bcd46d16`, `6b0e56b07c7a33b1735d7e8c48e121ce8cc0168d0dbb32864ac5c9283aa9907d` | `src/boot.rs` | One image per layer instead of one file per expert slice. The checks are the same (size, sha256, missing, unlisted, every failure named), plus rank, layout and world. Files are hashed in parallel. `Expect` carries the image size, so tests use small images. | `tests/boot.rs` (4) |
| Resident index | `src/resident.rs` | `98d0772985768c699a6ce1d0548e391838c84ef54544fb8e772d64628df60a9a` | `src/resident.rs` | Indexes layer images instead of slice files. The grouped image is replaced by `layer_image`. New: `write_rank_dir`, the slicer. | `tests/boot.rs`, `tests/real_experts.rs` |
| Manifest | `crates/mimo26-repack/src/manifest.rs` (design) | — | `src/manifest.rs` | Reimplemented: a line-based text manifest of layer images (file, size, sha256, layout, rank, world, source). No code copied. | `manifest::tests` (2) |
| SHA-256 and its vectors | `crates/mimo26-repack/src/sha256.rs`, `crates/mimo26-repack/tests/sha256_vectors.rs` | `7404872ca1df37e1385fdfc68eb35bb8309f2280ec82f4e02aab787329709d66`, `d4aafcbe11066265d64c042e852ca9ef670df2a649e61b56edaa9d3431921a55` | `src/sha256.rs`, `tests/sha256_vectors.rs` | Module doc names this crate; test comments and one input string renamed | `tests/sha256_vectors.rs` (6) |
| Compact return encoding | `src/wire.rs` | `351c6e1bd89d0591047d0b3352bd252b4a2935e1b6ba17df2f77a182393a5c42` | `src/wire.rs` | Renames; "Spark" → "rank" in two doc comments | `wire::tests` (2) |
| Device buffers | `src/device.rs` | `edfc7b5fb2888841be407a0785b4ecbeb68239c13cc97a945214823043abce69` | `src/device.rs` | `CudaError` imported from `cuda.rs` (the source's `ffi.rs` is dropped); `upload_prefix_async` marked `unsafe` with a safety note (it takes a raw stream) | builds with `cuda` |
| CUDA runtime bindings | `src/cuda.rs` | `5598da4d531e3cbebb63ce9bcb3cd58d055d2419d85d686892b540a81132f4ca` | `src/cuda.rs` | `CudaError`/`CudaStream` defined here (were in `ffi.rs`) | builds with `cuda` |
| Timeline and trace switches | `src/timeline.rs` | `96caad0ca841c7801ed4f5cc8b6e3c2b68dbbce16da6d1acb0626bb5eb1eabbe` | `src/timeline.rs` | Env names; a reference to an internal build note removed | — |
| Daemon | `src/main.rs` | `ce90a69f23ffcbf6fdcdf30acbee3435a91c6b13eaebc05ec2d0e376f3303485` | `src/main.rs` | Kept: boot readback; start-up device gate; fabric guard; RDMA-or-TCP accept; page-locked receive ring; zero-copy receive with the slot re-posted before the return; return written in place into the send buffer; error returns that fail the request, not the rank; per-request timing and window stats. Changed: one loop serves both transports through `RequestView` (the source had separate B1/B2 branches); the MXFP4 B2 and B1 paths become the `ExpertKernel`; layers are prepared eagerly by default (`--lazy` for on-demand); `--allow-partial`; `slice` and `verify` commands. The hard-coded frame-dump path is replaced by `GLM53F_RANK_DUMP_FRAME` (and `GLM53F_RANK_DUMP_LAYER`). New: the peer mesh (`--peers`, `--peer-timeout-ms` and their environment fallbacks) and the reduce-scattered requests (FP32 partial, exchange, row-slice return, an `exchange` trace line); `--recv-slots` (`GLM53F_RANK_RECV_SLOTS`, 1 to 16, 4 by default), the RDMA receive slots a coordinator connection queues requests in. | `tests/daemon.rs`, glm53f-coordinator `tests/row_sharded.rs` |
| Build script | `build.rs` | `158278df8b9b43d53d477fbe24b104883946c5e895503c75cd1f0fe97c7ee214` | `build.rs` | Compiles one kernel file (`kernels/exl3_rank.cu`) instead of the layout-v2 and B1 sets. Keeps: skip without `cuda`, archive, link `cudart` and `stdc++`, bake the target architecture. Env: `GLM53F_NVCC`, `GLM53F_CUDA_ARCH` (default `sm_89`), `GLM53F_CUDA_LIB`. Adds `--fmad=false`, `--ftz=false`, `--prec-div=true`. | builds with `cuda` |
| Kernel boundary shape | `src/b1.rs`, `kernels/b1_serve.cu` (`m26s_b1_layer_new`, `m26s_b1_ffn`, `Dev`, error strings) | `72624fa81ed4fde5eb65f3a1fcffab960148c1afb1e5332566b0e5dd40e05cda`, `f60ac76f2b8a294e022adbd29f60953eb9669e1cb6ef97e7f9ed52f1830d896b` | `src/kernel.rs`, `src/exl3_cuda.rs`, `kernels/exl3_rank.cu` (C ABI) | Same shape (a prepared layer, a scratch with stream and events, FP8 rows at pitches, top-8 ids and weights, BF16 out, stage timings, an error string); new EXL3 implementation; an FP32 output (`ffn_f32`, `g53r_ffn_f32`) for the reduce-scatter | `tests/cuda_kernel.rs` |
| One-CTA route planner | `kernels/b1_serve.cu` (`plan_parallel`) | `f60ac76f…` | `kernels/exl3_rank.cu` (`plan_kernel`) | Adapted to 288 experts and to groups of 16·MT pairs. The fault word is kept. The FC1 chunk list is dropped. The serial prefix sum becomes a warp-shuffle scan (16 → 4 µs a call). The host no longer reads the group count back: grids are sized by a bound and surplus blocks exit. (The plan inside the gate/up blocks, `self_plan`, is written here; see below.) | `tests/cuda_kernel.rs`, `route::tests` |
| Boot tests | `tests/boot.rs` | `edb85d290bd8a7d71277a324b8b240db74feaea2f44454973fd4b2b5d981bd5b` | `tests/boot.rs` | Rewritten for layer images; the same cases (clean, corrupt, unlisted, rank ownership), plus truncated, missing and partial | — |
| Licence | `LICENSE` | `bcf2864b2403249319c65320d79c708236f6fd46c61eee568a52ac1e8816e8fa` | `LICENSE.mimo26f-afd` | Verbatim | — |

**Dropped from the source crate:** the MXFP4 B1 and layout-v2 kernel paths
(`decode.rs`, `ffi.rs`, `b1.rs`, `kernels/b1_serve.cu`, `kernels/b1_fc1_m64.cuh`)
and the examples `b1_bench.rs` and `replay_frame.rs`.

## From TensorFold

All from `src/tensorfold/families/glm5_next/cuda/` unless the path says otherwise.

| Unit | Source path | sha256 (source file) | Here | Delta | Pinned by |
|---|---|---|---|---|---|
| Tile decoder (`mcg2`, `decode_tile`), MMA wrapper (`mma16816`), 128-point warp butterfly (`fwht128`) | `exl3.cu` | `959606bd73e32cf60beb665ad30ca6b36834fdbb4e71cb182cdbe60c789e9728` | `kernels/exl3_rank.cu` (marked section) | Verbatim | `tests/cuda_kernel.rs` (stage checks) |
| Grouped trellis GEMM, gate/up epilogue with GLM's clamped BF16 SwiGLU, down epilogue | `exl3.cu` (`grouped_kernel`, `gateup_epilogue_kernel`, `down_epilogue_kernel`), `exl3_mm.py` (`routed`) | `959606bd…`, `52350fc994fa3398afd82eb98ceabb70e94bdf67a22d9c3437cacad7f0108ba8` | `kernels/exl3_rank.cu` (the split kernels `gateup_kernel`, `down_kernel`; the epilogue and reduce arithmetic `epilogue_item`, `reduce_item`, also run by `gateup_epilogue`, `reduce_kernel` and the fused steps) | Adapted. Shapes: TP4 slices (N = 512 for gate/up, K = 512 for down), 288 experts. The input rotation is computed per block in shared memory straight from the FP8 wire rows; TensorFold used a separate rotation kernel over BF16 inputs into a buffer. Warps split N (the four 128-column Hadamard blocks), not K. Up to two 16-row tiles per block. The next k tile's weights are prefetched; a block's rows' rotation inputs are loaded together. The down epilogue is fused into the weighted slot reduce, which adds BF16 output and fault words and loads every slot's inputs before the first is used. An FP32 SwiGLU option. Groups come from the plan kernel or the gate/up blocks instead of TensorFold's Triton `_group`. | `tests/cuda_kernel.rs` |
| EXL3 format and reference decoder | `exl3.py` | `acdf6f0be5a2af09a92c905e9ba826f3083f752bbd7a7e42aeafc78ded69e077` | `src/exl3.rs` | Ported to Rust from the format description and the numpy code: codebook, tile positions, states, unpack, rotation, dequantize, forward. FP16 handled at the bit level (`src/half.rs`). | `tests/exl3_golden.rs`: bit for bit against `exl3.py` run unmodified (goldens computed from it) |
| EXL3 TP split rules | `split.py` (`EXL3_RULES`) | `f8d876285405b05f886bf7a7732d102054e649a35d2707d3bc2eaa54ad3df7f0` | through `glm53f_model::slicing` | The rule is implemented in `glm53f-model`; this crate places the pieces (`src/layout.rs`) | `tests/tp4_slicing.rs`, `tests/real_experts.rs` |
| Reference expert computation (for the epilogue semantics) | `tests/cuda/test_glm_exl3.py` | `050e5edf3f48100f9fb9acafa7700e069c3a5fc25750fbba51788a2313e1b907` | `src/reference.rs` (`gateup_epilogue`, `swiglu` with BF16 roundings) | Read for the rounding order | `tests/cuda_kernel.rs` |
| Licence | `LICENSE` (repository root) | `be6a9ee4462784a762b21790954b8e76c4dcacc9fb6da2a40d03a8a9bddec93e` | `LICENSE.tensorfold` | Verbatim | — |

## From glmrt (designs; reimplemented, no code copied)

| Unit | Source path | sha256 (source file) | Here |
|---|---|---|---|
| EXL3 TP4 geometry for GLM-5.3 (intermediate 2,048; `% (4 × 128)` check; per-rank rotation vectors) | `rust/crates/glmrt-loader/src/exl3_format.rs` | `e6072978cb63dd808c5de65dfeb0e72db1dff7733c593dc65fb1ec8b4e6f0198` | README "The TP4 split" (evidence) |
| TP4 trellis slicing at load (`queue_route_cuda_exl3_trellis_tp4`) | `rust/crates/glmrt-daemon/src/commands/real_full/sparse_mlp/route.rs` | `ad5fafcb70b7274528ff48f4160ba1ae36f0896a065be0f07ad72adeb8e7160c` | README "The TP4 split" (evidence) |
| Spark-side reduce-scatter over RDMA (pair rings, frame header, balanced row partition, FP8 row-scaled exchange) | `rust/crates/glmrt-daemon/src/commands/real_full/rdma_reduction.rs`, `.../intermediate_sharding.rs` (`balanced_row_partition`) | `6ba74c2588a5455b23342e44c814025916208030fc33a77583bd544bd1f31c21`, `19264cdd3823fa2646253d6fce1682abb795669e6c02603039958c4b6a609766` | `src/reduce_scatter.rs` (`Exchange`, `outgoing`, `place_frames`), `src/mesh.rs` (a link per rank pair), glm53f-wire `src/row_shard.rs` (the partition, the exchange frame), README "Prefill reduce-scatter" |
| FP8 E4M3 row-scaled packing and combine (`amax / 448` scale, SATFINITE, non-finite → 0) | `native/cuda/kernels/residual.cu` | `060784597185601b02c8d4d72753a35e5cd7113e790d0b6dd60b4715cd4eca59` | `src/reduce_scatter.rs` (`encode_row`, `decode_row`) |

## Semantics only

| Unit | Source | sha256 | Here |
|---|---|---|---|
| Routed experts (clamped SwiGLU, gate weights per slot, index-add), router (sigmoid top-8, normalization, × 2.5) | transformers @ `7cd73d9df0` : `models/glm5_next/modeling_glm5_next.py` (`Glm5NextTextExperts`, `Glm5NextTextTopkRouter`, `Glm5NextTextMoE`) | `4fe6ed7703e4f8f1dc7e3995af2b619058be8fec1f5157b7f512fc6f6150503f` | `src/reference.rs`, `src/consts.rs`, README "The kernel boundary" |

## Written here

- **Numerics:**
  - `src/half.rs`, FP16 and BF16 at the bit level;
  - `src/fp8.rs`, wire rows: E4M3 and UE8M0 decode, encode for tests, NaN
    refusal.
- **Layout:** `src/layout.rs`, image layout e1 and its offsets, cutting from
  whole tensors or a checkpoint through `glm53f-model`, and scale checks.
- **References and kernel boundary:**
  - `src/reference.rs`, the dequantized FP32 FFN and the kernel-order
    emulation;
  - `src/kernel.rs`, the `ExpertKernel` trait, `Rows`, route checks and the CPU
    backend;
  - `src/exl3_cuda.rs`, the CUDA backend and the test hook for intermediates.
- **Reduce-scatter:** `src/reduce_scatter.rs` (the exchange's arithmetic and
  bookkeeping) and `src/mesh.rs` (the peer mesh: hellos, TCP and RDMA links,
  exchanges keyed by request and layer, timeouts), apart from the designs
  credited above.
- **Page cache:** `src/pagecache.rs`, the drop of a layer image's cached pages (libc's `posix_fadvise`
  with `POSIX_FADV_DONTNEED`, declared in the file) once the image is on the device and after
  `verify`'s hashing, and the `MemAvailable` and `MemFree` figures of the boot log; the idea is from
  a public recipe's preflight (`docs/REUSE.md`), no code copied. Tests: `tests/pagecache.rs`
  (the pages leave the cache, counted with `mincore(2)`), `tests/boot.rs`. 29 September 2026.
- **Test inputs:** `src/testkit.rs`, synthetic layers, rows and routes, with
  scale magnitudes read from the published checkpoint.
- **Kernel file:** `kernels/exl3_rank.cu` outside the marked TensorFold
  section: the FP8 row decode, the in-block rotation, the grids and
  configurations, faults, the layer upload and scale scan, the C ABI; and,
  written for prefill and for the decode fixed phases (28 September 2026):
  - the large-M kernels `gateup_big` and `down_big` (64- or 32-row groups,
    8 or 16 MMA warps with optional rotation warps, double-buffered rotated
    rows, `cp.async` staging of the down input, a ring of trellis words
    loaded ahead, `ldmatrix` A fragments, skipped padding tiles) and their
    persistent scheduling (`launch_persistent`, work items in grid order);
  - the fused epilogue and reduce (`gateup_tail`, `down_tail`: the last
    block of a group or of a (row, chunk) runs the shared epilogue or reduce
    code; self-clearing arrival counters) and the L2 policies
    (`discard.global.L2` of consumed partial sums, an evict-first policy for
    trellis loads);
  - the plan inside the split gate/up blocks (`self_plan`, `block_scan128`),
    which repeats `plan_kernel`'s checks with each expert's pairs in route
    order;
  - the device helpers `ldsm_a`, `cp_async16`, `row_load`/`row_rotate`;
  - and, for the split kernels at decode sizes (29 September 2026): the ring
    of trellis words loaded 1, 2 or 4 k tiles ahead with branch-free refills
    and the evict-first policy as a template parameter (`pf`, `l2`), the
    down input staged in shared memory, the block orders that keep partial
    sums in L2 (`ord`), the L2 prefetches of scale vectors (`prefetch_l2`),
    the programmatic dependent launch of the down kernel with down blocks
    that plan themselves (`pdl`: `griddepcontrol` and
    `cudaLaunchKernelEx`, as documented for CUDA's runtime), and phase events
    recorded only after phases that ran a kernel.
- **Configuration policy:** `src/exl3_cuda.rs` (`Cfg` and its text form,
  `resolve_cfg`, `Policy` with the `GLM53F_RANK_*` environment).
- **Tests:**
  - `tests/tp4_slicing.rs`, `tests/cuda_kernel.rs`, `tests/reduce_scatter.rs`,
    `tests/mesh.rs`, `tests/real_experts.rs` and `tests/daemon.rs`;
  - the goldens in `tests/exl3_golden.rs`, computed by running TensorFold's
    `exl3.py`.
- **Benchmark:** `examples/exl3_bench.rs` (real layer images from a rank
  directory, taken in turn when several are given, configuration lists and
  sweeps, minimum or median times, a digest of each configuration's outputs).
