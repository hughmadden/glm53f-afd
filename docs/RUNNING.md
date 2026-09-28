# Running glm53f-afd

Two binaries:

- **`glm53f-serve`**, the coordinator: one process on the GPU host (an RTX 5090, `sm_120`). It
  holds every non-expert weight, the KV cache and the sampler, and serves the
  OpenAI-compatible API.
- **`glm53f-rank`**, an expert rank: one daemon on each of the four DGX Sparks (GB10,
  `sm_121`). Each holds a quarter of every routed expert (EXL3 4-bit) and serves the MoE
  layers.

A third, **`glm53f-score`**, takes the coordinator's place for the KL gate: it loads the same
weights, connects the same ranks and writes teacher-forced logits instead of serving (below, and
[KL-GATE.md](KL-GATE.md) section 4).

The expert exchange runs over RoCE v2 RDMA, on a port of at least 100 Gb/s; both sides refuse
other networks. The API can use any network.

**Status.** The coordinator has run end to end against four rank daemons on one development GPU
(an RTX 4090, `sm_89`) over TCP loopback. Its `sm_120` build compiles but has not yet run on an
RTX 5090, and its RDMA path has not yet run on a fabric.

## Build

Rust (edition 2021) and the CUDA toolkit. No external crates. The kernels are compiled ahead of
time for one architecture, chosen by environment variables:

| Variable | Default | What |
|---|---|---|
| `GLM53F_CUDA_ARCH` | `sm_89` | The GPU the kernels are built for: `sm_120` for the RTX 5090, `sm_121` for a DGX Spark |
| `GLM53F_NVCC` | `/usr/local/cuda/bin/nvcc` | The `nvcc` to run |
| `GLM53F_CUDA_LIB` | `/usr/local/cuda/lib64` | The directory holding `libcudart` and `libcublas` |

The coordinator, on any x86-64 machine with CUDA 12.8 or later (the build needs no GPU):

```sh
GLM53F_CUDA_ARCH=sm_120 cargo build --release -p glm53f-serve --features cuda,rdma
```

At run time it needs the CUDA 12 runtime and cuBLAS (`libcudart.so.12`, `libcublas.so.12`,
`libcublasLt.so.12`) and, for `rdma`, `libibverbs.so.1` (rdma-core).

A rank, on a Spark (arm64; `sm_121` needs a CUDA 13 toolkit):

```sh
GLM53F_CUDA_ARCH=sm_121 cargo build --release -p glm53f-rank --features cuda,rdma
```

Each binary runs only on the architecture it was built for; the rank checks this at start.

## Weights

- **Coordinator:** the non-expert tensors of the official FP8 checkpoint
  (`zai-org/GLM-5.3-Flash`), with its `config.json`, `tokenizer.json` and `chat_template.jinja`.
  The whole checkpoint works too. To fetch only the coordinator's tensors:

  ```sh
  scripts/fetch_tensors.py --repo zai-org/GLM-5.3-Flash --revision <sha> --out <coordinator-dir> --select nonexpert
  ```

- **Ranks:** each rank's share, cut from the EXL3 checkpoint into a rank directory (one image
  per MoE layer and a manifest the daemon verifies at start):

  ```sh
  glm53f-rank slice --checkpoint <exl3-checkpoint> --rank R --out <rank-dir> --source "<repo>@<revision>"
  ```

  A rank directory holds about 38 GB (layers 3 to 44). All four ranks must be cut from the same
  checkpoint.

## Start

1. Each rank, on its fabric address:

   ```sh
   GLM53F_WIRE_NOCRC=1 glm53f-rank serve --rank 0 --dir <rank-dir> --listen 192.0.2.10:8600 \
     --peers 192.0.2.10:8601,192.0.2.11:8601,192.0.2.12:8601,192.0.2.13:8601   # ranks 1-3: .11, .12, .13
   ```

   It verifies every image (size and SHA-256), loads them onto the GPU and prints
   `listening on ...`. `GLM53F_WIRE_NOCRC=1` sends frames without their CRC32C; both sides must
   agree, and the coordinator's RDMA transport requires it. `--peers` (the same list on every
   rank) joins the ranks' mesh for the prefill reduce-scatter; `GLM53F_RDMA=1` on the ranks makes
   the mesh RDMA RC, TCP otherwise (`crates/glm53f-rank/README.md`).

2. The coordinator:

   ```sh
   GLM53F_RDMA=1 GLM53F_WIRE_NOCRC=1 glm53f-serve \
     --checkpoint <coordinator-dir> \
     --ranks 192.0.2.10:8600,192.0.2.11:8600,192.0.2.12:8600,192.0.2.13:8600 \
     --listen 0.0.0.0:8100
   ```

   Without `GLM53F_RDMA=1` the exchange runs over TCP on the same fabric addresses (with or
   without CRCs, as long as the ranks agree). It loads the weights (and with `--drafter` the
   drafter's), connects the ranks in rank order, allocates every buffer a pass uses (both
   prefill lanes' scratch for `--prefill-rows`, the verify scratch, the attention workspaces,
   the expert exchange's buffers, the drafter's tap buffer and working memory), then sizes the
   KV page pool from what is left less `--reserve-gib` (1 by default), logs what it allocated
   for what, and prints `serving the API on ...`.

3. A request:

   ```sh
   curl -N http://<api-host>:8100/v1/chat/completions -H 'Content-Type: application/json' \
     -d '{"model":"glm-5.3-flash","messages":[{"role":"user","content":"Hello"}],"stream":true}'
   ```

`glm53f-serve --help` lists the options; the crate documentation (`crates/glm53f-serve/src/lib.rs`)
describes each.

### Prefill lanes and device memory

- **Two lanes.** A prefill pass of `--prefill-rows` rows (4,096 by default) runs in
  `--prefill-lanes` lanes (2 by default): lane B's attention runs while the ranks compute lane
  A's routed experts, and the other way round. Over RDMA two exchanges are in flight (the ranks
  hold lane B's request while they compute lane A's); over TCP one. A lane is one exchange, at
  most 4,096 rows, so `--prefill-rows 8192 --prefill-lanes 2` gives two lanes of 4,096 and
  `--prefill-lanes 1` the serial pass. `crates/glm53f-serve/src/lib.rs` explains the default.
  With `--drafter` each lane captures the drafter's taps for its own rows, and the rows reach
  the drafter's context lane by lane, as two one-lane passes would give them.
- **Memory.** Nothing a pass, a draft or an append to the drafter's context uses is allocated
  after start-up. Snapshot marks (the KDA states
  of a prompt or turn end, 141 MiB) take pages of the KV pool, which admission counts, so a
  short pool evicts retained snapshots, skips a snapshot or refuses a request; it never runs the
  device out of memory in a pass. The start-up log's `device memory` lines list the weights, the
  forward's buffers (each lane, verify, workspaces, GEMM), the drafter's weights, tap buffer,
  working memory and per-slot ring, the expert exchange, the slots' state, the pool (with what
  the snapshot banks would take of it if full) and what was left free.
- **Tracing a pass.** With `GLM53F_PROFILE=1` each prefill pass prints a `PIPE` line: per MoE
  layer (median), the wall time, each lane's GPU time for its attention and its shared expert,
  the host's time waiting for the routes, in `submit` and in `finish` (blocked on the ranks,
  then the returns placed), and each lane's exchange as the coordinator sees it (`submit`
  returned to `finish` returned: about the other lane's work when the exchange is hidden). It
  ends with the wire's record of the pass: which request and return paths served, and per
  return path (row slices, four planes) the exchanges and the medians of the host's time in
  `submit` (of it, waiting for the device's copies), in `finish` waiting for the returns, and
  placing them. The ranks' own time per request comes from `GLM53F_RANK_TRACE=1` on the ranks.
- **The expert exchange's paths.** Prefill exchanges from `GLM53F_ROW_SHARDED_MIN_ROWS` rows
  are reduce-scattered by the ranks (each returns its quarter of the rows, summed; needs
  `--peers` on the ranks); decode and verify windows keep the four-plane return. Requests are
  built from the GPU into the page-locked request body, and returns read from the page-locked
  receive buffers (summed in place at decode sizes). Each has an off switch for comparisons;
  the results do not depend on them (`crates/glm53f-forward/src/remote.rs`).

## The KL gate

`glm53f-score` runs the engine's side of the gate against the BF16 teacher panel
([KL-GATE.md](KL-GATE.md)): for each window of the plan `harness/klgate.py plan` writes, a fresh
slot, the raw token ids in passes of `--pass-rows` rows, and the plan's rows' full logits written
to `<out>/<window>.safetensors`, with `<out>/run.json`. The ranks serve one coordinator at a time,
so stop `glm53f-serve` first:

```sh
GLM53F_CUDA_ARCH=sm_120 cargo build --release -p glm53f-score --features cuda,rdma
GLM53F_RDMA=1 GLM53F_WIRE_NOCRC=1 glm53f-score --checkpoint <coordinator-dir> \
  --ranks 192.0.2.10:8600,192.0.2.11:8600,192.0.2.12:8600,192.0.2.13:8600 \
  --plan plan.json --pass-rows 4096 --out engine-4096   # then --pass-rows 8 --out engine-8
```

The gate runs both pass sizes (the prefill path, and 8 rows or fewer: the decode path);
[KL-GATE.md](KL-GATE.md) section 4.3 has the whole sequence, from `klgate.py plan` to `compare`.
`--experts local` runs the official FP8 experts on the coordinator's GPU instead of the ranks;
`--dev-layers`, `--dev-load-layers` and `--experts zero` make a development run on one GPU, whose
logits are meaningless.

## Development on one GPU

- `--experts local` runs the official FP8 experts on the coordinator's GPU, loaded on demand
  from `--experts-dir` (a checkpoint holding the experts of the layers run).
- `--dev-layers 0-N` runs decoder layers 0 to N only, then the head, so the whole serving loop
  runs with a slice of the weights. **The output is meaningless text by design.**
- Four ranks can share one GPU over loopback: cut layers 3 and 4 for each rank
  (`--layers 3-4`), start each with `--listen 127.0.0.1:0 --allow-partial`, then point
  `glm53f-serve --dev-layers 0-4 --ranks ...` at the four printed addresses. Each rank with two
  layers takes about 2.1 GiB.

The tests that do this themselves:

```sh
# RemoteExperts against the oracle and the local FP8 experts, on four rank daemons; the row-sharded
# return and the fast paths against four planes and the host paths, per call and in two-lane prefill.
GLM53F_CHECKPOINT_DIR=... GLM53F_EXPERTS_DIR=... GLM53F_RANK_BIN=.../glm53f-rank \
GLM53F_RANK_DIRS=<rank-0>,<rank-1>,<rank-2>,<rank-3> \
  cargo test --release -p glm53f-forward --features coordinator --test remote_experts -- --nocapture --test-threads=1

# Two-lane prefill against one lane (bit for bit against the same rows as two passes) and the
# goldens; admission and snapshots under a tight pool.
GLM53F_CHECKPOINT_DIR=... GLM53F_EXPERTS_DIR=... \
  cargo test --release -p glm53f-forward --features coordinator --test lanes --test admission -- --nocapture --test-threads=1

# One streamed chat completion through glm53f-serve in development mode.
GLM53F_CHECKPOINT_DIR=... GLM53F_RANK_BIN=... GLM53F_RANK_DIRS=... \
  cargo test --release -p glm53f-serve --features cuda --test dev_mode -- --nocapture

# GlmForward::score against the forward's own passes; glm53f-score in development mode through
# harness/klgate.py (with the fetched teacher subset, its first window too).
GLM53F_CHECKPOINT_DIR=... \
  cargo test --release -p glm53f-forward --features cuda --test score -- --nocapture
GLM53F_CHECKPOINT_DIR=... GLM53F_EXPERTS_DIR=... [GLM53F_KL_TEACHER=<teacher-dir>] \
  cargo test --release -p glm53f-score --features cuda --test plumbing -- --nocapture
```

## Environment

**Coordinator** (each has a flag; the flag wins):

| Variable | Flag | What |
|---|---|---|
| `GLM53F_CHECKPOINT_DIR` | `--checkpoint` | The coordinator's weights |
| `GLM53F_TOKENIZER` | `--tokenizer` | `tokenizer.json` (default: in the checkpoint) |
| `GLM53F_SPARK_ADDRS` | `--ranks` | The four ranks, `host:port` in rank order |
| `GLM53F_EXPERTS_DIR` | `--experts-dir` | `--experts local`: the routed experts' checkpoint |
| `GLM53F_API_ADDR` | `--listen` | The API's address (default `127.0.0.1:8100`) |
| `GLM53F_MAX_SLOTS` | `--slots` | Requests with device state at once (default 16) |
| `GLM53F_DFLASH_DIR` | `--drafter` | The DFlash2 drafter: speculative decoding, up to 7 drafts a step (needs decoder layers 0-43) |
| `GLM53F_PREFILL_ROWS` | `--prefill-rows` | Rows of one prefill pass, every lane's together (default 4,096) |
| `GLM53F_PREFILL_LANES` | `--prefill-lanes` | Lanes of a prefill pass, 1 or 2 (default 2) |

**Serving shell:** `GLM53F_QUEUE_DEPTH` and `GLM53F_QUEUE_WAIT_MS` (the request queue),
`GLM53F_HOST_CACHE_GB` (the host RAM tier for KV snapshots; 0 turns it off; by default the
smaller of 32 GiB and 40% of the available RAM), `GLM53F_PREFILL_SEGMENT_MS`,
`GLM53F_PREFIX_CACHE_ENTRIES`, and `GLM53F_SPEC`, `GLM53F_SPEC_POLICY`, `GLM53F_SPEC_TAU`,
`GLM53F_SPEC_COST_A`, `GLM53F_SPEC_COST_B` (speculation, with `--drafter`); the daemon reads
`GLM53F_DFLASH_SAMPLED_WALK` (0: sampled requests draft with the greedy walk too).

**Expert wire:** `GLM53F_RDMA=1` (coordinator: RDMA RC, in an `rdma` build; the ranks follow the
coordinator's handshake), `GLM53F_WIRE_NOCRC=1` (frames without CRC32C; both sides must agree;
RDMA requires it), `GLM53F_WIRE_MIN_GBPS` (the fabric's floor rate, default 100),
`GLM53F_WIRE_ALLOW_LAN=1` (admit a non-fabric network: tests only), `GLM53F_WIRE_INFLIGHT=1` (one
exchange in flight over RDMA too, instead of two: the prefill lanes then take turns on the wire),
`GLM53F_TIMELINE=1` (cross-host timeline events), `GLM53F_PROFILE=1` (per-exchange timings on the
coordinator, and a `PIPE` line per prefill pass). The return path: `GLM53F_ROW_SHARDED_MIN_ROWS=N`
(exchanges of N rows and more, at least 4, reduce-scattered by the ranks; unset or 0: four planes
always; the design's value is 16) with `GLM53F_EXCHANGE_DTYPE` (`bf16`, the default, or `fp8`,
which needs the KL gate). The coordinator's fast paths, on by default:
`GLM53F_WIRE_DEVICE_ENCODE=0` (requests encoded on the host instead of copied from the GPU into
the request body), `GLM53F_WIRE_ZERO_COPY=0` (returns uploaded from pageable memory instead of
read from the page-locked receive buffers), `GLM53F_WIRE_FILL_ROWS=N` (requests of up to N rows
written by the frame-fill kernel instead of the copies; default 0) and
`GLM53F_WIRE_MAPPED_ROWS=N` (four-plane returns of up to N rows summed in place; default 64).

**Rank:** `GLM53F_RANK_TRACE=1` (a timing line per request), `GLM53F_RANK_DUMP_FRAME=<path>`
with `GLM53F_RANK_DUMP_LAYER` (write one request frame for offline replay).

**Tests:** `GLM53F_CHECKPOINT_DIR`, `GLM53F_EXPERTS_DIR`, `GLM53F_GOLDENS` (default
`oracle/goldens`), `GLM53F_TOKENIZER`, `GLM53F_EXL3_DIR`, `GLM53F_FP8_DIR`, `GLM53F_RANK_BIN`,
`GLM53F_RANK_DIRS`, `GLM53F_KL_TEACHER` (the teacher subset `klgate_fetch.py` writes),
`GLM53F_PYTHON` (default `python3`). Tests skip, and say why, when theirs are missing.
