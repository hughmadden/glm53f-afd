# Running glm53f-afd

Two binaries:

- **`glm53f-serve`**, the coordinator: one process on the GPU host (an RTX 5090, `sm_120`). It
  holds every non-expert weight, the KV cache and the sampler, and serves the
  OpenAI-compatible API.
- **`glm53f-rank`**, an expert rank: one daemon on each of the four DGX Sparks (GB10,
  `sm_121`). Each holds a quarter of every routed expert (EXL3 4-bit) and serves the MoE
  layers.

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
   GLM53F_WIRE_NOCRC=1 glm53f-rank serve --rank 0 --dir <rank-dir> --listen 192.0.2.10:8600   # ranks 1-3: .11, .12, .13
   ```

   It verifies every image (size and SHA-256), loads them onto the GPU and prints
   `listening on ...`. `GLM53F_WIRE_NOCRC=1` sends frames without their CRC32C; both sides must
   agree, and the coordinator's RDMA transport requires it.

2. The coordinator:

   ```sh
   GLM53F_RDMA=1 GLM53F_WIRE_NOCRC=1 glm53f-serve \
     --checkpoint <coordinator-dir> \
     --ranks 192.0.2.10:8600,192.0.2.11:8600,192.0.2.12:8600,192.0.2.13:8600 \
     --listen 0.0.0.0:8100
   ```

   Without `GLM53F_RDMA=1` the exchange runs over TCP on the same fabric addresses (with or
   without CRCs, as long as the ranks agree). It loads
   the weights, sizes the KV pool from the free GPU memory (less `--reserve-gib`, 4 by default),
   connects the ranks in rank order and prints `serving the API on ...`.

3. A request:

   ```sh
   curl -N http://<api-host>:8100/v1/chat/completions -H 'Content-Type: application/json' \
     -d '{"model":"glm-5.3-flash","messages":[{"role":"user","content":"Hello"}],"stream":true}'
   ```

`glm53f-serve --help` lists the options; the crate documentation (`crates/glm53f-serve/src/lib.rs`)
describes each.

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
# RemoteExperts against the oracle and the local FP8 experts, on four rank daemons.
GLM53F_CHECKPOINT_DIR=... GLM53F_EXPERTS_DIR=... GLM53F_RANK_BIN=.../glm53f-rank \
GLM53F_RANK_DIRS=<rank-0>,<rank-1>,<rank-2>,<rank-3> \
  cargo test --release -p glm53f-forward --features coordinator --test remote_experts -- --nocapture --test-threads=1

# One streamed chat completion through glm53f-serve in development mode.
GLM53F_CHECKPOINT_DIR=... GLM53F_RANK_BIN=... GLM53F_RANK_DIRS=... \
  cargo test --release -p glm53f-serve --features cuda --test dev_mode -- --nocapture
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

**Serving shell:** `GLM53F_QUEUE_DEPTH` and `GLM53F_QUEUE_WAIT_MS` (the request queue),
`GLM53F_HOST_CACHE_GB` (the host RAM tier for KV snapshots; 0 turns it off; by default the
smaller of 32 GiB and 40% of the available RAM), `GLM53F_PREFILL_SEGMENT_MS`,
`GLM53F_PREFIX_CACHE_ENTRIES`, and `GLM53F_SPEC`, `GLM53F_SPEC_POLICY`, `GLM53F_SPEC_TAU`,
`GLM53F_SPEC_COST_A`, `GLM53F_SPEC_COST_B` (speculation, once a drafter exists).

**Expert wire:** `GLM53F_RDMA=1` (coordinator: RDMA RC, in an `rdma` build; the ranks follow the
coordinator's handshake), `GLM53F_WIRE_NOCRC=1` (frames without CRC32C; both sides must agree;
RDMA requires it), `GLM53F_WIRE_MIN_GBPS` (the fabric's floor rate, default 100),
`GLM53F_WIRE_ALLOW_LAN=1` (admit a non-fabric network: tests only), `GLM53F_TIMELINE=1`
(cross-host timeline events), `GLM53F_PROFILE=1` (per-exchange timings on the coordinator).

**Rank:** `GLM53F_RANK_TRACE=1` (a timing line per request), `GLM53F_RANK_DUMP_FRAME=<path>`
with `GLM53F_RANK_DUMP_LAYER` (write one request frame for offline replay).

**Tests:** `GLM53F_CHECKPOINT_DIR`, `GLM53F_EXPERTS_DIR`, `GLM53F_GOLDENS` (default
`oracle/goldens`), `GLM53F_TOKENIZER`, `GLM53F_EXL3_DIR`, `GLM53F_FP8_DIR`, `GLM53F_RANK_BIN`,
`GLM53F_RANK_DIRS`. Tests skip, and say why, when theirs are missing.
