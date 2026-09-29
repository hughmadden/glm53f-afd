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

**Status (29 September 2026).** The whole engine runs on its target hardware: the coordinator on
an RTX 5090 and four ranks on DGX Sparks, over RDMA at 200 Gb/s ([PERFORMANCE.md](PERFORMANCE.md)
§0). The development setup below (one GPU, four rank daemons over TCP loopback) still runs every
model-path test.

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
  python3 scripts/fetch_tensors.py --repo zai-org/GLM-5.3-Flash --revision <sha> --out <coordinator-dir> --select nonexpert
  ```

- **Ranks:** each rank's share, cut from the EXL3 checkpoint into a rank directory (one image
  per MoE layer and a manifest the daemon verifies at start):

  ```sh
  glm53f-rank slice --checkpoint <exl3-checkpoint> --rank R --out <rank-dir> --source "<repo>@<revision>"
  ```

  A rank directory holds about 38 GB (layers 3 to 44). All four ranks must be cut from the same
  checkpoint. The EXL3 K4 checkpoint is `brandonmusic/GLM-5.3-Flash-tr3-4bpw` (mirrored at
  `Mia-AiLab/GLM-5.3-Flash-EXL3-TR3-4bpw`).

- **Drafter** (optional, `--drafter`): the DFlash2 checkpoint `incoai/GLM-5.3-Flash-DFlash2`
  (`config.json`, `model.safetensors`), on the coordinator.

## Start

1. Each rank, on its fabric address:

   ```sh
   GLM53F_WIRE_NOCRC=1 glm53f-rank serve --rank 0 --dir <rank-dir> --listen 192.0.2.10:8600 \
     --peers 192.0.2.10:8601,192.0.2.11:8601,192.0.2.12:8601,192.0.2.13:8601   # ranks 1-3: .11, .12, .13
   ```

   It verifies every image (size and SHA-256), loads them onto the GPU, gives back the page cache's
   copy of each (`memory ...` lines in the log show `MemAvailable` and `MemFree` on the way) and
   prints `listening on ...`. `GLM53F_WIRE_NOCRC=1` sends frames without their CRC32C; both sides must
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
   drafter's), connects the ranks in rank order, allocates every buffer a pass uses (every
   prefill lane's scratch for `--prefill-rows`, the verify scratch, the attention workspaces,
   the expert exchange's buffers, the drafter's tap buffer and working memory), then sizes the
   KV page pool from what is left less `--reserve-gib` (1 by default), logs what it allocated
   for what, and prints `serving the API on ...`. The API listens only from then on.

3. A request:

   ```sh
   curl -N http://<api-host>:8100/v1/chat/completions -H 'Content-Type: application/json' \
     -d '{"model":"glm-5.3-flash","messages":[{"role":"user","content":"Hello"}],"stream":true}'
   ```

`glm53f-serve --help` lists the options; the crate documentation (`crates/glm53f-serve/src/lib.rs`)
describes each.

### Health

`GET /health` (`curl -s http://<api-host>:8100/health`) is for health checks:

- **200** `{"status":"ok"}` while the engine can serve.
- **503** `{"status":"unavailable","reason":"..."}` once it cannot:
  - the expert wire failed: after any failed exchange the forward refuses every pass until the
    coordinator restarts and reconnects to the ranks;
  - or the scheduler's thread ended.

  Restart the coordinator in either case.
- It reads that state only, never the request queue or the GPU, so it answers at once under any
  load. A 200 does not prove that the next pass succeeds; only a request does.
- Before the engine is ready (loading, connecting the ranks, sizing the pool) nothing listens, so a
  probe's connection is refused: there is no "starting" answer.

### Numerics defaults

Each option changes the engine's arithmetic and became a default only after the KL gate and speed
runs on the target hardware ([KL-GATE.md](KL-GATE.md) §6). `glm53f-serve` and `glm53f-score` take the
same flags and variables, with the same defaults; the start-up line `[coordinator] numerics: ...`
names the options on.

| Option | Default | Off with | KL gate |
|---|---|---|---|
| BF16 KDA states (D8) | on | `--kda-state-f32`, `GLM53F_KDA_STATE_BF16=0` | passed (§6b) |
| Chunked KDA prefill | on | `--kda-chain-prefill`, `GLM53F_KDA_CHUNKED_PREFILL=0` | passed with W8A16 (§6d); failed without it (§6b) |
| W8A16 prefill projections | on | `--prefill-w8a8`, `GLM53F_PREFILL_W8A16=0` | passed with the chunked prefill (§6d) |
| FP8 KDA projections (D2) | off | (on with `--kda-fp8`) | failed (§6b) |
| D2 with power-of-two block scales | off | (on with `--kda-fp8-pow2`, `GLM53F_KDA_FP8_POW2=1`) | not yet gated (§6e) |
| D2 as MXFP8 (an E8M0 scale per row and 32 values of K) | off | (on with `--kda-mxfp8`, `GLM53F_KDA_MXFP8=1`) | not yet gated (§6e) |
| D2's KDA projections at W8A8 in prefill | off | (on with `--kda-prefill-w8a8`, with D2) | not gated alone; D2 failed (§6b) |

- The chunked KDA prefill and W8A16 passed as a pair, so turn them off together:
  `--kda-chain-prefill --prefill-w8a8`.
- With `--kda-state-f32` as well, every option is off: the engine's reference arithmetic, which
  section 6a of the KL gate scored.

### Prefill lanes and device memory

- **Lanes.** A prefill pass of `--prefill-rows` rows (8,192 by default) runs in
  `--prefill-lanes` lanes (4 by default, so lanes of 2,048 rows; 1 to 4): while the ranks compute one lane's routed
  experts, the GPU runs the next lanes' attention, the lanes in turn. Over RDMA one exchange per
  lane is in flight (the ranks queue the requests in their receive slots and compute them in
  order; a rank queues four by default, `--recv-slots`, and says so in the RDMA handshake); over
  TCP one. A lane is one exchange, at most 4,096 rows, so `--prefill-rows 8192 --prefill-lanes 2`
  gives two lanes of 4,096, `--prefill-rows 12288 --prefill-lanes 3` three, and
  `--prefill-lanes 1` the serial pass. A pass of R rows in N lanes cuts them evenly (lane i from
  row ceil(i R / N)); smaller passes take a lane per 64 rows. `crates/glm53f-serve/src/lib.rs`
  explains the default and when more lanes pay. With `--drafter` each lane captures the
  drafter's taps for its own rows, and the rows reach the drafter's context lane by lane, as
  one-lane passes of the same rows would give them.
- **Memory.** Nothing a pass, a draft or an append to the drafter's context uses is allocated
  after start-up. Snapshot marks (the KDA states
  of a prompt or turn end: 73 MiB with the default BF16 states, 141 MiB with `--kda-state-f32`) take
  pages of the KV pool, which admission counts, so a short pool evicts snapshots ([KV snapshots and
  the RAM tier](#kv-snapshots-and-the-ram-tier)), skips a snapshot or refuses a request; it never runs the
  device out of memory in a pass. The start-up log's `device memory` lines list the weights, the
  forward's buffers (each lane, verify, workspaces, GEMM), the drafter's weights, tap buffer,
  working memory and per-slot ring, the expert exchange, the slots' state, the pool (with the
  pages a snapshot mark takes) and what was left free.
- **Decode lanes.** `--decode-lanes MIN[-MAX]` runs decode and verify passes of MIN to MAX rows
  (over two requests or more) in the prefill's first two lanes, cut between requests (it needs
  `--prefill-lanes` 2 or more); `2-16` by default. Each
  lane is exactly a pass over its own requests (`crates/glm53f-forward/tests/decode_lanes.rs`),
  so it changes timing only. It overlaps one lane's coordinator work with the other's routed
  experts, but each lane reads the coordinator's weights and the ranks read the experts each
  lane's rows name, so two lanes of many rows read most experts twice: measure it.
- **Slots.** `--slots` (16 by default) sizes each slot's fixed state (about 113 MiB with the
  drafter and the default BF16 KDA states; 181 MiB with `--kda-state-f32`)
  and, with the drafter, the verify pass: every slot's window of 8 rows, capped by the step's row
  budget `GLM53F_SPEC_MAX_ROWS` (256 by default; the most likely drafts first), about 4.9 MiB a
  row. Both come out of the KV pool, and so does the chunked KDA prefill's workspace (on by
  default: 34 MiB a slot, 544 MiB at 16 slots, 1.59 GiB at 48). The start-up log states the largest
  request the pool admits and says plainly when a request of `--max-context` tokens (1,048,576 by
  default) does not fit. `crates/glm53f-serve/src/lib.rs` gives the plan at 16, 32 and 48 slots.
- **Tracing a pass.** With `GLM53F_PROFILE=1` each prefill pass prints a `PIPE` line: per MoE
  layer (median), the wall time, each lane's GPU time for its attention and its shared expert,
  the host's time waiting for the routes, in `submit` and in `finish` (blocked on the ranks,
  then the returns placed), and each lane's exchange as the coordinator sees it (`submit`
  returned to `finish` returned: about the other lanes' work when the exchange is hidden), and
  the calls in flight at most (`depth`). With N lanes the GPU stays busy while each lane's
  exchange takes at most the other N - 1 lanes' attention; time in `finish` is the GPU waiting
  for the ranks. It
  ends with the wire's record of the pass: which request and return paths served, and per
  return path (row slices, four planes) the exchanges and the medians of the host's time in
  `submit` (of it, waiting for the device's copies), in `finish` waiting for the returns, and
  placing them. The ranks' own time per request comes from `GLM53F_RANK_TRACE=1` on the ranks.
- **Profiling a prefill pass by operation.** With `GLM53F_PROFILE_OPS=1` the forward records a
  CUDA event after every operation of each prefill lane's attention sublayer and shared expert,
  and each prefill pass prints an `OPS` table after its `PIPE` line (the setting turns the lane
  trace on). Per layer kind (KDA MoE, DSA MoE, the dense layers 0–2) and lane size, the table
  gives each operation's median GPU time over the pass's layers and lanes and its share of the
  lane's time in that kind of layer.
  - An operation's time runs from the previous operation's event to its own, so the operations
    add up to the `PIPE` line's "GPU attention" and "shared". The table's last line repeats
    those two medians from its own events.
  - It needs only the CUDA runtime (no `nsys`).
  - On, it costs about 30 events per lane and layer and one synchronization per pass. On the
    development GPU that was within run-to-run noise: pass time +0.5% with lanes of 2,048 rows,
    +0.9% with lanes of 4,096. Off, it costs one check per operation.
  - `crates/glm53f-forward/examples/prefill_bench.rs` prints the same tables on one GPU: all 45
    layers on the weights of layers 0–4, routed experts returning zeros.
- **Tracing a decode step.** With `GLM53F_PROFILE=1` each decode step prints a `STEP` line: the
  mode (decode, or verify with its drafts and commit), the requests per lane, the host times of
  the step (the gap since the forward's last call, the drafts, the pass, the commit), the MoE
  layers' totals (wall, GPU busy, and the host blocked in `finish` waiting for the experts), then
  the same per-layer medians and wire record as a `PIPE` line.
- **The expert exchange's paths.** Prefill exchanges from `GLM53F_ROW_SHARDED_MIN_ROWS` rows
  are reduce-scattered by the ranks (each returns its quarter of the rows, summed; needs
  `--peers` on the ranks); decode and verify windows keep the four-plane return. Requests are
  built from the GPU into the page-locked request body, and returns read from the page-locked
  receive buffers (summed in place at decode sizes). Each has an off switch for comparisons;
  the results do not depend on them (`crates/glm53f-forward/src/remote.rs`).

### KV snapshots and the RAM tier

Every prompt end and completion end of 512 tokens or more stays on the GPU as a snapshot: the
request's KV rows, plus a copy of its KDA state (141 MiB, 376 pool pages; 73 MiB and 195 pages with
`--kda-state-bf16`) made on the GPU. A later prompt that starts with a snapshot's tokens resumes
there instead of prefilling them. The rule is **no tax unless loaded**:

- **No load.** A snapshot stays on the GPU, uncopied, however many there are. Nothing is copied to
  RAM while nothing needs the memory.
- **Load.** When an incoming request needs pool pages the pool does not have (a prompt being
  admitted or restored, a running request's growing reservation, a new snapshot's KDA state copy),
  snapshots are evicted one at a time, least recently used first, wherever they are: finished
  conversations' and running or prefilling requests' own. Each is copied to the RAM tier first,
  then its pages are freed. Eviction stops as soon as the request fits. A request whose snapshot is
  evicted runs on; only the snapshot moves. A new snapshot is skipped only if it still does not
  fit once no other request's snapshot is left. When a request needs a slot and none is free, the
  least recently used finished conversation's slot is evicted whole.
- **Restore.** A prompt that misses the GPU but matches a RAM snapshot restores from it instead of
  prefilling.
- `GLM53F_HOST_CACHE_GB` sizes the RAM tier. With `GLM53F_HOST_CACHE_GB=0` evicted snapshots are
  dropped.
- `GLM53F_PREFIX_CACHE_ENTRIES=N` caps the snapshots on the GPU at N per bank (prompt ends, turn
  ends) and moves the oldest to RAM past it, loaded or not. It is unset by default: no cap
  (mimo26f-afd's default was 24).
- **Log.** `[coordinator] device pressure: <request> needs <MiB>, <MiB> is free: evicting <whose>
  <bank> snapshot of <n> tokens, to RAM` names each eviction and the request it made room for
  (`slot pressure:` when a slot was needed). `[hostcache] store ... (<whose>, evicted for
  <request>)` is its copy to RAM and what it cost; `[hostcache] restore ...` a restore.

### Copy windows

With the drafter, a greedy request whose last 24 tokens occurred earlier in its prompt or output
verifies the up to 7 tokens that followed them instead of the drafter's proposals, and the drafter
skips it for that step (`crates/glm53f-coordinator/src/copy.rs`; TensorFold's idea, as mimo26f-afd
v1.3.0 ported it). Greedy output is unchanged; sampled requests never copy.
`--copy-windows off` (or `GLM53F_COPY_WINDOWS=0`) turns them off.

- **Log.** Every 64 speculative steps a `[copy]` line gives the windows copied, the copied tokens
  kept, and the tokens a copied and a drafted window delivered. The `[drafter]` line counts both
  kinds of window.
- **Measuring.** Tokens per verify round do not move with other traffic, so compare them with
  copy windows off and on (`[copy]` and `[drafter]` lines, or a client that counts the stream's
  bursts) on copy-heavy requests (a file written back with an edit, an `edit_file` call, a quote)
  and on fresh code and prose. On one GPU:

  ```sh
  # The copy index's host cost: 1 and 48 requests, contexts up to 1,048,576 tokens.
  cargo run --release -p glm53f-coordinator --example copy_cost
  # Copies replayed on real text (tokenizer and template, no model): how often a copy is found and
  # how many of its tokens a greedy target keeps, at an entry of 8 and of 24 tokens.
  GLM53F_TOKENIZER=... GLM53F_CHECKPOINT_DIR=... \
    cargo run --release -p glm53f-coordinator --example copy_replay
  ```

### The FP8 drafter and the L2 prefetch

Two options that change no committed token, both off until measured on the target hardware:

- **`--drafter-fp8`** (`GLM53F_DRAFTER_FP8=1`, with `--drafter`): the DFlash2 drafter's GEMM
  weights in FP8 E4M3 with 128 x 128 block scales, quantized at load, and its own FP8 copy of the
  LM head for drafting; the target's LM head is unchanged (`crates/glm53f-dflash/README.md`, "The
  FP8 drafter"). It moves only which tokens are drafted: the verify pass keeps a draft only where
  it equals the target's pick. The drafter takes 0.42 GiB less device memory, 0.37 GiB net of
  its larger working memory at 16 slots, which the KV pool gets ([SIZING.md](SIZING.md) §11). The
  start-up log names it; the `[drafter]` lines (drafts kept, tokens a window) measure its
  acceptance.
- **`--l2-prefetch off|auto|<MiB>`** (`GLM53F_L2_PREFETCH`): in decode and verify passes of one
  lane (one request, or more rows than the decode lanes take), once a MoE layer's routed experts
  are out and its shared expert is queued, the forward's stream reads the first bytes of the next
  layer's weights through L2, in the order that layer reads them
  (`crates/glm53f-forward/src/prefetch.rs`). Every result is bit for bit the same, and the
  prefetch (77 us at 48 MiB on an RTX 4090) ends long before the exchange it overlaps. Passes
  of two lanes, where the other lane's attention fills the exchange, never prefetch. `auto` is
  three quarters of the GPU's L2 (72 MiB on the RTX 5090); the start-up log's `decode and verify
  passes` line states it. What it gains depends on how long the real exchange leaves the GPU
  idle, which one GPU can only emulate: compare single-stream tok/s and the `STEP` lines
  (`GLM53F_PROFILE=1`) with it off and on, on the target hardware.

On one GPU:

```sh
# The prefetch kernel against one KDA layer's BF16 GEMVs: L2 flushed, a prefetch of 0-96 MiB, an
# idle gap, the GEMVs at 1 and 8 rows; per prefetch mode and stride.
cargo run --release -p glm53f-forward --features cuda --example l2_prefetch_bench
# Decode steps of layers 0-4 (GLM53F_BENCH_LAYERS=45: all) at 1, 4 and 8 requests
# (GLM53F_BENCH_BATCHES) and an 8-row verify, the exchange emulated (each MoE layer's routed
# experts back 360 us after the call), the prefetch off against on (GLM53F_BENCH_AB_LANES=1:
# the decode passes of two requests or more in two lanes, which never prefetch).
GLM53F_CHECKPOINT_DIR=... GLM53F_BENCH_EXCHANGE_US=360 GLM53F_BENCH_L2_PREFETCH_MIB=48 \
  cargo run --release -p glm53f-forward --features cuda --example decode_bench
# Bit for bit: the decode digest (6 decode steps and 6 verify passes after the prompt) is the same
# with the prefetch off and on.
GLM53F_CHECKPOINT_DIR=... GLM53F_DIGEST_DECODE=6 GLM53F_DIGEST_L2_PREFETCH_MIB=48 \
  cargo run --release -p glm53f-forward --features cuda --example logits_digest
# The drafter's draft and append times, BF16 then FP8.
GLM53F_DFLASH_DIR=... GLM53F_CHECKPOINT_DIR=... \
  cargo run --release -p glm53f-dflash --features cuda --example dflash_bench
# Acceptance on real prompts, BF16 against FP8: record the whole target teacher-forced on five chat
# cases (all 45 layers, the official FP8 experts loaded on demand from the full checkpoint; about
# five minutes a case on an RTX 4090; GLM53F_RECORD_CASES picks some), then replay both drafters
# over the recordings.
GLM53F_CHECKPOINT_DIR=... GLM53F_EXPERTS_DIR=<the official checkpoint> GLM53F_DFLASH_DIR=... \
GLM53F_TOKENIZER=... GLM53F_RECORD_OUT=<dir> GLM53F_RECORD_KDA_FP8=1 \
  cargo run --release -p glm53f-forward --features coordinator --example draft_record
GLM53F_DFLASH_DIR=... GLM53F_CHECKPOINT_DIR=... \
  cargo run --release -p glm53f-dflash --features cuda --example draft_replay -- <dir>
```

The model-path tests take both through `GLM53F_TEST_NUMERICS` (`l2-prefetch`, `drafter-fp8`):
`draft_lossless` and `copy_windows` with `drafter-fp8` check that speculation still changes no
token. `tests/decode_lanes.rs` checks the prefetch against no prefetch bit for bit, and
`glm53f-dflash`'s `tests/gpu_fp8.rs` the FP8 drafter's arithmetic.

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
`glm53f-score` cuts a prefill pass into two lanes unless `--prefill-lanes` says otherwise (the
coordinator's default is four). The numerics options (`--kda-fp8`, `--kda-fp8-pow2`, `--kda-mxfp8`,
`--kda-state-bf16` or `--kda-state-f32`, `--prefill-w8a16` or `--prefill-w8a8`,
`--kda-chunked-prefill` or `--kda-chain-prefill`, `--kda-prefill-w8a8`, as `glm53f-serve` takes
them, with its defaults: [Numerics defaults](#numerics-defaults)) are scored the same way into their
own directories and compared with the baseline at `--margin 0.002`; the engine line in every output
names them.
`--experts local` runs the official FP8 experts on the coordinator's GPU instead of the ranks;
`--dev-layers`, `--dev-load-layers` and `--experts zero` make a development run on one GPU, whose
logits are meaningless.

## Development on one GPU

- `--experts local` runs the official FP8 experts on the coordinator's GPU, loaded on demand
  from `--experts-dir` (a checkpoint holding the experts of the layers run).
- `--dev-layers 0-N` runs decoder layers 0 to N only, then the head, so the whole serving loop
  runs with a slice of the weights. **The output is meaningless text by design.**
- `crates/glm53f-forward/examples/prefill_bench.rs` times prefill passes in 1 to 4 lanes on one
  GPU (`GLM53F_BENCH_LANES=2,3,4`, `GLM53F_BENCH_LANE_ROWS`), the routed experts returning zeros:
  the GPU side of the lanes, not their overlap with the ranks.
- `crates/glm53f-forward/examples/logits_digest.rs` scores a fixed 6,000-token prompt in two-lane
  prefill passes and prints a digest of 162 rows of logits. A change meant to move no bit (a
  kernel's schedule, where buffers live) is checked by running it before and after the change on
  the same GPU: the two lines must be equal.
- Four ranks can share one GPU over loopback: cut layers 3 and 4 for each rank
  (`--layers 3-4`), start each with `--listen 127.0.0.1:0 --allow-partial`, then point
  `glm53f-serve --dev-layers 0-4 --ranks ...` at the four printed addresses. Each rank with two
  layers takes about 2.1 GiB.

The tests that do this themselves:

```sh
# RemoteExperts against the oracle and the local FP8 experts, on four rank daemons; prefill in 2 to
# 4 lanes against as many passes; the row-sharded return and the fast paths against four planes and
# the host paths, per call and in prefill lanes.
GLM53F_CHECKPOINT_DIR=... GLM53F_EXPERTS_DIR=... GLM53F_RANK_BIN=.../glm53f-rank \
GLM53F_RANK_DIRS=<rank-0>,<rank-1>,<rank-2>,<rank-3> \
  cargo test --release -p glm53f-forward --features coordinator --test remote_experts -- --nocapture --test-threads=1

# Prefill in 2 to 4 lanes against one lane (bit for bit against the same rows as that many passes,
# at every depth of calls in flight) and the goldens; admission and snapshots under a tight pool.
GLM53F_CHECKPOINT_DIR=... GLM53F_EXPERTS_DIR=... \
  cargo test --release -p glm53f-forward --features coordinator --test lanes --test admission -- --nocapture --test-threads=1

# Two-lane decode and verify against the same requests as two passes (bit for bit, the drafter's
# rings included), and speculation lossless with it.
GLM53F_CHECKPOINT_DIR=... GLM53F_EXPERTS_DIR=... GLM53F_DFLASH_DIR=... \
  cargo test --release -p glm53f-forward --features coordinator --test decode_lanes --test draft_lossless -- --nocapture --test-threads=1

# Copy windows lossless (plain, speculative, and speculative with copies, token for token), and
# tokens per verify round with copies off and on (ignored: --ignored). The forward copies its
# context only with every layer loaded, which fits 24 GB with FP8 KDA projections.
GLM53F_DRAFT_TEST_LAYERS=45 GLM53F_TEST_NUMERICS=kda-fp8 GLM53F_TOKENIZER=... \
GLM53F_CHECKPOINT_DIR=... GLM53F_EXPERTS_DIR=... GLM53F_DFLASH_DIR=... \
  cargo test --release -p glm53f-forward --features coordinator --test copy_windows -- --nocapture --test-threads=1

# One streamed chat completion through glm53f-serve in development mode.
GLM53F_CHECKPOINT_DIR=... GLM53F_RANK_BIN=... GLM53F_RANK_DIRS=... \
  cargo test --release -p glm53f-serve --features cuda --test dev_mode -- --nocapture

# Any model-path suite with a numerics option on (a comma-separated list of kda-fp8,
# kda-fp8-pow2, kda-mxfp8, kda-state-bf16, prefill-w8a16 and kda-prefill-w8a8, and l2-prefetch
# and drafter-fp8, which change no committed token; the forward's tests otherwise run every option
# off, whatever glm53f-serve's defaults), for example verify and commit with BF16 KDA states:
GLM53F_TEST_NUMERICS=kda-state-bf16 GLM53F_CHECKPOINT_DIR=... GLM53F_EXPERTS_DIR=... \
  cargo test --release -p glm53f-forward --features coordinator --test verify_commit -- --nocapture

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
| `GLM53F_COPY_WINDOWS` | `--copy-windows` | With `--drafter`: copy windows for greedy requests, `on` (default) or `off` (`0` in the environment too) ([Copy windows](#copy-windows)) |
| `GLM53F_DRAFTER_FP8=1` | `--drafter-fp8` | With `--drafter`: the FP8 drafter, off by default ([The FP8 drafter and the L2 prefetch](#the-fp8-drafter-and-the-l2-prefetch)) |
| `GLM53F_L2_PREFETCH` | `--l2-prefetch` | Decode and verify passes of one lane prefetch the next layer's weights into L2 while a MoE layer's experts are out: `off` (default), `auto` (three quarters of the L2) or MiB ([The FP8 drafter and the L2 prefetch](#the-fp8-drafter-and-the-l2-prefetch)) |
| `GLM53F_PREFILL_ROWS` | `--prefill-rows` | Rows of one prefill pass, every lane's together (default 8,192) |
| `GLM53F_PREFILL_LANES` | `--prefill-lanes` | Lanes of a prefill pass, 1 to 4 (default 4); at most 4,096 rows per lane |
| `GLM53F_DECODE_LANES` | `--decode-lanes` | Decode and verify passes of MIN to MAX rows in two lanes of whole requests: `off`, `MIN` or `MIN-MAX` (default `2-16`) (needs `--prefill-lanes` 2 or more) |
| `GLM53F_KDA_FP8=1` | `--kda-fp8` | Numerics under test, off by default (D2): the KDA projections quantized to FP8 block-128 at load ([SIZING.md](SIZING.md) §10) |
| `GLM53F_KDA_FP8_POW2=1` | `--kda-fp8-pow2` | Numerics under test, off by default: D2 with power-of-two block scales (the same layout, kernels and bytes; 82-89% of the q, k, v and o weights kept exactly, [SIZING.md](SIZING.md) §10, [KL-GATE.md](KL-GATE.md) §6e) |
| `GLM53F_KDA_MXFP8=1` | `--kda-mxfp8` | Numerics under test, off by default: D2 as MXFP8, an E8M0 scale per row and 32 values of K (the MXFP8 GEMMs; 4.13 GiB less rather than 4.26; the same error as `--kda-fp8-pow2` on these weights). Of the three KDA flags the last given sets the scales; in the environment `GLM53F_KDA_MXFP8` wins |
| `GLM53F_KDA_STATE_BF16` | `--kda-state-bf16` / `--kda-state-f32` | D8, **on by default** (passed the KL gate, docs/KL-GATE.md §6b): the KDA recurrent states stored in BF16, computed in f32; `0` or `--kda-state-f32` for F32 |
| `GLM53F_PREFILL_W8A16` | `--prefill-w8a16` / `--prefill-w8a8` | **On by default** with the chunked KDA prefill (the pair passed the KL gate, docs/KL-GATE.md §6d): FP8 projections over 8 rows with BF16 activations (W8A16); `0` or `--prefill-w8a8` for E4M3 activations (W8A8) |
| `GLM53F_KDA_PREFILL_W8A8=1` | `--kda-prefill-w8a8` | With `--kda-fp8` and W8A16: the FP8 KDA projections keep E4M3 activations over 8 rows |
| `GLM53F_KDA_CHUNKED_PREFILL` | `--kda-chunked-prefill` / `--kda-chain-prefill` | **On by default** with W8A16: the KDA of prefill passes through the chunked kernel instead of the serial chain (decode and verify keep the chain); `0` or `--kda-chain-prefill` for the chain. Without W8A16 the chunked kernel failed the KL gate, so turn the two off together |

**Serving shell:** `GLM53F_QUEUE_DEPTH` and `GLM53F_QUEUE_WAIT_MS` (the request queue),
`GLM53F_HOST_CACHE_GB` (the host RAM tier for KV snapshots; 0 turns it off; by default the
smaller of 32 GiB and 40% of the available RAM), `GLM53F_PREFILL_SEGMENT_MS`,
`GLM53F_PREFIX_CACHE_ENTRIES` (a cap on the KV snapshots kept on the GPU per bank, past which the
oldest go to RAM whatever the load; unset or 0, the default: no cap, and snapshots go to RAM only
when an incoming request needs their memory; [KV snapshots and the RAM
tier](#kv-snapshots-and-the-ram-tier)), and `GLM53F_SPEC`, `GLM53F_SPEC_POLICY`, `GLM53F_SPEC_TAU`,
`GLM53F_SPEC_COST_A`, `GLM53F_SPEC_COST_B`, `GLM53F_SPEC_MAX_ROWS` (speculation, with
`--drafter`; the last caps a step's verify rows, 256 by default, 0 for no cap, and sizes the
verify pass); the daemon reads `GLM53F_DFLASH_SAMPLED_WALK` (0: sampled requests draft with the
greedy walk too).

**Expert wire:** `GLM53F_RDMA=1` (coordinator: RDMA RC, in an `rdma` build; the ranks follow the
coordinator's handshake), `GLM53F_WIRE_NOCRC=1` (frames without CRC32C; both sides must agree;
RDMA requires it), `GLM53F_WIRE_MIN_GBPS` (the fabric's floor rate, default 100),
`GLM53F_WIRE_ALLOW_LAN=1` (admit a non-fabric network: tests only), `GLM53F_WIRE_INFLIGHT=N` (at
most N exchanges in flight over RDMA, instead of one per prefill lane; 1: the lanes take turns on
the wire),
`GLM53F_TIMELINE=1` (cross-host timeline events), `GLM53F_PROFILE=1` (per-exchange timings on the
coordinator, a `PIPE` line per prefill pass and a `STEP` line per decode step). The return path: `GLM53F_ROW_SHARDED_MIN_ROWS=N`
(exchanges of N rows and more, at least 4, reduce-scattered by the ranks; unset or 0: four planes
always; the design's value is 16) with `GLM53F_EXCHANGE_DTYPE` (`bf16`, the default, or `fp8`,
which needs the KL gate). The coordinator's fast paths, on by default:
`GLM53F_WIRE_DEVICE_ENCODE=0` (requests encoded on the host instead of copied from the GPU into
the request body), `GLM53F_WIRE_ZERO_COPY=0` (returns uploaded from pageable memory instead of
read from the page-locked receive buffers), `GLM53F_WIRE_FILL_ROWS=N` (requests of up to N rows
written by the frame-fill kernel instead of the copies; default 0) and
`GLM53F_WIRE_MAPPED_ROWS=N` (four-plane returns of up to N rows summed in place; default 64).

**Rank:** `GLM53F_RANK_TRACE=1` (a timing line per request), `GLM53F_RANK_DUMP_FRAME=<path>`
with `GLM53F_RANK_DUMP_LAYER` (write one request frame for offline replay),
`GLM53F_RANK_RECV_SLOTS` (`--recv-slots`: the requests an RDMA connection queues, 4 by default;
the coordinator keeps no more exchanges in flight than every rank queues).

**Tests:** `GLM53F_CHECKPOINT_DIR`, `GLM53F_EXPERTS_DIR`, `GLM53F_GOLDENS` (default
`oracle/goldens`), `GLM53F_TOKENIZER`, `GLM53F_EXL3_DIR`, `GLM53F_FP8_DIR`, `GLM53F_RANK_BIN`,
`GLM53F_RANK_DIRS`, `GLM53F_KL_TEACHER` (the teacher subset `klgate_fetch.py` writes),
`GLM53F_PYTHON` (default `python3`). Tests skip, and say why, when theirs are missing.
