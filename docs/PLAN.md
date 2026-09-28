# Plan

> **Note (29 September 2026).** This is the build plan as written before the code, kept as a
> record. Phases 0 to 3 are built, except constrained output and a needle ladder beyond 79K
> tokens; phase 4 is in part; phase 5 has not started. Current state: [README](../README.md) and
> [PERFORMANCE.md](PERFORMANCE.md) §0.

**Status: draft for discussion (28 September 2026).** Each phase ends at a gate
with a measured receipt. Effort figures are rough, for a builder working with
parallel helpers; the long pole is the new model kernels (KDA prefill, the k-pool
indexer), not the serving shell.

## Phase 0: decisions and the oracle (1–2 days)

- Settle D1–D8 ([SIZING.md](SIZING.md) §9).
- Build the reference oracle: the `transformers` `glm5_next` code on the official
  FP8 weights. Record per-layer goldens for a handful of short prompts, as fixtures
  with digests.
- Expert format (D6): **largely answered by published measurements** (DESIGN §5).
  EXL3 K4 scores a KLD of 0.0246 against BF16, official FP8 scores 0.0206, both on
  the same 25-window panel, measured offline with no KV-cache quantization. The
  engine uses EXL3 K4 by default. No separate bake-off is needed.
- The engine's end-to-end KL gate scores on that **same public panel**. It uses the
  BF16 teacher logits of `brandonmusic/GLM-5.3-Flash-BF16-Teacher-Logits` (the
  `logits/` set is about 32 GB; the gate scores a fixed subset of 189 rows per
  window, fetched by range). The method, the harness (`harness/klgate.py`) and the
  engine interface are in [KL-GATE.md](KL-GATE.md). The engine's number includes
  its own FP8 KV cache, FP8 wire rows and kernels on top of the expert format, so
  it is comparable with, not equal in scope to, the published 0.0246.
- **Gate:** decisions recorded; goldens reproducible.

## Phase 1: the Spark ranks (2–4 days)

- Bring in M's rank daemon, wire frames and RDMA transport.
- Add the expert kernel for the chosen format (DESIGN §5), exported ahead of time
  for 4,096 × 512 slices and 288 experts, with SwiGLU clamped at 10.
- Add G's FP8 reduce-scatter for prefill-sized batches.
- **Gate:**
  - each rank's expert output matches the dequantized reference;
  - read rate at 1–8 rows: at least 200 GB/s for 4-bit formats;
  - a 2K–4K-row batch completes within its dataflow estimate;
  - the prefill return carries one plane.

## Phase 2: the coordinator model path (5–10 days)

Build these as vertical slices, each checked against its golden layer:

1. embedding (host gather) → mHC → dense MLP layers 0–2;
2. KDA:
   - port T's chain and replay kernels;
   - write the chunked prefill kernel;
3. DSA:
   - indexer with k-pool 4 and the tail;
   - top-512 selection;
   - sparse MLA over the FP8 latent;
4. router, shared expert and the remote expert call;
5. final collapse, LM head, sampler.

**Gate:**
- first coherent tokens end to end;
- KL against the oracle within the agreed bound on the corpus;
- needle recall to 32K.

## Phase 3: serving (3–6 days)

- Scheduler: 16 slots in two lanes, the prefill wavefront, every row type batched.
- KV pool, radix, KDA snapshots, host RAM tier.
- API:
  - chat, streaming, GLM tool calls and reasoning;
  - stop strings in the engine;
  - queue with 429;
  - constrained output.
- DFlash2, greedy and sampled, with the S2 verify policy.

**Gate:**
- API contract rows pass;
- drafted output equals serial output (greedy, and seeded sampling);
- needle ladder to 1,048,576 with caching on;
- cache-pressure test passes;
- first benchmark of record against [PERFORMANCE.md](PERFORMANCE.md).

## Phase 4: performance (3–5 days)

Candidates, each kept only on a measured gain:
- device-ordered stage chains;
- GPU-written frames and in-place plane sums;
- 4K expert chunks with a mapped send;
- graphs for every steady shape;
- the online bandwidth-balance policy as an A/B against S2;
- FP8 KDA projections, if D2 allows.

**Gate:** decode and prefill at or above the model's lower bounds, with no KL
regression.

## Phase 5: breadth

- Vision (the `glm5_next` tower).
- MTP as a second drafter.
- Console and `/v1/stats`.
- Other Spark counts.
- A tagged release with benchmarks.

## Rules that apply throughout

- Every copied unit gets a row in `docs/REUSE.md` before it lands.
- Numerics changes need a KL gate before they become defaults.
- No deployment-specific names, addresses or paths in this repository; examples
  use `spark-0` … `spark-3` and 192.0.2.0/24.
- Long benchmarks run only at phase gates; each step's check stays short.
