# Design

**Status: draft for discussion (28 September 2026).** This is the design to
agree on before any engine code is written. Sizing is in
[SIZING.md](SIZING.md), expected performance in [PERFORMANCE.md](PERFORMANCE.md),
and the build order in [PLAN.md](PLAN.md).

## 1. Goal

Serve **GLM-5.3-Flash** over an OpenAI-compatible API on **one RTX 5090 and four
DGX Sparks**, connected by RoCE v2 RDMA. The engine uses attention–FFN
disaggregation (AFD):

- the 5090 runs attention, the KV cache, drafting, sampling and the API;
- the four Sparks run the routed experts.

The targets are about **100 tok/s** for one stream, **1,048,576-token requests**,
and several 256K sessions at once. The repository is public from the start, with
no deployment-specific details in it.

**Non-goals for version 1:**
- other topologies (2, 3, 5 or 6 Sparks; larger coordinators) — the design must
  not preclude them;
- audio;
- training;
- a Python serving path.

## 2. Principles

1. **Take the proven serving shell; write only the model.** Every piece of this
   engine except the GLM-5.3-Flash layers already exists in a working engine on
   this hardware class. Copy it in, with provenance. Write only what is new.
2. **One provenance ledger.** `docs/REUSE.md` records the source repository,
   commit and file digest of every copied unit. `NOTICE.md` carries the licences.
3. **Correctness before speed.**
   - A reference oracle comes first: the `transformers` `glm5_next` code, run on
     real weights for short prompts. Per-layer goldens are taken from it.
   - Each numerics change needs a KL gate before it becomes a default.
4. **Measured gates, not stub predictions.** Every performance step lands with a
   receipt: build, command, before and after.
5. **No Python, and no empirical memory reserves.**
   - Kernels that exist only as Triton or CuTe DSL are exported ahead of time to
     C launchers.
   - The memory plan is computed from components and checked against
     `cudaMemGetInfo` at start-up.

## 3. System shape

```text
                  OpenAI API (chat, streaming, tools, reasoning, images)
                                       │
   ┌───────────────────────────────────▼───────────────────────────────────┐
   │ Coordinator: RTX 5090                                                  │
   │  API · admission and queue · scheduler (16 slots, 2 lanes)             │
   │  embedding (host RAM) · mHC (4 streams) · KDA ×34 · DSA/MLA ×11        │
   │  dense MLP 0–2 · shared expert · router (sigmoid top-8 of 288)         │
   │  LM head · GPU sampler · DFlash2 drafter · MTP (optional)              │
   │  KV: MLA latent pages (FP8) + indexer pooled keys + KDA snapshots      │
   │  host RAM tier: evicted pages and snapshots                            │
   └───────────────┬───────────────────────────────────────▲───────────────┘
      routed rows (FP8 or NVFP4 hidden, routes, weights) │ expert output (BF16)
                   │   RDMA RC over RoCE v2, two lanes     │
   ┌───────────────▼───────────────────────────────────────┴───────────────┐
   │ Expert ranks: 4 × DGX Spark (TP4: a quarter of every expert's width)   │
   │  zero-copy frame input · grouped expert kernel · output to send slot   │
   │  prefill: FP8 reduce-scatter among ranks → one plane to coordinator    │
   └────────────────────────────────────────────────────────────────────────┘
```

## 4. Where each part comes from

**Sources.**
- **M:** [hughmadden/mimo26f-afd](https://github.com/hughmadden/mimo26f-afd) v1.2.0 (MiMo-V2.6-Flash on this hardware; MIT).
- **D:** [tpurtell/ds41rt](https://github.com/tpurtell/ds41rt) v15 (DeepSeek-V4.1-Flash; MIT).
- **G:** [tpurtell/glmrt-5.3-1rtx-4spark](https://github.com/tpurtell/glmrt-5.3-1rtx-4spark) v9 (full GLM-5.3; MIT).
- **T:** [ashhart/TensorFold](https://github.com/ashhart/TensorFold) `families/glm5_next/cuda` (GLM-5.3-Flash on two Sparks; MIT).
- **S:** [tpurtell/sparkinfer-glmrt](https://github.com/tpurtell/sparkinfer-glmrt) (b12x fork; Apache-2.0).
- **F:** [fla-org/flash-linear-attention](https://github.com/fla-org/flash-linear-attention) KDA ops (MIT).
- **R:** `transformers` `models/glm5_next` (the reference).

| Part | Take from | What changes for GLM-5.3-Flash | New work |
|---|---|---|---|
| RDMA transport and wire frames | **M** (`DS41RTE3` v3 frames, RC rings, RoCE-only guard; design from **D**) | Frame dimensions: hidden 4,096; top-8 of 288 | — |
| Prefill return path | **G** (Spark-side FP8 row reduce-scatter above 16 rows) | Ported into M's rank daemon | The coordinator receives one plane instead of four. This prevents 4→1 incast on switches without priority flow control. |
| Spark rank daemon | **M** (mapped-frame input, output written into the send slot, packed route upload) plus **D**'s single-CTA decode route planner | Expert shape 4,096 × 512 per rank; 288 experts | — |
| Spark expert kernel, EXL3 K4 | **T** `exl3.cu` (208–220 GB/s at 1–8 rows on GB10) or **S** W4A16 trellis fused MoE (about 230 GB/s marginal, as **G** uses it) | Written for GLM-Flash (**T**); new AOT export for 4,096 × 512 (**S**) | Choose by measurement (§5) |
| Spark expert kernel, NVFP4 | **S** NVFP4 MoE kernels (recent commits target E = 288, K = 4,096) or **M**'s B1 W4A8 with NVFP4 block scales | E4M3 scales per 16 values instead of E8M0 per 32 | Only if D6 picks NVFP4 |
| Spark expert kernel, FP8 | **S** block-FP8 grouped GEMM | — | Only if D6 picks FP8 |
| Coordinator projections | **D** native block-FP8 GEMMs (checkpoint FP8, 128 × 128 blocks, FP32 accumulate, AOT); BF16 GEMMs for KDA | — | FP8 KDA only if D2 says so |
| mHC (4 streams, Sinkhorn 20) | **D** (the DeepSeek-V4 formulation, fused boundary kernels) | The final stream collapse is an **unweighted mean** in GLM-5.3-Flash (a weighted head in DeepSeek-V4) | Small |
| KDA, decode and verify | **T** `kda.cu`: a fused per-layer chain over R rows (convolution, L2 norms, per-channel decay, beta, delta rule, gated RMSNorm), plus a `replay` kernel that rebuilds the state after the accepted prefix, bit-exact with serial steps | Port from a torch extension to the engine's launcher; batch across requests | Ported, not new |
| KDA, chunked prefill | **F** `chunk_kda` (WY/UT chunk form) as the algorithm reference | Handwritten CUDA, or a TileLang/Triton kernel exported ahead of time | **New:** the long-prefill kernel |
| DSA indexer (k-pool 4, top-512 pools plus tail) | **R** for semantics; **G** for the FP8 paged top-k scorer (b12x `index_topk_fp8`) | Pooled keys stored once per complete pool; tail of 3 or fewer raw tokens; a fused score-and-select for long context | **New:** k-pool store and selection |
| Sparse MLA (NoPE, 512 latent, 64 heads of 256, 2,051 tokens or fewer) | **G** (sparse MLA over an FP8 paged latent on sm_120); **D**'s 64-head KV-reuse kernel design (7.76 → 1.95 ms at 2,048 rows) and register rescale | No RoPE part; 528-byte FP8 record | Adapt |
| Router | **M** (sigmoid, correction bias, top-8, FP32) | 288 experts; routed scale 2.5 | — |
| Shared expert, dense layers 0–2 | **D**/**G** pattern: the shared expert runs during the remote wait | FP8 | — |
| LM head and sampler | **M** V3 GPU sampler (DS41RT v15 contract: greedy by default; seeded, position-keyed draws; vLLM filter order) | Vocabulary 154,880; three stop ids | — |
| DFlash2 drafter | **M** (DFlash on the coordinator) + **G**/**T** (GLM's DFlash2: taps, selector top-16, 2K window) | 5 layers; taps 5, 14, 24, 33, 42; block of 8 | Wiring |
| MTP drafter (optional) | **T** `mtp.py` + **G** | Its experts run on the Sparks as a 43rd MoE layer | Later phase |
| Verify policy | **M** S2 (stop drafting when the product of draft confidences falls below a threshold; beat a cost-model policy live) | — | **D** v13's online bandwidth-balance policy as an experiment |
| Speculation around grammars | **D** v15 (trim drafts only at the first token the grammar rejects) | GLM stop ids | — |
| Scheduler | **M** (16 slots, segmented prefill interleaved with decode, short-prompt batching) + **D** (two independent lanes) + **G** (chunk × layer prefill wavefront) | Every row type is batched (§7) | — |
| KV pool and prefix cache | **M** (paged pool, right-sized admission) + **D** (radix, prompt and turn banks) | MLA pages plus pooled index keys; KDA snapshots at prompt and turn end | **New:** hybrid snapshot bookkeeping |
| Host RAM tier | **M** K3 (the DS41RT on-evict design) with least-recently-used eviction | Snapshots include the 141 MiB KDA state | — |
| API | **M** (OpenAI chat, SSE, tools, reasoning, stop strings in the engine, bounded queue with 429) + **G** (GLM chat template; tool-call format; reasoning split on think tokens) | — | — |
| Constrained output | **D** (XGrammar masks on the native path) | GLM tool-call grammar | — |
| Vision | **M** (image decode crate) + **R** `glm5_next` vision tower (24 layers, 1,024 hidden, patch 14, merge 2) | New encoder kernels, as MiMo's were | Later phase |
| Console and statistics | **D** v15 (`/` console over a WebSocket, `/v1/stats`) | — | Optional |
| Test and benchmark harness | **M** (API contract rows, needle ladder, sampling checks, cache-pressure test) | GLM prompts and tool markup | Oracle and goldens (§9) |

## 5. Expert formats

The Sparks can hold any of the three formats (SIZING §6). The choice trades
quality against decode speed:

| Format | Per rank | M1 read per token | Kernel options | Quality evidence |
|---|---:|---:|---|---|
| EXL3 K4 (`mcg` trellis, routed experts only) | 38.2 GB | 1.06 GB | **T** (GLM-Flash native), **S** | Used by two GLM-Flash engines on Sparks; no published KL figure |
| NVFP4 (modelopt, group of 16) | 42.8 GB | 1.19 GB | **S** (GLM-Flash geometry), **M** B1 with new scales | Serves on Sparks in vLLM and SGLang; no published KL figure |
| FP8 (official) | 76.1 GB | 2.11 GB | **S** block-FP8 grouped | **The reference** |

Plan: measure all three against FP8 on the same prompt corpus: top-1
agreement, KL and DFlash2 acceptance. Choose using the joint table (D6). The
coordinator always loads the official checkpoint's non-expert tensors, since the
quantized checkpoints ship them in BF16.

## 6. KDA

- **Decode and verify.**
  - Each window of R rows (the current token plus drafts) runs one fused kernel
    per layer. The kernel writes each row's output, the state after the last row,
    and the replay inputs: normalized k, v, decay and beta per row.
  - After verification, `replay` rebuilds the state after the accepted prefix
    from the state saved before the round. Keeping a prefix therefore gives the
    same bits as serial steps.
  - Memory per slot stays at one state (136 MiB FP32), plus a small replay
    scratch per lane.
- **Prefill.**
  - A chunked kernel (the gated delta rule in WY form, 64-token chunks) runs the
    long prompts. Its result is checked against the serial chain for bit-level
    agreement.
  - If agreement holds only within tolerance, the chain kernel remains the
    reference, and the chunked kernel's KL is gated like any numerics change.
- **Batching.** Rows from different requests share the projections. The recurrent
  part is per request: one block per (request, head).
- **State precision.** FP32, as the reference keeps it (D8).

## 7. Scheduling and batching

**Slots and lanes.**
- 16 slots in two independent lanes of up to 8.
- Each lane runs its own decode and verify rounds, so one lane's coordinator
  work overlaps the other lane's expert wait.

**Every row type is batched.** Greedy, sampled, grammar-constrained, tool-calling
and first-step rows all join the same round. Per-row sampling parameters and
per-row grammar masks make this possible. In an engine that serialises the whole
set whenever one sampled or constrained request is active, a mixed C4 set falls
to about C1 throughput. This design rules that out.

**Prefill.**
- Long prompts run as a chunk × layer wavefront.
- In segments of about 2 s, decode rounds for the other slots run in between.
- Short prompts that arrive together prefill in one pass.

**Admission.**
- Each request reserves its prompt plus an output allowance, not the maximum
  context.
- A request that does not fit waits.
- Beyond the queue depth, the API answers 429 with `Retry-After` before the
  response starts.

## 8. Cache

- **Pool.** The coordinator's KV pool is paged. Each page holds the MLA latent
  (FP8 528-byte records, 11 layers) and the indexer's pooled keys (11 layers)
  for 64 tokens (16 pools).
- **Radix.** A radix tree over token ids and image identities shares pages
  between requests.
- **KDA snapshots.**
  - One is saved at the end of each prompt and of each completed turn.
  - A match resumes from the longest snapshot that lies within the match.
  - Optional periodic checkpoints (D4) cover divergent branches.
- **Eviction.** Least recently used, over retained entries only. Evicted pages and
  snapshots move to a page-locked host RAM tier, and exact or extending repeats
  restore from it.
- **Memory plan.** Computed at start-up: weights, drafter, per-slot state,
  workspace, then the pool. It is checked against free device memory, and an
  infeasible plan fails at start-up.

## 9. Correctness

1. **Oracle.** The `transformers` reference on the official FP8 weights. It is run
   once to record per-layer goldens for short prompts: KDA outputs and states,
   indexer selections, MLA outputs, mHC weights, router choices and final logits.
   The goldens are stored as small binary fixtures with digests.
2. **Kernel gates.** Each coordinator kernel is checked against its golden layer:
   bitwise where the arithmetic matches, otherwise within a stated tolerance.
   Each expert kernel is checked against a dequantized reference.
3. **End-to-end gates.** First-token logits, KL against the oracle over a corpus,
   greedy continuations, a needle ladder to 1M, API contract rows, and
   drafted-equals-serial checks for greedy and seeded sampling.
4. **Traps.** Known serving traps are tested for:
   - chat-template drift;
   - tool-call markup leaking into content;
   - reasoning lost on a later turn;
   - UTF-8 split across tokens;
   - stop strings ending inside a draft block.

## 10. Configuration and deployment

- **Addresses.** Rank addresses, RDMA devices and ports are configuration.
  Examples use documentation addresses (192.0.2.0/24) and `spark-0` … `spark-3`.
- **Fabric.** Inference traffic must run on a RoCE v2 port of at least 100 Gb/s.
  The engine refuses anything else. The API may use any network.
- **Binaries.** One binary per role: `coordinator` (x86-64, sm_120) and `rank`
  (arm64, sm_121). Both are built natively, with kernels compiled ahead of time
  for the device found at build time.

## 11. What this design does differently from its sources

| Where a source… | This engine… | Why |
|---|---|---|
| speculates only for greedy requests | speculates for sampled requests too, by sample-and-match | Sampled traffic otherwise decodes at the one-row target rate |
| serialises the batch when a sampled or constrained row is present | batches every row type | Mixed traffic otherwise loses up to 40% at C4 |
| applies stop strings after generation ends | stops in the engine and holds back partial matches while streaming | Avoids wasted slot time and damaged stream formatting |
| lets the draft cache limit target prefix reuse | resumes the drafter with a fresh window when needed | Shared prefixes otherwise re-prefill in full |
| scans every visible position on the host per layer | passes O(1) cache metadata | This cost grows with context |
| sizes memory from an empirical reserve | computes and verifies the plan | Different cards and headroom work without edits |
| gates its speculation cost profile on one GPU and power limit | starts from a measured policy and refits it online | Coordinators other than the reference card work |
| returns four partial planes at prefill sizes | reduce-scatters among ranks first | Avoids 4→1 incast |

## 12. Open questions

1. How much DFlash2 acceptance GLM-5.3-Flash gets by content type, and on
   BF16/FP8 versus quantized experts.
2. The expert format: joint quality and speed (D6).
3. Whether FP8 KDA projections pass the quality gate (D2).
4. Chunked KDA prefill: whether handwritten CUDA or an exported TileLang or Triton
   kernel reaches the Spark-bound prefill rate (about 5K tok/s needs well under
   200 µs per token on the coordinator).
5. The indexer's top-512 selection over up to 256K pools at 1M context: whether a
   fused score-and-select kernel is needed from the start.
