# Expected performance

**Status: measured and modelled (28 September 2026).** §0 gives the first
measurements of the whole engine on its target hardware. The other sections are
the model it was designed against, derived from published or measured numbers of
related engines; they are kept so each derivation can be checked against §0.
Layout and format letters refer to [SIZING.md](SIZING.md).

## 0. Measured on the target hardware

**Setup:**
- One RTX 5090 coordinator and four DGX Spark (GB10) expert ranks, over RoCE v2 RDMA at 200 Gb/s.
- EXL3 K4 experts, FP8 MLA cache, BF16 KDA projections, the embedding in host RAM.
- All 45 layers, 16 slots. Thinking on (the model's default) unless stated.
- Single runs; between runs, ±2–3% is typical.

**Decode, one stream:**

| Case | Code | Prose | Counting |
|---|---:|---:|---:|
| No drafter | 52.7 | 52.7 | 52.7 |
| DFlash2 (chain τ 0.7), greedy, thinking off | **111.3** | 62.4 | **128.4** |
| DFlash2, greedy, thinking on | 82.7 | 73.3 | 169.1 |
| DFlash2, sampled (T 0.7), thinking off | 113.5 | 67.0 | 123.1 |

- Figures are tok/s. Without a drafter, 52.7 tok/s is 19.0 ms per token.
- With the drafter, 51–71% of verified drafts are kept (τ 0.3–0.7), 3.0–3.4 tokens per verify window.
- The model below expected 36–43 tok/s without a drafter (§2) and 110–130 / 75–105 / 125–145 with it (§3).

**Where a one-row step goes** (traced medians per MoE layer): the coordinator's own work 0.25 ms,
the rank kernel 0.168 ms, the wire about 33 µs beyond the rank's compute.

**Concurrency, aggregate:**

| Streams | No drafter | DFlash2 (τ 0.7) |
|---|---:|---:|
| C4 | 126.5 tok/s | 143.3 tok/s |
| C16 | 241 tok/s | **299 tok/s** |

The model expected 350–480 tok/s at C16 (§4). At 16 requests the verify passes reach 128 rows, and
nothing yet overlaps the drafter with the forward or the coordinator with the ranks in decode.

**Prefill**, with two-lane pipelining (the coordinator computes one half of a pass while the ranks
serve the other; 4,096-row passes in two lanes of 2,048):

| Prompt | 4K | 19K | 79K |
|---|---:|---:|---:|
| Two lanes (default) | 2,717 tok/s | 2,964 tok/s | 2,899 tok/s |
| One lane | 1,633 tok/s | 1,725 tok/s | 1,715 tok/s |

- A 79K-token prompt takes 27 s.
- This is still a **MISS** against the 4.5–6K tok/s of §5.
- 8,192-row passes reach about 3.1K, but take 3 GiB more buffers, which shrinks the pool below a 1M-token request.
- Per MoE layer the GPU is busy 66% of a 27.5 ms layer. The rest is the host's per-lane encode (1.5 ms) and the upload and sum of four returned planes (3.2 ms).
- The reduce-scatter return and the RDMA fast paths address that host work.

**Against a vLLM recipe on the same four Sparks without a coordinator GPU**
([tonyd2wild/GLM-5.3-Flash-NVFP4-1M-KV-4x-DGX-Spark](https://github.com/tonyd2wild/GLM-5.3-Flash-NVFP4-1M-KV-4x-DGX-Spark), TP4, NVFP4 experts, its README's figures):

| Metric | That recipe | This engine |
|---|---|---|
| Single stream | ~55 tok/s | 108–128 on code and counting, 60 on prose |
| Aggregate | **530 tok/s at 48 streams** | 292 tok/s at 16 streams (16 slots) |
| Prefill, short prompts | **3.5–4.1K tok/s** (warmed, ~9K) | 2.7–3.0K |
| Prefill, 114K prompt | 1.9K tok/s | 2.9K at 79K |

This engine leads at one stream and on long prompts, and trails on aggregate throughput and
short-prompt prefill. The levers: two-lane decode, more slots and speculation that adapts to load
for the aggregate; the host path above for prefill.

**Start-up:** the coordinator is ready 7 s after launch (weights from the page cache). A rank is ready
in 44–49 s (§6).

**Quality spot checks:**
- a number hidden at 37% depth is retrieved from 8.8K and 79K tokens of filler, with and without the drafter;
- `harness/api_contract.py` passes all 12 rows on the real model.

The KL gate against the published BF16 teacher ([KL-GATE.md](KL-GATE.md)) waits for the engine's
score mode.

## 1. Anchors

| Anchor | Value | Source |
|---|---|---|
| MiMo-V2.6-Flash, 4 Sparks + 5090 (same hardware class, similar expert shape) | C1 109.7 tok/s; C16 416.7 tok/s aggregate; prefill 3,630 / 5,123 / 4,816 / 4,283 tok/s at 2K / 8K / 32K / 64K; a 993,795-token prompt in 17.3 min | [mimo26f-afd](https://github.com/hughmadden/mimo26f-afd) v1.1–1.2 benchmarks |
| MiMo plain decode (no drafter) | about 24.5 ms per token over 47 MoE layers | the same engine, measured |
| DS41RT one-row expert phase per layer | 328 µs, of which 157 µs is the kernel | [tpurtell/ds41rt](https://github.com/tpurtell/ds41rt) `docs/ds41-expert-boundary-breakdown.md` |
| EXL3 K4 expert slice on GB10 | 20.5 µs marginal per 4.72 MB TP4 slice, about 230 GB/s | [glmrt](https://github.com/tpurtell/glmrt-5.3-1rtx-4spark) v9 route-cost profile |
| **This engine's rank kernel on GB10 (measured)** | One MoE layer on one rank: 1 row 0.145 ms (175 GB/s; about 26 µs of it is fixed: plan, epilogue, reduce); 8 rows over about 58 distinct experts 0.87 ms (215 GB/s); 4,096 rows 16.9 ms (243K rows/s) | `crates/glm53f-rank/examples/exl3_bench.rs --sweep`, default tiling, real layer slices |
| DFlash2 on GLM-5.3 (the full model) | code replay 5.78 tokens per target cycle; 67.9% acceptance on an 8-type mix | glmrt v9 README |
| GLM-5.3-Flash on 2 × RTX PRO 6000 (vLLM, EXL3 4 bpw + DFlash2, no Sparks) | C1 reasoning coding 140.9 tok/s; 128K prefill 4,984 tok/s; 1M six-needle retrieval 6/6 | [glm-5.3-flash-ext3-2x-rtx](https://github.com/tpurtell/glm-5.3-flash-ext3-2x-rtx) v0.8.0 |

**Why MiMo is the closest anchor.** GLM-5.3-Flash and MiMo-V2.6-Flash have the
same expert shape: 4,096 × 2,048 with top-8, so each rank computes 3 × 4,096 ×
512 per route. GLM has **42 MoE layers against MiMo's 47**. It is heavier on the
coordinator: KDA projections in BF16, and 4 mHC streams.

## 2. Decode: one target cycle, one row (M1)

| Term | Derivation | Layout B, EXL3 K4 | Layout C (FP8 KDA) |
|---|---|---:|---:|
| Spark expert kernel | 42 layers × 0.145 ms, **measured** on GB10 at one row. It includes the kernel's own fixed phases, about 26 µs per layer. (The model said 4.6 ms: 8 slices × 3.17 MB at 230 GB/s.) | 6.1 ms | 6.1 ms |
| Coordinator weight reads | 13.96 GB (B) or 9.28 GB (C) at 85–90% of 1.79 TB/s | 8.7–9.2 ms | 5.8–6.1 ms |
| Fixed cost per MoE layer | 0.20–0.27 ms × 42. Covers exchange, Spark fixed cost and per-layer launches. The range spans DS41RT to MiMo. Up to 26 µs per layer of it is now inside the kernel row, so the cycle subtracts 0–1.1 ms. | 8.4–11.3 ms | 8.4–11.3 ms |
| KDA state, indexer, mHC | 0.3 GB of FP32 state per step; top-512 over n/4 pools; 90 mHC boundaries | 1.0–1.5 ms | 1.0–1.5 ms |
| **M1 cycle** | | **23.1–28.1 ms** | **20.2–25.0 ms** |
| **Target-only decode** | | **36–43 tok/s** | **40–49 tok/s** |

- The one-row kernel misses its 200 GB/s target. Fusing its fixed phases is the
  next kernel change.
- NVFP4 experts would add about 0.6 ms per cycle and FP8 experts about 4.6 ms
  (modelled at 230 GB/s, not measured).

## 3. Decode with DFlash2 (block of 8)

- **Extra experts per verify row.** Each row routes to new experts. Scaled from
  GLM-5.3's measured unique-expert counts, 8 rows touch about 40 distinct experts
  per layer, against 8 for one row.
- **Cost of 8 rows.** Measured on GB10, 8 rows over about 58 distinct experts
  take 0.87 ms per layer, about 14.6 µs per expert beyond the fixed phases. At the
  expected 40 experts that is about 0.60 ms per layer: 0.46 ms more than one row,
  or about 19 ms per round over 42 layers (the model said 18 ms). Add about 1 ms of
  coordinator row work and 1.5–2.5 ms for the draft pass (2.18 GiB of BF16
  weights).
- **Round length (layout B).** About 45–51 ms at 8 rows, 38–45 ms at 6 rows and
  33–40 ms at 4 rows. These were 43–48, 37–43 and 32–38 ms before the kernel was
  measured: each round gains the one-row cycle's 0.4–1.5 ms, and its extra experts
  cost 6.6% more (14.6 µs each measured, against 13.7 µs modelled).

| Workload | Tokens per round | Round | Expected C1 |
|---|---:|---:|---:|
| Code (high acceptance) | 5.5–5.8 | 45–51 ms | **110–130 tok/s** |
| Weighted mix (code, maths, prose, JSON, short answers) | about 3.3 | 35–40 ms | **75–105 tok/s** |
| Low-entropy text (counting, templated output) | about 6.5 | 45–51 ms | **125–145 tok/s** |

- **Layout C** (FP8 KDA) adds about 5–8% to every row.
- **Sampled requests** (temperature > 0) use the same drafter with sample-and-match
  verification. Expect 3–14% below greedy, the penalty MiMo measured.
- **For comparison:** MiMo on the same hardware measures 110 tok/s weighted, and
  GLM-5.3-Flash on two RTX PRO 6000 measures 140.9 tok/s on reasoning code.

## 4. Concurrency

**16 slots in two lanes.** Expect **350–480 tok/s aggregate at C16**, against
MiMo's 416.7. What pushes the estimate each way:

- **Better than MiMo:** 11% fewer MoE layers. The coordinator's weights are read
  once per step and shared by all rows.
- **Worse than MiMo:**
  - each active request reads and writes 272 MiB of FP32 KDA state per step. The
    recurrent part measured 3.0–3.5 ms per step for 8 requests of one row on an
    RTX 4090, so C16 on the 5090 costs nearer **4 ms** per step. This corrects an
    earlier estimate of 2.5 ms, which assumed full bandwidth;
  - the BF16 KDA projections.

## 5. Prefill

**Two sides, two lanes.** Prefill pipelines the coordinator and the Sparks across
two lanes, so it runs at the slower side's rate.

| Side | Work per token | Rate if it were the limit |
|---|---|---|
| **Sparks** | 42 × 8 × 25.2 M = 8.46 G MAC, 0.89× MiMo | MiMo's rate ÷ 0.89: about **5.4–5.8K tok/s** at 8K–32K. **Measured** kernel: 4,096 rows per layer in 16.9 ms on one rank (243K rows/s), so 42 layers bound a 4,096-token chunk at about 5.8K tok/s before any exchange cost |
| **Coordinator, projections** | about 6.8 G MAC (13.6 GFLOP), most of it BF16 KDA | **7–14K tok/s** |

**Context-dependent coordinator work** comes on top of the projections:

- **Sparse MLA:** each query reads at most 2,051 tokens, about 3 GFLOP per token.
- **Indexer:** 2,048 × n FLOP per token per layer over n/4 pools, plus top-512
  selection. Averaged over a 1M prompt this is about 11 GFLOP per token, and about
  50 µs per token of selection.
- **KDA:** linear in n, so the attention cost stays nearly flat as context grows,
  unlike the global-attention layers in MiMo. It needs a **chunked prefill
  kernel** (the gated delta rule in WY form, 64-row chunks). Run serially, the
  recurrence alone measured 46 µs per token over 34 layers on an RTX 4090, and the
  full fused chain 86–101 µs: half the coordinator's budget. The chunked form is
  about 0.3 GFLOP per token, or 6–10 µs on the 5090, if its intermediates stay in
  FP32 and its passes are fused.

| Prompt | Expected | Derivation |
|---|---:|---|
| 2K fresh | 3.3–4.0K tok/s | Short chunks; MiMo 3.6K |
| 8K–32K fresh | **4.5–6.0K tok/s** | Spark-bound |
| 128K fresh | 4.0–5.0K tok/s | Indexer and sparse MLA add 30–60 µs per token |
| 1M fresh | **4–6 minutes** | About 250 µs per token on average. MiMo: 17.3 min |
| Follow-up turn from a snapshot | **0.2–0.4 s** time to first token at 64K–256K | Restore KV and one KDA snapshot, then prefill only the new tokens (MiMo: 0.16–0.23 s at 64K) |

**Return path.** At prefill sizes the four ranks reduce among themselves before
returning one plane (see [DESIGN.md](DESIGN.md) §3). The coordinator therefore
receives a quarter of the bytes that a four-plane return would carry. This
matters on switches without priority flow control, where four ranks converging on
one port drop packets.

## 6. Start-up

- **Sparks:** each rank loads its 38–43 GB expert quarter from local NVMe in parallel.
  **Measured** for EXL3 K4 (38.39 GB per rank), with the images in the page cache:
  44–49 s per rank from start to listening. That is the boot readback (size and
  SHA-256 of 42 layer images, about 17–19 s) plus device preparation (26–31 s). A
  cold read from NVMe adds its read time.
- **Coordinator:** loads about 14 GB.
- **Graphs:** captured lazily, per shape.
- **Expected time to ready: 30–60 s.**

## 7. What would change these numbers most

1. **Fixed cost per MoE layer.** It is the largest uncertain term (8–11 ms of a
   20–27 ms cycle). DS41RT-class exchange code sits at the low end.
2. **DFlash2 acceptance on GLM-5.3-Flash by content type.** Only low-entropy
   acceptance (about 93–94%) has been measured.
3. **D2** (FP8 KDA): about −2.7 ms per step and +4.4 GiB of pool.
4. **D6** (expert format): FP8 experts cost about 20% at C1, and more when
   verifying.
