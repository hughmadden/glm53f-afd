# Expected performance

**Status: model, not measurement (28 September 2026).** Every figure below is
derived from published or measured numbers of related engines on the same class
of hardware. Each derivation is shown so it can be checked and replaced as real
receipts arrive. Layout and format letters refer to [SIZING.md](SIZING.md).

## 1. Anchors

| Anchor | Value | Source |
|---|---|---|
| MiMo-V2.6-Flash, 4 Sparks + 5090 (same hardware class, similar expert shape) | C1 109.7 tok/s; C16 416.7 tok/s aggregate; prefill 3,630 / 5,123 / 4,816 / 4,283 tok/s at 2K / 8K / 32K / 64K; a 993,795-token prompt in 17.3 min | [mimo26f-afd](https://github.com/hughmadden/mimo26f-afd) v1.1–1.2 benchmarks |
| MiMo plain decode (no drafter) | about 24.5 ms per token over 47 MoE layers | the same engine, measured |
| DS41RT one-row expert phase per layer | 328 µs, of which 157 µs is the kernel | [tpurtell/ds41rt](https://github.com/tpurtell/ds41rt) `docs/ds41-expert-boundary-breakdown.md` |
| EXL3 K4 expert slice on GB10 | 20.5 µs marginal per 4.72 MB TP4 slice, about 230 GB/s | [glmrt](https://github.com/tpurtell/glmrt-5.3-1rtx-4spark) v9 route-cost profile |
| DFlash2 on GLM-5.3 (the full model) | code replay 5.78 tokens per target cycle; 67.9% acceptance on an 8-type mix | glmrt v9 README |
| GLM-5.3-Flash on 2 × RTX PRO 6000 (vLLM, EXL3 4 bpw + DFlash2, no Sparks) | C1 reasoning coding 140.9 tok/s; 128K prefill 4,984 tok/s; 1M six-needle retrieval 6/6 | [glm-5.3-flash-ext3-2x-rtx](https://github.com/tpurtell/glm-5.3-flash-ext3-2x-rtx) v0.8.0 |

**Why MiMo is the closest anchor.** GLM-5.3-Flash and MiMo-V2.6-Flash have the
same expert shape: 4,096 × 2,048 with top-8, so each rank computes 3 × 4,096 ×
512 per route. GLM has **42 MoE layers against MiMo's 47**. It is heavier on the
coordinator: KDA projections in BF16, and 4 mHC streams.

## 2. Decode: one target cycle, one row (M1)

| Term | Derivation | Layout B, EXL3 K4 | Layout C (FP8 KDA) |
|---|---|---:|---:|
| Spark expert reads | 42 layers × 8 slices × 3.15 MB at 230 GB/s | 4.6 ms | 4.6 ms |
| Coordinator weight reads | 13.96 GB (B) or 9.27 GB (C) at 85–90% of 1.79 TB/s | 8.7–9.2 ms | 5.8–6.1 ms |
| Fixed cost per MoE layer | 0.20–0.27 ms × 42. Covers exchange, Spark fixed cost and per-layer launches. The range spans DS41RT to MiMo. | 8.4–11.3 ms | 8.4–11.3 ms |
| KDA state, indexer, mHC | 0.3 GB of FP32 state per step; top-512 over n/4 pools; 90 mHC boundaries | 1.0–1.5 ms | 1.0–1.5 ms |
| **M1 cycle** | | **22.7–26.6 ms** | **19.8–23.5 ms** |
| **Target-only decode** | | **38–44 tok/s** | **43–51 tok/s** |

NVFP4 experts add about 0.6 ms per cycle; FP8 experts add about 4.6 ms.

## 3. Decode with DFlash2 (block of 8)

- **Extra experts per verify row.** Each row routes to new experts. Scaled from
  GLM-5.3's measured unique-expert counts, 8 rows touch about 40 distinct experts
  per layer, against 8 for one row.
- **Cost of 8 rows.** The extra experts cost (40 − 8) × 42 × 13.7 µs ≈ 18 ms on
  the Sparks. Add about 1 ms of coordinator row work and 1.5–2.5 ms for the draft
  pass (2.18 GiB of BF16 weights).
- **Round length (layout B).** About 43–48 ms at 8 rows, 37–43 ms at 6 rows and
  32–38 ms at 4 rows.

| Workload | Tokens per round | Round | Expected C1 |
|---|---:|---:|---:|
| Code (high acceptance) | 5.5–5.8 | 43–48 ms | **115–135 tok/s** |
| Weighted mix (code, maths, prose, JSON, short answers) | about 3.3 | 34–38 ms | **80–110 tok/s** |
| Low-entropy text (counting, templated output) | about 6.5 | 43–48 ms | **135–150 tok/s** |

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
  - each active request reads and writes 272 MiB of FP32 KDA state per step, so
    C16 adds about 2.5 ms per step;
  - the BF16 KDA projections.

## 5. Prefill

**Two sides, two lanes.** Prefill pipelines the coordinator and the Sparks across
two lanes, so it runs at the slower side's rate.

| Side | Work per token | Rate if it were the limit |
|---|---|---|
| **Sparks** | 42 × 8 × 25.2 M = 8.46 G MAC, 0.89× MiMo | MiMo's rate ÷ 0.89: about **5.4–5.8K tok/s** at 8K–32K |
| **Coordinator, projections** | about 6.8 G MAC (13.6 GFLOP), most of it BF16 KDA | **7–14K tok/s** |

**Context-dependent coordinator work** comes on top of the projections:

- **Sparse MLA:** each query reads at most 2,051 tokens, about 3 GFLOP per token.
- **Indexer:** 2,048 × n FLOP per token per layer over n/4 pools, plus top-512
  selection. Averaged over a 1M prompt this is about 11 GFLOP per token, and about
  50 µs per token of selection.
- **KDA:** linear in n, so the attention cost stays nearly flat as context grows,
  unlike the global-attention layers in MiMo.

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
