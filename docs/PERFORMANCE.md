# Expected performance

**Status: measured and modelled (29 September 2026).** §0 gives the first
measurements of the whole engine on its target hardware. The other sections are
the model it was designed against, derived from published or measured numbers of
related engines; they are kept so each derivation can be checked against §0.
Layout and format letters refer to [SIZING.md](SIZING.md).

## 0. Measured on the target hardware

**Setup:**
- One RTX 5090 coordinator and four DGX Spark (GB10) expert ranks, over RoCE v2 RDMA at 200 Gb/s.
- EXL3 K4 experts, FP8 MLA cache, BF16 KDA projections, the embedding in host RAM.
- All 45 layers, 16 slots, the DFlash2 drafter.
- Defaults since 29 September 2026:
  - four prefill lanes of 2,048 rows;
  - BF16 KDA states (D8, which passed the KL gate);
  - two decode lanes for passes of 2–16 rows;
  - copy windows for greedy requests;
  - the rank's GB10 decode schedule;
  - the chunked KDA prefill with W8A16 projections, which passed the KL gate in a second run that
    day. That run's rows below name its engine, `073b553`, and say whether the pair was on.
- Thinking on (the model's default) unless stated. Single runs; between runs, ±2–3% is typical.
- `GLM53F_PROFILE=1` (the lane trace most of these runs used) costs nothing measurable: on
  `073b553` traced and untraced boots decoded at the same speed within noise (for example 117.6 /
  66.3 / 168.2 against 117.7 / 66.8 / 168.1 tok/s) with identical replies.

**Decode, one stream** (tok/s; 1,024-token code and counting, a 903-token prose answer):

| Case | Code | Prose | Counting |
|---|---:|---:|---:|
| No drafter (28 Sep) | 52.7 | 52.7 | 52.7 |
| DFlash2 (chain τ 0.7), greedy, **thinking off (the template's Low effort)**, 29 Sep, `073b553` | **125.6** | **68.8** | **180.3** |
| The same with the chunked KDA prefill and W8A16 (today's defaults) | 122.6 | 68.2 | 179.5 |
| DFlash2 (chain τ 0.7), greedy, empty think block, 29 Sep (copy windows off) | 115.0 | 63.7 | 133.1 |
| The same prompts, 29 Sep, `073b553` with its defaults (D8, decode lanes, copy windows on) | 117.7 | 66.8 | 168.1 (266 tokens) |
| The same, 28 Sep | 111.3 | 62.4 | 128.4 |
| DFlash2, greedy, thinking on (28 Sep) | 82.7 | 73.3 | 169.1 |
| DFlash2, sampled (T 0.7), empty think block (28 Sep) | 113.5 | 67.0 | 123.1 |

- The empty think block was "thinking off" until 29 September; since then thinking off is the template's Low effort, and the API no longer renders the empty block (`docs/DESIGN.md`). On `073b553` the empty block's counting answer stopped after 266 tokens instead of counting to 400. That failure is the mode's, not D8's: with F32 KDA states the empty block fails the count too, and Low counts to 400 with either.
- The chunked KDA prefill with W8A16 changes the replies (all six differed within their first 340 characters: the prompt's own prefill takes the new arithmetic) and the speed by −2% to +5%.
- Without a drafter, 52.7 tok/s is 19.0 ms per token.
- With the drafter, 51–71% of verified drafts are kept (τ 0.3–0.7), 3.0–3.4 tokens per verify window.
- The model below expected 36–43 tok/s without a drafter (§2) and 110–130 / 75–105 / 125–145 with it (§3).

**Like-for-like with the fastest public four-Spark TP4 recipe** ([mmastrac/glm-5.3-flash-4x-gx10](https://github.com/mmastrac/glm-5.3-flash-4x-gx10) branch `perf-2026-09-27`):
- Its own `dev/repro/decode.py` prompts and method: 512 tokens, T 0, median of three after a warm-up, tok/s including the time to the first token.
- Its "thinking off" renders the model's `Reasoning Effort: Low` with `<think>` left open. Here that is `reasoning_effort: "low"`.

| Mode | Structured | Code | Prose | Drafts kept |
|---|---:|---:|---:|---|
| **This engine, `reasoning_effort: "low"`** (what its thinking off renders since 29 Sep) | **186.1** | **142.3** | **78.3** | 84.3%, 4.22 tokens a window |
| The same, 29 Sep, `073b553` | 185.6 | 136.6 | 76.8 | 82.2%, 4.04 tokens a window |
| This engine, empty `<think></think>` (its thinking off until 29 Sep, and `reasoning_effort: "none"` too; the API no longer renders it) | 167.2 | 124.1 | 71.4 | — |
| The same, `073b553` | 169.8 | 129.0 | 75.1 | 82.0%, 3.85 |
| This engine, thinking on (default effort) | 167.5 | 131.3 | 85.7 | 84.2% |
| The same, `073b553` | 170.5 | 124.5 | 85.2 | 86.4%, 4.17 |
| That recipe (commit `3e03894`, its message; its thinking off) | 167.2 | 118.7 | 64.4 | 88.5–97.2% on structured |

In the matched mode this engine is 11% / 20% / 22% faster (11% / 15% / 19% on `073b553`), with 4-bit experts that keep BF16 activations. That recipe's fastest build takes 4-bit activations in its prefill experts. `073b553` runs BF16 KDA states, whose greedy replies differ from the F32 states' of the first run, so drafts kept and reply lengths move with them (an inference: with the same options the two builds score identical logits, [KL-GATE.md](KL-GATE.md) §6d).

**RigMark** ([alexellis/rigmark](https://github.com/alexellis/rigmark) `c5a0db0`, MIT; 29 Sep, `073b553`):
- Its agent workloads at reasoning effort Low, 4,096 tokens, 5 runs, `--skip-prefill`.
- The comparison id `ringside-redhat-rowsplit-20260926` and seed 20260905 that mmastrac's gate uses (`gate/run.sh` at `203fc05`), so the prompts are byte-identical to that recipe's.
- Its decode rate leaves out the time to the first token.

| Workload | This engine (median; range) | Basic gates | mmastrac's recipe, as reported |
|---|---:|---|---:|
| Code | **123.9** (122.9–128.7) | 5/5 | 107.9 |
| Prose | **65.8** (65.6–68.5) | 5/5 | 61.7 |
| Structured | **174.9** (173.8–176.3) | 5/5 | 157.1 |

- 15 of 15 gates passed; +15% / +7% / +11%. That recipe's figures are those of its `experimental/README.md` at `f88710f`.
- RigMark's capped concurrent phase (short code, 256 tokens a stream) gave 97.7 / 137.2 / 189.2 tok/s aggregate at 1 / 2 / 4 streams.

**Concurrency with a distinct prompt per stream** (aggregate tok/s; 29 Sep, `073b553`, before the chunked KDA prefill and W8A16 became the default; 512-token greedy streams at Low effort, DFlash2 τ 0.7, the median of three; 1–16 streams at 16 slots, 32 and 48 at 48 slots):

| Prompts | C1 | C4 | C16 | C32 | C48 |
|---|---:|---:|---:|---:|---:|
| **Mixed** (code, prose and structured in turn) | 118.8 | **159.5** | **288.5** | **412.5** | **525.8** |
| Code | 118.5 | 194.2 | 383.5 | 576.8 | 738.9 |
| Structured | 131.5 | 186.7 | 353.6 | 447.8 | 583.9 |
| Prose | 73.9 | 156.4 | 268.7 | 383.6 | 495.2 |
| One topic on every stream, the same build (400-token streams, thinking on) | — | 168.5 | 321.3 | 485.7 | 605.0 |

- **Mixed against one topic:** −5% at C4, −10% at C16, −15% at C32, −13% at C48. Prose is 7–21% below the one-topic figure; code is 15–22% above it (code drafts well).
- **Why they differ:** streams that route alike share expert reads (103 experts per exchange at 41 rows, against about 197 under independent routing). The two probes also differ in tokens, effort and content, so read the gap as a range, not as the cost of routing alone.
- **The chunked KDA prefill with W8A16** (today's defaults) measured decode-neutral: code 121.9 / 204.3 / 380.0 and mixed 121.8 / 156.8 / 286.4 tok/s at C1 / C4 / C16, against 118.5 / 194.2 / 383.5 and 118.8 / 159.5 / 288.5 without it.
- **mmastrac's recipe** reports 317 and 451 tok/s for code at 16 and 32 streams with a prompt per stream, and 404 for 50 mixed streams (the message of its commit `e9839ea`). Its prompts are not these.

**Concurrency with one topic on every stream** (aggregate tok/s; 400-token streams, DFlash2 τ 0.7; the earlier runs, optimistic for mixed traffic as above):

| Streams | 16 slots, 29 Sep (copy windows off) | 16 slots, `--decode-lanes 2-16` (now the default) | 48 slots, D8, four prefill lanes, 29 Sep | 16 slots, 28 Sep | 48 slots, 28 Sep |
|---|---:|---:|---:|---:|---:|
| C2 | 107.7 | **116.2** | — | — | — |
| C4 | 147.7 | 157 (28 Sep) | — | 141 | — |
| C8 | 210.4 | **216.6** | — | — | — |
| C16 | 316.5 (325.9 with copy windows on) | 299 (28 Sep) | 325.9 | 300 | 295 |
| C32 | — | — | 476.8 | — | 456 |
| C48 | — | — | **600.8** | — | 573 (427 without drafting) |

- **Context at 48 slots:** with D8 and four lanes the pool is 4.94 GiB (859,072 tokens) and the largest request 850,816 tokens (28 Sep: 470,592 and 462,336). At 16 slots a 1,048,576-token request fits.
  - The chunked KDA prefill's workspace grows with the slots (34 MiB a slot: 544 MiB at 16, 1.59 GiB at 48), and W8A16 takes 64 MiB of GEMM scratch. At 16 slots the pool measured 8.73 GiB (1,519,424 tokens) with them, against 9.33 GiB (1,622,784) without: a 1,048,576-token request still fits.
  - At 48 slots the two take about 1.66 GiB of the pool, which leaves about 3.28 GiB (about 570K tokens). This is computed from the workspace's size, not measured. `--kda-chain-prefill --prefill-w8a8` gives the 4.94 GiB back.
- **Decode is bound by the ranks' weight reads.** At C16 the verify passes carry about 41 rows. Of each MoE layer's 2.24 ms (28 Sep), 1.73 ms was the exchange and 0.34 ms the coordinator's attention.
  - The rank's GB10 decode schedule (`crates/glm53f-rank/README.md`, "Measured on GB10") reads expert weights at 221–237 GB/s in its bench and 198–233 GB/s in service (28 Sep: 187–234). GB10's reads top out at about 225–241 GB/s in practice (273 on paper).
  - More rows per call share more of those reads: 48 streams reach 601 tok/s.
- **Drafting** is worth +34% at C48. A draft budget of 128 rows or τ 0.5 each cost about 1.5%.

**Copy windows** (greedy requests copy spans of their own context in place of drafts; lossless):
- Measured with mimo26f-afd's `harness/copy_bench.py` at v1.3.0, thinking off, 3 runs.
- Replies are byte-identical with copy windows on and off.

| Case | On | Off | Change |
|---|---:|---:|---:|
| Rewrite a file (two cases) | 173.5 / 167.1 | 133.1 / 127.1 | **+30% / +31%** |
| Quote a function | 117.9 | 107.7 | +9% |
| An `edit_file` tool call (end to end) | 165.4 | 158.3 | +4.5% |
| Fresh code / fresh prose | 114.4 / 76.2 | 114.4 / 76.3 | 0 / 0 |

Copied windows averaged 7.8 tokens with 97.1% of copied tokens kept.

**Prefill** (tok/s; one prompt at a time):

| Prompt | 4K | 19K | 79K |
|---|---:|---:|---:|
| **Four lanes of 2,048 with the chunked KDA prefill and W8A16 (the defaults since 29 Sep; `073b553`, two boots)** | **4,979–4,998** | **5,128–5,180** | **5,132–5,188** |
| Four lanes of 2,048 without them (the default until then; `073b553`, two boots) | 4,099–4,129 | 4,103–4,104 | 4,051–4,053 |
| Four lanes of 2,048, 29 Sep (the first run) | 4,145 | 4,138 | 4,089 |
| Three lanes of 2,048 | 4,138 | 4,048 | 4,074 |
| Two lanes of 4,096 | 3,771 | 4,041 | 4,013 |
| Two lanes of 2,048, 29 Sep | 3,505 | 4,014 | 3,978 |
| Four lanes with the chunked KDA prefill and W8A16, the first run (on an unbalanced bond) | 4,956 | 5,180 | 5,190 |
| Two lanes of 2,048, 28 Sep (rank kernel tuned for GB10) | 3,338 | 3,678 | 3,626 |
| Two lanes of 2,048, 28 Sep (exchange fast paths, earlier rank kernel) | 3,033 | 3,369 | 3,350 |
| Two lanes, host encode and pageable uploads | 2,717 | 2,964 | 2,899 |
| One lane | 1,633 | 1,725 | 1,715 |

- **The chunked KDA prefill with W8A16 projections** is 21–28% faster than without them: +21–22% at 4K, +25–26% at 19K, +27–28% at 79K. Every `073b553` row comes from a boot whose RDMA bond was balanced (each port carrying 42–58% of the return traffic).
  - A MoE layer of four 2,048-row lanes takes 28.2–29.6 ms against 42.8, and the GPU is 66–69% busy against 86%: the exchange sets the pace again.
  - It passed the 125-window paired KL comparison ([KL-GATE.md](KL-GATE.md) §6d: mean −0.0014 nats, upper bound +0.0002 against the 0.002 margin), and is the default since 29 September.
  - The first run's row came from an unbalanced bond (62.8% of the return traffic on one port) and matches the balanced boots: the bond did not limit it at these rates.
- **With four lanes, without the pair:**
  - A 79K-token prompt takes 19.3 s; a 207K-token one 53.7 s (3.9K tok/s).
  - The GPU is 86% busy, attention is the bound again, and each lane's exchange hides behind the other lanes' attention.
  - At 16 slots all forward buffers take 2.80 GiB and a 1M-token request fits (KV pool 8.26 GiB with the first run's F32 KDA states; 9.33 GiB with BF16 states on `073b553`). With the pair the buffers take 3.40 GiB (its KDA workspace 544 MiB, W8A16's GEMM scratch 64 MiB) and the pool 8.73 GiB.
- **The 5090's attention per 2,048-row lane** (`GLM53F_PROFILE_OPS=1`, the median; §5a has the development GPU):

  | Op | Without the pair (29 Sep, F32 KDA states) | With it (`073b553`) |
  |---|---:|---:|
  | KDA layer: attention | 8.515 ms | **4.259** |
  | – the chain (`kda_core`), then the chunked kernel | 5.249 | **0.983** |
  | – the `[q\|k\|v\|b]` projection (BF16) | 1.940 | 1.948 |
  | Shared expert (FP8) | 0.405 | 0.610 |
  | DSA layer: attention | 9.656 | 12.163 |
  | – the indexer's selection | 0.180 | 1.875 |
  | – `o_proj`, `q_b`, `q_a` (FP8) | 1.013, 0.313, 0.134 | 1.535, 0.530, 0.200 |
  | – the sparse attention | 5.234 | 5.300 |

  - The chunked kernel takes 4.3 ms off each KDA lane.
  - W8A16 makes each FP8 GEMM about 1.5 times slower.
  - The indexer's selection is ten times slower in this arm. An inference, not isolated: it is W8A16's doing, since the chunked kernel touches KDA layers only and the indexer's queries stay BF16.
  - Faster W8A16 GEMMs and that selection are the next prefill lever: up to about 2.5 ms per DSA lane.
- **The earlier story, in order:**
  - Without the exchange fast paths, the GPU was busy 66% of a 27.5 ms MoE layer. With them it was 72% of 25.1 ms.
  - With the GB10-tuned rank kernel a 2,048-row lane's exchange fell to 12 ms, longer than the other lane's attention (8.7 ms). That is why two lanes left the GPU idle and four do not.
  - The reduce-scatter return is slower over a TCP mesh between the ranks: they encode and sum peer rows on the CPU. It stays off.

**KV snapshots and the host RAM tier** (first run on the target, 29 Sep; `GLM53F_HOST_CACHE_GB=24`):

| Step | Time to first token |
|---|---:|
| A 207,436-token prompt, cold | 53.67 s |
| The same prompt again (device snapshot) | 0.01 s |
| Again, after 30 other long prompts pushed its snapshot out of the device (restored from RAM) | 0.04 s |

The restore itself took 21.4 ms. A 36K-token snapshot's store to RAM took 9.3 ms.

**Snapshots go to RAM only on demand** (29 Sep, `073b553`, the same workload at 16 slots, every store and restore counted from the log):
- Cold 53.22 s; again 0.01 s (the device snapshot); after the 30 other prompts, 0.04 s: the 207,436-token snapshot came back from RAM in 22.5 ms.
- Nothing was stored to RAM while the pool and the slots had room. Once every slot held a finished conversation, each new prompt of about 36K tokens took the least recently used one's slot and sent that conversation to RAM (the 207K one at the 16th prompt: 3,241 pages in 38.7 ms).
- 20 stores in all, every one such an eviction; one restore; no errors.
- Not yet exercised on the target: the page-pressure path, where an incoming request needs pages the pool lacks. With 16 slots and 36K-token prompts the slots ran out long before the 1.62M-token pool.

**Known limitation: a running stream nearly stops while a long prompt prefills** (29 Sep, `073b553`, before the prefill pair; the design of the prefill-blocking probe of mmastrac's recipe, `experimental/quality/prefill_block.py` at `0784b1b`, reimplemented):
- A 64,596-token prompt, sent while another stream was generating, prefilled in 16.52 s (3,911 tok/s).
- Meanwhile the running stream got 1.0 tok/s, with gaps of up to 3.96 s (75.5 tok/s before, 67.8 after).
- Six short requests sent during the prefill got their first tokens in 1.1–3.1 s, all before the long prompt's: new requests are admitted between its passes.
- The fix belongs to the scheduler and is not built: bounded prefill slices, or a decode step between prefill passes.

**Against the public four-Spark recipes** (their reported figures; this engine as above). Both
recipes run on four GB10 systems alone, with no RTX 5090, so the differences belong to the added
GPU and this engine together:

| Metric | [tonyd2wild](https://github.com/tonyd2wild/GLM-5.3-Flash-NVFP4-1M-KV-4x-DGX-Spark) (vLLM TP4, NVFP4) | mmastrac `perf-2026-09-27` (vLLM TP4, NVFP4) | This engine |
|---|---|---|---|
| Single stream | ~55 tok/s | 167.2 / 118.7 / 64.4 (structured / code / prose); RigMark 107.9 / 61.7 / 157.1 (code / prose / structured) | 186.1 / 142.3 / 78.3 in the same mode; RigMark 123.9 / 65.8 / 174.9 |
| Aggregate | 530 tok/s at 48 streams | 253 tok/s at 16 streams; code with a prompt per stream 317 / 451 at 16 / 32 | a prompt per stream, mixed: 288.5 at 16, **525.8 at 48** (48 slots); code alone 383.5 / 576.8 / 738.9 at 16 / 32 / 48 |
| Prefill | 3.5–4.1K tok/s short; 1.9K at 114K | 4,956 / 4,808 at 32K / 128K, cold | **5.0–5.2K** at 4K–79K (4.1K without the default prefill pair) |
| Context | 1M | 512K | 1M (16 slots); 851K at 48 slots without the default prefill pair, about 570K with it (computed) |

- mmastrac's prefill takes 4-bit activations in its experts, which its own test puts 13–62% away from BF16 activations at the MoE output.
- This engine's experts keep BF16 activations. Its KL against the BF16 teacher equals the published figure for its 4-bit expert checkpoint.
- Sources, pinned: tonyd2wild's README at `2ac4e8d` (its 1M fp8 lane for the aggregate, the 114K prefill and the context). For mmastrac, the single-stream and 16-stream figures are those in the message of commit `3e03894` on `perf-2026-09-27`, and the prefill figures those of its `experimental/README.md` from `889a456`. That branch has moved on since: at `74faf89` the same file reports 170.3 / 120.8 / 65.8, 251.3 at 16 streams and 4,946 / 4,750 prefill. Its RigMark figures are in the same file at `f88710f`, and the code figures with a prompt per stream in the message of `e9839ea`.

**Start-up:** the coordinator is ready 7 s after launch (weights from the page cache). A rank is ready
in 44–49 s (§6).

**Quality spot checks:**
- a number hidden at 37% depth is retrieved from 8.8K and 79K tokens of filler, with and without the drafter;
- `harness/api_contract.py` passes all 12 rows on the real model.

**KL gate** against the published BF16 teacher ([KL-GATE.md](KL-GATE.md) §6a, §6b, §6d):
- decode path 0.0245 nats, top-1 95.1%, equal within its standard error to the published 0.0246
  for the same 4-bit experts; prefill path 0.0282 nats, top-1 94.7%; both pass;
- every change merged by 29 September left the engine's output bit-identical;
- D8 passed and is on; D2 (FP8 KDA projections at 128 × 128 scales) failed and stays off;
- the chunked KDA prefill with W8A16 passed a paired comparison on 125 windows (mean −0.0014
  nats, upper bound +0.0002 against the 0.002 margin) and is on.

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

### 5a. Where a lane's coordinator time goes (development GPU, 28 September 2026)

**Setup:**
- One RTX 4090 (`sm_89`, 128 SMs). This card is capped at 250 W; its SM clock read 2.52 GHz
  under load.
- `crates/glm53f-forward/examples/prefill_bench.rs`: all 45 layers on the weights of layers
  0–4 (every later KDA MoE layer runs layer 4's, every DSA layer layer 3's), the routed experts
  returning zeros, one request in two-lane passes.
- `GLM53F_PROFILE_OPS=1` medians of the third pass (positions 8,192–12,288 with lanes of 2,048
  rows; 16,384–24,576 with lanes of 4,096), so every DSA row selects its full 2,051 tokens.
- Cross-checked once with an `nsys` timeline of the same pass: kernel medians within 1–3% of
  the op times, which also count launch gaps.
- The GPU is shared with other jobs. Runs were repeated; a figure a repeat did not match within
  about 2% says so.

**Peaks assumed:**

| | RTX 4090 | RTX 5090 | Ratio |
|---|---:|---:|---:|
| BF16/F16 tensor, FP32 accumulation, dense (NVIDIA's Ada and Blackwell GeForce whitepapers) | 165.2 TFLOPS | 209.5 TFLOPS | 1.27 |
| FP32, CUDA cores | 82.6 TFLOPS | 104.8 TFLOPS | 1.27 |
| DRAM | 1,008 GB/s | 1,792 GB/s | 1.78 |

- cuBLAS reached **145 TFLOPS** of BF16 here, so the 4090's BF16 rate is not the "about 83"
  sometimes quoted: that is its TF32 or CUDA-core FP32 rate. The 5090's tensor and FP32 rates
  are only **1.27×** the 4090's (tensor cores × clock); its DRAM is 1.78×.
- FP8 GEMMs are scaled by the same 1.27. No FP8 peak is assumed.
- "5090 floor" is the op's work at the 5090's peak for its bound. "5090 estimate" is the 4090
  time divided by the ratio for that bound, the 4090's efficiency kept.

**KDA MoE layer** (31 of the 42 MoE layers; the `PIPE` line's median is one of these):

| Op | 2,048 rows (ms) | 4,096 rows (ms) | Work per 2,048-row lane | Bound | 5090 floor | 5090 estimate |
|---|---:|---:|---|---|---:|---:|
| `attn_hc_project`: mHC attention boundary, expand and project | 0.199 | 0.402 | 172 MB | DRAM | 0.096 | 0.11 |
| `attn_hc_finish`: weights, Sinkhorn, collapse, norm, E4M3 | 0.097 | 0.222 | 99 MB | DRAM | 0.055 | 0.06 |
| `kda_proj`: the `[q\|k\|v\|b]` projection (cuBLAS BF16) | 2.852 | 5.671 | 413 GFLOP | BF16 tensor | 1.97 | 2.24 |
| `kda_proj`: the forget and output gate projections | 0.136 | 0.254 | 13 GFLOP, writes 67 MB | BF16, DRAM | 0.06 | 0.09 |
| `kda_core`: the KDA chain (conv, norms, gates, delta rule, gated norm) and the conv shift | 6.313 | 12.331 | 2,048 dependent steps per head | latency | — | ≈5.0 (inferred) |
| `kda_o`: `o_proj` (cuBLAS BF16) | 1.004 | 1.955 | 137 GFLOP | BF16 tensor | 0.66 | 0.79 |
| `ffn_hc_project`: mHC FFN boundary, expand and project | 0.181 | 0.346 | 158 MB | DRAM | 0.088 | 0.10 |
| `ffn_hc_finish` | 0.095 | 0.222 | 99 MB | DRAM | 0.055 | 0.06 |
| `router`: FP32 logits, top-8 of 288 | 0.241 | 0.459 | 4.8 GFLOP | FP32 | 0.046 | 0.19 |
| **attention** | **11.21** | **22.04** | | | | **8.68 measured on the target** |
| shared expert (FP8 gate and up, SwiGLU, down) | 0.538 | 1.003 | 103 GFLOP | FP8 tensor | — | 0.42 (target: 0.39) |

- The profile reports a KDA layer's three stages (`kda_proj`, `kda_core`, `kda_o`) from the marks
  in its code, which carries no hooks of its own. The rows here split `kda_proj` with finer
  events in a development build.
- The chain is serial in the rows: one block of 1,024 threads per head, 64 blocks. It spends
  3.1 µs per row per layer here and occupies 64 of the 5090's 170 SMs.
- Its 5090 figure is **inferred**, not measured: the target's measured 8.68 ms less the other
  ops' estimates (3.6 ms).
- On the 5090 the BF16 projections are then about 3.1 ms of the 8.7 and the chain about 5.0.

**DSA MoE layer** (11 of 42):

| Op | 2,048 rows (ms) | 4,096 rows (ms) | Work per 2,048-row lane | Bound | 5090 floor | 5090 estimate |
|---|---:|---:|---|---|---:|---:|
| mHC boundaries, both (as above) | 0.561 | 1.212 | 528 MB | DRAM | 0.29 | 0.32 |
| `q_a`, `kv_a`, the `q_a` norm and its E4M3 form (FP8) | 0.254 | 0.349 | 34 GFLOP | FP8 tensor | — | 0.20 |
| `q_b` (FP8) | 0.492 | 0.961 | 103 GFLOP | FP8 tensor | — | 0.39 |
| indexer projections (cuBLAS BF16) | 0.222 | 0.467 | 31 GFLOP | BF16 tensor | 0.15 | 0.18 |
| widening the index query and gates to f32 | 0.060 | 0.129 | 60 MB | DRAM | 0.03 | 0.03 |
| latent, pooled-key and tail writes | 0.019 | 0.022 | small | launches | — | 0.02 |
| index selection (top-512 pools) | 0.480 | 1.755 | grows with the context (8–12K, 16–24K) | F16 tensor, latency | — | 0.38 |
| absorb (`W_UK` into the query; FP32, separate multiply and add, bit-exact with the CPU) | 1.258 | 2.628 | 34 G FP32 instructions | FP32 issue | 0.66 | 0.99 |
| sparse MLA over 2,051 latents (F16 tensor cores) | 5.711 | 12.374 | 551 GFLOP | F16 tensor | 2.63 | 4.50 |
| un-absorb (`W_UV`; 3 FP32 instructions per 2 products), BF16 out | 1.533 | 3.292 | 26 G FP32 instructions | FP32 issue | 0.49 | 1.21 |
| E4M3 form of the heads' output | 0.095 | 0.195 | 100 MB | DRAM | 0.056 | 0.05 |
| `o_proj` (FP8) | 1.182 | 2.609 | 275 GFLOP | FP8 tensor | — | 0.93 |
| router | 0.226 | 0.496 | 4.8 GFLOP | FP32 | 0.046 | 0.18 |
| **attention** | **12.08** | **26.87** | | | | **≈9.4** |

- The sparse MLA kernel runs at 58% of the 4090's F16 peak, and the absorb at 66% of its FP32
  issue rate. The un-absorb (41%) and the selection have the most headroom.
- The selection grows with the context: 0.16 ms at 0–4K, 0.48 ms at 8–12K and 1.76 ms at
  16–24K (4,096-row lanes), about 0.2 µs per row per 10K tokens of context.
- At 79K it would be about 3.5 ms a lane, the layer's second-largest op (extrapolated, not
  measured). There it is bound by its F16 products: 162 MFLOP a row.

**What this package changed** (bit-identical; the same digest of 162 rows of logits before and
after, and with MLA blocks of 8, 64, 100, 512 and 1,000 rows; `examples/logits_digest.rs`):

| Per lane and layer | 2,048 rows: before → after | 4,096 rows: before → after |
|---|---:|---:|
| Un-absorb, 16 rows per block in shared memory, 32 outputs per warp reduced by recursive halving, BF16 out | 2.643 → 1.533 | 5.668 → 3.292 |
| Query widening (the absorb reads BF16) | 0.286 → 0.060 | 0.567 → 0.129 |
| f32 → BF16 copy (now in the un-absorb) and E4M3 | 0.285 → 0.095 | 0.614 → 0.195 |
| Router, 4 experts × 8 rows per warp | 0.33–0.35 → 0.23–0.24 | 0.66–0.71 → 0.46–0.50 |
| mHC finish, collapsed row written only for taps | 0.110 → 0.096 | 0.245 → 0.222 |
| **DSA MoE layer** (the changed ops: −1.66 ms at 2,048 rows) | **13.98 → 12.08** | **31.46 → 26.87** |
| **KDA MoE layer** (the changed ops: −0.14 ms) | 11.60 → 11.21 | 22.75 → 22.04 |
| **Pass wall** (both lanes, all 45 layers) | **1,214 → 1,143–1,146 ms** | **2,491 → 2,311 ms** (2,455 in a run another job shared) |

The layer rows also move by the run-to-run variation of unchanged kernels: the chain and the
BF16 projections vary by about 3% between runs.

- The un-absorb kernel with 32 rows per block and one block per SM, and one reading o_lat
  through L1, both measured slower.
- 32 rows per absorb block measured no faster: the absorb is bound by its separate multiplies
  and adds.
- With 2 head groups per sparse MLA block instead of 4 (the same bits), and with MLA blocks of
  256, 512 or a whole lane, sparse MLA timed within 1.5% of the default on the 4090.
- On `sm_120` the 4-group kernel spills registers (the DSA crate's README), so the target
  should time 2 groups: `GLM53F_BENCH_HEAD_GROUPS=2` in `prefill_bench`.

**The lane scratch** (`ForwardBuffers`, 16 slots, 128 verify rows, page tables for 1M tokens;
allocated and measured on the development GPU):

| | Before | After |
|---|---:|---:|
| Per row, lane A (lane B has no tap buffer: 32 KiB less) | 799,416 B | 286,136 B |
| Sparse MLA block buffers, shared by the lanes (`mla_block_rows` 512) | — | 128 MiB |
| Lanes of 2,048 rows | 1.60 + 1.46 GiB | 0.62 + 0.48 GiB |
| Lanes of 4,096 rows | 3.13 + 2.93 GiB | 1.17 + 0.97 GiB |
| All forward buffers, lanes of 4,096 | 6.67 GiB | 2.88 GiB |

- The "before" rows are the figures the target's start-up log printed.
- A KDA layer's buffers, a DSA layer's and an FFN's are now views of one region per lane:
  129,796 B a row, the DSA set, the largest.
- The sparse MLA core's absorbed queries and latent outputs (256 KiB a row) are sized for one
  block of rows.
- Lanes of 4,096 now take less memory than lanes of 2,048 took before.

**Levers**, by expected ms saved per 2,048-row lane on the 5090 (estimates from the tables
above):

| # | Lever | Per KDA layer | Per DSA layer | Per pass per lane (45 layers) | Bits | Cost |
|---|---|---:|---:|---:|---|---|
| 1 | **Chunked KDA prefill** (the kernel exists: `ForwardConfig::kda_chunked_prefill`). Measured here: chain 6.31 → 1.92–2.12 ms (12.3 → 4.0–4.1 at 4,096 rows), pass 1,206–1,228 → 980–1,045 ms | ≈ −3.5 | — | ≈ −120 | change (f32 rounding; KL gate) | **done**: with W8A16 it passed the gate and is on by default since 29 September; on the 5090 the chain's 5.25 ms became 0.98 (§0) |
| 2 | **FP8 KDA projections (D2)** with E4M3 activations in prefill (`--kda-fp8 --kda-prefill-w8a8`). Measured on the 4090 per 2,048-row lane: projections 3.31–3.63 → 2.31 ms and `o_proj` 1.04–1.06 → 0.70–0.86 ms, about −1.3 to −1.5 ms (−30%); scaled by 1.27 | ≈ −1.0 to −1.2 | — | ≈ −35 to −40 | change (KL gate) | the flags exist; a gate run |
| 3 | The KDA chain alongside the projections: row blocks of `[q\|k\|v\|b]` and `o_proj` on the 106 SMs the chain leaves idle | up to −3 | — | up to −100 | same, with a GEMM algorithm fixed per row block | moderate; moot after (1) |
| 4 | Sparse MLA prefill: consecutive rows' selections shared (each latent tile decoded once for several rows), a causal kernel for the dense start | — | −1 to −2 | −11 to −22 | same, if each row keeps its tile order and products | a new kernel |
| 5 | Index selection with FP8 × FP8 products: at long context it is bound by its F16 products (row blocks, the DSA README's other step, save only L2 traffic there) | — | small now; ≈ −1.4 at 79K | ≈ −15 at 79K | change (KL gate) | a kernel change |
| 6 | **Done here:** the DSA core, router and boundary changes above | −0.1 | ≈ −1.3 | ≈ −18 | same | done |
| 7 | Un-absorb staging double-buffered (1.21 against a floor of 0.49) | — | ≈ −0.5 | ≈ −5 | same | small |
| 8 | Absorb with fused multiply-adds or on tensor cores | — | ≈ −0.6 | ≈ −7 | change (the absorb is bit-exact with the CPU by contract) | small, plus the gate |
| 9 | mHC finish fused into the projection (the last CTA of a row finishes it, as the decode boundary does) | ≈ −0.05 | ≈ −0.05 | ≈ −2 | same | moderate |

- (1), (2) and (3) change the KDA kernels or their dispatch: (1) passed the KL gate with W8A16
  and is a default (§0), (2) failed it in this form (D2), and (3) is moot after (1).
- D2 is chiefly a decode lever (half the KDA weight bytes). In prefill it saves about 12–14% of
  a KDA layer on the 5090, and about 20–23% once the chunked kernel has removed the chain.
  From the same measurements: `--prefill-w8a16` (BF16 activations for the FP8 projections over
  8 rows) costs 12–14 µs a token, and D2 with `--kda-prefill-w8a8` saves about 12 µs a token net
  (estimated).

### 5b. Sparse MLA in passes of 9 to 170 rows (development GPU, 29 September 2026)

A decode, verify or short prefill pass of more than 8 rows runs the sparse attention in one split
(the split count moves bits, so it stays fixed) and, until now, with 4 head groups per block: one
512-thread block per row. A block holds over half of a multiprocessor's shared memory, so such a
pass filled as many multiprocessors as it had rows: 41 of the 5090's 170 for the verify windows of
16 streams, 32 of the 4090's 128 at 32 rows.

The head groups only partition the heads, so 1, 2 and 4 give the same bits at every row count
(`sparse_attn_head_groups_are_bitwise`, 1 to 256 rows; and the same logits digest, before and
after, for two-lane prefills in lanes of 12, 32, 48, 64 and 100 rows). The forward now takes the
fewest groups whose grid (`rows × 4 / groups` blocks) still fits one wave of the multiprocessors
(`mla_head_groups`): 1 group up to a quarter as many rows as the GPU has multiprocessors (42 rows
on the 5090, 32 on the 4090), 2 up to half (85, 64), then 4 as before. Passes that fill the GPU
with 4 groups, prefill lanes of hundreds of rows among them, are unchanged, and there is no new
setting (`prefill_head_groups` is the most).

Measured with `dsa_ab -- mid` on the RTX 4090 (128 multiprocessors): 2,051 selected tokens per
row, one call after an L2 flush, best of three, µs. "Before" is the previous build, 4 groups:

| Rows | Groups now | Blocks | 128K context: before → after | 1M context: before → after |
|---:|---:|---|---:|---:|
| 9 | 1 | 9 → 36 | 314 → 108 (2.9×) | 317 → 114 (2.8×) |
| 16 | 1 | 16 → 64 | 315 → 109 (2.9×) | 317 → 111 (2.9×) |
| 32 | 1 | 32 → 128 | 316 → 113 (2.8×) | 320 → 115 (2.8×) |
| 41 | 2 | 41 → 82 | 317 → 166 (1.9×) | 321 → 167 (1.9×) |
| 48 | 2 | 48 → 96 | 317 → 171 (1.9×) | 322 → 180 (1.8×) |
| 64 | 2 | 64 → 128 | 322 → 187 (1.7×) | 333 → 197 (1.7×) |

- A block over 2,051 tokens takes 316, 160 and 109 µs with 4, 2 and 1 groups (9 to 32 rows: one
  block to a multiprocessor). So 2 groups cost the same multiprocessor time per row as 4, and 1
  group about 40% more; once the blocks outnumber the multiprocessors that extra time is lost,
  which is why the choice stops at one wave. Over the 11 DSA layers, 16 rows save about 2.3 ms a
  pass on the 4090 (inferred from this table, not measured end to end).
- In the forward (`prefill_bench`: one lane, all 45 layers on the weights of layers 0–4, a
  2,880-token prompt so that the last pass selects 2,051 tokens, op profile on) a DSA layer's
  sparse attention in the last pass fell from 0.310 to 0.102 ms at 16 rows, from 0.308 to 0.105 ms
  at 32 and from 0.310 to 0.161 ms at 48, and the median pass from 32.1 to 29.5, 33.8 to 32.1 and
  35.3 to 34.5 ms (the median counts the early passes, which select few tokens).
- Above half as many rows as multiprocessors the settings are close: at 96 and 128 rows 4 groups
  is within 11% of the best; at 170 rows, just over one wave, 2 groups timed 15–21% faster; at 256
  and 512 rows the winner changes with the context. Those passes keep the prefill setting.
- Not measured on the 5090, which has 170 multiprocessors and whose 4-group kernel spills more
  registers than on the 4090 (the DSA crate's README): run `dsa_ab -- mid` there. If 2 groups win
  above half the multiprocessors' rows, set `prefill_head_groups` to 2 for it; the rule takes it
  as the most.

## 6. Start-up

- **Sparks:** each rank loads its 38–43 GB expert quarter from local NVMe in parallel.
  **Measured** for EXL3 K4 (38.39 GB per rank), with the images in the page cache:
  44–49 s per rank from start to listening. That is the boot readback (size and
  SHA-256 of 42 layer images, about 17–19 s) plus device preparation (26–31 s). A
  cold read from NVMe adds its read time.
  - **Since 29 September 2026** a rank gives each image's cached pages back once it is uploaded
    (`crates/glm53f-rank/README.md`, "The page cache"), so every start reads its images from NVMe.
    The cold figure has not been measured on the target hardware; the boot log now prints
    `MemAvailable` and `MemFree` before the readback, after it and after preparation.
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
