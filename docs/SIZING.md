# Context and memory sizing

**Status: draft for discussion (28 September 2026).** Nothing here has been
measured on the engine yet, because the engine does not exist yet. Model and
checkpoint byte counts are exact: they come from `config.json`, the
safetensors headers of each checkpoint and the reference modeling code. Runtime
and workspace figures are estimates, labelled as such.

Target: **GLM-5.3-Flash** (`zai-org/GLM-5.3-Flash`, architecture
`glm5_next`) on **one RTX 5090 (32 GB)** coordinator and **four DGX Spark (GB10,
128 GB unified memory)** expert ranks.

Sources:

- Model: `zai-org/GLM-5.3-Flash` @ `eb9eb208` (official FP8 checkpoint, 328.3 GB).
- Expert-quantized checkpoints:
  - `brandonmusic/GLM-5.3-Flash-tr3-4bpw` @ `a5fee929` (EXL3 4 bpw, 175.6 GB);
  - `LibertAIDAI/GLM-5.3-Flash-NVFP4` @ `caca4e6a` (NVFP4, 194.7 GB).
- Drafter: `incoai/GLM-5.3-Flash-DFlash2` @ `bf582e4e` (2.34 GB).
- Cache formats: `transformers` `models/glm5_next/modeling_glm5_next.py`
  (upstream main, September 2026).

## 1. The model, as far as memory is concerned

- **45 layers**, hidden size 4,096, with **4 mHC residual streams** (manifold-constrained
  hyper-connections, the DeepSeek-V4 formulation, 20 Sinkhorn iterations).
- **34 KDA layers** (Kimi Delta Attention, linear attention): 64 heads × 128.
  Each request carries a fixed-size recurrent state. **Nothing grows per token.**
- **11 DSA layers** (3, 7, 11, …, 43): MLA with a 512-dim latent and **no RoPE**
  (`qk_rope_head_dim = 0`), 64 heads of 256.
  - Every DSA layer runs its own indexer (32 heads × 128).
  - The indexer pools keys **4 tokens → 1** with a learned gate. It selects the
    top **512 pools (2,048 tokens)** and adds the incomplete tail pool (up to 3
    tokens), so sparse attention reads at most 2,051 cached tokens per layer.
- **MLP:** layers 0–2 are dense (width 12,288). Layers 3–44 are MoE: **288 routed
  experts, top-8**, plus 1 shared expert, expert width 2,048. Routing uses
  sigmoid scores with bias correction, a routed scaling factor of 2.5, and SwiGLU
  clamped at 10.
- **Drafters:** MTP layer 45 (native), and the DFlash2 drafter (5 layers, block
  of 8, 2,048-token sliding window).
- **Maximum context:** 1,048,576 tokens (`max_position_embeddings`).

## 2. What grows with context (per token)

| Store | FP8 | BF16 | Note |
|---|---:|---:|---|
| MLA latent, 11 layers | 5,808 B | 11,264 B | FP8 record: 512 FP8 values + 16 B of scales = 528 B per layer |
| Indexer pooled keys, 11 layers | 363 B | 704 B | One 128-dim key per 4 tokens, plus a 4 B scale for FP8 |
| **Total per token** | **6,171 B** | **11,968 B** | |
| One 256K request | 1.51 GiB | 2.92 GiB | |
| One 1M request | 6.03 GiB | 11.69 GiB | |

For comparison, the full GLM-5.3 (78 layers, MLA with RoPE on every layer)
needs about 57 KB per token at FP8, **nine times** as much.

## 3. What is fixed per active request

| Store | Size | Note |
|---|---:|---|
| KDA recurrent state, FP32 | 136 MiB | 34 × 64 × 128 × 128 × 4 B. The reference implementation keeps it in FP32. |
| Short-convolution state | 4.8 MiB | 3 steps × 24,576 channels × 34 layers, BF16 |
| DFlash2 draft KV | 40.2 MiB | 5 layers × 8 KV heads × 128 × K and V, BF16, 2,056 tokens |
| **Total per slot** | **181 MiB** | 16 slots: 2.8 GiB. 8 slots: 1.4 GiB |

Speculative decoding does not multiply the KDA state. After a verify pass the
engine replays the accepted rows from the state saved before the round, instead
of storing one state per draft position (8 × 136 MiB per request).

## 4. Coordinator weights (RTX 5090)

Byte counts are from the official checkpoint headers.

| Group | Official checkpoint | Resident on the 5090 |
|---|---|---:|
| KDA attention, 34 layers | **BF16**, 9.37 GB | 9.37 GB, or 4.69 GB if we quantize its projections to FP8 ourselves (128 × 128 block scales; conv, norms, `A_log` and `dt_bias` as shipped) |
| DSA attention and indexer, 11 layers | FP8 + BF16, 1.64 GB | 1.64 GB |
| Shared experts, 42 layers | FP8, 1.06 GB | 1.06 GB |
| Dense MLP, layers 0–2 | FP8, 0.45 GB | 0.45 GB |
| Routers, mHC, norms | 0.17 GB | 0.17 GB |
| LM head | BF16, 1.27 GB | 1.27 GB |
| Embedding | BF16, 1.27 GB | 1.27 GB, or 0 if kept in host RAM (row gather) |
| **Total** | | **14.18 GiB** (13.00 GiB with the embedding in host RAM; 8.64 GiB with FP8 KDA as well) |
| DFlash2 drafter | BF16, 2.34 GB | 2.18 GiB (shares the target's embedding and LM head) |
| MTP layer (optional second drafter) | 0.24 GB here | its experts sit on the Sparks |
| Vision tower (optional) | BF16, 1.13 GB | paged in per image request |

- **The official FP8 checkpoint leaves KDA attention in BF16**, and that group is
  61% of the coordinator's weights. Quantizing it to FP8 is our own change, and it
  needs a quality gate (decision D2).
- The NVFP4 and EXL3 checkpoints ship every non-expert tensor in BF16 (15.4 GiB
  without the embedding). So the engine loads the coordinator's tensors from the
  official checkpoint, whatever expert format the Sparks use.

## 5. RTX 5090 budget

The 5090 reports 32,607 MiB (31.84 GiB). The estimate for runtime and workspace
is **3–5 GiB**: CUDA context, graphs, prefill activations (4 mHC streams × 4,096
per token), KDA chunk buffers and indexer score tiles. The KV pool is whatever
remains.

**16 slots, DFlash2 resident.** Ranges run from 5 GiB of runtime to 3 GiB.

| Layout | KV pool | Tokens, FP8 KV | 256K requests at once | 1M requests at once | Tokens, BF16 KV |
|---|---:|---:|---:|---:|---:|
| A. Official precision, embedding on GPU | 7.7–9.7 GiB | 1.33–1.68 M | 5–6 | 1 | 0.69–0.87 M |
| **B. Official precision, embedding in host RAM** | 8.8–10.8 GiB | **1.54–1.89 M** | **5–7** | **1** | 0.79–0.97 M |
| C. B plus FP8 KDA projections | 13.2–15.2 GiB | 2.30–2.64 M | 8–10 | 2 | 1.18–1.36 M |
| D. All-BF16 non-expert (quantized checkpoints as shipped) | 6.4–8.4 GiB | 1.11–1.46 M | 4–5 | 1 | 0.57–0.75 M |

- With 8 slots instead of 16, each layout gains about 1.4 GiB (about 0.25 M FP8 tokens).
- **A single request can use the model's full 1,048,576 tokens in every layout
  with FP8 KV.** With BF16 KV, layouts A, B and D cap one request at about
  0.6–1.0 M tokens.
- The pool is shared: admission reserves each request's prompt plus its output
  allowance, not the maximum context.

## 6. Spark budget, per rank (TP4)

Each rank holds a quarter of every routed expert, split over the expert's
2,048-wide intermediate dimension.

| Expert format | Per rank | MTP experts | Left of ~113–116 GiB | Read per token at one row (M1) |
|---|---:|---:|---:|---:|
| EXL3 K4 (`tr3-4bpw`) | 38.4 GB (35.8 GiB) | 0.9 GB | ~75 GiB | 1.07 GB, 4.6 ms at 230 GB/s |
| NVFP4 (modelopt) | 42.8 GB (39.9 GiB) | 1.0 GB | ~70 GiB | 1.19 GB, 5.2 ms |
| FP8 (official) | 76.1 GB (70.9 GiB) | 1.8 GB | ~40 GiB | 2.11 GB, 9.2 ms |

- A DGX Spark exposes about 121 GiB to Linux. About 113–116 GiB is available when
  the system is otherwise idle (measured on the reference setup).
- Context never touches the Sparks: all KV and KDA state lives on the coordinator.
- An EXL3 rank holds slightly more than a quarter of the routed bytes: every rank keeps the
  Hadamard sign vectors and codebook markers of its sliced projections whole (0.22 GB per rank).
- The spare memory keeps the **FP8 experts** option open. They fit with room to
  spare, at roughly twice the Spark read time per token (decision D6).
- 230 GB/s is the marginal read rate an EXL3 K4 kernel reached on GB10 for
  GLM-5.3 slices of the same format.

## 7. Host RAM tier (the coordinator's host)

- **KV:** 64 GiB of page-locked RAM holds 11.1 M tokens of FP8 KV.
- **KDA snapshots:** a KDA layer has no per-token cache, so a prefix can resume
  only where its recurrent state was saved. Each snapshot is 141 MiB (state plus
  convolution). A retained conversation keeps two: prompt end and turn end.
- **Capacity:** 64 GiB holds about **98 retained 64K conversations** or about **35
  retained 256K conversations**.

## 8. Prefix reuse with KDA

1. **MLA and indexer pages** are shared through a radix tree at pool (4-token)
   granularity, as usual.
2. **KDA state** is saved at the end of every prompt and every completed turn.
   That covers the common agent case, where each new turn extends the last. An
   exact or extending match resumes with no recompute.
3. **Optional checkpoints** every N tokens during long prefills. At N = 32K, a
   256K prompt keeps 8 snapshots (1.1 GiB), which belong in host RAM. They let a
   divergent branch resume from the last checkpoint before the divergence,
   instead of from zero.
4. **A match between two snapshots** resumes from the earlier one and re-prefills
   the rest.

## 9. Decisions to discuss

| # | Decision | Options | Proposal |
|---|---|---|---|
| D1 | KV precision | FP8 528-B record · BF16 · (NVFP4) | **FP8.** 1M on every layout and twice the capacity. Published, on one window: 4-bit experts with an FP8 MLA cache score a KLD of 0.0246; with an NVFP4 cache 0.0548, a configuration that failed the card's task-level test, so NVFP4 is out. Gate: the engine's own 25-window KL against BF16 ([KL-GATE.md](KL-GATE.md)), plus a needle ladder to 1M. |
| D2 | KDA projection weights | BF16 as shipped · our own FP8 | **Start BF16.** Measure FP8: +4.4 GiB of pool and about −2.7 ms per decode step, but the official checkpoint deliberately keeps these in BF16. |
| D3 | Slots | 16 (2 lanes × 8) · 8 | **16**, as in the engines this borrows from. |
| D4 | KDA snapshot cadence | prompt and turn only · plus every 32K | **Prompt and turn first.** Add periodic checkpoints if branch reuse shows up in real traffic. |
| D5 | Request cap on the API | 1,048,576 · lower default | **1M,** with admission reserving prompt plus output allowance. |
| D6 | Expert format on the Sparks | EXL3 K4 · EXL3 K6 · NVFP4 · FP8 | **EXL3 K4.** Published KLD is 0.0246 against 0.0206 for official FP8 (25 windows, offline, no KV-cache quantization), at half the bytes. K6 (0.0137) is the upgrade path; see [DESIGN.md](DESIGN.md) §5. |
| D7 | Embedding table | GPU · host RAM | **Host RAM** (+1.18 GiB of pool). It costs a gather of M rows × 8 KB per step. |
| D8 | KDA state precision | FP32 (reference) · BF16 | **FP32.** BF16 saves only 68 MiB per slot. |
