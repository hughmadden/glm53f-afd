# Context and memory sizing

**Status: draft for discussion (28 September 2026).** Nothing here has been
measured on the engine yet, because the engine does not exist yet. Model and
checkpoint byte counts are exact: they come from `config.json`, the
safetensors headers of each checkpoint and the reference modeling code. Runtime
and workspace figures are estimates, labelled as such.

> **Note (29 September 2026).** Written before the engine; §10 was added as the options were
> built. Since then the engine has run on its target hardware ([PERFORMANCE.md](PERFORMANCE.md)
> §0), and the KL gate has decided two of the decisions below ([KL-GATE.md](KL-GATE.md) §6b):
> D8 (BF16 KDA states) passed and is on by default, halving the per-slot KDA state of §3 to
> 68 MiB; D2 (FP8 KDA projections) failed and stays off. Later that day the chunked KDA prefill
> with W8A16 projections passed on 125 windows (KL-GATE.md §6d) and is on by default too: its
> workspace takes 34 MiB a slot (544 MiB at 16 slots, 1.59 GiB at 48) and W8A16 64 MiB of GEMM
> scratch, so at 16 slots the measured pool is 8.73 GiB (1.52 M tokens; a 1,048,576-token request
> fits). The measured pools and largest requests are in PERFORMANCE.md §0, and the coordinator's
> start-up log prints its own plan.

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
| D2 | KDA projection weights | BF16 as shipped · our own FP8 | **Start BF16.** Measure FP8: +4.4 GiB of pool and about −2.7 ms per decode step, but the official checkpoint deliberately keeps these in BF16. *Built, off by default (`--kda-fp8`, §10): 4.26 GiB of weights less. Failed the KL gate on 29 September (KL-GATE.md §6b); stays off.* |
| D3 | Slots | 16 (2 lanes × 8) · 8 | **16**, as in the engines this borrows from. |
| D4 | KDA snapshot cadence | prompt and turn only · plus every 32K | **Prompt and turn first.** Add periodic checkpoints if branch reuse shows up in real traffic. |
| D5 | Request cap on the API | 1,048,576 · lower default | **1M,** with admission reserving prompt plus output allowance. |
| D6 | Expert format on the Sparks | EXL3 K4 · EXL3 K6 · NVFP4 · FP8 | **EXL3 K4.** Published KLD is 0.0246 against 0.0206 for official FP8 (25 windows, offline, no KV-cache quantization), at half the bytes. K6 (0.0137) is the upgrade path; see [DESIGN.md](DESIGN.md) §5. |
| D7 | Embedding table | GPU · host RAM | **Host RAM** (+1.18 GiB of pool). It costs a gather of M rows × 8 KB per step. |
| D8 | KDA state precision | FP32 (reference) · BF16 | **FP32.** BF16 saves only 68 MiB per slot. *Built (`--kda-state-bf16`, §10): 3.19 GiB at 48 slots; measured drift in §10. Passed the KL gate on 29 September (KL-GATE.md §6b) and is **on by default** since; `--kda-state-f32` restores FP32.* |

## 10. D2, D8 and the prefill activations as built (28 September 2026)

Three numerics options (the first with three kinds of scales, the third with a variant), each
**off by default** when built and each a flag of
`glm53f-serve` and `glm53f-score` (with an environment fallback). Each becomes a default only after the KL gate
([KL-GATE.md](KL-GATE.md) §6, `compare --margin 0.002` against the same engine without it) and
speed runs on the target hardware. Development-GPU figures are an RTX 4090 shared with other work.
*Outcome (29 September 2026, KL-GATE.md §6b): D8 passed and is now on by default; D2 failed;
W8A16, alone or with the chunked KDA prefill, lowered the mean KL but could not yet be shown
non-inferior on 25 windows, so it stays opt-in.* *Later on 29 September (KL-GATE.md §6d): W8A16
with the chunked KDA prefill passed on 125 windows, and both are on by default since;
`--prefill-w8a8 --kda-chain-prefill` turns them off. The chunked prefill's workspace takes 34 MiB
a slot and W8A16's scratch 64 MiB.*

**D2, FP8 KDA projections** (`--kda-fp8`, `GLM53F_KDA_FP8=1`).
- At load, the fused q|k|v|b projection and `o_proj` of the 34 KDA layers are quantized on the
  GPU to FP8 E4M3 with 128 × 128 block scales, the checkpoint's own scheme (`scale = amax / 448`,
  `q = e4m3(w / scale)`). q|k|v|b has 24,640 rows: 192 whole blocks and a 64-row block for beta
  with scales of its own (the FP8 GEMMs now take a partial last block of rows). The gate
  projections (6 MB a layer; the forget gate's decays compound), conv, norms, `A_log` and
  `dt_bias` stay as shipped.
- Memory: 275.5 → 141.0 MB a layer, 9.37 → 4.80 GB over 34 layers, **4.26 GiB less**.
- Decode (layers 0–4, `decode_bench`): per KDA layer `kda_proj` 0.23 → 0.12 ms and `kda_o` 0.077
  → 0.041 ms at one row, **about −4.9 ms per step over 34 layers** on the 4090 (about −2.8 ms
  scaled by the 5090's bandwidth: an estimate, not a measurement).
- Prefill: the KDA projections take the FP8 GEMMs, W8A8 (E4M3 activations; the default when this
  was measured) or W8A16 with `--prefill-w8a16` (the default since 29 September). Per KDA layer
  and pass (`decode_bench`, one lane, the chunked KDA kernel, three runs; the projections are
  q|k|v|b with the gate GEMMs, then `o_proj`):

  | Rows per pass | BF16 (today) | D2, W8A8 | D2 with `--prefill-w8a16` |
  |---:|---:|---:|---:|
  | 2,048 | 3.31–3.63 + 1.04–1.06 ms | 2.31 + 0.70–0.86 ms (−1.3 to −1.5 ms, −30%) | 3.57–3.93 + 1.13–1.24 ms (+0.3 to +0.5 ms) |
  | 4,096 | 6.60–8.10 + 2.14–2.20 ms | 4.49–4.68 + 1.45–1.78 ms (−2.8 ms) | 6.89–7.57 + 2.18–2.53 ms (+0.3 to +1.0 ms) |

  Over 34 layers D2 saves about 48 ms per 2,048-row pass on the 4090 (23 µs per token); with W8A16
  as well the KDA projections cost about 14 ms more than BF16 (7 µs per token), on top of the
  W8A16 cost of the other FP8 projections below. `--kda-prefill-w8a8` (with `--kda-fp8
  --prefill-w8a16`) keeps the FP8 KDA projections at W8A8 and the others at W8A16: the KDA
  layers then run at D2's speed (2.26–2.51 + 0.70 ms per 2,048 rows, 4.48–4.95 + 1.45–1.46 per
  4,096), about −12 µs per token against today over the whole model (an estimate from the parts).
  The prefill of 8,192 tokens through layers 0–4 in passes of 2,048 rows: 222–230 ms, 195–204
  with D2, 256–259 with D2 and W8A16, 223–233 with the three.
- Error against the oracle (layers 0–4, `goldens_chain`): the projections move by 1.4–1.9%
  relative RMS (0.2% for the reference in BF16); head logits 1.55e-2 (8-row passes) and 2.7e-2
  (one pass) against 0.97e-2 and 2.4e-2 without it; argmax 7/9, the two rows that differ being the
  golden's near-ties (top-2 logit gaps 0.042 and 0.006).

**D2 with power-of-two scales** (`--kda-fp8-pow2`, `GLM53F_KDA_FP8_POW2=1`) **and as MXFP8**
(`--kda-mxfp8`, `GLM53F_KDA_MXFP8=1`), built 29 September 2026 after D2 failed the KL gate
([KL-GATE.md](KL-GATE.md) §6e has why, and the weights' error).
- 82–89% of the q, k, v and o weights have at most 3 significant mantissa bits, and a power-of-two
  scale (the smallest ≥ amax / 448) keeps them exactly. Against the BF16 weights, q, k, v and o
  move by 3.2–6.2e-4 (relative RMS) instead of D2's 2.3–2.8e-2; `b_proj` stays at 2.7e-2.
- `--kda-fp8-pow2` keeps D2's 128 × 128 blocks, layout, kernels and bytes.
- `--kda-mxfp8` is the OCP Microscaling layout: an E8M0 scale (one byte) per row and 32 values of K.
  - Memory: 141.0 → 145.2 MB a layer, 4.94 GB over 34 layers, **4.13 GiB less** than BF16 (0.13
    GiB more than D2).
  - Its own GEMM entry points: the decode GEMM scales each 16-value step by its row's scale; the
    W8A8 prefill GEMM scales each k32 product sum by its column's scale as it adds it to the block
    sum (the k32 structure, always); the W8A16 path's BF16 tiles take the scale in the product.
  - Activations as D2's: BF16 in decode and with W8A16 (the default since 29 September), E4M3
    per 128 with f32 scales in W8A8 (`--prefill-w8a8`). The weights are in the layout of sm_120's
    block-scaled MMA, which would need MXFP8 activations as well (not built).
- Speed on the 4090 (`gemm_bench`, five runs, minima; `prefill_bench`, three runs; `decode_bench`,
  two runs). Powers of two run D2's kernels at D2's speed.
  - Decode: the MXFP8 GEMMs take D2's time within 3%. Per KDA layer at one row, `kda_proj` 0.127
    ms and `kda_o` 0.042 ms (D2 0.122 and 0.040, BF16 0.229 and 0.076).
  - Prefill GEMMs, W8A8 per 2,048 rows: q|k|v|b 1.93 ms (D2 1.90), `o_proj` 0.61 ms (D2 0.52: 18%
    slower). The 34 KDA layers' projections in one pass: 87.4 ms (D2 83.0, BF16 121.6).
  - A whole prefill (`prefill_bench`, four lanes of 2,048 rows, 45 layers, with the KDA chain and
    W8A8 as the bench then ran): 3,770 tok/s, against 3,910 with D2, 3,880 with powers of two and
    3,550 with BF16 KDA projections.
- Error against the oracle (layers 0–4, `goldens_chain`, the prompt's rows; MXFP8, with BF16 and
  D2 in brackets):
  - In 8-row passes q, k and v are where BF16 has them: layer 0's q|k|v|b output 2.3e-3 (2.2e-3,
    1.4e-2). Its `b_logits`, whose weights keep D2's error, 5.4e-3 (1.8e-3, 5.3e-3). Head logits
    0.95e-2 (0.97e-2, 1.55e-2), argmax 9/9 (9/9, 8/9).
  - In one pass the E4M3 activations dominate: q|k|v|b 1.3e-2 (2.2e-3, 1.9e-2), head logits
    2.5e-2 (2.4e-2, 2.7e-2), argmax 8/9 (9/9, 7/9).
  - Powers of two per 128 × 128: the same per layer; head logits 1.02e-2 and 2.6e-2 in the chain.

**D8, BF16 KDA states** (`--kda-state-bf16`, `GLM53F_KDA_STATE_BF16=1`).
- The state is stored in BF16 and every kernel computes in f32. The chain and the replay round
  the state after every row, so a verify round committed at k rows still gives the bits of k
  serial decode steps (`verify_commit` passes with it). The chunked prefill rounds after every
  16-row chunk.
- Memory: 136 → 68 MiB per slot and 376 → 195 pool pages per snapshot mark (141 → 73 MiB, and
  the host tier's images likewise): **3.19 GiB at 48 slots**.
- Speed (`kda_bench`, 34 layers, the recurrent part of a step): decode chain 0.60 → 0.34 ms at
  one request and 3.34 → 2.02 ms at eight; the commit replay of 8 × 8 rows 3.39 → 2.56 ms.
- **Drift** (`crates/glm53f-kda/tests/gpu_bf16_state.rs`, 8 heads, synthetic inputs): after 8K
  rows the state is 2.1e-3 (relative RMS) from exact arithmetic with gates across the range and
  **2.6e-2 when every channel decays slowly** (multipliers ≥ 0.9993: one step's decay is below
  half a BF16 ulp, so rounding swallows it); the f32 chain is at 5.6e-8 and 6.0e-7. Over 64K rows
  in eight segments the difference from the f32 state stays flat (2.0e-3 and 2.6e-2 at every
  segment): it does not grow, but the slow channels carry it. On the oracle's real prompt 5.3–6.1%
  of the key channels of layers 0 and 4 decay slower than 0.998 per token. The KL gate on 2,048-
  token windows decides. An FP16 state (same bytes, 3 more mantissa bits) measured 7.7× less drift
  on the CPU model (not built).

**W8A16 prefill projections** (`--prefill-w8a16`, `GLM53F_PREFILL_W8A16=1`; on by default since 29
September, with the chunked KDA prefill; `--prefill-w8a8` or `GLM53F_PREFILL_W8A16=0` turns it off).
- FP8 projections over 8 rows take BF16 activations instead of E4M3 per 128-group (the likely
  source of the prefill path's +0.0038 nats, KL-GATE.md §6a): each weight is dequantized to BF16
  tiles of rows (`bf16(e4m3 × scale)`, one rounding) in 64 MiB of scratch and multiplied by cuBLAS.
  Decode (8 rows or fewer) is unchanged.
- Error (`tests/fp8_gemm.rs`, against exact W8A16 products): mean 2e-4 to 3e-5 of Σ|x·w| against
  2e-3 to 4e-4 for W8A8, about 10× less.
- Speed (`gemm_bench prefill`, four runs): the FP8 projections of one pass of the model take
  43–48 → 67–74 ms at 2,048 rows and 91–99 → 144–157 ms at 4,096 rows on the 4090, **+12 to +14 µs
  per token**. With D2 as well, the KDA projections run at about the speed of today's BF16 ones
  (133–157 against 120–148 ms per 2,048 rows). A prefill of 8,192 tokens through layers 0–4
  (`decode_bench`, one lane, the chunked KDA kernel): 222–236 ms, 250–256 with W8A16, 195–208 with
  D2, 256–275 with both.
- Error against the oracle (`goldens_chain`, the prompt in one pass): head logits 2.41e-2 → 0.97e-2
  (relative RMS), the level of the 8-row decode path (0.97e-2); the dense MLP of layer 0 3.3e-2 →
  0.98e-2. Every FP8 projection contributes to the prefill path's excess (the outputs of q_a,
  kv_a, q_b, the DSA `o_proj`, the shared and the dense MLPs are each 1.6–5 times further from the
  oracle with E4M3 activations than in 8-row passes), so the option covers them all.

**A development-model proxy** (not the gate): `glm53f-score --experts zero --dev-load-layers 5`
(all 45 layers on repeats of layers 0–4, routed outputs of zeros) on 7 panel windows × 189 rows,
KL in nats against the same engine with every option off. The prefill path against the decode path
is 3.6e-2 (top-1 0.81); with `--prefill-w8a16` 0.79e-2 (0.90). Against the decode path, D2 is
4.9e-2 (0.77) in decode, 6.7e-2 in prefill, 5.0e-2 in prefill with W8A16 and 5.7e-2 with
`--kda-prefill-w8a8`; D8 1.6e-2 (0.85). On 29 September, with BF16 states, against the decode
path: powers of two 1.15e-2 (0.88) and MXFP8 0.97e-2 (0.89) in decode, where D2 is 5.1e-2 (0.78);
both 4.6e-2 (0.79) in prefill (D2 6.9e-2, neither 3.5e-2); MXFP8 in prefill with W8A16 1.2e-2
(0.88; D2 4.8e-2, neither 0.86e-2). Two arms whose weights differ in one of 10,000 (powers of two
per 128 × 128 and MXFP8) are 1.2e-2 apart in decode, so that is this proxy's floor. The
development model's sensitivity is not the real model's; the KL gate on the target hardware
decides.

