# Reference oracle and golden fixtures

> **Payloads are not in git.** The committed `manifest.json` and `tensors.tsv` files pin every
> fixture's sha256; the `.bin` payloads (about 136 MB) are regenerated with `golden_layers.py
> generate` (about 1–4 minutes on CPU, weights required) and checked with `verify`. A regeneration
> that does not reproduce the committed digests is a failure.


The engine is tested against **goldens**: what the reference implementation of GLM-5.3-Flash
computes, layer by layer, for one fixed prompt and eight decode steps. The reference is the
`glm5_next` code in `transformers` 5.17.0, run unmodified on the CPU with the official FP8 weights.
This directory holds the recipe (a pinned container image and `golden_layers.py`) and the fixtures
it wrote (`goldens/`).

The whole model does not fit in one machine's memory, so `golden_layers.py` builds the reference
modules one decoder layer at a time and loads their weights itself. `selftest_tiny.py` checks that
shortcut against `from_pretrained` and the whole-model forward on a tiny random checkpoint in the
official format: every parameter and every recorded output is bit-identical (276 checks).

## What is recorded

- **Prompt:** one user turn, rendered with the checkpoint's `chat_template.jinja` and tokenised
  with its `tokenizer.json`: 33 tokens, `[gMASK]<sop><|system|>Reasoning Effort: Max<|user|>Briefly,
  in two sentences: why does the sky look blue during the day but red at sunset?<|assistant|><think>`.
  33 is 1 mod 4, so the last prompt position has a one-token k-pool tail.
- **Decode:** 8 more tokens (`Sunlight scatters off air molecules,`), fed one at a time with the
  reference cache. They are fixed text, not samples, so the goldens do not depend on the logits.
- **Layers:** 0 to 4 run as a chain, so layers 3 and 4 see their real inputs. Three are recorded:
  - layer 0: KDA + dense MLP;
  - layer 3: DSA (MLA with its k-pool indexer) + MoE;
  - layer 4: KDA + MoE.
- **Head:** the embedding rows, and the HyperHead mean, final RMSNorm and LM head applied to the
  last layer's output (layer 4 here; see Limits).

The chain runs layer-major: one layer processes the prompt and all eight steps, then the next layer
loads. With fixed decode tokens this is bit-identical to running token by token through the whole
stack (the self-test compares against the whole-model forward), and it keeps one layer in memory.

The token ids, the rendered text, the digests of the tokenizer and template files, the model
revision and the runtime are in every manifest's `source` block.

## Golden sets

| Set | Contents | Size |
|---|---|---:|
| `layer00-prefill` | layer 0 over the prompt: streams in, after the attention site, out; mHC pre, post, comb and collapsed input at both sites; both norms; KDA q/k/v projections before the conv, q/k/v after it, forget-gate projection, g, beta logits, beta, output gate, core output, gated-norm output, recurrent state and conv cache after the prompt; attention and MLP outputs; the layer's small KDA weights (`weights.*`) | 26.5 MB |
| `layer00-decode` | layer 0 for each decode step, stacked `[steps, ...]`: the same per-token tensors; the KDA state after every step for 4 heads and after the last step for all 64; the conv cache after the last step; `weights.*` again | 12.2 MB |
| `layer03-prefill` | layer 3 over the prompt: streams, mHC, norms; MLA query latent, query, `kv_a`, the 512-dim latent, attention probabilities, per-head output; indexer q, k, gate scores, head weights, pooled keys and their token indices, scores, top-k indices with the tail; router logits, top-8 ids and weights, routed, shared and MoE outputs; an `index_topk=16` variant (`k16.*`) | 19.4 MB |
| `layer03-decode` | layer 3 for each decode step; pooled keys, pool indices and scores per step (`decode.sN.*`), because the pool count grows | 4.7 MB |
| `layer04-prefill` | layer 4 over the prompt (KDA as layer 0, MoE as layer 3) | 27.6 MB |
| `layer04-decode` | layer 4 for each decode step | 12.4 MB |
| `head` | embedding rows for all 41 tokens; the head's input streams, HyperHead mean, final norm and FP32 logits for the 9 rows a decoder turns into logits (the last prompt token, then each step) | 7.1 MB |
| `native` | the reference in its native dtypes (BF16) for layers 0, 3 and 4 and the head, each fed its FP32 golden input: layer outputs; for KDA the kernel's inputs (q/k/v projections, forget-gate projection, beta logits, output gate), q/k/v after the conv, g, beta, core output and states; for DSA the top-k indices and latents; for MoE the routing and routed output; with the measured difference from the FP32 goldens | 25.7 MB |

Together about 136 MB. `goldens/source-tensors.tsv` lists every checkpoint tensor read (2,207),
with the digest of its raw bytes. The `template/` and `tokenizer/` directories next to these sets
hold chat-template and tokenizer goldens in their own formats, written by `template_goldens.py` and
`tokenizer_goldens.py` in the same image; `verify` and `compare` skip them.

## Fixture format

Each set is a directory `goldens/<set>/`:

- `manifest.json`:
  - `tensors`: `{name: {file, dtype, shape, sha256, desc}}`, in the order they were recorded;
  - `source`: model repo and revision, `transformers` version and wheel digest, runtime, prompt and
    decode token ids, dtype policy, weights digest;
  - `notes`: layout details for the set.
- `tensors.tsv`: the same `name, file, dtype, shape, sha256` rows, tab-separated, for readers
  without a JSON parser.
- `<name>.bin`: raw little-endian values, row-major, no header.
  - `f32`: IEEE binary32;
  - `bf16`: the 16-bit bfloat16 patterns;
  - `i32`: signed 32-bit integers (token and expert indices; -1 means "none").

Names:
- `prefill.<x>`: one row per prompt token (33);
- `decode.<x>`: stacked over the 8 steps, one token per step (the token axis is dropped);
- `decode.sN.<x>`: step N alone, where the shape changes from step to step;
- `native` set: `LNN.<x>` rows are the prompt tokens then the decode tokens.

Every tensor's meaning is in its `desc`. Conventions to know:
- **Streams** are `[tokens, 4, 4096]`. An mHC site feeds its block `sum_i pre[i] * old[i]`, then
  updates the streams as `new[i] = sum_j comb[j][i] * old[j] + post[i] * block_out` (the reference
  multiplies by the transpose of `comb`).
- **KDA state** is `[heads, 128 (k), 128 (v)]` in FP32; the output is `q^T S` per head. The state
  before decode step 0 is `prefill.kda.state` in the matching `-prefill` set.
- **KDA conv cache** holds the last 4 pre-conv inputs per channel (q, k, v channels in that order),
  oldest first. Only the last 3 reach an output.
- **KDA weights in the fixtures:** each KDA set carries `weights.self_attn.{q,k,v}_conv1d.weight`,
  `A_log`, `dt_bias` and `o_norm.weight` exactly as the checkpoint stores them (BF16 or F32), about
  0.23 MB, so a layer check needs only the large projections from the checkpoint. The module's conv
  weight is the concatenation of the q, k and v conv weights along the channel axis.
- **Indexer top-k rows** (width 2,051 = 2,048 + 3):
  - first, `min(512, complete pools)` pools x 4 token indices, in descending score order. Pools not
    yet visible to that query still take a slot, filled with -1;
  - then 3 tail slots: the tokens of the incomplete pool that the query can see, else -1;
  - then -1 padding.

  With 33 prompt tokens every complete pool is selected, so the `k16.*` variant (4 pools + tail,
  same weights, `index_topk=16`) is there to exercise pool dropping. It is not the model's setting.
- **Router ids** are recorded as the reference returns them (`torch.topk(sorted=False)`) and sorted
  by expert id (`*_sorted`). The routed output sums the experts in ascending id order.

## Numerics

**FP32 is the primary contract.** Every FP8 weight is dequantised with transformers' own
`Fp8Dequantize._dequantize_one`, the function `from_pretrained` uses when it dequantises this
checkpoint: `fp32(w) * weight_scale_inv` per 128 x 128 block, rounded once to FP32. BF16 and F32
tensors are upcast exactly. Then every module runs in FP32 on the CPU, with the reference's eager
attention and eager expert loop.

**The native set records what the reference computes in its own dtypes.** On a machine without
FP8 GPU kernels, `from_pretrained` dequantises the checkpoint to BF16 and runs in BF16. The
parameter dtypes it chooses (checked by the self-test) are:
- BF16 for everything, including the KDA short-conv weights: `conv1d` is listed in the model's
  `_keep_in_fp32_modules_strict`, but the loader keeps the checkpoint dtype for renamed tensors, and
  the conv weight is concatenated from three BF16 tensors;
- F32 for the tensors stored as F32 in the checkpoint: `A_log`, `dt_bias`, the mHC `base` and
  `scale`, and the router's `e_score_correction_bias`.

The reference upcasts internally where its code says so: the mHC mixing, the router, the KDA
recurrence and state, the norms. Activations between modules are BF16. Three BF16 roundings stand
out:
- the decoder layer casts mHC `post` and `comb` to BF16 before applying them;
- the expert loop adds each expert's weighted output into a BF16 accumulator;
- the KDA short conv is the unfused PyTorch fallback, so natively it rounds twice: `F.conv1d` rounds
  its output to BF16, then SiLU rounds again. A fused conv + SiLU that rounds once differs from it
  by 1 BF16 ulp on about 23% of the outputs (at most 3 ulps) on layer 0; `native/manifest.json`
  records this per KDA layer (`LNN.conv_fused_vs_reference`).

Each native layer is fed its FP32 golden input cast to BF16, with its own cache, so the
differences in `native/manifest.json` (`notes.diff_vs_fp32`) are one layer's rounding. They give a
scale for kernel tolerances:

| Native vs FP32 (41 rows) | Layer 0 | Layer 3 | Layer 4 |
|---|---:|---:|---:|
| output streams, relative RMS | 0.58% | 0.55% | 0.31% |
| MLP / MoE output, relative RMS | 1.0% | 0.61% | 3.8% |
| rows with the same top-8 experts | — | 36 | 39 |
| conv outputs a fused conv + SiLU moves by 1 ulp | 22.6% | — | 24.4% |

The head's native logits differ by 0.35% (relative RMS) and give the same argmax on all 9 rows.

**Routing sits on near-ties.** The FP32 router's 8th and 9th choice scores (sigmoid + correction
bias) are a median 0.002 apart at layers 3 and 4, and 14 of the 33 prompt rows of layer 3 are within
1e-3 (the closest, 5e-6). BF16 alone swaps an expert on 5 of 41 rows at layer 3 and 2 of 41 at layer
4, which is what moves layer 4's MoE output by 3.8%. Engine tests should compare routing with a
near-tie allowance, and check the expert path with the golden routing (`moe.topk_ids`,
`moe.topk_weights`) rather than their own. The indexer's pool scores are further apart: the
closest consecutive pair in layer 3's prompt is 0.013, on scores of 15 to 131.

**Row count changes the bits.** A CPU GEMM's rounding depends on how many rows it multiplies, so a
row computed alone differs in the last bits from the same row inside a batch. The goldens follow the
reference: the prompt goes through in one forward, each decode step is its own one-row forward,
and the head runs one row per forward, as generation does (`logits_to_keep=1`). The same holds for
KDA: the reference's chunked prefill and its token-by-token recurrence agree to about 1e-7
(relative RMS) in FP32, not bit for bit. An engine that batches differently should expect
differences at that level.

## Build and run

The base image must provide `python3` (3.12) with torch 2.13.0 (x86-64 build with MKL), numpy,
packaging, filelock, fsspec, typing-extensions and jinja2. The published goldens were made from a
CUDA 13.2 base with the torch 2.13.0+cu132 wheel; the oracle uses only the CPU. Network access is
needed only for the pip step; the wheels are pinned by version and sha256 in `requirements.lock`.

```sh
docker build -t glm53f-oracle:1 --build-arg BASE_IMAGE=<image with Python 3.12 + torch 2.13.0> oracle/
```

**Weights.** Either the official checkpoint, or two subsets with the original tensor names, fetched
with `scripts/fetch_tensors.py` (about 30 GB instead of 328 GB):

```sh
python3 scripts/fetch_tensors.py --repo zai-org/GLM-5.3-Flash --revision <revision> \
  --out /models/GLM-5.3-Flash-coordinator --select nonexpert
python3 scripts/fetch_tensors.py --repo zai-org/GLM-5.3-Flash --revision <revision> \
  --out /models/GLM-5.3-Flash-experts-L3-4 --select experts:3,4
```

**Generate.** No network, weights read-only, output owned by the calling user:

```sh
docker run --rm --network none --user "$(id -u):$(id -g)" \
  -v /models/GLM-5.3-Flash-coordinator:/weights/coordinator:ro \
  -v /models/GLM-5.3-Flash-experts-L3-4:/weights/experts:ro \
  -v "$PWD/oracle/goldens:/out" \
  glm53f-oracle:1 generate --weights /weights/coordinator --weights /weights/experts --out /out
```

The run takes about a minute on 16 threads once the weights are in the page cache (about 4 minutes
from a cold cache) and peaks at about 16 GiB of RAM; no GPU is used.

With the full checkpoint, pass one `--weights` directory and the revision (the subsets carry it in
their index; the official index does not). `--last-layer 44` then runs all 45 layers, and
`head/head.logits` become the model's own logits for the last prompt token and each decode step:

```sh
docker run ... -v /models/GLM-5.3-Flash:/weights/full:ro ... glm53f-oracle:1 generate \
  --weights /weights/full --revision <revision> --last-layer 44 --out /out
```

**Verify** the files against their manifests (digests and sizes):

```sh
docker run --rm --network none -v "$PWD/oracle/goldens:/g:ro" glm53f-oracle:1 verify /g
```

**Self-test** (no weights needed, a few seconds):

```sh
docker run --rm --network none --entrypoint python3 glm53f-oracle:1 /oracle/selftest_tiny.py
```

## Reproducibility

Before torch loads, `golden_layers.py` pins the CPU code paths: AVX2 kernels in ATen, MKL and
oneDNN (`ATEN_CPU_CAPABILITY=avx2`, `MKL_CBWR=AVX2,STRICT`, `ONEDNN_MAX_CPU_ISA=AVX2`), and it runs
with deterministic algorithms and a fixed thread count. Bit-for-bit regeneration needs:

- the same image: Python 3.12, the same torch 2.13.0 build, the pinned wheels;
- an x86-64 CPU with AVX2;
- 16 threads, the default. The thread count is part of the contract: when a reduction has fewer
  output rows than threads (every single-token decode step), ATen splits it across threads, and
  the split changes the rounding.

**Determinism check:** generate twice into two directories (here under `runs/`, which git ignores),
then compare the digests. The manifests carry no timestamps or paths, so they must match byte for
byte too. Comparing a fresh run with `oracle/goldens` checks the published sets the same way.

```sh
docker run ... -v "$PWD/runs:/out" glm53f-oracle:1 generate ... --out /out/run-a
docker run ... -v "$PWD/runs:/out" glm53f-oracle:1 generate ... --out /out/run-b
docker run --rm --network none -v "$PWD/runs:/r:ro" glm53f-oracle:1 compare /r/run-a /r/run-b
docker run --rm --network none -v "$PWD/runs:/r:ro" -v "$PWD/oracle/goldens:/g:ro" \
  glm53f-oracle:1 compare /g /r/run-a
```

Result for the published goldens: two independent runs gave the same digests for all 301 tensors,
byte-identical manifests and the same `source-tensors.tsv`.

## Limits

- **Five layers, not 45.** The head's input is layer 4's output, so `head/head.logits` tests the
  final collapse, norm and LM head, not the model's predictions. The model's logits need the full
  checkpoint (`--last-layer 44`, above). That path is tested on the tiny checkpoint, where the last
  layer is layer 4, but has not yet run on the full checkpoint.
- **CPU numerics.** The native set is the reference on a CPU. On a GPU, the BF16 GEMMs round
  differently, and with FP8 kernels the reference would also quantise activations.
- **Heads subset.** Per-step KDA states are stored for heads 0, 21, 42 and 63; all 64 heads are
  stored after the prompt and after the last step. Per-step core outputs cover all heads.
- **One short prompt.** It selects every complete pool; the `k16` variant covers selection.
