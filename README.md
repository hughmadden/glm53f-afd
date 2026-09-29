# glm53f-afd

An inference engine for [zai-org/GLM-5.3-Flash](https://huggingface.co/zai-org/GLM-5.3-Flash)
that runs the model on **one RTX 5090 and four NVIDIA DGX Spark (GB10) systems**, connected by
RoCE v2 RDMA. It is written in Rust and CUDA, uses no external crates, and serves an
OpenAI-compatible API.

The RTX 5090, the *coordinator*, runs attention (Kimi delta attention and DeepSeek sparse
attention), the KV cache, drafting, sampling and the API. The four Sparks, the *ranks*, run the
routed experts. This split is attention–FFN disaggregation (AFD).

**Status (29 September 2026).** The engine serves the whole model on its target hardware, and the
figures below were measured there. It is young: most figures are single runs on one set of
machines, and some parts are opt-in or not built yet
([Status and limitations](#status-and-limitations)).

## Why this split

GLM-5.3-Flash's context is cheap to store. Only 11 of its 45 layers keep a per-token cache (a
512-dim latent with no RoPE, plus pooled indexer keys): about 6.2 KB per token at FP8. The other
34 layers use linear attention, with a fixed state per request (136 MiB in FP32, 68 MiB in BF16,
the default). So the 5090 can hold a full 1,048,576-token request next to the model's non-expert
weights (about 14 GB).

What is large is the routed experts: 42 MoE layers of 288 experts. At 4 bits they take about 38 GB
per Spark, each rank holding a quarter of every expert's width. Decode is bound by the ranks'
reads of those weights, and the coordinator's work overlaps them.

The serving shell (wire protocol, RDMA transport, scheduler, API) comes from engines already
proven on this hardware class, chiefly
[mimo26f-afd](https://github.com/hughmadden/mimo26f-afd). The GLM-5.3-Flash layers are written
from their reference implementation and tested against it ([docs/DESIGN.md](docs/DESIGN.md)).

## Hardware

| Role | Hardware | Toolkit |
|---|---|---|
| Coordinator | One RTX 5090 (32 GB) in an x86-64 host. Host RAM holds the embedding and the KV snapshot tier | CUDA 12.8 or later (`sm_120`) |
| Ranks | Four DGX Spark (GB10, 128 GB unified memory), about 38 GB of experts each | CUDA 13 (`sm_121`) |
| Fabric | A RoCE v2 port of at least 100 Gb/s on each machine (measured with 200 Gb/s ports; the coordinator's is a bond of two, below). Expert traffic refuses any other network; the API can use any | rdma-core (`libibverbs`) |

**The measured coordinator's network** is a ConnectX-7 with two 200 Gb/s ports bonded, in a PCIe
Gen5 x8 slot. The prefill figures come from boots where both ports carried the return traffic,
each 42–58% of it. A single 200 Gb/s port also works; prefill with one port is untested.
[docs/RUNNING.md](docs/RUNNING.md#bonded-coordinator-nic) says how to check the balance.

## Measured (28–29 September 2026)

**Setup:**
- One RTX 5090 and four DGX Sparks over RoCE v2 at 200 Gb/s.
- EXL3 K4 experts, an FP8 MLA cache, the DFlash2 drafter, and the defaults of 29 September.
- Single runs; between runs ±2–3% is typical.

Every table, method and earlier figure is in [docs/PERFORMANCE.md](docs/PERFORMANCE.md) §0.

**Decode, one stream** (tok/s, greedy; 1,024-token code and counting answers, a 903-token prose
answer):

| Case | Code | Prose | Counting |
|---|---:|---:|---:|
| No drafter (28 Sep) | 52.7 | 52.7 | 52.7 |
| DFlash2, thinking off (the template's Low effort; 29 Sep) | 125.6 | 68.8 | 180.3 |
| DFlash2, no reasoning (an empty think block, no longer served; 29 Sep) | 115.0 | 63.7 | 133.1 |
| DFlash2, thinking on (the model's default; 28 Sep) | 82.7 | 73.3 | 169.1 |

The decode figures predate the chunked KDA prefill and W8A16 as defaults; with them, decode moved
by −2% to +5% (122.6 / 68.2 / 179.5 in the thinking-off row).

**Concurrency** (aggregate tok/s; 512-token streams, each with its own prompt, code, prose and
structured in turn): 159.5 / 288.5 at 4 / 16 streams with 16 slots; 412.5 / 525.8 at 32 / 48 streams
with 48 slots. Code prompts alone reach 383.5 / 576.8 / 738.9 at 16 / 32 / 48 streams.

**Prefill** (one prompt at a time, the defaults: four lanes of 2,048 rows, the chunked KDA prefill
and W8A16 projections): 4,979–4,998 / 5,128–5,180 / 5,132–5,188 tok/s at 4K / 19K / 79K tokens,
21–28% above the 4,099–4,129 / 4,103–4,104 / 4,051–4,053 of the same build without the pair. A
207K-token prompt took 53.7 s without it.

**Prompt cache:** a repeated 207,436-token prompt returns its first token in 0.01 s (53.67 s cold).
After 30 other long prompts pushed its snapshot out of the GPU, it returns it in 0.04 s, restored
from host RAM in 21–23 ms. Snapshots go to RAM only when an incoming request needs their room.

**Copy windows** (on by default for greedy requests with the drafter): rewriting a file decodes
30–31% faster. Fresh code and prose are unchanged, and replies are byte-identical either way.

**Context:** 16 slots admit a 1,048,576-token request (a pool of 1.52M tokens with the defaults). At
48 slots the chunked KDA prefill's workspace (34 MiB a slot) leaves room for about 570K tokens
(computed, not measured); without it 48 slots admitted up to 850,816.

**Start-up:** the coordinator is ready 7 s after launch and a rank in 44–49 s (weights in the page
cache).

### Against the public four-Spark recipes

The two public recipes for GLM-5.3-Flash on four GB10 systems (DGX Spark or ASUS GX10) both run
vLLM at TP4 with NVFP4 experts, on those four machines alone: **neither uses an RTX 5090**. The
differences below therefore come from the added GPU and this engine together, not from the code
alone. Their figures are as they report them; they were not re-run here.

| | [tonyd2wild](https://github.com/tonyd2wild/GLM-5.3-Flash-NVFP4-1M-KV-4x-DGX-Spark) | [mmastrac](https://github.com/mmastrac/glm-5.3-flash-4x-gx10) (branch `perf-2026-09-27`) | This engine |
|---|---|---|---|
| Single stream (tok/s) | ~55 | 167.2 / 118.7 / 64.4 (structured / code / prose); RigMark 107.9 / 61.7 / 157.1 (code / prose / structured) | 186.1 / 142.3 / 78.3 in the same mode; RigMark 123.9 / 65.8 / 174.9 |
| Aggregate (tok/s) | 530 at 48 streams | 253 at 16 streams | 288.5 at 16; 525.8 at 48 (48 slots); a prompt per stream (see below) |
| Prefill (tok/s) | 3.5–4.1K short; 1.9K at 114K | 4,956 / 4,808 at 32K / 128K, cold | 5.0–5.2K at 4K–79K |
| Context | 1M | 512K | 1M (16 slots); about 570K at 48 slots (851K without the default prefill pair) |

- **The aggregate row gives each stream its own prompt**, as the other recipes' figures do (code, prose and structured prompts in turn; theirs are other prompts, so the row is close to like for like, not exact). These mixed aggregates are 5% / 10% / 15% / 13% below the same build's with one prompt topic on every stream, at 4 / 16 / 32 / 48 streams: streams that route alike read fewer expert weights. mmastrac's recipe reports 317 / 451 tok/s for code at 16 / 32 streams with a prompt per stream; code here gives 383.5 / 576.8.
- **RigMark** ([alexellis/rigmark](https://github.com/alexellis/rigmark) `c5a0db0`) with the comparison id and seed mmastrac's gate uses (`ringside-redhat-rowsplit-20260926`, 20260905) sends both engines byte-identical prompts: reasoning effort Low, 4,096 tokens. This engine passed all 15 basic output gates and ran 15% / 7% / 11% faster than the figures that recipe reports (its `experimental/README.md` at `f88710f`).
- **The single-stream row is like for like.** This engine ran mmastrac's own `dev/repro/decode.py`
  prompts and method: 512 tokens, temperature 0, the median of three runs after a warm-up, tok/s
  including the time to the first token.
  - That recipe's "thinking off" renders the model's low reasoning effort. So does this engine's
    since 29 September (`reasoning_effort: "low"` renders the same).
  - Its figures are the ones reported in its commit
    [`3e03894`](https://github.com/mmastrac/glm-5.3-flash-4x-gx10/commit/3e03894ef0) on that
    branch. Against them this engine is 11% / 20% / 22% faster. The branch has moved on since:
    at [`74faf89`](https://github.com/mmastrac/glm-5.3-flash-4x-gx10/tree/74faf894dd24) it
    reports 170.3 / 120.8 / 65.8.
- **Activations.** mmastrac's prefill takes 4-bit activations in its experts. This engine's experts
  keep BF16 activations.

## Quality

**The KL gate** ([docs/KL-GATE.md](docs/KL-GATE.md)) scores the engine's next-token distributions
against the published BF16 teacher logits
([brandonmusic/GLM-5.3-Flash-BF16-Teacher-Logits](https://huggingface.co/datasets/brandonmusic/GLM-5.3-Flash-BF16-Teacher-Logits)).
It follows the method of the published quantization figures: 25 windows of 2,048 tokens, here
with 189 rows scored per window. A run passes when its mean KL plus 1.96 standard errors is below
0.040 nats and top-1 agreement is at least 0.93.

| Engine path | Mean KL (nats) | Top-1 agreement | Gate |
|---|---:|---:|---|
| Decode kernels (passes of 8 rows or fewer), F32 KDA states, 28 Sep | 0.0245 | 95.1% | pass |
| Prefill kernels (4,096-row passes), F32 KDA states, 28 Sep | 0.0282 | 94.7% | pass |

- **The published figure** for the same EXL3 K4 experts is 0.024555 nats, with top-1 agreement
  0.9526. It was measured offline, with no KV-cache quantization, on every position of the same 25
  windows.
- **The decode path matches it** within its standard error, although it adds an FP8 cache and FP8
  wire rows. The scopes differ: 189 rows per window here, every row there.
- **BF16 KDA states** (D8, the default since 29 September) passed a paired comparison against F32
  states: 0.0240 nats on the decode path, 0.0277 on the prefill path.
- **The chunked KDA prefill with W8A16 projections** (the default since 29 September) passed a
  paired comparison on 125 windows against the defaults before it: mean −0.0014 nats, upper
  bound +0.0002 against the 0.002 margin ([docs/KL-GATE.md](docs/KL-GATE.md) §6d).
- **FP8 KDA projections** (D2) failed and stay off.
- **Spot checks:** a number hidden at 37% depth is found in 8.8K and 79K tokens of filler, with and
  without the drafter. `harness/api_contract.py` passes all 12 of its rows on the real model.

## Build and run

Every option, the environment and the tests: [docs/RUNNING.md](docs/RUNNING.md).

**Build.** Rust (edition 2021) and the CUDA toolkit. The kernels are compiled ahead of time for one
architecture, and each binary runs only on that one.

The release was built with these toolchains:

| | Coordinator | Ranks |
|---|---|---|
| Host | x86-64, Ubuntu 24.04, gcc 13.3 | DGX Spark (arm64), Ubuntu 24.04, gcc 13.3 |
| CUDA toolkit | 12.8 (`nvcc` 12.8.93) | 13.0 (`nvcc` 13.0.88) |
| Rust | 1.98.1 (`48a229cea`, 2026-09-01) | 1.98.1 (the same) |

The workspace manifest's `rust-version` is that Rust; no `rust-toolchain` file pins it.

```sh
# The coordinator: x86-64, CUDA 12.8 or later (the build needs no GPU)
GLM53F_CUDA_ARCH=sm_120 cargo build --release -p glm53f-serve --features cuda,rdma
# A rank, on each Spark: arm64, CUDA 13
GLM53F_CUDA_ARCH=sm_121 cargo build --release -p glm53f-rank --features cuda,rdma
```

Those two values are the defaults. The tests run on an RTX 4090, a development GPU and not a
target, and build for it with `GLM53F_CUDA_ARCH=sm_89`
([docs/RUNNING.md](docs/RUNNING.md#development-on-one-gpu)).

**Weights** (not included; the revisions this repository's documents cite):

| For | Hugging Face repository | Revision |
|---|---|---|
| The coordinator: every non-expert tensor, with `config.json`, `tokenizer.json` and `chat_template.jinja` | [zai-org/GLM-5.3-Flash](https://huggingface.co/zai-org/GLM-5.3-Flash), the official FP8 checkpoint. The whole checkpoint works; `python3 scripts/fetch_tensors.py --select nonexpert` fetches only these tensors | `eb9eb208` |
| The ranks: the routed experts in EXL3 K4 | [brandonmusic/GLM-5.3-Flash-tr3-4bpw](https://huggingface.co/brandonmusic/GLM-5.3-Flash-tr3-4bpw) (the same weights, byte for byte, are mirrored at [Mia-AiLab/GLM-5.3-Flash-EXL3-TR3-4bpw](https://huggingface.co/Mia-AiLab/GLM-5.3-Flash-EXL3-TR3-4bpw), revision `25a44fdb`) | `a5fee929` |
| The drafter (optional: `--drafter`) | [incoai/GLM-5.3-Flash-DFlash2](https://huggingface.co/incoai/GLM-5.3-Flash-DFlash2) | `bf582e4e` |

Any Hugging Face client fetches them, for example `hf download <repo> --revision <revision>
--local-dir <dir>`.

The rank images the published figures ran on were cut from the mirror's revision `25a44fdb`; their
SHA-256 are in [docs/rank-images.sha256](docs/rank-images.sha256), and the pinned revision above cuts
the same images ([docs/RUNNING.md](docs/RUNNING.md#weights)).

**Run.** The examples use documentation addresses (192.0.2.0/24) for the fabric.

```sh
# 1. On each Spark: cut its quarter of the experts (about 38 GB; all four ranks from the same
#    checkpoint), then serve it on the fabric. Rank 0 shown; ranks 1-3 listen on .11, .12, .13.
glm53f-rank slice --checkpoint <exl3-checkpoint> --rank 0 --out <rank-dir> --source "<repo>@<revision>"
GLM53F_WIRE_NOCRC=1 glm53f-rank serve --rank 0 --dir <rank-dir> --listen 192.0.2.10:8600 \
  --peers 192.0.2.10:8601,192.0.2.11:8601,192.0.2.12:8601,192.0.2.13:8601

# 2. On the coordinator
GLM53F_RDMA=1 GLM53F_WIRE_NOCRC=1 glm53f-serve --checkpoint <coordinator-dir> \
  --ranks 192.0.2.10:8600,192.0.2.11:8600,192.0.2.12:8600,192.0.2.13:8600 \
  --drafter <dflash2-dir> --listen 0.0.0.0:8100

# 3. A request (the model id is glm-5.3-flash)
curl -N http://<api-host>:8100/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"glm-5.3-flash","messages":[{"role":"user","content":"Hello"}],"stream":true}'
```

The API serves `GET /v1/models`, `GET /health` and `POST /v1/chat/completions`, streamed or not,
with tool calls and reasoning (`reasoning_content`). Thinking is on by default, as the model's chat
template renders it.
- `GET /health` answers 200 `{"status":"ok"}` while the engine can serve, and 503 with the reason
  once it cannot (the expert wire failed or the scheduler stopped: restart the coordinator). It
  reads that state only, so it answers at once under any load. The API listens only once the
  engine is ready.
- `--api-key-file <file>` (or `GLM53F_API_KEY_FILE`) gives the API a key, the file's first line:
  every `/v1/*` request then needs `Authorization: Bearer <key>` and is a 401 without it;
  `GET /health` stays open. Without the option the API serves every request that reaches it
  ([docs/RUNNING.md](docs/RUNNING.md#api-key)).
- The template has no thinking-off mode; it renders a reasoning effort of Low, High or Max, Max by
  default. The server renders only those efforts, never an empty think block.
- A request that turns thinking off (`chat_template_kwargs.enable_thinking: false` or its alias
  `chat_template_kwargs.thinking: false`, or `thinking.type: "disabled"`) gets the template's Low
  effort: a short plan under `reasoning_content`, then the answer.
- `reasoning_effort: "none"` and `"minimal"` are the lowest effort, which is Low: the same request
  as thinking off, and the same prompt. The short plan counts against `max_tokens`, so a very small
  budget can end inside it.
- `reasoning_effort: "low"` / `"high"` / `"max"` choose the effort directly. Any other value goes
  to the template as sent, and the template renders Max for it (`"medium"` is Max, not a middle
  effort).
- A reply with tool calls carries the text the model wrote before them as `content` (its ending
  whitespace dropped; `null` when there is none), streamed or not.
- A tool call the model writes badly (its closing tag missing, markup in its name, arguments but no
  name) is not dropped and never fails the request: its text comes back as `content`, streamed or
  not, and the server logs why. One shape is recovered: a name followed by a stray closing tag,
  when the rest is a tool the request offered.
- A streamed reply is never quiet for more than 15 s: a tool call is held back until it is
  complete, so while the model writes a long one (or a long prompt prefills) the API sends the SSE
  comment `: keepalive` whenever nothing has been written for 15 s, and a proxy or client with an
  idle timeout keeps the connection. The comment carries no data; no event changes.

## Status and limitations

**On by default**, each measured on the target hardware (the numerics changes also passed the KL
gate):
- four prefill lanes of 2,048 rows;
- BF16 KDA states (D8);
- the chunked KDA prefill with W8A16 projections (prefill 21–28% faster, decode unchanged;
  `--kda-chain-prefill --prefill-w8a8` restores the previous arithmetic);
- decode and verify passes of 2–16 rows in two lanes;
- copy windows for greedy requests, with the drafter;
- the rank kernel's GB10 decode schedule;
- an FP8 MLA cache and FP8 wire rows;
- the embedding and a KV snapshot tier in host RAM;
- `GET /health` for health checks (above).

**Opt-in or under test:**
- **`--kda-fp8` (D2):** failed the KL gate in this form.
- **The ranks' prefill reduce-scatter** (`GLM53F_ROW_SHARDED_MIN_ROWS`): slower than the
  four-plane return over the ranks' TCP mesh, so it is off. The RDMA mesh is built but untested.

**Known limitations:**
- **A running stream slows while a long prompt prefills.** On `073b553`, while a 64.6K-token
  prompt prefilled (16.5 s), a stream already generating got 1.0 tok/s, with gaps of up to 3.96 s;
  short requests sent meanwhile got their first tokens in 1.1–3.1 s. Since then a prefill round
  holds one pass of a long prompt (about 1.6 s), and the running requests keep a share of the
  time between rounds (`--decode-share`, 0.2 by default). Computed from that run's figures at
  today's prefill rate: the stream keeps about 21% of its rate, and the prompt's first token comes
  about 24% later. Not yet measured on the target hardware.
- **At 48 slots the default prefill pair costs context:** its workspace grows with the slots, and
  leaves room for about 570K tokens at 48 (computed; 851K without it).

**Measured once, or not yet:**
- Most figures are single runs, and the KL gate scores 189 of each window's 2,047 rows.
- The longest prompt run end to end is 207,436 tokens. A 1,048,576-token request fits the memory
  plan at 16 slots, but no prompt near 1M tokens, and no needle beyond 79K tokens, has run yet.
- The host RAM tier has had two runs on the target hardware. Its page-pressure path (an incoming
  request needing pool pages, not a slot) has not run there yet.

**Not built:**
- image input (the API refuses media parts);
- the model's MTP drafter;
- constrained output (a constrained `tool_choice` is refused);
- any number of ranks other than four.

**Topology.** Exactly four ranks and one coordinator; the ranks serve one coordinator at a time.

## Repository

| Crate | What |
|---|---|
| `glm53f-model` | Configuration, checkpoint catalog, memory planner |
| `glm53f-kda` | Kimi delta attention: fused decode chain and replay, chunked prefill |
| `glm53f-dsa` | DeepSeek sparse attention: indexer, top-k selection, sparse MLA over an FP8 latent cache |
| `glm53f-layers` | mHC hyper-connections, router, dense and shared MLPs, FP8 GEMMs |
| `glm53f-forward` | The model on the coordinator GPU: weights, KV pages, prefill in up to four lanes that overlap the ranks' work, decode and verify (in two lanes at 2–16 rows), commit; remote experts |
| `glm53f-coordinator` | Serving shell: scheduler, slot pool, prefix index, host RAM tier, sampler, copy windows, expert wire client |
| `glm53f-dflash` | The DFlash2 speculative drafter (`glm53f-serve --drafter`) |
| `glm53f-serve` | The coordinator daemon |
| `glm53f-score` | The KL gate's engine side: teacher-forced logits for the rows of a `harness/klgate.py` plan |
| `glm53f-rank` | The expert-rank daemon (EXL3 4-bit experts, TP4) |
| `glm53f-wire`, `glm53f-rdma` | The expert wire protocol and its RDMA transport |
| `glm53f-api`, `glm53f-tokenizer` | OpenAI-compatible API, tokenizer, chat template and GLM tool-call dialect |

| Directory | What |
|---|---|
| `harness/` | The API contract rows and the KL gate (Python standard library only) |
| `oracle/` | Reference goldens from the `transformers` implementation: digests committed, payloads regenerated |
| `scripts/` | A range fetcher for checkpoint subsets, and the pre-commit check that keeps deployment details out of this repository |

Documents:

- [docs/RUNNING.md](docs/RUNNING.md): building for each GPU, running the ranks and the coordinator,
  every option.
- [docs/PERFORMANCE.md](docs/PERFORMANCE.md): the measurements (§0), and the performance model the
  engine was designed against.
- [docs/KL-GATE.md](docs/KL-GATE.md): the quality gate, its method and its results.
- [docs/DESIGN.md](docs/DESIGN.md): the architecture, and where each part comes from.
- [docs/SIZING.md](docs/SIZING.md): context and memory sizing, and the decisions D1–D8.
- [docs/REUSE.md](docs/REUSE.md): the provenance ledger.
- [docs/PLAN.md](docs/PLAN.md): the build plan as written before the code.

## Credits

This engine is built on others' published work. [NOTICE.md](NOTICE.md) and
[docs/REUSE.md](docs/REUSE.md) record every unit taken, with its commit and file digest; each
crate's `PROVENANCE.md` gives the detail.

- [mimo26f-afd](https://github.com/hughmadden/mimo26f-afd) (MIT): the serving shell, wire codec,
  RDMA transport, API and tokenizer, and the copy windows (v1.3.0).
- [DS41RT](https://github.com/tpurtell/ds41rt) by T.J. Purtell
  ([@wrldsuksgo2mars](https://x.com/wrldsuksgo2mars)) (MIT): the mHC and router kernels and the
  sparse MLA kernel's structure; through mimo26f-afd, the wire format, the RDMA design and the
  sampling contract.
- [TensorFold](https://github.com/ashhart/TensorFold) (MIT): the KDA chain and replay kernels, the
  EXL3 tile decoder, and the idea of copy windows.
- [glmrt](https://github.com/tpurtell/glmrt-5.3-1rtx-4spark) by T.J. Purtell (MIT): the designs of
  the TP4 EXL3 split and the prefill reduce-scatter.
- [flash-linear-attention](https://github.com/fla-org/flash-linear-attention) (MIT): the structure
  of the chunked KDA prefill.
- [z-lab/dflash](https://github.com/z-lab/dflash) (MIT) and
  [SGLang](https://github.com/sgl-project/sglang) (Apache-2.0): the DFlash2 drafter's semantics.
- [b12x](https://github.com/local-inference-lab/b12x), through
  [sparkinfer-glmrt](https://github.com/tpurtell/sparkinfer-glmrt) (Apache-2.0): the MLA latent
  record layout.
- [ExLlamaV3](https://github.com/turboderp-org/exllamav3) (MIT): the EXL3 format.
- [transformers](https://github.com/huggingface/transformers) (Apache-2.0): the `glm5_next`
  reference that every layer is tested against.
- The weights and measurements the engine is checked against: zai-org (GLM-5.3-Flash),
  brandonmusic (the EXL3 K4 experts, the BF16 teacher logits and the KL method), incoai (the
  DFlash2 drafter) and malaiwah (the quant-fidelity registry).
- The public four-Spark recipes compared above: mmastrac and tonyd2wild.

## Licence

MIT; see [LICENSE](LICENSE). Code taken from other projects keeps its own licence, recorded in
[NOTICE.md](NOTICE.md). The model weights and the drafter are not included; their authors publish
them under their own terms.
