# glm53f-afd

An inference engine for [zai-org/GLM-5.3-Flash](https://huggingface.co/zai-org/GLM-5.3-Flash)
that runs the model across **one RTX 5090 and four NVIDIA DGX Spark (GB10)
systems**, connected by RoCE v2 RDMA.

The GPU runs attention (Kimi delta attention and DeepSeek sparse attention),
the KV cache, drafting, sampling and the API. The four Sparks run the routed
experts. This split is attention–FFN disaggregation (AFD).

**Status: in development (September 2026).** Every part of the model path exists
and is tested against the reference implementation on a development GPU, and the
expert ranks' kernel has run on GB10. The engine has not yet served the whole
model on its target hardware (one RTX 5090 and four Sparks). Do not use it for
anything that matters yet.

| Crate | What |
|---|---|
| `glm53f-model` | Configuration, checkpoint catalog, memory planner |
| `glm53f-kda` | Kimi delta attention: fused decode chain and replay, chunked prefill |
| `glm53f-dsa` | DeepSeek sparse attention: indexer, top-k selection, sparse MLA over an FP8 latent cache |
| `glm53f-layers` | mHC hyper-connections, router, dense and shared MLPs, FP8 GEMMs |
| `glm53f-forward` | The model on the coordinator GPU: weights, KV pages, prefill (in two lanes that overlap the ranks' work), decode, verify and commit; remote experts |
| `glm53f-coordinator` | Serving shell: scheduler, slot pool, prefix index, host RAM tier, sampler, expert wire client |
| `glm53f-dflash` | The DFlash2 speculative drafter (wired into the forward: `glm53f-serve --drafter`) |
| `glm53f-serve` | The coordinator daemon |
| `glm53f-rank` | The expert-rank daemon (EXL3 4-bit experts, TP4) |
| `glm53f-wire`, `glm53f-rdma` | The expert wire protocol and its RDMA transport |
| `glm53f-api`, `glm53f-tokenizer` | OpenAI-compatible API, tokenizer, chat template and GLM tool-call dialect |
| `oracle/` | Reference goldens from the `transformers` implementation |

Documents:

1. [docs/SIZING.md](docs/SIZING.md): context and memory sizing, and the
   decisions to settle first.
2. [docs/DESIGN.md](docs/DESIGN.md): the architecture, and where each part
   comes from.
3. [docs/PERFORMANCE.md](docs/PERFORMANCE.md): expected performance, derived
   from measured numbers of related engines, with this engine's first
   measurements.
4. [docs/PLAN.md](docs/PLAN.md): the build order and its gates.
5. [docs/RUNNING.md](docs/RUNNING.md): building for each GPU, and running the
   ranks and the coordinator.

## In one paragraph

GLM-5.3-Flash's context is cheap to store. Only 11 of its 45 layers keep a
per-token cache: a 512-dim latent with no RoPE, plus pooled indexer keys. That
costs about 6.2 KB per token at FP8, so the 5090 can hold a full
1,048,576-token request next to the model's non-expert weights. The other 34
layers use linear attention, with a fixed 136 MiB state per request. The routed
experts, 42 layers of 288 experts, fit on four Sparks at 4 bits (about 38 GB per
Spark) or even at FP8 (about 76 GB). The design borrows its serving shell from
proven engines on this hardware class. The GLM-5.3-Flash layers come from their
reference implementations.

## License

MIT; see [LICENSE](LICENSE).

- **Third-party code:** each unit copied or adapted from another project is
  recorded in [docs/REUSE.md](docs/REUSE.md) and [NOTICE.md](NOTICE.md), with
  per-crate detail in each crate's `PROVENANCE.md`.
- **Weights:** model weights and the draft model are not included. They are
  published by their authors under their own terms.
