# glm53f-afd

An inference engine for [zai-org/GLM-5.3-Flash](https://huggingface.co/zai-org/GLM-5.3-Flash)
that runs the model across **one RTX 5090 and four NVIDIA DGX Spark (GB10)
systems**, connected by RoCE v2 RDMA.

The GPU runs attention (Kimi delta attention and DeepSeek sparse attention),
the KV cache, drafting, sampling and the API. The four Sparks run the routed
experts. This split is attention–FFN disaggregation (AFD).

**Status: design phase.** There is no engine code yet. The documents below are
the design to agree on first:

1. [docs/SIZING.md](docs/SIZING.md): context and memory sizing, and the
   decisions to settle first.
2. [docs/DESIGN.md](docs/DESIGN.md): the architecture, and where each part
   comes from.
3. [docs/PERFORMANCE.md](docs/PERFORMANCE.md): expected performance, derived
   from measured numbers of related engines.
4. [docs/PLAN.md](docs/PLAN.md): the build order and its gates.

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

- **Third-party code:** each unit copied from another project will be recorded
  in `docs/REUSE.md` and `NOTICE.md` when it lands.
- **Weights:** model weights and the draft model are not included. They are
  published by their authors under their own terms.
