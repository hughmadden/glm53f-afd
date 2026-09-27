# Provenance ledger

Every unit copied or transcribed from another project gets a row here before it
lands: the source repository and commit, the source path, the file digest, where
it lives in this repository, what changed, and the tests that pin it. Units that
are reimplemented from a description (no code copied) are recorded too, marked
"reimplemented". Crate-level notes live in each crate's `PROVENANCE.md` and are
consolidated here.

| Unit | Source (repo @ commit : path) | sha256 (source file) | Here | Delta | Pinned by | Date |
|---|---|---|---|---|---|---|
| Wire codec (`DS41RTE3` v3 frames, CRC32C, layouts; 20 files) | hughmadden/mimo26f-afd @ `bab9fa2` (v1.2.0) : `crates/mimo26-wire/` | per file in the crate's PROVENANCE.md | `crates/glm53f-wire/` | renames at the boundary; docs recomputed for 42 MoE layers; new `tests/glm_experts.rs` pins expert ids 256–287 | wire_layout, l4_integrity, async_api, in_place_return, r8_rank_order, request_view, glm_experts | 2026-09-28 |
| RDMA RC transport (rdma-core verbs shim; 10 files incl. vendored headers) | hughmadden/mimo26f-afd @ `bab9fa2` : `crates/mimo26-rdma/` | per file in PROVENANCE.md | `crates/glm53f-rdma/` | renames; test GID moved to a documentation address; symbol prefix and handshake unchanged | unit tests; `--features rdma` builds and links | 2026-09-28 |
| OpenAI-compatible API (chat, SSE, tools, queue, JSON; 13 files) | hughmadden/mimo26f-afd @ `bab9fa2` : `crates/mimo26-api/` | per file in PROVENANCE.md | `crates/glm53f-api/` | new `Dialect` seam (MiMo parser kept as the reference dialect); image decoding moved behind `Engine::decode_image`; model id `glm-5.3-flash` | api unit (17), acceptance (22) | 2026-09-28 |
