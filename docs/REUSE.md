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
| JSON codec (with exact integers, duplicate-key check) | hughmadden/mimo26f-afd @ `bab9fa2` : `crates/mimo26-api/src/json.rs` | in the crate's PROVENANCE.md | `crates/glm53f-model/src/json.rs` | `Json::Int`, `check_unique_keys`, `as_u64`/`as_i64` | config, catalog tests | 2026-09-28 |
| Safetensors reader (all dtypes, exact coverage, multi-shard index, strided runs) | hughmadden/mimo26f-afd @ `bab9fa2` : `crates/mimo26-repack/src/safetensors.rs` | in PROVENANCE.md | `crates/glm53f-model/src/safetensors.rs` | adapted and extended | checkpoint, headers tests | 2026-09-28 |
| Model catalog, TP4 slicing, memory planner | new (EXL3 TP4 rule cross-checked against glmrt v9 and TensorFold `split.py`, no code copied) | — | `crates/glm53f-model/` | — | 50 tests; SIZING tables reproduced from real headers | 2026-09-28 |
| KDA chain, replay, replay-layers and conv-shift kernels (the per-row arithmetic unchanged; C ABI, batching, in-place states) | ashhart/TensorFold @ `bb4b4a3` : `src/tensorfold/families/glm5_next/cuda/kda.cu`, `kda.cpp`, `kda.py`, `forward.py` | in the crate's PROVENANCE.md | `crates/glm53f-kda/kernels/kda.cu`, `src/kernel.rs` | see PROVENANCE.md; the verbatim source is kept under `kernels/parity/` for a bitwise parity test | `tests/gpu.rs` (bitwise against the source, R = 1..8, every prefix), `tests/cpu_reference.rs` | 2026-09-28 |
| KDA CPU references (recurrent and chunked) | transformers @ `7cd73d9df0` : `models/glm5_next/modeling_glm5_next.py` (semantics only) | `4fe6ed77…` | `crates/glm53f-kda/src/cpu.rs`, `src/chunked.rs` | reimplemented | `tests/cpu_reference.rs` | 2026-09-28 |
