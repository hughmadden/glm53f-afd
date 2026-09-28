# Provenance: glm53f-score

Rows in the format of [docs/REUSE.md](../../docs/REUSE.md). All dates are 28 September 2026.
No code is copied from outside this repository. The scorer implements the engine side that
[docs/KL-GATE.md](../../docs/KL-GATE.md) section 4 specified, reads and writes the formats of
`harness/klgate.py`, and loads the model as `glm53f-serve` does. Within this repository it calls
`glm53f-forward` (the forward, `GlmForward::score`), `glm53f-model` (the JSON codec, the checkpoint
reader), `glm53f-dsa` (SHA-256) and `glm53f-serve` (`parse_dev_layers`, `RANKS`,
`MAX_LANE_ROWS`).

Sources (all this repository @ `9a94f89`, MIT):

- **K**: `docs/KL-GATE.md` section 4 and `harness/klgate.py` (the plan's schema, the output's
  tensors and metadata, `tokens_sha256` as little-endian u32).
- **S**: `crates/glm53f-serve/src/lib.rs` and `src/main.rs` (the options, the loading sequence).
- **F**: `crates/glm53f-forward/tests/drafting/mod.rs` (`SomeExperts`: local experts for some MoE
  layers, zeros for the rest).

| Unit | Source (repo @ commit : path) | sha256 (source file) | Here | Delta | Pinned by | Date |
|---|---|---|---|---|---|---|
| The options: flags with environment fallbacks, the four ranks as `host:port`, the checks | S : `src/lib.rs` (`Options::parse`, `parse_ranks`, `number`) | — | `src/lib.rs` | Pattern, rewritten for the scorer's flags (`--plan`, `--out`, `--pass-rows`, `--windows`, the numerics options, `--dev-load-layers`, `--experts zero`); `parse_ranks` and `number` restated (private in S) | unit tests in `src/lib.rs` | 2026-09-28 |
| Loading: the weights (the embedding in host RAM), the experts (ranks or local), the forward's buffers before the KV pool, the memory log | S : `src/main.rs` (`daemon::run`, steps 1 to 5) | — | `src/engine.rs` (`load`) | Pattern, rewritten: one slot sized for the longest window, no drafter, no engine or API; `DeviceModel::load_repeating` for `--dev-load-layers`; the local experts' coverage checked per MoE layer | `tests/plumbing.rs` | 2026-09-28 |
| Local experts for some MoE layers, zeros for the rest | F : `SomeExperts` | — | `src/engine.rs` (`PartialExperts`) | Pattern: the layers come from the experts directory instead of a constant; development runs only | `tests/plumbing.rs` | 2026-09-28 |
| The plan (schema `glm53f-kl-plan.v1`) and its checks; the window files (`positions` I32, `logits` F32, metadata `window_id`, `tokens_sha256`, `plan_sha256`, `engine`); `run.json` | K | — | `src/plan.rs`, `src/out.rs`, `src/main.rs` | **Written here** to K. The files are streamed row by row and renamed into place when complete. | unit tests in `src/plan.rs` and `src/out.rs`; `tests/format.rs` (through `klgate.py`) and `tests/plumbing.rs` | 2026-09-28 |

## Written here

| File | What | Date |
|---|---|---|
| build.rs | The build's commit (`-dirty` when the engine's sources differ from it) and the kernels' target, for the engine line | 2026-09-28 |
| src/main.rs | The binary: the plan read and checked, the engine loaded, each window scored and written, `run.json` after each | 2026-09-28 |
| tests/common/mod.rs | A teacher panel in the dataset's layout (manifest, token arrays, `teacher-rows/`), `klgate.py`, the scorer's files read back | 2026-09-28 |
| tests/format.rs | The plan and the files through `klgate.py plan` and `score` on the CPU | 2026-09-28 |
| tests/plumbing.rs | The binary in development mode on one GPU through `klgate.py`, and against the real teacher rows | 2026-09-28 |
| Cargo.toml | The manifest | 2026-09-28 |

## Test data

None is carried. `tests/format.rs` writes its own synthetic panel. `tests/plumbing.rs` reads the
coordinator's weights and the experts of layers 3 and 4 (`GLM53F_CHECKPOINT_DIR`,
`GLM53F_EXPERTS_DIR`) and, with `GLM53F_KL_TEACHER`, the teacher subset `harness/klgate_fetch.py`
writes from `brandonmusic/GLM-5.3-Flash-BF16-Teacher-Logits` (see `harness/PROVENANCE-klgate.md`).
