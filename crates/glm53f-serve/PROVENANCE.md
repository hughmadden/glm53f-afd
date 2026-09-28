# Provenance: glm53f-serve

Rows in the format of [docs/REUSE.md](../../docs/REUSE.md). Dates are 28 and 29 September 2026.
No code is copied. The daemon follows the shape of mimo26f-afd v1.2.0's coordinator binary and
wires this repository's crates together.

## Units

| Unit | Source (repo @ commit : path) | sha256 (source file) | Here | Delta | Pinned by | Date |
|---|---|---|---|---|---|---|
| The coordinator binary's sequence: load the coordinator's weights, build the device forward, connect the four ranks (addresses from configuration, no default), load the tokenizer, start the engine, serve the API | mimo26f-afd @ bab9fa2 : crates/mimo26-coordinator/src/main.rs (lines 10-74, `serving_main`) | `2a6ab00576731ba5ac320815e5da80b420bdbc31379961ccc5748154e3ec9ed2` | src/main.rs | Structure, rewritten over this repository's crates: `GlmForward` and `ServedForward`, `RemoteExperts` or `LocalFp8Experts`, `CoordinatorEngine::start` with the scheduler, `GlmPrompts`, `GlmDialect`. Flags with environment fallbacks (`MIMO26_WEIGHTS_DIR`, `MIMO26_SPARK_ADDRS` and `MIMO26_API_ADDR` became `GLM53F_CHECKPOINT_DIR`, `GLM53F_SPARK_ADDRS` and `GLM53F_API_ADDR`); the API defaults to loopback (was `0.0.0.0:8100`); the KV pool sized from the free memory; the development mode is new; the source's host-forward path, drafter loading, page-locking of the wire buffers and CPU smoke are not ported. | tests/dev_mode.rs (one streamed chat completion through four rank daemons), the unit tests in src/lib.rs | 2026-09-28 |

## Written here

| File | What | Date |
|---|---|---|
| src/lib.rs | The options (flags, environment fallbacks, checks, `--drafter`), the KV pool's page count, the development-mode banner, and their tests | 2026-09-28 |
| src/main.rs (the drafter) | The DFlash2 drafter loaded next to the weights before the KV is sized (its ring in each slot's fixed state), attached to the forward, verify passes sized for every slot's window, its memory logged; off in a development mode of fewer than 44 layers | 2026-09-28 |
| src/main.rs (start-up memory) | The allocation order (the experts' and the forward's buffers before the KV pool, the pool from what is left) and the memory plan logged at start-up; the prefill lanes and the lane trace (`GLM53F_PROFILE`) | 2026-09-28 |
| tests/dev_mode.rs | Four rank daemons and the daemon in development mode on one GPU; one streamed chat completion over HTTP | 2026-09-28 |
| src/lib.rs, src/main.rs (copy windows) | `--copy-windows on\|off` (`GLM53F_COPY_WINDOWS`), on by default, handed to the scheduler (`SchedulerConfig::copy_windows`); its documentation and test | 2026-09-29 |
| src/lib.rs, src/main.rs (snapshots) | `GLM53F_PREFIX_CACHE_ENTRIES` documented as an optional cap (none by default) and the rule snapshots follow without it; the start-up memory line gives a mark's pages and the cap, if any, instead of the banks' size at 24 | 2026-09-29 |
| Cargo.toml | The manifest | 2026-09-28 |

## Test data

None is carried. `tests/dev_mode.rs` reads the coordinator's weights named by
`GLM53F_CHECKPOINT_DIR` and runs a `glm53f-rank` binary (`GLM53F_RANK_BIN`) on rank directories
cut from the EXL3 checkpoint (`GLM53F_RANK_DIRS`).
