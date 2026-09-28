# glm53f-coordinator provenance

Ported from [hughmadden/mimo26f-afd](https://github.com/hughmadden/mimo26f-afd) v1.2.0,
commit `bab9fa2f2fc1e22ae67b56fbc1c209278f6a9d79` (MIT): the model-agnostic parts of
`crates/mimo26-coordinator`, plus two device functions from `crates/mimo26-attn` and the E4M3
codec of `crates/mimo26-load`. The sha256 column is the digest of the whole source file at that
commit (several units take part of a file).

**Renames applied throughout** (not repeated in the Delta column): `mimo26_wire` →
`glm53f_wire`, `mimo26_rdma` → `glm53f_rdma`, the environment prefix `MIMO26_` → `GLM53F_`, and
the kernels' symbol prefix `m26c_` → `glm53f_coord_`.

**Kept on purpose:** comment labels such as perf reset K3, Q1, S1, S2, V2, V3, W4, P6, P9, R2, R4, R8,
T10 and D7 name mimo26f-afd's public design records (`ARCHITECTURE.md`,
`docs/design/perf-reset-vs-ds41rt.md`) and stay as provenance. The sampling contract, the
queue's admission rule and the host cache's `on-evict` design are DS41RT v15's
([tpurtell/ds41rt](https://github.com/tpurtell/ds41rt), MIT), reimplemented in mimo26f-afd; no
DS41RT code is copied here.

## Units

| Unit | Source (repo @ commit : path) | sha256 (source file) | Here | Delta | Pinned by | Date |
|---|---|---|---|---|---|---|
| Scheduler loop, request records, slot pool, snapshot points, admission (`Job`, `Active`, `Prefilling`, `Point`, `Retained`, `Pool`, `admit_rows`, `admit_check`, `extends`, `start`, `scheduler`) | mimo26f-afd @ bab9fa2 : crates/mimo26-coordinator/src/api.rs (lines 109-121, 207-293, 324-724, 743-1102) | `a0e03f550971e72b38bf23093d75bc23959e92a25c4cf70cfac81ea5d2f84232` | src/scheduler.rs, src/pool.rs | Ported behind `KvSlot` / `ModelForward`; MiMo's structure kept. Every behavioural change is listed below | tests/scheduler.rs | 2026-09-28 |
| Bounded queue and `Engine::admit` | same file (lines 97-107, 1218-1249) | as above | src/queue.rs | A type of its own (`Queue::admit`); depth default = slot count, 16 | unit tests in the file; tests/engine.rs | 2026-09-28 |
| `Engine` implementation (`generate`, keep-alive, cancel, `LAST_ENCODE`, `emit_delta`, `CancelOnDrop`, `image_token_id`, `image_spans`, `encode_marked`) | same file (lines 33-40, 130-205, 1197-1394, tests 1460-1519) | as above | src/engine.rs | Text side behind `PromptCodec`; `render_prompt` / `tokenize_prompt` with `PromptOptions`; template errors returned by `generate`; `encode_marked` became `expand_image_marker` (the GLM template renders the image delimiters itself); the host reference backend dropped | unit tests in the file; tests/engine.rs | 2026-09-28 |
| Verify-length policy (`SpecPolicy`, `chain_length`, `verify_lengths`) | mimo26f-afd @ bab9fa2 : crates/mimo26-coordinator/src/dforward.rs (lines 937-994, test 2280-2295) | `c048732e72d1b78886a73fc4bb50b820a2fb0077c208909d1fc269d3a62a25fb` | src/spec.rs | Draft probabilities as slices of any length (were `[f32; 7]`); every length capped at the drafts the model returned; `from_env` gathers the three variables | unit tests in the file | 2026-09-28 |
| Speculative step's acceptance (`spec_step`, the accept loop of `verify_batch`'s caller) and the selection (`select`, `select_host_row`, `select_rows_host`) | same file (lines 134-160, 1611-1675, 1798-1833) | as above | src/scheduler.rs (`spec_step`), src/gpu.rs (`Sampler`, `select_rows_host`) | Acceptance moved into the scheduler over `ModelForward::draft` / `verify` / `commit`; selection is a `Sampler` any model's forward calls, with masks | tests/scheduler.rs, tests/gpu.rs | 2026-09-28 |
| Host RAM tier | mimo26f-afd @ bab9fa2 : crates/mimo26-coordinator/src/hostcache.rs | `ec6931686c930754ede3d9d128e5bd720f25cec0fb701f4b7a85c002eb087a70` | src/hostcache.rs | Over `KvSlot` page and mark export; page size, state size and budget split are parameters; radix lookup; see below. Tests `eviction_is_least_recently_used_first` and `page_chain_identifies_prefixes` kept (the latter at 256- and 64-token pages) | unit tests in the file; tests/host_tier.rs; tests/scheduler.rs | 2026-09-28 |
| Sampling contract and CPU reference | mimo26f-afd @ bab9fa2 : crates/mimo26-coordinator/src/sampling.rs | `e9a1393a691766c98762af07a1986acaf612a682a59a6537833e36e9fd7e8679` | src/sampling.rs | Module doc (bound, masks); `After::from_logits` takes the id bound; new `greedy` (the kernel's rule, bounded), `Mask`, `apply_mask`, `select_pick`. `Sampling`, `DeviceRow`, `select` and their tests verbatim | unit tests in the file (7 kept, 4 new); tests/gpu.rs | 2026-09-28 |
| Streaming holdback (`flush_pending`) | mimo26f-afd @ bab9fa2 : crates/mimo26-coordinator/src/streaming.rs | `8fc947479f0113a589145f958cdbfa4f313c5bb5476135f7c9cd8853c9346e54` | src/streaming.rs | Takes a decode function (ids to bytes) in place of MiMo's tokenizer; the tests' toy tokenizers became toy decoders; test bodies verbatim | unit tests in the file | 2026-09-28 |
| Expert wire client and FP8 K32 quantizer | mimo26f-afd @ bab9fa2 : crates/mimo26-coordinator/src/wire.rs | `4a9a1fb866b914cec532c45d51d2e7e1ab558a1b978c28afd63c24059e3f2561` | src/wire.rs | `WireConfig` (experts 288, routed scale, `ReturnPath`); route validation; the scale on the host sum; `moe_recv_during`; `returned`; host-only batched quantizer; module doc; one pointer to a private document removed; tests reworked around a four-rank mock (below) | unit tests in the file | 2026-09-28 |
| E4M3 codec | mimo26f-afd @ bab9fa2 : crates/mimo26-load/src/e4m3.rs (lines 14-81) | `767fc973a21aa65c09616c3acfa74eaa32f729dab33c7eeac9f7c5948a4bcfe3` | src/fp8.rs | `E4M3_MAX`, `decode_e4m3`, `decode_table`, `encode_e4m3` verbatim; module doc; two comments name the source repository; tests written here | unit tests in the file | 2026-09-28 |
| Sampling kernel | mimo26f-afd @ bab9fa2 : crates/mimo26-coordinator/kernels/sample.cu | `18267fac23b314a6ec4e09f0bcb72b91db143074a1c5615a7f9186d91e068ed5` | kernels/sample.cu | Entry point renamed; three comment lines. Kernel body verbatim | tests/gpu.rs, examples/sample_check.rs | 2026-09-28 |
| Device argmax | mimo26f-afd @ bab9fa2 : crates/mimo26-coordinator/kernels/dflash.cu (lines 163-229, 256-267) | `ecb17cd49a5f1b9e636ab8d051155bae0bc87eee4d3757f0ed3833cd9b8f4caa` | kernels/select.cu | `argmax_kernel` verbatim; its two entry points renamed; called with the tokenizer's id bound | tests/gpu.rs | 2026-09-28 |
| Wire kernels: UE8M0 scales, rank sum, frame fill, `grid_for` | mimo26f-afd @ bab9fa2 : crates/mimo26-coordinator/kernels/glue.cu (lines 125-143, 200-213, 222-229, 263-274, 311-348) | `79a306467d6937235e00299d8422fc2328b3b4a73824a47af798f8bc9dddc4e7` | kernels/wire.cu | Kernels verbatim except the rank sum, which multiplies by a scale after the four adds (`__fmul_rn`; 1.0 gives the source's bits); all kernels in the file's anonymous namespace (the source defined `frame_fill_kernel` inside its `extern "C"` block); entry points renamed, the rank sum's with the scale parameter | tests/gpu.rs | 2026-09-28 |
| Device E4M3 encoder | mimo26f-afd @ bab9fa2 : crates/mimo26-attn/kernels/include/mimo26_attn_device.cuh (lines 33-55) | `4a4028f07604020f6adf69fb91f3af1ec9639eeab740fb4ac9af8eb9e2a9f5d2` | kernels/wire.cu | Body verbatim; its comment rewritten without a pointer to a private document | tests/gpu.rs | 2026-09-28 |
| Device hidden quantizer | mimo26f-afd @ bab9fa2 : crates/mimo26-attn/kernels/kv_cache_fp8.cu (lines 160-169, 171-178) | `861f85df38b3641374a2a4ea1478c9c67502f9b4975e539da37dc0c569a5d5c3` | kernels/wire.cu | Kernel verbatim (namespace qualifier dropped); entry point renamed, takes a `cudaStream_t` | tests/gpu.rs | 2026-09-28 |
| Sampler check | mimo26f-afd @ bab9fa2 : crates/mimo26-coordinator/examples/sample_check.rs | `0fd93f37f84a1524b090abedb2d5d09ee942b6548b02ad94abd12f9df6539e99` | examples/sample_check.rs | GLM-5.3-Flash's widths (154,880 rows, ids below 154,856); `Pick`s through `gpu::select_rows_host` against `select_pick`; the timing covers the whole selection | `cargo run --release --features cuda --example sample_check` | 2026-09-28 |
| Build script | mimo26f-afd @ bab9fa2 : crates/mimo26-coordinator/build.rs | `a40762247399ff0f0c32e242210ba51322c07007b882cf12d483d4cc722fe15b` | build.rs | Rewritten in the shape of this repository's other kernel crates (`GLM53F_NVCC`, `GLM53F_CUDA_ARCH`, `GLM53F_CUDA_LIB`, cudart only); the source's nvcc flags kept | `--features cuda` builds | 2026-09-28 |

## Written here

| File | What | Date |
|---|---|---|
| src/model.rs | `KvSlot` and `ModelForward`, the cut between the shell and a model, with `Pick`, `Segment`, `DecodeRow`, `DraftRow`, `Draft`, `Window`, `Limits`, `ImageSpan` | 2026-09-28 |
| src/radix.rs | `RadixIndex`: a compressed trie over token ids with exact hits and a granular shared-prefix count | 2026-09-28 |
| src/glm_prompt.rs | `GlmPrompts`: GLM-5.3-Flash's tokenizer and chat template (`glm53f-tokenizer`) behind `PromptCodec`, the official template enforced | 2026-09-28 |
| src/gpu.rs | FFI to the kernels and the few CUDA runtime calls, `DeviceBuffer`, `Sampler`, page-locking | 2026-09-28 |
| src/lib.rs, Cargo.toml | Crate root and manifest | 2026-09-28 |
| kernels/select.cu (`mask_rows_kernel`, `glm53f_coord_mask_rows`) | The per-row grammar mask | 2026-09-28 |
| kernels/glm53f_coord.h | The kernels' C ABI | 2026-09-28 |
| tests/common/mod.rs | A toy deterministic model and slot behind the traits (its positional state is a hash of the whole prefix, checked on every call), simulated memory and time, a call log | 2026-09-28 |
| tests/scheduler.rs, tests/host_tier.rs, tests/engine.rs, tests/glm_prompt.rs, tests/gpu.rs | See Tests | 2026-09-28 |
| PROVENANCE.md | This ledger | 2026-09-28 |

## Behavioural differences from the source

### Scheduler and slot pool

1. **The model is behind two traits.** `DeviceForward` / `DeviceKv` became `ModelForward` /
   `KvSlot`. The loop is MiMo's: admission (a job deferred behind an identical prefill, one that
   did not fit waiting first in first out, refusal on an idle device), a prefill round (short
   prompts batched in one pass, then segments round robin until about 2 s is spent), bank
   enforcement, cancelled requests retired, running requests grown, one speculative or decode
   step.
2. **Prefix matching is a radix index.** The source scanned every retained slot's token history
   for the longest point that prefixes the prompt (`device_hit`) and scanned every RAM snapshot
   (`HostCache::lookup`). Both are `RadixIndex` lookups now, with the same result: the longest
   point whose tokens are a prefix of the prompt, a point at the prompt's full length giving way
   to a shorter one when it cannot serve the request. Among points of equal length a retained
   slot's last point comes first (used in place), as in the source; among identical keys the
   earliest indexed wins (the source took the first in its vector order).
3. **Points of running requests can be forked.** The source only resumed from retained slots, so
   a prompt deferred behind an identical prefill (its comment: "then forks its snapshot") found
   no point once the first request was decoding and prefilled again. Points of running and
   prefilling requests are indexed too and are forked into a free slot (never used in place).
4. **Granularity and the branch gap.** The index reports how much of a prompt some stored key
   shares, in whole granules (4 tokens, GLM-5.3-Flash's indexer pool, by default); the pool counts
   the part past the resume point (`PoolStats::branch_gap_tokens`): prefill that periodic
   checkpoints would save. New; resumes stay exact.
5. **No truncation of committed rows.** The source's speculative step truncated the rejected rows
   inside the forward, and `retire` truncated a slot back to the request's history when it
   stopped inside an accepted run. A KDA state cannot be truncated, so here the model's `commit`
   keeps exactly the rows of the tokens the scheduler delivered (the accepted drafts up to the
   stop, the window's first row included), and the slot is at the history when the request
   retires. A slot found elsewhere (a bug) is freed instead of retained.
6. **Deliver, then commit.** The source committed inside `spec_step` before sending tokens. Here
   the accepted tokens go out first (the commit needs to know how many were delivered); a commit
   that fails then reaches clients after their tokens.
7. **Every pass returns token ids.** Each row carries a `Pick`; the model applies it with the GPU
   sampler, so the scheduler no longer handles logits. The source downloaded a long prompt's last
   logit row every segment and took its argmax on the host, and took decode rows' argmax on the
   host when no row was sampled. A prompt segment now asks for the argmax (and, past 512 tokens,
   the logits, kept by a snapshot) through its pick.
8. **Greedy is bounded.** The argmax runs over ids below the tokenizer's bound, so the LM head's
   padding rows (154,856 to 154,879 for GLM-5.3-Flash) can never be picked. The source's argmax ran
   over the whole LM head; only draws were bounded.
9. **Memory accounting through the traits.** `KvSlot::need_bytes` and `ModelForward::free_bytes`
   replace the source's GA growth plus transient plus SWA working set against `cudaMemGetInfo`
   less 512 MiB; the margin is the model's. The drafter's state belongs to the slot (the source's
   `attach_draft` is gone). The maximum context is the largest request `need_bytes` admits
   (a search) less the output allowance, capped at the model's context (the source used a closed
   formula without the growth transient and the 4,096-row rounding).
10. **Images are the model's.** The scheduler makes room (`ModelForward::image_bytes`) and passes
    each segment the image spans it reaches; the source encoded the images itself and kept the
    encoded rows per prompt.
11. **Defaults and knobs.** 16 slots (the source: 8; `slot_count()` reads `GLM53F_MAX_SLOTS`),
    created by the caller at their base capacity (the source built 4,096-row slots itself); queue depth defaults to the slot count; every constant of the source
    (`MIN_RETAIN` 512, bank 24, 2,000 ms segments, `SEG_QUANTUM` 8,192, `SEG_MAX` 65,536, the
    1,024 / 8,192 / 64 allowance) is a `SchedulerConfig` field with the source's value; the clock
    is injectable; `Scheduler::step` runs one pass (the source's loop is `run`).
12. **Growth headroom** before a step is `2 * drafts + 2` rows with the model's draft count (the
    source always used 7 drafts).
13. **Two lanes** are the model's: a pass gets the whole batch and may split it into lanes, as
    the source's forward did. DS41RT's independent lanes, each with its own rounds, are not
    implemented.
14. Not ported: the host reference backend (one request at a time on the CPU), vision encoding.
    New: counters (`SchedStats`, `PoolStats`).

### Host RAM tier

1. **Snapshot anatomy is the slot's.** Pages of `KvSlot::page_tokens` tokens through
   `export_page` / `import_page` and a positional-state image through `export_state` /
   `import_state` replace the source's GA rows, SWA rows and draft rings (the snapshot's
   `swa_rows` and `has_draft` fields are gone). Page size is a parameter (the source: 256 rows).
2. **Budget split.** States get the share that fills page and state slots at the same rate for
   64K-token conversations with two snapshots (42% for GLM-5.3-Flash's 141 MiB states; 8.5% for
   MiMo's sizes). The source gave states a fixed 10%. Clamps (4 to 256 states, at least 16 pages)
   unchanged.
3. **Lookup** is a radix walk (the source compared page chains, then tail tokens, snapshot by
   snapshot); same result. Identical captures refresh the held snapshot, as before.
4. **Failed captures give everything back.** An export error in the source returned early and
   kept the page slot and the page references taken so far.
5. **Eviction order** is the source's (least recently used; at equal use a prompt snapshot
   before a turn snapshot); ties between snapshots of equal use and kind go to the oldest id
   (the source: vector order, which `swap_remove` reshuffled).
6. Page-locking only with the `cuda` feature (`HostTierConfig::pin`); the minimum snapshot is a
   config field (512).

### Sampler

1. **Masks** (new): `glm53f_coord_mask_rows` sets the logits of tokens a row's bitset does not
   allow to minus infinity before the argmax and the draw; `apply_mask` and `select_pick` are the
   CPU references.
2. The argmax is called with the id bound (above). `sampling::greedy` models the kernel's rule
   exactly: the source's host `greedy` let a NaN at index 0 win.
3. `Sampler::select` uploads the mask and sampled-row tables per call; the kernels are the
   source's.

### Engine and queue

1. The text side is a `PromptCodec`; GLM-5.3-Flash's (`GlmPrompts`) renders with the official
   chat template and refuses any other. `render_prompt` / `tokenize_prompt` carry the request's
   thinking switch, reasoning effort and `clear_thinking`.
2. A chat-template error (the API's `render_prompt` cannot fail) comes back from `generate`.
3. End-of-sequence ids and the `max_tokens` cap (65,536) are configuration.

### Wire client

1. The expert count is a parameter (288); host paths refuse an expert id at or past it.
2. A routed scale can multiply the ranks' sum on the coordinator (host sum here, a parameter of
   the device rank sum). The default is 1.0, which leaves the source's bits: GLM-5.3-Flash's
   2.5 travels inside the gate weights, as the reference and `glm53f-layers`' router compute
   them; 2.5 at the sum is for a coordinator that sends normalized weights.
3. `moe_recv_during` runs a hook (the shared expert) between the send and the collection.
4. `ReturnPath` holds the return handling: `FourPlaneSum` is the source's; `RowSharded` (the
   ranks' reduce-scatter, as the rank daemon designs it) is refused until `glm53f-wire` has its
   frame changes.
5. `quantize_hidden_batched` encodes on the host (the source used the attention crate's device
   encoder under `cuda`); the device quantizer is `glm53f_coord_quantize_hidden`.

## Tests

- **CPU** (`cargo test -p glm53f-coordinator`): 41 unit tests (sampling, radix, host tier, queue,
  spec, streaming, wire over four mock ranks, E4M3, engine helpers) and 24 integration tests:
  `tests/scheduler.rs` (11: batched decode, mixed greedy and sampled rows, admission waiting and
  refusal, prefill segments between decode steps, verify windows with partial accepts, stops
  inside an accepted run, radix reuse exact / extending / divergent with the branch gap, a
  deferred identical prompt forking a running request, snapshots through RAM under bank pressure,
  retained slots through RAM under memory pressure, a parked prefill resumed), `tests/host_tier.rs`
  (3: page sharing and exact restores, the v1.1.1 eviction case, prompt before turn),
  `tests/engine.rs` (7), `tests/glm_prompt.rs` (3: 41 reference template renders through the
  API's parser; with `GLM53F_TOKENIZER` set, the official tokenizer's ids for 40 of them).
- **GPU** (`cargo test -p glm53f-coordinator --features cuda`): `tests/gpu.rs` (6): the sampler
  at GLM-5.3-Flash's width against `select_pick` (greedy, sampled, masked; padding peaks), the
  mask kernel and argmax bit for bit, the quantizer, rank sum and frame fill against the host.
