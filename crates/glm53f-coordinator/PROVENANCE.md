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
| Snapshot eviction only under load (the source's pressure eviction and bank overflow, the design's `on-evict` store mode) | mimo26f-afd @ bab9fa2 : crates/mimo26-coordinator/src/api.rs (lines 389-442 `evict_lru`, `take_free`, `make_room`, `make_room_bytes`; 631-688 `enforce_banks`, `grow_active`; 772 the bank's default) | as above | src/pool.rs (`Want`, `make_room`, `evict_point`, `evict_lru`, `take_free`, `enforce_banks`, `grow_active`), src/scheduler.rs (`SchedulerConfig::bank`), src/hostcache.rs (`victim` shared, `capture_for`) | Written here (2026-09-29) on the source's design: no bank cap by default; room for an incoming request made one point at a time, least recently used first, wherever it lives, running and prefilling requests' included; see Scheduler and slot pool, item 15 | tests/paging.rs; tests/scheduler.rs and tests/host_tier.rs unchanged, also under `GLM53F_PREFIX_CACHE_ENTRIES=24`; glm53f-forward tests/paging.rs | 2026-09-29 |
| Bounded queue and `Engine::admit` | same file (lines 97-107, 1218-1249) | as above | src/queue.rs | A type of its own (`Queue::admit`); depth default = slot count, 16 | unit tests in the file; tests/engine.rs | 2026-09-28 |
| `Engine` implementation (`generate`, keep-alive, cancel, `LAST_ENCODE`, `emit_delta`, `CancelOnDrop`, `image_token_id`, `image_spans`, `encode_marked`) | same file (lines 33-40, 130-205, 1197-1394, tests 1460-1519) | as above | src/engine.rs | Text side behind `PromptCodec`; `render_prompt` / `tokenize_prompt` with `PromptOptions`; template errors returned by `generate`; `encode_marked` became `expand_image_marker` (the GLM template renders the image delimiters itself); the host reference backend dropped. `health` (written here, 2026-09-29): serving while the scheduler's thread runs and a check the daemon adds passes (`with_health`; the expert wire's state); it touches neither the queue nor the model | unit tests in the file; tests/engine.rs | 2026-09-28 |
| Verify-length policy (`SpecPolicy`, `chain_length`, `verify_lengths`) | mimo26f-afd @ bab9fa2 : crates/mimo26-coordinator/src/dforward.rs (lines 937-994, test 2280-2295) | `c048732e72d1b78886a73fc4bb50b820a2fb0077c208909d1fc269d3a62a25fb` | src/spec.rs | Draft probabilities as slices of any length (were `[f32; 7]`); every length capped at the drafts the model returned; `from_env` gathers the three variables | unit tests in the file | 2026-09-28 |
| Speculative step's acceptance (`spec_step`, the accept loop of `verify_batch`'s caller) and the selection (`select`, `select_host_row`, `select_rows_host`) | same file (lines 134-160, 1611-1675, 1798-1833) | as above | src/scheduler.rs (`spec_step`), src/gpu.rs (`Sampler`, `select_rows_host`) | Acceptance moved into the scheduler over `ModelForward::draft` / `verify` / `commit`; selection is a `Sampler` any model's forward calls, with masks | tests/scheduler.rs, tests/gpu.rs | 2026-09-28 |
| Host RAM tier | mimo26f-afd @ bab9fa2 : crates/mimo26-coordinator/src/hostcache.rs | `ec6931686c930754ede3d9d128e5bd720f25cec0fb701f4b7a85c002eb087a70` | src/hostcache.rs | Over `KvSlot` page and mark export; page size, state size and budget split are parameters; radix lookup; see below. Tests `eviction_is_least_recently_used_first` and `page_chain_identifies_prefixes` kept (the latter at 256- and 64-token pages) | unit tests in the file; tests/host_tier.rs; tests/scheduler.rs | 2026-09-28 |
| Sampling contract and CPU reference | mimo26f-afd @ bab9fa2 : crates/mimo26-coordinator/src/sampling.rs | `e9a1393a691766c98762af07a1986acaf612a682a59a6537833e36e9fd7e8679` | src/sampling.rs | Module doc (bound, masks); `After::from_logits` takes the id bound; new `greedy` (the kernel's rule, bounded), `Mask`, `apply_mask`, `select_pick`. `Sampling`, `DeviceRow`, `select` and their tests verbatim | unit tests in the file (7 kept, 4 new); tests/gpu.rs | 2026-09-28 |
| Streaming holdback (`flush_pending`) | mimo26f-afd @ bab9fa2 : crates/mimo26-coordinator/src/streaming.rs | `8fc947479f0113a589145f958cdbfa4f313c5bb5476135f7c9cd8853c9346e54` | src/streaming.rs | Takes a decode function (ids to bytes) in place of MiMo's tokenizer; the tests' toy tokenizers became toy decoders; test bodies verbatim | unit tests in the file | 2026-09-28 |
| Expert wire client and FP8 K32 quantizer | mimo26f-afd @ bab9fa2 : crates/mimo26-coordinator/src/wire.rs | `4a9a1fb866b914cec532c45d51d2e7e1ab558a1b978c28afd63c24059e3f2561` | src/wire.rs | `WireConfig` (experts 288, routed scale, `ReturnPath`); route validation; the scale on the host sum; `moe_recv_during`; `returned`; host-only batched quantizer; module doc; one pointer to a private document removed; tests reworked around a four-rank mock (below); the row-sharded return path (`ReturnPath::RowSharded`, `collected`, request ids from a random base per connection; below); exchanges in flight (written here, 2026-09-29): `WireConfig::depth` (1 to `MAX_DEPTH`, 4) instead of the fixed two over RDMA, a receive slot per exchange in flight and rank, the depth lowered to the receive slots every rank names in its handshake (an older rank: two), `WireClient::depth` in place of the private limit, `host_bytes` | unit tests in the file (`the_depth_is_bounded_and_one_over_tcp`); tests/row_sharded.rs | 2026-09-28 |
| E4M3 codec | mimo26f-afd @ bab9fa2 : crates/mimo26-load/src/e4m3.rs (lines 14-81) | `767fc973a21aa65c09616c3acfa74eaa32f729dab33c7eeac9f7c5948a4bcfe3` | src/fp8.rs | `E4M3_MAX`, `decode_e4m3`, `decode_table`, `encode_e4m3` verbatim; module doc; two comments name the source repository; tests written here | unit tests in the file | 2026-09-28 |
| Sampling kernel | mimo26f-afd @ bab9fa2 : crates/mimo26-coordinator/kernels/sample.cu | `18267fac23b314a6ec4e09f0bcb72b91db143074a1c5615a7f9186d91e068ed5` | kernels/sample.cu | Entry point renamed; three comment lines. Kernel body verbatim | tests/gpu.rs, examples/sample_check.rs | 2026-09-28 |
| Device argmax | mimo26f-afd @ bab9fa2 : crates/mimo26-coordinator/kernels/dflash.cu (lines 163-229, 256-267) | `ecb17cd49a5f1b9e636ab8d051155bae0bc87eee4d3757f0ed3833cd9b8f4caa` | kernels/select.cu | `argmax_kernel` verbatim; its two entry points renamed; called with the tokenizer's id bound | tests/gpu.rs | 2026-09-28 |
| Wire kernels: UE8M0 scales, rank sum, frame fill, `grid_for` | mimo26f-afd @ bab9fa2 : crates/mimo26-coordinator/kernels/glue.cu (lines 125-143, 200-213, 222-229, 263-274, 311-348) | `79a306467d6937235e00299d8422fc2328b3b4a73824a47af798f8bc9dddc4e7` | kernels/wire.cu | Kernels verbatim except the rank sum, which multiplies by a scale after the four adds (`__fmul_rn`; 1.0 gives the source's bits); all kernels in the file's anonymous namespace (the source defined `frame_fill_kernel` inside its `extern "C"` block); entry points renamed, the rank sum's with the scale parameter | tests/gpu.rs | 2026-09-28 |
| Device E4M3 encoder | mimo26f-afd @ bab9fa2 : crates/mimo26-attn/kernels/include/mimo26_attn_device.cuh (lines 33-55) | `4a4028f07604020f6adf69fb91f3af1ec9639eeab740fb4ac9af8eb9e2a9f5d2` | kernels/wire.cu | Body verbatim; its comment rewritten without a pointer to a private document | tests/gpu.rs | 2026-09-28 |
| Device hidden quantizer | mimo26f-afd @ bab9fa2 : crates/mimo26-attn/kernels/kv_cache_fp8.cu (lines 160-169, 171-178) | `861f85df38b3641374a2a4ea1478c9c67502f9b4975e539da37dc0c569a5d5c3` | kernels/wire.cu | Kernel verbatim (namespace qualifier dropped); entry point renamed, takes a `cudaStream_t` | tests/gpu.rs | 2026-09-28 |
| Sampler check | mimo26f-afd @ bab9fa2 : crates/mimo26-coordinator/examples/sample_check.rs | `0fd93f37f84a1524b090abedb2d5d09ee942b6548b02ad94abd12f9df6539e99` | examples/sample_check.rs | GLM-5.3-Flash's widths (154,880 rows, ids below 154,856); `Pick`s through `gpu::select_rows_host` against `select_pick`; the timing covers the whole selection | `cargo run --release --features cuda --example sample_check` | 2026-09-28 |
| Build script | mimo26f-afd @ bab9fa2 : crates/mimo26-coordinator/build.rs | `a40762247399ff0f0c32e242210ba51322c07007b882cf12d483d4cc722fe15b` | build.rs | Rewritten in the shape of this repository's other kernel crates (`GLM53F_NVCC`, `GLM53F_CUDA_ARCH`, `GLM53F_CUDA_LIB`, cudart only); the source's nvcc flags kept | `--features cuda` builds | 2026-09-28 |
| Copy windows' index (`CopyIndex`: grams, the chain of earlier occurrences, the longest backward match, the periodic run) | mimo26f-afd @ c6cc2ff (v1.3.0) : crates/mimo26-coordinator/src/copy.rs | `6ae2b7aad8e5874f054edfc8c2da7ae6c520be16a86a2d63621e3b24702f29e8` | src/copy.rs | Token ids `u32`; an entry of 24 matching tokens (`ENTRY`, `with_entry`) over the source's 8-token grams; grams keyed by a 64-bit fingerprint in the standard library's keyed map, each candidate checked against the context; at most `CATCH_UP` positions indexed a call, no copy until caught up; `reserve`; ids at or past the model's bound end a copy, and a copy under 2 tokens is none; `matched`, `indexed`, `bytes`. See Copy windows below | `copy::tests` (the source's 4 tests at its entry of 8 through `with_entry`, with the bound; 3 new); tests/copy_windows.rs; glm53f-forward tests/copy_windows.rs | 2026-09-29 |
| Copy windows in the speculative step (greedy requests only; a copy verified in place of drafts; the drafter skips the request, whose kept rows still reach its context) | mimo26f-afd @ c6cc2ff : crates/mimo26-coordinator/src/api.rs (lines 221-222, 721, 767, 1029-1034) and src/dforward.rs (lines 1652-1695) | `5557d695ed06ab52b8e8f1d3be8bfbaa361148a9d5a583b9bcb9e7c0057cc1d1` (api.rs), `f6f449c5e9bb7fb73c2dab4003b0c3b41658fee9f24f78507279dc1549a50508` (dforward.rs) | src/scheduler.rs (`copies`, `spec_step`), src/pool.rs (`Active::copy`) | In the model-agnostic scheduler over `ModelForward::draft` (asked only for the requests without a copy); copies join the step's row budget at `copy::TOKEN_P` a token; `SchedulerConfig::copy_windows`, off in the shell (the daemon's `--copy-windows`, on by default, in place of `MIMO26_COPY`); counters and the `[copy]` line | tests/copy_windows.rs; glm53f-forward tests/copy_windows.rs | 2026-09-29 |

## Written here

| File | What | Date |
|---|---|---|
| src/model.rs | `KvSlot` and `ModelForward`, the cut between the shell and a model, with `Pick`, `Segment`, `DecodeRow`, `DraftRow` (with the pick of its window's first row, which a drafter may sample with), `Draft`, `Window`, `Limits`, `ImageSpan` | 2026-09-28 |
| src/radix.rs | `RadixIndex`: a compressed trie over token ids with exact hits and a granular shared-prefix count | 2026-09-28 |
| src/glm_prompt.rs | `GlmPrompts`: GLM-5.3-Flash's tokenizer and chat template (`glm53f-tokenizer`) behind `PromptCodec`, the official template enforced | 2026-09-28 |
| src/gpu.rs | FFI to the kernels and the few CUDA runtime calls, `DeviceBuffer`, `Sampler`, page-locking | 2026-09-28 |
| src/lib.rs, Cargo.toml | Crate root and manifest | 2026-09-28 |
| examples/copy_cost.rs | The copy index's host cost: the indexing rate, and a step's lookup at 1 and 48 requests with nothing and with everything matching, contexts up to 1,048,576 tokens | 2026-09-29 |
| examples/copy_replay.rs | Copy windows replayed on real text through the tokenizer and template (no model): rounds with copies at an entry of 8 and of 24 tokens, and a copy at every position by the match behind it | 2026-09-29 |
| kernels/select.cu (`mask_rows_kernel`, `glm53f_coord_mask_rows`) | The per-row grammar mask | 2026-09-28 |
| kernels/glm53f_coord.h | The kernels' C ABI | 2026-09-28 |
| tests/common/mod.rs | A toy deterministic model and slot behind the traits (its positional state is a hash of the whole prefix, checked on every call), simulated memory and time, a call log | 2026-09-28 |
| tests/common/mod.rs (`copying_logits`, `copying_reference`, `MockModel::copying`), tests/copy_windows.rs | A copying variant of the toy model (its pick follows the last earlier occurrence of its last 8 tokens, one of them may differ, with edits) and the copy windows' tests (see Tests) | 2026-09-29 |
| tests/scheduler.rs, tests/host_tier.rs, tests/engine.rs, tests/glm_prompt.rs, tests/gpu.rs | See Tests | 2026-09-28 |
| tests/common/mod.rs (`MockModel::panic_in_prefill`), tests/engine.rs (`health_follows_the_scheduler_and_the_added_check`) | A toy model whose prefill panics, so the scheduler's thread ends, and the engine's health against it, an added check and a full queue | 2026-09-29 |
| tests/paging.rs; tests/common/mod.rs (`test_config` reads `GLM53F_PREFIX_CACHE_ENTRIES`) | Snapshots and the RAM tier under load and without (see Tests); the whole suite can run under a bank cap | 2026-09-29 |
| tests/row_sharded.rs | Four real rank daemons on one GPU over loopback: the row-sharded return against the four-plane sum within the rank README's bound (BF16 and FP8 exchange) and both against the oracle's layers 3 and 4; decode stays four-plane; loopback timings; a dead peer fails the request | 2026-09-28 |
| examples/wire_bench.rs | The exchange alone against four running ranks, four-plane and row-sharded on the same synthetic rows: per-exchange times, bytes into the coordinator, the two outputs' difference | 2026-09-28 |
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
    1,024 / 8,192 / 64 allowance) is a `SchedulerConfig` field with the source's value, except
    the bank, 0 (no cap) since item 15; the clock
    is injectable; `Scheduler::step` runs one pass (the source's loop is `run`).
12. **Growth headroom** before a step is `2 * drafts + 2` rows with the model's draft count (the
    source always used 7 drafts).
13. **Two lanes** are the model's: a pass gets the whole batch and may split it into lanes, as
    the source's forward did. DS41RT's independent lanes, each with its own rounds, are not
    implemented.
14. Not ported: the host reference backend (one request at a time on the CPU), vision encoding.
    New: counters (`SchedStats`, `PoolStats`).
15. **Snapshots leave the device only under load (written here, 2026-09-29).** The source copied
    a snapshot to RAM in two cases: its bank over 24 points (`MIMO26_PREFIX_CACHE_ENTRIES`), with
    free memory or not, and pressure, which evicted whole retained slots only, so running
    requests' marks (141 MiB each for GLM-5.3-Flash) could hold the pool while an incoming prefill
    was refused. Here:
    - the cap is optional, 0 (none) by default; `GLM53F_PREFIX_CACHE_ENTRIES=N` restores the
      source's bank overflow unchanged (`Pool::enforce_banks`), with a `[coordinator] bank cap:`
      log line;
    - when an admission (a prefill, a restore, an in-place resume growing its slot), a running
      request's growth or the image encoder needs memory the check does not find, points are
      evicted one at a time (`Pool::evict_point`), least recently used first by the RAM tier's
      victim rule (`hostcache::victim`, now shared), wherever they live: retained slots' points
      and the marks of running and prefilling requests, which run on. Each is stored to RAM first
      when the tier is on, then its mark dropped; a retained slot left without points is freed.
      Eviction stops as soon as the check passes. At a full tie a retained slot's point goes
      first (it never frees less than a running request's mark). The source evicted whole
      retained slots here; `take_free` still does when a slot is needed;
    - a point that serves a fork is marked used whatever holds it (the source refreshed only
      retained points), so a running request's hot prompt point is not the least recently used;
    - the in-place rewind's later points and a relocation through RAM count as evicted points;
    - `PoolStats` counts `evicted_points`, `evicted_in_flight` and `bank_overflows`; the
      `[coordinator] device pressure:` and `slot pressure:` lines name each evicted point, its
      holder, and the request that needed the room, and the `[hostcache] store` line says why
      (`HostCache::capture_for`).

### Copy windows (mimo26f-afd v1.3.0)

1. **The entry is 24 tokens.** The source proposed a copy on any match of its 8-token grams, as
   TensorFold's tree proposer does; TensorFold's chain proposer (`DFlashProposer.propose`) replaces
   the drafter's block only on a match of 24 (`confident_match`). This engine's drafter is a chain
   cut at a confidence of 0.7, whose windows are short where it is unsure, while a copy verifies up
   to 8 rows. Replayed on real text (`examples/copy_replay.rs`), an entry of 8 copied in 101
   rounds of a 3,959-token fresh-code reply, keeping 37% of the copied tokens; 24 copied in 2, and
   in none of a prose reply, while rewrites, edits and quotes copied 86-96% of their replies at
   7.9 tokens a copy round. The index keeps the source's 8-token grams; the longest match among the
   candidates must reach the entry.
2. **Fingerprinted grams.** The source keyed its map by the 8 ids; here by a 64-bit fingerprint
   (half the table at a million positions) in the standard library's keyed map, so a prompt cannot
   flood a bucket. Each candidate is compared with the context, so a collision costs a candidate.
3. **Bounded indexing.** The source indexed the whole context at a request's first proposal. Here
   a call indexes at most `CATCH_UP` (16,384) positions and copies nothing until it has caught up,
   so a million-token prompt (prefilled, or resumed from a snapshot) takes 64 steps, 1.1 ms each
   (the median; 13.6 ms for the first, which first touches the reserved table), instead of one
   step of about 90 ms (`examples/copy_cost.rs`). The scheduler reserves room for every position a
   request can reach, so the map never rehashes (unreserved, growing maps stalled a step of 48
   requests for up to 87 ms).
4. **Ids past the bound end a copy** (image rows), and a copy cut below 2 tokens is none.
5. **Where it runs.** The source proposed in its API loop and verified in its forward's
   `spec_step`, which drafted only for the requests without a copy. Here the scheduler proposes
   and asks `ModelForward::draft` only for the others; a copy skips the verify-length policy, as in
   the source, and joins the step's row budget (new here) at `copy::TOKEN_P` a token.
6. **The switch.** `SchedulerConfig::copy_windows`, off in the shell; the daemon's
   `--copy-windows on|off` (`GLM53F_COPY_WINDOWS`), on by default, in place of `MIMO26_COPY`.
   Counters in `SchedStats` and a `[copy]` line every 64 speculative steps (new).

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
4. `ReturnPath` holds the return handling: `FourPlaneSum` is the source's; `RowSharded` is the
   ranks' reduce-scatter (`DS41RTE3` v4, `glm53f_wire::row_shard`). The choice is made per
   exchange by its row count (`WireConfig::row_sharded`: from `min_rows`, at least 4, the
   design's 16; smaller exchanges stay four-plane) and travels with the exchange in flight, so
   the two RDMA lanes may differ. Reduce-scattered requests carry `FLAG_REDUCE_SCATTER` (and
   `FLAG_EXCHANGE_FP8` for the FP8 exchange), also on the in-place RDMA paths
   (`set_request_flags`); returns are validated as the rank's row slice (version 4, the row-slice
   flag, its partition's rows and first row); the host path assembles them with
   `CoordinatorSum::row_sharded`; `collected` hands out planes or row slices, and `returned`
   stays the four-plane accessor. `WireConfig::from_env` reads `GLM53F_ROW_SHARDED_MIN_ROWS` and
   `GLM53F_EXCHANGE_DTYPE`.
5. `quantize_hidden_batched` encodes on the host (the source used the attention crate's device
   encoder under `cuda`); the device quantizer is `glm53f_coord_quantize_hidden`.
6. Request ids start at a random base per connection (they started at 1): the ranks' peer mesh
   keys exchange frames by request id and layer, so a stale frame of an earlier connection can
   never match a new exchange. The host-sum path also checks each return's request and layer.
7. The in-place sends and the receive buffers work over TCP too (the source's were RDMA only):
   `send_buffers` allocates a staging body laid out as the RDMA one, `moe_send_mapped` and
   `moe_send_device` build the frame in it and write each rank's frame from it (its executor
   id, that connection's sequence and, unless disabled, its CRC32C), one exchange in flight;
   `plane_buffers` also hands out the connections' receive buffers, grown once to a whole
   receive slot so that they never move (the source's grew with each larger return). A
   caller's device paths then run on one machine; over TCP they save nothing.

## Tests

- **CPU** (`cargo test -p glm53f-coordinator`): 53 unit tests (sampling, radix, host tier, queue,
  spec, copy windows' index, streaming, wire over four mock ranks with the in-place sends' frames
  byte for byte, E4M3, engine helpers) and 36 integration tests (also green under
  `GLM53F_PREFIX_CACHE_ENTRIES=24`, the source's cap): `tests/paging.rs` (7: no load, no RAM
  traffic, every repeat a device hit, 152 RAM stores under the source's cap of 24 instead; an
  incoming prompt evicts exactly the seven least recently used points it needs, the last a running
  request's mark, with the tokens of a run without pressure and later restores from RAM; a running
  request's mark evicted as it runs; a prefilling request's point evicted; a growing request
  evicting its own mark last; the tier off drops them; a cap of 24 moves the oldest points as the
  source did),
  `tests/scheduler.rs` (12: batched decode, mixed greedy and sampled rows, admission waiting and
  refusal, prefill segments between decode steps, verify windows with partial accepts, the verify
  row budget, stops inside an accepted run, radix reuse exact / extending / divergent with the
  branch gap, a deferred identical prompt forking a running request, snapshots through RAM under
  bank pressure, retained slots through RAM under memory pressure, a parked prefill resumed),
  `tests/copy_windows.rs` (4, on the copying toy model: every token serial decoding's with copy
  windows on and off, copied windows never drafted for; sampled requests never copy; a stop inside
  a copied run commits only what was delivered; copies held to the row budget), `tests/host_tier.rs`
  (3: page sharing and exact restores, the v1.1.1 eviction case, prompt before turn),
  `tests/engine.rs` (8), `tests/glm_prompt.rs` (3: 41 reference template renders through the
  API's parser; with `GLM53F_TOKENIZER` set, the official tokenizer's ids for 40 of them).
- **GPU** (`cargo test -p glm53f-coordinator --features cuda`): `tests/gpu.rs` (6): the sampler
  at GLM-5.3-Flash's width against `select_pick` (greedy, sampled, masked; padding peaks), the
  mask kernel and argmax bit for bit, the quantizer, rank sum and frame fill against the host.
