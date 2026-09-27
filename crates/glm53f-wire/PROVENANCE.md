# glm53f-wire provenance

Copied from [hughmadden/mimo26f-afd](https://github.com/hughmadden/mimo26f-afd)
v1.2.0, commit `bab9fa2f2fc1e22ae67b56fbc1c209278f6a9d79` (MIT), directory
`crates/mimo26-wire`. The sha256 column is the digest of the file at that commit.

**Renames applied to every file** (not repeated in the Delta column):
`mimo26-wire` → `glm53f-wire`, `mimo26_wire` → `glm53f_wire`, and the environment
prefix `MIMO26_` → `GLM53F_` (`GLM53F_SPIKE_NAIVE`, `GLM53F_WIRE_NAIVE`,
`GLM53F_WIRE_NOCRC`).

**Kept on purpose:** the frame magic `DS41RTE3`, version 3 and every byte layout.
They are the contract with the expert ranks. Comment labels such as ADVISOR-I4 §…,
I4/I5 items, perf reset R2/P6/P9, R8, A6 and ARCHITECTURE.md §… refer to
mimo26f-afd's design records and are kept verbatim as provenance.

**288 experts.** Nothing in this crate hard-codes MiMo's 256 experts: expert ids
are `u32` route-entry fields and no code bounds them. Hidden 4,096, top-8 and four
ranks are the same for GLM-5.3-Flash. `tests/glm_experts.rs` pins ids 256–287.

| Unit | Source (repo @ commit : path) | sha256 (source file) | Here | Delta | Pinned by | Date |
|---|---|---|---|---|---|---|
| Manifest | mimo26f-afd @ bab9fa2 : crates/mimo26-wire/Cargo.toml | `de6e2c81d73200b7362d57eee3db00ae288fb10ed1289fb093e741efb67c89b2` | Cargo.toml | Header comment: build and trap-suite commands for this repository (`cargo test -p glm53f-wire`, `GLM53F_SPIKE_NAIVE=1`); ds41rt source named by its public repository; pointers to unpublished documents and build tooling removed | build | 2026-09-28 |
| Crate root, static row guards | mimo26f-afd @ bab9fa2 : crates/mimo26-wire/src/lib.rs | `5fa653562afc9ec782aea849c5ea381ba5f73ba650e282574b1d5e4a12aa0101` | src/lib.rs | Docs only: byte budgets recomputed for GLM-5.3-Flash's 42 MoE layers (4,360 × 42 × 4 = 0.73 MB out, 8,192 × 42 × 4 = 1.38 MB in per token; were 47 layers, 0.82 / 1.54 MB); note that 288 experts need no layout change; ds41rt derivation names tpurtell/ds41rt v10 in place of an unpublished path and port document | static guards (build); tests/wire_layout.rs | 2026-09-28 |
| Layout, offsets, wire codes | mimo26f-afd @ bab9fa2 : crates/mimo26-wire/src/layout.rs | `28fa05d70bbb2f5f3b88cb4a4e6dbc826ef07dc3c641303688ef3d372b1c6fb3` | src/layout.rs | Docs only: same budget recomputation and derivation pointer; `HIDDEN` and `TOPK` comments name GLM-5.3-Flash (hidden 4,096; top-8 of 288). No constant changed | tests/wire_layout.rs, tests/glm_experts.rs | 2026-09-28 |
| Frame codec, `RequestView`, in-place encoders | mimo26f-afd @ bab9fa2 : crates/mimo26-wire/src/frame.rs | `43d9f63b3e0e6b87f5a58a7eece74542711c6707cf822195317abde2652e50dc` | src/frame.rs | Renames (`GLM53F_WIRE_NOCRC`); derivation comment names tpurtell/ds41rt v10 in place of an unpublished port document | tests/wire_layout.rs, l4_integrity.rs, request_view.rs, request_meta_into.rs, in_place_return.rs, glm_experts.rs | 2026-09-28 |
| L4 ladder, `CoordinatorSum` | mimo26f-afd @ bab9fa2 : crates/mimo26-wire/src/l4.rs | `ea6ba1b97d8f0f43a2a40ed78b8ff6f22aebf09cbe2aac0f9c629eb961a61d4e` | src/l4.rs | Verbatim | tests/l4_integrity.rs, r8_rank_order.rs | 2026-09-28 |
| Non-blocking expert client | mimo26f-afd @ bab9fa2 : crates/mimo26-wire/src/async_api.rs | `45c8747713a5491a5d76ab0e8d13e854032166c3e0791744faf6d2ca6b32efb1` | src/async_api.rs | Verbatim | tests/async_api.rs | 2026-09-28 |
| CRC32C (table, SSE4.2, AArch64 CRC) | mimo26f-afd @ bab9fa2 : crates/mimo26-wire/src/crc32c.rs | `5193c371401ba40371c9484e8f46f0f4244304d39216f0fb0ca9e91a83ba0b1c` | src/crc32c.rs | Renames only (the bench command in a comment) | unit tests in the file; tests/wire_layout.rs (RFC 3720 vector, bitwise reference) | 2026-09-28 |
| BF16 conversions | mimo26f-afd @ bab9fa2 : crates/mimo26-wire/src/bf16.rs | `50699cc618a4bcb3b14edc5abd970f83c8a26d582e62858b8610e4cedc7fdcbf` | src/bf16.rs | Verbatim | tests/wire_layout.rs (`n_bf16_rounds_to_nearest_even`) | 2026-09-28 |
| Error taxonomy and dispositions | mimo26f-afd @ bab9fa2 : crates/mimo26-wire/src/error.rs | `b0ac84184b1c96d55b9416661fac7a8982cb29c85fd4e70ddb233b50e8a71b18` | src/error.rs | Verbatim | tests/l4_integrity.rs (`disposition_classification`) | 2026-09-28 |
| Trap switches | mimo26f-afd @ bab9fa2 : crates/mimo26-wire/src/naive.rs | `a274fdb10ad8e6133074ee479757a5ed2e46326461722d628de17fca0308a1bd` | src/naive.rs | Renames only | tests/wire_layout.rs (`every_naive_bit_is_a_killable_wrong_impl`); the naive run | 2026-09-28 |
| Test fixtures and L4 session harness | mimo26f-afd @ bab9fa2 : crates/mimo26-wire/tests/common/mod.rs | `f46f05ecbc397d2327da470eab22284262c76eacc059561531187c4e95d0c226` | tests/common/mod.rs | Renames only | — | 2026-09-28 |
| Layout trap suite | mimo26f-afd @ bab9fa2 : crates/mimo26-wire/tests/wire_layout.rs | `b56418b77cdd7b00e16a9f28ae0b4461848f6940094c63b1fe236391ed607627` | tests/wire_layout.rs | Renames only | — | 2026-09-28 |
| L4 trap suite | mimo26f-afd @ bab9fa2 : crates/mimo26-wire/tests/l4_integrity.rs | `659b4b808eff3d35ff7a86c2fa68f25236f54c829d5e8f0d6f7622f60ff0b22e` | tests/l4_integrity.rs | Renames only | — | 2026-09-28 |
| Async API tests | mimo26f-afd @ bab9fa2 : crates/mimo26-wire/tests/async_api.rs | `389d94adb82ca03de7f7a0dcd14f11ebea18640bb912f12f6fbd07dbfc39bab0` | tests/async_api.rs | Renames only | — | 2026-09-28 |
| In-place return test | mimo26f-afd @ bab9fa2 : crates/mimo26-wire/tests/in_place_return.rs | `e28b82b3373d6e556d5326014746c6d8cc56fef8a55da10340e65c0df3c26be9` | tests/in_place_return.rs | Renames only | — | 2026-09-28 |
| Rank-order sum test | mimo26f-afd @ bab9fa2 : crates/mimo26-wire/tests/r8_rank_order.rs | `cff7bfeeb2ff6a1d7191e6fdf2628e8de622ab92e11300827433ca7a06236257` | tests/r8_rank_order.rs | Renames only | — | 2026-09-28 |
| In-place request encoder test | mimo26f-afd @ bab9fa2 : crates/mimo26-wire/tests/request_meta_into.rs | `97d2449e13d12eca589af2cf369678f2a392831749bf790bbb6f376a96923caf` | tests/request_meta_into.rs | Renames only | — | 2026-09-28 |
| Zero-copy view test | mimo26f-afd @ bab9fa2 : crates/mimo26-wire/tests/request_view.rs | `9fda7f3df2fb58614bc6e52467c2ad76e4752f472afe5d407cd577775ca14157` | tests/request_view.rs | Renames only | — | 2026-09-28 |

## Written here

| File | What | Date |
|---|---|---|
| tests/glm_experts.rs | Expert ids 256–287 (GLM-5.3-Flash's 288 experts) round-trip through `encode_request_seq`, `decode_frame`, `RequestView` and `encode_request_meta_into`; the 12-B entry bytes for expert 287. Both runs. | 2026-09-28 |
| PROVENANCE.md | This ledger. | 2026-09-28 |
