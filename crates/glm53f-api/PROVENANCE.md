# glm53f-api provenance

Copied from [hughmadden/mimo26f-afd](https://github.com/hughmadden/mimo26f-afd)
v1.2.0, commit `bab9fa2f2fc1e22ae67b56fbc1c209278f6a9d79` (MIT), directory
`crates/mimo26-api`, plus one golden file from its `harness/`. The sha256 column is
the digest of the file at that commit.

**Renames applied to every Rust file** (not repeated in the Delta column):
`mimo26-api` → `glm53f-api`, `mimo26_api` → `glm53f_api`, and the environment
prefixes `MIMO26_` / `MIMO26F_` → `GLM53F_` (`GLM53F_PYTHON` in the tests).

**The model boundary.** The source served MiMo-V2.6-Flash only. The parts that
depend on the model now sit behind two seams, so a GLM-5.3-Flash dialect and
engine drop in without touching the HTTP code:
- `Dialect` (`src/dialect/`): splits a completion into content, reasoning and tool
  calls; gives the streaming splitter its tags; says whether the prompt opened the
  reasoning block. `MimoDialect` is the MiMo parser, unchanged, behind it.
- `Engine::decode_image`: image decoding and token counting depend on the vision
  tower's patch geometry, so they moved from the API (which called the
  `mimo26-image` crate) to the engine. The default refuses, as an engine without
  an image encoder did before.

The served model id is `glm-5.3-flash`. Comment labels such as A8, I5-R8, D2b,
D4, D6, T24/T27/T29 (COHERENCE-TRAPS), ADVISOR-I3 §… and perf reset Q2/V2/V3 refer
to mimo26f-afd's design records and are kept verbatim as provenance.

| Unit | Source (repo @ commit : path) | sha256 (source file) | Here | Delta | Pinned by | Date |
|---|---|---|---|---|---|---|
| Manifest | mimo26f-afd @ bab9fa2 : crates/mimo26-api/Cargo.toml | `285d5dfe15fca09ebe99f9a86b955769352ad9f8044dbbe7f0bad446994225d4` | Cargo.toml | `mimo26-image` dependency removed (no dependencies left); description and header comment describe the `Dialect` and `Engine` seams; team-role and lockfile notes removed | build | 2026-09-28 |
| Crate root, `serve`, router | mimo26f-afd @ bab9fa2 : crates/mimo26-api/src/lib.rs | `ee79547ae9a73125b9b818b211f24d49e11ca7b74fb438b00f8ad4522425f722` | src/lib.rs | `pub mod parser` → `pub mod dialect`; re-exports `Dialect`, `ParseResult`, `ParsedCall`, `StreamTags`; `serve(addr, engine, dialect)` and the router pass an `Arc<dyn Dialect>`; crate doc reworded | tests/acceptance.rs | 2026-09-28 |
| Chat handler, SSE stream, holdback | mimo26f-afd @ bab9fa2 : crates/mimo26-api/src/chat.rs | `1b933d74b41244a479d55929bca97c186619c8da4a3c1b7f1d130e19cd81c6f6` | src/chat.rs | `handle(engine, dialect, body)`; completions parsed by `dialect.parse(text, tools, thinking, cap)`; the MiMo tag constants moved to the MiMo dialect; `StreamSplit` takes the dialect's `StreamTags` and starts inside reasoning when `dialect.reasoning_first(thinking)`; the request parse gets the engine's image decoder. Splitting logic otherwise unchanged | tests/acceptance.rs (streaming, holdback, think-block, tool-call and seam tests) | 2026-09-28 |
| MiMo tool-call parser → `MimoDialect` | mimo26f-afd @ bab9fa2 : crates/mimo26-api/src/parser.rs | `76568b6d6b9d4b21ad22f487dfa640cdb5e87e1629fc240a4c284b1e3a383aea` | src/dialect/mimo.rs | Parser and its unit tests verbatim; `ParsedCall` and `ParseResult` moved to `dialect/mod.rs`; adds `MimoDialect` (`parse` ignores `thinking`; tags `<think>`, `</think>`, `<tool_call>`; `reasoning_first` false); module doc names the dialect | unit tests in the file; tests/acceptance.rs (`t29_parser_goldens_pass`, `nameless_tool_call_is_rejected_400`, tool-call tests) | 2026-09-28 |
| `Engine` trait, params, images, queue place | mimo26f-afd @ bab9fa2 : crates/mimo26-api/src/engine.rs | `d9cb64654e862db7cc5e1cf1dcb358db29e3829cfb2c378bbbb59d28df560787` | src/engine.rs | Adds `Engine::decode_image(data_url)` with a refusing default; the module doc and the `image_marker` doc no longer name MiMo's template tokens | tests/acceptance.rs (image, sampling and queue tests) | 2026-09-28 |
| Request types and validation | mimo26f-afd @ bab9fa2 : crates/mimo26-api/src/types.rs | `a9675fa6c01ff8b61ead01c64a48621fe5e220c7a7f243de2d6c510b6f96b448` | src/types.rs | `MODEL_ID` = `glm-5.3-flash` (was `mimo-v2.6-flash`); `ChatRequest::parse(body, decode)` takes a `DecodeImage` and calls it in place of `mimo26_image::decode_data_url` + `merged_tokens`. The `data:`-URL rule, the FNV-1a hash, the 16-image cap and the markers are unchanged | unit tests in the file; tests/acceptance.rs | 2026-09-28 |
| `/v1/models` | mimo26f-afd @ bab9fa2 : crates/mimo26-api/src/models.rs | `a786209af320882057803248b82b858079feae60d6a79ac9da1ed2eb579bbd6a` | src/models.rs | Doc names the new id | `models_lists_the_model_id` | 2026-09-28 |
| HTTP/1.1 server | mimo26f-afd @ bab9fa2 : crates/mimo26-api/src/http.rs | `f0abe6e1cf1fa21e86ce5aa5dead48c9d2c95c4dd16f4c9e55b9253f19899022` | src/http.rs | One doc line reworded | tests/acceptance.rs | 2026-09-28 |
| JSON codec | mimo26f-afd @ bab9fa2 : crates/mimo26-api/src/json.rs | `52c52df6c298bf29877e26d976b1e9227c8e58abc45fe2601670558fcd116aa5` | src/json.rs | One doc line reworded | unit tests in the file | 2026-09-28 |
| Acceptance tests | mimo26f-afd @ bab9fa2 : crates/mimo26-api/tests/acceptance.rs | `d69299ce18e80fc2dd29eaad4cc5511243d4e95ff9e01a46fac303234ccdcd71` | tests/acceptance.rs | Stubs are served through `MimoDialect` (`start_engine_with`); parser calls go to `dialect::mimo::parse`; request bodies send `glm-5.3-flash`; T29 goldens read from `tests/fixtures/mimo/`; harness-driven tests find their files under `GLM53F_HARNESS` or `harness/` and skip with a message while absent; `VisionStub` implements `decode_image` with a fixed 8 × 8, 4-token result (real PNG decoding belongs to the image crate's tests); two comments no longer name a client tool and a proxy product; one test added (below) | — | 2026-09-28 |
| MiMo parser goldens (T27/T29) | mimo26f-afd @ bab9fa2 : harness/goldens/t29_parser_goldens.json | `2f7f4773aa5eb063166fd68a2e2a21164f339845a711494f185e079ff4595a3e` | tests/fixtures/mimo/t29_parser_goldens.json | Verbatim; kept with the crate because they pin the MiMo dialect | `t29_parser_goldens_pass` | 2026-09-28 |

## Written here

| File | What | Date |
|---|---|---|
| src/dialect/mod.rs | The `Dialect` trait and `StreamTags`; `ParsedCall` and `ParseResult`, moved verbatim from mimo26f-afd's `src/parser.rs`. | 2026-09-28 |
| tests/acceptance.rs (end) | `reasoning_opened_by_the_prompt_streams_as_reasoning`: a test-local dialect whose prompt opens the reasoning block (as GLM-5.3-Flash's template does); streamed reasoning and content match the non-stream parse. | 2026-09-28 |
| PROVENANCE.md | This ledger. | 2026-09-28 |
