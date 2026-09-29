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
| Acceptance tests | mimo26f-afd @ bab9fa2 : crates/mimo26-api/tests/acceptance.rs | `d69299ce18e80fc2dd29eaad4cc5511243d4e95ff9e01a46fac303234ccdcd71` | tests/acceptance.rs | Stubs are served through `MimoDialect` (`start_engine_with`); parser calls go to `dialect::mimo::parse`; request bodies send `glm-5.3-flash`; T29 goldens read from `tests/fixtures/mimo/`; harness-driven tests find their files under `GLM53F_HARNESS` or `harness/` and skip with a message while absent; `VisionStub` implements `decode_image` with a fixed 8 × 8, 4-token result (real PNG decoding belongs to the image crate's tests); two comments no longer name a client tool and a proxy product; tests added (below) | — | 2026-09-28 |
| MiMo parser goldens (T27/T29) | mimo26f-afd @ bab9fa2 : harness/goldens/t29_parser_goldens.json | `2f7f4773aa5eb063166fd68a2e2a21164f339845a711494f185e079ff4595a3e` | tests/fixtures/mimo/t29_parser_goldens.json | Verbatim; kept with the crate because they pin the MiMo dialect | `t29_parser_goldens_pass` | 2026-09-28 |

## Written here

| File | What | Date |
|---|---|---|
| src/dialect/mod.rs | The `Dialect` trait and `StreamTags`; `ParsedCall` and `ParseResult`, moved verbatim from mimo26f-afd's `src/parser.rs`. | 2026-09-28 |
| tests/acceptance.rs (end) | `reasoning_opened_by_the_prompt_streams_as_reasoning`: a test-local dialect whose prompt opens the reasoning block (as GLM-5.3-Flash's template does); streamed reasoning and content match the non-stream parse. | 2026-09-28 |
| tests/acceptance.rs (last section) | `GlmScript`, a scripted GLM-5.3-Flash stand-in served with `GlmDialect` (a small GLM-like prompt whose length is its token count, answers to the contract harness's prompts, three-character deltas); `reasoning_is_reasoning_content_whole_and_streamed`, `every_chunk_carries_the_completion_head`, and `api_contract_harness_passes_against_the_glm_script`, which runs `harness/api_contract.py` against it and requires every row to pass. | 2026-09-28 |
| src/dialect/glm.rs | `GlmDialect`, GLM-5.3-Flash's completion markup: reasoning first when the prompt opened the think block (`reasoning_first(thinking) = thinking`), thinking on by default; the tool-call envelope read in the order the streaming splitter walks it; argument values typed by the tool's JSON schema, inverting the chat template (a string property keeps its text exactly; other values are JSON when the schema allows); every loss reported, a nameless call with arguments an error, the tool-call cap. Written here; the tool-call format and the schema-typed values follow the GLM-5.3-Flash chat template (sha256 `0c4099f3382d6c92700dfb99725025360966fd73032f0ecf32377c0d9e6309c5`) and were cross-read with tpurtell/glmrt-5.3-1rtx-4spark @ `dc6d9b8` : `rust/crates/glmrt-api/src/tooling.rs` (sha256 `cb657995662cbca79bbb2515f822de2d8d3e14ec11a7987d185a61db38eed93f`), no code copied. Tests in the file: parser cases (reasoning, parallel calls, nested JSON, strings holding tags, malformed calls, cap, schema typing, template round trip), request-field mapping, two server tests (a call split across stream deltas; thinking off end to end) and a source-hygiene check. | 2026-09-28 |
| src/health.rs | `GET /health`: 200 `{"status":"ok"}` while `Engine::health` passes, 503 `{"status":"unavailable","reason":...}` with its reason when not; own code, in the shape of `models.rs`. | 2026-09-29 |
| src/auth.rs | The API key: `ApiKey` (never empty, a `Debug` that shows nothing of it), `ApiKey::check` (a request for `/v1...` must send `Authorization: Bearer <key>`, the scheme in any case), the constant-time comparison; own code, in the style of vLLM's `--api-key`. | 2026-09-29 |
| PROVENANCE.md | This ledger. | 2026-09-28 |

## Changed here (the GLM dialect and its request fields)

Additive changes to the imported files, so a GLM engine gets the request's thinking switch,
reasoning effort, `clear_thinking` and the history fields its chat template reads. The MiMo
dialect's behaviour is unchanged (its default thinking stays off).

| File | Change | Pinned by | Date |
|---|---|---|---|
| src/dialect/mod.rs | Registers `glm` (`GlmDialect` re-exported); new trait method `Dialect::default_thinking` (default off) | glm.rs tests | 2026-09-28 |
| src/types.rs | `ChatMessage` gains `reasoning_content` (from `reasoning_content`, else `reasoning`; image-marker characters removed as from content) and `tool_call_id`. `ChatRequest::enable_thinking` becomes `Option<bool>`, the first of `chat_template_kwargs.enable_thinking`, top-level `enable_thinking`, `thinking.type` (`disabled` off, any other type on) and `reasoning_effort: "none"` (off); new `reasoning_effort` (top level, else `chat_template_kwargs`) and `clear_thinking` (`chat_template_kwargs`, else `thinking`) | glm.rs `request_fields_map_to_the_template_switches` | 2026-09-28 |
| src/engine.rs | New `PromptOptions` (thinking, reasoning effort, clear_thinking); new `Engine::render_prompt` and `Engine::tokenize_prompt`, defaulting to `render_chat` and `tokenize` with the thinking switch alone, so existing engines are unchanged | glm.rs server tests; tests/acceptance.rs | 2026-09-28 |
| src/chat.rs | The thinking switch is the request's, else `Dialect::default_thinking`; the handler calls `render_prompt` and `tokenize_prompt` with the request's options, and passes the resolved switch to generation and parsing | glm.rs server tests; tests/acceptance.rs | 2026-09-28 |

## Changed here (one reasoning field, the chunk head)

The source streamed reasoning as `delta.reasoning` while its non-streamed message used
`reasoning_content`, and put `id`, `object`, `created` and `model` on the first streamed chunk
only. The first was found on the first run of the whole model, the second by the CREATED row of
`harness/api_contract.py`, which reproduces both on the development server (rows CREATED,
STREAM-MARKUP and TOOLS-stream).

| File | Change | Pinned by | Date |
|---|---|---|---|
| src/chat.rs | `REASONING_FIELD` = `reasoning_content`, used by the non-streamed message and every streamed reasoning delta (live, and after generation for a think block behind a tool call). Chunks are built by `ChunkHead`, which puts the completion's `id`, `object`, `created` and `model` on every chunk, the usage chunk included; the event builders became its methods, their shapes otherwise unchanged | glm.rs `reasoning_after_a_tool_call_streams_under_the_same_field`; tests/acceptance.rs `reasoning_is_reasoning_content_whole_and_streamed`, `every_chunk_carries_the_completion_head`, `api_contract_harness_passes_against_the_glm_script` | 2026-09-28 |
| src/lib.rs | Crate doc: the reasoning field, the chunk head, and the thinking switch's sources in precedence order | — | 2026-09-28 |
| src/dialect/glm.rs (tests) | The server stub records `tokenize_prompt`'s options as well as `render_prompt`'s. Streamed reasoning is read from `reasoning_content` and a `reasoning` key fails the test. New: `the_thinking_switch_forms_and_their_precedence_reach_the_engine` (17 request forms), `reasoning_after_a_tool_call_streams_under_the_same_field`; `thinking_off_reaches_the_engine_and_the_parse` also streams | the tests themselves | 2026-09-28 |
| tests/acceptance.rs | `streaming_think_block_is_reasoning_only_and_matches_non_stream` and `reasoning_opened_by_the_prompt_streams_as_reasoning` read streamed reasoning from `reasoning_content` and fail on a `reasoning` key | the tests themselves | 2026-09-28 |

## Changed here (tool calls that do not parse, the text before a call, the `thinking` spelling of the switch)

A call the GLM dialect could not parse left nothing at the client (an empty `stop` turn), a nameless
one failed the request (a 400, or `finish_reason` "error" streamed), its reports were never read, the
text before a call reached a streaming client but not a whole reply, one malformed shape the model
writes ("markup after the name": a stray closing tag between the name and the first argument) was
lost, and `chat_template_kwargs.thinking` was ignored. Found by reading the API against the
real-output parser cases and the chat template of a public recipe (mmastrac/glm-5.3-flash-4x-gx10
@ `5ea4121`: `dev/patch-tests/_glm47_failclosed_stream_test.py`, sha256
`c39fee31c28b47ed91ef3ad3269a9d5498ae4fb7aea39194b524ae2548db7594`, and `image/chat-template.jinja`
line 3, sha256 `f02c2c536ac51deeb2675125064f56578d4ca729389511a782a8ccdc6721e95e`). The cases are
reimplemented as shapes (own tool names and values, no code or text copied); the recipe's parser
refuses such calls with a sentinel argument, this one returns their text as `content` and refuses
nothing. Each of the first three items below is a separate hunk.

| File | Change | Pinned by | Date |
|---|---|---|---|
| src/dialect/glm.rs | **Lost calls kept.** A lost call's text, opening tag included, stays in `content` where it stood; "markup after the name" is recovered, and reported, when what is left of the name (`without_closing_tags`) is a tool the request offered; module doc | glm.rs `a_lost_calls_text_stays_in_the_content`, `markup_after_the_name_is_recovered_only_as_an_offered_tool` | 2026-09-29 |
| src/dialect/glm.rs | **Nameless calls lost.** A call with arguments but no name is a lost call (was `error`, so a 400 or `finish_reason` "error"); `Call::Nameless` removed. The MiMo dialect keeps its error | glm.rs `malformed_calls_are_reported_not_dropped`, `a_lost_calls_text_stays_in_the_content`; tests/acceptance.rs cases "arguments without a name" | 2026-09-29 |
| src/dialect/glm.rs, src/chat.rs, src/dialect/mod.rs | **The text before a call.** `parse` keeps text only until the first call has parsed (`keep_text`); `reply_content` is the content of a reply, whole and streamed: as parsed, but without the whitespace that ends it when there are calls; `message_content` returns it (null when empty beside calls). The stream holds back whitespace that ends the text so far (a call that follows drops it) and, after generation, sends the rest of `reply_content` beyond what it sent live (`StreamSplit::content_bytes`): a lost call's text | tests/acceptance.rs `glm_tool_calls_as_the_model_writes_them` (cases "text before a call", "whitespace before a call", "text between and after calls", "a plain reply", and the lost calls), `a_think_block_inside_the_text_before_a_call_leaves_whole_and_streamed_alike`, `streaming_multibyte_text_does_not_panic_in_holdback` (now reads the joined content: the space ending the first fragment goes out with the next text) | 2026-09-29 |
| src/chat.rs | `log_reports`: every parse report to stderr under the completion's id, whole and streamed | tests/acceptance.rs `parse_reports_are_logged_under_the_completion_id` | 2026-09-29 |
| src/types.rs | `chat_template_kwargs.thinking` (a boolean) is an alias of `enable_thinking`: read after it and before the top-level one | glm.rs `request_fields_map_to_the_template_switches`, `the_thinking_switch_forms_and_their_precedence_reach_the_engine`; tests/acceptance.rs `glm_thinking_off_is_low_effort` | 2026-09-29 |
| src/lib.rs | Crate doc: the switch's sources, tool calls | — | 2026-09-29 |
| tests/acceptance.rs | `glm_tool_calls_as_the_model_writes_them` (the recipe's six shapes, lost calls, the text around calls: each whole and streamed a character and a token at a time), the think-block test, `parse_reports_are_logged_under_the_completion_id` (the server logs from its own threads, so a child process serves the requests and the test reads its stderr) | the tests themselves | 2026-09-29 |
| harness/test_api_contract.py | The fake server reads `chat_template_kwargs.thinking` as the API does | the self-test | 2026-09-29 |

## Changed here (the lowest effort's names)

`reasoning_effort: "none"` was rendered as an empty think block under the template's Max effort. It
and `"minimal"`, the names of the lowest effort, are now thinking off: the template's Low effort with
the think block open, so the API renders only the efforts the template has (`docs/DESIGN.md`,
"Thinking switch and reasoning history"). Own logic; nothing is copied.

| File | Change | Pinned by | Date |
|---|---|---|---|
| src/types.rs | `lowest_effort` ("none", "minimal"); it is the last link of the thinking switch's chain (was `reasoning_effort: "none"` alone) | glm.rs `request_fields_map_to_the_template_switches` | 2026-09-29 |
| src/chat.rs | A dialect with a lowest effort (`thinking_off_effort`) renders thinking on at that effort for a request that turns thinking off or names the lowest effort, whatever the switch says; the `effort_none` case (thinking off, the empty block) is removed | glm.rs `reasoning_effort_none_and_minimal_reach_the_engine_as_the_low_effort`, `the_thinking_switch_forms_and_their_precedence_reach_the_engine`; tests/acceptance.rs `glm_thinking_off_is_low_effort`, `glm_lowest_effort_names_render_the_low_prompt`; glm53f-coordinator tests/glm_prompt.rs `effort_requests_render_the_templates_efforts` (the real template) | 2026-09-29 |
| src/engine.rs, src/dialect/mod.rs, src/dialect/glm.rs, src/lib.rs | Docs: `PromptOptions::reasoning_effort`, `Dialect::thinking_off_effort`, the GLM dialect, the crate doc | — | 2026-09-29 |
| harness/api_contract.py, harness/test_api_contract.py | THINK-OFF: `reasoning_effort=none` is judged as the other off forms are (a short reasoning under the reasoning field is allowed), `=minimal` is a new off form, and the streamed check reads the field name and markup, not "no reasoning"; the fake server maps both names to Low (new broken mode `effort-minimal-ignored`) | the self-test | 2026-09-29 |

## Changed here (`GET /health`)

A production health check and a test ladder probe `GET /health`; the source served no such route (a
request for it was a 404). The engine answers from its state, so the route never waits behind the
request queue or the model. Own code; nothing is copied.

| File | Change | Pinned by | Date |
|---|---|---|---|
| src/engine.rs | New `Engine::health`, defaulting to `Ok(())`, so existing engines are unchanged | tests/acceptance.rs (health tests) | 2026-09-29 |
| src/lib.rs | The router serves `GET /health` (`health::handle`); new `serve_listener`, `serve` on an already-bound listener, which `serve` calls; crate doc | tests/acceptance.rs `health_is_ok_while_the_engine_can_serve_and_takes_no_queue_place`, `health_is_503_with_the_reason_once_the_engine_cannot_serve` (both through `serve_listener`) | 2026-09-29 |
| src/http.rs | The reason phrase of 503 (`Service Unavailable`) | the 503 test | 2026-09-29 |
| Cargo.toml | The header comment lists the route | — | 2026-09-29 |
| tests/acceptance.rs | `start_served` (the crate's own routes on a loopback port), `raw_get`, the `WireDown` stub and the two health tests | the tests themselves | 2026-09-29 |

## Changed here (an optional API key)

The API served every request that reached it, and the engine has no accounts, so anything on the
network could use it directly. `glm53f-serve --api-key-file` now gives it one key, the file's first
line; every `/v1/*` request must then carry it as `Authorization: Bearer <key>`, and `GET /health`
stays open. Own code; nothing is copied.

| File | Change | Pinned by | Date |
|---|---|---|---|
| src/lib.rs | New `serve_with_key` and `serve_listener_with_key`, which check the key before routing (a refusal reaches no handler); `serve` and `serve_listener` call them without a key, so their behaviour is unchanged; `ApiKey` re-exported; crate doc | tests/acceptance.rs `a_keyed_api_refuses_v1_requests_without_the_key`, `a_keyed_api_leaves_health_open`, `an_api_without_a_key_is_open` | 2026-09-29 |
| src/types.rs | New `ApiError::unauthorized` (401, code `invalid_api_key`); `ApiError::body` gives a 401 the type `invalid_request_error`, as OpenAI answers a bad key (the other errors keep their code as their type) | auth.rs unit tests; the acceptance tests above | 2026-09-29 |
| src/http.rs | The reason phrase of 401 (`Unauthorized`) and its `WWW-Authenticate: Bearer` header (RFC 9110, 11.6.1) | the acceptance tests above | 2026-09-29 |
| Cargo.toml | The header comment names the key | — | 2026-09-29 |
| tests/acceptance.rs | `start_keyed` (`start_served` calls it without a key), `raw_request`, and the three tests above | the tests themselves | 2026-09-29 |

## Changed here (a keepalive while a tool call is held back)

The engine sends an empty delta after a wait of 15 s for a token (a long prefill), which the API
writes as the SSE comment `: keepalive`. A tool call is held back until it is complete, so while the
model writes one, tokens arrive, nothing is written, and the engine has nothing to send: measured on
the target hardware, a 3,043-token `write` call left 25.1 s of silence, which proxies and clients with
a shorter idle timeout cut. The stream now writes the same comment whenever it has written nothing for
the engine's keepalive interval. Own code; nothing is copied.

| File | Change | Pinned by | Date |
|---|---|---|---|
| src/chat.rs | `StreamSplit` keeps the time of its last write (`send`, which every chunk it writes goes through) and `keepalive_if_idle` writes the comment once that is `Engine::keepalive` ago, after each delta; the engine's empty delta goes through `keepalive` too. Delta content and order are unchanged | tests/acceptance.rs `a_held_back_tool_call_sends_keepalives_and_the_same_reply`, `output_that_flows_needs_no_keepalive`, `an_engines_empty_delta_is_a_keepalive_comment` | 2026-09-29 |
| src/engine.rs | New `Engine::keepalive`, defaulting to 15 s, so existing engines are unchanged; `generate`'s doc says an empty delta is a keepalive | the acceptance tests above; glm53f-coordinator tests/engine.rs `the_keepalive_is_the_engines_configuration` | 2026-09-29 |
| src/lib.rs | Crate doc: the keepalive | — | 2026-09-29 |
| tests/acceptance.rs | `SlowGlm` (GLM-5.3-Flash writing slowly, on an engine with a keepalive interval), `keepalives`, `events_without_id` and the three tests above | the tests themselves | 2026-09-29 |
