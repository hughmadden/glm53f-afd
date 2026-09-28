# glm53f-tokenizer provenance

GLM-5.3-Flash's `tokenizer.json` and `chat_template.jinja` in Rust. One unit is adapted from
mimo26f-afd (MIT); the rest is written here against the reference `tokenizers` and
`transformers`, and checked against goldens those libraries recorded (below). No external crates.

## Adapted

| Unit | Source (repo @ commit : path) | sha256 (source file) | Here | Delta | Pinned by | Date |
|---|---|---|---|---|---|---|
| Byte-level BPE tokenizer: GPT-2 byte map, added-token split, pre-tokenize then BPE per piece, byte decode | [hughmadden/mimo26f-afd](https://github.com/hughmadden/mimo26f-afd) @ `bab9fa2f2fc1e22ae67b56fbc1c209278f6a9d79` (v1.2.0) : `crates/mimo26-coordinator/src/tokenizer.rs` | `0be0de186b5b894c41cf2466bf1ead9f90d1125935e5f04800ad27c278923e8f` | `src/tokenizer.rs`, `src/pretok.rs`, `src/bpe.rs` | Rewritten for GLM-5.3-Flash and made exact. **Pre-tokenizer:** GLM's regex (cl100k form: digits in runs of one to three) in place of MiMo's; the character classes come from tables probed from the reference (`src/unicode.rs`) instead of `char::is_alphabetic` / `is_numeric` / `is_whitespace`, which differ from the regex's `\p{L}` and `\p{N}` outside ASCII; the case-insensitive contraction also matches U+017F for `s`, as the reference does. **BPE:** the reference's merge order (a heap keyed by rank then position, stale entries skipped) in place of repeated lowest-rank scans, O(n log n) instead of quadratic in a piece's length; `ignore_merges`. **Added tokens:** longest match by first byte. **Decode:** a token with a character outside the byte map decodes as its own UTF-8 bytes, whole, as the reference's decoder does; `skip_special` is a parameter (MiMo always skipped special tokens). **Loading:** reads with this crate's JSON reader; validates the whole pipeline in tokenizer.json and refuses anything not implemented (normalizer, other pre-tokenizers, dropout, unknown token, subword affixes, byte fallback, space-stripping added tokens). MiMo's NFC scope note is gone: GLM's tokenizer has no normalizer | `tests/goldens.rs` (encode, decode, pre-tokenizer pieces); unit tests | 2026-09-28 |

## Written here

| File | What | Pinned by |
|---|---|---|
| `src/json.rs` | JSON reader that keeps what Python's `json.loads` keeps (key order, exact integers, floats, first-position/last-value duplicate keys, `NaN`/`Infinity`); `json.dumps(ensure_ascii=False)` writer with Python's float `repr` | unit tests; the template goldens |
| `src/unicode.rs`, `src/unicode_tables.rs` | The regex's `\p{L}`, `\p{N}` and `\s`. The tables are generated, not hand-written: `examples/gen_unicode_tables.rs` reads `oracle/goldens/tokenizer/unicode_classes.json` | `tests/goldens.rs` (`character_tables_match_the_reference_probe`); unit tests |
| `src/pretok.rs` | The regex's seven alternatives, leftmost-first, as direct scans | `tests/goldens.rs` (pieces of all 120 cases); unit tests |
| `src/bpe.rs` | The merge loop of `tokenizers`' `Word::merge_all`, step for step | goldens; unit tests |
| `src/stream.rs` | Streaming decode: holds back only a valid prefix of a split character; deltas concatenate to `from_utf8_lossy` of the whole | unit test against `from_utf8_lossy` (5,000 random splits); `tests/goldens.rs` (streamed decode of all 420 sequences) |
| `src/template.rs` | The chat template, rendered by hand; thinking off as documented in the module; `ToolCall::from_openai` (wire-form arguments); `check_template`, a chat-template drift guard (the length and FNV-1a-64 digest of the template it reproduces) | `tests/goldens.rs` (48 reference renders, 46 thinking-off prompts, token ids, the checkpoint's template against the guard); unit tests |
| `examples/gen_unicode_tables.rs` | Table generator (compiles `src/json.rs` in by path) | the table test |
| `tests/goldens.rs` | Golden harness; tests needing tokenizer.json read `GLM53F_TOKENIZER` and skip while unset | - |
| `tests/hygiene.rs` | Every source file here and both oracle scripts are ASCII and spell no complete think or tool-call tag (a decoded escape would fail it) | - |
| `oracle/tokenizer_goldens.py`, `oracle/template_goldens.py` | The golden recorders, run inside the oracle image | - |

The regenerate commands:

```text
cargo run -p glm53f-tokenizer --example gen_unicode_tables -- \
    oracle/goldens/tokenizer/unicode_classes.json > crates/glm53f-tokenizer/src/unicode_tables.rs
```

## References (read, no code copied)

| Reference | sha256 | What it fixed |
|---|---|---|
| zai-org/GLM-5.3-Flash `tokenizer.json` | `19e773648cb4e65de8660ea6365e10acca112d42a854923df93db4a6f333a82d` | The pipeline: no normalizer; Split (regex, Isolated) then ByteLevel (no prefix space, no regex); BPE with `ignore_merges`; 154,820 tokens, 321,649 merges, 36 added tokens (18 special) |
| zai-org/GLM-5.3-Flash `tokenizer_config.json` | `98b1271574f41abf89427ae2dda030d94dc9478f0edc5a8bd240db213c6fd5fc` | No cleanup of tokenization spaces |
| zai-org/GLM-5.3-Flash `chat_template.jinja` | `0c4099f3382d6c92700dfb99725025360966fd73032f0ecf32377c0d9e6309c5` | The renderer |
| zai-org/GLM-5.3-Flash `generation_config.json` | - | Stop ids 154,820, 154,827, 154,829 (`STOP_IDS`) |
| huggingface/tokenizers 0.23.2 (in the oracle image) | - | Semantics checked by the goldens: leftmost-longest added-token split, `ignore_merges`, `Word::merge_all`, the ByteLevel decoder, `decode` skipping special tokens |
| huggingface/transformers 5.17.0 (in the oracle image) | - | `apply_chat_template`'s Jinja environment: `trim_blocks`, `lstrip_blocks`, and `tojson` as `json.dumps(ensure_ascii=False, sort_keys=False)` |
| [tpurtell/glmrt-5.3-1rtx-4spark](https://github.com/tpurtell/glmrt-5.3-1rtx-4spark) @ `dc6d9b8` : `rust/crates/glmrt-api/src/request.rs` | `aeb248c87774bfab796c74fb87ae6a7a849e7fd8991e2906c65511b1437e3eb0` | Diffed against the Flash template (below) |
| same @ `dc6d9b8` : `rust/crates/glmrt-api/src/tooling.rs` | `cb657995662cbca79bbb2515f822de2d8d3e14ec11a7987d185a61db38eed93f` | Tool-call format and schema-typed argument values (used by the GLM dialect in `glm53f-api`) |

**glmrt's GLM-5.3 prompt against the GLM-5.3-Flash template.** glmrt renders full GLM-5.3 by
hand. What carries over: the `[gMASK]<sop>` prefix, the tools block text, the tool-call format and
image markup. What differs in Flash, all taken from the Flash template and its goldens:
- The effort line is always written, as Low, High or Max. glmrt writes it only with thinking on,
  and only as High or Max.
- An assistant turn's content is stripped. Reasoning written inline in the content moves into
  the think block, and the text after the last closing tag stays content. `clear_thinking` is
  honoured. glmrt does not strip, drops inline reasoning (keeping the text after the first
  closing tag), and has no `clear_thinking`.
- Tool results are ordered by the preceding calls' ids when the ids allow it. glmrt keeps
  message order. Both group a run of results under one observation token.
- Argument values and tool schemas are written with Python's separators. glmrt uses compact
  serde_json.
- Unknown roles render nothing; glmrt writes the role name.

glmrt's thinking-off prompt (an empty think block after the assistant token) is the form this crate
uses for thinking off.

## Goldens

Recorded by the two oracle scripts in the oracle image (`tokenizers` 0.23.2, `transformers`
5.17.0, Python 3.12.3); each directory's `manifest.json` has the versions and file digests. The
files are ASCII JSON with angle brackets escaped (as the MiMo parser goldens are), so none spells
markup. Rerunning both scripts reproduces them byte for byte.
- `oracle/goldens/tokenizer/`:
  - 120 texts (`cases.json`), with ids, pre-tokenizer pieces and both decodes (`ids.json`);
  - 300 random id sequences with both decodes (`decode.json`);
  - the regex's classes probed over all 1,112,064 scalar values (`unicode_classes.json`). They
    are Unicode 16.0's, one version ahead of Python 3.12's `unicodedata` (15.0), which the file
    records for comparison.
- `oracle/goldens/template/`: 48 conversations with the reference render and ids, and for 46 of
  them the render of a reasoning-free answer turn (the thinking-off check).
