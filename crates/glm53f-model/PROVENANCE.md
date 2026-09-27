# glm53f-model: provenance

Every unit this crate copied, adapted or reimplemented from another project,
and every data file it carries. Rows are consolidated into `docs/REUSE.md`.

## Code

| Unit | Source (repo @ commit : path) | sha256 (source file) | Here | Delta | Pinned by |
|---|---|---|---|---|---|
| JSON codec | [hughmadden/mimo26f-afd](https://github.com/hughmadden/mimo26f-afd) @ `bab9fa2f2fc1e22ae67b56fbc1c209278f6a9d79` (v1.2.0) : `crates/mimo26-api/src/json.rs` (MIT) | `52c52df6c298bf29877e26d976b1e9227c8e58abc45fe2601670558fcd116aa5` | `src/json.rs` | Integer literals parse to a new `Json::Int(i64)`, so offsets and dimensions are exact and a float where an integer is required is an error; `as_u64`, `as_i64`; `as_f64` accepts both; new `check_unique_keys`; `Int` serialization. Parser, string/escape/UTF-8 handling and serializer otherwise unchanged; comments that cite the source project's defect list dropped. The source's six tests kept (two adjusted for `Int`), three added. | `json::tests` (9) |
| Safetensors reader | hughmadden/mimo26f-afd @ `bab9fa2` (v1.2.0) : `crates/mimo26-repack/src/safetensors.rs` (MIT) | `a9fe0ca79b1c59e2c35ef43f477883c33894cf2de2f9383c8c101fc3b2d40d87` | `src/safetensors.rs` | Kept: the header-length and fit checks, reading a tensor by its offsets, strict failure on anything unexpected. Changed: every dtype (the source reads `U8` only); header parsed with `src/json.rs` instead of the source's private parser; each entry's byte length must equal shape x dtype size; the data region must be covered without holes or overlaps and end at the file's end; unknown entry fields rejected; `__metadata__` kept. Added: multi-shard `Checkpoint` with a `model.safetensors.index.json` cross-check, header bundles, positional (`pread`) reads, strided byte runs (`Runs`) for tensor-parallel slices, `serialize` for fixtures. | `safetensors::tests`, `tests/checkpoint.rs`, `tests/slicing.rs` |
| EXL3 TP split rule (reimplemented; no code copied) | [tpurtell/glmrt-5.3-1rtx-4spark](https://github.com/tpurtell/glmrt-5.3-1rtx-4spark) @ `dc6d9b8e1600e001cb1d4228bd911f4df8091f99` (v9) : `rust/crates/glmrt-loader/src/exl3_format.rs` (MIT); [ashhart/TensorFold](https://github.com/ashhart/TensorFold) @ `bb4b4a35863af562fc4ccb2586300d8f94b5d6de` (v0.3.4.1) : `src/tensorfold/families/glm5_next/cuda/split.py`, `.../cuda/exl3.py` (MIT) | `e6072978cb63dd808c5de65dfeb0e72db1dff7733c593dc65fb1ec8b4e6f0198`; `f8d876285405b05f886bf7a7732d102054e649a35d2707d3bc2eaa54ad3df7f0`; `acdf6f0be5a2af09a92c905e9ba826f3083f752bbd7a7e42aeafc78ded69e077` | `src/slicing.rs` (`split_rule`, `granularity`) | Read for the rule, then written from the format's definition (trellis `[in/16, out/16, 16*bits]`, `suh [in]`, `svh [out]`, `mcg`, block-diagonal 128-point Hadamard). glmrt: per-rank residency keeps hidden-side rotations whole and slices intermediate-side rotations, requiring `I % (4 x 128) == 0` (TP4, full GLM-5.3). TensorFold: gate/up split by tile columns and `svh`, down by tile rows and `suh`, the rest replicated (TP2, GLM-5.3-Flash). Here: the same rule at any world size whose share is a multiple of 128 channels, plus the FP8 and NVFP4 rules. | `tests/slicing.rs` (`exl3_output_split_is_exact`, `exl3_input_split_is_exact`, `exl3_split_inside_a_hadamard_block_is_wrong_and_refused`), `tests/headers.rs` |
| DFlash2 tensor list (reimplemented) | the published `incoai/GLM-5.3-Flash-DFlash2` @ `bf582e4e` checkpoint header; TensorFold @ `bb4b4a3` : `src/tensorfold/families/glm5_next/cuda/dflash2.py` (MIT), for the meaning of `base_kernel [2, 2, hidden]` (two branches x two taps) | `0e408f910e611af8c927a6a7c0f912ed59c304ecd845cc3ba84b8db1838e5684` (dflash2.py) | `src/config.rs` (`DraftConfig::tensors`, `DFLASH2_CONV_BRANCHES`) | Names and shapes derived from the drafter config; the branch count (2) is not a config field. | `tests/config.rs` (total 2,342,160,896 B), `tests/checkpoint.rs` (against the real header when `GLM53F_TEST_DRAFTER_DIR` is set) |

Everything else in `src/` (typed config and invariants, tensor catalog and
name classifier, planner) is new, written from `transformers`
`models/glm5_next` (the reference modeling code), the published configs and
headers, and `docs/SIZING.md`.

## Test data

Verbatim copies of published configs (`tests/data/`). The model config is kept
whole (not trimmed): `tests/config.rs` checks the catalog's FP8 policy against
its full `modules_to_not_convert` list.

| File | Source | sha256 |
|---|---|---|
| `zai-org_GLM-5.3-Flash.config.json` | `zai-org/GLM-5.3-Flash` @ `eb9eb208eb0d988989d07a6a12d0fdeb5f52574a` : `config.json` | `bb8f01c42cb92a52ca72e65afb4d5bd8d11aef083cd210e8de25dfb904f23e9f` |
| `incoai_GLM-5.3-Flash-DFlash2.config.json` | `incoai/GLM-5.3-Flash-DFlash2` @ `bf582e4eacc1810f76656d1811693ff6c6737d2a` : `config.json` | `c4aeac0101196a6e26705b34c45230bcd0c7c68ee2d2d1efdb242087f3712573` |
| `brandonmusic_GLM-5.3-Flash-tr3-4bpw.config.json` | `brandonmusic/GLM-5.3-Flash-tr3-4bpw` : `config.json` (revision not recorded) | `4f5341e048984459471bfb9c894e6bf87e69b9c67402672af901631d1349f265` |
| `LibertAIDAI_GLM-5.3-Flash-NVFP4.config.json` | `LibertAIDAI/GLM-5.3-Flash-NVFP4` : `config.json` (revision not recorded) | `5db46f44956e4a8a0cc8ed54b6d77bf99dd7c1ec90c58975d1952560768513d5` |

The expected tensor counts and bytes per group in `tests/common/mod.rs` were
derived from the three checkpoints' full safetensors headers, which are not
committed. `tests/headers.rs` recomputes them from the headers when
`GLM53F_TEST_HEADERS_DIR` is set.
