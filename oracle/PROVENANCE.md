# Oracle provenance

What the reference oracle takes from other projects, and what it changes. Nothing below is vendored
into this repository: the reference code runs from the pinned wheel inside the oracle image.

## Reference implementation

| Unit | Source (repo @ version : path) | sha256 | Used as | Changed |
|---|---|---|---|---|
| `glm5_next` modeling code: `Glm5NextTextDecoderLayer`, `Glm5NextTextLinearAttention` (KDA), `Glm5NextTextAttention` + `Glm5NextTextIndexer` (DSA), `Glm5NextTextHyperConnection` / `Glm5NextTextHyperHead` (mHC), `Glm5NextTextMLP`, `Glm5NextTextMoE`, `Glm5NextTextTopkRouter`, `Glm5NextTextExperts`, `Glm5NextTextRMSNorm` | huggingface/transformers @ 5.17.0 (PyPI wheel `transformers-5.17.0-py3-none-any.whl`, sha256 `78ec1ce21579b38dfb83950a0658cd119f87212a2fcfdff478096ce9d6c03801`) : `transformers/models/glm5_next/modeling_glm5_next.py` | `5a885692edc74056f370d70af10ba746c0b12d59d245fb782c8ec8964059ea6e` | imported and run unmodified | nothing; forward hooks and call wrappers record intermediates |
| `Glm5NextConfig` / `Glm5NextTextConfig` | same wheel : `transformers/models/glm5_next/configuration_glm5_next.py` | `f12b5876701d000f930cd8102124e07d05675a79a5591db29c18244f5f025dec` | reads the checkpoint's `config.json` | attention and expert implementations set to `eager` |
| `DynamicCache` (KDA conv and recurrent state, MLA and indexer caches) | same wheel : `transformers/cache_utils.py` | `702144bb44553f6339ea1bf23c8205a708bb5f8c7c09cb3a2db484182646743c` | the reference cache, unmodified | nothing |
| `Fp8Dequantize._dequantize_one` | same wheel : `transformers/integrations/finegrained_fp8.py` | `00441402ce986ef453363f93df52ce57ad5fe4eda95f877078faecd6e8382201` | called directly to dequantise FP8 E4M3 weights with their 128 x 128 `weight_scale_inv` blocks | nothing |
| Checkpoint-to-module name mapping for `glm5_next` (renames of `f_a_proj`, `f_b_proj`, `dt_bias`, `A_log`, `hc_{attn,ffn}_{fn,base,scale}`; q/k/v short-conv concatenation; per-expert stacking into `gate_up_proj` and `down_proj`) | same wheel : `transformers/conversion_mapping.py` (the `"glm5_next"` entry) | `38c364608a8cdb43eb2177e645c3de47cd2197c8e42d09b5848541b6c90e0d71` | **reimplemented** in `golden_layers.py` (`RENAMES`, `build_layer`, `ExpertBank`) | the stacked expert tensors are materialised one expert at a time on first access instead of at load; `selftest_tiny.py` checks every parameter, experts included, against `from_pretrained` bit for bit |

### Why 5.17.0

- `transformers` 5.16.0 on PyPI has no `models/glm5_next`, although the checkpoint's `config.json`
  records `transformers_version` 5.16.0. 5.16.1 and 5.17.0 have it.
- 5.16.1's chunked KDA builds its decay mask with `exp()` before masking the upper triangle; 5.17.0
  masks first. The masked entries are discarded either way, so the values agree, but 5.17.0 is the
  later and cleaner code.
- Against the upstream main branch at commit `7cd73d9df0` (the code the design documents cite,
  `modeling_glm5_next.py` sha256 `4fe6ed7703e4f8f1dc7e3995af2b619058be8fec1f5157b7f512fc6f6150503f`),
  5.17.0 differs in ways that do not change the goldens:
  - the MLA cache: 5.17.0 caches the expanded keys and values, main caches the 512-dim latent and
    expands it on every step. The expansion is the same `kv_b_proj` on the same latent;
  - the expert loop: main sizes its one-hot mask for an extra sentinel expert id (for expert
    parallelism). No sentinel occurs here;
  - the DSA layer-type name: `deepseek_sparse_attention` in 5.17.0 (which the checkpoint's
    `config.json` uses) versus `indexed_attention` in main;
  - the mHC forward is refactored (same operations in the same order), plus docstrings.

## Model

- Weights: [zai-org/GLM-5.3-Flash](https://huggingface.co/zai-org/GLM-5.3-Flash) (MIT licence), official
  FP8 checkpoint. The revision used is recorded in every manifest (`source.model`), and the digest of
  every tensor read is in `goldens/source-tensors.tsv`.
- The goldens are activations computed from those weights; no weights are stored in this repository.

## Written for this repository

- `golden_layers.py`: layer-by-layer construction and weight loading, recording hooks, the prompt and
  decode driver, the fixture writer, `verify` and `compare`.
- `selftest_tiny.py`: a tiny random checkpoint in the official on-disk format (its FP8 blockwise
  quantiser follows the checkpoint's layout: E4M3 values, F32 `weight_scale_inv` per 128 x 128 block)
  and the bit-for-bit comparison against `from_pretrained` and the whole-model forward.
- `Dockerfile`, `requirements.lock`: the image recipe and hash-pinned wheels (digests checked against
  PyPI's JSON API).
