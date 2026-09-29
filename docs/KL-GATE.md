# KL gate

**Status (28 September 2026): the gate has run on the target hardware, and both engine paths
pass** (section 6a). The decode path scores 0.0245 nats against the BF16 teacher, equal within its
standard error to the published figure for the same 4-bit experts. On 29 September the numerics
options were gated (section 6b): BF16 KDA states (D8) passed and are now the default; FP8 KDA
projections (D2) failed. A 125-window panel for paired comparisons (section 6c) had its first
result the same day (section 6d).

The gate measures how far the engine's next-token distributions are from the BF16 model's, on the
public panel that the published GLM-5.3-Flash quantization figures were measured on, with the
same method. It answers two questions:

1. **Absolute:** is the engine's configuration (official FP8 non-expert weights, EXL3 4-bit
   experts, FP8 MLA latent and index keys) in the regime of the published configurations?
2. **Relative:** does a numerics change (FP8 KDA projections, FP8 wire rows, W8A8 prefill GEMMs,
   the chunked KDA prefill) make it measurably worse? A paired comparison on the same positions.

Files: [`harness/klgate.py`](../harness/klgate.py) (the gate; `klgate.py selftest`),
[`harness/klgate_fetch.py`](../harness/klgate_fetch.py) (the teacher subset; its test
[`harness/test_klgate_fetch.py`](../harness/test_klgate_fetch.py) runs against a dataset in memory),
[`harness/PROVENANCE-klgate.md`](../harness/PROVENANCE-klgate.md) (sources). The tools use only the Python
standard library (tested with Python 3.12). The engine's logits come from
[`crates/glm53f-score`](../crates/glm53f-score) and `GlmForward::score` (section 4).

## 1. The teacher panel

[`brandonmusic/GLM-5.3-Flash-BF16-Teacher-Logits`](https://huggingface.co/datasets/brandonmusic/GLM-5.3-Flash-BF16-Teacher-Logits),
pinned at revision **`95f4fdd94bf29989db2e0d1054e4931f55edb6aa`** (27 August 2026). The panel's
identity: `dataset-manifest.json` sha256 `1c6cba530a60af71ff62e5c2180f2edd26090d8d5845747c85f2e0b53aafc736`,
its `dataset_sha256` field `61faf80c9a8c7bb60bcefbfd6208c7f63609ddc4089798c2f317bfcabc8569a4`.

**Teacher.** `zai-org/GLM-5.3-Flash-BF16` at `a6c167b62691b2bac901344b65cb651a70f53e43` (the
released BF16 weights, F32 tensors kept), `transformers` 5.16.1, torch 2.11.0+cu130, eager
attention, no KV cache (`use_cache: false`), TF32 off, expert-parallel over four B200s
(`backend.json`, `plan.json`).

**What the qualification panel holds** (the only part the gate uses):

| Path | Files | Size | Content |
|---|---:|---:|---|
| `logits/window-0000.safetensors` … `window-0024.safetensors` | 25 | 1,268,157,840 B each, 31.70 GB | safetensors: a 392-byte JSON header, then one tensor `logits`, F32, shape [2047, 154880], row-major. Row r = the teacher's logits after tokens[0..r], predicting tokens[r+1]. Header metadata: `window_id`, `model_revision`, `token_ids_sha256`, `attention_mask_sha256`, `capture_role`. |
| `calibration/panel-v1/arrays/final-0000.tokens.npy` … `final-0024` | 25 | 8,320 B each | `.npy` v1, int32, shape (2048,): the token ids. Raw packed text: no BOS, no chat template. |
| `calibration/panel-v1/arrays/causal-mask-2048.npy` | 1 | 2,176 B | uint8 (2048,), all ones (plain causal attention over the whole window). |
| `dataset-manifest.json` | 1 | 12,431 B | per window: path, bytes, sha256, `token_ids_sha256` (the sha256 of the `.npy` file), document, domain, role `final`, 2,047 positions; `vocab_size` 154880, `logits_dtype` float32. |
| `capture-receipt.json`, `backend.json`, `plan.json`, `token-panel-receipt.json`, `README.md` | 5 | 143 KB | the capture's identity; `kld_direction: teacher_to_student`. |
| `calibration/panel-v1/{panel,panel.receipt,tokenizer.receipt,corpus.receipt}.json` | 4 | 413 KB | window metadata (`teacher_row_start` 0, `teacher_row_end` 2047: rows 0 to 2046) and receipts; `tokenizer.receipt.json`: tokenizer vocabulary 154,856 (`maximum_token_id_exclusive`). |

- **Full logits, not top-k:** every one of the 154,880 LM-head columns, in FP32, at every
  next-token position. No log-softmax is applied.
- **Vocabulary and padding:** the tokenizer's ids end at 154,856; the last 24 stored columns are
  the LM head's padding rows. In the 4,725 fetched teacher rows, a row's 24 padded values agree
  to within 0.031 (between −4.7 and +3.3 across rows, against a median row maximum of 21.8),
  never hold the row's maximum, and carry at most **2.1e-5** of the teacher's probability.
- **Sequences:** 25 windows of 2,048 tokens, **51,175 scored positions** (2,047 per window). Four
  domains from four packed source documents: general 7 windows, legal 6, code/agentic 6,
  reasoning/termination 6. Role `final` marks them qualification-only (never used to calibrate);
  a third-party scan found shared text between 8 of them and the calibration windows, hence the
  "clean17" scope (section 6).
- **The rest of the repository:** `logits/full-panel/` (640 more windows in four other roles,
  811.6 GB; section 6c takes 100 of them for paired comparisons), and, not used,
  `calibration/main-ep4-full/` (464.4 GB of hidden states and router choices),
  `calibration/mtp45-ep4-full/` (10.8 GB), `source-inventory.json` (13.4 MB). 1,517 files,
  1.32 TB in all.

## 2. The published method

| Source | Pinned at | Used for |
|---|---|---|
| `brandonmusic/GLM-5.3-Flash-tr3-4bpw` model card, `README.md` | `a5fee929cf4888b1824323e33e8a19b60129e025`, sha256 `6701140b…6543e` | the K4 figures, their scope and regime |
| same repo, `scripts/measure_glm53_packed_student_kld.py`, `scripts/measure_glm53_tp_runtime_window_kld.py` | same, `3948b677…` and `460062c3…` | the kernel of the headline figures |
| same repo, `results/five-cold-run-kld.json`, `results/run-1-kld-report.json` | same, `d955bfae…` and `bc5c1e24…` | K4, 25 windows: mean, per window, per domain, top-1 |
| same repo, `runtime-results/v75/kld/{fp8,nvfp4}-five-run-kld.json` | same, `409a3487…` and `416b4470…` | K4 with an FP8 or NVFP4 MLA cache, window `final-0000` |
| same repo, `eval/kld/` (`kld_eval` package, `protocol.yaml`, `results/RUN_SUMMARY.md`) | same, `752af674…` (`core.py`), `3b6547ae…` (`stats.py`), `4d1d91ad…` (`protocol.yaml`) | masked-vocabulary kernel, statistics, protocol |
| GitHub `brandonmmusic-max/glm-5.3-flash-exl3-4bpw`, `kld quantization fidelity report.md` | commit `24784d71`, sha256 `692ff9e5…` (the digest `protocol.yaml` records) | the governing report |
| `malaiwah/quant-fidelity-registry` (dataset): `data/measurements.jsonl`, `protocol/glm53-joint-kld-protocol.v1.json`, `protocol/per-window/*.json` | `394b64750b55c325899c5cd121a415cc6fc99c15`; `82c5ade5…`, `80df521e…` | FP8, K6 and floor figures; canaries; per-window means |
| `brandonmusic/GLM-5.3-Flash-tr3-4bpw`, `runtime-results-v44.json` | `a5fee929…`, sha256 `f47b8aad…` | the window-0 split: first 64 positions against the rest |
| `Mia-AiLab/GLM-5.3-Flash-EXL3-TR3-4bpw` model card | `9eaebb7c4e96d983dcd538e18624622ba5b820a8` | a byte-identical mirror of the K4 checkpoint (upstream revision `5ab363a8`); no figures of its own; its `runtime-results-v44.json` is the upstream file (same git blob) |

Full digests are in [`harness/PROVENANCE-klgate.md`](../harness/PROVENANCE-klgate.md).

**The method, as the published figures compute it:**

- **Direction:** KL(teacher ‖ candidate) = Σ_v p(v) [ln p(v) − ln q(v)], in nats, p the teacher.
  The capture receipt says `teacher_to_student`, the model card `reference_to_candidate`.
- **Positions:** teacher-forced on the fixed token ids; every next-token position of each window
  (rows 0 to 2046); row r of the candidate against row r of the teacher, never shifted. One
  window per forward from a fresh state; no prefix cache, no MTP or drafting, no sampling.
- **Precision:** log-softmax and every sum in float64 from the float32 logits.
- **Vocabulary:** the headline figures (K4 0.024555, K6 0.013723, FP8 0.020615, both floors, the
  window-0 runtime figures) softmax over **all 154,880 stored columns** on both sides (the two
  `measure_*` scripts; the registry records `vocab_masking_policy: full_stored_vocab`). The
  `kld_eval` harness and the registry's joint protocol drop the 24 padded columns on both sides
  instead. With at most 2.1e-5 of the teacher's probability on the padding, the two policies
  agree closely for an engine whose padding carries a similar share; the default (all columns)
  also penalizes an engine that puts mass there.
- **Averaging:** the **token mean** over all scored positions of all windows: 51,175 for the
  panel. Every window has 2,047 positions, so it equals the mean of the 25 window means. The
  "five-run" figures are the mean of five runs' means; the offline K4 runs were bit-identical.
- **Top-1 agreement:** the fraction of positions where the teacher's and the candidate's argmax
  agree.
- **Uncertainty:** window-level block bootstrap, B = 5,000, seed 20260829, percentile and BCa
  intervals (jackknife acceleration); the cluster-robust SE with windows as clusters. The panel
  has only four source documents, so these intervals describe resampling windows of these
  documents, nothing wider.
- **Canaries:** the teacher against itself is exactly 0 at every position; the same rows shifted
  by one reach at least 3x the teacher's mean entropy; per window, the teacher's top-1 equals the
  next token at between 0.2 and 0.995 of positions.

**Published figures on this panel:**

| Configuration | KLD (nats) | Top-1 | Scope | Lane |
|---|---:|---:|---|---|
| EXL3 K4 experts (`tr3-4bpw`) decoded to BF16, BF16 non-expert weights, `transformers` EP4 | **0.024555** | 0.9526 | 25 windows, 51,175 positions; 5 runs, bit-identical | the checkpoint author's offline decode |
| same checkpoint, custom vLLM runtime v75 (TP2/EP2/DCP2, eager), **FP8 MLA cache** | **0.024611** (runs 0.02427–0.02497) | 0.9373 | window `final-0000` only, 2,047 positions; 5 runs | the author's runtime |
| same runtime, **NVFP4 MLA cache** | **0.054757** | 0.9150 | window `final-0000` only | the author's runtime; below the author's 0.06 KLD threshold but failed the card's task-level quality gate (an earlier v44 NVFP4 cache scored 0.0605 and failed the KLD threshold) |
| official FP8 checkpoint, replayed on vLLM | **0.020615** | 0.9563 | 25 windows | third party, cross-stack |
| EXL3 K6 experts | **0.013723** | 0.9656 (streaming lane) | 25 windows | third party |
| BF16 replayed on a different stack (the floor) | 0.012712 | 0.9665 | 25 windows | third party, cross-stack |
| BF16 through the streaming harness (that lane's floor) | 0.011506 | — | 25 windows | third party |
| K4 on the author's TP4 serving runtime, padded columns dropped | 0.030480, BCa 95% [0.024965, 0.037419] | 0.9467 | 25 windows | the author's `kld_eval` |

What this means for the gate:

- The design documents' **0.0246** for "K4 with an FP8 MLA cache" is the single-window runtime
  figure. The 25-window K4 figure is the offline one with no cache quantization; both round to
  0.0246. The NVFP4-cache figure is also single-window.
- **Window matters.** On window `final-0000` the offline K4 run scores 0.0318; across the 25
  windows its window means run from 0.0062 to 0.1019. Compare only numbers on the same windows.
- **Stack floors.** The unquantized BF16 model scores 0.0115–0.0127 against this teacher when run
  by a different implementation. Part of any engine's number is its own numerics, not
  quantization.
- **The start of a window dominates.** On window `final-0000` (the author's runtime v44 receipt,
  `runtime-results-v44.json`, FP8 cache), the first 64 positions average 0.192 nats against 0.019
  for the rest: 3% of the positions carry 24% of the mean. The teacher's own entropy shows why:
  7.36 nats at position 0, 5.08, 4.25, 3.36 and 3.31 at positions 1 to 4, 2.96 at 5, 1.78 over
  6–63, 1.06 over 64–255 and 0.8 after that (the fetched rows, all 25 windows).
- **No published figure matches this engine:** official FP8 non-expert weights (W8A8 beyond 8
  rows) with K4 experts, an FP8 latent and FP8 index keys. The first run sets its baseline.

## 3. The first gate

### Rows

Downloading the whole panel is 31.7 GB. The first gate uses **4,725 teacher rows: in each of the
25 windows, positions 0–4 and 184 evenly spaced positions** (the middle of each of 184 equal
strata: 5, 16, 27, … 2041). That is 2.93 GB, fetched with range requests:

```sh
python3 harness/klgate_fetch.py --revision 95f4fdd94bf29989db2e0d1054e4931f55edb6aa \
    --rows-per-window 184 --head-rows 5 --rate 3e6 --max-bytes 3e9 --out <teacher-dir>
```

The tool verifies every whole file against the Hub listing at that revision, each window's
redirect headers (commit, sha256 and size of the linked file) and safetensors header against
the manifest, every range response and every value's finiteness, and writes `FETCH-MANIFEST.json`
(byte ranges, digests) and `SHA256SUMS`. It resumes if interrupted. Its files depend only on
the revision and the row plan; the copy made for this document has a `SHA256SUMS` (61 files)
with sha256 `8ffc57ffafa16bee16938fb6c22a7b5761706e89dd5be9c77fab854039f04a7e`.

On these rows `klgate.py canary` passes: the teacher against itself is exactly 0 at all 4,725
rows; each row against the next available row of its window averages 15.6 nats, 16x the
teacher's mean entropy; per window, the teacher's top-1 is the next token at 0.51–0.90 of rows.
Against a stand-in engine (the teacher's rows plus noise), `score`'s per-row KL matched torch's
float64 `log_softmax` computation, the published scripts' arithmetic, to 1.8e-13 nats, with the
same top-1 on all 4,725 rows.

**Estimator.** Each scored row stands for the positions nearest to it (a tie is split), so the
estimate is the panel's position mean; with every position scored it is exactly the published
token mean. The first five positions are scored exactly because the KL per position is largest
and falls fastest there: on power-law profiles matching the window-0 receipt (first 64 positions
10x the rest), evenly spaced rows alone underestimate the panel mean by 0.6–21%, the steeper the
head the more; with positions 0–4 exact the error is +0.2% to +1.2% (`klgate.py selftest` checks
one such profile).

**Expected precision.** Treating each window's sampled rows as a simple random sample of the
positions they stand for, and taking the published K4 run's per-window SDs, the standard error of
the panel estimate is about **0.0014 nats (5.6%)** for a K4-like engine and 0.0009 (6.9%) for a
K6-like one; the 95% half-width is about 0.0027. That separates:

- a K4-like result (0.025) from an NVFP4-cache-like regression (0.055) by more than 20 SE;
- K4-like from K6-like (0.014) by about 8 SE;
- but K4-like from FP8-like (0.021) by only about 3 SE.

The window bootstrap, the published interval, is much wider (for K4, BCa [0.0194, 0.0359]
around 0.0246) because it describes resampling windows; the windows are the same for every
configuration on this panel, so it does not enter a comparison with the published figures.
**Paired differences** between two engine configurations on the same rows are far more precise
than either absolute value, because both runs spike on the same positions.

### Criteria

A run passes when:

1. **Canaries:** `klgate.py selftest` and `klgate.py canary --teacher <teacher-dir>` pass; `score`
   finds every teacher row in the engine output, the engine's token digests match, and every
   window's teacher top-1 equals the next token at 0.2–0.995 of its rows.
2. **Absolute** (`--max-mean 0.040 --min-top1 0.93`): the panel estimate plus 1.96 SE is below
   0.040 nats, between the passing 0.0246 and the failing 0.0548 cache configurations, and top-1
   agreement is at least 0.93. These are starting thresholds; after the first run, tighten to the
   engine's own baseline.
3. **Numerics changes** (`compare <candidate>.json <baseline>.json --margin 0.002`): the upper 95%
   bound (window bootstrap) of the paired mean increase is below 0.002 nats, about 8% of the K4
   figure. The baseline is the same engine without the change, scored on the same rows.

The full panel (all 51,175 positions, 31.7 GB of teacher logits) is the publication-grade
measurement and needs no change to either tool: `klgate_fetch.py --rows-per-window 2047
--head-rows 0`, or the dataset's own files, which `klgate.py` reads directly.

## 4. The engine side (implemented)

In serving, the forward computes logits only for its logit rows: each request's last row in
prefill, every row in decode and verify. Scoring needs the logits of chosen rows of a
teacher-forced window. Two additive pieces do it, and serving does not change.

### 4.1 `GlmForward::score`

In `crates/glm53f-forward/src/forward.rs`:

```rust
/// Teacher-forced scoring: append `tokens` to `kv` (a fresh slot) in passes of `pass_rows`
/// rows and return the f32 logits [rows.len()][VOCAB] of `rows` (ascending indices into
/// `tokens`), padding columns included.
pub fn score(&mut self, kv: &mut GlmKv, tokens: &[u32], rows: &[usize], pass_rows: usize)
    -> Result<Vec<f32>>;
/// The same, handing each row's logits to `sink` (the row, its 154,880 logits) in order as
/// they are computed.
pub fn score_each(&mut self, kv: &mut GlmKv, tokens: &[u32], rows: &[usize], pass_rows: usize,
    sink: impl FnMut(usize, &[f32]) -> Result<()>) -> Result<()>;
```

- **The passes** are those `prefill` runs for one segment cut into chunks of `pass_rows` rows
  (at most the forward's `prefill_rows()`). A pass of 8 rows or fewer runs the row-independent
  decode and verify kernels in one lane, whose bits equal serial decode steps. A larger pass runs
  the prefill kernels (tensor-core GEMMs, E4M3 activations for the FP8 projections), in two lanes
  in a two-lane forward when it has at least twice `min_lane_rows` rows (128 by default) or one
  lane cannot hold it.
- **The head.** Each pass runs the head's final norm over every lane's rows, as any pass
  through the head does, but not the LM head. The requested rows of the pass are then copied from
  their lane's head output (lane A's scratch holds the pass's first rows, lane B's the rest), and
  the LM head runs over them alone, in groups of up to 8 rows: the BF16 GEMV a prefill's last row
  takes. So a row's logits do not depend on which other rows are scored, they equal the forward's
  own logits for that row, and no buffer beyond the forward's own is needed however many rows are
  scored. `score_each` hands the rows over group by group: a caller that writes them out holds at
  most 8.
- **Nothing else changes.** One field of the forward records where lane B's rows start in the
  last pass. The existing passes keep their bits: a digest of every logit, pick and slot byte of
  two-lane and one-lane prefills, chunked and batched prefills, decode, verify and commit on the
  real weights of layers 0-4 is the same before and after the change, and the forward's existing
  tests pass.

**Tests** (`crates/glm53f-forward/tests/score.rs`, the real weights of layers 0-4 and the head,
routed experts returning zeros):

1. A 150-token prompt scored in passes of 64 rows (two lanes each: 32 + 32, 32 + 32, 11 + 11):
   at rows 63, 127 and 149 the bits of the logits a prefill of the prompt up to that row writes
   for its last row, and the same argmax as its pick; the same rows scored on their own, the same
   bits; the slot left as the prefill of the whole prompt leaves it (KDA states, conv windows, DSA
   tail, latent pages).
2. 128 rows in one pass of two lanes (64 + 64) against two one-lane passes of 64 rows: every row
   bit for bit.
3. 29 tokens in passes of 8 and of 5 rows: every row bit for bit serial decode steps.
4. 200 tokens, every row, in passes of 8 rows against one pass of 200 rows (two lanes of 100):
   logits relative RMS 2.1e-2 on average and 3.7e-2 on the worst row (the chain test's bound is
   5e-2), the same argmax on all 125 rows whose best two logits are at least 0.25 apart, and no
   row bit for bit.

### 4.2 `glm53f-score`

A binary crate, `crates/glm53f-score`. It loads the coordinator's weights and connects the
routed experts as `glm53f-serve` does, and builds the forward as `glm53f-serve --prefill-rows R
--prefill-lanes N` would, with one slot and no drafter:

```text
glm53f-score --checkpoint <dir> --ranks <a,b,c,d> --plan <plan.json> --out <dir>
             [--pass-rows <r>] [--windows <id,...>] [--prefill-lanes 1-4]
             [--fp8-act bf16|dynamic] [--no-promote-k32] [--kda-fp8]
             [--kda-state-bf16|--kda-state-f32] [--prefill-w8a16|--prefill-w8a8]
             [--kda-chunked-prefill|--kda-chain-prefill] [--kda-prefill-w8a8]
```

`--pass-rows` is 4096 by default (at most 4,096 per lane). `--experts local` runs the official
FP8 experts on the coordinator's GPU instead of the ranks. The crate documentation
(`crates/glm53f-score/src/lib.rs`) lists every option.

**Input.** `klgate.py plan --teacher <teacher-dir> --out plan.json` writes the plan: schema
`glm53f-kl-plan.v1`, the teacher panel's identity (with the full-panel manifest's when section
6c's windows are in it, and `panel`: the window count, roles and the sha256 of the window ids),
`vocab` 154880, and per window `window_id`, `role`, `tokens` (2,048 ids), `tokens_sha256` (sha256
of the ids as little-endian u32) and `positions` (the rows to write: the teacher rows available,
ascending). The scorer checks the schema, the vocabulary, every id (below 154,856), each window's
digest, and that the positions are distinct, ascending rows below the window's last token.

**Per window:**

1. A fresh slot: empty KV, zero KDA state. No prefix cache, host RAM tier or sharing between
   windows.
2. The ids as they are: no BOS, no template.
3. `score_each` with the plan's positions, in passes of `--pass-rows`: row r is the output at
   input position r, predicting token r + 1. No sampling, no drafting (DFlash and MTP off), no
   grammar. At `--pass-rows 4096` a 2,048-token window is one pass in two lanes of 1,024 rows
   (`glm53f-score`'s default `--prefill-lanes 2`; until 29 September `glm53f-serve`'s default
   `--prefill-rows 4096 --prefill-lanes 2` cut a 2,048-row pass the same way, and its default
   since, `--prefill-rows 8192 --prefill-lanes 4`, cuts it into four lanes of 512); at
   `--pass-rows 8` it is 256 passes.
4. The rows streamed to the window's file as they come, then the slot released.

**Output**, `<out>/<window_id>.safetensors` (written as `<window_id>.safetensors.partial` and
renamed when complete):

| Entry | Content |
|---|---|
| `__metadata__` | `window_id`; `tokens_sha256` (of the ids fed); `plan_sha256` (sha256 of the plan file); `engine`: one line naming the build (its commit, `-dirty` when the sources differ from it) and every numerics choice (the GPU and kernel target, the layers run, the FP8 projections' activations, the routed experts, the KV format, the KDA prefill, the pass rows and lanes, the LM head) |
| `positions` | I32 [k], ascending: the plan's positions for the window |
| `logits` | F32 [k, 154880]: row i is the logits of `positions[i]`, all LM-head columns, as the head computes them (no softmax, temperature or masking) |

Header as in any safetensors file: an 8-byte little-endian length, the JSON, spaces to an
8-byte boundary, then `positions` and `logits` in that order. Also `<out>/run.json`, rewritten
after every window (`complete` is true at the end): the build's revision and engine line, the
plan's digest and teacher identity, the options (with the expert wire's variables), the forward's
configuration, the layers run, the load time, and per window the token count, rows, passes, wall
time, writing time and bytes.

**Full logits, never API log-probabilities.** The teacher stores full logits, so the engine must
write full rows: a top-k list (the OpenAI API's `logprobs`) biases KL low exactly where the
distribution is broad. `klgate.py` refuses rows narrower than the scored columns; with
`--vocab tokenizer` it needs only the first 154,856.

**Size.** The first gate's output is 4,725 rows × 619,520 B = 2.93 GB; every position, 31.7 GB.

**Development mode.** `--dev-layers 0-N` (a prefix of the layers, as in `glm53f-serve`),
`--dev-load-layers N` (layers 0 to N - 1 loaded, all 45 run on repeats of them) and `--experts
zero` (routed outputs of zeros) make a development run on one GPU. Its logits are meaningless; it
says so at start, and its engine line begins with `DEVELOPMENT`. In a development run `--experts
local` gives zeros for the MoE layers the experts directory lacks; otherwise the scorer refuses to
start without every MoE layer's experts.

**Tests** (`crates/glm53f-score`):

- `cargo test -p glm53f-score` (CPU): the options, the plan's checks and the file writer (unit
  tests), and `tests/format.rs` (needs `python3`): a synthetic teacher panel in the dataset's
  layout, `klgate.py plan`, the plan read back, and the teacher's own rows written by the
  scorer's writer and scored by `klgate.py score`: KL exactly 0 at every row; one changed row
  alone above 0; a wrong token digest and a missing row refused.
- `tests/plumbing.rs` (`--features cuda`, one GPU with the coordinator's weights and the FP8
  experts of layers 3 and 4): the binary in development mode (all 45 layers on repeats of layers
  0-4, local experts for layers 3 and 4, zeros for the other 40 MoE layers) on two 160-token
  windows, half of whose tokens are the model's own greedy picks, at `--pass-rows 8` and `4096`.
  A teacher made of the 8-row run's rows gives KL exactly 0 at all 28 rows and top-1 agreement 1
  through `klgate.py score`. With `GLM53F_KL_TEACHER` (the fetched subset), window `final-0000`
  of the real panel at both pass sizes: `klgate.py score` reads all 189 rows against the
  teacher's. The values mean nothing with 5 layers' weights; the formats line up.
- By hand on the same GPU, the whole plan (25 windows, routed outputs of zeros) at both pass
  sizes: `klgate.py score` read all 4,725 rows of each run (28 s each with 8 processes) and
  failed the gate, as a development model must (mean KL 13.05 nats, top-1 agreement 0.0011);
  `compare` paired the two runs. And the `--ranks` form: `--dev-layers 0-4` against four
  `glm53f-rank` daemons on loopback (over TCP, serving the EXL3 shares of layers 3 and 4), window
  `final-0000` at `--pass-rows 2048` and `8`; `klgate.py score` read its 189 rows both times.

### 4.3 On the target hardware

The ranks serve one coordinator at a time: stop `glm53f-serve` before scoring and start it
again afterwards. The ranks run as for serving ([RUNNING.md](RUNNING.md), `GLM53F_WIRE_NOCRC=1`).

```sh
# Build (x86-64, CUDA 12.8 or later; no GPU needed to build).
GLM53F_CUDA_ARCH=sm_120 cargo build --release -p glm53f-score --features cuda,rdma

# The teacher side (once).
python3 harness/klgate.py selftest
python3 harness/klgate.py canary --teacher <teacher-dir>
python3 harness/klgate.py plan --teacher <teacher-dir> --out plan.json

# The engine, twice: the prefill path (the published figures are prefill-shaped) and the
# decode path (generation runs it).
RANKS=192.0.2.10:8600,192.0.2.11:8600,192.0.2.12:8600,192.0.2.13:8600
GLM53F_RDMA=1 GLM53F_WIRE_NOCRC=1 target/release/glm53f-score --checkpoint <coordinator-dir> \
    --ranks "$RANKS" --plan plan.json --pass-rows 4096 --out engine-4096
GLM53F_RDMA=1 GLM53F_WIRE_NOCRC=1 target/release/glm53f-score --checkpoint <coordinator-dir> \
    --ranks "$RANKS" --plan plan.json --pass-rows 8 --out engine-8

# The gate on each, then the two paths paired.
python3 harness/klgate.py score --teacher <teacher-dir> --engine engine-4096 \
    --json prefill.json --max-mean 0.040 --min-top1 0.93
python3 harness/klgate.py score --teacher <teacher-dir> --engine engine-8 \
    --json decode.json --max-mean 0.040 --min-top1 0.93
python3 harness/klgate.py compare decode.json prefill.json
```

Each run prints its engine line and, per window, the tokens, passes, rows and times; `run.json`
keeps them next to the windows' files. A numerics change is scored the same way into its own
directory and compared with `compare <candidate>.json <baseline>.json --margin 0.002`
(section 3). The numerics options (`--kda-fp8`, `--kda-state-bf16`, `--prefill-w8a16`,
`--kda-chunked-prefill`, `--kda-prefill-w8a8` and the flags that turn the defaults off;
[SIZING.md](SIZING.md) §10) are flags of both binaries, with the same defaults
([RUNNING.md](RUNNING.md), "Numerics defaults"), and the engine line names them.

## 5. Cost

- **Engine:** the 25 windows are 51,200 tokens, 25 fresh slots, the LM head on 4,725 rows (6 TFLOP
  in BF16, well under a second) and 2.93 GB written. Measured on the target hardware, model load
  included (sections 6a and 6b): 17–28 s at `--pass-rows 4096` and 247–283 s at `--pass-rows 8`
  (6,400 passes of 8 rows, about 40 ms each). The 125 windows of section 6c took 83–88 s at 4,096
  rows (section 6d). Scoring every row changes only the output size, not the prefill. On the
  development GPU (an RTX 4090; all 45 layers on repeats of layers 0-4, routed outputs of zeros,
  so no expert exchange), the whole plan ran in 20.4 s at `--pass-rows 4096` (0.7-0.9 s a window)
  and 136.8 s at `--pass-rows 8` (5.4 s a window, about 21 ms a pass), model load included, each
  writing 2.93 GB.
- **Harness** (measured on the fetched subset with a stand-in engine): 37 ms per row per core
  (pure Python, float64), so **30 s** for the first gate's 4,725 rows with 8 processes (175 s of
  CPU); the teacher canary, twice the rows, 54 s. The full panel, 51,175 rows, is about 32 CPU
  minutes: about 4 minutes with 8 processes, plus reading 63 GB.
- **Teacher download:** 2.93 GB, about 17 minutes at 3 MB/s (done once).

## 6. Running it

```sh
python3 harness/klgate.py selftest
python3 harness/klgate.py canary --teacher <teacher-dir>
python3 harness/klgate.py plan --teacher <teacher-dir> --out plan.json
glm53f-score --checkpoint <coordinator-dir> --ranks <a,b,c,d> --plan plan.json \
    --pass-rows 4096 --out <engine-dir>          # and --pass-rows 8 (section 4.3)
python3 harness/klgate.py score --teacher <teacher-dir> --engine <engine-dir> \
    --json baseline.json --max-mean 0.040 --min-top1 0.93
# a numerics change: score it the same way, then
python3 harness/klgate.py compare candidate.json baseline.json --margin 0.002
```

`score` prints the position-weighted mean with the SE of the subsample, the window bootstrap
(percentile and BCa), the clustered SE and design effect, top-1 agreement, ln(PPL ratio),
quantiles, per domain, per role, per position bucket (0–256, 256–1024, 1024 on) and per window;
`--json` keeps every row's KL for later paired comparisons. By default the commands take the
panel the fetch recorded (`FETCH-MANIFEST.json`), else the final windows. `--windows`, `--roles`
(e.g. `final`) and `--exclude` select windows; the registry's calibration-clean scope ("clean17")
excludes `final-0003`, `-0007`, `-0011`, `-0015`, `-0019`, `-0021`, `-0022` and `-0023`, and is
compared only with clean17 figures. Exit status: 0 pass, 1 error, 3 gate failed.

## 6a. First result on the target hardware (28 September 2026)

**Configuration:**
- One RTX 5090 and four GB10 expert ranks over RDMA, all 45 layers.
- Official FP8 non-expert weights, BF16 KDA projections, EXL3 K4 routed experts.
- FP8 MLA cache, FP8 wire rows, no drafting.
- The first gate's rows: 25 windows × 189.

| Engine path | Mean KL (nats) | + 1.96 SE | Top-1 agreement | Gate |
|---|---:|---:|---:|---|
| `--pass-rows 8` (decode kernels; FP8 projections take BF16 activations) | **0.02446** | 0.02662 | 0.9510 | PASS |
| `--pass-rows 4096` (prefill kernels, two lanes; E4M3 activations) | 0.02825 | 0.03099 | 0.9471 | PASS |

**Against the published figures** (section 2, all 25 windows):
- EXL3 K4 measures 0.024555, offline, with no KV-cache quantization. The decode path matches it
  within its standard error, although it adds an FP8 cache and FP8 wire rows.
- The scopes differ: 189 rows per window here, every row there.

**Prefill path against decode path** (`compare`, paired over 4,725 rows):
- The prefill path is worse by **0.0038 nats**, 95% window bootstrap [0.0004, 0.0090].
- Top-1 disagreements: 125 rows one way against 100 the other (McNemar p = 0.11).
- The likely source is the prefill FP8 GEMMs' activation quantization (E4M3 per 128 columns),
  which the decode path does not use. A W8A16 or finer-scaled prefill path is the candidate fix.
- The weakest window in both runs is `final-0021` (legal text): 0.047–0.052 nats, top-1 0.89–0.90.

**Timings:** loading 4.3 s, then 23.8 s for the whole plan at 4,096 rows per pass and 279 s at 8.

## 6b. The numerics options on the target hardware (29 September 2026)

**The regression check:**
- The engine at main `af0c565`, with every option off, gave **exactly** section 6a's result at 4,096 rows: mean 0.028249107…, top-1 0.94707.
- The changes merged between the two runs are these:
  - the rank kernel's large-M and split schedules;
  - the lane scratch and MLA row blocks;
  - up to four prefill lanes;
  - copy windows;
  - the options themselves, when off.
- None of them moved a bit of the engine's output. Section 6a's 8-row run therefore stands as the 8-row baseline.

**Each option, paired against that baseline** (`compare --margin 0.002`, 4,725 rows). "Mean A−B" is the change in mean KL over the same rows; "upper" is the 95% window-bootstrap bound.

| Option | Rows per pass | Mean KL | Mean A−B | Upper | Verdict |
|---|---:|---:|---:|---:|---|
| `--kda-state-bf16` (D8) | 8 | 0.02398 | −0.0005 | +0.0008 | **PASS** |
| `--kda-state-bf16` (D8) | 4,096 | 0.02767 | −0.0006 | +0.0010 | **PASS** |
| `--kda-fp8` (D2, 128 × 128 weight scales) | 8 | 0.02831 | +0.0039 | +0.0083 | FAIL (top-1 McNemar p = 0.027) |
| `--kda-fp8` | 4,096 | 0.03136 | +0.0031 | +0.0074 | FAIL |
| `--prefill-w8a16` | 4,096 | 0.02613 | −0.0021 | +0.0021 | FAIL (interval) |
| `--kda-chunked-prefill` | 4,096 | 0.03036 | +0.0021 | +0.0047 | FAIL |
| `--kda-chunked-prefill --prefill-w8a16` | 4,096 | 0.02626 | −0.0020 | +0.0026 | FAIL (interval) |
| the same plus D8 | 4,096 | 0.02678 | −0.0015 | +0.0029 | FAIL (interval) |
| D2 combined with W8A16, D8 or chunked | 4,096 / 8 | 0.0265–0.0292 | −0.0018 to +0.0047 | +0.0021 to +0.0087 | FAIL |

**Decisions:**
- **D8 is on by default.** `glm53f-serve` and `glm53f-score` both take it. `--kda-state-f32` (or `GLM53F_KDA_STATE_BF16=0`) reproduces section 6a's configuration.
- **D2 is rejected in this form:** it moves decode by +0.0039 nats and flips top-1 on significantly more rows. A finer weight scale (MXFP8's one scale per 32 values) is the variant to gate next.
- **W8A16, and the chunked KDA prefill with it, lower the mean KL** (they close most of the prefill-versus-decode gap of section 6a). But at this per-row correlation (0.72–0.76), 25 windows give an interval of about ±0.005 nats, too wide to show non-inferiority at 0.002. They stay opt-in until the 125-window panel of section 6c decides (section 6d has its first result; making
them defaults also waits for the speed comparison).
  - Speed is the reason to try: with four lanes they prefill about 5.2K tok/s against 4.1K (`docs/PERFORMANCE.md` §0).
  - The chunked kernel alone is worse (+0.0021), so it would only ever be paired with W8A16.

## 6c. A larger panel for paired comparisons: 125 windows

**More windows, not more rows.** In section 6b's reports the interval's width comes from how much
the windows differ: for the W8A16 pairs the row subsample adds only 13–18% of the variance of the
windows' paired means. More rows per window would barely narrow it; more windows do.

**What the dataset holds.** 665 windows of 2,048 tokens (2,047 positions each; a file of
1,268,157,840 or 1,268,157,848 bytes a window, 117.1 MB for the 189 rows), from 20 packed source
documents, five per domain, each document in a single role:

| Role | Windows | Documents (windows each) | Listed in |
|---|---:|---|---|
| `final` | 25 | 4 (7, 6, 6, 6) | `dataset-manifest.json` |
| `confirmation` | 64 | 4 (17, 16, 16, 15) | `logits/full-panel/full-panel-manifest.json` |
| `selection` | 64 | 4 (16 each) | the same |
| `conditional-fit` | 128 | 4 (38, 37, 37, 16) | the same |
| `fit` | 384 | 4 (132, 104, 132, 16) | the same |

- A role's windows take its four documents (one per domain) in turn, while they last.
- The full panel's capture receipts record the final panel's method: the same model revision,
  eager attention, no KV cache, four-way expert parallelism, F32 tensors kept. Its file headers
  carry the same metadata as the final windows'.
- `full-panel-manifest.json` (sha256 `c0c70608c6436324852732720afa6d060e7e220f2a6ce2945e8fe63ba8b99a8f`;
  its `full_panel_manifest_sha256` field `8397ef9d3eedbb256d09f2166fdb3e337dde907fd7b710510bb293b7190c7917`)
  binds each file's size, sha256 and token digest. Both tools check it against
  `dataset-manifest.json`: the same model revision, token panel and vocabulary, and no repeated or
  final window.
- **Caveat.** The dataset's README says captures over the non-final windows were used by the
  checkpoint author's routed-expert campaigns. The K4 experts may have been fitted or chosen on
  them, so their absolute KL may be lower than on unseen text. They serve paired comparisons of
  non-expert numerics, where both arms have the same experts; absolute figures stay on the final
  windows (`--roles final`).

**The rule.** `klgate_fetch.py --panel N` takes the first N windows in the order final,
confirmation, selection, conditional-fit, fit, each role in window-id order, with the first gate's
189 rows each. The panels are nested and N = 25 is the first gate's, so earlier reports stay
comparable on their rows. The order puts first the roles whose names suggest the least use in
building a checkpoint (an inference from the names).

**How many windows.** For window w, d_w is the mean of A − B over its positions (`compare`'s
interval is the percentile bootstrap of the mean of the d_w; every window has 2,047 positions, so
that is the plain mean). The three pairs of section 6b that bear on the prefill options, from their
per-row reports (4,096 rows per pass, the 25 final windows):

| Pair | Mean A − B | SD of d_w | Skew | Row sampling's share of the variance |
|---|---:|---:|---:|---:|
| chunked + W8A16 against flags off | −0.00199 | 0.0136 | −2.2 | 13% |
| W8A16 against flags off | −0.00212 | 0.0129 | −2.6 | 14% |
| chunked + W8A16 + D8 against D8 (today's defaults) | −0.00089 | 0.0115 | −0.7 | 18% |

Two windows carry much of the spread. On `final-0004` the prefill path without W8A16 scores
0.124–0.137 (the decode path 0.082–0.089) and W8A16 closes the gap: d = −0.041 to −0.057. On `final-0000`
W8A16 is worse than both paths: d = +0.025 to +0.032. Without those two windows the SDs are
0.004–0.005.

**Windows needed for 90% power.** The probability that `compare --margin 0.002` passes depends on
the true mean difference δ. Each cell is the smallest panel with at least 90% power, by the
normal approximation / by the bootstrap of the rule / by the bootstrap keeping today's 25 windows
as measured (methods below; the bootstrap sizes are good to about ±5):

| Pair | δ = −0.002 | δ = −0.001 | δ = 0.000 |
|---|---:|---:|---:|
| chunked + W8A16 against flags off | 123 / 90 / 80 | 217 / 180 / 150 | 489 / 400 / 350 |
| W8A16 against flags off | 110 / 75 / 65 | 195 / 145 / 115 | 438 / 365 / 290 |
| chunked + W8A16 + D8 against D8 | 87 / 80 / 85 | 154 / 140 / 130 | 346 / 325 / 285 |

**What 125 windows decide.** The probability that the panel of this section passes, at a true
difference δ (normal / bootstrap / keeping the 25):

| δ | chunked + W8A16 against flags off | W8A16 against flags off | chunked + W8A16 + D8 against D8 |
|---:|---|---|---|
| −0.0030 | 0.98 / 1.00 / 1.00 | 0.99 / 1.00 / 1.00 | 1.00 / 1.00 / 1.00 |
| −0.0020 | 0.91 / 0.96 / 0.98 | 0.93 / 0.99 / 0.99 | 0.97 / 0.98 / 0.98 |
| −0.0015 | 0.82 / 0.91 / 0.94 | 0.86 / 0.94 / 0.97 | 0.93 / 0.94 / 0.94 |
| −0.0010 | 0.69 / 0.79 / 0.87 | 0.74 / 0.86 / 0.92 | 0.83 / 0.85 / 0.88 |
| −0.0005 | 0.54 / 0.60 / 0.74 | 0.58 / 0.68 / 0.82 | 0.68 / 0.72 / 0.77 |
| 0.0000 | 0.37 / 0.43 / 0.59 | 0.41 / 0.47 / 0.67 | 0.50 / 0.54 / 0.63 |
| +0.0010 | 0.13 / 0.12 / 0.25 | 0.14 / 0.13 / 0.30 | 0.16 / 0.18 / 0.27 |
| +0.0020 | 0.03 / 0.02 / 0.06 | 0.03 / 0.02 / 0.07 | 0.03 / 0.02 / 0.06 |

- **The rule at 125 windows** passes when the panel's mean difference is below about zero
  (−0.0002 to +0.0001): the bound sits 0.0019–0.0022 above the mean and the margin is 0.002. A pass
  shows, with 95% confidence, that the candidate is not worse than the baseline by 0.002 nats, 8%
  of the K4 figure. (The half-width, 1.96 SD/√W for the first pair: 0.0053, 0.0038, 0.0027, 0.0024
  and 0.0019 at 25, 50, 100, 125 and 200 windows.)
- **A gain of 0.002 or more** passes with probability 0.91–0.99. Against today's defaults the 25
  windows measured a gain of 0.0009 (the 125, 0.0014: section 6d): 0.83–0.88 at a true 0.001.
- **No change (0.000)** passes 37–67% of the time. 125 windows cannot show non-inferiority for a
  candidate that is only as good as the baseline; that takes 285–490 windows (the last column
  above; the dataset has 665 windows).
- **A change worse by the margin (+0.002)** passes 2–7% of the time; worse by 0.001, 12–30%.
- So a pass is a finding. A fail with a mean difference between 0 and +0.002 is inconclusive, not a
  sign of harm: the interval then holds both zero and the margin. `compare` prints the difference
  per role, so an effect that differs between the final windows and the others shows in the report.

**The method.**

- **The rule** is `compare`'s: the upper end of the 95% percentile bootstrap (B = 5,000) of
  mean(d_w) below the margin.
- **Normal:** SE = s/√W with s the SD of the 25 observed d_w; the rule passes when the mean plus
  1.96 SE is below the margin, so the power is Φ((0.002 − δ)/SE − 1.96) and 90% power needs
  W = (1.96 + 1.2816)² s²/(0.002 − δ)².
- **Bootstrap:** the rule itself, by Monte Carlo. A panel of W windows is drawn with replacement
  from the 25 observed d_w shifted to mean δ (this keeps their skew and the two outlying windows)
  and judged by `compare`'s percentile bootstrap; the power is the fraction of panels that pass.
  The sizes come from 1,500 panels of 1,500 resamples at each size searched (steps of about 12% of
  the normal answer, interpolated linearly); simulated again at the sizes found, with 4,000 panels
  of 5,000 resamples, they give 0.89–0.91. The power at 125 windows is from 4,000 panels of 5,000
  resamples (Monte Carlo error about ±0.015).
- **Keeping the 25:** the same, but the panel is today's 25 windows as measured plus W − 25 drawn
  ones. That is the design: the panel contains the 25, which a run of the panel measures again on
  the same rows, and only the added windows are unknown. It is higher wherever the 25's mean is
  below δ.
- **Assumed:** the added windows' d_w follow the final windows' distribution; windows are
  independent (a one-way analysis by domain puts the between-domain share of the variance at 0 for
  all three pairs, but with one document per domain, domain and document cannot be told apart);
  the true difference is the same in every role. The teacher's own statistics do not tell the roles
  apart (mean entropy per window 0.86, 0.89 and 0.87 nats for final, confirmation and selection
  windows, SD 0.44–0.46; the teacher's top-1 is the next token at 0.73, 0.73 and 0.74 of the rows),
  which supports the assumption weakly and does not test it: the added windows' d_w are what the run
  measures.

Its limits:
- The SD rests on 25 windows, two of which dominate. Bootstrapping the 25, the SD's 10th, 50th and
  90th percentiles are 0.0046, 0.0133 and 0.0188 (first pair), which means 14, 116 or 231 windows
  for 90% power at δ = −0.002 (normal), and 56, 466 or 924 at δ = 0. If windows like those two are
  rare among the added ones, 125 is far more than needed; if they are as common as in the 25 (2 in
  25), it is about enough for a true gain of 0.002.
- Against today's defaults (D8 on), chunked + W8A16 measured −0.0009 over the 25 windows, not
  −0.002. At a true −0.001, 125 windows pass with probability 0.83–0.88 for that pair, and 90%
  needs 130–154. `--panel 153`, the next balanced size (all the confirmation and selection
  windows: 28 more, 3.3 GB), gives 0.90 (normal).
- Fix the panel before scoring. Growing it after an inconclusive result is a second look, which
  the 95% bound does not account for.

**125 windows** is the smallest balanced size with at least 90% power at δ = −0.002 for all three
pairs by the normal method (which needs 123, 110 and 87; the bootstrap needs fewer): the 25 final
windows, the 64 confirmation windows and the first 36 selection windows, nine from each selection
document. Domains 33/31/31/30 windows (general, legal, code, reasoning), 12 documents, window-id
sha256 `4e25ad0a446cfd796348aae53c218e5200e7e6e81db93b70fde2bc94a37e4645` (`panel.window_ids_sha256`
in `FETCH-MANIFEST.json`, and in every plan and report on it).

**Fetching and checking it.**

```sh
python3 harness/klgate_fetch.py --revision 95f4fdd94bf29989db2e0d1054e4931f55edb6aa \
    --panel 125 --reuse <first-gate-dir> --rate 5e6 --max-bytes 12e9 --connections 24 --out <teacher-dir>
```

- **What it holds:** 125 windows × 189 rows = 23,625 rows, 14.64 GB in 262 files (`SHA256SUMS`
  sha256 `0a0ebf6efb78be6e5293281904a6022d936ada654f0a9aea2b6596c659a73035`). The first 25
  windows and one more came from earlier fetches; the other 99 windows' 18,711 rows and the whole
  files, 11.59 GB in all, took 39 minutes over 24 connections (4.95 MB/s).
- **A resumed fetch.** This run stopped after its last row, inside its closing step (every row
  read again, the files renamed, the manifest written). Running the same command finished it:
  no row was fetched again, 0.5 MB came from the Hub (its listing and each window's header).
  `FETCH-MANIFEST.json` records the 35 windows the stopped run had finished by their rows only
  (`rows_sha256` null); `SHA256SUMS` binds their bytes.
- **Checked:** `sha256sum -c SHA256SUMS` (262 files). `klgate.py canary` over the 125 windows: the
  teacher against itself exactly 0 at 23,625 rows; each row against the next row of its window
  15.4 nats on average, 15.8 times the teacher's mean entropy (0.98); the teacher's top-1 equal to
  the next token at 0.51–0.92 of every window's rows; at most 5.9e-5 of the teacher's probability
  on the padded columns (2.1e-5 in the final windows). `klgate.py plan`: 1,336,059 bytes, sha256
  `121684e8bfa728211e41216b7cb50f2b312d8f8fa467aafd0c9514d815d16323`, 23,625 rows, 14.64 GB of F32
  logits per engine run. `glm53f-score`'s plan reader accepts it as it is
  (`GLM53F_KL_TEACHER=<teacher-dir> cargo test -p glm53f-score` plans the directory and reads
  the plan back with the model's widths).
- **Consistency with the first gate:** the 25 final windows of this directory, scored with
  `--roles final` against the fourteen engine outputs of section 6b's runs (25 windows each),
  reproduce those reports' every row: each window's KL and top-1 at every row, and the means,
  exactly.
- **Stand-in engines.** Two (the teacher's rows plus Gaussian noise of 0.05 and 0.06, written in
  `glm53f-score`'s format) over the whole plan: `score` reads all 23,625 rows; its report does not
  depend on the number of processes; the final windows score identically alone and inside the
  125-window run; `compare` pairs the two, and refuses a 125-window report against a final-only one.

**Running it.** As section 4.3, with the plan of this directory, and `score` twice per arm: over
the whole panel (what `compare` pairs), and over `--roles final` for the absolute gate, whose
thresholds were set on the final windows.

```sh
python3 harness/klgate.py plan --teacher <teacher-dir> --out plan-125.json
glm53f-score --checkpoint <coordinator-dir> --ranks <a,b,c,d> --plan plan-125.json \
    --pass-rows 4096 --out <engine-dir>           # per arm; the options under test as flags
python3 harness/klgate.py score --teacher <teacher-dir> --engine <engine-dir> --json <arm>.json
python3 harness/klgate.py score --teacher <teacher-dir> --engine <engine-dir> --roles final \
    --max-mean 0.040 --min-top1 0.93 --json <arm>-final.json
python3 harness/klgate.py compare <candidate>.json <baseline>.json --margin 0.002
```

**Cost.** The engine scored the 125 windows in 88 s at 4,096 rows per pass with the defaults and in
83 s with the chunked KDA prefill and W8A16 (section 6d; the 4.3 s model load included), and wrote
14.64 GB per run; at 8 rows per pass it is five times the 25-window runs of section 6b, about 20–22
minutes (an estimate). `klgate.py score` over the 125 windows is about 1,000 CPU-seconds (44 ms a
row): 69 s with 16 processes, 115 s with 8 (the default), reading 29 GB (the engine's rows and the
teacher's; page cache warm) on a shared machine; `compare` takes under a second. The teacher's
11.6 GB took 39 minutes at 5 MB/s (once). Two engine runs for a comparison are 29 GB on disk.

## 6d. First use of the 125-window panel (29 September 2026)

Target hardware, 29 September, engine `073b553`, the plan of section 6c, `--pass-rows 4096`. A is
the chunked KDA prefill with W8A16 (`--kda-chunked-prefill --prefill-w8a16`), B today's defaults
(BF16 KDA states on). Each engine run scored the 125 windows in 88 s (B) and 83 s (A), the 4.3 s
model load included, and wrote 14.64 GB.

```text
paired over 23,625 rows in 125 windows: A 0.024928, B 0.026295
  mean difference A - B       -0.001367  window bootstrap 95% [-0.002988, +0.000192]  clustered SE 0.000800
  ratio A / B                 0.9480  95% [0.8931, 1.0079]
  per-row correlation         0.8560
  top-1 (rows): A agrees and B not 477, B agrees and A not 497; McNemar p 0.543
  mean A - B per role         final 25 windows -0.000891; confirmation 64 windows -0.001186; selection 36 windows -0.002019
gate: PASS: upper 95% bound of A - B +0.000192 < margin 0.002
```

The absolute gate on the 25 final windows (`--roles final --max-mean 0.040 --min-top1 0.93`): B
0.02767 (top-1 0.9470), A 0.02678 (0.9482); both pass.

- **Result:** chunked KDA prefill with W8A16 is not worse than the defaults by more than the
  margin: the upper bound, +0.0002, is a tenth of it. The mean gain is 0.0014 nats (5%); its
  interval reaches just above zero, so the gain itself is not shown at 95%. Top-1 agreement moves
  on as many rows one way as the other (477 against 497, McNemar p = 0.54). The difference has the
  same sign in every role.
- **Regression check:** the defaults at 4,096 rows per pass, scored on the first gate's 25-window plan,
  are identical row for row (4,725 rows) to section 6b's D8 arm: the changes merged since leave the
  defaults' output bit for bit. The 25 final windows of this panel give the same d_w as before
  (−0.000891).
- **The KL result only:** the two options stay opt-in. Making them defaults waits for the speed
  comparison.

**How the estimate of section 6c held.** It was made before this run, from the 25 final windows.
The panel's window-level spread, for this pair:

| Windows | n | Mean A − B | SD of d_w | Skew |
|---|---:|---:|---:|---:|
| final | 25 | −0.00089 | 0.0115 | −0.7 |
| confirmation | 64 | −0.00119 | 0.0061 | +0.6 |
| selection | 36 | −0.00202 | 0.0112 | −1.9 |
| added (confirmation and selection) | 100 | −0.00149 | 0.0083 | −1.6 |
| all | 125 | −0.00137 | 0.0090 | −1.3 |

- **The assumption held for this pair, on the side of caution.** The added windows have outliers
  of the finals' size (`selection-0033`, d = −0.053, and `selection-0009`, +0.032, against
  `final-0004`, −0.041, and `final-0000`, +0.032), but fewer of them: two windows with |d_w| above
  0.03 in 100 against two in 25. The SD over the 125 windows is 0.0090, not 0.0115, so the bound
  sits 0.0016 above the mean where the estimate expected 0.0019 for this pair. The measured gain,
  −0.0014, lies between the rows −0.001 and −0.0015 of the table for this pair (0.83–0.94); the
  panel passed.
- **The per-row correlation** of A and B is 0.856 over the 125 windows and 0.694 in the 25 final
  windows alone; the three pairs of section 6c have 0.69–0.72 on the final windows (section 6b
  quoted 0.72–0.76 for its pairs). The estimate does not assume a correlation: it takes the pairs'
  window-level spread as measured, and the spread carries it. That the added windows' paired rows
  agree more closely than the finals' is part of why the interval came out narrower.
- **For planning** another comparison of this kind, the 125-window SD, 0.0090, is a better input than
  the 25-window 0.0115: 90% power (normal) needs 94 windows at δ = −0.001 and 210 at δ = 0.000
  (154 and 346 by the 25's SD), and 125 windows pass with probability 0.96 and 0.71.

**29 September 2026: the pair became the default** after the speed comparison, run on the target
hardware the same day with the same engine (`073b553`; [PERFORMANCE.md](PERFORMANCE.md) §0):
- prefill 4,979–4,998 / 5,128–5,180 / 5,132–5,188 tok/s at 4K / 19K / 79K tokens against B's
  4,099–4,129 / 4,103–4,104 / 4,051–4,053, 21–28% faster;
- decode −2% to +5%, so neutral. Greedy replies change: all six single-stream replies differed,
  within their first 340 characters, because a prompt's own prefill takes the new arithmetic;
- at 16 slots the KV pool is 8.73 GiB against 9.33 (the chunked prefill's workspace and W8A16's
  GEMM scratch), and a 1,048,576-token request still fits.

`glm53f-serve` and `glm53f-score` now run the chunked KDA prefill with W8A16 unless told
otherwise. B, the defaults this comparison was made against, is `--kda-chain-prefill
--prefill-w8a8`; adding `--kda-state-f32` gives section 6a's configuration. The chunked kernel
without W8A16 failed section 6b's gate (+0.0021), so the two are turned off together.

## 7. Open points

- The non-final windows of section 6c may have served in building the K4 experts; they are used
  only for paired comparisons.
- Section 6c's estimate assumed that the added windows' paired differences resemble the final
  windows'. Section 6d's first result supports it for one pair (the added windows' SD was lower, not
  higher); another pair's spread is measured by its own run.
- The SE of the subsample treats systematic sampling as simple random sampling, and the per-window
  SDs behind the expected precision come from the K4 offline run, not this engine.
- The window-0 runtime figures (0.0246 and 0.0548) and the 25-window figures are different
  scopes; the gate's absolute threshold leans on both.
- Rows 5–10 of each window stand for themselves and their five neighbours, where the KL is still
  falling; the residual bias on the synthetic profiles is +0.2% to +1.2%.
