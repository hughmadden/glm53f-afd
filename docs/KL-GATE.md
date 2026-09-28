# KL gate

**Status (28 September 2026): the harness and the teacher subset are ready; the engine side is
specified here and not implemented.** Until the engine can write per-position logits, the
model's quality has only been spot-checked (arithmetic, needle retrieval, coherent text).

The gate measures how far the engine's next-token distributions are from the BF16 model's, on the
public panel that the published GLM-5.3-Flash quantization figures were measured on, with the
same method. It answers two questions:

1. **Absolute:** is the engine's configuration (official FP8 non-expert weights, EXL3 4-bit
   experts, FP8 MLA latent and index keys) in the regime of the published configurations?
2. **Relative:** does a numerics change (FP8 KDA projections, FP8 wire rows, W8A8 prefill GEMMs,
   the chunked KDA prefill) make it measurably worse? A paired comparison on the same positions.

Files: [`harness/klgate.py`](../harness/klgate.py) (the gate),
[`harness/klgate_fetch.py`](../harness/klgate_fetch.py) (the teacher subset),
[`harness/PROVENANCE-klgate.md`](../harness/PROVENANCE-klgate.md) (sources). Both tools use only the Python
standard library (tested with Python 3.12).

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
- **The rest of the repository (not used):** `logits/full-panel/` (640 calibration windows,
  811.6 GB), `calibration/main-ep4-full/` (464.4 GB of hidden states and router choices),
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
| K4 on the author's production TP4 runtime, padded columns dropped | 0.030480, BCa 95% [0.024965, 0.037419] | 0.9467 | 25 windows | the author's `kld_eval` |

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

## 4. The engine side (specified, not implemented)

Today the forward returns logits only for its logit rows: each request's last row in prefill,
every row in decode and verify. Scoring needs the logits of chosen rows of a teacher-forced
window. Two additive pieces, and no change to serving:

### 4.1 `GlmForward::score`

```rust
/// Teacher-forced scoring: append `tokens` to `kv` (a fresh slot) in passes of `pass_rows`
/// rows and return the f32 logits [rows.len()][VOCAB] of `rows` (ascending indices into
/// `tokens`), padding columns included.
pub fn score(&mut self, kv: &mut GlmKv, tokens: &[u32], rows: &[usize], pass_rows: usize)
    -> Result<Vec<f32>>;
```

- It runs the same pass as `prefill`, with the pass's logit rows set to the requested rows that
  fall in the chunk (the pass already takes its logit rows as a list). The first gate asks for at
  most 28 rows in any 256-row chunk, so the default 64-row logits scratch suffices; scoring every
  row needs it sized for `pass_rows` (256 × 154,880 × 4 B = 159 MB) or the head run in
  sub-batches.
- `pass_rows` above 8 exercises the prefill path (tensor-core GEMMs, E4M3 activations for the FP8
  projections); 8 or fewer, the row-independent decode and verify kernels, whose bits equal serial
  decode. The gate runs both: the published figures are prefill-shaped, and generation runs the
  decode path.
- It belongs in `glm53f-forward` once the prefill work in progress has landed; nothing else in the
  crate changes.

### 4.2 `glm53f-score`

A new binary crate, loading the coordinator exactly as `glm53f-serve` does:

```text
glm53f-score --checkpoint <dir> --ranks <a,b,c,d> --plan <plan.json> --out <dir>
             [--pass-rows <r>] [--windows <id,...>] [the forward's numerics options]
```

**Input.** `klgate.py plan --teacher <teacher-dir> --out plan.json` writes the plan: schema
`glm53f-kl-plan.v1`, the teacher panel's identity, `vocab` 154880, and per window `window_id`,
`tokens` (2,048 ids), `tokens_sha256` (sha256 of the ids as little-endian u32) and `positions`
(the rows to write: the teacher rows available, ascending).

**Per window:**

1. A fresh slot: empty KV, zero KDA state. No prefix cache, host RAM tier or sharing between
   windows.
2. Feed the 2,048 ids as they are: no BOS, no template. Check every id is below 154,856 and the
   sha256 of the ids fed equals `tokens_sha256`.
3. `score` with the plan's positions: row r is the output at input position r, predicting
   token r + 1. No sampling, no drafting (DFlash and MTP off), no grammar.
4. Write the rows and release the slot.

**Output**, `<out>/<window_id>.safetensors`:

| Entry | Content |
|---|---|
| `__metadata__` | `window_id`; `tokens_sha256` (of the ids actually fed); `plan_sha256` (sha256 of the plan file); `engine`: one line naming the build and every numerics choice (expert format, KV format, pass rows, FP8 options) |
| `positions` | I32 [k], ascending: the plan's positions for the window |
| `logits` | F32 [k, 154880]: row i is the logits of `positions[i]`, all LM-head columns, as the head computes them (no softmax, temperature or masking) |

Header as in any safetensors file: an 8-byte little-endian length, the JSON, spaces to an
8-byte boundary, then the tensors in `data_offsets` order. Also `<out>/run.json`: the build's
revision, the options, and per-window token counts and wall times.

**Full logits, never API log-probabilities.** The teacher stores full logits, so the engine must
write full rows: a top-k list (the OpenAI API's `logprobs`) biases KL low exactly where the
distribution is broad. `klgate.py` refuses rows narrower than the scored columns; with
`--vocab tokenizer` it needs only the first 154,856.

**Size.** The first gate's output is 4,725 rows × 619,520 B = 2.93 GB; every position, 31.7 GB.

## 5. Cost

- **Engine:** the 25 windows are 51,200 tokens. At the current prefill rate of about 1.7K tok/s
  that is **about 30 s**, plus 25 fresh slots, the LM head on 4,725 rows (6 TFLOP in BF16, well
  under a second) and writing 2.93 GB: **under a minute** after the model is loaded. Scoring
  every row changes only the output size, not the prefill. The decode-path run (`--pass-rows 8`)
  is 6,400 passes of 8 rows: a few minutes if a pass takes tens of milliseconds (an estimate;
  not measured).
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
glm53f-score ... --plan plan.json --out <engine-dir>
python3 harness/klgate.py score --teacher <teacher-dir> --engine <engine-dir> \
    --json baseline.json --max-mean 0.040 --min-top1 0.93
# a numerics change: score it the same way, then
python3 harness/klgate.py compare candidate.json baseline.json --margin 0.002
```

`score` prints the position-weighted mean with the SE of the subsample, the window bootstrap
(percentile and BCa), the clustered SE and design effect, top-1 agreement, ln(PPL ratio),
quantiles, per domain, per position bucket (0–256, 256–1024, 1024 on) and per window; `--json`
keeps every row's KL for later paired comparisons. `--windows` and `--exclude` select windows; the
registry's calibration-clean scope ("clean17") excludes `final-0003`, `-0007`, `-0011`, `-0015`,
`-0019`, `-0021`, `-0022` and `-0023`, and is compared only with clean17 figures. Exit status: 0
pass, 1 error, 3 gate failed.

## 7. Open points

- The SE of the subsample treats systematic sampling as simple random sampling, and the per-window
  SDs behind the expected precision come from the K4 offline run, not this engine.
- The window-0 runtime figures (0.0246 and 0.0548) and the 25-window figures are different
  scopes; the gate's absolute threshold leans on both.
- Rows 5–10 of each window stand for themselves and their five neighbours, where the KL is still
  falling; the residual bias on the synthetic profiles is +0.2% to +1.2%.
