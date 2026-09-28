# KL gate

**Status (28 September 2026): the gate has run on the target hardware, and both engine paths
pass** (section 6a). The decode path scores 0.0245 nats against the BF16 teacher, equal within its
standard error to the published figure for the same 4-bit experts. On 29 September the numerics
options were gated (section 6b): BF16 KDA states (D8) passed and are now the default; FP8 KDA
projections (D2) failed.

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
             [--kda-chunked-prefill] [--fp8-act bf16|dynamic] [--no-promote-k32]
             [--kda-fp8] [--kda-state-bf16] [--prefill-w8a16] [--kda-prefill-w8a8]
```

`--pass-rows` is 4096 by default (at most 4,096 per lane). `--experts local` runs the official
FP8 experts on the coordinator's GPU instead of the ranks. The crate documentation
(`crates/glm53f-score/src/lib.rs`) lists every option.

**Input.** `klgate.py plan --teacher <teacher-dir> --out plan.json` writes the plan: schema
`glm53f-kl-plan.v1`, the teacher panel's identity, `vocab` 154880, and per window `window_id`,
`tokens` (2,048 ids), `tokens_sha256` (sha256 of the ids as little-endian u32) and `positions`
(the rows to write: the teacher rows available, ascending). The scorer checks the schema, the
vocabulary, every id (below 154,856), each window's digest, and that the positions are distinct,
ascending rows below the window's last token.

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
(section 3). The numerics options under test (`--kda-fp8`, `--kda-state-bf16`, `--prefill-w8a16`,
`--kda-prefill-w8a8`; [SIZING.md](SIZING.md) §10) are flags of both binaries, named in the engine
line.

## 5. Cost

- **Engine:** the 25 windows are 51,200 tokens. At the current prefill rate of about 1.7K tok/s
  that is **about 30 s**, plus 25 fresh slots, the LM head on 4,725 rows (6 TFLOP in BF16, well
  under a second) and writing 2.93 GB: **under a minute** after the model is loaded. Scoring
  every row changes only the output size, not the prefill. The decode-path run (`--pass-rows 8`)
  is 6,400 passes of 8 rows: a few minutes if a pass takes tens of milliseconds (an estimate for
  the target hardware; not measured there). On the development GPU (an RTX 4090; all 45 layers
  on repeats of layers 0-4, routed outputs of zeros, so no expert exchange), the whole plan ran
  in 20.4 s at `--pass-rows 4096` (0.7-0.9 s a window) and 136.8 s at `--pass-rows 8` (5.4 s a
  window, about 21 ms a pass), model load included, each writing 2.93 GB. *Measured on the target
  hardware since (section 6a): 23.8 s for the whole plan at 4,096 rows per pass and 279 s at 8,
  after 4.3 s of loading.*
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
quantiles, per domain, per position bucket (0–256, 256–1024, 1024 on) and per window; `--json`
keeps every row's KL for later paired comparisons. `--windows` and `--exclude` select windows; the
registry's calibration-clean scope ("clean17") excludes `final-0003`, `-0007`, `-0011`, `-0015`,
`-0019`, `-0021`, `-0022` and `-0023`, and is compared only with clean17 figures. Exit status: 0
pass, 1 error, 3 gate failed.

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
- **W8A16, and the chunked KDA prefill with it, lower the mean KL** (they close most of the prefill-versus-decode gap of section 6a). But at this per-row correlation (0.72–0.76), 25 windows give an interval of about ±0.005 nats, too wide to show non-inferiority at 0.002. They stay opt-in until a larger panel (about 100 windows, which halves the interval) decides.
  - Speed is the reason to try: with four lanes they prefill about 5.2K tok/s against 4.1K (`docs/PERFORMANCE.md` §0).
  - The chunked kernel alone is worse (+0.0021), so it would only ever be paired with W8A16.

## 7. Open points

- The SE of the subsample treats systematic sampling as simple random sampling, and the per-window
  SDs behind the expected precision come from the K4 offline run, not this engine.
- The window-0 runtime figures (0.0246 and 0.0548) and the 25-window figures are different
  scopes; the gate's absolute threshold leans on both.
- Rows 5–10 of each window stand for themselves and their five neighbours, where the KL is still
  falling; the residual bias on the synthetic profiles is +0.2% to +1.2%.
