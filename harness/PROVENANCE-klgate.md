# KL gate provenance

`klgate.py` and `klgate_fetch.py` were written for this repository. No code is copied from
another project. The KL method they implement follows the published measurements of
GLM-5.3-Flash quantizations; the table below pins the sources it follows. Revisions are Hugging
Face commits unless marked GitHub. Digests are the sha256 of the file bytes as fetched.

## Method sources

| What `klgate.py` follows | Source (repo @ revision : path) | sha256 | Licence (as the source states it) |
|---|---|---|---|
| The per-position kernel: KL(teacher ‖ student) in float64 over float32 logits, row r against row r, argmax top-1; all 154,880 stored columns (the policy of the headline figures) | `brandonmusic/GLM-5.3-Flash-tr3-4bpw` @ `a5fee929cf4888b1824323e33e8a19b60129e025` : `scripts/measure_glm53_packed_student_kld.py` | `3948b677a8ad31b99e74a1b38e0b7a48b2503cf9c6b39e40b2b68aed2d7bfdbb` | model card: `license: other`, `license_name: shapleymcg-license-1.0` |
| The same kernel for the runtime (window `final-0000`) figures | same repo @ same revision : `scripts/measure_glm53_tp_runtime_window_kld.py` | `460062c3171e13bb9b8d2e2f5457ff9966ddbddc72c62a041c69579d9cea6b4d` | as above |
| The masked-vocabulary policy (154,856 tokenizer ids on both sides), per-position diagnostics | same repo @ same revision : `eval/kld/kld_eval/kld/core.py` | `752af6740595791f300cf47f15ef81f8ca85245d5282fc600ef26ee57eae41e6` | as above |
| Statistics: clustered SE with windows as clusters, window-level block bootstrap (B = 5000, seed 20260829) with percentile and BCa intervals (jackknife acceleration), the 100-exceedance rule for quantiles, paired comparison with McNemar on top-1 | same repo @ same revision : `eval/kld/kld_eval/analysis/stats.py` | `3b6547aeb1a9ac97ef39b40e9bec34fe6964758c9e4fb7c84eed7e9dbb5307af` | as above |
| Scoring protocol: context 2,048, 2,047 positions per window, one window per forward with fresh state, no MTP, no prefix cache, no sampling | same repo @ same revision : `eval/kld/protocol/protocol.yaml` | `4d1d91adbd42621824c3df4e7e4d4a1fdfa85dcbd5072411e6289ea453d88106` | as above |
| Canaries: self-KL exactly 0, a one-row shift at least 3x the teacher's mean entropy, teacher top-1 equal to the next token in [0.2, 0.995] of a window's positions; position buckets | `malaiwah/quant-fidelity-registry` (dataset) @ `394b64750b55c325899c5cd121a415cc6fc99c15` : `protocol/glm53-joint-kld-protocol.v1.json` | `80df521eb46fba68538dd90aa3f2baf22b1e440b8b560555646ff9bbeb35961b` | `cc-by-4.0` |
| The governing report (direction, token-weighted mean, cluster bootstrap, paired differences, padded columns, no serving-API logprobs) | GitHub `brandonmmusic-max/glm-5.3-flash-exl3-4bpw` @ `24784d718cb7bbb99feaee2050eaddcfbc913386` : `kld quantization fidelity report.md` | `692ff9e50bc70e716f1a94f1d9a4f3fb2c6d797f639dc8da84b17b069a20b9fc` | GitHub reports the licence as "Other" |

## Differences from the sources

- **Standard library only.** The sources use numpy, torch and scipy; `klgate.py` computes the
  same sums with `math.fsum` in float64 and draws the bootstrap from Python's `random` with the
  same B and seed, so its interval endpoints differ from numpy's generator at Monte Carlo
  precision. The self-test checks the kernel against closed-form values and an independent
  two-pass computation (agreement to 1e-12 or better). On the 4,725 fetched teacher rows against
  a stand-in engine, its per-row KL matched torch's float64 `log_softmax` computation (the
  arithmetic of the scripts above, run on the CPU in the oracle image) to 1.8e-13 nats, with the
  same top-1 on every row.
- **Row subsets.** The sources score every position. `klgate.py` also accepts a subset of each
  window's positions and weights every scored row by the positions it stands for (the nearest
  scored row takes each position); with every position scored the weights are 1 and the
  estimate is the sources' token mean. It adds the standard error of that subsample.
- **Top-1 argmax** is taken over the scored columns: all 154,880 by default, as the scripts
  above; the 154,856 tokenizer ids with `--vocab tokenizer`, as `core.py`.

## Data

| Data | Source | Licence (as the source states it) |
|---|---|---|
| BF16 teacher logits, token panel and receipts | `brandonmusic/GLM-5.3-Flash-BF16-Teacher-Logits` (dataset) @ `95f4fdd94bf29989db2e0d1054e4931f55edb6aa` | dataset card: `license: other` |
| Teacher model | `zai-org/GLM-5.3-Flash-BF16` @ `a6c167b62691b2bac901344b65cb651a70f53e43` (recorded in the dataset's receipts) | model card: MIT |

The 125-window panel of `docs/KL-GATE.md` section 6c takes 100 more windows (roles `confirmation`
and `selection`) from the same dataset at the same revision, listed in
`logits/full-panel/full-panel-manifest.json` (file sha256
`c0c70608c6436324852732720afa6d060e7e220f2a6ce2945e8fe63ba8b99a8f`).

Nothing from these datasets is stored in this repository. `klgate_fetch.py` downloads a subset
to a directory given on its command line.
