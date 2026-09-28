# Third-party notices

This repository is MIT-licensed ([LICENSE](LICENSE)). Components taken from other
projects keep their own licences; each is listed here with its source, what was
taken and where its licence text lives, as units land. [docs/REUSE.md](docs/REUSE.md)
records the commit and file digest of every copied unit.

Model weights and draft models are not included; they are published by their
authors under their own terms.

## mimo26f-afd: MIT

- **Source:** <https://github.com/hughmadden/mimo26f-afd> v1.2.0 (`bab9fa2`). Copyright (c) 2026 Turquoise Bay AI Pty Ltd.
- **What was taken:** the wire codec (`crates/glm53f-wire`), the RDMA RC transport (`crates/glm53f-rdma`), the OpenAI-compatible API with its MiMo tool-call parser as the reference dialect and that parser's goldens (`crates/glm53f-api`), and the BPE tokenizer, adapted (`crates/glm53f-tokenizer`). Each crate's `PROVENANCE.md` lists the files, digests and changes.
- **Licence:** the same MIT terms and holder as this repository's [LICENSE](LICENSE).

## DS41RT: MIT

- **Source:** <https://github.com/tpurtell/ds41rt>. Copyright (c) 2026 T.J. Purtell.
- **What was taken, through mimo26f-afd:** the designs of the `DS41RTE3` v3 wire format and the RDMA transport (reimplemented there, no code copied), and the sampling-parameter validation ranges in `crates/glm53f-api/src/types.rs` (transcribed from v15).
- **What was taken directly** (ds41rt @ `3067d06`):
  - the mHC Sinkhorn, collapse and expansion kernels, the router logits and top-k selection, and the split-K reduction, adapted in `crates/glm53f-layers/kernels/`;
  - the sparse MLA attention kernel structure and the selection-key encoding, adapted in `crates/glm53f-dsa/kernels/`.
  Each crate's `PROVENANCE.md` lists the lines and changes.
- **Licence text:** `crates/glm53f-layers/LICENSE.ds41rt`, `crates/glm53f-dsa/LICENSES/ds41rt-MIT.txt`.

## b12x / SparkInfer fork: Apache License 2.0

- **Source:** <https://github.com/tpurtell/sparkinfer-glmrt> @ `7fcc094e` (a fork of <https://github.com/local-inference-lab/b12x>).
- **What was taken:** the 528-byte GLM-5.3-Flash MLA latent record layout (512 E4M3 values and four f32 scales), for format compatibility (`crates/glm53f-dsa/src/cache.rs`).
- **Licence text:** `crates/glm53f-dsa/LICENSES/Apache-2.0.txt`.

## rdma-core headers: GPL-2.0 or OpenIB.org BSD

- **Source:** the verbs headers in `crates/glm53f-rdma/native/include/` come from linux-rdma/rdma-core (via mimo26f-afd).
- **Licence:** dual-licensed; used here under the OpenIB.org BSD licence. Each file's header carries its copyright notices and the licence text.

## TensorFold: MIT

- **Source:** <https://github.com/ashhart/TensorFold> @ `bb4b4a3`. Copyright (c) 2026 TensorFold contributors (as in its licence).
- **What was taken:**
  - the fused KDA chain, replay and conv-shift kernels for GLM-5.3-Flash (`crates/glm53f-kda/kernels/`), ported to a C ABI with batching; the source file is kept verbatim for a bitwise parity test;
  - the EXL3 trellis tile decoder, fragment MMA and Hadamard butterfly (`crates/glm53f-rank/kernels/exl3_rank.cu`), and the EXL3 reference decoder, ported to Rust (`crates/glm53f-rank/src/exl3.rs`).
- **Licence text:** `crates/glm53f-kda/LICENSE.tensorfold`, `crates/glm53f-rank/LICENSE.tensorfold`.

## ExLlamaV3: MIT

- **Source:** <https://github.com/turboderp-org/exllamav3>.
- **What was taken:** nothing is copied. The EXL3 trellis format, codebook and Hadamard scheme that the rank's kernels read originate there, through TensorFold's implementation.
