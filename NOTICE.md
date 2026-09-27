# Third-party notices

This repository is MIT-licensed ([LICENSE](LICENSE)). Components taken from other
projects keep their own licences; each is listed here with its source, what was
taken and where its licence text lives, as units land. [docs/REUSE.md](docs/REUSE.md)
records the commit and file digest of every copied unit.

Model weights and draft models are not included; they are published by their
authors under their own terms.

## mimo26f-afd: MIT

- **Source:** <https://github.com/hughmadden/mimo26f-afd> v1.2.0 (`bab9fa2`). Copyright (c) 2026 Turquoise Bay AI Pty Ltd.
- **What was taken:** the wire codec (`crates/glm53f-wire`), the RDMA RC transport (`crates/glm53f-rdma`), and the OpenAI-compatible API with its MiMo tool-call parser as the reference dialect and that parser's goldens (`crates/glm53f-api`). Each crate's `PROVENANCE.md` lists the files, digests and changes.
- **Licence:** the same MIT terms and holder as this repository's [LICENSE](LICENSE).

## DS41RT: MIT

- **Source:** <https://github.com/tpurtell/ds41rt>. Copyright (c) 2026 T.J. Purtell.
- **What was taken, through mimo26f-afd:** the designs of the `DS41RTE3` v3 wire format and the RDMA transport (reimplemented there, no code copied), and the sampling-parameter validation ranges in `crates/glm53f-api/src/types.rs` (transcribed from v15). No DS41RT source file is copied yet.

## rdma-core headers: GPL-2.0 or OpenIB.org BSD

- **Source:** the verbs headers in `crates/glm53f-rdma/native/include/` come from linux-rdma/rdma-core (via mimo26f-afd).
- **Licence:** dual-licensed; used here under the OpenIB.org BSD licence. Each file's header carries its copyright notices and the licence text.
