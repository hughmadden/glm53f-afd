# Provenance ledger

Every unit copied or transcribed from another project gets a row here before it
lands: the source repository and commit, the source path, the file digest, where
it lives in this repository, what changed, and the tests that pin it. Units that
are reimplemented from a description (no code copied) are recorded too, marked
"reimplemented". Crate-level notes live in each crate's `PROVENANCE.md` and are
consolidated here.

| Unit | Source (repo @ commit : path) | sha256 (source file) | Here | Delta | Pinned by | Date |
|---|---|---|---|---|---|---|
