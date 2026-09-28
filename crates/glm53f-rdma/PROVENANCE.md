# glm53f-rdma provenance

Copied from [hughmadden/mimo26f-afd](https://github.com/hughmadden/mimo26f-afd)
v1.2.0, commit `bab9fa2f2fc1e22ae67b56fbc1c209278f6a9d79` (MIT), directory
`crates/mimo26-rdma`. The sha256 column is the digest of the file at that commit.

**Renames applied to every Rust and C file** (not repeated in the Delta column):
`mimo26-rdma` → `glm53f-rdma`, `mimo26_rdma` → `glm53f_rdma` (so the static
library is `libglm53f_rdma.a`), and the environment prefix `MIMO26_` → `GLM53F_`
(`GLM53F_WIRE_MIN_GBPS`, `GLM53F_WIRE_ALLOW_LAN`, `GLM53F_RDMA_TIMEOUT`, and
`GLM53F_SPARK_ADDRS` in an error message).

**Kept on purpose:** the handshake magic `M26RDMA1` that the coordinator writes on
the TCP socket before the queue-pair exchange (a protocol value shared with the
rank daemon, like the frame magic), the C symbol prefix `m26r_`, and the `rdma`
cargo feature: without it the native shim is not built and every entry point
returns an error, so the default build needs no RDMA stack.

**Third-party code:** the headers under `native/include/` are rdma-core 50's verbs
headers, dual-licensed GPL-2.0 or OpenIB.org BSD and used under the BSD licence.
Each file's header carries its copyright notices and the licence text; the source
crate carries no separate licence file.

| Unit | Source (repo @ commit : path) | sha256 (source file) | Here | Delta | Pinned by | Date |
|---|---|---|---|---|---|---|
| Manifest (feature `rdma`) | mimo26f-afd @ bab9fa2 : crates/mimo26-rdma/Cargo.toml | `b6559283b4a96eb2c5c45b631e71c508049e882e5139e9753829292698840764` | Cargo.toml | Header comment names mimo26f-afd for the design document and points to this ledger | build (default and `--features rdma`) | 2026-09-28 |
| Build script (native shim, libibverbs link) | mimo26f-afd @ bab9fa2 : crates/mimo26-rdma/build.rs | `9101187827d3a9b7946740f5026243366c7913ba21ff5d2b2e7097344051d1f7` | build.rs | Renames only | `cargo build -p glm53f-rdma --features rdma` | 2026-09-28 |
| Verbs RC shim | mimo26f-afd @ bab9fa2 : crates/mimo26-rdma/native/rdma.c | `93ad637d3349dd73d636c0b7d189abbccdae79d8505a9ce722e93c880c64fd31` | native/rdma.c | Header comment names the source project; the measurement-run reference is marked as mimo26f-afd's. No code change beyond the environment name | `--features rdma` build (compiles clean with `-Wall -Wextra`) | 2026-09-28 |
| Endpoint, RoCE v2 GID lookup, fabric guard | mimo26f-afd @ bab9fa2 : crates/mimo26-rdma/src/lib.rs | `6c9f1632805aeb1236f89c7209ab91908b5345b238906d124bc0686b26476e2e` | src/lib.rs | The fabric-guard doc no longer names a deployment's API proxy; unit test `parses_ipv4_mapped_gid` uses 192.0.2.3 (documentation range) in place of a private address. Receive slots in the handshake (written here, 2026-09-29): the `Info` word the source reserved (zero, not serialized) carries the receive slots a side posts (`recv_slots`, serialized at bytes 28-31; `peer_slots` reads a zero from an older peer as two, `LEGACY_RECV_SLOTS`); the magic, the length and the other fields are unchanged, and the C shim's struct keeps its `reserved` name | unit tests `info_round_trips`, `a_handshake_without_receive_slots_means_two`, `parses_ipv4_mapped_gid` | 2026-09-28 |
| rdma-core header | mimo26f-afd @ bab9fa2 : crates/mimo26-rdma/native/include/infiniband/verbs.h | `fab5bd6f7d17b23f9afac2b4e4ad06315f964a69d695d0ea06402e8858adb8f2` | native/include/infiniband/verbs.h | Verbatim | `--features rdma` build | 2026-09-28 |
| rdma-core header | mimo26f-afd @ bab9fa2 : crates/mimo26-rdma/native/include/infiniband/verbs_api.h | `6d37f6f7f417e87cc2c3ba8c0220d6d56f11b54ee4665ce8553500d65d076554` | native/include/infiniband/verbs_api.h | Verbatim | `--features rdma` build | 2026-09-28 |
| rdma-core header | mimo26f-afd @ bab9fa2 : crates/mimo26-rdma/native/include/infiniband/ib_user_ioctl_verbs.h | `455b073d4a7f57d73bc24a4f9a457acf62458a2647d31f4be5b05da20f9ad34b` | native/include/infiniband/ib_user_ioctl_verbs.h | Verbatim | `--features rdma` build | 2026-09-28 |
| rdma-core header | mimo26f-afd @ bab9fa2 : crates/mimo26-rdma/native/include/infiniband/tm_types.h | `c15d540f2ad545c2ebcda8eba2424d31e5bc2999e571809f9f1ad61ccb892406` | native/include/infiniband/tm_types.h | Verbatim | `--features rdma` build | 2026-09-28 |
| rdma-core header | mimo26f-afd @ bab9fa2 : crates/mimo26-rdma/native/include/rdma/ib_user_verbs.h | `7e32869c330291f4ccda8073c57fab2b7df898546eaba16062ee555ee964f861` | native/include/rdma/ib_user_verbs.h | Verbatim | `--features rdma` build | 2026-09-28 |

## Written here

| File | What | Date |
|---|---|---|
| PROVENANCE.md | This ledger. | 2026-09-28 |
