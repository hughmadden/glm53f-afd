//! `glm53f-wire` — DS41RTE3 v3 wire codec + L4 integrity ladder (I4 items
//! 5+6; ADVISOR-I4 §3.2 items 5-6 and §3.3, ARCHITECTURE.md §9 item 2 +
//! §11.5).
//!
//! # The two row contracts (pinned by tests, not comments)
//!
//! * **Request row = 4,360 B** per token per layer per Spark: 40-B row
//!   descriptor + 4,224-B `Fp8E4m3Ue8m0K32` hidden row (4,096 FP8 E4M3 payload
//!   + 128 per-token UE8M0 K32 scales) + 8 x 12-B route entries (row index,
//!   expert id, gate weight). 4,360 x 42 MoE layers x 4 Sparks = 0.73 MB
//!   out/token for GLM-5.3-Flash.
//! * **Compact return = 8,192 B BF16** per token per layer per Spark: the
//!   Spark pre-sums its 8 weighted route partials into one 4,096-wide BF16
//!   row; the coordinator sums the 4 partials in FP32. 8,192 x 42 x 4 = 1.38
//!   MB in/token. **Never per-route FP32 returns** — that is the 24.6 MB/token
//!   wall (ADVISOR-I4 A6); `tests/wire_layout.rs` pins route-count
//!   independence of the return size.
//!
//! Expert ids travel as `u32` route-entry fields, so GLM-5.3-Flash's 288
//! routed experts need no layout change (`tests/glm_experts.rs`).
//!
//! # Format derivation (copy-in; `PROVENANCE.md`)
//!
//! Byte-level formats derive from ds41rt-transport `protocol_v2.rs` +
//! `request.rs` (READ-ONLY source: tpurtell/ds41rt v10,
//! `rust/crates/ds41rt-transport/src/`): magic `DS41RTE3` v3 (`protocol_v2.rs:6-7`),
//! 40-B row descriptor + 12-B route entry lengths and byte layouts
//! (`:11-16`, `:580-620`), dtype wire codes `:45-60`, source kinds `:122-128`,
//! `Fp8E4m3Ue8m0K32` = elements + elements/32 (`:83-90`), request sectioning
//! (descriptors | routes | hidden payload, `request.rs:364-410`). No code body
//! is copied verbatim from ds41rt — mimo26f-afd's `docs/REUSE.md` records that
//! derivation. Magic stays `DS41RTE3` (ARCHITECTURE.md §9 item 2 default "keep").
//!
//! Deltas vs ds41rt v3 (we own both ends): unified 128-B header for both
//! directions; L4 tail (seq @112, CRC32C @120 replacing the SHA-256 debug
//! checksum); frame-level `source_kind` / `token_position` / `executor_id`;
//! bare BF16 return rows (no descriptors — the 8,192 B/token contract); return
//! frames must carry `SPARK_REDUCTION | V41_COMPACT_BF16` or decode fails loud.
//!
//! # L4 ladder (CRC32C + sequence per frame)
//!
//! See [`l4`]: bitflip / reorder / truncation / duplicate are detected and
//! retried (bounded, then fail loud) or dropped idempotently — never silently
//! summed. [`CoordinatorSum`] keys partials by header identity and refuses
//! duplicate slots, so corruption cannot land in the FP32 sum undetected.
//!
//! # Version 4: the prefill reduce-scatter
//!
//! See [`row_shard`]: a reduce-scattered request, a row-slice return and the
//! rank-to-rank exchange frame, on the same header and L4 tail. Only the frames
//! that use the extension are version 4, so a version-3 peer refuses exactly
//! those, by version, and keeps working for everything else.
//! [`CoordinatorSum::row_sharded`] assembles the row slices.
//!
//! # Naive discipline (suite convention)
//!
//! Trap wrong-implementation switches live in [`naive`]; NEGATIVE tests call
//! the `*_env` entry points and FAIL under `GLM53F_SPIKE_NAIVE=1` (alias
//! `GLM53F_WIRE_NAIVE=1`); BOTH-RUNS tests pass explicit flags. Classification
//! is listed in `tests/wire_layout.rs` and `tests/l4_integrity.rs`.

pub mod async_api;
pub mod bf16;
pub mod crc32c;
pub mod error;
pub mod frame;
pub mod l4;
pub mod layout;
pub mod naive;
pub mod row_shard;

pub use async_api::{ExpertClient, Ticket, Transport};
pub use error::{Disposition, WireError};
pub use frame::{
    decode_frame, decode_frame_env, encode_request, encode_request_env, encode_return,
    encode_return_env, frame_seq, Frame, HiddenRow, RequestFrame, ReturnFrame, ReturnRow,
    RouteEntry, RowDescriptor, FLAG_EXCHANGE_FP8, FLAG_REDUCE_SCATTER, FLAG_RETURN_REQUIRED,
    FLAG_ROW_SLICE, FLAG_SPARK_REDUCTION, FLAG_V41_COMPACT_BF16,
};
pub use l4::{retry_until, CoordinatorSum, SlotKey, StreamReceiver, StreamSender};
pub use row_shard::{row_partition, ExchangeDtype, ExchangeHeader, ExchangeView};
pub use layout::{
    request_row_bytes, request_row_bytes_env, return_row_bytes, return_row_bytes_env, Dtype,
    SourceKind, Status, HIDDEN, HIDDEN_ROW_BYTES, HEADER_LEN, REQUEST_ROW_BYTES, RETURN_ROW_BYTES,
    ROW_DESCRIPTOR_LEN, ROUTE_ENTRY_LEN, SPARKS, TOPK,
};
pub use naive::{naive_from_env, WireNaive};

// Static contract guards: the build breaks if the row sizes drift from the
// ADVISOR-I4 numbers (runtime twins live in tests/wire_layout.rs).
const _: [(); 4360] = [(); REQUEST_ROW_BYTES];
const _: [(); 8192] = [(); RETURN_ROW_BYTES];
