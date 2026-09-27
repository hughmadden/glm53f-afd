//! Naive-implementation switch — the two-run trap suite convention
//! (same discipline as `mimo26-load::naive_from_env`).
//!
//! A `WireNaive` bit selects ONE known-wrong implementation ("trap"). The
//! correct implementation is always [`WireNaive::NONE`]. NEGATIVE tests call
//! the `*_env` entry points so they FAIL when the wrong implementation is
//! selected (`GLM53F_SPIKE_NAIVE=1`, alias `GLM53F_WIRE_NAIVE=1` selects
//! [`WireNaive::ALL`]); BOTH-RUNS tests pass explicit flags and pass both runs.
//! NAIVE DISCIPLINE: wrong implementations live in this crate behind these
//! flags, never behind test-side mocks.

/// Bit flags selecting known-wrong wire implementations (traps).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WireNaive(pub u32);

impl WireNaive {
    /// The correct implementation.
    pub const NONE: WireNaive = WireNaive(0);
    /// Every trap at once (the naive run).
    pub const ALL: WireNaive = WireNaive(u32::MAX);

    /// Scale-grid slip: UE8M0 scales every 16 values instead of every 32
    /// (T10-adjacent; the classic K-block misalignment).
    pub const K16_SCALES: u32 = 1 << 0;
    /// Wrong dtype variant: `Fp8E4m3RowScaled` (one FP32 scale per row)
    /// instead of `Fp8E4m3Ue8m0K32`.
    pub const ROW_SCALED_DTYPE: u32 = 1 << 1;
    /// Compact return in FP32 (16,384 B/token) instead of BF16 (8,192 B).
    pub const FP32_RETURN: u32 = 1 << 2;
    /// Per-route return rows instead of one pre-summed partial (A6 wall).
    pub const PER_ROUTE_RETURN: u32 = 1 << 3;
    /// Route entry `expert_id`/`gate_weight` fields swapped on the wire.
    pub const ROUTE_FIELD_SWAP: u32 = 1 << 4;
    /// Row descriptor field drift (`source_request_id`/`token_position` swap).
    pub const DESC_FIELD_DRIFT: u32 = 1 << 5;
    /// Dtype wire codes renumbered (enum-order drift).
    pub const DTYPE_RECODE: u32 = 1 << 6;
    /// CRC-32 (IEEE) instead of CRC32C (Castagnoli).
    pub const CRC32_IEEE: u32 = 1 << 7;
    /// Checksum covers the body only — header fields ride unprotected.
    pub const CRC_HEADER_EXEMPT: u32 = 1 << 8;
    /// Decode skips checksum verification ("trust the transport").
    pub const CRC_UNCHECKED: u32 = 1 << 9;
    /// Receiver accepts any sequence — reorders and duplicates pass silently.
    pub const SEQ_IGNORED: u32 = 1 << 10;
    /// Coordinator accumulates duplicate partials twice (no slot idempotence).
    pub const DUP_DOUBLE_COUNT: u32 = 1 << 11;
    /// Coordinator matches partials by arrival order, not header identity.
    pub const ACCUM_BY_ARRIVAL: u32 = 1 << 12;
    /// f32 -> BF16 truncation instead of round-to-nearest-even.
    pub const BF16_TRUNCATE: u32 = 1 << 13;
    /// Coordinator sums the four rank rows in arrival order instead of the
    /// fixed rank order 0 → 3 (R8, ADVISOR-I4:487 — an FP32 sum that depends
    /// on delivery order is not reproducible).
    pub const RANK_SUM_BY_ARRIVAL: u32 = 1 << 14;

    pub const fn has(self, bit: u32) -> bool {
        self.0 & bit != 0
    }

    pub const fn bits(self) -> u32 {
        self.0
    }

    /// Single-bit flag set (for per-trap negatives and meta tests).
    pub const fn of(bit: u32) -> WireNaive {
        WireNaive(bit)
    }
}

/// Env-default naive flag: `GLM53F_SPIKE_NAIVE=1` (suite convention) or the
/// crate alias `GLM53F_WIRE_NAIVE=1` selects [`WireNaive::ALL`].
pub fn naive_from_env() -> WireNaive {
    let on = |key: &str| std::env::var(key).map(|v| v == "1").unwrap_or(false);
    if on("GLM53F_SPIKE_NAIVE") || on("GLM53F_WIRE_NAIVE") {
        WireNaive::ALL
    } else {
        WireNaive::NONE
    }
}
