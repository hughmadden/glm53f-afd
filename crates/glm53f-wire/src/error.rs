//! Wire error taxonomy with the L4 disposition policy (ADVISOR-I4 §3.2 item 6:
//! "every one is detected and retried or fails loud, and none produces a silent
//! wrong sum").
//!
//! Dispositions:
//! * [`Disposition::Retry`] — retransmit the expected sequence and continue
//!   (bounded; exhaustion converts to [`crate::WireError::RetryExhausted`],
//!   which is [`Disposition::FailLoud`]).
//! * [`Disposition::DropIdempotent`] — DETECTED duplicate; dropping it is safe
//!   and it must never be accumulated twice.
//! * [`Disposition::FailLoud`] — protocol/layout mismatch: never paper over it.

use std::fmt;

use crate::l4::SlotKey;

/// What the caller must do about an error.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Disposition {
    /// Retransmit the expected frame; bounded retries then fail loud.
    Retry,
    /// Detected duplicate — drop without side effects, report it.
    DropIdempotent,
    /// Stop now; the stream or frame is not interpretable.
    FailLoud,
}

/// Errors for the DS41RTE3 v3 codec and the L4 ladder.
#[derive(Debug, Clone, PartialEq)]
pub enum WireError {
    /// Fewer bytes than the fixed header (a cut frame fragment).
    TooShort { need: usize, got: usize },
    /// Frame shorter than its declared `wire_bytes` (truncation).
    Truncated { declared: usize, got: usize },
    /// Frame longer than its declared `wire_bytes` (stream desync / garbage).
    TrailingBytes { declared: usize, got: usize },
    /// Not a DS41RTE3 frame.
    BadMagic([u8; 8]),
    /// Wrong frame version (silent cross-version misparse guard).
    BadVersion(u16),
    /// Unknown message kind.
    BadKind(u16),
    /// Header length is not the v3 128-B header.
    BadHeaderLen(u32),
    /// A structural field disagrees with the frame geometry.
    DimMismatch { field: &'static str, want: usize, got: usize },
    /// Unknown enum wire code.
    BadCode { field: &'static str, code: u32 },
    /// Checksum mismatch (bitflip, corruption in flight).
    Corrupt { seq: u64, want: u32, got: u32 },
    /// Sequence already accepted (duplicate/replay).
    Duplicate { seq: u64, expected: u64 },
    /// Sequence gap: earlier frame(s) missing or delivered out of order.
    OutOfOrder { expected: u64, got: u64 },
    /// A route entry points at a row index outside the frame.
    RouteRowMismatch { entry: usize, row_index: u32, rows: u32 },
    /// A row's route slice falls outside the route table.
    RouteRangeBad { row: usize },
    /// Return frame not marked as pre-summed compact BF16 (A6 guard).
    UnpresummedReturn { flags: u32 },
    /// Coordinator slot already filled — a duplicate partial was refused.
    SlotFilled { slot: SlotKey },
    /// Spark reported status = Error for this return.
    SparkReportedError { executor_id: u64 },
    /// Fewer partials than the window needs.
    Incomplete { filled: usize, need: usize },
    /// Retry budget exhausted — the fail-loud end of the ladder.
    RetryExhausted { attempts: u32, cause: Box<WireError> },
    /// A return frame arrived for a request the client is not tracking (I4
    /// item 8 async API) — a protocol error, never silently dropped.
    UnknownRequest { request_id: u64 },
}

impl WireError {
    /// L4 policy class for this error.
    pub fn disposition(&self) -> Disposition {
        use Disposition::*;
        use WireError::*;
        match self {
            TooShort { .. } | Truncated { .. } | Corrupt { .. } | OutOfOrder { .. } => Retry,
            Duplicate { .. } | SlotFilled { .. } => DropIdempotent,
            TrailingBytes { .. }
            | BadMagic(_)
            | BadVersion(_)
            | BadKind(_)
            | BadHeaderLen(_)
            | DimMismatch { .. }
            | BadCode { .. }
            | RouteRowMismatch { .. }
            | RouteRangeBad { .. }
            | UnpresummedReturn { .. }
            | SparkReportedError { .. }
            | Incomplete { .. }
            | RetryExhausted { .. }
            | UnknownRequest { .. } => FailLoud,
        }
    }
}

impl fmt::Display for WireError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        use WireError::*;
        match self {
            TooShort { need, got } => write!(f, "frame shorter than header: need {need}, got {got}"),
            Truncated { declared, got } => {
                write!(f, "frame truncated: declared {declared} bytes, got {got}")
            }
            TrailingBytes { declared, got } => {
                write!(f, "trailing bytes after frame: declared {declared}, got {got}")
            }
            BadMagic(m) => write!(f, "bad frame magic: {m:?}"),
            BadVersion(v) => write!(f, "unsupported frame version {v}"),
            BadKind(k) => write!(f, "unknown message kind {k}"),
            BadHeaderLen(h) => write!(f, "bad header length {h}"),
            DimMismatch { field, want, got } => {
                write!(f, "dimension mismatch in {field}: want {want}, got {got}")
            }
            BadCode { field, code } => write!(f, "unknown wire code {code} for {field}"),
            UnknownRequest { request_id } => {
                write!(f, "return frame for untracked request {request_id}")
            }
            Corrupt { seq, want, got } => {
                write!(f, "CRC32C mismatch on seq {seq}: want {want:#010x}, got {got:#010x}")
            }
            Duplicate { seq, expected } => {
                write!(f, "duplicate frame seq {seq} (expected {expected}) — dropped")
            }
            OutOfOrder { expected, got } => {
                write!(f, "out-of-order frame seq {got} (expected {expected})")
            }
            RouteRowMismatch { entry, row_index, rows } => write!(
                f,
                "route entry {entry} references row {row_index}, frame has {rows} rows"
            ),
            RouteRangeBad { row } => write!(f, "row {row} route slice outside route table"),
            UnpresummedReturn { flags } => write!(
                f,
                "return frame flags {flags:#010x} missing SPARK_REDUCTION|V41_COMPACT_BF16"
            ),
            SlotFilled { slot } => write!(f, "coordinator slot already filled: {slot:?}"),
            SparkReportedError { executor_id } => {
                write!(f, "Spark {executor_id} reported status=Error")
            }
            Incomplete { filled, need } => {
                write!(f, "coordinator sum incomplete: {filled} of {need} partials")
            }
            RetryExhausted { attempts, cause } => {
                write!(f, "retry budget exhausted after {attempts} attempts: {cause}")
            }
        }
    }
}

impl std::error::Error for WireError {}
