//! L4 integrity ladder — CRC32C + sequence number per frame (ADVISOR-I4 §3.2
//! item 6): bitflip, reorder, truncation and duplicate are all detected and
//! retried or fail loud, and none of them may produce a silent wrong sum.
//!
//! Policy ([`crate::error::Disposition`]):
//! * Corrupt / Truncated / OutOfOrder -> **Retry**: bounded retransmission of
//!   the expected sequence ([`retry_until`]); budget exhaustion is
//!   [`crate::WireError::RetryExhausted`] = **fail loud**.
//! * Duplicate / SlotFilled -> **DropIdempotent**: detected, reported, and
//!   never accumulated twice (two layers: sequence policy + coordinator slots).
//! * Protocol/layout mismatches -> **fail loud** immediately.
//!
//! The coordinator FP32 sum ([`CoordinatorSum`]) keys partials by header
//! identity `(request_id, layer_id, token_position, executor_id)` — never by
//! arrival order — so a reordered delivery cannot land a partial in the wrong
//! token's sum.

use crate::error::{Disposition, WireError};
use crate::frame::{self, Frame, RequestFrame, ReturnFrame};
use crate::naive::WireNaive;

/// Identity of one Spark partial row in a coordinator sum window.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SlotKey {
    pub request_id: u64,
    pub layer_id: u32,
    pub token_position: u64,
    pub executor_id: u64,
}

/// Sender side of the ladder: stamps monotonically increasing sequence numbers.
pub struct StreamSender {
    naive: WireNaive,
    next_seq: u64,
}

impl StreamSender {
    pub fn new(naive: WireNaive) -> Self {
        Self { naive, next_seq: 0 }
    }

    pub fn new_env() -> Self {
        Self::new(crate::naive::naive_from_env())
    }

    pub fn next_seq(&self) -> u64 {
        self.next_seq
    }

    /// Consume the next L4 sequence without encoding (a caller that stamps a
    /// shared frame body's header itself, perf reset R2 RDMA path).
    pub fn take_seq(&mut self) -> u64 {
        let seq = self.next_seq;
        self.next_seq += 1;
        seq
    }

    /// The naive (trap) flags this sender encodes with.
    pub fn naive(&self) -> WireNaive {
        self.naive
    }

    pub fn encode_request(&mut self, f: &RequestFrame) -> Result<Vec<u8>, WireError> {
        let seq = self.next_seq;
        self.next_seq += 1;
        frame::encode_request_seq(f, seq, self.naive)
    }

    pub fn encode_return(&mut self, f: &ReturnFrame) -> Result<Vec<u8>, WireError> {
        let seq = self.next_seq;
        self.next_seq += 1;
        frame::encode_return_seq(f, seq, self.naive)
    }
}

/// Receiver side of the ladder: decode (CRC32C) + strict sequence policy.
pub struct StreamReceiver {
    naive: WireNaive,
    expected: u64,
}

impl StreamReceiver {
    pub fn new(naive: WireNaive) -> Self {
        Self { naive, expected: 0 }
    }

    pub fn new_env() -> Self {
        Self::new(crate::naive::naive_from_env())
    }

    /// Next sequence this receiver will accept.
    pub fn expected(&self) -> u64 {
        self.expected
    }

    /// Accept one frame: checksum verify, then
    /// `seq == expected` -> accept and advance;
    /// `seq <  expected` -> [`WireError::Duplicate`] (detected, idempotent drop);
    /// `seq >  expected` -> [`WireError::OutOfOrder`] (gap: retry the expected).
    /// The sequence discipline of [`accept`] alone, for a caller that validated
    /// the frame itself (the coordinator's zero-copy return path, perf reset R2).
    pub fn accept_seq(&mut self, seq: u64) -> Result<(), WireError> {
        if self.naive.has(WireNaive::SEQ_IGNORED) {
            return Ok(());
        }
        if seq == self.expected {
            self.expected += 1;
            Ok(())
        } else if seq < self.expected {
            Err(WireError::Duplicate { seq, expected: self.expected })
        } else {
            Err(WireError::OutOfOrder { expected: self.expected, got: seq })
        }
    }

    pub fn accept(&mut self, bytes: &[u8]) -> Result<Frame, WireError> {
        let f = frame::decode_frame(bytes, self.naive)?;
        let seq = f.seq();
        if self.naive.has(WireNaive::SEQ_IGNORED) {
            // TRAP: no ordering discipline at all — reorders and duplicates
            // pass silently.
            return Ok(f);
        }
        if seq == self.expected {
            self.expected += 1;
            Ok(f)
        } else if seq < self.expected {
            Err(WireError::Duplicate { seq, expected: self.expected })
        } else {
            Err(WireError::OutOfOrder { expected: self.expected, got: seq })
        }
    }
}

/// Bounded retry driver: Retry-class errors re-run `op` up to `max_attempts`;
/// exhaustion is [`WireError::RetryExhausted`] (fail loud). DropIdempotent and
/// FailLoud errors return immediately.
pub fn retry_until<T, F>(max_attempts: u32, mut op: F) -> Result<T, WireError>
where
    F: FnMut() -> Result<T, WireError>,
{
    let mut attempts = 0u32;
    loop {
        match op() {
            Ok(v) => return Ok(v),
            Err(e) => match e.disposition() {
                Disposition::Retry if attempts + 1 < max_attempts => {
                    attempts += 1;
                }
                Disposition::Retry => {
                    return Err(WireError::RetryExhausted { attempts, cause: Box::new(e) })
                }
                _ => return Err(e),
            },
        }
    }
}

/// Coordinator-side FP32 accumulation of the 4 Spark partials (ARCHITECTURE.md
/// §11.5: one compact BF16 row per token per Spark, summed on the coordinator).
/// Exactly one partial per `(request_id, layer_id, token_position, executor)`
/// slot is accepted — a duplicate is detected at the slot layer even if the
/// sequence layer were bypassed.
///
/// R8 (ADVISOR-I4:487): the four rank partials for a token are buffered per
/// rank and summed in fixed rank order `0 → 3`, so the FP32 result is
/// bit-reproducible regardless of delivery order — `[2^24, 1, −2^24, 1]` sums
/// to 1 in rank order but 2 if the middle two arrive first.
pub struct CoordinatorSum {
    naive: WireNaive,
    rows: usize,
    hidden: usize,
    seen: Vec<SlotKey>,
    distinct: usize,
    acc: Vec<f32>,
    arrivals: usize,
    /// R8: per-rank buffered partials (`SPARKS * rows * hidden`), summed into
    /// `acc` in rank order once a token slot has all `SPARKS` ranks.
    rank_buf: Vec<f32>,
    /// Distinct ranks landed per token slot (drives the rank-ordered sum).
    slot_rank_count: Vec<u8>,
}

impl CoordinatorSum {
    /// `rows` token rows x `hidden` elements, `SPARKS` partials per row.
    pub fn new(rows: usize, hidden: usize, naive: WireNaive) -> Self {
        Self {
            naive,
            rows,
            hidden,
            seen: Vec::new(),
            distinct: 0,
            acc: vec![0.0; rows * hidden],
            arrivals: 0,
            rank_buf: vec![0.0; crate::layout::SPARKS * rows * hidden],
            slot_rank_count: vec![0u8; rows],
        }
    }

    pub fn new_env(rows: usize, hidden: usize) -> Self {
        Self::new(rows, hidden, crate::naive::naive_from_env())
    }

    /// Distinct slots filled.
    pub fn filled(&self) -> usize {
        self.distinct
    }

    pub fn is_complete(&self) -> bool {
        self.distinct == self.rows * crate::layout::SPARKS
    }

    /// Accumulate one accepted return frame (its rows as BF16 -> FP32).
    pub fn accumulate(&mut self, f: &ReturnFrame) -> Result<(), WireError> {
        if f.status != crate::layout::Status::Ok {
            return Err(WireError::SparkReportedError { executor_id: f.executor_id });
        }
        // R8: a rank outside 0..SPARKS cannot be placed in the rank-ordered
        // buffer — refuse it before any indexed write.
        if f.executor_id >= crate::layout::SPARKS as u64 {
            return Err(WireError::DimMismatch {
                field: "executor_id",
                want: crate::layout::SPARKS,
                got: f.executor_id as usize,
            });
        }
        let rank = f.executor_id as usize;
        for (r, row) in f.rows.iter().enumerate() {
            let token = f
                .token_position
                .checked_add(r as u64)
                .ok_or(WireError::DimMismatch {
                    field: "token_position",
                    want: self.rows,
                    got: usize::MAX,
                })?;
            if token as usize >= self.rows {
                return Err(WireError::DimMismatch {
                    field: "token_position",
                    want: self.rows,
                    got: token as usize,
                });
            }
            if row.codes.len() != self.hidden {
                return Err(WireError::DimMismatch {
                    field: "return_row_codes",
                    want: self.hidden,
                    got: row.codes.len(),
                });
            }
            let key = SlotKey {
                request_id: f.request_id,
                layer_id: f.layer_id,
                token_position: token,
                executor_id: f.executor_id,
            };
            let is_dup = self.seen.contains(&key);
            if is_dup && !self.naive.has(WireNaive::DUP_DOUBLE_COUNT) {
                // Layer 2 idempotence: never double-count a partial.
                return Err(WireError::SlotFilled { slot: key });
            }
            let slot_row = if self.naive.has(WireNaive::ACCUM_BY_ARRIVAL) {
                // TRAP: arrival-order matching — a reordered delivery lands
                // partials in the wrong token's sum.
                (self.arrivals / crate::layout::SPARKS) % self.rows
            } else {
                token as usize
            };
            self.arrivals += 1;
            self.seen.push(key);
            if !is_dup {
                self.distinct += 1;
                self.slot_rank_count[slot_row] += 1;
            }
            let base = slot_row * self.hidden;
            if self.naive.has(WireNaive::RANK_SUM_BY_ARRIVAL) {
                // TRAP: arrival-order summation — the FP32 result depends on
                // delivery order (R8). Direct add, no rank buffering.
                for (i, &code) in row.codes.iter().enumerate() {
                    self.acc[base + i] += crate::bf16::bf16_to_f32(code);
                }
            } else {
                // R8: buffer this rank's partial, then — once all `SPARKS`
                // ranks for the slot have landed — sum them in rank order.
                let rbase = rank * (self.rows * self.hidden) + base;
                for (i, &code) in row.codes.iter().enumerate() {
                    self.rank_buf[rbase + i] += crate::bf16::bf16_to_f32(code);
                }
                // Re-sum once the slot has all `SPARKS` ranks. When
                // `DUP_DOUBLE_COUNT` admits a duplicate, the duplicate writes
                // into the buffer and the re-sum (still triggered, the count
                // is already `SPARKS`) folds the double-count into `acc`.
                if self.slot_rank_count[slot_row] as usize == crate::layout::SPARKS {
                    for i in 0..self.hidden {
                        let mut s = 0.0f32;
                        for rr in 0..crate::layout::SPARKS {
                            s += self.rank_buf[rr * (self.rows * self.hidden) + base + i];
                        }
                        self.acc[base + i] = s;
                    }
                }
            }
        }
        Ok(())
    }

    /// The FP32 sum once every slot is filled (else fail loud `Incomplete`).
    pub fn result(&self) -> Result<&[f32], WireError> {
        let need = self.rows * crate::layout::SPARKS;
        if self.distinct != need {
            return Err(WireError::Incomplete { filled: self.distinct, need });
        }
        Ok(&self.acc)
    }
}
