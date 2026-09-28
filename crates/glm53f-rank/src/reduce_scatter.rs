//! Prefill return path: the reduce-scatter among the four ranks. This module is
//! the protocol's arithmetic and bookkeeping (partition, exchange frames, the
//! sum in rank order); the peer mesh (`crate::mesh`) moves the frames, and
//! `tests/reduce_scatter.rs` runs all four ranks on the CPU against the
//! four-plane sum. The design (when it runs, frames, error bounds, traffic) is
//! in README.md, "Prefill reduce-scatter"; the frames are the wire's version-4
//! exchange frames (`glm53f_wire::row_shard`).
//!
//! After glmrt v9 (`glmrt-daemon` `real_full/rdma_reduction.rs`,
//! `intermediate_sharding.rs::balanced_row_partition`, and the row-scaled FP8
//! kernels `bf16_rows_to_fp8_e4m3_row_scaled` /
//! `combine_bf16_fp8_e4m3_row_scaled_to_fp8` in `native/cuda/kernels/residual.cu`;
//! MIT). Reimplemented, no code copied. Two differences: the reduced rows go
//! back to the coordinator as the wire's BF16 rows (glmrt re-quantizes them to
//! FP8), and the exchange dtype is a choice ([`ExchangeDtype`]): FP8 row-scaled
//! as glmrt, or BF16 (README.md explains why BF16 is the default).
//!
//! Per reduce-scattered request (the coordinator sets the request flag from 16
//! rows) and for each rank `r`:
//!
//! 1. compute the full partial `P_r` [rows, 4096] in FP32 (the kernel's reduce
//!    output before its BF16 rounding, `ExpertKernel::ffn_f32`);
//! 2. rows are partitioned `[start_q, start_q + count_q)` over the four ranks
//!    ([`row_partition`]); send each peer `q` its rows of `P_r` in the exchange
//!    dtype ([`Exchange::write_frame`]);
//! 3. receive the three peers' rows of this rank's partition and add, per
//!    element and in rank order 0..3, the local FP32 row and the three decoded
//!    peer rows, in FP32 ([`Exchange::reduce`]);
//! 4. round to BF16 and return only this partition's rows to the coordinator.
//!
//! The coordinator then receives each row once instead of four times.

use glm53f_wire::bf16::{bf16_to_f32, f32_to_bf16_rne};
use glm53f_wire::frame::seal_in_place;
use glm53f_wire::row_shard::{ExchangeHeader, ExchangeView};
use glm53f_wire::WireNaive;
pub use glm53f_wire::row_shard::{row_partition, ExchangeDtype};

use crate::consts::{HIDDEN, WORLD};
use crate::fp8::{e4m3_to_f32, f32_to_e4m3, E4M3_MAX};

/// The row count from which the design reduce-scatters a request (the
/// coordinator's threshold, configurable there): below it, the four-plane
/// return's latency is better.
pub const MIN_ROWS: usize = 16;

/// Rows per worker thread when encoding and summing: large exchanges are split
/// across threads, small ones stay on the calling thread.
const ROWS_PER_THREAD: usize = 64;

/// Encode one FP32 row for a peer. FP8: one FP32 scale (`amax / 448`, or 1
/// for an all-zero row) and E4M3 of `v / scale`, non-finite values as 0, as
/// glmrt's packer does. BF16: nearest even.
pub fn encode_row(dtype: ExchangeDtype, row: &[f32], out: &mut [u8]) {
    assert_eq!(row.len(), HIDDEN);
    assert_eq!(out.len(), dtype.row_bytes());
    match dtype {
        ExchangeDtype::Fp8RowScaled => {
            let amax = row.iter().filter(|v| v.is_finite()).fold(0f32, |m, v| m.max(v.abs()));
            let scale = if amax > 0.0 { amax / E4M3_MAX } else { 1.0 };
            for (o, &v) in out[..HIDDEN].iter_mut().zip(row) {
                *o = f32_to_e4m3(if v.is_finite() { v / scale } else { 0.0 });
            }
            out[HIDDEN..].copy_from_slice(&scale.to_le_bytes());
        }
        ExchangeDtype::Bf16 => {
            for (o, &v) in out.chunks_exact_mut(2).zip(row) {
                o.copy_from_slice(&f32_to_bf16_rne(v).to_le_bytes());
            }
        }
    }
}

/// Decode a row [`encode_row`] wrote.
pub fn decode_row(dtype: ExchangeDtype, bytes: &[u8], out: &mut [f32]) {
    assert_eq!(bytes.len(), dtype.row_bytes());
    match dtype {
        ExchangeDtype::Fp8RowScaled => {
            let scale = f32::from_le_bytes(bytes[HIDDEN..].try_into().unwrap());
            for (o, &c) in out.iter_mut().zip(&bytes[..HIDDEN]) {
                *o = e4m3_to_f32(c) * scale;
            }
        }
        ExchangeDtype::Bf16 => {
            for (o, c) in out.iter_mut().zip(bytes.chunks_exact(2)) {
                *o = bf16_to_f32(u16::from_le_bytes([c[0], c[1]]));
            }
        }
    }
}

/// The largest error one FP8 encode/decode round trip can put on `v` in a row
/// of scale `scale`: half an E4M3 step, relative `2^-4` in the normal range
/// and `2^-10 * scale` among the subnormals, plus the FP32 roundings of the
/// scale division and multiplication.
pub fn quant_bound(v: f32, scale: f32) -> f32 {
    (v.abs() * 2f32.powi(-4)).max(scale * 2f32.powi(-10)) + v.abs() * 2f32.powi(-22)
}

/// Worker threads for `rows` rows of work.
fn threads_for(rows: usize) -> usize {
    let hw = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1);
    rows.div_ceil(ROWS_PER_THREAD).clamp(1, hw.clamp(1, 16))
}

/// Run `f(first_row, rows_out)` over `out` cut into whole rows of `row_len`
/// items, on up to [`threads_for`] threads; the first error wins.
fn for_row_chunks<T: Send>(
    out: &mut [T],
    row_len: usize,
    f: impl Fn(usize, &mut [T]) -> Result<(), String> + Sync,
) -> Result<(), String> {
    let rows = out.len() / row_len;
    let threads = threads_for(rows);
    if threads == 1 {
        return f(0, out);
    }
    let per = rows.div_ceil(threads);
    std::thread::scope(|s| {
        let handles: Vec<_> = out
            .chunks_mut(per * row_len)
            .enumerate()
            .map(|(i, chunk)| {
                let f = &f;
                s.spawn(move || f(i * per, chunk))
            })
            .collect();
        let results: Vec<Result<(), String>> =
            handles.into_iter().map(|h| h.join().unwrap_or_else(|_| Err("reduce-scatter worker panicked".into()))).collect();
        results.into_iter().collect()
    })
}

/// One reduce-scattered request as rank `rank` serves it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Exchange {
    pub request_id: u64,
    pub layer: u32,
    /// The request's row count (at least one per rank).
    pub rows: usize,
    pub dtype: ExchangeDtype,
    pub rank: usize,
}

impl Exchange {
    /// This rank's partition: `(first row, rows)`.
    pub fn own(&self) -> (usize, usize) {
        row_partition(self.rows, WORLD, self.rank)
    }

    /// The header of the frame this rank sends peer `q`.
    pub fn header_to(&self, q: usize) -> ExchangeHeader {
        ExchangeHeader::new(self.request_id, self.layer, self.rows, self.rank, q, self.dtype)
    }

    /// Write the whole frame for peer `q` into `out` (exactly
    /// `header_to(q).frame_len()` bytes): the header with L4 sequence `seq`,
    /// `q`'s rows of this rank's FP32 `partial` [rows, 4096] in the exchange
    /// dtype, the CRC unless disabled for the process.
    pub fn write_frame(&self, q: usize, partial: &[f32], seq: u64, out: &mut [u8]) -> Result<(), String> {
        if partial.len() != self.rows * HIDDEN {
            return Err(format!("reduce-scatter: partial holds {} values for {} rows", partial.len(), self.rows));
        }
        let h = self.header_to(q);
        if out.len() != h.frame_len() {
            return Err(format!("reduce-scatter: frame buffer of {} bytes for {}", out.len(), h.frame_len()));
        }
        h.write(seq, out).map_err(|e| e.to_string())?;
        let (rb, first) = (self.dtype.row_bytes(), h.first_row);
        let dtype = self.dtype;
        for_row_chunks(&mut out[glm53f_wire::HEADER_LEN..], rb, |row0, chunk| {
            for (i, o) in chunk.chunks_exact_mut(rb).enumerate() {
                let r = first + row0 + i;
                encode_row(dtype, &partial[r * HIDDEN..(r + 1) * HIDDEN], o);
            }
            Ok(())
        })?;
        seal_in_place(out, WireNaive::NONE);
        Ok(())
    }

    /// Whether a received frame's header names this exchange (its request and
    /// layer): the frames of another exchange in flight are left for it.
    pub fn is_for(&self, h: &ExchangeHeader) -> bool {
        h.request_id == self.request_id && h.layer_id == self.layer
    }

    /// Check a frame of this exchange ([`Exchange::is_for`]): from a peer, to
    /// this rank, for the same rows in the same dtype. The wire checked that it
    /// carries this rank's partition.
    pub fn check(&self, h: &ExchangeHeader) -> Result<(), String> {
        let ok = self.is_for(h) && h.dst == self.rank && h.src != self.rank && h.src < WORLD && h.rows == self.rows && h.dtype == self.dtype;
        if !ok {
            return Err(format!("rank {}: a peer frame does not match this exchange ({self:?}): {h:?}", self.rank));
        }
        Ok(())
    }

    /// The reduction: this rank's rows of its own FP32 `partial` [rows, 4096]
    /// and of the three peers' frames (`peers[r]` holds rank `r`'s frame, this
    /// rank's slot empty), added per element in rank order 0..3 in FP32, then
    /// BF16 (nearest even) into `out` [count * 4096]. A missing peer, a frame
    /// that is not this exchange's, and a non-finite sum are errors.
    pub fn reduce(&self, partial: &[f32], peers: &[Option<ExchangeView<'_>>; WORLD], out: &mut [u16]) -> Result<(), String> {
        let (start, count) = self.own();
        if partial.len() != self.rows * HIDDEN || out.len() != count * HIDDEN {
            return Err(format!("reduce-scatter: {} partial values and {} outputs for {} rows", partial.len(), out.len(), self.rows));
        }
        for (r, p) in peers.iter().enumerate() {
            match p {
                None if r != self.rank => return Err(format!("rank {}: no frame from rank {r}", self.rank)),
                Some(_) if r == self.rank => return Err(format!("rank {}: a frame in its own slot", self.rank)),
                Some(v) if v.header.src != r => return Err(format!("rank {}: rank {}'s frame in slot {r}", self.rank, v.header.src)),
                Some(v) => self.check(&v.header)?,
                None => {}
            }
        }
        let (rank, dtype) = (self.rank, self.dtype);
        for_row_chunks(out, HIDDEN, |row0, chunk| {
            let mut bufs = [vec![0f32; HIDDEN], vec![0f32; HIDDEN], vec![0f32; HIDDEN], vec![0f32; HIDDEN]];
            for (i, o) in chunk.chunks_exact_mut(HIDDEN).enumerate() {
                let row = row0 + i;
                for (r, buf) in bufs.iter_mut().enumerate() {
                    match &peers[r] {
                        Some(v) => decode_row(dtype, v.row(row), buf),
                        None => buf.copy_from_slice(&partial[(start + row) * HIDDEN..(start + row + 1) * HIDDEN]),
                    }
                }
                for (h, o) in o.iter_mut().enumerate() {
                    let mut acc = 0f32;
                    for buf in &bufs {
                        acc += buf[h];
                    }
                    if !acc.is_finite() {
                        return Err(format!("rank {rank}: row {} column {h} sums to {acc}", start + row));
                    }
                    *o = f32_to_bf16_rne(acc);
                }
            }
            Ok(())
        })
    }
}

/// What rank `x.rank` sends, as whole frames (L4 sequence `seq` each): for
/// every peer, that peer's rows of the FP32 `partial`. For the CPU simulation;
/// the daemon writes each frame where its link sends it from.
pub fn outgoing(x: &Exchange, partial: &[f32], seq: u64) -> Result<Vec<Vec<u8>>, String> {
    (0..WORLD)
        .filter(|&q| q != x.rank)
        .map(|q| {
            let mut f = vec![0u8; x.header_to(q).frame_len()];
            x.write_frame(q, partial, seq, &mut f)?;
            Ok(f)
        })
        .collect()
}

/// Parse the three frames rank `x.rank` received (in any order), check them
/// against the exchange and place them by source rank, for
/// [`Exchange::reduce`]. A duplicate, a missing or a foreign frame is an error.
pub fn place_frames<'a>(x: &Exchange, frames: &'a [Vec<u8>]) -> Result<[Option<ExchangeView<'a>>; WORLD], String> {
    let mut peers: [Option<ExchangeView<'a>>; WORLD] = Default::default();
    for f in frames {
        let v = ExchangeView::parse(f, WireNaive::NONE).map_err(|e| format!("rank {}: {e}", x.rank))?;
        x.check(&v.header)?;
        let src = v.header.src;
        if peers[src].replace(v).is_some() {
            return Err(format!("rank {}: two frames from rank {src}", x.rank));
        }
    }
    Ok(peers)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn row_round_trips_are_within_their_bounds() {
        let row: Vec<f32> = (0..HIDDEN)
            .map(|i| ((i * 37 % 1001) as f32 - 500.0) * 0.013 * if i % 97 == 0 { 30.0 } else { 1.0 })
            .collect();
        let mut back = vec![0f32; HIDDEN];
        let mut bytes = vec![0u8; ExchangeDtype::Fp8RowScaled.row_bytes()];
        encode_row(ExchangeDtype::Fp8RowScaled, &row, &mut bytes);
        decode_row(ExchangeDtype::Fp8RowScaled, &bytes, &mut back);
        let scale = f32::from_le_bytes(bytes[HIDDEN..].try_into().unwrap());
        for (v, b) in row.iter().zip(&back) {
            assert!((v - b).abs() <= quant_bound(*v, scale), "{v} -> {b}");
        }
        let mut bytes = vec![0u8; ExchangeDtype::Bf16.row_bytes()];
        encode_row(ExchangeDtype::Bf16, &row, &mut bytes);
        decode_row(ExchangeDtype::Bf16, &bytes, &mut back);
        for (v, b) in row.iter().zip(&back) {
            assert!((v - b).abs() <= v.abs() * 2f32.powi(-8), "{v} -> {b}");
        }
        // An all-zero row keeps FP8 scale 1 and decodes to zeros.
        let mut bytes = vec![0u8; ExchangeDtype::Fp8RowScaled.row_bytes()];
        encode_row(ExchangeDtype::Fp8RowScaled, &vec![0f32; HIDDEN], &mut bytes);
        decode_row(ExchangeDtype::Fp8RowScaled, &bytes, &mut back);
        assert!(back.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn frames_carry_the_peers_rows_whatever_the_thread_count() {
        // 300 rows: the encode splits across threads; every row must land at its place.
        let rows = 300;
        let partial: Vec<f32> = (0..rows * HIDDEN).map(|i| (i % 7919) as f32 * 0.25 - 900.0).collect();
        let x = Exchange { request_id: 9, layer: 5, rows, dtype: ExchangeDtype::Bf16, rank: 2 };
        let frames = outgoing(&x, &partial, 3).unwrap();
        assert_eq!(frames.len(), 3);
        for f in &frames {
            let v = ExchangeView::parse(f, WireNaive::NONE).unwrap();
            assert_eq!((v.header.src, v.seq), (2, 3));
            let (first, count) = row_partition(rows, WORLD, v.header.dst);
            assert_eq!((v.header.first_row, v.header.row_count), (first, count));
            let mut back = vec![0f32; HIDDEN];
            for i in 0..count {
                decode_row(x.dtype, v.row(i), &mut back);
                let want = &partial[(first + i) * HIDDEN..(first + i + 1) * HIDDEN];
                assert!(back.iter().zip(want).all(|(b, w)| *b == bf16_to_f32(f32_to_bf16_rne(*w))), "row {}", first + i);
            }
        }
    }
}
