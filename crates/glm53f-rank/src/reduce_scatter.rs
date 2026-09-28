//! Prefill return path: the reduce-scatter among the four ranks, as a CPU
//! simulation of the protocol's arithmetic and bookkeeping. The design (when
//! it runs, frames, error bounds, traffic) is in README.md, "Prefill
//! reduce-scatter"; this module is what that design computes, so its tests can
//! pin the result against the four-plane sum.
//!
//! After glmrt v9 (`glmrt-daemon` `real_full/rdma_reduction.rs`,
//! `intermediate_sharding.rs::balanced_row_partition`, and the row-scaled FP8
//! kernels `bf16_rows_to_fp8_e4m3_row_scaled` /
//! `combine_bf16_fp8_e4m3_row_scaled_to_fp8` in `native/cuda/kernels/residual.cu`;
//! MIT). Reimplemented, no code copied. Two differences: the reduced rows go
//! back to the coordinator as the wire's BF16 rows (glmrt re-quantizes them to
//! FP8), and the exchange dtype is a choice ([`PeerDtype`]): FP8 row-scaled as
//! glmrt, or BF16 (README.md explains why BF16 is the proposed default).
//!
//! Per request of `rows >= 16` rows and for each rank `r`:
//!
//! 1. compute the full partial `P_r` [rows, 4096] in FP32 (the kernel's reduce
//!    output before its BF16 rounding);
//! 2. rows are partitioned `[start_q, start_q + count_q)` over the four ranks
//!    ([`partition`]); send each peer `q` its rows of `P_r` in the exchange
//!    dtype;
//! 3. receive the three peers' rows of this rank's partition and add, per
//!    element and in rank order 0..3, the local FP32 row and the three decoded
//!    peer rows, in FP32;
//! 4. round to BF16 and return only this partition's rows to the coordinator.
//!
//! The coordinator then receives each row once instead of four times.

use glm53f_wire::bf16::{bf16_to_f32, f32_to_bf16_rne};
use glm53f_wire::crc32c::crc32c;

use crate::consts::{HIDDEN, WORLD};
use crate::fp8::{e4m3_to_f32, f32_to_e4m3, E4M3_MAX};

/// The row count from which a request's return is reduce-scattered.
pub const MIN_ROWS: usize = 16;

/// How peer rows travel between ranks (codes as glmrt's reduction dtypes).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PeerDtype {
    /// 4,096 BF16 values (8,192 bytes a row).
    Bf16 = 1,
    /// 4,096 E4M3 values and one FP32 scale, `amax / 448` (4,100 bytes a row).
    Fp8RowScaled = 2,
}

impl PeerDtype {
    pub fn row_bytes(self) -> usize {
        match self {
            PeerDtype::Bf16 => 2 * HIDDEN,
            PeerDtype::Fp8RowScaled => HIDDEN + 4,
        }
    }

    fn from_code(c: u16) -> Option<Self> {
        match c {
            1 => Some(PeerDtype::Bf16),
            2 => Some(PeerDtype::Fp8RowScaled),
            _ => None,
        }
    }
}

/// The rows rank `rank` of `world` owns: `(start, count)`, the first
/// `rows % world` ranks holding one row more (glmrt's balanced partition).
pub fn partition(rows: usize, world: usize, rank: usize) -> (usize, usize) {
    assert!(world > 0 && rank < world);
    let (base, extra) = (rows / world, rows % world);
    (rank * base + rank.min(extra), base + usize::from(rank < extra))
}

/// Encode one FP32 row for a peer. FP8: one FP32 scale (`amax / 448`, or 1
/// for an all-zero row) and E4M3 of `v / scale`, non-finite values as 0, as
/// glmrt's packer does. BF16: nearest even.
pub fn encode_row(dtype: PeerDtype, row: &[f32], out: &mut [u8]) {
    assert_eq!(row.len(), HIDDEN);
    assert_eq!(out.len(), dtype.row_bytes());
    match dtype {
        PeerDtype::Fp8RowScaled => {
            let amax = row.iter().filter(|v| v.is_finite()).fold(0f32, |m, v| m.max(v.abs()));
            let scale = if amax > 0.0 { amax / E4M3_MAX } else { 1.0 };
            for (o, &v) in out[..HIDDEN].iter_mut().zip(row) {
                *o = f32_to_e4m3(if v.is_finite() { v / scale } else { 0.0 });
            }
            out[HIDDEN..].copy_from_slice(&scale.to_le_bytes());
        }
        PeerDtype::Bf16 => {
            for (o, &v) in out.chunks_exact_mut(2).zip(row) {
                o.copy_from_slice(&f32_to_bf16_rne(v).to_le_bytes());
            }
        }
    }
}

/// Decode a row [`encode_row`] wrote.
pub fn decode_row(dtype: PeerDtype, bytes: &[u8], out: &mut [f32]) {
    assert_eq!(bytes.len(), dtype.row_bytes());
    match dtype {
        PeerDtype::Fp8RowScaled => {
            let scale = f32::from_le_bytes(bytes[HIDDEN..].try_into().unwrap());
            for (o, &c) in out.iter_mut().zip(&bytes[..HIDDEN]) {
                *o = e4m3_to_f32(c) * scale;
            }
        }
        PeerDtype::Bf16 => {
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

/// Header of one rank-to-rank message (64 bytes, little-endian): magic
/// `G53RRS01` @0, request id @8, layer @16, source rank @20, destination rank
/// @22, the request's row count @24, first row @28 and rows carried @32, the
/// exchange dtype @36, the per-link sequence @40, a CRC32C @48 over the whole
/// message with that field zeroed; the rest reserved (zero).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PeerHeader {
    pub request_id: u64,
    pub layer: u32,
    pub src: u16,
    pub dst: u16,
    pub rows: u32,
    pub row_start: u32,
    pub row_count: u32,
    pub dtype: PeerDtype,
    pub seq: u64,
}

pub const PEER_MAGIC: [u8; 8] = *b"G53RRS01";
pub const PEER_HEADER_LEN: usize = 64;

impl PeerHeader {
    /// Encode a whole message: header then `row_count` rows of payload.
    pub fn encode(&self, payload: &[u8]) -> Vec<u8> {
        assert_eq!(payload.len(), self.row_count as usize * self.dtype.row_bytes());
        let mut m = vec![0u8; PEER_HEADER_LEN + payload.len()];
        m[0..8].copy_from_slice(&PEER_MAGIC);
        m[8..16].copy_from_slice(&self.request_id.to_le_bytes());
        m[16..20].copy_from_slice(&self.layer.to_le_bytes());
        m[20..22].copy_from_slice(&self.src.to_le_bytes());
        m[22..24].copy_from_slice(&self.dst.to_le_bytes());
        m[24..28].copy_from_slice(&self.rows.to_le_bytes());
        m[28..32].copy_from_slice(&self.row_start.to_le_bytes());
        m[32..36].copy_from_slice(&self.row_count.to_le_bytes());
        m[36..38].copy_from_slice(&(self.dtype as u16).to_le_bytes());
        m[40..48].copy_from_slice(&self.seq.to_le_bytes());
        m[PEER_HEADER_LEN..].copy_from_slice(payload);
        let crc = crc32c(&m);
        m[48..52].copy_from_slice(&crc.to_le_bytes());
        m
    }

    /// Decode and check a message: magic, CRC, reserved bytes, dtype, and that
    /// the payload holds exactly the rows the header claims.
    pub fn decode(m: &[u8]) -> Result<(PeerHeader, &[u8]), String> {
        if m.len() < PEER_HEADER_LEN || m[0..8] != PEER_MAGIC {
            return Err("peer message: bad magic or truncated header".into());
        }
        let u32_at = |o: usize| u32::from_le_bytes(m[o..o + 4].try_into().unwrap());
        let u64_at = |o: usize| u64::from_le_bytes(m[o..o + 8].try_into().unwrap());
        let mut z = m.to_vec();
        z[48..52].fill(0);
        if crc32c(&z) != u32_at(48) {
            return Err("peer message: CRC mismatch".into());
        }
        if m[38..40].iter().chain(&m[52..64]).any(|&b| b != 0) {
            return Err("peer message: reserved bytes are not zero".into());
        }
        let dtype = PeerDtype::from_code(u16::from_le_bytes([m[36], m[37]])).ok_or("peer message: unknown dtype")?;
        let h = PeerHeader {
            request_id: u64_at(8),
            layer: u32_at(16),
            src: u16::from_le_bytes([m[20], m[21]]),
            dst: u16::from_le_bytes([m[22], m[23]]),
            rows: u32_at(24),
            row_start: u32_at(28),
            row_count: u32_at(32),
            dtype,
            seq: u64_at(40),
        };
        let payload = &m[PEER_HEADER_LEN..];
        if payload.len() != h.row_count as usize * dtype.row_bytes() {
            return Err("peer message: payload does not match its row count".into());
        }
        Ok((h, payload))
    }
}

/// One request's identity on the exchange.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Exchange {
    pub request_id: u64,
    pub layer: u32,
    pub seq: u64,
    pub rows: usize,
    pub dtype: PeerDtype,
}

/// What rank `rank` sends: for each peer, a message with that peer's rows of
/// this rank's FP32 partial `partial` [rows, 4096].
pub fn outgoing(rank: usize, x: &Exchange, partial: &[f32]) -> Vec<Vec<u8>> {
    assert_eq!(partial.len(), x.rows * HIDDEN);
    let rb = x.dtype.row_bytes();
    (0..WORLD)
        .filter(|&q| q != rank)
        .map(|q| {
            let (start, count) = partition(x.rows, WORLD, q);
            let mut payload = vec![0u8; count * rb];
            for i in 0..count {
                encode_row(x.dtype, &partial[(start + i) * HIDDEN..(start + i + 1) * HIDDEN], &mut payload[i * rb..(i + 1) * rb]);
            }
            PeerHeader {
                request_id: x.request_id,
                layer: x.layer,
                src: rank as u16,
                dst: q as u16,
                rows: x.rows as u32,
                row_start: start as u32,
                row_count: count as u32,
                dtype: x.dtype,
                seq: x.seq,
            }
            .encode(&payload)
        })
        .collect()
}

/// Rank `rank`'s reduction: its own FP32 rows of its partition plus the three
/// peers' decoded rows, added per element in rank order, then BF16. `incoming`
/// holds the peers' messages in any order; every peer must appear once, for
/// this request, layer, dtype and partition. Returns `(row_start, rows of BF16)`.
pub fn reduce(rank: usize, x: &Exchange, partial: &[f32], incoming: &[Vec<u8>]) -> Result<(usize, Vec<u16>), String> {
    let (start, count) = partition(x.rows, WORLD, rank);
    let rb = x.dtype.row_bytes();
    let mut peer: [Option<&[u8]>; WORLD] = [None; WORLD];
    for m in incoming {
        let (h, payload) = PeerHeader::decode(m)?;
        let src = h.src as usize;
        let ok = h.request_id == x.request_id
            && h.layer == x.layer
            && h.seq == x.seq
            && h.dtype == x.dtype
            && h.dst as usize == rank
            && src < WORLD
            && src != rank
            && h.rows as usize == x.rows
            && (h.row_start as usize, h.row_count as usize) == (start, count);
        if !ok {
            return Err(format!("rank {rank}: a peer message does not match this request's partition: {h:?}"));
        }
        if peer[src].replace(payload).is_some() {
            return Err(format!("rank {rank}: two messages from rank {src}"));
        }
    }
    let mut out = vec![0u16; count * HIDDEN];
    let mut bufs = [vec![0f32; HIDDEN], vec![0f32; HIDDEN], vec![0f32; HIDDEN], vec![0f32; HIDDEN]];
    for i in 0..count {
        for (r, buf) in bufs.iter_mut().enumerate() {
            if r == rank {
                buf.copy_from_slice(&partial[(start + i) * HIDDEN..(start + i + 1) * HIDDEN]);
            } else {
                let p = peer[r].ok_or_else(|| format!("rank {rank}: no message from rank {r}"))?;
                decode_row(x.dtype, &p[i * rb..(i + 1) * rb], buf);
            }
        }
        for h in 0..HIDDEN {
            let mut acc = 0f32;
            for buf in &bufs {
                acc += buf[h];
            }
            out[i * HIDDEN + h] = f32_to_bf16_rne(acc);
        }
    }
    Ok((start, out))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn partitions_cover_the_rows_once() {
        assert_eq!(
            (0..4).map(|r| partition(1006, 4, r)).collect::<Vec<_>>(),
            vec![(0, 252), (252, 252), (504, 251), (755, 251)]
        );
        for rows in [16usize, 17, 18, 19, 4096] {
            let mut next = 0;
            for r in 0..4 {
                let (s, c) = partition(rows, 4, r);
                assert_eq!(s, next);
                next += c;
            }
            assert_eq!(next, rows);
        }
    }

    #[test]
    fn peer_headers_round_trip_and_catch_corruption() {
        for dtype in [PeerDtype::Fp8RowScaled, PeerDtype::Bf16] {
            let payload = vec![7u8; 2 * dtype.row_bytes()];
            let h = PeerHeader { request_id: 9, layer: 12, src: 1, dst: 3, rows: 8, row_start: 6, row_count: 2, dtype, seq: 44 };
            let m = h.encode(&payload);
            let (g, p) = PeerHeader::decode(&m).unwrap();
            assert_eq!((g, p), (h, &payload[..]));
            let mut bad = m.clone();
            bad[PEER_HEADER_LEN + 100] ^= 1;
            assert!(PeerHeader::decode(&bad).is_err());
            assert!(PeerHeader::decode(&m[..m.len() - 1]).is_err());
        }
    }

    #[test]
    fn row_round_trips_are_within_their_bounds() {
        let row: Vec<f32> = (0..HIDDEN)
            .map(|i| ((i * 37 % 1001) as f32 - 500.0) * 0.013 * if i % 97 == 0 { 30.0 } else { 1.0 })
            .collect();
        let mut back = vec![0f32; HIDDEN];
        let mut bytes = vec![0u8; PeerDtype::Fp8RowScaled.row_bytes()];
        encode_row(PeerDtype::Fp8RowScaled, &row, &mut bytes);
        decode_row(PeerDtype::Fp8RowScaled, &bytes, &mut back);
        let scale = f32::from_le_bytes(bytes[HIDDEN..].try_into().unwrap());
        for (v, b) in row.iter().zip(&back) {
            assert!((v - b).abs() <= quant_bound(*v, scale), "{v} -> {b}");
        }
        let mut bytes = vec![0u8; PeerDtype::Bf16.row_bytes()];
        encode_row(PeerDtype::Bf16, &row, &mut bytes);
        decode_row(PeerDtype::Bf16, &bytes, &mut back);
        for (v, b) in row.iter().zip(&back) {
            assert!((v - b).abs() <= v.abs() * 2f32.powi(-8), "{v} -> {b}");
        }
        // An all-zero row keeps FP8 scale 1 and decodes to zeros.
        let mut bytes = vec![0u8; PeerDtype::Fp8RowScaled.row_bytes()];
        encode_row(PeerDtype::Fp8RowScaled, &vec![0f32; HIDDEN], &mut bytes);
        decode_row(PeerDtype::Fp8RowScaled, &bytes, &mut back);
        assert!(back.iter().all(|&v| v == 0.0));
    }
}
