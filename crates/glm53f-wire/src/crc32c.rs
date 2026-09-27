//! CRC32C (Castagnoli) — the L4 per-frame checksum family (ADVISOR-I4 §3.2
//! item 6). Reflected form, poly 0x82F63B78 (0x1EDC6F41 forward), init and
//! final xor 0xFFFFFFFF — the RFC 3720 / iSCSI CRC32C, pinned by the test
//! vector `crc32c(b"123456789") == 0xE3069283`. The naive run may substitute
//! CRC-32 (IEEE) or skip/shorten coverage — every variant is a trap with a
//! negative test.

use crate::naive::WireNaive;

/// Checksum families the ladder distinguishes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CrcFamily {
    /// CRC32C, Castagnoli — the contract.
    Castagnoli,
    /// CRC-32, IEEE — a TRAP (wrong polynomial family).
    Ieee,
}

const fn build_table(poly: u32) -> [u32; 256] {
    let mut table = [0u32; 256];
    let mut i = 0usize;
    while i < 256 {
        let mut c = i as u32;
        let mut k = 0;
        while k < 8 {
            c = if c & 1 != 0 { poly ^ (c >> 1) } else { c >> 1 };
            k += 1;
        }
        table[i] = c;
        i += 1;
    }
    table
}

const CRC32C_TABLE: [u32; 256] = build_table(0x82F6_3B78);
const CRC32_IEEE_TABLE: [u32; 256] = build_table(0xEDB8_8320);

/// Byte-at-a-time 256-entry table walk — the reference path and the fallback
/// when no hardware CRC32C is available (and the oracle the hardware path is
/// bit-compared against in tests).
fn update_table(table: &[u32; 256], mut crc: u32, bytes: &[u8]) -> u32 {
    for &b in bytes {
        crc = table[((crc ^ b as u32) & 0xFF) as usize] ^ (crc >> 8);
    }
    crc
}

/// Hardware CRC32C over 8-byte words + a byte tail (x86_64 SSE4.2). Same
/// reflected Castagnoli polynomial as the table, bit-identical.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse4.2")]
unsafe fn update_sse42(mut crc: u32, bytes: &[u8]) -> u32 {
    use std::arch::x86_64::{_mm_crc32_u64, _mm_crc32_u8};
    // The u64 variant keeps the running CRC in a u64 (result in the low 32 bits).
    let mut crc64 = crc as u64;
    let mut chunks = bytes.chunks_exact(8);
    for c in &mut chunks {
        let word = u64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]);
        crc64 = _mm_crc32_u64(crc64, word);
    }
    crc = crc64 as u32;
    for &b in chunks.remainder() {
        crc = _mm_crc32_u8(crc, b);
    }
    crc
}

/// Hardware CRC32C over 4-byte words + a byte tail (aarch64 CRC). Same
/// reflected Castagnoli polynomial as the table, bit-identical.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "crc")]
unsafe fn update_arm_crc(mut crc: u32, bytes: &[u8]) -> u32 {
    use std::arch::aarch64::{__crc32cb, __crc32cd};
    // __crc32cd is the 64-bit (doubleword) CRC32C; __crc32cb the byte form.
    let mut chunks = bytes.chunks_exact(8);
    for c in &mut chunks {
        let word = u64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]);
        crc = __crc32cd(crc, word);
    }
    for &b in chunks.remainder() {
        crc = __crc32cb(crc, b);
    }
    crc
}

/// Castagnoli CRC update: hardware when available, table otherwise. The IEEE
/// naive family never takes the hardware path — the CRC32C instruction is
/// Castagnoli-only.
fn update_castagnoli(crc: u32, bytes: &[u8]) -> u32 {
    #[cfg(target_arch = "x86_64")]
    {
        if std::arch::is_x86_feature_detected!("sse4.2") {
            // SAFETY: the SSE4.2 feature was just detected for this CPU.
            return unsafe { update_sse42(crc, bytes) };
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("crc") {
            // SAFETY: the CRC feature was just detected for this CPU.
            return unsafe { update_arm_crc(crc, bytes) };
        }
    }
    update_table(&CRC32C_TABLE, crc, bytes)
}

/// CRC update for the selected family.
fn update_family(family: CrcFamily, crc: u32, bytes: &[u8]) -> u32 {
    match family {
        CrcFamily::Castagnoli => update_castagnoli(crc, bytes),
        CrcFamily::Ieee => update_table(&CRC32_IEEE_TABLE, crc, bytes),
    }
}

/// Checksum family selection for a naive flag set.
pub fn family_for(naive: WireNaive) -> CrcFamily {
    if naive.has(WireNaive::CRC32_IEEE) {
        CrcFamily::Ieee
    } else {
        CrcFamily::Castagnoli
    }
}

/// Checksum with explicit family.
pub fn crc32c_with(family: CrcFamily, bytes: &[u8]) -> u32 {
    !update_family(family, !0, bytes)
}

/// CRC32C over `bytes` with the 4 bytes at `crc_off` treated as zero, streamed
/// in three segments so the frame is never copied (the header CRC field lives
/// at `crc_off` = byte 120). The same value as checksumming `bytes` with those
/// 4 bytes zeroed in place.
pub fn crc32c_zeroed(family: CrcFamily, bytes: &[u8], crc_off: usize) -> u32 {
    let crc = !0u32;
    let crc = update_family(family, crc, &bytes[..crc_off]);
    let crc = update_family(family, crc, &[0u8; 4]);
    let crc = update_family(family, crc, &bytes[crc_off + 4..]);
    !crc
}

/// The contract checksum: CRC32C over `bytes`.
pub fn crc32c(bytes: &[u8]) -> u32 {
    crc32c_with(CrcFamily::Castagnoli, bytes)
}

/// Castagnoli via the table path only (no hardware) — the reference the
/// hardware path is bit-compared against in tests.
pub fn crc32c_table(bytes: &[u8]) -> u32 {
    !update_table(&CRC32C_TABLE, !0, bytes)
}

/// Env-default checksum (NEGATIVE tests; flips family in the naive run).
pub fn crc32c_env(bytes: &[u8]) -> u32 {
    crc32c_with(family_for(crate::naive::naive_from_env()), bytes)
}

/// Bit-by-bit reference implementation — the independent oracle that kills
/// table-generation errors (BOTH-RUNS cross-check in tests).
pub fn crc32c_bitwise(bytes: &[u8]) -> u32 {
    let mut crc = 0xFFFF_FFFFu32;
    for &b in bytes {
        crc ^= b as u32;
        for _ in 0..8 {
            crc = if crc & 1 != 0 {
                0x82F6_3B78 ^ (crc >> 1)
            } else {
                crc >> 1
            };
        }
    }
    !crc
}

#[cfg(test)]
mod tests {
    use super::*;

    fn xorshift(state: &mut u32) -> u32 {
        *state ^= *state << 13;
        *state ^= *state >> 17;
        *state ^= *state << 5;
        *state
    }

    /// The hardware path (whatever `crc32c` dispatches to) must be bit-identical
    /// to the table path for every length 0..=4096, plus a 1 MiB and a 17.86 MB
    /// frame. A length-0 or short buffer exercises the tail path.
    #[test]
    fn hardware_matches_table_all_lengths_and_large_frames() {
        let mut state = 0x1234_5678u32;
        let mut buf = vec![0u8; 4096];
        for len in 0..=4096 {
            for b in buf.iter_mut().take(len) {
                *b = (xorshift(&mut state) >> 24) as u8;
            }
            assert_eq!(
                crc32c(&buf[..len]),
                crc32c_table(&buf[..len]),
                "hardware != table at length {len}"
            );
        }

        // 1 MiB random.
        let mut big = vec![0u8; 1 << 20];
        for b in &mut big {
            *b = (xorshift(&mut state) >> 24) as u8;
        }
        assert_eq!(crc32c(&big), crc32c_table(&big), "hardware != table at 1 MiB");

        // 17.86 MB frame (the prefill request size).
        let mut frame = vec![0u8; 17_860_000];
        for b in &mut frame {
            *b = (xorshift(&mut state) >> 24) as u8;
        }
        assert_eq!(crc32c(&frame), crc32c_table(&frame), "hardware != table at 17.86 MB");
    }

    /// The no-copy `crc32c_zeroed` (the frame seal/verify path) must equal the
    /// explicit zero-the-field-then-checksum value at several offsets.
    #[test]
    fn zeroed_stream_matches_explicit_zeroing() {
        let mut state = 0xdead_beefu32;
        let mut buf = vec![0u8; 256];
        for b in &mut buf {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            *b = (state >> 24) as u8;
        }
        for off in [0usize, 1, 4, 120, 252] {
            let mut copy = buf.clone();
            if off + 4 <= copy.len() {
                copy[off..off + 4].fill(0);
            }
            assert_eq!(
                crc32c_zeroed(CrcFamily::Castagnoli, &buf, off),
                crc32c_with(CrcFamily::Castagnoli, &copy),
                "zeroed stream mismatch at offset {off}"
            );
        }
    }

    /// Timing probe (ignored in CI): record the table vs hardware CRC cost over
    /// a 17.86 MB frame, so the before/after seal+verify number is captured on
    /// coordinator (x86_64) and the Sparks (aarch64). Run with
    /// `cargo test --release -p glm53f-wire --lib bench_table_vs_hardware -- --ignored --nocapture`.
    #[test]
    #[ignore]
    fn bench_table_vs_hardware_17_86mb() {
        use std::time::Instant;
        let mut frame = vec![0u8; 17_860_000];
        let mut state = 0x9e37_79b9u32;
        for b in &mut frame {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            *b = (state >> 24) as u8;
        }
        let hw = crc32c(&frame);
        let tab = crc32c_table(&frame);
        assert_eq!(hw, tab);

        // 5 runs each, report the mean.
        let mut hw_total = 0.0f64;
        let mut tab_total = 0.0f64;
        for _ in 0..5 {
            let t = Instant::now();
            std::hint::black_box(crc32c(std::hint::black_box(&frame)));
            hw_total += t.elapsed().as_secs_f64();
            let t = Instant::now();
            std::hint::black_box(crc32c_table(std::hint::black_box(&frame)));
            tab_total += t.elapsed().as_secs_f64();
        }
        let hw_ms = hw_total / 5.0 * 1e3;
        let tab_ms = tab_total / 5.0 * 1e3;
        eprintln!(
            "CRC 17.86MB: table={tab_ms:.3} ms ({:.1} GB/s) hardware={hw_ms:.3} ms ({:.1} GB/s)",
            17.86 / tab_ms,
            17.86 / hw_ms
        );
    }
}
