//! Cache formats of the DSA layers.
//!
//! Per token and DSA layer the engine keeps one **MLA latent record**; per
//! complete pool of 4 tokens it keeps one **pooled index key**; per request and
//! layer it keeps the **tail**: raw index keys and gates of the (at most 3)
//! tokens of the incomplete pool.
//!
//! ```text
//! Latent record, 528 bytes (one token, one layer):
//!   [  0, 512)  512 x E4M3 codes, latent channel order
//!   [512, 528)  4 x f32 LE scales, one per 128-channel group (channels 128g .. 128g+127)
//!   value[i] = e4m3(code[i]) * scale[i / 128]
//!
//! Pooled index key, 132 bytes of payload (one pool, one layer):
//!   128 x E4M3 codes + 1 x f32 scale; value[i] = e4m3(code[i]) * scale
//!   stored structure-of-arrays per page: 16 x 128 code bytes, then 16 x f32 scales
//!
//! Tail (one request, one layer), 1,552 bytes:
//!   [ 0,  4)  u32 count (0..=3)          [ 4, 16) reserved (zero)
//!   [16, 16 + 512 c)  per tail token c: 128 x BF16 key, then 128 x BF16 gate
//!
//! Page (64 tokens = 16 pools), per DSA layer, 35,904 bytes:
//!   [     0, 33,792)  64 latent records (token slot t at 528 t)
//!   [33,792, 35,840)  16 x 128 pooled-key codes (pool slot p at 33,792 + 128 p)
//!   [35,840, 35,904)  16 x f32 pooled-key scales
//! ```
//!
//! Scales are written as the smallest power of two `s` with `amax <= 448 s`
//! ([`ScaleMode::Pow2`]); readers accept any f32 scale. Why this layout and
//! granularity is chosen is argued in the crate README (section "Cache formats").

use crate::fp8::{self, ScaleMode};
use crate::num::{bf16_bits_to_f32, f32_to_bf16_bits};

/// Latent channels.
pub const LATENT_DIM: usize = 512;
/// Channels per latent scale group.
pub const LATENT_GROUP: usize = 128;
/// Scale groups per latent record.
pub const LATENT_GROUPS: usize = LATENT_DIM / LATENT_GROUP;
/// Bytes per latent record.
pub const LATENT_RECORD_BYTES: usize = LATENT_DIM + 4 * LATENT_GROUPS; // 528
/// Index key channels.
pub const INDEX_DIM: usize = 128;
/// Bytes of one pooled key (codes + scale).
pub const INDEX_KEY_BYTES: usize = INDEX_DIM + 4; // 132
/// Tokens per page.
pub const PAGE_TOKENS: usize = 64;
/// Tokens per pool.
pub const KPOOL: usize = 4;
/// Pools per page.
pub const PAGE_POOLS: usize = PAGE_TOKENS / KPOOL; // 16
/// Byte offset of the pooled-key codes within a layer's page block.
pub const PAGE_POOL_CODES_OFFSET: usize = PAGE_TOKENS * LATENT_RECORD_BYTES; // 33,792
/// Byte offset of the pooled-key scales within a layer's page block.
pub const PAGE_POOL_SCALES_OFFSET: usize = PAGE_POOL_CODES_OFFSET + PAGE_POOLS * INDEX_DIM; // 35,840
/// Bytes of one layer's block in a page.
pub const PAGE_LAYER_BYTES: usize = PAGE_POOL_SCALES_OFFSET + PAGE_POOLS * 4; // 35,904
/// Maximum tail tokens (`kpool - 1`).
pub const TAIL_MAX: usize = KPOOL - 1;
/// Bytes of one tail token: BF16 key + BF16 gate.
pub const TAIL_TOKEN_BYTES: usize = 2 * INDEX_DIM * 2; // 512
/// Bytes of one tail record.
pub const TAIL_BYTES: usize = 16 + TAIL_MAX * TAIL_TOKEN_BYTES; // 1,552

/// Encode one latent (512 values) as a 528-byte record.
pub fn encode_latent(latent: &[f32], mode: ScaleMode) -> [u8; LATENT_RECORD_BYTES] {
    assert_eq!(latent.len(), LATENT_DIM);
    let mut rec = [0u8; LATENT_RECORD_BYTES];
    for g in 0..LATENT_GROUPS {
        let r = g * LATENT_GROUP..(g + 1) * LATENT_GROUP;
        let s = fp8::quantize_block(&latent[r.clone()], &mut rec[r], mode);
        rec[LATENT_DIM + 4 * g..LATENT_DIM + 4 * g + 4].copy_from_slice(&s.to_le_bytes());
    }
    rec
}

/// Scale of group `g` of a latent record.
#[inline]
pub fn latent_scale(rec: &[u8], g: usize) -> f32 {
    let o = LATENT_DIM + 4 * g;
    f32::from_le_bytes([rec[o], rec[o + 1], rec[o + 2], rec[o + 3]])
}

/// Decode a 528-byte latent record to f32 (`code * scale`, exact for power-of-two scales).
pub fn decode_latent(rec: &[u8]) -> Vec<f32> {
    assert!(rec.len() >= LATENT_RECORD_BYTES);
    let mut out = vec![0.0f32; LATENT_DIM];
    for g in 0..LATENT_GROUPS {
        let r = g * LATENT_GROUP..(g + 1) * LATENT_GROUP;
        fp8::dequantize_block(&rec[r.clone()], latent_scale(rec, g), &mut out[r]);
    }
    out
}

/// Encode a pooled index key: 128 codes and one scale.
pub fn encode_index_key(key: &[f32], mode: ScaleMode) -> ([u8; INDEX_DIM], f32) {
    assert_eq!(key.len(), INDEX_DIM);
    let mut codes = [0u8; INDEX_DIM];
    let s = fp8::quantize_block(key, &mut codes, mode);
    (codes, s)
}

/// Decode a pooled index key.
pub fn decode_index_key(codes: &[u8], scale: f32) -> Vec<f32> {
    let mut out = vec![0.0f32; codes.len()];
    fp8::dequantize_block(codes, scale, &mut out);
    out
}

/// Encode a pooled index key as BF16 (the 256-byte alternative format).
pub fn encode_index_key_bf16(key: &[f32]) -> Vec<u16> {
    key.iter().map(|v| f32_to_bf16_bits(*v)).collect()
}

/// The tail of one request and layer: raw keys and gates of up to 3 tokens.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Tail {
    /// `(key, gate)` per tail token, each `INDEX_DIM` BF16 values (as bits).
    pub tokens: Vec<(Vec<u16>, Vec<u16>)>,
}

impl Tail {
    /// Serialize to the 1,552-byte record.
    pub fn encode(&self) -> [u8; TAIL_BYTES] {
        assert!(self.tokens.len() <= TAIL_MAX);
        let mut b = [0u8; TAIL_BYTES];
        b[0..4].copy_from_slice(&(self.tokens.len() as u32).to_le_bytes());
        for (c, (k, g)) in self.tokens.iter().enumerate() {
            let o = 16 + c * TAIL_TOKEN_BYTES;
            for (i, v) in k.iter().chain(g.iter()).enumerate() {
                b[o + 2 * i..o + 2 * i + 2].copy_from_slice(&v.to_le_bytes());
            }
        }
        b
    }

    /// Parse a tail record.
    pub fn decode(b: &[u8]) -> Result<Self, String> {
        if b.len() < TAIL_BYTES {
            return Err(format!("tail record is {} bytes, need {TAIL_BYTES}", b.len()));
        }
        let n = u32::from_le_bytes([b[0], b[1], b[2], b[3]]) as usize;
        if n > TAIL_MAX {
            return Err(format!("tail count {n} > {TAIL_MAX}"));
        }
        let mut tokens = Vec::with_capacity(n);
        for c in 0..n {
            let o = 16 + c * TAIL_TOKEN_BYTES;
            let vals: Vec<u16> =
                (0..2 * INDEX_DIM).map(|i| u16::from_le_bytes([b[o + 2 * i], b[o + 2 * i + 1]])).collect();
            tokens.push((vals[..INDEX_DIM].to_vec(), vals[INDEX_DIM..].to_vec()));
        }
        Ok(Self { tokens })
    }

    /// Add a token (key and gate rounded to BF16, as the reference caches them).
    pub fn push(&mut self, key: &[f32], gate: &[f32]) {
        self.tokens.push((
            key.iter().map(|v| f32_to_bf16_bits(*v)).collect(),
            gate.iter().map(|v| f32_to_bf16_bits(*v)).collect(),
        ));
    }

    /// Keys and gates as f32.
    pub fn values(&self) -> Vec<(Vec<f32>, Vec<f32>)> {
        self.tokens
            .iter()
            .map(|(k, g)| (k.iter().map(|v| bf16_bits_to_f32(*v)).collect(), g.iter().map(|v| bf16_bits_to_f32(*v)).collect()))
            .collect()
    }
}

/// A paged byte image of the DSA cache for one layer (the device layout; used to
/// build GPU test inputs and to decode GPU outputs).
///
/// Physical page `p` occupies bytes `[p * page_stride, p * page_stride + PAGE_LAYER_BYTES)`.
/// A request's logical page `i` (tokens `64 i .. 64 i + 63`) maps to physical
/// page `page_table[i]`.
#[derive(Clone, Debug)]
pub struct PagedLayer {
    pub bytes: Vec<u8>,
    pub page_stride: usize,
    pub pages: usize,
}

impl PagedLayer {
    pub fn new(pages: usize, page_stride: usize) -> Self {
        assert!(page_stride >= PAGE_LAYER_BYTES && page_stride % 16 == 0);
        Self { bytes: vec![0u8; pages * page_stride], page_stride, pages }
    }

    /// Byte offset of token `t`'s latent record.
    pub fn latent_offset(&self, page_table: &[i32], t: usize) -> usize {
        let p = page_table[t / PAGE_TOKENS] as usize;
        p * self.page_stride + (t % PAGE_TOKENS) * LATENT_RECORD_BYTES
    }

    /// Byte offsets of pool `q`'s codes and scale.
    pub fn pool_offsets(&self, page_table: &[i32], q: usize) -> (usize, usize) {
        let p = page_table[q / PAGE_POOLS] as usize;
        let base = p * self.page_stride;
        (base + PAGE_POOL_CODES_OFFSET + (q % PAGE_POOLS) * INDEX_DIM, base + PAGE_POOL_SCALES_OFFSET + (q % PAGE_POOLS) * 4)
    }

    pub fn write_latent(&mut self, page_table: &[i32], t: usize, rec: &[u8; LATENT_RECORD_BYTES]) {
        let o = self.latent_offset(page_table, t);
        self.bytes[o..o + LATENT_RECORD_BYTES].copy_from_slice(rec);
    }

    pub fn read_latent(&self, page_table: &[i32], t: usize) -> &[u8] {
        let o = self.latent_offset(page_table, t);
        &self.bytes[o..o + LATENT_RECORD_BYTES]
    }

    pub fn write_pool(&mut self, page_table: &[i32], q: usize, codes: &[u8], scale: f32) {
        let (c, s) = self.pool_offsets(page_table, q);
        self.bytes[c..c + INDEX_DIM].copy_from_slice(codes);
        self.bytes[s..s + 4].copy_from_slice(&scale.to_le_bytes());
    }

    pub fn read_pool(&self, page_table: &[i32], q: usize) -> (&[u8], f32) {
        let (c, s) = self.pool_offsets(page_table, q);
        let sc = f32::from_le_bytes([self.bytes[s], self.bytes[s + 1], self.bytes[s + 2], self.bytes[s + 3]]);
        (&self.bytes[c..c + INDEX_DIM], sc)
    }
}

/// Bytes per token over all 11 DSA layers (latent plus a quarter pooled key).
pub const fn bytes_per_token_all_layers() -> usize {
    PAGE_LAYER_BYTES * crate::config::DSA_LAYERS.len() / PAGE_TOKENS
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rng::Rng;

    #[test]
    fn sizes_match_the_sizing_document() {
        assert_eq!(LATENT_RECORD_BYTES, 528);
        assert_eq!(INDEX_KEY_BYTES, 132);
        assert_eq!(PAGE_LAYER_BYTES, 35_904);
        assert_eq!(PAGE_LAYER_BYTES, PAGE_TOKENS * LATENT_RECORD_BYTES + PAGE_POOLS * INDEX_KEY_BYTES);
        // SIZING.md section 2: 5,808 B latent + 363 B pooled keys per token over 11 layers.
        assert_eq!(bytes_per_token_all_layers(), 6_171);
        assert_eq!(TAIL_BYTES, 1_552);
        assert_eq!(LATENT_RECORD_BYTES % 16, 0, "records stay 16-byte aligned");
    }

    #[test]
    fn latent_record_round_trip() {
        let mut rng = Rng::new(3);
        let mut v = rng.normals(LATENT_DIM, 1.0);
        v[5] = 40.0; // an outlier channel only perturbs its own group
        let rec = encode_latent(&v, ScaleMode::Pow2);
        let back = decode_latent(&rec);
        for g in 0..LATENT_GROUPS {
            let s = latent_scale(&rec, g);
            assert_eq!(s.to_bits() & 0x7F_FFFF, 0);
            for i in g * 128..(g + 1) * 128 {
                let tol = (v[i].abs() / 16.0).max(s * 2f32.powi(-10));
                assert!((v[i] - back[i]).abs() <= tol * 1.0001);
            }
        }
        // Re-encoding a decoded record keeps every value (the scale may halve when
        // a group's largest code rounded down to 224).
        assert_eq!(decode_latent(&encode_latent(&back, ScaleMode::Pow2)), back);
    }

    #[test]
    fn tail_round_trip() {
        let mut t = Tail::default();
        let mut rng = Rng::new(4);
        for _ in 0..3 {
            t.push(&rng.normals(INDEX_DIM, 1.0), &rng.normals(INDEX_DIM, 1.0));
        }
        let b = t.encode();
        assert_eq!(Tail::decode(&b).unwrap(), t);
        assert_eq!(Tail::decode(&Tail::default().encode()).unwrap().tokens.len(), 0);
    }

    #[test]
    fn paged_layout_offsets() {
        let mut l = PagedLayer::new(4, PAGE_LAYER_BYTES);
        let table = [2i32, 0, 3];
        let rec = encode_latent(&vec![1.0; LATENT_DIM], ScaleMode::Pow2);
        l.write_latent(&table, 70, &rec); // logical page 1 -> physical 0, slot 6
        assert_eq!(l.latent_offset(&table, 70), 6 * 528);
        assert_eq!(l.read_latent(&table, 70), &rec[..]);
        let (c, s) = l.pool_offsets(&table, 17); // logical page 1 -> physical 0, pool slot 1
        assert_eq!((c, s), (PAGE_POOL_CODES_OFFSET + 128, PAGE_POOL_SCALES_OFFSET + 4));
        l.write_pool(&table, 3, &[7u8; 128], 0.5);
        let (codes, sc) = l.read_pool(&table, 3);
        assert_eq!((codes[0], sc), (7, 0.5));
    }
}
