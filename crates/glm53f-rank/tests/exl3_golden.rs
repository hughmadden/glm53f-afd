//! The EXL3 decoder against TensorFold's reference, `exl3.py` (TensorFold @
//! bb4b4a3, `src/tensorfold/families/glm5_next/cuda/exl3.py`, MIT).
//!
//! The constants below were produced by running that file unmodified (its
//! numpy code, with a numpy stand-in for the few torch calls it makes) on the
//! same pseudo-random inputs this test builds: SplitMix64 streams, trellis
//! words from the low 16 bits of each draw, FP16 scale vectors with magnitudes
//! in [0.5, 2). Hashes are FNV-1a 64 over the little-endian bytes of the u16
//! arrays. The decoder (codebook, tile order, state extraction, unpacking) must
//! match bit for bit; the float64 dequantization and forward pass must match
//! to rounding (the Python side multiplies by the Hadamard matrix, this side
//! runs butterflies).

use glm53f_rank::exl3::{self, TILE_BYTES};
use glm53f_rank::half::{f16_to_f64, f64_to_f16};

struct SplitMix64(u64);

impl SplitMix64 {
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
}

fn fnv_u16(v: &[u16]) -> u64 {
    let mut h: u64 = 0xCBF2_9CE4_8422_2325;
    for x in v {
        for b in x.to_le_bytes() {
            h ^= b as u64;
            h = h.wrapping_mul(0x0000_0100_0000_01B3);
        }
    }
    h
}

fn trellis(seed: u64, kt: usize, nt: usize) -> Vec<u8> {
    let mut r = SplitMix64(seed);
    let mut out = Vec::with_capacity(kt * nt * TILE_BYTES);
    for _ in 0..kt * nt * 64 {
        out.extend_from_slice(&((r.next_u64() & 0xFFFF) as u16).to_le_bytes());
    }
    out
}

fn f16_vec(seed: u64, n: usize) -> Vec<u16> {
    let (lo, hi) = (0.5f64, 2.0f64);
    let mut r = SplitMix64(seed);
    (0..n)
        .map(|_| {
            let z = r.next_u64();
            let mag = lo + (hi - lo) * ((z >> 11) as f64 / (1u64 << 53) as f64);
            f64_to_f16(if z >> 63 != 0 { -mag } else { mag })
        })
        .collect()
}

fn f32_vec(seed: u64, n: usize) -> Vec<f64> {
    let mut r = SplitMix64(seed);
    (0..n)
        .map(|_| {
            let u = (r.next_u64() >> 11) as f64 / (1u64 << 53) as f64;
            ((2.0 * u - 1.0) as f32) as f64
        })
        .collect()
}

fn u16s(b: &[u8]) -> Vec<u16> {
    b.chunks_exact(2).map(|c| u16::from_le_bytes([c[0], c[1]])).collect()
}

#[test]
fn codebook_matches_exl3_py() {
    let cb = exl3::codebook();
    assert_eq!(fnv_u16(cb), 0x7bb3_6b05_a3b8_85a7);
    for (s, want) in [
        (0x0000usize, 0x3f60u16),
        (0x0001, 0x304e),
        (0x0002, 0xba13),
        (0x1234, 0x3e0f),
        (0x8000, 0xb915),
        (0xffff, 0x3acd),
        (0xbeef, 0xc293),
    ] {
        assert_eq!(cb[s], want, "state {s:#06x}");
    }
    assert_eq!(f16_to_f64(cb[0xbeef]), -3.287109375);
}

#[test]
fn tile_positions_match_exl3_py() {
    let mut v = Vec::new();
    for p in 0..256 {
        let (r, c) = exl3::tile_position(p);
        v.push(r as u16);
        v.push(c as u16);
    }
    assert_eq!(fnv_u16(&v), 0x385b_68bf_97cc_7f25);
}

/// (seed, k tiles, n tiles, trellis hash, states hash, W_q hash, spot values).
type Case = (u64, usize, usize, u64, u64, u64, [((usize, usize), u16); 5]);

const CASES: [Case; 3] = [
    (
        0x5EED_0001,
        4,
        3,
        0xb2d0_d508_391a_3e87,
        0x883f_fdfa_b3f9_7705,
        0x30ed_22b4_33f6_ad87,
        [((0, 0), 0xad50), ((1, 2), 0xbbba), ((15, 15), 0x3d9c), ((63, 47), 0xbbd2), ((7, 9), 0x3c4c)],
    ),
    // The shape of a rank's gate/up slice: K = 4,096, N = 512.
    (
        0x5EED_0002,
        256,
        32,
        0x55f8_3703_625c_43ac,
        0x9496_8327_968f_8c7b,
        0x00a4_10ca_2748_3e91,
        [((0, 0), 0x374a), ((1, 2), 0x387e), ((15, 15), 0x4064), ((4095, 511), 0xc198), ((7, 9), 0x33e0)],
    ),
    // The shape of a rank's down slice: K = 512, N = 4,096.
    (
        0x5EED_0003,
        32,
        256,
        0x8ef9_37b3_6a41_f5bb,
        0xd294_7476_c34c_15db,
        0x0e53_3e4d_a080_29e9,
        [((0, 0), 0x30d8), ((1, 2), 0xb0aa), ((15, 15), 0x383d), ((511, 4095), 0x33e2), ((7, 9), 0x37e2)],
    ),
];

#[test]
fn states_and_unpack_match_exl3_py_bit_for_bit() {
    for (seed, kt, nt, t_hash, s_hash, w_hash, spots) in CASES {
        let t = trellis(seed, kt, nt);
        assert_eq!(fnv_u16(&u16s(&t)), t_hash, "trellis {seed:#x} (the input generator)");
        let mut states = Vec::with_capacity(kt * nt * 256);
        for tile in t.chunks_exact(TILE_BYTES) {
            let w = exl3::tile_words(tile);
            for p in 0..256 {
                states.push(exl3::tile_state(&w, p));
            }
        }
        assert_eq!(fnv_u16(&states), s_hash, "states {seed:#x}");
        let wq = exl3::unpack(&t, kt, nt);
        assert_eq!(fnv_u16(&wq), w_hash, "W_q {seed:#x}");
        for ((k, n), want) in spots {
            assert_eq!(wq[k * nt * 16 + n], want, "W_q[{k},{n}] {seed:#x}");
        }
    }
}

fn close(a: f64, b: f64, rel: f64) -> bool {
    (a - b).abs() <= rel * b.abs().max(1.0)
}

#[test]
fn dequantize_and_forward_match_exl3_py() {
    let (seed, kt, nt) = (0x5EED_0010u64, 16usize, 24usize);
    let (k, n) = (kt * 16, nt * 16);
    let t = trellis(seed, kt, nt);
    let suh = f16_vec(0x5EED_0011, k);
    let svh = f16_vec(0x5EED_0012, n);
    assert_eq!(fnv_u16(&suh), 0x227f_d76c_8fa3_0b92);
    assert_eq!(fnv_u16(&svh), 0x9726_551c_0cf7_e5e2);
    let w = exl3::dequantize(&t, kt, nt, &suh, &svh);
    for ((r, c), want) in [
        ((0usize, 0usize), -0.7242898202748624f64),
        ((1, 2), -0.48290348295267865),
        ((127, 128), 0.5546841259083515),
        ((128, 127), -1.3615442480358986),
        ((255, 383), -1.8943148755261063),
        ((200, 17), -1.0883147672702755),
    ] {
        assert!(close(w[r * n + c], want, 1e-12), "W[{r},{c}] = {} want {want}", w[r * n + c]);
    }
    let sum: f64 = w.iter().sum();
    let abs: f64 = w.iter().map(|v| v.abs()).sum();
    let sq: f64 = w.iter().map(|v| v * v).sum();
    assert!(close(sum, -28.76148294026987, 1e-9), "sum {sum}");
    assert!(close(abs, 150410.30607602728, 1e-12), "abs sum {abs}");
    assert!(close(sq, 458395.3603214598, 1e-12), "square sum {sq}");

    let x = f32_vec(0x5EED_0013, 3 * k);
    let y = exl3::forward(&x, 3, &t, kt, nt, &suh, &svh);
    for ((r, c), want) in [
        ((0usize, 0usize), 9.87604709176315f64),
        ((1, 5), 10.74031091398918),
        ((2, 383), 2.4989257209420193),
        ((2, 128), -11.32514636145083),
    ] {
        assert!(close(y[r * n + c], want, 1e-12), "y[{r},{c}] = {} want {want}", y[r * n + c]);
    }
    // The factored forward pass is x W.
    for r in 0..3 {
        for c in (0..n).step_by(37) {
            let direct: f64 = (0..k).map(|i| x[r * k + i] * w[i * n + c]).sum();
            assert!(close(y[r * n + c], direct, 1e-12));
        }
    }
}

#[test]
fn had_scale_is_the_f32_of_one_over_sqrt_128() {
    assert_eq!(exl3::HAD_SCALE_F32, (1.0f64 / 128f64.sqrt()) as f32);
}
