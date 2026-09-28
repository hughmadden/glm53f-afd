//! The rank's weight image, layout `glm53f-exl3-k4-tp4-e1`, and how it is cut
//! from the checkpoint.
//!
//! A layer image holds rank `r`'s share of all 288 experts of one MoE layer,
//! expert after expert (`expert_id == slot`), each expert block
//! [`EXPERT_BYTES`] = 3,173,376 bytes:
//!
//! | Offset | Bytes | What | Source tensor, rank share |
//! |---:|---:|---|---|
//! | 0 | 1,048,576 | gate trellis `[256][32][64]` I16 | `gate_proj.trellis` [256, 128, 64], tile columns `[32 r, 32 r + 32)` |
//! | 1,048,576 | 1,048,576 | up trellis `[256][32][64]` | `up_proj.trellis`, the same columns |
//! | 2,097,152 | 1,048,576 | down trellis `[32][256][64]` | `down_proj.trellis` [128, 256, 64], tile rows `[32 r, 32 r + 32)` |
//! | 3,145,728 | 8,192 | gate `suh` [4,096] F16 | whole (input scales) |
//! | 3,153,920 | 8,192 | up `suh` [4,096] | whole |
//! | 3,162,112 | 1,024 | gate `svh` [512] | `[512 r, 512 r + 512)` (output scales) |
//! | 3,163,136 | 1,024 | up `svh` [512] | the same |
//! | 3,164,160 | 1,024 | down `suh` [512] | `[512 r, 512 r + 512)` (input scales) |
//! | 3,165,184 | 8,192 | down `svh` [4,096] | whole (output scales) |
//!
//! The `mcg` markers are checked when slicing and not stored. A layer image is
//! 288 x 3,173,376 = 913,932,288 bytes; 42 layers are 38.39 GB per rank.
//!
//! The split itself is `glm53f_model::slicing` (the rule and its proof live
//! there; README.md explains why it is exact for EXL3). This module only
//! places the pieces and checks them.

use glm53f_model::catalog::{Catalog, Part, Proj, Quant};
use glm53f_model::safetensors::Checkpoint;
use glm53f_model::slicing::{slice_tensor, split_rule, Tp};

use crate::consts::{EXPERTS, HIDDEN, INTERMEDIATE, RANK_WIDTH, WORLD};
use crate::exl3::{MCG_MULT, TILE, TILE_BYTES};

/// Layout name, recorded in the manifest.
pub const LAYOUT: &str = "glm53f-exl3-k4-tp4-e1";

/// One trellis slice: 4,096 x 512 weights at 4 bits.
pub const TRELLIS_BYTES: usize = HIDDEN * RANK_WIDTH / 2;
pub const GATE_TRELLIS: usize = 0;
pub const UP_TRELLIS: usize = GATE_TRELLIS + TRELLIS_BYTES;
pub const DOWN_TRELLIS: usize = UP_TRELLIS + TRELLIS_BYTES;
pub const GATE_SUH: usize = DOWN_TRELLIS + TRELLIS_BYTES;
pub const UP_SUH: usize = GATE_SUH + HIDDEN * 2;
pub const GATE_SVH: usize = UP_SUH + HIDDEN * 2;
pub const UP_SVH: usize = GATE_SVH + RANK_WIDTH * 2;
pub const DOWN_SUH: usize = UP_SVH + RANK_WIDTH * 2;
pub const DOWN_SVH: usize = DOWN_SUH + RANK_WIDTH * 2;
/// Bytes of one expert's rank share.
pub const EXPERT_BYTES: usize = DOWN_SVH + HIDDEN * 2;
/// Bytes of one layer image (all 288 experts).
pub const LAYER_BYTES: usize = EXPERTS * EXPERT_BYTES;

const _: () = assert!(EXPERT_BYTES == 3_173_376);
const _: () = assert!(LAYER_BYTES == 913_932_288);
const _: () = assert!(EXPERT_BYTES.is_multiple_of(256), "expert blocks stay 256-byte aligned");

/// Trellis tiles of a rank's gate/up slice (k tiles, n tiles) and down slice.
pub const GATE_UP_TILES: (usize, usize) = (HIDDEN / TILE, RANK_WIDTH / TILE);
pub const DOWN_TILES: (usize, usize) = (RANK_WIDTH / TILE, HIDDEN / TILE);

/// One EXL3 linear in checkpoint form (whole tensors, little-endian bytes).
#[derive(Clone, Debug)]
pub struct Exl3Linear {
    /// `trellis` I16 [in/16, out/16, 64].
    pub trellis: Vec<u8>,
    /// `suh` F16 [in].
    pub suh: Vec<u8>,
    /// `svh` F16 [out].
    pub svh: Vec<u8>,
    /// `mcg` I32 [1].
    pub mcg: Vec<u8>,
}

/// One routed expert in checkpoint form.
#[derive(Clone, Debug)]
pub struct Exl3Expert {
    pub gate: Exl3Linear,
    pub up: Exl3Linear,
    pub down: Exl3Linear,
}

/// Checkpoint shape of each part of a projection: (trellis, suh, svh).
fn shapes(proj: Proj) -> ([u64; 3], [u64; 1], [u64; 1]) {
    let (h, i) = (HIDDEN as u64, INTERMEDIATE as u64);
    match proj {
        Proj::Gate | Proj::Up => ([h / 16, i / 16, 64], [h], [i]),
        Proj::Down => ([i / 16, h / 16, 64], [i], [h]),
    }
}

fn check_mcg(bytes: &[u8], what: &str) -> Result<(), String> {
    if bytes.len() != 4 {
        return Err(format!("{what}: mcg marker has {} bytes, want 4", bytes.len()));
    }
    let v = u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
    if v != MCG_MULT {
        return Err(format!("{what}: mcg marker is {v:#010x}, want {MCG_MULT:#010x} (the mcg codebook)"));
    }
    Ok(())
}

/// Where each part of `proj` goes in an expert block.
fn offsets(proj: Proj) -> (usize, usize, usize) {
    match proj {
        Proj::Gate => (GATE_TRELLIS, GATE_SUH, GATE_SVH),
        Proj::Up => (UP_TRELLIS, UP_SUH, UP_SVH),
        Proj::Down => (DOWN_TRELLIS, DOWN_SUH, DOWN_SVH),
    }
}

fn part_len(proj: Proj, part: Part) -> usize {
    match (proj, part) {
        (_, Part::Exl3Trellis) => TRELLIS_BYTES,
        (Proj::Gate | Proj::Up, Part::Exl3Suh) | (Proj::Down, Part::Exl3Svh) => HIDDEN * 2,
        _ => RANK_WIDTH * 2,
    }
}

fn tp(rank: usize) -> Result<Tp, String> {
    if rank >= WORLD {
        return Err(format!("rank {rank} is outside 0..{WORLD}"));
    }
    Ok(Tp { rank: rank as u64, world: WORLD as u64 })
}

const QUANT: Quant = Quant::Exl3 { bits: 4 };

/// Write rank `rank`'s share of a whole expert into `block` (one expert block).
pub fn slice_expert(full: &Exl3Expert, rank: usize, block: &mut [u8]) -> Result<(), String> {
    if block.len() != EXPERT_BYTES {
        return Err(format!("expert block has {} bytes, want {EXPERT_BYTES}", block.len()));
    }
    let tp = tp(rank)?;
    for (proj, lin) in [(Proj::Gate, &full.gate), (Proj::Up, &full.up), (Proj::Down, &full.down)] {
        check_mcg(&lin.mcg, proj.stem())?;
        let (t_shape, suh_shape, svh_shape) = shapes(proj);
        let (t_off, suh_off, svh_off) = offsets(proj);
        for (part, bytes, shape, esize, off) in [
            (Part::Exl3Trellis, &lin.trellis, &t_shape[..], 2u64, t_off),
            (Part::Exl3Suh, &lin.suh, &suh_shape[..], 2, suh_off),
            (Part::Exl3Svh, &lin.svh, &svh_shape[..], 2, svh_off),
        ] {
            let want: u64 = shape.iter().product::<u64>() * esize;
            if bytes.len() as u64 != want {
                return Err(format!("{} {part:?}: {} bytes, want {want}", proj.stem(), bytes.len()));
            }
            let split = split_rule(QUANT, proj, part).ok_or("no split rule")?;
            let (_, runs) = slice_tensor(shape, esize, split, tp).map_err(|e| e.to_string())?;
            let got = runs.gather(bytes);
            let len = part_len(proj, part);
            if got.len() != len {
                return Err(format!("{} {part:?}: rank share is {} bytes, want {len}", proj.stem(), got.len()));
            }
            block[off..off + len].copy_from_slice(&got);
        }
    }
    Ok(())
}

/// Read rank `rank`'s share of `(layer, expert)` from a checkpoint into
/// `block`, through the catalog's TP4 slicing plan.
pub fn slice_checkpoint_expert(
    cp: &Checkpoint,
    cat: &Catalog,
    layer: u32,
    expert: usize,
    rank: usize,
    block: &mut [u8],
) -> Result<(), String> {
    if block.len() != EXPERT_BYTES {
        return Err(format!("expert block has {} bytes, want {EXPERT_BYTES}", block.len()));
    }
    let tp = tp(rank)?;
    for proj in Proj::ALL {
        let lin = cat
            .expert(layer as u64, expert as u64, proj)
            .ok_or_else(|| format!("layer {layer} expert {expert} {}: not in the checkpoint", proj.stem()))?;
        let slice = glm53f_model::slicing::slice_expert(cat, lin, tp).map_err(|e| e.to_string())?;
        let (t_off, suh_off, svh_off) = offsets(proj);
        for ts in &slice.tensors {
            let name = &cat.tensors[ts.tensor].name;
            let bytes = cp.read_runs(name, &ts.runs).map_err(|e| e.to_string())?;
            let off = match ts.part {
                Part::Exl3Trellis => t_off,
                Part::Exl3Suh => suh_off,
                Part::Exl3Svh => svh_off,
                Part::Exl3Mcg => {
                    check_mcg(&bytes, name)?;
                    continue;
                }
                p => return Err(format!("{name}: unexpected part {p:?} for EXL3")),
            };
            let len = part_len(proj, ts.part);
            if bytes.len() != len {
                return Err(format!("{name}: rank share is {} bytes, want {len}", bytes.len()));
            }
            // The split the catalog chose must be the one this layout assumes.
            let want = split_rule(QUANT, proj, ts.part);
            if want != Some(ts.split) {
                return Err(format!("{name}: split {:?}, layout expects {want:?}", ts.split));
            }
            block[off..off + len].copy_from_slice(&bytes);
        }
    }
    Ok(())
}

/// A whole layer image for `rank`, read from a checkpoint.
pub fn slice_checkpoint_layer(cp: &Checkpoint, cat: &Catalog, layer: u32, rank: usize) -> Result<Vec<u8>, String> {
    let mut image = vec![0u8; LAYER_BYTES];
    for (e, block) in image.chunks_exact_mut(EXPERT_BYTES).enumerate() {
        slice_checkpoint_expert(cp, cat, layer, e, rank, block)?;
    }
    Ok(image)
}

/// One expert's block of a layer image.
pub fn expert_block(image: &[u8], expert: usize) -> &[u8] {
    &image[expert * EXPERT_BYTES..(expert + 1) * EXPERT_BYTES]
}

/// FP16 bits of an F16 vector stored at `off..off + 2 n` of a block.
pub fn f16s(block: &[u8], off: usize, n: usize) -> Vec<u16> {
    block[off..off + 2 * n].chunks_exact(2).map(|c| u16::from_le_bytes([c[0], c[1]])).collect()
}

/// The pieces of one expert block, decoded to their natural types.
pub struct ExpertParts<'a> {
    pub gate_trellis: &'a [u8],
    pub up_trellis: &'a [u8],
    pub down_trellis: &'a [u8],
    pub gate_suh: Vec<u16>,
    pub up_suh: Vec<u16>,
    pub gate_svh: Vec<u16>,
    pub up_svh: Vec<u16>,
    pub down_suh: Vec<u16>,
    pub down_svh: Vec<u16>,
}

pub fn parts(block: &[u8]) -> ExpertParts<'_> {
    assert_eq!(block.len(), EXPERT_BYTES);
    ExpertParts {
        gate_trellis: &block[GATE_TRELLIS..GATE_TRELLIS + TRELLIS_BYTES],
        up_trellis: &block[UP_TRELLIS..UP_TRELLIS + TRELLIS_BYTES],
        down_trellis: &block[DOWN_TRELLIS..DOWN_TRELLIS + TRELLIS_BYTES],
        gate_suh: f16s(block, GATE_SUH, HIDDEN),
        up_suh: f16s(block, UP_SUH, HIDDEN),
        gate_svh: f16s(block, GATE_SVH, RANK_WIDTH),
        up_svh: f16s(block, UP_SVH, RANK_WIDTH),
        down_suh: f16s(block, DOWN_SUH, RANK_WIDTH),
        down_svh: f16s(block, DOWN_SVH, HIDDEN),
    }
}

/// Check an expert block's scale vectors are finite (the kernel multiplies by
/// them unchecked). Returns the first bad `(offset, bits)`.
pub fn check_block(block: &[u8]) -> Result<(), String> {
    for (off, n) in [
        (GATE_SUH, HIDDEN),
        (UP_SUH, HIDDEN),
        (GATE_SVH, RANK_WIDTH),
        (UP_SVH, RANK_WIDTH),
        (DOWN_SUH, RANK_WIDTH),
        (DOWN_SVH, HIDDEN),
    ] {
        for (i, h) in f16s(block, off, n).into_iter().enumerate() {
            if h & 0x7C00 == 0x7C00 {
                return Err(format!("scale vector at offset {off}: element {i} is not finite ({h:#06x})"));
            }
        }
    }
    Ok(())
}

/// Bytes of one tile row of a gate/up slice (32 tiles), for callers that
/// stream trellis rows.
pub const GATE_UP_TILE_ROW_BYTES: usize = (RANK_WIDTH / TILE) * TILE_BYTES;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn offsets_are_contiguous() {
        assert_eq!(GATE_SUH, 3_145_728);
        assert_eq!(DOWN_SVH, 3_165_184);
        assert_eq!(GATE_UP_TILES, (256, 32));
        assert_eq!(DOWN_TILES, (32, 256));
        assert_eq!(GATE_UP_TILES.0 * GATE_UP_TILES.1 * TILE_BYTES, TRELLIS_BYTES);
        assert_eq!(GATE_UP_TILE_ROW_BYTES, 4096);
    }

    #[test]
    fn splits_are_the_documented_ones() {
        use glm53f_model::slicing::Split::*;
        assert_eq!(split_rule(QUANT, Proj::Gate, Part::Exl3Trellis), Some(Axis1));
        assert_eq!(split_rule(QUANT, Proj::Up, Part::Exl3Suh), Some(Replicate));
        assert_eq!(split_rule(QUANT, Proj::Up, Part::Exl3Svh), Some(Axis0));
        assert_eq!(split_rule(QUANT, Proj::Down, Part::Exl3Trellis), Some(Axis0));
        assert_eq!(split_rule(QUANT, Proj::Down, Part::Exl3Suh), Some(Axis0));
        assert_eq!(split_rule(QUANT, Proj::Down, Part::Exl3Svh), Some(Replicate));
    }
}
