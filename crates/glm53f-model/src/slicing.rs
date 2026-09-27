//! Tensor-parallel slicing of the routed experts over the expert ranks.
//!
//! Each of the `world` ranks holds a `1/world` share of every routed expert,
//! split along the expert's intermediate dimension I (2,048 -> 4 x 512 at
//! TP4). Rank `r` owns intermediate channels `[r*w, (r+1)*w)` with `w = I /
//! world`. For `gate_proj` and `up_proj` (W [I, H]) those are output channels;
//! for `down_proj` (W [H, I]) they are input channels, so each rank produces a
//! partial `down` output and the partials are summed.
//!
//! # The rule, per stored part
//!
//! | Format | Part | gate / up (split outputs) | down (split inputs) |
//! |---|---|---|---|
//! | FP8 | `weight` [out, in] | rows | columns |
//! | FP8 | `weight_scale_inv` [out/128, in/128] | rows | columns |
//! | EXL3 | `trellis` [in/16, out/16, 16*bits] | axis 1 (tile columns) | axis 0 (tile rows) |
//! | EXL3 | `suh` [in] | replicated | sliced |
//! | EXL3 | `svh` [out] | sliced | replicated |
//! | EXL3 | `mcg` [1] | replicated | replicated |
//! | NVFP4 | `weight` [out, in/2] (U8) | rows | columns (packed bytes) |
//! | NVFP4 | `weight_scale` [out, in/16] | rows | columns |
//! | NVFP4 | `weight_scale_2`, `input_scale` [] | replicated | replicated |
//!
//! # Why it is exact
//!
//! A rank's share `w` must be a whole number of the format's blocks
//! ([`granularity`]), so no block straddles two ranks:
//!
//! - **FP8 (w a multiple of 128).** Each F32 scale covers one 128 x 128 block,
//!   so a rank's rows (or columns) carry exactly their own scales and every
//!   weight dequantizes to the same value as in the whole tensor.
//! - **NVFP4 (w a multiple of 16).** Each E4M3 scale covers 16 consecutive
//!   values of one row; a byte packs two neighbouring values, and w is even.
//!   `weight_scale_2` and `input_scale` are per-tensor scalars, replicated.
//! - **EXL3 (w a multiple of 128).** The layer computes `y = ((((x * suh) H_K)
//!   W_q) H_N) * svh` with `H` the Hadamard transform applied to each block of
//!   128 inputs (`H_K`) or outputs (`H_N`), and `W_q` stored as independent
//!   16 x 16 tiles. `H_K` and `H_N` are block-diagonal, so when w is a multiple
//!   of 128 every Hadamard block (and every tile) lies inside one rank:
//!   - gate/up: output block `j` of `y` needs all inputs (so all of `suh`, all
//!     tile rows), the tile columns of block `j`, and `svh` of block `j`;
//!   - down: the input transform of the rank's 512 inputs needs only their
//!     `suh`; the rank's tile rows give a partial `z_r`, and because `H_N` and
//!     `svh` are linear, `sum_r (z_r H_N) * svh = ((sum_r z_r) H_N) * svh`,
//!     so each rank applies the full output transform (all of `svh`) to its
//!     partial and the partials add up to the unsplit result.
//!
//!   The replicated `suh` (gate, up) and `svh` (down) hold H = 4,096 F16
//!   values each, so every rank keeps three 8 KiB vectors whole: 18 KiB per
//!   expert more than an even quarter, about 0.22 GB per rank over 42 layers.
//!   The only numerical difference from an unsplit layer is the order of the
//!   final sum over ranks, as for any split of a reduction dimension.
//!
//! This is the rule glmrt uses for full GLM-5.3 at TP4 (resident hidden-side
//! rotation tables per rank, intermediate-side rotations sliced, `I % (4 x
//! 128) == 0` required) and TensorFold uses for GLM-5.3-Flash at TP2 (whole
//! tiles and whole Hadamard blocks per rank). See `PROVENANCE.md`.

use crate::catalog::{
    Catalog, Component, Group, Linear, Part, Proj, Quant, EXL3_HADAMARD, EXL3_TILE, FP8_BLOCK,
    NVFP4_GROUP,
};
use crate::dtype::numel;
use crate::error::{Error, Result};
use crate::safetensors::Runs;

/// One rank of a tensor-parallel group.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Tp {
    pub rank: u64,
    pub world: u64,
}

/// How one stored tensor is divided among the ranks.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Split {
    /// Every rank holds the whole tensor.
    Replicate,
    /// Contiguous blocks of the first axis.
    Axis0,
    /// Blocks of the second axis: one run per index of the first axis.
    Axis1,
}

/// A rank's share of one stored tensor.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TensorSlice {
    /// Index into [`Catalog::tensors`].
    pub tensor: usize,
    pub part: Part,
    pub split: Split,
    /// Shape of the rank-local tensor.
    pub shape: Vec<u64>,
    /// Where its bytes are within the whole tensor.
    pub runs: Runs,
}

/// A rank's share of one routed-expert projection.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ExpertSlice {
    pub layer: u64,
    pub expert: u64,
    pub proj: Proj,
    pub tensors: Vec<TensorSlice>,
}

impl ExpertSlice {
    pub fn bytes(&self) -> u64 {
        self.tensors.iter().map(|t| t.runs.bytes()).sum()
    }
}

/// The intermediate channels a rank's share must be a multiple of, so that no
/// scale block, scale group or Hadamard block straddles two ranks.
pub fn granularity(q: Quant) -> u64 {
    match q {
        Quant::Bf16 => 1,
        Quant::Fp8Block => FP8_BLOCK,
        Quant::Exl3 { .. } => EXL3_HADAMARD.max(EXL3_TILE),
        Quant::Nvfp4 => NVFP4_GROUP,
    }
}

/// Which axis of a stored part carries the intermediate dimension.
pub fn split_rule(q: Quant, proj: Proj, part: Part) -> Option<Split> {
    use Split::*;
    let outputs = proj != Proj::Down;
    let (gate_up, down) = match (q, part) {
        (Quant::Bf16 | Quant::Fp8Block, Part::Weight | Part::Fp8ScaleInv) => (Axis0, Axis1),
        (Quant::Exl3 { .. }, Part::Exl3Trellis) => (Axis1, Axis0),
        (Quant::Exl3 { .. }, Part::Exl3Suh) => (Replicate, Axis0),
        (Quant::Exl3 { .. }, Part::Exl3Svh) => (Axis0, Replicate),
        (Quant::Exl3 { .. }, Part::Exl3Mcg) => (Replicate, Replicate),
        (Quant::Nvfp4, Part::Nvfp4Weight | Part::Nvfp4Scale) => (Axis0, Axis1),
        (Quant::Nvfp4, Part::Nvfp4Scale2 | Part::Nvfp4InputScale) => (Replicate, Replicate),
        _ => return None,
    };
    Some(if outputs { gate_up } else { down })
}

/// A rank's share of a tensor of `shape` with `esize`-byte elements.
pub fn slice_tensor(shape: &[u64], esize: u64, split: Split, tp: Tp) -> Result<(Vec<u64>, Runs)> {
    let bad = |m: String| Error::Slicing(format!("{shape:?}: {m}"));
    if tp.world == 0 || tp.rank >= tp.world {
        return Err(bad(format!("rank {} of {}", tp.rank, tp.world)));
    }
    let axis = match split {
        Split::Replicate => return Ok((shape.to_vec(), Runs::whole(numel(shape) * esize))),
        Split::Axis0 => 0,
        Split::Axis1 => 1,
    };
    let d = *shape
        .get(axis)
        .ok_or_else(|| bad(format!("no axis {axis}")))?;
    if !d.is_multiple_of(tp.world) {
        return Err(bad(format!(
            "axis {axis} ({d}) does not split {} ways",
            tp.world
        )));
    }
    let per = d / tp.world;
    let inner = numel(&shape[axis + 1..]) * esize;
    let outer = numel(&shape[..axis]);
    let mut local = shape.to_vec();
    local[axis] = per;
    let runs = Runs {
        offset: tp.rank * per * inner,
        len: per * inner,
        stride: d * inner,
        count: outer,
    };
    Ok((local, runs))
}

/// A rank's share of one routed-expert projection.
pub fn slice_expert(cat: &Catalog, lin: &Linear, tp: Tp) -> Result<ExpertSlice> {
    let s = &lin.spec;
    let Component::RoutedExpert { expert, proj } = s.component else {
        return Err(Error::Slicing(format!(
            "{} is not a routed expert",
            s.module
        )));
    };
    let inter = if proj == Proj::Down {
        s.in_features
    } else {
        s.out_features
    };
    let g = granularity(s.quant);
    if tp.world == 0 || !inter.is_multiple_of(tp.world) || !(inter / tp.world).is_multiple_of(g) {
        return Err(Error::Slicing(format!(
            "{}: {} intermediate channels over {} ranks is not a whole number of {g}-channel blocks per rank",
            s.module, inter, tp.world
        )));
    }
    let mut tensors = Vec::with_capacity(lin.parts.len());
    for &(part, i) in &lin.parts {
        let t = &cat.tensors[i];
        let split = split_rule(s.quant, proj, part).ok_or_else(|| {
            Error::Slicing(format!("{}: no rule for {part:?} of {:?}", t.name, s.quant))
        })?;
        let (shape, runs) = slice_tensor(&t.shape, t.dtype.size(), split, tp)?;
        tensors.push(TensorSlice {
            tensor: i,
            part,
            split,
            shape,
            runs,
        });
    }
    Ok(ExpertSlice {
        layer: s.layer.unwrap_or(0),
        expert,
        proj,
        tensors,
    })
}

/// One rank's share of every routed expert in a catalog.
#[derive(Clone, Debug)]
pub struct RankPlan {
    pub tp: Tp,
    /// Decoder-layer experts, then MTP experts, in catalog order.
    pub experts: Vec<ExpertSlice>,
    /// Bytes for the decoder layers' experts.
    pub bytes: u64,
    /// Bytes for the MTP layer's experts.
    pub mtp_bytes: u64,
    /// Of `bytes + mtp_bytes`, the bytes of tensors every rank holds whole.
    pub replicated_bytes: u64,
}

pub fn rank_plan(cat: &Catalog, tp: Tp) -> Result<RankPlan> {
    let mut plan = RankPlan {
        tp,
        experts: Vec::new(),
        bytes: 0,
        mtp_bytes: 0,
        replicated_bytes: 0,
    };
    for lin in cat
        .linears
        .iter()
        .filter(|l| matches!(l.spec.component, Component::RoutedExpert { .. }))
    {
        let e = slice_expert(cat, lin, tp)?;
        let mtp = cat.tensors[lin.parts[0].1].group == Group::MtpRoutedExperts;
        *(if mtp {
            &mut plan.mtp_bytes
        } else {
            &mut plan.bytes
        }) += e.bytes();
        plan.replicated_bytes += e
            .tensors
            .iter()
            .filter(|t| t.split == Split::Replicate)
            .map(|t| t.runs.bytes())
            .sum::<u64>();
        plan.experts.push(e);
    }
    Ok(plan)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn axis_slices_cover_the_tensor_once() {
        // [6, 6, 2] of 2-byte elements, 3 ranks, both axes.
        for split in [Split::Axis0, Split::Axis1] {
            let shape = [6u64, 6, 2];
            let bytes = 6 * 6 * 2 * 2;
            let mut hit = vec![0u32; bytes as usize];
            for rank in 0..3 {
                let (_, r) = slice_tensor(&shape, 2, split, Tp { rank, world: 3 }).unwrap();
                assert!(r.fits(bytes));
                for i in 0..r.count {
                    for b in 0..r.len {
                        hit[(r.offset + i * r.stride + b) as usize] += 1;
                    }
                }
            }
            assert!(hit.iter().all(|&h| h == 1), "{split:?}");
        }
    }

    #[test]
    fn uneven_splits_are_refused() {
        assert!(slice_tensor(&[6, 5], 1, Split::Axis1, Tp { rank: 0, world: 2 }).is_err());
        assert!(slice_tensor(&[6], 1, Split::Axis1, Tp { rank: 0, world: 2 }).is_err());
        assert!(slice_tensor(&[6], 1, Split::Axis0, Tp { rank: 2, world: 2 }).is_err());
    }

    #[test]
    fn granularity_per_format() {
        assert_eq!(granularity(Quant::Fp8Block), 128);
        assert_eq!(granularity(Quant::Exl3 { bits: 4 }), 128);
        assert_eq!(granularity(Quant::Nvfp4), 16);
    }
}
