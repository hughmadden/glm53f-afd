//! The rank's kernel boundary.
//!
//! It keeps the shape of mimo26f-afd's B1 path: a layer is prepared once from
//! its resident weights ([`ExpertKernel::prepare_layer`]); a call runs one
//! rank's FFN on the wire rows as they arrived, FP8 E4M3 with UE8M0 K32 scales,
//! plus top-8 expert ids and FP32 gate weights per row, and writes the rank's
//! BF16 partial rows ([`ExpertKernel::ffn`]), or the same rows before their
//! BF16 rounding for the prefill reduce-scatter ([`ExpertKernel::ffn_f32`]).
//!
//! Two backends: [`CpuKernel`] (the kernel-order reference of
//! [`crate::reference`], for the CPU gate and for serving tests) and, with the
//! `cuda` feature, `exl3_cuda::CudaKernel`.

use std::cell::RefCell;
use std::collections::HashMap;
use std::sync::Arc;

use crate::consts::{EXPERTS, HIDDEN, MAX_ROWS, TOPK};
use crate::fp8::SCALES_PER_ROW;
use crate::layout::{check_block, expert_block, LAYER_BYTES};
use crate::reference::{expert_partial_kernel_order, rank_row, rank_row_f32, KernelSlice};

/// Hidden rows as they arrived: `rows` rows of 4,096 E4M3 bytes at
/// `payload_pitch` and 128 UE8M0 bytes at `scales_pitch`.
#[derive(Clone, Copy, Debug)]
pub struct Rows<'a> {
    pub payload: &'a [u8],
    pub payload_pitch: usize,
    pub scales: &'a [u8],
    pub scales_pitch: usize,
    pub rows: usize,
}

impl<'a> Rows<'a> {
    /// Separate payload [rows * 4096] and scale [rows * 128] arrays.
    pub fn separate(payload: &'a [u8], scales: &'a [u8], rows: usize) -> Result<Self, String> {
        let r = Rows { payload, payload_pitch: HIDDEN, scales, scales_pitch: SCALES_PER_ROW, rows };
        r.check()?;
        Ok(r)
    }

    /// A request frame's hidden rows, read in place: `rows` rows of `stride`
    /// bytes, each 4,096 E4M3 bytes then 128 scale bytes.
    pub fn interleaved(hidden: &'a [u8], stride: usize, rows: usize) -> Result<Self, String> {
        if hidden.len() < HIDDEN {
            return Err("rows: hidden region shorter than one row".into());
        }
        let r = Rows { payload: hidden, payload_pitch: stride, scales: &hidden[HIDDEN..], scales_pitch: stride, rows };
        r.check()?;
        Ok(r)
    }

    /// Extents and pitches cover `rows` rows.
    pub fn check(&self) -> Result<(), String> {
        if self.rows == 0 || self.rows > MAX_ROWS {
            return Err(format!("rows: {} rows (1..={MAX_ROWS})", self.rows));
        }
        if self.payload_pitch < HIDDEN || self.scales_pitch < SCALES_PER_ROW {
            return Err("rows: pitch shorter than a row".into());
        }
        let need = |pitch: usize, width: usize| (self.rows - 1) * pitch + width;
        if self.payload.len() < need(self.payload_pitch, HIDDEN) || self.scales.len() < need(self.scales_pitch, SCALES_PER_ROW) {
            return Err(format!("rows: buffers do not hold {} rows", self.rows));
        }
        Ok(())
    }

    pub fn payload(&self, i: usize) -> &'a [u8] {
        &self.payload[i * self.payload_pitch..i * self.payload_pitch + HIDDEN]
    }

    pub fn scales(&self, i: usize) -> &'a [u8] {
        &self.scales[i * self.scales_pitch..i * self.scales_pitch + SCALES_PER_ROW]
    }
}

/// Stage timings of one call (ms), after mimo26f-afd's `b1::FfnStats`.
#[derive(Debug, Clone, Copy, Default)]
pub struct FfnStats {
    /// Host staging and uploads.
    pub upload_ms: f32,
    /// GPU time (plan to reduce).
    pub gpu_ms: f32,
    /// Download and checks.
    pub tail_ms: f32,
    /// Expert groups the plan formed.
    pub groups: u32,
    /// GPU phases: plan, gate/up, epilogue, down, reduce.
    pub phase_ms: [f32; 5],
}

/// One rank's expert FFN.
pub trait ExpertKernel {
    type Layer;
    /// Prepare one layer from its image (layout `glm53f-exl3-k4-tp4-e1`,
    /// [`LAYER_BYTES`]).
    fn prepare_layer(&mut self, image: &[u8]) -> Result<Self::Layer, String>;
    /// The rank's partial rows: for each row, `sum_slot weights[slot] *
    /// expert_partial[slot]` as BF16 into `out` [rows * 4096]. `ids` and
    /// `weights` are [rows * 8], row-major.
    fn ffn(&mut self, layer: &Self::Layer, rows: Rows<'_>, ids: &[i32], weights: &[f32], out: &mut [u16])
        -> Result<FfnStats, String>;
    /// [`ExpertKernel::ffn`] with the rows left in FP32 [rows * 4096]: the
    /// sums before their BF16 rounding, which the prefill reduce-scatter adds
    /// across the ranks first. The same arithmetic: rounding each value to
    /// BF16 (nearest even) gives `ffn`'s output bit for bit.
    fn ffn_f32(&mut self, layer: &Self::Layer, rows: Rows<'_>, ids: &[i32], weights: &[f32], out: &mut [f32])
        -> Result<FfnStats, String>;
}

/// The device planner's route checks, on the host: every id in `0..288`,
/// every weight finite and not negative, no expert twice in one row.
pub fn check_routes(ids: &[i32], weights: &[f32], rows: usize) -> Result<(), String> {
    if ids.len() != rows * TOPK || weights.len() != rows * TOPK {
        return Err(format!("routes: {} ids / {} weights for {rows} rows of {TOPK}", ids.len(), weights.len()));
    }
    for (r, (&e, &w)) in ids.iter().zip(weights).enumerate() {
        if e < 0 || e as usize >= EXPERTS {
            return Err(format!("routes: route {r} names expert {e}"));
        }
        if !(w >= 0.0 && w.is_finite()) {
            return Err(format!("routes: route {r} has gate weight {w}"));
        }
        let first = r / TOPK * TOPK;
        if ids[first..r].contains(&e) {
            return Err(format!("routes: row {} names expert {e} twice", r / TOPK));
        }
    }
    Ok(())
}

/// The CPU backend: the kernel-order reference, each expert decoded on first
/// use. Slow (f64 matrix products) and meant for tests and the CPU gate.
#[derive(Default)]
pub struct CpuKernel;

/// A layer for [`CpuKernel`]: the image and the experts decoded so far.
pub struct CpuLayer {
    image: Arc<Vec<u8>>,
    decoded: RefCell<HashMap<usize, Arc<KernelSlice>>>,
}

impl CpuLayer {
    fn slice(&self, e: usize) -> Arc<KernelSlice> {
        if let Some(s) = self.decoded.borrow().get(&e) {
            return Arc::clone(s);
        }
        let s = Arc::new(KernelSlice::from_block(expert_block(&self.image, e)));
        self.decoded.borrow_mut().insert(e, Arc::clone(&s));
        s
    }
}

impl ExpertKernel for CpuKernel {
    type Layer = CpuLayer;

    fn prepare_layer(&mut self, image: &[u8]) -> Result<CpuLayer, String> {
        if image.len() != LAYER_BYTES {
            return Err(format!("layer image has {} bytes, want {LAYER_BYTES}", image.len()));
        }
        for e in 0..EXPERTS {
            check_block(expert_block(image, e)).map_err(|m| format!("expert {e}: {m}"))?;
        }
        Ok(CpuLayer { image: Arc::new(image.to_vec()), decoded: RefCell::new(HashMap::new()) })
    }

    fn ffn(&mut self, layer: &CpuLayer, rows: Rows<'_>, ids: &[i32], weights: &[f32], out: &mut [u16])
        -> Result<FfnStats, String> {
        cpu_ffn(layer, rows, ids, weights, out.len(), |i, ys, w| rank_row(ys, w, &mut out[i * HIDDEN..(i + 1) * HIDDEN]))
    }

    fn ffn_f32(&mut self, layer: &CpuLayer, rows: Rows<'_>, ids: &[i32], weights: &[f32], out: &mut [f32])
        -> Result<FfnStats, String> {
        cpu_ffn(layer, rows, ids, weights, out.len(), |i, ys, w| rank_row_f32(ys, w, &mut out[i * HIDDEN..(i + 1) * HIDDEN]))
    }
}

/// The CPU backend's call: checks, then per row the 8 expert partials in
/// kernel order, handed to `row(i, partials, weights)` to reduce into the
/// output (`out_len` values).
fn cpu_ffn(
    layer: &CpuLayer,
    rows: Rows<'_>,
    ids: &[i32],
    weights: &[f32],
    out_len: usize,
    mut row: impl FnMut(usize, &[Vec<f32>], &[f32]) -> Result<(), String>,
) -> Result<FfnStats, String> {
    rows.check()?;
    check_routes(ids, weights, rows.rows)?;
    if out_len != rows.rows * HIDDEN {
        return Err(format!("ffn: output holds {out_len} values, want {}", rows.rows * HIDDEN));
    }
    let t = std::time::Instant::now();
    let mut x = vec![0f32; HIDDEN];
    for i in 0..rows.rows {
        crate::fp8::decode_row(rows.payload(i), rows.scales(i), &mut x)?;
        let ys: Vec<Vec<f32>> = (0..TOPK)
            .map(|s| expert_partial_kernel_order(&x, &layer.slice(ids[i * TOPK + s] as usize), true))
            .collect();
        row(i, &ys, &weights[i * TOPK..(i + 1) * TOPK])?;
    }
    Ok(FfnStats { gpu_ms: t.elapsed().as_secs_f32() * 1e3, ..FfnStats::default() })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn route_checks_match_the_planner() {
        let ids: Vec<i32> = (0..16).collect();
        let w = vec![0.25f32; 16];
        assert!(check_routes(&ids, &w, 2).is_ok());
        let mut bad = ids.clone();
        bad[3] = 288;
        assert!(check_routes(&bad, &w, 2).is_err());
        bad[3] = -1;
        assert!(check_routes(&bad, &w, 2).is_err());
        let mut dup = ids.clone();
        dup[5] = dup[1];
        assert!(check_routes(&dup, &w, 2).is_err());
        // The same expert in two different rows is fine.
        let mut two = ids.clone();
        two[9] = two[1];
        assert!(check_routes(&two, &w, 2).is_ok());
        for v in [f32::NAN, f32::INFINITY, -0.5] {
            let mut ww = w.clone();
            ww[7] = v;
            assert!(check_routes(&ids, &ww, 2).is_err(), "{v}");
        }
    }

    #[test]
    fn interleaved_rows_read_in_place() {
        let stride = 4224;
        let mut hidden = vec![0u8; 3 * stride];
        hidden[stride + 5] = 7;
        hidden[2 * stride + HIDDEN + 1] = 9;
        let r = Rows::interleaved(&hidden, stride, 3).unwrap();
        assert_eq!(r.payload(1)[5], 7);
        assert_eq!(r.scales(2)[1], 9);
        assert!(Rows::interleaved(&hidden[..2 * stride], stride, 3).is_err());
    }
}
