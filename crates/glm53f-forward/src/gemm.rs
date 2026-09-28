//! The forward's weight GEMMs (feature `cuda`).
//!
//! - **BF16** (the KDA projections, the indexer's BF16 projections, the LM head): up to
//!   [`GemmPolicy::gemv_max_rows`] rows (8) run this crate's GEMV (`kernels/gemv_bf16.cu`):
//!   bandwidth-bound, and row-independent, so a verify window of R rows gives the bits of R
//!   serial decode steps. More rows run cuBLAS (`cublasGemmEx`, BF16 tensor cores, f32
//!   accumulation, f32 split-K reductions).
//! - **FP8 block-128** (the DSA projections, shared experts, dense MLPs, local routed experts,
//!   and the KDA projections when they are loaded in FP8): `glm53f-layers`' kernels. Up to 8
//!   rows: the decode GEMM with its K splits fused (BF16 activations by default, or the
//!   checkpoint's dynamic per-128 E4M3 activations); more rows: the FP8 tensor-core GEMM, which
//!   takes E4M3 activations (W8A8), or, with [`GemmPolicy::prefill_w8a16`], BF16 activations
//!   (W8A16): the weight dequantized to BF16 tiles of rows (`glm53f_fp8_dequant_bf16`, one
//!   rounded product per weight) in [`DEQUANT_BYTES`] of scratch, each multiplied by cuBLAS as
//!   the BF16 GEMMs are.

use core::ffi::c_void;

use glm53f_layers::ffi as lffi;
use glm53f_layers::mlp::decode_ksplit;

use crate::cublas::{Blas, GemmDesc};
use crate::device::{launched, DeviceBuffer, Stream};
use crate::error::{invalid, Result};
use crate::ffi;

/// A BF16 weight `[groups][n][k]` on the device (row `o` of group `g` at
/// `ptr + g * gstride + o * ld`).
#[derive(Clone, Copy, Debug)]
pub struct Bf16Mat {
    pub ptr: *const u16,
    pub n: usize,
    pub k: usize,
    pub ld: usize,
    pub groups: usize,
    pub gstride: usize,
}

impl Bf16Mat {
    pub fn bytes(&self) -> usize {
        self.groups * self.n * self.k * 2
    }
}

/// An FP8 E4M3 weight `[n][k]` with f32 scales `[n / 128][k / 128]` on the device.
#[derive(Clone, Copy, Debug)]
pub struct Fp8Mat {
    pub w: *const u8,
    pub scales: *const f32,
    pub n: usize,
    pub k: usize,
}

impl Fp8Mat {
    pub fn bytes(&self) -> usize {
        self.n * self.k + self.n.div_ceil(128) * self.k.div_ceil(128) * 4
    }
}

/// Activations for an FP8 projection: BF16 rows `[rows][k]`, and their E4M3 form (`[rows][k]`
/// codes, `[rows][k / 128]` scales) when it exists.
#[derive(Clone, Copy, Debug)]
pub struct Fp8Input {
    pub bf16: *const u16,
    pub q: *const u8,
    pub scales: *const f32,
}

/// How the FP8 decode GEMM (up to 8 rows) meets its activations.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Fp8Act {
    /// BF16 activations against the FP8 weights (W8A16): every product exact in f32.
    Bf16,
    /// The checkpoint's `activation_scheme: dynamic`: E4M3 per row and 128-group (W8A8).
    Dynamic128,
}

/// Which kernel each GEMM uses.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GemmPolicy {
    /// BF16 GEMMs with at most this many rows run the GEMV (8 at most; 0 sends every BF16
    /// GEMM to cuBLAS).
    pub gemv_max_rows: usize,
    /// FP8 decode GEMM activations.
    pub fp8_act: Fp8Act,
    /// The FP8 tensor-core GEMM adds every k32 partial sum in f32 (more accurate, about 9%
    /// slower) instead of accumulating whole 128-blocks in the tensor core.
    pub prefill_promote_k32: bool,
    /// FP8 GEMMs over more than 8 rows take BF16 activations (W8A16) through BF16 tiles of the
    /// weight and cuBLAS, instead of E4M3 activations (W8A8) on the FP8 tensor cores. Needs
    /// [`DEQUANT_BYTES`] of scratch, allocated with the engine.
    pub prefill_w8a16: bool,
    /// With `prefill_w8a16`: the FP8 KDA projections ([`Gemm::fp8_kda`], decision D2) keep E4M3
    /// activations over 8 rows, so they keep D2's prefill speed while the other projections take
    /// W8A16. Nothing without `prefill_w8a16`.
    pub kda_prefill_w8a8: bool,
}

impl Default for GemmPolicy {
    fn default() -> Self {
        GemmPolicy {
            gemv_max_rows: 8,
            fp8_act: Fp8Act::Bf16,
            prefill_promote_k32: true,
            prefill_w8a16: false,
            kda_prefill_w8a8: false,
        }
    }
}

impl GemmPolicy {
    /// Whether FP8 GEMMs over more than 8 rows take BF16 activations (for the KDA projections
    /// when `kda`).
    pub fn w8a16(&self, kda: bool) -> bool {
        self.prefill_w8a16 && !(kda && self.kda_prefill_w8a8)
    }

    /// Whether an FP8 GEMM over `rows` rows needs E4M3 activations.
    pub fn fp8_needs_quant(&self, rows: usize) -> bool {
        self.needs_quant(rows, false)
    }

    /// The same for an FP8 KDA projection (`kda`) or another one.
    pub fn needs_quant(&self, rows: usize, kda: bool) -> bool {
        if rows > 8 {
            !self.w8a16(kda)
        } else {
            self.fp8_act == Fp8Act::Dynamic128
        }
    }
}

/// Split-K scratch sizes: f32 partials and zeroed counters.
const PARTIAL_FLOATS: usize = 1 << 20;
const SYNC_COUNTERS: usize = 1 << 16;
/// The cuBLAS workspace.
const BLAS_WORKSPACE: usize = 32 << 20;
/// The BF16 weight tiles of the W8A16 prefill path ([`GemmPolicy::prefill_w8a16`]): 8,192 rows
/// of a 4,096-wide weight, 2,048 of the widest (16,384).
pub const DEQUANT_BYTES: usize = 64 << 20;

/// The GEMM engine: a cuBLAS handle and the split-K scratch the fused kernels share (one
/// stream: launches run one after another, and the kernels leave the counters zeroed).
pub struct Gemm {
    blas: Blas,
    partials: DeviceBuffer,
    sync: DeviceBuffer,
    /// BF16 weight tiles for the W8A16 prefill path (allocated when the policy has it).
    dequant: Option<DeviceBuffer>,
    pub policy: GemmPolicy,
}

impl Gemm {
    pub fn new(stream: &Stream, policy: GemmPolicy) -> Result<Gemm> {
        if policy.gemv_max_rows > 8 {
            return Err(invalid!("the GEMV takes at most 8 rows"));
        }
        Ok(Gemm {
            blas: Blas::new(stream, BLAS_WORKSPACE)?,
            partials: DeviceBuffer::alloc(PARTIAL_FLOATS * 4)?,
            sync: DeviceBuffer::zeroed(SYNC_COUNTERS * 4)?,
            dequant: if policy.prefill_w8a16 {
                Some(DeviceBuffer::alloc(DEQUANT_BYTES)?)
            } else {
                None
            },
            policy,
        })
    }

    /// Device bytes of its scratch (the cuBLAS workspace, the split-K buffers and the W8A16
    /// path's weight tiles).
    pub fn bytes(&self) -> usize {
        BLAS_WORKSPACE
            + self.partials.bytes()
            + self.sync.bytes()
            + self.dequant.as_ref().map_or(0, |d| d.bytes())
    }

    /// Rows of the W8A16 path's weight tiles for an `[n][k]` weight: tiles of equal size (a
    /// multiple of 128 rows, the last one shorter), as few as [`DEQUANT_BYTES`] allows.
    pub fn w8a16_tile_rows(n: usize, k: usize) -> usize {
        let max = (DEQUANT_BYTES / (2 * k)) / 128 * 128;
        let tiles = n.div_ceil(max.max(128));
        n.div_ceil(tiles).div_ceil(128) * 128
    }

    /// The GEMV's K splits for `w` (shape only).
    pub fn gemv_ksplit(w: &Bf16Mat) -> usize {
        // SAFETY: a pure host function.
        unsafe { ffi::glm53f_fwd_gemv_ksplit(w.groups as i32, w.n as i32, w.k as i32) as usize }
    }

    /// `out[g][m][o] = x[g][m] . w[g][o]` for `rows` rows: BF16 inputs, f32 accumulation, BF16
    /// (or f32) output. `x` rows are `ldx` apart and groups `x_gstride`; `out` rows `ldo` and
    /// groups `o_gstride`.
    ///
    /// # Safety
    ///
    /// `x`, `out` and the weight are device pointers to live buffers that hold the shapes named
    /// (with their strides), and stay alive until the stream's work completes.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn bf16(
        &self,
        x: *const u16,
        ldx: usize,
        x_gstride: usize,
        w: &Bf16Mat,
        rows: usize,
        out: *mut c_void,
        ldo: usize,
        o_gstride: usize,
        out_f32: bool,
        stream: &Stream,
    ) -> Result<()> {
        if rows == 0 {
            return Ok(());
        }
        if rows <= self.policy.gemv_max_rows {
            self.gemv(
                x, ldx, x_gstride, w, rows, out, ldo, o_gstride, out_f32, stream,
            )
        } else {
            self.cublas(x, ldx, x_gstride, w, rows, out, ldo, o_gstride, out_f32)
        }
    }

    /// The GEMV (1..8 rows), whatever the policy.
    ///
    /// # Safety
    ///
    /// As for [`Gemm::bf16`].
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn gemv(
        &self,
        x: *const u16,
        ldx: usize,
        x_gstride: usize,
        w: &Bf16Mat,
        rows: usize,
        out: *mut c_void,
        ldo: usize,
        o_gstride: usize,
        out_f32: bool,
        stream: &Stream,
    ) -> Result<()> {
        let ksplit = Self::gemv_ksplit(w);
        if ksplit > 1
            && (w.groups * ksplit * rows * w.n > PARTIAL_FLOATS
                || w.groups * w.n / 8 > SYNC_COUNTERS)
        {
            return Err(invalid!("GEMV split-K scratch too small for {w:?}"));
        }
        // SAFETY: the pointers name live device buffers the caller sized for the shapes; the
        // scratch was checked above.
        let code = unsafe {
            ffi::glm53f_fwd_gemv_bf16(
                x,
                ldx as i64,
                x_gstride as i64,
                w.ptr,
                w.ld as i64,
                w.gstride as i64,
                w.groups as i32,
                rows as i32,
                w.n as i32,
                w.k as i32,
                ksplit as i32,
                self.partials.ptr(0),
                self.sync.ptr(0),
                out,
                i32::from(out_f32),
                ldo as i64,
                o_gstride as i64,
                stream.raw(),
            )
        };
        launched(code, "glm53f_fwd_gemv_bf16")
    }

    /// cuBLAS, whatever the policy.
    ///
    /// # Safety
    ///
    /// As for [`Gemm::bf16`].
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn cublas(
        &self,
        x: *const u16,
        ldx: usize,
        x_gstride: usize,
        w: &Bf16Mat,
        rows: usize,
        out: *mut c_void,
        ldo: usize,
        o_gstride: usize,
        out_f32: bool,
    ) -> Result<()> {
        self.blas.gemm(&GemmDesc {
            rows,
            n: w.n,
            k: w.k,
            x,
            ldx,
            w: w.ptr,
            ldw: w.ld,
            out,
            ldo,
            out_f32,
            groups: w.groups,
            x_gstride,
            w_gstride: w.gstride,
            o_gstride,
        })
    }

    /// `out [rows][n] = x . w^T` (BF16 output) for an FP8 block-128 weight.
    ///
    /// # Safety
    ///
    /// The activations (BF16, and E4M3 with scales when the GEMM needs them), the weight and
    /// `out` are device pointers to live buffers of `rows` rows (the weight: its shape).
    pub unsafe fn fp8(
        &self,
        x: &Fp8Input,
        w: &Fp8Mat,
        rows: usize,
        out: *mut u16,
        stream: &Stream,
    ) -> Result<()> {
        // SAFETY: the caller's contract.
        unsafe { self.fp8_as(x, w, rows, out, stream, false) }
    }

    /// [`Gemm::fp8`] for an FP8 KDA projection (decision D2), which
    /// [`GemmPolicy::kda_prefill_w8a8`] keeps at W8A8 over 8 rows.
    ///
    /// # Safety
    ///
    /// As for [`Gemm::fp8`].
    pub unsafe fn fp8_kda(
        &self,
        x: &Fp8Input,
        w: &Fp8Mat,
        rows: usize,
        out: *mut u16,
        stream: &Stream,
    ) -> Result<()> {
        // SAFETY: the caller's contract.
        unsafe { self.fp8_as(x, w, rows, out, stream, true) }
    }

    unsafe fn fp8_as(
        &self,
        x: &Fp8Input,
        w: &Fp8Mat,
        rows: usize,
        out: *mut u16,
        stream: &Stream,
        kda: bool,
    ) -> Result<()> {
        if rows == 0 {
            return Ok(());
        }
        let quant = self.policy.needs_quant(rows, kda);
        if quant && (x.q.is_null() || x.scales.is_null()) {
            return Err(invalid!(
                "an FP8 GEMM over {rows} rows needs E4M3 activations"
            ));
        }
        if rows <= 8 {
            let ksplit = decode_ksplit(w.n, w.k);
            if ksplit > 1 && (ksplit * rows * w.n > PARTIAL_FLOATS || w.n / 8 > SYNC_COUNTERS) {
                return Err(invalid!("FP8 split-K scratch too small for {w:?}"));
            }
            let (xp, xs, a8): (*const c_void, *const f32, i32) = if quant {
                (x.q.cast(), x.scales, 1)
            } else {
                (x.bf16.cast(), core::ptr::null(), 0)
            };
            // SAFETY: live device buffers sized by the caller; scratch checked above.
            let code = unsafe {
                lffi::glm53f_fp8_gemm_decode_fused(
                    xp,
                    xs,
                    a8,
                    w.w,
                    w.scales,
                    rows as i32,
                    w.n as i32,
                    w.k as i32,
                    ksplit as i32,
                    self.partials.ptr(0),
                    self.sync.ptr(0),
                    out,
                    stream.raw().cast(),
                )
            };
            launched(code, "glm53f_fp8_gemm_decode_fused")
        } else if self.policy.w8a16(kda) {
            // SAFETY: as documented; the tiles' scratch is the engine's.
            unsafe { self.fp8_w8a16(x.bf16, w, rows, out, stream) }
        } else {
            let flags = if self.policy.prefill_promote_k32 {
                lffi::PREFILL_PROMOTE_K32
            } else {
                0
            };
            // SAFETY: live device buffers sized by the caller.
            let code = unsafe {
                lffi::glm53f_fp8_gemm_prefill(
                    x.q,
                    x.scales,
                    w.w,
                    w.scales,
                    rows as i32,
                    w.n as i32,
                    w.k as i32,
                    flags,
                    out,
                    core::ptr::null_mut(),
                    stream.raw().cast(),
                )
            };
            launched(code, "glm53f_fp8_gemm_prefill")
        }
    }
}

impl Gemm {
    /// `out [rows][n] = x . w^T` with BF16 activations for an FP8 block-128 weight: tiles of the
    /// weight's rows dequantized to BF16 (`glm53f_fp8_dequant_bf16`), each multiplied by cuBLAS
    /// into its columns of `out`.
    ///
    /// # Safety
    ///
    /// As for [`Gemm::fp8`] (the BF16 activations `[rows][k]`).
    pub unsafe fn fp8_w8a16(
        &self,
        x: *const u16,
        w: &Fp8Mat,
        rows: usize,
        out: *mut u16,
        stream: &Stream,
    ) -> Result<()> {
        let tiles = self
            .dequant
            .as_ref()
            .ok_or_else(|| invalid!("the W8A16 prefill path needs an engine built for it"))?;
        let tile = Self::w8a16_tile_rows(w.n, w.k);
        if tile * w.k * 2 > tiles.bytes() {
            return Err(invalid!("W8A16 tiles too small for {w:?}"));
        }
        let mut n0 = 0;
        while n0 < w.n {
            let nt = tile.min(w.n - n0);
            // SAFETY: rows n0 .. n0 + nt of the weight; the tile buffer holds nt x k BF16.
            let code = unsafe {
                lffi::glm53f_fp8_dequant_bf16(
                    w.w,
                    w.scales,
                    w.n as i32,
                    w.k as i32,
                    n0 as i32,
                    nt as i32,
                    tiles.ptr(0),
                    stream.raw().cast(),
                )
            };
            launched(code, "glm53f_fp8_dequant_bf16")?;
            let t = Bf16Mat {
                ptr: tiles.ptr(0),
                n: nt,
                k: w.k,
                ld: w.k,
                groups: 1,
                gstride: 0,
            };
            // SAFETY: x [rows][k]; out's columns n0 .. n0 + nt of rows of n.
            unsafe {
                self.cublas(
                    x,
                    w.k,
                    0,
                    &t,
                    rows,
                    out.wrapping_add(n0).cast(),
                    w.n,
                    0,
                    false,
                )
            }?;
            n0 += nt;
        }
        Ok(())
    }
}

/// Quantize BF16 rows `[rows][cols]` to E4M3 per 128-group (`glm53f-layers`' `act_quant`).
///
/// # Safety
///
/// `x`, `q` and `scales` are device pointers to live buffers of `rows` rows of `cols` values
/// (`scales`: `cols / 128` per row).
pub unsafe fn act_quant(
    x: *const u16,
    q: *mut u8,
    scales: *mut f32,
    rows: usize,
    cols: usize,
    stream: &Stream,
) -> Result<()> {
    if rows == 0 {
        return Ok(());
    }
    // SAFETY: live device buffers sized by the caller.
    let code = unsafe {
        lffi::glm53f_act_quant(x, q, scales, rows as i32, cols as i32, stream.raw().cast())
    };
    launched(code, "glm53f_act_quant")
}
