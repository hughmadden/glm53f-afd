//! The few cuBLAS calls the BF16 GEMMs use (feature `cuda`), declared against `libcublas`.
//!
//! Row-major `Y [m][n] = X [m][k] . W^T` with `W [n][k]` is, in cuBLAS's column-major terms,
//! `C (n x m, ldc = ldy) = op(A) op(B)` with `A = W` read transposed (`lda = ldw`) and
//! `B = X` (`ldb = ldx`): `cublasGemmEx(T, N, n, m, k, W, ldw, X, ldx, Y, ldy)`.

use core::ffi::{c_int, c_void};

use crate::cuda::RawStream;
use crate::device::{DeviceBuffer, Stream};
use crate::error::{Error, Result};

type Handle = *mut c_void;
type Status = c_int;

const OP_N: c_int = 0;
const OP_T: c_int = 1;
/// `CUDA_R_16BF`, `CUDA_R_32F`.
const R_16BF: c_int = 14;
const R_32F: c_int = 0;
/// `CUBLAS_COMPUTE_32F`.
const COMPUTE_32F: c_int = 68;
/// `CUBLAS_GEMM_DEFAULT`.
const GEMM_DEFAULT: c_int = -1;
/// `CUBLAS_DEFAULT_MATH | CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION`: f32 split-K
/// reductions even with a BF16 output.
const MATH_NO_REDUCED_REDUCTION: c_int = 16;

unsafe extern "C" {
    fn cublasCreate_v2(handle: *mut Handle) -> Status;
    fn cublasDestroy_v2(handle: Handle) -> Status;
    fn cublasSetStream_v2(handle: Handle, stream: RawStream) -> Status;
    fn cublasSetMathMode(handle: Handle, mode: c_int) -> Status;
    fn cublasSetWorkspace_v2(handle: Handle, workspace: *mut c_void, bytes: usize) -> Status;
    fn cublasGemmEx(
        handle: Handle,
        transa: c_int,
        transb: c_int,
        m: c_int,
        n: c_int,
        k: c_int,
        alpha: *const c_void,
        a: *const c_void,
        atype: c_int,
        lda: c_int,
        b: *const c_void,
        btype: c_int,
        ldb: c_int,
        beta: *const c_void,
        c: *mut c_void,
        ctype: c_int,
        ldc: c_int,
        compute: c_int,
        algo: c_int,
    ) -> Status;
    fn cublasGemmStridedBatchedEx(
        handle: Handle,
        transa: c_int,
        transb: c_int,
        m: c_int,
        n: c_int,
        k: c_int,
        alpha: *const c_void,
        a: *const c_void,
        atype: c_int,
        lda: c_int,
        stride_a: i64,
        b: *const c_void,
        btype: c_int,
        ldb: c_int,
        stride_b: i64,
        beta: *const c_void,
        c: *mut c_void,
        ctype: c_int,
        ldc: c_int,
        stride_c: i64,
        batch: c_int,
        compute: c_int,
        algo: c_int,
    ) -> Status;
}

fn ok(status: Status, what: &str) -> Result<()> {
    if status == 0 {
        Ok(())
    } else {
        Err(Error::Cuda {
            code: status,
            what: format!("{what}: cuBLAS status {status}"),
        })
    }
}

fn int(v: usize, what: &str) -> Result<c_int> {
    c_int::try_from(v).map_err(|_| Error::Invalid(format!("{what} = {v} does not fit in 32 bits")))
}

/// One BF16 GEMM of the row-major form above, possibly batched over `groups`.
#[derive(Clone, Copy, Debug)]
pub struct GemmDesc {
    pub rows: usize,
    pub n: usize,
    pub k: usize,
    pub x: *const u16,
    pub ldx: usize,
    pub w: *const u16,
    pub ldw: usize,
    pub out: *mut c_void,
    pub ldo: usize,
    pub out_f32: bool,
    /// Groups and the element strides between them (1: a plain GEMM).
    pub groups: usize,
    pub x_gstride: usize,
    pub w_gstride: usize,
    pub o_gstride: usize,
}

/// A cuBLAS handle bound to one stream, with its own workspace.
pub struct Blas {
    handle: Handle,
    _workspace: DeviceBuffer,
}

// SAFETY: the handle is used from one thread at a time (the forward's owner).
unsafe impl Send for Blas {}

impl Blas {
    /// A handle on `stream` with a `workspace_bytes` workspace (cuBLAS's own recommendation
    /// on recent GPUs is 32 MiB).
    pub fn new(stream: &Stream, workspace_bytes: usize) -> Result<Blas> {
        let mut handle = core::ptr::null_mut();
        // SAFETY: a valid out-pointer.
        ok(unsafe { cublasCreate_v2(&mut handle) }, "cublasCreate")?;
        let blas = Blas {
            handle,
            _workspace: DeviceBuffer::alloc(workspace_bytes)?,
        };
        // SAFETY: a live handle, stream and workspace.
        unsafe {
            ok(cublasSetStream_v2(handle, stream.raw()), "cublasSetStream")?;
            ok(
                cublasSetMathMode(handle, MATH_NO_REDUCED_REDUCTION),
                "cublasSetMathMode",
            )?;
            ok(
                cublasSetWorkspace_v2(handle, blas._workspace.ptr::<c_void>(0), workspace_bytes),
                "cublasSetWorkspace",
            )?;
        }
        Ok(blas)
    }

    /// Launch `d` (BF16 inputs, f32 accumulation, BF16 or f32 output) on the handle's stream.
    ///
    /// # Safety
    ///
    /// The descriptor's pointers name live device buffers that hold its shapes and strides.
    pub unsafe fn gemm(&self, d: &GemmDesc) -> Result<()> {
        let alpha = 1.0f32;
        let beta = 0.0f32;
        let ctype = if d.out_f32 { R_32F } else { R_16BF };
        let (m, n, k) = (int(d.n, "n")?, int(d.rows, "rows")?, int(d.k, "k")?);
        let (lda, ldb, ldc) = (int(d.ldw, "ldw")?, int(d.ldx, "ldx")?, int(d.ldo, "ldo")?);
        // SAFETY: the caller's descriptor names live device buffers large enough for the shapes.
        let status = unsafe {
            if d.groups <= 1 {
                cublasGemmEx(
                    self.handle,
                    OP_T,
                    OP_N,
                    m,
                    n,
                    k,
                    (&alpha as *const f32).cast(),
                    d.w.cast(),
                    R_16BF,
                    lda,
                    d.x.cast(),
                    R_16BF,
                    ldb,
                    (&beta as *const f32).cast(),
                    d.out,
                    ctype,
                    ldc,
                    COMPUTE_32F,
                    GEMM_DEFAULT,
                )
            } else {
                cublasGemmStridedBatchedEx(
                    self.handle,
                    OP_T,
                    OP_N,
                    m,
                    n,
                    k,
                    (&alpha as *const f32).cast(),
                    d.w.cast(),
                    R_16BF,
                    lda,
                    d.w_gstride as i64,
                    d.x.cast(),
                    R_16BF,
                    ldb,
                    d.x_gstride as i64,
                    (&beta as *const f32).cast(),
                    d.out,
                    ctype,
                    ldc,
                    d.o_gstride as i64,
                    int(d.groups, "groups")?,
                    COMPUTE_32F,
                    GEMM_DEFAULT,
                )
            }
        };
        ok(status, "cublasGemmEx")
    }
}

impl Drop for Blas {
    fn drop(&mut self) {
        // SAFETY: created by cublasCreate, destroyed once.
        unsafe { cublasDestroy_v2(self.handle) };
    }
}
