//! The one cuBLAS call the forward's GEMMs use (feature `cuda`): BF16 inputs, f32 accumulation,
//! f32 output.
//!
//! Row-major `Y [m][n] = X [m][k] . W^T` with `W [n][k]` is, in cuBLAS's column-major terms,
//! `C (n x m, ldc = ldy) = op(A) op(B)` with `A = W` transposed (`lda = ldw`) and `B = X`
//! (`ldb = ldx`): `cublasGemmEx(T, N, n, m, k, W, ldw, X, ldx, Y, ldy)`. The same mapping and
//! handle setup as `crates/glm53f-forward/src/cublas.rs` (see PROVENANCE.md).

use core::ffi::{c_int, c_void};

use crate::cuda::RawStream;
use crate::device::{DeviceBuffer, Stream};

type Handle = *mut c_void;

const OP_N: c_int = 0;
const OP_T: c_int = 1;
/// `CUDA_R_16BF`, `CUDA_R_32F`.
const R_16BF: c_int = 14;
const R_32F: c_int = 0;
/// `CUBLAS_COMPUTE_32F`.
const COMPUTE_32F: c_int = 68;
/// `CUBLAS_GEMM_DEFAULT`.
const GEMM_DEFAULT: c_int = -1;
/// `CUBLAS_DEFAULT_MATH | CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION`.
const MATH_NO_REDUCED_REDUCTION: c_int = 16;

unsafe extern "C" {
    fn cublasCreate_v2(handle: *mut Handle) -> c_int;
    fn cublasDestroy_v2(handle: Handle) -> c_int;
    fn cublasSetStream_v2(handle: Handle, stream: RawStream) -> c_int;
    fn cublasSetMathMode(handle: Handle, mode: c_int) -> c_int;
    fn cublasSetWorkspace_v2(handle: Handle, workspace: *mut c_void, bytes: usize) -> c_int;
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
    ) -> c_int;
}

fn ok(status: c_int, what: &str) -> Result<(), String> {
    if status == 0 {
        Ok(())
    } else {
        Err(format!("{what}: cuBLAS status {status}"))
    }
}

fn int(v: usize, what: &str) -> Result<c_int, String> {
    c_int::try_from(v).map_err(|_| format!("{what} = {v} does not fit in 32 bits"))
}

/// A cuBLAS handle bound to one stream, with its own workspace.
pub struct Blas {
    handle: Handle,
    _workspace: DeviceBuffer,
}

// SAFETY: the handle is used from one thread at a time (its owner).
unsafe impl Send for Blas {}

impl Blas {
    pub fn new(stream: &Stream, workspace_bytes: usize) -> Result<Blas, String> {
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

    /// `y [rows][n] (f32, row stride ldy) = x [rows][k] (bf16, ldx) . w^T`, `w [n][k]` bf16 (ldw).
    ///
    /// # Safety
    ///
    /// The pointers name live device buffers holding these shapes and strides.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn gemm(
        &self,
        rows: usize,
        n: usize,
        k: usize,
        x: *const u16,
        ldx: usize,
        w: *const u16,
        ldw: usize,
        y: *mut f32,
        ldy: usize,
    ) -> Result<(), String> {
        if rows == 0 || n == 0 {
            return Ok(());
        }
        let (alpha, beta) = (1.0f32, 0.0f32);
        // SAFETY: the caller's pointers are live and large enough.
        let status = unsafe {
            cublasGemmEx(
                self.handle,
                OP_T,
                OP_N,
                int(n, "n")?,
                int(rows, "rows")?,
                int(k, "k")?,
                (&alpha as *const f32).cast(),
                w.cast(),
                R_16BF,
                int(ldw, "ldw")?,
                x.cast(),
                R_16BF,
                int(ldx, "ldx")?,
                (&beta as *const f32).cast(),
                y.cast(),
                R_32F,
                int(ldy, "ldy")?,
                COMPUTE_32F,
                GEMM_DEFAULT,
            )
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
