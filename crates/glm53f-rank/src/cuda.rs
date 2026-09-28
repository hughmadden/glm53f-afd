//! Minimal CUDA runtime bindings for the daemon's device-memory path.
//!
//! Only the calls the serving binary needs: allocation, H2D/D2H copy, memset,
//! stream create/sync/destroy, and error-string readback. Declarations only —
//! they resolve against `libcudart` (linked by `build.rs` under the `cuda`
//! feature). On the CPU merge gate these symbols are never referenced, so no
//! nvcc or cudart is required there.

use core::ffi::{c_char, c_int, c_uint, c_void};

/// `cudaError_t`: an enum, `cudaSuccess == 0`.
pub type CudaError = c_int;

/// `cudaStream_t`: an opaque pointer.
pub type CudaStream = *mut c_void;

/// `cudaSuccess`.
pub const SUCCESS: CudaError = 0;
/// `cudaMemcpyHostToDevice`.
pub const H2D: c_int = 1;
/// `cudaMemcpyDeviceToHost`.
pub const D2H: c_int = 2;
/// `cudaHostRegisterMapped` — pin + map host memory into the device address
/// space (zero-copy on unified memory).
pub const HOST_REGISTER_MAPPED: c_uint = 0x02;

unsafe extern "C" {
    pub fn cudaMalloc(ptr: *mut *mut c_void, size: usize) -> CudaError;
    pub fn cudaFree(ptr: *mut c_void) -> CudaError;
    pub fn cudaMemcpy(dst: *mut c_void, src: *const c_void, count: usize, kind: c_int) -> CudaError;
    pub fn cudaMemcpyAsync(dst: *mut c_void, src: *const c_void, count: usize, kind: c_int, stream: *mut c_void) -> CudaError;
    pub fn cudaMemset(ptr: *mut c_void, value: c_int, count: usize) -> CudaError;
    pub fn cudaDeviceSynchronize() -> CudaError;
    pub fn cudaStreamCreate(stream: *mut CudaStream) -> CudaError;
    pub fn cudaStreamSynchronize(stream: CudaStream) -> CudaError;
    pub fn cudaStreamDestroy(stream: CudaStream) -> CudaError;
    pub fn cudaGetLastError() -> CudaError;
    pub fn cudaGetErrorString(err: CudaError) -> *const c_char;
    pub fn cudaHostRegister(ptr: *mut c_void, size: usize, flags: c_uint) -> CudaError;
    pub fn cudaHostGetDevicePointer(ptr: *mut *mut c_void, host_ptr: *mut c_void, flags: c_uint) -> CudaError;
    pub fn cudaHostUnregister(ptr: *mut c_void) -> CudaError;
}

/// The runtime's description of a CUDA error, or a fallback string on a null.
pub fn error_string(err: CudaError) -> String {
    if err == SUCCESS {
        return "cudaSuccess".to_string();
    }
    // SAFETY: cudaGetErrorString returns a pointer to a static NUL-terminated
    // string owned by the runtime; it is only read, never freed.
    let p = unsafe { cudaGetErrorString(err) };
    if p.is_null() {
        return format!("cuda error {err}");
    }
    // SAFETY: p points to a static NUL-terminated C string.
    let bytes = unsafe { core::ffi::CStr::from_ptr(p) }.to_bytes();
    String::from_utf8_lossy(bytes).into_owned()
}
