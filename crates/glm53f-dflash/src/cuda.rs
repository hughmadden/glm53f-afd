//! The few CUDA runtime calls this crate uses (feature `cuda`), declared against `libcudart`.

use core::ffi::{c_char, c_int, c_void};

/// `cudaError_t`.
pub type CudaError = c_int;
/// `cudaStream_t`.
pub type RawStream = *mut c_void;
/// `cudaEvent_t`.
pub type RawEvent = *mut c_void;

pub const SUCCESS: CudaError = 0;
pub const MEMCPY_H2D: c_int = 1;
pub const MEMCPY_D2H: c_int = 2;
pub const MEMCPY_D2D: c_int = 3;

unsafe extern "C" {
    pub fn cudaMalloc(ptr: *mut *mut c_void, size: usize) -> CudaError;
    pub fn cudaFree(ptr: *mut c_void) -> CudaError;
    pub fn cudaMemcpyAsync(
        dst: *mut c_void,
        src: *const c_void,
        count: usize,
        kind: c_int,
        stream: RawStream,
    ) -> CudaError;
    pub fn cudaMemsetAsync(
        ptr: *mut c_void,
        value: c_int,
        count: usize,
        stream: RawStream,
    ) -> CudaError;
    pub fn cudaMemGetInfo(free: *mut usize, total: *mut usize) -> CudaError;
    pub fn cudaGetDeviceCount(count: *mut c_int) -> CudaError;
    pub fn cudaStreamCreate(stream: *mut RawStream) -> CudaError;
    pub fn cudaStreamSynchronize(stream: RawStream) -> CudaError;
    pub fn cudaStreamDestroy(stream: RawStream) -> CudaError;
    pub fn cudaEventCreate(event: *mut RawEvent) -> CudaError;
    pub fn cudaEventRecord(event: RawEvent, stream: RawStream) -> CudaError;
    pub fn cudaEventSynchronize(event: RawEvent) -> CudaError;
    pub fn cudaEventElapsedTime(ms: *mut f32, start: RawEvent, end: RawEvent) -> CudaError;
    pub fn cudaEventDestroy(event: RawEvent) -> CudaError;
    pub fn cudaGetLastError() -> CudaError;
    pub fn cudaGetErrorString(err: CudaError) -> *const c_char;
}

/// The runtime's name for an error code.
pub fn error_string(err: CudaError) -> String {
    // SAFETY: cudaGetErrorString returns a static NUL-terminated string (or null).
    let p = unsafe { cudaGetErrorString(err) };
    if p.is_null() {
        return format!("CUDA error {err}");
    }
    // SAFETY: p is a static NUL-terminated C string owned by the runtime.
    unsafe { core::ffi::CStr::from_ptr(p) }
        .to_string_lossy()
        .into_owned()
}

/// `Ok` for `cudaSuccess`, else an error naming `what`.
pub fn check(code: CudaError, what: &str) -> Result<(), String> {
    if code == SUCCESS {
        Ok(())
    } else {
        Err(format!("{what}: {} ({code})", error_string(code)))
    }
}
