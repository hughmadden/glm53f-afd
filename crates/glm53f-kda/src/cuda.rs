//! The few CUDA runtime calls the wrappers, tests and benchmark need (feature `cuda`),
//! declared directly against `libcudart`.

use core::ffi::{c_char, c_int, c_void};

use crate::ffi::{CudaError, Stream};

pub const SUCCESS: CudaError = 0;
pub const MEMCPY_H2D: c_int = 1;
pub const MEMCPY_D2H: c_int = 2;
pub const MEMCPY_D2D: c_int = 3;
/// `cudaDevAttrMultiProcessorCount`.
pub const ATTR_SM_COUNT: c_int = 16;
/// `cudaDevAttrComputeCapabilityMajor`.
pub const ATTR_CC_MAJOR: c_int = 75;
/// `cudaDevAttrComputeCapabilityMinor`.
pub const ATTR_CC_MINOR: c_int = 76;

/// `cudaEvent_t`.
pub type Event = *mut c_void;

unsafe extern "C" {
    pub fn cudaMalloc(ptr: *mut *mut c_void, size: usize) -> CudaError;
    pub fn cudaFree(ptr: *mut c_void) -> CudaError;
    pub fn cudaMemcpy(dst: *mut c_void, src: *const c_void, count: usize, kind: c_int)
        -> CudaError;
    pub fn cudaMemset(ptr: *mut c_void, value: c_int, count: usize) -> CudaError;
    pub fn cudaMemGetInfo(free: *mut usize, total: *mut usize) -> CudaError;
    pub fn cudaGetDeviceCount(count: *mut c_int) -> CudaError;
    pub fn cudaDeviceGetAttribute(value: *mut c_int, attr: c_int, device: c_int) -> CudaError;
    pub fn cudaDeviceSynchronize() -> CudaError;
    pub fn cudaStreamCreate(stream: *mut Stream) -> CudaError;
    pub fn cudaStreamSynchronize(stream: Stream) -> CudaError;
    pub fn cudaStreamDestroy(stream: Stream) -> CudaError;
    pub fn cudaEventCreate(event: *mut Event) -> CudaError;
    pub fn cudaEventRecord(event: Event, stream: Stream) -> CudaError;
    pub fn cudaEventSynchronize(event: Event) -> CudaError;
    pub fn cudaEventElapsedTime(ms: *mut f32, start: Event, end: Event) -> CudaError;
    pub fn cudaEventDestroy(event: Event) -> CudaError;
    pub fn cudaGetLastError() -> CudaError;
    pub fn cudaGetErrorString(err: CudaError) -> *const c_char;
}

/// The runtime's name for an error code.
pub fn error_string(err: CudaError) -> String {
    // SAFETY: cudaGetErrorString returns a pointer to a static NUL-terminated string (or null).
    let p = unsafe { cudaGetErrorString(err) };
    if p.is_null() {
        return format!("CUDA error {err}");
    }
    // SAFETY: p is a static NUL-terminated C string owned by the runtime.
    unsafe { core::ffi::CStr::from_ptr(p) }
        .to_string_lossy()
        .into_owned()
}
