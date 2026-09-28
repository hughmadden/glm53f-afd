//! The CUDA runtime calls this crate makes (feature `cuda`), declared against `libcudart`.

use core::ffi::{c_char, c_int, c_uint, c_void};

/// `cudaError_t` (0 = success).
pub type CudaError = i32;
/// `cudaStream_t`.
pub type RawStream = *mut c_void;
/// `cudaEvent_t`.
pub type RawEvent = *mut c_void;

pub const SUCCESS: CudaError = 0;
pub const MEMCPY_H2D: c_int = 1;
pub const MEMCPY_D2H: c_int = 2;
pub const MEMCPY_D2D: c_int = 3;
/// `cudaHostAllocMapped`: page-locked and mapped into the device's address space.
pub const HOST_ALLOC_MAPPED: c_uint = 2;
/// `cudaHostAllocPortable`.
pub const HOST_ALLOC_PORTABLE: c_uint = 1;
/// `cudaStreamNonBlocking`.
pub const STREAM_NON_BLOCKING: c_uint = 1;
/// `cudaDevAttrMultiProcessorCount`.
pub const ATTR_SM_COUNT: c_int = 16;
/// `cudaDevAttrMemoryClockRate` (kHz) and `cudaDevAttrGlobalMemoryBusWidth` (bits).
pub const ATTR_MEM_CLOCK_KHZ: c_int = 36;
pub const ATTR_MEM_BUS_BITS: c_int = 37;
/// `cudaDevAttrComputeCapabilityMajor` / `Minor`.
pub const ATTR_CC_MAJOR: c_int = 75;
pub const ATTR_CC_MINOR: c_int = 76;

unsafe extern "C" {
    pub fn cudaMalloc(ptr: *mut *mut c_void, size: usize) -> CudaError;
    pub fn cudaFree(ptr: *mut c_void) -> CudaError;
    pub fn cudaHostAlloc(ptr: *mut *mut c_void, size: usize, flags: c_uint) -> CudaError;
    pub fn cudaFreeHost(ptr: *mut c_void) -> CudaError;
    pub fn cudaHostGetDevicePointer(
        dptr: *mut *mut c_void,
        hptr: *mut c_void,
        flags: c_uint,
    ) -> CudaError;
    pub fn cudaMemcpy(dst: *mut c_void, src: *const c_void, count: usize, kind: c_int)
        -> CudaError;
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
    pub fn cudaDeviceGetAttribute(value: *mut c_int, attr: c_int, device: c_int) -> CudaError;
    pub fn cudaDeviceSynchronize() -> CudaError;
    pub fn cudaStreamCreateWithFlags(stream: *mut RawStream, flags: c_uint) -> CudaError;
    pub fn cudaStreamSynchronize(stream: RawStream) -> CudaError;
    pub fn cudaStreamDestroy(stream: RawStream) -> CudaError;
    pub fn cudaStreamWaitEvent(stream: RawStream, event: RawEvent, flags: c_uint) -> CudaError;
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
    // SAFETY: a static C string owned by the runtime.
    unsafe { core::ffi::CStr::from_ptr(p) }
        .to_string_lossy()
        .into_owned()
}
