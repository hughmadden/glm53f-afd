//! Minimal device memory, streams and events (feature `cuda`).

use core::ffi::c_void;
use std::fmt;

use crate::cuda;
use crate::ffi::{CudaError, Stream as RawStream};

/// An error from the CUDA runtime, or an argument the wrappers rejected.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Error {
    Cuda(CudaError, String),
    Invalid(String),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::Cuda(code, what) => write!(f, "{what}: {} ({code})", cuda::error_string(*code)),
            Error::Invalid(m) => write!(f, "invalid argument: {m}"),
        }
    }
}

impl std::error::Error for Error {}

/// `Ok` for `cudaSuccess`, else the error with `what` as context.
pub fn check(code: CudaError, what: &str) -> Result<(), Error> {
    if code == cuda::SUCCESS {
        Ok(())
    } else {
        Err(Error::Cuda(code, what.to_string()))
    }
}

/// Element types a buffer can be viewed as.
pub trait Pod: Copy + Default + 'static {}
impl Pod for f32 {}
impl Pod for u16 {}
impl Pod for i32 {}
impl Pod for i64 {}
impl Pod for u8 {}

/// One device allocation, freed on drop. Device memory is outside Rust's aliasing rules, so
/// every method takes `&self`; kernels write through the raw pointer.
pub struct DeviceBuffer {
    ptr: *mut c_void,
    bytes: usize,
}

// SAFETY: a device allocation is usable from any host thread of the process; the buffer frees
// it exactly once.
unsafe impl Send for DeviceBuffer {}
unsafe impl Sync for DeviceBuffer {}

impl DeviceBuffer {
    /// `bytes` of uninitialized device memory.
    pub fn alloc(bytes: usize) -> Result<Self, Error> {
        let mut ptr = core::ptr::null_mut();
        if bytes > 0 {
            // SAFETY: ptr is a valid out-pointer.
            check(unsafe { cuda::cudaMalloc(&mut ptr, bytes) }, "cudaMalloc")?;
        }
        Ok(DeviceBuffer { ptr, bytes })
    }

    /// `bytes` of zeroed device memory.
    pub fn zeroed(bytes: usize) -> Result<Self, Error> {
        let b = Self::alloc(bytes)?;
        b.zero()?;
        Ok(b)
    }

    /// A buffer holding a copy of `data`.
    pub fn from_slice<T: Pod>(data: &[T]) -> Result<Self, Error> {
        let b = Self::alloc(std::mem::size_of_val(data))?;
        b.upload(data)?;
        Ok(b)
    }

    pub fn as_ptr(&self) -> *mut c_void {
        self.ptr
    }

    /// The device pointer as `*mut T`, offset by `offset` elements (bounds are the caller's).
    pub fn ptr<T>(&self, offset: usize) -> *mut T {
        (self.ptr as *mut T).wrapping_add(offset)
    }

    pub fn bytes(&self) -> usize {
        self.bytes
    }

    /// Whole elements of `T` the buffer holds.
    pub fn len<T>(&self) -> usize {
        self.bytes / std::mem::size_of::<T>()
    }

    pub fn is_empty(&self) -> bool {
        self.bytes == 0
    }

    /// Copy `data` to the start of the buffer.
    pub fn upload<T: Pod>(&self, data: &[T]) -> Result<(), Error> {
        self.upload_at(0, data)
    }

    /// Copy `data` into the buffer at element `offset`.
    pub fn upload_at<T: Pod>(&self, offset: usize, data: &[T]) -> Result<(), Error> {
        let n = std::mem::size_of_val(data);
        let at = offset * std::mem::size_of::<T>();
        if at + n > self.bytes {
            return Err(Error::Invalid(format!(
                "upload of {n} bytes at {at} into {} bytes",
                self.bytes
            )));
        }
        if n == 0 {
            return Ok(());
        }
        // SAFETY: the destination range lies inside this allocation; data is a host slice of n bytes.
        check(
            unsafe {
                cuda::cudaMemcpy(
                    self.ptr.cast::<u8>().add(at).cast(),
                    data.as_ptr().cast(),
                    n,
                    cuda::MEMCPY_H2D,
                )
            },
            "cudaMemcpy H2D",
        )
    }

    /// Copy the first `n` elements to the host.
    pub fn download<T: Pod>(&self, n: usize) -> Result<Vec<T>, Error> {
        self.download_at(0, n)
    }

    /// Copy `n` elements from element `offset` to the host.
    pub fn download_at<T: Pod>(&self, offset: usize, n: usize) -> Result<Vec<T>, Error> {
        let bytes = n * std::mem::size_of::<T>();
        let at = offset * std::mem::size_of::<T>();
        if at + bytes > self.bytes {
            return Err(Error::Invalid(format!(
                "download of {bytes} bytes at {at} from {} bytes",
                self.bytes
            )));
        }
        let mut v = vec![T::default(); n];
        if bytes > 0 {
            // SAFETY: the source range lies inside this allocation; v holds n elements.
            check(
                unsafe {
                    cuda::cudaMemcpy(
                        v.as_mut_ptr().cast(),
                        self.ptr.cast::<u8>().add(at).cast(),
                        bytes,
                        cuda::MEMCPY_D2H,
                    )
                },
                "cudaMemcpy D2H",
            )?;
        }
        Ok(v)
    }

    /// The whole buffer as `T`.
    pub fn to_vec<T: Pod>(&self) -> Result<Vec<T>, Error> {
        self.download(self.len::<T>())
    }

    /// Copy all of `src` (no larger than this buffer) to the start of this buffer.
    pub fn copy_from(&self, src: &DeviceBuffer) -> Result<(), Error> {
        if src.bytes > self.bytes {
            return Err(Error::Invalid(
                "copy_from: source larger than destination".into(),
            ));
        }
        if src.bytes == 0 {
            return Ok(());
        }
        // SAFETY: both ranges are inside their allocations.
        check(
            unsafe { cuda::cudaMemcpy(self.ptr, src.ptr, src.bytes, cuda::MEMCPY_D2D) },
            "cudaMemcpy D2D",
        )
    }

    pub fn zero(&self) -> Result<(), Error> {
        if self.bytes == 0 {
            return Ok(());
        }
        // SAFETY: the range is this allocation.
        check(
            unsafe { cuda::cudaMemset(self.ptr, 0, self.bytes) },
            "cudaMemset",
        )
    }
}

impl Drop for DeviceBuffer {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            // SAFETY: allocated by cudaMalloc, freed once. cudaFree waits for work in flight.
            unsafe { cuda::cudaFree(self.ptr) };
        }
    }
}

/// A CUDA stream, destroyed on drop.
pub struct Stream(RawStream);

impl Stream {
    pub fn new() -> Result<Self, Error> {
        let mut s = core::ptr::null_mut();
        // SAFETY: s is a valid out-pointer.
        check(
            unsafe { cuda::cudaStreamCreate(&mut s) },
            "cudaStreamCreate",
        )?;
        Ok(Stream(s))
    }

    pub fn raw(&self) -> RawStream {
        self.0
    }

    pub fn synchronize(&self) -> Result<(), Error> {
        // SAFETY: a live stream.
        check(
            unsafe { cuda::cudaStreamSynchronize(self.0) },
            "cudaStreamSynchronize",
        )
    }
}

impl Drop for Stream {
    fn drop(&mut self) {
        // SAFETY: created by cudaStreamCreate, destroyed once.
        unsafe { cuda::cudaStreamDestroy(self.0) };
    }
}

/// A CUDA event for timing, destroyed on drop.
pub struct Event(cuda::Event);

impl Event {
    pub fn new() -> Result<Self, Error> {
        let mut e = core::ptr::null_mut();
        // SAFETY: e is a valid out-pointer.
        check(unsafe { cuda::cudaEventCreate(&mut e) }, "cudaEventCreate")?;
        Ok(Event(e))
    }

    pub fn record(&self, stream: &Stream) -> Result<(), Error> {
        // SAFETY: a live event and stream.
        check(
            unsafe { cuda::cudaEventRecord(self.0, stream.raw()) },
            "cudaEventRecord",
        )
    }

    /// Milliseconds from `start` to this event (waits for this event).
    pub fn elapsed_ms_since(&self, start: &Event) -> Result<f32, Error> {
        // SAFETY: live events.
        check(
            unsafe { cuda::cudaEventSynchronize(self.0) },
            "cudaEventSynchronize",
        )?;
        let mut ms = 0.0f32;
        // SAFETY: ms is a valid out-pointer; both events are live and recorded.
        check(
            unsafe { cuda::cudaEventElapsedTime(&mut ms, start.0, self.0) },
            "cudaEventElapsedTime",
        )?;
        Ok(ms)
    }
}

impl Drop for Event {
    fn drop(&mut self) {
        // SAFETY: created by cudaEventCreate, destroyed once.
        unsafe { cuda::cudaEventDestroy(self.0) };
    }
}

/// Number of CUDA devices (0 when the driver or a device is missing).
pub fn device_count() -> usize {
    let mut n = 0;
    // SAFETY: n is a valid out-pointer.
    let rc = unsafe { cuda::cudaGetDeviceCount(&mut n) };
    if rc == cuda::SUCCESS {
        n as usize
    } else {
        // Clear the error so later calls are not affected.
        // SAFETY: no arguments.
        unsafe { cuda::cudaGetLastError() };
        0
    }
}

/// An integer attribute of device 0.
pub fn attribute(attr: i32) -> Result<i32, Error> {
    let mut v = 0;
    // SAFETY: v is a valid out-pointer.
    check(
        unsafe { cuda::cudaDeviceGetAttribute(&mut v, attr, 0) },
        "cudaDeviceGetAttribute",
    )?;
    Ok(v)
}

/// Free and total device memory, in bytes.
pub fn mem_info() -> Result<(usize, usize), Error> {
    let (mut free, mut total) = (0, 0);
    // SAFETY: valid out-pointers.
    check(
        unsafe { cuda::cudaMemGetInfo(&mut free, &mut total) },
        "cudaMemGetInfo",
    )?;
    Ok((free, total))
}

pub fn synchronize() -> Result<(), Error> {
    // SAFETY: no arguments.
    check(
        unsafe { cuda::cudaDeviceSynchronize() },
        "cudaDeviceSynchronize",
    )
}
