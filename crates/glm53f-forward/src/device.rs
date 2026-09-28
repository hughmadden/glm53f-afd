//! Device and page-locked host memory, streams and events (feature `cuda`).

use core::ffi::c_void;

use crate::cuda::{self, CudaError, RawEvent, RawStream};
use crate::error::{Error, Result};

/// `Ok` for `cudaSuccess`, else the error with `what` and the runtime's name for it.
pub fn check(code: CudaError, what: &str) -> Result<()> {
    if code == cuda::SUCCESS {
        Ok(())
    } else {
        Err(Error::Cuda {
            code,
            what: format!("{what}: {}", cuda::error_string(code)),
        })
    }
}

/// A kernel launcher's status: 1 is an argument the launcher rejected.
pub fn launched(code: i32, what: &str) -> Result<()> {
    match code {
        0 => Ok(()),
        1 => Err(Error::Invalid(format!("{what}: rejected by the launcher"))),
        _ => check(code, what),
    }
}

/// Plain values a buffer can hold.
pub trait Pod: Copy + Default + 'static {}
impl Pod for f32 {}
impl Pod for u16 {}
impl Pod for i32 {}
impl Pod for u32 {}
impl Pod for i64 {}
impl Pod for u8 {}

/// One device allocation, freed on drop. Device memory is outside Rust's aliasing rules:
/// every method takes `&self`, and kernels write through the raw pointers.
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
    pub fn alloc(bytes: usize) -> Result<DeviceBuffer> {
        let mut ptr = core::ptr::null_mut();
        if bytes > 0 {
            // SAFETY: a valid out-pointer.
            let code = unsafe { cuda::cudaMalloc(&mut ptr, bytes) };
            if code != cuda::SUCCESS {
                // SAFETY: clear the sticky-free error so later calls are unaffected.
                unsafe { cuda::cudaGetLastError() };
                return Err(Error::OutOfMemory(format!(
                    "cudaMalloc of {bytes} bytes: {}",
                    cuda::error_string(code)
                )));
            }
        }
        Ok(DeviceBuffer { ptr, bytes })
    }

    /// `bytes` of zeroed device memory.
    pub fn zeroed(bytes: usize) -> Result<DeviceBuffer> {
        let b = DeviceBuffer::alloc(bytes)?;
        b.zero()?;
        Ok(b)
    }

    /// A buffer holding a copy of `data`.
    pub fn from_slice<T: Pod>(data: &[T]) -> Result<DeviceBuffer> {
        let b = DeviceBuffer::alloc(std::mem::size_of_val(data))?;
        b.upload(data)?;
        Ok(b)
    }

    pub fn bytes(&self) -> usize {
        self.bytes
    }

    pub fn is_empty(&self) -> bool {
        self.bytes == 0
    }

    /// Whole elements of `T` the buffer holds.
    pub fn len<T>(&self) -> usize {
        self.bytes / std::mem::size_of::<T>()
    }

    /// The device pointer as `*mut T`, offset by `offset` elements (bounds are the caller's).
    pub fn ptr<T>(&self, offset: usize) -> *mut T {
        (self.ptr as *mut T).wrapping_add(offset)
    }

    /// The device pointer offset by `offset` bytes.
    pub fn byte_ptr(&self, offset: usize) -> *mut u8 {
        (self.ptr as *mut u8).wrapping_add(offset)
    }

    fn check_range(&self, at: usize, n: usize, what: &str) -> Result<()> {
        if at.checked_add(n).is_none_or(|e| e > self.bytes) {
            return Err(Error::Invalid(format!(
                "{what}: bytes [{at}, {at} + {n}) of a {}-byte buffer",
                self.bytes
            )));
        }
        Ok(())
    }

    /// Copy `data` to the start of the buffer (synchronous).
    pub fn upload<T: Pod>(&self, data: &[T]) -> Result<()> {
        self.upload_at(0, data)
    }

    /// Copy `data` into the buffer at element `offset` (synchronous).
    pub fn upload_at<T: Pod>(&self, offset: usize, data: &[T]) -> Result<()> {
        let n = std::mem::size_of_val(data);
        let at = offset * std::mem::size_of::<T>();
        self.check_range(at, n, "upload")?;
        if n == 0 {
            return Ok(());
        }
        // SAFETY: the range lies inside this allocation; data is a host slice of n bytes.
        check(
            unsafe {
                cuda::cudaMemcpy(
                    self.byte_ptr(at).cast(),
                    data.as_ptr().cast(),
                    n,
                    cuda::MEMCPY_H2D,
                )
            },
            "cudaMemcpy H2D",
        )
    }

    /// Copy `data` into the buffer at byte `at`, ordered on `stream`. The host slice may be
    /// reused when the call returns (pageable memory is staged before it returns).
    pub fn upload_bytes_async(&self, stream: &Stream, at: usize, data: &[u8]) -> Result<()> {
        self.check_range(at, data.len(), "upload")?;
        if data.is_empty() {
            return Ok(());
        }
        // SAFETY: as in upload_at.
        check(
            unsafe {
                cuda::cudaMemcpyAsync(
                    self.byte_ptr(at).cast(),
                    data.as_ptr().cast(),
                    data.len(),
                    cuda::MEMCPY_H2D,
                    stream.raw(),
                )
            },
            "cudaMemcpyAsync H2D",
        )
    }

    /// Typed [`DeviceBuffer::upload_bytes_async`] at element `offset`.
    pub fn upload_async<T: Pod>(&self, stream: &Stream, offset: usize, data: &[T]) -> Result<()> {
        // SAFETY: T is plain data; the byte view covers exactly the slice.
        let bytes = unsafe {
            std::slice::from_raw_parts(data.as_ptr().cast::<u8>(), std::mem::size_of_val(data))
        };
        self.upload_bytes_async(stream, offset * std::mem::size_of::<T>(), bytes)
    }

    /// Copy the first `n` elements to the host (synchronous).
    pub fn download<T: Pod>(&self, n: usize) -> Result<Vec<T>> {
        self.download_at(0, n)
    }

    /// Copy `n` elements from element `offset` to the host (synchronous).
    pub fn download_at<T: Pod>(&self, offset: usize, n: usize) -> Result<Vec<T>> {
        let bytes = n * std::mem::size_of::<T>();
        let at = offset * std::mem::size_of::<T>();
        self.check_range(at, bytes, "download")?;
        let mut v = vec![T::default(); n];
        if bytes > 0 {
            // SAFETY: the range lies inside this allocation; v holds n elements.
            check(
                unsafe {
                    cuda::cudaMemcpy(
                        v.as_mut_ptr().cast(),
                        self.byte_ptr(at).cast(),
                        bytes,
                        cuda::MEMCPY_D2H,
                    )
                },
                "cudaMemcpy D2H",
            )?;
        }
        Ok(v)
    }

    /// Copy bytes `[at, at + dst.len())` into a host slice, ordered on `stream`, and wait.
    pub fn download_bytes(&self, stream: &Stream, at: usize, dst: &mut [u8]) -> Result<()> {
        self.check_range(at, dst.len(), "download")?;
        if dst.is_empty() {
            return Ok(());
        }
        // SAFETY: both ranges are valid for dst.len() bytes.
        check(
            unsafe {
                cuda::cudaMemcpyAsync(
                    dst.as_mut_ptr().cast(),
                    self.byte_ptr(at).cast(),
                    dst.len(),
                    cuda::MEMCPY_D2H,
                    stream.raw(),
                )
            },
            "cudaMemcpyAsync D2H",
        )?;
        stream.synchronize()
    }

    /// Device-to-device copy of `bytes` from `src` at `src_at` to this buffer at `at`, on `stream`.
    pub fn copy_from(
        &self,
        stream: &Stream,
        at: usize,
        src: &DeviceBuffer,
        src_at: usize,
        bytes: usize,
    ) -> Result<()> {
        self.check_range(at, bytes, "copy destination")?;
        src.check_range(src_at, bytes, "copy source")?;
        if bytes == 0 {
            return Ok(());
        }
        // SAFETY: both ranges lie inside their allocations.
        check(
            unsafe {
                cuda::cudaMemcpyAsync(
                    self.byte_ptr(at).cast(),
                    src.byte_ptr(src_at).cast(),
                    bytes,
                    cuda::MEMCPY_D2D,
                    stream.raw(),
                )
            },
            "cudaMemcpyAsync D2D",
        )
    }

    /// Zero the whole buffer (synchronous with respect to the device's earlier work).
    pub fn zero(&self) -> Result<()> {
        if self.bytes == 0 {
            return Ok(());
        }
        // SAFETY: the range is this allocation; the null stream orders it after earlier work.
        check(
            unsafe { cuda::cudaMemsetAsync(self.ptr, 0, self.bytes, core::ptr::null_mut()) },
            "cudaMemsetAsync",
        )?;
        synchronize()
    }

    /// Zero bytes `[at, at + n)` on `stream`.
    pub fn zero_async(&self, stream: &Stream, at: usize, n: usize) -> Result<()> {
        self.check_range(at, n, "memset")?;
        if n == 0 {
            return Ok(());
        }
        // SAFETY: the range lies inside this allocation.
        check(
            unsafe { cuda::cudaMemsetAsync(self.byte_ptr(at).cast(), 0, n, stream.raw()) },
            "cudaMemsetAsync",
        )
    }
}

impl Drop for DeviceBuffer {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            // SAFETY: allocated by cudaMalloc, freed once; cudaFree waits for work in flight.
            unsafe { cuda::cudaFree(self.ptr) };
        }
    }
}

/// Page-locked host memory, mapped into the device's address space (`cudaHostAllocMapped`):
/// kernels can read it through [`PinnedBuffer::device_ptr`], and copies from it run at full
/// PCIe speed.
pub struct PinnedBuffer {
    host: *mut c_void,
    device: *mut c_void,
    bytes: usize,
}

// SAFETY: page-locked host memory is usable from any thread; freed once.
unsafe impl Send for PinnedBuffer {}
unsafe impl Sync for PinnedBuffer {}

impl PinnedBuffer {
    /// `bytes` of zeroed page-locked host memory, mapped for the device.
    pub fn alloc(bytes: usize) -> Result<PinnedBuffer> {
        let mut host = core::ptr::null_mut();
        let mut device = core::ptr::null_mut();
        if bytes > 0 {
            // SAFETY: valid out-pointers.
            let code = unsafe {
                cuda::cudaHostAlloc(
                    &mut host,
                    bytes,
                    cuda::HOST_ALLOC_MAPPED | cuda::HOST_ALLOC_PORTABLE,
                )
            };
            if code != cuda::SUCCESS {
                // SAFETY: clear the error.
                unsafe { cuda::cudaGetLastError() };
                return Err(Error::OutOfMemory(format!(
                    "cudaHostAlloc of {bytes} bytes: {}",
                    cuda::error_string(code)
                )));
            }
            // SAFETY: host is a live mapped allocation of `bytes` bytes.
            unsafe { std::ptr::write_bytes(host.cast::<u8>(), 0, bytes) };
            // SAFETY: host was allocated mapped.
            check(
                unsafe { cuda::cudaHostGetDevicePointer(&mut device, host, 0) },
                "cudaHostGetDevicePointer",
            )?;
        }
        Ok(PinnedBuffer {
            host,
            device,
            bytes,
        })
    }

    pub fn bytes(&self) -> usize {
        self.bytes
    }

    /// The device's pointer to this memory.
    pub fn device_ptr<T>(&self) -> *const T {
        self.device as *const T
    }

    pub fn host_ptr<T>(&self) -> *mut T {
        self.host as *mut T
    }

    /// The whole buffer as bytes.
    pub fn as_bytes(&self) -> &[u8] {
        if self.bytes == 0 {
            return &[];
        }
        // SAFETY: a live allocation of `bytes` bytes.
        unsafe { std::slice::from_raw_parts(self.host.cast::<u8>(), self.bytes) }
    }

    /// The whole buffer as mutable bytes. The caller keeps device reads of this memory from
    /// overlapping the writes (a stream synchronization in between).
    #[allow(clippy::mut_from_ref)]
    pub fn as_bytes_mut(&self) -> &mut [u8] {
        if self.bytes == 0 {
            return &mut [];
        }
        // SAFETY: a live allocation of `bytes` bytes; exclusive use is the caller's contract.
        unsafe { std::slice::from_raw_parts_mut(self.host.cast::<u8>(), self.bytes) }
    }
}

impl Drop for PinnedBuffer {
    fn drop(&mut self) {
        if !self.host.is_null() {
            // SAFETY: allocated by cudaHostAlloc, freed once.
            unsafe { cuda::cudaFreeHost(self.host) };
        }
    }
}

/// A non-blocking CUDA stream, destroyed on drop.
pub struct Stream(RawStream);

// SAFETY: a stream handle may be used from any host thread (the scheduler owns it).
unsafe impl Send for Stream {}
unsafe impl Sync for Stream {}

impl Stream {
    pub fn new() -> Result<Stream> {
        let mut s = core::ptr::null_mut();
        // SAFETY: a valid out-pointer.
        check(
            unsafe { cuda::cudaStreamCreateWithFlags(&mut s, cuda::STREAM_NON_BLOCKING) },
            "cudaStreamCreate",
        )?;
        Ok(Stream(s))
    }

    pub fn raw(&self) -> RawStream {
        self.0
    }

    pub fn synchronize(&self) -> Result<()> {
        // SAFETY: a live stream.
        check(
            unsafe { cuda::cudaStreamSynchronize(self.0) },
            "cudaStreamSynchronize",
        )
    }

    /// Make later work on this stream wait for `event`.
    pub fn wait(&self, event: &Event) -> Result<()> {
        // SAFETY: a live stream and event.
        check(
            unsafe { cuda::cudaStreamWaitEvent(self.0, event.0, 0) },
            "cudaStreamWaitEvent",
        )
    }
}

impl Drop for Stream {
    fn drop(&mut self) {
        // SAFETY: created by cudaStreamCreateWithFlags, destroyed once.
        unsafe { cuda::cudaStreamDestroy(self.0) };
    }
}

/// A CUDA event, destroyed on drop.
pub struct Event(RawEvent);

// SAFETY: as for streams.
unsafe impl Send for Event {}
unsafe impl Sync for Event {}

impl Event {
    pub fn new() -> Result<Event> {
        let mut e = core::ptr::null_mut();
        // SAFETY: a valid out-pointer.
        check(unsafe { cuda::cudaEventCreate(&mut e) }, "cudaEventCreate")?;
        Ok(Event(e))
    }

    pub fn record(&self, stream: &Stream) -> Result<()> {
        // SAFETY: a live event and stream.
        check(
            unsafe { cuda::cudaEventRecord(self.0, stream.raw()) },
            "cudaEventRecord",
        )
    }

    pub fn synchronize(&self) -> Result<()> {
        // SAFETY: a live event.
        check(
            unsafe { cuda::cudaEventSynchronize(self.0) },
            "cudaEventSynchronize",
        )
    }

    /// Milliseconds from `start` to this event (waits for this event).
    pub fn elapsed_ms_since(&self, start: &Event) -> Result<f32> {
        self.synchronize()?;
        let mut ms = 0.0f32;
        // SAFETY: live, recorded events.
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
    // SAFETY: a valid out-pointer.
    let rc = unsafe { cuda::cudaGetDeviceCount(&mut n) };
    if rc == cuda::SUCCESS {
        n as usize
    } else {
        // SAFETY: clear the error so later calls are not affected.
        unsafe { cuda::cudaGetLastError() };
        0
    }
}

/// An integer attribute of device 0.
pub fn attribute(attr: i32) -> Result<i32> {
    let mut v = 0;
    // SAFETY: a valid out-pointer.
    check(
        unsafe { cuda::cudaDeviceGetAttribute(&mut v, attr, 0) },
        "cudaDeviceGetAttribute",
    )?;
    Ok(v)
}

/// Multiprocessors of device 0.
pub fn sm_count() -> Result<i32> {
    attribute(cuda::ATTR_SM_COUNT)
}

/// The DRAM bandwidth the attributes describe (2 x memory clock x bus width), bytes per second.
pub fn peak_bandwidth() -> Result<f64> {
    let khz = attribute(cuda::ATTR_MEM_CLOCK_KHZ)? as f64;
    let bits = attribute(cuda::ATTR_MEM_BUS_BITS)? as f64;
    Ok(2.0 * khz * 1e3 * bits / 8.0)
}

/// Free and total device memory, in bytes.
pub fn mem_info() -> Result<(usize, usize)> {
    let (mut free, mut total) = (0, 0);
    // SAFETY: valid out-pointers.
    check(
        unsafe { cuda::cudaMemGetInfo(&mut free, &mut total) },
        "cudaMemGetInfo",
    )?;
    Ok((free, total))
}

/// Wait for all work on the device.
pub fn synchronize() -> Result<()> {
    // SAFETY: no arguments.
    check(
        unsafe { cuda::cudaDeviceSynchronize() },
        "cudaDeviceSynchronize",
    )
}
