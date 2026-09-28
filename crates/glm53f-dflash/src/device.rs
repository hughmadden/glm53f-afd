//! Minimal device memory, a stream and timing events (feature `cuda`).

use core::ffi::c_void;

use crate::cuda::{self, check, RawEvent, RawStream};

/// Element types a buffer is copied as.
pub trait Pod: Copy + Default + 'static {}
impl Pod for f32 {}
impl Pod for u16 {}
impl Pod for i32 {}
impl Pod for i64 {}
impl Pod for u64 {}
impl Pod for u8 {}

/// One device allocation, freed on drop. Kernels write through the raw pointer, so every method
/// takes `&self`.
pub struct DeviceBuffer {
    ptr: *mut c_void,
    bytes: usize,
}

// SAFETY: a device allocation is usable from any host thread; the buffer frees it once.
unsafe impl Send for DeviceBuffer {}
unsafe impl Sync for DeviceBuffer {}

impl DeviceBuffer {
    /// `bytes` of uninitialized device memory.
    pub fn alloc(bytes: usize) -> Result<DeviceBuffer, String> {
        let mut ptr = core::ptr::null_mut();
        if bytes > 0 {
            // SAFETY: ptr is a valid out-pointer.
            check(
                unsafe { cuda::cudaMalloc(&mut ptr, bytes) },
                &format!("cudaMalloc({bytes})"),
            )?;
        }
        Ok(DeviceBuffer { ptr, bytes })
    }

    /// A buffer holding `data` (copied on `stream`, which is then synchronized).
    pub fn from_slice<T: Pod>(data: &[T], stream: &Stream) -> Result<DeviceBuffer, String> {
        let b = DeviceBuffer::alloc(std::mem::size_of_val(data))?;
        b.upload(data, stream)?;
        stream.synchronize()?;
        Ok(b)
    }

    /// The device pointer as `*mut T`, `offset` elements in (bounds are the caller's).
    pub fn ptr<T>(&self, offset: usize) -> *mut T {
        (self.ptr as *mut T).wrapping_add(offset)
    }

    pub fn bytes(&self) -> usize {
        self.bytes
    }

    /// Copy `data` to element `offset` on `stream`. A pageable source is staged before the call
    /// returns, so `data` may be dropped afterwards.
    pub fn upload_at<T: Pod>(
        &self,
        offset: usize,
        data: &[T],
        stream: &Stream,
    ) -> Result<(), String> {
        let n = std::mem::size_of_val(data);
        let at = offset * std::mem::size_of::<T>();
        if at + n > self.bytes {
            return Err(format!(
                "upload of {n} bytes at {at} into {} bytes",
                self.bytes
            ));
        }
        if n == 0 {
            return Ok(());
        }
        // SAFETY: the destination range is inside this allocation; data holds n bytes.
        check(
            unsafe {
                cuda::cudaMemcpyAsync(
                    self.ptr.cast::<u8>().add(at).cast(),
                    data.as_ptr().cast(),
                    n,
                    cuda::MEMCPY_H2D,
                    stream.raw(),
                )
            },
            "cudaMemcpyAsync H2D",
        )
    }

    pub fn upload<T: Pod>(&self, data: &[T], stream: &Stream) -> Result<(), String> {
        self.upload_at(0, data, stream)
    }

    /// Copy `n` elements from element `offset` to the host (synchronizes `stream`).
    pub fn download_at<T: Pod>(
        &self,
        offset: usize,
        n: usize,
        stream: &Stream,
    ) -> Result<Vec<T>, String> {
        let bytes = n * std::mem::size_of::<T>();
        let at = offset * std::mem::size_of::<T>();
        if at + bytes > self.bytes {
            return Err(format!(
                "download of {bytes} bytes at {at} from {} bytes",
                self.bytes
            ));
        }
        let mut v = vec![T::default(); n];
        if bytes > 0 {
            // SAFETY: the source range is inside this allocation; v holds n elements.
            check(
                unsafe {
                    cuda::cudaMemcpyAsync(
                        v.as_mut_ptr().cast(),
                        self.ptr.cast::<u8>().add(at).cast(),
                        bytes,
                        cuda::MEMCPY_D2H,
                        stream.raw(),
                    )
                },
                "cudaMemcpyAsync D2H",
            )?;
        }
        stream.synchronize()?;
        Ok(v)
    }

    pub fn download<T: Pod>(&self, n: usize, stream: &Stream) -> Result<Vec<T>, String> {
        self.download_at(0, n, stream)
    }

    /// Zero the whole buffer on `stream`.
    pub fn zero(&self, stream: &Stream) -> Result<(), String> {
        if self.bytes == 0 {
            return Ok(());
        }
        // SAFETY: the range is this allocation.
        check(
            unsafe { cuda::cudaMemsetAsync(self.ptr, 0, self.bytes, stream.raw()) },
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

/// A device buffer that grows to the largest size asked of it (scratch).
#[derive(Default)]
pub struct Scratch {
    buf: Option<DeviceBuffer>,
}

impl Scratch {
    /// At least `bytes`, reallocating (contents lost) when smaller.
    pub fn get(&mut self, bytes: usize) -> Result<&DeviceBuffer, String> {
        if self.buf.as_ref().is_none_or(|b| b.bytes() < bytes) {
            self.buf = None;
            self.buf = Some(DeviceBuffer::alloc(bytes.max(256))?);
        }
        Ok(self.buf.as_ref().expect("allocated"))
    }

    /// Device bytes held now.
    pub fn bytes(&self) -> usize {
        self.buf.as_ref().map_or(0, |b| b.bytes())
    }
}

/// A CUDA stream, destroyed on drop (unless borrowed).
pub struct Stream {
    raw: RawStream,
    owned: bool,
}

// SAFETY: a stream handle is usable from any host thread.
unsafe impl Send for Stream {}

impl Stream {
    pub fn new() -> Result<Stream, String> {
        let mut s = core::ptr::null_mut();
        // SAFETY: s is a valid out-pointer.
        check(
            unsafe { cuda::cudaStreamCreate(&mut s) },
            "cudaStreamCreate",
        )?;
        Ok(Stream {
            raw: s,
            owned: true,
        })
    }

    /// A stream another component created and destroys (the target forward's): work queued on
    /// it is ordered with that component's without events or host waits.
    ///
    /// # Safety
    ///
    /// `raw` is a live stream that outlives every use of the returned value.
    pub unsafe fn borrowed(raw: RawStream) -> Stream {
        Stream { raw, owned: false }
    }

    pub fn raw(&self) -> RawStream {
        self.raw
    }

    pub fn synchronize(&self) -> Result<(), String> {
        // SAFETY: a live stream.
        check(
            unsafe { cuda::cudaStreamSynchronize(self.raw) },
            "cudaStreamSynchronize",
        )
    }
}

impl Drop for Stream {
    fn drop(&mut self) {
        if self.owned {
            // SAFETY: created by cudaStreamCreate, destroyed once.
            unsafe { cuda::cudaStreamDestroy(self.raw) };
        }
    }
}

/// A CUDA event for timing, destroyed on drop.
pub struct Event(RawEvent);

impl Event {
    pub fn new() -> Result<Event, String> {
        let mut e = core::ptr::null_mut();
        // SAFETY: e is a valid out-pointer.
        check(unsafe { cuda::cudaEventCreate(&mut e) }, "cudaEventCreate")?;
        Ok(Event(e))
    }

    pub fn record(&self, stream: &Stream) -> Result<(), String> {
        // SAFETY: a live event and stream.
        check(
            unsafe { cuda::cudaEventRecord(self.0, stream.raw()) },
            "cudaEventRecord",
        )
    }

    /// Milliseconds from `start` to this event (waits for this event).
    pub fn elapsed_ms_since(&self, start: &Event) -> Result<f32, String> {
        // SAFETY: live events.
        check(
            unsafe { cuda::cudaEventSynchronize(self.0) },
            "cudaEventSynchronize",
        )?;
        let mut ms = 0f32;
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

/// CUDA devices visible (0 when the driver or a device is missing).
pub fn device_count() -> usize {
    let mut n = 0;
    // SAFETY: n is a valid out-pointer.
    let rc = unsafe { cuda::cudaGetDeviceCount(&mut n) };
    if rc == cuda::SUCCESS {
        n as usize
    } else {
        // SAFETY: clears the sticky error so later calls are not affected.
        unsafe { cuda::cudaGetLastError() };
        0
    }
}

/// Free and total device memory, in bytes.
pub fn mem_info() -> Result<(usize, usize), String> {
    let (mut free, mut total) = (0, 0);
    // SAFETY: valid out-pointers.
    check(
        unsafe { cuda::cudaMemGetInfo(&mut free, &mut total) },
        "cudaMemGetInfo",
    )?;
    Ok((free, total))
}
