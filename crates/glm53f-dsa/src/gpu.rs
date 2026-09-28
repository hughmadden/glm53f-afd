//! Device buffers, error handling and timing for the kernel tests and
//! benchmarks (feature `cuda`).

use core::ffi::c_void;
use std::ptr;

use crate::ffi::{self, CudaError};

/// Turn a CUDA status into a `Result` with the runtime's message.
pub fn check(rc: CudaError, what: &str) -> Result<(), String> {
    if rc == 0 {
        return Ok(());
    }
    // SAFETY: cudaGetErrorString returns a static NUL-terminated string.
    let msg = unsafe {
        let p = ffi::cudaGetErrorString(rc);
        if p.is_null() {
            format!("cuda error {rc}")
        } else {
            core::ffi::CStr::from_ptr(p).to_string_lossy().into_owned()
        }
    };
    Err(format!("{what}: {msg} ({rc})"))
}

/// One device allocation, freed on drop.
pub struct DeviceBuffer {
    ptr: *mut c_void,
    bytes: usize,
}

impl DeviceBuffer {
    pub fn alloc(bytes: usize) -> Result<Self, String> {
        let mut p: *mut c_void = ptr::null_mut();
        if bytes > 0 {
            // SAFETY: valid out-pointer.
            check(unsafe { ffi::cudaMalloc(&mut p, bytes) }, "cudaMalloc")?;
        }
        Ok(Self { ptr: p, bytes })
    }

    /// Allocate and zero.
    pub fn zeroed(bytes: usize) -> Result<Self, String> {
        let b = Self::alloc(bytes)?;
        b.zero()?;
        Ok(b)
    }

    /// Allocate and upload a slice of plain values.
    pub fn from_slice<T: Copy>(v: &[T]) -> Result<Self, String> {
        let b = Self::alloc(std::mem::size_of_val(v))?;
        b.upload(v)?;
        Ok(b)
    }

    pub fn ptr(&self) -> *mut c_void {
        self.ptr
    }

    pub fn as_ptr<T>(&self) -> *const T {
        self.ptr as *const T
    }

    pub fn as_mut_ptr<T>(&self) -> *mut T {
        self.ptr as *mut T
    }

    pub fn bytes(&self) -> usize {
        self.bytes
    }

    pub fn upload<T: Copy>(&self, v: &[T]) -> Result<(), String> {
        let n = std::mem::size_of_val(v);
        assert!(n <= self.bytes, "upload of {n} bytes into {}", self.bytes);
        if n == 0 {
            return Ok(());
        }
        // SAFETY: the device range holds at least n bytes; the host slice is n bytes.
        check(unsafe { ffi::cudaMemcpy(self.ptr, v.as_ptr() as *const c_void, n, ffi::MEMCPY_H2D) }, "upload")
    }

    pub fn download<T: Copy + Default>(&self, n: usize) -> Result<Vec<T>, String> {
        let bytes = n * std::mem::size_of::<T>();
        assert!(bytes <= self.bytes, "download of {bytes} bytes from {}", self.bytes);
        let mut v = vec![T::default(); n];
        if bytes > 0 {
            // SAFETY: both ranges are `bytes` long.
            check(unsafe { ffi::cudaMemcpy(v.as_mut_ptr() as *mut c_void, self.ptr, bytes, ffi::MEMCPY_D2H) }, "download")?;
        }
        Ok(v)
    }

    pub fn zero(&self) -> Result<(), String> {
        if self.bytes == 0 {
            return Ok(());
        }
        // SAFETY: the allocation is `bytes` long.
        check(unsafe { ffi::cudaMemset(self.ptr, 0, self.bytes) }, "memset")
    }
}

impl Drop for DeviceBuffer {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            // SAFETY: allocated by cudaMalloc, freed once.
            unsafe { ffi::cudaFree(self.ptr) };
        }
    }
}

/// Wait for the device and surface any asynchronous error.
pub fn sync() -> Result<(), String> {
    // SAFETY: plain runtime calls.
    check(unsafe { ffi::cudaDeviceSynchronize() }, "synchronize")?;
    check(unsafe { ffi::cudaGetLastError() }, "last error")
}

/// Free and total device memory in bytes.
pub fn mem_info() -> Result<(usize, usize), String> {
    let (mut f, mut t) = (0usize, 0usize);
    // SAFETY: valid out-pointers.
    check(unsafe { ffi::cudaMemGetInfo(&mut f, &mut t) }, "cudaMemGetInfo")?;
    Ok((f, t))
}

/// Multiprocessor count and compute capability of device 0.
pub fn device_info() -> Result<(i32, i32, i32), String> {
    let (mut sms, mut maj, mut min) = (0, 0, 0);
    // SAFETY: valid out-pointers.
    unsafe {
        check(ffi::cudaDeviceGetAttribute(&mut sms, ffi::ATTR_SM_COUNT, 0), "attr")?;
        check(ffi::cudaDeviceGetAttribute(&mut maj, ffi::ATTR_CC_MAJOR, 0), "attr")?;
        check(ffi::cudaDeviceGetAttribute(&mut min, ffi::ATTR_CC_MINOR, 0), "attr")?;
    }
    Ok((sms, maj, min))
}

/// One-time kernel setup (shared-memory limits).
pub fn init() -> Result<(), String> {
    // SAFETY: plain runtime call.
    check(unsafe { ffi::glm53f_dsa_init() }, "glm53f_dsa_init")
}

/// A CUDA stream, destroyed on drop. Non-blocking, so work on the legacy default
/// stream (from other threads, for example) never depends on it, which also
/// keeps a graph capture on it valid.
pub struct Stream(pub ffi::CudaStream);

impl Stream {
    pub fn new() -> Result<Self, String> {
        let mut s: ffi::CudaStream = ptr::null_mut();
        // SAFETY: valid out-pointer.
        check(unsafe { ffi::cudaStreamCreateWithFlags(&mut s, ffi::STREAM_NON_BLOCKING) }, "cudaStreamCreate")?;
        Ok(Self(s))
    }

    pub fn sync(&self) -> Result<(), String> {
        // SAFETY: a live stream.
        check(unsafe { ffi::cudaStreamSynchronize(self.0) }, "cudaStreamSynchronize")
    }
}

impl Drop for Stream {
    fn drop(&mut self) {
        // SAFETY: created by cudaStreamCreate, destroyed once.
        unsafe { ffi::cudaStreamDestroy(self.0) };
    }
}

/// Mean time per call of `f(stream)` in microseconds, launched eagerly on
/// `stream`: `warmup` untimed calls, then `iters` timed calls between two events.
pub fn time_stream_us(
    stream: &Stream,
    warmup: usize,
    iters: usize,
    mut f: impl FnMut(ffi::CudaStream) -> Result<(), String>,
) -> Result<f64, String> {
    for _ in 0..warmup {
        f(stream.0)?;
    }
    stream.sync()?;
    let (mut a, mut b): (*mut c_void, *mut c_void) = (ptr::null_mut(), ptr::null_mut());
    // SAFETY: events are created, used and destroyed here.
    unsafe {
        check(ffi::cudaEventCreate(&mut a), "event")?;
        check(ffi::cudaEventCreate(&mut b), "event")?;
        check(ffi::cudaEventRecord(a, stream.0), "record")?;
    }
    for _ in 0..iters {
        f(stream.0)?;
    }
    let mut ms = 0.0f32;
    // SAFETY: as above.
    unsafe {
        check(ffi::cudaEventRecord(b, stream.0), "record")?;
        check(ffi::cudaEventSynchronize(b), "event sync")?;
        check(ffi::cudaEventElapsedTime(&mut ms, a, b), "elapsed")?;
        ffi::cudaEventDestroy(a);
        ffi::cudaEventDestroy(b);
    }
    Ok(ms as f64 * 1000.0 / iters as f64)
}

/// Mean time per call of `f(stream)` in microseconds when replayed from a CUDA
/// graph: `reps` calls are captured into one graph, which is launched once to
/// warm up and then `launches` times between two events. This is how a serving
/// loop runs steady shapes: it removes host launch overhead but keeps the
/// device-side gaps between dependent kernels. The capture is thread-local: other
/// threads may use the runtime meanwhile.
pub fn time_graph_us(
    stream: &Stream,
    reps: usize,
    launches: usize,
    mut f: impl FnMut(ffi::CudaStream) -> Result<(), String>,
) -> Result<f64, String> {
    let mut graph: *mut c_void = ptr::null_mut();
    let mut exec: *mut c_void = ptr::null_mut();
    // SAFETY: capture, instantiate and launch on a live stream; handles freed below.
    unsafe {
        check(ffi::cudaStreamBeginCapture(stream.0, ffi::CAPTURE_THREAD_LOCAL), "begin capture")?;
        for _ in 0..reps {
            if let Err(e) = f(stream.0) {
                let _ = ffi::cudaStreamEndCapture(stream.0, &mut graph);
                return Err(e);
            }
        }
        check(ffi::cudaStreamEndCapture(stream.0, &mut graph), "end capture")?;
        check(ffi::cudaGraphInstantiate(&mut exec, graph, 0), "instantiate")?;
        check(ffi::cudaGraphLaunch(exec, stream.0), "graph launch")?;
    }
    stream.sync()?;
    let (mut a, mut b): (*mut c_void, *mut c_void) = (ptr::null_mut(), ptr::null_mut());
    let mut ms = 0.0f32;
    // SAFETY: as above.
    unsafe {
        check(ffi::cudaEventCreate(&mut a), "event")?;
        check(ffi::cudaEventCreate(&mut b), "event")?;
        check(ffi::cudaEventRecord(a, stream.0), "record")?;
        for _ in 0..launches {
            check(ffi::cudaGraphLaunch(exec, stream.0), "graph launch")?;
        }
        check(ffi::cudaEventRecord(b, stream.0), "record")?;
        check(ffi::cudaEventSynchronize(b), "event sync")?;
        check(ffi::cudaEventElapsedTime(&mut ms, a, b), "elapsed")?;
        ffi::cudaEventDestroy(a);
        ffi::cudaEventDestroy(b);
        ffi::cudaGraphExecDestroy(exec);
        ffi::cudaGraphDestroy(graph);
    }
    Ok(ms as f64 * 1000.0 / (reps * launches) as f64)
}

/// Mean time in microseconds of one call of `f(stream)` with a cold L2 cache:
/// before each of `iters` calls, `flush(stream)` evicts it (it should read more
/// data than the L2 holds; reads leave clean lines, so the timed call pays no
/// write-backs), and only the call itself is timed (events on both sides).
pub fn time_cold_us(
    stream: &Stream,
    iters: usize,
    mut flush: impl FnMut(ffi::CudaStream) -> Result<(), String>,
    mut f: impl FnMut(ffi::CudaStream) -> Result<(), String>,
) -> Result<f64, String> {
    f(stream.0)?; // warm-up (instruction cache, lazy setup)
    stream.sync()?;
    let mut ev: Vec<*mut c_void> = vec![ptr::null_mut(); 2 * iters];
    // SAFETY: events are created, recorded on a live stream and destroyed here.
    unsafe {
        for e in ev.iter_mut() {
            check(ffi::cudaEventCreate(e), "event")?;
        }
        for i in 0..iters {
            flush(stream.0)?;
            check(ffi::cudaEventRecord(ev[2 * i], stream.0), "record")?;
            f(stream.0)?;
            check(ffi::cudaEventRecord(ev[2 * i + 1], stream.0), "record")?;
        }
    }
    stream.sync()?;
    let mut total = 0.0f64;
    // SAFETY: as above.
    unsafe {
        for i in 0..iters {
            let mut ms = 0.0f32;
            check(ffi::cudaEventElapsedTime(&mut ms, ev[2 * i], ev[2 * i + 1]), "elapsed")?;
            total += ms as f64;
        }
        for e in ev {
            ffi::cudaEventDestroy(e);
        }
    }
    Ok(total * 1000.0 / iters as f64)
}

/// Time `f` (which launches work on the default stream) with CUDA events:
/// `warmup` untimed runs, then the mean over `iters` runs, in microseconds.
pub fn time_us(warmup: usize, iters: usize, mut f: impl FnMut() -> Result<(), String>) -> Result<f64, String> {
    for _ in 0..warmup {
        f()?;
    }
    sync()?;
    let (mut a, mut b): (*mut c_void, *mut c_void) = (ptr::null_mut(), ptr::null_mut());
    // SAFETY: event handles are created, used and destroyed here.
    unsafe {
        check(ffi::cudaEventCreate(&mut a), "event")?;
        check(ffi::cudaEventCreate(&mut b), "event")?;
        check(ffi::cudaEventRecord(a, ptr::null_mut()), "record")?;
    }
    for _ in 0..iters {
        f()?;
    }
    let mut ms = 0.0f32;
    // SAFETY: as above.
    unsafe {
        check(ffi::cudaEventRecord(b, ptr::null_mut()), "record")?;
        check(ffi::cudaEventSynchronize(b), "event sync")?;
        check(ffi::cudaEventElapsedTime(&mut ms, a, b), "elapsed")?;
        ffi::cudaEventDestroy(a);
        ffi::cudaEventDestroy(b);
    }
    Ok(ms as f64 * 1000.0 / iters as f64)
}
