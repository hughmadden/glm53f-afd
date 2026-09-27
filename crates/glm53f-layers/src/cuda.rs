//! The CUDA runtime calls the tests and the benchmark need (`cuda` feature): device
//! buffers, streams, events, graphs and device attributes. Declarations resolve against
//! `libcudart`, which `build.rs` links.

use core::ffi::{c_char, c_int, c_void};

use crate::ffi::{CudaError, CudaStream};

pub type CudaEvent = *mut c_void;
pub type CudaGraph = *mut c_void;
pub type CudaGraphExec = *mut c_void;

const H2D: c_int = 1;
const D2H: c_int = 2;

unsafe extern "C" {
    fn cudaMalloc(ptr: *mut *mut c_void, size: usize) -> CudaError;
    fn cudaFree(ptr: *mut c_void) -> CudaError;
    fn cudaMemcpy(dst: *mut c_void, src: *const c_void, count: usize, kind: c_int) -> CudaError;
    fn cudaMemcpyAsync(
        dst: *mut c_void,
        src: *const c_void,
        count: usize,
        kind: c_int,
        stream: CudaStream,
    ) -> CudaError;
    fn cudaMemset(ptr: *mut c_void, value: c_int, count: usize) -> CudaError;
    fn cudaDeviceSynchronize() -> CudaError;
    fn cudaStreamCreate(stream: *mut CudaStream) -> CudaError;
    fn cudaStreamSynchronize(stream: CudaStream) -> CudaError;
    fn cudaStreamDestroy(stream: CudaStream) -> CudaError;
    fn cudaEventCreate(event: *mut CudaEvent) -> CudaError;
    fn cudaEventDestroy(event: CudaEvent) -> CudaError;
    fn cudaEventRecord(event: CudaEvent, stream: CudaStream) -> CudaError;
    fn cudaEventSynchronize(event: CudaEvent) -> CudaError;
    fn cudaEventElapsedTime(ms: *mut f32, start: CudaEvent, end: CudaEvent) -> CudaError;
    fn cudaGetErrorString(err: CudaError) -> *const c_char;
    fn cudaMemGetInfo(free: *mut usize, total: *mut usize) -> CudaError;
    fn cudaDeviceGetAttribute(value: *mut c_int, attr: c_int, device: c_int) -> CudaError;
    fn cudaStreamBeginCapture(stream: CudaStream, mode: c_int) -> CudaError;
    fn cudaStreamEndCapture(stream: CudaStream, graph: *mut CudaGraph) -> CudaError;
    fn cudaGraphInstantiate(exec: *mut CudaGraphExec, graph: CudaGraph, flags: u64) -> CudaError;
    fn cudaGraphLaunch(exec: CudaGraphExec, stream: CudaStream) -> CudaError;
    fn cudaGraphExecDestroy(exec: CudaGraphExec) -> CudaError;
    fn cudaGraphDestroy(graph: CudaGraph) -> CudaError;
}

/// The runtime's text for an error code.
pub fn error_string(err: CudaError) -> String {
    // SAFETY: cudaGetErrorString returns a static NUL-terminated string (or null).
    let p = unsafe { cudaGetErrorString(err) };
    if p.is_null() {
        return format!("cuda error {err}");
    }
    // SAFETY: p is a static C string owned by the runtime.
    unsafe { core::ffi::CStr::from_ptr(p) }
        .to_string_lossy()
        .into_owned()
}

/// `Ok` for `cudaSuccess`, otherwise the error text with `what`.
pub fn check(err: CudaError, what: &str) -> Result<(), String> {
    if err == 0 {
        Ok(())
    } else {
        Err(format!("{what}: {} ({err})", error_string(err)))
    }
}

/// A device allocation, freed on drop.
pub struct DeviceBuffer {
    ptr: *mut c_void,
    bytes: usize,
}

impl DeviceBuffer {
    pub fn new(bytes: usize) -> Result<Self, String> {
        let mut ptr = core::ptr::null_mut();
        // SAFETY: cudaMalloc writes a device pointer into ptr.
        check(unsafe { cudaMalloc(&mut ptr, bytes.max(1)) }, "cudaMalloc")?;
        Ok(DeviceBuffer { ptr, bytes })
    }

    pub fn zeroed(bytes: usize) -> Result<Self, String> {
        let b = Self::new(bytes)?;
        // SAFETY: b.ptr is a live allocation of at least `bytes` bytes.
        check(unsafe { cudaMemset(b.ptr, 0, bytes) }, "cudaMemset")?;
        Ok(b)
    }

    pub fn from_slice<T: Copy>(data: &[T]) -> Result<Self, String> {
        let bytes = std::mem::size_of_val(data);
        let b = Self::new(bytes)?;
        // SAFETY: copies `bytes` bytes from a live host slice into a live allocation.
        check(
            unsafe { cudaMemcpy(b.ptr, data.as_ptr().cast(), bytes, H2D) },
            "cudaMemcpy H2D",
        )?;
        Ok(b)
    }

    pub fn upload<T: Copy>(&self, data: &[T]) -> Result<(), String> {
        let bytes = std::mem::size_of_val(data);
        assert!(bytes <= self.bytes, "upload larger than the buffer");
        // SAFETY: as in from_slice, within this allocation.
        check(
            unsafe { cudaMemcpy(self.ptr, data.as_ptr().cast(), bytes, H2D) },
            "cudaMemcpy H2D",
        )
    }

    pub fn download<T: Copy + Default>(&self, n: usize) -> Result<Vec<T>, String> {
        let bytes = n * std::mem::size_of::<T>();
        assert!(bytes <= self.bytes, "download larger than the buffer");
        let mut v = vec![T::default(); n];
        // SAFETY: copies `bytes` bytes of this allocation into a host vector of that size.
        check(
            unsafe { cudaMemcpy(v.as_mut_ptr().cast(), self.ptr, bytes, D2H) },
            "cudaMemcpy D2H",
        )?;
        Ok(v)
    }

    pub fn bytes(&self) -> usize {
        self.bytes
    }
    pub fn ptr<T>(&self) -> *const T {
        self.ptr as *const T
    }
    pub fn mut_ptr<T>(&self) -> *mut T {
        self.ptr as *mut T
    }
}

impl Drop for DeviceBuffer {
    fn drop(&mut self) {
        // SAFETY: ptr came from cudaMalloc and is freed once.
        unsafe {
            cudaFree(self.ptr);
        }
    }
}

/// A CUDA stream, destroyed on drop.
pub struct Stream(pub CudaStream);

impl Stream {
    pub fn new() -> Result<Self, String> {
        let mut s = core::ptr::null_mut();
        // SAFETY: writes a new stream handle.
        check(unsafe { cudaStreamCreate(&mut s) }, "cudaStreamCreate")?;
        Ok(Stream(s))
    }
    pub fn sync(&self) -> Result<(), String> {
        // SAFETY: a live stream.
        check(
            unsafe { cudaStreamSynchronize(self.0) },
            "cudaStreamSynchronize",
        )
    }

    /// Capture the launches `f` issues on this stream into a graph.
    pub fn capture(&self, f: impl FnOnce(&Stream) -> Result<(), String>) -> Result<Graph, String> {
        // SAFETY: begin/end capture on a live stream; mode 2 = relaxed.
        check(
            unsafe { cudaStreamBeginCapture(self.0, 2) },
            "cudaStreamBeginCapture",
        )?;
        let r = f(self);
        let mut g = core::ptr::null_mut();
        let e = unsafe { cudaStreamEndCapture(self.0, &mut g) };
        r?;
        check(e, "cudaStreamEndCapture")?;
        let mut x = core::ptr::null_mut();
        // SAFETY: instantiates the captured graph.
        let e = unsafe { cudaGraphInstantiate(&mut x, g, 0) };
        if e != 0 {
            unsafe { cudaGraphDestroy(g) };
            return Err(format!("cudaGraphInstantiate: {}", error_string(e)));
        }
        Ok(Graph { graph: g, exec: x })
    }
}

impl Drop for Stream {
    fn drop(&mut self) {
        // SAFETY: a live stream, destroyed once.
        unsafe {
            cudaStreamDestroy(self.0);
        }
    }
}

/// An instantiated CUDA graph.
pub struct Graph {
    graph: CudaGraph,
    exec: CudaGraphExec,
}

impl Graph {
    pub fn launch(&self, s: &Stream) -> Result<(), String> {
        // SAFETY: a live executable graph on a live stream.
        check(
            unsafe { cudaGraphLaunch(self.exec, s.0) },
            "cudaGraphLaunch",
        )
    }
}

impl Drop for Graph {
    fn drop(&mut self) {
        // SAFETY: live handles, destroyed once.
        unsafe {
            cudaGraphExecDestroy(self.exec);
            cudaGraphDestroy(self.graph);
        }
    }
}

/// Milliseconds between two points on a stream.
pub struct Timer {
    start: CudaEvent,
    end: CudaEvent,
}

impl Timer {
    pub fn new() -> Result<Self, String> {
        let (mut a, mut b) = (core::ptr::null_mut(), core::ptr::null_mut());
        // SAFETY: create two events.
        check(unsafe { cudaEventCreate(&mut a) }, "cudaEventCreate")?;
        check(unsafe { cudaEventCreate(&mut b) }, "cudaEventCreate")?;
        Ok(Timer { start: a, end: b })
    }
    pub fn start(&self, s: &Stream) -> Result<(), String> {
        check(
            unsafe { cudaEventRecord(self.start, s.0) },
            "cudaEventRecord",
        )
    }
    pub fn stop_ms(&self, s: &Stream) -> Result<f32, String> {
        let mut ms = 0f32;
        // SAFETY: record, wait for and read two live events.
        unsafe {
            check(cudaEventRecord(self.end, s.0), "cudaEventRecord")?;
            check(cudaEventSynchronize(self.end), "cudaEventSynchronize")?;
            check(
                cudaEventElapsedTime(&mut ms, self.start, self.end),
                "cudaEventElapsedTime",
            )?;
        }
        Ok(ms)
    }
}

impl Drop for Timer {
    fn drop(&mut self) {
        // SAFETY: live events, destroyed once.
        unsafe {
            cudaEventDestroy(self.start);
            cudaEventDestroy(self.end);
        }
    }
}

/// Copy `bytes` from `src` to `dst` on the device, ordered on `s`.
pub fn copy_d2d(
    dst: &DeviceBuffer,
    src: &DeviceBuffer,
    bytes: usize,
    s: &Stream,
) -> Result<(), String> {
    assert!(bytes <= dst.bytes() && bytes <= src.bytes());
    // SAFETY: both ranges lie inside live allocations; kind 3 = device to device.
    check(
        unsafe { cudaMemcpyAsync(dst.mut_ptr(), src.ptr(), bytes, 3, s.0) },
        "cudaMemcpyAsync D2D",
    )
}

pub fn device_sync() -> Result<(), String> {
    // SAFETY: no arguments.
    check(unsafe { cudaDeviceSynchronize() }, "cudaDeviceSynchronize")
}

/// Free and total device memory, in bytes.
pub fn mem_info() -> Result<(usize, usize), String> {
    let (mut f, mut t) = (0usize, 0usize);
    // SAFETY: writes two sizes.
    check(unsafe { cudaMemGetInfo(&mut f, &mut t) }, "cudaMemGetInfo")?;
    Ok((f, t))
}

/// A device attribute of device 0 (`cudaDeviceAttr` value).
pub fn attribute(attr: i32) -> Result<i32, String> {
    let mut v = 0;
    // SAFETY: writes one int.
    check(
        unsafe { cudaDeviceGetAttribute(&mut v, attr, 0) },
        "cudaDeviceGetAttribute",
    )?;
    Ok(v)
}

/// Streaming multiprocessors of device 0.
pub fn sm_count() -> Result<i32, String> {
    attribute(16)
}

/// Theoretical DRAM bandwidth of device 0 in bytes per second (2 x memory clock x bus
/// width, the double-data-rate peak the attributes describe).
pub fn peak_bandwidth() -> Result<f64, String> {
    let khz = attribute(36)? as f64;
    let bits = attribute(37)? as f64;
    Ok(2.0 * khz * 1e3 * bits / 8.0)
}

/// L2 cache size of device 0 in bytes.
pub fn l2_bytes() -> Result<i32, String> {
    attribute(38)
}
