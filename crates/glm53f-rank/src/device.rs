//! A minimal safe device-memory buffer: `cudaMalloc`/`cudaFree` with RAII.
//!
//! Single-owner, single-threaded (the daemon's serve loop is one thread for
//! now), so the buffer is deliberately not `Send`/`Sync`. A buffer owns exactly
//! one `cudaMalloc` allocation and frees it once on drop.

use core::ffi::c_void;

use crate::cuda;
use crate::cuda::CudaError;

/// One device allocation, freed on drop.
pub struct DeviceBuffer {
    ptr: *mut c_void,
    bytes: usize,
}

impl DeviceBuffer {
    /// Allocate `bytes` on the current device.
    pub fn alloc(bytes: usize) -> Result<Self, CudaError> {
        if bytes == 0 {
            return Ok(Self { ptr: core::ptr::null_mut(), bytes: 0 });
        }
        let mut ptr: *mut c_void = core::ptr::null_mut();
        // SAFETY: &mut ptr is a valid out-pointer; cudaMalloc writes the device
        // pointer into it.
        let rc = unsafe { cuda::cudaMalloc(&mut ptr, bytes) };
        if rc != cuda::SUCCESS {
            return Err(rc);
        }
        Ok(Self { ptr, bytes })
    }

    /// The device pointer (for passing to the kernels).
    pub fn as_ptr(&self) -> *mut c_void {
        self.ptr
    }

    /// Byte length.
    pub fn bytes(&self) -> usize {
        self.bytes
    }

    /// Copy `src` (exactly `bytes()` long) to the device.
    pub fn upload(&self, src: &[u8]) -> Result<(), CudaError> {
        assert_eq!(src.len(), self.bytes, "upload size mismatch");
        if self.bytes == 0 {
            return Ok(());
        }
        // SAFETY: self.ptr is a valid cudaMalloc allocation of self.bytes; src
        // is a host slice of the same length; H2D copy is within bounds.
        let rc = unsafe {
            cuda::cudaMemcpy(self.ptr, src.as_ptr() as *const c_void, src.len(), cuda::H2D)
        };
        if rc != cuda::SUCCESS {
            return Err(rc);
        }
        Ok(())
    }

    /// Copy `bytes()` bytes from the device into `dst`.
    pub fn download(&self, dst: &mut [u8]) -> Result<(), CudaError> {
        assert_eq!(dst.len(), self.bytes, "download size mismatch");
        if self.bytes == 0 {
            return Ok(());
        }
        // SAFETY: self.ptr is a valid cudaMalloc allocation of self.bytes; dst
        // is a host slice of the same length; D2H copy is within bounds.
        let rc = unsafe {
            cuda::cudaMemcpy(dst.as_mut_ptr() as *mut c_void, self.ptr, dst.len(), cuda::D2H)
        };
        if rc != cuda::SUCCESS {
            return Err(rc);
        }
        Ok(())
    }

    /// Copy the first `src.len()` bytes of `src` into this (larger or equal)
    /// buffer. For a pooled max-size buffer that is reused across requests.
    pub fn upload_prefix(&self, src: &[u8]) -> Result<(), CudaError> {
        assert!(src.len() <= self.bytes, "upload exceeds buffer capacity");
        if src.is_empty() {
            return Ok(());
        }
        // SAFETY: self.ptr is a valid cudaMalloc allocation of self.bytes >=
        // src.len(); the H2D copy of src.len() bytes stays in bounds.
        let rc = unsafe {
            cuda::cudaMemcpy(self.ptr, src.as_ptr() as *const c_void, src.len(), cuda::H2D)
        };
        if rc != cuda::SUCCESS {
            return Err(rc);
        }
        Ok(())
    }

    /// Copy the first `src.len()` bytes of `src` into this (larger or equal)
    /// buffer, asynchronously on `stream` (the source must be page-locked). The
    /// caller must ensure the stream is synchronized before the device reads the
    /// buffer; used for the small per-request plan/gather arrays whose sync copy
    /// dispatch dominates the decode FFN.
    ///
    /// # Safety
    /// `stream` must be a live CUDA stream (or null for the default stream),
    /// and `src` must stay valid and unchanged until the copy completes.
    pub unsafe fn upload_prefix_async(&self, src: &[u8], stream: *mut c_void) -> Result<(), CudaError> {
        assert!(src.len() <= self.bytes, "upload exceeds buffer capacity");
        if src.is_empty() {
            return Ok(());
        }
        // SAFETY: self.ptr is a valid cudaMalloc allocation >= src.len(); src is
        // page-locked; the async H2D copy of src.len() bytes stays in bounds.
        let rc = unsafe {
            cuda::cudaMemcpyAsync(self.ptr, src.as_ptr() as *const c_void, src.len(), cuda::H2D, stream)
        };
        if rc != cuda::SUCCESS {
            return Err(rc);
        }
        Ok(())
    }

    /// Copy the first `dst.len()` bytes of this buffer into `dst`. For a pooled
    /// max-size buffer whose live region is shorter than the allocation.
    pub fn download_prefix(&self, dst: &mut [u8]) -> Result<(), CudaError> {
        assert!(dst.len() <= self.bytes, "download exceeds buffer capacity");
        if dst.is_empty() {
            return Ok(());
        }
        // SAFETY: self.ptr is a valid cudaMalloc allocation of self.bytes >=
        // dst.len(); the D2H copy of dst.len() bytes stays in bounds.
        let rc = unsafe {
            cuda::cudaMemcpy(dst.as_mut_ptr() as *mut c_void, self.ptr, dst.len(), cuda::D2H)
        };
        if rc != cuda::SUCCESS {
            return Err(rc);
        }
        Ok(())
    }

    /// Zero the buffer on the device.
    pub fn zero(&self) -> Result<(), CudaError> {
        if self.bytes == 0 {
            return Ok(());
        }
        // SAFETY: self.ptr is a valid cudaMalloc allocation of self.bytes.
        let rc = unsafe { cuda::cudaMemset(self.ptr, 0, self.bytes) };
        if rc != cuda::SUCCESS {
            return Err(rc);
        }
        Ok(())
    }
}

impl Drop for DeviceBuffer {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            // SAFETY: ptr was allocated by cudaMalloc and is owned exclusively
            // by this buffer; freeing exactly once at drop.
            unsafe { cuda::cudaFree(self.ptr) };
        }
    }
}

/// A persistent pinned host buffer: `cudaHostRegister` once at allocation, so
/// per-request H2D uploads read from page-locked memory without paying a
/// register/unregister per request. The daemon's hidden-row staging buffer.
pub struct PinnedBuffer {
    bytes: Vec<u8>,
}

impl PinnedBuffer {
    /// Allocate and pin `capacity` bytes (the kernel may clamp the pin).
    pub fn new(capacity: usize) -> Result<Self, CudaError> {
        let mut bytes = vec![0u8; capacity];
        // SAFETY: bytes.as_mut_ptr is a valid host allocation of bytes.len(); the
        // pin lives until `drop` unregisters it.
        let rc = unsafe {
            cuda::cudaHostRegister(bytes.as_mut_ptr() as *mut c_void, bytes.len(), 0)
        };
        if rc != cuda::SUCCESS {
            return Err(rc);
        }
        Ok(Self { bytes })
    }

    /// Byte length of the pinned buffer.
    pub fn len(&self) -> usize {
        self.bytes.len()
    }

    /// The pinned host bytes.
    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        &mut self.bytes
    }
}

impl Drop for PinnedBuffer {
    fn drop(&mut self) {
        // SAFETY: bytes was registered by `new` and is owned exclusively here.
        unsafe { cuda::cudaHostUnregister(self.bytes.as_mut_ptr() as *mut c_void) };
    }
}
