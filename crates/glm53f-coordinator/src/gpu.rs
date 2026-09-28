//! The shell's kernels on the device (feature `cuda`): the sampler (masks, argmax, draws) and the
//! wire kernels, with the few CUDA runtime calls they need. The C ABI is documented in
//! `kernels/glm53f_coord.h`.
//!
//! [`Sampler`] is what a model's forward calls to turn its device logit rows into token ids:
//! `mask_rows` for rows with a mask, `argmax_rows` over every row, `sample_rows` over the sampled
//! ones, one download of the ids. It gives the bits of [`crate::sampling::select_pick`] except
//! where the device `expf` and the host's differ in the last bit and a draw lands within about
//! 1e-7 of a boundary.

use core::ffi::{c_char, c_int, c_void};

use crate::model::Pick;
use crate::sampling::DeviceRow;

/// `cudaError_t`.
pub type CudaError = c_int;
/// `cudaStream_t`.
pub type Stream = *mut c_void;

const SUCCESS: CudaError = 0;
const H2D: c_int = 1;
const D2H: c_int = 2;
/// The legacy default stream.
const STREAM: Stream = core::ptr::null_mut();

unsafe extern "C" {
    fn cudaMalloc(ptr: *mut *mut c_void, size: usize) -> CudaError;
    fn cudaFree(ptr: *mut c_void) -> CudaError;
    fn cudaMemcpy(dst: *mut c_void, src: *const c_void, count: usize, kind: c_int) -> CudaError;
    fn cudaDeviceSynchronize() -> CudaError;
    fn cudaGetDeviceCount(count: *mut c_int) -> CudaError;
    fn cudaGetErrorString(err: CudaError) -> *const c_char;
    fn cudaHostRegister(ptr: *mut c_void, size: usize, flags: u32) -> CudaError;
    fn cudaHostUnregister(ptr: *mut c_void) -> CudaError;

    pub fn glm53f_coord_argmax_rows(x: *const f32, ld: i64, rows: i32, vocab: i32, out: *mut i32, s: Stream) -> CudaError;
    pub fn glm53f_coord_argmax_prob_rows(x: *const f32, ld: i64, rows: i32, vocab: i32, out: *mut i32, prob: *mut f32,
        s: Stream) -> CudaError;
    pub fn glm53f_coord_mask_rows(x: *mut f32, ld: i64, rows: i32, cols: i32, masks: *const *const u32, bits: i32,
        s: Stream) -> CudaError;
    pub fn glm53f_coord_sample_rows(x: *const f32, ld: i64, vocab: i32, rows: *const c_void, n: i32, out: *mut i32,
        s: Stream) -> CudaError;
    pub fn glm53f_coord_quant_scales(x: *const f32, n_blocks: i64, scales: *mut u8, scale_inv: *mut f32, s: Stream)
        -> CudaError;
    pub fn glm53f_coord_quantize_hidden(x: *const f32, scale_inv: *const f32, payload: *mut u8, n_elem: i64, s: Stream)
        -> CudaError;
    pub fn glm53f_coord_rank_sum_bf16(p0: *const u16, p1: *const u16, p2: *const u16, p3: *const u16, out: *mut f32,
        n: i64, scale: f32, s: Stream) -> CudaError;
    pub fn glm53f_coord_frame_fill(idx: *const i32, wts: *const f32, payload: *const u8, scales: *const u8, t: i32,
        topk: i32, hid: i32, routes: *mut u8, hidden: *mut u8, pitch: i32, s: Stream) -> CudaError;
}

/// The runtime's name for an error code.
pub fn error_string(err: CudaError) -> String {
    // SAFETY: cudaGetErrorString returns a static NUL-terminated string (or null).
    let p = unsafe { cudaGetErrorString(err) };
    if p.is_null() {
        return format!("CUDA error {err}");
    }
    // SAFETY: a static NUL-terminated C string owned by the runtime.
    unsafe { core::ffi::CStr::from_ptr(p) }.to_string_lossy().into_owned()
}

/// `Ok` for a success code, else the error named after `what`.
pub fn check(rc: CudaError, what: &str) -> Result<(), String> {
    if rc == SUCCESS {
        Ok(())
    } else {
        Err(format!("{what}: {}", error_string(rc)))
    }
}

/// Devices the runtime sees (0 when there is none or no driver).
pub fn device_count() -> usize {
    let mut n = 0;
    // SAFETY: writes one int.
    if unsafe { cudaGetDeviceCount(&mut n) } == SUCCESS { n.max(0) as usize } else { 0 }
}

/// Wait for all queued device work.
pub fn synchronize() -> Result<(), String> {
    // SAFETY: no arguments.
    check(unsafe { cudaDeviceSynchronize() }, "cudaDeviceSynchronize")
}

/// Page-lock `len` bytes of host memory at `ptr` (the host tier's arenas).
pub fn host_register(ptr: *mut u8, len: usize) -> Result<(), String> {
    // SAFETY: the caller owns a live allocation of `len` bytes at `ptr` and unregisters it
    // before freeing it.
    check(unsafe { cudaHostRegister(ptr as *mut c_void, len, 0) }, "cudaHostRegister")
}

/// Undo [`host_register`].
pub fn host_unregister(ptr: *mut u8) {
    // SAFETY: `ptr` was registered by host_register.
    unsafe { cudaHostUnregister(ptr as *mut c_void) };
}

/// A device allocation, freed on drop.
pub struct DeviceBuffer {
    ptr: *mut c_void,
    bytes: usize,
}

impl DeviceBuffer {
    pub fn alloc(bytes: usize) -> Result<Self, String> {
        let mut ptr = core::ptr::null_mut();
        // SAFETY: cudaMalloc writes a device pointer.
        check(unsafe { cudaMalloc(&mut ptr, bytes.max(1)) }, "cudaMalloc")?;
        Ok(DeviceBuffer { ptr, bytes: bytes.max(1) })
    }

    /// A buffer holding a copy of `data`.
    pub fn from_slice<T: Copy>(data: &[T]) -> Result<Self, String> {
        let b = Self::alloc(std::mem::size_of_val(data))?;
        b.upload(data)?;
        Ok(b)
    }

    pub fn bytes(&self) -> usize {
        self.bytes
    }

    pub fn ptr<T>(&self) -> *mut T {
        self.ptr as *mut T
    }

    /// Copy `data` to the start of the buffer.
    pub fn upload<T: Copy>(&self, data: &[T]) -> Result<(), String> {
        let n = std::mem::size_of_val(data);
        if n > self.bytes {
            return Err(format!("upload of {n} B into {} B", self.bytes));
        }
        // SAFETY: `n` bytes fit in the allocation; `data` is plain old data.
        check(unsafe { cudaMemcpy(self.ptr, data.as_ptr() as *const c_void, n, H2D) }, "upload")
    }

    /// Copy the start of the buffer into `out`.
    pub fn download<T: Copy>(&self, out: &mut [T]) -> Result<(), String> {
        let n = std::mem::size_of_val(out);
        if n > self.bytes {
            return Err(format!("download of {n} B from {} B", self.bytes));
        }
        // SAFETY: as upload.
        check(unsafe { cudaMemcpy(out.as_mut_ptr() as *mut c_void, self.ptr, n, D2H) }, "download")
    }

    /// Grow (never shrink) to at least `bytes`, dropping the contents when it grows.
    pub fn ensure(&mut self, bytes: usize) -> Result<(), String> {
        if bytes > self.bytes {
            *self = Self::alloc(bytes)?;
        }
        Ok(())
    }
}

impl Drop for DeviceBuffer {
    fn drop(&mut self) {
        // SAFETY: allocated by cudaMalloc.
        unsafe { cudaFree(self.ptr) };
    }
}

/// The device sampler's scratch: the rows' ids, the sampled-row table and the mask tables.
pub struct Sampler {
    idx: DeviceBuffer,
    rows: DeviceBuffer,
    mask_ptrs: DeviceBuffer,
    mask_words: DeviceBuffer,
}

impl Sampler {
    pub fn new() -> Result<Self, String> {
        Ok(Sampler {
            idx: DeviceBuffer::alloc(256)?,
            rows: DeviceBuffer::alloc(256)?,
            mask_ptrs: DeviceBuffer::alloc(256)?,
            mask_words: DeviceBuffer::alloc(256)?,
        })
    }

    /// The tokens `picks` select from the first `picks.len()` rows of the device logits `x` (row
    /// stride `ld`), over ids below `vocab`. Rows with a mask are masked in place first (`x` is
    /// changed: copy out any row whose unmasked logits are still needed).
    pub fn select(&mut self, x: *mut f32, ld: usize, vocab: usize, picks: &[Pick]) -> Result<Vec<u32>, String> {
        let rows = picks.len();
        if rows == 0 {
            return Ok(Vec::new());
        }
        if vocab == 0 || vocab > ld {
            return Err(format!("select: vocabulary {vocab} of rows of {ld}"));
        }
        self.idx.ensure(rows * 4)?;
        // 1. Masks: one upload of every mask's words, and a table of per-row pointers.
        if picks.iter().any(|p| p.mask.is_some()) {
            let bits = picks.iter().find_map(|p| p.mask.as_ref().map(|m| m.bits())).expect("a mask");
            if picks.iter().filter_map(|p| p.mask.as_ref()).any(|m| m.bits() != bits) {
                return Err("select: masks of different sizes in one call".into());
            }
            let per = bits.div_ceil(32);
            let masked: Vec<usize> = (0..rows).filter(|&r| picks[r].mask.is_some()).collect();
            let mut words = Vec::with_capacity(masked.len() * per);
            for &r in &masked {
                words.extend_from_slice(picks[r].mask.as_ref().expect("mask").words());
            }
            self.mask_words.ensure(words.len().max(1) * 4)?;
            self.mask_words.upload(&words)?;
            let base = self.mask_words.ptr::<u32>() as usize;
            let mut ptrs = vec![0u64; rows];
            for (k, &r) in masked.iter().enumerate() {
                ptrs[r] = (base + k * per * 4) as u64;
            }
            self.mask_ptrs.ensure(rows * 8)?;
            self.mask_ptrs.upload(&ptrs)?;
            // SAFETY: x holds `rows` rows of `ld` floats; the tables hold `rows` entries.
            check(unsafe {
                glm53f_coord_mask_rows(x, ld as i64, rows as i32, ld as i32, self.mask_ptrs.ptr(), bits as i32, STREAM)
            }, "mask_rows")?;
        }
        // 2. Every row's argmax below the bound.
        // SAFETY: as above; idx holds `rows` ints.
        check(unsafe { glm53f_coord_argmax_rows(x, ld as i64, rows as i32, vocab as i32, self.idx.ptr(), STREAM) },
            "argmax_rows")?;
        // 3. The sampled rows' draws overwrite their argmax.
        let sampled: Vec<DeviceRow> =
            picks.iter().enumerate().filter_map(|(r, p)| p.draw.map(|(s, pos)| s.row(r, pos))).collect();
        if !sampled.is_empty() {
            self.rows.ensure(std::mem::size_of_val(sampled.as_slice()))?;
            self.rows.upload(&sampled)?;
            // SAFETY: the table holds `sampled.len()` rows, each naming a row below `rows`.
            check(unsafe {
                glm53f_coord_sample_rows(x, ld as i64, vocab as i32, self.rows.ptr(), sampled.len() as i32,
                    self.idx.ptr(), STREAM)
            }, "sample_rows")?;
        }
        let mut out = vec![0i32; rows];
        self.idx.download(&mut out)?;
        Ok(out.into_iter().map(|v| v as u32).collect())
    }
}

/// [`Sampler::select`] over host logit rows (uploaded first): the selected ids and the wall time
/// of a warm selection (tables, kernels and the ids' download), in ms. For tests and
/// `examples/sample_check.rs`.
pub fn select_rows_host(logits: &[f32], ld: usize, vocab: usize, picks: &[Pick]) -> Result<(Vec<u32>, f64), String> {
    if logits.len() < picks.len() * ld {
        return Err(format!("select_rows_host: {} logits for {} rows of {ld}", logits.len(), picks.len()));
    }
    let x = DeviceBuffer::from_slice(&logits[..picks.len() * ld])?;
    let mut s = Sampler::new()?;
    // Warm the scratch, then time a second run on fresh logits.
    s.select(x.ptr(), ld, vocab, picks)?;
    x.upload(&logits[..picks.len() * ld])?;
    synchronize()?;
    let t0 = std::time::Instant::now();
    let ids = s.select(x.ptr(), ld, vocab, picks)?;
    Ok((ids, t0.elapsed().as_secs_f64() * 1e3))
}
