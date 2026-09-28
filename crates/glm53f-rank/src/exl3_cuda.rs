//! The CUDA backend: the Rust side of `kernels/exl3_rank.cu` (after
//! mimo26f-afd's `crates/mimo26-spark/src/b1.rs`).

use core::ffi::{c_char, c_int};

use crate::consts::{HIDDEN, TOPK};
use crate::kernel::{ExpertKernel, FfnStats, Rows};
use crate::layout::LAYER_BYTES;

#[repr(C)]
struct RawLayer {
    _p: [u8; 0],
}

#[repr(C)]
struct RawScratch {
    _p: [u8; 0],
}

/// A kernel configuration: 16-row tiles per expert group (`mt`, 1 or 2), gate/up
/// K splits (`sk`) and down K splits (`skd`), both dividing 32; zero fields
/// take the defaults for the row count. `fp32_swiglu` = 1 computes the SwiGLU
/// in FP32 instead of with the reference model's BF16 roundings (the default).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Cfg {
    pub mt: c_int,
    pub sk: c_int,
    pub skd: c_int,
    pub fp32_swiglu: c_int,
}

unsafe extern "C" {
    fn g53r_baked_arch() -> c_int;
    fn g53r_device_identity(arch: *mut c_int, sms: *mut c_int, name: *mut c_char, namelen: usize) -> c_int;
    fn g53r_default_cfg(rows: u32, out: *mut Cfg);
    fn g53r_layer_new(host: *const u8, bytes: u64, out: *mut *mut RawLayer, err: *mut c_char, errlen: usize) -> c_int;
    fn g53r_layer_free(l: *mut RawLayer);
    fn g53r_scratch_new(out: *mut *mut RawScratch, err: *mut c_char, errlen: usize) -> c_int;
    fn g53r_scratch_free(s: *mut RawScratch);
    fn g53r_ffn(
        l: *const RawLayer,
        s: *mut RawScratch,
        payload: *const u8,
        payload_pitch: usize,
        scales: *const u8,
        scales_pitch: usize,
        ids: *const i32,
        weights: *const f32,
        rows: u32,
        bf16_out: *mut u16,
        cfg: *const Cfg,
        ms: *mut f32,
        err: *mut c_char,
        errlen: usize,
    ) -> c_int;
}

unsafe extern "C" {
    fn g53r_debug_copy(
        s: *const RawScratch,
        z: *mut f32,
        z_count: usize,
        pair_route: *mut i32,
        pr_count: usize,
        xd: *mut u16,
        xd_count: usize,
        zd: *mut f32,
        zd_count: usize,
        err: *mut c_char,
        errlen: usize,
    ) -> c_int;
}

/// The intermediates of the last call (a test hook): gate/up split partials
/// `z` [2][sk][routes][512], each grouped pair's route (`row * 8 + slot`),
/// the down input `xd` [routes][512] (FP16 bits), the down split partials
/// `zd` [skd][routes][4096].
pub struct Intermediates {
    pub cfg: Cfg,
    pub routes: usize,
    pub z: Vec<f32>,
    pub pair_route: Vec<i32>,
    pub xd: Vec<u16>,
    pub zd: Vec<f32>,
}

fn err_string(buf: &[u8]) -> String {
    let n = buf.iter().position(|&b| b == 0).unwrap_or(buf.len());
    String::from_utf8_lossy(&buf[..n]).into_owned()
}

/// The live device: compute capability (major * 10 + minor), SM count, name.
pub fn device_identity() -> Result<(i32, i32, String), String> {
    let (mut arch, mut sms) = (0, 0);
    let mut name = [0u8; 256];
    // SAFETY: out-pointers valid for writes; name holds 256 bytes.
    let rc = unsafe { g53r_device_identity(&mut arch, &mut sms, name.as_mut_ptr() as *mut c_char, name.len()) };
    if rc != 0 {
        return Err("no CUDA device".into());
    }
    Ok((arch, sms, err_string(&name)))
}

/// The architecture the kernels were compiled for (`GLM53F_CUDA_ARCH`).
pub fn baked_arch() -> i32 {
    // SAFETY: a pure function of the compiled object.
    unsafe { g53r_baked_arch() }
}

/// The start-up gate: the kernels must have been compiled for the live
/// device's architecture. Returns a one-line identity for the boot log.
pub fn check_device() -> Result<String, String> {
    let (arch, sms, name) = device_identity()?;
    let baked = baked_arch();
    if arch != baked {
        return Err(format!(
            "device {name} is sm_{arch} ({sms} SMs) but the kernels were built for sm_{baked} (GLM53F_CUDA_ARCH={}): rebuild on this host",
            env!("GLM53F_RANK_CUDA_ARCH")
        ));
    }
    Ok(format!("{name}, sm_{arch}, {sms} SMs, kernels built for {}", env!("GLM53F_RANK_CUDA_ARCH")))
}

/// The default configuration for `rows` rows.
pub fn default_cfg(rows: usize) -> Cfg {
    let mut c = Cfg::default();
    // SAFETY: c is a valid out-pointer.
    unsafe { g53r_default_cfg(rows as u32, &mut c) };
    c
}

/// One layer's image in device memory.
pub struct CudaLayer(*mut RawLayer);

// SAFETY: used from the serving thread only; the device memory has no thread affinity.
unsafe impl Send for CudaLayer {}

impl Drop for CudaLayer {
    fn drop(&mut self) {
        // SAFETY: created by g53r_layer_new, freed once.
        unsafe { g53r_layer_free(self.0) }
    }
}

/// The CUDA backend: a stream with its events and grow-only device scratch.
pub struct CudaKernel {
    scratch: *mut RawScratch,
    /// A fixed configuration for every call (tests and benchmarks), or the
    /// defaults by row count.
    pub cfg: Option<Cfg>,
}

// SAFETY: as for CudaLayer.
unsafe impl Send for CudaKernel {}

impl CudaKernel {
    pub fn new() -> Result<Self, String> {
        let mut out = core::ptr::null_mut();
        let mut err = [0u8; 256];
        // SAFETY: out/err valid for writes.
        let rc = unsafe { g53r_scratch_new(&mut out, err.as_mut_ptr() as *mut c_char, err.len()) };
        if rc != 0 {
            return Err(err_string(&err));
        }
        Ok(Self { scratch: out, cfg: None })
    }
}

impl CudaKernel {
    /// Copy the intermediates of the last [`ExpertKernel::ffn`] call of `rows`
    /// rows (a test hook; the configuration must be the one that call used).
    pub fn intermediates(&self, rows: usize) -> Result<Intermediates, String> {
        let mut cfg = default_cfg(rows);
        if let Some(c) = self.cfg {
            if c.mt != 0 {
                cfg.mt = c.mt;
            }
            if c.sk != 0 {
                cfg.sk = c.sk;
            }
            if c.skd != 0 {
                cfg.skd = c.skd;
            }
            cfg.fp32_swiglu = c.fp32_swiglu;
        }
        let routes = rows * TOPK;
        let (sk, skd) = (cfg.sk as usize, cfg.skd as usize);
        let mut out = Intermediates {
            cfg,
            routes,
            z: vec![0f32; 2 * sk * routes * crate::consts::RANK_WIDTH],
            pair_route: vec![0i32; routes],
            xd: vec![0u16; routes * crate::consts::RANK_WIDTH],
            zd: vec![0f32; skd * routes * HIDDEN],
        };
        let mut err = [0u8; 256];
        // SAFETY: every destination holds the count passed with it.
        let rc = unsafe {
            g53r_debug_copy(
                self.scratch,
                out.z.as_mut_ptr(),
                out.z.len(),
                out.pair_route.as_mut_ptr(),
                out.pair_route.len(),
                out.xd.as_mut_ptr(),
                out.xd.len(),
                out.zd.as_mut_ptr(),
                out.zd.len(),
                err.as_mut_ptr() as *mut c_char,
                err.len(),
            )
        };
        if rc != 0 {
            return Err(err_string(&err));
        }
        Ok(out)
    }
}

impl Drop for CudaKernel {
    fn drop(&mut self) {
        // SAFETY: created by g53r_scratch_new, freed once.
        unsafe { g53r_scratch_free(self.scratch) }
    }
}

impl ExpertKernel for CudaKernel {
    type Layer = CudaLayer;

    fn prepare_layer(&mut self, image: &[u8]) -> Result<CudaLayer, String> {
        if image.len() != LAYER_BYTES {
            return Err(format!("layer image has {} bytes, want {LAYER_BYTES}", image.len()));
        }
        let mut out = core::ptr::null_mut();
        let mut err = [0u8; 256];
        // SAFETY: image is LAYER_BYTES of host memory; out/err valid for writes.
        let rc = unsafe {
            g53r_layer_new(image.as_ptr(), image.len() as u64, &mut out, err.as_mut_ptr() as *mut c_char, err.len())
        };
        if rc != 0 {
            return Err(err_string(&err));
        }
        Ok(CudaLayer(out))
    }

    fn ffn(&mut self, layer: &CudaLayer, rows: Rows<'_>, ids: &[i32], weights: &[f32], out: &mut [u16])
        -> Result<FfnStats, String> {
        rows.check()?;
        let n = rows.rows;
        if ids.len() != n * TOPK || weights.len() != n * TOPK || out.len() != n * HIDDEN {
            return Err(format!("ffn: extents do not match {n} rows"));
        }
        let mut ms = [0f32; 9];
        let mut err = [0u8; 256];
        let cfg = self.cfg;
        let cfg_ptr = cfg.as_ref().map_or(core::ptr::null(), |c| c as *const Cfg);
        // SAFETY: rows.check() bounds every row read; ids/weights/out are sized above.
        let rc = unsafe {
            g53r_ffn(
                layer.0,
                self.scratch,
                rows.payload.as_ptr(),
                rows.payload_pitch,
                rows.scales.as_ptr(),
                rows.scales_pitch,
                ids.as_ptr(),
                weights.as_ptr(),
                n as u32,
                out.as_mut_ptr(),
                cfg_ptr,
                ms.as_mut_ptr(),
                err.as_mut_ptr() as *mut c_char,
                err.len(),
            )
        };
        if rc != 0 {
            return Err(err_string(&err));
        }
        Ok(FfnStats {
            upload_ms: ms[0],
            gpu_ms: ms[1],
            tail_ms: ms[2],
            groups: ms[3] as u32,
            phase_ms: [ms[4], ms[5], ms[6], ms[7], ms[8]],
        })
    }
}
