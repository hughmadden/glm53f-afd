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

/// A kernel configuration; zero fields take the defaults for the row count
/// (README "The kernel"):
///
/// - `mt`: 16-row tiles per expert group (1 or 2 with the split kernels, 2
///   or 4 with the large-M kernels);
/// - `sk`, `skd`: gate/up and down K splits, dividing 32 (the large-M down
///   kernel takes 1, 2 or 4);
/// - `fp32_swiglu` = 1 computes the SwiGLU in FP32 instead of with the
///   reference model's BF16 roundings (the default);
/// - `big`: 1 the split kernels, 2 the large-M kernels;
/// - `nt`: large-M down, 16-column tiles per warp (1, 2 or 4), blocks of
///   `128 nt` output columns;
/// - `fuse`: 1 separate epilogue and reduce kernels, 2 fused into gate/up
///   and down;
/// - `discard`: 1 keep the partial sums, 2 drop them from L2 once consumed
///   (fused only; [`CudaKernel::intermediates`] then refuses);
/// - `gw`: large-M gate/up, MMA warps per block (8 or 16);
/// - `gp`: large-M gate/up, rotation warps per block (2, with `gw` 8 and `mt`
///   2; or -1 for none: the MMA warps rotate their own input);
/// - `plan`: 1 the planning kernel, 2 the split gate/up blocks plan the call
///   themselves (split kernels, up to 512 routes; the planning kernel above);
/// - `l2`: 1 default caching of the trellis words, 2 an L2 evict-first policy
///   for them (both kernel families);
/// - `pf`: split kernels, trellis words loaded 1, 2 or 4 k tiles ahead (1
///   with the large-M kernels);
/// - `ord`: split kernels' block order, 1 the split slowest, 2 a group's
///   gate/up blocks adjacent and the down blocks chunk by chunk with the
///   splits fastest, so the fused steps read partial sums that were just
///   written (1 with the large-M kernels);
/// - `pdl`: 2 launches the split down kernel as a programmatic dependent
///   launch (`sm_90` and later): its blocks plan themselves from the routes
///   and load their first trellis words while the gate/up kernel ends (split
///   kernels with `plan` 2 and `fuse` 2, up to 512 routes; older devices run
///   the same kernels in stream order).
///
/// Only `sk`, `skd` and `fp32_swiglu` change a bit of the output (the tests
/// compare the others bit for bit).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Cfg {
    pub mt: c_int,
    pub sk: c_int,
    pub skd: c_int,
    pub fp32_swiglu: c_int,
    pub big: c_int,
    pub nt: c_int,
    pub fuse: c_int,
    pub discard: c_int,
    pub gw: c_int,
    pub gp: c_int,
    pub plan: c_int,
    pub l2: c_int,
    pub pf: c_int,
    pub ord: c_int,
    pub pdl: c_int,
}

impl Cfg {
    /// Parse `key=value` pairs separated by commas (`mt=4,sk=1,nt=2`), over
    /// `self`. Keys are the field names.
    pub fn parse_over(mut self, s: &str) -> Result<Self, String> {
        for kv in s.split(',').map(str::trim).filter(|x| !x.is_empty()) {
            let (k, v) = kv.split_once('=').ok_or_else(|| format!("kernel configuration: {kv:?} is not key=value"))?;
            let v: c_int = v.trim().parse().map_err(|_| format!("kernel configuration: {kv:?} has no integer value"))?;
            match k.trim() {
                "mt" => self.mt = v,
                "sk" => self.sk = v,
                "skd" => self.skd = v,
                "fp32_swiglu" => self.fp32_swiglu = v,
                "big" => self.big = v,
                "nt" => self.nt = v,
                "fuse" => self.fuse = v,
                "discard" => self.discard = v,
                "gw" => self.gw = v,
                "gp" => self.gp = v,
                "plan" => self.plan = v,
                "l2" => self.l2 = v,
                "pf" => self.pf = v,
                "ord" => self.ord = v,
                "pdl" => self.pdl = v,
                other => return Err(format!("kernel configuration: unknown key {other:?}")),
            }
        }
        Ok(self)
    }

    /// `mt=.. sk=.. ...` (every field).
    pub fn text(&self) -> String {
        format!(
            "mt={},sk={},skd={},fp32_swiglu={},big={},nt={},gw={},gp={},plan={},fuse={},discard={},l2={},pf={},ord={},pdl={}",
            self.mt,
            self.sk,
            self.skd,
            self.fp32_swiglu,
            self.big,
            self.nt,
            self.gw,
            self.gp,
            self.plan,
            self.fuse,
            self.discard,
            self.l2,
            self.pf,
            self.ord,
            self.pdl
        )
    }
}

unsafe extern "C" {
    fn g53r_baked_arch() -> c_int;
    fn g53r_device_identity(arch: *mut c_int, sms: *mut c_int, name: *mut c_char, namelen: usize) -> c_int;
    fn g53r_default_cfg(rows: u32, out: *mut Cfg);
    fn g53r_resolve_cfg(rows: u32, cfg: *const Cfg, out: *mut Cfg) -> c_int;
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
    fn g53r_ffn_f32(
        l: *const RawLayer,
        s: *mut RawScratch,
        payload: *const u8,
        payload_pitch: usize,
        scales: *const u8,
        scales_pitch: usize,
        ids: *const i32,
        weights: *const f32,
        rows: u32,
        f32_out: *mut f32,
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

/// The kernel's compiled-in default configuration for `rows` rows.
pub fn default_cfg(rows: usize) -> Cfg {
    let mut c = Cfg::default();
    // SAFETY: c is a valid out-pointer.
    unsafe { g53r_default_cfg(rows as u32, &mut c) };
    c
}

/// The configuration a call of `rows` rows runs with `cfg` (zero fields
/// filled from the defaults for the row count), or an error if the kernel
/// refuses it.
pub fn resolve_cfg(rows: usize, cfg: Option<Cfg>) -> Result<Cfg, String> {
    let mut out = Cfg::default();
    let p = cfg.as_ref().map_or(core::ptr::null(), |c| c as *const Cfg);
    // SAFETY: p is null or a valid Cfg; out is a valid out-pointer.
    match unsafe { g53r_resolve_cfg(rows.max(1) as u32, p, &mut out) } {
        0 => Ok(out),
        _ => Err(format!("kernel configuration {} is not valid for {rows} rows", out.text())),
    }
}

/// Which configuration each call runs when [`CudaKernel::cfg`] is unset, by
/// row count: `small` up to `small_max` rows (decode and verify windows: one
/// configuration, so a row's bits do not depend on the window size), `mid` up
/// to `mid_max` rows and `large` above. `mid` and `large` must share their K
/// splits and SwiGLU (the only fields that change a bit), so a prefill row's
/// bits do not depend on the row count either. Built from the compiled-in
/// defaults and the environment:
///
/// - `GLM53F_RANK_SMALL_MAX` (default 64) and `GLM53F_RANK_MID_MAX` (default
///   2,048): the regimes' largest row counts;
/// - `GLM53F_RANK_SMALL`, `GLM53F_RANK_MID`, `GLM53F_RANK_LARGE`: `key=value`
///   pairs over each regime's defaults (`mt=4,nt=1,discard=1`; keys as in
///   [`Cfg`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Policy {
    pub small_max: usize,
    pub mid_max: usize,
    pub small: Cfg,
    pub mid: Cfg,
    pub large: Cfg,
}

impl Policy {
    /// The compiled-in defaults.
    pub fn defaults() -> Self {
        Policy {
            small_max: 64,
            mid_max: 2048,
            small: default_cfg(1),
            mid: default_cfg(2048),
            large: default_cfg(crate::consts::MAX_ROWS),
        }
    }

    /// The defaults with the environment's overrides, each regime checked.
    pub fn from_env() -> Result<Self, String> {
        let mut p = Self::defaults();
        let var = |k: &str| std::env::var(k).ok().filter(|v| !v.trim().is_empty());
        for (key, max) in [("GLM53F_RANK_SMALL_MAX", &mut p.small_max), ("GLM53F_RANK_MID_MAX", &mut p.mid_max)] {
            if let Some(v) = var(key) {
                *max = v.trim().parse().map_err(|_| format!("{key}={v:?} is not a row count"))?;
            }
        }
        let regimes = [("GLM53F_RANK_SMALL", &mut p.small), ("GLM53F_RANK_MID", &mut p.mid), ("GLM53F_RANK_LARGE", &mut p.large)];
        for (key, cfg) in regimes {
            if let Some(v) = var(key) {
                *cfg = cfg.parse_over(&v).map_err(|e| format!("{key}: {e}"))?;
            }
        }
        if p.small_max > p.mid_max {
            return Err(format!("GLM53F_RANK_SMALL_MAX {} is above GLM53F_RANK_MID_MAX {}", p.small_max, p.mid_max));
        }
        let small = resolve_cfg(1, Some(p.small)).map_err(|e| format!("GLM53F_RANK_SMALL: {e}"))?;
        let mid = resolve_cfg(crate::consts::MAX_ROWS, Some(p.mid)).map_err(|e| format!("GLM53F_RANK_MID: {e}"))?;
        let large = resolve_cfg(crate::consts::MAX_ROWS, Some(p.large)).map_err(|e| format!("GLM53F_RANK_LARGE: {e}"))?;
        if (mid.sk, mid.skd, mid.fp32_swiglu) != (large.sk, large.skd, large.fp32_swiglu) {
            return Err(format!(
                "GLM53F_RANK_MID and GLM53F_RANK_LARGE must share sk, skd and fp32_swiglu \
                 (a prefill row's bits would depend on the row count): {} vs {}",
                mid.text(),
                large.text()
            ));
        }
        (p.small, p.mid, p.large) = (small, mid, large);
        Ok(p)
    }

    /// The configuration for a call of `rows` rows.
    pub fn for_rows(&self, rows: usize) -> Cfg {
        if rows <= self.small_max {
            self.small
        } else if rows <= self.mid_max {
            self.mid
        } else {
            self.large
        }
    }

    /// One line for the boot log.
    pub fn summary(&self) -> String {
        format!(
            "kernel: up to {} rows {}; up to {} rows {}; above {}",
            self.small_max,
            self.small.text(),
            self.mid_max,
            self.mid.text(),
            self.large.text()
        )
    }
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
    /// A fixed configuration for every call (tests and benchmarks; zero
    /// fields take the compiled-in defaults), or `None` for the policy.
    pub cfg: Option<Cfg>,
    /// The configurations by row count when `cfg` is `None`.
    pub policy: Policy,
}

// SAFETY: as for CudaLayer.
unsafe impl Send for CudaKernel {}

impl CudaKernel {
    /// A kernel with the environment's policy ([`Policy::from_env`]).
    pub fn new() -> Result<Self, String> {
        let policy = Policy::from_env()?;
        let mut out = core::ptr::null_mut();
        let mut err = [0u8; 256];
        // SAFETY: out/err valid for writes.
        let rc = unsafe { g53r_scratch_new(&mut out, err.as_mut_ptr() as *mut c_char, err.len()) };
        if rc != 0 {
            return Err(err_string(&err));
        }
        Ok(Self { scratch: out, cfg: None, policy })
    }

    /// The configuration a call of `rows` rows runs.
    pub fn cfg_for(&self, rows: usize) -> Result<Cfg, String> {
        match self.cfg {
            Some(c) => resolve_cfg(rows, Some(c)),
            None => resolve_cfg(rows, Some(self.policy.for_rows(rows))),
        }
    }
}

impl CudaKernel {
    /// Copy the intermediates of the last [`ExpertKernel::ffn`] call of `rows`
    /// rows (a test hook; the configuration must be the one that call used,
    /// with `discard` 1).
    pub fn intermediates(&self, rows: usize) -> Result<Intermediates, String> {
        let cfg = self.cfg_for(rows)?;
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
        let (l, s) = (layer.0, self.scratch);
        self.call(rows, ids, weights, out.len(), |cfg, ms, err, errlen| {
            // SAFETY: `call` checked the rows (every read in bounds) and the extents of ids,
            // weights and out.
            unsafe {
                g53r_ffn(l, s, rows.payload.as_ptr(), rows.payload_pitch, rows.scales.as_ptr(), rows.scales_pitch,
                    ids.as_ptr(), weights.as_ptr(), rows.rows as u32, out.as_mut_ptr(), cfg, ms, err, errlen)
            }
        })
    }

    fn ffn_f32(&mut self, layer: &CudaLayer, rows: Rows<'_>, ids: &[i32], weights: &[f32], out: &mut [f32])
        -> Result<FfnStats, String> {
        let (l, s) = (layer.0, self.scratch);
        self.call(rows, ids, weights, out.len(), |cfg, ms, err, errlen| {
            // SAFETY: as in `ffn`.
            unsafe {
                g53r_ffn_f32(l, s, rows.payload.as_ptr(), rows.payload_pitch, rows.scales.as_ptr(), rows.scales_pitch,
                    ids.as_ptr(), weights.as_ptr(), rows.rows as u32, out.as_mut_ptr(), cfg, ms, err, errlen)
            }
        })
    }
}

impl CudaKernel {
    /// The checks and statistics around one of the FFN entry points: `ffi(cfg, ms, err, errlen)`
    /// makes the call once the rows and the extents (`out_len` output values) are checked.
    fn call(
        &mut self,
        rows: Rows<'_>,
        ids: &[i32],
        weights: &[f32],
        out_len: usize,
        ffi: impl FnOnce(*const Cfg, *mut f32, *mut c_char, usize) -> c_int,
    ) -> Result<FfnStats, String> {
        rows.check()?;
        let n = rows.rows;
        if ids.len() != n * TOPK || weights.len() != n * TOPK || out_len != n * HIDDEN {
            return Err(format!("ffn: extents do not match {n} rows"));
        }
        let mut ms = [0f32; 9];
        let mut err = [0u8; 256];
        let cfg = self.cfg.unwrap_or_else(|| self.policy.for_rows(n));
        if ffi(&cfg, ms.as_mut_ptr(), err.as_mut_ptr() as *mut c_char, err.len()) != 0 {
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
