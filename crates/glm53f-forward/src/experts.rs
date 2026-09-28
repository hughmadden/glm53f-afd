//! Routed experts behind a trait (feature `cuda`).
//!
//! The forward computes the router on the device, copies the routes to the host (the one host
//! round trip of a step: a remote backend needs them to build its frames), and hands the MoE
//! layer's rows to an [`ExpertBackend`] in two phases: [`ExpertBackend::submit`], then the
//! forward runs the shared expert on its stream, then [`ExpertBackend::finish`], after which the
//! routed sum must be in [`ExpertCall::out`] for later work on the stream.
//!
//! - [`LocalFp8Experts`]: the official FP8 experts on this GPU, loaded on demand from the
//!   checkpoint into a device cache with a byte budget. For tests on one GPU.
//! - [`ZeroExperts`]: a routed output of zeros (tests that only need the coordinator's path).
//! - The remote backend (the expert ranks over RDMA) comes from the serving shell's wire
//!   client: `submit` writes the frames (routes, FP8 or NVFP4 hidden rows) and posts them,
//!   `finish` waits for the returned plane and copies it into `out` on the stream.

use std::collections::HashMap;
use std::path::Path;

use glm53f_layers::ffi as lffi;
use glm53f_model::catalog::LAYERS;
use glm53f_model::safetensors::Checkpoint;

use crate::device::{launched, DeviceBuffer, Stream};
use crate::error::{invalid, Result};
use crate::ffi;
use crate::gemm::{Fp8Input, Gemm, GemmPolicy};
use crate::shape::{HIDDEN, MOE_INTER, TOP_K};
use crate::weights::{Fp8W, Loader};

/// One MoE layer's routed-expert work for one pass.
pub struct ExpertCall<'a> {
    /// Decoder layer.
    pub layer: usize,
    pub rows: usize,
    /// The FFN input after `post_attention_layernorm`: BF16 `[rows][4096]` on the device.
    pub x: *const u16,
    /// Its E4M3 form: codes `[rows][4096]` and f32 scales `[rows][32]` on the device.
    pub x_q: *const u8,
    pub x_scales: *const f32,
    /// The routes on the device: expert ids i32 `[rows][8]` and weights f32 `[rows][8]` (the
    /// normalized sigmoid scores times 2.5).
    pub ids: *const i32,
    pub weights: *const f32,
    /// The same routes on the host.
    pub host_ids: &'a [i32],
    pub host_weights: &'a [f32],
    /// Where the routed sum goes: BF16 `[rows][4096]` on the device.
    pub out: *mut u16,
}

/// Runs the routed experts of MoE layers.
pub trait ExpertBackend: Send {
    /// Start one layer's routed experts (work may be enqueued on `stream`).
    fn submit(&mut self, call: &ExpertCall<'_>, stream: &Stream) -> Result<()>;
    /// The routed sum is in `call.out` for work enqueued on `stream` after this returns.
    fn finish(&mut self, call: &ExpertCall<'_>, stream: &Stream) -> Result<()>;
}

/// Routed output of zeros.
pub struct ZeroExperts;

impl ExpertBackend for ZeroExperts {
    fn submit(&mut self, _call: &ExpertCall<'_>, _stream: &Stream) -> Result<()> {
        Ok(())
    }
    fn finish(&mut self, call: &ExpertCall<'_>, stream: &Stream) -> Result<()> {
        let n = call.rows * HIDDEN * 2;
        // SAFETY: `out` holds rows x hidden BF16 values.
        crate::device::check(
            unsafe { crate::cuda::cudaMemsetAsync(call.out.cast(), 0, n, stream.raw()) },
            "cudaMemsetAsync",
        )
    }
}

struct Resident {
    gate_up: Fp8W,
    down: Fp8W,
    last_use: u64,
}

impl Resident {
    fn bytes(&self) -> usize {
        self.gate_up.mat().bytes() + self.down.mat().bytes()
    }
}

/// The official FP8 routed experts on this GPU (tests on one GPU).
///
/// Each expert's rows go through `glm53f-layers`' FP8 decode GEMM in groups of up to 8 rows
/// (row-independent, so a row's routed output is the same whatever pass it is in): gate and
/// up, SwiGLU with the clamp, down. The routed sum is the reference's eager loop: per row, in
/// ascending expert id, `acc = bf16(acc + bf16(y * weight))`.
pub struct LocalFp8Experts {
    ckpt: Checkpoint,
    gemm: Gemm,
    budget: usize,
    used: usize,
    clock: u64,
    resident: HashMap<(usize, usize), Resident>,
    max_rows: usize,
    // Scratch for 8 rows of one expert, and every (row, slot) output of a call.
    xg: DeviceBuffer,
    xqg: DeviceBuffer,
    xsg: DeviceBuffer,
    gu: DeviceBuffer,
    act: DeviceBuffer,
    act_q: DeviceBuffer,
    act_s: DeviceBuffer,
    y: DeviceBuffer,
    ys: DeviceBuffer,
    idx: DeviceBuffer,
    /// Experts loaded from the checkpoint so far (a count, for reports).
    pub loads: usize,
}

impl LocalFp8Experts {
    /// Experts from the checkpoint at `dir` (the official checkpoint, or a subset holding the
    /// routed experts of the layers to run), with at most `budget` device bytes resident, for
    /// passes of up to `max_rows` rows.
    pub fn new(
        dir: &Path,
        budget: usize,
        max_rows: usize,
        stream: &Stream,
        act: crate::gemm::Fp8Act,
    ) -> Result<Self> {
        let ckpt = Checkpoint::open(dir)?;
        let policy = GemmPolicy {
            fp8_act: act,
            ..GemmPolicy::default()
        };
        Ok(LocalFp8Experts {
            ckpt,
            gemm: Gemm::new(stream, policy)?,
            budget,
            used: 0,
            clock: 0,
            resident: HashMap::new(),
            max_rows,
            xg: DeviceBuffer::alloc(8 * HIDDEN * 2)?,
            xqg: DeviceBuffer::alloc(8 * HIDDEN)?,
            xsg: DeviceBuffer::alloc(8 * (HIDDEN / 128) * 4)?,
            gu: DeviceBuffer::alloc(8 * 2 * MOE_INTER * 2)?,
            act: DeviceBuffer::alloc(8 * MOE_INTER * 2)?,
            act_q: DeviceBuffer::alloc(8 * MOE_INTER)?,
            act_s: DeviceBuffer::alloc(8 * (MOE_INTER / 128) * 4)?,
            y: DeviceBuffer::alloc(8 * HIDDEN * 2)?,
            ys: DeviceBuffer::alloc(max_rows * TOP_K * HIDDEN * 2)?,
            idx: DeviceBuffer::alloc(max_rows * TOP_K * 2 * 4)?,
            loads: 0,
        })
    }

    /// Device bytes of the resident experts.
    pub fn resident_bytes(&self) -> usize {
        self.used
    }

    fn ensure(&mut self, layer: usize, e: usize) -> Result<()> {
        self.clock += 1;
        if let Some(r) = self.resident.get_mut(&(layer, e)) {
            r.last_use = self.clock;
            return Ok(());
        }
        let p = format!("{LAYERS}{layer}.mlp.experts.{e}.");
        let mut ld = Loader::new(&self.ckpt);
        let gate_up = ld.fp8(
            &[&format!("{p}gate_proj"), &format!("{p}up_proj")],
            &[MOE_INTER, MOE_INTER],
            HIDDEN,
        )?;
        let down = ld.fp8(&[&format!("{p}down_proj")], &[HIDDEN], MOE_INTER)?;
        let r = Resident {
            gate_up,
            down,
            last_use: self.clock,
        };
        let need = r.bytes();
        // Evict the least recently used until it fits (cudaFree waits for work in flight).
        while self.used + need > self.budget && !self.resident.is_empty() {
            let (&key, _) = self
                .resident
                .iter()
                .min_by_key(|(_, v)| v.last_use)
                .unwrap();
            let old = self.resident.remove(&key).unwrap();
            self.used -= old.bytes();
        }
        self.used += need;
        self.loads += 1;
        self.resident.insert((layer, e), r);
        Ok(())
    }
}

impl ExpertBackend for LocalFp8Experts {
    fn submit(&mut self, call: &ExpertCall<'_>, stream: &Stream) -> Result<()> {
        let rows = call.rows;
        if rows > self.max_rows {
            return Err(invalid!(
                "{rows} rows for local experts sized for {}",
                self.max_rows
            ));
        }
        if call.host_ids.len() != rows * TOP_K || call.host_weights.len() != rows * TOP_K {
            return Err(invalid!("routes for {rows} rows"));
        }
        // Rows and output slots of every expert, in ascending expert id.
        let mut by_expert: Vec<(usize, Vec<(i32, i32)>)> = Vec::new();
        {
            let mut map: HashMap<usize, Vec<(i32, i32)>> = HashMap::new();
            for r in 0..rows {
                for j in 0..TOP_K {
                    let id = call.host_ids[r * TOP_K + j];
                    if id < 0 {
                        continue;
                    }
                    if id as usize >= crate::shape::EXPERTS {
                        return Err(invalid!("expert id {id}"));
                    }
                    map.entry(id as usize)
                        .or_default()
                        .push((r as i32, (r * TOP_K + j) as i32));
                }
            }
            let mut keys: Vec<usize> = map.keys().copied().collect();
            keys.sort_unstable();
            for k in keys {
                by_expert.push((k, map.remove(&k).unwrap()));
            }
        }
        // One upload of every gather and scatter index: rows, then slots, per expert.
        let mut idx: Vec<i32> = Vec::with_capacity(rows * TOP_K * 2);
        let mut offsets = Vec::with_capacity(by_expert.len());
        for (_, list) in &by_expert {
            offsets.push(idx.len());
            idx.extend(list.iter().map(|&(r, _)| r));
            idx.extend(list.iter().map(|&(_, s)| s));
        }
        self.idx.upload_async(stream, 0, &idx)?;
        let st = stream.raw();
        for ((e, list), &off) in by_expert.iter().zip(&offsets) {
            self.ensure(call.layer, *e)?;
            let r = &self.resident[&(call.layer, *e)];
            let (gu_m, down_m) = (r.gate_up.mat(), r.down.mat());
            let n = list.len();
            let mut done = 0;
            while done < n {
                let m = (n - done).min(8);
                let rows_idx: *const i32 = self.idx.ptr(off + done);
                let slots_idx: *const i32 = self.idx.ptr(off + n + done);
                // SAFETY (all launches below): device buffers sized for 8 rows of this expert and
                // `rows` rows of the call; index entries are rows < `rows` and slots < rows x 8.
                launched(
                    unsafe {
                        ffi::glm53f_fwd_gather_rows(
                            call.x.cast(),
                            (HIDDEN * 2) as i64,
                            rows_idx,
                            self.xg.ptr(0),
                            (HIDDEN * 2) as i64,
                            m as i32,
                            (HIDDEN * 2) as i64,
                            st,
                        )
                    },
                    "expert gather",
                )?;
                let quant = self.gemm.policy.fp8_needs_quant(m);
                if quant {
                    if call.x_q.is_null() || call.x_scales.is_null() {
                        return Err(invalid!("W8A8 local experts need the E4M3 input"));
                    }
                    launched(
                        unsafe {
                            ffi::glm53f_fwd_gather_rows(
                                call.x_q,
                                HIDDEN as i64,
                                rows_idx,
                                self.xqg.ptr(0),
                                HIDDEN as i64,
                                m as i32,
                                HIDDEN as i64,
                                st,
                            )
                        },
                        "expert gather q",
                    )?;
                    launched(
                        unsafe {
                            ffi::glm53f_fwd_gather_rows(
                                call.x_scales.cast(),
                                (HIDDEN / 128 * 4) as i64,
                                rows_idx,
                                self.xsg.ptr(0),
                                (HIDDEN / 128 * 4) as i64,
                                m as i32,
                                (HIDDEN / 128 * 4) as i64,
                                st,
                            )
                        },
                        "expert gather scales",
                    )?;
                }
                let xin = Fp8Input {
                    bf16: self.xg.ptr(0),
                    q: self.xqg.ptr(0),
                    scales: self.xsg.ptr(0),
                };
                unsafe { self.gemm.fp8(&xin, &gu_m, m, self.gu.ptr(0), stream) }?;
                launched(
                    unsafe {
                        lffi::glm53f_swiglu(
                            self.gu.ptr(0),
                            self.act.ptr(0),
                            if quant {
                                self.act_q.ptr(0)
                            } else {
                                core::ptr::null_mut()
                            },
                            if quant {
                                self.act_s.ptr(0)
                            } else {
                                core::ptr::null_mut()
                            },
                            m as i32,
                            MOE_INTER as i32,
                            st.cast(),
                        )
                    },
                    "expert swiglu",
                )?;
                let ain = Fp8Input {
                    bf16: self.act.ptr(0),
                    q: self.act_q.ptr(0),
                    scales: self.act_s.ptr(0),
                };
                unsafe { self.gemm.fp8(&ain, &down_m, m, self.y.ptr(0), stream) }?;
                launched(
                    unsafe {
                        ffi::glm53f_fwd_scatter_rows(
                            self.y.ptr(0),
                            (HIDDEN * 2) as i64,
                            slots_idx,
                            self.ys.ptr(0),
                            (HIDDEN * 2) as i64,
                            m as i32,
                            (HIDDEN * 2) as i64,
                            st,
                        )
                    },
                    "expert scatter",
                )?;
                done += m;
            }
        }
        // SAFETY: ys holds every (row, slot) output the ids name; out holds rows x hidden.
        launched(
            unsafe {
                ffi::glm53f_fwd_moe_combine(
                    self.ys.ptr(0),
                    call.ids,
                    call.weights,
                    rows as i32,
                    TOP_K as i32,
                    HIDDEN as i32,
                    call.out,
                    st,
                )
            },
            "expert combine",
        )
    }

    fn finish(&mut self, _call: &ExpertCall<'_>, _stream: &Stream) -> Result<()> {
        Ok(())
    }
}
