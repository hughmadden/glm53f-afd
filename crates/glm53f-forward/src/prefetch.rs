//! L2 prefetch of the next layer's weights while a MoE layer's routed experts are out (feature
//! `cuda`; [`crate::forward::ForwardConfig::l2_prefetch`], off by default).
//!
//! In a decode or verify pass the coordinator runs a MoE layer's attention and router, sends the
//! rows to the expert ranks, runs the shared expert, and then waits for the ranks: in a pass of
//! one lane the GPU is idle for most of each exchange (the ranks' kernel alone takes about 0.15 ms
//! at one row, `docs/PERFORMANCE.md` §1). The next layer's attention is bound by its weight reads
//! (the BF16 KDA projections are about 275 MB a layer). With the prefetch on, once the shared
//! expert is queued the forward queues after it, on its own stream, a kernel that reads the first
//! `l2_prefetch` bytes of the next layer's weights, in the order that layer reads them, through L2
//! ([`Mode::Load`]). The next layer's GEMVs then find those bytes in L2.
//!
//! - **Bits.** Nothing is written and no value changes: the GEMVs read the same bytes, only from
//!   another level of the memory. `tests/decode_lanes.rs` checks every output with the prefetch
//!   on against the same run with it off, bit for bit, and `examples/logits_digest.rs` gives the
//!   same decode digest with it on and off.
//! - **On the forward's stream, in passes of one lane only.** The exchange's consumer (the routed
//!   sum, the next layer) is queued behind the prefetch: today by the host once the exchange has
//!   returned. A kernel queued behind a device-side wait (a stream memory-operation wait, as an
//!   exchange driven from the device would use) was seen on the development GPU to be held until
//!   a kernel of another stream ended, so the prefetch shares the forward's stream rather than
//!   racing it from another, and runs only where it ends first: in a pass of one lane it follows
//!   the shared expert and takes 77 us at 48 MiB on the RTX 4090 (`examples/l2_prefetch_bench.rs`),
//!   against an exchange of at least the ranks' kernel time. In a pass of two lanes the other
//!   lane's attention fills the exchange, and a prefetch could hold up the lane whose routed
//!   experts are back: those passes never prefetch. The kernel reads one byte of every `stride`
//!   bytes (the lines come whole) and discards them, from one block of 256 threads per
//!   multiprocessor.
//! - **No host synchronisation.** The hook is one launch on the forward's stream, its ranges
//!   passed by value: a stream capture records it as it is.
//! - **What.** A layer's weights in the order a decode pass reads them ([`read_order`]): the
//!   attention boundary's projection and norm, the attention (KDA: q|k|v|b, f_a|g_a, f_b|g_b, the
//!   conv and gate vectors, o_proj; DSA: q_a, kv_a, q_b, the indexer's projections, kv_b, o_proj),
//!   then the FFN boundary, the router and the shared expert. After the last layer, the head (the
//!   final norm, then the LM head). A prefix of at most `budget` bytes is prefetched.
//! - **When.** Decode and verify passes of one lane (one request, or more rows than the decode
//!   lanes take), after each MoE layer's call went out and its shared expert was queued. Prefill
//!   passes keep the GPU busy with the other lanes' attention and never prefetch.
//!
//! How much to prefetch depends on the GPU's L2 (72 MiB on the RTX 4090, 96 MB on the RTX 5090 by
//! its specification; [`crate::device::l2_bytes`] reads it) and on how long the exchange leaves
//! the GPU idle. `examples/l2_prefetch_bench.rs` measures the kernel against one KDA layer's
//! GEMVs, and `examples/decode_bench.rs` decode steps with the exchange emulated, on one GPU.

use core::ffi::c_void;

use crate::device::{self, launched, DeviceBuffer, Stream};
use crate::error::Result;
use crate::ffi;
use crate::weights::{AttnW, DeviceModel, FfnW, Fp8W, HcW, HeadW, LayerW, MlpW, ProjW};

/// Ranges one launch takes at most (`GLM53F_FWD_PREFETCH_MAX_RANGES`).
pub const MAX_RANGES: usize = 16;

/// A device range: its first byte and its length.
pub type Range = (*const u8, usize);

fn buf(v: &mut Vec<Range>, b: &DeviceBuffer) {
    v.push((b.ptr::<u8>(0).cast_const(), b.bytes()));
}

fn fp8(v: &mut Vec<Range>, w: &Fp8W) {
    buf(v, &w.w);
    buf(v, &w.scales);
}

fn proj(v: &mut Vec<Range>, p: &ProjW) {
    match p {
        ProjW::Bf16(w) => buf(v, &w.buf),
        ProjW::Fp8(w) => fp8(v, w),
    }
}

fn hc(v: &mut Vec<Range>, h: &HcW) {
    buf(v, &h.fn_);
    buf(v, &h.base);
    buf(v, &h.scale);
}

fn mlp(v: &mut Vec<Range>, m: &MlpW) {
    fp8(v, &m.gate_up);
    fp8(v, &m.down);
}

/// The device ranges of a decoder layer's weights in the order a decode or verify pass reads
/// them (`crate::forward`'s `attention`, `kda`, `dsa`, `mlp`, `moe_router`).
pub fn read_order(l: &LayerW) -> Vec<Range> {
    let mut v = Vec::with_capacity(32);
    hc(&mut v, &l.attn_hc);
    buf(&mut v, &l.input_norm);
    match &l.attn {
        AttnW::Kda(w) => {
            proj(&mut v, &w.qkvb);
            buf(&mut v, &w.fga.buf);
            buf(&mut v, &w.fgb.buf);
            buf(&mut v, &w.conv_w);
            buf(&mut v, &w.a_log);
            buf(&mut v, &w.dt_bias);
            buf(&mut v, &w.o_norm);
            proj(&mut v, &w.o);
        }
        AttnW::Dsa(w) => {
            fp8(&mut v, &w.q_a);
            fp8(&mut v, &w.kv_a);
            buf(&mut v, &w.q_a_norm);
            fp8(&mut v, &w.q_b);
            buf(&mut v, &w.idx_wq_b.buf);
            buf(&mut v, &w.idx_proj.buf);
            buf(&mut v, &w.kv_a_norm);
            buf(&mut v, &w.k_norm_w);
            buf(&mut v, &w.k_norm_b);
            buf(&mut v, &w.ape);
            buf(&mut v, &w.kv_b);
            fp8(&mut v, &w.o);
        }
    }
    hc(&mut v, &l.ffn_hc);
    buf(&mut v, &l.post_attn_norm);
    match &l.ffn {
        FfnW::Dense(m) => mlp(&mut v, m),
        FfnW::Moe {
            router,
            bias,
            shared,
        } => {
            buf(&mut v, router);
            buf(&mut v, bias);
            mlp(&mut v, shared);
        }
    }
    v
}

/// The head's ranges in read order: the final norm, then the LM head.
pub fn head_order(h: &HeadW) -> Vec<Range> {
    let mut v = Vec::with_capacity(2);
    buf(&mut v, &h.norm);
    buf(&mut v, &h.lm_head.buf);
    v
}

/// The first `budget` bytes of `ranges` (the last range cut short), at most [`MAX_RANGES`]
/// ranges.
pub fn prefix(ranges: &[Range], budget: usize) -> Vec<Range> {
    let mut out = Vec::with_capacity(MAX_RANGES);
    let mut left = budget;
    for &(p, n) in ranges {
        if left == 0 || out.len() == MAX_RANGES {
            break;
        }
        let take = n.min(left);
        if take > 0 {
            out.push((p, take));
        }
        left -= take;
    }
    out
}

/// How the prefetch kernel brings lines into L2 (`glm53f_forward.h`, `GLM53F_FWD_PREFETCH_*`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Mode {
    /// One-byte loads through L2 (`ld.global.cg`, `.L2::256B`), their values discarded: the
    /// kernel waits for the lines. The default.
    Load = 0,
    /// `prefetch.global.L2`, a hint the memory system may drop (on the RTX 4090 it dropped most).
    Hint = 1,
}

/// Every layer's read order and the kernel's settings.
pub struct L2Prefetch {
    /// Per decoder layer, then the head.
    plans: Vec<Vec<Range>>,
    blocks: i32,
    /// Bytes between two touches of a range (32, 64, 128 or 256).
    pub stride: i32,
    pub mode: Mode,
    /// Prefetches queued so far.
    pub launches: u64,
}

// SAFETY: the ranges are device addresses of the model's weights, usable from any host thread;
// the forward that owns the model owns this too, and its prefetches run on the forward's stream.
unsafe impl Send for L2Prefetch {}

impl L2Prefetch {
    /// No plan: [`L2Prefetch::launch`] only (benchmarks).
    pub fn new() -> Result<L2Prefetch> {
        Ok(L2Prefetch {
            plans: Vec::new(),
            blocks: device::sm_count()?.max(1),
            stride: 128,
            mode: Mode::Load,
            launches: 0,
        })
    }

    /// The prefetch plan of `model`'s layers and head.
    pub fn for_model(model: &DeviceModel) -> Result<L2Prefetch> {
        let mut p = L2Prefetch::new()?;
        p.plans = model.layers.iter().map(|l| read_order(l)).collect();
        p.plans.push(head_order(&model.head));
        Ok(p)
    }

    /// The ranges a prefetch of `budget` bytes before decoder layer `next` takes (`next` past the
    /// last layer: the head's).
    pub fn ranges(&self, next: usize, budget: usize) -> Vec<Range> {
        match self.plans.len() {
            0 => Vec::new(),
            n => prefix(&self.plans[next.min(n - 1)], budget),
        }
    }

    /// Queue on `stream` the prefetch of the first `budget` bytes decoder layer `next` reads (past
    /// the last layer: the head's).
    pub fn layer(&mut self, stream: &Stream, next: usize, budget: usize) -> Result<()> {
        let r = self.ranges(next, budget);
        if r.is_empty() {
            return Ok(());
        }
        self.launch(stream, &r)
    }

    /// Queue a prefetch of `ranges` (at most [`MAX_RANGES`]) on `stream`.
    pub fn launch(&mut self, stream: &Stream, ranges: &[Range]) -> Result<()> {
        let ptrs: Vec<*const c_void> = ranges.iter().map(|r| r.0.cast()).collect();
        let bytes: Vec<i64> = ranges.iter().map(|r| r.1 as i64).collect();
        // SAFETY: host arrays of `ranges.len()` entries (checked against the kernel's limit by the
        // callee); every range is part of a live weight buffer, and a prefetch writes nothing.
        launched(
            unsafe {
                ffi::glm53f_fwd_l2_prefetch(
                    ptrs.as_ptr(),
                    bytes.as_ptr(),
                    ranges.len() as i32,
                    self.stride,
                    self.blocks,
                    self.mode as i32,
                    stream.raw(),
                )
            },
            "glm53f_fwd_l2_prefetch",
        )?;
        self.launches += 1;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_prefix_of_the_ranges() {
        let r: Vec<Range> = (0..20)
            .map(|i| ((0x1000 * (i + 1)) as *const u8, 100))
            .collect();
        let p = prefix(&r, 250);
        assert_eq!(p.len(), 3);
        assert_eq!((p[0].1, p[1].1, p[2].1), (100, 100, 50));
        assert_eq!(p[2].0, r[2].0);
        assert!(prefix(&r, 0).is_empty());
        // At most MAX_RANGES ranges, whatever the budget.
        assert_eq!(prefix(&r, usize::MAX).len(), MAX_RANGES);
        let with_empty = [(r[0].0, 0), r[1]];
        assert_eq!(prefix(&with_empty, 50), vec![(r[1].0, 50)]);
    }
}
