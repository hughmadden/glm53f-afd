//! The op profile of prefill passes (feature `cuda`): the GPU time of every operation of each
//! lane's attention sublayer and shared expert, per layer, from CUDA events on the stream.
//!
//! [`crate::forward::GlmForward::set_op_trace`] turns it on (`GLM53F_PROFILE_OPS=1` does at
//! construction, with the lane trace: every prefill pass then prints its `PIPE` line and its
//! `OPS` table to stderr). Each lane's layer has two **segments**, which start at the lane
//! trace's own marks:
//!
//! - **attention**: from the lane's `AttnStart` to the end of its router, the interval the
//!   `PIPE` line calls "GPU attention" (a dense layer's segment runs on through its MLP, which
//!   the `PIPE` line does not count);
//! - **shared**: the MoE shared expert, the `PIPE` line's "shared".
//!
//! An event is recorded after each operation's launches, so an op's time is the stream's time
//! from the previous event to its own: its kernels and any gap before them. A segment's ops
//! therefore add up to the segment exactly, and the attention segment of an MoE layer to the
//! lane trace's "GPU attention" but for the two back-to-back events at its ends.
//!
//! Off, the forward's hooks cost one `RefCell` check each; on, one event per op (about 30 per
//! lane and layer, 2,700 in a two-lane pass of 45 layers) and a stream synchronization at the
//! end of each prefill pass.

use crate::device::{Event, Stream};
use crate::error::Result;

/// The part of a lane's layer an op belongs to.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Segment {
    /// The attention sublayer with both mHC boundaries and the router (and, in a dense layer,
    /// its MLP).
    Attention,
    /// The MoE shared expert.
    Shared,
}

/// A layer's kind, as the `OPS` table groups them.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum LayerKind {
    /// KDA attention, MoE FFN (31 of GLM-5.3-Flash's layers).
    KdaMoe,
    /// DSA attention, MoE FFN (11).
    DsaMoe,
    /// KDA attention, dense MLP (layers 0-2).
    KdaDense,
    /// DSA attention, dense MLP (none in GLM-5.3-Flash; development shapes only).
    DsaDense,
}

impl LayerKind {
    pub fn new(dsa: bool, moe: bool) -> LayerKind {
        match (dsa, moe) {
            (false, true) => LayerKind::KdaMoe,
            (true, true) => LayerKind::DsaMoe,
            (false, false) => LayerKind::KdaDense,
            (true, false) => LayerKind::DsaDense,
        }
    }

    pub fn is_moe(self) -> bool {
        matches!(self, LayerKind::KdaMoe | LayerKind::DsaMoe)
    }

    fn label(self) -> &'static str {
        match self {
            LayerKind::KdaMoe => "KDA MoE layers",
            LayerKind::DsaMoe => "DSA MoE layers",
            LayerKind::KdaDense => "KDA dense layers (attention and dense MLP)",
            LayerKind::DsaDense => "DSA dense layers (attention and dense MLP)",
        }
    }
}

/// One op's GPU time in one lane's layer.
#[derive(Clone, Debug)]
pub struct OpTime {
    pub layer: usize,
    pub lane: usize,
    /// The lane's rows.
    pub rows: usize,
    pub kind: LayerKind,
    pub segment: Segment,
    pub op: &'static str,
    pub ms: f64,
}

/// One op's median over the layers of a kind and the lanes of a row count.
#[derive(Clone, Debug)]
pub struct OpMedian {
    pub segment: Segment,
    pub op: &'static str,
    pub ms: f64,
}

/// The ops of one layer kind at one lane size: medians over its (layer, lane) samples.
#[derive(Clone, Debug)]
pub struct KindSummary {
    pub kind: LayerKind,
    pub rows: usize,
    /// Distinct layers and (layer, lane) samples.
    pub layers: usize,
    pub samples: usize,
    /// Each op's median, in pass order.
    pub ops: Vec<OpMedian>,
    /// Medians of the samples' segment totals.
    pub attention_ms: f64,
    pub shared_ms: f64,
}

/// The op times of one traced prefill pass.
#[derive(Clone, Debug, Default)]
pub struct OpProfile {
    /// Rows per lane.
    pub rows: Vec<usize>,
    /// Every op of every lane's layer, in the order recorded.
    pub ops: Vec<OpTime>,
}

fn median(mut v: Vec<f64>) -> f64 {
    if v.is_empty() {
        return 0.0;
    }
    v.sort_by(f64::total_cmp);
    v[v.len() / 2]
}

impl OpProfile {
    /// The segment totals of each (layer, lane): `(layer, lane, rows, kind, attention ms,
    /// shared ms)`, in pass order.
    pub fn segments(&self) -> Vec<(usize, usize, usize, LayerKind, f64, f64)> {
        let mut out: Vec<(usize, usize, usize, LayerKind, f64, f64)> = Vec::new();
        for o in &self.ops {
            let i = match out.iter().position(|s| s.0 == o.layer && s.1 == o.lane) {
                Some(i) => i,
                None => {
                    out.push((o.layer, o.lane, o.rows, o.kind, 0.0, 0.0));
                    out.len() - 1
                }
            };
            match o.segment {
                Segment::Attention => out[i].4 += o.ms,
                Segment::Shared => out[i].5 += o.ms,
            }
        }
        out
    }

    /// Per layer kind and lane size: each op's median over the layers and lanes, and the
    /// medians of the segment totals.
    pub fn summary(&self) -> Vec<KindSummary> {
        let segs = self.segments();
        let mut groups: Vec<(LayerKind, usize)> = segs.iter().map(|s| (s.3, s.2)).collect();
        groups.sort();
        groups.dedup();
        groups
            .into_iter()
            .map(|(kind, rows)| {
                // Per (layer, lane): each op's total (an op may run several times in a layer,
                // e.g. per block of rows), in the order the ops first ran.
                let mut names: Vec<(Segment, &'static str)> = Vec::new();
                let mut samples: Vec<((usize, usize), Vec<f64>)> = Vec::new();
                for o in self.ops.iter().filter(|o| o.kind == kind && o.rows == rows) {
                    let n = match names.iter().position(|x| *x == (o.segment, o.op)) {
                        Some(n) => n,
                        None => {
                            names.push((o.segment, o.op));
                            names.len() - 1
                        }
                    };
                    let i = match samples.iter().position(|x| x.0 == (o.layer, o.lane)) {
                        Some(i) => i,
                        None => {
                            samples.push(((o.layer, o.lane), Vec::new()));
                            samples.len() - 1
                        }
                    };
                    let t = &mut samples[i].1;
                    if t.len() <= n {
                        t.resize(n + 1, 0.0);
                    }
                    t[n] += o.ms;
                }
                let ops = names
                    .iter()
                    .enumerate()
                    .map(|(n, &(segment, op))| OpMedian {
                        segment,
                        op,
                        ms: median(
                            samples
                                .iter()
                                .map(|s| s.1.get(n).copied().unwrap_or(0.0))
                                .collect(),
                        ),
                    })
                    .collect();
                let s: Vec<_> = segs.iter().filter(|s| s.3 == kind && s.2 == rows).collect();
                let mut layers: Vec<usize> = s.iter().map(|s| s.0).collect();
                layers.sort_unstable();
                layers.dedup();
                KindSummary {
                    kind,
                    rows,
                    layers: layers.len(),
                    samples: s.len(),
                    ops,
                    attention_ms: median(s.iter().map(|s| s.4).collect()),
                    shared_ms: median(s.iter().map(|s| s.5).collect()),
                }
            })
            .collect()
    }

    /// Per lane, the medians over the MoE layers of the attention and shared segments: what
    /// the `PIPE` line prints as "GPU attention" and "shared", from these events.
    pub fn moe_medians(&self) -> Vec<(f64, f64)> {
        let segs = self.segments();
        (0..self.rows.len())
            .map(|x| {
                let s: Vec<_> = segs.iter().filter(|s| s.1 == x && s.3.is_moe()).collect();
                (
                    median(s.iter().map(|s| s.4).collect()),
                    median(s.iter().map(|s| s.5).collect()),
                )
            })
            .collect()
    }

    /// The `OPS` lines: per layer kind and lane size, each op's median GPU time over the
    /// pass's layers of that kind and its lanes, and its share of the lane's time in that kind
    /// of layer (attention and shared expert).
    pub fn table(&self) -> String {
        let rows: Vec<String> = self.rows.iter().map(|r| r.to_string()).collect();
        let mut out = format!(
            "OPS prefill pass, lanes of {} rows: per op, the median GPU time (ms) over the pass's \
             layers of a kind and its lanes, and its share of the lane's time in that kind of \
             layer\n",
            rows.join(" + ")
        );
        for k in self.summary() {
            let lane = k.attention_ms + k.shared_ms;
            out.push_str(&format!(
                "OPS {}, {} rows ({} layers x lanes = {} samples): attention {:.3} ms{}\n",
                k.kind.label(),
                k.rows,
                k.layers,
                k.samples,
                k.attention_ms,
                if k.kind.is_moe() {
                    format!(", shared expert {:.3} ms", k.shared_ms)
                } else {
                    String::new()
                }
            ));
            for seg in [Segment::Attention, Segment::Shared] {
                let ops: Vec<&OpMedian> = k.ops.iter().filter(|o| o.segment == seg).collect();
                if ops.is_empty() {
                    continue;
                }
                for o in &ops {
                    out.push_str(&format!(
                        "OPS   {:<22} {:>9.4} {:>6.1}%\n",
                        o.op,
                        o.ms,
                        100.0 * o.ms / lane.max(1e-12)
                    ));
                }
                let (name, total) = match seg {
                    Segment::Attention => ("attention", k.attention_ms),
                    Segment::Shared => ("shared expert", k.shared_ms),
                };
                out.push_str(&format!(
                    "OPS   {:<22} {:>9.4} {:>6.1}%   (sum of the op medians {:.4})\n",
                    format!("= {name}"),
                    total,
                    100.0 * total / lane.max(1e-12),
                    ops.iter().map(|o| o.ms).sum::<f64>()
                ));
            }
        }
        let moe = self.moe_medians();
        if !moe.is_empty() && self.ops.iter().any(|o| o.kind.is_moe()) {
            let a: Vec<String> = moe.iter().map(|m| format!("{:.2}", m.0)).collect();
            let s: Vec<String> = moe.iter().map(|m| format!("{:.2}", m.1)).collect();
            out.push_str(&format!(
                "OPS all MoE layers, median per lane: attention {}, shared {} (the PIPE line's \
                 \"GPU attention\" and \"shared\")\n",
                a.join(" + "),
                s.join(" + ")
            ));
        }
        out
    }
}

/// Where the ops being recorded go.
#[derive(Clone, Copy)]
struct Cur {
    layer: usize,
    lane: usize,
    rows: usize,
    kind: LayerKind,
    segment: Segment,
    /// The event the next op starts from.
    from: usize,
}

/// One op as recorded: where, and its start and end events.
#[derive(Clone, Copy)]
struct Rec {
    cur: Cur,
    op: &'static str,
    to: usize,
}

/// The recorder (the forward holds one while the op profile is on).
pub(crate) struct OpTrace {
    pub print: bool,
    /// Recording the current pass (a prefill).
    active: bool,
    events: Vec<Event>,
    used: usize,
    cur: Option<Cur>,
    recs: Vec<Rec>,
    rows: Vec<usize>,
    pub last: Option<OpProfile>,
}

impl OpTrace {
    pub fn new(print: bool) -> OpTrace {
        OpTrace {
            print,
            active: false,
            events: Vec::new(),
            used: 0,
            cur: None,
            recs: Vec::new(),
            rows: Vec::new(),
            last: None,
        }
    }

    fn record(&mut self, stream: &Stream) -> Result<usize> {
        if self.used == self.events.len() {
            self.events.push(Event::new()?);
        }
        self.events[self.used].record(stream)?;
        self.used += 1;
        Ok(self.used - 1)
    }

    /// A pass of lanes of `rows` rows starts: record it if `on` (a prefill pass).
    pub fn begin_pass(&mut self, on: bool, rows: Vec<usize>) {
        self.active = on;
        self.used = 0;
        self.cur = None;
        self.recs.clear();
        self.rows = rows;
    }

    /// A lane's segment of a layer starts here on the stream.
    pub fn begin(
        &mut self,
        stream: &Stream,
        layer: usize,
        lane: usize,
        kind: LayerKind,
        segment: Segment,
    ) -> Result<()> {
        if !self.active {
            return Ok(());
        }
        let rows = self.rows.get(lane).copied().unwrap_or(0);
        let from = self.record(stream)?;
        self.cur = Some(Cur {
            layer,
            lane,
            rows,
            kind,
            segment,
            from,
        });
        Ok(())
    }

    /// Op `op` of the current segment ends here on the stream (nothing outside a segment).
    pub fn op(&mut self, stream: &Stream, op: &'static str) -> Result<()> {
        let Some(cur) = self.cur.filter(|_| self.active) else {
            return Ok(());
        };
        let to = self.record(stream)?;
        self.recs.push(Rec { cur, op, to });
        self.cur = Some(Cur { from: to, ..cur });
        Ok(())
    }

    /// The current segment ends (ops until the next [`OpTrace::begin`] are not recorded).
    pub fn end(&mut self) {
        self.cur = None;
    }

    /// The pass ended: its profile from the events (waits for the stream), printed with
    /// `print`.
    pub fn finish(&mut self, stream: &Stream) -> Result<()> {
        if !std::mem::take(&mut self.active) {
            return Ok(());
        }
        self.cur = None;
        stream.synchronize()?;
        let mut ops = Vec::with_capacity(self.recs.len());
        for r in &self.recs {
            let ms = self.events[r.to].elapsed_ms_since(&self.events[r.cur.from])? as f64;
            ops.push(OpTime {
                layer: r.cur.layer,
                lane: r.cur.lane,
                rows: r.cur.rows,
                kind: r.cur.kind,
                segment: r.cur.segment,
                op: r.op,
                ms,
            });
        }
        let p = OpProfile {
            rows: std::mem::take(&mut self.rows),
            ops,
        };
        if self.print && !p.ops.is_empty() {
            eprint!("{}", p.table());
        }
        self.last = Some(p);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn t(
        layer: usize,
        lane: usize,
        kind: LayerKind,
        seg: Segment,
        op: &'static str,
        ms: f64,
    ) -> OpTime {
        OpTime {
            layer,
            lane,
            rows: 16,
            kind,
            segment: seg,
            op,
            ms,
        }
    }

    #[test]
    fn medians_group_by_kind_and_add_up_per_sample() {
        use LayerKind::*;
        use Segment::*;
        let mut ops = Vec::new();
        for (l, x, a, b) in [(3, 0, 1.0, 2.0), (3, 1, 3.0, 4.0), (4, 0, 5.0, 6.0)] {
            let kind = if l == 3 { DsaMoe } else { KdaMoe };
            // "a" runs twice in a layer (two blocks): its time is the sum.
            ops.push(t(l, x, kind, Attention, "a", a / 2.0));
            ops.push(t(l, x, kind, Attention, "a", a / 2.0));
            ops.push(t(l, x, kind, Attention, "router", b));
            ops.push(t(l, x, kind, Shared, "s", 0.5));
        }
        let p = OpProfile {
            rows: vec![16, 16],
            ops,
        };
        let s = p.summary();
        assert_eq!(s.len(), 2);
        let dsa = s.iter().find(|k| k.kind == DsaMoe).unwrap();
        assert_eq!((dsa.layers, dsa.samples), (1, 2));
        // Medians of two samples take the upper one; the segment totals are per sample.
        assert_eq!(dsa.ops[0].ms, 3.0);
        assert_eq!(dsa.attention_ms, 7.0);
        assert_eq!(dsa.shared_ms, 0.5);
        let kda = s.iter().find(|k| k.kind == KdaMoe).unwrap();
        assert_eq!(kda.attention_ms, 11.0);
        // Lane 0 over layers 3 and 4: 3 and 11; lane 1: layer 3 only.
        assert_eq!(p.moe_medians(), vec![(11.0, 0.5), (7.0, 0.5)]);
        let table = p.table();
        assert!(table.contains("OPS DSA MoE layers, 16 rows (1 layers x lanes = 2 samples)"));
        assert!(table.lines().all(|l| l.starts_with("OPS")));
    }
}
