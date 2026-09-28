//! One request through the rank: routes, the expert kernel, the compact return.
//!
//! Ported from mimo26f-afd v1.2.0 `crates/mimo26-spark/src/serve.rs`: the
//! owned path (`serve_return`, a decoded request to an un-stamped return
//! frame), the in-place return (`serve_into`, the BF16 rows written straight
//! into the transport's registered send buffer) and the in-place receive
//! (`serve_view`, the hidden rows read where the NIC landed them). The kernel
//! is behind [`ExpertKernel`] instead of MiMo's B1 FFI, so the whole path runs
//! on the CPU backend in tests.

use std::time::Instant;

use glm53f_wire::frame::RequestView;
use glm53f_wire::RequestFrame;

use crate::consts::{HIDDEN, MTP_LAYER, TOPK};
use crate::fp8::SCALES_PER_ROW;
use crate::kernel::{ExpertKernel, FfnStats, Rows};
use crate::route::{request_routes, view_routes};
use crate::wire;

/// Per-request serve timings (ms). The daemon composes these with its receive,
/// load and send timings into one line under `GLM53F_RANK_TRACE=1`.
#[derive(Default, Clone, Copy, Debug)]
pub struct Timings {
    /// Reading the route table.
    pub plan_ms: f64,
    /// The kernel call (uploads, GPU, download).
    pub ffn_ms: f64,
    /// Kept for the timing line's shape; the kernel reduces on the device.
    pub reduce_ms: f64,
}

/// Prepared layers by model layer id (3..=44, and 45 for MTP experts).
pub struct Layers<L> {
    slots: Vec<Option<L>>,
}

impl<L> Default for Layers<L> {
    fn default() -> Self {
        Self { slots: (0..=MTP_LAYER).map(|_| None).collect() }
    }
}

impl<L> Layers<L> {
    pub fn get(&self, layer: u32) -> Option<&L> {
        self.slots.get(layer as usize).and_then(|s| s.as_ref())
    }

    pub fn insert(&mut self, layer: u32, l: L) {
        self.slots[layer as usize] = Some(l);
    }

    pub fn len(&self) -> usize {
        self.slots.iter().filter(|s| s.is_some()).count()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// The return frame's metadata for `request` (no rows): what
/// `wire::rank_bf16_to_return_frame` stamps, for a frame assembled in place.
pub fn return_meta(request: &RequestFrame) -> glm53f_wire::ReturnFrame {
    glm53f_wire::ReturnFrame {
        request_id: request.request_id,
        placement_version: request.placement_version,
        layer_id: request.layer_id,
        executor_id: request.executor_id,
        token_position: request.token_position,
        status: glm53f_wire::layout::Status::Ok,
        flags: glm53f_wire::frame::FLAG_RETURN_REQUIRED,
        route_count: TOPK,
        seq: 0,
        rows: Vec::new(),
    }
}

/// [`return_meta`] for a request validated in place.
pub fn return_meta_view(v: &RequestView<'_>) -> glm53f_wire::ReturnFrame {
    glm53f_wire::ReturnFrame {
        request_id: v.request_id,
        placement_version: v.placement_version,
        layer_id: v.layer_id,
        executor_id: v.executor_id,
        token_position: v.token_position,
        status: glm53f_wire::layout::Status::Ok,
        flags: glm53f_wire::frame::FLAG_RETURN_REQUIRED,
        route_count: TOPK,
        seq: 0,
        rows: Vec::new(),
    }
}

fn trace_line(layer: u32, rows: usize, t: &Timings, st: &FfnStats, how: &str) {
    if crate::timeline::trace() {
        eprintln!(
            "ffn layer={layer} rows={rows} groups={} routes={:.3} upload={:.3} gpu={:.3} [plan={:.3} gate/up={:.3} epi={:.3} down={:.3} reduce={:.3}] tail={:.3} ms ({how})",
            st.groups, t.plan_ms, st.upload_ms, st.gpu_ms, st.phase_ms[0], st.phase_ms[1], st.phase_ms[2], st.phase_ms[3],
            st.phase_ms[4], st.tail_ms
        );
    }
}

/// Serve a decoded request, writing the rank's BF16 partial rows into `out`
/// (`[rows * 4096]`, for example the transport's send buffer after the
/// header: no return-row copies).
pub fn serve_into<K: ExpertKernel>(
    request: &RequestFrame,
    kernel: &mut K,
    layer: &K::Layer,
    timings: &mut Timings,
    out: &mut [u16],
) -> Result<FfnStats, String> {
    let rows = request.rows.len();
    let t = Instant::now();
    let (ids, weights) = request_routes(request)?;
    let mut payload = Vec::with_capacity(rows * HIDDEN);
    let mut scales = Vec::with_capacity(rows * SCALES_PER_ROW);
    for h in &request.hidden_rows {
        if h.payload.len() != HIDDEN || h.scales.len() != SCALES_PER_ROW {
            return Err("serve: hidden row is not 4,096 E4M3 + 128 scales".into());
        }
        payload.extend_from_slice(&h.payload);
        scales.extend_from_slice(&h.scales);
    }
    timings.plan_ms = t.elapsed().as_secs_f64() * 1e3;
    let t = Instant::now();
    let st = kernel.ffn(layer, Rows::separate(&payload, &scales, rows)?, &ids, &weights, out)?;
    timings.ffn_ms = t.elapsed().as_secs_f64() * 1e3;
    timings.reduce_ms = 0.0;
    trace_line(request.layer_id, rows, timings, &st, "in place");
    Ok(st)
}

/// [`serve_into`] for a request validated in place (zero-copy receive): the
/// routes are read from the frame's entries and the hidden rows go to the
/// kernel from where the NIC landed them.
pub fn serve_view<K: ExpertKernel>(
    view: &RequestView<'_>,
    kernel: &mut K,
    layer: &K::Layer,
    timings: &mut Timings,
    out: &mut [u16],
) -> Result<FfnStats, String> {
    if view.row_stride != HIDDEN + SCALES_PER_ROW {
        return Err(format!("serve: hidden row stride {} is not 4,096 E4M3 + 128 scales", view.row_stride));
    }
    let t = Instant::now();
    let (ids, weights) = view_routes(view)?;
    timings.plan_ms = t.elapsed().as_secs_f64() * 1e3;
    let t = Instant::now();
    let rows = Rows::interleaved(view.hidden(), view.row_stride, view.rows)?;
    let st = kernel.ffn(layer, rows, &ids, &weights, out)?;
    timings.ffn_ms = t.elapsed().as_secs_f64() * 1e3;
    timings.reduce_ms = 0.0;
    trace_line(view.layer_id, view.rows, timings, &st, "zero-copy");
    Ok(st)
}

/// Serve a decoded request to an un-stamped compact return frame (the server
/// stamps the L4 sequence before encoding).
pub fn serve_return<K: ExpertKernel>(
    request: &RequestFrame,
    kernel: &mut K,
    layer: &K::Layer,
    timings: &mut Timings,
) -> Result<glm53f_wire::ReturnFrame, String> {
    let rows = request.rows.len();
    let mut bf16 = vec![0u16; rows * HIDDEN];
    serve_into(request, kernel, layer, timings, &mut bf16)?;
    wire::rank_bf16_to_return_frame(
        &bf16,
        rows,
        request.request_id,
        request.placement_version,
        request.layer_id,
        request.executor_id,
        request.token_position,
    )
    .map_err(|e| e.to_string())
}

/// Serve one request end to end, encoded (no L4 sequence stamp).
pub fn serve<K: ExpertKernel>(
    request: &RequestFrame,
    kernel: &mut K,
    layer: &K::Layer,
    naive: glm53f_wire::WireNaive,
) -> Result<Vec<u8>, String> {
    let frame = serve_return(request, kernel, layer, &mut Timings::default())?;
    glm53f_wire::frame::encode_return(&frame, naive).map_err(|e| e.to_string())
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::kernel::check_routes;
    use glm53f_wire::frame::{decode_frame, Frame, HiddenRow, RouteEntry, RowDescriptor};
    use glm53f_wire::{SourceKind, WireNaive};

    /// A stand-in kernel for plumbing tests: row `i`'s output is
    /// `sum_slot w * (expert + 1) * x[i]`, so every id, weight and row must
    /// reach the right place.
    #[derive(Default)]
    pub(crate) struct SumKernel {
        pub calls: usize,
    }

    impl ExpertKernel for SumKernel {
        type Layer = ();
        fn prepare_layer(&mut self, _image: &[u8]) -> Result<(), String> {
            Ok(())
        }
        fn ffn(&mut self, _l: &(), rows: Rows<'_>, ids: &[i32], w: &[f32], out: &mut [u16]) -> Result<FfnStats, String> {
            rows.check()?;
            check_routes(ids, w, rows.rows)?;
            self.calls += 1;
            let mut x = vec![0f32; HIDDEN];
            for i in 0..rows.rows {
                crate::fp8::decode_row(rows.payload(i), rows.scales(i), &mut x)?;
                let k: f32 = (0..TOPK).map(|s| w[i * TOPK + s] * (ids[i * TOPK + s] + 1) as f32).sum();
                for h in 0..HIDDEN {
                    out[i * HIDDEN + h] = glm53f_wire::bf16::f32_to_bf16_rne(k * x[h]);
                }
            }
            Ok(FfnStats::default())
        }
    }

    pub(crate) fn request(id: u64, layer: u32, rows: usize) -> RequestFrame {
        let (payload, scales) = crate::testkit::wire_rows(id, rows);
        let (ids, w) = crate::testkit::routes(id + 1, rows, 0);
        RequestFrame {
            request_id: id,
            placement_version: 1,
            layer_id: layer,
            executor_id: 2,
            source_kind: SourceKind::Decode,
            token_position: 9,
            flags: 0,
            seq: 0,
            rows: (0..rows)
                .map(|i| RowDescriptor {
                    row_id: i as u64,
                    source_kind: SourceKind::Decode,
                    source_request_id: id,
                    token_position: 9 + i as u64,
                    route_offset: (i * TOPK) as u32,
                    route_count: TOPK as u32,
                })
                .collect(),
            routes: (0..rows * TOPK)
                .map(|r| RouteEntry { row_index: (r / TOPK) as u32, expert_id: ids[r] as u32, gate_weight: w[r] })
                .collect(),
            hidden_rows: (0..rows)
                .map(|i| HiddenRow {
                    payload: payload[i * HIDDEN..(i + 1) * HIDDEN].to_vec(),
                    scales: scales[i * SCALES_PER_ROW..(i + 1) * SCALES_PER_ROW].to_vec(),
                })
                .collect(),
        }
    }

    #[test]
    fn serve_round_trips_a_request() {
        let req = request(7, 5, 3);
        let mut k = SumKernel::default();
        let bytes = serve(&req, &mut k, &(), WireNaive::NONE).expect("serve");
        assert_eq!(bytes.len(), 128 + 3 * 8192, "3 rows => 3 x 8,192 B return");
        let Frame::Return(ret) = decode_frame(&bytes, WireNaive::NONE).expect("decode") else {
            panic!("expected a return frame");
        };
        assert_eq!((ret.request_id, ret.layer_id, ret.executor_id, ret.route_count), (7, 5, 2, 8));
        let (ids, w) = request_routes(&req).unwrap();
        let mut want = vec![0u16; 3 * HIDDEN];
        let payload: Vec<u8> = req.hidden_rows.iter().flat_map(|h| h.payload.clone()).collect();
        let scales: Vec<u8> = req.hidden_rows.iter().flat_map(|h| h.scales.clone()).collect();
        SumKernel::default().ffn(&(), Rows::separate(&payload, &scales, 3).unwrap(), &ids, &w, &mut want).unwrap();
        for (i, row) in ret.rows.iter().enumerate() {
            assert_eq!(row.codes, want[i * HIDDEN..(i + 1) * HIDDEN], "row {i}");
        }
    }

    #[test]
    fn the_zero_copy_view_serves_the_same_rows() {
        let req = request(11, 3, 4);
        let bytes = glm53f_wire::frame::encode_request(&req, WireNaive::NONE).unwrap();
        let view = RequestView::parse(&bytes, WireNaive::NONE).unwrap();
        let mut k = SumKernel::default();
        let (mut a, mut b) = (vec![0u16; 4 * HIDDEN], vec![1u16; 4 * HIDDEN]);
        serve_view(&view, &mut k, &(), &mut Timings::default(), &mut a).unwrap();
        serve_into(&req, &mut k, &(), &mut Timings::default(), &mut b).unwrap();
        assert_eq!(a, b);
    }

    #[test]
    fn malformed_route_tables_are_refused() {
        let mut req = request(13, 3, 2);
        req.routes[9].row_index = 0; // a row-1 route claims row 0
        assert!(serve(&req, &mut SumKernel::default(), &(), WireNaive::NONE).is_err());
        let mut req = request(13, 3, 2);
        req.rows[1].route_count = 7;
        assert!(serve(&req, &mut SumKernel::default(), &(), WireNaive::NONE).is_err());
        let mut req = request(13, 3, 2);
        req.routes[3].expert_id = 288;
        assert!(serve(&req, &mut SumKernel::default(), &(), WireNaive::NONE).is_err());
    }
}
