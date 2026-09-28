//! The shell's kernels against their CPU references (feature `cuda`; each test skips, with a
//! note, when no device is present):
//!
//! - the sampler at GLM-5.3-Flash's width (154,880 LM head rows, ids below 154,856): greedy,
//!   sampled and masked rows mixed in one call, each with a large logit on a padding row, equal to
//!   `sampling::select_pick` (the port of mimo26f-afd's `sample_check`);
//! - the mask kernel and the argmax, bit for bit;
//! - the wire kernels: the FP8 quantizer, the rank sum with the routed scale and the frame fill,
//!   byte for byte against `wire.rs` and `glm53f-wire`'s encoder and `CoordinatorSum`.
#![cfg(feature = "cuda")]

use glm53f_coordinator::gpu::{self, DeviceBuffer, Sampler};
use glm53f_coordinator::model::Pick;
use glm53f_coordinator::sampling::{apply_mask, greedy, select_pick, Mask, Sampling};
use glm53f_coordinator::wire::{quantize_hidden_batched, quantize_hidden_scales};
use glm53f_wire::{WireNaive, HIDDEN, HIDDEN_ROW_BYTES};

const LD: usize = 154_880;
const VOCAB: usize = 154_856;

fn have_gpu() -> bool {
    let n = gpu::device_count();
    if n == 0 {
        eprintln!("skipped: no CUDA device");
    }
    n > 0
}

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    fn f(&mut self) -> f32 {
        (self.next() >> 40) as f32 / 16_777_216.0
    }
    fn normal(&mut self) -> f32 {
        let (a, b) = (self.f().max(1e-7), self.f());
        (-2.0 * a.ln()).sqrt() * (std::f32::consts::TAU * b).cos()
    }
    fn pick<T: Copy>(&mut self, xs: &[T]) -> T {
        xs[(self.next() % xs.len() as u64) as usize]
    }
}

/// A logit row of one of five kinds (sample_check's): realistic, flat, tied, damaged, one-hot,
/// with a peak on a padding row that must never be picked.
fn logits(rng: &mut Rng, kind: usize) -> Vec<f32> {
    let mut l: Vec<f32> = match kind {
        1 => (0..LD).map(|_| 0.1 * rng.normal()).collect(),
        2 => (0..LD).map(|_| (2.0 * (3.0 * rng.normal() - 4.0)).round() * 0.5).collect(),
        4 => vec![-40.0; LD],
        _ => (0..LD).map(|_| 3.0 * rng.normal() - 5.0).collect(),
    };
    let peaks = 1 + (rng.next() % 20) as usize;
    for _ in 0..peaks {
        let i = (rng.next() % VOCAB as u64) as usize;
        l[i] = if kind == 4 { 40.0 } else { 10.0 + 15.0 * rng.f() };
        if kind == 2 {
            l[i] = l[i].round();
        }
    }
    if kind == 3 {
        for _ in 0..50 {
            let i = (rng.next() % VOCAB as u64) as usize;
            l[i] = if rng.next() % 2 == 0 { f32::NAN } else { f32::NEG_INFINITY };
        }
    }
    l[VOCAB + (rng.next() % (LD - VOCAB) as u64) as usize] = 60.0;
    l
}

/// A mask allowing about `keep` of the ids, always including one of the row's peaks.
fn mask(rng: &mut Rng, row: &[f32], keep: f64) -> Mask {
    let top = greedy(row, VOCAB) as u32;
    let allowed: Vec<u32> = (0..VOCAB as u32).filter(|&i| i == top || (rng.f() as f64) < keep).collect();
    Mask::from_allowed(VOCAB, allowed)
}

#[test]
fn the_sampler_matches_the_cpu_reference_at_glm_width() {
    if !have_gpu() {
        return;
    }
    let mut rng = Rng(0x5eed_1234_abcd_ef01);
    let mut sampler = Sampler::new().unwrap();
    let (mut rows_done, mut bad) = (0usize, Vec::new());
    for chunk in 0..4 {
        let m = 96;
        let mut x = Vec::with_capacity(m * LD);
        let mut picks = Vec::with_capacity(m);
        for r in 0..m {
            let row = logits(&mut rng, r % 5);
            let s = Sampling::new(rng.pick(&[0.2, 0.6, 1.0, 1.4, 2.0]), rng.pick(&[1.0, 0.95, 0.9, 0.5, 0.05]),
                rng.pick(&[0usize, 0, 5, 50, 400]), rng.pick(&[0.0, 0.0, 0.02, 0.3]), Some(rng.next()))
                .unwrap();
            // Rows cycle through greedy, sampled, masked greedy and masked sampled.
            let draw = if r % 4 == 1 || r % 4 == 3 { s.map(|s| (s, rng.next() % 4096)) } else { None };
            let keep = rng.pick(&[0.001, 0.3, 0.9]);
            let mask = if r % 4 >= 2 { Some(mask(&mut rng, &row, keep)) } else { None };
            picks.push(Pick { draw, mask });
            x.extend(row);
        }
        let dev = DeviceBuffer::from_slice(&x).unwrap();
        let got = sampler.select(dev.ptr(), LD, VOCAB, &picks).unwrap();
        for (r, p) in picks.iter().enumerate() {
            let want = select_pick(&x[r * LD..(r + 1) * LD], VOCAB, p);
            if got[r] != want || got[r] as usize >= VOCAB || p.mask.as_ref().is_some_and(|m| !m.allows(got[r] as usize)) {
                bad.push(format!("chunk {chunk} row {r}: gpu {} cpu {want} ({:?})", got[r], p.draw.map(|d| d.0)));
            }
        }
        rows_done += m;
    }
    assert!(bad.is_empty(), "{} of {rows_done} rows differ: {bad:#?}", bad.len());
}

#[test]
fn the_mask_kernel_and_the_argmax_are_bitwise() {
    if !have_gpu() {
        return;
    }
    let mut rng = Rng(77);
    let rows = 12;
    let mut x = Vec::with_capacity(rows * LD);
    let mut picks = Vec::new();
    for r in 0..rows {
        let mut row = logits(&mut rng, r % 5);
        if r == 5 {
            row.iter_mut().for_each(|v| *v = f32::NAN); // all NaN: id 0
        }
        if r == 6 {
            row[..VOCAB].iter_mut().for_each(|v| *v = f32::NEG_INFINITY); // all -inf below the bound
        }
        if r == 7 {
            row[100] = 30.0;
            row[90_000] = 30.0; // a tie: the first index
        }
        let m = if r % 3 == 0 { Some(mask(&mut rng, &row, 0.5)) } else { None };
        picks.push(Pick { draw: None, mask: m });
        x.extend(row);
    }
    let dev = DeviceBuffer::from_slice(&x).unwrap();
    let got = Sampler::new().unwrap().select(dev.ptr(), LD, VOCAB, &picks).unwrap();
    // The masked logits on the device, row by row, against apply_mask.
    let mut back = vec![0f32; rows * LD];
    dev.download(&mut back).unwrap();
    for r in 0..rows {
        let mut want = x[r * LD..(r + 1) * LD].to_vec();
        if let Some(m) = &picks[r].mask {
            apply_mask(&mut want, m);
        }
        let same = want.iter().zip(&back[r * LD..(r + 1) * LD]).all(|(a, b)| a.to_bits() == b.to_bits());
        assert!(same, "row {r}: masked logits differ");
        assert_eq!(got[r] as usize, greedy(&want, VOCAB), "row {r}: argmax");
    }
    assert_eq!(got[5], 0);
    assert_eq!(got[6], 0);
    assert_eq!(got[7], 100);
}

#[test]
fn the_wire_quantizer_matches_the_host() {
    if !have_gpu() {
        return;
    }
    let mut rng = Rng(5);
    let rows = 5;
    let mut hidden: Vec<f32> = (0..rows * HIDDEN).map(|_| rng.normal() * 3.0).collect();
    hidden[..HIDDEN].iter_mut().for_each(|v| *v = 0.0); // an all-zero row
    hidden[HIDDEN + 7] = 1e6; // a block far past 448
    hidden[2 * HIDDEN + 33] = 1e-30; // subnormal territory
    let want = quantize_hidden_batched(&hidden).unwrap();
    let (want_scales, want_inv) = quantize_hidden_scales(&hidden).unwrap();
    let x = DeviceBuffer::from_slice(&hidden).unwrap();
    let blocks = hidden.len() / 32;
    let scales = DeviceBuffer::alloc(blocks).unwrap();
    let inv = DeviceBuffer::alloc(blocks * 4).unwrap();
    let payload = DeviceBuffer::alloc(hidden.len()).unwrap();
    unsafe {
        gpu::check(gpu::glm53f_coord_quant_scales(x.ptr(), blocks as i64, scales.ptr(), inv.ptr(), core::ptr::null_mut()), "scales")
            .unwrap();
        gpu::check(gpu::glm53f_coord_quantize_hidden(x.ptr(), inv.ptr(), payload.ptr(), hidden.len() as i64, core::ptr::null_mut()),
            "quantize").unwrap();
    }
    let (mut s, mut i, mut p) = (vec![0u8; blocks], vec![0f32; blocks], vec![0u8; hidden.len()]);
    scales.download(&mut s).unwrap();
    inv.download(&mut i).unwrap();
    payload.download(&mut p).unwrap();
    assert_eq!(s, want_scales);
    assert!(i.iter().zip(&want_inv).all(|(a, b)| a.to_bits() == b.to_bits()));
    for (t, row) in want.iter().enumerate() {
        assert_eq!(&p[t * HIDDEN..(t + 1) * HIDDEN], &row.payload[..], "row {t} payload");
        assert_eq!(&s[t * HIDDEN / 32..(t + 1) * HIDDEN / 32], &row.scales[..], "row {t} scales");
    }
}

#[test]
fn the_rank_sum_matches_the_coordinator_sum_with_the_routed_scale() {
    if !have_gpu() {
        return;
    }
    use glm53f_wire::{bf16::f32_to_bf16, CoordinatorSum, ReturnFrame, ReturnRow};
    let mut rng = Rng(9);
    let tokens = 3;
    let planes: Vec<Vec<u16>> =
        (0..4).map(|_| (0..tokens * HIDDEN).map(|_| f32_to_bf16(rng.normal() * 4.0, WireNaive::NONE)).collect()).collect();
    // The host sum: CoordinatorSum over four return frames, in rank order.
    let mut sum = CoordinatorSum::new(tokens, HIDDEN, WireNaive::NONE);
    for (r, p) in planes.iter().enumerate() {
        let f = ReturnFrame {
            request_id: 1,
            placement_version: 1,
            layer_id: 3,
            executor_id: r as u64,
            token_position: 0,
            status: glm53f_wire::Status::Ok,
            flags: glm53f_wire::FLAG_RETURN_REQUIRED,
            route_count: 8,
            seq: 0,
            rows: p.chunks_exact(HIDDEN).map(|c| ReturnRow { codes: c.to_vec() }).collect(),
        };
        sum.accumulate(&f).unwrap();
    }
    let host = sum.result().unwrap().to_vec();
    let bufs: Vec<DeviceBuffer> = planes.iter().map(|p| DeviceBuffer::from_slice(p).unwrap()).collect();
    let out = DeviceBuffer::alloc(tokens * HIDDEN * 4).unwrap();
    for scale in [1.0f32, 2.5] {
        unsafe {
            gpu::check(gpu::glm53f_coord_rank_sum_bf16(bufs[0].ptr(), bufs[1].ptr(), bufs[2].ptr(), bufs[3].ptr(), out.ptr(),
                (tokens * HIDDEN) as i64, scale, core::ptr::null_mut()), "rank sum").unwrap();
        }
        let mut got = vec![0f32; tokens * HIDDEN];
        out.download(&mut got).unwrap();
        let same = got.iter().zip(&host).all(|(g, h)| g.to_bits() == (h * scale).to_bits());
        assert!(same, "scale {scale}: the device sum differs from the host's");
    }
}

#[test]
fn the_frame_fill_writes_the_host_encoders_bytes() {
    if !have_gpu() {
        return;
    }
    use glm53f_wire::frame::{encode_request_desc_into, encode_request_meta_into};
    let mut rng = Rng(21);
    let (t, topk) = (5usize, 8usize);
    let hidden: Vec<f32> = (0..t * HIDDEN).map(|_| rng.normal()).collect();
    let rows = quantize_hidden_batched(&hidden).unwrap();
    let idx: Vec<i32> = (0..t * topk).map(|_| (rng.next() % 288) as i32).collect();
    let wts: Vec<f32> = (0..t * topk).map(|_| rng.f()).collect();
    let routes: Vec<(u32, f32)> = idx.iter().zip(&wts).map(|(&e, &w)| (e as u32, w)).collect();
    // The host encoder: descriptors and routes, then the hidden rows copied in.
    let len = t * (40 + topk * 12 + HIDDEN_ROW_BYTES);
    let mut want = vec![0u8; len];
    let (hw, hidden_off, body_len) = encode_request_meta_into(&mut want, 7, 11, 0, 0, &routes, topk, WireNaive::NONE).unwrap();
    assert_eq!(body_len, len);
    for (r, row) in rows.iter().enumerate() {
        let o = hidden_off + r * HIDDEN_ROW_BYTES;
        want[o..o + HIDDEN].copy_from_slice(&row.payload);
        want[o + HIDDEN..o + HIDDEN_ROW_BYTES].copy_from_slice(&row.scales);
    }
    // The device fill: descriptors from the host, routes and hidden rows from the kernel.
    let mut got = vec![0u8; len];
    let (hg, routes_off, hidden_off2, _) = encode_request_desc_into(&mut got, 7, 11, 0, 0, t, topk, WireNaive::NONE).unwrap();
    assert_eq!((hg, hidden_off2), (hw, hidden_off));
    let payload: Vec<u8> = rows.iter().flat_map(|r| r.payload.clone()).collect();
    let scales: Vec<u8> = rows.iter().flat_map(|r| r.scales.clone()).collect();
    let (di, dw) = (DeviceBuffer::from_slice(&idx).unwrap(), DeviceBuffer::from_slice(&wts).unwrap());
    let (dp, ds) = (DeviceBuffer::from_slice(&payload).unwrap(), DeviceBuffer::from_slice(&scales).unwrap());
    let droutes = DeviceBuffer::alloc(t * topk * 12).unwrap();
    let dhidden = DeviceBuffer::alloc(t * HIDDEN_ROW_BYTES).unwrap();
    unsafe {
        gpu::check(gpu::glm53f_coord_frame_fill(di.ptr(), dw.ptr(), dp.ptr(), ds.ptr(), t as i32, topk as i32, HIDDEN as i32,
            droutes.ptr(), dhidden.ptr(), HIDDEN_ROW_BYTES as i32, core::ptr::null_mut()), "frame fill").unwrap();
    }
    droutes.download(&mut got[routes_off..routes_off + t * topk * 12]).unwrap();
    dhidden.download(&mut got[hidden_off..hidden_off + t * HIDDEN_ROW_BYTES]).unwrap();
    assert!(got == want, "the device-filled frame body differs from the host encoder's");
}

/// The host tier's arenas page-locked (as the engine runs them), captured and restored.
#[test]
fn the_host_tier_page_locks_its_arenas() {
    if !have_gpu() {
        return;
    }
    use glm53f_coordinator::{HostCache, HostTierConfig};
    let cfg = HostTierConfig::balanced(64 << 20, 64, 64 * 6_171, 8 << 20);
    assert!(cfg.pin);
    let hc = HostCache::new(cfg).expect("page-locked arenas");
    assert_eq!(hc.slots(), (cfg.pages, cfg.states));
    drop(hc);
}
