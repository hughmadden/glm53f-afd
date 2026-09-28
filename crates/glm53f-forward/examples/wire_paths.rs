//! The expert exchange's host-device transfer options on this GPU, on synthetic buffers of the
//! wire's sizes: the choices `RemoteExperts` (`src/remote.rs`) makes for the requests and the
//! returns.
//!
//! ```sh
//! cargo run --release -p glm53f-forward --features coordinator --example wire_paths
//! ```
//!
//! **Returns** of an exchange of R rows, four BF16 planes of R x 8,192 bytes (or four row slices
//! of R / 4 rows each), written into the output on the stream:
//!
//! - `pageable`: four uploads from pageable memory, then the rank sum (the host path);
//! - `page-locked DMA`: four copies from page-locked (registered) memory, then the rank sum;
//! - `in place`: the rank sum reading the four page-locked planes where they lie (mapped);
//! - row slices: `pageable` and `page-locked DMA` copies of the slices to their rows.
//!
//! **Requests** of R rows, R x 4,224 bytes of wire rows on the device:
//!
//! - `pageable download`: to pageable memory (the host path; the wire client then encodes the
//!   frame and copies it into the request body, not timed here);
//! - `DMA into the body` (P6): two 2-D copies into a page-locked body at the frame's row pitch;
//! - `frame fill` (P9): the kernel writing the route entries and the rows into the mapped body.
//!
//! Each is timed on the host (until the calls return: what the host thread spends), on the GPU
//! (events around the work) and to completion (from the first call to the stream's end), median
//! of `GLM53F_BENCH_REPS` (default 30). Every path's bytes are checked against the pageable one.
//! The numbers are this machine's (its PCIe link and a GPU shared with other work).

use core::ffi::{c_int, c_uint, c_void};
use std::time::Instant;

use glm53f_coordinator::gpu as cgpu;
use glm53f_forward::cuda;
use glm53f_forward::device::{check, DeviceBuffer, Event, Stream};
use glm53f_forward::shape::{HIDDEN, TOP_K};

const SCALES: usize = HIDDEN / 32;
const ROW: usize = HIDDEN + SCALES;

unsafe extern "C" {
    fn cudaHostRegister(ptr: *mut c_void, size: usize, flags: c_uint) -> i32;
    fn cudaHostUnregister(ptr: *mut c_void) -> i32;
    fn cudaMemcpy2DAsync(
        dst: *mut c_void,
        dpitch: usize,
        src: *const c_void,
        spitch: usize,
        width: usize,
        height: usize,
        kind: c_int,
        stream: *mut c_void,
    ) -> i32;
}

/// Page-aligned host memory, page-locked and mapped (`cudaHostRegisterMapped`), as
/// `RemoteExperts` registers the wire client's buffers.
struct Locked {
    ptr: *mut u8,
    len: usize,
    dev: *mut u8,
}

impl Locked {
    fn new(len: usize) -> Locked {
        let layout = std::alloc::Layout::from_size_align(len, 4096).unwrap();
        // SAFETY: a non-zero size and a valid alignment.
        let ptr = unsafe { std::alloc::alloc_zeroed(layout) };
        assert!(!ptr.is_null());
        // SAFETY: a live allocation of `len` bytes, unregistered before it is freed.
        check(
            unsafe { cudaHostRegister(ptr.cast(), len, 2) },
            "cudaHostRegister",
        )
        .unwrap();
        let mut dev = core::ptr::null_mut();
        check(
            unsafe { cuda::cudaHostGetDevicePointer(&mut dev, ptr.cast(), 0) },
            "cudaHostGetDevicePointer",
        )
        .unwrap();
        Locked {
            ptr,
            len,
            dev: dev.cast(),
        }
    }

    fn bytes(&self) -> &[u8] {
        // SAFETY: a live allocation of `len` bytes.
        unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
    }

    fn bytes_mut(&mut self) -> &mut [u8] {
        // SAFETY: as above, uniquely borrowed.
        unsafe { std::slice::from_raw_parts_mut(self.ptr, self.len) }
    }
}

impl Drop for Locked {
    fn drop(&mut self) {
        // SAFETY: registered and allocated in `new`.
        unsafe {
            cudaHostUnregister(self.ptr.cast());
            std::alloc::dealloc(
                self.ptr,
                std::alloc::Layout::from_size_align(self.len, 4096).unwrap(),
            );
        }
    }
}

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(f64::total_cmp);
    v[v.len() / 2]
}

/// (host, GPU, done) medians in milliseconds of `f` enqueueing its work on `stream`, from an
/// idle stream each time.
fn time(stream: &Stream, reps: usize, mut f: impl FnMut()) -> (f64, f64, f64) {
    let (e0, e1) = (Event::new().unwrap(), Event::new().unwrap());
    f();
    stream.synchronize().unwrap();
    let (mut host, mut gpu, mut done) = (Vec::new(), Vec::new(), Vec::new());
    for _ in 0..reps {
        stream.synchronize().unwrap();
        let t = Instant::now();
        e0.record(stream).unwrap();
        f();
        e1.record(stream).unwrap();
        host.push(t.elapsed().as_secs_f64() * 1e3);
        stream.synchronize().unwrap();
        done.push(t.elapsed().as_secs_f64() * 1e3);
        gpu.push(e1.elapsed_ms_since(&e0).unwrap() as f64);
    }
    (median(host), median(gpu), median(done))
}

fn h2d(dst: *mut u8, src: *const u8, n: usize, stream: &Stream) {
    // SAFETY: the callers pass buffers of at least `n` bytes.
    check(
        unsafe { cuda::cudaMemcpyAsync(dst.cast(), src.cast(), n, cuda::MEMCPY_H2D, stream.raw()) },
        "cudaMemcpyAsync",
    )
    .unwrap();
}

fn line(what: &str, rows: usize, mb: f64, (host, gpu, done): (f64, f64, f64)) {
    eprintln!(
        "{what:<34} {rows:>5} rows {mb:>7.2} MB: host {host:>7.3} ms, GPU {gpu:>7.3} ms, done {done:>7.3} ms ({:>5.1} GB/s to done)",
        mb / 1e3 / (done / 1e3)
    );
}

fn main() {
    if glm53f_forward::device::device_count() == 0 {
        eprintln!("no CUDA device");
        return;
    }
    let reps: usize = std::env::var("GLM53F_BENCH_REPS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(30);
    let stream = Stream::new().unwrap();
    let sizes = [1usize, 8, 64, 512, 2048, 4096];
    let max = *sizes.last().unwrap();
    let plane_max = max * HIDDEN * 2;

    // Returns: four planes, pageable and page-locked, with a pattern of BF16 values.
    let mut pageable: Vec<Vec<u8>> = (0..4).map(|_| vec![0u8; plane_max]).collect();
    let mut locked: Vec<Locked> = (0..4).map(|_| Locked::new(plane_max)).collect();
    for r in 0..4 {
        for (i, c) in pageable[r].chunks_exact_mut(2).enumerate() {
            let v = f32::from_bits(0x3f80_0000 + ((i * 7919 + r * 104_729) as u32 % 0x0040_0000));
            let b = ((v.to_bits() >> 16) as u16).to_le_bytes();
            c.copy_from_slice(&b);
        }
        locked[r].bytes_mut().copy_from_slice(&pageable[r]);
    }
    let planes = DeviceBuffer::alloc(4 * plane_max).unwrap();
    let sum = DeviceBuffer::alloc(max * HIDDEN * 4).unwrap();
    let out = DeviceBuffer::alloc(plane_max).unwrap();
    eprintln!("Returns ({reps} reps, median):");
    for &rows in &sizes {
        let plane = rows * HIDDEN * 2;
        let n = (rows * HIDDEN) as i64;
        let summed = |p: [*const u16; 4], st: &Stream| {
            check(
                unsafe {
                    cgpu::glm53f_coord_rank_sum_bf16(
                        p[0],
                        p[1],
                        p[2],
                        p[3],
                        sum.ptr(0),
                        n,
                        1.0,
                        st.raw(),
                    )
                },
                "rank sum",
            )
            .unwrap();
        };
        let staged: [*const u16; 4] =
            std::array::from_fn(|r| planes.byte_ptr(r * plane) as *const u16);
        let mb = 4.0 * plane as f64 / 1e6;
        let a = time(&stream, reps, || {
            for (r, p) in pageable.iter().enumerate() {
                h2d(planes.byte_ptr(r * plane), p.as_ptr(), plane, &stream);
            }
            summed(staged, &stream);
        });
        let want: Vec<f32> = sum.download(rows * HIDDEN).unwrap();
        line("four planes, pageable", rows, mb, a);
        let b = time(&stream, reps, || {
            for (r, p) in locked.iter().enumerate() {
                h2d(planes.byte_ptr(r * plane), p.ptr, plane, &stream);
            }
            summed(staged, &stream);
        });
        assert!(
            sum.download::<f32>(rows * HIDDEN).unwrap() == want,
            "page-locked DMA: other bits"
        );
        line("four planes, page-locked DMA", rows, mb, b);
        let c = time(&stream, reps, || {
            summed(
                std::array::from_fn(|r| locked[r].dev as *const u16),
                &stream,
            );
        });
        assert!(
            sum.download::<f32>(rows * HIDDEN).unwrap() == want,
            "in place: other bits"
        );
        line("four planes, in place (mapped)", rows, mb, c);
        if rows >= 4 {
            // Row slices: rank r's quarter of the rows, from the start of its buffer.
            let q = rows / 4;
            let slice = q * HIDDEN * 2;
            let mb = 4.0 * slice as f64 / 1e6;
            let a = time(&stream, reps, || {
                for (r, p) in pageable.iter().enumerate() {
                    h2d(out.byte_ptr(r * slice), p.as_ptr(), slice, &stream);
                }
            });
            let want: Vec<u16> = out.download(4 * q * HIDDEN).unwrap();
            line("row slices, pageable", rows, mb, a);
            let b = time(&stream, reps, || {
                for (r, p) in locked.iter().enumerate() {
                    h2d(out.byte_ptr(r * slice), p.ptr, slice, &stream);
                }
            });
            assert!(
                out.download::<u16>(4 * q * HIDDEN).unwrap() == want,
                "row slices: other bytes"
            );
            line("row slices, page-locked DMA", rows, mb, b);
        }
    }

    // Requests: wire rows on the device, and the routes.
    let q = DeviceBuffer::alloc(max * ROW).unwrap();
    let init: Vec<u8> = (0..max * ROW).map(|i| (i * 31 % 251) as u8).collect();
    q.upload(&init).unwrap();
    let ids: Vec<i32> = (0..max * TOP_K).map(|i| (i * 37 % 288) as i32).collect();
    let w: Vec<f32> = (0..max * TOP_K)
        .map(|i| 0.1 + (i % 8) as f32 * 0.25)
        .collect();
    let (di, dw) = (
        DeviceBuffer::from_slice(&ids).unwrap(),
        DeviceBuffer::from_slice(&w).unwrap(),
    );
    let mut host = vec![0u8; max * ROW];
    let body = Locked::new(max * (ROW + 40 + TOP_K * 12) + 4096);
    eprintln!("Requests ({reps} reps, median):");
    for &rows in &sizes {
        let (payload, scales) = (
            q.byte_ptr(0) as *const u8,
            q.byte_ptr(rows * HIDDEN) as *const u8,
        );
        let n = rows * ROW;
        let mb = n as f64 / 1e6;
        let a = time(&stream, reps, || {
            check(
                unsafe {
                    cuda::cudaMemcpyAsync(
                        host.as_mut_ptr().cast(),
                        payload.cast(),
                        n,
                        cuda::MEMCPY_D2H,
                        stream.raw(),
                    )
                },
                "download",
            )
            .unwrap();
        });
        line("pageable download (host path)", rows, mb, a);
        // The frame body: descriptors (not written here), the route entries, then the rows.
        let routes_off = rows * 40;
        let hidden_off = routes_off + rows * TOP_K * 12;
        let dst = body.ptr.wrapping_add(hidden_off);
        let b = time(&stream, reps, || unsafe {
            check(
                cudaMemcpy2DAsync(
                    dst.cast(),
                    ROW,
                    payload.cast(),
                    HIDDEN,
                    HIDDEN,
                    rows,
                    cuda::MEMCPY_D2H,
                    stream.raw().cast(),
                ),
                "payload",
            )
            .unwrap();
            check(
                cudaMemcpy2DAsync(
                    dst.add(HIDDEN).cast(),
                    ROW,
                    scales.cast(),
                    SCALES,
                    SCALES,
                    rows,
                    cuda::MEMCPY_D2H,
                    stream.raw().cast(),
                ),
                "scales",
            )
            .unwrap();
        });
        let rows_at = |b: &[u8]| -> Vec<u8> { b[hidden_off..hidden_off + n].to_vec() };
        let want: Vec<u8> = (0..rows)
            .flat_map(|t| {
                let mut r = host[t * HIDDEN..(t + 1) * HIDDEN].to_vec();
                r.extend_from_slice(
                    &host[rows * HIDDEN + t * SCALES..rows * HIDDEN + (t + 1) * SCALES],
                );
                r
            })
            .collect();
        assert!(
            rows_at(body.bytes()) == want,
            "DMA into the body: other bytes"
        );
        line("DMA into the body (P6)", rows, mb, b);
        let c = time(&stream, reps, || {
            check(
                unsafe {
                    cgpu::glm53f_coord_frame_fill(
                        di.ptr(0),
                        dw.ptr(0),
                        payload,
                        scales,
                        rows as i32,
                        TOP_K as i32,
                        HIDDEN as i32,
                        body.dev.wrapping_add(routes_off),
                        body.dev.wrapping_add(hidden_off),
                        ROW as i32,
                        stream.raw(),
                    )
                },
                "frame fill",
            )
            .unwrap();
        });
        assert!(rows_at(body.bytes()) == want, "frame fill: other row bytes");
        let routes: Vec<u8> = (0..rows * TOP_K)
            .flat_map(|i| {
                let mut e = ((i / TOP_K) as u32).to_le_bytes().to_vec();
                e.extend_from_slice(&(ids[i] as u32).to_le_bytes());
                e.extend_from_slice(&w[i].to_le_bytes());
                e
            })
            .collect();
        assert!(
            body.bytes()[routes_off..hidden_off] == routes[..],
            "frame fill: other route entries"
        );
        line(
            "frame fill into the mapped body (P9)",
            rows,
            mb + (rows * TOP_K * 12) as f64 / 1e6,
            c,
        );
    }

    // The host's side of a request over RDMA: the host path builds the frame from the downloaded
    // rows and copies its body into the request body; the device paths encode only the row
    // descriptors and route entries in place (the rows come by DMA or the fill).
    eprintln!("Host encode of a request ({reps} reps, median; no device work):");
    let mut staging = vec![0u8; max * (ROW + 40 + TOP_K * 12)];
    for &rows in &sizes {
        let routes: Vec<(u32, f32)> = (0..rows * TOP_K).map(|i| (ids[i] as u32, w[i])).collect();
        let (p, s) = host[..rows * ROW].split_at(rows * HIDDEN);
        let host_path = || {
            use glm53f_wire::frame::{HiddenRow, RequestFrame, RouteEntry, RowDescriptor};
            use glm53f_wire::SourceKind;
            let frame = RequestFrame {
                request_id: 1,
                placement_version: 1,
                layer_id: 3,
                executor_id: 0,
                source_kind: SourceKind::Decode,
                token_position: 0,
                flags: 0,
                seq: 0,
                rows: (0..rows)
                    .map(|t| RowDescriptor {
                        row_id: t as u64,
                        source_kind: SourceKind::Decode,
                        source_request_id: 1,
                        token_position: t as u64,
                        route_offset: (t * TOP_K) as u32,
                        route_count: TOP_K as u32,
                    })
                    .collect(),
                routes: routes
                    .iter()
                    .enumerate()
                    .map(|(i, &(expert_id, gate_weight))| RouteEntry {
                        row_index: (i / TOP_K) as u32,
                        expert_id,
                        gate_weight,
                    })
                    .collect(),
                hidden_rows: (0..rows)
                    .map(|t| HiddenRow {
                        payload: p[t * HIDDEN..(t + 1) * HIDDEN].to_vec(),
                        scales: s[t * SCALES..(t + 1) * SCALES].to_vec(),
                    })
                    .collect(),
            };
            glm53f_wire::frame::encode_request_seq(&frame, 0, glm53f_wire::WireNaive::NONE).unwrap()
        };
        let mut th = Vec::new();
        let mut tm = Vec::new();
        for _ in 0..reps {
            let t = Instant::now();
            let bytes = host_path();
            staging[..bytes.len() - 128].copy_from_slice(&bytes[128..]);
            th.push(t.elapsed().as_secs_f64() * 1e3);
            let t = Instant::now();
            glm53f_wire::frame::encode_request_meta_into(
                &mut staging,
                1,
                3,
                0,
                0,
                &routes,
                TOP_K,
                glm53f_wire::WireNaive::NONE,
            )
            .unwrap();
            tm.push(t.elapsed().as_secs_f64() * 1e3);
        }
        eprintln!(
            "{rows:>5} rows: host path (rows to HiddenRows, frame encode, copy into the body) {:>7.3} ms; in place (descriptors and route entries only) {:>7.3} ms",
            median(th),
            median(tm)
        );
    }
    eprintln!("every path's bytes equal the pageable path's");
}
