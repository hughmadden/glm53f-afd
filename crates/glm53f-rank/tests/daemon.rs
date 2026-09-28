//! The daemon binary end to end over TCP: a rank directory with one synthetic
//! layer image, `glm53f-rank serve` on loopback, requests from a coordinator
//! stand-in with the L4 ladder, returns checked against the CPU reference. A
//! request for a layer the rank does not hold gets an error return and the
//! connection closes; the daemon keeps listening.
//!
//! Heavy (a 0.91 GB image is written, hashed at boot and served): release
//! builds only. With `--features cuda` the daemon serves on the GPU.

use std::io::{BufRead, BufReader};
use std::process::{Child, Command, Stdio};

use glm53f_rank::consts::{HIDDEN, TOPK};
use glm53f_rank::kernel::{CpuKernel, ExpertKernel, Rows};
use glm53f_rank::manifest::{file_name, LayerEntry, Manifest};
use glm53f_rank::testkit;
use glm53f_rank::transport::{ByteTransport, TcpTransport};
use glm53f_wire::frame::{HiddenRow, RequestFrame, RouteEntry, RowDescriptor};
use glm53f_wire::l4::{StreamReceiver, StreamSender};
use glm53f_wire::layout::Status;
use glm53f_wire::{bf16::bf16_to_f32, Frame, SourceKind, WireNaive};

struct Daemon(Child);

impl Drop for Daemon {
    fn drop(&mut self) {
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}

fn request(id: u64, layer: u32, rows: usize) -> (RequestFrame, Vec<u8>, Vec<u8>, Vec<i32>, Vec<f32>) {
    let (payload, scales) = testkit::wire_rows(id * 31 + 1, rows);
    let (ids, w) = testkit::routes(id * 31 + 2, rows, 0);
    let f = RequestFrame {
        request_id: id,
        placement_version: 1,
        layer_id: layer,
        executor_id: 0,
        source_kind: SourceKind::Decode,
        token_position: 0,
        flags: 0,
        seq: 0,
        rows: (0..rows)
            .map(|i| RowDescriptor {
                row_id: i as u64,
                source_kind: SourceKind::Decode,
                source_request_id: id,
                token_position: i as u64,
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
                scales: scales[i * 128..(i + 1) * 128].to_vec(),
            })
            .collect(),
    };
    (f, payload, scales, ids, w)
}

#[test]
#[cfg_attr(debug_assertions, ignore = "writes and hashes a 0.91 GB image: run with --release")]
fn the_daemon_serves_over_tcp_and_fails_requests_not_the_rank() {
    let dir = std::path::PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join(format!("glm53f-rank-daemon-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let image = testkit::layer_image(0xDAE0_0001);
    let mut m = Manifest::new(0, "synthetic");
    std::fs::write(dir.join(file_name(3, 0)), &image).unwrap();
    m.layers.push(LayerEntry {
        layer: 3,
        file: file_name(3, 0),
        bytes: image.len() as u64,
        sha256: glm53f_rank::sha256::hex(&glm53f_rank::sha256::sha256(&image)),
    });
    m.write(&dir).unwrap();

    let mut child = Command::new(env!("CARGO_BIN_EXE_glm53f-rank"))
        .args(["serve", "--rank", "0", "--dir"])
        .arg(&dir)
        .args(["--listen", "127.0.0.1:0", "--allow-partial"])
        .stdout(Stdio::piped())
        .stderr(Stdio::inherit())
        .spawn()
        .expect("spawn the daemon");
    let stdout = child.stdout.take().unwrap();
    let daemon = Daemon(child);
    let mut addr = None;
    for line in BufReader::new(stdout).lines() {
        let line = line.unwrap();
        eprintln!("daemon: {line}");
        if let Some(rest) = line.strip_prefix("listening on ") {
            addr = Some(rest.split(' ').next().unwrap().to_string());
            break;
        }
    }
    let addr = addr.expect("the daemon did not start listening");

    let mut cpu = CpuKernel;
    let layer = cpu.prepare_layer(&image).unwrap();
    let mut t = TcpTransport::connect(&addr).unwrap();
    let (mut tx, mut rx) = (StreamSender::new(WireNaive::NONE), StreamReceiver::new(WireNaive::NONE));
    for (id, rows) in [(1u64, 1usize), (2, 3)] {
        let (f, p, s, ids, w) = request(id, 3, rows);
        t.send(tx.encode_request(&f).unwrap()).unwrap();
        let Frame::Return(ret) = rx.accept(&t.recv().expect("return")).expect("L4 accept") else {
            panic!("expected a return frame");
        };
        assert_eq!((ret.request_id, ret.layer_id, ret.status, ret.rows.len()), (id, 3, Status::Ok, rows));
        let mut want = vec![0u16; rows * HIDDEN];
        cpu.ffn(&layer, Rows::separate(&p, &s, rows).unwrap(), &ids, &w, &mut want).unwrap();
        // The reduce-scatter's FP32 rows are the same arithmetic before the BF16 rounding.
        let mut f32s = vec![0f32; rows * HIDDEN];
        cpu.ffn_f32(&layer, Rows::separate(&p, &s, rows).unwrap(), &ids, &w, &mut f32s).unwrap();
        assert!(f32s.iter().zip(&want).all(|(&v, &b)| glm53f_wire::bf16::f32_to_bf16_rne(v) == b), "request {id}: ffn_f32");
        for (i, row) in ret.rows.iter().enumerate() {
            let (mut d, mut r) = (0f64, 0f64);
            for (&g, &x) in row.codes.iter().zip(&want[i * HIDDEN..(i + 1) * HIDDEN]) {
                let (g, x) = (bf16_to_f32(g) as f64, bf16_to_f32(x) as f64);
                d += (g - x) * (g - x);
                r += x * x;
            }
            // Bit-exact on the CPU backend; the GPU's accumulation differs slightly.
            assert!((d / r).sqrt() < 3e-3, "request {id} row {i}: rms {:.2e}", (d / r).sqrt());
        }
    }
    // Layer 4 is not resident: the request fails, the connection closes.
    let (f, ..) = request(3, 4, 2);
    t.send(tx.encode_request(&f).unwrap()).unwrap();
    let Frame::Return(ret) = rx.accept(&t.recv().expect("error return")).unwrap() else {
        panic!("expected a return frame");
    };
    assert_eq!(ret.status, Status::Error);
    assert!(t.recv().is_none(), "the connection closes after a failed request");
    // The daemon keeps listening: a new connection serves again.
    let mut t2 = TcpTransport::connect(&addr).unwrap();
    let (mut tx2, mut rx2) = (StreamSender::new(WireNaive::NONE), StreamReceiver::new(WireNaive::NONE));
    let (f, ..) = request(4, 3, 1);
    t2.send(tx2.encode_request(&f).unwrap()).unwrap();
    assert!(matches!(rx2.accept(&t2.recv().unwrap()).unwrap(), Frame::Return(r) if r.status == Status::Ok));
    drop(daemon);
    std::fs::remove_dir_all(&dir).ok();
}
