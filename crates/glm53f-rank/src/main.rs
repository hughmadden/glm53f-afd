//! `glm53f-rank`: the expert-rank daemon of glm53f-afd (ported from
//! mimo26f-afd v1.2.0 `crates/mimo26-spark/src/main.rs`).
//!
//! ```text
//! glm53f-rank serve  --rank R --dir DIR [--listen ADDR] [--recv-slots N] [--peers A0,A1,A2,A3] [--peer-timeout-ms MS] [--lazy] [--allow-partial]
//! glm53f-rank slice  --checkpoint CKPT --rank R --out DIR [--layers 3-44] [--mtp] [--source TEXT]
//! glm53f-rank verify --rank R --dir DIR [--allow-partial]
//! ```
//!
//! `serve`:
//!
//! 1. verifies every layer image in DIR against its manifest (size and
//!    SHA-256) and refuses to serve on any mismatch ([`glm53f_rank::boot`]);
//! 2. checks that the kernels were built for this GPU (`cuda` builds);
//! 3. prepares every layer on the device before listening (`--lazy`: on first
//!    use instead); `--allow-partial` serves a directory that holds only some
//!    layers (bring-up and tests; a request for a missing layer fails);
//! 4. listens, refuses connections that do not arrive on the RDMA fabric
//!    (`glm53f_rdma::fabric_port`), and serves `DS41RTE3` v3 request frames
//!    with the L4 ladder: over RDMA the rows are read where the NIC landed
//!    them and the BF16 return is written into the registered send buffer;
//!    over TCP the frames are owned buffers. Over RDMA up to `--recv-slots`
//!    requests (`GLM53F_RANK_RECV_SLOTS`, 4 by default) queue in the receive
//!    slots it posts, served in order: the coordinator's prefill lanes keep that
//!    many exchanges in flight, and the handshake tells it how many;
//! 5. with `--peers` (the four ranks' peer-mesh addresses in rank order,
//!    `GLM53F_RANK_PEERS`), joins the peer mesh ([`glm53f_rank::mesh`]) and
//!    serves reduce-scattered requests (`DS41RTE3` v4): the partial is kept in
//!    FP32, each peer gets its rows, and only this rank's partition returns,
//!    summed. An exchange that misses a peer for `--peer-timeout-ms`
//!    (`GLM53F_RANK_PEER_TIMEOUT_MS`, 10,000 by default) fails the request.
//!    `GLM53F_RDMA=1` opens the links this rank dials as RDMA RC.
//!
//! `slice` cuts rank R's share of an EXL3 checkpoint into DIR (one image per
//! layer and a manifest). `verify` runs the boot readback alone.
//!
//! Environment: `GLM53F_RANK_TRACE=1` (a line per request, and one per
//! reduce-scattered exchange), `GLM53F_TIMELINE=1` (cross-host timeline
//! events), `GLM53F_RANK_SMALL`/`_MID`/`_LARGE` and `GLM53F_RANK_SMALL_MAX`/
//! `_MID_MAX` (the kernel configuration by row count, `exl3_cuda::Policy`;
//! the boot log prints it), `GLM53F_RANK_DUMP_FRAME=<path>` with `GLM53F_RANK_DUMP_LAYER=<id>`
//! (default 3): write the first request frame of that layer to `<path>` for
//! offline replay. `GLM53F_WIRE_ALLOW_LAN=1` admits connections off the RDMA
//! fabric (tests only), from the coordinator and from peers.

use std::io::Write;
use std::path::PathBuf;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use glm53f_rank::consts::{is_moe_layer, FIRST_MOE_LAYER, HIDDEN, LAST_MOE_LAYER, MTP_LAYER, RANK_WIDTH, WORLD};
use glm53f_rank::kernel::ExpertKernel;
use glm53f_rank::manifest::Expect;
use glm53f_rank::mesh::{self, Mesh, MeshConfig};
use glm53f_rank::resident::{write_rank_dir, Resident};
use glm53f_rank::serve::{return_meta_view, serve_view, serve_view_f32, Layers, Timings};
use glm53f_rank::transport::{ByteTransport, RdmaTransport, TcpTransport, RECV_SLOTS};
use glm53f_rank::{boot, pagecache, timeline};
use glm53f_wire::frame::{RequestView, ReturnFrame, ReturnRow, FLAG_RETURN_REQUIRED, FLAG_ROW_SLICE};
use glm53f_wire::l4::{StreamReceiver, StreamSender};
use glm53f_wire::layout::Status;
use glm53f_wire::WireNaive;

const USAGE: &str = "usage:
  glm53f-rank serve  --rank <0..3> --dir <rank-dir> [--listen <addr:port>] [--recv-slots <1..16>] [--peers <a0,a1,a2,a3>] [--peer-timeout-ms <ms>] [--lazy] [--allow-partial]
  glm53f-rank slice  --checkpoint <exl3-checkpoint-dir> --rank <0..3> --out <rank-dir> [--layers 3-44] [--mtp] [--source <text>]
  glm53f-rank verify --rank <0..3> --dir <rank-dir> [--allow-partial]";

struct Args {
    cmd: String,
    rank: usize,
    dir: Option<PathBuf>,
    listen: String,
    /// RDMA receive slots per coordinator connection: the requests it may queue.
    recv_slots: u32,
    lazy: bool,
    expect: Expect,
    checkpoint: Option<PathBuf>,
    out: Option<PathBuf>,
    layers: Vec<u32>,
    source: String,
    /// The peer mesh, from `--peers` and `--peer-timeout-ms` or their
    /// environment fallbacks.
    mesh: Option<MeshConfig>,
}

fn parse_layers(s: &str) -> Result<Vec<u32>, String> {
    let mut out = Vec::new();
    for part in s.split(',') {
        let (a, b) = part.split_once('-').unwrap_or((part, part));
        let a: u32 = a.parse().map_err(|_| format!("bad layer {a}"))?;
        let b: u32 = b.parse().map_err(|_| format!("bad layer {b}"))?;
        for l in a..=b {
            if !is_moe_layer(l, true) {
                return Err(format!("layer {l} is not a MoE layer ({FIRST_MOE_LAYER}..={LAST_MOE_LAYER}, {MTP_LAYER})"));
            }
            out.push(l);
        }
    }
    Ok(out)
}

fn parse_args() -> Result<Args, String> {
    let mut it = std::env::args().skip(1);
    let cmd = it.next().ok_or("missing command")?;
    let mut a = Args {
        cmd,
        rank: usize::MAX,
        dir: None,
        listen: "0.0.0.0:8600".into(),
        recv_slots: RECV_SLOTS,
        lazy: false,
        expect: Expect::SERVING,
        checkpoint: None,
        out: None,
        layers: (FIRST_MOE_LAYER..=LAST_MOE_LAYER).collect(),
        source: "unspecified".into(),
        mesh: None,
    };
    let mut mtp = false;
    let env = |k: &str| std::env::var(k).ok().filter(|v| !v.is_empty());
    let (mut peers, mut peer_timeout) = (env("GLM53F_RANK_PEERS"), env("GLM53F_RANK_PEER_TIMEOUT_MS"));
    let mut recv_slots = env("GLM53F_RANK_RECV_SLOTS");
    while let Some(k) = it.next() {
        let mut val = || it.next().ok_or(format!("{k} needs a value"));
        match k.as_str() {
            "--rank" => a.rank = val()?.parse().map_err(|_| "bad --rank")?,
            "--dir" => a.dir = Some(PathBuf::from(val()?)),
            "--listen" => a.listen = val()?,
            "--recv-slots" => recv_slots = Some(val()?),
            "--peers" => peers = Some(val()?),
            "--peer-timeout-ms" => peer_timeout = Some(val()?),
            "--lazy" => a.lazy = true,
            "--allow-partial" => a.expect.all_layers = false,
            "--checkpoint" => a.checkpoint = Some(PathBuf::from(val()?)),
            "--out" => a.out = Some(PathBuf::from(val()?)),
            "--layers" => a.layers = parse_layers(&val()?)?,
            "--mtp" => mtp = true,
            "--source" => a.source = val()?,
            other => return Err(format!("unknown argument {other}")),
        }
    }
    if a.rank >= WORLD {
        return Err(format!("--rank must be 0..{WORLD}"));
    }
    if let Some(n) = recv_slots {
        a.recv_slots = n.parse().ok().filter(|n| (1..=16).contains(n)).ok_or(format!("--recv-slots {n}: 1 to 16"))?;
    }
    if let Some(p) = peers {
        let timeout = match peer_timeout {
            Some(t) => std::time::Duration::from_millis(t.parse().map_err(|_| format!("bad --peer-timeout-ms {t}"))?),
            None => mesh::DEFAULT_TIMEOUT,
        };
        a.mesh = Some(MeshConfig::new(a.rank, &p, timeout, env("GLM53F_RDMA").as_deref() == Some("1"))?);
    }
    if mtp && !a.layers.contains(&MTP_LAYER) {
        a.layers.push(MTP_LAYER);
    }
    Ok(a)
}

fn main() {
    let args = match parse_args() {
        Ok(a) => a,
        Err(e) => {
            eprintln!("{USAGE}\nerror: {e}");
            std::process::exit(2);
        }
    };
    let rc = match args.cmd.as_str() {
        "serve" => serve(&args),
        "slice" => slice(&args),
        "verify" => verify(&args),
        other => {
            eprintln!("{USAGE}\nerror: unknown command {other}");
            2
        }
    };
    std::process::exit(rc);
}

fn slice(a: &Args) -> i32 {
    let (Some(ckpt), Some(out)) = (&a.checkpoint, &a.out) else {
        eprintln!("{USAGE}\nerror: slice needs --checkpoint and --out");
        return 2;
    };
    match write_rank_dir(ckpt, a.rank, &a.layers, out, &a.source) {
        Ok(m) => {
            println!("rank {}: {} layer images in {}", a.rank, m.layers.len(), out.display());
            0
        }
        Err(e) => {
            eprintln!("slice failed: {e}");
            1
        }
    }
}

fn verify(a: &Args) -> i32 {
    let Some(dir) = &a.dir else {
        eprintln!("{USAGE}\nerror: verify needs --dir");
        return 2;
    };
    match boot::readback(dir, a.rank, &a.expect) {
        Ok(r) => {
            println!("{}", r.summary);
            // Nothing reads the images after this: give back the pages the hashing cached.
            if let Ok(resident) = Resident::load_manifest(dir, a.rank, &a.expect) {
                for layer in resident.layers() {
                    drop_cached(&resident, layer);
                }
            }
            0
        }
        Err(e) => {
            eprintln!("{e}");
            3
        }
    }
}

/// Give back the page cache's copy of `layer`'s image; a failed hint is only logged.
fn drop_cached(resident: &Resident, layer: u32) {
    if let Err(e) = resident.drop_cache(layer) {
        eprintln!("layer {layer}: page cache not dropped: {e}");
    }
}

/// The boot log's memory line: `MemFree` beside `MemAvailable` (the page cache counts as
/// available, but a CUDA allocation can fail before the kernel reclaims it).
fn log_memory(when: &str) {
    if let Some(m) = pagecache::memory_line() {
        println!("memory {when}: {m}");
    }
}

fn serve(a: &Args) -> i32 {
    let Some(dir) = &a.dir else {
        eprintln!("{USAGE}\nerror: serve needs --dir");
        return 2;
    };
    log_memory("before the boot readback");
    // Boot identity readback: refuse to serve on any mismatch.
    match boot::readback(dir, a.rank, &a.expect) {
        Ok(r) => println!("boot readback: {}", r.summary),
        Err(e) => {
            eprintln!("{e}");
            return 3;
        }
    }
    log_memory("after the boot readback");
    let resident = match Resident::load_manifest(dir, a.rank, &a.expect) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("resident index failed: {e}");
            return 4;
        }
    };
    println!(
        "resident: rank {} holds intermediate channels [{}, {}) of every expert in {} layers",
        a.rank,
        a.rank * RANK_WIDTH,
        (a.rank + 1) * RANK_WIDTH,
        resident.files.len()
    );

    #[cfg(feature = "cuda")]
    let kernel = {
        match glm53f_rank::exl3_cuda::check_device() {
            Ok(id) => println!("device: {id}"),
            Err(e) => {
                eprintln!("device gate failed: {e}");
                return 6;
            }
        }
        match glm53f_rank::exl3_cuda::CudaKernel::new() {
            Ok(k) => {
                println!("{}", k.policy.summary());
                k
            }
            Err(e) => {
                eprintln!("kernel scratch failed: {e}");
                return 7;
            }
        }
    };
    #[cfg(not(feature = "cuda"))]
    let kernel = {
        eprintln!("warning: built without `cuda`; serving on the CPU reference backend (tests only)");
        glm53f_rank::kernel::CpuKernel
    };
    run(a, resident, kernel)
}

/// The rank's kernel, its prepared layers and where to load the rest.
struct Rank<K: ExpertKernel> {
    kernel: K,
    layers: Layers<K::Layer>,
    resident: Resident,
    /// A reduce-scattered request's FP32 partial [rows, 4096] (grow-only).
    partial: Vec<f32>,
}

/// The BF16 rows of a return frame assembled in place: `frame` after its
/// 128-byte header, `rows` x 8,192 bytes, 2-byte aligned.
fn bf16_rows(frame: &mut [u8], rows: usize) -> Result<&mut [u16], String> {
    let body = &mut frame[glm53f_wire::HEADER_LEN..];
    if body.len() != rows * HIDDEN * 2 || !(body.as_ptr() as usize).is_multiple_of(2) {
        return Err("return buffer is not rows x 8,192 aligned bytes".into());
    }
    // SAFETY: exact length and 2-byte alignment checked above.
    Ok(unsafe { std::slice::from_raw_parts_mut(body.as_mut_ptr() as *mut u16, rows * HIDDEN) })
}

impl<K: ExpertKernel> Rank<K> {
    /// Prepare `layer` if it is not yet; returns the load time in ms.
    fn ensure(&mut self, layer: u32) -> Result<f64, String> {
        if self.layers.get(layer).is_some() {
            return Ok(0.0);
        }
        let t = Instant::now();
        let image = self.resident.layer_image(layer)?;
        let prepared = self.kernel.prepare_layer(&image).map_err(|e| format!("prepare layer {layer}: {e}"))?;
        self.layers.insert(layer, prepared);
        let ms = t.elapsed().as_secs_f64() * 1e3;
        // The image is on the device and this was its last read (the boot readback's was the
        // first): give back the pages the two reads cached.
        drop_cached(&self.resident, layer);
        Ok(ms)
    }

    /// Serve `view` into `frame` (header space, then `rows` BF16 rows).
    fn compute_into(&mut self, view: &RequestView<'_>, frame: &mut [u8], timings: &mut Timings) -> Result<(), String> {
        let body16 = bf16_rows(frame, view.rows)?;
        let layer = self.layers.get(view.layer_id).ok_or_else(|| format!("layer {} is not prepared", view.layer_id))?;
        serve_view(view, &mut self.kernel, layer, timings, body16).map(|_| ())
    }

    /// Serve a reduce-scattered `view` into [`Rank::partial`]: all its rows,
    /// in FP32, before the ranks add them.
    fn compute_f32(&mut self, view: &RequestView<'_>, timings: &mut Timings) -> Result<(), String> {
        let n = view.rows * HIDDEN;
        if self.partial.len() < n {
            self.partial.resize(n, 0.0);
        }
        let layer = self.layers.get(view.layer_id).ok_or_else(|| format!("layer {} is not prepared", view.layer_id))?;
        serve_view_f32(view, &mut self.kernel, layer, timings, &mut self.partial[..n]).map(|_| ())
    }
}

/// A `Status::Error` return for a failed request: one zero BF16 row per row,
/// so the coordinator decodes it and fails the request instead of seeing a
/// dropped connection.
fn error_return_for(meta: &ReturnFrame, rows: usize) -> ReturnFrame {
    ReturnFrame {
        status: Status::Error,
        flags: 0,
        rows: (0..rows).map(|_| ReturnRow { codes: vec![0u16; HIDDEN] }).collect(),
        ..meta.clone()
    }
}

/// CUDA page-lock of an RDMA receive ring, released before the transport frees
/// the ring (declare it after the transport: locals drop in reverse).
#[cfg(feature = "cuda")]
struct RingRegistration(Option<*mut u8>);

#[cfg(feature = "cuda")]
impl Drop for RingRegistration {
    fn drop(&mut self) {
        if let Some(p) = self.0 {
            // SAFETY: registered by this guard; the transport still owns the ring.
            unsafe { glm53f_rank::cuda::cudaHostUnregister(p as *mut core::ffi::c_void) };
        }
    }
}

fn run<K: ExpertKernel>(a: &Args, resident: Resident, kernel: K) -> i32 {
    let mut rank = Rank { kernel, layers: Layers::default(), resident, partial: Vec::new() };
    let with_mtp = rank.resident.files.contains_key(&MTP_LAYER);
    if !a.lazy {
        let t = Instant::now();
        for layer in rank.resident.layers() {
            if let Err(e) = rank.ensure(layer) {
                eprintln!("{e}");
                return 5;
            }
        }
        println!("prepared {} layers in {:.1} s", rank.layers.len(), t.elapsed().as_secs_f64());
        log_memory("after preparing the layers");
    }
    // The peer mesh lives for the daemon's lifetime, across coordinator connections.
    let mut mesh = match &a.mesh {
        None => None,
        Some(cfg) => match Mesh::start(cfg) {
            Ok(m) => {
                println!(
                    "peer mesh: rank {} on {} ({}), exchange timeout {} ms",
                    a.rank,
                    m.local_addr(),
                    if cfg.rdma { "dialling RDMA RC" } else { "TCP" },
                    cfg.timeout.as_millis()
                );
                Some(m)
            }
            Err(e) => {
                eprintln!("{e}");
                return 5;
            }
        },
    };
    let listener = match std::net::TcpListener::bind(&a.listen) {
        Ok(l) => l,
        Err(e) => {
            eprintln!("bind {} failed: {e}", a.listen);
            return 5;
        }
    };
    println!("listening on {} (rank {})", listener.local_addr().map(|x| x.to_string()).unwrap_or_default(), a.rank);
    let _ = std::io::stdout().flush();

    let naive = WireNaive::NONE;
    let dump_path = std::env::var_os("GLM53F_RANK_DUMP_FRAME").map(PathBuf::from);
    let dump_layer: u32 =
        std::env::var("GLM53F_RANK_DUMP_LAYER").ok().and_then(|v| v.parse().ok()).unwrap_or(FIRST_MOE_LAYER);
    let mut dumped = false;
    for stream in listener.incoming() {
        let stream = match stream {
            Ok(s) => s,
            Err(e) => {
                eprintln!("accept failed: {e}");
                continue;
            }
        };
        // Inference runs only on the RDMA fabric: a coordinator that dialled
        // this daemon on another network is refused, TCP or RDMA.
        match stream.local_addr().map_err(|e| e.to_string()).and_then(|x| glm53f_rdma::fabric_port(x.ip())) {
            Ok(Some((dev, port, _, gbps))) => eprintln!("connection on fabric {dev} port {port} at {gbps} Gb/s"),
            Ok(None) => eprintln!("connection on loopback or GLM53F_WIRE_ALLOW_LAN=1, fabric check skipped"),
            Err(e) => {
                eprintln!("refused connection from {:?}: {e}", stream.peer_addr());
                continue;
            }
        }
        // A coordinator that opens with the RDMA handshake gets an RC queue
        // pair; anything else stays on TCP.
        let rdma = match RdmaTransport::accept(&stream, a.recv_slots) {
            Ok(r) => r,
            Err(e) => {
                eprintln!("rdma handshake failed: {e}");
                continue;
            }
        };
        let mut transport: Box<dyn ByteTransport> = match rdma {
            Some(t) => Box::new(t),
            None => match TcpTransport::from_stream(stream) {
                Ok(t) => Box::new(t),
                Err(e) => {
                    eprintln!("transport setup failed: {e}");
                    continue;
                }
            },
        };
        let in_place = transport.in_place_recv();
        // Page-lock the RDMA receive ring so the device copies of the rows run as DMA.
        #[cfg(feature = "cuda")]
        let _ring = RingRegistration(if in_place {
            transport.recv_ring().and_then(|(p, n)| {
                // SAFETY: the live receive ring of this connection's transport.
                let rc = unsafe { glm53f_rank::cuda::cudaHostRegister(p as *mut core::ffi::c_void, n, 0) };
                if rc == glm53f_rank::cuda::SUCCESS {
                    Some(p)
                } else {
                    eprintln!("cudaHostRegister of the receive ring failed ({rc}); copies go unpinned");
                    None
                }
            })
        } else {
            None
        });
        let mut rx = StreamReceiver::new(naive);
        let mut tx = StreamSender::new(naive);
        let mut window = timeline::Window::default();
        loop {
            let recv_start = Instant::now();
            // The request: where the NIC landed it (RDMA) or an owned frame (TCP).
            let owned: Vec<u8>;
            let bytes: &[u8] = if in_place {
                let Some((ptr, len)) = transport.recv_slot() else { break };
                // SAFETY: the slot stays valid, and the NIC cannot write it, until
                // release_slot below; `bytes` is not used after that.
                unsafe { std::slice::from_raw_parts(ptr, len) }
            } else {
                let Some(b) = transport.recv() else { break };
                owned = b;
                &owned
            };
            let recv_ms = recv_start.elapsed().as_secs_f64() * 1e3;
            let view = match RequestView::parse(bytes, naive) {
                Ok(v) => v,
                Err(e) => {
                    eprintln!("L4 accept failed: {e}");
                    break;
                }
            };
            if let Err(e) = rx.accept_seq(view.seq) {
                eprintln!("L4 accept failed: {e}");
                break;
            }
            timeline::tl("crc_done", Some(view.layer_id));
            let layer = view.layer_id;
            if !is_moe_layer(layer, with_mtp) {
                eprintln!("request layer {layer} is not served here ({FIRST_MOE_LAYER}..={LAST_MOE_LAYER}{})", if with_mtp { ", 45" } else { "" });
                break;
            }
            if let (Some(path), false) = (&dump_path, dumped) {
                if layer == dump_layer {
                    match std::fs::write(path, bytes) {
                        Ok(()) => eprintln!("dumped a layer-{layer} request frame ({} B) to {}", bytes.len(), path.display()),
                        Err(e) => eprintln!("dump frame failed: {e}"),
                    }
                    dumped = true;
                }
            }
            let rows = view.rows;
            // A reduce-scattered request returns only this rank's partition, summed
            // over the four ranks (version 4); any other returns every row.
            let sharded = mesh::exchange_for(&view, a.rank);
            let plain = return_meta_view(&view);
            let (ret_rows, meta) = match sharded {
                Some(x) => {
                    let (first, count) = x.own();
                    let slice = ReturnFrame { flags: FLAG_RETURN_REQUIRED | FLAG_ROW_SLICE, token_position: first as u64, ..plain.clone() };
                    (count, slice)
                }
                None => (rows, plain.clone()),
            };
            let len = glm53f_wire::HEADER_LEN + ret_rows * glm53f_wire::RETURN_ROW_BYTES;
            let mut timings = Timings::default();
            // The return is assembled in place: in the transport's registered
            // send buffer (RDMA) or an owned frame (TCP).
            let mut owned_ret: Vec<u8> = Vec::new();
            let mut in_buffer = false;
            let computed = rank.ensure(layer).and_then(|load_ms| {
                if sharded.is_some() {
                    if mesh.is_none() {
                        return Err("a reduce-scattered request, and this rank has no peer mesh (--peers)".into());
                    }
                    return rank.compute_f32(&view, &mut timings).map(|_| load_ms);
                }
                match transport.send_buffer() {
                    Some(b) if len > b.len() => Err(format!("return frame {len} B exceeds the send buffer")),
                    Some(b) => {
                        in_buffer = true;
                        rank.compute_into(&view, &mut b[..len], &mut timings)
                    }
                    None => {
                        owned_ret = vec![0u8; len];
                        rank.compute_into(&view, &mut owned_ret, &mut timings)
                    }
                }
                .map(|_| load_ms)
            });
            // The rows are on the device (the kernel synchronized): hand the slot
            // back before the return goes out, so the next request finds it posted.
            // `view` and `bytes` borrow the slot and are not used past this point.
            if in_place {
                if let Err(e) = transport.release_slot() {
                    eprintln!("rdma: re-post failed: {e}");
                    break;
                }
            }
            // The reduce-scatter: the peers get their rows of the partial, and theirs of
            // this rank's partition are summed with its own straight into the return.
            let mut peer_times = None;
            let computed = match (computed, sharded, mesh.as_mut()) {
                (Ok(load_ms), Some(x), Some(m)) => {
                    let frame = match transport.send_buffer() {
                        Some(b) if len > b.len() => Err(format!("return frame {len} B exceeds the send buffer")),
                        Some(b) => {
                            in_buffer = true;
                            Ok(&mut b[..len])
                        }
                        None => {
                            owned_ret = vec![0u8; len];
                            Ok(&mut owned_ret[..])
                        }
                    };
                    frame
                        .and_then(|f| bf16_rows(f, ret_rows))
                        .and_then(|out| m.exchange(&x, &rank.partial[..rows * HIDDEN], out))
                        .map(|t| {
                            peer_times = Some(t);
                            load_ms
                        })
                }
                (c, _, _) => c,
            };
            let load_ms = match computed {
                Ok(ms) => ms,
                Err(e) => {
                    // Fail the request, not the rank: an error return, then close
                    // this connection and keep listening.
                    eprintln!("serve layer {layer} failed: {e}");
                    if let Ok(stamped) = tx.encode_return(&error_return_for(&plain, rows)) {
                        let _ = transport.send(stamped);
                    }
                    break;
                }
            };
            let header = match glm53f_wire::frame::return_header_seq(&meta, ret_rows, tx.take_seq(), naive) {
                Ok(h) => h,
                Err(e) => {
                    eprintln!("L4 header failed: {e}");
                    break;
                }
            };
            let send_start = Instant::now();
            let sent = if in_buffer {
                let Some(buf) = transport.send_buffer() else {
                    eprintln!("rdma: send buffer lost");
                    break;
                };
                buf[..glm53f_wire::HEADER_LEN].copy_from_slice(&header);
                glm53f_wire::frame::seal_in_place(&mut buf[..len], naive);
                timeline::tl("seal_done", Some(layer));
                transport.send_in_place(len)
            } else {
                owned_ret[..glm53f_wire::HEADER_LEN].copy_from_slice(&header);
                glm53f_wire::frame::seal_in_place(&mut owned_ret, naive);
                timeline::tl("seal_done", Some(layer));
                transport.send(std::mem::take(&mut owned_ret))
            };
            if let Err(e) = sent {
                eprintln!("send failed: {e}");
                break;
            }
            let send_ms = send_start.elapsed().as_secs_f64() * 1e3;
            window.add(rows, timings.ffn_ms);
            if timeline::trace() {
                let epoch_ms = SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_millis()).unwrap_or(0);
                eprintln!(
                    "timing t={epoch_ms} layer={layer} rows={rows} recv={recv_ms:.3} load={load_ms:.3} plan={:.3} ffn={:.3} send={send_ms:.3} ms",
                    timings.plan_ms, timings.ffn_ms,
                );
                if let (Some(x), Some(t)) = (sharded, peer_times) {
                    let (first, count) = x.own();
                    eprintln!(
                        "exchange t={epoch_ms} layer={layer} rows={rows} part={first}+{count} dtype={} send={:.3} wait={:.3} sum={:.3} return={send_ms:.3} ms",
                        x.dtype.name(),
                        t.send_ms,
                        t.wait_ms,
                        t.sum_ms
                    );
                }
            }
        }
    }
    0
}
