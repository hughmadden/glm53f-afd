//! The ranks' peer mesh: the links the prefill reduce-scatter's exchange
//! frames travel on (README.md, "Prefill reduce-scatter").
//!
//! **Why a mesh.** In a reduce-scattered request every rank sends each peer
//! that peer's rows and needs its own partition's rows from all three, so every
//! pair of ranks has a link: six in all, one per pair. The lower rank of a pair
//! connects and the higher one accepts, so a rank dials the ranks above it and
//! listens for the ranks below.
//!
//! **Configuration** ([`MeshConfig`]): the four ranks' peer addresses in rank
//! order (the daemon's `--peers`, `GLM53F_RANK_PEERS`); rank `r` listens on
//! entry `r`. The daemon's fabric guard applies to these connections too
//! (`glm53f_rdma::fabric_port`: loopback and `GLM53F_WIRE_ALLOW_LAN=1` are the
//! test exemptions), on both sides.
//!
//! **Links.** A link opens with a 16-byte hello each way (magic `G53RMESH`, the
//! wire version, the world size, source and destination ranks, the transport),
//! so a peer at the wrong address, of another version or world is refused
//! before any frame. Then either
//!
//! - TCP (tests, and without RDMA): a reader thread per link validates each
//!   frame (header, CRC unless disabled, L4 sequence) and hands it to the
//!   serving thread; the serving thread writes its frames itself. The reader
//!   always drains the socket, so two ranks sending each other megabytes at
//!   once cannot block each other;
//! - RDMA RC (`MeshConfig::rdma`, `GLM53F_RDMA=1` in an `rdma` build, on the
//!   target hardware): the transport of the coordinator's link
//!   (`RdmaTransport`), sized for exchange frames; the connecting side opens
//!   it, the accepting side follows the hello. The serving thread writes frames
//!   into the registered send halves and busy-polls the receive queue, as the
//!   RDMA design polls completions.
//!
//! Every frame is a `DS41RTE3` version-4 exchange frame with the L4 ladder per
//! link and direction (`glm53f_wire::row_shard`).
//!
//! **Exchanges** are keyed by request id and layer (`reduce_scatter::Exchange`).
//! Frames of another exchange (another of the coordinator's prefill lanes, which a
//! faster peer may already be sending) wait in a pending list until their
//! exchange claims them; frames nobody claims within twice the timeout, and
//! the oldest beyond [`MAX_PENDING`] per peer, are dropped, which only
//! happens after a failed exchange. The coordinator starts each connection's
//! request ids at a random base, so a stale frame never names a later
//! exchange.
//!
//! **Failure.** An exchange fails closed and never hangs: a peer without a
//! link, or whose frame has not arrived, within the timeout (`--peer-timeout-ms`,
//! `GLM53F_RANK_PEER_TIMEOUT_MS`, 10 s by default); a link lost while its frame
//! is awaited (at once); a frame that names this exchange but not its rows,
//! dtype or ranks; a duplicate. A failed link is dropped, and the lower rank
//! dials again. The daemon turns the failure into an error return.

use std::io::{Read, Write};
use std::net::{Shutdown, SocketAddr, TcpListener, TcpStream, ToSocketAddrs};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::mpsc::{channel, Receiver, RecvTimeoutError, Sender, TryRecvError};
use std::sync::Arc;
use std::time::{Duration, Instant};

use glm53f_wire::l4::{StreamReceiver, StreamSender};
use glm53f_wire::row_shard::{ExchangeDtype, ExchangeHeader, ExchangeView};
use glm53f_wire::WireNaive;

use crate::consts::{MAX_ROWS, WORLD};
use crate::reduce_scatter::Exchange;
use crate::transport::{ByteTransport, RdmaTransport, TcpTransport};

/// The hello's magic.
const HELLO_MAGIC: [u8; 8] = *b"G53RMESH";
const HELLO_LEN: usize = 16;
/// How long a link's hello (and RDMA handshake) may take.
const HANDSHAKE_TIMEOUT: Duration = Duration::from_secs(5);
/// Frames kept per peer for exchanges not yet started.
pub const MAX_PENDING: usize = 8;
/// The default exchange timeout.
pub const DEFAULT_TIMEOUT: Duration = Duration::from_secs(10);
/// The largest exchange frame: a quarter of the largest request, in BF16.
const MAX_FRAME: usize = glm53f_wire::HEADER_LEN + MAX_ROWS.div_ceil(WORLD) * 2 * crate::consts::HIDDEN;
/// Receive slots of an RDMA link. Two are enough however many requests the coordinator queues:
/// the serving thread copies a frame out as soon as it polls it, and a peer is at most one
/// exchange ahead (it cannot finish an exchange without this rank's frame).
const MESH_SLOTS: u32 = 2;

static NEXT_LINK: AtomicU64 = AtomicU64::new(1);

/// Where the mesh listens and dials, and how long an exchange may wait.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MeshConfig {
    pub rank: usize,
    /// The four ranks' peer addresses (`host:port`) in rank order.
    pub peers: Vec<String>,
    pub timeout: Duration,
    /// Open RDMA RC queue pairs on the links this rank dials (accepted links
    /// follow the dialling rank's hello).
    pub rdma: bool,
}

impl MeshConfig {
    /// `peers`: four comma-separated addresses in rank order.
    pub fn new(rank: usize, peers: &str, timeout: Duration, rdma: bool) -> Result<Self, String> {
        let peers: Vec<String> = peers.split(',').map(|s| s.trim().to_string()).filter(|s| !s.is_empty()).collect();
        if peers.len() != WORLD || rank >= WORLD {
            return Err(format!("peers: need {WORLD} addresses in rank order and a rank below {WORLD}, got {} and {rank}", peers.len()));
        }
        if timeout.is_zero() {
            return Err("peers: the exchange timeout must be positive".into());
        }
        Ok(MeshConfig { rank, peers, timeout, rdma })
    }
}

fn hello(src: usize, dst: usize, rdma: bool) -> [u8; HELLO_LEN] {
    let mut h = [0u8; HELLO_LEN];
    h[..8].copy_from_slice(&HELLO_MAGIC);
    h[8..10].copy_from_slice(&glm53f_wire::layout::VERSION_ROW_SHARD.to_le_bytes());
    h[10] = WORLD as u8;
    h[11] = src as u8;
    h[12] = dst as u8;
    h[13] = u8::from(rdma);
    h
}

/// A peer's hello: `(source rank, RDMA)`, checked against this rank.
fn read_hello(s: &mut TcpStream, me: usize) -> Result<(usize, bool), String> {
    let mut h = [0u8; HELLO_LEN];
    s.read_exact(&mut h).map_err(|e| format!("hello: {e}"))?;
    let version = u16::from_le_bytes([h[8], h[9]]);
    if h[..8] != HELLO_MAGIC || version != glm53f_wire::layout::VERSION_ROW_SHARD || h[10] as usize != WORLD || h[14..] != [0, 0] {
        return Err(format!("hello: not a version-{} mesh peer of world {WORLD}: {h:02x?}", glm53f_wire::layout::VERSION_ROW_SHARD));
    }
    let (src, dst, rdma) = (h[11] as usize, h[12] as usize, h[13]);
    if dst != me || src >= WORLD || src == me || rdma > 1 {
        return Err(format!("hello: from rank {src} to rank {dst} (this is rank {me})"));
    }
    Ok((src, rdma == 1))
}

/// The inference-fabric rule, for peer connections as for the coordinator's.
fn fabric_guard(s: &TcpStream) -> Result<(), String> {
    let local = s.local_addr().map_err(|e| e.to_string())?;
    glm53f_rdma::fabric_port(local.ip()).map(|_| ())
}

/// One live link to a peer.
struct Link {
    id: u64,
    tx: Tx,
    /// L4 sequence stamps of this link's outgoing frames.
    seq: StreamSender,
    /// Shut down when the link is dropped, so its reader thread and the peer
    /// see the close.
    ctl: TcpStream,
}

enum Tx {
    /// The write half; a reader thread owns the read half.
    Tcp(TcpTransport),
    /// Both directions, polled by the serving thread.
    Rdma { t: Box<RdmaTransport>, rx: StreamReceiver },
}

impl Drop for Link {
    fn drop(&mut self) {
        let _ = self.ctl.shutdown(Shutdown::Both);
    }
}

impl Link {
    /// Send exchange `x`'s frame for peer `q`.
    fn send(&mut self, x: &Exchange, q: usize, partial: &[f32]) -> Result<(), String> {
        let seq = self.seq.take_seq();
        let len = x.header_to(q).frame_len();
        match &mut self.tx {
            Tx::Tcp(t) => {
                let mut f = vec![0u8; len];
                x.write_frame(q, partial, seq, &mut f)?;
                t.send(f).map_err(|e| format!("send to rank {q}: {e}"))
            }
            Tx::Rdma { t, .. } => {
                let buf = t.send_buffer().ok_or_else(|| format!("rdma: no send buffer for rank {q}"))?;
                if len > buf.len() {
                    return Err(format!("rdma: a {len}-byte frame exceeds the send buffer"));
                }
                x.write_frame(q, partial, seq, &mut buf[..len])?;
                t.send_in_place(len).map_err(|e| format!("send to rank {q}: {e}"))
            }
        }
    }
}

enum Event {
    Up { peer: usize, link: Link },
    Frame { peer: usize, header: ExchangeHeader, bytes: Vec<u8> },
    Down { peer: usize, id: u64, why: String },
}

/// A frame received and validated, not yet claimed by its exchange.
struct Pending {
    peer: usize,
    at: Instant,
    header: ExchangeHeader,
    bytes: Vec<u8>,
}

/// Timings of one exchange (ms), for the daemon's trace line.
#[derive(Debug, Clone, Copy, Default)]
pub struct ExchangeTimes {
    /// Encoding and posting the three frames (and any wait for a link).
    pub send_ms: f64,
    /// Waiting for the peers' frames.
    pub wait_ms: f64,
    /// The sum in rank order and the BF16 rounding.
    pub sum_ms: f64,
}

/// This rank's end of the mesh, owned by the serving thread. Dropping it
/// closes its links and stops its threads.
pub struct Mesh {
    rank: usize,
    timeout: Duration,
    events: Receiver<Event>,
    links: [Option<Link>; WORLD],
    /// For the ranks above this one: asks their dialling thread for a new link.
    redial: Vec<Option<Sender<()>>>,
    pending: Vec<Pending>,
    local: SocketAddr,
    /// Tells the listening thread to stop (it is woken by a connection).
    stop: Arc<AtomicBool>,
}

impl Drop for Mesh {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::SeqCst);
        let _ = TcpStream::connect_timeout(&self.local, Duration::from_millis(200));
    }
}

impl Mesh {
    /// Bind this rank's peer address and start dialling the ranks above it.
    pub fn start(cfg: &MeshConfig) -> Result<Mesh, String> {
        let addr = &cfg.peers[cfg.rank];
        let listener = TcpListener::bind(addr).map_err(|e| format!("peers: bind {addr}: {e}"))?;
        Mesh::with_listener(cfg, listener)
    }

    /// [`Mesh::start`] on a bound listener (tests bind port 0 first).
    pub fn with_listener(cfg: &MeshConfig, listener: TcpListener) -> Result<Mesh, String> {
        if cfg.peers.len() != WORLD || cfg.rank >= WORLD {
            return Err(format!("peers: {} addresses for rank {}", cfg.peers.len(), cfg.rank));
        }
        let local = listener.local_addr().map_err(|e| e.to_string())?;
        let (tx, events) = channel();
        let (rank, timeout) = (cfg.rank, cfg.timeout);
        let (accepted, stop) = (tx.clone(), Arc::new(AtomicBool::new(false)));
        let stopped = stop.clone();
        std::thread::Builder::new()
            .name(format!("mesh-listen-{rank}"))
            .spawn(move || listen(rank, listener, timeout, accepted, stopped))
            .map_err(|e| e.to_string())?;
        let mut redial = Vec::with_capacity(WORLD);
        for q in 0..WORLD {
            if q <= rank {
                redial.push(None);
                continue;
            }
            let (ask, asked) = channel();
            let (addr, rdma, tx) = (cfg.peers[q].clone(), cfg.rdma, tx.clone());
            std::thread::Builder::new()
                .name(format!("mesh-dial-{rank}-{q}"))
                .spawn(move || dialler(rank, q, addr, rdma, timeout, tx, asked))
                .map_err(|e| e.to_string())?;
            redial.push(Some(ask));
        }
        Ok(Mesh { rank, timeout, events, links: Default::default(), redial, pending: Vec::new(), local, stop })
    }

    /// The address this rank's peers dial.
    pub fn local_addr(&self) -> SocketAddr {
        self.local
    }

    pub fn timeout(&self) -> Duration {
        self.timeout
    }

    /// Wait (up to the timeout) until the links to all three peers are up.
    pub fn wait_ready(&mut self) -> Result<(), String> {
        let deadline = Instant::now() + self.timeout;
        self.wait_links(deadline)
    }

    /// One exchange: send each peer its rows of this rank's FP32 `partial`
    /// [rows, 4096], wait for the three peers' rows of this rank's partition,
    /// and write their sum with this rank's own rows, in rank order, as BF16
    /// into `out` [count * 4096] (`Exchange::reduce`). Fails, never hangs:
    /// each step waits at most the timeout.
    pub fn exchange(&mut self, x: &Exchange, partial: &[f32], out: &mut [u16]) -> Result<ExchangeTimes, String> {
        let t0 = Instant::now();
        self.send(x, partial)?;
        let send_ms = t0.elapsed().as_secs_f64() * 1e3;
        let (wait_ms, sum_ms) = self.finish(x, partial, out)?;
        Ok(ExchangeTimes { send_ms, wait_ms, sum_ms })
    }

    /// The second half of [`Mesh::exchange`], after [`Mesh::send`]: wait (up
    /// to the timeout) for the peers' frames and reduce. Returns the wait and
    /// the sum times (ms).
    pub fn finish(&mut self, x: &Exchange, partial: &[f32], out: &mut [u16]) -> Result<(f64, f64), String> {
        let t0 = Instant::now();
        let got = self.collect(x, t0 + self.timeout)?;
        let t1 = Instant::now();
        let peers: [Option<ExchangeView<'_>>; WORLD] = std::array::from_fn(|r| {
            got[r].as_ref().map(|p| ExchangeView { header: p.header, seq: 0, payload: &p.bytes[glm53f_wire::HEADER_LEN..] })
        });
        x.reduce(partial, &peers, out)?;
        let ms = |a: Instant, b: Instant| (b - a).as_secs_f64() * 1e3;
        Ok((ms(t0, t1), ms(t1, Instant::now())))
    }

    /// Send exchange `x`'s three frames, waiting (within the timeout) for
    /// links that are not up yet, as at start-up or after a peer restarted.
    pub fn send(&mut self, x: &Exchange, partial: &[f32]) -> Result<(), String> {
        if x.rank != self.rank {
            return Err(format!("mesh: rank {}'s exchange on rank {}'s mesh", x.rank, self.rank));
        }
        self.gc();
        self.wait_links(Instant::now() + self.timeout)?;
        let rank = self.rank;
        let results: Vec<(usize, Result<(), String>)> = std::thread::scope(|s| {
            let handles: Vec<_> = self
                .links
                .iter_mut()
                .enumerate()
                .filter(|(q, _)| *q != rank)
                .map(|(q, link)| {
                    let link = link.as_mut().expect("links checked above");
                    (q, s.spawn(move || link.send(x, q, partial)))
                })
                .collect();
            handles.into_iter().map(|(q, h)| (q, h.join().unwrap_or_else(|_| Err("send thread panicked".into())))).collect()
        });
        let mut first = None;
        for (q, r) in results {
            if let Err(e) = r {
                self.drop_link(q, &e);
                first.get_or_insert(format!("rank {rank}: {e}"));
            }
        }
        first.map_or(Ok(()), Err)
    }

    /// Wait for the peers' frames of exchange `x` until `deadline`, returning
    /// them by source rank (this rank's slot empty).
    fn collect(&mut self, x: &Exchange, deadline: Instant) -> Result<[Option<Pending>; WORLD], String> {
        let rank = self.rank;
        let mut got: [Option<Pending>; WORLD] = Default::default();
        let mut lost = [false; WORLD];
        loop {
            let mut i = 0;
            while i < self.pending.len() {
                if !x.is_for(&self.pending[i].header) {
                    i += 1;
                    continue;
                }
                let p = self.pending.swap_remove(i);
                x.check(&p.header)?;
                let src = p.header.src;
                if p.peer != src {
                    return Err(format!("rank {rank}: rank {src}'s frame on rank {}'s link", p.peer));
                }
                if got[src].replace(p).is_some() {
                    return Err(format!("rank {rank}: two frames from rank {src} (request {}, layer {})", x.request_id, x.layer));
                }
            }
            let missing: Vec<usize> = (0..WORLD).filter(|&q| q != rank && got[q].is_none()).collect();
            if missing.is_empty() {
                return Ok(got);
            }
            if let Some(&q) = missing.iter().find(|&&q| lost[q]) {
                return Err(format!("rank {rank}: the link to rank {q} was lost before its frame arrived (request {}, layer {})", x.request_id, x.layer));
            }
            let now = Instant::now();
            if now >= deadline {
                return Err(format!(
                    "rank {rank}: no frame from rank(s) {missing:?} within {:?} (request {}, layer {})",
                    self.timeout, x.request_id, x.layer
                ));
            }
            if self.links.iter().any(|l| matches!(l, Some(Link { tx: Tx::Rdma { .. }, .. }))) {
                // RDMA completions are polled; the TCP side's events are drained between polls.
                self.poll_rdma(&mut lost);
                match self.events.try_recv() {
                    Ok(ev) => self.handle(ev, &mut lost),
                    Err(TryRecvError::Empty) => std::thread::yield_now(),
                    Err(TryRecvError::Disconnected) => return Err("mesh: event channel closed".into()),
                }
            } else {
                match self.events.recv_timeout(deadline - now) {
                    Ok(ev) => self.handle(ev, &mut lost),
                    Err(RecvTimeoutError::Timeout) => {}
                    Err(RecvTimeoutError::Disconnected) => return Err("mesh: event channel closed".into()),
                }
            }
        }
    }

    /// Wait until every peer's link is up, or fail at `deadline`.
    fn wait_links(&mut self, deadline: Instant) -> Result<(), String> {
        let mut lost = [false; WORLD];
        loop {
            while let Ok(ev) = self.events.try_recv() {
                self.handle(ev, &mut lost);
            }
            let missing: Vec<usize> = (0..WORLD).filter(|&q| q != self.rank && self.links[q].is_none()).collect();
            if missing.is_empty() {
                return Ok(());
            }
            let now = Instant::now();
            if now >= deadline {
                return Err(format!("rank {}: no link to rank(s) {missing:?} within {:?}", self.rank, self.timeout));
            }
            match self.events.recv_timeout((deadline - now).min(Duration::from_millis(50))) {
                Ok(ev) => self.handle(ev, &mut lost),
                Err(RecvTimeoutError::Timeout) => {}
                Err(RecvTimeoutError::Disconnected) => return Err("mesh: event channel closed".into()),
            }
        }
    }

    fn handle(&mut self, ev: Event, lost: &mut [bool; WORLD]) {
        match ev {
            Event::Up { peer, link } => {
                eprintln!("mesh: rank {} <-> rank {peer} up ({})", self.rank, if matches!(link.tx, Tx::Rdma { .. }) { "RDMA RC" } else { "TCP" });
                self.links[peer] = Some(link);
            }
            Event::Frame { peer, header, bytes } => self.pending.push(Pending { peer, at: Instant::now(), header, bytes }),
            Event::Down { peer, id, why } => {
                if self.links[peer].as_ref().is_some_and(|l| l.id == id) {
                    self.drop_link(peer, &why);
                    lost[peer] = true;
                }
            }
        }
    }

    /// Poll every RDMA link's receive queue until it is empty.
    fn poll_rdma(&mut self, lost: &mut [bool; WORLD]) {
        for (q, lost_q) in lost.iter_mut().enumerate() {
            while let Some(Link { tx: Tx::Rdma { t, rx }, .. }) = self.links[q].as_mut() {
                let frame = match t.try_recv() {
                    Ok(Some(bytes)) => ExchangeView::parse(&bytes, WireNaive::NONE)
                        .map_err(|e| e.to_string())
                        .and_then(|v| rx.accept_seq(v.seq).map(|_| v.header).map_err(|e| e.to_string()))
                        .and_then(|h| if h.src == q { Ok(h) } else { Err(format!("a frame from rank {} on rank {q}'s link", h.src)) })
                        .map(|h| Some((h, bytes))),
                    Ok(None) => Ok(None),
                    Err(e) => Err(e),
                };
                match frame {
                    Ok(Some((header, bytes))) => self.pending.push(Pending { peer: q, at: Instant::now(), header, bytes }),
                    Ok(None) => break,
                    Err(e) => {
                        self.drop_link(q, &e);
                        *lost_q = true;
                        break;
                    }
                }
            }
        }
    }

    /// Drop a failed link; the lower rank of the pair dials again.
    fn drop_link(&mut self, peer: usize, why: &str) {
        if self.links[peer].take().is_some() {
            eprintln!("mesh: rank {} dropped its link to rank {peer}: {why}", self.rank);
            if let Some(Some(ask)) = self.redial.get(peer) {
                let _ = ask.send(());
            }
        }
    }

    /// Drop frames no exchange claimed: older than twice the timeout, or the
    /// oldest beyond [`MAX_PENDING`] of a peer.
    fn gc(&mut self) {
        let limit = 2 * self.timeout;
        let rank = self.rank;
        let stale = |p: &Pending| {
            eprintln!("mesh: rank {rank} dropped an unclaimed frame from rank {} (request {}, layer {})", p.peer, p.header.request_id, p.header.layer_id);
        };
        self.pending.retain(|p| {
            let keep = p.at.elapsed() < limit;
            if !keep {
                stale(p);
            }
            keep
        });
        for q in 0..WORLD {
            while self.pending.iter().filter(|p| p.peer == q).count() > MAX_PENDING {
                let (i, _) = self.pending.iter().enumerate().filter(|(_, p)| p.peer == q).min_by_key(|(_, p)| p.at).expect("counted above");
                stale(&self.pending.remove(i));
            }
        }
    }
}

/// Accept the lower ranks' links, until the mesh is dropped.
fn listen(rank: usize, listener: TcpListener, timeout: Duration, events: Sender<Event>, stop: Arc<AtomicBool>) {
    for conn in listener.incoming() {
        if stop.load(Ordering::SeqCst) {
            return;
        }
        let Ok(stream) = conn else { continue };
        let events = events.clone();
        // A slow or foreign peer must not hold up the others' hellos.
        std::thread::spawn(move || {
            if let Err(e) = accept(rank, stream, timeout, &events) {
                eprintln!("mesh: rank {rank} refused a peer connection: {e}");
            }
        });
    }
}

fn accept(rank: usize, mut stream: TcpStream, timeout: Duration, events: &Sender<Event>) -> Result<(), String> {
    fabric_guard(&stream)?;
    stream.set_nodelay(true).map_err(|e| e.to_string())?;
    stream.set_read_timeout(Some(HANDSHAKE_TIMEOUT)).map_err(|e| e.to_string())?;
    let (src, rdma) = read_hello(&mut stream, rank)?;
    if src > rank {
        return Err(format!("rank {src} dialled rank {rank}: the lower rank dials"));
    }
    stream.write_all(&hello(rank, src, rdma)).map_err(|e| format!("hello: {e}"))?;
    open_link(src, stream, rdma, false, timeout, events)
}

/// Keep a link to rank `peer` (above `rank`) up: dial, then wait until the
/// mesh reports it lost, and dial again; stop when the mesh is gone.
fn dialler(rank: usize, peer: usize, addr: String, rdma: bool, timeout: Duration, events: Sender<Event>, asked: Receiver<()>) {
    let mut failures = 0u64;
    loop {
        match dial(rank, peer, &addr, rdma, timeout, &events) {
            Ok(()) => {
                failures = 0;
                if asked.recv().is_err() {
                    return;
                }
            }
            Err(e) => {
                failures += 1;
                if failures.is_power_of_two() {
                    eprintln!("mesh: rank {rank} -> rank {peer} at {addr}: {e} (attempt {failures}; retrying)");
                }
                if let Err(RecvTimeoutError::Disconnected) = asked.recv_timeout(Duration::from_millis(200)) {
                    return;
                }
            }
        }
    }
}

fn dial(rank: usize, peer: usize, addr: &str, rdma: bool, timeout: Duration, events: &Sender<Event>) -> Result<(), String> {
    let sa = addr.to_socket_addrs().map_err(|e| e.to_string())?.next().ok_or("no address")?;
    let mut stream = TcpStream::connect_timeout(&sa, HANDSHAKE_TIMEOUT).map_err(|e| e.to_string())?;
    fabric_guard(&stream)?;
    stream.set_nodelay(true).map_err(|e| e.to_string())?;
    stream.set_read_timeout(Some(HANDSHAKE_TIMEOUT)).map_err(|e| e.to_string())?;
    stream.write_all(&hello(rank, peer, rdma)).map_err(|e| format!("hello: {e}"))?;
    let (src, echo) = read_hello(&mut stream, rank)?;
    if src != peer || echo != rdma {
        return Err(format!("hello: rank {src} answered at rank {peer}'s address (RDMA {echo}, asked {rdma})"));
    }
    open_link(peer, stream, rdma, true, timeout, events)
}

/// Open the transport on a stream whose hellos passed and hand the link to
/// the serving thread (then, for TCP, start its reader).
fn open_link(peer: usize, stream: TcpStream, rdma: bool, dialled: bool, timeout: Duration, events: &Sender<Event>) -> Result<(), String> {
    let id = NEXT_LINK.fetch_add(1, Ordering::Relaxed);
    let ctl = stream.try_clone().map_err(|e| e.to_string())?;
    // A peer that stops reading fails the send instead of blocking it forever.
    stream.set_write_timeout(Some(timeout)).map_err(|e| e.to_string())?;
    let seq = StreamSender::new(WireNaive::NONE);
    if rdma {
        let t = if dialled {
            let t = RdmaTransport::connect_sized(&stream, MAX_FRAME, MAX_FRAME, MESH_SLOTS).map_err(|e| e.to_string())?;
            // Both queue pairs are connected once this byte is out: the peer may send.
            ready(&ctl, true)?;
            t
        } else {
            let t = RdmaTransport::accept_sized(&stream, MAX_FRAME, MAX_FRAME, MESH_SLOTS)
                .map_err(|e| e.to_string())?
                .ok_or("the peer's hello asked for RDMA but it did not open it")?;
            ready(&ctl, false)?;
            t
        };
        let link = Link { id, tx: Tx::Rdma { t: Box::new(t), rx: StreamReceiver::new(WireNaive::NONE) }, seq, ctl };
        return events.send(Event::Up { peer, link }).map_err(|_| "the mesh is gone".into());
    }
    stream.set_read_timeout(None).map_err(|e| e.to_string())?;
    let reader = TcpTransport::from_stream(stream.try_clone().map_err(|e| e.to_string())?).map_err(|e| e.to_string())?;
    let writer = TcpTransport::from_stream(stream).map_err(|e| e.to_string())?;
    events.send(Event::Up { peer, link: Link { id, tx: Tx::Tcp(writer), seq, ctl } }).map_err(|_| "the mesh is gone")?;
    let events = events.clone();
    std::thread::Builder::new()
        .name(format!("mesh-read-{peer}"))
        .spawn(move || read_link(peer, id, reader, events))
        .map_err(|e| e.to_string())?;
    Ok(())
}

/// After the RDMA handshake the dialling side writes one byte and the
/// accepting side waits for it: the dialler's queue pair is connected before
/// the accepting side's first send.
fn ready(ctl: &TcpStream, dialled: bool) -> Result<(), String> {
    let mut s = ctl.try_clone().map_err(|e| e.to_string())?;
    s.set_nonblocking(false).map_err(|e| e.to_string())?;
    let r = if dialled {
        s.write_all(b"R").map_err(|e| format!("ready: {e}"))
    } else {
        s.set_read_timeout(Some(HANDSHAKE_TIMEOUT)).map_err(|e| e.to_string())?;
        let mut b = [0u8; 1];
        s.read_exact(&mut b).map_err(|e| format!("ready: {e}")).and_then(|_| if b == *b"R" { Ok(()) } else { Err("ready: bad byte".into()) })
    };
    // The RDMA transport watches this socket for the peer's close without blocking.
    s.set_nonblocking(true).map_err(|e| e.to_string())?;
    r
}

/// A TCP link's reader: validate each frame (CRC unless disabled, L4
/// sequence, the sending rank) and hand it on; report the link down on the
/// first failure or the close.
fn read_link(peer: usize, id: u64, mut t: TcpTransport, events: Sender<Event>) {
    let mut rx = StreamReceiver::new(WireNaive::NONE);
    let why = loop {
        let Some(bytes) = t.recv() else { break "closed".to_string() };
        let header = match ExchangeView::parse(&bytes, WireNaive::NONE) {
            Ok(v) => match rx.accept_seq(v.seq) {
                Ok(()) => v.header,
                Err(e) => break format!("L4: {e}"),
            },
            Err(e) => break e.to_string(),
        };
        if header.src != peer {
            break format!("a frame from rank {} on rank {peer}'s link", header.src);
        }
        if events.send(Event::Frame { peer, header, bytes }).is_err() {
            return;
        }
    };
    let _ = events.send(Event::Down { peer, id, why });
}

/// The exchange for a reduce-scattered request (`None` for a four-plane one).
pub fn exchange_for(view: &glm53f_wire::frame::RequestView<'_>, rank: usize) -> Option<Exchange> {
    ExchangeDtype::from_request_flags(view.flags).map(|dtype| Exchange {
        request_id: view.request_id,
        layer: view.layer_id,
        rows: view.rows,
        dtype,
        rank,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hellos_name_the_ranks_version_and_transport() {
        let h = hello(1, 3, true);
        assert_eq!((&h[..8], u16::from_le_bytes([h[8], h[9]]), h[10], h[11], h[12], h[13]), (&b"G53RMESH"[..], 4, 4, 1, 3, 1));
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let mut a = TcpStream::connect(listener.local_addr().unwrap()).unwrap();
        let (mut b, _) = listener.accept().unwrap();
        a.write_all(&h).unwrap();
        assert_eq!(read_hello(&mut b, 3).unwrap(), (1, true));
        for bad in [hello(1, 2, false), hello(3, 3, false), hello(5, 3, false)] {
            a.write_all(&bad).unwrap();
            assert!(read_hello(&mut b, 3).is_err());
        }
        let mut other = hello(1, 3, false);
        other[8] = 3; // a version-3 peer
        a.write_all(&other).unwrap();
        assert!(read_hello(&mut b, 3).is_err());
    }

    #[test]
    fn configurations_need_four_peers_and_a_rank_among_them() {
        let four = "127.0.0.1:1,127.0.0.1:2,127.0.0.1:3,127.0.0.1:4";
        assert_eq!(MeshConfig::new(2, four, DEFAULT_TIMEOUT, false).unwrap().peers.len(), 4);
        assert!(MeshConfig::new(4, four, DEFAULT_TIMEOUT, false).is_err());
        assert!(MeshConfig::new(0, "127.0.0.1:1,127.0.0.1:2", DEFAULT_TIMEOUT, false).is_err());
        assert!(MeshConfig::new(0, four, Duration::ZERO, false).is_err());
    }
}
