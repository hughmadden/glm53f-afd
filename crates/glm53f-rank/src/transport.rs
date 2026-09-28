//! Byte transport + the L4 fault-injection shim (A1: a test shim at the sender
//! inside the window protocol, never live buffers).
//!
//! The L4 ladder (`StreamSender`/`StreamReceiver`) is codec-level and already
//! pinned; this module is the transport under it. [`Loopback`] is an in-memory
//! two-ended transport (the TCP stand-in for CPU tests); [`FaultInjector`]
//! corrupts the transient frame bytes at the sender, after encode, so the test
//! exercises bitflip/truncation/reorder/duplicate over a real byte channel
//! without touching any live buffer.

use std::sync::mpsc::{channel, Receiver, Sender};

/// A byte-oriented transport: send a frame's bytes, receive a frame's bytes.
/// `recv` returns `Some(frame)` for a complete frame and `None` once the peer
/// closed; it may block until a whole frame arrives (`TcpTransport` reads
/// blocking, `LoopbackEnd` polls). Transport failures are I/O errors; wire-level
/// faults are the receiver's `WireError`s.
pub trait ByteTransport {
    fn send(&mut self, bytes: Vec<u8>) -> std::io::Result<()>;
    fn recv(&mut self) -> Option<Vec<u8>>;
    /// A registered buffer the next outgoing frame can be assembled in, in place
    /// (RDMA); `None` for transports that only send owned bytes.
    fn send_buffer(&mut self) -> Option<&mut [u8]> {
        None
    }
    /// Send the first `len` bytes of [`ByteTransport::send_buffer`].
    fn send_in_place(&mut self, _len: usize) -> std::io::Result<()> {
        Err(std::io::Error::new(std::io::ErrorKind::Unsupported, "no in-place send buffer"))
    }
    /// Whether [`ByteTransport::recv_slot`] is available (RDMA): a frame can be
    /// read where the NIC landed it.
    fn in_place_recv(&self) -> bool {
        false
    }
    /// In-place receive: the next frame `(ptr, len)`, left in its registered
    /// receive slot. It stays valid, and the slot un-posted, until
    /// [`ByteTransport::release_slot`]. `None`: the peer closed.
    fn recv_slot(&mut self) -> Option<(*const u8, usize)> {
        None
    }
    /// Hand the slot from [`ByteTransport::recv_slot`] back to the NIC.
    fn release_slot(&mut self) -> std::io::Result<()> {
        Ok(())
    }
    /// The receive ring `(base, bytes)`, so the caller can page-lock it for DMA.
    fn recv_ring(&mut self) -> Option<(*mut u8, usize)> {
        None
    }
}

/// One end of a loopback transport.
pub struct LoopbackEnd {
    tx: Sender<Vec<u8>>,
    rx: Receiver<Vec<u8>>,
}

impl ByteTransport for LoopbackEnd {
    fn send(&mut self, bytes: Vec<u8>) -> std::io::Result<()> {
        self.tx
            .send(bytes)
            .map_err(|_| std::io::Error::other("loopback peer dropped"))
    }

    fn recv(&mut self) -> Option<Vec<u8>> {
        self.rx.try_recv().ok()
    }
}

/// Any `&mut T` is a transport when `T` is, so shims can wrap a borrowed end.
impl<T: ByteTransport + ?Sized> ByteTransport for &mut T {
    fn send(&mut self, bytes: Vec<u8>) -> std::io::Result<()> {
        (**self).send(bytes)
    }
    fn recv(&mut self) -> Option<Vec<u8>> {
        (**self).recv()
    }
}

/// A connected in-memory transport: `a` and `b` are the two ends.
pub struct Loopback {
    pub a: LoopbackEnd,
    pub b: LoopbackEnd,
}

impl Loopback {
    pub fn new() -> Self {
        let (a_tx, a_rx) = channel();
        let (b_tx, b_rx) = channel();
        Self {
            a: LoopbackEnd { tx: a_tx, rx: b_rx },
            b: LoopbackEnd { tx: b_tx, rx: a_rx },
        }
    }
}

impl Default for Loopback {
    fn default() -> Self {
        Self::new()
    }
}

/// A one-shot byte-corruption fault to inject at the sender.
#[derive(Debug, Clone, Copy)]
pub enum Fault {
    /// Flip `mask` bits at byte `byte` (clamped to the frame length).
    Bitflip { byte: usize, mask: u8 },
    /// Truncate the frame to `keep` bytes.
    Truncate { keep: usize },
}

/// A sender-side fault-injection shim: corrupts the next frame's transient bytes
/// (post-encode, pre-transmit), then passes through. Never touches live buffers.
pub struct FaultInjector<T> {
    inner: T,
    pending: Option<Fault>,
}

impl<T> FaultInjector<T> {
    pub fn new(inner: T) -> Self {
        Self { inner, pending: None }
    }

    /// Queue a fault for the next send.
    pub fn inject(&mut self, fault: Fault) {
        self.pending = Some(fault);
    }
}

impl<T: ByteTransport> ByteTransport for FaultInjector<T> {
    fn send(&mut self, mut bytes: Vec<u8>) -> std::io::Result<()> {
        if let Some(fault) = self.pending.take() {
            match fault {
                Fault::Bitflip { byte, mask } => {
                    if byte < bytes.len() {
                        bytes[byte] ^= mask;
                    }
                }
                Fault::Truncate { keep } => bytes.truncate(keep),
            }
        }
        self.inner.send(bytes)
    }

    fn recv(&mut self) -> Option<Vec<u8>> {
        self.inner.recv()
    }
}

/// A real TCP transport with wire-protocol framing. Reads are BLOCKING and read
/// directly into the frame buffer (the 128-byte header first, then the body
/// sized by the header's `wire_bytes`); `send` writes the full frame bytes with
/// Nagle disabled. The wire protocol is the framing layer, so no length prefix
/// is added. Socket buffers are left to Linux TCP autotuning (an explicit
/// SO_RCVBUF/SO_SNDBUF would disable autotuning and be clamped to the low
/// `net.core.rmem_max` on the Sparks).
pub struct TcpTransport {
    stream: std::net::TcpStream,
    /// True once a read observes EOF (the peer closed the connection).
    closed: bool,
}

impl TcpTransport {
    /// Connect to `addr` (blocking socket, Nagle off).
    pub fn connect(addr: &str) -> std::io::Result<Self> {
        let stream = std::net::TcpStream::connect(addr)?;
        Self::from_stream(stream)
    }

    /// Wrap an existing stream: disable Nagle and keep the stream BLOCKING so
    /// `recv` reads a whole frame with `read_exact` (no yield busy-poll between
    /// partial reads). No SO_RCVBUF/SO_SNDBUF — Linux autotuning grows them up
    /// to `tcp_rmem`/`tcp_wmem` max, which exceeds an explicit request clamped
    /// to `net.core.rmem_max` (208 KB on the Sparks).
    pub fn from_stream(stream: std::net::TcpStream) -> std::io::Result<Self> {
        stream.set_nodelay(true)?;
        Ok(Self { stream, closed: false })
    }

    /// Whether the peer has closed the connection (distinguishes a partial
    /// frame from a dead peer).
    pub fn peer_closed(&self) -> bool {
        self.closed
    }
}

impl ByteTransport for TcpTransport {
    fn send(&mut self, bytes: Vec<u8>) -> std::io::Result<()> {
        use std::io::Write;
        // Blocking one-shot write: a 33 MB return exceeds the socket buffer, so
        // `write_all` blocks until the peer drains it (mirror of 3c3c7af).
        self.stream.write_all(&bytes)
    }

    fn recv(&mut self) -> Option<Vec<u8>> {
        use std::io::Read;
        // Blocking frame read: the 128-byte header first, then the body sized by
        // the header's wire_bytes, read directly into the frame buffer. Returns
        // None only on EOF before a complete frame (the peer closed). The header
        // is read with a manual loop so the first-byte arrival is timestamped for
        // the cross-host timeline.
        let mut header = [0u8; glm53f_wire::HEADER_LEN];
        let mut got = 0usize;
        while got < glm53f_wire::HEADER_LEN {
            match self.stream.read(&mut header[got..]) {
                Ok(0) => {
                    self.closed = true;
                    return None;
                }
                Ok(n) => {
                    if got == 0 {
                        crate::timeline::tl("req_first_byte", None);
                    }
                    got += n;
                }
                Err(e) if e.kind() == std::io::ErrorKind::Interrupted => continue,
                Err(_) => {
                    self.closed = true;
                    return None;
                }
            }
        }
        // The DS41RTE3 header carries the total wire length at byte 76 (u64).
        let wb = u64::from_le_bytes(header[76..84].try_into().unwrap()) as usize;
        if wb < glm53f_wire::HEADER_LEN {
            self.closed = true;
            return None;
        }
        let mut frame = vec![0u8; wb];
        frame[..glm53f_wire::HEADER_LEN].copy_from_slice(&header);
        if self.stream.read_exact(&mut frame[glm53f_wire::HEADER_LEN..]).is_err() {
            self.closed = true;
            return None;
        }
        crate::timeline::tl("req_last_byte", None);
        Some(frame)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use glm53f_wire::frame::{HiddenRow, RequestFrame, ReturnFrame, ReturnRow, RouteEntry, RowDescriptor};
    use glm53f_wire::l4::{retry_until, StreamReceiver, StreamSender};
    use glm53f_wire::layout::Status;
    use glm53f_wire::{Frame, SourceKind, WireNaive};

    fn request() -> RequestFrame {
        RequestFrame {
            request_id: 1,
            placement_version: 1,
            layer_id: 1,
            executor_id: 0,
            source_kind: SourceKind::Decode,
            token_position: 0,
            flags: 0,
            seq: 0,
            rows: vec![RowDescriptor {
                row_id: 0,
                source_kind: SourceKind::Decode,
                source_request_id: 1,
                token_position: 0,
                route_offset: 0,
                route_count: 1,
            }],
            routes: vec![RouteEntry { row_index: 0, expert_id: 0, gate_weight: 1.0 }],
            hidden_rows: vec![glm53f_wire::frame::HiddenRow {
                payload: vec![0u8; 4096],
                scales: vec![0u8; 128],
            }],
        }
    }

    /// The L4 ladder over the loopback: send a frame, corrupt it at the sender,
    /// and confirm the receiver classifies it — then that a retry recovers.
    #[test]
    fn bitflip_is_detected_and_retry_recovers() {
        let mut loopback = Loopback::new();
        let mut sender = StreamSender::new(WireNaive::NONE);
        let mut rx = StreamReceiver::new(WireNaive::NONE);

        let clean = sender.encode_request(&request()).expect("encode");

        // Send a bitflipped frame through the shim: the receiver must report
        // Corrupt (Retry disposition), never accept it.
        let mut shim = FaultInjector::new(&mut loopback.b);
        shim.inject(Fault::Bitflip { byte: 300, mask: 0x01 });
        shim.send(clean.clone()).expect("send corrupt");

        let outcome = loopback.a.recv().expect("frame arrived");
        match rx.accept(&outcome) {
            Err(e) => assert_eq!(e.disposition(), glm53f_wire::Disposition::Retry, "{e:?}"),
            Ok(_) => panic!("corrupt frame must not be accepted"),
        }

        // A clean retransmit (same seq is fine — the receiver never advanced)
        // must be accepted.
        loopback.b.send(clean).expect("retransmit");
        let outcome = loopback.a.recv().expect("retransmit arrived");
        assert!(matches!(rx.accept(&outcome), Ok(Frame::Request(_))));
    }

    #[test]
    fn truncation_is_detected() {
        let mut loopback = Loopback::new();
        let mut sender = StreamSender::new(WireNaive::NONE);
        let mut rx = StreamReceiver::new(WireNaive::NONE);
        let clean = sender.encode_request(&request()).expect("encode");

        let mut shim = FaultInjector::new(&mut loopback.b);
        shim.inject(Fault::Truncate { keep: 100 });
        shim.send(clean).expect("send truncated");

        let outcome = loopback.a.recv().expect("frame");
        match rx.accept(&outcome) {
            Err(e) => assert_eq!(e.disposition(), glm53f_wire::Disposition::Retry, "{e:?}"),
            Ok(_) => panic!("truncated frame must not be accepted"),
        }
    }

    #[test]
    fn reorder_and_duplicate_are_detected() {
        let mut loopback = Loopback::new();
        let mut sender = StreamSender::new(WireNaive::NONE);
        let mut rx = StreamReceiver::new(WireNaive::NONE);

        let f0 = sender.encode_request(&request()).expect("f0");
        let f1 = sender.encode_request(&request()).expect("f1");

        // Reorder: deliver f1 (seq 1) before f0 (seq 0) -> OutOfOrder.
        loopback.b.send(f1.clone()).expect("f1 first");
        let outcome = loopback.a.recv().expect("f1");
        assert!(matches!(
            rx.accept(&outcome),
            Err(glm53f_wire::WireError::OutOfOrder { expected: 0, got: 1 })
        ));

        // Deliver f0 (seq 0) -> accepted.
        loopback.b.send(f0.clone()).expect("f0");
        let outcome = loopback.a.recv().expect("f0");
        assert!(matches!(rx.accept(&outcome), Ok(Frame::Request(_))));

        // Duplicate: deliver f0 again (seq 0 < expected 1) -> Duplicate.
        loopback.b.send(f0).expect("f0 dup");
        let outcome = loopback.a.recv().expect("f0 dup");
        assert!(matches!(
            rx.accept(&outcome),
            Err(glm53f_wire::WireError::Duplicate { seq: 0, .. })
        ));
    }

    #[test]
    fn retry_until_recovers_from_one_bitflip() {
        let mut loopback = Loopback::new();
        let mut sender = StreamSender::new(WireNaive::NONE);
        let mut rx = StreamReceiver::new(WireNaive::NONE);
        let clean = sender.encode_request(&request()).expect("encode");

        let mut attempts = 0;
        let outcome = retry_until(3, || {
            attempts += 1;
            // First attempt injects a bitflip; later attempts are clean.
            let mut bytes = clean.clone();
            if attempts == 1 {
                bytes[300] ^= 0x01;
            }
            loopback.b.send(bytes).expect("send");
            let got = loopback.a.recv().expect("recv");
            rx.accept(&got)
        });
        assert!(matches!(outcome, Ok(Frame::Request(_))));
        assert_eq!(attempts, 2, "one corruption + one clean retransmit");
    }

    #[test]
    fn tcp_transport_delivers_the_exact_frame_bytes() {
        let listener = std::net::TcpListener::bind("127.0.0.1:0").expect("bind");
        let addr = listener.local_addr().expect("addr");
        let client_stream = std::net::TcpStream::connect(addr).expect("connect");
        let (server_stream, _) = listener.accept().expect("accept");

        let mut server_tx = TcpTransport::from_stream(server_stream).expect("server transport");
        let mut client_tx = TcpTransport::from_stream(client_stream).expect("client transport");

        let mut sender = StreamSender::new(WireNaive::NONE);
        let bytes = sender.encode_request(&request()).expect("encode");
        client_tx.send(bytes.clone()).expect("send");

        // The TCP byte stream is framed by the wire protocol's header wire_bytes;
        // recv() must return the exact frame, not a partial or concatenated read.
        let got = server_tx.recv().expect("frame");
        assert_eq!(got, bytes, "TCP must deliver the exact frame bytes");

        // A second frame round-trips too (framing stays aligned).
        let bytes2 = sender.encode_request(&request()).expect("encode 2");
        client_tx.send(bytes2.clone()).expect("send 2");
        let got2 = server_tx.recv().expect("frame 2");
        assert_eq!(got2, bytes2);
    }

    /// I5-R6 §1 negative: a rank that drops the connection mid-frame must fail
    /// the request with an error, never hang the coordinator. `recv()` must
    /// return `None` (the caller then fails the request) rather than surfacing
    /// a partial frame as a complete one or blocking forever.
    #[test]
    fn dropped_connection_never_hangs_and_never_surfaces_a_partial_frame() {
        use std::io::Write;
        let listener = std::net::TcpListener::bind("127.0.0.1:0").expect("bind");
        let addr = listener.local_addr().expect("addr");
        let mut client_stream = std::net::TcpStream::connect(addr).expect("connect");
        let (server_stream, _) = listener.accept().expect("accept");
        let mut server_tx = TcpTransport::from_stream(server_stream).expect("server transport");

        let mut sender = StreamSender::new(WireNaive::NONE);
        let full = sender.encode_request(&request()).expect("encode");
        assert!(full.len() > glm53f_wire::HEADER_LEN);

        // Write a partial frame (past the header, short of wire_bytes) then drop.
        client_stream.write_all(&full[..glm53f_wire::HEADER_LEN + 16]).expect("partial write");
        drop(client_stream); // the Spark rank drops the connection mid-frame

        // Must return None (not a partial frame, not a hang): the coordinator
        // turns None into a request error (Incomplete / connection lost).
        assert!(server_tx.recv().is_none(), "dropped connection must yield None, not a partial frame");
        // A second read also returns None — it never blocks or panics.
        assert!(server_tx.recv().is_none());
    }

    fn large_request(tokens: usize) -> RequestFrame {
        let mut rows = Vec::with_capacity(tokens);
        let mut routes = Vec::with_capacity(tokens * 8);
        let mut hidden_rows = Vec::with_capacity(tokens);
        for t in 0..tokens {
            rows.push(RowDescriptor {
                row_id: t as u64,
                source_kind: SourceKind::Decode,
                source_request_id: 1,
                token_position: t as u64,
                route_offset: (t * 8) as u32,
                route_count: 8,
            });
            for k in 0..8u32 {
                routes.push(RouteEntry { row_index: t as u32, expert_id: k % 8, gate_weight: 1.0 });
            }
            hidden_rows.push(HiddenRow { payload: vec![0u8; 4096], scales: vec![127u8; 128] });
        }
        RequestFrame {
            request_id: 1,
            placement_version: 1,
            layer_id: 1,
            executor_id: 0,
            source_kind: SourceKind::Decode,
            token_position: 0,
            flags: 0,
            seq: 0,
            rows,
            routes,
            hidden_rows,
        }
    }

    fn large_return(tokens: usize) -> ReturnFrame {
        let mut rows = Vec::with_capacity(tokens);
        for _ in 0..tokens {
            rows.push(ReturnRow { codes: vec![0u16; 4096] });
        }
        ReturnFrame {
            request_id: 1,
            placement_version: 1,
            layer_id: 1,
            executor_id: 0,
            token_position: 0,
            status: Status::Ok,
            flags: 0,
            route_count: 8,
            seq: 0,
            rows,
        }
    }

    /// D5 residual: after a panic mid-forward the wire must resync by
    /// reconnecting (fresh L4 sequence state on both ends), never by reusing the
    /// stale connection (mimo26f-afd's wire-resync design).
    #[test]
    fn resync_after_mid_forward_panic_is_a_reconnect() {
        use glm53f_wire::Disposition;

        // --- connection 1: a request is accepted, then the serve "panics"
        // mid-forward before producing a return. The counters have diverged. ---
        let mut lb = Loopback::new();
        let mut coord_tx = StreamSender::new(WireNaive::NONE);
        let _coord_rx = StreamReceiver::new(WireNaive::NONE);
        let mut spark_rx = StreamReceiver::new(WireNaive::NONE);

        // Coordinator sends request seq 0; the Spark accepts it (spark expected -> 1).
        let r0 = coord_tx.encode_request(&request()).expect("encode req 0");
        lb.a.send(r0).expect("send req 0");
        let got = lb.b.recv().expect("spark recv req 0");
        spark_rx.accept(&got).expect("spark accept req 0");

        // Panic mid-forward: no return is produced. coord_tx.next_seq is 1,
        // coord_rx.expected is 0, spark_rx.expected is 1 — drifted.

        // --- the desync: reusing the stale coordinator sender against a freshly
        // connected Spark receiver is OutOfOrder, never a clean resync. ---
        let mut fresh_spark_rx = StreamReceiver::new(WireNaive::NONE);
        let r1 = coord_tx.encode_request(&request()).expect("encode req 1"); // seq 1
        lb.a.send(r1).expect("send req 1");
        let got = lb.b.recv().expect("fresh spark recv");
        match fresh_spark_rx.accept(&got) {
            Err(e) => assert_eq!(e.disposition(), Disposition::Retry, "{e:?}"),
            Ok(_) => panic!("a stale seq must not be accepted by a fresh receiver"),
        }

        // --- the resync: reconnect both ends (fresh L4 state, seq 0), then a
        // full request -> return round trip succeeds. ---
        let mut lb2 = Loopback::new();
        let mut coord_tx2 = StreamSender::new(WireNaive::NONE);
        let mut coord_rx2 = StreamReceiver::new(WireNaive::NONE);
        let mut spark_rx2 = StreamReceiver::new(WireNaive::NONE);
        let mut spark_tx2 = StreamSender::new(WireNaive::NONE);

        let req = coord_tx2.encode_request(&request()).expect("encode fresh req");
        lb2.a.send(req).expect("send fresh req");
        let got = lb2.b.recv().expect("spark recv fresh req");
        let frame = match spark_rx2.accept(&got).expect("spark accept fresh req") {
            Frame::Request(r) => r,
            Frame::Return(_) => panic!("expected request"),
        };

        let ret = ReturnFrame {
            request_id: frame.request_id,
            placement_version: frame.placement_version,
            layer_id: frame.layer_id,
            executor_id: frame.executor_id,
            token_position: frame.token_position,
            status: Status::Ok,
            flags: glm53f_wire::FLAG_RETURN_REQUIRED,
            route_count: 1,
            seq: 0,
            rows: frame.rows.iter().map(|_| ReturnRow { codes: vec![0u16; 4096] }).collect(),
        };
        let stamped = spark_tx2.encode_return(&ret).expect("encode return");
        lb2.b.send(stamped).expect("send return");
        let got = lb2.a.recv().expect("coord recv return");
        match coord_rx2.accept(&got).expect("coord accept return") {
            Frame::Return(r) => assert_eq!(r.request_id, frame.request_id),
            Frame::Request(_) => panic!("expected return"),
        }
    }

    /// Read one complete frame, retrying across partial reads (the daemon's
    /// loop pattern for large multi-segment frames).
    fn recv_all(t: &mut TcpTransport) -> Vec<u8> {
        loop {
            match t.recv() {
                Some(b) => return b,
                None => {
                    assert!(!t.peer_closed(), "peer closed before a complete frame");
                    std::thread::yield_now();
                }
            }
        }
    }

    /// A 4,096-token request (~17.9 MB) and return (~33.5 MB) must round-trip
    /// exactly over the real TCP transport: `send` is a blocking one-shot write
    /// (the non-blocking `write_all` would EAGAIN on a 33 MB frame), and `recv`
    /// completes across multiple TCP segments via `recv_all`. The writer and the
    /// reader run concurrently — a blocking large write needs the peer to drain.
    #[test]
    fn large_frames_round_trip_over_tcp() {
        let listener = std::net::TcpListener::bind("127.0.0.1:0").expect("bind");
        let addr = listener.local_addr().expect("addr");
        let client_stream = std::net::TcpStream::connect(addr).expect("connect");
        let (server_stream, _) = listener.accept().expect("accept");
        let mut server_tx = TcpTransport::from_stream(server_stream).expect("server transport");
        let mut client_tx = TcpTransport::from_stream(client_stream).expect("client transport");

        let mut sender = StreamSender::new(WireNaive::NONE);
        let req = sender.encode_request(&large_request(4096)).expect("encode large request");
        let ret = sender.encode_return(&large_return(4096)).expect("encode large return");
        assert!(req.len() > 16_000_000, "request is {} B", req.len());
        assert!(ret.len() > 32_000_000, "return is {} B", ret.len());

        let req2 = req.clone();
        let ret2 = ret.clone();
        let client = std::thread::spawn(move || {
            client_tx.send(req2).expect("send large request");
            recv_all(&mut client_tx)
        });

        assert_eq!(recv_all(&mut server_tx), req, "large request round-trip");
        server_tx.send(ret.clone()).expect("send large return");
        let got = client.join().expect("client thread");
        assert_eq!(got, ret2, "large return round-trip");
    }
}

/// RDMA RC transport (perf reset R2 part 2): frames arrive by SEND into a
/// two-slot registered receive ring and leave by SEND from a registered buffer;
/// completions are busy-polled. The TCP socket stays open for peer-shutdown
/// detection only. Selected per connection by the coordinator's handshake.
///
/// The ranks' peer mesh (`crate::mesh`) uses the same transport, sized for its
/// exchange frames: the rank that connects opens it ([`RdmaTransport::connect_sized`],
/// the coordinator's side of the handshake) and polls it without blocking
/// ([`RdmaTransport::try_recv`]).
pub struct RdmaTransport {
    ep: glm53f_rdma::Endpoint,
    recv: glm53f_rdma::AlignedBuf,
    /// Two send halves of `half` bytes: a return is posted without waiting,
    /// and the next one is built in the other half (perf reset: the ~1.2 ms
    /// send completion leaves the critical path).
    send: glm53f_rdma::AlignedBuf,
    half: usize,
    send_cur: usize,
    send_pending: [bool; 2],
    /// Receive slot lent out by `recv_slot`, re-posted by `release_slot`.
    held: Option<u32>,
    stream: std::net::TcpStream,
}

/// Largest request frame (4,096 rows, the kernel's launch cap) and return
/// frame, page-rounded.
const RDMA_REQ_SLOT: usize = (glm53f_wire::HEADER_LEN + 4096 * glm53f_wire::layout::REQUEST_ROW_BYTES).div_ceil(4096) * 4096;
const RDMA_RET_MAX: usize = (glm53f_wire::HEADER_LEN + 4096 * glm53f_wire::layout::RETURN_ROW_BYTES).div_ceil(4096) * 4096;

/// The RoCE v2 device, port and GID index that own `stream`'s local IPv4 (the
/// inference-fabric rule of `glm53f_rdma::fabric_port`; loopback and the test
/// override fall back to any RoCE v2 GID of that address).
fn roce_for(stream: &std::net::TcpStream) -> std::io::Result<(String, u8, i32)> {
    use std::io::{Error, ErrorKind};
    let ip = match stream.local_addr()? {
        std::net::SocketAddr::V4(a) => *a.ip(),
        a => return Err(Error::new(ErrorKind::Unsupported, format!("rdma: IPv6 {a}"))),
    };
    match glm53f_rdma::fabric_port(std::net::IpAddr::V4(ip)) {
        Ok(Some((d, p, g, _))) => Ok((d, p, g)),
        Ok(None) => glm53f_rdma::find_roce_v2(ip)
            .ok_or_else(|| Error::new(ErrorKind::NotFound, format!("rdma: no RoCE v2 GID for {ip}"))),
        Err(e) => Err(Error::new(ErrorKind::PermissionDenied, e)),
    }
}

impl RdmaTransport {
    /// If the peer opened with the RDMA handshake, complete it and return the
    /// transport; `Ok(None)` means a plain TCP peer (nothing was consumed).
    pub fn accept(stream: &std::net::TcpStream) -> std::io::Result<Option<Self>> {
        Self::accept_sized(stream, RDMA_REQ_SLOT, RDMA_RET_MAX)
    }

    /// [`RdmaTransport::accept`] with two receive slots of `recv_slot` bytes
    /// and two send halves of `send_half` bytes (both page-rounded).
    pub fn accept_sized(stream: &std::net::TcpStream, recv_slot: usize, send_half: usize) -> std::io::Result<Option<Self>> {
        use glm53f_rdma::{AlignedBuf, Endpoint, Info, HANDSHAKE_LEN, HANDSHAKE_MAGIC};
        use std::io::{Error, ErrorKind, Read, Write};
        // Blocking peek, no timeout: a TCP coordinator may connect long before its
        // first frame. Wait for all 8 magic bytes (or EOF) before deciding.
        let mut peek = [0u8; 8];
        stream.set_read_timeout(None)?;
        let n = loop {
            let n = stream.peek(&mut peek)?;
            if n >= 8 || n == 0 {
                break n;
            }
            std::thread::sleep(std::time::Duration::from_millis(1));
        };
        if n < 8 || &peek != HANDSHAKE_MAGIC {
            return Ok(None);
        }
        let mut s = stream.try_clone()?;
        let mut hello = [0u8; HANDSHAKE_LEN];
        s.read_exact(&mut hello)?;
        s.set_read_timeout(None)?;
        let remote = Info::from_bytes(&hello[8..]).ok_or_else(|| Error::new(ErrorKind::InvalidData, "short rdma hello"))?;
        let (dev, port, gid) = roce_for(stream)?;
        let (recv_slot, half) = (recv_slot.div_ceil(4096) * 4096, send_half.div_ceil(4096) * 4096);
        let mut recv = AlignedBuf::new(2 * recv_slot);
        let mut send = AlignedBuf::new(2 * half);
        let mut ep = Endpoint::open(&dev, port, gid, Some(&mut send), &mut recv, 2, None)
            .map_err(|e| Error::new(ErrorKind::Other, e))?;
        ep.post_recv(0).map_err(|e| Error::new(ErrorKind::Other, e))?;
        ep.post_recv(1).map_err(|e| Error::new(ErrorKind::Other, e))?;
        ep.connect(&remote).map_err(|e| Error::new(ErrorKind::Other, e))?;
        let mut reply = Vec::with_capacity(HANDSHAKE_LEN);
        reply.extend_from_slice(HANDSHAKE_MAGIC);
        reply.extend_from_slice(&ep.local_info().to_bytes());
        s.write_all(&reply)?;
        eprintln!("rdma: RC on {dev} port {port} gid {gid} (qpn {} -> {})", ep.local_info().qpn, remote.qpn);
        stream.set_nonblocking(true)?;
        Ok(Some(Self { ep, recv, send, half, send_cur: 0, send_pending: [false; 2], held: None, stream: s }))
    }

    /// The connecting side of the handshake (as the coordinator's wire client
    /// does it): open an RC queue pair on the RoCE device that owns `stream`'s
    /// local IPv4, post both receive slots, send `M26RDMA1` and this side's
    /// queue-pair coordinates, and connect to the answer. Sizes as
    /// [`RdmaTransport::accept_sized`].
    pub fn connect_sized(stream: &std::net::TcpStream, recv_slot: usize, send_half: usize) -> std::io::Result<Self> {
        use glm53f_rdma::{AlignedBuf, Endpoint, Info, HANDSHAKE_LEN, HANDSHAKE_MAGIC};
        use std::io::{Error, ErrorKind, Read, Write};
        let (dev, port, gid) = roce_for(stream)?;
        let (recv_slot, half) = (recv_slot.div_ceil(4096) * 4096, send_half.div_ceil(4096) * 4096);
        let mut recv = AlignedBuf::new(2 * recv_slot);
        let mut send = AlignedBuf::new(2 * half);
        let mut ep = Endpoint::open(&dev, port, gid, Some(&mut send), &mut recv, 2, None)
            .map_err(|e| Error::new(ErrorKind::Other, e))?;
        ep.post_recv(0).map_err(|e| Error::new(ErrorKind::Other, e))?;
        ep.post_recv(1).map_err(|e| Error::new(ErrorKind::Other, e))?;
        let mut s = stream.try_clone()?;
        let mut msg = Vec::with_capacity(HANDSHAKE_LEN);
        msg.extend_from_slice(HANDSHAKE_MAGIC);
        msg.extend_from_slice(&ep.local_info().to_bytes());
        s.write_all(&msg)?;
        let mut reply = [0u8; HANDSHAKE_LEN];
        s.read_exact(&mut reply)?;
        if &reply[..8] != HANDSHAKE_MAGIC {
            return Err(Error::new(ErrorKind::InvalidData, "rdma: the peer answered without the RDMA magic"));
        }
        let remote = Info::from_bytes(&reply[8..]).ok_or_else(|| Error::new(ErrorKind::InvalidData, "short rdma hello"))?;
        ep.connect(&remote).map_err(|e| Error::new(ErrorKind::Other, e))?;
        eprintln!("rdma: RC on {dev} port {port} gid {gid} (qpn {} -> {})", ep.local_info().qpn, remote.qpn);
        stream.set_nonblocking(true)?;
        Ok(Self { ep, recv, send, half, send_cur: 0, send_pending: [false; 2], held: None, stream: s })
    }

    /// A received frame if one is waiting (copied out, its slot re-posted),
    /// without blocking; an error once the peer closed its TCP side or the
    /// queue pair failed.
    pub fn try_recv(&mut self) -> Result<Option<Vec<u8>>, String> {
        match self.ep.wait_recv(std::time::Duration::ZERO, Some(std::time::Duration::ZERO))? {
            Some((slot, len)) => {
                let base = slot as usize * self.ep.slot_len();
                let frame = self.recv.as_slice()[base..base + len].to_vec();
                self.ep.post_recv(slot)?;
                Ok(Some(frame))
            }
            None if self.peer_gone() => Err("rdma: the peer closed the connection".into()),
            None => Ok(None),
        }
    }

    /// Wait for the next receive completion: `Some((slot, len))`, or `None`
    /// when the peer closed or the queue pair failed.
    fn next_frame(&mut self) -> Option<(u32, usize)> {
        loop {
            // Spin 5 ms (decode layers arrive every ~1-2 ms), then back off; wake
            // every 100 ms to notice a closed TCP side.
            match self.ep.wait_recv(std::time::Duration::from_millis(5), Some(std::time::Duration::from_millis(100))) {
                Ok(Some(x)) => return Some(x),
                Ok(None) => {
                    if self.peer_gone() {
                        return None;
                    }
                }
                Err(e) => {
                    eprintln!("rdma: recv failed: {e}");
                    return None;
                }
            }
        }
    }

    /// True once the coordinator closed its TCP side (EOF) or errored.
    fn peer_gone(&self) -> bool {
        let mut b = [0u8; 1];
        match self.stream.peek(&mut b) {
            Ok(0) => true,
            Ok(_) => false,
            Err(e) => e.kind() != std::io::ErrorKind::WouldBlock,
        }
    }
}

impl ByteTransport for RdmaTransport {
    fn send(&mut self, bytes: Vec<u8>) -> std::io::Result<()> {
        use std::io::{Error, ErrorKind};
        let buf = self.send_buffer().ok_or_else(|| Error::new(ErrorKind::Other, "rdma: no send buffer"))?;
        if bytes.len() > buf.len() {
            return Err(Error::new(ErrorKind::InvalidInput, "rdma: return frame exceeds the send buffer"));
        }
        buf[..bytes.len()].copy_from_slice(&bytes);
        self.send_in_place(bytes.len())
    }

    /// The half the next return goes in. It is reused only after its previous
    /// SEND completed: RC completions arrive in posting order, so with the two
    /// halves alternating, the oldest outstanding completion is this half's.
    fn send_buffer(&mut self) -> Option<&mut [u8]> {
        let i = self.send_cur;
        if self.send_pending[i] {
            if let Err(e) = self.ep.wait_send(Some(std::time::Duration::from_secs(30))) {
                eprintln!("rdma: send completion failed: {e}");
                return None;
            }
            self.send_pending[i] = false;
        }
        let half = self.half;
        Some(&mut self.send.as_mut_slice()[i * half..(i + 1) * half])
    }

    /// Post the first `len` bytes of the current half without waiting for the
    /// completion ([`ByteTransport::send_buffer`] reaps it before reuse).
    fn send_in_place(&mut self, len: usize) -> std::io::Result<()> {
        use std::io::{Error, ErrorKind};
        if len > self.half {
            return Err(Error::new(ErrorKind::InvalidInput, "rdma: in-place frame exceeds the send buffer"));
        }
        let i = self.send_cur;
        self.ep.post_send(None, Some((i * self.half, len))).map_err(|e| Error::new(ErrorKind::Other, e))?;
        self.send_pending[i] = true;
        self.send_cur ^= 1;
        Ok(())
    }

    fn recv(&mut self) -> Option<Vec<u8>> {
        let (slot, len) = self.next_frame()?;
        let base = slot as usize * self.ep.slot_len();
        let frame = self.recv.as_slice()[base..base + len].to_vec();
        if let Err(e) = self.ep.post_recv(slot) {
            eprintln!("rdma: re-post recv failed: {e}");
            return None;
        }
        Some(frame)
    }

    fn in_place_recv(&self) -> bool {
        true
    }

    fn recv_slot(&mut self) -> Option<(*const u8, usize)> {
        if self.held.is_some() {
            eprintln!("rdma: recv_slot while a slot is still held");
            return None;
        }
        let (slot, len) = self.next_frame()?;
        self.held = Some(slot);
        // SAFETY: within the registered ring; the slot is not re-posted (so the
        // NIC cannot write it) until release_slot.
        Some((unsafe { self.recv.as_ptr().add(slot as usize * self.ep.slot_len()) } as *const u8, len))
    }

    fn release_slot(&mut self) -> std::io::Result<()> {
        if let Some(slot) = self.held.take() {
            self.ep.post_recv(slot).map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;
        }
        Ok(())
    }

    fn recv_ring(&mut self) -> Option<(*mut u8, usize)> {
        Some((self.recv.as_ptr(), self.recv.len()))
    }
}

impl Drop for RdmaTransport {
    /// Let posted returns (e.g. an error Status sent just before the connection
    /// closes) finish before the queue pair is destroyed.
    fn drop(&mut self) {
        for i in 0..2 {
            if self.send_pending[i] {
                let _ = self.ep.wait_send(Some(std::time::Duration::from_secs(2)));
                self.send_pending[i] = false;
            }
        }
    }
}
