//! The rank's wire server loop: recv → L4 accept → serve → stamp → send.
//!
//! Ported from mimo26f-afd v1.2.0 `crates/mimo26-spark/src/server.rs`. One
//! [`RankServer::step`] pulls a frame off the transport, applies the L4
//! sequence policy (`StreamReceiver`), serves the request through an
//! [`ExpertKernel`] and the layer the request names, stamps the return with
//! the next sequence (`StreamSender`), and asserts the live 8 KB per-row
//! return. L4 faults surface as [`ServeError::L4`] for the window protocol to
//! retry/fail loud; a serve failure is a [`ServeError::Serve`].

use std::fmt;

use glm53f_wire::frame::Frame;
use glm53f_wire::l4::{StreamReceiver, StreamSender};
use glm53f_wire::{WireError, WireNaive};

use crate::kernel::ExpertKernel;
use crate::serve::Layers;
use crate::transport::ByteTransport;

/// The rank-side wire server.
pub struct RankServer<T: ByteTransport> {
    transport: T,
    rx: StreamReceiver,
    tx: StreamSender,
}

/// A server-loop failure: an L4 fault (retry/drop class), a serve failure, or a
/// transport I/O error.
#[derive(Debug)]
pub enum ServeError {
    L4(WireError),
    Serve(String),
    Io(std::io::Error),
}

impl fmt::Display for ServeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ServeError::L4(e) => write!(f, "L4: {e}"),
            ServeError::Serve(e) => write!(f, "serve: {e}"),
            ServeError::Io(e) => write!(f, "io: {e}"),
        }
    }
}

impl std::error::Error for ServeError {}

impl<T: ByteTransport> RankServer<T> {
    pub fn new(transport: T, naive: WireNaive) -> Self {
        Self { transport, rx: StreamReceiver::new(naive), tx: StreamSender::new(naive) }
    }

    /// One non-blocking step. `Ok(None)` when no frame is ready; `Ok(Some(seq))`
    /// after serving one request (the stamped return sequence); `Err` on a
    /// detected L4 fault, a serve failure (including a layer that is not
    /// resident), or a transport I/O error.
    pub fn step<K: ExpertKernel>(&mut self, kernel: &mut K, layers: &Layers<K::Layer>) -> Result<Option<u64>, ServeError> {
        let Some(bytes) = self.transport.recv() else {
            return Ok(None);
        };
        let frame = self.rx.accept(&bytes).map_err(ServeError::L4)?;
        let req = match frame {
            Frame::Request(r) => r,
            Frame::Return(_) => return Err(ServeError::L4(WireError::BadKind(glm53f_wire::layout::KIND_RETURN))),
        };
        let layer = layers
            .get(req.layer_id)
            .ok_or_else(|| ServeError::Serve(format!("layer {} is not resident", req.layer_id)))?;
        let ret = crate::serve::serve_return(&req, kernel, layer, &mut crate::serve::Timings::default())
            .map_err(ServeError::Serve)?;
        let stamped_seq = self.tx.next_seq();
        let stamped = self.tx.encode_return(&ret).map_err(ServeError::L4)?;
        // Live 8 KB-per-row assertion (the compact return contract).
        assert_eq!(
            stamped.len(),
            glm53f_wire::HEADER_LEN + ret.rows.len() * glm53f_wire::RETURN_ROW_BYTES,
            "return frame is not 8,192 B per token"
        );
        self.transport.send(stamped).map_err(ServeError::Io)?;
        Ok(Some(stamped_seq))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::serve::tests::{request, SumKernel};
    use glm53f_wire::l4::StreamReceiver;
    use glm53f_wire::Frame;

    /// The full server loop over the loopback: coordinator sends two requests,
    /// the server serves both, and the coordinator's receiver accepts the
    /// stamped returns in order.
    #[test]
    fn server_serves_two_requests_over_the_loopback() {
        let mut lb = crate::transport::Loopback::new();
        let mut coord_tx = glm53f_wire::l4::StreamSender::new(WireNaive::NONE);
        let mut coord_rx = StreamReceiver::new(WireNaive::NONE);
        let mut kernel = SumKernel::default();
        let mut layers = Layers::default();
        layers.insert(3, ());

        let mut server = RankServer::new(&mut lb.b, WireNaive::NONE);

        let r0 = coord_tx.encode_request(&request(0, 3, 1)).expect("encode req 0");
        lb.a.send(r0).expect("send req 0");
        let r1 = coord_tx.encode_request(&request(1, 3, 2)).expect("encode req 1");
        lb.a.send(r1).expect("send req 1");

        let seq0 = server.step(&mut kernel, &layers).expect("step 0").expect("served");
        let seq1 = server.step(&mut kernel, &layers).expect("step 1").expect("served");
        assert_eq!((seq0, seq1), (0, 1));
        assert_eq!(kernel.calls, 2);

        let ret0 = coord_rx.accept(&lb.a.recv().expect("ret 0")).expect("accept ret 0");
        let ret1 = coord_rx.accept(&lb.a.recv().expect("ret 1")).expect("accept ret 1");
        match (ret0, ret1) {
            (Frame::Return(r0), Frame::Return(r1)) => {
                assert_eq!((r0.request_id, r0.seq, r0.rows.len()), (0, 0, 1));
                assert_eq!((r1.request_id, r1.seq, r1.rows.len()), (1, 1, 2));
            }
            _ => panic!("expected return frames"),
        }
    }

    /// A request for a layer that is not resident fails the step, loudly.
    #[test]
    fn a_missing_layer_is_a_serve_error() {
        let mut lb = crate::transport::Loopback::new();
        let mut coord_tx = glm53f_wire::l4::StreamSender::new(WireNaive::NONE);
        let mut layers = Layers::default();
        layers.insert(3, ());
        let mut server = RankServer::new(&mut lb.b, WireNaive::NONE);
        lb.a.send(coord_tx.encode_request(&request(5, 4, 1)).unwrap()).unwrap();
        match server.step(&mut SumKernel::default(), &layers) {
            Err(ServeError::Serve(m)) => assert!(m.contains("layer 4"), "{m}"),
            other => panic!("expected a serve error, got {other:?}"),
        }
    }
}
