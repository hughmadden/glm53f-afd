//! `async_api` — non-blocking expert submit/collect API (I4 item 8; ADVISOR-I4
//! §3.2 step 8).
//!
//! I5 needs to overlap the attention of micro-batch B with the experts of
//! micro-batch A (two-batch overlap) at C >= 8 and during 2,048-chunk prefill.
//! I4 only has to make the API **non-blocking**:
//!
//! * [`ExpertClient::submit`] encodes one request frame per Spark, hands them to
//!   the transport, and returns a [`Ticket`] immediately.  It **never** drains
//!   the receive side — pinned by `submit_never_drains_the_receive_side`.
//! * [`ExpertClient::collect`] drains only what has already arrived
//!   (`Transport::try_recv` returns `None` when the queue is empty) and returns
//!   `Ok(None)` until every Spark partial for that ticket is in, then the FP32
//!   sum.  It never waits.
//!
//! The transport is a trait so LaneSim can drive it with a virtual clock and a
//! loopback-TCP implementation can land later without touching this API.

use crate::error::WireError;
use crate::frame::{HiddenRow, RequestFrame, ReturnFrame, RouteEntry, RowDescriptor};
use crate::l4::CoordinatorSum;
use crate::layout::{SourceKind, SPARKS};
use crate::naive::WireNaive;

/// Opaque handle for one submitted layer's expert work.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Ticket {
    pub id: u64,
    pub layer_id: u32,
    pub rows: usize,
    pub sparks: usize,
}

/// Non-blocking transport contract.  `try_recv` MUST return `None` rather than
/// wait; a blocking implementation is a bug (I5 overlap depends on it).
pub trait Transport {
    fn send(&mut self, frame: &RequestFrame) -> Result<(), WireError>;
    fn try_recv(&mut self) -> Option<ReturnFrame>;
}

struct Pending {
    ticket: Ticket,
    request_id: u64,
    sum: CoordinatorSum,
}

/// Coordinator-side expert client: submit now, collect later.
pub struct ExpertClient<T: Transport> {
    transport: T,
    hidden: usize,
    naive: WireNaive,
    next_id: u64,
    pending: Vec<Pending>,
}

impl<T: Transport> ExpertClient<T> {
    pub fn new(transport: T, hidden: usize, naive: WireNaive) -> Self {
        Self { transport, hidden, naive, next_id: 1, pending: Vec::new() }
    }

    pub fn new_env(transport: T, hidden: usize) -> Self {
        Self::new(transport, hidden, crate::naive::naive_from_env())
    }

    /// Encode + send one request frame per Spark and return a ticket at once.
    /// Non-blocking: the receive side is untouched.
    pub fn submit(
        &mut self,
        layer_id: u32,
        rows: &[RowDescriptor],
        routes: &[RouteEntry],
        hidden_rows: &[HiddenRow],
    ) -> Result<Ticket, WireError> {
        let request_id = self.next_id;
        self.next_id += 1;
        for executor in 0..SPARKS as u64 {
            let frame = RequestFrame {
                request_id,
                placement_version: 1,
                layer_id,
                executor_id: executor,
                source_kind: SourceKind::Decode,
                token_position: 0,
                flags: 0,
                seq: 0,
                rows: rows.to_vec(),
                routes: routes.to_vec(),
                hidden_rows: hidden_rows.to_vec(),
            };
            self.transport.send(&frame)?;
        }
        let ticket = Ticket {
            id: request_id,
            layer_id,
            rows: rows.len(),
            sparks: SPARKS,
        };
        self.pending.push(Pending {
            ticket: ticket.clone(),
            request_id,
            sum: CoordinatorSum::new(rows.len(), self.hidden, self.naive),
        });
        Ok(ticket)
    }

    /// Drain what has arrived; `Ok(None)` until the ticket is complete.
    pub fn collect(&mut self, ticket: &Ticket) -> Result<Option<Vec<f32>>, WireError> {
        while let Some(frame) = self.transport.try_recv() {
            let Some(p) = self
                .pending
                .iter_mut()
                .find(|p| p.request_id == frame.request_id && p.ticket.layer_id == frame.layer_id)
            else {
                // A frame for an unknown/retired ticket is a protocol error, not
                // something to silently drop.
                return Err(WireError::UnknownRequest { request_id: frame.request_id });
            };
            p.sum.accumulate(&frame)?;
        }
        let Some(p) = self.pending.iter().find(|p| p.ticket == *ticket) else {
            return Err(WireError::UnknownRequest { request_id: ticket.id });
        };
        if !p.sum.is_complete() {
            return Ok(None);
        }
        Ok(Some(p.sum.result()?.to_vec()))
    }

    /// Drop a completed ticket (frees its accumulator).
    pub fn retire(&mut self, ticket: &Ticket) {
        self.pending.retain(|p| p.ticket != *ticket);
    }

    pub fn pending_len(&self) -> usize {
        self.pending.len()
    }

    pub fn transport_mut(&mut self) -> &mut T {
        &mut self.transport
    }
}
