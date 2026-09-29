//! Staging an operand: storage, transports, pipelined slots and the barriers between fill and read.

mod base;
mod pipeline;
mod rendezvous;
mod staged;
mod transport;

pub use base::*;
pub use pipeline::*;
pub use rendezvous::*;
pub(crate) use staged::StageElement;
pub(crate) use transport::*;
