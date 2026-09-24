//! Bringing an operand's values to where an instruction reads them: where a stage lives
//! ([`base`]) and the tile it is staged into ([`stage`]), which transport moves its cells
//! ([`transport`]), the slots a walk fills ahead of
//! ([`pipeline`]), and the barriers that sequence a fill against the read it feeds
//! ([`rendezvous`]).

mod base;
mod pipeline;
mod rendezvous;
mod stage;
mod transport;

pub use base::*;
pub use pipeline::*;
pub use rendezvous::*;
