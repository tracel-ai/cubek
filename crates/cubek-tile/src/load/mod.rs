//! Bringing an operand's values to where an instruction reads them: where a stage lives
//! ([`base`]), the slots a walk fills ahead of its reads ([`pipeline`]), and the barriers that
//! sequence a fill against the read it feeds ([`rendezvous`]).

mod base;
mod pipeline;
mod rendezvous;

pub use base::*;
pub use pipeline::*;
pub use rendezvous::*;
