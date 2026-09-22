//! Accumulators: how one is opened over a tile ([`base`]) and how its cells reach memory
//! ([`drain`]).

mod base;
mod drain;

pub use base::*;
pub(crate) use drain::*;
