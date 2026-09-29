//! Accumulators: opening ([`base`]), draining ([`drain`]) and the shared-memory destination
//! ([`smem_accumulation`]).

mod base;
mod drain;
mod smem_accumulation;

pub use base::*;
pub(crate) use drain::*;
pub use smem_accumulation::*;
