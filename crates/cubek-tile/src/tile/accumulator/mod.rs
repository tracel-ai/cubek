//! Accumulators: opening ([`base`]), draining ([`drain`]) and the shared-memory destination
//! ([`smem_accumulation`]).

pub(crate) mod base;
pub(crate) mod drain;
pub(crate) mod smem_accumulation;

pub(crate) use base::*;
pub(crate) use drain::*;
