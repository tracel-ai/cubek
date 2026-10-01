//! Accumulators: opening ([`base`]), draining ([`drain`]), the shared-memory destinations
//! planes meet in, atomically ([`smem_accumulation`]) or in cyclic rounds ([`smem_cyclic`]), and a
//! cube's output as its planes reach it ([`planes_output`]).

pub(crate) mod base;
pub(crate) mod drain;
pub(crate) mod planes_output;
pub(crate) mod smem_accumulation;
pub(crate) mod smem_cyclic;

pub(crate) use base::*;
pub(crate) use drain::*;
