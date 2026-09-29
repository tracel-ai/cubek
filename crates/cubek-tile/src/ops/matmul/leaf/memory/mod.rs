//! The register nest for an accumulator in memory, seeded from and committed to the sink per
//! visit.

mod base;
mod direct;
mod gather;
mod shape;

pub(crate) use base::{contract, contracted_per_step, resolve_nd_coords};
