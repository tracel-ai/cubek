//! The software nest for an accumulator that lives in memory: seeded from the sink per visit and
//! committed back, so partials round-trip through its element.
//!
//! [`base`] resolves each operand's packing and routes to [`direct`] (2-D, one contracted axis off
//! directly addressed operands) or [`gather`] (N-D, coordinates resolved per tap).

mod base;
mod direct;
mod gather;
mod shape;

pub(crate) use base::{contract, contracted_per_step, resolve_nd_coords};
