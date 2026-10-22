//! Hypercube routines: several cubes share a box of the output, each contracting its own run of
//! `K`, the sum reaching the output once: taking turns at a carry
//! ([`HypercubeRelayMatmulRoutine`]), or added up by the last cube to arrive
//! ([`HypercubeLastArrivalMatmulRoutine`]).

mod last_arrival;
mod relay;

pub use last_arrival::*;
pub use relay::*;
