//! Hypercube routines: several cubes share a box of the output, each contracting its own run of
//! `K`, the sum reaching the output once, added up by the last cube to arrive: each cube a staged
//! cube ([`HypercubeLastArrivalMatmulRoutine`]) or one plane of the shuffle
//! ([`HypercubeLastArrivalShuffleMatmulRoutine`]).

mod blueprint;
mod last_arrival;
mod last_arrival_shuffle;

pub use last_arrival::*;
pub use last_arrival_shuffle::*;
