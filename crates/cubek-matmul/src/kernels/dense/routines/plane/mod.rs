//! Plane routines: a plane owns a tile of the output, contracted by an instruction
//! ([`PlaneFragmentMatmulRoutine`]), or every row of a run of the output's columns, its units
//! splitting `K` ([`PlaneShuffleMatmulRoutine`]).

mod fragment;
mod shuffle;

pub use fragment::*;
pub use shuffle::*;
