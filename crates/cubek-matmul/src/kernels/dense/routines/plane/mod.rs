//! Plane routines: a plane owns a tile of the output, contracted by an instruction
//! ([`PlaneFragmentMatmulRoutine`]), or every row of a run of the output's columns, its units
//! splitting `K` ([`PlaneShuffleMatmulRoutine`]); and the dequantize that decodes a packed weight
//! into a float one for the float product ([`PlaneDequantizeMatmulRoutine`]).

mod dequantize;
mod fragment;
mod shuffle;

pub use dequantize::*;
pub use fragment::*;
pub use shuffle::*;
