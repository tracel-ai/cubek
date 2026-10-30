//! Plane routines: a plane owns every row of a run of the output's columns, its units splitting
//! `K` ([`PlaneShuffleMatmulRoutine`]); and the dequantize that decodes a packed input into a
//! float one for the float product ([`PlaneDequantizeMatmulRoutine`]).

mod dequantize;
mod shuffle;

pub use dequantize::*;
pub use shuffle::*;
