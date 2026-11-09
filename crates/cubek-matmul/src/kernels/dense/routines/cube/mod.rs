//! Cube routines: a cube owns a box of the output and stages it for its planes
//! ([`CubeStagedMatmulRoutine`]), or every row of a run of the output's columns, its planes
//! splitting `K` ([`CubeShuffleMatmulRoutine`]).

mod shuffle;
mod staged;

pub use shuffle::*;
pub use staged::*;
