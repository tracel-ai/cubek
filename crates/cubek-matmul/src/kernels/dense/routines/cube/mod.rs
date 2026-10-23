//! Cube routines: a cube owns a box of the output and stages it for its planes
//! ([`CubeStagedMatmulRoutine`]), every row of a run of the output's columns, its planes
//! splitting `K` ([`CubeShuffleMatmulRoutine`]), or every row of one stored tile group of a
//! packed input, its planes taking the group's steps in turns ([`CubeTileGroupMatmulRoutine`]).

mod shuffle;
mod staged;
mod tile_group;

pub use shuffle::*;
pub use staged::*;
pub use tile_group::*;
