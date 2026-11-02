//! The cube scope: a cube owns a row, its planes splitting it and meeting in shared memory, or a
//! cube holds several rows, a group of its units each ([`CubeGroupsRmsNormRoutine`]).

mod base;
mod groups;
mod kernel;

pub use base::*;
pub use groups::*;
