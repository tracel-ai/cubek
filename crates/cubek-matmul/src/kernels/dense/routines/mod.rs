//! The float product's routines, a folder per scope of the microkernel each places, and a folder
//! per routine inside it.

mod cube;
mod hypercube;
mod plane;
mod unit;

pub use cube::*;
pub use hypercube::*;
pub use plane::*;
pub use unit::*;
