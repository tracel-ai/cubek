//! The reduce routines, one folder per scope of the microkernel each is written around.

mod cube;
mod hypercube;
mod plane;
mod unit;

pub use cube::*;
pub use hypercube::*;
pub use plane::*;
pub use unit::*;
