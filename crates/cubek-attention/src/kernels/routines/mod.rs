//! The attention routines, one folder or file per scope of the microkernel each is written
//! around, and in a scope's folder one routine for every plane microkernel it holds.

mod cube;
mod hypercube;
mod plane;
mod unit;

pub use cube::*;
pub use hypercube::*;
pub use plane::*;
pub use unit::*;
