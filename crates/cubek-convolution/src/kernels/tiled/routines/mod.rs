//! The convolution routines, one folder per scope of the microkernel each is written around.

mod base;
mod cube;
mod plane;

pub use base::*;
pub use cube::*;
pub use plane::*;

#[cfg(test)]
mod test_support;
