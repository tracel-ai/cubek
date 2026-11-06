//! The convolution routines, one folder per scope of the microkernel each is written around.

mod base;
mod cube;

pub use base::*;
pub use cube::*;

#[cfg(test)]
mod test_support;
