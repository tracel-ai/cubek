//! The convolution routines, one folder per scope of the microkernel each is written around.

mod base;
mod cube;
mod settings;

pub use base::*;
pub use cube::*;
pub use settings::*;

#[cfg(test)]
pub(crate) mod test_support;
