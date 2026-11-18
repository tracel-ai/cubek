//! The convolution routines, one folder per scope of the microkernel each is written around.

mod cube;
mod kind;
mod plane;
mod settings;

pub use cube::*;
pub use kind::*;
pub use plane::*;
pub use settings::*;

#[cfg(test)]
pub(crate) mod test_support;
