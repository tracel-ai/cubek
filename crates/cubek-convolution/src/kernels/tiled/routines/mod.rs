//! The convolution routines, one folder per scope of the microkernel each is written around.

mod cube;
mod kind;
mod settings;
mod unit;

pub use cube::*;
pub use kind::*;
pub use settings::*;
pub use unit::*;

#[cfg(test)]
pub(crate) mod test_support;
