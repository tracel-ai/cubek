//! How the matmul family is measured: the tuning [`key`] that makes two products the same product,
//! and the candidate every float routine is listed as. A caller builds its tables from them.

mod candidate;
pub mod key;
#[cfg(test)]
mod tests;

pub use candidate::*;
