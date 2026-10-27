//! How the matmul family is measured: the tuning [`key`] that makes two products the same product,
//! the candidate every float routine is listed as, and the one a routine is listed as at the
//! settings it lists for a packed input. A caller builds its tables from them.

mod candidate;
pub mod key;
mod packed_candidate;
#[cfg(test)]
mod tests;

pub use candidate::*;
pub use packed_candidate::*;
