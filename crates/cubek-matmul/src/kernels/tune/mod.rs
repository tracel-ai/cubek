//! How the matmul family is measured: the tuning [`key`] that makes two products the same product,
//! the candidate every float routine is listed as, and the one a routine is listed as at the
//! settings it lists for a packed input. A caller builds its autotunes from them.

mod autotunes;
mod candidate;
#[cfg(feature = "autotune-checks")]
mod checks;
pub mod key;
mod packed_candidate;
mod routine_settings;
#[cfg(test)]
mod tests;

pub use autotunes::*;
pub use candidate::*;
#[cfg(feature = "autotune-checks")]
pub use checks::*;
pub use packed_candidate::*;
pub use routine_settings::*;
