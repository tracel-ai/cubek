//! How the matmul family is measured: the tuning [`key`] that makes two products the same product,
//! the candidate every float routine is listed as, and the one a routine is listed as at the
//! settings it lists for a packed input. A caller builds its tables from them.

mod candidate;
#[cfg(feature = "autotune-checks")]
mod checks;
pub mod key;
mod packed_candidate;
mod product;
mod routine_settings;
mod tables;
#[cfg(test)]
mod tests;

pub use candidate::*;
#[cfg(feature = "autotune-checks")]
pub use checks::*;
pub use packed_candidate::*;
pub use product::{FLOAT_SETTING_PACKED_INPUT, PACKED_SETTING_FLOAT_INPUTS};
pub(crate) use product::{ListedFor, Listing, Product, first_served};
pub use routine_settings::*;
pub use tables::*;
