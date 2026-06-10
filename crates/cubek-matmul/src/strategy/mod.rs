//! Strategy selection and dispatch: the user-facing [`Strategy`] enum, the kernel
//! selector, and the autotune key.

mod select_kernel;
#[allow(clippy::module_inception)]
mod strategy;
#[cfg(feature = "extended")]
pub mod test_only;
mod tune_key;

pub use select_kernel::*;
pub use strategy::*;
pub use tune_key::*;
