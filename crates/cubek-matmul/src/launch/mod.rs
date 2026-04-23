pub mod launch_naive;
pub mod launch_tiling;
pub mod launch_vecmat_plane_parallel;
pub mod launch_vecmat_unit_perpendicular;
#[cfg(feature = "extended")]
pub mod test_only;

mod args;
mod base;
mod select_kernel;
mod strategy;
mod tune_key;

pub use args::*;
pub use base::*;
pub use select_kernel::*;
pub use strategy::*;
pub use tune_key::*;
