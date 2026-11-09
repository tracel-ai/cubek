//! The cube staged routines: a cube owns a box of the output and stages it for its planes
//! ([`CubeStagedMicrokernel`](cubek_microkernel::matmul::cube::CubeStagedMicrokernel)). One routine
//! a way the stages are filled and the box stored, each refusing a problem its engines cannot
//! serve; what they share is in `base`.

mod async_copy;
mod base;
mod block_scaled;
mod kernel;
#[cfg(test)]
mod tests;
mod tma;
mod tma_filler;
mod tma_stored;
mod units;
mod wgmma;

pub use async_copy::*;
pub use base::CubeStagedMatmulRoutineBlueprint;
pub(crate) use base::widths_in_blocks;
pub use tma::*;
pub use tma_filler::*;
pub use tma_stored::*;
pub use units::*;
pub use wgmma::*;
