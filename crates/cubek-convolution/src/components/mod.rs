pub mod global;
pub mod stage;

mod config;
mod error;
mod problem;
mod selection;

pub use config::*;
use cubek_matmul::components::tile::cmma::CmmaMatmul;
use cubek_std::tile::Strided;
pub use error::*;
pub use problem::*;
pub use selection::*;

/// Convolution using `AcceleratedMatmul`
pub type AcceleratedConv = CmmaMatmul<Option<Strided>>;
