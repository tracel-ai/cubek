//! Tile variants that map to a tile-level matmul instruction. Each
//! variant carries the hardware- or software-specific fragment + config
//! consumed by [`crate::tile::Tile::mma`].

pub(crate) mod cmma;
pub(crate) mod interleaved;
pub(crate) mod mma;
pub(crate) mod plane_vec;
pub(crate) mod register;

pub use cmma::*;
pub use interleaved::*;
pub use mma::*;
pub use plane_vec::*;
pub use register::*;
