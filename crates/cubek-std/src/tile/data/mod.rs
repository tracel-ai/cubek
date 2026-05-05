//! Tile data structures: every per-variant `*Tile` carrier, the strided/shared
//! tiles that wrap stage memory, the `TileKind` family, and the `RowWise`
//! support type. Compute that operates on these tiles lives in
//! [`crate::tile::compute`].

mod bounce;
mod cmma;
mod interleaved;
mod kind;
mod mma;
mod plane_vec;
mod register;
mod rowwise;
mod strided;
mod unit;
mod whitebox_fragment;

pub use bounce::*;
pub use cmma::*;
pub use interleaved::*;
pub use kind::*;
pub use mma::*;
pub use plane_vec::*;
pub use register::*;
pub use rowwise::*;
pub use strided::*;
pub use unit::*;
pub use whitebox_fragment::*;
