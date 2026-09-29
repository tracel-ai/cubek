//! The matmul reading of a [`Tile`](crate::Tile): `c.mma(a, b)` treats the trailing two axes as
//! the `row × col` matrix, leading axes as a batch, and contracts `K`.

mod base;
mod config;
mod instruction;
pub(crate) mod leaf;

pub use config::*;
pub use instruction::*;
pub use leaf::*;
