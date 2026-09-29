//! The matmul reading of a [`Tile`](crate::Tile): `c.mma(a, b)` treats the trailing two axes as
//! the `row × col` matrix, leading axes as a batch, and contracts `K`.

pub(crate) mod base;
pub(crate) mod config;
pub(crate) mod instruction;
pub(crate) mod leaf;

pub(crate) use leaf::*;

pub use config::{LoadMethod, MmaIo};
pub use leaf::scale::Side;
