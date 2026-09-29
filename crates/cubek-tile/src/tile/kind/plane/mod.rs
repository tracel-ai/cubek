//! What one plane holds of a tile: the encodings ([`cmma`], [`mma`], [`registers`], [`lines`])
//! and the grid of them a plane owns ([`base`]).

mod base;
mod cmma;
mod lines;
mod load_matrix;
mod mma;
mod registers;

pub use base::*;
pub(crate) use cmma::*;
pub(crate) use lines::*;
pub use mma::*;
pub(crate) use registers::*;
