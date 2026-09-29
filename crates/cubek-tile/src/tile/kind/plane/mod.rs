//! What one plane holds of a tile: the encodings ([`cmma`], [`mma`], [`registers`], [`lines`])
//! and the grid of them a plane owns ([`base`]).

pub(crate) mod base;
pub(crate) mod cmma;
pub(crate) mod lines;
pub(crate) mod load_matrix;
pub(crate) mod mma;
pub(crate) mod registers;

pub(crate) use base::*;
pub(crate) use cmma::*;
pub(crate) use lines::*;
pub(crate) use mma::*;
pub(crate) use registers::*;
