//! The softmax reading of a [`Tile`](crate::Tile).
//!
//! [`OnlineSoftmax`] folds a score where its holder keeps it, the ownership read off the score.
//! The rest ([`RowState`], [`MaskProbe`], `Tile::softmax`) is the older step, whose caller states
//! who owns the rows; it stays until its last caller moves.

mod leaf;
mod online;
mod planewise;
mod rowwise;
mod state;

pub use online::*;
pub use state::*;
