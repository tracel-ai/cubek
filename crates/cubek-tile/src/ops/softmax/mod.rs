//! The softmax reading of a [`Tile`](crate::Tile): one online-softmax step along the score
//! axis absent from the state's space.

mod leaf;
pub mod logsumexp;
mod planewise;
mod rowwise;
mod state;

pub use state::*;
