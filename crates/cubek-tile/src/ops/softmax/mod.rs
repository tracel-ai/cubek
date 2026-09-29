//! The softmax reading of a [`Tile`](crate::Tile): one online-softmax step along the score
//! axis absent from the state's space.

pub(crate) mod leaf;
pub(crate) mod planewise;
pub(crate) mod rowwise;
pub(crate) mod state;

pub(crate) use state::*;

pub use state::{LOGIT_MASKED, MaskProbe, RowShare, RowState, masked_recip};
