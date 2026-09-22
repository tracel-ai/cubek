//! The layouts a leaf reads a memory tile through, and the views that wrap them: flat, 2-D
//! matrix, gathered, and masked.

mod flat;
mod masked;
mod matrix;
mod projected;

pub use flat::*;
pub use masked::*;
pub(crate) use matrix::*;
pub(crate) use projected::*;
