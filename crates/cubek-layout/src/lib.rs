//! The layout-request vocabulary: a strategy's intrinsic wish for the physical (storage)
//! layout of its operands, plus the labeled [`Axis`] it ranges over. A leaf crate shared by
//! the tile engine and every kernel client (matmul, reduce, pool), with no tile-engine or
//! GPU coupling.

mod axis;
mod request;

pub use axis::*;
pub use request::*;
