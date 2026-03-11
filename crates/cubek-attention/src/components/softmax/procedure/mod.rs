//! Handwritten CubeCL logic that utilizes low-level fragment and
//! plane primitives while maintaining explicit control over the normalization sequence.

mod accumulator;
mod base;
mod broadcast_reducer;
mod naive_reducer;
mod reduce_op;
mod unit_reducer;

pub use accumulator::*;
pub use base::*;
pub use broadcast_reducer::*;
pub use reduce_op::*;
pub use unit_reducer::*;
