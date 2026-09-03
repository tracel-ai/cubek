//! How a level of the space splits, and the [`Walk`] it produces.

mod distribution;
mod operand;
mod partitioner;
mod tiling;
mod walk;
mod walk_order;

pub use distribution::*;
pub use operand::*;
pub use partitioner::*;
pub use tiling::*;
pub use walk::*;
pub use walk_order::*;
