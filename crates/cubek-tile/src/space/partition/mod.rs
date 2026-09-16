//! How a level of the space splits ([`Level`]), the [`Walk`] it produces, and the space
//! with all of its levels ([`Partitioning`]).

mod distribution;
mod level;
mod partitioning;
mod tiling;
mod walk;
mod walk_order;

pub use distribution::*;
pub use level::*;
pub use partitioning::*;
pub use tiling::*;
pub use walk::*;
pub use walk_order::*;
