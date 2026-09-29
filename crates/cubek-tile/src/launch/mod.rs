//! Binding tensors to a kernel and launching it.

pub(crate) mod arg;
pub(crate) mod base;
pub(crate) mod spec;

pub(crate) use arg::*;

pub use arg::accumulate::{AccumulateArg, AccumulateArgLaunch};
pub use arg::base::{Bound, BoundaryPolicy, Unbound};
pub use arg::delivery::Delivery;
pub use arg::destination::{Buffered, Destination, DestinationLaunch};
pub use arg::operand::{Input, InputArgs, Output, OutputArgs};
pub use arg::scale::{MaybeTile, MaybeTileExpand, scale_tile};
pub use arg::tma::{TmaBox, TmaOperand, TmaTileArgLaunch};
pub use base::Grid;
