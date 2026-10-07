//! Binding tensors to a kernel and launching it.

pub(crate) mod arg;
pub(crate) mod base;
pub(crate) mod refusal;
pub(crate) mod spec;

pub use arg::accumulate::{AccumulateArg, AccumulateArgLaunch};
pub use arg::base::{Arg, Bound, Labelled, Unbound, Unlabelled};
pub use arg::boundary_policy::BoundaryPolicy;
pub use arg::delivery::Delivery;
pub use arg::destination::{Buffered, Destination, DestinationLaunch};
pub use arg::operand::{Input, InputArgs, Output, OutputArgs};
pub use arg::relay::Relay;
pub use arg::scale::{MaybeTile, MaybeTileExpand, scale_tile};
pub use arg::tma::TmaTileArg;
pub use arg::tma::{TmaBox, TmaOperand, TmaTileArgLaunch};
pub use base::Grid;
pub use refusal::Refusal;
