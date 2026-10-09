//! Binding tensors to a kernel and launching it: the [`Launcher`](crate::Launcher), each operand's
//! comptime `spec`, the `binding` that builds it from a tensor, and the `arg`s the kernel receives.

pub(crate) mod arg;
pub(crate) mod base;
pub(crate) mod binding;
pub(crate) mod refusal;
pub(crate) mod spec;

pub use arg::accumulate::{AccumulateArg, AccumulateArgLaunch};
pub use arg::destination::{Buffered, Destination, DestinationLaunch};
pub use arg::last_arrival::LastArrival;
pub use arg::operand::{Input, InputArgs, Output, OutputArgs};
pub use arg::relay::Relay;
pub use arg::scale::{MaybeTile, MaybeTileExpand, scale_tile};
pub use arg::tma::TmaTileArg;
pub use arg::tma::{TmaBox, TmaOperand, TmaTileArgLaunch};
pub use base::Grid;
pub use binding::{Arg, Bound, Labelled, Unbound, Unlabelled};
pub use refusal::Refusal;
pub use spec::{BoundaryPolicy, Delivery};
