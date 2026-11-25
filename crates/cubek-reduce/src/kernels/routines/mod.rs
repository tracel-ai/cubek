//! The reduce routines, one folder each, named for the routine: its scope, how it moves its data,
//! and its operation; and the trait they implement, in a module of its own.

mod cube_staged_reduce;
mod hypercube_atomic_reduce;
mod hypercube_last_arrival_reduce;
mod plane_shuffle_reduce;
mod reduce_routine;
mod unit_direct_reduce;
mod unit_direct_zero_reduce;

pub use cube_staged_reduce::*;
pub use hypercube_atomic_reduce::*;
pub use hypercube_last_arrival_reduce::*;
pub use plane_shuffle_reduce::*;
pub(crate) use reduce_routine::*;
pub use unit_direct_reduce::*;
pub use unit_direct_zero_reduce::*;
