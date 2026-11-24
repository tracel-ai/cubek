//! How one unit writes its carrier: to its window of the output, which every reduce microkernel
//! ends on ([`UnitCarrierWriteMicrokernel`]), or one unit a column of rows
//! ([`UnitWalkWriteMicrokernel`]); or as a partial that a later unit reduces
//! ([`UnitPartialStoreMicrokernel`]).

mod carrier;
mod partial;
mod walk;

pub use carrier::*;
pub use partial::*;
pub use walk::*;
