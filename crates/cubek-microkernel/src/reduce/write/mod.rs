//! How one unit writes its carrier: to its window of the output, which every reduce microkernel
//! ends on ([`UnitCarrierWriteMicrokernel`], then [`UnitLinesWriteMicrokernel`] once its lanes are
//! met), where an owner's units met ([`UnitRegionWriteMicrokernel`]), or one unit a column of rows
//! ([`UnitWalkWriteMicrokernel`]); or as a partial that a later unit reduces
//! ([`UnitPartialStoreMicrokernel`]).

mod carrier;
mod lines;
mod partial;
mod region;
mod walk;

pub use carrier::*;
pub use lines::*;
pub use partial::*;
pub use region::*;
pub use walk::*;
