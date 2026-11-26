//! How one unit writes its carrier: to its window of the output, which every reduce microkernel
//! ends on ([`UnitCarrierWriteMicrokernel`], then [`UnitLinesWriteMicrokernel`] once its lanes are
//! met), where an owner's units met ([`UnitRegionWriteMicrokernel`]), or one unit a column of rows
//! ([`UnitGroupColumnWriteMicrokernel`]); or as a partial that a later unit reduces
//! ([`UnitPartialStoreMicrokernel`]).

mod carrier;
mod columns;
mod lines;
mod partial;
mod region;

pub use carrier::*;
pub use columns::*;
pub use lines::*;
pub use partial::*;
pub use region::*;
