//! The RMS-norm microkernels, one folder per scope ([`plane`], [`cube`]), and what every one of
//! them shares: the axes, the operands, the work item, the partitioning, and the row an owner
//! holds in its units' registers between its two passes.
//!
//! `out = x / sqrt(mean(x²) + eps) · weight` over the last dim: the work item is one row, and its
//! owner reads it once, squares it into a partial in `f32` as it lands, sums the partials of its
//! units, and writes the row scaled out of the registers it read it into. What tells the scopes
//! apart is only where the partials meet: across a plane's lanes, across a cube's planes after
//! each plane has summed its own, or in shared memory between a group of the cube's units that
//! need not be a plane ([`cube::CubeGroupsRmsNormMicrokernel`]).

pub mod cube;
mod held_row;
mod operand;
mod partitioning;
pub mod plane;
mod scale;
mod work_item;

pub use held_row::*;
pub use operand::*;
pub use partitioning::*;
pub use scale::*;
pub use work_item::*;
