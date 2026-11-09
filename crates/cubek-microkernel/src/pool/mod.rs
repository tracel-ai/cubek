//! The max pool microkernels, one folder per scope ([`cube`]), and what every one of them shares:
//! the operands and the work item.
//!
//! Each output position reads a window of input taps, gathered into a tile over the window's
//! axes, and reduces it to its largest tap.

pub mod cube;
mod operand;
mod work_item;

pub use operand::*;
pub use work_item::*;
