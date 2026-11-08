//! The pool microkernels, one folder per scope ([`cube`]), and what every one of them shares: the
//! operands and the work item.
//!
//! Each output position reads a window of input taps, gathered into a tile over the window's
//! axes. A max pools the window as a reduce. An average contracts it against weights, a
//! procedural tile over the same axes that sums to one over the taps it counts.

pub mod cube;
mod operand;
mod work_item;

pub use operand::*;
pub use work_item::*;
