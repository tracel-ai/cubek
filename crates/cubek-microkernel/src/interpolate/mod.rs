//! The interpolate microkernels, one folder per scope ([`cube`]), and what every one of them
//! shares: the operands and the work item.
//!
//! Each output position expands into a square of source taps. The input is gathered into a tile
//! holding those samples, the separable filter weights are a procedural tile over the same axes,
//! and the output is their tile product: every tap of a position reduced into its channels.

pub mod cube;
mod operand;
mod work_item;

pub use operand::*;
pub use work_item::*;
