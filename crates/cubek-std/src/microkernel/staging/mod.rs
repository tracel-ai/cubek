//! What every family's staged walk shares, whatever operands it stages: the ring that holds its
//! stages, how a stage lays out its rows, and what a stage of each operand costs.

mod stage_layout;
mod stage_ring;
mod staged_operand;

pub use stage_layout::*;
pub use stage_ring::*;
pub use staged_operand::*;
