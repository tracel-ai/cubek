//! What every family's staged walk shares, whatever operands it stages: the ring that holds its
//! stages, and what a stage of each operand costs.

mod stage_ring;
mod staged_operand;

pub use stage_ring::*;
pub use staged_operand::*;
