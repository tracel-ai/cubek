//! The SwiGLU microkernels, one folder per scope ([`cube`]), and what every one of them shares:
//! the axes, the operands, the work item and the partitioning.
//!
//! `out = silu(gate) ⊙ up` over a fused `[gate | up]` row: the fused rows are one operand,
//! `[r, h, d]`, its two halves along `h`, and the work item is a run of one row, both halves of
//! it. Its owner reads each line of the run once from either half, computes `g · sigmoid(g) · u`
//! in `f32`, and writes the line. Nothing meets: an owner's units each take their lines and share
//! nothing, so a run is as long as its owner's units and lines make it, and a row is as many runs
//! as it takes.

pub mod cube;
mod operand;
mod partitioning;
mod work_item;

pub use operand::*;
pub use partitioning::*;
pub use work_item::*;
