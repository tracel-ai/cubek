//! The reduce microkernels, one folder per scope ([`unit`](mod@unit), [`plane`], [`cube`],
//! [`hypercube`]), and what every one of them shares: the operations, the operands, the work item
//! and what a row accumulates into.
//!
//! A reduction reduces the values of a row under a monoid, each value mapped as it is read
//! ([`ReduceOperation`]). The values accumulate into a carrier ([`ReduceAccumulator`]): the values themselves, a
//! value with its index for an argmax, or the `k` best with their indices for a top-k. An owner
//! reads its lines through its window of the input and accumulates each as it lands, never holding the
//! row, so a row as long as a vocabulary costs registers for one line; what tells the scopes apart
//! is only where the partials meet.

mod accumulator;
pub mod cube;
pub mod hypercube;
mod operand;
mod operation;
pub mod plane;
pub mod unit;
mod work_item;

pub use accumulator::*;
pub use operand::*;
pub use operation::*;
pub use work_item::*;
