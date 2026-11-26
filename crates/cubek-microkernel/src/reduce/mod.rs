//! The reduce microkernels, one folder per scope ([`unit`](mod@unit), [`plane`], [`cube`],
//! [`hypercube`]), and what every one of them shares: the operations, the operands, the work item
//! and what a row accumulates into.
//!
//! A reduction reduces the values of a row under a monoid, each value mapped as it is read
//! ([`ReduceOperation`]). The values accumulate into a carrier ([`Accumulable`]): the values themselves, a
//! value with its index for an argmax, or the `k` best for a top-k, with their indices where it
//! writes them. An owner
//! reads its lines through its window of the input and accumulates each as it lands, never holding the
//! row, so a row as long as a vocabulary costs registers for one line; what tells the scopes apart
//! is only where the partials meet. Each ends on one unit writing its carrier, or one a column where
//! its units spread across lines of rows ([`write`](mod@write)).

#[cfg(test)]
#[macro_use]
mod testing;

mod accumulator;
mod columns;
pub mod cube;
pub mod hypercube;
mod operand;
mod operation;
mod partial_indices;
pub mod plane;
pub mod read;
pub mod unit;
mod work_item;
pub mod write;

pub use accumulator::*;
pub(crate) use columns::*;
pub use operand::*;
pub use operation::*;
pub(crate) use partial_indices::*;
pub use work_item::*;
