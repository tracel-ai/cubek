//! The reduce microkernels, one folder per scope ([`unit`], [`plane`], [`cube`], [`hypercube`]),
//! and what every one of them shares: the operations, the operands, the work item and what a row
//! folds into.
//!
//! A reduction folds the values of a row under a monoid, each value mapped as it is read
//! ([`ReduceOperation`]). The values fold into a carrier ([`Folded`]): the values themselves, a
//! value with its index for an argmax, or the `k` best with their indices for a top-k. An owner
//! reads its lines through its window of the input and folds each as it lands, never holding the
//! row, so a row as long as a vocabulary costs registers for one line; what tells the scopes apart
//! is only where the partials meet.

pub mod cube;
mod folded;
pub mod hypercube;
mod operand;
mod operation;
pub mod plane;
pub mod unit;
mod work_item;

pub use folded::*;
pub use operand::*;
pub use operation::*;
pub use work_item::*;
