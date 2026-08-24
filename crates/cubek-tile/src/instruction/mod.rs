//! Leaf instructions: one hardware operation, or one fixed instruction sequence with no loop
//! over data.
//!
//! Executed only at final tiles ([`Partitioner::Final`](crate::Partitioner::Final)). Zero
//! awareness of global spaces, memory stages, or global tile walks, and nothing here reaches
//! up into [`instruction`](crate::instruction::registers) or the verbs: the register loop nests that issue
//! these repeatedly are one layer up, and the leaf dispatch that picks between a hardware leaf
//! and a nest is the verb's, one layer above that.

pub mod logsumexp;
mod mma;
mod op;
pub mod plane;
pub mod registers;

pub use op::*;
