//! Leaf instructions: one hardware operation, or one fixed instruction sequence with no loop
//! over data.
//!
//! Executed only at final tiles ([`Partitioner::Final`](crate::Partitioner::Final)). Zero
//! awareness of global spaces, memory stages, or global tile walks. The register loop nests that
//! issue these repeatedly are one layer up, in [`microkernel`](crate::microkernel); the leaf
//! dispatcher that picks between a hardware leaf and a nest is [`mma_leaf`](fn@mma::mma_leaf).

pub mod logsumexp;
pub mod mma;
mod op;
pub mod plane;

pub use op::*;
