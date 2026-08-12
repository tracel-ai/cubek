//! Pure Leaf Instructions (microkernels on registers / fragments).
//!
//! Executed only at final tiles ([`Partitioner::Final`](crate::Partitioner::Final)).
//! Zero awareness of global spaces, memory stages, or global tile walks.

pub mod logsumexp;
pub mod max;
pub mod min;
pub mod mma;
pub mod sum;
