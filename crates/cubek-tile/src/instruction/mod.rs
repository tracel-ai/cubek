//! Pure Leaf Instructions (microkernels on registers / fragments).
//!
//! Executed only at final tiles ([`Partitioner::Final`](crate::Partitioner::Final)).
//! Zero awareness of global spaces, memory stages, or global tile walks.

pub mod extrema;
pub mod logsumexp;
pub mod mma;
pub mod sum;

pub use extrema::{max, min};
pub use mma::mma_leaf;
pub use sum::fold_group;
