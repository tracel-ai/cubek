mod config;
mod matmul;
mod partition;
mod setup;

pub use partition::{ColMajorGlobalPartitionMatmul, RowMajorGlobalPartitionMatmul};
pub use setup::PartitionedBatchMatmulFamily;
