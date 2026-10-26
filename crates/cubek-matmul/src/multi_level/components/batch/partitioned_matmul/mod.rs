mod config;
mod matmul;
mod partition;
mod setup;

pub use partition::RowMajorGlobalPartitionMatmul;
pub use setup::PartitionedBatchMatmulFamily;
