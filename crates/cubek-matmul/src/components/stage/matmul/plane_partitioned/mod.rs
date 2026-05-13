mod setup;

pub use setup::PlaneMatmulFamily;

/// Partitioner used by the plane-partitioned family. See
/// `crate::components::stage::matmul::partitioned_matmul::StagePartitioner`.
pub use super::partitioned_matmul::PlanePartitioner;
