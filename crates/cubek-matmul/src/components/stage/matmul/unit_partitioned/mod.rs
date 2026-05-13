mod setup;

pub use setup::UnitMatmulFamily;

/// Partitioner used by the unit-partitioned family. See
/// `crate::components::stage::matmul::partitioned_matmul::StagePartitioner`.
pub use super::partitioned_matmul::UnitPartitioner;
