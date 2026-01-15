use crate::components::global::memory::GlobalLayoutConfig;
use crate::components::{batch::BatchConfig, global::GlobalConfig};
use crate::definition::GlobalPartitionSize;

#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
/// Configuration for partitioned batch matmul
pub struct PartitionedBatchConfig<G: GlobalConfig> {
    pub global_config: G,
    pub global_partition_size: GlobalPartitionSize,
}

impl<G: GlobalConfig> BatchConfig for PartitionedBatchConfig<G> {
    fn lhs_global_layout_config(&self) -> GlobalLayoutConfig {
        self.global_config.lhs_reader_config().gmem_config.into()
    }

    fn rhs_global_layout_config(&self) -> GlobalLayoutConfig {
        self.global_config.rhs_reader_config().gmem_config.into()
    }

    fn out_global_layout_config(&self) -> GlobalLayoutConfig {
        self.global_config.writer_config().gmem_config.into()
    }
}

impl<G: GlobalConfig> PartitionedBatchConfig<G> {
    /// Create a new config for partitioned batch matmul
    pub fn new(global_config: G, global_partition_size: GlobalPartitionSize) -> Self {
        Self {
            global_config,
            global_partition_size,
        }
    }
}
