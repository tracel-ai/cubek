use cubecl::CubeDim;

use crate::definition::{GlobalPartitionSize, MatmulLineSizes, MatmulProblem, MatmulSetupError};
use crate::{
    components::{
        batch::BatchConfig,
        global::{GlobalConfig, GlobalReaderConfig, GlobalWriterConfig},
    },
    definition::HypercubeBlueprint,
};

#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
/// Configuration for partitioned batch matmul
pub struct PartitionedBatchConfig<G: GlobalConfig> {
    pub global_config: G,
    hypercube_config: HypercubeBlueprint,
    pub global_partition_size: GlobalPartitionSize,
}

impl<G: GlobalConfig> BatchConfig for PartitionedBatchConfig<G> {
    fn cube_dim(&self) -> CubeDim {
        self.global_config.cube_dim()
    }

    fn line_sizes(&self) -> MatmulLineSizes {
        self.global_config.global_line_sizes()
    }

    fn hypercube_blueprint(&self) -> HypercubeBlueprint {
        self.hypercube_config
    }

    fn can_yield_extra_cubes(&self) -> bool {
        self.hypercube_config
            .cube_count_plan_config
            .can_yield_extra_cubes()
    }

    fn lhs_global_reader_config(&self) -> GlobalReaderConfig {
        self.global_config.lhs_reader_config()
    }

    fn rhs_global_reader_config(&self) -> GlobalReaderConfig {
        self.global_config.rhs_reader_config()
    }

    fn global_writer_config(&self) -> GlobalWriterConfig {
        self.global_config.writer_config()
    }
}

impl<G: GlobalConfig> PartitionedBatchConfig<G> {
    /// Create a new config for partitioned batch matmul
    pub fn new(
        global_config: G,
        hypercube_config: HypercubeBlueprint,
        global_partition_size: GlobalPartitionSize,
    ) -> Self {
        Self {
            global_config,
            hypercube_config,
            global_partition_size,
        }
    }

    /// May return an error if:
    /// - hypercube config is invalid
    pub fn validate(self, problem: &MatmulProblem) -> Result<Self, MatmulSetupError> {
        self.hypercube_config.validate(problem)?;
        Ok(self)
    }
}
