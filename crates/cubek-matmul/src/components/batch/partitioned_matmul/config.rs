use cubecl::{CubeCount, CubeDim};

use crate::components::global::memory::GlobalLayoutConfig;
use crate::definition::{
    CubeCountPlan, GlobalPartitionSize, MatmulLineSizes, MatmulProblem, MatmulSetupError,
};
use crate::{
    components::{batch::BatchConfig, global::GlobalConfig},
    definition::HypercubeConfig,
};

#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
/// Configuration for partitioned batch matmul
pub struct PartitionedBatchConfig<G: GlobalConfig> {
    pub global_config: G,
    pub hypercube_config: HypercubeConfig,
    pub global_partition_size: GlobalPartitionSize,
}

impl<G: GlobalConfig> BatchConfig for PartitionedBatchConfig<G> {
    fn cube_dim(&self) -> CubeDim {
        self.global_config.cube_dim()
    }

    fn line_sizes(&self) -> MatmulLineSizes {
        self.global_config.global_line_sizes()
    }

    fn can_yield_extra_cubes(&self) -> bool {
        self.hypercube_config
            .cube_count_plan_blueprint
            .can_yield_extra_cubes()
    }

    fn lhs_global_layout_config(&self) -> GlobalLayoutConfig {
        self.global_config.lhs_reader_config().gmem_config.into()
    }

    fn rhs_global_layout_config(&self) -> GlobalLayoutConfig {
        self.global_config.rhs_reader_config().gmem_config.into()
    }

    fn out_global_layout_config(&self) -> GlobalLayoutConfig {
        self.global_config.writer_config().gmem_config.into()
    }

    fn cube_count_plan(
        &self,
        problem: &MatmulProblem,
        max_cube_count: &CubeCount,
    ) -> CubeCountPlan {
        self.hypercube_config
            .cube_count_plan(problem, max_cube_count)
    }
}

impl<G: GlobalConfig> PartitionedBatchConfig<G> {
    /// Create a new config for partitioned batch matmul
    pub fn new(
        global_config: G,
        hypercube_config: HypercubeConfig,
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
