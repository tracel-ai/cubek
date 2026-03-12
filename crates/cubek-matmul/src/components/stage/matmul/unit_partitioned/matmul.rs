use crate::components::global::PlaneFlowPartition;
use crate::components::global::PlaneFlowPartitionRule;
use crate::components::stage::matmul::partitioned_matmul::PartitionedStageMatmul;
use crate::components::stage::matmul::partitioned_matmul::StagePartitioner;
use crate::components::tile::TileMatmul;
use crate::definition::MatmulTypes;
use crate::definition::MatrixTypes;
use cubecl::prelude::*;
use cubecl::std::tensor::layout::Coords2d;

use crate::components::{stage::matmul::partition::SharedPartitionMatmulConfig, tile::TileConfig};

#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
/// Configuration for the unit partitioned stage matmul
pub struct UnitPartitionedStageConfig<TC: TileConfig> {
    pub shared: SharedPartitionMatmulConfig<TC>,
}

impl<TC: TileConfig> UnitPartitionedStageConfig<TC> {
    pub fn from_shared_partition_config(shared: SharedPartitionMatmulConfig<TC>) -> Self {
        Self { shared }
    }
}

#[allow(type_alias_bounds)]
/// [PartitionedStageMatmul] partitioned across units
pub type UnitMatmul<
    MP: MatmulTypes,
    TM: TileMatmul<
            <MP::Lhs as MatrixTypes>::Register,
            <MP::Rhs as MatrixTypes>::Register,
            <MP::Acc as MatrixTypes>::Register,
        >,
    StageLhs,
    StageRhs,
    StageAcc,
    StageOut,
> = PartitionedStageMatmul<MP, TM, StageLhs, StageRhs, StageAcc, StageOut, UnitPartitioner>;

/// Defines how to partition across units
pub struct UnitPartitioner {}

#[cube]
impl StagePartitioner for UnitPartitioner {
    fn coordinates(
        #[comptime] role_rule_config: PlaneFlowPartitionRule,
        #[comptime] plane_dim: u32,
        #[comptime] num_partitions_col: u32,
    ) -> Coords2d {
        let plane_id = PlaneFlowPartition::new(role_rule_config).compute_index();

        let absolute_index = UNIT_POS_X + plane_dim * plane_id;

        (
            absolute_index / num_partitions_col,
            absolute_index % num_partitions_col,
        )
    }
}
