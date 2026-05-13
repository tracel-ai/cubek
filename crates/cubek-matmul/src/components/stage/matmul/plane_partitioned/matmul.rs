use crate::{
    components::global::PlaneFlowPartition, components::global::PlaneFlowPartitionRule,
    components::stage::matmul::partitioned_matmul::PartitionedStageMatmul,
    components::stage::matmul::partitioned_matmul::StagePartitioner, definition::MatmulTypes,
};
use cubecl::{prelude::*, std::tensor::layout::Coords2d};
use cubek_std::tile::Plane;

#[allow(type_alias_bounds)]
/// [PartitionedStageMatmul] partitioned across units
pub type PlaneMatmul<MP: MatmulTypes, StageLhs, StageRhs, StageAcc, StageOut> =
    PartitionedStageMatmul<MP, StageLhs, StageRhs, StageAcc, StageOut, PlanePartitioner>;

/// Defines how to partition across planes
pub struct PlanePartitioner {}

#[cube]
impl StagePartitioner for PlanePartitioner {
    type Scope = Plane;

    /// Returns the (row, col) of the current compute primitive within the stage.
    fn coordinates(
        #[comptime] role_rule_config: PlaneFlowPartitionRule,
        #[comptime] _plane_dim: u32,
        #[comptime] num_partitions_col: u32,
    ) -> Coords2d {
        let absolute_index = PlaneFlowPartition::new(role_rule_config).compute_index();

        (
            absolute_index / num_partitions_col,
            absolute_index % num_partitions_col,
        )
    }
}
