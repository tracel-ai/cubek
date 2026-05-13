//! The `StagePartitioner` trait — picks per-primitive partition coordinates
//! for the partition matmul flow — and its two impls (`PlanePartitioner`,
//! `UnitPartitioner`). These are the remaining cubek-matmul concepts that
//! drive the partition scheduler from the global matmul layer.
//!
//! The old `PartitionedStageMatmul<MP, …>` impl-struct that wrapped the
//! `StageMatmul` trait around these partitioners has been deleted along with
//! the trait — the global matmul flows now use `PartitionMatmul::<…>` direct
//! calls + `write_partition_to_stage` instead.

use crate::components::global::{PlaneFlowPartition, PlaneFlowPartitionRule};
use cubecl::{prelude::*, std::tensor::layout::Coords2d};
use cubek_std::tile::{Plane, TileScope, Unit};

#[cube]
/// Defines how the stage is partitioned among compute primitives (e.g., units
/// or planes). Controls global writeback and compute indexing.
pub trait StagePartitioner: Send + Sync + 'static {
    /// Compute primitive that runs each partition.
    type Scope: TileScope;

    /// Returns the (row, col) of the current compute primitive within the stage.
    fn coordinates(
        #[comptime] role_rule_config: PlaneFlowPartitionRule,
        #[comptime] plane_dim: u32,
        #[comptime] num_partitions_col: u32,
    ) -> Coords2d;
}

/// Partitions across planes — `Scope = Plane`.
pub struct PlanePartitioner {}

#[cube]
impl StagePartitioner for PlanePartitioner {
    type Scope = Plane;

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

/// Partitions across units — `Scope = Unit`.
pub struct UnitPartitioner {}

#[cube]
impl StagePartitioner for UnitPartitioner {
    type Scope = Unit;

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
