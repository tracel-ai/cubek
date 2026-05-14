//! The `StagePartitioner` trait — pairs a [`Partitioner`] with the matmul-flow
//! glue cubek-matmul needs (the matching [`StageMatmulKind`] variant and a
//! [`partition_coordinates`] helper that derives `compute_index` from a
//! [`PlaneFlowPartitionRule`]).
//!
//! The actual `PlanePartitioner` / `UnitPartitioner` types and their
//! coordinate math live in [`cubek_std::tile`]; this file only attaches the
//! matmul-flow glue to them.

use crate::components::global::{PlaneFlowPartition, PlaneFlowPartitionRule};
use crate::components::stage::stage_matmul::StageMatmulKind;
use cubecl::{prelude::*, std::tensor::layout::Coords2d};
use cubek_std::tile::{Partitioner, Plane, PlanePartitioner, TileScope, Unit, UnitPartitioner};

/// Defines how the stage is partitioned among compute primitives (e.g., units
/// or planes). Controls global writeback and compute indexing. Each impl pairs
/// a [`TileScope`] with the matching [`StageMatmulKind`] variant — callers can
/// hop from a partitioner type to the right `StageMatmulKind` via `SP::KIND`.
pub trait StagePartitioner: Partitioner {
    /// Compute primitive that runs each partition.
    type Scope: TileScope;

    /// Comptime selector for the `StageMatmul` variant this partitioner pairs
    /// with. Callers route `kind.expand_stage_matmul(…)` /
    /// `kind.cubedim_resource(…)` / `kind.validate_blueprint(…)` through this.
    const KIND: StageMatmulKind;
}

impl StagePartitioner for PlanePartitioner {
    type Scope = Plane;
    const KIND: StageMatmulKind = StageMatmulKind::PlanePartitioned;
}

impl StagePartitioner for UnitPartitioner {
    type Scope = Unit;
    const KIND: StageMatmulKind = StageMatmulKind::UnitPartitioned;
}

#[cube]
/// Returns the `(row, col)` of the current compute primitive within the stage,
/// deriving `compute_index` from `role_rule_config` via
/// [`PlaneFlowPartition`] and delegating the per-scope math to
/// [`Partitioner::coordinates`].
pub fn partition_coordinates<P: Partitioner>(
    #[comptime] role_rule_config: PlaneFlowPartitionRule,
    #[comptime] plane_dim: u32,
    #[comptime] num_partitions_col: u32,
) -> Coords2d {
    let compute_index = PlaneFlowPartition::new(role_rule_config).compute_index();
    P::coordinates(compute_index, plane_dim, num_partitions_col)
}
