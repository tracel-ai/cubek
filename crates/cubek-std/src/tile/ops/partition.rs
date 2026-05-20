//! `Tile::partition` dispatcher + the [`Partitioner`] strategies it dispatches
//! through. A partitioner takes a tile at one [`TileScope`] and produces a
//! per-primitive view at a lower scope (e.g. a plane-scope stage yields a
//! unit-scope partition view via [`UnitPartitioner`]).
//!
//! The math in [`UnitPartitioner::coordinates`] / [`PlanePartitioner::coordinates`]
//! migrates the previous `cubek_matmul::components::stage::StagePartitioner`
//! impls. The plane-specialization machinery (`PlaneFlowPartition`) stays in
//! cubek-matmul; callers there compute the `compute_index` runtime value
//! and pass it in as an argument to keep cubek-std free of matmul-flow
//! abstractions.
//!
//! The Stage-arm body of [`Tile::partition`] — materializing the per-primitive
//! partition tile from a [`StridedStage`] payload — lands in PR 4 alongside
//! the partition-matmul body migration; PR 3 only lays the API surface.

use std::marker::PhantomData;

use cubecl::{prelude::*, std::tensor::layout::Coords2d};

use crate::tile::{
    PartitionTile, Plane, Tile, TileExpand, TileKind, TileKindExpand, TileScope, Unit,
};

/// Strategy that maps the current compute primitive to its `(row, col)`
/// coordinates within a partition grid. The associated [`OutputScope`]
/// selects which [`TileScope`] the resulting partition tile is observed at:
/// e.g. [`UnitPartitioner`] partitions a higher-scope tile into per-unit
/// views.
///
/// [`OutputScope`]: Partitioner::OutputScope
#[cube]
pub trait Partitioner: 'static + Send + Sync {
    /// Scope of the partition tile produced by `Tile::partition`.
    type OutputScope: TileScope;

    /// `(row, col)` of this compute primitive in the partition grid. The
    /// inputs are the same comptime/runtime values today's
    /// `StagePartitioner::coordinates` consumes:
    ///
    /// - `compute_index` is the runtime plane-id-within-compute-flow,
    ///   produced by the caller (today via `PlaneFlowPartition::compute_index`).
    /// - `plane_dim` and `num_partitions_col` are comptime shape parameters
    ///   (plane width and number of partition columns in the stage).
    fn coordinates(
        compute_index: u32,
        #[comptime] plane_dim: u32,
        #[comptime] num_partitions_col: u32,
    ) -> Coords2d;
}

/// Partitions a higher-scope tile into per-unit views. Output scope is
/// [`Unit`]. The math mirrors the old
/// `cubek_matmul::components::stage::UnitPartitioner::coordinates`.
#[derive(Clone, Copy)]
pub struct UnitPartitioner;

#[cube]
impl Partitioner for UnitPartitioner {
    type OutputScope = Unit;

    fn coordinates(
        compute_index: u32,
        #[comptime] plane_dim: u32,
        #[comptime] num_partitions_col: u32,
    ) -> Coords2d {
        let absolute_index = UNIT_POS_X + plane_dim * compute_index;

        (
            absolute_index / num_partitions_col,
            absolute_index % num_partitions_col,
        )
    }
}

/// Partitions a higher-scope tile into per-plane views. Output scope is
/// [`Plane`]. The math mirrors the old
/// `cubek_matmul::components::stage::PlanePartitioner::coordinates`.
#[derive(Clone, Copy)]
pub struct PlanePartitioner;

#[cube]
impl Partitioner for PlanePartitioner {
    type OutputScope = Plane;

    fn coordinates(
        compute_index: u32,
        #[comptime] _plane_dim: u32,
        #[comptime] num_partitions_col: u32,
    ) -> Coords2d {
        (
            compute_index / num_partitions_col,
            compute_index % num_partitions_col,
        )
    }
}

#[cube]
impl<N: Numeric, Sc: TileScope> Tile<N, Sc> {
    /// Produce this compute primitive's view of `self` at the partitioner's
    /// output scope. Today the only valid source is a
    /// [`TileKind::Stage`] tile; the Stage-arm body is implemented in PR 4
    /// alongside the partition-matmul body migration.
    pub fn partition<P: Partitioner>(
        &self,
        compute_index: u32,
        #[comptime] plane_dim: u32,
        #[comptime] num_partitions_col: u32,
    ) -> Tile<N, P::OutputScope> {
        let (p_row, p_col) = P::coordinates(compute_index, plane_dim, num_partitions_col);
        match &self.kind {
            TileKind::Stage(stage) => {
                let m_tiles = comptime!(stage.config.tiles_per_partition_along_row);
                let n_tiles = comptime!(stage.config.tiles_per_partition_along_col);

                let mut tiles = Sequence::new();

                #[unroll]
                for m in 0..m_tiles {
                    #[unroll]
                    for n in 0..n_tiles {
                        let global = (p_row * m_tiles + m, p_col * n_tiles + n);
                        let shared = stage.get_tile(global);
                        tiles.push(Tile::<N, P::OutputScope>::new_SharedMemory(shared));
                    }
                }

                Tile::new_Partition(PartitionTile::<N, P::OutputScope> {
                    tiles,
                    rows: m_tiles,
                    cols: n_tiles,
                    _phantom: PhantomData,
                })
            }
            TileKind::SharedMemory(_)
            | TileKind::Cmma(_)
            | TileKind::Mma(_)
            | TileKind::Register(_)
            | TileKind::PlaneVec(_)
            | TileKind::Interleaved(_)
            | TileKind::Unit(_)
            | TileKind::WhiteboxFragment(_)
            | TileKind::Bounce(_)
            | TileKind::Partition(_)
            | TileKind::None => {
                panic!("Tile::partition: source variant cannot be partitioned")
            }
        }
    }
}
