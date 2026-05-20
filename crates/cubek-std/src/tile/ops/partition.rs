//! `Tile::partition` and the [`Partitioner`] strategies. A partitioner takes
//! a tile at one [`TileScope`] and yields a per-primitive view at a lower
//! scope.

use std::marker::PhantomData;

use cubecl::{prelude::*, std::tensor::layout::Coords2d};

use crate::tile::{
    PartitionTile, Plane, Tile, TileExpand, TileKind, TileKindExpand, TileScope, Unit,
};

/// Maps the current compute primitive to `(row, col)` in a partition grid.
#[cube]
pub trait Partitioner: 'static + Send + Sync {
    type OutputScope: TileScope;

    fn coordinates(
        compute_index: u32,
        #[comptime] plane_dim: u32,
        #[comptime] num_partitions_col: u32,
    ) -> Coords2d;
}

/// Per-unit views of a higher-scope tile.
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

/// Per-plane views of a higher-scope tile.
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
    /// View of `self` at the partitioner's output scope. Source must be a
    /// `TileKind::Stage`.
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
                        tiles.push(Tile::<N, P::OutputScope>::new_SharedTile(shared));
                    }
                }

                Tile::new_Partition(PartitionTile::<N, P::OutputScope> {
                    tiles,
                    rows: m_tiles,
                    cols: n_tiles,
                    _phantom: PhantomData,
                })
            }
            TileKind::SharedTile(_)
            | TileKind::Cmma(_)
            | TileKind::Mma(_)
            | TileKind::Register(_)
            | TileKind::PlaneVec(_)
            | TileKind::Interleaved(_)
            | TileKind::Unit(_)
            | TileKind::WhiteboxFragment(_)
            | TileKind::RowWise(_)
            | TileKind::Bounce(_)
            | TileKind::Partition(_)
            | TileKind::Pipelined(_)
            | TileKind::None => {
                panic!("Tile::partition: source variant cannot be partitioned")
            }
        }
    }
}
