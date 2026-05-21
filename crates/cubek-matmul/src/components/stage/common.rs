//! Shared stage-level data + glue used by both variants and the dispatch
//! enums. `mod.rs` stays strictly dispatching; everything that isn't a
//! per-variant dispatcher lives here.

use std::marker::PhantomData;

use cubecl::prelude::*;
use cubek_std::{
    PartitionSize, StageSize,
    stage::StageMemoryConfig,
    tile::{PartitionSchedulerScheme, PartitionTile, PipelinedTile, Plane, Tile, TileScope, Unit},
};

pub use cubek_std::tile::{PartitionBuffering, Partitioner, PlanePartitioner, UnitPartitioner};

use crate::components::global::PlaneFlowConfig;
use crate::components::tile::{TileMatmul, allocate_acc, allocate_lhs, allocate_rhs};
use crate::definition::{AccRE, LhsRE, MatmulTypes, MatrixTypes, RhsRE};

use super::StageMatmulKind;

type AccPartitionTile<MT, Sc> = PartitionTile<AccRE<MT>, Sc>;
type PipelinedBTile<MT, Sc> =
    PipelinedTile<<<MT as MatmulTypes>::Rhs as MatrixTypes>::Register, Sc>;

// =====================================================================
// NumStages — buffer count
// =====================================================================

#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
/// Number of stages in one shared memory, i.e. buffers for double buffering
pub struct NumStages {
    pub lhs: u32,
    pub rhs: u32,
}

impl From<(u32, u32)> for NumStages {
    fn from(value: (u32, u32)) -> Self {
        NumStages {
            lhs: value.0,
            rhs: value.1,
        }
    }
}

// =====================================================================
// Partitioner glue — pair cubek-std's `Partitioner` with the matmul-flow
// `StageMatmulKind` selector + the role-rule-derived coordinate helper.
// =====================================================================

/// Defines how the stage is partitioned among compute primitives (e.g., units
/// or planes).
pub trait StagePartitioner: Partitioner {
    /// Compute primitive that runs each partition.
    type Scope: TileScope;
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

// =====================================================================
// PartitionedStageMatmul — shared per-variant data.
// =====================================================================

/// Data carried by both [`StageMatmul`](super::StageMatmul) variants. Today
/// the unit- and plane-partitioned flows hold the same fields.
#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
pub struct PartitionedStageMatmul {
    pub tile_matmul: TileMatmul,
    pub partition_size: PartitionSize,
    pub partition_buffering: PartitionBuffering,
    pub plane_flow_config: PlaneFlowConfig,
    pub plane_dim: u32,
    pub stage_size: StageSize,
    pub partition_schedule_scheme: PartitionSchedulerScheme,
    pub lhs_smem_config: StageMemoryConfig,
    pub rhs_smem_config: StageMemoryConfig,
    pub acc_smem_config: StageMemoryConfig,
    pub out_smem_config: StageMemoryConfig,
}

// =====================================================================
// Per-partition init helpers
// =====================================================================

#[cube]
/// Initialize the per-`m` lhs register fragments.
///
/// # Safety
///
/// This may point towards uninitialized memory. Make sure to load fragments
/// before execution.
pub fn init_a_fragment<MT: MatmulTypes, Sc: TileScope>(
    #[comptime] shared_config: PartitionedStageMatmul,
) -> Sequence<Tile<<MT::Lhs as MatrixTypes>::Register, Sc>> {
    let mut lhs = Sequence::new();
    #[unroll]
    for _ in 0..shared_config.partition_size.m() {
        lhs.push(allocate_lhs::<LhsRE<MT>, RhsRE<MT>, AccRE<MT>, Sc>(
            shared_config.lhs_smem_config.matrix_layout,
            shared_config.tile_matmul,
        ));
    }
    lhs
}

#[cube]
/// Initialize the rhs register fragment(s) as a `Pipelined`-kind [`Tile`].
/// The inner sequence's comptime length is 1 (single-buffered) or 2
/// (double-buffered);
/// [`Tile::mma_partition`](cubek_std::tile::Tile::mma_partition) reads it to
/// pick the buffering strategy.
///
/// # Safety
///
/// This may point towards uninitialized memory.
pub fn init_b_fragments<MT: MatmulTypes, Sc: TileScope>(
    #[comptime] shared_config: PartitionedStageMatmul,
) -> Tile<<MT::Rhs as MatrixTypes>::Register, Sc> {
    let mut fragments = Sequence::new();
    let n_buffers = comptime!(match shared_config.partition_buffering {
        PartitionBuffering::Single => 1usize,
        PartitionBuffering::Double => 2usize,
    });
    #[unroll]
    for _ in 0..n_buffers {
        fragments.push(allocate_rhs::<LhsRE<MT>, RhsRE<MT>, AccRE<MT>, Sc>(
            shared_config.rhs_smem_config.matrix_layout,
            shared_config.tile_matmul,
        ));
    }
    Tile::<<MT::Rhs as MatrixTypes>::Register, Sc>::new_Pipelined(PipelinedBTile::<MT, Sc> {
        fragments,
    })
}

#[cube]
/// Initialize accumulators as a partition-kind tile.
///
/// # Safety
///
/// This may point towards uninitialized memory. Make sure to call
/// [`load_partition_from_stage`](cubek_std::tile::load_partition_from_stage)
/// prior to [`Tile::mma_partition`](cubek_std::tile::Tile::mma_partition).
pub fn init_accumulator<MT: MatmulTypes, Sc: TileScope>(
    #[comptime] shared_config: PartitionedStageMatmul,
) -> Tile<AccRE<MT>, Sc> {
    let mut tiles = Sequence::new();

    #[unroll]
    for _ in 0..shared_config.partition_size.mn() {
        tiles.push(allocate_acc::<LhsRE<MT>, RhsRE<MT>, AccRE<MT>, Sc>(
            shared_config.out_smem_config.matrix_layout,
            shared_config.tile_matmul,
        ));
    }

    let partition = AccPartitionTile::<MT, Sc> {
        tiles,
        rows: comptime!(shared_config.partition_size.m()),
        cols: comptime!(shared_config.partition_size.n()),
        _phantom: PhantomData,
    };
    Tile::<AccRE<MT>, Sc>::new_Partition(partition)
}
