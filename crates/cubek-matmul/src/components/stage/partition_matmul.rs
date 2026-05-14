//! Per-partition init / load helpers. Each function carries the
//! [`MatmulTypes`] plumbing so callers can resolve element types from a
//! single `MT: MatmulTypes` generic and avoid spelling per-element-type
//! generics at every call site. The partition matmul execute itself is
//! invoked through
//! [`Tile::mma_partition`](cubek_std::tile::Tile::mma_partition) at the call
//! site; the `b_fragments` sequence length (1 or 2) picks single- vs
//! double-buffered rhs.

use std::marker::PhantomData;

use crate::definition::{AccSE, AccSS};
use crate::{components::stage::stage_matmul::PartitionedStageMatmul, definition::Acc};
use crate::{
    components::{
        stage::{PartitionBuffering, Stage},
        tile::TileMatmul,
    },
    definition::{AccRE, LhsRE, MatmulTypes, MatrixTypes, RhsRE},
};
use cubecl::prelude::*;
use cubek_std::tile::{
    PartitionScheduler, PartitionTile, PipelinedTile, Tile, TileScope, cmma_allocate_acc,
    cmma_allocate_lhs, cmma_allocate_rhs, interleaved_allocate_acc, interleaved_allocate_lhs,
    interleaved_allocate_rhs, mma_allocate_acc, mma_allocate_lhs, mma_allocate_rhs,
    planevec_allocate_acc, planevec_allocate_lhs, planevec_allocate_rhs, register_allocate_acc,
    register_allocate_lhs, register_allocate_rhs,
};
use cubek_std::{MatrixLayout, StageIdent};

type STy<T> = crate::definition::Stage<T>;
type AccPartitionTile<MT, Sc> = PartitionTile<AccRE<MT>, Sc, ReadWrite>;
type PipelinedBTile<MT, Sc> =
    PipelinedTile<<<MT as MatmulTypes>::Rhs as MatrixTypes>::Register, Sc, ReadWrite>;

#[cube]
/// Initialize the per-`m` lhs register fragments.
///
/// # Safety
///
/// This may point towards uninitialized memory. Make sure to load fragments
/// before execution.
pub fn init_a_fragment<MT: MatmulTypes, Sc: TileScope>(
    #[comptime] shared_config: PartitionedStageMatmul,
) -> Sequence<Tile<<MT::Lhs as MatrixTypes>::Register, Sc, ReadWrite>> {
    let mut lhs = Sequence::new();
    #[unroll]
    for _ in 0..shared_config.partition_size.m() {
        lhs.push(allocate_lhs::<MT, Sc>(
            shared_config.lhs_smem_config.matrix_layout,
            shared_config.tile_matmul,
        ));
    }
    lhs
}

#[cube]
/// Initialize the rhs register fragment(s) as an [`PipelinedTile`]-kind
/// [`Tile`]. The inner sequence's comptime length is 1 (single-buffered) or
/// 2 (double-buffered);
/// [`Tile::mma_partition`](cubek_std::tile::Tile::mma_partition) reads it to
/// pick the buffering strategy.
///
/// # Safety
///
/// This may point towards uninitialized memory.
pub fn init_b_fragments<MT: MatmulTypes, Sc: TileScope>(
    #[comptime] shared_config: PartitionedStageMatmul,
) -> Tile<<MT::Rhs as MatrixTypes>::Register, Sc, ReadWrite> {
    let mut fragments = Sequence::new();
    let n_buffers = comptime!(match shared_config.partition_buffering {
        PartitionBuffering::Single => 1usize,
        PartitionBuffering::Double => 2usize,
    });
    #[unroll]
    for _ in 0..n_buffers {
        fragments.push(allocate_rhs::<MT, Sc>(
            shared_config.rhs_smem_config.matrix_layout,
            shared_config.tile_matmul,
        ));
    }
    Tile::<<MT::Rhs as MatrixTypes>::Register, Sc, ReadWrite>::new_Pipelined(
        PipelinedBTile::<MT, Sc> { fragments },
    )
}

#[cube]
/// Initialize accumulators as a partition-kind tile.
///
/// # Safety
///
/// This may point towards uninitialized memory. Make sure to call
/// [`load_accumulator`] prior to
/// [`Tile::mma_partition`](cubek_std::tile::Tile::mma_partition).
pub fn init_accumulator<MT: MatmulTypes, Sc: TileScope>(
    #[comptime] shared_config: PartitionedStageMatmul,
) -> Tile<AccRE<MT>, Sc, ReadWrite> {
    let mut tiles = Sequence::new();

    #[unroll]
    for _ in 0..shared_config.partition_size.mn() {
        tiles.push(allocate_acc::<MT, Sc>(
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
    Tile::<AccRE<MT>, Sc, ReadWrite>::new_Partition(partition)
}

#[cube]
/// Fill accumulators through a stage.
pub fn load_accumulator<
    MT: MatmulTypes,
    StageAcc: Stage<STy<Acc<MT>>, ReadOnly>,
    Sc: TileScope,
>(
    stage: &StageAcc,
    acc: &mut Tile<AccRE<MT>, Sc, ReadWrite>,
    partition_scheduler: &PartitionScheduler,
    #[comptime] shared_config: PartitionedStageMatmul,
) {
    #[unroll]
    for m in 0..shared_config.partition_size.m() as usize {
        let m_stage = partition_scheduler.map_m(m as u32);

        #[unroll]
        for n in 0..shared_config.partition_size.n() as usize {
            let n_stage = partition_scheduler.map_n(n as u32);

            let acc_tile =
                acc.partition_tile_at_mut(m, n, shared_config.partition_size.n() as usize);
            let tile = StageAcc::tile::<Sc>(stage, (m_stage, n_stage));
            acc_tile.copy_from::<AccSE<MT>, AccSS<MT>, LhsRE<MT>, RhsRE<MT>, AccRE<MT>, ReadOnly>(
                &tile,
                StageIdent::Acc,
            );
        }
    }
}

#[cube]
fn allocate_lhs<MT: MatmulTypes, Sc: TileScope>(
    #[comptime] layout: MatrixLayout,
    #[comptime] tile_matmul: TileMatmul,
) -> Tile<LhsRE<MT>, Sc, ReadWrite> {
    match tile_matmul {
        TileMatmul::Cmma(c) => cmma_allocate_lhs::<LhsRE<MT>, Sc>(layout, c.tile_size),
        TileMatmul::Mma(c) => mma_allocate_lhs::<LhsRE<MT>, RhsRE<MT>, AccRE<MT>, Sc>(layout, c),
        TileMatmul::Register(c) => register_allocate_lhs::<LhsRE<MT>, Sc>(layout, c),
        TileMatmul::PlaneVec(c) => planevec_allocate_lhs::<LhsRE<MT>, Sc>(layout, c),
        TileMatmul::Interleaved(c) => interleaved_allocate_lhs::<LhsRE<MT>, Sc>(layout, c),
    }
}

#[cube]
fn allocate_rhs<MT: MatmulTypes, Sc: TileScope>(
    #[comptime] layout: MatrixLayout,
    #[comptime] config: TileMatmul,
) -> Tile<RhsRE<MT>, Sc, ReadWrite> {
    match config {
        TileMatmul::Cmma(c) => cmma_allocate_rhs::<RhsRE<MT>, Sc>(layout, c.tile_size),
        TileMatmul::Mma(c) => mma_allocate_rhs::<RhsRE<MT>, LhsRE<MT>, AccRE<MT>, Sc>(layout, c),
        TileMatmul::Register(c) => register_allocate_rhs::<RhsRE<MT>, Sc>(layout, c),
        TileMatmul::PlaneVec(c) => planevec_allocate_rhs::<RhsRE<MT>, Sc>(layout, c),
        TileMatmul::Interleaved(c) => interleaved_allocate_rhs::<RhsRE<MT>, Sc>(layout, c),
    }
}

#[cube]
fn allocate_acc<MT: MatmulTypes, Sc: TileScope>(
    #[comptime] layout: MatrixLayout,
    #[comptime] tile_matmul: TileMatmul,
) -> Tile<AccRE<MT>, Sc, ReadWrite> {
    match tile_matmul {
        TileMatmul::Cmma(c) => cmma_allocate_acc::<AccRE<MT>, Sc>(layout, c.tile_size),
        TileMatmul::Mma(c) => mma_allocate_acc::<AccRE<MT>, LhsRE<MT>, RhsRE<MT>, Sc>(layout, c),
        TileMatmul::Register(c) => register_allocate_acc::<AccRE<MT>, Sc>(layout, c),
        TileMatmul::PlaneVec(c) => planevec_allocate_acc::<AccRE<MT>, Sc>(layout, c),
        TileMatmul::Interleaved(c) => interleaved_allocate_acc::<AccRE<MT>, Sc>(layout, c),
    }
}
