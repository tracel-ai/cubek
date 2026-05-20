//! Per-partition instruction-tile fragments + allocators used by the
//! partition-matmul body. Replaces the cubek-matmul `Accumulators` /
//! `RhsTile` / `allocate_*` helpers; parameterized over individual
//! element types (no `MatmulTypes` dependency).

use std::marker::PhantomData;

use cubecl::prelude::*;

use crate::{
    MatrixLayout, PartitionSize,
    tile::{
        PartitionTile, Tile, TileScope, cmma_allocate_acc, cmma_allocate_lhs, cmma_allocate_rhs,
        interleaved_allocate_acc, interleaved_allocate_lhs, interleaved_allocate_rhs,
        mma_allocate_acc, mma_allocate_lhs, mma_allocate_rhs, planevec_allocate_acc,
        planevec_allocate_lhs, planevec_allocate_rhs, register_allocate_acc, register_allocate_lhs,
        register_allocate_rhs, variants::stage::tile_matmul::TileMatmul,
    },
};

#[derive(CubeType)]
/// Per-partition rhs fragments. Single buffering keeps one fragment;
/// double buffering keeps two and rotates between them to overlap loads
/// with computes.
pub enum RhsTile<Rhs: CubeType> {
    Single(Rhs),
    Double((Rhs, Rhs)),
}

#[cube]
/// Allocate the lhs instruction-tile sequence for one partition.
pub fn allocate_lhs_fragment<L: Numeric, R: Numeric, A: Numeric, Sc: TileScope>(
    #[comptime] partition_size: PartitionSize,
    #[comptime] layout: MatrixLayout,
    #[comptime] tile_matmul: TileMatmul,
) -> Sequence<Tile<L, Sc>> {
    let mut lhs = Sequence::new();

    #[unroll]
    for _ in 0..partition_size.m() {
        lhs.push(allocate_lhs::<L, R, A, Sc>(layout, tile_matmul));
    }

    lhs
}

#[cube]
/// Allocate the rhs instruction-tile (single or double buffered).
pub fn allocate_rhs_fragment<L: Numeric, R: Numeric, A: Numeric, Sc: TileScope>(
    #[comptime] partition_buffering: PartitionBuffering,
    #[comptime] layout: MatrixLayout,
    #[comptime] tile_matmul: TileMatmul,
) -> RhsTile<Tile<R, Sc>> {
    match partition_buffering {
        PartitionBuffering::Single => {
            RhsTile::new_Single(allocate_rhs::<L, R, A, Sc>(layout, tile_matmul))
        }
        PartitionBuffering::Double => RhsTile::new_Double((
            allocate_rhs::<L, R, A, Sc>(layout, tile_matmul),
            allocate_rhs::<L, R, A, Sc>(layout, tile_matmul),
        )),
    }
}

#[cube]
/// Allocate the accumulator partition tile (a `m × n` grid of instruction
/// accumulators). Equivalent to today's `Accumulators::new`.
pub fn allocate_acc_partition<L: Numeric, R: Numeric, A: Numeric, Sc: TileScope>(
    #[comptime] partition_size: PartitionSize,
    #[comptime] layout: MatrixLayout,
    #[comptime] tile_matmul: TileMatmul,
) -> Tile<A, Sc> {
    let mut tiles = Sequence::new();

    #[unroll]
    for _ in 0..partition_size.mn() {
        tiles.push(allocate_acc::<L, R, A, Sc>(layout, tile_matmul));
    }

    Tile::new_Partition(PartitionTile::<A, Sc> {
        tiles,
        rows: comptime!(partition_size.m()),
        cols: comptime!(partition_size.n()),
        _phantom: PhantomData,
    })
}

#[cube]
fn allocate_lhs<L: Numeric, R: Numeric, A: Numeric, Sc: TileScope>(
    #[comptime] layout: MatrixLayout,
    #[comptime] tile_matmul: TileMatmul,
) -> Tile<L, Sc> {
    match tile_matmul {
        TileMatmul::Cmma(c) => cmma_allocate_lhs::<L, Sc>(layout, c.tile_size),
        TileMatmul::Mma(c) => mma_allocate_lhs::<L, R, A, Sc>(layout, c),
        TileMatmul::Register(c) => register_allocate_lhs::<L, Sc>(layout, c),
        TileMatmul::PlaneVec(c) => planevec_allocate_lhs::<L, Sc>(layout, c),
        TileMatmul::Interleaved(c) => interleaved_allocate_lhs::<L, Sc>(layout, c),
    }
}

#[cube]
fn allocate_rhs<L: Numeric, R: Numeric, A: Numeric, Sc: TileScope>(
    #[comptime] layout: MatrixLayout,
    #[comptime] tile_matmul: TileMatmul,
) -> Tile<R, Sc> {
    match tile_matmul {
        TileMatmul::Cmma(c) => cmma_allocate_rhs::<R, Sc>(layout, c.tile_size),
        TileMatmul::Mma(c) => mma_allocate_rhs::<R, L, A, Sc>(layout, c),
        TileMatmul::Register(c) => register_allocate_rhs::<R, Sc>(layout, c),
        TileMatmul::PlaneVec(c) => planevec_allocate_rhs::<R, Sc>(layout, c),
        TileMatmul::Interleaved(c) => interleaved_allocate_rhs::<R, Sc>(layout, c),
    }
}

#[cube]
fn allocate_acc<L: Numeric, R: Numeric, A: Numeric, Sc: TileScope>(
    #[comptime] layout: MatrixLayout,
    #[comptime] tile_matmul: TileMatmul,
) -> Tile<A, Sc> {
    match tile_matmul {
        TileMatmul::Cmma(c) => cmma_allocate_acc::<A, Sc>(layout, c.tile_size),
        TileMatmul::Mma(c) => mma_allocate_acc::<A, L, R, Sc>(layout, c),
        TileMatmul::Register(c) => register_allocate_acc::<A, Sc>(layout, c),
        TileMatmul::PlaneVec(c) => planevec_allocate_acc::<A, Sc>(layout, c),
        TileMatmul::Interleaved(c) => interleaved_allocate_acc::<A, Sc>(layout, c),
    }
}

// Re-export so the body can name it without re-importing.
pub use crate::tile::PartitionBuffering;
