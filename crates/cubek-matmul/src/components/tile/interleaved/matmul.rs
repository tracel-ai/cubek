use cubecl::prelude::*;
use std::marker::PhantomData;

use crate::components::tile::interleaved::config::InterleavedMatmulConfig;
use crate::components::tile::io::Strided;
use crate::components::tile::{TileMatmul, io::Filled};
use crate::components::tile::{io::TileKind, tile_data::StridedTile};
use crate::definition::MatrixLayout;

/// Computes a tile matmul where each unit of the plane accumulates an interleaved (by plane_dim)
/// partial dot-product over K.
///
/// Important: the plane must combine those contributions at the end of the global matmul.
pub struct InterleavedMatmul<Acc: TileKind = Filled> {
    _ty: PhantomData<Acc>,
}

#[derive(CubeType)]
pub struct UnitFragment<E: Numeric> {
    pub array: Array<E>,
    #[cube(comptime)]
    pub layout: MatrixLayout,
}

#[cube]
impl<L: Numeric, R: Numeric, A: Numeric> TileMatmul<L, R, A> for InterleavedMatmul {
    type Config = InterleavedMatmulConfig;

    type LhsFragment = UnitFragment<L>;
    type RhsFragment = UnitFragment<R>;
    type AccFragment = UnitFragment<A>;

    type LhsTile = Strided;
    type RhsTile = Strided;
    type AccTile = Filled;
    type OutTile = Strided;

    fn execute(
        lhs: &Self::LhsFragment,
        rhs: &Self::RhsFragment,
        acc: &mut Self::AccFragment,
        #[comptime] config: Self::Config,
    ) {
    }

    fn allocate_lhs(
        #[comptime] layout: MatrixLayout,
        #[comptime] config: Self::Config,
    ) -> Self::LhsFragment {
        todo!()
    }

    fn allocate_rhs(
        #[comptime] layout: MatrixLayout,
        #[comptime] config: Self::Config,
    ) -> Self::RhsFragment {
        todo!()
    }

    fn allocate_acc(
        #[comptime] layout: MatrixLayout,
        #[comptime] config: Self::Config,
    ) -> Self::AccFragment {
        todo!()
    }

    fn load_lhs<E: Numeric>(
        tile: &StridedTile<E>,
        lhs: &mut Self::LhsFragment,
        #[comptime] config: Self::Config,
    ) {
    }

    fn load_rhs<E: Numeric>(
        tile: &StridedTile<E>,
        rhs: &mut Self::RhsFragment,
        #[comptime] config: Self::Config,
    ) {
    }

    fn load_acc<E: Numeric>(
        tile: &E,
        acc: &mut Self::AccFragment,
        #[comptime] config: Self::Config,
    ) {
    }

    fn write_results<E: Numeric>(
        tile: &mut StridedTile<E, ReadWrite>,
        acc: &Self::AccFragment,
        #[comptime] config: Self::Config,
    ) {
    }
}
