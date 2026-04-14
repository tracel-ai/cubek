use std::marker::PhantomData;

use cubecl::prelude::*;
use cubek_std::{
    tile::Strided,
    {MatrixLayout, as_cmma_layout},
};

use crate::components::tile::{
    Operands, Plane, SharedTileConfig, StandardTileIO, TileLayout, TileMatmul, TileStorage, Tilex,
    cmma::reader::{CmmaFragmentReader, CmmaStageReader},
    tile_copy, tile_matmul,
};
use cubecl::cmma;

/// Uses one plane to perform a small matmul using accelerated instructions.
pub struct CmmaMatmul {}

// #[derive(CubeType)]
// pub struct FragmentOperands<L: Numeric, VL: Size, R: Numeric, VR: Size, A: Numeric, VA: Size> {
//     #[cube(comptime)]
//     _phantom: PhantomData<(L, VL, R, VR, A, VA)>,
// }

// impl<L: Numeric, VL: Size, R: Numeric, VR: Size, A: Numeric, VA: Size> Operands
//     for FragmentOperands<L, VL, R, VR, A, VA>
// {
//     type Lhs = Tilex<L, VL, Plane, ReadOnly>;
//     type Rhs = Tilex<R, VR, Plane, ReadOnly>;
//     type Acc = Tilex<A, VA, Plane, ReadWrite>;
// }

#[cube]
impl<L: Numeric, VL: Size, R: Numeric, VR: Size, A: Numeric, VA: Size>
    TileMatmul<L, VL, R, VR, A, VA> for CmmaMatmul
{
    type Config = SharedTileConfig;
    type Scope = Plane;

    fn execute(
        lhs: &Tilex<L, VL, Self::Scope, ReadWrite>,
        rhs: &Tilex<R, VR, Self::Scope, ReadWrite>,
        acc: &mut Tilex<A, VA, Self::Scope, ReadWrite>,
        #[comptime] _config: Self::Config,
    ) {
        todo!()
    }

    fn allocate_lhs(
        #[comptime] layout: MatrixLayout,
        #[comptime] config: Self::Config,
    ) -> Tilex<L, VL, Self::Scope, ReadWrite> {
        todo!()
    }

    fn allocate_rhs(
        #[comptime] layout: MatrixLayout,
        #[comptime] config: Self::Config,
    ) -> Tilex<R, VR, Self::Scope, ReadWrite> {
        todo!()
    }

    fn allocate_acc(
        #[comptime] layout: MatrixLayout,
        #[comptime] config: Self::Config,
    ) -> Tilex<A, VA, Self::Scope, ReadWrite> {
        todo!()
    }

    fn load_lhs<E: Numeric, ES: Size>(
        tile: &Tilex<E, ES, Self::Scope, ReadOnly>,
        lhs: &mut Tilex<L, VL, Self::Scope, ReadWrite>,
        #[comptime] _config: Self::Config,
    ) {
        todo!()
    }

    fn load_rhs<E: Numeric, ES: Size>(
        tile: &Tilex<E, ES, Self::Scope, ReadOnly>,
        rhs: &mut Tilex<R, VR, Self::Scope, ReadWrite>,
        #[comptime] _config: Self::Config,
    ) {
        todo!()
    }

    fn load_acc<E: Numeric, ES: Size>(
        tile: &Tilex<E, ES, Self::Scope, ReadOnly>,
        acc: &mut Tilex<A, VA, Self::Scope, ReadWrite>,
        #[comptime] _config: Self::Config,
    ) {
        todo!()
    }

    fn write_results<E: Numeric, ES: Size>(
        tile: &mut Tilex<E, ES, Self::Scope, ReadWrite>,
        out: &mut Tilex<A, VA, Self::Scope, ReadWrite>,
        #[comptime] _config: Self::Config,
    ) {
        todo!()
    }
}
