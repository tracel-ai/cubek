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

#[derive(CubeType)]
pub struct FragmentOperands<L: Numeric, VL: Size, R: Numeric, VR: Size, A: Numeric, VA: Size> {
    #[cube(comptime)]
    _phantom: PhantomData<(L, VL, R, VR, A, VA)>,
}

impl<L: Numeric, VL: Size, R: Numeric, VR: Size, A: Numeric, VA: Size> Operands
    for FragmentOperands<L, VL, R, VR, A, VA>
{
    type Lhs = Tilex<L, VL, Plane, ReadWrite>;
    type Rhs = Tilex<R, VR, Plane, ReadWrite>;
    type Acc = Tilex<A, VA, Plane, ReadWrite>;
}

#[cube]
impl<L: Numeric, VL: Size, R: Numeric, VR: Size, A: Numeric, VA: Size>
    TileMatmul<L, VL, R, VR, A, VA> for CmmaMatmul
where
    CmmaStageReader<Option<Strided>>: CmmaFragmentReader,
{
    type Config = SharedTileConfig;
    type Operands = FragmentOperands<L, VL, R, VR, A, VA>;
    type TileIO = StandardTileIO;
    type Scope = Plane;

    fn execute(
        lhs: &Tilex<L, VL, Self::Scope, ReadOnly>,
        rhs: &Tilex<R, VR, Self::Scope, ReadOnly>,
        acc: &mut Tilex<A, VA, Self::Scope, ReadWrite>,
        #[comptime] _config: Self::Config,
    ) {
        tile_matmul(lhs, rhs, acc);
    }

    fn allocate_lhs(
        #[comptime] layout: MatrixLayout,
        #[comptime] config: Self::Config,
    ) -> Tilex<L, VL, Self::Scope, ReadWrite> {
        let size = config.tile_size;

        Tilex::<L, VL, Self::Scope, ReadWrite> {
            storage: TileStorage::new_Cmma(unsafe {
                cmma::Matrix::<L>::uninitialized(
                    cmma::MatrixIdent::A,
                    size.m() as usize,
                    size.n() as usize,
                    size.k() as usize,
                    as_cmma_layout(layout),
                )
            }),
            layout: TileLayout::new_contiguous(layout),
            _scope: PhantomData,
        }
    }

    fn allocate_rhs(
        #[comptime] layout: MatrixLayout,
        #[comptime] config: Self::Config,
    ) -> Tilex<R, VR, Self::Scope, ReadWrite> {
        let size = config.tile_size;

        Tilex::<R, VR, Self::Scope, ReadWrite> {
            storage: TileStorage::new_Cmma(unsafe {
                cmma::Matrix::<R>::uninitialized(
                    cmma::MatrixIdent::B,
                    size.m() as usize,
                    size.n() as usize,
                    size.k() as usize,
                    as_cmma_layout(layout),
                )
            }),
            layout: TileLayout::new_contiguous(layout),
            _scope: PhantomData,
        }
    }

    fn allocate_acc(
        #[comptime] layout: MatrixLayout,
        #[comptime] config: Self::Config,
    ) -> Tilex<A, VA, Self::Scope, ReadWrite> {
        let size = config.tile_size;

        Tilex::<A, VA, Self::Scope, ReadWrite> {
            storage: TileStorage::new_Cmma(unsafe {
                cmma::Matrix::<A>::uninitialized(
                    cmma::MatrixIdent::Accumulator,
                    size.m() as usize,
                    size.n() as usize,
                    size.k() as usize,
                    cmma::MatrixLayout::Undefined,
                )
            }),
            layout: TileLayout::new_contiguous(layout),
            _scope: PhantomData,
        }
    }

    fn load_lhs<E: Numeric>(
        tile: &Tilex<L, VL, Self::Scope, ReadOnly>,
        lhs: &mut Tilex<L, VL, Self::Scope, ReadWrite>,
        #[comptime] _config: Self::Config,
    ) {
        tile_copy(tile, lhs);
    }

    fn load_rhs<E: Numeric>(
        tile: &Tilex<E, VR, Self::Scope, ReadOnly>,
        rhs: &mut Tilex<R, VR, Self::Scope, ReadWrite>,
        #[comptime] _config: Self::Config,
    ) {
        tile_copy(tile, rhs);
    }

    fn load_acc<E: Numeric>(
        tile: &Tilex<E, VA, Self::Scope, ReadOnly>,
        acc: &mut Tilex<A, VA, Self::Scope, ReadWrite>,
        #[comptime] _config: Self::Config,
    ) {
        tile_copy(tile, acc);
    }

    fn write_results<E: Numeric>(
        tile: &mut Tilex<E, VA, Self::Scope, ReadWrite>,
        out: &mut Tilex<A, VA, Self::Scope, ReadOnly>,
        #[comptime] _config: Self::Config,
    ) {
        // let out: Tilex<N, Self::Scope> = out.cast();
        tile_copy(&out, tile)
        // CmmaStageWriter::store_fragment(tile, &out);
    }
}
