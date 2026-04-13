use std::marker::PhantomData;

use cubecl::prelude::*;
use cubek_std::{
    tile::{Strided, StridedTile},
    {MatrixLayout, as_cmma_layout},
};

use crate::components::tile::{
    NumericVector, Operands, SharedTileConfig, StandardTileIO, TileLayout, TileMatmul, TileStorage,
    Tilex,
    cmma::{
        reader::{CmmaFragmentReader, CmmaStageReader},
        writer::CmmaStageWriter,
    },
    tile_copy, tile_matmul,
};
use cubecl::cmma;

/// Uses one plane to perform a small matmul using accelerated instructions.
pub struct CmmaMatmul {}

#[derive(CubeType)]
pub struct FragmentOperands<L: NumericVector, R: NumericVector, A: NumericVector> {
    #[cube(comptime)]
    _phantom: PhantomData<(L, R, A)>,
}

impl<L: NumericVector, R: NumericVector, A: NumericVector> Operands for FragmentOperands<L, R, A> {
    type Lhs = Tilex<L>;
    type Rhs = Tilex<R>;
    type Acc = Tilex<A>;
}

#[cube]
impl<L: NumericVector, R: NumericVector, A: NumericVector> TileMatmul<L, R, A> for CmmaMatmul
where
    CmmaStageReader<Option<Strided>>: CmmaFragmentReader,
{
    type Config = SharedTileConfig;
    type Operands = FragmentOperands<L, R, A>;
    type TileIO = StandardTileIO;

    fn execute(
        lhs: &Tilex<L>,
        rhs: &Tilex<R>,
        acc: &mut Tilex<A>,
        #[comptime] _config: Self::Config,
    ) {
        tile_matmul(lhs, rhs, acc);
    }

    fn allocate_lhs(
        #[comptime] layout: MatrixLayout,
        #[comptime] config: Self::Config,
    ) -> Tilex<L> {
        let size = config.tile_size;

        Tilex::<L> {
            storage: TileStorage::new_Cmma(unsafe {
                cmma::Matrix::<L::Elem>::uninitialized(
                    cmma::MatrixIdent::A,
                    size.m() as usize,
                    size.n() as usize,
                    size.k() as usize,
                    as_cmma_layout(layout),
                )
            }),
            layout: TileLayout::new_Contiguous(layout),
        }
    }

    fn allocate_rhs(
        #[comptime] layout: MatrixLayout,
        #[comptime] config: Self::Config,
    ) -> Tilex<R> {
        let size = config.tile_size;

        Tilex::<R> {
            storage: TileStorage::new_Cmma(unsafe {
                cmma::Matrix::<R::Elem>::uninitialized(
                    cmma::MatrixIdent::B,
                    size.m() as usize,
                    size.n() as usize,
                    size.k() as usize,
                    as_cmma_layout(layout),
                )
            }),
            layout: TileLayout::new_Contiguous(layout),
        }
    }

    fn allocate_acc(
        #[comptime] layout: MatrixLayout,
        #[comptime] config: Self::Config,
    ) -> Tilex<A> {
        let size = config.tile_size;

        Tilex::<A> {
            storage: TileStorage::new_Cmma(unsafe {
                cmma::Matrix::<A::Elem>::uninitialized(
                    cmma::MatrixIdent::Accumulator,
                    size.m() as usize,
                    size.n() as usize,
                    size.k() as usize,
                    cmma::MatrixLayout::Undefined,
                )
            }),
            layout: TileLayout::new_Contiguous(layout),
        }
    }

    fn load_lhs<N: NumericVector>(
        tile: &Tilex<N>,
        lhs: &mut Tilex<L, ReadWrite>,
        #[comptime] _config: Self::Config,
    ) {
        tile_copy(tile, lhs);
    }

    fn load_rhs<N: NumericVector>(
        tile: &Tilex<N>,
        rhs: &mut Tilex<R, ReadWrite>,
        #[comptime] _config: Self::Config,
    ) {
        tile_copy(tile, rhs);
    }

    fn load_acc<N: NumericVector>(
        tile: &Tilex<N>,
        acc: &mut Tilex<A, ReadWrite>,
        #[comptime] _config: Self::Config,
    ) {
        tile_copy(tile, acc);
    }

    fn write_results<N: NumericVector>(
        tile: &mut Tilex<N, ReadWrite>,
        out: &mut Tilex<A>,
        #[comptime] _config: Self::Config,
    ) {
        let out = out.cast();
        tile_copy(&out, tile)
        // CmmaStageWriter::store_fragment(tile, &out);
    }
}
