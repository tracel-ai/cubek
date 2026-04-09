use cubecl::prelude::*;
use cubek_std::{
    tile::{Strided, StridedTile},
    {MatrixLayout, as_cmma_layout},
};

use crate::components::tile::{
    SharedTileConfig, StandardTileIO, TileMatmul,
    cmma::{
        reader::{CmmaFragmentReader, CmmaStageReader},
        writer::CmmaStageWriter,
    },
};
use cubecl::cmma;

/// Uses one plane to perform a small matmul using accelerated instructions.
pub struct CmmaMatmul {}

#[derive(CubeType)]
pub struct Fragment<E: Numeric> {
    fragment: cmma::Matrix<E>,
    #[cube(comptime)]
    layout: MatrixLayout,
}

#[cube]
impl<L: Numeric, R: Numeric, A: Numeric> TileMatmul<L, R, A> for CmmaMatmul
where
    CmmaStageReader<Option<Strided>>: CmmaFragmentReader,
{
    type Config = SharedTileConfig;

    type LhsFragment = Fragment<L>;
    type RhsFragment = Fragment<R>;
    type AccFragment = Fragment<A>;

    type TileIO = StandardTileIO;

    fn execute(
        lhs: &Self::LhsFragment,
        rhs: &Self::RhsFragment,
        out: &mut Self::AccFragment,
        #[comptime] _config: Self::Config,
    ) {
        cmma::execute::<L, R, A, A>(&lhs.fragment, &rhs.fragment, &out.fragment, &out.fragment);
    }

    fn allocate_lhs(
        #[comptime] layout: MatrixLayout,
        #[comptime] config: Self::Config,
    ) -> Self::LhsFragment {
        let size = config.tile_size;

        Fragment::<L> {
            fragment: unsafe {
                cmma::Matrix::<L>::uninitialized(
                    cmma::MatrixIdent::A,
                    size.m() as usize,
                    size.n() as usize,
                    size.k() as usize,
                    as_cmma_layout(layout),
                )
            },
            layout,
        }
    }

    fn allocate_rhs(
        #[comptime] layout: MatrixLayout,
        #[comptime] config: Self::Config,
    ) -> Self::RhsFragment {
        let size = config.tile_size;

        Fragment::<R> {
            fragment: unsafe {
                cmma::Matrix::uninitialized(
                    cmma::MatrixIdent::B,
                    size.m() as usize,
                    size.n() as usize,
                    size.k() as usize,
                    as_cmma_layout(layout),
                )
            },
            layout,
        }
    }

    fn allocate_acc(
        #[comptime] layout: MatrixLayout,
        #[comptime] config: Self::Config,
    ) -> Self::AccFragment {
        let size = config.tile_size;

        Fragment::<A> {
            fragment: unsafe {
                cmma::Matrix::<A>::uninitialized(
                    cmma::MatrixIdent::Accumulator,
                    size.m() as usize,
                    size.n() as usize,
                    size.k() as usize,
                    cmma::MatrixLayout::Undefined,
                )
            },
            layout,
        }
    }

    fn load_lhs<E: Numeric, N: Size>(
        tile: &StridedTile<E, N>,
        lhs: &mut Self::LhsFragment,
        #[comptime] _config: Self::Config,
    ) {
        CmmaStageReader::<Strided>::load_fragment(
            tile,
            &mut lhs.fragment,
            ComptimeOption::new_None(),
        );
    }

    fn load_rhs<E: Numeric, N: Size>(
        tile: &StridedTile<E, N>,
        rhs: &mut Self::RhsFragment,
        #[comptime] _config: Self::Config,
    ) {
        CmmaStageReader::<Strided>::load_fragment(
            tile,
            &mut rhs.fragment,
            ComptimeOption::new_None(),
        );
    }

    fn load_acc<E: Numeric, N: Size>(
        tile: &ComptimeOption<StridedTile<E, N>>,
        acc: &mut Self::AccFragment,
        #[comptime] _config: Self::Config,
    ) {
        CmmaStageReader::<Option<Strided>>::load_fragment(
            tile,
            &mut acc.fragment,
            ComptimeOption::new_Some(as_cmma_layout(acc.layout)),
        );
    }

    fn write_results<E: Numeric, N: Size>(
        tile: &mut StridedTile<E, N, ReadWrite>,
        out: &mut Self::AccFragment,
        #[comptime] _config: Self::Config,
    ) {
        let out = cmma::cast::<A, E>(&out.fragment);
        CmmaStageWriter::store_fragment(tile, &out);
    }
}
