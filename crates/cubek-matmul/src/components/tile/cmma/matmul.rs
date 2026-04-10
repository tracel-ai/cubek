use std::marker::PhantomData;

use cubecl::prelude::*;
use cubek_std::{
    tile::{Strided, StridedTile},
    {MatrixLayout, as_cmma_layout},
};

use crate::components::tile::{
    Operands, SharedTileConfig, StandardTileIO, Tilex, TileLayout, TileMatmul, TileScalar,
    TileStorage,
    cmma::{
        reader::{CmmaFragmentReader, CmmaStageReader},
        writer::CmmaStageWriter,
    },
    tile_matmul,
};
use cubecl::cmma;

/// Uses one plane to perform a small matmul using accelerated instructions.
pub struct CmmaMatmul {}

#[derive(CubeType)]
pub struct FragmentOperands<L: Numeric, R: Numeric, A: Numeric> {
    #[cube(comptime)]
    _phantom: PhantomData<(L, R, A)>,
}

impl<L: Numeric, R: Numeric, A: Numeric> Operands for FragmentOperands<L, R, A> {
    type Lhs = TileScalar<L>;
    type Rhs = TileScalar<R>;
    type Acc = TileScalar<A>;
}

#[cube]
impl<L: Numeric, R: Numeric, A: Numeric> TileMatmul<L, R, A> for CmmaMatmul
where
    CmmaStageReader<Option<Strided>>: CmmaFragmentReader,
{
    type Config = SharedTileConfig;
    type Operands = FragmentOperands<L, R, A>;
    type TileIO = StandardTileIO;

    fn execute(
        lhs: &TileScalar<L>,
        rhs: &TileScalar<R>,
        acc: &mut TileScalar<A>,
        #[comptime] _config: Self::Config,
    ) {
        tile_matmul(lhs, rhs, acc);
    }

    fn allocate_lhs(
        #[comptime] layout: MatrixLayout,
        #[comptime] config: Self::Config,
    ) -> TileScalar<L> {
        let size = config.tile_size;

        TileScalar::<L> {
            storage: TileStorage::new_Cmma(unsafe {
                cmma::Matrix::<L>::uninitialized(
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
    ) -> TileScalar<R> {
        let size = config.tile_size;

        TileScalar::<R> {
            storage: TileStorage::new_Cmma(unsafe {
                cmma::Matrix::<R>::uninitialized(
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
    ) -> TileScalar<A> {
        let size = config.tile_size;

        TileScalar::<A> {
            storage: TileStorage::new_Cmma(unsafe {
                cmma::Matrix::<A>::uninitialized(
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

    fn load_lhs<E: Numeric, N: Size>(
        tile: &Tilex<E, N>,
        lhs: &mut TileScalar<L>,
        #[comptime] _config: Self::Config,
    ) {
        CmmaStageReader::<Strided>::load_fragment(
            tile,
            &mut lhs.fragment,
            ComptimeOption::new_None(),
        );
    }

    fn load_rhs<E: Numeric, N: Size>(
        tile: &Tilex<E, N>,
        rhs: &mut TileScalar<R>,
        #[comptime] _config: Self::Config,
    ) {
        CmmaStageReader::<Strided>::load_fragment(
            tile,
            &mut rhs.fragment,
            ComptimeOption::new_None(),
        );
    }

    fn load_acc<E: Numeric, N: Size>(
        tile: &Tilex<E, N>,
        acc: &mut TileScalar<A>,
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
        out: &mut TileScalar<A>,
        #[comptime] _config: Self::Config,
    ) {
        let out = cmma::cast::<A, E>(&out.fragment);
        CmmaStageWriter::store_fragment(tile, &out);
    }
}
