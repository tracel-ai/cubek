use cubecl::{define_size, prelude::*};
use cubek_std::MatrixLayout;
use cubek_std::tile::Strided;
use cubek_std::tile::StridedTile;
use cubek_std::tile::TileKind;
use std::marker::PhantomData;

use crate::components::tile::plane_vec_mat_inner_product::config::PlaneVecMatInnerProductConfig;
use crate::components::tile::plane_vec_mat_inner_product::reader::MatrixStageReader;
use crate::components::tile::plane_vec_mat_inner_product::reader::VectorStageReader;
use crate::components::tile::{
    TileMatmul,
    plane_vec_mat_inner_product::{reader::MatrixFragmentReader, writer::MatrixStageWriter},
};

/// Uses one unit to perform a small matmul directly in registers
pub struct PlaneVecMatInnerProduct<Acc: TileKind> {
    _ty: PhantomData<Acc>,
}

define_size!(NR);

#[derive(CubeType)]
pub struct LineContainer<E: Numeric> {
    pub line: Line<E, NR>,
}

#[cube]
impl<E: Numeric> LineContainer<E> {
    fn new() -> LineContainer<E> {
        LineContainer::<E> {
            line: Line::empty(),
        }
    }
}

#[cube]
impl<L: Numeric, R: Numeric, A: Numeric, AccTile: TileKind> TileMatmul<L, R, A>
    for PlaneVecMatInnerProduct<AccTile>
where
    MatrixStageReader<AccTile>: MatrixFragmentReader<TileKind = AccTile>,
{
    type Config = PlaneVecMatInnerProductConfig;

    // One line per unit in the plane
    type LhsFragment = LineContainer<L>;
    // For each n: one line per unit in the plane
    type RhsFragment = Sequence<LineContainer<R>>;

    // For each n: one line stored at unit pos 0, that will be reduced to a scalar only when writing at the end
    type AccFragment = Sequence<LineContainer<A>>;

    type LhsTile = Strided;
    type RhsTile = Strided;
    type AccTile = AccTile;
    type OutTile = Strided;

    fn execute(
        lhs: &Self::LhsFragment,
        rhs: &Self::RhsFragment,
        acc: &mut Self::AccFragment,
        #[comptime] config: Self::Config,
    ) {
        #[unroll]
        #[allow(clippy::explicit_counter_loop)]
        for n in 0..config.shared.tile_size.n() as usize {
            let lhs: Line<A, NR> = Line::cast_from(lhs.line);
            let rhs: Line<A, NR> = Line::cast_from(rhs[n].line);

            plane_sum_lined(
                lhs * rhs,
                acc.index_mut(n),
                config.reduce_line_size as usize,
            );
        }
    }

    fn allocate_lhs(
        #[comptime] _layout: MatrixLayout,
        #[comptime] config: Self::Config,
    ) -> Self::LhsFragment {
        register_line_size(config.reduce_line_size);
        LineContainer::<L>::new()
    }

    fn allocate_rhs(
        #[comptime] _layout: MatrixLayout,
        #[comptime] config: Self::Config,
    ) -> Self::RhsFragment {
        register_line_size(config.reduce_line_size);
        let mut rhs = Sequence::new();
        #[unroll]
        for _ in 0..config.shared.tile_size.n() {
            rhs.push(LineContainer::new())
        }
        rhs
    }

    fn allocate_acc(
        #[comptime] _layout: MatrixLayout,
        #[comptime] config: Self::Config,
    ) -> Self::AccFragment {
        register_line_size(config.reduce_line_size);
        let mut acc = Sequence::new();
        #[unroll]
        for _ in 0..config.shared.tile_size.n() {
            acc.push(LineContainer::new())
        }
        acc
    }

    fn load_lhs<E: Numeric, N: Size>(
        tile: &StridedTile<E, N>,
        lhs: &mut Self::LhsFragment,
        #[comptime] _config: Self::Config,
    ) {
        VectorStageReader::load_fragment(tile, lhs)
    }

    fn load_rhs<E: Numeric, N: Size>(
        tile: &StridedTile<E, N>,
        rhs: &mut Self::RhsFragment,
        #[comptime] config: Self::Config,
    ) {
        MatrixStageReader::<Strided>::load_fragment(tile, rhs, config.shared.tile_size.n())
    }

    fn load_acc<E: Numeric, N: Size>(
        tile: &AccTile::Tile<E, N>,
        acc: &mut Self::AccFragment,
        #[comptime] config: Self::Config,
    ) {
        MatrixStageReader::<AccTile>::load_fragment(tile, acc, config.shared.tile_size.n());
    }

    fn write_results<E: Numeric, N: Size>(
        tile: &mut StridedTile<E, N, ReadWrite>,
        acc: &mut Self::AccFragment,
        #[comptime] config: Self::Config,
    ) {
        MatrixStageWriter::store_fragment(
            tile,
            acc,
            config.shared.tile_size.n(),
            config.reduce_line_size as usize,
        )
    }
}

#[cube]
fn plane_sum_lined<E: Numeric, N: Size>(
    line_to_sum: Line<E, N>,
    line_accumulator: &mut LineContainer<E>,
    #[comptime] line_size: LineSize,
) {
    #[unroll]
    #[allow(clippy::explicit_counter_loop)]
    for line_iterator in 0..line_size {
        line_accumulator.line[line_iterator] += plane_sum(line_to_sum[line_iterator]);
    }
}

#[cube]
#[allow(unused)]
fn register_line_size(#[comptime] line_size: u32) {
    intrinsic!(|scope| {
        scope.register_size::<NR>(line_size as usize);
    })
}
