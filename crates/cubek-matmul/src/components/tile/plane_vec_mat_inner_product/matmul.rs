use cubecl::prelude::*;
use std::marker::PhantomData;

use crate::components::tile::plane_vec_mat_inner_product::config::PlaneVecMatInnerProductConfig;
use crate::components::tile::{
    TileMatmul,
    plane_vec_mat_inner_product::{reader::MatrixFragmentReader, writer::MatrixStageWriter},
};
use crate::components::tile::{
    io::Strided, plane_vec_mat_inner_product::reader::MatrixStageReader, tile_data::StridedTile,
};
use crate::components::tile::{
    io::TileKind, plane_vec_mat_inner_product::reader::VectorStageReader,
};
use crate::definition::MatrixLayout;

/// Uses one unit to perform a small matmul directly in registers
pub struct PlaneVecMatInnerProduct<Acc: TileKind> {
    _ty: PhantomData<Acc>,
}

#[derive(CubeType)]
pub struct LineContainer<E: Numeric> {
    pub line: Line<E>,
}

#[cube]
impl<E: Numeric> LineContainer<E> {
    fn new(#[comptime] size: LineSize) -> LineContainer<E> {
        LineContainer::<E> {
            line: Line::empty(size),
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
            let lhs: Line<A> = Line::cast_from(lhs.line);
            let rhs: Line<A> = Line::cast_from(rhs[n].line);

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
        LineContainer::<L>::new(config.reduce_line_size as usize)
    }

    fn allocate_rhs(
        #[comptime] _layout: MatrixLayout,
        #[comptime] config: Self::Config,
    ) -> Self::RhsFragment {
        let mut rhs = Sequence::new();
        #[unroll]
        for _ in 0..config.shared.tile_size.n() {
            rhs.push(LineContainer::new(config.reduce_line_size as usize))
        }
        rhs
    }

    fn allocate_acc(
        #[comptime] _layout: MatrixLayout,
        #[comptime] config: Self::Config,
    ) -> Self::AccFragment {
        let mut acc = Sequence::new();
        #[unroll]
        for _ in 0..config.shared.tile_size.n() {
            acc.push(LineContainer::new(config.reduce_line_size as usize))
        }
        acc
    }

    fn load_lhs<E: Numeric>(
        tile: &StridedTile<E>,
        lhs: &mut Self::LhsFragment,
        #[comptime] _config: Self::Config,
    ) {
        VectorStageReader::load_fragment(tile, lhs)
    }

    fn load_rhs<E: Numeric>(
        tile: &StridedTile<E>,
        rhs: &mut Self::RhsFragment,
        #[comptime] config: Self::Config,
    ) {
        MatrixStageReader::<Strided>::load_fragment(tile, rhs, config.shared.tile_size.n())
    }

    fn load_acc<E: Numeric>(
        tile: &AccTile::Tile<E>,
        acc: &mut Self::AccFragment,
        #[comptime] config: Self::Config,
    ) {
        MatrixStageReader::<AccTile>::load_fragment(tile, acc, config.shared.tile_size.n());
    }

    fn write_results<E: Numeric>(
        tile: &mut StridedTile<E, ReadWrite>,
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
fn plane_sum_lined<E: Numeric>(
    line_to_sum: Line<E>,
    line_accumulator: &mut LineContainer<E>,
    #[comptime] line_size: LineSize,
) {
    #[unroll]
    #[allow(clippy::explicit_counter_loop)]
    for line_iterator in 0..line_size {
        line_accumulator.line[line_iterator] += plane_sum(line_to_sum[line_iterator]);
    }
}
