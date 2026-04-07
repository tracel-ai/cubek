use cubecl::{define_size, prelude::*};
use cubek_std::{MatrixLayout, tile::Strided, tile::StridedTile, tile::TileKind};
use std::marker::PhantomData;

use crate::components::tile::{
    TileMatmul,
    plane_vec_mat_inner_product::{reader::MatrixFragmentReader, writer::MatrixStageWriter},
};
use crate::{
    components::tile::plane_vec_mat_inner_product::config::PlaneVecMatInnerProductConfig,
    components::tile::plane_vec_mat_inner_product::reader::MatrixStageReader,
    components::tile::plane_vec_mat_inner_product::reader::VectorStageReader,
};

/// Uses one unit to perform a small matmul directly in registers
pub struct PlaneVecMatInnerProduct<Acc: TileKind> {
    _ty: PhantomData<Acc>,
}

define_size!(pub NR);

#[derive(CubeType)]
pub struct VectorContainer<E: Numeric> {
    pub vector: Vector<E, NR>,
}

#[cube]
impl<E: Numeric> VectorContainer<E> {
    fn new() -> VectorContainer<E> {
        VectorContainer::<E> {
            vector: Vector::empty(),
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

    // One vector per unit in the plane
    type LhsFragment = VectorContainer<L>;
    // For each n: one vector per unit in the plane
    type RhsFragment = Sequence<VectorContainer<R>>;

    // For each n: one vector stored at unit pos 0, that will be reduced to a scalar only when writing at the end
    type AccFragment = Sequence<VectorContainer<A>>;

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
            let lhs: Vector<A, NR> = Vector::cast_from(lhs.vector);
            let rhs: Vector<A, NR> = Vector::cast_from(rhs[n].vector);

            plane_sum_vectorized(lhs * rhs, acc.index_mut(n));
        }
    }

    fn allocate_lhs(
        #[comptime] _layout: MatrixLayout,
        #[comptime] config: Self::Config,
    ) -> Self::LhsFragment {
        register_vector_size(config.reduce_vector_size);
        VectorContainer::<L>::new()
    }

    fn allocate_rhs(
        #[comptime] _layout: MatrixLayout,
        #[comptime] config: Self::Config,
    ) -> Self::RhsFragment {
        register_vector_size(config.reduce_vector_size);
        let mut rhs = Sequence::new();
        #[unroll]
        for _ in 0..config.shared.tile_size.n() {
            rhs.push(VectorContainer::new())
        }
        rhs
    }

    fn allocate_acc(
        #[comptime] _layout: MatrixLayout,
        #[comptime] config: Self::Config,
    ) -> Self::AccFragment {
        register_vector_size(config.reduce_vector_size);
        let mut acc = Sequence::new();
        #[unroll]
        for _ in 0..config.shared.tile_size.n() {
            acc.push(VectorContainer::new())
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
            config.reduce_vector_size as usize,
        )
    }
}

#[cube]
fn plane_sum_vectorized<E: Numeric, N: Size>(
    vector_to_sum: Vector<E, N>,
    vector_accumulator: &mut VectorContainer<E>,
) {
    #[unroll]
    #[allow(clippy::explicit_counter_loop)]
    for vector_iterator in 0..N::value() {
        vector_accumulator.vector[vector_iterator] += plane_sum(vector_to_sum[vector_iterator]);
    }
}

#[cube]
#[allow(unused)]
fn register_vector_size(#[comptime] vector_size: u32) {
    intrinsic!(|scope| {
        scope.register_size::<NR>(vector_size as usize);
    })
}
