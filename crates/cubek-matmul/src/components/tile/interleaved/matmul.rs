use cubecl::prelude::*;
use std::marker::PhantomData;

use crate::components::tile::interleaved::config::InterleavedMatmulConfig;
use crate::components::tile::interleaved::reader::InterleavedStageReader;
use crate::components::tile::interleaved::writer::InterleavedStageWriter;
use crate::components::tile::io::Strided;
use crate::components::tile::register::RegisterMatmul;
use crate::components::tile::tile_data::StridedTile;
use crate::components::tile::{TileMatmul, io::Filled};
use crate::definition::{MatrixLayout, StageIdent};

/// Computes a tile matmul where each unit of the plane accumulates an interleaved (by plane_dim)
/// partial dot-product over K.
///
/// Important: the plane must combine those contributions at the end of the global matmul.
pub struct InterleavedMatmul {}

#[derive(CubeType)]
/// InterleavedFragment: each unit owns a stripe of the input tile.
pub struct InterleavedFragment<E: Numeric> {
    pub array: Array<E>,
    #[cube(comptime)]
    pub layout: MatrixLayout,
}

#[derive(CubeType)]
/// InterleavedAccumulator: each unit holds a full accumulator with partial K contributions,
/// combined later via `consolidate`.
pub struct InterleavedAccumulator<E: Numeric> {
    pub array: Array<E>,
    #[cube(comptime)]
    pub layout: MatrixLayout,
    #[cube(comptime)]
    m: usize,
    #[cube(comptime)]
    n: usize,
}

#[cube]
impl<E: Numeric> InterleavedAccumulator<E> {
    /// Every unit will hold the sum
    pub fn consolidate(&mut self) {
        #[unroll]
        for i in 0..self.m * self.n {
            self.array[i] = plane_sum(self.array[i])
        }
    }
}

// u = k / plane_dim (exact division only)

#[cube]
impl<L: Numeric, R: Numeric, A: Numeric> TileMatmul<L, R, A> for InterleavedMatmul {
    type Config = InterleavedMatmulConfig;

    // Size m * u
    type LhsFragment = InterleavedFragment<L>;
    // Size u * n
    type RhsFragment = InterleavedFragment<R>;
    // Size m * n
    type AccFragment = InterleavedAccumulator<A>;

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
        RegisterMatmul::<Self::AccTile>::inner_product(
            &lhs.array,
            &rhs.array,
            &mut acc.array,
            config.local_tile_size(),
        );
    }

    fn allocate_lhs(
        #[comptime] layout: MatrixLayout,
        #[comptime] config: Self::Config,
    ) -> Self::LhsFragment {
        InterleavedFragment::<L> {
            array: Array::new(config.elements_per_unit_m() * config.elements_per_unit_k()),
            layout,
        }
    }

    fn allocate_rhs(
        #[comptime] layout: MatrixLayout,
        #[comptime] config: Self::Config,
    ) -> Self::RhsFragment {
        InterleavedFragment::<R> {
            array: Array::new(config.elements_per_unit_k() * config.elements_per_unit_n()),
            layout,
        }
    }

    fn allocate_acc(
        #[comptime] layout: MatrixLayout,
        #[comptime] config: Self::Config,
    ) -> Self::AccFragment {
        let m = config.elements_per_unit_m();
        let n = config.elements_per_unit_n();
        InterleavedAccumulator::<A> {
            array: Array::new(m * n),
            layout,
            m,
            n,
        }
    }

    fn load_lhs<E: Numeric>(
        tile: &StridedTile<E>,
        lhs: &mut Self::LhsFragment,
        #[comptime] config: Self::Config,
    ) {
        InterleavedStageReader::load_fragment(tile, lhs, StageIdent::Lhs, config);
    }

    fn load_rhs<E: Numeric>(
        tile: &StridedTile<E>,
        rhs: &mut Self::RhsFragment,
        #[comptime] config: Self::Config,
    ) {
        InterleavedStageReader::load_fragment(tile, rhs, StageIdent::Rhs, config);
    }

    fn load_acc<E: Numeric>(
        tile: &E,
        acc: &mut Self::AccFragment,
        #[comptime] config: Self::Config,
    ) {
        InterleavedStageReader::load_accumulator::<A, E>(tile, acc, config);
    }

    fn write_results<E: Numeric>(
        tile: &mut StridedTile<E, ReadWrite>,
        acc: &Self::AccFragment,
        #[comptime] config: Self::Config,
    ) {
        InterleavedStageWriter::store_fragment(tile, acc, config)
    }
}
