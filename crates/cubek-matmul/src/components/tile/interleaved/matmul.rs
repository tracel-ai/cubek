use cubecl::{
    prelude::*,
    std::{CubeOption, CubeOptionExpand},
};

use crate::components::tile::TileMatmul;
use crate::components::tile::interleaved::config::InterleavedMatmulConfig;
use crate::components::tile::interleaved::reader::InterleavedStageReader;
use crate::components::tile::interleaved::writer::InterleavedStageWriter;
use crate::components::tile::io::Strided;
use crate::components::tile::tile_data::StridedTile;
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
    #[cube(comptime)]
    row_count: usize,
    #[cube(comptime)]
    col_count: usize,
}

#[cube]
impl<E: Numeric> InterleavedFragment<E> {
    fn get(&self, i: usize, j: usize) -> E {
        match comptime!(self.layout) {
            MatrixLayout::RowMajor => self.array[i * self.col_count + j],
            MatrixLayout::ColMajor => self.array[j * self.row_count + i],
        }
    }
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
        for i in 0..comptime!(self.m * self.n) {
            self.array[i] = plane_sum(self.array[i])
        }
    }
}

#[cube]
impl<L: Numeric, R: Numeric, A: Numeric> TileMatmul<L, R, A> for InterleavedMatmul {
    type Config = InterleavedMatmulConfig;

    // Size m * k_local
    type LhsFragment = InterleavedFragment<L>;
    // Size k_local * n
    type RhsFragment = InterleavedFragment<R>;
    // Size m * n
    type AccFragment = InterleavedAccumulator<A>;

    type LhsTile = Strided;
    type RhsTile = Strided;
    type AccTile = CubeOption<Strided>;
    type OutTile = Strided;

    fn execute(
        lhs: &Self::LhsFragment,
        rhs: &Self::RhsFragment,
        acc: &mut Self::AccFragment,
        #[comptime] config: Self::Config,
    ) {
        let m = config.elements_per_unit_m();
        let n = config.elements_per_unit_n();
        let local_k = config.elements_per_unit_k();

        #[unroll]
        for m_ in 0..m {
            #[unroll]
            for n_ in 0..n {
                #[unroll]
                for k_ in 0..local_k {
                    let lhs_elem = A::cast_from(lhs.get(m_, k_));
                    let rhs_elem = A::cast_from(rhs.get(k_, n_));
                    acc.array[m_ * n + n_] += lhs_elem * rhs_elem;
                }
            }
        }
    }

    fn allocate_lhs(
        #[comptime] layout: MatrixLayout,
        #[comptime] config: Self::Config,
    ) -> Self::LhsFragment {
        let row_count = config.elements_per_unit_m();
        let col_count = config.elements_per_unit_k();
        InterleavedFragment::<L> {
            array: Array::new(row_count * col_count),
            layout,
            row_count,
            col_count,
        }
    }

    fn allocate_rhs(
        #[comptime] layout: MatrixLayout,
        #[comptime] config: Self::Config,
    ) -> Self::RhsFragment {
        let row_count = config.elements_per_unit_k();
        let col_count = config.elements_per_unit_n();
        InterleavedFragment::<R> {
            array: Array::new(row_count * col_count),
            layout,
            row_count,
            col_count,
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
        tile: &CubeOption<StridedTile<E>>,
        acc: &mut Self::AccFragment,
        #[comptime] config: Self::Config,
    ) {
        match tile {
            CubeOption::Some(_) => {
                todo!("Not yet implemented")
            }
            CubeOption::None => {
                let value = E::from_int(0);
                InterleavedStageReader::load_accumulator::<A, E>(&value, acc, config);
            }
        }
    }

    fn write_results<E: Numeric>(
        tile: &mut StridedTile<E, ReadWrite>,
        acc: &mut Self::AccFragment,
        #[comptime] config: Self::Config,
    ) {
        acc.consolidate();
        InterleavedStageWriter::store_fragment(tile, acc, config)
    }
}
