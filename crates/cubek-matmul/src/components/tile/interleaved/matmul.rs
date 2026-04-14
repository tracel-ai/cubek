use std::marker::PhantomData;

use cubecl::prelude::*;
use cubek_std::{MatrixLayout, tile::StridedTile};

use crate::{
    components::tile::{
        Operands, Plane, StandardTileIO, TileMatmul,
        interleaved::{
            config::InterleavedMatmulConfig, reader::InterleavedStageReader,
            writer::InterleavedStageWriter,
        },
    },
    definition::StageIdent,
};

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

#[derive(CubeType)]
pub struct InterleavedOperands<L: Numeric, R: Numeric, A: Numeric> {
    #[cube(comptime)]
    _phantom: PhantomData<(L, R, A)>,
}

impl<L: Numeric, R: Numeric, A: Numeric> Operands for InterleavedOperands<L, R, A> {
    // Size m * k_local
    type Lhs = InterleavedFragment<L>;
    // Size k_local * n
    type Rhs = InterleavedFragment<R>;
    // Size m * n
    type Acc = InterleavedAccumulator<A>;
}

#[cube]
impl<L: Numeric, VL: Size, R: Numeric, VR: Size, A: Numeric, VA: Size>
    TileMatmul<L, VL, R, VR, A, VA> for InterleavedMatmul
{
    type Config = InterleavedMatmulConfig;
    type Operands = InterleavedOperands<L, R, A>;
    type TileIO = StandardTileIO;
    type Scope = Plane;

    fn execute(
        lhs: &InterleavedFragment<L>,
        rhs: &InterleavedFragment<R>,
        acc: &mut InterleavedAccumulator<A>,
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
    ) -> InterleavedFragment<L> {
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
    ) -> InterleavedFragment<R> {
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
    ) -> InterleavedAccumulator<A> {
        let m = config.elements_per_unit_m();
        let n = config.elements_per_unit_n();
        InterleavedAccumulator::<A> {
            array: Array::new(m * n),
            layout,
            m,
            n,
        }
    }

    fn load_lhs<E: Numeric, N: Size>(
        tile: &StridedTile<E, N>,
        lhs: &mut InterleavedFragment<L>,
        #[comptime] config: Self::Config,
    ) {
        InterleavedStageReader::load_fragment(tile, lhs, StageIdent::Lhs, config);
    }

    fn load_rhs<E: Numeric, N: Size>(
        tile: &StridedTile<E, N>,
        rhs: &mut InterleavedFragment<R>,
        #[comptime] config: Self::Config,
    ) {
        InterleavedStageReader::load_fragment(tile, rhs, StageIdent::Rhs, config);
    }

    fn load_acc<E: Numeric, N: Size>(
        tile: &ComptimeOption<StridedTile<E, N>>,
        acc: &mut InterleavedAccumulator<A>,
        #[comptime] config: Self::Config,
    ) {
        match tile {
            ComptimeOption::Some(_) => {
                todo!("Not yet implemented")
            }
            ComptimeOption::None => {
                let value = E::from_int(0);
                InterleavedStageReader::load_accumulator::<A, E>(&value, acc, config);
            }
        }
    }

    fn write_results<E: Numeric, N: Size>(
        tile: &mut StridedTile<E, N, ReadWrite>,
        acc: &mut InterleavedAccumulator<A>,
        #[comptime] config: Self::Config,
    ) {
        acc.consolidate();
        InterleavedStageWriter::store_fragment(tile, acc, config)
    }
}
