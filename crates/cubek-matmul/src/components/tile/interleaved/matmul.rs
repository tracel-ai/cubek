use std::marker::PhantomData;

use cubecl::prelude::*;
use cubek_std::{MatrixLayout, tile::StridedTile};

use crate::{
    components::tile::{
        Operands, Plane, StandardTileIO, TileMatmul, TileStorage, Tilex,
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
    type Scope = Plane;

    fn execute(
        lhs: &Tilex<L, VL, Self::Scope, ReadOnly>,
        rhs: &Tilex<R, VR, Self::Scope, ReadOnly>,
        acc: &mut Tilex<R, VR, Self::Scope, ReadWrite>,
        #[comptime] config: Self::Config,
    ) {
        todo!()
    }

    fn allocate_lhs(
        #[comptime] layout: MatrixLayout,
        #[comptime] config: Self::Config,
    ) -> Tilex<L, VL, Self::Scope, ReadWrite> {
        todo!()
    }

    fn allocate_rhs(
        #[comptime] layout: MatrixLayout,
        #[comptime] config: Self::Config,
    ) -> Tilex<R, VR, Self::Scope, ReadWrite> {
        todo!()
    }

    fn allocate_acc(
        #[comptime] layout: MatrixLayout,
        #[comptime] config: Self::Config,
    ) -> Tilex<A, VA, Self::Scope, ReadWrite> {
        todo!()
    }

    fn load_lhs<E: Numeric, ES: Size>(
        tile: &Tilex<E, ES, Self::Scope, ReadOnly>,
        lhs: &mut Tilex<L, VL, Self::Scope, ReadWrite>,
        #[comptime] config: Self::Config,
    ) {
        todo!()
    }

    fn load_rhs<E: Numeric, ES: Size>(
        tile: &Tilex<E, ES, Self::Scope, ReadOnly>,
        rhs: &mut Tilex<R, VR, Self::Scope, ReadWrite>,
        #[comptime] config: Self::Config,
    ) {
        todo!()
    }

    fn load_acc<E: Numeric, ES: Size>(
        tile: &Tilex<E, ES, Self::Scope, ReadOnly>,
        acc: &mut Tilex<A, VA, Self::Scope, ReadWrite>,
        #[comptime] config: Self::Config,
    ) {
        todo!()
    }

    fn write_results<E: Numeric, ES: Size>(
        tile: &mut Tilex<E, ES, Self::Scope, ReadWrite>,
        acc: &mut Tilex<A, VA, Self::Scope, ReadWrite>,
        #[comptime] config: Self::Config,
    ) {
        todo!()
    }
}
