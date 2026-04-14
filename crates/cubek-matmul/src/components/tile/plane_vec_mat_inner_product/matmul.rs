use cubecl::{define_size, prelude::*};
use cubek_std::MatrixLayout;

use crate::components::tile::{
    TileMatmul, Tilex,
    plane_vec_mat_inner_product::config::PlaneVecMatInnerProductConfig,
    tilex_allocate, tilex_execute, tilex_load, tilex_write,
};

/// Uses one unit to perform a small matmul directly in registers
pub struct PlaneVecMatInnerProduct {}

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

// #[derive(CubeType)]
// pub struct VectorOperands<L: Numeric, R: Numeric, A: Numeric> {
//     #[cube(comptime)]
//     _phantom: PhantomData<(L, R, A)>,
// }

// impl<L: Numeric, R: Numeric, A: Numeric> Operands for VectorOperands<L, R, A> {
//     type Lhs = VectorContainer<L>;
//     type Rhs = Sequence<VectorContainer<R>>;
//     type Acc = Sequence<VectorContainer<A>>;
// }

#[cube]
impl<L: Numeric, VL: Size, R: Numeric, VR: Size, A: Numeric, VA: Size>
    TileMatmul<L, VL, R, VR, A, VA> for PlaneVecMatInnerProduct
{
    type Config = PlaneVecMatInnerProductConfig;

    // TODO dummy
    type Scope = u32;

    fn execute(
        lhs: &Tilex<L, VL, Self::Scope, ReadWrite>,
        rhs: &Tilex<R, VR, Self::Scope, ReadWrite>,
        acc: &mut Tilex<A, VA, Self::Scope, ReadWrite>,
        #[comptime] _config: Self::Config,
    ) {
        tilex_execute(lhs, rhs, acc);
    }

    fn allocate_lhs(
        #[comptime] layout: MatrixLayout,
        #[comptime] _config: Self::Config,
    ) -> Tilex<L, VL, Self::Scope, ReadWrite> {
        tilex_allocate(layout)
    }

    fn allocate_rhs(
        #[comptime] layout: MatrixLayout,
        #[comptime] _config: Self::Config,
    ) -> Tilex<R, VR, Self::Scope, ReadWrite> {
        tilex_allocate(layout)
    }

    fn allocate_acc(
        #[comptime] layout: MatrixLayout,
        #[comptime] _config: Self::Config,
    ) -> Tilex<A, VA, Self::Scope, ReadWrite> {
        tilex_allocate(layout)
    }

    fn load_lhs<E: Numeric, ES: Size>(
        tile: &Tilex<E, ES, Self::Scope, ReadOnly>,
        lhs: &mut Tilex<L, VL, Self::Scope, ReadWrite>,
        #[comptime] _config: Self::Config,
    ) {
        tilex_load(tile, lhs);
    }

    fn load_rhs<E: Numeric, ES: Size>(
        tile: &Tilex<E, ES, Self::Scope, ReadOnly>,
        rhs: &mut Tilex<R, VR, Self::Scope, ReadWrite>,
        #[comptime] _config: Self::Config,
    ) {
        tilex_load(tile, rhs);
    }

    fn load_acc<E: Numeric, ES: Size>(
        tile: &Tilex<E, ES, Self::Scope, ReadOnly>,
        acc: &mut Tilex<A, VA, Self::Scope, ReadWrite>,
        #[comptime] _config: Self::Config,
    ) {
        tilex_load(tile, acc);
    }

    fn write_results<E: Numeric, ES: Size>(
        tile: &mut Tilex<E, ES, Self::Scope, ReadWrite>,
        acc: &mut Tilex<A, VA, Self::Scope, ReadWrite>,
        #[comptime] _config: Self::Config,
    ) {
        tilex_write(tile, acc);
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
