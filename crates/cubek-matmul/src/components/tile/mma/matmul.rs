use cubecl::{define_size, prelude::*};
use cubek_std::MatrixLayout;

use crate::components::tile::{
    TileMatmul, Tilex, mma::config::MmaMatmulConfig,
    tilex_allocate, tilex_execute, tilex_load, tilex_write,
};
use cubecl::{cmma::MmaDefinition, ir::MatrixIdent};

/// Uses one plane to perform a small matmul using accelerated instructions, with manual register
/// management.
/// Currently requires matrix layout to match the platform's preferred layout.
pub struct MmaMatmul {}

define_size!(pub NL);
define_size!(pub NR);
define_size!(pub NA);

#[derive(CubeType)]
pub struct MmaFragment<E: Numeric, N: Size> {
    fragment: Array<Vector<E, N>>,
    #[cube(comptime)]
    layout: MatrixLayout,
}

// #[derive(CubeType)]
// pub struct MmaOperands<L: Numeric, R: Numeric, A: Numeric> {
//     #[cube(comptime)]
//     _phantom: PhantomData<(L, R, A)>,
// }

// impl<L: Numeric, R: Numeric, A: Numeric> Operands for MmaOperands<L, R, A> {
//     type Lhs = MmaFragment<L, NL>;
//     type Rhs = MmaFragment<R, NR>;
//     type Acc = MmaFragment<A, NA>;
// }

#[cube]
impl<L: Numeric, VL: Size, R: Numeric, VR: Size, A: Numeric, VA: Size>
    TileMatmul<L, VL, R, VR, A, VA> for MmaMatmul
{
    type Config = MmaMatmulConfig;

    // TODO  Dummy
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
        out: &mut Tilex<A, VA, Self::Scope, ReadWrite>,
        #[comptime] _config: Self::Config,
    ) {
        tilex_write(tile, out);
    }
}

#[cube]
pub(super) fn mma_definition<L: Numeric, R: Numeric, A: Numeric>(
    #[comptime] config: MmaMatmulConfig,
) -> MmaDefinition<L, R, A> {
    let size = config.shared.tile_size;
    MmaDefinition::new(size.m() as usize, size.n() as usize, size.k() as usize)
}

#[cube]
#[allow(unused_variables)]
pub(super) fn register_vector_sizes<L: Numeric, R: Numeric, A: Numeric>(
    def: MmaDefinition<L, R, A>,
) {
    let vector_size_a = def.vector_size(MatrixIdent::A);
    let vector_size_b = def.vector_size(MatrixIdent::B);
    let vector_size_acc = def.vector_size(MatrixIdent::Accumulator);
    intrinsic!(|scope| {
        scope.register_size::<NL>(vector_size_a);
        scope.register_size::<NR>(vector_size_b);
        scope.register_size::<NA>(vector_size_acc);
    });
}
