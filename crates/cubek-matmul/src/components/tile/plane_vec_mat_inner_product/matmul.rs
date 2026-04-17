use cubecl::prelude::*;
use cubek_std::MatrixLayout;

use crate::components::tile::{
    TileMatmul, Tilex, plane_vec_mat_inner_product::config::PlaneVecMatInnerProductConfig,
    planevec_allocate_acc, planevec_allocate_lhs, planevec_allocate_rhs, tilex_execute, tilex_load,
    tilex_write,
};
use crate::definition::StageIdent;

/// Performs a small matmul using one vector per unit.
pub struct PlaneVecMatInnerProduct {}

#[cube]
impl<L: Numeric, VL: Size, R: Numeric, VR: Size, A: Numeric, VA: Size>
    TileMatmul<L, VL, R, VR, A, VA> for PlaneVecMatInnerProduct
{
    type Config = PlaneVecMatInnerProductConfig;

    fn execute(
        lhs: &Tilex<L, VL, ReadWrite>,
        rhs: &Tilex<R, VR, ReadWrite>,
        acc: &mut Tilex<A, VA, ReadWrite>,
        #[comptime] _config: Self::Config,
    ) {
        tilex_execute::<L, VL, R, VR, A, VA>(lhs, rhs, acc);
    }

    fn allocate_lhs(
        #[comptime] layout: MatrixLayout,
        #[comptime] config: Self::Config,
    ) -> Tilex<L, VL, ReadWrite> {
        planevec_allocate_lhs::<L, VL>(layout, config.shared, config.reduce_vector_size)
    }

    fn allocate_rhs(
        #[comptime] layout: MatrixLayout,
        #[comptime] config: Self::Config,
    ) -> Tilex<R, VR, ReadWrite> {
        planevec_allocate_rhs::<R, VR>(layout, config.shared, config.reduce_vector_size)
    }

    fn allocate_acc(
        #[comptime] layout: MatrixLayout,
        #[comptime] config: Self::Config,
    ) -> Tilex<A, VA, ReadWrite> {
        planevec_allocate_acc::<A, VA>(layout, config.shared, config.reduce_vector_size)
    }

    fn load_lhs<E: Numeric, ES: Size>(
        tile: &Tilex<E, ES, ReadOnly>,
        lhs: &mut Tilex<L, VL, ReadWrite>,
        #[comptime] _config: Self::Config,
    ) {
        tilex_load::<E, ES, L, VL, L, R, A>(tile, lhs, StageIdent::Lhs);
    }

    fn load_rhs<E: Numeric, ES: Size>(
        tile: &Tilex<E, ES, ReadOnly>,
        rhs: &mut Tilex<R, VR, ReadWrite>,
        #[comptime] _config: Self::Config,
    ) {
        tilex_load::<E, ES, R, VR, L, R, A>(tile, rhs, StageIdent::Rhs);
    }

    fn load_acc<E: Numeric, ES: Size>(
        tile: &Tilex<E, ES, ReadOnly>,
        acc: &mut Tilex<A, VA, ReadWrite>,
        #[comptime] _config: Self::Config,
    ) {
        tilex_load::<E, ES, A, VA, L, R, A>(tile, acc, StageIdent::Acc);
    }

    fn write_results<E: Numeric, ES: Size>(
        tile: &mut Tilex<E, ES, ReadWrite>,
        out: &mut Tilex<A, VA, ReadWrite>,
        #[comptime] _config: Self::Config,
    ) {
        tilex_write::<E, ES, A, VA, L, R>(tile, out);
    }
}
