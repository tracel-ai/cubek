use cubecl::prelude::*;
use cubek_std::MatrixLayout;

use crate::components::tile_matmul::{
    Plane, Tile, TileMatmul, plane_vec_mat_inner_product::config::PlaneVecMatInnerProductConfig,
    planevec_allocate_acc, planevec_allocate_lhs, planevec_allocate_rhs,
};
use crate::definition::StageIdent;

/// Performs a small matmul using one vector per unit.
pub struct PlaneVecMatInnerProduct {}

#[cube]
impl<L: Numeric, VL: Size, R: Numeric, VR: Size, A: Numeric, VA: Size>
    TileMatmul<L, VL, R, VR, A, VA> for PlaneVecMatInnerProduct
{
    type Config = PlaneVecMatInnerProductConfig;
    type Scope = Plane;

    fn execute(
        lhs: &Tile<L, VL, Self::Scope, ReadWrite>,
        rhs: &Tile<R, VR, Self::Scope, ReadWrite>,
        acc: &mut Tile<A, VA, Self::Scope, ReadWrite>,
        #[comptime] _config: Self::Config,
    ) {
        acc.mma(lhs, rhs);
    }

    fn allocate_lhs(
        #[comptime] layout: MatrixLayout,
        #[comptime] config: Self::Config,
    ) -> Tile<L, VL, Self::Scope, ReadWrite> {
        planevec_allocate_lhs::<L, VL, Self::Scope>(
            layout,
            config.shared,
            config.reduce_vector_size,
        )
    }

    fn allocate_rhs(
        #[comptime] layout: MatrixLayout,
        #[comptime] config: Self::Config,
    ) -> Tile<R, VR, Self::Scope, ReadWrite> {
        planevec_allocate_rhs::<R, VR, Self::Scope>(
            layout,
            config.shared,
            config.reduce_vector_size,
        )
    }

    fn allocate_acc(
        #[comptime] layout: MatrixLayout,
        #[comptime] config: Self::Config,
    ) -> Tile<A, VA, Self::Scope, ReadWrite> {
        planevec_allocate_acc::<A, VA, Self::Scope>(
            layout,
            config.shared,
            config.reduce_vector_size,
        )
    }

    fn load_lhs<E: Numeric, ES: Size>(
        tile: &Tile<E, ES, Self::Scope, ReadOnly>,
        lhs: &mut Tile<L, VL, Self::Scope, ReadWrite>,
        #[comptime] _config: Self::Config,
    ) {
        lhs.copy_from::<E, ES, L, R, A, ReadOnly>(tile, StageIdent::Lhs);
    }

    fn load_rhs<E: Numeric, ES: Size>(
        tile: &Tile<E, ES, Self::Scope, ReadOnly>,
        rhs: &mut Tile<R, VR, Self::Scope, ReadWrite>,
        #[comptime] _config: Self::Config,
    ) {
        rhs.copy_from::<E, ES, L, R, A, ReadOnly>(tile, StageIdent::Rhs);
    }

    fn load_acc<E: Numeric, ES: Size>(
        tile: &Tile<E, ES, Self::Scope, ReadOnly>,
        acc: &mut Tile<A, VA, Self::Scope, ReadWrite>,
        #[comptime] _config: Self::Config,
    ) {
        acc.copy_from::<E, ES, L, R, A, ReadOnly>(tile, StageIdent::Acc);
    }

    fn write_results<E: Numeric, ES: Size>(
        tile: &mut Tile<E, ES, Self::Scope, ReadWrite>,
        out: &Tile<A, VA, Self::Scope, ReadWrite>,
        #[comptime] _config: Self::Config,
    ) {
        tile.copy_from::<A, VA, L, R, A, ReadWrite>(out, StageIdent::Out);
    }
}
