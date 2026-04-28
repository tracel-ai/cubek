use cubecl::prelude::*;
use cubek_std::MatrixLayout;

use crate::components::tile_matmul::dispatch::config::DispatchConfig;
use crate::components::tile_matmul::{
    Plane, Tile, cmma_allocate_acc, cmma_allocate_lhs, cmma_allocate_rhs, interleaved_allocate_acc,
    interleaved_allocate_lhs, interleaved_allocate_rhs, mma_allocate_acc, mma_allocate_lhs,
    mma_allocate_rhs, planevec_allocate_acc, planevec_allocate_lhs, planevec_allocate_rhs,
    register_allocate_acc, register_allocate_lhs, register_allocate_rhs,
};

#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
pub enum TileMatmul {
    Cmma,
    Mma,
    Register,
    PlaneVec,
    Interleaved,
}

#[cube]
pub(crate) fn allocate_lhs_tile<L: Numeric, VL: Size, R: Numeric, A: Numeric>(
    #[comptime] layout: MatrixLayout,
    #[comptime] config: DispatchConfig,
) -> Tile<L, VL, Plane, ReadWrite> {
    match config {
        DispatchConfig::Cmma(config) => cmma_allocate_lhs::<L, VL, Plane>(layout, config.tile_size),
        DispatchConfig::Mma(config) => {
            mma_allocate_lhs::<L, VL, R, A, Plane>(layout, config.shared, config.mma_io_config)
        }
        DispatchConfig::Register(config) => {
            register_allocate_lhs::<L, VL, Plane>(layout, config.shared, config.product_type)
        }
        DispatchConfig::PlaneVec(config) => {
            planevec_allocate_lhs::<L, VL, Plane>(layout, config.shared, config.reduce_vector_size)
        }
        DispatchConfig::Interleaved(config) => {
            interleaved_allocate_lhs::<L, VL, Plane>(layout, config.shared)
        }
    }
}

#[cube]
pub(crate) fn allocate_rhs_tile<R: Numeric, VR: Size, L: Numeric, A: Numeric>(
    #[comptime] layout: MatrixLayout,
    #[comptime] config: DispatchConfig,
) -> Tile<R, VR, Plane, ReadWrite> {
    match config {
        DispatchConfig::Cmma(config) => cmma_allocate_rhs::<R, VR, Plane>(layout, config.tile_size),
        DispatchConfig::Mma(config) => {
            mma_allocate_rhs::<R, VR, L, A, Plane>(layout, config.shared, config.mma_io_config)
        }
        DispatchConfig::Register(config) => {
            register_allocate_rhs::<R, VR, Plane>(layout, config.shared, config.product_type)
        }
        DispatchConfig::PlaneVec(config) => {
            planevec_allocate_rhs::<R, VR, Plane>(layout, config.shared, config.reduce_vector_size)
        }
        DispatchConfig::Interleaved(config) => {
            interleaved_allocate_rhs::<R, VR, Plane>(layout, config.shared)
        }
    }
}

#[cube]
pub(crate) fn allocate_acc_tile<A: Numeric, VA: Size, L: Numeric, R: Numeric>(
    #[comptime] layout: MatrixLayout,
    #[comptime] config: DispatchConfig,
) -> Tile<A, VA, Plane, ReadWrite> {
    match config {
        DispatchConfig::Cmma(config) => cmma_allocate_acc::<A, VA, Plane>(layout, config.tile_size),
        DispatchConfig::Mma(config) => {
            mma_allocate_acc::<A, VA, L, R, Plane>(layout, config.shared, config.mma_io_config)
        }
        DispatchConfig::Register(config) => {
            register_allocate_acc::<A, VA, Plane>(layout, config.shared, config.product_type)
        }
        DispatchConfig::PlaneVec(config) => {
            planevec_allocate_acc::<A, VA, Plane>(layout, config.shared, config.reduce_vector_size)
        }
        DispatchConfig::Interleaved(config) => {
            interleaved_allocate_acc::<A, VA, Plane>(layout, config.shared)
        }
    }
}
