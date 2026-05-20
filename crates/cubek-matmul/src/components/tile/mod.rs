//! Tile-level matmul: per-variant matmul-configs + the [`TileMatmul`] /
//! [`TileMatmulKind`] enums that dispatch over them.
//!
//! Each tile-kind owns its config struct, validator, and `expand` constructor
//! in its own module and implements [`TileVariant`]. This file holds *only*
//! dispatch: the enums themselves, accessor methods that read fields of the
//! carried struct, and per-method 5-arm matches that forward into the
//! variant's [`TileVariant`] impl. The `allocate_*` dispatchers at the bottom
//! follow the same shape, unpacking each matmul-config into the primitive
//! comptime fields the cubek-std allocators take.

mod cmma;
mod common;
mod interleaved;
mod mma;
mod plane_vec;
mod register;
mod variant;

pub use cmma::CmmaMatmul;
pub use interleaved::InterleavedMatmul;
pub use mma::MmaMatmul;
pub use plane_vec::PlaneVecMatInnerProduct;
pub use register::RegisterMatmul;
pub use variant::TileVariant;

use cubecl::{
    features::MmaConfig,
    ir::{DeviceProperties, StorageType},
    prelude::*,
};
use cubek_std::{
    CubeDimResource, InvalidConfigError, MatrixLayout, TileSize,
    tile::{
        Tile, TileScope, cmma_allocate_acc, cmma_allocate_lhs, cmma_allocate_rhs,
        interleaved_allocate_acc, interleaved_allocate_lhs, interleaved_allocate_rhs,
        mma_allocate_acc, mma_allocate_lhs, mma_allocate_rhs, planevec_allocate_acc,
        planevec_allocate_lhs, planevec_allocate_rhs, register_allocate_acc, register_allocate_lhs,
        register_allocate_rhs,
    },
};

use crate::definition::{MatmulElems, MatmulSetupError, MatmulVectorSizes, TilingBlueprint};

/// Tile-level matmul configuration. Each variant carries the per-kind config.
///
/// This is both the runtime selector and the comptime configuration
#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
pub enum TileMatmul {
    Cmma(CmmaMatmul),
    Mma(MmaMatmul),
    Register(RegisterMatmul),
    PlaneVec(PlaneVecMatInnerProduct),
    Interleaved(InterleavedMatmul),
}

impl TileMatmul {
    pub fn elements_in_tile_m(&self) -> u32 {
        match self {
            TileMatmul::Cmma(c) => c.tile_size.m(),
            TileMatmul::Mma(c) => c.tile_size.m(),
            TileMatmul::Register(c) => c.tile_size.m(),
            TileMatmul::PlaneVec(c) => c.tile_size.m(),
            TileMatmul::Interleaved(c) => c.tile_size.m(),
        }
    }

    pub fn elements_in_tile_n(&self) -> u32 {
        match self {
            TileMatmul::Cmma(c) => c.tile_size.n(),
            TileMatmul::Mma(c) => c.tile_size.n(),
            TileMatmul::Register(c) => c.tile_size.n(),
            TileMatmul::PlaneVec(c) => c.tile_size.n(),
            TileMatmul::Interleaved(c) => c.tile_size.n(),
        }
    }

    pub fn elements_in_tile_k(&self) -> u32 {
        match self {
            TileMatmul::Cmma(c) => c.tile_size.k(),
            TileMatmul::Mma(c) => c.tile_size.k(),
            TileMatmul::Register(c) => c.tile_size.k(),
            TileMatmul::PlaneVec(c) => c.tile_size.k(),
            TileMatmul::Interleaved(c) => c.tile_size.k(),
        }
    }
}

/// Selector for the tile-level matmul kind, used before per-kind config exists.
///
/// All methods on this enum are pure 5-arm dispatchers that forward to the
/// matching variant's [`TileVariant`] impl.
#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
pub enum TileMatmulKind {
    Cmma,
    Mma,
    Register,
    PlaneVec,
    Interleaved,
}

impl TileMatmulKind {
    pub fn requires_accelerator(&self) -> bool {
        match self {
            TileMatmulKind::Cmma => CmmaMatmul::requires_accelerator(),
            TileMatmulKind::Mma => MmaMatmul::requires_accelerator(),
            TileMatmulKind::Register => RegisterMatmul::requires_accelerator(),
            TileMatmulKind::PlaneVec => PlaneVecMatInnerProduct::requires_accelerator(),
            TileMatmulKind::Interleaved => InterleavedMatmul::requires_accelerator(),
        }
    }

    pub fn can_cast_stage_element(&self) -> bool {
        match self {
            TileMatmulKind::Cmma => CmmaMatmul::can_cast_stage_element(),
            TileMatmulKind::Mma => MmaMatmul::can_cast_stage_element(),
            TileMatmulKind::Register => RegisterMatmul::can_cast_stage_element(),
            TileMatmulKind::PlaneVec => PlaneVecMatInnerProduct::can_cast_stage_element(),
            TileMatmulKind::Interleaved => InterleavedMatmul::can_cast_stage_element(),
        }
    }

    pub fn should_swizzle<R: Runtime>(&self, client: &ComputeClient<R>) -> bool {
        match self {
            TileMatmulKind::Cmma => CmmaMatmul::should_swizzle(client),
            TileMatmulKind::Mma => MmaMatmul::should_swizzle(client),
            TileMatmulKind::Register => RegisterMatmul::should_swizzle(client),
            TileMatmulKind::PlaneVec => PlaneVecMatInnerProduct::should_swizzle(client),
            TileMatmulKind::Interleaved => InterleavedMatmul::should_swizzle(client),
        }
    }

    pub fn cubedim_resource(&self) -> Result<CubeDimResource, InvalidConfigError> {
        match self {
            TileMatmulKind::Cmma => CmmaMatmul::cubedim_resource(),
            TileMatmulKind::Mma => MmaMatmul::cubedim_resource(),
            TileMatmulKind::Register => RegisterMatmul::cubedim_resource(),
            TileMatmulKind::PlaneVec => PlaneVecMatInnerProduct::cubedim_resource(),
            TileMatmulKind::Interleaved => InterleavedMatmul::cubedim_resource(),
        }
    }

    pub fn is_supported<R: Runtime>(&self, client: &ComputeClient<R>, config: MmaConfig) -> bool {
        match self {
            TileMatmulKind::Cmma => CmmaMatmul::is_supported(client, config),
            TileMatmulKind::Mma => MmaMatmul::is_supported(client, config),
            TileMatmulKind::Register => RegisterMatmul::is_supported(client, config),
            TileMatmulKind::PlaneVec => PlaneVecMatInnerProduct::is_supported(client, config),
            TileMatmulKind::Interleaved => InterleavedMatmul::is_supported(client, config),
        }
    }

    pub fn supported_sizes<R: Runtime>(
        &self,
        client: &ComputeClient<R>,
        lhs_ty: StorageType,
        rhs_ty: StorageType,
        acc_ty: StorageType,
    ) -> Vec<TileSize> {
        match self {
            TileMatmulKind::Cmma => CmmaMatmul::supported_sizes(client, lhs_ty, rhs_ty, acc_ty),
            TileMatmulKind::Mma => MmaMatmul::supported_sizes(client, lhs_ty, rhs_ty, acc_ty),
            TileMatmulKind::Register => {
                RegisterMatmul::supported_sizes(client, lhs_ty, rhs_ty, acc_ty)
            }
            TileMatmulKind::PlaneVec => {
                PlaneVecMatInnerProduct::supported_sizes(client, lhs_ty, rhs_ty, acc_ty)
            }
            TileMatmulKind::Interleaved => {
                InterleavedMatmul::supported_sizes(client, lhs_ty, rhs_ty, acc_ty)
            }
        }
    }

    pub fn expand_tile_matmul(
        &self,
        device_props: &DeviceProperties,
        blueprint: &TilingBlueprint,
        dtypes: &MatmulElems,
        vector_sizes: &MatmulVectorSizes,
    ) -> Result<TileMatmul, MatmulSetupError> {
        Ok(match self {
            TileMatmulKind::Cmma => TileMatmul::Cmma(CmmaMatmul::expand(
                device_props,
                blueprint,
                dtypes,
                vector_sizes,
            )),
            TileMatmulKind::Mma => TileMatmul::Mma(MmaMatmul::expand(
                device_props,
                blueprint,
                dtypes,
                vector_sizes,
            )),
            TileMatmulKind::Register => TileMatmul::Register(RegisterMatmul::expand(
                device_props,
                blueprint,
                dtypes,
                vector_sizes,
            )),
            TileMatmulKind::PlaneVec => TileMatmul::PlaneVec(PlaneVecMatInnerProduct::expand(
                device_props,
                blueprint,
                dtypes,
                vector_sizes,
            )),
            TileMatmulKind::Interleaved => TileMatmul::Interleaved(InterleavedMatmul::expand(
                device_props,
                blueprint,
                dtypes,
                vector_sizes,
            )),
        })
    }

    pub fn validate_blueprint<R: Runtime>(
        &self,
        client: &ComputeClient<R>,
        blueprint: &TilingBlueprint,
        dtypes: &MatmulElems,
        vector_sizes: &MatmulVectorSizes,
    ) -> Result<(), MatmulSetupError> {
        match self {
            TileMatmulKind::Cmma => CmmaMatmul::validate(client, blueprint, dtypes, vector_sizes),
            TileMatmulKind::Mma => MmaMatmul::validate(client, blueprint, dtypes, vector_sizes),
            TileMatmulKind::Register => {
                RegisterMatmul::validate(client, blueprint, dtypes, vector_sizes)
            }
            TileMatmulKind::PlaneVec => {
                PlaneVecMatInnerProduct::validate(client, blueprint, dtypes, vector_sizes)
            }
            TileMatmulKind::Interleaved => {
                InterleavedMatmul::validate(client, blueprint, dtypes, vector_sizes)
            }
        }
    }
}

// =====================================================================
// Per-kind allocator dispatchers
// =====================================================================

#[cube]
/// Allocate the lhs instruction-level tile for the selected `tile_matmul`
/// kind. `LhsRE` / `RhsRE` / `AccRE` are the register element types; `RhsRE`
/// and `AccRE` are only consulted on the MMA path.
pub fn allocate_lhs<LhsRE: Numeric, RhsRE: Numeric, AccRE: Numeric, Sc: TileScope>(
    #[comptime] layout: MatrixLayout,
    #[comptime] tile_matmul: TileMatmul,
) -> Tile<LhsRE, Sc> {
    match tile_matmul {
        TileMatmul::Cmma(c) => cmma_allocate_lhs::<LhsRE, Sc>(layout, c.tile_size),
        TileMatmul::Mma(c) => {
            mma_allocate_lhs::<LhsRE, RhsRE, AccRE, Sc>(layout, c.tile_size, c.mma_io_config)
        }
        TileMatmul::Register(c) => {
            register_allocate_lhs::<LhsRE, Sc>(layout, c.tile_size, c.product_type)
        }
        TileMatmul::PlaneVec(c) => {
            planevec_allocate_lhs::<LhsRE, Sc>(layout, c.tile_size, c.reduce_vector_size)
        }
        TileMatmul::Interleaved(c) => {
            interleaved_allocate_lhs::<LhsRE, Sc>(layout, c.tile_size, c.plane_dim)
        }
    }
}

#[cube]
/// Allocate the rhs instruction-level tile for the selected `tile_matmul`
/// kind.
pub fn allocate_rhs<LhsRE: Numeric, RhsRE: Numeric, AccRE: Numeric, Sc: TileScope>(
    #[comptime] layout: MatrixLayout,
    #[comptime] tile_matmul: TileMatmul,
) -> Tile<RhsRE, Sc> {
    match tile_matmul {
        TileMatmul::Cmma(c) => cmma_allocate_rhs::<RhsRE, Sc>(layout, c.tile_size),
        TileMatmul::Mma(c) => {
            mma_allocate_rhs::<RhsRE, LhsRE, AccRE, Sc>(layout, c.tile_size, c.mma_io_config)
        }
        TileMatmul::Register(c) => {
            register_allocate_rhs::<RhsRE, Sc>(layout, c.tile_size, c.product_type)
        }
        TileMatmul::PlaneVec(c) => {
            planevec_allocate_rhs::<RhsRE, Sc>(layout, c.tile_size, c.reduce_vector_size)
        }
        TileMatmul::Interleaved(c) => {
            interleaved_allocate_rhs::<RhsRE, Sc>(layout, c.tile_size, c.plane_dim)
        }
    }
}

#[cube]
/// Allocate the accumulator instruction-level tile for the selected
/// `tile_matmul` kind.
pub fn allocate_acc<LhsRE: Numeric, RhsRE: Numeric, AccRE: Numeric, Sc: TileScope>(
    #[comptime] layout: MatrixLayout,
    #[comptime] tile_matmul: TileMatmul,
) -> Tile<AccRE, Sc> {
    match tile_matmul {
        TileMatmul::Cmma(c) => cmma_allocate_acc::<AccRE, Sc>(layout, c.tile_size),
        TileMatmul::Mma(c) => {
            mma_allocate_acc::<AccRE, LhsRE, RhsRE, Sc>(layout, c.tile_size, c.mma_io_config)
        }
        TileMatmul::Register(c) => {
            register_allocate_acc::<AccRE, Sc>(layout, c.tile_size, c.product_type)
        }
        TileMatmul::PlaneVec(c) => {
            planevec_allocate_acc::<AccRE, Sc>(layout, c.tile_size, c.reduce_vector_size)
        }
        TileMatmul::Interleaved(c) => {
            interleaved_allocate_acc::<AccRE, Sc>(layout, c.tile_size, c.plane_dim)
        }
    }
}
