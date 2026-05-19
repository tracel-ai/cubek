use cubecl::{
    features::MmaConfig,
    ir::{DeviceProperties, StorageType},
    prelude::*,
};
use cubek_std::{
    CubeDimResource, InvalidConfigError, MatrixLayout, SwizzleModes, TileSize,
    tile::{Plane, TileScope},
};

use crate::definition::{MatmulElems, MatmulSetupError, MatmulVectorSizes, TilingBlueprint};

use super::{common::check_types_available, variant::TileVariant};

#[derive(Copy, Clone, Eq, PartialEq, Hash, Debug)]
pub struct PlaneVecMatInnerProduct {
    pub tile_size: TileSize,
    pub plane_dim: u32,
    pub swizzle_modes: SwizzleModes,
    pub reduce_vector_size: u32,
}

impl PlaneVecMatInnerProduct {
    pub fn new(
        tile_size: TileSize,
        plane_dim: u32,
        swizzle_modes: SwizzleModes,
        reduce_vector_size: u32,
    ) -> Self {
        Self {
            tile_size,
            plane_dim,
            swizzle_modes,
            reduce_vector_size,
        }
    }
}

impl TileVariant for PlaneVecMatInnerProduct {
    fn requires_accelerator() -> bool {
        false
    }

    fn can_cast_stage_element() -> bool {
        true
    }

    fn should_swizzle<R: Runtime>(_client: &ComputeClient<R>) -> bool {
        // Supported but needs tuning, currently off.
        false
    }

    fn cubedim_resource() -> Result<CubeDimResource, InvalidConfigError> {
        Ok(Plane::default_resource())
    }

    fn is_supported<R: Runtime>(_client: &ComputeClient<R>, _config: MmaConfig) -> bool {
        true
    }

    fn supported_sizes<R: Runtime>(
        _client: &ComputeClient<R>,
        _lhs_ty: StorageType,
        _rhs_ty: StorageType,
        _acc_ty: StorageType,
    ) -> Vec<TileSize> {
        Vec::new()
    }

    fn expand(
        _device_props: &DeviceProperties,
        blueprint: &TilingBlueprint,
        _dtypes: &MatmulElems,
        vector_sizes: &MatmulVectorSizes,
    ) -> Self {
        Self::new(
            blueprint.tiling_scheme.tile_size,
            blueprint.plane_dim,
            blueprint.swizzle_modes,
            vector_sizes.lhs as u32,
        )
    }

    fn validate<R: Runtime>(
        client: &ComputeClient<R>,
        blueprint: &TilingBlueprint,
        dtypes: &MatmulElems,
        vector_sizes: &MatmulVectorSizes,
    ) -> Result<(), MatmulSetupError> {
        check_types_available(client, dtypes, true)?;

        if blueprint.lhs_layout != MatrixLayout::RowMajor {
            return Err(MatmulSetupError::InvalidConfig(Box::new(
                "Only Row Major layout is supported for Lhs",
            )));
        }

        if blueprint.rhs_layout != MatrixLayout::ColMajor {
            return Err(MatmulSetupError::InvalidConfig(Box::new(
                "Only Col Major layout is supported for Rhs",
            )));
        }

        let m = blueprint.tiling_scheme.tile_size.m();
        let n = blueprint.tiling_scheme.tile_size.n();
        let k = blueprint.tiling_scheme.tile_size.k();

        let lhs_vector = vector_sizes.lhs as u32;
        let rhs_vector = vector_sizes.rhs as u32;
        let out_vector = vector_sizes.out as u32;

        if m != 1 {
            return Err(MatmulSetupError::InvalidConfig(Box::new(format!(
                "Only m=1 is supported, got m={m:?}",
            ))));
        }

        if lhs_vector != rhs_vector {
            return Err(MatmulSetupError::InvalidConfig(Box::new(format!(
                "Lhs and Rhs must have same vector size, got lhs={lhs_vector:?} and rhs={rhs_vector:?}",
            ))));
        }

        if k != blueprint.plane_dim * lhs_vector {
            return Err(MatmulSetupError::InvalidConfig(Box::new(format!(
                "k must be equal to plane_dim times vector size (of both lhs and rhs), got k={:?}, plane_dim={:?} vector_size={:?}",
                k, blueprint.plane_dim, lhs_vector
            ))));
        }

        if !n.is_multiple_of(out_vector) {
            return Err(MatmulSetupError::InvalidConfig(Box::new(format!(
                "n must be divisible by out vector size, got n={n:?}, out_vector_size={out_vector:?}",
            ))));
        }

        Ok(())
    }
}
