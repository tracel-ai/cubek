use cubecl::{
    features::MmaConfig,
    ir::{DeviceProperties, StorageType},
    prelude::*,
};
use cubek_std::{
    CubeDimResource, InvalidConfigError, MatrixLayout, SwizzleModes, TileSize,
    tile::{ProductType, TileScope, Unit},
};

use crate::definition::{MatmulElems, MatmulSetupError, MatmulVectorSizes, TilingBlueprint};

use super::{common::check_types_available, variant::TileVariant};

#[derive(Copy, Clone, Eq, PartialEq, Hash, Debug)]
pub struct RegisterMatmul {
    pub tile_size: TileSize,
    pub plane_dim: u32,
    pub swizzle_modes: SwizzleModes,
    pub product_type: ProductType,
}

impl RegisterMatmul {
    pub fn new(
        lhs_layout: MatrixLayout,
        rhs_layout: MatrixLayout,
        tile_size: TileSize,
        plane_dim: u32,
        swizzle_modes: SwizzleModes,
    ) -> Self {
        Self {
            tile_size,
            plane_dim,
            swizzle_modes,
            product_type: ProductType::from_layouts(lhs_layout, rhs_layout, tile_size),
        }
    }
}

impl TileVariant for RegisterMatmul {
    fn requires_accelerator() -> bool {
        false
    }

    fn can_cast_stage_element() -> bool {
        true
    }

    fn should_swizzle<R: Runtime>(client: &ComputeClient<R>) -> bool {
        // Doesn't get rid of all conflicts with the current load strategy, but
        // does reduce them significantly (avg 18 vs avg 5). Tune in the future.
        client.properties().features.alignment
    }

    fn cubedim_resource() -> Result<CubeDimResource, InvalidConfigError> {
        Ok(Unit::default_resource())
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
        _vector_sizes: &MatmulVectorSizes,
    ) -> Self {
        Self::new(
            blueprint.lhs_layout,
            blueprint.rhs_layout,
            blueprint.tiling_scheme.tile_size,
            blueprint.plane_dim,
            blueprint.swizzle_modes,
        )
    }

    fn validate<R: Runtime>(
        client: &ComputeClient<R>,
        blueprint: &TilingBlueprint,
        dtypes: &MatmulElems,
        vector_sizes: &MatmulVectorSizes,
    ) -> Result<(), MatmulSetupError> {
        check_types_available(client, dtypes, false)?;

        let m = blueprint.tiling_scheme.tile_size.m();
        let n = blueprint.tiling_scheme.tile_size.n();
        let k = blueprint.tiling_scheme.tile_size.k();

        let lhs = vector_sizes.lhs as u32;
        let rhs = vector_sizes.rhs as u32;
        let out = vector_sizes.out as u32;

        match blueprint.lhs_layout {
            MatrixLayout::RowMajor => {
                if !k.is_multiple_of(lhs) {
                    return Err(MatmulSetupError::InvalidConfig(Box::new(format!(
                        "Tile shape in vectorized axis k({k:?}) should be divisible by vector size lhs({lhs:?})"
                    ))));
                }
            }
            MatrixLayout::ColMajor => {
                if !m.is_multiple_of(lhs) {
                    return Err(MatmulSetupError::InvalidConfig(Box::new(format!(
                        "Tile shape in vectorized axis m({m:?}) should be divisible by vector size lhs({lhs:?})"
                    ))));
                }
            }
        }
        match blueprint.rhs_layout {
            MatrixLayout::RowMajor => {
                if !n.is_multiple_of(rhs) {
                    return Err(MatmulSetupError::InvalidConfig(Box::new(format!(
                        "Tile shape in vectorized axis n({n:?}) should be divisible by vector size rhs({rhs:?})"
                    ))));
                }
            }
            MatrixLayout::ColMajor => {
                if !k.is_multiple_of(rhs) {
                    return Err(MatmulSetupError::InvalidConfig(Box::new(format!(
                        "Tile shape in vectorized axis k({k:?}) should be divisible by vector size rhs({rhs:?})"
                    ))));
                }
            }
        }

        if !n.is_multiple_of(out) {
            return Err(MatmulSetupError::InvalidConfig(Box::new(format!(
                "Tile shape in vectorized axis n({n:?}) should be divisible by vector size out({out:?})"
            ))));
        }

        Ok(())
    }
}
