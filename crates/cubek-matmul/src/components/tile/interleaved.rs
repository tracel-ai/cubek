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
pub struct InterleavedMatmul {
    pub tile_size: TileSize,
    pub plane_dim: u32,
    pub swizzle_modes: SwizzleModes,
}

impl InterleavedMatmul {
    pub fn new(tile_size: TileSize, plane_dim: u32, swizzle_modes: SwizzleModes) -> Self {
        Self {
            tile_size,
            plane_dim,
            swizzle_modes,
        }
    }

    pub fn elements_per_unit_m(&self) -> usize {
        self.tile_size.m() as usize
    }

    pub fn elements_per_unit_n(&self) -> usize {
        self.tile_size.n() as usize
    }

    pub fn local_tile_size(&self) -> TileSize {
        TileSize {
            m: self.tile_size.m(),
            n: self.tile_size.n(),
            k: self.tile_size.k(),
        }
    }

    pub fn elements_per_unit_k(&self) -> usize {
        let k = self.tile_size.k() as usize;
        let plane_dim = self.plane_dim as usize;
        assert!(
            k.is_multiple_of(plane_dim),
            "k must be divisible by plane_dim. Got k={:?}, plane_dim={:?}",
            k,
            plane_dim
        );

        k / plane_dim
    }
}

impl TileVariant for InterleavedMatmul {
    fn requires_accelerator() -> bool {
        false
    }

    fn can_cast_stage_element() -> bool {
        true
    }

    fn should_swizzle<R: Runtime>(client: &ComputeClient<R>) -> bool {
        // Same alignment reasoning as the register path.
        client.properties().features.alignment
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
        _vector_sizes: &MatmulVectorSizes,
    ) -> Self {
        Self::new(
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

        let plane_dim = blueprint.plane_dim;
        if !k.is_multiple_of(plane_dim) {
            return Err(MatmulSetupError::InvalidConfig(Box::new(format!(
                "k must be divisible by plane_dim. Got k={:?}, plane_dim={:?}",
                k, plane_dim,
            ))));
        }

        let k_local = k / plane_dim;

        let lhs = vector_sizes.lhs as u32;
        let rhs = vector_sizes.rhs as u32;
        let out = vector_sizes.out as u32;

        match blueprint.lhs_layout {
            MatrixLayout::RowMajor => {
                if !k_local.is_multiple_of(lhs) {
                    return Err(MatmulSetupError::InvalidConfig(Box::new(format!(
                        "Local shape in vectorized axis k ({k_local:?}) should be divisible by vector size lhs ({lhs:?})"
                    ))));
                }
            }
            MatrixLayout::ColMajor => {
                if !m.is_multiple_of(lhs) {
                    return Err(MatmulSetupError::InvalidConfig(Box::new(format!(
                        "Tile shape in vectorized axis m ({m:?}) should be divisible by vector size lhs ({lhs:?})"
                    ))));
                }
            }
        }
        match blueprint.rhs_layout {
            MatrixLayout::RowMajor => {
                if !n.is_multiple_of(rhs) {
                    return Err(MatmulSetupError::InvalidConfig(Box::new(format!(
                        "Tile shape in vectorized axis n ({n:?}) should be divisible by vector size rhs ({rhs:?})"
                    ))));
                }
            }
            MatrixLayout::ColMajor => {
                if !k_local.is_multiple_of(rhs) {
                    return Err(MatmulSetupError::InvalidConfig(Box::new(format!(
                        "Local shape in vectorized axis k ({k_local:?}) should be divisible by vector size rhs ({rhs:?})"
                    ))));
                }
            }
        }

        if !n.is_multiple_of(out) {
            return Err(MatmulSetupError::InvalidConfig(Box::new(format!(
                "Tile shape in vectorized axis out ({n:?}) should be divisible by vector size out ({out:?})"
            ))));
        }

        Ok(())
    }
}
