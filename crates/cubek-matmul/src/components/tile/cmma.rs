use cubecl::{
    features::MmaConfig,
    ir::{DeviceProperties, StorageType},
    prelude::*,
};
use cubek_std::{
    CubeDimResource, InvalidConfigError, SwizzleModes, TileSize,
    tile::{Plane, TileScope},
};

use crate::definition::{
    MatmulAvailabilityError, MatmulElems, MatmulSetupError, MatmulVectorSizes, TilingBlueprint,
};

use super::variant::TileVariant;

#[derive(Copy, Clone, Eq, PartialEq, Hash, Debug)]
pub struct CmmaMatmul {
    pub tile_size: TileSize,
    pub plane_dim: u32,
    pub swizzle_modes: SwizzleModes,
}

impl CmmaMatmul {
    pub fn new(tile_size: TileSize, plane_dim: u32, swizzle_modes: SwizzleModes) -> Self {
        Self {
            tile_size,
            plane_dim,
            swizzle_modes,
        }
    }
}

impl TileVariant for CmmaMatmul {
    fn requires_accelerator() -> bool {
        true
    }

    fn can_cast_stage_element() -> bool {
        false
    }

    fn should_swizzle<R: Runtime>(_client: &ComputeClient<R>) -> bool {
        false
    }

    fn cubedim_resource() -> Result<CubeDimResource, InvalidConfigError> {
        Ok(Plane::default_resource())
    }

    fn is_supported<R: Runtime>(client: &ComputeClient<R>, config: MmaConfig) -> bool {
        client.properties().features.matmul.cmma.contains(&config)
    }

    fn supported_sizes<R: Runtime>(
        client: &ComputeClient<R>,
        lhs_ty: StorageType,
        rhs_ty: StorageType,
        acc_ty: StorageType,
    ) -> Vec<TileSize> {
        client
            .properties()
            .features
            .matmul
            .cmma
            .iter()
            .filter(|it| it.a_type == lhs_ty && it.b_type == rhs_ty && it.cd_type == acc_ty)
            .map(|it| (it.m, it.n, it.k).into())
            .collect()
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
        _vector_sizes: &MatmulVectorSizes,
    ) -> Result<(), MatmulSetupError> {
        let lhs = dtypes.lhs_register;
        let rhs = dtypes.rhs_register;
        let acc = dtypes.acc_register;

        let size = blueprint.tiling_scheme.tile_size;
        if !client
            .properties()
            .features
            .matmul
            .cmma
            .contains(&MmaConfig {
                a_type: lhs,
                b_type: rhs,
                cd_type: acc,
                m: size.m(),
                k: size.k(),
                n: size.n(),
            })
        {
            return Err(MatmulSetupError::Unavailable(
                MatmulAvailabilityError::CmmaInstructionUnavailable {
                    lhs,
                    rhs,
                    output: acc,
                    size: Some(TileSize::new(size.m(), size.n(), size.k())),
                },
            ));
        }

        if blueprint.swizzle_modes.has_swizzle() {
            return Err(MatmulSetupError::InvalidConfig(Box::new(
                "This tile matmul doesn't support swizzling",
            )));
        }

        Ok(())
    }
}
