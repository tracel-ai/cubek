use crate::{
    components::{
        resource::CubeDimResource,
        tile_matmul::{
            Plane, SharedTileConfig, TileMatmulFamily,
            interleaved::{InterleavedMatmul, config::InterleavedMatmulConfig},
        },
    },
    definition::{
        MatmulAvailabilityError, MatmulElems, MatmulSetupError, MatmulVectorSizes, TilingBlueprint,
    },
};
use cubecl::{
    ir::{ElemType, FloatKind},
    prelude::*,
    {features::TypeUsage, ir::DeviceProperties},
};
use cubek_std::{InvalidConfigError, MatrixLayout};

impl TileMatmulFamily for InterleavedMatmul {
    type Config = InterleavedMatmulConfig;
    type Scope = Plane;
    type Matmul<L: Numeric, VL: Size, R: Numeric, VR: Size, A: Numeric, VA: Size> =
        InterleavedMatmul;

    fn requires_accelerator() -> bool {
        false
    }

    fn can_cast_stage_element() -> bool {
        true
    }

    fn cubedim_resource() -> Result<CubeDimResource, InvalidConfigError> {
        Ok(CubeDimResource::Planes(1))
    }

    fn expand_config(
        _device_props: &DeviceProperties,
        blueprint: &TilingBlueprint,
        _dtypes: &MatmulElems,
        _vector_sizes: &MatmulVectorSizes,
    ) -> Result<Self::Config, MatmulSetupError> {
        Ok(InterleavedMatmulConfig::from_shared_tile_config(
            SharedTileConfig::new(
                blueprint.tiling_scheme.tile_size,
                blueprint.plane_dim,
                blueprint.swizzle_modes,
            ),
        ))
    }

    fn should_swizzle<R: Runtime>(client: &ComputeClient<R>) -> bool {
        // Selection isn't getting rid of all conflicts with the current load strategy, but does
        // reduce conflicts significantly (i.e. average 18 vs average 5). Should try to find more
        // optimal settings in the future.
        client.properties().features.alignment
    }

    fn validate_blueprint<R: Runtime>(
        client: &ComputeClient<R>,
        blueprint: &TilingBlueprint,
        dtypes: &MatmulElems,
        vector_sizes: &MatmulVectorSizes,
    ) -> Result<(), MatmulSetupError> {
        check_availability(client, dtypes)?;

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
                "Tile shape in vectorized axis n ({n:?}) should be divisible by vector size out ({out:?})"
            ))));
        }

        Ok(())
    }
}

fn check_availability<R: Runtime>(
    client: &ComputeClient<R>,
    dtypes: &MatmulElems,
) -> Result<(), MatmulSetupError> {
    let lhs = dtypes.lhs_register;
    let rhs = dtypes.rhs_register;
    let acc = dtypes.acc_register;

    let lhs = match lhs {
        StorageType::Scalar(ElemType::Float(FloatKind::Flex32)) => {
            ElemType::Float(FloatKind::F32).into()
        }
        _ => lhs,
    };
    let rhs = match rhs {
        StorageType::Scalar(ElemType::Float(FloatKind::Flex32)) => {
            ElemType::Float(FloatKind::F32).into()
        }
        _ => rhs,
    };

    let output = match acc {
        StorageType::Scalar(ElemType::Float(FloatKind::Flex32)) => {
            ElemType::Float(FloatKind::F32).into()
        }
        _ => acc,
    };

    if !(client
        .properties()
        .features
        .type_usage(lhs)
        .contains(TypeUsage::Arithmetic)
        && client
            .properties()
            .features
            .type_usage(rhs)
            .contains(TypeUsage::Arithmetic)
        && client
            .properties()
            .features
            .type_usage(output)
            .contains(TypeUsage::Arithmetic))
    {
        return Err(MatmulSetupError::Unavailable(
            MatmulAvailabilityError::TypesUnavailable { lhs, rhs, output },
        ));
    }

    Ok(())
}
