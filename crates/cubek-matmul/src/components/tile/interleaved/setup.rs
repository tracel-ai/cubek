use crate::components::resource::CubeDimResource;
use crate::components::tile::SharedTileConfig;
use crate::components::tile::interleaved::InterleavedMatmul;
use crate::components::tile::interleaved::config::InterleavedMatmulConfig;
use crate::components::tile::{
    TileMatmulFamily,
    io::{Filled, Strided},
};
use crate::definition::TilingBlueprint;
use crate::definition::{InvalidConfigError, MatmulAvailabilityError, MatmulElems};
use crate::definition::{MatmulLineSizes, MatmulSetupError};
use cubecl::ir::{ElemType, FloatKind};
use cubecl::prelude::*;
use cubecl::{features::TypeUsage, ir::DeviceProperties};

impl TileMatmulFamily for InterleavedMatmul {
    type Config = InterleavedMatmulConfig;
    type Matmul<L: Numeric, R: Numeric, A: Numeric> = InterleavedMatmul;

    type LhsTile = Strided;
    type RhsTile = Strided;
    type AccTile = Filled;
    type OutTile = Strided;

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
        _line_sizes: &MatmulLineSizes,
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
        line_sizes: &MatmulLineSizes,
    ) -> Result<(), MatmulSetupError> {
        check_availability(client, dtypes)?;

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
