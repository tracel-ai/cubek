use crate::components::tile::SharedTileConfig;
use crate::components::tile::io::TileKind;
use crate::components::tile::register::config::RegisterMatmulConfig;
use crate::components::tile::register::matmul::RegisterMatmul;
use crate::components::tile::{TileMatmulFamily, io::Strided};
use crate::components::{
    resource::CubeDimResource,
    tile::register::reader::{RegisterFragmentReader, RegisterStageReader},
};
use crate::definition::TilingBlueprint;
use crate::definition::{InvalidConfigError, MatmulAvailabilityError, MatmulElems, MatrixLayout};
use crate::definition::{MatmulLineSizes, MatmulSetupError};
use cubecl::features::TypeUsage;
use cubecl::ir::{ElemType, FloatKind};
use cubecl::prelude::*;

impl<AccTile: TileKind> TileMatmulFamily for RegisterMatmul<AccTile>
where
    RegisterStageReader<AccTile>: RegisterFragmentReader<TileKind = AccTile>,
{
    type Config = RegisterMatmulConfig;
    type Matmul<L: Numeric, R: Numeric, A: Numeric> = RegisterMatmul<AccTile>;

    type LhsTile = Strided;
    type RhsTile = Strided;
    type AccTile = AccTile;
    type OutTile = Strided;

    fn requires_accelerator() -> bool {
        false
    }

    fn can_cast_stage_element() -> bool {
        true
    }

    fn cubedim_resource() -> Result<CubeDimResource, InvalidConfigError> {
        Ok(CubeDimResource::Units(1))
    }

    fn expand_config(
        blueprint: &TilingBlueprint,
        _dtypes: &MatmulElems,
        _line_sizes: &MatmulLineSizes,
    ) -> Result<Self::Config, MatmulSetupError> {
        Ok(RegisterMatmulConfig::from_shared_tile_config(
            blueprint.lhs_layout,
            blueprint.rhs_layout,
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

        let m = blueprint.tiling_scheme.tile_size.m();
        let n = blueprint.tiling_scheme.tile_size.n();
        let k = blueprint.tiling_scheme.tile_size.k();

        let lhs = line_sizes.lhs as u32;
        let rhs = line_sizes.rhs as u32;
        let out = line_sizes.out as u32;

        match blueprint.lhs_layout {
            MatrixLayout::RowMajor => {
                if !k.is_multiple_of(lhs) {
                    return Err(MatmulSetupError::InvalidConfig(Box::new(format!(
                        "Tile shape in lined axis k({k:?}) should be divisible by line size lhs({lhs:?})"
                    ))));
                }
            }
            MatrixLayout::ColMajor => {
                if !m.is_multiple_of(lhs) {
                    return Err(MatmulSetupError::InvalidConfig(Box::new(format!(
                        "Tile shape in lined axis m({m:?}) should be divisible by line size lhs({lhs:?})"
                    ))));
                }
            }
        }
        match blueprint.rhs_layout {
            MatrixLayout::RowMajor => {
                if !n.is_multiple_of(rhs) {
                    return Err(MatmulSetupError::InvalidConfig(Box::new(format!(
                        "Tile shape in lined axis n({n:?}) should be divisible by line size rhs({rhs:?})"
                    ))));
                }
            }
            MatrixLayout::ColMajor => {
                if !k.is_multiple_of(rhs) {
                    return Err(MatmulSetupError::InvalidConfig(Box::new(format!(
                        "Tile shape in lined axis k({k:?}) should be divisible by line size rhs({rhs:?})"
                    ))));
                }
            }
        }

        if !n.is_multiple_of(out) {
            return Err(MatmulSetupError::InvalidConfig(Box::new(format!(
                "Tile shape in lined axis n({n:?}) should be divisible by line size out({out:?})"
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
