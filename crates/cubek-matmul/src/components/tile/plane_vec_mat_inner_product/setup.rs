use crate::components::tile::SharedTileConfig;
use crate::components::tile::io::TileKind;
use crate::components::tile::plane_vec_mat_inner_product::config::PlaneVecMatInnerProductConfig;
use crate::components::tile::plane_vec_mat_inner_product::matmul::PlaneVecMatInnerProduct;
use crate::components::tile::{TileMatmulFamily, io::Strided};
use crate::components::{
    resource::CubeDimResource,
    tile::plane_vec_mat_inner_product::reader::{MatrixFragmentReader, MatrixStageReader},
};
use crate::definition::{
    InvalidConfigError, MatmulAvailabilityError, MatmulElems, MatmulSetupError, MatrixLayout,
};
use crate::definition::{MatmulLineSizes, TilingBlueprint};
use cubecl::features::{Plane, TypeUsage};
use cubecl::ir::{ElemType, FloatKind};
use cubecl::prelude::*;

impl<Kind: TileKind> TileMatmulFamily for PlaneVecMatInnerProduct<Kind>
where
    MatrixStageReader<Kind>: MatrixFragmentReader<TileKind = Kind>,
{
    type Config = PlaneVecMatInnerProductConfig;
    type Matmul<L: Numeric, R: Numeric, A: Numeric> = PlaneVecMatInnerProduct<Kind>;

    type LhsTile = Strided;
    type RhsTile = Strided;
    type AccTile = Kind;
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
        blueprint: &TilingBlueprint,
        _dtypes: &MatmulElems,
        line_sizes: &MatmulLineSizes,
    ) -> Result<PlaneVecMatInnerProductConfig, MatmulSetupError> {
        Ok(PlaneVecMatInnerProductConfig::new(
            SharedTileConfig::new(
                blueprint.tiling_scheme.tile_size,
                blueprint.plane_dim,
                blueprint.swizzle_modes,
            ),
            line_sizes.lhs as u32,
        ))
    }

    fn should_swizzle<R: Runtime>(_client: &ComputeClient<R>) -> bool {
        // Supported but need to find good settings for this tiling. Currently tuned for `ldmatrix`.
        // Need to profile at some point
        false
    }

    fn validate_blueprint<R: Runtime>(
        client: &ComputeClient<R>,
        blueprint: &TilingBlueprint,
        dtypes: &MatmulElems,
        line_sizes: &MatmulLineSizes,
    ) -> Result<(), MatmulSetupError> {
        check_availability(client, dtypes)?;

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

        let lhs_line = line_sizes.lhs as u32;
        let rhs_line = line_sizes.rhs as u32;
        let out_line = line_sizes.out as u32;

        if m != 1 {
            return Err(MatmulSetupError::InvalidConfig(Box::new(format!(
                "Only m=1 is supported, got m={m:?}",
            ))));
        }

        if lhs_line != rhs_line {
            return Err(MatmulSetupError::InvalidConfig(Box::new(format!(
                "Lhs and Rhs must have same line size, got lhs={lhs_line:?} and rhs={rhs_line:?}",
            ))));
        }

        if k != blueprint.plane_dim * lhs_line {
            return Err(MatmulSetupError::InvalidConfig(Box::new(format!(
                "k must be equal to plane_dim times line size (of both lhs and rhs), got k={:?}, plane_dim={:?} line_size={:?}",
                k, blueprint.plane_dim, lhs_line
            ))));
        }

        if !n.is_multiple_of(out_line) {
            return Err(MatmulSetupError::InvalidConfig(Box::new(format!(
                "n must be divisible by out line size, got n={n:?}, out_line_size={out_line:?}",
            ))));
        }

        Ok(())
    }
}

fn check_availability<R: Runtime>(
    client: &ComputeClient<R>,
    dtypes: &MatmulElems,
) -> Result<(), MatmulSetupError> {
    if !client.properties().features.plane.contains(Plane::Ops) {
        return Err(MatmulSetupError::Unavailable(
            MatmulAvailabilityError::PlaneOpsUnavailable,
        ));
    }

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
