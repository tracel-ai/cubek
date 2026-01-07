use crate::components::tile::SharedTileConfig;
use crate::components::tile::mma::config::StoreMethod;
use crate::components::tile::mma::config::{LoadMethod, MmaMatmulConfig};
use crate::components::tile::{
    TileMatmulFamily,
    mma::{
        MmaMatmul,
        reader::{MmaFragmentReader, MmaStageReader},
    },
};
use crate::components::{
    resource::CubeDimResource,
    tile::io::{Strided, TileKind},
};
use crate::definition::{
    InvalidConfigError, MatmulAvailabilityError, MatmulElems, MatmulSetupError, TileSize,
};
use crate::definition::{MatmulLineSizes, TilingBlueprint};
use cubecl::features::MmaConfig;
use cubecl::{ir::StorageType, prelude::*};

impl<LhsTile: TileKind, RhsTile: TileKind, AccTile: TileKind> TileMatmulFamily
    for MmaMatmul<LhsTile, RhsTile, AccTile>
where
    MmaStageReader<LhsTile>: MmaFragmentReader<TileKind = LhsTile>,
    MmaStageReader<RhsTile>: MmaFragmentReader<TileKind = RhsTile>,
    MmaStageReader<AccTile>: MmaFragmentReader<TileKind = AccTile>,
{
    type Config = MmaMatmulConfig;

    type Matmul<L: Numeric, R: Numeric, A: Numeric> = MmaMatmul<LhsTile, RhsTile, AccTile>;
    type LhsTile = LhsTile;
    type RhsTile = RhsTile;
    type AccTile = AccTile;
    type OutTile = Strided;

    fn requires_accelerator() -> bool {
        true
    }

    fn can_cast_stage_element() -> bool {
        true
    }

    fn cubedim_resource() -> Result<CubeDimResource, InvalidConfigError> {
        Ok(CubeDimResource::Planes(1))
    }

    fn expand_config(
        blueprint: &TilingBlueprint,
        dtypes: &MatmulElems,
        _line_sizes: &MatmulLineSizes,
    ) -> Result<Self::Config, MatmulSetupError> {
        Ok(MmaMatmulConfig::from_shared_tile_config(
            SharedTileConfig {
                tile_size: blueprint.tiling_scheme.tile_size,
                plane_dim: blueprint.plane_dim,
                swizzle_modes: blueprint.swizzle_modes,
            },
            load_method(dtypes.lhs_stage),
            load_method(dtypes.rhs_stage),
            load_method(dtypes.acc_stage),
            store_method(dtypes.acc_stage),
        ))
    }

    fn should_swizzle<R: Runtime>(client: &ComputeClient<R>) -> bool {
        // No alignment means swizzling can't be properly used, since it needs to be applied to
        // the address, and alignment guarantees the offset is aligned to the pattern repeat.
        client.properties().features.alignment
    }

    fn is_supported<R: Runtime>(client: &ComputeClient<R>, config: MmaConfig) -> bool {
        client.properties().features.mma.contains(&config)
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
            .mma
            .iter()
            .filter(|it| it.a_type == lhs_ty && it.b_type == rhs_ty && it.cd_type == acc_ty)
            .map(|it| (it.m, it.n, it.k).into())
            .collect()
    }

    fn validate_blueprint<R: Runtime>(
        client: &ComputeClient<R>,
        blueprint: &TilingBlueprint,
        dtypes: &MatmulElems,
        _line_sizes: &MatmulLineSizes,
    ) -> Result<(), MatmulSetupError> {
        let lhs = dtypes.lhs_register;
        let rhs = dtypes.rhs_register;
        let acc = dtypes.acc_register;

        let size = blueprint.tiling_scheme.tile_size;
        if !client.properties().features.mma.contains(&MmaConfig {
            a_type: lhs,
            b_type: rhs,
            cd_type: acc,
            m: size.m(),
            k: size.k(),
            n: size.n(),
        }) {
            return Err(MatmulSetupError::Unavailable(
                MatmulAvailabilityError::CmmaInstructionUnavailable {
                    lhs,
                    rhs,
                    output: acc,
                    size: Some(TileSize::new(size.m(), size.n(), size.k())),
                },
            ));
        }

        Ok(())
    }
}

fn load_method(dtype: StorageType) -> LoadMethod {
    if !matches!(dtype, StorageType::Packed(_, _))
        && comptime::device_properties()
            .features
            .ldmatrix
            .contains(&dtype)
    {
        LoadMethod::LoadMatrix
    } else {
        LoadMethod::Manual
    }
}

fn store_method(dtype: StorageType) -> StoreMethod {
    if !matches!(dtype, StorageType::Packed(_, _))
        && comptime::device_properties()
            .features
            .stmatrix
            .contains(&dtype)
    {
        StoreMethod::StoreMatrix
    } else {
        StoreMethod::Manual
    }
}
