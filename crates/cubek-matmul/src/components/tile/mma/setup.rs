use crate::components::resource::CubeDimResource;
use crate::components::tile::SharedTileConfig;
use crate::components::tile::mma::config::MmaMatmulConfig;
use crate::components::tile::{TileMatmulFamily, mma::MmaMatmul};
use crate::definition::{MatmulAvailabilityError, MatmulElems, MatmulSetupError};
use crate::definition::{MatmulLineSizes, TilingBlueprint};
use cubecl::{features::MmaConfig, ir::DeviceProperties};
use cubecl::{ir::StorageType, prelude::*};
use cubek_std::tile::mma::{MmaFragmentReader, MmaIOConfig, MmaStageReader};
use cubek_std::tile::{Strided, TileKind};
use cubek_std::{InvalidConfigError, TileSize};

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
        device_props: &DeviceProperties,
        blueprint: &TilingBlueprint,
        dtypes: &MatmulElems,
        _line_sizes: &MatmulLineSizes,
    ) -> Result<Self::Config, MatmulSetupError> {
        Ok(MmaMatmulConfig {
            shared: SharedTileConfig {
                tile_size: blueprint.tiling_scheme.tile_size,
                plane_dim: blueprint.plane_dim,
                swizzle_modes: blueprint.swizzle_modes,
            },
            mma_io_config: MmaIOConfig::new(
                device_props,
                dtypes.lhs_stage,
                dtypes.rhs_stage,
                dtypes.acc_stage,
            ),
        })
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
