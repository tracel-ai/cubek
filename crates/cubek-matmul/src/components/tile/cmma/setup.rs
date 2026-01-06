use crate::components::tile::SharedTileConfig;
use crate::components::tile::cmma::matmul::CmmaMatmul;
use crate::components::tile::io::Strided;
use crate::components::tile::{
    TileMatmulFamily,
    cmma::reader::{CmmaFragmentReader, CmmaStageReader},
};
use crate::components::{resource::CubeDimResource, tile::io::TileKind};
use crate::definition::{
    InvalidConfigError, MatmulAvailabilityError, MatmulLineSizes, MatmulSetupError, TileSize,
};
use crate::definition::{MatmulElems, TilingBlueprint};
use cubecl::features::MmaConfig;
use cubecl::{ir::StorageType, prelude::*};

impl<Tile: TileKind> TileMatmulFamily for CmmaMatmul<Tile>
where
    CmmaStageReader<Tile>: CmmaFragmentReader<TileKind = Tile>,
{
    type Config = SharedTileConfig;
    type Matmul<L: Numeric, R: Numeric, A: Numeric> = CmmaMatmul<Tile>;
    type LhsTile = Strided;
    type RhsTile = Strided;
    type AccTile = Tile;
    type OutTile = Strided;

    fn requires_accelerator() -> bool {
        true
    }

    fn can_cast_stage_element() -> bool {
        false
    }

    fn cubedim_resource() -> Result<CubeDimResource, InvalidConfigError> {
        Ok(CubeDimResource::Planes(1))
    }

    fn expand_config(
        blueprint: &TilingBlueprint,
        _dtypes: &MatmulElems,
        _line_sizes: &MatmulLineSizes,
    ) -> Result<SharedTileConfig, MatmulSetupError> {
        Ok(SharedTileConfig::new(
            blueprint.tiling_scheme.tile_size,
            blueprint.plane_dim,
            blueprint.swizzle_modes,
        ))
    }

    fn should_swizzle<R: Runtime>(_client: &ComputeClient<R>) -> bool {
        // Unsupported
        false
    }

    fn is_supported<R: Runtime>(client: &ComputeClient<R>, config: MmaConfig) -> bool {
        client.properties().features.cmma.contains(&config)
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
            .cmma
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
        if !client.properties().features.cmma.contains(&MmaConfig {
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

        if blueprint.swizzle_modes.has_swizzle() {
            return Err(MatmulSetupError::InvalidConfig(Box::new(
                "This tile matmul doesn't support swizzling",
            )));
        }

        Ok(())
    }
}
