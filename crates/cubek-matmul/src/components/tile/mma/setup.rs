use crate::{
    components::{
        resource::CubeDimResource,
        tile::{
            SharedTileConfig, TileMatmulFamily,
            mma::{MmaMatmul, config::MmaMatmulConfig},
        },
    },
    definition::{
        MatmulAvailabilityError, MatmulElems, MatmulSetupError, MatmulVectorSizes, TilingBlueprint,
    },
};
use cubecl::{
    {features::MmaConfig, ir::DeviceProperties},
    {ir::StorageType, prelude::*},
};
use cubek_std::{
    InvalidConfigError, TileSize,
    tile::{
        Strided,
        mma::{MmaFragmentReader, MmaIOConfig, MmaStageReader},
    },
};

impl TileMatmulFamily for MmaMatmul
where
    MmaStageReader<Strided>: MmaFragmentReader<TileKind = Strided>,
    MmaStageReader<Strided>: MmaFragmentReader<TileKind = Strided>,
    MmaStageReader<Option<Strided>>: MmaFragmentReader<TileKind = Option<Strided>>,
{
    type Config = MmaMatmulConfig;

    type Matmul<L: Numeric, NL: Size, R: Numeric, NR: Size, A: Numeric, NA: Size> = MmaMatmul;

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
        _vector_sizes: &MatmulVectorSizes,
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
        client.properties().features.matmul.mma.contains(&config)
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
            .mma
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

        Ok(())
    }
}
