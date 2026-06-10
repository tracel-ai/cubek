use std::fmt::Display;

use cubecl::{CubeCount, CubeDim, Runtime, client::ComputeClient, ir::AddressType};
use cubek_std::tile::{ColMajorTilingOrder, RowMajorTilingOrder};

use crate::components::batch::{PartitionedBatchMatmulFamily, RowMajorGlobalPartitionMatmul};
use crate::components::global::multi_stage::double_buffering::DoubleBufferingMatmulFamily;
use crate::components::global::{
    PlaneWriterFamily, read::sync_partial_tilewise::SyncPartialTilewiseLoading,
};
use crate::components::{
    batch::BatchMatmulFamily, global::read::sync_full_cyclic::SyncFullCyclicLoading,
};
use crate::definition::{
    BatchMatmulBlueprint, CubeMappingLaunch, MatmulElems, MatmulProblem, MatmulSetupError,
    MatmulVectorSizes, MultiRowStrategy,
};
use crate::{
    args::{ConfigRuntimeArg, InputRuntimeArg, MatmulArgs, OutputRuntimeArg, RuntimeConfig},
    routines::DeviceSettings,
    routines::selector::{PlaneTilingBlueprintOptions, infer_blueprint_plane},
    routines::{BlueprintStrategy, LaunchInfo, TilingArgs, base, batch_validate_blueprint},
};
use crate::{
    components::global::read::{
        async_full_cyclic::AsyncFullCyclicLoading, async_full_strided::AsyncFullStridedLoading,
        async_full_tma::AsyncFullTmaLoading, async_partial_cyclic::AsyncPartialCyclicLoading,
        async_partial_strided::AsyncPartialStridedLoading,
        async_partial_tma::AsyncPartialTmaLoading, sync_full_tilewise::SyncFullTilewiseLoading,
        sync_partial_cyclic::SyncPartialCyclicLoading,
    },
    routines::ExpandInfo,
};
use crate::{
    components::stage::{NumStages, PlanePartitioner},
    components::tile::TileMatmulKind,
};

/// Plane accelerated double buffered matmul with cyclic readers
pub struct CyclicDoubleBufferingAlgorithm;

/// Plane accelerated double buffered matmul with cyclic readers
pub struct AsyncCyclicDoubleBufferingAlgorithm;

/// Plane accelerated double buffered matmul with tilewise readers
pub struct TilewiseDoubleBufferingAlgorithm;

/// Plane accelerated double buffered matmul with tilewise reader on Lhs and cyclic on Rhs
pub struct HybridDoubleBufferingAlgorithm;

/// Plane accelerated double buffered matmul with TMA readers
pub struct TmaDoubleBufferingAlgorithm;

/// Plane accelerated double buffered matmul with cyclic readers
pub struct AsyncStridedDoubleBufferingAlgorithm;

#[derive(Debug, Clone, Copy)]
pub struct DoubleBufferingArgs {
    pub tile_matmul: TileMatmulKind,
    pub specialized: bool,
}

impl Default for DoubleBufferingArgs {
    fn default() -> Self {
        Self {
            tile_matmul: TileMatmulKind::Cmma,
            specialized: false,
        }
    }
}

impl TilingArgs for DoubleBufferingArgs {
    fn set_tile_matmul(&mut self, kind: TileMatmulKind) {
        self.tile_matmul = kind;
    }
}

impl Display for DoubleBufferingArgs {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(if self.specialized { "_specialized" } else { "" })
    }
}

macro_rules! double_buffering_impl {
    ($algo:ident, $batch:ty) => {
        impl<RC> base::Routine<RC> for $algo
        where
            RC: RuntimeConfig,
        {
            type Strategy = DoubleBufferingArgs;
            type Blueprint = BatchMatmulBlueprint;
        }

        impl<RC> base::BatchMatmulRoutine<RC> for $algo
        where
            RC: RuntimeConfig,
        {
            #[allow(clippy::too_many_arguments, clippy::result_large_err)]
            fn launch<MA: MatmulArgs<Config = RC>, R: Runtime>(
                client: &ComputeClient<R>,
                cube_dim: CubeDim,
                cube_count: CubeCount,
                address_type: AddressType,
                input: InputRuntimeArg<MA, R>,
                output: OutputRuntimeArg<MA, R>,
                config: ConfigRuntimeArg<MA, R>,
                cube_count_input: CubeMappingLaunch<R>,
                blueprint: Self::Blueprint,
                dtypes: &MatmulElems,
                vector_sizes: &MatmulVectorSizes,
            ) -> Result<(), MatmulSetupError> {
                {
                    unsafe {
                        <$batch>::launch_unchecked::<MA, R>(
                            client,
                            cube_dim,
                            cube_count,
                            address_type,
                            input,
                            output,
                            config,
                            cube_count_input,
                            blueprint,
                            dtypes,
                            vector_sizes,
                        )?
                    }
                    Ok(())
                }
            }

            #[allow(clippy::result_large_err)]
            fn validate_blueprint<R: Runtime>(
                client: &ComputeClient<R>,
                blueprint: &Self::Blueprint,
                problem: &MatmulProblem,
                dtypes: &MatmulElems,
                vector_sizes: &MatmulVectorSizes,
            ) -> Result<(), MatmulSetupError> {
                batch_validate_blueprint::<$batch, RC, R>(
                    client,
                    blueprint,
                    problem,
                    dtypes,
                    vector_sizes,
                )
            }

            fn num_stages() -> NumStages {
                <$batch>::num_stages()
            }

            fn expand_blueprint<R: Runtime>(
                problem: &MatmulProblem,
                device_settings: &DeviceSettings<R>,
                strategy: &BlueprintStrategy<RC, Self>,
            ) -> Result<ExpandInfo<Self::Blueprint>, MatmulSetupError> {
                let mut dtypes = MatmulElems::from_globals(&problem.global_dtypes);

                let tile_matmul = match strategy {
                    BlueprintStrategy::Forced(blueprint) => blueprint.tile_matmul,
                    BlueprintStrategy::Inferred(args) => args.tile_matmul,
                };

                if tile_matmul.can_cast_stage_element() {
                    dtypes.adjust_stage_dtypes();
                }

                let (blueprint, dtypes) = match strategy {
                    BlueprintStrategy::Forced(blueprint) => (blueprint.clone(), dtypes),
                    BlueprintStrategy::Inferred(strategy) => infer_blueprint_plane::<R>(
                        tile_matmul,
                        &device_settings.client,
                        problem,
                        device_settings.plane_dim,
                        dtypes,
                        &device_settings.vector_sizes,
                        PlaneTilingBlueprintOptions {
                            specialized: strategy.specialized,
                            multi_row_strategy: MultiRowStrategy::Adaptive {
                                minimum_stage_count: 8,
                            },
                            swizzled: tile_matmul.should_swizzle(&device_settings.client),
                            ..Default::default()
                        },
                    )?,
                };
                Ok(ExpandInfo { blueprint, dtypes })
            }

            fn prepare<R: Runtime>(
                problem: &MatmulProblem,
                device_settings: &DeviceSettings<R>,
                expand_info: ExpandInfo<Self::Blueprint>,
            ) -> Result<LaunchInfo<BatchMatmulBlueprint>, MatmulSetupError> {
                let ExpandInfo { blueprint, dtypes } = expand_info;

                <Self as base::BatchMatmulRoutine<RC>>::validate_blueprint(
                    &device_settings.client,
                    &blueprint,
                    problem,
                    &dtypes,
                    &device_settings.vector_sizes,
                )?;

                let cubedim_resource =
                    <$batch>::cubedim_resource(&blueprint, &dtypes, &device_settings.vector_sizes)?;

                LaunchInfo::new(
                    blueprint,
                    dtypes,
                    problem,
                    cubedim_resource,
                    device_settings,
                )
            }
        }
    };
}

double_buffering_impl!(
    CyclicDoubleBufferingAlgorithm,
    PartitionedBatchMatmulFamily<
        RC,
        DoubleBufferingMatmulFamily<
            PlanePartitioner,
            RC,
            SyncPartialCyclicLoading<RowMajorTilingOrder>,
            SyncPartialCyclicLoading<RowMajorTilingOrder>,
            SyncFullCyclicLoading<RowMajorTilingOrder>,
            PlaneWriterFamily,
        >,
        RowMajorGlobalPartitionMatmul,
    >
);

double_buffering_impl!(
    AsyncCyclicDoubleBufferingAlgorithm,
    PartitionedBatchMatmulFamily<
        RC,
        DoubleBufferingMatmulFamily<
            PlanePartitioner,
            RC,
            AsyncPartialCyclicLoading<RowMajorTilingOrder>,
            AsyncPartialCyclicLoading<RowMajorTilingOrder>,
            AsyncFullCyclicLoading<RowMajorTilingOrder>,
            PlaneWriterFamily,
        >,
        RowMajorGlobalPartitionMatmul,
    >
);

double_buffering_impl!(
    TilewiseDoubleBufferingAlgorithm,
    PartitionedBatchMatmulFamily<
        RC,
        DoubleBufferingMatmulFamily<
            PlanePartitioner,
            RC,
            SyncPartialTilewiseLoading<RowMajorTilingOrder>,
            SyncPartialTilewiseLoading<ColMajorTilingOrder>,
            SyncFullTilewiseLoading<ColMajorTilingOrder>,
            PlaneWriterFamily,
        >,
        RowMajorGlobalPartitionMatmul,
    >
);

double_buffering_impl!(
    HybridDoubleBufferingAlgorithm,
    PartitionedBatchMatmulFamily<
        RC,
        DoubleBufferingMatmulFamily<
            PlanePartitioner,
            RC,
            SyncPartialTilewiseLoading<RowMajorTilingOrder>,
            SyncPartialCyclicLoading<RowMajorTilingOrder>,
            SyncFullCyclicLoading<RowMajorTilingOrder>,
            PlaneWriterFamily,
        >,
        RowMajorGlobalPartitionMatmul,
    >
);

double_buffering_impl!(
    TmaDoubleBufferingAlgorithm,
    PartitionedBatchMatmulFamily<
        RC,
        DoubleBufferingMatmulFamily<
            PlanePartitioner,
            RC,
            AsyncPartialTmaLoading,
            AsyncPartialTmaLoading,
            AsyncFullTmaLoading,
            PlaneWriterFamily,
        >,
        RowMajorGlobalPartitionMatmul,
    >
);

double_buffering_impl!(
    AsyncStridedDoubleBufferingAlgorithm,
    PartitionedBatchMatmulFamily<
        RC,
        DoubleBufferingMatmulFamily<
            PlanePartitioner,
            RC,
            AsyncPartialStridedLoading,
            AsyncPartialStridedLoading,
            AsyncFullStridedLoading,
            PlaneWriterFamily,
        >,
        RowMajorGlobalPartitionMatmul,
    >
);
