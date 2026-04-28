use std::fmt::Display;

use cubecl::Runtime;

use crate::components::batch::{PartitionedBatchMatmulFamily, RowMajorGlobalPartitionMatmul};
use crate::components::global::{
    PlaneWriterFamily, read::sync_partial_tilewise::SyncPartialTilewiseLoading,
};
use crate::components::{
    batch::BatchMatmulFamily, global::read::sync_full_cyclic::SyncFullCyclicLoading,
};
use crate::components::{
    global::multi_stage::double_buffering::DoubleBufferingMatmulFamily, stage::StridedStageFamily,
};
use crate::definition::{
    MatmulElems, MatmulProblem, MatmulSetupError, MultiRowStrategy, TilingBlueprint,
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
    components::stage::{ColMajorTilingOrder, PlaneMatmulFamily, RowMajorTilingOrder},
    components::tile_matmul::{TileMatmul, TileMatmulFamily as _},
};
use crate::{
    launch::RuntimeConfig,
    routines::DeviceSettings,
    routines::selector::{PlaneTilingBlueprintOptions, infer_blueprint_plane},
    routines::{BlueprintStrategy, LaunchInfo, TilingArgs, base},
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
    pub tile_matmul: TileMatmul,
    pub specialized: bool,
}

impl Default for DoubleBufferingArgs {
    fn default() -> Self {
        Self {
            tile_matmul: TileMatmul::Cmma,
            specialized: false,
        }
    }
}

impl TilingArgs for DoubleBufferingArgs {
    fn set_tile_matmul(&mut self, kind: TileMatmul) {
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
            type BatchMatmul = $batch;
            type Blueprint = TilingBlueprint;
            type Config = <Self::BatchMatmul as BatchMatmulFamily<RC>>::Config;

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
            ) -> Result<LaunchInfo<TilingBlueprint>, MatmulSetupError> {
                let ExpandInfo { blueprint, dtypes } = expand_info;

                <Self as base::Routine<RC>>::validate_blueprint(
                    &device_settings.client,
                    &blueprint,
                    problem,
                    &dtypes,
                    &device_settings.vector_sizes,
                )?;

                let cubedim_resource = Self::BatchMatmul::cubedim_resource(
                    &blueprint,
                    &dtypes,
                    &device_settings.vector_sizes,
                )?;

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
            PlaneMatmulFamily<StridedStageFamily, StridedStageFamily, Option<StridedStageFamily>>,
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
            PlaneMatmulFamily<StridedStageFamily, StridedStageFamily, Option<StridedStageFamily>>,
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
            PlaneMatmulFamily<StridedStageFamily, StridedStageFamily, Option<StridedStageFamily>>,
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
            PlaneMatmulFamily<StridedStageFamily, StridedStageFamily, Option<StridedStageFamily>>,
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
            PlaneMatmulFamily<StridedStageFamily, StridedStageFamily, Option<StridedStageFamily>>,
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
            PlaneMatmulFamily<StridedStageFamily, StridedStageFamily, Option<StridedStageFamily>>,
            RC,
            AsyncPartialStridedLoading,
            AsyncPartialStridedLoading,
            AsyncFullStridedLoading,
            PlaneWriterFamily,
        >,
        RowMajorGlobalPartitionMatmul,
    >
);
