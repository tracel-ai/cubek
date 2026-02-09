use std::fmt::Display;
use std::marker::PhantomData;

use cubecl::{Runtime, std::CubeOption};

use crate::components::global::{
    PlaneWriterFamily, read::sync_partial_tilewise::SyncPartialTilewiseLoading,
};
use crate::components::stage::{ColMajorTilingOrder, PlaneMatmulFamily, RowMajorTilingOrder};
use crate::components::tile;
use crate::components::{
    batch::BatchMatmulFamily, global::read::sync_full_cyclic::SyncFullCyclicLoading,
};
use crate::components::{
    batch::{PartitionedBatchMatmulFamily, RowMajorGlobalPartitionMatmul},
    tile::io::Strided,
};
use crate::components::{
    global::multi_stage::double_buffering::DoubleBufferingMatmulFamily, stage::StridedStageFamily,
};
use crate::definition::{
    MatmulElems, MatmulProblem, MatmulSetupError, MultiRowStrategy, TilingBlueprint,
};
use crate::launch::RuntimeConfig;
use crate::routines::selector::{PlaneTilingBlueprintOptions, infer_blueprint_plane};
use crate::routines::{BlueprintStrategy, LaunchInfo, base};
use crate::routines::{DeviceSettings, Routine};
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

/// Plane accelerated double buffered matmul with cyclic readers
pub struct CyclicDoubleBufferingAlgorithm<TMM> {
    pub _phantom: PhantomData<TMM>,
}

/// Plane accelerated double buffered matmul with cyclic readers
pub struct AsyncCyclicDoubleBufferingAlgorithm<TMM> {
    pub _phantom: PhantomData<TMM>,
}

/// Plane accelerated double buffered matmul with tilewise readers
pub struct TilewiseDoubleBufferingAlgorithm<TMM> {
    pub _phantom: PhantomData<TMM>,
}

/// Plane accelerated double buffered matmul with tilewise reader on Lhs and cyclic on Rhs
pub struct HybridDoubleBufferingAlgorithm<TMM> {
    pub _phantom: PhantomData<TMM>,
}

/// Plane accelerated double buffered matmul with TMA readers
pub struct TmaDoubleBufferingAlgorithm<TMM> {
    pub _phantom: PhantomData<TMM>,
}

/// Plane accelerated double buffered matmul with cyclic readers
pub struct AsyncStridedDoubleBufferingAlgorithm<TMM> {
    pub _phantom: PhantomData<TMM>,
}

#[derive(Default, Debug, Clone, Copy)]
pub struct DoubleBufferingArgs {
    pub specialized: bool,
}

impl Display for DoubleBufferingArgs {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(if self.specialized { "_specialized" } else { "" })
    }
}

impl<TMM, RC> base::Routine<RC> for CyclicDoubleBufferingAlgorithm<TMM>
where
    TMM: tile::TileMatmulFamily<
            LhsTile = Strided,
            RhsTile = Strided,
            AccTile = CubeOption<Strided>,
            OutTile = Strided,
        >,
    RC: RuntimeConfig,
{
    type Strategy = DoubleBufferingArgs;

    type BatchMatmul = PartitionedBatchMatmulFamily<
        RC,
        DoubleBufferingMatmulFamily<
            PlaneMatmulFamily<
                TMM,
                StridedStageFamily,
                StridedStageFamily,
                Option<StridedStageFamily>,
            >,
            RC,
            SyncPartialCyclicLoading<RowMajorTilingOrder>,
            SyncPartialCyclicLoading<RowMajorTilingOrder>,
            SyncFullCyclicLoading<RowMajorTilingOrder>,
            PlaneWriterFamily,
        >,
        RowMajorGlobalPartitionMatmul,
    >;
    type Blueprint = TilingBlueprint;
    type Config = <Self::BatchMatmul as BatchMatmulFamily<RC>>::Config;

    fn expand_blueprint<R: Runtime>(
        problem: &MatmulProblem,
        device_settings: &DeviceSettings<R>,
        strategy: &BlueprintStrategy<RC, Self>,
    ) -> Result<ExpandInfo<Self::Blueprint>, MatmulSetupError> {
        let mut dtypes = MatmulElems::from_globals(&problem.global_dtypes);

        if TMM::can_cast_stage_element() {
            dtypes.adjust_stage_dtypes();
        }

        let (blueprint, dtypes) = match strategy {
            BlueprintStrategy::Forced(blueprint) => (blueprint.clone(), dtypes),
            BlueprintStrategy::Inferred(strategy) => infer_blueprint_plane::<TMM, R>(
                &device_settings.client,
                problem,
                device_settings.plane_dim,
                dtypes,
                &device_settings.line_sizes,
                PlaneTilingBlueprintOptions {
                    specialized: strategy.specialized,
                    multi_row_strategy: MultiRowStrategy::Adaptive {
                        minimum_stage_count: 8,
                    },
                    swizzled: TMM::should_swizzle(&device_settings.client),
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
            &device_settings.line_sizes,
        )?;

        let cubedim_resource =
            Self::BatchMatmul::cubedim_resource(&blueprint, &dtypes, &device_settings.line_sizes)?;

        LaunchInfo::new(
            blueprint,
            dtypes,
            problem,
            cubedim_resource,
            device_settings,
        )
    }
}

impl<TMM, RC> base::Routine<RC> for AsyncCyclicDoubleBufferingAlgorithm<TMM>
where
    TMM: tile::TileMatmulFamily<
            LhsTile = Strided,
            RhsTile = Strided,
            AccTile = CubeOption<Strided>,
            OutTile = Strided,
        >,
    RC: RuntimeConfig,
{
    type Strategy = DoubleBufferingArgs;
    type BatchMatmul = PartitionedBatchMatmulFamily<
        RC,
        DoubleBufferingMatmulFamily<
            PlaneMatmulFamily<
                TMM,
                StridedStageFamily,
                StridedStageFamily,
                Option<StridedStageFamily>,
            >,
            RC,
            AsyncPartialCyclicLoading<RowMajorTilingOrder>,
            AsyncPartialCyclicLoading<RowMajorTilingOrder>,
            AsyncFullCyclicLoading<RowMajorTilingOrder>,
            PlaneWriterFamily,
        >,
        RowMajorGlobalPartitionMatmul,
    >;
    type Blueprint = TilingBlueprint;
    type Config = <Self::BatchMatmul as BatchMatmulFamily<RC>>::Config;

    fn expand_blueprint<R: Runtime>(
        problem: &MatmulProblem,
        device_settings: &DeviceSettings<R>,
        strategy: &BlueprintStrategy<RC, Self>,
    ) -> Result<ExpandInfo<Self::Blueprint>, MatmulSetupError> {
        let mut dtypes = MatmulElems::from_globals(&problem.global_dtypes);

        if TMM::can_cast_stage_element() {
            dtypes.adjust_stage_dtypes();
        }

        let (blueprint, dtypes) = match strategy {
            BlueprintStrategy::Forced(blueprint) => (blueprint.clone(), dtypes),
            BlueprintStrategy::Inferred(strategy) => infer_blueprint_plane::<TMM, R>(
                &device_settings.client,
                problem,
                device_settings.plane_dim,
                dtypes,
                &device_settings.line_sizes,
                PlaneTilingBlueprintOptions {
                    specialized: strategy.specialized,
                    multi_row_strategy: MultiRowStrategy::Adaptive {
                        minimum_stage_count: 8,
                    },
                    swizzled: TMM::should_swizzle(&device_settings.client),
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
            &device_settings.line_sizes,
        )?;

        let cubedim_resource =
            Self::BatchMatmul::cubedim_resource(&blueprint, &dtypes, &device_settings.line_sizes)?;

        LaunchInfo::new(
            blueprint,
            dtypes,
            problem,
            cubedim_resource,
            device_settings,
        )
    }
}

impl<TMM, RC> Routine<RC> for TilewiseDoubleBufferingAlgorithm<TMM>
where
    TMM: tile::TileMatmulFamily<
            LhsTile = Strided,
            RhsTile = Strided,
            AccTile = CubeOption<Strided>,
            OutTile = Strided,
        >,
    RC: RuntimeConfig,
{
    type Strategy = DoubleBufferingArgs;

    type BatchMatmul = PartitionedBatchMatmulFamily<
        RC,
        DoubleBufferingMatmulFamily<
            PlaneMatmulFamily<
                TMM,
                StridedStageFamily,
                StridedStageFamily,
                Option<StridedStageFamily>,
            >,
            RC,
            // Other tiling orders are not supported
            SyncPartialTilewiseLoading<RowMajorTilingOrder>,
            SyncPartialTilewiseLoading<ColMajorTilingOrder>,
            SyncFullTilewiseLoading<ColMajorTilingOrder>,
            PlaneWriterFamily,
        >,
        RowMajorGlobalPartitionMatmul,
    >;
    type Blueprint = TilingBlueprint;
    type Config = <Self::BatchMatmul as BatchMatmulFamily<RC>>::Config;

    fn expand_blueprint<R: Runtime>(
        problem: &MatmulProblem,
        device_settings: &DeviceSettings<R>,
        strategy: &BlueprintStrategy<RC, Self>,
    ) -> Result<ExpandInfo<Self::Blueprint>, MatmulSetupError> {
        let mut dtypes = MatmulElems::from_globals(&problem.global_dtypes);

        if TMM::can_cast_stage_element() {
            dtypes.adjust_stage_dtypes();
        }

        let (blueprint, dtypes) = match strategy {
            BlueprintStrategy::Forced(blueprint) => (blueprint.clone(), dtypes),
            BlueprintStrategy::Inferred(strategy) => infer_blueprint_plane::<TMM, R>(
                &device_settings.client,
                problem,
                device_settings.plane_dim,
                dtypes,
                &device_settings.line_sizes,
                PlaneTilingBlueprintOptions {
                    specialized: strategy.specialized,
                    multi_row_strategy: MultiRowStrategy::Adaptive {
                        minimum_stage_count: 8,
                    },
                    swizzled: TMM::should_swizzle(&device_settings.client),
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
            &device_settings.line_sizes,
        )?;

        let cubedim_resource =
            Self::BatchMatmul::cubedim_resource(&blueprint, &dtypes, &device_settings.line_sizes)?;

        LaunchInfo::new(
            blueprint,
            dtypes,
            problem,
            cubedim_resource,
            device_settings,
        )
    }
}

impl<TMM, RC> base::Routine<RC> for HybridDoubleBufferingAlgorithm<TMM>
where
    TMM: tile::TileMatmulFamily<
            LhsTile = Strided,
            RhsTile = Strided,
            AccTile = CubeOption<Strided>,
            OutTile = Strided,
        >,
    RC: RuntimeConfig,
{
    type Strategy = DoubleBufferingArgs;

    type BatchMatmul = PartitionedBatchMatmulFamily<
        RC,
        DoubleBufferingMatmulFamily<
            PlaneMatmulFamily<
                TMM,
                StridedStageFamily,
                StridedStageFamily,
                Option<StridedStageFamily>,
            >,
            RC,
            SyncPartialTilewiseLoading<RowMajorTilingOrder>,
            SyncPartialCyclicLoading<RowMajorTilingOrder>,
            SyncFullCyclicLoading<RowMajorTilingOrder>,
            PlaneWriterFamily,
        >,
        RowMajorGlobalPartitionMatmul,
    >;
    type Blueprint = TilingBlueprint;
    type Config = <Self::BatchMatmul as BatchMatmulFamily<RC>>::Config;

    fn expand_blueprint<R: Runtime>(
        problem: &MatmulProblem,
        device_settings: &DeviceSettings<R>,
        strategy: &BlueprintStrategy<RC, Self>,
    ) -> Result<ExpandInfo<Self::Blueprint>, MatmulSetupError> {
        let mut dtypes = MatmulElems::from_globals(&problem.global_dtypes);

        if TMM::can_cast_stage_element() {
            dtypes.adjust_stage_dtypes();
        }

        let (blueprint, dtypes) = match strategy {
            BlueprintStrategy::Forced(blueprint) => (blueprint.clone(), dtypes),
            BlueprintStrategy::Inferred(strategy) => infer_blueprint_plane::<TMM, R>(
                &device_settings.client,
                problem,
                device_settings.plane_dim,
                dtypes,
                &device_settings.line_sizes,
                PlaneTilingBlueprintOptions {
                    specialized: strategy.specialized,
                    multi_row_strategy: MultiRowStrategy::Adaptive {
                        minimum_stage_count: 8,
                    },
                    swizzled: TMM::should_swizzle(&device_settings.client),
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
            &device_settings.line_sizes,
        )?;

        let cubedim_resource =
            Self::BatchMatmul::cubedim_resource(&blueprint, &dtypes, &device_settings.line_sizes)?;

        LaunchInfo::new(
            blueprint,
            dtypes,
            problem,
            cubedim_resource,
            device_settings,
        )
    }
}

impl<TMM, RC> base::Routine<RC> for TmaDoubleBufferingAlgorithm<TMM>
where
    TMM: tile::TileMatmulFamily<
            LhsTile = Strided,
            RhsTile = Strided,
            AccTile = CubeOption<Strided>,
            OutTile = Strided,
        >,
    RC: RuntimeConfig,
{
    type Strategy = DoubleBufferingArgs;
    type BatchMatmul = PartitionedBatchMatmulFamily<
        RC,
        DoubleBufferingMatmulFamily<
            PlaneMatmulFamily<
                TMM,
                StridedStageFamily,
                StridedStageFamily,
                Option<StridedStageFamily>,
            >,
            RC,
            AsyncPartialTmaLoading,
            AsyncPartialTmaLoading,
            AsyncFullTmaLoading,
            PlaneWriterFamily,
        >,
        RowMajorGlobalPartitionMatmul,
    >;
    type Blueprint = TilingBlueprint;
    type Config = <Self::BatchMatmul as BatchMatmulFamily<RC>>::Config;

    fn expand_blueprint<R: Runtime>(
        problem: &MatmulProblem,
        device_settings: &DeviceSettings<R>,
        strategy: &BlueprintStrategy<RC, Self>,
    ) -> Result<ExpandInfo<Self::Blueprint>, MatmulSetupError> {
        let mut dtypes = MatmulElems::from_globals(&problem.global_dtypes);

        if TMM::can_cast_stage_element() {
            dtypes.adjust_stage_dtypes();
        }

        let (blueprint, dtypes) = match strategy {
            BlueprintStrategy::Forced(blueprint) => (blueprint.clone(), dtypes),
            BlueprintStrategy::Inferred(strategy) => infer_blueprint_plane::<TMM, R>(
                &device_settings.client,
                problem,
                device_settings.plane_dim,
                dtypes,
                &device_settings.line_sizes,
                PlaneTilingBlueprintOptions {
                    specialized: strategy.specialized,
                    multi_row_strategy: MultiRowStrategy::Adaptive {
                        minimum_stage_count: 8,
                    },
                    swizzled: TMM::should_swizzle(&device_settings.client),
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
            &device_settings.line_sizes,
        )?;

        let cubedim_resource =
            Self::BatchMatmul::cubedim_resource(&blueprint, &dtypes, &device_settings.line_sizes)?;

        LaunchInfo::new(
            blueprint,
            dtypes,
            problem,
            cubedim_resource,
            device_settings,
        )
    }
}

impl<TMM, RC> base::Routine<RC> for AsyncStridedDoubleBufferingAlgorithm<TMM>
where
    TMM: tile::TileMatmulFamily<
            LhsTile = Strided,
            RhsTile = Strided,
            AccTile = CubeOption<Strided>,
            OutTile = Strided,
        >,
    RC: RuntimeConfig,
{
    type Strategy = DoubleBufferingArgs;
    type BatchMatmul = PartitionedBatchMatmulFamily<
        RC,
        DoubleBufferingMatmulFamily<
            PlaneMatmulFamily<
                TMM,
                StridedStageFamily,
                StridedStageFamily,
                Option<StridedStageFamily>,
            >,
            RC,
            AsyncPartialStridedLoading,
            AsyncPartialStridedLoading,
            AsyncFullStridedLoading,
            PlaneWriterFamily,
        >,
        RowMajorGlobalPartitionMatmul,
    >;
    type Blueprint = TilingBlueprint;
    type Config = <Self::BatchMatmul as BatchMatmulFamily<RC>>::Config;

    fn expand_blueprint<R: Runtime>(
        problem: &MatmulProblem,
        device_settings: &DeviceSettings<R>,
        strategy: &BlueprintStrategy<RC, Self>,
    ) -> Result<ExpandInfo<Self::Blueprint>, MatmulSetupError> {
        let mut dtypes = MatmulElems::from_globals(&problem.global_dtypes);

        if TMM::can_cast_stage_element() {
            dtypes.adjust_stage_dtypes();
        }

        let (blueprint, dtypes) = match strategy {
            BlueprintStrategy::Forced(blueprint) => (blueprint.clone(), dtypes),
            BlueprintStrategy::Inferred(strategy) => infer_blueprint_plane::<TMM, R>(
                &device_settings.client,
                problem,
                device_settings.plane_dim,
                dtypes,
                &device_settings.line_sizes,
                PlaneTilingBlueprintOptions {
                    specialized: strategy.specialized,
                    multi_row_strategy: MultiRowStrategy::Adaptive {
                        minimum_stage_count: 8,
                    },
                    swizzled: TMM::should_swizzle(&device_settings.client),
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
            &device_settings.line_sizes,
        )?;

        let cubedim_resource =
            Self::BatchMatmul::cubedim_resource(&blueprint, &dtypes, &device_settings.line_sizes)?;

        LaunchInfo::new(
            blueprint,
            dtypes,
            problem,
            cubedim_resource,
            device_settings,
        )
    }
}
