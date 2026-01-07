use cubecl::{Runtime, client::ComputeClient};

use std::{fmt::Display, marker::PhantomData};

use crate::{
    components::{
        batch::{BatchMatmulFamily, PartitionedBatchMatmulFamily, RowMajorGlobalPartitionMatmul},
        global::{
            UnitWriterFamily,
            read::{FullLoadingStrategy, sync_full_cyclic::SyncFullCyclicLoading},
            single_stage::simple::SimpleMatmulFamily,
        },
        stage::{
            ColMajorTilingOrder, FilledStageFamily, RowMajorTilingOrder, StridedStageFamily,
            UnitMatmulFamily,
        },
        tile::{TileMatmulFamily, io::Filled, register::RegisterMatmul},
    },
    definition::{MatmulElems, MatmulLineSizes, MatmulProblem, MatmulSetupError, TilingBlueprint},
    routines::{
        BlueprintStrategy, DeviceSettings, LaunchInfo,
        selector::{
            PartitionScaling, StageScaling, TileSizeSelection, UnitTilingBlueprintOptions,
            infer_blueprint_unit,
        },
    },
};

use super::Routine;

/// Unit single stage matmul with configurable readers (default to cyclic)
pub struct SimpleUnitAlgorithm<
    LL = SyncFullCyclicLoading<ColMajorTilingOrder>,
    RL = SyncFullCyclicLoading<RowMajorTilingOrder>,
> {
    pub _ll: PhantomData<LL>,
    pub _rl: PhantomData<RL>,
}

#[derive(Default, Clone, Debug)]
pub struct SimpleUnitSelectionArgs {
    pub tile_size: TileSizeSelection,
}

impl Display for SimpleUnitSelectionArgs {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "_{}", self.tile_size)
    }
}

impl<LL, RL> Routine for SimpleUnitAlgorithm<LL, RL>
where
    LL: FullLoadingStrategy,
    RL: FullLoadingStrategy<SyncStrategy = LL::SyncStrategy>,
{
    type Strategy = SimpleUnitSelectionArgs;
    type BatchMatmul = PartitionedBatchMatmulFamily<
        SimpleMatmulFamily<
            UnitMatmulFamily<RegisterMatmul<Filled>, StridedStageFamily, FilledStageFamily>,
            LL,
            RL,
            UnitWriterFamily,
        >,
        RowMajorGlobalPartitionMatmul,
    >;
    type Blueprint = TilingBlueprint;
    type Config = <Self::BatchMatmul as BatchMatmulFamily>::Config;

    fn prepare<R: Runtime>(
        problem: &MatmulProblem,
        device_settings: &DeviceSettings<R>,
        strategy: &BlueprintStrategy<Self>,
    ) -> Result<LaunchInfo<TilingBlueprint>, MatmulSetupError> {
        let mut dtypes = MatmulElems::from_globals(&problem.global_dtypes);

        if RegisterMatmul::<Filled>::can_cast_stage_element() {
            dtypes.adjust_stage_dtypes();
        }

        let (blueprint, dtypes) = match strategy {
            BlueprintStrategy::Forced(blueprint) => (blueprint.clone(), dtypes),
            BlueprintStrategy::Inferred(strategy) => infer_blueprint_unit(
                &device_settings.client,
                problem,
                device_settings.plane_dim,
                false,
                &device_settings.line_sizes,
                UnitTilingBlueprintOptions {
                    tile: strategy.tile_size,
                    stage: match strategy.tile_size {
                        TileSizeSelection::MinTileSize => StageScaling::Enabled(2),
                        TileSizeSelection::MaxTileSize => StageScaling::Disabled,
                    },
                    partition: match strategy.tile_size {
                        TileSizeSelection::MinTileSize => PartitionScaling::Disabled,
                        TileSizeSelection::MaxTileSize => PartitionScaling::Enabled,
                    },
                    swizzle: <RegisterMatmul as TileMatmulFamily>::should_swizzle(
                        &device_settings.client,
                    ),
                },
                &problem.global_dtypes,
            ),
        };

        Self::validate_blueprint(
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

    fn device_settings<R: Runtime>(
        client: &ComputeClient<R>,
        line_sizes: MatmulLineSizes,
    ) -> DeviceSettings<R> {
        let plane_dim = match client.properties().hardware.plane_size_min {
            0 => 32,
            plane_dim => plane_dim,
        };

        DeviceSettings {
            client: client.clone(),
            plane_dim,
            line_sizes,
            max_cube_count: client.properties().hardware.max_cube_count,
        }
    }
}
