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
    definition::{MatmulElems, MatmulProblem, MatmulSetupError, TilingBlueprint},
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
        match strategy {
            BlueprintStrategy::Forced(blueprint) => Ok(LaunchInfo {
                blueprint: blueprint.clone(),
                dtypes: MatmulElems::from_globals(&problem.global_dtypes),
            }),
            BlueprintStrategy::Inferred(strategy) => Ok(infer_blueprint_unit(
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
            )),
        }
    }

    fn select_plane_dim<R: Runtime>(client: &ComputeClient<R>) -> u32 {
        client.properties().hardware.plane_size_min
    }

    fn can_cast_stage_element() -> bool {
        RegisterMatmul::<Filled>::can_cast_stage_element()
    }
}
