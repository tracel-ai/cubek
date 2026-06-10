use cubecl::{CubeCount, CubeDim, Runtime, client::ComputeClient, ir::AddressType};
use cubek_std::tile::{ColMajorTilingOrder, RowMajorTilingOrder};

use std::{fmt::Display, marker::PhantomData};

use crate::{
    args::{ConfigRuntimeArg, InputRuntimeArg, MatmulArgs, OutputRuntimeArg, RuntimeConfig},
    components::{
        batch::{BatchMatmulFamily, PartitionedBatchMatmulFamily, RowMajorGlobalPartitionMatmul},
        global::{
            UnitWriterFamily,
            read::{FullLoadingStrategy, sync_full_cyclic::SyncFullCyclicLoading},
            single_stage::simple::SimpleMatmulFamily,
        },
        stage::{NumStages, UnitPartitioner},
        tile::TileMatmulKind,
    },
    definition::{
        BatchMatmulBlueprint, CubeMappingLaunch, MatmulElems, MatmulProblem, MatmulSetupError,
        MatmulVectorSizes,
    },
    routines::{
        BlueprintStrategy, DeviceSettings, ExpandInfo, LaunchInfo,
        selector::{
            PartitionScaling, StageScaling, TileSizeSelection, UnitTilingBlueprintOptions,
            infer_blueprint_unit,
        },
    },
};

use crate::routines::{BatchMatmulRoutine, Routine, batch_validate_blueprint};

/// The batch-matmul family powering [`SimpleUnitAlgorithm`].
type SimpleUnitBatch<RC, LL, RL, AL> = PartitionedBatchMatmulFamily<
    RC,
    SimpleMatmulFamily<UnitPartitioner, RC, LL, RL, AL, UnitWriterFamily>,
    RowMajorGlobalPartitionMatmul,
>;

/// Unit single stage matmul with configurable readers (default to cyclic)
pub struct SimpleUnitAlgorithm<
    LL = SyncFullCyclicLoading<ColMajorTilingOrder>,
    RL = SyncFullCyclicLoading<RowMajorTilingOrder>,
    AL = SyncFullCyclicLoading<RowMajorTilingOrder>,
> {
    pub _ll: PhantomData<LL>,
    pub _rl: PhantomData<RL>,
    pub _al: PhantomData<AL>,
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

impl<RC, LL, RL, AL> Routine<RC> for SimpleUnitAlgorithm<LL, RL, AL>
where
    RC: RuntimeConfig,
    LL: FullLoadingStrategy<RC>,
    RL: FullLoadingStrategy<RC, Stage = LL::Stage, SyncStrategy = LL::SyncStrategy>,
    AL: FullLoadingStrategy<RC, SyncStrategy = LL::SyncStrategy>,
{
    type Strategy = SimpleUnitSelectionArgs;
    type Blueprint = BatchMatmulBlueprint;
}

impl<RC, LL, RL, AL> BatchMatmulRoutine<RC> for SimpleUnitAlgorithm<LL, RL, AL>
where
    RC: RuntimeConfig,
    LL: FullLoadingStrategy<RC>,
    RL: FullLoadingStrategy<RC, Stage = LL::Stage, SyncStrategy = LL::SyncStrategy>,
    AL: FullLoadingStrategy<RC, SyncStrategy = LL::SyncStrategy>,
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
                <SimpleUnitBatch<RC, LL, RL, AL>>::launch_unchecked::<MA, R>(
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
        batch_validate_blueprint::<SimpleUnitBatch<RC, LL, RL, AL>, RC, R>(
            client,
            blueprint,
            problem,
            dtypes,
            vector_sizes,
        )
    }

    fn num_stages() -> NumStages {
        SimpleUnitBatch::<RC, LL, RL, AL>::num_stages()
    }

    fn expand_blueprint<R: Runtime>(
        problem: &MatmulProblem,
        device_settings: &DeviceSettings<R>,
        strategy: &BlueprintStrategy<RC, Self>,
    ) -> Result<ExpandInfo<Self::Blueprint>, MatmulSetupError> {
        let mut dtypes = MatmulElems::from_globals(&problem.global_dtypes);
        let tile_matmul = TileMatmulKind::Register;

        if tile_matmul.can_cast_stage_element() {
            dtypes.adjust_stage_dtypes();
        }

        let (blueprint, dtypes) = match strategy {
            BlueprintStrategy::Forced(blueprint) => (blueprint.clone(), dtypes),
            BlueprintStrategy::Inferred(strategy) => infer_blueprint_unit(
                &device_settings.client,
                problem,
                device_settings.plane_dim,
                false,
                &device_settings.vector_sizes,
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
                    swizzle: tile_matmul.should_swizzle(&device_settings.client),
                },
                &problem.global_dtypes,
            ),
        };
        Ok(ExpandInfo { blueprint, dtypes })
    }

    fn prepare<R: Runtime>(
        problem: &MatmulProblem,
        device_settings: &DeviceSettings<R>,
        expand_info: ExpandInfo<Self::Blueprint>,
    ) -> Result<LaunchInfo<Self::Blueprint>, MatmulSetupError> {
        let ExpandInfo { blueprint, dtypes } = expand_info;

        Self::validate_blueprint(
            &device_settings.client,
            &blueprint,
            problem,
            &dtypes,
            &device_settings.vector_sizes,
        )?;

        let cubedim_resource = SimpleUnitBatch::<RC, LL, RL, AL>::cubedim_resource(
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

    fn device_settings<R: Runtime>(
        client: &ComputeClient<R>,
        vector_sizes: MatmulVectorSizes,
    ) -> DeviceSettings<R> {
        let plane_dim = match client.properties().hardware.plane_size_min {
            0 => 32,
            plane_dim => plane_dim,
        };

        DeviceSettings {
            client: client.clone(),
            plane_dim,
            vector_sizes,
            max_cube_count: client.properties().hardware.max_cube_count,
        }
    }
}
