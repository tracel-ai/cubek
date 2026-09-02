use std::fmt::Display;

use cubecl::{CubeCount, CubeDim, client::ComputeClient, ir::AddressType};

use crate::{
    definition::{MatmulElems, MatmulProblem, MatmulSetupError, MatmulVectorSizes},
    multi_level::{
        BatchMatmulRoutine, ExpandInfo, LaunchInfo,
        args::{ConfigRuntimeArg, InputRuntimeArg, MatmulArgs, OutputRuntimeArg, RuntimeConfig},
        batch_validate_blueprint,
        components::{
            batch::{
                BatchMatmulFamily, PartitionedBatchMatmulFamily, RowMajorGlobalPartitionMatmul,
            },
            global::{
                UnitWriterFamily,
                multi_stage::double_buffering::DoubleBufferingMatmulFamily,
                read::{
                    sync_full_cyclic::SyncFullCyclicLoading,
                    sync_partial_cyclic::SyncPartialCyclicLoading,
                },
            },
            stage::{NumStages, UnitPartitioner},
            tile::TileMatmulKind,
        },
        definition::{BatchMatmulBlueprint, CubeMappingLaunch},
        routines::selector::{TileSizeSelection, UnitTilingBlueprintOptions, infer_blueprint_unit},
        tile::RowMajorTilingOrder,
    },
    routine::{BlueprintStrategy, DeviceSettings, Routine},
};

/// The batch-matmul family powering [`DoubleUnitAlgorithm`].
type DoubleUnitBatch<RC> = PartitionedBatchMatmulFamily<
    RC,
    DoubleBufferingMatmulFamily<
        UnitPartitioner,
        RC,
        SyncPartialCyclicLoading<RowMajorTilingOrder>,
        SyncPartialCyclicLoading<RowMajorTilingOrder>,
        SyncFullCyclicLoading<RowMajorTilingOrder>,
        UnitWriterFamily,
    >,
    RowMajorGlobalPartitionMatmul,
>;

/// Unit double buffered matmul with cyclic readers
pub struct DoubleUnitAlgorithm {}

#[derive(Default, Clone, Debug)]
pub struct DoubleUnitSelectionArgs {
    pub tile_size: TileSizeSelection,
}

impl Display for DoubleUnitSelectionArgs {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "_{}", self.tile_size)
    }
}

impl<RC: RuntimeConfig> Routine<RC> for DoubleUnitAlgorithm {
    type Strategy = DoubleUnitSelectionArgs;
    type Blueprint = BatchMatmulBlueprint;
}

impl<RC: RuntimeConfig> BatchMatmulRoutine<RC> for DoubleUnitAlgorithm {
    #[allow(clippy::too_many_arguments, clippy::result_large_err)]
    fn launch<MA: MatmulArgs<Config = RC>>(
        client: &ComputeClient,
        cube_dim: CubeDim,
        cube_count: CubeCount,
        address_type: AddressType,
        input: InputRuntimeArg<MA>,
        output: OutputRuntimeArg<MA>,
        config: ConfigRuntimeArg<MA>,
        cube_count_input: CubeMappingLaunch,
        blueprint: Self::Blueprint,
        dtypes: &MatmulElems,
        vector_sizes: &MatmulVectorSizes,
    ) -> Result<(), MatmulSetupError> {
        {
            unsafe {
                <DoubleUnitBatch<RC>>::launch_unchecked::<MA>(
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
    fn validate_blueprint(
        client: &ComputeClient,
        blueprint: &Self::Blueprint,
        problem: &MatmulProblem,
        dtypes: &MatmulElems,
        vector_sizes: &MatmulVectorSizes,
    ) -> Result<(), MatmulSetupError> {
        batch_validate_blueprint::<DoubleUnitBatch<RC>, RC>(
            client,
            blueprint,
            problem,
            dtypes,
            vector_sizes,
        )
    }

    fn num_stages() -> NumStages {
        DoubleUnitBatch::<RC>::num_stages()
    }

    fn expand_blueprint(
        problem: &MatmulProblem,
        device_settings: &DeviceSettings,
        strategy: &BlueprintStrategy<RC, Self>,
    ) -> Result<ExpandInfo<Self::Blueprint>, MatmulSetupError> {
        let mut dtypes = MatmulElems::from_globals(&problem.global_dtypes);

        if TileMatmulKind::Register.can_cast_stage_element() {
            dtypes.adjust_stage_dtypes();
        }

        let (blueprint, dtypes) = match strategy {
            BlueprintStrategy::Forced(blueprint) => (blueprint.clone(), dtypes),
            BlueprintStrategy::Inferred(strategy) => infer_blueprint_unit(
                &device_settings.client,
                problem,
                device_settings.plane_dim,
                true,
                &device_settings.vector_sizes,
                UnitTilingBlueprintOptions {
                    tile: strategy.tile_size,
                    ..Default::default()
                },
                &problem.global_dtypes,
            ),
        };
        Ok(ExpandInfo { blueprint, dtypes })
    }

    fn prepare(
        problem: &MatmulProblem,
        device_settings: &DeviceSettings,
        expand_info: ExpandInfo<Self::Blueprint>,
    ) -> Result<LaunchInfo<BatchMatmulBlueprint>, MatmulSetupError> {
        let ExpandInfo { blueprint, dtypes } = expand_info;

        <Self as BatchMatmulRoutine<RC>>::validate_blueprint(
            &device_settings.client,
            &blueprint,
            problem,
            &dtypes,
            &device_settings.vector_sizes,
        )?;

        let cubedim_resource = DoubleUnitBatch::<RC>::cubedim_resource(
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

    fn device_settings(client: &ComputeClient, vector_sizes: MatmulVectorSizes) -> DeviceSettings {
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
