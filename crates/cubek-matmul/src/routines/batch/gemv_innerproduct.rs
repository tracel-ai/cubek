use std::fmt::Display;

use cubecl::{CubeCount, CubeDim, Runtime, client::ComputeClient, ir::AddressType};
use cubek_std::{
    PartitionSize, TileSize,
    cube_count::{CubeCountStrategy, GlobalOrder, HypercubeBlueprint, SmAllocation},
    tile::{ColMajorTilingOrder, RowMajorTilingOrder},
};

use crate::{
    args::{ConfigRuntimeArg, InputRuntimeArg, MatmulArgs, OutputRuntimeArg, RuntimeConfig},
    components::{
        batch::{BatchMatmulFamily, PartitionedBatchMatmulFamily, RowMajorGlobalPartitionMatmul},
        global::{
            PlaneWriterFamily,
            multi_stage::double_buffering::DoubleBufferingMatmulFamily,
            read::{
                sync_full_cyclic::SyncFullCyclicLoading,
                sync_partial_cyclic::SyncPartialCyclicLoading,
            },
            single_stage::simple::SimpleMatmulFamily,
        },
        stage::{NumStages, PartitionBuffering, PlanePartitioner},
        tile::TileMatmulKind,
    },
    definition::{
        BatchMatmulBlueprint, CubeMappingLaunch, MatmulElems, MatmulProblem, MatmulSetupError,
        MatmulVectorSizes, TilingScheme,
    },
    routines::{
        BatchMatmulRoutine, BlueprintStrategy, DeviceSettings, ExpandInfo, LaunchInfo, Routine,
        batch_validate_blueprint,
    },
};

/// The batch-matmul family powering [`VecMatInnerProductAlgorithm`].
type VecMatBatch<RC> = PartitionedBatchMatmulFamily<
    RC,
    SimpleMatmulFamily<
        PlanePartitioner,
        RC,
        SyncFullCyclicLoading<RowMajorTilingOrder>,
        SyncFullCyclicLoading<ColMajorTilingOrder>,
        SyncFullCyclicLoading<ColMajorTilingOrder>,
        PlaneWriterFamily,
    >,
    RowMajorGlobalPartitionMatmul,
>;

/// The batch-matmul family powering [`DoubleVecMatInnerProductAlgorithm`].
type DoubleVecMatBatch<RC> = PartitionedBatchMatmulFamily<
    RC,
    DoubleBufferingMatmulFamily<
        PlanePartitioner,
        RC,
        SyncPartialCyclicLoading<RowMajorTilingOrder>,
        SyncPartialCyclicLoading<ColMajorTilingOrder>,
        SyncFullCyclicLoading<ColMajorTilingOrder>,
        PlaneWriterFamily,
    >,
    RowMajorGlobalPartitionMatmul,
>;

pub struct VecMatInnerProductAlgorithm {}

#[derive(Default, Clone)]
pub struct VecMatInnerProductStrategy {}

impl Display for VecMatInnerProductStrategy {
    fn fmt(&self, _f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        Ok(())
    }
}

impl From<()> for VecMatInnerProductStrategy {
    fn from(_value: ()) -> Self {
        Self {}
    }
}

impl<RC: RuntimeConfig> Routine<RC> for VecMatInnerProductAlgorithm {
    type Strategy = VecMatInnerProductStrategy;
    type Blueprint = BatchMatmulBlueprint;
}

impl<RC: RuntimeConfig> BatchMatmulRoutine<RC> for VecMatInnerProductAlgorithm {
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
                <VecMatBatch<RC>>::launch_unchecked::<MA, R>(
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
        batch_validate_blueprint::<VecMatBatch<RC>, RC, R>(
            client,
            blueprint,
            problem,
            dtypes,
            vector_sizes,
        )
    }

    fn num_stages() -> NumStages {
        VecMatBatch::<RC>::num_stages()
    }

    fn expand_blueprint<R: Runtime>(
        problem: &MatmulProblem,
        device_settings: &DeviceSettings<R>,
        strategy: &BlueprintStrategy<RC, Self>,
    ) -> Result<ExpandInfo<Self::Blueprint>, MatmulSetupError> {
        let mut dtypes = MatmulElems::from_globals(&problem.global_dtypes);

        if TileMatmulKind::PlaneVec.can_cast_stage_element() {
            dtypes.adjust_stage_dtypes();
        }

        let blueprint = match strategy {
            BlueprintStrategy::Forced(blueprint) => blueprint.clone(),
            BlueprintStrategy::Inferred(_) => {
                let vector_sizes = device_settings.vector_sizes;
                let plane_dim = device_settings.plane_dim;

                infer_blueprint_vecmat(
                    &device_settings.client,
                    problem,
                    (
                        1,
                        vector_sizes.out as u32,
                        plane_dim * vector_sizes.lhs as u32,
                    )
                        .into(),
                    plane_dim,
                )
            }
        };
        Ok(ExpandInfo { blueprint, dtypes })
    }

    fn prepare<R: Runtime>(
        problem: &MatmulProblem,
        device_settings: &DeviceSettings<R>,
        expand_info: ExpandInfo<Self::Blueprint>,
    ) -> Result<LaunchInfo<Self::Blueprint>, MatmulSetupError> {
        let ExpandInfo { blueprint, dtypes } = expand_info;

        <Self as BatchMatmulRoutine<RC>>::validate_blueprint(
            &device_settings.client,
            &blueprint,
            problem,
            &dtypes,
            &device_settings.vector_sizes,
        )?;

        let cubedim_resource = VecMatBatch::<RC>::cubedim_resource(
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

pub struct DoubleVecMatInnerProductAlgorithm {}

impl<RC: RuntimeConfig> Routine<RC> for DoubleVecMatInnerProductAlgorithm {
    type Strategy = VecMatInnerProductStrategy;
    type Blueprint = BatchMatmulBlueprint;
}

impl<RC: RuntimeConfig> BatchMatmulRoutine<RC> for DoubleVecMatInnerProductAlgorithm {
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
                <DoubleVecMatBatch<RC>>::launch_unchecked::<MA, R>(
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
        batch_validate_blueprint::<DoubleVecMatBatch<RC>, RC, R>(
            client,
            blueprint,
            problem,
            dtypes,
            vector_sizes,
        )
    }

    fn num_stages() -> NumStages {
        DoubleVecMatBatch::<RC>::num_stages()
    }

    fn expand_blueprint<R: Runtime>(
        problem: &MatmulProblem,
        device_settings: &DeviceSettings<R>,
        strategy: &BlueprintStrategy<RC, Self>,
    ) -> Result<ExpandInfo<Self::Blueprint>, MatmulSetupError> {
        let mut dtypes = MatmulElems::from_globals(&problem.global_dtypes);

        if TileMatmulKind::PlaneVec.can_cast_stage_element() {
            dtypes.adjust_stage_dtypes();
        }

        let blueprint = match strategy {
            BlueprintStrategy::Forced(blueprint) => blueprint.clone(),
            BlueprintStrategy::Inferred(_) => {
                let vector_sizes = device_settings.vector_sizes;
                let plane_dim = device_settings.plane_dim;

                infer_blueprint_vecmat(
                    &device_settings.client,
                    problem,
                    (
                        1,
                        vector_sizes.out as u32,
                        plane_dim * vector_sizes.lhs as u32,
                    )
                        .into(),
                    plane_dim,
                )
            }
        };
        Ok(ExpandInfo { blueprint, dtypes })
    }

    fn prepare<R: Runtime>(
        problem: &MatmulProblem,
        device_settings: &DeviceSettings<R>,
        expand_info: ExpandInfo<Self::Blueprint>,
    ) -> Result<LaunchInfo<Self::Blueprint>, MatmulSetupError> {
        let ExpandInfo { blueprint, dtypes } = expand_info;

        <Self as BatchMatmulRoutine<RC>>::validate_blueprint(
            &device_settings.client,
            &blueprint,
            problem,
            &dtypes,
            &device_settings.vector_sizes,
        )?;

        let cubedim_resource = DoubleVecMatBatch::<RC>::cubedim_resource(
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

fn infer_blueprint_vecmat<R: Runtime>(
    client: &ComputeClient<R>,
    problem: &MatmulProblem,
    tile_size: TileSize,
    plane_dim: u32,
) -> BatchMatmulBlueprint {
    let tiling_scheme = TilingScheme::builder()
        .with_tile_size(tile_size)
        .with_partition_size(PartitionSize::new(1, 1, 1))
        .with_stage_size((1, 1, 1).into())
        .build()
        .unwrap();
    let cube_count_strategy = match client.properties().hardware.num_streaming_multiprocessors {
        Some(num_sms) => CubeCountStrategy::Sm {
            num_sms,
            sm_usage: SmAllocation::Exact,
            cubes_first: true,
        },
        None => CubeCountStrategy::FromProblem,
    };

    let hypercube = HypercubeBlueprint::builder()
        .global_order(GlobalOrder::SwizzleRow(2))
        .cube_count_strategy(cube_count_strategy)
        .build();

    BatchMatmulBlueprint::builder(TileMatmulKind::PlaneVec, tiling_scheme, plane_dim, problem)
        .partition_buffering(PartitionBuffering::Single)
        .hypercube_blueprint(hypercube)
        .build()
}
