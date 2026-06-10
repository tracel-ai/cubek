use cubecl::{
    CubeCount, CubeDim,
    features::MmaConfig,
    ir::AddressType,
    {Runtime, client::ComputeClient},
};
use cubek_std::{
    cube_count::{CubeCountStrategy, GlobalOrder, HypercubeBlueprint, SmAllocation},
    tile::{ColMajorTilingOrder, RowMajorTilingOrder},
};
use std::{fmt::Display, marker::PhantomData};

use crate::definition::{
    BatchMatmulBlueprint, CubeMappingLaunch, MatmulElems, MatmulProblem, MatmulSetupError,
    MatmulVectorSizes, MultiRowStrategy, TilingScheme, adjust_dtypes,
};
use crate::{
    components::{
        batch::{PartitionedBatchMatmulFamily, RowMajorGlobalPartitionMatmul},
        global::{
            PlaneWriterFamily,
            read::{FullLoadingStrategy, sync_full_cyclic::SyncFullCyclicLoading},
            single_stage::simple::SimpleMatmulFamily,
        },
        stage::{NumStages, PartitionBuffering, PlanePartitioner},
        tile::TileMatmulKind,
    },
    routines::{
        BatchMatmulRoutine, Routine, batch_validate_blueprint,
        selector::{PlaneTilingBlueprintOptions, infer_blueprint_plane},
    },
};
use crate::{
    routines::ExpandInfo,
    routines::{BlueprintStrategy, DeviceSettings, LaunchInfo},
    {
        args::{ConfigRuntimeArg, InputRuntimeArg, MatmulArgs, OutputRuntimeArg, RuntimeConfig},
        components::batch::BatchMatmulFamily,
    },
};

/// Plane accelerated single stage matmul with configurable readers (default to cyclic)
pub struct InterleavedAlgorithm<
    LL = SyncFullCyclicLoading<ColMajorTilingOrder>,
    RL = SyncFullCyclicLoading<RowMajorTilingOrder>,
    AL = SyncFullCyclicLoading<RowMajorTilingOrder>,
> {
    pub _ll: PhantomData<LL>,
    pub _rl: PhantomData<RL>,
    pub _al: PhantomData<AL>,
}

#[derive(Default, Debug, Clone)]
pub struct InterleavedArgs {
    // Uses an optimized multi rows strategy.
    pub multi_rows: bool,
}

impl Display for InterleavedArgs {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(if self.multi_rows { "_multi_rows" } else { "" })
    }
}

/// The batch-matmul family powering [`InterleavedAlgorithm`].
type InterleavedBatch<RC, LL, RL, AL> = PartitionedBatchMatmulFamily<
    RC,
    SimpleMatmulFamily<PlanePartitioner, RC, LL, RL, AL, PlaneWriterFamily>,
    RowMajorGlobalPartitionMatmul,
>;

impl<LL, RL, AL, RC> Routine<RC> for InterleavedAlgorithm<LL, RL, AL>
where
    RC: RuntimeConfig,
    LL: FullLoadingStrategy<RC>,
    RL: FullLoadingStrategy<RC, SyncStrategy = LL::SyncStrategy>,
    AL: FullLoadingStrategy<RC, SyncStrategy = LL::SyncStrategy>,
{
    type Strategy = InterleavedArgs;
    type Blueprint = BatchMatmulBlueprint;
}

impl<LL, RL, AL, RC> BatchMatmulRoutine<RC> for InterleavedAlgorithm<LL, RL, AL>
where
    RC: RuntimeConfig,
    LL: FullLoadingStrategy<RC>,
    RL: FullLoadingStrategy<RC, SyncStrategy = LL::SyncStrategy>,
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
                <InterleavedBatch<RC, LL, RL, AL>>::launch_unchecked::<MA, R>(
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
        batch_validate_blueprint::<InterleavedBatch<RC, LL, RL, AL>, RC, R>(
            client,
            blueprint,
            problem,
            dtypes,
            vector_sizes,
        )
    }

    fn num_stages() -> NumStages {
        InterleavedBatch::<RC, LL, RL, AL>::num_stages()
    }

    fn expand_blueprint<R: Runtime>(
        problem: &MatmulProblem,
        device_settings: &DeviceSettings<R>,
        strategy: &BlueprintStrategy<RC, Self>,
    ) -> Result<ExpandInfo<Self::Blueprint>, MatmulSetupError> {
        let mut dtypes = MatmulElems::from_globals(&problem.global_dtypes);
        let tile_matmul = TileMatmulKind::Interleaved;

        if tile_matmul.can_cast_stage_element() {
            dtypes.adjust_stage_dtypes();
        }

        let client = &device_settings.client;
        let (blueprint, dtypes) = match strategy {
            BlueprintStrategy::Forced(blueprint) => (blueprint.clone(), dtypes),
            BlueprintStrategy::Inferred(strategy) => {
                if strategy.multi_rows {
                    infer_blueprint_multi_rows::<R>(
                        tile_matmul,
                        client,
                        problem,
                        device_settings.plane_dim,
                        dtypes,
                        &device_settings.vector_sizes,
                    )
                } else {
                    infer_blueprint_plane::<R>(
                        tile_matmul,
                        client,
                        problem,
                        device_settings.plane_dim,
                        dtypes,
                        &device_settings.vector_sizes,
                        PlaneTilingBlueprintOptions {
                            partition_buffering: Some(PartitionBuffering::Single),
                            tiny_selection_enabled: true,
                            swizzled: tile_matmul.should_swizzle(client),
                            ..Default::default()
                        },
                    )
                }?
            }
        };
        Ok(ExpandInfo { blueprint, dtypes })
    }

    fn prepare<R: Runtime>(
        problem: &MatmulProblem,
        device_settings: &DeviceSettings<R>,
        expand_info: ExpandInfo<Self::Blueprint>,
    ) -> Result<LaunchInfo<BatchMatmulBlueprint>, MatmulSetupError> {
        let ExpandInfo { blueprint, dtypes } = expand_info;
        let client = &device_settings.client;

        Self::validate_blueprint(
            client,
            &blueprint,
            problem,
            &dtypes,
            &device_settings.vector_sizes,
        )?;

        let cubedim_resource = InterleavedBatch::<RC, LL, RL, AL>::cubedim_resource(
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

fn infer_blueprint_multi_rows<R: Runtime>(
    tile_matmul: TileMatmulKind,
    client: &ComputeClient<R>,
    problem: &MatmulProblem,
    plane_dim: u32,
    mut dtypes: MatmulElems,
    vector_sizes: &MatmulVectorSizes,
) -> Result<(BatchMatmulBlueprint, MatmulElems), MatmulSetupError> {
    adjust_dtypes(client, &mut dtypes, tile_matmul.requires_accelerator());

    let supported = |m: u32, n: u32, k: u32| {
        tile_matmul.is_supported(
            client,
            MmaConfig {
                a_type: dtypes.lhs_register,
                b_type: dtypes.rhs_register,
                cd_type: dtypes.acc_register,
                m,
                n,
                k,
            },
        )
    };
    let cube_count_strategy = match client.properties().hardware.num_streaming_multiprocessors {
        Some(num_sms) => CubeCountStrategy::Sm {
            num_sms,
            sm_usage: SmAllocation::Exact,
            cubes_first: true,
        },
        None => CubeCountStrategy::Flattened,
    };

    if supported(8, 32, 16) {
        // A lot of multi-rows balanced with a
        // tile size of (8, 32, 16)
        let tiling_scheme = TilingScheme::builder()
            .with_tile_size((8, 32, 16).into())
            .with_partition_size((4, 4, 2).into())
            .with_stage_size((4, 1, 1).into())
            .build()
            .unwrap();

        let hypercube = HypercubeBlueprint::builder()
            .global_order(GlobalOrder::SwizzleRow(4))
            .cube_count_strategy(cube_count_strategy)
            .build();

        Ok((
            BatchMatmulBlueprint::builder(
                TileMatmulKind::Interleaved,
                tiling_scheme,
                plane_dim,
                problem,
            )
            .partition_buffering(PartitionBuffering::Single)
            .hypercube_blueprint(hypercube)
            .build(),
            dtypes,
        ))
    } else if supported(8, 8, 8) {
        let tiling_scheme = TilingScheme::builder()
            .with_tile_size((8, 8, 8).into())
            .with_partition_size((4, 8, 2).into())
            .with_stage_size((4, 1, 1).into())
            .build()
            .unwrap();
        let hypercube = HypercubeBlueprint::builder()
            .global_order(GlobalOrder::SwizzleRow(4))
            .cube_count_strategy(cube_count_strategy)
            .build();

        Ok((
            BatchMatmulBlueprint::builder(
                TileMatmulKind::Interleaved,
                tiling_scheme,
                plane_dim,
                problem,
            )
            .partition_buffering(PartitionBuffering::Single)
            .hypercube_blueprint(hypercube)
            .build(),
            dtypes,
        ))
    } else {
        infer_blueprint_plane::<R>(
            tile_matmul,
            client,
            problem,
            plane_dim,
            dtypes,
            vector_sizes,
            PlaneTilingBlueprintOptions {
                partition_buffering: Some(PartitionBuffering::Single),
                multi_row_strategy: MultiRowStrategy::Always(2),
                partition_k: Some(2),
                ..Default::default()
            },
        )
    }
}
