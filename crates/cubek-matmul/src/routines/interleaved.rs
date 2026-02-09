use cubecl::features::MmaConfig;
use cubecl::{Runtime, client::ComputeClient};
use std::fmt::Display;
use std::marker::PhantomData;

use crate::definition::{
    CubeCountStrategy, GlobalOrderStrategy, HypercubeBlueprint, MatmulElems, MatmulLineSizes,
    MatmulProblem, MatmulSetupError, MultiRowStrategy, SmAllocation, TilingBlueprint, TilingScheme,
    adjust_dtypes,
};
use crate::routines::{BlueprintStrategy, DeviceSettings, LaunchInfo};
use crate::{components::batch::BatchMatmulFamily, launch::RuntimeConfig};
use crate::{
    components::tile::{interleaved::InterleavedMatmul, io::Strided},
    routines::ExpandInfo,
};
use crate::{
    components::{
        batch::{PartitionedBatchMatmulFamily, RowMajorGlobalPartitionMatmul},
        global::{
            PlaneWriterFamily,
            read::{FullLoadingStrategy, sync_full_cyclic::SyncFullCyclicLoading},
            single_stage::simple::SimpleMatmulFamily,
        },
        stage::{ColMajorTilingOrder, PartitionBuffering, PlaneMatmulFamily, RowMajorTilingOrder},
        tile::TileMatmulFamily,
    },
    routines::{
        Routine,
        selector::{PlaneTilingBlueprintOptions, infer_blueprint_plane},
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

impl<LL, RL, AL, RC> Routine<RC> for InterleavedAlgorithm<LL, RL, AL>
where
    RC: RuntimeConfig,
    LL: FullLoadingStrategy<RC, TileKind = Strided>,
    RL: FullLoadingStrategy<RC, TileKind = Strided, SyncStrategy = LL::SyncStrategy>,
    AL: FullLoadingStrategy<RC, TileKind = Strided, SyncStrategy = LL::SyncStrategy>,
{
    type Strategy = InterleavedArgs;
    type BatchMatmul = PartitionedBatchMatmulFamily<
        RC,
        SimpleMatmulFamily<
            PlaneMatmulFamily<InterleavedMatmul, LL::Stage, RL::Stage, Option<AL::Stage>>,
            RC,
            LL,
            RL,
            AL,
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

        if InterleavedMatmul::can_cast_stage_element() {
            dtypes.adjust_stage_dtypes();
        }

        let client = &device_settings.client;
        let (blueprint, dtypes) = match strategy {
            BlueprintStrategy::Forced(blueprint) => (blueprint.clone(), dtypes),
            BlueprintStrategy::Inferred(strategy) => {
                if strategy.multi_rows {
                    infer_blueprint_multi_rows::<R, InterleavedMatmul>(
                        client,
                        problem,
                        device_settings.plane_dim,
                        dtypes,
                        &device_settings.line_sizes,
                    )
                } else {
                    infer_blueprint_plane::<InterleavedMatmul, R>(
                        client,
                        problem,
                        device_settings.plane_dim,
                        dtypes,
                        &device_settings.line_sizes,
                        PlaneTilingBlueprintOptions {
                            partition_buffering: Some(PartitionBuffering::Single),
                            tiny_selection_enabled: true,
                            swizzled: InterleavedMatmul::should_swizzle(client),
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
    ) -> Result<LaunchInfo<TilingBlueprint>, MatmulSetupError> {
        let ExpandInfo { blueprint, dtypes } = expand_info;
        let client = &device_settings.client;

        Self::validate_blueprint(
            client,
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

fn infer_blueprint_multi_rows<R: Runtime, TMM: TileMatmulFamily>(
    client: &ComputeClient<R>,
    problem: &MatmulProblem,
    plane_dim: u32,
    mut dtypes: MatmulElems,
    line_sizes: &MatmulLineSizes,
) -> Result<(TilingBlueprint, MatmulElems), MatmulSetupError> {
    adjust_dtypes(client, &mut dtypes, TMM::requires_accelerator());

    let supported = |m: u32, n: u32, k: u32| {
        TMM::is_supported(
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

        let hypercube = HypercubeBlueprint::builder(&tiling_scheme)
            .global_order_strategy(GlobalOrderStrategy::SwizzleRow {
                m: problem.m as u32,
                w: 4,
            })
            .cube_count_strategy(cube_count_strategy)
            .build();

        Ok((
            TilingBlueprint::builder(tiling_scheme, plane_dim, problem)
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
        let hypercube = HypercubeBlueprint::builder(&tiling_scheme)
            .global_order_strategy(GlobalOrderStrategy::SwizzleRow {
                m: problem.m as u32,
                w: 4,
            })
            .cube_count_strategy(cube_count_strategy)
            .build();

        Ok((
            TilingBlueprint::builder(tiling_scheme, plane_dim, problem)
                .partition_buffering(PartitionBuffering::Single)
                .hypercube_blueprint(hypercube)
                .build(),
            dtypes,
        ))
    } else {
        infer_blueprint_plane::<TMM, R>(
            client,
            problem,
            plane_dim,
            dtypes,
            line_sizes,
            PlaneTilingBlueprintOptions {
                partition_buffering: Some(PartitionBuffering::Single),
                multi_row_strategy: MultiRowStrategy::Always(2),
                partition_k: Some(2),
                ..Default::default()
            },
        )
    }
}
