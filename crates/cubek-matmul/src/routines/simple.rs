use cubecl::features::MmaConfig;
use cubecl::{Runtime, client::ComputeClient};
use std::marker::PhantomData;

use crate::components::batch::BatchMatmulFamily;
use crate::definition::{
    CubeCountPlanBlueprint, GlobalOrderBlueprint, HypercubeBlueprint, MatmulElems, MatmulLineSizes,
    MatmulProblem, MatmulSetupError, MultiRowStrategy, SmAllocation, TilingBlueprint, TilingScheme,
    adjust_dtypes,
};
use crate::{
    components::{
        batch::{PartitionedBatchMatmulFamily, RowMajorGlobalPartitionMatmul},
        global::{
            PlaneWriterFamily,
            read::{
                FullLoadingStrategy, async_full_tma::AsyncFullTmaLoading,
                sync_full_cyclic::SyncFullCyclicLoading,
            },
            single_stage::simple::SimpleMatmulFamily,
        },
        stage::{
            ColMajorTilingOrder, FilledStageFamily, PartitionBuffering, PlaneMatmulFamily,
            RowMajorTilingOrder, StridedStageFamily,
        },
        tile::{
            TileMatmulFamily,
            io::{Filled, Strided},
        },
    },
    routines::{
        Routine,
        selector::{PlaneTilingBlueprintOptions, infer_blueprint_plane},
    },
};

/// Plane accelerated single stage matmul with configurable readers (default to cyclic)
pub struct SimpleAlgorithm<
    TMM,
    LL = SyncFullCyclicLoading<ColMajorTilingOrder>,
    RL = SyncFullCyclicLoading<RowMajorTilingOrder>,
> {
    pub _tmm: PhantomData<TMM>,
    pub _ll: PhantomData<LL>,
    pub _rl: PhantomData<RL>,
}

pub type SimpleTmaAlgorithm<TMM> = SimpleAlgorithm<TMM, AsyncFullTmaLoading, AsyncFullTmaLoading>;
pub type SimpleBarrierAlgorithm<TMM, L> = SimpleAlgorithm<TMM, L, L>;

#[derive(Default, Debug, Clone)]
pub struct SimpleArgs {
    // Uses an optimized multi rows strategy.
    pub multi_rows: bool,
}

impl<TMM, LL, RL> Routine for SimpleAlgorithm<TMM, LL, RL>
where
    TMM:
        TileMatmulFamily<LhsTile = Strided, RhsTile = Strided, AccTile = Filled, OutTile = Strided>,
    LL: FullLoadingStrategy,
    RL: FullLoadingStrategy<SyncStrategy = LL::SyncStrategy>,
{
    type Strategy = SimpleArgs;
    type BatchMatmul = PartitionedBatchMatmulFamily<
        SimpleMatmulFamily<
            PlaneMatmulFamily<TMM, StridedStageFamily, StridedStageFamily, FilledStageFamily>,
            LL,
            RL,
            PlaneWriterFamily,
        >,
        RowMajorGlobalPartitionMatmul,
    >;
    type Blueprint = TilingBlueprint;
    type Config = <Self::BatchMatmul as BatchMatmulFamily>::Config;

    fn prepare<R: Runtime>(
        client: &ComputeClient<R>,
        problem: &MatmulProblem,
        plane_dim: u32,
        line_sizes: &MatmulLineSizes,
        args: &Self::Strategy,
        dtypes: &mut MatmulElems,
    ) -> Result<TilingBlueprint, MatmulSetupError> {
        if args.multi_rows {
            infer_blueprint_multi_rows::<R, TMM>(client, problem, plane_dim, dtypes, line_sizes)
        } else {
            infer_blueprint_plane::<TMM, R>(
                client,
                problem,
                plane_dim,
                dtypes,
                line_sizes,
                PlaneTilingBlueprintOptions {
                    partition_buffering: Some(PartitionBuffering::Single),
                    tiny_selection_enabled: true,
                    swizzled: TMM::should_swizzle(client),
                    ..Default::default()
                },
            )
        }
    }

    fn can_cast_stage_element() -> bool {
        TMM::can_cast_stage_element()
    }
}

fn infer_blueprint_multi_rows<R: Runtime, TMM: TileMatmulFamily>(
    client: &ComputeClient<R>,
    problem: &MatmulProblem,
    plane_dim: u32,
    dtypes: &mut MatmulElems,
    line_sizes: &MatmulLineSizes,
) -> Result<TilingBlueprint, MatmulSetupError> {
    adjust_dtypes(client, dtypes, TMM::requires_accelerator());

    let supported = |m: u32, n: u32, k: u32| {
        TMM::is_supported(
            client,
            MmaConfig {
                a_type: *dtypes.lhs_register,
                b_type: *dtypes.rhs_register,
                cd_type: *dtypes.acc_register,
                m,
                n,
                k,
            },
        )
    };
    let cube_count_plan = match client.properties().hardware.num_streaming_multiprocessors {
        Some(num_sms) => CubeCountPlanBlueprint::Sm {
            num_sms,
            sm_usage: SmAllocation::Exact,
            cubes_first: true,
        },
        None => CubeCountPlanBlueprint::Flattened,
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
            .global_order(GlobalOrderBlueprint::SwizzleRow {
                m: problem.m as u32,
                w: 4,
            })
            .cube_count_plan(cube_count_plan)
            .build();

        Ok(TilingBlueprint::builder(tiling_scheme, plane_dim)
            .partition_buffering(PartitionBuffering::Single)
            .hypercube_config(hypercube)
            .build())
    } else if supported(8, 8, 8) {
        let tiling_scheme = TilingScheme::builder()
            .with_tile_size((8, 8, 8).into())
            .with_partition_size((4, 8, 2).into())
            .with_stage_size((4, 1, 1).into())
            .build()
            .unwrap();
        let hypercube = HypercubeBlueprint::builder(&tiling_scheme)
            .global_order(GlobalOrderBlueprint::SwizzleRow {
                m: problem.m as u32,
                w: 4,
            })
            .cube_count_plan(cube_count_plan)
            .build();

        Ok(TilingBlueprint::builder(tiling_scheme, plane_dim)
            .partition_buffering(PartitionBuffering::Single)
            .hypercube_config(hypercube)
            .build())
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
