use std::fmt::Display;

use cubecl::{Runtime, client::ComputeClient};

use crate::{
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
        stage::{
            ColMajorTilingOrder, FilledStageFamily, PartitionBuffering, PlaneMatmulFamily,
            RowMajorTilingOrder, StridedStageFamily,
        },
        tile::{
            TileMatmulFamily, io::Filled, plane_vec_mat_inner_product::PlaneVecMatInnerProduct,
        },
    },
    definition::{
        CubeCountPlanBlueprint, GlobalOrderBlueprint, HypercubeBlueprint, MatmulElems,
        MatmulProblem, MatmulSetupError, PartitionSize, SmAllocation, TileSize, TilingBlueprint,
        TilingScheme,
    },
    routines::{BlueprintStrategy, DeviceSettings, LaunchInfo, Routine},
};

pub struct SimpleVecMatAlgorithm {}

#[derive(Default, Clone)]
pub struct VecMatStrategy {}

impl Display for VecMatStrategy {
    fn fmt(&self, _f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        Ok(())
    }
}

impl From<()> for VecMatStrategy {
    fn from(_value: ()) -> Self {
        Self {}
    }
}

impl Routine for SimpleVecMatAlgorithm {
    type Strategy = VecMatStrategy;
    type BatchMatmul = PartitionedBatchMatmulFamily<
        SimpleMatmulFamily<
            PlaneMatmulFamily<
                PlaneVecMatInnerProduct<Filled>,
                StridedStageFamily,
                StridedStageFamily,
                FilledStageFamily,
            >,
            SyncFullCyclicLoading<RowMajorTilingOrder>,
            SyncFullCyclicLoading<ColMajorTilingOrder>,
            PlaneWriterFamily,
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
        let blueprint = match strategy {
            BlueprintStrategy::Forced(blueprint) => blueprint.clone(),
            BlueprintStrategy::Inferred(_) => {
                let line_sizes = device_settings.line_sizes;
                let plane_dim = device_settings.plane_dim;

                infer_blueprint_vecmat(
                    &device_settings.client,
                    problem,
                    (1, line_sizes.out as u32, plane_dim * line_sizes.lhs as u32).into(),
                    plane_dim,
                )
            }
        };

        Ok(LaunchInfo {
            blueprint,
            dtypes: MatmulElems::from_globals(&problem.global_dtypes),
        })
    }

    fn can_cast_stage_element() -> bool {
        PlaneVecMatInnerProduct::<Filled>::can_cast_stage_element()
    }
}

pub struct DoubleVecMatAlgorithm {}

impl Routine for DoubleVecMatAlgorithm {
    type Strategy = VecMatStrategy;

    type BatchMatmul = PartitionedBatchMatmulFamily<
        DoubleBufferingMatmulFamily<
            PlaneMatmulFamily<
                PlaneVecMatInnerProduct<Filled>,
                StridedStageFamily,
                StridedStageFamily,
                FilledStageFamily,
            >,
            SyncPartialCyclicLoading<RowMajorTilingOrder>,
            SyncPartialCyclicLoading<ColMajorTilingOrder>,
            PlaneWriterFamily,
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
        let blueprint = match strategy {
            BlueprintStrategy::Forced(blueprint) => blueprint.clone(),
            BlueprintStrategy::Inferred(_) => {
                let line_sizes = device_settings.line_sizes;
                let plane_dim = device_settings.plane_dim;

                infer_blueprint_vecmat(
                    &device_settings.client,
                    problem,
                    (1, line_sizes.out as u32, plane_dim * line_sizes.lhs as u32).into(),
                    plane_dim,
                )
            }
        };

        Ok(LaunchInfo {
            blueprint,
            dtypes: MatmulElems::from_globals(&problem.global_dtypes),
        })
    }

    fn can_cast_stage_element() -> bool {
        PlaneVecMatInnerProduct::<Filled>::can_cast_stage_element()
    }
}

fn infer_blueprint_vecmat<R: Runtime>(
    client: &ComputeClient<R>,
    problem: &MatmulProblem,
    tile_size: TileSize,
    plane_dim: u32,
) -> TilingBlueprint {
    let tiling_scheme = TilingScheme::builder()
        .with_tile_size(tile_size)
        .with_partition_size(PartitionSize::new(1, 1, 1))
        .with_stage_size((1, 1, 1).into())
        .build()
        .unwrap();
    let cube_count_plan = match client.properties().hardware.num_streaming_multiprocessors {
        Some(num_sms) => CubeCountPlanBlueprint::Sm {
            num_sms,
            sm_usage: SmAllocation::Exact,
            cubes_first: true,
        },
        None => CubeCountPlanBlueprint::FromProblem,
    };

    let hypercube = HypercubeBlueprint::builder(&tiling_scheme)
        .global_order(GlobalOrderBlueprint::SwizzleRow {
            m: problem.m as u32,
            w: 2,
        })
        .cube_count_plan(cube_count_plan)
        .build();

    TilingBlueprint::builder(tiling_scheme, plane_dim)
        .partition_buffering(PartitionBuffering::Single)
        .hypercube_config(hypercube)
        .build()
}
