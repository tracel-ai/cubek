use std::fmt::Display;

use cubecl::{Runtime, client::ComputeClient};
use cubek_std::{
    PartitionSize, TileSize,
    cube_count::{CubeCountStrategy, GlobalOrder, HypercubeBlueprint, SmAllocation},
    tile::{Filled, Strided},
};

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
            ColMajorTilingOrder, PartitionBuffering, PlaneMatmulFamily, RowMajorTilingOrder,
            StridedStageFamily,
        },
        tile::{TileMatmulFamily, plane_vec_mat_inner_product::PlaneVecMatInnerProduct},
    },
    definition::{MatmulElems, MatmulProblem, MatmulSetupError, TilingBlueprint, TilingScheme},
    launch::RuntimeConfig,
    routines::{BlueprintStrategy, DeviceSettings, ExpandInfo, LaunchInfo, Routine},
};

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
    type BatchMatmul = PartitionedBatchMatmulFamily<
        RC,
        SimpleMatmulFamily<
            PlaneMatmulFamily<
                PlaneVecMatInnerProduct<Option<Strided>>,
                StridedStageFamily,
                StridedStageFamily,
                Option<StridedStageFamily>,
            >,
            RC,
            SyncFullCyclicLoading<RowMajorTilingOrder>,
            SyncFullCyclicLoading<ColMajorTilingOrder>,
            SyncFullCyclicLoading<ColMajorTilingOrder>,
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

        if PlaneVecMatInnerProduct::<Filled>::can_cast_stage_element() {
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

        <Self as Routine<RC>>::validate_blueprint(
            &device_settings.client,
            &blueprint,
            problem,
            &dtypes,
            &device_settings.vector_sizes,
        )?;

        let cubedim_resource = Self::BatchMatmul::cubedim_resource(
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

    type BatchMatmul = PartitionedBatchMatmulFamily<
        RC,
        DoubleBufferingMatmulFamily<
            PlaneMatmulFamily<
                PlaneVecMatInnerProduct<Option<Strided>>,
                StridedStageFamily,
                StridedStageFamily,
                Option<StridedStageFamily>,
            >,
            RC,
            SyncPartialCyclicLoading<RowMajorTilingOrder>,
            SyncPartialCyclicLoading<ColMajorTilingOrder>,
            SyncFullCyclicLoading<ColMajorTilingOrder>,
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

        if PlaneVecMatInnerProduct::<Filled>::can_cast_stage_element() {
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

        <Self as Routine<RC>>::validate_blueprint(
            &device_settings.client,
            &blueprint,
            problem,
            &dtypes,
            &device_settings.vector_sizes,
        )?;

        let cubedim_resource = Self::BatchMatmul::cubedim_resource(
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
) -> TilingBlueprint {
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

    TilingBlueprint::builder(tiling_scheme, plane_dim, problem)
        .partition_buffering(PartitionBuffering::Single)
        .hypercube_blueprint(hypercube)
        .build()
}
