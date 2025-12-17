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
        CubeCountPlanSelection, GlobalOrderSelection, HypercubeSelection, MatmulElems,
        MatmulLineSizes, MatmulProblem, MatmulSelection, MatmulSetupError, PartitionSize,
        SmAllocation, TileSize, TilingScheme,
    },
    routines::Routine,
};

pub struct SimpleVecMatAlgorithm {}

impl Routine for SimpleVecMatAlgorithm {
    type Strategy = ();
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
    type Blueprint = MatmulSelection;
    type Config = <Self::BatchMatmul as BatchMatmulFamily>::Config;

    fn prepare<R: Runtime>(
        client: &ComputeClient<R>,
        problem: &MatmulProblem,
        plane_dim: u32,
        line_sizes: &MatmulLineSizes,
        _args: &Self::Strategy,
        _dtypes: &mut MatmulElems,
    ) -> Result<MatmulSelection, MatmulSetupError> {
        Ok(selection_vecmat(
            client,
            problem,
            (1, line_sizes.out as u32, plane_dim * line_sizes.lhs as u32).into(),
            plane_dim,
        ))
    }

    fn can_cast_stage_element() -> bool {
        PlaneVecMatInnerProduct::<Filled>::can_cast_stage_element()
    }
}

pub struct DoubleVecMatAlgorithm {}

impl Routine for DoubleVecMatAlgorithm {
    type Strategy = ();

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
    type Blueprint = MatmulSelection;
    type Config = <Self::BatchMatmul as BatchMatmulFamily>::Config;

    fn prepare<R: Runtime>(
        client: &ComputeClient<R>,
        problem: &MatmulProblem,
        plane_dim: u32,
        line_sizes: &MatmulLineSizes,
        _args: &Self::Strategy,
        _dtypes: &mut MatmulElems,
    ) -> Result<MatmulSelection, MatmulSetupError> {
        Ok(selection_vecmat(
            client,
            problem,
            (1, line_sizes.out as u32, plane_dim * line_sizes.lhs as u32).into(),
            plane_dim,
        ))
    }

    fn can_cast_stage_element() -> bool {
        PlaneVecMatInnerProduct::<Filled>::can_cast_stage_element()
    }
}

fn selection_vecmat<R: Runtime>(
    client: &ComputeClient<R>,
    problem: &MatmulProblem,
    tile_size: TileSize,
    plane_dim: u32,
) -> MatmulSelection {
    let tiling_scheme = TilingScheme::builder()
        .with_tile_size(tile_size)
        .with_partition_size(PartitionSize::new(1, 1, 1))
        .with_stage_size((1, 1, 1).into())
        .build()
        .unwrap();
    let cube_count_plan = match client.properties().hardware.num_streaming_multiprocessors {
        Some(num_sms) => CubeCountPlanSelection::Sm {
            num_sms,
            sm_usage: SmAllocation::Exact,
            cubes_first: true,
        },
        None => CubeCountPlanSelection::FromProblem,
    };

    let hypercube = HypercubeSelection::builder(&tiling_scheme)
        .global_order(GlobalOrderSelection::SwizzleRow {
            m: problem.m as u32,
            w: 2,
        })
        .cube_count_plan(cube_count_plan)
        .build();

    MatmulSelection::builder(tiling_scheme, plane_dim)
        .partition_buffering(PartitionBuffering::Single)
        .hypercube_config(hypercube)
        .build()
}
