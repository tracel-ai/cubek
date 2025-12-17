use std::marker::PhantomData;

use cubecl::Runtime;
use cubecl::client::ComputeClient;
use cubecl::features::MmaConfig;

use crate::components::batch::BatchMatmulFamily;
use crate::components::stage::PlaneMatmulFamily;
use crate::components::tile;
use crate::components::{
    batch::{PartitionedBatchMatmulFamily, RowMajorGlobalPartitionMatmul},
    tile::io::{Filled, Strided},
};
use crate::components::{global::PlaneWriterFamily, stage::StageFamily};
use crate::components::{stage::FilledStageFamily, tile::TileMatmulFamily};
use crate::definition::{
    CubeCountPlanBlueprint, GlobalOrderBlueprint, HypercubeBlueprint, MatmulLineSizes,
    MatmulProblem, MatmulSetupError, MatrixLayout, SmAllocation, SwizzleBlueprint, TilingBlueprint,
    adjust_dtypes,
};
use crate::routines::base;
use crate::routines::selector::{PlaneTilingBlueprintOptions, infer_blueprint_plane};
use crate::{
    components::global::{
        multi_stage::specialized::SpecializedMatmulFamily,
        read::{AsyncPartialLoadingStrategy, async_partial_tma::AsyncPartialTmaLoading},
    },
    definition::{MatmulElems, MultiRowStrategy, TilingScheme},
};
use crate::{
    components::{
        global::{LoadSpecializationConfig, SpecializationTensorConfig},
        stage::PartitionBuffering,
    },
    routines::selector::select_swizzle,
};

/// Plane accelerated specialized matmul with TMA readers
pub struct SpecializedAlgorithm<TMM, L = AsyncPartialTmaLoading> {
    pub _phantom: PhantomData<(TMM, L)>,
}

impl<TMM, L> base::Routine for SpecializedAlgorithm<TMM, L>
where
    TMM: tile::TileMatmulFamily<
            LhsTile = <L::Stage as StageFamily>::TileKind,
            RhsTile = <L::Stage as StageFamily>::TileKind,
            AccTile = Filled,
            OutTile = Strided,
        >,
    L: AsyncPartialLoadingStrategy,
{
    type Strategy = ();
    type BatchMatmul = PartitionedBatchMatmulFamily<
        SpecializedMatmulFamily<
            PlaneMatmulFamily<TMM, L::Stage, L::Stage, FilledStageFamily>,
            L,
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
        _args: &Self::Strategy,
        dtypes: &mut MatmulElems,
    ) -> Result<TilingBlueprint, MatmulSetupError> {
        infer_blueprint_plane::<TMM, R>(
            client,
            problem,
            plane_dim,
            dtypes,
            line_sizes,
            PlaneTilingBlueprintOptions {
                specialized: true,
                multi_row_strategy: MultiRowStrategy::Adaptive {
                    minimum_stage_count: 8,
                },
                swizzled: TMM::should_swizzle(client),
                ..Default::default()
            },
        )
    }

    fn can_cast_stage_element() -> bool {
        TMM::can_cast_stage_element()
    }
}

#[allow(unused, reason = "needs more tuning")]
fn infer_blueprint_specialized<R: Runtime, TMM: TileMatmulFamily>(
    client: &ComputeClient<R>,
    problem: &MatmulProblem,
    plane_dim: u32,
    swizzle: bool,
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

    let tiling_scheme = if supported(16, 8, 16) {
        TilingScheme::builder()
            .with_tile_size((16, 8, 16).into())
            .with_partition_size((1, 8, 2).into())
            .with_stage_size((4, 1, 1).into())
            .build()
            .unwrap()
    } else if supported(16, 16, 16) {
        TilingScheme::builder()
            .with_tile_size((16, 16, 16).into())
            .with_partition_size((1, 4, 2).into())
            .with_stage_size((4, 1, 1).into())
            .build()
            .unwrap()
    } else {
        return infer_blueprint_plane::<TMM, R>(
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
        );
    };

    let hypercube = HypercubeBlueprint::builder(&tiling_scheme)
        .global_order(GlobalOrderBlueprint::SwizzleRow {
            m: problem.m as u32,
            w: 4,
        })
        .cube_count_plan(cube_count_plan)
        .build();

    let mut builder = TilingBlueprint::builder(tiling_scheme, plane_dim)
        .partition_buffering(PartitionBuffering::Single)
        .hypercube_config(hypercube)
        .load_specialization_config(LoadSpecializationConfig {
            lhs: SpecializationTensorConfig::LoadFlowOnly,
            rhs: SpecializationTensorConfig::LoadFlowOnly,
        });

    if swizzle {
        let lhs_swizzle_dim = match problem.lhs_layout {
            MatrixLayout::RowMajor => tiling_scheme.elements_per_stage_along_k(),
            MatrixLayout::ColMajor => tiling_scheme.elements_per_stage_along_m(),
        };
        let rhs_swizzle_dim = match problem.rhs_layout {
            MatrixLayout::RowMajor => tiling_scheme.elements_per_stage_along_n(),
            MatrixLayout::ColMajor => tiling_scheme.elements_per_stage_along_k(),
        };

        let lhs = select_swizzle(lhs_swizzle_dim, *dtypes.lhs_stage, line_sizes.lhs);
        let rhs = select_swizzle(rhs_swizzle_dim, *dtypes.rhs_stage, line_sizes.rhs);
        builder = builder.shared_swizzle(SwizzleBlueprint {
            lhs,
            rhs,
            ..Default::default()
        });
    }

    Ok(builder.build())
}
