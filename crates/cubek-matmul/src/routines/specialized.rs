use std::fmt::Display;
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
    CubeCountStrategy, GlobalOrderStrategy, HypercubeBlueprint, MatmulLineSizes, MatmulProblem,
    MatmulSetupError, MatrixLayout, SmAllocation, SwizzleModes, TilingBlueprint, adjust_dtypes,
};
use crate::routines::selector::{PlaneTilingBlueprintOptions, infer_blueprint_plane};
use crate::routines::{BlueprintStrategy, DeviceSettings, LaunchInfo, base};
use crate::{
    components::global::{
        multi_stage::specialized::SpecializedMatmulFamily,
        read::{AsyncPartialLoadingStrategy, async_partial_tma::AsyncPartialTmaLoading},
    },
    definition::{MatmulElems, MultiRowStrategy, TilingScheme},
};
use crate::{
    components::{
        global::{InputLoadFlow, LoadFlows},
        stage::PartitionBuffering,
    },
    routines::selector::select_swizzle,
};

/// Plane accelerated specialized matmul with TMA readers
pub struct SpecializedAlgorithm<TMM, L = AsyncPartialTmaLoading> {
    pub _phantom: PhantomData<(TMM, L)>,
}

#[derive(Default, Clone)]
pub struct SpecializedStrategy {}

impl Display for SpecializedStrategy {
    fn fmt(&self, _f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        Ok(())
    }
}

impl From<()> for SpecializedStrategy {
    fn from(_value: ()) -> Self {
        Self {}
    }
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
    type Strategy = SpecializedStrategy;
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
        problem: &MatmulProblem,
        device_settings: &DeviceSettings<R>,
        strategy: &BlueprintStrategy<Self>,
    ) -> Result<LaunchInfo<TilingBlueprint>, MatmulSetupError> {
        let mut dtypes = MatmulElems::from_globals(&problem.global_dtypes);

        if TMM::can_cast_stage_element() {
            dtypes.adjust_stage_dtypes();
        }

        let (blueprint, dtypes) = match strategy {
            BlueprintStrategy::Forced(blueprint) => (blueprint.clone(), dtypes),
            BlueprintStrategy::Inferred(_) => infer_blueprint_plane::<TMM, R>(
                &device_settings.client,
                problem,
                device_settings.plane_dim,
                dtypes,
                &device_settings.line_sizes,
                PlaneTilingBlueprintOptions {
                    specialized: true,
                    multi_row_strategy: MultiRowStrategy::Adaptive {
                        minimum_stage_count: 8,
                    },
                    swizzled: TMM::should_swizzle(&device_settings.client),
                    ..Default::default()
                },
            )?,
        };

        Self::validate_blueprint(
            &device_settings.client,
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

#[allow(unused, reason = "needs more tuning")]
fn infer_blueprint_specialized<R: Runtime, TMM: TileMatmulFamily>(
    client: &ComputeClient<R>,
    problem: &MatmulProblem,
    plane_dim: u32,
    swizzle: bool,
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
        .global_order_strategy(GlobalOrderStrategy::SwizzleRow {
            m: problem.m as u32,
            w: 4,
        })
        .cube_count_strategy(cube_count_strategy)
        .build();

    let mut builder = TilingBlueprint::builder(tiling_scheme, plane_dim, problem)
        .partition_buffering(PartitionBuffering::Single)
        .hypercube_blueprint(hypercube)
        .load_specialization_config(LoadFlows {
            lhs: InputLoadFlow::LoadOnly,
            rhs: InputLoadFlow::LoadOnly,
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

        let lhs = select_swizzle(lhs_swizzle_dim, dtypes.lhs_stage, line_sizes.lhs);
        let rhs = select_swizzle(rhs_swizzle_dim, dtypes.rhs_stage, line_sizes.rhs);
        builder = builder.shared_swizzle(SwizzleModes {
            lhs,
            rhs,
            ..Default::default()
        });
    }

    Ok((builder.build(), dtypes))
}
