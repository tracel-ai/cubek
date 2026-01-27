use std::fmt::Display;
use std::marker::PhantomData;

use cubecl::Runtime;

use crate::components::batch::BatchMatmulFamily;
use crate::components::global::PlaneWriterFamily;
use crate::components::stage::{PlaneMatmulFamily, RowMajorTilingOrder};
use crate::components::tile;
use crate::components::{
    batch::{PartitionedBatchMatmulFamily, RowMajorGlobalPartitionMatmul},
    stage::{FilledStageFamily, StridedStageFamily},
};
use crate::components::{
    global::multi_stage::ordered::OrderedDoubleBufferingMatmulFamily, tile::io::Filled,
};
use crate::components::{
    global::read::sync_partial_cyclic::SyncPartialCyclicLoading, tile::io::Strided,
};
use crate::definition::{
    MatmulElems, MatmulProblem, MatmulSetupError, MultiRowStrategy, TilingBlueprint,
};
use crate::routines::selector::{PlaneTilingBlueprintOptions, infer_blueprint_plane};
use crate::routines::{BlueprintStrategy, DeviceSettings, LaunchInfo, Routine};

/// Plane accelerated double buffered matmul ordered on Lhs with cyclic reader on Rhs
pub struct OrderedDoubleBufferingAlgorithm<TMM> {
    pub _phantom: PhantomData<TMM>,
}

#[derive(Debug, Clone, Default)]
pub struct OrderedSelectionArgs {
    pub partition_k: Option<u32>,
    pub row_count: Option<u32>,
    pub rows_per_plane: Option<u32>,
}

impl Display for OrderedSelectionArgs {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if let Some(k) = self.partition_k {
            f.write_fmt(format_args!("_partition_k{}", k))?;
        }
        if let Some(r) = self.row_count {
            f.write_fmt(format_args!("_row_count{}", r))?;
        }
        if let Some(r) = self.rows_per_plane {
            f.write_fmt(format_args!("_rows_per_plane{}", r))?;
        }

        Ok(())
    }
}

impl<TMM> Routine for OrderedDoubleBufferingAlgorithm<TMM>
where
    TMM: tile::TileMatmulFamily<
            LhsTile = Strided,
            RhsTile = Strided,
            AccTile = Filled,
            OutTile = Strided,
        >,
{
    type Strategy = OrderedSelectionArgs;
    type BatchMatmul = PartitionedBatchMatmulFamily<
        OrderedDoubleBufferingMatmulFamily<
            PlaneMatmulFamily<TMM, StridedStageFamily, StridedStageFamily, FilledStageFamily>,
            SyncPartialCyclicLoading<RowMajorTilingOrder>,
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
            BlueprintStrategy::Inferred(strategy) => infer_blueprint_plane::<TMM, R>(
                &device_settings.client,
                problem,
                device_settings.plane_dim,
                dtypes,
                &device_settings.line_sizes,
                PlaneTilingBlueprintOptions {
                    partition_k: strategy.partition_k,
                    row_count: strategy.row_count,
                    multi_row_strategy: strategy
                        .rows_per_plane
                        .map(MultiRowStrategy::Always)
                        .unwrap_or_else(|| MultiRowStrategy::Adaptive {
                            minimum_stage_count: 8,
                        }),
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
