use std::fmt::Display;

use cubecl::{CubeCount, CubeDim, Runtime, client::ComputeClient, ir::AddressType};
use cubek_std::tile::RowMajorTilingOrder;

use crate::components::batch::{PartitionedBatchMatmulFamily, RowMajorGlobalPartitionMatmul};
use crate::components::{
    batch::BatchMatmulFamily, global::read::sync_full_cyclic::SyncFullCyclicLoading,
};
use crate::definition::{
    BatchMatmulBlueprint, CubeMappingLaunch, MatmulElems, MatmulProblem, MatmulSetupError,
    MatmulVectorSizes, MultiRowStrategy,
};
use crate::{
    args::{ConfigRuntimeArg, InputRuntimeArg, MatmulArgs, OutputRuntimeArg, RuntimeConfig},
    routines::selector::{PlaneTilingBlueprintOptions, infer_blueprint_plane},
    routines::{
        BatchMatmulRoutine, BlueprintStrategy, DeviceSettings, LaunchInfo, Routine, TilingArgs,
        batch_validate_blueprint,
    },
    {components::global::PlaneWriterFamily, routines::ExpandInfo},
};
use crate::{
    components::global::multi_stage::ordered::OrderedDoubleBufferingMatmulFamily,
    components::global::read::sync_partial_cyclic::SyncPartialCyclicLoading,
    components::stage::{NumStages, PlanePartitioner},
    components::tile::TileMatmulKind,
};

/// The batch-matmul family powering [`OrderedDoubleBufferingAlgorithm`].
type OrderedDoubleBufferingBatch<RC> = PartitionedBatchMatmulFamily<
    RC,
    OrderedDoubleBufferingMatmulFamily<
        PlanePartitioner,
        RC,
        SyncPartialCyclicLoading<RowMajorTilingOrder>,
        SyncFullCyclicLoading<RowMajorTilingOrder>,
        PlaneWriterFamily,
    >,
    RowMajorGlobalPartitionMatmul,
>;

/// Plane accelerated double buffered matmul ordered on Lhs with cyclic reader on Rhs
pub struct OrderedDoubleBufferingAlgorithm;

#[derive(Debug, Clone)]
pub struct OrderedSelectionArgs {
    pub tile_matmul: TileMatmulKind,
    pub partition_k: Option<u32>,
    pub row_count: Option<u32>,
    pub rows_per_plane: Option<u32>,
}

impl Default for OrderedSelectionArgs {
    fn default() -> Self {
        Self {
            tile_matmul: TileMatmulKind::Cmma,
            partition_k: None,
            row_count: None,
            rows_per_plane: None,
        }
    }
}

impl TilingArgs for OrderedSelectionArgs {
    fn set_tile_matmul(&mut self, kind: TileMatmulKind) {
        self.tile_matmul = kind;
    }
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

impl<RC> Routine<RC> for OrderedDoubleBufferingAlgorithm
where
    RC: RuntimeConfig,
{
    type Strategy = OrderedSelectionArgs;
    type Blueprint = BatchMatmulBlueprint;
}

impl<RC> BatchMatmulRoutine<RC> for OrderedDoubleBufferingAlgorithm
where
    RC: RuntimeConfig,
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
                <OrderedDoubleBufferingBatch<RC>>::launch_unchecked::<MA, R>(
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
        batch_validate_blueprint::<OrderedDoubleBufferingBatch<RC>, RC, R>(
            client,
            blueprint,
            problem,
            dtypes,
            vector_sizes,
        )
    }

    fn num_stages() -> NumStages {
        OrderedDoubleBufferingBatch::<RC>::num_stages()
    }

    fn expand_blueprint<R: Runtime>(
        problem: &MatmulProblem,
        device_settings: &DeviceSettings<R>,
        strategy: &BlueprintStrategy<RC, Self>,
    ) -> Result<ExpandInfo<Self::Blueprint>, MatmulSetupError> {
        let mut dtypes = MatmulElems::from_globals(&problem.global_dtypes);

        let tile_matmul = match strategy {
            BlueprintStrategy::Forced(blueprint) => blueprint.tile_matmul,
            BlueprintStrategy::Inferred(args) => args.tile_matmul,
        };

        if tile_matmul.can_cast_stage_element() {
            dtypes.adjust_stage_dtypes();
        }

        let (blueprint, dtypes) = match strategy {
            BlueprintStrategy::Forced(blueprint) => (blueprint.clone(), dtypes),
            BlueprintStrategy::Inferred(strategy) => infer_blueprint_plane::<R>(
                tile_matmul,
                &device_settings.client,
                problem,
                device_settings.plane_dim,
                dtypes,
                &device_settings.vector_sizes,
                PlaneTilingBlueprintOptions {
                    partition_k: strategy.partition_k,
                    row_count: strategy.row_count,
                    multi_row_strategy: strategy
                        .rows_per_plane
                        .map(MultiRowStrategy::Always)
                        .unwrap_or_else(|| MultiRowStrategy::Adaptive {
                            minimum_stage_count: 8,
                        }),
                    swizzled: tile_matmul.should_swizzle(&device_settings.client),
                    ..Default::default()
                },
            )?,
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

        let cubedim_resource = OrderedDoubleBufferingBatch::<RC>::cubedim_resource(
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
