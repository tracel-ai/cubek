use crate::components::stage::matmul::plane_partitioned::{
    PlaneMatmul, PlanePartitionedStageConfig,
};
use crate::components::{
    CubeDimResource,
    global::{MatmulPlaneCounts, PartitionedStage, PartitionedStageFamily, PlaneFlowConfig},
    stage::{
        NumStages, PartitionBuffering, PartitionSchedulerScheme, StageFamily, StageMatmulFamily,
        matmul::{
            partition::SharedPartitionMatmulConfig, partitioned_matmul::PartitionMatmulConfig,
        },
    },
    tile_matmul::TileMatmulFamily,
};
use crate::definition::{
    MatmulElems, MatmulSetupError, MatmulTypes, MatmulVectorSizes, MatrixTypes, TilingBlueprint,
};
use crate::{
    components::stage::TilingLayout,
    definition::{Acc, Lhs, Rhs},
};
use core::marker::PhantomData;
use cubecl::{ir::DeviceProperties, prelude::*};
use cubek_std::{
    stage::StageMemoryConfig,
    {InvalidConfigError, MatrixLayout},
};

type STy<T> = crate::definition::Stage<T>;
type SSz<T> = crate::definition::StageSize<T>;

/// Plane Matmul family for any precision
pub struct PlaneMatmulFamily<
    TM: TileMatmulFamily,
    StageLhs: StageFamily,
    StageRhs: StageFamily,
    StageAcc: StageFamily,
> {
    _phantom: PhantomData<(TM, StageLhs, StageRhs, StageAcc)>,
}

impl<TM: TileMatmulFamily, StageLhs: StageFamily, StageRhs: StageFamily, StageAcc: StageFamily>
    StageMatmulFamily for PlaneMatmulFamily<TM, StageLhs, StageRhs, StageAcc>
{
    type Scope = TM::Scope;
    type LhsStage = StageLhs;
    type RhsStage = StageRhs;
    type AccStage = StageAcc;
    type OutStage = PartitionedStageFamily;

    type Matmul<
        MP: MatmulTypes,
        TL: TilingLayout,
        TR: TilingLayout,
        TA: TilingLayout,
        TO: TilingLayout,
    > = PlaneMatmul<
        MP,
        TM::Matmul<
            <MP::Lhs as MatrixTypes>::Register,
            <MP::Lhs as MatrixTypes>::RegisterSize,
            <MP::Rhs as MatrixTypes>::Register,
            <MP::Rhs as MatrixTypes>::RegisterSize,
            <MP::Acc as MatrixTypes>::Register,
            <MP::Acc as MatrixTypes>::RegisterSize,
        >,
        StageLhs::Stage<STy<Lhs<MP>>, SSz<Lhs<MP>>, TL>,
        StageRhs::Stage<STy<Rhs<MP>>, SSz<Rhs<MP>>, TR>,
        StageAcc::Stage<STy<Acc<MP>>, SSz<Acc<MP>>, TA>,
        PartitionedStage<STy<Acc<MP>>, SSz<Acc<MP>>>,
    >;

    type Config = PartitionMatmulConfig<TM::Config>;

    fn expand_config(
        device_props: &DeviceProperties,
        blueprint: &TilingBlueprint,
        plane_flow_config: PlaneFlowConfig,
        num_stages: NumStages,
        dtypes: &MatmulElems,
        vector_sizes: &MatmulVectorSizes,
    ) -> Result<Self::Config, MatmulSetupError> {
        let plane_counts = MatmulPlaneCounts::new(blueprint.load_flows, plane_flow_config.counts);

        let lhs_smem_config = StageMemoryConfig {
            num_planes: plane_counts.lhs,
            elements_per_tile_along_row: blueprint.tiling_scheme.tile_size.m,
            elements_per_tile_along_col: blueprint.tiling_scheme.tile_size.k,
            tiles_per_partition_along_row: blueprint.tiling_scheme.partition_size.m as u32,
            tiles_per_partition_along_col: blueprint.tiling_scheme.partition_size.k as u32,
            partitions_per_stage_along_row: blueprint.tiling_scheme.stage_size.m as u32,
            partitions_per_stage_along_col: blueprint.tiling_scheme.stage_size.k as u32,
            vector_size: vector_sizes.lhs as u32,
            matrix_layout: blueprint.lhs_layout,
            swizzle: blueprint.swizzle_modes.lhs,
            num_stages: num_stages.lhs,
            dtype: dtypes.lhs_stage,
        };

        let rhs_smem_config = StageMemoryConfig {
            num_planes: plane_counts.rhs,
            elements_per_tile_along_row: blueprint.tiling_scheme.tile_size.k,
            elements_per_tile_along_col: blueprint.tiling_scheme.tile_size.n,
            tiles_per_partition_along_row: blueprint.tiling_scheme.partition_size.k as u32,
            tiles_per_partition_along_col: blueprint.tiling_scheme.partition_size.n as u32,
            partitions_per_stage_along_row: blueprint.tiling_scheme.stage_size.k as u32,
            partitions_per_stage_along_col: blueprint.tiling_scheme.stage_size.n as u32,
            vector_size: vector_sizes.rhs as u32,
            matrix_layout: blueprint.rhs_layout,
            swizzle: blueprint.swizzle_modes.rhs,
            num_stages: num_stages.rhs,
            dtype: dtypes.rhs_stage,
        };

        let out_smem_config = StageMemoryConfig {
            num_planes: plane_counts.out,
            elements_per_tile_along_row: blueprint.tiling_scheme.tile_size.m,
            elements_per_tile_along_col: blueprint.tiling_scheme.tile_size.n,
            tiles_per_partition_along_row: blueprint.tiling_scheme.partition_size.m as u32,
            tiles_per_partition_along_col: blueprint.tiling_scheme.partition_size.n as u32,
            partitions_per_stage_along_row: blueprint.tiling_scheme.stage_size.m as u32,
            partitions_per_stage_along_col: blueprint.tiling_scheme.stage_size.n as u32,
            vector_size: vector_sizes.out as u32,
            matrix_layout: MatrixLayout::RowMajor,
            swizzle: blueprint.swizzle_modes.out,
            num_stages: 1,
            dtype: dtypes.acc_stage,
        };

        Ok(PartitionMatmulConfig::Plane(
            PlanePartitionedStageConfig::from_shared_partition_config(
                SharedPartitionMatmulConfig::new(
                    TM::expand_config(device_props, blueprint, dtypes, vector_sizes)?,
                    blueprint.tiling_scheme.partition_size,
                    blueprint.partition_buffering,
                    plane_flow_config,
                    blueprint.plane_dim,
                    blueprint.tiling_scheme.stage_size,
                    PartitionSchedulerScheme::Naive,
                    lhs_smem_config,
                    rhs_smem_config,
                    out_smem_config,
                    out_smem_config,
                ),
            ),
        ))
    }

    fn cubedim_resource(
        blueprint: &TilingBlueprint,
    ) -> Result<CubeDimResource, InvalidConfigError> {
        if let CubeDimResource::Planes(planes) = TM::cubedim_resource()? {
            Ok(CubeDimResource::Planes(
                planes
                    * blueprint.tiling_scheme.partitions_per_stage_along_m()
                    * blueprint.tiling_scheme.partitions_per_stage_along_n(),
            ))
        } else {
            Err(Box::new(
                "Error: Tried to use a plane stage matmul with a unit tile matmul.".to_string(),
            ))
        }
    }

    fn validate_blueprint<R: Runtime>(
        client: &ComputeClient<R>,
        blueprint: &TilingBlueprint,
        dtypes: &MatmulElems,
        vector_sizes: &MatmulVectorSizes,
    ) -> Result<(), MatmulSetupError> {
        let num_planes_needed = blueprint.tiling_scheme.partitions_per_stage_along_m()
            * blueprint.tiling_scheme.partitions_per_stage_along_n();
        let num_compute_planes =
            Self::cubedim_resource(blueprint)?.num_planes(blueprint.plane_dim)?;

        if num_compute_planes != num_planes_needed {
            return Err(MatmulSetupError::InvalidConfig(Box::new(format!(
                "Error: Number of compute planes {num_compute_planes} should be {num_planes_needed}."
            ))));
        }

        if blueprint.partition_buffering == PartitionBuffering::Double
            && blueprint.tiling_scheme.tiles_per_stage_partition_along_n() < 2
        {
            return Err(MatmulSetupError::InvalidConfig(Box::new(
                "Error: Tried doing double buffering with only one tile to compute.".to_string(),
            )));
        }

        TM::validate_blueprint(client, blueprint, dtypes, vector_sizes)
    }
}
