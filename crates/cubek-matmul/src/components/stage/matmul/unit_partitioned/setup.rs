use crate::components::CubeDimResource;
use crate::components::global::MatmulPlaneCounts;
use crate::components::global::MaxGlobalReaderPlanes;
use crate::components::global::PartitionedStage;
use crate::components::global::PartitionedStageFamily;
use crate::components::global::PlaneRoleConfig;
use crate::components::stage::NumStages;
use crate::components::stage::PartitionBuffering;
use crate::components::stage::PartitionSchedulerScheme;
use crate::components::stage::StageFamily;
use crate::components::stage::StageMemoryConfig;
use crate::components::stage::matmul::partition::SharedPartitionMatmulConfig;
use crate::components::stage::matmul::partitioned_matmul::PartitionMatmulConfig;
use crate::components::stage::matmul::unit_partitioned::UnitMatmul;
use crate::components::stage::matmul::unit_partitioned::UnitPartitionedStageConfig;
use crate::components::stage::{StageMatmulFamily, TilingLayout};
use crate::components::tile::TileMatmulFamily;
use crate::components::tile::io::Strided;
use crate::definition::AccS;
use crate::definition::InvalidConfigError;
use crate::definition::LhsS;
use crate::definition::MatmulPrecision;
use crate::definition::MatmulSetupError;
use crate::definition::MatrixLayout;
use crate::definition::MatrixPrecision;
use crate::definition::RhsS;
use crate::definition::TilingBlueprint;
use core::marker::PhantomData;
use cubecl::prelude::*;

/// Unit Matmul family for any precision
pub struct UnitMatmulFamily<TM: TileMatmulFamily, StageIn: StageFamily, StageAcc: StageFamily> {
    _phantom: PhantomData<(TM, StageIn, StageAcc)>,
}

impl<
    TM: TileMatmulFamily<
            LhsTile = StageIn::TileKind,
            RhsTile = StageIn::TileKind,
            AccTile = StageAcc::TileKind,
            OutTile = Strided,
        >,
    StageIn: StageFamily,
    StageAcc: StageFamily,
> StageMatmulFamily for UnitMatmulFamily<TM, StageIn, StageAcc>
{
    type LhsStage = StageIn;
    type RhsStage = StageIn;
    type AccStage = StageAcc;
    type OutStage = PartitionedStageFamily;

    type Matmul<
        MP: MatmulPrecision,
        TL: TilingLayout,
        TR: TilingLayout,
        TA: TilingLayout,
        TO: TilingLayout,
    > = UnitMatmul<
        MP,
        TM::Matmul<
            <MP::Lhs as MatrixPrecision>::Register,
            <MP::Rhs as MatrixPrecision>::Register,
            <MP::Acc as MatrixPrecision>::Register,
        >,
        StageIn::Stage<LhsS<MP>, TL>,
        StageIn::Stage<RhsS<MP>, TR>,
        StageAcc::Stage<AccS<MP>, TA>,
        PartitionedStage<AccS<MP>>,
    >;

    type Config = PartitionMatmulConfig<TM::Config>;

    fn expand_config(
        blueprint: &TilingBlueprint,
        reader_tasks: Option<MaxGlobalReaderPlanes>,
        num_stages: NumStages,
    ) -> Result<Self::Config, MatmulSetupError> {
        let num_planes = Self::cubedim_resource(blueprint)?.num_planes(blueprint.plane_dim)?;

        let plane_role_config = PlaneRoleConfig::new(
            blueprint.load_specialization_config,
            reader_tasks,
            num_planes,
        )?;

        let plane_counts = MatmulPlaneCounts::new(
            blueprint.load_specialization_config,
            plane_role_config.plane_roles,
        );

        let lhs_smem_config = StageMemoryConfig {
            num_planes: plane_counts.lhs,
            elements_per_tile_along_row: blueprint.tiling_scheme.tile_size.m,
            elements_per_tile_along_col: blueprint.tiling_scheme.tile_size.k,
            tiles_per_partition_along_row: blueprint.tiling_scheme.partition_size.m as u32,
            tiles_per_partition_along_col: blueprint.tiling_scheme.partition_size.k as u32,
            partitions_per_stage_along_row: blueprint.tiling_scheme.stage_size.m as u32,
            partitions_per_stage_along_col: blueprint.tiling_scheme.stage_size.k as u32,
            line_size: blueprint.line_sizes.lhs as u32,
            matrix_layout: blueprint.lhs_layout,
            swizzle: blueprint.swizzle_modes.lhs,
            num_stages: num_stages.lhs,
        };

        let rhs_smem_config = StageMemoryConfig {
            num_planes: plane_counts.rhs,
            elements_per_tile_along_row: blueprint.tiling_scheme.tile_size.k,
            elements_per_tile_along_col: blueprint.tiling_scheme.tile_size.n,
            tiles_per_partition_along_row: blueprint.tiling_scheme.partition_size.k as u32,
            tiles_per_partition_along_col: blueprint.tiling_scheme.partition_size.n as u32,
            partitions_per_stage_along_row: blueprint.tiling_scheme.stage_size.k as u32,
            partitions_per_stage_along_col: blueprint.tiling_scheme.stage_size.n as u32,
            line_size: blueprint.line_sizes.rhs as u32,
            matrix_layout: blueprint.rhs_layout,
            swizzle: blueprint.swizzle_modes.rhs,
            num_stages: num_stages.rhs,
        };

        let out_smem_config = StageMemoryConfig {
            num_planes: plane_counts.out,
            elements_per_tile_along_row: blueprint.tiling_scheme.tile_size.m,
            elements_per_tile_along_col: blueprint.tiling_scheme.tile_size.n,
            tiles_per_partition_along_row: blueprint.tiling_scheme.partition_size.m as u32,
            tiles_per_partition_along_col: blueprint.tiling_scheme.partition_size.n as u32,
            partitions_per_stage_along_row: blueprint.tiling_scheme.stage_size.m as u32,
            partitions_per_stage_along_col: blueprint.tiling_scheme.stage_size.n as u32,
            line_size: blueprint.line_sizes.out as u32,
            matrix_layout: MatrixLayout::RowMajor,
            swizzle: blueprint.swizzle_modes.out,
            num_stages: 1,
        };

        Ok(PartitionMatmulConfig::Unit(
            UnitPartitionedStageConfig::from_shared_partition_config(
                SharedPartitionMatmulConfig::new(
                    TM::expand_config(blueprint)?,
                    blueprint.tiling_scheme.partition_size,
                    blueprint.partition_buffering,
                    plane_role_config,
                    blueprint.plane_dim,
                    blueprint.tiling_scheme.stage_size,
                    PartitionSchedulerScheme::Naive,
                    lhs_smem_config,
                    rhs_smem_config,
                    out_smem_config,
                ),
            ),
        ))
    }

    fn cubedim_resource(
        blueprint: &TilingBlueprint,
    ) -> Result<CubeDimResource, InvalidConfigError> {
        if let CubeDimResource::Units(units) = TM::cubedim_resource()? {
            Ok(CubeDimResource::Units(
                units
                    * blueprint.tiling_scheme.partitions_per_stage_along_m()
                    * blueprint.tiling_scheme.partitions_per_stage_along_n(),
            ))
        } else {
            return Err(Box::new(
                "Error: Tried to use a unit stage matmul with a plane tile matmul.".to_string(),
            ));
        }
    }

    fn validate_blueprint<R: Runtime>(
        client: &ComputeClient<R>,
        blueprint: &TilingBlueprint,
        num_stages: NumStages,
    ) -> Result<(), MatmulSetupError> {
        let working_units = blueprint.tiling_scheme.partitions_per_stage_along_m()
            * blueprint.tiling_scheme.partitions_per_stage_along_n();
        let num_compute_planes =
            Self::cubedim_resource(blueprint)?.num_planes(blueprint.plane_dim)?;
        let num_units = blueprint.plane_dim * num_compute_planes;

        if num_units != working_units {
            return Err(MatmulSetupError::InvalidConfig(Box::new(format!(
                "Error: Number of units {num_units} should be {working_units}."
            ))));
        }

        if blueprint.partition_buffering == PartitionBuffering::Double
            && blueprint.tiling_scheme.tiles_per_stage_partition_along_n() < 2
        {
            return Err(MatmulSetupError::InvalidConfig(Box::new(
                "Error: Tried doing partition double buffering with only one tile to compute.",
            )));
        }

        let lhs_smem_size = blueprint.tiling_scheme.elements_per_stage_along_m()
            * blueprint.tiling_scheme.elements_per_stage_along_k()
            * num_stages.lhs;
        let rhs_smem_size = blueprint.tiling_scheme.elements_per_stage_along_k()
            * blueprint.tiling_scheme.elements_per_stage_along_n()
            * num_stages.rhs;
        let out_smem_size =
            blueprint.tiling_scheme.tile_size.m * blueprint.tiling_scheme.tile_size.n * num_units;
        let smem_total_size = blueprint.dtypes.lhs_stage.size() as u32 * lhs_smem_size
            + blueprint.dtypes.rhs_stage.size() as u32 * rhs_smem_size
            + blueprint.dtypes.acc_stage.size() as u32 * out_smem_size;

        let smem_limit = client.properties().hardware.max_shared_memory_size as u32;
        if smem_total_size > smem_limit {
            return Err(MatmulSetupError::InvalidConfig(Box::new(format!(
                "This algorithm needs {smem_total_size:?} shared memory bytes but hardware limit is {smem_limit:?}. "
            ))));
        }

        Ok(())
    }
}
