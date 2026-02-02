use crate::components::global::{
    SharedGlobalMatmulConfig,
    read::{AsyncPartialLoadingStrategy, PartialLoadingStrategy, async_copy::ASYNC_COPY_WIDTH},
};
use crate::components::{global::memory::GlobalIterator, stage::TilingValidation};
use crate::components::{global::read::async_copy::async_copy_from, stage::StridedStageMemory};
use crate::components::{global::read::stage::FullStageLayout, stage::StridedStageFamily};
use crate::components::{global::read::validate_swizzle_atom_size, stage::StageConfig};
use crate::components::{
    global::{GlobalReaderConfig, PlaneFlowPartition},
    tile::io::Strided,
};
use crate::components::{
    global::{
        multi_stage::LoadMaxRoundPlaneCount,
        read::{async_barrier::AsyncCopy, validate_async_copy},
    },
    stage::StridedTilingLayout,
};
use crate::definition::{
    InvalidConfigError, MatmulElems, MatmulPrecision, MatmulProblem, StageIdent,
};
use crate::{
    components::global::read::{validate_async_barrier, validate_async_copy_with_problem},
    launch::RuntimeConfig,
};
use cubecl::prelude::*;
use cubecl::std::tensor::layout::{Layout, LayoutExpand};
use cubecl::{ir::DeviceProperties, prelude::barrier::Barrier};

use super::{LoadingJob, LoadingValidation};

#[derive(CubeType, Clone, Copy)]
/// Loads the content of all tiles in the stage using all planes.
/// Unit with pos X loads lines with indices X, X + NUM_UNITS, X + 2 * NUM_UNITS, ...
pub struct AsyncPartialStridedLoading {}

impl LoadingValidation for AsyncPartialStridedLoading {
    fn validate_with_config(
        device_props: &DeviceProperties,
        config: &GlobalReaderConfig,
    ) -> Result<(), InvalidConfigError> {
        let line_size = ASYNC_COPY_WIDTH / config.smem_config.dtype.size_bits() as u32;

        // Needs separate check because copy size may be larger than global line size
        if !config
            .smem_config
            .elements_per_stage_along_contiguous_dim()
            .is_multiple_of(line_size)
        {
            return Err(Box::new("Stage size isn't divisible by copy line size"));
        }

        let num_stage_lines = config.smem_config.elements_per_stage() / line_size;
        let total_units = config.loading_units_count();

        if !num_stage_lines.is_multiple_of(total_units) {
            return Err(Box::new(format!(
                "Too many data will be loaded, resulting in out of bounds.
        Try setting line size and number of planes so that total unit count {total_units:?} divides number of lines in stage.",
            )));
        }

        validate_swizzle_atom_size(config.smem_config)?;
        validate_async_barrier(device_props)?;
        validate_async_copy(
            device_props,
            &config.gmem_config.dtype,
            &config.smem_config.dtype,
        )?;
        StridedTilingLayout::check(config.smem_config)?;

        Ok(())
    }

    fn validate_with_problem(
        problem: &MatmulProblem,
        dtypes: &MatmulElems,
        ident: StageIdent,
    ) -> Result<(), InvalidConfigError> {
        validate_async_copy_with_problem(problem, dtypes, ident)
    }
}

impl LoadMaxRoundPlaneCount for AsyncPartialStridedLoading {
    fn max_round_plane_count(
        elements_per_tile: u32,
        tiles_per_stage: u32,
        _line_size: LineSize,
        plane_dim: u32,
        dtype: StorageType,
    ) -> u32 {
        let line_size = ASYNC_COPY_WIDTH / dtype.size_bits() as u32;
        let num_lines_per_tile = elements_per_tile / line_size;
        let total_num_lines = tiles_per_stage * num_lines_per_tile;
        total_num_lines.div_ceil(plane_dim)
    }
}

#[cube]
impl<RC: RuntimeConfig> PartialLoadingStrategy<RC> for AsyncPartialStridedLoading {
    type TilingLayout = StridedTilingLayout;
    type SyncStrategy = AsyncCopy;
    type Job<EG: Numeric, ES: Numeric> = AsyncPartialStridedJob;
    type Stage = StridedStageFamily;
    type TileKind = Strided;

    fn new_job<EG: Numeric, ES: Numeric>(
        _runtime_config: RC,
        #[comptime] stage_index: u32,
        #[comptime] _line_size: LineSize,
        #[comptime] config: GlobalReaderConfig,
    ) -> Self::Job<EG, ES> {
        let type_size = ES::type_size_bits().comptime();
        let line_size = ASYNC_COPY_WIDTH / type_size as u32;

        let num_stage_lines = config.smem_config.elements_per_stage() / line_size;
        let unit_count = config.loading_planes_count() * config.plane_dim;
        let num_tasks_per_unit = num_stage_lines / unit_count;

        let unit_position_base = PlaneFlowPartition::new(config.plane_flow_config.partition_rule)
            .load_index(config.input_load_flow)
            * config.plane_dim
            + UNIT_POS_X;

        AsyncPartialStridedJob {
            stage_index,
            unit_position_base,
            num_tasks_per_unit,
            unit_count,
            copy_line_size: line_size,
        }
    }
}

#[derive(CubeType, Clone, Copy)]
pub struct AsyncPartialStridedJob {
    unit_position_base: u32,

    #[cube(comptime)]
    stage_index: u32,
    #[cube(comptime)]
    num_tasks_per_unit: u32,
    #[cube(comptime)]
    unit_count: u32,
    #[cube(comptime)]
    copy_line_size: u32,
}

#[cube]
impl<EG: Numeric, ES: Numeric> LoadingJob<EG, ES, StridedTilingLayout, AsyncCopy>
    for AsyncPartialStridedJob
{
    type Stage = StridedStageFamily;

    fn execute_task(
        this: &mut Self,
        #[comptime] task_id: u32,
        global_iter: &GlobalIterator<Line<EG>>,
        stage: &mut StridedStageMemory<ES, StridedTilingLayout>,
        _barrier: &mut Shared<Barrier>,
        #[comptime] config: GlobalReaderConfig,
    ) {
        let mut stage = stage.with_buffer_index(this.stage_index);

        let unit_position = this.unit_position_base + task_id * this.unit_count;
        let unit_position_abs = unit_position * this.copy_line_size;

        let layout = FullStageLayout::new(config.smem_config);
        let view = global_iter.view();

        let pos = layout.to_source_pos(unit_position_abs);
        let pos = match config.stage_ident {
            StageIdent::Lhs => (
                pos.0,
                pos.1 + config.smem_config.elements_per_stage_along_col() * this.stage_index,
            ),
            StageIdent::Rhs => (
                pos.0 + config.smem_config.elements_per_stage_along_row() * this.stage_index,
                pos.1,
            ),
            _ => pos,
        };

        let stage_offset = unit_position_abs / stage.smem.line_size() as u32;

        async_copy_from(
            view,
            pos,
            &mut stage,
            stage_offset,
            config,
            this.copy_line_size,
        );
    }

    fn task_count(this: &Self) -> comptime_type!(u32) {
        this.num_tasks_per_unit
    }
}

#[cube]
impl<RC: RuntimeConfig> AsyncPartialLoadingStrategy<RC> for AsyncPartialStridedLoading {
    fn arrival_count<S: StageConfig>(#[comptime] config: SharedGlobalMatmulConfig<S>) -> u32 {
        let total_load_units = config.plane_flow_config().counts.load_only * config.plane_dim();
        total_load_units.runtime()
    }

    fn barrier_post_init() {}

    fn arrive<MP: MatmulPrecision, S: StageConfig>(
        barrier: &mut Barrier,
        #[comptime] _config: SharedGlobalMatmulConfig<S>,
    ) {
        barrier.commit_copy_async();
        barrier.arrive();
    }

    fn is_elected<S: StageConfig>(#[comptime] config: SharedGlobalMatmulConfig<S>) -> bool {
        let role_rule = PlaneFlowPartition::new(config.plane_flow_config().partition_rule);
        role_rule.is_load_plane()
    }
}
