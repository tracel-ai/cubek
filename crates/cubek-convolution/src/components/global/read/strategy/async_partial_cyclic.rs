use std::marker::PhantomData;

use cubecl::prelude::*;
use cubecl::std::tensor::layout::{Layout, LayoutExpand};
use cubecl::{ir::DeviceProperties, prelude::barrier::Barrier};
use cubek_matmul::{
    components::{
        global::{
            GlobalReaderConfig, PlaneFlowPartition, SharedGlobalMatmulConfig,
            memory::GlobalIterator,
            multi_stage::LoadMaxRoundPlaneCount,
            read::{
                AsyncPartialLoadingStrategy, LoadingJob, LoadingValidation, PartialLoadingStrategy,
                ReaderMode, async_barrier::AsyncCopy,
                async_partial_cyclic::AsyncPartialCyclicLoading as MatmulCyclicLoading,
                tiled::TiledLayout,
            },
        },
        stage::{
            ContiguousTilingLayout, StageConfig, StridedStageFamily, StridedStageMemory,
            TilingOrder,
        },
        tile::io::Strided,
    },
    definition::{InvalidConfigError, MatmulElems, MatmulPrecision, MatmulProblem, StageIdent},
};

use crate::components::global::{
    args::RuntimeArgs,
    read::strategy::async_copy::{ASYNC_COPY_WIDTH, async_copy_from},
};

#[derive(CubeType, Clone, Copy)]
/// Loads the content of all tiles in the stage using all planes.
/// Unit with pos X loads lines with indices X, X + NUM_UNITS, X + 2 * NUM_UNITS, ...
pub struct AsyncPartialCyclicLoading<T: TilingOrder> {
    #[cube(comptime)]
    _phantom: PhantomData<T>,
}

impl<TO: TilingOrder> LoadingValidation for AsyncPartialCyclicLoading<TO> {
    fn validate_with_config(
        device_props: &DeviceProperties,
        config: &GlobalReaderConfig,
    ) -> Result<(), InvalidConfigError> {
        MatmulCyclicLoading::<TO>::validate_with_config(device_props, config)
    }

    fn validate_with_problem(
        problem: &MatmulProblem,
        dtypes: &MatmulElems,
        ident: StageIdent,
    ) -> Result<(), InvalidConfigError> {
        MatmulCyclicLoading::<TO>::validate_with_problem(problem, dtypes, ident)
    }
}

impl<TO: TilingOrder> LoadMaxRoundPlaneCount for AsyncPartialCyclicLoading<TO> {
    fn max_round_plane_count(
        elements_per_tile: u32,
        tiles_per_stage: u32,
        line_size: LineSize,
        plane_dim: u32,
        dtype: StorageType,
    ) -> u32 {
        MatmulCyclicLoading::<TO>::max_round_plane_count(
            elements_per_tile,
            tiles_per_stage,
            line_size,
            plane_dim,
            dtype,
        )
    }
}

#[cube]
impl<TO: TilingOrder> PartialLoadingStrategy<RuntimeArgs> for AsyncPartialCyclicLoading<TO> {
    type TilingLayout = ContiguousTilingLayout<TO>;
    type SyncStrategy = AsyncCopy;
    type Stage = StridedStageFamily;
    type TileKind = Strided;

    type Job<EG: Numeric, ES: Numeric> = AsyncPartialCyclicJob;

    fn new_job<EG: Numeric, ES: Numeric>(
        runtime_args: RuntimeArgs,
        #[comptime] stage_index: u32,
        #[comptime] _line_size: LineSize,
        #[comptime] config: GlobalReaderConfig,
    ) -> AsyncPartialCyclicJob {
        let type_size = ES::type_size_bits().comptime();
        let line_size = ASYNC_COPY_WIDTH / type_size as u32;
        let num_stage_elements = config.smem_config.elements_per_stage();

        let tile_size = config.smem_config.elements_per_tile();
        let tile_count_row = config.smem_config.tiles_per_stage_along_row();
        let tile_count_col = config.smem_config.tiles_per_stage_along_col();

        let num_lines_per_tile = tile_size / line_size;
        let total_units = config.loading_units_count();

        let num_tiles_in_stage = tile_count_row * tile_count_col;
        let total_num_lines = num_tiles_in_stage * num_lines_per_tile;
        let balanced_workload = total_num_lines.is_multiple_of(total_units);
        let num_tasks_per_unit = total_num_lines.div_ceil(total_units);
        let jump_length = total_units * line_size;

        let plane_id = PlaneFlowPartition::new(config.plane_flow_config.partition_rule)
            .load_index(config.input_load_flow);
        let unit_id = plane_id * config.plane_dim + UNIT_POS_X;
        let unit_position_base = unit_id * line_size;

        AsyncPartialCyclicJob {
            unit_position_base,
            runtime_args,
            num_tasks_per_unit,
            stage_index,
            jump_length,
            num_lines_per_tile,
            balanced_workload,
            num_stage_elements,
            copy_line_size: line_size,
            reader_mode: config.reader_mode,
        }
    }
}

#[derive(CubeType, Clone)]
pub struct AsyncPartialCyclicJob {
    unit_position_base: u32,
    runtime_args: RuntimeArgs,

    #[cube(comptime)]
    num_tasks_per_unit: u32,
    #[cube(comptime)]
    stage_index: u32,
    #[cube(comptime)]
    jump_length: u32,
    #[cube(comptime)]
    num_lines_per_tile: u32,
    #[cube(comptime)]
    balanced_workload: bool,
    #[cube(comptime)]
    num_stage_elements: u32,
    #[cube(comptime)]
    reader_mode: ReaderMode,
    #[cube(comptime)]
    copy_line_size: u32,
}

#[cube]
impl<EG: Numeric, ES: Numeric, TO: TilingOrder>
    LoadingJob<EG, ES, ContiguousTilingLayout<TO>, AsyncCopy> for AsyncPartialCyclicJob
{
    type Stage = StridedStageFamily;

    fn execute_task(
        this: &mut Self,
        #[comptime] task_id: u32,
        global_iter: &GlobalIterator<Line<EG>>,
        stage: &mut StridedStageMemory<ES, ContiguousTilingLayout<TO>>,
        _barrier: &mut Shared<Barrier>,
        #[comptime] config: GlobalReaderConfig,
    ) {
        let unit_position = this.unit_position_base + task_id * this.jump_length;
        let mut stage = stage.with_buffer_index(this.stage_index);

        #[allow(clippy::collapsible_else_if)]
        if comptime!(this.reader_mode == ReaderMode::Strict || this.balanced_workload) {
            copy_line::<EG, ES, TO>(
                this,
                unit_position,
                global_iter,
                &mut stage,
                &this.runtime_args,
                config,
            );
        } else {
            if unit_position < this.num_stage_elements {
                copy_line::<EG, ES, TO>(
                    this,
                    unit_position,
                    global_iter,
                    &mut stage,
                    &this.runtime_args,
                    config,
                );
            }
        }
    }

    fn task_count(this: &Self) -> comptime_type!(u32) {
        this.num_tasks_per_unit
    }
}

#[cube]
pub(crate) fn copy_line<EG: Numeric, ES: Numeric, TO: TilingOrder>(
    job: &AsyncPartialCyclicJob,
    unit_position: u32,
    global_iter: &GlobalIterator<Line<EG>>,
    stage: &mut StridedStageMemory<ES, ContiguousTilingLayout<TO>>,
    runtime_args: &RuntimeArgs,
    #[comptime] config: GlobalReaderConfig,
) {
    let layout = TiledLayout::new(config.stage_ident, config.smem_config);
    let view = global_iter.view();

    let tile_size = config.smem_config.elements_per_tile();
    let tile_count_row = config.smem_config.tiles_per_stage_along_row();
    let tile_count_col = config.smem_config.tiles_per_stage_along_col();

    let tile_index = unit_position / tile_size;
    let pos_within_tile = unit_position % tile_size;

    let (tile_x_within_stage, tile_y_within_stage) = TO::to_row_col(
        tile_index,
        tile_count_row,
        tile_count_col,
        config.smem_config,
    );

    let tile = match config.stage_ident {
        StageIdent::Lhs => (
            tile_x_within_stage,
            job.stage_index * tile_count_col + tile_y_within_stage,
        ),
        StageIdent::Rhs => (
            job.stage_index * tile_count_row + tile_x_within_stage,
            tile_y_within_stage,
        ),
        _ => unreachable!(),
    };

    let pos = layout.to_source_pos((tile, pos_within_tile));

    let tile_start = tile_index * job.num_lines_per_tile * job.copy_line_size;
    let stage_offset = (tile_start + pos_within_tile) / stage.smem.line_size() as u32;

    async_copy_from(
        view,
        pos,
        stage,
        stage_offset,
        runtime_args,
        global_iter.offset(),
        config,
        job.copy_line_size,
    );
}

#[cube]
impl<TO: TilingOrder> AsyncPartialLoadingStrategy<RuntimeArgs> for AsyncPartialCyclicLoading<TO> {
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
