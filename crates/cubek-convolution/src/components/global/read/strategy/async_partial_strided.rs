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
                async_barrier::AsyncCopy,
                async_partial_strided::AsyncPartialStridedLoading as MatmulStridedLoading,
                stage::FullStageLayout,
            },
        },
        stage::{StageConfig, StridedStageFamily, StridedStageMemory, StridedTilingLayout},
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
pub struct AsyncPartialStridedLoading {}

impl LoadingValidation for AsyncPartialStridedLoading {
    fn validate_with_config(
        device_props: &DeviceProperties,
        config: &GlobalReaderConfig,
    ) -> Result<(), InvalidConfigError> {
        MatmulStridedLoading::validate_with_config(device_props, config)
    }

    fn validate_with_problem(
        problem: &MatmulProblem,
        dtypes: &MatmulElems,
        ident: StageIdent,
    ) -> Result<(), InvalidConfigError> {
        MatmulStridedLoading::validate_with_problem(problem, dtypes, ident)
    }
}

impl LoadMaxRoundPlaneCount for AsyncPartialStridedLoading {
    fn max_round_plane_count(
        elements_per_tile: u32,
        tiles_per_stage: u32,
        line_size: LineSize,
        plane_dim: u32,
        dtype: StorageType,
    ) -> u32 {
        MatmulStridedLoading::max_round_plane_count(
            elements_per_tile,
            tiles_per_stage,
            line_size,
            plane_dim,
            dtype,
        )
    }
}

#[cube]
impl PartialLoadingStrategy<RuntimeArgs> for AsyncPartialStridedLoading {
    type TilingLayout = StridedTilingLayout;
    type SyncStrategy = AsyncCopy;
    type Job<EG: Numeric, ES: Numeric> = AsyncPartialStridedJob;
    type Stage = StridedStageFamily;
    type TileKind = Strided;

    fn new_job<EG: Numeric, ES: Numeric>(
        runtime_args: RuntimeArgs,
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
            runtime_args,
            unit_position_base,
            num_tasks_per_unit,
            unit_count,
            copy_line_size: line_size,
        }
    }
}

#[derive(CubeType, Clone)]
pub struct AsyncPartialStridedJob {
    unit_position_base: u32,
    runtime_args: RuntimeArgs,

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
            &this.runtime_args,
            global_iter.offset(),
            config,
            this.copy_line_size,
        );
    }

    fn task_count(this: &Self) -> comptime_type!(u32) {
        this.num_tasks_per_unit
    }
}

#[cube]
impl AsyncPartialLoadingStrategy<RuntimeArgs> for AsyncPartialStridedLoading {
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
